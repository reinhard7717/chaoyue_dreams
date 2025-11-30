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

    def _prepare_behavioral_data(self, intraday_data: pd.DataFrame, daily_data: pd.Series, tick_data: pd.DataFrame = None, level5_data: pd.DataFrame = None, realtime_data: pd.DataFrame = None) -> tuple:
        """
        【V47.2 · 遗漏逻辑补全版】
        - 核心修复: 补全在V47.0重构中遗漏的 retail_ofi 和 order_book_liquidity_supply 的前置数据准备逻辑。
        """
        import numpy as np
        daily_total_volume = daily_data.get('vol', 0) * 100
        daily_total_amount = pd.to_numeric(daily_data.get('amount', 0), errors='coerce') * 1000
        daily_vwap = daily_total_amount / daily_total_volume if daily_total_volume > 0 else np.nan
        atr = daily_data.get('atr_14d')
        day_open, day_close = daily_data.get('open_qfq'), daily_data.get('close_qfq')
        day_high, day_low = daily_data.get('high_qfq'), daily_data.get('low_qfq')
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
                    merged_hf, realtime_prepped, left_index=True, right_index=True,
                    direction='backward', suffixes=('_tick', '_realtime')
                )
            if not merged_hf.empty:
                merged_hf.rename(columns={'volume_tick': 'volume'}, inplace=True)
                merged_hf['mid_price'] = (merged_hf['buy_price1'] + merged_hf['sell_price1']) / 2
                merged_hf['prev_mid_price'] = merged_hf['mid_price'].shift(1)
                buy_pressure = np.where(merged_hf['mid_price'] >= merged_hf['prev_mid_price'], merged_hf['buy_volume1'].shift(1), 0)
                sell_pressure = np.where(merged_hf['mid_price'] <= merged_hf['prev_mid_price'], merged_hf['sell_volume1'].shift(1), 0)
                merged_hf['ofi'] = buy_pressure - sell_pressure
                is_main_force_trade = merged_hf['amount'] > 200000
                # 修改代码行：新增 is_retail_trade 的定义
                is_retail_trade = merged_hf['amount'] < 50000
                merged_hf['main_force_ofi'] = np.where(is_main_force_trade, merged_hf['ofi'], 0)
                # 修改代码行：新增 retail_ofi 列的计算
                merged_hf['retail_ofi'] = np.where(is_retail_trade, merged_hf['ofi'], 0)
                merged_hf['mid_price_change'] = merged_hf['mid_price'].diff()
                if 'volume_realtime' in merged_hf.columns and 'snapshot_time' in merged_hf.columns:
                    snapshot_changed_mask = merged_hf['snapshot_time'] != merged_hf['snapshot_time'].shift(1)
                    volume_delta = merged_hf['volume_realtime'].diff().fillna(0)
                    merged_hf['market_vol_delta'] = np.where(snapshot_changed_mask, volume_delta, 0)
                merged_hf['prev_a1_p'] = merged_hf['sell_price1'].shift(1)
                merged_hf['prev_b1_p'] = merged_hf['buy_price1'].shift(1)
                # 修改代码行：新增 prev_a1_v 和 prev_b1_v 的计算，为 exhaustion_rate 做准备
                merged_hf['prev_a1_v'] = merged_hf['sell_volume1'].shift(1)
                merged_hf['prev_b1_v'] = merged_hf['buy_volume1'].shift(1)
                try:
                    weighted_buy_vol = pd.Series(0, index=merged_hf.index); weighted_sell_vol = pd.Series(0, index=merged_hf.index)
                    # 修改代码行：新增 total_buy_value 和 total_sell_value 的初始化
                    total_buy_value = pd.Series(0, index=merged_hf.index); total_sell_value = pd.Series(0, index=merged_hf.index)
                    for i in range(1, 6):
                        weight = 1 / i
                        weighted_buy_vol += merged_hf[f'buy_volume{i}'] * weight
                        weighted_sell_vol += merged_hf[f'sell_volume{i}'] * weight
                        # 修改代码行：新增 total_buy_value 和 total_sell_value 的计算
                        total_buy_value += merged_hf[f'buy_volume{i}'] * merged_hf[f'buy_price{i}']
                        total_sell_value += merged_hf[f'sell_volume{i}'] * merged_hf[f'sell_price{i}']
                    merged_hf['imbalance'] = (weighted_buy_vol - weighted_sell_vol) / (weighted_buy_vol + weighted_sell_vol).replace(0, np.nan)
                    # 修改代码行：新增 liquidity_supply_ratio 列的计算
                    merged_hf['liquidity_supply_ratio'] = total_buy_value / total_sell_value.replace(0, np.nan)
                except Exception:
                    merged_hf['imbalance'] = np.nan
                    # 修改代码行：新增异常情况下的默认值
                    merged_hf['liquidity_supply_ratio'] = np.nan
                hf_analysis_df = merged_hf
        common_data = {
            'daily_total_volume': daily_total_volume, 'daily_total_amount': daily_total_amount,
            'daily_vwap': daily_vwap, 'atr': atr, 'day_open': day_open, 'day_close': day_close,
            'day_high': day_high, 'day_low': day_low
        }
        return hf_analysis_df, common_data

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
        【V12.1 · 诊断驾驶舱升级版】
        - 核心升级: 引入 A-系列 (Aggregation Service) 探针，监控服务层的数据流转和关键上游计算节点。
        """
        all_metrics_list = []
        attributed_minute_data_map = {}
        failures = []
        prev_metrics = memory.copy() if memory is not None else {}
        num_days = len(merged_df)
        enable_probe = self.debug_params.get('enable_mfca_probe', False)
        target_date_str = self.debug_params.get('target_date')
        for i, (trade_date, daily_data_series) in enumerate(merged_df.iterrows()):
            debug_mode = (i == num_days - 1)
            date_obj = trade_date.date()
            is_target_date = str(date_obj) == target_date_str
            # 修改代码行：升级【A.1 - 服务入口探针】
            if enable_probe and is_target_date:
                print(f"\n{'='*20} [探针 A.1 - 服务入口 @ {date_obj}] {'='*20}")
                print(f"  - 传入的日线级数据 (daily_data_series):")
                print(daily_data_series.to_string())
                print(f"  - 传入的记忆体 (prev_metrics): {prev_metrics}")
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
            daily_data_series_with_mem = pd.concat([daily_data_series, pd.Series(prev_metrics, name=daily_data_series.name)])
            attribution_weights_df = self._calculate_intraday_attribution_weights(intraday_data, daily_data_series_with_mem)
            # 修改代码行：新增【A.2 - 归因权重探针】
            if enable_probe and is_target_date:
                print(f"\n{'='*20} [探针 A.2 - 归因权重审计 @ {date_obj}] {'='*20}")
                print("  - 归因权重DataFrame统计信息 (.describe()):")
                weight_cols = [c for c in attribution_weights_df.columns if 'weight' in c]
                print(attribution_weights_df[weight_cols].describe().to_string())
            probabilistic_costs_dict, attributed_minute_df = self._calculate_probabilistic_costs(stock_code, attribution_weights_df, daily_data_series_with_mem, debug_mode=debug_mode)
            # 修改代码行：新增【A.3 - 概率成本探针】
            if enable_probe and is_target_date:
                print(f"\n{'='*20} [探针 A.3 - 概率成本审计 @ {date_obj}] {'='*20}")
                print("  - 计算出的概率成本字典:")
                print(probabilistic_costs_dict)
            # 修改代码行：新增【A.4 - 引擎调用探针】
            if enable_probe and is_target_date:
                print(f"\n{'='*20} [探针 A.4 - 引擎调用 @ {date_obj}] {'='*20}")
                print("  - 即将调用 _calculate_all_metrics_for_day 核心行为指标计算引擎...")
            day_metrics, _ = self._calculate_all_metrics_for_day(
                stock_code, daily_data_series_with_mem, intraday_data, attributed_minute_df, probabilistic_costs_dict,
                tick_data_for_day=tick_data_map.get(date_obj),
                level5_data_for_day=level5_data_map.get(date_obj),
                realtime_data_for_day=realtime_data_map.get(date_obj),
                debug_mode=debug_mode
            )
            # 修改代码行：升级【A.5 - 服务出口探针】
            if enable_probe and is_target_date:
                print(f"\n{'='*20} [探针 A.5 - 服务出口 @ {date_obj}] {'='*20}")
                print(f"  - 资金流指标计算完成。最终日度指标字典 (day_metrics) 预览:")
                # 打印部分关键指标以供快速检查
                preview_keys = ['main_force_net_flow_calibrated', 'avg_cost_main_buy', 'main_force_conviction_index', 'dip_absorption_power', 'rally_distribution_pressure']
                preview_dict = {k: day_metrics.get(k) for k in preview_keys if k in day_metrics}
                print(preview_dict)
            all_metrics_list.append(day_metrics)
            attributed_minute_data_map[date_obj] = attributed_minute_df.copy(deep=True)
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
        【V49.0 · 架构净化版】
        - 核心重构: 彻底移除为高频指标设置 np.nan 占位符的“反模式”，让方法回归单一职责。
        - 重构原因: 探针 V48.9 证明了问题根源不在于具体指标的计算，而在于数据组装流程的架构缺陷。
                     不完整的占位符列表导致了最终字段的缺失。
        - 核心实现:
          - 删除所有为 main_force_activity_ratio, main_force_conviction_index 等高频指标预设的 np.nan 占位符。
          - 此方法现在只计算并返回真正能从日线数据中派生的指标。
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
        # 修改代码块：移除所有为高频指标设置的 np.nan 占位符
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
        【V48.9 · CMF诊断探针版】
        - 核心升级: 引入 B-系列 (Behavioral Engine) 探针，监控引擎的输入数据健康度和计算过程。
        """
        is_target_date = str(daily_data.name.date()) == self.debug_params.get('target_date')
        enable_probe = self.debug_params.get('enable_mfca_probe', False)
        if enable_probe and is_target_date:
            print(f"\n{'='*20} [探针 B.1 - 引擎入口 @ {daily_data.name.date()}] {'='*20}")
            print("  - 进入 `_compute_all_behavioral_metrics` 行为指标计算引擎...")
        results = {}
        if intraday_data.empty:
            return results
        hf_analysis_df, common_data = self._prepare_behavioral_data(
            intraday_data, daily_data, tick_data, level5_data, realtime_data
        )
        if enable_probe and is_target_date:
            print(f"\n{'='*20} [探针 B.2 - 引擎输入审计 @ {daily_data.name.date()}] {'='*20}")
            print("  - 通用日线数据 (common_data):")
            print(f"    {common_data}")
            if not hf_analysis_df.empty:
                print("  - 高频分析DataFrame (hf_analysis_df) 健康检查:")
                print(f"    - Shape: {hf_analysis_df.shape}")
                print(f"    - 关键列 'ofi' 的空值数量: {hf_analysis_df['ofi'].isnull().sum()}")
                print(f"    - 关键列 'main_force_ofi' 的空值数量: {hf_analysis_df['main_force_ofi'].isnull().sum()}")
                print("    - 数据预览 (head):")
                print(hf_analysis_df[['mid_price', 'ofi', 'main_force_ofi', 'imbalance']].head(3).to_string())
            else:
                print("  - [!!!] 关键警告: 高频分析DataFrame (hf_analysis_df) 为空！所有高频指标将无法计算。")
        if not hf_analysis_df.empty:
            results.update(self._calculate_core_hf_metrics(hf_analysis_df, daily_data, common_data, is_target_date, enable_probe))
        results.update(self._calculate_vwap_related_metrics(intraday_data, common_data))
        results.update(self._calculate_opening_battle_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        results.update(self._calculate_shadow_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        results.update(self._calculate_dip_rally_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        results.update(self._calculate_reversal_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        results.update(self._calculate_panic_cascade_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        # 修改代码行：将 is_target_date 和 enable_probe 探针信号传递给 CMF 计算方法
        results.update(self._calculate_cmf_metrics(intraday_data, is_target_date, enable_probe))
        results.update(self._calculate_vpoc_metrics(intraday_data, common_data))
        results.update(self._calculate_liquidity_swap_metrics(intraday_data))
        results.update(self._calculate_closing_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        results.update(self._calculate_retail_sentiment_metrics(intraday_data, hf_analysis_df, daily_data, common_data, is_target_date, enable_probe))
        results.update(self._calculate_hidden_accumulation_metrics(intraday_data, hf_analysis_df, common_data, is_target_date, enable_probe))
        results.update(self._calculate_misc_minute_metrics(intraday_data, common_data))
        results.update(self._calculate_misc_daily_metrics(daily_data, main_force_net_flow_calibrated))
        return results

    def _calculate_core_hf_metrics(self, hf_analysis_df: pd.DataFrame, daily_data: pd.Series, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V47.7 · 信念标尺归一版】
        - 核心精炼: 对 main_force_conviction_index 的 Cost Tolerance 组件进行ATR归一化。
        - 精炼原因: 原有的成本差异百分比缺乏统一标尺，在不同价格、不同波动的股票间可比性差。
        - 核心实现: 将成本容忍度公式从 (买成本/卖成本)-1 修改为 (买成本-卖成本)/ATR，
                     使其度量的是“主力愿意承受相当于当日平均波幅百分之多少的成本差异”，
                     从而让信念指数的评估体系更加稳健和科学。
        """
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        metrics['main_force_ofi'] = hf_analysis_df['main_force_ofi'].sum()
        metrics['retail_ofi'] = hf_analysis_df['retail_ofi'].sum()
        mf_ofi_cumsum = hf_analysis_df['main_force_ofi'].cumsum().fillna(0)
        aggressiveness_component = 0.0
        if not mf_ofi_cumsum.empty and mf_ofi_cumsum.nunique() > 1:
            time_index = np.arange(len(mf_ofi_cumsum))
            trend_quality = np.corrcoef(time_index, mf_ofi_cumsum)[0, 1]
            trend_quality = np.nan_to_num(trend_quality)
            ofi_min, ofi_max = mf_ofi_cumsum.min(), mf_ofi_cumsum.max()
            closing_strength = (mf_ofi_cumsum.iloc[-1] - ofi_min) / (ofi_max - ofi_min) if (ofi_max - ofi_min) > 0 else 0.0
            closing_strength = np.nan_to_num(closing_strength)
            aggressiveness_component = trend_quality * closing_strength
        avg_cost_main_buy = daily_data.get('avg_cost_main_buy')
        avg_cost_main_sell = daily_data.get('avg_cost_main_sell')
        # 修改代码块：引入ATR对成本容忍度进行归一化
        cost_tolerance_component = 0.0
        if pd.notna(avg_cost_main_buy) and pd.notna(avg_cost_main_sell) and pd.notna(atr) and atr > 0:
            cost_tolerance_component = (avg_cost_main_buy - avg_cost_main_sell) / atr
        market_pressure_zone = hf_analysis_df['ofi'] < 0
        mf_resilience_ofi = hf_analysis_df.loc[market_pressure_zone, 'main_force_ofi'].clip(lower=0).sum()
        total_mf_positive_ofi = hf_analysis_df['main_force_ofi'].clip(lower=0).sum()
        resilience_component = mf_resilience_ofi / total_mf_positive_ofi if total_mf_positive_ofi > 0 else 0.0
        metrics['main_force_conviction_index'] = (0.4 * aggressiveness_component + 0.4 * cost_tolerance_component + 0.2 * resilience_component) * 100
        if enable_probe and is_target_date:
            print(f"\n--- [探针 B.3.1] main_force_conviction_index (主力信念) ---")
            # 修改代码行：更新探针输出，增加atr并标明Cost Tolerance已归一化
            print(f"    - 上游输入 avg_cost_main_buy: {avg_cost_main_buy:.4f}, avg_cost_main_sell: {avg_cost_main_sell:.4f}, atr: {atr:.4f}")
            print(f"    - 组件 Aggressiveness: {aggressiveness_component:.4f} = (TrendQuality {trend_quality:.4f} * ClosingStrength {closing_strength:.4f})")
            print(f"    - 组件 Cost Tolerance (norm by ATR): {cost_tolerance_component:.4f}, 组件 Resilience: {resilience_component:.4f}")
            print(f"    -> 最终得分: {metrics['main_force_conviction_index']:.2f}")
        mf_trades = hf_analysis_df[hf_analysis_df['amount'] > 200000].copy()
        if not mf_trades.empty and 'prev_mid_price' in mf_trades.columns:
            buy_trades_mask = mf_trades['type'] == 'B'
            sell_trades_mask = mf_trades['type'] == 'S'
            mf_trades['slippage'] = np.nan
            mf_trades.loc[buy_trades_mask, 'slippage'] = (mf_trades.loc[buy_trades_mask, 'price'] - mf_trades.loc[buy_trades_mask, 'prev_mid_price']).values
            mf_trades.loc[sell_trades_mask, 'slippage'] = (mf_trades.loc[sell_trades_mask, 'prev_mid_price'] - mf_trades.loc[sell_trades_mask, 'price']).values
            mf_trades['slippage'] = mf_trades['slippage'].clip(lower=0)
            total_mf_vol = mf_trades['volume'].sum()
            if total_mf_vol > 0:
                weighted_avg_slippage = (mf_trades['slippage'] * mf_trades['volume']).sum() / total_mf_vol
                if pd.notna(atr) and atr > 0:
                    metrics['main_force_price_impact_ratio'] = (weighted_avg_slippage / atr) * 100
                    if enable_probe and is_target_date:
                        print(f"\n--- [探针 B.3.2] main_force_price_impact_ratio (价格冲击) ---")
                        print(f"    - 主力总成交量: {total_mf_vol:,.0f}, 平均滑点: {weighted_avg_slippage:.4f}元, ATR: {atr:.2f}")
                        print(f"    -> 最终得分 (滑点/ATR %): {metrics['main_force_price_impact_ratio']:.2f}")
            if total_mf_vol > 0:
                offensive_buy_mask = (mf_trades['type'] == 'B') & (mf_trades['price'] >= mf_trades['sell_price1'])
                offensive_sell_mask = (mf_trades['type'] == 'S') & (mf_trades['price'] <= mf_trades['buy_price1'])
                offensive_volume = mf_trades[offensive_buy_mask | offensive_sell_mask]['volume'].sum()
                passive_volume = total_mf_vol - offensive_volume
                metrics['main_force_posture_index'] = ((offensive_volume - passive_volume) / total_mf_vol) * 100
                metrics['main_force_activity_ratio'] = (total_mf_vol / daily_total_volume) * 100 if daily_total_volume > 0 else np.nan
                if enable_probe and is_target_date:
                    print(f"\n--- [探针 B.3.3] main_force_posture_index (主力姿态) ---")
                    print(f"    - 主力总成交: {total_mf_vol:,.0f}, 进攻性成交: {offensive_volume:,.0f}, 被动性成交: {passive_volume:,.0f}")
                    print(f"    -> 最终得分: {metrics['main_force_posture_index']:.2f}")
                    print(f"\n--- [探针 B.3.4] main_force_activity_ratio (主力参与度) ---")
                    print(f"    - 主力总成交: {total_mf_vol:,.0f}, 当日总成交: {daily_total_volume:,.0f}")
                    print(f"    -> 最终得分: {metrics['main_force_activity_ratio']:.2f}")
                mf_buy_vol = mf_trades.loc[buy_trades_mask, 'volume'].sum()
                mf_sell_vol = mf_trades.loc[sell_trades_mask, 'volume'].sum()
                mf_total_activity_vol = mf_buy_vol + mf_sell_vol
                if mf_total_activity_vol > 0:
                    mf_net_vol = mf_buy_vol - mf_sell_vol
                    metrics['main_force_flow_directionality'] = (mf_net_vol / mf_total_activity_vol) * 100
                    if enable_probe and is_target_date:
                        print(f"\n--- [探针 B.3.5] main_force_flow_directionality (主力流向性) ---")
                        print(f"    - 主力买入量: {mf_buy_vol:,.0f}, 主力卖出量: {mf_sell_vol:,.0f}, 主力总活动量: {mf_total_activity_vol:,.0f}")
                        print(f"    -> 最终得分: {metrics['main_force_flow_directionality']:.2f}")
        large_orders_df = hf_analysis_df[hf_analysis_df['amount'] > 200000]
        if not large_orders_df.empty:
            metrics['observed_large_order_size_avg'] = large_orders_df['amount'].mean()
        up_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] > 0]
        down_ticks = hf_analysis_df[hf_analysis_df['mid_price_change'] < 0]
        if not up_ticks.empty and not down_ticks.empty and up_ticks['mid_price_change'].sum() > 0 and down_ticks['mid_price_change'].abs().sum() > 0:
            vol_per_tick_up = up_ticks['volume'].sum() / (up_ticks['mid_price_change'].sum() * 100)
            vol_per_tick_down = down_ticks['volume'].sum() / (down_ticks['mid_price_change'].abs().sum() * 100)
            if vol_per_tick_down > 1e-6:
                asymmetry_ratio = vol_per_tick_up / vol_per_tick_down
                metrics['micro_price_impact_asymmetry'] = np.log1p(asymmetry_ratio) if asymmetry_ratio > 0 else np.nan
        ask_clearing_mask = (hf_analysis_df['type'] == 'B') & (hf_analysis_df['price'] == hf_analysis_df['prev_a1_p'])
        ask_clearing_vol = hf_analysis_df.loc[ask_clearing_mask, 'volume'].sum()
        bid_clearing_mask = (hf_analysis_df['type'] == 'S') & (hf_analysis_df['price'] == hf_analysis_df['prev_b1_p'])
        bid_clearing_vol = hf_analysis_df.loc[bid_clearing_mask, 'volume'].sum()
        total_cleared_vol = ask_clearing_vol + bid_clearing_vol
        if daily_total_volume > 0:
            metrics['order_book_clearing_rate'] = (total_cleared_vol / daily_total_volume) * 100
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
                metrics['wash_trade_intensity'] = (wash_trade_vol / daily_total_volume) * 100
        except Exception:
            metrics['wash_trade_intensity'] = np.nan
        try:
            time_diffs = hf_analysis_df.index.to_series().diff().dt.total_seconds().fillna(0)
            if time_diffs.sum() > 0:
                metrics['order_book_imbalance'] = np.average(hf_analysis_df['imbalance'].dropna(), weights=time_diffs[hf_analysis_df['imbalance'].notna()]) * 100
                metrics['order_book_liquidity_supply'] = np.average(hf_analysis_df['liquidity_supply_ratio'].dropna(), weights=time_diffs[hf_analysis_df['liquidity_supply_ratio'].notna()])
            if 'market_vol_delta' in hf_analysis_df.columns and hf_analysis_df['imbalance'].var() > 1e-9 and hf_analysis_df['market_vol_delta'].var() > 1e-9:
                correlation_value = hf_analysis_df['imbalance'].corr(hf_analysis_df['market_vol_delta'])
                metrics['imbalance_effectiveness'] = correlation_value
                if enable_probe and is_target_date:
                    print(f"\n--- [探针 B.3.6] imbalance_effectiveness (盘口诚实度) ---")
                    print(f"    - Imbalance vs MarketVolDelta Corr: {correlation_value:.4f}")
                    print(f"    -> 最终得分: {metrics['imbalance_effectiveness']:.2f}")
        except Exception:
            pass
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
                metrics['large_order_pressure'] = (pressure_strength / total_trading_seconds) * 100
                metrics['large_order_support'] = (support_strength / total_trading_seconds) * 100
        except Exception:
            metrics['large_order_pressure'] = np.nan; metrics['large_order_support'] = np.nan
        try:
            buy_exhaustion_mask = hf_analysis_df['sell_price1'] > hf_analysis_df['prev_a1_p']
            buy_exhausted_vol = hf_analysis_df.loc[buy_exhaustion_mask, 'prev_a1_v'].sum()
            sell_exhaustion_mask = hf_analysis_df['buy_price1'] < hf_analysis_df['prev_b1_p']
            sell_exhausted_vol = hf_analysis_df.loc[sell_exhaustion_mask, 'prev_b1_v'].sum()
            if daily_total_volume > 0:
                metrics['buy_quote_exhaustion_rate'] = (buy_exhausted_vol / daily_total_volume) * 100
                metrics['sell_quote_exhaustion_rate'] = (sell_exhausted_vol / daily_total_volume) * 100
        except Exception:
            metrics['buy_quote_exhaustion_rate'] = np.nan; metrics['sell_quote_exhaustion_rate'] = np.nan
        mf_ofi_series = hf_analysis_df['main_force_ofi']
        price_change_series = hf_analysis_df['mid_price_change']
        if mf_ofi_series.var() > 0 and price_change_series.var() > 0:
            metrics['ofi_price_impact_factor'] = mf_ofi_series.corr(price_change_series)
        return metrics

    def _calculate_opening_battle_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V47.0 · 模块化重构版】
        新增方法: 计算开盘战役结果指标。
        """
        from datetime import time
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        opening_battle_df = intraday_data[(intraday_data.index.time >= time(9, 30)) & (intraday_data.index.time <= time(9, 45))]
        if not opening_battle_df.empty and len(opening_battle_df) > 1 and pd.notna(atr) and atr > 0:
            if not hf_analysis_df.empty:
                opening_hf_df = hf_analysis_df[(hf_analysis_df.index.time >= time(9, 30)) & (hf_analysis_df.index.time <= time(9, 45))]
                if not opening_hf_df.empty:
                    price_gain_hf = (opening_hf_df['price'].iloc[-1] - opening_hf_df['price'].iloc[0]) / atr
                    mf_ofi_opening = opening_hf_df['main_force_ofi'].sum()
                    total_abs_ofi_opening = opening_hf_df['ofi'].abs().sum()
                    mf_ofi_dominance = mf_ofi_opening / total_abs_ofi_opening if total_abs_ofi_opening > 0 else 0
                    metrics['opening_battle_result'] = price_gain_hf * (1 + mf_ofi_dominance) * 100
                    if enable_probe and is_target_date:
                        print(f"  [探针] opening_battle_result (高频-开盘战役) 计算:")
                        print(f"    - Price Gain (norm by ATR): {price_gain_hf:.4f}, MF OFI Dominance: {mf_ofi_dominance:.4f}")
                        print(f"    -> Final Score: {metrics['opening_battle_result']:.2f}")
            else:
                if 'close' in opening_battle_df.columns and 'open' in opening_battle_df.columns and 'vol_shares' in opening_battle_df.columns and 'minute_vwap' in opening_battle_df.columns and 'main_force_net_vol' in opening_battle_df.columns:
                    price_gain = (opening_battle_df['close'].iloc[-1] - opening_battle_df['open'].iloc[0]) / atr
                    battle_amount = (opening_battle_df['vol_shares'] * opening_battle_df['minute_vwap']).sum()
                    if battle_amount > 0:
                        mf_power = opening_battle_df['main_force_net_vol'].sum() * opening_battle_df['minute_vwap'].mean() / battle_amount
                        metrics['opening_battle_result'] = np.sign(price_gain) * np.sqrt(abs(price_gain)) * (1 + mf_power) * 100
        return metrics

    def _calculate_shadow_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V48.1 · 影线博弈对称版】
        - 核心升级: 对 upper_shadow_selling_pressure 指标进行高频化重构，使其与 lower_shadow_absorption_strength 形成逻辑对称。
        - 升级原因: 原有分钟级上影线指标过于简单，无法揭示“拉高出货”等复杂博弈行为。
        - 核心实现:
          - 高频路径: 引入与下影线指标对称的“派发流量”、“价格拒绝”、“盘口压制(OFI)”三组件模型。
          - 降级路径: 保留原分钟级逻辑作为无高频数据时的备用方案。
        """
        import numpy as np
        metrics = {}
        day_open, day_close = common_data['day_open'], common_data['day_close']
        day_high, day_low = common_data['day_high'], common_data['day_low']
        if pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low):
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            day_range = day_high - day_low
            if day_low < body_low and day_range > 0:
                RecoveryComponent = (body_low - day_low) / day_range
                if not hf_analysis_df.empty:
                    hf_shadow_zone = hf_analysis_df[hf_analysis_df['price'] < body_low]
                    if not hf_shadow_zone.empty:
                        mf_trades_in_shadow = hf_shadow_zone[hf_shadow_zone['amount'] > 200000]
                        mf_buy_vol_in_shadow = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'B']['volume'].sum()
                        mf_sell_vol_in_shadow = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'S']['volume'].sum()
                        mf_net_vol_in_shadow = mf_buy_vol_in_shadow - mf_sell_vol_in_shadow
                        total_vol_in_shadow = hf_shadow_zone['volume'].sum()
                        FlowComponent = np.tanh(mf_net_vol_in_shadow / total_vol_in_shadow) if total_vol_in_shadow > 0 else 0.0
                        mf_ofi_in_shadow = hf_shadow_zone['main_force_ofi'].sum()
                        total_abs_ofi_day = hf_analysis_df['ofi'].abs().sum()
                        OFI_Component = mf_ofi_in_shadow / total_abs_ofi_day if total_abs_ofi_day > 0 else 0.0
                        metrics['lower_shadow_absorption_strength'] = (0.5 * FlowComponent + 0.3 * RecoveryComponent + 0.2 * OFI_Component) * 100
                        if enable_probe and is_target_date:
                            print(f"  [探针] lower_shadow_absorption_strength (高频-下影线) 计算:")
                            print(f"    - body_low: {body_low:.2f}, day_low: {day_low:.2f}, day_range: {day_range:.2f}")
                            print(f"    - HF FlowComponent (from net vol): {FlowComponent:.4f}, RecoveryComponent: {RecoveryComponent:.4f}, OFI_Component: {OFI_Component:.4f}")
                            print(f"    -> Final Score: {metrics['lower_shadow_absorption_strength']:.2f}")
                else:
                    lower_shadow_df = intraday_data[intraday_data['low'] < body_low]
                    if not lower_shadow_df.empty and 'vol_shares' in lower_shadow_df.columns and 'main_force_net_vol' in lower_shadow_df.columns and lower_shadow_df['vol_shares'].sum() > 0:
                        shadow_volume = lower_shadow_df['vol_shares'].sum()
                        mf_net_in_shadow = lower_shadow_df['main_force_net_vol'].sum()
                        FlowComponent = np.tanh(mf_net_in_shadow / shadow_volume)
                        metrics['lower_shadow_absorption_strength'] = (0.7 * FlowComponent + 0.3 * RecoveryComponent) * 100
            # 修改代码块：为上影线新增高频计算路径
            if day_high > body_high and day_range > 0:
                RejectionComponent = (day_high - body_high) / day_range
                if not hf_analysis_df.empty:
                    hf_shadow_zone = hf_analysis_df[hf_analysis_df['price'] > body_high]
                    if not hf_shadow_zone.empty:
                        mf_trades_in_shadow = hf_shadow_zone[hf_shadow_zone['amount'] > 200000]
                        mf_buy_vol_in_shadow = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'B']['volume'].sum()
                        mf_sell_vol_in_shadow = mf_trades_in_shadow[mf_trades_in_shadow['type'] == 'S']['volume'].sum()
                        mf_net_vol_in_shadow = mf_buy_vol_in_shadow - mf_sell_vol_in_shadow
                        total_vol_in_shadow = hf_shadow_zone['volume'].sum()
                        FlowComponent = np.tanh(-mf_net_vol_in_shadow / total_vol_in_shadow) if total_vol_in_shadow > 0 else 0.0
                        mf_ofi_in_shadow = hf_shadow_zone['main_force_ofi'].sum()
                        total_abs_ofi_day = hf_analysis_df['ofi'].abs().sum()
                        OFI_Component = -mf_ofi_in_shadow / total_abs_ofi_day if total_abs_ofi_day > 0 else 0.0
                        metrics['upper_shadow_selling_pressure'] = (0.5 * FlowComponent + 0.3 * RejectionComponent + 0.2 * OFI_Component) * 100
                        if enable_probe and is_target_date:
                            print(f"  [探针] upper_shadow_selling_pressure (高频-上影线) 计算:")
                            print(f"    - body_high: {body_high:.2f}, day_high: {day_high:.2f}, day_range: {day_range:.2f}")
                            print(f"    - HF FlowComponent (from net vol): {FlowComponent:.4f}, RejectionComponent: {RejectionComponent:.4f}, OFI_Component: {OFI_Component:.4f}")
                            print(f"    -> Final Score: {metrics['upper_shadow_selling_pressure']:.2f}")
                else:
                    upper_shadow_df = intraday_data[intraday_data['high'] > body_high]
                    if not upper_shadow_df.empty and 'vol_shares' in upper_shadow_df.columns and 'main_force_net_vol' in upper_shadow_df.columns and upper_shadow_df['vol_shares'].sum() > 0:
                        mf_sell_in_shadow = abs(upper_shadow_df[upper_shadow_df['main_force_net_vol'] < 0]['main_force_net_vol'].sum())
                        metrics['upper_shadow_selling_pressure'] = (mf_sell_in_shadow / upper_shadow_df['vol_shares'].sum())
        return metrics

    def _calculate_dip_rally_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V47.4 · 操盘技艺升维版】
        - 核心升级: 对 rally_distribution_pressure 指标进行逻辑升维，用更鲁棒的“操盘技艺”组件替代原有的“诱饵效率”。
        - 升级原因: 发现原基于OFI的“诱饵效率”会被高明的“假买真卖”手法干扰，导致指标失灵。
        - 核心实现: 新的 artistry_component 直接基于主力在拉升期间的“净成交量”进行评判，
                     能更准确地奖励“边拉边撤”的高明派发行为，而不会被虚假的盘口意图迷惑。
        """
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        metrics = {}
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns:
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            if not hf_analysis_df.empty and pd.notna(daily_vwap):
                absorption_zone_hf = hf_analysis_df[hf_analysis_df['price'] < daily_vwap]
                if not absorption_zone_hf.empty:
                    offensive_ofi = absorption_zone_hf['main_force_ofi'].clip(lower=0).sum()
                    total_abs_ofi_day = hf_analysis_df['ofi'].abs().sum()
                    OffensiveOFI_Component = offensive_ofi / total_abs_ofi_day if total_abs_ofi_day > 0 else 0.0
                    mf_buy_cost_in_dip = (absorption_zone_hf['price'] * absorption_zone_hf['main_force_ofi'].clip(lower=0)).sum() / offensive_ofi if offensive_ofi > 0 else np.nan
                    CostAdvantage_Component = (daily_vwap - mf_buy_cost_in_dip) / atr if pd.notna(mf_buy_cost_in_dip) and pd.notna(atr) and atr > 0 else 0.0
                    mf_trades_hf = hf_analysis_df[hf_analysis_df['amount'] > 200000]
                    total_mf_buy_vol_day_hf = mf_trades_hf[mf_trades_hf['type'] == 'B']['volume'].sum()
                    mf_buy_vol_in_dip_hf = mf_trades_hf[(mf_trades_hf['type'] == 'B') & (mf_trades_hf['price'] < daily_vwap)]['volume'].sum()
                    VolumeDominance_Component = mf_buy_vol_in_dip_hf / total_mf_buy_vol_day_hf if total_mf_buy_vol_day_hf > 0 else 0.0
                    metrics['dip_absorption_power'] = (0.5 * OffensiveOFI_Component + 0.3 * CostAdvantage_Component + 0.2 * VolumeDominance_Component) * 100
                    if enable_probe and is_target_date:
                        print(f"  [探针] dip_absorption_power (高频-战略吸筹) 计算:")
                        print(f"    - OffensiveOFI: {OffensiveOFI_Component:.4f}, CostAdvantage: {CostAdvantage_Component:.4f}, VolumeDominance (HF): {VolumeDominance_Component:.4f}")
                        print(f"    -> Final Score: {metrics['dip_absorption_power']:.2f}")
                total_rally_price_change = 0
                total_deceptive_pressure = 0
                daily_retail_trades = hf_analysis_df[hf_analysis_df['amount'] < 50000]
                daily_avg_retail_buy_vol_per_min = 0
                if not daily_retail_trades.empty:
                    daily_retail_buy_vol = daily_retail_trades[daily_retail_trades['type'] == 'B']['volume'].sum()
                    if daily_retail_buy_vol > 0:
                        daily_avg_retail_buy_vol_per_min = daily_retail_buy_vol / 240
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
                            total_vol_in_rally_hf = rally_hf_df['volume'].sum()
                            mf_trades_in_rally = rally_hf_df[rally_hf_df['amount'] > 200000]
                            # 修改代码块：废弃 bait_efficiency，引入 artistry_component
                            mf_buy_vol_in_rally_hf = mf_trades_in_rally[mf_trades_in_rally['type'] == 'B']['volume'].sum()
                            mf_sell_vol_in_rally_hf = mf_trades_in_rally[mf_trades_in_rally['type'] == 'S']['volume'].sum()
                            mf_net_vol_in_rally = mf_buy_vol_in_rally_hf - mf_sell_vol_in_rally_hf
                            artistry_component = (1 + np.tanh(-mf_net_vol_in_rally / total_vol_in_rally_hf)) if total_vol_in_rally_hf > 0 else 1.0
                            distribution_dominance = mf_sell_vol_in_rally_hf / total_vol_in_rally_hf if total_vol_in_rally_hf > 0 else 0.0
                            follower_frenzy = 0.0
                            retail_hf_in_rally = rally_hf_df[rally_hf_df['amount'] < 50000]
                            if not retail_hf_in_rally.empty:
                                retail_buy_trades_in_rally = retail_hf_in_rally[retail_hf_in_rally['type'] == 'B']
                                total_retail_buy_vol_in_rally = retail_buy_trades_in_rally['volume'].sum()
                                cost_component = 0.0
                                if total_retail_buy_vol_in_rally > 0 and price_change_in_rally > 0:
                                    retail_buy_vwap = (retail_buy_trades_in_rally['price'] * retail_buy_trades_in_rally['volume']).sum() / total_retail_buy_vol_in_rally
                                    cost_component = (retail_buy_vwap - window_df['minute_vwap'].iloc[0]) / price_change_in_rally
                                rally_duration_mins = (end_time - start_time).total_seconds() / 60
                                avg_retail_buy_vol_in_rally = total_retail_buy_vol_in_rally / rally_duration_mins if rally_duration_mins > 0 else 0
                                volume_spike_component = np.log1p(avg_retail_buy_vol_in_rally / daily_avg_retail_buy_vol_per_min) if daily_avg_retail_buy_vol_per_min > 0 else 0.0
                                aggressive_buy_mask = retail_buy_trades_in_rally['price'] >= retail_buy_trades_in_rally['sell_price1']
                                aggressive_buy_vol = retail_buy_trades_in_rally[aggressive_buy_mask]['volume'].sum()
                                aggression_component = aggressive_buy_vol / total_retail_buy_vol_in_rally if total_retail_buy_vol_in_rally > 0 else 0.0
                                follower_frenzy = cost_component * volume_spike_component * (1 + aggression_component)
                            # 修改代码行：使用新的 artistry_component
                            deceptive_pressure_score = artistry_component * distribution_dominance * follower_frenzy
                            total_deceptive_pressure += deceptive_pressure_score * price_change_in_rally
                if total_rally_price_change > 0:
                    weighted_avg_pressure = total_deceptive_pressure / total_rally_price_change
                    metrics['rally_distribution_pressure'] = weighted_avg_pressure * 100
                    if enable_probe and is_target_date:
                        print(f"  [探针] rally_distribution_pressure (高频-诡道派发) 计算:")
                        print(f"    - Weighted Avg Deceptive Artistry: {weighted_avg_pressure:.4f}")
                        # 修改代码行：更新探针输出
                        print(f"      - [Components] Artistry(NetVol): {artistry_component:.2f}, DistDom(GrossSell): {distribution_dominance:.2f}, Frenzy: {follower_frenzy:.2f}")
                        print(f"    -> Final Score: {metrics['rally_distribution_pressure']:.2f}")
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
                                    metrics['dip_absorption_power'] = (absorption_vol / daily_total_volume) * cost_deviation * 100
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
                    metrics['rally_distribution_pressure'] = (rally_dist_vol / total_rally_vol) * 100
        return metrics

    def _calculate_reversal_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V47.6 · 反转力度稳健重构版】
        - 核心重构: 将 reversal_power_index 的计算范式从脆弱的“乘法模型”升级为稳健的“加权加法模型”。
        - 重构原因: 发现原公式中的 Exhaustion (衰竭) 组件在强趋势行情中经常失效为0，导致整个指标评判逻辑出现偏差。
        - 核心实现: 废弃不可靠的 Exhaustion 组件，聚焦于“价格收复”和“主力反击”两大核心要素，
                     采用 0.6 * PriceRecovery + 0.4 * CounterAttack 的加权求和方式，使指标更稳定、更可靠。
        """
        import numpy as np
        metrics = {}
        day_open, day_close = common_data['day_open'], common_data['day_close']
        day_high, day_low = common_data['day_high'], common_data['day_low']
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        if len(intraday_data) >= 10 and pd.notna(day_open) and pd.notna(day_close) and pd.notna(day_high) and pd.notna(day_low) and pd.notna(atr) and atr > 0 and daily_total_volume > 0:
            day_range = day_high - day_low
            if day_range > 0:
                is_v_shape = (day_close - day_open) > 0
                turn_point_idx = np.argmin(intraday_data['low'].values) if is_v_shape else np.argmax(intraday_data['high'].values)
                if 0 < turn_point_idx < len(intraday_data) - 1:
                    if not hf_analysis_df.empty:
                        turn_point_time = intraday_data.index[turn_point_idx]
                        reversal_phase_hf = hf_analysis_df[hf_analysis_df.index >= turn_point_time]
                        if not reversal_phase_hf.empty:
                            turn_point_price = intraday_data.iloc[turn_point_idx]['low'] if is_v_shape else intraday_data.iloc[turn_point_idx]['high']
                            PriceRecovery_Component = abs(day_close - turn_point_price) / day_range
                            # 修改代码块：移除 Exhaustion_Component 的计算
                            reversal_ofi = reversal_phase_hf['main_force_ofi']
                            CounterAttack_Component = np.tanh(reversal_ofi.sum() / daily_total_volume)
                            # 修改代码块：采用新的加权加法模型
                            power_score = (0.6 * PriceRecovery_Component + 0.4 * CounterAttack_Component)
                            metrics['reversal_power_index'] = power_score * 100
                            if enable_probe and is_target_date:
                                print(f"  [探针] reversal_power_index (高频-V型反转) 计算:")
                                # 修改代码块：更新探针输出
                                print(f"    - PriceRecovery: {PriceRecovery_Component:.4f} (权重 0.6), CounterAttack: {CounterAttack_Component:.4f} (权重 0.4)")
                                print(f"    -> Final Score: {metrics['reversal_power_index']:.2f}")
                    else:
                        initial_phase = intraday_data.iloc[:turn_point_idx]
                        reversal_phase = intraday_data.iloc[turn_point_idx:]
                        vol_initial, vol_reversal = initial_phase['vol_shares'].sum(), reversal_phase['vol_shares'].sum()
                        if vol_initial > 0 and vol_reversal > 0 and 'main_force_net_vol' in reversal_phase.columns:
                            turn_point_vwap = intraday_data['minute_vwap'].iloc[turn_point_idx]
                            price_recovery = abs(day_close - turn_point_vwap) / day_range
                            vol_shift = np.log1p(vol_reversal / vol_initial)
                            reversal_mf_net_vol = reversal_phase['main_force_net_vol'].sum()
                            reversal_conviction = reversal_mf_net_vol / vol_reversal if vol_reversal > 0 else 0
                            # 保持分钟级数据的乘法逻辑，因为它更依赖于成交量 shift
                            power_score = price_recovery * vol_shift * reversal_conviction
                            metrics['reversal_power_index'] = power_score if is_v_shape else -power_score
        return metrics

    def _calculate_closing_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V47.1 · 类型修复版】
        - 核心修复: 修正了在筛选14:57之前的高频数据时，因直接比较 DatetimeIndex 与 time 对象导致的 TypeError。
                     通过使用 .index.time 访问器，确保比较的是两个同为时间类型的对象。
        """
        from datetime import time
        import numpy as np
        metrics = {}
        day_close = common_data['day_close']
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and pd.notna(atr) and atr > 0:
            auction_df = intraday_data[intraday_data.index.time >= time(14, 57)]
            if not auction_df.empty:
                avg_minute_vol = continuous_trading_df['vol_shares'].mean()
                auction_vol = auction_df['vol_shares'].sum()
                VolumeAnomaly = np.log1p((auction_vol / 3) / avg_minute_vol) if avg_minute_vol > 0 else 0.0
                if not hf_analysis_df.empty:
                    # 修改代码行：使用 .index.time 提取时间部分进行比较，修复TypeError
                    pre_auction_df = hf_analysis_df[hf_analysis_df.index.time < time(14, 57)]
                    if not pre_auction_df.empty:
                        pre_auction_snapshot = pre_auction_df.iloc[-1]
                        pre_auction_mid = pre_auction_snapshot['mid_price']
                        pre_auction_imbalance = pre_auction_snapshot['imbalance']
                        PriceDeviation = (day_close - pre_auction_mid) / atr if pd.notna(pre_auction_mid) else 0.0
                        Deception = -np.sign(PriceDeviation) * pre_auction_imbalance if pd.notna(pre_auction_imbalance) else 0.0
                        metrics['closing_auction_ambush'] = PriceDeviation * VolumeAnomaly * (1 + Deception) * 100
                        if enable_probe and is_target_date:
                            print(f"  [探针] closing_auction_ambush (高频-竞价伏击) 计算:")
                            print(f"    - pre_auction_mid: {pre_auction_mid:.2f}, day_close: {day_close:.2f}, atr: {atr:.2f}")
                            print(f"    - PriceDeviation: {PriceDeviation:.4f}, VolumeAnomaly: {VolumeAnomaly:.4f}, Deception: {Deception:.4f}")
                            print(f"    -> Final Score: {metrics['closing_auction_ambush']:.2f}")
                else:
                    pre_auction_close = continuous_trading_df['close'].iloc[-1]
                    PriceImpact = (day_close - pre_auction_close) / atr if pd.notna(pre_auction_close) else 0.0
                    metrics['closing_auction_ambush'] = PriceImpact * VolumeAnomaly * 100
            posturing_df = continuous_trading_df[continuous_trading_df.index.time >= time(14, 30)]
            if pd.notna(daily_vwap) and not posturing_df.empty:
                if not hf_analysis_df.empty:
                    posturing_hf_df = hf_analysis_df[hf_analysis_df.index.time >= time(14, 30)]
                    if not posturing_hf_df.empty:
                        time_diffs = posturing_hf_df.index.to_series().diff().dt.total_seconds().fillna(0)
                        if time_diffs.sum() > 0:
                            avg_imbalance = np.average(posturing_hf_df['imbalance'].dropna(), weights=time_diffs[posturing_hf_df['imbalance'].notna()])
                            avg_spread = (posturing_hf_df['sell_price1'] - posturing_hf_df['buy_price1']).mean()
                            normalized_imbalance = avg_imbalance * (avg_spread / atr) if pd.notna(avg_spread) and avg_spread > 0 else 0
                            metrics['pre_closing_posturing'] = normalized_imbalance * 100
                            if enable_probe and is_target_date:
                                print(f"  [探针] pre_closing_posturing (高频-收盘姿态) 计算:")
                                print(f"    - Avg Imbalance: {avg_imbalance:.4f}, Avg Spread: {avg_spread:.4f}, ATR: {atr:.2f}")
                                print(f"    -> Final Score: {metrics['pre_closing_posturing']:.2f}")
                else:
                    if 'vol_shares' in posturing_df.columns and 'minute_vwap' in posturing_df.columns and 'main_force_net_vol' in posturing_df.columns:
                        posturing_vwap = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum() / posturing_df['vol_shares'].sum()
                        price_posture = (posturing_vwap - daily_vwap) / atr
                        posturing_amount = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum()
                        if posturing_amount > 0:
                            force_posture = (posturing_df['main_force_net_vol'].sum() * posturing_vwap) / posturing_amount
                            metrics['pre_closing_posturing'] = (0.6 * price_posture + 0.4 * force_posture) * 100
        return metrics

    def _calculate_hidden_accumulation_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V47.0 · 模块化重构版】
        新增方法: 计算隐蔽吸筹强度指标。
        """
        import numpy as np
        metrics = {}
        daily_vwap = common_data['daily_vwap']
        if not hf_analysis_df.empty and pd.notna(daily_vwap):
            absorption_zone = hf_analysis_df[hf_analysis_df['mid_price'] < daily_vwap].copy()
            if not absorption_zone.empty:
                passive_absorption_mask = (absorption_zone['type'] == 'S') & (absorption_zone['price'] <= absorption_zone['prev_b1_p'])
                passive_absorption_vol = absorption_zone.loc[passive_absorption_mask, 'volume'].sum()
                total_vol_below_vwap = absorption_zone['volume'].sum()
                passive_absorption_component = passive_absorption_vol / total_vol_below_vwap if total_vol_below_vwap > 0 else 0.0
                impact_suppression_component = 0.0
                if not absorption_zone.empty and absorption_zone['main_force_ofi'].var() > 0 and absorption_zone['mid_price_change'].var() > 0:
                    correlation = absorption_zone['main_force_ofi'].corr(absorption_zone['mid_price_change'])
                    impact_suppression_component = -np.tanh(correlation) if pd.notna(correlation) else 0.0
                total_book_depth = absorption_zone[[f'{d}_volume{i}' for d in ['buy', 'sell'] for i in range(1, 6)]].sum(axis=1)
                bid_depth_ratio = absorption_zone['buy_volume1'] / total_book_depth.replace(0, np.nan)
                liquidity_commitment_component = bid_depth_ratio.mean() if not bid_depth_ratio.empty else 0.0
                metrics['hidden_accumulation_intensity'] = (0.5 * passive_absorption_component + 0.3 * impact_suppression_component + 0.2 * liquidity_commitment_component) * 100
                if enable_probe and is_target_date:
                    print(f"  [探针] hidden_accumulation_intensity (高频-隐蔽吸筹) 计算:")
                    print(f"    - PassiveAbsorption (Efficiency): {passive_absorption_component:.4f}, ImpactSuppression: {impact_suppression_component:.4f}, LiquidityCommitment: {liquidity_commitment_component:.4f}")
                    print(f"    -> Final Score: {metrics['hidden_accumulation_intensity']:.2f}")
        else:
            dip_or_flat_df = intraday_data[intraday_data['close'] <= intraday_data['open']]
            if not dip_or_flat_df.empty:
                total_vol_dip = dip_or_flat_df['vol_shares'].sum()
                if total_vol_dip > 0 and 'main_force_net_vol' in dip_or_flat_df.columns:
                    mf_net_buy_on_dip = dip_or_flat_df['main_force_net_vol'].clip(lower=0).sum()
                    metrics['hidden_accumulation_intensity'] = (mf_net_buy_on_dip / total_vol_dip) * 100
        return metrics

    def _calculate_vwap_related_metrics(self, intraday_data: pd.DataFrame, common_data: dict) -> dict:
        """
        【V47.0 · 模块化重构版】
        新增方法: 计算与VWAP相关的指标。
        """
        import numpy as np
        import pandas as pd
        metrics = {}
        daily_vwap = common_data['daily_vwap']
        daily_total_volume = common_data['daily_total_volume']
        atr = common_data['atr']
        if pd.notna(daily_vwap) and daily_total_volume > 0 and pd.notna(atr) and atr > 0 and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            price_deviation_value = (intraday_data['minute_vwap'] - daily_vwap) * intraday_data['vol_shares']
            metrics['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
            price_dev_series = intraday_data['minute_vwap'] - daily_vwap
            mf_net_flow_series = intraday_data['main_force_net_vol']
            if price_dev_series.var() != 0 and mf_net_flow_series.var() != 0 and len(price_dev_series) > 1:
                correlation = price_dev_series.corr(mf_net_flow_series)
                metrics['main_force_vwap_guidance'] = correlation if pd.notna(correlation) else np.nan
            position_vs_vwap = np.sign(intraday_data['minute_vwap'] - daily_vwap)
            crossings = position_vs_vwap.diff().ne(0)
            metrics['vwap_crossing_intensity'] = intraday_data.loc[crossings, 'vol_shares'].sum() / daily_total_volume
            twap = intraday_data['minute_vwap'].mean()
            if pd.notna(twap) and twap > 0:
                metrics['vwap_structure_skew'] = (daily_vwap - twap) / twap * 100
        return metrics

    def _calculate_cmf_metrics(self, intraday_data: pd.DataFrame, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V48.9 · CMF诊断探针版】
        - 核心升级: 移除 V48.8 的防御性硬化编码，转而植入诊断探针，以定位 cmf_divergence_score 缺失的根本原因。
        - 升级原因: 遵循指令，采用主动诊断而非被动防御的策略来解决 60 vs 61 的数据完整性问题。
        - 核心实现:
          - 恢复 metrics 字典的按需创建模式。
          - 新增一个探针，在目标调试日期打印 CMF 指标的中间值，并明确报告背离度指标的计算条件是否满足。
        """
        import numpy as np
        import pandas as pd
        # 修改代码块：移除指标硬化逻辑，恢复为按需创建
        metrics = {}
        if 'high' in intraday_data.columns and 'low' in intraday_data.columns and 'close' in intraday_data.columns and 'vol_shares' in intraday_data.columns and 'main_force_net_vol' in intraday_data.columns:
            high_low_range = intraday_data['high'] - intraday_data['low']
            money_flow_multiplier = np.where(
                high_low_range > 0,
                ((intraday_data['close'] - intraday_data['low']) - (intraday_data['high'] - intraday_data['close'])) / high_low_range,
                0
            )
            cmf_period = 20
            vol_sum = intraday_data['vol_shares'].rolling(window=cmf_period, min_periods=1).sum()
            mfv_main_force = money_flow_multiplier * intraday_data['main_force_net_vol']
            mfv_sum_main_force = mfv_main_force.rolling(window=cmf_period, min_periods=1).sum()
            main_force_cmf_series = mfv_sum_main_force / vol_sum
            mfv_holistic = money_flow_multiplier * intraday_data['vol_shares']
            mfv_sum_holistic = mfv_holistic.rolling(window=cmf_period, min_periods=1).sum()
            holistic_cmf_series = mfv_sum_holistic / vol_sum
            axiom_mfv = intraday_data['main_force_net_vol'] * abs(money_flow_multiplier)
            axiom_mfv_sum = axiom_mfv.rolling(window=cmf_period, min_periods=1).sum()
            main_force_axiom_index_series = axiom_mfv_sum / vol_sum
            main_force_cmf_value = main_force_cmf_series.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1] if not main_force_cmf_series.empty else None
            holistic_cmf_value = holistic_cmf_series.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1] if not holistic_cmf_series.empty else None
            axiom_index_value = main_force_axiom_index_series.replace([np.inf, -np.inf], np.nan).ffill().iloc[-1] if not main_force_axiom_index_series.empty else None
            if pd.notna(main_force_cmf_value):
                metrics['main_force_cmf'] = main_force_cmf_value
            if pd.notna(holistic_cmf_value):
                metrics['holistic_cmf'] = holistic_cmf_value
            if pd.notna(axiom_index_value):
                metrics['main_force_flow_axiom_index'] = axiom_index_value
            # 修改代码块：新增 CMF 背离度计算的诊断探针
            if enable_probe and is_target_date:
                print(f"  [探针] cmf_divergence_score (CMF背离度) 计算前置检查:")
                print(f"    - main_force_cmf_value: {main_force_cmf_value}")
                print(f"    - holistic_cmf_value: {holistic_cmf_value}")
                condition_met = pd.notna(main_force_cmf_value) and pd.notna(holistic_cmf_value)
                print(f"    - 计算条件 (两者均非NaN): {'满足' if condition_met else '不满足'}")
                if not condition_met:
                    print("    -> [诊断结论] 因上游指标存在NaN，cmf_divergence_score 将不会被计算。")
            if pd.notna(main_force_cmf_value) and pd.notna(holistic_cmf_value):
                metrics['cmf_divergence_score'] = (main_force_cmf_value - holistic_cmf_value) * 100
        return metrics

    def _calculate_vpoc_metrics(self, intraday_data: pd.DataFrame, common_data: dict) -> dict:
        """
        【V47.0 · 模块化重构版】
        新增方法: 计算成交量分布(Volume Profile)相关指标。
        """
        import pandas as pd
        import numpy as np
        metrics = {}
        daily_total_amount = common_data['daily_total_amount']
        if 'main_force_net_vol' in intraday_data.columns and 'minute_vwap' in intraday_data.columns and 'vol_shares' in intraday_data.columns:
            vp_global = intraday_data.groupby(pd.cut(intraday_data['minute_vwap'], bins=30, duplicates='drop'))['vol_shares'].sum()
            global_vpoc_price = np.nan
            if not vp_global.empty:
                vpoc_interval = vp_global.idxmax()
                global_vpoc_price = vpoc_interval.mid
                peak_zone_df = intraday_data[(intraday_data['minute_vwap'] >= vpoc_interval.left) & (intraday_data['minute_vwap'] < vpoc_interval.right)]
                if not peak_zone_df.empty:
                    mf_net_vol_on_peak = peak_zone_df['main_force_net_vol'].sum()
                    if daily_total_amount > 0:
                        normalized_mf_on_peak_flow = np.tanh((mf_net_vol_on_peak * global_vpoc_price) / daily_total_amount)
                        metrics['main_force_on_peak_flow'] = normalized_mf_on_peak_flow
            mf_net_buy_df = intraday_data[intraday_data['main_force_net_vol'] > 0]
            if not mf_net_buy_df.empty:
                vp_mf = mf_net_buy_df.groupby(pd.cut(mf_net_buy_df['minute_vwap'], bins=30, duplicates='drop'))['main_force_net_vol'].sum()
                if not vp_mf.empty:
                    mf_vpoc = vp_mf.idxmax().mid
                    metrics['main_force_vpoc'] = mf_vpoc
                    if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                        metrics['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
        return metrics

    def _calculate_liquidity_swap_metrics(self, intraday_data: pd.DataFrame) -> dict:
        """
        【V47.0 · 模块化重构版】
        新增方法: 计算主力与散户流动性交换指标。
        """
        metrics = {}
        if 'main_force_net_vol' in intraday_data.columns and 'retail_net_vol' in intraday_data.columns:
            mf_net_series = intraday_data['main_force_net_vol']
            retail_net_series = intraday_data['retail_net_vol']
            if mf_net_series.var() != 0 and retail_net_series.var() != 0 and len(mf_net_series) > 1:
                rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                metrics['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
        return metrics

    def _calculate_retail_sentiment_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, daily_data: pd.Series, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V48.7 · 健壮性修复版】
        - 核心升级: 修复在分钟级降级逻辑中，因笔误使用错误的列名 'net_vol' 替代 'retail_net_vol' 导致的 KeyError。
        - 升级原因: 线上任务在无高频数据的场景下执行降级逻辑时发生崩溃。
        - 核心实现:
          - 统一在降级逻辑中对散户净成交量的引用为 'retail_net_vol'。
        """
        from datetime import time
        import pandas as pd
        import numpy as np
        metrics = {
            'retail_fomo_premium_index': np.nan,
            'retail_panic_surrender_index': np.nan
        }
        day_high, day_low = common_data['day_high'], common_data['day_low']
        atr = common_data['atr']
        if not hf_analysis_df.empty and pd.notna(atr) and atr > 0:
            # --- FOMO Index Calculation ---
            total_weighted_fomo_score = 0
            total_fomo_volume = 0
            max_fomo_event_info = {'score': -1}
            hf_analysis_df['is_new_high'] = hf_analysis_df['price'] > hf_analysis_df['price'].cummax().shift(1).fillna(0)
            fomo_events = hf_analysis_df['is_new_high'].ne(hf_analysis_df['is_new_high'].shift()).cumsum()
            fomo_zones = hf_analysis_df[hf_analysis_df['is_new_high']]
            daily_retail_buy_vol = hf_analysis_df[(hf_analysis_df['amount'] < 50000) & (hf_analysis_df['type'] == 'B')]['volume'].sum()
            avg_retail_buy_rate = daily_retail_buy_vol / (4 * 3600) if daily_retail_buy_vol > 0 else 0
            for _, event_df in fomo_zones.groupby(fomo_events):
                if event_df.empty: continue
                retail_buy_trades = event_df[(event_df['amount'] < 50000) & (event_df['type'] == 'B')]
                if retail_buy_trades.empty: continue
                fomo_vol_in_event = retail_buy_trades['volume'].sum()
                cost_fomo = (retail_buy_trades['price'] * retail_buy_trades['volume']).sum() / fomo_vol_in_event
                cost_mf_sell = daily_data.get('avg_cost_main_sell')
                if pd.notna(cost_mf_sell) and cost_mf_sell > 0:
                    cost_premium_component = (cost_fomo - cost_mf_sell) / atr
                    aggressive_buy_mask = retail_buy_trades['price'] >= retail_buy_trades['sell_price1']
                    aggression_component = retail_buy_trades[aggressive_buy_mask]['volume'].sum() / fomo_vol_in_event
                    duration_seconds = (event_df.index.max() - event_df.index.min()).total_seconds() + 1
                    event_buy_rate = fomo_vol_in_event / duration_seconds
                    volume_spike_component = np.log1p(event_buy_rate / avg_retail_buy_rate) if avg_retail_buy_rate > 0 else 0
                    event_fomo_score = cost_premium_component * aggression_component * volume_spike_component
                    total_weighted_fomo_score += event_fomo_score * fomo_vol_in_event
                    total_fomo_volume += fomo_vol_in_event
                    if event_fomo_score > max_fomo_event_info['score']:
                        max_fomo_event_info = {
                            'score': event_fomo_score, 'premium': cost_premium_component,
                            'aggression': aggression_component, 'spike': volume_spike_component
                        }
            if total_fomo_volume > 0:
                weighted_avg_fomo_score = total_weighted_fomo_score / total_fomo_volume
                metrics['retail_fomo_premium_index'] = weighted_avg_fomo_score * 100
                if enable_probe and is_target_date:
                    print(f"  [探针] retail_fomo_premium_index (高频-散户FOMO) 计算:")
                    print(f"    - Weighted Avg FOMO Score: {weighted_avg_fomo_score:.4f}")
                    if max_fomo_event_info['score'] > -1:
                        print(f"      - [最强FOMO事件分解] Premium: {max_fomo_event_info['premium']:.2f}, Aggression: {max_fomo_event_info['aggression']:.2f}, Spike: {max_fomo_event_info['spike']:.2f}")
                    print(f"    -> Final Score: {metrics['retail_fomo_premium_index']:.2f}")
            # --- Panic Index Calculation (Symmetrical Logic) ---
            total_weighted_panic_score = 0
            total_panic_volume = 0
            max_panic_event_info = {'score': -1}
            hf_analysis_df['is_new_low'] = hf_analysis_df['price'] < hf_analysis_df['price'].cummin().shift(1).fillna(float('inf'))
            panic_events = hf_analysis_df['is_new_low'].ne(hf_analysis_df['is_new_low'].shift()).cumsum()
            panic_zones = hf_analysis_df[hf_analysis_df['is_new_low']]
            daily_retail_sell_vol = hf_analysis_df[(hf_analysis_df['amount'] < 50000) & (hf_analysis_df['type'] == 'S')]['volume'].sum()
            avg_retail_sell_rate = daily_retail_sell_vol / (4 * 3600) if daily_retail_sell_vol > 0 else 0
            for _, event_df in panic_zones.groupby(panic_events):
                if event_df.empty: continue
                retail_sell_trades = event_df[(event_df['amount'] < 50000) & (event_df['type'] == 'S')]
                if retail_sell_trades.empty: continue
                panic_vol_in_event = retail_sell_trades['volume'].sum()
                cost_panic = (retail_sell_trades['price'] * retail_sell_trades['volume']).sum() / panic_vol_in_event
                cost_mf_buy = daily_data.get('avg_cost_main_buy')
                if pd.notna(cost_mf_buy) and cost_mf_buy > 0:
                    cost_discount_component = (cost_mf_buy - cost_panic) / atr
                    aggressive_sell_mask = retail_sell_trades['price'] <= retail_sell_trades['buy_price1']
                    aggression_component = retail_sell_trades[aggressive_sell_mask]['volume'].sum() / panic_vol_in_event
                    duration_seconds = (event_df.index.max() - event_df.index.min()).total_seconds() + 1
                    event_sell_rate = panic_vol_in_event / duration_seconds
                    volume_spike_component = np.log1p(event_sell_rate / avg_retail_sell_rate) if avg_retail_sell_rate > 0 else 0
                    event_panic_score = cost_discount_component * aggression_component * volume_spike_component
                    total_weighted_panic_score += event_panic_score * panic_vol_in_event
                    total_panic_volume += panic_vol_in_event
                    if event_panic_score > max_panic_event_info['score']:
                        max_panic_event_info = {
                            'score': event_panic_score, 'discount': cost_discount_component,
                            'aggression': aggression_component, 'spike': volume_spike_component
                        }
            if total_panic_volume > 0:
                weighted_avg_panic_score = total_weighted_panic_score / total_panic_volume
                metrics['retail_panic_surrender_index'] = weighted_avg_panic_score * 100
                if enable_probe and is_target_date:
                    print(f"  [探针] retail_panic_surrender_index (高频-散户恐慌) 计算:")
                    print(f"    - Weighted Avg Panic Score: {weighted_avg_panic_score:.4f}")
                    if max_panic_event_info['score'] > -1:
                        print(f"      - [最强恐慌事件分解] Discount: {max_panic_event_info['discount']:.2f}, Aggression: {max_panic_event_info['aggression']:.2f}, Spike: {max_panic_event_info['spike']:.2f}")
                    print(f"    -> Final Score: {metrics['retail_panic_surrender_index']:.2f}")
        else:
            # --- Fallback Logic for both metrics ---
            continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
            if pd.notna(day_high) and pd.notna(day_low):
                day_range = day_high - day_low
                if day_range > 0:
                    # FOMO Fallback
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
                                    metrics['retail_fomo_premium_index'] = premium * (fomo_vol / total_retail_buy_vol) * 100
                    # Panic Fallback
                    panic_zone_threshold = day_low + 0.25 * day_range
                    panic_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < panic_zone_threshold]
                    if not panic_zone_df.empty and 'retail_net_vol' in panic_zone_df.columns and 'retail_sell_vol' in continuous_trading_df.columns and 'minute_vwap' in panic_zone_df.columns:
                        panic_retail_df = panic_zone_df[panic_zone_df['retail_net_vol'] < 0]
                        if not panic_retail_df.empty:
                            # 修改代码行：修复笔误，使用 'retail_net_vol'
                            panic_vol = abs(panic_retail_df['retail_net_vol'].sum())
                            total_retail_sell_vol = continuous_trading_df[continuous_trading_df['retail_sell_vol'] > 0]['retail_sell_vol'].sum()
                            if panic_vol > 0 and total_retail_sell_vol > 0:
                                # 修改代码行：修复笔误，使用 'retail_net_vol'
                                cost_panic = (panic_retail_df['minute_vwap'] * abs(panic_retail_df['retail_net_vol'])).sum() / panic_vol
                                cost_mf_buy = daily_data.get('avg_cost_main_buy')
                                if pd.notna(cost_mf_buy) and cost_mf_buy > 0:
                                    discount = (cost_mf_buy - cost_panic) / cost_mf_buy
                                    metrics['retail_panic_surrender_index'] = discount * (panic_vol / total_retail_sell_vol) * 100
        return metrics

    def _calculate_panic_cascade_metrics(self, intraday_data: pd.DataFrame, hf_analysis_df: pd.DataFrame, common_data: dict, is_target_date: bool, enable_probe: bool) -> dict:
        """
        【V48.0 · 恐慌微观解构版】
        - 核心升级: 重构 panic_selling_cascade 指标，从分钟级主力行为升级为高频微观结构解构。
        - 升级原因: 原有分钟级指标无法捕捉恐慌的本质——流动性真空与散户情绪崩溃的共振。
        - 核心实现:
          - 高频路径: 在下跌波段中，综合评估三个核心要素：
            1. 流动性真空 (LiquidityVacuum): 通过买卖盘深度比量化。
            2. 散户投降 (RetailCapitulation): 通过散户主动性卖出量占比量化。
            3. 价格冲击 (PriceImpact): 通过ATR标准化的价格跌幅量化。
          - 降级路径: 若无高频数据，则回退至原有的分钟级主力净卖出占比逻辑。
        """
        from scipy.signal import find_peaks
        from datetime import time
        import numpy as np
        metrics = {}
        atr = common_data['atr']
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'minute_vwap' in continuous_trading_df.columns and pd.notna(atr) and atr > 0:
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            if not hf_analysis_df.empty:
                total_weighted_panic_score = 0
                total_price_drop = 0
                max_panic_leg_info = {'score': -1}
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2: continue
                    if window_df['minute_vwap'].iloc[-1] < window_df['minute_vwap'].iloc[0]:
                        start_time, end_time = window_df.index[0], window_df.index[-1]
                        panic_hf_df = hf_analysis_df[(hf_analysis_df.index >= start_time) & (hf_analysis_df.index <= end_time)]
                        if not panic_hf_df.empty:
                            price_drop_in_leg = window_df['minute_vwap'].iloc[0] - window_df['minute_vwap'].iloc[-1]
                            total_price_drop += price_drop_in_leg
                            price_impact_component = price_drop_in_leg / atr
                            ask_depth = panic_hf_df[[f'sell_volume{i}' for i in range(1, 6)]].sum(axis=1).mean()
                            bid_depth = panic_hf_df[[f'buy_volume{i}' for i in range(1, 6)]].sum(axis=1).mean()
                            liquidity_vacuum_component = np.tanh(np.log1p(ask_depth / bid_depth)) if bid_depth > 0 else 1.0
                            retail_trades_in_leg = panic_hf_df[panic_hf_df['amount'] < 50000]
                            retail_sell_trades = retail_trades_in_leg[retail_trades_in_leg['type'] == 'S']
                            total_retail_sell_vol = retail_sell_trades['volume'].sum()
                            if total_retail_sell_vol > 0:
                                aggressive_sell_mask = retail_sell_trades['price'] <= retail_sell_trades['buy_price1']
                                aggressive_retail_sell_vol = retail_sell_trades[aggressive_sell_mask]['volume'].sum()
                                retail_capitulation_component = aggressive_retail_sell_vol / total_retail_sell_vol
                            else:
                                retail_capitulation_component = 0.0
                            leg_panic_score = price_impact_component * liquidity_vacuum_component * retail_capitulation_component
                            total_weighted_panic_score += leg_panic_score * price_drop_in_leg
                            if leg_panic_score > max_panic_leg_info['score']:
                                max_panic_leg_info = {
                                    'score': leg_panic_score,
                                    'impact': price_impact_component,
                                    'vacuum': liquidity_vacuum_component,
                                    'capitulation': retail_capitulation_component
                                }
                if total_price_drop > 0:
                    weighted_avg_panic_score = total_weighted_panic_score / total_price_drop
                    metrics['panic_selling_cascade'] = weighted_avg_panic_score * 100
                    if enable_probe and is_target_date:
                        print(f"  [探针] panic_selling_cascade (高频-恐慌级联) 计算:")
                        print(f"    - Weighted Avg Panic Score: {weighted_avg_panic_score:.4f}")
                        if max_panic_leg_info['score'] > -1:
                            print(f"      - [最恐慌波段分解] Impact: {max_panic_leg_info['impact']:.2f}, Vacuum: {max_panic_leg_info['vacuum']:.2f}, Capitulation: {max_panic_leg_info['capitulation']:.2f}")
                        print(f"    -> Final Score: {metrics['panic_selling_cascade']:.2f}")
            else:
                panic_vol, total_panic_vol = 0, 0
                for i in range(len(turning_points) - 1):
                    start_idx, end_idx = turning_points[i], turning_points[i+1]
                    window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                    if window_df.empty or len(window_df) < 2 or 'minute_vwap' not in window_df.columns or 'main_force_net_vol' not in window_df.columns or 'vol_shares' not in window_df.columns:
                        continue
                    if window_df['minute_vwap'].iloc[-1] <= window_df['minute_vwap'].iloc[0]:
                        total_panic_vol += window_df['vol_shares'].sum()
                        mf_net_vol = window_df['main_force_net_vol'].sum()
                        if mf_net_vol < 0:
                            panic_vol += abs(mf_net_vol)
                if total_panic_vol > 0:
                    metrics['panic_selling_cascade'] = (panic_vol / total_panic_vol) * 100
        return metrics

    def _calculate_misc_minute_metrics(self, intraday_data: pd.DataFrame, common_data: dict) -> dict:
        """
        【V48.0 · 恐慌微观解构版】
        - 核心重构: 移除旧的 panic_selling_cascade 计算逻辑，其功能已由新的 _calculate_panic_cascade_metrics 方法接管。
        """
        from datetime import time
        from scipy.signal import find_peaks
        import numpy as np
        import pandas as pd
        metrics = {}
        day_open, day_close = common_data['day_open'], common_data['day_close']
        day_high, day_low = common_data['day_high'], common_data['day_low']
        daily_vwap = common_data['daily_vwap']
        atr = common_data['atr']
        daily_total_volume = common_data['daily_total_volume']
        if 'main_force_buy_vol' in intraday_data.columns and 'main_force_sell_vol' in intraday_data.columns and pd.notna(atr) and atr > 0:
            mf_activity_ratio = (intraday_data['main_force_buy_vol'].sum() + intraday_data['main_force_sell_vol'].sum()) / daily_total_volume if daily_total_volume > 0 else 0.0
            if mf_activity_ratio > 0:
                price_outcome = (day_close - day_open) / atr
                metrics['trend_conviction_ratio'] = price_outcome / mf_activity_ratio
        continuous_trading_df = intraday_data[intraday_data.index.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and 'close' in continuous_trading_df.columns and 'open' in continuous_trading_df.columns:
            up_minutes = continuous_trading_df[continuous_trading_df['close'] > continuous_trading_df['open']]
            down_minutes = continuous_trading_df[continuous_trading_df['close'] < continuous_trading_df['open']]
            if not up_minutes.empty and not down_minutes.empty:
                up_price_change = (up_minutes['close'] - up_minutes['open']).sum()
                down_price_change = (down_minutes['open'] - down_minutes['close']).sum()
                avg_up_speed = up_price_change / len(up_minutes) if len(up_minutes) > 0 else 0
                avg_down_speed = down_price_change / len(down_minutes) if len(down_minutes) > 0 else 0
                if avg_up_speed > 0 and avg_down_speed > 0:
                    metrics['volatility_asymmetry_index'] = np.log(avg_up_speed / avg_down_speed)
        if pd.notna(day_high) and pd.notna(day_low) and pd.notna(day_close) and pd.notna(daily_vwap) and pd.notna(atr) and atr > 0 and daily_total_volume > 0:
            day_range = day_high - day_low
            if day_range > 0:
                range_pos_factor = ((day_close - day_low) / day_range) * 2 - 1
                value_dev_factor = np.tanh((day_close - daily_vwap) / atr)
                force_balance_factor = intraday_data['main_force_net_vol'].sum() / daily_total_volume if 'main_force_net_vol' in intraday_data.columns else 0.0
                metrics['closing_price_deviation_score'] = (0.5 * range_pos_factor + 0.3 * value_dev_factor + 0.2 * force_balance_factor) * 100
        # 修改代码块：移除整个 panic_selling_cascade 的计算逻辑
        if 'main_force_net_vol' in intraday_data.columns and 'close' in intraday_data.columns:
            intraday_data['price_change'] = intraday_data['close'].diff()
            if pd.notna(atr) and atr > 0:
                intraday_data['price_change_norm'] = intraday_data['price_change'] / atr
                mf_net_vol_series = intraday_data['main_force_net_vol']
                price_change_norm_series = intraday_data['price_change_norm']
                if mf_net_vol_series.var() > 0 and price_change_norm_series.var() > 0:
                    metrics['microstructure_efficiency_index'] = mf_net_vol_series.corr(price_change_norm_series)
        return metrics

    def _calculate_misc_daily_metrics(self, daily_data: pd.Series, main_force_net_flow_calibrated: float) -> dict:
        """
        【V47.0 · 模块化重构版】
        新增方法: 计算主要基于日线数据的杂项指标。
        """
        import pandas as pd
        import numpy as np
        metrics = {}
        WAN = 10000.0
        try:
            pct_change = pd.to_numeric(daily_data.get('pct_change'), errors='coerce')
            circ_mv_yuan = pd.to_numeric(daily_data.get('circ_mv'), errors='coerce') * WAN
            if pd.notna(circ_mv_yuan) and circ_mv_yuan > 0 and pd.notna(main_force_net_flow_calibrated) and pd.notna(pct_change):
                mf_flow_yuan = main_force_net_flow_calibrated * WAN
                flow_input = mf_flow_yuan / circ_mv_yuan
                if abs(flow_input) > 1e-9:
                    efficiency = (pct_change / 100) / flow_input
                    metrics['flow_efficiency_index'] = np.sign(efficiency) * np.log1p(abs(efficiency))
        except Exception:
            metrics['flow_efficiency_index'] = np.nan
        try:
            trade_count = pd.to_numeric(daily_data.get('trade_count'), errors='coerce')
            turnover_amount_yuan = pd.to_numeric(daily_data.get('amount'), errors='coerce') * 1000
            if pd.notna(trade_count) and trade_count > 0 and pd.notna(turnover_amount_yuan) and turnover_amount_yuan > 0:
                metrics['inferred_active_order_size'] = turnover_amount_yuan / trade_count
        except Exception:
            metrics['inferred_active_order_size'] = np.nan
        return metrics

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

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        records_to_save_df = final_df
        stock_code = stock_info.stock_code
        is_target_date_in_df = False
        target_date = None
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
        # 修改代码行：升级【S.1 - 保存前终审探针】
        if enable_probe and is_target_date_in_df:
            print(f"\n{'='*20} [探针 S.1 - 保存前终审 @ {target_date}] {'='*20}")
            filtered_cols_set = set(df_filtered.columns)
            model_fields_set = set(model_fields)
            print(f"  - 准备保存的DataFrame列数: {len(filtered_cols_set)}")
            print(f"  - 模型定义的字段数 (不含关系): {len(model_fields_set)}")
            missing_in_df = model_fields_set - filtered_cols_set - {'id', 'stock_id', 'trade_time'}
            extra_in_df = filtered_cols_set - model_fields_set
            if not missing_in_df and not extra_in_df:
                print("  - [检查通过] 数据列与模型字段完美匹配。")
            else:
                if missing_in_df:
                    print(f"  - [!!!] 根本原因定位：模型中定义但DataFrame中缺失的字段为: {missing_in_df}")
                if extra_in_df:
                    print(f"  - [!!!] 警告：DataFrame中存在但模型中未定义的字段为: {extra_in_df}")
            target_row = df_filtered.loc[df_filtered.index.date == target_date]
            if not target_row.empty:
                print("  - 目标日期最终待保存数据行预览:")
                preview_cols = ['main_force_conviction_index', 'dip_absorption_power', 'rally_distribution_pressure', 'hidden_accumulation_intensity', 'lower_shadow_absorption_strength']
                print(target_row[preview_cols].to_string())
            else:
                print("  - [!!!] 警告: 在最终待保存的DataFrame中未找到目标日期行！")
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






