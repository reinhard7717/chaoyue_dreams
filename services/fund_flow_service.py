# services/fund_flow_service.py

import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta
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
class AdvancedFundFlowMetricsService:
    """
    【V1.0 · 兵工厂模式】高级资金流指标服务
    - 核心职责: 封装所有高级资金流指标的加载、计算、融合与存储逻辑。
    - 架构优势: 实现业务逻辑与任务调度的完全解耦。
    """
    def __init__(self):
        self.max_lookback_days = 300

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
        【新增】安全地获取一个列作为数值型Series，并保留NaN。
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
        # 移除所有调试性质的print语句
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
            daily_vwap_series = await self._calculate_daily_vwap(stock_info, chunk_raw_data_df.index)
            self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, chunk_raw_data_df.index)
            chunk_new_metrics_df = self._synthesize_and_forge_metrics(stock_code, chunk_raw_data_df, daily_vwap_series)
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
        
        
    async def _load_and_merge_sources(self, stock_info, data_dfs: dict):
        # 移除所有探针性质的print语句
        def standardize_and_prepare(df: pd.DataFrame, source: str) -> pd.DataFrame:
            if df.empty: return df
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            cols_to_numeric = [col for col in df.columns if col != 'trade_time' and 'code' not in col and 'name' not in col]
            df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce')
            if source == 'tushare':
                df['net_flow_tushare'] = df['net_mf_amount']
                df['main_force_net_flow_tushare'] = df['buy_lg_amount'] + df['buy_elg_amount'] - df['sell_lg_amount'] - df['sell_elg_amount']
                df['retail_net_flow_tushare'] = df['buy_sm_amount'] + df['buy_md_amount'] - df['sell_sm_amount'] - df['sell_md_amount']
                df['net_xl_amount_tushare'] = df['buy_elg_amount'] - df['sell_elg_amount']
                df['net_lg_amount_tushare'] = df['buy_lg_amount'] - df['sell_lg_amount']
                df['net_md_amount_tushare'] = df['buy_md_amount'] - df['sell_md_amount']
                df['net_sh_amount_tushare'] = df['buy_sm_amount'] - df['sell_sm_amount']
                return df
            elif source == 'ths':
                df = df.rename(columns={'net_amount': 'net_flow_ths', 'buy_lg_amount': 'net_lg_amount_ths', 'buy_md_amount': 'net_md_amount_ths', 'buy_sm_amount': 'net_sh_amount_ths'})
                df['main_force_net_flow_ths'] = df.get('net_lg_amount_ths', 0)
                df['retail_net_flow_ths'] = df.get('net_md_amount_ths', 0).fillna(0) + df.get('net_sh_amount_ths', 0).fillna(0)
                return df
            elif source == 'dc':
                df = df.rename(columns={'net_amount': 'main_force_net_flow_dc', 'buy_elg_amount': 'net_xl_amount_dc', 'buy_lg_amount': 'net_lg_amount_dc', 'buy_md_amount': 'net_md_amount_dc', 'buy_sm_amount': 'net_sh_amount_dc'})
                df['net_flow_dc'] = df.get('main_force_net_flow_dc', 0).fillna(0) + df.get('net_md_amount_dc', 0).fillna(0) + df.get('net_sh_amount_dc', 0).fillna(0)
                df['retail_net_flow_dc'] = df.get('net_md_amount_dc', 0).fillna(0) + df.get('net_sh_amount_dc', 0).fillna(0)
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
                merged_df = pd.merge(merged_df, right_df, on='trade_time', how='left')
        merged_df = merged_df.sort_values('trade_time').set_index('trade_time')
        merged_df = merged_df.drop(columns=['close'], errors='ignore')
        daily_dfs_to_join = []
        if not data_dfs['daily'].empty:
            daily_df = data_dfs['daily'].set_index(pd.to_datetime(data_dfs['daily']['trade_time'])).drop(columns='trade_time')
            daily_dfs_to_join.append(daily_df)
        if not data_dfs['daily_basic'].empty:
            daily_basic_df = data_dfs['daily_basic'].set_index(pd.to_datetime(data_dfs['daily_basic']['trade_time'])).drop(columns='trade_time')
            daily_dfs_to_join.append(daily_basic_df)
        if daily_dfs_to_join:
            merged_df = merged_df.join(daily_dfs_to_join, how='left')
        return merged_df
        

    async def _calculate_daily_vwap(self, stock_info: StockInfo, date_index: pd.DatetimeIndex) -> pd.Series:
        """【V1.3 · 时区修正版】从分钟数据计算日度VWAP"""
        # 修正时区查询BUG
        minute_df = await self._get_daily_grouped_minute_data(stock_info, date_index, fetch_full_cols=False)
        if minute_df is None or minute_df.empty:
            return pd.Series(np.nan, index=date_index)
        # 使用新的、更可靠的辅助函数进行计算
        return self._calculate_daily_vwap_from_df(minute_df, date_index)
        
    async def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex, fetch_full_cols: bool = True):
        from django.utils import timezone
        from datetime import datetime, time
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        # 移除所有调试性质的print语句
        if not MinuteModel:
            return None
        if date_index.empty:
            return None
        @sync_to_async(thread_sensitive=True)
        def get_data(model, stock_pk, start_dt, end_dt):
            cols_to_fetch = ('trade_time', 'amount', 'vol', 'open', 'close', 'high', 'low') if fetch_full_cols else ('trade_time', 'amount', 'vol')
            qs = model.objects.filter(
                stock_id=stock_pk,
                trade_time__gte=start_dt,
                trade_time__lt=end_dt
            ).values(*cols_to_fetch)
            return pd.DataFrame.from_records(qs)
        min_date, max_date = date_index.min().date(), date_index.max().date()
        start_datetime = timezone.make_aware(datetime.combine(min_date, time.min))
        end_datetime = timezone.make_aware(datetime.combine(max_date, time.max))
        minute_df = await get_data(MinuteModel, stock_info.pk, start_datetime, end_datetime)
        if minute_df.empty:
            return None
        return self._group_minute_data_from_df(minute_df)
        
        
    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame, daily_vwap_series: pd.Series) -> tuple[pd.DataFrame, dict]:
        df = merged_df.copy()
        df['daily_vwap'] = daily_vwap_series
        result_df = df.copy()
        attributed_minute_map = {}
        # 移除所有探针性质的print语句
        consensus_map = {
            'net_flow_consensus': ['net_flow_tushare', 'net_flow_ths', 'net_flow_dc'],
            'main_force_net_flow_consensus': ['main_force_net_flow_tushare', 'main_force_net_flow_dc'],
            'retail_net_flow_consensus': ['retail_net_flow_tushare', 'retail_net_flow_ths', 'retail_net_flow_dc'],
            'net_xl_amount_consensus': ['net_xl_amount_tushare', 'net_xl_amount_dc'],
            'net_lg_amount_consensus': ['net_lg_amount_tushare', 'net_lg_amount_ths', 'net_lg_amount_dc'],
            'net_md_amount_consensus': ['net_md_amount_tushare', 'net_md_amount_ths', 'net_md_amount_dc'],
            'net_sh_amount_consensus': ['net_sh_amount_tushare', 'net_sh_amount_ths', 'net_sh_amount_dc'],
        }
        for target_col, source_cols in consensus_map.items():
            existing_cols = [col for col in source_cols if col in df.columns]
            if existing_cols:
                result_df[target_col] = df[existing_cols].mean(axis=1)
        source_cols = ['main_force_net_flow_tushare', 'main_force_net_flow_ths', 'main_force_net_flow_dc']
        existing_sources = [col for col in source_cols if col in df.columns]
        minute_df_daily_grouped = getattr(self, '_minute_df_daily_grouped', None)
        if minute_df_daily_grouped is not None and not minute_df_daily_grouped.empty and 'buy_sm_vol' in df.columns:
            pvwap_costs_df, attributed_minute_map = self._calculate_probabilistic_costs(df, minute_df_daily_grouped)
            result_df = result_df.join(pvwap_costs_df)
            pnl_matrix_df = self._upgrade_intraday_profit_metric(result_df)
            result_df = result_df.join(pnl_matrix_df)
            if 'net_position_change_value' in result_df.columns:
                result_df['cost_weighted_main_flow'] = result_df['net_position_change_value']
            if 'main_force_net_flow_consensus' in result_df.columns and 'pnl_matrix_confidence_score' in result_df.columns:
                result_df['consensus_calibrated_main_flow'] = result_df['main_force_net_flow_consensus'] * result_df['pnl_matrix_confidence_score']
            mf_flow = self._get_numeric_series_with_nan(result_df, 'main_force_net_flow_consensus')
            retail_flow = self._get_numeric_series_with_nan(result_df, 'retail_net_flow_consensus')
            numerator = (mf_flow - retail_flow).abs()
            denominator = mf_flow.abs() + retail_flow.abs()
            result_df['flow_internal_friction_ratio'] = numerator / denominator.replace(0, np.nan)
            if len(existing_sources) > 1:
                flows = df[existing_sources]
                result_df['cross_source_divergence_std'] = flows.std(axis=1)
            if 'cross_source_divergence_std' in result_df.columns and 'main_force_net_flow_consensus' in result_df.columns:
                mean_abs_flow = result_df['main_force_net_flow_consensus'].abs().mean()
                denominator_consistency = np.nan if mean_abs_flow == 0 else mean_abs_flow
                consistency_ratio = result_df['cross_source_divergence_std'] / denominator_consistency
                result_df['source_consistency_score'] = (1 - consistency_ratio).clip(lower=0)
            if 'avg_cost_main_buy' in result_df.columns:
                result_df['cost_divergence_mf_vs_retail'] = result_df['avg_cost_main_buy'] - result_df.get('avg_cost_retail_sell', np.nan)
                result_df['main_buy_cost_advantage'] = np.divide(result_df['avg_cost_main_buy'], df['close'], out=np.full_like(df['close'].values, np.nan, dtype=float), where=df['close']!=0) - 1
                result_df['market_cost_battle'] = result_df['avg_cost_main_buy'] - result_df.get('avg_cost_retail_buy', np.nan)
                if 'daily_vwap' in result_df.columns:
                    result_df['main_buy_cost_vs_vwap'] = result_df['avg_cost_main_buy'] - result_df['daily_vwap']
                    result_df['main_sell_cost_vs_vwap'] = result_df.get('avg_cost_main_sell', np.nan) - result_df['daily_vwap']
            behavioral_metrics_df = self._upgrade_behavioral_metrics(result_df, attributed_minute_map)
            result_df = result_df.join(behavioral_metrics_df)
            structure_metrics_df = self._calculate_intraday_structure_metrics(result_df, minute_df_daily_grouped)
            result_df = result_df.join(structure_metrics_df)
        if len(existing_sources) > 1:
            flows = df[existing_sources]
            median_flow = flows.median(axis=1)
            deviations = flows.sub(median_flow, axis=0).abs()
            weights = 1 / (1 + deviations)
            weighted_flows = flows.multiply(weights.values)
            result_df['consensus_flow_weighted'] = weighted_flows.sum(axis=1) / weights.sum(axis=1).replace(0, np.nan)
        else:
            result_df['consensus_flow_weighted'] = result_df.get('main_force_net_flow_consensus', np.nan)
        if 'main_force_net_flow_tushare' in df.columns and 'main_force_net_flow_ths' in df.columns:
            result_df['divergence_ts_ths'] = df['main_force_net_flow_tushare'] - df['main_force_net_flow_ths']
        if 'main_force_net_flow_tushare' in df.columns and 'main_force_net_flow_dc' in df.columns:
            result_df['divergence_ts_dc'] = df['main_force_net_flow_tushare'] - df['main_force_net_flow_dc']
        if 'main_force_net_flow_ths' in df.columns and 'main_force_net_flow_dc' in df.columns:
            result_df['divergence_ths_dc'] = df['main_force_net_flow_ths'] - df['main_force_net_flow_dc']
        safe_denom = lambda v: v.replace(0, np.nan)
        if 'trade_count' in df.columns and 'amount' in df.columns and 'close' in df.columns:
            total_turnover_yuan = pd.to_numeric(df['amount'], errors='coerce') * 1000
            trade_count_np = pd.to_numeric(df['trade_count'], errors='coerce')
            result_df['avg_order_value'] = np.divide(total_turnover_yuan, trade_count_np, out=np.full_like(total_turnover_yuan, np.nan, dtype=float), where=trade_count_np!=0)
            close_price_np = pd.to_numeric(df['close'], errors='coerce')
            avg_order_value_np = result_df['avg_order_value'].values
            result_df['avg_order_value_norm_price'] = np.divide(avg_order_value_np, close_price_np, out=np.full_like(avg_order_value_np, np.nan, dtype=float), where=close_price_np!=0)
        if 'amount' in df.columns and 'circ_mv' in df.columns:
            total_turnover_yuan = pd.to_numeric(df['amount'], errors='coerce') * 1000
            circ_mv_yuan = pd.to_numeric(df['circ_mv'], errors='coerce') * 10000
            if 'main_force_net_flow_consensus' in result_df.columns:
                main_force_net_flow_yuan = pd.to_numeric(result_df['main_force_net_flow_consensus'], errors='coerce') * 10000
                result_df['main_force_flow_impact_ratio'] = main_force_net_flow_yuan / safe_denom(circ_mv_yuan)
                result_df['main_force_flow_intensity_ratio'] = main_force_net_flow_yuan / safe_denom(total_turnover_yuan)
                result_df['main_force_buy_rate_consensus'] = (main_force_net_flow_yuan / safe_denom(circ_mv_yuan)) * 100
            if 'avg_order_value' in result_df.columns:
                result_df['trade_granularity_impact'] = result_df['avg_order_value'] / safe_denom(circ_mv_yuan)
            if 'net_xl_amount_consensus' in result_df.columns:
                total_xl_trade_yuan = pd.to_numeric(result_df['net_xl_amount_consensus'], errors='coerce').abs() * 10000
                result_df['trade_concentration_index'] = total_xl_trade_yuan / safe_denom(total_turnover_yuan)
        if 'main_force_net_flow_consensus' in result_df.columns and 'retail_net_flow_consensus' in result_df.columns:
            result_df['flow_divergence_mf_vs_retail'] = result_df['main_force_net_flow_consensus'] - result_df['retail_net_flow_consensus']
        if 'main_force_net_flow_consensus' in result_df.columns and 'net_xl_amount_consensus' in result_df.columns:
            result_df['main_force_vs_xl_divergence'] = result_df['main_force_net_flow_consensus'] - result_df['net_xl_amount_consensus']
        if 'net_xl_amount_consensus' in result_df.columns and 'net_lg_amount_consensus' in result_df.columns:
            result_df['main_force_conviction_ratio'] = result_df['net_xl_amount_consensus'] / safe_denom(result_df['net_lg_amount_consensus'])
        return result_df, attributed_minute_map
        

    def _calculate_probabilistic_costs(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        【V6.0 · 显式返回版】不再依赖副作用，显式返回归因后的分钟数据字典。
        """
        if minute_df_grouped is None:
            # 确保在任何分支都返回两个值
            return pd.DataFrame(index=daily_df.index), {}
            
        results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        from scipy.spatial.distance import jensenshannon
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = self._calculate_intraday_attribution_weights(minute_df_grouped.loc[[date_key]].copy(), daily_data)
            day_results = {'trade_time': date}
            for cost_type in cost_types:
                size, direction = cost_type.split('_')
                db_vol_key = f'{direction}_{size}_vol'
                daily_vol_shares = pd.to_numeric(daily_data.get(db_vol_key), errors='coerce') * 100
                if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                    day_results[f'avg_cost_{cost_type}'] = np.nan
                    minute_data_for_day[f'{cost_type}_vol_attr'] = 0
                    continue
                weight_col = f'{size}_{direction}_weight'
                weight_series = minute_data_for_day[weight_col]
                if weight_series.sum() < 1e-9:
                    day_results[f'avg_cost_{cost_type}'] = np.nan
                    minute_data_for_day[f'{cost_type}_vol_attr'] = 0
                    continue
                attributed_vol = weight_series * daily_vol_shares
                minute_data_for_day[f'{cost_type}_vol_attr'] = attributed_vol
                attributed_value = attributed_vol * minute_data_for_day['minute_vwap']
                total_attributed_value = attributed_value.sum()
                total_attributed_vol = attributed_vol.sum()
                day_results[f'avg_cost_{cost_type}'] = total_attributed_value / total_attributed_vol if total_attributed_vol > 0 else np.nan
            p_dist = minute_data_for_day['vol_shares'].fillna(0).values / minute_data_for_day['vol_shares'].sum() if minute_data_for_day['vol_shares'].sum() > 0 else np.zeros(len(minute_data_for_day))
            q_dist = np.full_like(p_dist, 1.0 / len(p_dist)) if len(p_dist) > 0 else np.array([])
            day_results['volume_profile_jsd_vs_uniform'] = jensenshannon(p_dist, q_dist)**2 if p_dist.size > 0 and q_dist.size > 0 else np.nan
            trade_time_local = minute_data_for_day['trade_time'].dt.tz_convert('Asia/Shanghai')
            first_hour_mask = (trade_time_local.dt.hour == 9) & (trade_time_local.dt.minute >= 30) | \
                              (trade_time_local.dt.hour == 10) & (trade_time_local.dt.minute <= 30)
            first_hour_vol = minute_data_for_day[first_hour_mask]['vol_shares'].sum()
            total_day_vol = minute_data_for_day['vol_shares'].sum()
            day_results['aggression_index_opening'] = first_hour_vol / total_day_vol if total_day_vol else np.nan
            day_results['minute_data_attributed'] = minute_data_for_day
            results[date] = day_results
        if not results:
            # 确保在任何分支都返回两个值
            return pd.DataFrame(), {}
            
        # 不再设置实例属性，而是创建局部变量并返回
        attributed_minute_map = {date: res.pop('minute_data_attributed') for date, res in results.items() if 'minute_data_attributed' in res}
        pvwap_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        aggregate_costs_df = self._calculate_aggregate_pvwap_costs(pvwap_df, daily_df)
        final_df = pvwap_df.join(aggregate_costs_df)
        return final_df, attributed_minute_map

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.7 · 聚合单位修正版】在加权平均聚合成本时，将成交量单位从“手”统一转换为“股”。
        """
        temp_df = pvwap_df.copy()
        vol_cols = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'
        ]
        existing_vol_cols = [col for col in vol_cols if col in daily_df.columns]
        agg_cols = ['avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell', 'vwap_tracking_error']
        if not existing_vol_cols:
            return pd.DataFrame(columns=agg_cols, index=pvwap_df.index)
        temp_df = temp_df.join(daily_df[existing_vol_cols])
        def weighted_avg_cost(cost_cols, vol_cols):
            numerator = pd.Series(0.0, index=temp_df.index)
            denominator = pd.Series(0.0, index=temp_df.index)
            for cost_col, vol_col in zip(cost_cols, vol_cols):
                if cost_col in temp_df.columns and vol_col in temp_df.columns:
                    cost = temp_df[cost_col]
                    # 将成交量从“手”转换为“股” (乘以100)
                    volume_shares = pd.to_numeric(temp_df[vol_col], errors='coerce').fillna(0) * 100
                    value_contribution = (cost * volume_shares).fillna(0)
                    
                    numerator += value_contribution
                    denominator += volume_shares.where(cost.notna(), 0)
            return numerator / denominator.replace(0, np.nan)
        result_agg_df = pd.DataFrame(index=pvwap_df.index)
        result_agg_df['avg_cost_main_buy'] = weighted_avg_cost(
            ['avg_cost_lg_buy', 'avg_cost_elg_buy'],
            ['buy_lg_vol', 'buy_elg_vol']
        )
        result_agg_df['avg_cost_main_sell'] = weighted_avg_cost(
            ['avg_cost_lg_sell', 'avg_cost_elg_sell'],
            ['sell_lg_vol', 'sell_elg_vol']
        )
        result_agg_df['avg_cost_retail_buy'] = weighted_avg_cost(
            ['avg_cost_sm_buy', 'avg_cost_md_buy'],
            ['buy_sm_vol', 'buy_md_vol']
        )
        result_agg_df['avg_cost_retail_sell'] = weighted_avg_cost(
            ['avg_cost_sm_sell', 'avg_cost_md_sell'],
            ['sell_sm_vol', 'sell_md_vol']
        )
        if 'avg_cost_main_buy' in result_agg_df.columns and 'daily_vwap' in daily_df.columns:
            result_agg_df['vwap_tracking_error'] = result_agg_df['avg_cost_main_buy'] - daily_df['daily_vwap']
        return result_agg_df

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """【V3.7 · 导数协议修正版】为二阶导数（加速度）使用独立的、正确的短计算窗口。"""
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        # [代码新增开始] 为加速度定义一个独立的、符合数学定义的短窗口
        ACCEL_WINDOW = 2
        # [代码新增结束]
        sum_cols = [
            'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
            'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
            'net_sh_amount_consensus', 'cost_weighted_main_flow',
            'consensus_calibrated_main_flow',
            'consensus_flow_weighted',
            'divergence_ts_ths', 'divergence_ts_dc', 'divergence_ths_dc',
            'realized_profit_on_exchange', 'net_position_change_value', 'unrealized_pnl_on_net_change',
        ]
        UNIFIED_PERIODS = [1, 5, 13, 21, 55]
        for p in UNIFIED_PERIODS:
            if p <= 1: continue
            min_p = max(2, int(p * 0.8))
            for col in sum_cols:
                if col in consensus_df.columns:
                    source_series_for_sum = pd.to_numeric(consensus_df[col], errors='coerce')
                    sum_col_name = f'{col}_sum_{p}d'
                    derivatives_df[sum_col_name] = source_series_for_sum.rolling(window=p, min_periods=min_p).sum()
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
            if source_series.isnull().all():
                continue
            for p in UNIFIED_PERIODS:
                calc_window = max(2, p) if p > 1 else 2
                slope_col_name = f'{col}_slope_{p}d'
                slope_series = ta.slope(close=source_series.astype(float), length=calc_window)
                derivatives_df[slope_col_name] = slope_series
                if slope_series is not None and not slope_series.empty:
                    accel_col_name = f'{col}_accel_{p}d'
                    # 强制为加速度计算使用独立的短窗口 ACCEL_WINDOW
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
        

    def _upgrade_intraday_profit_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.2 · 会计准则修正版】修正 main_force_intraday_profit 的计算逻辑。
        - 核心修正: 遵循金融第一性原理，将总盈亏修正为“已实现盈亏 + 未实现盈亏”。
        """
        results_df = pd.DataFrame(index=df.index)
        cost_buy = self._get_numeric_series_with_nan(df, 'avg_cost_main_buy')
        cost_sell = self._get_numeric_series_with_nan(df, 'avg_cost_main_sell')
        close_price = self._get_numeric_series_with_nan(df, 'close')
        vol_buy_shares = (self._get_safe_numeric_series(df, 'buy_lg_vol') + self._get_safe_numeric_series(df, 'buy_elg_vol')) * 100
        vol_sell_shares = (self._get_safe_numeric_series(df, 'sell_lg_vol') + self._get_safe_numeric_series(df, 'sell_elg_vol')) * 100
        exchanged_volume = np.minimum(vol_buy_shares, vol_sell_shares)
        realized_profit_yuan = (cost_sell - cost_buy) * exchanged_volume
        results_df['realized_profit_on_exchange'] = realized_profit_yuan / 10000 # 转换为万元
        net_pos_change_vol = vol_buy_shares - vol_sell_shares
        net_pos_change_cost = np.where(net_pos_change_vol > 0, cost_buy, cost_sell)
        net_pos_change_value_yuan = net_pos_change_vol * net_pos_change_cost
        results_df['net_position_change_value'] = net_pos_change_value_yuan / 10000 # 转换为万元
        unrealized_pnl_yuan = (close_price - net_pos_change_cost) * net_pos_change_vol
        results_df['unrealized_pnl_on_net_change'] = unrealized_pnl_yuan / 10000 # 转换为万元
        # 修正主力日内盈亏的计算逻辑
        # 旧的错误逻辑: total_profit_yuan = (cost_sell * vol_sell_shares) - (total_buy_value_yuan = cost_buy * vol_buy_shares)
        # 这是净现金流，不是利润。
        # 新的正确逻辑: 总利润 = 已实现利润 + 未实现（浮动）利润
        total_profit_yuan = realized_profit_yuan + unrealized_pnl_yuan
        results_df['main_force_intraday_profit'] = total_profit_yuan / 10000 # 将最终利润转换为“万元”
        
        dir_ts = np.sign(results_df['net_position_change_value'].fillna(0))
        dir_ths = np.sign(self._get_safe_numeric_series(df, 'main_force_net_flow_ths'))
        dir_dc = np.sign(self._get_safe_numeric_series(df, 'main_force_net_flow_dc'))
        agreement_count = pd.Series(0, index=df.index)
        valid_sources = 0
        if 'main_force_net_flow_ths' in df.columns:
            agreement_count += (dir_ts == dir_ths).astype(int)
            valid_sources += 1
        if 'main_force_net_flow_dc' in df.columns:
            agreement_count += (dir_ts == dir_dc).astype(int)
            valid_sources += 1
        if 'main_force_net_flow_ths' in df.columns and 'main_force_net_flow_dc' in df.columns:
             agreement_count += (dir_ths == dir_dc).astype(int)
             valid_sources += 1
        if valid_sources > 0:
            results_df['pnl_matrix_confidence_score'] = agreement_count / max(1, valid_sources)
        else:
            results_df['pnl_matrix_confidence_score'] = np.nan
        return results_df

    def _upgrade_behavioral_metrics(self, daily_df: pd.DataFrame, minute_df_attributed_grouped: dict) -> pd.DataFrame:
        # 移除调试性质的print语句
        if not minute_df_attributed_grouped:
            return pd.DataFrame(index=daily_df.index)
        
        results = {}
        for date, daily_data in daily_df.iterrows():
            if date not in minute_df_attributed_grouped:
                continue
            minute_data_for_day = minute_df_attributed_grouped[date]
            day_results = {'trade_time': date}
            minute_data_for_day['main_force_buy_vol'] = minute_data_for_day.get('lg_buy_vol_attr', 0) + minute_data_for_day.get('elg_buy_vol_attr', 0)
            minute_data_for_day['main_force_sell_vol'] = minute_data_for_day.get('lg_sell_vol_attr', 0) + minute_data_for_day.get('elg_sell_vol_attr', 0)
            minute_data_for_day['main_force_net_vol'] = minute_data_for_day['main_force_buy_vol'] - minute_data_for_day['main_force_sell_vol']
            minute_data_for_day['retail_buy_vol'] = minute_data_for_day.get('sm_buy_vol_attr', 0) + minute_data_for_day.get('md_buy_vol_attr', 0)
            minute_data_for_day['retail_sell_vol'] = minute_data_for_day.get('sm_sell_vol_attr', 0) + minute_data_for_day.get('md_sell_vol_attr', 0)
            minute_data_for_day['retail_net_vol'] = minute_data_for_day['retail_buy_vol'] - minute_data_for_day['retail_sell_vol']
            low_threshold = minute_data_for_day['minute_vwap'].quantile(0.1)
            bottom_zone_minutes = minute_data_for_day[minute_data_for_day['minute_vwap'] <= low_threshold]
            if not bottom_zone_minutes.empty:
                support_net_flow = bottom_zone_minutes['main_force_net_vol'].sum()
                total_main_buy = minute_data_for_day['main_force_buy_vol'].sum()
                day_results['main_force_support_strength'] = support_net_flow / total_main_buy if total_main_buy else np.nan
            else:
                day_results['main_force_support_strength'] = 0
            high_threshold = minute_data_for_day['minute_vwap'].quantile(0.9)
            top_zone_minutes = minute_data_for_day[minute_data_for_day['minute_vwap'] >= high_threshold]
            if not top_zone_minutes.empty:
                distribution_net_flow = top_zone_minutes['main_force_net_vol'].sum()
                total_main_sell = minute_data_for_day['main_force_sell_vol'].sum()
                day_results['main_force_distribution_pressure'] = -distribution_net_flow / total_main_sell if total_main_sell else np.nan
            else:
                day_results['main_force_distribution_pressure'] = 0
            minute_data_for_day['price_return_5min'] = minute_data_for_day['minute_vwap'].pct_change(5)
            panic_minutes = minute_data_for_day[minute_data_for_day['price_return_5min'] < -0.015]
            if not panic_minutes.empty:
                panic_sell_vol = panic_minutes['retail_sell_vol'].sum()
                total_retail_sell = minute_data_for_day['retail_sell_vol'].sum()
                day_results['retail_capitulation_score'] = panic_sell_vol / total_retail_sell if total_retail_sell else np.nan
            else:
                day_results['retail_capitulation_score'] = 0
            main_force_net_flow_series = minute_data_for_day['main_force_net_vol'].fillna(0)
            price_change_series = minute_data_for_day['minute_vwap'].diff().fillna(0)
            if not main_force_net_flow_series.empty and not price_change_series.empty and main_force_net_flow_series.std() > 0 and price_change_series.std() > 0:
                correlation = main_force_net_flow_series.corr(price_change_series)
                day_results['intraday_execution_alpha'] = -correlation if pd.notna(correlation) else 0
            else:
                day_results['intraday_execution_alpha'] = 0
            results[date] = day_results
        if not results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')

    def _calculate_intraday_structure_metrics(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame) -> pd.DataFrame:
        # 移除调试性质的print语句
        if minute_df_grouped is None:
            return pd.DataFrame(index=daily_df.index)
        
        results = {}
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = minute_df_grouped.loc[[date_key]]
            day_results = {'trade_time': date}
            minute_returns = minute_data_for_day['minute_vwap'].pct_change().dropna()
            day_results['intraday_volatility'] = minute_returns.std() if not minute_returns.empty else 0
            intraday_high = minute_data_for_day['minute_vwap'].max()
            intraday_low = minute_data_for_day['minute_vwap'].min()
            close_price = daily_data.get('close')
            price_range = intraday_high - intraday_low
            if pd.notna(close_price) and price_range > 0:
                day_results['closing_strength_index'] = (close_price - intraday_low) / price_range
            else:
                day_results['closing_strength_index'] = np.nan
            daily_vwap = daily_data.get('daily_vwap')
            if pd.notna(close_price) and pd.notna(daily_vwap) and daily_vwap > 0:
                day_results['close_vs_vwap_ratio'] = (close_price / daily_vwap) - 1
            else:
                day_results['close_vs_vwap_ratio'] = np.nan
            trade_time_local = minute_data_for_day['trade_time'].dt.tz_convert('Asia/Shanghai')
            final_hour_mask = trade_time_local.dt.hour >= 14
            final_hour_vol = minute_data_for_day[final_hour_mask]['vol_shares'].sum()
            total_day_vol = minute_data_for_day['vol_shares'].sum()
            if total_day_vol > 0:
                day_results['final_hour_momentum'] = final_hour_vol / total_day_vol
            else:
                day_results['final_hour_momentum'] = np.nan
            results[date] = day_results
        if not results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')

    def _calculate_intraday_attribution_weights(self, minute_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """
        【V7.1 · 级联选择重构版】使用np.select重构分钟成交额分类逻辑，根除“空桶”问题。
        """
        df = minute_data_for_day.copy()
        if 'amount_yuan' not in df.columns or df['amount_yuan'].sum() < 1e-6:
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_weight'] = df[f'{size}_sell_weight'] = 0
            return df
        valid_amounts = df['amount_yuan'][df['amount_yuan'] > 0]
        if valid_amounts.empty:
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_weight'] = df[f'{size}_sell_weight'] = 0
            return df
        thresholds = {
            'elg': valid_amounts.quantile(0.95),
            'lg': valid_amounts.quantile(0.85),
            'md': valid_amounts.quantile(0.60)
        }
        score_source_col = 'amount_yuan'
        # 使用级联选择（np.select）重构分类逻辑，使其对重合的分位数具有鲁棒性
        conditions = [
            df[score_source_col] >= thresholds['elg'],
            df[score_source_col] >= thresholds['lg'],
            df[score_source_col] >= thresholds['md'],
        ]
        choices = ['elg', 'lg', 'md']
        # np.select按顺序匹配，第一个为True的条件决定分类，完美解决分位数重合问题
        df['category'] = np.select(conditions, choices, default='sm')
        scores = {}
        for size in ['sm', 'md', 'lg', 'elg']:
            scores[size] = df[score_source_col].where(df['category'] == size, 0)
        
        price_range = df['high'] - df['low']
        prev_close = df['close'].shift(1)
        conditions_pressure = [
            price_range > 0,
            (price_range == 0) & (df['close'] > prev_close),
            (price_range == 0) & (df['close'] < prev_close),
        ]
        choices_pressure = [
            (df['close'] - df['low']) / price_range,
            1.0,
            0.0,
        ]
        buy_pressure_proxy = np.select(conditions_pressure, choices_pressure, default=0.5)
        sell_pressure_proxy = 1.0 - buy_pressure_proxy
        for size, score_series in scores.items():
            unnormalized_buy_score = score_series * buy_pressure_proxy
            unnormalized_sell_score = score_series * sell_pressure_proxy
            total_buy_score_day = unnormalized_buy_score.sum()
            total_sell_score_day = unnormalized_sell_score.sum()
            if total_buy_score_day > 1e-9:
                df[f'{size}_buy_weight'] = unnormalized_buy_score / total_buy_score_day
            else:
                df[f'{size}_buy_weight'] = 0
            if total_sell_score_day > 1e-9:
                df[f'{size}_sell_weight'] = unnormalized_sell_score / total_sell_score_day
            else:
                df[f'{size}_sell_weight'] = 0
        return df

    def _calculate_daily_vwap_from_df(self, minute_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.Series:
        """【V1.3 · API单位对齐版】从预加载的DataFrame计算日度VWAP"""
        if minute_df.empty:
            return pd.Series(np.nan, index=date_index)
        df = minute_df.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df[['amount', 'vol']] = df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
        # 根据API文档，分钟线的amount单位是元，vol单位是股，无需转换。
        df['total_value'] = df['amount']
        df['total_volume'] = df['vol']
        
        daily_agg = df.groupby(df['trade_time'].dt.date)
        daily_total_value = daily_agg['total_value'].sum()
        daily_total_volume = daily_agg['total_volume'].sum()
        daily_vwap = daily_total_value / daily_total_volume.replace(0, np.nan)
        daily_vwap.index = pd.to_datetime(daily_vwap.index)
        return daily_vwap.reindex(date_index)

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【V1.3 · API单位对齐版】从预加载的DataFrame构建按日分组的数据。"""
        if minute_df is None or minute_df.empty:
            return None
        df = minute_df.copy()
        df.sort_values('trade_time', inplace=True)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df['date'] = df['trade_time'].dt.date
        df[['amount', 'vol']] = df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
        # 根据API文档，分钟线的amount单位是元，vol单位是股，无需转换。
        df['amount_yuan'] = df['amount']
        df['vol_shares'] = df['vol']
        
        df['minute_vwap'] = df['amount_yuan'] / df['vol_shares'].replace(0, np.nan)
        daily_total_vol = df.groupby('date')['vol_shares'].transform('sum')
        df['vol_weight'] = df['vol_shares'] / daily_total_vol.replace(0, np.nan)
        return df.set_index('date')

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """
        【V2.1 · 数据类型净化版】从数据库加载并净化历史高级资金流指标。
        - 核心修正: 在加载后立即将所有数值列转换为float，防止后续拼接时产生object类型污染。
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
            df = df.set_index(pd.to_datetime(df['trade_time']))
            # 在数据源头进行类型净化，杜绝object类型污染
            # 遍历所有非索引列，将它们强制转换为float类型，无法转换的将变为NaN
            for col in df.columns:
                if col != 'trade_time': # trade_time已经是索引了
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df







