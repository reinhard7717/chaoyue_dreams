# services/fund_flow_service.py

import asyncio
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
        
    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame) -> tuple[pd.DataFrame, dict, list]:
        """
        【V20.3 · VWAP固化修复版】
        - 核心修复: 彻底修复VWAP传递链。在probabilistic_costs计算后，立即将返回的VWAP固化到主数据流result_df_day中，确保其能被下游的advanced_behavioral_metrics正确获取。
        """
        all_metrics_list = []
        attributed_minute_map = {}
        all_failures = []
        for trade_date, daily_data_series in merged_df.iterrows():
            daily_data = daily_data_series.to_frame().T
            daily_data.index.name = 'trade_time'
            minute_df_daily_grouped = getattr(self, '_minute_df_daily_grouped', None)
            has_minute_data = minute_df_daily_grouped is not None and not minute_df_daily_grouped.empty and trade_date.date() in minute_df_daily_grouped.index
            has_daily_flow_data = 'buy_sm_vol' in daily_data.columns and pd.notna(daily_data['buy_sm_vol'].iloc[0])
            if not has_daily_flow_data:
                reason = "日线资金流数据缺失"
                print(f"DEBUG PROBE: [{stock_code}] [{trade_date.date()}] 跳过，原因: {reason}")
                all_failures.append({'stock_code': stock_code, 'trade_date': str(trade_date.date()), 'reason': f"[资金流服务] {reason}"})
                continue
            result_df_day = daily_data.copy()
            daily_derived_metrics = self._calculate_daily_derived_metrics(result_df_day.iloc[0])
            if daily_derived_metrics:
                result_df_day = result_df_day.join(pd.DataFrame(daily_derived_metrics, index=[trade_date]))
            if has_minute_data:
                pvwap_costs_df, day_attributed_minute_map, pvwap_failures = self._calculate_probabilistic_costs(result_df_day, minute_df_daily_grouped, stock_code)
                all_failures.extend(pvwap_failures)
                if not pvwap_costs_df.empty:
                    # [代码修改开始]
                    # 修复：将包含VWAP的成本数据合并到主数据流
                    result_df_day = result_df_day.join(pvwap_costs_df)
                    # [代码修改结束]
                    attributed_minute_map.update(day_attributed_minute_map)
                    advanced_behavioral_df = self._calculate_advanced_behavioral_metrics(result_df_day, day_attributed_minute_map)
                    result_df_day = result_df_day.join(advanced_behavioral_df)
            all_metrics_list.append(result_df_day)
        if not all_metrics_list:
            return pd.DataFrame(), {}, all_failures
        final_df = pd.concat(all_metrics_list)
        return final_df, attributed_minute_map, all_failures

    def _calculate_daily_derived_metrics(self, daily_data_series: pd.Series) -> dict:
        """
        【V23.2 · 幽灵笔误修复版】
        - 核心修复: 将错误的 'pct_chg' 修正为正确的 'pct_change'，确保能从数据流中正确获取涨跌幅。
        - 新增探针: 在 flow_efficiency_index 计算前植入探针，监控所有输入参数的状态。
        """
        results = {}
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
            if pd.notna(base_value):
                confirmation_score = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)) and np.sign(base_value) == np.sign(pd.to_numeric(daily_data_series.get(conf_col), errors='coerce')))
                available_sources = sum(1 for conf_col in confirm_cols if pd.notna(daily_data_series.get(conf_col)))
                calibration_factor = (1 + confirmation_score) / (1 + available_sources) if available_sources > 0 else 1.0
                results[target_col] = base_value * calibration_factor
            else:
                results[target_col] = np.nan
        turnover_amount_yuan = pd.to_numeric(daily_data_series.get('amount'), errors='coerce') * 1000
        if pd.notna(turnover_amount_yuan) and turnover_amount_yuan > 0:
            base_flow = pd.to_numeric(daily_data_series.get('main_force_net_flow_tushare'), errors='coerce')
            confirm_flows = [pd.to_numeric(daily_data_series.get(c), errors='coerce') for c in ['main_force_net_flow_ths', 'main_force_net_flow_dc']]
            if pd.notna(base_flow):
                deviations = [abs(conf_flow - base_flow) / turnover_amount_yuan for conf_flow in confirm_flows if pd.notna(conf_flow)]
                results['flow_credibility_index'] = (1.0 - np.mean(deviations)) * 100 if deviations else 50.0
            mf_flow = results.get('main_force_net_flow_calibrated')
            retail_flow = results.get('retail_net_flow_calibrated')
            if pd.notna(mf_flow) and pd.notna(retail_flow):
                battle_volume = min(abs(mf_flow), abs(retail_flow))
                results['mf_retail_battle_intensity'] = np.sign(mf_flow) * (battle_volume / (turnover_amount_yuan / 10000)) * 100
        mf_buy = np.nansum([
            pd.to_numeric(daily_data_series.get('buy_lg_amount'), errors='coerce'),
            pd.to_numeric(daily_data_series.get('buy_elg_amount'), errors='coerce')
        ])
        mf_sell = np.nansum([
            pd.to_numeric(daily_data_series.get('sell_lg_amount'), errors='coerce'),
            pd.to_numeric(daily_data_series.get('sell_elg_amount'), errors='coerce')
        ])
        mf_total_activity = mf_buy + mf_sell
        total_turnover_wan = turnover_amount_yuan / 10000
        if total_turnover_wan > 0:
            results['main_force_activity_ratio'] = (mf_total_activity / total_turnover_wan) * 100
        if mf_total_activity > 0:
            mf_net_calibrated = results.get('main_force_net_flow_calibrated')
            if pd.notna(mf_net_calibrated):
                results['main_force_flow_directionality'] = (mf_net_calibrated / mf_total_activity) * 100
        def get_directionality(buy_c, sell_c):
            b = np.nan_to_num(pd.to_numeric(daily_data_series.get(buy_c), errors='coerce'))
            s = np.nan_to_num(pd.to_numeric(daily_data_series.get(sell_c), errors='coerce'))
            return (b - s) / (b + s) if (b + s) > 0 else 0.0
        xl_directionality = get_directionality('buy_elg_amount', 'sell_elg_amount')
        lg_directionality = get_directionality('buy_lg_amount', 'sell_lg_amount')
        results['main_force_conviction_index'] = ((xl_directionality + lg_directionality) / 2.0) * (1.0 - abs(xl_directionality - lg_directionality)) * 100
        mf_flow = results.get('main_force_net_flow_calibrated')
        retail_flow = results.get('retail_net_flow_calibrated')
        if pd.notna(mf_flow) and pd.notna(retail_flow):
            total_opinionated_flow = abs(mf_flow) + abs(retail_flow)
            if total_opinionated_flow > 0:
                dominance_ratio = abs(retail_flow) / total_opinionated_flow
                divergence_penalty = 1 if np.sign(mf_flow) != np.sign(retail_flow) and mf_flow != 0 and retail_flow != 0 else 0
                results['retail_flow_dominance_index'] = np.sign(retail_flow) * dominance_ratio * (1 + divergence_penalty) * 100
        # [代码修改开始]
        # 修复：将错误的 'pct_chg' 修正为正确的 'pct_change'
        pct_change = pd.to_numeric(daily_data_series.get('pct_change'), errors='coerce')
        # [代码修改结束]
        if pd.notna(pct_change) and pd.notna(mf_flow) and total_turnover_wan > 0:
            standardized_mf_flow = mf_flow / total_turnover_wan
            if abs(standardized_mf_flow) > 1e-6:
                results['main_force_price_impact_ratio'] = (pct_change / 100) / standardized_mf_flow
            else:
                results['main_force_price_impact_ratio'] = 0.0 if pct_change == 0 else np.inf * np.sign(pct_change)
        else:
            results['main_force_price_impact_ratio'] = 0.0
        if total_turnover_wan > 0:
            results['main_force_buy_rate_consensus'] = (mf_buy / total_turnover_wan) * 100
        results['flow_efficiency_index'] = 0.0
        circ_mv = pd.to_numeric(daily_data_series.get('circ_mv'), errors='coerce')
        if pd.notna(circ_mv) and circ_mv > 0 and pd.notna(mf_flow) and pd.notna(pct_change):
            flow_input = mf_flow / circ_mv
            if abs(flow_input) > 1e-9:
                efficiency = (pct_change / 100) / flow_input
                results['flow_efficiency_index'] = np.sign(efficiency) * np.log1p(abs(efficiency))
        results['inferred_active_order_size'] = 0.0
        trade_count = pd.to_numeric(daily_data_series.get('trade_count'), errors='coerce')
        if pd.notna(trade_count) and trade_count > 0 and pd.notna(turnover_amount_yuan) and turnover_amount_yuan > 0:
            results['inferred_active_order_size'] = turnover_amount_yuan * 1000 / trade_count
        return results

    def _calculate_probabilistic_costs(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame, stock_code: str) -> tuple[pd.DataFrame, dict, list]:
        """
        【V6.9 · VWAP源头固化版】
        - 核心修复: 在本方法内部计算daily_vwap，并将其作为返回DataFrame的一部分，确保VWAP值能被上游调用者接收和使用。
        """
        if minute_df_grouped is None:
            return pd.DataFrame(index=daily_df.index), {}, []
        results = {}
        failures_list = []
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        required_daily_fund_flow_cols = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'
        ]
        from scipy.spatial.distance import jensenshannon
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            missing_cols = [col for col in required_daily_fund_flow_cols if col not in daily_data or pd.isna(daily_data[col])]
            if missing_cols:
                reason = f"缺失资金流原料数据: {missing_cols}"
                failures_list.append({'stock_code': stock_code, 'trade_date': str(date.date()), 'reason': reason})
                continue
            minute_data_full = minute_df_grouped.loc[[date_key]].copy()
            if minute_data_full.empty:
                continue
            daily_vwap_from_minute = (minute_data_full['amount_yuan'].sum() / minute_data_full['vol_shares'].sum()) if minute_data_full['vol_shares'].sum() > 0 else np.nan
            daily_data['daily_vwap'] = daily_vwap_from_minute
            minute_data_for_day = self._calculate_intraday_attribution_weights(minute_data_full, daily_data)
            day_results = {'trade_time': date}
            # [代码修改开始]
            # 修复：将计算出的VWAP直接存入结果字典，以便最终成为DataFrame的一列
            day_results['daily_vwap'] = daily_vwap_from_minute
            # [代码修改结束]
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
            minute_data_for_day = self._attribute_minute_volume_to_players(minute_data_for_day)
            p_dist = minute_data_for_day['vol_shares'].fillna(0).values / minute_data_for_day['vol_shares'].sum() if minute_data_for_day['vol_shares'].sum() > 0 else np.zeros(len(minute_data_for_day))
            q_dist = np.full_like(p_dist, 1.0 / len(p_dist)) if len(p_dist) > 0 else np.array([])
            day_results['volume_profile_jsd_vs_uniform'] = jensenshannon(p_dist, q_dist)**2 if p_dist.size > 0 and q_dist.size > 0 else np.nan
            day_results['minute_data_attributed'] = minute_data_for_day
            results[date] = day_results
        if not results:
            return pd.DataFrame(), {}, failures_list
        attributed_minute_map = {date: res.pop('minute_data_attributed') for date, res in results.items() if 'minute_data_attributed' in res}
        pvwap_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        # [代码修改开始]
        # 修复：现在pvwap_df自身就包含了daily_vwap列，直接将其传递给下游即可
        aggregate_costs_df = self._calculate_aggregate_pvwap_costs(pvwap_df, daily_df)
        # [代码修改结束]
        final_df = pvwap_df.join(aggregate_costs_df)
        return final_df, attributed_minute_map, failures_list

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.7 · 弹药型号修复版】
        - 核心修复: 在合并数据时，同时补充成交量(_vol)和成交额(_amount)列，根除T+0效率分母为零的错误。
        """
        temp_df = pvwap_df.copy()
        # [代码修改开始]
        # 修复：同时定义需要补充的成交量和成交额列
        cols_to_join = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol',
            'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'sell_md_amount',
            'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount'
        ]
        existing_cols_to_join = [col for col in cols_to_join if col in daily_df.columns]
        # [代码修改结束]
        agg_cols = [
            'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
            'main_force_cost_alpha', 'retail_cost_beta', 'main_force_t0_spread_ratio',
            'main_force_execution_alpha', 'main_force_t0_efficiency', 'flow_temperature_premium'
        ]
        if not existing_cols_to_join:
            return pd.DataFrame(columns=agg_cols, index=pvwap_df.index)
        # [代码修改开始]
        # 修复：合并所有需要的列
        temp_df = temp_df.join(daily_df[existing_cols_to_join])
        # [代码修改结束]
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
            return numerator / denominator.replace(0, np.nan)
        result_agg_df = pd.DataFrame(index=pvwap_df.index)
        result_agg_df['avg_cost_main_buy'] = weighted_avg_cost(['avg_cost_lg_buy', 'avg_cost_elg_buy'], ['buy_lg_vol', 'buy_elg_vol'])
        result_agg_df['avg_cost_main_sell'] = weighted_avg_cost(['avg_cost_lg_sell', 'avg_cost_elg_sell'], ['sell_lg_vol', 'sell_elg_vol'])
        result_agg_df['avg_cost_retail_buy'] = weighted_avg_cost(['avg_cost_sm_buy', 'avg_cost_md_buy'], ['buy_sm_vol', 'buy_md_vol'])
        result_agg_df['avg_cost_retail_sell'] = weighted_avg_cost(['avg_cost_sm_sell', 'avg_cost_md_sell'], ['sell_sm_vol', 'sell_md_vol'])
        daily_vwap = pvwap_df.get('daily_vwap')
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
        result_agg_df['flow_temperature_premium'] = 0.0
        if 'avg_cost_main_buy' in result_agg_df.columns and daily_vwap is not None and not daily_vwap.empty:
            safe_vwap = daily_vwap.replace(0, np.nan)
            result_agg_df['flow_temperature_premium'] = (result_agg_df['avg_cost_main_buy'] / safe_vwap - 1) * 100
        if 'avg_cost_main_sell' in result_agg_df.columns and 'avg_cost_main_buy' in result_agg_df.columns and daily_vwap is not None and not daily_vwap.empty:
            t0_spread = result_agg_df['avg_cost_main_sell'] - result_agg_df['avg_cost_main_buy']
            spread_ratio = (t0_spread / daily_vwap.replace(0, np.nan)) * 100
            result_agg_df['main_force_t0_spread_ratio'] = spread_ratio
        else:
            result_agg_df['main_force_t0_spread_ratio'] = np.nan
        result_agg_df['main_force_execution_alpha'] = 0.0
        mf_buy_vol = np.nansum([
            pd.to_numeric(temp_df.get('buy_lg_vol'), errors='coerce'),
            pd.to_numeric(temp_df.get('buy_elg_vol'), errors='coerce')
        ], axis=0) * 100
        mf_sell_vol = np.nansum([
            pd.to_numeric(temp_df.get('sell_lg_vol'), errors='coerce'),
            pd.to_numeric(temp_df.get('sell_elg_vol'), errors='coerce')
        ], axis=0) * 100
        total_mf_vol = mf_buy_vol + mf_sell_vol
        if daily_vwap is not None and not daily_vwap.empty:
            safe_vwap = daily_vwap.replace(0, np.nan)
            alpha_buy = ((safe_vwap - result_agg_df['avg_cost_main_buy']) / safe_vwap).fillna(0)
            alpha_sell = ((result_agg_df['avg_cost_main_sell'] - safe_vwap) / safe_vwap).fillna(0)
            weighted_alpha = (alpha_buy * mf_buy_vol + alpha_sell * mf_sell_vol)
            result_agg_df['main_force_execution_alpha'] = (weighted_alpha / np.where(total_mf_vol == 0, np.nan, total_mf_vol)) * 100
        result_agg_df['main_force_t0_efficiency'] = 0.0
        mf_buy_amount = np.nansum([
            pd.to_numeric(temp_df.get('buy_lg_amount'), errors='coerce'),
            pd.to_numeric(temp_df.get('buy_elg_amount'), errors='coerce')
        ], axis=0)
        mf_sell_amount = np.nansum([
            pd.to_numeric(temp_df.get('sell_lg_amount'), errors='coerce'),
            pd.to_numeric(temp_df.get('sell_elg_amount'), errors='coerce')
        ], axis=0)
        t0_vol = pd.min(mf_buy_vol, mf_sell_vol) if isinstance(mf_buy_vol, pd.Series) else min(mf_buy_vol, mf_sell_vol)
        if 'avg_cost_main_sell' in result_agg_df.columns and 'avg_cost_main_buy' in result_agg_df.columns:
            t0_profit = (result_agg_df['avg_cost_main_sell'] - result_agg_df['avg_cost_main_buy']) * t0_vol
            total_mf_amount = (mf_buy_amount + mf_sell_amount) * 10000
            result_agg_df['main_force_t0_efficiency'] = (t0_profit / np.where(total_mf_amount == 0, np.nan, total_mf_amount)) * 100
        if 'market_cost_battle_premium' in result_agg_df.columns:
            result_agg_df = result_agg_df.drop(columns=['market_cost_battle_premium'])
        return result_agg_df

    def _attribute_minute_volume_to_players(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 · 新增】将基础成交量归因为主力/散户的核心辅助函数。
        - 核心职责: 聚合基础的 *_vol_attr 列，生成 main_force_* 和 retail_* 级别的成交量列。
        """
        df = minute_df.copy()
        # 聚合计算主力成交量
        df['main_force_buy_vol'] = df.get('lg_buy_vol_attr', 0) + df.get('elg_buy_vol_attr', 0)
        df['main_force_sell_vol'] = df.get('lg_sell_vol_attr', 0) + df.get('elg_sell_vol_attr', 0)
        df['main_force_net_vol'] = df['main_force_buy_vol'] - df['main_force_sell_vol']
        # 聚合计算散户成交量
        df['retail_buy_vol'] = df.get('sm_buy_vol_attr', 0) + df.get('md_buy_vol_attr', 0)
        df['retail_sell_vol'] = df.get('sm_sell_vol_attr', 0) + df.get('md_sell_vol_attr', 0)
        df['retail_net_vol'] = df['retail_buy_vol'] - df['retail_sell_vol']
        return df

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 · 战略精简版】为所有核心资金流指标计算斜率和加速度。
        - 核心裁撤: 从 `sum_cols` 列表中移除了所有已废弃的 PnL 评估类及中间过程指标。
        - 核心同步: `sum_cols` 列表已与当前模型定义严格对齐。
        """
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        # 为加速度定义一个独立的、符合数学定义的短窗口
        ACCEL_WINDOW = 2
        # 根据精简后的模型定义，更新需要计算累计值的列
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

    def _compute_all_behavioral_metrics(self, minute_data: pd.DataFrame, daily_data: pd.Series) -> dict:
        """
        【V3.13 · 分析语义终极修复版】
        - 核心修复: 将所有行为指标的默认值从 0.0 修改为 np.nan。这能从根本上区分“指标值为零”和“指标因条件不满足而未计算”这两种情况，对于下游的量化分析和模型训练至关重要。
        """
        from scipy.signal import find_peaks
        results = {}
        if minute_data.empty:
            return results
        daily_total_volume = daily_data.get('vol', 0) * 100
        daily_vwap = daily_data.get('daily_vwap')
        atr = daily_data.get('atr_14d')
        day_open, day_close = daily_data.get('open_qfq'), daily_data.get('close_qfq')
        day_high, day_low = daily_data.get('high_qfq'), daily_data.get('low_qfq')
        # [代码修改开始]
        # 修复：将所有指标的默认值从 0.0 改为 np.nan，以保证分析语义的正确性
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
        # [代码修改结束]
        gatekeeper_condition = all(pd.notna(v) for v in [daily_vwap, daily_total_volume, atr]) and daily_total_volume > 0 and atr > 0
        if gatekeeper_condition:
            price_deviation_value = (minute_data['minute_vwap'] - daily_vwap) * minute_data['vol_shares']
            results['vwap_control_strength'] = price_deviation_value.sum() / (atr * daily_total_volume)
            price_dev_series = minute_data['minute_vwap'] - daily_vwap
            mf_net_flow_series = minute_data['main_force_net_vol']
            if not price_dev_series.var() == 0 and not mf_net_flow_series.var() == 0 and len(price_dev_series) > 1:
                correlation = price_dev_series.corr(mf_net_flow_series)
                # [代码修改开始]
                results['main_force_vwap_guidance'] = correlation if pd.notna(correlation) else np.nan
                # [代码修改结束]
            position_vs_vwap = np.sign(minute_data['minute_vwap'] - daily_vwap)
            crossings = position_vs_vwap.diff().ne(0)
            results['vwap_crossing_intensity'] = minute_data.loc[crossings, 'vol_shares'].sum() / daily_total_volume
            twap = minute_data['minute_vwap'].mean()
            if pd.notna(twap) and twap > 0:
                results['vwap_structure_skew'] = (daily_vwap - twap) / twap * 100
        opening_battle_df = minute_data[(minute_data['trade_time'].dt.time >= time(9, 30)) & (minute_data['trade_time'].dt.time <= time(9, 45))]
        if not opening_battle_df.empty and len(opening_battle_df) > 1 and pd.notna(atr) and atr > 0:
            price_gain = (opening_battle_df['close'].iloc[-1] - opening_battle_df['open'].iloc[0]) / atr
            battle_amount = (opening_battle_df['vol_shares'] * opening_battle_df['minute_vwap']).sum()
            if battle_amount > 0:
                mf_power = opening_battle_df['main_force_net_vol'].sum() * opening_battle_df['minute_vwap'].mean() / battle_amount
                results['opening_battle_result'] = np.sign(price_gain) * np.sqrt(abs(price_gain)) * (1 + mf_power) * 100
        if all(pd.notna(v) for v in [day_open, day_close, day_high, day_low]):
            body_high, body_low = max(day_open, day_close), min(day_open, day_close)
            upper_shadow_df = minute_data[minute_data['high'] > body_high]
            if not upper_shadow_df.empty and upper_shadow_df['vol_shares'].sum() > 0:
                mf_sell_in_shadow = abs(upper_shadow_df[upper_shadow_df['main_force_net_vol'] < 0]['main_force_net_vol'].sum())
                results['upper_shadow_selling_pressure'] = (mf_sell_in_shadow / upper_shadow_df['vol_shares'].sum())
            lower_shadow_df = minute_data[minute_data['low'] < body_low]
            if not lower_shadow_df.empty and lower_shadow_df['vol_shares'].sum() > 0:
                mf_net_flow = lower_shadow_df['main_force_net_vol'].sum()
                results['lower_shadow_absorption_strength'] = mf_net_flow / lower_shadow_df['vol_shares'].sum()
        if len(minute_data) >= 10 and all(pd.notna(v) for v in [day_open, day_close, day_high, day_low, atr]) and daily_total_volume > 0 and atr > 0:
            mf_activity_ratio = (minute_data['main_force_buy_vol'].sum() + minute_data['main_force_sell_vol'].sum()) / daily_total_volume
            if mf_activity_ratio > 0:
                price_outcome = (day_close - day_open) / atr
                results['trend_conviction_ratio'] = price_outcome / mf_activity_ratio
            day_range = day_high - day_low
            if day_range > 0:
                is_v_shape = (day_close - day_open) > 0
                turn_point_idx = np.argmin(minute_data['low'].values) if is_v_shape else np.argmax(minute_data['high'].values)
                if 0 < turn_point_idx < len(minute_data) - 1:
                    initial_phase = minute_data.iloc[:turn_point_idx]
                    reversal_phase = minute_data.iloc[turn_point_idx:]
                    vol_initial, vol_reversal = initial_phase['vol_shares'].sum(), reversal_phase['vol_shares'].sum()
                    if vol_initial > 0 and vol_reversal > 0:
                        turn_point_vwap = minute_data['minute_vwap'].iloc[turn_point_idx]
                        price_recovery = abs(day_close - turn_point_vwap) / day_range
                        vol_shift = np.log1p(vol_reversal / vol_initial)
                        reversal_mf_net_vol = reversal_phase['main_force_net_vol'].sum()
                        reversal_conviction = reversal_mf_net_vol / vol_reversal if vol_reversal > 0 else 0
                        power_score = price_recovery * vol_shift * reversal_conviction
                        results['reversal_power_index'] = power_score if is_v_shape else -power_score
        if 'minute_vwap' in minute_data.columns:
            df_cmf = minute_data.copy()
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
                results['cmf_divergence_score'] = results.get('main_force_cmf', 0.0) - results.get('holistic_cmf', 0.0)
        if 'main_force_net_vol' in minute_data.columns and pd.notna(day_close):
            vp_global = minute_data.groupby(pd.cut(minute_data['minute_vwap'], bins=30))['vol_shares'].sum()
            if not vp_global.empty:
                vpoc_interval = vp_global.idxmax()
                global_vpoc_price = vpoc_interval.mid
                peak_zone_df = minute_data[
                    (minute_data['minute_vwap'] >= vpoc_interval.left) &
                    (minute_data['minute_vwap'] < vpoc_interval.right)
                ]
                if not peak_zone_df.empty:
                    mf_net_vol_on_peak = peak_zone_df['main_force_net_vol'].sum()
                    results['main_force_on_peak_flow'] = (mf_net_vol_on_peak * global_vpoc_price) / 10000
                mf_net_buy_df = minute_data[minute_data['main_force_net_vol'] > 0]
                if not mf_net_buy_df.empty:
                    vp_mf = mf_net_buy_df.groupby(pd.cut(mf_net_buy_df['minute_vwap'], bins=30))['main_force_net_vol'].sum()
                    if not vp_mf.empty:
                        mf_vpoc = vp_mf.idxmax().mid
                        results['main_force_vpoc'] = mf_vpoc
                        if pd.notna(global_vpoc_price) and global_vpoc_price > 0 and pd.notna(mf_vpoc):
                            results['mf_vpoc_premium'] = (mf_vpoc / global_vpoc_price - 1) * 100
        if 'main_force_net_vol' in minute_data.columns and 'retail_net_vol' in minute_data.columns:
            mf_net_series = minute_data['main_force_net_vol']
            retail_net_series = minute_data['retail_net_vol']
            if not mf_net_series.var() == 0 and not retail_net_series.var() == 0 and len(mf_net_series) > 1:
                rolling_corr = mf_net_series.rolling(window=30).corr(retail_net_series)
                results['mf_retail_liquidity_swap_corr'] = rolling_corr.mean()
        continuous_trading_df = minute_data[minute_data['trade_time'].dt.time < time(14, 57)].copy()
        if not continuous_trading_df.empty and gatekeeper_condition:
            up_minutes = continuous_trading_df[continuous_trading_df['close'] > continuous_trading_df['open']]
            down_minutes = continuous_trading_df[continuous_trading_df['close'] < continuous_trading_df['open']]
            if not up_minutes.empty and not down_minutes.empty:
                up_price_change = (up_minutes['close'] - up_minutes['open']).sum()
                up_vol = up_minutes['vol_shares'].sum()
                down_price_change = (down_minutes['open'] - down_minutes['close']).sum()
                down_vol = down_minutes['vol_shares'].sum()
                if up_vol > 0 and down_price_change > 0:
                    upward_efficacy = (up_price_change / up_vol) if up_vol > 0 else 0
                    downward_resistance = (down_vol / down_price_change) if down_price_change > 0 else np.inf
                    log_thrust_ratio = np.log(upward_efficacy * atr) - np.log(downward_resistance * atr)
                    results['asymmetric_volume_thrust'] = log_thrust_ratio if np.isfinite(log_thrust_ratio) else np.nan
                avg_up_speed = up_price_change / len(up_minutes) if len(up_minutes) > 0 else 0
                avg_down_speed = down_price_change / len(down_minutes) if len(down_minutes) > 0 else 0
                if avg_up_speed > 0 and avg_down_speed > 0:
                    results['volatility_asymmetry_index'] = np.log(avg_up_speed / avg_down_speed)
            day_range = day_high - day_low
            if day_range > 0:
                range_pos_factor = ((day_close - day_low) / day_range) * 2 - 1
                value_dev_factor = np.tanh((day_close - daily_vwap) / atr)
                force_balance_factor = minute_data['main_force_net_vol'].sum() / daily_total_volume if daily_total_volume > 0 else 0
                results['closing_price_deviation_score'] = (0.5 * range_pos_factor + 0.3 * value_dev_factor + 0.2 * force_balance_factor) * 100
            auction_df = minute_data[minute_data['trade_time'].dt.time >= time(14, 57)]
            if not auction_df.empty and not continuous_trading_df.empty:
                pre_auction_close = continuous_trading_df['close'].iloc[-1]
                auction_vol = auction_df['vol_shares'].sum()
                avg_minute_vol = continuous_trading_df['vol_shares'].mean()
                if auction_vol > 0 and avg_minute_vol > 0:
                    price_impact = (day_close - pre_auction_close) / atr
                    volume_impact = np.log1p((auction_vol / 3) / avg_minute_vol)
                    auction_amount = (auction_df['vol_shares'] * auction_df['minute_vwap']).sum()
                    mf_participation = auction_df['main_force_net_vol'].sum() * auction_df['minute_vwap'].mean() / auction_amount if auction_amount > 0 else 0
                    results['closing_auction_ambush'] = price_impact * volume_impact * (1 + mf_participation) * 100
            absorption_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] < daily_vwap]
            if not absorption_zone_df.empty:
                mf_absorption_df = absorption_zone_df[absorption_zone_df['main_force_net_vol'] > 0]
                if not mf_absorption_df.empty:
                    absorption_vol = mf_absorption_df['main_force_net_vol'].sum()
                    if absorption_vol > 0:
                        absorption_cost = (mf_absorption_df['minute_vwap'] * mf_absorption_df['main_force_net_vol']).sum() / absorption_vol
                        cost_deviation = (daily_vwap - absorption_cost) / atr
                        if daily_total_volume > 0:
                            results['dip_absorption_power'] = (absorption_vol / daily_total_volume) * cost_deviation * 100
            peaks, _ = find_peaks(continuous_trading_df['minute_vwap'].values)
            troughs, _ = find_peaks(-continuous_trading_df['minute_vwap'].values)
            turning_points = sorted(list(set(np.concatenate(([0], troughs, peaks, [len(continuous_trading_df)-1])))))
            rally_dist_vol, panic_vol, total_rally_vol, total_panic_vol = 0, 0, 0, 0
            for i in range(len(turning_points) - 1):
                start_idx, end_idx = turning_points[i], turning_points[i+1]
                window_df = continuous_trading_df.iloc[start_idx:end_idx+1]
                if window_df.empty or len(window_df) < 2: continue
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
            if total_rally_vol > 0:
                results['rally_distribution_pressure'] = (rally_dist_vol / total_rally_vol) * 100
            if total_panic_vol > 0:
                results['panic_selling_cascade'] = (panic_vol / total_panic_vol) * 100
            posturing_df = continuous_trading_df[continuous_trading_df['trade_time'].dt.time >= time(14, 30)]
            if not posturing_df.empty and pd.notna(daily_vwap) and atr > 0:
                posturing_vwap = (posturing_df['minute_vwap'] * posturing_df['vol_shares']).sum() / posturing_df['vol_shares'].sum()
                price_posture = (posturing_vwap - daily_vwap) / atr
                posturing_amount = (posturing_df['vol_shares'] * posturing_df['minute_vwap']).sum()
                if posturing_amount > 0:
                    force_posture = (posturing_df['main_force_net_vol'].sum() * posturing_vwap) / posturing_amount
                    results['pre_closing_posturing'] = (0.6 * price_posture + 0.4 * force_posture) * 100
            if day_range > 0:
                fomo_zone_threshold = day_low + 0.75 * day_range
                fomo_zone_df = continuous_trading_df[continuous_trading_df['minute_vwap'] > fomo_zone_threshold]
                if not fomo_zone_df.empty:
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
                if not panic_zone_df.empty:
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
        return results

    def _calculate_intraday_attribution_weights(self, minute_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """
        【V9.0 · 行为归因重构版】
        - 核心革命: 废弃“一体适用”的权重模型，为超大单、大单、中单、小单引入各自独特的、基于行为特征的权重分配逻辑。
        - 核心思想:
          - 超大单(ELG) -> 脉冲修正: 权重集中在成交量和振幅剧增的“暴力分钟”。
          - 大单(LG) -> VWAP修正: 权重与价格偏离VWAP的程度相关，体现战术意图。
          - 中单(MD) -> 动量修正: 权重与短期价格动量相关，体现追涨杀跌特性。
          - 小单(SM) -> 基准压力: 沿用原有的K线形态压力模型作为基准。
        """
        df = minute_data_for_day.copy()
        if 'vol_shares' not in df.columns or df['vol_shares'].sum() < 1e-6 or len(df) < 5:
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_weight'] = 0; df[f'{size}_sell_weight'] = 0
            return df
        # --- 1. 计算基础买卖压力和行为修正因子 ---
        price_range = df['high'] - df['low']
        conditions = [price_range > 0, (price_range == 0) & (df['close'] > df['open']), (price_range == 0) & (df['close'] < df['open'])]
        choices = [(df['close'] - df['low']) / price_range, 1.0, 0.0]
        buy_pressure_proxy = np.select(conditions, choices, default=0.5)
        # 脉冲修正因子 (用于ELG)
        vol_ma = df['vol_shares'].rolling(window=20, min_periods=1).mean()
        range_ma = price_range.rolling(window=20, min_periods=1).mean()
        impulse_modifier = (df['vol_shares'] / vol_ma) * (price_range / range_ma.replace(0, 1))
        impulse_modifier = impulse_modifier.fillna(1).clip(0, 10) # 归一化并限制极端值
        # VWAP修正因子 (用于LG)
        daily_vwap = daily_data.get('daily_vwap')
        if pd.notna(daily_vwap):
            vwap_deviation = (df['minute_vwap'] - daily_vwap) / daily_vwap
            # 买单权重在VWAP下方更高，卖单权重在VWAP上方更高
            lg_buy_modifier = np.exp(-np.maximum(0, vwap_deviation) * 5) # 价格越高，权重越低
            lg_sell_modifier = np.exp(np.minimum(0, vwap_deviation) * 5) # 价格越低，权重越低
        else:
            lg_buy_modifier = pd.Series(1.0, index=df.index); lg_sell_modifier = pd.Series(1.0, index=df.index)
        # 动量修正因子 (用于MD)
        momentum_modifier = df['minute_vwap'].pct_change().rolling(window=5).mean().fillna(0)
        md_buy_modifier = np.exp(momentum_modifier * 50) # 动量为正时权重高
        md_sell_modifier = np.exp(-momentum_modifier * 50) # 动量为负时权重高
        # --- 2. 计算各类订单的非归一化权重 ---
        # 小单 (SM): 使用基础压力模型
        sm_buy_score = df['vol_shares'] * buy_pressure_proxy
        sm_sell_score = df['vol_shares'] * (1 - buy_pressure_proxy)
        # 中单 (MD): 基础压力 + 动量修正
        md_buy_score = sm_buy_score * md_buy_modifier
        md_sell_score = sm_sell_score * md_sell_modifier
        # 大单 (LG): 基础压力 + VWAP修正
        lg_buy_score = sm_buy_score * lg_buy_modifier
        lg_sell_score = sm_sell_score * lg_sell_modifier
        # 超大单 (ELG): 基础压力 + 脉冲修正
        elg_buy_score = sm_buy_score * impulse_modifier
        elg_sell_score = sm_sell_score * impulse_modifier
        # --- 3. 分别归一化并赋值 ---
        scores = {
            'sm': (sm_buy_score, sm_sell_score), 'md': (md_buy_score, md_sell_score),
            'lg': (lg_buy_score, lg_sell_score), 'elg': (elg_buy_score, elg_sell_score)
        }
        for size, (buy_score, sell_score) in scores.items():
            total_buy_score = buy_score.sum()
            df[f'{size}_buy_weight'] = buy_score / total_buy_score if total_buy_score > 1e-9 else 0
            total_sell_score = sell_score.sum()
            df[f'{size}_sell_weight'] = sell_score / total_sell_score if total_sell_score > 1e-9 else 0
        return df

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【V1.5 · 数据完整性修复版】从预加载的DataFrame构建按日分组的数据。"""
        if minute_df is None or minute_df.empty:
            return None
        df = minute_df.copy()
        df.sort_values('trade_time', inplace=True)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        if df['trade_time'].dt.tz is None:
            df['trade_time'] = df['trade_time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        else:
            df['trade_time'] = df['trade_time'].dt.tz_convert('Asia/Shanghai')
        df['date'] = df['trade_time'].dt.date
        df[['amount', 'vol']] = df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
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








