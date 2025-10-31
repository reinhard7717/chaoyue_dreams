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
            # [代码修改开始]
            # 核心修正：移除独立的 daily_vwap 计算步骤，将其整合到核心合成方法中
            self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, chunk_raw_data_df.index)
            chunk_new_metrics_df, _, _ = self._synthesize_and_forge_metrics(stock_code, chunk_raw_data_df)
            # [代码修改结束]
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
        【V20.0 · 双核驱动版】
        - 核心重构: 将计算流程重构为“日线衍生核心”与“分钟衍生核心”两大引擎驱动。
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
                print(f"[资金流服务] [{stock_code}] [{trade_date.date()}] 跳过计算，原因：{reason}")
                all_failures.append({'stock_code': stock_code, 'trade_date': str(trade_date.date()), 'reason': f"[资金流服务] {reason}"})
                continue
            result_df_day = daily_data.copy()
            # [代码修改开始]
            # --- 步骤1: 调用日线衍生指标统一计算核心 ---
            daily_derived_metrics = self._calculate_daily_derived_metrics(result_df_day.iloc[0])
            if daily_derived_metrics:
                result_df_day = result_df_day.join(pd.DataFrame(daily_derived_metrics, index=[trade_date]))
            # --- 步骤2: 计算概率成本 (连接日线与分钟线的桥梁) ---
            if has_minute_data:
                pvwap_costs_df, day_attributed_minute_map, pvwap_failures = self._calculate_probabilistic_costs(result_df_day, minute_df_daily_grouped, stock_code)
                all_failures.extend(pvwap_failures)
                if not pvwap_costs_df.empty:
                    result_df_day = result_df_day.join(pvwap_costs_df)
                    attributed_minute_map.update(day_attributed_minute_map)
                    # --- 步骤3: 调用分钟线衍生指标统一计算核心 ---
                    advanced_behavioral_df = self._calculate_advanced_behavioral_metrics(result_df_day, day_attributed_minute_map)
                    result_df_day = result_df_day.join(advanced_behavioral_df)
            # [代码修改结束]
            all_metrics_list.append(result_df_day)
        if not all_metrics_list:
            return pd.DataFrame(), {}, all_failures
        final_df = pd.concat(all_metrics_list)
        return final_df, attributed_minute_map, all_failures

    def _calculate_daily_derived_metrics(self, daily_data_series: pd.Series) -> dict:
        """
        【V20.0 · 新增 · 日线衍生指标统一计算核心】
        - 核心革命: 整合所有仅依赖日线数据的分散计算方法到一个统一的计算核心中。
        - 核心思想:
          1. 职责单一: 本方法专注于从单日的日线数据中衍生出所有“力量结构”和“共识”相关的指标。
          2. 逻辑内聚: 将校准共识、可信度、博弈烈度、主力动态、交易结构等计算逻辑集中管理。
        """
        # [代码新增开始]
        results = {}
        # --- 1. 校准共识资金流 (Calibrated Consensus) ---
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
        # --- 2. 资金流可信度与博弈烈度 (Credibility & Battle Intensity) ---
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
        # --- 3. 主力动态 (Main Force Dynamics) ---
        mf_buy = pd.to_numeric(daily_data_series.get('buy_lg_amount'), errors='coerce', default=0) + pd.to_numeric(daily_data_series.get('buy_elg_amount'), errors='coerce', default=0)
        mf_sell = pd.to_numeric(daily_data_series.get('sell_lg_amount'), errors='coerce', default=0) + pd.to_numeric(daily_data_series.get('sell_elg_amount'), errors='coerce', default=0)
        mf_total_activity = mf_buy + mf_sell
        total_turnover_wan = turnover_amount_yuan / 10000
        if total_turnover_wan > 0:
            results['main_force_activity_ratio'] = (mf_total_activity / total_turnover_wan) * 100
        if mf_total_activity > 0:
            mf_net_calibrated = results.get('main_force_net_flow_calibrated')
            if pd.notna(mf_net_calibrated):
                results['main_force_flow_directionality'] = (mf_net_calibrated / mf_total_activity) * 100
        # --- 4. 交易结构动态 (Trade Structure Dynamics) ---
        def get_directionality(buy_c, sell_c):
            b = pd.to_numeric(daily_data_series.get(buy_c), errors='coerce', default=0)
            s = pd.to_numeric(daily_data_series.get(sell_c), errors='coerce', default=0)
            return (b - s) / (b + s) if (b + s) > 0 else 0.0
        xl_directionality = get_directionality('buy_elg_amount', 'sell_elg_amount')
        lg_directionality = get_directionality('buy_lg_amount', 'sell_lg_amount')
        results['xl_order_flow_directionality'] = xl_directionality * 100
        results['main_force_conviction_index'] = ((xl_directionality + lg_directionality) / 2.0) * (1.0 - abs(xl_directionality - lg_directionality)) * 100
        # --- 5. 力量格局动态 (Power Structure Dynamics) ---
        mf_flow = results.get('main_force_net_flow_calibrated')
        retail_flow = results.get('retail_net_flow_calibrated')
        if pd.notna(mf_flow) and pd.notna(retail_flow):
            total_opinionated_flow = abs(mf_flow) + abs(retail_flow)
            if total_opinionated_flow > 0:
                dominance_ratio = abs(retail_flow) / total_opinionated_flow
                divergence_penalty = 1 if np.sign(mf_flow) != np.sign(retail_flow) and mf_flow != 0 and retail_flow != 0 else 0
                results['retail_flow_dominance_index'] = np.sign(retail_flow) * dominance_ratio * (1 + divergence_penalty) * 100
        pct_change = pd.to_numeric(daily_data_series.get('pct_chg'), errors='coerce')
        if pd.notna(pct_change) and pd.notna(mf_flow) and total_turnover_wan > 0:
            standardized_mf_flow = mf_flow / total_turnover_wan
            if abs(standardized_mf_flow) > 1e-6:
                results['main_force_price_impact_ratio'] = (pct_change / 100) / standardized_mf_flow
            else:
                results['main_force_price_impact_ratio'] = 0.0 if pct_change == 0 else np.inf * np.sign(pct_change)
        # --- 6. 其他日线级指标 ---
        if total_turnover_yuan > 0:
            results['main_force_buy_rate_consensus'] = (mf_buy * 10000 / total_turnover_yuan) * 100
        return results
        # [代码新增结束]

    def _calculate_probabilistic_costs(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame, stock_code: str) -> tuple[pd.DataFrame, dict, list]:
        """
        【V6.5 · 职责净化版】移除已废弃的 aggression_index_opening 计算逻辑。
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
                print(f"[{stock_code}] [{date.date()}] 跳过概率成本计算，{reason}")
                failures_list.append({'stock_code': stock_code, 'trade_date': str(date.date()), 'reason': reason})
                continue
            minute_data_full = minute_df_grouped.loc[[date_key]].copy()
            minute_data_continuous = minute_data_full[minute_data_full['is_continuous_trading']].copy()
            if minute_data_continuous.empty:
                continue
            minute_data_for_day = self._calculate_intraday_attribution_weights(minute_data_continuous, daily_data)
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
            minute_data_for_day = self._attribute_minute_volume_to_players(minute_data_for_day)
            p_dist = minute_data_for_day['vol_shares'].fillna(0).values / minute_data_for_day['vol_shares'].sum() if minute_data_for_day['vol_shares'].sum() > 0 else np.zeros(len(minute_data_for_day))
            q_dist = np.full_like(p_dist, 1.0 / len(p_dist)) if len(p_dist) > 0 else np.array([])
            day_results['volume_profile_jsd_vs_uniform'] = jensenshannon(p_dist, q_dist)**2 if p_dist.size > 0 and q_dist.size > 0 else np.nan
            # [代码修改开始]
            # 核心修正：移除已废弃的 aggression_index_opening 计算逻辑
            # [代码修改结束]
            day_results['minute_data_attributed'] = minute_data_for_day
            results[date] = day_results
        if not results:
            return pd.DataFrame(), {}, failures_list
        attributed_minute_map = {date: res.pop('minute_data_attributed') for date, res in results.items() if 'minute_data_attributed' in res}
        pvwap_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        aggregate_costs_df = self._calculate_aggregate_pvwap_costs(pvwap_df, daily_df)
        final_df = pvwap_df.join(aggregate_costs_df)
        return final_df, attributed_minute_map, failures_list

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.0 · 成本博弈分析版】
        - 核心重构: 废弃简单的成本聚合，引入全新的“主力成本Alpha”和“散户成本Beta”指标。
        - 核心思想:
          - 主力成本Alpha: 量化主力买入行为相对于市场平均的“超额收益能力”。
          - 散户成本Beta: 量化散户买入行为相对于主力卖出行为的“被收割程度”。
        """
        temp_df = pvwap_df.copy()
        vol_cols = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'
        ]
        existing_vol_cols = [col for col in vol_cols if col in daily_df.columns]
        # [代码修改开始]
        # 核心修改: 聚合列中加入新的成本博弈指标
        agg_cols = [
            'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
            'main_force_cost_alpha', 'retail_cost_beta', 'main_force_t0_spread_ratio'
        ]
        # [代码修改结束]
        if not existing_vol_cols:
            return pd.DataFrame(columns=agg_cols, index=pvwap_df.index)
        temp_df = temp_df.join(daily_df[existing_vol_cols])
        def weighted_avg_cost(cost_cols, vol_cols):
            numerator = pd.Series(0.0, index=temp_df.index)
            denominator = pd.Series(0.0, index=temp_df.index)
            for cost_col, vol_col in zip(cost_cols, vol_cols):
                if cost_col in temp_df.columns and vol_col in temp_df.columns:
                    cost = temp_df[cost_col]
                    volume_shares = pd.to_numeric(temp_df[vol_col], errors='coerce').fillna(0) * 100
                    value_contribution = (cost * volume_shares).fillna(0)
                    numerator += value_contribution
                    denominator += volume_shares.where(cost.notna(), 0)
            return numerator / denominator.replace(0, np.nan)
        result_agg_df = pd.DataFrame(index=pvwap_df.index)
        result_agg_df['avg_cost_main_buy'] = weighted_avg_cost(['avg_cost_lg_buy', 'avg_cost_elg_buy'], ['buy_lg_vol', 'buy_elg_vol'])
        result_agg_df['avg_cost_main_sell'] = weighted_avg_cost(['avg_cost_lg_sell', 'avg_cost_elg_sell'], ['sell_lg_vol', 'sell_elg_vol'])
        result_agg_df['avg_cost_retail_buy'] = weighted_avg_cost(['avg_cost_sm_buy', 'avg_cost_md_buy'], ['buy_sm_vol', 'buy_md_vol'])
        result_agg_df['avg_cost_retail_sell'] = weighted_avg_cost(['avg_cost_sm_sell', 'avg_cost_md_sell'], ['sell_sm_vol', 'sell_md_vol'])
        daily_vwap = daily_df.get('daily_vwap')
        # [代码修改开始]
        # 核心新增: 计算主力成本Alpha和散户成本Beta
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
        # [代码修改结束]
        if 'avg_cost_main_sell' in result_agg_df.columns and 'avg_cost_main_buy' in result_agg_df.columns and daily_vwap is not None and not daily_vwap.empty:
            t0_spread = result_agg_df['avg_cost_main_sell'] - result_agg_df['avg_cost_main_buy']
            spread_ratio = (t0_spread / daily_vwap.replace(0, np.nan)) * 100
            result_agg_df['main_force_t0_spread_ratio'] = spread_ratio
        else:
            result_agg_df['main_force_t0_spread_ratio'] = np.nan
        # 移除旧的、意义不明确的 market_cost_battle_premium
        if 'market_cost_battle_premium' in result_agg_df.columns:
            result_agg_df = result_agg_df.drop(columns=['market_cost_battle_premium'])
        return result_agg_df

    def _attribute_minute_volume_to_players(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 · 新增】将基础成交量归因为主力/散户的核心辅助函数。
        - 核心职责: 聚合基础的 *_vol_attr 列，生成 main_force_* 和 retail_* 级别的成交量列。
        """
        # [代码新增开始]
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
        # [代码新增结束]

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """【V3.7 · 导数协议修正版】为二阶导数（加速度）使用独立的、正确的短计算窗口。"""
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        # 为加速度定义一个独立的、符合数学定义的短窗口
        ACCEL_WINDOW = 2
        
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

    def _calculate_advanced_behavioral_metrics(self, daily_df: pd.DataFrame, minute_df_attributed_grouped: dict) -> pd.DataFrame:
        """
        【V22.0 · 行为指标最终整合版】
        - 核心革命: 将孤立的 `_upgrade_behavioral_metrics` 方法的逻辑，完全整合进本统一计算核心中。
        - 核心思想: 激活 `opening_gambit_score`, `main_force_absorption_strength` 等指标，实现所有分钟级行为指标的“单一入口、一次计算”。
        """
        if not minute_df_attributed_grouped:
            return pd.DataFrame(index=daily_df.index)
        import pandas_ta as ta
        all_results = {}
        for date, daily_data in daily_df.iterrows():
            if date not in minute_df_attributed_grouped:
                continue
            minute_data = minute_df_attributed_grouped[date].copy()
            if minute_data.empty:
                continue
            day_results = {'trade_time': date}
            minute_data['trade_time_local'] = minute_data['trade_time'].dt.tz_convert('Asia/Shanghai').dt.time
            # --- 1. 战术流动强度 (Tactical Flow Intensity) ---
            if len(minute_data) >= 20:
                minute_data.ta.bbands(length=60, append=True)
                lower_band_col, upper_band_col = 'BBL_60_2.0', 'BBU_60_2.0'
                if lower_band_col in minute_data.columns:
                    dip_zone_mask = minute_data['minute_vwap'] < minute_data[lower_band_col]
                    dip_minutes = minute_data[dip_zone_mask]
                    day_results['dip_absorption_power'] = (dip_minutes['main_force_net_vol'].sum() / dip_minutes['vol_shares'].sum() * 100) if not dip_minutes.empty and dip_minutes['vol_shares'].sum() > 0 else 0
                    rally_zone_mask = minute_data['minute_vwap'] > minute_data[upper_band_col]
                    rally_minutes = minute_data[rally_zone_mask]
                    day_results['rally_distribution_pressure'] = (rally_minutes['main_force_net_vol'].sum() / rally_minutes['vol_shares'].sum() * 100) if not rally_minutes.empty and rally_minutes['vol_shares'].sum() > 0 else 0
            daily_total_volume = daily_data.get('vol', 0) * 100
            if daily_total_volume > 0 and not minute_data.empty:
                minute_data['price_pos'] = (minute_data['close'] - minute_data['low']) / (minute_data['high'] - minute_data['low']).replace(0, np.nan)
                avg_minute_vol = minute_data['vol_shares'].mean()
                is_panic_minute = (minute_data['price_pos'] < 0.25) & (minute_data['vol_shares'] > 1.5 * avg_minute_vol)
                panic_streak = is_panic_minute.rolling(window=3).sum()
                cascade_mask = (panic_streak >= 3).rolling(window=3).max().fillna(0).astype(bool)
                cascade_minutes = minute_data[cascade_mask]
                if not cascade_minutes.empty:
                    retail_net_sell_in_cascade = cascade_minutes['retail_net_vol'].sum()
                    day_results['panic_selling_cascade'] = (-retail_net_sell_in_cascade / daily_total_volume) * 100 if retail_net_sell_in_cascade < 0 else 0
                else:
                    day_results['panic_selling_cascade'] = 0
            # --- 2. 关键时刻动态 (Key Moment Dynamics) ---
            pre_close = daily_data.get('pre_close')
            open_price = daily_data.get('open')
            opening_15min = minute_data[minute_data['trade_time_local'] < time(9, 45)]
            if not opening_15min.empty and pd.notna(pre_close) and pre_close > 0 and pd.notna(open_price) and open_price > 0:
                gap_strength = (open_price - pre_close) / pre_close
                vwap_15min_vol = opening_15min['vol_shares'].sum()
                vwap_15min = (opening_15min['minute_vwap'] * opening_15min['vol_shares']).sum() / vwap_15min_vol if vwap_15min_vol > 0 else open_price
                gap_defense_strength = (vwap_15min - open_price) / open_price
                mf_net_vol_15min = opening_15min['main_force_net_vol'].sum()
                mf_confirmation_factor = 1 + (mf_net_vol_15min / vwap_15min_vol if vwap_15min_vol > 0 else 0)
                day_results['opening_battle_result'] = (gap_strength + gap_defense_strength) * mf_confirmation_factor * 100
            pre_closing_minutes = minute_data[(minute_data['trade_time_local'] >= time(14, 30)) & (minute_data['trade_time_local'] < time(14, 57))]
            if not pre_closing_minutes.empty and daily_total_volume > 0:
                mf_total_activity = pre_closing_minutes['main_force_buy_vol'].sum() + pre_closing_minutes['main_force_sell_vol'].sum()
                tail_directionality = pre_closing_minutes['main_force_net_vol'].sum() / mf_total_activity if mf_total_activity > 0 else 0
                tail_volume_share = pre_closing_minutes['vol_shares'].sum() / daily_total_volume
                day_results['pre_closing_posturing'] = tail_directionality * tail_volume_share * 100
            final_close = daily_data.get('close')
            atr = daily_data.get('atr_14d')
            continuous_minutes = minute_data[minute_data['is_continuous_trading']].copy()
            auction_minutes = minute_data[minute_data['trade_time_local'] >= time(14, 57)]
            if not continuous_minutes.empty and not auction_minutes.empty and pd.notna(final_close) and pd.notna(atr) and atr > 0:
                price_1457 = continuous_minutes.iloc[-1]['close']
                price_surprise_factor = (final_close - price_1457) / atr
                last_hour_avg_vol = minute_data[minute_data['trade_time_local'] >= time(14, 0)]['vol_shares'].mean()
                if pd.notna(last_hour_avg_vol) and last_hour_avg_vol > 0:
                    volume_energy_factor = np.log1p(auction_minutes['vol_shares'].sum() / last_hour_avg_vol)
                    day_results['closing_auction_ambush'] = price_surprise_factor * volume_energy_factor * 10
            # --- 3. 成本博弈 (Cost Game) ---
            costs = {'main_buy': daily_data.get('avg_cost_main_buy'), 'main_sell': daily_data.get('avg_cost_main_sell'), 'retail_buy': daily_data.get('avg_cost_retail_buy'), 'retail_sell': daily_data.get('avg_cost_retail_sell')}
            vols = {'main_buy_vol': (daily_data.get('buy_lg_vol', 0) + daily_data.get('buy_elg_vol', 0)) * 100, 'main_sell_vol': (daily_data.get('sell_lg_vol', 0) + daily_data.get('sell_elg_vol', 0)) * 100}
            mf_buy_minutes = minute_data[minute_data['main_force_net_vol'] > 0]
            if not mf_buy_minutes.empty and pd.notna(costs['main_buy']):
                mf_active_vol = mf_buy_minutes['vol_shares'].sum()
                mf_active_vwap = (mf_buy_minutes['minute_vwap'] * mf_buy_minutes['vol_shares']).sum() / mf_active_vol if mf_active_vol > 0 else np.nan
                if pd.notna(mf_active_vwap) and mf_active_vwap > 0:
                    day_results['main_force_execution_alpha'] = ((mf_active_vwap - costs['main_buy']) / mf_active_vwap) * 100
            if pd.notna(costs['main_buy']) and costs['main_buy'] > 0 and pd.notna(costs['retail_sell']):
                day_results['retail_panic_surrender_index'] = ((costs['main_buy'] - costs['retail_sell']) / costs['main_buy']) * 100
            if pd.notna(costs['main_sell']) and costs['main_sell'] > 0 and pd.notna(costs['retail_buy']):
                day_results['retail_fomo_premium_index'] = ((costs['retail_buy'] - costs['main_sell']) / costs['main_sell']) * 100
            if pd.notna(costs['main_buy']) and pd.notna(costs['main_sell']) and vols['main_buy_vol'] > 0 and vols['main_sell_vol'] > 0:
                t0_profit = min(vols['main_buy_vol'], vols['main_sell_vol']) * (costs['main_sell'] - costs['main_buy'])
                total_capital_activity = (vols['main_buy_vol'] * costs['main_buy']) + (vols['main_sell_vol'] * costs['main_sell'])
                if total_capital_activity > 0: day_results['main_force_t0_efficiency'] = (t0_profit / total_capital_activity) * 100
            # --- 4. 市场脉搏 (Market Pulse) ---
            def get_window_vwap(start_time, end_time):
                window_df = minute_data[(minute_data['trade_time_local'] >= start_time) & (minute_data['trade_time_local'] < end_time)]
                return (window_df['minute_vwap'] * window_df['vol_shares']).sum() / window_df['vol_shares'].sum() if not window_df.empty and window_df['vol_shares'].sum() > 0 else np.nan
            opening_vwap = get_window_vwap(time(9, 30), time(9, 45))
            midday_vwap = get_window_vwap(time(9, 45), time(14, 45))
            closing_vwap = get_window_vwap(time(14, 45), time(15, 1))
            if all(pd.notna(v) for v in [opening_vwap, midday_vwap, closing_vwap, atr]) and atr > 0:
                day_results['vwap_structure_skew'] = (((opening_vwap + closing_vwap) / 2 - midday_vwap) / atr) * 100
            pct_chg = daily_data.get('pct_chg')
            mf_net_flow = daily_data.get('main_force_net_flow_calibrated')
            total_turnover = daily_data.get('amount')
            if all(pd.notna(v) for v in [pct_chg, mf_net_flow, total_turnover]) and total_turnover > 0:
                mf_strength = mf_net_flow / (total_turnover / 10)
                if abs(mf_strength) > 1e-6: day_results['flow_efficiency_index'] = (pct_chg / 100) / mf_strength
            up_vol = minute_data[minute_data['close'] > minute_data['open']]['vol_shares'].sum()
            down_vol = minute_data[minute_data['close'] < minute_data['open']]['vol_shares'].sum()
            if down_vol > 0: day_results['asymmetric_volume_thrust'] = np.log(up_vol / down_vol) if up_vol > 0 else -np.inf
            else: day_results['asymmetric_volume_thrust'] = np.inf if up_vol > 0 else 0
            # --- 5. 盈亏矩阵 (P&L Matrix) ---
            realized_profit_yuan = (costs['main_sell'] - costs['main_buy']) * min(vols['main_buy_vol'], vols['main_sell_vol'])
            net_pos_change_vol = vols['main_buy_vol'] - vols['main_sell_vol']
            net_pos_change_cost = np.where(net_pos_change_vol > 0, costs['main_buy'], costs['main_sell'])
            unrealized_pnl_yuan = (final_close - net_pos_change_cost) * net_pos_change_vol
            day_results['t0_arbitrage_profit'] = realized_profit_yuan / 10000
            day_results['positional_pnl'] = unrealized_pnl_yuan / 10000
            day_results['total_trading_pnl'] = (realized_profit_yuan.fillna(0) + unrealized_pnl_yuan.fillna(0)) / 10000
            daily_vwap = daily_data.get('daily_vwap')
            if pd.notna(daily_vwap):
                actual_net_cashflow = (costs['main_buy'] * vols['main_buy_vol']) - (costs['main_sell'] * vols['main_sell_vol'])
                benchmark_net_cashflow = net_pos_change_vol * daily_vwap
                total_activity_value = (costs['main_buy'] * vols['main_buy_vol']) + (costs['main_sell'] * vols['main_sell_vol'])
                if pd.notna(total_activity_value) and total_activity_value > 0:
                    day_results['execution_cost_alpha'] = ((benchmark_net_cashflow - actual_net_cashflow) / total_activity_value) * 100
            # --- 6. 最终评估 (Final Assessment) ---
            total_pnl = day_results.get('total_trading_pnl', 0)
            exec_alpha = day_results.get('execution_cost_alpha', 0)
            if 'total_activity_value' in locals() and total_activity_value > 0:
                pnl_efficiency = (total_pnl * 10000) / total_activity_value
                day_results['pnl_quality_score'] = pnl_efficiency * (1 + (exec_alpha / 100)) * 100
            minute_returns = minute_data['minute_vwap'].pct_change()
            upside_vol = minute_returns[minute_returns > 0].std()
            downside_vol = minute_returns[minute_returns < 0].std()
            if pd.notna(downside_vol) and downside_vol > 1e-9:
                day_results['volatility_asymmetry_index'] = np.log((upside_vol if pd.notna(upside_vol) else 0) / downside_vol)
            else:
                day_results['volatility_asymmetry_index'] = np.inf if pd.notna(upside_vol) and upside_vol > 0 else 0
            midday_df = minute_data[(minute_data['trade_time_local'] >= time(9, 45)) & (minute_data['trade_time_local'] < time(14, 57))]
            if not midday_df.empty and not auction_minutes.empty and pd.notna(atr) and atr > 0 and pd.notna(final_close):
                midday_avg_vol = midday_df['vol_shares'].mean()
                if pd.notna(midday_vwap) and pd.notna(midday_avg_vol) and midday_avg_vol > 0:
                    price_deviation = (final_close - midday_vwap) / atr
                    volume_energy = np.log1p(auction_minutes['vol_shares'].sum() / midday_avg_vol)
                    day_results['closing_price_deviation_score'] = price_deviation * (1 + np.sign(price_deviation) * volume_energy) * 10
            # --- 7. 战术日志指标 (Tactical Log Metrics) ---
            if daily_total_volume > 0:
                opening_mask = minute_data['trade_time_local'] < time(10, 0)
                day_results['main_force_opening_blitz'] = (minute_data.loc[opening_mask, 'main_force_net_vol'].sum() / daily_total_volume) * 100
                closing_mask = minute_data['trade_time_local'] >= time(14, 30)
                day_results['main_force_closing_assault'] = (minute_data.loc[closing_mask, 'main_force_net_vol'].sum() / daily_total_volume) * 100
            if pd.notna(daily_vwap) and daily_vwap > 0 and 'total_activity_value' in locals():
                buy_side_alpha = (daily_vwap - costs['main_buy']) * vols['main_buy_vol'] if pd.notna(costs['main_buy']) and vols['main_buy_vol'] > 0 else 0
                sell_side_alpha = (costs['main_sell'] - daily_vwap) * vols['main_sell_vol'] if pd.notna(costs['main_sell']) and vols['main_sell_vol'] > 0 else 0
                if total_activity_value > 0:
                    day_results['main_force_vwap_alpha'] = ((buy_side_alpha + sell_side_alpha) / total_activity_value) * 100
            if not continuous_minutes.empty:
                continuous_minutes['cumulative_mf_net_vol'] = continuous_minutes['main_force_net_vol'].cumsum()
                def get_flow_slope(df_period):
                    if len(df_period) < 2: return 0.0
                    return np.polyfit(np.arange(len(df_period)), df_period['cumulative_mf_net_vol'].values, 1)[0]
                slope_am = get_flow_slope(continuous_minutes[continuous_minutes['trade_time_local'].dt.time < time(12, 0)])
                slope_pm = get_flow_slope(continuous_minutes[continuous_minutes['trade_time_local'].dt.time >= time(13, 0)])
                avg_vol_per_minute = daily_total_volume / 240 if daily_total_volume > 0 else 1
                day_results['flow_acceleration_index'] = (slope_pm - slope_am) / avg_vol_per_minute
                main_force_net_flows = continuous_minutes['main_force_net_vol'].to_numpy()
                n = len(main_force_net_flows)
                if n > 0:
                    denominator = n * np.sum(np.abs(main_force_net_flows))
                    day_results['flow_timing_index'] = np.sum(np.arange(1, n + 1) * main_force_net_flows) / denominator if denominator > 0 else 0.5
                if 'vol_shares' in continuous_minutes.columns and daily_total_volume > 0:
                    vol_ma_20 = continuous_minutes['vol_shares'].rolling(window=20, min_periods=1).mean()
                    impulse_minutes_df = continuous_minutes[continuous_minutes['vol_shares'] > 3 * vol_ma_20]
                    day_results['volume_impulse_index'] = (impulse_minutes_df['main_force_net_vol'].sum() / daily_total_volume) * 100 if not impulse_minutes_df.empty else 0.0
                calc_df = pd.DataFrame({'price_change': continuous_minutes['close'].diff(), 'main_force_net_vol': continuous_minutes['main_force_net_vol']}).dropna()
                if len(calc_df) < 2 or calc_df['price_change'].var() == 0 or calc_df['main_force_net_vol'].var() == 0:
                    day_results['volume_price_momentum'] = 0.0
                else:
                    correlation = calc_df['price_change'].corr(calc_df['main_force_net_vol'])
                    day_results['volume_price_momentum'] = correlation if pd.notna(correlation) else 0.0
            # [代码新增开始]
            # --- 8. 升级版行为指标 (Upgraded Behavioral Metrics) ---
            opening_window_mask = minute_data['trade_time_local'] < time(9, 45)
            opening_minutes = minute_data[opening_window_mask]
            if not opening_minutes.empty and pd.notna(pre_close) and pre_close > 0:
                opening_volume = opening_minutes['vol_shares'].sum()
                if opening_volume > 0:
                    vwap_opening = (opening_minutes['minute_vwap'] * opening_minutes['vol_shares']).sum() / opening_volume
                    price_thrust = (vwap_opening - pre_close) / pre_close
                    force_multiplier = opening_minutes['main_force_net_vol'].sum() / opening_volume
                    day_results['opening_gambit_score'] = price_thrust * force_multiplier * 10000
            if pd.notna(daily_vwap) and daily_vwap > 0:
                absorption_minutes = minute_data[minute_data['minute_vwap'] < daily_vwap]
                if not absorption_minutes.empty and absorption_minutes['vol_shares'].sum() > 0:
                    day_results['main_force_absorption_strength'] = (absorption_minutes['main_force_buy_vol'].sum() / absorption_minutes['vol_shares'].sum()) * 100
                distribution_minutes = minute_data[minute_data['minute_vwap'] > daily_vwap]
                if not distribution_minutes.empty and distribution_minutes['vol_shares'].sum() > 0:
                    day_results['main_force_distribution_intensity'] = (distribution_minutes['main_force_sell_vol'].sum() / distribution_minutes['vol_shares'].sum()) * 100
            minute_returns = minute_data['minute_vwap'].pct_change().fillna(0)
            panic_minutes = minute_data[(minute_returns < (minute_returns.mean() - 2 * minute_returns.std())) & (minute_data['vol_shares'] > (minute_data['vol_shares'].mean() + 2 * minute_data['vol_shares'].std()))]
            if not panic_minutes.empty:
                total_sell_in_panic = panic_minutes['main_force_sell_vol'].sum() + panic_minutes['retail_sell_vol'].sum()
                if total_sell_in_panic > 0:
                    retail_sell_dominance = panic_minutes['retail_sell_vol'].sum() / total_sell_in_panic
                    total_buy_in_panic = panic_minutes['main_force_buy_vol'].sum() + panic_minutes['retail_buy_vol'].sum()
                    main_force_absorption_rate = panic_minutes['main_force_buy_vol'].sum() / total_buy_in_panic if total_buy_in_panic > 0 else 0
                    day_results['panic_selling_intensity'] = retail_sell_dominance * main_force_absorption_rate * 100
            if len(minute_data) > 1 and not continuous_minutes.empty:
                if pd.notna(final_close) and pd.notna(price_1457) and pd.notna(daily_vwap) and daily_vwap > 0 and pd.notna(avg_minute_vol) and avg_minute_vol > 0:
                    price_impact = (final_close - price_1457) / daily_vwap
                    volume_power = auction_minutes['vol_shares'].sum() / avg_minute_vol
                    day_results['closing_auction_conviction'] = price_impact * volume_power * 100
            tail_minutes = minute_data[minute_data['trade_time_local'] >= time(14, 30)]
            if not tail_minutes.empty and daily_total_volume > 0:
                day_results['tail_thrust_intensity'] = (tail_minutes['main_force_net_vol'].sum() / daily_total_volume) * 100
            price_change_series = minute_data['minute_vwap'].diff().fillna(0)
            if not main_force_net_flows.empty and not price_change_series.empty and main_force_net_flows.std() > 0 and price_change_series.std() > 0:
                correlation = main_force_net_flows.corr(price_change_series)
                day_results['intraday_execution_alpha'] = -correlation if pd.notna(correlation) else 0
            # [代码新增结束]
            all_results[date] = day_results
        if not all_results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(all_results, orient='index').set_index('trade_time')

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
        # [代码修改开始]
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
        # [代码修改结束]

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【V1.4 · 竞价隔离版】从预加载的DataFrame构建按日分组的数据。"""
        if minute_df is None or minute_df.empty:
            return None
        df = minute_df.copy()
        df.sort_values('trade_time', inplace=True)
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        # 强制转换为北京时间，以便进行时间过滤
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
        # 标记集合竞价时段
        CONTINUOUS_TRADING_END_TIME = time(14, 57, 0)
        df['is_continuous_trading'] = df['trade_time'].dt.time < CONTINUOUS_TRADING_END_TIME
        
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













