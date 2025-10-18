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
        self.max_lookback_days = 200

    async def run_precomputation(self, stock_code: str, is_incremental: bool):
        """【V1.2 · 数据链路修复版】服务层主执行器"""
        # 1. 初始化上下文
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental
        )
        # 2. 加载并合并所有日线级别数据源
        merged_df = await self._load_and_merge_sources(stock_info, fetch_start_date)
        if merged_df.empty:
            return 0
        # 3. 【升维】加载分钟数据并计算日度VWAP
        daily_vwap_series = await self._calculate_daily_vwap(stock_info, merged_df.index)
        merged_df['daily_vwap'] = daily_vwap_series
        # [代码修改开始] 将 stock_code 参数改为 stock_info 对象，确保查询的准确性
        self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, merged_df.index)
        # [代码修改结束]
        # 4. 一站式合成、交叉验证并锻造高级指标（已注入VWAP）
        df_with_advanced_metrics = self._synthesize_and_forge_metrics(stock_code, merged_df)
        # 5. 计算所有指标的衍生值（斜率、加速度等）
        final_metrics_df = self._calculate_derivatives(stock_code, df_with_advanced_metrics)
        # 6. 准备并保存数据
        processed_count = await self._prepare_and_save_data(
            stock_info, MetricsModel, final_metrics_df, is_incremental_final, last_metric_date
        )
        if hasattr(self, '_minute_df_daily_grouped'):
            del self._minute_df_daily_grouped
        return processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool):
        """初始化任务上下文"""
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        last_metric_date = None
        if is_incremental:
            @sync_to_async(thread_sensitive=True)
            def get_latest_metric_async(model, stock_info_obj):
                try:
                    return model.objects.filter(stock=stock_info_obj).latest('trade_time')
                except model.DoesNotExist:
                    return None
            latest_metric = await get_latest_metric_async(MetricsModel, stock_info)
            if latest_metric:
                last_metric_date = latest_metric.trade_time
            else:
                is_incremental = False
        fetch_start_date = None
        if is_incremental and last_metric_date:
            fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

    async def _load_and_merge_sources(self, stock_info, fetch_start_date):
        """加载、标准化并合并多源数据"""
        @sync_to_async(thread_sensitive=True)
        def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_date=None):
            if not model: return pd.DataFrame()
            qs = model.objects.filter(stock=stock_info_obj)
            if start_date:
                qs = qs.filter(**{f'{date_field}__gte': start_date})
            return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
        data_tasks = {
            "tushare": get_data_async(get_fund_flow_model_by_code(stock_info.stock_code), stock_info, start_date=fetch_start_date),
            "ths": get_data_async(get_fund_flow_ths_model_by_code(stock_info.stock_code), stock_info, start_date=fetch_start_date),
            "dc": get_data_async(get_fund_flow_dc_model_by_code(stock_info.stock_code), stock_info, start_date=fetch_start_date),
            "daily": get_data_async(get_daily_data_model_by_code(stock_info.stock_code), stock_info, fields=('trade_time', 'amount', 'close'), start_date=fetch_start_date),
            "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'circ_mv', 'turnover_rate'), start_date=fetch_start_date),
        }
        results = await asyncio.gather(*data_tasks.values())
        data_dfs = dict(zip(data_tasks.keys(), results))
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
                df = df.rename(columns={'net_amount': 'net_flow_ths', 'buy_lg_amount': 'main_force_net_flow_ths', 'buy_md_amount': 'net_md_amount_ths', 'buy_sm_amount': 'net_sh_amount_ths'})
                df['retail_net_flow_ths'] = df.get('net_md_amount_ths', 0).fillna(0) + df.get('net_sh_amount_ths', 0).fillna(0)
                return df[['trade_time', 'net_flow_ths', 'main_force_net_flow_ths', 'retail_net_flow_ths']]
            elif source == 'dc':
                df = df.rename(columns={'net_amount': 'main_force_net_flow_dc', 'buy_elg_amount': 'net_xl_amount_dc', 'buy_lg_amount': 'net_lg_amount_dc', 'buy_md_amount': 'net_md_amount_dc', 'buy_sm_amount': 'net_sh_amount_dc'})
                df['net_flow_dc'] = df.get('main_force_net_flow_dc', 0).fillna(0) + df.get('net_md_amount_dc', 0).fillna(0) + df.get('net_sh_amount_dc', 0).fillna(0)
                df['retail_net_flow_dc'] = df.get('net_md_amount_dc', 0).fillna(0) + df.get('net_sh_amount_dc', 0).fillna(0)
                return df[['trade_time', 'net_flow_dc', 'main_force_net_flow_dc', 'retail_net_flow_dc', 'net_xl_amount_dc']]
            return df
        df_tushare = standardize_and_prepare(data_dfs['tushare'], 'tushare')
        df_ths = standardize_and_prepare(data_dfs['ths'], 'ths')
        df_dc = standardize_and_prepare(data_dfs['dc'], 'dc')
        dfs_to_merge = [df for df in [df_tushare, df_ths, df_dc] if not df.empty]
        if not dfs_to_merge:
            raise ValueError("所有资金流数据源均为空")
        merged_df = reduce(lambda left, right: pd.merge(left, right, on='trade_time', how='outer'), dfs_to_merge)
        merged_df = merged_df.sort_values('trade_time').set_index('trade_time')
        daily_dfs_to_join = []
        if not data_dfs['daily'].empty:
            daily_df = data_dfs['daily'].set_index(pd.to_datetime(data_dfs['daily']['trade_time'])).drop(columns='trade_time')
            daily_dfs_to_join.append(daily_df)
        if not data_dfs['daily_basic'].empty:
            daily_basic_df = data_dfs['daily_basic'].set_index(pd.to_datetime(data_dfs['daily_basic']['trade_time'])).drop(columns='trade_time')
            daily_dfs_to_join.append(daily_basic_df)
        if daily_dfs_to_join:
            merged_df = merged_df.join(daily_dfs_to_join, how='inner')
        return merged_df

    async def _calculate_daily_vwap(self, stock_info, date_index: pd.DatetimeIndex) -> pd.Series:
        """【V1.1 · 修正版】从分钟数据计算日度VWAP"""
        # [代码修改开始] 修正了获取分钟级别数据模型的方法名，以匹配 model_helpers.py 中的定义
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        # [代码修改结束]
        if not MinuteModel or date_index.empty:
            print(f"调试信息: 未能为 {stock_info.stock_code} 找到1分钟线模型，VWAP计算将跳过。")
            return pd.Series(np.nan, index=date_index)
        @sync_to_async(thread_sensitive=True)
        def get_minute_data(model, stock_info_obj, start_date, end_date):
            qs = model.objects.filter(
                stock=stock_info_obj,
                trade_time__date__gte=start_date,
                trade_time__date__lte=end_date
            ).values('trade_time', 'amount', 'vol')
            return pd.DataFrame.from_records(qs)
        min_date, max_date = date_index.min().date(), date_index.max().date()
        minute_df = await get_minute_data(MinuteModel, stock_info, min_date, max_date)
        if minute_df.empty:
            return pd.Series(np.nan, index=date_index)
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'])
        minute_df[['amount', 'vol']] = minute_df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
        minute_df['total_value'] = minute_df['amount'] * 1000 # 成交额单位是千元，转换为元
        minute_df['total_volume'] = minute_df['vol'] * 100 # 成交量单位是手，转换为股
        daily_agg = minute_df.groupby(minute_df['trade_time'].dt.date)
        daily_total_value = daily_agg['total_value'].sum()
        daily_total_volume = daily_agg['total_volume'].sum()
        daily_vwap = daily_total_value / daily_total_volume.replace(0, np.nan)
        daily_vwap.index = pd.to_datetime(daily_vwap.index)
        return daily_vwap.reindex(date_index)

    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame) -> pd.DataFrame:
        """【V1.5 · 逻辑修复版】修复因分钟数据缺失导致下游计算流程中断的BUG"""
        df = merged_df.copy()
        tushare_cols_exist = 'buy_sm_vol' in df.columns
        if tushare_cols_exist:
            minute_df_daily_grouped = getattr(self, '_minute_df_daily_grouped', None)
            # [代码修改开始] 重构逻辑，确保所有高级指标计算都在分钟数据有效时执行
            if minute_df_daily_grouped is None or minute_df_daily_grouped.empty:
                print(f"调试信息: {stock_code} 在 {df.index.min().date()} 到 {df.index.max().date()} 期间分钟数据未预加载或为空，跳过PVWAP及所有基于分钟线的高级指标计算。")
                # 使用旧的、不依赖分钟数据的成本计算作为后备方案
                cost_pairs = {
                    'avg_cost_sm_buy': ('buy_sm_amount', 'buy_sm_vol'), 'avg_cost_sm_sell': ('sell_sm_amount', 'sell_sm_vol'),
                    'avg_cost_md_buy': ('buy_md_amount', 'buy_md_vol'), 'avg_cost_md_sell': ('sell_md_amount', 'sell_md_vol'),
                    'avg_cost_lg_buy': ('buy_lg_amount', 'buy_lg_vol'), 'avg_cost_lg_sell': ('sell_lg_amount', 'sell_lg_vol'),
                    'avg_cost_elg_buy': ('buy_elg_amount', 'buy_elg_vol'), 'avg_cost_elg_sell': ('sell_elg_amount', 'sell_elg_vol'),
                }
                for new_col, (amount_col, vol_col) in cost_pairs.items():
                    amount = pd.to_numeric(df[amount_col], errors='coerce') * 10000
                    vol = pd.to_numeric(df[vol_col], errors='coerce') * 100
                    df[new_col] = amount / vol.replace(0, np.nan)
                # 计算聚合成本的后备方法
                df['avg_cost_main_buy'] = (df['buy_lg_amount'] * 10000 + df['buy_elg_amount'] * 10000) / (df['buy_lg_vol'] * 100 + df['buy_elg_vol'] * 100).replace(0, np.nan)
                df['avg_cost_main_sell'] = (df['sell_lg_amount'] * 10000 + df['sell_elg_amount'] * 10000) / (df['sell_lg_vol'] * 100 + df['sell_elg_vol'] * 100).replace(0, np.nan)
                df['avg_cost_retail_buy'] = (df['buy_sm_amount'] * 10000 + df['buy_md_amount'] * 10000) / (df['buy_sm_vol'] * 100 + df['buy_md_vol'] * 100).replace(0, np.nan)
                df['avg_cost_retail_sell'] = (df['sell_sm_amount'] * 10000 + df['sell_md_amount'] * 10000) / (df['sell_sm_vol'] * 100 + df['sell_md_vol'] * 100).replace(0, np.nan)
            else:
                # 只有在分钟数据有效时，才执行所有依赖它的高级计算
                print(f"调试信息: {stock_code} 分钟数据加载成功，开始计算所有高级指标。")
                pvwap_costs_df = self._calculate_probabilistic_costs(df, minute_df_daily_grouped)
                df = df.join(pvwap_costs_df)
                pnl_matrix_df = self._upgrade_intraday_profit_metric(df)
                df = df.join(pnl_matrix_df)
                behavioral_metrics_df = self._upgrade_behavioral_metrics(df, minute_df_daily_grouped)
                df = df.join(behavioral_metrics_df)
                structure_metrics_df = self._calculate_intraday_structure_metrics(df, minute_df_daily_grouped)
                df = df.join(structure_metrics_df)
            # 无论成本如何计算，这些衍生指标都依赖于成本字段，因此放在if/else之后
            df['cost_divergence_mf_vs_retail'] = df['avg_cost_main_buy'] - df['avg_cost_retail_sell']
            df['cost_weighted_main_flow'] = df.get('main_force_net_flow_tushare', np.nan) * df['avg_cost_main_buy']
            df['main_buy_cost_advantage'] = np.divide(df['avg_cost_main_buy'], df['close'], out=np.full_like(df['close'].values, np.nan, dtype=float), where=df['close']!=0) - 1
            df['market_cost_battle'] = df['avg_cost_main_buy'] - df['avg_cost_retail_buy']
            if 'daily_vwap' in df.columns:
                df['main_buy_cost_vs_vwap'] = df['avg_cost_main_buy'] - df['daily_vwap']
                df['main_sell_cost_vs_vwap'] = df['avg_cost_main_sell'] - df['daily_vwap']
            if 'trade_count' in df.columns and 'amount' in df.columns:
                total_turnover_yuan = pd.to_numeric(df['amount'], errors='coerce').values * 1000
                trade_count_np = pd.to_numeric(df['trade_count'], errors='coerce').values
                df['avg_order_value'] = np.divide(total_turnover_yuan, trade_count_np, out=np.full_like(total_turnover_yuan, np.nan, dtype=float), where=trade_count_np!=0)
                close_price_np = pd.to_numeric(df['close'], errors='coerce').values
                avg_order_value_np = df['avg_order_value'].values
                df['avg_order_value_norm_price'] = np.divide(avg_order_value_np, close_price_np, out=np.full_like(avg_order_value_np, np.nan, dtype=float), where=close_price_np!=0)
            # [代码修改结束]
        consensus_map = {
            'net_flow_consensus': ['net_flow_tushare', 'net_flow_ths', 'net_flow_dc'],
            'main_force_net_flow_consensus': ['main_force_net_flow_tushare', 'main_force_net_flow_ths', 'main_force_net_flow_dc'],
            'retail_net_flow_consensus': ['retail_net_flow_tushare', 'retail_net_flow_ths', 'retail_net_flow_dc'],
            'net_xl_amount_consensus': ['net_xl_amount_tushare', 'net_xl_amount_dc'],
            'net_lg_amount_consensus': ['net_lg_amount_tushare'],
            'net_md_amount_consensus': ['net_md_amount_tushare', 'net_md_amount_ths', 'net_md_amount_dc'],
            'net_sh_amount_consensus': ['net_sh_amount_tushare', 'net_sh_amount_ths', 'net_sh_amount_dc'],
        }
        for target_col, source_cols in consensus_map.items():
            existing_cols = [col for col in source_cols if col in df.columns]
            if existing_cols:
                df[target_col] = df[existing_cols].mean(axis=1)
            else:
                df[target_col] = np.nan
        if 'main_force_net_flow_tushare' in df.columns and 'main_force_net_flow_ths' in df.columns:
            df['divergence_ts_ths'] = df['main_force_net_flow_tushare'] - df['main_force_net_flow_ths']
        if 'main_force_net_flow_tushare' in df.columns and 'main_force_net_flow_dc' in df.columns:
            df['divergence_ts_dc'] = df['main_force_net_flow_tushare'] - df['main_force_net_flow_dc']
        if 'main_force_net_flow_ths' in df.columns and 'main_force_net_flow_dc' in df.columns:
            df['divergence_ths_dc'] = df['main_force_net_flow_ths'] - df['main_force_net_flow_dc']
        safe_denom = lambda v: v.replace(0, np.nan)
        total_turnover_yuan = pd.to_numeric(df.get('amount'), errors='coerce') * 1000
        main_force_net_flow_yuan = pd.to_numeric(df.get('main_force_net_flow_consensus'), errors='coerce') * 10000
        circ_mv_yuan = pd.to_numeric(df.get('circ_mv'), errors='coerce') * 10000
        df['main_force_flow_impact_ratio'] = main_force_net_flow_yuan / safe_denom(circ_mv_yuan)
        if 'avg_order_value' in df.columns:
            df['trade_granularity_impact'] = df['avg_order_value'] / safe_denom(circ_mv_yuan)
        df['main_force_flow_intensity_ratio'] = main_force_net_flow_yuan / safe_denom(total_turnover_yuan)
        df['main_force_buy_rate_consensus'] = (main_force_net_flow_yuan / safe_denom(circ_mv_yuan)) * 100
        df['flow_divergence_mf_vs_retail'] = df.get('main_force_net_flow_consensus', np.nan) - df.get('retail_net_flow_consensus', np.nan)
        df['main_force_vs_xl_divergence'] = df.get('main_force_net_flow_consensus', np.nan) - df.get('net_xl_amount_consensus', np.nan)
        net_lg_consensus = df.get('net_lg_amount_consensus')
        if net_lg_consensus is not None:
            df['main_force_conviction_ratio'] = df.get('net_xl_amount_consensus', np.nan) / safe_denom(net_lg_consensus)
        else:
            df['main_force_conviction_ratio'] = np.nan
        total_xl_trade_yuan = pd.to_numeric(df.get('net_xl_amount_consensus'), errors='coerce').abs() * 10000
        df['trade_concentration_index'] = total_xl_trade_yuan / safe_denom(total_turnover_yuan)
        return df

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标的衍生指标（斜率、加速度等）"""
        final_df = consensus_df.copy()
        # [代码修改开始] 将新的派发压力指标加入衍生计算列表
        CORE_METRICS_TO_DERIVE = [
            'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
            'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus', 'net_sh_amount_consensus',
            'flow_divergence_mf_vs_retail', 'main_force_flow_intensity_ratio', 'main_force_vs_xl_divergence',
            'main_force_flow_impact_ratio', 'trade_granularity_impact', 'avg_order_value_norm_price',
            'trade_concentration_index', 'avg_order_value', 'main_force_conviction_ratio',
            'avg_cost_sm_buy', 'avg_cost_sm_sell', 'avg_cost_md_buy', 'avg_cost_md_sell',
            'avg_cost_lg_buy', 'avg_cost_lg_sell', 'avg_cost_elg_buy', 'avg_cost_elg_sell',
            'avg_cost_main_buy', 'avg_cost_main_sell', 'avg_cost_retail_buy', 'avg_cost_retail_sell',
            'cost_divergence_mf_vs_retail', 'cost_weighted_main_flow', 'main_buy_cost_advantage',
            'market_cost_battle',
            'daily_vwap', 'main_buy_cost_vs_vwap', 'main_sell_cost_vs_vwap',
            'divergence_ts_ths', 'divergence_ts_dc', 'divergence_ths_dc',
            'realized_profit_on_exchange', 'net_position_change_value', 'unrealized_pnl_on_net_change',
            'pnl_matrix_confidence_score', 'vwap_tracking_error', 'volume_profile_jsd_vs_uniform',
            'aggression_index_opening', 'main_force_support_strength', 'main_force_distribution_pressure',
            'retail_capitulation_score', 'intraday_execution_alpha',
            'intraday_volatility', 'closing_strength_index', 'close_vs_vwap_ratio', 'final_hour_momentum',
        ]
        # [代码修改结束]
        sum_cols = [
            'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
            'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
            'net_sh_amount_consensus', 'cost_weighted_main_flow',
            'divergence_ts_ths', 'divergence_ts_dc', 'divergence_ths_dc',
            'realized_profit_on_exchange', 'net_position_change_value', 'unrealized_pnl_on_net_change',
        ]
        UNIFIED_PERIODS = [1, 5, 13, 21, 55]
        for p in UNIFIED_PERIODS:
            if p <= 1: continue
            for col in sum_cols:
                if col in final_df.columns:
                    sum_col_name = f'{col}_sum_{p}d'
                    final_df[sum_col_name] = final_df[col].rolling(window=p, min_periods=max(2, p // 2)).sum()
        all_cols_to_derive = CORE_METRICS_TO_DERIVE + [f'{c}_sum_{p}d' for c in sum_cols for p in UNIFIED_PERIODS if p > 1]
        for col in all_cols_to_derive:
            if col in final_df.columns:
                source_series = final_df[col].astype(float)
                for p in UNIFIED_PERIODS:
                    calc_window = 2 if p == 1 else p
                    slope_col_name = f'{col}_slope_{p}d'
                    slope_series = final_df.ta.slope(close=source_series, length=calc_window)
                    final_df[slope_col_name] = slope_series
                    if slope_series is not None and not slope_series.empty:
                        accel_col_name = f'{col}_accel_{p}d'
                        final_df[accel_col_name] = final_df.ta.slope(close=slope_series.astype(float), length=calc_window)
        return final_df

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame, is_incremental: bool, last_metric_date):
        """准备并保存最终计算结果到数据库"""
        if is_incremental and last_metric_date:
            records_to_save_df = final_df[final_df.index.date > last_metric_date]
        else:
            records_to_save_df = final_df
        if records_to_save_df.empty:
            return 0
        records_to_save_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = records_to_save_df[[col for col in records_to_save_df.columns if col in model_fields]]
        records_list = df_filtered.to_dict('records')
        records_to_create = []
        for record_date, record_data in zip(df_filtered.index, records_list):
            safe_record_data = {
                key: None if isinstance(value, float) and not np.isfinite(value) else value
                for key, value in record_data.items()
            }
            records_to_create.append(
                MetricsModel(
                    stock=stock_info,
                    trade_time=record_date.date(),
                    **safe_record_data
                )
            )
        @sync_to_async(thread_sensitive=True)
        def save_metrics_async(model, stock_info_obj, records_to_create_list, do_delete_first: bool):
            with transaction.atomic():
                if do_delete_first:
                    model.objects.filter(stock=stock_info_obj).delete()
                model.objects.bulk_create(records_to_create_list, batch_size=2000)
        await save_metrics_async(MetricsModel, stock_info, records_to_create, not is_incremental)
        return len(records_to_create)

    @sync_to_async(thread_sensitive=True)
    # [代码修改开始] 修改方法签名，接收 stock_info 对象而非 stock_code 字符串
    def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex):
        """【V1.1 · 修复与增强版】获取并按日聚合分钟数据，为PVWAP计算提供材料"""
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        # [代码修改开始] 增加更精细的调试信息
        if not MinuteModel:
            print(f"调试信息: {stock_info.stock_code} 未能找到对应的1分钟线数据模型。")
            return None
        if date_index.empty:
            print(f"调试信息: {stock_info.stock_code} 的日期索引为空，无法查询分钟数据。")
            return None
        min_date, max_date = date_index.min().date(), date_index.max().date()
        # [代码修改开始] 使用 stock_info 对象进行外键查询，更准确高效
        qs = MinuteModel.objects.filter(
            stock=stock_info,
            trade_time__date__gte=min_date,
            trade_time__date__lte=max_date
        ).values('trade_time', 'amount', 'vol')
        # [代码修改结束]
        minute_df = pd.DataFrame.from_records(qs)
        # [代码修改开始] 增加更精细的调试信息
        if minute_df.empty:
            print(f"调试信息: {stock_info.stock_code} 在 {min_date} 到 {max_date} 期间的数据库查询结果为空。")
            return None
        # [代码修改结束]
        minute_df['trade_time'] = pd.to_datetime(minute_df['trade_time'])
        minute_df['date'] = minute_df['trade_time'].dt.date
        minute_df[['amount', 'vol']] = minute_df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
        minute_df['amount_yuan'] = minute_df['amount'] * 1000
        minute_df['vol_shares'] = minute_df['vol'] * 100
        minute_df['minute_vwap'] = minute_df['amount_yuan'] / minute_df['vol_shares'].replace(0, np.nan)
        daily_total_vol = minute_df.groupby('date')['vol_shares'].transform('sum')
        minute_df['vol_weight'] = minute_df['vol_shares'] / daily_total_vol.replace(0, np.nan)
        return minute_df.set_index('date')

    def _calculate_probabilistic_costs(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.2 · 算法指纹版】计算PVWAP成本，并同步提取算法交易指纹
        """
        if minute_df_grouped is None:
            print("调试信息: 分钟数据为空，无法计算PVWAP成本及算法指纹。")
            # Fallback to old method if no minute data
            cost_pairs = {
                'avg_cost_sm_buy': ('buy_sm_amount', 'buy_sm_vol'), 'avg_cost_sm_sell': ('sell_sm_amount', 'sell_sm_vol'),
                'avg_cost_md_buy': ('buy_md_amount', 'buy_md_vol'), 'avg_cost_md_sell': ('sell_md_amount', 'sell_md_vol'),
                'avg_cost_lg_buy': ('buy_lg_amount', 'buy_lg_vol'), 'avg_cost_lg_sell': ('sell_lg_amount', 'sell_lg_vol'),
                'avg_cost_elg_buy': ('buy_elg_amount', 'buy_elg_vol'), 'avg_cost_elg_sell': ('sell_elg_amount', 'sell_elg_vol'),
            }
            results_df = pd.DataFrame(index=daily_df.index)
            for new_col, (amount_col, vol_col) in cost_pairs.items():
                amount = pd.to_numeric(daily_df[amount_col], errors='coerce') * 10000
                vol = pd.to_numeric(daily_df[vol_col], errors='coerce') * 100
                results_df[new_col] = amount / vol.replace(0, np.nan)
            return self._calculate_aggregate_pvwap_costs(results_df, daily_df)
        results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        # [代码新增开始] 导入 scipy 用于计算JS散度
        from scipy.spatial.distance import jensenshannon
        # [代码新增结束]
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = minute_df_grouped.loc[[date_key]]
            day_results = {'trade_time': date}
            # --- 1. 计算PVWAP成本 ---
            for cost_type in cost_types:
                daily_vol_shares = pd.to_numeric(daily_data.get(f'{cost_type}_vol'), errors='coerce') * 100
                if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                    day_results[f'avg_cost_{cost_type}'] = np.nan
                    continue
                attributed_vol = minute_data_for_day['vol_weight'] * daily_vol_shares
                attributed_value = attributed_vol * minute_data_for_day['minute_vwap']
                total_attributed_value = attributed_value.sum()
                total_attributed_vol = attributed_vol.sum()
                day_results[f'avg_cost_{cost_type}'] = total_attributed_value / total_attributed_vol if total_attributed_vol else np.nan
            # [代码新增开始] 2. 计算算法交易指纹 ---
            # 指纹A: VWAP跟踪误差 (需要先在下游计算出聚合成本avg_cost_main_buy)
            # 我们将在下游 _calculate_aggregate_pvwap_costs 之后计算
            # 指纹B: 成交量分布均匀度 (JS散度)
            p_dist = minute_data_for_day['vol_weight'].fillna(0).values
            q_dist = np.full_like(p_dist, 1.0 / len(p_dist)) # 完美均匀分布
            day_results['volume_profile_jsd_vs_uniform'] = jensenshannon(p_dist, q_dist)**2 # JS散度平方，使其范围在0-1
            # 指纹C: 开盘进攻性指数
            first_hour_mask = (minute_data_for_day['trade_time'].dt.hour == 9) & (minute_data_for_day['trade_time'].dt.minute >= 30) | \
                              (minute_data_for_day['trade_time'].dt.hour == 10) & (minute_data_for_day['trade_time'].dt.minute < 30)
            first_hour_vol = minute_data_for_day[first_hour_mask]['vol_shares'].sum()
            total_day_vol = minute_data_for_day['vol_shares'].sum()
            day_results['aggression_index_opening'] = first_hour_vol / total_day_vol if total_day_vol else np.nan
            # [代码新增结束]
            results[date] = day_results
        if not results:
            return pd.DataFrame()
        final_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        # --- 3. 计算聚合成本 ---
        final_df = self._calculate_aggregate_pvwap_costs(final_df, daily_df)
        # [代码新增开始] 4. 计算依赖于聚合成本的VWAP跟踪误差 ---
        if 'avg_cost_main_buy' in final_df.columns and 'daily_vwap' in daily_df.columns:
            # 合并日度VWAP以便计算
            final_df = final_df.join(daily_df['daily_vwap'])
            final_df['vwap_tracking_error'] = final_df['avg_cost_main_buy'] - final_df['daily_vwap']
        # [代码新增结束]
        return final_df

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 精确加权版】基于PVWAP基础成本，计算按量加权的聚合成本
        - 核心修正: 引入日线资金流数据(daily_df)，使用各分力（大单、特大单等）的真实成交量作为权重，
                      替代了原有的简单算术平均，极大提升了聚合成本的准确性。
        """
        df = pvwap_df.copy()
        # 为了加权，我们需要从 daily_df 中获取成交量（单位：手）
        vol_cols = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'
        ]
        # 将成交量数据合并到 pvwap_df 中，以便计算
        df = df.join(daily_df[vol_cols])
        # 定义计算函数
        def weighted_avg_cost(cost_cols, vol_cols):
            total_value = pd.Series(0.0, index=df.index)
            total_volume = pd.Series(0.0, index=df.index)
            for cost_col, vol_col in zip(cost_cols, vol_cols):
                cost = df.get(cost_col, 0).fillna(0)
                volume = pd.to_numeric(df.get(vol_col, 0), errors='coerce').fillna(0)
                total_value += cost * volume
                total_volume += volume
            return total_value / total_volume.replace(0, np.nan)
        # [代码修改开始] 使用新的加权函数计算聚合成本
        # 主力买入成本
        df['avg_cost_main_buy'] = weighted_avg_cost(
            ['avg_cost_lg_buy', 'avg_cost_elg_buy'],
            ['buy_lg_vol', 'buy_elg_vol']
        )
        # 主力卖出成本
        df['avg_cost_main_sell'] = weighted_avg_cost(
            ['avg_cost_lg_sell', 'avg_cost_elg_sell'],
            ['sell_lg_vol', 'sell_elg_vol']
        )
        # 散户买入成本
        df['avg_cost_retail_buy'] = weighted_avg_cost(
            ['avg_cost_sm_buy', 'avg_cost_md_buy'],
            ['buy_sm_vol', 'buy_md_vol']
        )
        # 散户卖出成本
        df['avg_cost_retail_sell'] = weighted_avg_cost(
            ['avg_cost_sm_sell', 'avg_cost_md_sell'],
            ['sell_sm_vol', 'sell_md_vol']
        )
        # [代码修改结束]
        # 返回时不包含临时的成交量列
        return df.drop(columns=vol_cols, errors='ignore')

    def _upgrade_intraday_profit_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【新增】构建“主力日内三维P&L矩阵”，并融合多源数据计算可信度评分。
        """
        results_df = pd.DataFrame(index=df.index)
        # --- 准备数据 ---
        cost_buy = df.get('avg_cost_main_buy').fillna(0)
        cost_sell = df.get('avg_cost_main_sell').fillna(0)
        vol_buy = pd.to_numeric(df.get('buy_lg_vol'), errors='coerce').fillna(0) + pd.to_numeric(df.get('buy_elg_vol'), errors='coerce').fillna(0)
        vol_sell = pd.to_numeric(df.get('sell_lg_vol'), errors='coerce').fillna(0) + pd.to_numeric(df.get('sell_elg_vol'), errors='coerce').fillna(0)
        close_price = df.get('close').fillna(0)
        # 转换成交量单位：手 -> 股
        vol_buy_shares = vol_buy * 100
        vol_sell_shares = vol_sell * 100
        # --- 维度一: 已实现利润 (Realized P&L) ---
        exchanged_volume = np.minimum(vol_buy_shares, vol_sell_shares)
        results_df['realized_profit_on_exchange'] = (cost_sell - cost_buy) * exchanged_volume
        # --- 维度二: 净头寸变动 (Net Position Change) ---
        net_pos_change_vol = vol_buy_shares - vol_sell_shares
        # 净买入时，成本为买入成本；净卖出时，成本为卖出成本
        net_pos_change_cost = np.where(net_pos_change_vol > 0, cost_buy, cost_sell)
        results_df['net_position_change_value'] = net_pos_change_vol * net_pos_change_cost
        # --- 维度三: 浮动盈亏 (Unrealized P&L) ---
        results_df['unrealized_pnl_on_net_change'] = (close_price - net_pos_change_cost) * net_pos_change_vol
        # --- 可信度评分 (Confidence Score) ---
        # 1. Tushare 净流入方向 (基于我们的计算)
        dir_ts = np.sign(results_df['net_position_change_value'])
        # 2. THS 净流入方向
        dir_ths = np.sign(df.get('main_force_net_flow_ths', 0).fillna(0))
        # 3. DC 净流入方向
        dir_dc = np.sign(df.get('main_force_net_flow_dc', 0).fillna(0))
        # 计算一致性
        agreement_count = (dir_ts == dir_ths).astype(int) + (dir_ts == dir_dc).astype(int) + (dir_ths == dir_dc).astype(int)
        # 3个方向一致 -> agreement_count=3 -> score=1.0
        # 2个方向一致 -> agreement_count=1 -> score=0.67 (近似2/3)
        # 3个方向混战 -> agreement_count=0 -> score=0.33 (近似1/3)
        # 注意：这里用 (agreement_count / 3 * 2 + 1) / 3 是一个映射技巧
        results_df['pnl_matrix_confidence_score'] = ((agreement_count / 3 * 2) + 1) / 3
        return results_df

    def _upgrade_behavioral_metrics(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 攻防一体版】升维战术行为指标：支撑强度、派发压力、投降分、执行Alpha
        """
        if minute_df_grouped is None:
            print("调试信息: 分钟数据为空，无法升维战术行为指标。")
            return pd.DataFrame(index=daily_df.index)
        results = {}
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = minute_df_grouped.loc[[date_key]].copy()
            day_results = {'trade_time': date}
            # --- 准备分钟级归属资金流数据 ---
            fund_types = ['main_force', 'retail']
            for fund_type in fund_types:
                if fund_type == 'main_force': # 主力包含大单和特大单
                    daily_buy_vol = (pd.to_numeric(daily_data.get('buy_lg_vol'), 'coerce') + pd.to_numeric(daily_data.get('buy_elg_vol'), 'coerce')).fillna(0) * 100
                    daily_sell_vol = (pd.to_numeric(daily_data.get('sell_lg_vol'), 'coerce') + pd.to_numeric(daily_data.get('sell_elg_vol'), 'coerce')).fillna(0) * 100
                else: # 散户包含小单和中单
                    daily_buy_vol = (pd.to_numeric(daily_data.get('buy_sm_vol'), 'coerce') + pd.to_numeric(daily_data.get('buy_md_vol'), 'coerce')).fillna(0) * 100
                    daily_sell_vol = (pd.to_numeric(daily_data.get('sell_sm_vol'), 'coerce') + pd.to_numeric(daily_data.get('sell_md_vol'), 'coerce')).fillna(0) * 100
                minute_data_for_day[f'{fund_type}_net_vol'] = (daily_buy_vol - daily_sell_vol) * minute_data_for_day['vol_weight']
                minute_data_for_day[f'{fund_type}_buy_vol'] = daily_buy_vol * minute_data_for_day['vol_weight']
                minute_data_for_day[f'{fund_type}_sell_vol'] = daily_sell_vol * minute_data_for_day['vol_weight']
            # --- 1. `main_force_support_strength` (主力支撑强度) ---
            low_threshold = minute_data_for_day['minute_vwap'].quantile(0.1)
            bottom_zone_minutes = minute_data_for_day[minute_data_for_day['minute_vwap'] <= low_threshold]
            if not bottom_zone_minutes.empty:
                support_net_flow = bottom_zone_minutes['main_force_net_vol'].sum()
                total_main_buy = minute_data_for_day['main_force_buy_vol'].sum()
                day_results['main_force_support_strength'] = support_net_flow / total_main_buy if total_main_buy else np.nan
            else:
                day_results['main_force_support_strength'] = 0
            # [代码新增开始] 2. `main_force_distribution_pressure` (主力派发压力) ---
            high_threshold = minute_data_for_day['minute_vwap'].quantile(0.9)
            top_zone_minutes = minute_data_for_day[minute_data_for_day['minute_vwap'] >= high_threshold]
            if not top_zone_minutes.empty:
                distribution_net_flow = top_zone_minutes['main_force_net_vol'].sum() # 派发时，此值为负
                total_main_sell = minute_data_for_day['main_force_sell_vol'].sum()
                # 主力在顶部区间的净卖出量 / 主力全天总卖出量 (取负值使其为正数)
                day_results['main_force_distribution_pressure'] = -distribution_net_flow / total_main_sell if total_main_sell else np.nan
            else:
                day_results['main_force_distribution_pressure'] = 0
            # [代码新增结束]
            # --- 3. `retail_capitulation_score` (散户投降分) ---
            minute_data_for_day['price_return_5min'] = minute_data_for_day['minute_vwap'].pct_change(5)
            panic_minutes = minute_data_for_day[minute_data_for_day['price_return_5min'] < -0.015]
            if not panic_minutes.empty:
                panic_sell_vol = panic_minutes['retail_sell_vol'].sum()
                total_retail_sell = minute_data_for_day['retail_sell_vol'].sum()
                day_results['retail_capitulation_score'] = panic_sell_vol / total_retail_sell if total_retail_sell else np.nan
            else:
                day_results['retail_capitulation_score'] = 0
            # --- 4. `intraday_execution_alpha` (日内执行Alpha) ---
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
        """
        【新增】计算“日内交易结构”指标，为每个交易日绘制画像
        """
        if minute_df_grouped is None:
            print("调试信息: 分钟数据为空，无法计算日内交易结构指标。")
            return pd.DataFrame(index=daily_df.index)
        results = {}
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = minute_df_grouped.loc[[date_key]]
            day_results = {'trade_time': date}
            # --- 1. 日内波动率 (intraday_volatility) ---
            # 计算分钟收益率的标准差，作为日内波动性的度量
            minute_returns = minute_data_for_day['minute_vwap'].pct_change().dropna()
            day_results['intraday_volatility'] = minute_returns.std() if not minute_returns.empty else 0
            # --- 2. 收盘强度指数 (closing_strength_index) ---
            intraday_high = minute_data_for_day['minute_vwap'].max()
            intraday_low = minute_data_for_day['minute_vwap'].min()
            close_price = daily_data.get('close')
            price_range = intraday_high - intraday_low
            if pd.notna(close_price) and price_range > 0:
                day_results['closing_strength_index'] = (close_price - intraday_low) / price_range
            else:
                day_results['closing_strength_index'] = np.nan
            # --- 3. 收盘价与VWAP偏离度 (close_vs_vwap_ratio) ---
            daily_vwap = daily_data.get('daily_vwap')
            if pd.notna(close_price) and pd.notna(daily_vwap) and daily_vwap > 0:
                day_results['close_vs_vwap_ratio'] = (close_price / daily_vwap) - 1
            else:
                day_results['close_vs_vwap_ratio'] = np.nan
            # --- 4. 尾盘动能 (final_hour_momentum) ---
            final_hour_mask = minute_data_for_day['trade_time'].dt.hour >= 14
            final_hour_vol = minute_data_for_day[final_hour_mask]['vol_shares'].sum()
            total_day_vol = minute_data_for_day['vol_shares'].sum()
            if total_day_vol > 0:
                # 计算尾盘成交量占全天成交量的比例
                day_results['final_hour_momentum'] = final_hour_vol / total_day_vol
            else:
                day_results['final_hour_momentum'] = np.nan
            results[date] = day_results
        if not results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')








