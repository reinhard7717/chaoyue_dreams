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
from stock_models.fund_flow import BaseAdvancedFundFlowMetrics
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
        """【V1.3 · 分块处理核心实现】服务层主执行器"""
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental
        )
        # 引入分块处理逻辑
        total_processed_count = 0
        # 如果是增量更新，则直接执行一次计算即可
        if is_incremental_final:
            print(f"调试信息: {stock_code} 启动增量计算，起始日期: {fetch_start_date}")
            merged_df = await self._load_and_merge_sources(stock_info, fetch_start_date)
            if not merged_df.empty:
                daily_vwap_series = await self._calculate_daily_vwap(stock_info, merged_df.index)
                merged_df['daily_vwap'] = daily_vwap_series
                self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, merged_df.index)
                df_with_advanced_metrics = self._synthesize_and_forge_metrics(stock_code, merged_df)
                final_metrics_df = self._calculate_derivatives(stock_code, df_with_advanced_metrics)
                total_processed_count = await self._prepare_and_save_data(
                    stock_info, MetricsModel, final_metrics_df, is_incremental_final, last_metric_date
                )
                if hasattr(self, '_minute_df_daily_grouped'):
                    del self._minute_df_daily_grouped
        else:
            # 全量计算，启动分块处理循环
            print(f"调试信息: {stock_code} 启动全量分块计算...")
            # 首次运行时，清空旧数据
            await sync_to_async(MetricsModel.objects.filter(stock=stock_info).delete)()
            # 获取所有日线数据的交易日历作为循环基准
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_trade_dates = pd.to_datetime(
                await sync_to_async(list)(DailyModel.objects.filter(stock=stock_info).values_list('trade_time', flat=True).order_by('trade_time'))
            )
            if all_trade_dates.empty:
                print(f"调试信息: {stock_code} 无日线数据，无法进行计算。")
                return 0
            chunk_size = 180  # 每个区块处理180天的数据
            for i in range(0, len(all_trade_dates), chunk_size):
                chunk_dates = all_trade_dates[i:i + chunk_size]
                if chunk_dates.empty:
                    continue
                # 为确保衍生指标计算的连续性，需要向前加载一个缓冲期的数据
                chunk_start_date = chunk_dates.min().date()
                buffer_start_date = chunk_start_date - timedelta(days=self.max_lookback_days)
                chunk_end_date = chunk_dates.max().date()
                print(f"--- 正在处理区块 {i//chunk_size + 1}，日期范围: {chunk_start_date} to {chunk_end_date} ---")
                merged_df = await self._load_and_merge_sources(stock_info, buffer_start_date)
                if merged_df.empty:
                    continue
                # 仅为当前区块的日期计算高级指标
                target_df = merged_df[merged_df.index.date >= chunk_start_date]
                if target_df.empty:
                    continue
                daily_vwap_series = await self._calculate_daily_vwap(stock_info, target_df.index)
                target_df['daily_vwap'] = daily_vwap_series
                self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, target_df.index)
                df_with_advanced_metrics = self._synthesize_and_forge_metrics(stock_code, target_df)
                # 衍生指标计算需要用到缓冲期的数据，所以在完整的merged_df上计算
                full_df_with_metrics = merged_df.join(df_with_advanced_metrics, rsuffix='_calc')
                final_metrics_df = self._calculate_derivatives(stock_code, full_df_with_metrics)
                # 只保存属于当前区块的计算结果
                chunk_to_save = final_metrics_df[final_metrics_df.index.date >= chunk_start_date]
                processed_count = await self._prepare_and_save_data(
                    stock_info, MetricsModel, chunk_to_save, is_incremental=True, last_metric_date=chunk_start_date - timedelta(days=1)
                )
                total_processed_count += processed_count
                if hasattr(self, '_minute_df_daily_grouped'):
                    del self._minute_df_daily_grouped
        return total_processed_count
        

    async def _initialize_context(self, stock_code: str, is_incremental: bool):
        """【V1.1 · 分块处理版】初始化任务上下文，为分块处理做准备"""
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        last_metric_date = None
        # 调整逻辑以支持分块处理
        # 在增量模式下，我们只计算最近的一小部分数据，无需分块
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
                # 增量更新时，计算的起始日期为最后记录日期减去衍生指标最大回溯期
                fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
            else:
                # 如果没有历史数据，则转为全量计算
                is_incremental = False
                fetch_start_date = None # 全量计算从头开始
        else:
            # 全量计算模式，起始日期设为None，由主循环处理
            fetch_start_date = None
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

    async def _calculate_daily_vwap(self, stock_info: StockInfo, date_index: pd.DatetimeIndex) -> pd.Series:
        """【V1.2 · 通信协议升级版】从分钟数据计算日度VWAP"""
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        if not MinuteModel or date_index.empty:
            print(f"调试信息: 未能为 {stock_info.stock_code} 找到1分钟线模型，VWAP计算将跳过。")
            return pd.Series(np.nan, index=date_index)
        @sync_to_async(thread_sensitive=True)
        def get_minute_data(model, stock_code_pk, start_date, end_date):
            # 使用主键(stock_id)进行查询，避免跨线程传递复杂对象
            qs = model.objects.filter(
                stock_id=stock_code_pk,
                trade_time__date__gte=start_date,
                trade_time__date__lte=end_date
            ).values('trade_time', 'amount', 'vol')
            
            return pd.DataFrame.from_records(qs)
        min_date, max_date = date_index.min().date(), date_index.max().date()
        # 传递主键 stock_info.stock_code 而非整个 stock_info 对象
        minute_df = await get_minute_data(MinuteModel, stock_info.stock_code, min_date, max_date)
        
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
        """【V2.1 · 德尔菲神谕引擎植入版】"""
        df = merged_df.copy()
        tushare_cols_exist = 'buy_sm_vol' in df.columns
        if tushare_cols_exist:
            minute_df_daily_grouped = getattr(self, '_minute_df_daily_grouped', None)
            if minute_df_daily_grouped is None or minute_df_daily_grouped.empty:
                print(f"调试信息: {stock_code} 在 {df.index.min().date()} 到 {df.index.max().date()} 期间分钟数据未预加载或为空，跳过PVWAP及所有基于分钟线的高级指标计算。")
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
                df['avg_cost_main_buy'] = (df['buy_lg_amount'] * 10000 + df['buy_elg_amount'] * 10000) / (df['buy_lg_vol'] * 100 + df['buy_elg_vol'] * 100).replace(0, np.nan)
                df['avg_cost_main_sell'] = (df['sell_lg_amount'] * 10000 + df['sell_elg_amount'] * 10000) / (df['sell_lg_vol'] * 100 + df['sell_elg_vol'] * 100).replace(0, np.nan)
                df['avg_cost_retail_buy'] = (df['buy_sm_amount'] * 10000 + df['buy_md_amount'] * 10000) / (df['buy_sm_vol'] * 100 + df['buy_md_vol'] * 100).replace(0, np.nan)
                df['avg_cost_retail_sell'] = (df['sell_sm_amount'] * 10000 + df['sell_md_amount'] * 10000) / (df['sell_sm_vol'] * 100 + df['sell_md_vol'] * 100).replace(0, np.nan)
            else:
                print(f"调试信息: {stock_code} 分钟数据加载成功，开始计算所有高级指标。")
                pvwap_costs_df = self._calculate_probabilistic_costs(df, minute_df_daily_grouped)
                df = df.join(pvwap_costs_df)
                pnl_matrix_df = self._upgrade_intraday_profit_metric(df)
                df = df.join(pnl_matrix_df)
                behavioral_metrics_df = self._upgrade_behavioral_metrics(df, minute_df_daily_grouped)
                df = df.join(behavioral_metrics_df)
                structure_metrics_df = self._calculate_intraday_structure_metrics(df, minute_df_daily_grouped)
                df = df.join(structure_metrics_df)
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
        # 植入“德尔菲神谕”加权与分歧分析引擎
        source_cols = ['main_force_net_flow_tushare', 'main_force_net_flow_ths', 'main_force_net_flow_dc']
        existing_sources = [col for col in source_cols if col in df.columns]
        if len(existing_sources) > 1:
            # 1. 计算多源分歧标准差
            df['cross_source_divergence_std'] = df[existing_sources].std(axis=1)
            # 2. 计算加权共识主力净流入
            flows = df[existing_sources]
            median_flow = flows.median(axis=1)
            # 计算每个源与中位数的绝对偏差
            deviations = flows.sub(median_flow, axis=0).abs()
            # 计算权重：偏差越小，权重越高。使用 1 / (1 + deviation) 的形式避免除以零
            weights = 1 / (1 + deviations)
            # 计算加权平均值
            weighted_flows = flows.multiply(weights.values)
            df['consensus_flow_weighted'] = weighted_flows.sum(axis=1) / weights.sum(axis=1).replace(0, np.nan)
        else:
            df['cross_source_divergence_std'] = np.nan
            df['consensus_flow_weighted'] = df.get('main_force_net_flow_consensus', np.nan) # 若只有一个源，则加权共识等于其自身
        
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
        """【V3.0 · SSOT原则重构版】计算衍生指标，并动态读取模型的排除列表。"""
        final_df = consensus_df.copy()
        # 直接从模型类读取“生产管制清单”，不再硬编码
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        # 同时，动态读取所有核心指标，使衍生计算更具扩展性
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        
        sum_cols = [
            'net_flow_consensus', 'main_force_net_flow_consensus', 'retail_net_flow_consensus',
            'net_xl_amount_consensus', 'net_lg_amount_consensus', 'net_md_amount_consensus',
            'net_sh_amount_consensus', 'cost_weighted_main_flow',
            'consensus_calibrated_main_flow', 'consensus_flow_weighted',
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
                base_col_name = col.split('_sum_')[0] if '_sum_' in col else col
                if base_col_name in SLOPE_ACCEL_EXCLUSIONS:
                    continue
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

    # 重构为标准的 async 方法，并将ORM操作封装在内联函数中
    async def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex):
        """【V1.2 · 通信协议升级与重构版】获取并按日聚合分钟数据"""
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        if not MinuteModel:
            print(f"调试信息: {stock_info.stock_code} 未能找到对应的1分钟线数据模型。")
            return None
        if date_index.empty:
            print(f"调试信息: {stock_info.stock_code} 的日期索引为空，无法查询分钟数据。")
            return None
        @sync_to_async(thread_sensitive=True)
        def get_data(model, stock_code_pk, start_date, end_date):
            # 使用主键(stock_id)进行查询，这是跨线程通信的最佳实践
            qs = model.objects.filter(
                stock_id=stock_code_pk,
                trade_time__date__gte=start_date,
                trade_time__date__lte=end_date
            ).values('trade_time', 'amount', 'vol')
            return pd.DataFrame.from_records(qs)
        min_date, max_date = date_index.min().date(), date_index.max().date()
        # 传递主键 stock_info.stock_code 而非整个 stock_info 对象
        minute_df = await get_data(MinuteModel, stock_info.stock_code, min_date, max_date)
        if minute_df.empty:
            print(f"调试信息: {stock_info.stock_code} 在 {min_date} 到 {max_date} 期间的数据库查询结果为空。")
            return None
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
        【V2.0 · 奥丁之眼重构版】使用订单规模似然权重计算PVWAP成本
        """
        if minute_df_grouped is None:
            print("调试信息: 分钟数据为空，无法计算PVWAP成本及算法指纹。")
            # 如果没有分钟数据，则不进行任何计算，返回空DataFrame
            return pd.DataFrame(index=daily_df.index)
        results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        from scipy.spatial.distance import jensenshannon
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = minute_df_grouped.loc[[date_key]].copy()
            # 调用新的权重计算方法
            minute_data_for_day = self._calculate_order_size_likelihood_weights(minute_data_for_day, daily_data)
            
            day_results = {'trade_time': date}
            # --- 1. 计算PVWAP成本 ---
            for cost_type in cost_types:
                size = cost_type.split('_')[0] # sm, md, lg, elg
                daily_vol_shares = pd.to_numeric(daily_data.get(f'{cost_type}_vol'), errors='coerce') * 100
                if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                    day_results[f'avg_cost_{cost_type}'] = np.nan
                    # 同时记录归因后的分钟成交量，供下游使用
                    minute_data_for_day[f'{cost_type}_vol_attr'] = 0
                    
                    continue
                # 使用新的、分规模的权重进行归因
                weight_col = f'{size}_weight'
                attributed_vol = minute_data_for_day[weight_col] * daily_vol_shares
                minute_data_for_day[f'{cost_type}_vol_attr'] = attributed_vol # 记录归因后的分钟成交量
                
                attributed_value = attributed_vol * minute_data_for_day['minute_vwap']
                total_attributed_value = attributed_value.sum()
                total_attributed_vol = attributed_vol.sum()
                day_results[f'avg_cost_{cost_type}'] = total_attributed_value / total_attributed_vol if total_attributed_vol > 0 and not np.isclose(total_attributed_vol, 0) else np.nan
            # --- 2. 计算算法交易指纹 (逻辑不变，但基于更精确的分钟数据) ---
            p_dist = minute_data_for_day['vol_shares'].fillna(0).values / minute_data_for_day['vol_shares'].sum() if minute_data_for_day['vol_shares'].sum() > 0 else np.zeros(len(minute_data_for_day))
            q_dist = np.full_like(p_dist, 1.0 / len(p_dist)) if len(p_dist) > 0 else np.array([])
            day_results['volume_profile_jsd_vs_uniform'] = jensenshannon(p_dist, q_dist)**2 if p_dist.size > 0 and q_dist.size > 0 else np.nan
            first_hour_mask = (minute_data_for_day['trade_time'].dt.hour == 9) & (minute_data_for_day['trade_time'].dt.minute >= 30) | \
                              (minute_data_for_day['trade_time'].dt.hour == 10) & (minute_data_for_day['trade_time'].dt.minute < 30)
            first_hour_vol = minute_data_for_day[first_hour_mask]['vol_shares'].sum()
            total_day_vol = minute_data_for_day['vol_shares'].sum()
            day_results['aggression_index_opening'] = first_hour_vol / total_day_vol if total_day_vol else np.nan
            # 将带有归因成交量的分钟数据存入day_results，供下游使用
            day_results['minute_data_attributed'] = minute_data_for_day
            
            results[date] = day_results
        if not results:
            return pd.DataFrame()
        # 调整数据处理流程，以传递分钟数据给下游
        final_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        # 提取分钟数据以供下游使用
        self._minute_df_attributed_daily_grouped = {date: res.pop('minute_data_attributed') for date, res in results.items()}
        final_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        
        final_df = self._calculate_aggregate_pvwap_costs(final_df, daily_df)
        if 'avg_cost_main_buy' in final_df.columns and 'daily_vwap' in daily_df.columns:
            final_df = final_df.join(daily_df['daily_vwap'])
            final_df['vwap_tracking_error'] = final_df['avg_cost_main_buy'] - final_df['daily_vwap']
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
        # 使用新的加权函数计算聚合成本
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
        【V2.0 · 奥丁之眼适配版】使用精确归因后的分钟资金流计算战术行为指标
        """
        # 检查新的、带有归因数据的分钟数据是否存在
        minute_df_attributed_grouped = getattr(self, '_minute_df_attributed_daily_grouped', None)
        if minute_df_attributed_grouped is None:
            print("调试信息: 精确归因后的分钟数据为空，无法升维战术行为指标。")
            return pd.DataFrame(index=daily_df.index)
        
        results = {}
        for date, daily_data in daily_df.iterrows():
            # 从新的数据源获取带有归因成交量的分钟数据
            if date not in minute_df_attributed_grouped:
                continue
            minute_data_for_day = minute_df_attributed_grouped[date]
            
            day_results = {'trade_time': date}
            # --- 准备分钟级归属资金流数据 ---
            # 直接使用已归因的成交量数据，不再需要vol_weight
            minute_data_for_day['main_force_buy_vol'] = minute_data_for_day['lg_buy_vol_attr'] + minute_data_for_day['elg_buy_vol_attr']
            minute_data_for_day['main_force_sell_vol'] = minute_data_for_day['lg_sell_vol_attr'] + minute_data_for_day['elg_sell_vol_attr']
            minute_data_for_day['main_force_net_vol'] = minute_data_for_day['main_force_buy_vol'] - minute_data_for_day['main_force_sell_vol']
            minute_data_for_day['retail_buy_vol'] = minute_data_for_day['sm_buy_vol_attr'] + minute_data_for_day['md_buy_vol_attr']
            minute_data_for_day['retail_sell_vol'] = minute_data_for_day['sm_sell_vol_attr'] + minute_data_for_day['md_sell_vol_attr']
            minute_data_for_day['retail_net_vol'] = minute_data_for_day['retail_buy_vol'] - minute_data_for_day['retail_sell_vol']
            
            # --- 1. `main_force_support_strength` (主力支撑强度) ---
            low_threshold = minute_data_for_day['minute_vwap'].quantile(0.1)
            bottom_zone_minutes = minute_data_for_day[minute_data_for_day['minute_vwap'] <= low_threshold]
            if not bottom_zone_minutes.empty:
                support_net_flow = bottom_zone_minutes['main_force_net_vol'].sum()
                total_main_buy = minute_data_for_day['main_force_buy_vol'].sum()
                day_results['main_force_support_strength'] = support_net_flow / total_main_buy if total_main_buy else np.nan
            else:
                day_results['main_force_support_strength'] = 0
            # --- 2. `main_force_distribution_pressure` (主力派发压力) ---
            high_threshold = minute_data_for_day['minute_vwap'].quantile(0.9)
            top_zone_minutes = minute_data_for_day[minute_data_for_day['minute_vwap'] >= high_threshold]
            if not top_zone_minutes.empty:
                distribution_net_flow = top_zone_minutes['main_force_net_vol'].sum()
                total_main_sell = minute_data_for_day['main_force_sell_vol'].sum()
                day_results['main_force_distribution_pressure'] = -distribution_net_flow / total_main_sell if total_main_sell else np.nan
            else:
                day_results['main_force_distribution_pressure'] = 0
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

    def _calculate_order_size_likelihood_weights(self, minute_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """【新增】奥丁之眼算法核心：计算订单规模似然权重"""
        df = minute_data_for_day.copy()
        circ_mv = pd.to_numeric(daily_data.get('circ_mv'), errors='coerce') * 10000 # 转换为元
        if pd.isna(circ_mv) or circ_mv == 0:
            # 如果没有流通市值，退回到旧的、基于成交量的权重
            total_day_vol = df['vol_shares'].sum()
            df['sm_weight'] = df['md_weight'] = df['lg_weight'] = df['elg_weight'] = df['vol_shares'] / total_day_vol if total_day_vol > 0 else 0
            return df
        # 1. 定义动态绝对阈值
        elg_threshold = circ_mv * 0.001  # 特大单门槛: 流通市值的千分之一 (e.g., 100亿 -> 1000万)
        lg_threshold = circ_mv * 0.0002  # 大单门槛: 流通市值的万分之二 (e.g., 100亿 -> 200万)
        md_threshold = circ_mv * 0.00005 # 中单门槛: 流通市值的十万分之五 (e.g., 100亿 -> 50万)
        # 2. 计算各规模的似然分数 (Likelihood Score)
        # 核心逻辑：只有当分钟成交额达到相应门槛，才认为它可能包含该规模的订单
        df['elg_score'] = df['amount_yuan'].where(df['amount_yuan'] >= elg_threshold, 0)
        df['lg_score'] = df['amount_yuan'].where((df['amount_yuan'] >= lg_threshold) & (df['amount_yuan'] < elg_threshold), 0)
        df['md_score'] = df['amount_yuan'].where((df['amount_yuan'] >= md_threshold) & (df['amount_yuan'] < lg_threshold), 0)
        df['sm_score'] = df['amount_yuan'].where(df['amount_yuan'] < md_threshold, 0)
        # 3. 归一化似然分数，生成最终的归因权重
        for size in ['sm', 'md', 'lg', 'elg']:
            total_score = df[f'{size}_score'].sum()
            df[f'{size}_weight'] = df[f'{size}_score'] / total_score if total_score > 0 else 0
        return df







