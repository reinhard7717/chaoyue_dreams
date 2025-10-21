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
        self.max_lookback_days = 300

    def _get_safe_numeric_series(self, df: pd.DataFrame, col_name: str, default_value=0) -> pd.Series:
        """
        【V2.0 · 单行兼容版】类型安全的列获取辅助函数。
        修正了对单行DataFrame处理时返回标量导致后续链式调用失败的BUG。
        """
        # [代码修改开始] 彻底修正单行DataFrame问题
        if col_name not in df.columns:
            # 如果列不存在，创建一个填充了默认值的Series
            return pd.Series(default_value, index=df.index, dtype=float)
        # 使用 df[col_name] 保证返回的是Series，而不是标量，从根本上解决问题
        series = df[col_name]
        # 先转换为数值类型，再填充NaN
        return pd.to_numeric(series, errors='coerce').fillna(default_value)
        # [代码修改结束]

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
        """【V6.0 · 回滚式增量引擎版】服务层主执行器"""
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        total_processed_count = 0
        # [代码修改开始] 重新定义增量模式下的作战范围和前置操作
        if not is_incremental_final: # 全量计算模式
            print(f"调试信息: [{stock_code}] 启动全量计算模式。")
            await sync_to_async(MetricsModel.objects.filter(stock=stock_info).delete)()
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        else: # 增量或部分全量模式
            mode = "部分全量" if start_date_str else "增量"
            # 确定回滚删除的起点
            rollback_start_date = fetch_start_date if fetch_start_date else start_date_str
            if rollback_start_date:
                deleted_count, _ = await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=rollback_start_date).delete)()
                print(f"调试信息: [{stock_code}] {mode}模式，已回滚删除从 {rollback_start_date} 开始的 {deleted_count} 条旧指标数据。")
            else:
                # 如果是首次全量计算，则无需回滚
                print(f"调试信息: [{stock_code}] 首次计算，无需回滚。")
            # 增量模式下，需要重新计算的日期是从 fetch_start_date 开始的所有日期
            DailyModel = get_daily_data_model_by_code(stock_code)
            all_dates_qs = DailyModel.objects.filter(stock=stock_info, trade_time__gte=fetch_start_date).values_list('trade_time', flat=True).order_by('trade_time')
            dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            print(f"调试信息: [{stock_code}] 无需计算的日期，任务终止。")
            return 0
        # 步骤 2: 初始化【所有】历史指标作为基线上下文
        initial_history_end_date = dates_to_process.min()
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, initial_history_end_date)
        # 步骤 3: 【核心指标锻造循环】，分块处理，保证低内存
        CHUNK_SIZE = 50
        all_new_core_metrics_df = pd.DataFrame()
        for i in range(0, len(dates_to_process), CHUNK_SIZE):
            chunk_dates = dates_to_process[i:i + CHUNK_SIZE]
            if chunk_dates.empty:
                continue
            chunk_start_date, chunk_end_date = chunk_dates.min(), chunk_dates.max()
            print(f"--- 正在锻造核心指标 区块 {i//CHUNK_SIZE + 1}，日期范围: {chunk_start_date.date()} to {chunk_end_date.date()} ---")
            chunk_raw_data_df = await self._load_and_merge_sources(stock_info, start_date=chunk_start_date, end_date=chunk_end_date)
            if chunk_raw_data_df.empty:
                print(f"警告: 区块 {chunk_start_date.date()} to {chunk_end_date.date()} 原始数据为空，跳过。")
                continue
            daily_vwap_series = await self._calculate_daily_vwap(stock_info, chunk_raw_data_df.index)
            self._minute_df_daily_grouped = await self._get_daily_grouped_minute_data(stock_info, chunk_raw_data_df.index)
            chunk_new_metrics_df = self._synthesize_and_forge_metrics(stock_code, chunk_raw_data_df, daily_vwap_series)
            all_new_core_metrics_df = pd.concat([all_new_core_metrics_df, chunk_new_metrics_df])
        if hasattr(self, '_minute_df_daily_grouped'):
            del self._minute_df_daily_grouped
        if all_new_core_metrics_df.empty:
            print(f"调试信息: [{stock_code}] 所有区块均未能计算出核心指标，任务终止。")
            return 0
        # 步骤 4: 【衍生指标升维】，在全局视角下一次性完成
        print(f"--- 核心指标锻造完毕，共 {len(all_new_core_metrics_df)} 条。开始全局衍生指标升维... ---")
        full_sequence_for_derivatives = pd.concat([historical_metrics_df, all_new_core_metrics_df])
        full_sequence_for_derivatives.sort_index(inplace=True)
        final_metrics_df = self._calculate_derivatives(stock_code, full_sequence_for_derivatives)
        # 步骤 5: 精确切分出本次任务计算的最终结果并保存
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(all_new_core_metrics_df.index)]
        total_processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        return total_processed_count
        # [代码修改结束]

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        """【V2.0 · 回滚式增量定义版】初始化任务上下文。"""
        from datetime import datetime
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_fund_flow_metrics_model_by_code(stock_code)
        last_metric_date = None
        fetch_start_date = None
        # [代码修改开始] 修正增量和部分全量模式的 fetch_start_date 定义
        if start_date_str:
            print(f"调试信息: [{stock_code}] 检测到资金流起始日期覆盖: {start_date_str}，将执行部分全量计算。")
            try:
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                is_incremental = True
                last_metric_date = start_date_obj - timedelta(days=1)
                # 数据拉取起点，需要回溯足够天数以计算最长周期的二阶导数
                # 55日加速度需要大约 55(斜率) + 55(加速度) = 110个交易日
                # 200天大约是140个交易日，提供足够缓冲
                fetch_start_date = start_date_obj - timedelta(days=self.max_lookback_days)
            except (ValueError, TypeError):
                print(f"警告: [{stock_code}] 提供的起始日期 '{start_date_str}' 格式错误，将忽略并执行默认逻辑。")
                is_incremental = True # 强制走默认增量
        if is_incremental and not start_date_str: # 仅在纯增量模式下查询最新日期
            @sync_to_async(thread_sensitive=True)
            def get_latest_metric_async(model, stock_info_obj):
                try:
                    return model.objects.filter(stock=stock_info_obj).latest('trade_time')
                except model.DoesNotExist:
                    return None
            latest_metric = await get_latest_metric_async(MetricsModel, stock_info)
            if latest_metric:
                last_metric_date = latest_metric.trade_time
                # 同样，回溯足够天数以修复可能被污染的历史衍生指标
                fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
            else:
                # 数据库为空，转为全量计算
                is_incremental = False
                fetch_start_date = None
        # 如果是全量计算，fetch_start_date 为 None，将加载所有数据
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date
        # [代码修改结束]

    async def _load_and_merge_sources(self, stock_info, start_date=None, end_date=None):
        """【V1.4 · 精确范围查询版】加载、标准化并合并多源数据"""
        @sync_to_async(thread_sensitive=True)
        def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_date=None, end_date=None):
            # [代码修改开始] 增加 end_date 过滤
            if not model: return pd.DataFrame()
            qs = model.objects.filter(stock=stock_info_obj)
            if start_date:
                qs = qs.filter(**{f'{date_field}__gte': start_date})
            if end_date:
                qs = qs.filter(**{f'{date_field}__lte': end_date})
            return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
            # [代码修改结束]
        
        data_tasks = {
            "tushare": get_data_async(get_fund_flow_model_by_code(stock_info.stock_code), stock_info, start_date=start_date, end_date=end_date),
            "ths": get_data_async(get_fund_flow_ths_model_by_code(stock_info.stock_code), stock_info, start_date=start_date, end_date=end_date),
            "dc": get_data_async(get_fund_flow_dc_model_by_code(stock_info.stock_code), stock_info, start_date=start_date, end_date=end_date),
            "daily": get_data_async(get_daily_data_model_by_code(stock_info.stock_code), stock_info, fields=('trade_time', 'amount', 'close'), start_date=start_date, end_date=end_date),
            "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'circ_mv', 'turnover_rate'), start_date=start_date, end_date=end_date),
        }
        results = await asyncio.gather(*data_tasks.values())
        data_dfs = dict(zip(data_tasks.keys(), results))
        
        # ... 后续的 standardize_and_prepare 和 merge 逻辑保持不变 ...
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
            # 在区块模式下，单个区块没有数据是正常的，不应抛出异常
            return pd.DataFrame()
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
        """【V1.3 · 时区修正版】从分钟数据计算日度VWAP"""
        # [代码修改开始] 修正时区查询BUG
        minute_df = await self._get_daily_grouped_minute_data(stock_info, date_index, fetch_full_cols=False)
        if minute_df is None or minute_df.empty:
            return pd.Series(np.nan, index=date_index)
        # 使用新的、更可靠的辅助函数进行计算
        return self._calculate_daily_vwap_from_df(minute_df, date_index)
        # [代码修改结束]

    async def _get_daily_grouped_minute_data(self, stock_info: StockInfo, date_index: pd.DatetimeIndex, fetch_full_cols: bool = True):
        """【V1.3 · 时区修正与重构版】获取并按日聚合分钟数据"""
        # [代码修改开始] 修正时区查询BUG
        from django.utils import timezone
        from datetime import datetime, time
        MinuteModel = get_minute_data_model_by_code_and_timelevel(stock_info.stock_code, '1')
        if not MinuteModel:
            print(f"调试信息: {stock_info.stock_code} 未能找到对应的1分钟线数据模型。")
            return None
        if date_index.empty:
            print(f"调试信息: {stock_info.stock_code} 的日期索引为空，无法查询分钟数据。")
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
            print(f"调试信息: {stock_info.stock_code} 在 {min_date} 到 {max_date} 期间的数据库查询结果为空。")
            return None
        # 使用新的、更可靠的辅助函数进行分组
        return self._group_minute_data_from_df(minute_df)
        # [代码修改结束]

    def _synthesize_and_forge_metrics(self, stock_code: str, merged_df: pd.DataFrame, daily_vwap_series: pd.Series) -> pd.DataFrame:
        """【V2.6 · 成本指标重构版】重组成本衍生指标计算逻辑，并修正概念错误。"""
        df = merged_df.copy()
        df['daily_vwap'] = daily_vwap_series
        print(f"调试信息: [{stock_code}] 进入指标合成引擎，传入数据形状: {df.shape}, 列: {df.columns.tolist()}")
        final_cols = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        result_df = pd.DataFrame(index=df.index, columns=final_cols)
        result_df.update(df)
        # [代码修改开始] 重构成本指标及其衍生指标的计算流程
        # 模块一：分钟数据依赖的高级指标计算 (包含所有成本相关指标)
        minute_df_daily_grouped = getattr(self, '_minute_df_daily_grouped', None)
        if minute_df_daily_grouped is not None and not minute_df_daily_grouped.empty and 'buy_sm_vol' in df.columns:
            # print(f"调试信息: [{stock_code}] 分钟数据和成交量数据满足，开始计算所有高级指标。")
            # 1. 计算核心的PVWAP成本
            pvwap_costs_df = self._calculate_probabilistic_costs(df, minute_df_daily_grouped)
            result_df.update(pvwap_costs_df)
            # 2. 【结构重组】紧接着计算所有依赖于PVWAP成本的衍生指标
            # 确保只有在成本计算成功后才执行
            if 'avg_cost_main_buy' in result_df.columns:
                # 2.1 成本分歧度 (主力买 vs 散户卖)
                result_df['cost_divergence_mf_vs_retail'] = result_df['avg_cost_main_buy'] - result_df.get('avg_cost_retail_sell', np.nan)
                # 2.2 【概念重塑】成本加权主力资金流 (使用成交量而非成交额)
                main_force_net_vol = self._get_safe_numeric_series(df, 'buy_lg_vol') + self._get_safe_numeric_series(df, 'buy_elg_vol') - self._get_safe_numeric_series(df, 'sell_lg_vol') - self._get_safe_numeric_series(df, 'sell_elg_vol')
                main_force_net_vol_shares = main_force_net_vol * 100 # 从“手”转换为“股”
                result_df['cost_weighted_main_flow'] = main_force_net_vol_shares * result_df['avg_cost_main_buy']
                # 2.3 主力成本优势 (vs 收盘价)
                result_df['main_buy_cost_advantage'] = np.divide(result_df['avg_cost_main_buy'], df['close'], out=np.full_like(df['close'].values, np.nan, dtype=float), where=df['close']!=0) - 1
                # 2.4 市场成本博弈 (主力买 vs 散户买)
                result_df['market_cost_battle'] = result_df['avg_cost_main_buy'] - result_df.get('avg_cost_retail_buy', np.nan)
                # 2.5 主力成本与VWAP的偏离
                if 'daily_vwap' in result_df.columns:
                    result_df['main_buy_cost_vs_vwap'] = result_df['avg_cost_main_buy'] - result_df['daily_vwap']
                    result_df['main_sell_cost_vs_vwap'] = result_df.get('avg_cost_main_sell', np.nan) - result_df['daily_vwap']
            # 3. 计算P&L矩阵 (它也依赖成本数据)
            pnl_matrix_df = self._upgrade_intraday_profit_metric(result_df)
            result_df.update(pnl_matrix_df)
            # 4. 计算其他行为和结构指标
            behavioral_metrics_df = self._upgrade_behavioral_metrics(result_df, minute_df_daily_grouped)
            result_df.update(behavioral_metrics_df)
            structure_metrics_df = self._calculate_intraday_structure_metrics(result_df, minute_df_daily_grouped)
            result_df.update(structure_metrics_df)
        else:
            print(f"警告: [{stock_code}] 在 {df.index.min().date()} 到 {df.index.max().date()} 期间分钟数据或成交量数据缺失，跳过大部分高级指标计算。")
        # [代码修改结束]
        # 模块二：共识指标计算 (填充result_df)
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
                result_df[target_col] = df[existing_cols].mean(axis=1)
        # 模块三：衍生共识与分歧度指标 (填充result_df)
        source_cols = ['main_force_net_flow_tushare', 'main_force_net_flow_ths', 'main_force_net_flow_dc']
        existing_sources = [col for col in source_cols if col in df.columns]
        if len(existing_sources) > 1:
            flows = df[existing_sources]
            result_df['cross_source_divergence_std'] = flows.std(axis=1)
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
        # 模块四：最终比率指标 (填充result_df)
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
        print(f"调试信息: [{stock_code}] 指标合成引擎执行完毕，返回数据形状: {result_df.shape}")
        return result_df

    def _calculate_probabilistic_costs(self, daily_df: pd.DataFrame, minute_df_grouped: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.1 · 方向性权重应用版】使用订单规模似然权重计算PVWAP成本。
        - 核心修正: 使用上游生成的、带有方向的score来构建权重，确保买卖成本的差异性。
        """
        if minute_df_grouped is None:
            print("调试信息: 分钟数据为空，无法计算PVWAP成本及算法指纹。")
            return pd.DataFrame(index=daily_df.index)
        results = {}
        cost_types = ['sm_buy', 'sm_sell', 'md_buy', 'md_sell', 'lg_buy', 'lg_sell', 'elg_buy', 'elg_sell']
        from scipy.spatial.distance import jensenshannon
        for date, daily_data in daily_df.iterrows():
            date_key = date.date()
            if date_key not in minute_df_grouped.index:
                continue
            minute_data_for_day = self._calculate_order_size_likelihood_scores(minute_df_grouped.loc[[date_key]].copy(), daily_data)
            day_results = {'trade_time': date}
            for cost_type in cost_types:
                size, direction = cost_type.split('_')
                db_vol_key = f'{direction}_{size}_vol'
                daily_vol_shares = pd.to_numeric(daily_data.get(db_vol_key), errors='coerce') * 100
                if pd.isna(daily_vol_shares) or daily_vol_shares == 0:
                    day_results[f'avg_cost_{cost_type}'] = np.nan
                    minute_data_for_day[f'{cost_type}_vol_attr'] = 0
                    continue
                # [代码修改开始] 使用带有方向的、正确的score_col
                score_col = f'{size}_{direction}_score'
                # [代码修改结束]
                total_score = minute_data_for_day[score_col].sum()
                if total_score > 0:
                    weight_series = minute_data_for_day[score_col] / total_score
                else:
                    weight_series = minute_data_for_day['vol_weight']
                attributed_vol = weight_series * daily_vol_shares
                minute_data_for_day[f'{cost_type}_vol_attr'] = attributed_vol
                attributed_value = attributed_vol * minute_data_for_day['minute_vwap']
                total_attributed_value = attributed_value.sum()
                total_attributed_vol = attributed_vol.sum()
                day_results[f'avg_cost_{cost_type}'] = total_attributed_value / total_attributed_vol if total_attributed_vol > 0 else np.nan
            p_dist = minute_data_for_day['vol_shares'].fillna(0).values / minute_data_for_day['vol_shares'].sum() if minute_data_for_day['vol_shares'].sum() > 0 else np.zeros(len(minute_data_for_day))
            q_dist = np.full_like(p_dist, 1.0 / len(p_dist)) if len(p_dist) > 0 else np.array([])
            day_results['volume_profile_jsd_vs_uniform'] = jensenshannon(p_dist, q_dist)**2 if p_dist.size > 0 and q_dist.size > 0 else np.nan
            first_hour_mask = (minute_data_for_day['trade_time'].dt.hour == 9) & (minute_data_for_day['trade_time'].dt.minute >= 30) | \
                              (minute_data_for_day['trade_time'].dt.hour == 10) & (minute_data_for_day['trade_time'].dt.minute < 30)
            first_hour_vol = minute_data_for_day[first_hour_mask]['vol_shares'].sum()
            total_day_vol = minute_data_for_day['vol_shares'].sum()
            day_results['aggression_index_opening'] = first_hour_vol / total_day_vol if total_day_vol else np.nan
            day_results['minute_data_attributed'] = minute_data_for_day
            results[date] = day_results
        if not results:
            return pd.DataFrame()
        self._minute_df_attributed_daily_grouped = {date: res.pop('minute_data_attributed') for date, res in results.items() if 'minute_data_attributed' in res}
        pvwap_df = pd.DataFrame.from_dict(results, orient='index').set_index('trade_time')
        final_df = self._calculate_aggregate_pvwap_costs(pvwap_df, daily_df)
        return final_df

    def _calculate_aggregate_pvwap_costs(self, pvwap_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.5 · 数据净化版】确保在聚合成本时正确处理NaN值。
        """
        df = pvwap_df.copy()
        vol_cols = [
            'buy_sm_vol', 'sell_sm_vol', 'buy_md_vol', 'sell_md_vol',
            'buy_lg_vol', 'sell_lg_vol', 'buy_elg_vol', 'sell_elg_vol'
        ]
        existing_vol_cols = [col for col in vol_cols if col in daily_df.columns]
        if not existing_vol_cols:
            return df
        df = df.join(daily_df[existing_vol_cols])
        # [代码修改开始] 重构聚合逻辑，根除数据污染
        def weighted_avg_cost(cost_cols, vol_cols):
            """一个健壮的加权平均函数，能正确处理NaN"""
            numerator = pd.Series(0.0, index=df.index)
            denominator = pd.Series(0.0, index=df.index)
            for cost_col, vol_col in zip(cost_cols, vol_cols):
                if cost_col in df.columns and vol_col in df.columns:
                    cost = df[cost_col] # 直接获取Series，不填充NaN
                    volume = pd.to_numeric(df[vol_col], errors='coerce').fillna(0)
                    # 计算价值贡献，其中cost为NaN的部分，结果也是NaN，我们用0填充这部分贡献
                    value_contribution = (cost * volume).fillna(0)
                    numerator += value_contribution
                    # 关键：只在成本有效（非NaN）的地方累加成交量到分母
                    denominator += volume.where(cost.notna(), 0)
            return numerator / denominator.replace(0, np.nan)
        # [代码修改结束]
        df['avg_cost_main_buy'] = weighted_avg_cost(
            ['avg_cost_lg_buy', 'avg_cost_elg_buy'],
            ['buy_lg_vol', 'buy_elg_vol']
        )
        df['avg_cost_main_sell'] = weighted_avg_cost(
            ['avg_cost_lg_sell', 'avg_cost_elg_sell'],
            ['sell_lg_vol', 'sell_elg_vol']
        )
        df['avg_cost_retail_buy'] = weighted_avg_cost(
            ['avg_cost_sm_buy', 'avg_cost_md_buy'],
            ['buy_sm_vol', 'buy_md_vol']
        )
        df['avg_cost_retail_sell'] = weighted_avg_cost(
            ['avg_cost_sm_sell', 'avg_cost_md_sell'],
            ['sell_sm_vol', 'sell_md_vol']
        )
        if 'avg_cost_main_buy' in df.columns and 'daily_vwap' in daily_df.columns:
            df['vwap_tracking_error'] = df['avg_cost_main_buy'] - daily_df['daily_vwap']
        return df.drop(columns=existing_vol_cols, errors='ignore')

    def _calculate_derivatives(self, stock_code: str, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """【V3.3 · 强制净化版】在计算前强制转换数据类型，根除object污染。"""
        final_df = consensus_df.copy()
        import pandas_ta as ta
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedFundFlowMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedFundFlowMetrics.CORE_METRICS.keys())
        # [代码修改开始] 在衍生计算前，对所有目标列进行强制的、全面的数据类型净化
        for col in CORE_METRICS_TO_DERIVE:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        # [代码修改结束]
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
            min_p = max(2, int(p * 0.8))
            for col in sum_cols:
                if col in final_df.columns:
                    # 在这里也进行一次净化，确保sum计算的输入是纯净的
                    source_series_for_sum = pd.to_numeric(final_df[col], errors='coerce')
                    sum_col_name = f'{col}_sum_{p}d'
                    final_df[sum_col_name] = source_series_for_sum.rolling(window=p, min_periods=min_p).sum()
        all_cols_to_derive = CORE_METRICS_TO_DERIVE + [f'{c}_sum_{p}d' for c in sum_cols for p in UNIFIED_PERIODS if p > 1]
        for col in all_cols_to_derive:
            # 此时，由于前面的强制净化，这个if条件对于所有数值列都应该为True
            if col in final_df.columns and col not in final_df.select_dtypes(include=['object', 'bool']).columns:
                base_col_name = col.split('_sum_')[0] if '_sum_' in col else col
                if base_col_name in SLOPE_ACCEL_EXCLUSIONS:
                    continue
                source_series = final_df[col].astype(float) # 再次确认，双重保险
                for p in UNIFIED_PERIODS:
                    calc_window = max(2, p) if p > 1 else 2
                    slope_col_name = f'{col}_slope_{p}d'
                    slope_series = ta.slope(close=source_series, length=calc_window)
                    final_df[slope_col_name] = slope_series
                    if slope_series is not None and not slope_series.empty:
                        accel_col_name = f'{col}_accel_{p}d'
                        final_df[accel_col_name] = ta.slope(close=slope_series.astype(float), length=calc_window)
        return final_df

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """【V1.3 · 纯净保存版 + 黑匣子】准备并保存最终计算结果到数据库。"""
        records_to_save_df = final_df
        # [代码新增开始] 增加黑匣子诊断信息
        stock_code = stock_info.stock_code
        print(f"调试信息: [{stock_code}] [节点6] 进入保存函数，待保存记录数: {len(records_to_save_df)}")
        # [代码新增结束]
        if records_to_save_df.empty:
            return 0
        records_to_save_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = records_to_save_df[[col for col in records_to_save_df.columns if col in model_fields]]
        # [代码新增开始] 增加黑匣子诊断信息
        print(f"调试信息: [{stock_code}] [节点7] 按模型字段过滤后，待保存列: {df_filtered.columns.tolist()}")
        # [代码新增结束]
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
        # [代码新增开始] 增加黑匣子诊断信息
        print(f"调试信息: [{stock_code}] [节点8] 已创建 {len(records_to_create)} 个待保存的ORM对象。")
        # [代码新增结束]
        @sync_to_async(thread_sensitive=True)
        def save_metrics_async(model, records_to_create_list):
            with transaction.atomic():
                model.objects.bulk_create(records_to_create_list, batch_size=2000)
        await save_metrics_async(MetricsModel, records_to_create)
        # [代码新增开始] 增加黑匣子诊断信息
        print(f"调试信息: [{stock_code}] [节点9] 批量保存执行完毕。")
        # [代码新增结束]
        return len(records_to_create)

    def _upgrade_intraday_profit_metric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.2 · 鲁棒性重构版】使用健壮的辅助函数重写P&L矩阵计算。
        - 核心修正: 引入_get_numeric_series_with_nan处理成本，保留NaN。
        - 核心修正: 使用修正后的_get_safe_numeric_series处理成交量，填充0。
        """
        # [代码修改开始] 全面采用新的、对单行数据免疫的辅助函数
        results_df = pd.DataFrame(index=df.index)
        # 使用新辅助函数获取成本，保留NaN
        cost_buy = self._get_numeric_series_with_nan(df, 'avg_cost_main_buy')
        cost_sell = self._get_numeric_series_with_nan(df, 'avg_cost_main_sell')
        close_price = self._get_numeric_series_with_nan(df, 'close')
        # 使用修正后的辅助函数获取成交量，填充0
        vol_buy = (self._get_safe_numeric_series(df, 'buy_lg_vol') +
                   self._get_safe_numeric_series(df, 'buy_elg_vol'))
        vol_sell = (self._get_safe_numeric_series(df, 'sell_lg_vol') +
                    self._get_safe_numeric_series(df, 'sell_elg_vol'))
        vol_buy_shares = vol_buy * 100
        vol_sell_shares = vol_sell * 100
        # 后续计算逻辑在Numpy/Pandas的广播机制下，对NaN具有天然的传播性，无需修改
        exchanged_volume = np.minimum(vol_buy_shares, vol_sell_shares)
        results_df['realized_profit_on_exchange'] = (cost_sell - cost_buy) * exchanged_volume
        net_pos_change_vol = vol_buy_shares - vol_sell_shares
        net_pos_change_cost = np.where(net_pos_change_vol > 0, cost_buy, cost_sell)
        results_df['net_position_change_value'] = net_pos_change_vol * net_pos_change_cost
        results_df['unrealized_pnl_on_net_change'] = (close_price - net_pos_change_cost) * net_pos_change_vol
        results_df['main_force_intraday_profit'] = (cost_sell - cost_buy) * vol_buy_shares
        # 此处使用fillna(0)是安全的，因为我们只关心符号
        dir_ts = np.sign(results_df['net_position_change_value'].fillna(0))
        dir_ths = np.sign(self._get_safe_numeric_series(df, 'main_force_net_flow_ths'))
        dir_dc = np.sign(self._get_safe_numeric_series(df, 'main_force_net_flow_dc'))
        agreement_count = (dir_ts == dir_ths).astype(int) + (dir_ts == dir_dc).astype(int) + (dir_ths == dir_dc).astype(int)
        results_df['pnl_matrix_confidence_score'] = ((agreement_count / 3 * 2) + 1) / 3
        return results_df
        # [代码修改结束]

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

    def _calculate_order_size_likelihood_scores(self, minute_data_for_day: pd.DataFrame, daily_data: pd.Series) -> pd.DataFrame:
        """
        【V3.0 · 方向性注入重构版】计算带有买卖方向的订单规模似然分数。
        - 核心修正: 引入 close vs open 作为方向代理，为每个规模生成独立的 buy_score 和 sell_score。
        """
        df = minute_data_for_day.copy()
        circ_mv = pd.to_numeric(daily_data.get('circ_mv'), errors='coerce') * 10000
        if pd.isna(circ_mv) or circ_mv == 0:
            for size in ['sm', 'md', 'lg', 'elg']:
                df[f'{size}_buy_score'] = 0
                df[f'{size}_sell_score'] = 0
            return df
        # [代码修改开始] 引入方向性代理，并生成方向性分数
        elg_threshold = circ_mv * 0.001
        lg_threshold = circ_mv * 0.0002
        md_threshold = circ_mv * 0.00005
        # 1. 计算无方向的原始分数
        elg_score = df['amount_yuan'].where(df['amount_yuan'] >= elg_threshold, 0)
        lg_score = df['amount_yuan'].where((df['amount_yuan'] >= lg_threshold) & (df['amount_yuan'] < elg_threshold), 0)
        md_score = df['amount_yuan'].where((df['amount_yuan'] >= md_threshold) & (df['amount_yuan'] < lg_threshold), 0)
        sm_score = df['amount_yuan'].where(df['amount_yuan'] < md_threshold, 0)
        # 2. 根据分钟线涨跌（方向代理）分配分数
        # close > open: 认为是买方驱动，分数全部分配给buy_score
        buy_driven_mask = df['close'] > df['open']
        # close < open: 认为是卖方驱动，分数全部分配给sell_score
        sell_driven_mask = df['close'] < df['open']
        # close == open: 平盘，买卖均衡，分数对半分
        neutral_mask = df['close'] == df['open']
        # 3. 循环生成带方向的分数
        for size, score_series in [('elg', elg_score), ('lg', lg_score), ('md', md_score), ('sm', sm_score)]:
            df[f'{size}_buy_score'] = np.where(buy_driven_mask, score_series, 0) + np.where(neutral_mask, score_series / 2, 0)
            df[f'{size}_sell_score'] = np.where(sell_driven_mask, score_series, 0) + np.where(neutral_mask, score_series / 2, 0)
        return df
        # [代码修改结束]

    def _calculate_daily_vwap_from_df(self, minute_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.Series:
        """【新增 V1.0】从预加载的DataFrame计算日度VWAP"""
        if minute_df.empty:
            return pd.Series(np.nan, index=date_index)
        df = minute_df.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df[['amount', 'vol']] = df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
        df['total_value'] = df['amount'] * 1000
        df['total_volume'] = df['vol'] * 100
        daily_agg = df.groupby(df['trade_time'].dt.date)
        daily_total_value = daily_agg['total_value'].sum()
        daily_total_volume = daily_agg['total_volume'].sum()
        daily_vwap = daily_total_value / daily_total_volume.replace(0, np.nan)
        daily_vwap.index = pd.to_datetime(daily_vwap.index)
        return daily_vwap.reindex(date_index)

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【新增 V1.0】从预加载的DataFrame构建按日分组的数据"""
        if minute_df is None or minute_df.empty:
            return None
        df = minute_df.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df['date'] = df['trade_time'].dt.date
        df[['amount', 'vol']] = df[['amount', 'vol']].apply(pd.to_numeric, errors='coerce')
        df['amount_yuan'] = df['amount'] * 1000
        df['vol_shares'] = df['vol'] * 100
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
            # [代码修改开始] 在数据源头进行类型净化，杜绝object类型污染
            # 遍历所有非索引列，将它们强制转换为float类型，无法转换的将变为NaN
            for col in df.columns:
                if col != 'trade_time': # trade_time已经是索引了
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            # [代码修改结束]
        return df




