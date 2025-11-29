# services/advanced_chip_metrics_service.py
import asyncio
import pandas as pd
import numpy as np
from datetime import timedelta, datetime, time
from functools import reduce
from django.db import transaction
from asgiref.sync import sync_to_async
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockDailyBasic, StockCyqPerf
from stock_models.advanced_metrics import BaseAdvancedChipMetrics
from utils.model_helpers import (
    get_advanced_chip_metrics_model_by_code,
    get_cyq_chips_model_by_code,
    get_daily_data_model_by_code,
    get_minute_data_model_by_code_and_timelevel,
)
from services.chip_feature_calculator import ChipFeatureCalculator
from strategies.trend_following.utils import get_param_value, get_params_block
import logging
logger = logging.getLogger(__name__)

class AdvancedChipMetricsService:
    """
    【V1.0 · 兵工厂模式】高级筹码指标服务
    - 核心职责: 封装所有高级筹码指标的加载、计算、融合与存储逻辑。
    - 架构优势: 实现业务逻辑与任务调度的完全解耦，并为“深度融合”架构提供标准接口。
    """
    def __init__(self):
        self.max_lookback_days = 200

    async def run_precomputation(self, stock_code: str, is_incremental: bool, start_date_str: str = None, fund_flow_attributed_minute_map: dict = None, debug_params: dict = None):
        """
        【V1.1 · 融合接口版】服务层主执行器。
        - 新增: 接收由资金流服务预先计算好的 `fund_flow_attributed_minute_map`。
        - 【修正】新增 `debug_params` 参数，用于控制内部探针的输出。
        """
        stock_info, MetricsModel, is_incremental_final, last_metric_date, fetch_start_date = await self._initialize_context(
            stock_code, is_incremental, start_date_str
        )
        # 统一以日线数据确定处理范围
        DailyModel = get_daily_data_model_by_code(stock_code)
        date_filter = {'stock': stock_info}
        if fetch_start_date:
            date_filter['trade_time__gte'] = fetch_start_date
        if last_metric_date and is_incremental_final:
            date_filter['trade_time__gt'] = last_metric_date
        all_dates_qs = DailyModel.objects.filter(**date_filter).values_list('trade_time', flat=True).order_by('trade_time')
        dates_to_process = pd.to_datetime(await sync_to_async(list)(all_dates_qs))
        if dates_to_process.empty:
            logger.info(f"[{stock_code}] [筹码服务] 无需计算的日期，任务终止。")
            return 0
        # 统一加载所有数据
        data_dfs = await self._load_all_sources(stock_info, dates_to_process.min(), dates_to_process.max())
        merged_df = self._preprocess_and_merge_data(stock_code, data_dfs)
        minute_data_map = await self._load_minute_data_for_range(stock_info, dates_to_process.min(), dates_to_process.max())
        # 核心指标锻造
        base_metrics_df, _, _ = self._synthesize_and_forge_metrics(
            stock_info, merged_df, minute_data_map, fund_flow_attributed_minute_map, debug_params=debug_params
        )
        if base_metrics_df.empty:
            logger.info(f"[{stock_code}] [筹码服务] 未能计算出任何新的基础指标。")
            return 0
        # 衍生指标升维
        historical_metrics_df = await self._load_historical_metrics(MetricsModel, stock_info, base_metrics_df.index.min())
        full_sequence_df = pd.concat([historical_metrics_df, base_metrics_df]).sort_index()
        final_metrics_df = self._calculate_derivatives(full_sequence_df)
        # 精确切分并保存
        chunk_to_save = final_metrics_df[final_metrics_df.index.isin(base_metrics_df.index)]
        processed_count = await self._prepare_and_save_data(stock_info, MetricsModel, chunk_to_save)
        return processed_count

    async def _initialize_context(self, stock_code: str, is_incremental: bool, start_date_str: str = None):
        """初始化任务上下文，确定计算模式和日期范围。"""
        stock_info = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)
        MetricsModel = get_advanced_chip_metrics_model_by_code(stock_code)
        last_metric_date = None
        fetch_start_date = None
        if start_date_str:
            try:
                start_date_obj = datetime.strptime(start_date_str, '%Y-%m-%d').date()
                is_incremental = True # 强制为增量模式，以便进行回滚删除
                last_metric_date = start_date_obj - timedelta(days=1)
                fetch_start_date = start_date_obj - timedelta(days=self.max_lookback_days)
                deleted_count, _ = await sync_to_async(MetricsModel.objects.filter(stock=stock_info, trade_time__gte=start_date_obj).delete)()
                logger.info(f"[{stock_code}] [筹码服务] 部分全量模式，已回滚删除从 {start_date_obj} 开始的 {deleted_count} 条旧指标。")
            except (ValueError, TypeError):
                logger.error(f"[{stock_code}] 提供的起始日期 '{start_date_str}' 格式错误，将忽略。")
                is_incremental = True
        if is_incremental and not start_date_str:
            latest_metric = await sync_to_async(
                lambda: MetricsModel.objects.filter(stock=stock_info).only('trade_time').latest('trade_time')
            )()
            if latest_metric:
                last_metric_date = latest_metric.trade_time
                fetch_start_date = last_metric_date - timedelta(days=self.max_lookback_days)
            else:
                is_incremental = False
        return stock_info, MetricsModel, is_incremental, last_metric_date, fetch_start_date

    async def _load_all_sources(self, stock_info: StockInfo, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """统一加载并审计所有筹码计算所需的数据源。"""
        @sync_to_async(thread_sensitive=True)
        def get_data_async(model, stock_info_obj, fields: tuple = None, date_field='trade_time', start_dt=None, end_dt=None):
            if not model: return pd.DataFrame()
            qs = model.objects.filter(stock=stock_info_obj, **{f'{date_field}__gte': start_dt, f'{date_field}__lte': end_dt})
            return pd.DataFrame.from_records(qs.values(*fields) if fields else qs.values())
        chip_model = get_cyq_chips_model_by_code(stock_info.stock_code)
        daily_data_model = get_daily_data_model_by_code(stock_info.stock_code)
        data_tasks = {
            "cyq_chips": get_data_async(chip_model, stock_info, fields=('trade_time', 'price', 'percent'), start_dt=start_date, end_dt=end_date),
            "daily_data": get_data_async(daily_data_model, stock_info, fields=('trade_time', 'close_qfq', 'vol', 'high_qfq', 'low_qfq'), start_dt=start_date, end_dt=end_date),
            "daily_basic": get_data_async(StockDailyBasic, stock_info, fields=('trade_time', 'float_share', 'circ_mv'), start_dt=start_date, end_dt=end_date),
            "cyq_perf": get_data_async(StockCyqPerf, stock_info, start_dt=start_date, end_dt=end_date),
        }
        results = await asyncio.gather(*data_tasks.values())
        data_dfs = dict(zip(data_tasks.keys(), results))
        # 数据审计
        for name, df in data_dfs.items():
            if df is None or df.empty:
                raise ValueError(f"[审计失败] 核心数据源 '{name}' 在日期范围 {start_date.date()} to {end_date.date()} 为空！")
        return data_dfs

    def _preprocess_and_merge_data(self, stock_code: str, data_dfs: dict, base_daily_df: pd.DataFrame, close_map: dict, date_20d_ago_map: dict, atr_map: dict, high_20d_map: dict, low_20d_map: dict, high_5d_map: dict, low_5d_map: dict, turnover_vol_5d_map: dict) -> pd.DataFrame:
        """
        【V4.0 · 瘦身重构版】
        - 核心重构: 剥离日线和基础面数据的合并逻辑，改为接收由上游任务统一准备好的 `base_daily_df`。
        - 核心职责: 仅负责合并筹码特有数据源（CYQ、CYQPerf），并与标准化的 `base_daily_df` 进行最终连接。
        """
        cyq_chips_df = data_dfs['cyq_chips'].copy()
        cyq_perf_df = data_dfs['cyq_perf'].copy()
        cyq_chips_df['trade_time'] = pd.to_datetime(cyq_chips_df['trade_time'])
        cyq_perf_df['trade_time'] = pd.to_datetime(cyq_perf_df['trade_time'])
        cyq_perf_df.drop(columns=['id', 'stock_id'], errors='ignore', inplace=True)
        cyq_perf_df.set_index('trade_time', inplace=True)
        # 架构升级：直接与上游传入的 base_daily_df 合并，不再自行处理 daily_data 和 daily_basic
        daily_combined_df = base_daily_df.join(cyq_perf_df, how='left')
        merged_df = pd.merge(cyq_chips_df, daily_combined_df.reset_index(), on='trade_time', how='right')
        merged_df.sort_values(by=['trade_time', 'price'], inplace=True)
        merged_df['prev_20d_trade_time'] = merged_df['trade_time'].map(date_20d_ago_map)
        merged_df['prev_20d_close'] = merged_df['prev_20d_trade_time'].map(close_map)
        merged_df['atr_14d'] = merged_df['trade_time'].map(atr_map)
        merged_df['high_20d'] = merged_df['trade_time'].map(high_20d_map)
        merged_df['low_20d'] = merged_df['trade_time'].map(low_20d_map)
        merged_df['high_5d'] = merged_df['trade_time'].map(high_5d_map)
        merged_df['low_5d'] = merged_df['trade_time'].map(low_5d_map)
        merged_df['turnover_vol_5d'] = merged_df['trade_time'].map(turnover_vol_5d_map)
        merged_df.drop(columns=['prev_20d_trade_time'], inplace=True)
        return merged_df

    def _synthesize_and_forge_metrics(self, stock_info: StockInfo, merged_df: pd.DataFrame, minute_data_map: dict, fund_flow_attributed_minute_map: dict, memory: dict = None, historical_components: pd.DataFrame = None, debug_params: dict = None, tick_data_map: dict = None, realtime_data_map: dict = None, level5_data_map: dict = None) -> tuple[pd.DataFrame, dict, list]:
        """
        【V1.11 · 记忆补录修复版】
        - 核心修复: 在构建传递给下一日的记忆字典 `next_prev_metrics` 时，补充了对 `main_force_cumulative_cost`
                     的记录。此举修复了因遗漏导致累积成本状态无法跨区块传递，从而引发全量回测中“失忆”的根本问题。
        """
        stock_code = stock_info.stock_code
        all_metrics_list = []
        failures_list = []
        prev_metrics = memory.copy() if memory is not None else {}
        hist_comp_cols = None
        if historical_components is not None and not historical_components.columns.empty:
            hist_comp_cols = historical_components.columns
        else:
            hist_comp_cols = pd.Index([
                'concentration_70pct', 'cost_divergence_normalized', 'dominant_peak_profit_margin',
                'main_force_cost_advantage', 'suppressive_accumulation_intensity', 'upward_impulse_purity',
                'active_winner_profit_margin', 'winner_conviction_index'
            ])
        hist_comp_dict = historical_components.to_dict('index') if historical_components is not None and not historical_components.empty else {}
        grouped_data = merged_df.groupby('trade_time')
        required_daily_chip_cols = ['close_qfq', 'vol', 'float_share', 'circ_mv', 'weight_avg', 'winner_rate', 'pre_close_qfq', 'open_qfq', 'high_qfq', 'low_qfq']
        is_first_day_in_batch = True
        debug_params = debug_params if debug_params is not None else {}
        for i, (trade_date, daily_full_df) in enumerate(grouped_data):
            date_obj = trade_date.date()
            enable_probe = debug_params.get('enable_mfca_probe', False)
            target_date_str = debug_params.get('target_date')
            is_target_date = str(date_obj) == target_date_str
            if enable_probe and is_target_date:
                print(f"\n[探针 C.1 - {stock_code} @ {date_obj}] 进入筹码服务 _synthesize_and_forge_metrics...")
                print(f"  - 传入的 daily_full_df 的列: {daily_full_df.columns.tolist()}")
                if 'avg_cost_main_buy' in daily_full_df.columns:
                    print(f"  - 成功接收到 avg_cost_main_buy: {daily_full_df['avg_cost_main_buy'].iloc[0]}")
                else:
                    print("  - 警告: daily_full_df 中未找到 avg_cost_main_buy 列！")
            context_data = daily_full_df.iloc[0].to_dict()
            chip_data_for_calc = daily_full_df[['price', 'percent']].dropna()
            if chip_data_for_calc.empty:
                reason = "当日源筹码分布(cyq_chips)数据缺失"
                logger.warning(f"[{stock_code}] [{trade_date.date()}] 预警：{reason}。")
                failures_list.append({'stock_code': stock_code, 'trade_date': str(trade_date.date()), 'reason': reason})
                if is_first_day_in_batch: is_first_day_in_batch = False
                continue
            missing_keys = [key for key in required_daily_chip_cols if key not in context_data or pd.isna(context_data[key])]
            if not chip_data_for_calc.empty and chip_data_for_calc['percent'].sum() < 0.1:
                missing_keys.append('valid_chip_distribution')
            if missing_keys:
                reason = f"缺失核心原料数据: {missing_keys}"
                logger.warning(f"[{stock_code}] [{trade_date.date()}] 跳过筹码计算，{reason}")
                failures_list.append({'stock_code': stock_code, 'trade_date': str(trade_date.date()), 'reason': reason})
                continue
            context_for_calc = context_data.copy()
            daily_amount = pd.to_numeric(context_data.get('amount'), errors='coerce') * 1000
            daily_vol_shares = pd.to_numeric(context_data.get('vol'), errors='coerce') * 100
            if pd.notna(daily_amount) and pd.notna(daily_vol_shares) and daily_vol_shares > 0:
                context_for_calc['daily_vwap'] = daily_amount / daily_vol_shares
            else:
                context_for_calc['daily_vwap'] = np.nan
            close_price_today = context_data.get('close_qfq')
            recent_closes_list = prev_metrics.get('recent_closes_queue', [])
            if len(recent_closes_list) >= 10:
                recent_closes_list.pop(0)
            recent_closes_list.append(close_price_today)
            total_chip_volume_today = context_data.get('float_share', 0) * 10000
            context_for_calc.update({
                'close_price': close_price_today,
                'high_price': context_data.get('high_qfq'),
                'low_price': context_data.get('low_qfq'),
                'open_price': context_data.get('open_qfq'),
                'pre_close': context_data.get('pre_close_qfq'),
                'daily_turnover_volume': context_data.get('vol', 0) * 100,
                'total_chip_volume': total_chip_volume_today, 'stock_code': stock_code, 'trade_date': trade_date.date(),
                'circ_mv': context_data.get('circ_mv'), 'is_first_day_in_batch': is_first_day_in_batch,
                'high_20d': context_data.get('high_20d'), 'low_20d': context_data.get('low_20d'),
                'atr_14d': context_data.get('atr_14d'),
                'high_5d': context_data.get('high_5d'), 'low_5d': context_data.get('low_5d'),
                'turnover_vol_5d': context_data.get('turnover_vol_5d'),
                'debug_params': debug_params,
                'prev_metrics': prev_metrics,
            })
            if enable_probe and is_target_date:
                print(f"[探针 C.2 - {stock_code} @ {date_obj}] 准备传递给计算器的 context_for_calc...")
                if 'avg_cost_main_buy' in context_for_calc:
                    print(f"  - 成功包含 avg_cost_main_buy: {context_for_calc.get('avg_cost_main_buy')}")
                else:
                    print("  - 致命错误: context_for_calc 中仍未找到 avg_cost_main_buy！")
            historical_data_for_day = {k: v for k, v in hist_comp_dict.items() if k < trade_date}
            if historical_data_for_day:
                context_for_calc['historical_components'] = pd.DataFrame.from_dict(historical_data_for_day, orient='index')
            else:
                context_for_calc['historical_components'] = pd.DataFrame(columns=hist_comp_cols)
            if fund_flow_attributed_minute_map and date_obj in fund_flow_attributed_minute_map:
                enhanced_intraday_data = fund_flow_attributed_minute_map[date_obj]
            else:
                enhanced_intraday_data = minute_data_map.get(date_obj, pd.DataFrame())
            context_for_calc['intraday_data'] = enhanced_intraday_data
            context_for_calc['tick_data'] = tick_data_map.get(date_obj, pd.DataFrame()) if tick_data_map else pd.DataFrame()
            context_for_calc['level5_data'] = level5_data_map.get(date_obj, pd.DataFrame()) if level5_data_map else pd.DataFrame()
            merged_realtime_df = pd.DataFrame()
            tick_df_day = tick_data_map.get(date_obj)
            level5_df_day = level5_data_map.get(date_obj)
            if tick_df_day is not None and not tick_df_day.empty and level5_df_day is not None and not level5_df_day.empty:
                if all(col in tick_df_day.columns for col in ['price', 'volume']) and 'buy_price1' in level5_df_day.columns:
                    tick_df_sorted = tick_df_day.sort_index()
                    merged_realtime_df = pd.merge_asof(
                        left=tick_df_sorted.reset_index(),
                        right=level5_df_day.reset_index(),
                        on='trade_time',
                        direction='backward'
                    )
                    if 'trade_time' in merged_realtime_df.columns:
                        merged_realtime_df.set_index('trade_time', inplace=True)
                    column_rename_map = {
                        **{f'buy_price{i}': f'b{i}_p' for i in range(1, 6)},
                        **{f'buy_volume{i}': f'b{i}_v' for i in range(1, 6)},
                        **{f'sell_price{i}': f'a{i}_p' for i in range(1, 6)},
                        **{f'sell_volume{i}': f'a{i}_v' for i in range(1, 6)},
                    }
                    merged_realtime_df.rename(columns=column_rename_map, inplace=True)
            context_for_calc['realtime_data'] = merged_realtime_df
            calculator = ChipFeatureCalculator(chip_data_for_calc, context_for_calc)
            daily_metrics = calculator.calculate_all_metrics()
            if daily_metrics:
                daily_metrics['trade_time'] = trade_date
                daily_metrics['prev_20d_close'] = context_data.get('prev_20d_close')
                all_metrics_list.append(daily_metrics)
                today_metrics_for_hist = {k: [daily_metrics.get(k)] for k in hist_comp_cols}
                today_df = pd.DataFrame(today_metrics_for_hist, index=[trade_date])
                hist_comp_dict.update(today_df.to_dict('index'))
            # [修改代码块] 构建下一日的记忆
            next_prev_metrics = {
                'chip_distribution': chip_data_for_calc,
                'close_price': context_data.get('close_qfq'),
                'prev_20d_close': context_data.get('prev_20d_close'),
                'high_20d': context_data.get('high_20d'),
                'low_20d': context_data.get('low_20d'),
                'total_chip_volume': total_chip_volume_today,
                'recent_closes_queue': recent_closes_list,
                'atr_14d': context_data.get('atr_14d'),
            }
            if daily_metrics:
                # 核心修复：确保将当日计算出的累积成本存入记忆
                next_prev_metrics['main_force_cumulative_cost'] = daily_metrics.get('main_force_cumulative_cost')
                next_prev_metrics.update(daily_metrics)
            else:
                next_prev_metrics.update({
                    'concentration_90pct': None, 'winner_avg_cost': None,
                    'dominant_peak_cost': None, 'chip_fatigue_index': 0.0,
                    'cost_gini_coefficient': None, 'total_winner_rate': None,
                    'price_volume_entropy': None, 'strategic_phase_score': None,
                    'main_force_cumulative_cost': prev_metrics.get('main_force_cumulative_cost'), # 如果当天计算失败，继承前一天的
                })
            prev_metrics = next_prev_metrics
            if is_first_day_in_batch: is_first_day_in_batch = False
        if not all_metrics_list:
            return pd.DataFrame(), prev_metrics, failures_list
        return pd.DataFrame(all_metrics_list).set_index('trade_time'), prev_metrics, failures_list

    def _enhance_minute_data_fallback(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.2 · 竞价数据保全版】当资金流数据缺失时，提供一个基础的分钟数据增强。
        - 核心修复: 移除对集合竞价时段的错误过滤，确保包含15:00竞价数据的完整分钟线被传递给下游计算器。
        """
        if minute_df.empty: return minute_df
        df = minute_df.copy()
        df['amount_yuan'] = pd.to_numeric(df['amount'], errors='coerce')
        df['vol_shares'] = pd.to_numeric(df['vol'], errors='coerce')
        df['minute_vwap'] = df['amount_yuan'] / df['vol_shares'].replace(0, np.nan)
        return df

    async def _load_minute_data_for_range(self, stock_info: StockInfo, start_date: pd.Timestamp, end_date: pd.Timestamp, tick_data_map: dict = None, minute_data_map: dict = None):
        """
        【V1.8 · 探针清理版】不再查询数据库，仅处理由上游任务传入的日内数据maps。
        - 核心重构: 移除所有数据库查询逻辑，职责单一化为数据处理与聚合。
        - 核心逻辑: 遍历所需日期，优先使用tick_data_map，若无则回退到minute_data_map。
        - 核心修复: 确保返回的DataFrame都经过 `_group_minute_data_from_df` 处理，并以 `trade_time` 为 `DatetimeIndex`，保持数据结构一致性。
        - 【维护】移除了所有用于调试的 `print` 探针语句。
        """
        from stock_models.time_trade import StockDailyBasic
        intraday_data_map = {}
        all_dates_in_range_qs = StockDailyBasic.objects.filter(stock=stock_info, trade_time__gte=start_date, trade_time__lte=end_date).values_list('trade_time', flat=True)
        all_required_dates = {d.date() for d in pd.to_datetime(await sync_to_async(list)(all_dates_in_range_qs))}
        for date_obj in sorted(list(all_required_dates)):
            processed_intraday_for_day = None
            if tick_data_map and date_obj in tick_data_map:
                tick_df = tick_data_map[date_obj].copy()
                if not all(col in tick_df.columns for col in ['price', 'volume', 'amount']):
                    logger.warning(f"[{stock_info.stock_code}] [筹码服务] 日期 {date_obj} 逐笔数据缺少'price', 'volume'或'amount'列，将尝试回退到分钟数据。")
                else:
                    current_price_col = 'price'
                    current_volume_col = 'volume'
                    current_amount_col = 'amount'
                    if 'type' not in tick_df.columns:
                        tick_df['type'] = 'M'
                    buy_vol_per_minute = tick_df[tick_df['type'] == 'B'].resample('1min')[current_volume_col].sum()
                    sell_vol_per_minute = tick_df[tick_df['type'] == 'S'].resample('1min')[current_volume_col].sum()
                    minute_df_from_ticks = tick_df.resample('1min').agg(
                        open=(current_price_col, 'first'), high=(current_price_col, 'max'), low=(current_price_col, 'min'),
                        close=(current_price_col, 'last'), vol=(current_volume_col, 'sum'), amount=(current_amount_col, 'sum')
                    ).dropna(subset=['open', 'high', 'low', 'close', 'vol', 'amount'])
                    minute_df_from_ticks['buy_vol_raw'] = buy_vol_per_minute
                    minute_df_from_ticks['sell_vol_raw'] = sell_vol_per_minute
                    minute_df_from_ticks.fillna(0, inplace=True)
                    processed_intraday_for_day = self._group_minute_data_from_df(minute_df_from_ticks)
            if processed_intraday_for_day is None and minute_data_map and date_obj in minute_data_map:
                processed_intraday_for_day = self._group_minute_data_from_df(minute_data_map[date_obj])
            if processed_intraday_for_day is not None:
                intraday_data_map[date_obj] = processed_intraday_for_day
            else:
                # 已移除: 清理了在未找到预加载日内数据时打印调试信息的逻辑
                logger.info(f"[{stock_info.stock_code}] [筹码服务] 日期 {date_obj} 未找到任何预加载的日内数据。")
        return intraday_data_map

    async def _load_historical_metrics(self, model, stock_info, end_date):
        """从数据库加载历史高级筹码指标。"""
        @sync_to_async
        def get_data():
            core_metric_cols = list(BaseAdvancedChipMetrics.CORE_METRICS.keys())
            required_cols = ['trade_time'] + [col for col in core_metric_cols if hasattr(model, col)]
            qs = model.objects.filter(stock=stock_info, trade_time__lt=end_date).order_by('trade_time')
            return pd.DataFrame.from_records(qs.values(*required_cols))
        df = await get_data()
        if not df.empty:
            # 修复：分两步操作，先转换类型，再用列名设置索引，确保 'trade_time' 列被正确移除
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df = df.set_index('trade_time')
            for col in df.columns:
                # 此处的 if col != 'trade_time' 检查现在是多余但无害的
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _calculate_derivatives(self, consensus_df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.3 · 动态尺度重构版】
        - 核心重构(加速度): 废弃固定的2日加速度窗口，引入与斜率周期p动态关联的加速度窗口(accel_window = max(2, p // 4))。
        - 核心思想: 解决了“望远镜与显微镜悖论”。现在，对长期趋势（如55日斜率）的加速度评估，是在一个更长、
                     更合理的尺度（13日）上进行，有效过滤短期噪声，使“加速度”指标能真正揭示趋势的系统性拐点。
        """
        derivatives_df = pd.DataFrame(index=consensus_df.index)
        import pandas_ta as ta
        SLOPE_ACCEL_EXCLUSIONS = BaseAdvancedChipMetrics.SLOPE_ACCEL_EXCLUSIONS
        CORE_METRICS_TO_DERIVE = list(BaseAdvancedChipMetrics.CORE_METRICS.keys())
        UNIFIED_PERIODS = BaseAdvancedChipMetrics.UNIFIED_PERIODS
        # [修改代码块] 废弃固定的加速度窗口
        # ACCEL_WINDOW = 2
        for col in CORE_METRICS_TO_DERIVE:
            if col in consensus_df.columns and col not in SLOPE_ACCEL_EXCLUSIONS and col not in BaseAdvancedChipMetrics.BOOLEAN_FIELDS:
                source_series = pd.to_numeric(consensus_df[col], errors='coerce')
                if source_series.isnull().all():
                    continue
                for p in UNIFIED_PERIODS:
                    calc_window = max(2, p)
                    slope_col_name = f'{col}_slope_{p}d'
                    slope_series = ta.slope(close=source_series, length=calc_window)
                    derivatives_df[slope_col_name] = slope_series
                    if slope_series is not None and not slope_series.empty:
                        accel_col_name = f'{col}_accel_{p}d'
                        # [修改代码块] 引入与斜率周期p动态关联的加速度窗口
                        # 新逻辑：加速度的观察尺度应与速度的观察尺度相匹配
                        accel_window = max(2, p // 4)
                        derivatives_df[accel_col_name] = ta.slope(close=slope_series.astype(float), length=accel_window)
        return derivatives_df

    async def _prepare_and_save_data(self, stock_info, MetricsModel, final_df: pd.DataFrame):
        """准备并以“更新或创建”的方式原子化保存数据。"""
        if final_df.empty: return 0
        final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # 健壮性修复：确保所有布尔字段在保存前都有确定的值 (True/False)，而不是 NaN。
        # NaN 在保存时会变成 NULL，导致数据库 NOT NULL 约束错误。
        boolean_fields = BaseAdvancedChipMetrics.BOOLEAN_FIELDS
        for col in boolean_fields:
            if col in final_df.columns:
                # 将 NaN 值填充为 False，这是布尔字段最安全的默认值
                final_df[col] = final_df[col].fillna(False)
        model_fields = {f.name for f in MetricsModel._meta.get_fields() if not f.is_relation and f.name != 'id'}
        df_filtered = final_df[[col for col in final_df.columns if col in model_fields]]
        records_list = df_filtered.to_dict('records')
        @sync_to_async(thread_sensitive=True)
        def save_atomically(model, stock_obj, records_to_process):
            processed_count = 0
            for record_data in records_to_process:
                trade_time = record_data.pop('trade_time').date()
                defaults_data = {key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in record_data.items()}
                try:
                    obj, created = model.objects.update_or_create(stock=stock_obj, trade_time=trade_time, defaults=defaults_data)
                    processed_count += 1
                except Exception as e:
                    logger.error(f"[{stock_obj.stock_code}] [筹码保存失败] 日期: {trade_time}, 错误: {e}")
            return processed_count
        records_for_atomic_save = []
        for record_date, record_data in zip(df_filtered.index, records_list):
            record_data['trade_time'] = record_date
            records_for_atomic_save.append(record_data)
        processed_count = await save_atomically(MetricsModel, stock_info, records_for_atomic_save)
        return processed_count

    def _group_minute_data_from_df(self, minute_df: pd.DataFrame):
        """【V1.4 · 数据完整性修复版 - 辅助列添加 - 智能列名识别】从预加载的DataFrame构建按日分组的数据。
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
            # 如果索引是 naive，假定它是 UTC（因为DAO层应该输出UTC aware，但可能在某些操作后丢失时区信息）
            df.index = df.index.tz_localize('UTC', ambiguous='infer').tz_convert(timezone.get_current_timezone())
        else:
            # 如果索引是 aware，直接转换为目标时区
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






