# 文件: services/realtime_services.py

import logging
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from asgiref.sync import sync_to_async
from celery import group
from chaoyue_dreams.celery import app as celery_app
import pytz # <--- 1. 导入 pytz
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time, date # <--- 2. 导入 time
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager
from stock_models.stock_analytics import FavoriteStockTracker, TrendFollowStrategySignalLog
from stock_models.index import TradeCalendar
from utils.cash_key import StockCashKey

from celery import group, chord # 确保 group 和 chord 都已导入

logger = logging.getLogger("services")


@celery_app.task(name='services.realtime_services.cpu_bound_calculation_task', queue='cpu_intensive_queue')
def cpu_bound_calculation_task(
    stock_data_package: Tuple[str, Optional[dict], Optional[dict], Optional[dict]],
    time_level: str,
    slope_window: int,
    stats_window: int
) -> Optional[dict]:
    """
    【V5.3 - 修复Decimal/Float冲突版】
    在计算前，将所有 Decimal 类型强制转换为 float，以保证数值计算的兼容性。
    """
    stock_code, serialized_minute, serialized_ticks, serialized_level5 = stock_data_package

    try:
        def _reconstruct_df_from_dict(data: Optional[dict]) -> Optional[pd.DataFrame]:
            if data is None:
                return None
            df = pd.DataFrame(data['data'], columns=data['columns'], index=data['index'])
            df.index = pd.to_datetime(df.index)
            return df

        df_minute = _reconstruct_df_from_dict(serialized_minute)
        df_ticks = _reconstruct_df_from_dict(serialized_ticks)
        df_level5 = _reconstruct_df_from_dict(serialized_level5)

        if df_minute is None or df_minute.empty:
            print(f"    -> [Celery Worker] 跳过 {stock_code}，因为没有分钟K线数据。")
            return None

        # ▼▼▼ 核心修改：强制转换 Decimal 类型为 float ▼▼▼
        # 定义可能为 Decimal 类型的列名列表
        decimal_cols_minute = ['open', 'high', 'low', 'close', 'turnover_value']
        decimal_cols_ticks = ['current_price', 'buy_price1', 'sell_price1'] # 根据您的tick数据结构调整
        
        for col in decimal_cols_minute:
            if col in df_minute.columns:
                # 使用 pd.to_numeric 转换，它比 .astype 更稳健
                df_minute[col] = pd.to_numeric(df_minute[col], errors='coerce')

        if df_ticks is not None:
            for col in decimal_cols_ticks:
                if col in df_ticks.columns:
                    df_ticks[col] = pd.to_numeric(df_ticks[col], errors='coerce')
        # ▲▲▲ 类型转换结束，后续所有计算都将在 float 类型上进行 ▲▲▲

        # --- 1. 聚合Tick数据 (逻辑不变) ---
        if df_ticks is not None and not df_ticks.empty and 'buy_volume1' in df_ticks.columns:
            df_ticks['buy_com'] = df_ticks[['buy_volume1', 'buy_volume2', 'buy_volume3', 'buy_volume4', 'buy_volume5']].sum(axis=1)
            df_ticks['sell_com'] = df_ticks[['sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5']].sum(axis=1)
            df_ticks['energy_ratio'] = (df_ticks['buy_com'] / (df_ticks['sell_com'] + 1e-6)).clip(0, 100)
            df_ticks['aggressive_buy_volume'] = df_ticks.apply(lambda row: row['volume'] if row['current_price'] >= row['sell_price1'] else 0, axis=1)
            df_ticks['aggressive_sell_volume'] = df_ticks.apply(lambda row: row['volume'] if row['current_price'] <= row['buy_price1'] else 0, axis=1)
            aggregation_rules = {
                'buy_com': ['mean'], 'sell_com': ['mean'], 'energy_ratio': ['mean', 'max', 'min'],
                'aggressive_buy_volume': ['sum'], 'aggressive_sell_volume': ['sum'],
                'volume': ['sum', 'count']
            }
            df_aggregated = df_ticks.resample(time_level).agg(aggregation_rules)
            df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
            df_aggregated.rename(columns={
                'buy_com_mean': 'buy_com_mean', 'sell_com_mean': 'sell_com_mean',
                'energy_ratio_mean': 'energy_ratio_mean', 'energy_ratio_max': 'energy_ratio_max',
                'energy_ratio_min': 'energy_ratio_min', 'aggressive_buy_volume_sum': 'agg_buy_vol_sum',
                'aggressive_sell_volume_sum': 'agg_sell_vol_sum', 'volume_sum': 'agg_volume',
                'volume_count': 'tick_count',
            }, inplace=True)
            df_minute = df_minute.join(df_aggregated, how='left')

        # --- 2. 计算各种技术指标 (逻辑不变) ---
        if 'turnover_value' in df_minute.columns and 'volume' in df_minute.columns:
            df_minute['vwap'] = df_minute['turnover_value'].cumsum() / (df_minute['volume'].cumsum() + 1e-6)
        if df_minute.empty: return None
        if 'agg_buy_vol_sum' in df_minute.columns and 'agg_sell_vol_sum' in df_minute.columns:
            df_minute['net_aggressive_volume'] = df_minute['agg_buy_vol_sum'] - df_minute['agg_sell_vol_sum']
        df_minute['price_pct_change'] = df_minute.ta.percent_return(length=1, cores=1)
        custom_strategy = ta.Strategy(
            name="Intraday_Advanced_Features",
            ta=[
                {"kind": "slope", "close": "net_aggressive_volume", "length": slope_window, "col_names": "net_agg_vol_slope"},
                {"kind": "slope", "close": "buy_com_mean", "length": slope_window, "col_names": "buy_com_slope"},
                {"kind": "slope", "close": "energy_ratio_mean", "length": slope_window, "col_names": "energy_ratio_slope"},
                {"kind": "zscore", "close": "volume", "length": stats_window, "col_names": "volume_zscore"},
                {"kind": "zscore", "close": "net_aggressive_volume", "length": stats_window, "col_names": "net_agg_vol_zscore"},
                {"kind": "ema", "close": "net_aggressive_volume", "length": 10, "col_names": "net_agg_vol_ema10"},
            ]
        )
        df_minute.ta.strategy(custom_strategy, cores=1)
        if 'net_agg_vol_slope' in df_minute.columns:
            df_minute['net_agg_vol_accel'] = df_minute.ta.slope(close=df_minute['net_agg_vol_slope'], length=slope_window, cores=1)
        if 'net_aggressive_volume' in df_minute.columns:
            bbands_df = df_minute.ta.bbands(close=df_minute['net_aggressive_volume'], length=stats_window, col_names=('BBL', 'BBM', 'BBU', 'BBB', 'BBP'), cores=1)
            df_minute = df_minute.join(bbands_df)
        stdev = df_minute.ta.stdev(length=stats_window, cores=1)
        sma = df_minute.ta.sma(length=stats_window, cores=1)
        df_minute['price_cv'] = stdev / (sma + 1e-6)
        if 'net_aggressive_volume' in df_minute.columns:
            df_minute['corr_price_net_agg_vol'] = df_minute['price_pct_change'].rolling(stats_window).corr(df_minute['net_aggressive_volume'])
        try:
            df_minute.ta.fractal(append=True, cores=1)
            rename_map = {'FRACTAL_low_2': 'fractal_low', 'FRACTAL_high_2': 'fractal_high'}
            df_minute.rename(columns=rename_map, inplace=True)
        except Exception:
            pass
            
        df_minute.reset_index(inplace=True)
        return df_minute.to_dict('records')

    except Exception as e:
        # 增加 exc_info=True 来打印完整的堆栈跟踪，方便调试
        logger.error(f"    -> [Celery Worker] 处理 {stock_code} 时发生严重错误: {e}", exc_info=True)
        return None


class RealtimeServices:
    """
    【盘中引擎 - 服务层 V4.3 - 使用数据库交易日历】
    - 改造: 使用数据库中的 TradeCalendar 模型替代外部工具类。
    """
    def __init__(self, cache_manager_instance: CacheManager):
        self.cache_manager = cache_manager_instance
        self.realtime_dao = StockRealtimeDAO(cache_manager_instance)
        self.timetrade_dao = StockTimeTradeDAO(cache_manager_instance)
        self.cache_key = StockCashKey()
        # self.trade_calendar = ATradeCalendar() # <-- 删除这一行
        self.slope_window = 5
        self.stats_window = 20

    @sync_to_async
    def _get_monitoring_pool_from_sources(self, trade_date: date) -> tuple[list[str], list[str]]:
        """
        【内部方法】从数据库并发获取策略Top100股和所有自选股。
        (此方法保持不变)
        """
        # ... 此方法内部代码完全不变 ...
        try:
            top_stocks_qs = TrendFollowStrategySignalLog.objects.filter(
                trade_time__date=trade_date,
                timeframe='D',
                entry_signal=True
            ).order_by('-entry_score').values_list('stock__stock_code', flat=True)[:100]
            strategy_stocks = list(top_stocks_qs)
        except Exception as e:
            logger.error(f"获取 {trade_date} Top策略股时出错: {e}", exc_info=True)
            strategy_stocks = []
        try:
            watchlist_qs = FavoriteStockTracker.objects.filter(
                status='HOLDING'
            ).select_related('stock').values_list('stock__stock_code', flat=True).distinct()
            watchlist_stocks = list(watchlist_qs)
        except Exception as e:
            logger.error(f"获取所有自选股时出错: {e}", exc_info=True)
            watchlist_stocks = []
        return strategy_stocks, watchlist_stocks

    # [修改后方法]
    async def update_and_cache_monitoring_pool(self):
        """
        【盘前任务入口 - V2.0】使用数据库中的TradeCalendar模型更新监控池。
        """
        print("开始更新盘中监控股票池...")
        
        # ▼▼▼ 核心修改 ▼▼▼
        # 1. 将同步的类方法包装成异步函数
        get_prev_date_async = sync_to_async(TradeCalendar.get_latest_trade_date, thread_sensitive=True)
        
        # 2. 异步调用，获取前一个交易日
        previous_trade_date = await get_prev_date_async(reference_date=date.today())
        
        # 3. 关键的健壮性检查
        if not previous_trade_date:
            logger.error("无法从数据库TradeCalendar中获取到前一个交易日，盘前准备任务终止！")
            return
        # ▲▲▲ 核心修改结束 ▲▲▲

        print(f"  -> 目标策略日期 (前一交易日): {previous_trade_date}")

        strategy_stocks, watchlist_stocks = await self._get_monitoring_pool_from_sources(previous_trade_date)
        
        print(f"  -> 获取到自选股: {len(watchlist_stocks)} 支")
        print(f"  -> 获取到策略Top100股: {len(strategy_stocks)} 支")

        final_pool_set = set(watchlist_stocks) | set(strategy_stocks)
        final_pool_list = list(final_pool_set)
        
        print(f"  -> 合并去重后，最终监控池大小为: {len(final_pool_list)} 支")

        if not final_pool_list:
            logger.warning("生成的最终监控股票池为空，不进行缓存操作。")
            return

        redis_key = self.cache_key.intraday_monitoring_pool()
        await self.cache_manager.delete(redis_key)
        await self.cache_manager.sadd(redis_key, *final_pool_list)
        await self.cache_manager.expire(redis_key, 86400)
        
        print(f"  -> 成功将 {len(final_pool_list)} 支股票存入Redis缓存键: {redis_key}")

    # [以下方法保持不变]
    async def get_monitoring_pool_from_cache(self) -> List[str]:
        """
        【盘中任务入口 - V2.1 修正版】从Redis中获取盘中监控的股票池。
        - 修正: 移除了对已反序列化字符串的多余 .decode() 操作。
        """
        redis_key = self.cache_key.intraday_monitoring_pool()
        
        # 1. 调用 self.cache_manager.smembers，它已经返回了反序列化后的字符串列表
        stock_codes_from_cache = await self.cache_manager.smembers(redis_key)
        
        # 2. 检查返回结果
        if stock_codes_from_cache is None:
            # smembers 在连接失败时可能返回 None
            logger.error(f"从Redis缓存键 {redis_key} 获取股票池失败（连接错误？）。")
            return []
        
        if not stock_codes_from_cache:
            logger.warning(f"无法从Redis缓存键 {redis_key} 中获取到任何股票，监控池为空！")
            return []
            
        # 3. 直接使用结果，不再需要解码
        stock_codes = stock_codes_from_cache # 直接赋值即可
        
        print(f"成功从Redis加载监控池，共 {len(stock_codes)} 支股票。")
        return stock_codes

    async def process_all_stocks_intraday_data(self, stock_codes: list, time_level: str, trade_date: str):
        """
        【V5.3 - Chord异步工作流版】
        使用 Celery Chord 将并行计算与结果聚合解耦，避免阻塞。
        """
        from tasks.stock_analysis_tasks import aggregate_intraday_results
        print(f"【V5.3】开始为 {len(stock_codes)} 支股票构建 Chord 异步工作流...")
        
        # --- 步骤 1: I/O 阶段 (逻辑不变) ---
        # ... (这部分数据准备的代码完全不变) ...
        print("  -> 阶段 1/2: 正在集中获取所有股票的原始数据...")
        all_bulk_data = await self.realtime_dao.get_daily_ticks_and_level5_in_bulk(stock_codes, trade_date)
        if not all_bulk_data:
            logger.warning(f"未能从缓存中批量获取到任何股票的 Ticks/Level5 数据，日期: {trade_date}。任务终止。")
            return
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        target_date_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
        start_dt_aware = shanghai_tz.localize(datetime.combine(target_date_obj, time(9, 25, 0)))
        end_dt_aware = shanghai_tz.localize(datetime.combine(target_date_obj, time(15, 5, 0)))
        kline_tasks = [
            self.timetrade_dao.get_minute_kline_by_daterange(code, time_level, start_dt_aware, end_dt_aware)
            for code in stock_codes
        ]
        all_minute_klines_results = await asyncio.gather(*kline_tasks, return_exceptions=True)
        job_packages = []
        for i, code in enumerate(stock_codes):
            df_minute = all_minute_klines_results[i]
            if isinstance(df_minute, Exception) or df_minute is None or df_minute.empty:
                continue
            bulk_data = all_bulk_data.get(code)
            df_ticks, df_level5 = (bulk_data[0], bulk_data[1]) if bulk_data else (None, None)
            serialized_minute = df_minute.to_dict('split') if df_minute is not None else None
            serialized_ticks = df_ticks.to_dict('split') if df_ticks is not None else None
            serialized_level5 = df_level5.to_dict('split') if df_level5 is not None else None
            job_packages.append((code, serialized_minute, serialized_ticks, serialized_level5))
        
        print(f"  -> 数据获取完成，准备将 {len(job_packages)} 个有效计算任务打包到 Chord 中。")

        # --- 步骤 2: 构建并启动 Chord 工作流 ---
        if not job_packages:
            print("没有可执行的计算任务，流程结束。")
            return

        # 2.1 创建并行任务组 (Header)
        calculation_group = group(
            cpu_bound_calculation_task.s(
                stock_data_package=pkg,
                time_level=time_level,
                slope_window=self.slope_window,
                stats_window=self.stats_window
            ) for pkg in job_packages
        )

        # 2.2 创建回调任务签名 (Body)
        # .s() 创建一个不带参数的签名，它将自动接收 group 的结果
        callback_task = aggregate_intraday_results.s()

        # 2.3 构建 Chord 并异步执行
        # chord(header, body)
        workflow = chord(calculation_group, callback_task)
        workflow.apply_async()
        
        # ▼▼▼ 核心修改：不再有 .get()，任务启动后立即结束 ▼▼▼
        print(f"  -> Celery Chord 工作流已成功启动，包含 {len(job_packages)} 个计算任务。主任务退出，结果将由回调任务处理。")










