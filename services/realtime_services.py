# 文件: services/realtime_services.py

import logging
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
import pytz # <--- 1. 导入 pytz
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time # <--- 2. 导入 time
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager

from concurrent.futures import ProcessPoolExecutor
import functools

logger = logging.getLogger("services")

def _cpu_bound_calculation_worker(
    stock_data_package: Tuple[str, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]],
    time_level: str,
    slope_window: int,
    stats_window: int
) -> Optional[pd.DataFrame]:
    """
    【并行计算工人】这是一个纯CPU计算函数，用于在独立的进程中处理单只股票的数据。
    它不包含任何I/O操作，因此可以完美地并行化。
    """
    stock_code, df_minute, df_ticks, df_level5 = stock_data_package

    if df_minute is None or df_minute.empty:
        # 在工作进程中打印日志可能不直接显示在主控台，但有助于调试
        print(f"    -> [Worker] 跳过 {stock_code}，因为没有分钟K线数据。")
        return None

    try:
        # --- 1. 聚合Tick数据 (复制自原_calculate_tick_level_indicators等方法的逻辑) ---
        if df_ticks is not None and not df_ticks.empty and 'buy_volume1' in df_ticks.columns:
            # 计算Tick级指标
            df_ticks['buy_com'] = df_ticks[['buy_volume1', 'buy_volume2', 'buy_volume3', 'buy_volume4', 'buy_volume5']].sum(axis=1)
            df_ticks['sell_com'] = df_ticks[['sell_volume1', 'sell_volume2', 'sell_volume3', 'sell_volume4', 'sell_volume5']].sum(axis=1)
            df_ticks['energy_ratio'] = (df_ticks['buy_com'] / (df_ticks['sell_com'] + 1e-6)).clip(0, 100)
            df_ticks['aggressive_buy_volume'] = df_ticks.apply(lambda row: row['volume'] if row['current_price'] >= row['sell_price1'] else 0, axis=1)
            df_ticks['aggressive_sell_volume'] = df_ticks.apply(lambda row: row['volume'] if row['current_price'] <= row['buy_price1'] else 0, axis=1)
            
            # 定义聚合规则
            aggregation_rules = {
                'buy_com': ['mean'], 'sell_com': ['mean'], 'energy_ratio': ['mean', 'max', 'min'],
                'aggressive_buy_volume': ['sum'], 'aggressive_sell_volume': ['sum'],
                'volume': ['sum', 'count']
            }
            
            # 重采样和聚合
            df_aggregated = df_ticks.resample(time_level).agg(aggregation_rules)
            df_aggregated.columns = ['_'.join(col).strip() for col in df_aggregated.columns.values]
            
            # 重命名
            df_aggregated.rename(columns={
                'buy_com_mean': 'buy_com_mean', 'sell_com_mean': 'sell_com_mean',
                'energy_ratio_mean': 'energy_ratio_mean', 'energy_ratio_max': 'energy_ratio_max',
                'energy_ratio_min': 'energy_ratio_min', 'aggressive_buy_volume_sum': 'agg_buy_vol_sum',
                'aggressive_sell_volume_sum': 'agg_sell_vol_sum', 'volume_sum': 'agg_volume',
                'volume_count': 'tick_count',
            }, inplace=True)
            
            # 合并到分钟线
            df_minute = df_minute.join(df_aggregated, how='left')

        # --- 2. 计算各种技术指标 (复制自原_calculate_advanced_features_with_ta的逻辑) ---
        if 'turnover_value' in df_minute.columns and 'volume' in df_minute.columns:
            df_minute['vwap'] = df_minute['turnover_value'].cumsum() / (df_minute['volume'].cumsum() + 1e-6)
        
        if df_minute.empty: return df_minute
        
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
            pass # 在worker中静默处理分形指标的异常
            
        return df_minute

    except Exception as e:
        print(f"    -> [Worker] 处理 {stock_code} 时发生错误: {e}")
        return None

class RealtimeServices:
    """
    【盘中引擎 - 服务层 V4.1 - 最终形态】
    - 核心能力: 将原始Tick和分钟K线，通过深度解析、聚合和高级数学计算，
                转化为一个包含力学、统计学、分形等多维度特征的战术情报矩阵。
    - 技术栈: Pandas, Numpy, Pandas-TA
    """
    def __init__(self, cache_manager_instance: CacheManager):
        # 【核心修复】接收 cache_manager_instance
        
        # 使用传入的实例来创建 DAO
        self.realtime_dao = StockRealtimeDAO(cache_manager_instance)
        self.timetrade_dao = StockTimeTradeDAO(cache_manager_instance)
        self.slope_window = 5
        self.stats_window = 20

    async def process_all_stocks_intraday_data(self, stock_codes: list, time_level: str, trade_date: str):
        """
        【V5.0 - 多进程并行计算版】
        此方法利用 ProcessPoolExecutor 将 CPU 密集型计算分发到多个核心，大幅提升处理速度。
        """
        print(f"【V5.0】开始为 {len(stock_codes)} 支股票并行处理盘中数据...")
        
        # --- 步骤 1: I/O 阶段 - 集中获取所有原始数据 ---
        print("  -> 阶段 1/3: 正在集中获取所有股票的原始数据...")
        
        # 1.1 批量获取所有 Ticks 和 Level5 数据
        all_bulk_data = await self.realtime_dao.get_daily_ticks_and_level5_in_bulk(stock_codes, trade_date)
        if not all_bulk_data:
            logger.warning(f"未能从缓存中批量获取到任何股票的 Ticks/Level5 数据，日期: {trade_date}。任务终止。")
            return

        # 1.2 异步并发获取所有分钟K线数据
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        target_date_obj = datetime.strptime(trade_date, '%Y-%m-%d').date()
        start_dt_aware = shanghai_tz.localize(datetime.combine(target_date_obj, time(9, 25, 0)))
        end_dt_aware = shanghai_tz.localize(datetime.combine(target_date_obj, time(15, 5, 0)))

        kline_tasks = [
            self.timetrade_dao.get_minute_kline_by_daterange(code, time_level, start_dt_aware, end_dt_aware)
            for code in stock_codes
        ]
        all_minute_klines_results = await asyncio.gather(*kline_tasks, return_exceptions=True)
        
        # 1.3 准备计算任务的数据包
        job_packages = []
        for i, code in enumerate(stock_codes):
            df_minute = all_minute_klines_results[i]
            if isinstance(df_minute, Exception) or df_minute is None or df_minute.empty:
                continue # 如果分钟线获取失败或为空，则跳过该股票

            bulk_data = all_bulk_data.get(code)
            df_ticks, df_level5 = (bulk_data[0], bulk_data[1]) if bulk_data else (None, None)
            
            # 将所有需要的数据打包成一个元组
            job_packages.append((code, df_minute, df_ticks, df_level5))
        
        print(f"  -> 数据获取完成，准备将 {len(job_packages)} 个有效计算任务提交到进程池。")

        # --- 步骤 2: CPU 计算阶段 - 使用进程池并行处理 ---
        print(f"  -> 阶段 2/3: 正在将计算任务提交到多核CPU进程池...")
        
        loop = asyncio.get_running_loop()
        # 使用 with 语句确保进程池被正确关闭
        with ProcessPoolExecutor() as executor:
            # 使用 functools.partial 预先绑定不变的参数
            worker_func = functools.partial(
                _cpu_bound_calculation_worker,
                time_level=time_level,
                slope_window=self.slope_window,
                stats_window=self.stats_window
            )
            
            # run_in_executor 可以在 asyncio 事件循环中运行阻塞的、CPU密集型的代码
            # executor.map 会将 job_packages 中的每个元素作为参数传递给 worker_func
            results = await loop.run_in_executor(
                executor,
                list, # 将 map 的结果转换为列表
                map(worker_func, job_packages)
            )
        
        print(f"  -> 所有 {len(results)} 个计算任务已在子进程中完成。")

        # --- 步骤 3: 结果处理阶段 ---
        print(f"  -> 阶段 3/3: 正在处理计算结果...")
        processed_count = 0
        for result_df in results:
            if result_df is not None and not result_df.empty:
                # 在这里，你可以对计算完成的 DataFrame 进行后续操作
                # 例如：保存到数据库、推送到消息队列等
                # 为了保持和之前逻辑一致，我们暂时只打印信息
                stock_code = result_df['stock_code'].iloc[0]
                print(f"    -> 成功生成股票 {stock_code} 的盘中数据矩阵，共 {len(result_df)} 条。")
                processed_count += 1
        
        print(f"【V5.0】所有股票的盘中数据并行处理完成，成功处理了 {processed_count} / {len(job_packages)} 支股票。")













