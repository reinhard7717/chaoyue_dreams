# 文件: services/realtime_services.py

import logging
import asyncio
import pandas as pd
import numpy as np
import pandas_ta as ta
from asgiref.sync import sync_to_async
from celery import group
from chaoyue_dreams.celery import app as celery_app
import pytz
from typing import List, Dict, Optional, Tuple
from datetime import datetime, time, date
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from utils.cache_manager import CacheManager
from stock_models.index import TradeCalendar
from utils.cash_key import StockCashKey

logger = logging.getLogger("services")

# ▼▼▼ 特征工程引擎类 ▼▼▼
class IntradayFeatureEngine:
    """
    【V11.0 - 真实逐笔驱动版】
    - 基于真实的逐笔成交数据 (realtime_tick) 进行特征计算。
    - 核心逻辑:
      1. 对逐笔数据进行主动性判断。
      2. 将逐笔数据聚合到分钟级别，计算出真正的“主动买卖量”。
      3. 将主动性指标与分钟K线数据融合。
      4. 计算基于净主动成交量的高级衍生指标 (Z-Score, Slope, Correlation)。
    """
    def __init__(self, slope_window: int = 5, stats_window: int = 20, corr_window: int = 10):
        """
        初始化特征引擎。
        Args:
            slope_window (int): 计算斜率的窗口期。
            stats_window (int): 计算统计指标 (如Z-Score, CV) 的窗口期。
            corr_window (int): 计算相关性的窗口期。
        """
        self.slope_window = slope_window
        self.stats_window = stats_window
        self.corr_window = corr_window
    def generate_features(self, stock_code: str, df_quotes: pd.DataFrame, df_real_ticks: Optional[pd.DataFrame], time_level: str) -> Optional[pd.DataFrame]:
        """
        执行完整的、基于真实逐笔数据的特征计算流水线。
        Args:
            stock_code (str): 股票代码。
            df_quotes (pd.DataFrame): 分钟级别的行情快照数据 (含OHLCV, VWAP等)。
            df_real_ticks (Optional[pd.DataFrame]): 当日开盘至今的全部真实逐笔数据。
            time_level (str): K线的时间级别，如 '1min'。
        Returns:
            Optional[pd.DataFrame]: 包含了所有计算特征的分钟级别DataFrame。
        """
        if df_quotes is None or df_quotes.empty:
            logger.warning(f"[{stock_code}] 输入的行情快照数据为空，特征计算终止。")
            return None
        # 1. 计算基础特征 (基于行情快照)
        df_minute = self._calculate_primary_features(df_quotes)
        # 2. 如果有真实逐笔数据，则计算主动性特征并融合
        if df_real_ticks is not None and not df_real_ticks.empty:
            df_agg_features = self._aggregate_tick_data(df_real_ticks, time_level)
            if df_agg_features is not None:
                # 使用 left join，以分钟K线为主体，合并主动性指标
                df_minute = df_minute.join(df_agg_features, how='left')
        # 3. 使用 pandas-ta 计算衍生技术指标
        df_minute = self._apply_pandas_ta_strategy(df_minute)
        # 4. 最终处理
        df_minute['stock_code'] = stock_code
        df_minute.reset_index(inplace=True)
        df_minute.rename(columns={'index': 'trade_time'}, inplace=True)
        # 填充由滚动窗口计算产生的初始NaN值
        df_minute.fillna(0, inplace=True)
        return df_minute
    def _calculate_primary_features(self, df_quotes: pd.DataFrame) -> pd.DataFrame:
        """计算基于分钟K线的基础特征，如VWAP。"""
        # 确保列是数值类型
        df_quotes['turnover_value'] = pd.to_numeric(df_quotes['turnover_value'], errors='coerce')
        df_quotes['volume'] = pd.to_numeric(df_quotes['volume'], errors='coerce')
        # 计算当日累计VWAP (Volume Weighted Average Price)
        df_quotes['vwap'] = df_quotes['turnover_value'].cumsum() / (df_quotes['volume'].cumsum() + 1e-9)
        return df_quotes
    def _aggregate_tick_data(self, df_ticks: pd.DataFrame, time_level: str) -> Optional[pd.DataFrame]:
        """
        对真实逐笔数据进行处理和分钟级聚合，计算主动性指标。
        """
        try:
            # 1. 判断主动性
            # '买盘' 表示主动买入，'卖盘' 表示主动卖出
            df_ticks['aggressive_buy_volume'] = np.where(df_ticks['type'] == '买盘', df_ticks['volume'], 0)
            df_ticks['aggressive_sell_volume'] = np.where(df_ticks['type'] == '卖盘', df_ticks['volume'], 0)
            # 2. 定义聚合规则
            aggregation_rules = {
                'aggressive_buy_volume': 'sum',
                'aggressive_sell_volume': 'sum',
            }
            # 3. 按分钟级别重新采样和聚合
            df_aggregated = df_ticks.resample(time_level).agg(aggregation_rules)
            # 4. 重命名列，使其更清晰
            df_aggregated.rename(columns={
                'aggressive_buy_volume': 'agg_buy_vol_sum',
                'aggressive_sell_volume': 'agg_sell_vol_sum'
            }, inplace=True)
            return df_aggregated
        except Exception as e:
            logger.error(f"聚合真实逐笔数据时发生异常: {e}", exc_info=True)
            return None
    def _apply_pandas_ta_strategy(self, df_minute: pd.DataFrame) -> pd.DataFrame:
        """
        使用 pandas-ta 库计算各种衍生技术指标。
        """
        # --- 基础量价指标 ---
        # 价格波动率 (Coefficient of Variation)
        stdev = df_minute['close'].rolling(window=self.stats_window, min_periods=1).std()
        sma = df_minute['close'].rolling(window=self.stats_window, min_periods=1).mean()
        df_minute['price_cv'] = stdev / (sma + 1e-9)
        # 成交量Z-Score
        if 'volume' in df_minute.columns:
            df_minute.ta.zscore(close=df_minute['volume'], length=self.stats_window, append=True, col_names="volume_zscore")
        # --- 主动性衍生指标 (如果存在) ---
        if 'agg_buy_vol_sum' in df_minute.columns and 'agg_sell_vol_sum' in df_minute.columns:
            # 填充可能因join产生的NaN
            df_minute['agg_buy_vol_sum'].fillna(0, inplace=True)
            df_minute['agg_sell_vol_sum'].fillna(0, inplace=True)
            # 1. 净主动成交量
            df_minute['net_aggressive_volume'] = df_minute['agg_buy_vol_sum'] - df_minute['agg_sell_vol_sum']
            # 2. 净主动成交量Z-Score
            df_minute.ta.zscore(close=df_minute['net_aggressive_volume'], length=self.stats_window, append=True, col_names="net_agg_vol_zscore")
            # 3. 净主动成交量斜率
            df_minute.ta.slope(close=df_minute['net_aggressive_volume'], length=self.slope_window, append=True, col_names="net_agg_vol_slope")
            # 4. 价格与净主动成交量的滚动相关性
            corr = df_minute['close'].rolling(window=self.corr_window).corr(df_minute['net_aggressive_volume'])
            df_minute['corr_price_net_agg_vol'] = corr
        return df_minute

# ▼▼▼ 修改后：重构为服务编排器 ▼▼▼
@celery_app.task(name='services.realtime_services.cpu_bound_calculation_task', queue='cpu_intensive_queue')
def cpu_bound_calculation_task(
    stock_data_package: Tuple[str, Optional[dict], Optional[dict], Optional[dict]],
    time_level: str,
    slope_window: int,
    stats_window: int
) -> None:
    """
    【V9.0 - 重构版】
    - 职责: 作为一个轻量级的服务编排器。
    - 行为: 1. 调用IntradayFeatureEngine执行计算。
             2. 将计算结果分派给下一个策略分析任务。
    """
    stock_code = stock_data_package[0]
    # print(f"    -> [WORKER V9.0] 开始处理 {stock_code}...")
    try:
        # 1. 初始化特征引擎
        feature_engine = IntradayFeatureEngine(slope_window, stats_window)
        # 2. 调用引擎生成所有特征
        df_features = feature_engine.generate_features(stock_data_package, time_level)
        # 3. 检查结果并触发下游任务
        if df_features is None or df_features.empty:
            print(f"    -> [WORKER V9.0] {stock_code} 未生成有效特征数据，任务结束。")
            return
        # print(f"    -> [WORKER V9.0] {stock_code} 特征计算完成，正在触发策略分析任务...")
        from tasks.stock_analysis_tasks import run_realtime_strategy_for_stock
        calculated_data = df_features.to_dict('records')
        run_realtime_strategy_for_stock.apply_async(
            args=[calculated_data],
            queue='cpu_intensive_queue',
            routing_key='cpu_intensive_queue'
        )
    except Exception as e:
        logger.error(f"    -> [WORKER V9.0] 处理 {stock_code} 时发生严重错误: {e}", exc_info=True)


class RealtimeServices:
    """
    【盘中引擎 - 服务层 V4.3 - 使用数据库交易日历】
    - 使用数据库中的 TradeCalendar 模型替代外部工具类。
    """
    def __init__(self, cache_manager_instance: CacheManager):
        self.cache_manager = cache_manager_instance
        self.realtime_dao = StockRealtimeDAO(cache_manager_instance)
        self.timetrade_dao = StockTimeTradeDAO(cache_manager_instance)
        self.cache_key = StockCashKey()
        self.feature_engine = IntradayFeatureEngine(slope_window=5, stats_window=20, corr_window=10)
        # 将窗口参数定义在服务实例中
        self.slope_window = 5
        self.stats_window = 20
    @sync_to_async
    def _get_monitoring_pool_from_sources(self, trade_date: date) -> tuple[list[str], list[str]]:
        """
        【内部方法】从数据库并发获取策略Top100股和所有自选股。
        (此方法保持不变)
        """
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
    async def update_and_cache_monitoring_pool(self):
        """
        【盘前任务入口 - V2.0】使用数据库中的TradeCalendar模型更新监控池。
        """
        print("开始更新盘中监控股票池...")
        get_prev_date_async = sync_to_async(TradeCalendar.get_latest_trade_date, thread_sensitive=True)
        previous_trade_date = await get_prev_date_async(reference_date=date.today())
        if not previous_trade_date:
            logger.error("无法从数据库TradeCalendar中获取到前一个交易日，盘前准备任务终止！")
            return
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
    async def get_monitoring_pool_from_cache(self) -> List[str]:
        """
        【盘中任务入口 - V2.1 修正版】从Redis中获取盘中监控的股票池。
        - 修正: 移除了对已反序列化字符串的多余 .decode() 操作。
        """
        redis_key = self.cache_key.intraday_monitoring_pool()
        stock_codes_from_cache = await self.cache_manager.smembers(redis_key)
        if stock_codes_from_cache is None:
            logger.error(f"从Redis缓存键 {redis_key} 获取股票池失败（连接错误？）。")
            return []
        if not stock_codes_from_cache:
            logger.warning(f"无法从Redis缓存键 {redis_key} 中获取到任何股票，监控池为空！")
            return []
        stock_codes = stock_codes_from_cache
        print(f"成功从Redis加载监控池，共 {len(stock_codes)} 支股票。")
        return stock_codes
    async def process_all_stocks_intraday_data(self, stock_codes: List[str], time_level: str, trade_date: str):
        """
        【V4.0 - 重构版】
        对给定的股票列表，执行完整的数据获取、特征计算和策略分析流程。
        """
        print(f"-> [服务层 V4.0] 开始处理 {len(stock_codes)} 支股票的盘中数据...")
        # 1. 并发获取所有需要的基础数据
        #    - 行情快照 (用于生成分钟K线)
        #    - 真实逐笔 (用于计算主动性指标)
        quotes_map, real_ticks_map = await self._get_all_base_data(stock_codes, trade_date)
        # 2. 准备Celery任务组
        calculation_tasks = []
        for stock_code in stock_codes:
            df_quotes = quotes_map.get(stock_code)
            df_real_ticks = real_ticks_map.get(stock_code)
            if df_quotes is None or df_quotes.empty:
                print(f"  - 警告: {stock_code} 缺少行情快照数据，无法进行分析，已跳过。")
                continue
            # 3. 调用特征引擎计算特征
            #    现在传递了所有必需的参数
            df_features = self.feature_engine.generate_features(
                stock_code=stock_code,
                df_quotes=df_quotes,
                df_real_ticks=df_real_ticks,
                time_level=time_level
            )
            if df_features is None or df_features.empty:
                print(f"  - 警告: {stock_code} 特征计算失败或结果为空，已跳过。")
                continue
            # 4. 将计算结果打包，准备发送给策略分析任务
            #    将DataFrame转换为list of dicts以便Celery传输
            calculated_data_list = df_features.to_dict('records')
            # 创建Celery任务签名
            task_signature = run_realtime_strategy_for_stock.s(calculated_data_list)
            calculation_tasks.append(task_signature)
        # 5. 并行执行所有策略分析任务
        if calculation_tasks:
            print(f"-> [服务层 V4.0] 准备将 {len(calculation_tasks)} 个分析任务分派到 'cpu_intensive_queue' 队列...")
            workflow = group(calculation_tasks)
            workflow.apply_async()
            print("-> [服务层 V4.0] 所有分析任务已成功分派。")
        else:
            print("-> [服务层 V4.0] 没有需要分析的任务。")
    async def _get_all_base_data(self, stock_codes: List[str], trade_date: str) -> tuple[dict, dict]:
        """
        并发获取所有股票的行情快照和真实逐笔数据。
        """
        print(f"  -> [数据获取] 开始为 {len(stock_codes)} 支股票并发获取基础数据...")
        # 使用 asyncio.gather 并发执行两个批量获取任务
        tasks = [
            self.realtime_dao.get_daily_quotes_and_level5_in_bulk(stock_codes, trade_date),
            self._get_all_real_ticks_in_bulk(stock_codes, trade_date)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 处理行情快照和Level5数据
        quotes_level5_map = results[0] if not isinstance(results[0], Exception) else {}
        quotes_map = {}
        if quotes_level5_map:
            for code, (df_quotes, df_level5) in quotes_level5_map.items():
                if df_quotes is not None and df_level5 is not None:
                    # 合并数据
                    quotes_map[code] = pd.merge_asof(df_quotes.sort_index(), df_level5.sort_index(), left_index=True, right_index=True, direction='backward')
                elif df_quotes is not None:
                    quotes_map[code] = df_quotes
        # 处理真实逐笔数据
        real_ticks_map = results[1] if not isinstance(results[1], Exception) else {}
        print(f"  -> [数据获取] 完成。获取到 {len(quotes_map)} 支股票的快照数据和 {len(real_ticks_map)} 支股票的逐笔数据。")
        return quotes_map, real_ticks_map
    async def _get_all_real_ticks_in_bulk(self, stock_codes: List[str], trade_date: str) -> Dict[str, pd.DataFrame]:
        """
        并发获取多只股票的真实逐笔数据。
        """
        tasks = [self.realtime_dao.get_daily_real_ticks(code, trade_date) for code in stock_codes]
        results = await asyncio.gather(*tasks)
        return {code: df for code, df in zip(stock_codes, results) if df is not None and not df.empty}




