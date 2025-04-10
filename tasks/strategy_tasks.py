# tasks/strategy_tasks.py

# --- 移除 async_to_sync 导入 ---
# from asgiref.sync import async_to_sync
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app
import pandas as pd
# ... 其他导入 ...
from services.indicator_services import IndicatorService
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollStrategy
from utils.cache_set import StrategyCacheSet

logger = logging.getLogger('strategy') # 或者你使用的 logger 名称

@celery_app.task(bind=True, name='tasks.strategy.run_macd_rsi_kdj_boll_strategy_for_stock_main') # 建议给 Celery 任务一个不同的名字
async def run_macd_rsi_kdj_boll_strategy_main_task(self): # 函数名也建议区分
    """
    定时任务：计算策略信号 (主任务，分发给单个股票处理)
    """
    logger.info("开始执行策略信号计算主任务")
    # --- 确保 DAO 在异步函数内正确使用 ---
    # 如果 StockBasicDAO 的方法是异步的，需要 await
    # 如果是同步的，需要用 async_to_sync 包装或确保它们是线程安全的
    # 假设 get_all_favorite_stocks 和 get_all_stocks 是异步的
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    stock_basic_dao = StockBasicDAO() # 假设 DAO 实例化是安全的

    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        logger.info(f"获取到 {len(favorite_stocks)} 只自选股")
        for favorite_stock in favorite_stocks:
            logger.debug(f"处理自选股: {favorite_stock.stock_code}")
            # 直接 await 调用处理单个股票的异步函数
            await process_single_stock_strategy(favorite_stock.stock_code)

        stocks = await stock_basic_dao.get_stock_list()
        logger.info(f"获取到 {len(stocks)} 只股票")
        # --- 优化：避免重复计算自选股 ---
        favorite_codes = {fs.stock_code for fs in favorite_stocks}
        for stock in stocks:
            if stock.stock_code not in favorite_codes:
                 logger.debug(f"处理普通股票: {stock.stock_code}")
                 await process_single_stock_strategy(stock.stock_code)
            else:
                 logger.debug(f"跳过已处理的自选股: {stock.stock_code}")
        # --- 结束优化 ---

    except Exception as e:
         logger.error(f"获取股票列表或分发任务时出错: {e}", exc_info=True)
    finally:
        # 如果 DAO 需要关闭
        if hasattr(stock_basic_dao, 'close') and callable(stock_basic_dao.close):
             # 假设 close 是同步的，如果异步需要 await
             try:
                 stock_basic_dao.close()
                 logger.info("StockBasicDAO closed.")
             except Exception as close_err:
                 logger.error(f"关闭 StockBasicDAO 时出错: {close_err}")

    logger.info("所有股票的macd_rsi_kdj_boll策略信号计算任务完成")


# --- 将核心逻辑放在一个独立的 async 函数中 ---
async def process_single_stock_strategy(stock_code: str):
    """处理单个股票的策略计算和缓存"""
    logger.info(f"开始为股票 {stock_code} 运行 MACD+RSI+KDJ+BOLL 策略")

    service = None
    strategy = None
    cache_setter = None
    merged_data = None # 初始化 merged_data

    try:
        logger.info(f"[{stock_code}] 准备实例化 IndicatorService...")
        service = IndicatorService()
        logger.info(f"[{stock_code}] IndicatorService 实例化完成.")

        logger.info(f"[{stock_code}] 准备实例化 MacdRsiKdjBollStrategy...")
        strategy = MacdRsiKdjBollStrategy()
        logger.info(f"[{stock_code}] MacdRsiKdjBollStrategy 实例化完成.")

        logger.info(f"[{stock_code}] 准备实例化 StrategyCacheSet...")
        cache_setter = StrategyCacheSet()
        logger.info(f"[{stock_code}] StrategyCacheSet 实例化完成.")

        main_timeframe = '15'
        strategy_timeframes = ['5', '15', '30', '60'] # 从 strategy 实例获取更佳

        if not all([service, strategy, cache_setter]):
             logger.error(f"[{stock_code}] 某个核心对象未能成功实例化，任务终止。")
             return # 直接返回，不再继续处理此股票

        # --- 1. 准备数据 ---
        logger.info(f"[{stock_code}] 准备调用 service.prepare_strategy_dataframe...")
        # --- 修改点：使用 await ---
        merged_data = await service.prepare_strategy_dataframe(
            stock_code=stock_code,
            timeframes=strategy.timeframes, # 使用 strategy 实例的 timeframes
            strategy_params=strategy.params,
            limit_per_tf=1500
        )
        # --- 结束修改点 ---
        logger.info(f"[{stock_code}] service.prepare_strategy_dataframe 调用完成.")

        if merged_data is None or merged_data.empty:
            logger.warning(f"[{stock_code}] 未能准备策略所需数据，策略无法运行。")
            # --- 修改点：缓存空信号时也使用 await ---
            try:
                latest_timestamp_ref = pd.Timestamp.now(tz='UTC') # 无数据时用当前时间
                signal_data_to_cache = {
                    'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                    'time_level': main_timeframe, 'timestamp': latest_timestamp_ref.isoformat(),
                    'signal': None, 'signal_display': 'No Data', 'score': None, # 明确是无数据
                    'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
                }
                cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                    stock_code=stock_code,
                    time_level=main_timeframe,
                    data_to_cache=signal_data_to_cache
                )
                if cache_success: logger.info(f"[{stock_code}] 数据准备失败，已缓存空信号状态。")
                else: logger.warning(f"[{stock_code}] 缓存空信号状态失败。")
            except Exception as cache_err: logger.error(f"[{stock_code}] 缓存空信号状态时发生错误: {cache_err}", exc_info=False)
            # --- 结束修改点 ---
            return # 不再继续处理此股票

        # --- 2. 运行策略 ---
        logger.info(f"[{stock_code}] 准备运行策略...")
        # 策略的 run 方法是同步的，不需要 await
        signal_series = strategy.run(merged_data)
        logger.info(f"[{stock_code}] 策略运行完成.")

        latest_signal = None
        latest_timestamp = None
        signal_display = "No Signal" # 默认值

        if not signal_series.empty:
            valid_signals = signal_series.dropna()
            if not valid_signals.empty:
                latest_signal = valid_signals.iloc[-1]
                latest_timestamp = valid_signals.index[-1]
                signal_map = {
                    strategy.SIGNAL_STRONG_BUY: "Strong Buy", strategy.SIGNAL_BUY: "Buy",
                    strategy.SIGNAL_HOLD: "Hold", strategy.SIGNAL_SELL: "Sell",
                    strategy.SIGNAL_STRONG_SELL: "Strong Sell",
                }
                signal_display = signal_map.get(latest_signal, "Unknown")
            else:
                 logger.info(f"[{stock_code}] 策略运行完成，但所有信号均为 NaN。")
                 signal_display = "No Signal (NaN)"
                 latest_timestamp = merged_data.index[-1] # 使用数据最后时间戳
        else:
            logger.info(f"[{stock_code}] 策略运行完成，但未生成信号序列。")
            signal_display = "No Signal (Empty Series)"
            latest_timestamp = merged_data.index[-1] # 使用数据最后时间戳

        # --- 3. 缓存结果 (无论是否有有效信号，都缓存当前状态) ---
        logger.info(f"[{stock_code}] 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
        try:
            signal_data_to_cache = {
                'stock_code': stock_code,
                'strategy_name': strategy.strategy_name,
                'time_level': main_timeframe,
                'timestamp': latest_timestamp.isoformat() if latest_timestamp else pd.Timestamp.now(tz='UTC').isoformat(), # 处理 None
                'signal': int(latest_signal) if latest_signal is not None and not pd.isna(latest_signal) else None,
                'signal_display': signal_display,
                'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
            }
            # --- 修改点：使用 await ---
            cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                stock_code=stock_code,
                time_level=main_timeframe,
                data_to_cache=signal_data_to_cache
            )
            # --- 结束修改点 ---
            if cache_success: logger.info(f"[{stock_code}] 策略结果/状态成功缓存到 Redis。")
            else: logger.warning(f"[{stock_code}] 缓存策略结果/状态到 Redis 失败。")
        except Exception as cache_err:
            logger.error(f"[{stock_code}] 缓存策略结果/状态时发生错误: {cache_err}", exc_info=True)

        # --- 4. TODO: 其他信号处理逻辑 (数据库、通知等) ---

    except Exception as e:
        logger.error(f"为股票 {stock_code} 处理策略时发生严重错误: {e}", exc_info=True)
        # 这里可以选择是否返回错误状态，或者让主任务继续处理其他股票

# ... (文件末尾的其他任务定义，如果它们也需要调用异步代码，同样需要改为 async def 和 await) ...
