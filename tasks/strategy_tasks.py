# tasks/strategy_tasks.py

# --- 移除不再需要的 async_to_sync 导入 ---
# from asgiref.sync import async_to_sync
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app
import pandas as pd
# ... 其他导入 ...
from services.indicator_services import IndicatorService
from strategies.base import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_BUY, SIGNAL_STRONG_SELL
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollStrategy
from utils.cache_set import StrategyCacheSet
from dao_manager.daos.stock_basic_dao import StockBasicDAO # 假设主任务需要

logger = logging.getLogger('strategy') # 或者你使用的 logger 名称

# --- 主任务保持 async def ---
@celery_app.task(bind=True, name='tasks.strategy.run_macd_rsi_kdj_boll_strategy_for_stock_main')
async def run_macd_rsi_kdj_boll_strategy_main_task(self):
    logger.info("开始执行策略信号计算主任务")
    stock_basic_dao = StockBasicDAO()
    favorite_stocks = []
    stocks = []
    try:
        # 假设 DAO 方法是异步的
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        logger.info(f"获取到 {len(favorite_stocks)} 只自选股")
        stocks = await stock_basic_dao.get_all_stocks()
        logger.info(f"获取到 {len(stocks)} 只股票")
    except Exception as e:
        logger.error(f"获取股票列表时出错: {e}", exc_info=True)
        return # 获取列表失败则不继续

    favorite_codes = {fs.stock_code for fs in favorite_stocks}

    for favorite_stock in favorite_stocks:
        logger.debug(f"处理自选股: {favorite_stock.stock_code}")
        try:
            # 使用 await 调用处理单个股票的异步函数
            await strategy_macd_rsi_kdj_boll_strategy_for_stock(favorite_stock.stock_code)
        except Exception as e_inner:
             logger.error(f"处理股票 {favorite_stock.stock_code} 时发生未捕获异常: {e_inner}", exc_info=True)


    for stock in stocks:
        if stock.stock_code not in favorite_codes:
            logger.debug(f"处理普通股票: {stock.stock_code}")
            try:
                await strategy_macd_rsi_kdj_boll_strategy_for_stock(stock.stock_code)
            except Exception as e_inner:
                 logger.error(f"处理股票 {stock.stock_code} 时发生未捕获异常: {e_inner}", exc_info=True)
        # else: # 跳过已处理的自选股逻辑是隐式的
        #     logger.debug(f"跳过已处理的自选股: {stock.stock_code}")

    # 关闭 DAO (如果需要且方法存在)
    if hasattr(stock_basic_dao, 'close') and callable(stock_basic_dao.close):
        try:
            # 假设 close 是同步的
            stock_basic_dao.close()
            logger.info("StockBasicDAO closed.")
        except Exception as close_err:
            logger.error(f"关闭 StockBasicDAO 时出错: {close_err}")

    logger.info("所有股票的macd_rsi_kdj_boll策略信号计算任务完成")


# --- 单个股票处理函数保持 async def ---
async def strategy_macd_rsi_kdj_boll_strategy_for_stock(stock_code: str):
    # --- 实例化部分保持不变 ---
    logger.info(f"开始为股票 {stock_code} 运行 MACD+RSI+KDJ+BOLL 策略")
    service = None
    strategy = None
    cache_setter = None
    merged_data = None

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

        if not all([service, strategy, cache_setter]):
             logger.error(f"[{stock_code}] 某个核心对象未能成功实例化，任务终止。")
             return # 返回 None 或其他表示失败的值

        # --- 第一个 try 块：获取数据并处理 ---
        logger.info(f"[{stock_code}] 准备调用 service.prepare_strategy_dataframe...")
        # --- !!! 修改点：将 async_to_sync 改为 await !!! ---
        merged_data = await service.prepare_strategy_dataframe(
            stock_code=stock_code,
            timeframes=strategy.timeframes, # 使用 strategy 实例的 timeframes
            strategy_params=strategy.params,
            limit_per_tf=1500
        )
        # --- 结束修改点 ---
        logger.info(f"[{stock_code}] service.prepare_strategy_dataframe 调用完成.")

        # --- 后续逻辑基于第一个 try 块获取的 merged_data ---
        if merged_data is None or merged_data.empty:
            logger.warning(f"[{stock_code}] 未能准备策略所需数据，策略无法运行。")
            # 缓存空信号状态 (使用 await)
            try:
                latest_timestamp_ref = pd.Timestamp.now(tz='UTC')
                signal_data_to_cache = {
                    'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                    'time_level': main_timeframe, 'timestamp': latest_timestamp_ref.isoformat(),
                    'signal': None, 'signal_display': 'No Data', 'score': None,
                    'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
                }
                cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                    stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
                )
                if cache_success: logger.info(f"[{stock_code}] 数据准备失败，已缓存空信号状态。")
                else: logger.warning(f"[{stock_code}] 缓存空信号状态失败。")
            except Exception as cache_err: logger.error(f"[{stock_code}] 缓存空信号状态时发生错误: {cache_err}", exc_info=False)
            return # 返回 None 或其他表示失败的值

        # 运行策略 (同步调用)
        logger.info(f"[{stock_code}] 准备运行策略...")
        signal_series = strategy.run(merged_data)
        logger.info(f"[{stock_code}] 策略运行完成.")

        # 处理信号结果 (逻辑保持不变)
        latest_signal = None
        latest_timestamp = None
        signal_display = "No Signal"
        if not signal_series.empty:
            valid_signals = signal_series.dropna()
            if not valid_signals.empty:
                latest_signal = valid_signals.iloc[-1]
                latest_timestamp = valid_signals.index[-1]
                signal_map = {
                    SIGNAL_STRONG_BUY: "Strong Buy", 
                    SIGNAL_BUY: "Buy",
                    SIGNAL_HOLD: "Hold", 
                    SIGNAL_SELL: "Sell",
                    SIGNAL_STRONG_SELL: "Strong Sell",
                }
                signal_display = signal_map.get(latest_signal, "Unknown")
            else:
                 logger.info(f"[{stock_code}] 策略运行完成，但所有信号均为 NaN。")
                 signal_display = "No Signal (NaN)"
                 latest_timestamp = merged_data.index[-1]
        else:
            logger.info(f"[{stock_code}] 策略运行完成，但未生成信号序列。")
            signal_display = "No Signal (Empty Series)"
            latest_timestamp = merged_data.index[-1]

        # 缓存结果 (使用 await)
        logger.info(f"[{stock_code}] 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
        try:
            signal_data_to_cache = {
                'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                'time_level': main_timeframe,
                'timestamp': latest_timestamp.isoformat() if latest_timestamp else pd.Timestamp.now(tz='UTC').isoformat(),
                'signal': int(latest_signal) if latest_signal is not None and not pd.isna(latest_signal) else None,
                'signal_display': signal_display,
                'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
            }
            cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
            )
            if cache_success: logger.info(f"[{stock_code}] 策略结果/状态成功缓存到 Redis。")
            else: logger.warning(f"[{stock_code}] 缓存策略结果/状态到 Redis 失败。")
        except Exception as cache_err:
            logger.error(f"[{stock_code}] 缓存策略结果/状态时发生错误: {cache_err}", exc_info=True)

    except Exception as e:
        logger.error(f"为股票 {stock_code} 处理策略时发生严重错误: {e}", exc_info=True)
        # 这里可以选择是否返回错误状态，或者让主任务继续处理其他股票
        # return f"[{stock_code}] 策略运行出错: {e}" # 可以取消注释以返回错误信息

    # --- !!! 移除重复的 try...except 块 !!! ---
    # 原来从这里开始的第二个 try 块是重复且错误的，已删除

# ... (文件末尾的其他任务定义) ...
