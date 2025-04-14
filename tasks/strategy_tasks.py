# tasks/strategy_tasks.py

import asyncio  # 保留异步支持
import logging
from celery import shared_task  # 使用 shared_task 以确保兼容
from chaoyue_dreams.celery import app as celery_app  # 假设这是您的Celery app
import pandas as pd
from channels.layers import get_channel_layer
from channels.db import database_sync_to_async  # 用于同步 ORM 查询
from services.indicator_services import IndicatorService
from strategies.base import SIGNAL_BUY, SIGNAL_HOLD, SIGNAL_SELL, SIGNAL_STRONG_BUY, SIGNAL_STRONG_SELL
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy
from utils.cache_set import StrategyCacheSet
from dao_manager.daos.stock_basic_dao import StockBasicDAO  # 假设主任务需要

logger = logging.getLogger('tasks')  # 或者你使用的 logger 名称

# 自选股队列
FAVORITE_CALCULATE_STRATEGY_QUEUE = 'favorite_calculate_strategy'
# 非自选股队列
STOCKS_CALCULATE_STRATEGY_QUEUE = 'calculate_strategy'

@database_sync_to_async  # 将同步的 ORM 查询包装成异步
def _get_favorited_user_ids(self, stock_id: int) -> list[int]:
    """根据股票 ID 获取关注该股票的所有用户 ID 列表"""
    from users.models import FavoriteStock
    return list(FavoriteStock.objects.filter(stock_id=stock_id).values_list('user_id', flat=True))

async def _run_strategy_for_single_stock_task(self, stock_code: str):
    """
    为单支股票执行 MACD+RSI+KDJ+BOLL 策略计算并缓存结果。
    这是实际执行策略计算的 Celery Worker 任务。
    """
    stock_dao = StockBasicDAO()
    stock = await stock_dao.get_stock_by_code(stock_code)
    log_prefix = f"[StrategyTask] [{stock}]"
    # logger.info(f"{log_prefix} 开始执行 MACD+RSI+KDJ+BOLL 策略计算任务")
    service = None
    strategy = None
    cache_setter = None
    merged_data = None
    main_timeframe = '15'  # 或者从配置/策略对象获取
    try:
        # 1. 实例化所需对象
        service = IndicatorService()
        strategy = MacdRsiKdjBollEnhancedStrategy()  # 假设这个策略的时间框架和参数是固定的
        cache_setter = StrategyCacheSet()
        if not all([service, strategy, cache_setter]):
            logger.error(f"{log_prefix} 某个核心对象未能成功实例化，任务终止。")
            return {'status': 'failed', 'error': 'Core object instantiation failed'}  # 返回错误字典
        # 2. 准备数据
        logger.debug(f"{log_prefix} 准备策略数据...")
        merged_data = await service.prepare_strategy_dataframe(
            stock_code=stock_code,
            timeframes=strategy.timeframes,
            strategy_params=strategy.params,
            limit_per_tf=1500  # 保持与原逻辑一致
        )
        # 3. 处理数据准备失败的情况
        if merged_data is None or merged_data.empty:
            logger.warning(f"{log_prefix} 未能准备策略所需数据，策略无法运行。将缓存空信号状态。")
            try:
                latest_timestamp_ref = pd.Timestamp.now(tz='UTC')  # 使用 UTC 时间戳
                signal_data_to_cache = {
                    'stock_code': stock_code, 'strategy_name': strategy.strategy_name,
                    'time_level': main_timeframe, 'timestamp': latest_timestamp_ref.isoformat(),
                    'signal': None, 'signal_display': 'No Data', 'score': None,  # 假设没有 score
                    'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
                }
                # 使用 await 调用异步缓存方法
                cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                    stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
                )
                if cache_success:
                    logger.info(f"{log_prefix} 数据准备失败，已缓存空信号状态。")
                else:
                    logger.warning(f"{log_prefix} 缓存空信号状态失败。")
            except Exception as cache_err:
                logger.error(f"{log_prefix} 缓存空信号状态时发生错误: {cache_err}", exc_info=False)
            return {'status': 'success', 'message': 'Data preparation failed, cached empty signal'}  # 返回结果
        # 4. 运行策略 (策略的 run 方法本身是同步的，但我们在这里处理)
        logger.debug(f"{log_prefix} 运行策略计算...")
        signal_series = strategy.run(merged_data)  # 假设 run 是同步的
        # 5. 处理和解析信号结果
        latest_signal = None
        latest_timestamp = None
        signal_display = "No Signal"  # 默认值
        if signal_series is not None and not signal_series.empty:
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
                signal_display = signal_map.get(latest_signal, "Unknown Signal Value")
                if signal_display == 'Hold':
                    # logger.info(f"{log_prefix} 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
                    pass
                else:
                    logger.info(f"{log_prefix} 策略信号 @ {latest_timestamp}: {latest_signal} ({signal_display})")
            else:
                # logger.info(f"{log_prefix} 策略运行完成，但所有信号均为 NaN。")
                signal_display = "No Signal (NaN)"
                if not merged_data.empty:
                    latest_timestamp = merged_data.index[-1]
                else:
                    latest_timestamp = pd.Timestamp.now(tz='UTC')
        else:
            # logger.info(f"{log_prefix} 策略运行完成，但未生成有效信号序列。")
            signal_display = "No Signal (Empty Series)"
            if not merged_data.empty:
                latest_timestamp = merged_data.index[-1]
            else:
                latest_timestamp = pd.Timestamp.now(tz='UTC')
        # 6. 缓存策略结果
        logger.debug(f"{log_prefix} 准备缓存策略结果...")
        try:
            if latest_timestamp and latest_timestamp.tzinfo is None:
                latest_timestamp = latest_timestamp.tz_localize('Asia/Shanghai').tz_convert('UTC')
            elif latest_timestamp is None:
                latest_timestamp = pd.Timestamp.now(tz='UTC')
            signal_data_to_cache = {
                'stock_code': stock_code,
                'strategy_name': strategy.strategy_name,
                'time_level': main_timeframe,
                'timestamp': latest_timestamp.isoformat(),
                'signal': int(latest_signal) if latest_signal is not None and not pd.isna(latest_signal) else None,
                'signal_display': signal_display,
                'generated_at': pd.Timestamp.now(tz='UTC').isoformat()
            }
            cache_success = await cache_setter.macd_rsi_kdj_boll_data(
                stock_code=stock_code, time_level=main_timeframe, data_to_cache=signal_data_to_cache
            )
            if cache_success:
                # logger.debug(f"{log_prefix} 策略结果/状态成功缓存到 Redis。")
                if signal_data_to_cache.get('signal_display') == 'Hold':
                    pass
                else:
                    # 1. 获取关注该股票的所有用户 ID 列表
                    # 2. 将信号数据推送给所有关注该股票的用户
                    pass
            else:
                logger.warning(f"{log_prefix} 缓存策略结果/状态到 Redis 失败。")
        except Exception as cache_err:
            logger.error(f"{log_prefix} 缓存策略结果/状态时发生错误: {cache_err}", exc_info=True)
    except Exception as e:
        logger.error(f"{log_prefix} 处理策略时发生严重错误: {e}", exc_info=True)
        return {'status': 'failed', 'error': str(e)}
    finally:
        # logger.info(f"{log_prefix} 策略计算任务执行完毕。")
        return {'status': 'success'}  # 返回成功状态

# Celery 任务：对单个股票执行策略计算。
@celery_app.task(bind=True, name='tasks.stock_processing.run_stock_strategy')
def run_stock_strategy_task(self, stock_code: str):
    """
    Celery 任务：对单个股票执行策略计算。
    假定上一个任务成功时会传递 stock_code。
    """
    if not stock_code:
         logger.warning(f"任务跳过 (策略计算): run_stock_strategy_task - 未收到有效的 stock_code (可能前序任务失败)")
         return None
    queue_name = self.request.delivery_info.get('routing_key', '未知')
    # logger.info(f"任务启动 (策略计算): run_stock_strategy_task - 处理股票 {stock_code} (队列: {queue_name})")
    async def _run_async_strategy():
        try:
            # logger.debug(f"策略计算: 开始运行 {stock_code} 的策略...")
            await _run_strategy_for_single_stock_task(stock_code)
            # logger.debug(f"策略计算: 完成运行 {stock_code} 的策略。")
            return True
        except Exception as e:
            logger.error(f"策略计算: 运行股票 {stock_code} 策略时出错: {e}", exc_info=True)
            return False
    try:
        success = asyncio.run(_run_async_strategy()) # 假设策略函数是异步的
        if success:
            # logger.info(f"任务成功 (策略计算): run_stock_strategy_task - 完成处理股票 {stock_code}")
            return f"Strategy calculation completed for {stock_code}" # 链的最终结果
        else:
            logger.error(f"任务失败 (策略计算): run_stock_strategy_task - 处理股票 {stock_code} 失败")
            raise Exception(f"Failed to run strategy for {stock_code}")
    except Exception as e:
        logger.error(f"执行 run_stock_strategy_task (同步包装器) 时出错: {e}", exc_info=True)
        raise

# Celery 任务：对单个股票执行策略计算。
@celery_app.task(bind=True, name='tasks.stock_processing.run_stock_strategy')
def run_stock_strategy_task(self, stock_code: str):
    """
    Celery 任务：对单个股票执行策略计算。
    假定上一个任务成功时会传递 stock_code。
    """
    if not stock_code:
         logger.warning(f"任务跳过 (策略计算): run_stock_strategy_task - 未收到有效的 stock_code (可能前序任务失败)")
         return None
    queue_name = self.request.delivery_info.get('routing_key', '未知')
    # logger.info(f"任务启动 (策略计算): run_stock_strategy_task - 处理股票 {stock_code} (队列: {queue_name})")
    async def _run_async_strategy():
        try:
            # logger.debug(f"策略计算: 开始运行 {stock_code} 的策略...")
            await _run_strategy_for_single_stock_task(stock_code)
            # logger.debug(f"策略计算: 完成运行 {stock_code} 的策略。")
            return True
        except Exception as e:
            logger.error(f"策略计算: 运行股票 {stock_code} 策略时出错: {e}", exc_info=True)
            return False
    try:
        success = asyncio.run(_run_async_strategy()) # 假设策略函数是异步的
        if success:
            # logger.info(f"任务成功 (策略计算): run_stock_strategy_task - 完成处理股票 {stock_code}")
            return f"Strategy calculation completed for {stock_code}" # 链的最终结果
        else:
            logger.error(f"任务失败 (策略计算): run_stock_strategy_task - 处理股票 {stock_code} 失败")
            raise Exception(f"Failed to run strategy for {stock_code}")
    except Exception as e:
        logger.error(f"执行 run_stock_strategy_task (同步包装器) 时出错: {e}", exc_info=True)
        raise


# --- 异步辅助函数：获取需要处理的股票代码 (区分自选和非自选) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    stock_basic_dao = StockBasicDAO()
    favorite_stock_codes = set()
    all_stock_codes = set()

    # 获取自选股
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        for fav in favorite_stocks:
            favorite_stock_codes.add(fav.stock_id)
        logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)

    # 获取所有A股 (或者你需要的范围)
    try:
        # 注意：如果 get_stock_list() 返回大量数据，考虑分页或流式处理
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            all_stock_codes.add(stock.stock_code)
        logger.info(f"获取到 {len(all_stock_codes)} 个全市场股票代码")
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)

    # 计算非自选股代码 (在所有代码中，但不在自选代码中)
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes) # 转换为列表

    total_unique_stocks = len(favorite_stock_codes) + len(non_favorite_stock_codes)
    # logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
         logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes

# 任务调度：计算所有股票的指标
@celery_app.task(bind=True, name='tasks.strategy_tasks.calculate_stock_strategy')
def calculate_stock_strategy(self):
    """
    修改后的调度器任务：
    1. 获取自选股和非自选股代码。
    2. 为每只股票创建任务链 (获取数据 -> 计算指标 -> 执行策略)，并分派到指定的队列。
    3. 将自选股任务分派到 FAVORITE_CALCULATE_STRATEGY_QUEUE 队列。
    4. 将非自选股任务分派到 STOCKS_CALCULATE_STRATEGY_QUEUE 队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: calculate_stock_strategy (调度器模式) - 获取股票列表并分派细粒度任务链")
    try:
        # 在同步任务中运行异步代码来获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return "未获取到股票代码"

        total_dispatched_chains = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股任务链到 FAVORITE_CALCULATE_STRATEGY_QUEUE 队列
        for stock_code in favorite_codes:
            sig = run_stock_strategy_task.s(stock_code).set(queue='FAVORITE_CALCULATE_STRATEGY_QUEUE')
            sig.apply_async()  # 分派任务
            total_dispatched_chains += 1  # 计数分派的任务

        # 2. 分派非自选股任务链到 STOCKS_CALCULATE_STRATEGY_QUEUE 队列
        for stock_code in non_favorite_codes:
            sig = run_stock_strategy_task.s(stock_code).set(queue='STOCKS_CALCULATE_STRATEGY_QUEUE')
            sig.apply_async()  # 分派任务
            total_dispatched_chains += 1  # 计数分派的任务

        logger.info(f"任务结束: calculate_stock_strategy (调度器模式) - 共分派 {total_dispatched_chains} 个任务链")
        return f"已为 {total_favorite_stocks} 自选股和 {total_non_favorite_stocks} 非自选股分派 {total_dispatched_chains} 个任务链"

    except Exception as e:
        logger.error(f"执行 calculate_stock_strategy (调度器模式) 时出错: {e}", exc_info=True)
        # 可以考虑重试机制
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"

