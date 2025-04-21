# tasks\stock_time_trade_tasks.py
import asyncio
import logging
import math
from typing import List
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from celery import Celery, chain, group # 导入 group
from celery.utils.log import get_task_logger
from core.constants import TIME_TEADE_TIME_LEVELS_LITE, TIME_TEADE_TIME_LEVELS_PER_TRADING
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService

logger = logging.getLogger('tasks')

FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_save_api_data_TimeTrade'
STOCKS_SAVE_API_DATA_QUEUE = 'save_api_data_TimeTrade'

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

# --- 新增：处理单个股票的子任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_latest_trade')
def process_single_stock_latest_trade(self, stock_code: str):
    """
    获取并保存单个股票的最新实时数据 (子任务)
    """
    # logger.info(f"子任务启动: process_single_stock_realtime_data for {stock_code}")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO

    stock_indicators_dao = None # 初始化为 None
    task_result = f"处理股票 {stock_code} 数据失败" # 默认失败结果

    try:
        stock_indicators_dao = StockIndicatorsDAO()
        # 在同步的 Celery 任务中运行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_latest_time_trade_by_stock_code(stock_code))
        # task_result = f"成功处理股票 {stock_code} 最新实时数据"
        # logger.info(task_result)
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 数据时发生错误: {e}", exc_info=True)
        task_result = f"处理股票 {stock_code} 数据失败: {e}"
        # 可以选择性地重新抛出异常，让 Celery 知道任务失败
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise Ignore() # 或者使用 Ignore() 避免重试（如果设置了重试策略）
    finally:
        # 确保 DAO 被关闭，即使它在 try 块中未能成功初始化
        if stock_indicators_dao:
            try:
                # DAO 的 close 方法也可能是异步的
                asyncio.run(stock_indicators_dao.close())
                logger.debug(f"DAO for {stock_code} closed.")
            except Exception as close_err:
                logger.error(f"关闭股票 {stock_code} 的 DAO 时出错: {close_err}", exc_info=True)
        # logger.info(f"子任务结束: process_single_stock_realtime_data for {stock_code}")

    return task_result # 返回单个任务的结果

# --- 新增：处理单个股票的子任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_latest_trade_trading_hours')
def process_single_stock_latest_trade_trading_hours(self, stock_code: str, time_level: str):
    """
    获取并保存单个股票的最新实时数据 (子任务)
    """
    # logger.info(f"子任务启动: process_single_stock_realtime_data for {stock_code}")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO

    stock_indicators_dao = None # 初始化为 None
    task_result = f"处理股票 {stock_code} 数据失败" # 默认失败结果

    try:
        stock_indicators_dao = StockIndicatorsDAO()
        # 在同步的 Celery 任务中运行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_latest_time_trade(stock_code, time_level))
        # task_result = f"成功处理股票 {stock_code} 最新实时数据"
        # logger.info(task_result)
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 数据时发生错误: {e}", exc_info=True)
        task_result = f"处理股票 {stock_code} 数据失败: {e}"
        # 可以选择性地重新抛出异常，让 Celery 知道任务失败
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise Ignore() # 或者使用 Ignore() 避免重试（如果设置了重试策略）
    finally:
        # 确保 DAO 被关闭，即使它在 try 块中未能成功初始化
        if stock_indicators_dao:
            try:
                # DAO 的 close 方法也可能是异步的
                if asyncio.iscoroutinefunction(getattr(stock_indicators_dao, 'close', None)):
                     asyncio.run(stock_indicators_dao.close())
                elif callable(getattr(stock_indicators_dao, 'close', None)):
                     stock_indicators_dao.close()
                logger.debug(f"DAO for {stock_code} closed.")
            except Exception as close_err:
                logger.error(f"关闭股票 {stock_code} 的 DAO 时出错: {close_err}", exc_info=True)
        # logger.info(f"子任务结束: process_single_stock_realtime_data for {stock_code}")

    return task_result # 返回单个任务的结果

@celery_app.task(bind=True, name='tasks.stock_indicators.process_stocks_latest_trade_by_time_level')
def process_stocks_latest_trade_by_time_level(self, time_level: str, stock_codes: List[str]):
    """
    获取并保存所有股票的最新实时数据 (子任务)
    """
    # logger.info(f"子任务启动: process_single_stock_realtime_data for {stock_code}")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO

    stock_indicators_dao = None # 初始化为 None
    task_result = f"处理时间级别 {time_level} 全部股票数据失败" # 默认失败结果

    try:
        stock_indicators_dao = StockIndicatorsDAO()
        # 在同步的 Celery 任务中运行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_latest_time_trade_by_time_level_and_stock_codes(stock_codes, time_level))
        # task_result = f"成功处理股票 {stock_code} 最新实时数据"
        # logger.info(task_result)
    except Exception as e:
        logger.error(f"处理时间级别 {time_level} 全部股票数据时发生错误: {e}", exc_info=True)
        task_result = f"处理时间级别 {time_level} 全部股票数据失败: {e}"
    finally:
        # 确保 DAO 被关闭，即使它在 try 块中未能成功初始化
        if stock_indicators_dao:
            try:
                # DAO 的 close 方法也可能是异步的
                asyncio.run(stock_indicators_dao.close())
                # logger.debug(f"DAO for {time_level} closed.")
            except Exception as close_err:
                logger.error(f"关闭股票 {time_level} 的 stock_indicators_dao 时出错: {close_err}", exc_info=True)
        # logger.info(f"子任务结束: process_single_stock_realtime_data for {stock_code}")

    return task_result # 返回单个任务的结果

# 任务调度：计算所有股票的指标
@celery_app.task(bind=True, name='tasks.stock_time_trade_tasks.save_latest_trade_datas_by_time_level')
def save_latest_trade_datas_by_time_level(self, time_level: str):
    """
    修改后的调度器任务：
    1. 获取自选股和非自选股代码。
    2. 为每只股票创建任务链 (获取数据 -> 计算指标 -> 执行策略)，并分派到指定的队列。
    3. 将自选股任务分派到 FAVORITE_SAVE_API_DATA_QUEUE 队列。
    4. 将非自选股任务分派到 STOCKS_SAVE_API_DATA_QUEUE 队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: save_latest_trade_datas_by_time_level (调度器模式) - 获取股票列表并分派细粒度任务链")
    try:
        # 在同步任务中运行异步代码来获取列表
        sig = process_stocks_latest_trade_by_time_level.s(time_level).set(queue=STOCKS_SAVE_API_DATA_QUEUE)
        sig.apply_async()  # 分派任务

        logger.info(f"任务结束: save_latest_trade_datas_by_time_level (调度器模式)")
        return f"已分派 获取最新 {time_level} 级别股票数据 任务"

    except Exception as e:
        logger.error(f"执行 save_latest_trade_datas_by_time_level (调度器模式) 时出错: {e}", exc_info=True)
        # 可以考虑重试机制
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"


# 任务调度：计算所有股票的指标
@celery_app.task(bind=True, name='tasks.stock_time_trade_tasks.save_latest_trade_datas')
def save_latest_trade_datas(self, time_level: str, batch_size: int = 200):
    """
    修改后的调度器任务：
    1. 获取自选股和非自选股代码。
    2. 为每只股票创建任务链 (获取数据 -> 计算指标 -> 执行策略)，并分派到指定的队列。
    3. 将自选股任务分派到 FAVORITE_SAVE_API_DATA_QUEUE 队列。
    4. 将非自选股任务分派到 STOCKS_SAVE_API_DATA_QUEUE 队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: save_latest_trade_datas (调度器模式) - 获取股票列表并分派细粒度任务链")
    try:
        # 在同步任务中运行异步代码来获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return "未获取到股票代码"

        total_dispatched_batches = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股任务链到 FAVORITE_SAVE_API_DATA_QUEUE 队列
        for i in range(0, total_favorite_stocks, batch_size):
            batch = favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                process_stocks_latest_trade_by_time_level.s(batch, time_level).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        # 2. 分派非自选股任务链到 STOCKS_SAVE_API_DATA_QUEUE 队列
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                process_stocks_latest_trade_by_time_level.s(batch, time_level).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                total_non_favorite_stocks += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {total_non_favorite_stocks} 个批次任务。")

        logger.info(f"任务结束: save_latest_trade_datas (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_latest_trade_datas (调度器模式) 时出错: {e}", exc_info=True)
        # 可以考虑重试机制
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"
