# tasks/stock_realtime_tasks.py
import asyncio
import logging
import datetime
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
# from celery import chain # 不再需要 chain，除非有后续步骤
from celery.utils.log import get_task_logger

from dao_manager.tushare_daos.basic_info_dao import BasicInfoDao
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_save_api_data_RealTime'
STOCKS_SAVE_API_DATA_QUEUE = 'save_api_data_RealTime'
logger = get_task_logger(__name__)

def is_trading_time():
    now = datetime.datetime.now()
    # 交易日判断略，假设已是交易日
    if now.hour in [9, 10, 11, 13, 14]:
        if now.hour == 11 and now.minute >= 30:
            return False
        if now.hour == 9 and now.minute < 30:
            return False
        return True
    return False

# --- 辅助函数：获取需要处理的股票代码 (保持不变) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    stock_basic_dao = BasicInfoDao()
    favorite_stock_codes = set()
    all_stock_codes = set()
    # 获取自选股
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        if favorite_stocks is not None:
            for fav in favorite_stocks:
                code = str(fav.stock_id)
                if code.startswith('6'):
                    code = f"{code}.SH"
                else:
                    code = f"{code}.SZ"
                favorite_stock_codes.add(code)
            logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    # 获取所有A股
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            code = str(stock.stock_code)
            if code.startswith('6'):
                code = f"{code}.SH"
            else:
                code = f"{code}.SZ"
            all_stock_codes.add(code)
        logger.info(f"获取到 {len(all_stock_codes)} 个全市场股票代码")
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)

    # 计算非自选股代码
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)

    total_unique_stocks = len(favorite_stock_codes_list) + len(non_favorite_stock_codes)
    logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
        logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes

#  ================ 实时(Tick)数据任务 ================
# --- 新的批量处理工作任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime.save_tick_data_batch')
def save_tick_data_batch(self, stock_codes: List[str]):
    """
    从Tushare批量获取实时Tick交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的批次...")
    # 在任务开始时创建一次 DAO 实例
    stock_realtime_dao = StockRealtimeDAO()
    try:
        asyncio.run(stock_realtime_dao.tushare_save_tick_data_by_stock_codes(stock_codes))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)
    finally:
        # 无论成功失败，都在任务结束时关闭 DAO 持有的 Session
        # 需要在异步上下文中关闭，所以再次使用 asyncio.run
        if stock_realtime_dao:
            try:
                asyncio.run(stock_realtime_dao.close())
            except Exception as close_err:
                logger.error(f"关闭 StockRealtimeDAO 时出错: {close_err}", exc_info=True)
    return {}

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime.save_stocks_tick_data_task')
def save_stocks_tick_data_task(self, batch_size: int = 50): # sina数据最多每次50个
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    if not is_trading_time():
        return
    logger.info(f"任务启动: save_stocks_tick_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        # 在同步任务中运行异步代码获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return {"status": "warning", "message": "未获取到股票代码", "dispatched_batches": 0}

        total_dispatched_batches = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股批量任务
        logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
        for i in range(0, total_favorite_stocks, batch_size):
            batch = favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_tick_data_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_tick_data_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_tick_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_tick_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


#  ================ 实时(分钟)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime.save_realtime_min_data_batch')
def save_realtime_min_data_batch(self, stock_codes: List[str], time_level: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的批次...")
    # 在任务开始时创建一次 DAO 实例
    stock_realtime_dao = StockRealtimeDAO()
    try:
        asyncio.run(stock_realtime_dao.save_realtime_min_data(stock_codes, time_level))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)
    finally:
        # 无论成功失败，都在任务结束时关闭 DAO 持有的 Session
        # 需要在异步上下文中关闭，所以再次使用 asyncio.run
        if stock_realtime_dao:
            try:
                asyncio.run(stock_realtime_dao.close())
            except Exception as close_err:
                logger.error(f"关闭 StockRealtimeDAO 时出错: {close_err}", exc_info=True)
    return {}

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime.save_stocks_realtime_min_data_task')
def save_stocks_realtime_min_data_task(self, batch_size: int = 1000, time_level: str = '5'): # 限量：单次最大1000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    if not is_trading_time():
        return
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size}, 时间级别: {time_level})")
    try:
        # 在同步任务中运行异步代码获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return {"status": "warning", "message": "未获取到股票代码", "dispatched_batches": 0}

        total_dispatched_batches = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股批量任务
        logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
        for i in range(0, total_favorite_stocks, batch_size):
            batch = favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_realtime_min_data_batch.s(batch, time_level).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_realtime_min_data_batch.s(batch, time_level).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

# 移除旧的单任务处理函数 (如果不再需要)
# @celery_app.task(bind=True, name='tasks.stock_realtime.save_realtime_data')
# def save_realtime_data(self, stock_code: str):
#     ...
