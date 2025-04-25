# tasks\stock_time_trade_tasks.py
import asyncio
import logging
import datetime
from typing import List
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO

logger = logging.getLogger('tasks')

FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveData_TimeTrade'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveData_TimeTrade'

def is_trading_time():
    now = datetime.datetime.now()
    # 交易日判断略，假设已是交易日
    if now.hour in [9, 10, 11, 13, 14, 15, 16]:
        if now.hour == 11 and now.minute >= 30:
            return False
        if now.hour == 9 and now.minute < 30:
            return False
        if now.hour == 16 and now.minute >= 2:
            return False
        return True
    return False

# --- 异步辅助函数：获取需要处理的股票代码 (区分自选和非自选) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    stock_basic_dao = StockBasicInfoDao()
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

    return sorted(favorite_stock_codes_list), sorted(non_favorite_stock_codes)

#  ================ 实时(分钟)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_minute_data_realtime_batch')
def save_minute_data_realtime_batch(self, stock_codes: List[str], time_level: str):
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
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_minute_time_trade_realtime_by_stock_codes_and_time_level(stock_codes, time_level))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_realtime_task')
def save_stocks_minute_data_realtime_task(self, batch_size: int = 1000, time_level: str = '5'): # 限量：单次最大1000行数据
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
                save_minute_data_realtime_batch.s(batch, time_level).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
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
                save_minute_data_realtime_batch.s(batch, time_level).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


#  ================ 历史(分钟)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_minute_data_history_batch')
def save_minute_data_history_batch(self, stock_codes: List[str], time_level: str):
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
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_minute_time_trade_realtime_by_stock_codes_and_time_level(stock_codes, time_level))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_history_task')
def save_stocks_minute_data_history_task(self, batch_size: int = 2, time_level: str = '5'): # 限量：单次最大8000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
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
                save_minute_data_history_batch.s(batch, time_level).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
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
                save_minute_data_history_batch.s(batch, time_level).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 历史(日线)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_day_data_history_batch')
def save_day_data_history_batch(self, stock_codes: List[str]):
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
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        return_info = asyncio.run(stock_time_trade_dao.save_daily_time_trade_history_by_stock_codes(stock_codes))
        logger.info(f"保存日线数据完成. {return_info}，起始stock_code: {stock_codes[0]}，结束stock_code: {stock_codes[-1]}")
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_day_data_history_task')
def save_stocks_day_data_history_task(self, batch_size: int = 2): # 限量：单次最大6000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
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
                save_day_data_history_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
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
                save_day_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 历史(每日基本信息)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_basic_data_history_batch')
def save_daily_basic_data_history_batch(self, stock_code: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_code:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理{stock_code} 股票...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_stock_daily_basic_history_by_stock_code(stock_code))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_basic_data_history_task')
def save_stocks_daily_basic_data_history_task(self):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务")
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
        for favorite_code in favorite_codes:
            if favorite_code:
                logger.info(f"创建自选股批次任务 (大小: {len(favorite_code)})...")
                # 使用新的批量任务，并指定队列
                save_daily_basic_data_history_batch.s(favorite_code).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {favorite_code})")

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for non_favorite_code in non_favorite_codes:
            if non_favorite_code:
                logger.info(f"创建非自选股批次任务 (大小: {len(non_favorite_code)})...")
                # 使用新的批量任务，并指定队列
                save_daily_basic_data_history_batch.s(non_favorite_code).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {non_favorite_code})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 今日基本信息 数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_daily_basic_data_today_batch')
def save_daily_basic_data_today_batch(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """

    logger.info(f"开始处理今日股票重要的基本面指标...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_today_stock_basic_info())
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_today_task')
def save_stocks_daily_basic_data_today_task(self):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务")
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
        for favorite_code in favorite_codes:
            if favorite_code:
                logger.info(f"创建自选股批次任务 (大小: {len(favorite_code)})...")
                # 使用新的批量任务，并指定队列
                save_daily_basic_data_today_batch.s(favorite_code).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {favorite_code})")

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for non_favorite_code in non_favorite_codes:
            if non_favorite_code:
                logger.info(f"创建非自选股批次任务 (大小: {len(non_favorite_code)})...")
                # 使用新的批量任务，并指定队列
                save_daily_basic_data_today_batch.s(non_favorite_code).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {non_favorite_code})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


#  ================ 历史(周线)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_week_data_history_batch')
def save_week_data_history_batch(self, stock_codes: List[str], time_level: str):
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
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_weekly_time_trade_by_stock_codes(stock_codes))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_week_data_history_task')
def save_stocks_week_data_history_task(self, batch_size: int = 5): # 限量：单次最大4500行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
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
                save_week_data_history_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
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
                save_week_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


#  ================ 历史(月线)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_month_data_history_batch')
def save_month_data_history_batch(self, stock_codes: List[str], time_level: str):
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
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_monthly_time_trade_by_stock_codes(stock_codes))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_month_data_history_task')
def save_stocks_month_data_history_task(self, batch_size: int = 10): # 限量：单次最大4500行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
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
                save_month_data_history_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
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
                save_month_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}
















