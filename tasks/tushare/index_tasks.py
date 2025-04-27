# tasks/tushare/index_tasks.py
import asyncio
import logging
import datetime
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
from celery.utils.log import get_task_logger
from dao_manager.tushare_daos import fund_flow_dao

logger = logging.getLogger('tasks')

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


#  ================ （历史）日级资金流向数据 ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_batch')
def save_fund_flow_daily_data_history_batch(self, stock_code: str):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的批次...")
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_history_fund_flow_daily_data_by_stock_code(stock_code))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_task')
def save_fund_flow_daily_data_history_task(self, batch_size: int = 2): # 限量：单次最大8000行数据
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
        for stock_code in favorite_codes:
            logger.info(f"创建自选股批次任务 (股票代码: {stock_code})...")
            # 使用新的批量任务，并指定队列
            save_fund_flow_daily_data_history_batch.s(stock_code).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for stock_code in non_favorite_codes:
            logger.info(f"创建非自选股批次任务 (股票代码: {stock_code})...")
            # 使用新的批量任务，并指定队列
            save_fund_flow_daily_data_history_batch.s(stock_code).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
            non_favorite_batches_dispatched += 1
        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")
        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}
