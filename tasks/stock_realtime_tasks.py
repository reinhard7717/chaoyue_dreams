# tasks\stock_realtime_tasks.py
import asyncio
import logging
import math
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from celery import Celery, chain, group # 导入 group
from celery.utils.log import get_task_logger

from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_save_api_data'
STOCKS_SAVE_API_DATA_QUEUE = 'save_api_data'
logger = get_task_logger('tasks')

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
            favorite_stock_codes.add(fav.stock_code)
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
    logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
         logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes


@celery_app.task(bind=True, name='tasks.stock_realtime.save_realtime_data')
async def save_realtime_data(self, stock_code: str):
    """
    从API获取实时交易数据并保存到数据库
    Args:
        stock_code: 股票代码
    """
    stock_realtime_dao = StockRealtimeDAO()
    try:
        await stock_realtime_dao.fetch_and_save_realtime_data(stock_code)
    except Exception as e:
        logger.error(f"保存股票[{stock_code}]实时数据失败: {str(e)}")
        raise self.retry(exc=e)

@celery_app.task(bind=True, name='tasks.stock_realtime.get_realtime_data_task')  
def get_realtime_data_task(self):
    """
    修改后的调度器任务：
    1. 获取自选股和非自选股代码。
    2. 为每只股票创建任务链 (获取数据 -> 计算指标 -> 执行策略)。
    3. 将任务链中的任务分派到指定的队列 (favorite_* 或 stocks_*/calculate_*)。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: get_realtime_data_task (调度器模式) - 获取股票列表并分派细粒度任务链")
    try:
        # 在同步任务中运行异步代码来获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return "未获取到股票代码"

        total_dispatched_chains = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股任务链
        logger.info(f"准备为 {total_favorite_stocks} 个自选股分派任务链...")
        for stock_code in favorite_codes:
            logger.debug(f"创建自选股 {stock_code} 的任务链...")
            # 创建任务链，并为每个任务指定队列
            task_chain = chain(
                # 任务签名 .s()，并用 .set() 指定队列
                save_realtime_data.s(stock_code).set(queue=STOCKS_SAVE_API_DATA_QUEUE),
            )
            # 异步执行任务链
            task_chain.apply_async()
            total_dispatched_chains += 1
            logger.debug(f"已分派自选股 {stock_code} 的任务链")
            # 可选：短暂延迟以避免瞬间产生大量请求
            # import time
            # time.sleep(0.01)

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派任务链。")

        # 2. 分派非自选股任务链
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派任务链...")
        for stock_code in non_favorite_codes:
            logger.debug(f"创建非自选股 {stock_code} 的任务链...")
            # 创建任务链，并为每个任务指定队列
            task_chain = chain(
                save_realtime_data.s(stock_code).set(queue=FAVORITE_SAVE_API_DATA_QUEUE),
            )
            # 异步执行任务链
            task_chain.apply_async()
            total_dispatched_chains += 1
            logger.debug(f"已分派非自选股 {stock_code} 的任务链")
            # 可选：短暂延迟
            # import time
            # time.sleep(0.01)

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派任务链。")

        logger.info(f"任务结束: get_realtime_data_task (调度器模式) - 共分派 {total_dispatched_chains} 个任务链")
        return f"已为 {total_favorite_stocks} 自选股和 {total_non_favorite_stocks} 非自选股分派 {total_dispatched_chains} 个任务链"

    except Exception as e:
        logger.error(f"执行 get_realtime_data_task (调度器模式) 时出错: {e}", exc_info=True)
        # 可以考虑重试机制
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"
        