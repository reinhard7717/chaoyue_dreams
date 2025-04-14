# tasks/stock_realtime_tasks.py
import asyncio
import logging
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
# from celery import chain # 不再需要 chain，除非有后续步骤
from celery.utils.log import get_task_logger

from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_save_api_data_RealTime'
STOCKS_SAVE_API_DATA_QUEUE = 'save_api_data_RealTime'
logger = get_task_logger(__name__)

# --- 辅助函数：获取需要处理的股票代码 (保持不变) ---
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

    # 获取所有A股
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            all_stock_codes.add(stock.stock_code)
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

# --- 新的批量处理工作任务 ---
@celery_app.task(bind=True, name='tasks.stock_realtime.save_realtime_data_batch')
def save_realtime_data_batch(self, stock_codes: List[str]):
    """
    从API批量获取实时交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的批次...")
    # 在任务开始时创建一次 DAO 实例
    stock_realtime_dao = StockRealtimeDAO()
    results = []
    processed_count = 0
    success_count = 0
    error_count = 0
    async def _run_batch():
        nonlocal results, processed_count, success_count, error_count
        tasks = []
        for stock_code in stock_codes:
            # 为每个股票代码创建异步处理任务
            task = asyncio.create_task(stock_realtime_dao.fetch_and_save_realtime_data(stock_code))
            tasks.append(task)
        # 并发执行所有任务，并收集结果 (return_exceptions=True 避免一个失败导致全部中断)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 统计结果
        processed_count = len(results)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"处理股票 {stock_codes[i]} 时发生未捕获异常: {result}", exc_info=result)
                error_count += 1
            elif isinstance(result, dict) and result.get('错误'):
                # fetch_and_save_realtime_data 内部捕获并标记了错误
                error_count += result.get('错误', 1)
            elif isinstance(result, dict):
                 # 假设返回字典表示成功处理（即使创建/更新为0）
                 success_count += 1
            else:
                logger.warning(f"处理股票 {stock_codes[i]} 返回未知结果类型: {result}")
                error_count += 1 # 算作错误
    try:
        # 在同步 Celery 任务中运行主异步逻辑
        asyncio.run(_run_batch())
        logger.info(f"批次处理完成: 总数 {processed_count}, 成功 {success_count}, 失败 {error_count}")
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)
        # 可以选择重试整个批次，但这可能导致部分数据重复处理，需谨慎
        # raise self.retry(exc=e)
        error_count = len(stock_codes) # 标记整个批次失败
        success_count = 0
    finally:
        # 无论成功失败，都在任务结束时关闭 DAO 持有的 Session
        # 需要在异步上下文中关闭，所以再次使用 asyncio.run
        try:
            asyncio.run(stock_realtime_dao.close())
            # logger.info("StockRealtimeDAO session 已关闭。")
        except Exception as close_err:
            logger.error(f"关闭 StockRealtimeDAO 时出错: {close_err}", exc_info=True)

    return {"processed": processed_count, "success": success_count, "errors": error_count}

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.stock_realtime.get_realtime_data_task')
def get_realtime_data_task(self, batch_size: int = 20): # 增加批次大小参数
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: get_realtime_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
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
                logger.debug(f"创建自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_realtime_data_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
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
                logger.debug(f"创建非自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_realtime_data_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: get_realtime_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 get_realtime_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

# 移除旧的单任务处理函数 (如果不再需要)
# @celery_app.task(bind=True, name='tasks.stock_realtime.save_realtime_data')
# def save_realtime_data(self, stock_code: str):
#     ...
