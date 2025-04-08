import asyncio
import logging
from celery import group
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService
from core.constants import TIME_TEADE_TIME_LEVELS, TimeLevel
from chaoyue_dreams.celery import app as celery_app

logger = logging.getLogger(__name__)

# --- 内部异步逻辑：计算单支股票 ---
async def _calculate_stock_indicators_async(stock_code: str):
    """实际执行异步计算的内部函数"""
    service = IndicatorService()
    logger.info(f"开始异步计算股票 {stock_code} 的指标...")
    try:
        tasks = [
            service.calculate_and_save_all_indicators(stock_code, time_level)
            for time_level in TIME_TEADE_TIME_LEVELS
        ]
        # 注意：确保 service.calculate_and_save_all_indicators 也是 async def
        await asyncio.gather(*tasks)
        logger.info(f"成功完成股票 {stock_code} 的异步指标计算。")
        return f"Success: {stock_code}"
    except Exception as e:
        logger.error(f"异步计算股票 {stock_code} 指标时出错: {e}", exc_info=True)
        # 返回错误信息，而不是重新抛出异常，以便 Celery 可以记录结果
        return f"Failed: {stock_code} - {str(e)}"

# --- 工作任务（同步包装器）：处理单支股票 ---
@celery_app.task(bind=True, name='tasks.indicators.calculate_stock_indicators_for_single_stock')
def calculate_stock_indicators_for_single_stock(self, stock_code: str):
    """
    Celery 工作任务（同步）：计算并保存指定股票在所有时间级别上的指标。
    它调用内部的异步函数来完成工作。
    """
    logger.info(f"Celery 任务开始处理股票 {stock_code}...")
    # 使用 asyncio.run() 在同步任务中运行异步代码
    result = asyncio.run(_calculate_stock_indicators_async(stock_code))
    logger.info(f"Celery 任务完成处理股票 {stock_code}，结果: {result}")
    # 返回异步函数的结果，这个结果应该是可序列化的（字符串）
    return result


# --- 内部异步逻辑：分发任务 ---
async def _dispatch_all_stock_indicator_updates_async():
    """实际执行异步分发的内部函数"""
    stock_basic_dao = StockBasicDAO()
    logger.info("开始异步分发所有股票的指标更新任务...")
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("未找到任何股票，无需分发任务。")
            return "No stocks found to process."

        stock_count = len(all_stocks)
        logger.info(f"获取到 {stock_count} 支股票，准备创建并发任务组。")

        tasks_to_run = [
            calculate_stock_indicators_for_single_stock.s(stock.stock_code)
            for stock in all_stocks
        ]
        task_group = group(tasks_to_run)
        group_result = task_group.apply_async()

        result_message = f"Dispatched indicator calculation tasks for {stock_count} stocks. Group ID: {group_result.id}"
        logger.info(f"已成功异步分发 {stock_count} 个股票指标计算任务。任务组 ID: {group_result.id}")
        return result_message

    except Exception as e:
        logger.error(f"异步分发股票指标更新任务时出错: {e}", exc_info=True)
        return f"Dispatch failed: {str(e)}"

# --- 分发任务（同步包装器）：获取股票列表并分发工作任务 ---
@celery_app.task(bind=True, name='tasks.indicators.dispatch_all_stock_indicator_updates')
def dispatch_all_stock_indicator_updates(self):
    """
    Celery 分发任务（同步）：获取所有股票列表，并为每支股票分发一个指标计算任务。
    调用内部异步函数完成分发逻辑。
    """
    logger.info("Celery 分发任务启动...")
    # 使用 asyncio.run() 在同步任务中运行异步代码
    result = asyncio.run(_dispatch_all_stock_indicator_updates_async())
    logger.info(f"Celery 分发任务完成，结果: {result}")
    # 返回异步函数的结果，这个结果应该是可序列化的（字符串）
    return result

