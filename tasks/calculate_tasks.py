# tasks\calculate_tasks.py
import asyncio
from asgiref.sync import async_to_sync
import logging
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from core.constants import TIME_TEADE_TIME_LEVELS_PER_TRADING
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from services.indicator_services import IndicatorService
from utils.cache_manager import CacheManager

logger = logging.getLogger('tasks')

FAVORITE_CALCULATE_INDICATORS_QUEUE = 'favorite_calculate_indicators'
STOCKS_CALCULATE_INDICATORS_QUEUE = 'calculate_indicators'

# --- 内部异步逻辑：计算单支股票 ---
async def _calculate_stock_indicators_async(stock_code: str):
    """实际执行异步计算的内部函数"""
    cache_manager = CacheManager()
    service = IndicatorService(cache_manager)
    # logger.info(f"开始异步计算股票 {stock_code} 的指标...")
    try:
        tasks = [
            service.calculate_and_save_all_indicators(stock_code, time_level)
            for time_level in TIME_TEADE_TIME_LEVELS_PER_TRADING
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
    # logger.info(f"Celery 任务开始处理股票 {stock_code}...")
    # 使用 async_to_sync() 在同步任务中运行异步代码
    result = async_to_sync(_calculate_stock_indicators_async)(stock_code)
    # logger.info(f"Celery 任务完成处理股票 {stock_code}，结果: {result}")
    # 返回异步函数的结果，这个结果应该是可序列化的（字符串）
    return result

# --- 异步辅助函数：获取需要处理的股票代码 (区分自选和非自选) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    cache_manager = CacheManager()
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    favorite_stock_codes = set()
    all_stock_codes = set()
    # 获取自选股
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        for fav in favorite_stocks:
            favorite_stock_codes.add(fav.get("stock_code"))
        logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    # 获取所有A股 (或者你需要的范围)
    try:
        # 注意：如果 get_stock_list() 返回大量数据，考虑分页或流式处理
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            code = str(stock.stock_code)
            if not code.endswith('.BJ'):
                all_stock_codes.add(code)
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
@celery_app.task(bind=True, name='tasks.calculate_tasks.calculate_stock_indicators')
def calculate_stock_indicators(self):
    """
    修改后的调度器任务：
    1. 获取自选股和非自选股代码。
    2. 为每只股票创建任务链 (获取数据 -> 计算指标 -> 执行策略)，并分派到指定的队列。
    3. 将自选股任务分派到 FAVORITE_CALCULATE_INDICATORS_QUEUE 队列。
    4. 将非自选股任务分派到 STOCKS_CALCULATE_INDICATORS_QUEUE 队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: calculate_stock_indicators (调度器模式) - 获取股票列表并分派细粒度任务链")
    try:
        # 在同步任务中运行异步代码来获取列表
        favorite_codes, non_favorite_codes = async_to_sync(_get_all_relevant_stock_codes_for_processing)()
        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return "未获取到股票代码"
        total_dispatched_chains = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)
        # 1. 分派自选股任务链到 FAVORITE_CALCULATE_INDICATORS_QUEUE 队列
        for stock_code in favorite_codes:
            sig = calculate_stock_indicators_for_single_stock.s(stock_code).set(queue=FAVORITE_CALCULATE_INDICATORS_QUEUE)
            sig.apply_async()  # 分派任务
            total_dispatched_chains += 1  # 计数分派的任务
        # 2. 分派非自选股任务链到 STOCKS_CALCULATE_INDICATORS_QUEUE 队列
        for stock_code in non_favorite_codes:
            sig = calculate_stock_indicators_for_single_stock.s(stock_code).set(queue=STOCKS_CALCULATE_INDICATORS_QUEUE)
            sig.apply_async()  # 分派任务
            total_dispatched_chains += 1  # 计数分派的任务
        logger.info(f"任务结束: calculate_stock_indicators (调度器模式) - 共分派 {total_dispatched_chains} 个任务链")
        return f"已为 {total_favorite_stocks} 自选股和 {total_non_favorite_stocks} 非自选股分派 {total_dispatched_chains} 个任务链"
    except Exception as e:
        logger.error(f"执行 calculate_stock_indicators (调度器模式) 时出错: {e}", exc_info=True)
        # 可以考虑重试机制
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"






