import asyncio
import logging
from celery import group
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService
from core.constants import TIME_TEADE_TIME_LEVELS, TimeLevel
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app

# 设置日志记录器
logger = logging.getLogger(__name__)

# --- 工作任务：处理单支股票 ---
@celery_app.task(bind=True, name='tasks.indicators.calculate_stock_indicators_for_single_stock')
async def calculate_stock_indicators_for_single_stock(self, stock_code: str):
    """
    计算并保存指定股票在所有时间级别上的指标。
    这是一个实际执行计算的工作任务。
    """
    service = IndicatorService()
    logger.info(f"开始计算股票 {stock_code} 的指标...")
    try:
        # 如果 calculate_and_save_all_indicators 本身可以并发处理不同 time_level，
        # 并且它是 async 函数，可以考虑使用 asyncio.gather
        tasks = [
            service.calculate_and_save_all_indicators(stock_code, time_level)
            for time_level in TIME_TEADE_TIME_LEVELS
        ]
        await asyncio.gather(*tasks)

        # 如果需要按顺序处理时间级别，或者 service 方法不是为并发设计的
        # for time_level in TIME_TEADE_TIME_LEVELS:
        #      logger.debug(f"计算 {stock_code} 在 {time_level} 级别指标")
        #      await service.calculate_and_save_all_indicators(stock_code, time_level)

        logger.info(f"成功完成股票 {stock_code} 的指标计算。")
        return f"Success: {stock_code}"
    except Exception as e:
        logger.error(f"计算股票 {stock_code} 指标时出错: {e}", exc_info=True)
        # 可以选择重试任务
        # raise self.retry(exc=e, countdown=60) # 60秒后重试
        # 或者直接标记失败
        return f"Failed: {stock_code} - {str(e)}"

# --- 分发任务：获取股票列表并分发工作任务 ---
@celery_app.task(bind=True, name='tasks.indicators.dispatch_all_stock_indicator_updates')
async def dispatch_all_stock_indicator_updates(self):
    """
    获取所有股票列表，并为每支股票分发一个指标计算任务。
    这个任务本身不执行计算，只负责调度。
    """
    stock_basic_dao = StockBasicDAO()
    logger.info("开始分发所有股票的指标更新任务...")
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("未找到任何股票，无需分发任务。")
            return "No stocks found to process."

        logger.info(f"获取到 {len(all_stocks)} 支股票，准备创建并发任务组。")

        # 为每支股票创建一个工作任务签名 (signature)
        # 使用 .s() 创建签名，而不是直接调用任务函数
        tasks_to_run = [
            calculate_stock_indicators_for_single_stock.s(stock.stock_code)
            for stock in all_stocks
        ]

        # 使用 group 将所有任务签名组合起来，以便并发执行
        task_group = group(tasks_to_run)

        # 异步执行任务组
        # apply_async 不会阻塞，它会立即返回一个 AsyncResult 对象，代表整个组的结果
        group_result = task_group.apply_async()

        logger.info(f"已成功分发 {len(all_stocks)} 个股票指标计算任务。任务组 ID: {group_result.id}")
        # === 调试日志 ===
        return_value = f"Dispatched indicator calculation tasks for {len(all_stocks)} stocks. Group ID: {group_result.id}"
        logger.info(f"任务即将返回，类型: {type(return_value)}, 值: {return_value}")
        # ===============

        # 注意：这个任务完成仅代表分发完成，不代表所有子任务都已执行完毕。
        return f"Dispatched indicator calculation tasks for {len(all_stocks)} stocks. Group ID: {group_result.id}"

    except Exception as e:
        logger.error(f"分发股票指标更新任务时出错: {e}", exc_info=True)
        # 可以根据需要进行重试或其他错误处理
        return f"Dispatch failed: {str(e)}"

# 调用示例
# await update_stock_indicators('000001', '1d')
# await update_stock_indicators('600519', TimeLevel.MIN_60)
