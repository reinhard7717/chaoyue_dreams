"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from celery import Celery, group # 导入 group
from celery.utils.log import get_task_logger
from core.constants import TIME_TEADE_TIME_LEVELS


from services.indicator_services import IndicatorService


logger = logging.getLogger(__name__)

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
                if asyncio.iscoroutinefunction(getattr(stock_indicators_dao, 'close', None)):
                     asyncio.run(stock_indicators_dao.close())
                elif callable(getattr(stock_indicators_dao, 'close', None)):
                     stock_indicators_dao.close()
                logger.debug(f"DAO for {stock_code} closed.")
            except Exception as close_err:
                logger.error(f"关闭股票 {stock_code} 的 DAO 时出错: {close_err}", exc_info=True)
        # logger.info(f"子任务结束: process_single_stock_realtime_data for {stock_code}")

    return task_result # 返回单个任务的结果

@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_realtime_trade')
def process_single_stock_realtime_trade(self, stock_code: str):
    """
    获取并保存单个股票的最新实时数据 (子任务)
    """
    # logger.info(f"子任务启动: process_single_stock_realtime_data for {stock_code}")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO

    stock_indicators_dao = None # 初始化为 None
    task_result = f"处理股票 {stock_code} 数据失败" # 默认失败结果

    try:
        stock_realtime_dao = StockRealtimeDAO()
        # 在同步的 Celery 任务中运行异步 DAO 方法
        asyncio.run(stock_realtime_dao.fetch_and_save_time_deals(stock_code))
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
        if stock_realtime_dao:
            try:
                # DAO 的 close 方法也可能是异步的
                if asyncio.iscoroutinefunction(getattr(stock_realtime_dao, 'close', None)):
                     asyncio.run(stock_realtime_dao.close())
                elif callable(getattr(stock_realtime_dao, 'close', None)):
                     stock_realtime_dao.close()
                logger.debug(f"DAO for {stock_code} closed.")
            except Exception as close_err:
                logger.error(f"关闭股票 {stock_code} 的 DAO 时出错: {close_err}", exc_info=True)
        # logger.info(f"子任务结束: process_single_stock_realtime_data for {stock_code}")

    return task_result # 返回单个任务的结果


# --- 新增：处理单个股票的子任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_latest_trade_trading_hours')
def process_single_stock_latest_trade_trading_hours(self, stock_code: str):
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
        asyncio.run(stock_indicators_dao.fetch_and_save_latest_time_trade_trading_hours_by_stock_code(stock_code))
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

# --- 新增：处理单个股票单个时间级别历史数据的子任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_history_trade')
def process_single_stock_history_trade(self, stock_code: str):
    """
    获取并保存单个股票在指定时间级别的历史分时/K线数据 (子任务)
    """
    # logger.info(f"子任务启动: process_single_stock_history_trade for {stock_code} ({time_level})")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO

    stock_indicators_dao = None # 初始化
    task_result = f"保存历史数据失败 for {stock_code}" # 默认失败结果

    try:
        # 在任务内部实例化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        # 在同步的 Celery 任务中运行异步 DAO 方法
        # 注意：这里假设 fetch_and_save_history_time_trade_by_stock_code 是 async
        asyncio.run(stock_indicators_dao.fetch_and_save_history_time_trade_by_stock_code(stock_code))
        task_result = f"成功保存历史数据 for {stock_code}"
        logger.info(task_result)
    except Exception as e:
        logger.error(f"保存历史数据时发生错误 for {stock_code} : {e}", exc_info=True)
        task_result = f"保存历史数据失败 for {stock_code}: {e}"
        # 可以选择性地失败任务
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise Ignore()
    finally:
        # 确保 DAO 被关闭
        if stock_indicators_dao:
            try:
                # 假设 close 是异步的
                if asyncio.iscoroutinefunction(getattr(stock_indicators_dao, 'close', None)):
                     asyncio.run(stock_indicators_dao.close())
                elif callable(getattr(stock_indicators_dao, 'close', None)):
                     stock_indicators_dao.close()
                logger.debug(f"DAO for {stock_code} closed.")
            except Exception as close_err:
                logger.error(f"关闭 DAO for {stock_code} 时出错: {close_err}", exc_info=True)
        # logger.info(f"子任务结束: process_single_stock_history_trade for {stock_code} ({time_level})")

    return task_result


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


@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_indicators')
def process_single_stock_indicators(self, stock_code: str):
    """
    计算并保存单个股票的最新MACD数据 (子任务)
    :param self: Celery 任务实例
    :param stock_code: 要处理的股票代码
    :return: 任务执行结果描述字符串
    """
    logger.info(f"子任务启动: process_single_stock_indicators for {stock_code}")
    stock_indicators_dao = None # 初始化 DAO 实例变量
    task_result = f"计算技术指标失败 for {stock_code}" # 默认失败结果
    from services.indicator_services import IndicatorService
    try:
        # 在任务内部实例化 DAO
        service = IndicatorService()
        logger.debug(f"[{stock_code}] StockIndicatorsDAO initialized.")
        # 使用 asyncio.run 在同步任务中执行异步 DAO 方法
        asyncio.run(service.calculate_and_save_all_indicators(stock_code))
        # ---------------------------------------------
        task_result = f"成功计算技术指标 for {stock_code}"
        logger.info(task_result)
    except Exception as e:
        # 记录包括堆栈跟踪的详细错误信息
        logger.error(f"计算技术指标时发生错误 for {stock_code}: {e}", exc_info=True)
        task_result = f"计算技术指标失败 for {stock_code}: {e}"
        # 根据需要，可以更新任务状态为失败
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise Ignore() # 如果不希望 Celery 将此视为错误重试或计入失败总数
    finally:
        # --- 确保关闭 DAO 连接 ---
        if stock_indicators_dao:
            try:
                # 检查 close 方法是否是异步的
                if asyncio.iscoroutinefunction(getattr(stock_indicators_dao, 'close', None)):
                     asyncio.run(stock_indicators_dao.close())
                     logger.debug(f"[{stock_code}] StockIndicatorsDAO (async) closed.")
                # 否则，假设它是同步的
                elif callable(getattr(stock_indicators_dao, 'close', None)):
                     stock_indicators_dao.close()
                     logger.debug(f"[{stock_code}] StockIndicatorsDAO (sync) closed.")
            except Exception as close_err:
                logger.error(f"关闭 StockIndicatorsDAO for {stock_code} 时出错: {close_err}", exc_info=True)
        logger.info(f"子任务结束: process_single_stock_latest_macd for {stock_code}")
    return task_result

@celery_app.task(bind=True, name='tasks.stock_indicators.get_trade_and_calculate_and_strategy')
async def get_trade_and_calculate_and_strategy(self, stock_code: str):
    """
    获取股票最新实时数据并计算技术指标
    """
    logger.info(f"任务启动: get_trade_and_calculate for {stock_code}")
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    from tasks.strategy_tasks import strategy_macd_rsi_kdj_boll_strategy_for_stock
    stock_basic_dao = StockBasicDAO()
    favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
    for favorite_stock in favorite_stocks:
        process_single_stock_latest_trade_trading_hours(favorite_stock.stock_code)
        process_single_stock_realtime_trade(favorite_stock.stock_code)
        process_single_stock_indicators(favorite_stock.stock_code)
        strategy_macd_rsi_kdj_boll_strategy_for_stock(favorite_stock.stock_code)
    stocks = await stock_basic_dao.get_stock_list()
    for stock in stocks:
        process_single_stock_latest_trade_trading_hours(stock.stock_code)
        process_single_stock_realtime_trade(stock.stock_code)
        process_single_stock_indicators(stock.stock_code)
        strategy_macd_rsi_kdj_boll_strategy_for_stock(stock.stock_code)
    logger.info(f"任务结束: get_trade_and_calculate for {stock_code}")




















