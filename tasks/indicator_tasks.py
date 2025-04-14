import asyncio
import logging
from celery import group
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService
from core.constants import TIME_TEADE_TIME_LEVELS, TimeLevel
from chaoyue_dreams.celery import app as celery_app
from tasks.stock_indicator_tasks import calculate_stock_indicators_for_single_stock

logger = logging.getLogger('tasks')


@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_latest_kdj')
def process_single_stock_latest_kdj(self, stock_code: str):
    """
    计算并保存单个股票的最新KDJ数据 (子任务)

    :param self: Celery 任务实例
    :param stock_code: 要处理的股票代码
    :return: 任务执行结果描述字符串
    """
    logger.info(f"子任务启动: process_single_stock_latest_kdj for {stock_code}")
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = None # 初始化 DAO 实例变量
    task_result = f"保存最新KDJ失败 for {stock_code}" # 默认失败结果
    try:
        # 在任务内部实例化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        logger.debug(f"[{stock_code}] StockIndicatorsDAO initialized.")
        # 使用 asyncio.run 在同步任务中执行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_latest_kdj_by_stock_code(stock_code))
        # ---------------------------------------------
        task_result = f"成功保存最新KDJ for {stock_code}"
        logger.info(task_result)
    except Exception as e:
        # 记录包括堆栈跟踪的详细错误信息
        logger.error(f"处理最新KDJ时发生错误 for {stock_code}: {e}", exc_info=True)
        task_result = f"保存最新KDJ失败 for {stock_code}: {e}"
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
        logger.info(f"子任务结束: process_single_stock_latest_kdj for {stock_code}")
    return task_result

@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_history_kdj')
def process_single_stock_history_kdj(self, stock_code: str):
    """
    计算并保存单个股票的历史KDJ数据 (子任务)

    :param self: Celery 任务实例
    :param stock_code: 要处理的股票代码
    :return: 任务执行结果描述字符串
    """
    logger.info(f"子任务启动: process_single_stock_history_kdj for {stock_code}")
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = None # 初始化 DAO 实例变量
    task_result = f"保存历史KDJ失败 for {stock_code}" # 默认失败结果
    try:
        # 在任务内部实例化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        logger.debug(f"[{stock_code}] StockIndicatorsDAO initialized.")
        # 使用 asyncio.run 在同步任务中执行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_history_kdj_by_stock_code(stock_code))
        # ---------------------------------------------
        task_result = f"成功保存历史KDJ for {stock_code}"
        logger.info(task_result)
    except Exception as e:
        # 记录包括堆栈跟踪的详细错误信息
        logger.error(f"处理历史KDJ时发生错误 for {stock_code}: {e}", exc_info=True)
        task_result = f"保存历史KDJ失败 for {stock_code}: {e}"
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
        logger.info(f"子任务结束: process_single_stock_history_kdj for {stock_code}")
    return task_result

@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_latest_macd')
def process_single_stock_latest_macd(self, stock_code: str):
    """
    计算并保存单个股票的最新MACD数据 (子任务)

    :param self: Celery 任务实例
    :param stock_code: 要处理的股票代码
    :return: 任务执行结果描述字符串
    """
    logger.info(f"子任务启动: process_single_stock_latest_macd for {stock_code}")
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = None # 初始化 DAO 实例变量
    task_result = f"保存最新MACD失败 for {stock_code}" # 默认失败结果
    try:
        # 在任务内部实例化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        logger.debug(f"[{stock_code}] StockIndicatorsDAO initialized.")
        # 使用 asyncio.run 在同步任务中执行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_latest_macd_by_stock_code(stock_code))
        # ---------------------------------------------
        task_result = f"成功保存最新MACD for {stock_code}"
        logger.info(task_result)
    except Exception as e:
        # 记录包括堆栈跟踪的详细错误信息
        logger.error(f"处理最新MACD时发生错误 for {stock_code}: {e}", exc_info=True)
        task_result = f"保存最新MACD失败 for {stock_code}: {e}"
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

@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_history_macd')
def process_single_stock_history_macd(self, stock_code: str):
    """
    计算并保存单个股票的历史MACD数据 (子任务)

    :param self: Celery 任务实例
    :param stock_code: 要处理的股票代码
    :return: 任务执行结果描述字符串
    """
    logger.info(f"子任务启动: process_single_stock_history_macd for {stock_code}")
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = None # 初始化 DAO 实例变量
    task_result = f"保存历史MACD失败 for {stock_code}" # 默认失败结果
    try:
        # 在任务内部实例化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        logger.debug(f"[{stock_code}] StockIndicatorsDAO initialized.")
        # 使用 asyncio.run 在同步任务中执行异步 DAO 方法
        asyncio.run(stock_indicators_dao.fetch_and_save_history_macd_by_stock_code(stock_code))
        # ---------------------------------------------
        task_result = f"成功保存历史MACD for {stock_code}"
        logger.info(task_result)
    except Exception as e:
        # 记录包括堆栈跟踪的详细错误信息
        logger.error(f"处理历史MACD时发生错误 for {stock_code}: {e}", exc_info=True)
        task_result = f"保存历史MACD失败 for {stock_code}: {e}"
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
        logger.info(f"子任务结束: process_single_stock_history_macd for {stock_code}")
    return task_result


@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_ma')
def save_stock_all_latest_ma(self):
    """
    保存股票最新MA数据
    """
    logger.info("任务启动: save_stock_all_latest_ma")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_latest_ma())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票最新MA数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_stock_all_latest_ma")

async def _run_save_stock_all_latest_ma():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_latest_ma()
        return "保存股票最新MA数据完成"
    except Exception as e:
        logger.error(f"异步保存股票最新MA数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_boll')
def save_stock_all_latest_boll(self):
    """
    保存股票最新BOLL数据
    """
    logger.info("任务启动: save_stock_all_latest_boll")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_latest_boll())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票最新BOLL数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_stock_all_latest_boll")

async def _run_save_stock_all_latest_boll():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_latest_boll()
        return "保存股票最新BOLL数据完成"
    except Exception as e:
        logger.error(f"异步保存股票最新BOLL数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_ma')
def save_all_history_ma(stock_code: str):
    """
    保存股票历史MA数据
    """
    logger.info("任务启动: save_all_history_ma")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_history_ma())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票历史MA数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_all_history_ma")

async def _run_save_stock_all_history_ma():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_history_ma()
        return "保存股票历史MA数据完成"
    except Exception as e:
        logger.error(f"异步保存股票历史MA数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_boll')
def save_all_history_boll(stock_code: str):
    """
    保存股票历史BOLL数据
    """
    logger.info("任务启动: save_all_history_boll")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_history_boll())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票历史BOLL数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_all_history_boll")

async def _run_save_stock_all_history_boll():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_history_boll()
        return "保存股票历史BOLL数据完成"
    except Exception as e:
        logger.error(f"异步保存股票历史BOLL数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()
