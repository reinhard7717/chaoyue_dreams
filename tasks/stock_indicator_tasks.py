"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app

logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_realtime_data')
def save_stock_all_latest_realtime_data(self):
    """
    保存股票最新实时数据 (推荐方案)
    """
    logger.info("任务启动: save_stock_all_latest_realtime_data")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_latest_realtime_data())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        # 捕获从 _run_save_stock_all_latest_realtime_data 抛出的异常
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票最新实时数据失败: {e}"
        # 根据需要决定是否让 Celery 任务失败
        # raise # 如果希望任务状态为 FAILURE
    finally:
        logger.info("任务执行流程结束: save_stock_all_latest_realtime_data")

    return task_result

async def _run_save_stock_all_latest_realtime_data():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_latest_time_trade()
        return "保存股票最新实时数据完成"
    except Exception as e:
        logger.error(f"异步保存股票最新实时数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_kdj')
def save_stock_all_latest_kdj(self):
    """
    保存股票最新KDJ数据
    """
    logger.info("任务启动: save_stock_all_latest_kdj")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_latest_kdj())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票最新KDJ数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_stock_all_latest_kdj")

async def _run_save_stock_all_latest_kdj():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_latest_kdj()
        return "保存股票最新KDJ数据完成"
    except Exception as e:
        logger.error(f"异步保存股票最新KDJ数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_macd')
def save_stock_all_latest_macd(self):
    """
    保存股票最新MACD数据
    """
    logger.info("任务启动: save_stock_all_latest_macd")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_latest_macd())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票最新MACD数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_stock_all_latest_macd")

async def _run_save_stock_all_latest_macd():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_latest_macd()
        return "保存股票最新MACD数据完成"
    except Exception as e:
        logger.error(f"异步保存股票最新MACD数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

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

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_time_trade')
def save_all_history_time_trade(stock_code: str):
    """
    保存股票历史实时数据
    """
    logger.info("任务启动: save_all_history_time_trade")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_history_time_trade())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票历史实时数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_all_history_time_trade")

async def _run_save_stock_all_history_time_trade():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_history_time_trade()
        return "保存股票历史实时数据完成"
    except Exception as e:
        logger.error(f"异步保存股票历史实时数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_kdj')
def save_all_history_kdj(stock_code: str):
    """
    保存股票历史KDJ数据
    """
    logger.info("任务启动: save_all_history_kdj")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_history_kdj())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票历史KDJ数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_all_history_kdj")

async def _run_save_stock_all_history_kdj():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_history_kdj()
        return "保存股票历史KDJ数据完成"
    except Exception as e:
        logger.error(f"异步保存股票历史KDJ数据出错: {e}", exc_info=True)
        # 将异常重新抛出，由外层捕获
        raise
    finally:
        # 确保无论成功或失败都关闭 DAO
        if stock_indicators_dao:
            await stock_indicators_dao.close()

@celery_app.task(bind=True, name='tasks.stock_indicators.save_all_history_macd')
def save_all_history_macd(stock_code: str):
    """
    保存股票历史MACD数据
    """
    logger.info("任务启动: save_all_history_macd")
    task_result = "任务出现意外未能正常完成" # 默认结果
    try:
        # 使用 asyncio.run 运行整个异步逻辑块
        task_result = asyncio.run(_run_save_stock_all_history_macd())
        logger.info(f"任务成功完成: {task_result}")
    except Exception as e:
        logger.error(f"任务执行期间发生错误: {e}", exc_info=True)
        task_result = f"保存股票历史MACD数据失败: {e}"
    finally:
        logger.info("任务执行流程结束: save_all_history_macd")

async def _run_save_stock_all_history_macd():
    """封装所有异步操作的辅助函数"""
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    stock_indicators_dao = StockIndicatorsDAO()
    try:
        await stock_indicators_dao.fetch_and_save_all_history_macd()
        return "保存股票历史MACD数据完成"
    except Exception as e:
        logger.error(f"异步保存股票历史MACD数据出错: {e}", exc_info=True)
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

























