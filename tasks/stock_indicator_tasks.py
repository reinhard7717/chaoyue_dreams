"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
from celery import shared_task
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from celery import Celery, group # 导入 group
from celery.utils.log import get_task_logger

logger = logging.getLogger(__name__)
TIME_LEVELS = ['5','15','30','60','Day','Day_qfq','Day_hfq','Week','Week_qfq','Week_hfq','Month','Month_qfq','Month_hfq','Year','Year_qfq','Year_hfq']


# --- 新增：处理单个股票的子任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_single_stock_realtime_data')
def process_single_stock_realtime_data(self, stock_code: str):
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

# --- 修改后的主任务 (分发器) ---
@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_latest_realtime_data')
def save_stock_all_latest_realtime_data(self):
    """
    分发任务以并发保存所有股票的最新实时数据 (主任务/分发器)
    """
    logger.info("主任务启动: save_stock_all_latest_realtime_data (分发器)")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_basic_dao import StockBasicDAO

    stock_basic_dao = None # 初始化为 None
    try:
        stock_basic_dao = StockBasicDAO()

        # 获取股票列表 - 如果 get_stock_list 是异步的，也需要用 asyncio.run
        # 假设 get_stock_list 也是异步的
        try:
            stocks = asyncio.run(stock_basic_dao.get_stock_list())
            logger.info(f"获取到 {len(stocks)} 支股票列表")
        except Exception as e:
             logger.error(f"获取股票列表时出错: {e}", exc_info=True)
             raise # 获取列表失败，则无法继续，直接抛出异常使主任务失败

        if not stocks:
            logger.warning("未获取到任何股票信息，任务结束")
            return "未获取到股票列表，无法分发任务"

        # 创建子任务签名列表
        # 使用 .s() 创建签名，传递 stock_code 参数
        tasks_signatures = [process_single_stock_realtime_data.s(stock.stock_code) for stock in stocks]

        # 创建任务组
        task_group = group(tasks_signatures)
        logger.info(f"已创建包含 {len(tasks_signatures)} 个子任务的任务组")

        # 异步执行任务组
        # apply_async 不会阻塞，它会立即返回一个 AsyncResult 对象
        group_result = task_group.apply_async()

        logger.info(f"任务组已提交执行，Group ID: {group_result.id}")
        # 主任务的目的是分发，通常不需要等待所有子任务完成
        # 如果需要等待并获取结果（不推荐，可能阻塞 worker），可以使用：
        # completed_results = group_result.get()
        # logger.info("所有子任务已完成")
        # return f"所有 {len(completed_results)} 个子任务已完成处理"

        # 返回分发成功的消息
        return f"成功分发 {len(tasks_signatures)} 个股票数据处理子任务，Group ID: {group_result.id}"

    except Exception as e:
        logger.error(f"主任务执行期间发生错误: {e}", exc_info=True)
        # 让 Celery 知道主任务（分发器）失败了
        # 可以根据需要决定是否 raise
        # raise # 如果希望任务状态为 FAILURE
        return f"分发股票数据处理任务失败: {e}"
    finally:
        # 关闭主任务中使用的 DAO
        if stock_basic_dao:
            try:
                 # 假设 close 也是异步的
                if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                     asyncio.run(stock_basic_dao.close())
                elif callable(getattr(stock_basic_dao, 'close', None)):
                     stock_basic_dao.close()
                logger.debug("主任务 DAO closed.")
            except Exception as close_err:
                logger.error(f"关闭主任务 DAO 时出错: {close_err}", exc_info=True)
        logger.info("主任务执行流程结束: save_stock_all_latest_realtime_data (分发器)")

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

# --- 修改后的主任务 (分发器) ---
# 注意：移除了原签名中的 stock_code 参数，因为逻辑是处理所有股票
@celery_app.task(bind=True, name='tasks.stock_indicators.save_stock_all_history_time_trade')
def save_stock_all_history_time_trade(self):
    """
    分发任务以并发保存所有股票在所有时间级别的历史分时/K线数据 (主任务/分发器)
    """
    logger.info("主任务启动: dispatch_history_time_trade_saving (分发器)")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_basic_dao import StockBasicDAO

    stock_basic_dao = None # 初始化
    try:
        stock_basic_dao = StockBasicDAO()

        # 获取股票列表 (假设 get_stock_list 是 async)
        try:
            stocks = asyncio.run(stock_basic_dao.get_stock_list())
            logger.info(f"获取到 {len(stocks)} 支股票列表")
        except Exception as e:
             logger.error(f"获取股票列表时出错: {e}", exc_info=True)
             raise # 获取列表失败，则无法继续，直接抛出异常使主任务失败

        if not stocks:
            logger.warning("未获取到任何股票信息，任务结束")
            return "未获取到股票列表，无法分发任务"

        # 创建子任务签名列表
        tasks_signatures = []
        for stock in stocks:
            tasks_signatures.append(process_single_stock_history_trade.s(stock.stock_code))
        if not tasks_signatures:
             logger.warning("没有生成任何子任务签名，任务结束")
             return "没有有效的股票或时间级别组合，未分发任务"

        # 创建任务组
        task_group = group(tasks_signatures)
        logger.info(f"已创建包含 {len(tasks_signatures)} 个历史数据保存子任务的任务组")

        # 异步执行任务组
        group_result = task_group.apply_async()

        logger.info(f"任务组已提交执行，Group ID: {group_result.id}")

        # 返回分发成功的消息
        return f"成功分发 {len(tasks_signatures)} 个历史数据保存子任务，Group ID: {group_result.id}"

    except Exception as e:
        logger.error(f"主任务执行期间发生错误: {e}", exc_info=True)
        # 让 Celery 知道主任务（分发器）失败了
        # raise
        return f"分发历史数据保存任务失败: {e}"
    finally:
        # 关闭主任务中使用的 DAO
        if stock_basic_dao:
            try:
                 # 假设 close 是异步的
                if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                     asyncio.run(stock_basic_dao.close())
                elif callable(getattr(stock_basic_dao, 'close', None)):
                     stock_basic_dao.close()
                logger.debug("主任务 DAO closed.")
            except Exception as close_err:
                logger.error(f"关闭主任务 DAO 时出错: {close_err}", exc_info=True)
        logger.info("主任务执行流程结束: dispatch_history_time_trade_saving (分发器)")



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

# --- 新增：处理单个股票单个时间级别指标计算的子任务 ---
@celery_app.task(bind=True, name='tasks.indicator_calculation.calculate_single_stock_indicator')
def calculate_single_stock_indicator(self, stock_code: str, time_level: str):
    """
    计算并保存单个股票在指定时间级别的指标 (子任务)
    """
    # logger.info(f"子任务启动: calculate_single_stock_indicator for {stock_code} ({time_level})")
    # 导入 Service，注意路径根据你的项目结构调整
    from services.indicator_services import BaseIndicatorService

    indicator_services = None # 初始化
    task_result = f"计算指标失败 for {stock_code} ({time_level})" # 默认失败结果

    try:
        # 在任务内部实例化 Service
        indicator_services = BaseIndicatorService()
        # 在同步的 Celery 任务中运行异步 Service 方法
        # 注意：这里假设 calculate_and_save_stock_indicators 是 async
        asyncio.run(indicator_services.calculate_and_save_stock(stock_code, time_level))
        task_result = f"成功计算并保存指标 for {stock_code} ({time_level})"
        logger.info(task_result)
    except Exception as e:
        logger.error(f"计算指标时发生错误 for {stock_code} ({time_level}): {e}", exc_info=True)
        task_result = f"计算指标失败 for {stock_code} ({time_level}): {e}"
        # 可以选择性地失败任务
        # self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
        # raise Ignore()
    finally:
        # 如果 IndicatorServices 需要显式关闭资源（比如内部持有的DAO），在这里处理
        # 例如:
        # if indicator_services and hasattr(indicator_services, 'close') and callable(indicator_services.close):
        #     try:
        #         # 假设 close 是同步的，如果是异步则用 asyncio.run
        #         indicator_services.close()
        #         logger.debug(f"IndicatorServices for {stock_code} ({time_level}) closed.")
        #     except Exception as close_err:
        #         logger.error(f"关闭 IndicatorServices for {stock_code} ({time_level}) 时出错: {close_err}", exc_info=True)
        logger.info(f"子任务结束: calculate_single_stock_indicator for {stock_code} ({time_level})")

    return task_result

# --- 新增：分发指标计算任务的主任务 ---
@celery_app.task(bind=True, name='tasks.indicator_calculation.dispatch_all')
def dispatch_indicator_calculation(self):
    """
    分发任务以并发计算所有股票在所有时间级别的指标 (主任务/分发器)
    """
    logger.info("主任务启动: dispatch_indicator_calculation (分发器)")
    # 导入 DAO，注意路径根据你的项目结构调整
    from dao_manager.daos.stock_basic_dao import StockBasicDAO

    stock_basic_dao = None # 初始化
    try:
        stock_basic_dao = StockBasicDAO()

        # 获取股票列表 (假设 get_stock_list 是 async)
        try:
            stocks = asyncio.run(stock_basic_dao.get_stock_list())
            logger.info(f"获取到 {len(stocks)} 支股票列表")
        except Exception as e:
             logger.error(f"获取股票列表时出错: {e}", exc_info=True)
             raise # 获取列表失败，则无法继续，直接抛出异常使主任务失败

        if not stocks:
            logger.warning("未获取到任何股票信息，任务结束")
            return "未获取到股票列表，无法分发任务"

        # 创建子任务签名列表
        tasks_signatures = []
        for stock in stocks:
            for time_level in TIME_LEVELS:
                # 为每个 stock_code 和 time_level 组合创建签名
                tasks_signatures.append(calculate_single_stock_indicator.s(stock.stock_code, time_level))

        if not tasks_signatures:
             logger.warning("没有生成任何子任务签名，任务结束")
             return "没有有效的股票或时间级别组合，未分发任务"

        # 创建任务组
        task_group = group(tasks_signatures)
        logger.info(f"已创建包含 {len(tasks_signatures)} 个指标计算子任务的任务组")

        # 异步执行任务组
        group_result = task_group.apply_async()

        logger.info(f"任务组已提交执行，Group ID: {group_result.id}")

        # 返回分发成功的消息
        return f"成功分发 {len(tasks_signatures)} 个指标计算子任务，Group ID: {group_result.id}"

    except Exception as e:
        logger.error(f"主任务执行期间发生错误: {e}", exc_info=True)
        # 让 Celery 知道主任务（分发器）失败了
        # raise
        return f"分发指标计算任务失败: {e}"
    finally:
        # 关闭主任务中使用的 DAO
        if stock_basic_dao:
            try:
                 # 假设 close 是异步的
                if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                     asyncio.run(stock_basic_dao.close())
                elif callable(getattr(stock_basic_dao, 'close', None)):
                     stock_basic_dao.close()
                logger.debug("主任务 DAO closed.")
            except Exception as close_err:
                logger.error(f"关闭主任务 DAO 时出错: {close_err}", exc_info=True)
        logger.info("主任务执行流程结束: dispatch_indicator_calculation (分发器)")

























