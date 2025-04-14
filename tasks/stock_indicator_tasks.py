"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
import math
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from celery import Celery, chain, group # 导入 group
from celery.utils.log import get_task_logger
from core.constants import TIME_TEADE_TIME_LEVELS_LITE, TIME_TEADE_TIME_LEVELS_PER_TRADING
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService



logger = logging.getLogger('tasks')

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
        asyncio.run(stock_realtime_dao.fetch_and_save_time_deals(stock_code))
    except Exception as e:
        logger.error(f"处理股票 {stock_code} 数据时发生错误: {e}", exc_info=True)
        task_result = f"处理股票 {stock_code} 数据失败: {e}"
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

@celery_app.task(bind=True, name='tasks.stock_indicators.fetch_single_stock_history_trade_data')
def fetch_single_stock_history_trade_data(self, stock_code):
    """从数据库获取单只股票的历史交易数据并缓存
    Args:
        stock_code (str): 股票代码
    Returns:
        dict: 任务执行结果信息
    """
    logger.info(f"开始获取并缓存股票 {stock_code} 的历史交易数据")
    from utils.cash_key import StockCashKey
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    from dao_manager.daos.stock_basic_dao import StockBasicDAO
    from stock_models.stock_basic import StockTimeTrade
    from asgiref.sync import sync_to_async
    cache_limit = 333
    try:
        # 初始化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        stock_basic_dao = StockBasicDAO()
        # 获取单只股票信息
        stock = asyncio.run(stock_basic_dao.get_stock_by_code(stock_code))
        if not stock:
            logger.error(f"未找到股票代码 {stock_code} 的股票信息")
            return {"status": "error", "message": f"未找到股票代码 {stock_code} 的股票信息"}
        processed_count = 0
        # 处理不同的时间级别
        for time_level in TIME_TEADE_TIME_LEVELS_PER_TRADING:
            # 查询数据库获取历史数据
            datas = list(
                StockTimeTrade.objects.filter(
                    stock=stock, time_level=time_level
                ).order_by('-trade_time')[:cache_limit]
            )
            # 构建缓存键并记录日志
            cache_key = StockCashKey()
            cache_key_str = cache_key.history_time_trade(stock.stock_code, time_level)
            if datas:
                stock_indicators_dao.cache_manager.delete(cache_key_str)
                logger.info(f"缓存股票 {stock.stock_code} {time_level} 级别历史数据，共 {len(datas)} 条")
                for item in datas:
                    # 格式化数据并缓存
                    cache_data = stock_indicators_dao.data_format_process.set_time_trade_data(stock, time_level, item)
                    asyncio.run(stock_indicators_dao.cache_set.history_time_trade(stock.stock_code, time_level, cache_data))
                    processed_count += 1
                # 修剪缓存大小
                asyncio.run(stock_indicators_dao.cache_manager.trim_cache_zset(cache_key_str, cache_limit))
                logger.info(f"成功缓存股票 {stock.stock_code} {time_level} 级别历史数据，并修剪缓存大小为 {cache_limit}")
        
        logger.info(f"完成股票 {stock_code} 的历史交易数据缓存，共处理 {processed_count} 条记录")
        return {
            "status": "success",
            "message": f"完成股票 {stock_code} 的历史交易数据缓存",
            "processed_count": processed_count
        }
    except Exception as e:
        logger.exception(f"处理股票 {stock_code} 的历史交易数据时出错: {str(e)}")
        raise self.retry(exc=e, countdown=60)
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
        if stock_basic_dao:
            try:
                # 假设 close 是异步的
                if asyncio.iscoroutinefunction(getattr(stock_basic_dao, 'close', None)):
                     asyncio.run(stock_basic_dao.close())
                elif callable(getattr(stock_basic_dao, 'close', None)):
                     stock_basic_dao.close()
                logger.debug(f"DAO for {stock_code} closed.")
            except Exception as close_err:
                logger.error(f"关闭 DAO for {stock_code} 时出错: {close_err}", exc_info=True)

# --- 内部异步逻辑：计算单支股票 ---
async def _calculate_stock_indicators_async(stock_code: str):
    """实际执行异步计算的内部函数"""
    service = IndicatorService()
    logger.info(f"开始异步计算股票 {stock_code} 的指标...")
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
    logger.info(f"Celery 任务开始处理股票 {stock_code}...")
    # 使用 asyncio.run() 在同步任务中运行异步代码
    result = asyncio.run(_calculate_stock_indicators_async(stock_code))
    logger.info(f"Celery 任务完成处理股票 {stock_code}，结果: {result}")
    # 返回异步函数的结果，这个结果应该是可序列化的（字符串）
    return result


# --- 新增：定义细粒度任务的队列名称 ---
# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_save_api_data_TimeTrade'
FAVORITE_CALCULATE_INDICATORS_QUEUE = 'favorite_calculate_indicators'
FAVORITE_CALCULATE_STRATEGY_QUEUE = 'favorite_calculate_strategy'
# 非自选股队列
STOCKS_SAVE_API_DATA_QUEUE = 'save_api_data_TimeTrade'
STOCKS_CALCULATE_INDICATORS_QUEUE = 'calculate_indicators'
STOCKS_CALCULATE_STRATEGY_QUEUE = 'calculate_strategy'


# --- 新增：细粒度的 Celery Worker 任务 ---

@celery_app.task(bind=True, name='tasks.stock_processing.fetch_stock_api_data')
def fetch_stock_api_data_task(self, stock_code: str):
    """
    Celery 任务：获取单个股票的最新交易时间和分时成交数据。
    """
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
    queue_name = self.request.delivery_info.get('routing_key', '未知')
    logger.info(f"任务启动 (API数据): fetch_stock_api_data_task - 处理股票 {stock_code} (队列: {queue_name})")
    async def _run_async_fetch():
        stock_indicators_dao = StockIndicatorsDAO()
        stock_realtime_dao = StockRealtimeDAO()
        try:
            logger.debug(f"API数据: 开始获取 {stock_code} 的交易数据...")
            await stock_indicators_dao.fetch_and_save_latest_time_trade_trading_hours_by_stock_code(stock_code)
            await stock_realtime_dao.fetch_and_save_time_deals(stock_code)
            logger.debug(f"API数据: 完成获取 {stock_code} 的交易数据。")
            return True
        except Exception as e:
            logger.error(f"API数据: 获取股票 {stock_code} 数据时出错: {e}", exc_info=True)
            return False # 返回失败状态，链会停止
    try:
        success = asyncio.run(_run_async_fetch())
        if success:
            logger.info(f"任务成功 (API数据): fetch_stock_api_data_task - 完成处理股票 {stock_code}")
            return stock_code # 成功时传递股票代码给下一个任务
        else:
            logger.error(f"任务失败 (API数据): fetch_stock_api_data_task - 处理股票 {stock_code} 失败")
            # 抛出异常以确保链停止 (或者根据需要决定是否继续)
            raise Exception(f"Failed to fetch API data for {stock_code}")
    except Exception as e:
        logger.error(f"执行 fetch_stock_api_data_task (同步包装器) 时出错: {e}", exc_info=True)
        # Celery 会自动处理重试或记录失败，这里可以选择重新抛出
        raise # 重新抛出异常，标记任务失败

@celery_app.task(bind=True, name='tasks.stock_processing.calculate_stock_indicators')
def calculate_stock_indicators_task(self, stock_code: str):
    """
    Celery 任务：计算单个股票的 MACD 指标。
    假定上一个任务成功时会传递 stock_code。
    """
    if not stock_code: # 检查上一个任务是否成功传递了 stock_code
         logger.warning(f"任务跳过 (指标计算): calculate_stock_indicators_task - 未收到有效的 stock_code (可能前序任务失败)")
         return None # 或者根据需要处理
    queue_name = self.request.delivery_info.get('routing_key', '未知')
    logger.info(f"任务启动 (指标计算): calculate_stock_indicators_task - 处理股票 {stock_code} (队列: {queue_name})")
    async def _run_async_calculate():
        indicator_services = IndicatorService()
        try:
            logger.debug(f"指标计算: 开始计算 {stock_code} 的指标...")
            # 注意：这里只计算了 MACD，如果需要计算其他指标，也应在此处添加
            for time_level in TIME_TEADE_TIME_LEVELS_LITE:
                 await indicator_services.calculate_and_save_macd_indicators(stock_code, time_level)
            logger.debug(f"指标计算: 完成计算 {stock_code} 的指标。")
            return True
        except Exception as e:
            logger.error(f"指标计算: 计算股票 {stock_code} 指标时出错: {e}", exc_info=True)
            return False

    try:
        success = asyncio.run(_run_async_calculate())
        if success:
            logger.info(f"任务成功 (指标计算): calculate_stock_indicators_task - 完成处理股票 {stock_code}")
            return stock_code # 成功时传递股票代码给下一个任务
        else:
            logger.error(f"任务失败 (指标计算): calculate_stock_indicators_task - 处理股票 {stock_code} 失败")
            raise Exception(f"Failed to calculate indicators for {stock_code}")
    except Exception as e:
        logger.error(f"执行 calculate_stock_indicators_task (同步包装器) 时出错: {e}", exc_info=True)
        raise

# Celery 任务：对单个股票执行策略计算。
@celery_app.task(bind=True, name='tasks.stock_processing.run_stock_strategy')
def run_stock_strategy_task(self, stock_code: str):
    """
    Celery 任务：对单个股票执行策略计算。
    假定上一个任务成功时会传递 stock_code。
    """
    from tasks.strategy_tasks import run_strategy_for_single_stock_task
    if not stock_code:
         logger.warning(f"任务跳过 (策略计算): run_stock_strategy_task - 未收到有效的 stock_code (可能前序任务失败)")
         return None
    queue_name = self.request.delivery_info.get('routing_key', '未知')
    logger.info(f"任务启动 (策略计算): run_stock_strategy_task - 处理股票 {stock_code} (队列: {queue_name})")
    async def _run_async_strategy():
        try:
            logger.debug(f"策略计算: 开始运行 {stock_code} 的策略...")
            await run_strategy_for_single_stock_task(stock_code)
            logger.debug(f"策略计算: 完成运行 {stock_code} 的策略。")
            return True
        except Exception as e:
            logger.error(f"策略计算: 运行股票 {stock_code} 策略时出错: {e}", exc_info=True)
            return False
    try:
        success = asyncio.run(_run_async_strategy()) # 假设策略函数是异步的
        if success:
            logger.info(f"任务成功 (策略计算): run_stock_strategy_task - 完成处理股票 {stock_code}")
            return f"Strategy calculation completed for {stock_code}" # 链的最终结果
        else:
            logger.error(f"任务失败 (策略计算): run_stock_strategy_task - 处理股票 {stock_code} 失败")
            raise Exception(f"Failed to run strategy for {stock_code}")
    except Exception as e:
        logger.error(f"执行 run_stock_strategy_task (同步包装器) 时出错: {e}", exc_info=True)
        raise

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
    logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
         logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.get_trade_and_calculate_and_strategy')
def get_trade_and_calculate_and_strategy(self):
    """
    修改后的调度器任务：
    1. 获取自选股和非自选股代码。
    2. 为每只股票创建任务链 (获取数据 -> 计算指标 -> 执行策略)。
    3. 将任务链中的任务分派到指定的队列 (favorite_* 或 stocks_*/calculate_*)。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: get_trade_and_calculate_and_strategy (调度器模式) - 获取股票列表并分派细粒度任务链")
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
                fetch_stock_api_data_task.s(stock_code).set(queue=FAVORITE_SAVE_API_DATA_QUEUE),
                calculate_stock_indicators_task.s().set(queue=FAVORITE_CALCULATE_INDICATORS_QUEUE), # 注意：后续任务参数由前序任务传递
                run_stock_strategy_task.s().set(queue=FAVORITE_CALCULATE_STRATEGY_QUEUE)
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
                fetch_stock_api_data_task.s(stock_code).set(queue=STOCKS_SAVE_API_DATA_QUEUE),
                calculate_stock_indicators_task.s().set(queue=STOCKS_CALCULATE_INDICATORS_QUEUE),
                run_stock_strategy_task.s().set(queue=STOCKS_CALCULATE_STRATEGY_QUEUE)
            )
            # 异步执行任务链
            task_chain.apply_async()
            total_dispatched_chains += 1
            logger.debug(f"已分派非自选股 {stock_code} 的任务链")
            # 可选：短暂延迟
            # import time
            # time.sleep(0.01)

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派任务链。")

        logger.info(f"任务结束: get_trade_and_calculate_and_strategy (调度器模式) - 共分派 {total_dispatched_chains} 个任务链")
        return f"已为 {total_favorite_stocks} 自选股和 {total_non_favorite_stocks} 非自选股分派 {total_dispatched_chains} 个任务链"

    except Exception as e:
        logger.error(f"执行 get_trade_and_calculate_and_strategy (调度器模式) 时出错: {e}", exc_info=True)
        # 可以考虑重试机制
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"













