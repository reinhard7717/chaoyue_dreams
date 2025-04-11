"""
指数相关任务
提供指数数据的定时更新和手动触发任务
"""
import asyncio
import logging
import math
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from celery import Celery, group # 导入 group
from celery.utils.log import get_task_logger
from core.constants import TIME_TEADE_TIME_LEVELS, TIME_TEADE_TIME_LEVELS_LITE
from dao_manager.daos.stock_basic_dao import StockBasicDAO
from services.indicator_services import IndicatorService

logger = logging.getLogger("celery")

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

@celery_app.task(bind=True, name='tasks.stock_indicators.fetch_single_stock_history_trade_data', max_retries=3)
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
    cache_limit = 233 * 3
    try:
        # 初始化 DAO
        stock_indicators_dao = StockIndicatorsDAO()
        stock_basic_dao = StockBasicDAO()
        # 获取单只股票信息
        stock = stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.error(f"未找到股票代码 {stock_code} 的股票信息")
            return {"status": "error", "message": f"未找到股票代码 {stock_code} 的股票信息"}
        processed_count = 0
        # 处理不同的时间级别
        for time_level in TIME_TEADE_TIME_LEVELS_LITE:
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
                logger.info(f"缓存股票 {stock.stock_code} {time_level} 级别历史数据，共 {len(datas)} 条")
                for item in datas:
                    # 格式化数据并缓存
                    cache_data = stock_indicators_dao.data_format_process.set_time_trade_data(stock, time_level, item)
                    stock_indicators_dao.cache_set.history_time_trade(stock.stock_code, time_level, cache_data)
                    processed_count += 1
                # 修剪缓存大小
                stock_indicators_dao.cache_manager.trim_cache_zset(cache_key_str, cache_limit)
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
            for time_level in TIME_TEADE_TIME_LEVELS_LITE
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

# --- 配置 ---
STOCK_PROCESSING_BATCH_SIZE = 100  # 每个批处理任务处理的股票数量
PRIORITY_QUEUE_NAME = 'priority_tasks' # 定义高优先级队列名称
DEFAULT_QUEUE_NAME = 'celery' # 定义默认队列名称 (或者你的默认队列名)

# --- 批处理 Worker 任务 (保持不变) ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_stock_batch_with_original_logic')
def process_stock_batch_with_original_logic(self, stock_codes_batch):
    """
    处理一个批次的股票（同步任务包装器）。
    内部使用 asyncio.run() 执行异步逻辑。
    (代码内容和之前一样，这里省略以保持简洁)
    """
    # ... (之前的 process_stock_batch_with_original_logic 实现) ...
    batch_size = len(stock_codes_batch)
    logger.info(f"任务启动 (同步包装器): process_stock_batch_with_original_logic - 处理 {batch_size} 只股票 (队列: {self.request.delivery_info.get('routing_key', '未知')})")
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
    from services.indicator_services import IndicatorService
    from tasks.strategy_tasks import strategy_macd_rsi_kdj_boll_strategy_for_stock

    async def _run_async_batch_logic(batch):
        processed_count = 0
        error_count = 0
        indicator_services = IndicatorService()
        stock_indicators_dao = StockIndicatorsDAO()
        stock_realtime_dao = StockRealtimeDAO()

        for stock_code in batch:
            try:
                logger.debug(f"批处理 (异步逻辑): 开始处理股票 {stock_code}")
                await stock_indicators_dao.fetch_and_save_latest_time_trade_trading_hours_by_stock_code(stock_code)
                await stock_realtime_dao.fetch_and_save_time_deals(stock_code)
                logger.debug(f"批处理 (异步逻辑): 计算 {stock_code} 的指标...")
                for time_level in TIME_TEADE_TIME_LEVELS_LITE:
                    await indicator_services.calculate_and_save_macd_indicators(stock_code, time_level)
                    # 其他指标计算...
                await strategy_macd_rsi_kdj_boll_strategy_for_stock(stock_code)
                logger.debug(f"批处理 (异步逻辑): 完成处理股票 {stock_code}")
                processed_count += 1
            except Exception as e:
                error_count += 1
                logger.error(f"批处理 (异步逻辑): 处理股票 {stock_code} 时出错: {e}", exc_info=True)
        return processed_count, error_count

    try:
        processed_count, error_count = asyncio.run(_run_async_batch_logic(stock_codes_batch))
        logger.info(f"任务结束 (同步包装器): process_stock_batch_with_original_logic - 处理完成 {processed_count}/{batch_size} 只股票，失败 {error_count} 只")
        if error_count > 0:
            pass
        return f"批处理 (原逻辑) 完成: {processed_count} 成功, {error_count} 失败"
    except Exception as e:
        logger.error(f"执行 process_stock_batch_with_original_logic (同步包装器) 时出错: {e}", exc_info=True)
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
            favorite_stock_codes.add(fav.stock_code)
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
    修改后的任务：获取自选股和非自选股代码，优先分派自选股任务到高优先级队列，
    然后分派非自选股任务到默认队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: get_trade_and_calculate_and_strategy (调度器模式) - 获取股票列表并按优先级分派批处理任务")
    try:
        # 在同步任务中运行异步代码来获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())
        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return "未获取到股票代码"
        total_dispatched_tasks = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)
        # 1. 分派自选股任务到高优先级队列
        if favorite_codes:
            num_fav_batches = math.ceil(total_favorite_stocks / STOCK_PROCESSING_BATCH_SIZE)
            logger.info(f"准备分派 {total_favorite_stocks} 个自选股，分为 {num_fav_batches} 批，发送到队列 '{PRIORITY_QUEUE_NAME}'")
            for i in range(num_fav_batches):
                start_index = i * STOCK_PROCESSING_BATCH_SIZE
                end_index = start_index + STOCK_PROCESSING_BATCH_SIZE
                batch = favorite_codes[start_index:end_index]
                if batch:
                    # 使用 apply_async 并指定 queue 参数
                    process_stock_batch_with_original_logic.apply_async(
                        args=[batch],
                        queue=PRIORITY_QUEUE_NAME
                    )
                    logger.info(f"已分派自选股批次 {i+1}/{num_fav_batches} ({len(batch)} 只股票) 到队列 '{PRIORITY_QUEUE_NAME}'")
                    total_dispatched_tasks += 1
                    # import time # 如果需要，可以稍微延迟
                    # time.sleep(0.05)
        else:
            logger.info("没有自选股需要处理。")
        # 2. 分派非自选股任务到默认队列
        if non_favorite_codes:
            num_non_fav_batches = math.ceil(total_non_favorite_stocks / STOCK_PROCESSING_BATCH_SIZE)
            logger.info(f"准备分派 {total_non_favorite_stocks} 个非自选股，分为 {num_non_fav_batches} 批，发送到队列 '{DEFAULT_QUEUE_NAME}'")
            for i in range(num_non_fav_batches):
                start_index = i * STOCK_PROCESSING_BATCH_SIZE
                end_index = start_index + STOCK_PROCESSING_BATCH_SIZE
                batch = non_favorite_codes[start_index:end_index]
                if batch:
                    # 使用 apply_async 或 delay 发送到默认队列
                    process_stock_batch_with_original_logic.apply_async(
                        args=[batch],
                        queue=DEFAULT_QUEUE_NAME # 或者直接用 .delay() 如果 DEFAULT_QUEUE_NAME 就是 Celery 的默认队列 'celery'
                        # process_stock_batch_with_original_logic.delay(batch) # 如果默认队列就是 'celery'
                    )
                    logger.info(f"已分派非自选股批次 {i+1}/{num_non_fav_batches} ({len(batch)} 只股票) 到队列 '{DEFAULT_QUEUE_NAME}'")
                    total_dispatched_tasks += 1
                    # import time # 如果需要，可以稍微延迟
                    # time.sleep(0.05)
        else:
            logger.info("没有非自选股需要处理。")
        logger.info(f"任务结束: get_trade_and_calculate_and_strategy (调度器模式) - 共分派 {total_dispatched_tasks} 个批处理任务")
        return f"已为 {total_favorite_stocks} 自选股和 {total_non_favorite_stocks} 非自选股分派 {total_dispatched_tasks} 个批处理任务"
    except Exception as e:
        logger.error(f"执行 get_trade_and_calculate_and_strategy (调度器模式) 时出错: {e}", exc_info=True)
        # raise self.retry(exc=e, countdown=300, max_retries=1) # 可以考虑重试
        return "调度任务执行失败"















