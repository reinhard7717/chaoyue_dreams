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
# 假设这是你要调用的服务
from services.indicator_services import IndicatorService
# 假设这是你要用的 DAO
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
STOCK_PROCESSING_BATCH_SIZE = 100  # 每个批处理任务处理的股票数量，根据内存和任务复杂度调整

# --- 新的批处理 Worker 任务 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.process_stock_batch_with_original_logic')
async def process_stock_batch_with_original_logic(self, stock_codes_batch):
    """
    处理一个批次的股票，执行原有的获取数据和计算指标逻辑。
    这是一个异步任务，因为它内部调用了异步的指标计算。
    """
    batch_size = len(stock_codes_batch)
    logger.info(f"任务启动: process_stock_batch_with_original_logic - 处理 {batch_size} 只股票")
    processed_count = 0
    error_count = 0
    from dao_manager.daos.stock_indicators_dao import StockIndicatorsDAO
    from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
    stock_indicators_dao = StockIndicatorsDAO()
    stock_realtime_dao = StockRealtimeDAO()
    # 在任务内部实例化，确保隔离性
    indicator_services = IndicatorService()
    # 注意：如果 process_single_stock... 是同步函数或非 Celery 任务，你需要相应调整调用方式

    for stock_code in stock_codes_batch:
        try:
            logger.debug(f"批处理 (原逻辑): 开始处理股票 {stock_code}")
            
            await stock_indicators_dao.fetch_and_save_latest_time_trade_trading_hours_by_stock_code(stock_code)
            await stock_realtime_dao.fetch_and_save_time_deals(stock_code)

            # --- 执行原循环内部的逻辑 ---
            # 1. 调用其他获取数据的任务 (如果它们是 Celery tasks)
            # process_single_stock_latest_trade_trading_hours.delay(stock_code)
            # process_single_stock_realtime_trade.delay(stock_code)
            # 注意： .delay() 是异步发送，计算指标的步骤会立即执行，
            # 如果计算指标依赖这些任务的结果，你需要更复杂的同步机制（如 .get() 或回调），
            # 或者更好的方式是将这些数据获取逻辑也直接包含在这个批处理任务中（如下面的注释示例）。

            # --- 或者，直接在此处执行数据获取逻辑 (推荐，保证顺序) ---
            # logger.debug(f"批处理 (原逻辑): 获取 {stock_code} 的最新交易数据...")
            # await fetch_latest_trade_data(stock_code) # 实现或调用相应函数
            # logger.debug(f"批处理 (原逻辑): 获取 {stock_code} 的实时交易数据...")
            # await fetch_realtime_trade_data(stock_code) # 实现或调用相应函数

            # 2. 计算并保存指标
            logger.debug(f"批处理 (原逻辑): 计算 {stock_code} 的指标...")
            for time_level in TIME_TEADE_TIME_LEVELS_LITE:
                await indicator_services.calculate_and_save_macd_indicators(stock_code, time_level)
                # 如果还有其他指标计算，也放在这里
                # await indicator_services.calculate_and_save_kdj_indicators(stock_code, time_level)

            # 3. （可选）执行策略逻辑
            # logger.debug(f"批处理 (原逻辑): 执行 {stock_code} 的策略...")
            # await run_strategy_for_stock(stock_code) # 实现或调用相应函数

            # --- 原循环内部逻辑结束 ---

            logger.debug(f"批处理 (原逻辑): 完成处理股票 {stock_code}")
            processed_count += 1

        except Exception as e:
            error_count += 1
            logger.error(f"批处理 (原逻辑): 处理股票 {stock_code} 时出错: {e}", exc_info=True)
            # 可以选择重试单只股票，但这会增加复杂度。通常是记录错误，继续处理批次内其他股票。
            # continue

    logger.info(f"任务结束: process_stock_batch_with_original_logic - 处理完成 {processed_count}/{batch_size} 只股票，失败 {error_count} 只")
    if error_count > 0:
        # 可以根据需要决定是否让整个批次任务失败
        # raise Exception(f"批处理任务中有 {error_count} 只股票处理失败")
        pass
    return f"批处理 (原逻辑) 完成: {processed_count} 成功, {error_count} 失败"


# --- 异步辅助函数：获取所有需要处理的股票代码 ---
#    （与上一个方案相同）
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表（自选+所有）"""
    stock_basic_dao = StockBasicDAO()
    stock_codes = set()
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        for fav in favorite_stocks:
            stock_codes.add(fav.stock_code)
        logger.info(f"获取到 {len(stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)

    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        original_count = len(stock_codes)
        for stock in all_stocks:
            stock_codes.add(stock.stock_code)
        logger.info(f"获取到 {len(all_stocks)} 个全市场股票代码，总计（去重后） {len(stock_codes)} 个")
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)

    if not stock_codes:
         logger.warning("未能获取到任何需要处理的股票代码")
         return []

    return list(stock_codes)


# --- 修改原有的任务，使其成为调度器 ---
@celery_app.task(bind=True, name='tasks.stock_indicators.get_trade_and_calculate_and_strategy')
def get_trade_and_calculate_and_strategy(self):
    """
    修改后的任务：获取所有相关股票代码，并分批派发给处理任务。
    这个任务由 Celery Beat 调度。
    """
    logger.info("任务启动: get_trade_and_calculate_and_strategy (调度器模式) - 获取股票列表并分派批处理任务")

    try:
        # 在同步任务中运行异步代码来获取列表
        stock_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not stock_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return "未获取到股票代码"

        total_stocks = len(stock_codes)
        num_batches = math.ceil(total_stocks / STOCK_PROCESSING_BATCH_SIZE)
        logger.info(f"获取到 {total_stocks} 个股票代码，将分 {num_batches} 批处理 (每批最多 {STOCK_PROCESSING_BATCH_SIZE} 只)")

        # 分批派发任务
        for i in range(num_batches):
            start_index = i * STOCK_PROCESSING_BATCH_SIZE
            end_index = start_index + STOCK_PROCESSING_BATCH_SIZE
            batch = stock_codes[start_index:end_index]

            if batch:
                # 为每个批次异步调用新的批处理任务
                process_stock_batch_with_original_logic.delay(batch)
                logger.info(f"已分派批次 {i+1}/{num_batches} ({len(batch)} 只股票) 的处理任务 (使用原逻辑)")
                # 可以在批次之间短暂休眠，避免瞬间冲击 Broker
                # import time
                # time.sleep(0.1)

        logger.info(f"任务结束: get_trade_and_calculate_and_strategy (调度器模式) - 所有 {num_batches} 个批处理任务已分派")
        return f"已为 {total_stocks} 只股票分派 {num_batches} 个批处理任务 (使用原逻辑)"

    except Exception as e:
        logger.error(f"执行 get_trade_and_calculate_and_strategy (调度器模式) 时出错: {e}", exc_info=True)
        # 可以添加重试逻辑
        # raise self.retry(exc=e, countdown=300, max_retries=1)
        return "调度任务执行失败"


















