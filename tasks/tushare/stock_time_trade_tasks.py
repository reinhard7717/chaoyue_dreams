# tasks\stock_time_trade_tasks.py
import asyncio
import logging
import datetime
import time
from typing import List
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from stock_models.stock_basic import StockInfo

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveData_TimeTrade'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveData_TimeTrade'
logger = logging.getLogger('tasks')

def is_trading_time():
    now = datetime.datetime.now()
    # 交易日判断略，假设已是交易日
    if now.hour in [9, 10, 11, 13, 14, 15, 16]:
        if now.hour == 11 and now.minute >= 30:
            return False
        if now.hour == 9 and now.minute < 30:
            return False
        if now.hour == 16 and now.minute >= 2:
            return False
        return True
    return False

# 获取本周一和本周五的日期
def get_this_monday_and_friday():
    """获取本周一和本周五的日期"""
    today = datetime.date.today()
    this_monday = today - datetime.timedelta(days=today.weekday())
    this_friday = this_monday + datetime.timedelta(days=4)
    return this_monday, this_friday

# 获取上周一和上周五的日期
def get_last_monday_and_friday():
    """获取上周一和上周五的日期"""
    today = datetime.date.today()
    this_monday = today - datetime.timedelta(days=today.weekday())
    last_monday = this_monday - datetime.timedelta(days=7)
    last_friday = last_monday + datetime.timedelta(days=4)
    return last_monday, last_friday

# --- 异步辅助函数：获取需要处理的股票代码 (区分自选和非自选) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    stock_basic_dao = StockBasicInfoDao()
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
    # logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
         logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes

# ===================================================
#                      当日任务
# ===================================================
#  ================ 整体任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.run_daily_data_ingestion_task')
def run_daily_data_ingestion_task(self, trade_time_str=None):
    """
    整体任务：按顺序执行当日收盘后的数据采集任务。
    包括：分钟数据、日线数据（含筹码）、当日基本信息。
    这个任务由 Celery Beat 调度，通常在收盘后执行。
    """
    logger.info("整体任务启动: run_daily_data_ingestion_task - 开始执行当日数据采集流程")

    try:
        # 步骤 1: 执行分钟数据采集调度任务
        # 这个任务会获取所有股票并分批派发 save_stocks_minute_data_today_batch
        logger.info("开始执行: 分钟数据采集调度任务...")
        # 使用 .delay() 或 .apply_async() 异步触发子任务
        # .delay() 是 .apply_async() 的简化版
        minute_task_result = save_stocks_minute_data_today_task.delay(trade_time_str=trade_time_str)
        logger.info(f"已分派分钟数据采集调度任务。任务ID: {minute_task_result.id}")

        # 注意：这里使用 .delay() 意味着整体任务会立即返回，而子任务会在后台异步执行。
        # 如果你需要严格等待前一个任务完成后再开始下一个，你需要使用 Celery 的 Chain 或 Group/Chord，
        # 或者在当前任务中调用子任务的 .get() 方法（但这会阻塞当前 worker，不推荐用于长时间任务）。
        # 对于数据采集，通常分派出去即可，让 worker 自己处理并发和顺序（如果子任务内部有依赖）。
        # 当前设计是整体任务负责“分派”子任务，子任务各自执行。

        # 步骤 2: 执行日线数据（含筹码）采集任务
        # 这个任务会采集日线数据，并且内部包含了筹码数据的采集
        logger.info("开始执行: 日线数据（含筹码）采集任务...")
        daily_data_task_result = save_day_data_today_task.delay()
        logger.info(f"已分派日线数据（含筹码）采集任务。任务ID: {daily_data_task_result.id}")

        # 步骤 3: 执行当日基本信息采集任务
        logger.info("开始执行: 当日基本信息采集任务...")
        daily_basic_task_result = save_stocks_daily_basic_data_today_task.delay()
        logger.info(f"已分派当日基本信息采集任务。任务ID: {daily_basic_task_result.id}")

        logger.info("整体任务结束: run_daily_data_ingestion_task - 所有当日数据采集任务已分派。")

        # 返回一些信息，方便查看任务状态
        return {
            "status": "success",
            "message": "所有当日数据采集任务已分派",
            "dispatched_tasks": {
                "minute_data_scheduler": minute_task_result.id,
                "daily_data_and_chips": daily_data_task_result.id,
                "daily_basic_info": daily_basic_task_result.id,
            }
        }

    except Exception as e:
        logger.error(f"整体任务 run_daily_data_ingestion_task 执行失败: {e}", exc_info=True)
        # 记录异常并返回错误状态
        return {"status": "error", "message": f"整体任务执行失败: {e}"}

#  ================ 分钟数据任务（当日收盘后） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_today_batch')
def save_stocks_minute_data_today_batch(self, stock_codes, trade_time_str=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        print("开始保存 分钟数据任务（当日）...")
        result = asyncio.run(stock_time_trade_dao.save_minute_time_trade_history_today(stock_codes=stock_codes,trade_time_str=trade_time_str))
        print(f"保存股票 {len(stock_codes)} 个的分钟级交易数据完成。结果: {result}")
    except Exception as e:
        logger.error(f"save_stocks_minute_data_today_batch.执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_today_task')
def save_stocks_minute_data_today_task(self, trade_time_str=None, batch_size: int = 310): # 最大循环10万个，每310个一组循环一次是99510个
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_minute_data_today_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        total_dispatched_batches = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        all_stock_codes = [stock.stock_code for stock in all_stocks]
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        total_codes_count = len(all_stocks)  # 用于统计总代码数量
        logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
        for i in range(0, total_codes_count, batch_size):
            batch_codes = all_stock_codes[i:i + batch_size]
            if batch_codes:
                logger.info(f"创建自选股批次任务 (大小: {len(batch_codes)})...")
                # 使用新的批量任务，并指定队列
                save_stocks_minute_data_today_batch.s(stock_codes=batch_codes, trade_time_str=trade_time_str).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch_codes)-1})")
        logger.info(f"已为 {total_codes_count} 个股票分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_stocks_minute_data_today_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_minute_data_today_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 今日基本信息 数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_today_task', queue='SaveData_TimeTrade')
def save_stocks_daily_basic_data_today_task(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    logger.info(f"开始处理今日股票重要的基本面指标...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        print("开始保存 今日股票重要的基本面指标...")
        result = asyncio.run(stock_time_trade_dao.save_today_stock_basic_info())
        print(f"保存 今日股票重要的基本面指标 完成。result: {result}")
    except Exception as e:
        logger.error(f"save_stocks_daily_basic_data_today_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

#  ================ 日线数据任务（当日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_day_data_today_task', queue='SaveData_TimeTrade')
def save_day_data_today_task(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        print("开始保存 日线数据任务（当日）...")
        result = asyncio.run(stock_time_trade_dao.save_daily_time_trade_today())
        print(f"保存 日线数据任务（当日） 完成。result: {result}")
        result = asyncio.run(stock_time_trade_dao.save_today_cyq_chips())
        print(f"保存 每日筹码分布 数据完成。 result: {result} ")
        result = asyncio.run(stock_time_trade_dao.save_today_cyq_perf())
        print(f"保存 每日筹码及胜率 数据完成。 result: {result} ")
    except Exception as e:
        logger.error(f"save_day_data_history_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

# ============== 每日筹码分布任务（当日） ==============
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_chips_today_batch', queue='SaveData_TimeTrade')
def save_cyq_chips_today_batch(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        result = asyncio.run(stock_time_trade_dao.save_today_cyq_chips())
        print(f"保存 每日筹码分布 数据完成。 result: {result} ")
    except Exception as e:
        logger.error(f"save_day_data_history_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_perf_today_batch', queue='SaveData_TimeTrade')
def save_cyq_perf_today_batch(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        result = asyncio.run(stock_time_trade_dao.save_today_cyq_perf())
        print(f"保存 每日筹码及胜率 数据完成。 result: {result} ")
    except Exception as e:
        logger.error(f"save_day_data_history_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_today_task')
def save_cyq_data_today_task(self):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_cyq_data_today_task (调度器模式) - 获取股票列表并分派批量任务")
    try:
        save_cyq_chips_today_batch.s().set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
        save_cyq_perf_today_batch.s().set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
        logger.info(f"任务结束: save_cyq_data_today_task (调度器模式)")
    except Exception as e:
        logger.error(f"执行 save_cyq_data_today_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

# ===================================================
#                      本周任务
# ===================================================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.run_this_week_data_ingestion_task', queue='celery')
def run_this_week_data_ingestion_task(self):
    """
    整体任务：按顺序执行本周的数据采集任务。
    包括：分钟数据、日线数据、每日基本信息、每日筹码数据。
    这个任务由 Celery Beat 调度，通常在周末或周初执行。
    """
    logger.info("整体任务启动: run_this_week_data_ingestion_task - 开始执行本周数据采集流程")
    try:
        # 步骤 1: 执行本周分钟数据采集调度任务
        # 这个任务会获取所有股票并分派 save_minute_data_this_week_batch 给每个股票和时间级别
        logger.info("开始执行: 本周分钟数据采集调度任务...")
        minute_weekly_task_result = save_stocks_minute_data_this_week_task.delay()
        logger.info(f"已分派本周分钟数据采集调度任务。任务ID: {minute_weekly_task_result.id}")
        # 步骤 2: 执行本周日线数据采集任务
        # 这个任务会采集本周的日线数据
        logger.info("开始执行: 本周日线数据采集任务...")
        daily_weekly_task_result = save_day_data_this_week_batch.delay()
        logger.info(f"已分派本周日线数据采集任务。任务ID: {daily_weekly_task_result.id}")
        # 步骤 3: 执行本周每日基本信息采集任务
        # 这个任务会循环采集本周每一天的基本信息
        logger.info("开始执行: 本周每日基本信息采集任务...")
        daily_basic_weekly_task_result = save_stocks_daily_basic_data_this_week_task.delay()
        logger.info(f"已分派本周每日基本信息采集任务。任务ID: {daily_basic_weekly_task_result.id}")
        # 步骤 4: 执行本周每日筹码数据采集调度任务
        # 这个任务会循环分派本周每一天的筹码数据采集任务
        logger.info("开始执行: 本周每日筹码数据采集调度任务...")
        cyq_weekly_task_result = save_cyq_data_this_week_task.delay()
        logger.info(f"已分派本周每日筹码数据采集调度任务。任务ID: {cyq_weekly_task_result.id}")

        logger.info("整体任务结束: run_this_week_data_ingestion_task - 所有本周数据采集任务已分派。")

        # 返回一些信息，方便查看任务状态
        return {
            "status": "success",
            "message": "所有本周数据采集任务已分派",
            "dispatched_tasks": {
                "minute_data_weekly_scheduler": minute_weekly_task_result.id,
                "daily_data_weekly": daily_weekly_task_result.id,
                "daily_basic_info_weekly": daily_basic_weekly_task_result.id,
                "cyq_data_weekly_scheduler": cyq_weekly_task_result.id,
            }
        }

    except Exception as e:
        logger.error(f"整体任务 run_this_week_data_ingestion_task 执行失败: {e}", exc_info=True)
        # 记录异常并返回错误状态
        return {"status": "error", "message": f"整体任务执行失败: {e}"}

#  ================ 分钟数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_minute_data_this_week_batch')
def save_minute_data_this_week_batch(self, stock_codes: str):
    """
    分钟数据任务（本周）
    从Tushare批量获取数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    this_monday, this_friday = get_this_monday_and_friday()
    this_monday_str = this_monday.strftime('%Y%m%d') + " 00:00:00"
    this_friday_str = this_friday.strftime('%Y%m%d') + " 16:00:00"
    try:
        result = asyncio.run(stock_time_trade_dao.save_minute_time_trade_history_by_stock_codes(stock_codes=stock_codes, start_date=this_monday_str, end_date=this_friday_str))
        logger.info(f"保存股票 {stock_codes} 的分钟级交易数据完成. 结果: {result}")
    except Exception as e:
        logger.error(f"save_minute_data_this_week_batch.执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_this_week_task')
def save_stocks_minute_data_this_week_task(self, batch_size: int = 50): # 限量：单次最大8000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_minute_data_today_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        total_dispatched_batches = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        all_stock_codes = [stock.stock_code for stock in all_stocks]
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        total_codes_count = len(all_stocks)  # 用于统计总代码数量
        logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
        for i in range(0, total_codes_count, batch_size):
            batch_codes = all_stock_codes[i:i + batch_size]
            if batch_codes:
                logger.info(f"创建自选股批次任务 (大小: {len(batch_codes)})...")
                # 使用新的批量任务，并指定队列
                save_minute_data_this_week_batch.s(stock_codes=batch_codes).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch_codes)-1})")
        logger.info(f"已为 {total_codes_count} 个股票分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_stocks_minute_data_today_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_minute_data_today_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 日线数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_day_data_this_week_batch')
def save_day_data_this_week_batch(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    this_monday, this_friday = get_this_monday_and_friday()
    logger.info(f"开始处理包含 {this_monday} - {this_friday} 历史(日线)数据任务...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        return_info = asyncio.run(stock_time_trade_dao.save_daily_time_trade_history_by_trade_dates(start_date=this_monday, end_date=this_friday))
        print(f"完成 {this_monday} - {this_friday} 的日线数据保存，{return_info}。")
        time.sleep(5)
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

#  ================ 每日基本信息数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_this_week_task', queue='SaveData_TimeTrade')
def save_stocks_daily_basic_data_this_week_task(self):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    logger.info(f"开始处理今日股票重要的基本面指标...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    this_monday, this_friday = get_this_monday_and_friday()
    for i in range(5):
        day = this_monday + datetime.timedelta(days=i)
        try:
            print(f"开始保存 {day} 今日股票重要的基本面指标...")
            result = asyncio.run(stock_time_trade_dao.save_today_stock_basic_info(trade_date=day))
            print(f"保存 {day} 今日股票重要的基本面指标 完成。result: {result}")
        except Exception as e:
            logger.error(f"save_stocks_daily_basic_data_this_week_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

# ============== 每日筹码分布任务 ==============
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_chips_this_week_batch', queue='SaveData_TimeTrade')
def save_cyq_chips_this_week_batch(self, trade_date: datetime.date):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        result = asyncio.run(stock_time_trade_dao.save_today_cyq_chips(trade_date=trade_date))
        print(f"保存 每日筹码分布 数据完成。 result: {result} ")
    except Exception as e:
        logger.error(f"save_day_data_history_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_perf_this_week_batch', queue='SaveData_TimeTrade')
def save_cyq_perf_this_week_batch(self, trade_date: datetime.date):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        result = asyncio.run(stock_time_trade_dao.save_today_cyq_perf(trade_date=trade_date))
        print(f"保存 每日筹码及胜率 数据完成。 result: {result} ")
    except Exception as e:
        logger.error(f"save_day_data_history_task.执行批量保存任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_this_week_task')
def save_cyq_data_this_week_task(self):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_cyq_data_this_week_task (调度器模式) - 获取股票列表并分派批量任务")
    try:
        this_monday, this_friday = get_this_monday_and_friday()
        for i in range(5):
            day = this_monday + datetime.timedelta(days=i)
            save_cyq_chips_this_week_batch.s(trade_date=day).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
            save_cyq_perf_this_week_batch.s(trade_date=day).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
        logger.info(f"任务结束: save_cyq_data_this_week_task (调度器模式)")
    except Exception as e:
        logger.error(f"执行 save_cyq_data_this_week_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


# ===================================================
#                      历史任务
# ===================================================

#  ================ 分钟数据任务（历史） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_minute_data_history_batch')
def save_minute_data_history_batch(self, stock_code: str, time_level: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_code:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理股票{stock_code}的 历史({time_level}分钟)数据任务...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        result = asyncio.run(stock_time_trade_dao.save_minute_time_trade_history_by_stock_code_and_time_level(stock_code, time_level))
        logger.info(f"保存股票 {stock_code} 的{time_level}分钟级交易数据完成. 结果: {result}")
    except Exception as e:
        logger.error(f"save_minute_data_history_batch.执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_history_task')
def save_stocks_minute_data_history_task(self): # 限量：单次最大8000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_minute_data_history_task (调度器模式) - 获取股票列表并分派批量任务.")
    try:
        total_dispatched_batches = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        all_stock_codes = [stock.stock_code for stock in all_stocks][::-1]
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        total_codes_count = len(all_stocks)  # 用于统计总代码数量
        logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
        for stock_code in all_stock_codes:
            logger.info(f"创建自选股任务 ({stock_code})...")
            # 使用新的批量任务，并指定队列
            for time_level in ["5", "15", "30", "60"]:
                save_minute_data_history_batch.s(stock_code=stock_code, time_level=time_level).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
        logger.info(f"已为 {total_codes_count} 个股票分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_stocks_minute_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_minute_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 日线数据任务（历史） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_day_data_history_batch')
def save_day_data_history_batch(self, stock_codes: List[str]):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(日线)数据任务...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        return_info = asyncio.run(stock_time_trade_dao.save_daily_time_trade_history_by_stock_codes(stock_codes))
        print(f"完成 {len(stock_codes)} 个股票的日线数据保存，{return_info}。")
        time.sleep(5)
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_day_data_history_task')
def save_stocks_day_data_history_task(self, batch_size: int = 13): # 限量：单次最大6000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_day_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_day_data_history_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        total_dispatched_batches = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        all_stock_codes = [stock.stock_code for stock in all_stocks]
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        total_codes_count = len(all_stocks)  # 用于统计总代码数量
        logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
        for i in range(0, total_codes_count, batch_size):
            batch_codes = all_stock_codes[i:i + batch_size]
            if batch_codes:
                logger.info(f"创建自选股批次任务 (大小: {len(batch_codes)})...")
                # 使用新的批量任务，并指定队列
                save_day_data_history_batch.s(stock_codes=batch_codes).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch_codes)-1})")
        logger.info(f"已为 {total_codes_count} 个股票分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_stocks_day_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_day_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 每日基本信息数据任务（历史） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_basic_data_history_batch')
def save_daily_basic_data_history_batch(self, stock_codes: List[str]):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理{','.join(stock_codes)} 历史(每日基本信息)数据任务...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_stock_daily_basic_history_by_stock_codes(stock_codes))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_history_task')
def save_stocks_daily_basic_data_history_task(self, batch_size: int = 16):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_daily_basic_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_daily_basic_data_history_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        total_dispatched_batches = 0
        stock_basic_dao = StockBasicInfoDao()
        all_stocks = asyncio.run(stock_basic_dao.get_stock_list())
        all_stock_codes = [stock.stock_code for stock in all_stocks]
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        total_codes_count = len(all_stocks)  # 用于统计总代码数量
        logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
        for i in range(0, total_codes_count, batch_size):
            batch_codes = all_stock_codes[i:i + batch_size]
            if batch_codes:
                logger.info(f"创建自选股批次任务 (大小: {len(batch_codes)})...")
                # 使用新的批量任务，并指定队列
                save_daily_basic_data_history_batch.s(stock_codes=batch_codes).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch_codes)-1})")
        logger.info(f"已为 {total_codes_count} 个股票分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_stocks_daily_basic_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_daily_basic_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}
    except Exception as e:
        logger.error(f"执行 save_stocks_daily_basic_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 历史(周线)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_week_data_history_batch')
def save_week_data_history_batch(self, stock_codes: List[str], time_level: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(周线)数据任务 批次...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_weekly_time_trade_by_stock_codes(stock_codes))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_week_data_history_task')
def save_stocks_week_data_history_task(self, batch_size: int = 5): # 限量：单次最大4500行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        # 在同步任务中运行异步代码获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return {"status": "warning", "message": "未获取到股票代码", "dispatched_batches": 0}

        total_dispatched_batches = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股批量任务
        logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
        for i in range(0, total_favorite_stocks, batch_size):
            batch = favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_week_data_history_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_week_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


#  ================ 历史(月线)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_month_data_history_batch')
def save_month_data_history_batch(self, stock_codes: List[str], time_level: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(月线)数据任务 批次...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_monthly_time_trade_by_stock_codes(stock_codes))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_month_data_history_task')
def save_stocks_month_data_history_task(self, batch_size: int = 10): # 限量：单次最大4500行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    try:
        # 在同步任务中运行异步代码获取列表
        favorite_codes, non_favorite_codes = asyncio.run(_get_all_relevant_stock_codes_for_processing())

        if not favorite_codes and not non_favorite_codes:
            logger.warning("未能获取到需要处理的股票代码列表，调度任务结束")
            return {"status": "warning", "message": "未获取到股票代码", "dispatched_batches": 0}

        total_dispatched_batches = 0
        total_favorite_stocks = len(favorite_codes)
        total_non_favorite_stocks = len(non_favorite_codes)

        # 1. 分派自选股批量任务
        logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
        for i in range(0, total_favorite_stocks, batch_size):
            batch = favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_month_data_history_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                logger.debug(f"已分派自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_month_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"已为 {total_non_favorite_stocks} 个非自选股分派了 {non_favorite_batches_dispatched} 个批次任务。")

        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}
















