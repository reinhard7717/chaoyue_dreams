# tasks/tushare/stock_realtime_tasks.py
import asyncio
from asgiref.sync import async_to_sync
from celery import chain
import logging
import datetime
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
# from celery import chain # 不再需要 chain，除非有后续步骤
from celery.utils.log import get_task_logger

from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from tasks.stock_analysis_tasks import analyze_batch_stocks


# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveData_RealTime'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveData_RealTime'
logger = logging.getLogger('tasks')

def is_trading_time():
    now = datetime.datetime.now()
    # 交易日判断略，假设已是交易日
    if now.hour in [9, 10, 11, 13, 14, 15]:
        if now.hour == 11 and now.minute >= 30:
            return False
        if now.hour == 9 and now.minute < 25:
            return False
        if now.hour == 15 and now.minute >= 2:
            return False
        return True
    return False

# --- 辅助函数：获取需要处理的股票代码 (保持不变) ---
async def _get_all_relevant_stock_codes_for_processing():
    """异步获取所有需要处理的股票代码列表，区分为自选股和非自选股"""
    stock_basic_dao = StockBasicInfoDao()
    favorite_stock_codes = set()
    all_stock_codes = set()
    # 获取自选股
    try:
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        if favorite_stocks is not None:
            for fav in favorite_stocks:
                code = str(fav.stock_id)
                favorite_stock_codes.add(code)
            # logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    # 获取所有A股
    try:
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            code = str(stock.stock_code)
            all_stock_codes.add(code)
        # logger.info(f"获取到 {len(all_stock_codes)} 个全市场股票代码")
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)

    # 计算非自选股代码
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)

    total_unique_stocks = len(favorite_stock_codes_list) + len(non_favorite_stock_codes)
    logger.info(f"总计需要处理的股票: {total_unique_stocks} (自选: {len(favorite_stock_codes_list)}, 非自选: {len(non_favorite_stock_codes)})")

    if not favorite_stock_codes_list and not non_favorite_stock_codes:
        logger.warning("未能获取到任何需要处理的股票代码")

    return favorite_stock_codes_list, non_favorite_stock_codes

#  ================ 实时(Tick)数据任务 ================
# --- 新的批量处理工作任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime_tasks.save_tick_data_batch')
def save_tick_data_batch(self, stock_codes: List[str]):
    """
    从Tushare批量获取实时Tick交易数据并保存到数据库（异步并发处理），并推送到前台
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 实时(Tick)数据任务...")
    stock_realtime_dao = StockRealtimeDAO()
    try:
        # 1. 保存tick数据
        asyncio.run(stock_realtime_dao.save_tick_data_by_stock_codes(stock_codes))
        # logger.info("批量tick数据保存完成，准备推送到前台")
        from users.models import FavoriteStock
        from dashboard.tasks import send_update_to_user_task_celery

        for code in stock_codes:
            # 获取所有关注该股票的用户
            user_ids = list(FavoriteStock.objects.filter(stock__stock_code=code).values_list('user_id', flat=True))
            if not user_ids:
                # logger.info(f"股票{code}没有关注用户，跳过推送")
                continue
            # 获取最新tick数据（调用异步方法，转同步）
            latest_tick = async_to_sync(stock_realtime_dao.get_latest_tick_data)(code)
            if not latest_tick:
                logger.warning(f"未获取到股票{code}的最新tick数据，跳过推送")
                continue
            # --- 保证signal字段为对象 ---
            signal = latest_tick.get('signal')
            if not isinstance(signal, dict):
                signal = {'type': 'hold', 'text': signal or 'N/A'}
            # --- 构造payload，字段名与前端updateStockRow完全一致 ---
            payload = {
                'code': code,
                'current_price': latest_tick.get('current_price'),
                'high_price': latest_tick.get('high_price'),
                'low_price': latest_tick.get('low_price'),
                'open_price': latest_tick.get('open_price'),
                'prev_close_price': latest_tick.get('prev_close_price'),
                'trade_time': latest_tick.get('trade_time'),
                'turnover_value': latest_tick.get('turnover_value'),
                'volume': latest_tick.get('volume'),
                'change_percent': latest_tick.get("change_percent"),
                'signal': signal,
            }
            # 推送给所有关注该股票的用户
            for uid in user_ids:
                send_update_to_user_task_celery.apply_async(
                    args=[uid, 'realtime_tick_update', payload],
                    queue='dashboard'  # 指定队列为dashboard
                )
                # print(f"已推送{code}最新tick数据到用户{uid}")
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)


# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime_tasks.save_stocks_tick_data_task')
def save_stocks_tick_data_task(self, batch_size: int = 50): # sina数据最多每次50个
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    if not is_trading_time():
        return
    logger.info(f"任务启动: save_stocks_tick_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
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
        # logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
        for i in range(0, total_favorite_stocks, batch_size):
            batch = favorite_codes[i:i + batch_size]
            if batch:
                # 使用新的批量任务，并指定队列
                save_tick_data_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1

        # logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
        favorite_batches_dispatched = total_dispatched_batches

        # 2. 分派非自选股批量任务
        # logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                # logger.info(f"创建非自选股批次任务 (大小: {len(batch)})...")
                # 使用新的批量任务，并指定队列
                save_tick_data_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")

        logger.info(f"任务结束: save_stocks_tick_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}

    except Exception as e:
        logger.error(f"执行 save_stocks_tick_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

#  ================ 实时(分钟)数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime_tasks.save_minute_data_realtime_batch', queue='SaveData_TimeTrade')
def save_minute_data_realtime_batch(self, stock_codes: List[str], time_level: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 实时(分钟)数据任务...")
    # 在任务开始时创建一次 DAO 实例
    stock_time_trade_dao = StockTimeTradeDAO()
    try:
        asyncio.run(stock_time_trade_dao.save_minute_time_trade_realtime_by_stock_codes_and_time_level(stock_codes, time_level))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', queue='celery')
def save_stocks_minute_data_realtime_task(self, batch_size: int = 300, time_level: str = '5', params_file: str = "default_params.json", day_count: int = 5):
    """
    调度器任务：保存分钟数据后自动分析
    """
    if not is_trading_time():
        return
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size}, 时间级别: {time_level})")
    try:
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
                # 链式：先保存分钟数据，再分析
                task_chain = chain(
                    save_minute_data_realtime_batch.s(batch, time_level),
                    analyze_batch_stocks.s(params_file, day_count)
                )
                task_chain.apply_async()
                total_dispatched_batches += 1
        favorite_batches_dispatched = total_dispatched_batches
        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                task_chain = chain(
                    save_minute_data_realtime_batch.s(batch, time_level),
                    analyze_batch_stocks.s(params_file, day_count)
                )
                task_chain.apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")
        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


