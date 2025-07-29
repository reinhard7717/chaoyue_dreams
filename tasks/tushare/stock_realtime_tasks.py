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
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from utils.cache_manager import CacheManager

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
async def _get_all_relevant_stock_codes_for_processing(stock_basic_dao: StockBasicInfoDao):
    """
    【V2.0 依赖注入版】
    异步获取所有需要处理的股票代码列表。
    - 核心修改: 不再自己创建DAO，而是接收一个外部传入的DAO实例。
    """
    favorite_stock_codes = set()
    all_stock_codes = set()
    
    try:
        # 直接使用传入的DAO实例
        favorite_stocks = await stock_basic_dao.get_all_favorite_stocks()
        if favorite_stocks:
            for fav in favorite_stocks:
                if fav and fav.get("stock_code"):
                    favorite_stock_codes.add(fav.get("stock_code"))
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
        
    try:
        # 直接使用传入的DAO实例
        all_stocks = await stock_basic_dao.get_stock_list()
        if all_stocks:
            for stock in all_stocks:
                if stock and not stock.stock_code.endswith('.BJ'):
                    all_stock_codes.add(stock.stock_code)
    except Exception as e:
        logger.error(f"获取全市场股票列表时出错: {e}", exc_info=True)
        
    non_favorite_stock_codes = list(all_stock_codes - favorite_stock_codes)
    favorite_stock_codes_list = list(favorite_stock_codes)
    
    # 返回排序后的列表，保证每次结果一致
    return sorted(favorite_stock_codes_list), sorted(non_favorite_stock_codes)

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

    async def main():
        # 创建CacheManager实例
        cache_manager = CacheManager()
        # 创建DAO实例并注入cache_manager
        stock_realtime_dao = StockRealtimeDAO(cache_manager)
        strategy_dao = StrategiesDAO(cache_manager)
        # 1. 保存tick数据
        await stock_realtime_dao.save_tick_data_by_stock_codes(stock_codes)
        from users.models import FavoriteStock
        from dashboard.tasks import send_update_to_user_task_celery
        for code in stock_codes:
            user_ids = list(FavoriteStock.objects.filter(stock__stock_code=code).values_list('user_id', flat=True))
            if not user_ids:
                continue
            latest_tick = await stock_realtime_dao.get_latest_tick_data(code)
            latest_strategy_result = await strategy_dao.get_latest_strategy_result(code)
            if not latest_tick:
                logger.warning(f"未获取到股票{code}的最新tick数据，跳过推送")
                continue
            signal = latest_strategy_result.score
            if not isinstance(signal, dict):
                signal = {'type': 'hold', 'text': signal or 'N/A'}
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
            for uid in user_ids:
                send_update_to_user_task_celery.apply_async(
                    args=[uid, 'realtime_tick_update', payload],
                    queue='dashboard'
                )
    try:
        async_to_sync(main)()
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
        # 初始化用于接收结果的列表
        favorite_codes = []
        non_favorite_codes = []

        # 1. 定义一个异步 main 函数，用于安全地执行所有需要异步环境的操作
        async def main():
            # nonlocal 关键字允许内部函数修改外部函数的变量
            nonlocal favorite_codes, non_favorite_codes
            
            # 在异步上下文中创建 CacheManager 和 DAO
            cache_manager_instance = CacheManager()
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
            
            # 调用改造后的辅助函数，并将DAO实例作为参数传递进去
            fav_codes, non_fav_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
            
            # 将获取到的结果赋值给外部变量
            favorite_codes.extend(fav_codes)
            non_favorite_codes.extend(non_fav_codes)

        # 2. 在同步代码中，安全地执行异步的 main 函数来准备数据
        async_to_sync(main)()

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
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime_tasks.save_minute_data_realtime_batch', queue='SaveData_TimeTrade', rate_limit='180/m')
def save_minute_data_realtime_batch(self, stock_codes: List[str], time_level: str):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 实时(分钟) ({time_level}) 数据任务...")

    async def main():
        # 创建CacheManager实例
        cache_manager = CacheManager()
        # 创建DAO实例并注入cache_manager
        stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
        # 执行业务逻辑
        await stock_time_trade_dao.save_minute_time_trade_realtime_by_stock_codes_and_time_level(stock_codes, time_level)

    try:
        async_to_sync(main)()
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)
    return stock_codes


# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', queue='celery')
def save_stocks_minute_data_realtime_task(self, batch_size: int = 300, time_level: str = '5', params_file: str = "config/indicator_parameters.json"):
    """
    调度器任务：保存分钟数据后自动分析
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size}, 时间级别: {time_level})")
    try:
        # 初始化用于接收结果的列表
        favorite_codes = []
        non_favorite_codes = []

        # 1. 定义一个异步 main 函数，用于安全地执行所有需要异步环境的操作
        async def main():
            # nonlocal 关键字允许内部函数修改外部函数的变量
            nonlocal favorite_codes, non_favorite_codes
            
            # 在异步上下文中创建 CacheManager 和 DAO
            cache_manager_instance = CacheManager()
            stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
            
            # 调用改造后的辅助函数，并将DAO实例作为参数传递进去
            fav_codes, non_fav_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
            
            # 将获取到的结果赋值给外部变量
            favorite_codes.extend(fav_codes)
            non_favorite_codes.extend(non_fav_codes)

        # 2. 在同步代码中，安全地执行异步的 main 函数来准备数据
        async_to_sync(main)()
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
                save_minute_data_realtime_batch.s(batch, time_level).apply_async()
                total_dispatched_batches += 1
        favorite_batches_dispatched = total_dispatched_batches
        # 2. 分派非自选股批量任务
        logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
        non_favorite_batches_dispatched = 0
        for i in range(0, total_non_favorite_stocks, batch_size):
            batch = non_favorite_codes[i:i + batch_size]
            if batch:
                save_minute_data_realtime_batch.s(batch, time_level).apply_async()
                total_dispatched_batches += 1
                non_favorite_batches_dispatched += 1
                logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")
        logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_stocks_realtime_min_data_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}


