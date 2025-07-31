# tasks/tushare/stock_realtime_tasks.py
import asyncio
from collections import defaultdict
from asgiref.sync import async_to_sync
from asgiref.sync import sync_to_async # 异步转换工具
from celery import chain
import logging
import datetime
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
# from celery import chain # 不再需要 chain，除非有后续步骤
from dashboard.tasks import send_update_to_user_task_celery
from utils.task_helpers import with_cache_manager
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from users.models import FavoriteStock
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

# =================================================================
# =================== 1. 行情快照 (Quote) 数据任务 ==================
# =================================================================

@celery_app.task(queue="SaveData_RealTime")
@with_cache_manager
def save_quote_data_batch(stock_codes: List[str], cache_manager=None):
    """
    【V3.0 - 职责明确版】
    获取并保存行情快照(realtime_quote)，并负责后续的用户推送。
    此任务处理的是可以批量获取的数据。
    """
    if not stock_codes:
        logger.info("行情快照任务收到空列表，任务结束。")
        return
    logger.info(f"开始处理 {len(stock_codes)} 个股票的行情快照(Quote)数据任务...")
    
    stock_realtime_dao = StockRealtimeDAO(cache_manager)
    strategy_dao = StrategiesDAO(cache_manager)

    async def main():
        # 1. 批量保存行情快照数据
        await stock_realtime_dao.save_quote_data_by_stock_codes(stock_codes)
        
        # 2. 执行用户推送逻辑 (这部分逻辑保持不变)
        @sync_to_async(thread_sensitive=True)
        def get_user_ids_for_codes(codes: List[str]) -> Dict[str, List[int]]:
            favorites = FavoriteStock.objects.filter(stock__stock_code__in=codes).values('stock__stock_code', 'user_id')
            user_map = defaultdict(list)
            for fav in favorites:
                user_map[fav['stock__stock_code']].append(fav['user_id'])
            return user_map
        
        user_ids_map = await get_user_ids_for_codes(stock_codes)
        push_tasks = []
        for code in stock_codes:
            user_ids = user_ids_map.get(code)
            if not user_ids: continue
            
            latest_tick, latest_strategy_result = await asyncio.gather(
                stock_realtime_dao.get_latest_tick_data(code),
                strategy_dao.get_latest_strategy_result(code)
            )
            
            if not latest_tick: continue
            
            signal_score = getattr(latest_strategy_result, 'score', None)
            signal = signal_score if isinstance(signal_score, dict) else {'type': 'hold', 'text': signal_score or 'N/A'}
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
    async_to_sync(main)()

# =================================================================
# =================== 2. 真实逐笔 (Tick) 数据任务 ===================
# =================================================================

# ▼▼▼ 新增: 处理单只股票真实逐笔数据的工作任务 ▼▼▼
@celery_app.task(queue="SaveData_RealTime")
@with_cache_manager
def save_real_tick_data_single(stock_code: str, cache_manager=None):
    """
    【新增】获取并保存单只股票的当日全部真实逐笔数据 (realtime_tick)。
    这是一个高频、IO密集型任务。
    """
    if not stock_code:
        return
    
    logger.info(f"开始处理 {stock_code} 的真实逐笔(Tick)数据任务...")
    stock_realtime_dao = StockRealtimeDAO(cache_manager)
    trade_date = datetime.datetime.now().strftime('%Y-%m-%d')
    
    async def main():
        # 调用我们之前在DAO中创建的、包含完整持久化逻辑的方法
        await stock_realtime_dao.save_realtime_tick_in_bulk([stock_code], trade_date)
        
    async_to_sync(main)()
# ▲▲▲ 新增结束 ▲▲▲

# =================================================================
# =================== 3. 统一调度器任务 ============================
# =================================================================

@celery_app.task(name='tasks.tushare.stock_realtime_tasks.save_stocks_tick_data_task', queue='celery')
@with_cache_manager
def save_stocks_tick_data_task(quote_batch_size: int = 50, cache_manager=None):
    """
    【V3.0 - 统一调度版】
    此任务由 Celery Beat 调度，统一分发“行情快照”和“真实逐笔”两种数据获取任务。
    """
    if not is_trading_time():
        return
    
    logger.info(f"任务启动: 统一调度器 save_stocks_tick_data_task 启动...")
    
    # 1. 获取需要处理的股票列表
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    favorite_codes, non_favorite_codes = async_to_sync(
        _get_all_relevant_stock_codes_for_processing
    )(stock_basic_dao)
    
    if not favorite_codes and not non_favorite_codes:
        logger.warning("未能获取到股票列表，统一调度任务结束。")
        return
        
    # 2. 分派“行情快照(Quote)”批量任务 (逻辑不变)
    logger.info("--- 开始分派行情快照(Quote)任务 ---")
    total_quote_batches = 0
    for i in range(0, len(favorite_codes), quote_batch_size):
        batch = favorite_codes[i:i + quote_batch_size]
        if batch:
            save_quote_data_batch.s(batch).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
            total_quote_batches += 1
            
    for i in range(0, len(non_favorite_codes), quote_batch_size):
        batch = non_favorite_codes[i:i + quote_batch_size]
        if batch:
            save_quote_data_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
            total_quote_batches += 1
    logger.info(f"--- 行情快照任务分派完成，共 {total_quote_batches} 个批次。 ---")

    # 3. 分派“真实逐笔(Tick)”单票任务
    logger.info("--- 开始分派真实逐笔(Tick)任务 ---")
    all_codes = favorite_codes + non_favorite_codes
    for stock_code in all_codes:
        # 为每一只股票分派一个独立的任务
        save_real_tick_data_single.s(stock_code).set(queue="SaveData_RealTime").apply_async()
    logger.info(f"--- 真实逐笔任务分派完成，共 {len(all_codes)} 个任务。 ---")

    return {
        "status": "success",
        "dispatched_quote_batches": total_quote_batches,
        "dispatched_real_tick_tasks": len(all_codes)
    }

#  ================ 实时(分钟)数据任务 ================
@celery_app.task(queue='SaveData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_minute_data_realtime_batch(stock_codes: List[str], time_level: str, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 实时(分钟) ({time_level}) 数据任务...")
    async def main():
        # 执行业务逻辑
        await stock_time_trade_dao.save_minute_time_trade_realtime_by_stock_codes_and_time_level(stock_codes, time_level)
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(name='tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_realtime_task(batch_size: int = 300, time_level: str = '5', cache_manager=None):
    """
    【无绑定版】
    调度器任务：保存分钟数据后自动分析
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size}, 时间级别: {time_level})")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    favorite_codes = []
    non_favorite_codes = []
    async def main():
        nonlocal favorite_codes, non_favorite_codes
        fav_codes, non_fav_codes = await _get_all_relevant_stock_codes_for_processing(stock_basic_dao)
        favorite_codes.extend(fav_codes)
        non_favorite_codes.extend(non_fav_codes)
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
    # 2. 分派非自选股批量任务
    logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
    for i in range(0, total_non_favorite_stocks, batch_size):
        batch = non_favorite_codes[i:i + batch_size]
        if batch:
            save_minute_data_realtime_batch.s(batch, time_level).apply_async()
            total_dispatched_batches += 1
            logger.debug(f"已分派非自选股批次任务 (索引 {i} 到 {i+len(batch)-1})")
    logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

