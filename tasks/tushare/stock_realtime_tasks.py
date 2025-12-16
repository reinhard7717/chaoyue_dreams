# tasks/tushare/stock_realtime_tasks.py
# tasks/tushare/stock_realtime_tasks.py
# tasks/tushare/stock_realtime_tasks.py
import asyncio
from collections import defaultdict
from asgiref.sync import async_to_sync
from asgiref.sync import sync_to_async # 异步转换工具
from celery import chain
import logging
import datetime
from datetime import timedelta # 引入timedelta用于日期迭代
from typing import List, Dict, Any # 引入 List, Dict, Any
import pandas as pd # 引入pandas库，用于高效处理数据
from django.utils import timezone # 引入Django的时区工具，用于处理时间
from chaoyue_dreams.celery import app as celery_app
# from celery import chain # 不再需要 chain # 不再需要 chain，除非有后续步骤
from dashboard.tasks import send_update_to_user_task_celery
from utils.task_helpers import with_cache_manager
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.realtime_data_dao import StockRealtimeDAO
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from users.models import FavoriteStock
from utils.cache_manager import CacheManager
from stock_models.stock_basic import StockInfo
# 修改代码行: 引入交易日历模型，并移除不再使用的分钟线模型辅助函数
from stock_models.index import TradeCalendar
from utils.model_helpers import get_stock_tick_data_model_by_code

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveData_RealTime'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveData_RealTime'
logger = logging.getLogger('tasks')

def is_trading_time():
    """
    【修改】检查当前时间是否为A股的交易时间。
    该函数现在会同时检查日期和时间。
    """
    now = datetime.datetime.now()
    # 新增代码行: 首先检查当天是否为交易日
    if not TradeCalendar.is_trade_date(check_date=now.date()):
        return False # 如果不是交易日，则直接返回False
    # 如果是交易日，再继续判断时间是否在交易时段内
    if now.hour in [9, 10, 11, 13, 14, 15]:
        if now.hour == 11 and now.minute >= 31:
            return False
        if now.hour == 9 and now.minute < 25:
            return False
        if now.hour == 15 and now.minute >= 2:
            return False
        return True
    return False

# --- 辅助函数：获取需要处理的股票代码 ---
async def _get_all_relevant_stock_codes_for_processing(stock_basic_dao: StockBasicInfoDao):
    """
    【V2.0 依赖注入版】
    异步获取所有需要处理的股票代码列表。
    - 不再自己创建DAO，而是接收一个外部传入的DAO实例。
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

@celery_app.task(queue="SaveData_RealTime_Quote")
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
    # logger.info(f"开始处理 {len(stock_codes)} 个股票的行情快照(Quote)数据任务...")
    stock_realtime_dao = StockRealtimeDAO(cache_manager)
    # strategy_dao = StrategiesDAO(cache_manager)
    async def main():
        # 1. 批量保存行情快照数据
        await stock_realtime_dao.save_quote_data_by_stock_codes(stock_codes)
        # 2. 执行用户推送逻辑 (这部分逻辑保持不变)
        # @sync_to_async(thread_sensitive=True)
        # def get_user_ids_for_codes(codes: List[str]) -> Dict[str, List[int]]:
        #     favorites = FavoriteStock.objects.filter(stock__stock_code__in=codes).values('stock__stock_code', 'user_id')
        #     user_map = defaultdict(list)
        #     for fav in favorites:
        #         user_map[fav['stock__stock_code']].append(fav['user_id'])
        #     return user_map
        # user_ids_map = await get_user_ids_for_codes(stock_codes)
        # push_tasks = []
        # for code in stock_codes:
        #     user_ids = user_ids_map.get(code)
        #     if not user_ids: continue
        #     latest_tick, latest_strategy_result = await asyncio.gather(
        #         stock_realtime_dao.get_latest_tick_data(code),
        #         strategy_dao.get_latest_strategy_result(code)
        #     )
        #     if not latest_tick: continue
        #     signal_score = getattr(latest_strategy_result, 'score', None)
        #     signal = signal_score if isinstance(signal_score, dict) else {'type': 'hold', 'text': signal_score or 'N/A'}
        #     payload = {
        #         'code': code,
        #         'current_price': latest_tick.get('current_price'),
        #         'high_price': latest_tick.get('high_price'),
        #         'low_price': latest_tick.get('low_price'),
        #         'open_price': latest_tick.get('open_price'),
        #         'prev_close_price': latest_tick.get('prev_close_price'),
        #         'trade_time': latest_tick.get('trade_time'),
        #         'turnover_value': latest_tick.get('turnover_value'),
        #         'volume': latest_tick.get('volume'),
        #         'change_percent': latest_tick.get("change_percent"),
        #         'signal': signal,
        #     }
        #     for uid in user_ids:
        #         send_update_to_user_task_celery.apply_async(
        #             args=[uid, 'realtime_tick_update', payload],
        #             queue='dashboard'
        #         )
    async_to_sync(main)()

# =================================================================
# =================== 2. 真实逐笔 (Tick) 数据任务 ===================
# =================================================================
@celery_app.task(
    queue="SaveData_RealTime",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 5, 'countdown': 15}
)
@with_cache_manager
def save_real_tick_data_single(stock_code: str, cache_manager=None):
    """
    获取并保存单只股票的当日全部真实逐笔数据 (realtime_tick)。
    这是一个高频、IO密集型任务。
    """
    if not stock_code:
        return
    logger.info(f"开始处理 {stock_code} 的真实逐笔(Tick)数据任务...")
    stock_realtime_dao = StockRealtimeDAO(cache_manager)
    trade_date = datetime.datetime.now().strftime('%Y-%m-%d')
    try:
        async def main():
            print(f"开始处理 {stock_code} 的真实逐笔(Tick)数据任务...")
            # 接收DAO返回的 success 和 message
            success, message = await stock_realtime_dao.save_realtime_tick_in_bulk([stock_code], trade_date)
            if not success:
                # 修改代码开始: 增加对“无数据”情况的判断，避免不必要的重试
                # 检查是否是因为没有获取到数据，这种情况（例如，股票暂时无成交）不应视为需要重试的严重错误
                if "未能获取到任何股票的实时逐笔数据" in message:
                    logger.warning(f"股票 {stock_code} 未获取到实时逐笔数据，可能当前无成交。任务正常结束。消息: {message}")
                    # 不再抛出异常，让任务成功结束
                else:
                    # 对于其他类型的失败，仍然抛出异常以触发Celery重试
                    print(f"股票 {stock_code} 的真实逐笔数据保存失败: {message}。触发 Celery 重试。")
                    raise Exception(f"股票 {stock_code} 的真实逐笔数据保存失败: {message}")
                # 修改代码结束
        async_to_sync(main)()
    except Exception as e:
        logger.error(f"处理 {stock_code} 的真实逐笔(Tick)数据任务时发生未预期异常: {e}", exc_info=False)
        raise e

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
    # if not is_trading_time():
    #     return
    # 1. 获取需要处理的股票列表
    stock_codes = list(StockInfo.objects.filter(list_status='L').exclude(stock_code__endswith='.BJ').values_list('stock_code', flat=True))
    # 2. 分派“行情快照(Quote)”批量任务 
    logger.info("--- 开始分派行情快照(Quote)任务 ---")
    total_quote_batches = 0
    for i in range(0, len(stock_codes), quote_batch_size):
        batch = stock_codes[i:i + quote_batch_size]
        if batch:
            save_quote_data_batch.s(batch).set(queue="SaveData_RealTime_Quote").apply_async()
            total_quote_batches += 1
    logger.info(f"--- 行情快照任务分派完成，共 {total_quote_batches} 个批次。 ---")
    # 3. 分派“真实逐笔(Tick)”单票任务
    logger.info("--- 开始分派真实逐笔(Tick)任务 ---")
    dispatched_count = 0 # 初始化已分派任务计数器
    failed_dispatch_count = 0 # 初始化分派失败计数器
    for stock_code in stock_codes:
        try: # 添加 try-except 块捕获 apply_async 异常
            # 为每一只股票分派一个独立的任务
            save_real_tick_data_single.s(stock_code).set(queue="SaveData_RealTime_Tick").apply_async()
            dispatched_count += 1 # 成功分派则计数
        except Exception as e: # 捕获分派异常
            failed_dispatch_count += 1 # 失败分派则计数
            logger.error(f"分派 {stock_code} 的真实逐笔(Tick)数据任务失败: {e}", exc_info=True) # 记录分派失败日志
    logger.info(f"--- 真实逐笔任务分派完成，共 {dispatched_count} 个任务成功分派，{failed_dispatch_count} 个任务分派失败。 ---") # 打印详细分派结果
    return {
        "status": "success",
        "dispatched_quote_batches": 0,
        "dispatched_real_tick_tasks": dispatched_count # 返回成功分派的任务数量
    }

# 单独：“行情快照(Quote)”数据获取任务。
@celery_app.task(name='tasks.tushare.stock_realtime_tasks.dispatch_stocks_quote_data_task', queue='celery') # 新的行情快照调度器
@with_cache_manager
def dispatch_stocks_quote_data_task(quote_batch_size: int = 50, cache_manager=None):
    """
    【新增-调度器】
    此任务由 Celery Beat 调度，统一分发“行情快照(Quote)”数据获取任务。
    """
    if not is_trading_time():
        return
    # 1. 获取需要处理的股票列表
    stock_codes = list(StockInfo.objects.filter(list_status='L').exclude(stock_code__endswith='.BJ').values_list('stock_code', flat=True))
    # 2. 分派“行情快照(Quote)”批量任务
    logger.info("--- 开始分派行情快照(Quote)任务 ---")
    total_quote_batches = 0
    for i in range(0, len(stock_codes), quote_batch_size):
        batch = stock_codes[i:i + quote_batch_size]
        if batch:
            save_quote_data_batch.s(batch).set(queue="SaveData_RealTime_Quote").apply_async()
            total_quote_batches += 1
    logger.info(f"--- 行情快照任务分派完成，共 {total_quote_batches} 个批次。 ---")
    return {
        "status": "success",
        "dispatched_quote_batches": total_quote_batches,
    }

# 单独：真实逐笔(Tick)”数据获取任务。
@celery_app.task(name='tasks.tushare.stock_realtime_tasks.dispatch_stocks_real_tick_task', queue='celery') # 重命名原任务
@with_cache_manager
def dispatch_stocks_real_tick_task(cache_manager=None): # 移除不再需要的 quote_batch_size 参数
    """
    【修改-调度器】
    此任务由 Celery Beat 调度，统一分发“真实逐笔(Tick)”数据获取任务。
    """
    # if not is_trading_time():
    #     return
    # 1. 获取需要处理的股票列表
    stock_codes = list(StockInfo.objects.filter(list_status='L').exclude(stock_code__endswith='.BJ').values_list('stock_code', flat=True))
    # 2. 分派“真实逐笔(Tick)”单票任务
    logger.info("--- 开始分派真实逐笔(Tick)任务 ---")
    dispatched_count = 0
    failed_dispatch_count = 0
    for stock_code in stock_codes:
        try:
            save_real_tick_data_single.s(stock_code).set(queue="SaveData_RealTime_Tick").apply_async()
            dispatched_count += 1
        except Exception as e:
            failed_dispatch_count += 1
            logger.error(f"分派 {stock_code} 的真实逐笔(Tick)数据任务失败: {e}", exc_info=True)
    logger.info(f"--- 真实逐笔任务分派完成，共 {dispatched_count} 个任务成功分派，{failed_dispatch_count} 个任务分派失败。 ---")
    return {
        "status": "success",
        "dispatched_real_tick_tasks": dispatched_count
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


#  ================ 清理错误的tick数据任务 ================
@celery_app.task(queue='SaveData_RealTime_Quote')
def clean_tick_data_for_stock(stock_code: str, trade_date_str: str):
    """
    【重构】清理Tick数据。
    1. 删除所有非交易日的Tick数据。
    2. 通过与上一交易日对比，清理重复的Tick数据。
    Args:
        stock_code (str): 股票代码。
        trade_date_str (str): 交易日期字符串，格式 'YYYY-MM-DD'。
    """
    TickModel = get_stock_tick_data_model_by_code(stock_code)
    if not TickModel:
        logger.error(f"[{stock_code}] 无法找到对应的Tick模型，清洗任务终止。")
        return
    try:
        local_tz = timezone.get_default_timezone()
        current_date = datetime.datetime.strptime(trade_date_str, '%Y-%m-%d').date()
        if not TradeCalendar.is_trade_date(check_date=current_date):
            start_dt = datetime.datetime.combine(current_date, datetime.time.min, tzinfo=local_tz)
            end_dt = datetime.datetime.combine(current_date, datetime.time.max, tzinfo=local_tz)
            deleted_count, _ = TickModel.objects.filter(
                stock_id=stock_code,
                trade_time__range=(start_dt, end_dt)
            ).delete()
            if deleted_count > 0:
                logger.info(f"[{stock_code}] {trade_date_str} 非交易日，清理了 {deleted_count} 条异常Tick数据。")
            return
        prev_trade_date = TradeCalendar.get_latest_trade_date(reference_date=current_date)
        if not prev_trade_date:
            logger.info(f"[{stock_code}] 在 {trade_date_str} 找不到上一个交易日，无法进行数据比对。")
            return
        def get_ticks_for_date(trade_date):
            start_dt = datetime.datetime.combine(trade_date, datetime.time.min, tzinfo=local_tz)
            end_dt = datetime.datetime.combine(trade_date, datetime.time.max, tzinfo=local_tz)
            qs = TickModel.objects.filter(
                stock_id=stock_code,
                trade_time__range=(start_dt, end_dt)
            ).values('id', 'trade_time', 'price', 'volume', 'amount', 'type')
            return pd.DataFrame.from_records(qs)
        current_day_df = get_ticks_for_date(current_date)
        prev_day_df = get_ticks_for_date(prev_trade_date)
        if current_day_df.empty or prev_day_df.empty:
            logger.info(f"[{stock_code}] 在 {trade_date_str} 或其上一交易日数据为空，无法清理。")
            return
        for df in [current_day_df, prev_day_df]:
            df['trade_time'] = df['trade_time'].dt.tz_localize(None).dt.tz_localize(local_tz)
            df['time_of_day'] = df['trade_time'].dt.time
            df['price'] = df['price'].astype(float)
            df['volume'] = df['volume'].astype(int)
            df['amount'] = df['amount'].astype(float)
        merge_columns = ['time_of_day', 'price', 'volume', 'amount', 'type']
        merged_df = pd.merge(
            current_day_df,
            prev_day_df[merge_columns],
            on=merge_columns,
            how='inner'
        )
        ids_to_delete = merged_df['id'].tolist()
        if ids_to_delete:
            deleted_count, _ = TickModel.objects.filter(id__in=ids_to_delete).delete()
            logger.info(f"[{stock_code}] 在 {trade_date_str} 清理了 {deleted_count} 条与上一交易日重复的Tick数据。")
    except Exception as e:
        logger.error(f"[{stock_code}] 在 {trade_date_str} 清洗Tick数据时发生错误: {e}", exc_info=True)

@celery_app.task(name='tasks.tushare.stock_realtime_tasks.dispatch_tick_data_cleaning_task', queue='celery')
def dispatch_tick_data_cleaning_task(start_date_str: str, end_date_str: str = None):
    """
    【修改-调度器】
    分发清理指定日期范围内Tick数据的任务到任务队列。
    Args:
        start_date_str (str): 需要清理的起始交易日期 'YYYY-MM-DD'。
        end_date_str (str, optional): 需要清理的结束交易日期 'YYYY-MM-DD'。
                                      如果为None，则默认为当天。
    """
    try:
        start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
        if end_date_str:
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
        else:
            end_date = datetime.datetime.now(timezone.get_default_timezone()).date()
        dates_to_process = []
        current_date = start_date
        while current_date <= end_date:
            dates_to_process.append(current_date)
            current_date += timedelta(days=1)
    except ValueError:
        logger.error(f"日期格式错误。应为 'YYYY-MM-DD'。收到的 start_date: '{start_date_str}', end_date: '{end_date_str}'")
        return {"status": "error", "message": "日期格式错误"}
    logger.info(f"--- 开始为日期范围 {start_date_str} 至 {end_date.strftime('%Y-%m-%d')} 分派Tick数据清理任务 ---")
    stock_codes = list(StockInfo.objects.filter(list_status='L').values_list('stock_code', flat=True))
    total_dispatched_count = 0
    for process_date in dates_to_process:
        date_str = process_date.strftime('%Y-%m-%d')
        for stock_code in stock_codes:
            clean_tick_data_for_stock.s(stock_code, date_str).set(queue="SaveData_RealTime_Quote").apply_async()
            total_dispatched_count += 1
    logger.info(f"--- Tick数据清理任务分派完成，共为 {len(dates_to_process)} 天、{len(stock_codes)} 只股票分派了 {total_dispatched_count} 个任务。 ---")
    return {
        "status": "success",
        "date_range": f"{start_date_str} to {end_date.strftime('%Y-%m-%d')}",
        "dispatched_tasks": total_dispatched_count
    }



