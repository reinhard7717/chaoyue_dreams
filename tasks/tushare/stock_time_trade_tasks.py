# tasks\stock_time_trade_tasks.py
import asyncio
import logging
import datetime
from asgiref.sync import async_to_sync
from celery import chord, group
from django.db import models
from django.utils import timezone
import time
from typing import List
from django.db.models.functions import TruncDate
from utils.task_helpers import with_cache_manager
from chaoyue_dreams.celery import app as celery_app  # 从 celery.py 导入 app 实例并重命名为 celery_app
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from services.indicator_services import IndicatorService
from stock_models.index import TradeCalendar
from stock_models.time_trade import StockCyqChipsBJ, StockCyqChipsCY, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsSZ, StockMinuteData_15_SZ, StockMinuteData_30_SZ, StockMinuteData_5_SZ, StockMinuteData_60_SZ, StockMinuteData_5_SH, StockMinuteData_15_SH, StockMinuteData_30_SH, StockMinuteData_60_SH, StockMinuteData_5_BJ, StockMinuteData_15_BJ, StockMinuteData_30_BJ, StockMinuteData_60_BJ,StockMinuteData_5_CY, StockMinuteData_15_CY, StockMinuteData_30_CY, StockMinuteData_60_CY, StockMinuteData_5_KC, StockMinuteData_15_KC, StockMinuteData_30_KC, StockMinuteData_60_KC, StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, StockDailyData_KC, StockDailyData_BJ
from utils.cache_manager import CacheManager

# 自选股队列
FAVORITE_SAVE_API_DATA_QUEUE = 'favorite_SaveHistoryData_TimeTrade'
STOCKS_SAVE_API_DATA_QUEUE = 'SaveHistoryData_TimeTrade'
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
    today_date = timezone.now().date()
    this_monday = today_date - datetime.timedelta(days=today_date.weekday())
    this_friday = this_monday + datetime.timedelta(days=4)
    return this_monday, this_friday

# 获取上周一和上周五的日期
def get_last_monday_and_friday():
    """获取上周一和上周五的日期"""
    today_date = timezone.now().date()
    this_monday = today_date - datetime.timedelta(days=today_date.weekday())
    last_monday = this_monday - datetime.timedelta(days=7)
    last_friday = last_monday + datetime.timedelta(days=4)
    return last_monday, last_friday

# --- 异步辅助函数：获取需要处理的股票代码 (区分自选和非自选) ---
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

# ===================================================
#                      当日任务
# ===================================================

#  ================ 分钟数据任务（当日收盘后） ================
@celery_app.task(queue='SaveData_TimeTrade')
@with_cache_manager
def save_stocks_minute_data_today_batch(stock_codes, trade_time_str=None, cache_manager=None):
    # 1. 直接使用由装饰器注入的 cache_manager 实例
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    if trade_time_str:
        start_date_str = trade_time_str
        end_date_str = trade_time_str
        print(f"开始保存 分钟数据任务（当日特定时间: {trade_time_str}）...")
    else:
        start_date_str = f"{today_date} 00:00:00"
        end_date_str = f"{today_date} 23:59:59"
        print(f"开始保存 分钟数据任务（当日全天）...")
    async def main():
        # 2. 执行业务逻辑
        await stock_time_trade_dao.save_minute_time_trade_history_by_stock_codes(
            stock_codes=stock_codes,
            start_date_str=start_date_str,
            end_date_str=end_date_str
        )
    async_to_sync(main)()
    print(f"保存股票 {len(stock_codes)} 个的当日分钟级交易数据完成。")

# --- 修改后的调度器任务 ---
@celery_app.task(queue='SaveData_TimeTrade')
def on_all_minute_data_saved(results, trade_time_str=None):
    logger.info(f"所有分钟数据批量任务已全部完成。")
    # 这里可以做收尾工作，比如汇总、通知等
    return {"status": "success", "batches": len(results)}

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_today_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_today_task(self, trade_time_str=None, batch_size: int = 310, cache_manager=None):
    # 【代码修改】将 cache_manager_instance 移到 try...finally 外部作用域
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    logger.info(f"任务启动: save_stocks_minute_data_today_task (调度器模式) - 批次大小: {batch_size}")
    all_stocks = None
    if not TradeCalendar.is_trade_date():
        message = f"今天 ({timezone.now().date()}) 不是交易日，跳过所有分钟数据获取任务。"
        logger.info(message)
        return {"status": "skipped", "message": message}
    async def main():
        return await stock_basic_dao.get_stock_list()
    all_stocks = async_to_sync(main)()
    if not all_stocks:
        logger.warning("未能从DAO获取到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未能从DAO获取到任何股票代码"}
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    total_codes_count = len(all_stock_codes)
    logger.info(f"今天是交易日，准备为 {total_codes_count} 个股票分派批量任务...")
    batch_tasks = [
        save_stocks_minute_data_today_batch.s(stock_codes=all_stock_codes[i:i + batch_size], trade_time_str=trade_time_str)
        for i in range(0, total_codes_count, batch_size)
        if all_stock_codes[i:i + batch_size]
    ]
    if batch_tasks:
        callback = on_all_minute_data_saved.s(trade_time_str=trade_time_str)
        job = chord(batch_tasks)(callback)
        logger.info(f"已分派 {len(batch_tasks)} 个批量任务，等待全部完成后自动回调。")
        return {"status": "dispatched", "dispatched_batches": len(batch_tasks), "chord_id": job.id}
    else:
        logger.warning("没有需要分派的分钟数据批量任务。")
        return {"status": "skipped", "message": "没有需要分派的分钟数据批量任务。"}

@celery_app.task(queue='SaveData_TimeTrade')
@with_cache_manager
def save_stocks_minute_data_yesterday_batch(stock_codes: list, cache_manager=None):
    """
    从Tushare批量获取并保存上一个交易日的分钟级交易数据（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    latest_trade_date = TradeCalendar.get_latest_trade_date()
    if not latest_trade_date:
        logger.warning("未能从交易日历中获取到上一个交易日，任务终止。")
        print("调试: 未能获取到上一个交易日，任务终止。")
        return
    start_date_str = f"{latest_trade_date} 00:00:00"
    end_date_str = f"{latest_trade_date} 23:59:59"
    print(f"开始保存 {len(stock_codes)} 个股票, 交易日 {latest_trade_date} 的分钟数据任务...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_minute_time_trade_history_by_stock_codes(
            stock_codes=stock_codes,
            start_date_str=start_date_str,
            end_date_str=end_date_str
        )
    async_to_sync(main)()
    print(f"保存股票 {len(stock_codes)} 个的上一个交易日 ({latest_trade_date}) 分钟级交易数据完成。")

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_yesterday_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_yesterday_task(self, batch_size: int = 310, cache_manager=None): # 最大循环10万个，每310个一组循环一次是99510个
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_minute_data_today_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    total_dispatched_batches = 0
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def main():
        # 2. 【核心修复】使用 await 来执行异步方法，并用 return 返回结果
        return await stock_basic_dao.get_stock_list()
    all_stocks = async_to_sync(main)()
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    total_codes_count = len(all_stocks)  # 用于统计总代码数量
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
            # 使用新的批量任务，并指定队列
            save_stocks_minute_data_yesterday_batch.s(stock_codes=batch_codes).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"任务结束: save_stocks_minute_data_today_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

#  ================ 今日基本信息 数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_today_task', queue='SaveData_TimeTrade')
@with_cache_manager
def save_stocks_daily_basic_data_today_task(self, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    logger.info(f"开始处理今日股票重要的基本面指标...")
    service = IndicatorService(cache_manager)
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    print("开始保存 今日股票重要的基本面指标...")
    async def main():
        # 3. 执行业务逻辑
        await stock_time_trade_dao.save_stock_daily_basic_history_by_trade_date(trade_date=today_date)
        print(f"保存 今日股票重要的基本面指标 完成。result: {result}")
        rotation_report = async_to_sync(service.analyze_industry_rotation)(datetime.date.today(), lookback_days=10)
        print("--- 行业轮动强度报告 ---")
        print(rotation_report.head(10))
    async_to_sync(main)()

#  ================ 昨日基本信息 数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_yesterday_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_stocks_daily_basic_data_yesterday_task(self, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    logger.info(f"开始处理今日股票重要的基本面指标...")
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)
    print("开始保存 昨日股票重要的基本面指标...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_stock_daily_basic_history_by_trade_date(trade_date=yesterday)
    result = async_to_sync(main)()
    print(f"保存 昨日股票重要的基本面指标 完成。result: {result}")

#  ================ 日线数据任务（当日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_day_data_today_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_day_data_today_task(self, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    print("开始保存 日线数据任务（当日）...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_daily_time_trade_history_by_trade_dates(trade_date=today_date)
    result = async_to_sync(main)()
    print(f"保存 日线数据任务（当日） 完成。result: {result}")

#  ================ 日线数据任务（昨日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_day_data_yesterday_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_day_data_yesterday_task(self, cache_manager=None):
    """
    保存昨日日线数据，并等待筹码数据任务完成后再返回。
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)
    print("开始保存 日线数据任务（昨日）...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_daily_time_trade_history_by_trade_dates(trade_date=yesterday)
    result = async_to_sync(main)()
    print(f"保存 日线数据任务（昨日） 完成。result: {result}")

#  ================ 周线数据任务（当日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_week_data_today_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_week_data_today_task(self, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    day_str = today_date.strftime('%Y%m%d')
    print("开始保存 周线数据任务（当日）...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_weekly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 周线数据任务（当日） 完成。result: {result}")


#  ================ 周线数据任务（昨日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_week_data_yesterday_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_week_data_yesterday_task(self, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)
    day_str = yesterday.strftime('%Y%m%d')
    print("开始保存 周线数据任务（昨日）...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_weekly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 周线数据任务（昨日） 完成。result: {result}")

#  ================ 月线数据任务（当日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_month_data_today_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_month_data_today_task(self, cache_manager=None):
    """
    从Tushare批量获取月线数据（当日）并保存到数据库（异步并发处理）
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    day_str = today_date.strftime('%Y%m%d')
    print("开始保存 月线数据（当日）...")
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_weekly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 月线数据（当日） 完成。result: {result}")

#  ================ 月线数据任务（昨日） ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_month_data_yesterday_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_month_data_yesterday_task(self, cache_manager=None):
    """
    从Tushare批量获取月线数据（昨日）并保存到数据库（异步并发处理）
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    print("开始保存 月线数据（昨日）...")
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)
    day_str = yesterday.strftime('%Y%m%d')
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_weekly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 月线数据（昨日） 完成。result: {result}")
    save_cyq_data_yesterday_task.delay()
    print(f"保存 月线数据（昨日） 数据完成。")

# ============== 每日筹码分布任务（当日） ==============
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_cyq_chips_today_batch(cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    async def main():
        return await stock_time_trade_dao.save_all_cyq_chips_history(trade_date=today_date)
    result = async_to_sync(main)()
    print(f"保存 每日筹码分布 数据完成。 result: {result} ")

@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_cyq_perf_today_batch(cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    async def main():
        return await stock_time_trade_dao.save_all_cyq_perf_history(trade_date=today_date)
    result = async_to_sync(main)()
    print(f"保存 每日筹码及胜率 数据完成。 result: {result} ")

@celery_app.task(queue='SaveHistoryData_TimeTrade')
def on_cyq_two_tasks_done(results):
    logger.info("两个CYQ子任务已完成。")
    return {"status": "success", "results": results}

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_today_task', queue='celery')
def save_cyq_data_today_task(self):
    logger.info(f"任务启动: save_cyq_data_today_task (调度器模式)")
    try:
        job = chord([
            save_cyq_chips_today_batch.s(),
            save_cyq_perf_today_batch.s()
        ])(on_cyq_two_tasks_done.s())
        logger.info("已分派两个CYQ子任务，等待全部完成后自动回调。")
        return {"status": "dispatched", "chord_id": job.id}
    except Exception as e:
        logger.error(f"执行 save_cyq_data_today_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# ============== 每日筹码分布任务（昨日） ==============
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_cyq_chips_yesterday_batch(cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
    async def main():
        return await stock_time_trade_dao.save_all_cyq_chips_history(trade_date=yesterday)
    result = async_to_sync(main)()
    print(f"保存 每日筹码分布 数据完成。 result: {result} ")

@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_cyq_perf_yesterday_batch(cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        cache_manager: 缓存管理器实例
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
    async def main():
        return await stock_time_trade_dao.save_all_cyq_perf_history(trade_date=yesterday)
    result = async_to_sync(main)()
    print(f"保存 每日筹码及胜率 数据完成。 result: {result} ")

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_yesterday_task', queue='celery')
def save_cyq_data_yesterday_task(self):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_cyq_data_yesterday_task (调度器模式) - 获取股票列表并分派批量任务")
    try:
        save_cyq_chips_yesterday_batch.s().set().apply_async()
        save_cyq_perf_yesterday_batch.s().set().apply_async()
        logger.info(f"任务结束: save_cyq_data_yesterday_task (调度器模式)")
    except Exception as e:
        logger.error(f"执行 save_cyq_data_today_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

# ===================================================
#                      本周任务
# ===================================================

#  ================ （本周）分钟数据任务 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_minute_data_this_week_batch(stock_codes: str, cache_manager=None):
    """
    分钟数据任务（本周）
    从Tushare批量获取数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    this_monday, this_friday = get_this_monday_and_friday()
    this_monday_str = this_monday.strftime('%Y%m%d') + " 00:00:00"
    this_friday_str = this_friday.strftime('%Y%m%d') + " 16:00:00"
    # 2. 创建 DAO 实例，并注入 cache_manager
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        return await stock_time_trade_dao.save_minute_time_trade_history_by_stock_codes(stock_codes=stock_codes, start_date_str=this_monday_str, end_date_str=this_friday_str)
    result = async_to_sync(main)()
    # logger.info(f"保存股票 {stock_codes} 的分钟级交易数据完成. 结果: {result}")nute_data_this_week_batch.执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_this_week_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_this_week_task(self, batch_size: int = 10, cache_manager=None): # 限量：单次最大8000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_minute_data_today_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    total_dispatched_batches = 0
    async def main():
        # 2. 【核心修复】使用 await 来执行异步方法，并用 return 返回结果
        return await stock_basic_dao.get_stock_list()
    all_stocks = async_to_sync(main)()
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    total_codes_count = len(all_stocks)  # 用于统计总代码数量
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
            # 使用新的批量任务，并指定队列
            save_minute_data_this_week_batch.s(stock_codes=batch_codes).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"任务结束: save_stocks_minute_data_today_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

#  ================ （本周）日线数据任务 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_day_data_this_week_batch(cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    this_monday, this_friday = get_this_monday_and_friday()
    logger.info(f"开始处理包含 {this_monday} - {this_friday} 历史(日线)数据任务...")
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager_instance)
    async def main():
        return await stock_time_trade_dao.save_daily_time_trade_history_by_trade_dates(start_date=this_monday, end_date=this_friday)
    return_info = async_to_sync(main)()
    print(f"完成 {this_monday} - {this_friday} 的日线数据保存，{return_info}。")
    time.sleep(5)

#  ================ （本周）每日基本信息数据任务 ================
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_this_week_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_stocks_daily_basic_data_this_week_task(self, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    """
    logger.info(f"开始处理今日股票重要的基本面指标...")
    this_monday, this_friday = get_this_monday_and_friday()
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        return await stock_time_trade_dao.save_stock_daily_basic_history_by_trade_date(start_date=this_monday, end_date=this_friday)
    print(f"开始保存 本周 股票重要的基本面指标...")
    result = async_to_sync(main)()

# ============== （本周）每日筹码分布任务 ==============
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_cyq_chips_this_week_batch(ts_code: str, start_date: datetime.date, end_date: datetime.date, cache_manager=None):
    """
    从Tushare获取单只股票的每日筹码分布数据。
    【V2.0 - 异步上下文修复版】
    """
    print(f"开始处理 ts_code: {ts_code} 的每日筹码分布数据...")
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    # 使用 async_to_sync 运行这个总的 main 函数
    async def main():
        # 3. 执行业务逻辑
        stock = await stock_basic_dao.get_stock_by_code(stock_code=ts_code)
        if not stock:
            logger.warning(f"在数据库中未找到代码为 {ts_code} 的股票，跳过此任务。")
            return
        result = await stock_time_trade_dao.save_cyq_chips_for_stock(stock=stock, start_date=start_date, end_date=end_date)
        print(f"保存 {ts_code} （本周）每日筹码分布 数据完成。 result: {result} ")
    async_to_sync(main)()

@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_cyq_perf_this_week_batch(start_date: datetime.date, end_date: datetime.date, cache_manager=None):
    """
    从Tushare批量获取所有股票的每日筹码及胜率数据。
    【V2.0 - 异步上下文修复版】
    """
    print(f"开始处理 {start_date} 到 {end_date} 的每日筹码及胜率数据...")
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    # 使用 async_to_sync 运行这个总的 main 函数
    async def main():
        # 3. 执行业务逻辑
        result = await stock_time_trade_dao.save_all_cyq_perf_history(start_date=start_date, end_date=end_date)
        print(f"保存 （本周）每日筹码及胜率 数据完成。 result: {result} ")
    async_to_sync(main)()


@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_this_week_task', queue='celery')
@with_cache_manager
def save_cyq_data_this_week_task(self, cache_manager=None):
    """
    调度器任务：获取股票列表并分派筹码数据获取子任务。
    【V2.0 - 异步上下文修复版】
    """
    logger.info(f"任务启动: save_cyq_data_this_week_task (调度器模式)")
    all_stocks = []
    stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
    async def get_stocks_main():
        nonlocal all_stocks # 声明 all_stocks 是外部变量
        # 获取股票列表
        stocks_list = await stock_basic_dao.get_stock_list()
        if stocks_list:
            all_stocks = stocks_list
    async_to_sync(get_stocks_main)()

    # 3. 回到同步代码中，执行分派任务的逻辑
    try:
        if not all_stocks:
            logger.warning("未能获取到任何股票，任务终止。")
            return {"status": "skipped", "message": "未能获取到任何股票"}

        this_monday, this_friday = get_this_monday_and_friday()
        
        logger.info(f"获取到 {len(all_stocks)} 只股票，开始为每只股票分派筹码分布任务...")
        for stock in all_stocks:
            # 分派获取单只股票筹码分布的任务
            save_cyq_chips_this_week_batch.s(
                ts_code=stock.stock_code, 
                start_date=this_monday, 
                end_date=this_friday
            ).set(queue='SaveHistoryData_TimeTrade').apply_async()
            
        logger.info(f"所有股票的筹码分布任务已分派完毕。")

        # 分派获取所有股票筹码胜率的任务（这个任务本身会处理所有股票）
        logger.info(f"开始分派 （本周）每日筹码及胜率 任务...")
        save_cyq_perf_this_week_batch.s(
            start_date=this_monday, 
            end_date=this_friday
        ).set(queue='SaveHistoryData_TimeTrade').apply_async()
        
        logger.info(f"任务结束: save_cyq_data_this_week_task (调度器模式) - 任务已全部分派。")
        return {"status": "dispatched", "stock_count": len(all_stocks)}

    except Exception as e:
        logger.error(f"在 save_cyq_data_this_week_task 中分派子任务时出错: {e}", exc_info=True)
        return {"status": "error", "message": "分派子任务失败"}

# ===================================================
#                      历史任务
# ===================================================

#  ================ 分钟数据任务（历史） ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='480/m')
@with_cache_manager
def save_minute_data_history_batch(stock_code: str, time_level: str, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        stock_code: 股票代码
        time_level: 分钟级别
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        if not stock_code:
            logger.info("收到空的股票代码列表，任务结束")
            return {"processed": 0, "success": 0, "errors": 0}
        logger.info(f"开始处理股票{stock_code}的 历史({time_level}分钟)数据任务...")
        # 3. 执行业务逻辑
        result = await stock_time_trade_dao.save_minute_time_trade_history_by_stock_code_and_time_level(stock_code, time_level)
        logger.info(f"保存股票 {stock_code} 的{time_level}分钟级交易数据完成. 结果: {result}")
        # 用同步方式运行异步main
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_history_task(self, cache_manager=None): # 限量：单次最大8000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_minute_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_minute_data_history_task (调度器模式) - 获取股票列表并分派批量任务.")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def main():
        all_stocks = await stock_basic_dao.get_stock_list()
        all_stock_codes = [stock.stock_code for stock in all_stocks]
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        total_codes_count = len(all_stocks)  # 用于统计总代码数量
        logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
        for stock_code in all_stock_codes:
            logger.info(f"创建自选股任务 ({stock_code})...")
            # 使用新的批量任务，并指定队列
            for time_level in ["1", "5", "15", "30", "60"]:
                save_minute_data_history_batch.s(stock_code=stock_code, time_level=time_level).set().apply_async()
            total_dispatched_batches += 1
        logger.info(f"已为 {total_codes_count} 个股票分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_stocks_minute_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    async_to_sync(main)()

#  ================ 日线数据任务（历史） ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_day_data_history_batch(stock_codes: List[str], cache_manager=None):
    """
    从Tushare批量获取日线交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(日线)数据任务...")

    async def main():
        # 3. 执行业务逻辑
        return_info = await stock_time_trade_dao.save_daily_time_trade_history_by_stock_codes(stock_codes)
        print(f"完成 {len(stock_codes)} 个股票的日线数据保存，{return_info}。")
        time.sleep(5)  # 保持原有逻辑
        async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_day_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_day_data_history_task(self, batch_size: int = 13, cache_manager=None): # 限量：单次最大6000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_day_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_day_data_history_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    total_codes_count = len(all_stocks)  # 用于统计总代码数量
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
            # 使用新的批量任务，并指定队列
            save_day_data_history_batch.s(stock_codes=batch_codes).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"任务结束: save_stocks_day_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

#  ================ 每日基本信息数据任务（历史） ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_daily_basic_data_history_batch(stock_codes: List[str], cache_manager=None):
    """
    从Tushare批量获取每日基本信息数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理{','.join(stock_codes)} 历史(每日基本信息)数据任务...")
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_stock_daily_basic_history_by_stock_codes(stock_codes)
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_daily_basic_data_history_task(self, batch_size: int = 16, cache_manager=None):
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_daily_basic_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_daily_basic_data_history_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    total_dispatched_batches = 0
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    total_codes_count = len(all_stocks)  # 用于统计总代码数量
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
            # 使用新的批量任务，并指定队列
            save_daily_basic_data_history_batch.s(stock_codes=batch_codes).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"任务结束: save_stocks_daily_basic_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

#  ================ 历史(周线)数据任务 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='480/m')
@with_cache_manager
def save_week_data_history_batch(stock_codes: List[str], cache_manager=None):
    """
    从Tushare批量获取周线交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(周线)数据任务 批次...")

    async def main():
        stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_weekly_time_trade_by_stock_codes(stock_codes)
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_week_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_week_data_history_task(self, batch_size: int = 15, cache_manager=None): # 限量：单次最大4500行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_realtime_min_data_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    # 在同步任务中运行异步代码获取列表
    # 初始化用于接收结果的列表
    favorite_codes = []
    non_favorite_codes = []
    # 在异步上下文中创建 CacheManager 和 DAO
    stock_basic_dao = StockBasicInfoDao(cache_manager)

    # 1. 定义一个异步 main 函数，用于安全地执行所有需要异步环境的操作
    async def main():
        # nonlocal 关键字允许内部函数修改外部函数的变量
        nonlocal favorite_codes, non_favorite_codes
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
            # 使用新的批量任务，并指定队列
            save_week_data_history_batch.s(batch).set().apply_async()
            total_dispatched_batches += 1
    # logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
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
    logger.info(f"任务结束: save_stocks_realtime_min_data_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

#  ================ 历史(月线)数据任务 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_month_data_history_batch(stock_codes: List[str], cache_manager=None):
    """
    从Tushare批量获取月线交易数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(月线)数据任务 批次...")
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        # 3. 执行业务逻辑
        result = await stock_time_trade_dao.save_monthly_time_trade_by_stock_codes(stock_codes)
        logger.info(f"历史(月线)数据任务 结果：{result}")
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_stocks_month_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_month_data_history_task(self, batch_size: int = 50, cache_manager=None): # 限量：单次最大6000行数据
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_realtime_data_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_stocks_month_data_history_task (调度器模式) - 获取股票列表并分派批量任务 (批次大小: {batch_size})")
    favorite_codes = []
    non_favorite_codes = []
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    # 1. 定义一个异步 main 函数，用于安全地执行所有需要异步环境的操作
    async def main():
        # nonlocal 关键字允许内部函数修改外部函数的变量
        nonlocal favorite_codes, non_favorite_codes
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
            # 使用新的批量任务，并指定队列
            save_month_data_history_batch.s(batch).set().apply_async()
            total_dispatched_batches += 1
    # logger.info(f"已为 {total_favorite_stocks} 个自选股分派了 {total_dispatched_batches} 个批次任务。")
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
    logger.info(f"任务结束: save_stocks_month_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}


# ============== （历史）每日筹码分布任务 ==============
@celery_app.task(queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_cyq_chips_history_batch(ts_code: str, start_date: datetime.date = None, end_date: datetime.date = None, cache_manager=None):
    """
    从Tushare批量获取实时分钟级交易数据并保存到数据库（异步并发处理）
    Args:
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    print(f"开始处理 ts_code: {ts_code} 的每日筹码分布数据...")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        # 3. 根据ts_code从数据库获取StockInfo对象
        stock = await stock_basic_dao.get_stock_by_code(stock_code=ts_code)
        if not stock:
            logger.warning(f"在数据库中未找到代码为 {ts_code} 的股票，跳过此任务。")
            return
        # 4. 使用从数据库获取的stock对象进行后续操作
        result = await stock_time_trade_dao.save_cyq_chips_for_stock(stock=stock, start_date=start_date, end_date=end_date)
        print(f"保存 {ts_code} （历史）每日筹码分布 数据完成。 result: {result} ")
    async_to_sync(main)()

@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='190/m')
@with_cache_manager
def save_cyq_perf_for_stocky_batch(ts_code: str, start_date: datetime.date = None, end_date: datetime.date = None, cache_manager=None):
    """
    从Tushare获取单只股票的每日筹码及胜率数据并保存。
    通过将大任务拆分为每个股票的小任务，利用Celery的并发能力和速率限制，
    实现并行处理，从而大幅提升整体效率。
    Args:
        ts_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    """
    print(f"开始处理 ts_code: {ts_code} 的每日筹码及胜率数据...")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
    # 3. 获取股票对象
        stock = await stock_basic_dao.get_stock_by_code(stock_code=ts_code)
        if not stock:
            logger.warning(f"在数据库中未找到代码为 {ts_code} 的股票，跳过此任务。")
            return
        # 4. 调用DAO方法处理单个股票
        result = await stock_time_trade_dao.save_cyq_perf_for_stock(stock=stock, start_date=start_date, end_date=end_date)
        print(f"保存 {ts_code} 每日筹码及胜率 数据完成。 result: {result} ")
    async_to_sync(main)()

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_history_task', queue='celery')
@with_cache_manager
def save_cyq_data_history_task(self, cache_manager=None):
    """
    调度器任务（已优化）：
    1. 获取所有股票列表。
    2. 为每只股票分派 save_cyq_chips_history_batch 任务（筹码分布）。
    3. 为每只股票分派 save_cyq_perf_for_stock_task 任务（筹码及胜率）。
    这两个任务队列将由Celery Workers并行处理，并各自遵守其速率限制。
    """
    logger.info(f"任务启动: save_cyq_data_history_task (调度器模式) - 获取股票列表并分派并行任务")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    today_date = timezone.now().date()
    all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
    logger.info(f"获取到 {len(all_stocks)} 只股票，开始为每只股票分派任务...")
    for stock in all_stocks:
        # 分派筹码分布任务
        save_cyq_chips_history_batch.s(ts_code=stock.stock_code, start_date=None, end_date=today_date).apply_async()
        # [修改] 在同一个循环中，为每只股票分派新的筹码及胜率任务
        save_cyq_perf_for_stocky_batch.s(ts_code=stock.stock_code, start_date=None, end_date=today_date).apply_async()
    logger.info(f"所有 {len(all_stocks)} 只股票的筹码分布和筹码胜率任务已全部分派完毕。")
    logger.info(f"任务结束: save_cyq_data_history_task (调度器模式)")

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.refetch_incomplete_cyq_chips', queue='celery')
def refetch_incomplete_cyq_chips(self, record_threshold=1000):
    from django.db.models import Count
    """
    Celery 定时任务：查找所有 StockCyqChips* 表中，按股票分组后记录数少于指定阈值的股票，
    并为这些股票重新触发全量历史数据拉取任务。

    这个任务通常用于数据修复或补充，例如当发现某些股票的历史筹码数据不完整时。

    Args:
        self: Celery 任务实例，由 bind=True 自动注入。
        record_threshold (int): 记录数的阈值。如果一只股票的筹码数据记录总数低于此值，
                                 将被视为不完整。默认为 800。
    """
    print(f"开始执行任务：检查并重新拉取记录数少于 {record_threshold} 条的筹码分布数据...")
    logger.info(f"开始执行任务：检查并重新拉取记录数少于 {record_threshold} 条的筹码分布数据...")
    # 定义所有需要检查的筹码分布数据表模型
    chips_models = [
        StockCyqChipsCY,
        StockCyqChipsSZ,
        StockCyqChipsKC,
        StockCyqChipsSH,
        StockCyqChipsBJ,
    ]
    # 使用集合来存储需要重新拉取数据的股票代码，以自动处理重复项
    stocks_to_refetch = set()
    try:
        # 遍历每一个筹码分布模型进行检查
        for model in chips_models:
            table_name = model._meta.db_table
            print(f"正在检查表: {table_name}...")
            logger.info(f"正在检查表: {table_name}...")
            # 【核心查询逻辑】
            # 1. `values('stock__stock_code')`: 按关联的 StockInfo 模型的 stock_code 字段进行分组。
            # 2. `annotate(record_count=Count('id'))`: 为每个分组计算其拥有的记录数量，并命名为 record_count。
            # 3. `filter(record_count__lt=record_threshold)`: 筛选出记录数量小于指定阈值的分组。
            # 4. `values_list('stock__stock_code', flat=True)`: 提取这些分组的 stock_code，并返回一个扁平化的列表，如 ['000001.SZ', '000002.SZ']。
            incomplete_stocks_query = model.objects.values('stock__stock_code') \
                .annotate(record_count=Count('id')) \
                .filter(record_count__lt=record_threshold) \
                .values_list('stock__stock_code', flat=True)
            # 执行查询并将结果转换为列表
            found_codes = list(incomplete_stocks_query)
            if found_codes:
                count = len(found_codes)
                print(f"在表 {table_name} 中发现 {count} 只股票记录不完整。")
                logger.info(f"在表 {table_name} 中发现 {count} 只股票记录不完整。")
                # 将新发现的股票代码添加到集合中
                stocks_to_refetch.update(found_codes)
            else:
                print(f"在表 {table_name} 中未发现记录不完整的股票。")
                logger.info(f"在表 {table_name} 中未发现记录不完整的股票。")
        total_count = len(stocks_to_refetch)
        print(f"检查完成。总共发现 {total_count} 只股票需要重新拉取数据。")
        logger.info(f"检查完成。总共发现 {total_count} 只股票需要重新拉取数据。")
        # 如果有需要处理的股票，则为每一只股票分发一个Celery任务
        if total_count > 0:
            print("开始分发数据拉取任务...")
            for stock_code in stocks_to_refetch:
                print(f"为股票 {stock_code} 创建数据拉取任务。")
                # 调用已有的 `save_cyq_chips_history_batch` 任务来拉取数据。
                # 我们只传递 ts_code，让任务使用其默认的日期范围（通常是全量拉取）。
                save_cyq_chips_history_batch.delay(ts_code=stock_code, start_date=None, end_date=None)
            print(f"已为 {total_count} 只股票成功分发任务。")
            logger.info(f"已为 {total_count} 只股票成功分发任务。")
        return f"任务完成。共检查 {len(chips_models)} 个表，为 {total_count} 只股票分发了更新任务。"
    except Exception as e:
        print(f"执行任务 refetch_incomplete_cyq_chips 时发生严重错误: {e}")
        logger.error(f"执行任务 refetch_incomplete_cyq_chips 时发生严重错误: {e}", exc_info=True)
        # 发生错误时，可以选择让Celery重试任务，例如5分钟后重试
        # self.retry(exc=e, countdown=300)
        raise # 重新抛出异常，以便在Celery监控工具中看到任务失败状态

#  ================ 清理非交易日数据任务 ================

# 将所有需要清理的模型统一放入一个列表中，方便遍历和维护
# 这样做的好处是，未来如果新增了类似的模型，只需要在这里添加即可
DATA_MODELS_TO_CLEAN = [
    # 日线数据模型
    StockDailyData_SZ, StockDailyData_SH, StockDailyData_CY, StockDailyData_KC, StockDailyData_BJ,
    # 分钟数据模型
    StockMinuteData_5_SZ, StockMinuteData_15_SZ, StockMinuteData_30_SZ, StockMinuteData_60_SZ,
    StockMinuteData_5_SH, StockMinuteData_15_SH, StockMinuteData_30_SH, StockMinuteData_60_SH,
    StockMinuteData_5_BJ, StockMinuteData_15_BJ, StockMinuteData_30_BJ, StockMinuteData_60_BJ,
    StockMinuteData_5_CY, StockMinuteData_15_CY, StockMinuteData_30_CY, StockMinuteData_60_CY,
    StockMinuteData_5_KC, StockMinuteData_15_KC, StockMinuteData_30_KC, StockMinuteData_60_KC,
]

@celery_app.task(bind=True, name='tasks.tushare.stock_time_trade_tasks.cleanup_non_trade_day_data', queue='clean_data')
def cleanup_non_trade_day_data(self):
    """
    一个Celery任务，用于清理所有股票数据表中在非交易日产生的无效数据。
    任务会遍历所有指定的数据模型，高效地找出并删除这些记录。
    【已修正】能够自动识别 DateField 和 DateTimeField 并采用正确的查询策略。
    """
    print("开始执行【清理非交易日数据】任务...")
    total_deleted_count = 0

    # 1. 首先，从交易日历中获取所有非交易日的日期集合，这样后续可以快速判断
    all_non_trade_dates = set(
        TradeCalendar.objects.filter(is_open=False).values_list('cal_date', flat=True)
    )
    if not all_non_trade_dates:
        print("警告：交易日历中未找到任何休市日期记录，任务提前终止。请检查TradeCalendar数据。")
        return "任务终止，未找到休市日期。"
    
    print(f"已从交易日历加载 {len(all_non_trade_dates)} 个休市日期。")

    # 2. 遍历每一个需要清理的数据模型
    for model in DATA_MODELS_TO_CLEAN:
        model_name = model._meta.verbose_name_plural or model.__name__
        table_name = model._meta.db_table
        print(f"\n--- 正在处理模型: {model_name} (数据表: {table_name}) ---")

        # --- 修改/新增代码开始 ---
        # 3. 动态检查 'trade_time' 字段的类型，以决定使用哪种查询方式
        field_object = model._meta.get_field('trade_time')
        is_datetime_field = isinstance(field_object, models.DateTimeField)
        
        print(f"检测到 'trade_time' 字段类型为: {'DateTimeField' if is_datetime_field else 'DateField'}")

        # 4. 根据字段类型，使用正确的查询方式从数据表中提取所有唯一的日期
        if is_datetime_field:
            # 对于 DateTimeField，使用 TruncDate
            dates_in_table = set(
                model.objects.annotate(trade_date=TruncDate('trade_time'))
                .values_list('trade_date', flat=True)
                .distinct()
            )
        else:
            # 对于 DateField，直接获取值即可
            dates_in_table = set(
                model.objects.values_list('trade_time', flat=True).distinct()
            )
        # --- 修改/新增代码结束 ---

        if not dates_in_table:
            print(f"数据表 {table_name} 中没有数据，跳过。")
            continue
        
        print(f"在表 {table_name} 中发现 {len(dates_in_table)} 个唯一日期。")

        # 5. 计算出需要被删除的日期（即存在于数据表中的日期，同时也是我们已知的休市日期）
        dates_to_delete = dates_in_table & all_non_trade_dates

        if not dates_to_delete:
            print(f"在表 {table_name} 中未发现任何非交易日数据，无需清理。")
            continue

        print(f"在表 {table_name} 中发现 {len(dates_to_delete)} 个非交易日需要清理") # .: {sorted(list(dates_to_delete))}

        # --- 修改/新增代码开始 ---
        # 6. 根据字段类型，使用正确的过滤器进行批量删除
        if is_datetime_field:
            # 对于 DateTimeField，使用 '__date__in'
            deleted_info = model.objects.filter(trade_time__date__in=dates_to_delete).delete()
        else:
            # 对于 DateField，使用 '__in'
            deleted_info = model.objects.filter(trade_time__in=dates_to_delete).delete()
        # --- 修改/新增代码结束 ---
        
        deleted_count = deleted_info[0]
        total_deleted_count += deleted_count

        print(f"成功从 {table_name} 中删除了 {deleted_count} 条非交易日记录。")

    print(f"\n--- 任务执行完毕 ---")
    print(f"总共删除了 {total_deleted_count} 条非交易日记录。")
    return f"任务完成，总共删除 {total_deleted_count} 条记录。"










