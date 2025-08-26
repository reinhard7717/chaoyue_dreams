# tasks\stock_time_trade_tasks.py
import asyncio
import logging
import datetime
from django.conf import settings
from asgiref.sync import async_to_sync
from celery import chord, group, chain
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

@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_today_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_today_task(trade_time_str=None, batch_size: int = 310, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派当日分钟数据获取任务。
    """
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    logger.info(f"任务启动: save_stocks_minute_data_today_task (调度器模式) - 批次大小: {batch_size}")
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
def save_stocks_minute_data_batch(stock_codes: list, trade_date_str: str, cache_manager=None):
    """
    从Tushare批量获取并保存指定交易日的分钟级交易数据（异步并发处理）
    Args:
        stock_codes: 股票代码列表
        trade_date_str: 需要获取数据的交易日字符串 (格式: 'YYYY-MM-DD')
    """
    # [修改] 直接使用传入的 trade_date_str 参数，不再内部计算日期
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    
    # [修改] 根据传入的日期构造时间范围
    start_date_str = f"{trade_date_str} 00:00:00"
    end_date_str = f"{trade_date_str} 23:59:59"
    
    # [修改] 更新日志和调试信息
    print(f"开始保存 {len(stock_codes)} 个股票, 交易日 {trade_date_str} 的分钟数据任务...")
    logger.info(f"开始处理 {len(stock_codes)} 只股票在 {trade_date_str} 的分钟数据。")

    async def main():
        # 3. 执行业务逻辑
        return await stock_time_trade_dao.save_minute_time_trade_history_by_stock_codes(
            stock_codes=stock_codes,
            start_date_str=start_date_str,
            end_date_str=end_date_str
        )
    
    async_to_sync(main)()
    
    # [修改] 更新完成后的日志和调试信息
    print(f"保存股票 {len(stock_codes)} 个的交易日 ({trade_date_str}) 分钟级交易数据完成。")
    logger.info(f"完成处理 {len(stock_codes)} 只股票在 {trade_date_str} 的分钟数据。")

# 2. 重构并重命名调度器任务，以处理最近N天的数据
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_latest_days_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_latest_days_task(batch_size: int = 310, num_days: int = 5, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派获取最近N个交易日分钟数据的任务。
    Args:
        batch_size (int): 每个批次处理的股票数量。
        num_days (int): 需要获取最近多少个交易日的数据。
    """
    logger.info(f"任务启动: save_stocks_minute_data_latest_days_task - 获取最近 {num_days} 天数据, 批次大小: {batch_size}")

    # [新增] 使用 TradeCalendar 获取最近 num_days 个交易日
    trade_dates = TradeCalendar.get_latest_n_trade_dates(n=num_days)
    if not trade_dates:
        logger.warning("未能从交易日历中获取到最近的交易日列表，任务终止。")
        print("调试: 未能获取到最近的交易日列表，任务终止。")
        return {"status": "skipped", "message": "未能获取到交易日列表"}

    logger.info(f"成功获取到最近 {len(trade_dates)} 个交易日: {trade_dates}")
    print(f"调试: 将为以下交易日分派任务: {trade_dates}")

    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def main():
        return await stock_basic_dao.get_stock_list()
    all_stocks = async_to_sync(main)()

    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}

    all_stock_codes = [stock.stock_code for stock in all_stocks]
    total_codes_count = len(all_stock_codes)
    total_dispatched_batches = 0
    logger.info(f"准备为 {total_codes_count} 个股票分派任务...")

    # [修改] 增加外层循环，遍历所有需要处理的交易日
    for trade_date in trade_dates:
        trade_date_str = trade_date.strftime('%Y-%m-%d')
        logger.info(f"开始为交易日 {trade_date_str} 分派批量任务...")
        print(f"调试: 正在为交易日 {trade_date_str} 创建批次...")
        
        # 内层循环，将所有股票代码分批处理
        for i in range(0, total_codes_count, batch_size):
            batch_codes = all_stock_codes[i:i + batch_size]
            if batch_codes:
                # [修改] 调用新的工作任务，并传入股票批次和对应的交易日
                save_stocks_minute_data_batch.s(
                    stock_codes=batch_codes, 
                    trade_date_str=trade_date_str
                ).set().apply_async()
                total_dispatched_batches += 1
    
    logger.info(f"任务结束: save_stocks_minute_data_latest_days_task - 共为 {len(trade_dates)} 个交易日分派了 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches, "trade_days_processed": len(trade_dates)}

#  ================ 今日基本信息 数据任务 ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_today_task', queue='SaveData_TimeTrade')
@with_cache_manager
def save_stocks_daily_basic_data_today_task(cache_manager=None):
    """
    【V2.1 健壮性增强版】
    获取并保存今日股票重要的基本面指标，并执行行业轮动分析。
    - 核心修复: 增加了明确的返回值和异常捕获。
    - 核心增强: 在DAO调用后增加了对结果的检查，如果获取数据为空，会明确记录日志。
    """
    task_name = 'save_stocks_daily_basic_data_today_task'
    logger.info(f"任务启动: {task_name} - 开始处理今日股票重要的基本面指标...")
    try:
        service = IndicatorService(cache_manager)
        stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
        today_date = timezone.now().date()
        print(f"[{task_name}] 开始保存 今日股票重要的基本面指标...")
        async def main():
            # 执行业务逻辑并捕获返回值
            result = await stock_time_trade_dao.save_stock_daily_basic_history_by_trade_date(trade_date=today_date)
            if not result or result.get("创建/更新成功", 0) == 0:
                logger.warning(f"[{task_name}] 从Tushare获取每日指标数据为空或未保存任何数据 for {today_date}。这可能是因为数据尚未就绪。")
            else:
                print(f"[{task_name}] 保存 今日股票重要的基本面指标 完成。result: {result}")
            # 无论是否获取到数据，都尝试执行行业轮动分析
            rotation_report = await service.analyze_industry_rotation(datetime.date.today(), lookback_days=10)
            print(f"--- [{task_name}] 行业轮动强度报告 ---")
            print(rotation_report.head(10))
            return {
                "status": "success",
                "message": f"每日基本面指标和行业轮动分析完成 for {today_date}",
                "db_result": result
            }
        task_result = async_to_sync(main)()
        logger.info(f"任务成功: {task_name} - {task_result.get('message')}")
        return task_result
    except Exception as e:
        logger.error(f"任务失败: {task_name} 执行时发生严重错误: {e}", exc_info=True)
        raise e

#  ================ 昨日基本信息 数据任务 ================
# 重命名任务并更新Celery name属性，使其能处理最近N天的数据
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_latest_days_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_stocks_daily_basic_data_latest_days_task(num_days: int = 5, cache_manager=None):
    """
    【无绑定版】
    获取并保存最近N个交易日所有股票重要的基本面指标。
    Args:
        num_days (int): 需要获取最近多少个交易日的数据。
    """
    # [修改] 更新启动日志
    logger.info(f"任务启动: save_stocks_daily_basic_data_latest_days_task - 获取最近 {num_days} 个交易日的基本面指标。")

    # [新增] 使用 TradeCalendar 获取最近 num_days 个交易日
    trade_dates = TradeCalendar.get_latest_n_trade_dates(n=num_days)
    if not trade_dates:
        logger.warning("未能从交易日历中获取到最近的交易日列表，任务终止。")
        print("调试: 未能获取到最近的交易日列表，任务终止。")
        return {"status": "skipped", "message": "未能获取到交易日列表"}
    
    logger.info(f"成功获取到最近 {len(trade_dates)} 个交易日: {trade_dates}")
    print(f"调试: 将为以下交易日获取基本面指标: {trade_dates}")

    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    # [新增] 用于存储每个交易日处理结果的字典
    results_summary = {}

    # [修改] 循环处理获取到的每一个交易日
    for trade_date in trade_dates:
        trade_date_str = trade_date.strftime('%Y-%m-%d')
        logger.info(f"开始处理交易日 {trade_date_str} 的股票基本面指标...")
        print(f"开始保存交易日 {trade_date_str} 的股票基本面指标...")

        async def main():
            # [修改] 调用DAO时传入循环中的当前交易日
            return await stock_time_trade_dao.save_stock_daily_basic_history_by_trade_date(trade_date=trade_date)
        
        result = async_to_sync(main)()
        # [新增] 将当天的处理结果存入汇总字典
        results_summary[trade_date_str] = result
        
        print(f"保存交易日 {trade_date_str} 的股票基本面指标完成。result: {result}")
        logger.info(f"处理交易日 {trade_date_str} 的数据完成。")

    # [修改] 更新结束日志和返回信息
    logger.info(f"任务结束: save_stocks_daily_basic_data_latest_days_task - 共处理了 {len(trade_dates)} 个交易日。")
    return {"status": "success", "processed_days": len(trade_dates), "details": results_summary}

#  ================ 日线数据任务（当日） ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_day_data_today_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_day_data_today_task(cache_manager=None):
    """
    【无绑定版】
    获取并保存当日日线数据。
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    print("开始保存 日线数据任务（当日）...")
    async def main():
        return await stock_time_trade_dao.save_daily_time_trade_history_by_trade_dates(trade_date=today_date)
    result = async_to_sync(main)()
    print(f"保存 日线数据任务（当日） 完成。result: {result}")

#  ================ 日线数据任务（昨日） ================
STOCK_BATCH_SIZE = 400

@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_day_data_latest_days_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_day_data_latest_days_task(num_days: int = 5, cache_manager=None):
    """
    【V2.0 - 批量优化版】
    保存最近N个交易日的日线数据。
    优化点: 按天循环，在每天内部将所有股票分批，批量从Tushare获取数据。
    这极大地减少了API请求次数 (从 每天N次API请求 优化为 每天 ceil(总股票数/批大小) 次)。
    
    Args:
        num_days (int): 需要获取最近多少个交易日的数据。
    """
    logger.info(f"任务启动: save_day_data_latest_days_task - 获取最近 {num_days} 个交易日的日线数据。")
    
    # 1. 获取需要处理的交易日列表
    trade_dates = TradeCalendar.get_latest_n_trade_dates(n=num_days)
    if not trade_dates:
        logger.warning("未能从交易日历中获取到最近的交易日列表，任务终止。")
        return {"status": "skipped", "message": "未能获取到交易日列表"}
    logger.info(f"成功获取到最近 {len(trade_dates)} 个交易日: {trade_dates}")

    # 2. 一次性获取所有股票代码
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    try:
        all_stocks = async_to_sync(stock_basic_dao.get_stock_list)()
        all_stock_codes = [s.stock_code for s in all_stocks]
        if not all_stock_codes:
            logger.warning("未能获取到任何股票列表，任务终止。")
            return {"status": "skipped", "message": "未能获取到股票列表"}
        logger.info(f"获取到 {len(all_stock_codes)} 只股票待处理。")
    except Exception as e:
        logger.error(f"获取股票列表时发生错误: {e}")
        return {"status": "failed", "message": f"获取股票列表失败: {e}"}

    # 3. 按天、按股票批次处理数据
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    results_summary = {}
    total_batches = ceil(len(all_stock_codes) / STOCK_BATCH_SIZE)

    for trade_date in trade_dates:
        trade_date_str = trade_date.strftime('%Y-%m-%d')
        trade_date_api_format = trade_date.strftime('%Y%m%d') # Tushare API格式
        
        logger.info(f"===== 开始处理交易日 {trade_date_str} 的日线数据... =====")
        print(f"===== 开始保存交易日 {trade_date_str} 的日线数据... =====")
        
        day_results = {}
        
        # 将所有股票代码分批处理
        for i in range(0, len(all_stock_codes), STOCK_BATCH_SIZE):
            batch_num = (i // STOCK_BATCH_SIZE) + 1
            stock_batch = all_stock_codes[i:i + STOCK_BATCH_SIZE]
            
            logger.info(f"[{trade_date_str}] 处理批次 {batch_num}/{total_batches} (含 {len(stock_batch)} 只股票)")
            print(f"调试: [{trade_date_str}] 处理批次 {batch_num}/{total_batches}...")

            async def main():
                # 调用修改后的DAO方法，并传入 trade_date
                return await stock_time_trade_dao.save_daily_time_trade_history_by_stock_codes(
                    stock_codes=stock_batch,
                    trade_date=trade_date_api_format
                )
            
            try:
                result = async_to_sync(main)()
                day_results[f"batch_{batch_num}"] = result
                print(f"调试: [{trade_date_str}] 批次 {batch_num} 保存完成。result: {result}")
                logger.info(f"[{trade_date_str}] 批次 {batch_num} 处理完成。")
            except Exception as e:
                error_msg = f"[{trade_date_str}] 批次 {batch_num} 处理失败: {e}"
                logger.error(error_msg)
                print(f"调试: {error_msg}")
                day_results[f"batch_{batch_num}"] = {"status": "error", "message": str(e)}

        results_summary[trade_date_str] = day_results
        logger.info(f"===== 处理交易日 {trade_date_str} 的日线数据完成。 =====")

    logger.info(f"任务结束: save_day_data_latest_days_task - 共处理了 {len(trade_dates)} 个交易日。")
    return {"status": "success", "processed_days": len(trade_dates), "details": results_summary}

#  ================ 周线数据任务（当日） ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_week_data_today_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_week_data_today_task(cache_manager=None):
    """
    【无绑定版】
    获取并保存当日周线数据。
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    day_str = today_date.strftime('%Y%m%d')
    print("开始保存 周线数据任务（当日）...")
    async def main():
        return await stock_time_trade_dao.save_weekly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 周线数据任务（当日） 完成。result: {result}")

#  ================ 周线数据任务（昨日） ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_week_data_yesterday_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_week_data_yesterday_task(cache_manager=None):
    """
    【无绑定版】
    获取并保存昨日周线数据。
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)
    day_str = yesterday.strftime('%Y%m%d')
    print("开始保存 周线数据任务（昨日）...")
    async def main():
        return await stock_time_trade_dao.save_weekly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 周线数据任务（昨日） 完成。result: {result}")

#  ================ 月线数据任务（当日） ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_month_data_today_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_month_data_today_task(cache_manager=None):
    """
    【无绑定版】
    获取并保存当日月线数据。
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    today_date = timezone.now().date()
    day_str = today_date.strftime('%Y%m%d')
    print("开始保存 月线数据（当日）...")
    async def main():
        # [代码已修复] 此处应调用月线方法 save_monthly_time_trade
        return await stock_time_trade_dao.save_monthly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 月线数据（当日） 完成。result: {result}")

#  ================ 月线数据任务（昨日） ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_month_data_yesterday_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_month_data_yesterday_task(cache_manager=None):
    """
    【无绑定版】
    获取并保存昨日月线数据。
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    print("开始保存 月线数据（昨日）...")
    today_date = timezone.now().date()
    yesterday = today_date - datetime.timedelta(days=1)
    day_str = yesterday.strftime('%Y%m%d')
    async def main():
        # [代码已修复] 此处应调用月线方法 save_monthly_time_trade
        return await stock_time_trade_dao.save_monthly_time_trade(trade_date=day_str)
    result = async_to_sync(main)()
    print(f"保存 月线数据（昨日） 完成。result: {result}")
    save_cyq_data_yesterday_task.delay()
    print(f"保存 月线数据（昨日） 数据完成。")

# ===================================================
#      每日筹码分布任务（新版 - 两级分发模式）
# ===================================================
# --- 1. 执行器任务 (Executor Task) ---
# 这个任务是真正干活的，只处理单个股票
@celery_app.task(
    name='tasks.tushare.stock_time_trade_tasks.save_single_stock_cyq_chips',
    queue="SaveHistoryData_TimeTrade",
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3},
    retry_backoff=True,
    retry_backoff_max=300
)
@with_cache_manager
# [修改] 修改函数签名，增加 start_date_str 和 end_date_str，并为 trade_date_str 设置默认值 None 以实现兼容
def save_single_stock_cyq_chips(stock_code: str, trade_date_str: str = None, *, start_date_str: str = None, end_date_str: str = None, cache_manager=None):
    """
    【执行器】获取并保存【单个】股票在【指定日期或日期范围】的CYQ筹码分布数据。
    [修改] 适配新的DAO方法，并支持日期范围。
    """
    # print(f"执行器任务[CYQ Chips]启动: stock={stock_code}, trade_date={trade_date_str}, start_date={start_date_str}, end_date={end_date_str}")

    async def _async_task():
        """将所有异步逻辑封装在一个协程中"""
        stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
        stock_basic_dao = StockBasicInfoDao(cache_manager)
        
        stock_obj = await stock_basic_dao.get_stock_by_code(stock_code)
        if not stock_obj:
            logger.warning(f"执行器[CYQ Chips]: 未找到股票 {stock_code} 的信息，任务终止。")
            return

        # [修改] 增加逻辑判断，以兼容单日和日期范围两种模式
        start_date, end_date = None, None
        if trade_date_str:
            # 兼容旧的单日模式
            start_date = datetime.datetime.strptime(trade_date_str, '%Y-%m-%d').date()
            end_date = start_date # 单日查询时，开始和结束日期相同
            print(f"调试: 执行器接收到单日参数: stock={stock_code}, date={start_date}")
        elif start_date_str and end_date_str:
            # 新的日期范围模式
            start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d').date()
            end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d').date()
            print(f"调试: 执行器接收到日期范围参数: stock={stock_code}, start={start_date}, end={end_date}")
        else:
            logger.error(f"执行器[CYQ Chips]错误: 必须提供 trade_date_str 或 (start_date_str 和 end_date_str)。stock={stock_code}")
            return

        # [修改] 调用DAO方法时，同时传入 start_date 和 end_date
        await stock_time_trade_dao.save_cyq_chips_for_stock(
            stock=stock_obj, 
            start_date=start_date,
            end_date=end_date # [修改] 传入结束日期
        )

    try:
        asyncio.run(_async_task())
    except Exception as e:
        # [修改] 完善错误日志
        log_ctx = f"stock={stock_code}, trade_date={trade_date_str}, start={start_date_str}, end={end_date_str}"
        logger.error(f"执行器任务[CYQ Chips]失败: {log_ctx}, error={e}", exc_info=True)
        raise

# --- 2. 分发器任务 (Dispatcher Task) ---
# [修改] 修改函数签名，增加 start_date_str 和 end_date_str，并为 trade_date_str 设置默认值 None 以实现兼容
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.dispatch_cyq_tasks_for_date', queue='celery', bind=True)
@with_cache_manager
def dispatch_cyq_tasks_for_date(self, trade_date_str: str = None, *, start_date_str: str = None, end_date_str: str = None, cache_manager: CacheManager):
    """
    【分发器 V2.3 - 兼容日期范围】
    - 核心修改: 能够接收 start_date_str 和 end_date_str，为指定范围分发任务。
                同时保留对 trade_date_str 的支持，以兼容旧的调用方式。
    """
    # [修改] 增加逻辑判断，以兼容单日和日期范围两种模式
    final_start_date_str, final_end_date_str = None, None
    log_date_info = ""
    if trade_date_str:
        # 兼容旧的单日模式
        final_start_date_str = trade_date_str
        final_end_date_str = trade_date_str
        log_date_info = f"日期 {trade_date_str}"
    elif start_date_str and end_date_str:
        # 新的日期范围模式
        final_start_date_str = start_date_str
        final_end_date_str = end_date_str
        log_date_info = f"日期范围 {start_date_str} 到 {end_date_str}"
    else:
        logger.error("分发器错误: 必须提供 trade_date_str 或 (start_date_str 和 end_date_str)。")
        return {"status": "error", "message": "Invalid date arguments"}

    print(f"分发器任务[V2.3 范围兼容版]启动，准备为 {log_date_info} 分发CYQ任务...")
    async def main():
        stock_dao = StockBasicInfoDao(cache_manager_instance=cache_manager)
        print("分发器：正在通过 DAO (含缓存) 获取股票列表...")
        stock_list = await stock_dao.get_stock_list()
        all_stock_codes = [stock.stock_code for stock in stock_list]
        if not all_stock_codes:
            logger.warning(f"分发器：未能通过DAO获取到任何股票代码，{log_date_info} 的任务未分发。")
            return {"status": "skipped", "message": "no stocks found via DAO"}
        
        stock_count = len(all_stock_codes)
        chunk_size_per_stock = getattr(settings, 'CYQ_TASK_CHUNK_SIZE', 190) 
        delay_between_chunks = getattr(settings, 'CYQ_TASK_CHUNK_DELAY', 60)
        print(f"分发器：获取到 {stock_count} 只股票，将以每批 {chunk_size_per_stock} 只、间隔 {delay_between_chunks} 秒的速率平滑分发...")
        
        all_tasks = []
        for stock_code in all_stock_codes:
            # [修改] 调用执行器任务时，使用新的关键字参数传递日期范围
            all_tasks.append(save_single_stock_cyq_chips.s(
                stock_code=stock_code, 
                start_date_str=final_start_date_str,
                end_date_str=final_end_date_str
            ))
        
        total_tasks = len(all_tasks)
        task_chunk_size = chunk_size_per_stock * 1
        batch_num = 0
        for i in range(0, total_tasks, task_chunk_size):
            chunk_of_tasks = all_tasks[i : i + task_chunk_size]
            task_group = group(chunk_of_tasks)
            countdown = batch_num * delay_between_chunks
            task_group.apply_async(countdown=countdown)
            print(f"  -> 第 {batch_num + 1} 批任务 (共 {len(chunk_of_tasks)} 个) 已调度，将在 {countdown} 秒后执行。")
            batch_num += 1
        
        message = f"分发器：成功调度了 {stock_count} 只股票的CYQ任务 (共 {total_tasks} 个)，已分 {batch_num} 批平滑处理。"
        print(message)
        logger.info(message)
        return {"status": "dispatched_smoothly", "stock_count": stock_count, "chunk_count": batch_num}

    try:
        return async_to_sync(main)()
    except Exception as e:
        logger.error(f"分发器任务失败，日期信息: {log_date_info}, error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# --- 3. 调度器任务 (Scheduler Task) ---
# 这个任务由Celery Beat调用，它只负责调用分发器
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_today_task', queue='celery')
def save_cyq_data_today_task():
    """
    【调度器】用于获取【当天】CYQ数据的入口任务。
    它只调用分发器任务。
    """
    logger.info(f"调度器任务[CYQ Today]启动...")
    today_date_str = timezone.now().date().strftime('%Y-%m-%d')
    # 调用分发器，并传入当天的日期 (保持旧的调用方式)
    dispatch_cyq_tasks_for_date.delay(trade_date_str=today_date_str)
    return {"status": "dispatcher_called", "date": today_date_str}

@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_latest_days_task', queue='celery')
def save_cyq_data_latest_days_task(num_days: int = 5):
    """
    【调度器 V2.0 - 范围优化版】用于获取【最近N个交易日】CYQ数据的入口任务。
    它会计算日期范围，然后调用一次分发器任务。
    Args:
        num_days (int): 需要获取最近多少个交易日的数据。
    """
    logger.info(f"调度器任务[CYQ Latest Days - 范围优化版]启动: 准备为最近 {num_days} 个交易日分派任务。")

    trade_dates = TradeCalendar.get_latest_n_trade_dates(n=num_days)
    if not trade_dates:
        logger.warning("未能从交易日历中获取到最近的交易日列表，任务终止。")
        print("调试: 未能获取到最近的交易日列表，任务终止。")
        return {"status": "skipped", "message": "未能获取到交易日列表"}

    # [修改] 核心逻辑变更：不再循环，而是确定日期范围
    # trade_dates 列表是降序的（从近到远）
    end_date = trade_dates[0]      # 最近的交易日作为结束日期
    start_date = trade_dates[-1]   # 最远的交易日作为开始日期

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    logger.info(f"获取到 {len(trade_dates)} 个交易日，确定的处理范围为: {start_date_str} 到 {end_date_str}")
    print(f"调试: 将为日期范围 {start_date_str} - {end_date_str} 调用一次分发器。")

    # [修改] 一次性调用分发器，传入开始和结束日期
    dispatch_cyq_tasks_for_date.delay(
        start_date_str=start_date_str,
        end_date_str=end_date_str
    )

    logger.info(f"调度器任务[CYQ Latest Days - 范围优化版]完成: 已为日期范围 {start_date_str} - {end_date_str} 调用了分发器。")
    return {
        "status": "dispatcher_called_with_range", 
        "processed_days_count": len(trade_dates), 
        "start_date": start_date_str,
        "end_date": end_date_str
    }

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
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_this_week_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_this_week_task(batch_size: int = 10, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派本周分钟数据获取任务。
    """
    logger.info(f"任务启动: save_stocks_minute_data_this_week_task (调度器模式) - 批次大小: {batch_size}")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def main():
        return await stock_basic_dao.get_stock_list()
    all_stocks = async_to_sync(main)()
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    total_codes_count = len(all_stock_codes)
    total_dispatched_batches = 0
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
            save_minute_data_this_week_batch.s(stock_codes=batch_codes).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"任务结束: save_stocks_minute_data_this_week_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
    return {"status": "success", "dispatched_batches": total_dispatched_batches}

#  ================ （本周）日线数据任务 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_day_data_this_week_batch(cache_manager=None):
    """
    获取并保存本周的日线数据。
    """
    this_monday, this_friday = get_this_monday_and_friday()
    logger.info(f"开始处理 {this_monday} - {this_friday} 的历史(日线)数据任务...")
    # [代码已修复] 修复变量名错误，使用由装饰器注入的 cache_manager
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        return await stock_time_trade_dao.save_daily_time_trade_history_by_trade_dates(start_date=this_monday, end_date=this_friday)
    return_info = async_to_sync(main)()
    print(f"完成 {this_monday} - {this_friday} 的日线数据保存，{return_info}。")
    time.sleep(5)

#  ================ （本周）每日基本信息数据任务 ================
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_this_week_task', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_stocks_daily_basic_data_this_week_task(cache_manager=None):
    """
    【无绑定版】
    获取并保存本周股票重要的基本面指标。
    """
    logger.info(f"开始处理本周股票重要的基本面指标...")
    this_monday, this_friday = get_this_monday_and_friday()
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    async def main():
        print(f"开始保存 本周({this_monday} - {this_friday}) 股票重要的基本面指标...")
        # [代码已修复] 执行业务逻辑并捕获返回值
        result = await stock_time_trade_dao.save_stock_daily_basic_history_by_trade_date(start_date=this_monday, end_date=this_friday)
        # [代码已修复] 在 print 语句中使用捕获的返回值
        print(f"保存 本周 股票重要的基本面指标 完成。result: {result}")
    async_to_sync(main)()

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
        # print(f"保存 {ts_code} （本周）每日筹码分布 数据完成。 result: {result} ")
    async_to_sync(main)()


@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_this_week_task', queue='celery')
@with_cache_manager
def save_cyq_data_this_week_task(cache_manager=None):
    """
    【无绑定版】
    调度器任务：获取股票列表并分派本周筹码数据获取子任务。
    """
    logger.info(f"任务启动: save_cyq_data_this_week_task (调度器模式)")
    all_stocks = []
    # [代码已修复] 修复变量名错误，使用由装饰器注入的 cache_manager
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def get_stocks_main():
        nonlocal all_stocks
        stocks_list = await stock_basic_dao.get_stock_list()
        if stocks_list:
            all_stocks = stocks_list
    async_to_sync(get_stocks_main)()
    try:
        if not all_stocks:
            logger.warning("未能获取到任何股票，任务终止。")
            return {"status": "skipped", "message": "未能获取到任何股票"}
        this_monday, this_friday = get_this_monday_and_friday()
        logger.info(f"获取到 {len(all_stocks)} 只股票，开始为每只股票分派筹码分布任务...")
        for stock in all_stocks:
            save_cyq_chips_this_week_batch.s(
                ts_code=stock.stock_code, 
                start_date=this_monday, 
                end_date=this_friday
            ).set(queue='SaveHistoryData_TimeTrade').apply_async()
        logger.info(f"所有股票的筹码分布任务已分派完毕。")
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
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_minute_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_minute_data_history_task(cache_manager=None):
    """
    【无绑定版】
    调度器任务：为所有股票分派历史分钟数据获取任务。
    """
    logger.info(f"任务启动: save_stocks_minute_data_history_task (调度器模式)")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    async def main():
        all_stocks = await stock_basic_dao.get_stock_list()
        if not all_stocks:
            logger.warning("未找到任何股票代码，跳过任务")
            return {"status": "skipped", "message": "未找到任何股票代码"}
        all_stock_codes = [stock.stock_code for stock in all_stocks]
        total_codes_count = len(all_stock_codes)
        # [代码已修复] 在异步函数内部初始化计数器
        total_dispatched_stocks = 0
        logger.info(f"准备为 {total_codes_count} 个股票分派历史分钟数据任务...")
        for stock_code in all_stock_codes:
            for time_level in ["1", "5", "15", "30", "60"]:
                save_minute_data_history_batch.s(stock_code=stock_code, time_level=time_level).apply_async()
            total_dispatched_stocks += 1
        logger.info(f"任务结束: save_stocks_minute_data_history_task (调度器模式) - 共为 {total_dispatched_stocks} 个股票分派了任务。")
        return {"status": "success", "dispatched_stocks": total_dispatched_stocks}
    async_to_sync(main)()

#  ================ 日线数据任务（历史） ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_day_data_history_batch(stock_codes: List[str], cache_manager=None):
    """
    从Tushare批量获取日线交易数据并保存到数据库（异步并发处理）
    """
    stock_time_trade_dao = StockTimeTradeDAO(cache_manager)
    if not stock_codes:
        logger.info("收到空的股票代码列表，任务结束")
        return {"processed": 0, "success": 0, "errors": 0}
    logger.info(f"开始处理包含 {len(stock_codes)} 个股票的 历史(日线)数据任务...")
    async def main():
        return_info = await stock_time_trade_dao.save_daily_time_trade_history_by_stock_codes(stock_codes)
        print(f"完成 {len(stock_codes)} 个股票的日线数据保存，{return_info}。")
        time.sleep(5)
    # [代码已修复] 移除错误的递归调用，只执行一次 main 函数
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_day_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_day_data_history_task(batch_size: int = 13, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派历史日线数据获取任务。
    """
    logger.info(f"任务启动: save_stocks_day_data_history_task (调度器模式) - 批次大小: {batch_size}")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    all_stocks = []
    async def main():
        nonlocal all_stocks
        all_stocks = await stock_basic_dao.get_stock_list()
    async_to_sync(main)()
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    total_codes_count = len(all_stock_codes)
    # [代码已修复] 初始化计数器
    total_dispatched_batches = 0
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
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
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_daily_basic_data_history_task(batch_size: int = 16, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派历史每日基本信息数据获取任务。
    """
    logger.info(f"任务启动: save_stocks_daily_basic_data_history_task (调度器模式) - 批次大小: {batch_size}")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    all_stocks = []
    async def main():
        nonlocal all_stocks
        all_stocks = await stock_basic_dao.get_stock_list()
    async_to_sync(main)()
    if not all_stocks:
        logger.warning("未找到任何股票代码，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    all_stock_codes = [stock.stock_code for stock in all_stocks]
    total_codes_count = len(all_stock_codes)
    # [代码已修复] 初始化计数器
    total_dispatched_batches = 0
    logger.info(f"准备为 {total_codes_count} 个股票分派批量任务...")
    for i in range(0, total_codes_count, batch_size):
        batch_codes = all_stock_codes[i:i + batch_size]
        if batch_codes:
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
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_week_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_week_data_history_task(batch_size: int = 15, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派历史周线数据获取任务。
    """
    logger.info(f"任务启动: save_stocks_week_data_history_task (调度器模式) - 批次大小: {batch_size}")
    favorite_codes = []
    non_favorite_codes = []
    stock_basic_dao = StockBasicInfoDao(cache_manager)
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
    logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
    for i in range(0, total_favorite_stocks, batch_size):
        batch = favorite_codes[i:i + batch_size]
        if batch:
            save_week_data_history_batch.s(batch).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
    for i in range(0, total_non_favorite_stocks, batch_size):
        batch = non_favorite_codes[i:i + batch_size]
        if batch:
            save_week_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
    logger.info(f"任务结束: save_stocks_week_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
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
        result = await stock_time_trade_dao.save_monthly_time_trade(stock_codes)
        logger.info(f"历史(月线)数据任务 结果：{result}")
    async_to_sync(main)()

# --- 修改后的调度器任务 ---
@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_stocks_month_data_history_task', queue='celery')
@with_cache_manager
def save_stocks_month_data_history_task(batch_size: int = 50, cache_manager=None):
    """
    【无绑定版】
    调度器任务：分派历史月线数据获取任务。
    """
    logger.info(f"任务启动: save_stocks_month_data_history_task (调度器模式) - 批次大小: {batch_size}")
    favorite_codes = []
    non_favorite_codes = []
    stock_basic_dao = StockBasicInfoDao(cache_manager)
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
    logger.info(f"准备为 {total_favorite_stocks} 个自选股分派批量任务...")
    for i in range(0, total_favorite_stocks, batch_size):
        batch = favorite_codes[i:i + batch_size]
        if batch:
            save_month_data_history_batch.s(batch).set().apply_async()
            total_dispatched_batches += 1
    logger.info(f"准备为 {total_non_favorite_stocks} 个非自选股分派批量任务...")
    for i in range(0, total_non_favorite_stocks, batch_size):
        batch = non_favorite_codes[i:i + batch_size]
        if batch:
            save_month_data_history_batch.s(batch).set(queue=STOCKS_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
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

@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.save_cyq_data_history_task', queue='celery')
@with_cache_manager
def save_cyq_data_history_task(cache_manager=None):
    """
    【无绑定版】
    调度器任务：为所有股票分派历史筹码数据获取任务。
    """
    logger.info(f"任务启动: save_cyq_data_history_task (调度器模式)")
    stock_basic_dao = StockBasicInfoDao(cache_manager)
    today_date = timezone.now().date()
    all_stocks = []
    async def main():
        nonlocal all_stocks
        all_stocks = await stock_basic_dao.get_stock_list()
    async_to_sync(main)()
    if not all_stocks:
        logger.warning("未能获取到任何股票，跳过任务")
        return {"status": "skipped", "message": "未找到任何股票代码"}
    logger.info(f"获取到 {len(all_stocks)} 只股票，开始为每只股票分派任务...")
    for stock in all_stocks:
        save_cyq_chips_history_batch.s(ts_code=stock.stock_code, start_date=None, end_date=today_date).apply_async()
        save_cyq_perf_for_stock_batch.s(ts_code=stock.stock_code, start_date=None, end_date=today_date).apply_async()
    logger.info(f"所有 {len(all_stocks)} 只股票的筹码分布和筹码胜率任务已全部分派完毕。")
    return {"status": "success", "dispatched_stocks": len(all_stocks)}

@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.refetch_incomplete_cyq_chips', queue='celery')
def refetch_incomplete_cyq_chips(record_threshold=1000):
    """
    【无绑定版】
    Celery 定时任务：查找并重新拉取历史筹码数据不完整的股票。
    """
    from django.db.models import Count
    print(f"开始执行任务：检查并重新拉取记录数少于 {record_threshold} 条的筹码分布数据...")
    logger.info(f"开始执行任务：检查并重新拉取记录数少于 {record_threshold} 条的筹码分布数据...")
    chips_models = [
        StockCyqChipsCY, StockCyqChipsSZ, StockCyqChipsKC, StockCyqChipsSH, StockCyqChipsBJ,
    ]
    stocks_to_refetch = set()
    try:
        for model in chips_models:
            table_name = model._meta.db_table
            print(f"正在检查表: {table_name}...")
            logger.info(f"正在检查表: {table_name}...")
            incomplete_stocks_query = model.objects.values('stock__stock_code') \
                .annotate(record_count=Count('id')) \
                .filter(record_count__lt=record_threshold) \
                .values_list('stock__stock_code', flat=True)
            found_codes = list(incomplete_stocks_query)
            if found_codes:
                count = len(found_codes)
                print(f"在表 {table_name} 中发现 {count} 只股票记录不完整。")
                logger.info(f"在表 {table_name} 中发现 {count} 只股票记录不完整。")
                stocks_to_refetch.update(found_codes)
            else:
                print(f"在表 {table_name} 中未发现记录不完整的股票。")
                logger.info(f"在表 {table_name} 中未发现记录不完整的股票。")
        total_count = len(stocks_to_refetch)
        print(f"检查完成。总共发现 {total_count} 只股票需要重新拉取数据。")
        logger.info(f"检查完成。总共发现 {total_count} 只股票需要重新拉取数据。")
        if total_count > 0:
            print("开始分发数据拉取任务...")
            for stock_code in stocks_to_refetch:
                print(f"为股票 {stock_code} 创建数据拉取任务。")
                save_cyq_chips_history_batch.delay(ts_code=stock_code, start_date=None, end_date=None)
            print(f"已为 {total_count} 只股票成功分发任务。")
            logger.info(f"已为 {total_count} 只股票成功分发任务。")
        return f"任务完成。共检查 {len(chips_models)} 个表，为 {total_count} 只股票分发了更新任务。"
    except Exception as e:
        print(f"执行任务 refetch_incomplete_cyq_chips 时发生严重错误: {e}")
        logger.error(f"执行任务 refetch_incomplete_cyq_chips 时发生严重错误: {e}", exc_info=True)
        raise

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

@celery_app.task(name='tasks.tushare.stock_time_trade_tasks.cleanup_non_trade_day_data', queue='clean_data')
def cleanup_non_trade_day_data():
    """
    【无绑定版】
    一个Celery任务，用于清理所有股票数据表中在非交易日产生的无效数据。
    """
    print("开始执行【清理非交易日数据】任务...")
    total_deleted_count = 0
    all_non_trade_dates = set(
        TradeCalendar.objects.filter(is_open=False).values_list('cal_date', flat=True)
    )
    if not all_non_trade_dates:
        print("警告：交易日历中未找到任何休市日期记录，任务提前终止。请检查TradeCalendar数据。")
        return "任务终止，未找到休市日期。"
    print(f"已从交易日历加载 {len(all_non_trade_dates)} 个休市日期。")
    for model in DATA_MODELS_TO_CLEAN:
        model_name = model._meta.verbose_name_plural or model.__name__
        table_name = model._meta.db_table
        print(f"\n--- 正在处理模型: {model_name} (数据表: {table_name}) ---")
        field_object = model._meta.get_field('trade_time')
        is_datetime_field = isinstance(field_object, models.DateTimeField)
        print(f"检测到 'trade_time' 字段类型为: {'DateTimeField' if is_datetime_field else 'DateField'}")
        if is_datetime_field:
            dates_in_table = set(
                model.objects.annotate(trade_date=TruncDate('trade_time'))
                .values_list('trade_date', flat=True)
                .distinct()
            )
        else:
            dates_in_table = set(
                model.objects.values_list('trade_time', flat=True).distinct()
            )
        if not dates_in_table:
            print(f"数据表 {table_name} 中没有数据，跳过。")
            continue
        print(f"在表 {table_name} 中发现 {len(dates_in_table)} 个唯一日期。")
        dates_to_delete = dates_in_table & all_non_trade_dates
        if not dates_to_delete:
            print(f"在表 {table_name} 中未发现任何非交易日数据，无需清理。")
            continue
        print(f"在表 {table_name} 中发现 {len(dates_to_delete)} 个非交易日需要清理")
        if is_datetime_field:
            deleted_info = model.objects.filter(trade_time__date__in=dates_to_delete).delete()
        else:
            deleted_info = model.objects.filter(trade_time__in=dates_to_delete).delete()
        deleted_count = deleted_info[0]
        total_deleted_count += deleted_count
        print(f"成功从 {table_name} 中删除了 {deleted_count} 条非交易日记录。")
    print(f"\n--- 任务执行完毕 ---")
    print(f"总共删除了 {total_deleted_count} 条非交易日记录。")
    return f"任务完成，总共删除 {total_deleted_count} 条记录。"










