# tasks/tushare/fund_flow_tasks.py
import asyncio
import logging
import datetime
import time
from django.db.models import Q
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
# from celery import chain # 不再需要 chain，除非有后续步骤
from celery.utils.log import get_task_logger
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO

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

    return sorted(favorite_stock_codes_list), sorted(non_favorite_stock_codes)

#  ================ （当日）个股日级资金流向数据 （三种渠道） ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_today', queue=STOCKS_SAVE_API_DATA_QUEUE)
def save_fund_flow_daily_data_today(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理（当日）日级资金流向数据 （三种渠道）...")
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_today_fund_flow_daily_data())
        asyncio.run(fund_flow_dao.save_today_fund_flow_daily_ths_data())
        asyncio.run(fund_flow_dao.save_today_fund_flow_daily_dc_data())
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

#  ================ （历史）日级资金流向数据（三种渠道） ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_batch')
def save_fund_flow_daily_data_history_batch(self, trade_date_str: str):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理 {trade_date_str} 的 （历史）日级资金流向数据 （三种渠道）...")
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        # 异步获取数据并保存
        result = asyncio.run(fund_flow_dao.save_history_fund_flow_daily_data_by_trade_date(trade_date_str))
        print(f"{trade_date_str} 日级资金流向数据 保存完成。{result}")
        # 同花顺
        result = asyncio.run(fund_flow_dao.save_history_fund_flow_daily_ths_data_by_trade_date(trade_date_str))
        print(f"{trade_date_str} 日级资金流向数据(同花顺) 保存完成。{result}")
        # 东方财富
        result = asyncio.run(fund_flow_dao.save_history_fund_flow_daily_dc_data_trade_date(trade_date_str))
        print(f"{trade_date_str} 日级资金流向数据(东方财富) 保存完成。{result}")
        time.sleep(2)
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# --- 修改后的调度器任务 ---
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_task')
def save_fund_flow_daily_data_history_task(self): 
    """
    调度器任务：
    1. 获取自选股和非自选股代码。
    2. 将代码分成批次。
    3. 为每个批次分派 save_fund_flow_daily_data_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_history_task (调度器模式) - 获取股票列表并分派批量任务")
    try:
        total_dispatched_batches = 0
        index_basic_dao = IndexBasicDAO()
        trade_days_list = asyncio.run(index_basic_dao.get_last_n_trade_cal_open())
        for trade_date in trade_days_list:
            trade_date_str = trade_date.strftime('%Y%m%d')
            logger.info(f"创建自选股批次任务 (抓取日期: {trade_date_str})...")
            # 使用新的批量任务，并指定队列
            save_fund_flow_daily_data_history_batch.s(trade_date_str).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
        logger.info(f"已分派了 {total_dispatched_batches} 个批次任务。")
        logger.info(f"任务结束: save_fund_flow_daily_data_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

# ================ （当日）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_today', queue=STOCKS_SAVE_API_DATA_QUEUE)
def save_fund_flow_daily_data_ths_today(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理（当日）板块、行业资金流向数据 - 同花顺...")
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_today_fund_flow_daily_ths_data())
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# ================ （历史）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_history_batch')
def save_fund_flow_daily_data_ths_history_batch(self, trade_date: datetime.date):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理 {trade_date} 的 （历史）板块资金流向数据 - 同花顺...")
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_history_fund_flow_cnt_ths_data(trade_date))
        asyncio.run(fund_flow_dao.save_history_fund_flow_industry_ths_data(trade_date))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_history_task')
def save_fund_flow_daily_data_ths_history_task(self):
    """
    调度器任务：
    1. 获取最近60天的A股交易日。
    2. 为每个交易日分派 save_fund_flow_daily_data_ths_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
 
    logger.info(f"任务启动: save_fund_flow_daily_data_ths_history_task (调度器模式)")
    try:
        index_info_dao = IndexBasicDAO()
        trade_days = asyncio.run(index_info_dao.get_last_n_trade_cal_open(n=1500))
        total_dispatched_batches = 0
        for cal_date in trade_days:
            # cal_date格式为'YYYYMMDD'，转为date对象
            save_fund_flow_daily_data_ths_history_batch.s(cal_date).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1

        logger.info(f"任务结束: save_fund_flow_daily_data_ths_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_ths_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}










