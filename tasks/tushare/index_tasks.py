# tasks/tushare/index_tasks.py
import asyncio
import logging
import datetime
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
from celery.utils.log import get_task_logger
from dao_manager.tushare_daos import fund_flow_dao
from dao_manager.tushare_daos import index_basic_dao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao

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

#  ================ 交易日历数据 ================
@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_trade_cal', queue='SaveData_TimeTrade')
def save_trade_cal(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    print(f"开始处理 交易日历数据...")
    # 在任务开始时创建一次 DAO 实例
    index_info_dao = IndexBasicDAO()
    try:
        # 异步获取数据并保存
        asyncio.run(index_info_dao.save_trade_cal())
        print("任务完成 - 交易日历数据")
    except Exception as e:
        logger.error(f"执行 交易日历数据 任务时发生意外错误: {e}", exc_info=True)

#  ================ 指数基本信息 ================
@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_infos', queue='SaveData_TimeTrade')
def save_index_infos(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    try:
        # 异步获取数据并保存
        print(f"开始处理 指数基本信息...")
        asyncio.run(index_basic_dao.save_indexs())
        print("任务完成 - 指数基本信息")
        # asyncio.run(index_basic_dao.save_index_weight_monthly())
        # print("任务完成 - 指数成分和权重")
    except Exception as e:
        logger.error(f"执行 指数基本信息 任务时发生意外错误: {e}", exc_info=True)

#  ================ 指数每日指标 ================
@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_basic_today', queue='SaveData_TimeTrade')
def save_index_daily_basic_today(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    try:
        asyncio.run(index_basic_dao.save_index_daily_basic_today())
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)


#  ================ 指数历史指标 ================
@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_basic_history', queue='SaveData_TimeTrade')
def save_index_daily_basic_history(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    try:
        asyncio.run(index_basic_dao.save_index_daily_basic_history())
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)








