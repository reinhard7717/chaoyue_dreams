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

# 获取本周一和本周五的日期
def get_this_monday_and_friday():
    """获取本周一和本周五的日期"""
    today = datetime.date.today()
    this_monday = today - datetime.timedelta(days=today.weekday())
    this_friday = this_monday + datetime.timedelta(days=4)
    return this_monday, this_friday


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
@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_today_task', queue='SaveData_TimeTrade')
def save_index_daily_today_task(self):
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

@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_this_week', queue='SaveData_TimeTrade')
def save_index_daily_this_week_task(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    this_monday, this_friday = get_this_monday_and_friday()
    try:
        asyncio.run(index_basic_dao.save_index_daily_history(start_date=this_monday, end_date=this_friday))
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)


@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_history', queue='SaveData_TimeTrade')
def save_index_daily_history_task(self, slice_size: int = 500):# 定义切片大小，每500个指数一个切片
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    修改为每500个index切片分配任务。
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    try:
        print(f"开始处理 指数每日指标...")
        # 获取所有需要处理的指数列表
        # 注意：这里需要在同步任务中运行异步方法获取所有指数
        print("正在获取所有指数列表...")
        all_indices = asyncio.run(index_basic_dao.get_indexs_by_publisher(publisher="中证指数有限公司")) # 修改：先获取所有指数列表
        print(f"共获取到 {len(all_indices)} 个指数，将按每 {slice_size} 个进行切片处理。")
        # 对指数列表进行切片并逐个处理
        for i in range(0, len(all_indices), slice_size): # 修改：循环遍历指数切片
            index_slice = all_indices[i:i + slice_size] # 修改：获取当前切片
            print(f"开始处理第 {i // slice_size + 1} 个切片 (索引 {i} 到 {min(i + slice_size, len(all_indices)) - 1})，包含 {len(index_slice)} 个指数...") # 修改：打印切片信息
            # 为每个切片调用保存数据的异步方法
            # 注意：这里需要在同步任务中运行异步方法处理切片数据
            asyncio.run(index_basic_dao.save_index_daily_history(indexs=index_slice)) # 修改：对当前切片调用保存方法
            print(f"第 {i // slice_size + 1} 个切片处理完成。") # 修改：打印切片处理完成信息

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
        print(f"开始处理 指数每日指标...")
        asyncio.run(index_basic_dao.save_index_daily_basic_history())
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)








