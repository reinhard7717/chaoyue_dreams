# tasks\tushare\industry_tasks.py

import datetime
import os
import logging
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
import pandas as pd
import asyncio
# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app

logger = logging.getLogger("tasks")

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

@celery_app.task(bind=True, name='tasks.tushare.industry_tasks.save_ths_index_member_task', queue='SaveData_TimeTrade', rate_limit='180/m')
def save_ths_index_member_task(self):
    industry_dao = IndustryDao()
    logger.info(f"开始获取同花顺概念板块成分...")
    result_member = asyncio.run(industry_dao.save_ths_index_member())
    logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")

# 每日任务：同花顺板块 & 指数行情
@celery_app.task(bind=True, name='tasks.tushare.industry_tasks.save_ths_index_today_task', queue='SaveData_TimeTrade')
def save_ths_index_today_task(self):
    # --- 代码修改开始 ---
    """
    修改说明:
    原始实现中，对每个异步(async)函数都使用了一次 asyncio.run()。
    这会导致每次调用都创建和销毁一个新的事件循环，效率较低。

    优化后的实现将所有需要按顺序执行的异步操作封装在一个主异步函数 (main_logic) 中，
    然后只调用一次 asyncio.run() 来执行这个主函数。
    这样做的好处是：
    1. 效率更高：所有异步任务在同一个事件循环中运行，避免了重复创建和销毁循环的开销。
    2. 代码更简洁：将异步逻辑集中管理，使主任务函数更清晰。
    3. 遵循最佳实践：这是从同步代码中调用多个异步操作的标准模式。

    同时，增加了完整的 try...except 错误处理块，以确保任务的健壮性，
    在发生任何步骤的错误时都能正确记录日志并报告任务失败。
    业务逻辑和执行顺序保持不变。
    """
    # 打印任务开始信息，self.request.id 是Celery提供的唯一任务ID
    print(f"开始执行Celery任务: save_ths_index_today_task, Task ID: {self.request.id}")
    try:
        # 1. 实例化DAO，这个操作在异步流程外进行
        industry_dao = IndustryDao()
        today = datetime.date.today()

        # 2. 定义一个主异步函数，用于按顺序编排所有异步调用
        async def main_logic():
            # 步骤一：保存同花顺板块指数列表 (顺序不变)
            print("任务步骤1: 开始获取同花顺板块指数...")
            logger.info("任务步骤1: 开始获取同花顺板块指数...")
            result_list = await industry_dao.save_ths_index_list()
            print(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")
            logger.info(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")

            # 步骤二：保存同花顺板块成分 (顺序不变)
            print("任务步骤2: 开始获取同花顺概念板块成分...")
            logger.info("任务步骤2: 开始获取同花顺概念板块成分...")
            result_member = await industry_dao.save_ths_index_member()
            print(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")
            logger.info(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")

            # 步骤三：保存当日的同花顺板块指数行情 (顺序不变)
            print(f"任务步骤3: 开始获取 {today} 同花顺板块指数行情...")
            logger.info(f"任务步骤3: 开始获取 {today} 同花顺板块指数行情...")
            result_daily = await industry_dao.save_ths_index_daily_by_trade_date(trade_date=today)
            print(f"任务步骤3: 保存 {today} 同花顺板块指数行情完成，结果：{result_daily}")
            logger.info(f"任务步骤3: 保存 {today} 同花顺板块指数行情完成，结果：{result_daily}")

        # 3. 使用一次 asyncio.run() 来统一执行整个异步逻辑流程
        asyncio.run(main_logic())

        print(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 全部步骤执行成功。")
        logger.info(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 全部步骤执行成功。")
        return "所有同花顺板块相关数据保存成功。"

    except Exception as e:
        # 捕获在任何步骤中发生的异常
        print(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 执行失败: {e}")
        logger.error(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 执行失败: {e}", exc_info=True)
        # 重新抛出异常，以便Celery将任务状态标记为FAILURE
        raise

@celery_app.task(bind=True, name='tasks.tushare.industry_tasks.save_ths_index_yesterday_task', queue='SaveData_TimeTrade')
def save_ths_index_yesterday_task(self):
    industry_dao = IndustryDao()
    logger.info(f"开始获取同花顺板块指数...")
    result_list = asyncio.run(industry_dao.save_ths_index_list())
    logger.info(f"保存同花顺概念和行业指数， 结果：{result_list}")

    logger.info(f"开始获取同花顺概念板块成分...")
    result_member = asyncio.run(industry_dao.save_ths_index_member())
    logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")

    logger.info(f"开始获取同花顺板块指数行情...")
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期时间
    result_daily = asyncio.run(industry_dao.save_ths_index_daily_by_trade_date(trade_date=yesterday))
    logger.info(f"保存 {today} 同花顺板块指数行情， 结果：{result_daily}")

# 任务：同花顺板块 & 指数行情
@celery_app.task(bind=True, name='tasks.tushare.industry_tasks.save_ths_index_history_task', queue='SaveData_TimeTrade')
def save_ths_index_history_task(self):
    industry_dao = IndustryDao()
    logger.info(f"开始获取同花顺板块指数...")
    result_list = asyncio.run(industry_dao.save_ths_index_list())
    logger.info(f"保存同花顺概念和行业指数， 结果：{result_list}")

    logger.info(f"开始获取同花顺概念板块成分...")
    result_member = asyncio.run(industry_dao.save_ths_index_member())
    logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")

    logger.info(f"开始获取同花顺板块指数行情...")
    index_dao = IndexBasicDAO()
    trade_days = asyncio.run(index_dao.get_last_n_trade_cal_open())
    for day in trade_days:
        result_daily = asyncio.run(industry_dao.save_ths_index_daily_by_trade_date(trade_date=day))
        logger.info(f"保存 {day} 同花顺板块指数行情， 结果：{result_daily}")

@celery_app.task(bind=True, name='tasks.tushare.industry_tasks.save_ths_index_this_week_task', queue='SaveData_TimeTrade')
def save_ths_index_this_week_task(self):
    industry_dao = IndustryDao()
    logger.info(f"开始获取同花顺板块指数...")
    result_list = asyncio.run(industry_dao.save_ths_index_list())
    logger.info(f"保存同花顺概念和行业指数， 结果：{result_list}")

    logger.info(f"开始获取同花顺概念板块成分...")
    result_member = asyncio.run(industry_dao.save_ths_index_member())
    logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")

    logger.info(f"开始获取同花顺板块指数行情...")
    this_monday, this_friday = get_this_monday_and_friday()
    result_daily = asyncio.run(industry_dao.save_ths_index_daily_history(start_date=this_monday, end_date=this_friday))
    logger.info(f"保存 {this_monday} - {this_friday} 同花顺板块指数行情， 结果：{result_daily}")








