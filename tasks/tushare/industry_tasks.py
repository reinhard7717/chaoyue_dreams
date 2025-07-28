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
    """
    Celery调度器任务：保存同花顺板块相关数据（指数列表、成分、当日行情）。
    优化点：
    1. 只创建一次事件循环，所有异步操作在同一循环内顺序执行。
    2. 业务逻辑顺序不变，异常捕获和日志记录更健壮。
    """
    print(f"开始执行Celery任务: save_ths_index_today_task, Task ID: {self.request.id}")
    try:
        industry_dao = IndustryDao()
        today = datetime.date.today()

        async def main_logic():
            # 步骤1：保存同花顺板块指数列表
            print("任务步骤1: 开始获取同花顺板块指数...")
            logger.info("任务步骤1: 开始获取同花顺板块指数...")
            result_list = await industry_dao.save_ths_index_list()
            print(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")
            logger.info(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")

            # 步骤2：保存同花顺板块成分
            print("任务步骤2: 开始获取同花顺概念板块成分...")
            logger.info("任务步骤2: 开始获取同花顺概念板块成分...")
            result_member = await industry_dao.save_ths_index_member()
            print(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")
            logger.info(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")

            # 步骤3：保存当日同花顺板块指数行情
            print(f"任务步骤3: 开始获取 {today} 同花顺板块指数行情...")
            logger.info(f"任务步骤3: 开始获取 {today} 同花顺板块指数行情...")
            result_daily = await industry_dao.save_ths_index_daily_by_trade_date(trade_date=today)
            print(f"任务步骤3: 保存 {today} 同花顺板块指数行情完成，结果：{result_daily}")
            logger.info(f"任务步骤3: 保存 {today} 同花顺板块指数行情完成，结果：{result_daily}")

        # 只创建一次事件循环，执行所有异步步骤
        asyncio.run(main_logic())

        print(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 全部步骤执行成功。")
        logger.info(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 全部步骤执行成功。")
        return "所有同花顺板块相关数据保存成功。"

    except Exception as e:
        print(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 执行失败: {e}")
        logger.error(f"Celery任务: save_ths_index_today_task, Task ID: {self.request.id} 执行失败: {e}", exc_info=True)
        raise  # 保证Celery能正确标记任务失败

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








