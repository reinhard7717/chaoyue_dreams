# tasks\tushare\industry_tasks.py

import datetime
import logging
import asyncio
from asgiref.sync import async_to_sync
from utils.task_helpers import with_cache_manager
# 假设 StockBasicInfoDao 存在且可用
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.industry_dao import IndustryDao

# 假设 celery 实例存在且可用
from chaoyue_dreams.celery import app as celery_app
from utils.cache_manager import CacheManager

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

@celery_app.task(name='tasks.tushare.industry_tasks.save_ths_index_member_task', queue='SaveData_TimeTrade', rate_limit='180/m')
@with_cache_manager
def save_ths_index_member_task(cache_manager=None):
    """
    【无绑定版】
    保存同花顺概念板块成分。
    """
    industry_dao = IndustryDao(cache_manager)
    async def main():
        logger.info(f"开始获取同花顺概念板块成分...")
        result_member = await industry_dao.save_ths_index_member()
        logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")
    async_to_sync(main)()

# 每日任务：同花顺板块 & 指数行情
@celery_app.task(name='tasks.tushare.industry_tasks.save_ths_index_today_task', queue='SaveData_TimeTrade')
@with_cache_manager
def save_ths_index_today_task(cache_manager=None):
    """
    【无绑定版】
    Celery调度器任务：保存同花顺板块相关数据（指数列表、成分、当日行情）。
    """
    print(f"开始执行Celery任务: save_ths_index_today_task")
    industry_dao = IndustryDao(cache_manager)
    today = datetime.date.today()
    async def main():
        print("任务步骤1: 开始获取同花顺板块指数...")
        logger.info("任务步骤1: 开始获取同花顺板块指数...")
        result_list = await industry_dao.save_ths_index_list()
        print(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")
        logger.info(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")
        print("任务步骤2: 开始获取同花顺概念板块成分...")
        logger.info("任务步骤2: 开始获取同花顺概念板块成分...")
        result_member = await industry_dao.save_ths_index_member()
        print(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")
        logger.info(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")
        print(f"任务步骤3: 开始获取 {today} 同花顺板块指数行情...")
        logger.info(f"任务步骤3: 开始获取 {today} 同花顺板块指数行情...")
        result_daily = await industry_dao.save_ths_index_daily_by_trade_date(trade_date=today)
        print(f"任务步骤3: 保存 {today} 同花顺板块指数行情完成，结果：{result_daily}")
        logger.info(f"任务步骤3: 保存 {today} 同花顺板块指数行情完成，结果：{result_daily}")
    async_to_sync(main)()
    print(f"Celery任务: save_ths_index_today_task 全部步骤执行成功。")
    logger.info(f"Celery任务: save_ths_index_today_task 全部步骤执行成功。")
    return "所有同花顺板块相关数据保存成功。"


@celery_app.task(name='tasks.tushare.industry_tasks.save_ths_index_yesterday_task', queue='SaveData_TimeTrade')
@with_cache_manager
def save_ths_index_yesterday_task(cache_manager=None):
    """
    【无绑定版】
    保存昨日同花顺板块相关数据（指数列表、成分、昨日行情）。
    """
    industry_dao = IndustryDao(cache_manager)
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    async def main():
        logger.info(f"开始获取同花顺板块指数...")
        result_list = await industry_dao.save_ths_index_list()
        logger.info(f"保存同花顺概念和行业指数， 结果：{result_list}")
        logger.info(f"开始获取同花顺概念板块成分...")
        result_member = await industry_dao.save_ths_index_member()
        logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")
        logger.info(f"开始获取同花顺板块指数行情...")
        result_daily = await industry_dao.save_ths_index_daily_by_trade_date(trade_date=yesterday)
        logger.info(f"保存 {yesterday} 同花顺板块指数行情， 结果：{result_daily}")
    async_to_sync(main)()


# 任务：同花顺板块 & 指数行情
@celery_app.task(name='tasks.tushare.industry_tasks.save_ths_index_history_task', queue='SaveData_TimeTrade')
@with_cache_manager
def save_ths_index_history_task(cache_manager=None):
    """
    【无绑定版 & 性能优化版】
    保存历史同花顺板块相关数据（指数列表、成分、历史行情）。
    """
    industry_dao = IndustryDao(cache_manager)
    async def main():
        logger.info("任务步骤1: 开始获取同花顺板块指数...")
        result_list = await industry_dao.save_ths_index_list()
        logger.info(f"任务步骤1: 保存同花顺概念和行业指数完成，结果：{result_list}")
        logger.info("任务步骤2: 开始获取同花顺概念板块成分...")
        result_member = await industry_dao.save_ths_index_member()
        logger.info(f"任务步骤2: 保存同花顺概念板块成分完成，结果：{result_member}")
        # --- 性能优化：避免N+1查询 ---
        # [代码已修改] 不再循环单日查询，而是获取日期范围后进行一次性批量查询
        logger.info("任务步骤3: 开始获取历史同花顺板块指数行情...")
        index_dao = IndexBasicDAO(cache_manager)
        trade_days = await index_dao.get_last_n_trade_cal_open()
        if trade_days:
            # 获取日期范围的开始和结束日期
            start_date = min(trade_days)
            end_date = max(trade_days)
            logger.info(f"准备获取 {start_date} 到 {end_date} 的历史行情...")
            # 调用支持日期范围的批量方法
            result_daily = await industry_dao.save_ths_index_daily_history(start_date=start_date, end_date=end_date)
            logger.info(f"保存 {start_date} 到 {end_date} 的同花顺板块指数行情完成，结果：{result_daily}")
        else:
            logger.warning("未能获取到任何交易日，跳过历史行情保存步骤。")
    async_to_sync(main)()


@celery_app.task(name='tasks.tushare.industry_tasks.save_ths_index_this_week_task', queue='SaveData_TimeTrade')
@with_cache_manager
def save_ths_index_this_week_task(cache_manager=None):
    """
    【无绑定版】
    保存本周同花顺板块相关数据（指数列表、成分、本周行情）。
    """
    industry_dao = IndustryDao(cache_manager)
    this_monday, this_friday = get_this_monday_and_friday()
    async def main():
        logger.info(f"开始获取同花顺板块指数...")
        result_list = await industry_dao.save_ths_index_list()
        logger.info(f"保存同花顺概念和行业指数， 结果：{result_list}")
        logger.info(f"开始获取同花顺概念板块成分...")
        result_member = await industry_dao.save_ths_index_member()
        logger.info(f"保存同花顺概念板块成分， 结果：{result_member}")
        logger.info(f"开始获取本周同花顺板块指数行情...")
        result_daily = await industry_dao.save_ths_index_daily_history(start_date=this_monday, end_date=this_friday)
        logger.info(f"保存 {this_monday} - {this_friday} 同花顺板块指数行情， 结果：{result_daily}")
    async_to_sync(main)()








