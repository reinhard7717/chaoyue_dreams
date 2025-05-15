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

@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_history_slice', queue='SaveData_TimeTrade')
def save_index_daily_history_slice(self, index_codes_slice: List[str]):
    """
    执行保存单个指数切片的历史日级指标数据到数据库
    Args:
        index_codes_slice: 指数代码列表切片
    """
    # 在任务开始时创建 DAO 实例
    index_basic_dao = IndexBasicDAO() # 修改：在执行任务内部创建DAO实例
    try:
        print(f"执行任务 - 处理指数切片，包含 {len(index_codes_slice)} 个指数代码: {index_codes_slice[:5]}...") # 修改：打印正在处理的切片信息

        # 根据指数代码列表获取 IndexInfo 对象列表
        # !!! 注意：这里需要 IndexBasicDAO 提供一个方法来根据代码列表获取 IndexInfo 对象
        # 假设 IndexBasicDAO 有一个异步方法 get_index_infos_by_codes(codes: List[str]) -> List[IndexInfo]
        # 或者使用 Django ORM 直接查询 (需要在异步环境中)
        async def fetch_index_infos_async():
             # Placeholder: Replace with actual DAO method or ORM query
             # Example using a hypothetical async DAO method:
             # return await index_basic_dao.get_index_infos_by_codes(index_codes_slice)
             # Example fetching all and filtering (less efficient but works for demo if get_indexs_by_publisher is async):
             all_indices = await index_basic_dao.get_indexs_by_publisher(publisher="中证指数有限公司") # 假设这个方法是异步的
             return [idx for idx in all_indices if idx.index_code in index_codes_slice]

        # 在同步任务中运行异步代码获取 IndexInfo 对象
        fetched_index_infos = asyncio.run(fetch_index_infos_async()) # 修改：在同步任务中运行异步代码获取 IndexInfo 对象

        if not fetched_index_infos:
            print(f"未找到对应的 IndexInfo 对象，跳过处理切片: {index_codes_slice}")
            return

        # 调用 IndexBasicDAO 中的异步方法来保存数据
        # 这个方法就是用户提供的 save_index_daily_history
        # 注意：save_index_daily_history 方法需要能够接受一个 IndexInfo 列表作为参数
        asyncio.run(index_basic_dao.save_index_daily_history(indexs=fetched_index_infos)) # 修改：调用DAO方法处理获取到的IndexInfo列表

        print(f"任务完成 - 指数切片处理完成，包含 {len(index_codes_slice)} 个指数代码。") # 修改：打印切片处理完成信息
    except Exception as e:
        logger.error(f"执行 指数切片任务时发生意外错误 (切片: {index_codes_slice[:5]}...): {e}", exc_info=True) # 修改：记录切片任务错误

@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_history_task', queue='celery')
def save_index_daily_history_task(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（调度任务）
    将指数列表切片后分配给执行任务处理。
    """
    # 在任务开始时创建一次 DAO 实例 (仅用于获取所有指数代码)
    index_basic_dao = IndexBasicDAO()
    slice_size = 500 # 定义切片大小，每500个指数一个切片
    try:
        print(f"开始调度 指数每日指标任务...")
        # 获取所有需要处理的指数列表，只需要代码
        print("正在获取所有指数代码列表...")
        # 注意：这里需要在同步任务中运行异步方法获取所有指数
        async def get_all_index_codes_async():
             all_indices = await index_basic_dao.get_index_list() # 假设这个方法是异步的
             return [idx.index_code for idx in all_indices]

        # 在同步任务中运行异步代码获取所有指数代码
        all_index_codes = asyncio.run(get_all_index_codes_async()) # 修改：先获取所有指数代码列表
        print(f"共获取到 {len(all_index_codes)} 个指数代码，将按每 {slice_size} 个进行切片并分配任务。")

        if not all_index_codes:
            print("未获取到任何指数代码，调度任务结束。")
            return

        # 对指数代码列表进行切片并分配执行任务
        # 导入执行任务函数
        # from . import save_index_basic_daily_slice # 假设执行任务函数在同一个文件或可导入路径
        for i in range(0, len(all_index_codes), slice_size): # 修改：循环遍历指数代码切片
            index_codes_slice = all_index_codes[i:i + slice_size] # 修改：获取当前指数代码切片
            print(f"调度任务 - 分配第 {i // slice_size + 1} 个切片任务 (索引 {i} 到 {min(i + slice_size, len(all_index_codes)) - 1})，包含 {len(index_codes_slice)} 个指数代码...") # 修改：打印切片分配信息
            # 分配执行任务
            save_index_daily_history_slice.delay(index_codes_slice) # 修改：调用执行任务并传递切片参数

        print("调度任务完成 - 所有指数切片任务已分配。") # 修改：打印调度任务完成信息
    except Exception as e:
        logger.error(f"执行 指数每日指标调度任务时发生意外错误: {e}", exc_info=True) # 修改：记录调度任务错误


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








