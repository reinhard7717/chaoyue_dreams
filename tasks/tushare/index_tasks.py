# tasks/tushare/index_tasks.py
import asyncio
from asgiref.sync import async_to_sync
import logging
import datetime
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
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
        async_to_sync(index_info_dao.save_trade_cal)()
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
        async_to_sync(index_basic_dao.save_indexs)()
        print("任务完成 - 指数基本信息")
        # async_to_sync(index_basic_dao.save_index_weight_monthly())
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
        async_to_sync(index_basic_dao.save_index_daily_basic_today)()
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_yesterday_task', queue='SaveData_TimeTrade')
def save_index_daily_yesterday_task(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    try:
        async_to_sync(index_basic_dao.save_index_daily_basic_yesterday)()
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
        async_to_sync(index_basic_dao.save_index_daily_history)(start_date=this_monday, end_date=this_friday)
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)

INDEX_SLICE_SIZE = 100 # 优化：将切片大小从10增加到100，减少任务总数，降低系统开销

@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_history_slice', queue='SaveData_TimeTrade', rate_limit='180/m')
def save_index_daily_history_slice(self, index_codes_slice: List[str]):
    """
    【优化版】执行保存单个指数切片的历史日级指标数据到数据库
    Args:
        index_codes_slice: 指数代码列表切片
    """
    # 在任务开始时创建 DAO 实例
    index_basic_dao = IndexBasicDAO()
    task_id = self.request.id # 代码修改处: 获取任务ID用于日志追踪
    
    try:
        # 代码修改处: 使用Celery logger记录信息，并包含任务ID
        logger.info(f"[{task_id}] 开始执行任务 - 处理指数切片，包含 {len(index_codes_slice)} 个代码: {index_codes_slice[:3]}...")
        
        # 调用DAO方法处理获取到的IndexInfo列表
        async_to_sync(index_basic_dao.save_index_daily_history)(index_codes=index_codes_slice)

        logger.info(f"[{task_id}] 任务成功 - 指数切片处理完成，包含 {len(index_codes_slice)} 个代码。")
    except Exception as e:
        # 代码修改处: 使用Celery logger记录错误，并包含任务ID
        logger.error(f"[{task_id}] 执行指数切片任务时发生错误 (切片: {index_codes_slice[:3]}...): {e}", exc_info=True)
        # 可以选择重新抛出异常，让Celery根据配置进行重试
        # raise self.retry(exc=e, countdown=60)


@celery_app.task(bind=True, name='tasks.tushare.index_tasks.save_index_daily_history_task', queue='celery') # 代码修改处: 任务名修改为 dispatch_... 更清晰
def save_index_daily_history_task(self):
    """
    【优化版】从数据库获取所有指数代码，切片后分发给执行器任务进行处理（调度任务）
    """
    # 在任务开始时创建一次 DAO 实例
    index_basic_dao = IndexBasicDAO()
    task_id = self.request.id
    
    try:
        logger.info(f"[{task_id}] 开始调度 [指数每日指标] 任务...")
        
        # 代码修改处: 直接调用高效的DAO方法获取所有指数代码列表
        # 无需在Celery任务中再嵌套一层 async_to_sync
        logger.info(f"[{task_id}] 正在从数据库获取所有指数代码列表...")
        all_index_codes = async_to_sync(index_basic_dao.get_all_index_codes)()
        
        if not all_index_codes:
            logger.warning(f"[{task_id}] 未获取到任何指数代码，调度任务结束。")
            return

        logger.info(f"[{task_id}] 共获取到 {len(all_index_codes)} 个指数代码，将按每 {INDEX_SLICE_SIZE} 个进行切片并分配任务。")

        # 对指数代码列表进行切片并分配执行任务
        for i in range(0, len(all_index_codes), INDEX_SLICE_SIZE):
            index_codes_slice = all_index_codes[i:i + INDEX_SLICE_SIZE]
            
            # 代码修改处: 优化日志输出，使其更清晰
            log_msg = (
                f"[{task_id}] 调度中: 分配第 {i // INDEX_SLICE_SIZE + 1} 个切片 "
                f"(含 {len(index_codes_slice)} 个代码) 到执行队列 [SaveData_TimeTrade]..."
            )
            logger.info(log_msg)
            
            # 分配执行任务
            save_index_daily_history_slice.delay(index_codes_slice)

        logger.info(f"[{task_id}] 调度任务完成 - 所有 {len(range(0, len(all_index_codes), INDEX_SLICE_SIZE))} 个指数切片任务已成功分配。")
    except Exception as e:
        logger.error(f"[{task_id}] 执行 [指数每日指标] 调度任务时发生严重错误: {e}", exc_info=True)
        # 调度任务失败通常需要手动干预，可以根据需要设置重试
        # raise self.retry(exc=e, countdown=300, max_retries=3)


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
    this_monday, this_friday = get_this_monday_and_friday()
    try:
        print(f"开始处理 指数每日指标...")
        async_to_sync(index_basic_dao.save_index_daily_basic_history)()
        print("任务完成 - 大盘指数每日指标")
    except Exception as e:
        logger.error(f"执行 指数每日指标 任务时发生意外错误: {e}", exc_info=True)








