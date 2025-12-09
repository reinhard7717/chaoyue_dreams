# tasks/tushare/fund_flow_tasks.py
import asyncio
from asgiref.sync import async_to_sync
import logging
import datetime
from django.utils import timezone
from utils.task_helpers import with_cache_manager
from celery import group, chord, current_task # 【修改】新增导入 chord
from django.db.models import Q
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
from dao_manager.tushare_daos.fund_flow_dao import FundFlowDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from stock_models.index import TradeCalendar
from utils.cache_manager import CacheManager

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

# 获取上周一和上周五的日期
def get_last_monday_and_friday():
    """获取上周一和上周五的日期"""
    today = datetime.date.today()
    this_monday = today - datetime.timedelta(days=today.weekday())
    last_monday = this_monday - datetime.timedelta(days=7)
    last_friday = last_monday + datetime.timedelta(days=4)
    return last_monday, last_friday

#  ================ （当日）个股日级资金流向数据 （三种渠道） ================
# 创建一个通用的、原子化的子任务，用于执行DAO中的异步保存方法
@celery_app.task(queue='SaveHistoryData_TimeTrade', acks_late=True)
@with_cache_manager
def execute_save_today_fund_flow_method(method_name: str, trade_date: datetime.date, cache_manager=None):
    """
    【无绑定版】
    通用子任务：执行FundFlowDao中的指定异步方法来保存当日数据。
    """
    # 使用 current_task.request.id 获取任务ID
    task_id = current_task.request.id
    fund_flow_dao = FundFlowDao(cache_manager)
    logger.info(f"子任务启动: {task_id} - {method_name}")
    print(f"调试信息：子任务 {task_id} 启动，执行异步方法: {method_name}")
    async def main():
        save_method = getattr(fund_flow_dao, method_name)
        return await save_method(trade_date)
    async_to_sync(main)()
    logger.info(f"子任务成功: {task_id} ({method_name})")
    print(f"调试信息：子任务 {task_id} ({method_name}) 执行成功。")
    return {"status": "success", "method": method_name, "task_id": task_id}

# 原任务被重构为编排和分派任务
@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_today', queue='SaveHistoryData_TimeTrade')
def save_fund_flow_daily_data_today():
    """
    【无绑定版】
    调度器任务（编排者）：负责并行分派获取当日三种渠道资金流数据的子任务，并在完成后记录日志。
    """
    # 使用 current_task.request.id 获取任务ID
    task_id = current_task.request.id
    logger.info(f"任务启动: {task_id} (编排者模式) - 准备分派并行子任务")
    print(f"调试信息：主任务 {task_id} 启动，准备分派当日资金流数据获取任务组。")
    try:
        today_date = timezone.now().date()
        target_methods = [
            'save_history_fund_flow_daily_data',
            'save_history_fund_flow_daily_ths_data',
            'save_history_fund_flow_daily_dc_data'
        ]
        task_signatures = [
            execute_save_today_fund_flow_method.s(method_name=method, trade_date=today_date)
            for method in target_methods
        ]
        # 【修改】使用 chord 替代 group().get() 以避免在任务中阻塞。
        # chord 结构: chord(并行任务组)(完成后执行的回调任务)
        # 当 task_signatures 中的所有任务完成后，会自动调用 log_fund_flow_group_completion 任务。
        callback = log_fund_flow_group_completion.s()
        task_chord = chord(task_signatures, callback)
        result = task_chord.apply_async()
        # 【修改】日志记录分派动作，而不是等待完成。result.id 是回调任务的ID。
        logger.info(f"任务弦(Chord)成功分派. 包含 {len(target_methods)} 个并行子任务和一个回调任务. Callback Task ID: {result.id}")
        print(f"调试信息：任务弦(Chord) {result.id} 已成功分派，包含 {len(target_methods)} 个子任务和一个回调任务。")
        # 【修改】主任务分派后立即返回，实现真正的异步。返回状态改为 "dispatched"。
        return {"status": "dispatched", "callback_task_id": result.id, "dispatched_tasks": len(target_methods)}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_today (编排者模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to dispatch task group: {e}"}

# 用于 chord 的回调任务，在任务组完成后执行
@celery_app.task(queue='SaveHistoryData_TimeTrade', ignore_result=True)
def log_fund_flow_group_completion(results):
    """
    Chord callback: 记录每日资金流任务组的完成情况。
    'results' 参数是组中所有任务返回值的列表，由Celery自动传入。
    """
    # 这部分逻辑是从原 save_fund_flow_daily_data_today 任务中移动过来的
    logger.info(f"所有当日资金流子任务已全部完成。收到了 {len(results)} 个子任务的结果。")
    print(f"调试信息：当日资金流数据获取任务组所有子任务已完成。")


@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_hm_detail_data_today', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_hm_detail_data_today(cache_manager=None):
    """
    【无绑定版】
    Celery任务：获取并保存【当天】的游资每日明细数据。
    """
    # 使用 current_task.request.id 获取任务ID
    task_id = current_task.request.id
    dao = FundFlowDao(cache_manager)
    print(f"开始执行Celery任务: 保存【当天】的游资每日明细数据。 save_hm_detail_data_today, Task ID: {task_id}")
    async def main():        
        return await dao.save_hm_detail_data()
    async_to_sync(main)()
    print(f"Celery任务: save_hm_detail_data_today, Task ID: {task_id} 执行成功。")
    return f"成功获取并保存了当天的游资明细数据。"

#  ================ （昨日）个股日级资金流向数据 （三种渠道） ================
@celery_app.task(queue='SaveHistoryData_TimeTrade', acks_late=True)
@with_cache_manager
def execute_fund_flow_dao_method(method_name: str, trade_date: str, cache_manager=None):
    """
    【无绑定版】
    通用执行者子任务：执行FundFlowDao中的指定异步方法。
    """
    # 使用 current_task.request.id 获取任务ID
    task_id = current_task.request.id
    fund_flow_dao = FundFlowDao(cache_manager)
    logger.info(f"通用子任务启动: {task_id} - {method_name}")
    print(f"调试信息：子任务 {task_id} 启动，执行异步方法: {method_name}")
    async def main():
        save_method = getattr(fund_flow_dao, method_name)
        return await save_method(trade_date)
    async_to_sync(main)()
    logger.info(f"通用子任务成功: {task_id} ({method_name})")
    print(f"调试信息：子任务 {task_id} ({method_name}) 执行成功。")
    return {"status": "success", "method": method_name, "task_id": task_id}
    
# 原任务被重构为编排和分派任务
@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_yesterday', queue='SaveHistoryData_TimeTrade')
def save_fund_flow_daily_data_yesterday():
    """
    【无绑定版】
    调度器任务（编排者）：负责并行分派获取【昨日】三种渠道资金流数据的子任务。
    """
    # 使用 current_task.request.id 获取任务ID
    task_id = current_task.request.id
    logger.info(f"任务启动: {task_id} (编排者模式) - 准备分派并行子任务")
    print(f"调试信息：主任务 {task_id} 启动，准备分派【昨日】资金流数据获取任务组。")
    try:
        today_date = timezone.now().date()
        yesterday = today_date - datetime.timedelta(days=1)
        target_methods = [
            'save_history_fund_flow_daily_data',
            'save_history_fund_flow_daily_ths_data',
            'save_history_fund_flow_daily_dc_data'
        ]
        task_signatures = [
            execute_fund_flow_dao_method.s(method_name=method,trade_date=yesterday)
            for method in target_methods
        ]
        task_group = group(task_signatures)
        result = task_group.apply_async()
        logger.info(f"任务组成功分派. Group ID: {result.id}. 包含 {len(target_methods)} 个子任务.")
        print(f"调试信息：任务组 {result.id} 已成功分派，包含 {len(target_methods)} 个子任务。")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(target_methods)}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_yesterday (编排者模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to dispatch task group: {e}"}

#  ================ （本周）日级资金流向数据（三种渠道） ================
# 创建一个通用的、原子化的子任务，用于执行具体的数据保存操作
@celery_app.task(queue='SaveData_TimeTrade', acks_late=True)
@with_cache_manager
def execute_save_fund_flow_method(method_name: str, start_date: str, end_date: str, cache_manager=None):
    """
    通用子任务：执行FundFlowDao中的指定方法来保存数据。
    """
    task_id = current_task.request.id
    print(f"调试信息：子任务 {task_id} 启动，执行方法: {method_name}")
    logger.info(f"子任务启动: {task_id} - {method_name}, 日期范围: {start_date} 到 {end_date}")
    try:
        ff_dao = FundFlowDao(cache_manager)
        async def main():
            save_method = getattr(ff_dao, method_name)
            return await save_method(start_date=start_date, end_date=end_date)
        async_to_sync(main)()
        logger.info(f"子任务成功: {task_id} ({method_name})")
        print(f"调试信息：子任务 {task_id} ({method_name}) 执行成功。")
        return {"status": "success", "method": method_name, "task_id": task_id}
    except Exception as e:
        logger.error(f"子任务 {task_id} ({method_name}) 执行失败: {e}", exc_info=True)
        raise

# 原任务被重构为编排和分派任务
@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_data_this_week_task', queue='celery')
def save_fund_flow_data_this_week_task():
    """
    【无绑定版】
    调度器任务（编排者）：分派本周资金流数据更新子任务。
    """
    logger.info(f"任务启动: save_fund_flow_data_this_week_task (编排者模式) - 准备分派并行子任务")
    try:
        this_monday, this_friday = get_this_monday_and_friday()
        target_methods = [
            'save_history_fund_flow_daily_data',
            'save_history_fund_flow_daily_ths_data',
            'save_history_fund_flow_daily_dc_data'
        ]
        task_signatures = [
            execute_save_fund_flow_method.s(method_name=method, start_date=this_monday, end_date=this_friday)
            for method in target_methods
        ]
        task_group = group(task_signatures)
        print(f"调试信息：准备分派 {len(target_methods)} 个资金流数据保存子任务...")
        result = task_group.apply_async()
        logger.info(f"任务组成功分派. Group ID: {result.id}. 包含 {len(target_methods)} 个子任务.")
        print(f"调试信息：任务组 {result.id} 已成功分派。")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(target_methods)}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_data_this_week_task (编排者模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to dispatch task group: {e}"}


#  ================ （历史）日级资金流向数据（三种渠道） ================
@celery_app.task(queue="SaveHistoryData_TimeTrade")
@with_cache_manager
def save_fund_flow_daily_data_history_batch(start_date: datetime.date, end_date: datetime.date, cache_manager=None):
    """
    【优化版 V2】从Tushare批量获取指定日期范围内的历史日级资金流向数据并保存。
    - 此任务被设计为处理一个明确的、不宜过大的日期范围。
    - [重构] 并发执行三个数据源的获取，提升效率。
    - [重构] 增强错误处理，单个数据源失败不影响其他数据源。
    """
    log_msg = f"开始并发处理 {start_date} 到 {end_date} 的历史日级资金流向数据..."
    logger.info(log_msg)
    fund_flow_dao = FundFlowDao(cache_manager)
    # [优化] 定义一个异步主函数来使用asyncio.gather并发执行所有数据获取任务
    async def main():
        # 将三个独立的异步任务放入一个列表中
        tasks = [
            fund_flow_dao.save_history_fund_flow_daily_data(start_date=start_date, end_date=end_date),
            fund_flow_dao.save_history_fund_flow_daily_ths_data(start_date=start_date, end_date=end_date),
            fund_flow_dao.save_history_fund_flow_daily_dc_data(start_date=start_date, end_date=end_date)
        ]
        # 使用 asyncio.gather 并发运行所有任务，并设置 return_exceptions=True 以便捕获所有异常而不是中途停止
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # 检查每个任务的结果，进行精细化的日志记录
        source_names = ["Tushare", "同花顺", "东方财富"]
        has_error = False
        for name, result in zip(source_names, results):
            if isinstance(result, Exception):
                logger.error(f"数据源 [{name}] 在处理日期范围 {start_date}-{end_date} 时失败: {result}", exc_info=False) # exc_info设为False避免日志过于冗长
                has_error = True
            else:
                logger.info(f"数据源 [{name}] 在处理日期范围 {start_date}-{end_date} 时成功。")
        return not has_error # 如果没有错误，返回True
    # 只需调用一次 async_to_sync() 来执行异步主函数
    success = async_to_sync(main)()
    if success:
        logger.info(f"成功完成日期范围 {start_date}-{end_date} 的所有资金流数据保存任务。")
        return {"status": "success"}
    else:
        logger.warning(f"日期范围 {start_date}-{end_date} 的资金流数据保存任务部分失败。")
        # 即使部分失败，也认为是可接受的完成，以便Celery不重试。错误已记录。
        return {"status": "partial_success"}

@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_task', queue='celery')
def save_fund_flow_daily_data_history_task(): 
    """
    【无绑定版】
    调度器任务：将历史资金流数据获取任务切片并分派。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_history_task (调度器-任务分片模式)")
    try:
        NUM_DAYS_TO_FETCH = 1800
        CHUNK_SIZE_DAYS = 5
        trade_days_list = TradeCalendar.get_latest_n_trade_dates(n=NUM_DAYS_TO_FETCH)
        if not trade_days_list:
            logger.warning("未能从TradeCalendar获取到交易日历，无法分派任务。")
            return {"status": "skipped", "message": "Trade calendar is empty."}
        total_start_date = trade_days_list[-1]
        total_end_date = trade_days_list[0]
        logger.info(f"计算出的总处理日期范围为: {total_start_date} 到 {total_end_date} (共 {len(trade_days_list)} 个交易日)")
        dispatched_tasks_count = 0
        current_end_date = total_end_date
        while current_end_date >= total_start_date:
            current_start_date = max(total_start_date, current_end_date - datetime.timedelta(days=CHUNK_SIZE_DAYS - 1))
            print(f"准备分派任务，范围: {current_start_date} -> {current_end_date}")
            save_fund_flow_daily_data_history_batch.s(
                start_date=current_start_date, 
                end_date=current_end_date
            ).apply_async()
            dispatched_tasks_count += 1
            current_end_date = current_start_date - datetime.timedelta(days=1)
        logger.info(f"任务结束: 成功将总范围分派成 {dispatched_tasks_count} 个子任务进行处理。")
        return {"status": "success", "dispatched_tasks": dispatched_tasks_count, "total_date_range": f"{total_start_date}_to_{total_end_date}"}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": 0}

@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_last_n_days_task', queue='celery')
def save_fund_flow_daily_data_last_n_days_task(num_days: int):
    """
    【无绑定版】
    调度器任务：获取最近N个交易日的个股日级资金流向数据（三种渠道）。
    Args:
        num_days (int): 需要获取的交易日数量。
    """
    task_id = current_task.request.id
    logger.info(f"任务启动: {task_id} - save_fund_flow_daily_data_last_n_days_task (调度器模式) - 准备分派最近 {num_days} 个交易日的资金流数据任务。")
    print(f"调试信息：主任务 {task_id} 启动，准备分派最近 {num_days} 个交易日的个股资金流数据获取任务组。")
    try:
        # 获取最新的N个交易日
        trade_days_list = TradeCalendar.get_latest_n_trade_dates(n=num_days)
        if not trade_days_list:
            logger.warning(f"任务 {task_id}: 未能从TradeCalendar获取到最近 {num_days} 个交易日，无法分派任务。")
            print(f"调试信息：任务 {task_id} 警告：未能获取到交易日。")
            return {"status": "skipped", "message": "Trade calendar is empty or num_days is invalid."}
        # 为每个交易日创建子任务签名
        task_signatures = []
        for trade_date in trade_days_list:
            # save_fund_flow_daily_data_history_batch 已经支持处理单个日期（start_date=end_date）
            task_signatures.append(
                save_fund_flow_daily_data_history_batch.s(
                    start_date=trade_date, 
                    end_date=trade_date
                ).set(queue='SaveHistoryData_TimeTrade') # 指定子任务的队列
            )
        if not task_signatures:
            logger.info(f"任务 {task_id}: 没有需要分派的资金流数据子任务。")
            print(f"调试信息：任务 {task_id}：没有子任务需要分派。")
            return {"status": "success", "dispatched_tasks": 0}

        # 将所有子任务组成一个组并异步分派
        task_group = group(task_signatures)
        result = task_group.apply_async()
        logger.info(f"任务 {task_id}: 成功分派 {len(task_signatures)} 个个股资金流数据子任务组。Group ID: {result.id}")
        print(f"调试信息：任务 {task_id}: 成功分派 {len(task_signatures)} 个个股资金流数据子任务组。Group ID: {result.id}")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(task_signatures), "num_days_requested": num_days}
    except Exception as e:
        logger.error(f"任务 {task_id}: 执行 save_fund_flow_daily_data_last_n_days_task 时出错: {e}", exc_info=True)
        print(f"调试信息：任务 {task_id} 错误：{e}")
        return {"status": "error", "message": f"Failed to dispatch task group: {e}", "num_days_requested": num_days}


# ================ （当日）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_today', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_fund_flow_daily_data_ths_today(cache_manager: CacheManager):
    """
    【无绑定版】
    获取并保存【当日】的板块、行业资金流向数据（同花顺）。
    """
    fund_flow_dao = FundFlowDao(cache_manager)
    logger.info(f"开始处理（当日）板块、行业资金流向数据 - 同花顺...")
    async def main():
        today_date = timezone.now().date()
        return await fund_flow_dao.save_history_fund_flow_cnt_ths_data(trade_date=today_date)
    async_to_sync(main)()

# ================ （昨日）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_yesterday', queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_fund_flow_daily_data_ths_yesterday(cache_manager: CacheManager):
    """
    【无绑定版】
    获取并保存【昨日】的板块、行业资金流向数据（同花顺）。
    """
    fund_flow_dao = FundFlowDao(cache_manager)
    logger.info(f"开始处理（昨日）板块、行业资金流向数据 - 同花顺...")
    async def main():
        today_date = timezone.now().date()
        yesterday = today_date - datetime.timedelta(days=1)
        return await fund_flow_dao.save_history_fund_flow_cnt_ths_data(trade_date=yesterday)
    async_to_sync(main)()

# ================ （本周）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_fund_flow_daily_data_ths_this_week_batch(this_monday: datetime.date, this_friday: datetime.date, cache_manager: CacheManager):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    """
    fund_flow_dao = FundFlowDao(cache_manager)
    logger.info(f"开始处理 {this_monday} - {this_friday} 的 （历史）板块资金流向数据 - 同花顺...")
    async def main():
        await fund_flow_dao.save_history_fund_flow_cnt_ths_data(start_date=this_monday, end_date=this_friday)
        await fund_flow_dao.save_history_fund_flow_industry_ths_data(start_date=this_monday, end_date=this_friday)
    async_to_sync(main)()

@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_this_week_task')
def save_fund_flow_daily_data_ths_this_week_task():
    """
    【无绑定版】
    调度器任务：分派获取本周板块、行业资金流向数据的任务。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_ths_this_week_task (调度器模式)")
    try:
        this_monday, this_friday = get_this_monday_and_friday()
        save_fund_flow_daily_data_ths_this_week_batch.s(this_monday, this_friday).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
        logger.info(f"任务结束: save_fund_flow_daily_data_ths_this_week_task (调度器模式)")
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_ths_this_week_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# ================ （历史）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(queue='SaveHistoryData_TimeTrade')
@with_cache_manager
def save_fund_flow_daily_data_ths_history_batch(trade_date: datetime.date, cache_manager: CacheManager):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    """
    fund_flow_dao = FundFlowDao(cache_manager)
    logger.info(f"开始处理 {trade_date} 的 （历史）板块资金流向数据 - 同花顺...")
    async def main():
        await fund_flow_dao.save_history_fund_flow_cnt_ths_data(trade_date)
        await fund_flow_dao.save_history_fund_flow_industry_ths_data(trade_date)
    async_to_sync(main)()

@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_history_task', queue='celery')
def save_fund_flow_daily_data_ths_history_task():
    """
    【无绑定版】
    调度器任务：为历史交易日分派板块、行业资金流向数据获取任务。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_ths_history_task (调度器模式)")
    try:
        cache_manager_instance = CacheManager()
        index_info_dao = IndexBasicDAO(cache_manager_instance)
        trade_days = async_to_sync(index_info_dao.get_last_n_trade_cal_open)(n=1500)
        total_dispatched_batches = 0
        for cal_date in trade_days:
            save_fund_flow_daily_data_ths_history_batch.s(cal_date).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
            total_dispatched_batches += 1
        logger.info(f"任务结束: save_fund_flow_daily_data_ths_history_task (调度器模式) - 共分派 {total_dispatched_batches} 个批量任务")
        return {"status": "success", "dispatched_batches": total_dispatched_batches}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_ths_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

@celery_app.task(name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_ths_data_last_n_days_task', queue='celery')
def save_fund_flow_daily_ths_data_last_n_days_task(num_days: int):
    """
    【无绑定版】
    调度器任务：获取最近N个交易日的板块、行业资金流向数据（同花顺）。
    Args:
        num_days (int): 需要获取的交易日数量。
    """
    task_id = current_task.request.id
    logger.info(f"任务启动: {task_id} - save_fund_flow_daily_ths_data_last_n_days_task (调度器模式) - 准备分派最近 {num_days} 个交易日的板块/行业资金流数据任务。")
    print(f"调试信息：主任务 {task_id} 启动，准备分派最近 {num_days} 个交易日的板块/行业资金流数据获取任务组。")
    try:
        # 获取最新的N个交易日
        trade_days_list = TradeCalendar.get_latest_n_trade_dates(n=num_days)
        if not trade_days_list:
            logger.warning(f"任务 {task_id}: 未能从TradeCalendar获取到最近 {num_days} 个交易日，无法分派任务。")
            print(f"调试信息：任务 {task_id} 警告：未能获取到交易日。")
            return {"status": "skipped", "message": "Trade calendar is empty or num_days is invalid."}
        # 为每个交易日创建子任务签名
        task_signatures = []
        for trade_date in trade_days_list:
            # save_fund_flow_daily_data_ths_history_batch 已经支持处理单个交易日
            task_signatures.append(
                save_fund_flow_daily_data_ths_history_batch.s(
                    trade_date=trade_date
                ).set(queue=FAVORITE_SAVE_API_DATA_QUEUE) # 沿用现有历史任务的队列配置
            )
        if not task_signatures:
            logger.info(f"任务 {task_id}: 没有需要分派的板块/行业资金流数据子任务。")
            print(f"调试信息：任务 {task_id}：没有子任务需要分派。")
            return {"status": "success", "dispatched_tasks": 0}

        # 将所有子任务组成一个组并异步分派
        task_group = group(task_signatures)
        result = task_group.apply_async()
        logger.info(f"任务 {task_id}: 成功分派 {len(task_signatures)} 个板块/行业资金流数据子任务组。Group ID: {result.id}")
        print(f"调试信息：任务 {task_id}: 成功分派 {len(task_signatures)} 个板块/行业资金流数据子任务组。Group ID: {result.id}")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(task_signatures), "num_days_requested": num_days}
    except Exception as e:
        logger.error(f"任务 {task_id}: 执行 save_fund_flow_daily_ths_data_last_n_days_task 时出错: {e}", exc_info=True)
        print(f"调试信息：任务 {task_id} 错误：{e}")
        return {"status": "error", "message": f"Failed to dispatch task group: {e}", "num_days_requested": num_days}








