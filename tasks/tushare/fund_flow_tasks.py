# tasks/tushare/fund_flow_tasks.py
import asyncio
import logging
import datetime
from django.utils import timezone
from celery import group
from django.db.models import Q
from typing import List, Dict, Any # 引入 List, Dict, Any
from chaoyue_dreams.celery import app as celery_app
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
            favorite_stock_codes.add(fav.get("stock_code"))
        logger.info(f"获取到 {len(favorite_stock_codes)} 个自选股代码")
    except Exception as e:
        logger.error(f"获取自选股列表时出错: {e}", exc_info=True)
    # 获取所有A股 (或者你需要的范围)
    try:
        # 注意：如果 get_stock_list() 返回大量数据，考虑分页或流式处理
        all_stocks = await stock_basic_dao.get_stock_list()
        for stock in all_stocks:
            code = str(stock.stock_code)
            if not code.endswith('.BJ'):
                all_stock_codes.add(code)
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
# [新增] 创建一个通用的、原子化的子任务，用于执行DAO中的异步保存方法
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.execute_save_today_fund_flow_method', queue=STOCKS_SAVE_API_DATA_QUEUE, acks_late=True)
def execute_save_today_fund_flow_method(self, method_name: str):
    """
    通用子任务：执行FundFlowDao中的指定异步方法来保存当日数据。
    此任务是原子化的，专注于单一的数据保存操作。
    :param method_name: FundFlowDao中需要被调用的异步方法名 (字符串格式)。
    """
    logger.info(f"子任务启动: {method_name}")
    print(f"调试信息：子任务 {self.request.id} 启动，执行异步方法: {method_name}")
    try:
        fund_flow_dao = FundFlowDao()
        # 使用getattr动态获取DAO实例的异步方法
        save_method = getattr(fund_flow_dao, method_name)
        # [关键修改] 在独立的子任务中正确使用 asyncio.run()
        asyncio.run(save_method())
        logger.info(f"子任务成功: {method_name}")
        print(f"调试信息：子任务 {self.request.id} ({method_name}) 执行成功。")
        return {"status": "success", "method": method_name}
    except Exception as e:
        logger.error(f"执行子任务 {method_name} 时出错: {e}", exc_info=True)
        print(f"调试信息：子任务 {self.request.id} ({method_name}) 执行失败: {e}")
        # 触发Celery的重试机制
        raise self.retry(exc=e, countdown=60, max_retries=3)

# [修改] 原任务被重构为编排和分派任务
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_today', queue=STOCKS_SAVE_API_DATA_QUEUE)
def save_fund_flow_daily_data_today(self):
    """
    [修改] 调度器任务（编排者）：
    负责并行分派获取当日三种渠道资金流数据的子任务。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_today (编排者模式) - 准备分派并行子任务")
    print(f"调试信息：主任务 {self.request.id} 启动，准备分派当日资金流数据获取任务组。")
    try:
        # [修改] 定义需要并行执行的所有异步数据保存方法名
        target_methods = [
            'save_today_fund_flow_daily_data',
            'save_today_fund_flow_daily_ths_data',
            'save_today_fund_flow_daily_dc_data'
        ]
        # [修改] 使用列表推导式和 .s() 方法创建一组任务签名
        task_signatures = [
            execute_save_today_fund_flow_method.s(method_name=method)
            for method in target_methods
        ]
        # [修改] 使用 group 将所有任务签名组合成一个可并行执行的任务组
        task_group = group(task_signatures)
        # [修改] 异步执行任务组
        result = task_group.apply_async()
        logger.info(f"任务组成功分派. Group ID: {result.id}. 包含 {len(target_methods)} 个子任务.")
        print(f"调试信息：任务组 {result.id} 已成功分派，包含 {len(target_methods)} 个子任务。")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(target_methods)}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_today (编排者模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to dispatch task group: {e}"}


#  ================ （昨日）个股日级资金流向数据 （三种渠道） ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.execute_fund_flow_dao_method', queue=STOCKS_SAVE_API_DATA_QUEUE, acks_late=True)
def execute_fund_flow_dao_method(self, method_name: str):
    """
    [重构] 通用执行者子任务：
    执行FundFlowDao中的指定异步方法。此任务是原子化的，可重试的，并且可被任何编排者任务调用。
    :param method_name: FundFlowDao中需要被调用的异步方法名 (字符串格式)。
    """
    logger.info(f"通用子任务启动: {method_name}")
    print(f"调试信息：子任务 {self.request.id} 启动，执行异步方法: {method_name}")
    try:
        fund_flow_dao = FundFlowDao()
        # 使用getattr动态获取DAO实例的异步方法
        save_method = getattr(fund_flow_dao, method_name)
        # 在独立的子任务中正确使用 asyncio.run() 来桥接同步和异步
        asyncio.run(save_method())
        logger.info(f"通用子任务成功: {method_name}")
        print(f"调试信息：子任务 {self.request.id} ({method_name}) 执行成功。")
        return {"status": "success", "method": method_name}
    except Exception as e:
        logger.error(f"执行通用子任务 {method_name} 时出错: {e}", exc_info=True)
        print(f"调试信息：子任务 {self.request.id} ({method_name}) 执行失败: {e}")
        # 触发Celery的重试机制，例如60秒后重试，最多3次
        raise self.retry(exc=e, countdown=60, max_retries=3)

# [修改] 原任务被重构为编排和分派任务
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_yesterday', queue=STOCKS_SAVE_API_DATA_QUEUE)
def save_fund_flow_daily_data_yesterday(self):
    """
    [修改] 调度器任务（编排者）：
    负责并行分派获取【昨日】三种渠道资金流数据的子任务。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_yesterday (编排者模式) - 准备分派并行子任务")
    print(f"调试信息：主任务 {self.request.id} 启动，准备分派【昨日】资金流数据获取任务组。")
    try:
        # [修改] 定义需要并行执行的所有【昨日】数据保存方法名
        target_methods = [
            'save_yesterday_fund_flow_daily_data',
            'save_yesterday_fund_flow_daily_ths_data',
            'save_yesterday_fund_flow_daily_dc_data'
        ]
        # [修改] 使用列表推导式和 .s() 方法创建一组对【通用执行者】的任务签名
        task_signatures = [
            execute_fund_flow_dao_method.s(method_name=method)
            for method in target_methods
        ]
        # [修改] 使用 group 将所有任务签名组合成一个可并行执行的任务组
        task_group = group(task_signatures)
        # [修改] 异步执行任务组
        result = task_group.apply_async()
        logger.info(f"任务组成功分派. Group ID: {result.id}. 包含 {len(target_methods)} 个子任务.")
        print(f"调试信息：任务组 {result.id} 已成功分派，包含 {len(target_methods)} 个子任务。")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(target_methods)}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_yesterday (编排者模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": f"Failed to dispatch task group: {e}"}

#  ================ （本周）日级资金流向数据（三种渠道） ================
# [新增] 创建一个通用的、原子化的子任务，用于执行具体的数据保存操作
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.execute_save_fund_flow_method', queue='SaveData_TimeTrade', acks_late=True)
def execute_save_fund_flow_method(self, method_name: str, start_date: str, end_date: str):
    """
    通用子任务：执行FundFlowDao中的指定方法来保存数据。
    这个任务是原子化的，专注于单一的数据保存操作，便于重试和并发。
    :param method_name: FundFlowDao中需要被调用的方法名 (字符串格式)。
    :param start_date: 开始日期。
    :param end_date: 结束日期。
    """
    logger.info(f"子任务启动: {method_name}, 日期范围: {start_date} 到 {end_date}")
    print(f"调试信息：子任务 {self.request.id} 启动，执行方法: {method_name}")
    try:
        ff_dao = FundFlowDao()
        # 使用getattr动态获取并调用DAO实例的方法
        save_method = getattr(ff_dao, method_name)
        save_method(start_date=start_date, end_date=end_date)
        logger.info(f"子任务成功: {method_name}")
        print(f"调试信息：子任务 {self.request.id} ({method_name}) 执行成功。")
        return {"status": "success", "method": method_name}
    except Exception as e:
        logger.error(f"执行子任务 {method_name} 时出错: {e}", exc_info=True)
        print(f"调试信息：子任务 {self.request.id} ({method_name}) 执行失败: {e}")
        # 触发Celery的重试机制，例如在60秒后重试，最多3次
        raise self.retry(exc=e, countdown=60, max_retries=3)

# [修改] 原任务被重构为编排和分派任务
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_data_this_week_task', queue='celery')
def save_fund_flow_data_this_week_task(self):
    """
    [修改] 调度器任务（编排者）：
    1. 获取本周的起止日期。
    2. 定义所有需要执行的数据保存任务列表。
    3. 使用 Celery group 将这些任务作为一组并行分派到队列中。
    这个任务由 Celery Beat 调度，负责启动一周数据的并行更新流程。
    """
    logger.info(f"任务启动: save_fund_flow_data_this_week_task (编排者模式) - 准备分派并行子任务")
    try:
        this_monday, this_friday = get_this_monday_and_friday()
        # [修改] 定义需要并行执行的所有数据保存方法名
        target_methods = [
            'save_history_fund_flow_daily_data_by_trade_date',
            'save_history_fund_flow_daily_ths_data_by_trade_date',
            'save_history_fund_flow_cnt_ths_data',
            'save_history_fund_flow_industry_ths_data'
        ]
        # [修改] 使用列表推导式和 .s() 方法创建一组任务签名 (signature)
        # .s() 创建了一个任务的“签名”，它包含了任务名和所有参数，但不会立即执行
        task_signatures = [
            execute_save_fund_flow_method.s(method_name=method, start_date=this_monday, end_date=this_friday)
            for method in target_methods
        ]
        # [修改] 使用 group 将所有任务签名组合成一个可并行执行的任务组
        task_group = group(task_signatures)
        print(f"调试信息：准备分派 {len(target_methods)} 个资金流数据保存子任务...")
        # [修改] 异步执行任务组
        result = task_group.apply_async()
        logger.info(f"任务组成功分派. Group ID: {result.id}. 包含 {len(target_methods)} 个子任务.")
        print(f"调试信息：任务组 {result.id} 已成功分派。")
        return {"status": "dispatched", "group_id": result.id, "dispatched_tasks": len(target_methods)}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_data_this_week_task (编排者模式) 时出错: {e}", exc_info=True)
        # [修改] 返回值更清晰地反映了错误情况
        return {"status": "error", "message": f"Failed to dispatch task group: {e}"}


#  ================ （历史）日级资金流向数据（三种渠道） ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_batch', queue="SaveHistoryData_TimeTrade")
def save_fund_flow_daily_data_history_batch(self, trade_date: datetime.date = None, start_date: datetime.date = None, end_date: datetime.date = None):
    """
    【优化版】从Tushare批量获取指定日期或日期范围内的历史日级资金流向数据并保存。
    - 优先使用 start_date 和 end_date 进行范围查询。
    - 如果只提供了 trade_date，则进行单日查询。
    """
    # [修改] 日志记录更清晰，能反映任务处理的范围
    if start_date and end_date:
        log_msg = f"开始处理 {start_date} 到 {end_date} 的历史日级资金流向数据..."
    elif trade_date:
        log_msg = f"开始处理 {trade_date} 的历史日级资金流向数据..."
    else:
        logger.warning("未提供任何有效日期参数，任务终止。")
        return {"status": "skipped", "message": "No date provided."}
    
    logger.info(log_msg)
    
    fund_flow_dao = FundFlowDao()
    try:
        # [修改] 将所有参数直接传递给已优化的DAO方法，它能智能处理
        # DAO方法内部会优先使用start_date和end_date
        asyncio.run(fund_flow_dao.save_history_fund_flow_daily_data_by_trade_date(
            trade_date=trade_date, 
            start_date=start_date, 
            end_date=end_date
        ))
        
        # 如果未来要启用同花顺和东方财富的数据源，也应改造它们以接受日期范围
        # logger.info("处理同花顺数据...")
        # asyncio.run(fund_flow_dao.save_history_fund_flow_daily_ths_data_by_trade_date(start_date=start_date, end_date=end_date))
        # logger.info("处理东方财富数据...")
        # asyncio.run(fund_flow_dao.save_history_fund_flow_daily_dc_data_trade_date(start_date=start_date, end_date=end_date))
        
        logger.info(f"成功完成日期范围 {start_date}-{end_date} (或单日 {trade_date}) 的资金流数据保存任务。")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

# [修改] 重构调度器任务，使其分派单个范围任务
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_history_task', queue="celery")
def save_fund_flow_daily_data_history_task(self): 
    """
    【优化版】调度器任务：
    1. 获取最近N个交易日。
    2. 计算起始和结束日期。
    3. 分派一个【单个】的批量任务来处理整个日期范围。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_history_task (调度器模式) - 获取交易日历并分派单个范围任务")
    try:
        index_basic_dao = IndexBasicDAO()
        trade_days_list = asyncio.run(index_basic_dao.get_last_n_trade_cal_open(n=1500))
        
        # [修改] 不再循环，而是计算范围
        if not trade_days_list:
            logger.warning("未能获取到交易日历，无法分派任务。")
            return {"status": "skipped", "message": "Trade calendar is empty."}
            
        # [修改] 获取起始和结束日期
        start_date = trade_days_list[-1]
        end_date = trade_days_list[0]
        
        logger.info(f"计算出的处理日期范围为: {start_date} 到 {end_date}")
        
        # [修改] 只分派一个任务，并传递日期范围
        save_fund_flow_daily_data_history_batch.s(
            start_date=start_date, 
            end_date=end_date
        ).apply_async()
        
        logger.info(f"任务结束: 成功分派1个批量任务处理从 {start_date} 到 {end_date} 的数据。")
        return {"status": "success", "dispatched_tasks": 1, "date_range": f"{start_date}_to_{end_date}"}
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_tasks": 0}

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
        today_date = timezone.now().date()
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_history_fund_flow_cnt_ths_data(trade_date=today_date))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# ================ （昨日）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_yesterday', queue=STOCKS_SAVE_API_DATA_QUEUE)
def save_fund_flow_daily_data_ths_yesterday(self):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理（当日）板块、行业资金流向数据 - 同花顺...")
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        today_date = timezone.now().date()
        yesterday = today_date - datetime.timedelta(days=1)  # 用timedelta减去1天，得到昨天的日期
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_history_fund_flow_cnt_ths_data(trade_date=yesterday))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

# ================ （本周）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_this_week_batch', queue=STOCKS_SAVE_API_DATA_QUEUE)
def save_fund_flow_daily_data_ths_this_week_batch(self, this_monday: datetime.date, this_friday: datetime.date):
    """
    从Tushare批量获取历史日级资金流向数据并保存到数据库（异步并发处理）
    Args:
        stock_codes: 股票代码列表
    """
    logger.info(f"开始处理 {this_monday} - {this_friday} 的 （历史）板块资金流向数据 - 同花顺...")
    # 在任务开始时创建一次 DAO 实例
    fund_flow_dao = FundFlowDao()
    try:
        # 异步获取数据并保存
        asyncio.run(fund_flow_dao.save_history_fund_flow_cnt_ths_data(start_date=this_monday, end_date=this_friday))
        asyncio.run(fund_flow_dao.save_history_fund_flow_industry_ths_data(start_date=this_monday, end_date=this_friday))
    except Exception as e:
        logger.error(f"执行批量保存任务时发生意外错误: {e}", exc_info=True)

@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_this_week_task')
def save_fund_flow_daily_data_ths_this_week_task(self):
    """
    调度器任务：
    1. 获取最近60天的A股交易日。
    2. 为每个交易日分派 save_fund_flow_daily_data_ths_history_batch 任务到指定队列。
    这个任务由 Celery Beat 调度。
    """
    logger.info(f"任务启动: save_fund_flow_daily_data_ths_history_task (调度器模式)")
    try:
        this_monday, this_friday = get_this_monday_and_friday()
        save_fund_flow_daily_data_ths_this_week_batch.s(this_monday, this_friday).set(queue=FAVORITE_SAVE_API_DATA_QUEUE).apply_async()
        logger.info(f"任务结束: save_fund_flow_daily_data_ths_history_task (调度器模式)")
    except Exception as e:
        logger.error(f"执行 save_fund_flow_daily_data_ths_history_task (调度器模式) 时出错: {e}", exc_info=True)
        return {"status": "error", "message": str(e), "dispatched_batches": 0}

# ================ （历史）板块、行业资金流向数据 - 同花顺 ================
@celery_app.task(bind=True, name='tasks.tushare.fund_flow_tasks.save_fund_flow_daily_data_ths_history_batch', queue=STOCKS_SAVE_API_DATA_QUEUE)
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










