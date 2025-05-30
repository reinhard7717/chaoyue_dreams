# 每日收盘任务整合
import logging
from typing import List
from chaoyue_dreams.celery import app as celery_app
from tasks.tushare.fund_flow_tasks import save_fund_flow_daily_data_this_week_task, save_fund_flow_daily_data_ths_this_week_task, save_fund_flow_daily_data_ths_today, save_fund_flow_daily_data_ths_yesterday, save_fund_flow_daily_data_today, save_fund_flow_daily_data_yesterday
from tasks.tushare.index_tasks import save_index_daily_basic_history, save_index_daily_today_task, save_index_daily_this_week_task, save_index_daily_yesterday_task
from tasks.tushare.stock_time_trade_tasks import save_cyq_data_this_week_task, save_day_data_this_week_batch, save_day_data_today_task, save_day_data_yesterday_task, save_stocks_daily_basic_data_this_week_task, save_stocks_daily_basic_data_today_task, save_stocks_daily_basic_data_yesterday_task, save_stocks_minute_data_this_week_task, save_stocks_minute_data_today_task, save_stocks_minute_data_yesterday_task  # 从 celery.py 导入 app 实例并重命名为 celery_app
from tasks.tushare.industry_tasks import save_ths_index_today_task, save_ths_index_yesterday_task


logger = logging.getLogger('tasks')

#  ================ （当日）整体任务 ================
@celery_app.task(bind=True, name='tasks.tushare.cal_daily_tasks.run_daily_data_ingestion_task', queue='celery')
def run_daily_data_ingestion_task(self, trade_time_str=None):
    """
    整体任务：按顺序执行当日收盘后的数据采集任务。
    包括：分钟数据、日线数据（含筹码）、当日基本信息。
    这个任务由 Celery Beat 调度，通常在收盘后执行。
    """
    logger.info("整体任务启动: run_daily_data_ingestion_task - 开始执行当日数据采集流程")

    try:
        # 步骤 1: 执行分钟数据采集调度任务
        # 这个任务会获取所有股票并分批派发 save_stocks_minute_data_today_batch
        logger.info("开始执行: 分钟数据采集调度任务...")
        # 使用 .delay() 或 .apply_async() 异步触发子任务
        # .delay() 是 .apply_async() 的简化版
        minute_task_result = save_stocks_minute_data_today_task.delay(trade_time_str=trade_time_str)
        logger.info(f"已分派分钟数据采集调度任务。任务ID: {minute_task_result.id}")

        # 步骤 2: 执行日线数据（含筹码）采集任务
        # 这个任务会采集日线数据，并且内部包含了筹码数据的采集
        logger.info("开始执行: 日线数据（含筹码）采集任务...")
        daily_data_task_result = save_day_data_today_task.delay()
        logger.info(f"已分派日线数据（含筹码）采集任务。任务ID: {daily_data_task_result.id}")

        # 步骤 3: 执行当日基本信息采集任务
        logger.info("开始执行: 当日基本信息采集任务...")
        daily_basic_task_result = save_stocks_daily_basic_data_today_task.delay()
        logger.info(f"已分派当日基本信息采集任务。任务ID: {daily_basic_task_result.id}")

        # 步骤4：执行指数每日指标
        logger.info("开始执行: 指数每日指标...")
        save_index_daily_today_task.delay()
        logger.info(f"已分派指数每日指标任务。")

        # 步骤5：今日资金流向 - 个股
        logger.info("开始执行: 个股日级资金流向数据...")
        save_fund_flow_daily_data_today.delay()
        logger.info(f"已分派个股日级资金流向数据任务。")
        
        # 每日任务：同花顺板块 & 指数行情
        logger.info("开始执行: 同花顺板块 & 指数行情")
        save_ths_index_today_task.delay()
        
        # 步骤5：板块、行业资金流向数据 - 同花顺
        logger.info("开始执行: 板块、行业资金流向数据 - 同花顺...")
        save_fund_flow_daily_data_ths_today.delay()
        logger.info(f"已分派板块、行业资金流向数据 - 同花顺任务。")

        logger.info("整体任务结束: run_daily_data_ingestion_task - 所有当日数据采集任务已分派。")

        # 返回一些信息，方便查看任务状态
        return {
            "status": "success",
            "message": "所有当日数据采集任务已分派",
            "dispatched_tasks": {
                "minute_data_scheduler": minute_task_result.id,
                "daily_data_and_chips": daily_data_task_result.id,
                "daily_basic_info": daily_basic_task_result.id,
            }
        }

    except Exception as e:
        logger.error(f"整体任务 run_daily_data_ingestion_task 执行失败: {e}", exc_info=True)
        # 记录异常并返回错误状态
        return {"status": "error", "message": f"整体任务执行失败: {e}"}

@celery_app.task(bind=True, name='tasks.tushare.cal_daily_tasks.run_yesterday_data_ingestion_task', queue='celery')
def run_yesterday_data_ingestion_task(self):
    """
    整体任务：按顺序执行当日收盘后的数据采集任务。
    包括：分钟数据、日线数据（含筹码）、当日基本信息。
    这个任务由 Celery Beat 调度，通常在收盘后执行。
    """
    logger.info("整体任务启动: run_yesterday_data_ingestion_task - 开始执行当日数据采集流程")

    try:
        # 步骤 1: 执行分钟数据采集调度任务
        # 这个任务会获取所有股票并分批派发 save_stocks_minute_data_today_batch
        logger.info("开始执行: （昨日）分钟数据采集调度任务...")
        # 使用 .delay() 或 .apply_async() 异步触发子任务
        # .delay() 是 .apply_async() 的简化版
        minute_task_result = save_stocks_minute_data_yesterday_task.delay()
        logger.info(f"已分派（昨日）分钟数据采集调度任务。任务ID: {minute_task_result.id}")

        # 步骤 2: 执行日线数据（含筹码）采集任务
        # 这个任务会采集日线数据，并且内部包含了筹码数据的采集
        logger.info("开始执行: （昨日）日线数据（含筹码）采集任务...")
        daily_data_task_result = save_day_data_yesterday_task.delay()
        logger.info(f"已分派（昨日）日线数据（含筹码）采集任务。任务ID: {daily_data_task_result.id}")

        # 步骤 3: 执行当日基本信息采集任务
        logger.info("开始执行: （昨日）基本信息采集任务...")
        daily_basic_task_result = save_stocks_daily_basic_data_yesterday_task.delay()
        logger.info(f"已分派（昨日）基本信息采集任务。任务ID: {daily_basic_task_result.id}")

        # 步骤4：执行指数每日指标
        logger.info("开始执行: （昨日）指数每日指标...")
        save_index_daily_yesterday_task.delay()
        logger.info(f"已分派（昨日）指数每日指标任务。")

        # 步骤5：今日资金流向 - 个股
        logger.info("开始执行: （昨日）个股日级资金流向数据...")
        save_fund_flow_daily_data_yesterday.delay()
        logger.info(f"已分派（昨日）个股日级资金流向数据任务。")
        
        # 每日任务：同花顺板块 & 指数行情
        logger.info("开始执行: （昨日）同花顺板块 & 指数行情")
        save_ths_index_yesterday_task.delay()
        
        # 步骤5：板块、行业资金流向数据 - 同花顺
        logger.info("开始执行: （昨日）板块、行业资金流向数据 - 同花顺...")
        save_fund_flow_daily_data_ths_yesterday.delay()
        logger.info(f"已分派（昨日）板块、行业资金流向数据 - 同花顺任务。")

        logger.info("整体任务结束: run_yesterday_data_ingestion_task - 所有当日数据采集任务已分派。")

        # 返回一些信息，方便查看任务状态
        return {
            "status": "success",
            "message": "所有（昨日）当日数据采集任务已分派",
            "dispatched_tasks": {
                "minute_data_scheduler": minute_task_result.id,
                "daily_data_and_chips": daily_data_task_result.id,
                "daily_basic_info": daily_basic_task_result.id,
            }
        }

    except Exception as e:
        logger.error(f"整体任务 run_yesterday_data_ingestion_task 执行失败: {e}", exc_info=True)
        # 记录异常并返回错误状态
        return {"status": "error", "message": f"整体任务执行失败: {e}"}


#  ================ （本周）整体任务 ================
@celery_app.task(bind=True, name='tasks.tushare.cal_daily_tasks.run_this_week_data_ingestion_task', queue='celery')
def run_this_week_data_ingestion_task(self, trade_time_str=None):
    """
    整体任务：按顺序执行当日收盘后的数据采集任务。
    包括：分钟数据、日线数据（含筹码）、当日基本信息。
    这个任务由 Celery Beat 调度，通常在收盘后执行。
    """
    logger.info("整体任务启动: run_this_week_data_ingestion_task - 开始执行当日数据采集流程")

    try:
        # 步骤 1: 执行分钟数据采集调度任务
        # 这个任务会获取所有股票并分批派发 save_stocks_minute_data_today_batch
        logger.info("开始执行: 分钟数据采集调度任务...")
        # 使用 .delay() 或 .apply_async() 异步触发子任务
        # .delay() 是 .apply_async() 的简化版
        minute_task_result = save_stocks_minute_data_this_week_task.delay()
        logger.info(f"已分派分钟数据采集调度任务。任务ID: {minute_task_result.id}")

        # 步骤 2: 执行日线数据（含筹码）采集任务
        # 这个任务会采集日线数据，并且内部包含了筹码数据的采集
        logger.info("开始执行: 日线数据（含筹码）采集任务...")
        daily_data_task_result = save_day_data_this_week_batch.delay()
        save_cyq_data_this_week_task.delay()
        logger.info(f"已分派日线数据（含筹码）采集任务。任务ID: {daily_data_task_result.id}")

        # 步骤 3: 执行当日基本信息采集任务
        logger.info("开始执行: 当日基本信息采集任务...")
        daily_basic_task_result = save_stocks_daily_basic_data_this_week_task.delay()
        logger.info(f"已分派当日基本信息采集任务。任务ID: {daily_basic_task_result.id}")

        # 步骤4：执行指数每日指标
        logger.info("开始执行: 指数每日指标...")
        save_index_daily_basic_history.delay()
        save_index_daily_this_week_task.delay()
        logger.info(f"已分派指数每日指标任务。")

        # 步骤5：今日资金流向 - 个股
        logger.info("开始执行: 个股日级资金流向数据...")
        save_fund_flow_daily_data_this_week_task.delay()
        logger.info(f"已分派个股日级资金流向数据任务。")

        # 步骤5：板块、行业资金流向数据 - 同花顺
        logger.info("开始执行: 板块、行业资金流向数据 - 同花顺...")
        save_fund_flow_daily_data_ths_this_week_task.delay()
        logger.info(f"已分派板块、行业资金流向数据 - 同花顺任务。")

        logger.info("整体任务结束: run_this_week_data_ingestion_task - 所有当日数据采集任务已分派。")

        # 返回一些信息，方便查看任务状态
        return {
            "status": "success",
            "message": "所有当日数据采集任务已分派",
            "dispatched_tasks": {
                "minute_data_scheduler": minute_task_result.id,
                "daily_data_and_chips": daily_data_task_result.id,
                "daily_basic_info": daily_basic_task_result.id,
            }
        }

    except Exception as e:
        logger.error(f"整体任务 run_this_week_data_ingestion_task 执行失败: {e}", exc_info=True)
        # 记录异常并返回错误状态
        return {"status": "error", "message": f"整体任务执行失败: {e}"}
