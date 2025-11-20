# 每日收盘任务整合
import logging
from typing import List
from celery import group, chord
from chaoyue_dreams.celery import app as celery_app
from tasks.stock_analysis_tasks import precompute_all_stocks_advanced_metrics
from tasks.tushare.fund_flow_tasks import save_fund_flow_daily_data_today, save_fund_flow_daily_data_ths_today, save_fund_flow_daily_data_ths_yesterday, save_fund_flow_daily_data_yesterday, save_fund_flow_data_this_week_task, save_hm_detail_data_today
from tasks.tushare.index_tasks import save_index_daily_basic_history, save_index_daily_today_task, save_index_daily_this_week_task, save_index_daily_yesterday_task, save_trade_cal
from tasks.tushare.stock_time_trade_tasks import save_cyq_data_this_week_task, save_cyq_data_today_task, save_day_data_this_week_batch, save_day_data_today_task, save_month_data_today_task, save_stocks_daily_basic_data_this_week_task, save_stocks_daily_basic_data_today_task, save_stocks_minute_data_this_week_task, save_stocks_minute_data_today_task, save_week_data_today_task  # 从 celery.py 导入 app 实例并重命名为 celery_app
from tasks.tushare.industry_tasks import save_all_daily_industry_concept_data_task, save_all_historical_data_task


logger = logging.getLogger('tasks')

#  ================ （当日）整体任务 ================
@celery_app.task(bind=True, name='tasks.tushare.cal_daily_tasks.dispatch_derived_data_tasks', queue='celery')
def dispatch_derived_data_tasks(self, primary_results, trade_time_str=None):
    """
    【V2.0 新增】第二阶段任务分发器。
    在第一阶段主要数据采集完成后被调用，负责分发那些依赖服务端后处理的、数据就绪慢的任务。
    """
    logger.info("第二阶段任务启动: dispatch_derived_data_tasks - 开始分发衍生数据采集任务")
    # 定义第二阶段需要执行的任务列表
    derived_tasks = [
        save_stocks_daily_basic_data_today_task.s(), # <-- 我们的问题任务，现在被安全地放在第二阶段
        save_fund_flow_daily_data_today.s(),
        save_fund_flow_daily_data_ths_today.s(),
        save_hm_detail_data_today.s(),
    ]
    # 再次使用 chord，确保所有衍生数据任务完成后，才执行最终的高级筹码计算
    final_callback = precompute_all_stocks_advanced_metrics.s()
    logger.info("开始执行: 所有衍生数据采集任务并行调度，全部完成后执行高级筹码指标预计算。")
    # 使用 countdown 参数，给数据提供商留出充足的处理时间（例如10分钟）
    # 这是一个非常重要的“保险丝”，确保万无一失
    chord(derived_tasks)(final_callback).apply_async(countdown=600) # 延迟10分钟执行
    logger.info("第二阶段任务分派完毕，将在10分钟后执行。")
    return {"status": "success", "message": "第二阶段衍生数据任务已分派"}


#  ================ （当日）整体任务 ================
@celery_app.task(bind=True, name='tasks.tushare.cal_daily_tasks.run_daily_data_ingestion_task', queue='celery')
def run_daily_data_ingestion_task(self, trade_time_str=None):
    """
    【V2.0 两阶段工作流版】
    - 将任务流重构为两个阶段，彻底解决数据就绪时间的依赖问题。
      - 阶段一: 并行执行数据就绪快的核心数据任务（分钟、日线、周线、月线、CYQ）。
      - 阶段二: 在阶段一完成后，通过回调启动一个新的分发器，该分发器负责执行数据就绪慢的衍生数据任务（每日指标、资金流等），并增加了延迟执行的保险机制。
    """
    logger.info("整体任务启动: run_daily_data_ingestion_task (V2.0 两阶段工作流版)")
    try:
        logger.info("开始执行: 更新交易日历...")
        save_trade_cal.delay()
        # 1. 定义第一阶段任务（数据就绪快）
        primary_tasks = [
            save_stocks_minute_data_today_task.s(trade_time_str=trade_time_str),
            save_day_data_today_task.s(),
            save_cyq_data_today_task.s(),
            save_week_data_today_task.s(),
            save_month_data_today_task.s(),
            save_index_daily_today_task.s(),
            save_all_daily_industry_concept_data_task.s(),
        ]
        # 2. 使用 chord 将第一阶段任务与第二阶段的分发器连接起来
        # 当所有 primary_tasks 完成后，会自动调用 dispatch_derived_data_tasks
        callback = dispatch_derived_data_tasks.s(trade_time_str=trade_time_str)
        logger.info("开始执行: 第一阶段主要数据采集任务并行调度，全部完成后将触发第二阶段任务分发器。")
        job = chord(primary_tasks)(callback)
        logger.info("整体任务分派完成: run_daily_data_ingestion_task - 所有任务已按两阶段工作流分派。")
        return {
            "status": "success",
            "message": "两阶段工作流已启动，等待第一阶段任务完成。",
            "chord_id": job.id,
        }
    except Exception as e:
        logger.error(f"整体任务 run_daily_data_ingestion_task 执行失败: {e}", exc_info=True)
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
        save_all_historical_data_task.delay()
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
        # 步骤5：个股、板块、行业资金流向数据
        logger.info("开始执行: 板块、行业资金流向数据...")
        save_fund_flow_data_this_week_task.delay()
        logger.info(f"已分派板块、行业资金流向数据 任务。")
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
