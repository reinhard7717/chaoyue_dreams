# 文件: tasks/intraday_engine_tasks.py
import logging
from celery import shared_task
from asgiref.sync import async_to_sync
from django_celery_beat.models import PeriodicTask, CrontabSchedule

from intraday_engine.orchestrator import IntradayEngineOrchestrator

logger = logging.getLogger("celery_tasks")

# --- 任务一：盘前准备任务 ---
@shared_task(bind=True, name='tasks.intraday_engine.prepare_pools')
def prepare_pools(self):
    """
    【最佳实践】盘前准备任务，只负责构建监控池并写入Redis。
    应在交易日开盘前（如 09:15）由Celery Beat触发一次。
    """
    try:
        logger.info("盘中引擎盘前准备任务启动...")
        params = {} # 从配置加载
        orchestrator = IntradayEngineOrchestrator(params)
        async_to_sync(orchestrator.initialize_pools)()
        logger.info("盘中引擎盘前准备任务完成。")
        return {"status": "success"}
    except Exception as e:
        logger.error(f"盘前准备任务失败: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# --- 任务二：核心盘中循环任务 ---
@shared_task(bind=True, name='tasks.intraday_engine.run_cycle')
def run_cycle(self):
    """
    【最佳实践】核心盘中循环任务，只负责执行一轮分析。
    它从Redis读取状态，执行计算，并将结果写回Redis。
    """
    try:
        params = {} # 从配置加载
        orchestrator = IntradayEngineOrchestrator(params)
        
        # 直接执行循环，不再需要初始化
        signals = async_to_sync(orchestrator.run_single_cycle)()
        
        if signals:
            logger.info(f"本轮循环产生 {len(signals)} 条交易信号。")
        
        return {"status": "success", "signals_found": len(signals)}
    except Exception as e:
        logger.error(f"盘中引擎循环任务失败: {e}", exc_info=True)
        return {"status": "error", "reason": str(e)}

# --- 任务三：引擎调度器 (启动/停止) ---
# 这部分可以简化为一个管理命令或在Django Admin中手动操作，
# 但用Celery任务来自动化是更佳实践。

@shared_task(name='tasks.intraday_engine.scheduler')
def scheduler(action: str):
    """
    【最佳实践】统一的引擎调度器，负责启动和停止盘中循环任务。
    """
    task_name = 'intraday-engine-main-loop'
    
    if action == 'start':
        logger.info("调度器：正在启动盘中引擎...")
        schedule, _ = CrontabSchedule.objects.get_or_create(
            minute='*', hour='9-11, 13-14', day_of_week='1-5',
            month_of_year='*', timezone='Asia/Shanghai'
        )
        PeriodicTask.objects.update_or_create(
            name=task_name,
            defaults={
                'task': 'tasks.intraday_engine.run_cycle',
                'crontab': schedule,
                'enabled': True,
                'queue': 'intraday_queue'
            }
        )
        logger.info("调度器：盘中引擎已设置为每分钟运行。")
        return {"status": "started"}
        
    elif action == 'stop':
        logger.info("调度器：正在停止盘中引擎...")
        try:
            task = PeriodicTask.objects.get(name=task_name)
            task.enabled = False
            task.save()
            logger.info("调度器：盘中引擎的定时任务已禁用。")
            return {"status": "stopped"}
        except PeriodicTask.DoesNotExist:
            logger.warning(f"未找到名为 '{task_name}' 的定时任务，无需停止。")
            return {"status": "not_found"}
    else:
        return {"status": "error", "reason": "Invalid action"}
