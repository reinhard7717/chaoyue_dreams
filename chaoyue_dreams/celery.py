"""
Celery配置文件
使用Redis作为消息代理和结果后端
"""
import os
from celery import Celery, signals

# 设置默认Django设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')

# 创建Celery应用
app = Celery('chaoyue_dreams')

# 使用Django的settings中的CELERY_前缀配置来配置Celery
app.config_from_object('django.conf:settings', namespace='CELERY')

# 延迟自动发现任务，确保Django已完全加载
# 仅在指定包内查找任务，并且命名为tasks.py的模块
# 使用 include 显式指定包含任务的模块 (推荐)
app.conf.update(
    # worker_concurrency = 16, # 保留你原来的设置
    # worker_prefetch_multiplier = 1, # 保留你原来的设置
    # task_time_limit = 1800, # 保留你原来的设置
    # worker_hijack_root_logger=False, # 保留你原来的设置
    # worker_log_color=False, # 保留你原来的设置

    # ****** 添加 include 配置 ******
    include=[
        'tasks.datacenter_tasks',
        'tasks.index_tasks',
        'tasks.stock_tasks',
        'tasks.stock_indicator_tasks',
        'tasks.strategy_tasks',      # <<<--- 添加包含策略任务的模块
        # 如果还有其他文件包含 Celery 任务，也一并添加到这里
        # 例如: 'tasks.other_tasks'
    ]
)

# 设置worker进程数（根据服务器CPU核心数调整）
app.conf.worker_concurrency = 16  # 或更多，取决于您的服务器资源

# 启用预取限制，避免任务分配不均
app.conf.worker_prefetch_multiplier = 1

# 为任务设置超时
app.conf.task_time_limit = 1800  # 30分钟

# 配置Celery日志
app.conf.update(
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# 导入Django设置后手动配置Celery日志
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    from django.conf import settings
    import logging.config
    logging.config.dictConfig(settings.LOGGING)


# 调试任务
@app.task(bind=True)
def debug_task(self):
    """
    用于调试Celery配置的任务
    """
    print(f'请求: {self.request!r}') 