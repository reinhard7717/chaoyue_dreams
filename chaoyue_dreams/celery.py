# 文件: chaoyue_dreams/celery.py

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
    accept_content=['pickle', 'json'],
    task_serializer='pickle',
    result_serializer='pickle',
    # ****** 包含的任务模块列表 ******
    include=[
        'tasks.tushare.stock_time_trade_tasks',
        'tasks.tushare.stock_tasks',
        'tasks.tushare.stock_realtime_tasks',
        'tasks.stock_analysis_tasks',
        'tasks.tushare.fund_flow_tasks',
        'tasks.tushare.index_tasks',
        'tasks.tushare.train_transformer_tasks',
        'tasks.tushare.industry_tasks',
        'tasks.tushare.cal_daily_tasks',
        'dashboard.tasks'
    ],
    # ****** 其他性能和健壮性配置 ******
    worker_concurrency=5,                 # Worker并发数
    worker_prefetch_multiplier=1,          # Worker预取因子 (设置为1，结合acks_late=True，确保worker完成当前任务后再取下一个)
    task_time_limit=14400,                  # 任务硬超时时间 (秒)，当前为4小时。
    task_acks_late=False,                   # 任务执行成功完成后才向Broker发送确认信号。
    broker_transport_options={             # 配置Broker（Redis）的传输选项。
        'visibility_timeout': 14400         # 任务可见性超时时间 (秒)，应大于等于task_time_limit。
    },
    worker_hijack_root_logger=False,       # 避免Celery劫持根日志记录器
    worker_log_color=True,                # 在Windows上或不需要颜色日志时可以禁用
)

# 导入Django设置后手动配置Celery日志
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    from django.conf import settings
    import logging.config
    logging.config.dictConfig(settings.LOGGING)

