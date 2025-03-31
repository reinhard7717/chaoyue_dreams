"""
专门为Windows环境优化的Celery配置
"""
import os
from celery import Celery

# 设置默认Django设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')

# 创建Celery应用
app = Celery('chaoyue_dreams')

# 使用Django的settings中的CELERY_前缀配置来配置Celery
app.config_from_object('django.conf:settings', namespace='CELERY')

# Windows环境专用优化配置
app.conf.update(
    worker_pool='solo',  # 使用solo池，Windows不支持prefork
    worker_concurrency=1,  # Windows下使用单进程
    broker_heartbeat=None,  # 禁用心跳（避免Windows上的问题）
    broker_pool_limit=1,  # 限制连接池（避免Windows上的问题）
    worker_prefetch_multiplier=1,  # 禁用预取，避免Windows内存问题
    task_acks_late=False,  # Windows下不延迟确认
)

# 延迟自动发现任务，确保Django已完全加载
# 仅在指定包内查找任务，并且命名为tasks.py的模块
app.autodiscover_tasks(lambda: ['tasks'])

# 调试任务
@app.task(bind=True)
def debug_task(self):
    """
    用于调试Celery配置的任务
    """
    print(f'请求: {self.request!r}') 