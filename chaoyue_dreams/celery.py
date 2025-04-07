"""
Celery配置文件
使用Redis作为消息代理和结果后端
"""
import os
from celery import Celery

# 设置默认Django设置模块
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')

# 创建Celery应用
app = Celery('chaoyue_dreams')

# 使用Django的settings中的CELERY_前缀配置来配置Celery
app.config_from_object('django.conf:settings', namespace='CELERY')

# 延迟自动发现任务，确保Django已完全加载
# 仅在指定包内查找任务，并且命名为tasks.py的模块
app.autodiscover_tasks(lambda: ['tasks'])

# 设置worker进程数（根据服务器CPU核心数调整）
app.conf.worker_concurrency = 8  # 或更多，取决于您的服务器资源

# 启用预取限制，避免任务分配不均
app.conf.worker_prefetch_multiplier = 1

# 为任务设置超时
app.conf.task_time_limit = 1800  # 30分钟

# 调试任务
@app.task(bind=True)
def debug_task(self):
    """
    用于调试Celery配置的任务
    """
    print(f'请求: {self.request!r}') 