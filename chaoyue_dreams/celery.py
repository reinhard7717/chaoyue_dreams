import os
from celery import Celery
from celery.schedules import crontab
from django.conf import settings

# 设置环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'chaoyue_dreams.settings')

# 创建Celery应用
app = Celery('chaoyue_dreams')

# 使用Django的settings配置Celery
app.config_from_object('django.conf:settings', namespace='CELERY')

# 自动从已注册的Django应用中加载任务
app.autodiscover_tasks()

# 定义定时任务
app.conf.beat_schedule = {
    # 指数相关定时任务
    'refresh_indexes_daily': {
        'task': 'tasks.index_tasks.refresh_indexes',
        'schedule': crontab(hour=8, minute=0),  # 每天早上8点
        'args': (),
    },
    'refresh_index_realtime_data': {
        'task': 'tasks.index_tasks.refresh_main_indexes_realtime_data',
        'schedule': crontab(minute='*/1', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每分钟
        'args': (),
    },
    'refresh_market_overview': {
        'task': 'tasks.index_tasks.refresh_market_overview',
        'schedule': crontab(minute='*/2', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每2分钟
        'args': (),
    },
    'refresh_5min_time_series': {
        'task': 'tasks.index_tasks.refresh_main_indexes_time_series',
        'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每5分钟
        'args': ('5',),
    },
    'refresh_15min_time_series': {
        'task': 'tasks.index_tasks.refresh_main_indexes_time_series',
        'schedule': crontab(minute='*/15', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每15分钟
        'args': ('15',),
    },
    'refresh_30min_time_series': {
        'task': 'tasks.index_tasks.refresh_main_indexes_time_series',
        'schedule': crontab(minute='*/30', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每30分钟
        'args': ('30',),
    },
    'refresh_60min_time_series': {
        'task': 'tasks.index_tasks.refresh_main_indexes_time_series',
        'schedule': crontab(minute='0', hour='10,11,13,14,15', day_of_week='1-5'),  # 交易时间每小时
        'args': ('60',),
    },
    'refresh_day_time_series': {
        'task': 'tasks.index_tasks.refresh_main_indexes_time_series',
        'schedule': crontab(hour=15, minute=45, day_of_week='1-5'),  # 每个交易日15:45
        'args': ('Day',),
    },
    'refresh_day_technical_indicators': {
        'task': 'tasks.index_tasks.refresh_main_indexes_technical_indicators',
        'schedule': crontab(hour=16, minute=0, day_of_week='1-5'),  # 每个交易日16:00
        'args': ('Day',),
    },

    # 资金流向相关定时任务
    'refresh_fund_flow_daily': {
        'task': 'tasks.fund_flow_tasks.refresh_popular_stocks_fund_flow',
        'schedule': crontab(hour=20, minute=30),  # 每天20:30
        'args': (),
    },
    'refresh_fund_flow_minute': {
        'task': 'tasks.fund_flow_tasks.refresh_active_stocks_fund_flow_minute',
        'schedule': crontab(hour=20, minute=45),  # 每天20:45
        'args': (),
    },

    # 股票池相关定时任务
    'refresh_limit_pools': {
        'task': 'tasks.stock_pool_tasks.refresh_daily_limit_pools',
        'schedule': crontab(minute='*/10', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每10分钟
        'args': (),
    },
    'refresh_strong_stocks': {
        'task': 'tasks.stock_pool_tasks.refresh_daily_strong_stocks',
        'schedule': crontab(minute='*/10', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每10分钟
        'args': (),
    },
}

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
