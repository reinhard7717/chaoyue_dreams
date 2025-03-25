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
    'refresh_sector_fund_flow': {
        'task': 'tasks.fund_flow_tasks.refresh_sector_fund_flow',
        'schedule': crontab(hour=16, minute=30, day_of_week='1-5'),  # 每个交易日16:30
        'args': (),
    },
    'refresh_market_main_force_phase': {
        'task': 'tasks.fund_flow_tasks.refresh_market_main_force_phase',
        'schedule': crontab(hour=16, minute=45, day_of_week='1-5'),  # 每个交易日16:45
        'args': (),
    },
    'refresh_stock_transaction_distribution': {
        'task': 'tasks.fund_flow_tasks.refresh_stock_transaction_distribution',
        'schedule': crontab(hour=17, minute=0, day_of_week='1-5'),  # 每个交易日17:00
        'args': (),
    },
    'refresh_north_south_fund_flow': {
        'task': 'tasks.fund_flow_tasks.refresh_north_south_fund_flow',
        'schedule': crontab(hour='10,11,14,15', minute=0, day_of_week='1-5'),  # 交易时间每小时
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
    'refresh_break_limit_pools': {
        'task': 'tasks.stock_pool_tasks.refresh_break_limit_pools',
        'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每5分钟
        'args': (),
    },
    'refresh_new_stock_pools': {
        'task': 'tasks.stock_pool_tasks.refresh_new_stock_pools',
        'schedule': crontab(hour=9, minute=0, day_of_week='1'),  # 每周一早上9:00
        'args': (),
    },
    'refresh_concept_top_stocks': {
        'task': 'tasks.stock_pool_tasks.refresh_concept_top_stocks',
        'schedule': crontab(hour=16, minute=15, day_of_week='1-5'),  # 每个交易日16:15
        'args': (),
    },
    'refresh_industry_top_stocks': {
        'task': 'tasks.stock_pool_tasks.refresh_industry_top_stocks',
        'schedule': crontab(hour=16, minute=20, day_of_week='1-5'),  # 每个交易日16:20
        'args': (),
    },

    # 股票相关定时任务
    'refresh_stock_basic_info': {
        'task': 'tasks.stock_tasks.refresh_stock_basic_info',
        'schedule': crontab(hour=7, minute=0),  # 每天早上7点
        'args': (),
    },
    'refresh_stock_industry_info': {
        'task': 'tasks.stock_tasks.refresh_stock_industry_info',
        'schedule': crontab(hour=7, minute=30, day_of_week='1'),  # 每周一早上7:30
        'args': (),
    },
    'refresh_stock_concept_info': {
        'task': 'tasks.stock_tasks.refresh_stock_concept_info',
        'schedule': crontab(hour=8, minute=0, day_of_week='1'),  # 每周一早上8:00
        'args': (),
    },
    'refresh_favorites_realtime_data': {
        'task': 'tasks.stock_tasks.refresh_favorites_realtime_data',
        'schedule': crontab(minute='*/1', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每分钟
        'args': (),
    },
    'refresh_active_stocks_realtime_data': {
        'task': 'tasks.stock_tasks.refresh_active_stocks_realtime_data',
        'schedule': crontab(minute='*/2', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每2分钟
        'args': (),
    },
    'refresh_stock_level5_data': {
        'task': 'tasks.stock_tasks.refresh_stock_level5_data',
        'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每5分钟
        'args': (),
    },
    'refresh_stock_1min_time_series': {
        'task': 'tasks.stock_tasks.refresh_stock_time_series',
        'schedule': crontab(minute='*/2', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每2分钟
        'args': ('1', None),
    },
    'refresh_stock_5min_time_series': {
        'task': 'tasks.stock_tasks.refresh_stock_time_series',
        'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每5分钟
        'args': ('5', None),
    },
    'refresh_stock_day_time_series': {
        'task': 'tasks.stock_tasks.refresh_stock_time_series',
        'schedule': crontab(hour=15, minute=50, day_of_week='1-5'),  # 每个交易日15:50
        'args': ('Day', None),
    },
    'refresh_stock_day_technical_indicators': {
        'task': 'tasks.stock_tasks.refresh_stock_technical_indicators',
        'schedule': crontab(hour=16, minute=10, day_of_week='1-5'),  # 每个交易日16:10
        'args': ('Day', None),
    },

    # 数据中心相关定时任务
    'refresh_financial_data': {
        'task': 'tasks.datacenter_tasks.refresh_financial_data',
        'schedule': crontab(hour=21, minute=0),  # 每天21:00
        'args': (),
    },
    'refresh_capital_flow_data': {
        'task': 'tasks.datacenter_tasks.refresh_capital_flow_data',
        'schedule': crontab(hour=21, minute=30),  # 每天21:30
        'args': (),
    },
    'refresh_lhb_data': {
        'task': 'tasks.datacenter_tasks.refresh_lhb_data',
        'schedule': crontab(hour=21, minute=45),  # 每天21:45
        'args': (),
    },
    'refresh_institution_data': {
        'task': 'tasks.datacenter_tasks.refresh_institution_data',
        'schedule': crontab(hour=22, minute=0, day_of_week='1'),  # 每周一22:00
        'args': (),
    },
    'refresh_north_south_data': {
        'task': 'tasks.datacenter_tasks.refresh_north_south_data',
        'schedule': crontab(hour=22, minute=15),  # 每天22:15
        'args': (),
    },
    'refresh_statistics_data': {
        'task': 'tasks.datacenter_tasks.refresh_statistics_data',
        'schedule': crontab(hour=22, minute=30),  # 每天22:30
        'args': (),
    },
    'refresh_market_data': {
        'task': 'tasks.datacenter_tasks.refresh_market_data',
        'schedule': crontab(hour=22, minute=45),  # 每天22:45
        'args': (),
    },

    # 策略相关定时任务
    'calculate_intraday_strategy': {
        'task': 'tasks.strategy_tasks.calculate_intraday_strategy',
        'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每5分钟
        'args': (),
    },
    'calculate_wave_tracking_strategy': {
        'task': 'tasks.strategy_tasks.calculate_wave_tracking_strategy',
        'schedule': crontab(hour=16, minute=5, day_of_week='1-5'),  # 每个交易日16:05
        'args': (),
    },
    'check_stock_reversal_morning': {
        'task': 'tasks.strategy_tasks.check_stock_reversal',
        'schedule': crontab(hour=8, minute=45, day_of_week='1-5'),  # 每个交易日开盘前8:45
        'args': (),
    },
    'check_stock_reversal_evening': {
        'task': 'tasks.strategy_tasks.check_stock_reversal',
        'schedule': crontab(hour=15, minute=35, day_of_week='1-5'),  # 每个交易日收盘后15:35
        'args': (),
    },
    'calculate_intraday_signals': {
        'task': 'tasks.strategy_tasks.calculate_intraday_signals',
        'schedule': crontab(minute='*/10', hour='9,10,11,13,14,15', day_of_week='1-5'),  # 交易时间每10分钟
        'args': (),
    },
    'calculate_daily_signals': {
        'task': 'tasks.strategy_tasks.calculate_daily_signals',
        'schedule': crontab(hour=15, minute=40, day_of_week='1-5'),  # 每个交易日收盘后15:40
        'args': (),
    },
    'calculate_market_signals': {
        'task': 'tasks.strategy_tasks.calculate_market_signals',
        'schedule': crontab(hour=15, minute=45, day_of_week='1-5'),  # 每个交易日收盘后15:45
        'args': (),
    },
}

@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
