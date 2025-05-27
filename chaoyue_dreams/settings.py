# chaoyue_dreams/chaoyue_dreams/settings.py

import os
import socket # <--- 导入 socket 模块
from pathlib import Path
from celery.schedules import crontab
from kombu import Queue
from datetime import timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-here'
# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
ALLOWED_HOSTS = []
# --- 开始: 动态获取本机IP并设置Redis主机 ---
def get_local_ip():
    """尝试获取本机的主要出站IP地址"""
    s = None
    try:
        # 连接到一个外部地址（不需要实际发送数据）来确定本机用于出站连接的IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 使用一个不太可能无法访问的公共DNS服务器
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        # 如果获取失败，尝试使用hostname，或者回退到localhost
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            ip = '127.0.0.1' # 最终回退
    finally:
        if s:
            s.close()
    return ip

SERVER_IP = get_local_ip()
TARGET_SERVER_IP = "39.101.65.133"
REDIS_PASSWORD = 'Asdf1234' # 将密码定义在这里，方便复用
REDIS_PORT = '6379'
BASE_DIR = Path(__file__).resolve().parent.parent

if SERVER_IP == TARGET_SERVER_IP or SERVER_IP == '172.30.93.156':
    REDIS_HOST_DYNAMIC = '127.0.0.1'
    MYSQL_HOST_DYNAMIC = '127.0.0.1'
    STRATEGY_DATA_DIR = '/data/chaoyue_dreams/models'
    print(f"检测到服务器IP为 {SERVER_IP}，Redis Host 设置为: 127.0.0.1，STRATEGY_DATA_DIR：{STRATEGY_DATA_DIR}")
else:
    REDIS_HOST_DYNAMIC = TARGET_SERVER_IP
    MYSQL_HOST_DYNAMIC = TARGET_SERVER_IP
    STRATEGY_DATA_DIR = str(BASE_DIR / 'models')
    print(f"检测到服务器IP为 {SERVER_IP} (非 {TARGET_SERVER_IP})，Redis Host 设置为: {TARGET_SERVER_IP}，STRATEGY_DATA_DIR：{STRATEGY_DATA_DIR}")

# --- 结束: 动态获取本机IP并设置Redis主机 ---

# Application definition
INSTALLED_APPS = [
    'daphne', # 必须放在 'django.contrib.admin' 等之前
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework', # 添加 DRF
    'channels',       # 添加 Channels
    'users',          # 你的用户 app
    'stock_models',   # 你的模型 app
    'dashboard',      # 新创建的主控台 app
    'api_manager',
    'dao_manager',
    'utils',
    'core',
    'strategies',
    'tasks',  # 添加任务应用
    'django_celery_results',  # 添加Celery结果存储应用
    'django_celery_beat',  # 添加Celery定时任务应用
    'bulk_update_or_create',
]

# 自定义用户模型 - 暂时注释掉，等数据库初始化完成后再启用
# AUTH_USER_MODEL = 'users.User'
LOGIN_URL = '/users/login/'
LOGIN_REDIRECT_URL = '/dashboard/' # 登录后重定向到主控台
LOGOUT_REDIRECT_URL = '/'

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'chaoyue_dreams.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],  # 添加全局模板目录
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'chaoyue_dreams.wsgi.application'

# --- Channels 配置 ---
ASGI_APPLICATION = 'chaoyue_dreams.asgi.application' # 指定 ASGI 入口点

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels_redis.core.RedisChannelLayer',
        'CONFIG': {
            "hosts": [f"redis://:{REDIS_PASSWORD}@{REDIS_HOST_DYNAMIC}:{REDIS_PORT}/1"],
        },
    },
}

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'beyond_dreams',
        'USER': 'stocker',
        'PASSWORD': 'Asdf+1234',
        'HOST': MYSQL_HOST_DYNAMIC,  # 数据库地址保持不变
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'; SET SESSION wait_timeout=86400;",  # 添加wait_timeout设置为2400秒
        },
        'CONN_MAX_AGE': 86400,  # 如上所述
    }
}

# Redis缓存设置
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        # 使用动态获取的 Redis 主机地址
        'LOCATION': f'redis://{REDIS_HOST_DYNAMIC}:{REDIS_PORT}/0',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'SERIALIZER': 'utils.custom_serializer.CustomJSONSerializer',
            'PASSWORD': REDIS_PASSWORD, # 使用定义的密码变量
            'SOCKET_CONNECT_TIMEOUT': 10,  # 从5秒增加到10秒
            'SOCKET_TIMEOUT': 15,  # 从5秒增加到15秒
            'RETRY_ON_TIMEOUT': True,
            'MAX_CONNECTIONS': 100,
            # 新增重试策略配置
            'RETRY_TIMES': 3,  # 重试3次
            'RETRY_DELAY': 0.5,  # 重试间隔0.5秒.
            # 添加连接池配置
            'CONNECTION_POOL_KWARGS': {
                'max_connections': 2000,
                'socket_timeout': 20,
                'socket_connect_timeout': 10,  # 连接超时
                'socket_keepalive': True,      # 保持连接活跃
                'health_check_interval': 30,   # 定期健康检查
            },
            # 压缩配置（减少网络传输数据量）
            'COMPRESSOR': 'django_redis.compressors.zlib.ZlibCompressor',
            'COMPRESS_MIN_LEN': 10,  # 最小压缩长度
        },
        'TIMEOUT': 300,
    }
}

# --- DRF 配置 (基础) ---
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        # 根据需要配置认证方式，例如 Session 或 Token
        'rest_framework.authentication.SessionAuthentication',
        # 'rest_framework.authentication.TokenAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        # API 的默认权限，开始可以宽松点，后续收紧
        'rest_framework.permissions.IsAuthenticatedOrReadOnly',
        # 'rest_framework.permissions.IsAuthenticated',
    ]
}

# 语言和时区设置
LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder', # 这个查找器负责在 STATICFILES_DIRS 中查找
    'django.contrib.staticfiles.finders.AppDirectoriesFinder', # 这个查找器负责在 app 的 static 子目录中查找
]

# 静态文件设置
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'), # 使用 os.path.join 拼接字符串路径
]

# 配置 indicator_naming.json 的路径
INDICATOR_NAMING_CONFIG_PATH = str(BASE_DIR / 'config' / 'indicator_naming_conventions.json')
INDICATOR_PARAMETERS_CONFIG_PATH = str(BASE_DIR / 'config' / 'indicator_parameters.json')

# 默认主键字段类型
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# API配置
API_BASE_URL = 'http://ig507.com'  # 修改为http方式
API_LICENCES_IG507 = ['6AEE029A-44A4-3404-A405-FB2C20085521', '03B3418F-04F6-5C44-A700-A245F74B8D77', '6AEE029A-44A4-3404-A405-FB2C20085521', '03B3418F-04F6-5C44-A700-A245F74B8D77']
API_LICENCES_TUSHARE = '0793156bc63040ee46008f217c6e76c8b7c415e2748ac0a7bb509d2c'
API_REQUEST_TIMEOUT = 10  # 请求超时时间(秒)
API_RATE_LIMIT = {
    'default': {'times': 10, 'seconds': 60},  # 默认每分钟10次
    'professional': {'times': 50, 'seconds': 1},  # 专业版每秒50次
}

# 日志配置
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
        'simple': {
            'format': '%(asctime)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'api.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'api': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'api.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'dao': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'dao.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'services': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'services.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'celery': {
            'level': 'WARNING',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'celery.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5, # 备份5个文件
            'formatter': 'verbose', # 使用verbose格式
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
         'strategy': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'strategy.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5,
            'formatter': 'verbose',
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
         'tasks': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'tasks.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5, # 备份5个文件
            'formatter': 'verbose', # 使用verbose格式
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'strategy_trend_following': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'strategy_trend_following.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5, # 备份5个文件
            'formatter': 'verbose', # 使用verbose格式
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'strategy_deep_learning_utils': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'strategy_deep_learning_utils.log'),
            'maxBytes': 1024 * 1024 * 5,  # 5MB
            'backupCount': 5, # 备份5个文件
            'formatter': 'verbose', # 使用verbose格式
            'encoding': 'utf-8',  # 设置编码为utf-8
        },
        'evaluation_file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'evaluation_results.log',  # 日志文件路径
            'formatter': 'verbose',
            'encoding': 'utf-8',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
        'api': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'dao': {
            'handlers': ['console', 'dao'],
            'level': 'INFO',
            'propagate': False,
        },
        'celery.task': {
            'handlers': ['celery'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'celery.worker': {
            'handlers': ['celery'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'celery.app': {
            'handlers': ['celery'],
            'level': 'INFO',
            'propagate': False,
        },
        'services': {
            'handlers': ['console', 'services'],
            'level': 'INFO',
            'propagate': False,
        },
        'strategy': {
            'handlers': ['console', 'strategy'],
            'level': 'INFO',
            'propagate': False,
        },
        'tasks': {
            'handlers': ['console', 'tasks'],
            'level': 'DEBUG',
            'propagate': False,
        },
        'strategy_trend_following': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'strategy_deep_learning_utils': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        # 专门用于evaluation_results日志的logger
        'evaluation_results': {
            'handlers': ['evaluation_file'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# 确保日志目录存在
if not os.path.exists(os.path.join(BASE_DIR, 'logs')):
    os.makedirs(os.path.join(BASE_DIR, 'logs'))

# 指数相关缓存配置
INDEX_CACHE_TIMEOUT = {
    'index_list': 86400,  # 指数列表缓存1天
    'realtime_data': 60,  # 实时数据缓存1分钟
    'market_overview': 120,  # 市场概览缓存2分钟
    'time_series': 300,  # 分时数据缓存5分钟
    'technical_indicators': 300,  # 技术指标缓存5分钟
}

# Celery基础配置
# 使用动态获取的 Redis 主机地址和密码变量
CELERY_BROKER_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST_DYNAMIC}:{REDIS_PORT}/1'  # 使用Redis作为消息代理
CELERY_RESULT_BACKEND = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST_DYNAMIC}:{REDIS_PORT}/2'  # 使用Redis作为结果后端

# 定义队列
CELERY_TASK_QUEUES = (
    Queue('priority_tasks', routing_key='priority.#'), # 高优先级队列
    Queue('celery', routing_key='celery.#'),          # 默认队列 (假设你的默认队列是 'celery')
    Queue('SaveData_RealTime', routing_key='save_api_data_RealTime.#'), # 保存API数据队列
    Queue('SaveData_TimeTrade', routing_key='save_api_data_TimeTrade.#'), # 保存API数据队列
    Queue('favorite_SaveData_RealTime', routing_key='favorite_save_api_data_RealTime.#'), # 保存API数据队列
    Queue('favorite_SaveData_TimeTrade', routing_key='favorite_save_api_data_TimeTrade.#'), # 保存API数据队列
    Queue('Train_Transformer_Prepare_Data', routing_key='Train_Transformer_Prepare_Data.#'), # 机器学习 - 准备数据队列
    Queue('Train_Transformer_Model', routing_key='Train_Transformer_Model.#'), # 机器学习 - 训练队列
    Queue('calculate_strategy', routing_key='calculate_strategy.#'), # 计算股票指标队列
    Queue('favorite_calculate_strategy', routing_key='favorite_calculate_strategy.#'), # 计算股票指标队列
    Queue('dashboard', routing_key='dashboard.#'), # DRF专用队列
    # 如果你的默认队列有其他名字，比如 'default_tasks'，就这样定义:
    # Queue('default_tasks', routing_key='default.#'),
)


# Celery Beat配置
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'  # 使用数据库作为调度器
CELERY_BEAT_SCHEDULE = {
    ############# 任务：每 60 秒为 所有自选股 运行一次策略执行引擎 #############
    '每 5 秒运行一次所有股票的实时Tick数据获取': {
        'task': 'tasks.tushare.stock_realtime_tasks.save_stocks_tick_data_task',
        'schedule': timedelta(seconds=5),  # 每5秒执行一次
        'options': {'queue': 'celery'},  # 添加此行：指定队列名称，这是调度器的队列
    },
    # '每 1 分钟运行一次所有股票的K线数据获取任务': {
    #     # 这里包含了获得最新数据、计算指标、执行策略等步骤
    #     'task': 'tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', # 任务函数名
    #     'schedule': crontab(minute='*/1', hour='9,10,11,13,14,15', day_of_week='mon,tue,wed,thu,fri'), # 交易时段每 5 分钟执行
    #     'kwargs': {'time_level': '1'},
    #     'options': {'queue': 'celery'},  # 添加此行：指定队列名称，这是调度器的队列
    # },
    # '每 5 分钟运行一次所有股票的K线数据获取任务': {
    #     # 这里包含了获得最新数据、计算指标、执行策略等步骤
    #     'task': 'tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', # 任务函数名
    #     'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='mon,tue,wed,thu,fri'), # 交易时段每 5 分钟执行
    #     'kwargs': {'time_level': '5'},
    #     'options': {'queue': 'celery'},  # 添加此行：指定队列名称，这是调度器的队列
    # },
    # '每 15 分钟运行一次所有股票的K线数据获取任务': {
    #     # 这里包含了获得最新数据、计算指标、执行策略等步骤
    #     'task': 'tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', # 任务函数名
    #     'schedule': crontab(minute='*/15', hour='9,10,11,13,14,15', day_of_week='mon,tue,wed,thu,fri'), # 交易时段每 5 分钟执行
    #     'kwargs': {'time_level': '15'},
    #     'options': {'queue': 'celery'},  # 添加此行：指定队列名称，这是调度器的队列
    # },
    # '每 30 分钟运行一次所有股票的K线数据获取任务': {
    #     # 这里包含了获得最新数据、计算指标、执行策略等步骤
    #     'task': 'tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', # 任务函数名
    #     'schedule': crontab(minute='*/30', hour='9,10,11,13,14,15', day_of_week='mon,tue,wed,thu,fri'), # 交易时段每 5 分钟执行
    #     'kwargs': {'time_level': '30'},
    #     'options': {'queue': 'celery'},  # 添加此行：指定队列名称，这是调度器的队列
    # },
    # '每 60 分钟运行一次所有股票的K线数据获取任务': {
    #     # 这里包含了获得最新数据、计算指标、执行策略等步骤
    #     'task': 'tasks.tushare.stock_realtime_tasks.save_stocks_minute_data_realtime_task', # 任务函数名
    #     'schedule': crontab(minute='*/60', hour='9,10,11,13,14,15', day_of_week='mon,tue,wed,thu,fri'), # 交易时段每 5 分钟执行
    #     'kwargs': {'time_level': '60'},
    #     'options': {'queue': 'celery'},  # 添加此行：指定队列名称，这是调度器的队列
    # },
    # '每 5 分钟运行一次所有股票的策略计算任务': {
    #     # 这里包含了获得最新数据、计算指标、执行策略等步骤
    #     'task': 'tasks.stock_analysis.analyze_all_stocks', # 任务函数名
    #     'schedule': crontab(minute='*/5', hour='9,10,11,13,14,15', day_of_week='mon,tue,wed,thu,fri'), # 交易时段每 5 分钟执行
    # },
    '每天运行一次保存股票列表数据任务': {
        'task': 'tasks.tushare.cal_daily_tasks.run_daily_data_ingestion_task',
        'schedule': crontab(minute=0, hour=19, day_of_week='mon,tue,wed,thu,fri'),  # 每天凌晨1点执行
        'options': {'queue': 'celery'}, # 指定队列为 celery
    },
    # '每天运行一次保存股票列表数据任务': {
    #     'task': 'tasks.tushare.stock_time_trade_tasks.save_stocks_daily_basic_data_today_task',
    #     'schedule': crontab(minute=30, hour=17, day_of_week='mon,tue,wed,thu,fri'),  # 每天凌晨1点执行
    #     'options': {'queue': 'celery'}, # 指定队列为 celery
    # },
}

# 结果存储设置
CELERY_RESULT_EXPIRES = 86400  # 结果过期时间（秒）
CELERY_RESULT_EXTENDED = True  # 扩展的结果信息
CELERY_RESULT_COMPRESSION = 'gzip'  # 结果压缩算法
CELERY_RESULT_DB_TABLENAMES = {  # 自定义结果表名
    'task': 'celery_tasks',
    'group': 'celery_groups',
}

# Celery监控和日志设置
CELERY_SEND_TASK_SENT_EVENT = True  # 发送任务发送事件
CELERY_SEND_EVENTS = True  # 发送worker事件
CELERY_EVENT_QUEUE_TTL = 60  # 事件队列TTL
CELERY_EVENT_QUEUE_EXPIRES = 60  # 事件队列过期时间
CELERY_EVENT_SERIALIZER = 'json'  # 事件序列化格式

# Celery安全设置
CELERY_SECURITY_KEY = os.path.join(BASE_DIR, 'private.key')  # 安全密钥
CELERY_SECURITY_CERTIFICATE = os.path.join(BASE_DIR, 'public.crt')  # 安全证书
CELERY_SECURITY_CERT_STORE = os.path.join(BASE_DIR, 'ca.crt')  # 证书存储

# Celery应用名称
CELERY_APP_NAME = 'chaoyue_dreams'
