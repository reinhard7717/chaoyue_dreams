# chaoyue_dreams/chaoyue_dreams/settings.py

import os
from pathlib import Path

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-your-secret-key-here'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = []

# Application definition
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'api_manager',
    'dao_manager',
    'stock_models',
    'utils',
    'users.apps.UsersConfig',  # 添加用户应用
    'tasks',  # 添加任务应用
]

# 自定义用户模型 - 暂时注释掉，等数据库初始化完成后再启用
# AUTH_USER_MODEL = 'users.User'

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

# 数据库配置
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'beyond_dreams',
        'USER': 'stocker',
        'PASSWORD': 'Asdf+1234',
        'HOST': '39.101.65.133',
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",
        },
    }
}

# Redis缓存设置
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://39.101.65.133:6379/0',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'SERIALIZER': 'django_redis.serializers.json.JSONSerializer',  # 使用JSON序列化
            'PASSWORD': 'Asdf1234',  # Redis密码
            'SOCKET_CONNECT_TIMEOUT': 5,
            'SOCKET_TIMEOUT': 5,
            'RETRY_ON_TIMEOUT': True,
            'MAX_CONNECTIONS': 1000,
            'CONNECTION_POOL_KWARGS': {'max_connections': 100},
        },
        'TIMEOUT': 300,  # 默认缓存超时时间，单位：秒
    }
}

# 语言和时区设置
LANGUAGE_CODE = 'zh-hans'
TIME_ZONE = 'Asia/Shanghai'
USE_I18N = True
USE_TZ = True

# 静态文件设置
STATIC_URL = 'static/'
STATICFILES_DIRS = [
    os.path.join(BASE_DIR, 'static'),
]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# 媒体文件设置
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# 默认主键字段类型
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# 登录设置
LOGIN_URL = '/users/login/'
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/users/login/'

# API配置
API_BASE_URL = 'https://ig507.com'
API_LICENCES = [
    '6AEE029A-44A4-3404-A405-FB2C20085521',
    '03B3418F-04F6-5C44-A700-A245F74B8D77',
    # 添加更多licence
]
API_REQUEST_TIMEOUT = 10  # 请求超时时间(秒)
API_RATE_LIMIT = {
    'default': {'times': 10, 'seconds': 60},  # 默认每分钟10次
    'professional': {'times': 50, 'seconds': 1},  # 专业版每秒50次
}


# API缓存超时设置（秒）
API_CACHE_TIMEOUTS = {
    'basic': 86400,  # 基础数据缓存24小时
    'market': 60,  # 市场数据缓存1分钟
    'technical': 300,  # 技术指标缓存5分钟
    'fund_flow': 60,  # 资金流向缓存1分钟
    'index': 60,  # 指数数据缓存1分钟
    'realtime': 5,  # 实时数据缓存5秒
    'default': 300,  # 默认缓存5分钟
}

# API请求设置
API_REQUEST_SETTINGS = {
    'timeout': 30,  # 总超时时间（秒）
    'connect_timeout': 10,  # 连接超时时间（秒）
    'max_connections': 100,  # 最大连接数
    'dns_cache_ttl': 300,  # DNS缓存时间（秒）
    'max_retries': 3,  # 最大重试次数
    'base_retry_delay': 1,  # 基础重试延迟（秒）
    'max_retry_delay': 30,  # 最大重试延迟（秒）
}

# API错误处理设置
API_ERROR_SETTINGS = {
    'error_threshold': 5,  # 错误阈值
    'base_cooldown': 60,  # 基础冷却时间（秒）
    'max_cooldown': 300,  # 最大冷却时间（秒）
    'error_backoff': 2.0,  # 错误退避因子
}

API_URL_PATTERNS = {
    # 实时数据类型 - 更新频率高，对时效性要求高
    'realtime': [
        r'/data/time/real/',           # 实时交易数据
        r'/data/time/real/trace/',     # 买卖五档、逐笔交易等
        r'/data/all/pzyd',             # 盘中异动
        r'/data/time/real/time/',      # 分时交易数据
        r'/quotes/',                    # 实时报价
        r'/tick/',                      # 实时逐笔
        r'/min/',                       # 分钟线数据
    ],
    
    # 技术指标类型 - 计算密集型
    'technical': [
        r'/data/time/real/kdj/',       # KDJ指标
        r'/data/time/real/macd/',      # MACD指标
        r'/data/time/real/ma/',        # 移动平均线
        r'/data/time/real/boll/',      # 布林带
        r'/data/time/history/kdj/',    # 历史KDJ
        r'/data/time/history/macd/',   # 历史MACD
        r'/data/time/history/ma/',     # 历史MA
        r'/data/time/history/boll/',   # 历史BOLL
        r'/indicators/',               # 通用技术指标
    ],
    
    # 资金流向类型 - 资金相关数据
    'fund_flow': [
        r'/data/time/zijin/',          # 资金流向相关接口
        r'/data/time/zdtgc/',          # 涨跌停、强势股池等
        r'/data/all/ld',               # 每日龙虎榜详情
        r'/data/all/gg/',              # 个股上榜统计
        r'/data/all/yyb/',             # 营业部上榜统计
        r'/data/all/jgzz/',            # 机构席位追踪
        r'/data/all/jgcj',             # 机构席位成交明细
    ],
    
    # 市场数据类型 - 市场整体数据
    'market': [
        r'/data/all/jdgd',             # 阶段最高最低
        r'/data/all/cxg',              # 盘中创新高个股
        r'/data/all/cxd',              # 盘中创新低个股
        r'/data/all/cjzz',             # 成交骤增个股
        r'/data/all/cjzj',             # 成交骤减个股
        r'/data/all/lxfl',             # 连续放量个股
        r'/data/all/lxsl',             # 连续缩量个股
        r'/data/all/lxsz',             # 连续上涨个股
        r'/data/all/lxxd',             # 连续下跌个股
        r'/data/all/zzdpm',            # 周涨跌排名
        r'/market/',                   # 市场概览
        r'/sector/',                   # 板块数据
        r'/industry/',                 # 行业数据
    ],
    
    # 指数数据类型
    'index': [
        r'/data/base/shsz',            # 沪深主要指数列表
        r'/data/base/sh',              # 沪市指数列表
        r'/data/base/sz',              # 深市指数列表
        r'/data/time/real/shszzdbl',   # 沪深两市上涨下跌数概览
        r'/index/',                    # 指数数据
        r'/indices/',                  # 指数数据
    ],
    
    # 基础数据类型 - 更新频率低，数据稳定
    'basic': [
        r'/data/base/gplist',          # 股票列表
        r'/data/all/xgrl',             # 新股日历
        r'/data/all/stgplist',         # 风险警示股票列表
        r'/data/time/f10/',            # 公司F10信息
        r'/basic/',                    # 基础数据
        r'/profile/',                  # 公司简介
        r'/company/',                  # 公司信息
        r'/summary/',                  # 摘要信息
    ],
}


# API频率限制设置
API_RATE_LIMITS = {
    'realtime': {  # 实时数据API
        'basic': {
            'rate': 0.5,  # 每2秒1个请求
            'burst': 1,  # 不允许突发
        },
        'pro': {
            'rate': 40,  # 每秒40个请求
            'burst': 50,  # 允许突发到50个请求
        },
    },
    'basic': {  # 基础数据API
        'basic': {
            'rate': 0.5,  # 每2秒1个请求
            'burst': 1,
        },
        'pro': {
            'rate': 40,  # 每秒40个请求
            'burst': 50,
        },
    },
    'index': {  # 指数数据API
        'basic': {
            'rate': 0.5,  # 每2秒1个请求
            'burst': 1,
        },
        'pro': {
            'rate': 40,  # 每秒40个请求
            'burst': 50,
        },
    },
    'market': {  # 市场数据API
        'basic': {
            'rate': 0.5,  # 每2秒1个请求
            'burst': 1,
        },
        'pro': {
            'rate': 0.167,  # 每6秒1个请求 - 根据API文档，市场数据请求频率限制为每分钟10次
            'burst': 2,  # 允许一定程度的突发
        },
    },
    'fund_flow': {  # 资金流向API
        'basic': {
            'rate': 0.5,  # 每2秒1个请求
            'burst': 1,
        },
        'pro': {
            'rate': 40,  # 每秒40个请求
            'burst': 50,
        },
    },
    'technical': {  # 技术指标API
        'basic': {
            'rate': 0.5,  # 每2秒1个请求
            'burst': 1,
        },
        'pro': {
            'rate': 3,  # 每秒3个请求 - 根据API文档，技术指标API请求频率限制为每秒3次
            'burst': 5,  # 允许一定程度的突发，但历史数据请求限制为每秒3次
        },
    },
    'default': {  # 默认限制
        'basic': {
            'rate': 0.167,  # 每6秒1个请求
            'burst': 1,
        },
        'pro': {
            'rate': 0.167,  # 每6秒1个请求
            'burst': 1,
        },
    },
}

# API自动更新时间设置（秒）
API_UPDATE_INTERVALS = {
    'realtime': 3,  # 实时数据每3秒更新一次
    'basic': 86400,  # 基础数据每24小时更新一次
    'index': 60,  # 指数数据每1分钟更新一次
    'market': 60,  # 市场数据每1分钟更新一次
    'fund_flow': 60,  # 资金流向每1分钟更新一次
    'technical': 60,  # 技术指标每1分钟更新一次
    'default': 300,  # 默认每5分钟更新一次
}

# API数据更新时间范围
API_UPDATE_TIME_RANGE = {
    'start_time': '09:15:00',  # 开始更新时间
    'end_time': '15:30:00',  # 结束更新时间
    'trading_days': [0, 1, 2, 3, 4],  # 周一到周五
}

# 技术指标计算设置
TECHNICAL_INDICATOR_SETTINGS = {
    'MA_PERIODS': [5, 10, 20, 60],
    'MACD_SETTINGS': {
        'FAST': 12,
        'SLOW': 26,
        'SIGNAL': 9,
    },
    'KDJ_SETTINGS': {
        'WINDOW': 9,
    },
    'RSI_SETTINGS': {
        'WINDOW': 14,
    },
    'BOLL_SETTINGS': {
        'WINDOW': 20,
        'STD_DEV': 2,
    },
}

# 策略参数设置
STRATEGY_SETTINGS = {
    'INTRADAY_SETTINGS': {
        'MIN_PROFIT_PCT': 0.02,  # 最小获利百分比
        'MAX_LOSS_PCT': 0.01,    # 最大亏损百分比
        'HOLDING_PERIODS': [5, 15, 30, 60],  # 分钟
    },
    'SWING_SETTINGS': {
        'MIN_TREND_STRENGTH': 0.7,  # 最小趋势强度
        'REVERSAL_THRESHOLD': 0.3,  # 反转阈值
        'HOLDING_PERIODS': [1, 3, 5, 10],  # 天
    },
    'POSITION_SETTINGS': {
        'MAX_POSITION': 0.3,     # 最大仓位
        'MIN_POSITION': 0.1,     # 最小仓位
        'POSITION_STEP': 0.1,    # 仓位步长
    },
}

# 日志配置
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
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
        },
    },
    'loggers': {
        'api': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}

# 确保日志目录存在
if not os.path.exists(os.path.join(BASE_DIR, 'logs')):
    os.makedirs(os.path.join(BASE_DIR, 'logs'))
