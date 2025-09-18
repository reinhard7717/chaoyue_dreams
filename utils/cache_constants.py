# your_app/cache_constants.py

# --- Cache Types ---
TYPE_STATIC = 'st'
TYPE_REALTIME = 'rt'
TYPE_TIMESERIES = 'ts'
TYPE_APP = "app"
TYPE_CALCULATION = 'calc'
TYPE_USER = 'user'
TYPE_STRATEGY = 'strategy'

# --- Entity Types ---
ENTITY_INDEX = 'index'
ENTITY_STOCK = 'stock'
ENTITY_MARKET = 'market'
ENTITY_STRATEGY = 'strategy'
ENTITY_USERINFO = 'user_info'
ENTITY_USERFAVORITES = 'favorite_stocks'
ENTITY_POOL = "pool"
# ... 其他实体类型

# --- Entity IDs / Subtypes / Identifiers ---
ID_ALL = 'all'
ID_INTRADAY_MONITORING = "intraday_monitoring"
ID_OVERVIEW = 'overview'
ID_TIME_SERIES = 'time_series'
ID_REALTIME = 'realtime'

SUBTYPE_BASIC_INFO = "basic"
SUBTYPE_QUOTE = 'quote'
SUBTYPE_KLINE = 'kline'
SUBTYPE_BASIC_INFO = 'info'
SUBTYPE_MACD = 'macd'
SUBTYPE_KDJ = 'kdj'
SUBTYPE_RSI = 'rsi'
SUBTYPE_BOLL = 'boll'
SUBTYPE_MA = 'ma'
SUBTYPE_REAL_PERCENT = 'real_percent'
SUBTYPE_BIG_DEAL = 'big_deal'
SUBTYPE_ABNORMAL_MOVEMENT = 'abnormal_movement'
SUBTYPE_TIME_DEAL = 'time_deal'
SUBTYPE_LEVEL5 = 'level5'
SUBTYPE_KLINE_MINUTE = 'kline_minute'
SUBTYPE_STRATEGY_TREND_FOLLOWING = 'trend_following'
SUBTYPE_CONCEPTS = 'concepts'

# ... 其他子类型或固定标识符

# --- Parameter Keys (如果 generate_key 使用 params) ---
PARAM_PERIOD = 'time_level'
PARAM_DATE = 'date'
# ... 其他参数名
