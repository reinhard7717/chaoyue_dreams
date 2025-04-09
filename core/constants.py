# core/constants.py
from enum import Enum

# 斐波那契数列周期
FIB_PERIODS = [5, 8, 13, 21, 34, 55, 89, 144, 233]

# 时间级别枚举 (如果还没有的话)
class TimeLevel(Enum):
    MIN_1 = '1m'
    MIN_5 = '5m'
    MIN_15 = '15m'
    MIN_30 = '30m'
    MIN_60 = '60m'
    DAY = '1d'
    WEEK = '1w'
    MONTH = '1M'
    YEAR = '1y'
    # ... 其他需要的级别

# finta 需要的列名
FINTA_OHLCV_MAP = {
    'open_price': 'open',
    'high_price': 'high',
    'low_price': 'low',
    'close_price': 'close',
    'volume': 'volume',
    'turnover': 'turnover', # 保留成交额用于计算
    'trade_time': 'trade_time' # 用于设置索引
}

TIME_TEADE_TIME_LEVELS = ['5','15','30','60','Day','Day_qfq','Day_hfq','Week','Week_qfq','Week_hfq','Month','Month_qfq','Month_hfq','Year','Year_qfq','Year_hfq']

TIME_TEADE_TIME_LEVELS_LITE = ['5','15','30','60','Day','Week','Month','Year']

TIME_TEADE_TIME_LEVELS_PER_TRADE_HOURS = ['5','15','30','60','Day']
