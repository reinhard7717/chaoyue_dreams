# core/constants.py
from enum import Enum

# 斐波那契数列周期
FIB_PERIODS = [5, 8, 13, 21, 34, 55, 89, 144, 233]

# 时间级别枚举 (如果还没有的话)
class TimeLevel(Enum):
    MIN_1 = '1'
    MIN_5 = '5'
    MIN_15 = '15'
    MIN_30 = '30'
    MIN_60 = '60'
    DAY = 'Day'
    WEEK = 'Week'
    MONTH = 'Month'
    YEAR = 'Year'
    # ... 其他需要的级别

# finta 需要的列名
FINTA_OHLCV_MAP = {
    'open': 'open', # 假设原始就是 open，映射为 open
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'vol': 'volume',       # <-- 确保这一条存在！将原始的 'vol' 映射为标准的 'volume'
    'amount': 'amount',    # 假设原始是 amount
    'trade_time': 'trade_time', # 时间列可能也需要标准化，但通常单独处理
}

TIME_TEADE_TIME_LEVELS = ['5','15','30','60','Day','Day_qfq','Day_hfq','Week','Week_qfq','Week_hfq','Month','Month_qfq','Month_hfq','Year','Year_qfq','Year_hfq']

TIME_TEADE_TIME_LEVELS_LITE = ['5','15','30','60','Day','Week','Month','Year']

TIME_TEADE_TIME_LEVELS_PER_TRADING = ['5','15','30','60','Day']

TIME_TEADE_TIME_LEVELS_AFTER_TRADE = ['Day_qfq','Day_hfq','Week','Week_qfq','Week_hfq','Month','Month_qfq','Month_hfq','Year','Year_qfq','Year_hfq']
