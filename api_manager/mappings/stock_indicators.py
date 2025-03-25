"""
股票分时和技术指标字段映射表，用于API响应字段到模型字段的映射
格式为：{API字段: 模型字段}
"""

# 分时交易数据字段映射
TIME_TRADE_MAPPING = {
    'd': 'trade_time',
    'o': 'open_price',
    'h': 'high_price',
    'l': 'low_price',
    'c': 'close_price',
    'v': 'volume',
    'e': 'turnover',
    'zf': 'amplitude',
    'hs': 'turnover_rate',
    'zd': 'price_change_percent',
    'zde': 'price_change'
}

# KDJ指标数据字段映射
KDJ_INDICATOR_MAPPING = {
    't': 'trade_time',
    'k': 'k_value',
    'd': 'd_value',
    'j': 'j_value'
}

# MACD指标数据字段映射
MACD_INDICATOR_MAPPING = {
    't': 'trade_time',
    'diff': 'diff',
    'dea': 'dea',
    'macd': 'macd',
    'ema12': 'ema12',
    'ema26': 'ema26'
}

# MA指标数据字段映射
MA_INDICATOR_MAPPING = {
    't': 'trade_time',
    'ma3': 'ma3',
    'ma5': 'ma5',
    'ma10': 'ma10',
    'ma15': 'ma15',
    'ma20': 'ma20',
    'ma30': 'ma30',
    'ma60': 'ma60',
    'ma120': 'ma120',
    'ma200': 'ma200',
    'ma250': 'ma250'
}

# BOLL指标数据字段映射
BOLL_INDICATOR_MAPPING = {
    't': 'trade_time',
    'u': 'upper',
    'd': 'lower',
    'm': 'mid'
}
