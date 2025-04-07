"""
股票实时数据字段映射表，用于API响应字段到模型字段的映射
格式为：{API字段: 模型字段}
"""

# 实时交易数据字段映射
REALTIME_DATA_MAPPING = {
    'o': 'open_price',
    'fm': 'five_min_change',
    'h': 'high_price',
    'hs': 'turnover_rate',
    'lb': 'volume_ratio',
    'l': 'low_price',
    'lt': 'tradable_market_value',
    'pe': 'pe_ratio',
    'pc': 'price_change_percent',
    'p': 'current_price',
    'sz': 'total_market_value',
    'cje': 'turnover_value',
    'ud': 'price_change',
    'v': 'volume',
    'yc': 'prev_close_price',
    'zf': 'amplitude',
    'zs': 'increase_speed',
    'sjl': 'pb_ratio',
    'zdf60': 'price_change_60d',
    'zdfnc': 'price_change_ytd',
    't': 'trade_time'
}

# 买卖五档盘口数据字段映射
LEVEL5_DATA_MAPPING = {
    't': 'trade_time',
    'vc': 'order_diff',
    'vb': 'order_ratio',
    'pb1': 'buy_price1',
    'vb1': 'buy_volume1',
    'pb2': 'buy_price2',
    'vb2': 'buy_volume2',
    'pb3': 'buy_price3',
    'vb3': 'buy_volume3',
    'pb4': 'buy_price4',
    'vb4': 'buy_volume4',
    'pb5': 'buy_price5',
    'vb5': 'buy_volume5',
    'ps1': 'sell_price1',
    'vs1': 'sell_volume1',
    'ps2': 'sell_price2',
    'vs2': 'sell_volume2',
    'ps3': 'sell_price3',
    'vs3': 'sell_volume3',
    'ps4': 'sell_price4',
    'vs4': 'sell_volume4',
    'ps5': 'sell_price5',
    'vs5': 'sell_volume5'
}

# 逐笔交易数据字段映射
TRADE_DETAIL_MAPPING = {
    'd': 'trade_date',
    't': 'trade_time',
    'v': 'volume',
    'p': 'price',
    'ts': 'trade_direction'
}

# 分时成交数据字段映射
TIME_DEAL_MAPPING = {
    'd': 'trade_date',
    't': 'trade_time',
    'v': 'volume',
    'p': 'price'
}

# 分价成交占比数据字段映射
PRICE_PERCENT_MAPPING = {
    'd': 'trade_date',
    'p': 'price',
    'v': 'volume',
    'b': 'percentage'
}

# 逐笔大单交易数据字段映射
BIG_DEAL_MAPPING = {
    'd': 'trade_date',
    't': 'trade_time',
    'v': 'volume',
    'p': 'price',
    'ts': 'trade_direction'
}

# 盘中异动数据字段映射
ABNORMAL_MOVEMENT_MAPPING = {
    'dm': 'stock_code',
    'mc': 'stock_name',
    't': 'movement_time',
    'type': 'movement_type',
    'xx': 'movement_info'
}
