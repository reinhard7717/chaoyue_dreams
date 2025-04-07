# mapping/stock_index_mapping.py

"""
股票指数字段映射表

用于将API响应字段映射到模型字段
"""

# 指数列表字段映射
INDEX_LIST_MAPPING = {
    'dm': 'code',            # 指数代码
    'mc': 'name',            # 指数名称
    'jys': 'exchange',       # 交易所
}

# 指数实时数据字段映射
INDEX_REALTIME_DATA_MAPPING = {
    'o': 'open_price',                  # 开盘价
    'h': 'high_price',                  # 最高价
    'l': 'low_price',                   # 最低价
    'p': 'current_price',               # 当前价格
    'yc': 'prev_close_price',           # 昨日收盘价
    'ud': 'price_change',               # 涨跌额
    'pc': 'price_change_percent',       # 涨跌幅(%)
    'fm': 'five_minute_change_percent', # 五分钟涨跌幅(%)
    'zf': 'amplitude',                  # 振幅(%)
    'zs': 'change_speed',               # 涨速(%)
    'zdf60': 'sixty_day_change_percent',# 60日涨跌幅(%)
    'zdfnc': 'ytd_change_percent',      # 年初至今涨跌幅(%)
    'v': 'volume',                      # 成交量(手)
    'cje': 'turnover',                  # 成交额(元)
    'hs': 'turnover_rate',              # 换手率(%)
    'lb': 'volume_ratio',               # 量比(%)
    'pe': 'pe_ratio',                   # 市盈率
    'sjl': 'pb_ratio',                  # 市净率
    'lt': 'circulating_market_value',   # 流通市值(元)
    'sz': 'total_market_value',         # 总市值(元)
    't': 'trade_time',                 # 更新时间
}

# 市场概览数据字段映射
MARKET_OVERVIEW_MAPPING = {
    'totalUp': 'total_up',           # 上涨总数
    'totalDown': 'total_down',       # 下跌总数
    'zt': 'limit_up',                # 涨停总数
    'dt': 'limit_down',              # 跌停总数
    'up8ToZt': 'up_8_to_limit',      # 上涨8%~涨停数量
    'up6To8': 'up_6_to_8',           # 上涨6%~8%数量
    'up4To6': 'up_4_to_6',           # 上涨4%~6%数量
    'up2To4': 'up_2_to_4',           # 上涨2%~4%数量
    'up0To2': 'up_0_to_2',           # 上涨0%~2%数量
    'down0To2': 'down_0_to_2',       # 下跌0%~2%数量
    'down2To4': 'down_2_to_4',       # 下跌2%~4%数量
    'down4To6': 'down_4_to_6',       # 下跌4%~6%数量
    'down6To8': 'down_6_to_8',       # 下跌6%~8%数量
    'down8ToDt': 'down_8_to_limit',  # 下跌8%~跌停数量
}

# 时间序列数据字段映射
TIME_SERIES_MAPPING = {
    'd': 'trade_time',           # 交易时间
    'o': 'open_price',           # 开盘价
    'h': 'high_price',           # 最高价
    'l': 'low_price',            # 最低价
    'c': 'close_price',          # 收盘价
    'v': 'volume',               # 成交量(手)
    'e': 'turnover',             # 成交额(元)
    'zf': 'amplitude',           # 振幅(%)
    'hs': 'turnover_rate',       # 换手率(%)
    'zd': 'change_percent',      # 涨跌幅(%)
    'zde': 'change_amount',      # 涨跌额(元)
}

# 添加到现有的映射文件中

# KDJ指标数据字段映射
KDJ_MAPPING = {
    't': 'trade_time',        # 交易时间
    'k': 'k_value',           # K值
    'd': 'd_value',           # D值
    'j': 'j_value',           # J值
}

# MACD指标数据字段映射
MACD_MAPPING = {
    't': 'trade_time',        # 交易时间
    'diff': 'diff',           # DIFF值
    'dea': 'dea',             # DEA值
    'macd': 'macd',           # MACD值
    'ema12': 'ema12',         # EMA(12)值
    'ema26': 'ema26',         # EMA(26)值
}

# MA指标数据字段映射
MA_MAPPING = {
    't': 'trade_time',        # 交易时间
    'ma3': 'ma3',             # MA3
    'ma5': 'ma5',             # MA5
    'ma10': 'ma10',           # MA10
    'ma15': 'ma15',           # MA15
    'ma20': 'ma20',           # MA20
    'ma30': 'ma30',           # MA30
    'ma60': 'ma60',           # MA60
    'ma120': 'ma120',         # MA120
    'ma200': 'ma200',         # MA200
    'ma250': 'ma250',         # MA250
}

# BOLL指标数据字段映射
BOLL_MAPPING = {
    't': 'trade_time',        # 交易时间
    'u': 'upper',             # 上轨
    'm': 'middle',            # 中轨
    'd': 'lower',             # 下轨
}

