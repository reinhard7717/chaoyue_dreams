# mapping/fund_flow_mapping.py

"""
资金流向相关字段映射表

将API响应字段映射到模型字段
"""

# 资金走势对照字段映射
FUND_FLOW_TREND_MAPPING = {
    't': 'trade_time',           # 时间
    'zdf': 'change_percent',     # 涨跌幅(%)
    'lrzj': 'inflow_amount',     # 总流入资金(元)
    'lczj': 'outflow_amount',    # 总流出资金(元)
    'jlr': 'net_inflow',         # 净流入(元)
    'jlrl': 'net_inflow_rate',   # 净流入率(%)
    'lrl': 'main_inflow_rate',   # 主力流入率(%)
    'shlrl': 'retail_inflow_rate', # 散户流入率(%)
}

# 日级资金流入趋势字段映射
DAILY_FUND_FLOW_MAPPING = {
    't': 'trade_date',               # 时间
    'zdf': 'change_percent',         # 涨跌幅(%)
    'hsl': 'turnover_rate',          # 换手率(%)
    'jlr': 'net_inflow',             # 净流入(元)
    'jlrl': 'net_inflow_rate',       # 净流入率(%)
    'zljlr': 'main_net_inflow',      # 主力净流入(元)
    'zljlrl': 'main_net_inflow_rate', # 主力净流入率(%)
    'hyjlr': 'industry_net_inflow',   # 行业净流入(元)
    'hyjlrl': 'industry_net_inflow_rate', # 行业净流入率(%)
}

# 阶段主力动向字段映射
MAIN_FORCE_PHASE_MAPPING = {
    't': 'trade_date',           # 时间
    'jlr3': 'inflow_3day',       # 近3日主力净流入(元)
    'jlr5': 'inflow_5day',       # 近5日主力净流入(元)
    'jlr10': 'inflow_10day',     # 近10日主力净流入(元)
    'jlrl3': 'inflow_rate_3day', # 近3日主力净流入率(%)
    'jlrl5': 'inflow_rate_5day', # 近5日主力净流入率(%)
    'jlrl10': 'inflow_rate_10day', # 近10日主力净流入率(%)
}

# 历史成交分布字段映射
TRANSACTION_DISTRIBUTION_MAPPING = {
    't': 'trade_date',               # 时间
    'c': 'close_price',              # 收盘价(元)
    'zdf': 'change_percent',         # 涨跌幅(%)
    'jlrl': 'net_inflow_rate',       # 净流入率(%)
    'hsl': 'turnover_rate',          # 换手率(%)
    'qbjlr': 'total_net_inflow',     # 全部净流入(元)
    'cddlr': 'super_large_inflow',   # 超大单流入(元)
    'cddjlr': 'super_large_net_inflow', # 超大单净流入(元)
    'ddlr': 'large_inflow',          # 大单流入(元)
    'ddjlr': 'large_net_inflow',     # 大单净流入(元)
    'xdlr': 'small_inflow',          # 小单流入(元)
    'xdjlr': 'small_net_inflow',     # 小单净流入(元)
    'sdlr': 'retail_inflow',         # 散单流入(元)
    'sdjlr': 'retail_net_inflow',    # 散单净流入(元)
}

"""
股票池相关字段映射表

将API响应字段映射到模型字段
"""

# 涨停股池字段映射
LIMIT_UP_POOL_MAPPING = {
    'dm': 'code',                       # 代码
    'mc': 'name',                       # 名称
    'p': 'price',                       # 价格(元)
    'zf': 'change_percent',             # 涨幅(%)
    'cje': 'turnover',                  # 成交额(元)
    'lt': 'circulating_market_value',   # 流通市值(元)
    'zsz': 'total_market_value',        # 总市值(元)
    'hs': 'turnover_rate',              # 换手率(%)
    'lbc': 'consecutive_limit_days',    # 连板数
    'fbt': 'first_limit_time',          # 首次封板时间
    'lbt': 'last_limit_time',           # 最后封板时间
    'zj': 'limit_funds',                # 封板资金(元)
    'zbc': 'break_times',               # 炸板次数
    'tj': 'limit_statistics',           # 涨停统计
}

# 跌停股池字段映射
LIMIT_DOWN_POOL_MAPPING = {
    'dm': 'code',                       # 代码
    'mc': 'name',                       # 名称
    'p': 'price',                       # 价格(元)
    'zf': 'change_percent',             # 跌幅(%)
    'cje': 'turnover',                  # 成交额(元)
    'lt': 'circulating_market_value',   # 流通市值(元)
    'zsz': 'total_market_value',        # 总市值(元)
    'pe': 'pe_ratio',                   # 动态市盈率
    'hs': 'turnover_rate',              # 换手率(%)
    'lbc': 'consecutive_limit_days',    # 连续跌停次数
    'lbt': 'last_limit_time',           # 最后封板时间
    'zj': 'limit_funds',                # 封单资金(元)
    'fba': 'turnover_on_limit',         # 板上成交额(元)
    'zbc': 'open_times',                # 开板次数
}

# 强势股池字段映射
STRONG_STOCK_POOL_MAPPING = {
    'dm': 'code',                       # 代码
    'mc': 'name',                       # 名称
    'p': 'price',                       # 价格(元)
    'ztp': 'limit_up_price',            # 涨停价(元)
    'zf': 'change_percent',             # 涨幅(%)
    'cje': 'turnover',                  # 成交额(元)
    'lt': 'circulating_market_value',   # 流通市值(元)
    'zsz': 'total_market_value',        # 总市值(元)
    'zs': 'change_speed',               # 涨速(%)
    'nh': 'is_new_high',                # 是否新高
    'lb': 'volume_ratio',               # 量比
    'hs': 'turnover_rate',              # 换手率(%)
    'tj': 'limit_statistics',           # 涨停统计
}

# 次新股池字段映射
NEW_STOCK_POOL_MAPPING = {
    'dm': 'code',                       # 代码
    'mc': 'name',                       # 名称
    'p': 'price',                       # 价格(元)
    'ztp': 'limit_up_price',            # 涨停价(元)
    'zf': 'change_percent',             # 涨跌幅(%)
    'cje': 'turnover',                  # 成交额(元)
    'lt': 'circulating_market_value',   # 流通市值(元)
    'zsz': 'total_market_value',        # 总市值(元)
    'nh': 'is_new_high',                # 是否新高
    'hs': 'turnover_rate',              # 转手率(%)
    'tj': 'limit_statistics',           # 涨停统计
    'kb': 'days_after_open',            # 开板几日
    'od': 'open_date',                  # 开板日期
    'ipod': 'ipo_date',                 # 上市日期
}

# 炸板股池字段映射
BREAK_LIMIT_POOL_MAPPING = {
    'dm': 'code',                       # 代码
    'mc': 'name',                       # 名称
    'p': 'price',                       # 价格(元)
    'ztp': 'limit_up_price',            # 涨停价(元)
    'zf': 'change_percent',             # 涨跌幅(%)
    'cje': 'turnover',                  # 成交额(元)
    'lt': 'circulating_market_value',   # 流通市值(元)
    'zsz': 'total_market_value',        # 总市值(元)
    'zs': 'change_speed',               # 涨速(%)
    'hs': 'turnover_rate',              # 转手率(%)
    'tj': 'limit_statistics',           # 涨停统计
    'fbt': 'first_limit_time',          # 首次封板时间
    'zbc': 'break_times',               # 炸板次数
}
