# api_manager/mappings/datacenter_mappings.py
# 龙虎榜相关映射
LHB_DETAIL_MAPPING = {
    "dm": "stock_code",    # 股票代码
    "mc": "stock_name",    # 股票名称
    "c": "close_price",    # 收盘价
    "val": "value",        # 对应值
    "v": "volume",         # 成交量(万股)
    "e": "amount"          # 成交额(万元)
}

LHB_DAILY_MAPPING = {
    "t": "trade_date",                         # 日期
    "dpl7": "decline_deviation_7pct",    # 跌幅偏离值达7%的证券
    "z20": "rise_cumulative_20pct",      # 连续三个交易日内，涨幅偏离值累计达20%的证券
    "zpl7": "rise_deviation_7pct",       # 涨幅偏离值达7%的证券
    "h20": "turnover_20pct",             # 换手率达20%的证券
    "st15": "st_rise_15pct",             # 连续三个交易日内，涨幅偏离值累计达到15%的ST证券
    "st12": "st_rise_12pct",             # 连续三个交易日内，涨幅偏离值累计达到12%的ST证券
    "std15": "st_decline_15pct",         # 连续三个交易日内，跌幅偏离值累计达到15%的ST证券
    "std12": "st_decline_12pct",         # 连续三个交易日内，跌幅偏离值累计达到12%的ST证券
    "zf15": "amplitude_15pct",           # 振幅值达15%的证券
    "df15": "decline_cumulative_15pct",  # 连续三个交易日内，跌幅偏离值累计达20%的证券
    "wxz": "no_price_limit",             # 无价格涨跌幅限制的证券
    "wxztp": "no_price_limit_halted"     # 当日无价格涨跌幅限制的A股，出现异常波动停牌的股票
}


STOCK_ON_LIST_MAPPING = {
    "dm": "stock_code",             # 股票代码
    "mc": "stock_name",             # 股票名称
    "count": "list_count",          # 上榜次数
    "totalb": "total_buy_amount",   # 累积获取额(万)
    "totals": "total_sell_amount",  # 累积卖出额(万)
    "netp": "net_amount",           # 净额(万)
    "xb": "buy_seat_count",         # 买入席位数
    "xs": "sell_seat_count",        # 卖出席位
    "days": "stats_days"            # 统计天数
}


BROKER_ON_LIST_MAPPING = {
    "yybmc": "broker_name",         # 营业部名称
    "count": "list_count",          # 上榜次数
    "totalb": "total_buy_amount",   # 累积获取额(万)
    "bcount": "buy_count",          # 买入席位
    "totals": "total_sell_amount",  # 累积卖出额(万)
    "scount": "sell_count",         # 卖出席位
    "top3": "top3_stocks",          # 买入前三股票
    "days": "stats_days"            # 统计天数
}


INSTITUTION_TRADE_TRACK_MAPPING = {
    "dm": "stock_code",        # 股票代码
    "mc": "stock_name",        # 股票名称
    "be": "buy_amount",        # 累积买入额(万)
    "bcount": "buy_count",     # 买入次数
    "se": "sell_amount",       # 累积卖出额(万)
    "scount": "sell_count",    # 卖出次数
    "ende": "net_amount",      # 净额(万)
    "days": "stats_days"       # 统计天数
}


INSTITUTION_TRADE_DETAIL_MAPPING = {
    "dm": "stock_code",       # 股票代码
    "mc": "stock_name",       # 股票名称
    "t": "trade_date",        # 交易日期
    "buy": "buy_amount",      # 机构席位买入额(万)
    "sell": "sell_amount",    # 机构席位卖出额(万)
    "type": "trade_type"      # 类型
}


# 个股统计相关映射
STAGE_HIGH_LOW_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "g5": "high_price_5d",             # 近5日最高价
    "d5": "low_price_5d",              # 近5日最低价
    "zd5": "change_rate_5d",           # 近5日涨跌幅
    "iscq5": "has_ex_dividend_5d",     # 近5日是否除权除息
    "g10": "high_price_10d",           # 近10日最高价
    "d10": "low_price_10d",            # 近10日最低价
    "zd10": "change_rate_10d",         # 近10日涨跌幅
    "iscq10": "has_ex_dividend_10d",   # 近10日是否除权除息
    "g20": "high_price_20d",           # 近20日最高价
    "d20": "low_price_20d",            # 近20日最低价
    "zd20": "change_rate_20d",         # 近20日涨跌幅
    "iscq20": "has_ex_dividend_20d",   # 近20日是否除权除息
    "g60": "high_price_60d",           # 近60日最高价
    "d60": "low_price_60d",            # 近60日最低价
    "zd60": "change_rate_60d",         # 近60日涨跌幅
    "iscq60": "has_ex_dividend_60d"    # 近60日是否除权除息
}


NEW_HIGH_STOCK_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "h": "high_price",                 # 最高价
    "l": "low_price",                  # 最低价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "hs": "turnover_rate",             # 换手率
    "zdf5": "change_rate_5d",          # 5日涨跌幅
    "iscq5": "has_ex_dividend_5d",     # 近5日是否除权除息
    "zdf10": "change_rate_10d",        # 10日涨跌幅
    "iscq10": "has_ex_dividend_10d",   # 近10日是否除权除息
    "zdf20": "change_rate_20d",        # 20日涨跌幅
    "iscq20": "has_ex_dividend_20d"    # 近20日是否除权除息
}


# 与NEW_HIGH_STOCK_MAPPING相同的字段，可以共用
NEW_LOW_STOCK_MAPPING = NEW_HIGH_STOCK_MAPPING

# 市场数据相关映射
VOLUME_INCREASE_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "v": "volume",                     # 成交量
    "pv": "previous_volume",           # 前一交易日成交量
    "zjl": "volume_change",            # 增减量
    "zjf": "volume_change_rate"        # 增减幅
}


# 与VOLUME_INCREASE_MAPPING相同的字段，可以共用
VOLUME_DECREASE_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "v": "volume",                     # 成交量
    "pv": "previous_volume",           # 前一交易日成交量
    "zjl": "volume_change",            # 增减量
    "zjf": "volume_change_rate"        # 增减幅
}


CONTINUOUS_VOLUME_INCREASE_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "v": "volume",                     # 成交量
    "pv": "previous_volume",           # 前一交易日成交量
    "flday": "volume_increase_days",   # 放量天数
    "pzdf": "period_change_rate",      # 阶段涨跌幅
    "ispcq": "period_has_ex_dividend", # 阶段是否除权除息
    "phs": "period_turnover_rate"      # 阶段换手率
}


# 与CONTINUOUS_VOLUME_INCREASE_MAPPING相同的字段，可以共用
CONTINUOUS_VOLUME_DECREASE_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "v": "volume",                     # 成交量
    "pv": "previous_volume",           # 前一交易日成交量
    "flday": "volume_decrease_days",   # 缩量天数
    "pzdf": "period_change_rate",      # 阶段涨跌幅
    "ispcq": "period_has_ex_dividend", # 阶段是否除权除息
    "phs": "period_turnover_rate"      # 阶段换手率
}


CONTINUOUS_RISE_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "v": "volume",                     # 成交量
    "hs": "turnover_rate",             # 换手
    "szday": "rising_days",            # 上涨天数
    "pzdf": "period_change_rate",      # 阶段涨跌幅
    "ispcq": "period_has_ex_dividend"  # 阶段是否除权除息
}


# 与CONTINUOUS_RISE_MAPPING相同的字段，可以共用
CONTINUOUS_FALL_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "c": "close_price",                # 收盘价
    "zdf": "change_rate",              # 涨跌幅
    "iscq": "is_ex_dividend",          # 当天是否除权除息
    "v": "volume",                     # 成交量
    "hs": "turnover_rate",             # 换手
    "szday": "falling_days",           # 下跌天数
    "pzdf": "period_change_rate",      # 阶段涨跌幅
    "ispcq": "period_has_ex_dividend"  # 阶段是否除权除息
}


# 财务指标相关映射
WEEKLY_RANK_CHANGE_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 代码
    "mc": "stock_name",                # 名称
    "zdf": "weekly_change_rate",       # 周涨跌幅
    "v": "weekly_volume",              # 周成交量
    "amount": "weekly_amount",         # 周成交额
    "hs": "weekly_turnover_rate",      # 周换手率
    "hp": "weekly_highest_price",      # 周最高价
    "lp": "weekly_lowest_price",       # 周最低价
    "zf": "weekly_amplitude"           # 周振幅
}

# 与WEEKLY_RANK_CHANGE_MAPPING相同的字段，可以共用
MONTHLY_RANK_CHANGE_MAPPING = {
    "t": "trade_date",                        # 日期
    "dm": "stock_code",                 # 代码
    "mc": "stock_name",                 # 名称
    "zdf": "monthly_change_rate",       # 月涨跌幅
    "v": "monthly_volume",              # 月成交量
    "amount": "monthly_amount",         # 月成交额
    "hs": "monthly_turnover_rate",      # 月换手率
    "hp": "monthly_highest_price",      # 月最高价
    "lp": "monthly_lowest_price",       # 月最低价
    "zf": "monthly_amplitude"           # 月振幅
}

WEEKLY_STRONG_STOCK_MAPPING = {
    "t": "trade_date",                         # 日期
    "dm": "stock_code",                  # 代码
    "mc": "stock_name",                  # 名称
    "zdf": "weekly_change_rate",         # 周涨跌幅
    "o": "weekly_open_price",            # 周开盘价
    "c": "weekly_close_price",           # 周收盘价
    "h": "weekly_highest_price",         # 周最高价
    "l": "weekly_lowest_price",          # 周最低价
    "v": "weekly_volume",                # 周成交量
    "hs": "weekly_turnover_rate",        # 周换手率
    "zf300": "hs300_weekly_change_rate"  # 本周沪深300涨幅
}

# 与WEEKLY_STRONG_STOCK_MAPPING相同的字段，可以共用
MONTHLY_STRONG_STOCK_MAPPING = {
    "t": "trade_date",                          # 日期
    "dm": "stock_code",                   # 代码
    "mc": "stock_name",                   # 名称
    "zdf": "monthly_change_rate",         # 月涨跌幅
    "o": "monthly_open_price",            # 月开盘价
    "c": "monthly_close_price",           # 月收盘价
    "h": "monthly_highest_price",         # 月最高价
    "l": "monthly_lowest_price",          # 月最低价
    "v": "monthly_volume",                # 月成交量
    "hs": "monthly_turnover_rate",        # 月换手率
    "zf300": "hs300_monthly_change_rate"  # 本月沪深300涨幅
}

CIRC_MARKET_VALUE_RANK_MAPPING = {
    "t": "trade_date",                         # 日期
    "dm": "stock_code",                  # 代码
    "mc": "stock_name",                  # 名称
    "c": "close_price",                  # 收盘价
    "zdf": "change_rate",                # 涨跌幅
    "v": "volume",                       # 成交量
    "hs": "turnover_rate",               # 换手率
    "ltsz": "circulating_market_value",  # 流通市值
    "zsz": "total_market_value"          # 总市值
}

PE_RATIO_RANK_MAPPING = {
    "t": "trade_date",                # 日期
    "dm": "stock_code",         # 代码
    "mc": "stock_name",         # 名称
    "c": "close_price",         # 收盘价
    "zdf": "change_rate",       # 涨跌幅
    "v": "volume",              # 成交量
    "hs": "turnover_rate",      # 换手率
    "jpe": "static_pe_ratio",   # 静态市盈率
    "dpe": "ttm_pe_ratio"       # 市盈率(TTM)
}

PB_RATIO_RANK_MAPPING = {
    "t": "trade_date",                    # 日期
    "dm": "stock_code",             # 代码
    "mc": "stock_name",             # 名称
    "c": "close_price",             # 收盘价
    "zdf": "change_rate",           # 涨跌幅
    "v": "volume",                  # 成交量
    "hs": "turnover_rate",          # 换手率
    "sjl": "pb_ratio",              # 市净率
    "jzc": "net_asset_per_share"    # 每股净资产
}

ROE_RANK_MAPPING = {
    "dm": "stock_code",                    # 股票代码
    "mc": "stock_name",                    # 股票名称
    "roe": "roe",                          # ROE
    "zsz": "total_market_value",           # 总市值
    "jzc": "net_assets",                   # 净资产
    "jlr": "net_profit",                   # 净利润
    "syld": "dynamic_pe_ratio",            # 市盈率(动)
    "sjl": "pb_ratio",                     # 市净率
    "mll": "gross_profit_margin",          # 毛利率
    "jll": "net_profit_margin",            # 净利率
    "hyroe": "industry_avg_roe",           # 行业平均ROE
    "hyzsz": "industry_avg_market_value",  # 行业平均总市值
    "hyjzc": "industry_avg_net_assets",    # 行业平均净资产
    "hyjlr": "industry_avg_net_profit",    # 行业平均净利润
    "hysyld": "industry_avg_dynamic_pe",   # 行业平均市盈率(动)
    "hysjl": "industry_avg_pb_ratio",      # 行业平均市净率
    "hymll": "industry_avg_gross_margin",  # 行业平均毛利率
    "hyjll": "industry_avg_net_margin",    # 行业平均净利率
    "roepm": "roe_industry_rank",          # ROE行业排名
    "zszpm": "market_value_industry_rank", # 总市值行业排名
    "jzcpm": "net_assets_industry_rank",   # 净资产行业排名
    "jlrpm": "net_profit_industry_rank",   # 净利润行业排名
    "syldpm": "pe_ratio_industry_rank",    # 市盈率行业排名
    "sjlpm": "pb_ratio_industry_rank",     # 市净率行业排名
    "mllpm": "gross_margin_industry_rank", # 毛利率行业排名
    "jllpm": "net_margin_industry_rank",   # 净利率行业排名
    "hym": "industry_name",                # 行业名
    "hygpzs": "industry_stock_count"       # 同行业股票总数量
}

# api_manager/mappings/datacenter_mappings.py
# 在现有映射后添加以下内容

# 机构持股相关映射
INSTITUTION_HOLDING_SUMMARY_MAPPING = {
    "t": "trade_date",                                 # 统计日期
    "dm": "stock_code",                          # 股票代码
    "mc": "stock_name",                          # 股票名称
    "jgcgs": "institution_count",                # 机构持股家数
    "jgcgzb": "institution_holding_ratio",       # 机构持股占比(%)
    "jgcgsz": "institution_holding_value",       # 机构持股市值(万元)
    "jjcgs": "fund_count",                       # 基金持股家数
    "jjcgzb": "fund_holding_ratio",              # 基金持股占比(%)
    "jjcgsz": "fund_holding_value",              # 基金持股市值(万元)
    "sbcgs": "social_security_count",            # 社保持股家数
    "sbcgzb": "social_security_holding_ratio",   # 社保持股占比(%)
    "sbcgsz": "social_security_holding_value",   # 社保持股市值(万元)
    "qfiicgs": "qfii_count",                     # QFII持股家数
    "qfiicgzb": "qfii_holding_ratio",            # QFII持股占比(%)
    "qfiicgsz": "qfii_holding_value",            # QFII持股市值(万元)
    "baoxcgs": "insurance_count",                # 保险持股家数
    "baoxcgzb": "insurance_holding_ratio",       # 保险持股占比(%)
    "baoxcgsz": "insurance_holding_value",       # 保险持股市值(万元)
    "year": "year",                              # 报告年份
    "quarter": "quarter"                         # 报告季度
}


FUND_HEAVY_POSITION_MAPPING = {
    "t": "trade_date",                          # 统计日期
    "dm": "stock_code",                   # 股票代码
    "mc": "stock_name",                   # 股票名称
    "jjsl": "fund_count",                 # 持有基金数
    "cgs": "holding_shares",              # 持股数(万股)
    "cgbl": "holding_ratio",              # 持股比例(%)
    "cgsz": "holding_value",              # 持股市值(万元)
    "cgszbl": "float_market_value_ratio", # 占流通市值比例(%)
    "zltgbl": "total_share_ratio",        # 占总股本比例(%)
    "jzc": "net_assets",                  # 净资产(万元)
    "jlr": "net_profit",                  # 净利润(万元)
    "c": "close_price",                   # 收盘价
    "zdf": "change_rate",                 # 涨跌幅(%)
    "year": "year",                       # 报告年份
    "quarter": "quarter"                  # 报告季度
}


SOCIAL_SECURITY_HEAVY_POSITION_MAPPING = {
    "t": "trade_date",                          # 统计日期
    "dm": "stock_code",                   # 股票代码
    "mc": "stock_name",                   # 股票名称
    "sbsl": "social_security_count",      # 持有社保基金数
    "cgs": "holding_shares",              # 持股数(万股)
    "cgbl": "holding_ratio",              # 持股比例(%)
    "cgsz": "holding_value",              # 持股市值(万元)
    "cgszbl": "float_market_value_ratio", # 占流通市值比例(%)
    "zltgbl": "total_share_ratio",        # 占总股本比例(%)
    "jzc": "net_assets",                  # 净资产(万元)
    "jlr": "net_profit",                  # 净利润(万元)
    "c": "close_price",                   # 收盘价
    "zdf": "change_rate",                 # 涨跌幅(%)
    "year": "year",                       # 报告年份
    "quarter": "quarter"                  # 报告季度
}


QFII_HEAVY_POSITION_MAPPING = {
    "t": "trade_date",                          # 统计日期
    "dm": "stock_code",                   # 股票代码
    "mc": "stock_name",                   # 股票名称
    "qfiis": "qfii_count",                # 持有QFII数
    "cgs": "holding_shares",              # 持股数(万股)
    "cgbl": "holding_ratio",              # 持股比例(%)
    "cgsz": "holding_value",              # 持股市值(万元)
    "cgszbl": "float_market_value_ratio", # 占流通市值比例(%)
    "zltgbl": "total_share_ratio",        # 占总股本比例(%)
    "jzc": "net_assets",                  # 净资产(万元)
    "jlr": "net_profit",                  # 净利润(万元)
    "c": "close_price",                   # 收盘价
    "zdf": "change_rate",                 # 涨跌幅(%)
    "year": "year",                       # 报告年份
    "quarter": "quarter"                  # 报告季度
}


# 资金流向相关映射
INDUSTRY_CAPITAL_FLOW_MAPPING = {
    "t": "trade_date",                             # 日期
    "hymc": "industry_name",                 # 行业名称
    "zjjlr": "net_inflow",                   # 资金净流入(万元)
    "zljlr": "main_force_net_inflow",        # 主力净流入(万元)
    "shjlr": "retail_net_inflow",            # 散户净流入(万元)
    "jlrl": "net_inflow_rate",               # 净流入率(%)
    "zljlrl": "main_force_net_inflow_rate",  # 主力净流入率(%)
    "shjlrl": "retail_net_inflow_rate",      # 散户净流入率(%)
    "jlrpj": "average_net_inflow",           # 净流入均额(万元)
    "zdf": "change_rate"                     # 行业涨跌幅(%)
}


CONCEPT_CAPITAL_FLOW_MAPPING = {
    "t": "trade_date",                             # 日期
    "gnmc": "concept_name",                  # 概念名称
    "zjjlr": "net_inflow",                   # 资金净流入(万元)
    "zljlr": "main_force_net_inflow",        # 主力净流入(万元)
    "shjlr": "retail_net_inflow",            # 散户净流入(万元)
    "jlrl": "net_inflow_rate",               # 净流入率(%)
    "zljlrl": "main_force_net_inflow_rate",  # 主力净流入率(%)
    "shjlrl": "retail_net_inflow_rate",      # 散户净流入率(%)
    "jlrpj": "average_net_inflow",           # 净流入均额(万元)
    "zdf": "change_rate"                     # 概念涨跌幅(%)
}


STOCK_CAPITAL_FLOW_MAPPING = {
    "t": "trade_date",                             # 日期
    "dm": "stock_code",                      # 股票代码
    "mc": "stock_name",                      # 股票名称
    "zjjlr": "net_inflow",                   # 资金净流入(万元)
    "zljlr": "main_force_net_inflow",        # 主力净流入(万元)
    "shjlr": "retail_net_inflow",            # 散户净流入(万元)
    "jlrl": "net_inflow_rate",               # 净流入率(%)
    "zljlrl": "main_force_net_inflow_rate",  # 主力净流入率(%)
    "shjlrl": "retail_net_inflow_rate",      # 散户净流入率(%)
    "zdf": "change_rate",                    # 涨跌幅(%)
    "cje": "trading_amount",                 # 成交额(万元)
    "zsz": "total_market_value",             # 总市值(万元)
    "hs": "turnover_rate"                    # 换手率(%)
}


# 南北向资金相关映射
NORTH_SOUTH_FUND_OVERVIEW_MAPPING = {
    "t": "trade_date",                       # 日期
    "hk2sh": "hk_to_shanghai",         # 沪股通(北向)(亿元)
    "hk2sz": "hk_to_shenzhen",         # 深股通(北向)(亿元)
    "bxzjlr": "northbound_net_inflow", # 北向资金净流入(亿元)
    "sh2hk": "shanghai_to_hk",         # 沪港通(南向)(亿元)
    "sz2hk": "shenzhen_to_hk",         # 深港通(南向)(亿元)
    "nxzjlr": "southbound_net_inflow"  # 南向资金净流入(亿元)
}


NORTH_FUND_TREND_MAPPING = {
    "t": "trade_date",                       # 日期
    "hk2sh": "hk_to_shanghai",         # 沪股通(亿元)
    "hk2sz": "hk_to_shenzhen",         # 深股通(亿元)
    "bxzjlr": "northbound_net_inflow", # 北向资金净流入(亿元)
    "hsIndex": "hs300_index"           # 沪深300指数
}


SOUTH_FUND_TREND_MAPPING = {
    "t": "trade_date",                       # 日期
    "sh2hk": "shanghai_to_hk",         # 沪港通(亿元)
    "sz2hk": "shenzhen_to_hk",         # 深港通(亿元)
    "nxzjlr": "southbound_net_inflow", # 南向资金净流入(亿元)
    "hsi": "hang_seng_index"           # 恒生指数
}


NORTH_STOCK_HOLDING_MAPPING = {
    "t": "trade_date",                       # 日期
    "dm": "stock_code",                # 股票代码
    "mc": "stock_name",                # 股票名称
    "cgs": "holding_shares",           # 持股数量(万股)
    "zltgbl": "float_share_ratio",     # 占流通股比例(%)
    "cgsz": "holding_value",           # 持股市值(万元)
    "djcgs": "daily_share_change",     # 当日持股变动(万股)
    "djcgsz": "daily_value_change",    # 当日市值变动(万元)
    "zdf": "price_change_rate",        # 涨跌幅(%)
    "zdbl": "holding_ratio_change",    # 持股占比变动(%)
    "period": "period"                 # 统计周期
}

