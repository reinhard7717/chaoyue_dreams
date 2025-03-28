# api_manager/mappings/stock_basic.py

"""
字段映射表，用于API响应字段到模型字段的映射
格式为：{API字段: 模型字段}
"""

# 股票基础信息字段映射
STOCK_BASIC_MAPPING = {
    "dm": "stock_code",  # 股票代码
    "mc": "stock_name",  # 股票名称
    "jys": "exchange"    # 交易所
}

# 指数、行业、概念树字段映射
MARKET_CATEGORY_MAPPING = {
    "name": "name",            # 名称
    "code": "code",            # 代码
    "type1": "type1",          # 一级分类
    "type2": "type2",          # 二级分类
    "level": "level",          # 层级
    "pcode": "pcode",          # 父节点代码
    "pname": "pname",          # 父节点名称
    "isleaf": "isleaf"         # 是否为叶子节点
}


# 新股日历字段映射
NEW_STOCK_CALENDAR_MAPPING = {
    "zqdm": "stock_code",                  # 股票代码
    "zqjc": "stock_short_name",            # 股票简称
    "sgdm": "subscription_code",           # 申购代码
    "fxsl": "issue_total_shares",          # 发行总数（股）
    "swfxsl": "online_issue_shares",       # 网上发行（股）
    "sgsx": "subscription_limit",          # 申购上限（股）
    "dgsz": "max_subscription_value",      # 顶格申购需配市值(元)
    "sgrq": "subscription_date",           # 申购日期
    "fxjg": "issue_price",                 # 发行价格（元）
    "zxj": "latest_price",                 # 最新价（元）
    "srspj": "first_day_close_price",      # 首日收盘价（元）
    "zqgbrq": "winning_announcement_date", # 中签号公布日
    "zqjkrq": "winning_payment_date",      # 中签缴款日
    "ssrq": "listing_date",                # 上市日期
    "syl": "issue_pe_ratio",               # 发行市盈率
    "hysyl": "industry_pe_ratio",          # 行业市盈率
    "wszql": "winning_rate",               # 中签率（%）
    "yzbsl": "consecutive_limit_boards",   # 连续一字板数量
    "zf": "price_increase_rate",           # 涨幅（%）
    "yqhl": "profit_per_winning",          # 每中一签获利（元）
    "zyyw": "main_business"                # 主营业务
}


# 风险警示股票列表字段映射
ST_STOCK_LIST_MAPPING = {
    "dm": "stock_code",  # 股票代码
    "mc": "stock_name",  # 股票名称
    "jys": "exchange"    # 交易所
}


# 公司简介字段映射
COMPANY_INFO_MAPPING = {
    # 注意：API路径参数中的股票代码应映射到 stock_code
    "name": "company_name",                # 公司名称
    "ename": "company_english_name",       # 公司英文名称
    "market": "market",                    # 上市市场
    "idea": "concepts",                    # 概念及板块
    "ldate": "listing_date",               # 上市日期
    "sprice": "issue_price",               # 发行价格（元）
    "principal": "lead_underwriter",       # 主承销商
    "rdate": "establishment_date",         # 成立日期
    "rprice": "registered_capital",        # 注册资本
    "instype": "institution_type",         # 机构类型
    "organ": "organization_form"           # 组织形式
}


# 所属指数字段映射
COMPANY_INDEX_MAPPING = {
    # 注意：API路径参数中的股票代码应映射到 stock_code
    "mc": "index_name",   # 指数名称
    "dm": "index_code",   # 指数代码
    "ind": "entry_date",  # 进入日期
    "outd": "exit_date"   # 退出日期
}


# 季度利润字段映射
QUARTERLY_PROFIT_MAPPING = {
    # 注意：API路径参数中的股票代码应映射到 stock_code
    "date": "report_date",                     # 截止日期
    "income": "operating_revenue",             # 营业收入（万元）
    "expend": "operating_expense",             # 营业支出（万元）
    "profit": "operating_profit",              # 营业利润（万元）
    "totalp": "total_profit",                  # 利润总额（万元）
    "reprofit": "net_profit",                  # 净利润（万元）
    "basege": "basic_earnings_per_share",      # 基本每股收益(元/股)
    "ettege": "diluted_earnings_per_share",    # 稀释每股收益(元/股)
    "otherp": "other_comprehensive_income",    # 其他综合收益（万元）
    "totalcp": "total_comprehensive_income"    # 综合收益总额（万元）
}

