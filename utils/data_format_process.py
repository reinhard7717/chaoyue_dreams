import datetime
import decimal
from django.utils import timezone
from typing import Any, Dict
import logging
import numpy as np
import math
# 导入 Django 的 Model 基类，用于判断是否是模型实例
from django.db.models import Model
from dao_manager.base_dao import BaseDAO
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS, FundFlowDaily, FundFlowIndustryTHS, FundFlowMarketDc
from stock_models.index import IndexDailyBasic, IndexInfo, IndexWeight, TradeCalendar
from stock_models.industry import DcIndex, DcIndexDaily, DcIndexMember, KplConcept, SwIndustry, SwIndustryDaily, SwIndustryMember, ThsIndex, ThsIndexMember, ThsIndexDaily
from stock_models.market import HmDetail, HmList, LimitCptList, LimitListD, LimitListThs, LimitStep, MarketDailyInfo
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import StockLevel5Data, StockRealtimeData
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyBasic, StockDailyData, StockMinuteData, StockMonthlyData, StockTimeTrade, StockWeeklyData, IndexDaily
from users.models import FavoriteStock

logger = logging.getLogger(__name__)

# 对所有字段做一次NaN/None清洗
def safe_value(val):
    # 修改代码行：首先检查值是否是 Django 模型实例，如果是则直接返回
    if isinstance(val, Model):
        return val
    # 递归处理 dict
    if isinstance(val, dict):
        return {k: safe_value(v) for k, v in val.items()}
    # 递归处理 list/tuple
    if isinstance(val, (list, tuple)):
        return [safe_value(v) for v in val]
    # 处理 float nan
    if isinstance(val, float) and (np.isnan(val) or math.isnan(val)):
        return None
    # 处理 decimal.Decimal
    if isinstance(val, decimal.Decimal):
        return float(val)
    # 处理 datetime/date
    if isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    return val

class UserDataFormatProcess(BaseDAO):
    def set_user_favorites(self, user_id: int, api_data: Any) -> Dict:
        if isinstance(api_data, FavoriteStock):
            data_dict = {
                'user_id': user_id,
                'added_at': api_data.added_at,
                'note': api_data.note,
                'is_pinned': api_data.is_pinned,
                'tags': api_data.tags,
            }
        else:
            data_dict = {
                'user_id': user_id,
                'added_at': api_data.get('added_at'),
                'note': api_data.get('note'),
                'is_pinned': api_data.get('is_pinned'),
                'tags': api_data.get('tags'),
            }
        return {k: safe_value(v) for k, v in data_dict.items()}

class IndexDataFormatProcess(BaseDAO):
    # 指数基础信息
    def set_index_info_data(self, api_data: Any) -> Dict:
        data_dict = {
            "index_code": getattr(api_data, "ts_code", getattr(api_data, "index_code", None)),  # 指数代码
            "name": getattr(api_data, "name", None),  # 简称
            "fullname": getattr(api_data, "fullname", None),  # 指数全称
            "market": getattr(api_data, "market", None),  # 市场
            "publisher": getattr(api_data, "publisher", None),  # 发布方
            "index_type": getattr(api_data, "index_type", None),  # 指数风格
            "category": getattr(api_data, "category", None),  # 指数类别
            "base_date": self._parse_datetime(getattr(api_data, "base_date", None)),  # 基期
            "base_point": self._parse_number(getattr(api_data, "base_point", None)),  # 基点
            "list_date": getattr(api_data, "list_date", None),  # 发布日期
            "weight_rule": getattr(api_data, "weight_rule", None),  # 加权方式
            "desc": getattr(api_data, "desc", None),  # 描述
            "exp_date": getattr(api_data, "exp_date", None),  # 终止日期
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 指数成分和权重
    def set_index_weight_data(self, index_info: IndexInfo, api_data: Any) -> Dict:
        data_dict = {
            "index": index_info,  # 指数代码
            "stock": getattr(api_data, "stock", getattr(api_data, "stock_code", None)),  # 股票代码
            "trade_date": self._parse_datetime(getattr(api_data, "trade_date", None)),  # 交易日期
            "weight": self._parse_number(getattr(api_data, "weight", None)),  # 权重
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 指数每日指标
    def set_index_daily_data(self, index_info: IndexInfo, api_data: Any) -> Dict:
        data_dict = {
            "index": index_info,  # 指数代码
            "trade_time": self._parse_datetime(
                getattr(api_data, "trade_date", getattr(api_data, "trade_time", None))
            ),  # 交易日期
            "close": self._parse_number(getattr(api_data, "close", None)),  # 收盘
            "open": self._parse_number(getattr(api_data, "open", None)),  # 开盘
            "high": self._parse_number(getattr(api_data, "high", None)),  # 最高
            "low": self._parse_number(getattr(api_data, "low", None)),  # 最低
            "pre_close": self._parse_number(getattr(api_data, "pre_close", None)),  # 昨收
            "change": self._parse_number(getattr(api_data, "change", None)),  # 涨跌额
            "pct_chg": self._parse_number(getattr(api_data, "pct_chg", None)),  # 涨跌幅
            "vol": self._parse_number(getattr(api_data, "vol", None)),  # 成交量
            "amount": self._parse_number(getattr(api_data, "amount", None)),  # 成交额
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 大盘指数每日指标
    def set_index_daily_basic_data(self, index_info: IndexInfo, api_data: Any) -> Dict:
        data_dict = {
            "index": index_info,  # 指数代码
            "trade_time": self._parse_datetime(
                getattr(api_data, "trade_date", getattr(api_data, "trade_time", None))
            ),  # 交易日期
            "total_mv": self._parse_number(getattr(api_data, "total_mv", None)),  # 总市值
            "float_mv": self._parse_number(getattr(api_data, "float_mv", None)),  # 流通市值
            "total_share": self._parse_number(getattr(api_data, "total_share", None)),  # 总股本
            "float_share": self._parse_number(getattr(api_data, "float_share", None)),  # 流通股本
            "free_share": self._parse_number(getattr(api_data, "free_share", None)),  # 自由流通股本
            "turnover_rate": self._parse_number(getattr(api_data, "turnover_rate", None)),  # 换手率
            "turnover_rate_f": self._parse_number(getattr(api_data, "turnover_rate_f", None)),  # 换手率(自由流通)
            "pe": self._parse_number(getattr(api_data, "pe", None)),  # 市盈率
            "pe_ttm": self._parse_number(getattr(api_data, "pe_ttm", None)),  # 市盈率TTM
            "pb": self._parse_number(getattr(api_data, "pb", None)),  # 市净率
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 交易日历
    def set_trade_calendar_data(self, api_data: Any) -> Dict:
        data_dict = {
            "exchange": getattr(api_data, "exchange", None),  # 交易所
            "cal_date": self._parse_datetime(getattr(api_data, "cal_date", None)),  # 日历日期
            "is_open": getattr(api_data, "is_open", None),  # 是否交易
            "pretrade_date": self._parse_datetime(getattr(api_data, "pretrade_date", None)),  # 上一个交易日
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class StockInfoFormatProcess(BaseDAO):
    def set_stock_info_data(self, api_data: Any) -> Dict:
        data_dict = {
            'stock_code': getattr(api_data, 'ts_code', getattr(api_data, 'stock_code', None)),  # 股票代码
            'stock_name': getattr(api_data, 'name', getattr(api_data, 'stock_name', None)),  # 股票名称
            'area': getattr(api_data, 'area', None),  # 地域
            'industry': getattr(api_data, 'industry', None),  # 所属行业
            'full_name': getattr(api_data, 'fullname', getattr(api_data, 'full_name', None)),  # 股票全称
            'en_name': getattr(api_data, 'enname', getattr(api_data, 'en_name', None)),  # 英文全称
            'cn_spell': getattr(api_data, 'cnspell', getattr(api_data, 'cn_spell', None)),  # 拼音缩写
            'market_type': getattr(api_data, 'market', getattr(api_data, 'market_type', None)),  # 市场类型
            'exchange': getattr(api_data, 'exchange', None),  # 交易所代码
            'currency_type': getattr(api_data, 'curr_type', getattr(api_data, 'currency_type', None)),  # 交易货币
            'list_status': getattr(api_data, 'list_status', None),  # 上市状态
            'list_date': getattr(api_data, 'list_date', None),  # 上市日期
            'delist_date': getattr(api_data, 'delist_date', None),  # 退市日期
            'is_hs': getattr(api_data, 'is_hs', None),  # 是否沪深港通标的
            'actual_controller': getattr(api_data, 'act_name', getattr(api_data, 'actual_controller', None)),  # 实控人名称
            'actual_controller_type': getattr(api_data, 'act_ent_type', getattr(api_data, 'actual_controller_type', None)),  # 实控人企业性质
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_stock_info_basic_data(self, api_data: Any) -> Dict:
        data_dict = {
            'stock_code': getattr(api_data, 'ts_code', getattr(api_data, 'stock_code', None)),  # 股票代码
            'stock_name': getattr(api_data, 'name', getattr(api_data, 'stock_name', None)),  # 股票名称
            'industry': getattr(api_data, 'industry', None),  # 所属行业
            'market_type': getattr(api_data, 'market', getattr(api_data, 'market_type', None)),  # 市场类型
            'exchange': getattr(api_data, 'exchange', None),  # 交易所代码
            'currency_type': getattr(api_data, 'curr_type', getattr(api_data, 'currency_type', None)),  # 交易货币
            'list_status': getattr(api_data, 'list_status', None),  # 上市状态
            'is_hs': getattr(api_data, 'is_hs', None),  # 是否沪深港通标的
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_company_info_data(self, stock: StockInfo, api_data: Any) -> Dict:
        data_dict = {
            'stock': stock,
            'com_name': getattr(api_data, 'com_name', None),
            'com_id': getattr(api_data, 'com_id', None),
            'exchange': getattr(api_data, 'exchange', None),
            'chairman': getattr(api_data, 'chairman', None),
            'manager': getattr(api_data, 'manager', None),
            'secretary': getattr(api_data, 'secretary', None),
            'reg_capital': getattr(api_data, 'reg_capital', None),
            'setup_date': getattr(api_data, 'setup_date', None),
            'province': getattr(api_data, 'province', None),
            'city': getattr(api_data, 'city', None),
            'introduction': getattr(api_data, 'introduction', None),
            'website': getattr(api_data, 'website', None),
            'email': getattr(api_data, 'email', None),
            'office': (getattr(api_data, 'office', '') or '')[:100],
            'employees': getattr(api_data, 'employees', None),
            'main_business': getattr(api_data, 'main_business', None),
            'business_scope': getattr(api_data, 'business_scope', None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_hs_const_data(self, stock: StockInfo, api_data: Any) -> Dict:
        data_dict = {
            'stock': stock,
            'hs_type': getattr(api_data, 'hs_type', None),
            'in_date': getattr(api_data, 'in_date', None),
            'out_date': getattr(api_data, 'out_date', None),
            'is_new': getattr(api_data, 'is_new', None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class StockTimeTradeFormatProcess(BaseDAO):
    def set_time_trade_day_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "open": getattr(df_data, "open", None),
            "high": getattr(df_data, "high", None),
            "low": getattr(df_data, "low", None),
            "close": getattr(df_data, "close", None),
            "pre_close": getattr(df_data, "pre_close", None),
            "change": getattr(df_data, "change", None),
            "pct_change": getattr(df_data, "pct_change", getattr(df_data, "pct_chg", None)),
            "vol": getattr(df_data, "vol", None),
            "amount": getattr(df_data, "amount", None),
            "adj_factor": getattr(df_data, "adj_factor", None),
            "open_qfq": getattr(df_data, "open_qfq", None),
            "high_qfq": getattr(df_data, "high_qfq", None),
            "low_qfq": getattr(df_data, "low_qfq", None),
            "close_qfq": getattr(df_data, "close_qfq", None),
            "pre_close_qfq": getattr(df_data, "pre_close_qfq", None),
            "open_hfq": getattr(df_data, "open_hfq", None),
            "high_hfq": getattr(df_data, "high_hfq", None),
            "low_hfq": getattr(df_data, "low_hfq", None),
            "close_hfq": getattr(df_data, "close_hfq", None),
            "pre_close_hfq": getattr(df_data, "pre_close_hfq", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_time_trade_minute_data(self, stock: StockInfo, df_data: Any) -> Dict:
        # 兼容 freq 和 time_level 字段
        time_level = getattr(df_data, "freq", getattr(df_data, "time_level", None)).lower()
        # 处理time_level，去掉min，转为int
        if isinstance(time_level, str) and time_level.endswith('min'):
            time_level_num = int(time_level.replace('min', ''))
        else:
            try:
                time_level_num = int(time_level)
            except Exception:
                return {}  # 不能转为数字的直接丢弃
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_time", getattr(df_data, "time", None))),
            "time_level": time_level_num,
            "open": getattr(df_data, "open", None),
            "high": getattr(df_data, "high", None),
            "low": getattr(df_data, "low", None),
            "close": getattr(df_data, "close", None),
            "vol": getattr(df_data, "vol", None),
            "amount": getattr(df_data, "amount", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_time_trade_week_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "open": self._parse_number(getattr(df_data, "open", None)),
            "high": self._parse_number(getattr(df_data, "high", None)),
            "low": self._parse_number(getattr(df_data, "low", None)),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "pre_close": self._parse_number(getattr(df_data, "pre_close", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", getattr(df_data, "pct_change", None))),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_time_trade_month_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "open": self._parse_number(getattr(df_data, "open", None)),
            "high": self._parse_number(getattr(df_data, "high", None)),
            "low": self._parse_number(getattr(df_data, "low", None)),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "pre_close": self._parse_number(getattr(df_data, "pre_close", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", getattr(df_data, "pct_change", None))),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_stock_daily_basic_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "turnover_rate": self._parse_number(getattr(df_data, "turnover_rate", None)),
            "turnover_rate_f": self._parse_number(getattr(df_data, "turnover_rate_f", None)),
            "volume_ratio": self._parse_number(getattr(df_data, "volume_ratio", None)),
            "pe": self._parse_number(getattr(df_data, "pe", None)),
            "pe_ttm": self._parse_number(getattr(df_data, "pe_ttm", None)),
            "pb": self._parse_number(getattr(df_data, "pb", None)),
            "ps": self._parse_number(getattr(df_data, "ps", None)),
            "ps_ttm": self._parse_number(getattr(df_data, "ps_ttm", None)),
            "total_share": self._parse_number(getattr(df_data, "total_share", None)),
            "float_share": self._parse_number(getattr(df_data, "float_share", None)),
            "free_share": self._parse_number(getattr(df_data, "free_share", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "circ_mv": self._parse_number(getattr(df_data, "circ_mv", None)),
            "limit_status": getattr(df_data, "limit_status", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_cyq_perf_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "his_low": self._parse_number(getattr(df_data, "his_low", None)),
            "his_high": self._parse_number(getattr(df_data, "his_high", None)),
            "cost_5pct": self._parse_number(getattr(df_data, "cost_5pct", None)),
            "cost_15pct": self._parse_number(getattr(df_data, "cost_15pct", None)),
            "cost_50pct": self._parse_number(getattr(df_data, "cost_50pct", None)),
            "cost_85pct": self._parse_number(getattr(df_data, "cost_85pct", None)),
            "cost_95pct": self._parse_number(getattr(df_data, "cost_95pct", None)),
            "weight_avg": self._parse_number(getattr(df_data, "weight_avg", None)),
            "winner_rate": self._parse_number(getattr(df_data, "winner_rate", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_cyq_chips_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "price": self._parse_number(getattr(df_data, "price", None)),
            "percent": self._parse_number(getattr(df_data, "percent", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class StockRealtimeDataFormatProcess(BaseDAO):
    # ================ 数据格式 ================
    def set_realtime_tick_data(self, stock: StockInfo, df_data: Any) -> Dict:
        # 兼容不同字段名
        date = getattr(df_data, "DATE", None)
        time = getattr(df_data, "TIME", None)
        trade_time = None
        if date and time:
            trade_time = self._parse_datetime(str(date) + str(time))
        else:
            trade_time = getattr(df_data, "trade_time", None)
        # print(f"DATE: {date}, TIME:{time}, trade_time: {trade_time}")
        data_dict = {
            "stock": stock,
            "trade_time": trade_time,
            "open_price": getattr(df_data, "OPEN", getattr(df_data, "open_price", None)),
            "prev_close_price": getattr(df_data, "PRE_CLOSE", getattr(df_data, "prev_close_price", None)),
            "current_price": getattr(df_data, "PRICE", getattr(df_data, "current_price", None)),
            "high_price": getattr(df_data, "HIGH", getattr(df_data, "high_price", None)),
            "low_price": getattr(df_data, "LOW", getattr(df_data, "low_price", None)),
            "volume": getattr(df_data, "VOLUME", None),
            "turnover_value": getattr(df_data, "AMOUNT", getattr(df_data, "turnover_value", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_level5_data(self, stock: StockInfo, df_data: Any) -> Dict:
        # 兼容不同字段名
        date = getattr(df_data, "DATE", None)
        time = getattr(df_data, "TIME", None)
        trade_time = None
        if date and time:
            trade_time = self._parse_datetime(str(date) + str(time))
        else:
            trade_time = getattr(df_data, "trade_time", None)

        # 买卖盘数据兼容
        b1_v = getattr(df_data, "B1_V", getattr(df_data, "buy_volume1", 0))
        b2_v = getattr(df_data, "B2_V", getattr(df_data, "buy_volume2", 0))
        b3_v = getattr(df_data, "B3_V", getattr(df_data, "buy_volume3", 0))
        b4_v = getattr(df_data, "B4_V", getattr(df_data, "buy_volume4", 0))
        b5_v = getattr(df_data, "B5_V", getattr(df_data, "buy_volume5", 0))
        s1_v = getattr(df_data, "S1_V", getattr(df_data, "sell_volume1", 0))
        s2_v = getattr(df_data, "S2_V", getattr(df_data, "sell_volume2", 0))
        s3_v = getattr(df_data, "S3_V", getattr(df_data, "sell_volume3", 0))
        s4_v = getattr(df_data, "S4_V", getattr(df_data, "sell_volume4", 0))
        s5_v = getattr(df_data, "S5_V", getattr(df_data, "sell_volume5", 0))

        b1_p = getattr(df_data, "B1_P", getattr(df_data, "buy_price1", 0))
        b2_p = getattr(df_data, "B2_P", getattr(df_data, "buy_price2", 0))
        b3_p = getattr(df_data, "B3_P", getattr(df_data, "buy_price3", 0))
        b4_p = getattr(df_data, "B4_P", getattr(df_data, "buy_price4", 0))
        b5_p = getattr(df_data, "B5_P", getattr(df_data, "buy_price5", 0))
        s1_p = getattr(df_data, "S1_P", getattr(df_data, "sell_price1", 0))
        s2_p = getattr(df_data, "S2_P", getattr(df_data, "sell_price2", 0))
        s3_p = getattr(df_data, "S3_P", getattr(df_data, "sell_price3", 0))
        s4_p = getattr(df_data, "S4_P", getattr(df_data, "sell_price4", 0))
        s5_p = getattr(df_data, "S5_P", getattr(df_data, "sell_price5", 0))

        # 盘口差和比率
        try:
            order_diff = b1_v - s1_v
            order_ratio = (b1_v + b2_v + b3_v + b4_v + b5_v) / (s1_v + s2_v + s3_v + s4_v + s5_v) if (s1_v + s2_v + s3_v + s4_v + s5_v) != 0 else 0
        except Exception:
            order_diff = 0
            order_ratio = 0

        data_dict = {
            "stock": stock,
            "trade_time": trade_time,
            "buy_volume1": self._parse_number(b1_v),
            "buy_price1": self._parse_number(b1_p),
            "buy_volume2": self._parse_number(b2_v),
            "buy_price2": self._parse_number(b2_p),
            "buy_volume3": self._parse_number(b3_v),
            "buy_price3": self._parse_number(b3_p),
            "buy_volume4": self._parse_number(b4_v),
            "buy_price4": self._parse_number(b4_p),
            "buy_volume5": self._parse_number(b5_v),
            "buy_price5": self._parse_number(b5_p),
            "sell_volume1": self._parse_number(s1_v),
            "sell_price1": self._parse_number(s1_p),
            "sell_volume2": self._parse_number(s2_v),
            "sell_price2": self._parse_number(s2_p),
            "sell_volume3": self._parse_number(s3_v),
            "sell_price3": self._parse_number(s3_p),
            "sell_volume4": self._parse_number(s4_v),
            "sell_price4": self._parse_number(s4_p),
            "sell_volume5": self._parse_number(s5_v),
            "sell_price5": self._parse_number(s5_p),
            "order_diff": order_diff,
            "order_ratio": order_ratio,
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class StrategiesDataFormatProcess(BaseDAO):
    def set_strategies_data(self, api_data: Dict) -> Dict:
        data_dict = {
            "generated_at": api_data.get('generated_at'),
            "signal": api_data.get('signal'),
            "signal_display": api_data.get('signal_display'),
            "stock_code": api_data.get('stock_code'),
            "strategy_name": api_data.get('strategy_name'),
            "time_level": api_data.get('time_level'),
            "timestamp": api_data.get('timestamp')
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class FundFlowFormatProcess(BaseDAO):
    def set_fund_flow_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "buy_sm_vol": getattr(df_data, "buy_sm_vol", None),
            "buy_sm_amount": getattr(df_data, "buy_sm_amount", None),
            "sell_sm_vol": getattr(df_data, "sell_sm_vol", None),
            "sell_sm_amount": getattr(df_data, "sell_sm_amount", None),
            "buy_md_vol": getattr(df_data, "buy_md_vol", None),
            "buy_md_amount": getattr(df_data, "buy_md_amount", None),
            "sell_md_vol": getattr(df_data, "sell_md_vol", None),
            "sell_md_amount": getattr(df_data, "sell_md_amount", None),
            "buy_lg_vol": getattr(df_data, "buy_lg_vol", None),
            "buy_lg_amount": getattr(df_data, "buy_lg_amount", None),
            "sell_lg_vol": getattr(df_data, "sell_lg_vol", None),
            "sell_lg_amount": getattr(df_data, "sell_lg_amount", None),
            "buy_elg_vol": getattr(df_data, "buy_elg_vol", None),
            "buy_elg_amount": getattr(df_data, "buy_elg_amount", None),
            "sell_elg_vol": getattr(df_data, "sell_elg_vol", None),
            "sell_elg_amount": getattr(df_data, "sell_elg_amount", None),
            "net_mf_vol": getattr(df_data, "net_mf_vol", None),
            "net_mf_amount": getattr(df_data, "net_mf_amount", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_fund_flow_data_ths(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "net_amount": getattr(df_data, "net_amount", None),
            "net_d5_amount": getattr(df_data, "net_d5_amount", None),
            "buy_lg_amount": getattr(df_data, "buy_lg_amount", None),
            "buy_lg_amount_rate": getattr(df_data, "buy_lg_amount_rate", None),
            "buy_md_amount": getattr(df_data, "buy_md_amount", None),
            "buy_md_amount_rate": getattr(df_data, "buy_md_amount_rate", None),
            "buy_sm_amount": getattr(df_data, "buy_sm_amount", None),
            "buy_sm_amount_rate": getattr(df_data, "buy_sm_amount_rate", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_fund_flow_data_dc(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "name": getattr(df_data, "name", None),
            "pct_change": getattr(df_data, "pct_change", None),
            "close": getattr(df_data, "close", None),
            "net_amount": getattr(df_data, "net_amount", None),
            "net_amount_rate": getattr(df_data, "net_amount_rate", None),
            "buy_elg_amount": getattr(df_data, "buy_elg_amount", None),
            "buy_elg_amount_rate": getattr(df_data, "buy_elg_amount_rate", None),
            "buy_lg_amount": getattr(df_data, "buy_lg_amount", None),
            "buy_lg_amount_rate": getattr(df_data, "buy_lg_amount_rate", None),
            "buy_md_amount": getattr(df_data, "buy_md_amount", None),
            "buy_md_amount_rate": getattr(df_data, "buy_md_amount_rate", None),
            "buy_sm_amount": getattr(df_data, "buy_sm_amount", None),
            "buy_sm_amount_rate": getattr(df_data, "buy_sm_amount_rate", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_fund_flow_cnt_ths_data(self, ths_index: 'ThsIndex', df_data: Any) -> Dict:
        data_dict = {
            "ths_index": ths_index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "lead_stock": getattr(df_data, "lead_stock", None),
            "close_price": self._parse_number(getattr(df_data, "close_price", None)),
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "industry_index": self._parse_number(getattr(df_data, "industry_index", None)),
            "company_num": self._parse_number(getattr(df_data, "company_num", None)),
            "pct_change_stock": self._parse_number(getattr(df_data, "pct_change_stock", None)),
            "net_buy_amount": self._parse_number(getattr(df_data, "net_buy_amount", None)),
            "net_sell_amount": self._parse_number(getattr(df_data, "net_sell_amount", None)),
            "net_amount": self._parse_number(getattr(df_data, "net_amount", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_fund_flow_cnt_dc_data(self, dc_index: 'DcIndex', df_data: Any) -> Dict:
        data_dict = {
            "dc_index": dc_index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "content_type": getattr(df_data, "content_type", None),
            "name": getattr(df_data, "name", None),
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "close": self._parse_number(getattr(df_data, "close_price", getattr(df_data, "close", None))),
            "net_amount": self._parse_number(getattr(df_data, "net_amount", None)),
            "net_amount_rate": self._parse_number(getattr(df_data, "net_amount_rate", None)),
            "buy_elg_amount": self._parse_number(getattr(df_data, "buy_elg_amount", None)),
            "buy_elg_amount_rate": self._parse_number(getattr(df_data, "buy_elg_amount_rate", None)),
            "buy_lg_amount": self._parse_number(getattr(df_data, "buy_lg_amount", None)),
            "buy_lg_amount_rate": self._parse_number(getattr(df_data, "buy_lg_amount_rate", None)),
            "buy_md_amount": self._parse_number(getattr(df_data, "buy_md_amount", None)),
            "buy_md_amount_rate": self._parse_number(getattr(df_data, "buy_md_amount_rate", None)),
            "buy_sm_amount": self._parse_number(getattr(df_data, "buy_sm_amount", None)),
            "buy_sm_amount_rate": self._parse_number(getattr(df_data, "buy_sm_amount_rate", None)),
            "buy_sm_amount_stock": getattr(df_data, "buy_sm_amount_stock", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_fund_flow_industry_ths_data(self, ths_index: ThsIndex, df_data: Any) -> Dict:
        data_dict = {
            "ths_index": ths_index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "industry": getattr(df_data, "industry", None),
            "lead_stock": getattr(df_data, "lead_stock", None),
            "close": getattr(df_data, "close", None),
            "pct_change": getattr(df_data, "pct_change", None),
            "company_num": getattr(df_data, "company_num", None),
            "pct_change_stock": getattr(df_data, "pct_change_stock", None),
            "close_price": getattr(df_data, "close_price", None),
            "net_buy_amount": getattr(df_data, "net_buy_amount", None),
            "net_sell_amount": getattr(df_data, "net_sell_amount", None),
            "net_amount": getattr(df_data, "net_amount", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    def set_fund_flow_market_dc_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", getattr(df_data, "trade_time", None))),
            "close_sh": getattr(df_data, "close_sh", None),
            "pct_change_sh": getattr(df_data, "pct_change_sh", None),
            "close_sz": getattr(df_data, "close_sz", None),
            "pct_change_sz": getattr(df_data, "pct_change_sz", None),
            "net_buy_amount": getattr(df_data, "net_buy_amount", None),
            "net_buy_amount_rate": getattr(df_data, "net_buy_amount_rate", None),
            "buy_elg_amount": getattr(df_data, "buy_elg_amount", None),
            "buy_elg_amount_rate": getattr(df_data, "buy_elg_amount_rate", None),
            "buy_lg_amount": getattr(df_data, "buy_lg_amount", None),
            "buy_lg_amount_rate": getattr(df_data, "buy_lg_amount_rate", None),
            "buy_md_amount": getattr(df_data, "buy_md_amount", None),
            "buy_md_amount_rate": getattr(df_data, "buy_md_amount_rate", None),
            "buy_sm_amount": getattr(df_data, "buy_sm_amount", None),
            "buy_sm_amount_rate": getattr(df_data, "buy_sm_amount_rate", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class IndustryFormatProcess(BaseDAO):
    # 申万行业分类
    def set_sw_industry_data(self, index: IndexInfo, df_data: Any) -> Dict:
        data_dict = {
            "index": index,
            "index_code": getattr(df_data, "index_code", None),
            "industry_name": getattr(df_data, "industry_name", None),
            "parent_code": getattr(df_data, "parent_code", None),
            "level": getattr(df_data, "level", None),
            "industry_code": getattr(df_data, "industry_code", None),
            "is_publish": getattr(df_data, "is_publish", None),
            "src": getattr(df_data, "src", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 申万行业成分
    def set_sw_industry_member_data(self, sw_industry: 'SwIndustry', stock: 'StockInfo', df_data: Any) -> Dict:
        data_dict = {
            "l3_industry": sw_industry,
            "stock": stock,
            "l1_code": getattr(df_data, "l1_code", None),
            "l1_name": getattr(df_data, "l1_name", None),
            "l2_code": getattr(df_data, "l2_code", None),
            "l2_name": getattr(df_data, "l2_name", None),
            "l3_name": getattr(df_data, "l3_name", None),
            "name": getattr(df_data, "name", None),
            "in_date": getattr(df_data, "in_date", None),
            "out_date": getattr(df_data, "out_date", None),
            "is_new": getattr(df_data, "is_new", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 申万行业日线行情
    def set_sw_industry_daily_data(self, index: IndexInfo, df_data: Any) -> Dict:
        data_dict = {
            "index": index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_time", None)),
            "name": getattr(df_data, "name", None),
            "open": self._parse_number(getattr(df_data, "open", None)),
            "high": self._parse_number(getattr(df_data, "high", None)),
            "low": self._parse_number(getattr(df_data, "low", None)),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
            "pe": self._parse_number(getattr(df_data, "pe", None)),
            "pb": self._parse_number(getattr(df_data, "pb", None)),
            "float_mv": self._parse_number(getattr(df_data, "float_mv", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "weight": self._parse_number(getattr(df_data, "weight", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 开盘啦题材库
    def set_kpl_concept_data(self, df_data: Any) -> Dict:
        data_dict = {
            "trade_time": getattr(df_data, "trade_time", None),
            "ts_code": getattr(df_data, "ts_code", None),
            "name": getattr(df_data, "name", None),
            "z_t_num": getattr(df_data, "z_t_num", None),
            "up_num": getattr(df_data, "up_num", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 开盘啦题材成分股
    def set_kpl_concept_member_data(self, kpl_concept: 'KplConcept', stock: 'StockInfo', df_data: Any) -> Dict:
        data_dict = {
            "concept": kpl_concept,
            "stock": stock,
            "name": getattr(df_data, "name", None),
            "con_name": getattr(df_data, "con_name", None),
            "trade_time": getattr(df_data, "trade_time", None),
            "desc": getattr(df_data, "desc", None),
            "hot_num": getattr(df_data, "hot_num", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 同花顺概念和行业指数
    def set_ths_index_data(self, df_data: Any) -> Dict:
        raw_count = getattr(df_data, "count", None)
        parsed_count = self._parse_number(raw_count)
        final_count_for_dict = parsed_count if parsed_count is not None else 0
        try:
            final_count_for_dict = int(final_count_for_dict)
        except Exception:
            final_count_for_dict = 0

        data_dict = {
            "ts_code": getattr(df_data, "ts_code", None),
            "name": getattr(df_data, "name", None),
            "count": final_count_for_dict,
            "exchange": getattr(df_data, "exchange", None),
            "list_date": self._parse_datetime(getattr(df_data, "list_date", None)),
            "type": getattr(df_data, "type", None),
        }
        result = {k: safe_value(v) for k, v in data_dict.items()}
        if result.get("count") is None:
            result["count"] = 0
        return result


    # 同花顺概念板块成分
    def set_ths_index_member_data(self, ths_index: 'ThsIndex', stock: 'StockInfo', df_data: Any) -> Dict:
        data_dict = {
            "ths_index": ths_index,
            "stock": stock,
            "weight": getattr(df_data, "weight", None),
            "in_date": self._parse_datetime(getattr(df_data, "in_date", None)),
            "out_date": self._parse_datetime(getattr(df_data, "out_date", None)),
            "is_new": getattr(df_data, "is_new", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 同花顺板块指数行情
    def set_ths_index_daily_data(self, ths_index: 'ThsIndex', df_data: Any) -> Dict:
        data_dict = {
            "ths_index": ths_index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "open": self._parse_number(getattr(df_data, "open", None)),
            "high": self._parse_number(getattr(df_data, "high", None)),
            "low": self._parse_number(getattr(df_data, "low", None)),
            "pre_close": self._parse_number(getattr(df_data, "pre_close", None)),
            "avg_price": self._parse_number(getattr(df_data, "avg_price", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "turnover_rate": self._parse_number(getattr(df_data, "turnover_rate", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "float_mv": self._parse_number(getattr(df_data, "float_mv", None)),
            "pe_ttm": self._parse_number(getattr(df_data, "pe_ttm", None)),
            "pb_mrq": self._parse_number(getattr(df_data, "pb_mrq", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 东方财富概念板块
    def set_dc_index_data(self, stock: 'StockInfo', df_data: Any) -> Dict:
        data_dict = {
            "ts_code": getattr(df_data, "ts_code", None),
            "name": getattr(df_data, "name", None),
            "exchange": getattr(df_data, "exchange", None),
            "type": getattr(df_data, "type", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 东方财富板块成分
    def set_dc_member_data(self, dc_index: 'DcIndex', stock: 'StockInfo', df_data: Any) -> Dict:
        data_dict = {
            "trade_time": getattr(df_data, "trade_time", None),
            "dc_index": dc_index,
            "stock": stock,
            "name": getattr(df_data, "name", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 东方财富板块指数行情
    def set_dc_index_daily_data(self, stock: 'StockInfo', dc_index: 'DcIndex', df_data: Any) -> Dict:
        data_dict = {
            "dc_index": dc_index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_time", None)),
            "name": getattr(df_data, "name", None),
            "leading": getattr(df_data, "leading", None),
            "stock": stock,
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "leading_pct": self._parse_number(getattr(df_data, "leading_pct", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "turnover_rate": self._parse_number(getattr(df_data, "turnover_rate", None)),
            "up_num": self._parse_number(getattr(df_data, "up_num", None)),
            "down_num": self._parse_number(getattr(df_data, "down_num", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

class MarketFormatProcess(BaseDAO):
    # 市场交易统计(MarketDailyInfo)
    def set_market_daily_info_data(self, df_data: Any) -> Dict:
        data_dict = {
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "ts_code": getattr(df_data, "ts_code", None),
            "ts_name": getattr(df_data, "ts_name", None),
            "com_count": self._parse_number(getattr(df_data, "com_count", None)),
            "total_share": self._parse_number(getattr(df_data, "total_share", None)),
            "float_share": self._parse_number(getattr(df_data, "float_share", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "float_mv": self._parse_number(getattr(df_data, "float_mv", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "trans_count": self._parse_number(getattr(df_data, "trans_count", None)),
            "pe": self._parse_number(getattr(df_data, "pe", None)),
            "trans_rate": self._parse_number(getattr(df_data, "trans_rate", None)),
            "exchange": getattr(df_data, "exchange", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 游资名录
    def set_hm_list_data(self, df_data: Any) -> Dict:
        data_dict = {
            "name": getattr(df_data, "name", None),
            "desc": getattr(df_data, "desc", None),
            "orgs": getattr(df_data, "orgs", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 游资每日明细
    def set_hm_detail_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "ts_name": getattr(df_data, "ts_name", None),
            "buy_amount": self._parse_number(getattr(df_data, "buy_amount", None)),
            "sell_amount": self._parse_number(getattr(df_data, "sell_amount", None)),
            "net_amount": self._parse_number(getattr(df_data, "net_amount", None)),
            "hm_name": getattr(df_data, "hm_name", None),
            "hm_orgs": getattr(df_data, "hm_orgs", None),
            "tag": getattr(df_data, "tag", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 同花顺板块指数行情
    def set_ths_daily_data(self, ths_index: 'ThsIndex', df_data: Any) -> Dict:
        data_dict = {
            "ths_index": ths_index,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "open": self._parse_number(getattr(df_data, "open", None)),
            "high": self._parse_number(getattr(df_data, "high", None)),
            "low": self._parse_number(getattr(df_data, "low", None)),
            "pre_close": self._parse_number(getattr(df_data, "pre_close", None)),
            "avg_price": self._parse_number(getattr(df_data, "avg_price", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "turnover_rate": self._parse_number(getattr(df_data, "turnover_rate", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "float_mv": self._parse_number(getattr(df_data, "float_mv", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 涨跌停榜单 - 同花顺
    def set_limit_list_ths_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "name": getattr(df_data, "name", None),
            "price": self._parse_number(getattr(df_data, "price", None)),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", None)),
            "open_num": self._parse_number(getattr(df_data, "open_num", None)),
            "lu_desc": getattr(df_data, "lu_desc", None),
            "limit_type": getattr(df_data, "limit_type", None),
            "tag": getattr(df_data, "tag", None),
            "status": getattr(df_data, "status", None),
            "first_lu_time": self._parse_datetime(getattr(df_data, "first_lu_time", None)),
            "last_lu_time": self._parse_datetime(getattr(df_data, "last_lu_time", None)),
            "first_ld_time": self._parse_datetime(getattr(df_data, "first_ld_time", None)),
            "last_ld_time": self._parse_datetime(getattr(df_data, "last_ld_time", None)),
            "limit_order": self._parse_number(getattr(df_data, "limit_order", None)),
            "limit_amount": self._parse_number(getattr(df_data, "limit_amount", None)),
            "turnover": self._parse_number(getattr(df_data, "turnover", None)),
            "rise_rate": self._parse_number(getattr(df_data, "rise_rate", None)),
            "sum_float": self._parse_number(getattr(df_data, "sum_float", None)),
            "market_type": getattr(df_data, "market_type", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 涨跌停列表
    def set_limit_list_d_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "industry": getattr(df_data, "industry", None),
            "name": getattr(df_data, "name", None),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
            "limit_amount": self._parse_number(getattr(df_data, "limit_amount", None)),
            "float_mv": self._parse_number(getattr(df_data, "float_mv", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "turnover_ratio": self._parse_number(getattr(df_data, "turnover_ratio", None)),
            "fd_amount": self._parse_number(getattr(df_data, "fd_amount", None)),
            "first_time": self._parse_datetime(getattr(df_data, "first_time", None)),
            "last_time": self._parse_datetime(getattr(df_data, "last_time", None)),
            "open_times": self._parse_datetime(getattr(df_data, "open_times", None)),
            "up_stat": getattr(df_data, "up_stat", None),
            "limit_times": self._parse_number(getattr(df_data, "limit_times", None)),
            "limit": getattr(df_data, "limit", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 连板天梯
    def set_limit_step_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "name": getattr(df_data, "name", None),
            "nums": self._parse_number(getattr(df_data, "nums", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}

    # 最强板块统计 - 同花顺
    def set_limit_cpt_list_data(self, ths_index: 'ThsIndex', df_data: Any) -> Dict:
        data_dict = {
            "ths_index": ths_index,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "name": getattr(df_data, "name", None),
            "days": self._parse_number(getattr(df_data, "days", None)),
            "up_stat": getattr(df_data, "up_stat", None),
            "cons_nums": self._parse_number(getattr(df_data, "cons_nums", None)),
            "up_nums": getattr(df_data, "up_nums", None),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", None)),
            "rank": getattr(df_data, "rank", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}


















