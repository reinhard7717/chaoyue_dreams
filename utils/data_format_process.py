import datetime
import decimal
from django.utils import timezone
from typing import Any, Dict, Optional
import logging
import numpy as np
import math
# 导入 Django 的 Model 基类，用于判断是否是模型实例
from django.db.models import Model
from dao_manager.base_dao import BaseDAO
from stock_models.index import IndexInfo
from stock_models.industry import (
    ConceptMaster, ConceptDaily, # 新增导入
    DcIndex, KplConceptInfo, KplConceptDaily, KplConceptConstituent, KplLimitList, SwIndustry, ThsIndex
)
from stock_models.stock_basic import StockInfo
from users.models import FavoriteStock
from utils.cache_manager import CacheManager

logger = logging.getLogger(__name__)

# 对所有字段做一次NaN/None清洗
def safe_value(val):
    # 首先检查值是否是 Django 模型实例，如果是则直接返回
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
    return val

class UserDataFormatProcess(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
        data_dict = {
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_time", getattr(df_data, "time", None))),
            # "time_level": time_level_num,
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
            "open": self._parse_number(getattr(df_data, "open", getattr(df_data, "open_qfq", None))),
            "high": self._parse_number(getattr(df_data, "high", getattr(df_data, "high_qfq", None))),
            "low": self._parse_number(getattr(df_data, "low", getattr(df_data, "low_qfq", None))),
            "close": self._parse_number(getattr(df_data, "close", getattr(df_data, "close_qfq", None))),
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
            "open": self._parse_number(getattr(df_data, "open", getattr(df_data, "open_qfq", None))),
            "high": self._parse_number(getattr(df_data, "high", getattr(df_data, "high_qfq", None))),
            "low": self._parse_number(getattr(df_data, "low", getattr(df_data, "low_qfq", None))),
            "close": self._parse_number(getattr(df_data, "close", getattr(df_data, "close_qfq", None))),
            "pre_close": self._parse_number(getattr(df_data, "pre_close", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", getattr(df_data, "pct_change", None))),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    def set_stock_daily_basic_data(self, stock: StockInfo, df_data: Any) -> Dict:
        # 负责将从Tushare获取的单行数据（通常是DataFrame的一行）转换为符合数据库模型要求的字典。
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
            "dv_ratio": self._parse_number(getattr(df_data, "dv_ratio", None)),
            "dv_ttm": self._parse_number(getattr(df_data, "dv_ttm", None)),
            "total_share": self._parse_number(getattr(df_data, "total_share", None)),
            "float_share": self._parse_number(getattr(df_data, "float_share", None)),
            "free_share": self._parse_number(getattr(df_data, "free_share", None)),
            "total_mv": self._parse_number(getattr(df_data, "total_mv", None)),
            "circ_mv": self._parse_number(getattr(df_data, "circ_mv", None)),
            "limit_status": getattr(df_data, "limit_status", None),
        }
        # 使用 safe_value 函数对字典中的所有值进行最终处理，确保数据安全
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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
    # ================ 数据格式 ================
    def set_realtime_tick_data(self, stock: Optional[StockInfo], df_data: Any) -> Dict:
        """
        【V3.0 - 单位修正版】
        - 修正了对 _parse_number 的调用，移除了非法的 to_type 参数。
        - 修正了 volume 的单位处理逻辑，根据 sina 接口规范，volume 单位是“股”，不再乘以100。
        """
        date = getattr(df_data, "DATE", None)
        time = getattr(df_data, "TIME", None)
        trade_time = self._parse_datetime(f"{date}{time}") if date and time else None
        # 先用 _parse_number 获取 Decimal 类型的值
        volume_decimal = self._parse_number(getattr(df_data, "VOLUME", None))
        data_dict = {
            "stock": stock,
            "trade_time": trade_time,
            "open_price": self._parse_number(getattr(df_data, "OPEN", None)),
            "prev_close_price": self._parse_number(getattr(df_data, "PRE_CLOSE", None)),
            "current_price": self._parse_number(getattr(df_data, "PRICE", None)),
            "high_price": self._parse_number(getattr(df_data, "HIGH", None)),
            "low_price": self._parse_number(getattr(df_data, "LOW", None)),
            # sina接口的VOLUME单位是“股”，直接转换为int即可
            "volume": int(volume_decimal) if volume_decimal is not None else None,
            # sina接口的AMOUNT单位是“元”
            "turnover_value": self._parse_number(getattr(df_data, "AMOUNT", None)),
        }
        return {k: v for k, v in data_dict.items() if v is not None}
    def set_level5_data(self, stock: Optional[StockInfo], df_data: Any) -> Dict:
        """
        【V3.0 - 单位修正版】
        - 修正了对 _parse_number 的调用，移除了非法的 to_type 参数。
        - 在 _parse_number 之后进行类型转换和单位乘法。
        """
        date = getattr(df_data, "DATE", None)
        time = getattr(df_data, "TIME", None)
        trade_time = self._parse_datetime(f"{date}{time}") if date and time else None
        def _process_volume(value: Any) -> Optional[int]:
            """辅助函数，用于处理盘口量：解析 -> 乘100 -> 转int"""
            parsed_val = self._parse_number(value)
            if parsed_val is not None:
                return int(parsed_val * 100)
            return None
        data_dict = {
            "stock": stock,
            "trade_time": trade_time,
            # sina接口的买卖盘量单位是“手”，数据库需要存“股”，所以乘以100
            "buy_volume1": _process_volume(getattr(df_data, "B1_V", None)),
            "buy_price1": self._parse_number(getattr(df_data, "B1_P", None)),
            "buy_volume2": _process_volume(getattr(df_data, "B2_V", None)),
            "buy_price2": self._parse_number(getattr(df_data, "B2_P", None)),
            "buy_volume3": _process_volume(getattr(df_data, "B3_V", None)),
            "buy_price3": self._parse_number(getattr(df_data, "B3_P", None)),
            "buy_volume4": _process_volume(getattr(df_data, "B4_V", None)),
            "buy_price4": self._parse_number(getattr(df_data, "B4_P", None)),
            "buy_volume5": _process_volume(getattr(df_data, "B5_V", None)),
            "buy_price5": self._parse_number(getattr(df_data, "B5_P", None)),
            "sell_volume1": _process_volume(getattr(df_data, "A1_V", None)),
            "sell_price1": self._parse_number(getattr(df_data, "A1_P", None)),
            "sell_volume2": _process_volume(getattr(df_data, "A2_V", None)),
            "sell_price2": self._parse_number(getattr(df_data, "A2_P", None)),
            "sell_volume3": _process_volume(getattr(df_data, "A3_V", None)),
            "sell_price3": self._parse_number(getattr(df_data, "A3_P", None)),
            "sell_volume4": _process_volume(getattr(df_data, "A4_V", None)),
            "sell_price4": self._parse_number(getattr(df_data, "A4_P", None)),
            "sell_volume5": _process_volume(getattr(df_data, "A5_V", None)),
            "sell_price5": self._parse_number(getattr(df_data, "A5_P", None)),
        }
        return {k: v for k, v in data_dict.items() if v is not None}

class StrategiesDataFormatProcess(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
    # 申万行业分类
    def set_sw_industry_data(self, index: IndexInfo, df_data: Any) -> Dict:
        data_dict = {
            "index": index,
            "index_code": getattr(df_data, "index_code", None),
            "industry_name": getattr(df_data, "industry_name", None),
            "parent_code": getattr(df_data, "parent_code", None),
            "level": getattr(df_data, "level", None),
            "industry_code": getattr(df_data, "industry_code", None),
            "is_pub": getattr(df_data, "is_pub", None),
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
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", None)),
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
        # return {k: safe_value(v) for k, v in data_dict.items()}
        return data_dict
    # 开盘啦题材字典
    def set_kpl_concept_info_data(self, df_data: Any) -> Dict:
        """【V2.0 新增】用于格式化 KplConceptInfo 主表数据"""
        data_dict = {
            "ts_code": getattr(df_data, "ts_code", None),
            "name": getattr(df_data, "name", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    # 开盘啦题材每日快照 (原 set_kpl_concept_data)
    def set_kpl_concept_daily_data(self, concept_info: 'KplConceptInfo', df_data: Any) -> Dict:
        """【V2.0 重构】用于格式化 KplConceptDaily 每日快照数据"""
        data_dict = {
            "concept_info": concept_info,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "z_t_num": self._parse_number(getattr(df_data, "z_t_num", None)),
            "up_num": self._parse_number(getattr(df_data, "up_num", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    # 开盘啦题材成分股
    def set_kpl_concept_member_data(self, concept_info: 'KplConceptInfo', stock: 'StockInfo', df_data: Any) -> Dict:
        """【V2.0 重构】更新外键为 KplConceptInfo"""
        data_dict = {
            "concept_info": concept_info, # 外键对象修改
            "stock": stock,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "desc": getattr(df_data, "desc", None),
            "hot_num": self._parse_number(getattr(df_data, "hot_num", None)),
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
    def set_dc_index_data(self, df_data: Any) -> Dict:
        data_dict = {
            "ts_code": getattr(df_data, "ts_code", None),
            "name": getattr(df_data, "name", None),
            "exchange": "DC",
            "type": "C",
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    # 东方财富板块成分
    def set_dc_index_member_data(self, dc_index: 'DcIndex', stock: 'StockInfo', df_data: Any) -> Dict:
        data_dict = {
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "dc_index": dc_index,
            "stock": stock,
            "name": getattr(df_data, "name", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    # 东方财富板块指数行情
    def set_dc_index_daily_data(self, dc_index: 'DcIndex', df_data: Any) -> Dict:
        """
        【V2.0 重构版】
        根据重构后的 DcIndexDaily 模型和 dc_daily API 的实际返回字段进行数据格式化。
        移除了不再需要的 leading_stock 参数。
        """
        data_dict = {
            "dc_index": dc_index,
            "trade_time": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "close": self._parse_number(getattr(df_data, "close", None)),
            "open": self._parse_number(getattr(df_data, "open", None)),
            "high": self._parse_number(getattr(df_data, "high", None)),
            "low": self._parse_number(getattr(df_data, "low", None)),
            "change": self._parse_number(getattr(df_data, "change", None)),
            "pct_change": self._parse_number(getattr(df_data, "pct_change", None)),
            "vol": self._parse_number(getattr(df_data, "vol", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
            "swing": self._parse_number(getattr(df_data, "swing", None)),
            "turnover_rate": self._parse_number(getattr(df_data, "turnover_rate", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    # --- 新增区域: ConceptDaily 适配器 ---
    def adapt_to_concept_daily(self, source: str, daily_dict: dict, concept_master: 'ConceptMaster') -> 'ConceptDaily':
        """
        【V3.0 总适配器】根据来源调用相应的具体适配器。
        """
        if source == 'sw':
            return self._adapt_sw_daily(daily_dict, concept_master)
        elif source == 'ths':
            return self._adapt_ths_daily(daily_dict, concept_master)
        elif source == 'dc':
            return self._adapt_dc_daily(daily_dict, concept_master)
        elif source == 'ci':
            return self._adapt_ci_daily(daily_dict, concept_master)
        else:
            # 提供一个通用后备，以防未来增加新来源
            return self._adapt_generic_daily(daily_dict, concept_master)
    def _adapt_ths_daily(self, daily_dict: dict, concept_master: 'ConceptMaster') -> 'ConceptDaily':
        """将 ThsIndexDaily 的数据字典适配到 ConceptDaily 模型实例。"""
        return ConceptDaily(
            concept=concept_master,
            trade_date=daily_dict.get('trade_time'),
            open=daily_dict.get('open'),
            high=daily_dict.get('high'),
            low=daily_dict.get('low'),
            close=daily_dict.get('close'),
            pre_close=daily_dict.get('pre_close'),
            pct_change=daily_dict.get('pct_change'),
            vol=daily_dict.get('vol'),  # 同花顺单位是'手'
            amount=daily_dict.get('amount'), # 同花顺单位是'千元'
            turnover_rate=daily_dict.get('turnover_rate')
        )
    def _adapt_sw_daily(self, daily_dict: dict, concept_master: 'ConceptMaster') -> 'ConceptDaily':
        """将 SwIndustryDaily 的数据字典适配到 ConceptDaily 模型实例。"""
        # 单位转换：申万成交量是“万股”，成交额是“万元”
        vol_standard = daily_dict.get('vol') * 100 if daily_dict.get('vol') is not None else None # 万股 -> 手
        amount_standard = daily_dict.get('amount') * 1000 if daily_dict.get('amount') is not None else None # 万元 -> 千元
        return ConceptDaily(
            concept=concept_master,
            trade_date=daily_dict.get('trade_time'),
            open=daily_dict.get('open'),
            high=daily_dict.get('high'),
            low=daily_dict.get('low'),
            close=daily_dict.get('close'),
            pre_close=None, # 申万API不提供昨收
            pct_change=daily_dict.get('pct_change'),
            vol=vol_standard,
            amount=amount_standard,
            turnover_rate=None # 申万API不提供换手率
        )
    def _adapt_dc_daily(self, daily_dict: dict, concept_master: 'ConceptMaster') -> 'ConceptDaily':
        """将 DcIndexDaily 的数据字典适配到 ConceptDaily 模型实例。"""
        # 东方财富单位：成交量是'手'，成交额是'千元'，与我们的目标单位一致
        return ConceptDaily(
            concept=concept_master,
            trade_date=daily_dict.get('trade_time'),
            open=daily_dict.get('open'),
            high=daily_dict.get('high'),
            low=daily_dict.get('low'),
            close=daily_dict.get('close'),
            pre_close=None, # 东方财富API不提供昨收
            pct_change=daily_dict.get('pct_change'),
            vol=daily_dict.get('vol'),
            amount=daily_dict.get('amount'),
            turnover_rate=daily_dict.get('turnover_rate')
        )
    def _adapt_ci_daily(self, daily_dict: dict, concept_master: 'ConceptMaster') -> 'ConceptDaily':
        """将 CiDaily 的数据字典适配到 ConceptDaily 模型实例。"""
        # 中信单位：成交量是'万股'，成交额是'万元'
        vol_standard = daily_dict.get('vol') * 100 if daily_dict.get('vol') is not None else None # 万股 -> 手
        amount_standard = daily_dict.get('amount') * 1000 if daily_dict.get('amount') is not None else None # 万元 -> 千元
        return ConceptDaily(
            concept=concept_master,
            trade_date=daily_dict.get('trade_time'),
            open=daily_dict.get('open'),
            high=daily_dict.get('high'),
            low=daily_dict.get('low'),
            close=daily_dict.get('close'),
            pre_close=daily_dict.get('pre_close'),
            pct_change=daily_dict.get('pct_change'),
            vol=vol_standard,
            amount=amount_standard,
            turnover_rate=None # 中信API不提供换手率
        )
    def _adapt_generic_daily(self, daily_dict: dict, concept_master: 'ConceptMaster') -> 'ConceptDaily':
        """一个通用的、尽力而为的后备适配器。"""
        return ConceptDaily(
            concept=concept_master,
            trade_date=daily_dict.get('trade_time') or daily_dict.get('trade_date'),
            open=daily_dict.get('open'),
            high=daily_dict.get('high'),
            low=daily_dict.get('low'),
            close=daily_dict.get('close'),
            pre_close=daily_dict.get('pre_close'),
            pct_change=daily_dict.get('pct_change') or daily_dict.get('pct_chg'),
            vol=daily_dict.get('vol'),
            amount=daily_dict.get('amount'),
            turnover_rate=daily_dict.get('turnover_rate') or daily_dict.get('turnover_ratio')
        )


class MarketFormatProcess(BaseDAO):
    def __init__(self, cache_manager_instance: CacheManager):
        """
        初始化 StockInfoFormatProcess。
        由于继承自 BaseDAO，必须接收 cache_manager_instance 并传递给父类。
        """
        # 调用父类的 __init__ 方法，并将“接力棒”传递下去
        # 因为这个类不直接操作某个特定模型，所以 model_class 可以是 None
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
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
            "open_num": self._parse_number(getattr(df_data, "open_num", None)) or 0,
            "lu_desc": getattr(df_data, "lu_desc", None) or '',
            "limit_type": getattr(df_data, "limit_type", None),
            "tag": getattr(df_data, "tag", None) or '',
            "status": getattr(df_data, "status", None) or '',
            "first_lu_time": getattr(df_data, "first_lu_time", None), # 保持为字符串
            "last_lu_time": getattr(df_data, "last_lu_time", None), # 保持为字符串
            "first_ld_time": getattr(df_data, "first_ld_time", None), # 保持为字符串
            "last_ld_time": getattr(df_data, "last_ld_time", None), # 保持为字符串
            "limit_order": self._parse_number(getattr(df_data, "limit_order", None)),
            "limit_amount": self._parse_number(getattr(df_data, "limit_amount", None)),
            "turnover": self._parse_number(getattr(df_data, "turnover", None)),
            "rise_rate": self._parse_number(getattr(df_data, "rise_rate", None)),
            "sum_float": self._parse_number(getattr(df_data, "sum_float", None)),
            "market_type": getattr(df_data, "market_type", None),
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
    def set_kpl_list_data(self, stock: StockInfo, df_data: Any) -> Dict:
        """
        将Tushare的kpl_list接口返回的单行数据，格式化为准备入库的字典。
        """
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(getattr(df_data, "trade_date", None)),
            "name": getattr(df_data, "name", None),
            "tag": getattr(df_data, "tag", None),
            "lu_time": getattr(df_data, "lu_time", None),
            "ld_time": getattr(df_data, "ld_time", None),
            "open_time": getattr(df_data, "open_time", None),
            "last_time": getattr(df_data, "last_time", None),
            "lu_desc": getattr(df_data, "lu_desc", None),
            "theme": getattr(df_data, "theme", None),
            "status": getattr(df_data, "status", None),
            "pct_chg": self._parse_number(getattr(df_data, "pct_chg", None)),
            "turnover_rate": self._parse_number(getattr(df_data, "turnover_rate", None)),
            "amount": self._parse_number(getattr(df_data, "amount", None)),
            "limit_order": self._parse_number(getattr(df_data, "limit_order", None)),
            "lu_limit_order": self._parse_number(getattr(df_data, "lu_limit_order", None)),
            "free_float": self._parse_number(getattr(df_data, "free_float", None)),
            "bid_change": self._parse_number(getattr(df_data, "bid_change", None)),
            "bid_turnover": self._parse_number(getattr(df_data, "bid_turnover", None)),
            "bid_pct_chg": self._parse_number(getattr(df_data, "bid_pct_chg", None)),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}
    # 涨跌停列表
    def set_limit_list_d_data(self, stock: StockInfo, df_data: Any) -> Dict:
        """
        将Tushare的limit_list_d接口返回的单行数据，格式化为准备入库的字典。
        """
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
            "first_time": getattr(df_data, "first_time", None), # 保持为字符串
            "last_time": getattr(df_data, "last_time", None), # 保持为字符串
            "open_times": self._parse_number(getattr(df_data, "open_times", None)),
            "up_stat": getattr(df_data, "up_stat", None) or '',
            "limit_times": self._parse_number(getattr(df_data, "limit_times", None)) or 0,
            "limit": getattr(df_data, "limit", None),
        }
        return {k: safe_value(v) for k, v in data_dict.items()}













