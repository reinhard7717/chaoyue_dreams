from django.utils import timezone
from typing import Any, Dict
import logging
from dao_manager.base_dao import BaseDAO
from stock_models.fund_flow import FundFlowCntDC, FundFlowCntTHS, FundFlowDaily, FundFlowIndustryTHS, FundFlowMarketDc
from stock_models.index import IndexDailyBasic, IndexInfo, IndexWeight
from stock_models.industry import DcIndex, DcMember, KplConcept, SwIndustry, ThsIndex, ThsMember
from stock_models.market import HmDetail, HmList, LimitCptList, LimitListD, LimitListThs, LimitStep, MarketDailyInfo, ThsDaily
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import StockLevel5Data, StockRealtimeData
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyBasic, StockDailyData, StockMinuteData, StockMonthlyData, StockTimeTrade, StockWeeklyData
from users.models import FavoriteStock

logger = logging.getLogger(__name__)

class UserDataFormatProcess(BaseDAO):
    def set_user_favorites(self, user_id: int, api_data: FavoriteStock) -> Dict:
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
        return data_dict

class IndexDataFormatProcess(BaseDAO):
    # 指数基础信息
    def set_index_info_data(self, api_data: Any) -> Dict:
        if isinstance(api_data, IndexInfo):
            data_dict = {
                "index_code": api_data.index_code,  # 指数代码
                "name": api_data.name,  # 简称
                "fullname": api_data.fullname,  # 指数全称
                "market": api_data.market,  # 市场
                "publisher": api_data.publisher,  # 发布方
                "index_type": api_data.index_type,  # 指数风格
                "category": api_data.category,  # 指数类别
                "base_date": api_data.base_date,  # 基期
                "base_point": api_data.base_point,  # 基点
                "list_date": api_data.list_date,  # 发布日期
                "weight_rule": api_data.weight_rule,  # 加权方式
                "desc": api_data.desc,  # 描述
                "exp_date": api_data.exp_date,  # 终止日期
            }
        else:
            data_dict = {
                "index_code": api_data.index_code,  # 指数代码
                "name": api_data.name,  # 简称
                "fullname": api_data.fullname,  # 指数全称
                "market": api_data.market,  # 市场
                "publisher": api_data.publisher,  # 发布方
                "index_type": api_data.index_type,  # 指数风格
                "category": api_data.category,  # 指数类别
                "base_date": self._parse_datetime(api_data.base_date),  # 基期
                "base_point": self._parse_number(api_data.base_point),  # 基点
                "list_date": api_data.list_date,  # 发布日期
                "weight_rule": api_data.weight_rule,  # 加权方式
                "desc": api_data.desc,  # 描述
                "exp_date": api_data.exp_date,  # 终止日期
            }
        return data_dict

    # 指数成分和权重
    def set_index_weight_data(self, index_info: IndexInfo, api_data: Any) -> Dict:
        if isinstance(api_data, IndexWeight):
            data_dict = {
                "index": index_info,  # 指数代码
                "stock": api_data.stock_code,  # 股票代码
                "trade_date": api_data.trade_date,  # 交易日期
                "weight": api_data.weight,  # 权重
            }
        else:
            data_dict = {
                "index": index_info,  # 指数代码
                "stock": api_data.stock,  # 股票代码
                "trade_date": self._parse_datetime(api_data.trade_date),  # 交易日期
                "weight": self._parse_number(api_data.weight),  # 权重
            }
        return data_dict

    # 大盘指数每日指标
    def set_index_daily_basic_data(self, api_data: Any) -> Dict:
        if isinstance(api_data, IndexDailyBasic):
            data_dict = {
                "index": index_info,  # 指数代码
                "trade_date": api_data.trade_date,  # 交易日期
                "total_mv": api_data.total_mv,  # 总市值
                "float_mv": api_data.float_mv,  # 流通市值
                "total_share": api_data.total_share,  # 总股本
                "float_share": api_data.float_share,  # 流通股本
                "free_share": api_data.free_share,  # 自由流通股本
                "turnover_rate": api_data.turnover_rate,  # 换手率
                "turnover_rate_f": api_data.turnover_rate_f,  # 换手率(自由流通)
                "pe": api_data.pe,  # 市盈率
                "pe_ttm": api_data.pe_ttm,  # 市盈率TTM
                "pb": api_data.pb,  # 市净率
            }
        else:
            data_dict = {
                "index": index_info,  # 指数代码
                "trade_date": self._parse_datetime(api_data.trade_date),  # 交易日期
                "total_mv": self._parse_number(api_data.total_mv),  # 总市值
                "float_mv": self._parse_number(api_data.float_mv),  # 流通市值
                "total_share": self._parse_number(api_data.total_share),  # 总股本
                "float_share": self._parse_number(api_data.float_share),  # 流通股本
                "free_share": self._parse_number(api_data.free_share),  # 自由流通股本
                "turnover_rate": self._parse_number(api_data.turnover_rate),  # 换手率
                "turnover_rate_f": self._parse_number(api_data.turnover_rate_f),  # 换手率(自由流通)
                "pe": self._parse_number(api_data.pe),  # 市盈率
                "pe_ttm": self._parse_number(api_data.pe_ttm),  # 市盈率TTM
                "pb": self._parse_number(api_data.pb),  # 市净率
            }
        return data_dict

class StockInfoFormatProcess(BaseDAO):
    def set_stock_info_data(self, api_data: Any) -> Dict:
        if isinstance(api_data, StockInfo):
            data_dict = {
                'stock_code': api_data.stock_code,  # 股票代码
                'stock_name': api_data.stock_name,  # 股票名称
                'area': api_data.area,  # 地域
                'industry': api_data.industry,  # 所属行业
                'full_name': api_data.full_name,  # 股票全称
                'en_name': api_data.en_name,  # 英文全称
                'cn_spell': api_data.cn_spell,  # 拼音缩写
                'market_type': api_data.market_type,  # 市场类型
                'exchange': api_data.exchange,  # 交易所代码
                'currency_type': api_data.currency_type,  # 交易货币
                'list_status': api_data.list_status,  # 上市状态
                'list_date': api_data.list_date,  # 上市日期
                'delist_date': api_data.delist_date,  # 退市日期
                'is_hs': api_data.is_hs,  # 是否沪深港通标的
                'actual_controller': api_data.actual_controller,  # 实控人名称
                'actual_controller_type': api_data.actual_controller_type,  # 实控人企业性质
            }
        else:
            data_dict = {
                'stock_code': api_data.ts_code,  # 股票代码
                'stock_name': api_data.name,  # 股票名称
                'area': api_data.area,  # 地域
                'industry': api_data.industry,  # 所属行业
                'full_name': api_data.fullname,  # 股票全称
                'en_name': api_data.enname,  # 英文全称
                'cn_spell': api_data.cnspell,  # 拼音缩写
                'market_type': api_data.market,  # 市场类型
                'exchange': api_data.exchange,  # 交易所代码
                'currency_type': api_data.curr_type,  # 交易货币
                'list_status': api_data.list_status,  # 上市状态
                'list_date': api_data.list_date,  # 上市日期
                'delist_date': api_data.delist_date,  # 退市日期
                'is_hs': api_data.is_hs,  # 是否沪深港通标的
                'actual_controller': api_data.act_name,  # 实控人名称
                'actual_controller_type': api_data.act_ent_type,  # 实控人企业性质
            }
        return data_dict
    
    def set_stock_info_basic_data(self, api_data: Any) -> Dict:
        if isinstance(api_data, StockInfo):
             data_dict = {
                'stock_code': api_data.stock_code,  # 股票代码
                'stock_name': api_data.stock_name,  # 股票名称
                'industry': api_data.industry,  # 所属行业
                'market_type': api_data.market_type,  # 市场类型
                'exchange': api_data.exchange,  # 交易所代码
                'currency_type': api_data.currency_type,  # 交易货币
                'list_status': api_data.list_status,  # 上市状态
                'is_hs': api_data.is_hs,  # 是否沪深港通标的
            }
        else:
            data_dict = {
                'stock_code': api_data.ts_code,  # 股票代码
                'stock_name': api_data.name,  # 股票名称
                'industry': api_data.industry,  # 所属行业
                'market_type': api_data.market,  # 市场类型
                'exchange': api_data.exchange,  # 交易所代码
                'list_status': api_data.list_status,  # 上市状态
                'is_hs': api_data.is_hs,  # 是否沪深港通标的
            }
        return data_dict

    def set_company_info_data(self, stock: StockInfo, api_data: Any) -> Dict:
        data_dict = {
            'stock': stock,
            'com_name': api_data.com_name,
            'com_id': api_data.com_id,
            'exchange': api_data.exchange,
            'chairman': api_data.chairman,
            'manager': api_data.manager,
            'secretary': api_data.secretary,
            'reg_capital': api_data.reg_capital,
            'setup_date': api_data.setup_date,
            'province': api_data.province,
            'city': api_data.city,
            'introduction': api_data.introduction,
            'website': api_data.website,
            'email': api_data.email,
            'office': api_data.office,
            'employees': api_data.employees,
            'main_business': api_data.main_business,
            'business_scope': api_data.business_scope,
        }
        return data_dict

    def set_hs_const_data(self, stock: StockInfo, api_data: Any) -> Dict:
        data_dict = {
            'stock': stock,
            'hs_type': api_data.hs_type,
            'in_date': api_data.in_date,
            'out_date': api_data.out_date,
            'is_new': api_data.is_new,
        }
        return data_dict

class StockTimeTradeFormatProcess(BaseDAO):
    def set_time_trade_day_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockDailyData):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "pre_close": df_data.pre_close,
                "change": df_data.change,
                "pct_change": df_data.pct_change,
                "vol": df_data.vol,
                "amount": df_data.amount,
                "adj_factor": df_data.adj_factor,
                "open_qfq": df_data.open_qfq,
                "high_qfq": df_data.high_qfq,
                "low_qfq": df_data.low_qfq,
                "close_qfq": df_data.close_qfq,
                "pre_close_qfq": df_data.pre_close_qfq,
                "open_hfq": df_data.open_hfq,
                "high_hfq": df_data.high_hfq,
                "low_hfq": df_data.low_hfq,
                "close_hfq": df_data.close_hfq,
                "pre_close_hfq": df_data.pre_close_hfq,            
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "pre_close": df_data.pre_close,
                "change": df_data.change,
                "pct_change": df_data.pct_change,
                "vol": df_data.vol,
                "amount": df_data.amount,
                "adj_factor": df_data.adj_factor,
                "open_qfq": df_data.open_qfq,
                "high_qfq": df_data.high_qfq,
                "low_qfq": df_data.low_qfq,
                "close_qfq": df_data.close_qfq,
                "pre_close_qfq": df_data.pre_close_qfq,
                "open_hfq": df_data.open_hfq,
                "high_hfq": df_data.high_hfq,
                "low_hfq": df_data.low_hfq,
                "close_hfq": df_data.close_hfq,
                "pre_close_hfq": df_data.pre_close_hfq,            
            }
        return data_dict
    
    def set_time_trade_minute_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockMinuteData):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "time_level": df_data.time_level,
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "vol": df_data.vol,
                "amount": df_data.amount,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_time),
                "time_level": df_data.freq,
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "vol": df_data.vol,
                "amount": df_data.amount,
            }
        return data_dict

    def set_time_trade_week_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockWeeklyData):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "pre_close": df_data.pre_close,
                "change": df_data.change,
                "pct_chg": df_data.pct_chg,
                "vol": df_data.vol,
                "amount": df_data.amount,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "pre_close": df_data.pre_close,
                "change": df_data.change,
                "pct_chg": df_data.pct_chg,
                "vol": df_data.vol,
                "amount": df_data.amount,
            }
        return data_dict
    
    def set_time_trade_month_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockMonthlyData):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "pre_close": df_data.pre_close,
                "change": df_data.change,
                "pct_chg": df_data.pct_chg,
                "vol": df_data.vol,
                "amount": df_data.amount,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "open": df_data.open,
                "high": df_data.high,
                "low": df_data.low,
                "close": df_data.close,
                "pre_close": df_data.pre_close,
                "change": df_data.change,
                "pct_chg": df_data.pct_chg,
                "vol": df_data.vol,
                "amount": df_data.amount,
            }
        return data_dict

    def set_stock_daily_basic_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockDailyBasic):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "close": df_data.close,
                "turnover_rate": df_data.turnover_rate,
                "turnover_rate_f": df_data.turnover_rate_f,
                "volume_ratio": df_data.volume_ratio,
                "pe": df_data.pe,
                "pe_ttm": df_data.pe_ttm,
                "pb": df_data.pb,
                "ps": df_data.ps,
                "ps_ttm": df_data.ps_ttm,
                "total_share": df_data.total_share,
                "float_share": df_data.float_share,
                "free_share": df_data.free_share,
                "total_mv": df_data.total_mv,
                "circ_mv": df_data.circ_mv,
                "limit_status": df_data.limit_status,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "close": df_data.close,
                "turnover_rate": df_data.turnover_rate,
                "turnover_rate_f": df_data.turnover_rate_f,
                "volume_ratio": df_data.volume_ratio,
                "pe": df_data.pe,
                "pe_ttm": df_data.pe_ttm,
                "pb": df_data.pb,
                "ps": df_data.ps,
                "ps_ttm": df_data.ps_ttm,
                "total_share": df_data.total_share,
                "float_share": df_data.float_share,
                "free_share": df_data.free_share,
                "total_mv": df_data.total_mv,
                "circ_mv": df_data.circ_mv,
                "limit_status": df_data.limit_status,
            }
        return data_dict

    def set_cyq_perf_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockCyqPerf):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "his_low": df_data.his_low,
                "his_high": df_data.his_high,
                "cost_5pct": df_data.cost_5pct,
                "cost_15pct": df_data.cost_15pct,
                "cost_50pct": df_data.cost_50pct,
                "cost_85pct": df_data.cost_85pct,
                "cost_95pct": df_data.cost_95pct,
                "weight_avg": df_data.weight_avg,
                "winner_rate": df_data.winner_rate,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "his_low": df_data.his_low,
                "his_high": df_data.his_high,
                "cost_5pct": df_data.cost_5pct,
                "cost_15pct": df_data.cost_15pct,
                "cost_50pct": df_data.cost_50pct,
                "cost_85pct": df_data.cost_85pct,
                "cost_95pct": df_data.cost_95pct,
                "weight_avg": df_data.weight_avg,
                "winner_rate": df_data.winner_rate,
            }
        return data_dict
    
    def set_cyq_chips_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockCyqChips):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "price": df_data.price,
                "percent": df_data.percent,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "price": df_data.price,
                "percent": df_data.percent,
            }
        return data_dict

class StockRealtimeDataFormatProcess(BaseDAO):
        # ================ 数据格式 ================
    def set_realtime_tick_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockRealtimeData):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "open_price": df_data.open_price,
                "prev_close_price": df_data.prev_close_price,
                "current_price": df_data.current_price,
                "high_price": df_data.high_price,
                "low_price": df_data.low_price,
                "volume": df_data.volume,
                "turnover_value": df_data.turnover_value,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.date + df_data.time),
                "open_price": df_data.open,
                "prev_close_price": df_data.pre_close,
                "current_price": df_data.price,
                "high_price": df_data.high,
                "low_price": df_data.low,
                "volume": df_data.volume,
                "turnover_value": df_data.amount,
            }
        return data_dict

    def set_level5_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, StockLevel5Data):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "buy_volume1": df_data.buy_volume1,
                "buy_price1": df_data.buy_price1,
                "buy_volume2": df_data.buy_volume2,
                "buy_price2": df_data.buy_price2,
                "buy_volume3": df_data.buy_volume3,
                "buy_price3": df_data.buy_price3,
                "buy_volume4": df_data.buy_volume4,
                "buy_price4": df_data.buy_price4,
                "buy_volume5": df_data.buy_volume5,
                "buy_price5": df_data.buy_price5,
                "sell_volume1": df_data.sell_volume1,
                "sell_price1": df_data.sell_price1,
                "sell_volume2": df_data.sell_volume2,
                "sell_price2": df_data.sell_price2,
                "sell_volume3": df_data.sell_volume3,
                "sell_price3": df_data.sell_price3,
                "sell_volume4": df_data.sell_volume4,
                "sell_price4": df_data.sell_price4,
                "sell_volume5": df_data.sell_volume5,
                "sell_price5": df_data.sell_price5,
                "order_diff": df_data.order_diff,
                "order_ratio": df_data.order_ratio,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.date + df_data.time),
                "buy_volume1": df_data.b1_v,
                "buy_price1": df_data.b1_p,
                "buy_volume2": df_data.b2_v,
                "buy_price2": df_data.b2_p,
                "buy_volume3": df_data.b3_v,
                "buy_price3": df_data.b3_p,
                "buy_volume4": df_data.b4_v,
                "buy_price4": df_data.b4_p,
                "buy_volume5": df_data.b5_v,
                "buy_price5": df_data.b5_p,
                "sell_volume1": df_data.s1_v,
                "sell_price1": df_data.s1_p,
                "sell_volume2": df_data.s2_v,
                "sell_price2": df_data.s2_p,
                "sell_volume3": df_data.s3_v,
                "sell_price3": df_data.s3_p,
                "sell_volume4": df_data.s4_v,
                "sell_price4": df_data.s4_p,
                "sell_volume5": df_data.s5_v,
                "sell_price5": df_data.s5_p,
                "order_diff": df_data.b1_v - df_data.s1_v,
                "order_ratio": (df_data.b1_v + df_data.b2_v + df_data.b3_v + df_data.b4_v + df_data.b5_v) / (df_data.s1_v + df_data.s2_v + df_data.s3_v + df_data.s4_v + df_data.s5_v),
            }
        return data_dict

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
        return data_dict

class FundFlowFormatProcess(BaseDAO):
    def set_fund_flow_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowDaily):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "buy_sm_vol": df_data.buy_sm_vol,
                "buy_sm_amount": df_data.buy_sm_amount,
                "sell_sm_vol": df_data.sell_sm_vol,
                "sell_sm_amount": df_data.sell_sm_amount,
                "buy_md_vol": df_data.buy_md_vol,
                "buy_md_amount": df_data.buy_md_amount,
                "sell_md_vol": df_data.sell_md_vol,
                "sell_md_amount": df_data.sell_md_amount,
                "buy_lg_vol": df_data.buy_lg_vol,
                "buy_lg_amount": df_data.buy_lg_amount,
                "sell_lg_vol": df_data.sell_lg_vol,
                "sell_lg_amount": df_data.sell_lg_amount,
                "buy_elg_vol": df_data.buy_elg_vol,
                "buy_elg_amount": df_data.buy_elg_amount,
                "sell_elg_vol": df_data.sell_elg_vol,
                "sell_elg_amount": df_data.sell_elg_amount,
                "net_mf_vol": df_data.net_mf_vol,
                "net_mf_amount": df_data.net_mf_amount,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "buy_sm_vol": df_data.buy_sm_vol,
                "buy_sm_amount": df_data.buy_sm_amount,
                "sell_sm_vol": df_data.sell_sm_vol,
                "sell_sm_amount": df_data.sell_sm_amount,
                "buy_md_vol": df_data.buy_md_vol,
                "buy_md_amount": df_data.buy_md_amount,
                "sell_md_vol": df_data.sell_md_vol,
                "sell_md_amount": df_data.sell_md_amount,
                "buy_lg_vol": df_data.buy_lg_vol,
                "buy_lg_amount": df_data.buy_lg_amount,
                "sell_lg_vol": df_data.sell_lg_vol,
                "sell_lg_amount": df_data.sell_lg_amount,
                "buy_elg_vol": df_data.buy_elg_vol,
                "buy_elg_amount": df_data.buy_elg_amount,
                "sell_elg_vol": df_data.sell_elg_vol,
                "sell_elg_amount": df_data.sell_elg_amount,
                "net_mf_vol": df_data.net_mf_vol,
                "net_mf_amount": df_data.net_mf_amount,
            }
        return data_dict

    def set_fund_flow_data_ths(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowCntTHS):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "net_amount": df_data.net_amount,
                "net_d5_amount": df_data.net_d5_amount,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "net_amount": df_data.net_amount,
                "net_d5_amount": df_data.net_d5_amount,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
            }
        return data_dict
    
    def set_fund_flow_data_dc(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowCntDC):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "content_type": df_data.content_type,
                "name": df_data.name,
                "pct_change": df_data.pct_change,
                "close": df_data.close,
                "net_amount": df_data.net_amount,
                "net_amount_rate": df_data.net_amount_rate,
                "buy_elg_amount": df_data.buy_elg_amount,
                "buy_elg_amount_rate": df_data.buy_elg_amount_rate,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "content_type": df_data.content_type,
                "name": df_data.name,
                "pct_change": df_data.pct_change,
                "close": df_data.close,
                "net_amount": df_data.net_amount,
                "net_amount_rate": df_data.net_amount_rate,
                "buy_elg_amount": df_data.buy_elg_amount,
                "buy_elg_amount_rate": df_data.buy_elg_amount_rate,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
            }
        return data_dict

    def set_fund_flow_cnt_ths_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowCntTHS):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "lead_stock": df_data.lead_stock,
                "pct_change": df_data.pct_change,
                "industry_index": df_data.industry_index,
                "company_num": df_data.company_num,
                "pct_change_stock": df_data.pct_change_stock,
                "net_buy_amount": df_data.net_buy_amount,
                "net_sell_amount": df_data.net_sell_amount,
                "net_amount": df_data.net_amount,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "lead_stock": df_data.lead_stock,
                "pct_change": df_data.pct_change,
                "industry_index": df_data.industry_index,
                "company_num": df_data.company_num,
                "pct_change_stock": df_data.pct_change_stock,
                "net_buy_amount": df_data.net_buy_amount,
                "net_sell_amount": df_data.net_sell_amount,
                "net_amount": df_data.net_amount,
            }
        return data_dict

    def set_fund_flow_cnt_dc_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowCntDC):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "content_type": df_data.content_type,
                "name": df_data.name,
                "pct_change": df_data.pct_change,
                "close": df_data.close_price,
                "net_amount": df_data.net_amount,
                "net_amount_rate": df_data.net_amount_rate,
                "buy_elg_amount": df_data.buy_elg_amount,
                "buy_elg_amount_rate": df_data.buy_elg_amount_rate,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
                "buy_sm_amount_stock": df_data.buy_sm_amount_stock,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "content_type": df_data.content_type,
                "name": df_data.name,
                "pct_change": df_data.pct_change,
                "close": df_data.close_price,
                "net_amount": df_data.net_amount,
                "net_amount_rate": df_data.net_amount_rate,
                "buy_elg_amount": df_data.buy_elg_amount,
                "buy_elg_amount_rate": df_data.buy_elg_amount_rate,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
                "buy_sm_amount_stock": df_data.buy_sm_amount_stock,
            }
        return data_dict

    def set_fund_flow_industry_ths_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowIndustryTHS):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "industry": df_data.industry,
                "lead_stock": df_data.lead_stock,
                "close": df_data.close,
                "pct_change": df_data.pct_change,
                "company_num": df_data.company_num,
                "pct_change_stock": df_data.pct_change_stock,
                "close_price": df_data.close_price,
                "net_buy_amount": df_data.net_buy_amount,
                "net_sell_amount": df_data.net_sell_amount,
                "net_amount": df_data.net_amount,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "industry": df_data.industry,
                "lead_stock": df_data.lead_stock,
                "close": df_data.close,
                "pct_change": df_data.pct_change,
                "company_num": df_data.company_num,
                "pct_change_stock": df_data.pct_change_stock,
                "close_price": df_data.close_price,
                "net_buy_amount": df_data.net_buy_amount,
                "net_sell_amount": df_data.net_sell_amount,
                "net_amount": df_data.net_amount,
            }
        return data_dict

    def set_fund_flow_market_dc_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, FundFlowMarketDc):
            data_dict = {
                "stock": stock,
                "trade_time": df_data.trade_time,
                "close_sh": df_data.close_sh,
                "pct_change_sh": df_data.pct_change_sh,
                "close_sz": df_data.close_sz,
                "pct_change_sz": df_data.pct_change_sz,
                "net_buy_amount": df_data.net_buy_amount,
                "net_buy_amount_rate": df_data.net_buy_amount_rate,
                "buy_elg_amount": df_data.buy_elg_amount,
                "buy_elg_amount_rate": df_data.buy_elg_amount_rate,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "close_sh": df_data.close_sh,
                "pct_change_sh": df_data.pct_change_sh,
                "close_sz": df_data.close_sz,
                "pct_change_sz": df_data.pct_change_sz,
                "net_buy_amount": df_data.net_buy_amount,
                "net_buy_amount_rate": df_data.net_buy_amount_rate,
                "buy_elg_amount": df_data.buy_elg_amount,
                "buy_elg_amount_rate": df_data.buy_elg_amount_rate,
                "buy_lg_amount": df_data.buy_lg_amount,
                "buy_lg_amount_rate": df_data.buy_lg_amount_rate,
                "buy_md_amount": df_data.buy_md_amount,
                "buy_md_amount_rate": df_data.buy_md_amount_rate,
                "buy_sm_amount": df_data.buy_sm_amount,
                "buy_sm_amount_rate": df_data.buy_sm_amount_rate,
            }
        return data_dict

    def set_lhb_daily_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, LhbDailyData):
            data_dict = {
                "stock": stock, # 股票代码
                "trade_time": df_data.trade_time, # 交易日期
                "name": df_data.name, # 股票名称
                "close": df_data.close, # 收盘价
                "pct_change": df_data.pct_change, # 涨跌幅
                "turnover_rate": df_data.turnover_rate, # 换手率
                "amount": df_data.amount, # 总成交额
                "l_sell": df_data.l_sell, # 龙虎榜卖出额
                "l_buy": df_data.l_buy, # 龙虎榜买入额
                "l_amount": df_data.l_amount, # 龙虎榜成交额
                "net_amount": df_data.net_amount, # 龙虎榜净买入额
                "net_rate": df_data.net_rate, # 龙虎榜净买入额占比
                "amount_rate": df_data.amount_rate, # 龙虎榜成交额占比
                "float_values": df_data.float_values, # 流通市值
                "reason": df_data.reason, # 上榜原因
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "name": df_data.name,
                "close": self._parse_number(df_data.close),
                "pct_change": self._parse_number(df_data.pct_change),
                "turnover_rate": self._parse_number(df_data.turnover_rate),
                "amount": self._parse_number(df_data.amount),
                "l_sell": self._parse_number(df_data.l_sell),
                "l_buy": self._parse_number(df_data.l_buy),
                "l_amount": self._parse_number(df_data.l_amount),
                "net_amount": self._parse_number(df_data.net_amount),
                "net_rate": self._parse_number(df_data.net_rate),
                "amount_rate": self._parse_number(df_data.amount_rate),
                "float_values": self._parse_number(df_data.float_values),
                "reason": df_data.reason,
            }
        return data_dict

    def set_lhb_inst_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, LhbInstData):
            data_dict = {
                "stock": stock, # 股票代码
                "trade_time": df_data.trade_time, # 交易日期
                "exalter": df_data.exalter, # 营业部
                "side": df_data.side, # 买卖买卖类型
                "buy": df_data.buy, # 买入额
                "buy_rate": df_data.buy_rate, # 买入占总成交比例
                "sell": df_data.sell, # 卖出额
                "sell_rate": df_data.sell_rate, # 卖出占总成交比例
                "net_buy": df_data.net_buy, # 净买入额
                "reason": df_data.reason, # 上榜原因
            }
        else:
            data_dict = {
                "stock": stock,
                "trade_time": self._parse_datetime(df_data.trade_date),
                "exalter": df_data.exalter, # 营业部
                "side": df_data.side, # 买卖买卖类型
                "buy": self._parse_number(df_data.buy), # 买入额
                "buy_rate": self._parse_number(df_data.buy_rate), # 买入占总成交比例
                "sell": self._parse_number(df_data.sell), # 卖出额
                "sell_rate": self._parse_number(df_data.sell_rate), # 卖出占总成交比例
                "net_buy": self._parse_number(df_data.net_buy), # 净买入额
                "reason": df_data.reason, # 上榜原因
            }
        return data_dict

class IndustryFormatProcess(BaseDAO):
    # 申万行业分类
    def set_sw_industry_data(self, index: IndexInfo, df_data: Any) -> Dict:
        if isinstance(df_data, SwIndustry):
            data_dict = {
                "index": index, # 指数代码
                "index_code": df_data.index_code, # 指数代码
                "industry_name": df_data.industry_name, # 行业名称
                "parent_code": df_data.parent_code, # 父级代码
                "level": df_data.level, # 行业分级
                "industry_code": df_data.industry_code, # 行业代码
                "is_publish": df_data.is_publish, # 是否发布指数
                "src": df_data.src, # 行业分类来源
            }
        else:
            data_dict = {
                "index": index, # 指数代码
                "index_code": df_data.index_code, # 指数代码
                "industry_name": df_data.industry_name, # 行业名称
                "parent_code": df_data.parent_code, # 父级代码
                "level": df_data.level, # 行业分级
                "industry_code": df_data.industry_code, # 行业代码
                "is_publish": df_data.is_publish, # 是否发布指数
                "src": df_data.src, # 行业分类来源
            }
        return data_dict

    # 申万行业成分
    def set_sw_industry_member_data(self, sw_industry: SwIndustry, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, SwIndustryMember):
            data_dict = {
                "l3_industry": sw_industry, # 三级行业
                "stock": stock, # 股票代码
                "l1_code": df_data.l1_code, # 一级行业代码
                "l1_name": df_data.l1_name, # 一级行业名称
                "l2_code": df_data.l2_code, # 二级行业代码
                "l2_name": df_data.l2_name, # 二级行业名称
                "l3_code": df_data.l3_code, # 三级行业代码
                "l3_name": df_data.l3_name, # 三级行业名称
                "name": df_data.name, # 成分股票名称
                "in_date": df_data.in_date, # 纳入日期
                "out_date": df_data.out_date, # 剔除日期
                "is_new": df_data.is_new, # 是否最新
            }
        else:
            data_dict = {
                "l3_industry": sw_industry, # 三级行业
                "stock": stock, # 股票代码
                "l1_code": df_data.l1_code, # 一级行业代码
                "l1_name": df_data.l1_name, # 一级行业名称
                "l2_code": df_data.l2_code, # 二级行业代码
                "l2_name": df_data.l2_name, # 二级行业名称
                "l3_code": df_data.l3_code, # 三级行业代码
                "l3_name": df_data.l3_name, # 三级行业名称
                "name": df_data.name, # 成分股票名称
                "in_date": df_data.in_date, # 纳入日期
                "out_date": df_data.out_date, # 剔除日期
                "is_new": df_data.is_new, # 是否最新
            }
        return data_dict

    # 申万行业日线行情
    def set_sw_industry_daily_data(self, sw_industry: SwIndustry, index: IndexInfo, df_data: Any) -> Dict:
        if isinstance(df_data, SwIndustryDaily):
            data_dict = {
                "industry": sw_industry, # 三级行业
                "index": index, # 指数代码
                "ts_code": df_data.ts_code, # 指数代码
                "trade_time": df_data.trade_time, # 交易日期
                "name": df_data.name, # 指数名称
                "open": df_data.open, # 开盘点位
                "high": df_data.high, # 最高点位
                "low": df_data.low, # 最低点位
                "close": df_data.close, # 收盘点位
                "change": df_data.change, # 涨跌点位
                "pct_change": df_data.pct_change, # 涨跌幅
                "vol": df_data.vol, # 成交量（万股）
                "amount": df_data.amount, # 成交额（万元）
                "pe": df_data.pe, # 市盈率
                "pb": df_data.pb, # 市净率
                "float_mv": df_data.float_mv, # 流通市值
                "total_mv": df_data.total_mv, # 总市值
            }
        else:
            data_dict = {
                "industry": sw_industry, # 三级行业
                "index": index, # 指数代码
                "ts_code": df_data.ts_code, # 指数代码
                "trade_time": self._parse_datetime(df_data.trade_time), # 交易日期
                "name": df_data.name, # 指数名称
                "open": self._parse_number(df_data.open), # 开盘点位
                "high": self._parse_number(df_data.high), # 最高点位
                "low": self._parse_number(df_data.low), # 最低点位
                "close": self._parse_number(df_data.close), # 收盘点位
                "change": self._parse_number(df_data.change), # 涨跌点位
                "pct_change": self._parse_number(df_data.pct_change), # 涨跌幅
                "vol": self._parse_number(df_data.vol), # 成交量（万股）
                "amount": self._parse_number(df_data.amount), # 成交额（万元）
                "pe": self._parse_number(df_data.pe), # 市盈率
                "pb": self._parse_number(df_data.pb), # 市净率
                "float_mv": self._parse_number(df_data.float_mv), # 流通市值
                "total_mv": self._parse_number(df_data.total_mv), # 总市值
            }
        return data_dict

    # 开盘啦题材库
    def set_kpl_concept_data(self, df_data: Any) -> Dict:
        if isinstance(df_data, KplConcept):
            data_dict = {
                "trade_time": df_data.trade_time, # 交易日期
                "ts_code": df_data.ts_code, # 题材代码
                "name": df_data.name, # 题材名称
                "z_t_num": df_data.z_t_num, # 涨停数
                "up_num": df_data.up_num, # 排名上升位数
            }
        else:
            data_dict = {
                "trade_time": df_data.trade_time, # 交易日期
                "ts_code": df_data.ts_code, # 题材代码
                "name": df_data.name, # 题材名称
                "z_t_num": df_data.z_t_num, # 涨停数
                "up_num": df_data.up_num, # 排名上升位数
            }
        return data_dict

    # 开盘啦题材成分股
    def set_kpl_concept_member_data(self, kpl_concept: KplConcept, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, KplConcept):
            data_dict = {
                "concept": kpl_concept, # 所属题材
                "stock": stock, # 股票代码
                "name": df_data.name, # 题材名称
                "con_name": df_data.con_name, # 成分股名称
                "trade_time": df_data.trade_time, # 交易日期
                "desc": df_data.desc, # 描述
                "hot_num": df_data.hot_num, # 人气值
            }
        else:
            data_dict = {
                "concept": kpl_concept, # 所属题材
                "stock": stock, # 股票代码
                "name": df_data.name, # 题材名称
                "con_name": df_data.con_name, # 成分股名称
                "trade_time": df_data.trade_time, # 交易日期
                "desc": df_data.desc, # 描述
                "hot_num": df_data.hot_num, # 人气值
            }
        return data_dict

    # 同花顺概念和行业指数
    def set_ths_index_data(self, df_data: Any) -> Dict:
        if isinstance(df_data, ThsIndex):
            data_dict = {
                "ts_code": df_data.ts_code, # 指数代码
                "name": df_data.name, # 指数名称
                "count": df_data.count, # 成分个数
                "exchange": df_data.exchange, # 交易所
                "list_date": df_data.list_date, # 上市日期
                "type": df_data.type, # 类型指数类型
            }
        else:
            data_dict = {
                "ts_code": df_data.ts_code, # 指数代码
                "name": df_data.name, # 指数名称
                "count": self._parse_number(df_data.count), # 成分个数
                "exchange": df_data.exchange, # 交易所
                "list_date": self._parse_datetime(df_data.list_date), # 上市日期
                "type": df_data.type, # 类型指数类型
            }
        return data_dict

    # 同花顺概念板块成分
    def set_ths_member_data(self, ths_index: ThsIndex, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, ThsMember):
            data_dict = {
                "ths_index": ths_index, # 所属概念
                "stock": stock, # 股票代码
                "weight": df_data.weight, # 权重
                "in_date": df_data.in_date, # 纳入日期
                "out_date": df_data.out_date, # 剔除日期
                "is_new": df_data.is_new, # 是否最新
            }
        else:
            data_dict = {
                "ths_index": ths_index, # 所属概念
                "stock": stock, # 股票代码
                "weight": df_data.weight, # 权重
                "in_date": self._parse_datetime(df_data.in_date), # 纳入日期
                "out_date": self._parse_datetime(df_data.out_date), # 剔除日期
                "is_new": df_data.is_new, # 是否最新
            }
        return data_dict

    # 东方财富概念板块
    def set_dc_index_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, DcIndex):
            data_dict = {
                "ts_code": df_data.ts_code, # 指数代码
                "trade_time": df_data.trade_time, # 交易日期
                "name": df_data.name, # 指数名称
                "leading_stock": df_data.leading_stock, # 领涨股
                "stock": stock,
                "pct_change": df_data.pct_change, # 涨跌幅
                "leading_pct": df_data.leading_pct, # 领涨股涨跌幅
                "total_mv": df_data.total_mv, # 总市值
                "turnover_rate": df_data.turnover_rate, # 换手率
                "up_num": df_data.up_num, # 排名上升位数
                "down_num": df_data.down_num, # 排名下降位数
            }
        else:
            data_dict = {
                "ts_code": df_data.ts_code, # 指数代码
                "trade_time": self._parse_datetime(df_data.trade_time), # 交易日期
                "name": df_data.name, # 指数名称
                "leading_stock": df_data.leading_stock, # 领涨股
                "stock": stock,
                "pct_change": self._parse_number(df_data.pct_change), # 涨跌幅
                "leading_pct": self._parse_number(df_data.leading_pct), # 领涨股涨跌幅
                "total_mv": self._parse_number(df_data.total_mv), # 总市值
                "turnover_rate": self._parse_number(df_data.turnover_rate), # 换手率
                "up_num": self._parse_number(df_data.up_num), # 排名上升位数
                "down_num": self._parse_number(df_data.down_num), # 排名下降位数
            }
        return data_dict

    # 东方财富板块成分
    def set_dc_member_data(self, dc_index: DcIndex, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, DcMember):
            data_dict = {
                "trade_time": df_data.trade_time, # 交易日期
                "dc_index": dc_index, # 所属概念
                "stock": stock, # 股票代码
                "name": df_data.name, # 成分股票名称
            }
        else:
            data_dict = {
                "trade_time": df_data.trade_time, # 交易日期
                "dc_index": dc_index, # 所属概念
                "stock": stock, # 股票代码
                "name": df_data.name, # 成分股票名称
            }
        return data_dict

class MarketFormatProcess(BaseDAO):
    # 市场交易统计(MarketDailyInfo)
    def set_market_daily_info_data(self, df_data: Any) -> Dict:
        if isinstance(df_data, MarketDailyInfo):
            data_dict = {
                "trade_date": df_data.trade_date, # 交易日期
                "ts_code": df_data.ts_code, # 市场代码
                "ts_name": df_data.ts_name, # 市场名称
                "com_count": df_data.com_count, # 挂牌数
                "total_share": df_data.total_share, # 总股本(亿股)
                "float_share": df_data.float_share, # 流通股本(亿股)
                "total_mv": df_data.total_mv, # 总市值(亿元)
                "float_mv": df_data.float_mv, # 流通市值(亿元)
                "amount": df_data.amount, # 成交金额(亿元)
                "vol": df_data.vol, # 成交量(亿股)
                "trans_count": df_data.trans_count, # 成交笔数(万笔)
                "pe": df_data.pe, # 市盈率
                "trans_rate": df_data.trans_rate, # 换手率(%)
                "exchange": df_data.exchange, # 交易所
            }
        else:
            data_dict = {
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "ts_code": df_data.ts_code, # 市场代码
                "ts_name": df_data.ts_name, # 市场名称
                "com_count": self._parse_number(df_data.com_count), # 挂牌数
                "total_share": self._parse_number(df_data.total_share), # 总股本(亿股)
                "float_share": self._parse_number(df_data.float_share), # 流通股本(亿股)
                "total_mv": self._parse_number(df_data.total_mv), # 总市值(亿元)
                "float_mv": self._parse_number(df_data.float_mv), # 流通市值(亿元)
                "amount": self._parse_number(df_data.amount), # 成交金额(亿元)
                "vol": self._parse_number(df_data.vol), # 成交量(亿股)
                "trans_count": self._parse_number(df_data.trans_count), # 成交笔数(万笔)
                "pe": self._parse_number(df_data.pe), # 市盈率
                "trans_rate": self._parse_number(df_data.trans_rate), # 换手率(%)
                "exchange": df_data.exchange, # 交易所
            }
        return data_dict

    # 游资名录
    def set_hm_list_data(self, df_data: Any) -> Dict:
        if isinstance(df_data, HmList):
            data_dict = {
                "name": df_data.name, # 游资名称
                "desc": df_data.desc, # 说明
                "orgs": df_data.orgs, # 关联机构
            }
        else:
            data_dict = {
                "name": df_data.name, # 游资名称
                "desc": df_data.desc, # 说明
                "orgs": df_data.orgs, # 关联机构
            }
        return data_dict

    # 游资每日明细
    def set_hm_detail_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, HmDetail):
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": df_data.trade_date, # 交易日期
                "ts_name": df_data.ts_name, # 股票名称
                "buy_amount": df_data.buy_amount, # 买入金额(元)
                "sell_amount": df_data.sell_amount, # 卖出金额(元)
                "net_amount": df_data.net_amount, # 净买卖(元)
                "hm_name": df_data.hm_name, # 游资名称
                "hm_orgs": df_data.hm_orgs, # 关联机构
                "tag": df_data.tag, # 标签
            }
        else:
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "ts_name": df_data.ts_name, # 股票名称
                "buy_amount": self._parse_number(df_data.buy_amount), # 买入金额(元)
                "sell_amount": self._parse_number(df_data.sell_amount), # 卖出金额(元)
                "net_amount": self._parse_number(df_data.net_amount), # 净买卖(元)
                "hm_name": df_data.hm_name, # 游资名称
                "hm_orgs": df_data.hm_orgs, # 关联机构
                "tag": df_data.tag, # 标签
            }
        return data_dict

    # 同花顺板块指数行情
    def set_ths_daily_data(self, ths_index: ThsIndex, df_data: Any) -> Dict:
        if isinstance(df_data, ThsDaily):
            data_dict = {
                "ths_index": ths_index, # 板块
                "trade_date": df_data.trade_date, # 交易日期
                "close": df_data.close, # 收盘价
                "open": df_data.open, # 开盘价
                "high": df_data.high, # 最高价
                "low": df_data.low, # 最低价
                "pre_close": df_data.pre_close, # 昨收价
                "avg_price": df_data.avg_price, # 平均价
                "change": df_data.change, # 涨跌额
                "pct_change": df_data.pct_change, # 涨跌幅
                "vol": df_data.vol, # 成交量
                "turnover_rate": df_data.turnover_rate, # 换手率
                "total_mv": df_data.total_mv, # 总市值
                "float_mv": df_data.float_mv, # 流通市值
            }
        else:
            data_dict = {
                "ths_index": ths_index, # 板块
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "close": self._parse_number(df_data.close), # 收盘价
                "open": self._parse_number(df_data.open), # 开盘价
                "high": self._parse_number(df_data.high), # 最高价
                "low": self._parse_number(df_data.low), # 最低价
                "pre_close": self._parse_number(df_data.pre_close), # 昨收价
                "avg_price": self._parse_number(df_data.avg_price), # 平均价
                "change": self._parse_number(df_data.change), # 涨跌额
                "pct_change": self._parse_number(df_data.pct_change), # 涨跌幅
                "vol": self._parse_number(df_data.vol), # 成交量
                "turnover_rate": self._parse_number(df_data.turnover_rate), # 换手率
                "total_mv": self._parse_number(df_data.total_mv), # 总市值
                "float_mv": self._parse_number(df_data.float_mv), # 流通市值
            }
        return data_dict

    # 涨跌停榜单 - 同花顺
    def set_limit_list_ths_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, LimitListThs):
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": df_data.trade_date, # 交易日期
                "name": df_data.name, # 股票名称
                "price": df_data.price, # 收盘价
                "pct_chg": df_data.pct_chg, # 涨跌幅%
                "open_num": df_data.open_num,
                "lu_desc": df_data.lu_desc, # 涨停原因
                "limit_type": df_data.limit_type, # 板单类别
                "tag": df_data.tag, # 涨停标签
                "status": df_data.status, # 涨停状态
                "first_lu_time": df_data.first_lu_time, # 首次涨停时间
                "last_lu_time": df_data.last_lu_time, # 最后涨停时间
                "first_ld_time": df_data.first_ld_time, # 首次跌停时间
                "last_ld_time": df_data.last_ld_time, # 最后跌停时间
                "limit_order": df_data.limit_order, # 封单量
                "limit_amount": df_data.limit_amount, # 封单额
                "turnover": df_data.turnover, # 成交额
                "rise_rate": df_data.rise_rate, # 涨速
                "sum_float": df_data.sum_float, # 总市值
                "market_type": df_data.market_type, # 股票类型
            }
        else:
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "name": df_data.name, # 股票名称
                "price": self._parse_number(df_data.price), # 收盘价
                "pct_chg": self._parse_number(df_data.pct_chg), # 涨跌幅%
                "open_num": self._parse_number(df_data.open_num), # 打开次数
                "lu_desc": df_data.lu_desc, # 涨停原因
                "limit_type": df_data.limit_type, # 板单类别
                "tag": df_data.tag, # 涨停标签
                "status": df_data.status, # 涨停状态
                "first_lu_time": self._parse_datetime(df_data.first_lu_time), # 首次涨停时间
                "last_lu_time": self._parse_datetime(df_data.last_lu_time), # 最后涨停时间
                "first_ld_time": self._parse_datetime(df_data.first_ld_time), # 首次跌停时间
                "last_ld_time": self._parse_datetime(df_data.last_ld_time), # 最后跌停时间
                "limit_order": self._parse_number(df_data.limit_order), # 封单量
                "limit_amount": self._parse_number(df_data.limit_amount), # 封单额
                "turnover": self._parse_number(df_data.turnover), # 成交额
                "rise_rate": self._parse_number(df_data.rise_rate), # 涨速
                "sum_float": self._parse_number(df_data.sum_float), # 总市值
                "market_type": df_data.market_type, # 股票类型
            }
        return data_dict

    # 涨跌停列表
    def set_limit_list_d_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, LimitListD):
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": df_data.trade_date, # 交易日期
                "industry": df_data.industry, # 所属行业
                "name": df_data.name, # 股票名称
                "close": df_data.close, # 收盘价
                "pct_chg": df_data.pct_chg, # 涨跌幅
                "amount": df_data.amount, # 成交额
                "limit_amount": df_data.limit_amount, # 板上成交金额
                "float_mv": df_data.float_mv, # 流通市值
                "total_mv": df_data.total_mv, # 总市值
                "turnover_ratio": df_data.turnover_ratio, # 换手率
                "fd_amount": df_data.fd_amount, # 封单金额
                "first_time": df_data.first_time, # 首次封板时间
                "last_time": df_data.last_time, # 最后封板时间
                "open_times": df_data.open_times, # 炸板次数
                "up_stat": df_data.up_stat, # 涨停统计
                "limit_times": df_data.limit_times, # 连板数
                "limit": df_data.limit, # 涨跌停类型
            }
        else:
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "industry": df_data.industry, # 所属行业
                "name": df_data.name, # 股票名称
                "close": self._parse_number(df_data.close), # 收盘价
                "pct_chg": self._parse_number(df_data.pct_chg), # 涨跌幅
                "amount": self._parse_number(df_data.amount), # 成交额
                "limit_amount": self._parse_number(df_data.limit_amount), # 板上成交金额
                "float_mv": self._parse_number(df_data.float_mv), # 流通市值
                "total_mv": self._parse_number(df_data.total_mv), # 总市值
                "turnover_ratio": self._parse_number(df_data.turnover_ratio), # 换手率
                "fd_amount": self._parse_number(df_data.fd_amount), # 封单金额
                "first_time": self._parse_datetime(df_data.first_time), # 首次封板时间
                "last_time": self._parse_datetime(df_data.last_time), # 最后封板时间
                "open_times": self._parse_datetime(df_data.open_times), # 炸板次数
                "up_stat": df_data.up_stat, # 涨停统计
                "limit_times": self._parse_number(df_data.limit_times), # 连板数
                "limit": df_data.limit, # 涨跌停类型
            }
        return data_dict

    # 连板天梯
    def set_limit_step_data(self, stock: StockInfo, df_data: Any) -> Dict:
        if isinstance(df_data, LimitStep):
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": df_data.trade_date, # 交易日期
                "name": df_data.name, # 股票名称
                "nums": df_data.nums, # 连板数
            }
        else:
            data_dict = {
                "stock": stock, # 股票代码
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "name": df_data.name, # 股票名称
                "nums": self._parse_number(df_data.nums), # 连板数
            }
        return data_dict

    # 最强板块统计 - 同花顺
    def set_limit_cpt_list_data(self, ths_index: ThsIndex, df_data: Any) -> Dict:
        if isinstance(df_data, LimitCptList):
            data_dict = {
                "ths_index": ths_index, # 板块名称
                "trade_date": df_data.trade_date, # 交易日期
                "name": df_data.name, # 板块名称
                "days": df_data.days, # 上榜天数
                "up_stat": df_data.up_stat, # 涨停统计
                "cons_nums": df_data.cons_nums, # 连板数
                "up_nums": df_data.up_nums, # 涨停数
                "pct_chg": df_data.pct_chg, # 涨跌幅
                "rank": df_data.rank, # 板块热点排名
            }
        else:
            data_dict = {
                "ths_index": ths_index, # 板块名称
                "trade_date": self._parse_datetime(df_data.trade_date), # 交易日期
                "name": df_data.name, # 板块名称
                "days": self._parse_number(df_data.days), # 上榜天数
                "up_stat": df_data.up_stat, # 涨停统计
                "cons_nums": self._parse_number(df_data.cons_nums), # 连板数
                "up_nums": df_data.up_nums, # 涨停数
                "pct_chg": self._parse_number(df_data.pct_chg), # 涨跌幅
                "rank": df_data.rank, # 板块热点排名
            }
        return data_dict


















