from django.utils import timezone
from typing import Any, Dict
import logging
from dao_manager.base_dao import BaseDAO
from stock_models.index import IndexInfo
from stock_models.stock_basic import StockInfo
from stock_models.time_trade import StockCyqChips, StockCyqPerf, StockDailyData, StockMinuteData, StockTimeTrade
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
    # ================ 数据格式 ================
    def set_index_data(self, api_data: Dict) -> Dict:
        # logger.info(f"api_data: {api_data}")
        if isinstance(api_data, IndexInfo):
            data_dict = {
                'code': api_data.code,  # 指数代码
                'name': api_data.name,  # 指数名称
                'exchange': api_data.exchange,  # 交易所代码
            }
        else:
            data_dict = {
                'code': api_data.get('dm', ''),  # 指数代码
                'name': api_data.get('mc', ''),  # 指数名称
                'exchange': api_data.get('jys', ''),  # 交易所代码
            }
        return data_dict
    
    def set_realtime_data(self, index: IndexInfo, api_data: Dict) -> Dict:
        data_dict = {
            'index': index,
            'open_price': self._parse_number(api_data.get('o')),  # 开盘价
            'high_price': self._parse_number(api_data.get('h')),  # 最高价
            'low_price': self._parse_number(api_data.get('l')),  # 最低价
            'current_price': self._parse_number(api_data.get('p')),  # 当前价格
            'prev_close_price': self._parse_number(api_data.get('yc')),  # 昨日收盘价
            'price_change': self._parse_number(api_data.get('ud')),  # 涨跌额
            'price_change_percent': self._parse_number(api_data.get('pc')),  # 涨跌幅
            'five_minute_change_percent': self._parse_number(api_data.get('fm')),  # 五分钟涨跌幅
            'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
            'change_speed': self._parse_number(api_data.get('zs')),  # 涨速
            'sixty_day_change_percent': self._parse_number(api_data.get('zdf60')),  # 60日涨跌幅
            'ytd_change_percent': self._parse_number(api_data.get('zdfnc')),  # 年初至今涨跌幅
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'turnover': self._parse_number(api_data.get('cje')),  # 成交额
            'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
            'volume_ratio': self._parse_number(api_data.get('lb')),  # 量比
            'pe_ratio': self._parse_number(api_data.get('pe')),  # 市盈率
            'pb_ratio': self._parse_number(api_data.get('sjl')),  # 市净率
            'circulating_market_value': self._parse_number(api_data.get('lt')),  # 流通市值
            'total_market_value': self._parse_number(api_data.get('sz')),  # 总市值
            'trade_time': self._parse_datetime(api_data.get('t')),  # 更新时间
        }
        return data_dict

    def set_time_series(self, index: IndexInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'index': index.id,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('d')),  # 交易时间
            'open_price': self._parse_number(api_data.get('o')),  # 开盘价
            'high_price': self._parse_number(api_data.get('h')),  # 最高价
            'low_price': self._parse_number(api_data.get('l')),  # 最低价
            'close_price': self._parse_number(api_data.get('c')),  # 收盘价
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'turnover': self._parse_number(api_data.get('e')),  # 成交额
            'amplitude': self._parse_number(api_data.get('zf')),  # 振幅
            'turnover_rate': self._parse_number(api_data.get('hs')),  # 换手率
            'change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
            'change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额
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
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(df_data.trade_date),
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
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(df_data.trade_date),
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
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(df_data.trade_date),
            "price": df_data.price,
            "percent": df_data.percent,
        }
        return data_dict

class StockRealtimeDataFormatProcess(BaseDAO):
        # ================ 数据格式 ================
    def set_realtime_tick_data(self, stock: StockInfo, time_level: str, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "time_level": time_level,
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
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
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
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
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
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
            "net_amount": df_data.net_amount,
            "net_amount_rate": df_data.net_amount_rate,
            "net_d5_amount": df_data.net_d5_amount,
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
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
            "lead_stock": df_data.lead_stock,
            "pct_change": df_data.pct_change,
            "index_close": df_data.industry_index,
            "company_num": df_data.company_num,
            "pct_change_stock": df_data.pct_change_stock,
            "net_buy_amount": df_data.net_buy_amount,
            "net_sell_amount": df_data.net_sell_amount,
            "net_amount": df_data.net_amount,
        }
        return data_dict

    def set_fund_flow_cnt_dc_data(self, stock: StockInfo, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
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
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
            "industry_name": df_data.industry,
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
        data_dict = {
            "stock": stock,
            "trade_date": df_data.trade_date,
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


















