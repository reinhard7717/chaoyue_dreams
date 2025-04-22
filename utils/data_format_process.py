from django.utils import timezone
from typing import Any, Dict, List, Optional

from dao_manager.base_dao import BaseDAO
from stock_models.index import IndexInfo
from stock_models.stock_basic import StockInfo, StockTimeTrade
from users.models import FavoriteStock
import logging

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

    def set_kdj_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'index': index,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'k_value': self._parse_number(api_data.get('k')),  # K值
            'd_value': self._parse_number(api_data.get('d')),  # D值
            'j_value': self._parse_number(api_data.get('j')),  # J值
        }
        return data_dict
    
    def set_macd_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'index': index,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'diff': self._parse_number(api_data.get('diff')),  # DIFF值
            'dea': self._parse_number(api_data.get('dea')),    # DEA值
            'macd': self._parse_number(api_data.get('macd')),  # MACD值
            'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
            'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
        }
        return data_dict
    
    def set_ma_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'index': index,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'ma3': self._parse_number(api_data.get('ma3')),  # K值
            'ma5': self._parse_number(api_data.get('ma5')),  # D值
            'ma10': self._parse_number(api_data.get('ma10')),  # J值
            'ma15': self._parse_number(api_data.get('ma15')),  # J值
            'ma20': self._parse_number(api_data.get('ma20')),  # J值
            'ma30': self._parse_number(api_data.get('ma30')),  # J值
            'ma60': self._parse_number(api_data.get('ma60')),  # J值
            'ma120': self._parse_number(api_data.get('ma120')),  # J值
            'ma200': self._parse_number(api_data.get('ma200')),  # J值
            'ma250': self._parse_number(api_data.get('ma250')),  # J值
        }
        return data_dict
    
    def set_boll_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'index': index,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'upper': self._parse_number(api_data.get('u')),
            'middle': self._parse_number(api_data.get('m')),
            'lower': self._parse_number(api_data.get('d')),
        }
        return data_dict

class StockInfoFormatProcess(BaseDAO):
    def set_stock_info_data(self, api_data: Dict) -> Dict:
        if isinstance(api_data, StockInfo):
            data_dict = {
                'stock_code': api_data.stock_code,  # 股票代码
                'stock_name': api_data.stock_name,  # 股票名称
                'exchange': api_data.exchange,  # 交易所代码
            }
        else:
            data_dict = {
                'stock_code': api_data.get('dm', ''),  # 股票代码
                'stock_name': api_data.get('mc', ''),  # 股票名称
                'exchange': api_data.get('jys', ''),  # 交易所代码
            }
        return data_dict

class StockIndicatorsDataFormatProcess(BaseDAO):
        # ================= 数据 =================
    def set_time_trade_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Dict:
        if isinstance(api_data, StockTimeTrade):
            # 确保时区转换正确
            trade_time = api_data.trade_time
            if trade_time.tzinfo is not None:
                trade_time = trade_time.astimezone(timezone.get_current_timezone())
            data_dict = {
                'stock': stock,
                'time_level': time_level,
                'trade_time': trade_time,  # 交易时间
                'open_price': self._parse_number(api_data.open_price),  # 开盘价
                'high_price': self._parse_number(api_data.high_price),  # 最高价
                'low_price': self._parse_number(api_data.low_price),  # 最低价
                'close_price': self._parse_number(api_data.close_price),  # 收盘价
                'volume': self._parse_number(api_data.volume),  # 成交量
                'turnover': self._parse_number(api_data.turnover),  # 成交额
                'amplitude': self._parse_number(api_data.amplitude),  # 振幅
                'turnover_rate': self._parse_number(api_data.turnover_rate),  # 换手率
                'price_change_percent': self._parse_number(api_data.price_change_percent),  # 涨跌幅
                'price_change_amount': self._parse_number(api_data.price_change_amount),  # 涨跌额   
            }
        else:
            data_dict = {
                'stock': stock,
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
                'price_change_percent': self._parse_number(api_data.get('zd')),  # 涨跌幅
                'price_change_amount': self._parse_number(api_data.get('zde')),  # 涨跌额   
            }
        return data_dict
    
    def set_kdj_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'k_value': self._parse_number(api_data.get('k')),  # K值
            'd_value': self._parse_number(api_data.get('d')),  # D值
            'j_value': self._parse_number(api_data.get('j')),  # J值
        }
        return data_dict
    
    def set_macd_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'diff': self._parse_number(api_data.get('diff')),  # DIFF值
            'dea': self._parse_number(api_data.get('dea')),  # DEA值
            'macd': self._parse_number(api_data.get('macd')),  # MACD值
            'ema12': self._parse_number(api_data.get('ema12')),  # EMA(12)值
            'ema26': self._parse_number(api_data.get('ema26')),  # EMA(26)值
        }
        return data_dict
    
    def set_ma_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'ma3': self._parse_number(api_data.get('ma3')),  # MA3值
            'ma5': self._parse_number(api_data.get('ma5')),  # MA5值
            'ma10': self._parse_number(api_data.get('ma10')),  # MA10值
            'ma15': self._parse_number(api_data.get('ma15')),  # MA15值
            'ma20': self._parse_number(api_data.get('ma20')),  # MA20值
            'ma30': self._parse_number(api_data.get('ma30')),  # MA30值 
            'ma60': self._parse_number(api_data.get('ma60')),  # MA60值
            'ma120': self._parse_number(api_data.get('ma120')),  # MA120值
            'ma200': self._parse_number(api_data.get('ma200')),  # MA200值
            'ma250': self._parse_number(api_data.get('ma250')),  # MA250值
        }
        return data_dict
    
    def set_boll_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'upper': self._parse_number(api_data.get('u')),  # 上轨
            'lower': self._parse_number(api_data.get('d')),  # 下轨
            'mid': self._parse_number(api_data.get('m')),  # 中轨
        }
        return data_dict

class StockRealtimeDataFormatProcess(BaseDAO):
        # ================ 数据格式 ================
    def set_realtime_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        from stock_models.stock_realtime import StockRealtimeData
        if isinstance(api_data, StockRealtimeData):
            trade_time = api_data.trade_time
            if trade_time.tzinfo is not None:
                trade_time = trade_time.astimezone(timezone.get_current_timezone())
            data_dict = {
                'stock': stock,
                'trade_time': trade_time,
                'open_price': api_data.open_price,
                'five_min_change': api_data.five_min_change,
                'high_price': api_data.high_price,
                'turnover_rate': api_data.turnover_rate,
                'volume_ratio': api_data.volume_ratio,
                'low_price': api_data.low_price,
                'tradable_market_value': api_data.tradable_market_value,
                'pe_ratio': api_data.pe_ratio,
                'price_change_percent': api_data.price_change_percent,
                'current_price': api_data.current_price,
                'total_market_value': api_data.total_market_value,
                'turnover_value': api_data.turnover_value,
                'price_change': api_data.price_change,
                'volume': api_data.volume,
                'prev_close_price': api_data.prev_close_price,
                'amplitude': api_data.amplitude,
                'increase_speed': api_data.increase_speed,
                'pb_ratio': api_data.pb_ratio,
                'price_change_60d': api_data.price_change_60d,
                'price_change_ytd': api_data.price_change_ytd,
            }
        else:
            trade_time = self._parse_datetime(api_data.get('t'))
            data_dict = {
                'stock': stock,
                'trade_time': trade_time,
                'open_price': self._parse_number(api_data.get('o')),
                'five_min_change': self._parse_number(api_data.get('fm')),
                'high_price': self._parse_number(api_data.get('h')),
                'turnover_rate': self._parse_number(api_data.get('hs')),
                'volume_ratio': self._parse_number(api_data.get('lb')),
                'low_price': self._parse_number(api_data.get('l')),
                'tradable_market_value': self._parse_number(api_data.get('lt')),
                'pe_ratio': self._parse_number(api_data.get('pe')),
                'price_change_percent': self._parse_number(api_data.get('pc')),
                'current_price': self._parse_number(api_data.get('p')),
                'total_market_value': self._parse_number(api_data.get('sz')),
                'turnover_value': self._parse_number(api_data.get('cje')),
                'price_change': self._parse_number(api_data.get('ud')),
                'volume': self._parse_number(api_data.get('v')),
                'prev_close_price': self._parse_number(api_data.get('yc')),
                'amplitude': self._parse_number(api_data.get('zf')),
                'increase_speed': self._parse_number(api_data.get('zs')),
                'pb_ratio': self._parse_number(api_data.get('sjl')),
                'price_change_60d': self._parse_number(api_data.get('zdf60')),
                'price_change_ytd': self._parse_number(api_data.get('zdfnc')),
            }
        data_dict['trade_time'] = trade_time
        return data_dict
    
    def set_level5_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'order_diff': self._parse_number(api_data.get('vc')),
            'order_ratio': self._parse_number(api_data.get('vb')),
            'buy_price1': self._parse_number(api_data.get('pb1')),
            'buy_volume1': self._parse_number(api_data.get('vb1')),
            'buy_price2': self._parse_number(api_data.get('pb2')),
            'buy_volume2': self._parse_number(api_data.get('vb2')),
            'buy_price3': self._parse_number(api_data.get('pb3')),
            'buy_volume3': self._parse_number(api_data.get('vb3')),
            'buy_price4': self._parse_number(api_data.get('pb4')),
            'buy_volume4': self._parse_number(api_data.get('vb4')),
            'buy_price5': self._parse_number(api_data.get('pb5')),
            'buy_volume5': self._parse_number(api_data.get('vb5')),
            'sell_price1': self._parse_number(api_data.get('ps1')),
            'sell_volume1': self._parse_number(api_data.get('vs1')),
            'sell_price2': self._parse_number(api_data.get('ps2')),
            'sell_volume2': self._parse_number(api_data.get('vs2')),
            'sell_price3': self._parse_number(api_data.get('ps3')),
            'sell_volume3': self._parse_number(api_data.get('vs3')),
            'sell_price4': self._parse_number(api_data.get('ps4')),
            'sell_volume4': self._parse_number(api_data.get('vs4')),
            'sell_price5': self._parse_number(api_data.get('ps5')),
            'sell_volume5': self._parse_number(api_data.get('vs5')),
        }
        return data_dict
    
    def set_onebyone_trade_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'trade_date': self._parse_datetime(api_data.get('d')),  # 交易时间
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'price': self._parse_number(api_data.get('p')),  # 成交价
            'trade_direction': self._parse_number(api_data.get('ts')),  # 交易方向
        }
        return data_dict
    
    def set_time_deal_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        from stock_models.stock_realtime import StockTimeDeal
        if isinstance(api_data, StockTimeDeal):
            trade_time = api_data.trade_time
            trade_date = api_data.trade_date
            data_dict = {
                'stock': stock,
                'trade_date': trade_date,
                'trade_time': trade_time,
                'volume': api_data.volume,
                'price': api_data.price,
            }
        else:
            trade_time = self._parse_datetime(api_data.get('t'))
            trade_date = self._parse_datetime(api_data.get('d'))
            data_dict = {
                'stock': stock,
                'trade_date': trade_date,
                'trade_time': trade_time,
                'volume': self._parse_number(api_data.get('v')),  # 成交量
                'price': self._parse_number(api_data.get('p')),  # 成交价
            }
        if trade_time.tzinfo is not None:
            trade_time = trade_time.astimezone(timezone.get_current_timezone())
        if trade_date.tzinfo is not None:
            trade_date = trade_date.astimezone(timezone.get_current_timezone())
        data_dict['trade_time'] = trade_time
        data_dict['trade_date'] = trade_date
        return data_dict
    
    def set_real_percent_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        from stock_models.stock_realtime import StockPricePercent
        if isinstance(api_data, StockPricePercent):
            trade_date = api_data.trade_date
            data_dict = {
                'stock': stock,
                'trade_date': trade_date,
                'price': api_data.price,
                'volume': api_data.volume,
                'percentage': api_data.percentage,
            }
        else:
            trade_date = self._parse_datetime(api_data.get('d'))
            data_dict = {
                'stock': stock,
                'trade_date': trade_date,
                'volume': self._parse_number(api_data.get('v')),  # 成交量
                'price': self._parse_number(api_data.get('p')),  # 成交价
                'percentage': self._parse_number(api_data.get('b')),  # 占比
            }
        if trade_date.tzinfo is not None:
            trade_date = trade_date.astimezone(timezone.get_current_timezone())
        data_dict['trade_date'] = trade_date
        return data_dict
    
    def set_big_deal_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        from stock_models.stock_realtime import StockBigDeal
        if isinstance(api_data, StockBigDeal):
            trade_time = api_data.trade_time
            trade_date = api_data.trade_date
            data_dict = {
                'stock': stock,
                'trade_date': trade_date,
                'trade_time': trade_time,
                'volume': api_data.volume,
                'price': api_data.price,
            }
        else:
            trade_time = self._parse_datetime(api_data.get('t'))
            trade_date = self._parse_datetime(api_data.get('d'))
            data_dict = {
                'stock': stock,
                'trade_date': trade_date,
                'trade_time': trade_time,
                'volume': self._parse_number(api_data.get('v')),  # 成交量
                'price': self._parse_number(api_data.get('p')),  # 成交价
                'trade_direction': self._parse_number(api_data.get('ts')),  # 交易方向
            }
        if trade_time.tzinfo is not None:
            trade_time = trade_time.astimezone(timezone.get_current_timezone())
        if trade_date.tzinfo is not None:
            trade_date = trade_date.astimezone(timezone.get_current_timezone())
        data_dict['trade_time'] = trade_time
        data_dict['trade_date'] = trade_date
        return data_dict
    
    def set_abnormal_movement_data(self, stock: StockInfo, api_data: Dict) -> Dict:
        data_dict = {
            'stock': stock,
            'movement_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'movement_type': api_data.get('type'),  # 异动类型
            'movement_info': api_data.get('xx'),  # 相关信息
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

class StockInfoFormatTuShare(BaseDAO):
    def set_stock_info_data(self, df_data: Any) -> Dict:
        data_dict = {
            "stock_code": df_data.ts_code,
            "stock_name": df_data.name,
            "area": df_data.area,
            "industry": df_data.industry,
            "full_name": df_data.fullname,
            "en_name": df_data.en_name,
            "cn_spell": df_data.cnspell,
            "market_type": df_data.market,
            "exchange": df_data.exchange,
            "currency_type": df_data.curr_type,
            "list_status": df_data.list_status,
            "list_date": df_data.list_date,
            "delist_date": df_data.delist_date,
            "is_hs": df_data.is_hs,
            "actual_controller": df_data.act_name,
            "actual_controller_type": df_data.act_ent_type,
        }
        return data_dict

class StockRealtimeDataFormatTuShare(BaseDAO):
    def set_realtime_data(self, stock: StockInfo, df_data: Any) -> Dict:
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

class StockTimeTradeFormatTuShare(BaseDAO):
    def set_time_trade_data(self, stock: StockInfo, time_level: str, df_data: Any) -> Dict:
        data_dict = {
            "stock": stock,
            "trade_date": self._parse_datetime(df_data.date),
            "time_level": time_level,
            "open_price": df_data.open,
            "high_price": df_data.high,
            "low_price": df_data.low,
            "close_price": df_data.close,
            "volume": df_data.vol,
            "price_change_amount": df_data.change,
            "price_change_percent": df_data.pct_chg,
            "turnover": df_data.amount,
        }
        return data_dict

