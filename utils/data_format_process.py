
from typing import Dict, List, Optional
from dao_manager.base_dao import BaseDAO
from stock_models.index import IndexInfo
from stock_models.stock_basic import StockInfo
import logging

logger = logging.getLogger(__name__)

class IndexDataFormatProcess(BaseDAO):

    # ================ 数据格式 ================
    async def set_index_data(self, api_data: Dict) -> Optional[List[Dict]]:
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
    
    async def set_realtime_data(self, index: IndexInfo, api_data: Dict) -> Optional[List[Dict]]:
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
            'update_time': self._parse_datetime(api_data.get('t')),  # 更新时间
        }
        return data_dict

    async def set_time_series(self, index: IndexInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'index': index,
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

    async def set_kdj_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'index': index,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'k_value': self._parse_number(api_data.get('k')),  # K值
            'd_value': self._parse_number(api_data.get('d')),  # D值
            'j_value': self._parse_number(api_data.get('j')),  # J值
        }
        return data_dict
    
    async def set_macd_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    
    async def set_ma_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    
    async def set_boll_data(self, index: IndexInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    async def set_stock_info_data(self, api_data: Dict) -> Optional[List[Dict]]:
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
    async def set_time_trade_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    
    async def set_kdj_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'time_level': time_level,
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'k_value': self._parse_number(api_data.get('k')),  # K值
            'd_value': self._parse_number(api_data.get('d')),  # D值
            'j_value': self._parse_number(api_data.get('j')),  # J值
        }
        return data_dict
    
    async def set_macd_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    
    async def set_ma_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    
    async def set_boll_data(self, stock: StockInfo, time_level: str, api_data: Dict) -> Optional[List[Dict]]:
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
    async def set_realtime_data(self, stock: StockInfo, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'update_time': self._parse_datetime(api_data.get('t')),
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
        return data_dict
    
    async def set_level5_data(self, stock: StockInfo, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'update_time': self._parse_datetime(api_data.get('t')),  # 交易时间
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
    
    async def set_onebyone_trade_data(self, stock: StockInfo, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'trade_date': self._parse_datetime(api_data.get('d')),  # 交易时间
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'price': self._parse_number(api_data.get('p')),  # 成交价
            'trade_direction': self._parse_number(api_data.get('ts')),  # 交易方向
        }
        return data_dict
    
    async def set_time_deal_data(self, stock: StockInfo, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'trade_date': self._parse_datetime(api_data.get('d')),  # 交易时间
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'price': self._parse_number(api_data.get('p')),  # 成交价
        }
        return data_dict
    
    async def set_real_percent_data(self, stock: StockInfo, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'trade_date': self._parse_datetime(api_data.get('d')),  # 交易时间
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'price': self._parse_number(api_data.get('p')),  # 成交价
            'percentage': self._parse_number(api_data.get('b')),  # 占比
        }
        return data_dict
    
    async def set_big_deal_data(self, stock: StockInfo, api_data: Dict) -> Optional[List[Dict]]:
        data_dict = {
            'stock': stock,
            'trade_date': self._parse_datetime(api_data.get('d')),  # 交易时间
            'trade_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'volume': self._parse_number(api_data.get('v')),  # 成交量
            'price': self._parse_number(api_data.get('p')),  # 成交价
            'trade_direction': self._parse_number(api_data.get('ts')),  # 交易方向
        }
        return data_dict
    
    async def set_abnormal_movement_data(self, api_data: Dict) -> Optional[List[Dict]]:
        stock = await self.stock_basic_dao.get_stock_by_code(api_data.get('dm'))
        data_dict = {
            'stock': stock,
            'movement_time': self._parse_datetime(api_data.get('t')),  # 交易时间
            'movement_type': api_data.get('type'),  # 异动类型
            'movement_info': api_data.get('xx'),  # 相关信息
        }
        return data_dict
    
