from typing import Dict, Any, List, Optional
from utils import cache_constants as cc
from utils.cache_manager import CacheManager

class IndexCashKey:
    """
    指数缓存键生成器
    负责生成与指数相关的缓存键
    """
    
    def __init__(self):
        """初始化缓存键生成器"""
        self.cache_manager = CacheManager()

    # ================ Cash_key 缓存键设置 ================
    def indexs_data(self) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=cc.ID_ALL
        )
        return cache_key
    
    def index_data(self, index_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_BASIC_INFO
        )
        return cache_key
    
    def realtime_data(self, index_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
                cache_type=cc.TYPE_REALTIME,    # 实时数据类型
                entity_type=cc.ENTITY_INDEX,    # 实体类型为指数
                entity_id=index_code,           # 实体ID为指数代码
                subtype=cc.SUBTYPE_QUOTE        # 子类型为报价/实时行情
            )
        return cache_key
    
    def latest_time_series(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_time_series(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_macd(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,      # 时间序列类型
            entity_type=cc.ENTITY_INDEX,        # 实体类型为指数
            entity_id=index_code,               # 实体ID为指数代码
            subtype=cc.SUBTYPE_MACD,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_macd(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_MACD,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_kdj(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KDJ,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_kdj(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KDJ,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_ma(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_MA,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_ma(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_MA,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_boll(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_BOLL,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_boll(self, index_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_BOLL,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

class UserCashKey:
    def __init__(self):
        self.cache_manager = CacheManager()

    def user_favorites(self, user_id: int) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_USER,
            entity_type=cc.ENTITY_USERFAVORITES,
            entity_id=user_id
        )
        return cache_key
    

class StockCashKey:
    def __init__(self):
        self.cache_manager = CacheManager()

    # ================= 缓存cache_key设置 =================
    def stocks_data(self) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=cc.ID_ALL
        )
        return cache_key
    
    def stock_data(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BASIC_INFO
        )
        return cache_key
    
    def latest_time_trade(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KLINE,        # 子类型为报价/实时行情
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key

    def latest_kdj(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KDJ,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key

    def latest_macd(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_MACD,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key

    def latest_ma(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_MA,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key

    def latest_boll(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BOLL,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key

    def history_time_trade(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

    def history_kdj(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KDJ,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

    def history_macd(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_MACD,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

    def history_ma(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_MA,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

    def history_boll(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BOLL,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

    def latest_realtime_data(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_QUOTE
        )
        return cache_key

    def history_realtime_data(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_QUOTE
        )
        return cache_key

    def latest_level5_data(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_LEVEL5
        )
        return cache_key
    
    def history_level5_data(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_LEVEL5
        )
        return cache_key
    
    def latest_time_deal(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_TIME_DEAL
        )
        return cache_key
    
    def history_time_deal(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_TIME_DEAL
        )
        return cache_key
    
    def latest_real_percent(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_REAL_PERCENT
        )
        return cache_key
    
    def history_real_percent(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_REAL_PERCENT
        )
        return cache_key
    
    def latest_big_deal(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BIG_DEAL
        )
        return cache_key
    
    def history_big_deal(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BIG_DEAL
        )
        return cache_key
    
    def latest_abnormal_movement(self, stock_code: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_ABNORMAL_MOVEMENT
        )
        return cache_key


class StrategyCashKey:
    def __init__(self):
        self.cache_manager = CacheManager()

    def macd_rsi_kdj_boll_data(self, stock_code: str, time_level: str) -> str:
        cache_key = self.cache_manager.generate_key(
            cache_type=cc.TYPE_STRATEGY,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_STRATEGY_MACD_RSI_KDJ_BOLL,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key


