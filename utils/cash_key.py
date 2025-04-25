from typing import Dict, Optional, Union
from datetime import datetime
from utils import cache_constants as cc
from utils.cache_manager import CacheManager

# 缓存类型枚举
CACHE_TYPES = {
    'rt': 'realtime',     # 实时数据
    'st': 'static',       # 静态数据
    'ts': 'timeseries',   # 时间序列
    'calc': 'calculation', # 计算结果
    'user': 'user',       # 用户数据
    'strategy': 'strategy', # 策略数据
}

class CashKey:
    def generate_key(self, cache_type: str, entity_type: str, entity_id: str,  subtype: Optional[str] = None, params: Optional[Dict] = None,
                    date: Optional[Union[str, datetime]] = None) -> str:
        """
        生成标准化的缓存键
        """
        if cache_type not in CACHE_TYPES:
            raise ValueError(f"无效的缓存类型: {cache_type}")
        key_parts = [str(cache_type), str(entity_type), str(entity_id)]
        if subtype:
            key_parts.append(str(subtype))
        if params:
            param_parts = []
            for k, v in sorted(params.items()):
                if v is not None:
                    param_parts.append(f"{str(k)}:{str(v)}")
            if param_parts:
                key_parts.append(":".join(param_parts))
        if date:
            if isinstance(date, datetime):
                date_str = date.strftime('%Y%m%d')
            else:
                date_str = str(date)
            key_parts.append(date_str)
        return ':'.join(key_parts)

class IndexCashKey(CashKey):
    """
    指数缓存键生成器
    负责生成与指数相关的缓存键
    """

    # ================ Cash_key 缓存键设置 ================
    def indexs_data(self) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=cc.ID_ALL
        )
        return cache_key
    
    def index_data(self, index_code: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_BASIC_INFO
        )
        return cache_key
    
    def realtime_data(self, index_code: str) -> str:
        cache_key = self.generate_key(
                cache_type=cc.TYPE_REALTIME,    # 实时数据类型
                entity_type=cc.ENTITY_INDEX,    # 实体类型为指数
                entity_id=index_code,           # 实体ID为指数代码
                subtype=cc.SUBTYPE_QUOTE        # 子类型为报价/实时行情
            )
        return cache_key
    
    def latest_time_series(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_time_series(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_macd(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,      # 时间序列类型
            entity_type=cc.ENTITY_INDEX,        # 实体类型为指数
            entity_id=index_code,               # 实体ID为指数代码
            subtype=cc.SUBTYPE_MACD,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_macd(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_MACD,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_kdj(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KDJ,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_kdj(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_KDJ,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_ma(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_MA,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_ma(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_MA,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key
    
    def latest_boll(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_BOLL,
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key
    
    def history_boll(self, index_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_BOLL,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

class UserCashKey(CashKey):
    def user_favorites(self, user_id: int) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_USER,
            entity_type=cc.ENTITY_USERFAVORITES,
            entity_id=user_id
        )
        return cache_key
    
    def all_favorites(self) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_USER,
            entity_type=cc.ENTITY_USERFAVORITES
        )
        return cache_key

class StockCashKey(CashKey):
    # ================= 缓存cache_key设置 =================
    # 所有股票的cache_key
    def stocks_data(self) -> str:
        """
        所有股票的cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=cc.ID_ALL
        )
        return cache_key
    
    # 单个股票的cache_key
    def stock_data(self, stock_code: str) -> str:
        """
        单个股票的cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BASIC_INFO
        )
        return cache_key

    def stock_day_basic_info(self, stock_code: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_DAY_BASIC_INFO
        )
        return cache_key

    # 单个股票的最新分时成交数据cache_key
    def latest_time_trade(self, stock_code: str, time_level: str) -> str:
        """
        单个股票的最新分时成交数据cache_key
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            str: 单个股票的最新分时成交数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KLINE,        # 子类型为报价/实时行情
            params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
        )
        return cache_key

    # 单个股票的历史分时成交数据cache_key
    def history_time_trade(self, stock_code: str, time_level: str) -> str:
        """
        单个股票的历史分时成交数据cache_key
        Args:
            stock_code: 股票代码
            time_level: 时间级别
        Returns:
            str: 单个股票的历史分时成交数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

    # 单个股票的最新实时数据cache_key
    def latest_realtime_data(self, stock_code: str) -> str:
        """
        单个股票的最新实时数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的最新实时数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_QUOTE
        )
        return cache_key

    # 单个股票的历史实时数据cache_key
    def history_realtime_data(self, stock_code: str) -> str:
        """
        单个股票的历史实时数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的历史实时数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_QUOTE
        )
        return cache_key

    # 单个股票的最新买卖五档盘口数据cache_key
    def latest_level5_data(self, stock_code: str) -> str:
        """
        单个股票的最新买卖五档盘口数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的最新买卖五档盘口数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_LEVEL5
        )
        return cache_key

    # 单个股票的历史买卖五档盘口数据cache_key
    def history_level5_data(self, stock_code: str) -> str:
        """
        单个股票的历史买卖五档盘口数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的历史买卖五档盘口数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_LEVEL5
        )
        return cache_key

    # 单个股票的最新逐笔交易数据cache_key
    def latest_time_deal(self, stock_code: str) -> str:
        """
        单个股票的最新逐笔交易数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的最新逐笔交易数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_TIME_DEAL
        )
        return cache_key

    # 单个股票的历史逐笔交易数据cache_key
    def history_time_deal(self, stock_code: str) -> str:
        """
        单个股票的历史逐笔交易数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的历史逐笔交易数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_TIME_DEAL
        )
        return cache_key

    # 单个股票的最新分价成交占比数据cache_key
    def latest_real_percent(self, stock_code: str) -> str:
        """
        单个股票的最新分价成交占比数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的最新分价成交占比数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_REAL_PERCENT
        )
        return cache_key

    # 单个股票的历史分价成交占比数据cache_key
    def history_real_percent(self, stock_code: str) -> str:
        """
        单个股票的历史分价成交占比数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的历史分价成交占比数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_REAL_PERCENT
        )
        return cache_key

    # 单个股票的最新逐笔大单交易数据cache_key
    def latest_big_deal(self, stock_code: str) -> str:
        """
        单个股票的最新逐笔大单交易数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的最新逐笔大单交易数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BIG_DEAL
        )
        return cache_key

    # 单个股票的历史逐笔大单交易数据cache_key
    def history_big_deal(self, stock_code: str) -> str:
        """
        单个股票的历史逐笔大单交易数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的历史逐笔大单交易数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_BIG_DEAL
        )
        return cache_key

    # 单个股票的最新盘中异动数据
    def latest_abnormal_movement(self, stock_code: str) -> str:
        """
        单个股票的最新盘中异动数据cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的最新盘中异动数据cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_ABNORMAL_MOVEMENT
        )
        return cache_key

    # 单个股票的每日筹码分布
    def latest_cyq_chips(self, stock_code: str) -> str:
        """
        单个股票的每日筹码分布cache_key
        Args:
            stock_code: 股票代码
        Returns:
            str: 单个股票的每日筹码分布cache_key
        """
        cache_key = self.generate_key(
            cache_type=cc.TYPE_REALTIME,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_CYQ_CHIPS
        )
        return cache_key

class StrategyCashKey(CashKey):
    def __init__(self):
        self.cache_manager = None  # 修改: 改为 None，等待异步初始化

    async def initialize(self):
        self.cache_manager = await CacheManager()  # 异步初始化

    def macd_rsi_kdj_boll_data(self, stock_code: str, time_level: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STRATEGY,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_STRATEGY_MACD_RSI_KDJ_BOLL,
            params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key


