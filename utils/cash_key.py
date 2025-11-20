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
    'app': 'app', # 应用数据
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
    def IndexWeight(self, index_code: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_INDEX,
            entity_id=index_code,
            subtype=cc.SUBTYPE_CONCEPTS
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
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_KLINE,
            params={cc.PARAM_PERIOD: "day_basic_info"}
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
            subtype=time_level,        # 子类型为报价/实时行情
            # params={cc.PARAM_PERIOD: time_level, 'tag': 'latest'}
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
    # 【盘中引擎专用】生成单个股票指定日期的分钟K线ZSET缓存键
    def intraday_minute_kline(self, stock_code: str, time_level: str, date_str: str) -> str:
        """
        【盘中引擎专用】生成单个股票指定日期的分钟K线ZSET缓存键。
        Args:
            stock_code (str): 股票代码
            time_level (str): 分钟级别 (e.g., '1', '5')
            date_str (str): 日期字符串, 格式 'YYYYMMDD'
        Returns:
            str: 用于存储分钟K线ZSET的缓存键
        """
        # 使用 'ts' (timeseries) 类型
        # 使用 'kline_minute' 作为子类型
        # 将 time_level 和 date 作为参数
        cache_key = self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=f"kline_{time_level}min",
            date=date_str
        )
        return cache_key
    # 【盘中引擎专用】生成单个股票指定日期的实时行情Tick ZSET缓存键
    def intraday_ticks_realtime(self, stock_code: str, date_str: str) -> str:
        """
        【盘中引擎专用】生成单个股票指定日期的实时行情Tick ZSET缓存键。
        """
        return self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype='ticks_realtime',
            date=date_str
        )
    # 【盘中引擎专用】生成单个股票指定日期的五档盘口Tick ZSET缓存键
    def intraday_ticks_level5(self, stock_code: str, date_str: str) -> str:
        """
        【盘中引擎专用】生成单个股票指定日期的五档盘口Tick ZSET缓存键。
        """
        return self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype='ticks_level5',
            date=date_str
        )
    # 【盘中引擎专用】生成当日监控股票池的缓存键
    def intraday_monitoring_pool(self) -> str:
        """
        【盘中引擎专用】生成用于存储当日监控股票池的缓存键。
        这是一个全局键，不针对任何特定股票。
        """
        # 使用 'app' (application) 类型，表示这是一个应用级别的、非特定实体的数据
        # 使用 'pool' (股票池) 作为实体类型
        # 使用 'intraday_monitoring' 作为ID，明确其用途
        cache_key = self.generate_key(
            cache_type=cc.TYPE_APP,
            entity_type=cc.ENTITY_POOL,
            entity_id=cc.ID_INTRADAY_MONITORING
        )
        return cache_key
    def intraday_real_ticks(self, stock_code: str, date_str: str) -> str:
        """
        为真实的、带有买卖盘属性的逐笔成交数据 (realtime_tick) 生成ZSET缓存键。
        """
        return self.generate_key(
            cache_type=cc.TYPE_TIMESERIES,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype='real_ticks', # 使用 'real_ticks' 以区别于之前的快照 'ticks_realtime'
            date=date_str
        )
    def stock_concepts(self, stock_code: str, source: str) -> str:
        """
        【V1.0 新增】生成单个股票特定来源的概念/行业列表的缓存键。
        例如: st:stock:000001.SZ:concepts:source:sw
        """
        return self.generate_key(
            cache_type=cc.TYPE_STATIC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_CONCEPTS,
            params={'source': source}
        )

class StrategyCashKey(CashKey):
    def __init__(self):
        self.cache_manager = None  # 改为 None，等待异步初始化
    async def initialize(self):
        self.cache_manager = await cache_manager  # 异步初始化
    def analyze_signals_trend_following(self, stock_code: str) -> str:
        cache_key = self.generate_key(
            cache_type=cc.TYPE_STRATEGY,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype=cc.SUBTYPE_STRATEGY_TREND_FOLLOWING,
            # params={cc.PARAM_PERIOD: time_level}
        )
        return cache_key

class IntradayEngineCashKey(CashKey):
    """
    盘中引擎专用缓存键生成器
    """
    def _engine_base_key(self, date_str: str) -> str:
        """引擎当日的基础键前缀"""
        return self.generate_key(
            cache_type=cc.TYPE_STRATEGY,
            entity_type='intraday_engine',
            entity_id=date_str
        )
    def watchlist_key(self, date_str: str) -> str:
        """待买入池的缓存键 (使用Redis Set)"""
        return f"{self._engine_base_key(date_str)}:watchlist"
    def position_list_key(self, date_str: str) -> str:
        """持仓监控池的缓存键 (使用Redis Hash)"""
        return f"{self._engine_base_key(date_str)}:position_list"
    def user_signals_key(self, user_id: int, date_str: str) -> str:
        """单个用户当日盘中信号的缓存键 (使用Redis List)"""
        return self.generate_key(
            cache_type=cc.TYPE_USER,
            entity_type='intraday_signals',
            entity_id=str(user_id),
            date=date_str
        )
    """
    盘中实时计算引擎专用的缓存键生成器。
    """
    def stock_calculated_data_key(self, stock_code: str, trade_date: str) -> str:
        """
        为单支股票的【完整计算结果】生成的键 (中间数据)。
        使用 'calc' 类型，表示这是一个计算过程的产物。
        """
        return self.generate_key(
            cache_type=cc.TYPE_CALC,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype='intraday_full_metrics', # 子类型明确指出是盘中全量指标
            date=trade_date.replace('-', '') # 使用 YYYYMMDD 格式的日期
        )
    def stock_signals_key(self, stock_code: str, trade_date: str) -> str:
        """
        为单支股票的【最终信号】生成的键 (给前端使用)。
        使用 'strategy' 类型，因为信号是策略的一部分。
        """
        return self.generate_key(
            cache_type=cc.TYPE_STRATEGY,
            entity_type=cc.ENTITY_STOCK,
            entity_id=stock_code,
            subtype='intraday_signals', # 子类型明确指出是盘中信号
            date=trade_date.replace('-', '') # 使用 YYYYMMDD 格式的日期
        )









