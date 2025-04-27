import asyncio
import logging
from typing import Any, List, Optional, Set, Union, Dict, Type
import pandas as pd
from django.db.models import Max # <--- 确保导入 Max
import numpy as np
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from django.db import models
from django.utils import timezone
from django.core.exceptions import FieldDoesNotExist
from dao_manager.base_dao import BaseDAO
from core.constants import TimeLevel, FIB_PERIODS, FINTA_OHLCV_MAP
from stock_models.time_trade import StockDailyData, StockMinuteData, StockMonthlyData, StockTimeTrade, StockWeeklyData
from utils.cache_get import  StockTimeTradeCacheGet
from utils.cache_manager import CacheManager

logger = logging.getLogger("dao")

class IndicatorDAO(BaseDAO):
    """
    指标数据访问对象，负责指标数据的读取和存储
    """
    def __init__(self):
        # 依赖注入基础DAO和缓存工具
        from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
        self.stock_basic_dao = StockBasicInfoDao()
        self.cache_manager = None
        self.cache_get = None

    async def initialize_cache_objects(self):
        self.cache_manager = CacheManager()  # 先实例化
        await self.cache_manager.initialize()  # 然后 await 其异步初始化方法，如果存在
        self.cache_get = StockTimeTradeCacheGet()  # 先实例化
        await self.cache_get.initialize()  # 添加异步初始化方法，如果需要

    async def get_history_time_trades_by_limit(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[List[StockTimeTrade]]:
        """
        获取指定股票、时间级别和数量限制的历史分时交易数据。
        查询顺序:
        1. 尝试从 Redis 缓存获取数据 (期望格式为 List[Dict])。
        2. 如果缓存命中，手动将字典列表转换为 `StockTimeTrade` 模型实例列表。
        3. 如果缓存未命中或转换失败，则从数据库查询。
        4. 返回按交易时间升序排列的模型实例列表。
        Args:
            stock_code: 股票代码。
            time_level: 时间周期级别 (可以是 TimeLevel 枚举或其字符串表示)。
            limit: 需要获取的最新数据条数。
        Returns:
            包含 `StockTimeTrade` 模型实例的列表 (按时间升序)，如果找不到数据或出错则返回 None。
        """
        # 确保缓存对象已初始化
        if self.cache_get is None or self.stock_basic_dao is None:
            await self.initialize_cache_objects()
        # 获取股票基础信息对象
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None
        # 标准化时间级别为字符串
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        # --- 1. 尝试从 Redis 缓存获取数据 ---
        cache_data: Optional[List[Dict]] = None
        try:
            cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level_str, limit)
        except Exception as e:
            logger.error(f"从 Redis 获取缓存数据时出错 for {stock} {time_level_str}: {e}", exc_info=True)
            cache_data = None
        # --- 2. 处理缓存数据 (如果命中) ---
        if cache_data and isinstance(cache_data, list):
            # logger.debug(f"缓存命中: 获取到 {stock_code} {time_level_str} 历史数据 {len(cache_data)} 条 (limit={limit})，进行手动转换...")
            model_instances = []
            conversion_errors = 0
            for item_dict_str in cache_data:
                try:
                    if isinstance(item_dict_str, bytes):
                        item_dict = self.cache_manager._deserialize(item_dict_str)
                    else:
                        item_dict = item_dict_str
                    # 手动将字典转换为 StockTimeTrade 模型实例
                    trade_time = self._safe_datetime(item_dict.get('trade_time'))
                    if not trade_time: # 如果时间无效，跳过此条记录
                        logger.warning(f"缓存数据中发现无效的 trade_time: {item_dict.get('trade_time')}")
                        conversion_errors += 1
                        continue
                    # 注意：这里不设置 id，因为缓存字典中没有
                    instance = StockMinuteData(
                        stock=stock,
                        trade_time=self._safe_decimal(item_dict.get('trade_time')),
                        time_level="Day",
                        open=self._safe_decimal(item_dict.get('open')),
                        high=self._safe_decimal(item_dict.get('high')),
                        low=self._safe_decimal(item_dict.get('low')),
                        close=self._safe_decimal(item_dict.get('close')),
                        vol=self._safe_decimal(item_dict.get('vol')),
                        amount=self._safe_decimal(item_dict.get('amount')),
                        # 其他 StockTimeTrade 可能有的字段，如果缓存中有，也需要在这里添加转换逻辑
                    )
                    model_instances.append(instance)
                    if time_level_str == "Day":
                        instance.time_level = "day" # 确保正确设置 time_level，避免重复设置
                    elif time_level_str == "week":
                        instance.time_level = "week"
                    elif time_level_str == "month":
                        instance.time_level = "month"
                    else:
                        instance.time_level = time_level_str
                except Exception as e_conv:
                    conversion_errors += 1
                    # 记录转换单个字典时的错误，避免打印过多日志可以选择不显示 exc_info
                    logger.error(f"转换缓存字典为 StockTimeTrade 实例时出错: {e_conv}. Dict: {item_dict}", exc_info=False)
            if conversion_errors > 0:
                 logger.warning(f"转换缓存数据时遇到 {conversion_errors} 个错误 for {stock_code} {time_level_str}")
            if not model_instances:
                logger.warning(f"缓存数据转换后为空列表 for {stock_code} {time_level_str}，将尝试从数据库获取。")
                # 不返回 None，继续尝试数据库
            else:
                # 按交易时间升序排序
                model_instances.sort(key=lambda x: x.trade_time)
                logger.debug(f"成功从缓存转换 {len(model_instances)} 条 StockTimeTrade 实例 for {stock_code} {time_level_str}")
                return model_instances # 返回成功转换的实例列表
        # --- 3. 缓存未命中或处理失败，从数据库获取 ---
        logger.debug(f"缓存未命中或处理失败 for {stock_code} {time_level_str}，从数据库获取...")
        try:
            # 查询数据库，按时间降序获取最新的 limit 条
            if time_level_str == "Day":
                data_qs = StockDailyData.objects.filter(
                    stock=stock,
                    time_level=time_level_str
                ).order_by('-trade_time')[:limit]
            elif time_level_str == "week":
                data_qs = StockWeeklyData.objects.filter(
                    stock=stock,
                    time_level=time_level_str
                ).order_by('-trade_time')[:limit]
            elif time_level_str == "month":
                data_qs = StockMonthlyData.objects.filter(
                    stock=stock,
                    time_level=time_level_str
                ).order_by('-trade_time')[:limit]
            else:
                data_qs = StockMinuteData.objects.filter(
                    stock=stock,
                    time_level=time_level_str
                ).order_by('-trade_time')[:limit]
            # 异步执行查询
            data_list = await sync_to_async(list)(data_qs)
            if not data_list:
                logger.warning(f"数据库中未找到 {stock_code} {time_level_str} 的历史数据")
                return None
            logger.debug(f"从数据库获取到 {stock_code} {time_level_str} {len(data_list)} 条历史数据")
            # 反转列表，得到升序排列
            data_list.reverse()
            return data_list # 返回从数据库获取并排序后的列表
        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None # 查询失败返回 None

    # 
    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取历史数据并转换为 finta 需要的 DataFrame 格式。
        返回的 DataFrame 按时间升序排列。
        (此方法逻辑不变，因为它依赖于 get_history_time_trades_by_limit 返回 List[StockTimeTrade])
        """
        time_level_str = ""
        if time_level == "5m":
            time_level_str = '5'
        elif time_level == '15m':
            time_level_str = '15'
        elif time_level == '30m':
            time_level_str = '30'
        elif time_level == '60m':
            time_level_str = '60'
        elif time_level == 'D':
            time_level_str = 'day'
        elif time_level == "W":
            time_level_str = 'week'
        elif time_level == "M":
            time_level_str = 'month'
        else:
            time_level_str = time_level
        history_trades = await self.get_history_time_trades_by_limit(stock_code, time_level_str, limit)
        if not history_trades:
            logger.warning(f"get_history_time_trades_by_limit 未返回数据 for {stock_code} {time_level}")
            return None
        try:
            # 1. 将模型列表转换为 DataFrame
            data = [
                {
                    # 使用 getattr 安全访问，并处理 None
                    'trade_time': getattr(trade, 'trade_time', None),
                    'open': float(getattr(trade, 'open', None)) if getattr(trade, 'open', None) is not None else np.nan,
                    'high': float(getattr(trade, 'high', None)) if getattr(trade, 'high', None) is not None else np.nan,
                    'low': float(getattr(trade, 'low', None)) if getattr(trade, 'low', None) is not None else np.nan,
                    'close': float(getattr(trade, 'close', None)) if getattr(trade, 'close', None) is not None else np.nan,
                    'vol': int(getattr(trade, 'vol', 0)) if getattr(trade, 'vol', None) is not None else 0, # finta 需要整数或浮点数
                    'amount': float(getattr(trade, 'amount', None)) if getattr(trade, 'amount', None) is not None else np.nan,
                }
                for trade in history_trades if getattr(trade, 'trade_time', None) is not None # 过滤掉没有时间的记录
            ]
            if not data:
                logger.warning(f"从 StockTimeTrade 实例转换的数据列表为空: {stock_code} {time_level}")
                return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 DataFrame 为空: {stock_code} {time_level}")
                return None
            # 2. 对非数值列应用分类类型 (在 df 创建之后)
            # 确保在访问 df.columns 之前 df 已经被定义
            for col in df.columns:
                # 检查列是否存在且类型为 object，并且不是时间索引列
                if col in df.columns and df[col].dtype == 'object' and col != 'trade_time':
                    try:
                        df[col] = df[col].astype('category')
                    except Exception as e_cat:
                        # 如果转换失败，记录警告但继续
                        logger.warning(f"转换列 '{col}' 为 category 类型失败: {e_cat}")
            # 3. 重命名列以匹配 finta 要求
            df.rename(columns=FINTA_OHLCV_MAP, inplace=True)
            # 4. 将 trade_time 设置为索引
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 5. 去除重复索引
            initial_len = len(df)
            if df.index.has_duplicates:
                # logger.warning(f"发现重复的时间戳索引 for {stock_code} {time_level}，将进行去重处理 (保留最后一个)")
                # 保留每个重复时间戳的最后一条记录
                df = df[~df.index.duplicated(keep='last')]
                # logger.info(f"索引去重完成 for {stock_code} {time_level}，记录数从 {initial_len} 变为 {len(df)}")
            # --- 去重结束 ---
            # 6. 确保数据按时间升序排列
            df.sort_index(ascending=True, inplace=True)
            # 7. 验证必要的列是否存在
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"DataFrame 缺少必要列: {stock_code} {time_level}. 需要: {required_cols}, 实际: {df.columns.tolist()}")
                return None
            # 移除完全是 NaN 的行 (如果需要)
            # df.dropna(subset=required_cols, how='all', inplace=True) # 谨慎使用，可能移除计算指标需要的数据点
            # 8. 检查是否有足够的非 NaN 数据行
            if df[required_cols].isnull().all(axis=1).sum() == len(df):
                 logger.warning(f"处理后 DataFrame 只包含 NaN 值: {stock_code} {time_level}")
                 return None # 如果全是 NaN，返回 None
            return df
        except Exception as e:
            logger.error(f"转换 {stock_code} {time_level} 历史数据为 DataFrame 失败: {str(e)}", exc_info=True)
            return None

    # --- 其他 DAO 方法 ---
    @staticmethod
    def _safe_decimal(value, default=None) -> Optional[Decimal]:
        """
        安全地将值转换为 Decimal。
        处理可能带小数点的字符串或浮点数。
        如果转换失败，返回 default 值。
        Args:
            value: 要转换的值，可以是任何类型。
            default: 转换失败时返回的默认值。
        Returns:
            转换后的 Decimal 值，如果转换失败则返回 default。
        """
        if value is None or value == '':
            return default
        try:
            # 如果已经是 Decimal，直接返回
            if isinstance(value, Decimal):
                return value
            # 从字符串或浮点数转换，推荐先转字符串保证精度
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value)}) 转换为 Decimal", exc_info=False)
            return default

    @staticmethod
    def _safe_int(value, default=None) -> Optional[int]:
        """
        安全地将值转换为 Integer。
        处理可能带小数点的字符串或浮点数。
        如果转换失败，返回 default 值。
        Args:
            value: 要转换的值，可以是任何类型。
            default: 转换失败时返回的默认值。
        Returns:
            转换后的 Integer 值，如果转换失败则返回 default。
        """
        if value is None or value == '':
            return default
        try:
            # 如果已经是 Int，直接返回
            if isinstance(value, int):
                return value
            # 处理可能带小数点的字符串或浮点数
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value)}) 转换为 Integer", exc_info=False)
            return default

    @staticmethod
    def _safe_datetime(value, default=None) -> Optional[timezone.datetime]:
        """安全地将值转换为 timezone-aware datetime"""
        if value is None:
            return default
        try:
            # pandas.to_datetime 是一个强大的解析器
            dt = pd.to_datetime(value)
            # 确保时区感知，如果已经是，则不变；如果是 naive，则设置为默认时区
            if timezone.is_naive(dt):
                return timezone.make_aware(dt, timezone.get_default_timezone())
            return dt
        except (ValueError, TypeError):
             logger.warning(f"无法将值 '{value}' (类型: {type(value)}) 转换为 Datetime", exc_info=False)
             return default
    
    @staticmethod
    def _prepare_decimal(value) -> Optional[Decimal]:
        """将计算结果（可能是 float 或 NaN）安全地转换为 Decimal 或 None"""
        if value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value))):
            return None
        try:
            # 使用之前定义的更安全的转换函数
            return IndicatorDAO._safe_decimal(value)
            # return Decimal(str(value)).quantize(Decimal("0.0001")) # 保留4位小数，根据模型调整
        except (InvalidOperation, TypeError):
            logger.warning(f"无法将值 {value} (类型: {type(value)}) 转换为 Decimal", exc_info=False)
            return None

    @staticmethod
    def _prepare_int(value) -> Optional[int]:
        """将计算结果安全地转换为 int 或 None"""
        if value is None or (isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value))):
            return None
        try:
             # 使用之前定义的更安全的转换函数
            return IndicatorDAO._safe_int(value)
            # return int(value)
        except (ValueError, TypeError):
            logger.warning(f"无法将值 {value} (类型: {type(value)}) 转换为 Integer", exc_info=False)
            return None














