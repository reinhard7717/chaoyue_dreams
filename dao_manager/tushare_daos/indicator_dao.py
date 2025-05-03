# dao_manager\tushare_daos\indicator_dao.py
import asyncio
import datetime
import logging
from typing import Any, List, Optional, Set, Union, Dict, Type
import pandas as pd
from django.db.models import Max # <--- 确保导入 Max
import numpy as np
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from django.utils import timezone
from dao_manager.base_dao import BaseDAO
from core.constants import TimeLevel, FIB_PERIODS, FINTA_OHLCV_MAP
from stock_models.time_trade import StockDailyData, StockMinuteData, StockMonthlyData, StockTimeTrade, StockWeeklyData
from utils.cache_get import  StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO

logger = logging.getLogger("dao")

def get_china_a_stock_kline_times(trade_days: list, time_level: str) -> list:
    """
    生成A股应有的K线时间点，基于实际交易日历。
    trade_days: list of datetime.date，实际交易日列表
    time_level: 'day', 'week', 'month', '5', '15', '30', '60'
    返回: list of pd.Timestamp (Asia/Shanghai)
    """
    times = []
    if time_level == 'day':
        for day in trade_days:
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz='Asia/Shanghai'))
    elif time_level == 'week':
        # 只保留每周最后一个交易日
        week_map = {}
        for day in trade_days:
            week = pd.Timestamp(day).isocalendar()[1]
            year = pd.Timestamp(day).year
            key = (year, week)
            if key not in week_map or day > week_map[key]:
                week_map[key] = day
        for day in week_map.values():
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz='Asia/Shanghai'))
    elif time_level == 'month':
        # 只保留每月最后一个交易日
        month_map = {}
        for day in trade_days:
            month = pd.Timestamp(day).month
            year = pd.Timestamp(day).year
            key = (year, month)
            if key not in month_map or day > month_map[key]:
                month_map[key] = day
        for day in month_map.values():
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz='Asia/Shanghai'))
    elif time_level in ['5', '15', '30', '60']:
        freq = int(time_level)
        for day in trade_days:
            # 上午交易时间：9:30 - 11:30
            morning_start = datetime.datetime.combine(day, datetime.time(9, 30))
            morning_end = datetime.datetime.combine(day, datetime.time(11, 30))
            t = morning_start
            while t <= morning_end:
                times.append(pd.Timestamp(t, tz='Asia/Shanghai'))
                t += datetime.timedelta(minutes=freq)
            # 下午交易时间：13:00 - 15:00
            afternoon_start = datetime.datetime.combine(day, datetime.time(13, 0))
            afternoon_end = datetime.datetime.combine(day, datetime.time(15, 0))
            t = afternoon_start
            while t <= afternoon_end:
                times.append(pd.Timestamp(t, tz='Asia/Shanghai'))
                t += datetime.timedelta(minutes=freq)
            # 去除超出收盘的时间点
            times = [x for x in times if (x.time() <= datetime.time(11, 30) or (x.time() >= datetime.time(13, 0) and x.time() <= datetime.time(15, 0)))]
    else:
        raise ValueError(f"不支持的K线类型: {time_level}")
    return sorted(times)

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
        if self.cache_get is None or self.stock_basic_dao is None:
            await self.initialize_cache_objects()
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        time_level_str = time_level_str.lower()
        cache_data: Optional[List[Dict]] = None
        try:
            cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level_str, limit)
        except Exception as e:
            logger.error(f"从 Redis 获取缓存数据时出错 for {stock} {time_level_str}: {e}", exc_info=True)
            cache_data = None
        if len(cache_data) >= limit:
            if cache_data and isinstance(cache_data, list):
                model_instances = []
                conversion_errors = 0
                for item_dict_str in cache_data:
                    try:
                        if isinstance(item_dict_str, bytes):
                            item_dict = self.cache_manager._deserialize(item_dict_str)
                        else:
                            item_dict = item_dict_str
                        trade_time = self._safe_datetime(item_dict.get('trade_time'))
                        if not trade_time:
                            logger.warning(f"缓存数据中发现无效的 trade_time: {item_dict.get('trade_time')}")
                            conversion_errors += 1
                            continue
                        # 修改：将缓存字段名映射为模型字段名 --- 注意新增字段名称对应
                        if time_level_str in ['day', 'week', 'month']:
                            # 使用对应日/周/月模型
                            if time_level_str == 'day':
                                ModelClass = StockDailyData
                            elif time_level_str == 'week':
                                ModelClass = StockWeeklyData
                            else:
                                ModelClass = StockMonthlyData
                            instance = ModelClass(
                                stock=stock,
                                trade_time=trade_time,
                                open=self._safe_decimal(item_dict.get('open')),
                                high=self._safe_decimal(item_dict.get('high')),
                                low=self._safe_decimal(item_dict.get('low')),
                                close=self._safe_decimal(item_dict.get('close')),
                                vol=self._safe_int(item_dict.get('vol')),
                                amount=self._safe_decimal(item_dict.get('amount')),
                            )
                            instance.time_level = time_level_str
                        else:
                            # 按分钟数据 StockMinuteData
                            instance = StockMinuteData(
                                stock=stock,
                                trade_time=trade_time,
                                time_level=time_level_str,
                                open=self._safe_decimal(item_dict.get('open')),
                                high=self._safe_decimal(item_dict.get('high')),
                                low=self._safe_decimal(item_dict.get('low')),
                                close=self._safe_decimal(item_dict.get('close')),
                                vol=self._safe_int(item_dict.get('vol')),
                                amount=self._safe_decimal(item_dict.get('amount')),
                            )
                        model_instances.append(instance)
                    except Exception as e_conv:
                        conversion_errors += 1
                        logger.error(f"转换缓存字典为 StockTimeTrade 实例时出错: {e_conv}. Dict: {item_dict}", exc_info=False)
                if conversion_errors > 0:
                    logger.warning(f"转换缓存数据时遇到 {conversion_errors} 个错误 for {stock_code} {time_level_str}")
                if not model_instances:
                    logger.warning(f"缓存数据转换后为空列表 for {stock_code} {time_level_str}，将尝试从数据库获取。")
                else:
                    model_instances.sort(key=lambda x: x.trade_time)
                    logger.debug(f"成功从缓存转换 {len(model_instances)} 条 StockTimeTrade 实例 for {stock_code} {time_level_str}")
                    return model_instances
            logger.debug(f"缓存未命中或处理失败 for {stock_code} {time_level_str}，从数据库获取...")
        try:
            if time_level_str == "day":
                data_qs = StockDailyData.objects.filter(
                    stock=stock,
                ).order_by('-trade_time')[:limit]
            elif time_level_str == "week":
                data_qs = StockWeeklyData.objects.filter(
                    stock=stock,
                ).order_by('-trade_time')[:limit]
            elif time_level_str == "month":
                data_qs = StockMonthlyData.objects.filter(
                    stock=stock,
                ).order_by('-trade_time')[:limit]
            else:
                # 对分钟线数据，由于分钟线有 time_level 字段，需过滤
                data_qs = StockMinuteData.objects.filter(
                    stock=stock,
                    time_level=time_level_str
                ).order_by('-trade_time')[:limit]
            data_list = await sync_to_async(list)(data_qs)
            if not data_list:
                logger.warning(f"数据库中未找到 {stock_code} {time_level_str} 的历史数据")
                return None
            data_list.reverse()

            # 1. 获取实际有的数据时间点
            trade_times = [getattr(trade, 'trade_time', None) for trade in data_list if getattr(trade, 'trade_time', None) is not None]
            trade_times = sorted([pd.to_datetime(t).tz_convert('Asia/Shanghai') if pd.to_datetime(t).tzinfo else pd.to_datetime(t).tz_localize('Asia/Shanghai') for t in trade_times])
            
            # 2. 获取应有的交易日
            index_basic_dao = IndexBasicDAO()
            start_date = trade_times[0].strftime('%Y%m%d') if trade_times else (timezone.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
            end_date = trade_times[-1].strftime('%Y%m%d') if trade_times else timezone.now().strftime('%Y%m%d')
            trade_days = await index_basic_dao.get_trade_cal_open(start_date, end_date)
            trade_days = [pd.to_datetime(day).date() for day in trade_days]  # 转为date对象
            
            # 3. 生成应有的K线时间点（基于实际交易日）
            expected_times = get_china_a_stock_kline_times(trade_days, time_level_str)
            
            # 4. 检查缺失比例并决定是否返回数据
            actual_times_set = set([t.replace(second=0, microsecond=0) for t in trade_times])
            expected_times_set = set([t.replace(second=0, microsecond=0) for t in expected_times])
            missing_times = expected_times_set - actual_times_set
            missing_ratio = len(missing_times) / len(expected_times_set) if expected_times_set else 0
            missing_threshold = 0.5 if time_level_str in ['5', '15', '30', '60'] else 0.1  # 分钟级别放宽阈值到50%
            
            if missing_times:
                logger.warning(f"原始K线数据时间序列有缺失: {stock_code} {time_level_str}，缺失数量: {len(missing_times)}，缺失比例: {missing_ratio:.2%}，缺失时间: {sorted(list(missing_times))[:5]} ...")
                if missing_ratio > missing_threshold:
                    logger.error(f"数据缺失比例 {missing_ratio:.2%} 超过阈值 {missing_threshold}，拒绝返回数据: {stock_code} {time_level_str}")
                    return None
            else:
                logger.info(f"原始K线数据时间序列无缺失: {stock_code} {time_level_str}")
            
            return data_list
        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None

    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取历史数据并转换为 finta 需要的 DataFrame 格式。
        返回的 DataFrame 按时间升序排列。
        根据时间级别明确调用对应模型数据读取。
        """
        # 1. 统一处理 time_level，支持 TimeLevel 枚举及字符串
        if isinstance(time_level, TimeLevel):
            time_level_val = time_level.value
        else:
            time_level_val = str(time_level)
        # 2. 将大写特殊时间级别转换为对应的小写别名
        # D => day； W => week； M => month； 如 5m => 5 等
        tl_lower = time_level_val.lower()
        if tl_lower == 'd' or tl_lower == 'day':
            time_level_str = 'day'
        elif tl_lower == 'w' or tl_lower == 'week':
            time_level_str = 'week'
        elif tl_lower == 'm' or tl_lower == 'month':
            time_level_str = 'month'
        elif tl_lower.endswith('m') and tl_lower[:-1].isdigit():
            # 类似 '5m', '15m' 变 '5', '15'
            time_level_str = tl_lower[:-1]
        else:
            time_level_str = tl_lower
        # 3. 调用：根据 time_level_str 获取对应模型数据
        history_trades = await self.get_history_time_trades_by_limit(stock_code, time_level_str, limit)
        if not history_trades:
            logger.warning(f"get_history_time_trades_by_limit 未返回数据 for {stock_code} {time_level_val}")
            return None
        try:
            # 4. 模型列表转换成字典列表
            data = []
            for trade in history_trades:
                tt = getattr(trade, 'trade_time', None)
                if tt is None:
                    continue
                data.append({
                    'trade_time': tt,
                    'open': float(getattr(trade, 'open', np.nan)) if getattr(trade, 'open', None) is not None else np.nan,
                    'high': float(getattr(trade, 'high', np.nan)) if getattr(trade, 'high', None) is not None else np.nan,
                    'low': float(getattr(trade, 'low', np.nan)) if getattr(trade, 'low', None) is not None else np.nan,
                    'close': float(getattr(trade, 'close', np.nan)) if getattr(trade, 'close', None) is not None else np.nan,
                    'volume': int(getattr(trade, 'vol', 0)) if getattr(trade, 'vol', None) is not None else 0,
                    'amount': float(getattr(trade, 'amount', np.nan)) if getattr(trade, 'amount', None) is not None else np.nan,
                })
            if not data:
                logger.warning(f"从 StockTimeTrade 实例转换的数据列表为空: {stock_code} {time_level_val}")
                return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 DataFrame 为空: {stock_code} {time_level_val}")
                return None
            # 5. 类型转换
            for col in df.columns:
                if col != 'trade_time' and df[col].dtype == 'object':
                    try:
                        df[col] = df[col].astype('category')
                    except Exception as e:
                        logger.warning(f"转换列 '{col}' 为 category 类型失败: {e}")
            # 6. 重命名列适配 finta
            df.rename(columns=FINTA_OHLCV_MAP, inplace=True)
            # 7. 时间列转换设为索引，丢弃解析失败行
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True, errors='coerce')
            df.dropna(subset=['trade_time'], inplace=True)
            df.set_index('trade_time', inplace=True)
            # 8. 去重索引，保留最后一个
            if df.index.has_duplicates:
                df = df[~df.index.duplicated(keep='last')]
            # 9. 按时间升序排序
            df.sort_index(ascending=True, inplace=True)
            # 10. 校验必要列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"DataFrame 缺少必要列: {stock_code} {time_level_val}. 需要: {required_cols}, 实际: {df.columns.tolist()}")
                return None
            # 11. 校验是否整张表全部NaN
            if df[required_cols].isnull().all(axis=1).sum() == len(df):
                logger.warning(f"处理后 DataFrame 只包含 NaN 值: {stock_code} {time_level_val}")
                return None
            
            # 12. 检查缺失比例（基于必要列）
            missing_ratio = df[required_cols].isnull().mean().mean()  # 计算必要列的平均缺失比例
            missing_threshold = 0.1  # 缺失比例阈值，10%
            if missing_ratio > missing_threshold:
                logger.error(f"数据缺失比例 {missing_ratio:.2%} 超过阈值 {missing_threshold}，拒绝返回 DataFrame: {stock_code} {time_level_val}")
                return None
            
            logger.info(f"返回 DataFrame，缺失比例: {missing_ratio:.2%}，数据量: {len(df)} 条: {stock_code} {time_level_val}")
            return df
        except Exception as e:
            logger.error(f"转换 {stock_code} {time_level_val} 历史数据为 DataFrame 失败: {str(e)}", exc_info=True)
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














