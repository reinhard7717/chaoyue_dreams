# dao_manager\tushare_daos\indicator_dao.py
import asyncio
import datetime
import logging
from typing import Any, List, Optional, Set, Union, Dict, Type
import pandas as pd
from django.db.models import Max # <--- 确保导入 Max
from django.db import models # <--- 确保导入 models 以支持类型提示
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
    Args:
        trade_days: list of datetime.date，实际交易日列表
        time_level: 'day', 'week', 'month', '5', '15', '30', '60'
    Returns:
        list of pd.Timestamp (Asia/Shanghai)
    """
    times = []
    if time_level == 'day':
        for day in trade_days:
            # 日线数据的时间点通常设为当日开始（午夜）
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz='Asia/Shanghai'))
    elif time_level == 'week':
        # 只保留每周最后一个交易日的时间点
        week_map = {}
        for day in trade_days:
            # 使用 ISO 年份和周次作为 key
            week = pd.Timestamp(day).isocalendar()[1]
            year = pd.Timestamp(day).year
            key = (year, week)
            if key not in week_map or day > week_map[key]:
                week_map[key] = day
        # 按照日期排序，转换为时间戳
        for day in sorted(week_map.values()):
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz='Asia/Shanghai'))
    elif time_level == 'month':
        # 只保留每月最后一个交易日的时间点
        month_map = {}
        for day in trade_days:
            month = pd.Timestamp(day).month
            year = pd.Timestamp(day).year
            key = (year, month)
            if key not in month_map or day > month_map[key]:
                month_map[key] = day
        # 按照日期排序，转换为时间戳
        for day in sorted(month_map.values()):
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz='Asia/Shanghai'))
    elif time_level in ['5', '15', '30', '60']:
        freq = int(time_level)
        for day in trade_days:
            # A股上午交易时间段：9:30 - 11:30
            morning_start = datetime.datetime.combine(day, datetime.time(9, 30))
            morning_end = datetime.datetime.combine(day, datetime.time(11, 30))
            t = morning_start
            while t <= morning_end:
                times.append(pd.Timestamp(t, tz='Asia/Shanghai'))
                t += datetime.timedelta(minutes=freq)
            # A股下午交易时间段：13:00 - 15:00
            afternoon_start = datetime.datetime.combine(day, datetime.time(13, 0))
            afternoon_end = datetime.datetime.combine(day, datetime.time(15, 0))
            t = afternoon_start
            while t <= afternoon_end:
                times.append(pd.Timestamp(t, tz='Asia/Shanghai'))
                t += datetime.timedelta(minutes=freq)

        # 过滤掉不在A股交易时间内的整点（例如 11:30 之后 13:00 之前的点，或者 15:00 之后的点）
        # 交易所数据通常提供的是某个时间点结束的K线，例如 9:35 的5分钟K线包含 9:30-9:35 的数据，时间点是 9:35
        # 因此这里生成的预期时间点应该是每个K线周期的结束时间
        # 对于 5分钟 K线，从 9:35 开始，到 11:30 结束，下午 13:05 开始，到 15:00 结束
        # 重新生成分钟线预期时间点，以 K线结束时间为准
        times = []
        for day in trade_days:
             # 上午 9:35, 9:40, ..., 11:30
             morning_times = pd.date_range(start=f'{day} 09:35:00', end=f'{day} 11:30:00', freq=f'{freq}T', tz='Asia/Shanghai')
             # 下午 13:05, 13:10, ..., 15:00
             afternoon_times = pd.date_range(start=f'{day} 13:05:00', end=f'{day} 15:00:00', freq=f'{freq}T', tz='Asia/Shanghai')
             times.extend(morning_times)
             times.extend(afternoon_times)

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
        self.cache_manager = None # 缓存管理器
        self.cache_get = None # 缓存获取工具

    async def initialize_cache_objects(self):
        """异步初始化缓存相关对象"""
        if self.cache_manager is None:
            # 假设 CacheManager 有异步初始化方法或者其 __init__ 是同步的
            self.cache_manager = CacheManager()
            # 如果 CacheManager 有需要 await 的初始化方法，在这里调用
            # if hasattr(self.cache_manager, 'initialize_async'):
            #     await self.cache_manager.initialize_async()

        if self.cache_get is None:
            # 假设 StockTimeTradeCacheGet 有异步初始化方法或者其 __init__ 是同步的
            self.cache_get = StockTimeTradeCacheGet()
            # 如果 StockTimeTradeCacheGet 有需要 await 的初始化方法，在这里调用
            # if hasattr(self.cache_get, 'initialize_async'):
            #     await self.cache_get.initialize_async()


    async def get_history_time_trades_by_limit(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[List[StockTimeTrade]]:
        """
        获取指定股票、时间级别和数量限制的历史分时交易数据（模型实例列表）。
        查询顺序: 缓存 -> 数据库。
        返回按交易时间升序排列的模型实例列表。
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

        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None

        # 统一时间级别字符串格式
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level)
        time_level_str = time_level_str.lower()

        cache_data: Optional[List[Dict]] = None
        try:
            # 尝试从 Redis 缓存获取数据
            # 假设缓存存储的是字典列表或可反序列化为字典列表的数据
            cache_data = await self.cache_get.history_time_trade_by_limit(stock_code, time_level_str, limit)
            # 检查缓存数据是否有效且数量满足要求
            if cache_data and isinstance(cache_data, list) and len(cache_data) >= limit:
                 logger.debug(f"从 Redis 获取到 {len(cache_data)} 条缓存数据 for {stock_code} {time_level_str}")
                 # 尝试将缓存数据（字典列表）转换为模型实例列表
                 model_instances = []
                 conversion_errors = 0
                 # TODO: 优化缓存反序列化和模型转换，避免手动处理每个字段
                 for item_dict_raw in cache_data:
                     try:
                         # 确保 item_dict_raw 是字典类型
                         item_dict = item_dict_raw
                         if isinstance(item_dict_raw, bytes):
                             # 如果缓存中存储的是 bytes，尝试反序列化 (取决于 cache_manager._deserialize 实现)
                              item_dict = self.cache_manager._deserialize(item_dict_raw)
                              # 反序列化后再次检查类型
                              if not isinstance(item_dict, dict):
                                   logger.warning(f"缓存反序列化结果不是字典: {type(item_dict_raw).__name__}, raw: {item_dict_raw[:100]}")
                                   conversion_errors += 1
                                   continue

                         # 根据时间级别选择对应的模型类
                         ModelClass: Type[models.Model] # Django Model 类型提示
                         if time_level_str == 'day':
                             ModelClass = StockDailyData
                         elif time_level_str == 'week':
                             ModelClass = StockWeeklyData
                         elif time_level_str == 'month':
                             ModelClass = StockMonthlyData
                         else:
                             ModelClass = StockMinuteData # 分钟线模型

                         # 安全获取并转换字段值
                         # 确保这里使用的键名与缓存中存储的字段名一致
                         trade_time = self._safe_datetime(item_dict.get('trade_time'))
                         open_price = self._safe_decimal(item_dict.get('open'))
                         high_price = self._safe_decimal(item_dict.get('high'))
                         low_price = self._safe_decimal(item_dict.get('low'))
                         close_price = self._safe_decimal(item_dict.get('close'))
                         volume = self._safe_int(item_dict.get('vol')) # 缓存中存储的字段名是 'vol'
                         amount = self._safe_decimal(item_dict.get('amount'))
                         # 添加其他字段根据模型实际情况添加，确保键名一致

                         if trade_time is None: # trade_time 是关键字段，如果为 None 则跳过此条
                             logger.warning(f"缓存数据中发现无效或缺失的 trade_time for {stock_code} {time_level_str}")
                             conversion_errors += 1
                             continue

                         # 实例化模型，分钟线需要 time_level 字段
                         if time_level_str in ['day', 'week', 'month']:
                            instance = ModelClass(
                                stock=stock,
                                trade_time=trade_time,
                                open=open_price,
                                high=high_price,
                                low=low_price,
                                close=close_price,
                                vol=volume,
                                amount=amount,
                                # 添加其他日/周/月模型特有的字段
                            )
                            # 日/周/月模型没有 time_level 字段
                         else:
                            instance = ModelClass(
                                stock=stock,
                                trade_time=trade_time,
                                time_level=time_level_str, # 分钟线需要 time_level 字段
                                open=open_price,
                                high=high_price,
                                low=low_price,
                                close=close_price,
                                vol=volume,
                                amount=amount,
                                # 添加其他分钟线模型特有的字段
                            )
                         model_instances.append(instance)

                     except Exception as e_conv:
                         conversion_errors += 1
                         # 仅记录错误类型和少量信息，避免日志过长
                         # 尝试打印导致错误的字典键，如果 item_dict 是字典
                         sample_keys = list(item_dict.keys()) if isinstance(item_dict, dict) else 'N/A'
                         logger.error(f"转换缓存字典为 Model 实例时出错 ({type(e_conv).__name__}) for {stock_code} {time_level_str}. Sample Dict Keys: {sample_keys}", exc_info=False)

                 if conversion_errors > 0:
                     logger.warning(f"从缓存转换 {stock_code} {time_level_str} 数据时遇到 {conversion_errors} 个错误。")

                 if model_instances:
                     # 按时间排序（尽管缓存应该是排序好的，再次排序确保万无一失）
                     model_instances.sort(key=lambda x: x.trade_time)
                     logger.debug(f"成功从缓存转换 {len(model_instances)} 条 Model 实例 for {stock_code} {time_level_str}")
                     return model_instances
                 else:
                      logger.warning(f"缓存数据转换后为空列表 for {stock_code} {time_level_str}，将尝试从数据库获取。")
            else:
                logger.debug(f"缓存未命中或数量不足 ({len(cache_data) if cache_data else 0} < {limit}) for {stock_code} {time_level_str}，从数据库获取...")

        except Exception as e:
            # 捕获缓存获取或初步处理时的异常
            logger.error(f"从 Redis 获取缓存数据时出错 for {stock_code} {time_level_str}: {e}", exc_info=True)
            # 缓存出错，继续从数据库获取

        # 从数据库查询
        try:
            # 根据时间级别选择对应的查询集
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

            # 执行查询并转为列表
            data_list = await sync_to_async(list)(data_qs)

            if not data_list:
                logger.warning(f"数据库中未找到 {stock_code} {time_level_str} 的历史数据")
                return None

            # 按时间升序排列
            data_list.reverse()

            # --- 以下是原始数据缺失检查部分 ---
            # 1. 获取实际有的数据时间点，并明确时区转换
            # 假设数据库返回的时间是 timezone aware 或 naive UTC
            trade_times_raw = [getattr(trade, 'trade_time', None) for trade in data_list if getattr(trade, 'trade_time', None) is not None]

            # 将数据库时间转换为上海时区，如果已经是 aware，则直接转；如果是 naive，假设是 UTC
            trade_times = []
            for t_raw in trade_times_raw:
                try:
                    # 使用 _safe_datetime 辅助函数进行安全转换和时区处理
                    safe_dt = self._safe_datetime(t_raw)
                    if safe_dt:
                        trade_times.append(safe_dt)
                    else:
                        logger.warning(f"从数据库获取的数据中发现无效或无法转换的 trade_time: {t_raw}")
                except Exception as e_tz_conv:
                    # _safe_datetime 内部已经有日志，这里可以简单处理
                    pass # 或者再次记录更高级别的警告

            trade_times.sort() # 确保时间排序

            # 2. 记录实际数据时间范围
            min_time: Optional[datetime.datetime] = None
            max_time: Optional[datetime.datetime] = None
            if trade_times:
                min_time = trade_times[0]
                max_time = trade_times[-1]
                # 对于分钟线，实际范围是 trade_times 中的最早和最晚时间
                # 对于日/周/月线，实际范围是 trade_times (日期) 中的最早和最晚日期
                logger.info(f"实际数据时间范围: {min_time} 至 {max_time}，数据量: {len(data_list)} 条，股票: {stock_code} {time_level_str}")
            else:
                logger.warning(f"无法确定实际数据时间范围，从数据库获取的 trade_times 列表为空，股票: {stock_code} {time_level_str}")
                # 如果没有时间点，但 data_list 不为空（理论上不发生），则返回原始列表
                # 但如果 trade_times 为空，说明所有 trade.trade_time 都是 None，数据有问题
                return None # 没有有效时间点，数据无效

            # 3. 获取应有的交易日，基于实际数据时间范围
            index_basic_dao = IndexBasicDAO()
            # 使用实际获取数据的日期范围来确定交易日历
            # 确保 start_date 和 end_date 是 YYYYMMDD 格式字符串
            # 如果 min_time 或 max_time 为 None，则使用默认日期范围
            start_date_str = min_time.strftime('%Y%m%d') if min_time else (timezone.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
            end_date_str = max_time.strftime('%Y%m%d') if max_time else timezone.now().strftime('%Y%m%d')

            trade_days = await index_basic_dao.get_trade_cal_open(start_date_str, end_date_str)
            trade_days_date = [pd.to_datetime(day).date() for day in trade_days]  # 转为date对象

            # 4. 生成应有的K线时间点（基于实际交易日）
            expected_times = get_china_a_stock_kline_times(trade_days_date, time_level_str)
            # expected_times 函数已经会生成在交易日内的标准时间点 (pd.Timestamp, Asia/Shanghai)

            # 5. 进一步过滤应有的时间点，使其落在实际获取数据的最小到最大时间范围内
            if min_time and max_time: # 确保 min_time 和 max_time 有效
                 # 将 min_time 和 max_time 转换为 pd.Timestamp 以便与 expected_times 中的元素 (pd.Timestamp) 比较
                 min_ts = pd.Timestamp(min_time)
                 max_ts = pd.Timestamp(max_time)
                 # 使用转换后的 pd.Timestamp 进行比较
                 expected_times_filtered = [t for t in expected_times if min_ts <= t <= max_ts]

                 # 记录调整后的预期时间点数量
                 if expected_times_filtered:
                     logger.info(f"预期时间点范围调整为: {expected_times_filtered[0]} 至 {expected_times_filtered[-1]}，调整后预期时间点数量: {len(expected_times_filtered)}，股票: {stock_code} {time_level_str}")
                 else:
                     logger.warning(f"在实际数据时间范围 {min_time} 至 {max_time} 内没有找到预期时间点，股票: {stock_code} {time_level_str}")
                     expected_times = [] # 如果过滤后为空，将预期时间点列表设为空
                 
                 expected_times = expected_times_filtered # 使用过滤后的预期时间点

            # 6. 检查缺失比例
            if not expected_times:
                logger.warning(f"无法生成任何预期时间点，跳过缺失检查。股票: {stock_code} {time_level_str}")
                # 即使无法检查缺失，如果 data_list 有数据，仍然返回
                return data_list

            # 将实际获取的时间点和预期时间点都转换为 DatetimeIndex 以便使用 difference 方法
            # 确保时间点精度一致以便比较
            # 对于分钟线，对齐到分钟；对于日/周/月线，只保留日期
            if time_level_str in ['5', '15', '30', '60']:
                freq = int(time_level_str)
                 # 实际时间点对齐到分钟间隔（如果原始数据不是精确对齐的，可能需要round）
                actual_times_index = pd.DatetimeIndex(trade_times).round(f'{freq}min')
                 # 预期时间点本身已经是对齐的 K线结束时间点
                expected_times_index = pd.DatetimeIndex(expected_times)
            else: # 日线、周线、月线
                actual_times_index = pd.DatetimeIndex(trade_times).normalize() # 只保留日期部分
                expected_times_index = pd.DatetimeIndex(expected_times).normalize() # 只保留日期部分

            # 找到缺失的时间点
            missing_index = expected_times_index.difference(actual_times_index)

            missing_count = len(missing_index)
            expected_count = len(expected_times_index)
            missing_ratio = missing_count / expected_count if expected_count else 0

            if missing_count > 0:
                # 打印缺失警告，包括数量、比例和部分缺失时间点
                # 将缺失时间转换为字符串列表以便日志打印
                logger.warning(f"原始K线数据时间序列有缺失: {stock_code} {time_level_str}，缺失数量: {missing_count}，缺失比例: {missing_ratio:.2%}，缺失时间 (部分): {[str(t) for t in missing_index[:5]]} ...")

                # --- 移除之前基于高阈值拒绝返回数据的逻辑 (已在上一次修改中移除) ---
                # 保留警告，但不在这里拒绝返回数据

            else:
                # 如果没有缺失，记录信息
                logger.info(f"原始K线数据时间序列无缺失: {stock_code} {time_level_str}")

            # 无论缺失多少，只要数据库查询有数据，就返回数据列表
            # 数据质量的最终判断和处理应由调用方 (Service 层) 负责
            return data_list

        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None

    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取历史数据并转换为 finta/pandas_ta 需要的 DataFrame 格式。
        返回的 DataFrame 按时间升序排列，并包含必要的 OHLCV 列（'open', 'high', 'low', 'close', 'volume'）
        以及可能的 'amount' 和 'turnover_rate' 等列。
        根据时间级别明确调用对应模型数据读取。
        """
        # 1. 统一处理 time_level，支持 TimeLevel 枚举及字符串
        if isinstance(time_level, TimeLevel):
            time_level_val = time_level.value
        else:
            time_level_val = str(time_level)

        # 2. 调用：根据 time_level_str 获取对应模型数据 (模型实例列表)
        # get_history_time_trades_by_limit 会处理 time_level 字符串到内部模型的映射
        history_trades = await self.get_history_time_trades_by_limit(stock_code, time_level_val, limit)

        if not history_trades:
            logger.warning(f"get_history_time_trades_by_limit 未返回数据 for {stock_code} {time_level_val}")
            return None

        try:
            # 3. 模型实例列表转换成字典列表
            data = []
            # 定义需要包含的字段及其类型转换方式
            fields_to_include = {
                'trade_time': lambda x: x, # 时间字段保留原始类型，后续 pd.to_datetime 处理
                'open': lambda x: float(x) if x is not None else np.nan,
                'high': lambda x: float(x) if x is not None else np.nan,
                'low': lambda x: float(x) if x is not None else np.nan,
                'close': lambda x: float(x) if x is not None else np.nan,
                'vol': lambda x: int(x) if x is not None else 0,
                'amount': lambda x: float(x) if x is not None else np.nan,
                # 根据您的模型，可能还需要 'turnover_rate' 等字段
                # 'turnover_rate': lambda x: float(x) if x is not None else np.nan,
                # 'pct_chg': lambda x: float(x) if x is not None else np.nan,
                # 'pe': lambda x: float(x) if x is not None else np.nan, # 日线等可能包含
                # ... 添加其他相关字段
            }

            for trade in history_trades:
                row_data = {}
                for field_name, converter in fields_to_include.items():
                    # 使用 getattr 安全地获取字段值
                    value = getattr(trade, field_name, None)
                    # 应用转换函数
                    try:
                         row_data[field_name] = converter(value)
                    except (ValueError, TypeError, InvalidOperation, AttributeError) as e:
                         # 如果转换失败，记录警告并使用默认的 NaN/0
                         logger.warning(f"转换字段 '{field_name}' 值 '{value}' 失败 for {stock_code} {time_level_val}: {e}", exc_info=False)
                         # 根据字段类型设置默认值
                         if 'vol' in field_name:
                             row_data[field_name] = 0
                         elif 'time' in field_name:
                              row_data[field_name] = pd.NaT
                         else:
                             row_data[field_name] = np.nan

                data.append(row_data)

            if not data:
                logger.warning(f"从 Model 实例转换的数据列表为空: {stock_code} {time_level_val}")
                return None

            df = pd.DataFrame(data)

            if df.empty:
                logger.warning(f"转换后的 DataFrame 为空: {stock_code} {time_level_val}")
                return None

            # 4. 列名标准化（适配 pandas_ta）
            # 注意：FINTA_OHLCV_MAP 应该是 {'open': 'open', 'high': 'high', ... 'volume': 'volume'} 这样的映射
            # pandas_ta 默认期望 'open', 'high', 'low', 'close', 'volume'
            # 这里的 rename 实际上是为了确保字段名是小写的 'open', 'high', 'low', 'close', 'volume', 'amount' 等
            # 更好的做法是直接在字段循环中将键名设为小写
            df.columns = [col.lower() for col in df.columns]
            # 如果您的模型字段名已经是小写，这一步是冗余但无害的

            # 5. 时间列转换为 DatetimeIndex 并设为索引
            # 确保 trade_time 列存在并是 DatetimeIndex
            if 'trade_time' not in df.columns:
                logger.error(f"转换后的 DataFrame 缺少 'trade_time' 列: {stock_code} {time_level_val}")
                return None

            # 强制转换为 datetime，处理错误并丢弃无效行
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True, errors='coerce')
            df.dropna(subset=['trade_time'], inplace=True) # 丢弃 trade_time 为 NaT 的行

            if df.empty:
                logger.warning(f"处理无效 trade_time 后 DataFrame 为空: {stock_code} {time_level_val}")
                return None

            df.set_index('trade_time', inplace=True)

            # 6. 去重索引，保留最后一个（最新数据）
            if df.index.has_duplicates:
                # 记录去重前的数量
                original_rows = len(df)
                df = df[~df.index.duplicated(keep='last')]
                logger.warning(f"移除重复索引，去重前: {original_rows}，去重后: {len(df)}，股票: {stock_code} {time_level_val}")

            # 7. 按时间升序排序索引
            df.sort_index(ascending=True, inplace=True)

            # 8. 校验必要列是否存在且不全为 NaN
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"DataFrame 缺少必要列: {stock_code} {time_level_val}. 需要: {required_cols}, 实际: {df.columns.tolist()}")
                return None

            # 检查必要列是否全部是 NaN
            if df[required_cols].isnull().all(axis=1).sum() == len(df):
                logger.warning(f"处理后 DataFrame 的必要列全部包含 NaN 值: {stock_code} {time_level_val}")
                # 即使必要列全NaN，为了让上层流程能拿到一个DataFrame，这里不直接返回None
                # 上层（Service层）在合并和填充后，会再进行最终的缺失检查

            # 9. 检查必要列的缺失比例（这个检查更合理，可以保留）
            # 计算必要列的平均缺失比例
            missing_ratio_required = df[required_cols].isnull().mean().mean()
            missing_threshold_df = 0.1  # DataFrame 层面缺失比例阈值，10%

            if missing_ratio_required > missing_threshold_df:
                 logger.warning(f"处理后 DataFrame 必要列平均缺失比例 {missing_ratio_required:.2%} 超过阈值 {missing_threshold_df}，可能影响后续计算: {stock_code} {time_level_val}. 各必要列缺失比例: {df[required_cols].isnull().mean().to_dict()}")
                 # 这里不拒绝返回，只打印警告，让 Service 层决定是否使用

            # 可以选择性地对 OHLCV 列进行简单的填充，例如前向填充
            # 但更建议在 Service 层合并多时间级别数据后再进行填充
            # df[required_cols] = df[required_cols].fillna(method='ffill')
            # df[required_cols] = df[required_cols].fillna(method='bfill')


            logger.info(f"返回 DataFrame，必要列平均缺失比例: {missing_ratio_required:.2%}，数据量: {len(df)} 条: {stock_code} {time_level_val}")

            return df

        except Exception as e:
            logger.error(f"转换 {stock_code} {time_level_val} 历史数据为 DataFrame 失败: {str(e)}", exc_info=True)
            return None

    # 添加安全转换辅助函数（确保存在且正确）
    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """将输入值安全转换为 Decimal 类型"""
        if value is None:
            return None
        try:
            # 尝试直接转换 Decimal
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 转换为 Decimal。")
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        """将输入值安全转换为 int 类型"""
        if value is None:
            return None
        try:
            # 先尝试转 float 处理科学计数法，再转 int
            return int(float(value))
        except (ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 转换为 int。")
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        """将输入值安全转换为 float 类型"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 转换为 float。")
            return None

    def _safe_datetime(self, value: Any) -> Optional[datetime.datetime]:
        """将输入值安全转换为时区感知的 datetime 对象 (假设默认时区为上海)"""
        if value is None:
            return None
        try:
            # 尝试解析为 datetime
            if isinstance(value, datetime.datetime):
                dt_obj = value
            elif isinstance(value, datetime.date):
                 # 如果是 date 对象，假设时间是午夜 00:00
                 dt_obj = datetime.datetime.combine(value, datetime.time(0,0))
            else:
                dt_obj = pd.to_datetime(value)

            # 如果是 Naive datetime，标记为默认时区
            if dt_obj.tzinfo is None:
                dt_obj = timezone.make_aware(dt_obj, timezone.get_default_timezone())
            else:
                 # 如果已经是 Aware，转换为默认时区
                 dt_obj = dt_obj.astimezone(timezone.get_default_timezone())
            return dt_obj
        except (ValueError, TypeError, Exception) as e: # 捕获更广泛的异常
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 datetime 对象: {e}", exc_info=False)
            return None














