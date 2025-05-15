# dao_manager\tushare_daos\indicator_dao.py
import asyncio
import datetime
import logging
from typing import Any, List, Optional, Set, Union, Dict, Type
import pandas as pd
import pandas_ta as ta
from django.db.models import Max
from django.db import models # 确保导入 models
import numpy as np
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from django.utils import timezone
from dao_manager.base_dao import BaseDAO
from core.constants import TimeLevel, FINTA_OHLCV_MAP # 确保 FINTA_OHLCV_MAP 导入且包含 'vol': 'volume'
from dao_manager.tushare_daos.industry_dao import IndustryDao
from stock_models.industry import ThsIndexDaily, ThsIndexMember # 导入 ThsIndexMember 模型
from stock_models.time_trade import IndexDaily, StockCyqPerf, StockDailyData, StockMinuteData, StockMonthlyData, StockTimeTrade, StockWeeklyData
# 导入资金流向相关模型
from stock_models.fund_flow import FundFlowDaily, FundFlowDailyTHS, FundFlowDailyDC, FundFlowCntTHS, FundFlowIndustryTHS
from utils.cache_get import  StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO

logger = logging.getLogger("dao")

main_indices = ['000300.SH', '000001.SH', '000905.SH'] # 沪深300, 上证指数, 中证500

# 假设 FINTA_OHLCV_MAP 包含必要的列名映射，例如 {'vol': 'volume'}
# 请确保您的 constants.py 文件中 FINTA_OHLCV_MAP 包含了 'vol': 'volume'
# 如果 FINTA_OHLCV_MAP 在 constants.py 中定义，这里无需重复定义

def get_china_a_stock_kline_times(trade_days: list, time_level: str) -> list:
    """
    生成A股应有的K线标准结束时间点，基于实际交易日历。
    Args:
        trade_days: list of datetime.date，实际交易日列表 (naive date)
        time_level: 'd', 'w', 'm', '5', '15', '30', '60'
    Returns:
        list of pd.Timestamp (Asia/Shanghai)，按时间升序排列
    """
    times = []
    default_tz = timezone.get_default_timezone() # 获取默认时区 (Asia/Shanghai)

    if time_level.lower() == 'd':
        for day in trade_days:
            # 日线数据的时间点通常设为当日开始（午夜 00:00:00），并标记为默认时区
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz=default_tz))
    elif time_level.lower() == 'w':
        # 只保留每周最后一个交易日的午夜时间点
        week_map = {}
        for day in trade_days:
            # 使用 ISO 年份和周次作为 key
            week = pd.Timestamp(day).isocalendar()[1]
            year = pd.Timestamp(day).year
            key = (year, week)
            if key not in week_map or day > week_map[key]:
                week_map[key] = day
        # 按照日期排序，转换为时区感知的 Timestamp
        for day in sorted(week_map.values()):
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz=default_tz))
    elif time_level.lower() == 'm':
        # 只保留每月最后一个交易日的午夜时间点
        month_map = {}
        for day in trade_days:
            month = pd.Timestamp(day).month
            year = pd.Timestamp(day).year
            key = (year, month)
            if key not in month_map or day > month_map[key]:
                month_map[key] = day
        # 按照日期排序，转换为时区感知的 Timestamp
        for day in sorted(month_map.values()):
            times.append(pd.Timestamp(datetime.datetime.combine(day, datetime.time(0, 0)), tz=default_tz))
    elif time_level in ['5', '15', '30', '60']:
        freq = int(time_level)
        times = [] # 清空 times 列表，重新生成分钟线时间点
        for day in trade_days:
             # A股上午交易时间段 K线标准结束时间点
             # 5min: 9:35, 9:40, ..., 11:30
             # 15min: 9:45, 10:00, ..., 11:30
             # 30min: 10:00, 10:30, ..., 11:30
             # 60min: 10:30, 11:30
             # 根据频率确定上午开始时间
             if freq == 5:
                 morning_start_str = '09:35:00'
             elif freq == 15:
                 morning_start_str = '09:45:00'
             elif freq == 30:
                 morning_start_str = '10:00:00'
             elif freq == 60:
                 morning_start_str = '10:30:00'
             else: # Should not happen based on outer if
                 continue # or raise error

             # A股下午交易时间段 K线标准结束时间点
             # 5min: 13:05, 13:10, ..., 15:00
             # 15min: 13:15, 13:30, ..., 15:00
             # 30min: 13:30, 14:00, ..., 15:00
             # 60min: 14:00, 15:00
              # 根据频率确定下午开始时间
             if freq == 5:
                 afternoon_start_str = '13:05:00'
             elif freq == 15:
                 afternoon_start_str = '13:15:00'
             elif freq == 30:
                 afternoon_start_str = '13:30:00'
             elif freq == 60:
                 afternoon_start_str = '14:00:00'
             else: # Should not happen
                 continue # or raise error

             morning_end_str = '11:30:00'
             afternoon_end_str = '15:00:00'

             try:
                # 生成上午时间序列
                morning_times = pd.date_range(start=f'{day} {morning_start_str}', end=f'{day} {morning_end_str}', freq=f'{freq}T', tz=default_tz)
                # 生成下午时间序列
                afternoon_times = pd.date_range(start=f'{day} {afternoon_start_str}', end=f'{day} {afternoon_end_str}', freq=f'{freq}T', tz=default_tz)
                times.extend(morning_times)
                times.extend(afternoon_times)
             except Exception as e:
                  logger.error(f"生成 {day} 的 {freq} 分钟K线标准时间点出错: {e}", exc_info=True)
                  continue # 跳过当前日期

    else:
        raise ValueError(f"不支持的K线类型: {time_level}")
    # 确保时间点是唯一的并排序
    # pandas.date_range 应该已经返回有序的 Timestamp
    # 如果有跨天的数据需要 extend，sorted 是必要的
    return sorted(list(set(times))) # 使用 set 去重并排序


class IndicatorDAO(BaseDAO):
    """
    指标数据访问对象，负责指标数据的读取和存储
    """
    def __init__(self):
         # 依赖注入基础DAO和缓存工具
        from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
        self.stock_basic_dao = StockBasicInfoDao()
        self.industry_dao = IndustryDao()
        self.index_basic_dao = IndexBasicDAO()  # 添加 IndexBasicDAO 的初始化
        self.cache_manager = None # 缓存管理器
        self.cache_get = None # 缓存获取工具
        self.ta = ta

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
                         if time_level_str.lower() == 'd':
                             ModelClass = StockDailyData
                         elif time_level_str.lower() == 'w':
                             ModelClass = StockWeeklyData
                         elif time_level_str.lower() == 'm':
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
                         logger.error(f"转换缓存字典为 Model 实例时出错 ({type(e_conv).__name__}) for {stock_code} {time_level_str}. Sample Dict Keys: {sample_keys}", exc_info=True)

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
            if time_level_str.lower() == "d":
                data_qs = StockDailyData.objects.filter(
                    stock=stock,
                ).order_by('-trade_time')[:limit]
            elif time_level_str.lower() == "w":
                data_qs = StockWeeklyData.objects.filter(
                    stock=stock,
                ).order_by('-trade_time')[:limit]
            elif time_level_str.lower() == "m":
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
            # 1. 获取实际有的数据时间点，并转换为时区感知的 datetime 对象
            trade_times_aware = []
            for trade in data_list:
                 t_raw = getattr(trade, 'trade_time', None)
                 if t_raw:
                     safe_dt = self._safe_datetime(t_raw)
                     if safe_dt:
                         trade_times_aware.append(safe_dt)
                     else:
                         logger.warning(f"从数据库获取的数据中发现无效或无法转换的 trade_time: {t_raw}")

            trade_times_aware.sort() # 确保时间排序

            # 2. 记录实际数据时间范围
            min_time: Optional[datetime.datetime] = None
            max_time: Optional[datetime.datetime] = None
            if trade_times_aware:
                min_time = trade_times_aware[0]
                max_time = trade_times_aware[-1]
                # 对于分钟线，实际范围是 trade_times_aware 中的最早和最晚时间
                # 对于日/周/月线，实际范围是 trade_times_aware (日期) 中的最早和最晚日期
                # logger.info(f"实际数据时间范围: {min_time} 至 {max_time}，数据量: {len(data_list)} 条，股票: {stock_code} {time_level_str}")
            else:
                logger.warning(f"无法确定实际数据时间范围，从数据库获取的 trade_times_aware 列表为空，股票: {stock_code} {time_level_str}")
                # 如果没有有效时间点，数据无效
                return None

            # 3. 获取应有的交易日，基于实际数据日期范围
            index_basic_dao = IndexBasicDAO()
            # 使用实际获取数据的日期范围来确定交易日历
            start_date_str = min_time.strftime('%Y%m%d') if min_time else (timezone.now() - datetime.timedelta(days=365)).strftime('%Y%m%d')
            end_date_str = max_time.strftime('%Y%m%d') if max_time else timezone.now().strftime('%Y%m%d')

            trade_days = await index_basic_dao.get_trade_cal_open(start_date_str, end_date_str)
            trade_days_date = [pd.to_datetime(day).date() for day in trade_days]  # 转为date对象

            # 4. 生成应有的K线标准结束时间点（基于实际交易日）
            # 注意：这里生成的仍然是标准的 K 线结束时间点，可能与实际数据的时间戳不匹配
            expected_times = get_china_a_stock_kline_times(trade_days_date, time_level_str)
            # expected_times 函数已经会生成在交易日内的标准时间点 (pd.Timestamp, Asia/Shanghai)

            # 5. 进一步过滤应有的时间点，使其落在实际获取数据的最小到最大时间范围内
            # 使用 Pandas Timestamp 转换为 datetime.datetime 对象进行比较，确保类型一致
            if min_time and max_time: # 确保 min_time 和 max_time 有效
                 # 将 min_time 和 max_time 转换为 pd.Timestamp 以便与 expected_times 中的元素 (pd.Timestamp) 比较
                 min_ts = pd.Timestamp(min_time)
                 max_ts = pd.Timestamp(max_time)
                 # 使用转换后的 pd.Timestamp 进行范围过滤
                 expected_times_filtered = [t for t in expected_times if min_ts <= t <= max_ts]

                 # 记录调整后的预期时间点数量
                 if expected_times_filtered:
                     logger.info(f"预期时间点范围调整为: {expected_times_filtered[0]} 至 {expected_times_filtered[-1]}，调整后预期时间点数量: {len(expected_times_filtered)}，股票: {stock_code} {time_level_str}")
                 else:
                     # 如果过滤后为空，说明获取到的数据的时间范围内没有任何标准的 K 线结束时间点
                     logger.warning(f"在实际数据时间范围 {min_time} 至 {max_time} 内没有找到预期的标准K线结束时间点，股票: {stock_code} {time_level_str}")
                     expected_times = [] # 将预期时间点列表设为空，后续缺失检查将跳过

                 expected_times = expected_times_filtered # 使用过滤后的预期时间点列表

            # 6. 检查缺失比例 (基于时间点集合差异，只记录警告)
            if not expected_times:
                logger.warning(f"无法生成任何预期时间点，跳过缺失检查。股票: {stock_code} {time_level_str}")
                # 即使无法检查缺失，如果 data_list 有数据，仍然返回
                return data_list

            # 将实际获取的时间点转换为 Pandas DatetimeIndex
            actual_times_index = pd.DatetimeIndex(trade_times_aware)
            expected_times_index = pd.DatetimeIndex(expected_times)

            # 找到缺失的时间点
            # 直接比较 DatetimeIndex 的集合差异，这将反映实际数据时间戳与标准时间戳的匹配程度
            missing_index = expected_times_index.difference(actual_times_index)

            missing_count = len(missing_index)
            expected_count = len(expected_times_index)
            missing_ratio = missing_count / expected_count if expected_count else 0

            # if missing_count > 0:
            #     # 打印缺失警告，包括数量、比例和部分缺失时间点
            #     # 将缺失时间转换为字符串列表以便日志打印，最多打印前 10 个
            #     logger.warning(f"原始K线数据时间序列有缺失: {stock_code} {time_level_str}，缺失数量: {missing_count}，缺失比例: {missing_ratio:.2%}，缺失时间 (部分): {[str(t) for t in missing_index[:10]]} ...")

            #     # --- 移除之前基于高阈值拒绝返回数据的逻辑 (已在上一次修改中移除) ---
            #     # 保留警告，但不在这里拒绝返回数据

            # else:
            #     # 如果没有缺失，记录信息
            #     logger.info(f"原始K线数据时间序列无缺失: {stock_code} {time_level_str}")

            # 无论缺失多少，只要数据库查询有数据，就返回数据列表
            # 数据质量的最终判断和处理应由调用方 (Service 层) 负责
            return data_list

        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None

    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000) -> Optional[pd.DataFrame]:
        """
        获取历史数据并转换为 pandas_ta 需要的 DataFrame 格式。
        返回的 DataFrame 按时间升序排列，并包含必要的 OHLCV 列（'open', 'high', 'low', 'close', 'volume'）
        以及可能的 'amount' 等列。
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
            # 注意：这里的键名应该与数据库模型字段名或缓存中存储的键名一致
            # 并在构建 DataFrame 后再统一转换为小写
            fields_to_include = {
                'trade_time': lambda x: x, # 时间字段保留原始类型，后续 pd.to_datetime 处理
                'open': lambda x: self._safe_float(x), # 使用安全函数转换为 float
                'high': lambda x: self._safe_float(x),
                'low': lambda x: self._safe_float(x),
                'close': lambda x: self._safe_float(x),
                'vol': lambda x: self._safe_int(x), # 使用安全函数转换为 int (对应模型字段名)
                'amount': lambda x: self._safe_float(x),
                # 根据您的模型，可能还需要 'turnover_rate' 等字段
                # 'turnover_rate': lambda x: self._safe_float(x),
                # 'pct_chg': lambda x: self._safe_float(x),
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
                    except Exception as e: # 捕获更广泛的异常
                        # 如果转换失败，记录警告并使用默认的 NaN/0
                        # 记录警告时，仅打印值和类型，避免日志过长
                        value_str = str(value)[:100] # 截断值字符串
                        logger.warning(f"转换字段 '{field_name}' 值 '{value_str}' (类型: {type(value).__name__}) 失败 for {stock_code} {time_level_val}: {e}", exc_info=True)
                        # 根据字段名设置默认值
                        if 'vol' in field_name: # 成交量默认 0
                            row_data[field_name] = 0
                        elif 'time' in field_name: # 时间默认 NaT
                            row_data[field_name] = pd.NaT
                        else: # 其他数值字段默认 NaN
                            row_data[field_name] = np.nan
                data.append(row_data)
            if not data:
                logger.warning(f"从 Model 实例转换的数据列表为空: {stock_code} {time_level_val}")
                return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 DataFrame 为空: {stock_code} {time_level_val}")
                return None
            # 4. 列名标准化：先转为小写，再根据需要映射为标准名称
            df.columns = [col.lower() for col in df.columns]
            # 将非标准列名映射为标准名称，例如将 'vol' 映射为 'volume'
            # 确保 FINTA_OHLCV_MAP 包含了 {'vol': 'volume'} 映射
            rename_map = {k.lower(): v.lower() for k, v in FINTA_OHLCV_MAP.items()} # 将映射表的键和值都转为小写
            # 只重命名那些存在于 df.columns 中且在 rename_map 的键中，并且映射后名称不同的列
            actual_rename_map = {
                col: rename_map[col]
                for col in df.columns
                if col in rename_map and col != rename_map[col]
            }
            if actual_rename_map:
                df.rename(columns=actual_rename_map, inplace=True)
                logger.debug(f"对列名进行重命名: {actual_rename_map} for {stock_code} {time_level_val}")
            # 5. 时间列转换为 DatetimeIndex 并设为索引
            # 确保 'trade_time' 列存在
            if 'trade_time' not in df.columns:
                logger.error(f"转换后的 DataFrame 缺少 'trade_time' 列: {stock_code} {time_level_val}")
                return None
            # 强制转换为 datetime，处理错误并丢弃无效行
            # 使用 utc=True 标记为 UTC 时间，errors='coerce' 将无效解析转为 NaT
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True, errors='coerce')
            df.dropna(subset=['trade_time'], inplace=True) # 丢弃 trade_time 为 NaT 的行
            if df.empty:
                logger.warning(f"处理无效 trade_time 后 DataFrame 为空: {stock_code} {time_level_val}")
                return None
            # 转换为默认时区 (假设是上海)
            default_tz = timezone.get_default_timezone()
            # pd.to_datetime(..., utc=True) 已经返回时区感知的 UTC Timestamp
            # 直接将其转换为默认时区
            df.index = df['trade_time'].dt.tz_convert(default_tz)
            df.drop(columns=['trade_time'], inplace=True) # 移除 trade_time 列
            # 6. 去重索引，保留最后一个（最新数据）
            if df.index.has_duplicates:
                # 记录去重前的数量
                original_rows = len(df)
                df = df[~df.index.duplicated(keep='last')]
                if len(df) < original_rows: # 仅在实际去重时才打印
                     logger.warning(f"移除重复索引，去重前: {original_rows}，去重后: {len(df)}，股票: {stock_code} {time_level_val}")
            # 7. 按时间升序排序索引
            df.sort_index(ascending=True, inplace=True)
            # --- 记录应用重命名和排序后的最终列名 ---
            logger.debug(f"转换并重命名后的 DataFrame 列名: {df.columns.tolist()} for {stock_code} {time_level_val}")
            # 8. 校验必要列是否存在
            required_cols = ['open', 'high', 'low', 'close', 'volume'] # 注意这里用的是小写标准列名
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                logger.error(f"DataFrame 缺少必要列: {stock_code} {time_level_val}. 缺失: {missing}, 实际: {df.columns.tolist()}")
                return None # 必要列缺失是严重问题，拒绝返回 DataFrame
            # 9. 检查必要列是否全部是 NaN (检查 DataFrame 是否包含有效数据)
            # df[required_cols].isnull().all(axis=1) 会得到一个布尔 Series，表示每一行在 required_cols 中是否全为 NaN
            # .all() 检查这个 Series 是否全部为 True (即所有行在必要列中都全为 NaN)
            # 如果必要列中任何一列的**所有值**都为 NaN，或者所有行的必要列值都为 NaN，则认为数据无效
            if df[required_cols].isnull().all().any() or df[required_cols].isnull().all(axis=1).all():
                 logger.warning(f"处理后 DataFrame 的必要列全部包含 NaN 值 或 某些必要列全为NaN: {stock_code} {time_level_val}")
                 # 如果必要列数据全为 NaN，这个 DataFrame 是无效的，拒绝返回
                 return None
            # 10. 检查必要列的缺失比例（这个检查更合理，可以保留）
            # 计算必要列的平均缺失比例
            missing_ratio_required = df[required_cols].isnull().mean().mean()
            missing_threshold_df = 0.1  # DataFrame 层面必要列平均缺失比例阈值，10%
            if missing_ratio_required > missing_threshold_df:
                 # 仅记录警告，不拒绝返回
                 # 打印各必要列的缺失比例详情
                 missing_details = df[required_cols].isnull().mean().apply(lambda x: f'{x:.2%}')
                 logger.warning(f"处理后 DataFrame 必要列平均缺失比例 {missing_ratio_required:.2%} 超过阈值 {missing_threshold_df}，可能影响后续计算: {stock_code} {time_level_val}. 必要列缺失比例详情: {missing_details.to_dict()}")
            # logger.info(f"返回 DataFrame，必要列平均缺失比例: {missing_ratio_required:.2%}，数据量: {len(df)} 条: {stock_code} {time_level_val}")
            return df
        except Exception as e:
            logger.error(f"转换 {stock_code} {time_level_val} 历史数据为 DataFrame 失败: {str(e)}", exc_info=True)
            return None

    # 新增方法：获取指数日线数据并转为 DataFrame
    async def get_index_daily_df(self, index_codes: List[str], start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定指数列表在日期范围内（包含起止日）的日线数据，并转换为 DataFrame。
        返回 DataFrame 包含 'index_code', 'trade_time', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_chg', 'vol', 'amount' 等列。
        DataFrame 的索引是时区感知的 pd.Timestamp。
        """
        if not index_codes:
            return None
        try:
            # 使用 filter(index__index_code__in=index_codes) 批量查询
            # 注意 IndexDaily 模型的外键关联 IndexInfo 的 index_code 字段
            # 用 sync_to_async 包裹整个 ORM 查询过程，保证异步安全
            data_list = await sync_to_async(
                lambda: list(
                    IndexDaily.objects.filter(
                        index__index_code__in=index_codes,
                        trade_time__gte=start_date,
                        trade_time__lte=end_date
                    ).select_related('index')
                )
            )()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到指数 {index_codes} 的日线数据")
                return None
            # 将模型实例列表转换为字典列表
            data = []
            for item in data_list:
                data.append({
                    'index_code': getattr(item.index, 'index_code', None), # 获取关联 IndexInfo 的 index_code
                    'trade_time': getattr(item, 'trade_time', None),
                    'open': self._safe_float(getattr(item, 'open', None)),
                    'high': self._safe_float(getattr(item, 'high', None)),
                    'low': self._safe_float(getattr(item, 'low', None)),
                    'close': self._safe_float(getattr(item, 'close', None)),
                    'pre_close': self._safe_float(getattr(item, 'pre_close', None)),
                    'change': self._safe_float(getattr(item, 'change', None)),
                    'pct_chg': self._safe_float(getattr(item, 'pct_chg', None)),
                    'vol': self._safe_float(getattr(item, 'vol', None)), # 指数 vol 是浮点型
                    'amount': self._safe_float(getattr(item, 'amount', None)),
                    # 添加其他需要的字段
                })
            if not data:
                 logger.warning(f"从 IndexDaily Model 转换的数据列表为空 for {index_codes}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 IndexDaily DataFrame 为空 for {index_codes}")
                return None
            # 处理 trade_time 列并设置为索引 (日线数据的时间点视为默认时区下的日期开始)
            default_tz = timezone.get_default_timezone()
            # 日线 trade_time 是 date 对象，转为 datetime.datetime 再标记时区
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True) # 丢弃无效时间行
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 IndexDaily DataFrame 为空 for {index_codes}")
                 return None
            df.index = df['trade_time']
            df.drop(columns=['trade_time'], inplace=True)
            # 按时间升序排序索引
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"成功获取并处理指数 {index_codes} 的日线数据，数据量: {len(df)} 条")
            return df

        except Exception as e:
            logger.error(f"获取指数日线数据失败 for {index_codes} 在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增方法：获取同花顺板块日线数据并转为 DataFrame
    async def get_ths_index_daily_df(self, ths_codes: List[str], start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定同花顺指数列表在日期范围内（包含起止日）的日线数据，并转换为 DataFrame。
        返回 DataFrame 包含 'ts_code', 'trade_time', 'close', 'open', 'high', 'low', 'pre_close', 'change', 'pct_change', 'vol', 'turnover_rate', 'total_mv', 'float_mv', 'pe_ttm', 'pb_mrq' 等列。
        DataFrame 的索引是时区感知的 pd.Timestamp。
        """
        if not ths_codes:
            return None
        try:
            # 使用 filter(ths_index__ts_code__in=ths_codes) 批量查询
            # 注意 ThsIndexDaily 模型的外键关联 ThsIndex 的 ts_code 字段
            data_list = await sync_to_async(
                lambda: list(
                    ThsIndexDaily.objects.filter(
                        ths_index__ts_code__in=ths_codes,
                        trade_time__gte=start_date,
                        trade_time__lte=end_date
                    ).select_related('ths_index')
                )
            )()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到同花顺指数 {ths_codes} 的日线数据")
                return None
            # 将模型实例列表转换为字典列表
            data = []
            for item in data_list:
                 data.append({
                    'ts_code': getattr(item.ths_index, 'ts_code', None), # 获取关联 ThsIndex 的 ts_code
                    'trade_time': getattr(item, 'trade_time', None),
                    'open': self._safe_float(getattr(item, 'open', None)),
                    'high': self._safe_float(getattr(item, 'high', None)),
                    'low': self._safe_float(getattr(item, 'low', None)),
                    'close': self._safe_float(getattr(item, 'close', None)),
                    'pre_close': self._safe_float(getattr(item, 'pre_close', None)),
                    'change': self._safe_float(getattr(item, 'change', None)),
                    'pct_change': self._safe_float(getattr(item, 'pct_change', None)), # 注意字段名不同于 IndexDaily
                    'vol': self._safe_float(getattr(item, 'vol', None)),
                    'turnover_rate': self._safe_float(getattr(item, 'turnover_rate', None)),
                    'total_mv': self._safe_float(getattr(item, 'total_mv', None)),
                    'float_mv': self._safe_float(getattr(item, 'float_mv', None)),
                    'pe_ttm': self._safe_float(getattr(item, 'pe_ttm', None)),
                    'pb_mrq': self._safe_float(getattr(item, 'pb_mrq', None)),
                     # 添加其他需要的字段
                })
            if not data:
                 logger.warning(f"从 ThsIndexDaily Model 转换的数据列表为空 for {ths_codes}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 ThsIndexDaily DataFrame 为空 for {ths_codes}")
                return None
            # 处理 trade_time 列并设置为索引 (日线数据的时间点视为默认时区下的日期开始)
            default_tz = timezone.get_default_timezone()
            # 日线 trade_time 是 date 对象，转为 datetime.datetime 再标记时区
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True) # 丢弃无效时间行
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 ThsIndexDaily DataFrame 为空 for {ths_codes}")
                 return None
            df.index = df['trade_time']
            df.drop(columns=['trade_time'], inplace=True)
            # 按时间升序排序索引
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"成功获取并处理同花顺指数 {ths_codes} 的日线数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取同花顺指数日线数据失败 for {ths_codes} 在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增方法：获取股票筹码分布汇总数据并转为 DataFrame
    async def get_stock_cyq_perf_df(self, stock_code: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定股票在日期范围内（包含起止日）的筹码分布汇总数据，并转换为 DataFrame。
        返回 DataFrame 包含 'stock_code', 'trade_time', 'his_low', 'his_high', 'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct', 'weight_avg', 'winner_rate' 等列。
        DataFrame 的索引是时区感知的 pd.Timestamp。
        """
        try:
            # 获取股票对象 (假设 get_stock_by_code 返回的是 StockInfo 模型实例)
            stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
            if not stock:
                 logger.warning(f"无法找到股票信息: {stock_code}，无法获取筹码数据")
                 return None
            # 用 sync_to_async 包裹 ORM 查询
            data_list = await sync_to_async(
                lambda: list(
                    StockCyqPerf.objects.filter( stock=stock, trade_time__gte=start_date, trade_time__lte=end_date ).select_related('stock')
                )
            )()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到股票 {stock_code} 的筹码分布汇总数据")
                return None
            # 将模型实例列表转换为字典列表
            data = []
            for item in data_list:
                 data.append({
                    'stock_code': getattr(item.stock, 'stock_code', None), # 获取关联 StockInfo 的 stock_code
                    'trade_time': getattr(item, 'trade_time', None),
                    'his_low': self._safe_float(getattr(item, 'his_low', None)),
                    'his_high': self._safe_float(getattr(item, 'his_high', None)),
                    'cost_5pct': self._safe_float(getattr(item, 'cost_5pct', None)),
                    'cost_15pct': self._safe_float(getattr(item, 'cost_15pct', None)),
                    'cost_50pct': self._safe_float(getattr(item, 'cost_50pct', None)),
                    'cost_85pct': self._safe_float(getattr(item, 'cost_85pct', None)),
                    'cost_95pct': self._safe_float(getattr(item, 'cost_95pct', None)),
                    'weight_avg': self._safe_float(getattr(item, 'weight_avg', None)),
                    'winner_rate': self._safe_float(getattr(item, 'winner_rate', None)),
                    # 添加其他需要的字段
                 })
            if not data:
                 logger.warning(f"从 StockCyqPerf Model 转换的数据列表为空 for {stock_code}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 StockCyqPerf DataFrame 为空 for {stock_code}")
                return None
            # 处理 trade_time 列并设置为索引 (日线数据的时间点视为默认时区下的日期开始)
            default_tz = timezone.get_default_timezone()
            # 日线 trade_time 是 date 对象，转为 datetime.datetime 再标记时区
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True) # 丢弃无效时间行
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 StockCyqPerf DataFrame 为空 for {stock_code}")
                 return None
            df.index = df['trade_time']
            df.drop(columns=['trade_time'], inplace=True)
            # 按时间升序排序索引
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"成功获取并处理股票 {stock_code} 的筹码分布汇总数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 筹码分布汇总数据失败在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增获取 FundFlowDaily 数据的异步方法
    async def get_fund_flow_daily_df(self, stock_code: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定股票在日期范围内（包含起止日）的日级资金流向数据，并转换为 DataFrame。
        """
        try:
            # 使用 filter(stock__stock_code=stock_code) 过滤股票
            data_list = await sync_to_async(
                lambda: list(
                    FundFlowDaily.objects.filter(
                        stock__stock_code=stock_code,
                        trade_time__gte=start_date,
                        trade_time__lte=end_date
                    ).select_related('stock')
                )
            )()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到股票 {stock_code} 的日级资金流向数据")
                return None
            data = []
            for item in data_list:
                data.append({
                    'trade_time': getattr(item, 'trade_time', None),
                    'ff_daily_buy_sm_vol': self._safe_int(getattr(item, 'buy_sm_vol', None)), # 添加前缀
                    'ff_daily_buy_sm_amount': self._safe_float(getattr(item, 'buy_sm_amount', None)), # 添加前缀
                    'ff_daily_sell_sm_vol': self._safe_int(getattr(item, 'sell_sm_vol', None)), # 添加前缀
                    'ff_daily_sell_sm_amount': self._safe_float(getattr(item, 'sell_sm_amount', None)), # 添加前缀
                    'ff_daily_buy_md_vol': self._safe_int(getattr(item, 'buy_md_vol', None)), # 添加前缀
                    'ff_daily_buy_md_amount': self._safe_float(getattr(item, 'buy_md_amount', None)), # 添加前缀
                    'ff_daily_sell_md_vol': self._safe_int(getattr(item, 'sell_md_vol', None)), # 添加前缀
                    'ff_daily_sell_md_amount': self._safe_float(getattr(item, 'sell_md_amount', None)), # 添加前缀
                    'ff_daily_buy_lg_vol': self._safe_int(getattr(item, 'buy_lg_vol', None)), # 添加前缀
                    'ff_daily_buy_lg_amount': self._safe_float(getattr(item, 'buy_lg_amount', None)), # 添加前缀
                    'ff_daily_sell_lg_vol': self._safe_int(getattr(item, 'sell_lg_vol', None)), # 添加前缀
                    'ff_daily_sell_lg_amount': self._safe_float(getattr(item, 'sell_lg_amount', None)), # 添加前缀
                    'ff_daily_buy_elg_vol': self._safe_int(getattr(item, 'buy_elg_vol', None)), # 添加前缀
                    'ff_daily_buy_elg_amount': self._safe_float(getattr(item, 'buy_elg_amount', None)), # 添加前缀
                    'ff_daily_sell_elg_vol': self._safe_int(getattr(item, 'sell_elg_vol', None)), # 添加前缀
                    'ff_daily_sell_elg_amount': self._safe_float(getattr(item, 'sell_elg_amount', None)), # 添加前缀
                    'ff_daily_net_mf_vol': self._safe_int(getattr(item, 'net_mf_vol', None)), # 添加前缀
                    'ff_daily_net_mf_amount': self._safe_float(getattr(item, 'net_mf_amount', None)), # 添加前缀
                })
            if not data:
                 logger.warning(f"从 FundFlowDaily Model 转换的数据列表为空 for {stock_code}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 FundFlowDaily DataFrame 为空 for {stock_code}")
                return None
            default_tz = timezone.get_default_timezone()
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 FundFlowDaily DataFrame 为空 for {stock_code}")
                 return None
            df.index = df['trade_time']
            df.drop(columns=['trade_time'], inplace=True)
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"成功获取并处理股票 {stock_code} 的日级资金流向数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 日级资金流向数据失败在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增获取 FundFlowDailyTHS 数据的异步方法
    async def get_fund_flow_daily_ths_df(self, stock_code: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定股票在日期范围内（包含起止日）的同花顺日级资金流向数据，并转换为 DataFrame。
        """
        try:
            data_list = await sync_to_async(
                lambda: list(
                    FundFlowDailyTHS.objects.filter(
                        stock__stock_code=stock_code,
                        trade_time__gte=start_date,
                        trade_time__lte=end_date
                    ).select_related('stock')
                )
            )()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到股票 {stock_code} 的同花顺日级资金流向数据")
                return None
            data = []
            for item in data_list:
                data.append({
                    'trade_time': getattr(item, 'trade_time', None),
                    'ff_ths_pct_change': self._safe_float(getattr(item, 'pct_change', None)), # 添加前缀
                    'ff_ths_net_amount': self._safe_float(getattr(item, 'net_amount', None)), # 添加前缀
                    'ff_ths_net_d5_amount': self._safe_float(getattr(item, 'net_d5_amount', None)), # 添加前缀
                    'ff_ths_buy_lg_amount': self._safe_float(getattr(item, 'buy_lg_amount', None)), # 添加前缀
                    'ff_ths_buy_lg_amount_rate': self._safe_float(getattr(item, 'buy_lg_amount_rate', None)), # 添加前缀
                    'ff_ths_buy_md_amount': self._safe_float(getattr(item, 'buy_md_amount', None)), # 添加前缀
                    'ff_ths_buy_md_amount_rate': self._safe_float(getattr(item, 'buy_md_amount_rate', None)), # 添加前缀
                    'ff_ths_buy_sm_amount': self._safe_float(getattr(item, 'buy_sm_amount', None)), # 添加前缀
                    'ff_ths_buy_sm_amount_rate': self._safe_float(getattr(item, 'buy_sm_amount_rate', None)), # 添加前缀
                })
            if not data:
                 logger.warning(f"从 FundFlowDailyTHS Model 转换的数据列表为空 for {stock_code}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 FundFlowDailyTHS DataFrame 为空 for {stock_code}")
                return None
            default_tz = timezone.get_default_timezone()
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 FundFlowDailyTHS DataFrame 为空 for {stock_code}")
                 return None
            df.index = df['trade_time']
            df.drop(columns=['trade_time'], inplace=True)
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"成功获取并处理股票 {stock_code} 的同花顺日级资金流向数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 同花顺日级资金流向数据失败在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增获取 FundFlowDailyDC 数据的异步方法
    async def get_fund_flow_daily_dc_df(self, stock_code: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定股票在日期范围内（包含起止日）的东方财富日级资金流向数据，并转换为 DataFrame。
        """
        try:
            data_list = await sync_to_async(lambda: list(
                FundFlowDailyDC.objects.filter(
                    stock__stock_code=stock_code,
                    trade_time__gte=start_date,
                    trade_time__lte=end_date
                ).select_related('stock') # 优化查询
            ))()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到股票 {stock_code} 的东方财富日级资金流向数据")
                return None
            data = []
            for item in data_list:
                data.append({
                    'trade_time': getattr(item, 'trade_time', None),
                    # 'name': getattr(item, 'name', None), # 名称字段通常不需要合并到数值特征中
                    'ff_dc_pct_change': self._safe_float(getattr(item, 'pct_change', None)), # 添加前缀
                    'ff_dc_close': self._safe_float(getattr(item, 'close', None)), # 添加前缀
                    'ff_dc_net_amount': self._safe_float(getattr(item, 'net_amount', None)), # 添加前缀
                    'ff_dc_net_amount_rate': self._safe_float(getattr(item, 'net_amount_rate', None)), # 添加前缀
                    'ff_dc_buy_elg_amount': self._safe_float(getattr(item, 'buy_elg_amount', None)), # 添加前缀
                    'ff_dc_buy_elg_amount_rate': self._safe_float(getattr(item, 'buy_elg_amount_rate', None)), # 添加前缀
                    'ff_dc_buy_lg_amount': self._safe_float(getattr(item, 'buy_lg_amount', None)), # 添加前缀
                    'ff_dc_buy_lg_amount_rate': self._safe_float(getattr(item, 'buy_lg_amount_rate', None)), # 添加前缀
                    'ff_dc_buy_md_amount': self._safe_float(getattr(item, 'buy_md_amount', None)), # 添加前缀
                    'ff_dc_buy_md_amount_rate': self._safe_float(getattr(item, 'buy_md_amount_rate', None)), # 添加前缀
                    'ff_dc_buy_sm_amount': self._safe_float(getattr(item, 'buy_sm_amount', None)), # 添加前缀
                    'ff_dc_buy_sm_amount_rate': self._safe_float(getattr(item, 'buy_sm_amount_rate', None)), # 添加前缀
                })
            if not data:
                 logger.warning(f"从 FundFlowDailyDC Model 转换的数据列表为空 for {stock_code}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 FundFlowDailyDC DataFrame 为空 for {stock_code}")
                return None
            default_tz = timezone.get_default_timezone()
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 FundFlowDailyDC DataFrame 为空 for {stock_code}")
                 return None
            df.index = df['trade_time']
            df.drop(columns=['trade_time'], inplace=True)
            df.sort_index(ascending=True, inplace=True)
            logger.info(f"成功获取并处理股票 {stock_code} 的东方财富日级资金流向数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取股票 {stock_code} 东方财富日级资金流向数据失败在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增获取 FundFlowCntTHS 数据的异步方法
    async def get_fund_flow_cnt_ths_df(self, ths_codes: List[str], start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定同花顺板块列表在日期范围内（包含起止日）的资金流向统计数据，并转换为 DataFrame。
        """
        if not ths_codes:
            return None
        try:
            data_list = await sync_to_async(lambda: list(
                FundFlowCntTHS.objects.filter(
                    ths_index__ts_code__in=ths_codes,
                    trade_time__gte=start_date,
                    trade_time__lte=end_date
                ).select_related('ths_index') # 优化查询
            ))()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到同花顺板块 {ths_codes} 的资金流向统计数据")
                return None
            data = []
            for item in data_list:
                data.append({
                    'ts_code': getattr(item.ths_index, 'ts_code', None), # 保留板块代码用于后续处理
                    'trade_time': getattr(item, 'trade_time', None),
                    # 'lead_stock': getattr(item, 'lead_stock', None), # 领涨股名称通常不需要合并
                    'ff_cnt_ths_close_price': self._safe_float(getattr(item, 'close_price', None)), # 添加前缀
                    'ff_cnt_ths_pct_change': self._safe_float(getattr(item, 'pct_change', None)), # 添加前缀
                    'ff_cnt_ths_industry_index': self._safe_float(getattr(item, 'industry_index', None)), # 添加前缀
                    'ff_cnt_ths_company_num': self._safe_int(getattr(item, 'company_num', None)), # 添加前缀
                    'ff_cnt_ths_pct_change_stock': self._safe_float(getattr(item, 'pct_change_stock', None)), # 添加前缀
                    'ff_cnt_ths_net_buy_amount': self._safe_float(getattr(item, 'net_buy_amount', None)), # 添加前缀
                    'ff_cnt_ths_net_sell_amount': self._safe_float(getattr(item, 'net_sell_amount', None)), # 添加前缀
                    'ff_cnt_ths_net_amount': self._safe_float(getattr(item, 'net_amount', None)), # 添加前缀
                })
            if not data:
                 logger.warning(f"从 FundFlowCntTHS Model 转换的数据列表为空 for {ths_codes}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 FundFlowCntTHS DataFrame 为空 for {ths_codes}")
                return None
            default_tz = timezone.get_default_timezone()
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 FundFlowCntTHS DataFrame 为空 for {ths_codes}")
                 return None
            # 不在这里设置索引，保留 ts_code 列用于后续分组和合并
            df.sort_values(by=['ts_code', 'trade_time'], ascending=True, inplace=True)
            logger.info(f"成功获取并处理同花顺板块 {ths_codes} 的资金流向统计数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取同花顺板块资金流向统计数据失败 for {ths_codes} 在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 新增获取 FundFlowIndustryTHS 数据的异步方法
    async def get_fund_flow_industry_ths_df(self, ths_codes: List[str], start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        获取指定同花顺行业列表在日期范围内（包含起止日）的资金流向统计数据，并转换为 DataFrame。
        """
        if not ths_codes:
            return None
        try:
            data_list = await sync_to_async(lambda: list(
                FundFlowIndustryTHS.objects.filter(
                    ths_index__ts_code__in=ths_codes,
                    trade_time__gte=start_date,
                    trade_time__lte=end_date
                ).select_related('ths_index') # 优化查询
            ))()
            if not data_list:
                logger.warning(f"在日期范围 {start_date} 到 {end_date} 未找到同花顺行业 {ths_codes} 的资金流向统计数据")
                return None
            data = []
            for item in data_list:
                data.append({
                    'ts_code': getattr(item.ths_index, 'ts_code', None), # 保留行业代码用于后续处理
                    'trade_time': getattr(item, 'trade_time', None),
                    # 'industry': getattr(item, 'industry', None), # 行业名称通常不需要合并
                    # 'lead_stock': getattr(item, 'lead_stock', None), # 领涨股名称通常不需要合并
                    'ff_ind_ths_close': self._safe_float(getattr(item, 'close', None)), # 添加前缀
                    'ff_ind_ths_pct_change': self._safe_float(getattr(item, 'pct_change', None)), # 添加前缀
                    'ff_ind_ths_company_num': self._safe_int(getattr(item, 'company_num', None)), # 添加前缀
                    'ff_ind_ths_pct_change_stock': self._safe_float(getattr(item, 'pct_change_stock', None)), # 添加前缀
                    'ff_ind_ths_close_price': self._safe_float(getattr(item, 'close_price', None)), # 添加前缀
                    'ff_ind_ths_net_buy_amount': self._safe_float(getattr(item, 'net_buy_amount', None)), # 添加前缀
                    'ff_ind_ths_net_sell_amount': self._safe_float(getattr(item, 'net_sell_amount', None)), # 添加前缀
                    'ff_ind_ths_net_amount': self._safe_float(getattr(item, 'net_amount', None)), # 添加前缀
                })
            if not data:
                 logger.warning(f"从 FundFlowIndustryTHS Model 转换的数据列表为空 for {ths_codes}")
                 return None
            df = pd.DataFrame(data)
            if df.empty:
                logger.warning(f"转换后的 FundFlowIndustryTHS DataFrame 为空 for {ths_codes}")
                return None
            default_tz = timezone.get_default_timezone()
            df['trade_time'] = df['trade_time'].apply(lambda x: timezone.make_aware(datetime.datetime.combine(x, datetime.time(0,0)), default_tz) if isinstance(x, datetime.date) else pd.NaT)
            df.dropna(subset=['trade_time'], inplace=True)
            if df.empty:
                 logger.warning(f"处理无效 trade_time 后 FundFlowIndustryTHS DataFrame 为空 for {ths_codes}")
                 return None
            # 不在这里设置索引，保留 ts_code 列用于后续分组和合并
            df.sort_values(by=['ts_code', 'trade_time'], ascending=True, inplace=True)
            logger.info(f"成功获取并处理同花顺行业 {ths_codes} 的资金流向统计数据，数据量: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取同花顺行业资金流向统计数据失败 for {ths_codes} 在日期范围 {start_date} 到 {end_date}: {str(e)}", exc_info=True)
            return None

    # 添加安全转换辅助函数（确保存在且正确）
    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """将输入值安全转换为 Decimal 类型"""
        if value is None:
            return None
        try:
            # 尝试直接转换 Decimal
            # 避免科学计数法字符串问题，先尝试转为字符串再创建 Decimal
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as e:
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 Decimal: {e}", exc_info=True)
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        """将输入值安全转换为 int 类型"""
        if value is None:
            return None
        try:
            # 确保值是非空的字符串或数字
            if isinstance(value, (str, int, float, Decimal)):
                 # 如果是 Decimal，先转换为 float 再 int
                 if isinstance(value, Decimal):
                      value = float(value)
                 # 尝试直接转换为 int，如果失败则尝试通过 float 转换
                 try:
                     return int(value)
                 except (ValueError, TypeError):
                     return int(float(value))
            else:
                 logger.warning(f"无法将非数字/字符串类型 '{type(value).__name__}' 的值 '{value}' 转换为 int。")
                 return None
        except (ValueError, TypeError) as e:
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 int: {e}", exc_info=True)
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        """将输入值安全转换为 float 类型"""
        if value is None:
            return None
        try:
            # 确保值是非空的字符串或数字
            if isinstance(value, (str, int, float, Decimal)):
                 # 如果是 Decimal，直接转换为 float
                 if isinstance(value, Decimal):
                      return float(value)
                 return float(value)
            else:
                 logger.warning(f"无法将非数字/字符串类型 '{type(value).__name__}' 的值 '{value}' 转换为 float。")
                 return None
        except (ValueError, TypeError) as e:
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 float: {e}", exc_info=True)
            return None

    def _safe_datetime(self, value: Any) -> Optional[datetime.datetime]:
        """
        将输入值安全转换为时区感知的 datetime 对象 (目标时区为默认时区，通常为上海)。
        假定输入的 naive datetime 或字符串代表的是 UTC 时间。
        """
        if value is None:
            return None
        try:
            dt_obj = None
            if isinstance(value, datetime.datetime):
                dt_obj = value
            elif isinstance(value, datetime.date):
                 # 如果是 date 对象，假设是默认时区下的午夜 00:00:00
                 # 日线数据的时间点通常应视为默认时区下的日期
                 dt_obj = timezone.make_aware(datetime.datetime.combine(value, datetime.time(0,0)), timezone.get_default_timezone())
            else:
                # 尝试使用 pd.to_datetime 解析字符串等，并假设原始字符串代表的是 UTC 时间
                # errors='coerce' 会将无法解析的转换为 NaT
                dt_obj = pd.to_datetime(value, utc=True, errors='coerce')

            # 如果解析失败 (NaT) 或原始就是 None/NaT
            if pd.isna(dt_obj):
                raise ValueError("解析为 datetime/Timestamp 失败或结果无效")

            # 确保最终结果是时区感知的 datetime.datetime 对象，并转换为默认时区
            if isinstance(dt_obj, pd.Timestamp):
                 # 如果是 Timestamp，已经是时区感知的 (UTC)，直接转换为默认时区的 datetime.datetime
                 return dt_obj.tz_convert(timezone.get_default_timezone()).to_pydatetime()
            elif isinstance(dt_obj, datetime.datetime):
                 # 如果已经是 datetime.datetime
                 if dt_obj.tzinfo is None:
                     # Naive datetime，根据我们对数据库存储的理解，标记为 UTC 再转换
                     aware_dt = timezone.make_aware(dt_obj, timezone.utc)
                     return aware_dt.astimezone(timezone.get_default_timezone())
                 else:
                     # Already aware datetime，直接转换为默认时区
                     return dt_obj.astimezone(timezone.get_default_timezone())
            else:
                 raise TypeError(f"转换结果不是 datetime 或 Timestamp: {type(dt_obj)}")

        except Exception as e: # 捕获更广泛的异常
            # 记录警告时，仅打印值和类型，避免日志过长
            value_str = str(value)[:100] # 截断值字符串
            logger.warning(f"无法将值 '{value_str}' (类型: {type(value).__name__}) 安全转换为时区感知 datetime 对象: {e}", exc_info=True)
            return None








