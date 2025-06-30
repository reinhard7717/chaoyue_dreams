# dao_manager\tushare_daos\indicator_dao.py
import asyncio
import datetime
import logging
from typing import Any, List, Optional, Set, Union, Dict, Type
import pandas as pd
from django.db.models import F
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
from stock_models.industry import ThsIndex, ThsIndexDaily, ThsIndexMember # 导入 ThsIndexMember 模型
from stock_models.time_trade import IndexDaily, StockCyqPerf, StockDailyBasic, StockDailyData, StockDailyData_BJ, StockDailyData_CY, StockDailyData_KC, StockDailyData_SH, StockDailyData_SZ, StockMinuteData, StockMinuteData_15_BJ, StockMinuteData_15_CY, StockMinuteData_15_KC, StockMinuteData_15_SH, StockMinuteData_15_SZ, StockMinuteData_30_BJ, StockMinuteData_30_CY, StockMinuteData_30_KC, StockMinuteData_30_SH, StockMinuteData_30_SZ, StockMinuteData_5_BJ, StockMinuteData_5_CY, StockMinuteData_5_KC, StockMinuteData_5_SH, StockMinuteData_5_SZ, StockMinuteData_60_BJ, StockMinuteData_60_CY, StockMinuteData_60_KC, StockMinuteData_60_SH, StockMinuteData_60_SZ, StockMonthlyData, StockTimeTrade, StockWeeklyData
# 导入资金流向相关模型
from stock_models.fund_flow import FundFlowDaily, FundFlowDailyTHS, FundFlowDailyDC, FundFlowCntTHS, FundFlowIndustryTHS
from utils.cache_get import  StockTimeTradeCacheGet
from utils.cache_manager import CacheManager
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO

logger = logging.getLogger("dao")

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

    async def get_history_time_trades_by_limit(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000, trade_time: Optional[str] = None) -> Optional[List[StockTimeTrade]]:
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
        # 先将trade_time字符串转换为datetime对象（如果提供）
        start_trade_time = None
        if trade_time:
            try:
                start_trade_time = self._safe_datetime(trade_time)
                if start_trade_time is None:
                    logger.warning(f"传入的trade_time无法解析为有效时间: {trade_time}")
            except Exception as e:
                logger.error(f"解析trade_time失败: {trade_time}, 错误: {e}")
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
                            if stock_code.startswith('3') and stock_code.endswith('.SZ'):
                                ModelClass =  StockDailyData_CY
                            elif stock_code.endswith('.SZ'):
                                ModelClass =  StockDailyData_SZ
                            elif stock_code.startswith('68') and stock_code.endswith('.SH'):
                                ModelClass =  StockDailyData_KC
                            elif stock_code.endswith('.SH'):
                                ModelClass =  StockDailyData_SH
                            elif stock_code.endswith('.BJ'):
                                ModelClass =  StockDailyData_BJ
                            else:
                                logger.warning(f"未识别的股票代码: {stock_code}，默认使用SZ主板日线表") # 修改行: print改为logger.warning
                                ModelClass =  StockDailyData_SZ  # 默认返回深市主板
                        elif time_level_str.lower() == 'w':
                            ModelClass = StockWeeklyData
                        elif time_level_str.lower() == 'm':
                            ModelClass = StockMonthlyData
                        else:
                            # 分钟线模型选择逻辑
                            if time_level_str not in ['5', '15', '30', '60']:
                                ModelClass = StockMinuteData # 1min默认用原表
                            elif stock_code.endswith('.SZ'):
                                if stock_code.startswith('3'):
                                    ModelClass = {
                                        '5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY
                                    }[time_level_str]
                                else:
                                    ModelClass = {
                                        '5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ
                                    }[time_level_str]
                            elif stock_code.endswith('.SH'):
                                if stock_code.startswith('68'):
                                    ModelClass = {
                                        '5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC
                                    }[time_level_str]
                                else:
                                    ModelClass = {
                                        '5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH
                                    }[time_level_str]
                            elif stock_code.endswith('.BJ'):
                                ModelClass = {
                                    '5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ
                                }[time_level_str]
                            else:
                                logger.warning(f"未识别的股票代码或分钟级别: {stock_code}, {time_level_str}，默认使用通用分钟表 StockMinuteData") # 修改行: print改为logger.warning
                                ModelClass = StockMinuteData
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
                        if time_level_str in ['d', 'w', 'm']: # 修改行: 'day', 'week', 'month' 改为 'd', 'w', 'm'
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
                        else:
                            # 对于拆分后的分钟表，ModelClass 已经包含了时间级别信息，不需要 time_level 字段
                            # 只有当 ModelClass 是 StockMinuteData (即1分钟或未识别的通用分钟表) 时，才需要 time_level 字段
                            if ModelClass == StockMinuteData: # 修改行: 增加判断，只有通用分钟表才需要 time_level 字段
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
                            else: # 拆分后的分钟表
                                instance = ModelClass(
                                    stock=stock,
                                    trade_time=trade_time,
                                    open=open_price,
                                    high=high_price,
                                    low=low_price,
                                    close=close_price,
                                    vol=volume,
                                    amount=amount,
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
        # 选择数据库查询集时，加入时间过滤条件
        try:
            ModelClass: Type[models.Model] # 新增行: 声明 ModelClass 类型
            if time_level_str.lower() == "d":
                # 修改行: 根据股票代码选择日线模型
                if stock_code.startswith('3') and stock_code.endswith('.SZ'):
                    ModelClass = StockDailyData_CY
                elif stock_code.endswith('.SZ'):
                    ModelClass = StockDailyData_SZ
                elif stock_code.startswith('68') and stock_code.endswith('.SH'):
                    ModelClass = StockDailyData_KC
                elif stock_code.endswith('.SH'):
                    ModelClass = StockDailyData_SH
                elif stock_code.endswith('.BJ'):
                    ModelClass = StockDailyData_BJ
                else:
                    logger.warning(f"未识别的股票代码: {stock_code}，默认使用SZ主板日线表") # 修改行: print改为logger.warning
                    ModelClass = StockDailyData_SZ  # 默认返回深市主板
                qs = ModelClass.objects.filter(stock=stock)
            elif time_level_str.lower() == "w":
                # 修改行: 周线数据，假设未拆表
                ModelClass = StockWeeklyData
                qs = ModelClass.objects.filter(stock=stock)
            elif time_level_str.lower() == "m":
                # 修改行: 月线数据，假设未拆表
                ModelClass = StockMonthlyData
                qs = ModelClass.objects.filter(stock=stock)
            else: # 修改行: 分钟线数据
                # 修改行: 根据股票代码和时间级别选择分钟线模型
                # 这里的逻辑与 get_minute_model 保持一致
                if time_level_str not in ['5', '15', '30', '60']:
                    # 修改行: 1min 或其他未拆分的分钟级别，使用原表
                    ModelClass = StockMinuteData
                    qs = ModelClass.objects.filter(stock=stock, time_level=time_level_str)
                elif stock_code.endswith('.SZ'):
                    if stock_code.startswith('3'):
                        ModelClass = {
                            '5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY
                        }[time_level_str]
                    else:
                        ModelClass = {
                            '5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ
                        }[time_level_str]
                    qs = ModelClass.objects.filter(stock=stock) # 修改行: 拆分表不再需要 time_level 字段过滤
                elif stock_code.endswith('.SH'):
                    if stock_code.startswith('68'):
                        ModelClass = {
                            '5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC
                        }[time_level_str]
                    else:
                        ModelClass = {
                            '5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH
                        }[time_level_str]
                    qs = ModelClass.objects.filter(stock=stock) # 修改行: 拆分表不再需要 time_level 字段过滤
                elif stock_code.endswith('.BJ'):
                    ModelClass = {
                        '5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ
                    }[time_level_str]
                    qs = ModelClass.objects.filter(stock=stock) # 修改行: 拆分表不再需要 time_level 字段过滤
                else:
                    # 修改行: 未识别的股票代码，或者 time_level_str 不在 ['5', '15', '30', '60'] 且未被前面的 if 捕获
                    # 修改行: 默认使用原 StockMinuteData 表，并带上 time_level 过滤
                    logger.warning(f"未识别的股票代码或分钟级别: {stock_code}, {time_level_str}，默认使用通用分钟表 StockMinuteData") # 修改行: print改为logger.warning
                    ModelClass = StockMinuteData
                    qs = ModelClass.objects.filter(stock=stock, time_level=time_level_str)
            # 如果提供了起点时间，加入过滤条件
            if start_trade_time:
                qs = qs.filter(trade_time__lte=start_trade_time)
            # 按时间倒序，限制数量
            qs = qs.order_by('-trade_time')[:limit]
            data_list = await sync_to_async(list)(qs)  # 用同步ORM，异步调用
            # 升序排列
            data_list = list(data_list)[::-1]
            # print(f"{stock} data_list_count: {len(data_list)}")
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
                     logger.debug(f"预期时间点范围调整为: {expected_times_filtered[0]} 至 {expected_times_filtered[-1]}，调整后预期时间点数量: {len(expected_times_filtered)}，股票: {stock_code} {time_level_str}")
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
            return data_list
        except Exception as e_db:
            logger.error(f"从数据库获取股票[{stock_code}] {time_level_str} 级别分时成交数据失败: {str(e_db)}", exc_info=True)
            return None

    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000, trade_time: Optional[str] = None) -> Optional[pd.DataFrame]:
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
        history_trades = await self.get_history_time_trades_by_limit(stock_code=stock_code, time_level=time_level_val, limit=limit, trade_time=trade_time)
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
            # print(f"get_history_ohlcv_df.转换并重命名后的 DataFrame 列名: {df.columns.tolist()} for {stock_code} {time_level_val}")
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

    # ▼▼▼【代码修改】: 新增行业分析相关的所有DAO方法 ▼▼▼
    async def get_all_industries(self, industry_type: str = '行业') -> List[ThsIndex]:
        """
        获取所有同花顺行业指数的基本信息。

        Args:
            industry_type (str): 指数类型，默认为'行业'，也可以是'概念'。

        Returns:
            List[ThsIndex]: ThsIndex模型对象的列表。
        """
        print(f"    [DAO] Fetching all industries with type: {industry_type}...")
        # 使用 Django ORM 的异步接口 afilter 和 alist
        industries = await self.sync_to_async_iterable(
            ThsIndex.objects.filter(type=industry_type)
        )
        print(f"    [DAO] Found {len(industries)} industries.")
        return industries

    async def get_industry_daily_data(self, industry_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        获取单个行业指数在指定时间范围内的日线行情数据。

        Args:
            industry_code (str): 行业代码 (e.g., '881101')。
            start_date (date): 开始日期。
            end_date (date): 结束日期。

        Returns:
            pd.DataFrame: 包含行业日线行情的DataFrame，按日期升序排列。
        """
        print(f"    [DAO] Fetching daily data for industry '{industry_code}' from {start_date} to {end_date}...")
        query_set = ThsIndexDaily.objects.filter(
            ths_index__ts_code=industry_code,
            trade_time__range=(start_date, end_date)
        ).order_by('trade_time')
        
        # 使用 .values() 直接获取字典列表，性能更优
        data = await self.sync_to_async_iterable(
            query_set.values(
                'trade_time', 'open', 'high', 'low', 'close', 'pct_change', 'vol'
            )
        )
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(data))
        # 将 trade_time 设置为索引，方便后续时间序列分析
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df.set_index('trade_time', inplace=True)
        print(f"    [DAO] Fetched {len(df)} daily records for industry '{industry_code}'.")
        return df

    async def get_industry_fund_flow(self, industry_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        获取单个行业在指定时间范围内的资金流数据。

        Args:
            industry_code (str): 行业代码。
            start_date (date): 开始日期。
            end_date (date): 结束日期。

        Returns:
            pd.DataFrame: 包含行业资金流数据的DataFrame，按日期升序排列。
        """
        print(f"    [DAO] Fetching fund flow for industry '{industry_code}' from {start_date} to {end_date}...")
        query_set = FundFlowIndustryTHS.objects.filter(
            ths_index__ts_code=industry_code,
            trade_time__range=(start_date, end_date)
        ).order_by('trade_time')

        data = await self.sync_to_async_iterable(
            query_set.values(
                'trade_time', 'net_amount', 'lead_stock', 'pct_change_stock'
            )
        )

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(list(data))
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df.set_index('trade_time', inplace=True)
        print(f"    [DAO] Fetched {len(df)} fund flow records for industry '{industry_code}'.")
        return df

    async def get_industry_members(self, industry_code: str) -> List[str]:
        """
        获取指定行业的所有当前成分股代码。

        Args:
            industry_code (str): 行业代码。

        Returns:
            List[str]: 股票代码列表 (e.g., ['000001.SZ', '600000.SH']).
        """
        print(f"    [DAO] Fetching members for industry '{industry_code}'...")
        # 假设 is_new='Y' 表示当前最新的成分股
        query_set = ThsIndexMember.objects.filter(
            ths_index__ts_code=industry_code,
            is_new='Y'
        ).select_related('stock') # 优化查询，避免N+1问题

        # 使用 values_list 直接获取股票代码列表，性能最高
        members = await self.sync_to_async_iterable(
            query_set.values_list('stock__stock_code', flat=True)
        )
        
        member_list = list(members)
        print(f"    [DAO] Found {len(member_list)} members for industry '{industry_code}'.")
        return member_list

    async def get_stocks_daily_close(self, stock_codes: List[str], trade_date: datetime.date) -> pd.DataFrame:
        """
        获取一批股票在指定交易日的收盘价和前收盘价。
        注意：这个方法需要一个日线行情表，这里假设它叫 `StockDailyData`。
        如果你的个股日线行情表是别的名字，请修改 `StockDailyData`。

        Args:
            stock_codes (List[str]): 股票代码列表。
            trade_date (date): 交易日期。

        Returns:
            pd.DataFrame: 包含 'stock_code', 'close', 'pre_close' 的DataFrame。
        """
        print(f"    [DAO] Fetching daily close for {len(stock_codes)} stocks on {trade_date}...")
        query_set = StockDailyData.objects.filter(
            stock__stock_code__in=stock_codes,
            trade_time=trade_date
        )

        data = await self.sync_to_async_iterable(
            query_set.values(
                stock_code=F('stock__stock_code'), # 通过外键获取股票代码
                close=F('close'),
                pre_close=F('pre_close')
            )
        )

        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(list(data))
        print(f"    [DAO] Fetched close prices for {len(df)} stocks.")
        return df

    @sync_to_async
    def get_latest_industry_fund_flow(self, industry_code: str, trade_date: datetime.date) -> Optional[FundFlowIndustryTHS]:
        """
        【已实现】获取单个行业在指定日期或之前的最新一条资金流数据。
        这对于获取当日的“领涨股”等信息至关重要。

        Args:
            industry_code (str): 同花顺行业代码。
            trade_date (datetime.date): 交易日期。

        Returns:
            Optional[FundFlowIndustryTHS]: 最新的行业资金流模型实例，或在找不到时返回 None。
        """
        # print(f"    [DAO] 正在查询行业 {industry_code} 在 {trade_date} 或之前的最新资金流...")
        try:
            # 筛选小于等于指定日期的记录，按日期降序排列，取第一个
            # .select_related('ths_index') 可以优化性能，如果后续需要访问行业名称
            flow_data = FundFlowIndustryTHS.objects.filter(
                ths_index__ts_code=industry_code,
                trade_time__lte=trade_date
            ).order_by('-trade_time').select_related('ths_index').first()
            if flow_data:
                print(f"    [DAO] 成功找到行业 {industry_code} 的资金流数据，日期为 {flow_data.trade_time}。")
                return flow_data
            flow_data = FundFlowCntTHS.objects.filter(
                    ths_index__ts_code=industry_code,
                    trade_time__lte=trade_date
                ).order_by('-trade_time').select_related('ths_index').first()
            if flow_data:
                print(f"    [DAO] 成功找到行业 {industry_code} 的资金流数据，日期为 {flow_data.trade_time}。")
                return flow_data
            else:
                print(f"    [DAO] 未找到行业 {industry_code} 在 {trade_date} 之前的资金流数据。")
            return flow_data
        except Exception as e:
            logger.error(f"查询行业 {industry_code} 最新资金流时出错: {e}")
            return None

    @sync_to_async
    def get_industry_members(self, industry_code: str) -> List[ThsIndexMember]:
        """
        【已实现】获取指定行业的所有当前成分股。

        Args:
            industry_code (str): 同花顺行业代码。

        Returns:
            List[ThsIndexMember]: 该行业的成分股模型实例列表。
        """
        # print(f"    [DAO] 正在查询行业 {industry_code} 的所有成分股...")
        try:
            # 假设 is_new='Y' 或类似字段表示当前成分股，如果模型没有该字段，则移除该过滤条件
            # .select_related('stock') 是关键的性能优化，避免 N+1 查询
            members = list(
                ThsIndexMember.objects.filter(
                    ths_index__ts_code=industry_code,
                    # is_new='Y' # 如果有该字段用于标识最新成分股，请取消注释
                ).select_related('stock')
            )
            print(f"    [DAO] 成功查询到行业 {industry_code} 的 {len(members)} 只成分股。")
            return members
        except Exception as e:
            logger.error(f"查询行业 {industry_code} 成分股时出错: {e}")
            return []

    @sync_to_async
    def get_stocks_daily_data(self, stock_codes: List[str], trade_date: datetime.date) -> List[StockDailyData_SZ]:
        """
        【已实现】批量获取多支股票在指定日期的日线行情数据。

        Args:
            stock_codes (List[str]): 股票代码列表。
            trade_date (datetime.date): 交易日期。

        Returns:
            List[StockDailyData_SZ]: 日线行情模型实例列表。
        """
        if not stock_codes:
            return []
        print(f"    [DAO] 正在批量查询 {len(stock_codes)} 支股票在 {trade_date} 的日线行情...")
        try:
            # 使用 __in 查询进行高效的批量获取
            daily_data = list(
                StockDailyData_SZ.objects.filter(
                    stock__stock_code__in=stock_codes,
                    trade_time=trade_date
                )
            )
            print(f"    [DAO] 成功查询到 {len(daily_data)} 条日线行情数据。")
            return daily_data
        except Exception as e:
            logger.error(f"批量查询股票日线行情时出错: {e}")
            return []

    @sync_to_async
    def get_stocks_daily_basic(self, stock_codes: List[str], trade_date: datetime.date) -> List[StockDailyBasic]:
        """
        【已实现】批量获取多支股票在指定日期的每日基本面指标（包含涨停状态）。

        Args:
            stock_codes (List[str]): 股票代码列表。
            trade_date (datetime.date): 交易日期。

        Returns:
            List[StockDailyBasic]: 每日基本面指标模型实例列表。
        """
        if not stock_codes:
            return []
        print(f"    [DAO] 正在批量查询 {len(stock_codes)} 支股票在 {trade_date} 的基本面指标（含涨停状态）...")
        try:
            # 使用 __in 查询进行高效的批量获取
            basic_data = list(
                StockDailyBasic.objects.filter(
                    stock__stock_code__in=stock_codes,
                    trade_time=trade_date
                )
            )
            print(f"    [DAO] 成功查询到 {len(basic_data)} 条基本面指标数据。")
            return basic_data
        except Exception as e:
            logger.error(f"批量查询股票每日基本面指标时出错: {e}")
            return []

    # --- 为了让之前的代码能跑通，还需要补齐这两个方法 ---
    @sync_to_async
    def get_all_industries(self) -> List[ThsIndex]:
        """获取所有同花顺行业/概念指数列表"""
        # print("    [DAO] 正在获取所有行业列表...")
        # 假设 type='N' 代表行业, 'C' 代表概念，根据你的实际情况调整
        return list(ThsIndex.objects.filter(type='N'))

    @sync_to_async
    def get_industry_daily_data(self, industry_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """获取行业指数的历史日线行情"""
        # print(f"    [DAO] 正在获取行业 {industry_code} 从 {start_date} 到 {end_date} 的指数行情...")
        qs = ThsIndexDaily.objects.filter(
            ths_index__ts_code=industry_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        df = pd.DataFrame(list(qs.values()))
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df.set_index('trade_time', inplace=True)
        return df

    @sync_to_async
    def get_industry_fund_flow(self, industry_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """获取行业的历史资金流数据"""
        # print(f"    [DAO] 正在获取行业 {industry_code} 从 {start_date} 到 {end_date} 的资金流...")
        qs = FundFlowIndustryTHS.objects.filter(
            ths_index__ts_code=industry_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        df = pd.DataFrame(list(qs.values()))
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df.set_index('trade_time', inplace=True)
        return df

    @sync_to_async
    def get_market_index_daily_data(self, market_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        【修改版】获取大盘基准指数的历史日线行情
        
        修改点:
        1. 查询的数据模型由 ThsIndexDaily 更改为 IndexDaily。
        2. 查询条件根据 IndexDaily 的外键关系调整为 'index__index_code'。
        """
        # print(f"    [DAO] 正在获取大盘指数 {market_code} 从 {start_date} 到 {end_date} 的行情...")
        
        # 代码修改处: 使用新的 IndexDaily 模型进行查询
        # 根据 IndexDaily 的外键 'index' 和其关联字段 'index_code' 进行过滤
        qs = IndexDaily.objects.filter(
            index__index_code=market_code, # 代码修改处: 过滤条件从 ths_index__ts_code 调整为 index__index_code
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        
        # 从查询结果中仅选择需要的字段，以提高效率
        df = pd.DataFrame(list(qs.values('trade_time', 'close')))
        
        # 后续数据处理逻辑保持不变
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'])
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'close': 'market_close'}, inplace=True)
            
        # print(f"    [DAO] 获取到 {len(df)} 条指数 {market_code} 的行情数据。")
        return df

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
                     aware_dt = timezone.make_aware(dt_obj, datetime.timezone.utc)
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








