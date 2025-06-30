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
from stock_models.fund_flow import FundFlowCntTHS, FundFlowIndustryTHS
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

    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000, trade_time: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        【V4.1 重构优化版】获取历史数据并直接高效转换为 pandas DataFrame。
        - 修复BUG: 采用先选择模型、后构建查询的清晰结构，根治因复杂分支导致过滤器失效的问题。
        - 结构优化: 遵循DRY原则，将模型选择与查询构建分离，代码更清晰、更健壮。
        - 保持高效: 仍然使用 Django ORM 的 .values() 方法直接查询数据库。
        """
        # 1. 统一时间级别 (代码无变化)
        time_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level).lower()
        
        print(f"    [DAO-数据库 V4.1] 正在为 {stock_code} ({time_level_str}) 从数据库查询 {limit} 条数据...")

        # 2. 获取股票实例 (代码无变化)
        if self.stock_basic_dao is None:
            await self.initialize_cache_objects()
        
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None

        try:
            # --- 【重构核心】步骤 3: 模型选择 ---
            # 在这个阶段，我们只负责一件事：根据条件找到正确的模型类(ModelClass)
            ModelClass: Optional[Type[models.Model]] = None
            extra_filters = {} # 用于处理像通用分钟线那样的额外过滤条件

            if time_level_str == "d":
                if stock_code.startswith('3') and stock_code.endswith('.SZ'): ModelClass = StockDailyData_CY
                elif stock_code.endswith('.SZ'): ModelClass = StockDailyData_SZ
                elif stock_code.startswith('68') and stock_code.endswith('.SH'): ModelClass = StockDailyData_KC
                elif stock_code.endswith('.SH'): ModelClass = StockDailyData_SH
                elif stock_code.endswith('.BJ'): ModelClass = StockDailyData_BJ
                else:
                    logger.warning(f"未识别的日线股票代码: {stock_code}，默认使用SZ主板日线表")
                    ModelClass = StockDailyData_SZ
            elif time_level_str == "w":
                ModelClass = StockWeeklyData
            elif time_level_str == "m":
                ModelClass = StockMonthlyData
            else: # 分钟线模型选择
                model_map = None
                if stock_code.endswith('.SZ'):
                    base_map = {'5': StockMinuteData_5_SZ, '15': StockMinuteData_15_SZ, '30': StockMinuteData_30_SZ, '60': StockMinuteData_60_SZ}
                    cy_map = {'5': StockMinuteData_5_CY, '15': StockMinuteData_15_CY, '30': StockMinuteData_30_CY, '60': StockMinuteData_60_CY}
                    model_map = cy_map if stock_code.startswith('3') else base_map
                elif stock_code.endswith('.SH'):
                    base_map = {'5': StockMinuteData_5_SH, '15': StockMinuteData_15_SH, '30': StockMinuteData_30_SH, '60': StockMinuteData_60_SH}
                    kc_map = {'5': StockMinuteData_5_KC, '15': StockMinuteData_15_KC, '30': StockMinuteData_30_KC, '60': StockMinuteData_60_KC}
                    model_map = kc_map if stock_code.startswith('68') else base_map
                elif stock_code.endswith('.BJ'):
                    model_map = {'5': StockMinuteData_5_BJ, '15': StockMinuteData_15_BJ, '30': StockMinuteData_30_BJ, '60': StockMinuteData_60_BJ}

                if model_map and time_level_str in model_map:
                    ModelClass = model_map[time_level_str]
                else: # 1分钟或未识别的通用分钟表
                    logger.warning(f"未找到特定分钟线表 for {stock_code} {time_level_str}, 使用通用分钟表 StockMinuteData")
                    ModelClass = StockMinuteData
                    extra_filters['time_level'] = time_level_str

            # --- 【重构核心】步骤 4: 统一构建和执行查询 ---
            # 如果没有找到对应的模型，直接退出
            if not ModelClass:
                logger.error(f"未能为 {stock_code} 在时间级别 {time_level_str} 找到对应的数据库模型。")
                return None

            # 从这里开始，我们基于已确定的 ModelClass 构建查询，保证过滤条件一定生效
            # 1. 基础查询，并应用最关键的股票代码过滤
            qs = ModelClass.objects.filter(stock=stock)

            # 2. 应用额外过滤条件 (主要用于通用分钟表)
            if extra_filters:
                qs = qs.filter(**extra_filters)

            # 3. 应用时间过滤条件
            if trade_time:
                trade_time_dt = self._safe_datetime(trade_time)
                if trade_time_dt:
                    qs = qs.filter(trade_time__lte=trade_time_dt)

            # 4. 排序、选择列、限制数量并执行查询
            fields = ['trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']
            data_values = await sync_to_async(list)(
                qs.order_by('-trade_time').values(*fields)[:limit]
            )

            # 后续的数据处理部分 (步骤 5, 6, 7, 8, 9) 与您的原代码相同，无需修改
            if not data_values:
                logger.warning(f"数据库未返回任何数据 for {stock_code} {time_level_str}")
                return None
            
            print(f"    [DAO-数据库] 成功从数据库获取 {len(data_values)} 条数据。")

            df = pd.DataFrame.from_records(data_values)
            df = df.iloc[::-1].reset_index(drop=True)
            df.rename(columns={'vol': 'volume'}, inplace=True)

            if 'trade_time' not in df.columns:
                logger.error(f"查询结果缺少 'trade_time' 列: {stock_code} {time_level_str}")
                return None
            
            df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce')
            df.dropna(subset=['trade_time'], inplace=True)
            df.set_index('trade_time', inplace=True)

            if df.empty:
                logger.warning(f"处理时间索引后 DataFrame 为空: {stock_code} {time_level_str}")
                return None

            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                logger.error(f"DataFrame 缺少必要列: {missing}, 实际列: {df.columns.tolist()}")
                return None
            
            print(f"    [DAO-DataFrame] 成功生成DataFrame，共 {len(df)} 行，数据准备返回。")
            return df

        except Exception as e:
            logger.error(f"从数据库获取并转换 {stock_code} {time_level_str} 数据失败: {e}", exc_info=True)
            return None

    # ▼▼▼【 新增行业分析相关的所有DAO方法 ▼▼▼
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

    async def get_stock_industry_info(self, stock_code: str) -> Optional[Dict[str, str]]:
        """
        【V1.0 新增】根据股票代码获取其当前所属的行业/概念板块信息。
        Args:
            stock_code (str): 股票代码 (例如 '600519.SH')。
        Returns:
            Optional[Dict[str, str]]: 包含行业代码和名称的字典，例如 {'code': '881152.TI', 'name': '白酒'}。
                                      如果未找到，则返回 None。
        """
        print(f"      - [DAO查询] 正在查询股票 {stock_code} 的所属行业...")
        try:
            # ▼▼▼【代码实现】: 这是您需要的新方法的核心逻辑 ▼▼▼
            # 使用 Django 异步 ORM 进行查询
            # 1. select_related('ths_index'): 预加载关联的行业信息，避免N+1查询。
            # 2. filter(stock__stock_code=...): 按股票代码过滤。
            # 3. filter(out_date__isnull=True): 只选择当前有效的成分股关系 (尚未被剔除)。
            # 4. afirst(): 异步获取第一个匹配的记录。
            membership = await ThsIndexMember.objects.select_related('ths_index').filter(
                stock__stock_code=stock_code,
                out_date__isnull=True
            ).afirst()
            # ▲▲▲【代码实现】: 核心逻辑结束 ▲▲▲

            if membership and membership.ths_index:
                # 如果找到了有效的成员关系，并且其关联的指数也存在
                industry_info = {
                    'code': membership.ths_index.ts_code,
                    'name': membership.ths_index.name
                }
                print(f"      - [DAO查询] 成功找到 {stock_code} 所属行业: {industry_info['name']} ({industry_info['code']})")
                return industry_info
            else:
                # 如果没有找到任何有效的成员关系
                print(f"      - [DAO查询] 未能找到 {stock_code} 的当前所属行业。")
                return None
        except Exception as e:
            logger.error(f"查询股票 {stock_code} 的行业信息时发生数据库错误: {e}", exc_info=True)
            print(f"      - [DAO查询] 查询 {stock_code} 行业信息时出错: {e}")
            return None

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
                # print(f"    [DAO] 成功找到行业 {industry_code} 的资金流数据，日期为 {flow_data.trade_time}。")
                return flow_data
            flow_data = FundFlowCntTHS.objects.filter(
                    ths_index__ts_code=industry_code,
                    trade_time__lte=trade_date
                ).order_by('-trade_time').select_related('ths_index').first()
            if flow_data:
                # print(f"    [DAO] 成功找到行业 {industry_code} 的资金流数据，日期为 {flow_data.trade_time}。")
                return flow_data
            else:
                pass
                # print(f"    [DAO] 未找到行业 {industry_code} 在 {trade_date} 之前的资金流数据。")
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
            # print(f"    [DAO] 成功查询到行业 {industry_code} 的 {len(members)} 只成分股。")
            return members
        except Exception as e:
            logger.error(f"查询行业 {industry_code} 成分股时出错: {e}")
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
        # print(f"    [DAO] 正在批量查询 {len(stock_codes)} 支股票在 {trade_date} 的基本面指标（含涨停状态）...")
        try:
            # 使用 __in 查询进行高效的批量获取
            basic_data = list(
                StockDailyBasic.objects.filter(
                    stock__stock_code__in=stock_codes,
                    trade_time=trade_date
                )
            )
            # print(f"    [DAO] 成功查询到 {len(basic_data)} 条基本面指标数据。")
            return basic_data
        except Exception as e:
            logger.error(f"批量查询股票每日基本面指标时出错: {e}")
            return []

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








