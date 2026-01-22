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
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from stock_models.industry import ThsIndex, ThsIndexDaily, ThsIndexMember # 导入 ThsIndexMember 模型
from stock_models.time_trade import IndexDaily, StockCyqPerf, StockDailyBasic, StockDailyData, StockMonthlyData, StockWeeklyData
# 导入资金流向相关模型
from stock_models.fund_flow import FundFlowCntTHS, FundFlowIndustryTHS
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from utils.cache_manager import CacheManager
from utils.model_helpers import get_daily_data_model_by_code, get_minute_data_model_by_code_and_timelevel

logger = logging.getLogger("dao")

# 假设 FINTA_OHLCV_MAP 包含必要的列名映射，例如 {'vol': 'volume'}
# 请确保您的 constants.py 文件中 FINTA_OHLCV_MAP 包含了 'vol': 'volume'
# 如果 FINTA_OHLCV_MAP 在 constants.py 中定义，这里无需重复定义

def get_china_a_stock_kline_times(trade_days: list, time_level: str) -> list:
    """
    【V2.0 向量化优化版】生成A股应有的K线标准结束时间点，基于实际交易日历。
    - 核心优化: 对分钟级别('5', '15', '30', '60')的时间点生成逻辑进行了完全向量化重构。
                通过Pandas和NumPy的广播机制，替代了原有的Python for循环，
                一次性计算出所有交易日的所有分钟K线时间点，大幅提升了执行效率。
    Args:
        trade_days: list of datetime.date，实际交易日列表 (naive date)
        time_level: 'd', 'w', 'm', '5', '15', '30', '60'
    Returns:
        list of pd.Timestamp (Asia/Shanghai)，按时间升序排列
    """
    times = []
    default_tz = timezone.get_default_timezone() # 获取Django项目配置的默认时区 (通常是 Asia/Shanghai)
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
    # 开始向量化处理分钟级别逻辑
    elif time_level in ['5', '15', '30', '60']:
        freq = int(time_level)
        if not trade_days:
            return []
        # 根据频率确定上午和下午的开始时间
        morning_start_map = {'5': '09:35:00', '15': '09:45:00', '30': '10:00:00', '60': '10:30:00'}
        afternoon_start_map = {'5': '13:05:00', '15': '13:15:00', '30': '13:30:00', '60': '14:00:00'}
        morning_start_str = morning_start_map.get(time_level)
        afternoon_start_str = afternoon_start_map.get(time_level)
        morning_end_str = '11:30:00'
        afternoon_end_str = '15:00:00'
        # 1. 将交易日列表转换为Pandas的DatetimeIndex，这是向量化操作的基础
        trade_dates_ts = pd.to_datetime(trade_days)
        # 2. 生成一天内的标准时间点（不带日期），并合并
        morning_times_of_day = pd.date_range(start=f'1970-01-01 {morning_start_str}', end=f'1970-01-01 {morning_end_str}', freq=f'{freq}T').time
        afternoon_times_of_day = pd.date_range(start=f'1970-01-01 {afternoon_start_str}', end=f'1970-01-01 {afternoon_end_str}', freq=f'{freq}T').time
        all_times_of_day = np.union1d(morning_times_of_day, afternoon_times_of_day)
        # 3. 将时间点转换为TimedeltaIndex，以便与日期进行向量化加法
        all_timedeltas = pd.to_timedelta([t.strftime('%H:%M:%S') for t in all_times_of_day])
        # 4. 核心向量化操作：通过NumPy的广播机制，将每个交易日与所有日内时间点相加，生成所有时间戳
        # trade_dates_ts.values[:, np.newaxis] 将日期数组变为 (N, 1) 的形状
        # all_timedeltas.values 是 (M,) 的形状
        # 两者相加会广播成 (N, M) 的结果，然后用 flatten() 展平成一维数组
        all_timestamps_naive = (trade_dates_ts.values[:, np.newaxis] + all_timedeltas.values).flatten()
        # 5. 转换为Pandas的DatetimeIndex并设置时区
        times_index = pd.to_datetime(all_timestamps_naive, errors='coerce').dropna()
        times_index_aware = times_index.tz_localize(default_tz)
        # 6. 转换为列表并返回
        return times_index_aware.to_list()
    # 结束向量化处理分钟级别逻辑
    else:
        raise ValueError(f"不支持的K线类型: {time_level}")
    # 确保时间点是唯一的并排序
    return sorted(list(set(times)))


class IndicatorDAO(BaseDAO):
    """
    指标数据访问对象，负责指标数据的读取和存储
    """
    def __init__(self, cache_manager_instance: CacheManager):
        # 调用 super() 时，将 cache_manager_instance 传递进去
        super().__init__(cache_manager_instance=cache_manager_instance, model_class=None)
        self.stock_basic_dao = StockBasicInfoDao(cache_manager_instance)
        self.industry_dao = IndustryDao(cache_manager_instance)
        self.index_basic_dao = IndexBasicDAO(cache_manager_instance)  # 添加 IndexBasicDAO 的初始化
        self.ta = ta
    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000, trade_time: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        【V118.12 分钟线自动聚合修复版】
        - 核心修复: 当请求非1分钟的分钟级数据（如'60'）且数据库仅有1分钟数据时，自动获取1分钟数据并重采样聚合。
        - 逻辑调整: 自动放大 limit 查询量，确保聚合后有足够的数据点。
        - 字段修复: 保持对分钟线模型不查询 'pre_close' 的逻辑。
        """
        # 1. 解析目标时间级别
        target_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level).lower()
        
        # 2. 判断是否需要从1分钟数据聚合
        # 如果是数字且不等于"1"（例如 "60", "30", "15"），或者显式以 "min" 结尾但不是 "1min"
        is_minute_aggregation = False
        aggregation_period = 1
        query_level_str = target_level_str # 默认查询级别等于目标级别

        if (target_level_str.isdigit() and target_level_str != "1") or \
           (target_level_str.endswith("min") and target_level_str != "1min"):
            is_minute_aggregation = True
            # 提取周期数字，例如 "60" -> 60
            aggregation_period = int(''.join(filter(str.isdigit, target_level_str)))
            query_level_str = "1" # 强制查询1分钟数据源
            # 放大 limit: 如果需要 100 个 60分钟K线，需要查询 100 * 60 个 1分钟K线
            # 增加一点 buffer (1.2倍) 防止数据缺失导致聚合后数量不足
            original_limit = limit
            limit = int(limit * aggregation_period * 1.2)
            # 限制最大查询量，防止内存溢出 (例如限制在 50000 条)
            limit = min(limit, 50000)

        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None

        try:
            ModelClass: Optional[Type[models.Model]] = None
            # 3. 根据 query_level_str 选择模型 (此时如果是聚合，query_level_str 已经是 "1")
            if query_level_str == "d":
                ModelClass = get_daily_data_model_by_code(stock_code)
            elif query_level_str == "w": ModelClass = StockWeeklyData
            elif query_level_str == "m": ModelClass = StockMonthlyData
            else:
                # 这里通常会获取到 Stock1MinData 相关的模型
                ModelClass = get_minute_data_model_by_code_and_timelevel(stock_code, query_level_str)
            if not ModelClass:
                logger.error(f"未能为 {stock_code} 在时间级别 {query_level_str} 找到对应的数据库模型。")
                return None
            model_name = ModelClass._meta.db_table
            qs = ModelClass.objects.filter(stock=stock)
            if trade_time:
                trade_time_dt = self._safe_datetime(trade_time)
                if trade_time_dt:
                    qs = qs.filter(trade_time__lte=trade_time_dt)
            # 4. 定义字段 (分钟线不含 pre_close)
            if query_level_str == "d":
                fields = ['trade_time', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq', 'pre_close_qfq', 'vol', 'amount']
                rename_map = {'open_qfq': 'open', 'high_qfq': 'high', 'low_qfq': 'low', 'close_qfq': 'close', 'pre_close_qfq': 'pre_close', 'vol': 'volume'}
            elif query_level_str in ["w", "m"]:
                fields = ['trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
                rename_map = {'vol': 'volume'}
            else:
                # 分钟线 (包括 "1")
                fields = ['trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']
                rename_map = {'vol': 'volume'}
            limited_qs = qs.order_by('-trade_time')[:limit]
            data_values = await sync_to_async(list)(limited_qs.values(*fields))
            if not data_values:
                logger.warning(f"数据库未返回任何数据 for {stock_code} {query_level_str} from table {model_name}")
                return None
            df = pd.DataFrame.from_records(data_values)
            df = df.iloc[::-1].reset_index(drop=True)
            df.rename(columns=rename_map, inplace=True)
            if 'trade_time' not in df.columns:
                logger.error(f"查询结果缺少 'trade_time' 列: {stock_code} {query_level_str}")
                return None
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True, errors='coerce')
            df.dropna(subset=['trade_time'], inplace=True)
            df.set_index('trade_time', inplace=True)
            if df.empty:
                return None
            # 5. 【核心新增】执行分钟线聚合
            if is_minute_aggregation:
                # 定义聚合规则
                agg_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'amount': 'sum'
                }
                # A股分钟线通常是 label='right', closed='right' (例如 10:30 的bar代表 10:29:00-10:30:00)
                # 使用 aggregation_period (例如 60) 进行重采样
                resample_rule = f"{aggregation_period}min"
                # 确保只对存在的列进行聚合
                valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
                df_resampled = df.resample(resample_rule, label='right', closed='right').agg(valid_agg_dict)
                df_resampled.dropna(inplace=True) # 删除因非交易时间产生的空行
                # 截取请求的原始数量 (因为我们之前放大了limit)
                if len(df_resampled) > original_limit:
                    df = df_resampled.iloc[-original_limit:]
                else:
                    df = df_resampled
                logger.info(f"[{stock_code}] 已将 1分钟 数据聚合为 {target_level_str}分钟 数据，结果行数: {len(df)}")
            # 6. 最终检查
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in df.columns]
                logger.error(f"DataFrame 缺少必要列: {missing}, 实际列: {df.columns.tolist()}")
                return None
            return df
        except Exception as e:
            logger.error(f"从数据库获取并转换 {stock_code} {target_level_str} 数据失败: {e}", exc_info=True)
            return None

    # ▼▼▼ 行业分析相关的所有DAO方法 ▼▼▼
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
        industries = await sync_to_async(list)(ThsIndex.objects.filter(type=industry_type))
        print(f"    [DAO] Found {len(industries)} industries.")
        return industries
    async def get_stocks_daily_close(self, stock_codes: List[str], trade_date: datetime.date) -> pd.DataFrame:
        """
        获取一批股票在指定交易日的收盘价和前收盘价。
        - 核心优化: 简化了DataFrame的创建过程，避免了不必要的数据复制。
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
        # 使用 sync_to_async(list) 执行查询，结果 data 已经是 List[Dict] 类型
        data = await sync_to_async(list)(query_set.values(
            stock_code=F('stock__stock_code'),
            close=F('close'),
            pre_close=F('pre_close')
        ))
        if not data:
            return pd.DataFrame()
        # 直接使用 data 创建DataFrame，无需再次调用 list()，避免了不必要的列表拷贝
        df = pd.DataFrame(data)
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
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
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
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'close': 'market_close'}, inplace=True)
        # print(f"    [DAO] 获取到 {len(df)} 条指数 {market_code} 的行情数据。")
        return df
    @sync_to_async
    def get_cyq_perf_for_stock_and_dates(self, stock_code: str, trade_dates: List[pd.Timestamp]) -> Optional[pd.DataFrame]:
        """
        获取指定股票在指定日期范围内的筹码表现数据(StockCyqPerf)。
        Args:
            stock_code (str): 股票代码。
            trade_dates (List[pd.Timestamp]): 交易日期列表。
        Returns:
            Optional[pd.DataFrame]: 包含筹码表现数据的DataFrame，以trade_time为索引。如果无数据则返回None。
        """
        if not trade_dates:
            return None
        start_date = min(trade_dates).date()
        end_date = max(trade_dates).date()
        try:
            # 使用Django ORM进行高效查询
            queryset = StockCyqPerf.objects.filter(
                stock__stock_code=stock_code,
                trade_time__range=(start_date, end_date)
            ).values(
                'trade_time',
                'cost_15pct',
                'cost_85pct',
                'weight_avg',
                'winner_rate'
            )
            if not queryset.exists():
                return None
            # 将查询结果转换为DataFrame
            df = pd.DataFrame.from_records(queryset)
            # 将trade_time转换为datetime类型以便合并
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df = df.set_index('trade_time')
            # 为列名添加前缀和后缀，以符合策略框架的规范
            df = df.rename(columns={
                'cost_15pct': 'CYQ_cost_15pct_D',
                'cost_85pct': 'CYQ_cost_85pct_D',
                'weight_avg': 'CYQ_weight_avg_D',
                'winner_rate': 'CYQ_winner_rate_D'
            })
            return df
        except Exception as e:
            print(f"[错误] 在CyqDao中获取股票 {stock_code} 的筹码数据时出错: {e}")
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








