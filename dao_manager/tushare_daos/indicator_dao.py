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
from stock_models.time_trade import IndexDaily, StockDailyBasic, StockMonthlyData, StockWeeklyData
from stock_models.chip import StockCyqPerf
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
    【V2.1 - 全面向量化版】生成A股应有的K线标准结束时间点。
    优化：
    1. 分钟级逻辑保持原有的向量化广播机制。
    2. 新增：日、周、月级别逻辑全部重构为 Pandas 向量化操作，移除 Python 循环。
       利用 to_period 和 groupby 瞬间完成数千个交易日的周期聚合。
    """
    if not trade_days:
        return []
    default_tz = timezone.get_default_timezone()
    # 统一转换为 DatetimeIndex，这是向量化的基础
    trade_dates_ts = pd.to_datetime(trade_days)
    if time_level.lower() == 'd':
        # 向量化：直接添加时间部分并本地化时区
        # normalize() 将时间置为 00:00:00
        times_index = trade_dates_ts.normalize().tz_localize(default_tz)
        return times_index.to_list()
    elif time_level.lower() == 'w':
        # 向量化：按周分组取最大值
        # to_period('W') 将日期转换为周对象，groupby 取每组的 max (即本周最后一个交易日)
        df = pd.DataFrame({'date': trade_dates_ts})
        # 使用 'W-SUN' 确保周日为一周结束，与 isocalendar 逻辑一致
        last_days = df.groupby(df['date'].dt.to_period('W-SUN'))['date'].max()
        times_index = last_days.dt.normalize().dt.tz_localize(default_tz)
        return sorted(times_index.to_list())
    elif time_level.lower() == 'm':
        # 向量化：按月分组取最大值
        df = pd.DataFrame({'date': trade_dates_ts})
        last_days = df.groupby(df['date'].dt.to_period('M'))['date'].max()
        times_index = last_days.dt.normalize().dt.tz_localize(default_tz)
        return sorted(times_index.to_list())
    elif time_level in ['5', '15', '30', '60']:
        freq = int(time_level)
        morning_start_map = {'5': '09:35:00', '15': '09:45:00', '30': '10:00:00', '60': '10:30:00'}
        afternoon_start_map = {'5': '13:05:00', '15': '13:15:00', '30': '13:30:00', '60': '14:00:00'}
        morning_start_str = morning_start_map.get(time_level)
        afternoon_start_str = afternoon_start_map.get(time_level)
        morning_end_str = '11:30:00'
        afternoon_end_str = '15:00:00'
        morning_times_of_day = pd.date_range(start=f'1970-01-01 {morning_start_str}', end=f'1970-01-01 {morning_end_str}', freq=f'{freq}T').time
        afternoon_times_of_day = pd.date_range(start=f'1970-01-01 {afternoon_start_str}', end=f'1970-01-01 {afternoon_end_str}', freq=f'{freq}T').time
        all_times_of_day = np.union1d(morning_times_of_day, afternoon_times_of_day)
        all_timedeltas = pd.to_timedelta([t.strftime('%H:%M:%S') for t in all_times_of_day])
        # 广播加法：(N, 1) + (M,) -> (N, M) -> flatten
        all_timestamps_naive = (trade_dates_ts.values[:, np.newaxis] + all_timedeltas.values).flatten()
        times_index = pd.to_datetime(all_timestamps_naive, errors='coerce').dropna()
        times_index_aware = times_index.tz_localize(default_tz)
        return times_index_aware.to_list()
    else:
        raise ValueError(f"不支持的K线类型: {time_level}")


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

    async def get_history_ohlcv_df(self, stock_code: str, time_level: Union[TimeLevel, str], limit: int = 1000, trade_time: Optional[str] = None, start_date: Optional[datetime.date] = None) -> Optional[pd.DataFrame]:
        """
        【V118.14 - 类型向量化优化版】
        优化：
        1. 增加 Decimal -> float 的向量化转换，避免 object 类型导致的计算龟速。
        2. 保持原有的聚合逻辑和时间过滤逻辑。
        """
        target_level_str = time_level.value if isinstance(time_level, TimeLevel) else str(time_level).lower()
        is_minute_aggregation = False
        aggregation_period = 1
        query_level_str = target_level_str
        if (target_level_str.isdigit() and target_level_str != "1") or \
           (target_level_str.endswith("min") and target_level_str != "1min"):
            is_minute_aggregation = True
            aggregation_period = int(''.join(filter(str.isdigit, target_level_str)))
            query_level_str = "1"
            if not start_date:
                original_limit = limit
                limit = int(limit * aggregation_period * 1.2)
                limit = min(limit, 50000)
            else:
                original_limit = limit
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            logger.warning(f"无法找到股票信息: {stock_code}")
            return None
        try:
            ModelClass: Optional[Type[models.Model]] = None
            if query_level_str == "d":
                ModelClass = get_daily_data_model_by_code(stock_code)
            elif query_level_str == "w": ModelClass = StockWeeklyData
            elif query_level_str == "m": ModelClass = StockMonthlyData
            else:
                ModelClass = get_minute_data_model_by_code_and_timelevel(stock_code, query_level_str)
            if not ModelClass:
                return None
            qs = ModelClass.objects.filter(stock=stock)
            if trade_time:
                trade_time_dt = self._safe_datetime(trade_time)
                if trade_time_dt:
                    qs = qs.filter(trade_time__lte=trade_time_dt)
            if start_date:
                qs = qs.filter(trade_time__gte=start_date)
            if query_level_str == "d":
                fields = ['trade_time', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq', 'pre_close_qfq', 'vol', 'amount']
                rename_map = {'open_qfq': 'open', 'high_qfq': 'high', 'low_qfq': 'low', 'close_qfq': 'close', 'pre_close_qfq': 'pre_close', 'vol': 'volume'}
            elif query_level_str in ["w", "m"]:
                fields = ['trade_time', 'open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount']
                rename_map = {'vol': 'volume'}
            else:
                fields = ['trade_time', 'open', 'high', 'low', 'close', 'vol', 'amount']
                rename_map = {'vol': 'volume'}
            if start_date:
                limited_qs = qs.order_by('-trade_time')[:50000] 
            else:
                limited_qs = qs.order_by('-trade_time')[:limit]
            data_values = await sync_to_async(list)(limited_qs.values(*fields))
            if not data_values:
                return None
            df = pd.DataFrame.from_records(data_values)
            df = df.iloc[::-1].reset_index(drop=True)
            df.rename(columns=rename_map, inplace=True)
            if 'trade_time' not in df.columns:
                return None
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True, errors='coerce')
            df.dropna(subset=['trade_time'], inplace=True)
            df.set_index('trade_time', inplace=True)
            # --- 核心优化：向量化类型转换 ---
            # 识别数值列，将 Decimal (object) 转换为 float64
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount']
            cols_to_convert = [c for c in numeric_cols if c in df.columns]
            if cols_to_convert:
                # astype(float) 能正确处理 Decimal 对象，比 apply(lambda) 快
                df[cols_to_convert] = df[cols_to_convert].astype(float)
            # ---------------------------
            if df.empty:
                return None
            if is_minute_aggregation:
                agg_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'amount': 'sum'
                }
                resample_rule = f"{aggregation_period}min"
                valid_agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}
                df_resampled = df.resample(resample_rule, label='right', closed='right').agg(valid_agg_dict)
                df_resampled.dropna(inplace=True)
                if not start_date:
                    if len(df_resampled) > original_limit:
                        df = df_resampled.iloc[-original_limit:]
                    else:
                        df = df_resampled
                else:
                    df = df_resampled
                logger.info(f"[{stock_code}] 已将 1分钟 数据聚合为 {target_level_str}分钟 数据，结果行数: {len(df)}")
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                return None
            return df
        except Exception as e:
            logger.error(f"从数据库获取并转换 {stock_code} {target_level_str} 数据失败: {e}", exc_info=True)
            return None

    # ▼▼▼ 行业分析相关的所有DAO方法 ▼▼▼
    async def get_all_industries(self, industry_type: str = '行业') -> List[ThsIndex]:
        """
        【V1.1 - 日志规范化】
        获取所有同花顺行业指数的基本信息。
        优化：使用 logger 替代 print。
        """
        logger.info(f"正在获取所有类型为 '{industry_type}' 的行业信息...")
        # 使用 Django ORM 的异步接口
        industries = await sync_to_async(list)(ThsIndex.objects.filter(type=industry_type))
        logger.info(f"共找到 {len(industries)} 个行业。")
        return industries

    async def get_stocks_daily_close(self, stock_codes: List[str], trade_date: datetime.date) -> pd.DataFrame:
        """
        【V2.0 - 类型优化版】获取一批股票在指定交易日的收盘价和前收盘价。
        优化：将 Decimal 类型的价格数据强制转换为 float，提升计算性能。
        """
        print(f"    [DAO] Fetching daily close for {len(stock_codes)} stocks on {trade_date}...")
        query_set = StockDailyData.objects.filter(
            stock__stock_code__in=stock_codes,
            trade_time=trade_date
        )
        data = await sync_to_async(list)(query_set.values(
            stock_code=F('stock__stock_code'),
            close=F('close'),
            pre_close=F('pre_close')
        ))
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        # 向量化类型转换
        numeric_cols = ['close', 'pre_close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(float)
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
        【V1.1 - ORM优化版】
        批量获取多支股票在指定日期的每日基本面指标（包含涨停状态）。
        优化：增加 select_related('stock') 避免 N+1 查询问题。
        """
        if not stock_codes:
            return []
        try:
            # 使用 __in 查询进行高效的批量获取
            # select_related('stock') 确保在访问 stock 外键时不会触发额外查询
            basic_data = list(
                StockDailyBasic.objects.filter(
                    stock__stock_code__in=stock_codes,
                    trade_time=trade_date
                ).select_related('stock')
            )
            return basic_data
        except Exception as e:
            logger.error(f"批量查询股票每日基本面指标时出错: {e}")
            return []

    @sync_to_async
    def get_industry_daily_data(self, industry_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        【V2.0 - 类型优化版】获取行业指数的历史日线行情
        优化：确保返回的 DataFrame 中数值列为 float 类型。
        """
        qs = ThsIndexDaily.objects.filter(
            ths_index__ts_code=industry_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        df = pd.DataFrame(list(qs.values()))
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            # 转换所有可能的数值列
            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'vol', 'amount', 'pct_chg']
            cols_to_convert = [c for c in numeric_cols if c in df.columns]
            if cols_to_convert:
                df[cols_to_convert] = df[cols_to_convert].astype(float)
        return df

    @sync_to_async
    def get_market_index_daily_data(self, market_code: str, start_date: datetime.date, end_date: datetime.date) -> pd.DataFrame:
        """
        【V2.0 - 类型优化版】获取大盘基准指数的历史日线行情
        优化：确保 close 字段转换为 float。
        """
        qs = IndexDaily.objects.filter(
            index__index_code=market_code,
            trade_time__gte=start_date,
            trade_time__lte=end_date
        ).order_by('trade_time')
        df = pd.DataFrame(list(qs.values('trade_time', 'close')))
        if not df.empty:
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df.set_index('trade_time', inplace=True)
            df.rename(columns={'close': 'market_close'}, inplace=True)
            # 类型转换
            if 'market_close' in df.columns:
                df['market_close'] = df['market_close'].astype(float)
        return df

    @sync_to_async
    def get_cyq_perf_for_stock_and_dates(self, stock_code: str, trade_dates: List[pd.Timestamp]) -> Optional[pd.DataFrame]:
        """
        【V2.0 - 类型优化版】获取指定股票在指定日期范围内的筹码表现数据。
        优化：将筹码成本和获利比例等字段强制转换为 float。
        """
        if not trade_dates:
            return None
        start_date = min(trade_dates).date()
        end_date = max(trade_dates).date()
        try:
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
            df = pd.DataFrame.from_records(queryset)
            df['trade_time'] = pd.to_datetime(df['trade_time'], utc=True)
            df = df.set_index('trade_time')
            df = df.rename(columns={
                'cost_15pct': 'CYQ_cost_15pct_D',
                'cost_85pct': 'CYQ_cost_85pct_D',
                'weight_avg': 'CYQ_weight_avg_D',
                'winner_rate': 'CYQ_winner_rate_D'
            })
            # 向量化类型转换
            numeric_cols = ['CYQ_cost_15pct_D', 'CYQ_cost_85pct_D', 'CYQ_weight_avg_D', 'CYQ_winner_rate_D']
            cols_to_convert = [c for c in numeric_cols if c in df.columns]
            if cols_to_convert:
                df[cols_to_convert] = df[cols_to_convert].astype(float)
            return df
        except Exception as e:
            print(f"[错误] 在CyqDao中获取股票 {stock_code} 的筹码数据时出错: {e}")
            return None

    # 添加安全转换辅助函数（确保存在且正确）
    def _safe_decimal(self, value: Any) -> Optional[Decimal]:
        """
        【V1.1 - 标量转换优化】
        将输入值安全转换为 Decimal 类型。
        优化：针对 int 和 Decimal 类型增加快速路径，避免不必要的字符串转换。
        """
        if value is None:
            return None
        
        # 快速路径
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int):
            return Decimal(value)
            
        try:
            # 对于 float，为了避免精度问题 (如 1.1 -> 1.1000000000000000888)，通常建议先转 str
            # 对于 string，直接转换
            return Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError) as e:
            # 仅在转换失败时记录日志，减少正常流程开销
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 Decimal: {e}")
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        """
        【V1.1 - 标量转换优化】
        将输入值安全转换为 int 类型。
        优化：优先处理数字类型，减少异常捕获开销。
        """
        if value is None:
            return None
        
        # 快速路径
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, Decimal):
            return int(value)
            
        try:
            # 处理字符串
            if isinstance(value, str):
                # 处理 "100.0" 这种情况，直接 int("100.0") 会报错
                if '.' in value:
                    return int(float(value))
                return int(value)
            
            # 其他情况尝试强转
            return int(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 int: {e}")
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        """
        【V1.1 - 标量转换优化】
        将输入值安全转换为 float 类型。
        优化：优先处理数字类型，减少异常捕获开销。
        """
        if value is None:
            return None
            
        # 快速路径
        if isinstance(value, float):
            return value
        if isinstance(value, (int, Decimal)):
            return float(value)
            
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"无法将值 '{value}' (类型: {type(value).__name__}) 安全转换为 float: {e}")
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








