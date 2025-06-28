# services\indicator_services.py
import asyncio
from collections import defaultdict
import datetime
from functools import reduce
import json
import os
import sys
import traceback
import warnings
import logging
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from dao_manager.tushare_daos.indicator_dao import IndicatorDAO
import numpy as np
import pandas as pd
import math
from django.utils import timezone
from typing import Any, Callable, List, Optional, Set, Tuple, Union, Dict
from django.db import models
import pandas_ta as ta
from dao_manager.tushare_daos.industry_dao import IndustryDao
from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from dao_manager.tushare_daos.index_basic_dao import IndexBasicDAO
from core.constants import TimeLevel
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO
from dao_manager.tushare_daos.strategies_dao import StrategiesDAO
from stock_models.time_trade import IndexDaily
from utils.config_loader import load_strategy_config

warnings.filterwarnings(action='ignore', category=UserWarning, message='.*drop timezone information.*')
warnings.filterwarnings(action='ignore', category=FutureWarning, message=".*Passing 'suffixes' which cause duplicate columns.*")
pd.options.mode.chained_assignment = None

logger = logging.getLogger("services")

DERIVATIVE_BASE_TO_REGISTERED_INDICATOR_MAP = {
    'K': 'KDJ',
    'D': 'KDJ',
    'J': 'KDJ',
    'ADX': 'DMI',
    'PDI': 'DMI',
    'NDI': 'DMI',
}

class IndicatorService:
    """
    技术指标计算服务 (使用 pandas-ta)
    负责获取多时间级别原始数据，进行时间序列标准化（重采样），计算指标，合并数据并进行最终填充。
    并行处理数据获取、重采样和指标计算任务以提高效率。
    """
    def __init__(self):
        """
        初始化 IndicatorService。
        设置 DAO 对象，并动态导入 pandas_ta 库。
        """
        self.indicator_dao = IndicatorDAO()
        self.industry_dao = IndustryDao()
        self.stock_basic_dao = StockBasicInfoDao()
        self.index_dao = IndexBasicDAO()
        self.strategies_dao = StrategiesDAO() # 实例化DAO
        
        try:
            global ta
            import pandas_ta as ta
            if ta is None:
                 logger.warning("pandas_ta 之前导入失败，尝试重新导入。")
                 import pandas_ta as ta
        except ImportError:
            logger.error("pandas-ta 库未安装，请运行 'pip install pandas-ta'")
            ta = None

    def _load_config(self, path: str) -> Dict:
        """
        【辅助函数】从给定的路径加载JSON配置文件。
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"    - 警告: 配置文件未找到: {path}")
            return {}
        except json.JSONDecodeError:
            print(f"    - 警告: 配置文件格式错误: {path}")
            return {}

    def _get_timeframe_in_minutes(self, tf: str) -> int:
        """
        【修正版】辅助函数：将时间周期字符串转换为【交易分钟数】。
        A股交易时间标准:
        - 1个交易日 = 4小时 = 240分钟
        - 1个交易周 = 5个交易日
        - 1个交易月 ≈ 21个交易日 (采用斐波那契数作为近似值)
        """
        tf_upper = str(tf).upper()
        
        # 处理数字格式的分钟周期，例如 '5', '15', '30', '60'
        if tf_upper.isdigit():
            return int(tf_upper)
        
        # 定义基础交易单位的分钟数
        MINUTES_PER_DAY = 4 * 60  # A股交易日: 4小时 * 60分钟/小时 = 240分钟

        # 使用交易时间而非日历时间进行计算
        if tf_upper == 'D':
            return MINUTES_PER_DAY
        if tf_upper == 'W':
            return 5 * MINUTES_PER_DAY  # 每周5个交易日
        if tf_upper == 'M':
            return 21 * MINUTES_PER_DAY # 每月约21个交易日
            
        # 对于无法识别的周期，返回一个极大的整数，以保证类型一致性
        # sys.maxsize 是Python中整数能表示的最大值，作为“无穷大”使用
        return sys.maxsize

    def _get_resample_freq_str(self, tf_str: str) -> Optional[str]:
        """
        将时间级别字符串转换为 pandas resample 函数使用的频率字符串。
        例如: '5' -> '5T', 'D' -> 'D'。

        Args:
            tf_str (str): 时间级别字符串。

        Returns:
            Optional[str]: pandas 频率字符串，如果不支持则返回 None。
        """
        tf_str = str(tf_str).upper()
        if tf_str.isdigit():
            return f'{tf_str}T'
        elif tf_str in ['D', 'W', 'M']:
            return tf_str
        else:
            logger.warning(f"不支持的时间级别 '{tf_str}' 进行重采样。")
            return None

    def _get_aggregation_source_tfs(self, target_tf: str, all_known_tfs: Set[str]) -> List[str]:
        """
        根据目标时间级别，确定所有可能用于聚合的更小时间级别。
        例如：对于 '30' 分钟，可能返回 ['5', '15']。对于 'D'，可能返回 ['5', '15', '30', '60']。
        Args:
            target_tf (str): 目标时间级别字符串。
            all_known_tfs (Set[str]): 策略所需的所有时间级别集合。
        Returns:
            List[str]: 用于聚合的源时间级别字符串列表，按分钟数从小到大排序。
        """
        target_minutes = self._get_timeframe_in_minutes(target_tf)
        if target_minutes is None:
            return []

        potential_sources = []
        for tf_str in all_known_tfs:
            source_minutes = self._get_timeframe_in_minutes(tf_str)
            # 确保源时间级别存在且小于目标时间级别
            if source_minutes is not None and source_minutes < target_minutes:
                potential_sources.append(tf_str)

        # 按分钟数从小到大排序，确保优先使用更精细的数据进行聚合
        potential_sources.sort(key=lambda tf: self._get_timeframe_in_minutes(tf) or float('inf'))
        print(f"Debug: 为目标时间级别 '{target_tf}' 找到的潜在聚合源: {potential_sources}") # 调试信息
        return potential_sources

    def _calculate_needed_bars_for_tf(self, target_tf: str, min_tf: str, base_needed_bars: int, global_max_lookback: int) -> int:
        """
        计算目标时间级别需要从 DAO 获取的原始 K 线数量。
        获取数量应足够覆盖基础时间级别所需的最早时间点及所有指标的最大回看期。

        Args:
            target_tf (str): 目标时间级别 (例如 '15', 'D')。
            min_tf (str): 基础（最小）时间级别 (例如 '5')。
            base_needed_bars (int): 基础时间级别要求的 K 线数量。
            global_max_lookback (int): 所有指标计算所需的最大回看期。

        Returns:
            int: 目标时间级别应获取的 K 线数量 (作为 DAO 的 limit 参数)。
        """
        total_duration_bars_at_min_tf = base_needed_bars + global_max_lookback
        min_tf_minutes = self._get_timeframe_in_minutes(min_tf)
        target_tf_minutes = self._get_timeframe_in_minutes(target_tf)
        
        if min_tf_minutes is None or target_tf_minutes is None or min_tf_minutes == 0:
            logger.warning(f"无法计算 {target_tf} 的时间比例 (min={min_tf_minutes}, target={target_tf_minutes})，将仅使用指标回看期 {global_max_lookback} 作为获取数量。")
            return global_max_lookback

        estimated_total_minutes = total_duration_bars_at_min_tf * min_tf_minutes
        estimated_bars_needed = math.ceil(estimated_total_minutes / target_tf_minutes) if target_tf_minutes > 0 else total_duration_bars_at_min_tf
        
        # 基础需求是推算出的K线数和全局回看期中的较大者，并增加100条缓冲
        needed = max(estimated_bars_needed, global_max_lookback) + 100
        
        # --- 为日线及以上级别设置独立的、更长的获取周期 ---
        # 解释: A股的日、周、月线数据量不大，直接获取一个足够长的周期（如4年）
        # 可以简化逻辑，并确保任何长周期指标（如60月线）都有充足数据，
        # 避免了分钟到日级别换算的复杂性和不准确性。
        # 240分钟代表一个完整的A股交易日。
        if target_tf_minutes >= 240:
             # 对于日线、周线、月线，我们不再依赖于从分钟级别推算，
             # 而是直接请求一个固定的、足够长的周期（例如：4年 * 250交易日/年 ≈ 1000条）
             # 再加上指标回看期和一些缓冲。这比原先的 365*2 更明确且更长，保证了策略的稳健性。
             long_period_bars = 365 * 4 # 定义一个清晰的4年周期作为基准
             needed = max(needed, long_period_bars + global_max_lookback)
            #  print(f"调试信息: {target_tf} 为日线或以上级别，请求至少 {long_period_bars + global_max_lookback} 条数据。")
        
        return math.ceil(needed)

    async def _get_ohlcv_data(self, stock_code: str, time_level: Union[TimeLevel, str], needed_bars: int, trade_time: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        异步获取足够用于计算的原始历史数据 DataFrame。
        此函数仅负责从 DAO 获取，不进行时间序列对齐或质量过滤。
        Args:
            stock_code (str): 股票代码。
            time_level (Union[TimeLevel, str]): 时间级别。
            needed_bars (int): 需要获取的 K 线数量。
        Returns:
            Optional[pd.DataFrame]: 包含 OHLCV 数据的 DataFrame，如果获取失败则为 None。
        """
        limit = needed_bars
        df = await self.indicator_dao.get_history_ohlcv_df(stock_code=stock_code, time_level=time_level, limit=limit, trade_time=trade_time)
        if df is None or df.empty:
            logger.warning(f"[{stock_code}] 时间级别 {time_level} 无法获取足够的原始历史数据 (请求 {limit} 条)。")
            return None
        if 'vol' in df.columns and 'volume' not in df.columns:
            df.rename(columns={'vol': 'volume'}, inplace=True) # 重命名 'vol' 为 'volume'
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 将原始数据列名 'vol' 重命名为 'volume'.")
        if 'amount_col_from_dao' in df.columns and 'amount' not in df.columns: # 示例性的其他列名标准化
            df.rename(columns={'amount_col_from_dao': 'amount'}, inplace=True)
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 示例性重命名 'amount_col_from_dao' 为 'amount'.")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is not None:
            logger.debug(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
            logger.warning(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，但索引不是时区感知的 DatetimeIndex。")
        # print("_get_ohlcv_data.df:")
        # print(df)
        # 确定：获取K线数据
        return df

    def _resample_to_daily(self, df: pd.DataFrame, source_tf: str) -> Optional[pd.DataFrame]:
        """
        【辅助新方法】将一个时间周期小于日线的DataFrame重采样到日线级别。
        """
        if df.index.tz is None:
            df.index = df.index.tz_localize('Asia/Shanghai')
        
        # 定义聚合规则
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }
        
        # 只对存在的列进行聚合
        agg_dict = {col: rule for col, rule in agg_rules.items() if col in df.columns}
        if not agg_dict:
            logger.warning(f"在重采样时，源DataFrame不包含任何OHLCVA列。")
            return None
            
        try:
            df_daily = df.resample('D').agg(agg_dict)
            # 删除没有交易的行
            df_daily.dropna(subset=['close'], inplace=True)
            # 重命名列以包含源周期信息
            df_daily.rename(columns=lambda c: f"{c}_{source_tf}", inplace=True)
            return df_daily
        except Exception as e:
            logger.error(f"将 {source_tf} 周期数据重采样到日线时出错: {e}", exc_info=True)
            return None

    def aggregate_latest_ohlcv(df_5min: pd.DataFrame, target_tf: str) -> pd.DataFrame:
        """
        只用5分钟数据聚合出最新一根高时间级别K线（15/30/60分钟/日线）。
        返回DataFrame，index为新K线的时间戳。
        """
        rule_map = {'15': ('15T', 3), '30': ('30T', 6), '60': ('60T', 12), 'D': ('1D', None)}
        if target_tf not in rule_map:
            raise ValueError(f"不支持的聚合目标时间级别: {target_tf}")
        rule, n = rule_map[target_tf]
        df = df_5min.copy()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        if target_tf == 'D':
            # 日线聚合：取当天所有5分钟数据
            last_day = df.index[-1].date()
            df_last = df[df.index.date == last_day]
        else:
            # 取最新n根5分钟K线
            df_last = df.iloc[-n:]
        if df_last.empty:
            return pd.DataFrame()
        agg_dict = {
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum', 'amount': 'sum'
        }
        # 计算新K线的结束时间
        if target_tf == 'D':
            end_time = df_last.index[-1].replace(hour=15, minute=0, second=0, microsecond=0)
        else:
            freq_minutes = int(rule.replace('T', ''))
            end_time = df_last.index[-1].replace(second=0, microsecond=0)
            # 对齐到整15/30/60分钟
            minute = (end_time.minute // freq_minutes) * freq_minutes
            end_time = end_time.replace(minute=minute)
        # 聚合
        o = df_last['open'].iloc[0]
        h = df_last['high'].max()
        l = df_last['low'].min()
        c = df_last['close'].iloc[-1]
        v = df_last['volume'].sum()
        a = df_last['amount'].sum()
        df_new = pd.DataFrame([[o, h, l, c, v, a]], columns=['open', 'high', 'low', 'close', 'volume', 'amount'], index=[end_time])
        # === 加后缀 ===
        cols_to_suffix = ['open', 'high', 'low', 'close', 'volume', 'amount']
        rename_map_with_suffix = {col: f"{col}_{target_tf}" for col in cols_to_suffix}
        df_new = df_new.rename(columns=rename_map_with_suffix)
        return df_new

    def _resample_and_clean_dataframe(self, df: pd.DataFrame, tf: str, min_periods: int = 1, fill_method: str = 'ffill') -> Optional[pd.DataFrame]:
        """
        对原始 DataFrame 进行重采样到标准的 K 线时间点，并进行初步填充。
        此函数为同步函数，设计为可通过 asyncio.to_thread 在单独线程中运行以实现并行重采样。

        Args:
            df (pd.DataFrame): 从 DAO 获取的原始 DataFrame (索引是 DatetimeIndex, tz-aware)。
            tf (str): 目标时间级别字符串 (例如 '5', '15', 'D')。
            min_periods (int): 重采样聚合所需的最小原始数据点数量。
            fill_method (str): 重采样后填充 NaN 的方法 ('ffill', 'bfill', None)。

        Returns:
            Optional[pd.DataFrame]: 重采样并初步填充后的 DataFrame，如果失败则返回 None。
        """
        if df is None or df.empty:
            print(f"Debug: _resample_and_clean_dataframe 收到空或None的DataFrame，时间级别 {tf}。") # 调试信息
            return None
        freq = self._get_resample_freq_str(tf)
        if freq is None:
            logger.error(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 无法转换为有效的重采样频率。")
            return None
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum',
        }
        agg_rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
        if not agg_rules:
            logger.warning(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 没有找到可以聚合的列，无法进行重采样。")
            return None
        try:
            # 确保索引是时区感知的，如果不是，尝试本地化
            if not isinstance(df.index, pd.DatetimeIndex) or df.index.tzinfo is None:
                try:
                    default_tz = timezone.get_default_timezone()
                    if df.index.tzinfo is None:
                        df.index = df.index.tz_localize(default_tz)
                    else:
                        df.index = df.index.tz_convert(default_tz)
                    print(f"Debug: _resample_and_clean_dataframe: 尝试将DataFrame索引转换为默认时区。") # 调试信息
                except Exception as e:
                    logger.error(f"转换DataFrame索引时区失败: {e}", exc_info=True)
                    return None
            
            # 只对当天的数据进行聚合：resample函数本身在处理 intraday 频率时，会自然地在每日边界处创建新的时间段。
            # 例如，从 5T 到 15T，它不会将跨越午夜的 5T 数据聚合到一个 15T 的 K 线中。
            # 因此，无需额外的按天分组逻辑。
            resampled_df = df.resample(freq, label='right', closed='right').agg(agg_rules, min_periods=min_periods)
            if resampled_df.empty:
                logger.warning(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 重采样后 DataFrame 为空。")
                return None
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            required_agg_cols = [col for col in required_cols if col in agg_rules]
            if resampled_df[required_agg_cols].isnull().all().all():
                logger.warning(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 重采样后必要列全部为 NaN，数据无效。")
                return None
            if fill_method == 'ffill':
                resampled_df.ffill(inplace=True)
            elif fill_method == 'bfill':
                resampled_df.bfill(inplace=True)
            missing_after_resample_fill = resampled_df.isnull().sum().sum()
            if missing_after_resample_fill > 0:
                 logger.warning(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 重采样并初步填充后仍存在 {missing_after_resample_fill} 个缺失值。")
                 missing_cols_detail = resampled_df.isnull().mean()
                 missing_cols_detail = missing_cols_detail[missing_cols_detail > 0].sort_values(ascending=False).head()
                 if not missing_cols_detail.empty:
                     logger.warning(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 重采样后缺失比例较高的列 (初步填充后): {missing_cols_detail.to_dict()}")
            cols_to_suffix = ['open', 'high', 'low', 'close', 'volume', 'amount']
            rename_map_with_suffix = {col: f"{col}_{tf}" for col in cols_to_suffix if col in resampled_df.columns}
            resampled_df_renamed = resampled_df.rename(columns=rename_map_with_suffix) # 对列名添加时间级别后缀
            return resampled_df_renamed
        except Exception as e:
            logger.error(f"[{df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}] 时间级别 {tf} 重采样和清理数据时出错: {e}", exc_info=True)
            return None

    def _build_indicator_base_name_for_lookup(self, base_name: str, params: Dict, stock_code: str) -> Optional[str]:
        """
        根据指标名称和参数字典构建基础指标列名（不含时间级别后缀），用于在DataFrame中查找列。
        此函数需要与 calculate_* 函数实际返回的列名格式严格一致。

        Args:
            base_name (str): 指标基础名称 (例如 'RSI', 'MACD', 'K')。
            params (Dict): 指标参数字典。
            stock_code (str): 股票代码，用于日志记录。

        Returns:
            Optional[str]: 构建的指标列名查找模式，如果无法构建则为 None。
        """
        if base_name == 'RSI':
            period = params.get('period', 14)
            return f"RSI_{period}"
        elif base_name in ['MACD', 'MACDh', 'MACDs']:
            p_fast = params.get('period_fast', 12)
            p_slow = params.get('period_slow', 26)
            p_signal = params.get('signal_period', 9)
            return f"{base_name}_{p_fast}_{p_slow}_{p_signal}"
        elif base_name in ['K', 'D', 'J']:
            p_k = params.get('period', 9)
            p_d = params.get('signal_period', 3)
            p_j = params.get('smooth_k_period', 3)
            return f"{base_name}_{p_k}_{p_d}_{p_j}"
        elif base_name in ['PDI', 'NDI', 'ADX']:
             period = params.get('period', 14)
             return f"{base_name}_{period}"
        elif base_name in ['BBL', 'BBM', 'BBU', 'BBW', 'BBP']:
             p = params.get('period', 20)
             std = params.get('std_dev', 2.0)
             return f"{base_name}_{p}_{std:.1f}"
        elif base_name in ['CCI', 'MFI', 'ROC', 'ATR', 'HV', 'MOM', 'WILLR', 'VROC', 'AROC']:
            period = params.get('period', None)
            if period is not None:
                 return f"{base_name}_{period}"
            return base_name
        elif base_name in ['EMA', 'SMA', 'AMT_MA', 'VOL_MA']:
             period = params.get('period', None)
             if period is not None:
                  return f"{base_name}_{period}"
             return None
        elif base_name in ['OBV', 'ADL']:
             return base_name
        elif base_name == 'VWAP':
             anchor = params.get('anchor', None)
             if anchor:
                 return f"VWAP_{anchor}"
             return "VWAP"
        elif base_name in ['KCL', 'KCM', 'KCU']:
             ema_p = params.get('ema_period', 20)
             atr_p = params.get('atr_period', 10)
             return f"{base_name}_{ema_p}_{atr_p}"
        logger.warning(f"[{stock_code}] 无法为基础指标 {base_name} 构建列名查找模式。参数: {params}")
        return None

    def _calculate_monthly_derived_features(self, df_monthly: pd.DataFrame, lookback_months: int) -> Optional[pd.DataFrame]:
        """
        【扩展版】在纯月线DataFrame上计算衍生指标。
        此版本计算了基于收盘价、最高价、最低价的盘整期指标。

        Args:
            df_monthly (pd.DataFrame): 纯月线OHLCV数据。
            lookback_months (int): 回看周期（月数）。

        Returns:
            Optional[pd.DataFrame]: 包含衍生指标的新DataFrame，索引为月线日期。
        """
        try:
            # 检查列时，额外要求 'high' 和 'low'
            required_cols = ['open_M', 'high_M', 'low_M', 'close_M', 'volume_M', 'amount_M']
            if df_monthly.empty or not all(c in df_monthly.columns for c in required_cols):
                logger.warning(f"月线数据不完整 (需要 {required_cols})，无法计算衍生指标。")
                return None

            # 确保数据按时间升序排列
            df_monthly = df_monthly.sort_index()

            # 定义所有新列的名称
            consolidation_close_high_col = f'consolidation_high_M_{lookback_months}' # 基于收盘价的盘整高点
            avg_volume_col = f'avg_volume_M_{lookback_months}' # 平均成交量
            
            # 新增: 基于最高价和最低价的盘整区间指标
            consolidation_period_high_col = f'consolidation_period_high_M_{lookback_months}' # 期间最高价
            consolidation_period_low_col = f'consolidation_period_low_M_{lookback_months}'   # 期间最低价

            # 计算逻辑:
            # .shift(1) 确保我们获取的是【不包含当月】的过去N个月的数据。
            min_p = max(1, lookback_months // 2) # 允许在初期数据不足时也进行计算

            # 计算过去N个月的收盘价最高点 (原逻辑)
            consolidation_close_high = df_monthly['close_M'].rolling(window=lookback_months, min_periods=min_p).max()

            # 新增: 计算过去N个月的【最高价】的最高点
            consolidation_period_high = df_monthly['high_M'].rolling(window=lookback_months, min_periods=min_p).max()

            # 新增: 计算过去N个月的【最低价】的最低点
            # 注意: 这里用 .min()
            consolidation_period_low = df_monthly['low_M'].rolling(window=lookback_months, min_periods=min_p).min()

            # 计算过去N个月的平均成交量 (原逻辑)
            avg_volume = df_monthly['volume_M'].rolling(window=lookback_months, min_periods=min_p).mean()

            # 创建一个新的DataFrame来存放这些衍生指标
            derived_df = pd.DataFrame(index=df_monthly.index)
            derived_df[consolidation_close_high_col] = consolidation_close_high
            derived_df[avg_volume_col] = avg_volume
            
            # 新增: 将新计算的指标加入DataFrame
            derived_df[consolidation_period_high_col] = consolidation_period_high
            derived_df[consolidation_period_low_col] = consolidation_period_low
            
            # print(f"【调试信息】月线衍生指标计算完成。回看周期: {lookback_months}个月。新增了期间高/低点。")

            return derived_df
        except Exception as e:
            logger.error(f"计算月线衍生指标时发生内部错误: {e}", exc_info=True)
            return None

    def _calculate_weekly_derived_features(self, df_weekly: pd.DataFrame, lookback_weeks: int) -> Optional[pd.DataFrame]:
        """
        【新增方法】在纯周线DataFrame上计算衍生指标。
        此方法是 _calculate_monthly_derived_features 的周线版本。

        Args:
            df_weekly (pd.DataFrame): 纯周线OHLCV数据，列名应以 _W 结尾。
            lookback_weeks (int): 回看周期（周数）。

        Returns:
            Optional[pd.DataFrame]: 包含衍生指标的新DataFrame，索引为周线日期。
        """
        try:
            # ▼▼▼ 修改/新增 ▼▼▼
            # 1. 修改检查的列名，从 _M 后缀改为 _W 后缀
            required_cols = ['open_W', 'high_W', 'low_W', 'close_W', 'volume_W', 'amount_W']
            if df_weekly.empty or not all(c in df_weekly.columns for c in required_cols):
                # 修改日志信息
                logger.warning(f"周线数据不完整 (需要 {required_cols})，无法计算衍生指标。")
                return None

            # 确保数据按时间升序排列
            df_weekly = df_weekly.sort_index()

            # 2. 修改所有新列的名称，使用 _W 后缀和 lookback_weeks
            consolidation_close_high_col = f'consolidation_high_W_{lookback_weeks}'
            avg_volume_col = f'avg_volume_W_{lookback_weeks}'
            consolidation_period_high_col = f'consolidation_period_high_W_{lookback_weeks}'
            consolidation_period_low_col = f'consolidation_period_low_W_{lookback_weeks}'

            # 3. 修改计算逻辑的参数和列名
            # 使用 lookback_weeks 作为回看周期
            min_p = max(1, lookback_weeks // 2)

            # 从周线数据列 ('close_W', 'high_W' 等) 进行计算
            consolidation_close_high = df_weekly['close_W'].rolling(window=lookback_weeks, min_periods=min_p).max()
            consolidation_period_high = df_weekly['high_W'].rolling(window=lookback_weeks, min_periods=min_p).max()
            consolidation_period_low = df_weekly['low_W'].rolling(window=lookback_weeks, min_periods=min_p).min()
            avg_volume = df_weekly['volume_W'].rolling(window=lookback_weeks, min_periods=min_p).mean()

            # 创建一个新的DataFrame来存放这些衍生指标
            derived_df = pd.DataFrame(index=df_weekly.index)
            derived_df[consolidation_close_high_col] = consolidation_close_high
            derived_df[avg_volume_col] = avg_volume
            derived_df[consolidation_period_high_col] = consolidation_period_high
            derived_df[consolidation_period_low_col] = consolidation_period_low
            
            # print(f"【调试信息】周线衍生指标计算完成。回看周期: {lookback_weeks}周。")
            # ▲▲▲ 修改/新增 ▲▲▲

            return derived_df
        except Exception as e:
            # ▼▼▼ 修改/新增 ▼▼▼
            # 修改错误日志信息
            logger.error(f"计算周线衍生指标时发生内部错误: {e}", exc_info=True)
            # ▲▲▲ 修改/新增 ▲▲▲
            return None

    def _calculate_daily_derived_features(self, df_daily: pd.DataFrame, lookback_months: int) -> Optional[pd.DataFrame]:
        """
        【新增方法】计算日线级别的衍生特征，特别是前期高点作为结构支撑位。
        """
        if df_daily is None or df_daily.empty:
            return None
        
        # 确保列名是标准的 'high_D'
        high_col = 'high_D'
        if high_col not in df_daily.columns:
            logger.warning(f"在计算日线衍生特征时，缺少必需的列: {high_col}")
            return None
            
        try:
            df = df_daily.copy()
            
            # 计算回看窗口（大约的交易日数）
            lookback_window = lookback_months * 21  # 每月大约21个交易日
            
            # 计算N周期内的最高价，作为潜在的压力位
            df['rolling_high'] = df[high_col].rolling(window=lookback_window).max()
            
            # 核心逻辑：当前期高点被突破后，它就从压力位转变为支撑位。
            # 我们将这个转换后的支撑位向前填充，模拟其持续有效性。
            # 当价格再次创出新高时，这个支撑位会被更新。
            
            # 1. 识别突破前期高点的时刻
            #    - shift(1) 是为了确保我们用的是“前期”的高点
            is_breakout = df[high_col] > df['rolling_high'].shift(1)
            
            # 2. 在突破发生时，记录当时的压力位（即前一个rolling_high）作为新的支撑位
            #    - 使用 .where() 方法，只在突破时保留值，否则为NaN
            new_support_level = df['rolling_high'].shift(1).where(is_breakout)
            
            # 3. 向前填充这个支撑位，直到下一个新的支撑位出现
            #    - 这模拟了“一旦一个压力位被突破，它就变成了支撑，直到更强的支撑出现”
            df['prev_high_support_D'] = new_support_level.ffill()
            
            logger.info(f"成功计算日线衍生特征 'prev_high_support_D'。")
            
            # 只返回我们新计算的列，以避免与原始数据冲突
            return df[['prev_high_support_D']]
            
        except Exception as e:
            logger.error(f"计算日线衍生特征时发生错误: {e}", exc_info=True)
            return None

    # 【新增辅助方法】将指数的DAO返回结果转换为DataFrame
    def _convert_index_daily_to_df(self, index_data: List['IndexDaily'], config: Dict) -> Optional[pd.DataFrame]:
        """
        将从DAO获取的IndexDaily模型对象列表转换为一个干净的、可合并的DataFrame。
        
        Args:
            index_data (List['IndexDaily']): IndexDaily模型对象的列表。
            config (Dict): 对应的 'index_sync' 配置字典。

        Returns:
            Optional[pd.DataFrame]: 转换后的DataFrame，索引为 'trade_time'。
        """
        if not index_data:
            return None
        
        # 从配置中获取目标列名，默认为 'index_close'
        rename_to = config.get('rename_to', 'index_close')
        
        # 我们只关心收盘价，以避免数据混淆
        records = [{'trade_time': d.trade_time, rename_to: d.close} for d in index_data]
        
        df = pd.DataFrame.from_records(records)
        if df.empty:
            return None
            
        df['trade_time'] = pd.to_datetime(df['trade_time']).dt.normalize()
        df.set_index('trade_time', inplace=True)
        
        # 去重，以防数据库中有重复数据
        df = df[~df.index.duplicated(keep='last')]
        
        return df

    async def prepare_minute_centric_dataframe(self, stock_code: str, params_file: str, timeframe: str, trade_time: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """
        【新增】为指定的分钟线周期准备一个包含其自身指标的DataFrame。
        这是一个简化的数据准备函数，专门服务于多时间框架分析中的战术周期。

        Args:
            stock_code (str): 股票代码。
            params_file (str): 该分钟线周期对应的配置文件路径。
            timeframe (str): 时间周期标识符 (e.g., '60', '30')。

        Returns:
            一个元组，包含处理好的DataFrame和加载的参数。
        """
        try:
            # 1. 加载该周期的特定参数
            params = load_strategy_config(params_file)
            fe_params = params.get('feature_engineering_params', {})
            indicators_to_calc = fe_params.get('indicators', {})
            
            # 2. 获取该周期的原始OHLCV数据
            # 我们需要一个合理的K线条数，这里可以硬编码一个较大的值或从配置读取
            needed_bars = fe_params.get('base_needed_bars', 1000)
            df_minute = await self._get_ohlcv_data(stock_code, timeframe, needed_bars)
            
            if df_minute is None or df_minute.empty:
                logger.warning(f"[{stock_code}] 无法获取周期 '{timeframe}' 的数据。")
                return None, None

            # 3. 根据配置计算指标
            # 为简化，我们这里只实现MACD的计算，您可以按需扩展
            # 一个更完整的实现会复用 prepare_daily_centric_dataframe 中的指标计算逻辑
            indicator_tasks = []
            if 'macd' in indicators_to_calc:
                macd_params = indicators_to_calc['macd']
                # 假设参数格式为: {'periods': [12, 26, 9], 'apply_on': ['60']}
                p = macd_params.get('periods', [12, 26, 9])
                if len(p) == 3:
                    task = self.calculate_macd(df_minute.copy(), period_fast=p[0], period_slow=p[1], signal_period=p[2])
                    indicator_tasks.append(task)
            
            calculated_indicators = await asyncio.gather(*indicator_tasks)
            
            # 4. 合并指标到分钟线DataFrame
            df_final = df_minute
            for indi_df in calculated_indicators:
                if indi_df is not None and not indi_df.empty:
                    df_final = df_final.join(indi_df, how='left')
            
            # 确保列名不带周期后缀，因为这个df本身就代表一个周期
            df_final.rename(columns=lambda c: c.replace(f'_{timeframe}', ''), inplace=True)

            return df_final, params

        except FileNotFoundError:
            logger.error(f"[{stock_code}] 分钟线配置文件未找到: {params_file}")
            return None, None
        except Exception as e:
            logger.error(f"[{stock_code}] 为周期 '{timeframe}' 准备数据时出错: {e}", exc_info=True)
            return None, None

    # 修改方法: 重构 prepare_multi_timeframe_data 以提高效率和正确性
    async def prepare_multi_timeframe_data(
        self, 
        stock_code: str, 
        tactical_configs: Dict[str, str], 
        trade_time: Optional[str] = None,
        precomputed_daily_centric_df: Optional[pd.DataFrame] = None,
        daily_params_file: Optional[str] = None, 
        weekly_params_file: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        【V2.2 终极修正版】为多时间框架分析准备数据字典，并对周线数据执行前向填充。
        - 新增：在拆分数据前，对合并后的周线列执行ffill()，解决周间NaN问题。
        """
        all_dfs = {}
        
        # --- 步骤 1 & 2 & 3: (保持不变) ---
        daily_centric_result_df = None
        
        if precomputed_daily_centric_df is not None and not precomputed_daily_centric_df.empty:
            logger.info(f"[{stock_code}] 使用了预计算的日线中心数据。")
            daily_centric_result_df = precomputed_daily_centric_df
            daily_centric_task = None
        else:
            logger.info(f"[{stock_code}] 未提供预计算数据，将内部计算日线中心数据。")
            if not weekly_params_file:
                logger.error(f"[{stock_code}] 未提供预计算数据，也未提供 weekly_params_file，无法准备日线/周线数据。")
                return {}
            daily_centric_task = self.prepare_daily_centric_dataframe(
                stock_code=stock_code, 
                params_file=weekly_params_file, 
                trade_time=trade_time
            )
        
        tactical_tasks = {
            tf: self.prepare_minute_centric_dataframe(stock_code=stock_code, params_file=config_path, timeframe=tf, trade_time=trade_time)
            for tf, config_path in tactical_configs.items()
        }
        
        tasks_to_run = list(tactical_tasks.values())
        if daily_centric_task:
            tasks_to_run.insert(0, daily_centric_task)

        results = await asyncio.gather(*tasks_to_run, return_exceptions=True)
        
        # --- 步骤 4: 解析结果 ---
        if daily_centric_task:
            daily_centric_result = results[0]
            if isinstance(daily_centric_result, Exception) or daily_centric_result is None or daily_centric_result[0] is None:
                logger.error(f"[{stock_code}] 内部准备日线/周线数据失败，分析终止。错误: {daily_centric_result}")
                return {}
            daily_centric_result_df, _ = daily_centric_result
            tactical_results_start_index = 1
        else:
            tactical_results_start_index = 0

        if daily_centric_result_df is None or daily_centric_result_df.empty:
            logger.error(f"[{stock_code}] 缺少有效的日线中心数据，无法继续。")
            return {}

        # --- 步骤 5: 前向填充周线数据 ---
        # 在拆分DataFrame之前，对所有周线列执行前向填充(ffill)。
        # 这会将每周五的周线指标值填充到下一周的周一至周四，确保策略在任何一天都能获取到有效的周线背景。
        weekly_cols_to_fill = [col for col in daily_centric_result_df.columns if col.endswith('_W')]
        if weekly_cols_to_fill:
            daily_centric_result_df[weekly_cols_to_fill] = daily_centric_result_df[weekly_cols_to_fill].ffill()
            # 填充后，最开始的几行可能仍然是NaN（因为没有更早的数据来填充），可以选择性删除
            # daily_centric_result_df.dropna(subset=weekly_cols_to_fill, inplace=True)
            print(f"    - [数据填充] 已对 {len(weekly_cols_to_fill)} 个周线列执行前向填充(ffill)。")

        # --- 步骤 6: 拆分数据到字典 ---
        w_cols = [c for c in daily_centric_result_df.columns if '_W' in c]
        if w_cols:
            all_dfs['W'] = daily_centric_result_df[w_cols].copy()
            all_dfs['W'].rename(columns=lambda c: c.replace('_W', ''), inplace=True)
        
        non_daily_suffixes = ['_W', '_M']
        d_cols = [c for c in daily_centric_result_df.columns if not any(suffix in c for suffix in non_daily_suffixes)]
        if d_cols:
            all_dfs['D'] = daily_centric_result_df[d_cols].copy()
            all_dfs['D'].rename(columns=lambda c: c.replace('_D', ''), inplace=True)

        # --- 步骤 7: 解析分钟线结果 (保持不变) ---
        tactical_results = results[tactical_results_start_index:]
        for i, tf in enumerate(tactical_tasks.keys()):
            res = tactical_results[i]
            if isinstance(res, Exception) or res is None or res[0] is None:
                logger.warning(f"[{stock_code}] 准备周期 '{tf}' 的数据失败。错误: {res}")
                continue
            df_tactical, _ = res
            all_dfs[tf] = df_tactical
                
        return all_dfs

    async def prepare_daily_centric_dataframe(self, stock_code: str, trade_time: str, daily_config_path: Optional[str] = None, weekly_config_path: Optional[str] = None) -> pd.DataFrame:
        """
        【总指挥 V2.1 终极版】准备以日线为中心，融合了周线指标的DataFrame。
        - 修正：使用 pd.merge_asof 替代 pd.merge，正确地将周线数据对齐到日线。
        - 优化：简化了列名添加和数据填充逻辑，使其更健壮。
        """
        # 1. 加载配置
        # 加载各自独立的配置文件，明确区分日线和周线配置
        daily_strategy_config = self._load_config(daily_config_path) if daily_config_path else {}
        weekly_strategy_config = self._load_config(weekly_config_path) if weekly_config_path else {}
        daily_indicator_config = daily_strategy_config.get('feature_engineering_params', {}).get('indicators', {})
        weekly_indicator_config = weekly_strategy_config.get('feature_engineering_params', {}).get('indicators', {})

        # 打印调试信息，确认指标配置加载数量
        print(f"--- [数据准备V2.7] 日线指标配置加载数量: {len(daily_indicator_config)} 个")
        print(f"--- [数据准备V2.7] 周线指标配置加载数量: {len(weekly_indicator_config)} 个")
        
        print(f"--- [数据准备V2.0] 开始为 {stock_code} 构建多时间框架数据 ---")

        # 2. 获取基础日线数据
        # (数据公用) 获取基础日线数据
        df_daily = await self._fetch_and_prepare_base_data(stock_code, trade_time)
        if df_daily.empty:
            return pd.DataFrame()
        
        # 步骤 2.5: 获取并合并资金流和筹码数据
        print(f"--- [数据准备V2.3] 正在调用整合DAO获取资金流和筹码信息... ---")
        
        # 将字符串格式的 trade_time 转换为 datetime 对象，以匹配DAO方法的类型提示
        trade_time_dt = pd.to_datetime(trade_time) if trade_time else None

        # 单次调用新的DAO方法
        df_fund_chips = await self.strategies_dao.get_fund_flow_and_chips_data(
            stock_code=stock_code,
            trade_time=trade_time_dt
        )

        # 将获取到的整合数据合并到日线DataFrame中
        if df_fund_chips is not None and not df_fund_chips.empty:
            if df_daily.index.tz is not None:
                print(f"    - [时区统一] 检测到 df_daily 索引带有时区 ({df_daily.index.tz})，正在移除...")
                df_daily.index = df_daily.index.tz_localize(None)
            
            if df_fund_chips.index.tz is not None:
                print(f"    - [时区统一] 检测到 df_fund_chips 索引带有时区 ({df_fund_chips.index.tz})，正在移除...")
                df_fund_chips.index = df_fund_chips.index.tz_localize(None)
            # 使用左连接将资金筹码数据合并到日线数据上
            df_daily = pd.merge(df_daily, df_fund_chips, left_index=True, right_index=True, how='left')
            print("    - [信息] 已成功合并资金流与筹码的整合数据。")
            
            # 对新合并的列中因左连接产生的NaN值进行填充
            # DAO内部的ffill处理了数据内部的缺失，这里的fillna(0)处理因对齐产生的头部缺失
            new_cols = list(df_fund_chips.columns)
            df_daily[new_cols] = df_daily[new_cols].fillna(0)
            print(f"    - [信息] 已对 {len(new_cols)} 个新增资金/筹码列的NaN值填充为0。")
        else:
            print("    - [警告] 未能获取到资金流和筹码数据。")

        # 3. 计算日线指标
        # (独立计算) 调用通用计算器，传入日线数据、日线配置和 "_D" 后缀
        df_daily = await self._calculate_indicators_for_timescale(df_daily, daily_indicator_config, '_D')

        # 4. 聚合为周线
        # (数据转换) 将日线数据聚合为周线
        df_weekly = self._resample_to_weekly(df_daily)

        # 5. 计算周线指标
        # (独立计算) 调用同一个通用计算器，传入周线数据、周线配置和 "_W" 后缀
        df_weekly = await self._calculate_indicators_for_timescale(df_weekly, weekly_indicator_config, '_W')

        # 6. 为基础列和指标列统一添加后缀
        # 确保日线和周线的所有列都有正确的后缀，为合并做准备
        df_daily.columns = [f"{col}_D" if not col.endswith('_D') else col for col in df_daily.columns]
        df_weekly.columns = [f"{col}_W" if not col.endswith('_W') else col for col in df_weekly.columns]

        # 7. (数据合并) 使用 merge_asof 将周线指标精确对齐合并回日线DataFrame
        # 解释：merge_asof 是处理时间序列对齐的专业工具。
        # 对于df_daily中的每一行（每一天），它会从df_weekly中找到时间上最近的上一条记录（即上一个周五）并合并其数据。
        # 这就从根本上解决了周一到周四数据为NaN的问题。
        # print(f"--- [数据准备V2.0] 准备将 {len(df_weekly.columns)} 个周线指标合并回日线 (使用 merge_asof)... ---")
        
        # 确保两个DataFrame的索引都是排序好的DateTimeIndex
        df_daily.sort_index(inplace=True)
        df_weekly.sort_index(inplace=True)

        df_final = pd.merge_asof(
            left=df_daily,
            right=df_weekly,
            left_index=True,
            right_index=True,
            direction='backward'  # 'backward' 表示使用前一个时间点的数据填充
        )
        
        # 8. (数据清洗) 清理因计算指标和合并产生的初始NaN行
        # 找到一个可靠的、周期较短的周线指标作为基准，例如EMA_21_W
        first_reliable_weekly_col = next((col for col in df_final.columns if col.endswith('_W') and 'EMA' in col), None)
        if first_reliable_weekly_col:
            initial_len = len(df_final)
            # 删除在该列上值为NaN的行，这些通常是数据序列最开始、无法计算周线指标的部分
            df_final.dropna(subset=[first_reliable_weekly_col], inplace=True)
            # print(f"    - [数据清洗] 已移除初始 {initial_len - len(df_final)} 行无周线数据的记录。")
        
        print(f"--- [数据准备V2.0] 数据准备完成 ---")
        return df_final

    async def _fetch_and_prepare_base_data(self, stock_code: str, trade_time: str) -> Optional[pd.DataFrame]:
        """
        【异步重构】使用注入的 indicator_dao 获取并准备基础日线数据。
        """
        print("  [步骤1] 正在异步获取基础日线数据...")
        # 解释：设置一个足够大的 limit 以计算所有可能的指标，例如500个交易日（约2年）。
        needed_bars = 500 
        
        # 解释：使用 await 调用您提供的异步方法。
        # time_level='D' 表示获取日线数据。
        df = await self.indicator_dao.get_history_ohlcv_df(
            stock_code=stock_code, 
            time_level='D', # 指定获取日线数据
            limit=needed_bars, 
            trade_time=trade_time
        )
        
        if df is None or df.empty:
            logger.warning(f"[{stock_code}] 无法从 indicator_dao 获取日线数据。")
            return None
            
        # 解释：复用您代码中的列名标准化逻辑，确保数据格式统一。
        if 'vol' in df.columns and 'volume' not in df.columns:
            df.rename(columns={'vol': 'volume'}, inplace=True)
        
        # 解释：确保数据有 DatetimeIndex，这是后续所有时间序列操作的基础。
        if not isinstance(df.index, pd.DatetimeIndex):
             # 如果索引不是datetime，假设有一个'trade_date'列
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
            else:
                logger.error(f"[{stock_code}] 获取的数据既没有DatetimeIndex，也没有'trade_date'列。")
                return None

        logger.info(f"[{stock_code}] 成功获取 {len(df)} 条日线数据。")
        return df

    def _resample_to_weekly(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.0 修正版】将日线数据聚合为周线数据。
        - 修正：在聚合后，为所有列统一添加 '_W' 后缀，确保数据一致性。
        - 优化：使用 dropna(how='all')，仅在某一周完全没有数据时才删除，更加健壮。
        """
        print("  [步骤3] 正在将日线聚合为周线...")
        # 'W-FRI' 表示以周五为每周的结束点，更符合A股交易习惯
        ohlc_dict = {
            'open': 'first', 
            'high': 'max', 
            'low': 'min', 
            'close': 'last', 
            'volume': 'sum'
        }
        df_weekly = df_daily.resample('W-FRI').agg(ohlc_dict)

        # 1. 为所有聚合后的列添加 '_W' 后缀，实现命名统一
        df_weekly.columns = [f"{col}_W" for col in df_weekly.columns]
        print(f"    - 已将周线基础列重命名为: {list(df_weekly.columns)}") # 增加一条调试信息

        # 2. 优化dropna逻辑，只删除所有数据都为NaN的行（例如国庆长假所在的周）
        df_weekly.dropna(how='all', inplace=True)

        return df_weekly

    async def _calculate_indicators_for_timescale(self, df: pd.DataFrame, config: dict, suffix: str) -> pd.DataFrame:
        """
        【V4.6 列名匹配修正版】确保标准指标生成的列名与复合指标的依赖项完全匹配。
        - 核心修正：在第一个循环中，每计算完一个标准指标，就立刻将其结果（不带后缀）更新回 df_for_calc，
                    从而确保后续的指标计算（包括复合指标）可以找到其依赖项。
        """
        timescale_name = "D" if suffix == '_D' else "W"
        print(f"  [步骤 {2 if timescale_name == 'D' else 4}] 正在计算 {timescale_name} 线指标 (V4.6 最终修正版)...")
        if not config:
            print(f"    - 警告: {timescale_name} 线没有配置任何指标。")
            return df

        # 1. 创建初始计算副本
        df_for_calc = df.copy()
        if suffix == '_W':
            rename_map = {col: col.replace(suffix, '') for col in df_for_calc.columns if col.endswith(suffix)}
            df_for_calc.rename(columns=rename_map, inplace=True)
            print(f"    - [周线适配] 已临时重命名 {len(rename_map)} 个列以进行计算 (e.g., 'close_W' -> 'close')")

        # 假设所有指标方法都已在类中定义
        indicator_method_map = {
            'ema': self.calculate_ema,
            'vol_ma': self.calculate_vol_ma,
            'trix': self.calculate_trix,
            'coppock': self.calculate_coppock,
            'rsi': self.calculate_rsi,
            'macd': self.calculate_macd,
            'dmi': self.calculate_dmi,
            'roc': self.calculate_roc,
            'boll_bands_and_width': self.calculate_boll_bands_and_width,
            'cmf': self.calculate_cmf,
            'bias': self.calculate_bias,
            'atrn': self.calculate_atrn,
            'atrr': self.calculate_atrr,
            'obv': self.calculate_obv,
            'kdj': self.calculate_kdj,
            'uo': self.calculate_uo,
            'consolidation_period': self.calculate_consolidation_period,
            'advanced_fund_features': self.calculate_advanced_fund_features,
        }
        
        composite_indicator_keys = ['consolidation_period', 'advanced_fund_features']
        composite_indicators_config = {}
        
        # 2. 循环计算所有标准指标，并实时更新计算副本
        for indicator_key, params in config.items():
            indicator_name = indicator_key.lower()

            if indicator_name in composite_indicator_keys:
                composite_indicators_config[indicator_name] = params
                continue

            if indicator_name in ['说明', 'index_sync'] or not params.get('enabled', False):
                continue
            if "apply_on" in params and suffix.strip('_') not in params["apply_on"]:
                continue
            if indicator_name not in indicator_method_map:
                print(f"    - 警告: 未在映射表中找到指标 '{indicator_name}' 的计算方法，已跳过。")
                continue
            
            method_to_call = indicator_method_map[indicator_name]
            try:
                periods = params.get('periods')
                if periods is None: # 适用于 OBV 等无 period 参数的指标
                    result_df = await method_to_call(df=df_for_calc)
                    if result_df is not None and not result_df.empty:
                        # ▼▼▼【核心修正】▼▼▼
                        # 将新计算的列添加到 df_for_calc (无后缀) 和主 df (有后缀)
                        for col in result_df.columns:
                            df_for_calc[col] = result_df[col] # 更新计算副本，供后续依赖
                            df[f"{col}{suffix}"] = result_df[col] # 更新最终结果
                        # ▲▲▲【核心修正】▲▲▲
                    else:
                        print(f"    - 警告: {indicator_name.upper()} 计算返回空值。")
                    continue
                
                is_nested_list = isinstance(periods[0], list)
                periods_to_process = [periods] if indicator_name in ['macd', 'trix', 'coppock', 'kdj', 'uo'] and not is_nested_list else periods

                for p_set in periods_to_process:
                    kwargs = {'df': df_for_calc}
                    if indicator_name == 'macd': kwargs.update({'period_fast': p_set[0], 'period_slow': p_set[1], 'signal_period': p_set[2]})
                    elif indicator_name == 'trix': kwargs.update({'period': p_set[0], 'signal_period': p_set[1]})
                    elif indicator_name == 'coppock': kwargs.update({'long_roc_period': p_set[0], 'short_roc_period': p_set[1], 'wma_period': p_set[2]})
                    elif indicator_name == 'kdj': kwargs.update({'period': p_set[0], 'signal_period': p_set[1], 'smooth_k_period': p_set[2]})
                    elif indicator_name == 'uo': kwargs.update({'short_period': p_set[0], 'medium_period': p_set[1], 'long_period': p_set[2]})
                    elif indicator_name == 'boll_bands_and_width':
                        std_val = float(params.get('std', 2.0)) # 兼容JSON中的'std'
                        kwargs.update({'period': p_set, 'std_dev': std_val}) # 假设函数签名为std_dev
                    else: # 适用于 vol_ma, roc, ema 等
                        kwargs['period'] = p_set
                    
                    result_df = await method_to_call(**kwargs)
                    if result_df is not None and not result_df.empty:
                        # ▼▼▼【核心修正】▼▼▼
                        # 同样，将新计算的列添加到 df_for_calc 和主 df 中
                        for col in result_df.columns:
                            df_for_calc[col] = result_df[col] # 更新计算副本，供后续依赖
                            df[f"{col}{suffix}"] = result_df[col] # 更新最终结果
                        # ▲▲▲【核心修正】▲▲▲
                    else:
                        print(f"    - 警告: {indicator_name.upper()} (参数: {p_set}) 计算返回空值。")
            except Exception as e:
                print(f"    - 计算指标 {indicator_name.upper()}{suffix} 时出错: {e}")
                traceback.print_exc()

        # 3. (逻辑已优化) 不再需要刷新df_for_calc，因为它在循环中已被实时更新
        if composite_indicators_config:
            print("    - [复合指标适配] 计算用DataFrame已包含所有标准指标，准备计算复合指标。")

        # 4. 循环计算所有复合指标
        if composite_indicators_config:
            for indicator_name, params in composite_indicators_config.items():
                if not params.get('enabled', False): continue
                if "apply_on" in params and suffix.strip('_') not in params["apply_on"]: continue

                method_to_call = indicator_method_map[indicator_name]
                try:
                    # 现在传递给函数的 df_for_calc 是最新的，包含了所有依赖项
                    kwargs = {'df': df_for_calc, 'params': params, 'suffix': suffix}
                    result_df = await method_to_call(**kwargs)

                    if result_df is not None and not result_df.empty:
                        for col in result_df.columns:
                            if f"{col}{suffix}" not in df.columns:
                                df[f"{col}{suffix}"] = result_df[col]
                            else:
                                df[f"{col}{suffix}"].update(result_df[col])
                        print(f"    - [成功] 复合指标 {indicator_name.upper()} 计算完成。")

                except Exception as e:
                    print(f"    - [严重警告] 复合指标 {indicator_name.upper()}{suffix} 计算过程中发生异常: {e}")
                    traceback.print_exc()
        return df

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, base_needed_bars: Optional[int] = None, trade_time: Optional[str] = None) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        根据策略 JSON 配置文件准备包含重采样基础数据和所有计算指标的 DataFrame。
        此方法通过并行化数据获取、重采样和指标计算来优化性能。

        执行步骤:
        1. 加载策略参数。
        2. 解析参数，识别时间级别和指标。
        3. 估算数据量。
        4. **修改点: 顺序获取/聚合原始 OHLCV 数据，优先从数据库获取，否则从更小时间级别聚合。**
        5. 并行计算所有配置的基础指标。
        6. 合并所有数据。
        7. 补充外部特征。
        8. 计算衍生特征。
        9. 最终缺失值填充。

        Args:
            stock_code (str): 股票代码。
            params_file (str): 策略 JSON 配置文件路径。
            base_needed_bars (Optional[int]): 基础所需最小时间级别数据条数，覆盖文件配置。

        Returns:
            Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]: 包含所有数据的 DataFrame 和指标配置列表，
                                                                如果准备失败则返回 None。
        """
        if 'ta' not in globals() or ta is None:
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None, None
        try:
            if not os.path.exists(params_file):
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None, None
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
        except Exception as e:
            logger.error(f"[{stock_code}] 加载或解析参数文件 {params_file} 失败: {e}", exc_info=True)
            return None, None
        main_index_codes = params.get('base_scoring', {}).get('main_index_codes', [])
        if not main_index_codes:
            logger.warning(f"[{stock_code}] JSON 参数中未配置 'base_scoring.main_index_codes'，相对强度计算将仅使用股票所属板块作为基准。")
        fe_params = params.get('feature_engineering_params', {})
        external_data_history_days = fe_params.get('external_data_history_days', 365)
        logger.info(f"[{stock_code}] 外部特征历史数据天数 (从 JSON 读取): {external_data_history_days}")
        all_time_levels_needed: Set[str] = set()
        indicator_configs: List[Dict[str, Any]] = []
        def _add_indicator_config(name: str, func: Callable, param_block_key: Optional[str], params_dict: Dict, applicable_tfs: Union[str, List[str]], param_override_key: Optional[str] = None):
            tfs = [applicable_tfs] if isinstance(applicable_tfs, str) else applicable_tfs
            all_time_levels_needed.update(tfs)
            indicator_configs.append({
                'name': name,
                'func': func,
                'params': params_dict,
                'timeframes': tfs,
                'param_block_key': param_block_key,
                'param_override_key': param_override_key
            })
        def _get_indicator_params(param_block: Dict, default_params: Dict, param_override_key: Optional[str] = None) -> Dict:
            indi_specific_params_json = param_block.get(param_override_key, param_block) if param_override_key else param_block
            final_calc_params = default_params.copy()
            for k, v_json in indi_specific_params_json.items():
                if k in final_calc_params:
                    final_calc_params[k] = v_json
            return final_calc_params
        bs_params = params.get('base_scoring', {})
        bs_timeframes = bs_params.get('timeframes', ['5', '15', '30', '60', 'D'])
        all_time_levels_needed.update(bs_timeframes)
        default_macd_p = {'period_fast': 12, 'period_slow': 26, 'signal_period': 9}
        default_rsi_p = {'period': 14}
        default_kdj_p = {'period': 9, 'signal_period': 3, 'smooth_k_period': 3}
        default_boll_p = {'period': 15, 'std_dev': 2.2}
        default_cci_p = {'period': 14}
        default_mfi_p = {'period': 14}
        default_roc_p = {'period': 12}
        default_dmi_p = {'period': 14}
        default_sar_p = {'af_step': 0.02, 'max_af': 0.2}
        default_stoch_p = {'k_period': 14, 'd_period': 3, 'smooth_k_period': 3}
        default_atr_p = {'period': 14}
        default_hv_p = {'period': 20, 'annual_factor': 252}
        default_kc_p = {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0}
        default_mom_p = {'period': 10}
        default_willr_p = {'period': 14}
        default_sma_ema_p = {'period': 20}
        default_ichimoku_p = {'tenkan_period': 9, 'kijun_period': 26, 'senkou_period': 52}
        for indi_key in bs_params.get('score_indicators', []):
            if indi_key == 'macd':
                macd_calc_params = {
                    'period_fast': bs_params.get('macd_fast', default_macd_p['period_fast']),
                    'period_slow': bs_params.get('macd_slow', default_macd_p['period_slow']),
                    'signal_period': bs_params.get('macd_signal', default_macd_p['signal_period'])
                }
                _add_indicator_config('MACD', self.calculate_macd, 'base_scoring', macd_calc_params, bs_timeframes)
            elif indi_key == 'rsi':
                calc_params = {'period': bs_params.get('rsi_period', default_rsi_p['period'])}
                _add_indicator_config('RSI', self.calculate_rsi, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'kdj':
                calc_params = {
                    'period': bs_params.get('kdj_period_k', default_kdj_p['period']),
                    'signal_period': bs_params.get('kdj_period_d', default_kdj_p['signal_period']),
                    'smooth_k_period': bs_params.get('kdj_period_j', default_kdj_p['smooth_k_period'])
                }
                _add_indicator_config('KDJ', self.calculate_kdj, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'boll':
                calc_params = {
                    'period': bs_params.get('boll_period', default_boll_p['period']),
                    'std_dev': bs_params.get('boll_std_dev', default_boll_p['std_dev'])
                }
                _add_indicator_config('BOLL', self.calculate_boll_bands_and_width, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'cci':
                calc_params = {'period': bs_params.get('cci_period', default_cci_p['period'])}
                _add_indicator_config('CCI', self.calculate_cci, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'mfi':
                calc_params = {'period': bs_params.get('mfi_period', default_mfi_p['period'])}
                _add_indicator_config('MFI', self.calculate_mfi, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'roc':
                calc_params = {'period': bs_params.get('roc_period', default_roc_p['period'])}
                _add_indicator_config('ROC', self.calculate_roc, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'dmi':
                calc_params = {'period': bs_params.get('dmi_period', default_dmi_p['period'])}
                _add_indicator_config('DMI', self.calculate_dmi, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'sar':
                calc_params = {
                    'af_step': bs_params.get('sar_step', default_sar_p['af_step']),
                    'max_af': bs_params.get('sar_max', default_sar_p['max_af'])
                }
                _add_indicator_config('SAR', self.calculate_sar, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'ema':
                ema_p = bs_params.get('ema_period', default_sma_ema_p['period'])
                ema_calc_params = {'period': ema_p}
                _add_indicator_config('EMA', self.calculate_ema, 'base_scoring', ema_calc_params, bs_timeframes, param_override_key='ema_params')
            elif indi_key == 'sma':
                sma_p = bs_params.get('sma_period', default_sma_ema_p['period'])
                sma_calc_params = {'period': sma_p}
                _add_indicator_config('SMA', self.calculate_sma, 'base_scoring', sma_calc_params, bs_timeframes, param_override_key='sma_params')
        vc_params = params.get('volume_confirmation', {})
        ia_params = params.get('indicator_analysis_params', {})
        vol_ana_tf_cfg = vc_params.get('timeframes', bs_timeframes)
        vol_ana_tfs_vc = [vol_ana_tf_cfg] if isinstance(vol_ana_tf_cfg, str) else vol_ana_tf_cfg if vc_params.get('enabled', False) else []
        ia_tfs_cfg = ia_params.get('timeframes', bs_timeframes)
        ia_tfs = [ia_tfs_cfg] if isinstance(ia_tfs_cfg, str) else ia_tfs_cfg if ia_params else []
        target_vol_ana_tfs = list(set(vol_ana_tfs_vc) | set(ia_tfs) | set(bs_timeframes))
        all_time_levels_needed.update(target_vol_ana_tfs)
        if vc_params.get('enabled', False) or ia_params.get('calculate_amt_ma', False):
            amt_ma_p = vc_params.get('amount_ma_period', ia_params.get('amount_ma_period', 20))
            amt_ma_calc_params = {'period': amt_ma_p}
            _add_indicator_config('AMT_MA', self.calculate_amount_ma, 'volume_confirmation' if vc_params.get('enabled', False) else 'indicator_analysis_params', amt_ma_calc_params, target_vol_ana_tfs, param_override_key='amount_ma_params')
        if vc_params.get('enabled', False) or ia_params.get('calculate_cmf', False):
            cmf_p = vc_params.get('cmf_period', ia_params.get('cmf_period', 20))
            cmf_calc_params = {'period': cmf_p}
            _add_indicator_config('CMF', self.calculate_cmf, 'volume_confirmation' if vc_params.get('enabled', False) else 'indicator_analysis_params', cmf_calc_params, target_vol_ana_tfs, param_override_key='cmf_params')
        if ia_params.get('calculate_vol_ma', False):
            vol_ma_p = ia_params.get('volume_ma_period', 20)
            vol_ma_calc_params = {'period': vol_ma_p}
            _add_indicator_config('VOL_MA', self.calculate_vol_ma, 'indicator_analysis_params', vol_ma_calc_params, target_vol_ana_tfs, param_override_key='volume_ma_params')
        ia_timeframes = ia_params.get('timeframes', bs_timeframes)
        all_time_levels_needed.update(ia_timeframes)
        if ia_params.get('calculate_stoch', False):
            stoch_calc_params = _get_indicator_params(ia_params, default_stoch_p, param_override_key='stoch_params')
            _add_indicator_config('STOCH', self.calculate_stoch, 'indicator_analysis_params', stoch_calc_params, ia_timeframes)
        if ia_params.get('calculate_vwap', False):
            vwap_calc_params = _get_indicator_params(ia_params, {'anchor': None}, param_override_key='vwap_params')
            _add_indicator_config('VWAP', self.calculate_vwap, 'indicator_analysis_params', vwap_calc_params, ia_timeframes)
        if ia_params.get('calculate_adl', False):
            _add_indicator_config('ADL', self.calculate_adl, 'indicator_analysis_params', {}, ia_timeframes, param_override_key='adl_params')
        if ia_params.get('calculate_ichimoku', False):
            ichimoku_calc_params = _get_indicator_params(ia_params, default_ichimoku_p, param_override_key='ichimoku_params')
            _add_indicator_config('Ichimoku', self.calculate_ichimoku, 'indicator_analysis_params', ichimoku_calc_params, ia_timeframes)
        if ia_params.get('calculate_pivot_points', False):
            pivot_calc_params = _get_indicator_params(ia_params, {}, param_override_key='pivot_params')
            _add_indicator_config('PivotPoints', self.calculate_pivot_points, 'indicator_analysis_params', pivot_calc_params, ['D'])
            all_time_levels_needed.add('D')
        fe_timeframes_cfg = fe_params.get('apply_on_timeframes', bs_timeframes)
        fe_timeframes = [fe_timeframes_cfg] if isinstance(fe_timeframes_cfg, str) else fe_timeframes_cfg if fe_params else []
        all_time_levels_needed.update(fe_timeframes)
        if fe_params.get('calculate_atr', False):
            atr_calc_params = _get_indicator_params(fe_params, default_atr_p, param_override_key='atr_params')
            _add_indicator_config('ATR', self.calculate_atr, 'feature_engineering_params', atr_calc_params, fe_timeframes)
        if fe_params.get('calculate_hv', False):
            hv_calc_params = _get_indicator_params(fe_params, default_hv_p, param_override_key='hv_params')
            _add_indicator_config('HV', self.calculate_historical_volatility, 'feature_engineering_params', hv_calc_params, fe_timeframes)
        if fe_params.get('calculate_kc', False):
            kc_calc_params = _get_indicator_params(fe_params, default_kc_p, param_override_key='kc_params')
            _add_indicator_config('KC', self.calculate_keltner_channels, 'feature_engineering_params', kc_calc_params, fe_timeframes)
        if fe_params.get('calculate_mom', False):
            mom_calc_params = _get_indicator_params(fe_params, default_mom_p, param_override_key='mom_params')
            _add_indicator_config('MOM', self.calculate_mom, 'feature_engineering_params', mom_calc_params, fe_timeframes)
        if fe_params.get('calculate_willr', False):
            willr_calc_params = _get_indicator_params(fe_params, default_willr_p, param_override_key='willr_params')
            _add_indicator_config('WILLR', self.calculate_willr, 'feature_engineering_params', willr_calc_params, fe_timeframes)
        if fe_params.get('calculate_vroc', False):
            vroc_calc_params = _get_indicator_params(fe_params, default_roc_p, param_override_key='vroc_params')
            _add_indicator_config('VROC', self.calculate_volume_roc, 'feature_engineering_params', vroc_calc_params, fe_timeframes)
        if fe_params.get('calculate_aroc', False):
            aroc_calc_params = _get_indicator_params(fe_params, default_roc_p, param_override_key='aroc_params')
            _add_indicator_config('AROC', self.calculate_amount_roc, 'feature_engineering_params', aroc_calc_params, fe_timeframes)
        for ma_type, ma_func in [('EMA', self.calculate_ema), ('SMA', self.calculate_sma)]:
            ma_periods = fe_params.get(f'{ma_type.lower()}_periods', [])
            if ma_periods:
                for p_val in ma_periods:
                    if isinstance(p_val, int) and p_val > 0:
                        ma_calc_params = {'period': p_val}
                        _add_indicator_config(ma_type, ma_func, 'feature_engineering_params', ma_calc_params, fe_timeframes)
            elif fe_params.get(f'calculate_{ma_type.lower()}', False):
                ma_p = fe_params.get(f'{ma_type.lower()}_period', default_sma_ema_p['period'])
                ma_calc_params = {'period': ma_p}
                _add_indicator_config(ma_type, ma_func, 'feature_engineering_params', ma_calc_params, fe_timeframes)
        if not any(conf['name'] == 'OBV' for conf in indicator_configs):
            _add_indicator_config('OBV', self.calculate_obv, None, {}, list(all_time_levels_needed))
        focus_tf = params.get('trend_following_params', {}).get('focus_timeframe', '30')
        min_time_level = None
        min_tf_minutes = float('inf')
        if not all_time_levels_needed:
            logger.error(f"[{stock_code}] 未能从参数文件中确定任何需要的时间级别。")
            return None, None
        sorted_time_levels = sorted(list(all_time_levels_needed), key=lambda tf: self._get_timeframe_in_minutes(tf) or float('inf'))
        if not sorted_time_levels:
            logger.error(f"[{stock_code}] 无法确定有效的最小时间级别从所需级别: {all_time_levels_needed}")
            return None, None
        min_time_level = sorted_time_levels[0]
        logger.info(f"[{stock_code}] 策略所需时间级别: {sorted_time_levels}, 最小时间级别: {min_time_level}")
        global_max_lookback = 0
        unique_configs_for_lookback = {}
        for config in indicator_configs:
            if config.get('param_block_key') is None and config['name'] != 'OBV':
                continue
            params_hashable = tuple(sorted(config['params'].items()))
            key = (config['name'], params_hashable)
            if key not in unique_configs_for_lookback:
                unique_configs_for_lookback[key] = config
        for config in unique_configs_for_lookback.values():
            current_max_period = 0
            period_keys = ['period', 'period_fast', 'period_slow', 'signal_period', 'k_period', 'd_period', 'smooth_k_period',
                           'ema_period', 'atr_period', 'tenkan_period', 'kijun_period', 'senkou_period', 'atr_multiplier']
            for p_key in period_keys:
                if p_key == 'atr_multiplier':
                    continue
                if p_key in config['params'] and isinstance(config['params'][p_key], (int, float)):
                    current_max_period = max(current_max_period, int(config['params'][p_key]))
            if config['name'] == 'MACD':
                 p_macd = config['params']
                 current_max_period = max(current_max_period, p_macd.get('period_slow',0) + p_macd.get('signal_period',0))
            if config['name'] == 'DMI' and 'period' in config['params']:
                current_max_period = max(current_max_period, int(config['params']['period'] * 2.5 + 10))
            if config['name'] == 'KC':
                p_kc = config['params']
                current_max_period = max(current_max_period, p_kc.get('ema_period', 0), p_kc.get('atr_period', 0))
            if config['name'] == 'Ichimoku':
                p_ichi = config['params']
                current_max_period = max(current_max_period, p_ichi.get('tenkan_period', 0), p_ichi.get('kijun_period', 0), p_ichi.get('senkou_period', 0))
            if config['name'] in ['EMA', 'SMA']:
                current_max_period = max(current_max_period, config['params'].get('period', 0))
            if config['name'] in ['AMT_MA', 'VOL_MA', 'MOM', 'WILLR', 'VROC', 'AROC', 'RSI', 'CCI', 'MFI', 'ROC', 'ATR', 'SAR']:
                current_max_period = max(current_max_period,
                                           config['params'].get('period', 0),
                                           config['params'].get('k_period', 0),
                                           config['params'].get('atr_period', 0))
            if config['name'] == 'KDJ':
                p_kdj = config['params']
                current_max_period = max(current_max_period, p_kdj.get('period', 0), p_kdj.get('signal_period', 0), p_kdj.get('smooth_k_period', 0))
            global_max_lookback = max(global_max_lookback, current_max_period)
        global_max_lookback += 100
        logger.info(f"[{stock_code}] 动态计算的全局指标最大回看期 (含缓冲): {global_max_lookback}")
        lstm_window_size = params.get('lstm_training_config',{}).get('lstm_window_size', 60)
        effective_base_needed_bars = base_needed_bars if base_needed_bars is not None else \
                                     lstm_window_size + global_max_lookback + 500
        min_usable_bars = math.ceil(effective_base_needed_bars * 0.6)
        final_ohlcv_dfs: Dict[str, pd.DataFrame] = {}
        for tf_process in sorted_time_levels:
            needed_bars_for_tf = self._calculate_needed_bars_for_tf(
                target_tf=tf_process, min_tf=min_time_level,
                base_needed_bars=effective_base_needed_bars,
                global_max_lookback=global_max_lookback
            )
            raw_df_from_db = await self._get_ohlcv_data(stock_code=stock_code, time_level=tf_process, needed_bars=needed_bars_for_tf, trade_time=trade_time)
            processed_df_for_tf = None
            if raw_df_from_db is not None and not raw_df_from_db.empty:
                processed_df_for_tf = await asyncio.to_thread(
                    self._resample_and_clean_dataframe, raw_df_from_db, tf_process, min_periods=1, fill_method='ffill'
                )
            else:
                if min_time_level == '5' and '5' in final_ohlcv_dfs and not final_ohlcv_dfs['5'].empty and tf_process in ['15', '30', '60', 'D']:
                    print(f"[{stock_code}] 调试信息: {tf_process}级别无数据，尝试用5分钟数据聚合最新一段。")
                    df_5min = final_ohlcv_dfs['5']
                    aggregate_rows = 1  # 默认聚合前10行数据  # 修改行：添加聚合行数参数
                    if len(df_5min) >= aggregate_rows:
                        latest_df_5min = df_5min.tail(aggregate_rows)  # 使用最新的前10行数据
                    else:
                        latest_df_5min = df_5min  # 如果数据不足，使用全部可用数据
                    latest_k_df = await asyncio.to_thread(
                        self.aggregate_latest_ohlcv, latest_df_5min, tf_process  # 修改行：传递更多行数据
                    )
                    if tf_process in final_ohlcv_dfs and not final_ohlcv_dfs[tf_process].empty:
                        old_df = final_ohlcv_dfs[tf_process]
                        if not latest_k_df.empty and latest_k_df.index[0] in old_df.index:
                            old_df = old_df.drop(latest_k_df.index[0])  # 避免重复
                        processed_df_for_tf = pd.concat([old_df, latest_k_df])
                    else:
                        processed_df_for_tf = latest_k_df
                    processed_df_for_tf = await asyncio.to_thread(
                        self._resample_and_clean_dataframe, processed_df_for_tf, tf_process, min_periods=1, fill_method='ffill'
                    )
                    final_ohlcv_dfs[tf_process] = processed_df_for_tf  # 更新数据
                else:
                    potential_source_tfs = self._get_aggregation_source_tfs(tf_process, all_time_levels_needed)
                    print(f"[{stock_code}] Debug: 时间级别 {tf_process}: 数据库无数据，尝试从更小时间级别聚合。潜在源: {potential_source_tfs}")
                    for source_tf in potential_source_tfs:
                        if source_tf in final_ohlcv_dfs and not final_ohlcv_dfs[source_tf].empty:
                            print(f"[{stock_code}] Debug: 尝试从已处理的 {source_tf} 聚合到 {tf_process}。")
                            source_df_for_agg = final_ohlcv_dfs[source_tf].copy()
                            source_df_for_agg.columns = [col.replace(f'_{source_tf}', '') if col.endswith(f'_{source_tf}') else col for col in source_df_for_agg.columns]
                            processed_df_for_tf = await asyncio.to_thread(
                                self._resample_and_clean_dataframe, source_df_for_agg, tf_process, min_periods=1, fill_method='ffill'
                            )
                            if processed_df_for_tf is not None and not processed_df_for_tf.empty:
                                print(f"[{stock_code}] Debug: 成功从 {source_tf} 聚合得到 {len(processed_df_for_tf)} 条 {tf_process} 数据。")
                                break
                            else:
                                print(f"[{stock_code}] Debug: 从 {source_tf} 聚合到 {tf_process} 失败或结果为空，尝试下一个潜在源。")
                        else:
                            print(f"[{stock_code}] Debug: 潜在源 {source_tf} 未处理或为空，跳过。")
                        print("prepare_strategy_dataframe.processed_df_for_tf（聚合数据）:")
                        print(processed_df_for_tf)
            if processed_df_for_tf is None or processed_df_for_tf.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf_process} 无法获取或聚合到有效数据，跳过。")
                final_ohlcv_dfs[tf_process] = pd.DataFrame()
                if tf_process == min_time_level:
                    logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 无法获取有效数据，终止流程。")
                    return None, None
                continue
            if tf_process == min_time_level and len(processed_df_for_tf) < min_usable_bars:
                logger.error(f"[{stock_code}] 最小时间级别 {tf_process} 处理后数据量 {len(processed_df_for_tf)} 条，少于最低可用阈值 {min_usable_bars} 条。无法继续。")
                return None, None
            if len(processed_df_for_tf) < global_max_lookback * 0.5:
                logger.warning(f"[{stock_code}] 时间级别 {tf_process} 处理后数据量 {len(processed_df_for_tf)} 条，显著少于全局指标最大回看期 {global_max_lookback} 条。计算的指标可能不可靠。")
            final_ohlcv_dfs[tf_process] = processed_df_for_tf
        if min_time_level not in final_ohlcv_dfs or final_ohlcv_dfs[min_time_level].empty:
             logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 的 OHLCV 数据不可用，无法进行合并。")
             return None, None
        base_index = final_ohlcv_dfs[min_time_level].index
        logger.debug(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的 OHLCV 索引作为合并基准，数量: {len(base_index)}。")
        indicator_calculation_tasks = []
        async def _calculate_single_indicator_async(tf_calc: str, base_df_with_suffix: pd.DataFrame, config_item: Dict) -> Optional[Tuple[str, pd.DataFrame]]:
            if base_df_with_suffix is None or base_df_with_suffix.empty:
                return None, None
            df_for_ta = base_df_with_suffix.copy()
            ohlcv_map_to_std = {
                f'open_{tf_calc}': 'open', f'high_{tf_calc}': 'high', f'low_{tf_calc}': 'low',
                f'close_{tf_calc}': 'close', f'volume_{tf_calc}': 'volume', f'amount_{tf_calc}': 'amount'
            }
            actual_rename_map_to_std = {k: v for k, v in ohlcv_map_to_std.items() if k in df_for_ta.columns}
            df_for_ta.rename(columns=actual_rename_map_to_std, inplace=True)
            required_cols_for_func = set(['high', 'low', 'close'])
            if config_item['name'] in ['MFI', 'OBV', 'VWAP', 'CMF', 'VOL_MA', 'ADL', 'VROC']:
                required_cols_for_func.add('volume')
            if config_item['name'] in ['AMT_MA', 'AROC']:
                required_cols_for_func.add('amount')
            if not all(col in df_for_ta.columns for col in required_cols_for_func):
                missing_cols_str = ", ".join(list(required_cols_for_func - set(df_for_ta.columns)))
                print(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} 时，df_for_ta 缺少必要列 ({missing_cols_str})。可用: {df_for_ta.columns.tolist()}")
                return None, None
            try:
                func_params_to_pass = config_item['params'].copy()
                indicator_result_df = await config_item['func'](df_for_ta, **func_params_to_pass)
                if indicator_result_df is None:
                    print(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为 None。")
                    return None, None
                if indicator_result_df.empty:
                    print(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为空。")
                    return None, None
                if not isinstance(indicator_result_df, pd.DataFrame):
                    logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算函数未返回DataFrame (返回类型: {type(indicator_result_df)})。尝试转换。")
                    if isinstance(indicator_result_df, pd.Series):
                        series_name = indicator_result_df.name if indicator_result_df.name else config_item['name']
                        indicator_result_df = indicator_result_df.to_frame(name=series_name)
                        print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 转换为DataFrame后列名: {indicator_result_df.columns.tolist()}")
                    else:
                        logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 返回类型 {type(indicator_result_df)} 无法处理。")
                        return None, None
                result_renamed_df = indicator_result_df.rename(columns=lambda x: f"{x}_{tf_calc}")
                return (tf_calc, result_renamed_df)
            except Exception as e_calc:
                logger.error(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}", exc_info=True)
                return None, None
        for config_item_loop in indicator_configs:
            for tf_conf in config_item_loop['timeframes']:
                if tf_conf in final_ohlcv_dfs and not final_ohlcv_dfs[tf_conf].empty:
                    base_ohlcv_df_for_tf_loop = final_ohlcv_dfs[tf_conf]
                    indicator_calculation_tasks.append(
                        _calculate_single_indicator_async(tf_conf, base_ohlcv_df_for_tf_loop, config_item_loop)
                    )
                else:
                    logger.warning(f"[{stock_code}] 时间框架 {tf_conf} 在 final_ohlcv_dfs 中未找到有效数据，无法为指标 {config_item_loop['name']} 创建计算任务。")
        calculated_results_tuples = await asyncio.gather(*indicator_calculation_tasks, return_exceptions=True)
        calculated_indicators_by_tf = defaultdict(list)
        all_cols = []
        
        # 对 asyncio.gather 返回的结果进行健壮性处理
        for res_tuple_item in calculated_results_tuples:
            # 首先，处理任务中可能抛出的异常
            if isinstance(res_tuple_item, Exception):
                logger.error(f"[{stock_code}] 指标计算任务发生异常: {res_tuple_item}", exc_info=res_tuple_item)
                continue # 跳过此异常结果，继续处理下一个

            # 其次，检查返回的是否是预期的元组格式
            if not (isinstance(res_tuple_item, tuple) and len(res_tuple_item) == 2):
                logger.warning(f"[{stock_code}] 指标计算任务返回非预期结果: {res_tuple_item}")
                continue # 跳过此异常结果，继续处理下一个

            # 解包元组
            tf_res, indi_df_res = res_tuple_item

            # 在对 indi_df_res 进行任何操作（如访问 .columns）之前，
            # 必须先检查它是否是一个有效的、非空的 DataFrame。
            if indi_df_res is not None and isinstance(indi_df_res, pd.DataFrame) and not indi_df_res.empty:
                # 只有在检查通过后，才安全地执行后续操作
                all_cols.extend(list(indi_df_res.columns))
                calculated_indicators_by_tf[tf_res].append(indi_df_res)
            # 如果 indi_df_res 是 None 或空的 DataFrame，则静默跳过，不执行任何操作，避免错误。
        unique_sorted_cols = sorted(set(all_cols))
        final_df = final_ohlcv_dfs.get(min_time_level)
        if final_df is None or final_df.empty:
            logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 的 OHLCV 数据不可用，无法进行合并。")
            return None, None
        dfs_to_merge = [final_df]
        for tf_merge in sorted_time_levels:
            if tf_merge != min_time_level:
                df_to_merge = final_ohlcv_dfs.get(tf_merge)
                if df_to_merge is not None and not df_to_merge.empty:
                    dfs_to_merge.append(df_to_merge)
        for tf_indi, indi_dfs_list in calculated_indicators_by_tf.items():
            if indi_dfs_list:
                merged_indi_df_for_tf = pd.concat(indi_dfs_list, axis=1)
                if not merged_indi_df_for_tf.empty:
                    dfs_to_merge.append(merged_indi_df_for_tf)
                else:
                    logger.warning(f"[{stock_code}] 时间级别 {tf_indi} 的指标数据合并后为空。")
        if not dfs_to_merge:
            logger.error(f"[{stock_code}] 没有可合并的数据。")
            return None, None
        final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='left'), dfs_to_merge)
        if final_df is None or final_df.empty:
            logger.error(f"[{stock_code}] 合并所有 OHLCV 和指标数据后 DataFrame 为空。")
            return None, None
        final_df.ffill(inplace=True)
        final_df.bfill(inplace=True)
        # === 合并后只对技术指标用fillna(0)，K线数据用ffill/bfill ===
        indicator_cols = [col for col in final_df.columns if any(ind in col for ind in ['ADX', 'EMA', 'SMA', 'OBV', 'MACD', 'RSI'])]
        final_df[indicator_cols] = final_df[indicator_cols].fillna(0)
        kline_cols = [col for col in final_df.columns if any(col.startswith(prefix) for prefix in ['open_', 'high_', 'low_', 'close_', 'volume_', 'amount_'])]
        final_df[kline_cols] = final_df[kline_cols].ffill().bfill()

        nan_count_after_fill = final_df.isnull().sum().sum()
        if nan_count_after_fill > 0:
            pass
        else:
            logger.info(f"[{stock_code}] 最终缺失值填充完成，无剩余 NaN。")
        close_cols = [col for col in final_df.columns if col.startswith('close_')]
        # print(f"[{stock_code}] 合并后DataFrame包含的close_xx字段: {close_cols}")
        # print(final_df[close_cols].tail(8))
        # print("所有唯一列名的list（已排序）：")
        # sorted_columns = sorted(final_df.columns)
        # print(sorted_columns)
        return final_df, indicator_configs

    # --- 计算价格通道（箱体） ---
    async def calculate_price_channel(self, df: pd.DataFrame, period: int = 20) -> Optional[pd.DataFrame]:
        """
        计算价格通道（箱体）。
        使用滚动窗口计算指定周期内的最高价和最低价。

        Args:
            df (pd.DataFrame): 输入的 OHLCV DataFrame，必须包含 'high' 和 'low' 列。
            period (int): 计算通道的周期。

        Returns:
            Optional[pd.DataFrame]: 包含通道上下轨（CHANNEL_U, CHANNEL_L）的 DataFrame。
        """
        # 检查输入数据是否有效
        if df is None or df.empty or not all(c in df.columns for c in ['high', 'low']):
            logger.warning(f"计算价格通道失败：输入DataFrame无效或缺少'high'/'low'列。")
            return None
        
        try:
            # 定义一个同步函数来执行 pandas 操作，以便通过 to_thread 运行
            def _calculate(df_sync: pd.DataFrame, period_sync: int) -> pd.DataFrame:
                # 获取高点和低点序列
                high = df_sync['high']
                low = df_sync['low']
                
                # 使用 rolling().max() 计算上轨（周期内最高价）
                upper_band = high.rolling(window=period_sync, min_periods=1).max()
                # 使用 rolling().min() 计算下轨（周期内最低价）
                lower_band = low.rolling(window=period_sync, min_periods=1).min()
                
                # 创建结果DataFrame
                result_df = pd.DataFrame(index=df_sync.index)
                result_df[f'CHANNEL_U_{period_sync}'] = upper_band
                result_df[f'CHANNEL_L_{period_sync}'] = lower_band
                return result_df

            # 使用 asyncio.to_thread 在单独的线程中运行计算，避免阻塞事件循环
            return await asyncio.to_thread(_calculate, df.copy(), period)
        except Exception as e:
            logger.error(f"计算价格通道时出错 (周期={period}): {e}", exc_info=True)
            return None

    async def enrich_features(self, df: pd.DataFrame, stock_code: str, main_indices: List[str], external_data_history_days: int) -> pd.DataFrame:
        """
        为K线DataFrame批量补充指数、板块、筹码、资金流向等外部特征。
        数据获取任务将并行执行。

        Args:
            df (pd.DataFrame): 股票 OHLCV DataFrame (索引是时区感知的 pd.Timestamp)。
            stock_code (str): 股票代码。
            main_indices (List[str]): 主要市场指数代码列表。
            external_data_history_days (int): 获取外部特征数据的历史天数。

        Returns:
            pd.DataFrame: 补充了新特征的 DataFrame。
        """
        if df.empty:
            logger.warning(f"输入 DataFrame 为空，跳过特征工程 for {stock_code}")
            return df
        if not isinstance(df.index, pd.DatetimeIndex) or df.index.tzinfo is None:
             logger.error(f"输入 DataFrame 的索引不是时区感知的 DatetimeIndex for {stock_code}")
             try:
                  default_tz = timezone.get_default_timezone()
                  if df.index.tzinfo is None:
                      df.index = df.index.tz_localize(default_tz)
                  else:
                      df.index = df.index.tz_convert(default_tz)
                  logger.warning(f"尝试将输入 DataFrame 的索引转换为默认时区 for {stock_code}")
             except Exception as e:
                  logger.error(f"转换输入 DataFrame 索引时区失败 for {stock_code}: {e}", exc_info=True)
                  return df
        start_date = df.index.min().date()
        end_date = df.index.max().date()
        logger.info(f"对股票 {stock_code} 在日期范围 {start_date} 到 {end_date} 进行特征工程")
        trade_days_for_external = await self.indicator_dao.index_basic_dao.get_last_n_trade_cal_open(n=external_data_history_days, trade_date=end_date)
        if not trade_days_for_external:
             logger.warning(f"无法获取用于确定外部特征起始日期的交易日历数据 for {stock_code} (请求 {external_data_history_days} 天，基准日期 {end_date})。跳过外部特征获取。")
             return df
        external_fetch_start_date = trade_days_for_external[-1]
        logger.info(f"对股票 {stock_code} 进行特征工程，外部特征获取范围: {external_fetch_start_date} 到 {end_date}")
        ths_member_objects = await self.industry_dao.get_stock_ths_indices(stock_code)
        if ths_member_objects is None:
            logger.warning(f"get_stock_ths_indices 返回 None for stock_code={stock_code},无法获取同花顺板块信息。")
            ths_member_objects = []
        ths_codes = [m.ths_index.ts_code for m in ths_member_objects if m.ths_index]
        # logger.info(f"股票 {stock_code} 所属同花顺板块代码: {ths_codes}")
        tasks = []
        if main_indices:
             tasks.append(self.indicator_dao.get_index_daily_df(main_indices, external_fetch_start_date, end_date))
        if ths_codes:
             tasks.append(self.indicator_dao.get_ths_index_daily_df(ths_codes, external_fetch_start_date, end_date))
             tasks.append(self.indicator_dao.get_fund_flow_cnt_ths_df(ths_codes, external_fetch_start_date, end_date))
             tasks.append(self.indicator_dao.get_fund_flow_industry_ths_df(ths_codes, external_fetch_start_date, end_date))
        tasks.append(self.indicator_dao.get_stock_cyq_perf_df(stock_code, external_fetch_start_date, end_date))
        tasks.append(self.indicator_dao.get_fund_flow_daily_df(stock_code, external_fetch_start_date, end_date))
        tasks.append(self.indicator_dao.get_fund_flow_daily_ths_df(stock_code, external_fetch_start_date, end_date))
        tasks.append(self.indicator_dao.get_fund_flow_daily_dc_df(stock_code, external_fetch_start_date, end_date))
        results = await asyncio.gather(*tasks)
        index_daily_df = None
        ths_daily_df = None
        cyq_perf_df = None
        fund_flow_daily_df = None
        fund_flow_daily_ths_df = None
        fund_flow_daily_dc_df = None
        fund_flow_cnt_ths_df_raw = None
        fund_flow_industry_ths_df_raw = None
        result_index = 0
        if main_indices:
             index_daily_df = results[result_index]
             result_index += 1
        if ths_codes:
             ths_daily_df = results[result_index]
             fund_flow_cnt_ths_df_raw = results[result_index + 1]
             fund_flow_industry_ths_df_raw = results[result_index + 2]
             result_index += 3
        cyq_perf_df = results[result_index]
        fund_flow_daily_df = results[result_index + 1]
        fund_flow_daily_ths_df = results[result_index + 2]
        fund_flow_daily_dc_df = results[result_index + 3]
        
        all_indices_dfs_list = [] # 优化合并逻辑，先收集再合并
        if index_daily_df is not None and not index_daily_df.empty:
            for index_code in main_indices:
                idx_df = index_daily_df[index_daily_df['index_code'] == index_code].copy()
                if not idx_df.empty:
                    idx_df.drop(columns=['index_code'], inplace=True)
                    idx_df.columns = [f'index_{index_code.replace(".", "_").lower()}_{col}' for col in idx_df.columns]
                    all_indices_dfs_list.append(idx_df) # 添加到列表
        if ths_daily_df is not None and not ths_daily_df.empty:
            for ths_code in ths_codes:
                ths_d_df = ths_daily_df[ths_daily_df['ts_code'] == ths_code].copy()
                if not ths_d_df.empty:
                    ths_d_df.drop(columns=['ts_code'], inplace=True)
                    ths_d_df.columns = [f'ths_{ths_code.replace(".", "_").lower()}_{col}' for col in ths_d_df.columns]
                    all_indices_dfs_list.append(ths_d_df) # 添加到列表

        all_indices_df = pd.DataFrame(index=df.index if not df.empty else None) # 初始化为空DF并尝试使用主DF索引
        if all_indices_dfs_list: # 如果列表不为空，则进行合并
            # 使用 reduce 进行合并，确保列名唯一性（通过前缀已经保证大部分，suffixes 主要处理意外情况）
            all_indices_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), all_indices_dfs_list)
        
        if not all_indices_df.empty: # 检查合并后的 all_indices_df
            logger.debug(f"已获取并合并指数/板块数据，数据量: {len(all_indices_df)} 条，列数: {len(all_indices_df.columns)}")
        else:
            logger.warning(f"未获取到任何指数/板块数据 for {stock_code}")
            all_indices_df = pd.DataFrame(index=df.index if not df.empty else None)

        if cyq_perf_df is not None and not cyq_perf_df.empty:
            cyq_perf_df.drop(columns=['stock_code'], inplace=True, errors='ignore')
            cyq_perf_df.columns = [f'cyq_{col}' for col in cyq_perf_df.columns]
            logger.debug(f"已获取股票 {stock_code} 的筹码分布汇总数据，数据量: {len(cyq_perf_df)} 条，列数: {len(cyq_perf_df.columns)}")
        else:
            logger.warning(f"未获取到股票 {stock_code} 的筹码分布汇总数据")
            cyq_perf_df = pd.DataFrame(index=df.index if not df.empty else None)

        fund_flow_cnt_ths_dfs_list = [] # 优化合并逻辑
        if fund_flow_cnt_ths_df_raw is not None and not fund_flow_cnt_ths_df_raw.empty:
             logger.debug(f"已获取同花顺板块资金流向统计数据 (原始)，数据量: {len(fund_flow_cnt_ths_df_raw)} 条")
             for ths_code_ff in fund_flow_cnt_ths_df_raw['ts_code'].unique(): # 修改变量名避免冲突
                  cnt_df = fund_flow_cnt_ths_df_raw[fund_flow_cnt_ths_df_raw['ts_code'] == ths_code_ff].copy()
                  if not cnt_df.empty:
                       cnt_df.drop(columns=['ts_code'], inplace=True)
                       cnt_df.columns = [f'ff_cnt_ths_{ths_code_ff.replace(".", "_").lower()}_{col}' for col in cnt_df.columns]
                       cnt_df.set_index('trade_time', inplace=True)
                       cnt_df.sort_index(ascending=True, inplace=True)
                       fund_flow_cnt_ths_dfs_list.append(cnt_df) # 添加到列表
        
        fund_flow_cnt_ths_df_processed = pd.DataFrame(index=df.index if not df.empty else None) # 修改行
        if fund_flow_cnt_ths_dfs_list: # 修改行
            fund_flow_cnt_ths_df_processed = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), fund_flow_cnt_ths_dfs_list)
            if not fund_flow_cnt_ths_df_processed.empty:
                 logger.debug(f"已处理同花顺板块资金流向统计数据，数据量: {len(fund_flow_cnt_ths_df_processed)} 条，列数: {len(fund_flow_cnt_ths_df_processed.columns)}")
            else:
                 logger.warning(f"处理同花顺板块资金流向统计数据后 DataFrame 为空 for {ths_codes}")
        
        fund_flow_industry_ths_dfs_list = [] # 优化合并逻辑
        if fund_flow_industry_ths_df_raw is not None and not fund_flow_industry_ths_df_raw.empty:
             logger.debug(f"已获取同花顺行业资金流向统计数据 (原始)，数据量: {len(fund_flow_industry_ths_df_raw)} 条")
             for ths_code_fi in fund_flow_industry_ths_df_raw['ts_code'].unique(): # 修改变量名
                  ind_df = fund_flow_industry_ths_df_raw[fund_flow_industry_ths_df_raw['ts_code'] == ths_code_fi].copy()
                  if not ind_df.empty:
                       ind_df.drop(columns=['ts_code'], inplace=True)
                       ind_df.columns = [f'ff_ind_ths_{ths_code_fi.replace(".", "_").lower()}_{col}' for col in ind_df.columns]
                       ind_df.set_index('trade_time', inplace=True)
                       ind_df.sort_index(ascending=True, inplace=True)
                       fund_flow_industry_ths_dfs_list.append(ind_df) # 添加到列表

        fund_flow_industry_ths_df_processed = pd.DataFrame(index=df.index if not df.empty else None) # 修改行
        if fund_flow_industry_ths_dfs_list: # 修改行
            fund_flow_industry_ths_df_processed = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), fund_flow_industry_ths_dfs_list)
            if not fund_flow_industry_ths_df_processed.empty:
                 logger.debug(f"已处理同花顺行业资金流向统计数据，数据量: {len(fund_flow_industry_ths_df_processed)} 条，列数: {len(fund_flow_industry_ths_df_processed.columns)}")
            else:
                 logger.warning(f"处理同花顺行业资金流向统计数据后 DataFrame 为空 for {ths_codes}")

        external_features_dfs = [
            all_indices_df,
            cyq_perf_df,
            fund_flow_daily_df,
            fund_flow_daily_ths_df,
            fund_flow_daily_dc_df,
            fund_flow_cnt_ths_df_processed,
            fund_flow_industry_ths_df_processed
        ]
        
        valid_external_dfs = [ext_df for ext_df in external_features_dfs if ext_df is not None and not ext_df.empty] # 过滤无效DF
        merged_external_df = pd.DataFrame(index=df.index if not df.empty else None) # 修改行
        if valid_external_dfs: # 使用 reduce 合并有效DF
            merged_external_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), valid_external_dfs)
        
        if not merged_external_df.empty:
            #  logger.info(f"所有外部特征数据合并完成，数据量: {len(merged_external_df)} 条，列数: {len(merged_external_df.columns)}")
            # 确保索引类型一致性，都转为不含时区（naive）的 DatetimeIndex 进行合并，或都为带时区的
            # 主 df.index 是时区感知的，外部特征的索引通常是 naive date 转换为 tz-aware DatetimeIndex
            # DAO 层应该已经统一了外部特征的索引为时区感知的 DatetimeIndex
            # 如果没有，这里可能需要 df.index.tz_localize(None) 和 merged_external_df.index.tz_localize(None)
            # 但由于主 df 是 tz-aware，外部特征也应该是，以避免错误
            final_df = pd.merge(df, merged_external_df, left_index=True, right_index=True, how='left')
            logger.info(f"外部特征已合并到主 DataFrame，合并后数据量: {len(final_df)} 条，列数: {len(final_df.columns)}")
            return final_df
        else:
            logger.warning(f"没有获取到任何外部特征数据 for {stock_code}，返回原始 DataFrame。")
            return df

    def filter_to_period_points(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        尝试保留周期对齐的时间点。
        在引入重采样后，此函数通常不再用于原始数据过滤，可能用于重采样后的额外处理或验证。

        Args:
            df (pd.DataFrame): 输入 DataFrame。
            tf (str): 时间级别字符串。

        Returns:
            pd.DataFrame: 过滤后的 DataFrame。
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"DataFrame 索引不是 DatetimeIndex，无法进行周期点过滤: {tf}")
            return df
        if tf.isdigit() and int(tf) > 0 and int(tf) < 1440:
            period = int(tf)
            try:
                trading_hours_mask = ((df.index.time >= datetime.time(9, 30)) & (df.index.time <= datetime.time(11, 30))) | \
                                      ((df.index.time >= datetime.time(13, 0)) & (df.index.time <= datetime.time(15, 0)))
                trading_minutes = df.index[trading_hours_mask].minute
                if trading_minutes.empty:
                    logger.warning(f"在交易时段内没有找到分钟数据点，无法确定周期点: {tf}")
                    return df
                mod_counts = trading_minutes % period
                if mod_counts.empty:
                    logger.warning(f"计算分钟余数时发现空数据，无法确定周期点: {tf}")
                    return df
                most_common_mod = mod_counts.value_counts().idxmax()
                mask = (df.index.minute % period == most_common_mod) & (df.index.second == 0)
                filtered_df = df[mask]
                logger.info(f"对齐分钟时间到 {period} 分钟间隔 (模数 {most_common_mod})，原始时间点数量: {len(df)}，对齐后数量: {len(filtered_df)}，股票: {df.index.name if hasattr(df.index, 'name') else 'N/A'} {tf}")
                if filtered_df.empty and not df.empty:
                     logger.warning(f"对齐周期点 {tf} 后 DataFrame 为空，原始数据量为 {len(df)}。请检查时间戳模式和对齐逻辑。")
                return filtered_df
            except Exception as e:
                 logger.error(f"过滤周期点 {tf} 时出错: {e}", exc_info=True)
                 return df
        else:
            return df

    def _log_dataframe_missing(self, df: pd.DataFrame, stock_code: str):
        """
        记录 DataFrame 中填充后的缺失值情况，只报告存在缺失的列。

        Args:
            df (pd.DataFrame): 待检查的 DataFrame。
            stock_code (str): 股票代码，用于日志。
        """
        if df.empty:
            logger.warning(f"[{stock_code}] DataFrame 为空，无法检查缺失值。")
            return
        missing_count = df.isna().sum()
        missing_ratio = (df.isna().mean() * 100).round(2)
        non_zero_missing_count = missing_count[missing_count > 0]
        non_zero_missing_ratio = missing_ratio[non_zero_missing_count.index]
        if not non_zero_missing_count.empty:
            logger.warning(f"[{stock_code}] 合并填充后存在缺失值的列 - 数量: {non_zero_missing_count.to_dict()}")
            logger.warning(f"[{stock_code}] 合并填充后存在缺失值的列 - 比例(%): {non_zero_missing_ratio.to_dict()}")
            key_indicators = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'kdj', 'boll', 'cci', 'mfi', 'roc', 'adx', 'dmp', 'dmn', 'sar', 'stoch', 'vol_ma', 'vwap', 'amount', 'turnover_rate', 'rs_', 'lag_', 'rolling_']
            key_cols = [col for col in df.columns if any(indicator_name in col.lower() for indicator_name in key_indicators)]
            key_missing_count_filtered = non_zero_missing_count.filter(items=key_cols)
            key_missing_ratio_filtered = non_zero_missing_ratio.filter(items=key_cols)
            if not key_missing_count_filtered.empty:
                logger.warning(f"[{stock_code}] 关键列缺失数量 (仅显示有缺失的): {key_missing_count_filtered.to_dict()}")
                logger.warning(f"[{stock_code}] 关键列缺失比例(%)(仅显示有缺失的): {key_missing_ratio_filtered.to_dict()}")
            top_missing = non_zero_missing_ratio.sort_values(ascending=False).head(5)
            if not top_missing.empty:
                logger.warning(f"[{stock_code}] 缺失比例最高的5列 (仅显示有缺失的): {top_missing.to_dict()}")
        else:
            logger.info(f"[{stock_code}] 合并填充后所有列数据完整，无缺失值。")
        all_nan_rows = df.isna().all(axis=1).sum()
        if all_nan_rows > 0:
            logger.warning(f"[{stock_code}] 注意：合并填充后 DataFrame 仍存在 {all_nan_rows} 行全部为 NaN 的数据！这些行可能无法用于训练/预测。")

    # --- 所有指标计算函数 async def calculate_* ---
    async def calculate_atr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 ATR (平均真实波幅)"""
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 ATR 缺少必要列: {required_cols}。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 ATR。")
            return None
        try:
            def _sync_atr():
                return ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            atr_series = await asyncio.to_thread(_sync_atr)
            if atr_series is None or atr_series.empty:
                logger.warning(f"ATR_{period} 计算结果为空。")
                return None
            df_results = pd.DataFrame({f'ATR_{period}': atr_series}, index=df.index)
            return df_results
        except Exception as e:
            logger.error(f"计算 ATR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_atrr(self, df: pd.DataFrame, period: int = 14, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        计算 ATRR (Average True Range Ratio)。
        ATRR = ATR / Close
        """
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period:
            return None
        
        try:
            # 1. 计算 ATR
            atr_series = ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            if atr_series is None or atr_series.empty:
                return None
            
            # 2. 计算 ATRR
            close_prices = df[close_col].replace(0, np.nan) # 避免除以零
            atrr_series = atr_series / close_prices
            
            # 3. 返回DataFrame
            return pd.DataFrame({f'ATRr_{period}': atrr_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 ATRR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_atrn(self, df: pd.DataFrame, period: int = 14, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> Optional[pd.DataFrame]:
        """
        计算 ATRN (归一化平均真实波幅)。
        此方法先计算ATR，然后手动进行归一化处理 (ATR / Close)。
        """
        # 步骤1: 输入验证。ATRN需要高、低、收三列数据。
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 ATRN 失败：DataFrame 中缺少必需的列。需要 {required_cols}。")
            return None
        
        # 步骤2: 数据长度验证。
        if len(df) < period:
            logger.warning(f"计算 ATRN 失败：数据行数 {len(df)} 小于周期 {period}。")
            return None
            
        try:
            # 步骤3: 定义一个内部同步函数来执行计算。
            def _sync_atrn():
                # 核心修改：pandas_ta 没有 atrn，我们先计算 atr
                atr_series = ta.atr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
                
                if atr_series is None or atr_series.empty:
                    logger.warning(f"ATRN 计算失败：基础 ATR 计算返回了空结果。")
                    return None

                # 获取收盘价序列，并处理收盘价为0的情况，避免除零错误
                close_prices = df[close_col].replace(0, np.nan)
                
                # 手动进行归一化计算： (ATR / Close) * 100
                atrn_series = (atr_series / close_prices) * 100
                
                # 将计算结果包装成一个DataFrame，并使用标准命名
                return pd.DataFrame({f'ATRN_{period}': atrn_series})
            
            # 步骤4: 异步执行同步函数。
            atrn_df = await asyncio.to_thread(_sync_atrn)
            
            # 步骤5: 检查计算结果是否有效。
            if atrn_df is None or atrn_df.empty:
                logger.warning(f"ATRN 计算返回了空结果。")
                return None
                
            # 步骤6: 返回计算成功的DataFrame。
            return atrn_df
            
        except Exception as e:
            # 步骤7: 捕获并记录异常。
            logger.error(f"计算 ATRN (period={period}) 时出错: {e}", exc_info=True)
            return None

    async def calculate_boll_bands_and_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, close_col='close') -> Optional[pd.DataFrame]:
        """
        【V1.1 标准化版】计算布林带 (BBANDS) 及其宽度 (BBW) 和百分比B (%B)
        - 核心修正: 对 pandas-ta 返回的 BBB% (带宽百分比) 列进行标准化，将其除以 100，转换为标准比率。
        """
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"计算布林带缺少必要列: {close_col}。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的布林带。")
            return None
        try:
            def _sync_bbands():
                # 使用 ta.bbands() 直接调用，返回一个新的DataFrame
                return ta.bbands(close=df[close_col], length=period, std=std_dev, append=False)

            bbands_df = await asyncio.to_thread(_sync_bbands)
            if bbands_df is None or bbands_df.empty:
                logger.warning(f"布林带 (周期 {period}) 计算结果为空。")
                return None

            # ▼▼▼【核心修正】▼▼▼
            # pandas-ta 返回的 'BBB' 列是百分比形式，我们需要将其转换为标准比率
            # 1. 确定 pandas-ta 输出的原始列名
            bbw_source_col = f'BBB_{period}_{std_dev:.1f}'
            
            # 2. 检查该列是否存在，然后进行标准化
            if bbw_source_col in bbands_df.columns:
                print(f"  [指标标准化] 检测到 pandas-ta 的 '{bbw_source_col}' 列。")
                print(f"  [指标标准化] 将其值除以 100.0 以从百分比转换为标准比率。")
                # 核心操作：将百分比转换为比率
                bbands_df[bbw_source_col] = bbands_df[bbw_source_col] / 100.0
            # ▲▲▲【核心修正】▲▲▲

            # 3. 现在可以安全地重命名了，重命名后的 'BBW' 列将包含正确的比率值
            rename_map = {
                f'BBL_{period}_{std_dev:.1f}': f'BBL_{period}_{std_dev:.1f}',
                f'BBM_{period}_{std_dev:.1f}': f'BBM_{period}_{std_dev:.1f}',
                f'BBU_{period}_{std_dev:.1f}': f'BBU_{period}_{std_dev:.1f}',
                bbw_source_col: f'BBW_{period}_{std_dev:.1f}', # 使用源列名进行重命名
                f'BBP_{period}_{std_dev:.1f}': f'BBP_{period}_{std_dev:.1f}'
            }
            
            result_df = bbands_df.rename(columns=rename_map)
            
            # 筛选出我们需要的列
            final_columns = list(rename_map.values())
            result_df = result_df[[col for col in final_columns if col in result_df.columns]]
            
            return result_df if not result_df.empty else None
        except Exception as e:
            logger.error(f"计算布林带及宽度 (周期 {period}, 标准差 {std_dev}) 出错: {e}", exc_info=True)
            return None

    async def calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20, window_type: Optional[str] = None, close_col='close', annual_factor: int = 252) -> Optional[pd.DataFrame]:
        """计算历史波动率 (HV)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_hv():
                log_returns = np.log(df[close_col] / df[close_col].shift(1))
                hv_series = log_returns.rolling(window=period, min_periods=max(1, int(period * 0.5))).std() * np.sqrt(annual_factor)
                return pd.DataFrame({f'HV_{period}': hv_series}, index=df.index)
            return await asyncio.to_thread(_sync_hv)
        except Exception as e:
            logger.error(f"计算历史波动率 (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_keltner_channels(self, df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算肯特纳通道 (KC)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            logger.warning(f"计算肯特纳通道缺少必要列。")
            return None
        if len(df) < max(ema_period, atr_period):
            logger.warning(f"数据行数 ({len(df)}) 不足以计算肯特纳通道。")
            return None
        try:
            def _sync_kc():
                # --- 代码修改: 统一使用 ta.kc() 直接调用，其行为更可预测 ---
                return ta.kc(high=df[high_col], low=df[low_col], close=df[close_col], length=ema_period, atr_length=atr_period, scalar=atr_multiplier, mamode="ema", append=False)
            
            kc_df = await asyncio.to_thread(_sync_kc)
            
            if kc_df is None or kc_df.empty:
                logger.warning(f"肯特纳通道 (EMA周期 {ema_period}) 计算结果为空。")
                return None
            # 检查返回的列数是否符合预期
            if kc_df.shape[1] != 3:
                logger.error(f"肯特纳通道计算返回的列数不为3，实际为 {kc_df.shape[1]}。列名: {kc_df.columns.tolist()}")
                return None
            # 构建我们期望的、与原始代码意图完全一致的列名
            target_lower_col = f'KCL_{ema_period}_{atr_period}'
            target_middle_col = f'KCM_{ema_period}_{atr_period}'
            target_upper_col = f'KCU_{ema_period}_{atr_period}'
            # 创建一个新的 DataFrame，使用原始索引和我们期望的列名
            result_df = pd.DataFrame({
                target_lower_col: kc_df.iloc[:, 0], # 第1列是下轨
                target_middle_col: kc_df.iloc[:, 1], # 第2列是中轨
                target_upper_col: kc_df.iloc[:, 2]  # 第3列是上轨
            }, index=df.index)
            return result_df
        except Exception as e:
            logger.error(f"计算肯特纳通道 (EMA周期 {ema_period}, ATR周期 {atr_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cci(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 CCI (商品渠道指数)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        if len(df) < period: return None
        try:
            def _sync_cci():
                return ta.cci(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            cci_series = await asyncio.to_thread(_sync_cci)
            if cci_series is None or cci_series.empty: return None
            return pd.DataFrame({f'CCI_{period}': cci_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 CCI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cmf(self, df: pd.DataFrame, period: int = 20, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """
        【V2 修正版】计算 CMF (蔡金货币流量)。
        - 修正了列名问题，确保与上层合并逻辑兼容。
        - 简化了异步实现，使其更直接、更健壮。
        """
        required_cols = [high_col, low_col, close_col, volume_col]
        if df is None or df.empty or not all(col in df.columns for col in required_cols):
            logger.warning(f"计算 CMF (周期 {period}) 失败：输入DataFrame为空或缺少必需列 {required_cols}。")
            return None
        
        if len(df) < period:
            logger.warning(f"计算 CMF (周期 {period}) 失败：数据长度 {len(df)} 小于周期 {period}。")
            return None
            
        try:
            # 直接调用 pandas_ta，它会返回一个带有正确列名（如 'CMF_20'）的 Series
            # 我们不需要手动创建 DataFrame 或重命名列
            # print(f"调试信息: [IndicatorService] 正在为周期 {period} 计算 CMF...")
            cmf_series = ta.cmf(
                high=df[high_col], 
                low=df[low_col], 
                close=df[close_col], 
                volume=df[volume_col], 
                length=period, 
                append=False # 确保不修改原始df
            )
            
            if cmf_series is None or cmf_series.empty:
                logger.warning(f"计算 CMF (周期 {period}) 返回了空结果。")
                return None
            
            # print(f"调试信息: [IndicatorService] CMF (周期 {period}) 计算完成，结果类型: {type(cmf_series)}，列名: {cmf_series.name}")
            # 将返回的 Series 转换为 DataFrame，后续的合并逻辑会处理它
            return cmf_series.to_frame()

        except Exception as e:
            logger.error(f"计算 CMF (周期 {period}) 时发生未知异常: {e}", exc_info=True)
            return None

    async def calculate_kdj(self, df: pd.DataFrame, period: int = 9, signal_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算KDJ指标"""
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < period + signal_period:
            return None

        try:
            def _sync_stoch():
                # pandas-ta的stoch函数计算的就是KDJ
                return ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=period, d=signal_period, smooth_k=smooth_k_period, append=False)
            
            stoch_df = await asyncio.to_thread(_sync_stoch)

            # ▼▼▼【修改】增加对 None 返回值的健壮性检查 ▼▼▼
            if stoch_df is None or stoch_df.empty:
                logger.warning(f"KDJ (p={period}, sig={signal_period}, smooth={smooth_k_period}) 计算结果为空，可能数据量不足。")
                return None
            # ▲▲▲ 修改结束 ▲▲▲

            # 重命名列以符合KDJ的习惯
            # STOCHk_9_3_3 -> K_9_3_3, STOCHd_9_3_3 -> D_9_3_3
            stoch_df.rename(columns=lambda x: x.replace('STOCHk', 'K').replace('STOCHd', 'D'), inplace=True)
            
            # 计算J值: J = 3*K - 2*D
            k_col = f'K_{period}_{signal_period}_{smooth_k_period}'
            d_col = f'D_{period}_{signal_period}_{smooth_k_period}'
            j_col = f'J_{period}_{signal_period}_{smooth_k_period}'
            
            if k_col in stoch_df and d_col in stoch_df:
                stoch_df[j_col] = 3 * stoch_df[k_col] - 2 * stoch_df[d_col]
            
            return stoch_df

        except Exception as e:
            logger.error(f"计算 KDJ (p={period}, sig={signal_period}, smooth={smooth_k_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ema(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 EMA (指数移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_ema():
                return ta.ema(close=df[close_col], length=period, append=False)
            ema_series = await asyncio.to_thread(_sync_ema)
            if ema_series is None or not isinstance(ema_series, pd.Series) or ema_series.empty: return None
            return pd.DataFrame({f'EMA_{period}': ema_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 EMA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
            return None

    async def calculate_dmi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 DMI (动向指标), 包括 PDI (+DI), NDI (-DI), ADX"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        if len(df) < period: return None
        try:
            def _sync_dmi():
                return ta.adx(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            dmi_df = await asyncio.to_thread(_sync_dmi)
            if dmi_df is None or dmi_df.empty: return None
            rename_map = {
                f'DMP_{period}': f'PDI_{period}',
                f'DMN_{period}': f'NDI_{period}',
                f'ADX_{period}': f'ADX_{period}'
            }
            result_df = dmi_df.rename(columns={k: v for k, v in rename_map.items() if k in dmi_df.columns})
            return result_df
        except Exception as e:
            logger.error(f"计算 DMI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ichimoku(self, df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, name_suffix: Optional[str] = None, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """
        计算一目均衡表 (Ichimoku Cloud) 的时间对齐特征。
        Args:
            df (pd.DataFrame): 输入的K线数据。
            tenkan_period (int): 转换线周期。
            kijun_period (int): 基准线周期。
            senkou_period (int): 先行带B周期。
            name_suffix (Optional[str]): 可选的名称后缀，用于附加到所有列名后 (例如 '15', 'D')。
            ...
        Returns:
            Optional[pd.DataFrame]: 包含一目均衡表指标的DataFrame。
        """
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        if len(df) < max(tenkan_period, kijun_period, senkou_period): return None
        try:
            def _sync_ichimoku():
                # 使用 ta.ichimoku() 直接调用，获取时间对齐的特征
                # 返回的第一个元素是包含 ITS, IKS, ISA, ISB, ICS 的 DataFrame
                ichimoku_data, _ = ta.ichimoku(high=df[high_col], low=df[low_col], close=df[close_col],
                                           tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period,
                                           append=False)
                return ichimoku_data
            ichi_df = await asyncio.to_thread(_sync_ichimoku)
            if ichi_df is None or ichi_df.empty: return None
            # 源列名 (由 pandas-ta 生成)
            source_tenkan = f'ITS_{tenkan_period}'
            source_kijun = f'IKS_{kijun_period}'
            source_senkou_a = f'ISA_{tenkan_period}'
            source_senkou_b = f'ISB_{kijun_period}' # 注意: pandas-ta 使用 kijun 周期命名
            source_chikou = f'ICS_{kijun_period}'   # 注意: pandas-ta 使用 kijun 周期命名
            # 目标列名 (基础部分，与原始代码意图一致)
            target_tenkan = f'TENKAN_{tenkan_period}'
            target_kijun = f'KIJUN_{kijun_period}'
            target_senkou_a = f'SENKOU_A_{tenkan_period}'
            target_senkou_b = f'SENKOU_B_{senkou_period}' # 目标名使用 senkou 周期，更符合逻辑
            target_chikou = f'CHIKOU_{kijun_period}'
            rename_map = {
                source_tenkan: target_tenkan,
                source_kijun: target_kijun,
                source_senkou_a: target_senkou_a,
                source_senkou_b: target_senkou_b,
                source_chikou: target_chikou,
            }
            
            result_df = ichi_df.rename(columns=rename_map)
            # 筛选出我们成功重命名的列，避免携带非预期的列
            final_columns = list(rename_map.values())
            result_df = result_df[[col for col in final_columns if col in result_df.columns]]
            # 如果提供了后缀，则附加到所有列名上
            if name_suffix:
                result_df.columns = [f'{col}_{name_suffix}' for col in result_df.columns]
            return result_df if not result_df.empty else None
        except Exception as e:
            logger.error(f"计算 Ichimoku (t={tenkan_period}, k={kijun_period}, s={senkou_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sma(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 SMA (简单移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_sma():
                return ta.sma(close=df[close_col], length=period, append=False)
            sma_series = await asyncio.to_thread(_sync_sma)
            if sma_series is None or not isinstance(sma_series, pd.Series) or sma_series.empty: return None
            return pd.DataFrame({f'SMA_{period}': sma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 SMA (周期 {period}) 时发生未知错误: {e}", exc_info=True)
            return None

    async def calculate_amount_ma(self, df: pd.DataFrame, period: int = 20, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的移动平均线 (AMT_MA)"""
        if df is None or df.empty or amount_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_amount_ma():
                return df[amount_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            amt_ma_series = await asyncio.to_thread(_sync_amount_ma)
            return pd.DataFrame({f'AMT_MA_{period}': amt_ma_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 AMT_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_macd(self, df: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        """计算移动平均收敛散度 (MACD)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period_slow + signal_period:
            return None

        try:
            def _sync_macd():
                return ta.macd(close=df[close_col], fast=period_fast, slow=period_slow, signal=signal_period, append=False)
            
            macd_df = await asyncio.to_thread(_sync_macd)

            # ▼▼▼【修改】增加对 None 返回值的健壮性检查 ▼▼▼
            if macd_df is None or macd_df.empty:
                logger.warning(f"MACD (f={period_fast},s={period_slow},sig={signal_period}) 计算结果为空，可能数据量不足。")
                return None
            # ▲▲▲ 修改结束 ▲▲▲
            
            return macd_df

        except Exception as e:
            logger.error(f"计算 MACD (f={period_fast},s={period_slow},sig={signal_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mfi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 MFI (资金流量指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]): return None
        if len(df) < period: return None
        try:
            def _sync_mfi():
                return ta.mfi(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False)
            mfi_series = await asyncio.to_thread(_sync_mfi)
            if mfi_series is None or mfi_series.empty: return None
            return pd.DataFrame({f'MFI_{period}': mfi_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 MFI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mom(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 MOM (动量指标)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) < period: return None
        try:
            def _sync_mom():
                return ta.mom(close=df[close_col], length=period)
            mom_series = await asyncio.to_thread(_sync_mom)
            if mom_series is None or mom_series.empty: return None
            return pd.DataFrame({f'MOM_{period}': mom_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 MOM (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_obv(self, df: pd.DataFrame, close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 OBV (能量潮指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [close_col, volume_col]): return None
        try:
            def _sync_obv():
                return ta.obv(close=df[close_col], volume=df[volume_col], append=False)
            obv_series = await asyncio.to_thread(_sync_obv)
            if obv_series is None or obv_series.empty: return None
            return pd.DataFrame({'OBV': obv_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 OBV 出错: {e}", exc_info=True)
            return None

    async def calculate_roc(self, df: pd.DataFrame, period: int = 12, close_col='close') -> Optional[pd.DataFrame]:
        """计算 ROC (价格变化率)"""
        if df is None or df.empty or close_col not in df.columns: return None
        if len(df) <= period: return None
        try:
            def _sync_roc():
                return ta.roc(close=df[close_col], length=period, append=False)
            roc_series = await asyncio.to_thread(_sync_roc)
            if roc_series is None or roc_series.empty: return None
            return pd.DataFrame({f'ROC_{period}': roc_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_amount_roc(self, df: pd.DataFrame, period: int, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的 ROC (AROC)"""
        if df is None or df.empty or amount_col not in df.columns:
            logger.warning(f"输入 DataFrame 为空或缺少 '{amount_col}' 列，无法计算 AROC。")
            return None
        # --- 调试信息：打印输入DataFrame的形状和列名 ---
        # print(f"调试信息: [AROC_{period}] 输入 df 的形状: {df.shape}, 列: {df.columns.tolist()}")
        if len(df) <= period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 AROC。")
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_aroc():
                target_series = df[amount_col]
                # 直接调用 ta.roc 函数，传入 Series
                return ta.roc(close=target_series, length=period, append=False)
                # --- 代码修改结束 ---
            aroc_series = await asyncio.to_thread(_sync_aroc)
            if aroc_series is None or aroc_series.empty:
                logger.warning(f"AROC_{period} 计算结果为空。")
                return None
            # 将结果构建为 DataFrame，列名格式化
            df_results = pd.DataFrame({f'AROC_{period}': aroc_series})
            # 将无穷大值替换为NaN，便于后续处理
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            # 记录详细的错误信息，包括堆栈跟踪
            logger.error(f"计算 Amount ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_volume_roc(self, df: pd.DataFrame, period: int, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的 ROC (VROC)"""
        if df is None or df.empty or volume_col not in df.columns:
            logger.warning(f"输入 DataFrame 为空或缺少 '{volume_col}' 列，无法计算 VROC。")
            return None
        if len(df) <= period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 VROC。")
            return None
            
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_vroc():
                target_series = df[volume_col]
                # 直接调用 ta.roc 函数，传入 Series。注意：pandas_ta 的 roc 函数使用 'close' 作为通用输入参数名。
                return ta.roc(close=target_series, length=period)
            vroc_series = await asyncio.to_thread(_sync_vroc)
            if vroc_series is None or vroc_series.empty:
                logger.warning(f"VROC_{period} 计算结果为空。")
                return None
            df_results = pd.DataFrame({f'VROC_{period}': vroc_series}, index=df.index)
            # 将无穷大值替换为NaN，便于后续处理
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Volume ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_rsi(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算相对强弱指数 (RSI)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period:
            return None
        
        try:
            # 异步执行 pandas_ta 计算
            def _sync_rsi():
                return ta.rsi(close=df[close_col], length=period, append=False)
            rsi_series = await asyncio.to_thread(_sync_rsi)

            # ▼▼▼【修改】增加对 None 返回值的健壮性检查 ▼▼▼
            if rsi_series is None or not isinstance(rsi_series, pd.Series) or rsi_series.empty:
                logger.warning(f"RSI (周期 {period}) 计算结果为空或无效，可能数据量不足。")
                return None
            
            # 【修改】在创建DataFrame时显式传入索引，更加安全
            return pd.DataFrame({f'RSI_{period}': rsi_series}, index=df.index)
            # ▲▲▲ 修改结束 ▲▲▲

        except Exception as e:
            logger.error(f"计算 RSI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sar(self, df: pd.DataFrame, af_step: float = 0.02, max_af: float = 0.2, high_col='high', low_col='low') -> Optional[pd.DataFrame]:
        """计算 SAR (抛物线转向指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_psar():
                return df.ta.psar(high=df[high_col], low=df[low_col], af0=af_step, af=af_step, max_af=max_af, append=False)
            psar_df = await asyncio.to_thread(_sync_psar)
            if psar_df is None or psar_df.empty: return None
            long_sar_col = next((col for col in psar_df.columns if col.startswith('PSARl')), None)
            short_sar_col = next((col for col in psar_df.columns if col.startswith('PSARs')), None)
            if long_sar_col and short_sar_col:
                sar_values = psar_df[long_sar_col].fillna(psar_df[short_sar_col])
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': sar_values})
            elif long_sar_col:
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': psar_df[long_sar_col]})
            elif short_sar_col:
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': psar_df[short_sar_col]})
            else:
                logger.warning(f"计算 SAR 未找到 PSARl 或 PSARs 列。返回列: {psar_df.columns.tolist()}")
                return None
        except Exception as e:
            logger.error(f"计算 SAR (af={af_step:.2f}, max_af={max_af:.2f}) 出错: {e}", exc_info=True)
            return None

    async def calculate_stoch(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 STOCH (随机指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_stoch():
                return df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=k_period, d=d_period, smooth_k=smooth_k_period, append=False)
            stoch_df = await asyncio.to_thread(_sync_stoch)
            if stoch_df is None or stoch_df.empty: return None
            return stoch_df
        except Exception as e:
            logger.error(f"计算 STOCH (k={k_period},d={d_period},s={smooth_k_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_adl(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 ADL (累积/派发线)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_adl():
                return df.ta.ad(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], append=False)
            adl_series = await asyncio.to_thread(_sync_adl)
            if adl_series is None or adl_series.empty: return None
            return pd.DataFrame({'ADL': adl_series})
        except Exception as e:
            logger.error(f"计算 ADL 出错: {e}", exc_info=True)
            return None

    async def calculate_pivot_points(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算经典枢轴点和斐波那契枢轴点 (基于前一周期数据)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # --- 将同步的计算逻辑移至线程中执行 ---
            def _sync_pivot():
                results = pd.DataFrame(index=df.index)
                prev_high = df[high_col].shift(1)
                prev_low = df[low_col].shift(1)
                prev_close = df[close_col].shift(1)
                PP = (prev_high + prev_low + prev_close) / 3
                results['PP'] = PP
                results['S1'] = (2 * PP) - prev_high
                results['S2'] = PP - (prev_high - prev_low)
                results['S3'] = results['S1'] - (prev_high - prev_low)
                results['S4'] = results['S2'] - (prev_high - prev_low)
                results['R1'] = (2 * PP) - prev_low
                results['R2'] = PP + (prev_high - prev_low)
                results['R3'] = results['R1'] + (prev_high - prev_low)
                results['R4'] = results['R2'] + (prev_high - prev_low)
                diff = prev_high - prev_low
                results['F_R1'] = PP + 0.382 * diff
                results['F_R2'] = PP + 0.618 * diff
                results['F_R3'] = PP + 1.000 * diff
                results['F_S1'] = PP - 0.382 * diff
                results['F_S2'] = PP - 0.618 * diff
                results['F_S3'] = PP - 1.000 * diff
                return results.iloc[1:]
            return await asyncio.to_thread(_sync_pivot)
        except Exception as e:
            logger.error(f"计算 Pivot Points 出错: {e}", exc_info=True)
            return None

    async def calculate_vol_ma(self, df: pd.DataFrame, period: int = 20, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的移动平均线 (VOL_MA)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            # --- 将同步的 rolling 计算移至线程中执行 ---
            def _sync_vol_ma():
                return df[volume_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            vol_ma_series = await asyncio.to_thread(_sync_vol_ma)
            return pd.DataFrame({f'VOL_MA_{period}': vol_ma_series})
        except Exception as e:
            logger.error(f"计算 VOL_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_vwap(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume', anchor: Optional[str] = None) -> Optional[pd.DataFrame]:
        """计算 VWAP (成交量加权平均价)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_vwap():
                return df.ta.vwap(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], anchor=anchor, append=False)
            vwap_series = await asyncio.to_thread(_sync_vwap)
            if vwap_series is None or vwap_series.empty: return None
            col_name = 'VWAP' if anchor is None else f'VWAP_{anchor}'
            return pd.DataFrame({col_name: vwap_series})
        except Exception as e:
            logger.error(f"计算 VWAP (anchor={anchor}) 出错: {e}", exc_info=True)
            return None

    async def calculate_willr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算威廉姆斯 %R (WILLR)"""
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            logger.warning(f"输入 DataFrame 为空或缺少必要的列 {required_cols}，无法计算 WILLR。")
            return None
        if len(df) < period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 WILLR。")
            return None
            
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_willr():
                return ta.willr(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            willr_series = await asyncio.to_thread(_sync_willr)
            if willr_series is None or willr_series.empty:
                logger.warning(f"WILLR_{period} 计算结果为空。")
                return None
            df_results = pd.DataFrame({f'WILLR_{period}': willr_series})            
            return df_results
        except Exception as e:
            logger.error(f"计算 WILLR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    def calculate_relative_strength(self, df: pd.DataFrame, stock_close_col: str, benchmark_codes: List[str], periods: List[int], time_level: str) -> pd.DataFrame:
        """
        计算股票相对于基准指数/板块的相对强度/超额收益。
        使用对数收益率的累积差值进行计算。此为同步函数。
        Args:
            df (pd.DataFrame): 包含股票和基准数据的 DataFrame。
            stock_close_col (str): 股票收盘价列名 (已带时间级别后缀)。
            benchmark_codes (List[str]): 基准指数/板块代码列表。
            periods (List[int]): 计算相对强度的周期列表。
            time_level (str): 当前计算的时间级别。
        Returns:
            pd.DataFrame: 补充了相对强度特征的 DataFrame。
        """
        if df is None or df.empty or stock_close_col not in df.columns:
            logger.warning(f"计算相对强度失败，输入 DataFrame 无效或缺少股票收盘价列 {stock_close_col}。")
            print(f"计算相对强度失败，输入 DataFrame 无效或缺少股票收盘价列 {stock_close_col}。")
            return df
        df_processed = df.copy()
        stock_close_shifted = df_processed[stock_close_col].shift(1)
        stock_returns = np.log(df_processed[stock_close_col] / stock_close_shifted)
        stock_returns = stock_returns.replace([np.inf, -np.inf], np.nan)
        for benchmark_code in benchmark_codes:
            if '.' in benchmark_code:
                benchmark_col_prefix = f'index_{benchmark_code.replace(".", "_").lower()}_'
            else:
                benchmark_col_prefix = f'ths_{benchmark_code.replace(".", "_").lower()}_'
            benchmark_close_col = f'{benchmark_col_prefix}close'
            if benchmark_close_col in df_processed.columns:
                benchmark_close_shifted = df_processed[benchmark_close_col].shift(1)
                benchmark_returns = np.log(df_processed[benchmark_close_col] / benchmark_close_shifted)
                benchmark_returns = benchmark_returns.replace([np.inf, -np.inf], np.nan)
                for period in periods:
                    cumulative_stock_log_return = stock_returns.rolling(window=period).sum()
                    cumulative_benchmark_log_return = benchmark_returns.rolling(window=period).sum()
                    excess_log_return = cumulative_stock_log_return - cumulative_benchmark_log_return
                    rs_col_name = f'RS_{benchmark_code.replace(".", "_").lower()}_{period}_{time_level}'
                    df_processed[rs_col_name] = excess_log_return
        return df_processed

    async def calculate_trix(self, df: pd.DataFrame, period: int = 14, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        """
        计算 TRIX (三重指数平滑移动平均线) 及其信号线。
        Args:
            df (pd.DataFrame): 输入数据。
            period (int): TRIX 计算周期。
            signal_period (int): 信号线计算周期。
            close_col (str): 收盘价列名。
        Returns:
            Optional[pd.DataFrame]: 包含 TRIX 和 TRIX_signal 列的 DataFrame。
        """
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < period * 3: # TRIX 需要更长的启动数据
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {period} 的 TRIX。")
            return None
        try:
            def _sync_trix():
                # 使用 pandas-ta 直接计算 TRIX 和其信号线
                return ta.trix(close=df[close_col], length=period, signal=signal_period, append=False)
            trix_df = await asyncio.to_thread(_sync_trix)
            if trix_df is None or trix_df.empty:
                return None
            # pandas-ta 默认返回的列名是 'TRIX_14_9' 和 'TRIXs_14_9'，这已经很清晰，直接返回即可
            return trix_df
        except Exception as e:
            logger.error(f"计算 TRIX (period={period}, signal={signal_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_coppock(self, df: pd.DataFrame, long_roc_period: int = 14, short_roc_period: int = 11, wma_period: int = 10, close_col='close') -> Optional[pd.DataFrame]:
        """
        计算 Coppock Curve (估波指标)。
        Args:
            df (pd.DataFrame): 输入数据。
            long_roc_period (int): 长变化率周期。
            short_roc_period (int): 短变化率周期。
            wma_period (int): 加权移动平均周期。
            close_col (str): 收盘价列名。
        Returns:
            Optional[pd.DataFrame]: 包含 Coppock Curve 的 DataFrame。
        """
        if df is None or df.empty or close_col not in df.columns:
            return None
        if len(df) < long_roc_period + wma_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算 Coppock Curve。")
            return None
        try:
            def _sync_coppock():
                # 使用 pandas-ta 直接计算
                return ta.coppock(close=df[close_col], length=short_roc_period, long=long_roc_period, wma=wma_period, append=False)
            coppock_series = await asyncio.to_thread(_sync_coppock)
            if coppock_series is None or coppock_series.empty:
                return None
            # 返回一个标准的 DataFrame
            col_name = f'COPPOCK_{long_roc_period}_{short_roc_period}_{wma_period}'
            return pd.DataFrame({col_name: coppock_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 Coppock Curve 出错: {e}", exc_info=True)
            return None

    async def calculate_uo(self, df: pd.DataFrame, short_period: int = 7, medium_period: int = 14, long_period: int = 28, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """
        计算 Ultimate Oscillator (终极波动指标)。
        Args:
            df (pd.DataFrame): 输入数据。
            short_period (int): 短周期。
            medium_period (int): 中周期。
            long_period (int): 长周期。
            ...
        Returns:
            Optional[pd.DataFrame]: 包含 UO 指标的 DataFrame。
        """
        required_cols = [high_col, low_col, close_col]
        if df is None or df.empty or not all(c in df.columns for c in required_cols):
            return None
        if len(df) < long_period:
            logger.warning(f"数据行数 ({len(df)}) 不足以计算周期为 {long_period} 的 UO。")
            return None
        try:
            def _sync_uo():
                # 使用 pandas-ta 直接计算
                return ta.uo(high=df[high_col], low=df[low_col], close=df[close_col], fast=short_period, medium=medium_period, slow=long_period, append=False)
            uo_series = await asyncio.to_thread(_sync_uo)
            if uo_series is None or uo_series.empty:
                return None
            # 返回一个标准的 DataFrame
            col_name = f'UO_{short_period}_{medium_period}_{long_period}'
            return pd.DataFrame({col_name: uo_series}, index=df.index)
        except Exception as e:
            logger.error(f"计算 Ultimate Oscillator 出错: {e}", exc_info=True)
            return None

    async def calculate_bias(self, df: pd.DataFrame, period: int = 20, close_col='close') -> Optional[pd.DataFrame]:
        """
        【V1.3 最终修正版】计算 BIAS，并强制重命名列以符合系统标准。
        """
        # (此函数前面的调试代码和检查代码保持不变)
        if df is None or df.empty or close_col not in df.columns:
            logger.warning(f"BIAS计算失败：输入的DataFrame为空或缺少'{close_col}'列。")
            return None
        
        if len(df) < period:
            logger.warning(f"BIAS计算失败：数据长度 {len(df)} 小于所需周期 {period}。")
            return None

        try:
            def _sync_bias() -> Optional[pd.Series]:
                # pandas_ta.bias 会生成一个名为 'BIAS_SMA_{period}' 的列
                return df.ta.bias(close=df[close_col], length=period, append=False)

            bias_series = await asyncio.to_thread(_sync_bias)

            if bias_series is None or bias_series.empty:
                logger.warning(f"pandas_ta.bias 未能为周期 {period} 生成有效结果。")
                return None

            # 这是解决问题的核心：将 pandas_ta 生成的列名 'BIAS_SMA_20' 重命名为我们需要的标准格式 'BIAS_20'
            target_col_name = f"BIAS_{period}"
            bias_series.name = target_col_name

            # 将重命名后的 Series 转换为 DataFrame
            result_df = pd.DataFrame(bias_series)
            return result_df

        except Exception as e:
            logger.error(f"计算 BIAS (period={period}) 时发生未知错误: {e}", exc_info=True)
            return None
    
    async def calculate_consolidation_period(self, df: pd.DataFrame, params: Dict, suffix: str) -> Optional[pd.DataFrame]:
        """
        【V2.0 多因子共振模型】根据波动率、趋势和成交量共振识别盘整期。
        - 波动率因子: BBW低于其自身的动态历史分位数 (无未来函数)。
        - 趋势因子: ROC绝对值低于指定阈值，表示缺乏强劲趋势。
        - 成交量因子: 当前成交量低于其长期移动平均线，表示市场交投清淡。
        - 只有当三个条件同时满足时，才被识别为盘整期。
        """
        # 1. 从配置中获取V2.0模型的所有参数
        boll_period = params.get('boll_period', 20)
        boll_std = params.get('boll_std', 2.0)
        bbw_quantile = params.get('bbw_quantile', 0.25)
        roc_period = params.get('roc_period', 12)
        roc_threshold = params.get('roc_threshold', 5.0)
        vol_ma_period = params.get('vol_ma_period', 55)
        min_expanding_periods = boll_period * 2 # 为动态分位数计算设置一个最小观测窗口

        # 2. 构建所有依赖列的名称
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}"
        roc_col = f"ROC_{roc_period}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}"
        
        # 3. 依赖检查，现在需要检查BBW, ROC, 和 VOL_MA
        required_cols = [bbw_col, roc_col, vol_ma_col, 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"    - [依赖错误] V2.0箱体计算跳过，依赖的列 '{', '.join(missing)}{suffix}' 不存在。")
            print(f"    - [检查清单] 请确保 'boll_bands_and_width', 'roc', 'vol_ma' 指标已在JSON中为'{suffix}'正确启用并配置。")
            return None

        # 4. 创建一个包含所有目标列（初始值为NaN）的DataFrame
        result_df = pd.DataFrame(index=df.index)
        output_cols = [
            'is_consolidating', # 新增：用于调试和分析的布尔标记
            'dynamic_bbw_threshold', # 新增：用于观察动态阈值的变化
            'dynamic_consolidation_high', 
            'dynamic_consolidation_low', 
            'dynamic_consolidation_avg_vol', 
            'dynamic_consolidation_duration'
        ]
        for col in output_cols:
            result_df[col] = np.nan if col not in ['is_consolidating'] else False

        # 5. 【核心逻辑】定义多因子共振条件
        
        # 5.1 波动率因子: BBW必须低于其自身的动态历史分位数 (无未来函数)
        # 使用 .expanding() 确保在每个时间点，我们只使用过去的数据来计算分位数
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile)
        cond_volatility = df[bbw_col] < dynamic_bbw_threshold
        result_df['dynamic_bbw_threshold'] = dynamic_bbw_threshold # 存入结果用于分析

        # 5.2 趋势因子: ROC绝对值必须低于阈值，表示趋势停滞
        cond_trend = df[roc_col].abs() < roc_threshold

        # 5.3 成交量因子: 当前成交量必须低于其移动平均线，表示缩量
        cond_volume = df['volume'] < df[vol_ma_col]

        # 5.4 共振: 三个条件必须同时满足
        is_consolidating = cond_volatility & cond_trend & cond_volume
        result_df['is_consolidating'] = is_consolidating

        # 如果没有任何盘整期，直接返回包含诊断信息的result_df
        if not is_consolidating.any():
            print(f"    - [信息] V2.0箱体计算：在整个周期内未发现满足多因子共振的盘整期。")
            return result_df

        # 6. 计算箱体指标 (此部分逻辑与V1.1相同，但作用于新的 is_consolidating 序列)
        consolidation_blocks = (is_consolidating != is_consolidating.shift()).cumsum()
        consolidating_df = df[is_consolidating].copy()
        grouped = consolidating_df.groupby(consolidation_blocks[is_consolidating])

        consolidation_high = grouped['high'].transform('max')
        consolidation_low = grouped['low'].transform('min')
        consolidation_avg_vol = grouped['volume'].transform('mean')
        consolidation_duration = grouped['high'].transform('size')

        # 7. 将计算结果填充到 result_df 中
        result_df['dynamic_consolidation_high'].update(consolidation_high)
        result_df['dynamic_consolidation_low'].update(consolidation_low)
        result_df['dynamic_consolidation_avg_vol'].update(consolidation_avg_vol)
        result_df['dynamic_consolidation_duration'].update(consolidation_duration)

        # 向前填充(ffill)结果，使得在整个盘整期内，箱体值保持不变
        # 注意：只填充箱体指标，不填充诊断列
        fill_cols = [
            'dynamic_consolidation_high', 'dynamic_consolidation_low', 
            'dynamic_consolidation_avg_vol', 'dynamic_consolidation_duration'
        ]
        result_df[fill_cols] = result_df[fill_cols].ffill()
        
        # 8. 返回包含新列的DataFrame
        print(f"    - [成功] V2.0箱体计算完成，共识别出 {is_consolidating.sum()} 个盘整周期点。")
        return result_df

    async def calculate_advanced_fund_features(self, df: pd.DataFrame, params: dict, suffix: str) -> Optional[pd.DataFrame]:
        """
        【V1.0】计算基于资金流和筹码的衍生特征。
        - 核心逻辑: 对原始资金流和筹码数据进行二次加工，生成趋势和偏离度等特征。
        - 依赖项: 
            - 资金流数据 (e.g., 'buy_lg_amount')
            - 筹码数据 (e.g., 'weight_avg')
            - 基础行情数据 ('close')
        - 输出: 
            - fund_buy_lg_amount_ma5: 大单净买入5日移动平均
            - fund_buy_lg_amount_ma10: 大单净买入10日移动平均
            - chip_cost_deviation: 收盘价相对筹码平均成本的偏离度
        """
        # 1. 检查依赖列是否存在于传入的DataFrame中
        #    这是关键一步，确保数据合并已在上游完成
        required_cols = ['buy_lg_amount', 'weight_avg', 'close']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"    - [依赖错误] 资金流衍生特征计算跳过，因缺失必要列: {missing}。请确保上游数据已正确合并。")
            return None

        try:
            # 2. 初始化一个空的DataFrame用于存放结果
            derived_features = pd.DataFrame(index=df.index)

            # 3. 从params或默认值获取参数
            ma_periods = params.get('ma_periods', [5, 10])

            # 4. 计算资金流特征：大单净买入的移动平均
            #    使用循环，更具扩展性
            for period in ma_periods:
                # 注意：输出列名不带后缀，由调用者统一添加
                derived_features[f'fund_buy_lg_amount_ma{period}'] = df['buy_lg_amount'].rolling(window=period).mean()

            # 5. 计算筹码特征：收盘价与平均成本的偏离度
            #    (收盘价 / 平均成本) - 1。 >0 代表股价在成本之上，<0 代表股价在成本之下。
            #    使用 .replace(0, np.nan) 避免除以零的错误
            cost_basis = df['weight_avg'].replace(0, np.nan)
            derived_features['chip_cost_deviation'] = df['close'] / cost_basis - 1
            
            print("    - [信息] 已成功计算资金流和筹码衍生特征。")
            return derived_features

        except Exception as e:
            # 使用Django的logger或您自己的日志系统
            print(f"    - [严重错误] 计算资金流和筹码衍生特征时发生意外: {e}")
            return None

    def add_lagged_features(self, df: pd.DataFrame, columns_to_lag_with_suffix: List[str], lags: List[int]) -> pd.DataFrame:
        """
        为指定的列添加滞后特征。此为同步函数。

        Args:
            df (pd.DataFrame): 输入 DataFrame。
            columns_to_lag_with_suffix (List[str]): 需要添加滞后特征的列名列表。
            lags (List[int]): 滞后周期列表。

        Returns:
            pd.DataFrame: 补充了滞后特征的 DataFrame。
        """
        if df is None or df.empty or not columns_to_lag_with_suffix or not lags:
            logger.warning("添加滞后特征失败，输入 DataFrame 无效或配置不完整。")
            return df
        for col in columns_to_lag_with_suffix:
            if col in df.columns:
                for lag in lags:
                    if lag <= 0:
                         logger.warning(f"无效的滞后周期: {lag}，跳过。")
                         continue
                    new_col_name = f'{col}_lag_{lag}'
                    if new_col_name not in df.columns:
                         df[new_col_name] = df[col].shift(lag)
                    else:
                         logger.debug(f"列 {new_col_name} 已存在，跳过添加。")
            else:
                logger.warning(f"添加滞后特征失败，未找到列: {col}")
        return df

    def add_rolling_features(self, df: pd.DataFrame, columns_to_roll_with_suffix: List[str], windows: List[int], stats: List[str]) -> pd.DataFrame:
        """
        为指定的列添加滚动统计特征。此为同步函数。

        Args:
            df (pd.DataFrame): 输入 DataFrame。
            columns_to_roll_with_suffix (List[str]): 需要添加滚动统计特征的列名列表。
            windows (List[int]): 滚动窗口大小列表。
            stats (List[str]): 滚动统计类型列表。

        Returns:
            pd.DataFrame: 补充了滚动统计特征的 DataFrame。
        """
        if df is None or df.empty or not columns_to_roll_with_suffix or not windows or not stats:
            logger.warning("添加滚动统计特征失败，输入 DataFrame 无效或配置不完整。")
            return df
        valid_stats = ['mean', 'std', 'min', 'max', 'sum', 'median', 'count', 'var', 'skew', 'kurt']
        stats_to_apply = [s for s in stats if s in valid_stats]
        if not stats_to_apply:
             logger.warning(f"未指定有效的滚动统计类型。支持类型: {valid_stats}")
             return df
        for col in columns_to_roll_with_suffix:
            if col in df.columns:
                for window in windows:
                    if window <= 0:
                         logger.warning(f"无效的滚动窗口大小: {window}，跳过。")
                         continue
                    rolling_obj = df[col].rolling(window=window, min_periods=max(1, int(window*0.5)))
                    for stat in stats_to_apply:
                        new_col_name = f'{col}_rolling_{stat}_{window}'
                        if new_col_name not in df.columns:
                            try:
                                if stat == 'mean': df[new_col_name] = rolling_obj.mean()
                                elif stat == 'std': df[new_col_name] = rolling_obj.std()
                                elif stat == 'min': df[new_col_name] = rolling_obj.min()
                                elif stat == 'max': df[new_col_name] = rolling_obj.max()
                                elif stat == 'sum': df[new_col_name] = rolling_obj.sum()
                                elif stat == 'median': df[new_col_name] = rolling_obj.median()
                                elif stat == 'count': df[new_col_name] = rolling_obj.count()
                                elif stat == 'var': df[new_col_name] = rolling_obj.var()
                                elif stat == 'skew': df[new_col_name] = rolling_obj.skew()
                                elif stat == 'kurt': df[new_col_name] = rolling_obj.kurt()
                            except Exception as e:
                                logger.error(f"计算滚动统计 {stat} for 列 {col} (窗口 {window}) 出错: {e}", exc_info=True)
                                df[new_col_name] = np.nan
                        else:
                             logger.debug(f"列 {new_col_name} 已存在，跳过添加。")
            else:
                logger.warning(f"添加滚动统计特征失败，未找到列: {col}")
        return df

    async def get_5_min_kline_time_by_day_count(self, stock_code: str, day_count: int) -> List[datetime.datetime]:
        """
        获取指定股票在前N个交易日内所有的5分钟K线的交易时间（UTC，datetime对象）
        """
        stock = await self.stock_basic_dao.get_stock_by_code(stock_code)
        if not stock:
            print(f"未找到股票代码：{stock_code}")
            return []

        index_dao = IndexBasicDAO()
        trade_days = await index_dao.get_last_n_trade_cal_open(n=day_count)
        if not trade_days:
            print("未获取到交易日列表")
            return []
        else:
            print(f"获取到交易日列表: {trade_days}")

        trade_times_set = set()
        stt_dao = StockTimeTradeDAO()
        for trade_day in trade_days:
            # 获取当天的所有5分钟K线时间（假设返回字符串列表）
            daily_trade_times = await stt_dao.get_5_min_kline_time_by_day(stock_code=stock_code, date=trade_day)
            # 转为带UTC时区的datetime对象
            for t_str in daily_trade_times:
                # 假设格式为"2024-06-08 09:35:00"
                t_dt = datetime.datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc)  # 加入UTC时区
                trade_times_set.add(t_dt)

        trade_times_list = sorted(trade_times_set)
        print(f"总共获取到的交易时间点数量：{stock} - {len(trade_times_list)}")
        return trade_times_list



    # logger.info(f"[{stock_code}] 开始补充外部特征 (指数、板块、筹码、资金流向)...")
    # final_df = await self.enrich_features(df=final_df, stock_code=stock_code, main_indices=main_index_codes, external_data_history_days=external_data_history_days)
    # logger.info(f"[{stock_code}] 外部特征补充完成。最终 DataFrame Shape: {final_df.shape}, 列数: {len(final_df.columns)}")
    # actual_rsi_period = bs_params.get('rsi_period', default_rsi_p['period'])
    # actual_macd_fast = bs_params.get('macd_fast', default_macd_p['period_fast'])
    # actual_macd_slow = bs_params.get('macd_slow', default_macd_p['period_slow'])
    # actual_macd_signal = bs_params.get('macd_signal', default_macd_p['signal_period'])
    # fe_config = params.get('feature_engineering_params', {})
    # apply_on_tfs = fe_config.get('apply_on_timeframes', bs_timeframes)

    # # 相对强度
    # rs_config = fe_config.get('relative_strength', {})
    # if rs_config.get('enabled', False):
    #      ths_indexs_for_rs_objects = await self.industry_dao.get_stock_ths_indices(stock_code)
    #      if ths_indexs_for_rs_objects is None:
    #           logger.warning(f"[{stock_code}] 无法获取股票 {stock_code} 的同花顺板块信息。相对强度计算将跳过。")
    #           ths_codes_for_rs = []
    #      else:
    #         ths_codes_for_rs = [m.ths_index.ts_code for m in ths_indexs_for_rs_objects if m.ths_index]
    #      all_benchmark_codes_for_rs = list(set(main_index_codes + ths_codes_for_rs))
    #      periods = rs_config.get('periods', [5, 10, 20])
    #      if all_benchmark_codes_for_rs and periods:
    #           for tf_apply in apply_on_tfs:
    #                stock_close_col = f'close_{tf_apply}'
    #                if stock_close_col in final_df.columns:
    #                     final_df = self.calculate_relative_strength(df=final_df, stock_close_col=stock_close_col, benchmark_codes=all_benchmark_codes_for_rs, periods=periods, time_level=tf_apply)
    #                else:
    #                     logger.warning(f"[{stock_code}] 计算相对强度 for TF {tf_apply} 失败，未找到股票收盘价列: {stock_close_col}")
    #           logger.info(f"[{stock_code}] 相对强度/超额收益特征计算完成。")
    #      else:
    #           logger.warning(f"[{stock_code}] 相对强度/超额收益特征未启用或配置不完整 (基准代码或周期列表为空)。")
    # # 滞后特征
    # lag_config = fe_config.get('lagged_features', {})
    # if lag_config.get('enabled', False):
    #     columns_to_lag_from_json = lag_config.get('columns_to_lag', [])
    #     lags = lag_config.get('lags', [1, 2, 3])
    #     if columns_to_lag_from_json and lags:
    #         logger.debug(f"[{stock_code}] 开始添加滞后特征...")
    #         for tf_apply in apply_on_tfs:
    #             actual_cols_for_lagging = []
    #             for col_template_from_json in columns_to_lag_from_json:
    #                 effective_base_name = col_template_from_json
    #                 if col_template_from_json.startswith("RSI"):
    #                     effective_base_name = f"RSI_{actual_rsi_period}"
    #                 elif col_template_from_json.startswith("MACD_") and \
    #                     not col_template_from_json.startswith("MACDh_") and \
    #                     not col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDh_"):
    #                     effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 col_with_suffix = f"{effective_base_name}_{tf_apply}"
    #                 if col_with_suffix in final_df.columns:
    #                     actual_cols_for_lagging.append(col_with_suffix)
    #                 else:
    #                     if effective_base_name in final_df.columns:
    #                         actual_cols_for_lagging.append(effective_base_name)
    #                         logger.debug(f"[{stock_code}] TF {tf_apply}: 找到不带时间级别后缀的列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 进行滞后计算。")
    #                     else:
    #                         logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到指定列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 或其带后缀形式 {col_with_suffix} 进行滞后计算。")
    #             if actual_cols_for_lagging:
    #                 logger.debug(f"[{stock_code}] TF {tf_apply}: 准备为以下列添加滞后特征: {actual_cols_for_lagging}")
    #                 final_df = self.add_lagged_features(final_df, actual_cols_for_lagging, lags)
    #                 logger.debug(f"[{stock_code}] 添加滞后特征 for TF {tf_apply} 完成。")
    #             else:
    #                 logger.warning(f"[{stock_code}] 添加滞后特征 for TF {tf_apply} 失败，未找到任何有效列。JSON配置: {columns_to_lag_from_json}")
    #         logger.info(f"[{stock_code}] 滞后特征添加完成。")
    #     else:
    #         logger.warning(f"[{stock_code}] 滞后特征未启用或配置不完整。")
    # # 滚动特征
    # roll_config = fe_config.get('rolling_features', {})
    # if roll_config.get('enabled', False):
    #     columns_to_roll_from_json = roll_config.get('columns_to_roll', [])
    #     windows = roll_config.get('windows', [5, 10, 20])
    #     stats = roll_config.get('stats', ["mean", "std"])
    #     if columns_to_roll_from_json and windows and stats:
    #         logger.debug(f"[{stock_code}] 开始添加滚动统计特征...")
    #         for tf_apply in apply_on_tfs:
    #             actual_cols_for_rolling = []
    #             for col_template_from_json in columns_to_roll_from_json:
    #                 effective_base_name = col_template_from_json
    #                 if col_template_from_json.startswith("RSI"):
    #                     effective_base_name = f"RSI_{actual_rsi_period}"
    #                 elif col_template_from_json.startswith("MACD_") and \
    #                     not col_template_from_json.startswith("MACDh_") and \
    #                     not col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDh_"):
    #                     effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 elif col_template_from_json.startswith("MACDs_"):
    #                     effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
    #                 col_with_suffix = f"{effective_base_name}_{tf_apply}"
    #                 if col_with_suffix in final_df.columns:
    #                     actual_cols_for_rolling.append(col_with_suffix)
    #                 else:
    #                     if effective_base_name in final_df.columns:
    #                         actual_cols_for_rolling.append(effective_base_name)
    #                         logger.debug(f"[{stock_code}] TF {tf_apply}: 找到不带时间级别后缀的列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 进行滚动统计计算。")
    #                     else:
    #                         logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到指定列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 或其带后缀形式 {col_with_suffix} 进行滚动统计计算。")
    #             if actual_cols_for_rolling:
    #                 logger.debug(f"[{stock_code}] TF {tf_apply}: 准备为以下列添加滚动统计特征: {actual_cols_for_rolling}")
    #                 final_df = self.add_rolling_features(final_df, actual_cols_for_rolling, windows, stats)
    #                 logger.debug(f"[{stock_code}] 添加滚动统计特征 for TF {tf_apply} 完成。")
    #             else:
    #                 logger.warning(f"[{stock_code}] 滚动统计特征 for TF {tf_apply} 失败，未找到任何指定列。JSON配置: {columns_to_roll_from_json}")
    #         logger.info(f"[{stock_code}] 滚动统计特征添加完成。")
    #     else:
    #         logger.warning(f"[{stock_code}] 滚动统计特征未启用或配置不完整。")
    # original_nan_count = final_df.isnull().sum().sum()














