# services\indicator_services.py
import asyncio
from collections import defaultdict
import datetime
from functools import reduce
import json
import os
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
from core.constants import TimeLevel
from dao_manager.tushare_daos.stock_time_trade_dao import StockTimeTradeDAO

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
        
        try:
            global ta
            import pandas_ta as ta
            if ta is None:
                 logger.warning("pandas_ta 之前导入失败，尝试重新导入。")
                 import pandas_ta as ta
        except ImportError:
            logger.error("pandas-ta 库未安装，请运行 'pip install pandas-ta'")
            ta = None

    def _get_timeframe_in_minutes(self, tf_str: str) -> Optional[int]:
        """
        将时间级别字符串（如 '5', '15', 'D', 'W', 'M'）转换为近似的分钟数。
        'D', 'W', 'M' 基于标准交易时间估算。

        Args:
            tf_str (str): 时间级别字符串。

        Returns:
            Optional[int]: 对应的分钟数，如果无法转换则返回 None。
        """
        tf_str = str(tf_str).upper()
        if tf_str.isdigit():
            return int(tf_str)
        elif tf_str == 'D':
            return 240
        elif tf_str == 'W':
            return 240 * 5
        elif tf_str == 'M':
            return 240 * 21
        else:
            logger.warning(f"无法将时间级别 '{tf_str}' 转换为分钟数，将返回 None。")
            return None

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
        needed = max(estimated_bars_needed, global_max_lookback) + 100
        if target_tf_minutes >= 240:
             needed = max(estimated_bars_needed, global_max_lookback * (target_tf_minutes / 240) + 100)
             if target_tf_minutes >= 240:
                  needed = max(needed, global_max_lookback + 365*2)
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
            df.rename(columns={'vol': 'volume'}, inplace=True) # 修改行: 重命名 'vol' 为 'volume'
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 将原始数据列名 'vol' 重命名为 'volume'.")
        if 'amount_col_from_dao' in df.columns and 'amount' not in df.columns: # 示例性的其他列名标准化
            df.rename(columns={'amount_col_from_dao': 'amount'}, inplace=True)
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 示例性重命名 'amount_col_from_dao' 为 'amount'.")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is not None:
            logger.debug(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
            logger.warning(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，但索引不是时区感知的 DatetimeIndex。")
        return df

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
            print(f"Debug: _resample_and_clean_dataframe 收到空或None的DataFrame，时间级别 {tf}。") # 修改行: 调试信息
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
                    print(f"Debug: _resample_and_clean_dataframe: 尝试将DataFrame索引转换为默认时区。") # 修改行: 调试信息
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
            resampled_df_renamed = resampled_df.rename(columns=rename_map_with_suffix) # 修改行: 对列名添加时间级别后缀
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
        
        all_indices_dfs_list = [] # 修改行: 优化合并逻辑，先收集再合并
        if index_daily_df is not None and not index_daily_df.empty:
            for index_code in main_indices:
                idx_df = index_daily_df[index_daily_df['index_code'] == index_code].copy()
                if not idx_df.empty:
                    idx_df.drop(columns=['index_code'], inplace=True)
                    idx_df.columns = [f'index_{index_code.replace(".", "_").lower()}_{col}' for col in idx_df.columns]
                    all_indices_dfs_list.append(idx_df) # 修改行: 添加到列表
        if ths_daily_df is not None and not ths_daily_df.empty:
            for ths_code in ths_codes:
                ths_d_df = ths_daily_df[ths_daily_df['ts_code'] == ths_code].copy()
                if not ths_d_df.empty:
                    ths_d_df.drop(columns=['ts_code'], inplace=True)
                    ths_d_df.columns = [f'ths_{ths_code.replace(".", "_").lower()}_{col}' for col in ths_d_df.columns]
                    all_indices_dfs_list.append(ths_d_df) # 修改行: 添加到列表

        all_indices_df = pd.DataFrame(index=df.index if not df.empty else None) # 修改行: 初始化为空DF并尝试使用主DF索引
        if all_indices_dfs_list: # 修改行: 如果列表不为空，则进行合并
            # 使用 reduce 进行合并，确保列名唯一性（通过前缀已经保证大部分，suffixes 主要处理意外情况）
            all_indices_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), all_indices_dfs_list)
        
        if not all_indices_df.empty: # 修改行: 检查合并后的 all_indices_df
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

        fund_flow_cnt_ths_dfs_list = [] # 修改行: 优化合并逻辑
        if fund_flow_cnt_ths_df_raw is not None and not fund_flow_cnt_ths_df_raw.empty:
             logger.debug(f"已获取同花顺板块资金流向统计数据 (原始)，数据量: {len(fund_flow_cnt_ths_df_raw)} 条")
             for ths_code_ff in fund_flow_cnt_ths_df_raw['ts_code'].unique(): # 修改变量名避免冲突
                  cnt_df = fund_flow_cnt_ths_df_raw[fund_flow_cnt_ths_df_raw['ts_code'] == ths_code_ff].copy()
                  if not cnt_df.empty:
                       cnt_df.drop(columns=['ts_code'], inplace=True)
                       cnt_df.columns = [f'ff_cnt_ths_{ths_code_ff.replace(".", "_").lower()}_{col}' for col in cnt_df.columns]
                       cnt_df.set_index('trade_time', inplace=True)
                       cnt_df.sort_index(ascending=True, inplace=True)
                       fund_flow_cnt_ths_dfs_list.append(cnt_df) # 修改行: 添加到列表
        
        fund_flow_cnt_ths_df_processed = pd.DataFrame(index=df.index if not df.empty else None) # 修改行
        if fund_flow_cnt_ths_dfs_list: # 修改行
            fund_flow_cnt_ths_df_processed = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), fund_flow_cnt_ths_dfs_list)
            if not fund_flow_cnt_ths_df_processed.empty:
                 logger.debug(f"已处理同花顺板块资金流向统计数据，数据量: {len(fund_flow_cnt_ths_df_processed)} 条，列数: {len(fund_flow_cnt_ths_df_processed.columns)}")
            else:
                 logger.warning(f"处理同花顺板块资金流向统计数据后 DataFrame 为空 for {ths_codes}")
        
        fund_flow_industry_ths_dfs_list = [] # 修改行: 优化合并逻辑
        if fund_flow_industry_ths_df_raw is not None and not fund_flow_industry_ths_df_raw.empty:
             logger.debug(f"已获取同花顺行业资金流向统计数据 (原始)，数据量: {len(fund_flow_industry_ths_df_raw)} 条")
             for ths_code_fi in fund_flow_industry_ths_df_raw['ts_code'].unique(): # 修改变量名
                  ind_df = fund_flow_industry_ths_df_raw[fund_flow_industry_ths_df_raw['ts_code'] == ths_code_fi].copy()
                  if not ind_df.empty:
                       ind_df.drop(columns=['ts_code'], inplace=True)
                       ind_df.columns = [f'ff_ind_ths_{ths_code_fi.replace(".", "_").lower()}_{col}' for col in ind_df.columns]
                       ind_df.set_index('trade_time', inplace=True)
                       ind_df.sort_index(ascending=True, inplace=True)
                       fund_flow_industry_ths_dfs_list.append(ind_df) # 修改行: 添加到列表

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
        
        valid_external_dfs = [ext_df for ext_df in external_features_dfs if ext_df is not None and not ext_df.empty] # 修改行: 过滤无效DF
        merged_external_df = pd.DataFrame(index=df.index if not df.empty else None) # 修改行
        if valid_external_dfs: # 修改行: 使用 reduce 合并有效DF
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
            # print(f"[{stock_code}] Debug: pandas_ta 未加载。")
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None, None
        try:
            if not os.path.exists(params_file):
                # print(f"[{stock_code}] Debug: 策略参数文件未找到: {params_file}")
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None, None
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
        except Exception as e:
            # print(f"[{stock_code}] Debug: 加载或解析参数文件失败: {e}")
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
        print(f"[{stock_code}] Debug: 策略关注的时间级别 (focus_tf): {focus_tf}")
        print(f"[{stock_code}] Debug: 所有策略所需时间级别集合: {sorted(list(all_time_levels_needed))}")
        min_time_level = None
        min_tf_minutes = float('inf')
        if not all_time_levels_needed:
            logger.error(f"[{stock_code}] 未能从参数文件中确定任何需要的时间级别。")
            return None, None
        # 按照分钟数从小到大排序时间级别，以便先处理小级别数据
        sorted_time_levels = sorted(list(all_time_levels_needed), key=lambda tf: self._get_timeframe_in_minutes(tf) or float('inf'))
        if not sorted_time_levels:
            logger.error(f"[{stock_code}] 无法确定有效的最小时间级别从所需级别: {all_time_levels_needed}")
            return None, None
        min_time_level = sorted_time_levels[0] # 最小时间级别
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
        # print(f"[{stock_code}] Debug: 用于回看期计算的唯一指标配置数量: {len(unique_configs_for_lookback)}")
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
        min_usable_bars = math.ceil(effective_base_needed_bars * 0.6) # 用于检查数据量是否足够

        # 顺序获取/聚合 OHLCV 数据
        final_ohlcv_dfs: Dict[str, pd.DataFrame] = {}
        for tf_process in sorted_time_levels:
            needed_bars_for_tf = self._calculate_needed_bars_for_tf(
                target_tf=tf_process, min_tf=min_time_level,
                base_needed_bars=effective_base_needed_bars,
                global_max_lookback=global_max_lookback
            )
            # print(f"[{stock_code}] Debug: 正在处理时间级别 {tf_process}。所需条数: {needed_bars_for_tf}")

            # 1. 尝试从数据库直接获取该时间级别的数据
            raw_df_from_db = await self._get_ohlcv_data(stock_code=stock_code, time_level=tf_process, needed_bars=needed_bars_for_tf, trade_time=trade_time)
            processed_df_for_tf = None

            if raw_df_from_db is not None and not raw_df_from_db.empty:
                # 如果数据库有数据，直接重采样和清洗
                # print(f"[{stock_code}] Debug: 时间级别 {tf_process}: 从数据库获取到 {len(raw_df_from_db)} 条数据，进行重采样。")
                processed_df_for_tf = await asyncio.to_thread(
                    self._resample_and_clean_dataframe, raw_df_from_db, tf_process, min_periods=1, fill_method='ffill'
                )
            else:
                # 2. 如果数据库没有数据，尝试从更小的时间级别聚合
                # 获取所有潜在的更精细的源时间级别，按分钟数从小到大排序
                potential_source_tfs = self._get_aggregation_source_tfs(tf_process, all_time_levels_needed)
                # print(f"[{stock_code}] Debug: 时间级别 {tf_process}: 数据库无数据，尝试从更小时间级别聚合。潜在源: {potential_source_tfs}")

                for source_tf in potential_source_tfs:
                    # 检查该源时间级别的数据是否已经成功处理并存在
                    if source_tf in final_ohlcv_dfs and not final_ohlcv_dfs[source_tf].empty:
                        # print(f"[{stock_code}] Debug: 尝试从已处理的 {source_tf} 聚合到 {tf_process}。")
                        source_df_for_agg = final_ohlcv_dfs[source_tf].copy()
                        # 移除后缀，以便resample能识别'open', 'high', 'low', 'close', 'volume', 'amount'
                        # 这一步非常重要，因为 _resample_and_clean_dataframe 期望输入是标准OHLCV列名
                        source_df_for_agg.columns = [col.replace(f'_{source_tf}', '') if col.endswith(f'_{source_tf}') else col for col in source_df_for_agg.columns]
                        
                        processed_df_for_tf = await asyncio.to_thread(
                            self._resample_and_clean_dataframe, source_df_for_agg, tf_process, min_periods=1, fill_method='ffill'
                        )
                        if processed_df_for_tf is not None and not processed_df_for_tf.empty:
                            print(f"[{stock_code}] Debug: 成功从 {source_tf} 聚合得到 {len(processed_df_for_tf)} 条 {tf_process} 数据。")
                            break # 成功聚合，跳出内部循环，不再尝试其他源
                        else:
                            print(f"[{stock_code}] Debug: 从 {source_tf} 聚合到 {tf_process} 失败或结果为空，尝试下一个潜在源。")
                    else:
                        print(f"[{stock_code}] Debug: 潜在源 {source_tf} 未处理或为空，跳过。")

            if processed_df_for_tf is None or processed_df_for_tf.empty:
                # print(f"[{stock_code}] Debug: 时间级别 {tf_process}: 无法从数据库获取数据，也无法从任何更小时间级别聚合。")
                logger.warning(f"[{stock_code}] 时间级别 {tf_process} 无法获取或聚合到有效数据，跳过。")
                final_ohlcv_dfs[tf_process] = pd.DataFrame() # 标记为空DataFrame
                if tf_process == min_time_level:
                    logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 无法获取有效数据，终止流程。")
                    return None, None
                continue
            
            # 对最小时间级别的数据量进行硬性检查
            if tf_process == min_time_level and len(processed_df_for_tf) < min_usable_bars:
                logger.error(f"[{stock_code}] 最小时间级别 {tf_process} 处理后数据量 {len(processed_df_for_tf)} 条，少于最低可用阈值 {min_usable_bars} 条。无法继续。")
                return None, None # 数据量不足，终止流程
            
            # 检查数据量是否显著少于全局最大回看期
            if len(processed_df_for_tf) < global_max_lookback * 0.5:
                logger.warning(f"[{stock_code}] 时间级别 {tf_process} 处理后数据量 {len(processed_df_for_tf)} 条，显著少于全局指标最大回看期 {global_max_lookback} 条。计算的指标可能不可靠。")
            
            final_ohlcv_dfs[tf_process] = processed_df_for_tf
            # print(f"[{stock_code}] Debug: 时间级别 {tf_process} 处理完成，数据量: {len(final_ohlcv_dfs[tf_process])}。")
        # 修改结束: 顺序获取/聚合 OHLCV 数据

        if min_time_level not in final_ohlcv_dfs or final_ohlcv_dfs[min_time_level].empty:
             logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 的 OHLCV 数据不可用，无法进行合并。")
             return None, None
        
        base_index = final_ohlcv_dfs[min_time_level].index
        logger.debug(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的 OHLCV 索引作为合并基准，数量: {len(base_index)}。")
        
        indicator_calculation_tasks = []
        async def _calculate_single_indicator_async(tf_calc: str, base_df_with_suffix: pd.DataFrame, config_item: Dict) -> Optional[Tuple[str, pd.DataFrame]]:
            """
            异步计算单个时间级别上的单个指标。
            核心的 pandas-ta 计算将通过 asyncio.to_thread 在单独线程中运行，以避免阻塞事件循环。
            """
            if base_df_with_suffix is None or base_df_with_suffix.empty:
                # print(f"[{stock_code}] Debug: TF {tf_calc}: 基础OHLCV数据为空，无法计算指标 {config_item['name']}")
                return None, None
            df_for_ta = base_df_with_suffix.copy()
            ohlcv_map_to_std = {
                f'open_{tf_calc}': 'open', f'high_{tf_calc}': 'high', f'low_{tf_calc}': 'low',
                f'close_{tf_calc}': 'close', f'volume_{tf_calc}': 'volume', f'amount_{tf_calc}': 'amount'
            }
            actual_rename_map_to_std = {k: v for k, v in ohlcv_map_to_std.items() if k in df_for_ta.columns}
            df_for_ta.rename(columns=actual_rename_map_to_std, inplace=True)
            required_cols_for_func = set(['high', 'low', 'close'])
            if config_item['name'] in ['MFI', 'OBV', 'VWAP', 'CMF', 'VOL_MA', 'ADL', 'VROC']: # VROC 也需要 volume
                required_cols_for_func.add('volume')
            if config_item['name'] in ['AMT_MA', 'AROC']:
                required_cols_for_func.add('amount')
            if not all(col in df_for_ta.columns for col in required_cols_for_func):
                missing_cols_str = ", ".join(list(required_cols_for_func - set(df_for_ta.columns)))
                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 缺少必要列 ({missing_cols_str})，跳过计算。")
                logger.debug(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} 时，df_for_ta 缺少必要列 ({missing_cols_str})。可用: {df_for_ta.columns.tolist()}")
                return None, None
            try:
                func_params_to_pass = config_item['params'].copy()
                indicator_result_df = await config_item['func'](df_for_ta, **func_params_to_pass)
                if indicator_result_df is None:
                    # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算结果为 None。")
                    logger.debug(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为 None。")
                    return None, None
                if indicator_result_df.empty:
                    # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算结果为 Empty DataFrame。")
                    logger.debug(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为空。")
                    return None, None
                if not isinstance(indicator_result_df, pd.DataFrame):
                    logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算函数未返回DataFrame (返回类型: {type(indicator_result_df)})。尝试转换。")
                    if isinstance(indicator_result_df, pd.Series):
                        series_name = indicator_result_df.name if indicator_result_df.name else config_item['name']
                        indicator_result_df = indicator_result_df.to_frame(name=series_name)
                        # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 转换为DataFrame后列名: {indicator_result_df.columns.tolist()}")
                    else:
                        # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 返回类型 {type(indicator_result_df)} 无法处理。")
                        logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 返回类型 {type(indicator_result_df)} 无法处理。")
                        return None, None
                result_renamed_df = indicator_result_df.rename(columns=lambda x: f"{x}_{tf_calc}")
                return (tf_calc, result_renamed_df)
            except Exception as e_calc:
                # print(f"[{stock_code}] Debug: TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}")
                logger.error(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}", exc_info=True)
                return None, None
        
        for config_item_loop in indicator_configs:
            for tf_conf in config_item_loop['timeframes']:
                if tf_conf in final_ohlcv_dfs and not final_ohlcv_dfs[tf_conf].empty: # 确保数据有效
                    base_ohlcv_df_for_tf_loop = final_ohlcv_dfs[tf_conf]
                    indicator_calculation_tasks.append(
                        _calculate_single_indicator_async(tf_conf, base_ohlcv_df_for_tf_loop, config_item_loop)
                    )
                else:
                    # print(f"[{stock_code}] Debug: 时间框架 {tf_conf} 的基础数据未找到或无效 ({config_item_loop['name']})，无法创建计算任务。")
                    logger.warning(f"[{stock_code}] 时间框架 {tf_conf} 在 final_ohlcv_dfs 中未找到有效数据，无法为指标 {config_item_loop['name']} 创建计算任务。")
        
        calculated_results_tuples = await asyncio.gather(*indicator_calculation_tasks, return_exceptions=True)
        calculated_indicators_by_tf = defaultdict(list)
        for res_tuple_item in calculated_results_tuples:
            if isinstance(res_tuple_item, tuple) and len(res_tuple_item) == 2:
                tf_res, indi_df_res = res_tuple_item
                if indi_df_res is not None and not indi_df_res.empty:
                    calculated_indicators_by_tf[tf_res].append(indi_df_res)
            elif isinstance(res_tuple_item, Exception):
                # print(f"[{stock_code}] Debug: 指标计算任务发生异常: {res_tuple_item}")
                logger.error(f"[{stock_code}] 指标计算任务发生异常: {res_tuple_item}", exc_info=res_tuple_item)
            else:
                # print(f"[{stock_code}] Debug: 指标计算任务返回非预期结果: {res_tuple_item}")
                logger.warning(f"[{stock_code}] 指标计算任务返回非预期结果: {res_tuple_item}")
        
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
        logger.info(f"[{stock_code}] 所有 OHLCV 和指标数据合并完成，最终 Shape: {final_df.shape}, 列数: {len(final_df.columns)}")
        logger.info(f"[{stock_code}] 开始补充外部特征 (指数、板块、筹码、资金流向)...")
        final_df = await self.enrich_features(df=final_df, stock_code=stock_code, main_indices=main_index_codes, external_data_history_days=external_data_history_days)
        logger.info(f"[{stock_code}] 外部特征补充完成。最终 DataFrame Shape: {final_df.shape}, 列数: {len(final_df.columns)}")
        actual_rsi_period = bs_params.get('rsi_period', default_rsi_p['period'])
        actual_macd_fast = bs_params.get('macd_fast', default_macd_p['period_fast'])
        actual_macd_slow = bs_params.get('macd_slow', default_macd_p['period_slow'])
        actual_macd_signal = bs_params.get('macd_signal', default_macd_p['signal_period'])
        fe_config = params.get('feature_engineering_params', {})
        apply_on_tfs = fe_config.get('apply_on_timeframes', bs_timeframes)
        rs_config = fe_config.get('relative_strength', {})
        if rs_config.get('enabled', False):
             ths_indexs_for_rs_objects = await self.industry_dao.get_stock_ths_indices(stock_code)
             if ths_indexs_for_rs_objects is None:
                  logger.warning(f"[{stock_code}] 无法获取股票 {stock_code} 的同花顺板块信息。相对强度计算将跳过。")
                  ths_codes_for_rs = []
             else:
                ths_codes_for_rs = [m.ths_index.ts_code for m in ths_indexs_for_rs_objects if m.ths_index]
             all_benchmark_codes_for_rs = list(set(main_index_codes + ths_codes_for_rs))
             periods = rs_config.get('periods', [5, 10, 20])
             if all_benchmark_codes_for_rs and periods:
                  for tf_apply in apply_on_tfs:
                       stock_close_col = f'close_{tf_apply}'
                       if stock_close_col in final_df.columns:
                            final_df = self.calculate_relative_strength(df=final_df, stock_close_col=stock_close_col, benchmark_codes=all_benchmark_codes_for_rs, periods=periods, time_level=tf_apply)
                       else:
                            logger.warning(f"[{stock_code}] 计算相对强度 for TF {tf_apply} 失败，未找到股票收盘价列: {stock_close_col}")
                  logger.info(f"[{stock_code}] 相对强度/超额收益特征计算完成。")
             else:
                  logger.warning(f"[{stock_code}] 相对强度/超额收益特征未启用或配置不完整 (基准代码或周期列表为空)。")
        lag_config = fe_config.get('lagged_features', {})
        if lag_config.get('enabled', False):
             columns_to_lag_from_json = lag_config.get('columns_to_lag', [])
             lags = lag_config.get('lags', [1, 2, 3])
             if columns_to_lag_from_json and lags:
                  logger.debug(f"[{stock_code}] 开始添加滞后特征...")
                  for tf_apply in apply_on_tfs:
                       actual_cols_for_lagging = []
                       for col_template_from_json in columns_to_lag_from_json:
                            effective_base_name = col_template_from_json
                            if col_template_from_json.startswith("RSI"):
                                effective_base_name = f"RSI_{actual_rsi_period}"
                            elif col_template_from_json.startswith("MACD_") and \
                                 not col_template_from_json.startswith("MACDh_") and \
                                 not col_template_from_json.startswith("MACDs_"):
                                effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                            elif col_template_from_json.startswith("MACDh_"):
                                effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                            elif col_template_from_json.startswith("MACDs_"):
                                effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                            col_with_suffix = f"{effective_base_name}_{tf_apply}"
                            if col_with_suffix in final_df.columns:
                                 actual_cols_for_lagging.append(col_with_suffix)
                            else:
                                 if effective_base_name in final_df.columns:
                                      actual_cols_for_lagging.append(effective_base_name)
                                      logger.debug(f"[{stock_code}] TF {tf_apply}: 找到不带时间级别后缀的列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 进行滞后计算。")
                                 else:
                                      logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到指定列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 或其带后缀形式 {col_with_suffix} 进行滞后计算。")
                       if actual_cols_for_lagging:
                            logger.debug(f"[{stock_code}] TF {tf_apply}: 准备为以下列添加滞后特征: {actual_cols_for_lagging}")
                            final_df = self.add_lagged_features(final_df, actual_cols_for_lagging, lags)
                            logger.debug(f"[{stock_code}] 添加滞后特征 for TF {tf_apply} 完成。")
                       else:
                            logger.warning(f"[{stock_code}] 添加滞后特征 for TF {tf_apply} 失败，未找到任何有效列。JSON配置: {columns_to_lag_from_json}")
                  logger.info(f"[{stock_code}] 滞后特征添加完成。")
             else:
                logger.warning(f"[{stock_code}] 滞后特征未启用或配置不完整。")
        roll_config = fe_config.get('rolling_features', {})
        if roll_config.get('enabled', False):
             columns_to_roll_from_json = roll_config.get('columns_to_roll', [])
             windows = roll_config.get('windows', [5, 10, 20])
             stats = roll_config.get('stats', ["mean", "std"])
             if columns_to_roll_from_json and windows and stats:
                  logger.debug(f"[{stock_code}] 开始添加滚动统计特征...")
                  for tf_apply in apply_on_tfs:
                       actual_cols_for_rolling = []
                       for col_template_from_json in columns_to_roll_from_json:
                            effective_base_name = col_template_from_json
                            if col_template_from_json.startswith("RSI"):
                                effective_base_name = f"RSI_{actual_rsi_period}"
                            elif col_template_from_json.startswith("MACD_") and \
                                not col_template_from_json.startswith("MACDh_") and \
                                not col_template_from_json.startswith("MACDs_"):
                                effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                            elif col_template_from_json.startswith("MACDh_"):
                                effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                            elif col_template_from_json.startswith("MACDs_"):
                                effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                            col_with_suffix = f"{effective_base_name}_{tf_apply}"
                            if col_with_suffix in final_df.columns:
                                actual_cols_for_rolling.append(col_with_suffix)
                            else:
                                if effective_base_name in final_df.columns:
                                    actual_cols_for_rolling.append(effective_base_name)
                                    logger.debug(f"[{stock_code}] TF {tf_apply}: 找到不带时间级别后缀的列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 进行滚动统计计算。")
                                else:
                                    logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到指定列 {effective_base_name} (来自JSON模板 {col_template_from_json}) 或其带后缀形式 {col_with_suffix} 进行滚动统计计算。")
                       if actual_cols_for_rolling:
                            logger.debug(f"[{stock_code}] TF {tf_apply}: 准备为以下列添加滚动统计特征: {actual_cols_for_rolling}")
                            final_df = self.add_rolling_features(final_df, actual_cols_for_rolling, windows, stats)
                            logger.debug(f"[{stock_code}] 添加滚动统计特征 for TF {tf_apply} 完成。")
                       else:
                            logger.warning(f"[{stock_code}] 滚动统计特征 for TF {tf_apply} 失败，未找到任何指定列。JSON配置: {columns_to_roll_from_json}")
                  logger.info(f"[{stock_code}] 滚动统计特征添加完成。")
             else:
                logger.warning(f"[{stock_code}] 滚动统计特征未启用或配置不完整。")
        original_nan_count = final_df.isnull().sum().sum()
        final_df.ffill(inplace=True)
        final_df.bfill(inplace=True)
        nan_count_after_fill = final_df.isnull().sum().sum()
        if nan_count_after_fill > 0:
            logger.warning(f"[{stock_code}] 最终填充后仍存在 {nan_count_after_fill} 个缺失值 (原始 {original_nan_count})。缺失列详情 (部分): {final_df.isnull().sum()[final_df.isnull().sum() > 0].head().to_dict()}")
        else:
            logger.info(f"[{stock_code}] 最终缺失值填充完成，无剩余 NaN。")
        return final_df, indicator_configs

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
    # --- 以下所有 calculate_* 方法均已修改 ---
    # --- 将核心的、潜在CPU密集型的 pandas-ta 调用放入 asyncio.to_thread() 中执行，以实现真正的异步并行计算 ---
    async def calculate_atr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 ATR (平均真实波幅)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            logger.debug(f"计算 ATR 缺少必要列: high, low, close。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_atr():
                return df.ta.atr(length=period, high=df[high_col], low=df[low_col], close=df[close_col], append=False)
            atr_series = await asyncio.to_thread(_sync_atr)
            if atr_series is None or atr_series.empty:
                return None
            return pd.DataFrame({f'ATR_{period}': atr_series})
        except Exception as e:
            logger.error(f"计算 ATR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_boll_bands_and_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, close_col='close') -> Optional[pd.DataFrame]:
        """计算布林带 (BBANDS) 及其宽度 (BBW) 和百分比B (%B)"""
        if df is None or df.empty or close_col not in df.columns:
            logger.debug(f"计算布林带缺少必要列: {close_col}。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_bbands():
                return df.ta.bbands(length=period, std=std_dev, close=df[close_col], append=False)
            bbands_df = await asyncio.to_thread(_sync_bbands)
            if bbands_df is None or bbands_df.empty:
                return None
            lower_col_name_ta = f'BBL_{period}_{std_dev:.1f}'
            middle_col_name_ta = f'BBM_{period}_{std_dev:.1f}'
            upper_col_name_ta = f'BBU_{period}_{std_dev:.1f}'
            bbw_col_name_ta = f'BBB_{period}_{std_dev:.1f}'
            bbp_col_name_ta = f'BBP_{period}_{std_dev:.1f}'
            lower_col_name_out = f'BBL_{period}_{std_dev:.1f}'
            middle_col_name_out = f'BBM_{period}_{std_dev:.1f}'
            upper_col_name_out = f'BBU_{period}_{std_dev:.1f}'
            bbw_col_name_out = f'BBW_{period}_{std_dev:.1f}'
            bbp_col_name_out = f'BBP_{period}_{std_dev:.1f}'
            result_df = pd.DataFrame(index=df.index)
            has_any_col = False
            if lower_col_name_ta in bbands_df.columns:
                result_df[lower_col_name_out] = bbands_df[lower_col_name_ta]
                has_any_col = True
            if middle_col_name_ta in bbands_df.columns:
                result_df[middle_col_name_out] = bbands_df[middle_col_name_ta]
                has_any_col = True
            if upper_col_name_ta in bbands_df.columns:
                result_df[upper_col_name_out] = bbands_df[upper_col_name_ta]
                has_any_col = True
            if all(c in result_df.columns for c in [lower_col_name_out, middle_col_name_out, upper_col_name_out]):
                result_df[bbw_col_name_out] = np.where(
                    np.abs(result_df[middle_col_name_out]) > 1e-9,
                    (result_df[upper_col_name_out] - result_df[lower_col_name_out]) / result_df[middle_col_name_out],
                    np.nan
                )
                has_any_col = True
            elif bbw_col_name_ta in bbands_df.columns:
                result_df[bbw_col_name_out] = bbands_df[bbw_col_name_ta]
                has_any_col = True
            if bbp_col_name_ta in bbands_df.columns:
                result_df[bbp_col_name_out] = bbands_df[bbp_col_name_ta]
                has_any_col = True
            return result_df if has_any_col else None
        except Exception as e:
            logger.error(f"计算布林带及宽度 (周期 {period}, 标准差 {std_dev}) 出错: {e}", exc_info=True)
            return None

    async def calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20, window_type: Optional[str] = None, close_col='close', annual_factor: int = 252) -> Optional[pd.DataFrame]:
        """计算历史波动率 (HV)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        try:
            # --- 修改行: 将同步的计算逻辑移至线程中执行 ---
            def _sync_hv():
                log_returns = np.log(df[close_col] / df[close_col].shift(1))
                hv_series = log_returns.rolling(window=period, min_periods=max(1, int(period * 0.5))).std() * np.sqrt(annual_factor)
                return pd.DataFrame({f'HV_{period}': hv_series})
            return await asyncio.to_thread(_sync_hv)
        except Exception as e:
            logger.error(f"计算历史波动率 (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_keltner_channels(self, df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算肯特纳通道 (KC)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            logger.debug(f"计算肯特纳通道缺少必要列。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_kc():
                return df.ta.kc(high=df[high_col], low=df[low_col], close=df[close_col], length=ema_period, atr_length=atr_period, scalar=atr_multiplier, mamode="ema", append=False)
            kc_df = await asyncio.to_thread(_sync_kc)
            if kc_df is None or kc_df.empty:
                return None
            lower_col_ta = kc_df.columns[0]
            basis_col_ta = kc_df.columns[1]
            upper_col_ta = kc_df.columns[2]
            result_df = pd.DataFrame({
                f'KCL_{ema_period}_{atr_period}': kc_df[lower_col_ta],
                f'KCM_{ema_period}_{atr_period}': kc_df[basis_col_ta],
                f'KCU_{ema_period}_{atr_period}': kc_df[upper_col_ta]
            })
            return result_df
        except Exception as e:
            logger.error(f"计算肯特纳通道 (EMA周期 {ema_period}, ATR周期 {atr_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cci(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 CCI (商品渠道指数)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_cci():
                return df.ta.cci(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            cci_series = await asyncio.to_thread(_sync_cci)
            if cci_series is None or cci_series.empty: return None
            return pd.DataFrame({f'CCI_{period}': cci_series})
        except Exception as e:
            logger.error(f"计算 CCI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cmf(self, df: pd.DataFrame, period: int = 20, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 CMF (蔡金货币流量)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_cmf():
                return df.ta.cmf(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False)
            cmf_series = await asyncio.to_thread(_sync_cmf)
            if cmf_series is None or cmf_series.empty:
                return None
            return pd.DataFrame({f'CMF_{period}': cmf_series})
        except Exception as e:
            logger.error(f"计算 CMF (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_dmi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 DMI (动向指标), 包括 PDI (+DI), NDI (-DI), ADX"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_dmi():
                return ta.adx(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            dmi_df = await asyncio.to_thread(_sync_dmi)
            if dmi_df is None or dmi_df.empty:
                return None
            rename_map = {}
            if f'DMP_{period}' in dmi_df.columns:
                rename_map[f'DMP_{period}'] = f'PDI_{period}'
            if f'DMN_{period}' in dmi_df.columns:
                rename_map[f'DMN_{period}'] = f'NDI_{period}'
            if f'ADX_{period}' in dmi_df.columns:
                rename_map[f'ADX_{period}'] = f'ADX_{period}'
            return dmi_df.rename(columns=rename_map)
        except Exception as e:
            logger.error(f"计算 DMI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ichimoku(self, df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, chikou_period: Optional[int] = None, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算一目均衡表 (Ichimoku Cloud)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_ichimoku():
                return df.ta.ichimoku(high=df[high_col], low=df[low_col], close=df[close_col],
                                   tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period,
                                   append=False)
            ichi_df, _ = await asyncio.to_thread(_sync_ichimoku) # span DataFrame 未使用
            if ichi_df is None or ichi_df.empty:
                return None
            result_df = pd.DataFrame(index=df.index)
            if f'ITS_{tenkan_period}' in ichi_df.columns:
                result_df[f'TENKAN_{tenkan_period}'] = ichi_df[f'ITS_{tenkan_period}']
            if f'IKS_{kijun_period}' in ichi_df.columns:
                result_df[f'KIJUN_{kijun_period}'] = ichi_df[f'IKS_{kijun_period}']
            
            # Chikou Span (迟行线): 当前收盘价向后移动 kijun_period 个周期
            # pandas_ta 的 ICS_26 (假设 kijun_period=26) 是将收盘价向前移动26周期，所以是未来的价格在当前行
            # 通常Chikou是当前价格相对于历史价格的位置，或者说，历史价格的图形向未来平移。
            # 如果作为特征，通常是当前收盘价与 (kijun_period) 周期前的价格比较，或者将 (kijun_period) 周期前的收盘价作为特征。
            # pandas_ta 的 ICS_26 列是 close.shift(-26)，即把未来的价格放到当前行。
            # 我们这里计算的是将当前价格后移，作为特征。
            # 若用作图表绘制，Chikou Line 是当前 Close 向过去回移 kijun_period。
            # 作为特征，我们通常使用 close.shift(kijun_period)，即 kijun_period 之前的收盘价。
            # 但 pandas-ta ichimoku ICS_X 列是 `close.shift(-X)`。这里保持与 ta 一致但注意含义。
            if f'ICS_{kijun_period}' in ichi_df.columns:
                 result_df[f'CHIKOU_{kijun_period}'] = ichi_df[f'ICS_{kijun_period}'] # 这是将未来的收盘价移到当前时间点
            else: # 如果ta未提供，则手动计算 close.shift(-kijun_period)
                 result_df[f'CHIKOU_{kijun_period}'] = df[close_col].shift(-kijun_period)

            # Senkou Span A (先行上线A): (Tenkan + Kijun) / 2, 绘制在未来 kijun_period 的位置
            # 作为特征，使用当前计算值或未来 kijun_period 对应的值
            # pandas_ta ISA_X 是未平移的 (Tenkan + Kijun) / 2
            if f'ITS_{tenkan_period}' in ichi_df.columns and f'IKS_{kijun_period}' in ichi_df.columns:
                senkou_a_calc = (ichi_df[f'ITS_{tenkan_period}'] + ichi_df[f'IKS_{kijun_period}']) / 2
                result_df[f'SENKOU_A_{tenkan_period}_{kijun_period}'] = senkou_a_calc.shift(kijun_period) # 向前平移
            elif f'ISA_{tenkan_period}' in ichi_df.columns:
                result_df[f'SENKOU_A_{tenkan_period}_{kijun_period}'] = ichi_df[f'ISA_{tenkan_period}'].shift(kijun_period) # 向前平移

            # Senkou Span B (先行上线B): (过去 senkou_period周期最高价 + 最低价) / 2, 绘制在未来 kijun_period 的位置
            # pandas_ta ISB_X 是 (X周期高+X周期低)/2，未平移
            # 标准 Senkou B 使用 senkou_period (通常是52)
            def _sync_senkou_b_calc(): # 将 senkou B 的滚动计算也放入线程
                rolling_high_senkou = df[high_col].rolling(window=senkou_period, min_periods=1).max()
                rolling_low_senkou = df[low_col].rolling(window=senkou_period, min_periods=1).min()
                return (rolling_high_senkou + rolling_low_senkou) / 2
            
            if f'ISB_{kijun_period}' in ichi_df.columns and senkou_period == kijun_period:
                result_df[f'SENKOU_B_{kijun_period}'] = ichi_df[f'ISB_{kijun_period}'].shift(kijun_period) # 向前平移
            else:
                senkou_b_calc = await asyncio.to_thread(_sync_senkou_b_calc)
                result_df[f'SENKOU_B_{senkou_period}'] = senkou_b_calc.shift(kijun_period) # 向前平移
            return result_df if not result_df.empty else None
        except Exception as e:
            logger.error(f"计算 Ichimoku (t={tenkan_period}, k={kijun_period}, s={senkou_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_kdj(self, df: pd.DataFrame, period: int = 9, signal_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 KDJ 指标"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_stoch(): # KDJ基于Stochastic Oscillator
                return df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=period, d=signal_period, smooth_k=smooth_k_period, append=False)
            stoch_df = await asyncio.to_thread(_sync_stoch)
            if stoch_df is None or stoch_df.empty: return None
            k_col_name_ta = stoch_df.columns[0]
            d_col_name_ta = stoch_df.columns[1]
            kdj_df = pd.DataFrame(index=df.index)
            k_series = stoch_df[k_col_name_ta]
            d_series = stoch_df[d_col_name_ta]
            kdj_df[f'K_{period}_{signal_period}_{smooth_k_period}'] = k_series
            kdj_df[f'D_{period}_{signal_period}_{smooth_k_period}'] = d_series
            kdj_df[f'J_{period}_{signal_period}_{smooth_k_period}'] = 3 * k_series - 2 * d_series
            return kdj_df
        except Exception as e:
            logger.error(f"计算 KDJ (p={period}, sig={signal_period}, smooth={smooth_k_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ema(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 EMA (指数移动平均线)"""
        if df is None or df.empty or close_col not in df.columns:
            # logger.warning(f"calculate_ema: 输入 df 为 None、为空或 '{close_col}' 列不存在。") # 根据需要取消注释日志
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_ema():
                # df.ta.ema 需要 df[close_col] 是一个 Series
                # 如果 df[close_col] 由于某种原因不是 Series (例如 df 结构异常)，pandas-ta 内部会处理或报错
                return df.ta.ema(close=df[close_col], length=period, append=False)

            ema_series = await asyncio.to_thread(_sync_ema)

            # --- 调试信息 Start ---
            # print(f"DEBUG calculate_ema: period={period}, input df rows={len(df)}")
            # print(f"DEBUG calculate_ema: ema_series type: {type(ema_series)}")
            # if ema_series is not None:
            #     print(f"DEBUG calculate_ema: ema_series value (first 5 if Series else value): {ema_series.head(5) if isinstance(ema_series, pd.Series) else ema_series}")
            #     if hasattr(ema_series, 'empty'):
            #         print(f"DEBUG calculate_ema: ema_series.empty: {ema_series.empty}")
            #     if hasattr(ema_series, 'index'):
            #         print(f"DEBUG calculate_ema: ema_series.index: {ema_series.index}")
            #     from pandas.api.types import is_scalar
            #     print(f"DEBUG calculate_ema: is_scalar(ema_series): {is_scalar(ema_series)}")
            # else:
            #     print(f"DEBUG calculate_ema: ema_series is None")
            # --- 调试信息 End ---

            if ema_series is None: # 修改: 首先检查 ema_series 是否为 None
                # logger.warning(f"EMA (周期 {period}) 计算后 _sync_ema 返回 None. 输入 df 行数: {len(df)}.")
                return None

            if not isinstance(ema_series, pd.Series): # 修改: 严格检查返回类型是否为 pd.Series
                logger.error(f"EMA (周期 {period}) 计算结果不是 pandas.Series. "
                             f"实际类型: {type(ema_series)}, 值 (部分): {str(ema_series)[:200]}. "
                             f"输入 df 行数: {len(df)}.")
                return None

            if ema_series.empty: # 修改: 如果是 pd.Series，再检查是否为空
                # logger.warning(f"EMA (周期 {period}) 计算结果为 空的 pandas.Series. 输入 df 行数: {len(df)}.")
                return None
            
            # 此时 ema_series 是一个非空的 pd.Series
            return pd.DataFrame({f'EMA_{period}': ema_series})
        except AttributeError as ae: # 修改: 捕获 AttributeError，例如在非 Series 对象上调用 .empty
            logger.error(f"计算 EMA (周期 {period}) 时发生 AttributeError: {ae}. "
                         f"ema_series 当前类型: {type(ema_series)}, 值 (部分): {str(ema_series)[:200]}. "
                         f"这可能意味着 ema_series 不是预期的 Series 类型，并且类型检查未完全捕获。", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"计算 EMA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sma(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 SMA (简单移动平均线)"""
        if df is None or df.empty or close_col not in df.columns:
            # logger.warning(f"calculate_sma: 输入 df 为 None、为空或 '{close_col}' 列不存在。") # 根据需要取消注释日志
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_sma():
                return df.ta.sma(close=df[close_col], length=period, append=False)

            sma_series = await asyncio.to_thread(_sync_sma)
            
            # --- 调试信息 Start ---
            # print(f"DEBUG calculate_sma: period={period}, input df rows={len(df)}")
            # print(f"DEBUG calculate_sma: sma_series type: {type(sma_series)}")
            # if sma_series is not None:
            #     print(f"DEBUG calculate_sma: sma_series value (first 5 if Series else value): {sma_series.head(5) if isinstance(sma_series, pd.Series) else sma_series}")
            #     if hasattr(sma_series, 'empty'):
            #         print(f"DEBUG calculate_sma: sma_series.empty: {sma_series.empty}")
            #     if hasattr(sma_series, 'index'):
            #         print(f"DEBUG calculate_sma: sma_series.index: {sma_series.index}")
            #     from pandas.api.types import is_scalar
            #     print(f"DEBUG calculate_sma: is_scalar(sma_series): {is_scalar(sma_series)}")
            # else:
            #     print(f"DEBUG calculate_sma: sma_series is None")
            # --- 调试信息 End ---

            if sma_series is None: # 修改: 首先检查 sma_series 是否为 None
                # logger.warning(f"SMA (周期 {period}) 计算后 _sync_sma 返回 None. 输入 df 行数: {len(df)}.")
                return None

            if not isinstance(sma_series, pd.Series): # 修改: 严格检查返回类型是否为 pd.Series
                logger.error(f"SMA (周期 {period}) 计算结果不是 pandas.Series. "
                             f"实际类型: {type(sma_series)}, 值 (部分): {str(sma_series)[:200]}. "
                             f"输入 df 行数: {len(df)}.")
                return None

            if sma_series.empty: # 修改: 如果是 pd.Series，再检查是否为空
                # logger.warning(f"SMA (周期 {period}) 计算结果为 空的 pandas.Series. 输入 df 行数: {len(df)}.")
                return None

            # 此时 sma_series 是一个非空的 pd.Series
            return pd.DataFrame({f'SMA_{period}': sma_series})
        except AttributeError as ae: # 修改: 捕获 AttributeError
            logger.error(f"计算 SMA (周期 {period}) 时发生 AttributeError: {ae}. "
                         f"sma_series 当前类型: {type(sma_series)}, 值 (部分): {str(sma_series)[:200]}. ", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"计算 SMA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_amount_ma(self, df: pd.DataFrame, period: int = 20, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的移动平均线 (AMT_MA)"""
        if df is None or df.empty or amount_col not in df.columns: return None
        try:
            # --- 修改行: 将同步的 rolling 计算移至线程中执行 ---
            def _sync_amount_ma():
                return df[amount_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            amt_ma_series = await asyncio.to_thread(_sync_amount_ma)
            return pd.DataFrame({f'AMT_MA_{period}': amt_ma_series})
        except Exception as e:
            logger.error(f"计算 AMT_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_macd(self, df: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        """计算 MACD (异同移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_macd():
                return df.ta.macd(close=df[close_col], fast=period_fast, slow=period_slow, signal=signal_period, append=False)
            macd_df = await asyncio.to_thread(_sync_macd)
            if macd_df is None or macd_df.empty: return None
            return macd_df
        except Exception as e:
            logger.error(f"计算 MACD (f={period_fast},s={period_slow},sig={signal_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mfi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 MFI (资金流量指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_mfi():
                return df.ta.mfi(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False)
            mfi_series = await asyncio.to_thread(_sync_mfi)
            if mfi_series is None or mfi_series.empty: return None
            return pd.DataFrame({f'MFI_{period}': mfi_series})
        except Exception as e:
            logger.error(f"计算 MFI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mom(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 MOM (动量指标)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_mom():
                return df.ta.mom(close=df[close_col], length=period, append=False)
            mom_series = await asyncio.to_thread(_sync_mom)
            if mom_series is None or mom_series.empty: return None
            return pd.DataFrame({f'MOM_{period}': mom_series})
        except Exception as e:
            logger.error(f"计算 MOM (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_obv(self, df: pd.DataFrame, close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 OBV (能量潮指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [close_col, volume_col]): return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_obv():
                return df.ta.obv(close=df[close_col], volume=df[volume_col], append=False)
            obv_series = await asyncio.to_thread(_sync_obv)
            if obv_series is None or obv_series.empty: return None
            return pd.DataFrame({'OBV': obv_series})
        except Exception as e:
            logger.error(f"计算 OBV 出错: {e}", exc_info=True)
            return None

    async def calculate_roc(self, df: pd.DataFrame, period: int = 12, close_col='close') -> Optional[pd.DataFrame]:
        """计算 ROC (价格变化率)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_roc():
                return df.ta.roc(close=df[close_col], length=period, append=False)
            roc_series = await asyncio.to_thread(_sync_roc)
            if roc_series is None or roc_series.empty: return None
            return pd.DataFrame({f'ROC_{period}': roc_series})
        except Exception as e:
            logger.error(f"计算 ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_amount_roc(self, df: pd.DataFrame, period: int, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的 ROC (AROC)"""
        if df is None or df.empty or amount_col not in df.columns: return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_aroc():
                return df.ta.roc(close=df[amount_col], length=period, append=False)
            aroc_series = await asyncio.to_thread(_sync_aroc)
            if aroc_series is None or aroc_series.empty: return None
            df_results = pd.DataFrame({f'AROC_{period}': aroc_series})
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Amount ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_volume_roc(self, df: pd.DataFrame, period: int, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的 ROC (VROC)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_vroc():
                return df.ta.roc(close=df[volume_col], length=period, append=False)
            vroc_series = await asyncio.to_thread(_sync_vroc)
            if vroc_series is None or vroc_series.empty: return None
            df_results = pd.DataFrame({f'VROC_{period}': vroc_series})
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Volume ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_rsi(self, df: pd.DataFrame, period: int = 14, close_col='close') -> Optional[pd.DataFrame]:
        """计算 RSI (相对强弱指数)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_rsi():
                return df.ta.rsi(close=df[close_col], length=period, append=False)
            rsi_series = await asyncio.to_thread(_sync_rsi)
            if rsi_series is None or rsi_series.empty: return None
            return pd.DataFrame({f'RSI_{period}': rsi_series})
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
            # --- 修改行: 将同步的计算逻辑移至线程中执行 ---
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
            # --- 修改行: 将同步的 rolling 计算移至线程中执行 ---
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
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # --- 将同步的 pandas-ta 调用移至线程中执行 ---
            def _sync_willr():
                return df.ta.willr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            willr_series = await asyncio.to_thread(_sync_willr)
            if willr_series is None or willr_series.empty: return None
            return pd.DataFrame({f'WILLR_{period}': willr_series})
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


















