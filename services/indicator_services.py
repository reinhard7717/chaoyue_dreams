# services\indicator_services.py
import asyncio
from collections import defaultdict
import datetime
from functools import reduce
import json
import os
import warnings
import logging
import concurrent.futures # 修改行: 导入concurrent.futures
import multiprocessing # 修改行: 导入multiprocessing以获取cpu_count等
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

# 修改行: 定义模块级静态包装函数，用于在子进程中执行指标计算
def _static_indicator_execution_wrapper(
    indicator_function: Callable, # 这是一个静态方法引用
    stock_code_for_log: str,
    tf_calc: str,
    base_df_with_suffix: pd.DataFrame,
    config_params: Dict,
    original_indicator_name: str
) -> Union[Tuple[str, pd.DataFrame], Exception, None]: # 修改行: 返回类型可以包含Exception
    """
    在子进程中执行单个指标计算的静态包装函数。
    接收一个静态的指标计算函数、数据和参数。
    """
    # print(f"[{stock_code_for_log}] Subprocess PID {os.getpid()}: TF {tf_calc}, calculating {original_indicator_name} using {indicator_function.__name__ if hasattr(indicator_function, '__name__') else 'unknown_func'}")
    df_for_ta = base_df_with_suffix.copy()
    ohlcv_map_to_std = {
        f'open_{tf_calc}': 'open', f'high_{tf_calc}': 'high', f'low_{tf_calc}': 'low',
        f'close_{tf_calc}': 'close', f'volume_{tf_calc}': 'volume', f'amount_{tf_calc}': 'amount'
    }
    actual_rename_map_to_std = {k: v for k, v in ohlcv_map_to_std.items() if k in df_for_ta.columns}
    df_for_ta.rename(columns=actual_rename_map_to_std, inplace=True)

    required_cols_for_func = set(['high', 'low', 'close'])
    # 简化版必需列检查，实际应更精细
    if original_indicator_name in ['MFI', 'OBV', 'VWAP', 'CMF', 'VOL_MA', 'ADL', 'VROC', 'Volume ROC']:
        required_cols_for_func.add('volume')
    if original_indicator_name in ['AMT_MA', 'AROC', 'Amount ROC']:
        required_cols_for_func.add('amount')

    if not all(col in df_for_ta.columns for col in required_cols_for_func):
        missing_cols_str = ", ".join(list(required_cols_for_func - set(df_for_ta.columns)))
        # 注意: logger 在子进程中直接使用可能无效，除非特殊配置。使用print进行调试。
        print(f"[{stock_code_for_log}] Subprocess PID {os.getpid()}: TF {tf_calc}, Indicator {original_indicator_name}: Missing required columns ({missing_cols_str}). Available: {df_for_ta.columns.tolist()}. Skipping calculation.")
        return None

    try:
        indicator_result_df = indicator_function(df_for_ta, **config_params)
        
        if indicator_result_df is None or indicator_result_df.empty:
            return None
        
        if not isinstance(indicator_result_df, pd.DataFrame):
            if isinstance(indicator_result_df, pd.Series):
                series_name = indicator_result_df.name
                if not series_name:
                    # 构造一个可接受的默认名
                    param_str = "_".join(map(str, config_params.values())) if config_params else ""
                    series_name = f"{original_indicator_name}_{param_str}".rstrip('_') if param_str else original_indicator_name
                indicator_result_df = indicator_result_df.to_frame(name=series_name)
            else:
                print(f"[{stock_code_for_log}] Subprocess PID {os.getpid()}: TF {tf_calc}, Indicator {original_indicator_name}: Returned non-DataFrame, non-Series type {type(indicator_result_df)}. Skipping.")
                return None
        
        result_renamed_df = indicator_result_df.rename(columns=lambda x: f"{x}_{tf_calc}")
        return (tf_calc, result_renamed_df)
    except Exception as e_calc:
        # print(f"[{stock_code_for_log}] Subprocess PID {os.getpid()}: Error in TF {tf_calc}, Indicator {original_indicator_name} (params: {config_params}): {e_calc}")
        # 让主进程处理异常，这里直接返回异常对象
        return e_calc # 修改行: 返回异常本身

class IndicatorService:
    """
    技术指标计算服务 (使用 pandas-ta)
    负责获取多时间级别原始数据，进行时间序列标准化（重采样），计算指标，合并数据并进行最终填充。
    通过多进程并行处理数据获取、重采样和指标计算任务以提高效率。
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
            global ta # ta 在模块级别导入
            if ta is None: # 检查 ta 是否成功导入
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

    def _calculate_needed_bars_for_tf( self, target_tf: str, min_tf: str, base_needed_bars: int, global_max_lookback: int ) -> int:
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
        if target_tf_minutes >= 240: # 日线及以上级别数据获取逻辑
             needed = max(estimated_bars_needed, global_max_lookback * (target_tf_minutes / 240) + 100) # 按比例增加
             if target_tf_minutes >= 240: # 确保日周月有足够历史数据
                  needed = max(needed, global_max_lookback + 365*2) # 例如至少两年日线数据
        return math.ceil(needed)

    async def _get_ohlcv_data(self, stock_code: str, time_level: Union[TimeLevel, str], needed_bars: int) -> Optional[pd.DataFrame]:
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
        df = await self.indicator_dao.get_history_ohlcv_df(stock_code, time_level, limit=limit)
        if df is None or df.empty:
            logger.warning(f"[{stock_code}] 时间级别 {time_level} 无法获取足够的原始历史数据 (请求 {limit} 条)。")
            return None
        if 'vol' in df.columns and 'volume' not in df.columns:
            df.rename(columns={'vol': 'volume'}, inplace=True)
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 将原始数据列名 'vol' 重命名为 'volume'.")
        if 'amount_col_from_dao' in df.columns and 'amount' not in df.columns: 
            df.rename(columns={'amount_col_from_dao': 'amount'}, inplace=True)
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 示例性重命名 'amount_col_from_dao' 为 'amount'.")
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is not None:
             logger.info(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
             logger.warning(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，但索引不是时区感知的 DatetimeIndex。")
        return df
    
    # _resample_and_clean_dataframe 现在是一个普通方法，将被进程池调用
    def _resample_and_clean_dataframe(self, df: pd.DataFrame, tf: str, min_periods: int = 1, fill_method: str = 'ffill') -> Optional[pd.DataFrame]:
        """
        对原始 DataFrame 进行重采样到标准的 K 线时间点，并进行初步填充。
        此函数为同步函数，设计为可通过 ProcessPoolExecutor 在单独进程中运行以实现并行重采样。

        Args:
            df (pd.DataFrame): 从 DAO 获取的原始 DataFrame (索引是 DatetimeIndex, tz-aware)。
            tf (str): 目标时间级别字符串 (例如 '5', '15', 'D')。
            min_periods (int): 重采样聚合所需的最小原始数据点数量。
            fill_method (str): 重采样后填充 NaN 的方法 ('ffill', 'bfill', None)。

        Returns:
            Optional[pd.DataFrame]: 重采样并初步填充后的 DataFrame，如果失败则返回 None。
        """
        # print(f"Subprocess PID {os.getpid()} for resample: Resampling TF {tf} for index {df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}")
        if df is None or df.empty:
            return None
        freq = self._get_resample_freq_str(tf) # self._get_resample_freq_str 可以在子进程中调用，因为它不依赖不可pickle的实例状态
        if freq is None:
            # logger.error(...) # 在子进程中，logger 可能需要特殊处理
            print(f"Error in subprocess {os.getpid()}: TF {tf} for index {df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'}无法转换为有效的重采样频率。")
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
            print(f"Warning in subprocess {os.getpid()}: TF {tf} for index {df.index.name if hasattr(df.index, 'name') else 'UnknownIndex'} 没有找到可以聚合的列，无法进行重采样。")
            return None
        try:
            resampled_df = df.resample(freq, label='right', closed='right').agg(agg_rules, min_periods=min_periods)
            if resampled_df.empty:
                # logger.warning(...)
                return None
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            required_agg_cols = [col for col in required_cols if col in agg_rules]
            if resampled_df[required_agg_cols].isnull().all().all():
                # logger.warning(...)
                return None
            if fill_method == 'ffill':
                resampled_df.ffill(inplace=True)
            elif fill_method == 'bfill':
                resampled_df.bfill(inplace=True)
            
            missing_after_resample_fill = resampled_df.isnull().sum().sum()
            if missing_after_resample_fill > 0:
                 # logger.warning(...)
                 pass # 在子进程中可以简化日志记录或通过队列发送
            
            cols_to_suffix = ['open', 'high', 'low', 'close', 'volume', 'amount']
            rename_map_with_suffix = {col: f"{col}_{tf}" for col in cols_to_suffix if col in resampled_df.columns}
            resampled_df_renamed = resampled_df.rename(columns=rename_map_with_suffix)
            return resampled_df_renamed
        except Exception as e:
            # logger.error(...)
            print(f"Error in subprocess {os.getpid()} for resample TF {tf}: {e}")
            return None # 或者重新抛出异常，由主进程处理

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
            logger.info(f"已获取并合并指数/板块数据，数据量: {len(all_indices_df)} 条，列数: {len(all_indices_df.columns)}")
        else:
            logger.warning(f"未获取到任何指数/板块数据 for {stock_code}")
            all_indices_df = pd.DataFrame(index=df.index if not df.empty else None)

        if cyq_perf_df is not None and not cyq_perf_df.empty:
            cyq_perf_df.drop(columns=['stock_code'], inplace=True, errors='ignore')
            cyq_perf_df.columns = [f'cyq_{col}' for col in cyq_perf_df.columns]
            logger.info(f"已获取股票 {stock_code} 的筹码分布汇总数据，数据量: {len(cyq_perf_df)} 条，列数: {len(cyq_perf_df.columns)}")
        else:
            logger.warning(f"未获取到股票 {stock_code} 的筹码分布汇总数据")
            cyq_perf_df = pd.DataFrame(index=df.index if not df.empty else None)

        fund_flow_cnt_ths_dfs_list = [] # 修改行: 优化合并逻辑
        if fund_flow_cnt_ths_df_raw is not None and not fund_flow_cnt_ths_df_raw.empty:
             logger.info(f"已获取同花顺板块资金流向统计数据 (原始)，数据量: {len(fund_flow_cnt_ths_df_raw)} 条")
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
                 logger.info(f"已处理同花顺板块资金流向统计数据，数据量: {len(fund_flow_cnt_ths_df_processed)} 条，列数: {len(fund_flow_cnt_ths_df_processed.columns)}")
            else:
                 logger.warning(f"处理同花顺板块资金流向统计数据后 DataFrame 为空 for {ths_codes}")
        
        fund_flow_industry_ths_dfs_list = [] # 修改行: 优化合并逻辑
        if fund_flow_industry_ths_df_raw is not None and not fund_flow_industry_ths_df_raw.empty:
             logger.info(f"已获取同花顺行业资金流向统计数据 (原始)，数据量: {len(fund_flow_industry_ths_df_raw)} 条")
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
                 logger.info(f"已处理同花顺行业资金流向统计数据，数据量: {len(fund_flow_industry_ths_df_processed)} 条，列数: {len(fund_flow_industry_ths_df_processed.columns)}")
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
             logger.info(f"所有外部特征数据合并完成，数据量: {len(merged_external_df)} 条，列数: {len(merged_external_df.columns)}")
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

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, base_needed_bars: Optional[int] = None) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        根据策略 JSON 配置文件准备包含重采样基础数据和所有计算指标的 DataFrame。
        此方法通过并行化数据获取、重采样和指标计算来优化性能。

        执行步骤:
        1. 加载策略参数。
        2. 解析参数，识别时间级别和指标。
        3. 估算数据量。
        4. 并行获取原始 OHLCV 数据。
        5. (多进程)并行重采样和初步清洗原始数据。
        6. (多进程)并行计算所有配置的基础指标。
        7. 合并所有数据。
        8. 补充外部特征。
        9. 计算衍生特征。
        10. 最终缺失值填充。

        Args:
            stock_code (str): 股票代码。
            params_file (str): 策略 JSON 配置文件路径。
            base_needed_bars (Optional[int]): 基础所需最小时间级别数据条数，覆盖文件配置。

        Returns:
            Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]: 包含所有数据的 DataFrame 和指标配置列表，
                                                                如果准备失败则返回 None。
        """
        if 'ta' not in globals() or ta is None: # 检查 ta 是否在全局（主进程）加载
            print(f"[{stock_code}] Debug: pandas_ta 未加载。")
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None
        try:
            if not os.path.exists(params_file):
                print(f"[{stock_code}] Debug: 策略参数文件未找到: {params_file}")
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None
            with open(params_file, 'r', encoding='utf-8') as f:
                params_json_config = json.load(f) # 修改变量名 params 为 params_json_config 避免与参数 params 混淆
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
        except Exception as e:
            print(f"[{stock_code}] Debug: 加载或解析参数文件失败: {e}")
            logger.error(f"[{stock_code}] 加载或解析参数文件 {params_file} 失败: {e}", exc_info=True)
            return None

        # 修改行: 初始化 ProcessPoolExecutor
        # 使用cpu核心数，或者可以根据实际情况调整
        num_workers = multiprocessing.cpu_count() 
        # print(f"[{stock_code}] Debug: Initializing ProcessPoolExecutor with {num_workers} workers.")
        # executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) # 在 try-finally 中创建和关闭
        loop = asyncio.get_event_loop()

        main_index_codes = params_json_config.get('base_scoring', {}).get('main_index_codes', [])
        if not main_index_codes:
             logger.warning(f"[{stock_code}] JSON 参数中未配置 'base_scoring.main_index_codes'，相对强度计算将仅使用股票所属板块作为基准。")
        fe_params = params_json_config.get('feature_engineering_params', {})
        external_data_history_days = fe_params.get('external_data_history_days', 365)
        
        all_time_levels_needed: Set[str] = set()
        indicator_configs: List[Dict[str, Any]] = []

        # _add_indicator_config 内部注册的是静态方法引用
        def _add_indicator_config(name: str, func: Callable, param_block_key: Optional[str], params_dict: Dict, applicable_tfs: Union[str, List[str]], param_override_key: Optional[str] = None):
            tfs = [applicable_tfs] if isinstance(applicable_tfs, str) else applicable_tfs
            all_time_levels_needed.update(tfs)
            indicator_configs.append({
                'name': name, # 指标的原始名称，如 "MACD", "RSI"
                'func': func, # 指向静态的 IndicatorService.calculate_xyz 方法
                'params': params_dict,
                'timeframes': tfs,
                'param_block_key': param_block_key,
                'param_override_key': param_override_key
            })
        
        def _get_indicator_params(param_block: Dict, default_params: Dict, param_override_key: Optional[str] = None) -> Dict:
             indi_specific_params_json = param_block.get(param_override_key, param_block) if param_override_key else param_block
             final_calc_params = default_params.copy()
             for k, v_json in indi_specific_params_json.items():
                 if k in final_calc_params: # 只覆盖默认参数中存在的键
                      final_calc_params[k] = v_json
             return final_calc_params

        bs_params = params_json_config.get('base_scoring', {})
        bs_timeframes = bs_params.get('timeframes', ['5', '15', '30', '60', 'D'])
        all_time_levels_needed.update(bs_timeframes)

        default_macd_p = {'period_fast': 12, 'period_slow': 26, 'signal_period': 9}
        default_rsi_p = {'period': 14}
        default_kdj_p = {'period': 9, 'signal_period': 3, 'smooth_k_period': 3} # 这些参数对应stoch的k,d,smooth_k
        default_boll_p = {'period': 15, 'std_dev': 2.2}
        default_cci_p = {'period': 14}
        default_mfi_p = {'period': 14}
        default_roc_p = {'period': 12}
        default_dmi_p = {'period': 14}
        default_sar_p = {'af_step': 0.02, 'max_af': 0.2}
        default_stoch_p = {'k_period': 14, 'd_period': 3, 'smooth_k_period': 3}
        default_atr_p = {'period': 14}
        default_hv_p = {'period': 20, 'annual_factor': 252} # hv参数
        default_kc_p = {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0}
        default_mom_p = {'period': 10}
        default_willr_p = {'period': 14}
        default_sma_ema_p = {'period': 20}
        default_ichimoku_p = {'tenkan_period': 9, 'kijun_period': 26, 'senkou_period': 52}

        # 注册指标计算函数 (使用静态方法)
        for indi_key in bs_params.get('score_indicators', []):
            if indi_key == 'macd':
                macd_calc_params = {
                    'period_fast': bs_params.get('macd_fast', default_macd_p['period_fast']),
                    'period_slow': bs_params.get('macd_slow', default_macd_p['period_slow']),
                    'signal_period': bs_params.get('macd_signal', default_macd_p['signal_period'])
                }
                _add_indicator_config('MACD', IndicatorService.calculate_macd, 'base_scoring', macd_calc_params, bs_timeframes)
            elif indi_key == 'rsi':
                calc_params = {'period': bs_params.get('rsi_period', default_rsi_p['period'])}
                _add_indicator_config('RSI', IndicatorService.calculate_rsi, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'kdj': # KDJ 参数对应 STOCH 的参数
                calc_params = {
                    'period': bs_params.get('kdj_period_k', default_kdj_p['period']), # k_period for stoch
                    'signal_period': bs_params.get('kdj_period_d', default_kdj_p['signal_period']), # d_period for stoch
                    'smooth_k_period': bs_params.get('kdj_period_j', default_kdj_p['smooth_k_period']) # smooth_k for stoch
                }
                _add_indicator_config('KDJ', IndicatorService.calculate_kdj, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'boll':
                calc_params = {
                    'period': bs_params.get('boll_period', default_boll_p['period']),
                    'std_dev': bs_params.get('boll_std_dev', default_boll_p['std_dev'])
                }
                _add_indicator_config('BOLL', IndicatorService.calculate_boll_bands_and_width, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'cci':
                calc_params = {'period': bs_params.get('cci_period', default_cci_p['period'])}
                _add_indicator_config('CCI', IndicatorService.calculate_cci, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'mfi':
                calc_params = {'period': bs_params.get('mfi_period', default_mfi_p['period'])}
                _add_indicator_config('MFI', IndicatorService.calculate_mfi, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'roc':
                calc_params = {'period': bs_params.get('roc_period', default_roc_p['period'])}
                _add_indicator_config('ROC', IndicatorService.calculate_roc, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'dmi':
                calc_params = {'period': bs_params.get('dmi_period', default_dmi_p['period'])}
                _add_indicator_config('DMI', IndicatorService.calculate_dmi, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'sar':
                calc_params = {
                    'af_step': bs_params.get('sar_step', default_sar_p['af_step']),
                    'max_af': bs_params.get('sar_max', default_sar_p['max_af'])
                }
                _add_indicator_config('SAR', IndicatorService.calculate_sar, 'base_scoring', calc_params, bs_timeframes)
            elif indi_key == 'ema':
                ema_p = bs_params.get('ema_period', default_sma_ema_p['period'])
                _add_indicator_config('EMA', IndicatorService.calculate_ema, 'base_scoring', {'period': ema_p}, bs_timeframes, param_override_key='ema_params')
            elif indi_key == 'sma':
                sma_p = bs_params.get('sma_period', default_sma_ema_p['period'])
                _add_indicator_config('SMA', IndicatorService.calculate_sma, 'base_scoring', {'period': sma_p}, bs_timeframes, param_override_key='sma_params')
        
        vc_params = params_json_config.get('volume_confirmation', {})
        ia_params = params_json_config.get('indicator_analysis_params', {})
        vol_ana_tf_cfg = vc_params.get('timeframes', bs_timeframes)
        vol_ana_tfs_vc = [vol_ana_tf_cfg] if isinstance(vol_ana_tf_cfg, str) else vol_ana_tf_cfg if vc_params.get('enabled', False) else []
        ia_tfs_cfg = ia_params.get('timeframes', bs_timeframes)
        ia_tfs = [ia_tfs_cfg] if isinstance(ia_tfs_cfg, str) else ia_tfs_cfg if ia_params else []
        target_vol_ana_tfs = list(set(vol_ana_tfs_vc) | set(ia_tfs) | set(bs_timeframes))
        all_time_levels_needed.update(target_vol_ana_tfs)

        if vc_params.get('enabled', False) or ia_params.get('calculate_amt_ma', False):
             amt_ma_p = vc_params.get('amount_ma_period', ia_params.get('amount_ma_period', 20))
             _add_indicator_config('AMT_MA', IndicatorService.calculate_amount_ma, 'volume_confirmation' if vc_params.get('enabled', False) else 'indicator_analysis_params', {'period': amt_ma_p}, target_vol_ana_tfs, param_override_key='amount_ma_params')
        if vc_params.get('enabled', False) or ia_params.get('calculate_cmf', False):
            cmf_p = vc_params.get('cmf_period', ia_params.get('cmf_period', 20))
            _add_indicator_config('CMF', IndicatorService.calculate_cmf, 'volume_confirmation' if vc_params.get('enabled', False) else 'indicator_analysis_params', {'period': cmf_p}, target_vol_ana_tfs, param_override_key='cmf_params')
        if ia_params.get('calculate_vol_ma', False):
            vol_ma_p = ia_params.get('volume_ma_period', 20)
            _add_indicator_config('VOL_MA', IndicatorService.calculate_vol_ma, 'indicator_analysis_params', {'period': vol_ma_p}, target_vol_ana_tfs, param_override_key='volume_ma_params')

        ia_timeframes = ia_params.get('timeframes', bs_timeframes)
        all_time_levels_needed.update(ia_timeframes)
        if ia_params.get('calculate_stoch', False):
            stoch_calc_params = _get_indicator_params(ia_params, default_stoch_p, param_override_key='stoch_params')
            _add_indicator_config('STOCH', IndicatorService.calculate_stoch, 'indicator_analysis_params', stoch_calc_params, ia_timeframes)
        if ia_params.get('calculate_vwap', False):
            vwap_calc_params = _get_indicator_params(ia_params, {'anchor': None}, param_override_key='vwap_params')
            _add_indicator_config('VWAP', IndicatorService.calculate_vwap, 'indicator_analysis_params', vwap_calc_params, ia_timeframes)
        if ia_params.get('calculate_adl', False):
            _add_indicator_config('ADL', IndicatorService.calculate_adl, 'indicator_analysis_params', {}, ia_timeframes, param_override_key='adl_params')
        if ia_params.get('calculate_ichimoku', False):
            ichimoku_calc_params = _get_indicator_params(ia_params, default_ichimoku_p, param_override_key='ichimoku_params')
            _add_indicator_config('Ichimoku', IndicatorService.calculate_ichimoku, 'indicator_analysis_params', ichimoku_calc_params, ia_timeframes)
        if ia_params.get('calculate_pivot_points', False): # PivotPoints 仅用于日线
            pivot_calc_params = _get_indicator_params(ia_params, {}, param_override_key='pivot_params')
            _add_indicator_config('PivotPoints', IndicatorService.calculate_pivot_points, 'indicator_analysis_params', pivot_calc_params, ['D'])
            all_time_levels_needed.add('D')

        fe_timeframes_cfg = fe_params.get('apply_on_timeframes', bs_timeframes)
        fe_timeframes = [fe_timeframes_cfg] if isinstance(fe_timeframes_cfg, str) else fe_timeframes_cfg if fe_params else []
        all_time_levels_needed.update(fe_timeframes)
        if fe_params.get('calculate_atr', False):
             atr_calc_params = _get_indicator_params(fe_params, default_atr_p, param_override_key='atr_params')
             _add_indicator_config('ATR', IndicatorService.calculate_atr, 'feature_engineering_params', atr_calc_params, fe_timeframes)
        if fe_params.get('calculate_hv', False): # Historical Volatility
             hv_calc_params = _get_indicator_params(fe_params, default_hv_p, param_override_key='hv_params')
             _add_indicator_config('HV', IndicatorService.calculate_historical_volatility, 'feature_engineering_params', hv_calc_params, fe_timeframes)
        if fe_params.get('calculate_kc', False): # Keltner Channels
             kc_calc_params = _get_indicator_params(fe_params, default_kc_p, param_override_key='kc_params')
             _add_indicator_config('KC', IndicatorService.calculate_keltner_channels, 'feature_engineering_params', kc_calc_params, fe_timeframes)
        if fe_params.get('calculate_mom', False):
             mom_calc_params = _get_indicator_params(fe_params, default_mom_p, param_override_key='mom_params')
             _add_indicator_config('MOM', IndicatorService.calculate_mom, 'feature_engineering_params', mom_calc_params, fe_timeframes)
        if fe_params.get('calculate_willr', False):
             willr_calc_params = _get_indicator_params(fe_params, default_willr_p, param_override_key='willr_params')
             _add_indicator_config('WILLR', IndicatorService.calculate_willr, 'feature_engineering_params', willr_calc_params, fe_timeframes)
        if fe_params.get('calculate_vroc', False): # Volume ROC
             vroc_calc_params = _get_indicator_params(fe_params, default_roc_p, param_override_key='vroc_params') # 使用 roc 的默认参数
             _add_indicator_config('VROC', IndicatorService.calculate_volume_roc, 'feature_engineering_params', vroc_calc_params, fe_timeframes)
        if fe_params.get('calculate_aroc', False): # Amount ROC
             aroc_calc_params = _get_indicator_params(fe_params, default_roc_p, param_override_key='aroc_params') # 使用 roc 的默认参数
             _add_indicator_config('AROC', IndicatorService.calculate_amount_roc, 'feature_engineering_params', aroc_calc_params, fe_timeframes)

        for ma_type, ma_func_static in [('EMA', IndicatorService.calculate_ema), ('SMA', IndicatorService.calculate_sma)]:
            ma_periods = fe_params.get(f'{ma_type.lower()}_periods', [])
            if ma_periods: # 如果明确配置了 period 列表
                for p_val in ma_periods:
                     if isinstance(p_val, int) and p_val > 0:
                         _add_indicator_config(ma_type, ma_func_static, 'feature_engineering_params', {'period': p_val}, fe_timeframes)
            elif fe_params.get(f'calculate_{ma_type.lower()}', False): # 如果仅配置了 calculate_xx: true
                 ma_p = fe_params.get(f'{ma_type.lower()}_period', default_sma_ema_p['period']) # 使用默认周期或指定单一周期
                 _add_indicator_config(ma_type, ma_func_static, 'feature_engineering_params', {'period': ma_p}, fe_timeframes)
        
        # 确保 OBV 总是被计算（如果JSON中没有指定）
        if not any(conf['name'] == 'OBV' for conf in indicator_configs):
            _add_indicator_config('OBV', IndicatorService.calculate_obv, None, {}, list(all_time_levels_needed))
        
        focus_tf = params_json_config.get('trend_following_params', {}).get('focus_timeframe', '30')
        # print(f"[{stock_code}] Debug: 策略关注的时间级别 (focus_tf): {focus_tf}")
        # print(f"[{stock_code}] Debug: 所有策略所需时间级别集合: {sorted(list(all_time_levels_needed))}")

        min_time_level = None
        min_tf_minutes = float('inf')
        if not all_time_levels_needed:
            logger.error(f"[{stock_code}] 未能从参数文件中确定任何需要的时间级别。")
            return None
        for tf_str_loop in all_time_levels_needed: # 重命名变量避免与外部 tf 混淆
            minutes = self._get_timeframe_in_minutes(tf_str_loop)
            if minutes is not None and minutes < min_tf_minutes:
                 min_tf_minutes = minutes
                 min_time_level = tf_str_loop
        if min_time_level is None and bs_timeframes: # 如果 all_time_levels_needed 为空或都是无效tf
             min_time_level = bs_timeframes[0] # 回退
             # print(f"[{stock_code}] Debug: 无法精确确定最小时间级别 (按分钟)，回退使用基础时间级别列表的第一个: {min_time_level}")
        elif min_time_level is None:
            logger.error(f"[{stock_code}] 无法确定有效的最小时间级别从所需级别: {all_time_levels_needed}")
            return None
        logger.info(f"[{stock_code}] 策略所需时间级别: {sorted(list(all_time_levels_needed))}, 最小时间级别: {min_time_level} ({min_tf_minutes if min_tf_minutes != float('inf') else 'N/A'} 分钟)")

        global_max_lookback = 0
        unique_configs_for_lookback = {} # 使用字典去重配置
        for config in indicator_configs:
            if config.get('param_block_key') is None and config['name'] != 'OBV': # 只考虑来自JSON配置的指标用于回看期
                 continue
            # 使用参数的哈希able形式作为键的一部分
            params_hashable = tuple(sorted(config['params'].items()))
            key = (config['name'], params_hashable) # 用 (指标名, 参数元组) 作为唯一键
            if key not in unique_configs_for_lookback:
                 unique_configs_for_lookback[key] = config
        
        # print(f"[{stock_code}] Debug: 用于回看期计算的唯一指标配置数量: {len(unique_configs_for_lookback)}")
        for config in unique_configs_for_lookback.values(): # 遍历去重后的配置
            current_max_period = 0
            # 简化查找逻辑，直接检查常用周期参数名
            period_keys = ['period', 'period_fast', 'period_slow', 'signal_period', 'k_period', 'd_period', 'smooth_k_period',
                           'ema_period', 'atr_period', 'tenkan_period', 'kijun_period', 'senkou_period'] # 移除了 atr_multiplier
            for p_key in period_keys:
                if p_key in config['params'] and isinstance(config['params'][p_key], (int, float)):
                    current_max_period = max(current_max_period, int(config['params'][p_key]))
            # 特殊指标的回看期处理 (基于其典型计算方式)
            if config['name'] == 'MACD': # MACD 通常需要 slow + signal
                 p_macd = config['params']
                 current_max_period = max(current_max_period, p_macd.get('period_slow',0) + p_macd.get('signal_period',0))
            if config['name'] == 'DMI' and 'period' in config['params']: # DMI/ADX 通常需要更多数据来稳定
                current_max_period = max(current_max_period, int(config['params']['period'] * 2.5 + 10) ) # 经验值
            if config['name'] == 'KC': # Keltner Channel
                 p_kc = config['params'] # ema_period, atr_period
                 current_max_period = max(current_max_period, p_kc.get('ema_period', 0), p_kc.get('atr_period', 0))
            if config['name'] == 'Ichimoku':
                 p_ichi = config['params']
                 current_max_period = max(current_max_period, p_ichi.get('tenkan_period', 0), p_ichi.get('kijun_period', 0), p_ichi.get('senkou_period', 0))
            # 其他指标如RSI, SMA, EMA, CCI, MFI, ROC, ATR, MOM, WILLR, STOCH, SAR, VOL_MA, AMT_MA, VROC, AROC 的周期已在上面通用period_keys中处理
            global_max_lookback = max(global_max_lookback, current_max_period)
        
        global_max_lookback += 100 # 增加一个缓冲期
        logger.info(f"[{stock_code}] 动态计算的全局指标最大回看期 (含缓冲): {global_max_lookback}")

        ohlcv_tasks = {}
        lstm_window_size = params_json_config.get('lstm_training_config',{}).get('lstm_window_size', 60)
        # effective_base_needed_bars 是最小时间级别需要的数据条数，用于后续的对其和切片
        effective_base_needed_bars = base_needed_bars if base_needed_bars is not None else \
                                     lstm_window_size + global_max_lookback + 500 # 额外的500条作为通用缓冲

        for tf_fetch in all_time_levels_needed:
            needed_bars_for_tf = self._calculate_needed_bars_for_tf(
                target_tf=tf_fetch, min_tf=min_time_level,
                base_needed_bars=effective_base_needed_bars, # 使用估算的最小级别所需条数
                global_max_lookback=global_max_lookback
            )
            logger.info(f"[{stock_code}] 时间级别 {tf_fetch}: 基础({min_time_level})需(估算){effective_base_needed_bars}条, 指标需{global_max_lookback}条 -> 动态计算需获取 {needed_bars_for_tf} 条原始数据.")
            ohlcv_tasks[tf_fetch] = self._get_ohlcv_data(stock_code, tf_fetch, needed_bars=needed_bars_for_tf)
        
        # 并行获取所有时间级别的原始 OHLCV 数据
        ohlcv_results = await asyncio.gather(*ohlcv_tasks.values())
        raw_ohlcv_dfs = dict(zip(all_time_levels_needed, ohlcv_results))
        # print(f"[{stock_code}] Debug: 原始 OHLCV 数据获取完成。各级别数据概览:")
        # for tf_debug, df_debug in raw_ohlcv_dfs.items():
        #     print(f"  - TF {tf_debug}: {'None/Empty' if df_debug is None or df_debug.empty else f'Shape {df_debug.shape}, Columns: {df_debug.columns.tolist()}'}")
        
        # --- 修改开始: 使用 ProcessPoolExecutor 并行化重采样 ---
        resampled_ohlcv_dfs = {}
        min_usable_bars = math.ceil(effective_base_needed_bars * 0.6) # 最小可用数据量阈值
        
        resample_futures_map = {} # 使用字典存储 future 和对应的 tf
        valid_raw_dfs_for_resample = {} 

        process_executor = None # 在 try 块中初始化
        try: # 修改行: 添加 try-finally 以确保 executor 关闭
            process_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
            for tf_resample, raw_df_item in raw_ohlcv_dfs.items():
                if raw_df_item is None or raw_df_item.empty:
                    logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 没有获取到原始数据，跳过重采样。")
                    resampled_ohlcv_dfs[tf_resample] = None
                    continue
                valid_raw_dfs_for_resample[tf_resample] = raw_df_item # 存储有效的原始DataFrame
                # 提交到进程池执行，self._resample_and_clean_dataframe 是同步方法
                # 注意：传递 raw_df_item.copy() 以确保原始数据在主进程中不受影响 (如果子进程修改它)
                # 并且可以避免一些潜在的pickle问题如果df被多次引用且状态复杂。
                # print(f"[{stock_code}] Debug: Submitting resample task for TF {tf_resample} to ProcessPoolExecutor.")
                future = loop.run_in_executor(
                    process_executor,
                    self._resample_and_clean_dataframe, # 实例方法，self 会被 pickle（如果方法内不访问不可pickle的self成员则还好）
                                                      # _resample_and_clean_dataframe设计上不依赖太多self状态
                    raw_df_item.copy(), # 传递副本
                    tf_resample,
                    1, # min_periods
                    'ffill' # fill_method
                )
                resample_futures_map[tf_resample] = future
            
            if resample_futures_map:
                # 等待所有重采样任务完成
                tf_keys_ordered = list(resample_futures_map.keys())
                gathered_resample_results_or_exceptions = await asyncio.gather(
                    *[resample_futures_map[tf] for tf in tf_keys_ordered], 
                    return_exceptions=True
                )

                for i, tf_resample_key in enumerate(tf_keys_ordered):
                    result_or_exc = gathered_resample_results_or_exceptions[i]
                    if isinstance(result_or_exc, Exception):
                        logger.error(f"[{stock_code}] 时间级别 {tf_resample_key} 重采样任务失败: {result_or_exc}", exc_info=result_or_exc)
                        resampled_ohlcv_dfs[tf_resample_key] = None
                        continue
                    
                    resampled_df_processed = result_or_exc
                    if resampled_df_processed is None or resampled_df_processed.empty:
                        logger.warning(f"[{stock_code}] 时间级别 {tf_resample_key} 重采样后数据为空，跳过。")
                        resampled_ohlcv_dfs[tf_resample_key] = None
                        continue
                    
                    if tf_resample_key == min_time_level and len(resampled_df_processed) < min_usable_bars:
                        logger.error(f"[{stock_code}] 最小时间级别 {tf_resample_key} 重采样后数据量 {len(resampled_df_processed)} 条，少于最低可用阈值 {min_usable_bars} 条。无法继续。")
                        return None 
                    
                    if len(resampled_df_processed) < global_max_lookback * 0.5: # 检查数据量是否过少
                        logger.warning(f"[{stock_code}] 时间级别 {tf_resample_key} 重采样后数据量 {len(resampled_df_processed)} 条，可能少于指标最大回看期 {global_max_lookback} 的一半。计算的指标可能不可靠。")
                    
                    # print(f"[{stock_code}] Debug: TF {tf_resample_key} 重采样并重命名后的列: {resampled_df_processed.columns.tolist()[:10]}...")
                    resampled_ohlcv_dfs[tf_resample_key] = resampled_df_processed
            # --- 修改结束: 使用 ProcessPoolExecutor 并行化重采样 ---

            if min_time_level not in resampled_ohlcv_dfs or \
               resampled_ohlcv_dfs.get(min_time_level) is None or \
               resampled_ohlcv_dfs.get(min_time_level).empty:
                 logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 的重采样 OHLCV 数据不可用，无法进行合并。")
                 return None
            base_index = resampled_ohlcv_dfs[min_time_level].index
            logger.info(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的重采样索引作为合并基准，数量: {len(base_index)}。")

            # --- 修改开始: 使用 ProcessPoolExecutor 并行化指标计算 ---
            indicator_calculation_futures = [] # 修改行: 存储 futures
            # print(f"[{stock_code}] Debug: Preparing indicator calculation tasks for ProcessPoolExecutor.")
            for config_item_loop in indicator_configs:
                for tf_conf in config_item_loop['timeframes']:
                    if tf_conf in resampled_ohlcv_dfs and resampled_ohlcv_dfs[tf_conf] is not None and not resampled_ohlcv_dfs[tf_conf].empty:
                        base_ohlcv_df_for_tf_loop = resampled_ohlcv_dfs[tf_conf]
                        # print(f"[{stock_code}] Debug: Submitting indicator {config_item_loop['name']} for TF {tf_conf} to ProcessPoolExecutor.")
                        future = loop.run_in_executor(
                            process_executor,
                            _static_indicator_execution_wrapper, # 调用模块级/静态包装器
                            config_item_loop['func'],       # 静态指标计算方法
                            stock_code,                     # stock_code 用于日志
                            tf_conf,                        # 当前时间级别
                            base_ohlcv_df_for_tf_loop.copy(),# 传递DataFrame副本
                            config_item_loop['params'],     # 指标参数
                            config_item_loop['name']        # 原始指标名 (用于日志/列名)
                        )
                        indicator_calculation_futures.append(future)
                    else:
                        # print(f"[{stock_code}] Debug: 时间框架 {tf_conf} 的基础数据未找到或无效 ({config_item_loop['name']})，无法创建计算任务。")
                        logger.warning(f"[{stock_code}] 时间框架 {tf_conf} 在 resampled_ohlcv_dfs 中未找到有效数据，无法为指标 {config_item_loop['name']} 创建计算任务。")
            
            calculated_results_or_exceptions = [] # 修改行: 存储结果或异常
            if indicator_calculation_futures:
                # print(f"[{stock_code}] Debug: Waiting for {len(indicator_calculation_futures)} indicator calculation tasks to complete.")
                calculated_results_or_exceptions = await asyncio.gather(*indicator_calculation_futures, return_exceptions=False) # return_exceptions=False 会让第一个异常直接打断 gather
                                                                                                                            # 改为 True 以收集所有结果/异常
                # gathered_results = await asyncio.gather(*indicator_calculation_futures, return_exceptions=True)
                # print(f"[{stock_code}] Debug: All indicator calculation tasks completed.")


            calculated_indicators_by_tf = defaultdict(list)
            for res_item in calculated_results_or_exceptions: # 修改行: 变量名
                if isinstance(res_item, Exception): # 修改行: 处理直接返回的异常
                    logger.error(f"[{stock_code}] 指标计算子进程任务返回异常: {res_item}", exc_info=res_item if isinstance(res_item, BaseException) else None)
                    # print(f"[{stock_code}] Debug: 指标计算任务发生异常: {res_item}")
                elif isinstance(res_item, tuple) and len(res_item) == 2:
                    tf_res, indi_df_res = res_item
                    if indi_df_res is not None and not indi_df_res.empty:
                        calculated_indicators_by_tf[tf_res].append(indi_df_res)
                elif res_item is None: # Wrapper 可能返回 None
                    pass # 已在wrapper中打印日志
                else:
                    logger.warning(f"[{stock_code}] 指标计算任务返回非预期结果类型: {type(res_item)}, 内容: {str(res_item)[:200]}")
                    # print(f"[{stock_code}] Debug: 指标计算任务返回非预期结果: {res_item}")
            # --- 修改结束: 使用 ProcessPoolExecutor 并行化指标计算 ---

            final_df = resampled_ohlcv_dfs.get(min_time_level) # 初始的 final_df
            if final_df is None or final_df.empty:
                 logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 的重采样 OHLCV 数据在合并前不可用。")
                 return None
            
            dfs_to_merge = [final_df]
            # 合并其他时间级别的基础OHLCV数据 (已重命名列)
            for tf_merge, df_to_merge in resampled_ohlcv_dfs.items():
                 if tf_merge != min_time_level and df_to_merge is not None and not df_to_merge.empty:
                      dfs_to_merge.append(df_to_merge)
            
            # 合并计算得到的指标数据 (已重命名列)
            for tf_indi, indi_dfs_list in calculated_indicators_by_tf.items():
                 if indi_dfs_list:
                      # concat 之前检查是否有重复列，理论上不应有，因为列名已包含TF和指标特有参数
                      # merged_indi_df_for_tf = pd.concat(indi_dfs_list, axis=1)
                      # 为了更安全地处理潜在的重复列（尽管不太可能），可以使用合并
                      merged_indi_df_for_tf = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer', suffixes=(False, False)), indi_dfs_list)

                      if not merged_indi_df_for_tf.empty:
                           dfs_to_merge.append(merged_indi_df_for_tf)
                      else:
                           logger.warning(f"[{stock_code}] 时间级别 {tf_indi} 的指标数据合并后为空。")

            if not dfs_to_merge or dfs_to_merge[0] is None or dfs_to_merge[0].empty : # 确保第一个df有效
                logger.error(f"[{stock_code}] 没有可合并的数据，或基础数据无效。")
                return None

            # 使用 reduce 进行左连接合并，以 final_df (最小时间级别) 为基础
            # 确保索引都是 DatetimeIndex 且时区感知性一致 (resample 已处理)
            try:
                final_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='left', suffixes=(False, '_dup')), dfs_to_merge)
                # 检查是否有因suffixes产生的 '_dup' 列，理论上不应有太多
                dup_cols = [col for col in final_df.columns if '_dup' in col]
                if dup_cols:
                    logger.warning(f"[{stock_code}] 合并时产生重复列: {dup_cols}。请检查列命名和合并逻辑。")

            except Exception as merge_exc:
                logger.error(f"[{stock_code}] 合并所有 OHLCV 和指标数据时出错: {merge_exc}", exc_info=True)
                # print(f"[{stock_code}] Debug: 合并数据时出错: {merge_exc}")
                # print(f"[{stock_code}] Debug: 待合并的DataFrame数量: {len(dfs_to_merge)}")
                # for i, df_item in enumerate(dfs_to_merge):
                #     print(f"  DataFrame {i}: {'None' if df_item is None else df_item.shape}, Index type: {type(df_item.index) if df_item is not None else 'N/A'}, Columns (first 5): {df_item.columns[:5].tolist() if df_item is not None and not df_item.empty else 'N/A'}")
                return None

            if final_df is None or final_df.empty:
                 logger.error(f"[{stock_code}] 合并所有 OHLCV 和指标数据后 DataFrame 为空。")
                 return None
            logger.info(f"[{stock_code}] 所有 OHLCV 和指标数据合并完成，最终 Shape: {final_df.shape}, 列数: {len(final_df.columns)}")
            
            # 后续特征工程 (这些通常在合并后的 final_df 上操作，如果也需要并行化，需要更复杂的处理)
            logger.info(f"[{stock_code}] 开始补充外部特征 (指数、板块、筹码、资金流向)...")
            final_df = await self.enrich_features(df=final_df, stock_code=stock_code, main_indices=main_index_codes, external_data_history_days=external_data_history_days)
            logger.info(f"[{stock_code}] 外部特征补充完成。最终 DataFrame Shape: {final_df.shape}, 列数: {len(final_df.columns)}")
            
            actual_rsi_period = bs_params.get('rsi_period', default_rsi_p['period'])
            actual_macd_fast = bs_params.get('macd_fast', default_macd_p['period_fast'])
            actual_macd_slow = bs_params.get('macd_slow', default_macd_p['period_slow'])
            actual_macd_signal = bs_params.get('macd_signal', default_macd_p['signal_period'])

            apply_on_tfs = fe_params.get('apply_on_timeframes', bs_timeframes) # 确保apply_on_tfs是列表
            if isinstance(apply_on_tfs, str): apply_on_tfs = [apply_on_tfs]

            rs_config = fe_params.get('relative_strength', {})
            if rs_config.get('enabled', False):
                 ths_indexs_for_rs_objects = await self.industry_dao.get_stock_ths_indices(stock_code)
                 if ths_indexs_for_rs_objects is None:
                      logger.warning(f"[{stock_code}] 无法获取股票 {stock_code} 的同花顺板块信息。相对强度计算将跳过。")
                      ths_codes_for_rs = []
                 else:
                    ths_codes_for_rs = [m.ths_index.ts_code for m in ths_indexs_for_rs_objects if m.ths_index]
                 all_benchmark_codes_for_rs = list(set(main_index_codes + ths_codes_for_rs))
                 periods_rs = rs_config.get('periods', [5, 10, 20]) # 重命名变量避免冲突
                 if all_benchmark_codes_for_rs and periods_rs:
                      for tf_apply in apply_on_tfs: # 确保 apply_on_tfs 是列表
                           stock_close_col = f'close_{tf_apply}'
                           if stock_close_col in final_df.columns:
                                final_df = self.calculate_relative_strength(df=final_df, stock_close_col=stock_close_col, benchmark_codes=all_benchmark_codes_for_rs, periods=periods_rs, time_level=tf_apply)
                           else:
                                logger.warning(f"[{stock_code}] 计算相对强度 for TF {tf_apply} 失败，未找到股票收盘价列: {stock_close_col}")
                      logger.info(f"[{stock_code}] 相对强度/超额收益特征计算完成。")
                 else:
                      logger.warning(f"[{stock_code}] 相对强度/超额收益特征未启用或配置不完整 (基准代码或周期列表为空)。")
            
            lag_config = fe_params.get('lagged_features', {})
            if lag_config.get('enabled', False):
                 columns_to_lag_from_json = lag_config.get('columns_to_lag', [])
                 lags = lag_config.get('lags', [1, 2, 3])
                 if columns_to_lag_from_json and lags:
                      logger.info(f"[{stock_code}] 开始添加滞后特征...")
                      for tf_apply in apply_on_tfs:
                           actual_cols_for_lagging = []
                           for col_template_from_json in columns_to_lag_from_json:
                                # 构造列名，考虑指标参数
                                effective_base_name = self._build_indicator_base_name_for_lookup(
                                    base_name=col_template_from_json.split('_')[0] if '_' in col_template_from_json else col_template_from_json, # 粗略提取基础名
                                    params=bs_params, # 假设大部分滞后特征基于base_scoring的参数, 此处应更精确
                                    stock_code=stock_code
                                ) or col_template_from_json # 如果无法构建，使用原始模板

                                if col_template_from_json.startswith("RSI"): # 特殊处理RSI等
                                    effective_base_name = f"RSI_{actual_rsi_period}"
                                elif col_template_from_json.startswith("MACD_") and not col_template_from_json.startswith("MACDh_") and not col_template_from_json.startswith("MACDs_"):
                                    effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                                elif col_template_from_json.startswith("MACDh_"):
                                    effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                                elif col_template_from_json.startswith("MACDs_"):
                                    effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                                # ... 其他指标的参数化列名构造 ...

                                col_with_suffix = f"{effective_base_name}_{tf_apply}"
                                if col_with_suffix in final_df.columns:
                                     actual_cols_for_lagging.append(col_with_suffix)
                                elif effective_base_name in final_df.columns: # 有些特征可能不带时间后缀（如外部特征）
                                     actual_cols_for_lagging.append(effective_base_name)
                                else:
                                     logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到列 {col_with_suffix} 或 {effective_base_name} 进行滞后计算。")
                           if actual_cols_for_lagging:
                                final_df = self.add_lagged_features(final_df, actual_cols_for_lagging, lags)
                      logger.info(f"[{stock_code}] 滞后特征添加完成。")
                 else:
                      logger.warning(f"[{stock_code}] 滞后特征未启用或配置不完整。")

            roll_config = fe_params.get('rolling_features', {})
            if roll_config.get('enabled', False):
                 columns_to_roll_from_json = roll_config.get('columns_to_roll', [])
                 windows = roll_config.get('windows', [5, 10, 20])
                 stats = roll_config.get('stats', ["mean", "std"])
                 if columns_to_roll_from_json and windows and stats:
                      logger.info(f"[{stock_code}] 开始添加滚动统计特征...")
                      for tf_apply in apply_on_tfs:
                           actual_cols_for_rolling = []
                           for col_template_from_json in columns_to_roll_from_json:
                                effective_base_name = self._build_indicator_base_name_for_lookup(
                                    base_name=col_template_from_json.split('_')[0] if '_' in col_template_from_json else col_template_from_json,
                                    params=bs_params, # 同样假设基于base_scoring参数
                                    stock_code=stock_code
                                ) or col_template_from_json

                                if col_template_from_json.startswith("RSI"):
                                    effective_base_name = f"RSI_{actual_rsi_period}"
                                elif col_template_from_json.startswith("MACD_") and not col_template_from_json.startswith("MACDh_") and not col_template_from_json.startswith("MACDs_"):
                                     effective_base_name = f"MACD_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                                elif col_template_from_json.startswith("MACDh_"):
                                     effective_base_name = f"MACDh_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"
                                elif col_template_from_json.startswith("MACDs_"):
                                     effective_base_name = f"MACDs_{actual_macd_fast}_{actual_macd_slow}_{actual_macd_signal}"

                                col_with_suffix = f"{effective_base_name}_{tf_apply}"
                                if col_with_suffix in final_df.columns:
                                     actual_cols_for_rolling.append(col_with_suffix)
                                elif effective_base_name in final_df.columns:
                                     actual_cols_for_rolling.append(effective_base_name)
                                else:
                                     logger.warning(f"[{stock_code}] TF {tf_apply}: 未找到列 {col_with_suffix} 或 {effective_base_name} 进行滚动统计。")
                           if actual_cols_for_rolling:
                                final_df = self.add_rolling_features(final_df, actual_cols_for_rolling, windows, stats)
                      logger.info(f"[{stock_code}] 滚动统计特征添加完成。")
                 else:
                      logger.warning(f"[{stock_code}] 滚动统计特征未启用或配置不完整。")
            
            original_nan_count = final_df.isnull().sum().sum()
            final_df.ffill(inplace=True)
            final_df.bfill(inplace=True) # 再次填充以处理头部可能因ffill无法填充的NaN
            nan_count_after_fill = final_df.isnull().sum().sum()
            if nan_count_after_fill > 0:
                 logger.warning(f"[{stock_code}] 最终填充后仍存在 {nan_count_after_fill} 个缺失值 (原始 {original_nan_count})。缺失列详情 (部分): {final_df.isnull().sum()[final_df.isnull().sum() > 0].head().to_dict()}")
            else:
                 logger.info(f"[{stock_code}] 最终缺失值填充完成，无剩余 NaN。")
            
            # print(f"[{stock_code}] Debug: DataFrame preparation complete. Final shape: {final_df.shape}")
            return final_df, indicator_configs

        finally: # 修改行: 确保 executor 关闭
            if process_executor:
                # print(f"[{stock_code}] Debug: Shutting down ProcessPoolExecutor.")
                process_executor.shutdown(wait=True) # 等待所有任务完成
                # print(f"[{stock_code}] Debug: ProcessPoolExecutor shut down.")

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

    # --- 修改所有指标计算函数 async def calculate_* 为 def calculate_*(@staticmethod) ---
    # --- 并移除内部的 asyncio.to_thread ---
    @staticmethod # 修改行
    def calculate_atr(df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 ATR (平均真实波幅)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            # print(f"Debug (calculate_atr): Missing columns or empty df. Required: high, low, close. Available: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            # global ta # ta 应该在子进程中可用 (模块级导入)
            atr_series = df.ta.atr(length=period, high=df[high_col], low=df[low_col], close=df[close_col], append=False) # 修改行: 直接调用
            if atr_series is None or atr_series.empty:
                return None
            return pd.DataFrame({f'ATR_{period}': atr_series}) # pandas_ta 返回的列名已经是 ATR_period
        except Exception as e:
            # print(f"Error in calculate_atr (stock_code, tf_calc might be unavailable here, period {period}): {e}")
            return None

    @staticmethod # 修改行
    def calculate_boll_bands_and_width(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, close_col='close') -> Optional[pd.DataFrame]:
        """计算布林带 (BBANDS) 及其宽度 (BBW) 和百分比B (%B)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        try:
            bbands_df = df.ta.bbands(length=period, std=std_dev, close=df[close_col], append=False) # 修改行: 直接调用
            if bbands_df is None or bbands_df.empty:
                return None
            # pandas_ta 返回的列名通常是 BBL_period_std.dev, BBM_period_std.dev, BBU_period_std.dev, BBB_period_std.dev, BBP_period_std.dev
            # 我们这里直接使用它们，如果需要重命名，可以在这里处理或确保 _build_indicator_base_name_for_lookup 匹配
            # 例如，确保 std_dev 格式化为一位小数，如 2.0 -> _2.0
            std_dev_str = f"{std_dev:.1f}" # pandas-ta uses 1 decimal for std in column names
            
            # 构造期望的列名以匹配 pandas-ta 的输出
            lower_col_name_ta = f'BBL_{period}_{std_dev_str}'
            middle_col_name_ta = f'BBM_{period}_{std_dev_str}'
            upper_col_name_ta = f'BBU_{period}_{std_dev_str}'
            # bandwidth_col_name_ta = f'BBB_{period}_{std_dev_str}' # Bandwidth
            # percent_b_col_name_ta = f'BBP_{period}_{std_dev_str}' # Percent B

            result_df = pd.DataFrame(index=df.index)
            added_cols = False
            if lower_col_name_ta in bbands_df.columns:
                result_df[f'BBL_{period}_{std_dev:.1f}'] = bbands_df[lower_col_name_ta] # 使用我们期望的输出格式
                added_cols = True
            if middle_col_name_ta in bbands_df.columns:
                result_df[f'BBM_{period}_{std_dev:.1f}'] = bbands_df[middle_col_name_ta]
                added_cols = True
            if upper_col_name_ta in bbands_df.columns:
                result_df[f'BBU_{period}_{std_dev:.1f}'] = bbands_df[upper_col_name_ta]
                added_cols = True
            
            # 手动计算 BBW (Bandwidth) 和 BBP (Percent B) 以确保列名统一
            # pandas-ta 0.3.14b0 bbands() 会返回 BBB (bandwidth) 和 BBP (percent B)
            # 我们将依赖它，如果不存在则自己计算
            
            # BBW (布林带宽)
            bbw_col_out = f'BBW_{period}_{std_dev:.1f}'
            if f'BBB_{period}_{std_dev_str}' in bbands_df.columns: # pandas-ta name for bandwidth
                 result_df[bbw_col_out] = bbands_df[f'BBB_{period}_{std_dev_str}']
                 added_cols = True
            elif all(c in result_df.columns for c in [f'BBU_{period}_{std_dev:.1f}', f'BBL_{period}_{std_dev:.1f}', f'BBM_{period}_{std_dev:.1f}']):
                # Fallback: manual calculation if BBB not present
                result_df[bbw_col_out] = np.where(
                    np.abs(result_df[f'BBM_{period}_{std_dev:.1f}']) > 1e-9, # Avoid division by zero
                    (result_df[f'BBU_{period}_{std_dev:.1f}'] - result_df[f'BBL_{period}_{std_dev:.1f}']) / result_df[f'BBM_{period}_{std_dev:.1f}'],
                    np.nan
                )
                added_cols = True

            # BBP (%B)
            bbp_col_out = f'BBP_{period}_{std_dev:.1f}'
            if f'BBP_{period}_{std_dev_str}' in bbands_df.columns: # pandas-ta name for percent B
                 result_df[bbp_col_out] = bbands_df[f'BBP_{period}_{std_dev_str}']
                 added_cols = True
            elif all(c in result_df.columns for c in [f'BBU_{period}_{std_dev:.1f}', f'BBL_{period}_{std_dev:.1f}']) and close_col in df.columns:
                 # Fallback: manual calculation if BBP not present
                 denominator = result_df[f'BBU_{period}_{std_dev:.1f}'] - result_df[f'BBL_{period}_{std_dev:.1f}']
                 result_df[bbp_col_out] = np.where(
                     np.abs(denominator) > 1e-9, # Avoid division by zero
                     (df[close_col] - result_df[f'BBL_{period}_{std_dev:.1f}']) / denominator,
                     np.nan
                 )
                 added_cols = True
            
            return result_df if added_cols else None
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_historical_volatility(df: pd.DataFrame, period: int = 20, window_type: Optional[str] = None, close_col='close', annual_factor: int = 252) -> Optional[pd.DataFrame]:
        """计算历史波动率 (HV)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        try:
            log_returns = np.log(df[close_col] / df[close_col].shift(1))
            # window_type is not directly used in pandas rolling.std unless for specific win_types.
            # For simple rolling std, min_periods is more relevant.
            min_p = max(1, int(period * 0.8)) # 要求至少80%的窗口数据
            hv_series = log_returns.rolling(window=period, min_periods=min_p).std() * np.sqrt(annual_factor)
            return pd.DataFrame({f'HV_{period}': hv_series})
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_keltner_channels(df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算肯特纳通道 (KC)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            return None
        try:
            # pandas-ta kc() returns columns like KCL_ema_atr_multiplier, KCM_ema_atr_multiplier, KCU_ema_atr_multiplier
            kc_df = df.ta.kc(high=df[high_col], low=df[low_col], close=df[close_col], length=ema_period, atr_length=atr_period, scalar=atr_multiplier, mamode="ema", append=False) # 修改行
            if kc_df is None or kc_df.empty or kc_df.shape[1] < 3: # Expecting 3 columns
                return None
            
            # Construct names based on pandas-ta's typical output structure or inspect kc_df.columns
            # Assuming kc_df.columns are in order [lower, basis, upper]
            # Example pandas-ta column names: KCL_20_10_2.0, KCM_20_10_2.0, KCU_20_10_2.0
            # We need to match our _build_indicator_base_name_for_lookup which uses KCL_ema_atr format
            # So, we will rename them to our desired format.
            
            # A more robust way is to find columns by common prefixes if names are not exact
            lower_col_ta = kc_df.columns[0] # kc_df.filter(like='KCL').columns[0]
            basis_col_ta = kc_df.columns[1] # kc_df.filter(like='KCM').columns[0]
            upper_col_ta = kc_df.columns[2] # kc_df.filter(like='KCU').columns[0]

            result_df = pd.DataFrame({
                f'KCL_{ema_period}_{atr_period}': kc_df[lower_col_ta], # Our desired name
                f'KCM_{ema_period}_{atr_period}': kc_df[basis_col_ta], # Our desired name
                f'KCU_{ema_period}_{atr_period}': kc_df[upper_col_ta]  # Our desired name
            })
            return result_df
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_cci(df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 CCI (商品渠道指数)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            cci_series = df.ta.cci(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False) # 修改行
            if cci_series is None or cci_series.empty: return None
            return pd.DataFrame({f'CCI_{period}': cci_series}) # pandas_ta returns CCI_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_cmf(df: pd.DataFrame, period: int = 20, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 CMF (蔡金货币流量)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            cmf_series = df.ta.cmf(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False) # 修改行
            if cmf_series is None or cmf_series.empty:
                return None
            return pd.DataFrame({f'CMF_{period}': cmf_series}) # pandas_ta returns CMF_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_dmi(df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 DMI (动向指标), 包括 PDI (+DI), NDI (-DI), ADX"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # pandas_ta adx() returns ADX_period, DMP_period, DMN_period
            dmi_df = ta.adx(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False) # 修改行
            if dmi_df is None or dmi_df.empty:
                return None
            
            # Rename to PDI, NDI to match common usage, ADX is fine
            rename_map = {}
            # Ensure keys exist before trying to rename
            if f'DMP_{period}' in dmi_df.columns:
                rename_map[f'DMP_{period}'] = f'PDI_{period}'
            if f'DMN_{period}' in dmi_df.columns:
                rename_map[f'DMN_{period}'] = f'NDI_{period}'
            # ADX_period is usually the name, no need to rename if it matches
            # if f'ADX_{period}' in dmi_df.columns: # This is already the target name
            #     rename_map[f'ADX_{period}'] = f'ADX_{period}' 
            
            return dmi_df.rename(columns=rename_map) if rename_map else dmi_df # Return renamed or original
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_ichimoku(df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, chikou_period: Optional[int] = None, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算一目均衡表 (Ichimoku Cloud)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # pandas_ta ichimoku() returns two DataFrames: one with spans, one with signals (we usually use the first)
            # It returns ITS_tenkan, IKS_kijun, ISA_tenkan, ISB_kijun, ICS_kijun (chikou span shifted)
            # Senkou spans (ISA, ISB) are typically shifted forward for plotting, but as features, their current calculated values are used or shifted.
            # Here, we will take the unshifted values from pandas-ta for ISA, ISB, and shift them as required by convention.
            # The chikou span (ICS_kijun) from pandas-ta is close.shift(-kijun_period).
            
            # chikou_offset = chikou_period if chikou_period is not None else kijun_period # Default chikou offset
            # senkou_offset = kijun_period # Default senkou offset (for plotting)

            ichi_df_spans, ichi_df_signals = df.ta.ichimoku( # 修改行
                high=df[high_col], low=df[low_col], close=df[close_col],
                tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period,
                # chikou=chikou_offset, # pandas-ta uses this for the ICS_chikou_offset column
                # senkou_lead=senkou_offset, # pandas-ta uses this for shifting ISA/ISB if true
                include_chikou=True, # ensure chikou is calculated
                append=False
            )
            if ichi_df_spans is None or ichi_df_spans.empty:
                return None

            result_df = pd.DataFrame(index=df.index)
            # Tenkan-sen (Conversion Line)
            if f'ITS_{tenkan_period}' in ichi_df_spans.columns:
                result_df[f'TENKAN_{tenkan_period}'] = ichi_df_spans[f'ITS_{tenkan_period}']
            # Kijun-sen (Base Line)
            if f'IKS_{kijun_period}' in ichi_df_spans.columns:
                result_df[f'KIJUN_{kijun_period}'] = ichi_df_spans[f'IKS_{kijun_period}']
            
            # Chikou Span (Lagging Span) - current close plotted `kijun_period` periods in the past
            # As a feature, it's often `df[close_col].shift(kijun_period)` (past close)
            # pandas-ta `ICS_` is `close.shift(-offset)`, i.e., future price at current row.
            # We will use the `ICS_` column directly if present.
            chikou_col_name_ta = f'ICS_{kijun_period}' # Default based on pandas-ta output if chikou param not used
            if chikou_period is not None and f'ICS_{chikou_period}' in ichi_df_spans.columns: # If specific chikou period used
                chikou_col_name_ta = f'ICS_{chikou_period}'

            if chikou_col_name_ta in ichi_df_spans.columns:
                 result_df[f'CHIKOU_{kijun_period if chikou_period is None else chikou_period}'] = ichi_df_spans[chikou_col_name_ta]
            else: # Fallback if specific column not found, calculate standard chikou as past price
                 result_df[f'CHIKOU_{kijun_period}'] = df[close_col].shift(kijun_period)


            # Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2, plotted `kijun_period` (or `senkou_offset`) periods in the future
            # pandas_ta `ISA_` is the unshifted calculation.
            senkou_a_unshifted = None
            if f'ISA_{tenkan_period}' in ichi_df_spans.columns: # pandas-ta often names it with tenkan period
                senkou_a_unshifted = ichi_df_spans[f'ISA_{tenkan_period}']
            elif f'ITS_{tenkan_period}' in result_df.columns and f'KIJUN_{kijun_period}' in result_df.columns: # manual
                senkou_a_unshifted = (result_df[f'TENKAN_{tenkan_period}'] + result_df[f'KIJUN_{kijun_period}']) / 2
            
            if senkou_a_unshifted is not None:
                result_df[f'SENKOU_A_{tenkan_period}_{kijun_period}'] = senkou_a_unshifted.shift(kijun_period) # Shift forward

            # Senkou Span B (Leading Span B) - (senkou_period high + senkou_period low) / 2, plotted `kijun_period` periods in the future
            # pandas_ta `ISB_` is the unshifted calculation using `senkou_period`.
            senkou_b_unshifted = None
            if f'ISB_{senkou_period}' in ichi_df_spans.columns: # pandas-ta uses senkou_period for ISB name
                senkou_b_unshifted = ichi_df_spans[f'ISB_{senkou_period}']
            else: # manual
                rolling_high_senkou = df[high_col].rolling(window=senkou_period, min_periods=1).max()
                rolling_low_senkou = df[low_col].rolling(window=senkou_period, min_periods=1).min()
                senkou_b_unshifted = (rolling_high_senkou + rolling_low_senkou) / 2

            if senkou_b_unshifted is not None:
                result_df[f'SENKOU_B_{senkou_period}'] = senkou_b_unshifted.shift(kijun_period) # Shift forward
            
            return result_df if not result_df.empty else None
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_kdj(df: pd.DataFrame, period: int = 9, signal_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 KDJ 指标 (基于 Stochastic Oscillator)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # KDJ's K and D are stochastic's %K and %D. J = 3*%K - 2*%D
            # pandas-ta stoch() params: k=fast_k_period, d=slow_k_period (for %D), smooth_k=slow_d_period (for %K smoothing)
            # Typical KDJ: period for RSV (like fast_k in stoch), signal_period for D (like slow_k in stoch for %D), smooth_k_period for K smoothing (like smooth_k in stoch for %K)
            # So, map: KDJ period -> stoch k, KDJ signal_period -> stoch d, KDJ smooth_k_period -> stoch smooth_k
            stoch_df = df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], 
                                   k=period, d=signal_period, smooth_k=smooth_k_period, append=False) # 修改行
            if stoch_df is None or stoch_df.empty or stoch_df.shape[1] < 2: # Expect STOCHk and STOCHd
                return None
            
            # stoch_df columns are typically STOCHk_k_d_smooth and STOCHd_k_d_smooth
            # We need to find them. Example: STOCHk_9_3_3, STOCHd_9_3_3
            k_col_name_ta = stoch_df.columns[0] # Assume first is K
            d_col_name_ta = stoch_df.columns[1] # Assume second is D

            kdj_df = pd.DataFrame(index=df.index)
            k_series = stoch_df[k_col_name_ta]
            d_series = stoch_df[d_col_name_ta]
            
            # Our desired output names
            k_col_out = f'K_{period}_{signal_period}_{smooth_k_period}'
            d_col_out = f'D_{period}_{signal_period}_{smooth_k_period}'
            j_col_out = f'J_{period}_{signal_period}_{smooth_k_period}'

            kdj_df[k_col_out] = k_series
            kdj_df[d_col_out] = d_series
            kdj_df[j_col_out] = 3 * k_series - 2 * d_series
            return kdj_df
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_ema(df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 EMA (指数移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            ema_series = df.ta.ema(close=df[close_col], length=period, append=False) # 修改行
            if ema_series is None or ema_series.empty: return None
            return pd.DataFrame({f'EMA_{period}': ema_series}) # pandas_ta returns EMA_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_sma(df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 SMA (简单移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            sma_series = df.ta.sma(close=df[close_col], length=period, append=False) # 修改行
            if sma_series is None or sma_series.empty: return None
            return pd.DataFrame({f'SMA_{period}': sma_series}) # pandas_ta returns SMA_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_amount_ma(df: pd.DataFrame, period: int = 20, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的移动平均线 (AMT_MA)"""
        if df is None or df.empty or amount_col not in df.columns: return None
        try:
            min_p = max(1, int(period * 0.8))
            amt_ma_series = df[amount_col].rolling(window=period, min_periods=min_p).mean() # 修改行
            return pd.DataFrame({f'AMT_MA_{period}': amt_ma_series})
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_macd(df: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        """计算 MACD (异同移动平均线)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            # pandas_ta macd() returns MACD_fast_slow_signal, MACDh_f_s_s (histogram), MACDs_f_s_s (signal line)
            macd_df = df.ta.macd(close=df[close_col], fast=period_fast, slow=period_slow, signal=signal_period, append=False) # 修改行
            if macd_df is None or macd_df.empty: return None
            # Ensure column names are consistent with _build_indicator_base_name_for_lookup
            # e.g., MACD_12_26_9, MACDH_12_26_9, MACDS_12_26_9
            # pandas-ta might use lowercase 'h' and 's' for histogram and signal. We'll standardize.
            rename_map = {}
            expected_macd_col = f"MACD_{period_fast}_{period_slow}_{period_signal}"
            expected_macdh_col = f"MACDH_{period_fast}_{period_slow}_{period_signal}" # Our uppercase H
            expected_macds_col = f"MACDS_{period_fast}_{period_slow}_{period_signal}" # Our uppercase S

            # Find actual columns from pandas-ta output (they might vary slightly based on version or params)
            actual_macd_col = macd_df.filter(regex=f"MACD_{period_fast}_{period_slow}_{period_signal}").columns
            actual_macdh_col = macd_df.filter(regex=f"MACDH_{period_fast}_{period_slow}_{period_signal}|MACDh_{period_fast}_{period_slow}_{period_signal}").columns
            actual_macds_col = macd_df.filter(regex=f"MACDS_{period_fast}_{period_slow}_{period_signal}|MACDs_{period_fast}_{period_slow}_{period_signal}").columns
            
            if actual_macd_col.any(): rename_map[actual_macd_col[0]] = expected_macd_col
            if actual_macdh_col.any(): rename_map[actual_macdh_col[0]] = expected_macdh_col
            if actual_macds_col.any(): rename_map[actual_macds_col[0]] = expected_macds_col
            
            return macd_df.rename(columns=rename_map) if rename_map else macd_df
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_mfi(df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 MFI (资金流量指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]): return None
        try:
            mfi_series = df.ta.mfi(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False) # 修改行
            if mfi_series is None or mfi_series.empty: return None
            return pd.DataFrame({f'MFI_{period}': mfi_series}) # pandas_ta returns MFI_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_mom(df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 MOM (动量指标)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            mom_series = df.ta.mom(close=df[close_col], length=period, append=False) # 修改行
            if mom_series is None or mom_series.empty: return None
            return pd.DataFrame({f'MOM_{period}': mom_series}) # pandas_ta returns MOM_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_obv(df: pd.DataFrame, close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 OBV (能量潮指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [close_col, volume_col]): return None
        try:
            obv_series = df.ta.obv(close=df[close_col], volume=df[volume_col], append=False) # 修改行
            if obv_series is None or obv_series.empty: return None
            return pd.DataFrame({'OBV': obv_series}) # pandas_ta returns OBV
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_roc(df: pd.DataFrame, period: int = 12, close_col='close') -> Optional[pd.DataFrame]:
        """计算 ROC (价格变化率)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            roc_series = df.ta.roc(close=df[close_col], length=period, append=False) # 修改行
            if roc_series is None or roc_series.empty: return None
            return pd.DataFrame({f'ROC_{period}': roc_series}) # pandas_ta returns ROC_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_amount_roc(df: pd.DataFrame, period: int, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的 ROC (AROC)"""
        if df is None or df.empty or amount_col not in df.columns: return None
        try:
            # ROC on amount might produce inf if amount was 0 in the past.
            aroc_series = df.ta.roc(close=df[amount_col], length=period, append=False) # 修改行
            if aroc_series is None or aroc_series.empty: return None
            df_results = pd.DataFrame({f'AROC_{period}': aroc_series})
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinities
            return df_results
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_volume_roc(df: pd.DataFrame, period: int, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的 ROC (VROC)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            vroc_series = df.ta.roc(close=df[volume_col], length=period, append=False) # 修改行
            if vroc_series is None or vroc_series.empty: return None
            df_results = pd.DataFrame({f'VROC_{period}': vroc_series})
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinities
            return df_results
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_rsi(df: pd.DataFrame, period: int = 14, close_col='close') -> Optional[pd.DataFrame]:
        """计算 RSI (相对强弱指数)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            rsi_series = df.ta.rsi(close=df[close_col], length=period, append=False) # 修改行
            if rsi_series is None or rsi_series.empty: return None
            return pd.DataFrame({f'RSI_{period}': rsi_series}) # pandas_ta returns RSI_period
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_sar(df: pd.DataFrame, af_step: float = 0.02, max_af: float = 0.2, high_col='high', low_col='low') -> Optional[pd.DataFrame]:
        """计算 SAR (抛物线转向指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col]): return None
        try:
            # pandas_ta psar() returns PSARl (long), PSARs (short), PSARaf, PSARr (reversal)
            psar_df = df.ta.psar(high=df[high_col], low=df[low_col], af0=af_step, af=af_step, max_af=max_af, append=False) # 修改行
            if psar_df is None or psar_df.empty: return None
            
            # We are interested in the SAR value itself, which is either long or short SAR.
            # Combine PSARl and PSARs: PSARl is NaN when short, PSARs is NaN when long.
            # One of them will have the value.
            long_sar_col = next((col for col in psar_df.columns if col.startswith('PSARl')), None)
            short_sar_col = next((col for col in psar_df.columns if col.startswith('PSARs')), None)
            
            sar_values = None
            if long_sar_col and short_sar_col and long_sar_col in psar_df and short_sar_col in psar_df:
                sar_values = psar_df[long_sar_col].fillna(psar_df[short_sar_col])
            elif long_sar_col and long_sar_col in psar_df: # Only long found
                sar_values = psar_df[long_sar_col]
            elif short_sar_col and short_sar_col in psar_df: # Only short found
                sar_values = psar_df[short_sar_col]
            else: # No SAR columns found as expected
                # logger.warning(...) for subprocess
                return None

            if sar_values is not None:
                # Format af_step and max_af to match _build_indicator_base_name_for_lookup
                return pd.DataFrame({f'SAR_{af_step:.2f}_{max_af:.2f}': sar_values})
            return None
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_stoch(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 STOCH (随机指标)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # pandas_ta stoch() returns STOCHk_k_d_smooth and STOCHd_k_d_smooth
            stoch_df = df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=k_period, d=d_period, smooth_k=smooth_k_period, append=False) # 修改行
            if stoch_df is None or stoch_df.empty: return None
            # Column names from pandas_ta are like: STOCHk_14_3_3, STOCHd_14_3_3
            # These should be directly usable if _build_indicator_base_name_for_lookup matches this format.
            # For STOCH, _build_indicator_base_name_for_lookup returns STOCH_k_d_s or STOCHF_k_d if fast.
            # Here we use the direct output from pandas-ta, which has k,d,smooth in name.
            return stoch_df
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_adl(df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 ADL (累积/派发线) - pandas_ta calls it AD"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            adl_series = df.ta.ad(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], append=False) # 修改行
            if adl_series is None or adl_series.empty: return None
            return pd.DataFrame({'ADL': adl_series}) # pandas_ta returns ADOSC for AD Oscillator, AD for AD Line. We want AD Line. Name is 'AD'.
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_pivot_points(df: pd.DataFrame, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算经典枢轴点和斐波那契枢轴点 (基于前一周期数据). Typically for Daily TF."""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # Pivot points are calculated based on the previous period's HLC.
            # So, the values for today are based on yesterday's data.
            results = pd.DataFrame(index=df.index)
            prev_high = df[high_col].shift(1)
            prev_low = df[low_col].shift(1)
            prev_close = df[close_col].shift(1)
            
            # Check if enough data after shift
            if prev_high.iloc[1:].empty: # If after shifting, no data left (e.g., only 1 row in df)
                return pd.DataFrame(columns=['PP', 'S1', 'S2', 'S3', 'R1', 'R2', 'R3', 'F_S1', 'F_S2', 'F_S3', 'F_R1', 'F_R2', 'F_R3'])


            PP = (prev_high + prev_low + prev_close) / 3
            results['PP'] = PP
            # Classic Pivots
            results['S1'] = (2 * PP) - prev_high
            results['R1'] = (2 * PP) - prev_low
            results['S2'] = PP - (prev_high - prev_low)
            results['R2'] = PP + (prev_high - prev_low)
            results['S3'] = results['S1'] - (prev_high - prev_low) # PP - 2 * (prev_high - prev_low)
            results['R3'] = results['R1'] + (prev_high - prev_low) # PP + 2 * (prev_high - prev_low)
            # results['S4'] = PP - 2 * (prev_high - prev_low) - (prev_high - prev_low) # S3 - (H-L)
            # results['R4'] = PP + 2 * (prev_high - prev_low) + (prev_high - prev_low) # R3 + (H-L)
            
            # Fibonacci Pivots
            diff = prev_high - prev_low
            results['F_S1'] = PP - (0.382 * diff)
            results['F_S2'] = PP - (0.618 * diff)
            results['F_S3'] = PP - (1.000 * diff)
            results['F_R1'] = PP + (0.382 * diff)
            results['F_R2'] = PP + (0.618 * diff)
            results['F_R3'] = PP + (1.000 * diff)
            
            return results.iloc[1:] # Remove first row NaN due to shift
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_vol_ma(df: pd.DataFrame, period: int = 20, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的移动平均线 (VOL_MA)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            min_p = max(1, int(period * 0.8))
            vol_ma_series = df[volume_col].rolling(window=period, min_periods=min_p).mean() # 修改行
            return pd.DataFrame({f'VOL_MA_{period}': vol_ma_series})
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_vwap(df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume', anchor: Optional[str] = None) -> Optional[pd.DataFrame]:
        """计算 VWAP (成交量加权平均价)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            # pandas_ta vwap() returns VWAP if no anchor, or VWAP_D, VWAP_W if anchor is D, W etc.
            vwap_series = df.ta.vwap(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], anchor=anchor, append=False) # 修改行
            if vwap_series is None or vwap_series.empty: return None
            
            # Name construction should match _build_indicator_base_name_for_lookup
            col_name = 'VWAP'
            if anchor:
                # anchor can be 'D', 'W', 'M', 'Y', 'Q', 'S'(ession), 'H', 'MIN'
                # ta library typically appends it like VWAP_D.
                col_name = f'VWAP_{anchor.upper()}' # Match common output
            
            # If vwap_series.name matches col_name, great. Otherwise, create df with col_name.
            if vwap_series.name == col_name:
                 return vwap_series.to_frame()
            return pd.DataFrame({col_name: vwap_series})
        except Exception as e:
            return None

    @staticmethod # 修改行
    def calculate_willr(df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算威廉姆斯 %R (WILLR)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            willr_series = df.ta.willr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False) # 修改行
            if willr_series is None or willr_series.empty: return None
            return pd.DataFrame({f'WILLR_{period}': willr_series}) # pandas_ta returns WILLR_period
        except Exception as e:
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
