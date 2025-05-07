# services\indicator_services.py
import asyncio
from collections import defaultdict
import datetime
from functools import reduce
import json
import os
import warnings
import logging
from dao_manager.tushare_daos.indicator_dao import IndicatorDAO
import numpy as np
import pandas as pd
import math
from django.utils import timezone
from typing import Any, List, Optional, Union, Dict
from django.db import models # 确保导入 models
import pandas_ta as ta

# --- 添加以下代码来忽略 FutureWarning ---
# 仅忽略 FutureWarning
# warnings.simplefilter(action='ignore', category=FutureWarning)
# ---------------------------------------
# --- 添加以下代码来忽略特定的 UserWarning ---
# 仅忽略关于 "drop timezone information" 的 UserWarning
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*drop timezone information.*')
# -------------------------------------------

# 解决 pandas_ta 可能出现的 SettingWithCopyWarning，虽然通常不影响结果
pd.options.mode.chained_assignment = None # default='warn'

from dao_manager.tushare_daos.stock_basic_info_dao import StockBasicInfoDao
from core.constants import TimeLevel # 确保 TimeLevel 枚举和可能需要的其他常量导入

logger = logging.getLogger("services")

class IndicatorService:
    """
    技术指标计算服务 (使用 pandas-ta)
    负责获取多时间级别原始数据，进行时间序列标准化（重采样），计算指标，合并数据并进行最终填充。
    """
    def __init__(self):
        self.indicator_dao = IndicatorDAO()
        self.stock_basic_dao = StockBasicInfoDao() # 用于获取 StockInfo 对象
        # 动态导入 pandas_ta，避免在类实例化时就强制依赖
        try:
            global ta
            import pandas_ta as ta
            if ta is None: # 如果之前导入失败过
                 logger.warning("pandas_ta 之前导入失败，尝试重新导入。")
                 import pandas_ta as ta
            # 添加 ta.CommonStrategy 以便后面使用
            # if not hasattr(ta, 'CommonStrategy'):
            #     # 根据 pandas_ta 版本，可能需要导入或它已在 ta 命名空间下
            #     pass # 通常 ta.Strategy 即可
        except ImportError:
            logger.error("pandas-ta 库未安装，请运行 'pip install pandas-ta'")
            ta = None # 设置为 None，后续计算会失败但不会在导入时崩溃

    # --- 辅助函数：将时间级别字符串转换为近似分钟数 ---
    def _get_timeframe_in_minutes(self, tf_str: str) -> Optional[int]:
        """
        将时间级别字符串（如 '5', '15', 'D', 'W', 'M'）转换为近似的分钟数。
        注意：'D', 'W', 'M' 是基于标准交易时间的估算。
        """
        tf_str = str(tf_str).upper() # 转换为大写以便处理 'd', 'w', 'm'
        if tf_str.isdigit():
            return int(tf_str)
        elif tf_str == 'D':
            return 240 # A股主要交易时间 4 小时 * 60 分钟/小时
        elif tf_str == 'W':
            return 240 * 5 # 每周 5 个交易日
        elif tf_str == 'M':
            # 月度交易日数不固定，使用一个近似值，例如 21 天 * 4小时/天 * 60分钟/小时
            return 240 * 21
        else:
            logger.warning(f"无法将时间级别 '{tf_str}' 转换为分钟数，将返回 None。")
            return None

    # --- 辅助函数：将时间级别字符串转换为 pandas resample 频率字符串 ---
    def _get_resample_freq_str(self, tf_str: str) -> Optional[str]:
        """
        将时间级别字符串转换为 pandas resample 函数使用的频率字符串。
        例如: '5' -> '5T', '15' -> '15T', '60' -> '60T', 'D' -> 'D', 'W' -> 'W', 'M' -> 'M'.
        """
        tf_str = str(tf_str).upper()
        if tf_str.isdigit():
            return f'{tf_str}T' # T for minutes
        elif tf_str in ['D', 'W', 'M']:
            return tf_str
        else:
            logger.warning(f"不支持的时间级别 '{tf_str}' 进行重采样。")
            return None

    # --- 新增辅助函数：计算特定时间级别所需的 K 线数量 ---
    def _calculate_needed_bars_for_tf(
        self,
        target_tf: str,
        min_tf: str,
        base_needed_bars: int,
        global_max_lookback: int
    ) -> int:
        """
        计算目标时间级别需要从 DAO 获取的原始 K 线数量。
        获取的数量应该足够覆盖基础时间级别所需的最早时间点，以及所有指标的最大回看期。
        注意：这里计算的是应从数据源请求的“大致”数量，不是精确的数量或重采样后的数量。

        Args:
            target_tf (str): 目标时间级别 (e.g., '15', 'D').
            min_tf (str): 基础（最小）时间级别 (e.g., '5').
            base_needed_bars (int): 基础时间级别要求的 K 线数量（用于训练窗口等）.
            global_max_lookback (int): 所有指标计算所需的最大回看期.

        Returns:
            int: 最终计算出的目标时间级别应获取的 K 线数量 (作为 DAO 的 limit 参数).
        """
        # 首先确定需要覆盖的总时间跨度，基于最小时间级别的基础数量和全局最大回看期
        # 假设我们需要获取足够的数据来覆盖 min_tf 的 base_needed_bars + global_max_lookback 时长
        # 这是一个估算，因为交易日非连续
        total_duration_bars_at_min_tf = base_needed_bars + global_max_lookback

        min_tf_minutes = self._get_timeframe_in_minutes(min_tf)
        target_tf_minutes = self._get_timeframe_in_minutes(target_tf)

        if min_tf_minutes is None or target_tf_minutes is None or min_tf_minutes == 0:
            logger.warning(f"无法计算 {target_tf} 的时间比例 (min={min_tf_minutes}, target={target_tf_minutes})，将仅使用指标回看期 {global_max_lookback} 作为获取数量。")
            return global_max_lookback # 出错时，至少保证指标计算

        # 计算目标时间级别需要多少 bar 来覆盖相同的时间跨度
        # target_bars = total_duration_bars_at_min_tf * (min_tf_minutes / target_tf_minutes)
        # 更直观地，计算总时长（以分钟为单位，估算）
        estimated_total_minutes = total_duration_bars_at_min_tf * min_tf_minutes
        # 在目标时间级别下，覆盖这些分钟数需要的 bar 数量
        estimated_bars_needed = math.ceil(estimated_total_minutes / target_tf_minutes) if target_tf_minutes > 0 else total_duration_bars_at_min_tf

        # 确保获取的数量至少大于指标的最大回看期，并增加一些缓冲
        needed = max(estimated_bars_needed, global_max_lookback) + 100 # 增加 100 bar 缓冲获取更多原始数据用于重采样

        # 针对日、周、月线做特殊调整，可能不需要分钟线那么大的比例乘数
        # 对于 >= Day 的时间级别，可能只需要覆盖指标回看期 + 少量缓冲的 bar 数量
        if target_tf_minutes >= 240: # 如果是日线或更长
             needed = max(estimated_bars_needed, global_max_lookback * (target_tf_minutes / 240) + 100) # 日线回看期按天算
             # 简单一点，对于日线及以上，至少获取 回看期 + 一些额外天数（例如 365天 + 回看期）
             if target_tf_minutes >= 240: # 日线或以上
                  needed = max(needed, global_max_lookback + 365*2) # 确保获取足够年份数据

        # 向上取整，确保获取足够数量
        return math.ceil(needed)


    async def _get_ohlcv_data(self, stock_code: str, time_level: Union[TimeLevel, str], needed_bars: int) -> Optional[pd.DataFrame]:
        """
        获取足够用于计算的原始历史数据 DataFrame。
        此函数仅负责从 DAO 获取，不进行时间序列对齐或质量过滤。
        """
        limit = needed_bars # 直接使用计算好的 needed_bars 作为 limit
        # logger.debug(f"为计算指标 {stock_code} {time_level}，尝试获取 {limit} 条原始历史数据 (动态计算值)")
        df = await self.indicator_dao.get_history_ohlcv_df(stock_code, time_level, limit=limit)
        if df is None or df.empty:
            logger.warning(f"[{stock_code}] 时间级别 {time_level} 无法获取足够的原始历史数据 (请求 {limit} 条)。")
            return None

        # DAO 已经处理了列名小写、转换为数值、设为时区感知的 DatetimeIndex 并排序
        # 在这里不需要再次处理

        # --- 检查并标准化列名 ---
        # 您的日志显示 DAO 返回的成交量列名是 'vol'，但后续代码期望 'volume'
        # 在此处统一列名以确保后续处理的正确性
        if 'vol' in df.columns and 'volume' not in df.columns:
            df.rename(columns={'vol': 'volume'}, inplace=True)
            logger.debug(f"[{stock_code}] 时间级别 {time_level}: 将原始数据列名 'vol' 重命名为 'volume'.")
        # 可以根据需要添加其他列名的检查和标准化
        # 例如，如果 amount 列名也可能不一致，可以添加类似逻辑
        if 'amount_col_from_dao' in df.columns and 'amount' not in df.columns:
            df.rename(columns={'amount_col_from_dao': 'amount'}, inplace=True)
            logger.debug(...)
        # --------------------------------------

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tzinfo is not None:
             logger.info(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，时间范围: {df.index.min()} 至 {df.index.max()}")
        else:
             logger.warning(f"[{stock_code}] 时间级别 {time_level} 获取到 {len(df)} 条原始K线数据，但索引不是时区感知的 DatetimeIndex。")
             # 如果索引格式不正确，后续 resample 会出错，需要检查 DAO 确保其返回正确格式的索引。

        return df

    def _resample_and_clean_dataframe(self, df: pd.DataFrame, tf: str, min_periods: int = 1, fill_method: str = 'ffill') -> Optional[pd.DataFrame]:
        """
        对原始 DataFrame 进行重采样到标准的 K 线时间点，并进行初步填充。
        Args:
            df (pd.DataFrame): 从 DAO 获取的原始 DataFrame (index 是 DatetimeIndex, tz-aware)。
            tf (str): 目标时间级别字符串 (e.g., '5', '15', 'D').
            min_periods (int): 重采样聚合所需的最小原始数据点数量，小于此数量将产生 NaN。
            fill_method (str): 重采样后填充 NaN 的方法 ('ffill', 'bfill', None)。
        Returns:
            Optional[pd.DataFrame]: 重采样并初步填充后的 DataFrame，如果重采样后数据量过少或全为 NaN 则返回 None。
        """
        if df is None or df.empty:
            return None
        freq = self._get_resample_freq_str(tf)
        if freq is None:
            logger.error(f"[{df.index.name}] 时间级别 {tf} 无法转换为有效的重采样频率。")
            return None
        # 定义重采样聚合规则：OHLCV + amount + turnover_rate
        # 假设原始 DataFrame 包含 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate' (小写)
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum',
            # 如果有换手率等其他列需要聚合，根据其性质添加
            # 'turnover_rate': 'last' # 或者其他合适的聚合方式
        }
        # 过滤掉 DataFrame 中不存在的列
        agg_rules = {col: rule for col, rule in agg_rules.items() if col in df.columns}
        if not agg_rules:
            logger.warning(f"[{df.index.name}] 时间级别 {tf} 没有找到可以聚合的列，无法进行重采样。")
            return None
        try:
            # 进行重采样。 label='right', closed='right' 表示时间戳代表 K 线结束时间点
            # origin='start_day' 或 'end_day' 根据需要调整，或者使用默认行为
            resampled_df = df.resample(freq, label='right', closed='right').agg(agg_rules, min_periods=min_periods)

            # 重采样后可能会引入 NaN，特别是对于在某个时间周期内没有原始数据点的情况
            # 检查重采样后的数据量和质量
            if resampled_df.empty:
                logger.warning(f"[{df.index.name}] 时间级别 {tf} 重采样后 DataFrame 为空。")
                return None

            # 检查必要列（open, high, low, close, volume）是否在重采样后全为 NaN
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            # 确保必要列在 agg_rules 中被聚合了
            required_agg_cols = [col for col in required_cols if col in agg_rules]
            if resampled_df[required_agg_cols].isnull().all().all():
                logger.warning(f"[{df.index.name}] 时间级别 {tf} 重采样后必要列全部为 NaN，数据无效。")
                return None
            # 初步填充重采样引入的 NaN
            if fill_method == 'ffill':
                resampled_df.ffill(inplace=True)
            elif fill_method == 'bfill':
                resampled_df.bfill(inplace=True)
            # 记录重采样后的数据量和缺失情况 (初步填充后)
            missing_after_resample_fill = resampled_df.isnull().sum().sum()
            if missing_after_resample_fill > 0:
                 logger.warning(f"[{df.index.name}] 时间级别 {tf} 重采样并初步填充后仍存在 {missing_after_resample_fill} 个缺失值。")
                 # 打印缺失比例较高的列
                 missing_cols_detail = resampled_df.isnull().mean()
                 missing_cols_detail = missing_cols_detail[missing_cols_detail > 0].sort_values(ascending=False).head()
                 if not missing_cols_detail.empty:
                     logger.warning(f"[{df.index.name}] 时间级别 {tf} 重采样后缺失比例较高的列 (初步填充后): {missing_cols_detail.to_dict()}")
            # logger.info(f"[{df.index.name}] 时间级别 {tf} 重采样完成，数据量: {len(resampled_df)} 条。")
            return resampled_df
        except Exception as e:
            logger.error(f"[{df.index.name}] 时间级别 {tf} 重采样和清理数据时出错: {e}", exc_info=True)
            return None

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, base_needed_bars: int = None) -> Optional[pd.DataFrame]:
        """
        根据策略 JSON 配置文件准备包含重采样基础数据和所有计算指标的 DataFrame。
        处理数据源时间戳不规范的问题。
        Args:
            stock_code (str): 股票代码，用于标识具体股票。
            params_file (str): 策略参数 JSON 文件的路径，包含时间框架和指标参数。
            base_needed_bars (int, optional): 基础时间级别（最小级别）需要覆盖的大致 K 线数量（用于训练窗口等），影响 DAO 获取数据量。如果为 None，则从参数文件或使用默认值确定。
        Returns:
            Optional[pd.DataFrame]: 包含所有所需数据的 DataFrame，列名包含时间级别后缀（如 'RSI_12_15', 'close_60'）。
                                    数据已按最小时间级别索引对齐并最终填充。
                                    如果数据准备失败（如关键时间级别数据不可用），则返回 None。
        """
        if ta is None:
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None
        # 1. 加载 JSON 参数文件
        try:
            if not os.path.exists(params_file):
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
        except Exception as e:
            logger.error(f"[{stock_code}] 加载或解析参数文件 {params_file} 失败: {e}", exc_info=True)
            return None

        # 2. 识别需求：时间级别和全局指标最大回看期
        # 这部分逻辑保持不变
        all_time_levels = set()
        global_max_lookback = 0
        min_time_level = None
        min_tf_minutes = float('inf')

        # ... (识别时间级别和计算 global_max_lookback 的代码，与之前相同)
        try:
            bs_params = params['base_scoring']  # 基础评分参数
            vc_params = params['volume_confirmation']  # 成交量确认参数
            dd_params = params['divergence_detection']  # 背离检测参数
            kpd_params = params['kline_pattern_detection']  # K线形态检测参数
            ta_params = params['trend_analysis']  # 趋势分析参数
            ia_params = params['indicator_analysis_params']  # 指标分析参数
            tr_params = params.get('trend_reversal_params', {})  # 趋势反转参数
            t0_params = params.get('t_plus_0_signals', {})
            turnover_filter = tr_params.get('turnover_filter', {})
            # 将所有需要的时间级别添加到集合中
            all_time_levels.update(bs_params['timeframes'])
            if vc_params.get('enabled', False): all_time_levels.add(vc_params.get('tf'))
            if dd_params.get('enabled', False): all_time_levels.add(dd_params.get('tf'))
            if kpd_params.get('enabled', False): all_time_levels.add(kpd_params.get('tf'))

            # 确定分析所需的时间框架 (考虑默认值和优先级)
            analysis_tf_candidates = [
                 dd_params.get('tf') if dd_params.get('enabled', False) else None,
                 kpd_params.get('tf') if kpd_params.get('enabled', False) else None,
                 vc_params.get('tf') if vc_params.get('enabled', False) else None
            ]
            # 找到第一个非 None 的时间框架，如果没有，则使用 bs_params['timeframes'][0] 作为默认
            analysis_tf = next((tf for tf in analysis_tf_candidates if tf is not None), bs_params['timeframes'][0])
            all_time_levels.add(analysis_tf)

            # 为 T+0 策略添加 focus_timeframe（如果存在）
            t0_params = params.get('t_plus_0_signals', {})
            if t0_params.get('enabled', False):
                focus_timeframe = t0_params.get('focus_timeframe', '5')
                all_time_levels.add(focus_timeframe)
                logger.info(f"[{stock_code}] 为 T+0 策略添加时间级别: {focus_timeframe}")
            # 为换手率过滤添加时间框架（如果启用）
            turnover_filter = tr_params.get('turnover_filter', {})
            if turnover_filter.get('enabled', False):
                turnover_tf = turnover_filter.get('timeframe', analysis_tf) # 默认使用分析时间框架
                all_time_levels.add(turnover_tf)
                logger.info(f"[{stock_code}] 为换手率过滤添加时间级别: {turnover_tf}")

            # 移除 None 值，转换为字符串集合
            all_time_levels = {str(tf) for tf in all_time_levels if tf is not None}

            # --- 找到实际需要并可转换为分钟数的最小时间级别 ---
            valid_time_levels_with_minutes = {}
            for tf in all_time_levels:
                minutes = self._get_timeframe_in_minutes(tf)
                if minutes is not None:
                     valid_time_levels_with_minutes[tf] = minutes

            if not valid_time_levels_with_minutes:
                 logger.error(f"[{stock_code}] 未能从参数文件中确定任何可用的时间级别。请检查时间级别定义: {all_time_levels}")
                 return None

            # 找到分钟数最小的时间级别
            min_time_level = min(valid_time_levels_with_minutes, key=valid_time_levels_with_minutes.get)
            min_tf_minutes = valid_time_levels_with_minutes[min_time_level]

            if min_time_level is None:
                logger.error(f"[{stock_code}] 无法确定有效的最小时间级别，请检查参数文件中的时间级别定义: {all_time_levels}")
                return None
            logger.info(f"[{stock_code}] 识别出的最小时间级别为: {min_time_level} ({min_tf_minutes} 分钟)")


            # 计算最大回看期，考虑所有指标的参数
            # 确保 lookbacks 列表非空且包含有效数值
            lookbacks = [
                bs_params.get('rsi_period', 0),  bs_params.get('kdj_period_k', 0),
                bs_params.get('boll_period', 0),  bs_params.get('macd_slow', 0) + bs_params.get('macd_signal', 0),
                bs_params.get('cci_period', 0),  bs_params.get('mfi_period', 0),
                bs_params.get('roc_period', 0),  bs_params.get('dmi_period', 0) * 3,
                vc_params.get('amount_ma_period', 0), vc_params.get('cmf_period', 0), vc_params.get('obv_ma_period', 0),
                ia_params.get('stoch_k', 0) + ia_params.get('stoch_d', 0) + ia_params.get('stoch_smooth_k', 0),
                ia_params.get('volume_ma_period', 0),
                dd_params.get('lookback', 0) * 2,
                tr_params.get('turnover_filter', {}).get('timeframe_lookback', 0), # 换手率过滤回看期
                55  # SAR 默认回看期或固定值
            ]
            valid_lookbacks = [lb for lb in lookbacks if isinstance(lb, (int, float)) and lb > 0]

            if not valid_lookbacks:
                logger.warning(f"[{stock_code}] 未能从参数计算出有效的指标回看期，将使用默认值 100。")
                global_max_lookback = 100
            else:
                # 增加缓冲，确保重采样后数据量足够
                global_max_lookback = max(valid_lookbacks) + 100

            logger.info(f"[{stock_code}] 需要的时间级别: {all_time_levels}, 全局指标最大回看期 (含缓冲): {global_max_lookback}")

        except KeyError as e:
            logger.error(f"[{stock_code}] 参数文件 {params_file} 缺少键: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"[{stock_code}] 解析参数文件或确定时间级别时出错: {e}", exc_info=True)
            return None


        # 3. 并行获取原始 OHLCV 数据（使用动态计算的 needed_bars）
        # 并行任务字典：{tf: asyncio_task}
        ohlcv_tasks = {}
        for tf in all_time_levels:
            # 为当前时间级别计算需要从 DAO 获取的原始 K 线数量
            # 这个数量是为了覆盖足够长的时间范围，以便重采样和指标计算
            needed_bars_for_tf = self._calculate_needed_bars_for_tf(
                target_tf=tf,
                min_tf=min_time_level,
                base_needed_bars=base_needed_bars if base_needed_bars is not None else 10000, # 如果 base_needed_bars 为 None，使用默认值 10000
                global_max_lookback=global_max_lookback
            )
            logger.info(f"[{stock_code}] 时间级别 {tf}: 基础({min_time_level})需{base_needed_bars if base_needed_bars is not None else 10000}条, 指标需{global_max_lookback}条 -> 动态计算需获取 {needed_bars_for_tf} 条原始数据.")
            # 调用 DAO 获取原始数据
            ohlcv_tasks[tf] = self._get_ohlcv_data(stock_code, tf, needed_bars_for_tf) # 使用计算出的数量

        # 并行执行数据获取任务
        ohlcv_results = await asyncio.gather(*ohlcv_tasks.values())
        # 结果字典：{tf: DataFrame or None}
        raw_ohlcv_dfs = dict(zip(all_time_levels, ohlcv_results))

        # 4. 对获取到的原始数据进行重采样和初步清洗
        resampled_ohlcv_dfs = {}
        # 设定一个重采样后数据量阈值，如果小于此值则认为数据不可用
        # 例如，要求重采样后数据量至少是所需基础 bar 数量的 80%
        min_usable_bars = math.ceil((base_needed_bars if base_needed_bars is not None else 10000) * 0.8)
        for tf, raw_df in raw_ohlcv_dfs.items():
            if raw_df is None or raw_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf} 没有获取到原始数据，跳过重采样和计算。")
                continue
            # 对原始数据进行重采样和初步填充
            # min_periods=1 表示只要重采样周期内有 1 个原始数据点就进行聚合
            resampled_df = self._resample_and_clean_dataframe(raw_df, tf, min_periods=1, fill_method='ffill')
            if resampled_df is None or resampled_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf} 重采样后数据为空，跳过计算。")
                continue
            # 检查重采样后数据量是否足够（特别是最小时间级别）
            if tf == min_time_level and len(resampled_df) < min_usable_bars:
                 logger.error(f"[{stock_code}] 最小时间级别 {tf} 重采样后数据量 {len(resampled_df)} 条，少于最低可用阈值 {min_usable_bars} 条。无法继续。")
                 return None # 如果最小时间级别数据不可用，整个流程中断
            # 对于其他时间级别，如果数据量太少，也可能影响指标计算，可以设置一个相对阈值
            # 例如，重采样后数据量少于 global_max_lookback，可能指标计算会产生大量NaN
            if len(resampled_df) < global_max_lookback:
                 logger.warning(f"[{stock_code}] 时间级别 {tf} 重采样后数据量 {len(resampled_df)} 条，少于全局指标最大回看期 {global_max_lookback} 条。计算的指标可能包含大量 NaN。")
                 # 仍然保留数据，让后续填充处理
            # 重命名基础列，添加时间级别后缀
            rename_map = {}
            for col in resampled_df.columns:
                 # 只对 OHLCV 等基础列添加时间后缀
                 # 假设 DAO 返回的 DataFrame 列名已经是 'open', 'high', 'low', 'close', 'volume', 'amount' 等标准小写名
                 if col in ['open', 'high', 'low', 'close', 'volume', 'amount', 'turnover_rate']:
                      rename_map[col] = f"{col}_{tf}"
                 # 其他可能存在的列名（如 stock_code, time_level，虽然 DAO 返回的 DataFrame 索引是时间，这些列不应存在）
                 # 如果存在且需要保留，则不重命名或根据需要重命名
            # 如果没有需要重命名的列（例如 DataFrame 只有索引），则跳过 rename
            if rename_map:
                resampled_df_renamed = resampled_df.rename(columns=rename_map)
            else:
                resampled_df_renamed = resampled_df.copy()
            resampled_ohlcv_dfs[tf] = resampled_df_renamed
            # logger.info(f"[{stock_code}] 时间级别 {tf} 重采样并清洗完成，数据量: {len(resampled_df_renamed)} 条。")
        # 确保最小时间级别的数据可用
        if min_time_level not in resampled_ohlcv_dfs or resampled_ohlcv_dfs[min_time_level] is None or resampled_ohlcv_dfs[min_time_level].empty:
             logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 重采样后的数据不可用。终止数据准备。")
             return None
        # 确定最小时间级别重采样后的时间索引作为合并基准
        base_index = resampled_ohlcv_dfs[min_time_level].index
        logger.info(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的重采样索引作为合并基准，索引数量: {len(base_index)}。")
        # 5. 计算所有指标 - 使用并行任务，基于重采样后的 OHLCV 数据
        calculated_indicators = defaultdict(list)  # {tf: [indicator_df1, indicator_df2, ...]}
        # 辅助函数：安全计算指标并存储结果
        async def _calculate_and_store_async(tf, indicator_name, calculation_func, base_ohlcv_df, *args, **kwargs):
            """
            在指定的重采样 OHLCV 数据帧上计算指标。
            """
            if base_ohlcv_df is None or base_ohlcv_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf} 的基础 OHLCV 数据为空，无法计算 {indicator_name}")
                return None
            try:
                # 将重采样后的 OHLCV 数据（列名已添加时间后缀）
                # 暂时恢复标准列名以便 pandas-ta 使用
                # 移除时间后缀，例如 'open_5' -> 'open'
                df_for_ta = base_ohlcv_df.copy()
                rename_back_map = {
                     f'open_{tf}': 'open', f'high_{tf}': 'high', f'low_{tf}': 'low',
                     f'close_{tf}': 'close', f'volume_{tf}': 'volume', f'amount_{tf}': 'amount',
                     f'turnover_rate_{tf}': 'turnover_rate'
                }
                # 只重命名存在的列
                actual_rename_back_map = {k: v for k, v in rename_back_map.items() if k in df_for_ta.columns}

                if actual_rename_back_map:
                    df_for_ta.rename(columns=actual_rename_back_map, inplace=True)
                else:
                    # 如果没有标准列名，可能数据有问题，或者只包含一些非标准列
                    # 检查是否包含至少 OHLCV，如果没有，指标计算会失败
                    required_basic_cols = ['open', 'high', 'low', 'close', 'volume']
                    if not all(col in df_for_ta.columns for col in required_basic_cols):
                         logger.warning(f"[{stock_code}] 时间级别 {tf} 的基础 OHLCV 数据缺少标准列 ({required_basic_cols})，可能无法计算指标 {indicator_name}")
                         # 继续尝试计算，但可能会失败

                # 确保传递给 pandas_ta 的 DataFrame 包含必要的 OHLCV 列且是数值类型
                # 重采样后已经确保了索引和基本数值类型
                # pandas_ta 会在其内部处理 NaNs
                result = calculation_func(df_for_ta, *args, **kwargs) # 使用重采样并临时重命名列的 DF

                if result is not None and not result.empty:
                    # 统一为所有结果列添加时间后缀
                    result_renamed = result.rename(columns=lambda x: f"{x}_{tf}")
                    return (tf, result_renamed)
                else:
                    logger.warning(f"[{stock_code}] 计算 {indicator_name} for 时间级别 {tf} 结果为空或失败")
                    return None
            except Exception as e:
                logger.error(f"[{stock_code}] 计算指标 {indicator_name} (时间 {tf}) 时出错: {e}", exc_info=True)
                return None
            return None # Should not reach here
        # 创建并行任务列表 for indicator calculation
        indicator_tasks = []
        # 遍历所有已成功重采样并清洗的时间级别
        for tf, base_ohlcv_df in resampled_ohlcv_dfs.items():
            if base_ohlcv_df is None or base_ohlcv_df.empty:
                 continue # 跳过没有数据的级别
            # --- 添加所有指标计算任务 ---
            # 确保这里调用 calculate_* 函数时，传入的是临时的、列名不带后缀的 DataFrame
            # 辅助函数 _calculate_and_store_async 会处理列名恢复和结果列名添加后缀

            # 基础评分指标
            for indi_key in bs_params.get('score_indicators', []):
                if tf not in bs_params.get('timeframes', []): continue # 只在指定的时间级别计算评分指标
                if indi_key == 'macd':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'MACD', self.calculate_macd, base_ohlcv_df,
                                                                    period_fast=bs_params['macd_fast'],
                                                                    period_slow=bs_params['macd_slow'],
                                                                    signal_period=bs_params['macd_signal']))
                elif indi_key == 'rsi':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'RSI', self.calculate_rsi, base_ohlcv_df, period=bs_params['rsi_period']))
                elif indi_key == 'kdj':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'KDJ', self.calculate_kdj, base_ohlcv_df,
                                                                    period=bs_params['kdj_period_k'],
                                                                    signal_period=bs_params['kdj_period_d'],
                                                                    smooth_k_period=bs_params['kdj_period_j']))
                elif indi_key == 'boll':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'BOLL', self.calculate_boll, base_ohlcv_df, period=bs_params['boll_period'], std_dev=bs_params['boll_std_dev']))
                elif indi_key == 'cci':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'CCI', self.calculate_cci, base_ohlcv_df, period=bs_params['cci_period']))
                elif indi_key == 'mfi':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'MFI', self.calculate_mfi, base_ohlcv_df, period=bs_params['mfi_period']))
                elif indi_key == 'roc':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'ROC', self.calculate_roc, base_ohlcv_df, period=bs_params['roc_period']))
                elif indi_key == 'dmi':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'DMI', self.calculate_dmi, base_ohlcv_df, period=bs_params['dmi_period']))
                elif indi_key == 'sar':
                    indicator_tasks.append(_calculate_and_store_async(tf, 'SAR', self.calculate_sar, base_ohlcv_df, af=bs_params['sar_step'], max_af=bs_params['sar_max']))

            # 计算成交量确认指标 (仅在指定的时间级别计算)
            if vc_params.get('enabled', False) and tf == vc_params.get('tf'):
                indicator_tasks.append(_calculate_and_store_async(tf, 'AMT_MA', self.calculate_amount_ma, base_ohlcv_df, period=vc_params.get('amount_ma_period', 10)))
                indicator_tasks.append(_calculate_and_store_async(tf, 'CMF', self.calculate_cmf, base_ohlcv_df, period=vc_params.get('cmf_period', 20)))
            # OBV 均线在 OBV 计算后处理
            indicator_tasks.append(_calculate_and_store_async(tf, 'OBV', self.calculate_obv, base_ohlcv_df))

            # 计算分析所需的指标 (STOCH, VOL_MA, VWAP) - 在所有需要的时间级别计算
            # STOCH
            indicator_tasks.append(_calculate_and_store_async(tf, 'STOCH', self.calculate_stoch, base_ohlcv_df,
                                                            k_period=ia_params.get('stoch_k', 14),
                                                            d_period=ia_params.get('stoch_d', 3),
                                                            smooth_k_period=ia_params.get('stoch_smooth_k', 3)))
            # VOL_MA
            # 修正：df.ta.sma 返回的是 DataFrame，不需要 to_frame()，直接重命名列
            vol_ma_period = ia_params.get('volume_ma_period', 20)
            indicator_tasks.append(_calculate_and_store_async(tf, 'VOL_MA',
                lambda df: df.ta.sma(close='volume', length=vol_ma_period).rename(columns=lambda col_name: f'VOL_MA_{vol_ma_period}'),
                base_ohlcv_df))
            # VWAP
            indicator_tasks.append(_calculate_and_store_async(tf, 'VWAP', self.calculate_vwap, base_ohlcv_df))

        # 并行执行所有指标计算任务
        results = await asyncio.gather(*indicator_tasks, return_exceptions=True)
        for result in results:
            if result and isinstance(result, tuple):
                tf, indicator_df = result
                calculated_indicators[tf].append(indicator_df)

        # --- 修改：后处理 OBV 均线 (在所有成功计算 OBV 的级别上计算) ---
        obv_ma_period = vc_params.get('obv_ma_period', 10) # 获取均线周期参数
        # 遍历所有时间级别，检查是否计算出了 OBV
        for tf in all_time_levels: # 或者遍历 calculated_indicators.keys() 更安全
            if tf not in calculated_indicators: # 如果这个级别没有任何指标计算成功，跳过
                continue
            # 查找该时间级别下计算出的 OBV 数据帧
            obv_df = next((df for df in calculated_indicators.get(tf, []) if f'OBV_{tf}' in df.columns), None)
            if obv_df is not None:
                obv_series_name = f'OBV_{tf}'
                # 检查 OBV 列是否存在且包含有效数据
                if obv_series_name in obv_df.columns and not obv_df[obv_series_name].isnull().all():
                    try:
                        obv_ma_series = ta.sma(obv_df[obv_series_name], length=obv_ma_period)
                        # 检查均线计算结果是否有效
                        if obv_ma_series is not None and not obv_ma_series.isnull().all():
                            obv_ma_df = obv_ma_series.to_frame(name=f'OBV_MA_{obv_ma_period}_{tf}')
                            calculated_indicators[tf].append(obv_ma_df) # 添加到对应 tf 的列表中
                            # logger.debug(f"[{stock_code}] 时间级别 {tf} 计算 OBV 均线 (周期 {obv_ma_period}) 完成。") # 使用 debug 级别
                        else:
                            logger.warning(f"[{stock_code}] 时间级别 {tf} 计算 OBV 均线 (周期 {obv_ma_period}) 结果为空或全为 NaN。OBV 数据可能存在问题。")
                            # logger.debug(f"[{stock_code}] 时间级别 {tf} 用于计算 MA 的 OBV 数据 (前5行):\n{obv_df[obv_series_name].head()}") # 可选调试信息
                    except Exception as e:
                         logger.error(f"[{stock_code}] 时间级别 {tf} 计算 OBV 均线 (周期 {obv_ma_period}) 时出错: {e}", exc_info=True)
                else:
                    logger.warning(f"[{stock_code}] 时间级别 {tf} 未找到有效的 OBV 列 ({obv_series_name}) 或该列全为 NaN，无法计算 OBV 均线。")
            # else: # OBV 本身就没计算出来，MA 自然也无法计算
            #     logger.debug(f"[{stock_code}] 时间级别 {tf} 未找到 OBV 数据帧，跳过 OBV 均线计算。")

        # 后处理布林带列名修正 (已在 _calculate_and_store_async 添加后缀，这里可能不需要额外修正，除非 pandas_ta 返回的原始列名不是标准格式)
        # 检查 calculate_boll 的返回值列名格式，如果它是 BB_UPPER_period 这种格式，_calculate_and_store_async 会自动加上时间后缀
        # 如果它返回的就是 BB_UPPER 这种，那需要在这里根据 bs_params['boll_period'] 找到它并重命名
        # 假设 _calculate_and_store_async 已经处理了列名后缀，这里不需要额外修正

        # 6. 合并所有重采样后的 OHLCV 数据和计算的指标数据到一个 DataFrame
        all_dfs_to_merge = []
        # 添加重采样后的 OHLCV 数据
        for tf, df_tf in resampled_ohlcv_dfs.items():
            if df_tf is not None and not df_tf.empty:
                 all_dfs_to_merge.append(df_tf.reindex(base_index, method='ffill')) # 对齐到基准索引并前向填充

        # 添加计算的指标数据
        for tf, indicator_dfs_list in calculated_indicators.items():
             for indicator_df in indicator_dfs_list:
                 if indicator_df is not None and not indicator_df.empty:
                      all_dfs_to_merge.append(indicator_df.reindex(base_index, method='ffill')) # 对齐到基准索引并前向填充


        if not all_dfs_to_merge:
            logger.error(f"[{stock_code}] 没有可用的数据帧进行合并（重采样后或指标计算失败）。")
            return None

        # 使用 pd.concat 合并所有 DataFrame，按索引外连接对齐，并处理重复列名
        try:
            combined_df = pd.concat(all_dfs_to_merge, axis=1, join='outer', copy=False)
            # 处理重复列名，保留第一个出现的列 (例如，不同时间级别的 close 列)
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
            logger.info(f"[{stock_code}] 成功合并所有数据帧，形状: {combined_df.shape}")

            # --- 最终的填充逻辑 (填充合并后和指标计算初期产生的 NaN) ---
            # 这些 NaN 可能是由于不同时间级别数据起始点不同，指标计算初期回看期不足产生
            combined_df.ffill(inplace=True)
            combined_df.bfill(inplace=True) # bfill 用于填充序列最开始的 NaN 值

            # --- 记录最终填充后的缺失状态 (重要) ---
            logger.info(f"[{stock_code}] 合并后数据填充完成，最终形状: {combined_df.shape}")
            self._log_dataframe_missing(combined_df, stock_code) # 记录填充后的状态

            # --- 强制类型转换 ---
            # 在合并和填充后，再次确保所有列是数值类型
            # all_dfs_were_empty = all(df is None or df.empty for df in all_dfs_to_merge) # 检查是否所有输入都为空
            for col in combined_df.columns:
                # 使用 errors='coerce' 将无法转换为数值的强制转为 NaN
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                # if combined_df[col].isnull().all() and not all_dfs_were_empty:
                #      logger.warning(f"[{stock_code}] 列 '{col}' 在填充和数值转换后全部为 NaN。") # 此检查可能过于严格，因为有些列可能确实为空（如某些时间级别的某些指标）

            # 最终检查合并后的 DataFrame 是否有效 (例如，是否包含所有必要的特征，并且不是全NaN)
            # 具体的必要特征列表需要根据您的机器学习模型来确定
            # 至少，最小时间级别的 OHLCV 列应该存在且非空
            min_tf_ohlcv_cols = [f'{col}_{min_time_level}' for col in ['open', 'high', 'low', 'close', 'volume']]
            if not all(col in combined_df.columns for col in min_tf_ohlcv_cols):
                 missing_min_tf_cols = [col for col in min_tf_ohlcv_cols if col not in combined_df.columns]
                 logger.error(f"[{stock_code}] 合并后的 DataFrame 缺少最小时间级别 ({min_time_level}) 的关键 OHLCV 列: {missing_min_tf_cols}。终止数据准备。")
                 return None

            if combined_df[min_tf_ohlcv_cols].isnull().all().all() or combined_df.isnull().all().all():
                logger.error(f"[{stock_code}] 合并后的 DataFrame 在关键列 ({min_tf_ohlcv_cols}) 或所有列上全部为 NaN。数据无效。终止数据准备。")
                return None


            return combined_df
        except Exception as e:
            logger.error(f"[{stock_code}] 合并数据帧时出错: {e}", exc_info=True)
            return None

    # --- 周期对齐函数 ---
    # 这个函数在引入重采样后，不再用于主要的时间序列标准化，
    # 但可以在重采样后用于额外的验证或在特定情况下使用。
    # 当前修改方案中，主要依赖重采样。可以保留此函数，但确保在 prepare_strategy_dataframe 中不再用于原始数据过滤。
    def filter_to_period_points(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        尝试保留周期对齐的时间点。
        注意：在引入重采样后，此函数通常不再用于原始数据过滤，可能用于重采样后的额外处理或验证。
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"DataFrame 索引不是 DatetimeIndex，无法进行周期点过滤: {tf}")
            return df

        if tf.isdigit() and int(tf) > 0 and int(tf) < 1440: # 过滤分钟周期
            period = int(tf)
            # 尝试找到数据中最常见的分钟余数，并以此作为对齐点
            # 这对于不规则时间戳的数据来说可能不够准确
            try:
                # 只考虑交易时间内的分钟
                # 示例：上午 9:30-11:30, 下午 13:00-15:00
                trading_hours_mask = ((df.index.time >= datetime.time(9, 30)) & (df.index.time <= datetime.time(11, 30))) | \
                                      ((df.index.time >= datetime.time(13, 0)) & (df.index.time <= datetime.time(15, 0)))
                trading_minutes = df.index[trading_hours_mask].minute
                if trading_minutes.empty:
                     logger.warning(f"在交易时段内没有找到分钟数据点，无法确定周期点: {tf}")
                     return df # 或者返回空DataFrame，取决于需求

                mod_counts = trading_minutes % period

                # 检查 mod_counts 是否为空，避免在没有数据点时调用 idxmax
                if mod_counts.empty:
                    logger.warning(f"计算分钟余数时发现空数据，无法确定周期点: {tf}")
                    return df

                # 找到出现次数最多的余数作为对齐基准
                # 如果所有余数出现次数相同，idxmax 返回第一个
                most_common_mod = mod_counts.value_counts().idxmax()

                # 过滤出分钟部分与最常见余数匹配且秒为0的时间点
                # 注意：您的数据时间戳带有秒，这里秒==0的条件可能导致大量数据丢失
                # 更合适的做法是根据数据源特点调整这里的对齐逻辑
                # 暂时保留秒为0的条件，但要注意其潜在影响
                mask = (df.index.minute % period == most_common_mod) & (df.index.second == 0)
                filtered_df = df[mask]

                logger.info(f"对齐分钟时间到 {period} 分钟间隔 (模数 {most_common_mod})，原始时间点数量: {len(df)}，对齐后数量: {len(filtered_df)}，股票: {df.index.name if hasattr(df.index, 'name') else 'N/A'} {tf}")

                # 如果过滤后数据量为零，可能需要记录警告或返回原始 df
                if filtered_df.empty and not df.empty:
                     logger.warning(f"对齐周期点 {tf} 后 DataFrame 为空，原始数据量为 {len(df)}。请检查时间戳模式和对齐逻辑。")

                return filtered_df
            except Exception as e:
                 logger.error(f"过滤周期点 {tf} 时出错: {e}", exc_info=True)
                 return df # 出错时返回原始 DataFrame


        else: # 日、周、月等时间级别不进行此分钟级别的周期点过滤
            return df


    # --- 检查缺失项目并记录 ---
    def _log_dataframe_missing(self, df: pd.DataFrame, stock_code: str):
        """
        记录 DataFrame 中填充后的缺失值情况，只报告存在缺失的列。
        """
        if df.empty:
            logger.warning(f"[{stock_code}] DataFrame 为空，无法检查缺失值。")
            return

        # 计算所有列的缺失数量和比例
        missing_count = df.isna().sum()
        missing_ratio = (df.isna().mean() * 100).round(2)

        # 筛选出缺失数量大于 0 的列
        non_zero_missing_count = missing_count[missing_count > 0]
        non_zero_missing_ratio = missing_ratio[non_zero_missing_count.index] # 使用相同索引确保对应

        # 检查是否有任何列存在缺失
        if not non_zero_missing_count.empty:
            logger.warning(f"[{stock_code}] 合并填充后存在缺失值的列 - 数量: {non_zero_missing_count.to_dict()}")
            logger.warning(f"[{stock_code}] 合并填充后存在缺失值的列 - 比例(%): {non_zero_missing_ratio.to_dict()}")

            # 检查关键列的缺失情况 (只检查那些实际有缺失的关键列)
            # 关键列可能是包含 open, high, low, close, volume 的列，无论时间级别
            key_indicators = ['open', 'high', 'low', 'close', 'volume', 'macd', 'rsi', 'kdj', 'boll', 'cci', 'mfi', 'roc', 'adx', 'dmp', 'dmn', 'sar', 'stoch', 'vol_ma', 'vwap', 'amount', 'turnover_rate']
            key_cols = [col for col in df.columns if any(indicator_name in col.lower() for indicator_name in key_indicators)] # 不区分大小写匹配

            # 从有缺失的列中筛选出关键列
            key_missing_count_filtered = non_zero_missing_count.filter(items=key_cols)
            key_missing_ratio_filtered = non_zero_missing_ratio.filter(items=key_cols)

            if not key_missing_count_filtered.empty:
                logger.warning(f"[{stock_code}] 关键列缺失数量 (仅显示有缺失的): {key_missing_count_filtered.to_dict()}")
                logger.warning(f"[{stock_code}] 关键列缺失比例(%)(仅显示有缺失的): {key_missing_ratio_filtered.to_dict()}")

            # 报告缺失比例最高的列 (只考虑实际有缺失的列)
            top_missing = non_zero_missing_ratio.sort_values(ascending=False).head(5)
            if not top_missing.empty:
                 logger.warning(f"[{stock_code}] 缺失比例最高的5列 (仅显示有缺失的): {top_missing.to_dict()}")

        else:
            # 如果没有任何列缺失，打印一条信息日志
            logger.info(f"[{stock_code}] 合并填充后所有列数据完整，无缺失值。")

        # 检查是否存在整行都是 NaN 的情况 (这个检查仍然有用)
        all_nan_rows = df.isna().all(axis=1).sum()
        if all_nan_rows > 0:
            logger.warning(f"[{stock_code}] 注意：合并填充后 DataFrame 仍存在 {all_nan_rows} 行全部为 NaN 的数据！这些行可能无法用于训练/预测。")


    def calculate_atr(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 ATR。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): ATR 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 "ATR_period" 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            atr_series = ohlc.ta.atr(length=period)
            if atr_series is None: return None
            # 返回 DataFrame，列名为 ATR_period
            return atr_series.to_frame(name=f'ATR_{period}')
        except Exception as e:
            logger.error(f"计算 ATR(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_boll(self, ohlc: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Optional[pd.DataFrame]:
        """
        计算布林带 (标准周期)
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): 布林带计算周期。
            std_dev (float): 标准差倍数。
        Returns:
            Optional[pd.DataFrame]: 包含 'BB_UPPER_period', 'BB_MIDDLE_period', 'BB_LOWER_period'
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            bbands_df = ohlc.ta.bbands(length=period, std=std_dev)
            if bbands_df is None or bbands_df.empty:
                return None
            # 重命名列以匹配 finta 的输出，并包含 period 信息
            rename_map = {
                f'BBU_{period}_{std_dev}': f'BB_UPPER_{period}',
                f'BBM_{period}_{std_dev}': f'BB_MIDDLE_{period}',
                f'BBL_{period}_{std_dev}': f'BB_LOWER_{period}',
            }
            required_cols = list(rename_map.keys())
            actual_cols = {col: rename_map[col] for col in required_cols if col in bbands_df.columns}
            if len(actual_cols) != 3:
                logger.warning(f"pandas-ta bbands 未返回所有预期列 for period={period}, std={std_dev}. 返回: {bbands_df.columns.tolist()}")
                potential_map = {'BBU': f'BB_UPPER_{period}', 'BBM': f'BB_MIDDLE_{period}', 'BBL': f'BB_LOWER_{period}'}
                common_cols = {f'{k}_{period}_{std_dev}': v for k, v in potential_map.items() if f'{k}_{period}_{std_dev}' in bbands_df.columns}
                if len(common_cols) == 3:
                    actual_cols = common_cols
                else:
                    logger.error(f"无法从 pandas-ta bbands 结果中提取所需列。")
                    found_df = bbands_df[[col for col in actual_cols.keys()]].rename(columns=actual_cols)
                    return found_df if not found_df.empty else None
            return bbands_df[list(actual_cols.keys())].rename(columns=actual_cols)
        except Exception as e:
            logger.error(f"计算 BOLL({period}, {std_dev}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_cci(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 CCI。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): CCI 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'CCI_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            cci_series = ohlc.ta.cci(length=period)
            if cci_series is None: return None
            # 返回 DataFrame，列名为 CCI_period
            return cci_series.to_frame(name=f'CCI_{period}')
        except Exception as e:
            logger.error(f"计算 CCI(period={period}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_cmf(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 CMF。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'volume' 列)。
            period (int): CMF 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'CMF_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'volume' not in ohlc.columns:
            logger.error("计算 CMF 需要 'volume' 列")
            return None
        if len(ohlc) < period: return None
        try:
            cmf_series = ohlc.ta.cmf(length=period)
            if cmf_series is None: return None
            # 返回 DataFrame，列名为 CMF_period
            return cmf_series.to_frame(name=f'CMF_{period}')
        except Exception as e:
            logger.error(f"计算 CMF(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_dmi(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 DMI (+DI, -DI, ADX)。
        注意：不直接计算 ADXR。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): DMI 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 '+DI_period', '-DI_period', 'ADX_period' 列的 DataFrame。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # ADX 通常需要至少 2*period 的数据
        if ohlc is None or ohlc.empty or len(ohlc) < period * 2: return None
        try:
            adx_df = ohlc.ta.adx(length=period)
            if adx_df is None or adx_df.empty: return None
            # 提取所需的列并重命名
            pdi_col_ta = f'DMP_{period}'
            mdi_col_ta = f'DMN_{period}'
            adx_col_ta = f'ADX_{period}'
            pdi_col_strat = f'+DI_{period}'
            mdi_col_strat = f'-DI_{period}'
            adx_col_strat = f'ADX_{period}' # 保持 ADX 名称
            found_cols = {}
            if pdi_col_ta in adx_df.columns: found_cols[pdi_col_ta] = pdi_col_strat
            if mdi_col_ta in adx_df.columns: found_cols[mdi_col_ta] = mdi_col_strat
            if adx_col_ta in adx_df.columns: found_cols[adx_col_ta] = adx_col_strat
            if len(found_cols) < 3:
                 logger.warning(f"无法从 pandas-ta adx(length={period}) 结果中找到所有 PDI/MDI/ADX 列。返回列: {adx_df.columns.tolist()}")
                 # 尝试返回找到的列
                 if not found_cols: return None
                 return adx_df[list(found_cols.keys())].rename(columns=found_cols)
            return adx_df[list(found_cols.keys())].rename(columns=found_cols)
        except Exception as e:
            logger.error(f"计算 DMI(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_ichimoku(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算一目均衡表示线 (Ichimoku)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
        Returns:
            Optional[pd.DataFrame]: 包含 'TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B' 列的 DataFrame。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # Ichimoku 默认参数 9, 26, 52。需要数据量 > 52
        min_required = 52 # Kijun=26, Senkou=52 lookbacks
        # if ohlc is None or ohlc.empty or len(ohlc) < min_required:
            # logger.warning(f"数据不足 ({len(ohlc)} < {min_required}) 可能无法准确计算 Ichimoku")
            # return None # 保持原有逻辑，尝试计算，pandas-ta 内部会处理

        try:
            # pandas-ta ichimoku 返回 DataFrame 和 SPAN A/B 的 shifted 版本
            # 我们通常需要未 shift 的版本: ITS_9, IKS_26, ICS_26, ISA_9, ISB_26 (根据日志调整)
            # 参数: tenkan=9, kijun=26, senkou=52
            ichi_df, span_shifted_df = ohlc.ta.ichimoku(tenkan=9, kijun=26, senkou=52)

            if ichi_df is None or ichi_df.empty:
                logger.warning("pandas-ta ichimoku 计算返回空")
                return None

            # 重命名列以匹配 finta 的输出 ('TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU A', 'SENKOU B')
            # 注意：根据错误日志调整 ISA 和 ISB 的键名
            rename_map = {
                'ITS_9': 'TENKAN',         # Tenkan-sen (Conversion Line)
                'IKS_26': 'KIJUN',         # Kijun-sen (Base Line)
                'ICS_26': 'CHIKOU',        # Chikou Span (Lagging Span)
                # 'ISA_9_26_52': 'SENKOU A', # --- 旧的错误的键名 ---
                'ISA_9': 'SENKOU A',       # --- 根据日志调整后的键名 --- Senkou Span A (Leading Span A)
                # 'ISB_26_52': 'SENKOU B', # --- 旧的错误的键名 ---
                'ISB_26': 'SENKOU B',      # --- 根据日志调整后的键名 --- Senkou Span B (Leading Span B)
            }
            required_keys = list(rename_map.keys())
            actual_cols_map = {key: rename_map[key] for key in required_keys if key in ichi_df.columns}
            if len(actual_cols_map) != 5:
                # 记录更详细的错误，包括期望的和实际的
                expected_cols = required_keys
                returned_cols = ichi_df.columns.tolist()
                logger.error(f"无法从 pandas-ta ichimoku 结果中提取所有所需列。期望键名: {expected_cols}, 实际返回列: {returned_cols}")
                # 仍然尝试返回能找到的列，避免完全失败
                found_df = ichi_df[[key for key in actual_cols_map.keys()]].rename(columns=actual_cols_map)
                return found_df if not found_df.empty else None
            # 提取并重命名所需的列
            result_df = ichi_df[list(actual_cols_map.keys())].rename(columns=actual_cols_map)
            return result_df
        except Exception as e:
            logger.error(f"计算 Ichimoku 失败: {e}", exc_info=True)
            return None

    def calculate_kdj(self, ohlc: pd.DataFrame, period: int, signal_period: int, smooth_k_period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 KDJ (K, D, J)。
        【修改】: 返回列名为 K_period_signal, D_period_signal, J_period_signal
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): K 计算周期 (N)。
            signal_period (int): D 计算周期 (M1)。
            smooth_k_period (int): K 平滑周期 (M2)。
        Returns:
            Optional[pd.DataFrame]: 包含 'K_period_signal', 'D_period_signal', 'J_period_signal' 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period + signal_period + smooth_k_period: return None
        try:
            kdj_df = ohlc.ta.kdj(length=period, signal=signal_period, smooth_k=smooth_k_period)
            if kdj_df is None or kdj_df.empty: return None
            # 构建 pandas-ta 可能的列名
            k_col_ta = f'K_{period}_{signal_period}'
            d_col_ta = f'D_{period}_{signal_period}'
            j_col_ta = f'J_{period}_{signal_period}'
            # 策略期望的基础列名 (不带时间后缀)
            k_col_base = f'K_{period}_{signal_period}'
            d_col_base = f'D_{period}_{signal_period}'
            j_col_base = f'J_{period}_{signal_period}'
            rename_map = {}
            if k_col_ta in kdj_df.columns: rename_map[k_col_ta] = k_col_base
            if d_col_ta in kdj_df.columns: rename_map[d_col_ta] = d_col_base
            if j_col_ta in kdj_df.columns: rename_map[j_col_ta] = j_col_base
            # 回退到位置索引
            if len(rename_map) < 3:
                logger.warning(f"KDJ 列名不匹配预期。尝试位置索引。返回列: {kdj_df.columns.tolist()}")
                if k_col_base not in rename_map.values() and len(kdj_df.columns) >= 1:
                     # 检查原始列名是否已经是期望的基础名
                     if kdj_df.columns[0] == k_col_base:
                         rename_map[kdj_df.columns[0]] = k_col_base
                     else: # 否则用位置索引重命名
                         rename_map[kdj_df.columns[0]] = k_col_base
                if d_col_base not in rename_map.values() and len(kdj_df.columns) >= 2:
                     if kdj_df.columns[1] == d_col_base:
                          rename_map[kdj_df.columns[1]] = d_col_base
                     else:
                          rename_map[kdj_df.columns[1]] = d_col_base
                if j_col_base not in rename_map.values() and len(kdj_df.columns) >= 3:
                     if kdj_df.columns[2] == j_col_base:
                          rename_map[kdj_df.columns[2]] = j_col_base
                     else:
                          rename_map[kdj_df.columns[2]] = j_col_base
            if len(rename_map) < 3:
                 logger.error(f"无法从 pandas-ta kdj 结果中提取 K, D, J 列。找到的映射: {rename_map}")
                 # 只返回能找到的列，并使用期望的基础名称
                 if not rename_map: return None
                 return kdj_df[[col for col in rename_map.keys()]].rename(columns=rename_map)
            # 返回包含 K_period_signal, D_period_signal, J_period_signal 的 DataFrame
            return kdj_df[[col for col in rename_map.keys()]].rename(columns=rename_map)
        except Exception as e:
            logger.error(f"计算 KDJ(k={period}, s={signal_period}, sm={smooth_k_period}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_ema(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 EMA。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): EMA 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'EMA_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            # 假设计算收盘价的 EMA
            ema_series = ohlc.ta.ema(close='close', length=period) # 明确指定 close='close'
            if ema_series is None: return None
            # 返回 DataFrame，列名为 EMA_period
            return ema_series.to_frame(name=f'EMA_{period}')
        except Exception as e:
            logger.error(f"计算 EMA(period={period}) 失败: {e}", exc_info=True)
            return None
        
    def calculate_amount_ma(self, ohlc: pd.DataFrame, period: int, tf: str = "") -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交额 SMA，返回列名为 AMT_MA_周期_时间框，满足策略列名需求。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据，必须包含 'amount' 列。
            period (int): SMA 计算周期。
            tf (str): 时间框后缀，例如 '15', 用于生成列名。
        Returns:
            Optional[pd.DataFrame]: 计算结果DataFrame，列名如 'AMT_MA_15_15'。
                                计算失败返回 None。
        """
        if ta is None:
            logger.error("pandas-ta 未加载，无法计算成交额均线")
            return None
        if ohlc is None or ohlc.empty:
            logger.error("输入的 OHLCV 数据为空，无法计算成交额均线")
            return None
        if 'amount' not in ohlc.columns:
            logger.error("计算 Amount MA 需要数据包含 'amount' 列")
            return None
        if len(ohlc) < period:
            logger.warning(f"数据长度 {len(ohlc)} 小于所需周期 {period}，成交额均线计算返回 None")
            return None
        try:
            # pandas_ta 用 sma 计算成交额均线，输入列名为 'amount'
            amt_ma_series = ohlc.ta.sma(close='amount', length=period)
            if amt_ma_series is None or amt_ma_series.empty:
                logger.warning("成交额均线计算结果为空")
                return None

            # 构建列名，格式为 AMT_MA_周期_时间框
            col_name = f'AMT_MA_{period}'
            if tf:
                col_name = f'AMT_MA_{period}_{tf}'

            # 转换为 DataFrame，返回
            result_df = amt_ma_series.to_frame(name=col_name)
            return result_df

        except Exception as e:
            logger.error(f"计算成交额均线 AMT_MA(period={period}, tf={tf}) 失败: {e}", exc_info=True)
            return None

    def calculate_macd(self, ohlc: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9) -> Optional[pd.DataFrame]:
        """
        计算标准 MACD(fast, slow, signal)。
        返回包含 'MACD_f_s_g', 'MACDh_f_s_g', 'MACDs_f_s_g' 列的 DataFrame。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period_fast (int): 快线 EMA 周期。
            period_slow (int): 慢线 EMA 周期。
            signal_period (int): 信号线 EMA 周期。
        Returns:
            Optional[pd.DataFrame]: 包含 MACD, MACD Histogram, MACD Signal 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period_slow + signal_period: return None
        try:
            # 直接调用 pandas-ta macd
            macd_df = ohlc.ta.macd(fast=period_fast, slow=period_slow, signal=signal_period)
            if macd_df is None or macd_df.empty:
                logger.warning(f"pandas-ta macd({period_fast}, {period_slow}, {signal_period}) 计算失败或返回空")
                return None
            # 不再进行重命名，直接返回 pandas-ta 的结果
            # 列名将是 'MACD_f_s_g', 'MACDh_f_s_g', 'MACDs_f_s_g'
            return macd_df

        except Exception as e:
            logger.error(f"计算 MACD({period_fast}, {period_slow}, {signal_period}) 失败: {e}", exc_info=True)
            return None

    def calculate_mfi(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 MFI。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): MFI 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'MFI_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                mfi_series = ohlc.ta.mfi(length=period)
            if mfi_series is None: return None
            # 返回 DataFrame，列名为 MFI_period
            return mfi_series.to_frame(name=f'MFI_{period}')
        except Exception as e:
            logger.error(f"计算 MFI(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_mom(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 MOM。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): MOM 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'MOM_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            mom_series = ohlc.ta.mom(length=period)
            if mom_series is None: return None
            # 返回 DataFrame，列名为 MOM_period
            return mom_series.to_frame(name=f'MOM_{period}')
        except Exception as e:
            logger.error(f"计算 MOM(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_obv(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 OBV。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
        Returns:
            Optional[pd.DataFrame]: 包含 'OBV' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        # 假设 DAO 层已处理重复索引问题
        if ohlc is None or ohlc.empty: return None
        try:
            # pandas-ta obv 返回 Series，名称为 OBV
            obv_series = ohlc.ta.obv()
            if obv_series is None: return None
            # 返回 DataFrame，列名为 OBV
            return obv_series.to_frame(name='OBV')
        except Exception as e:
            logger.error(f"计算 OBV 失败: {e}", exc_info=True)
            return None

    def calculate_roc(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 ROC。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): ROC 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'ROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period + 1: return None
        try:
            roc_series = ohlc.ta.roc(length=period)
            if roc_series is None: return None
            # 返回 DataFrame，列名为 ROC_period
            df_result = roc_series.to_frame(name=f'ROC_{period}')
            df_result.replace([np.inf, -np.inf], np.nan, inplace=True) # 处理 inf
            return df_result
        except Exception as e:
            logger.error(f"计算 ROC(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_amount_roc(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交额 ROC。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'amount' 列)。
            period (int): ROC 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'AROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'amount' not in ohlc.columns:
             logger.error("计算 Amount ROC 需要 'amount' 列")
             return None
        if len(ohlc) < period + 1: return None
        try:
            aroc_series = ohlc.ta.roc(close='amount', length=period)
            if aroc_series is None: return None
            # 返回 DataFrame，列名为 AROC_period
            df_results = aroc_series.to_frame(name=f'AROC_{period}')
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Amount ROC(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_rsi(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 RSI。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): RSI 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'RSI_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period + 1: return None
        try:
            rsi_series = ohlc.ta.rsi(length=period)
            if rsi_series is None: return None
            # 返回 DataFrame，列名为 RSI_period
            return rsi_series.to_frame(name=f'RSI_{period}')
        except Exception as e:
            logger.error(f"计算 RSI(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_sar(self, ohlc: pd.DataFrame, af: float = 0.02, max_af: float = 0.2) -> Optional[pd.DataFrame]:
        """计算 SAR (使用 pandas-ta 的 psar)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < 2: return None
        try:
            # 使用 pandas-ta 的 psar 函数，注意参数名 af0 和 af_max
            # psar 返回一个包含多列的 DataFrame
            psar_df = ohlc.ta.psar(af0=af, af_max=max_af) # 使用 psar 并传入对应参数
            if psar_df is None or psar_df.empty:
                logger.warning("pandas-ta psar 计算返回空")
                return None
            # psar 返回的列名通常包含参数，例如 PSARl_0.02_0.2, PSARs_0.02_0.2
            # 我们需要找到实际的列名
            psarl_col = next((col for col in psar_df.columns if col.startswith('PSARl')), None)
            psars_col = next((col for col in psar_df.columns if col.startswith('PSARs')), None)
            if psarl_col and psars_col:
                # 合并 PSARl 和 PSARs 列得到单一的 SAR 值
                # 当 PSARl 为 NaN 时取 PSARs 的值，反之亦然
                sar_series = psar_df[psarl_col].fillna(psar_df[psars_col])
                # 返回 DataFrame，列名为 SAR，以兼容 DAO
                return sar_series.to_frame(name='SAR')
            else:
                logger.error(f"无法从 pandas-ta psar 结果中找到 PSARl 或 PSARs 列。返回列: {psar_df.columns.tolist()}")
                return None
        except Exception as e:
            # 捕获可能的其他错误，例如参数错误
            logger.error(f"计算 SAR (psar) 失败: {e}", exc_info=True)
            return None

    def calculate_vroc(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的成交量 ROC (VROC)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据 (必须包含 'volume' 列)。
            period (int): VROC 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'VROC_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or 'volume' not in ohlc.columns:
             logger.error("计算 VROC 需要 'volume' 列")
             return None
        if len(ohlc) < period + 1: return None
        try:
            vroc_series = ohlc.ta.roc(close='volume', length=period)
            if vroc_series is None: return None
            # 返回 DataFrame，列名为 VROC_period
            df_results = vroc_series.to_frame(name=f'VROC_{period}')
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 VROC(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_vwap(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 VWAP (日内 VWAP 需要 timestamp index)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
        Returns:
            Optional[pd.DataFrame]: 包含 'VWAP' 列的 DataFrame，如果计算失败则返回 None。
                                     列名通常由 pandas_ta 决定 (e.g., VWAP_D)，会尝试重命名为 'VWAP'。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty: return None
        # pandas-ta vwap 依赖于 DatetimeIndex 来确定重置周期 (通常是每日)
        # 假定输入 ohlc 的索引已经是 DatetimeIndex (在 _get_ohlcv_data 中处理)
        if not isinstance(ohlc.index, pd.DatetimeIndex):
            logger.warning("计算 VWAP 需要 DataFrame 的 index 是 DatetimeIndex。尝试转换...")
            try:
                original_index = ohlc.index # 保存原始索引以备恢复
                ohlc.index = pd.to_datetime(ohlc.index)
            except Exception as e:
                logger.error(f"无法将 index 转换为 DatetimeIndex，VWAP 计算可能不准确或失败: {e}")
                # 可以在这里决定是否返回 None 或继续尝试
                # return None
        try:
            # pandas-ta vwap 返回 Series，名称通常是 VWAP_D (日级别) 或类似
            vwap_series = ohlc.ta.vwap() # 默认计算日 VWAP
            if vwap_series is None:
                 logger.warning("pandas-ta vwap() 返回 None")
                 return None
            # 尝试将返回的 Series 重命名为 'VWAP'
            # 获取 pandas_ta 返回的实际列名 (通常只有一个)
            vwap_col_name = vwap_series.name if hasattr(vwap_series, 'name') else None
            if vwap_col_name:
                 return vwap_series.to_frame(name='VWAP')
            else:
                 # 如果没有名称，直接创建一个名为 VWAP 的列
                 logger.warning("pandas-ta vwap() 返回的 Series 没有名称，将尝试直接创建 'VWAP' 列")
                 return pd.DataFrame({'VWAP': vwap_series}, index=ohlc.index)
        except Exception as e:
            logger.error(f"计算 VWAP 失败: {e}", exc_info=True)
            return None
        finally:
            # 恢复索引 (如果之前转换过)
            if 'original_index' in locals():
                ohlc.index = original_index
                pass

    def calculate_wr(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 Williams %R (WR)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): WR 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'WR_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            wr_series = ohlc.ta.willr(length=period)
            if wr_series is None: return None
            # 返回 DataFrame，列名为 WR_period
            return wr_series.to_frame(name=f'WR_{period}')
        except Exception as e:
            logger.error(f"计算 WR(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_sma(self, ohlc: pd.DataFrame, period: int) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 SMA。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): SMA 计算周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'SMA_period' 列的 DataFrame，如果计算失败则返回 None。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < period: return None
        try:
            sma_series = ohlc.ta.sma(length=period)
            if sma_series is None: return None
            # 返回 DataFrame，列名为 SMA_period
            return sma_series.to_frame(name=f'SMA_{period}')
        except Exception as e:
            logger.error(f"计算 SMA(period={period}) 失败: {e}", exc_info=True)
            return None

    def calculate_kc(self, ohlc: pd.DataFrame, period: int, atr_length: int = 10, scalar: float = 2.0) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 Keltner Channels (KC)。
        使用 period 作为 EMA 的长度 (length)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            period (int): EMA 计算周期。
            atr_length (int): ATR 计算周期。
            scalar (float): ATR 乘数。
        Returns:
            Optional[pd.DataFrame]: 包含 'KC_LOWER_period', 'KC_BASIS_period', 'KC_UPPER_period' 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < max(period, atr_length): return None
        try:
            kc_df = ohlc.ta.kc(length=period, atr_length=atr_length, scalar=scalar)
            if kc_df is None or kc_df.empty: return None
            # 提取并重命名
            lower_col_ta = f'KCLe_{period}_{scalar}'
            basis_col_ta = f'KCBe_{period}_{scalar}'
            upper_col_ta = f'KCUe_{period}_{scalar}'
            lower_col_strat = f'KC_LOWER_{period}'
            basis_col_strat = f'KC_BASIS_{period}'
            upper_col_strat = f'KC_UPPER_{period}'
            found_cols = {}
            if lower_col_ta in kc_df.columns: found_cols[lower_col_ta] = lower_col_strat
            if basis_col_ta in kc_df.columns: found_cols[basis_col_ta] = basis_col_strat
            if upper_col_ta in kc_df.columns: found_cols[upper_col_ta] = upper_col_strat
             # 回退到位置索引
            if len(found_cols) < 3:
                 logger.warning(f"pandas-ta kc(p={period}, atr={atr_length}, s={scalar}) 列名不完全匹配预期。尝试位置索引。返回列: {kc_df.columns.tolist()}")
                 if lower_col_strat not in found_cols and len(kc_df.columns) >= 1:
                      found_cols[kc_df.columns[0]] = lower_col_strat
                 if basis_col_strat not in found_cols and len(kc_df.columns) >= 2:
                      found_cols[kc_df.columns[1]] = basis_col_strat
                 if upper_col_strat not in found_cols and len(kc_df.columns) >= 3:
                      found_cols[kc_df.columns[2]] = upper_col_strat
            if len(found_cols) < 3:
                 logger.error(f"无法从 pandas-ta kc 结果中提取 Lower, Basis, Upper 列。")
                 if not found_cols: return None
                 return kc_df[[col for col in found_cols.keys()]].rename(columns=found_cols)
            return kc_df[list(found_cols.keys())].rename(columns=found_cols)
        except Exception as e:
            logger.error(f"计算 KC(p={period}, atr={atr_length}, s={scalar}) 失败: {e}", exc_info=True)
            return None

    def calculate_stoch(self, ohlc: pd.DataFrame, k_period: int, d_period: int = 3, smooth_k_period: int = 3) -> Optional[pd.DataFrame]:
        """
        计算指定周期的 Stochastic Oscillator (%K, %D)。
        使用 k_period 作为 %K 的计算周期 (k)。
        Args:
            ohlc (pd.DataFrame): OHLCV 数据。
            k_period (int): %K 计算周期。
            d_period (int): %D 计算周期。
            smooth_k_period (int): Slow %K 平滑周期。
        Returns:
            Optional[pd.DataFrame]: 包含 'STOCH_K_k_period', 'STOCH_D_k_period' 列。
        """
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or len(ohlc) < k_period + d_period + smooth_k_period: return None
        try:
            stoch_df = ohlc.ta.stoch(k=k_period, d=d_period, smooth_k=smooth_k_period)
            if stoch_df is None or stoch_df.empty: return None
            # 提取并重命名
            k_col_ta = f'STOCHk_{k_period}_{d_period}_{smooth_k_period}'
            d_col_ta = f'STOCHd_{k_period}_{d_period}_{smooth_k_period}'
            k_col_strat = f'STOCH_K_{k_period}' # 策略期望的列名
            d_col_strat = f'STOCH_D_{k_period}' # 策略期望的列名 (基于 K 周期命名)
            found_cols = {}
            if k_col_ta in stoch_df.columns: found_cols[k_col_ta] = k_col_strat
            if d_col_ta in stoch_df.columns: found_cols[d_col_ta] = d_col_strat
            if len(found_cols) < 2:
                logger.warning(f"pandas-ta stoch(k={k_period}, d={d_period}, sm={smooth_k_period}) 列名不完全匹配预期。返回列: {stoch_df.columns.tolist()}")
                if not found_cols: return None
                return stoch_df[[col for col in found_cols.keys()]].rename(columns=found_cols)
            return stoch_df[list(found_cols.keys())].rename(columns=found_cols)
        except Exception as e:
             logger.error(f"计算 Stochastic(k={k_period}, d={d_period}, sm={smooth_k_period}) 失败: {e}", exc_info=True)
             return None

    def calculate_adl(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """计算 Accumulation/Distribution Line (ADL)"""
        if ta is None: logger.error("pandas-ta 未加载"); return None
        if ohlc is None or ohlc.empty or not all(c in ohlc.columns for c in ['high', 'low', 'close', 'volume']):
            logger.error("计算 ADL 需要 'high', 'low', 'close', 'volume' 列")
            return None
        try:
            # pandas-ta ADL 函数名为 ad
            adl_series = ohlc.ta.ad()
            if adl_series is None: return None
            # 返回 DataFrame，列名为 ADL
            return adl_series.to_frame(name='ADL')
        except Exception as e:
            logger.error(f"计算 ADL 失败: {e}", exc_info=True)
            return None

    def calculate_pivot_points(self, ohlc: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        计算 Pivot Points (PP, S1-S4, R1-R4)。
        依赖于 DatetimeIndex 来确定计算周期 (例如基于前一日计算当日)。
        """
        if ohlc is None or ohlc.empty: return None
        try:
            # 确保数据按日期排序
            ohlc = ohlc.sort_index()
            # 创建结果 DataFrame
            results = pd.DataFrame(index=ohlc.index)
            # 创建移位版本的价格数据来计算枢轴点
            # 使用前一交易日的 OHLC 计算当前交易日的枢轴点
            prev_high = ohlc['high'].shift(1)
            prev_low = ohlc['low'].shift(1)
            prev_close = ohlc['close'].shift(1)
            # 计算传统枢轴点
            # PP = (High + Low + Close) / 3
            PP = (prev_high + prev_low + prev_close) / 3
            results['PP'] = PP
            # 计算支撑位
            # S1 = (2 * PP) - High
            results['S1'] = (2 * PP) - prev_high
            # S2 = PP - (High - Low)
            results['S2'] = PP - (prev_high - prev_low)
            # S3 = S1 - (High - Low)
            results['S3'] = results['S1'] - (prev_high - prev_low)
            # S4 = S3 - (High - Low)
            results['S4'] = results['S3'] - (prev_high - prev_low)
            # 计算阻力位
            # R1 = (2 * PP) - Low
            results['R1'] = (2 * PP) - prev_low
            # R2 = PP + (High - Low)
            results['R2'] = PP + (prev_high - prev_low)
            # R3 = R1 + (High - Low)
            results['R3'] = results['R1'] + (prev_high - prev_low)
            # R4 = R3 + (High - Low)
            results['R4'] = results['R3'] + (prev_high - prev_low)
            # 斐波那契枢轴点
            diff = prev_high - prev_low
            results['F_R1'] = PP + 0.382 * diff
            results['F_R2'] = PP + 0.618 * diff
            results['F_R3'] = PP + 1.000 * diff
            results['F_S1'] = PP - 0.382 * diff
            results['F_S2'] = PP - 0.618 * diff
            results['F_S3'] = PP - 1.000 * diff
            # 删除第一行（无法计算）
            results = results.iloc[1:]
            return results
        except Exception as e:
            logger.error(f"计算 Pivot Points 失败: {e}", exc_info=True)
            return None










