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
from typing import Any, Callable, List, Optional, Set, Tuple, Union, Dict
from django.db import models # 确保导入 models
import pandas_ta as ta

# --- 忽略特定警告 ---
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*drop timezone information.*')
warnings.filterwarnings(action='ignore', category=FutureWarning, message=".*Passing 'suffixes' which cause duplicate columns.*")
pd.options.mode.chained_assignment = None

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

    # 辅助函数，用于根据指标名称和参数构建基础指标列名（不含时间级别后缀）
    # 这个函数主要用于在计算衍生特征时，根据已知参数和基本名称，构建出原始指标计算函数应该生成的列名
    # 请确保这个函数的逻辑与您的 calculate_* 函数实际返回的列名格式严格一致！
    def _build_indicator_base_name_for_lookup(self, base_name: str, params: Dict, stock_code: str) -> Optional[str]:
        """
        根据指标名称和参数字典构建基础指标列名（不含时间级别后缀），用于在DataFrame中查找列。
        这个函数需要与您的 calculate_* 函数实际返回的列名格式严格一致。
        如果无法构建，返回 None。
        """
        if base_name == 'RSI':
            # 示例: calculate_rsi(df, period=14) 返回名为 'RSI_14' 的 Series 或 DataFrame
            period = params.get('period', 14) # 使用默认值以防参数缺失
            return f"RSI_{period}"
        elif base_name in ['MACD', 'MACDh', 'MACDs']:
             # 示例: calculate_macd(df, period_fast=12, period_slow=26, signal_period=9) 返回 DataFrame 包含 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9' 列
            p_fast = params.get('period_fast', 12)
            p_slow = params.get('period_slow', 26)
            p_signal = params.get('signal_period', 9)
            return f"{base_name}_{p_fast}_{p_slow}_{p_signal}"
        elif base_name in ['K', 'D', 'J']:
             # 示例: calculate_kdj(df, period=9, signal_period=3, smooth_k_period=3) 返回 DataFrame 包含 'K_9_3_3', 'D_9_3_3', 'J_9_3_3' 列
            p_k = params.get('period', 9) # 注意 KDJ 参数命名可能需要调整以匹配您的 calculate_kdj
            p_d = params.get('signal_period', 3)
            p_j = params.get('smooth_k_period', 3)
            # 如果您的 calculate_kdj 参数键名不同，这里需要修改以匹配
            # 例如，如果 calculate_kdj 接收 kdj_period_k, kdj_period_d, kdj_period_j
            # p_k = params.get('kdj_period_k', 9) etc.
            return f"{base_name}_{p_k}_{p_d}_{p_j}"
        elif base_name in ['PDI', 'NDI', 'ADX']:
             # 示例: calculate_dmi(df, period=14) 返回 DataFrame 包含 'PDI_14', 'NDI_14', 'ADX_14' 列
             period = params.get('period', 14)
             return f"{base_name}_{period}"
        elif base_name in ['BBL', 'BBM', 'BBU', 'BBW', 'BBP']:
             # 示例: calculate_boll_bands_and_width(df, period=20, std_dev=2.0) 返回 DataFrame 包含 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBW_20_2.0', 'BBP_20_2.0' 列
             p = params.get('period', 20)
             std = params.get('std_dev', 2.0)
             return f"{base_name}_{p}_{std:.1f}" # 注意标准差的格式化
        elif base_name in ['CCI', 'MFI', 'ROC', 'ATR', 'HV', 'MOM', 'WILLR', 'VROC', 'AROC']:
            # 示例: CCI_14, MFI_14, ROC_12, ATR_14, HV_20, MOM_10, WILLR_14, VROC_10, AROC_10
            period = params.get('period', None)
            if period is not None:
                 return f"{base_name}_{period}"
            return base_name # 对于没有 period 参数但有基础列名的，如 ADL, OBV, VWAP
        elif base_name in ['EMA', 'SMA', 'AMT_MA', 'VOL_MA']:
             # 示例: EMA_5, SMA_20 etc.
             period = params.get('period', None)
             if period is not None:
                  return f"{base_name}_{period}"
             return None # 这些通常都有周期参数
        elif base_name in ['OBV', 'ADL']:
             # 示例: OBV, ADL - 这些通常没有参数在列名中
             return base_name
        elif base_name == 'VWAP':
             # 示例: VWAP, VWAP_D
             anchor = params.get('anchor', None)
             if anchor:
                 return f"VWAP_{anchor}"
             return "VWAP" # 默认无锚点VWAP列名
        elif base_name in ['KCL', 'KCM', 'KCU']:
             # 示例: KCL_20_10, KCM_20_10, KCU_20_10 (根据日志推测不包含乘数 2.0)
             ema_p = params.get('ema_period', 20)
             atr_p = params.get('atr_period', 10)
             # atr_m = params.get('atr_multiplier', 2.0) # 移除 atr_multiplier
             # atr_m_str = f"{atr_m:.1f}" # 移除 atr_multiplier 格式化

             # >>> 修改开始 >>>
             # 根据日志 KCL_20_10_5 和 JSON 约定 KCL_{ema_period}_{atr_period}，
             # 基础 KC 列名不包含 atr_multiplier 参数。
             return f"{base_name}_{ema_p}_{atr_p}"
             # <<< 修改结束 <<<

        # 如果是 Ichimoku 或 PivotPoints 的子列，可能需要更复杂的逻辑
        # 例如，如果 base_name 是 TENKAN，需要找到 Ichimoku 的参数来构建列名 TENKAN_9
        # 暂时不在这里处理 Ichimoku/PivotPoints 的子列查找，如果需要差分再添加
        # 例如，TENKAN_9, KIJUN_26, CHIKOU_26, SENKOU_A_9_26, SENKOU_B_52
        # PivotPoints: PP, S1, R1 等没有参数

        # >>> 新增调试日志 >>>
        logger.warning(f"[{stock_code}] 无法为基础指标 {base_name} 构建列名查找模式。参数: {params}")
        # <<< 新增调试日志 >>>
        return None # 无法构建列名，返回 None

# File 1: pasted_text_0.txt (Modified section within prepare_strategy_dataframe)

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, base_needed_bars: Optional[int] = None) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        根据策略 JSON 配置文件准备包含重采样基础数据和所有计算指标的 DataFrame。

        该函数执行以下步骤：
        1. 加载策略参数文件。
        2. 解析参数，识别所需的时间级别和要计算的指标及其参数。
        3. 确定最小时间级别和指标的最大回看期，估算需要获取的原始数据量。
        4. 并行获取所有所需时间级别的原始 OHLCV 数据。
        5. 对原始数据进行重采样和初步清洗，并给 OHLCV 列名添加时间周期后缀。
        6. 并行计算所有配置的指标，并给指标列名添加时间周期后缀。
        7. 将所有时间级别的 OHLCV 数据和计算出的指标数据合并到同一个 DataFrame 中。
        8. 计算基于合并后数据的衍生特征。
        9. 对最终的 DataFrame 进行缺失值填充。
        10. 返回最终的 DataFrame 和指标配置列表。

        Args:
            stock_code (str): 股票代码。
            params_file (str): 策略 JSON 配置文件的路径。
            base_needed_bars (Optional[int]): 如果提供，作为基础所需的最小时间级别数据条数，
                                               覆盖参数文件中的 lstm_window_size + buffer。

        Returns:
            Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]: 包含所有数据的 DataFrame 和指标配置列表，
                                                                如果准备失败则返回 None。
        """
        # 检查 pandas_ta 是否已加载
        if ta is None:
            print(f"[{stock_code}] Debug: pandas_ta 未加载。") # 调试输出
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None

        # 1. 加载 JSON 参数文件
        try:
            # 检查文件是否存在
            if not os.path.exists(params_file):
                print(f"[{stock_code}] Debug: 策略参数文件未找到: {params_file}") # 调试输出
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None
            # 打开并解析 JSON 文件
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
            # 添加参数加载成功的调试输出，可以部分打印参数
            print(f"[{stock_code}] Debug: 加载策略参数成功，部分参数: base_scoring={params.get('base_scoring', {})}, indicator_analysis_params={params.get('indicator_analysis_params', {})}, trend_following_params={params.get('trend_following_params', {})}") # 调试输出
        except Exception as e:
            # 记录加载或解析失败的错误
            print(f"[{stock_code}] Debug: 加载或解析参数文件失败: {e}") # 调试输出
            logger.error(f"[{stock_code}] 加载或解析参数文件 {params_file} 失败: {e}", exc_info=True)
            return None

        # 2. 识别需求：时间级别和全局指标最大回看期
        # 存储所有需要的时间级别，使用集合避免重复
        all_time_levels_needed: Set[str] = set()
        # 存储每个指标的计算配置 (名称, 函数, 参数, 适用的时间级别列表)
        indicator_configs: List[Dict[str, Any]] = []

        # 辅助函数：用于简化从参数中提取指标配置并添加到 indicator_configs 列表的过程
        def _add_indicator_config(
            name: str, # 指标名称，用于识别
            func: Callable, # 计算该指标的异步函数引用 (如 self.calculate_macd)
            param_block_key: str, # 参数块的键名，指示该指标的配置位于 JSON 的哪个部分 (如 'base_scoring')
            default_params: Dict, # 该指标计算函数自身的默认参数字典
            applicable_tfs: Union[str, List[str]], # 适用于该指标的时间级别，可以是单个字符串或字符串列表
            param_override_key: Optional[str] = None # 可选：在参数块内，用于覆盖默认参数的具体指标键名 (如 'rsi_period')
        ):
            # 从主 params 字典中获取指定的参数块
            param_block = params.get(param_block_key, {})
            # 根据 param_override_key 确定最终用于指标计算的参数字典来源
            # 如果指定了 param_override_key，则使用 param_block[param_override_key] 作为参数字典
            # 否则，使用 param_block 本身作为参数字典
            indi_specific_params_json = param_block.get(param_override_key, param_block) if param_override_key else param_block

            # 合并参数：以 default_params 为基础，使用 indi_specific_params_json 中的值覆盖
            final_calc_params = default_params.copy()
            # 遍历 JSON 中提供的参数，只覆盖 default_params 中存在的键，忽略 JSON 中的额外参数
            for k, v_json in indi_specific_params_json.items():
                # 检查 key 是否在 default_params 中，并且值类型是否匹配或可转换
                # 简单实现：只覆盖 default_params 中已有的键
                if k in final_calc_params:
                     final_calc_params[k] = v_json
                # 更严格的实现可能需要类型检查
                # if k in final_calc_params and isinstance(v_json, type(final_calc_params[k])):
                #     final_calc_params[k] = v_json
                # elif k in final_calc_params:
                #     logger.warning(f"[{stock_code}] 参数 {param_block_key}.{param_override_key or ''}.{k}: JSON值类型 {type(v_json)} 与默认类型 {type(final_calc_params[k])} 不匹配，忽略。")


            # 将适用的时间级别转换为列表，并添加到 all_time_levels_needed 集合中
            tfs = [applicable_tfs] if isinstance(applicable_tfs, str) else applicable_tfs
            all_time_levels_needed.update(tfs)

            # 将完整的指标配置信息添加到列表中
            indicator_configs.append({
                'name': name, # 指标名称
                'func': func, # 指标计算函数引用
                'params': final_calc_params, # 传递给计算函数的最终参数字典
                'timeframes': tfs, # 适用该指标的时间级别列表
                'param_block_key': param_block_key, # 参数块键名 (用于日志或调试)
                'param_override_key': param_override_key # 参数覆盖键名 (用于日志或调试)
            })

        # --- 从参数文件动态构建指标计算列表 ---
        # 获取基础评分相关的参数，包括时间级别列表
        bs_params = params.get('base_scoring', {})
        bs_timeframes = bs_params.get('timeframes', ['5', '15', '30', '60', 'D']) # 基础评分默认使用的时间级别
        # 将基础评分时间级别添加到总所需时间级别集合中
        all_time_levels_needed.update(bs_timeframes)

        # 定义常用指标的默认参数。这些默认值在 JSON 参数中未提供时使用。
        default_macd_p = {'period_fast': 12, 'period_slow': 26, 'signal_period': 9}
        default_rsi_p = {'period': 14}
        # KDJ 周期参数的命名可能因库而异，这里使用一个示例结构，需与 calculate_kdj 匹配
        default_kdj_p = {'period': 9, 'signal_period': 3, 'smooth_k_period': 3}
        # *** 注意：这里的 default_boll_p 周期默认值是 20, 2.0。
        default_boll_p = {'period': 20, 'std_dev': 2.0}
        default_cci_p = {'period': 14}
        default_mfi_p = {'period': 14}
        default_roc_p = {'period': 12}
        # *** 注意：这里的 default_dmi_p 周期默认值是 14。
        default_dmi_p = {'period': 14}
        default_sar_p = {'af_step': 0.02, 'max_af': 0.2}
        # STOCH 参数命名需与 calculate_stoch 匹配
        default_stoch_p = {'k_period': 14, 'd_period': 3, 'smooth_k_period': 3}
        default_atr_p = {'period': 14}
        default_hv_p = {'period': 20, 'annual_factor': 252} # 日线年化因子
        default_kc_p = {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0}
        default_mom_p = {'period': 10}
        default_willr_p = {'period': 14}
        default_sma_ema_p = {'period': 20} # 通用均线周期默认值
        default_ichimoku_p = {'tenkan_period': 9, 'kijun_period': 26, 'senkou_period': 52}


        # --- 注册基础评分指标的计算配置 ---
        # 遍历 base_scoring.score_indicators 列表中启用的指标键名
        for indi_key in bs_params.get('score_indicators', []):
            # 为每个启用的指标调用 _add_indicator_config 辅助函数
            if indi_key == 'macd': _add_indicator_config('MACD', self.calculate_macd, 'base_scoring', default_macd_p, bs_timeframes)
            elif indi_key == 'rsi': _add_indicator_config('RSI', self.calculate_rsi, 'base_scoring', default_rsi_p, bs_timeframes)
            elif indi_key == 'kdj':
                # KDJ 的周期参数在 JSON 中可能有特定的键名，这里从 bs_params 中提取并覆盖默认值
                kdj_calc_params = {
                    'period': bs_params.get('kdj_period_k', default_kdj_p['period']),
                    'signal_period': bs_params.get('kdj_period_d', default_kdj_p['signal_period']),
                    'smooth_k_period': bs_params.get('kdj_period_j', default_kdj_p['smooth_k_period'])
                }
                _add_indicator_config(
                    'KDJ', # 注册的配置名称是 'KDJ'
                    self.calculate_kdj,
                    'base_scoring',
                    kdj_calc_params, # 传递已经根据 JSON 配置好的参数字典
                    bs_timeframes
                    # param_override_key 不再需要
                )
            # *** 调试点：检查 BOLL 是否在 base_scoring.score_indicators 中并被注册 ***
            # *** 检查您的 JSON 参数文件 base_scoring.score_indicators 中是否包含 "boll" ***
            elif indi_key == 'boll':
                 # 注意：这里的默认周期是 20, 2.0。如果您的 JSON 中配置的是 15, 2.2，
                 # 且 base_scoring 块包含了 boll 的参数覆盖，例如 {"boll_period": 15, "boll_std_dev": 2.2}，
                 # 那么 _add_indicator_config 会正确合并这些参数，最终 default_boll_p 会被覆盖。
                 # 最终传递给 calculate_boll_bands_and_width 的将是 JSON 中的值。
                 # 之前警告中的 BBU_15_2.2_30 表明 JSON 中配置的周期可能是 15, 2.2。
                print(f"[{stock_code}] Debug: 注册 BOLL 计算配置，应用于时间框架: {bs_timeframes}") # 调试输出
                boll_calc_params = {
                     'period': bs_params.get('boll_period', default_boll_p['period']),
                     'std_dev': bs_params.get('boll_std_dev', default_boll_p['std_dev'])
                 }
                print(f"[{stock_code}] Debug: 注册 BOLL 计算配置，使用参数: {boll_calc_params} 应用于时间框架: {bs_timeframes}") # 调试输出
                # 使用获取到的参数注册计算配置
                _add_indicator_config(
                    'BOLL', # 注册的配置名称是 'BOLL'
                    self.calculate_boll_bands_and_width,
                    'base_scoring',
                    boll_calc_params, # 传递已经根据 JSON 配置好的参数字典
                    bs_timeframes
                    # param_override_key 不再需要，因为我们手动映射了参数
                )
            elif indi_key == 'cci': _add_indicator_config('CCI', self.calculate_cci, 'base_scoring', default_cci_p, bs_timeframes)
            elif indi_key == 'mfi': _add_indicator_config('MFI', self.calculate_mfi, 'base_scoring', default_mfi_p, bs_timeframes)
            elif indi_key == 'roc': _add_indicator_config('ROC', self.calculate_roc, 'base_scoring', default_roc_p, bs_timeframes)
            # *** 调试点：检查 DMI 是否在 base_scoring.score_indicators 中并被注册 ***
            # *** 检查您的 JSON 参数文件 base_scoring.score_indicators 中是否包含 "dmi" ***
            # *** 检查 base_scoring 块中是否有 dmi_period 覆盖默认值 14 ***
            # *** 之前警告中的 ADX_14_30 表明 JSON 中配置的 DMI 周期可能是 14。
            elif indi_key == 'dmi':
                print(f"[{stock_code}] Debug: 注册 DMI 计算配置，应用于时间框架: {bs_timeframes}") # 调试输出
                _add_indicator_config('DMI', self.calculate_dmi, 'base_scoring', default_dmi_p, bs_timeframes) # 注册的配置名称是 'DMI'
            elif indi_key == 'sar': _add_indicator_config('SAR', self.calculate_sar, 'base_scoring', default_sar_p, bs_timeframes)
            # EMA 和 SMA 通常不在 score_indicators 里，而是作为独立特征或趋势分析的一部分
            # 如果参数中明确要计算EMA/SMA作为评分指标 (这种情况较少，通常在 feature_engineering)
            elif indi_key == 'ema':
                 ema_p = bs_params.get('ema_period', default_sma_ema_p['period'])
                 _add_indicator_config(f'EMA_{ema_p}', self.calculate_ema, 'base_scoring', {'period': ema_p}, bs_timeframes, param_override_key='ema_params')
            elif indi_key == 'sma':
                 sma_p = bs_params.get('sma_period', default_sma_ema_p['period'])
                 _add_indicator_config(f'SMA_{sma_p}', self.calculate_sma, 'base_scoring', {'period': sma_p}, bs_timeframes, param_override_key='sma_params')

        # --- 注册成交量和 indicator_analysis 相关指标的计算配置 ---
        vc_params = params.get('volume_confirmation', {})
        ia_params = params.get('indicator_analysis_params', {})
        # 成交量/额分析通常在基础时间框架上进行，除非 volume_confirmation 中指定了特定的 'tf'
        vol_ana_tf = vc_params.get('tf', bs_timeframes) if vc_params.get('enabled', False) else bs_timeframes
        # 确保如果 ia_params 启用了某个计算，即使 vc_params 未启用，也应计算
        # 合并可能的不同来源的时间框架，并去重
        target_vol_ana_tfs = list(set([vol_ana_tf] + ia_params.get('timeframes', [])))
        all_time_levels_needed.update(target_vol_ana_tfs)


        # AMT_MA 计算
        if vc_params.get('enabled', False) or ia_params.get('calculate_amt_ma', False):
             _add_indicator_config('AMT_MA', self.calculate_amount_ma, 'volume_confirmation', {'period': vc_params.get('amount_ma_period', ia_params.get('amount_ma_period', 20))}, target_vol_ana_tfs, param_override_key='amount_ma_params')
        # CMF 计算
        if vc_params.get('enabled', False) or ia_params.get('calculate_cmf', False):
            _add_indicator_config('CMF', self.calculate_cmf, 'volume_confirmation', {'period': vc_params.get('cmf_period', ia_params.get('cmf_period', 20))}, target_vol_ana_tfs, param_override_key='cmf_params')
        # VOL_MA 计算
        if ia_params.get('calculate_vol_ma', False):
            _add_indicator_config('VOL_MA', self.calculate_vol_ma, 'indicator_analysis_params', {'period': ia_params.get('volume_ma_period', 20)}, target_vol_ana_tfs, param_override_key='volume_ma_params')

        # 其他分析指标 (来自 indicator_analysis_params)
        ia_timeframes = ia_params.get('timeframes', bs_timeframes)
        all_time_levels_needed.update(ia_timeframes)

        # STOCH 计算
        # *** 检查您的 JSON 参数文件 indicator_analysis_params 中是否包含 calculate_stoch: true ***
        # *** 检查 indicator_analysis_params 块中是否有 stoch_k, stoch_d, stoch_smooth_k 覆盖默认值 ***
        if ia_params.get('calculate_stoch', False):
            stoch_p = {
                'k_period': ia_params.get('stoch_k', default_stoch_p['k_period']),
                'd_period': ia_params.get('stoch_d', default_stoch_p['d_period']),
                'smooth_k_period': ia_params.get('stoch_smooth_k', default_stoch_p['smooth_k_period'])
            }
            _add_indicator_config('STOCH', self.calculate_stoch, 'indicator_analysis_params', stoch_p, ia_timeframes, param_override_key='stoch_params') # 注册的配置名称是 'STOCH'

        # VWAP 计算
        # *** 检查您的 JSON 参数文件 indicator_analysis_params 中是否包含 calculate_vwap: true ***
        # *** 检查 indicator_analysis_params 块中是否有 vwap_anchor 覆盖默认值 None ***
        if ia_params.get('calculate_vwap', False):
            vwap_p = {'anchor': ia_params.get('vwap_anchor', None)}
            _add_indicator_config('VWAP', self.calculate_vwap, 'indicator_analysis_params', vwap_p, ia_timeframes, param_override_key='vwap_params') # 注册的配置名称是 'VWAP'

        # ADL 计算
        if ia_params.get('calculate_adl', False):
             _add_indicator_config('ADL', self.calculate_adl, 'indicator_analysis_params', {}, ia_timeframes, param_override_key='adl_params') # ADL 通常无参数，注册的配置名称是 'ADL'

        # Ichimoku 计算
        if ia_params.get('calculate_ichimoku', False):
             ichimoku_p = {
                 'tenkan_period': ia_params.get('ichimoku_tenkan', default_ichimoku_p['tenkan_period']),
                 'kijun_period': ia_params.get('ichimoku_kijun', default_ichimoku_p['kijun_period']),
                 'senkou_period': ia_params.get('ichimoku_senkou', default_ichimoku_p['senkou_period'])
             }
             _add_indicator_config('Ichimoku', self.calculate_ichimoku, 'indicator_analysis_params', ichimoku_p, ia_timeframes, param_override_key='ichimoku_params') # 注册的配置名称是 'Ichimoku'

        # Pivot Points 计算 (通常基于日线计算，但代码中注册为 bs_timeframes，这里修正为只在 'D' 上计算)
        # *** 检查您的 JSON 参数文件 indicator_analysis_params 中是否包含 calculate_pivot_points: true ***
        if ia_params.get('calculate_pivot_points', False):
             # Pivot 通常基于日线计算，所以适用时间级别应为 ['D']
             _add_indicator_config('PivotPoints', self.calculate_pivot_points, 'indicator_analysis_params', {}, ['D'], param_override_key='pivot_params') # 注册的配置名称是 'PivotPoints'
             all_time_levels_needed.add('D') # 确保 'D' 被包含在所需时间级别中


        # --- 注册特征工程指标的计算配置 ---
        fe_params = params.get('feature_engineering_params', {})
        # 特征工程默认应用于基础时间框架，除非参数中指定了 apply_on_timeframes
        fe_timeframes = fe_params.get('apply_on_timeframes', bs_timeframes)
        all_time_levels_needed.update(fe_timeframes)

        # ATR 计算
        if fe_params.get('calculate_atr', False):
             _add_indicator_config('ATR', self.calculate_atr, 'feature_engineering_params', default_atr_p, fe_timeframes, param_override_key='atr_params') # 注册的配置名称是 'ATR'
        # 历史波动率 (HV) 计算
        if fe_params.get('calculate_hv', False):
             _add_indicator_config('HV', self.calculate_historical_volatility, 'feature_engineering_params', default_hv_p, fe_timeframes, param_override_key='hv_params') # 注册的配置名称是 'HV'
        # 肯特纳通道 (KC) 计算
        if fe_params.get('calculate_kc', False):
             _add_indicator_config('KC', self.calculate_keltner_channels, 'feature_engineering_params', default_kc_p, fe_timeframes, param_override_key='kc_params') # 注册的配置名称是 'KC'
        # 动量 (MOM) 计算
        if fe_params.get('calculate_mom', False):
             _add_indicator_config('MOM', self.calculate_mom, 'feature_engineering_params', default_mom_p, fe_timeframes, param_override_key='mom_params') # 注册的配置名称是 'MOM'
        # Williams %R (WILLR) 计算
        if fe_params.get('calculate_willr', False):
             _add_indicator_config('WILLR', self.calculate_willr, 'feature_engineering_params', default_willr_p, fe_timeframes, param_override_key='willr_params') # 注册的配置名称是 'WILLR'
        # 成交量变化率 (VROC) 计算
        if fe_params.get('calculate_vroc', False):
             _add_indicator_config('VROC', self.calculate_volume_roc, 'feature_engineering_params', default_roc_p, fe_timeframes, param_override_key='vroc_params') # 注册的配置名称是 'VROC'
        # 成交额变化率 (AROC) 计算
        if fe_params.get('calculate_aroc', False):
             _add_indicator_config('AROC', self.calculate_amount_roc, 'feature_engineering_params', default_roc_p, fe_timeframes, param_override_key='aroc_params') # 注册的配置名称是 'AROC'

        # 计算 EMA 和 SMA (如果参数中指定了周期列表)
        # 这些通常作为独立特征或用于计算与其他指标的关系
        for ma_type, ma_func in [('EMA', self.calculate_ema), ('SMA', self.calculate_sma)]:
            # 从 fe_params 中获取指定类型的均线周期列表，例如 "ema_periods": [5, 10, 20]
            ma_periods = fe_params.get(f'{ma_type.lower()}_periods', [])
            for p in ma_periods:
                if isinstance(p, int) and p > 0: # 确保周期是有效的正整数
                    # 为每个周期添加一个计算配置
                    _add_indicator_config(ma_type, ma_func, 'feature_engineering_params', {'period': p}, fe_timeframes, param_override_key=f'{ma_type.lower()}_params') # 注册的配置名称是 'EMA' 或 'SMA'


        # OBV 是基础的，通常都需要计算。确保只添加一次。
        # 检查 indicator_configs 列表中是否已经有 OBV 的配置
        if not any(conf['name'] == 'OBV' for conf in indicator_configs):
            # 将 OBV 添加到所有需要的时间级别上计算
            _add_indicator_config('OBV', self.calculate_obv, 'base_scoring', {}, list(all_time_levels_needed)) # 使用 list() 复制集合内容，注册的配置名称是 'OBV'


        # --- 调试点：确认需要的时间级别集合中是否包含目标 focus_tf (如 '30') ---
        # 获取 focus_timeframe (用于后续检查)
        focus_tf = params.get('trend_following_params', {}).get('focus_timeframe', '30')
        print(f"[{stock_code}] Debug: 策略关注的时间级别 (focus_tf): {focus_tf}") # 调试输出
        print(f"[{stock_code}] Debug: 所有策略所需时间级别集合: {sorted(list(all_time_levels_needed))}") # 调试输出


        # --- 确定最小时间级别 ---
        # 从 all_time_levels_needed 集合中找到分钟数最小的时间级别
        min_time_level = None
        min_tf_minutes = float('inf') # 初始化为无穷大
        if not all_time_levels_needed: # 如果没有任何时间级别被识别出来，记录错误并返回 None
            logger.error(f"[{stock_code}] 未能从参数文件中确定任何需要的时间级别。")
            return None

        # 遍历所有需要的时间级别，找到分钟数最小的那个
        for tf_str_loop in all_time_levels_needed:
            minutes = self._get_timeframe_in_minutes(tf_str_loop)
            # 排除无法转换为分钟的时间级别 None
            if minutes is not None and minutes < min_tf_minutes:
                 min_tf_minutes = minutes
                 min_time_level = tf_str_loop

        # 如果遍历完成后 min_time_level 仍然是 None (意味着所有时间级别都无法转换为分钟，或者集合为空)
        # 则尝试从 bs_timeframes 中取第一个作为最小级别，或者标记错误
        if min_time_level is None and bs_timeframes:
             min_time_level = bs_timeframes[0] # Fallback to the first base timeframe
             print(f"[{stock_code}] Debug: 无法精确确定最小时间级别 (按分钟)，回退使用基础时间级别列表的第一个: {min_time_level}") # 调试输出
        elif min_time_level is None:
            logger.error(f"[{stock_code}] 无法确定有效的最小时间级别从所需级别: {all_time_levels_needed}")
            return None

        logger.info(f"[{stock_code}] 策略所需时间级别: {sorted(list(all_time_levels_needed))}, 最小时间级别: {min_time_level} ({min_tf_minutes if min_tf_minutes != float('inf') else 'N/A'} 分钟)")


        # --- 动态计算 global_max_lookback ---
        # 估算所有指标计算所需的最大历史回看期，用于确定需要获取多少根 K 线数据
        global_max_lookback = 0
        for config in indicator_configs:
            current_max_period = 0
            # 检查所有可能的周期参数键名
            period_keys = ['period', 'period_fast', 'period_slow', 'signal_period', 'k_period', 'd_period', 'smooth_k_period',
                           'ema_period', 'atr_period', 'tenkan_period', 'kijun_period', 'senkou_period', 'atr_multiplier'] # 添加 atr_multiplier
            for p_key in period_keys:
                # 获取参数值，确保是数字且大于当前最大周期
                # 对于乘数等非周期参数，不用于计算回看期，跳过
                if p_key == 'atr_multiplier':
                    continue

                if p_key in config['params'] and isinstance(config['params'][p_key], (int, float)):
                    current_max_period = max(current_max_period, int(config['params'][p_key]))

            # 特殊处理组合周期，例如 MACD (慢周期 + 信号周期) 或 DMI/ADX (周期本身较大，ADX计算还需要平滑)
            # 这里采用一个更保守的估算，DMI/ADX 通常需要比周期长很多的数据
            if config['name'] == 'MACD':
                 p = config['params']
                 # 修正 MACD 回看期估算，通常只需慢周期长度 + 信号周期平滑长度
                 current_max_period = max(current_max_period, p.get('period_slow',0) + p.get('signal_period',0))
            if config['name'] == 'DMI' and 'period' in config['params']: # DMI/ADX
                # ADX 计算需要至少 2*period 根数据，外加 EMA 平滑，所以需要更长的回看期
                current_max_period = max(current_max_period, int(config['params']['period'] * 2.5 + 10)) # 稍微保守一些的估算
            if config['name'] == 'KC': # Keltner Channels 也依赖 EMA 和 ATR 周期
                 p = config['params']
                 current_max_period = max(current_max_period, p.get('ema_period', 0), p.get('atr_period', 0))


            global_max_lookback = max(global_max_lookback, current_max_period)

        # 添加一个固定的缓冲期，以应对计算起点、复权等问题
        global_max_lookback += 100
        logger.info(f"[{stock_code}] 动态计算的全局指标最大回看期 (含缓冲): {global_max_lookback}")

        # 3. 并行获取原始 OHLCV 数据
        ohlcv_tasks = {} # 存储异步数据获取任务
        # 确定基础所需数据条数，如果 base_needed_bars 未指定，则从参数文件获取 LSTM 窗口大小加上指标回看期和缓冲
        # 从参数中获取 LSTM 窗口大小，如果不存在则使用默认值 60
        lstm_window_size = params.get('lstm_training_config',{}).get('lstm_window_size', 60)
        effective_base_needed_bars = base_needed_bars if base_needed_bars is not None else \
                                     lstm_window_size + global_max_lookback + 500 # 额外加一个缓冲

        # 为每个所需时间级别创建一个数据获取任务
        for tf_fetch in all_time_levels_needed:
            # 根据目标时间级别和最小时间级别，估算需要获取的原始数据条数
            needed_bars_for_tf = self._calculate_needed_bars_for_tf(
                target_tf=tf_fetch, min_tf=min_time_level,
                base_needed_bars=effective_base_needed_bars, # 使用基础所需条数
                global_max_lookback=global_max_lookback
            )
            logger.info(f"[{stock_code}] 时间级别 {tf_fetch}: 基础({min_time_level})需(估算){effective_base_needed_bars}条, 指标需{global_max_lookback}条 -> 动态计算需获取 {needed_bars_for_tf} 条原始数据.")
            # 创建异步获取数据的任务
            ohlcv_tasks[tf_fetch] = self._get_ohlcv_data(stock_code, tf_fetch, needed_bars_for_tf)

        # 并行执行所有数据获取任务，并收集结果
        ohlcv_results = await asyncio.gather(*ohlcv_tasks.values())
        # 将结果按时间级别字典化
        raw_ohlcv_dfs = dict(zip(all_time_levels_needed, ohlcv_results))

        # *** 调试点：检查获取到的原始数据状态 ***
        # print(f"[{stock_code}] Debug: 原始 OHLCV 数据获取结果 (按时间级别):") # 调试输出
        for tf, df in raw_ohlcv_dfs.items():
            print(f"  - TF {tf}: {'None/Empty' if df is None or df.empty else f'Shape {df.shape}, Columns: {df.columns.tolist()}'}") # 调试输出

        # 4. 重采样和初步清洗
        resampled_ohlcv_dfs = {} # 存储重采样和清洗后的数据
        # 定义最小可用数据条数的硬性门槛 (例如，基础所需条数的 60%)
        min_usable_bars = math.ceil(effective_base_needed_bars * 0.6)

        # 遍历所有获取到的原始数据，进行重采样和清洗
        for tf_resample, raw_df in raw_ohlcv_dfs.items():
            # 检查原始数据是否有效
            if raw_df is None or raw_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 没有获取到原始数据，跳过重采样。")
                continue

            # 执行重采样和清洗（例如处理缺失 K 线、填充等）
            # min_periods=1 和 fill_method='ffill' 是示例参数
            resampled_df = self._resample_and_clean_dataframe(raw_df, tf_resample, min_periods=1, fill_method='ffill')

            # 检查重采样后的数据是否有效
            if resampled_df is None or resampled_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 重采样后数据为空，跳过。")
                continue

            # 对最小时间级别的数据量进行硬性检查
            if tf_resample == min_time_level and len(resampled_df) < min_usable_bars:
                 logger.error(f"[{stock_code}] 最小时间级别 {tf_resample} 重采样后数据量 {len(resampled_df)} 条，少于最低可用阈值 {min_usable_bars} 条。无法继续。")
                 return None # 数据量不足以支持后续分析，终止流程

            # 如果数据量显著少于全局最大回看期，记录警告
            if len(resampled_df) < global_max_lookback * 0.5:
                 logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 重采样后数据量 {len(resampled_df)} 条，显著少于全局指标最大回看期 {global_max_lookback} 条。计算的指标可能不可靠。")

            # *** 关键步骤：给基础 OHLCV 列名添加时间周期后缀 ***
            # 例如，将 'close' 列重命名为 'close_30'
            rename_map = {col: f"{col}_{tf_resample}" for col in ['open', 'high', 'low', 'close', 'volume', 'amount'] if col in resampled_df.columns}
            # 应用重命名，如果 rename_map 不为空则复制并重命名，否则只复制
            resampled_df_renamed = resampled_df.rename(columns=rename_map) if rename_map else resampled_df.copy()

            # *** 调试点：检查重采样并添加后缀后的列名 ***
            print(f"[{stock_code}] Debug: TF {tf_resample} 重采样并重命名后的列: {resampled_df_renamed.columns.tolist()[:20]}...") # 调试输出

            # 将处理后的 DataFrame 存储起来
            resampled_ohlcv_dfs[tf_resample] = resampled_df_renamed

        # 最终检查最小时间级别的数据是否存在且非空
        if min_time_level not in resampled_ohlcv_dfs or resampled_ohlcv_dfs[min_time_level].empty:
             logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 重采样后的数据不可用。终止。")
             return None # 最小时间级别数据缺失，终止流程

        # 使用最小时间级别的数据索引作为最终合并的基准索引
        base_index = resampled_ohlcv_dfs[min_time_level].index
        logger.info(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的重采样索引作为合并基准，数量: {len(base_index)}。")

        # 5. 计算所有配置的指标 (并行)
        indicator_calculation_tasks = [] # 存储异步指标计算任务

        # 定义一个异步辅助函数来计算单个指标
        async def _calculate_single_indicator_async(tf_calc: str, base_df_with_suffix: pd.DataFrame, config_item: Dict) -> Optional[Tuple[str, pd.DataFrame]]:
            """异步计算单个时间级别上的单个指标。"""
            print(f"[{stock_code}] Debug: 开始计算指标 {config_item['name']} for TF {tf_calc}...") # 调试输出
            # 检查输入的基础数据是否有效
            if base_df_with_suffix is None or base_df_with_suffix.empty:
                print(f"[{stock_code}] Debug: TF {tf_calc}: 基础OHLCV数据为空，无法计算指标 {config_item['name']}") # 调试输出
                return None

            # 复制 DataFrame，用于传递给指标计算函数。
            # 指标计算函数通常期望标准的列名 (high, low, close)，而不是带后缀的 (high_30, low_30, close_30)。
            df_for_ta = base_df_with_suffix.copy()

            # 创建一个映射，将带后缀的列名暂时重命名为标准的列名
            ohlcv_map_to_std = {
                f'open_{tf_calc}': 'open', f'high_{tf_calc}': 'high', f'low_{tf_calc}': 'low',
                f'close_{tf_calc}': 'close', f'volume_{tf_calc}': 'volume', f'amount_{tf_calc}': 'amount'
            }
            # 过滤掉 df_for_ta 中不存在的列，避免 KeyError
            actual_rename_map_to_std = {k: v for k, v in ohlcv_map_to_std.items() if k in df_for_ta.columns}
            # 应用临时重命名
            df_for_ta.rename(columns=actual_rename_map_to_std, inplace=True)

            # *** 调试点：检查用于指标计算的临时 DataFrame 的列名 ***
            # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 用于计算的临时 df_for_ta 列名: {df_for_ta.columns.tolist()}") # 调试输出

            # 动态确定指标函数需要的列 (high, low, close, volume, amount 等)
            # 这是一个简化版本，实际可能需要更复杂的参数签名检查或配置
            required_cols_for_func = set(['high', 'low', 'close']) # 默认需要 HLC
            if config_item['name'] in ['MFI', 'OBV', 'VWAP', 'CMF', 'VOL_MA', 'ADL']:
                required_cols_for_func.add('volume')
            if config_item['name'] in ['AMT_MA', 'AROC']:
                required_cols_for_func.add('amount')
            # Keltner Channels (KC) 需要高、低、收盘价以及 ATR，所以需要 ATR 列
            if config_item['name'] == 'KC':
                 # KC 计算函数 calculate_keltner_channels 需要 ATR_period 列
                 # 这里应该检查的是原始的 ATR 列是否能被计算，而不是计算前的 df_for_ta 里是否有 ATR 列
                 # 因为 ATR 应该作为独立指标计算，KC 直接使用 ATR 计算结果
                 # 所以如果 calculate_keltner_channels 依赖 ATR_period 列作为输入，
                 # 需要确保 ATR 在 KC 之前计算完成并存在于 df_for_ta 中 (已重命名回标准名)
                 # 但是 calculate_keltner_channels 接收 close, high, low 和参数 ema_period, atr_period, atr_multiplier
                 # 并在内部计算 ATR。所以这里不需要检查 ATR_period 列。
                 pass # 无需额外添加 required_cols_for_func for KC

            # 检查 df_for_ta 是否包含计算该指标所需的所有列
            if not all(col in df_for_ta.columns for col in required_cols_for_func):
                missing_cols_str = ", ".join(list(required_cols_for_func - set(df_for_ta.columns)))
                print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 缺少必要列 ({missing_cols_str})，跳过计算。") # 调试输出
                logger.debug(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} 时，df_for_ta 缺少必要列 ({missing_cols_str})。可用: {df_for_ta.columns.tolist()}")
                return None # 缺少必要列，无法计算

            try:
                # 获取为该指标准备好的参数副本，传递给计算函数
                func_params_to_pass = config_item['params'].copy()
                # *** 调试点：打印传递给计算函数的参数 ***
                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 传递的计算参数: {func_params_to_pass}") # 调试输出

                # 调用具体的指标计算函数 (这些函数使用标准的列名，如 'close')
                # calculate_dmi 和 calculate_boll_bands_and_width 就是在这里被调用的
                indicator_result_df = await config_item['func'](df_for_ta, **func_params_to_pass)

                # *** 调试点：检查指标计算函数的原始返回结果 ***
                if indicator_result_df is None:
                     print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算结果为 None。") # 调试输出
                     logger.debug(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为 None。")
                     return None
                if indicator_result_df.empty:
                     print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算结果为 Empty DataFrame。") # 调试输出
                     logger.debug(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为空。")
                     return None

                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算原始结果列名: {indicator_result_df.columns.tolist()}") # 调试输出
                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算原始结果 Shape: {indicator_result_df.shape}") # 调试输出


                # 确保返回的是 DataFrame 类型，如果不是，尝试转换为 DataFrame
                if not isinstance(indicator_result_df, pd.DataFrame):
                    # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算函数未返回DataFrame (返回类型: {type(indicator_result_df)})。尝试转换。") # 调试输出
                    logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算函数未返回DataFrame (返回类型: {type(indicator_result_df)})。尝试转换。")
                    if isinstance(indicator_result_df, pd.Series):
                        # 使用指标名称作为列名，如果 Series 没有 name
                        series_name = indicator_result_df.name if indicator_result_df.name else config_item['name']
                        # 检查 Series name 是否包含参数，如果只返回基础名称，需要根据 config_item 的 params 加上参数
                        # 例如， calculate_rsi 可能返回 Series，name 是 RSI_14
                        # calculate_macd 返回 dataframe，列是 MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
                        # 为了确保列名带参数，这里可以根据 config_item['name'] 和 config_item['params'] 重新构建列名
                        # 但是 _build_indicator_base_name 函数是用于查找已存在的列名，不是用于生成 Series 的列名
                        # 假设计算函数返回的 Series/DataFrame 列名已经包含参数
                        indicator_result_df = indicator_result_df.to_frame(name=series_name)
                        print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 转换为DataFrame后列名: {indicator_result_df.columns.tolist()}") # 调试输出
                    else:
                        print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 返回类型 {type(indicator_result_df)} 无法处理。") # 调试输出
                        logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 返回类型 {type(indicator_result_df)} 无法处理。")
                        return None # 无法处理的返回类型

                # *** 关键步骤：给计算出的指标列名添加时间周期后缀 ***
                # 例如，将 calculate_dmi 返回的 'ADX_14' 重命名为 'ADX_14_30'
                # lambda 函数 x: f"{x}_{tf_calc}" 对 DataFrame 的所有列名应用后缀
                result_renamed_df = indicator_result_df.rename(columns=lambda x: f"{x}_{tf_calc}")

                # *** 调试点：检查添加后缀后的指标 DataFrame 列名 ***
                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 添加后缀后的列名: {result_renamed_df.columns.tolist()}") # 调试输出

                # 返回时间级别和带有后缀的结果 DataFrame
                return (tf_calc, result_renamed_df)
            except Exception as e_calc:
                # 捕获指标计算过程中的异常并记录错误日志
                print(f"[{stock_code}] Debug: TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}") # 调试输出
                logger.error(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}", exc_info=True)
                return None # 计算出错，返回 None

        # 为每个指标配置和适用的时间级别创建一个计算任务
        for config_item_loop in indicator_configs:
            for tf_conf in config_item_loop['timeframes']:
                # 确保该时间级别的重采样数据已经准备好
                if tf_conf in resampled_ohlcv_dfs:
                    base_ohlcv_df_for_tf_loop = resampled_ohlcv_dfs[tf_conf]
                    # 创建异步计算任务并添加到列表中
                    indicator_calculation_tasks.append(
                        _calculate_single_indicator_async(tf_conf, base_ohlcv_df_for_tf_loop, config_item_loop)
                    )
                else:
                    print(f"[{stock_code}] Debug: 时间框架 {tf_conf} 的基础数据未找到 ({config_item_loop['name']})，无法创建计算任务。") # 调试输出
                    logger.warning(f"[{stock_code}] 时间框架 {tf_conf} 在 resampled_ohlcv_dfs 中未找到，无法为指标 {config_item_loop['name']} 创建计算任务。")

        # 并行执行所有指标计算任务
        # return_exceptions=True 确保即使某个任务失败，也不会中断整个 gather，而是返回异常对象
        calculated_results_tuples = await asyncio.gather(*indicator_calculation_tasks, return_exceptions=True)

        # 按时间级别对计算结果进行分组存储
        calculated_indicators_by_tf = defaultdict(list) # { 'tf' : [indi_df1, indi_df2, ...] }

        # *** 调试点：检查所有指标计算任务的结果 ***
        # print(f"[{stock_code}] Debug: 所有指标计算任务原始结果 tuple 列表 (总数: {len(calculated_results_tuples)}):") # 调试输出
        # 打印部分结果摘要，避免刷屏
        # print(f"[{stock_code}] Debug: 部分计算结果摘要: {[r for r in calculated_results_tuples[:10]]}...")

        for res_tuple_item in calculated_results_tuples:
            # 检查任务结果是否成功 (不是异常对象，且是预期的元组格式)
            if isinstance(res_tuple_item, tuple) and len(res_tuple_item) == 2:
                tf_res, indi_df_res = res_tuple_item
                # 检查返回的 DataFrame 是否有效
                if indi_df_res is not None and not indi_df_res.empty:
                    # 将结果 DataFrame 添加到对应时间级别列表中
                    calculated_indicators_by_tf[tf_res].append(indi_df_res)
                    # *** 调试点：记录成功添加到分组列表的指标 ***
                    # print(f"[{stock_code}] Debug: TF {tf_res}: 成功添加指标 DataFrame 到分组列表，列: {indi_df_res.columns.tolist()}") # 调试输出 (可能刷屏)
                else:
                    # 记录计算结果为空的情况 - 已经在 _calculate_single_indicator_async 中记录了
                    pass
            elif isinstance(res_tuple_item, Exception):
                # 如果任务返回的是异常，记录错误 - 已经在 _calculate_single_indicator_async 中记录了
                pass


        # --- 后处理 OBV_MA ---
        # OBV_MA 需要在 OBV 计算完成后才能计算
        # 从参数中获取 OBV_MA 的周期
        obv_ma_period_json = vc_params.get('obv_ma_period', ia_params.get('obv_ma_period', 10)) # 尝试从 volume_confirmation 或 indicator_analysis_params 获取

        # 检查是否需要计算 OBV_MA
        if vc_params.get('enabled', False) or ia_params.get('calculate_obv_ma', False): # 如果成交量确认或指标分析中启用了 OBV_MA 计算
            print(f"[{stock_code}] Debug: 尝试计算 OBV_MA (周期 {obv_ma_period_json})...") # 调试输出
            # 遍历所有计算了指标的时间级别
            for tf_obv_ma, df_list_obv_ma in calculated_indicators_by_tf.items():
                # 在该时间级别的指标列表中查找 OBV 列所在的 DataFrame
                # 期望的 OBV 列名是 'OBV_{tf_obv_ma}'
                obv_col_name = f'OBV_{tf_obv_ma}' # OBV 列名

                # 查找包含 OBV 列的 DataFrame
                obv_df_with_suffix = None
                for df_item in df_list_obv_ma:
                    if obv_col_name in df_item.columns:
                        obv_df_with_suffix = df_item
                        break

                # 如果找到了 OBV 数据
                if obv_df_with_suffix is not None:
                    try:
                        # 计算 OBV 的移动平均
                        obv_series = obv_df_with_suffix[obv_col_name].rolling(window=obv_ma_period_json, min_periods=max(1, int(obv_ma_period_json*0.5))).mean()
                        # 将计算结果转换为 DataFrame 并添加带有周期和时间框架后缀的列名
                        # 根据 JSON 衍生特征命名约定，OBV_MA 列名是 OBV_MA_{period} (在添加时间级别后缀前)
                        obv_ma_df_res = pd.DataFrame({f'OBV_MA_{obv_ma_period_json}_{tf_obv_ma}': obv_series}) # 添加时间级别后缀
                        # 将 OBV_MA 的结果添加到该时间级别的指标列表中，以便后续合并
                        calculated_indicators_by_tf[tf_obv_ma].append(obv_ma_df_res)
                        logger.debug(f"[{stock_code}] TF {tf_obv_ma}: OBV_MA_{obv_ma_period_json} 计算完成。")
                        print(f"[{stock_code}] Debug: TF {tf_obv_ma}: OBV_MA_{obv_ma_period_json} 计算并添加到分组列表。列名: {obv_ma_df_res.columns.tolist()}") # 调试输出
                    except Exception as e_obvma:
                        # 记录 OBV_MA 计算出错的日志
                        logger.error(f"[{stock_code}] TF {tf_obv_ma}: 计算 OBV_MA_{obv_ma_period_json} 出错: {e_obvma}", exc_info=True)
                        print(f"[{stock_code}] Debug: TF {tf_obv_ma}: 计算 OBV_MA_{obv_ma_period_json} 出错: {e_obvma}") # 调试输出
                else:
                    # 如果需要计算 OBV_MA 但没有找到 OBV 数据，记录警告
                    logger.warning(f"[{stock_code}] TF {tf_obv_ma}: 需要计算 OBV_MA_{obv_ma_period_json} 但没有找到 {obv_col_name} 列。")
                    print(f"[{stock_code}] Debug: TF {tf_obv_ma}: 需要计算 OBV_MA_{obv_ma_period_json} 但没有找到 {obv_col_name} 列。") # 调试输出


        # 6. 合并所有 DataFrame
        merged_indicators_by_tf = {}
        for tf_merge_indi, df_list_merge_indi in calculated_indicators_by_tf.items():
            if df_list_merge_indi:
                base_df_merge = resampled_ohlcv_dfs.get(tf_merge_indi)
                if base_df_merge is not None:
                    # 从基础OHLCV开始合并
                    merged_tf_df_res = base_df_merge.copy()
                    # *** 调试点：打印基础合并 DataFrame 的初始列 ***
                    # print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 合并起点 DataFrame 初始列 (来自重采样OHLCV): {merged_tf_df_res.columns.tolist()[:20]}...") # 调试输出

                    for indi_df_to_merge in df_list_merge_indi:
                        # 检查索引是否一致，如果不同，需要reindex (理论上应该一致)
                        if not merged_tf_df_res.index.equals(indi_df_to_merge.index):
                            logger.warning(f"[{stock_code}] TF {tf_merge_indi}: 指标 DataFrame (cols: {indi_df_to_merge.columns.tolist()}) 与基础数据索引不一致，尝试reindex。")
                            print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 指标 DataFrame (cols: {indi_df_to_merge.columns.tolist()}) 与基础数据索引不一致，尝试reindex。") # 调试输出
                            indi_df_to_merge = indi_df_to_merge.reindex(merged_tf_df_res.index) # 简单reindex，可能引入NaN

                        # *** 调试点：打印即将合并的指标 DataFrame 的列 ***
                        # print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 正在合并指标 DataFrame，列: {indi_df_to_merge.columns.tolist()}") # 调试输出

                        # 使用 pd.merge 按索引合并
                        # suffixes 参数用于处理列名冲突，理论上因为我们已经加了 _tf 后缀，不会有 OHLCV 列与指标列冲突
                        # 但如果不同指标计算函数内部有相同的临时列名且都返回了，suffixes 可以帮助区分
                        # 这里的 suffixes 只是为了防止极端情况，并且不会影响我们期望的带 _tf 后缀的列名
                        merged_tf_df_res = pd.merge(merged_tf_df_res, indi_df_to_merge, left_index=True, right_index=True, how='left', suffixes=('', '_indicator')) # 使用更通用的后缀

                        # 检查是否存在带 _indicator 后缀的列，并考虑移除或警告（如果这不是期望的）
                        # 例如， unexpected_cols = [col for col in merged_tf_df_res.columns if col.endswith('_indicator')]

                        # *** 调试点：打印合并后的 DataFrame 列数 (可选，如果列太多可能刷屏) ***
                        # print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 合并后的 DataFrame 当前列数: {len(merged_tf_df_res.columns)}") # 调试输出

                    merged_indicators_by_tf[tf_merge_indi] = merged_tf_df_res
                    # print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 指标合并完成，最终列数: {len(merged_indicators_by_tf[tf_merge_indi].columns)}") # 调试输出
                else:
                    logger.warning(f"[{stock_code}] TF {tf_merge_indi}: 基础重采样数据丢失，无法合并指标。")
                    print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 基础重采样数据丢失，无法合并指标。") # 调试输出
            elif tf_merge_indi in resampled_ohlcv_dfs: # 如果没有计算出任何指标，但有基础数据
                # 确保即使没有指标，该时间级别的数据也存在于 merged_indicators_by_tf 中
                merged_indicators_by_tf[tf_merge_indi] = resampled_ohlcv_dfs[tf_merge_indi]
                print(f"[{stock_code}] Debug: TF {tf_merge_indi}: 没有计算的指标，仅保留基础OHLCV数据。列数: {len(merged_indicators_by_tf[tf_merge_indi].columns)}") # 调试输出


        if not merged_indicators_by_tf or min_time_level not in merged_indicators_by_tf:
            logger.error(f"[{stock_code}] 没有可合并的数据，或最小时间级别 {min_time_level} 的数据丢失。")
            print(f"[{stock_code}] Debug: 没有可合并的数据，或最小时间级别 {min_time_level} 的数据丢失。") # 调试输出
            return None

        # 以最小时间级别的数据为基础，合并其他时间级别的数据
        final_merged_df = merged_indicators_by_tf[min_time_level].copy()

        # *** 调试点：打印最终合并起点 (最小时间级别) 的列 ***
        # print(f"[{stock_code}] Debug: 最终合并起点 (最小时间级别 {min_time_level}) 列数: {len(final_merged_df.columns)}. 部分列: {final_merged_df.columns.tolist()[:50]}...") # 调试输出

        # 对 all_time_levels_needed 进行排序，确保合并顺序一致性 (从小到大)
        sorted_time_levels_for_merge = sorted(list(all_time_levels_needed), key=lambda x: self._get_timeframe_in_minutes(x) or float('inf'))

        for tf_final_merge in sorted_time_levels_for_merge:
            # 跳过最小时间级别，因为它已经是合并的起点了
            if tf_final_merge == min_time_level:
                continue

            if tf_final_merge in merged_indicators_by_tf:
                # 获取要合并的时间级别数据，并 reindex 到最小时间级别的索引上，使用 ffill 填充缺失值
                df_to_merge_final = merged_indicators_by_tf[tf_final_merge].reindex(final_merged_df.index, method='ffill')

                # *** 调试点：打印即将合并的其他时间级别的数据列 ***
                # print(f"[{stock_code}] Debug: 最终合并: 正在合并 TF {tf_final_merge} 的数据，Shape: {df_to_merge_final.shape}, 列: {df_to_merge_final.columns.tolist()[:20]}...") # 调试输出

                # 执行合并操作
                # suffixes 参数用于处理可能出现的同名列冲突，虽然我们期望列名已经通过 _tf 后缀区分，
                # 但如果不同时间级别有相同的基础 OHLCV 列名（如 'open_5', 'open_15'），merge 默认会加后缀
                # 这里保留 suffixes 是为了显式处理，或者可以依赖于我们已经加好的 _tf 后缀
                # 使用更明确的后缀避免冲突
                final_merged_df = pd.merge(final_merged_df, df_to_merge_final, left_index=True, right_index=True, how='left', suffixes=('', f'_{tf_final_merge}_merge'))

                # *** 调试点：打印合并 TF {tf_final_merge} 后，检查是否有意外的列，例如带merge后缀的
                # unexpected_merge_cols = [col for col in final_merged_df.columns if col.endswith(f'_{tf_final_merge}_merge')]
                # if unexpected_merge_cols:
                #      print(f"[{stock_code}] Debug: 合并 TF {tf_final_merge} 后发现意外合并后缀列: {unexpected_merge_cols}") # 调试输出


                # *** 调试点：打印合并 TF {tf_final_merge} 后的最终 DataFrame 列数 ***
                # print(f"[{stock_code}] Debug: 最终合并: 合并 TF {tf_final_merge} 后，总列数: {len(final_merged_df.columns)}") # 调试输出

            else:
                logger.warning(f"[{stock_code}] 时间框架 {tf_final_merge} 的合并数据在 merged_indicators_by_tf 中未找到。")
                print(f"[{stock_code}] Debug: 时间框架 {tf_final_merge} 的合并数据在 merged_indicators_by_tf 中未找到。") # 调试输出


        # --- 7. 计算基于合并后数据的衍生特征 ---
        logger.info(f"[{stock_code}] 开始计算衍生特征...")
        print(f"[{stock_code}] Debug: 计算衍生特征前，DataFrame 形状: {final_merged_df.shape}, 列数: {len(final_merged_df.columns)}") # 调试输出
        # 打印一部分列名，确认基础列和指标列是否存在
        print(f"[{stock_code}] Debug: 计算衍生特征前，部分列: {final_merged_df.columns.tolist()[:50]}...") # 调试输出

        # >>> 新增调试输出：打印 indicator_configs 的内容 <<<
        print(f"[{stock_code}] Debug: Contents of indicator_configs before derivative feature calculation:") # Debug output
        if indicator_configs: # 检查列表是否为空
             for idx, cfg in enumerate(indicator_configs): # Debug output
                 # 只打印前几个参数，避免刷屏
                 params_summary = {k: cfg['params'][k] for k in list(cfg['params'].keys())[:min(len(cfg['params']), 5)]} if cfg['params'] else {} # 确保params非空且至少有5个键
                 print(f"  [{idx}] Name: {cfg.get('name')}, Timeframes: {cfg.get('timeframes')}, Params (partial): {params_summary}...") # Debug output
        else:
            print(f"  indicator_configs 列表为空。") # Debug output
        print("-" * 30) # Debug output
        # <<< 新增调试输出结束 >>>


        try:
            # 价格与均线的关系 (EMA, SMA)
            # ... (这部分代码保持不变)
            for ma_type_deriv in ['EMA', 'SMA']:
                # 获取用于计算关系的均线周期列表，优先使用 *_for_relation，否则使用 *_periods
                ma_periods_deriv = fe_params.get(f'{ma_type_deriv.lower()}_periods_for_relation', fe_params.get(f'{ma_type_deriv.lower()}_periods', []))
                for tf_str_deriv in all_time_levels_needed:
                    # 构造收盘价列名
                    close_col_tf_deriv = f'close_{tf_str_deriv}'
                    # 如果收盘价列不存在，跳过该时间级别的均线关系计算
                    if close_col_tf_deriv not in final_merged_df.columns:
                        print(f"[{stock_code}] Debug: TF {tf_str_deriv}: 缺少收盘价列 {close_col_tf_deriv}，跳过 {ma_type_deriv} 关系计算。") # 调试输出
                        continue

                    for p_deriv in ma_periods_deriv:
                        # 构造均线列名，这里假设均线列名格式是 MA_Period_Timeframe
                        ma_col_tf_deriv = f'{ma_type_deriv}_{p_deriv}_{tf_str_deriv}'
                        # 如果均线列存在，则计算价格与均线的关系
                        if ma_col_tf_deriv in final_merged_df.columns:
                            # 计算价格与均线的比率
                            final_merged_df[f'CLOSE_{ma_type_deriv}_RATIO_{p_deriv}_{tf_str_deriv}'] = final_merged_df[close_col_tf_deriv] / final_merged_df[ma_col_tf_deriv]
                            # 计算价格与均线的归一化差值 (收盘价 - 均线) / 均线
                            # 避免除以零，对均线列进行小数值检查再相除
                            denominator_ndiff = final_merged_df[ma_col_tf_deriv]
                            final_merged_df[f'CLOSE_{ma_type_deriv}_NDIFF_{p_deriv}_{tf_str_deriv}'] = np.where(
                                 np.abs(denominator_ndiff) > 1e-9, # 检查均线值是否接近零
                                 (final_merged_df[close_col_tf_deriv] - denominator_ndiff) / denominator_ndiff,
                                 0 # 如果均线接近零，归一化差值设为 0
                            )
                        else:
                            # 记录均线列缺失
                             print(f"[{stock_code}] Debug: TF {tf_str_deriv}: 缺少均线列 {ma_col_tf_deriv}，跳过关系计算。") # 调试输出


            # 指标的变化率/差分 (RSI, MACDh, MFI, CMF, ADX 等)
            # 注意这里的 base_name 应该对应 calculate_* 函数返回的列名，
            # 而 calc_config_name 用于查找注册时的配置 name
            indicators_to_diff_config = fe_params.get('indicators_for_difference', [
                 {'base_name': 'RSI', 'calc_config_name': 'RSI'},
                 {'base_name': 'MACD', 'calc_config_name': 'MACD'}, # MACD 快慢线
                 {'base_name': 'MACDh', 'calc_config_name': 'MACD'}, # MACD 柱状图
                 {'base_name': 'MACDs', 'calc_config_name': 'MACD'}, # MACD 信号线
                 {'base_name': 'MFI', 'calc_config_name': 'MFI'},
                 {'base_name': 'CCI', 'calc_config_name': 'CCI'},
                 {'base_name': 'ADX', 'calc_config_name': 'DMI'}, # ADX 是 DMI 计算的输出
                 {'base_name': 'PDI', 'calc_config_name': 'DMI'}, # PDI 是 DMI 计算的输出
                 {'base_name': 'NDI', 'calc_config_name': 'DMI'}, # NDI 是 DMI 计算的输出
                 {'base_name': 'K', 'calc_config_name': 'KDJ'}, # K 是 KDJ 计算的输出
                 {'base_name': 'D', 'calc_config_name': 'KDJ'}, # D 是 KDJ 计算的输出
                 {'base_name': 'J', 'calc_config_name': 'KDJ'}, # J 是 KDJ 计算的输出
                 {'base_name': 'BBW', 'calc_config_name': 'BOLL'}, # BBW 是 BOLL 计算的输出
                 {'base_name': 'BBP', 'calc_config_name': 'BOLL'}, # BBP 是 BOLL 计算的输出
                 {'base_name': 'BBU', 'calc_config_name': 'BOLL'}, # 上轨也可以计算差分
                 {'base_name': 'BBL', 'calc_config_name': 'BOLL'}, # 下轨也可以计算差分
                 {'base_name': 'BBM', 'calc_config_name': 'BOLL'}, # 中轨也可以计算差分
                 {'base_name': 'ATR', 'calc_config_name': 'ATR'},
                 {'base_name': 'HV', 'calc_config_name': 'HV'},
                 {'base_name': 'MOM', 'calc_config_name': 'MOM'},
                 {'base_name': 'WILLR', 'calc_config_name': 'WILLR'},
                 {'base_name': 'VROC', 'calc_config_name': 'VROC'},
                 {'base_name': 'AROC', 'calc_config_name': 'AROC'},
                 {'base_name': 'STOCHk', 'calc_config_name': 'STOCH'}, # STOCH 输出名
                 {'base_name': 'STOCHd', 'calc_config_name': 'STOCH'}  # STOCH 输出名
                 # 添加其他需要计算差分的指标配置
            ])
            diff_periods = fe_params.get('difference_periods', [1, 2]) # 计算1阶和2阶差分

            # 遍历需要计算差分的指标配置
            for indi_diff_conf in indicators_to_diff_config:
                base_name = indi_diff_conf.get('base_name') # 例如 'RSI', 'MACDh', 'K', 'ADX'
                # 增加 'calc_config_name' 字段，用于查找注册的计算配置，例如 'RSI', 'MACD', 'KDJ', 'DMI'
                # 如果未指定 calc_config_name，尝试使用 base_name 查找（用于如 RSI, ATR 等计算函数只生成一个同名主列的情况）
                calc_config_name = indi_diff_conf.get('calc_config_name', base_name)

                if not base_name:
                    logger.warning(f"[{stock_code}] 差分配置项缺少 'base_name'，跳过。配置: {indi_diff_conf}")
                    print(f"[{stock_code}] Debug: 差分配置项缺少 'base_name'，跳过。配置: {indi_diff_conf}")
                    continue

                # 查找计算该 base_name 对应的原始指标时使用的实际参数
                actual_params = None
                registered_timeframes = []
                # 修改查找逻辑：根据 calc_config_name 查找注册的配置
                # >>> 修改开始：改进查找逻辑，并增加调试输出 <<<
                found_config = None
                for registered_config in indicator_configs:
                    # 检查 name 是否存在并匹配
                    if registered_config.get('name') == calc_config_name:
                         found_config = registered_config
                         break # 找到匹配的计算配置

                if found_config:
                     actual_params = found_config['params']
                     registered_timeframes = found_config['timeframes'] # 获取注册时适用的时间级别
                     # 增加调试输出：确认找到了配置
                     print(f"[{stock_code}] Debug: Found indicator config for calc_config_name '{calc_config_name}' (for derivative base_name '{base_name}').")
                else:
                     # 如果没有找到匹配的注册配置，记录警告并跳过该指标的差分计算
                     # 修改警告信息，使其包含实际查找的 calc_config_name
                     logger.warning(f"[{stock_code}] 未找到指标计算配置 '{calc_config_name}' (用于衍生特征 '{base_name}')，无法确定实际参数。跳过。配置: {indi_diff_conf}")
                     print(f"[{stock_code}] Debug: 未找到指标计算配置 '{calc_config_name}' (用于衍生特征 '{base_name}')，无法确定实际参数。跳过。")
                     continue # 跳过当前指标的差分计算
                # >>> 修改结束 <<<


                # 遍历所有需要计算差分的时间级别
                # 默认对所有需要的时间级别计算差分，除非 fe_params 指定了 apply_on_timeframes
                # 确保只在原始指标计算过的时间级别上计算差分
                diff_timeframes_potential = fe_params.get('apply_on_timeframes', list(all_time_levels_needed))
                # 过滤，只保留在注册配置中实际计算过的时间框架
                diff_timeframes_for_this_indi = [tf for tf in diff_timeframes_potential if tf in registered_timeframes]

                # 增加调试输出：显示将计算差分的时间框架
                # print(f"[{stock_code}] Debug: Derivative '{base_name}' (config '{calc_config_name}') will be calculated on timeframes: {diff_timeframes_for_this_indi}")


                for tf_str_diff in diff_timeframes_for_this_indi:
                    # >>> 调用 _build_indicator_base_name_for_lookup 构建基础列名 <<<
                    # 构建不含时间级别的原始指标列名（带参数），使用查找到的实际参数
                    # 注意：这里传递的 base_name 是 'K', 'MACDh', 'ADX' 等，而不是 'KDJ', 'MACD', 'DMI'
                    indi_col_name_base = self._build_indicator_base_name_for_lookup(base_name, actual_params, stock_code)

                    if indi_col_name_base is None:
                        # _build_indicator_base_name_for_lookup 失败的情况，应该不太常见如果 base_name 配置正确
                        logger.warning(f"[{stock_code}] TF {tf_str_diff}: 调用 _build_indicator_base_name_for_lookup({base_name}, {actual_params}) 无法构建基础列名进行差分计算。跳过。")
                        print(f"[{stock_code}] Debug: TF {tf_str_diff}: 调用 _build_indicator_base_name_for_lookup({base_name}, {actual_params}) 无法构建基础列名进行差分计算。跳过。")
                        continue # 无法构建基础列名，跳过该时间级别和指标的差分计算


                    # 构造带有时间周期后缀的完整原始指标列名 (用于查找)
                    source_col_name = f"{indi_col_name_base}_{tf_str_diff}"

                    # >>> 新增调试输出：检查原始指标列是否存在 <<<
                    print(f"[{stock_code}] Debug: TF {tf_str_diff}: Attempting to calculate DIFF for '{base_name}'. Source column lookup name: '{source_col_name}'. Exists in DataFrame: {source_col_name in final_merged_df.columns}") # 增加调试输出
                    # <<< 新增调试输出结束 >>>

                    if source_col_name in final_merged_df.columns:
                        # 如果原始指标列存在，则计算其差分
                        for diff_p in diff_periods:
                            # 构造差分结果列名，遵循 JSON 衍生特征命名规范
                            # 格式: {original_indicator_name_with_params}_DIFF{diff_period}_{timeframe}
                            target_col_name = f"{indi_col_name_base}_DIFF{diff_p}_{tf_str_diff}"
                            # 计算差分 (当前值 - N周期前的值)
                            final_merged_df[target_col_name] = final_merged_df[source_col_name].diff(diff_p)
                    else:
                         # 记录原始指标列缺失的警告
                         # 这个警告现在应该只会出现在原始计算函数没有成功生成该列的情况下
                         print(f"[{stock_code}] Debug: TF {tf_str_diff}: 计算指标 {base_name} 差分时，原始指标列 {source_col_name} 不存在于 DataFrame 中。跳过。") # 调试输出
                         logger.debug(f"[{stock_code}] TF {tf_str_diff}: 计算指标 {base_name} 差分时，原始指标列 {source_col_name} 不存在于 DataFrame 中。跳过。")


            # 价格在布林带/肯特纳通道中的位置
            # 布林带位置 (%B 或类似)
            # ... (这部分代码保持不变，因为它查找的是 BOLL 的配置，根据日志，BOLL 配置是找到了的)
            # 获取布林带参数，优先使用 base_scoring 的，其次是 feature_engineering 的，最后是默认值
            # 这些参数的获取逻辑已在 _add_indicator_config 中处理，并存储在 indicator_configs 中
            # 我们需要查找注册的 BOLL 计算配置来获取实际参数


            # 查找 BOLL 的计算配置以获取实际参数
            boll_actual_params = None
            boll_registered_timeframes = []
            # 遍历已注册的指标配置列表
            found_boll_config = None # 增加查找结果变量
            for registered_config in indicator_configs:
                 # 如果找到名称为 'BOLL' 的配置项
                 if registered_config.get('name') == 'BOLL': # 使用 .get()
                      found_boll_config = registered_config # 找到配置
                      break # 找到后即退出循环

            if found_boll_config: # 检查是否找到了配置
                 boll_actual_params = found_boll_config['params']
                 boll_registered_timeframes = found_boll_config['timeframes'] # 获取实际计算的时间框架
                 # 增加调试输出
                 print(f"[{stock_code}] Debug: Found indicator config for 'BOLL'.")
            else:
                 logger.warning(f"[{stock_code}] 未找到指标计算配置 'BOLL'，无法确定实际参数构建布林带位置源列名。跳过布林带位置计算。")
                 print(f"[{stock_code}] Debug: 未找到指标计算配置 'BOLL'，无法确定实际参数构建布林带位置源列名。跳过布林带位置计算。")


            if boll_actual_params is not None: # 仅当找到 BOLL 配置时才计算位置
                 # 从实际参数中获取周期和乘数，用于构建查找的列名和衍生特征输出列名
                 boll_period_actual = boll_actual_params.get('period', default_boll_p['period'])
                 boll_std_actual = boll_actual_params.get('std_dev', default_boll_p['std_dev'])
                 # ATR 乘数可能用于基础指标列名 (尽管日志显示不包含)，但在衍生特征列名中通常不包含
                 # 我们依赖 _build_indicator_base_name_for_lookup 来构建查找名
                 # kc_atr_mult_actual = kc_actual_params.get('atr_multiplier', default_kc_p['atr_multiplier'])


                 # 遍历所有需要计算布林带位置的时间级别
                 # 只在 BOLL 实际计算过的时间级别上计算位置
                 # 使用注册配置中的实际时间级别列表
                 boll_pos_timeframes = boll_registered_timeframes


                 for tf_str_deriv_ch in boll_pos_timeframes:
                    # 构造收盘价列名 (所有时间级别都有收盘价)
                    close_col_ch = f'close_{tf_str_deriv_ch}'

                    # >>> 调用 _build_indicator_base_name_for_lookup 构建 BOLL 上下轨的基础列名 <<<
                    # 使用查找到的实际参数构建 BOLL 上下轨的基础列名（不含时间级别后缀）
                    lower_b_col_base = self._build_indicator_base_name_for_lookup('BBL', boll_actual_params, stock_code)
                    upper_b_col_base = self._build_indicator_base_name_for_lookup('BBU', boll_actual_params, stock_code)

                    # 检查是否成功构建了基础列名
                    if lower_b_col_base is None or upper_b_col_base is None:
                         logger.warning(f"[{stock_code}] TF {tf_str_deriv_ch}: 调用 _build_indicator_base_name_for_lookup(BBL/BBU, {boll_actual_params}) 无法构建基础列名进行布林带位置计算。跳过。")
                         print(f"[{stock_code}] Debug: TF {tf_str_deriv_ch}: 调用 _build_indicator_base_name_for_lookup(BBL/BBU, {boll_actual_params}) 无法构建基础列名进行布林带位置计算。跳过。")
                         continue # 无法构建基础列名，跳过该时间级别和布林带位置计算


                    # 构造带有时间周期后缀的完整 BOLL 上下轨列名 (用于查找)
                    lower_b_col_lookup = f'{lower_b_col_base}_{tf_str_deriv_ch}'
                    upper_b_col_lookup = f'{upper_b_col_base}_{tf_str_deriv_ch}'
                    # 中轨列名，虽然 %B 计算不需要，但有些地方可能需要
                    # middle_b_col = f'BBM_{boll_period_deriv}_{std_str_deriv}_{tf_str_deriv_ch}' # 保留以防万一，但查找不再使用手动拼接名

                    # 检查所有必要列是否存在 (使用通过 _build_indicator_base_name_for_lookup 构建的查找名称)
                    # >>> 新增调试输出：检查 BOLL 位置计算的原始列是否存在 <<<
                    print(f"[{stock_code}] Debug: TF {tf_str_deriv_ch}: Attempting to calculate BOLL POS. Required columns lookup: '{close_col_ch}', '{lower_b_col_lookup}', '{upper_b_col_lookup}'. Exists: {all(c in final_merged_df.columns for c in [close_col_ch, lower_b_col_lookup, upper_b_col_lookup])}") # 增加调试输出
                    # <<< 新增调试输出结束 <<<
                    if all(c in final_merged_df.columns for c in [close_col_ch, lower_b_col_lookup, upper_b_col_lookup]):
                        # 计算布林带宽度
                        band_width_b = final_merged_df[upper_b_col_lookup] - final_merged_df[lower_b_col_lookup]
                         # 计算价格在布林带中的相对位置 (类似于 %B)
                        # 避免除以零，对带宽进行小数值检查再相除
                        # 根据 JSON 衍生特征命名约定: CLOSE_BB_POS_{period}_{std_dev:.1f}_{timeframe}
                        # 衍生特征列名仍然需要手动构建，因为它有自己的命名约定，或者可以添加一个函数来构建衍生特征的输出列名
                        # 这里继续手动构建，但使用实际参数
                        # boll_period_actual 和 boll_std_actual 已经在循环外获取
                        boll_std_str_actual = f"{boll_std_actual:.1f}" # 标准差格式化为字符串，保留一位小数

                        bb_pos_col_name = f'CLOSE_BB_POS_{boll_period_actual}_{boll_std_str_actual}_{tf_str_deriv_ch}'

                        final_merged_df[bb_pos_col_name] = np.where(
                             np.abs(band_width_b) > 1e-9, # 检查带宽是否接近零
                            (final_merged_df[close_col_ch] - final_merged_df[lower_b_col_lookup]) / band_width_b,
                            0.5 # 带宽为零或接近零时，设为中轨位置
                        )
                    else:
                         # 记录缺失列的警告，显示期望查找的列名 (使用通过 _build_indicator_base_name_for_lookup 构建的名称)
                         missing_cols = [c for c in [close_col_ch, lower_b_col_lookup, upper_b_col_lookup] if c not in final_merged_df.columns]
                         print(f"[{stock_code}] Debug: TF {tf_str_deriv_ch}: 计算 BOLL 位置时，缺少列 {', '.join(missing_cols)}。跳过。") # 调试输出
                         logger.debug(f"[{stock_code}] TF {tf_str_deriv_ch}: 计算 BOLL 位置时，缺少列 {', '.join(missing_cols)}。跳过。")



            # 肯特纳通道位置 (类似处理)
            # 获取肯特纳通道参数，优先使用 feature_engineering 的，最后是默认值
            # 这些参数的获取逻辑已在 _add_indicator_config中处理，并存储在 indicator_configs 中
            # 我们需要查找注册的 KC 计算配置来获取实际参数

            # >>> 修改开始：获取实际使用的 KC 参数并构建基础列名进行查找和衍生特征命名 <<<
            # 查找 KC 的计算配置以获取实际参数
            kc_actual_params = None
            kc_registered_timeframes = []
            # 遍历已注册的指标配置列表
            found_kc_config = None # 增加查找结果变量
            for registered_config in indicator_configs:
                 # 如果找到名称为 'KC' 的配置项
                 if registered_config.get('name') == 'KC': # 使用 .get()
                      found_kc_config = registered_config # 找到配置
                      break # 找到后即退出循环

            if found_kc_config: # 检查是否找到了配置
                 kc_actual_params = found_kc_config['params']
                 kc_registered_timeframes = found_kc_config['timeframes'] # 获取实际计算的时间框架
                 # 增加调试输出
                 print(f"[{stock_code}] Debug: Found indicator config for 'KC'.")
                 # >>> 将实际参数的获取移到这里，并放在 if 块外面 <<<
                 kc_ema_p_actual = kc_actual_params.get('ema_period', default_kc_p['ema_period'])
                 kc_atr_p_actual = kc_actual_params.get('atr_period', default_kc_p['atr_period'])
                 # <<< 移动和修改结束 >>>
            else:
                 logger.warning(f"[{stock_code}] 未找到指标计算配置 'KC'，无法确定实际参数构建肯特纳通道位置源列名。跳过肯特纳通道位置计算。")
                 print(f"[{stock_code}] Debug: 未找到指标计算配置 'KC'，无法确定实际参数构建肯特纳通道位置源列名。跳过肯特纳通道位置计算。")

            # 仅当找到 KC 配置且实际参数不为 None 时才计算位置
            # 增加对 kc_actual_params 的 None 检查
            if kc_actual_params is not None:
                 # 从实际参数中获取周期，用于构建查找的列名和衍生特征输出列名
                 # 这些变量 (kc_ema_p_actual, kc_atr_p_actual) 现在在 if found_kc_config 块内定义，可以在这里访问到

                 # ATR 乘数可能用于基础指标列名 (尽管日志显示不包含)，但在衍生特征列名中通常不包含
                 # 我们依赖 _build_indicator_base_name_for_lookup 来构建查找名
                 # kc_atr_mult_actual = kc_actual_params.get('atr_multiplier', default_kc_p['atr_multiplier'])


                 # 遍历所有需要计算肯特纳通道位置的时间级别
                 # 只在 KC 实际计算过的时间级别上计算位置
                 # 使用注册配置中的实际时间级别列表
                 kc_pos_timeframes = kc_registered_timeframes


                 for tf_str_deriv_ch_kc in kc_pos_timeframes:
                    # 构造收盘价列名 (所有时间级别都有收盘价)
                    close_col_kc = f'close_{tf_str_deriv_ch_kc}'

                    # >>> 调用 _build_indicator_base_name_for_lookup 构建 KCL/KCU 的基础列名 <<<
                    # 使用查找到的实际参数构建 KCL/KCU 的基础列名（不含时间级别后缀）
                    # _build_indicator_base_name_for_lookup 已经修正为不包含 ATR 乘数在 KC 列名中
                    lower_kc_col_base = self._build_indicator_base_name_for_lookup('KCL', kc_actual_params, stock_code)
                    upper_kc_col_base = self._build_indicator_base_name_for_lookup('KCU', kc_actual_params, stock_code)

                    # 检查是否成功构建了基础列名
                    if lower_kc_col_base is None or upper_kc_col_base is None:
                         logger.warning(f"[{stock_code}] TF {tf_str_deriv_ch_kc}: 调用 _build_indicator_base_name_for_lookup(KCL/KCU, {kc_actual_params}) 无法构建基础列名进行肯特纳通道位置计算。跳过。")
                         print(f"[{stock_code}] Debug: TF {tf_str_deriv_ch_kc}: 调用 _build_indicator_base_name_for_lookup(KCL/KCU, {kc_actual_params}) 无法构建基础列名进行肯特纳通道位置计算。跳过。")
                         continue # 无法构建基础列名，跳过该时间级别和 KC 位置计算


                    # 构造带有时间周期后缀的完整 KCL/KCU 列名 (用于查找)
                    lower_kc_col_lookup = f'{lower_kc_col_base}_{tf_str_deriv_ch_kc}'
                    upper_kc_col_lookup = f'{upper_kc_col_base}_{tf_str_deriv_ch_kc}'


                    # 检查所有必要列是否存在 (使用通过 _build_indicator_base_name_for_lookup 构建的查找名称)
                     # >>> 新增调试输出：检查 KC 位置计算的原始列是否存在 <<<
                    print(f"[{stock_code}] Debug: TF {tf_str_deriv_ch_kc}: Attempting to calculate KC POS. Required columns lookup: '{close_col_kc}', '{lower_kc_col_lookup}', '{upper_kc_col_lookup}'. Exists: {all(c in final_merged_df.columns for c in [close_col_kc, lower_kc_col_lookup, upper_kc_col_lookup])}") # 增加调试输出
                    # <<< 新增调试输出结束 <<<
                    if all(c in final_merged_df.columns for c in [close_col_kc, lower_kc_col_lookup, upper_kc_col_lookup]):
                        # 计算肯特纳通道宽度
                        band_width_kc = final_merged_df[upper_kc_col_lookup] - final_merged_df[lower_kc_col_lookup]
                         # 计算价格在肯特纳通道中的相对位置
                        # 避免除以零，对带宽进行小数值检查再相除
                        # 根据 JSON 衍生特征命名约定: CLOSE_KC_POS_{ema_period}_{atr_period}_{timeframe}
                        # 衍生特征列名不包含 ATR 乘数，使用查找到的实际参数中的周期
                        # kc_ema_p_actual 和 kc_atr_p_actual 已经在循环外获取

                        kc_pos_col_name = f'CLOSE_KC_POS_{kc_ema_p_actual}_{kc_atr_p_actual}_{tf_str_deriv_ch_kc}' # 衍生特征列名不包含乘数

                        final_merged_df[kc_pos_col_name] = np.where(
                             np.abs(band_width_kc) > 1e-9, # 检查带宽是否接近零
                            (final_merged_df[close_col_kc] - final_merged_df[lower_kc_col_lookup]) / band_width_kc,
                            0.5 # 带宽为零或接近零时，设为中轨位置
                        )
                    else:
                        # 记录缺失列的警告，显示期望查找的列名 (现在是正确的，不含乘数)
                        missing_cols_kc = [c for c in [close_col_kc, lower_kc_col_lookup, upper_kc_col_lookup] if c not in final_merged_df.columns]
                        print(f"[{stock_code}] Debug: TF {tf_str_deriv_ch_kc}: 计算 KC 位置时，缺少列 {', '.join(missing_cols_kc)}。跳过。") # 调试输出
                        logger.debug(f"[{stock_code}] TF {tf_str_deriv_ch_kc}: 计算 KC 位置时，缺少列 {', '.join(missing_cols_kc)}。跳过。")

            # >>> 修改结束 <<<


            logger.info(f"[{stock_code}] 衍生特征计算完成。")
            print(f"[{stock_code}] Debug: 衍生特征计算完成，DataFrame 当前列数: {len(final_merged_df.columns)}") # 调试输出
        except Exception as e_deriv:
            logger.error(f"[{stock_code}] 计算衍生特征时出错: {e_deriv}", exc_info=True)
            print(f"[{stock_code}] Debug: 计算衍生特征时出错: {e_deriv}") # 调试输出


        # 8. 最终填充
        # ... (这部分代码保持不变)
        logger.info(f"[{stock_code}] 开始最终填充 NaN...")
        print(f"[{stock_code}] Debug: 开始最终填充 NaN...") # 调试输出
        nan_before_final_fill = final_merged_df.isnull().sum().sum() # 计算填充前总 NaN 数量
        final_merged_df.ffill(inplace=True) # 向前填充
        final_merged_df.bfill(inplace=True) # 向后填充
        nan_after_final_fill = final_merged_df.isnull().sum().sum() # 计算填充后总 NaN 数量

        logger.info(f"[{stock_code}] 最终填充完成。填充前 NaN 总数: {nan_before_final_fill}, 填充后 NaN 总数: {nan_after_final_fill}")
        print(f"[{stock_code}] Debug: 最终填充完成。填充前 NaN 总数: {nan_before_final_fill}, 填充后 NaN 总数: {nan_after_final_fill}") # 调试输出

        if nan_after_final_fill > 0:
            nan_cols_summary = final_merged_df.isnull().sum()
            nan_cols_summary = nan_cols_summary[nan_cols_summary > 0].sort_values(ascending=False)
            logger.warning(f"[{stock_code}] 最终填充后仍存在 NaN 的列 (前10条): {nan_cols_summary.head(10).to_dict()}")
            print(f"[{stock_code}] Debug: 最终填充后仍存在 NaN 的列 (前10条): {nan_cols_summary.head(10).to_dict()}") # 调试输出
            # 对于完全是NaN的列，可以考虑填充0或移除，但需谨慎
            # final_merged_df.fillna(0, inplace=True) # 可选：强制填充所有剩余NaN为0


        if final_merged_df.empty:
            logger.error(f"[{stock_code}] 最终合并和填充后的 DataFrame 为空。")
            print(f"[{stock_code}] Debug: 最终合并和填充后的 DataFrame 为空。") # 调试输出
            return None


        # 9. 返回最终结果
        logger.info(f"[{stock_code}] 策略数据准备完成，最终 DataFrame 形状: {final_merged_df.shape}, 列数: {len(final_merged_df.columns)}")
        print(f"[{stock_code}] Debug: 最终策略 DataFrame 准备完成.") # 调试输出
        print(f"[{stock_code}] Debug: 最终 DataFrame 形状: {final_merged_df.shape}") # 调试输出
        print(f"[{stock_code}] Debug: 最终 DataFrame 列数: {len(final_merged_df.columns)}") # 调试输出
        print(f"[{stock_code}] Debug: 最终 DataFrame 列名 (前50条): {final_merged_df.columns.tolist()[:50]}...") # 调试输出

        # 可以手动检查特定列是否存在，例如检查 'ADX_14_30' 和 'BBU_15_2.2_30'
        # 从 params 中获取用于构造列名的参数
        dmi_period_p = params.get('base_scoring',{}).get('dmi_period', 14)
        boll_period_p = params.get('base_scoring',{}).get('boll_period', fe_params.get('boll_period', default_boll_p['period'])) # 布林带周期，可能在 base_scoring 或 fe_params
        boll_std_dev_p = params.get('base_scoring',{}).get('boll_std_dev', fe_params.get('boll_std_dev', default_boll_p['std_dev'])) # 布林带标准差，可能在 base_scoring 或 fe_params
        boll_std_str_p = f"{boll_std_dev_p:.1f}" # 格式化标准差
        focus_tf_p = params.get('trend_following_params',{}).get('focus_timeframe', '30')

        # 构建期望的基础指标列名 (带参数，不含时间级别)
        # 这里的构建逻辑需要和 _build_indicator_base_name 或 calculate_* 返回的列名一致
        # 例如，如果 calculate_dmi 返回 ADX_14
        target_adx_col_base = f"ADX_{dmi_period_p}"
        # 如果 calculate_boll_bands_and_width 返回 BBU_15_2.2
        target_boll_upper_col_base = f"BBU_{boll_period_p}_{boll_std_str_p}" # 根据 JSON 约定和日志观察构建

        # 添加时间级别后缀后的完整列名
        target_adx_col_full = f"{target_adx_col_base}_{focus_tf_p}"
        target_boll_upper_col_full = f"{target_boll_upper_col_base}_{focus_tf_p}" # 根据 JSON 约定和日志观察构建

        print(f"[{stock_code}] Debug: 检查关键列 (for focus_tf='{focus_tf_p}'):") # 调试输出
        print(f"  - ADX 列 '{target_adx_col_full}': {'存在' if target_adx_col_full in final_merged_df.columns else '不存在'}") # 调试输出
        print(f"  - BOLL 上轨列 '{target_boll_upper_col_full}': {'存在' if target_boll_upper_col_full in final_merged_df.columns else '不存在'}") # 调试输出

        # 检查差分和 KC 位置衍生特征列名是否存在
        # 差分列名格式: {base_name}_{param_str}_DIFF{diff_period}_{timeframe}
        # 例如: RSI_14_DIFF1_30
        # 需要根据参数获取实际的列名
        rsi_params_for_check = {'period': params.get('base_scoring', {}).get('rsi_period', default_rsi_p['period'])} # 获取 RSI 参数用于检查
        rsi_base_name_for_check = self._build_indicator_base_name_for_lookup('RSI', rsi_params_for_check, stock_code)
        target_rsi_diff1_col = f"{rsi_base_name_for_check}_DIFF1_{focus_tf_p}" if rsi_base_name_for_check else f"RSI_14_DIFF1_{focus_tf_p}_fallback" # 使用构建函数或回退

        macd_params_for_check = params.get('base_scoring', {}).get('macd_params', default_macd_p) # 获取 MACD 参数用于检查
        macd_base_name_for_check = self._build_indicator_base_name_for_lookup('MACDh', macd_params_for_check, stock_code) # 检查 MACDh
        target_macd_diff1_col = f"{macd_base_name_for_check}_DIFF1_{focus_tf_p}" if macd_base_name_for_check else f"MACDh_12_26_9_DIFF1_{focus_tf_p}_fallback" # 使用构建函数或回退

        # KC 位置列名格式 (根据修改后代码，不包含 ATR 乘数在衍生特征列名中): CLOSE_KC_POS_{ema_period}_{atr_period}_{timeframe}
        # 从 KC 实际参数中获取周期用于构建检查列名
        # >>> 修改开始：在调试输出部分也使用实际参数构建 KC POS 列名 <<<
        # 检查 kc_actual_params 是否已定义且不为 None
        # 理论上在上面的 KC 位置计算部分找到了配置，这里的 kc_actual_params 应该不为 None
        # 但为了健壮性，增加检查
        kc_pos_check_col = "CLOSE_KC_POS_UNKNOWN" # 默认值
        if 'kc_actual_params' in locals() and kc_actual_params is not None:
             kc_ema_p_check = kc_actual_params.get('ema_period', default_kc_p['ema_period'])
             kc_atr_p_check = kc_actual_params.get('atr_period', default_kc_p['atr_period'])
             kc_pos_check_col = f"CLOSE_KC_POS_{kc_ema_p_check}_{kc_atr_p_check}_{focus_tf_p}"
        else:
             # 如果上面计算时都没找到 KC 配置，这里也无法构建正确的检查列名
             print(f"[{stock_code}] Debug: 未找到 KC 的实际计算参数，无法构建准确的 KC POS 检查列名。")


        print(f"[{stock_code}] Debug: 检查衍生特征关键列 (for focus_tf='{focus_tf_p}'):") # 调试输出
        print(f"  - RSI DIFF1 列 '{target_rsi_diff1_col}': {'存在' if target_rsi_diff1_col in final_merged_df.columns else '不存在'}") # 调试输出
        print(f"  - MACDh DIFF1 列 '{target_macd_diff1_col}': {'存在' if target_macd_diff1_col in final_merged_df.columns else '不存在'}") # 调试输出
        print(f"  - KC POS 列 '{kc_pos_check_col}': {'存在' if kc_pos_check_col in final_merged_df.columns else '不存在'}") # 调试输出
        # <<< 修改结束 >>>


        logger.debug(f"[{stock_code}] 最终 DataFrame 列名 (部分): {final_merged_df.columns.tolist()[:30]}...")
        return final_merged_df, indicator_configs

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

    # --- 所有指标计算函数 async def calculate_* ---
    async def calculate_atr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 ATR (Average True Range)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            logger.debug(f"计算 ATR 缺少必要列: high, low, close。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            atr_series = df.ta.atr(length=period, high=df[high_col], low=df[low_col], close=df[close_col], append=False)
            if atr_series is None or atr_series.empty:
                return None
            return pd.DataFrame({f'ATR_{period}': atr_series})
        except Exception as e:
            logger.error(f"计算 ATR (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_boll_bands_and_width(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0, close_col='close') -> Optional[pd.DataFrame]:
        """计算布林带及其宽度 BBW = (Upper - Lower) / Middle"""
        if df is None or df.empty or close_col not in df.columns:
            logger.debug(f"计算布林带缺少必要列: {close_col}。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            bbands_df = df.ta.bbands(length=period, std=std_dev, close=df[close_col], append=False)
            if bbands_df is None or bbands_df.empty:
                return None

            lower_col_name_ta = f'BBL_{period}_{std_dev:.1f}'
            middle_col_name_ta = f'BBM_{period}_{std_dev:.1f}'
            upper_col_name_ta = f'BBU_{period}_{std_dev:.1f}'
            # pandas_ta 还会生成 BBB_ (Bandwidth) 和 BBP_ (Percent B)
            bbw_col_name_ta = f'BBB_{period}_{std_dev:.1f}' # pandas_ta 的带宽列
            bbp_col_name_ta = f'BBP_{period}_{std_dev:.1f}' # pandas_ta 的 %B 列

            # 我们期望的列名
            lower_col_name_out = f'BBL_{period}_{std_dev:.1f}'
            middle_col_name_out = f'BBM_{period}_{std_dev:.1f}'
            upper_col_name_out = f'BBU_{period}_{std_dev:.1f}'
            bbw_col_name_out = f'BBW_{period}_{std_dev:.1f}' # 自定义宽度列名
            bbp_col_name_out = f'BBP_{period}_{std_dev:.1f}' # %B 列名

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

            # 计算自定义的 BBW (Upper - Lower) / Middle
            if all(c in result_df.columns for c in [lower_col_name_out, middle_col_name_out, upper_col_name_out]):
                # 避免除以零
                result_df[bbw_col_name_out] = np.where(
                    np.abs(result_df[middle_col_name_out]) > 1e-9, # 检查 middle band 是否接近零
                    (result_df[upper_col_name_out] - result_df[lower_col_name_out]) / result_df[middle_col_name_out],
                    np.nan # 如果 middle band 为零或接近零，则宽度为 NaN
                )
                has_any_col = True
            elif bbw_col_name_ta in bbands_df.columns: # 如果自定义计算失败，尝试用 pandas_ta 的带宽
                result_df[bbw_col_name_out] = bbands_df[bbw_col_name_ta] # 使用 pandas-ta 的 BBB 作为 BBW
                has_any_col = True


            if bbp_col_name_ta in bbands_df.columns: # 添加 %B
                result_df[bbp_col_name_out] = bbands_df[bbp_col_name_ta]
                has_any_col = True

            return result_df if has_any_col else None
        except Exception as e:
            logger.error(f"计算布林带及宽度 (周期 {period}, 标准差 {std_dev}) 出错: {e}", exc_info=True)
            return None

    async def calculate_historical_volatility(self, df: pd.DataFrame, period: int = 20, window_type: Optional[str] = None, close_col='close', annual_factor: int = 252) -> Optional[pd.DataFrame]:
        """计算历史波动率 (对数收益率的标准差，年化)"""
        if df is None or df.empty or close_col not in df.columns:
            return None
        try:
            log_returns = np.log(df[close_col] / df[close_col].shift(1))
            hv = log_returns.rolling(window=period, min_periods=max(1, int(period * 0.5))).std() * np.sqrt(annual_factor)
            return pd.DataFrame({f'HV_{period}': hv})
        except Exception as e:
            logger.error(f"计算历史波动率 (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_keltner_channels(self, df: pd.DataFrame, ema_period: int = 20, atr_period: int = 10, atr_multiplier: float = 2.0, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算肯特纳通道"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col]):
            logger.debug(f"计算肯特纳通道缺少必要列。可用列: {df.columns.tolist() if df is not None else 'None'}")
            return None
        try:
            kc_df = df.ta.kc(high=df[high_col], low=df[low_col], close=df[close_col], length=ema_period, atr_length=atr_period, scalar=atr_multiplier, mamode="ema", append=False)
            if kc_df is None or kc_df.empty:
                return None
            # pandas_ta kc 返回的列名类似: KCLe_20_2.0, KCBe_20_2.0, KCUe_20_2.0
            # 我们希望的列名: KCL_ema_atr, KCM_ema_atr, KCU_ema_atr
            # 注意：pandas_ta 的列名可能包含 EMA 周期和乘数，但不含 ATR 周期。我们需要自定义列名。
            lower_col_ta = kc_df.columns[0] # 通常是 Lower band
            basis_col_ta = kc_df.columns[1] # 通常是 Basis (EMA)
            upper_col_ta = kc_df.columns[2] # 通常是 Upper band

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
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            cci_series = df.ta.cci(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            if cci_series is None or cci_series.empty: return None
            return pd.DataFrame({f'CCI_{period}': cci_series})
        except Exception as e:
            logger.error(f"计算 CCI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_cmf(self, df: pd.DataFrame, period: int = 20, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 Chaikin Money Flow (CMF)"""
        if df is None or df.empty or not all(col in df.columns for col in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            cmf_series = df.ta.cmf(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False)
            if cmf_series is None or cmf_series.empty:
                return None
            return pd.DataFrame({f'CMF_{period}': cmf_series})
        except Exception as e:
            logger.error(f"计算 CMF (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_dmi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        # 检查输入DataFrame是否有效
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # 用pandas_ta.adx计算DMI相关指标（包含+DI, -DI, ADX）
            dmi_df = ta.adx(high=df[high_col], low=df[low_col], close=df[close_col], length=period)
            if dmi_df is None or dmi_df.empty:
                return None
            # pandas_ta.adx返回的列名: DMP_period (+DI), DMN_period (-DI), ADX_period
            rename_map = {}
            if f'DMP_{period}' in dmi_df.columns:
                rename_map[f'DMP_{period}'] = f'PDI_{period}'  # +DI
            if f'DMN_{period}' in dmi_df.columns:
                rename_map[f'DMN_{period}'] = f'NDI_{period}'  # -DI
            if f'ADX_{period}' in dmi_df.columns:
                rename_map[f'ADX_{period}'] = f'ADX_{period}'  # ADX
            # 重命名后返回
            return dmi_df.rename(columns=rename_map)
        except Exception as e:
            logger.error(f"计算 DMI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_ichimoku(self, df: pd.DataFrame, tenkan_period: int = 9, kijun_period: int = 26, senkou_period: int = 52, chikou_period: Optional[int] = None, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算一目均衡表 (Ichimoku Cloud)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            # pandas_ta ichimoku: tenkan, kijun, senkou_lead (chikou is derived from close)
            # chikou_period is usually same as kijun_period for the ICS_26 column
            # The senkou_lead parameter in ta.ichimoku is for the Senkou Span B lookback.
            # Senkou Span A is avg of tenkan and kijun, plotted senkou_period ahead.
            # Senkou Span B is avg of 52-period high and low, plotted senkou_period ahead.
            # Chikou Span is close plotted kijun_period behind.

            # ta.ichimoku returns two DataFrames: ichi (indicators) and span (shifted spans for plotting)
            # We are interested in 'ichi' for the raw indicator values.
            ichi_df, _ = df.ta.ichimoku(high=df[high_col], low=df[low_col], close=df[close_col],
                                        tenkan=tenkan_period, kijun=kijun_period, senkou=senkou_period,
                                        append=False)
            if ichi_df is None or ichi_df.empty:
                return None

            # Expected columns from pandas_ta ichimoku (may vary by version):
            # ITS_tenkan (Tenkan Sen), IKS_kijun (Kijun Sen),
            # ISA_tenkan (Senkou Span A), ISB_kijun (Senkou Span B - this is based on kijun period for some reason in ta, should be senkou_period)
            # ICS_kijun (Chikou Span - based on kijun period shift)

            # Let's define our desired output column names
            # TENKAN, KIJUN, CHIKOU, SENKOU_A, SENKOU_B
            result_df = pd.DataFrame(index=df.index)
            # Tenkan Sen
            if f'ITS_{tenkan_period}' in ichi_df.columns:
                result_df[f'TENKAN_{tenkan_period}'] = ichi_df[f'ITS_{tenkan_period}']
            # Kijun Sen
            if f'IKS_{kijun_period}' in ichi_df.columns:
                result_df[f'KIJUN_{kijun_period}'] = ichi_df[f'IKS_{kijun_period}']
            # Chikou Span (Lagging Span) - current close shifted BACK by kijun_period
            # pandas_ta ICS_kijun is already the correctly shifted close.
            if f'ICS_{kijun_period}' in ichi_df.columns: # This is Close price shifted BACK by kijun_period
                 result_df[f'CHIKOU_{kijun_period}'] = ichi_df[f'ICS_{kijun_period}']
            else: # Manual calculation if ICS is not present or named differently
                 result_df[f'CHIKOU_{kijun_period}'] = df[close_col].shift(-kijun_period)


            # Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2, plotted senkou_lead_period (usually kijun_period) AHEAD
            # pandas_ta ISA_tenkan is (ITS + IKS) / 2, but NOT shifted. We need to shift it.
            # The `span` DataFrame returned by ta.ichimoku contains the shifted versions.
            # However, for features, we usually want the unshifted calculated value, and let the model learn the lead.
            # Or, if we want the "current value of the future cloud", we'd use the shifted ones.
            # For now, let's calculate the unshifted Senkou A and B.
            if f'ITS_{tenkan_period}' in ichi_df.columns and f'IKS_{kijun_period}' in ichi_df.columns:
                senkou_a_calc = (ichi_df[f'ITS_{tenkan_period}'] + ichi_df[f'IKS_{kijun_period}']) / 2
                result_df[f'SENKOU_A_{tenkan_period}_{kijun_period}'] = senkou_a_calc
            elif f'ISA_{tenkan_period}' in ichi_df.columns: # If pandas_ta provides it directly (unshifted)
                result_df[f'SENKOU_A_{tenkan_period}_{kijun_period}'] = ichi_df[f'ISA_{tenkan_period}']


            # Senkou Span B (Leading Span B) - (senkou_period High + senkou_period Low) / 2, plotted senkou_lead_period AHEAD
            # pandas_ta ISB_kijun is (kijun_period high + kijun_period low) / 2. This is not standard.
            # Standard Senkou B uses a longer period (typically 52).
            if f'ISB_{kijun_period}' in ichi_df.columns and senkou_period == kijun_period: # If using kijun for senkou B as per ta's ISB col
                result_df[f'SENKOU_B_{kijun_period}'] = ichi_df[f'ISB_{kijun_period}']
            else: # Manual calculation for standard Senkou B
                rolling_high_senkou = df[high_col].rolling(window=senkou_period, min_periods=1).max()
                rolling_low_senkou = df[low_col].rolling(window=senkou_period, min_periods=1).min()
                senkou_b_calc = (rolling_high_senkou + rolling_low_senkou) / 2
                result_df[f'SENKOU_B_{senkou_period}'] = senkou_b_calc

            return result_df if not result_df.empty else None
        except Exception as e:
            logger.error(f"计算 Ichimoku (t={tenkan_period}, k={kijun_period}, s={senkou_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_kdj(self, df: pd.DataFrame, period: int = 9, signal_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            stoch_df = df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=period, d=signal_period, smooth_k=smooth_k_period, append=False)
            if stoch_df is None or stoch_df.empty: return None

            k_col_name_ta = stoch_df.columns[0] # STOCHk_...
            d_col_name_ta = stoch_df.columns[1] # STOCHd_...

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
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            ema_series = df.ta.ema(close=df[close_col], length=period, append=False)
            if ema_series is None or ema_series.empty: return None
            return pd.DataFrame({f'EMA_{period}': ema_series})
        except Exception as e:
            logger.error(f"计算 EMA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sma(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            sma_series = df.ta.sma(close=df[close_col], length=period, append=False)
            if sma_series is None or sma_series.empty: return None
            return pd.DataFrame({f'SMA_{period}': sma_series})
        except Exception as e:
            logger.error(f"计算 SMA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_amount_ma(self, df: pd.DataFrame, period: int = 20, amount_col='amount') -> Optional[pd.DataFrame]:
        if df is None or df.empty or amount_col not in df.columns: return None
        try:
            amt_ma_series = df[amount_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            return pd.DataFrame({f'AMT_MA_{period}': amt_ma_series})
        except Exception as e:
            logger.error(f"计算 AMT_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_macd(self, df: pd.DataFrame, period_fast: int = 12, period_slow: int = 26, signal_period: int = 9, close_col='close') -> Optional[pd.DataFrame]:
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            macd_df = df.ta.macd(close=df[close_col], fast=period_fast, slow=period_slow, signal=signal_period, append=False)
            if macd_df is None or macd_df.empty: return None
            # 列名: MACD_fast_slow_signal, MACDh_fast_slow_signal, MACDs_fast_slow_signal
            return macd_df
        except Exception as e:
            logger.error(f"计算 MACD (f={period_fast},s={period_slow},sig={signal_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mfi(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]): return None
        try:
            mfi_series = df.ta.mfi(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], length=period, append=False)
            if mfi_series is None or mfi_series.empty: return None
            return pd.DataFrame({f'MFI_{period}': mfi_series})
        except Exception as e:
            logger.error(f"计算 MFI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_mom(self, df: pd.DataFrame, period: int, close_col='close') -> Optional[pd.DataFrame]:
        """计算 Momentum (MOM)"""
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            mom_series = df.ta.mom(close=df[close_col], length=period, append=False)
            if mom_series is None or mom_series.empty: return None
            return pd.DataFrame({f'MOM_{period}': mom_series})
        except Exception as e:
            logger.error(f"计算 MOM (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_obv(self, df: pd.DataFrame, close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        if df is None or df.empty or not all(c in df.columns for c in [close_col, volume_col]): return None
        try:
            obv_series = df.ta.obv(close=df[close_col], volume=df[volume_col], append=False)
            if obv_series is None or obv_series.empty: return None
            return pd.DataFrame({'OBV': obv_series})
        except Exception as e:
            logger.error(f"计算 OBV 出错: {e}", exc_info=True)
            return None

    async def calculate_roc(self, df: pd.DataFrame, period: int = 12, close_col='close') -> Optional[pd.DataFrame]:
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            roc_series = df.ta.roc(close=df[close_col], length=period, append=False)
            if roc_series is None or roc_series.empty: return None
            return pd.DataFrame({f'ROC_{period}': roc_series})
        except Exception as e:
            logger.error(f"计算 ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_amount_roc(self, df: pd.DataFrame, period: int, amount_col='amount') -> Optional[pd.DataFrame]:
        """计算成交额的 ROC"""
        if df is None or df.empty or amount_col not in df.columns: return None
        try:
            aroc_series = df.ta.roc(close=df[amount_col], length=period, append=False) # 使用 amount 列作为 roc 的输入
            if aroc_series is None or aroc_series.empty: return None
            df_results = pd.DataFrame({f'AROC_{period}': aroc_series})
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True) # 处理除以0的情况
            return df_results
        except Exception as e:
            logger.error(f"计算 Amount ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_volume_roc(self, df: pd.DataFrame, period: int, volume_col='volume') -> Optional[pd.DataFrame]:
        """计算成交量的 ROC (VROC)"""
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            vroc_series = df.ta.roc(close=df[volume_col], length=period, append=False) # 使用 volume 列
            if vroc_series is None or vroc_series.empty: return None
            df_results = pd.DataFrame({f'VROC_{period}': vroc_series})
            df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df_results
        except Exception as e:
            logger.error(f"计算 Volume ROC (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_rsi(self, df: pd.DataFrame, period: int = 14, close_col='close') -> Optional[pd.DataFrame]:
        if df is None or df.empty or close_col not in df.columns: return None
        try:
            rsi_series = df.ta.rsi(close=df[close_col], length=period, append=False)
            if rsi_series is None or rsi_series.empty: return None
            return pd.DataFrame({f'RSI_{period}': rsi_series})
        except Exception as e:
            logger.error(f"计算 RSI (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_sar(self, df: pd.DataFrame, af_step: float = 0.02, max_af: float = 0.2, high_col='high', low_col='low') -> Optional[pd.DataFrame]:
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col]): return None
        try:
            psar_df = df.ta.psar(high=df[high_col], low=df[low_col], af0=af_step, af=af_step, max_af=max_af, append=False)
            if psar_df is None or psar_df.empty: return None
            # pandas_ta PSAR 返回多列，如 PSARl_0.02_0.2, PSARs_0.02_0.2
            # 我们通常需要合并的 SAR 线
            long_sar_col = next((col for col in psar_df.columns if col.startswith('PSARl')), None)
            short_sar_col = next((col for col in psar_df.columns if col.startswith('PSARs')), None)

            if long_sar_col and short_sar_col:
                sar_values = psar_df[long_sar_col].fillna(psar_df[short_sar_col])
                return pd.DataFrame({f'SAR_{af_step}_{max_af}': sar_values}) # 在列名中包含参数
            elif long_sar_col: # 只有 long
                return pd.DataFrame({f'SAR_{af_step}_{max_af}': psar_df[long_sar_col]})
            elif short_sar_col: # 只有 short
                return pd.DataFrame({f'SAR_{af_step}_{max_af}': psar_df[short_sar_col]})
            else:
                logger.warning(f"计算 SAR 未找到 PSARl 或 PSARs 列。返回列: {psar_df.columns.tolist()}")
                return None
        except Exception as e:
            logger.error(f"计算 SAR (af={af_step}, max_af={max_af}) 出错: {e}", exc_info=True)
            return None

    async def calculate_stoch(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k_period: int = 3, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            stoch_df = df.ta.stoch(high=df[high_col], low=df[low_col], close=df[close_col], k=k_period, d=d_period, smooth_k=smooth_k_period, append=False)
            if stoch_df is None or stoch_df.empty: return None
            # 列名: STOCHk_k_d_smooth, STOCHd_k_d_smooth
            return stoch_df
        except Exception as e:
            logger.error(f"计算 STOCH (k={k_period},d={d_period},s={smooth_k_period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_adl(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume') -> Optional[pd.DataFrame]:
        """计算 Accumulation/Distribution Line (ADL)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            adl_series = df.ta.ad(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], append=False)
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
            results = pd.DataFrame(index=df.index)
            prev_high = df[high_col].shift(1)
            prev_low = df[low_col].shift(1)
            prev_close = df[close_col].shift(1)

            PP = (prev_high + prev_low + prev_close) / 3
            results['PP'] = PP
            results['S1'] = (2 * PP) - prev_high
            results['S2'] = PP - (prev_high - prev_low)
            results['S3'] = results['S1'] - (prev_high - prev_low) # S3 = Low - 2 * (High - PP)
            results['S4'] = results['S2'] - (prev_high - prev_low) # S4 = Low - 3 * (High - PP)

            results['R1'] = (2 * PP) - prev_low
            results['R2'] = PP + (prev_high - prev_low)
            results['R3'] = results['R1'] + (prev_high - prev_low) # R3 = High + 2 * (PP - Low)
            results['R4'] = results['R2'] + (prev_high - prev_low) # R4 = High + 3 * (PP - Low)

            # Fibonacci Pivot Points
            diff = prev_high - prev_low
            results['F_R1'] = PP + 0.382 * diff
            results['F_R2'] = PP + 0.618 * diff
            results['F_R3'] = PP + 1.000 * diff
            results['F_S1'] = PP - 0.382 * diff
            results['F_S2'] = PP - 0.618 * diff
            results['F_S3'] = PP - 1.000 * diff

            return results.iloc[1:] # 第一行会是NaN
        except Exception as e:
            logger.error(f"计算 Pivot Points 出错: {e}", exc_info=True)
            return None

    async def calculate_vol_ma(self, df: pd.DataFrame, period: int = 20, volume_col='volume') -> Optional[pd.DataFrame]:
        if df is None or df.empty or volume_col not in df.columns: return None
        try:
            vol_ma_series = df[volume_col].rolling(window=period, min_periods=max(1, int(period*0.5))).mean()
            return pd.DataFrame({f'VOL_MA_{period}': vol_ma_series})
        except Exception as e:
            logger.error(f"计算 VOL_MA (周期 {period}) 出错: {e}", exc_info=True)
            return None

    async def calculate_vwap(self, df: pd.DataFrame, high_col='high', low_col='low', close_col='close', volume_col='volume', anchor: Optional[str] = None) -> Optional[pd.DataFrame]:
        """计算 VWAP。anchor='D' 可实现日内VWAP重置。"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col, volume_col]):
            return None
        try:
            vwap_series = df.ta.vwap(high=df[high_col], low=df[low_col], close=df[close_col], volume=df[volume_col], anchor=anchor, append=False)
            if vwap_series is None or vwap_series.empty: return None
            col_name = 'VWAP' if anchor is None else f'VWAP_{anchor}'
            return pd.DataFrame({col_name: vwap_series})
        except Exception as e:
            logger.error(f"计算 VWAP (anchor={anchor}) 出错: {e}", exc_info=True)
            return None

    async def calculate_willr(self, df: pd.DataFrame, period: int = 14, high_col='high', low_col='low', close_col='close') -> Optional[pd.DataFrame]:
        """计算 Williams %R (WILLR)"""
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]):
            return None
        try:
            willr_series = df.ta.willr(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)
            if willr_series is None or willr_series.empty: return None
            return pd.DataFrame({f'WILLR_{period}': willr_series})
        except Exception as e:
            logger.error(f"计算 WILLR (周期 {period}) 出错: {e}", exc_info=True)
            return None






