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

# 定义从衍生特征 base_name 到基础指标注册名称的映射表
# 注意：这个映射表本身主要服务于下游的衍生特征计算逻辑。
# 在 prepare_strategy_dataframe 中，我们通过额外注册配置来兼容下游可能不使用这个映射表
# 直接查找 K, D, J, ADX 等名称的行为。
DERIVATIVE_BASE_TO_REGISTERED_INDICATOR_MAP = {
    'K': 'KDJ',
    'D': 'KDJ',
    'J': 'KDJ',
    'ADX': 'DMI',
    'PDI': 'DMI',
    'NDI': 'DMI',
    # 其他 base_name 和注册名称一致的指标不需要在这里列出
    # 例如：'RSI': 'RSI', 'MACD': 'MACD', 'BOLL': 'BOLL', 'KC': 'KC', 'OBV': 'OBV', 等
}

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

    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, base_needed_bars: Optional[int] = None) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        根据策略 JSON 配置文件准备包含重采样基础数据和所有计算指标的 DataFrame。

        该函数执行以下步骤：
        1. 加载策略参数文件。
        2. 解析参数，识别所需的时间级别和要计算的指标及其参数。
        3. 确定最小时间级别和指标的最大回看期，估算需要获取的原始数据量。
        4. 并行获取所有所需时间级别的原始 OHLCV 数据。
        5. 对原始数据进行重采样和初步清洗，并给 OHLCV 列名添加时间周期后缀。
        6. 并行计算所有配置的基础指标。
        7. 将所有时间级别的 OHLCV 数据和计算出的指标数据合并到同一个 DataFrame 中，以最小时间级别索引为基准。
        8. (此方法不包含衍生特征计算，该步骤通常在合并数据后进行)
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
            print(f"[{stock_code}] Debug: pandas_ta 未加载。") # 调试输出：检查pandas_ta是否已导入
            logger.error(f"[{stock_code}] pandas_ta 未加载，无法准备策略数据。")
            return None

        # 1. 加载 JSON 参数文件
        try:
            # 检查文件是否存在
            if not os.path.exists(params_file):
                print(f"[{stock_code}] Debug: 策略参数文件未找到: {params_file}") # 调试输出：检查参数文件是否存在
                logger.error(f"[{stock_code}] 策略参数文件未找到: {params_file}")
                return None
            # 打开并解析 JSON 文件
            with open(params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"[{stock_code}] 从 {params_file} 加载策略参数成功。")
            # 添加参数加载成功的调试输出，可以部分打印参数
            print(f"[{stock_code}] Debug: 加载策略参数成功，部分参数: base_scoring={params.get('base_scoring', {})}, indicator_analysis_params={params.get('indicator_analysis_params', {})}, trend_following_params={params.get('trend_following_params', {})}") # 调试输出：打印部分加载的参数
        except Exception as e:
            # 记录加载或解析失败的错误
            print(f"[{stock_code}] Debug: 加载或解析参数文件失败: {e}") # 调试输出：记录参数文件加载/解析错误
            logger.error(f"[{stock_code}] 加载或解析参数文件 {params_file} 失败: {e}", exc_info=True)
            return None

        # 2. 识别需求：时间级别和全局指标最大回看期
        # 存储所有需要的时间级别，使用集合避免重复
        all_time_levels_needed: Set[str] = set()
        # 存储每个指标的计算配置 (名称, 函数, 参数, 适用的时间级别列表, 参数块键名, 参数覆盖键名)
        indicator_configs: List[Dict[str, Any]] = []

        # 辅助函数：用于简化从参数中提取指标配置并添加到 indicator_configs 列表的过程
        def _add_indicator_config(
            name: str, # 指标名称，用于识别和下游查找
            func: Callable, # 计算该指标的异步函数引用 (如 self.calculate_macd)
            param_block_key: Optional[str], # 参数块的键名，指示该指标的配置位于 JSON 的哪个部分 (如 'base_scoring')，None表示不来自特定参数块
            params_dict: Dict, # 传递给计算函数的参数字典
            applicable_tfs: Union[str, List[str]], # 适用于该指标的时间级别，可以是单个字符串或字符串列表
            param_override_key: Optional[str] = None # 可选：在参数块内，用于覆盖默认参数的具体指标键名
        ):
             # 将适用的时间级别转换为列表，并添加到 all_time_levels_needed 集合中
            tfs = [applicable_tfs] if isinstance(applicable_tfs, str) else applicable_tfs
            all_time_levels_needed.update(tfs)

            # 将完整的指标配置信息添加到列表中
            indicator_configs.append({
                'name': name, # 指标名称
                'func': func, # 指标计算函数引用
                'params': params_dict, # 传递给计算函数的最终参数字典
                'timeframes': tfs, # 适用该指标的时间级别列表
                'param_block_key': param_block_key, # 参数块键名 (用于日志或调试)
                'param_override_key': param_override_key # 参数覆盖键名 (用于日志或调试)
            })


        # 辅助函数：从参数中提取并合并单个指标的参数
        def _get_indicator_params(param_block: Dict, default_params: Dict, param_override_key: Optional[str] = None) -> Dict:
             """从参数块中提取并合并单个指标的参数。"""
             indi_specific_params_json = param_block.get(param_override_key, param_block) if param_override_key else param_block
             final_calc_params = default_params.copy()
             for k, v_json in indi_specific_params_json.items():
                 if k in final_calc_params:
                      final_calc_params[k] = v_json
             return final_calc_params


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
            if indi_key == 'macd':
                 macd_calc_params = _get_indicator_params(bs_params, default_macd_p)
                 _add_indicator_config('MACD', self.calculate_macd, 'base_scoring', macd_calc_params, bs_timeframes)
            elif indi_key == 'rsi':
                 rsi_calc_params = _get_indicator_params(bs_params, default_rsi_p)
                 _add_indicator_config('RSI', self.calculate_rsi, 'base_scoring', rsi_calc_params, bs_timeframes)
            elif indi_key == 'kdj':
                # KDJ 的周期参数在 JSON 中可能有特定的键名，这里从 bs_params 中提取并覆盖默认值
                kdj_calc_params = {
                    'period': bs_params.get('kdj_period_k', default_kdj_p['period']),
                    'signal_period': bs_params.get('kdj_period_d', default_kdj_p['signal_period']),
                    'smooth_k_period': bs_params.get('kdj_period_j', default_kdj_p['smooth_k_period'])
                }
                _add_indicator_config(
                    'KDJ', # 注册的配置名称是 'KDJ'
                    self.calculate_kdj, # 指向 KDJ 的计算函数
                    'base_scoring',
                    kdj_calc_params, # 传递已经根据 JSON 配置好的参数字典
                    bs_timeframes
                )
                # -------------------- 删除开始 --------------------
                # 移除为下游兼容性额外注册 K, D, J 的配置，避免生成重复列
                # print(f"[{stock_code}] Debug: 为下游兼容性额外注册 K, D, J 配置 (指向 KDJ)...") # 调试输出：额外注册KDJ组成部分
                # _add_indicator_config(
                #     'K', # 额外注册名称为 'K'
                #     self.calculate_kdj, # 指向 KDJ 的计算函数
                #     'base_scoring',
                #     kdj_calc_params, # 使用相同的 KDJ 参数
                #     bs_timeframes
                # )
                # _add_indicator_config(
                #     'D', # 额外注册名称为 'D'
                #     self.calculate_kdj, # 指向 KDJ 的计算函数
                #     'base_scoring',
                #     kdj_calc_params, # 使用相同的 KDJ 参数
                #     bs_timeframes
                # )
                # _add_indicator_config(
                #     'J', # 额外注册名称为 'J'
                #     self.calculate_kdj, # 指向 KDJ 的计算函数
                #     'base_scoring',
                #     kdj_calc_params, # 使用相同的 KDJ 参数
                #     bs_timeframes
                # )
                # -------------------- 删除结束 --------------------

            elif indi_key == 'boll':
                 boll_calc_params = _get_indicator_params(bs_params, default_boll_p)
                 print(f"[{stock_code}] Debug: 注册 BOLL 计算配置，使用参数: {boll_calc_params} 应用于时间框架: {bs_timeframes}") # 调试输出：打印BOLL参数
                 _add_indicator_config('BOLL', self.calculate_boll_bands_and_width, 'base_scoring', boll_calc_params, bs_timeframes)
            elif indi_key == 'cci':
                 cci_calc_params = _get_indicator_params(bs_params, default_cci_p)
                 _add_indicator_config('CCI', self.calculate_cci, 'base_scoring', cci_calc_params, bs_timeframes)
            elif indi_key == 'mfi':
                 mfi_calc_params = _get_indicator_params(bs_params, default_mfi_p)
                 _add_indicator_config('MFI', self.calculate_mfi, 'base_scoring', mfi_calc_params, bs_timeframes)
            elif indi_key == 'roc':
                 roc_calc_params = _get_indicator_params(bs_params, default_roc_p)
                 _add_indicator_config('ROC', self.calculate_roc, 'base_scoring', roc_calc_params, bs_timeframes)
            elif indi_key == 'dmi':
                print(f"[{stock_code}] Debug: 注册 DMI 计算配置，应用于时间框架: {bs_timeframes}") # 调试输出：注册DMI配置
                dmi_calc_params = _get_indicator_params(bs_params, default_dmi_p)
                _add_indicator_config('DMI', self.calculate_dmi, 'base_scoring', dmi_calc_params, bs_timeframes) # 注册的配置名称是 'DMI'
                # -------------------- 删除开始 --------------------
                # 移除为下游兼容性额外注册 ADX, PDI, NDI 的配置，避免生成重复列
                # print(f"[{stock_code}] Debug: 为下游兼容性额外注册 ADX, PDI, NDI 配置 (指向 DMI)...") # 调试输出：额外注册DMI组成部分
                # _add_indicator_config(
                #     'ADX', # 额外注册名称为 'ADX'
                #     self.calculate_dmi, # 指向 DMI 的计算函数
                #     'base_scoring',
                #     dmi_calc_params, # 使用相同的 DMI 参数
                #     bs_timeframes
                # )
                # _add_indicator_config(
                #     'PDI', # 额外注册名称为 'PDI'
                #     self.calculate_dmi, # 指向 DMI 的计算函数
                #     'base_scoring',
                #     dmi_calc_params, # 使用相同的 DMI 参数
                #     bs_timeframes
                # )
                # _add_indicator_config(
                #     'NDI', # 额外注册名称为 'NDI'
                #     self.calculate_dmi, # 指向 DMI 的计算函数
                #     'base_scoring',
                #     dmi_calc_params, # 使用相同的 DMI 参数
                #     bs_timeframes
                # )
                # -------------------- 删除结束 --------------------

            elif indi_key == 'sar':
                 sar_calc_params = _get_indicator_params(bs_params, default_sar_p)
                 _add_indicator_config('SAR', self.calculate_sar, 'base_scoring', sar_calc_params, bs_timeframes)
            # EMA 和 SMA 通常不在 score_indicators 里，而是作为独立特征或趋势分析的一部分
            # 如果参数中明确要计算EMA/SMA作为评分指标 (这种情况较少，通常在 feature_engineering)
            elif indi_key == 'ema':
                 # 注意：这里假设如果 EMA 在 score_indicators 里，只计算一个特定周期的 EMA
                 ema_p = bs_params.get('ema_period', default_sma_ema_p['period'])
                 ema_calc_params = {'period': ema_p} # 明确构建参数字典
                 _add_indicator_config('EMA', self.calculate_ema, 'base_scoring', ema_calc_params, bs_timeframes, param_override_key='ema_params') # 注册的配置名称是 'EMA'

            elif indi_key == 'sma':
                 # 注意：这里假设如果 SMA 在 score_indicators 里，只计算一个特定周期的 SMA
                 sma_p = bs_params.get('sma_period', default_sma_ema_p['period'])
                 sma_calc_params = {'period': sma_p} # 明确构建参数字典
                 _add_indicator_config('SMA', self.calculate_sma, 'base_scoring', sma_calc_params, bs_timeframes, param_override_key='sma_params') # 注册的配置名称是 'SMA'


        # --- 注册成交量和 indicator_analysis 相关指标的计算配置 ---
        vc_params = params.get('volume_confirmation', {})
        ia_params = params.get('indicator_analysis_params', {})
        # 成交量/额分析通常在基础时间框架上进行，除非 volume_confirmation 中指定了特定的 'tf'
        vol_ana_tf_cfg = vc_params.get('tf', bs_timeframes) # 从vc_params获取tf配置
        vol_ana_tfs_vc = [vol_ana_tf_cfg] if isinstance(vol_ana_tf_cfg, str) else vol_ana_tf_cfg if vc_params.get('enabled', False) else [] # 如果vc未启用，不考虑其tf
        ia_tfs_cfg = ia_params.get('timeframes', bs_timeframes) # 从ia_params获取timeframes配置
        ia_tfs = [ia_tfs_cfg] if isinstance(ia_tfs_cfg, str) else ia_tfs_cfg if ia_params else [] # 如果ia未启用，不考虑其timeframes

        # 合并所有可能的时间框架，并去重
        target_vol_ana_tfs = list(set(vol_ana_tfs_vc) | set(ia_tfs) | set(bs_timeframes)) # 确保基础时间级别也在内

        all_time_levels_needed.update(target_vol_ana_tfs)


        # AMT_MA 计算
        if vc_params.get('enabled', False) or ia_params.get('calculate_amt_ma', False):
             amt_ma_p = vc_params.get('amount_ma_period', ia_params.get('amount_ma_period', 20))
             amt_ma_calc_params = {'period': amt_ma_p}
             _add_indicator_config('AMT_MA', self.calculate_amount_ma, 'volume_confirmation' if vc_params.get('enabled', False) else 'indicator_analysis_params', amt_ma_calc_params, target_vol_ana_tfs, param_override_key='amount_ma_params')
        # CMF 计算
        if vc_params.get('enabled', False) or ia_params.get('calculate_cmf', False):
            cmf_p = vc_params.get('cmf_period', ia_params.get('cmf_period', 20))
            cmf_calc_params = {'period': cmf_p}
            _add_indicator_config('CMF', self.calculate_cmf, 'volume_confirmation' if vc_params.get('enabled', False) else 'indicator_analysis_params', cmf_calc_params, target_vol_ana_tfs, param_override_key='cmf_params')
        # VOL_MA 计算
        if ia_params.get('calculate_vol_ma', False):
            vol_ma_p = ia_params.get('volume_ma_period', 20)
            vol_ma_calc_params = {'period': vol_ma_p}
            _add_indicator_config('VOL_MA', self.calculate_vol_ma, 'indicator_analysis_params', vol_ma_calc_params, target_vol_ana_tfs, param_override_key='volume_ma_params')

        # 其他分析指标 (来自 indicator_analysis_params)
        ia_timeframes = ia_params.get('timeframes', bs_timeframes) # 这里重新获取 ia_params 的 timeframes 用于这些指标
        all_time_levels_needed.update(ia_timeframes)

        # STOCH 计算
        if ia_params.get('calculate_stoch', False):
            stoch_p = {
                'k_period': ia_params.get('stoch_k', default_stoch_p['k_period']),
                'd_period': ia_params.get('stoch_d', default_stoch_p['d_period']),
                'smooth_k_period': ia_params.get('stoch_smooth_k', default_stoch_p['smooth_k_period'])
            }
            stoch_calc_params = _get_indicator_params(ia_params, default_stoch_p, param_override_key='stoch_params') # 使用辅助函数提取参数
            _add_indicator_config('STOCH', self.calculate_stoch, 'indicator_analysis_params', stoch_calc_params, ia_timeframes) # 注册的配置名称是 'STOCH'

        # VWAP 计算
        if ia_params.get('calculate_vwap', False):
            vwap_p = {'anchor': ia_params.get('vwap_anchor', None)}
            vwap_calc_params = _get_indicator_params(ia_params, {'anchor': None}, param_override_key='vwap_params') # VWAP 默认 anchor 是 None
            _add_indicator_config('VWAP', self.calculate_vwap, 'indicator_analysis_params', vwap_calc_params, ia_timeframes) # 注册的配置名称是 'VWAP'

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
             ichimoku_calc_params = _get_indicator_params(ia_params, default_ichimoku_p, param_override_key='ichimoku_params') # 使用辅助函数提取参数
             _add_indicator_config('Ichimoku', self.calculate_ichimoku, 'indicator_analysis_params', ichimoku_calc_params, ia_timeframes) # 注册的配置名称是 'Ichimoku'

        # Pivot Points 计算 (通常基于日线计算，但代码中注册为 bs_timeframes，这里修正为只在 'D' 上计算)
        if ia_params.get('calculate_pivot_points', False):
             # Pivot 通常基于日线计算，所以适用时间级别应为 ['D']
             pivot_calc_params = _get_indicator_params(ia_params, {}, param_override_key='pivot_params') # Pivot Points 通常无参数
             _add_indicator_config('PivotPoints', self.calculate_pivot_points, 'indicator_analysis_params', pivot_calc_params, ['D']) # 注册的配置名称是 'PivotPoints'
             all_time_levels_needed.add('D') # 确保 'D' 被包含在所需时间级别中


        # --- 注册特征工程指标的计算配置 ---
        fe_params = params.get('feature_engineering_params', {})
        # 特征工程默认应用于基础时间框架，除非参数中指定了 apply_on_timeframes
        fe_timeframes_cfg = fe_params.get('apply_on_timeframes', bs_timeframes)
        fe_timeframes = [fe_timeframes_cfg] if isinstance(fe_timeframes_cfg, str) else fe_timeframes_cfg if fe_params else []
        all_time_levels_needed.update(fe_timeframes)

        # ATR 计算
        if fe_params.get('calculate_atr', False):
             atr_calc_params = _get_indicator_params(fe_params, default_atr_p, param_override_key='atr_params')
             _add_indicator_config('ATR', self.calculate_atr, 'feature_engineering_params', atr_calc_params, fe_timeframes) # 注册的配置名称是 'ATR'
        # 历史波动率 (HV) 计算
        if fe_params.get('calculate_hv', False):
             hv_calc_params = _get_indicator_params(fe_params, default_hv_p, param_override_key='hv_params')
             _add_indicator_config('HV', self.calculate_historical_volatility, 'feature_engineering_params', hv_calc_params, fe_timeframes) # 注册的配置名称是 'HV'
        # 肯特纳通道 (KC) 计算
        if fe_params.get('calculate_kc', False):
             kc_calc_params = _get_indicator_params(fe_params, default_kc_p, param_override_key='kc_params')
             _add_indicator_config('KC', self.calculate_keltner_channels, 'feature_engineering_params', kc_calc_params, fe_timeframes) # 注册的配置名称是 'KC'
        # 动量 (MOM) 计算
        if fe_params.get('calculate_mom', False):
             mom_calc_params = _get_indicator_params(fe_params, default_mom_p, param_override_key='mom_params')
             _add_indicator_config('MOM', self.calculate_mom, 'feature_engineering_params', mom_calc_params, fe_timeframes) # 注册的配置名称是 'MOM'
        # Williams %R (WILLR) 计算
        if fe_params.get('calculate_willr', False):
             willr_calc_params = _get_indicator_params(fe_params, default_willr_p, param_override_key='willr_params')
             _add_indicator_config('WILLR', self.calculate_willr, 'feature_engineering_params', willr_calc_params, fe_timeframes) # 注册的配置名称是 'WILLR'
        # 成交量变化率 (VROC) 计算
        if fe_params.get('calculate_vroc', False):
             vroc_calc_params = _get_indicator_params(fe_params, default_roc_p, param_override_key='vroc_params')
             _add_indicator_config('VROC', self.calculate_volume_roc, 'feature_engineering_params', vroc_calc_params, fe_timeframes) # 注册的配置名称是 'VROC'
        # 成交额变化率 (AROC) 计算
        if fe_params.get('calculate_aroc', False):
             aroc_calc_params = _get_indicator_params(fe_params, default_roc_p, param_override_key='aroc_params')
             _add_indicator_config('AROC', self.calculate_amount_roc, 'feature_engineering_params', aroc_calc_params, fe_timeframes) # 注册的配置名称是 'AROC'

        # 计算 EMA 和 SMA (如果参数中指定了周期列表)
        # 这些通常作为独立特征或用于计算与其他指标的关系
        for ma_type, ma_func in [('EMA', self.calculate_ema), ('SMA', self.calculate_sma)]:
            # 从 fe_params 中获取指定类型的均线周期列表，例如 "ema_periods": [5, 10, 20]
            ma_periods = fe_params.get(f'{ma_type.lower()}_periods', [])
            if ma_periods: # 如果配置了周期列表
                 # 为每个周期添加一个计算配置
                for p in ma_periods:
                     if isinstance(p, int) and p > 0: # 确保周期是有效的正整数
                         ma_calc_params = {'period': p}
                         # 使用通用的名称 'EMA' 或 'SMA' 注册，参数中带周期
                         _add_indicator_config(ma_type, ma_func, 'feature_engineering_params', ma_calc_params, fe_timeframes) # 注册的配置名称是 'EMA' 或 'SMA'
            elif fe_params.get(f'calculate_{ma_type.lower()}', False): # 如果没有指定列表，但calculate标志为True，则使用默认周期
                 ma_p = fe_params.get(f'{ma_type.lower()}_period', default_sma_ema_p['period'])
                 ma_calc_params = {'period': ma_p}
                 _add_indicator_config(ma_type, ma_func, 'feature_engineering_params', ma_calc_params, fe_timeframes) # 注册的配置名称是 'EMA' 或 'SMA'


        # OBV 是基础的，通常都需要计算。确保只添加一次。
        # OBV 的计算没有依赖特定的参数块，所以 param_block_key 可以是 None
        if not any(conf['name'] == 'OBV' for conf in indicator_configs):
            # 将 OBV 添加到所有需要的时间级别上计算
            _add_indicator_config('OBV', self.calculate_obv, None, {}, list(all_time_levels_needed)) # OBV 无参数，使用 None 作为 param_block_key，注册的配置名称是 'OBV'


        # --- 调试点：确认需要的时间级别集合中是否包含目标 focus_tf (如 '30') ---
        # 获取 focus_timeframe (用于后续检查)
        focus_tf = params.get('trend_following_params', {}).get('focus_timeframe', '30')
        print(f"[{stock_code}] Debug: 策略关注的时间级别 (focus_tf): {focus_tf}") # 调试输出：策略关注的时间级别
        print(f"[{stock_code}] Debug: 所有策略所需时间级别集合: {sorted(list(all_time_levels_needed))}") # 调试输出：所有所需时间级别

        # --- 调试点：打印注册完成的 indicator_configs 列表摘要 ---
        print(f"[{stock_code}] Debug: Contents of indicator_configs after registration (summary):") # 调试输出
        for i, conf in enumerate(indicator_configs):
             # 避免打印完整的 params 字典，只打印类型和部分键
             params_summary = ', '.join([f"{k}:{conf['params'][k]}" for k in list(conf['params'].keys())[: min(3, len(conf['params'])) ]]) + ('...' if len(conf['params']) > 3 else '')
             print(f"  [{i}] Name: {conf['name']}, Timeframes: {conf['timeframes']}, Params (partial): {{{params_summary}}}") # 调试输出
        print("-" * 30) # 分隔线


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
             print(f"[{stock_code}] Debug: 无法精确确定最小时间级别 (按分钟)，回退使用基础时间级别列表的第一个: {min_time_level}") # 调试输出：回退使用第一个基础时间级别
        elif min_time_level is None:
            logger.error(f"[{stock_code}] 无法确定有效的最小时间级别从所需级别: {all_time_levels_needed}")
            return None

        logger.info(f"[{stock_code}] 策略所需时间级别: {sorted(list(all_time_levels_needed))}, 最小时间级别: {min_time_level} ({min_tf_minutes if min_tf_minutes != float('inf') else 'N/A'} 分钟)")


        # --- 动态计算 global_max_lookback ---
        # 估算所有指标计算所需的最大历史回看期，用于确定需要获取多少根 K 线数据
        global_max_lookback = 0
        # 回看期计算应基于实际注册的核心指标配置
        # 我们可以直接遍历 indicator_configs，但需要注意同一个指标不同参数的配置
        # 一个简单的方法是只考虑那些在 params 中有对应配置块或 score_indicators 列表中的指标
        # 并且只计算其主要周期参数的回看期
        # 这里还是采用之前的去重思路，基于 name 和 params 来识别唯一的“计算任务”配置
        unique_configs_for_lookback = {}
        for config in indicator_configs:
            # 使用 name 和参数的哈希作为键进行去重
            # 注意：这里需要确保所有核心配置都能被包含进来，而不会因为 name 是 K, D, J 等而被过滤掉
            # 但是我们已经移除了 K, D, J 等的额外注册，所以现在 indicator_configs 应该只包含核心名称
            # 过滤掉 param_block_key 为 None 且 name 不是 OBV 的项（理论上不应该有）
            if config.get('param_block_key') is None and config['name'] != 'OBV':
                 continue # 跳过非 OBV 的无参数块配置

            params_hashable = tuple(sorted(config['params'].items()))
            key = (config['name'], params_hashable)
            if key not in unique_configs_for_lookback:
                 unique_configs_for_lookback[key] = config

        print(f"[{stock_code}] Debug: 用于回看期计算的唯一指标配置数量: {len(unique_configs_for_lookback)}") # 调试输出


        for config in unique_configs_for_lookback.values():
            current_max_period = 0
            # 检查所有可能的周期参数键名
            period_keys = ['period', 'period_fast', 'period_slow', 'signal_period', 'k_period', 'd_period', 'smooth_k_period',
                           'ema_period', 'atr_period', 'tenkan_period', 'kijun_period', 'senkou_period', 'atr_multiplier']
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
            # 注意：这里判断的是核心指标名称 'DMI'
            if config['name'] == 'DMI' and 'period' in config['params']: # DMI/ADX
                # ADX 计算需要至少 2*period 根数据，外加 EMA 平滑，所以需要更长的回看期
                current_max_period = max(current_max_period, int(config['params']['period'] * 2.5 + 10)) # 稍微保守一些的估算
            if config['name'] == 'KC': # Keltner Channels 也依赖 EMA 和 ATR 周期
                 p = config['params']
                 current_max_period = max(current_max_period, p.get('ema_period', 0), p.get('atr_period', 0))
            if config['name'] == 'Ichimoku': # Ichimoku depends on multiple periods
                 p = config['params']
                 current_max_period = max(current_max_period, p.get('tenkan_period', 0), p.get('kijun_period', 0), p.get('senkou_period', 0))
            if config['name'] in ['EMA', 'SMA']: # EMA/SMA 直接使用其 period
                 current_max_period = max(current_max_period, config['params'].get('period', 0))
            if config['name'] in ['AMT_MA', 'VOL_MA', 'MOM', 'WILLR', 'VROC', 'AROC', 'RSI', 'CCI', 'MFI', 'ROC', 'ATR', 'SAR']: # 其他单周期指标
                 # 检查 period, k_period, atr_period 等可能的周期键
                 current_max_period = max(current_max_period,
                                           config['params'].get('period', 0),
                                           config['params'].get('k_period', 0),
                                           config['params'].get('atr_period', 0)) # 添加 atr_period 检查，以防某些指标如 KC 注册时是核心，但此处未单独处理
            if config['name'] == 'KDJ': # KDJ 依赖其平滑周期
                 p = config['params']
                 current_max_period = max(current_max_period, p.get('period', 0), p.get('signal_period', 0), p.get('smooth_k_period', 0))
            # OBV, ADL, PivotPoints, VWAP 的回看期计算逻辑可能不同或较小，这里简化处理，主要考虑带周期的指标


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
                global_max_lookback=global_max_lookback # 考虑全局指标回看期
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
            print(f"  - TF {tf}: {'None/Empty' if df is None or df.empty else f'Shape {df.shape}, Columns: {df.columns.tolist()}'}") # 调试输出：检查原始数据状态

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
            print(f"[{stock_code}] Debug: TF {tf_resample} 重采样并重命名后的列: {resampled_df_renamed.columns.tolist()[:20]}...") # 调试输出：检查重采样和重命名后的列

            # 将处理后的 DataFrame 存储起来
            resampled_ohlcv_dfs[tf_resample] = resampled_df_renamed

        # 最终检查最小时间级别的数据是否存在且非空
        if min_time_level not in resampled_ohlcv_dfs or resampled_ohlcv_dfs[min_time_level].empty:
             logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 重采样后的数据不可用。终止。")
             return None # 最小时间级别数据缺失，终止流程

        # 使用最小时间级别的数据索引作为最终合并的基准索引
        base_index = resampled_ohlcv_dfs[min_time_level].index
        logger.info(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的重采样索引作为合并基准，数量: {len(base_index)}。")

        # 5. 计算所有配置的基础指标 (并行)
        indicator_calculation_tasks = [] # 存储异步指标计算任务

        # 定义一个异步辅助函数来计算单个指标
        async def _calculate_single_indicator_async(tf_calc: str, base_df_with_suffix: pd.DataFrame, config_item: Dict) -> Optional[Tuple[str, pd.DataFrame]]:
            """异步计算单个时间级别上的单个指标。"""
            print(f"[{stock_code}] Debug: 开始计算指标 {config_item['name']} for TF {tf_calc}...") # 调试输出：开始计算单个指标
            # 检查输入的基础数据是否有效
            if base_df_with_suffix is None or base_df_with_suffix.empty:
                print(f"[{stock_code}] Debug: TF {tf_calc}: 基础OHLCV数据为空，无法计算指标 {config_item['name']}") # 调试输出：基础数据为空
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
            # Keltner Channels (KC) 需要高、低、收盘价以及 ATR，calculate_keltner_channels 函数内部会计算或使用 ATR
            # 所以这里仅需要 OHLCV 列，不需要检查 ATR 列是否存在于 df_for_ta 中
            pass # 无需额外添加 required_cols_for_func for KC

            # 检查 df_for_ta 是否包含计算该指标所需的所有列
            if not all(col in df_for_ta.columns for col in required_cols_for_func):
                missing_cols_str = ", ".join(list(required_cols_for_func - set(df_for_ta.columns)))
                print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 缺少必要列 ({missing_cols_str})，跳过计算。") # 调试输出：缺少必要列
                logger.debug(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} 时，df_for_ta 缺少必要列 ({missing_cols_str})。可用: {df_for_ta.columns.tolist()}")
                return None # 缺少必要列，无法计算

            try:
                # 获取为该指标准备好的参数副本，传递给计算函数
                func_params_to_pass = config_item['params'].copy()
                # *** 调试点：打印传递给计算函数的参数 ***
                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 传递的计算参数: {func_params_to_pass}") # 调试输出

                # 调用具体的指标计算函数 (这些函数使用标准的列名，如 'close')
                # calculate_dmi 和 calculate_boll_bands_and_width 就是在这里被调用的
                # 注意：这里调用的是 config_item['func']，而不是根据 name 判断调用哪个函数
                # config_item['func'] 在 _add_indicator_config 时已经正确设定（例如 KDJ 配置的 func 是 calculate_kdj）
                indicator_result_df = await config_item['func'](df_for_ta, **func_params_to_pass)


                # *** 调试点：检查指标计算函数的原始返回结果 ***
                if indicator_result_df is None:
                     print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算结果为 None。") # 调试输出：计算结果为None
                     logger.debug(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为 None。")
                     return None
                if indicator_result_df.empty:
                     print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 计算结果为 Empty DataFrame。") # 调试输出：计算结果为空DataFrame
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
                        # 假设计算函数返回的 Series/DataFrame 列名已经包含参数
                        # 例如 pandas_ta 的 KDJ 会返回 K_x_y_z, D_x_y_z, J_x_y_z
                        # MACD 会返回 MACD_x_y_z, MACDh_x_y_z, MACDs_x_y_z
                        # RSI 返回 RSI_x
                        # DMI 返回 ADX_x, PDI_x, NDI_x
                        # OBV 返回 OBV
                        # 所以这里转换为 DataFrame 时，直接使用原始的 Series name 或 config_item['name'] 作为备用
                        indicator_result_df = indicator_result_df.to_frame(name=series_name)
                        print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 转换为DataFrame后列名: {indicator_result_df.columns.tolist()}") # 调试输出：Series转换为DataFrame后的列名
                    else:
                        print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 返回类型 {type(indicator_result_df)} 无法处理。") # 调试输出：无法处理的返回类型
                        logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 返回类型 {type(indicator_result_df)} 无法处理。")
                        return None # 无法处理的返回类型

                # *** 关键步骤：给计算出的指标列名添加时间周期后缀 ***
                # 例如，将 calculate_dmi 返回的 'ADX_14', 'PDI_14', 'NDI_14' 重命名为 'ADX_14_30', 'PDI_14_30', 'NDI_14_30'
                # lambda 函数 x: f"{x}_{tf_calc}" 对 DataFrame 的所有列名应用后缀
                result_renamed_df = indicator_result_df.rename(columns=lambda x: f"{x}_{tf_calc}")

                # *** 调试点：检查添加后缀后的指标 DataFrame 列名 ***
                # print(f"[{stock_code}] Debug: TF {tf_calc}, 指标 {config_item['name']}: 添加后缀后的列名: {result_renamed_df.columns.tolist()}") # 调试输出

                # 返回时间级别和带有后缀的结果 DataFrame
                return (tf_calc, result_renamed_df)
            except Exception as e_calc:
                # 捕获指标计算过程中的异常并记录错误日志
                print(f"[{stock_code}] Debug: TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}") # 调试输出：指标计算出错
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
                    print(f"[{stock_code}] Debug: 时间框架 {tf_conf} 的基础数据未找到 ({config_item_loop['name']})，无法创建计算任务。") # 调试输出：基础数据未找到
                    logger.warning(f"[{stock_code}] 时间框架 {tf_conf} 在 resampled_ohlcv_dfs 中未找到，无法为指标 {config_item_loop['name']} 创建计算任务。")

        # 并行执行所有指标计算任务
        # return_exceptions=True 确保即使某个任务失败，也不会中断整个 gather，而是返回异常对象
        calculated_results_tuples = await asyncio.gather(*indicator_calculation_tasks, return_exceptions=True)

        # 按时间级别对计算结果进行分组存储
        # defaultdict(list) 用于存储 { 'tf' : [indi_df1, indi_df2, ...] }
        calculated_indicators_by_tf = defaultdict(list)

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
        # OBV_MA 计算通常在 volume_confirmation 或 indicator_analysis_params 中配置
        obv_ma_period_vc = vc_params.get('obv_ma_period')
        obv_ma_period_ia = ia_params.get('obv_ma_period')
        # 优先使用 volume_confirmation 的配置，其次 indicator_analysis_params，最后默认值
        obv_ma_period = obv_ma_period_vc if obv_ma_period_vc is not None else \
                        obv_ma_period_ia if obv_ma_period_ia is not None else 10 # 默认周期为 10

        # 检查是否需要计算 OBV_MA
        # OBV_MA 是否启用取决于 volume_confirmation 或 indicator_analysis_params 中的 calculate_obv_ma 标志
        calculate_obv_ma_enabled = vc_params.get('calculate_obv_ma', False) or ia_params.get('calculate_obv_ma', False)

        if calculate_obv_ma_enabled: # 如果启用了 OBV_MA 计算
            print(f"[{stock_code}] Debug: 尝试计算 OBV_MA (周期 {obv_ma_period})...") # 调试输出：尝试计算OBV_MA
            # OBV_MA 的计算适用于所有已经计算了 OBV 的时间级别
            # 我们需要找到 OBV 列所在的 DataFrame
            # -------------------- 修改开始 --------------------
            # 检查是否在 indicator_configs 中注册了 OBV，并获取其时间级别
            obv_config = next((c for c in indicator_configs if c['name'] == 'OBV'), None)
            if obv_config:
                obv_timeframes = obv_config['timeframes']
                for tf_obv_ma in obv_timeframes:
                     # 在该时间级别的指标列表中查找 OBV 列所在的 DataFrame
                     # 期望的 OBV 列名是 'OBV_{tf_obv_ma}' (因为基础指标计算时已经添加了时间后缀)
                     obv_col_pattern = f'OBV_{tf_obv_ma}'
                     obv_df_found = None
                     # 遍历该时间级别下所有计算出的指标 DataFrame，查找包含 OBV 列的那个
                     if tf_obv_ma in calculated_indicators_by_tf:
                         for df in calculated_indicators_by_tf[tf_obv_ma]:
                              if obv_col_pattern in df.columns:
                                   obv_df_found = df
                                   break # 找到 OBV 所在的 DataFrame

                     if obv_df_found is not None:
                         # 找到了 OBV 所在的 DataFrame，现在计算 OBV 的移动平均
                         try:
                             # 确保 OBV 列是数值类型，防止移动平均计算出错
                             if not pd.api.types.is_numeric_dtype(obv_df_found[obv_col_pattern]):
                                  print(f"[{stock_code}] Debug: TF {tf_obv_ma}: OBV 列 '{obv_col_pattern}' 不是数值类型，无法计算 OBV_MA。") # 调试输出：OBV列非数值
                                  logger.warning(f"[{stock_code}] TF {tf_obv_ma}: OBV 列 '{obv_col_pattern}' 不是数值类型，无法计算 OBV_MA。")
                                  continue # 跳过当前时间级别的 OBV_MA 计算

                             # 使用 pandas 的 rolling().mean() 计算移动平均
                             obv_ma_col_name = f'OBV_MA_{obv_ma_period}_{tf_obv_ma}' # OBV MA 列名带周期和时间后缀
                             # 使用 apply 在 Series 上计算移动平均，确保 NaN 处理正确
                             obv_ma_series = obv_df_found[obv_col_pattern].rolling(window=obv_ma_period, min_periods=1).mean() # 计算移动平均，min_periods=1 表示窗口不足时也计算
                             obv_ma_df = obv_ma_series.to_frame(name=obv_ma_col_name) # 转换为 DataFrame

                             # 将计算出的 OBV_MA DataFrame 添加到该时间级别下的指标列表中
                             # 注意：这里直接修改了 defaultdict 中的列表
                             calculated_indicators_by_tf[tf_obv_ma].append(obv_ma_df)
                             print(f"[{stock_code}] Debug: TF {tf_obv_ma}: OBV_MA_{obv_ma_period} 计算并添加到分组列表。列名: [{obv_ma_col_name}]") # 调试输出：OBV_MA计算成功
                             logger.info(f"[{stock_code}] TF {tf_obv_ma}: OBV_MA_{obv_ma_period} 计算完成。")

                         except Exception as e_obv_ma:
                             print(f"[{stock_code}] Debug: TF {tf_obv_ma}: 计算 OBV_MA_{obv_ma_period} 时出错: {e_obv_ma}") # 调试输出：OBV_MA计算出错
                             logger.error(f"[{stock_code}] TF {tf_obv_ma}: 计算 OBV_MA_{obv_ma_period} 时出错: {e_obv_ma}", exc_info=True)
                     else:
                         # 如果该时间级别未找到 OBV 数据，记录警告
                         print(f"[{stock_code}] Debug: TF {tf_obv_ma}: 未找到 OBV 列 '{obv_col_pattern}'，跳过 OBV_MA 计算。") # 调试输出：未找到OBV列
                         logger.warning(f"[{stock_code}] TF {tf_obv_ma}: 未找到 OBV 列 '{obv_col_pattern}'，跳过 OBV_MA 计算。")
            else:
                 print(f"[{stock_code}] Debug: 未找到 OBV 的注册配置，无法计算 OBV_MA。") # 调试输出：未找到OBV配置
                 logger.warning(f"[{stock_code}] 未找到 OBV 的注册配置，无法计算 OBV_MA。")

            # -------------------- 修改结束 --------------------


        # 6. 将所有时间级别的 OHLCV 和指标数据合并到最终 DataFrame
        # 使用最小时间级别的 OHLCV 数据作为基础 DataFrame
        if min_time_level not in resampled_ohlcv_dfs:
             logger.error(f"[{stock_code}] 关键错误: 最小时间级别 {min_time_level} 的 OHLCV 数据在重采样后丢失。终止。")
             return None # 理论上不会发生，前面已经检查过

        final_df = resampled_ohlcv_dfs[min_time_level].copy()
        logger.info(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的数据作为最终 DataFrame 基础，形状: {final_df.shape}")

        # 遍历所有时间级别和其计算出的指标 DataFrame 列表
        for tf_merge, df_list_merge in calculated_indicators_by_tf.items():
            # 将该时间级别所有的指标 DataFrame concat 到一起
            # 注意：如果 df_list_merge 中的 DataFrame 包含重复列名，pd.concat 会保留所有重复列
            # 这是日志中出现多个 K_7_3_3_5, ADX_14_5 的原因
            # 修复方法是在注册时避免重复计算和返回 DataFrames，让每个基础指标只计算一次并返回一个 DataFrame
            # 这已经在前面的删除额外注册的步骤中完成
            if df_list_merge:
                # 确保合并的 DataFrame 具有相同的索引，这里应该没问题，因为它们都基于重采样数据
                merged_indicators_for_tf = pd.concat(df_list_merge, axis=1)

                # 将合并后的指标 DataFrame 与 final_df 按索引对齐合并
                # 使用 left 合并，以确保 final_df (基于最小时间级别索引) 的所有行都被保留
                # 如果 tf_merge 就是最小时间级别，则只合并指标列
                if tf_merge == min_time_level:
                     # 确保不合并 OHLCV 列 (尽管列名带有后缀，理论上不会冲突，但为了清晰)
                     # 实际上，因为基础 OHLCV 已经在 final_df 里了，merge 相同列名会自动处理（例如添加 _x, _y 后缀或报错），
                     # 但因为我们已经处理了重复注册，这里的 merged_indicators_for_tf 应该只包含指标列。
                     # 使用 join 更直接，因为它默认按索引对齐
                     final_df = final_df.join(merged_indicators_for_tf, how='left')
                     logger.info(f"[{stock_code}] 合并时间级别 {tf_merge} 的 {len(df_list_merge)} 个指标 DataFrame (到基础DF)，当前 final_df 形状: {final_df.shape}")
                else:
                     # 如果是非最小时间级别，需要先获取该时间级别的 OHLCV 数据
                     if tf_merge in resampled_ohlcv_dfs:
                         ohlcv_df_for_merge = resampled_ohlcv_dfs[tf_merge]
                         # 将 OHLCV 数据和指标数据先 concat 到一起
                         df_to_merge = pd.concat([ohlcv_df_for_merge, merged_indicators_for_tf], axis=1)
                         logger.info(f"[{stock_code}] 合并时间级别 {tf_merge} 的 OHLCV 数据和 {len(df_list_merge)} 个指标 DataFrame，形状: {df_to_merge.shape}")

                         # 将这个时间级别的数据 (OHLCV + 指标) 与 final_df 按索引对齐合并
                         final_df = final_df.join(df_to_merge, how='left')
                         logger.info(f"[{stock_code}] 将时间级别 {tf_merge} 的数据合并到 final_df，当前 final_df 形状: {final_df.shape}")
                     else:
                         logger.warning(f"[{stock_code}] 时间级别 {tf_merge} 的重采样 OHLCV 数据未找到，无法合并该时间级别的数据和指标。")

            else:
                logger.warning(f"[{stock_code}] 时间级别 {tf_merge} 没有计算出任何指标 DataFrame。")


        logger.info(f"[{stock_code}] 所有时间级别的数据和指标合并完成，最终 DataFrame 形状: {final_df.shape}, 列数: {len(final_df.columns)}")
        print(f"[{stock_code}] Debug: 合并完成后，DataFrame 形状: {final_df.shape}, 列数: {len(final_df.columns)}") # 调试输出：合并后形状
        print(f"[{stock_code}] Debug: 合并完成后，部分列: {final_df.columns.tolist()[:50]}...") # 调试输出：合并后部分列

        # 7. 最终填充 NaN
        # 在合并不同时间级别的数据后，可能引入 NaN (例如，分钟线没有对应的高级别数据)
        # 使用前向填充和后向填充组合，然后用 0 填充剩余的 NaN
        logger.info(f"[{stock_code}] 开始最终填充 NaN...")
        print(f"[{stock_code}] Debug: 开始最终填充 NaN...") # 调试输出：开始填充NaN

        # 填充前 NaN 总数
        nan_before = final_df.isnull().sum().sum()
        print(f"[{stock_code}] Debug: 填充前 NaN 总数: {nan_before}") # 调试输出：填充前NaN数量

        # 按列进行填充，可以提高效率，避免创建 DataFrame 副本
        for col in final_df.columns:
             # 先前向填充
             final_df[col].fillna(method='ffill', inplace=True)
             # 然后后向填充
             final_df[col].fillna(method='bfill', inplace=True)

        # 对于仍然存在的 NaN (例如 DataFrame 开头部分的 NaN，FFILL和BFILL无法处理)，用 0 填充
        final_df.fillna(0, inplace=True)

        # 填充后 NaN 总数
        nan_after = final_df.isnull().sum().sum()
        logger.info(f"[{stock_code}] 最终填充完成。填充前 NaN 总数: {nan_before}, 填充后 NaN 总数: {nan_after}")
        print(f"[{stock_code}] Debug: 最终填充完成。填充前 NaN 总数: {nan_before}, 填充后 NaN 总数: {nan_after}") # 调试输出：填充后NaN数量


        # 8. 策略数据准备完成
        logger.info(f"[{stock_code}] 策略数据准备完成，最终 DataFrame 形状: {final_df.shape}, 列数: {len(final_df.columns)}")
        print(f"[{stock_code}] Debug: 最终策略 DataFrame 准备完成.") # 调试输出
        print(f"[{stock_code}] Debug: 最终 DataFrame 形状: {final_df.shape}") # 调试输出
        print(f"[{stock_code}] Debug: 最终 DataFrame 列数: {len(final_df.columns)}") # 调试输出
        # 限制打印的列数，防止日志过长
        print(f"[{stock_code}] Debug: 最终 DataFrame 列名 (前50条): {final_df.columns.tolist()[:50]}...") # 调试输出


        # 返回最终的 DataFrame 和注册的指标配置列表
        # 指标配置列表 indicator_configs 中不再包含额外的冗余项，只包含核心配置
        return final_df, indicator_configs



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






