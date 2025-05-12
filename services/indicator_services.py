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
from typing import Any, Callable, List, Optional, Tuple, Union, Dict
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

    # --- 核心数据准备函数 ---
    async def prepare_strategy_dataframe(self, stock_code: str, params_file: str, base_needed_bars: Optional[int] = None) -> Optional[Tuple[pd.DataFrame, List[Dict[str, Any]]]]:
        """
        根据策略 JSON 配置文件准备包含重采样基础数据和所有计算指标的 DataFrame。
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
        all_time_levels_needed = set()
        indicator_configs = [] # 存储 (indicator_name, function, params_dict, timeframes_list)

        # 辅助函数，用于从参数中提取指标配置
        def _add_indicator_config(
            name: str,
            func: Callable,
            param_block_key: str, # 参数块的键名，如 'base_scoring', 'feature_engineering_params'
            default_params: Dict, # 指标计算函数自身的默认参数
            applicable_tfs: Union[str, List[str]],
            param_override_key: Optional[str] = None # 在参数块内，具体指标的参数覆盖键名，如 'rsi_period'
        ):
            # 从主参数中获取特定指标的参数字典
            param_block = params.get(param_block_key, {})
            # 如果有 override_key，则从块内获取该指标的特定参数，否则使用块本身作为参数源
            indi_specific_params_json = param_block.get(param_override_key, param_block) if param_override_key else param_block

            # 合并参数：默认参数 < JSON块参数 < JSON特定指标参数
            # 注意：这里简化处理，直接用 indi_specific_params_json 覆盖 default_params
            # 更精细的控制可能需要逐个检查参数是否存在
            final_calc_params = default_params.copy()
            # 只用 JSON 中存在的键去覆盖默认值
            for k, v_json in indi_specific_params_json.items():
                if k in final_calc_params: # 确保只覆盖指标函数实际关心的参数
                    final_calc_params[k] = v_json
                # 如果JSON中有指标函数不关心的额外参数，会被忽略，这是期望的行为

            tfs = [applicable_tfs] if isinstance(applicable_tfs, str) else applicable_tfs
            all_time_levels_needed.update(tfs)
            indicator_configs.append({
                'name': name,
                'func': func,
                'params': final_calc_params, # 传递给计算函数的参数
                'timeframes': tfs,
                'param_block_key': param_block_key, # 用于日志或调试
                'param_override_key': param_override_key # 用于日志或调试
            })

        # --- 从参数文件动态构建指标计算列表 ---
        bs_params = params.get('base_scoring', {})
        bs_timeframes = bs_params.get('timeframes', ['5', '15', '30', '60', 'D'])
        all_time_levels_needed.update(bs_timeframes)

        # 常用指标的默认参数（如果JSON中未提供，则使用这些）
        # 这些默认值应与 calculate_* 函数的默认值一致或作为其基础
        default_macd_p = {'period_fast': 12, 'period_slow': 26, 'signal_period': 9}
        default_rsi_p = {'period': 14}
        default_kdj_p = {'period': 9, 'signal_period': 3, 'smooth_k_period': 3}
        default_boll_p = {'period': 20, 'std_dev': 2.0}
        default_cci_p = {'period': 14}
        default_mfi_p = {'period': 14}
        default_roc_p = {'period': 12}
        default_dmi_p = {'period': 14}
        default_sar_p = {'af_step': 0.02, 'max_af': 0.2}
        default_stoch_p = {'k_period': 14, 'd_period': 3, 'smooth_k_period': 3}
        default_atr_p = {'period': 14}
        default_hv_p = {'period': 20, 'annual_factor': 252} # 日线年化因子
        default_kc_p = {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0}
        default_mom_p = {'period': 10}
        default_willr_p = {'period': 14}
        default_sma_ema_p = {'period': 20} # 通用均线周期

        # 注册基础评分指标
        for indi_key in bs_params.get('score_indicators', []):
            if indi_key == 'macd': _add_indicator_config('MACD', self.calculate_macd, 'base_scoring', default_macd_p, bs_timeframes)
            elif indi_key == 'rsi': _add_indicator_config('RSI', self.calculate_rsi, 'base_scoring', default_rsi_p, bs_timeframes)
            elif indi_key == 'kdj':
                # 从 bs_params 中提取 KDJ 特定的周期参数
                kdj_calc_params = {
                    'period': bs_params.get('kdj_period_k', default_kdj_p['period']), # 使用 kdj_period_k 覆盖默认的 period
                    'signal_period': bs_params.get('kdj_period_d', default_kdj_p['signal_period']), # 使用 kdj_period_d 覆盖默认的 signal_period
                    'smooth_k_period': bs_params.get('kdj_period_j', default_kdj_p['smooth_k_period']) # 使用 kdj_period_j 覆盖默认的 smooth_k_period
                }
                _add_indicator_config(
                    'KDJ',
                    self.calculate_kdj,
                    'base_scoring', # 仍然指明参数来源的块
                    kdj_calc_params, # 传递已经根据JSON配置好的参数字典
                    bs_timeframes
                    # param_override_key 不再需要，因为我们已经手动处理了参数映射
                )
            elif indi_key == 'boll': _add_indicator_config('BOLL', self.calculate_boll_bands_and_width, 'base_scoring', default_boll_p, bs_timeframes)
            elif indi_key == 'cci': _add_indicator_config('CCI', self.calculate_cci, 'base_scoring', default_cci_p, bs_timeframes)
            elif indi_key == 'mfi': _add_indicator_config('MFI', self.calculate_mfi, 'base_scoring', default_mfi_p, bs_timeframes)
            elif indi_key == 'roc': _add_indicator_config('ROC', self.calculate_roc, 'base_scoring', default_roc_p, bs_timeframes)
            elif indi_key == 'dmi': _add_indicator_config('DMI', self.calculate_dmi, 'base_scoring', default_dmi_p, bs_timeframes)
            elif indi_key == 'sar': _add_indicator_config('SAR', self.calculate_sar, 'base_scoring', default_sar_p, bs_timeframes)
            # EMA 和 SMA 通常不在 score_indicators 里，而是作为独立特征或趋势分析的一部分
            elif indi_key == 'ema': # 如果参数中明确要计算EMA作为评分指标
                 ema_p = bs_params.get('ema_period', default_sma_ema_p['period']) # 假设参数中可配ema_period
                 _add_indicator_config(f'EMA_{ema_p}', self.calculate_ema, 'base_scoring', {'period': ema_p}, bs_timeframes, param_override_key='ema_params')
            elif indi_key == 'sma':
                 sma_p = bs_params.get('sma_period', default_sma_ema_p['period'])
                 _add_indicator_config(f'SMA_{sma_p}', self.calculate_sma, 'base_scoring', {'period': sma_p}, bs_timeframes, param_override_key='sma_params')


        # 成交量相关指标 (volume_confirmation, indicator_analysis_params)
        vc_params = params.get('volume_confirmation', {})
        ia_params = params.get('indicator_analysis_params', {})
        # 默认在所有基础时间框架计算这些，除非参数中指定了特定tf
        vol_ana_tf = vc_params.get('tf', bs_timeframes) if vc_params.get('enabled', False) else bs_timeframes

        if vc_params.get('enabled', False) or ia_params.get('calculate_amt_ma', True): # 假设 ia_params 也可以控制
            _add_indicator_config('AMT_MA', self.calculate_amount_ma, 'volume_confirmation', {'period': vc_params.get('amount_ma_period',20)}, vol_ana_tf)
        if vc_params.get('enabled', False) or ia_params.get('calculate_cmf', True):
            _add_indicator_config('CMF', self.calculate_cmf, 'volume_confirmation', {'period': vc_params.get('cmf_period',20)}, vol_ana_tf)
        if ia_params.get('calculate_vol_ma', True):
            _add_indicator_config('VOL_MA', self.calculate_vol_ma, 'indicator_analysis_params', {'period': ia_params.get('volume_ma_period',20)}, bs_timeframes)

        # 其他分析指标 (indicator_analysis_params)
        _add_indicator_config('STOCH', self.calculate_stoch, 'indicator_analysis_params', default_stoch_p, bs_timeframes)
        _add_indicator_config('VWAP', self.calculate_vwap, 'indicator_analysis_params', {'anchor': ia_params.get('vwap_anchor', None)}, bs_timeframes) # VWAP anchor 可配置
        _add_indicator_config('ADL', self.calculate_adl, 'indicator_analysis_params', {}, bs_timeframes) # ADL 通常无参数
        _add_indicator_config('Ichimoku', self.calculate_ichimoku, 'indicator_analysis_params',
                              {'tenkan_period': ia_params.get('ichimoku_tenkan',9),
                               'kijun_period': ia_params.get('ichimoku_kijun',26),
                               'senkou_period': ia_params.get('ichimoku_senkou',52)},
                              bs_timeframes)
        _add_indicator_config('PivotPoints', self.calculate_pivot_points, 'indicator_analysis_params', {}, ['D']) # Pivot 通常基于日线计算

        # 特征工程参数 (feature_engineering_params)
        fe_params = params.get('feature_engineering_params', {})
        fe_timeframes = fe_params.get('apply_on_timeframes', bs_timeframes)

        if fe_params.get('calculate_atr', True):
            _add_indicator_config('ATR', self.calculate_atr, 'feature_engineering_params', default_atr_p, fe_timeframes, param_override_key='atr_params')
        if fe_params.get('calculate_hv', True):
            _add_indicator_config('HV', self.calculate_historical_volatility, 'feature_engineering_params', default_hv_p, fe_timeframes, param_override_key='hv_params')
        if fe_params.get('calculate_kc', True):
            _add_indicator_config('KC', self.calculate_keltner_channels, 'feature_engineering_params', default_kc_p, fe_timeframes, param_override_key='kc_params')
        if fe_params.get('calculate_mom', True):
            _add_indicator_config('MOM', self.calculate_mom, 'feature_engineering_params', default_mom_p, fe_timeframes, param_override_key='mom_params')
        if fe_params.get('calculate_willr', True):
            _add_indicator_config('WILLR', self.calculate_willr, 'feature_engineering_params', default_willr_p, fe_timeframes, param_override_key='willr_params')
        if fe_params.get('calculate_vroc', True):
            _add_indicator_config('VROC', self.calculate_volume_roc, 'feature_engineering_params', default_roc_p, fe_timeframes, param_override_key='vroc_params')
        if fe_params.get('calculate_aroc', True):
            _add_indicator_config('AROC', self.calculate_amount_roc, 'feature_engineering_params', default_roc_p, fe_timeframes, param_override_key='aroc_params')

        # 计算 EMA 和 SMA (如果参数中指定了周期列表)
        for ma_type, ma_func in [('EMA', self.calculate_ema), ('SMA', self.calculate_sma)]:
            ma_periods = fe_params.get(f'{ma_type.lower()}_periods', []) # e.g., "ema_periods": [5, 10, 20]
            for p in ma_periods:
                if isinstance(p, int) and p > 0:
                    _add_indicator_config(f'{ma_type}_{p}', ma_func, 'feature_engineering_params', {'period': p}, fe_timeframes, param_override_key=f'{ma_type.lower()}_params')

        # OBV 是基础的，通常都需要 (确保只添加一次)
        # 检查是否已有 OBV 配置，避免重复
        if not any(conf['name'] == 'OBV' for conf in indicator_configs):
            _add_indicator_config('OBV', self.calculate_obv, 'base_scoring', {}, all_time_levels_needed)

        # --- 确定最小时间级别 ---
        min_time_level = None
        min_tf_minutes = float('inf')
        if not all_time_levels_needed: # 如果没有任何时间级别被识别出来
            logger.error(f"[{stock_code}] 未能从参数文件中确定任何需要的时间级别。")
            return None

        for tf_str_loop in all_time_levels_needed: # 使用不同的变量名避免覆盖
            minutes = self._get_timeframe_in_minutes(tf_str_loop)
            if minutes is not None:
                if minutes < min_tf_minutes:
                    min_tf_minutes = minutes
                    min_time_level = tf_str_loop
        if min_time_level is None:
            logger.error(f"[{stock_code}] 无法确定有效的最小时间级别从所需级别: {all_time_levels_needed}")
            return None
        logger.info(f"[{stock_code}] 策略所需时间级别: {sorted(list(all_time_levels_needed))}, 最小时间级别: {min_time_level} ({min_tf_minutes} 分钟)")

        # --- 动态计算 global_max_lookback ---
        global_max_lookback = 0
        for config in indicator_configs:
            current_max_period = 0
            # 检查所有可能的周期参数名
            period_keys = ['period', 'period_slow', 'k_period', 'ema_period', 'atr_period', 'tenkan_period', 'kijun_period', 'senkou_period']
            for p_key in period_keys:
                if p_key in config['params']:
                    current_max_period = max(current_max_period, config['params'][p_key])

            # 特殊处理组合周期
            if 'period_slow' in config['params'] and 'signal_period' in config['params']: # MACD
                current_max_period = max(current_max_period, config['params']['period_slow'] + config['params']['signal_period'])
            if config['name'] == 'DMI' and 'period' in config['params']: # DMI/ADX
                current_max_period = max(current_max_period, config['params']['period'] * 2 + 10) # ADX 通常需要更长

            global_max_lookback = max(global_max_lookback, current_max_period)

        global_max_lookback += 100 # 固定缓冲
        logger.info(f"[{stock_code}] 动态计算的全局指标最大回看期 (含缓冲): {global_max_lookback}")

        # 3. 并行获取原始 OHLCV 数据
        ohlcv_tasks = {}
        effective_base_needed_bars = base_needed_bars if base_needed_bars is not None else \
                                     params.get('lstm_training_config',{}).get('lstm_window_size', 60) + global_max_lookback + 500

        for tf_fetch in all_time_levels_needed:
            needed_bars_for_tf = self._calculate_needed_bars_for_tf(
                target_tf=tf_fetch, min_tf=min_time_level,
                base_needed_bars=effective_base_needed_bars,
                global_max_lookback=global_max_lookback
            )
            logger.info(f"[{stock_code}] 时间级别 {tf_fetch}: 基础({min_time_level})需(估算){effective_base_needed_bars}条, 指标需{global_max_lookback}条 -> 动态计算需获取 {needed_bars_for_tf} 条原始数据.")
            ohlcv_tasks[tf_fetch] = self._get_ohlcv_data(stock_code, tf_fetch, needed_bars_for_tf)

        ohlcv_results = await asyncio.gather(*ohlcv_tasks.values())
        raw_ohlcv_dfs = dict(zip(all_time_levels_needed, ohlcv_results))

        # 4. 重采样和初步清洗
        resampled_ohlcv_dfs = {}
        min_usable_bars = math.ceil(effective_base_needed_bars * 0.6) # 降低到60%作为硬性门槛
        for tf_resample, raw_df in raw_ohlcv_dfs.items():
            if raw_df is None or raw_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 没有获取到原始数据，跳过。")
                continue
            resampled_df = self._resample_and_clean_dataframe(raw_df, tf_resample, min_periods=1, fill_method='ffill')
            if resampled_df is None or resampled_df.empty:
                logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 重采样后数据为空，跳过。")
                continue
            if tf_resample == min_time_level and len(resampled_df) < min_usable_bars:
                 logger.error(f"[{stock_code}] 最小时间级别 {tf_resample} 重采样后数据量 {len(resampled_df)} 条，少于最低可用阈值 {min_usable_bars} 条。无法继续。")
                 return None
            if len(resampled_df) < global_max_lookback * 0.5: # 如果数据量远少于回看期，警告
                 logger.warning(f"[{stock_code}] 时间级别 {tf_resample} 重采样后数据量 {len(resampled_df)} 条，显著少于全局指标最大回看期 {global_max_lookback} 条。")

            rename_map = {col: f"{col}_{tf_resample}" for col in ['open', 'high', 'low', 'close', 'volume', 'amount'] if col in resampled_df.columns}
            resampled_df_renamed = resampled_df.rename(columns=rename_map) if rename_map else resampled_df.copy()
            resampled_ohlcv_dfs[tf_resample] = resampled_df_renamed

        if min_time_level not in resampled_ohlcv_dfs or resampled_ohlcv_dfs[min_time_level].empty:
             logger.error(f"[{stock_code}] 最小时间级别 {min_time_level} 重采样后的数据不可用。终止。")
             return None
        base_index = resampled_ohlcv_dfs[min_time_level].index
        logger.info(f"[{stock_code}] 使用最小时间级别 {min_time_level} 的重采样索引作为合并基准，数量: {len(base_index)}。")

        # 5. 计算所有配置的指标 (并行)
        indicator_calculation_tasks = []

        async def _calculate_single_indicator_async(tf_calc: str, base_df_with_suffix: pd.DataFrame, config_item: Dict) -> Optional[Tuple[str, pd.DataFrame]]:
            if base_df_with_suffix is None or base_df_with_suffix.empty:
                print(f"[{stock_code}] TF {tf_calc}: 基础OHLCV数据为空，无法计算指标 {config_item['name']}")
                return None

            df_for_ta = base_df_with_suffix.copy()
            ohlcv_map_to_std = {
                f'open_{tf_calc}': 'open', f'high_{tf_calc}': 'high', f'low_{tf_calc}': 'low',
                f'close_{tf_calc}': 'close', f'volume_{tf_calc}': 'volume', f'amount_{tf_calc}': 'amount'
            }
            actual_rename_map_to_std = {k: v for k, v in ohlcv_map_to_std.items() if k in df_for_ta.columns}
            df_for_ta.rename(columns=actual_rename_map_to_std, inplace=True)

            # 动态确定指标函数需要的列 (high, low, close, volume, amount)
            # 这是一个简化版本，实际可能需要更复杂的参数签名检查
            required_cols_for_func = set(['high', 'low', 'close']) # 默认需要HLC
            if config_item['name'] in ['MFI', 'OBV', 'VWAP', 'CMF', 'VOL_MA', 'ADL']:
                required_cols_for_func.add('volume')
            if config_item['name'] in ['AMT_MA', 'AROC']:
                required_cols_for_func.add('amount')

            if not all(col in df_for_ta.columns for col in required_cols_for_func):
                missing_cols_str = ", ".join(list(required_cols_for_func - set(df_for_ta.columns)))
                logger.debug(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} 时，df_for_ta 缺少必要列 ({missing_cols_str})。可用: {df_for_ta.columns.tolist()}")
                return None
            try:
                func_params_to_pass = config_item['params'].copy() # 使用为该指标准备好的参数
                # print(f"{stock_code}] TF {tf_calc}: --- df_for_ta.shape: {df_for_ta.shape}, df_for_ta.head(): {df_for_ta.head()}, df_for_ta.isnull().sum()：{df_for_ta.isnull().sum()}")
                indicator_result_df = await config_item['func'](df_for_ta, **func_params_to_pass)

                if indicator_result_df is None or indicator_result_df.empty:
                    logger.debug(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算结果为空。")
                    return None
                # 确保返回的是DataFrame
                if not isinstance(indicator_result_df, pd.DataFrame):
                    logger.warning(f"[{stock_code}] TF {tf_calc}: 指标 {config_item['name']} 计算函数未返回DataFrame (返回类型: {type(indicator_result_df)})。尝试转换。")
                    if isinstance(indicator_result_df, pd.Series):
                        indicator_result_df = indicator_result_df.to_frame()
                    else:
                        return None # 无法处理的返回类型
                result_renamed_df = indicator_result_df.rename(columns=lambda x: f"{x}_{tf_calc}")
                return (tf_calc, result_renamed_df)
            except Exception as e_calc:
                logger.error(f"[{stock_code}] TF {tf_calc}: 计算指标 {config_item['name']} (参数: {config_item['params']}) 时出错: {e_calc}", exc_info=True)
                return None
        for config_item_loop in indicator_configs:
            for tf_conf in config_item_loop['timeframes']:
                if tf_conf in resampled_ohlcv_dfs:
                    base_ohlcv_df_for_tf_loop = resampled_ohlcv_dfs[tf_conf]
                    indicator_calculation_tasks.append(
                        _calculate_single_indicator_async(tf_conf, base_ohlcv_df_for_tf_loop, config_item_loop)
                    )
                else:
                    logger.warning(f"[{stock_code}] 时间框架 {tf_conf} 在 resampled_ohlcv_dfs 中未找到，无法为指标 {config_item_loop['name']} 创建计算任务。")
        calculated_results_tuples = await asyncio.gather(*indicator_calculation_tasks, return_exceptions=True)
        calculated_indicators_by_tf = defaultdict(list)
        for res_tuple_item in calculated_results_tuples:
            if isinstance(res_tuple_item, tuple) and len(res_tuple_item) == 2:
                tf_res, indi_df_res = res_tuple_item
                if indi_df_res is not None and not indi_df_res.empty:
                    calculated_indicators_by_tf[tf_res].append(indi_df_res)
            elif isinstance(res_tuple_item, Exception):
                logger.error(f"[{stock_code}] 一个指标计算任务失败: {res_tuple_item}", exc_info=True)
        # --- 后处理 OBV_MA ---
        obv_ma_period_json = vc_params.get('obv_ma_period', ia_params.get('obv_ma_period', 10)) # 尝试从多个地方获取
        if vc_params.get('enabled', False) or ia_params.get('calculate_obv_ma', True): # 如果启用或需要计算
            for tf_obv_ma, df_list_obv_ma in calculated_indicators_by_tf.items():
                obv_df_with_suffix = next((df_item for df_item in df_list_obv_ma if f'OBV_{tf_obv_ma}' in df_item.columns), None)
                if obv_df_with_suffix is not None:
                    try:
                        obv_ma_series = obv_df_with_suffix[f'OBV_{tf_obv_ma}'].rolling(window=obv_ma_period_json, min_periods=max(1, int(obv_ma_period_json*0.5))).mean()
                        obv_ma_df_res = pd.DataFrame({f'OBV_MA_{obv_ma_period_json}_{tf_obv_ma}': obv_ma_series})
                        calculated_indicators_by_tf[tf_obv_ma].append(obv_ma_df_res)
                        logger.debug(f"[{stock_code}] TF {tf_obv_ma}: OBV_MA_{obv_ma_period_json} 计算完成。")
                    except Exception as e_obvma:
                        logger.error(f"[{stock_code}] TF {tf_obv_ma}: 计算 OBV_MA_{obv_ma_period_json} 出错: {e_obvma}", exc_info=True)
        # 6. 合并所有 DataFrame
        merged_indicators_by_tf = {}
        for tf_merge_indi, df_list_merge_indi in calculated_indicators_by_tf.items():
            if df_list_merge_indi:
                base_df_merge = resampled_ohlcv_dfs.get(tf_merge_indi)
                if base_df_merge is not None:
                    # 使用 reduce 进行链式左合并，以 base_df_merge 为起点
                    # 确保所有 df_list_merge_indi 中的 DataFrame 都有与 base_df_merge 相同的索引
                    # 理论上 _calculate_single_indicator_async 返回的 df 索引与输入一致
                    merged_tf_df_res = base_df_merge.copy() # 从基础OHLCV开始
                    for indi_df_to_merge in df_list_merge_indi:
                        # 检查索引是否一致，如果不同，需要reindex (理论上应该一致)
                        if not merged_tf_df_res.index.equals(indi_df_to_merge.index):
                            logger.warning(f"[{stock_code}] TF {tf_merge_indi}: 指标 {indi_df_to_merge.columns[0].rsplit('_',1)[0]} 与基础数据索引不一致，尝试reindex。")
                            indi_df_to_merge = indi_df_to_merge.reindex(merged_tf_df_res.index) # 简单reindex，可能引入NaN

                        merged_tf_df_res = pd.merge(merged_tf_df_res, indi_df_to_merge, left_index=True, right_index=True, how='left', suffixes=('', f'_dup_{indi_df_to_merge.columns[0].split("_")[-1]}'))
                    merged_indicators_by_tf[tf_merge_indi] = merged_tf_df_res
                else:
                    logger.warning(f"[{stock_code}] TF {tf_merge_indi}: 基础重采样数据丢失，无法合并指标。")
            elif tf_merge_indi in resampled_ohlcv_dfs:
                merged_indicators_by_tf[tf_merge_indi] = resampled_ohlcv_dfs[tf_merge_indi]
        if not merged_indicators_by_tf or min_time_level not in merged_indicators_by_tf:
            logger.error(f"[{stock_code}] 没有可合并的数据，或最小时间级别 {min_time_level} 的数据丢失。")
            return None
        final_merged_df = merged_indicators_by_tf[min_time_level].copy()
        # 对 all_time_levels_needed 进行排序，确保合并顺序一致性，例如按时间级别从小到大
        sorted_time_levels_for_merge = sorted(list(all_time_levels_needed), key=lambda x: self._get_timeframe_in_minutes(x) or float('inf'))
        for tf_final_merge in sorted_time_levels_for_merge:
            if tf_final_merge == min_time_level:
                continue
            if tf_final_merge in merged_indicators_by_tf:
                df_to_merge_final = merged_indicators_by_tf[tf_final_merge].reindex(final_merged_df.index, method='ffill')
                # 在合并前，检查 df_to_merge_final 中是否有与 final_merged_df 中已存在的列重名（除了索引）
                # 这通常发生在不同时间周期的相同基础指标上，例如 close_D 和 close_5
                # 我们期望保留所有这些列，因为它们带有时间后缀
                # pd.merge 的 suffixes 参数在这里可能不是必须的，因为列名已经通过 _tf 后缀区分了
                # 但为了保险，可以保留一个简单的后缀
                final_merged_df = pd.merge(final_merged_df, df_to_merge_final, left_index=True, right_index=True, how='left', suffixes=('_base', f'_other'))
            else:
                logger.warning(f"[{stock_code}] 时间框架 {tf_final_merge} 的合并数据在 merged_indicators_by_tf 中未找到。")
        # --- 7. 计算衍生特征 ---
        logger.info(f"[{stock_code}] 开始计算衍生特征...")
        try:
            # 价格与均线的关系 (EMA, SMA)
            for ma_type_deriv in ['EMA', 'SMA']:
                ma_periods_deriv = fe_params.get(f'{ma_type_deriv.lower()}_periods_for_relation', fe_params.get(f'{ma_type_deriv.lower()}_periods', [])) # 从参数获取
                for tf_str_deriv in all_time_levels_needed:
                    close_col_tf_deriv = f'close_{tf_str_deriv}'
                    if close_col_tf_deriv not in final_merged_df.columns: continue
                    for p_deriv in ma_periods_deriv:
                        ma_col_tf_deriv = f'{ma_type_deriv}_{p_deriv}_{tf_str_deriv}'
                        if ma_col_tf_deriv in final_merged_df.columns:
                            final_merged_df[f'CLOSE_{ma_type_deriv}_RATIO_{p_deriv}_{tf_str_deriv}'] = final_merged_df[close_col_tf_deriv] / final_merged_df[ma_col_tf_deriv]
                            final_merged_df[f'CLOSE_{ma_type_deriv}_NDIFF_{p_deriv}_{tf_str_deriv}'] = (final_merged_df[close_col_tf_deriv] - final_merged_df[ma_col_tf_deriv]) / final_merged_df[ma_col_tf_deriv]
            # 指标的变化率/差分 (RSI, MACDh, MFI, CMF, ADX 等)
            indicators_to_diff = fe_params.get('indicators_for_difference', [
                {'base_name': 'RSI', 'params_key': 'rsi_period', 'default_period': 14},
                {'base_name': 'MACDh', 'params_key': ['macd_fast','macd_slow','macd_signal'], 'default_period': [12,26,9]}, # MACDh 列名较复杂
                {'base_name': 'MFI', 'params_key': 'mfi_period', 'default_period': 14},
                {'base_name': 'CMF', 'params_key': 'cmf_period', 'default_period': 20},
                {'base_name': 'ADX', 'params_key': 'dmi_period', 'default_period': 14},
            ])
            diff_periods = fe_params.get('difference_periods', [1, 2]) # 计算1阶和2阶差分
            for indi_diff_conf in indicators_to_diff:
                base_name = indi_diff_conf['base_name']
                param_keys = indi_diff_conf['params_key']
                default_p_values = indi_diff_conf['default_period']
                for tf_str_diff in all_time_levels_needed:
                    # 构建指标列名
                    param_values_for_col = []
                    if isinstance(param_keys, list): # 多个参数，如MACD
                        for i, pk in enumerate(param_keys):
                            # 尝试从 base_scoring 或 indicator_analysis_params 获取参数值
                            val = bs_params.get(pk, ia_params.get(pk, default_p_values[i]))
                            param_values_for_col.append(str(val))
                        param_str_for_col = "_".join(param_values_for_col)
                        # MACDh 的列名通常是 MACDh_fast_slow_signal
                        indi_col_name_diff = f"{base_name}_{param_str_for_col}_{tf_str_diff}"
                    else: # 单个参数
                        val = bs_params.get(param_keys, ia_params.get(param_keys, default_p_values))
                        param_str_for_col = str(val)
                        indi_col_name_diff = f"{base_name}_{param_str_for_col}_{tf_str_diff}"


                    if indi_col_name_diff in final_merged_df.columns:
                        for diff_p in diff_periods:
                            final_merged_df[f'{base_name}_DIFF{diff_p}_{param_str_for_col}_{tf_str_diff}'] = final_merged_df[indi_col_name_diff].diff(diff_p)
                            # final_merged_df[f'{base_name}_PCTCHG{diff_p}_{param_str_for_col}_{tf_str_diff}'] = final_merged_df[indi_col_name_diff].pct_change(diff_p) # 百分比变化可能导致inf
            # 价格在布林带/肯特纳通道中的位置
            # 布林带
            boll_period_deriv = bs_params.get('boll_period', default_boll_p['period'])
            boll_std_deriv = bs_params.get('boll_std_dev', default_boll_p['std_dev'])
            for tf_str_deriv_ch in all_time_levels_needed:
                close_col_ch = f'close_{tf_str_deriv_ch}'
                lower_b_col = f'BBL_{boll_period_deriv}_{boll_std_deriv:.1f}_{tf_str_deriv_ch}'
                upper_b_col = f'BBU_{boll_period_deriv}_{boll_std_deriv:.1f}_{tf_str_deriv_ch}'
                middle_b_col = f'BBM_{boll_period_deriv}_{boll_std_deriv:.1f}_{tf_str_deriv_ch}'
                if all(c in final_merged_df.columns for c in [close_col_ch, lower_b_col, upper_b_col, middle_b_col]):
                    band_width_b = final_merged_df[upper_b_col] - final_merged_df[lower_b_col]
                    final_merged_df[f'CLOSE_BB_POS_{boll_period_deriv}_{tf_str_deriv_ch}'] = np.where(
                        band_width_b > 1e-9,
                        (final_merged_df[close_col_ch] - final_merged_df[lower_b_col]) / band_width_b, 0.5
                    )
            # 肯特纳通道 (类似处理)
            kc_ema_p_deriv = fe_params.get('kc_ema_period', default_kc_p['ema_period'])
            kc_atr_p_deriv = fe_params.get('kc_atr_period', default_kc_p['atr_period'])
            for tf_str_deriv_ch_kc in all_time_levels_needed:
                close_col_kc = f'close_{tf_str_deriv_ch_kc}'
                lower_kc_col = f'KCL_{kc_ema_p_deriv}_{kc_atr_p_deriv}_{tf_str_deriv_ch_kc}'
                upper_kc_col = f'KCU_{kc_ema_p_deriv}_{kc_atr_p_deriv}_{tf_str_deriv_ch_kc}'
                if all(c in final_merged_df.columns for c in [close_col_kc, lower_kc_col, upper_kc_col]):
                    band_width_kc = final_merged_df[upper_kc_col] - final_merged_df[lower_kc_col]
                    final_merged_df[f'CLOSE_KC_POS_{kc_ema_p_deriv}_{kc_atr_p_deriv}_{tf_str_deriv_ch_kc}'] = np.where(
                        band_width_kc > 1e-9,
                        (final_merged_df[close_col_kc] - final_merged_df[lower_kc_col]) / band_width_kc, 0.5
                    )

            logger.info(f"[{stock_code}] 衍生特征计算完成。")
        except Exception as e_deriv:
            logger.error(f"[{stock_code}] 计算衍生特征时出错: {e_deriv}", exc_info=True)
        # 8. 最终填充
        nan_before_final_fill = final_merged_df.isnull().sum().sum()
        final_merged_df.ffill(inplace=True)
        final_merged_df.bfill(inplace=True)
        nan_after_final_fill = final_merged_df.isnull().sum().sum()
        logger.info(f"[{stock_code}] 最终填充完成。填充前 NaN 总数: {nan_before_final_fill}, 填充后 NaN 总数: {nan_after_final_fill}")
        if nan_after_final_fill > 0:
            nan_cols_summary = final_merged_df.isnull().sum()
            nan_cols_summary = nan_cols_summary[nan_cols_summary > 0].sort_values(ascending=False)
            logger.warning(f"[{stock_code}] 最终填充后仍存在 NaN 的列 (前10条): {nan_cols_summary.head(10).to_dict()}")
            # 对于完全是NaN的列，可以考虑填充0或移除，但需谨慎
            # final_merged_df.fillna(0, inplace=True) # 强制填充所有剩余NaN为0
        if final_merged_df.empty:
            logger.error(f"[{stock_code}] 最终合并和填充后的 DataFrame 为空。")
            return None
        logger.info(f"[{stock_code}] 策略数据准备完成，最终 DataFrame 形状: {final_merged_df.shape}, 列数: {len(final_merged_df.columns)}")
        logger.debug(f"[{stock_code}] 最终 DataFrame 列名 (部分): {final_merged_df.columns.tolist()[:30]}...") # 打印更多列名
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
        if df is None or df.empty or not all(c in df.columns for c in [high_col, low_col, close_col]): return None
        try:
            # MODIFIED: 添加日志，打印 pandas_ta.dm 的输入 DataFrame 信息
            print(f"calculate_dmi input shape: {df.shape}, head: {df.head()}, isnull().sum(): {df.isnull().sum()}") # 如果需要，可以再次打印输入信息

            dmi_df = df.ta.dm(high=df[high_col], low=df[low_col], close=df[close_col], length=period, append=False)

            # MODIFIED: 添加日志，打印 pandas_ta.dm 的输出 DataFrame 信息
            if dmi_df is not None:
                print(f"calculate_dmi (period {period}) pandas_ta output columns: {dmi_df.columns.tolist()}") # MODIFIED: 打印 pandas_ta 输出的列名
                print(f"calculate_dmi (period {period}) pandas_ta output isnull().sum(): {dmi_df.isnull().sum()}") # MODIFIED: 打印 pandas_ta 输出的 NaN 统计
            else:
                 logger.warning(f"calculate_dmi (period {period}) pandas_ta.dm 返回 None 或空。") # MODIFIED: 增加返回 None/空的日志

            if dmi_df is None or dmi_df.empty: return None
            # pandas_ta 列名: DMP_period (DI+), DMN_period (DI-), ADX_period
            rename_map = {}
            if f'DMP_{period}' in dmi_df.columns: rename_map[f'DMP_{period}'] = f'PDI_{period}' # Positive DI
            if f'DMN_{period}' in dmi_df.columns: rename_map[f'DMN_{period}'] = f'NDI_{period}' # Negative DI
            # MODIFIED: 检查 ADX 列是否存在并添加到重命名映射
            adx_col_name_ta = f'ADX_{period}'
            if adx_col_name_ta in dmi_df.columns:
                 rename_map[adx_col_name_ta] = adx_col_name_ta # ADX 通常不变
            else:
                 # MODIFIED: 如果 ADX 列不存在，记录警告
                 logger.warning(f"calculate_dmi (period {period}) pandas_ta.dm 返回的 DataFrame 中缺少 ADX 列 ({adx_col_name_ta})。")


            # MODIFIED: 仅对需要重命名的列进行操作，并确保返回的 DataFrame 包含所有原始列（如果需要）
            # pandas_ta.dm 默认 append=False，只返回指标列。这里只需要返回指标列。
            # 确保返回的 DataFrame 包含 PDI, NDI, ADX (如果存在)
            output_cols = [f'PDI_{period}', f'NDI_{period}']
            if adx_col_name_ta in dmi_df.columns:
                 output_cols.append(adx_col_name_ta)

            # 选择并重命名列
            # 确保只选择实际存在于 dmi_df.columns 且在 rename_map 中的列
            cols_to_select = [ta_col for ta_col, final_col in rename_map.items() if ta_col in dmi_df.columns]
            if not cols_to_select:
                 logger.warning(f"calculate_dmi (period {period}) pandas_ta.dm 返回的 DataFrame 中没有找到任何预期的指标列。")
                 return None # 没有找到任何指标列，返回 None

            result_df = dmi_df[cols_to_select].rename(columns=rename_map)

            # 检查结果 DataFrame 是否包含所有必需的评分列 (PDI, NDI, ADX)
            # 注意：这里检查的是 calculate_dmi 返回的 DataFrame，它应该包含 PDI, NDI, ADX (如果计算成功)
            # 评分函数需要的是 PDI, NDI, ADX Series，这些 Series 的列名在 calculate_all_indicator_scores 中会被映射
            # calculate_dmi 返回的 DataFrame 列名应该是 PDI_period, NDI_period, ADX_period
            expected_return_cols = [f'PDI_{period}', f'NDI_{period}']
            if adx_col_name_ta in dmi_df.columns: # 只有当 pandas_ta 返回了 ADX，我们才期望 calculate_dmi 返回它
                 expected_return_cols.append(adx_col_name_ta) # ADX 列名在 calculate_dmi 返回时通常不变

            # 检查 result_df 是否包含所有 expected_return_cols
            missing_return_cols = [col for col in expected_return_cols if col not in result_df.columns]
            if missing_return_cols:
                 logger.warning(f"calculate_dmi (period {period}) 返回的 DataFrame 缺少预期的列: {missing_return_cols}. 返回的列: {result_df.columns.tolist()}")
                 # 即使缺少 ADX，只要 PDI/NDI 存在，仍然返回，让评分函数处理缺失
                 # 如果 PDI 或 NDI 缺失，则返回 None
                 if f'PDI_{period}' not in result_df.columns or f'NDI_{period}' not in result_df.columns:
                      logger.error(f"calculate_dmi (period {period}) 返回的 DataFrame 缺少必需的 PDI 或 NDI 列。")
                      return None


            return result_df # 返回包含 PDI, NDI, ADX (如果存在) 的 DataFrame

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






