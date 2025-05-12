# strategies\utils\strategy_utils.py
from collections import defaultdict
import json
import os
from django.conf import settings
import pandas as pd
import numpy as np
import pandas_ta as ta
import re
import logging
from scipy.signal import find_peaks # 需要 scipy
import warnings
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("strategy_utils")

# --- 忽略特定警告 ---
warnings.filterwarnings(action='ignore', category=UserWarning, message='.*pandas_ta.*might not be installed.*')
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='.*invalid value encountered in scalar divide.*')
warnings.filterwarnings(action='ignore', category=RuntimeWarning, message='.*Mean of empty slice.*')
warnings.filterwarnings(action='ignore', category=FutureWarning, message='.*Passing method to Float64Index.*')
warnings.filterwarnings(action='ignore', category=FutureWarning, message='.*use_inf_as_na option is deprecated*')
pd.options.mode.chained_assignment = None # default='warn'

# --- 辅助函数区 ---

def _get_timeframe_in_minutes(tf_str: str) -> Optional[int]:
    """
    将时间级别字符串（如 '5', '15', 'D', 'W', 'M'）转换为近似的分钟数。
    此函数与 indicator_services 中的同名函数同步，确保一致性。
    注意：'D', 'W', 'M' 是基于标准交易时间的估算。
    """
    tf_str_upper = str(tf_str).upper() # 转换为大写以便处理 'd', 'w', 'm'
    if tf_str_upper.isdigit():
        return int(tf_str_upper)
    elif tf_str_upper == 'D':
        return 240 # A股主要交易时间 4 小时 * 60 分钟/小时
    elif tf_str_upper == 'W':
        return 240 * 5 # 每周 5 个交易日
    elif tf_str_upper == 'M':
        # 月度交易日数不固定，使用一个近似值，例如 21 天 * 4小时/天 * 60分钟/小时
        return 240 * 21
    else:
        logger.warning(f"无法将时间级别 '{tf_str}' 转换为分钟数，将返回 None。")
        return None

def get_find_peaks_params(time_level: str, base_lookback: int) -> Dict[str, Any]:
    """
    根据时间级别和基础回看期，返回适用于 find_peaks 的参数。
    短周期更敏感，长周期过滤更多噪音。
    """
    tf_minutes = _get_timeframe_in_minutes(time_level)
    if tf_minutes is None:
        logger.warning(f"无法确定时间级别 {time_level} 的分钟数，使用默认 find_peaks 参数。")
        # 确保默认参数的合理性
        distance = max(3, base_lookback // 3 if base_lookback > 0 else 3)
        prominence_factor = 0.3
        width = max(1, base_lookback // 6 if base_lookback > 0 else 1)
        return {'distance': distance, 'prominence_factor': prominence_factor, 'width': width}

    # 根据分钟数调整基础参数
    if tf_minutes <= 5: # 1-5分钟
        distance_factor = 4
        prominence_factor = 0.2
        width_factor = 4
    elif tf_minutes <= 15: # 6-15分钟
        distance_factor = 3
        prominence_factor = 0.3
        width_factor = 3
    elif tf_minutes <= 60: # 16-60分钟
        distance_factor = 2.5
        prominence_factor = 0.4
        width_factor = 2.5
    elif tf_minutes <= 240: # 61-240分钟 (日内更高周期)
        distance_factor = 2
        prominence_factor = 0.5
        width_factor = 2
    else: # 日线及以上 (D, W, M)
        distance_factor = 1.5
        prominence_factor = 0.8
        width_factor = 1.5

    # 确保 base_lookback > 0
    safe_lookback = max(1, base_lookback)
    distance = max(2, int(safe_lookback / distance_factor))
    width = max(1, int(distance / width_factor)) # 保持原逻辑，宽度依赖于计算出的distance

    if tf_minutes <= 5:
        distance = max(distance, 5)
        width = max(width, 2)

    params = {
        'distance': distance,
        'prominence_factor': prominence_factor,
        'width': width
    }
    logger.debug(f"时间级别 {time_level} ({tf_minutes}分钟)，基础回看期 {base_lookback}，生成 find_peaks 参数: {params}")
    return params

def find_divergence_for_indicator(price_series: pd.Series,
                                  indicator_series: pd.Series,
                                  lookback: int,
                                  find_peaks_params: Dict[str, Any],
                                  check_regular_bullish: bool,
                                  check_regular_bearish: bool,
                                  check_hidden_bullish: bool,
                                  check_hidden_bearish: bool
                                  ) -> pd.DataFrame:
    """
    辅助函数：检测单个价格序列与指标序列之间的背离。
    优化：
    - 移除 check_divergence_pairs 中未使用的 indicator_matches 参数。
    - 确保 prominence 计算的 min_periods 合理性。
    """
    result_df = pd.DataFrame({
        'regular_bullish': 0,
        'regular_bearish': 0,
        'hidden_bullish': 0,
        'hidden_bearish': 0
    }, index=price_series.index)

    min_data_points = max(lookback * 2, 30) # 确保 lookback 大于0
    if price_series.isnull().all() or indicator_series.isnull().all() or len(price_series) < min_data_points:
        logger.debug(f"数据不足或全为 NaN (价格: {len(price_series)}, 指标: {len(indicator_series)}, 最小需要: {min_data_points})，无法检测背离。")
        return result_df

    if not isinstance(price_series, pd.Series) or not isinstance(indicator_series, pd.Series) or not price_series.index.equals(indicator_series.index):
         logger.error("价格序列或指标序列不是 pandas Series，或索引不一致。")
         return result_df

    indicator_filled = indicator_series.ffill().bfill()
    if indicator_filled.isnull().all():
        logger.debug("填充后的指标序列全为 NaN，无法检测背离。")
        return result_df

    distance = find_peaks_params.get('distance', max(3, lookback // 3 if lookback > 0 else 3))
    width = find_peaks_params.get('width', max(1, distance // 2 if distance > 0 else 1)) # 确保 distance > 0
    prominence_factor = find_peaks_params.get('prominence_factor', 0.3)

    # 确保 lookback > 0 用于 rolling
    safe_lookback_rolling = max(1, lookback)
    min_periods_rolling = max(1, int(safe_lookback_rolling * 0.5))

    min_prominence_price_series = (price_series.rolling(safe_lookback_rolling, min_periods=min_periods_rolling).std() * prominence_factor).fillna(0).replace([np.inf, -np.inf], 0)
    min_prominence_indicator_series = (indicator_filled.rolling(safe_lookback_rolling, min_periods=min_periods_rolling).std() * prominence_factor).fillna(0).replace([np.inf, -np.inf], 0)

    min_prominence_price = np.maximum(min_prominence_price_series.values, 1e-9)
    min_prominence_indicator = np.maximum(min_prominence_indicator_series.values, 1e-9)

    try:
        price_peaks_indices, _ = find_peaks(price_series.values, distance=distance, prominence=min_prominence_price, width=width)
        indicator_peaks_indices, _ = find_peaks(indicator_filled.values, distance=distance, prominence=min_prominence_indicator, width=width)
        price_troughs_indices, _ = find_peaks(-price_series.values, distance=distance, prominence=min_prominence_price, width=width)
        indicator_troughs_indices, _ = find_peaks(-indicator_filled.values, distance=distance, prominence=min_prominence_indicator, width=width)
    except Exception as fp_err:
        logger.warning(f"查找峰值/谷值时出错: {fp_err}。跳过此指标的背离检测。")
        return result_df

    def find_matching_indicator_extremums(price_extremum_indices: np.ndarray, indicator_extremum_indices: np.ndarray, window: int) -> List[Tuple[int, Optional[int]]]:
        matches = []
        if len(indicator_extremum_indices) == 0: # 如果指标没有极值点，直接返回空匹配
            for p_idx in price_extremum_indices:
                matches.append((p_idx, None))
            return matches

        for p_idx in price_extremum_indices:
            lower_bound_idx = np.searchsorted(indicator_extremum_indices, p_idx - window, side='left')
            upper_bound_idx = np.searchsorted(indicator_extremum_indices, p_idx + window, side='right')
            nearby_indicator_indices = indicator_extremum_indices[lower_bound_idx:upper_bound_idx]

            if len(nearby_indicator_indices) > 0:
                closest_i_idx = nearby_indicator_indices[np.abs(nearby_indicator_indices - p_idx).argmin()]
                matches.append((p_idx, closest_i_idx))
            else:
                matches.append((p_idx, None))
        return matches
    
    def check_divergence_pairs(price_matches: List[Tuple[int, Optional[int]]], is_peak: bool, div_type: str ):
        sorted_price_matches = sorted([m for m in price_matches if m[1] is not None], key=lambda x: x[0])
        if len(sorted_price_matches) < 2:
            return
        last_idx = len(price_series) - 1
        safe_lookback_check = max(0, lookback) # 确保 lookback 不为负
        for k in range(len(sorted_price_matches) - 1):
            p1_idx, i1_idx = sorted_price_matches[k]
            p2_idx, i2_idx = sorted_price_matches[k+1]
            # i1_idx 和 i2_idx 必然不为 None，因为上面已经过滤了
            if i1_idx >= i2_idx or p2_idx < last_idx - safe_lookback_check:
                continue
            price1, price2 = price_series.iloc[p1_idx], price_series.iloc[p2_idx]
            indicator1, indicator2 = indicator_filled.iloc[i1_idx], indicator_filled.iloc[i2_idx]
            if pd.isna(price1) or pd.isna(price2) or pd.isna(indicator1) or pd.isna(indicator2):
                 continue
            signal_value = 0
            if is_peak:
                if div_type == 'regular_bearish' and price2 > price1 and indicator2 < indicator1:
                    signal_value = -1
                elif div_type == 'hidden_bearish' and price2 < price1 and indicator2 > indicator1:
                    signal_value = -1
            else:
                if div_type == 'regular_bullish' and price2 < price1 and indicator2 > indicator1:
                    signal_value = 1
                elif div_type == 'hidden_bullish' and price2 > price1 and indicator2 < indicator1:
                    signal_value = 1
            if signal_value != 0:
                 target_index = price_series.index[p2_idx]
                 current_signal = result_df.loc[target_index, div_type]
                 if abs(signal_value) > abs(current_signal) or current_signal == 0:
                      result_df.loc[target_index, div_type] = signal_value
    price_peak_matches = find_matching_indicator_extremums(price_peaks_indices, indicator_peaks_indices, distance)
    price_trough_matches = find_matching_indicator_extremums(price_troughs_indices, indicator_troughs_indices, distance)
    if check_regular_bearish:
        check_divergence_pairs(price_peak_matches, is_peak=True, div_type='regular_bearish')
    if check_hidden_bearish:
        check_divergence_pairs(price_peak_matches, is_peak=True, div_type='hidden_bearish')
    if check_regular_bullish:
        check_divergence_pairs(price_trough_matches, is_peak=False, div_type='regular_bullish')
    if check_hidden_bullish:
        check_divergence_pairs(price_trough_matches, is_peak=False, div_type='hidden_bullish')
    return result_df.fillna(0).astype(int)
    
def detect_divergence(data: pd.DataFrame, dd_params: Dict, indicator_configs: List[Dict]) -> pd.DataFrame:
    """
    检测价格与多个指定指标之间的常规和隐藏背离。
    优化：
    - 简化指标列名查找逻辑，使其更健壮。
    - 确保 lookback > 0 传递给 get_find_peaks_params。
    """
    all_divergence_signals = pd.DataFrame(index=data.index)
    all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = False
    all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = False

    if not dd_params.get('enabled', False):
        logger.info("参数中已禁用背离检测。")
        return all_divergence_signals

    timeframes_to_check = dd_params.get('timeframes', [])
    price_type = dd_params.get('price_type', 'close')
    lookback = dd_params.get('lookback', 14)
    safe_lookback = max(1, lookback) # 确保 lookback > 0

    base_find_peaks_params = dd_params.get('find_peaks_params', {})
    check_regular_bullish = dd_params.get('check_regular_bullish', True)
    check_regular_bearish = dd_params.get('check_regular_bearish', True)
    check_hidden_bullish = dd_params.get('check_hidden_bullish', True)
    check_hidden_bearish = dd_params.get('check_hidden_bearish', True)
    indicators_to_check = dd_params.get('indicators', {})

    if not timeframes_to_check:
        logger.warning("未指定用于背离检测的时间框架列表 (dd_params['timeframes'])。")
        return all_divergence_signals

    # 构建指标列名查找字典: {(indicator_base_name, tf): full_column_name}
    # indicator_configs 的结构: [{'name': 'RSI', 'params': {'length': 14}, 'timeframes': ['D', '60'], 'func': ...}, ...]
    # 生成的列名可能是: RSI_14_D, MACDh_12_26_9_60
    # 我们需要从 indicator_configs 更精确地构建这个映射
    col_name_map = {}
    for config in indicator_configs:
        base_name = config['name']
        params_str = "_".join(map(str, config.get('params', {}).values())) if config.get('params') else ""
        
        for tf_item in config['timeframes']: # tf_item could be str or dict
            current_tf = tf_item if isinstance(tf_item, str) else tf_item.get('tf', '')
            if not current_tf: continue

            # 构建标准列名
            # 对于 MACD，它会生成 MACD, MACDs, MACDh
            if base_name == 'MACD':
                # MACD_fast_slow_signal_tf
                # MACDs_fast_slow_signal_tf
                # MACDh_fast_slow_signal_tf
                macd_params_str = "_".join(map(str, [
                    config['params'].get('fast', 12),
                    config['params'].get('slow', 26),
                    config['params'].get('signal', 9)
                ]))
                col_name_map[('MACD', current_tf)] = f"MACD_{macd_params_str}_{current_tf}"
                col_name_map[('MACDs', current_tf)] = f"MACDs_{macd_params_str}_{current_tf}"
                col_name_map[('MACDh', current_tf)] = f"MACDh_{macd_params_str}_{current_tf}"
                col_name_map[('macd_hist', current_tf)] = f"MACDh_{macd_params_str}_{current_tf}" # alias for macd_hist
            else:
                col_name = f"{base_name}"
                if params_str:
                    col_name += f"_{params_str}"
                col_name += f"_{current_tf}"
                col_name_map[(base_name, current_tf)] = col_name
                # 兼容一些常见的 key，例如 rsi -> RSI
                col_name_map[(base_name.lower(), current_tf)] = col_name


    indicator_key_to_base_name_map = {
        'macd_hist': 'MACDh', 'rsi': 'RSI', 'mfi': 'MFI', 'obv': 'OBV',
        'cci': 'CCI', 'cmf': 'CMF', 'stoch_k': 'STOCHk', 'stoch_d': 'STOCHd',
        'stoch_j': 'J', 'roc': 'ROC', 'adx': 'ADX', 'pdi': 'PDI',
        'ndi': 'NDI', 'bbp': 'BBP'
        # Add other direct mappings if indicator_key in dd_params is different from base_name
    }

    for indicator_key, enabled in indicators_to_check.items():
        if not enabled:
            continue

        actual_base_name = indicator_key_to_base_name_map.get(indicator_key, indicator_key.upper())

        for tf_check in timeframes_to_check:
            price_col = f'{price_type}_{tf_check}'
            if price_col not in data.columns or data[price_col].isnull().all():
                logger.warning(f"价格列 '{price_col}' (TF {tf_check}) 不存在或全为 NaN。跳过。")
                continue
            price_series = data[price_col]

            indicator_col = col_name_map.get((actual_base_name, tf_check))
            # Fallback for cases where params might not be perfectly matched in col_name_map construction
            if not indicator_col or indicator_col not in data.columns:
                potential_cols = [col for col in data.columns if col.startswith(f"{actual_base_name}_") and col.endswith(f"_{tf_check}")]
                if potential_cols:
                    indicator_col = max(potential_cols, key=len) # Choose longest, assuming more params
                elif f"{actual_base_name}_{tf_check}" in data.columns: # For indicators without params like OBV_D
                     indicator_col = f"{actual_base_name}_{tf_check}"


            if indicator_col is None or indicator_col not in data.columns or data[indicator_col].isnull().all():
                logger.warning(f"指标 '{indicator_key}' (基名: {actual_base_name}) 在 TF {tf_check} 的列 '{indicator_col}' 未找到、全为 NaN 或未启用。跳过。")
                continue
            indicator_series = data[indicator_col]

            current_find_peaks_params = get_find_peaks_params(tf_check, safe_lookback) # Use safe_lookback
            current_find_peaks_params.update(base_find_peaks_params) # User-defined params override generated ones

            logger.debug(f"开始检测 TF {tf_check}: 价格 ('{price_col}') 与指标 ('{indicator_col}') 的背离 (lookback: {safe_lookback})...")
            div_result = find_divergence_for_indicator(
                price_series=price_series,
                indicator_series=indicator_series,
                lookback=safe_lookback, # Pass safe_lookback
                find_peaks_params=current_find_peaks_params,
                check_regular_bullish=check_regular_bullish,
                check_regular_bearish=check_regular_bearish,
                check_hidden_bullish=check_hidden_bullish,
                check_hidden_bearish=check_hidden_bearish
            )

            for div_type_col_name in div_result.columns: # e.g., 'regular_bullish'
                # 从 indicator_col (e.g., RSI_14_D) 提取指标名和参数部分
                parts = indicator_col.split('_')
                indi_name_from_col = parts[0] # RSI
                params_and_tf_from_col = "_".join(parts[1:]) # 14_D
                
                # 清理 div_type_col_name, e.g., 'regular_bullish' -> 'RegularBullish'
                clean_div_type = "".join(word.capitalize() for word in div_type_col_name.split('_'))

                detailed_col_name = f'DIV_{indi_name_from_col}_{params_and_tf_from_col}_{clean_div_type}'
                all_divergence_signals[detailed_col_name.upper()] = div_result[div_type_col_name]

    bullish_cols = [col for col in all_divergence_signals.columns if 'BULLISH' in col.upper() and col not in ['HAS_BULLISH_DIVERGENCE', 'HAS_BEARISH_DIVERGENCE']]
    if bullish_cols:
        all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = (all_divergence_signals[bullish_cols] > 0).any(axis=1)

    bearish_cols = [col for col in all_divergence_signals.columns if 'BEARISH' in col.upper() and col not in ['HAS_BULLISH_DIVERGENCE', 'HAS_BEARISH_DIVERGENCE']]
    if bearish_cols:
        all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = (all_divergence_signals[bearish_cols] < 0).any(axis=1)

    for col in all_divergence_signals.columns:
        if all_divergence_signals[col].dtype == 'bool':
            all_divergence_signals[col] = all_divergence_signals[col].fillna(False)
        else:
            all_divergence_signals[col] = all_divergence_signals[col].fillna(0)
    
    # 确保聚合列也是 bool 类型
    all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = all_divergence_signals['HAS_BULLISH_DIVERGENCE'].astype(bool)
    all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = all_divergence_signals['HAS_BEARISH_DIVERGENCE'].astype(bool)


    logger.info(f"背离检测完成。共生成 {len(all_divergence_signals.columns) - 2} 个详细信号列。")
    return all_divergence_signals

def detect_kline_patterns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    检测 K 线形态，使用指定时间框架的 OHLC 数据。
    优化：
    - 使用直接布尔索引 patterns_filtered.loc[condition, col_name] = value
    - 简化部分条件判断，利用向量化特性。
    - 确保滚动平均的 min_periods 合理性。
    - 修正 Doji 判断逻辑。
    - 移除了未使用的变量 full_range2, upper_shadow1, lower_shadow1。
    :param df: 包含原始数据的 DataFrame，列名应包含特定时间框架的 OHLCV 数据 (例如 'open_D', 'high_D', 'low_D', 'close_D', 'volume_D')。
    :param tf: 当前要检测的时间框架字符串 (例如 'D', '60', '15')。
    :return: 返回一个 DataFrame，索引与输入 df 一致。每列代表一个K线形态 (例如 'KAP_BULLISHENGULFING_D')，
             值为 1 表示看涨形态，-1 表示看跌形态，0 表示无此形态。
    """
    # 定义K线形态检测所需的基础列名
    required_cols_base = ['open', 'high', 'low', 'close', 'volume']
    # 根据时间框架(tf)构建实际需要的列名
    required_cols_tf = [f"{col}_{tf}" for col in required_cols_base]

    # 定义所有将要检测的K线形态名称，用于预先创建结果DataFrame的列
    pattern_names = [
        'BullishEngulfing', 'BearishEngulfing', 'Hammer', 'HangingMan',
        'MorningStar', 'EveningStar', 'PiercingLine', 'DarkCloudCover', 'Doji',
        'ThreeWhiteSoldiers', 'ThreeBlackCrows', 'BullishHarami', 'BearishHarami',
        'BullishHaramiCross', 'BearishHaramiCross', 'TweezerBottom', 'TweezerTop',
        'BullishMarubozu', 'BearishMarubozu', 'RisingThreeMethods', 'FallingThreeMethods',
        'UpsideGapTwoCrows', 'UpsideTasukiGap', 'DownsideTasukiGap',
        'BullishSeparatingLines', 'BearishSeparatingLines', 'BullishCounterattack', 'BearishCounterattack'
    ]
    # 初始化默认的结果DataFrame，所有形态信号默认为0，索引与输入df一致
    default_pattern_df = pd.DataFrame(0, index=df.index, columns=[f'KAP_{name.upper()}_{tf}' for name in pattern_names])

    # 检查所需列是否存在于输入DataFrame中
    if not all(col in df.columns for col in required_cols_tf):
        logger.warning(f"K 线形态检测 TF {tf} 所需列不完整: {required_cols_tf}。跳过此时间框架的K线形态检测。")
        return default_pattern_df # 返回全为0的默认DataFrame

    # 准备数据：复制所需OHLCV列，并重命名为通用名称（open, high, low, close, volume）以方便后续代码引用
    ohlcv_tf = df[required_cols_tf].copy()
    ohlcv_tf.columns = required_cols_base

    # 过滤掉 OHLC 数据中存在 NaN 值的行，仅对有效数据进行计算
    # df_calc_idx 保存了有效数据行的索引，后续计算结果将基于此索引
    df_calc_idx = ohlcv_tf[['open', 'high', 'low', 'close']].dropna().index
    
    # 如果有效数据点过少（少于5个），很多多K线组合形态无法检测，记录警告
    # 即使数据不足，也返回定义好的、全为0的DataFrame结构，以保证后续流程的统一性
    if len(df_calc_idx) < 5:
        logger.warning(f"TF {tf} 有效数据点不足 (<5)，部分K线形态可能无法准确检测。当前有效数据量: {len(df_calc_idx)}")
        if len(df_calc_idx) == 0: # 如果完全没有有效数据
            return default_pattern_df

    # 从过滤后的有效数据中提取 OHLCV Series
    o, h, l, c, v = (ohlcv_tf[col].loc[df_calc_idx] for col in required_cols_base)

    # --- 计算K线基础属性 ---
    # 前N个周期的OHLCV数据 (shift操作)
    o1, h1, l1, c1, v1 = (s.shift(1) for s in (o, h, l, c, v)) # 前1周期
    o2, h2, l2, c2, v2 = (s.shift(2) for s in (o, h, l, c, v)) # 前2周期
    o3, h3, l3, c3 = (s.shift(3) for s in (o, h, l, c))       # 前3周期 (成交量v3, v4在原逻辑中未使用)
    o4, h4, l4, c4 = (s.shift(4) for s in (o, h, l, c))       # 前4周期

    # K线实体大小 (当前及前N周期)
    body = abs(c - o)
    body1 = abs(c1 - o1).fillna(0) # shift后的NaN用0填充
    body2 = abs(c2 - o2).fillna(0)
    body3 = abs(c3 - o3).fillna(0)
    body4 = abs(c4 - o4).fillna(0)

    # K线整体波幅 (最高价-最低价)，避免除以零
    full_range = (h - l).replace(0, 1e-9)
    full_range1 = (h1 - l1).fillna(0).replace(0, 1e-9) # 前1周期波幅
    # full_range2 已被移除，因未使用

    # 上下影线长度 (当前周期)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(o, c) - l
    # upper_shadow1, lower_shadow1 已被移除，因未使用

    # 判断K线颜色 (阳线/阴线，当前及前N周期)
    # shift后的布尔序列NaN用False填充，表示非阳/非阴
    is_green = c > o
    is_red = c < o
    is_green1 = (c1 > o1).fillna(False)
    is_red1 = (c1 < o1).fillna(False)
    is_green2 = (c2 > o2).fillna(False)
    is_red2 = (c2 < o2).fillna(False)
    is_green3 = (c3 > o3).fillna(False)
    is_red3 = (c3 < o3).fillna(False)
    is_green4 = (c4 > o4).fillna(False)
    is_red4 = (c4 < o4).fillna(False)

    # 计算滚动平均实体大小和滚动平均波幅，用于形态判断的相对阈值
    # 确保滚动窗口大小不超过数据长度
    rolling_window = min(len(body), 20) # 使用最近20根K线（或实际可用K线数）
    min_periods_rolling = max(1, int(rolling_window * 0.3)) if rolling_window > 0 else 1 # 至少需要窗口期30%的数据
    
    # 平均实体，处理NaN和0值
    avg_body_val = body.rolling(rolling_window, min_periods=min_periods_rolling).mean()
    avg_body = avg_body_val.fillna(body.mean() if not body.empty else 1e-9).replace(0, 1e-9)

    # 平均波幅，处理NaN和0值
    avg_range_val = full_range.rolling(rolling_window, min_periods=min_periods_rolling).mean()
    avg_range = avg_range_val.fillna(full_range.mean() if not full_range.empty else 1e-9).replace(0, 1e-9)

    # 初始化用于计算的形态结果DataFrame，索引为有效数据索引，所有形态信号默认为0
    patterns_calc = pd.DataFrame(0, index=df_calc_idx, columns=default_pattern_df.columns)

    # --- 开始检测各种K线形态 ---
    # 使用向量化布尔索引 ( patterns_calc.loc[condition, column_name] = signal_value ) 进行赋值

    # 1. 十字星 (Doji)
    # 定义：实体长度远小于K线整体波幅，且实体本身不能过小（相对于波幅）。
    doji_body_max_ratio_of_range = 0.1  # 实体最大占全天波幅的10%
    doji_body_min_abs_ratio_of_range = 0.01 # 实体至少是全天波幅的1% (避免0实体或极小实体被误判)
    is_doji = (body <= full_range * doji_body_max_ratio_of_range) & \
              (body >= full_range * doji_body_min_abs_ratio_of_range)
    # Doji信号：阳十字为1，阴十字为-1 (或统一为1表示出现，方向由上下文判断)
    patterns_calc.loc[is_doji, f'KAP_DOJI_{tf}'] = np.where(is_green[is_doji], 1, -1)

    # 2. 吞没形态 (Engulfing)
    # 看涨吞没: 前阴，今阳，今日阳线实体完全吞没昨日阴线实体
    bull_engulf = is_red1 & is_green & (c > o1) & (o < c1) & (body > body1 * 1.01) # 今日实体比昨日实体大1%以上
    patterns_calc.loc[bull_engulf, f'KAP_BULLISHENGULFING_{tf}'] = 1
    # 看跌吞没: 前阳，今阴，今日阴线实体完全吞没昨日阳线实体
    bear_engulf = is_green1 & is_red & (o > c1) & (c < o1) & (body > body1 * 1.01)
    patterns_calc.loc[bear_engulf, f'KAP_BEARISHENGULFING_{tf}'] = -1

    # 3. 锤子线 (Hammer) / 上吊线 (Hanging Man)
    # 定义：实体小，下影线长（通常为实体的2倍以上），上影线短或无。
    small_body_max_ratio_of_range = 0.3       # 小实体定义：实体最大占全天波幅的30%
    long_lower_shadow_min_ratio_of_body = 2.0 # 长下影线定义：下影线至少是实体的2倍
    short_upper_shadow_max_ratio_of_body = 0.5# 短上影线定义：上影线最多是实体的0.5倍
    
    hammer_like_cond = (body > 1e-9) & \
                       (body <= full_range * small_body_max_ratio_of_range) & \
                       (lower_shadow >= body * long_lower_shadow_min_ratio_of_body) & \
                       (upper_shadow < body * short_upper_shadow_max_ratio_of_body)
    
    # 锤子线：出现在下跌趋势后（简化为前一根K线是阴线）
    is_hammer = hammer_like_cond & is_red1
    patterns_calc.loc[is_hammer, f'KAP_HAMMER_{tf}'] = 1
    # 上吊线：出现在上涨趋势后（简化为前一根K线是阳线）
    is_hanging = hammer_like_cond & is_green1
    patterns_calc.loc[is_hanging, f'KAP_HANGINGMAN_{tf}'] = -1

    # 4. 星线形态 (Morning Star / Evening Star) - 三K线组合
    # 定义：第一根长K线，第二根小实体星线（与第一根有跳空），第三根反向长K线（与第二根有跳空，并深入第一根实体）。
    star_body_max_ratio_of_range = 0.3 # 星线（第二根K线）的实体最大占其自身波幅的30%
    is_star1 = (body1 < full_range1 * star_body_max_ratio_of_range) & (body1 > 1e-9) # 前一日是星线
    
    # 早晨之星 (看涨反转)
    # 条件: 前2日大阴线, 前1日星线(与前2日阴线向下跳空), 当日大阳线(与前1日星线向上跳空，收盘深入前2日阴线实体过半)
    gap1_down_morning = (h1 < l2) | ((np.maximum(o1,c1) < np.minimum(o2,c2)) & (np.minimum(o1,c1) < np.minimum(o2,c2))) # 星线与前大阴向下跳空
    gap2_up_morning = (l > h1) | ((np.minimum(o,c) > np.maximum(o1,c1)) & (np.maximum(o,c) > np.maximum(o1,c1)))       # 当日阳线与星线向上跳空
    morning_star = is_red2 & (body2 > avg_body) & \
                   is_star1 & \
                   is_green & (body > body2 * 0.5) & (c > (o2 + c2) / 2) & \
                   gap1_down_morning & gap2_up_morning
    patterns_calc.loc[morning_star, f'KAP_MORNINGSTAR_{tf}'] = 1

    # 黄昏之星 (看跌反转)
    # 条件: 前2日大阳线, 前1日星线(与前2日阳线向上跳空), 当日大阴线(与前1日星线向下跳空，收盘深入前2日阳线实体过半)
    gap1_up_evening = (l1 > h2) | ((np.minimum(o1,c1) > np.maximum(o2,c2)) & (np.maximum(o1,c1) > np.maximum(o2,c2))) # 星线与前大阳向上跳空
    gap2_down_evening = (h < l1) | ((np.maximum(o,c) < np.minimum(o1,c1)) & (np.minimum(o,c) < np.minimum(o1,c1)))     # 当日阴线与星线向下跳空
    evening_star = is_green2 & (body2 > avg_body) & \
                   is_star1 & \
                   is_red & (body > body2 * 0.5) & (c < (o2 + c2) / 2) & \
                   gap1_up_evening & gap2_down_evening
    patterns_calc.loc[evening_star, f'KAP_EVENINGSTAR_{tf}'] = -1

    # 5. 刺透形态 (Piercing Line) / 乌云盖顶 (Dark Cloud Cover) - 两K线组合
    # 刺透线 (看涨反转): 前大阴，当日阳线低开（低于前阴最低价），收盘深入前阴实体一半以上但未完全吞没。
    piercing = is_red1 & (body1 > avg_body) & is_green & (o < l1) & (c > (o1 + c1) / 2) & (c < o1)
    patterns_calc.loc[piercing, f'KAP_PIERCINGLINE_{tf}'] = 1
    # 乌云盖顶 (看跌反转): 前大阳，当日阴线高开（高于前阳最高价），收盘深入前阳实体一半以上但未完全吞没。
    dark_cloud = is_green1 & (body1 > avg_body) & is_red & (o > h1) & (c < (o1 + c1) / 2) & (c > o1)
    patterns_calc.loc[dark_cloud, f'KAP_DARKCLOUDCOVER_{tf}'] = -1

    # 6. 三白兵 (Three White Soldiers) / 三只乌鸦 (Three Black Crows) - 三K线组合
    min_body_ratio_for_soldier_crow = 0.5 # 三兵/三鸦中每根K线实体至少是平均实体的0.5倍
    # 三白兵 (看涨持续): 连续三根阳线，每日开盘在前一日实体内，收盘高于前一日收盘，实体逐渐放大或相近。
    soldiers = is_green2 & (body2 > avg_body * min_body_ratio_for_soldier_crow) & \
               is_green1 & (body1 > avg_body * min_body_ratio_for_soldier_crow) & (c1 > c2) & (o1 < c2) & (o1 > o2) & \
               is_green & (body > avg_body * min_body_ratio_for_soldier_crow) & (c > c1) & (o < c1) & (o > o1)
    patterns_calc.loc[soldiers, f'KAP_THREEWHITESOLDIERS_{tf}'] = 1
    # 三只乌鸦 (看跌持续): 连续三根阴线，每日开盘在前一日实体内，收盘低于前一日收盘，实体逐渐放大或相近。
    crows = is_red2 & (body2 > avg_body * min_body_ratio_for_soldier_crow) & \
            is_red1 & (body1 > avg_body * min_body_ratio_for_soldier_crow) & (c1 < c2) & (o1 > c2) & (o1 < o2) & \
            is_red & (body > avg_body * min_body_ratio_for_soldier_crow) & (c < c1) & (o > c1) & (o < o1)
    patterns_calc.loc[crows, f'KAP_THREEBLACKCROWS_{tf}'] = -1

    # 7. 孕线形态 (Harami) - 两K线组合
    # 定义：第二根K线的实体完全被第一根K线的实体所包含。
    is_harami_body = (np.maximum(o, c) < np.maximum(o1, c1)) & (np.minimum(o, c) > np.minimum(o1, c1))
    # 看涨孕线: 前大阴，后小阳线或十字星被包含。
    bullish_harami = is_red1 & (body1 > avg_body) & is_harami_body & (is_green | is_doji) # is_doji 是当日的十字星判断
    patterns_calc.loc[bullish_harami, f'KAP_BULLISHHARAMI_{tf}'] = 1
    # 看跌孕线: 前大阳，后小阴线或十字星被包含。
    bearish_harami = is_green1 & (body1 > avg_body) & is_harami_body & (is_red | is_doji)
    patterns_calc.loc[bearish_harami, f'KAP_BEARISHHARAMI_{tf}'] = -1
    
    # 十字孕线 (Harami Cross): 孕线形态的第二根K线是十字星。
    bullish_harami_cross = is_red1 & (body1 > avg_body) & is_harami_body & is_doji
    patterns_calc.loc[bullish_harami_cross, f'KAP_BULLISHHARAMICROSS_{tf}'] = 1
    bearish_harami_cross = is_green1 & (body1 > avg_body) & is_harami_body & is_doji
    patterns_calc.loc[bearish_harami_cross, f'KAP_BEARISHHARAMICROSS_{tf}'] = -1

    # 8. 镊子顶 (Tweezer Top) / 镊子底 (Tweezer Bottom) - 两K线组合
    # 定义：连续两根K线的最高价（镊子顶）或最低价（镊子底）几乎相同。
    tweezer_tolerance_ratio_of_range = 0.02 # 价格相同的容忍度：平均波幅的2%
    # 镊子底: 两根K线最低点相近，通常前一根为阴线。
    tweezer_bottom = (abs(l - l1) < avg_range * tweezer_tolerance_ratio_of_range) # is_red1 可作为可选增强条件
    patterns_calc.loc[tweezer_bottom, f'KAP_TWEEZERBOTTOM_{tf}'] = 1
    # 镊子顶: 两根K线最高点相近，通常前一根为阳线。
    tweezer_top = (abs(h - h1) < avg_range * tweezer_tolerance_ratio_of_range) # is_green1 可作为可选增强条件
    patterns_calc.loc[tweezer_top, f'KAP_TWEEZERTOP_{tf}'] = -1

    # 9. 光头光脚K线 (Marubozu) - 单K线形态
    # 定义：实体非常饱满，几乎没有上下影线，实体占整个K线范围的比例非常高，且实体本身也较大。
    marubozu_body_min_ratio_of_range = 0.95    # 实体至少占全天波幅的95%
    marubozu_body_min_ratio_of_avg_body = 1.5 # 实体至少是平均实体的1.5倍
    is_marubozu_cond = (body >= full_range * marubozu_body_min_ratio_of_range) & \
                       (body > avg_body * marubozu_body_min_ratio_of_avg_body)
    # 看涨光头光脚 (大阳线)
    bull_marubozu = is_marubozu_cond & is_green
    patterns_calc.loc[bull_marubozu, f'KAP_BULLISHMARUBOZU_{tf}'] = 1
    # 看跌光头光脚 (大阴线)
    bear_marubozu = is_marubozu_cond & is_red
    patterns_calc.loc[bear_marubozu, f'KAP_BEARISHMARUBOZU_{tf}'] = -1
    
    # --- 多K线组合形态 (需要更多历史数据) ---
    # 10. 上升三法 (Rising Three Methods) / 下降三法 (Falling Three Methods) - 五K线组合
    if len(df_calc_idx) >= 5: # 确保有足够数据进行5根K线的判断
        consolidation_body_max_ratio_of_avg_body = 0.5 # 中间整理K线的实体最大为平均实体的0.5倍

        # 上升三法 (看涨持续)
        # 条件: 前4日长阳, 中间三日小K线(阴或阳实体小)整理且在前4日长阳实体内, 当日长阳收盘高于前4日收盘。
        rising_three = is_green4 & (body4 > avg_body * 1.5) & \
                       (is_red3 | (body3 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h3 < h4) & (l3 > l4) & \
                       (is_red2 | (body2 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h2 < h4) & (l2 > l4) & \
                       (is_red1 | (body1 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h1 < h4) & (l1 > l4) & \
                       is_green & (body > avg_body * 1.5) & (c > c4) & (o > l1) # 当日阳线开盘高于前一整理K线低点
        patterns_calc.loc[rising_three, f'KAP_RISINGTHREEMETHODS_{tf}'] = 1
        
        # 下降三法 (看跌持续)
        # 条件: 前4日长阴, 中间三日小K线整理且在前4日长阴实体内, 当日长阴收盘低于前4日收盘。
        falling_three = is_red4 & (body4 > avg_body * 1.5) & \
                        (is_green3 | (body3 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h3 < h4) & (l3 > l4) & \
                        (is_green2 | (body2 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h2 < h4) & (l2 > l4) & \
                        (is_green1 | (body1 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h1 < h4) & (l1 > l4) & \
                        is_red & (body > avg_body * 1.5) & (c < c4) & (o < h1) # 当日阴线开盘低于前一整理K线高点
        patterns_calc.loc[falling_three, f'KAP_FALLINGTHREEMETHODS_{tf}'] = -1

    # 11. 其他三K线组合形态
    if len(df_calc_idx) >= 3: # 确保有足够数据
        # 向上跳空两只乌鸦 (Upside Gap Two Crows) - 看跌反转
        # 条件: 前2日强阳, 前1日向上跳空收小阴, 当日再收阴线开盘高于前阴实体但收盘低于前阴实体，形成吞噬。
        upside_gap_two_crows = is_green2 & (body2 > avg_body) & \
                              is_red1 & (body1 < avg_body * 0.7) & (o1 > h2) & \
                              is_red & (o > o1) & (c < c1) & (c < h2) # 当日阴线实体吞没前小阴，收盘低于第一天高点
        patterns_calc.loc[upside_gap_two_crows, f'KAP_UPSIDEGAPTWOCROWS_{tf}'] = -1

        # 跳空并列线 (Tasuki Gap) - 持续形态
        # 向上跳空并列阳线 (Upside Tasuki Gap) - 看涨持续
        # 条件: 前2日阳线, 前1日向上跳空收阳, 当日阴线开盘于前1日阳线实体内，收盘于缺口之内但未完全封闭缺口。
        upside_tasuki_gap = is_green2 & \
                           is_green1 & (o1 > h2) & \
                           is_red & (o > o1) & (o < c1) & (c < o1) & (c > h2)
        patterns_calc.loc[upside_tasuki_gap, f'KAP_UPSIDETASUKIGAP_{tf}'] = 1
        
        # 向下跳空并列阴线 (Downside Tasuki Gap) - 看跌持续
        # 条件: 前2日阴线, 前1日向下跳空收阴, 当日阳线开盘于前1日阴线实体内，收盘于缺口之内但未完全封闭缺口。
        downside_tasuki_gap = is_red2 & \
                             is_red1 & (o1 < l2) & \
                             is_green & (o < o1) & (o > c1) & (c > o1) & (c < l2)
        patterns_calc.loc[downside_tasuki_gap, f'KAP_DOWNSIDETASUKIGAP_{tf}'] = -1

    # 12. 其他两K线组合形态
    if len(df_calc_idx) >= 2: # 确保有足够数据
        # 分离线 (Separating Lines) - 持续形态
        # 条件：两根K线颜色相反，开盘价相同。
        same_open_cond = abs(o - o1) < avg_range * 0.02 # 开盘价几乎相同
        # 看涨分离线: 下降趋势中（前阴），前阴后阳，开盘相同。
        bullish_sep_lines = is_red1 & is_green & same_open_cond
        patterns_calc.loc[bullish_sep_lines, f'KAP_BULLISHSEPARATINGLINES_{tf}'] = 1
        # 看跌分离线: 上升趋势中（前阳），前阳后阴，开盘相同。
        bearish_sep_lines = is_green1 & is_red & same_open_cond
        patterns_calc.loc[bearish_sep_lines, f'KAP_BEARISHSEPARATINGLINES_{tf}'] = -1

        # 反击线 (Counterattack Lines) - 反转形态
        # 条件：两根K线颜色相反，收盘价相同，且第二根K线大幅跳空开盘。
        same_close_cond = abs(c - c1) < avg_range * 0.02 # 收盘价几乎相同
        # 看涨反击线: 下降趋势中（前长阴），当日长阳大幅跳空低开，但收盘价与前一日相同。
        bullish_counter = is_red1 & (body1 > avg_body) & is_green & (body > avg_body) & same_close_cond & (o < l1)
        patterns_calc.loc[bullish_counter, f'KAP_BULLISHCOUNTERATTACK_{tf}'] = 1
        # 看跌反击线: 上升趋势中（前长阳），当日长阴大幅跳空高开，但收盘价与前一日相同。
        bearish_counter = is_green1 & (body1 > avg_body) & is_red & (body > avg_body) & same_close_cond & (o > h1)
        patterns_calc.loc[bearish_counter, f'KAP_BEARISHCOUNTERATTACK_{tf}'] = -1

    # --- 结果合并与返回 ---
    # 将在有效数据上计算得到的形态结果 (patterns_calc) 更新到完整索引的默认结果DataFrame (default_pattern_df) 中
    # .update() 方法会用 patterns_calc 中的非NaN值（这里是0, 1, -1）更新 default_pattern_df 对应索引位置的值
    default_pattern_df.update(patterns_calc)
    
    logger.info(f"TF {tf} K 线形态检测完成。")
    # 确保最终返回的DataFrame填充所有可能的NaN为0，并为整数类型
    return default_pattern_df.fillna(0).astype(int)

# --- 指标评分函数 (原 _get_xxx_score 改为公用) ---
def _safe_fillna_series(series_list: List[pd.Series], fill_values: List[Any]) -> List[pd.Series]:
    """辅助函数，安全地填充多个Series的NaN值，并确保索引一致（如果需要）。"""
    if not series_list:
        return []
    
    # 以第一个Series的索引为基准，如果后续Series索引不同，则reindex
    base_index = series_list[0].index
    processed_series = []
    for i, s in enumerate(series_list):
        series_to_fill = s
        if not series_to_fill.index.equals(base_index):
            logger.debug(f"Series {i} 索引与基准不一致，将进行reindex。")
            series_to_fill = series_to_fill.reindex(base_index)
        
        # 尝试ffill().bfill()，然后用指定值填充剩余NaN
        filled_s = series_to_fill.ffill().bfill()
        if fill_values[i] is not None: # None表示不使用特定值填充，仅依赖ffill/bfill
             filled_s = filled_s.fillna(fill_values[i])
        processed_series.append(filled_s)
    return processed_series

def calculate_macd_score(macd_series: pd.Series, macd_d: pd.Series, macd_h: pd.Series) -> pd.Series:
    """MACD 评分 (0-100)。"""
    # 确保索引一致并填充NaN
    # MACD线和DEA线中性值可以是0或前值，MACDh中性值是0
    macd_s, macd_d_s, macd_h_s = _safe_fillna_series(
        [macd_series, macd_d, macd_h],
        [0.0, 0.0, 0.0] # 假设0是合理的填充值，或者用mean()等
    )
    # 如果填充后仍有NaN（例如整个序列都是NaN），则返回全50分
    if macd_s.isnull().all() or macd_d_s.isnull().all() or macd_h_s.isnull().all():
        return pd.Series(50.0, index=macd_series.index).clip(0,100)

    score = pd.Series(50.0, index=macd_s.index)

    buy_cross = (macd_s.shift(1) < macd_d_s.shift(1)) & (macd_s >= macd_d_s)
    sell_cross = (macd_s.shift(1) > macd_d_s.shift(1)) & (macd_s <= macd_d_s)

    buy_cross_above_zero = buy_cross & (macd_s > 0)
    buy_cross_below_zero = buy_cross & (macd_s <= 0)
    sell_cross_above_zero = sell_cross & (macd_s >= 0)
    sell_cross_below_zero = sell_cross & (macd_s < 0)

    score.loc[buy_cross_above_zero] = 80.0
    score.loc[buy_cross_below_zero] = np.maximum(score.loc[buy_cross_below_zero], 70.0) # Use np.maximum to avoid overwriting 80 with 70
    score.loc[sell_cross_below_zero] = 20.0
    score.loc[sell_cross_above_zero] = np.minimum(score.loc[sell_cross_above_zero], 30.0) # Use np.minimum

    bullish_momentum = (macd_h_s > macd_h_s.shift(1)) & (macd_h_s > 0)
    bearish_momentum = (macd_h_s < macd_h_s.shift(1)) & (macd_h_s < 0)
    
    # Conditions for non-cross scenarios
    not_cross_cond = ~buy_cross & ~sell_cross
    
    score.loc[bullish_momentum & not_cross_cond] = np.maximum(score.loc[bullish_momentum & not_cross_cond], 60.0)
    score.loc[bearish_momentum & not_cross_cond] = np.minimum(score.loc[bearish_momentum & not_cross_cond], 40.0)

    above_zero_no_mom_increase = (macd_h_s > 0) & (~bullish_momentum) & not_cross_cond
    below_zero_no_mom_decrease = (macd_h_s < 0) & (~bearish_momentum) & not_cross_cond
    
    score.loc[above_zero_no_mom_increase] = np.maximum(score.loc[above_zero_no_mom_increase], 55.0)
    score.loc[below_zero_no_mom_decrease] = np.minimum(score.loc[below_zero_no_mom_decrease], 45.0)
    
    return score.clip(0, 100)

def calculate_rsi_score(rsi: pd.Series, params: Dict) -> pd.Series:
    """RSI 评分 (0-100)。"""
    rsi_s, = _safe_fillna_series([rsi], [50.0]) # RSI 中性50
    if rsi_s.isnull().all():
        return pd.Series(50.0, index=rsi.index).clip(0,100)

    score = pd.Series(50.0, index=rsi_s.index)
    os = params.get('rsi_oversold', 30)
    ob = params.get('rsi_overbought', 70)
    ext_os = params.get('rsi_extreme_oversold', 20)
    ext_ob = params.get('rsi_extreme_overbought', 80)

    score.loc[rsi_s < ext_os] = 95.0
    score.loc[rsi_s > ext_ob] = 5.0
    
    score.loc[(rsi_s >= ext_os) & (rsi_s < os)] = np.maximum(score.loc[(rsi_s >= ext_os) & (rsi_s < os)], 85.0)
    score.loc[(rsi_s <= ext_ob) & (rsi_s > ob)] = np.minimum(score.loc[(rsi_s <= ext_ob) & (rsi_s > ob)], 15.0)

    buy_signal = (rsi_s.shift(1) < os) & (rsi_s >= os)
    sell_signal = (rsi_s.shift(1) > ob) & (rsi_s <= ob)
    score.loc[buy_signal] = np.maximum(score.loc[buy_signal], 75.0) # Ensure not overwriting higher scores
    score.loc[sell_signal] = np.minimum(score.loc[sell_signal], 25.0) # Ensure not overwriting lower scores

    not_signal_cond = ~buy_signal & ~sell_signal
    neutral_zone_cond = (rsi_s >= os) & (rsi_s <= ob) & not_signal_cond

    bullish_trend = neutral_zone_cond & (rsi_s > rsi_s.shift(1))
    bearish_trend = neutral_zone_cond & (rsi_s < rsi_s.shift(1))
    
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)
    
    return score.clip(0, 100)

def calculate_kdj_score(k: pd.Series, d: pd.Series, j: pd.Series, params: Dict) -> pd.Series:
    """KDJ 评分 (0-100)。"""
    k_s, d_s, j_s = _safe_fillna_series([k, d, j], [50.0, 50.0, 50.0]) # KDJ 中性50
    if k_s.isnull().all(): # Check one, assume others similar after _safe_fillna_series
        return pd.Series(50.0, index=k.index).clip(0,100)

    score = pd.Series(50.0, index=k_s.index)
    os = params.get('kdj_oversold', 20)
    ob = params.get('kdj_overbought', 80)
    ext_os = params.get('kdj_extreme_oversold', 10)
    ext_ob = params.get('kdj_extreme_overbought', 90)

    score.loc[j_s < ext_os] = 95.0
    score.loc[j_s > ext_ob] = 5.0

    # Apply to k or d, ensuring not to overwrite extreme j scores
    score.loc[(k_s < os) | (d_s < os)] = np.maximum(score.loc[(k_s < os) | (d_s < os)], 85.0)
    score.loc[(k_s > ob) | (d_s > ob)] = np.minimum(score.loc[(k_s > ob) | (d_s > ob)], 15.0)

    buy_cross = (k_s.shift(1) < d_s.shift(1)) & (k_s >= d_s)
    sell_cross = (k_s.shift(1) > d_s.shift(1)) & (k_s <= d_s)

    buy_cross_os = buy_cross & (j_s < os)
    buy_cross_ob = buy_cross & (j_s > ob) # Potentially risky cross
    sell_cross_os = sell_cross & (j_s < os) # Potentially risky cross
    sell_cross_ob = sell_cross & (j_s > ob)

    score.loc[buy_cross_os] = np.maximum(score.loc[buy_cross_os], 80.0)
    score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)] = np.maximum(score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)], 75.0)
    score.loc[buy_cross_ob] = np.maximum(score.loc[buy_cross_ob], 60.0) # Still a cross, but in OB

    score.loc[sell_cross_ob] = np.minimum(score.loc[sell_cross_ob], 20.0)
    score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)] = np.minimum(score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)], 25.0)
    score.loc[sell_cross_os] = np.minimum(score.loc[sell_cross_os], 40.0) # Cross in OS

    not_cross_cond = ~buy_cross & ~sell_cross
    bullish_j_in_os = (j_s < os) & (j_s > j_s.shift(1)) & not_cross_cond
    bearish_j_in_ob = (j_s > ob) & (j_s < j_s.shift(1)) & not_cross_cond
    score.loc[bullish_j_in_os] = np.maximum(score.loc[bullish_j_in_os], 70.0)
    score.loc[bearish_j_in_ob] = np.minimum(score.loc[bearish_j_in_ob], 30.0)
    
    neutral_j_zone = (j_s >= os) & (j_s <= ob) & not_cross_cond
    bullish_j_trend_neutral = neutral_j_zone & (j_s > j_s.shift(1))
    bearish_j_trend_neutral = neutral_j_zone & (j_s < j_s.shift(1))
    score.loc[bullish_j_trend_neutral] = np.maximum(score.loc[bullish_j_trend_neutral], 55.0)
    score.loc[bearish_j_trend_neutral] = np.minimum(score.loc[bearish_j_trend_neutral], 45.0)
    
    return score.clip(0, 100)

def calculate_boll_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """BOLL 评分 (0-100)。"""
    # For BOLL, if bands are NaN, filling with close or mid might be tricky.
    # Original fill logic was complex. Let's simplify slightly but keep robustness.
    # If close is all NaN, it's problematic.
    if close.isnull().all():
        return pd.Series(50.0, index=close.index).clip(0,100)

    close_f = close.ffill().bfill() # Primary series
    # Fill bands relative to close_f or its mean if they are completely NaN
    mid_f = mid.ffill().bfill().fillna(close_f.rolling(20, min_periods=1).mean())
    # Estimate std_dev if bands are missing, for fallback
    std_dev_est = close_f.rolling(20, min_periods=1).std().fillna(close_f.std()).fillna(0.01 * close_f.mean()) # small val if std is 0

    upper_f = upper.ffill().bfill().fillna(mid_f + 2 * std_dev_est)
    lower_f = lower.ffill().bfill().fillna(mid_f - 2 * std_dev_est)
    
    # Re-check for all NaNs after filling attempts
    series_to_check = [close_f, upper_f, mid_f, lower_f]
    if any(s.isnull().all() for s in series_to_check):
         logger.warning("BOLL 评分：一个或多个关键序列在填充后仍全为NaN。")
         return pd.Series(50.0, index=close.index).clip(0,100)


    score = pd.Series(50.0, index=close_f.index)

    score.loc[close_f <= lower_f] = 90.0
    buy_support = (close_f.shift(1) < lower_f.shift(1)) & (close_f >= lower_f)
    score.loc[buy_support] = np.maximum(score.loc[buy_support], 80.0)

    score.loc[close_f >= upper_f] = 10.0
    sell_pressure = (close_f.shift(1) > upper_f.shift(1)) & (close_f <= upper_f)
    score.loc[sell_pressure] = np.minimum(score.loc[sell_pressure], 20.0)

    buy_mid_cross = (close_f.shift(1) < mid_f.shift(1)) & (close_f >= mid_f)
    sell_mid_cross = (close_f.shift(1) > mid_f.shift(1)) & (close_f <= mid_f)
    score.loc[buy_mid_cross] = np.maximum(score.loc[buy_mid_cross], 65.0)
    score.loc[sell_mid_cross] = np.minimum(score.loc[sell_mid_cross], 35.0)

    not_extreme_cond = (close_f > lower_f) & (close_f < upper_f)
    not_mid_cross_cond = ~buy_mid_cross & ~sell_mid_cross
    
    is_above_mid = not_extreme_cond & not_mid_cross_cond & (close_f > mid_f)
    is_below_mid = not_extreme_cond & not_mid_cross_cond & (close_f < mid_f)
    score.loc[is_above_mid] = np.maximum(score.loc[is_above_mid], 55.0)
    score.loc[is_below_mid] = np.minimum(score.loc[is_below_mid], 45.0)
    
    return score.clip(0, 100)

# --- Template for other scoring functions (showing CCI as example) ---
def calculate_cci_score(cci: pd.Series, params: Dict) -> pd.Series:
    """CCI 评分 (0-100)。"""
    cci_s, = _safe_fillna_series([cci], [0.0]) # CCI 中性0
    if cci_s.isnull().all():
        return pd.Series(50.0, index=cci.index).clip(0,100)
        
    score = pd.Series(50.0, index=cci_s.index)
    threshold = params.get('cci_threshold', 100)
    ext_threshold = params.get('cci_extreme_threshold', 200)

    score.loc[cci_s < -ext_threshold] = 95.0
    score.loc[cci_s > ext_threshold] = 5.0

    score.loc[(cci_s >= -ext_threshold) & (cci_s < -threshold)] = np.maximum(score.loc[(cci_s >= -ext_threshold) & (cci_s < -threshold)], 85.0)
    score.loc[(cci_s <= ext_threshold) & (cci_s > threshold)] = np.minimum(score.loc[(cci_s <= ext_threshold) & (cci_s > threshold)], 15.0)

    buy_signal = (cci_s.shift(1) < -threshold) & (cci_s >= -threshold)
    sell_signal = (cci_s.shift(1) > threshold) & (cci_s <= threshold)
    score.loc[buy_signal] = np.maximum(score.loc[buy_signal], 75.0)
    score.loc[sell_signal] = np.minimum(score.loc[sell_signal], 25.0)

    not_signal_cond = ~buy_signal & ~sell_signal
    neutral_zone_cond = (cci_s >= -threshold) & (cci_s <= threshold) & not_signal_cond
    
    bullish_trend = neutral_zone_cond & (cci_s > cci_s.shift(1))
    bearish_trend = neutral_zone_cond & (cci_s < cci_s.shift(1))
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)
    
    return score.clip(0, 100)

def calculate_mfi_score(mfi: pd.Series, params: Dict) -> pd.Series:
    """MFI 评分 (0-100)。"""
    mfi_s, = _safe_fillna_series([mfi], [50.0]) # MFI 中性50
    if mfi_s.isnull().all():
        return pd.Series(50.0, index=mfi.index).clip(0,100)

    score = pd.Series(50.0, index=mfi_s.index)
    os = params.get('mfi_oversold', 20)
    ob = params.get('mfi_overbought', 80)
    ext_os = params.get('mfi_extreme_oversold', 10)
    ext_ob = params.get('mfi_extreme_overbought', 90)

    score.loc[mfi_s < ext_os] = 95.0
    score.loc[mfi_s > ext_ob] = 5.0
    
    score.loc[(mfi_s >= ext_os) & (mfi_s < os)] = np.maximum(score.loc[(mfi_s >= ext_os) & (mfi_s < os)], 85.0)
    score.loc[(mfi_s <= ext_ob) & (mfi_s > ob)] = np.minimum(score.loc[(mfi_s <= ext_ob) & (mfi_s > ob)], 15.0)

    buy_signal = (mfi_s.shift(1) < os) & (mfi_s >= os)
    sell_signal = (mfi_s.shift(1) > ob) & (mfi_s <= ob)
    score.loc[buy_signal] = np.maximum(score.loc[buy_signal], 75.0)
    score.loc[sell_signal] = np.minimum(score.loc[sell_signal], 25.0)

    not_signal_cond = ~buy_signal & ~sell_signal
    neutral_zone_cond = (mfi_s >= os) & (mfi_s <= ob) & not_signal_cond

    bullish_trend = neutral_zone_cond & (mfi_s > mfi_s.shift(1))
    bearish_trend = neutral_zone_cond & (mfi_s < mfi_s.shift(1))
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)
    
    return score.clip(0, 100)

def calculate_roc_score(roc: pd.Series) -> pd.Series:
    """ROC 评分 (0-100)。"""
    roc_s, = _safe_fillna_series([roc], [0.0]) # ROC 中性0
    if roc_s.isnull().all():
        return pd.Series(50.0, index=roc.index).clip(0,100)

    score = pd.Series(50.0, index=roc_s.index)
    buy_cross = (roc_s.shift(1) < 0) & (roc_s >= 0)
    sell_cross = (roc_s.shift(1) > 0) & (roc_s <= 0)
    score.loc[buy_cross] = 70.0
    score.loc[sell_cross] = 30.0

    not_cross_cond = ~buy_cross & ~sell_cross
    bullish_trend = (roc_s > 0) & (roc_s > roc_s.shift(1)) & not_cross_cond
    bearish_trend = (roc_s < 0) & (roc_s < roc_s.shift(1)) & not_cross_cond
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 60.0)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 40.0)

    bullish_waning = (roc_s > 0) & (roc_s < roc_s.shift(1)) & ~sell_cross # Allow sell_cross to take precedence
    bearish_waning = (roc_s < 0) & (roc_s > roc_s.shift(1)) & ~buy_cross  # Allow buy_cross to take precedence
    score.loc[bullish_waning] = np.minimum(score.loc[bullish_waning], 55.0)
    score.loc[bearish_waning] = np.maximum(score.loc[bearish_waning], 45.0)
    
    return score.clip(0, 100)

def calculate_dmi_score(pdi: pd.Series, ndi: pd.Series, adx: pd.Series, params: Dict) -> pd.Series:
    """DMI 评分 (0-100)。"""
    pdi_s, ndi_s, adx_s = _safe_fillna_series([pdi, ndi, adx], [0.0, 0.0, 0.0]) # DMI/ADX 中性0
    if pdi_s.isnull().all():
        return pd.Series(50.0, index=pdi.index).clip(0,100)

    score = pd.Series(50.0, index=pdi_s.index)
    adx_th = params.get('adx_threshold', 25)
    adx_strong_th = params.get('adx_strong_threshold', 40)

    buy_cross = (pdi_s.shift(1) < ndi_s.shift(1)) & (pdi_s >= ndi_s)
    sell_cross = (ndi_s.shift(1) < pdi_s.shift(1)) & (ndi_s >= pdi_s)
    
    # Base scores for crosses
    score.loc[buy_cross] = 70.0
    score.loc[sell_cross] = 30.0

    adx_rising = adx_s > adx_s.shift(1)
    
    # ADX confirmed crosses (higher priority)
    score.loc[buy_cross & (adx_s > adx_th)] = np.maximum(score.loc[buy_cross & (adx_s > adx_th)], 75.0)
    score.loc[buy_cross & (adx_s > adx_strong_th) & adx_rising] = np.maximum(score.loc[buy_cross & (adx_s > adx_strong_th) & adx_rising], 85.0)
    
    score.loc[sell_cross & (adx_s > adx_th)] = np.minimum(score.loc[sell_cross & (adx_s > adx_th)], 25.0)
    score.loc[sell_cross & (adx_s > adx_strong_th) & adx_rising] = np.minimum(score.loc[sell_cross & (adx_s > adx_strong_th) & adx_rising], 15.0)

    not_cross_cond = ~buy_cross & ~sell_cross
    is_bullish_trend = (pdi_s > ndi_s) & not_cross_cond
    is_bearish_trend = (ndi_s > pdi_s) & not_cross_cond

    score.loc[is_bullish_trend & (adx_s > adx_strong_th)] = np.maximum(score.loc[is_bullish_trend & (adx_s > adx_strong_th)], 65.0)
    score.loc[is_bullish_trend & (adx_s > adx_th) & (adx_s <= adx_strong_th)] = np.maximum(score.loc[is_bullish_trend & (adx_s > adx_th) & (adx_s <= adx_strong_th)], 60.0)
    score.loc[is_bullish_trend & (adx_s <= adx_th)] = np.maximum(score.loc[is_bullish_trend & (adx_s <= adx_th)], 55.0)

    score.loc[is_bearish_trend & (adx_s > adx_strong_th)] = np.minimum(score.loc[is_bearish_trend & (adx_s > adx_strong_th)], 35.0)
    score.loc[is_bearish_trend & (adx_s > adx_th) & (adx_s <= adx_strong_th)] = np.minimum(score.loc[is_bearish_trend & (adx_s > adx_th) & (adx_s <= adx_strong_th)], 40.0)
    score.loc[is_bearish_trend & (adx_s <= adx_th)] = np.minimum(score.loc[is_bearish_trend & (adx_s <= adx_th)], 45.0)
    
    return score.clip(0, 100)

def calculate_sar_score(close: pd.Series, sar: pd.Series) -> pd.Series:
    """SAR 评分 (0-100)。"""
    # SAR can be tricky to fill if NaN. Filling with close means neutral.
    close_s, sar_s = _safe_fillna_series([close, sar], [None, None]) # Let ffill/bfill handle first
    # If sar_s is still NaN after ffill/bfill, fill with close_s
    sar_s = sar_s.fillna(close_s)
    # If close_s is all NaN, then sar_s might also be.
    if close_s.isnull().all() or sar_s.isnull().all():
        return pd.Series(50.0, index=close.index).clip(0,100)

    score = pd.Series(50.0, index=close_s.index)

    buy_signal = (sar_s.shift(1) > close_s.shift(1)) & (sar_s <= close_s)
    sell_signal = (sar_s.shift(1) < close_s.shift(1)) & (sar_s >= close_s)
    score.loc[buy_signal] = 75.0
    score.loc[sell_signal] = 25.0

    not_signal_cond = ~buy_signal & ~sell_signal
    score.loc[(close_s > sar_s) & not_signal_cond] = np.maximum(score.loc[(close_s > sar_s) & not_signal_cond], 60.0)
    score.loc[(close_s < sar_s) & not_signal_cond] = np.minimum(score.loc[(close_s < sar_s) & not_signal_cond], 40.0)
    
    return score.clip(0, 100)

def calculate_stoch_score(k: pd.Series, d: pd.Series, params: Dict) -> pd.Series:
    """随机指标 (STOCH) 评分 (0-100)。"""
    k_s, d_s = _safe_fillna_series([k, d], [50.0, 50.0]) # STOCH 中性50
    if k_s.isnull().all():
        return pd.Series(50.0, index=k.index).clip(0,100)

    score = pd.Series(50.0, index=k_s.index)
    os = params.get('stoch_oversold', 20)
    ob = params.get('stoch_overbought', 80)
    ext_os = params.get('stoch_extreme_oversold', 10)
    ext_ob = params.get('stoch_extreme_overbought', 90)

    score.loc[(k_s < ext_os) | (d_s < ext_os)] = 95.0
    score.loc[(k_s > ext_ob) | (d_s > ext_ob)] = 5.0

    score.loc[((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))] = \
        np.maximum(score.loc[((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))], 85.0)
    score.loc[((k_s <= ext_ob) & (k_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))] = \
        np.minimum(score.loc[((k_s <= ext_ob) & (k_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))], 15.0)

    buy_cross = (k_s.shift(1) < d_s.shift(1)) & (k_s >= d_s)
    sell_cross = (k_s.shift(1) > d_s.shift(1)) & (k_s <= d_s)

    buy_cross_os = buy_cross & (d_s < os)
    buy_cross_ob = buy_cross & (d_s > ob)
    sell_cross_os = sell_cross & (d_s < os)
    sell_cross_ob = sell_cross & (d_s > ob)

    score.loc[buy_cross_os] = np.maximum(score.loc[buy_cross_os], 80.0)
    score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)] = np.maximum(score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)], 75.0)
    score.loc[buy_cross_ob] = np.maximum(score.loc[buy_cross_ob], 60.0)

    score.loc[sell_cross_ob] = np.minimum(score.loc[sell_cross_ob], 20.0)
    score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)] = np.minimum(score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)], 25.0)
    score.loc[sell_cross_os] = np.minimum(score.loc[sell_cross_os], 40.0)
    
    not_cross_cond = ~buy_cross & ~sell_cross
    neutral_stoch_zone = (k_s >= os) & (k_s <= ob) & (d_s >= os) & (d_s <= ob) & not_cross_cond
    
    bullish_trend_neutral = neutral_stoch_zone & (k_s > k_s.shift(1)) & (d_s > d_s.shift(1))
    bearish_trend_neutral = neutral_stoch_zone & (k_s < k_s.shift(1)) & (d_s < d_s.shift(1))
    score.loc[bullish_trend_neutral] = np.maximum(score.loc[bullish_trend_neutral], 55.0)
    score.loc[bearish_trend_neutral] = np.minimum(score.loc[bearish_trend_neutral], 45.0)
        
    return score.clip(0, 100)

def calculate_ma_score(close: pd.Series, ma: pd.Series, params: Optional[Dict] = None) -> pd.Series: # params made optional
    """移动平均线 (MA) 评分 (0-100)。"""
    # MA can be filled with close's rolling mean if NaN.
    close_s, ma_s = _safe_fillna_series([close, ma], [None, None]) # ffill/bfill first
    if close_s.isnull().all(): # If close is all NaN, MA likely too or irrelevant
        return pd.Series(50.0, index=close.index).clip(0,100)
    
    # If ma_s is still NaN, fill with a rolling mean of close_s
    ma_s = ma_s.fillna(close_s.rolling(20, min_periods=1).mean())
    if ma_s.isnull().all(): # If still all NaN (e.g. close_s was also all NaN and short)
        return pd.Series(50.0, index=close.index).clip(0,100)


    score = pd.Series(50.0, index=close_s.index)
    buy_cross = (close_s.shift(1) < ma_s.shift(1)) & (close_s >= ma_s)
    sell_cross = (close_s.shift(1) > ma_s.shift(1)) & (close_s <= ma_s)
    score.loc[buy_cross] = 70.0
    score.loc[sell_cross] = 30.0

    not_cross_cond = ~buy_cross & ~sell_cross
    score.loc[(close_s > ma_s) & not_cross_cond] = np.maximum(score.loc[(close_s > ma_s) & not_cross_cond], 60.0)
    score.loc[(close_s < ma_s) & not_cross_cond] = np.minimum(score.loc[(close_s < ma_s) & not_cross_cond], 40.0)
        
    return score.clip(0, 100)

def calculate_atr_score(atr: pd.Series) -> pd.Series:
    """ATR 评分 (0-100)。"""
    atr_s, = _safe_fillna_series([atr], [None]) # ffill/bfill first
    if atr_s.isnull().all():
        return pd.Series(50.0, index=atr.index).clip(0,100)
    atr_s = atr_s.fillna(atr_s.mean()) # Fill remaining NaNs with mean
    if atr_s.isnull().all(): # If mean is also NaN (all original were NaN)
        return pd.Series(50.0, index=atr.index).clip(0,100)


    score = pd.Series(50.0, index=atr_s.index)
    # Ensure window size is not larger than series length
    rolling_window = min(len(atr_s), 20)
    min_periods_rolling = max(1, int(rolling_window * 0.5)) if rolling_window > 0 else 1

    atr_mean = atr_s.rolling(window=rolling_window, min_periods=min_periods_rolling).mean().fillna(atr_s.mean())
    atr_std = atr_s.rolling(window=rolling_window, min_periods=min_periods_rolling).std().fillna(atr_s.std()).fillna(0) # fill std NaN with 0

    high_volatility = atr_s > (atr_mean + 0.5 * atr_std)
    low_volatility = atr_s < (atr_mean - 0.5 * atr_std)
    
    # ATR score is less directional, more about volatility regime
    score.loc[high_volatility] = np.maximum(score.loc[high_volatility], 60.0) # Higher score for high vol
    score.loc[low_volatility] = np.minimum(score.loc[low_volatility], 40.0)   # Lower score for low vol
            
    return score.clip(0, 100)

def calculate_adl_score(adl: pd.Series) -> pd.Series:
    """ADL (Accumulation/Distribution Line) 评分 (0-100)。"""
    adl_s, = _safe_fillna_series([adl], [0.0]) # ADL can be filled with 0 or previous value
    if adl_s.isnull().all():
        return pd.Series(50.0, index=adl.index).clip(0,100)

    score = pd.Series(50.0, index=adl_s.index)
    bullish_trend = adl_s > adl_s.shift(1)
    bearish_trend = adl_s < adl_s.shift(1)
    
    score.loc[bullish_trend] = 60.0
    score.loc[bearish_trend] = 40.0
    # Neutral for adl_s == adl_s.shift(1) is already 50.0
            
    return score.clip(0, 100)

def calculate_vwap_score(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """VWAP (Volume Weighted Average Price) 评分 (0-100)。"""
    close_s, vwap_s = _safe_fillna_series([close, vwap], [None, None]) # ffill/bfill first
    if close_s.isnull().all():
        return pd.Series(50.0, index=close.index).clip(0,100)
    # If vwap_s is still NaN, fill with close_s
    vwap_s = vwap_s.fillna(close_s)
    if vwap_s.isnull().all():
        return pd.Series(50.0, index=close.index).clip(0,100)

    score = pd.Series(50.0, index=close_s.index)
    score.loc[close_s > vwap_s] = 60.0
    score.loc[close_s < vwap_s] = 40.0
    # score.loc[close_s == vwap_s] is already 50.0
            
    return score.clip(0, 100)

def calculate_ichimoku_score(close: pd.Series, tenkan: pd.Series, kijun: pd.Series, senkou_a: pd.Series, senkou_b: pd.Series, chikou: pd.Series) -> pd.Series:
    """Ichimoku (一目均衡表) 评分 (0-100)。Simplified NaN handling."""
    # Ichimoku lines have inherent NaNs due to shifts. ffill/bfill is a simplification.
    # A more rigorous approach would respect these NaNs or use a sufficiently long data period.
    # Filling with close_s is a pragmatic choice if exact Ichimoku NaN propagation isn't critical.
    
    if close.isnull().all(): # If no close data, cannot score
        return pd.Series(50.0, index=close.index).clip(0,100)

    idx = close.index
    c = close.ffill().bfill() # Use this as the base for filling others if they are all NaN
    
    tk = tenkan.reindex(idx).ffill().bfill().fillna(c)
    kj = kijun.reindex(idx).ffill().bfill().fillna(c)
    sa = senkou_a.reindex(idx).ffill().bfill().fillna(c) # Senkou spans are future-shifted
    sb = senkou_b.reindex(idx).ffill().bfill().fillna(c)
    cs = chikou.reindex(idx).ffill().bfill().fillna(c)   # Chikou is past-shifted price

    # Check if any series is still all NaN after filling attempts
    if any(s.isnull().all() for s in [c, tk, kj, sa, sb, cs]):
        logger.warning("Ichimoku: One or more lines are all NaN after filling.")
        return pd.Series(50.0, index=close.index).clip(0,100)

    score = pd.Series(50.0, index=idx)

    # Weights (can be adjusted)
    w_price_kijun, w_tk_kj_cross, w_price_cloud, w_cloud_twist, w_chikou_price = 0.2, 0.2, 0.3, 0.1, 0.2

    # 1. Price vs Kijun
    pk_up = (c.shift(1) < kj.shift(1)) & (c >= kj); score.loc[pk_up] = np.maximum(score.loc[pk_up], 50 + 50*w_price_kijun + 10) # Cross bonus
    pk_dn = (c.shift(1) > kj.shift(1)) & (c <= kj); score.loc[pk_dn] = np.minimum(score.loc[pk_dn], 50 - 50*w_price_kijun - 10)
    score.loc[(c > kj) & ~pk_up] = np.maximum(score.loc[(c > kj) & ~pk_up], 50 + 25*w_price_kijun)
    score.loc[(c < kj) & ~pk_dn] = np.minimum(score.loc[(c < kj) & ~pk_dn], 50 - 25*w_price_kijun)

    # 2. Tenkan/Kijun Cross
    tk_kj_up = (tk.shift(1) < kj.shift(1)) & (tk >= kj); score.loc[tk_kj_up] = np.maximum(score.loc[tk_kj_up], 50 + 50*w_tk_kj_cross)
    tk_kj_dn = (tk.shift(1) > kj.shift(1)) & (tk <= kj); score.loc[tk_kj_dn] = np.minimum(score.loc[tk_kj_dn], 50 - 50*w_tk_kj_cross)

    # 3. Price vs Cloud (Kumo)
    cloud_top = np.maximum(sa, sb)
    cloud_bottom = np.minimum(sa, sb)
    price_above_cloud = c > cloud_top; score.loc[price_above_cloud] = np.maximum(score.loc[price_above_cloud], 50 + 50*w_price_cloud)
    price_below_cloud = c < cloud_bottom; score.loc[price_below_cloud] = np.minimum(score.loc[price_below_cloud], 50 - 50*w_price_cloud)
    # In cloud:
    price_in_cloud = (c >= cloud_bottom) & (c <= cloud_top)
    score.loc[price_in_cloud & (c > c.shift(1))] = np.maximum(score.loc[price_in_cloud & (c > c.shift(1))], 55.0) # Rising in cloud
    score.loc[price_in_cloud & (c < c.shift(1))] = np.minimum(score.loc[price_in_cloud & (c < c.shift(1))], 45.0) # Falling in cloud
    
    # 4. Cloud Twist (Senkou A vs Senkou B) - Future signal
    cloud_twist_up = (sa.shift(1) < sb.shift(1)) & (sa >= sb); score.loc[cloud_twist_up] = np.maximum(score.loc[cloud_twist_up], 50 + 25*w_cloud_twist) # Milder effect
    cloud_twist_dn = (sa.shift(1) > sb.shift(1)) & (sa <= sb); score.loc[cloud_twist_dn] = np.minimum(score.loc[cloud_twist_dn], 50 - 25*w_cloud_twist)

    # 5. Chikou Span vs Price (Chikou is price shifted back 26 periods)
    # We need price 26 periods ago. If data is too short, this will be NaN.
    price_26_ago = c.shift(26) # This is the price that Chikou (cs) is compared against
    chikou_valid = price_26_ago.notna()
    
    cs_above_price = cs > price_26_ago; score.loc[cs_above_price & chikou_valid] = np.maximum(score.loc[cs_above_price & chikou_valid], 50 + 50*w_chikou_price)
    cs_below_price = cs < price_26_ago; score.loc[cs_below_price & chikou_valid] = np.minimum(score.loc[cs_below_price & chikou_valid], 50 - 50*w_chikou_price)
            
    return score.clip(0, 100)

def calculate_mom_score(mom: pd.Series) -> pd.Series:
    """MOM (Momentum) 评分 (0-100)。"""
    mom_s, = _safe_fillna_series([mom], [0.0]) # MOM 中性0
    if mom_s.isnull().all():
        return pd.Series(50.0, index=mom.index).clip(0,100)

    score = pd.Series(50.0, index=mom_s.index)
    buy_cross = (mom_s.shift(1) < 0) & (mom_s >= 0)
    sell_cross = (mom_s.shift(1) > 0) & (mom_s <= 0)
    score.loc[buy_cross] = 65.0
    score.loc[sell_cross] = 35.0

    not_cross_cond = ~buy_cross & ~sell_cross
    bullish_trend = (mom_s > 0) & (mom_s > mom_s.shift(1)) & not_cross_cond
    bearish_trend = (mom_s < 0) & (mom_s < mom_s.shift(1)) & not_cross_cond
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # Original was 55
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # Original was 45
            
    return score.clip(0, 100)

def calculate_willr_score(willr: pd.Series) -> pd.Series: # Removed params, as it's not used
    """WILLR (%R) 评分 (0-100)。"""
    willr_s, = _safe_fillna_series([willr], [-50.0]) # %R 中性-50
    if willr_s.isnull().all():
        return pd.Series(50.0, index=willr.index).clip(0,100)

    score = pd.Series(50.0, index=willr_s.index)
    ob_th, os_th = -20, -80
    ext_ob_th, ext_os_th = -10, -90

    score.loc[willr_s < ext_os_th] = 95.0
    score.loc[willr_s > ext_ob_th] = 5.0

    score.loc[(willr_s >= ext_os_th) & (willr_s < os_th)] = np.maximum(score.loc[(willr_s >= ext_os_th) & (willr_s < os_th)], 85.0)
    score.loc[(willr_s <= ext_ob_th) & (willr_s > ob_th)] = np.minimum(score.loc[(willr_s <= ext_ob_th) & (willr_s > ob_th)], 15.0)

    buy_signal = (willr_s.shift(1) < os_th) & (willr_s >= os_th)
    sell_signal = (willr_s.shift(1) > ob_th) & (willr_s <= ob_th)
    score.loc[buy_signal] = np.maximum(score.loc[buy_signal], 75.0)
    score.loc[sell_signal] = np.minimum(score.loc[sell_signal], 25.0)

    not_signal_cond = ~buy_signal & ~sell_signal
    neutral_zone_cond = (willr_s >= os_th) & (willr_s <= ob_th) & not_signal_cond
    
    # WILLR is inverted: lower values are more oversold (bullish), higher are overbought (bearish)
    # So, if WILLR is rising in neutral zone, it's moving towards overbought (bearish)
    # If WILLR is falling in neutral zone, it's moving towards oversold (bullish)
    trend_to_ob = neutral_zone_cond & (willr_s > willr_s.shift(1)) # Moving towards -20 (bearish)
    trend_to_os = neutral_zone_cond & (willr_s < willr_s.shift(1)) # Moving towards -80 (bullish)
    score.loc[trend_to_ob] = np.minimum(score.loc[trend_to_ob], 45.0) 
    score.loc[trend_to_os] = np.maximum(score.loc[trend_to_os], 55.0)
            
    return score.clip(0, 100)

def calculate_cmf_score(cmf: pd.Series) -> pd.Series:
    """CMF (Chaikin Money Flow) 评分 (0-100)。"""
    cmf_s, = _safe_fillna_series([cmf], [0.0]) # CMF 中性0
    if cmf_s.isnull().all():
        return pd.Series(50.0, index=cmf.index).clip(0,100)

    score = pd.Series(50.0, index=cmf_s.index)
    
    # Base score on position relative to zero
    score.loc[cmf_s > 0] = 60.0
    score.loc[cmf_s < 0] = 40.0

    # Modify based on trend, ensuring not to override stronger signals if they existed
    bullish_trend = cmf_s > cmf_s.shift(1)
    bearish_trend = cmf_s < cmf_s.shift(1)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # If CMF > 0 and rising, score remains 60. If CMF < 0 but rising, score becomes 55.
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # If CMF < 0 and falling, score remains 40. If CMF > 0 but falling, score becomes 45.
            
    return score.clip(0, 100)

def calculate_obv_score(obv: pd.Series) -> pd.Series:
    """OBV (On Balance Volume) 评分 (0-100)。"""
    obv_s, = _safe_fillna_series([obv], [None]) # ffill/bfill first
    if obv_s.isnull().all():
        return pd.Series(50.0, index=obv.index).clip(0,100)
    obv_s = obv_s.fillna(obv_s.mean()) # Fill remaining with mean
    if obv_s.isnull().all():
        return pd.Series(50.0, index=obv.index).clip(0,100)

    score = pd.Series(50.0, index=obv_s.index)
    bullish_trend = obv_s > obv_s.shift(1)
    bearish_trend = obv_s < obv_s.shift(1)
    score.loc[bullish_trend] = 60.0
    score.loc[bearish_trend] = 40.0
                
    return score.clip(0, 100)

def calculate_kc_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """KC (Keltner Channel) 评分 (0-100)。"""
    # Similar to BOLL, NaN handling for bands is key.
    if close.isnull().all():
        return pd.Series(50.0, index=close.index).clip(0,100)

    close_f = close.ffill().bfill()
    # Fallback for bands: use close's rolling mean and a typical ATR multiple (e.g., 1.5-2x of a default ATR calc)
    # This is a rough estimation if actual ATR isn't available or bands are missing.
    mid_f = mid.ffill().bfill().fillna(close_f.rolling(20, min_periods=1).mean())
    
    # Simplified fallback for band width if upper/lower are all NaN
    # A better approach would be to pass ATR or calculate it if HLC data is available
    typical_range = (close_f.rolling(20, min_periods=1).max() - close_f.rolling(20, min_periods=1).min()).fillna(0.02 * close_f.mean()) # Approx range
    
    upper_f = upper.ffill().bfill().fillna(mid_f + typical_range)
    lower_f = lower.ffill().bfill().fillna(mid_f - typical_range)

    if any(s.isnull().all() for s in [close_f, upper_f, mid_f, lower_f]):
         logger.warning("KC 评分：一个或多个关键序列在填充后仍全为NaN。")
         return pd.Series(50.0, index=close.index).clip(0,100)

    score = pd.Series(50.0, index=close_f.index)

    score.loc[close_f <= lower_f] = 90.0
    buy_support = (close_f.shift(1) < lower_f.shift(1)) & (close_f >= lower_f)
    score.loc[buy_support] = np.maximum(score.loc[buy_support], 80.0)

    score.loc[close_f >= upper_f] = 10.0
    sell_pressure = (close_f.shift(1) > upper_f.shift(1)) & (close_f <= upper_f)
    score.loc[sell_pressure] = np.minimum(score.loc[sell_pressure], 20.0)

    buy_mid_cross = (close_f.shift(1) < mid_f.shift(1)) & (close_f >= mid_f)
    sell_mid_cross = (close_f.shift(1) > mid_f.shift(1)) & (close_f <= mid_f)
    score.loc[buy_mid_cross] = np.maximum(score.loc[buy_mid_cross], 65.0)
    score.loc[sell_mid_cross] = np.minimum(score.loc[sell_mid_cross], 35.0)

    not_extreme_cond = (close_f > lower_f) & (close_f < upper_f)
    not_mid_cross_cond = ~buy_mid_cross & ~sell_mid_cross
    
    is_above_mid = not_extreme_cond & not_mid_cross_cond & (close_f > mid_f)
    is_below_mid = not_extreme_cond & not_mid_cross_cond & (close_f < mid_f)
    score.loc[is_above_mid] = np.maximum(score.loc[is_above_mid], 55.0)
    score.loc[is_below_mid] = np.minimum(score.loc[is_below_mid], 45.0)
                
    return score.clip(0, 100)

def calculate_hv_score(hv: pd.Series) -> pd.Series:
    """HV (Historical Volatility) 评分 (0-100)。"""
    hv_s, = _safe_fillna_series([hv], [None]) # ffill/bfill first
    if hv_s.isnull().all():
        return pd.Series(50.0, index=hv.index).clip(0,100)
    hv_s = hv_s.fillna(hv_s.mean())
    if hv_s.isnull().all():
        return pd.Series(50.0, index=hv.index).clip(0,100)

    score = pd.Series(50.0, index=hv_s.index)
    rolling_window = min(len(hv_s), 20)
    min_periods_rolling = max(1, int(rolling_window * 0.5)) if rolling_window > 0 else 1

    hv_mean = hv_s.rolling(window=rolling_window, min_periods=min_periods_rolling).mean().fillna(hv_s.mean())
    hv_std = hv_s.rolling(window=rolling_window, min_periods=min_periods_rolling).std().fillna(hv_s.std()).fillna(0)

    high_volatility = hv_s > (hv_mean + 0.5 * hv_std)
    low_volatility = hv_s < (hv_mean - 0.5 * hv_std)
    
    score.loc[high_volatility] = np.maximum(score.loc[high_volatility], 60.0)
    score.loc[low_volatility] = np.minimum(score.loc[low_volatility], 40.0)
                
    return score.clip(0, 100)

def calculate_vroc_score(vroc: pd.Series) -> pd.Series:
    """VROC (Volume Rate of Change) 评分 (0-100)。"""
    vroc_s, = _safe_fillna_series([vroc], [0.0]) # VROC 中性0
    if vroc_s.isnull().all():
        return pd.Series(50.0, index=vroc.index).clip(0,100)

    score = pd.Series(50.0, index=vroc_s.index)
    buy_cross = (vroc_s.shift(1) < 0) & (vroc_s >= 0)
    sell_cross = (vroc_s.shift(1) > 0) & (vroc_s <= 0)
    score.loc[buy_cross] = 55.0 # Volume supporting, slightly positive
    score.loc[sell_cross] = 45.0 # Volume waning, slightly negative

    not_cross_cond = ~buy_cross & ~sell_cross
    bullish_trend = (vroc_s > 0) & (vroc_s > vroc_s.shift(1)) & not_cross_cond
    bearish_trend = (vroc_s < 0) & (vroc_s < vroc_s.shift(1)) & not_cross_cond
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 52.0) # Mildly positive
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 48.0) # Mildly negative
                
    return score.clip(0, 100)

def calculate_aroc_score(aroc: pd.Series) -> pd.Series:
    """AROC (Absolute Rate of Change) 评分 (0-100)。"""
    # AROC is likely an alias for ROC, using same logic as calculate_roc_score
    # If AROC has a specific different interpretation (e.g. Aroon Oscillator), the logic would change.
    # Assuming it's similar to Price ROC:
    aroc_s, = _safe_fillna_series([aroc], [0.0]) # AROC 中性0
    if aroc_s.isnull().all():
        return pd.Series(50.0, index=aroc.index).clip(0,100)

    score = pd.Series(50.0, index=aroc_s.index)
    buy_cross = (aroc_s.shift(1) < 0) & (aroc_s >= 0)
    sell_cross = (aroc_s.shift(1) > 0) & (aroc_s <= 0)
    score.loc[buy_cross] = 65.0 # Stronger signal than VROC as it's price based
    score.loc[sell_cross] = 35.0

    not_cross_cond = ~buy_cross & ~sell_cross
    bullish_trend = (aroc_s > 0) & (aroc_s > aroc_s.shift(1)) & not_cross_cond
    bearish_trend = (aroc_s < 0) & (aroc_s < aroc_s.shift(1)) & not_cross_cond
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)
                
    return score.clip(0, 100)

def calculate_pivot_score(close: pd.Series, pivot_levels_df: pd.DataFrame,
                          tf: str, # 增加时间框架参数，用于构建标准列名
                          params: Optional[Dict] = None) -> pd.Series:
    """
    Pivot Points 评分 (0-100)。
    评分逻辑基于收盘价相对于 Pivot Point (PP) 和各支撑/阻力水平的位置。
    价格在 PP 上方偏多，下方偏空。突破阻力看涨，跌破支撑看跌。

    优化：
    - 改进列名查找和级别解析的健壮性。
    - 优化NaN处理，避免不当填充。
    - 结构化评分逻辑。

    Args:
        close (pd.Series): 收盘价序列。
        pivot_levels_df (pd.DataFrame): 包含 Pivot 水平的 DataFrame。
                                        列名应为标准格式如 'PP_D', 'R1_D', 'F_S1_D'。
        tf (str): 当前使用的时间框架，用于匹配正确的 Pivot 列。
        params (Dict, optional): 评分函数可能需要的额外参数。目前未使用。

    Returns:
        pd.Series: 计算出的 Pivot Points 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保索引一致
    if not close.index.equals(pivot_levels_df.index):
        logger.warning("Pivot Points 评分：输入 close 和 pivot_levels_df 索引不一致。将尝试对齐。")
        # 以 close 的索引为准对齐 pivot_levels_df，缺失的 pivot levels 会是 NaN
        pivot_levels_df = pivot_levels_df.reindex(close.index)
        # 如果 close 长度变化，score 也需要重新初始化
        if not score.index.equals(close.index):
            score = pd.Series(50.0, index=close.index)

    # 填充 close 的 NaN 值
    # 对于 pivot_levels_df 中的 NaN，我们不应随意填充，因为 NaN 代表该水平不存在或未计算。
    # 在比较时，如果 pivot level 是 NaN，则该比较应视为无效。
    close_filled = close.ffill().bfill()
    if close_filled.isnull().all():
        logger.warning("Pivot Points 评分：收盘价序列在填充后仍全为 NaN。")
        return score.clip(0, 100) # 返回默认中性分

    # 1. 价格与 Pivot Point (PP) 的相对位置
    pp_col_name = f'PP_{tf}' # 标准 Pivot Point 列名
    if pp_col_name in pivot_levels_df.columns and pivot_levels_df[pp_col_name].notna().any():
        pp_series = pivot_levels_df[pp_col_name]
        # 只在 pp_series 非 NaN 的地方进行比较和打分
        valid_pp_mask = pp_series.notna()
        score.loc[valid_pp_mask & (close_filled > pp_series)] = 55.0
        score.loc[valid_pp_mask & (close_filled < pp_series)] = 45.0
        score.loc[valid_pp_mask & (close_filled == pp_series)] = 50.0 # 精确在PP上
    else:
        logger.warning(f"未找到有效的 Pivot Point 列 '{pp_col_name}' 或该列全为 NaN，PP 相关评分跳过。")


    # 2. 价格突破/跌破支撑与阻力水平
    # 定义标准和斐波那契支撑/阻力级别的前缀和最大级别
    level_types = {
        'R': {'prefix': f'R', 'max_level': 4, 'base_score_breakout': 70, 'base_score_breakdown': None, 'level_multiplier': 5, 'is_resistance': True},
        'S': {'prefix': f'S', 'max_level': 4, 'base_score_breakout': None, 'base_score_breakdown': 30, 'level_multiplier': 5, 'is_resistance': False},
        'F_R': {'prefix': f'F_R', 'max_level': 3, 'base_score_breakout': 75, 'base_score_breakdown': None, 'level_multiplier': 5, 'is_resistance': True},
        'F_S': {'prefix': f'F_S', 'max_level': 3, 'base_score_breakout': None, 'base_score_breakdown': 25, 'level_multiplier': 5, 'is_resistance': False},
    }

    for type_key, config in level_types.items():
        for level in range(1, config['max_level'] + 1):
            level_col_name = f"{config['prefix']}{level}_{tf}"

            if level_col_name in pivot_levels_df.columns and pivot_levels_df[level_col_name].notna().any():
                level_series = pivot_levels_df[level_col_name]
                valid_level_mask = level_series.notna() # 只在 pivot level 非 NaN 的地方操作

                # 确保 shift 后的 series 也有相同的 valid_level_mask 应用，或者在比较前处理
                # close_prev = close_filled.shift(1)[valid_level_mask]
                # level_prev = level_series.shift(1)[valid_level_mask]
                # current_close = close_filled[valid_level_mask]
                # current_level = level_series[valid_level_mask]
                
                # 为了避免索引问题，直接在完整序列上操作，然后用 valid_level_mask 过滤结果
                close_prev_full = close_filled.shift(1)
                level_prev_full = level_series.shift(1)


                if config['is_resistance']: # 处理阻力位
                    # 价格向上突破阻力
                    breakout_cond_full = (close_prev_full < level_prev_full) & (close_filled >= level_series)
                    breakout_cond = breakout_cond_full & valid_level_mask # 应用掩码
                    
                    if breakout_cond.any():
                        breakout_score_value = config['base_score_breakout'] + level * config['level_multiplier']
                        score.loc[breakout_cond] = np.maximum(score.loc[breakout_cond], breakout_score_value)
                    
                    # 价格在阻力位下方（未突破时，作为阻力区的参考）
                    below_resistance_cond = (close_filled < level_series) & (~breakout_cond_full) & valid_level_mask
                    if below_resistance_cond.any():
                         # 越接近高级别阻力，分数越低（更看跌）
                        penalty = level * 2.5 # 示例惩罚值
                        score.loc[below_resistance_cond] = np.minimum(score.loc[below_resistance_cond], 50 - penalty)

                else: # 处理支撑位
                    # 价格向下跌破支撑
                    breakdown_cond_full = (close_prev_full > level_prev_full) & (close_filled <= level_series)
                    breakdown_cond = breakdown_cond_full & valid_level_mask # 应用掩码

                    if breakdown_cond.any():
                        breakdown_score_value = config['base_score_breakdown'] - level * config['level_multiplier']
                        score.loc[breakdown_cond] = np.minimum(score.loc[breakdown_cond], breakdown_score_value)

                    # 价格在支撑位上方（未跌破时，作为支撑区的参考）
                    above_support_cond = (close_filled > level_series) & (~breakdown_cond_full) & valid_level_mask
                    if above_support_cond.any():
                        # 越接近高级别支撑，分数越高（更看涨）
                        bonus = level * 2.5 # 示例奖励值
                        score.loc[above_support_cond] = np.maximum(score.loc[above_support_cond], 50 + bonus)
            # else:
            #     logger.debug(f"Pivot level列 '{level_col_name}' 不存在或全为 NaN，跳过。")


    return score.clip(0, 100)

def calculate_all_indicator_scores(data: pd.DataFrame,
                                   bs_params: Dict,
                                   indicator_configs: List[Dict]
                                   ) -> pd.DataFrame:
    """
    根据配置计算所有指定指标的评分 (0-100)。

    此函数遍历 base_scoring 参数中指定的需要评分的指标和时间框架，
    从输入的 DataFrame 中查找对应的指标数据列，并调用相应的评分计算函数。
    指标列名根据 indicator_naming_conventions.json 文件中的命名规范构建。

    :param data: 包含所有原始数据和指标的 DataFrame。列名应包含时间级别后缀，
                 例如 'close_15', 'RSI_14_30'。
    :param bs_params: base_scoring 参数字典，包含 'score_indicators' (需要评分的指标列表)
                      和 'timeframes' (需要计算评分的时间框架列表)，以及各指标的参数。
    :param indicator_configs: 由 indicator_services.prepare_strategy_dataframe 生成的，
                              包含每个指标计算函数、参数、时间框架的列表。用于辅助查找列名。
    :return: 返回一个 DataFrame，包含所有时间框架和指标的评分列。
             列名格式: SCORE_{指标名}_{时间级别}。
             如果某个指标或时间框架的数据缺失或计算失败，对应的评分列将填充默认中性分 50.0。
    """
    # 初始化用于存储所有评分结果的 DataFrame
    scoring_results = pd.DataFrame(index=data.index)

    # 获取需要评分的指标列表和时间框架列表
    score_indicators_keys = bs_params.get('score_indicators', [])
    score_timeframes = bs_params.get('timeframes', [])

    # 如果没有配置需要评分的指标或时间框架，则直接返回空的 DataFrame
    if not score_indicators_keys or not score_timeframes:
        logger.warning("未配置需要评分的指标或时间框架 (base_scoring.score_indicators 或 base_scoring.timeframes)。")
        return scoring_results

    logger.info(f"开始计算指标评分，指标: {score_indicators_keys}, 时间框架: {score_timeframes}")

    # 加载 indicator_naming_conventions.json 文件中的命名规范
    naming_conventions_path = settings.INDICATOR_PARAMETERS_CONFIG_PATH
    try:
        with open(naming_conventions_path, 'r', encoding='utf-8') as f:
            naming_conventions = json.load(f)
        indicator_naming = naming_conventions.get('indicator_naming_conventions', {})
        logger.info("成功加载指标命名规范配置文件。")
    except Exception as e:
        logger.error(f"加载指标命名规范配置文件失败: {e}，将使用默认命名逻辑。")
        indicator_naming = {}

    # 从 indicator_configs 中提取可能的列名信息
    # 修改：利用 indicator_configs 获取更精确的列名映射
    config_column_mapping = {}
    for config in indicator_configs:
        indicator_name = config.get('name', '').lower()
        timeframe = config.get('timeframe', '')
        output_columns = config.get('output_columns', [])
        if indicator_name and timeframe and output_columns:
            key = (indicator_name, str(timeframe))
            config_column_mapping[key] = output_columns
            logger.debug(f"从配置中提取指标 {indicator_name} 在时间框架 {timeframe} 的列名: {output_columns}")

    # 遍历需要评分的指标 key 和时间框架
    for indicator_key in score_indicators_keys:
        # 查找该 indicator_key 对应的评分函数
        score_func = None
        # 存储计算评分函数可能需要的参数字典 (从 bs_params 获取)
        score_func_params = {}
        # 存储用于查找列名的参数字典 (可能与评分函数参数不同)
        col_lookup_params = {}

        # 根据 indicator_key 确定评分函数和获取参数的方式
        if indicator_key == 'macd':
            score_func = calculate_macd_score
            col_lookup_params = {
                'period_fast': bs_params.get('macd_fast', 12),
                'period_slow': bs_params.get('macd_slow', 26),
                'signal_period': bs_params.get('macd_signal', 9)
            }
            score_func_params = {}
        elif indicator_key == 'rsi':
            score_func = calculate_rsi_score
            col_lookup_params = {'period': bs_params.get('rsi_period', 14)}
            score_func_params = {
                'rsi_oversold': bs_params.get('rsi_oversold', 30),
                'rsi_overbought': bs_params.get('rsi_overbought', 70),
                'rsi_extreme_oversold': bs_params.get('rsi_extreme_oversold', 20),
                'rsi_extreme_overbought': bs_params.get('rsi_extreme_overbought', 80)
            }
        elif indicator_key == 'kdj':
            score_func = calculate_kdj_score
            col_lookup_params = {
                'period': bs_params.get('kdj_period_k', 9),  # K周期
                'signal_period': bs_params.get('kdj_period_d', 3),  # D周期
                'smooth_k_period': bs_params.get('kdj_period_j', 3)  # J周期 (通常是K的平滑)
            }
            score_func_params = {
                'kdj_oversold': bs_params.get('kdj_oversold', 20),
                'kdj_overbought': bs_params.get('kdj_overbought', 80),
                'kdj_extreme_oversold': bs_params.get('kdj_extreme_oversold', 10),
                'kdj_extreme_overbought': bs_params.get('kdj_extreme_overbought', 90)
            }
        elif indicator_key == 'boll':
            score_func = calculate_boll_score
            col_lookup_params = {
                'period': bs_params.get('boll_period', 20),
                'std_dev': bs_params.get('boll_std_dev', 2.0)
            }
            score_func_params = {}
        elif indicator_key == 'cci':
            score_func = calculate_cci_score
            col_lookup_params = {'period': bs_params.get('cci_period', 14)}
            score_func_params = {
                'cci_threshold': bs_params.get('cci_threshold', 100),
                'cci_extreme_threshold': bs_params.get('cci_extreme_threshold', 200)
            }
        elif indicator_key == 'mfi':
            score_func = calculate_mfi_score
            col_lookup_params = {'period': bs_params.get('mfi_period', 14)}
            score_func_params = {
                'mfi_oversold': bs_params.get('mfi_oversold', 20),
                'mfi_overbought': bs_params.get('mfi_overbought', 80),
                'mfi_extreme_oversold': bs_params.get('mfi_extreme_oversold', 10),
                'mfi_extreme_overbought': bs_params.get('mfi_extreme_overbought', 90)
            }
        elif indicator_key == 'roc':
            score_func = calculate_roc_score
            col_lookup_params = {'period': bs_params.get('roc_period', 12)}
            score_func_params = {}
        elif indicator_key == 'dmi':
            score_func = calculate_dmi_score
            col_lookup_params = {'period': bs_params.get('dmi_period', 14)}
            score_func_params = {
                'adx_threshold': bs_params.get('adx_threshold', 25),
                'adx_strong_threshold': bs_params.get('adx_strong_threshold', 40)
            }
        elif indicator_key == 'sar':
            score_func = calculate_sar_score
            col_lookup_params = {'af_step': bs_params.get('sar_af_step', 0.02), 'max_af': bs_params.get('sar_max_af', 0.2)}
            score_func_params = {}
        elif indicator_key == 'stoch':
            score_func = calculate_stoch_score
            col_lookup_params = {
                'k_period': bs_params.get('stoch_k_period', 14),
                'd_period': bs_params.get('stoch_d_period', 3),
                'smooth_k_period': bs_params.get('stoch_smooth_k_period', 3)  # STOCH D线通常是K线的平滑
            }
            score_func_params = {
                'stoch_oversold': bs_params.get('stoch_oversold', 20),
                'stoch_overbought': bs_params.get('stoch_overbought', 80),
                'stoch_extreme_oversold': bs_params.get('stoch_extreme_oversold', 10),
                'stoch_extreme_overbought': bs_params.get('stoch_extreme_overbought', 90)
            }
        elif indicator_key == 'ema':
            score_func = calculate_ma_score  # 假设存在一个通用的 MA 评分函数
            col_lookup_params = {'period': bs_params.get('ema_params', {}).get('period', 20)}
            score_func_params = {'ma_type': 'ema'}  # 传递 MA 类型给通用评分函数
        elif indicator_key == 'sma':
            score_func = calculate_ma_score  # 假设存在一个通用的 MA 评分函数
            col_lookup_params = {'period': bs_params.get('sma_params', {}).get('period', 20)}
            score_func_params = {'ma_type': 'sma'}  # 传递 MA 类型给通用评分函数
        elif indicator_key == 'atr':
            score_func = calculate_atr_score  # 假设存在 calculate_atr_score 函数
            col_lookup_params = {'period': bs_params.get('atr_params', {}).get('period', 14)}
            score_func_params = {}  # ATR 评分函数可能不需要额外参数
        elif indicator_key == 'adl':
            score_func = calculate_adl_score  # 假设存在 calculate_adl_score 函数
            col_lookup_params = {}  # ADL 通常没有周期参数
            score_func_params = {}  # ADL 评分函数可能不需要额外参数
        elif indicator_key == 'vwap':
            score_func = calculate_vwap_score  # 假设存在 calculate_vwap_score 函数
            col_lookup_params = {'anchor': bs_params.get('vwap_anchor', None)}  # VWAP 可能有 anchor 参数
            score_func_params = {}  # VWAP 评分函数可能不需要额外参数
        elif indicator_key == 'ichimoku':
            score_func = calculate_ichimoku_score  # 假设存在 calculate_ichimoku_score 函数
            col_lookup_params = {
                'tenkan_period': bs_params.get('ichimoku_tenkan', 9),
                'kijun_period': bs_params.get('ichimoku_kijun', 26),
                'senkou_period': bs_params.get('ichimoku_senkou', 52)
            }
            score_func_params = {}  # Ichimoku 评分函数可能不需要额外参数
        elif indicator_key == 'mom':
            score_func = calculate_mom_score  # 假设存在 calculate_mom_score 函数
            col_lookup_params = {'period': bs_params.get('mom_params', {}).get('period', 10)}
            score_func_params = {}  # MOM 评分函数可能不需要额外参数
        elif indicator_key == 'willr':
            score_func = calculate_willr_score  # 假设存在 calculate_willr_score 函数
            col_lookup_params = {'period': bs_params.get('willr_params', {}).get('period', 14)}
            score_func_params = {}  # WILLR 评分函数可能不需要额外参数
        elif indicator_key == 'cmf':
            score_func = calculate_cmf_score  # 假设存在 calculate_cmf_score 函数
            col_lookup_params = {'period': bs_params.get('cmf_period', 20)}
            score_func_params = {}  # CMF 评分函数可能不需要额外参数
        elif indicator_key == 'obv':
            score_func = calculate_obv_score  # 假设存在 calculate_obv_score 函数
            col_lookup_params = {}  # OBV 通常没有周期参数
            score_func_params = {}  # OBV 评分函数可能不需要额外参数
        elif indicator_key == 'kc':
            score_func = calculate_kc_score  # 假设存在 calculate_kc_score 函数
            col_lookup_params = {
                'ema_period': bs_params.get('kc_params', {}).get('ema_period', 20),
                'atr_period': bs_params.get('kc_params', {}).get('atr_period', 10)
            }
            score_func_params = {}  # KC 评分函数可能不需要额外参数
        elif indicator_key == 'hv':
            score_func = calculate_hv_score  # 假设存在 calculate_hv_score 函数
            col_lookup_params = {'period': bs_params.get('hv_params', {}).get('period', 20)}
            score_func_params = {}  # HV 评分函数可能不需要额外参数
        elif indicator_key == 'vroc':
            score_func = calculate_vroc_score  # 假设存在 calculate_vroc_score 函数
            col_lookup_params = {'period': bs_params.get('vroc_params', {}).get('period', 10)}
            score_func_params = {}  # VROC 评分函数可能不需要额外参数
        elif indicator_key == 'aroc':
            score_func = calculate_aroc_score  # 假设存在 calculate_aroc_score 函数
            col_lookup_params = {'period': bs_params.get('aroc_params', {}).get('period', 10)}
            score_func_params = {}  # AROC 评分函数可能不需要额外参数
        elif indicator_key == 'pivot':
            score_func = calculate_pivot_score  # 假设存在 calculate_pivot_score 函数
            col_lookup_params = {}  # Pivot Points 通常基于日线，且列名固定
            score_func_params = {}

        # 如果指标 key 未找到对应的评分函数，则跳过
        if score_func is None:
            logger.warning(f"指标 '{indicator_key}' 未找到对应的评分函数，跳过评分计算。")
            continue

        # 遍历需要计算评分的时间框架
        for tf_score in score_timeframes:
            # 查找指标列名 (这部分逻辑需要根据指标类型和 pandas_ta 的命名规则来确定)
            indicator_cols_for_score = {}  # {参数名: 列名} 或 {指标线名: 列名}

            try:
                # 修改：添加更健壮的列名查找逻辑，尝试多种时间框架后缀格式
                possible_tf_suffixes = [
                    str(tf_score), f"{tf_score}m", f"{tf_score}min", tf_score.upper(),
                    f"{tf_score}M", f"{tf_score}MIN", f"T{tf_score}", f"t{tf_score}"
                ]
                found = False
                # 首先检查 indicator_configs 中的列名映射
                config_key = (indicator_key.lower(), str(tf_score))
                if config_key in config_column_mapping:
                    config_cols = config_column_mapping[config_key]
                    logger.debug(f"使用 indicator_configs 提供的列名映射: {config_cols} for {indicator_key} at {tf_score}")
                    if indicator_key == 'macd':
                        if len(config_cols) >= 3 and all(col in data.columns for col in config_cols[:3]):
                            indicator_cols_for_score = {
                                'macd_series': config_cols[0],  # MACD (diff)
                                'macd_d': config_cols[2],       # MACD signal (DEA)
                                'macd_h': config_cols[1]        # MACD hist
                            }
                            found = True
                            logger.debug(f"找到指标 'macd' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'rsi':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'rsi': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'rsi' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'kdj':
                        if len(config_cols) >= 3 and all(col in data.columns for col in config_cols[:3]):
                            indicator_cols_for_score = {'k': config_cols[0], 'd': config_cols[1], 'j': config_cols[2]}
                            found = True
                            logger.debug(f"找到指标 'kdj' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'boll':
                        if len(config_cols) >= 3 and all(col in data.columns for col in config_cols[:3]):
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and all(col in data.columns for col in [close_col] + config_cols[:3]):
                                indicator_cols_for_score = {'close': close_col, 'upper': config_cols[2], 'mid': config_cols[1], 'lower': config_cols[0]}
                                found = True
                                logger.debug(f"找到指标 'boll' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'cci':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'cci': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'cci' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'mfi':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'mfi': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'mfi' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'roc':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'roc': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'roc' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'dmi':
                        if len(config_cols) >= 3 and all(col in data.columns for col in config_cols[:3]):
                            indicator_cols_for_score = {'pdi': config_cols[0], 'ndi': config_cols[1], 'adx': config_cols[2]}
                            found = True
                            logger.debug(f"找到指标 'dmi' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'sar':
                        if config_cols and config_cols[0] in data.columns:
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and config_cols[0] in data.columns:
                                indicator_cols_for_score = {'close': close_col, 'sar': config_cols[0]}
                                found = True
                                logger.debug(f"找到指标 'sar' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'stoch':
                        if len(config_cols) >= 2 and all(col in data.columns for col in config_cols[:2]):
                            indicator_cols_for_score = {'k': config_cols[0], 'd': config_cols[1]}
                            found = True
                            logger.debug(f"找到指标 'stoch' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key in ['ema', 'sma']:
                        if config_cols and config_cols[0] in data.columns:
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and config_cols[0] in data.columns:
                                indicator_cols_for_score = {'close': close_col, 'ma': config_cols[0]}
                                found = True
                                logger.debug(f"找到指标 '{indicator_key}' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'atr':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'atr': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'atr' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'adl':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'adl': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'adl' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'vwap':
                        if config_cols and config_cols[0] in data.columns:
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and config_cols[0] in data.columns:
                                indicator_cols_for_score = {'close': close_col, 'vwap': config_cols[0]}
                                found = True
                                logger.debug(f"找到指标 'vwap' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'ichimoku':
                        if len(config_cols) >= 5 and all(col in data.columns for col in config_cols[:5]):
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and all(col in data.columns for col in config_cols[:5]):
                                indicator_cols_for_score = {
                                    'close': close_col, 'tenkan': config_cols[0], 'kijun': config_cols[1],
                                    'senkou_a': config_cols[3], 'senkou_b': config_cols[4], 'chikou': config_cols[2]
                                }
                                found = True
                                logger.debug(f"找到指标 'ichimoku' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'mom':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'mom': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'mom' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'willr':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'willr': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'willr' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'cmf':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'cmf': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'cmf' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'obv':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'obv': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'obv' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'kc':
                        if len(config_cols) >= 3 and all(col in data.columns for col in config_cols[:3]):
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and all(col in data.columns for col in config_cols[:3]):
                                indicator_cols_for_score = {'close': close_col, 'upper': config_cols[2], 'mid': config_cols[1], 'lower': config_cols[0]}
                                found = True
                                logger.debug(f"找到指标 'kc' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'hv':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'hv': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'hv' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'vroc':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'vroc': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'vroc' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'aroc':
                        if config_cols and config_cols[0] in data.columns:
                            indicator_cols_for_score = {'aroc': config_cols[0]}
                            found = True
                            logger.debug(f"找到指标 'aroc' 在时间框架 {tf_score} 的列，使用配置映射")
                    elif indicator_key == 'pivot':
                        if len(config_cols) >= 15 and all(col in data.columns for col in config_cols[:15]):
                            close_col = f"close_{tf_score}" if f"close_{tf_score}" in data.columns else next((c for c in data.columns if c.startswith("close_")), None)
                            if close_col and all(col in data.columns for col in config_cols[:15]):
                                indicator_cols_for_score = {'close': close_col, 'pivot_levels': config_cols[:15]}
                                found = True
                                logger.debug(f"找到指标 'pivot' 在时间框架 {tf_score} 的列，使用配置映射")

                # 如果配置映射未找到匹配项，则回退到原来的后缀尝试逻辑
                if not found:
                    for tf_suffix in possible_tf_suffixes:
                        if indicator_key == 'macd' and 'MACD' in indicator_naming:
                            p_fast = col_lookup_params.get('period_fast', 12)
                            p_slow = col_lookup_params.get('period_slow', 26)
                            p_sig = col_lookup_params.get('signal_period', 9)
                            macd_col = f"MACD_{p_fast}_{p_slow}_{p_sig}_{tf_suffix}"
                            dea_col = f"MACDs_{p_fast}_{p_slow}_{p_sig}_{tf_suffix}"  # pandas_ta MACD signal 列名通常是 MACDs
                            hist_col = f"MACDh_{p_fast}_{p_slow}_{p_sig}_{tf_suffix}"
                            if all(col in data.columns for col in [macd_col, dea_col, hist_col]):
                                indicator_cols_for_score = {
                                    'macd_series': macd_col,  # MACD (diff)
                                    'macd_d': dea_col,        # MACD signal (DEA)
                                    'macd_h': hist_col        # MACD hist
                                }
                                found = True
                                logger.debug(f"找到指标 'macd' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'rsi' and 'RSI' in indicator_naming:
                            p_rsi = col_lookup_params.get('period', 14)
                            rsi_col = f"RSI_{p_rsi}_{tf_suffix}"
                            if rsi_col in data.columns:
                                indicator_cols_for_score = {'rsi': rsi_col}
                                found = True
                                logger.debug(f"找到指标 'rsi' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'kdj' and 'KDJ' in indicator_naming:
                            p_k = col_lookup_params.get('period', 9)
                            p_d = col_lookup_params.get('signal_period', 3)
                            p_j = col_lookup_params.get('smooth_k_period', 3)
                            k_col = f"K_{p_k}_{p_d}_{p_j}_{tf_suffix}"
                            d_col = f"D_{p_k}_{p_d}_{p_j}_{tf_suffix}"
                            j_col = f"J_{p_k}_{p_d}_{p_j}_{tf_suffix}"
                            if all(col in data.columns for col in [k_col, d_col, j_col]):
                                indicator_cols_for_score = {'k': k_col, 'd': d_col, 'j': j_col}
                                found = True
                                logger.debug(f"找到指标 'kdj' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'boll' and 'BOLL' in indicator_naming:
                            p_boll = col_lookup_params.get('period', 20)
                            std_boll = col_lookup_params.get('std_dev', 2.0)
                            std_str = f"{std_boll:.1f}"
                            upper_col = f"BBU_{p_boll}_{std_str}_{tf_suffix}"
                            mid_col = f"BBM_{p_boll}_{std_str}_{tf_suffix}"
                            lower_col = f"BBL_{p_boll}_{std_str}_{tf_suffix}"
                            close_col = f"close_{tf_suffix}"  # BOLL评分需要收盘价
                            if all(col in data.columns for col in [close_col, upper_col, mid_col, lower_col]):
                                indicator_cols_for_score = {'close': close_col, 'upper': upper_col, 'mid': mid_col, 'lower': lower_col}
                                found = True
                                logger.debug(f"找到指标 'boll' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'cci' and 'CCI' in indicator_naming:
                            p_cci = col_lookup_params.get('period', 14)
                            cci_col = f"CCI_{p_cci}_{tf_suffix}"
                            if cci_col in data.columns:
                                indicator_cols_for_score = {'cci': cci_col}
                                found = True
                                logger.debug(f"找到指标 'cci' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'mfi' and 'MFI' in indicator_naming:
                            p_mfi = col_lookup_params.get('period', 14)
                            mfi_col = f"MFI_{p_mfi}_{tf_suffix}"
                            if mfi_col in data.columns:
                                indicator_cols_for_score = {'mfi': mfi_col}
                                found = True
                                logger.debug(f"找到指标 'mfi' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'roc' and 'ROC' in indicator_naming:
                            p_roc = col_lookup_params.get('period', 12)
                            roc_col = f"ROC_{p_roc}_{tf_suffix}"
                            if roc_col in data.columns:
                                indicator_cols_for_score = {'roc': roc_col}
                                found = True
                                logger.debug(f"找到指标 'roc' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'dmi' and 'DMI' in indicator_naming:
                            p_dmi = col_lookup_params.get('period', 14)
                            pdi_col = f"PDI_{p_dmi}_{tf_suffix}"
                            ndi_col = f"NDI_{p_dmi}_{tf_suffix}"
                            adx_col = f"ADX_{p_dmi}_{tf_suffix}"
                            if all(col in data.columns for col in [pdi_col, ndi_col, adx_col]):
                                indicator_cols_for_score = {'pdi': pdi_col, 'ndi': ndi_col, 'adx': adx_col}
                                found = True
                                logger.debug(f"找到指标 'dmi' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'sar' and 'SAR' in indicator_naming:
                            step = col_lookup_params.get('af_step', 0.02)
                            max_af = col_lookup_params.get('max_af', 0.2)
                            sar_col = f"SAR_{step}_{max_af}_{tf_suffix}"
                            close_col = f"close_{tf_suffix}"  # SAR评分需要收盘价
                            if all(col in data.columns for col in [close_col, sar_col]):
                                indicator_cols_for_score = {'close': close_col, 'sar': sar_col}
                                found = True
                                logger.debug(f"找到指标 'sar' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'stoch' and 'STOCH' in indicator_naming:
                            p_k = col_lookup_params.get('k_period', 14)
                            p_d = col_lookup_params.get('d_period', 3)
                            p_smooth_k = col_lookup_params.get('smooth_k_period', 3)
                            stochk_col = f"STOCHk_{p_k}_{p_d}_{p_smooth_k}_{tf_suffix}"
                            stochd_col = f"STOCHd_{p_k}_{p_d}_{p_smooth_k}_{tf_suffix}"
                            if all(col in data.columns for col in [stochk_col, stochd_col]):
                                indicator_cols_for_score = {'k': stochk_col, 'd': stochd_col}
                                found = True
                                logger.debug(f"找到指标 'stoch' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'ema' and 'EMA' in indicator_naming:
                            p_ema = col_lookup_params.get('period', 20)
                            ema_col = f"EMA_{p_ema}_{tf_suffix}"
                            close_col = f"close_{tf_suffix}"  # MA评分需要收盘价
                            if all(col in data.columns for col in [close_col, ema_col]):
                                indicator_cols_for_score = {'close': close_col, 'ma': ema_col}  # 通用MA评分函数需要close和ma
                                found = True
                                logger.debug(f"找到指标 'ema' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'sma' and 'SMA' in indicator_naming:
                            p_sma = col_lookup_params.get('period', 20)
                            sma_col = f"SMA_{p_sma}_{tf_suffix}"
                            close_col = f"close_{tf_suffix}"  # MA评分需要收盘价
                            if all(col in data.columns for col in [close_col, sma_col]):
                                indicator_cols_for_score = {'close': close_col, 'ma': sma_col}  # 通用MA评分函数需要close和ma
                                found = True
                                logger.debug(f"找到指标 'sma' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'atr' and 'ATR' in indicator_naming:
                            p_atr = col_lookup_params.get('period', 14)
                            atr_col = f"ATR_{p_atr}_{tf_suffix}"
                            if atr_col in data.columns:
                                indicator_cols_for_score = {'atr': atr_col}
                                found = True
                                logger.debug(f"找到指标 'atr' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'adl' and 'ADL' in indicator_naming:
                            adl_col = f"ADL_{tf_suffix}"  # ADL通常没有周期参数
                            if adl_col in data.columns:
                                indicator_cols_for_score = {'adl': adl_col}
                                found = True
                                logger.debug(f"找到指标 'adl' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'vwap' and 'VWAP' in indicator_naming:
                            anchor = col_lookup_params.get('anchor', None)
                            vwap_col = f"VWAP_{tf_suffix}" if anchor is None else f"VWAP_{anchor}_{tf_suffix}"
                            close_col = f"close_{tf_suffix}"  # VWAP评分需要收盘价
                            if all(col in data.columns for col in [close_col, vwap_col]):
                                indicator_cols_for_score = {'close': close_col, 'vwap': vwap_col}
                                found = True
                                logger.debug(f"找到指标 'vwap' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'ichimoku' and 'ICHIMOKU' in indicator_naming:
                            tenkan = col_lookup_params.get('tenkan_period', 9)
                            kijun = col_lookup_params.get('kijun_period', 26)
                            senkou_b_period = col_lookup_params.get('senkou_period', 52)
                            tenkan_col = f"TENKAN_{tenkan}_{tf_suffix}"
                            kijun_col = f"KIJUN_{kijun}_{tf_suffix}"
                            senkou_a_col = f"SENKOU_A_{tenkan}_{kijun}_{tf_suffix}"
                            senkou_b_col = f"SENKOU_B_{senkou_b_period}_{tf_suffix}"
                            chikou_col = f"CHIKOU_{kijun}_{tf_suffix}"  # Chikou 通常与 Kijun 同周期
                            close_col = f"close_{tf_suffix}"  # Ichimoku 评分可能需要收盘价
                            if all(col in data.columns for col in [close_col, tenkan_col, kijun_col, senkou_a_col, senkou_b_col, chikou_col]):
                                indicator_cols_for_score = {
                                    'close': close_col, 'tenkan': tenkan_col, 'kijun': kijun_col,
                                    'senkou_a': senkou_a_col, 'senkou_b': senkou_b_col, 'chikou': chikou_col
                                }
                                found = True
                                logger.debug(f"找到指标 'ichimoku' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'mom' and 'MOM' in indicator_naming:
                            p_mom = col_lookup_params.get('period', 10)
                            mom_col = f"MOM_{p_mom}_{tf_suffix}"
                            if mom_col in data.columns:
                                indicator_cols_for_score = {'mom': mom_col}
                                found = True
                                logger.debug(f"找到指标 'mom' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'willr' and 'WILLR' in indicator_naming:
                            p_willr = col_lookup_params.get('period', 14)
                            willr_col = f"WILLR_{p_willr}_{tf_suffix}"
                            if willr_col in data.columns:
                                indicator_cols_for_score = {'willr': willr_col}
                                found = True
                                logger.debug(f"找到指标 'willr' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'cmf' and 'CMF' in indicator_naming:
                            p_cmf = col_lookup_params.get('period', 20)
                            cmf_col = f"CMF_{p_cmf}_{tf_suffix}"
                            if cmf_col in data.columns:
                                indicator_cols_for_score = {'cmf': cmf_col}
                                found = True
                                logger.debug(f"找到指标 'cmf' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'obv' and 'OBV' in indicator_naming:
                            obv_col = f"OBV_{tf_suffix}"  # OBV通常没有周期参数
                            if obv_col in data.columns:
                                indicator_cols_for_score = {'obv': obv_col}
                                found = True
                                logger.debug(f"找到指标 'obv' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'kc' and 'KC' in indicator_naming:
                            ema_p = col_lookup_params.get('ema_period', 20)
                            atr_p = col_lookup_params.get('atr_period', 10)
                            upper_col = f"KCU_{ema_p}_{atr_p}_{tf_suffix}"
                            mid_col = f"KCM_{ema_p}_{atr_p}_{tf_suffix}"
                            lower_col = f"KCL_{ema_p}_{atr_p}_{tf_suffix}"
                            close_col = f"close_{tf_suffix}"  # KC评分需要收盘价
                            if all(col in data.columns for col in [close_col, upper_col, mid_col, lower_col]):
                                indicator_cols_for_score = {'close': close_col, 'upper': upper_col, 'mid': mid_col, 'lower': lower_col}
                                found = True
                                logger.debug(f"找到指标 'kc' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'hv' and 'HV' in indicator_naming:
                            p_hv = col_lookup_params.get('period', 20)
                            hv_col = f"HV_{p_hv}_{tf_suffix}"
                            if hv_col in data.columns:
                                indicator_cols_for_score = {'hv': hv_col}
                                found = True
                                logger.debug(f"找到指标 'hv' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'vroc' and 'VROC' in indicator_naming:
                            p_vroc = col_lookup_params.get('period', 10)
                            vroc_col = f"VROC_{p_vroc}_{tf_suffix}"
                            if vroc_col in data.columns:
                                indicator_cols_for_score = {'vroc': vroc_col}
                                found = True
                                logger.debug(f"找到指标 'vroc' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'aroc' and 'AROC' in indicator_naming:
                            p_aroc = col_lookup_params.get('period', 10)
                            aroc_col = f"AROC_{p_aroc}_{tf_suffix}"
                            if aroc_col in data.columns:
                                indicator_cols_for_score = {'aroc': aroc_col}
                                found = True
                                logger.debug(f"找到指标 'aroc' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break
                        elif indicator_key == 'pivot' and 'PIVOT_POINTS' in indicator_naming and tf_score == 'D':
                            pivot_cols = [f"PP_{tf_suffix}"] + [f"S{i}_{tf_suffix}" for i in range(1, 5)] + [f"R{i}_{tf_suffix}" for i in range(1, 5)] + \
                                         [f"F_S{i}_{tf_suffix}" for i in range(1, 4)] + [f"F_R{i}_{tf_suffix}" for i in range(1, 4)]
                            close_col = f"close_{tf_suffix}"  # Pivot评分需要收盘价
                            if all(col in data.columns for col in [close_col] + pivot_cols):
                                indicator_cols_for_score = {'close': close_col, 'pivot_levels': pivot_cols}  # 传递收盘价和所有Pivot水平列名
                                found = True
                                logger.debug(f"找到指标 'pivot' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}")
                                break

                # 修改：如果仍未找到匹配列，尝试模糊匹配列名（不依赖具体后缀）
                if not found:
                    logger.debug(f"未找到精确匹配的列名，尝试模糊匹配指标 '{indicator_key}' 在时间框架 {tf_score}")
                    for col in data.columns:
                        col_lower = col.lower()
                        if indicator_key == 'macd' and 'macd_' in col_lower and str(col_lookup_params.get('period_fast', 12)) in col and str(col_lookup_params.get('period_slow', 26)) in col:
                            if 'macds' in col_lower:
                                indicator_cols_for_score['macd_d'] = col
                            elif 'macdh' in col_lower:
                                indicator_cols_for_score['macd_h'] = col
                            else:
                                indicator_cols_for_score['macd_series'] = col
                        elif indicator_key == 'rsi' and 'rsi_' in col_lower and str(col_lookup_params.get('period', 14)) in col:
                            indicator_cols_for_score['rsi'] = col
                        elif indicator_key == 'kdj' and 'k_' in col_lower and str(col_lookup_params.get('period', 9)) in col:
                            if 'k_' in col_lower:
                                indicator_cols_for_score['k'] = col
                            elif 'd_' in col_lower:
                                indicator_cols_for_score['d'] = col
                            elif 'j_' in col_lower:
                                indicator_cols_for_score['j'] = col
                        elif indicator_key == 'boll' and 'bb' in col_lower and str(col_lookup_params.get('period', 20)) in col:
                            if 'bbu' in col_lower:
                                indicator_cols_for_score['upper'] = col
                            elif 'bbm' in col_lower:
                                indicator_cols_for_score['mid'] = col
                            elif 'bbl' in col_lower:
                                indicator_cols_for_score['lower'] = col
                        elif indicator_key == 'cci' and 'cci_' in col_lower and str(col_lookup_params.get('period', 14)) in col:
                            indicator_cols_for_score['cci'] = col
                        elif indicator_key == 'mfi' and 'mfi_' in col_lower and str(col_lookup_params.get('period', 14)) in col:
                            indicator_cols_for_score['mfi'] = col
                        elif indicator_key == 'roc' and 'roc_' in col_lower and str(col_lookup_params.get('period', 12)) in col:
                            indicator_cols_for_score['roc'] = col
                        elif indicator_key == 'dmi' and str(col_lookup_params.get('period', 14)) in col:
                            if 'pdi' in col_lower:
                                indicator_cols_for_score['pdi'] = col
                            elif 'ndi' in col_lower:
                                indicator_cols_for_score['ndi'] = col
                            elif 'adx' in col_lower:
                                indicator_cols_for_score['adx'] = col
                        elif indicator_key == 'sar' and 'sar_' in col_lower:
                            indicator_cols_for_score['sar'] = col
                        elif indicator_key == 'stoch' and 'stoch' in col_lower and str(col_lookup_params.get('k_period', 14)) in col:
                            if 'stochk' in col_lower:
                                indicator_cols_for_score['k'] = col
                            elif 'stochd' in col_lower:
                                indicator_cols_for_score['d'] = col
                        elif indicator_key == 'ema' and 'ema_' in col_lower and str(col_lookup_params.get('period', 20)) in col:
                            indicator_cols_for_score['ma'] = col
                        elif indicator_key == 'sma' and 'sma_' in col_lower and str(col_lookup_params.get('period', 20)) in col:
                            indicator_cols_for_score['ma'] = col
                        elif indicator_key == 'atr' and 'atr_' in col_lower and str(col_lookup_params.get('period', 14)) in col:
                            indicator_cols_for_score['atr'] = col
                        elif indicator_key == 'adl' and 'adl' in col_lower:
                            indicator_cols_for_score['adl'] = col
                        elif indicator_key == 'vwap' and 'vwap' in col_lower:
                            indicator_cols_for_score['vwap'] = col
                        elif indicator_key == 'ichimoku' and 'senkou' in col_lower or 'tenkan' in col_lower or 'kijun' in col_lower:
                            if 'tenkan' in col_lower:
                                indicator_cols_for_score['tenkan'] = col
                            elif 'kijun' in col_lower:
                                indicator_cols_for_score['kijun'] = col
                            elif 'senkou_a' in col_lower:
                                indicator_cols_for_score['senkou_a'] = col
                            elif 'senkou_b' in col_lower:
                                indicator_cols_for_score['senkou_b'] = col
                            elif 'chikou' in col_lower:
                                indicator_cols_for_score['chikou'] = col
                        elif indicator_key == 'mom' and 'mom_' in col_lower and str(col_lookup_params.get('period', 10)) in col:
                            indicator_cols_for_score['mom'] = col
                        elif indicator_key == 'willr' and 'willr_' in col_lower and str(col_lookup_params.get('period', 14)) in col:
                            indicator_cols_for_score['willr'] = col
                        elif indicator_key == 'cmf' and 'cmf_' in col_lower and str(col_lookup_params.get('period', 20)) in col:
                            indicator_cols_for_score['cmf'] = col
                        elif indicator_key == 'obv' and 'obv' in col_lower:
                            indicator_cols_for_score['obv'] = col
                        elif indicator_key == 'kc' and 'kc' in col_lower:
                            if 'kcu' in col_lower:
                                indicator_cols_for_score['upper'] = col
                            elif 'kcm' in col_lower:
                                indicator_cols_for_score['mid'] = col
                            elif 'kcl' in col_lower:
                                indicator_cols_for_score['lower'] = col
                        elif indicator_key == 'hv' and 'hv_' in col_lower and str(col_lookup_params.get('period', 20)) in col:
                            indicator_cols_for_score['hv'] = col
                        elif indicator_key == 'vroc' and 'vroc_' in col_lower and str(col_lookup_params.get('period', 10)) in col:
                            indicator_cols_for_score['vroc'] = col
                        elif indicator_key == 'aroc' and 'aroc_' in col_lower and str(col_lookup_params.get('period', 10)) in col:
                            indicator_cols_for_score['aroc'] = col
                        elif indicator_key == 'pivot' and 'pp_' in col_lower or 's1_' in col_lower or 'r1_' in col_lower:
                            if 'pivot_levels' not in indicator_cols_for_score:
                                indicator_cols_for_score['pivot_levels'] = []
                            indicator_cols_for_score['pivot_levels'].append(col)

                    # 检查是否收集到所有必要的列
                    if indicator_key == 'macd' and all(k in indicator_cols_for_score for k in ['macd_series', 'macd_d', 'macd_h']):
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'macd' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'rsi' and 'rsi' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'rsi' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'kdj' and all(k in indicator_cols_for_score for k in ['k', 'd', 'j']):
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'kdj' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'boll' and all(k in indicator_cols_for_score for k in ['upper', 'mid', 'lower']):
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 'boll' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'cci' and 'cci' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'cci' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'mfi' and 'mfi' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'mfi' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'roc' and 'roc' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'roc' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'dmi' and all(k in indicator_cols_for_score for k in ['pdi', 'ndi', 'adx']):
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'dmi' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'sar' and 'sar' in indicator_cols_for_score:
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 'sar' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'stoch' and all(k in indicator_cols_for_score for k in ['k', 'd']):
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'stoch' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key in ['ema', 'sma'] and 'ma' in indicator_cols_for_score:
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 '{indicator_key}' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'atr' and 'atr' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'atr' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'adl' and 'adl' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'adl' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'vwap' and 'vwap' in indicator_cols_for_score:
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 'vwap' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'ichimoku' and all(k in indicator_cols_for_score for k in ['tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou']):
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 'ichimoku' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'mom' and 'mom' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'mom' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'willr' and 'willr' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'willr' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'cmf' and 'cmf' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'cmf' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'obv' and 'obv' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'obv' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'kc' and all(k in indicator_cols_for_score for k in ['upper', 'mid', 'lower']):
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 'kc' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'hv' and 'hv' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'hv' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'vroc' and 'vroc' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'vroc' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'aroc' and 'aroc' in indicator_cols_for_score:
                        found = True
                        logger.debug(f"通过模糊匹配找到指标 'aroc' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")
                    elif indicator_key == 'pivot' and 'pivot_levels' in indicator_cols_for_score and len(indicator_cols_for_score['pivot_levels']) >= 15:
                        close_col = next((c for c in data.columns if c.startswith("close_")), None)
                        if close_col:
                            indicator_cols_for_score['close'] = close_col
                            found = True
                            logger.debug(f"通过模糊匹配找到指标 'pivot' 在时间框架 {tf_score} 的列: {indicator_cols_for_score}")

                # 如果未能找到该指标在该时间框架下的任何列，则跳过评分计算
                if not found:
                    logger.warning(f"未能为指标 '{indicator_key}' 在时间框架 {tf_score} 找到任何数据列进行评分。尝试的后缀: {possible_tf_suffixes}")
                    logger.debug(f"DataFrame 列名列表: {list(data.columns)}")
                    continue  # 跳过当前时间框架的评分

                # 提取 Series 并计算评分
                score_series = pd.Series(50.0, index=data.index)  # 初始化为中性分
                try:
                    # 根据指标 key 调用对应的评分函数，并传入相应的 Series 和参数
                    if indicator_key == 'macd':
                        score_series = score_func(
                            data[indicator_cols_for_score['macd_series']],  # MACD (diff)
                            data[indicator_cols_for_score['macd_d']],       # MACD signal (DEA)
                            data[indicator_cols_for_score['macd_h']]        # MACD hist
                        )
                    elif indicator_key == 'rsi':
                        score_series = score_func(data[indicator_cols_for_score['rsi']], params=score_func_params)
                    elif indicator_key == 'kdj':
                        score_series = score_func(
                            data[indicator_cols_for_score['k']],
                            data[indicator_cols_for_score['d']],
                            data[indicator_cols_for_score['j']],
                            params=score_func_params  # 传递评分函数所需参数 (超买超卖阈值)
                        )
                    elif indicator_key == 'boll':
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['upper']],
                            data[indicator_cols_for_score['mid']],
                            data[indicator_cols_for_score['lower']]
                        )
                    elif indicator_key == 'cci':
                        score_series = score_func(data[indicator_cols_for_score['cci']], params=score_func_params)
                    elif indicator_key == 'mfi':
                        score_series = score_func(data[indicator_cols_for_score['mfi']], params=score_func_params)
                    elif indicator_key == 'roc':
                        score_series = score_func(data[indicator_cols_for_score['roc']])  # ROC评分可能不需要额外阈值参数
                    elif indicator_key == 'dmi':
                        score_series = score_func(
                            data[indicator_cols_for_score['pdi']],
                            data[indicator_cols_for_score['ndi']],
                            data[indicator_cols_for_score['adx']],
                            params=score_func_params  # 传递评分函数所需参数 (ADX阈值)
                        )
                    elif indicator_key == 'sar':
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['sar']]
                        )
                    elif indicator_key == 'stoch':
                        score_series = score_func(
                            data[indicator_cols_for_score['k']],
                            data[indicator_cols_for_score['d']],
                            params=score_func_params  # 传递评分函数所需参数 (超买超卖阈值)
                        )
                    elif indicator_key in ['ema', 'sma']:  # 通用 MA 评分
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['ma']],
                            params=score_func_params  # 传递 MA 类型等参数
                        )
                    elif indicator_key == 'atr':
                        score_series = score_func(data[indicator_cols_for_score['atr']])  # ATR评分可能不需要额外参数
                    elif indicator_key == 'adl':
                        score_series = score_func(data[indicator_cols_for_score['adl']])  # ADL评分可能不需要额外参数
                    elif indicator_key == 'vwap':
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['vwap']]
                        )  # VWAP评分可能不需要额外参数
                    elif indicator_key == 'ichimoku':
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['tenkan']],
                            data[indicator_cols_for_score['kijun']],
                            data[indicator_cols_for_score['senkou_a']],
                            data[indicator_cols_for_score['senkou_b']],
                            data[indicator_cols_for_score['chikou']]
                        )  # Ichimoku评分可能不需要额外参数
                    elif indicator_key == 'mom':
                        score_series = score_func(data[indicator_cols_for_score['mom']])  # MOM评分可能不需要额外参数
                    elif indicator_key == 'willr':
                        score_series = score_func(data[indicator_cols_for_score['willr']])  # WILLR评分可能不需要额外参数
                    elif indicator_key == 'cmf':
                        score_series = score_func(data[indicator_cols_for_score['cmf']])  # CMF评分可能不需要额外参数
                    elif indicator_key == 'obv':
                        score_series = score_func(data[indicator_cols_for_score['obv']])  # OBV评分可能不需要额外参数
                    elif indicator_key == 'kc':
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['upper']],
                            data[indicator_cols_for_score['mid']],
                            data[indicator_cols_for_score['lower']]
                        )  # KC评分可能不需要额外参数
                    elif indicator_key == 'hv':
                        score_series = score_func(data[indicator_cols_for_score['hv']])  # HV评分可能不需要额外参数
                    elif indicator_key == 'vroc':
                        score_series = score_func(data[indicator_cols_for_score['vroc']])  # VROC评分可能不需要额外参数
                    elif indicator_key == 'aroc':
                        score_series = score_func(data[indicator_cols_for_score['aroc']])  # AROC评分可能不需要额外参数
                    elif indicator_key == 'pivot' and tf_score == 'D':
                        score_series = score_func(
                            data[indicator_cols_for_score['close']],
                            data[indicator_cols_for_score['pivot_levels']]  # 传递Pivot水平列名列表
                        )  # Pivot评分可能需要额外参数
                    else:
                        # 理论上不会到这里，因为前面已经检查并跳过
                        logger.error(f"内部错误：未知的指标 key '{indicator_key}' 在评分函数调用逻辑中。")
                        continue  # 跳过当前时间框架的评分

                    # 将评分结果添加到 DataFrame
                    # 修改：评分列名也参考 JSON 文件中的 SCORE 命名规范
                    if 'SCORE' in indicator_naming and any(indicator_key.upper() in col['name_pattern'] for col in indicator_naming['SCORE'].get('output_columns', [])):
                        score_col_name = next(
                            (col['name_pattern'].format(**{k: v for k, v in col_lookup_params.items() if k in col['name_pattern']}) + f"_{tf_score}"
                             for col in indicator_naming['SCORE'].get('output_columns', [])
                             if indicator_key.upper() in col['name_pattern']),
                            f"SCORE_{indicator_key.upper()}_{tf_score}"
                        )
                    else:
                        score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score}"
                    # 确保评分结果 Series 的索引与原始数据对齐，并填充 NaN 为 50.0
                    scoring_results[score_col_name] = score_series.reindex(data.index).fillna(50.0)
                    logger.debug(f"TF {tf_score}: 指标 {indicator_key} 评分计算完成，列名 '{score_col_name}'.")

                except Exception as e:
                    logger.error(f"计算指标 '{indicator_key}' 在时间框架 {tf_score} 的评分时出错: {e}", exc_info=False)
                    # 计算出错时，该指标在该时间框架的评分列将保持默认的 50.0 (如果前面没有成功赋值的话)
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score}"
                    if score_col_name not in scoring_results.columns:
                        scoring_results[score_col_name] = 50.0
                    else:
                        scoring_results[score_col_name].fillna(50.0, inplace=True)

            except Exception as e_col_lookup:
                logger.error(f"查找指标 '{indicator_key}' 在时间框架 {tf_score} 的列时出错: {e_col_lookup}", exc_info=False)
                # 列查找出错时，该指标在该时间框架的评分列将保持默认的 50.0
                score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score}"
                if score_col_name not in scoring_results.columns:
                    scoring_results[score_col_name] = 50.0
                else:
                    scoring_results[score_col_name].fillna(50.0, inplace=True)

    # 最终填充所有评分列中的 NaN 为 50 (中性分数)
    # 这一步是双重保险，前面的填充应该已经处理了大部分
    scoring_results = scoring_results.fillna(50.0)
    logger.info("所有配置的指标评分计算完成。")
    return scoring_results

def adjust_score_with_volume( preliminary_score: pd.Series, data: pd.DataFrame, vc_params: Dict ) -> pd.DataFrame:
    """
    使用量能相关指标（成交量、OBV、CMF等）对初步的策略评分 (0-100) 进行调整和确认。
    同时，此函数会计算并输出相关的量能分析信号。

    主要优化和深化点：
    1.  更健壮的量价背离检测逻辑（启发式，寻找价格与OBV的趋势不一致）。
    2.  改进了分数调整机制，使其按比例向目标值（0, 50, 100）移动。
    3.  清晰的参数配置和默认值。
    4.  加强了数据对齐、缺失值处理的鲁棒性。
    5.  详细的中文注释。
    Args:
        preliminary_score (pd.Series): 经过基础指标计算得到的初步分数 (0-100)，索引为时间。
        data (pd.DataFrame): 包含OHLCV价格数据以及计算好的CMF、OBV等指标的DataFrame。
                             列名需要与配置中构建的名称一致。
        vc_params (Dict): 量能确认与调整相关的参数字典，包含：
            'enabled': bool, 是否启用基于量能的分数调整 (True/False)。
            'volume_analysis_enabled': bool, 是否启用量能分析信号的计算（即使不调整分数）。
            'tf': str, 进行量能分析所使用的数据时间级别 (例如 'D', '60', '15')。
            'boost_factor': float, 量能确认趋势时，分数向极值（0或100）移动的比例 (0.0-1.0)。
            'penalty_factor': float, 量能与趋势矛盾时，分数向中性值（50）移动的比例 (0.0-1.0)。
            'volume_spike_threshold': float, 成交量突增的判断阈值（当前量 / 均量的倍数）。
            'volume_analysis_lookback': int, 计算成交量均线、寻找量价背离等的回看窗口期。
            'cmf_period': int, CMF指标的计算周期 (应与指标服务中的计算一致)。
            'cmf_confirmation_threshold': float, CMF确认趋势的阈值 (例如 0.05)。
            'obv_ma_period': int, OBV移动平均线的计算周期 (应与指标服务中的计算一致)。
            'vp_divergence_lookback': int, 量价背离检测的回看窗口期。
            'vp_divergence_peak_offset': int, 在量价背离中，比较当前极值点与之前第N个极值点。
            'vp_divergence_price_threshold': float, 价格创出新高/新低的最小幅度（相对于前一极值）。
            'vp_divergence_obv_threshold': float, OBV未能同步创出新高/新低的最小幅度。
            'vp_divergence_penalty_factor': float, 量价背离惩罚分数调整的比例。
            'volume_spike_adj_factor': float, 成交量突增时调整分数的比例。
    Returns:
        pd.DataFrame: 返回一个DataFrame，索引与 preliminary_score 一致，包含以下列:
            - 'ADJUSTED_SCORE': 经过量能调整后的最终分数 (0-100)。
            - 'VOL_CONFIRM_SIGNAL_{TF}': 量能确认信号 (1: 量能支持当前趋势, -1: 量能与当前趋势矛盾, 0: 中性)。
            - 'VOL_SPIKE_SIGNAL_{TF}': 成交量突增信号 (1: 检测到突增, 0: 正常)。
            - 'VOL_PRICE_DIV_SIGNAL_{TF}': 量价背离信号 (1: 看涨背离 - 价跌量升势头, -1: 看跌背离 - 价涨量缩势头, 0: 无明显背离)。
    """
    # --- 0. 初始化和参数准备 ---
    result_df = pd.DataFrame(index=preliminary_score.index)
    result_df['ADJUSTED_SCORE'] = preliminary_score.copy() # 默认调整后分数等于原始分数

    # 获取分析的时间级别，默认为日线 'D'
    vol_tf = vc_params.get('tf', 'D')

    # 预先创建所有信号列，并用0填充，确保它们总是存在
    confirm_signal_col = f'VOL_CONFIRM_SIGNAL_{vol_tf}'
    spike_signal_col = f'VOL_SPIKE_SIGNAL_{vol_tf}'
    div_signal_col = f'VOL_PRICE_DIV_SIGNAL_{vol_tf}'
    result_df[confirm_signal_col] = 0
    result_df[spike_signal_col] = 0
    result_df[div_signal_col] = 0

    # 检查是否启用了量能分析或分数调整
    score_adjustment_enabled = vc_params.get('enabled', False)
    volume_analysis_enabled = vc_params.get('volume_analysis_enabled', False)
    if not score_adjustment_enabled and not volume_analysis_enabled:
        logger.debug("量能分数调整和量能分析均未启用。")
        # 确保 ADJUSTED_SCORE 列的 NaN 被填充
        result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].fillna(50.0)
        return result_df

    # 提取各项配置参数
    boost_factor = vc_params.get('boost_factor', 0.20)  # 分数向极值移动20%的剩余距离
    penalty_factor = vc_params.get('penalty_factor', 0.30) # 分数向中性值50移动30%的剩余距离
    vol_spike_threshold = vc_params.get('volume_spike_threshold', 2.5)
    vol_analysis_lookback = vc_params.get('volume_analysis_lookback', 20)
    cmf_period = vc_params.get('cmf_period', 20)
    cmf_confirm_thresh = vc_params.get('cmf_confirmation_threshold', 0.05)
    obv_ma_period = vc_params.get('obv_ma_period', 10)

    # 量价背离相关参数
    vp_div_lookback = vc_params.get('vp_divergence_lookback', 21) # 例如过去一个月（21个交易日）
    # vp_div_peak_offset = vc_params.get('vp_divergence_peak_offset', 5) # 比较当前极值与向前数第5个交易日附近的极值
    # vp_price_thresh = vc_params.get('vp_divergence_price_threshold', 0.005) # 价格新高/低至少超过前高/低0.5%
    # vp_obv_thresh = vc_params.get('vp_divergence_obv_threshold', 0.005)   # OBV未能同步的幅度阈值

    # --- 1. 数据列名构建和有效性检查 ---
    close_col = f'close_{vol_tf}'
    high_col = f'high_{vol_tf}'
    low_col = f'low_{vol_tf}'
    volume_col = f'volume_{vol_tf}'
    cmf_col_name = f'CMF_{cmf_period}_{vol_tf}'
    obv_col_name = f'OBV_{vol_tf}'
    obv_ma_col_name = f'OBV_MA_{obv_ma_period}_{vol_tf}'

    required_cols = [close_col, high_col, low_col, volume_col, cmf_col_name, obv_col_name, obv_ma_col_name]
    missing_cols = [col for col in required_cols if col not in data.columns or data[col].isnull().all()]

    if missing_cols:
        logger.warning(f"量能调整/分析模块：时间框架 '{vol_tf}' 缺少必需的数据列: {missing_cols}。跳过此模块。")
        result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].fillna(50.0)
        return result_df # 返回包含默认0信号和原始（或填充后）分数的DataFrame

    # --- 2. 数据提取、对齐和填充 ---
    # 合并 preliminary_score 和所需数据列，确保索引对齐
    # 使用 outer join 避免丢失任一方的索引，然后 reindex 回 preliminary_score 的索引
    data_subset = data[required_cols].copy() # 操作副本
    merged_df = pd.concat([preliminary_score.rename("PRELIM_SCORE_TEMP_COL"), data_subset], axis=1, join='outer')
    merged_df = merged_df.reindex(preliminary_score.index) # 确保最终索引与输入分数一致

    # 对提取的序列进行填充
    close = merged_df[close_col].ffill().bfill() # 价格用前后值填充
    high = merged_df[high_col].ffill().bfill()
    low = merged_df[low_col].ffill().bfill()
    volume = merged_df[volume_col].fillna(0)          # 成交量NaN填充为0
    cmf = merged_df[cmf_col_name].fillna(0)           # CMF NaN填充为0 (代表中性资金流)
    obv = merged_df[obv_col_name].ffill().bfill()     # OBV是累积值，用前后值填充可能更合理
    obv_ma = merged_df[obv_ma_col_name].ffill().bfill() # OBV均线也用前后值填充

    # 再次检查关键数据是否在填充后仍然无效
    if close.isnull().all() or obv.isnull().all() or volume.isnull().all():
        logger.warning(f"量能调整/分析模块：时间框架 '{vol_tf}' 的关键数据 (收盘价/OBV/成交量) 在填充后仍无效。跳过。")
        result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].fillna(50.0)
        return result_df

    # --- 3. 计算量能分析信号 ---

    # 3.1. 量能确认信号 (VOL_CONFIRM_SIGNAL)
    # 定义：CMF 和 OBV 相对于其均线的方向是否支持当前价格趋势的初步判断
    # 看涨量能确认: CMF 为正且高于阈值，并且 OBV 在其均线上方
    bullish_volume_confirmation = (cmf > cmf_confirm_thresh) & (obv > obv_ma)
    # 看跌量能确认: CMF 为负且低于负阈值，并且 OBV 在其均线下方
    bearish_volume_confirmation = (cmf < -cmf_confirm_thresh) & (obv < obv_ma)
    result_df.loc[bullish_volume_confirmation, confirm_signal_col] = 1
    result_df.loc[bearish_volume_confirmation, confirm_signal_col] = -1

    # 3.2. 成交量突增信号 (VOL_SPIKE_SIGNAL)
    # 定义：当前成交量显著高于其近期移动平均值
    if vol_analysis_lookback > 0 and not volume.empty:
        # 计算成交量均线，替换0值为NaN以避免除零错误
        volume_mean = volume.rolling(window=vol_analysis_lookback, min_periods=max(1, vol_analysis_lookback // 2)).mean().replace(0, np.nan)
        # 仅在volume_mean有效时进行比较
        valid_mean_mask = volume_mean.notna()
        is_spike = pd.Series(False, index=volume.index) # 初始化为全False
        is_spike.loc[valid_mean_mask] = (volume.loc[valid_mean_mask] / volume_mean.loc[valid_mean_mask]) > vol_spike_threshold
        result_df[spike_signal_col] = is_spike.astype(int)

    # 3.3. 量价背离信号 (VOL_PRICE_DIV_SIGNAL) - 深化版启发式检测
    # 我们将寻找价格高低点与OBV高低点之间的不一致性
    # 注意：这仍然是启发式方法，更精确的背离需要类似`find_divergence_for_indicator`的峰谷匹配。
    if vp_div_lookback > 1 and not obv.isnull().all(): # 需要至少2个点进行比较
        # 寻找价格的近期高点和低点索引
        price_high_idx = high.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmax(), raw=True)
        price_low_idx = low.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmin(), raw=True)

        # 寻找OBV的近期高点和低点索引
        obv_high_idx = obv.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmax(), raw=True)
        obv_low_idx = obv.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmin(), raw=True)

        # 为了比较，我们需要前一个极值点。这里简化为比较当前点与回看期内的整体极值趋势。
        # 更精确的方法是找到序列中的P1, P2, I1, I2点。
        # 此处简化：如果价格创近期新高，但OBV未创近期新高（或OBV趋势向下），则为看跌背离。
        #           如果价格创近期新低，但OBV未创近期新低（或OBV趋势向上），则为看涨背离。

        # 条件1: 价格创N期新高
        is_price_new_high = (high == high.rolling(window=vp_div_lookback).max())
        # 条件2: 价格创N期新低
        is_price_new_low = (low == low.rolling(window=vp_div_lookback).min())

        # 条件3: OBV未创N期新高 (或者说，OBV在价格新高时，其值低于OBV的N期内高点)
        obv_not_new_high = (obv < obv.rolling(window=vp_div_lookback).max())
        # 条件4: OBV未创N期新低 (或者说，OBV在价格新低时，其值高于OBV的N期内低点)
        obv_not_new_low = (obv > obv.rolling(window=vp_div_lookback).min())
        
        # 另一种OBV趋势判断：比较OBV与其自身的短期均线
        obv_short_ma_period = max(3, vp_div_lookback // 4) # 短期OBV均线
        obv_vs_short_ma = obv - obv.rolling(window=obv_short_ma_period, min_periods=1).mean()


        # 看跌背离: 价格新高 & (OBV未新高 或 OBV < OBV短期均线)
        # 增加一个条件：价格确实比前几天有所上涨，避免横盘高位也被判为新高
        price_rising_flag = close.diff(1).fillna(0) > 0
        bearish_divergence = is_price_new_high & price_rising_flag & (obv_not_new_high | (obv_vs_short_ma < 0))
        result_df.loc[bearish_divergence, div_signal_col] = -1

        # 看涨背离: 价格新低 & (OBV未新低 或 OBV > OBV短期均线)
        price_falling_flag = close.diff(1).fillna(0) < 0
        bullish_divergence = is_price_new_low & price_falling_flag & (obv_not_new_low | (obv_vs_short_ma > 0))
        # 确保不覆盖已经标记的看跌背离
        result_df.loc[bullish_divergence & (result_df[div_signal_col] == 0), div_signal_col] = 1
    else:
        logger.debug(f"VP背离检测跳过，因 lookback ({vp_div_lookback}) 不足或OBV数据无效。")


    # --- 4. 应用量能调整到初步分数 (仅当 score_adjustment_enabled 为 True) ---
    if score_adjustment_enabled:
        current_score = result_df['ADJUSTED_SCORE'].copy() # 获取当前待调整的分数副本
        
        # 判断初步分数的趋势方向
        is_bullish_prelim_score = current_score > 55  # 初步看涨
        is_bearish_prelim_score = current_score < 45  # 初步看跌
        is_neutral_prelim_score = (~is_bullish_prelim_score) & (~is_bearish_prelim_score) # 初步中性

        # 4.1. 基于量能确认信号调整
        # a) 初步看涨
        #    - 量能支持 (confirm_signal == 1): 分数向100移动
        bull_confirmed_cond = is_bullish_prelim_score & (result_df[confirm_signal_col] == 1)
        current_score.loc[bull_confirmed_cond] = current_score + (100 - current_score) * boost_factor
        #    - 量能矛盾 (confirm_signal == -1): 分数向50移动
        bull_contradict_cond = is_bullish_prelim_score & (result_df[confirm_signal_col] == -1)
        current_score.loc[bull_contradict_cond] = current_score - (current_score - 50) * penalty_factor
        
        # b) 初步看跌
        #    - 量能支持 (confirm_signal == -1): 分数向0移动
        bear_confirmed_cond = is_bearish_prelim_score & (result_df[confirm_signal_col] == -1)
        current_score.loc[bear_confirmed_cond] = current_score - current_score * boost_factor
        #    - 量能矛盾 (confirm_signal == 1): 分数向50移动
        bear_contradict_cond = is_bearish_prelim_score & (result_df[confirm_signal_col] == 1)
        current_score.loc[bear_contradict_cond] = current_score + (50 - current_score) * penalty_factor

        # c) 初步中性 (分数在45-55之间)
        #    - 量能看涨 (confirm_signal == 1): 分数向看涨区域轻微移动 (例如60)
        neutral_to_bull_cond = is_neutral_prelim_score & (result_df[confirm_signal_col] == 1)
        current_score.loc[neutral_to_bull_cond] = current_score + (60 - current_score) * boost_factor * 0.5 # 调整幅度减半
        #    - 量能看跌 (confirm_signal == -1): 分数向看跌区域轻微移动 (例如40)
        neutral_to_bear_cond = is_neutral_prelim_score & (result_df[confirm_signal_col] == -1)
        current_score.loc[neutral_to_bear_cond] = current_score - (current_score - 40) * boost_factor * 0.5 # 调整幅度减半

        result_df['ADJUSTED_SCORE'] = current_score # 更新调整后的分数

        # 4.2. 基于量价背离信号调整 (通常是惩罚性或警示性)
        vp_div_penalty_factor = vc_params.get('vp_divergence_penalty_factor', 0.25) # 背离惩罚使分数向50移动25%
        # a) 初步看涨，但出现看跌量价背离 (div_signal == -1) -> 分数向50回调
        bull_score_with_bear_div = is_bullish_prelim_score & (result_df[div_signal_col] == -1)
        result_df.loc[bull_score_with_bear_div, 'ADJUSTED_SCORE'] = \
            result_df.loc[bull_score_with_bear_div, 'ADJUSTED_SCORE'] - \
            (result_df.loc[bull_score_with_bear_div, 'ADJUSTED_SCORE'] - 50) * vp_div_penalty_factor
        
        # b) 初步看跌，但出现看涨量价背离 (div_signal == 1) -> 分数向50回调
        bear_score_with_bull_div = is_bearish_prelim_score & (result_df[div_signal_col] == 1)
        result_df.loc[bear_score_with_bull_div, 'ADJUSTED_SCORE'] = \
            result_df.loc[bear_score_with_bull_div, 'ADJUSTED_SCORE'] + \
            (50 - result_df.loc[bear_score_with_bull_div, 'ADJUSTED_SCORE']) * vp_div_penalty_factor

        # 4.3. 基于成交量突增信号调整 (可选，可能加强趋势或预示反转)
        # 这里简化为：如果趋势与突增方向一致，则轻微加强分数；如果矛盾，则轻微拉回。
        vol_spike_adj_factor = vc_params.get('volume_spike_adj_factor', 0.10) # 突增调整幅度10%
        # a) 初步看涨 + 成交量突增: 轻微加强看涨分数
        bull_score_with_spike = is_bullish_prelim_score & (result_df[spike_signal_col] == 1)
        result_df.loc[bull_score_with_spike, 'ADJUSTED_SCORE'] = \
            result_df.loc[bull_score_with_spike, 'ADJUSTED_SCORE'] + \
            (100 - result_df.loc[bull_score_with_spike, 'ADJUSTED_SCORE']) * vol_spike_adj_factor
        
        # b) 初步看跌 + 成交量突增: 轻微加强看跌分数
        bear_score_with_spike = is_bearish_prelim_score & (result_df[spike_signal_col] == 1)
        result_df.loc[bear_score_with_spike, 'ADJUSTED_SCORE'] = \
            result_df.loc[bear_score_with_spike, 'ADJUSTED_SCORE'] - \
            result_df.loc[bear_score_with_spike, 'ADJUSTED_SCORE'] * vol_spike_adj_factor
            
        # 确保最终分数在0-100范围内
        result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].clip(0, 100)

    # --- 5. 最终填充和返回 ---
    # 确保 ADJUSTED_SCORE 列的 NaN 被填充为中性值 50
    result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].fillna(50.0)
    # 其他信号列的 NaN (理论上在初始化时已处理) 再次确保填充为0
    for col in [confirm_signal_col, spike_signal_col, div_signal_col]:
        if col in result_df.columns: # 确保列存在
            result_df[col] = result_df[col].fillna(0).astype(int) # 转换为整数类型

    logger.info(f"量能调整和分析模块（时间框架 {vol_tf}）处理完成。")
    return result_df

