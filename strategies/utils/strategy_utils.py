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
    print(f"时间级别 {time_level} ({tf_minutes}分钟)，基础回看期 {base_lookback}，生成 find_peaks 参数: {params}")
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
        print(f"数据不足或全为 NaN (价格: {len(price_series)}, 指标: {len(indicator_series)}, 最小需要: {min_data_points})，无法检测背离。")
        return result_df

    if not isinstance(price_series, pd.Series) or not isinstance(indicator_series, pd.Series) or not price_series.index.equals(indicator_series.index):
         logger.error("价格序列或指标序列不是 pandas Series，或索引不一致。")
         return result_df

    indicator_filled = indicator_series.ffill().bfill()
    if indicator_filled.isnull().all():
        print("填充后的指标序列全为 NaN，无法检测背离。")
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

            print(f"开始检测 TF {tf_check}: 价格 ('{price_col}') 与指标 ('{indicator_col}') 的背离 (lookback: {safe_lookback})...")
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
# MODIFIED: 复制 File 2 中的评分函数实现片段到此处
def _safe_fillna_series(series_list: List[pd.Series], fill_values: List[Any]) -> List[pd.Series]:
    """
    安全地填充 Series 中的 NaN 值，优先使用 ffill/bfill，然后使用指定值。
    确保所有 Series 具有相同的索引。
    """
    if not series_list:
        return []

    # 确保所有 Series 索引一致，以第一个 Series 的索引为准
    base_index = series_list[0].index
    aligned_series = [s.reindex(base_index) for s in series_list]

    filled_series = []
    for i, s in enumerate(aligned_series):
        if s.isnull().all():
             # 如果整个 Series 都是 NaN，尝试使用指定的填充值
             fill_val = fill_values[i] if i < len(fill_values) else 50.0 # 默认填充50
             if fill_val is None:
                  # 如果指定填充值为 None，则不填充，保留 NaN
                  filled_series.append(s)
             elif isinstance(fill_val, (int, float)):
                  # 如果是数值，填充为该数值
                  filled_series.append(s.fillna(fill_val))
             elif hasattr(fill_val, '__call__'):
                  # 如果是函数，调用函数获取填充值 (例如 s.mean())
                  try:
                      val = fill_val(s)
                      filled_series.append(s.fillna(val))
                  except Exception:
                      # 调用函数失败，填充为50
                      filled_series.append(s.fillna(50.0))
                      logger.warning(f"调用填充函数失败，使用默认值 50.0 填充 Series {i}.")
             else:
                  # 其他类型填充
                  filled_series.append(s.fillna(fill_val))
        else:
            # 否则，先进行 ffill/bfill
            s_filled = s.ffill().bfill()
            # 如果 ffill/bfill 后仍然有 NaN (例如开头或结尾)，尝试使用指定填充值
            fill_val = fill_values[i] if i < len(fill_values) else 50.0
            if fill_val is not None:
                if isinstance(fill_val, (int, float)):
                     s_filled = s_filled.fillna(fill_val)
                elif hasattr(fill_val, '__call__'):
                     try:
                         val = fill_val(s_filled) # 注意这里用 s_filled 而不是 s
                         s_filled = s_filled.fillna(val)
                     except Exception:
                         s_filled = s_filled.fillna(50.0)
                         logger.warning(f"调用填充函数失败，使用默认值 50.0 填充 Series {i}.")
                else:
                     s_filled = s_filled.fillna(fill_val)
            filled_series.append(s_filled)

    return filled_series

def calculate_macd_score(macd_series: pd.Series, macd_d: pd.Series, macd_h: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """MACD 评分 (0-100)。"""
    # 确保索引一致并填充NaN
    # MACD线和DEA线中性值可以是0或前值，MACDh中性值是0
    # MODIFIED: 使用 _safe_fillna_series 填充
    macd_s, macd_d_s, macd_h_s = _safe_fillna_series(
        [macd_series, macd_d, macd_h],
        [0.0, 0.0, 0.0] # 假设0是合理的填充值
    )
    # 如果填充后仍有NaN（例如整个序列都是NaN），则返回全50分
    # MODIFIED: 检查填充后的 Series 是否全为 NaN
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

def calculate_rsi_score(rsi: pd.Series, params: Dict) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """RSI 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    rsi_s, = _safe_fillna_series([rsi], [50.0]) # RSI 中性50
    if rsi_s.isnull().all():
        return pd.Series(50.0, index=rsi.index).clip(0,100)

    score = pd.Series(50.0, index=rsi_s.index)
    # MODIFIED: 从 params 字典获取参数
    os = params.get('oversold', 30) # 注意这里使用了评分函数内部的参数名 'oversold'，不是 bs_params 中的 'rsi_oversold'
    ob = params.get('overbought', 70)
    ext_os = params.get('extreme_oversold', 20)
    ext_ob = params.get('extreme_overbought', 80)
    # period 参数在评分逻辑中可能不会直接用到数值，主要用于标识是哪个周期的 RSI

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

def calculate_kdj_score(k: pd.Series, d: pd.Series, j: pd.Series, params: Dict) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """KDJ 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    k_s, d_s, j_s = _safe_fillna_series([k, d, j], [50.0, 50.0, 50.0]) # KDJ 中性50
    if k_s.isnull().all(): # Check one, assume others similar after _safe_fillna_series
        return pd.Series(50.0, index=k.index).clip(0,100)

    score = pd.Series(50.0, index=k_s.index)
    # MODIFIED: 从 params 字典获取参数
    os = params.get('oversold', 20)
    ob = params.get('overbought', 80)
    ext_os = params.get('extreme_oversold', 10)
    ext_ob = params.get('extreme_overbought', 90)
    # period, signal_period, smooth_k_period 参数在评分逻辑中可能不会直接用到数值，主要用于标识是哪个周期的 KDJ

    score.loc[j_s < ext_os] = 95.0
    score.loc[j_s > ext_ob] = 5.0

    # Apply to k or d, ensuring not to overwrite extreme j scores
    score.loc[((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))] = np.maximum(score.loc[((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))], 85.0)
    score.loc[((k_s <= ext_ob) & (k_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))] = np.minimum(score.loc[((k_s <= ext_ob) & (d_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))], 15.0)


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

def calculate_boll_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """BOLL 评分 (0-100)。"""
    # For BOLL, if bands are NaN, filling with close or mid might be tricky.
    # MODIFIED: 使用 _safe_fillna_series 填充，并提供后备填充逻辑
    close_s, upper_s, mid_s, lower_s = _safe_fillna_series(
        [close, upper, mid, lower],
        [
            None, # close 优先 ffill/bfill
            lambda s: s.mean() + 2 * s.std() if s.std() > 0 else s.mean() + 0.01 * s.mean(), # upper 填充后，如果全 NaN 估算
            lambda s: s.mean(), # mid 填充后，如果全 NaN 估算
            lambda s: s.mean() - 2 * s.std() if s.std() > 0 else s.mean() - 0.01 * s.mean()  # lower 填充后，如果全 NaN 估算
        ]
    )

    # Re-check for all NaNs after filling attempts
    if close_s.isnull().all() or upper_s.isnull().all() or mid_s.isnull().all() or lower_s.isnull().all():
         logger.warning("BOLL 评分：一个或多个关键序列在填充后仍全为NaN。")
         # MODIFIED: 确保返回 Series 的索引是原始 close 的索引
         return pd.Series(50.0, index=close.index).clip(0,100)


    score = pd.Series(50.0, index=close_s.index)

    score.loc[close_s <= lower_s] = 90.0
    buy_support = (close_s.shift(1) < lower_s.shift(1)) & (close_s >= lower_s)
    score.loc[buy_support] = np.maximum(score.loc[buy_support], 80.0)

    score.loc[close_s >= upper_s] = 10.0
    sell_pressure = (close_s.shift(1) > upper_s.shift(1)) & (close_s <= upper_s)
    score.loc[sell_pressure] = np.minimum(score.loc[sell_pressure], 20.0)

    buy_mid_cross = (close_s.shift(1) < mid_s.shift(1)) & (close_s >= mid_s)
    sell_mid_cross = (close_s.shift(1) > mid_s.shift(1)) & (close_s <= mid_s)
    score.loc[buy_mid_cross] = np.maximum(score.loc[buy_mid_cross], 65.0)
    score.loc[sell_mid_cross] = np.minimum(score.loc[sell_mid_cross], 35.0)

    not_extreme_cond = (close_s > lower_s) & (close_s < upper_s)
    not_mid_cross_cond = ~buy_mid_cross & ~sell_mid_cross

    is_above_mid = not_extreme_cond & not_mid_cross_cond & (close_s > mid_s)
    is_below_mid = not_extreme_cond & not_mid_cross_cond & (close_s < mid_s)
    score.loc[is_above_mid] = np.maximum(score.loc[is_above_mid], 55.0)
    score.loc[is_below_mid] = np.minimum(score.loc[is_below_mid], 45.0)

    return score.clip(0, 100)

def calculate_cci_score(cci: pd.Series, params: Dict) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """CCI 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    cci_s, = _safe_fillna_series([cci], [0.0]) # CCI 中性0
    if cci_s.isnull().all():
        return pd.Series(50.0, index=cci.index).clip(0,100)

    score = pd.Series(50.0, index=cci_s.index)
    # MODIFIED: 从 params 字典获取参数
    threshold = params.get('threshold', 100) # 注意这里使用了评分函数内部的参数名
    ext_threshold = params.get('extreme_threshold', 200)
    # period 参数在评分逻辑中可能不会直接用到数值

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

def calculate_mfi_score(mfi: pd.Series, params: Dict) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """MFI 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    mfi_s, = _safe_fillna_series([mfi], [50.0]) # MFI 中性50
    if mfi_s.isnull().all():
        return pd.Series(50.0, index=mfi.index).clip(0,100)

    score = pd.Series(50.0, index=mfi_s.index)
    # MODIFIED: 从 params 字典获取参数
    os = params.get('oversold', 20) # 注意这里使用了评分函数内部的参数名
    ob = params.get('overbought', 80)
    ext_os = params.get('extreme_oversold', 10)
    ext_ob = params.get('extreme_overbought', 90)
    # period 参数在评分逻辑中可能不会直接用到数值

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

def calculate_roc_score(roc: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """ROC 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_dmi_score(pdi: pd.Series, ndi: pd.Series, adx: pd.Series, params: Dict) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """DMI 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    pdi_s, ndi_s, adx_s = _safe_fillna_series([pdi, ndi, adx], [0.0, 0.0, 0.0]) # DMI/ADX 中性0
    if pdi_s.isnull().all():
        return pd.Series(50.0, index=pdi.index).clip(0,100)

    score = pd.Series(50.0, index=pdi_s.index)
    # MODIFIED: 从 params 字典获取参数
    adx_th = params.get('adx_threshold', 25) # 注意这里使用了评分函数内部的参数名
    adx_strong_th = params.get('adx_strong_threshold', 40)
    # period 参数在评分逻辑中可能不会直接用到数值

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

def calculate_sar_score(close: pd.Series, sar: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """SAR 评分 (0-100)。"""
    # SAR can be tricky to fill if NaN. Filling with close means neutral.
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_stoch_score(k: pd.Series, d: pd.Series, params: Dict) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """随机指标 (STOCH) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    k_s, d_s = _safe_fillna_series([k, d], [50.0, 50.0]) # STOCH 中性50
    if k_s.isnull().all():
        return pd.Series(50.0, index=k.index).clip(0,100)

    score = pd.Series(50.0, index=k_s.index)
    # MODIFIED: 从 params 字典获取参数
    os = params.get('stoch_oversold', 20) # 注意这里使用了评分函数内部的参数名
    ob = params.get('stoch_overbought', 80)
    ext_os = params.get('stoch_extreme_oversold', 10)
    ext_ob = params.get('stoch_extreme_overbought', 90)
    # k_period, d_period, smooth_k_period 参数在评分逻辑中可能不会直接用到数值

    score.loc[(k_s < ext_os) | (d_s < ext_os)] = 95.0
    score.loc[(k_s > ext_ob) | (d_s > ext_ob)] = 5.0

    score.loc[((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))] = \
        np.maximum(score.loc[((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))], 85.0)
    score.loc[((k_s <= ext_ob) & (k_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))] = \
        np.minimum(score.loc[((k_s <= ext_ob) & (d_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))], 15.0)

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

def calculate_ma_score(close: pd.Series, ma: pd.Series, params: Optional[Dict] = None) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """移动平均线 (MA) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    close_s, ma_s = _safe_fillna_series(
        [close, ma],
        [None, lambda s: s.rolling(20, min_periods=1).mean()] # ma 填充后，如果全 NaN 使用 close 的滚动平均
    )
    if close_s.isnull().all() or ma_s.isnull().all(): # If close is all NaN, MA likely too or irrelevant, or MA fill failed
        # MODIFIED: 确保返回 Series 的索引是原始 close 的索引
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

def calculate_atr_score(atr: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """ATR 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    atr_s, = _safe_fillna_series([atr], [lambda s: s.mean()]) # atr 填充后，如果全 NaN 使用平均值
    if atr_s.isnull().all() or atr_s.mean() == 0: # 如果平均值也为 NaN 或为 0
        # MODIFIED: 确保返回 Series 的索引是原始 atr 的索引
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

def calculate_adl_score(adl: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """ADL (Accumulation/Distribution Line) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    adl_s, = _safe_fillna_series([adl], [0.0]) # ADL 中性0
    if adl_s.isnull().all():
        return pd.Series(50.0, index=adl.index).clip(0,100)

    score = pd.Series(50.0, index=adl_s.index)
    bullish_trend = adl_s > adl_s.shift(1)
    bearish_trend = adl_s < adl_s.shift(1)

    score.loc[bullish_trend] = 60.0
    score.loc[bearish_trend] = 40.0
    # Neutral for adl_s == adl_s.shift(1) is already 50.0

    return score.clip(0, 100)

def calculate_vwap_score(close: pd.Series, vwap: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """VWAP (Volume Weighted Average Price) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    close_s, vwap_s = _safe_fillna_series([close, vwap], [None, lambda s: s.mean() if s.mean() is not np.nan else close.mean()]) # vwap 填充后，如果全 NaN 使用均值，再不行用 close 均值
    if close_s.isnull().all() or vwap_s.isnull().all():
        # MODIFIED: 确保返回 Series 的索引是原始 close 的索引
        return pd.Series(50.0, index=close.index).clip(0,100)

    score = pd.Series(50.0, index=close_s.index)
    score.loc[close_s > vwap_s] = 60.0
    score.loc[close_s < vwap_s] = 40.0
    # score.loc[close_s == vwap_s] is already 50.0

    return score.clip(0, 100)

def calculate_ichimoku_score(close: pd.Series, tenkan: pd.Series, kijun: pd.Series, senkou_a: pd.Series, senkou_b: pd.Series, chikou: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """Ichimoku (一目均衡表) 评分 (0-100)。Simplified NaN handling."""
    # Ichimoku lines have inherent NaNs due to shifts. ffill/bfill is a simplification.
    # A more rigorous approach would respect these NaNs or use a sufficiently long data period.
    # Filling with close_s is a pragmatic choice if exact Ichimoku NaN propagation isn't critical.
    # MODIFIED: 使用 _safe_fillna_series 填充
    series_list = [close, tenkan, kijun, senkou_a, senkou_b, chikou]
    fill_values = [
        None, # close 优先 ffill/bfill
        lambda s: s.mean() if s.mean() is not np.nan else close.mean(), # tenkan 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s.mean() is not np.nan else close.mean(), # kijun 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s.mean() is not np.nan else close.mean(), # senkou_a 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s.mean() is not np.nan else close.mean(), # senkou_b 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s.mean() is not np.nan else close.mean() # chikou 填充后，如果全 NaN 使用均值
        ]
    c, tk, kj, sa, sb, cs = _safe_fillna_series(series_list, fill_values)

    # Check if any series is still all NaN after filling attempts
    if c.isnull().all() or tk.isnull().all() or kj.isnull().all() or sa.isnull().all() or sb.isnull().all() or cs.isnull().all():
        logger.warning("Ichimoku: One or more lines are all NaN after filling.")
        # MODIFIED: 确保返回 Series 的索引是原始 close 的索引
        return pd.Series(50.0, index=close.index).clip(0,100)

    score = pd.Series(50.0, index=c.index)

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

def calculate_mom_score(mom: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """MOM (Momentum) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_willr_score(willr: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """WILLR (%R) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_cmf_score(cmf: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """CMF (Chaikin Money Flow) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_obv_score(obv: pd.Series, obv_ma: pd.Series = None, obv_ma_period: int = None) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名
    """OBV (On Balance Volume) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充 OBV
    obv_s, = _safe_fillna_series([obv], [None]) # ffill/bfill first
    if obv_s.isnull().all():
        return pd.Series(50.0, index=obv.index).clip(0,100)
    # 如果填充后仍有 NaN，用均值填充
    obv_s = obv_s.fillna(obv_s.mean())
    if obv_s.isnull().all():
        return pd.Series(50.0, index=obv.index).clip(0,100)

    score = pd.Series(50.0, index=obv_s.index)

    # 如果提供了 OBV_MA series，进行交叉判断
    if obv_ma is not None and not obv_ma.isnull().all():
         # MODIFIED: 填充 OBV_MA，确保索引一致
         obv_ma_s, = _safe_fillna_series([obv_ma], [lambda s: s.mean()])
         if not obv_ma_s.isnull().all():
              buy_cross = (obv_s.shift(1) < obv_ma_s.shift(1)) & (obv_s >= obv_ma_s)
              sell_cross = (obv_s.shift(1) > obv_ma_s.shift(1)) & (obv_s <= obv_ma_s)
              score.loc[buy_cross] = np.maximum(score.loc[buy_cross], 70.0)
              score.loc[sell_cross] = np.minimum(score.loc[sell_cross], 30.0)

              # 在未交叉时，根据 OBV 相对于 MA 的位置调整分数
              not_cross_cond = ~buy_cross & ~sell_cross
              score.loc[(obv_s > obv_ma_s) & not_cross_cond] = np.maximum(score.loc[(obv_s > obv_ma_s) & not_cross_cond], 60.0)
              score.loc[(obv_s < obv_ma_s) & not_cross_cond] = np.minimum(score.loc[(obv_s < obv_ma_s) & not_cross_cond], 40.0)

         else:
             logger.warning("OBV 评分：OBV_MA 序列在填充后仍全为NaN，跳过 OBV_MA 相关评分逻辑。")


    # 如果没有提供 OBV_MA 或 OBV_MA 无效，或者在没有交叉时，使用 OBV 本身的趋势
    # 检查当前 score 中还是 50 的位置，应用 OBV 趋势评分
    neutral_score_mask = (score == 50.0)
    bullish_trend_no_cross = (obv_s > obv_s.shift(1)) & neutral_score_mask
    bearish_trend_no_cross = (obv_s < obv_s.shift(1)) & neutral_score_mask

    score.loc[bullish_trend_no_cross] = np.maximum(score.loc[bullish_trend_no_cross], 55.0)
    score.loc[bearish_trend_no_cross] = np.minimum(score.loc[bearish_trend_no_cross], 45.0)


    return score.clip(0, 100)

def calculate_kc_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """KC (Keltner Channel) 评分 (0-100)。"""
    # Similar to BOLL, NaN handling for bands is key.
    # MODIFIED: 使用 _safe_fillna_series 填充
    close_s, upper_s, mid_s, lower_s = _safe_fillna_series(
        [close, upper, mid, lower],
        [
            None, # close 优先 ffill/bfill
             lambda s: s.mean() + 1.5 * (s.rolling(20, min_periods=1).max() - s.rolling(20, min_periods=1).min()).mean() if s.mean() is not np.nan else close.mean() + 1.5 * (close.rolling(20, min_periods=1).max() - close.rolling(20, min_periods=1).min()).mean(), # upper fallback
             lambda s: s.mean() if s.mean() is not np.nan else close.mean(), # mid fallback
             lambda s: s.mean() - 1.5 * (s.rolling(20, min_periods=1).max() - s.rolling(20, min_periods=1).min()).mean() if s.mean() is not np.nan else close.mean() - 1.5 * (close.rolling(20, min_periods=1).max() - close.rolling(20, min_periods=1).min()).mean() # lower fallback
        ]
    )
    # Re-check for all NaNs after filling attempts
    if close_s.isnull().all() or upper_s.isnull().all() or mid_s.isnull().all() or lower_s.isnull().all():
         logger.warning("KC 评分：一个或多个关键序列在填充后仍全为NaN。")
         # MODIFIED: 确保返回 Series 的索引是原始 close 的索引
         return pd.Series(50.0, index=close.index).clip(0,100)


    score = pd.Series(50.0, index=close_s.index)

    score.loc[close_s <= lower_s] = 90.0
    buy_support = (close_s.shift(1) < lower_s.shift(1)) & (close_s >= lower_s)
    score.loc[buy_support] = np.maximum(score.loc[buy_support], 80.0)

    score.loc[close_s >= upper_s] = 10.0
    sell_pressure = (close_s.shift(1) > upper_s.shift(1)) & (close_s <= upper_s)
    score.loc[sell_pressure] = np.minimum(score.loc[sell_pressure], 20.0)

    buy_mid_cross = (close_s.shift(1) < mid_s.shift(1)) & (close_s >= mid_s)
    sell_mid_cross = (close_s.shift(1) > mid_s.shift(1)) & (close_s <= mid_s)
    score.loc[buy_mid_cross] = np.maximum(score.loc[buy_mid_cross], 65.0)
    score.loc[sell_mid_cross] = np.minimum(score.loc[sell_mid_cross], 35.0)

    not_extreme_cond = (close_s > lower_s) & (close_s < upper_s)
    not_mid_cross_cond = ~buy_mid_cross & ~sell_mid_cross

    is_above_mid = not_extreme_cond & not_mid_cross_cond & (close_s > mid_s)
    is_below_mid = not_extreme_cond & not_mid_cross_cond & (close_s < mid_s)
    score.loc[is_above_mid] = np.maximum(score.loc[is_above_mid], 55.0)
    score.loc[is_below_mid] = np.minimum(score.loc[is_below_mid], 45.0)

    return score.clip(0, 100)

def calculate_hv_score(hv: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """HV (Historical Volatility) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
    hv_s, = _safe_fillna_series([hv], [lambda s: s.mean()]) # hv 填充后，如果全 NaN 使用均值
    if hv_s.isnull().all() or hv_s.mean() == 0:
        # MODIFIED: 确保返回 Series 的索引是原始 hv 的索引
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

def calculate_vroc_score(vroc: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """VROC (Volume Rate of Change) 评分 (0-100)。"""
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_aroc_score(aroc: pd.Series) -> pd.Series: # MODIFIED: 移除 data 和 params，根据 File 2 签名
    """AROC (Absolute Rate of Change) 评分 (0-100)。"""
    # AROC is likely an alias for ROC, using same logic as calculate_roc_score
    # If AROC has a specific different interpretation (e.g. Aroon Oscillator), the logic would change.
    # Assuming it's similar to Price ROC:
    # MODIFIED: 使用 _safe_fillna_series 填充
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

def calculate_pivot_score(close: pd.Series, pivot_levels: Dict[str, pd.Series],
                          tf: str, # 增加时间框架参数，用于构建标准列名
                          params: Optional[Dict] = None) -> pd.Series: # MODIFIED: 移除 data，根据 File 2 签名，且 pivot_levels 期望的是 Dict[str, pd.Series]
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
        pivot_levels (Dict[str, pd.Series]): 包含 Pivot 水平的 Series 字典。
                                         键是内部 key 如 'PP', 'R1', 'F_S1'，值是对应的 Series。
        tf (str): 当前使用的时间框架，用于日志等。
        params (Dict, optional): 评分函数可能需要的额外参数。目前未使用。

    Returns:
        pd.Series: 计算出的 Pivot Points 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保索引一致，并填充 close 的 NaN 值
    close_filled = close.ffill().bfill()
    if close_filled.isnull().all():
        logger.warning("Pivot Points 评分：收盘价序列在填充后仍全为 NaN。")
        # MODIFIED: 确保返回 Series 的索引是原始 close 的索引
        return score.clip(0, 100) # 返回默认中性分

    # 从 pivot_levels 字典中提取需要的 Series
    pp_series = pivot_levels.get('PP')
    r_series = {k: v for k, v in pivot_levels.items() if k.startswith('R') and k != 'R'} # R1, R2, R3, R4
    s_series = {k: v for k, v in pivot_levels.items() if k.startswith('S') and k != 'S'} # S1, S2, S3, S4
    fr_series = {k: v for k, v in pivot_levels.items() if k.startswith('F_R')} # F_R1, F_R2, F_R3
    fs_series = {k: v for k, v in pivot_levels.items() if k.startswith('F_S')} # F_S1, F_S2, F_S3


    # 1. 价格与 Pivot Point (PP) 的相对位置
    if pp_series is not None and pp_series.notna().any():
        # 对齐 PP series 的索引到 close
        pp_series_aligned = pp_series.reindex(close.index)
        valid_pp_mask = pp_series_aligned.notna()
        score.loc[valid_pp_mask & (close_filled > pp_series_aligned)] = np.maximum(score.loc[valid_pp_mask & (close_filled > pp_series_aligned)], 55.0) # MODIFIED: 确保不覆盖更高分
        score.loc[valid_pp_mask & (close_filled < pp_series_aligned)] = np.minimum(score.loc[valid_pp_mask & (close_filled < pp_series_aligned)], 45.0) # MODIFIED: 确保不覆盖更低分
        # 精确在PP上的情况已经默认为50

    # 2. 价格突破/跌破支撑与阻力水平
    # 定义标准和斐波那契支撑/阻力级别及其基础分数和级别权重
    # 使用字典来存储 Series，键是内部 key (R1, S1, F_R1, etc.)
    level_data = {
        'R': {'series': r_series, 'base_score_breakout': 70, 'base_score_breakdown': None, 'level_multiplier': 5, 'is_resistance': True},
        'S': {'series': s_series, 'base_score_breakout': None, 'base_score_breakdown': 30, 'level_multiplier': 5, 'is_resistance': False},
        'F_R': {'series': fr_series, 'base_score_breakout': 75, 'base_score_breakdown': None, 'level_multiplier': 5, 'is_resistance': True}, # 斐波那契阻力突破可能更强
        'F_S': {'series': fs_series, 'base_score_breakout': None, 'base_score_breakdown': 25, 'level_multiplier': 5, 'is_resistance': False}, # 斐波那契支撑跌破可能更弱
    }

    for type_key, config in level_data.items():
        # 遍历该类型的所有级别 Series
        for level_key, level_series in config['series'].items():
            # 从 level_key 中解析级别数字 (e.g., 'R1' -> 1, 'F_S3' -> 3)
            try:
                 # 移除前缀，剩下的数字就是级别
                 level_num_str = level_key.replace(config['prefix'], '')
                 level_num = int(level_num_str)
            except (ValueError, TypeError):
                 logger.warning(f"Pivot Points 评分：无法从内部 key '{level_key}' 解析级别数字。跳过该级别。")
                 continue

            if level_series is not None and level_series.notna().any():
                # 对齐 level series 的索引到 close
                level_series_aligned = level_series.reindex(close.index)
                valid_level_mask = level_series_aligned.notna() # 只在 pivot level 非 NaN 的地方操作

                # 获取前一时刻的价格和支撑/阻力位 (需要先确保索引一致)
                close_prev = close_filled.shift(1)
                level_prev = level_series_aligned.shift(1)

                if config['is_resistance']: # 处理阻力位
                    # 价格向上突破阻力
                    breakout_cond_full = (close_prev < level_prev) & (close_filled >= level_series_aligned)
                    breakout_cond = breakout_cond_full & valid_level_mask # 应用掩码

                    if breakout_cond.any():
                        breakout_score_value = config['base_score_breakout'] + level_num * config['level_multiplier']
                        score.loc[breakout_cond] = np.maximum(score.loc[breakout_cond], breakout_score_value)

                    # 价格在阻力位下方（未突破时，作为阻力区的参考）
                    # 越接近高级别阻力，分数越低（更看跌）
                    below_resistance_cond = (close_filled < level_series_aligned) & (~breakout_cond_full) & valid_level_mask
                    if below_resistance_cond.any():
                        penalty = level_num * 2.5 # 示例惩罚值，越高级别惩罚越多
                        score.loc[below_resistance_cond] = np.minimum(score.loc[below_resistance_cond], 50 - penalty)

                else: # 处理支撑位
                    # 价格向下跌破支撑
                    breakdown_cond_full = (close_prev > level_prev) & (close_filled <= level_series_aligned)
                    breakdown_cond = breakdown_cond_full & valid_level_mask # 应用掩码

                    if breakdown_cond.any():
                        breakdown_score_value = config['base_score_breakdown'] - level_num * config['level_multiplier']
                        score.loc[breakdown_cond] = np.minimum(score.loc[breakdown_cond], breakdown_score_value)

                    # 价格在支撑位上方（未跌破时，作为支撑区的参考）
                    # 越接近高级别支撑，分数越高（更看涨）
                    above_support_cond = (close_filled > level_series_aligned) & (~breakdown_cond_full) & valid_level_mask
                    if above_support_cond.any():
                        bonus = level_num * 2.5 # 示例奖励值，越高级别奖励越多
                        score.loc[above_support_cond] = np.maximum(score.loc[above_support_cond], 50 + bonus)


    return score.clip(0, 100)

def build_expected_col_name(indicator_key: str, internal_key: str, params: List[Any], tf_suffix: str) -> Optional[str]:
    """
    根据指标 key, 内部 key, 参数列表和时间框架后缀构建期望的列名。
    """
    if not tf_suffix:
         logger.error(f"构建列名失败: 时间框架后缀为空。 indicator_key={indicator_key}, internal_key={internal_key}")
         return None

    def format_param(p):
        if isinstance(p, float):
            # 尝试格式化为 .1f 或 .2f，取决于参数类型和指标
            if indicator_key == 'boll': return f"{p:.1f}" # BOLL std_dev 常见格式
            if indicator_key == 'sar':
                 # SAR af_step 和 max_af 可能有不同格式，根据约定文件
                 # Convention文件里是 SAR_0.02_0.2_{timeframe}
                 # 所以 af_step 是 .2f, max_af 是 .1f
                 if len(params) == 2:
                      if p == params[0]: return f"{p:.2f}" # af_step
                      if p == params[1]: return f"{p:.1f}" # max_af
                 # Fallback if cannot determine based on position
                 return f"{p:.2f}" # 默认 .2f
            # 其他浮点参数如果存在，需要根据实际命名规范调整
            return str(p) # 默认转换为字符串
        return str(p)

    # MODIFIED: 修正参数部分的构建逻辑，确保只包含实际参数
    # MACD_12_26_9, RSI_14, K_7_3_3, BOLL_20_2.0 等，参数在指标名后用下划线分隔
    param_str_parts = [format_param(p) for p in params]
    param_part = '_'.join(param_str_parts) if param_str_parts else ""
    param_suffix = f"_{param_part}" if param_part else ""


    if indicator_key == 'macd' and len(params) == 3:
        prefix_map = {'macd_series': 'MACD', 'macd_d': 'MACDs', 'macd_h': 'MACDh'}
        prefix = prefix_map.get(internal_key)
        if prefix:
            return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'rsi' and len(params) == 1:
        if internal_key == 'rsi':
            return f"RSI{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'kdj' and len(params) == 3:
        prefix_map = {'k': 'K', 'd': 'D', 'j': 'J'}
        prefix = prefix_map.get(internal_key)
        if prefix:
             return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'boll' and len(params) == 2:
        prefix_map = {'upper': 'BBU', 'mid': 'BBM', 'lower': 'BBL', 'close': 'close'}
        prefix = prefix_map.get(internal_key)
        if prefix:
             if internal_key == 'close':
                 return f"{prefix}_{tf_suffix}" # close 列名没有参数部分
             return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'cci' and len(params) == 1:
        if internal_key == 'cci':
             return f"CCI{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'mfi' and len(params) == 1:
        if internal_key == 'mfi':
             return f"MFI{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'roc' and len(params) == 1:
        if internal_key == 'roc':
             return f"ROC{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'dmi' and len(params) == 1:
        prefix_map = {'pdi': 'PDI', 'ndi': 'NDI', 'adx': 'ADX'}
        prefix = prefix_map.get(internal_key)
        if prefix:
             return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'sar' and len(params) == 2:
        prefix_map = {'sar': 'SAR', 'close': 'close'}
        prefix = prefix_map.get(internal_key)
        if prefix:
             if internal_key == 'close':
                  return f"{prefix}_{tf_suffix}" # close 列名没有参数部分
             # SAR 参数格式化特殊处理，af_step(.2f)_max_af(.1f)
             param_str_sar = f"{format_param(params[0])}_{format_param(params[1])}"
             return f"{prefix}_{param_str_sar}_{tf_suffix}" # MODIFIED: 修正 SAR 参数部分的构建

    elif indicator_key == 'stoch' and len(params) == 3:
        prefix_map = {'k': 'STOCHk', 'd': 'STOCHd'}
        prefix = prefix_map.get(internal_key)
        if prefix:
             return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key in ['ema', 'sma'] and len(params) == 1:
        prefix_map = {'ma': indicator_key.upper(), 'close': 'close'}
        prefix = prefix_map.get(internal_key)
        if prefix:
            if internal_key == 'close':
                 return f"{prefix}_{tf_suffix}" # close 列名没有参数部分
            return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'atr' and len(params) == 1:
        if internal_key == 'atr':
             return f"ATR{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'adl' and not params: # ADL 没有参数在列名中
        if internal_key == 'adl':
             return f"ADL_{tf_suffix}"

    elif indicator_key == 'vwap' and not params: # VWAP 通常没有参数在列名中 (除非有anchor)
         prefix_map = {'vwap': 'VWAP', 'close': 'close'}
         prefix = prefix_map.get(internal_key)
         if prefix:
             return f"{prefix}_{tf_suffix}" # VWAP/close 列名模式: NAME_timeframe

    elif indicator_key == 'ichimoku' and len(params) == 3:
         prefix_map = {
             'close': 'close', 'tenkan': 'TENKAN', 'kijun': 'KIJUN',
             'senkou_a': 'SENKOU_A', 'senkou_b': 'SENKOU_B', 'chikou': 'CHIKOU'
         }
         prefix = prefix_map.get(internal_key)
         if prefix:
             if internal_key == 'close':
                  return f"{prefix}_{tf_suffix}"

             p_tenkan, p_kijun, p_senkou_b = params # params 列表包含 Ichimoku 的主要参数
             # Ichimoku 列名模式复杂，根据 internal_key 和对应参数构建
             if internal_key == 'tenkan': return f"TENKAN_{p_tenkan}_{tf_suffix}"
             if internal_key == 'kijun': return f"KIJUN_{p_kijun}_{tf_suffix}"
             if internal_key == 'chikou': return f"CHIKOU_{p_kijun}_{tf_suffix}" # Chikou 参数是 Kijun 的
             if internal_key == 'senkou_a': return f"SENKOU_A_{p_tenkan}_{p_kijun}_{tf_suffix}" # Senkou A 参数是 Tenkan 和 Kijun 的
             if internal_key == 'senkou_b': return f"SENKOU_B_{p_senkou_b}_{tf_suffix}" # Senkou B 参数是自己的

    elif indicator_key == 'mom' and len(params) == 1:
        if internal_key == 'mom':
             return f"MOM{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'willr' and len(params) == 1:
        if internal_key == 'willr':
             return f"WILLR{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'cmf' and len(params) == 1:
        if internal_key == 'cmf':
             return f"CMF{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'obv':
        if internal_key == 'obv':
             return f"OBV_{tf_suffix}"
        # OBV_MA 需要单独构建，参数是 OBV_MA 的周期
        if internal_key == 'obv_ma' and len(params) == 1: # params 此时应该是 OBV_MA 的周期列表 [obv_ma_period]
             p_obv_ma = params[0]
             return f"OBV_MA_{p_obv_ma}_{tf_suffix}"


    elif indicator_key == 'kc' and len(params) == 2:
        prefix_map = {'upper': 'KCU', 'mid': 'KCM', 'lower': 'KCL', 'close': 'close'}
        prefix = prefix_map.get(internal_key)
        if prefix:
             if internal_key == 'close':
                 return f"{prefix}_{tf_suffix}"
             return f"{prefix}{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'hv' and len(params) == 1:
        if internal_key == 'hv':
             return f"HV{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'vroc' and len(params) == 1:
        if internal_key == 'vroc':
             return f"VROC{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'aroc' and len(params) == 1:
        if internal_key == 'aroc':
             return f"AROC{param_suffix}_{tf_suffix}" # MODIFIED: 使用修正后的 param_suffix

    elif indicator_key == 'pivot' and tf_suffix.upper() == 'D' and internal_key == 'close':
         return f"close_{tf_suffix}"
    # Pivot levels 不需要通过 build_expected_col_name 单独构建，它们是列表，在查找时统一处理

    logger.warning(f"无法为指标 '{indicator_key}' 构建列名，内部 key: '{internal_key}', 参数: {params}, 后缀: '{tf_suffix}'")
    return None

def parse_col_params(col_name: str, indicator_key: str, tf_suffix: str) -> List[Any] | None:
    """
    尝试从包含时间框架后缀的列名中解析指标参数。
    """
    if not col_name.endswith(f"_{tf_suffix}"):
         return None # 后缀不匹配

    base_name_with_params = col_name[:-len(f"_{tf_suffix}")] # 移除后缀
    parts = base_name_with_params.split('_')

    try:
        if indicator_key == 'macd' and parts[0] in ['MACD', 'MACDh', 'MACDs']:
            # MACD_fast_slow_signal
            if len(parts) >= 4: return [int(parts[1]), int(parts[2]), int(parts[3])]
        elif indicator_key == 'rsi' and parts[0] == 'RSI':
            # RSI_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'kdj' and parts[0] in ['K', 'D', 'J']:
            # K_period_signal_period_smooth_k_period
             if len(parts) >= 4: return [int(parts[1]), int(parts[2]), int(parts[3])]
        elif indicator_key == 'boll' and parts[0] in ['BBL', 'BBM', 'BBU']:
            # BB{L/M/U}_period_std_dev
             if len(parts) >= 3: return [int(parts[1]), float(parts[2])] # std_dev 是浮点数
        elif indicator_key == 'cci' and parts[0] == 'CCI':
            # CCI_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'mfi' and parts[0] == 'MFI':
            # MFI_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'roc' and parts[0] == 'ROC':
            # ROC_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'dmi' and parts[0] in ['PDI', 'NDI', 'ADX']:
            # DMI_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'sar' and parts[0] == 'SAR':
             # SAR_af_step_max_af (浮点数)
             if len(parts) >= 3: return [float(parts[1]), float(parts[2])]
        elif indicator_key == 'stoch' and parts[0] in ['STOCHk', 'STOCHd']:
            # STOCHk_k_period_d_period_smooth_k_period
             if len(parts) >= 4: return [int(parts[1]), int(parts[2]), int(parts[3])]
        elif indicator_key in ['ema', 'sma'] and parts[0] in ['EMA', 'SMA']:
            # MA_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'atr' and parts[0] == 'ATR':
             # ATR_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'adl' and parts[0] == 'ADL':
            # ADL_{timeframe}, 无参数
            if len(parts) == 1: return []
        elif indicator_key == 'vwap' and parts[0] == 'VWAP':
             # VWAP_{timeframe} 或 VWAP_{anchor}_{timeframe}
             # 根据日志，VWAP 列名是 VWAP_5 等，没有anchor参数
             if len(parts) == 1: return [] # 假设列名中不包含参数
        elif indicator_key == 'ichimoku' and parts[0] in ['TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU_A', 'SENKOU_B']:
            # Ichimoku 参数解析复杂，尝试从主要列名解析
            # TENKAN_period, KIJUN_period, CHIKOU_period, SENKOU_A_tenkan_kijun, SENKOU_B_period
            if parts[0] in ['TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU_B'] and len(parts) >= 2:
                 return [int(parts[1])] # 尝试解析单个周期 (Tenkan, Kijun, Chikou, Senkou B)
            elif parts[0] == 'SENKOU_A' and len(parts) >= 3:
                 return [int(parts[1]), int(parts[2])] # 尝试解析两个周期 (Senkou A)
            return None # 参数格式不匹配
        elif indicator_key == 'mom' and parts[0] == 'MOM':
            # MOM_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'willr' and parts[0] == 'WILLR':
            # WILLR_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'cmf' and parts[0] == 'CMF':
            # CMF_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'obv' and parts[0] == 'OBV':
             # OBV_{timeframe}, 无参数
            if len(parts) == 1: return []
        elif indicator_key == 'obv' and parts[0] == 'OBV_MA':
             # OBV_MA_period_{timeframe}
             if len(parts) >= 2: return [int(parts[1])] # OBV_MA 列名中的周期
        elif indicator_key == 'kc' and parts[0] in ['KCL', 'KCM', 'KCU']:
             # KC_ema_period_atr_period
             if len(parts) >= 3: return [int(parts[1]), int(parts[2])]
        elif indicator_key == 'hv' and parts[0] == 'HV':
            # HV_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'vroc' and parts[0] == 'VROC':
            # VROC_period
             if len(parts) >= 2: return [int(parts[1])]
        elif indicator_key == 'aroc' and parts[0] == 'AROC':
            # AROC_period
             if len(parts) >= 2: return [int(parts[1])]
        # Pivot 列名不包含参数，只有基础名和后缀，在主函数中特殊处理

        return None # 未知列名模式或解析失败
    except (ValueError, IndexError):
        logger.debug(f"从列名 '{col_name}' 解析参数失败 (indicator: {indicator_key}, suffix: {tf_suffix}).", exc_info=True)
        return None # 参数转换失败或索引越界

def calculate_all_indicator_scores(data: pd.DataFrame, bs_params: Dict, indicator_configs: List[Dict] ) -> pd.DataFrame:
    """
    根据配置计算所有指定指标的评分 (0-100)。

    此函数遍历 base_scoring 参数中指定的需要评分的指标和时间框架，
    从输入的 DataFrame 中查找对应的指标数据列，并调用相应的评分计算函数。
    指标列名根据 indicator_naming_conventions.json 文件中的命名规范构建，
    并能根据实际DataFrame中存在的列名（包含实际计算参数）进行匹配。

    :param data: 包含所有原始数据和指标的 DataFrame。列名应包含时间级别后缀和计算参数，
                 例如 'close_15', 'MACD_12_26_9_30'。
    :param bs_params: base_scoring 参数字典，包含 'score_indicators' (需要评分的指标列表)
                      和 'timeframes' (需要计算评分的时间框架列表)，以及各指标的评分参数。
                      注意：这里的指标参数是策略配置中定义的，键名可能与评分函数参数名不同。
    :param indicator_configs: 由 indicator_services.prepare_strategy_dataframe 生成的，
                              包含每个指标计算函数、参数、时间框架的列表。用于辅助查找列名。
    :return: 返回一个 DataFrame，包含所有时间框架和指标的评分列。
             列名格式: SCORE_{指标名}_{时间级别}。
             如果某个指标或时间框架的数据缺失或计算失败，对应的评分列将填充默认中性分 50.0。
    """
    # 初始化用于存储所有评分结果的 DataFrame
    scoring_results = pd.DataFrame(index=data.index)
    # 如果输入 DataFrame 是空的，也直接返回空结果
    if data.empty:
         logger.warning("输入 DataFrame 为空，无法计算指标评分。")
         return scoring_results
    # 获取需要评分的指标列表和时间框架列表
    score_indicators_keys = bs_params.get('score_indicators', [])
    score_timeframes = bs_params.get('timeframes', [])
    if not score_indicators_keys or not score_timeframes:
        logger.warning("未配置需要评分的指标或时间框架 (base_scoring.score_indicators 或 base_scoring.timeframes)。")
        return scoring_results

    logger.info(f"开始计算指标评分，指标: {score_indicators_keys}, 时间框架: {score_timeframes}")
    # 加载 indicator_naming_conventions.json 文件中的命名规范
    # naming_conventions_path = getattr(settings, 'INDICATOR_PARAMETERS_CONFIG_PATH', None)
    # naming_conventions = {}
    # indicator_naming = {}
    # derivative_naming = {}
    # if naming_conventions_path:
    #     try:
    #         with open(naming_conventions_path, 'r', encoding='utf-8') as f:
    #             naming_conventions = json.load(f)
    #         indicator_naming = naming_conventions.get('indicator_naming_conventions', {})
    #         derivative_naming = naming_conventions.get('derivative_feature_naming_conventions', {})
    #         logger.info(f"成功加载指标命名规范配置文件: {naming_conventions_path}")
    #     except FileNotFoundError:
    #         logger.error(f"指标命名规范配置文件未找到: {naming_conventions_path}，将使用默认命名逻辑。")
    #     except json.JSONDecodeError:
    #          logger.error(f"指标命名规范配置文件格式错误: {naming_conventions_path}，请检查JSON格式。将使用默认命名逻辑。")
    #     except Exception as e:
    #         logger.error(f"加载指标命名规范配置文件失败: {naming_conventions_path}: {e}，将使用默认命名逻辑。")
    # else:
    #      logger.warning("未配置 INDICATOR_PARAMETERS_CONFIG_PATH 路径，将使用默认命名逻辑。")
    # 从 indicator_configs 中提取可能的列名信息 (优先使用此映射)
    config_column_mapping_by_tf: Dict[str, Dict[str, List[str]]] = {} # {timeframe: {indicator_name: [col1, col2, ...]}}
    for config in indicator_configs:
        indicator_name = config.get('name', '').lower()
        timeframe = str(config.get('timeframe', '')) # 确保是字符串
        output_columns = config.get('output_columns', [])
        if isinstance(output_columns, str):
             output_columns = [output_columns]
        if indicator_name and timeframe and output_columns:
            if timeframe not in config_column_mapping_by_tf:
                config_column_mapping_by_tf[timeframe] = {}
            config_column_mapping_by_tf[timeframe][indicator_name] = output_columns
            # logger.debug(f"从配置中提取指标 {indicator_name} 在时间框架 {timeframe} 的列名: {output_columns}") # 调试信息
    # --- 集中配置指标的评分函数、所需内部键和列名前缀及参数映射 ---
    # 调整 indicator_scoring_info 结构，明确参数传递风格和映射
    indicator_scoring_info: Dict[str, Dict[str, Any]] = {
        'macd': { # File 2 signature: macd_series, macd_d, macd_h
            'func': calculate_macd_score, 'param_passing_style': 'none', # Based on File 2 signature
            'bs_param_key_to_score_func_arg': {}, # No params from bs_params passed to this function directly
            'defaults': {}, # No defaults for params used in this function
            'required_keys': ['macd_series', 'macd_d', 'macd_h'], 'prefixes': ['MACD_', 'MACDh_', 'MACDs_']
        },
        'rsi': { # File 2 signature: rsi, params
            'func': calculate_rsi_score, 'param_passing_style': 'dict',
            'bs_param_key_to_score_func_arg': {'rsi_period': 'period', 'rsi_oversold': 'oversold', 'rsi_overbought': 'overbought', 'rsi_extreme_oversold': 'extreme_oversold', 'rsi_extreme_overbought': 'extreme_overbought'},
            'defaults': {'period': 14, 'oversold': 30, 'overbought': 70, 'extreme_oversold': 20, 'extreme_overbought': 80},
            'required_keys': ['rsi'], 'prefixes': ['RSI_']
        },
        'kdj': { # File 2 signature: k, d, j, params
            'func': calculate_kdj_score, 'param_passing_style': 'dict',
            'bs_param_key_to_score_func_arg': {'kdj_period': 'period', 'kdj_signal_period': 'signal_period', 'kdj_smooth_k_period': 'smooth_k_period', 'kdj_oversold': 'oversold', 'kdj_overbought': 'overbought', 'kdj_extreme_oversold': 'extreme_oversold', 'kdj_extreme_overbought': 'extreme_overbought'},
            'defaults': {'period': 9, 'signal_period': 3, 'smooth_k_period': 3, 'oversold': 20, 'overbought': 80, 'extreme_oversold': 10, 'extreme_overbought': 90},
            'required_keys': ['k', 'd', 'j'], 'prefixes': ['K_', 'D_', 'J_']
        },
        'boll': { # File 2 signature: close, upper, mid, lower
           'func': calculate_boll_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, # No params from bs_params passed to this function directly
           'defaults': {}, # No defaults for params used in this function
           'required_keys': ['close', 'upper', 'mid', 'lower'], 'prefixes': ['BBL_', 'BBM_', 'BBU_']
        },
        'cci': { # File 2 signature: cci, params
           'func': calculate_cci_score, 'param_passing_style': 'dict',
           'bs_param_key_to_score_func_arg': {'cci_period': 'period', 'cci_threshold': 'threshold', 'cci_extreme_threshold': 'extreme_threshold'},
           'defaults': {'period': 14, 'threshold': 100, 'extreme_threshold': 200},
           'required_keys': ['cci'], 'prefixes': ['CCI_']
        },
        'mfi': { # File 2 signature: mfi, params
           'func': calculate_mfi_score, 'param_passing_style': 'dict',
           'bs_param_key_to_score_func_arg': {'mfi_period': 'period', 'mfi_oversold': 'oversold', 'mfi_overbought': 'overbought', 'mfi_extreme_oversold': 'extreme_oversold', 'mfi_extreme_overbought': 'extreme_overbought'},
           'defaults': {'period': 14, 'oversold': 20, 'overbought': 80, 'extreme_oversold': 10, 'extreme_overbought': 90},
           'required_keys': ['mfi'], 'prefixes': ['MFI_']
        },
        'roc': { # File 2 signature: roc
           'func': calculate_roc_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, # No params from bs_params passed to this function directly
           'defaults': {}, # No defaults for params used in this function
           'required_keys': ['roc'], 'prefixes': ['ROC_']
        },
        'dmi': { # File 2 signature: pdi, ndi, adx, params
           'func': calculate_dmi_score, 'param_passing_style': 'dict',
           'bs_param_key_to_score_func_arg': {'dmi_period': 'period', 'adx_threshold': 'adx_threshold', 'adx_strong_threshold': 'adx_strong_threshold'},
           'defaults': {'period': 14, 'adx_threshold': 25, 'adx_strong_threshold': 40},
           'required_keys': ['pdi', 'ndi', 'adx'], 'prefixes': ['PDI_', 'NDI_', 'ADX_']
        },
        'sar': { # File 2 signature: close, sar
           'func': calculate_sar_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, # No params from bs_params passed to this function directly
           'defaults': {}, # No defaults for params used in this function
           'required_keys': ['close', 'sar'], 'prefixes': ['SAR_']
        },
        'stoch': { # File 2 signature: k, d, params
           'func': calculate_stoch_score, 'param_passing_style': 'dict',
           'bs_param_key_to_score_func_arg': {'stoch_k_period': 'k_period', 'stoch_d_period': 'd_period', 'stoch_smooth_k_period': 'smooth_k_period', 'stoch_oversold': 'stoch_oversold', 'stoch_overbought': 'stoch_overbought', 'stoch_extreme_oversold': 'stoch_extreme_oversold', 'stoch_extreme_overbought': 'stoch_extreme_overbought'},
           'defaults': {'k_period': 14, 'd_period': 3, 'smooth_k_period': 3, 'stoch_oversold': 20, 'stoch_overbought': 80, 'stoch_extreme_oversold': 10, 'stoch_extreme_overbought': 90},
           'required_keys': ['k', 'd'], 'prefixes': ['STOCHk_', 'STOCHd_']
        },
        'ema': { # File 2 signature: close, ma, params
           'func': calculate_ma_score, 'param_passing_style': 'dict',
           'bs_param_key_to_score_func_arg': {'ema_period': 'period'},
           'defaults': {'period': 20}, 'required_keys': ['close', 'ma'], 'prefixes': ['EMA_']},
        'sma': { # File 2 signature: close, ma, params
           'func': calculate_ma_score, 'param_passing_style': 'dict',
           'bs_param_key_to_score_func_arg': {'sma_period': 'period'},
           'defaults': {'period': 20}, 'required_keys': ['close', 'ma'], 'prefixes': ['SMA_']
        },
        'atr': { # File 2 signature: atr
           'func': calculate_atr_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['atr'], 'prefixes': ['ATR_']
        },
        'adl': { # File 2 signature: adl
           'func': calculate_adl_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['adl'], 'prefixes': ['ADL_']
        },
        'vwap': { # File 2 signature: close, vwap
           'func': calculate_vwap_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['close', 'vwap'], 'prefixes': ['VWAP_']
        },
        'ichimoku': { # File 2 signature: close, tenkan, kijun, senkou_a, senkou_b, chikou
           'func': calculate_ichimoku_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['close', 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'], 'prefixes': ['TENKAN_', 'KIJUN_', 'CHIKOU_', 'SENKOU_A_', 'SENKOU_B_']
        },
        'mom': { # File 2 signature: mom
           'func': calculate_mom_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['mom'], 'prefixes': ['MOM_']
        },
        'willr': { # File 2 signature: willr
           'func': calculate_willr_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['willr'], 'prefixes': ['WILLR_']
        },
        'cmf': { # File 2 signature: cmf
           'func': calculate_cmf_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['cmf'], 'prefixes': ['CMF_']
        },
        'obv': { # File 2 signature: obv, obv_ma=None, obv_ma_period=None
           'func': calculate_obv_score, 'param_passing_style': 'individual', # Can accept obv_ma and obv_ma_period individually
           'bs_param_key_to_score_func_arg': {'obv_ma_period': 'obv_ma_period'}, # This parameter is passed individually
           'defaults': {'obv_ma_period': 10},
           'required_keys': ['obv'], 'prefixes': ['OBV_'] # Required key is 'obv', 'obv_ma' is optional for scoring function
        },
        'kc': { # File 2 signature: close, upper, mid, lower
           'func': calculate_kc_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['close', 'upper', 'mid', 'lower'], 'prefixes': ['KCL_', 'KCM_', 'KCU_']
        },
        'hv': { # File 2 signature: hv
           'func': calculate_hv_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['hv'], 'prefixes': ['HV_']
        },
        'vroc': { # File 2 signature: vroc
           'func': calculate_vroc_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['vroc'], 'prefixes': ['VROC_']
        },
        'aroc': { # File 2 signature: aroc
           'func': calculate_aroc_score, 'param_passing_style': 'none', # Based on File 2 signature
           'bs_param_key_to_score_func_arg': {}, 'defaults': {}, 'required_keys': ['aroc'], 'prefixes': ['AROC_']
        },
        'pivot': { # File 2 signature: close, pivot_levels, tf, params
           'func': calculate_pivot_score, 'param_passing_style': 'dict', # Accepts params: Dict
           'bs_param_key_to_score_func_arg': {}, # No params from bs_params go into the 'params' dict currently defined in File 2
           'defaults': {}, # No defaults for params dict in File 2 currently
           'required_keys': ['close', 'pivot_levels'], 'prefixes': [] # Requires close series and a dict/list for pivot_levels
        }
    }
    # 遍历需要评分的指标 key 和时间框架
    for indicator_key in score_indicators_keys:
        info = indicator_scoring_info.get(indicator_key)
        if not info:
             logger.warning(f"指标 '{indicator_key}' 未找到对应的评分函数定义或配置，跳过评分计算。")
             continue # 跳过当前指标的所有时间框架
        score_func = info['func']
        required_score_keys = info['required_keys']
        column_pattern_prefixes = info['prefixes']
        # MODIFIED: 获取参数传递风格和映射
        param_passing_style = info['param_passing_style']
        bs_param_key_to_score_func_arg = info['bs_param_key_to_score_func_arg']
        defaults = info['defaults']
        if score_func is None:
             logger.warning(f"评分函数 calculate_{indicator_key}_score 未定义，无法计算指标 '{indicator_key}' 的评分。")
             continue
        # 遍历需要计算评分的时间框架
        for tf_score in score_timeframes:
            indicator_cols_for_score: Dict[str, str | List[str]] = {}  # {内部 key: 实际列名 或 [实际列名列表]}
            found = False # 标记是否成功找到所有必要的列 (重置每个时间框架)
            tf_score_str = str(tf_score) # 确保时间框架是字符串，用于后缀匹配
            # 构建可能的时间框架后缀列表
            possible_tf_suffixes = [
                tf_score_str, f"{tf_score_str}m", f"{tf_score_str}min", tf_score_str.upper(),
                f"{tf_score_str}M", f"{tf_score_str}MIN", f"T{tf_score_str}", f"t{tf_score_str}"
            ]
            # 确保日线时间框架 'D' 被正确处理，并移除数字后缀
            if tf_score_str.upper() == 'D':
                 possible_tf_suffixes = ['D', 'd']
            # 移除重复项并保持顺序
            possible_tf_suffixes = list(dict.fromkeys(possible_tf_suffixes))
            # --- 查找当前时间框架下的指标列 ---
            # 优先尝试使用 indicator_configs 提供的列名映射
            tf_config_mapping = config_column_mapping_by_tf.get(tf_score_str, {})
            config_cols = tf_config_mapping.get(indicator_key, [])
            if config_cols:
                current_indicator_cols_attempt: Dict[str, str | List[str]] = {}
                config_mapping_successful = False
                 # 根据指标类型，从 config_cols 中提取对应的列名并检查是否存在
                 # MODIFIED: 改进 config_column_mapping 处理逻辑，特别是多列指标和特殊列 (close, pivot_levels)
                all_config_cols_exist = True
                temp_cols_from_config: Dict[str, str | List[str]] = {} # 暂存映射结果
                if indicator_key == 'pivot' and tf_score_str.upper() == 'D':
                    # Pivot 需要 close 列和 pivot_levels 列表
                    close_col = f"close_{tf_score_str}" if f"close_{tf_score_str}" in data.columns else None
                    if close_col:
                        temp_cols_from_config['close'] = close_col
                        # config_cols 应该是 pivot levels 的列表
                        if 'pivot_levels' in required_score_keys and isinstance(config_cols, list) and config_cols:
                            # 检查 config_cols 中的所有列是否存在
                            if all(col in data.columns for col in config_cols):
                                    temp_cols_from_config['pivot_levels'] = config_cols
                            else:
                                    all_config_cols_exist = False
                                    logger.warning(f"指标 'pivot' 在时间框架 {tf_score} 的配置映射中部分 pivot_levels 列在 DataFrame 中不存在。") # MODIFIED: 增加日志
                        else:
                            all_config_cols_exist = False
                            logger.warning(f"指标 'pivot' 在时间框架 {tf_score} 的配置映射中需要 pivot_levels，但 config_cols 格式不正确或为空。") # MODIFIED: 增加日志
                    else:
                        all_config_cols_exist = False
                        logger.warning(f"指标 'pivot' 在时间框架 {tf_score} 的配置映射中需要 close 列，但未找到 {f'close_{tf_score_str}'}。") # MODIFIED: 增加日志
                elif 'close' in required_score_keys: # 其他需要 close 列的指标
                    close_col = f"close_{tf_score_str}" if f"close_{tf_score_str}" in data.columns else None
                    if close_col:
                        temp_cols_from_config['close'] = close_col
                        # 检查 config_cols 中提供的列是否存在
                        if all(col in data.columns for col in config_cols):
                            # 尝试根据 required_score_keys 映射 config_cols 到 internal_key
                            # 这个映射逻辑依赖于 config_cols 的顺序或命名规范，比较脆弱。
                            # 最好是在 indicator_configs 中就提供 internal_key -> col_name 的映射
                            # 暂且假设 config_cols 严格按照 required_score_keys 中除了 'close' 之外的顺序
                            other_required_keys = [k for k in required_score_keys if k not in ['close', 'pivot_levels']]
                            if len(config_cols) == len(other_required_keys):
                                temp_cols_from_config.update(dict(zip(other_required_keys, config_cols)))
                            else:
                                all_config_cols_exist = False
                                logger.debug(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的 config_cols ({config_cols}) 数量与 required_score_keys ({other_required_keys}) 不匹配，尝试后缀匹配。") # MODIFIED: 增加日志
                        else:
                            all_config_cols_exist = False
                            logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的 config_cols ({config_cols}) 在 DataFrame 中不存在。") # MODIFIED: 增加日志
                    else:
                        all_config_cols_exist = False
                        logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的配置映射中需要 close 列，但未找到 {f'close_{tf_score_str}'}。") # MODIFIED: 增加日志
                else: # 不需要 close 列的指标
                    # 检查 config_cols 中提供的列是否存在
                    if all(col in data.columns for col in config_cols):
                        # 尝试根据 required_score_keys 映射 config_cols 到 internal_key
                        # 暂且假设 config_cols 严格按照 required_score_keys 的顺序
                        if len(config_cols) == len(required_score_keys):
                            temp_cols_from_config = dict(zip(required_score_keys, config_cols))
                        else:
                            all_config_cols_exist = False
                            logger.debug(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的 config_cols ({config_cols}) 数量与 required_score_keys ({required_score_keys}) 不匹配，尝试后缀匹配。") # MODIFIED: 增加日志
                    else:
                        all_config_cols_exist = False
                        logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的 config_cols ({config_cols}) 在 DataFrame 中不存在。") # MODIFIED: 增加日志
                 # 如果所有必需列都找到了 (通过配置映射)
                required_keys_from_config_present = all(k in temp_cols_from_config for k in required_score_keys)
                # 特别处理 Pivot，确保 'pivot_levels' 键存在且对应一个非空列表
                if indicator_key == 'pivot' and 'pivot_levels' in required_score_keys:
                    if not (isinstance(temp_cols_from_config.get('pivot_levels'), list) and temp_cols_from_config['pivot_levels']):
                        required_keys_from_config_present = False
                if all_config_cols_exist and required_keys_from_config_present:
                    indicator_cols_for_score = temp_cols_from_config
                    found = True
                    # logger.debug(f"通过 config_column_mapping 找到指标 '{indicator_key}' 在时间框架 {tf_score} 的列。") # 调试信息
        # 如果 config_column_mapping 未找到或不完整，尝试使用后缀匹配和参数解析
        if not found:
            # logger.debug(f"config_column_mapping 未成功找到，尝试后缀匹配指标 '{indicator_key}' 在时间框架 {tf_score}") # 调试信息
            for tf_suffix in possible_tf_suffixes:
                # 特殊处理 Pivot，Pivot 列名不包含参数，且通常只在日线 D 计算
                if indicator_key == 'pivot':
                    if tf_suffix.upper() != 'D': continue # Pivot通常只在日线计算
                    # 查找所有 Pivot level 列 (根据命名规范的基础名)
                    pivot_cols_base = ["PP", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4", "F_R1", "F_R2", "F_R3", "F_S1", "F_S2", "F_S3"] # 根据 convention 文件
                    pivot_cols_with_suffix = [f"{col}_{tf_suffix}" for col in pivot_cols_base]
                    close_col = f"close_{tf_suffix}"
                    # 检查 close 列和所有 pivot level 列是否存在
                    if close_col in data.columns and all(col in data.columns for col in pivot_cols_with_suffix):
                        # Pivot 评分函数期望 pivot_levels 参数是一个 {internal_key: Series} 字典
                        pivot_series_dict = {
                            p_base: data[f"{p_base}_{tf_suffix}"]
                            for p_base in pivot_cols_base if f"{p_base}_{tf_suffix}" in data.columns
                        }
                        # 确保所有预期的 level keys 都有对应的 Series
                        # 这里需要一个从 Pivot base name 到 internal key 的映射，例如 "PP" -> "PP" (假设), "R1" -> "R1"
                        # 简化处理，直接使用列名作为 internal key 传入评分函数（如果评分函数能处理）
                        # 或者构建一个 {internal_key: Series} 字典，internal_key 需要与评分函数期望的键一致
                        # 根据 File 2 的 calculate_pivot_score(..., pivot_levels: Dict[str, pd.Series], ...)，它期望键是字符串，值是 Series
                        # 这里直接使用列名前缀 (如 'PP', 'R1') 作为字典的键
                        pivot_levels_series_dict_for_score: Dict[str, pd.Series] = {}
                        all_pivot_levels_found_as_series = True
                        for p_base in pivot_cols_base:
                            col_name = f"{p_base}_{tf_suffix}"
                            if col_name in data.columns:
                                # Use the base name (PP, R1, S1 etc.) as the key for the scoring function
                                pivot_levels_series_dict_for_score[p_base] = data[col_name]
                            else:
                                all_pivot_levels_found_as_series = False
                                break # 缺少任何一个 Pivot level 列，就认为未找到
                        if all_pivot_levels_found_as_series and close_col in data.columns:
                            indicator_cols_for_score = {'close': close_col, 'pivot_levels': pivot_levels_series_dict_for_score}
                            found = True
                        # logger.debug(f"通过后缀匹配找到指标 'pivot' 在时间框架 {tf_score} 的列，使用后缀 {tf_suffix}") # 调试信息
                    break # Pivot处理完成后跳出后缀循环
                # 对于其他指标，查找符合模式和后缀的列
                potential_cols = [c for c in data.columns if any(c.startswith(prefix) for prefix in column_pattern_prefixes) and c.endswith(f"_{tf_suffix}")]
                if not potential_cols:
                    # logger.debug(f"在时间框架 {tf_score} 使用后缀 {tf_suffix} 未找到符合模式 {column_pattern_prefixes} 的列。") # 调试信息
                    continue # 尝试下一个后缀
                # 选择第一个找到的潜在列作为参考，尝试解析其参数
                reference_col = potential_cols[0]
                # logger.debug(f"找到参考列: {reference_col}，尝试解析参数...") # 调试信息
                # 使用 parse_col_params 助手函数解析参数
                params = parse_col_params(reference_col, indicator_key, tf_suffix)
                if params is None:
                    # logger.debug(f"从参考列 {reference_col} 解析参数失败，尝试下一个后缀。") # 调试信息
                    continue # 参数解析失败，尝试下一个后缀
                # 如果参数解析成功 (或指标无参数)，构建所有必需列的预期列名并检查是否存在
                expected_cols: Dict[str, str] = {}
                all_required_cols_found_with_params = True
                # 遍历必需的内部 key，构建对应的列名并检查
                for internal_key in required_score_keys:
                    # 特殊处理不需要通过 build_expected_col_name 构建的 key (如 Pivot levels)
                    if indicator_key == 'pivot' and internal_key == 'pivot_levels':
                        # Pivot levels 已在上方特殊处理
                        continue
                    # 特殊处理 OBV_MA，需要使用 OBV_MA 的周期参数，而不是基础指标的参数
                    if indicator_key == 'obv' and internal_key == 'obv_ma':
                        # OBV 评分函数签名是 calculate_obv_score(obv, obv_ma=None, obv_ma_period=None)
                        # OBV_MA Series 是一个可选参数，其列名需要 OBV_MA_period_tf 模式查找
                        # 即使 obv_ma 在 required_score_keys 里，这里也只尝试查找列名，并不会中断 found=False
                        # OBV_MA 列名需要 OBV_MA 的周期参数，从 bs_params 中获取
                        obv_ma_period = bs_params.get('obv_ma_period', defaults.get('obv_ma_period', 10))
                        obv_ma_params = [obv_ma_period] # OBV_MA 参数列表只包含其周期
                        expected_col_name = build_expected_col_name(indicator_key, internal_key, obv_ma_params, tf_suffix)
                    else:
                        # 其他指标使用解析出的参数 params
                        expected_col_name = build_expected_col_name(indicator_key, internal_key, params, tf_suffix)
                    if expected_col_name and expected_col_name in data.columns:
                        expected_cols[internal_key] = expected_col_name
                        # logger.debug(f"找到必需列: {expected_col_name} (内部 key: {internal_key})") # 调试信息
                    else:
                        # 如果是 OBV_MA 未找到，它不是评分必需的，不中断查找
                        if indicator_key == 'obv' and internal_key == 'obv_ma':
                            logger.debug(f"为指标 '{indicator_key}' 在时间框架 {tf_score} (后缀: {tf_suffix}) 未找到可选列 '{expected_col_name}' (内部 key: {internal_key})。") # MODIFIED: 增加日志
                        else:
                            # 其他必需列未找到，当前参数组和后缀不匹配
                            all_required_cols_found_with_params = False
                            # logger.debug(f"未找到必需列: {expected_col_name} (内部 key: {internal_key})") # 调试信息
                            break # 只要有一个必需列未找到，当前参数组和后缀就不匹配
                    # 如果所有必需列都找到了 (包括非 OBV_MA 的必需列)
                    if all_required_cols_found_with_params:
                        # 成功找到了一组具有相同参数和时间框架后缀的指标列
                        # 再次检查并添加 OBV_MA 列，如果它存在且评分函数需要
                        if indicator_key == 'obv' and 'obv_ma' in info['required_keys']: # 检查info['required_keys']看是否需要OBV_MA series
                            obv_ma_period = bs_params.get('obv_ma_period', defaults.get('obv_ma_period', 10))
                            obv_ma_params = [obv_ma_period]
                            expected_obv_ma_col_name = build_expected_col_name(indicator_key, 'obv_ma', obv_ma_params, tf_suffix)
                            if expected_obv_ma_col_name and expected_obv_ma_col_name in data.columns:
                                expected_cols['obv_ma'] = expected_obv_ma_col_name
                            else:
                                # 如果 OBV_MA 是必需的但没找到，则当前后缀查找失败
                                # Note: 根据File 2的OBV评分函数签名，obv_ma是Optional，即使没找到，评分函数也能运行
                                # 这里的逻辑需要与评分函数实际是否*必须*使用obv_ma保持一致。
                                # 假设如果配置了obv_ma_period且评分函数签名有obv_ma参数，就尝试找obv_ma列并传入
                                # 如果没找到，就传入 None 给 obv_ma 参数，评分函数内部处理 None
                                # 所以这里找到 OBV 主列就够了，OBV_MA 找不到不影响 found 状态
                                logger.debug(f"为指标 '{indicator_key}' 在时间框架 {tf_score} (后缀: {tf_suffix}) 未找到可选的 OBV_MA 列: {expected_obv_ma_col_name}") # MODIFIED: 增加日志
                        indicator_cols_for_score = expected_cols
                        found = True
                        # logger.debug(f"通过后缀匹配找到指标 '{indicator_key}' 在时间框架 {tf_score} 的所有必需列，使用后缀 {tf_suffix} 和参数 {params}") # 调试信息
                        break # 找到匹配项，跳出后缀循环
                # 在尝试所有后缀后，如果 still not found，打印错误信息
                if not found:
                    logger.warning(f"未能为指标 '{indicator_key}' 在时间框架 {tf_score} 找到所有必要的数据列进行评分。")
                    logger.info(f"尝试查找所需的内部键: {required_score_keys}. 尝试的后缀: {possible_tf_suffixes}.")
                    # 尝试列出该时间框架下匹配任何前缀的列
                    if indicator_key == 'pivot':
                        # 对于 Pivot，列出所有 Pivot 相关列
                        pivot_cols_base = ["PP", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4", "F_R1", "F_R2", "F_R3", "F_S1", "F_S2", "F_S3"]
                        relevant_cols_for_tf = [f"{col}_{tf_suffix}" for col in pivot_cols_base for tf_suffix in possible_tf_suffixes]
                        close_col_tf_list = [f"close_{tf_suffix}" for tf_suffix in possible_tf_suffixes]
                        relevant_cols_for_tf.extend(close_col_tf_list)
                        relevant_cols_for_tf = list(dict.fromkeys(relevant_cols_for_tf)) # 去重
                        relevant_cols_for_tf = [c for c in relevant_cols_for_tf if c in data.columns] # 只保留实际存在的列
                    else:
                        relevant_cols_for_tf = [c for c in data.columns if any(c.startswith(prefix) for prefix in column_pattern_prefixes) and any(c.endswith(f"_{s}") for s in possible_tf_suffixes)]
                        logger.info(f"DataFrame 中与该时间框架匹配的 '{indicator_key}' 相关列列表: {relevant_cols_for_tf}.")
                    # logger.debug(f"DataFrame 所有列列表: {list(data.columns)}.") # 打印所有列名，可能很长
            # --- 调用评分函数并存储结果 ---
            if found:
                try:
                    # 准备评分函数的参数字典
                    score_func_args: Dict[str, Any] = {}
                    # 传入指标数据 Series
                    for internal_key, actual_col_name in indicator_cols_for_score.items():
                        # MODIFIED: 特殊处理 pivot_levels，它在 indicator_cols_for_score 中已经是一个 {internal_key: Series} 字典
                        if indicator_key == 'pivot' and internal_key == 'pivot_levels' and isinstance(actual_col_name, dict):
                            # 直接将这个字典作为 'pivot_levels' 参数传入
                            score_func_args['pivot_levels'] = actual_col_name
                        elif isinstance(actual_col_name, str) and actual_col_name in data.columns:
                            # 将实际 Series 赋值给评分函数期望的参数名 (internal_key)
                            score_func_args[internal_key] = data[actual_col_name]
                        else:
                            # 理论上 found=True 应该保证列存在，这里是双重检查
                            # 对于 Pivot level 列表形式的 actual_col_name，这里不应该处理，应该在上面 special case 处理
                            # 这里的 else 应该只处理 str 类型的 actual_col_name 不存在的情况
                            if isinstance(actual_col_name, str):
                                logger.error(f"内部错误: 期望列 '{actual_col_name}' (内部 key: '{internal_key}') 在 DataFrame 中未找到，despite 'found' is True.")
                                raise KeyError(f"Missing column {actual_col_name}")
                            else:
                                # 如果 actual_col_name 不是 string 也不是 dict (for pivot_levels)，说明有问题
                                logger.error(f"内部错误: 期望的实际列名不是字符串或 pivot_levels 字典。internal_key: '{internal_key}', actual_col_name: {actual_col_name}")
                                raise ValueError(f"Invalid actual_col_name type for internal key {internal_key}")
                    # 传入评分逻辑参数 (根据 param_passing_style)
                    if param_passing_style == 'dict':
                        params_dict = {}
                        for bs_key, score_arg_name in bs_param_key_to_score_func_arg.items():
                            # 从 bs_params 获取值，如果不存在则使用 defaults 中的默认值
                            param_value = bs_params.get(bs_key, defaults.get(score_arg_name, None))
                            params_dict[score_arg_name] = param_value
                        score_func_args['params'] = params_dict # 将参数字典赋值给名为 'params' 的键

                    elif param_passing_style == 'individual':
                        for bs_key, score_arg_name in bs_param_key_to_score_func_arg.items():
                            # 从 bs_params 获取值，如果不存在则使用 defaults 中的默认值
                            param_value = bs_params.get(bs_key, defaults.get(score_arg_name, None))
                            score_func_args[score_arg_name] = param_value # 直接将参数作为关键字参数传入
                    # 特殊处理 Pivot，需要传入 tf 参数
                    if indicator_key == 'pivot':
                        score_func_args['tf'] = tf_score_str
                    # 调用评分函数
                    # score_func_args 此时包含所有必需的 Series (以其内部 key 为键) 和评分逻辑参数 (根据 style)
                    # 评分函数签名是 score_func(Series1, Series2, ..., param1=value1, param2=value2, ...) 或 score_func(Series1, Series2, ..., params=param_dict)
                    # 我们将 score_func_args 解包传入
                    scores: pd.Series = score_func(**score_func_args) # MODIFIED: 移除 positional 'data' argument
                    # 确保评分结果是 Series 且索引与输入 data 相同
                    if not isinstance(scores, pd.Series) or not scores.index.equals(data.index):
                        logger.error(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的评分函数返回结果格式不正确。")
                        scores = pd.Series(50.0, index=data.index) # 返回默认中性分
                    # 存储评分结果列
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str.replace('.', '_').replace('-', '_')}" # 格式化列名
                    scoring_results[score_col_name] = scores
                except Exception as e:
                    # 如果评分计算发生错误，记录错误并填充默认中性分
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str.replace('.', '_').replace('-', '_')}" # 格式化列名
                    scoring_results[score_col_name] = 50.0 # 发生错误时填充默认中性分
                    # 打印详细错误信息和传入的参数键名
                    arg_names = list(score_func_args.keys())
                    # MODIFIED: 尝试在日志中区分 Series 参数和普通参数，避免日志过长
                    series_args = {k: type(v) for k, v in score_func_args.items() if isinstance(v, (pd.Series, dict))} # dict for pivot_levels
                    other_args = {k: v for k, v in score_func_args.items() if not isinstance(v, (pd.Series, dict))}
                    logger.error(f"计算指标 '{indicator_key}' 在时间框架 {tf_score} 的评分时发生错误。"
                                f"传入 Series 参数: {series_args}. 其他参数: {other_args}. "
                                f"错误信息: {e}", exc_info=True)
            else:
                # 如果未找到必需的列，为该指标和时间框架添加一个填充默认中性分的列
                score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str.replace('.', '_').replace('-', '_')}" # 格式化列名
                scoring_results[score_col_name] = 50.0 # 列未找到时填充默认中性分

    # 填充最终结果中的 NaN 值为默认中性分 50.0
    scoring_results = scoring_results.fillna(50.0)

    logger.info("指标评分计算完成。")
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
        print("量能分数调整和量能分析均未启用。")
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
        # 修改点：移除 raw=True，确保 apply 接收 Series 并正确返回原始索引
        price_high_idx = high.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmax()) # REMOVED: raw=True
        # 修改点：移除 raw=True，确保 apply 接收 Series 并正确返回原始索引
        price_low_idx = low.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmin()) # REMOVED: raw=True

        # 寻找OBV的近期高点和低点索引
        # 修改点：移除 raw=True，确保 apply 接收 Series 并正确返回原始索引
        obv_high_idx = obv.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmax()) # REMOVED: raw=True
        # 修改点：移除 raw=True，确保 apply 接收 Series 并正确返回原始索引
        obv_low_idx = obv.rolling(window=vp_div_lookback, center=False).apply(lambda x: x.idxmin()) # REMOVED: raw=True


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
        print(f"VP背离检测跳过，因 lookback ({vp_div_lookback}) 不足或OBV数据无效。")


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



