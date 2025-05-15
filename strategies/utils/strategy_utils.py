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
from typing import Dict, Any, List, Optional, Tuple, Union

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
        check_divergence_pairs(price_trough_matches, is_peak=False, div_type='hidden_bullish') # MODIFIED: Corrected typo 'hidden_hidden_bullish' to 'hidden_bullish'
    return result_df.fillna(0).astype(int)

def detect_divergence(data: pd.DataFrame, dd_params: Dict, naming_config: Dict) -> pd.DataFrame:
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

    # 获取命名规范字典
    indicator_naming_conv = naming_config.get('indicator_naming_conventions', {})
    ohlcv_naming_conv = naming_config.get('ohlcv_naming_convention', {})
    timeframe_naming_conv = naming_config.get('timeframe_naming_convention', {}) # 新增：时间框架命名规范

    # 增加类型检查
    if not isinstance(indicator_naming_conv, dict): indicator_naming_conv = {}
    if not isinstance(ohlcv_naming_conv, dict): ohlcv_naming_conv = {}
    if not isinstance(timeframe_naming_conv, dict): timeframe_naming_conv = {}

    # 获取价格列模式
    price_pattern = None
    ohlcv_output_cols_conf = ohlcv_naming_conv.get('output_columns', [])
    if isinstance(ohlcv_output_cols_conf, list):
        for col_conf in ohlcv_output_cols_conf:
            if isinstance(col_conf, dict) and col_conf.get('name_pattern') == price_type:
                price_pattern = col_conf.get('name_pattern')
                break
    if not price_pattern:
        logger.error(f"命名规范中未找到价格类型 '{price_type}' 的模式。无法进行背离检测。")
        return all_divergence_signals

    # 遍历需要检测的指标键
    for indicator_key, enabled in indicators_to_check.items():
        if not enabled:
            continue

        # 获取该指标的命名配置
        indi_naming_conf = indicator_naming_conv.get(indicator_key.upper()) # 使用大写键查找
        if not isinstance(indi_naming_conf, dict):
            logger.warning(f"指标 '{indicator_key}' 在命名规范中未找到或配置无效，跳过背离检测。")
            continue

        # 获取该指标的主要输出列模式 (用于背离检测)
        # 假设 output_columns 列表中的第一个模式是主要模式，或者根据 internal_key 查找
        # 这里需要知道哪个 internal_key 对应背离检测使用的 Series (例如 RSI -> 'rsi', MACD -> 'macd_h')
        # 这个映射关系应该在 dd_params 或 indicator_scoring_info 中定义
        # 暂时硬编码一些常见映射，或者假设 dd_params['indicators'] 的键就是 internal_key
        # 假设 dd_params['indicators'] 的键就是 internal_key
        internal_key_for_div = indicator_key.lower() # 假设 dd_params 中的键是小写 internal_key

        indicator_pattern = None
        output_cols_patterns = indi_naming_conf.get('output_columns', [])
        if isinstance(output_cols_patterns, list):
            for col_conf in output_cols_patterns:
                if isinstance(col_conf, dict) and col_conf.get('name_pattern', '').lower().startswith(internal_key_for_div): # 尝试匹配模式开头
                     indicator_pattern = col_conf.get('name_pattern')
                     break
            # 特殊处理一些指标，如 KDJ, STOCH, DMI，它们有多个输出列，需要指定用哪个
            # 例如 KDJ 通常用 J 线，STOCH 用 K 或 D 线，DMI 用 ADX 或 PDI/NDI
            # 假设 dd_params['indicators'] 的键已经指定了具体用哪个线 (如 'macd_hist', 'stoch_k', 'kdj_j')
            # 那么 internal_key_for_div 可能是 'macd_hist', 'stoch_k', 'kdj_j' 等
            # 需要根据这个 internal_key_for_div 找到对应的模式
            if indicator_key.lower() == 'macd_hist': internal_key_for_div = 'macdh'
            elif indicator_key.lower() == 'stoch_k': internal_key_for_div = 'stochk'
            elif indicator_key.lower() == 'stoch_d': internal_key_for_div = 'stochd'
            elif indicator_key.lower() == 'kdj_j': internal_key_for_div = 'j'
            elif indicator_key.lower() == 'dmi_adx': internal_key_for_div = 'adx'
            elif indicator_key.lower() == 'dmi_pdi': internal_key_for_div = 'pdi'
            elif indicator_key.lower() == 'dmi_ndi': internal_key_for_div = 'ndi'
            # 再次尝试根据修正后的 internal_key_for_div 查找模式
            if not indicator_pattern and isinstance(output_cols_patterns, list):
                 for col_conf in output_cols_patterns:
                     if isinstance(col_conf, dict) and col_conf.get('name_pattern', '').lower().startswith(internal_key_for_div):
                          indicator_pattern = col_conf.get('name_pattern')
                          break
            # 特殊处理 OBV，它没有参数
            if indicator_key.lower() == 'obv':
                 indicator_pattern = 'OBV' # 硬编码 OBV 模式

        if not indicator_pattern:
            logger.warning(f"指标 '{indicator_key}' 在命名规范中未找到主要输出列模式，跳过背离检测。")
            continue

        # 获取该指标的参数信息 (从 indicator_scoring_info 或 dd_params)
        # 这是一个简化的参数获取，可能需要更精确的匹配逻辑
        # 尝试从 indicator_scoring_info 获取默认参数
        scoring_info = indicator_scoring_info.get(indicator_key.lower())
        indicator_params = {}
        if scoring_info and isinstance(scoring_info.get('defaults'), dict):
             indicator_params = scoring_info['defaults'].copy()
             # 尝试从 dd_params 中覆盖默认参数 (如果 dd_params 中有针对该指标的参数配置块)
             # 假设 dd_params 中有类似 {'rsi_params': {'period': 20}} 的结构
             indi_params_from_dd = dd_params.get(f'{indicator_key.lower()}_params', {})
             if isinstance(indi_params_from_dd, dict):
                  indicator_params.update(indi_params_from_dd)
        # 特殊处理 DMI，其周期参数在命名规范中是 'period'
        if indicator_key.lower() in ['dmi', 'dmi_adx', 'dmi_pdi', 'dmi_ndi']:
             dmi_period = dd_params.get('dmi_period', indicator_params.get('period', 14))
             indicator_params['period'] = dmi_period
        # 特殊处理 MACD，其参数在命名规范中是 'period_fast', 'period_slow', 'signal_period'
        if indicator_key.lower() in ['macd', 'macd_hist']:
             macd_params_dd = dd_params.get('macd_params', {})
             indicator_params['period_fast'] = macd_params_dd.get('period_fast', indicator_params.get('period_fast', 12))
             indicator_params['period_slow'] = macd_params_dd.get('period_slow', indicator_params.get('period_slow', 26))
             indicator_params['signal_period'] = macd_params_dd.get('signal_period', indicator_params.get('signal_period', 9))
        # 特殊处理 KDJ，其参数在命名规范中是 'period', 'signal_period', 'smooth_k_period'
        if indicator_key.lower() in ['kdj', 'kdj_j']:
             kdj_params_dd = dd_params.get('kdj_params', {})
             indicator_params['period'] = kdj_params_dd.get('period', indicator_params.get('period', 9))
             indicator_params['signal_period'] = kdj_params_dd.get('signal_period', indicator_params.get('signal_period', 3))
             indicator_params['smooth_k_period'] = kdj_params_dd.get('smooth_k_period', indicator_params.get('smooth_k_period', 3))
        # 特殊处理 STOCH，其参数在命名规范中是 'k_period', 'd_period', 'smooth_k_period'
        if indicator_key.lower() in ['stoch', 'stoch_k', 'stoch_d']:
             stoch_params_dd = dd_params.get('stoch_params', {})
             indicator_params['k_period'] = stoch_params_dd.get('k_period', indicator_params.get('k_period', 14))
             indicator_params['d_period'] = stoch_params_dd.get('d_period', indicator_params.get('d_period', 3))
             indicator_params['smooth_k_period'] = stoch_params_dd.get('smooth_k_period', indicator_params.get('smooth_k_period', 3))
        # 特殊处理 BOLL，其参数在命名规范中是 'period', 'std_dev'
        if indicator_key.lower() == 'boll':
             boll_params_dd = dd_params.get('boll_params', {})
             indicator_params['period'] = boll_params_dd.get('period', indicator_params.get('period', 20))
             indicator_params['std_dev'] = boll_params_dd.get('std_dev', indicator_params.get('std_dev', 2.0))
        # 特殊处理 SAR，其参数在命名规范中是 'af_step', 'max_af'
        if indicator_key.lower() == 'sar':
             sar_params_dd = dd_params.get('sar_params', {})
             indicator_params['af_step'] = sar_params_dd.get('af_step', indicator_params.get('af_step', 0.02))
             indicator_params['max_af'] = sar_params_dd.get('max_af', indicator_params.get('max_af', 0.2))
        # 特殊处理 VWAP，其参数在命名规范中是 'anchor'
        if indicator_key.lower() == 'vwap':
             vwap_params_dd = dd_params.get('vwap_params', {})
             indicator_params['anchor'] = vwap_params_dd.get('anchor', indicator_params.get('anchor', None))
        # 特殊处理 Ichimoku，其参数在命名规范中是 'tenkan_period', 'kijun_period', 'senkou_period'
        if indicator_key.lower() == 'ichimoku':
             ichimoku_params_dd = dd_params.get('ichimoku_params', {})
             indicator_params['tenkan_period'] = ichimoku_params_dd.get('tenkan_period', indicator_params.get('tenkan_period', 9))
             indicator_params['kijun_period'] = ichimoku_params_dd.get('kijun_period', indicator_params.get('kijun_period', 26))
             indicator_params['senkou_period'] = ichimoku_params_dd.get('senkou_period', indicator_params.get('senkou_period', 52))
        # 特殊处理 KC，其参数在命名规范中是 'ema_period', 'atr_period'
        if indicator_key.lower() == 'kc':
             kc_params_dd = dd_params.get('kc_params', {})
             indicator_params['ema_period'] = kc_params_dd.get('ema_period', indicator_params.get('ema_period', 20))
             indicator_params['atr_period'] = kc_params_dd.get('atr_period', indicator_params.get('atr_period', 10))
        # 特殊处理 OBV_MA，其参数在命名规范中是 'period'
        if indicator_key.lower() == 'obv_ma':
             obv_ma_params_dd = dd_params.get('obv_ma_params', {})
             indicator_params['period'] = obv_ma_params_dd.get('period', indicator_params.get('period', 10))


        # 遍历需要检测的时间框架
        for tf_check in timeframes_to_check:
            tf_check_str = str(tf_check)
            # 获取该时间框架可能的后缀列表
            possible_tf_suffixes_raw = timeframe_naming_conv.get('patterns', {}).get(tf_check_str.lower(), [tf_check_str])
            if not isinstance(possible_tf_suffixes_raw, list): possible_tf_suffixes = [str(possible_tf_suffixes_raw)]
            else: possible_tf_suffixes = [str(s) for s in possible_tf_suffixes_raw] # 确保是字符串
            if tf_check_str in possible_tf_suffixes: possible_tf_suffixes.remove(tf_check_str); possible_tf_suffixes.insert(0, tf_check_str)
            elif tf_check_str.upper() in possible_tf_suffixes: possible_tf_suffixes.remove(tf_check_str.upper()); possible_tf_suffixes.insert(0, tf_check_str.upper())
            else: possible_tf_suffixes.insert(0, tf_check_str)
            seen = set(); possible_tf_suffixes_unique = [];
            for suffix in possible_tf_suffixes:
                if suffix not in seen: seen.add(suffix); possible_tf_suffixes_unique.append(suffix)
            possible_tf_suffixes = possible_tf_suffixes_unique


            # 查找价格列
            price_col = None
            for suffix in possible_tf_suffixes:
                 potential_price_col = f'{price_pattern}_{suffix}'
                 if potential_price_col in data.columns:
                      price_col = potential_price_col
                      break
            if price_col is None or data[price_col].isnull().all():
                logger.warning(f"价格列 '{price_type}' 在 TF {tf_check} ({possible_tf_suffixes}) 未找到或全为 NaN。跳过。")
                continue
            price_series = data[price_col]

            # 查找指标列
            indicator_col = None
            # 尝试根据模式和参数构建列名并查找
            try:
                 # 格式化参数，需要根据 indicator_key 特殊处理浮点数格式
                 def format_param_for_div(p, key):
                     if isinstance(p, float):
                         if key == 'boll': return f"{p:.1f}"
                         if key == 'sar':
                              # SAR 参数格式化 SAR_0.02_0.2
                              if 'af_step' in indicator_params and p == indicator_params['af_step']: return f"{p:.2f}"
                              if 'max_af' in indicator_params and p == indicator_params['max_af']: return f"{p:.1f}"
                              return f"{p:.2f}" # Default float format
                         return str(p)
                     return str(p)

                 # 构建参数字符串部分
                 param_str_parts = []
                 # 根据 indicator_key 和 internal_key_for_div 确定参数顺序和键名
                 # 这是一个简化的映射，可能需要更精确的配置
                 if indicator_key.lower() in ['macd', 'macd_hist'] and 'period_fast' in indicator_params and 'period_slow' in indicator_params and 'signal_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period_fast'], indicator_key.lower()), format_param_for_div(indicator_params['period_slow'], indicator_key.lower()), format_param_for_div(indicator_params['signal_period'], indicator_key.lower())]
                 elif indicator_key.lower() in ['rsi', 'cci', 'mfi', 'roc', 'atr', 'mom', 'willr', 'vroc', 'aroc'] and 'period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], indicator_key.lower())]
                 elif indicator_key.lower() in ['kdj', 'kdj_j'] and 'period' in indicator_params and 'signal_period' in indicator_params and 'smooth_k_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], indicator_key.lower()), format_param_for_div(indicator_params['signal_period'], indicator_key.lower()), format_param_for_div(indicator_params['smooth_k_period'], indicator_key.lower())]
                 elif indicator_key.lower() in ['stoch', 'stoch_k', 'stoch_d'] and 'k_period' in indicator_params and 'd_period' in indicator_params and 'smooth_k_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['k_period'], indicator_key.lower()), format_param_for_div(indicator_params['d_period'], indicator_key.lower()), format_param_for_div(indicator_params['smooth_k_period'], indicator_key.lower())]
                 elif indicator_key.lower() in ['dmi', 'dmi_adx', 'dmi_pdi', 'dmi_ndi'] and 'period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], indicator_key.lower())]
                 elif indicator_key.lower() == 'boll' and 'period' in indicator_params and 'std_dev' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], indicator_key.lower()), format_param_for_div(indicator_params['std_dev'], indicator_key.lower())]
                 elif indicator_key.lower() == 'sar' and 'af_step' in indicator_params and 'max_af' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['af_step'], indicator_key.lower()), format_param_for_div(indicator_params['max_af'], indicator_key.lower())]
                 elif indicator_key.lower() == 'vwap' and 'anchor' in indicator_params and indicator_params['anchor'] is not None:
                      param_str_parts = [str(indicator_params['anchor'])]
                 elif indicator_key.lower() == 'ichimoku' and 'tenkan_period' in indicator_params and 'kijun_period' in indicator_params and 'senkou_period' in indicator_params:
                      # Ichimoku patterns are complex, need to match specific line patterns
                      # This simplified approach might not work well for Ichimoku
                      pass # Skip building param_str_parts for Ichimoku here
                 elif indicator_key.lower() == 'kc' and 'ema_period' in indicator_params and 'atr_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['ema_period'], indicator_key.lower()), format_param_for_div(indicator_params['atr_period'], indicator_key.lower())]
                 elif indicator_key.lower() == 'obv_ma' and 'period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], indicator_key.lower())]


                 param_part = '_'.join(param_str_parts) if param_str_parts else ""
                 param_suffix = f"_{param_part}" if param_part else ""

                 # Try to build the expected column name based on the pattern and parameters
                 expected_col_name = None
                 if indicator_key.lower() == 'obv': # OBV pattern is just 'OBV'
                      expected_col_name = f"OBV_{tf_check_str}" # OBV_tfSuffix
                 elif indicator_key.lower() == 'vwap': # VWAP patterns are 'VWAP' or 'VWAP_{anchor}'
                      if 'anchor' in indicator_params and indicator_params['anchor'] is not None:
                           expected_col_name = f"VWAP_{indicator_params['anchor']}_{tf_check_str}" # VWAP_anchor_tfSuffix
                      else:
                           expected_col_name = f"VWAP_{tf_check_str}" # VWAP_tfSuffix
                 elif indicator_key.lower() == 'ichimoku':
                      # Ichimoku needs specific line patterns
                      # This simplified lookup won't work well. Need to iterate through output_cols_patterns
                      # and match based on internal_key_for_div
                      if isinstance(output_cols_patterns, list):
                           for col_conf in output_cols_patterns:
                                if isinstance(col_conf, dict) and col_conf.get('name_pattern', '').lower().startswith(internal_key_for_div):
                                     pattern = col_conf.get('name_pattern')
                                     # Format Ichimoku patterns with their specific parameters
                                     if pattern:
                                          try:
                                               # This requires knowing which parameters map to which placeholders in the pattern
                                               # Example: SENKOU_A_{tenkan_period}_{kijun_period}
                                               # Need to map indicator_params keys to pattern placeholders
                                               # This is complex and error-prone without a clear mapping in naming_config
                                               # Let's skip Ichimoku for this simplified lookup and rely on the more robust lookup in calculate_all_indicator_scores
                                               pass # Skip Ichimoku for now
                                          except Exception as e:
                                               logger.warning(f"格式化 Ichimoku 模式 '{pattern}' 出错: {e}")
                                     break # Found a pattern, stop searching
                      pass # Ichimoku lookup skipped

                 else: # Most other indicators follow BASE_NAME_PARAMS_TF pattern
                      if indicator_pattern:
                           # Format the pattern with the parameter string part and time frame suffix
                           expected_col_name = f"{indicator_pattern}{param_suffix}_{tf_check_str}"
                           expected_col_name = expected_col_name.replace('__', '_').strip('_') # Clean up double underscores or leading/trailing underscores

                 # Check if the constructed column name exists
                 if expected_col_name and expected_col_name in data.columns:
                      indicator_col = expected_col_name
                      break # Found the column, break suffix loop

            except Exception as e:
                 logger.warning(f"尝试构建或查找指标 '{indicator_key}' (内部键: '{internal_key_for_div}') 在 TF {tf_check} 的列名时出错: {e}")
                 pass # Continue trying other suffixes or methods

            # Fallback: If building/matching pattern failed, try a simpler prefix/suffix match
            if indicator_col is None:
                 # Get possible prefixes for this indicator from indicator_scoring_info
                 prefixes = indicator_scoring_info.get(indicator_key.lower(), {}).get('prefixes', [])
                 # Also check prefixes from naming_config if available
                 indi_naming_prefixes = indi_naming_conf.get('prefixes', {}) # This might be a dict {internal_key: prefix}
                 if isinstance(indi_naming_prefixes, dict):
                      # Add prefixes for the relevant internal key
                      prefix_from_config = indi_naming_prefixes.get(internal_key_for_div)
                      if prefix_from_config and prefix_from_config not in prefixes:
                           prefixes.append(prefix_from_config)
                 elif isinstance(indi_naming_prefixes, list): # Some configs might list prefixes directly
                      for p in indi_naming_prefixes:
                           if p not in prefixes: prefixes.append(p)

                 # Filter columns by prefix and suffix
                 potential_cols = [c for c in data.columns if any(c.startswith(p) for p in prefixes) and c.endswith(f"_{tf_check_str}")]
                 if potential_cols:
                      # Choose the column that seems most likely (e.g., longest, or first)
                      # This is heuristic and might pick the wrong column if multiple exist with similar names
                      indicator_col = max(potential_cols, key=len) # Choose longest, assuming more params
                      logger.debug(f"Fallback lookup for '{indicator_key}' in TF {tf_check} found potential column: '{indicator_col}'")


            if indicator_col is None or data[indicator_col].isnull().all():
                logger.warning(f"指标 '{indicator_key}' (内部键: '{internal_key_for_div}') 在 TF {tf_check} 的列 '{indicator_col}' 未找到、全为 NaN 或未启用。跳过。")
                continue
            indicator_series = data[indicator_col]

            current_find_peaks_params = get_find_peaks_params(tf_check_str, safe_lookback) # Use safe_lookback
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
                # This parsing is heuristic and might fail for complex names
                parts = indicator_col.split('_')
                # Try to get the base indicator name part (e.g., RSI, MACDh)
                indi_name_part = parts[0]
                # Try to get the parameters part (e.g., 14, 12_26_9)
                params_part = "_".join(parts[1:-1]) if len(parts) > 2 else "" # Exclude TF suffix

                # Clean up div_type_col_name, e.g., 'regular_bullish' -> 'RegularBullish'
                clean_div_type = "".join(word.capitalize() for word in div_type_col_name.split('_'))

                # Construct the detailed column name: DIV_INDICATORNAME_PARAMS_TF_DIVTYPE
                # Example: DIV_RSI_14_D_RegularBullish
                detailed_col_name = f'DIV_{indi_name_part}'
                if params_part:
                     detailed_col_name += f'_{params_part}'
                detailed_col_name += f'_{tf_check_str}_{clean_div_type}'

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

    # Ensure聚合列也是 bool 类型
    all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = all_divergence_signals['HAS_BULLISH_DIVERGENCE'].astype(bool)
    all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = all_divergence_signals['HAS_BEARISH_DIVERGENCE'].astype(bool)


    logger.info(f"背离检测完成。共生成 {len(all_divergence_signals.columns) - 2} 个详细信号列。")
    return all_divergence_signals

def detect_kline_patterns(df: pd.DataFrame, tf: str, naming_config: Dict) -> pd.DataFrame: # MODIFIED: Added naming_config parameter
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
    :param naming_config: 包含列命名规范的字典。 # 新增参数说明
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
    # MODIFIED: Use naming_config to get pattern prefix if available, otherwise use default 'KAP_'
    kline_pattern_prefix = naming_config.get('kline_pattern_naming_convention', {}).get('prefix', 'KAP_') # 新增：从命名规范获取K线形态前缀
    default_pattern_df = pd.DataFrame(0, index=df.index, columns=[f'{kline_pattern_prefix}{name.upper()}_{tf}' for name in pattern_names]) # 使用获取到的前缀

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
    patterns_calc.loc[is_doji, f'{kline_pattern_prefix}DOJI_{tf}'] = np.where(is_green[is_doji], 1, -1) # 使用获取到的前缀

    # 2. 吞没形态 (Engulfing)
    # 看涨吞没: 前阴，今阳，今日阳线实体完全吞没昨日阴线实体
    bull_engulf = is_red1 & is_green & (c > o1) & (o < c1) & (body > body1 * 1.01) # 今日实体比昨日实体大1%以上
    patterns_calc.loc[bull_engulf, f'{kline_pattern_prefix}BULLISHENGULFING_{tf}'] = 1 # 使用获取到的前缀
    # 看跌吞没: 前阳，今阴，今日阴线实体完全吞没昨日阳线实体
    bear_engulf = is_green1 & is_red & (o > c1) & (c < o1) & (body > body1 * 1.01)
    patterns_calc.loc[bear_engulf, f'{kline_pattern_prefix}BEARISHENGULFING_{tf}'] = -1 # 使用获取到的前缀

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
    patterns_calc.loc[is_hammer, f'{kline_pattern_prefix}HAMMER_{tf}'] = 1 # 使用获取到的前缀
    # 上吊线：出现在上涨趋势后（简化为前一根K线是阳线）
    is_hanging = hammer_like_cond & is_green1
    patterns_calc.loc[is_hanging, f'{kline_pattern_prefix}HANGINGMAN_{tf}'] = -1 # 使用获取到的前缀

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
    patterns_calc.loc[morning_star, f'{kline_pattern_prefix}MORNINGSTAR_{tf}'] = 1 # 使用获取到的前缀

    # 黄昏之星 (看跌反转)
    # 条件: 前2日大阳线, 前1日星线(与前2日阳线向上跳空), 当日大阴线(与前1日星线向下跳空，收盘深入前2日阳线实体过半)
    gap1_up_evening = (l1 > h2) | ((np.minimum(o1,c1) > np.maximum(o2,c2)) & (np.maximum(o1,c1) > np.maximum(o2,c2))) # 星线与前大阳向上跳空
    gap2_down_evening = (h < l1) | ((np.maximum(o,c) < np.minimum(o1,c1)) & (np.minimum(o,c) < np.minimum(o1,c1)))     # 当日阴线与星线向下跳空
    evening_star = is_green2 & (body2 > avg_body) & \
                   is_star1 & \
                   is_red & (body > body2 * 0.5) & (c < (o2 + c2) / 2) & \
                   gap1_up_evening & gap2_down_evening
    patterns_calc.loc[evening_star, f'{kline_pattern_prefix}EVENINGSTAR_{tf}'] = -1 # 使用获取到的前缀

    # 5. 刺透形态 (Piercing Line) / 乌云盖顶 (Dark Cloud Cover) - 两K线组合
    # 刺透线 (看涨反转): 前大阴，当日阳线低开（低于前阴最低价），收盘深入前阴实体一半以上但未完全吞没。
    piercing = is_red1 & (body1 > avg_body) & is_green & (o < l1) & (c > (o1 + c1) / 2) & (c < o1)
    patterns_calc.loc[piercing, f'{kline_pattern_prefix}PIERCINGLINE_{tf}'] = 1 # 使用获取到的前缀
    # 乌云盖顶 (看跌反转): 前大阳，当日阴线高开（高于前阳最高价），收盘深入前阳实体一半以上但未完全吞没。
    dark_cloud = is_green1 & (body1 > avg_body) & is_red & (o > h1) & (c < (o1 + c1) / 2) & (c > o1)
    patterns_calc.loc[dark_cloud, f'{kline_pattern_prefix}DARKCLOUDCOVER_{tf}'] = -1 # 使用获取到的前缀

    # 6. 三白兵 (Three White Soldiers) / 三只乌鸦 (Three Black Crows) - 三K线组合
    min_body_ratio_for_soldier_crow = 0.5 # 三兵/三鸦中每根K线实体至少是平均实体的0.5倍
    # 三白兵 (看涨持续): 连续三根阳线，每日开盘在前一日实体内，收盘高于前一日收盘，实体逐渐放大或相近。
    soldiers = is_green2 & (body2 > avg_body * min_body_ratio_for_soldier_crow) & \
               is_green1 & (body1 > avg_body * min_body_ratio_for_soldier_crow) & (c1 > c2) & (o1 < c2) & (o1 > o2) & \
               is_green & (body > avg_body * min_body_ratio_for_soldier_crow) & (c > c1) & (o < c1) & (o > o1)
    patterns_calc.loc[soldiers, f'{kline_pattern_prefix}THREEWHITESOLDIERS_{tf}'] = 1 # 使用获取到的前缀
    # 三只乌鸦 (看跌持续): 连续三根阴线，每日开盘在前一日实体内，收盘低于前一日收盘，实体逐渐放大或相近。
    crows = is_red2 & (body2 > avg_body * min_body_ratio_for_soldier_crow) & \
            is_red1 & (body1 > avg_body * min_body_ratio_for_soldier_crow) & (c1 < c2) & (o1 > c2) & (o1 < o2) & \
            is_red & (body > avg_body * min_body_ratio_for_soldier_crow) & (c < c1) & (o > c1) & (o < o1)
    patterns_calc.loc[crows, f'{kline_pattern_prefix}THREEBLACKCROWS_{tf}'] = -1 # 使用获取到的前缀

    # 7. 孕线形态 (Harami) - 两K线组合
    # 定义：第二根K线的实体完全被第一根K线的实体所包含。
    is_harami_body = (np.maximum(o, c) < np.maximum(o1, c1)) & (np.minimum(o, c) > np.minimum(o1, c1))
    # 看涨孕线: 前大阴，后小阳线或十字星被包含。
    bullish_harami = is_red1 & (body1 > avg_body) & is_harami_body & (is_green | is_doji) # is_doji 是当日的十字星判断
    patterns_calc.loc[bullish_harami, f'{kline_pattern_prefix}BULLISHHARAMI_{tf}'] = 1 # 使用获取到的前缀
    # 看跌孕线: 前大阳，后小阴线或十字星被包含。
    bearish_harami = is_green1 & (body1 > avg_body) & is_harami_body & (is_red | is_doji)
    patterns_calc.loc[bearish_harami, f'{kline_pattern_prefix}BEARISHHARAMI_{tf}'] = -1 # 使用获取到的前缀

    # 十字孕线 (Harami Cross): 孕线形态的第二根K线是十字星。
    bullish_harami_cross = is_red1 & (body1 > avg_body) & is_harami_body & is_doji
    patterns_calc.loc[bullish_harami_cross, f'{kline_pattern_prefix}BULLISHHARAMICROSS_{tf}'] = 1 # 使用获取到的前缀
    bearish_harami_cross = is_green1 & (body1 > avg_body) & is_harami_body & is_doji
    patterns_calc.loc[bearish_harami_cross, f'{kline_pattern_prefix}BEARISHHARAMICROSS_{tf}'] = -1 # 使用获取到的前缀

    # 8. 镊子顶 (Tweezer Top) / 镊子底 (Tweezer Bottom) - 两K线组合
    # 定义：连续两根K线的最高价（镊子顶）或最低价（镊子底）几乎相同。
    tweezer_tolerance_ratio_of_range = 0.02 # 价格相同的容忍度：平均波幅的2%
    # 镊子底: 两根K线最低点相近，通常前一根为阴线。
    tweezer_bottom = (abs(l - l1) < avg_range * tweezer_tolerance_ratio_of_range) # is_red1 可作为可选增强条件
    patterns_calc.loc[tweezer_bottom, f'{kline_pattern_prefix}TWEEZERBOTTOM_{tf}'] = 1 # 使用获取到的前缀
    # 镊子顶: 两根K线最高点相近，通常前一根为阳线。
    tweezer_top = (abs(h - h1) < avg_range * tweezer_tolerance_ratio_of_range) # is_green1 可作为可选增强条件
    patterns_calc.loc[tweezer_top, f'{kline_pattern_prefix}TWEEZERTOP_{tf}'] = -1 # 使用获取到的前缀

    # 9. 光头光脚K线 (Marubozu) - 单K线形态
    # 定义：实体非常饱满，几乎没有上下影线，实体占整个K线范围的比例非常高，且实体本身也较大。
    marubozu_body_min_ratio_of_range = 0.95    # 实体至少占全天波幅的95%
    marubozu_body_min_ratio_of_avg_body = 1.5 # 实体至少是平均实体的1.5倍
    is_marubozu_cond = (body >= full_range * marubozu_body_min_ratio_of_range) & \
                       (body > avg_body * marubozu_body_min_ratio_of_avg_body)
    # 看涨光头光脚 (大阳线)
    bull_marubozu = is_marubozu_cond & is_green
    patterns_calc.loc[bull_marubozu, f'{kline_pattern_prefix}BULLISHMARUBOZU_{tf}'] = 1 # 使用获取到的前缀
    # 看跌光头光脚 (大阴线)
    bear_marubozu = is_marubozu_cond & is_red
    patterns_calc.loc[bear_marubozu, f'{kline_pattern_prefix}BEARISHMARUBOZU_{tf}'] = -1 # 使用获取到的前缀

    # --- 多K线组合形态 (需要更多历史数据) ---
    # 10. 上升三法 (Rising Three Methods) / 下降三法 (Falling Three Methods) - 五K线组合
    if len(df_calc_idx) >= 5: # 确保有足够数据进行5根K线的判断
        consolidation_body_max_ratio_of_avg_body = 0.5 # 中间整理K线的实体最大为平均实体的0.5倍

        # 上升三法 (看涨持续)
        # 条件: 前4日长阳, 中间三日小K线(阴或阳实体小)整理且在前4日长阳实体内, 当日长阳收盘高于前4日收盘。
        rising_three = is_green4 & (body4 > avg_body * 1.5) & \
                       (is_red3 | (body3 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h3 < h4) & (l3 > l4) & \
                       (is_red2 | (body2 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h2 < h4) & (l2 > l4) & \
                       is_red1 & (body1 < avg_body * consolidation_body_max_ratio_of_avg_body) & (h1 < h4) & (l1 > l4) & \
                       is_green & (body > avg_body * 1.5) & (c > c4) & (o > l1) # 当日阳线开盘高于前一整理K线低点 # MODIFIED: Corrected is_red1 condition
        patterns_calc.loc[rising_three, f'{kline_pattern_prefix}RISINGTHREEMETHODS_{tf}'] = 1 # 使用获取到的前缀

        # 下降三法 (Falling Three Methods) - 看跌持续
        # 条件: 前4日长阴, 中间三日小K线整理且在前4日长阴实体内, 当日长阴收盘低于前4日收盘。
        falling_three = is_red4 & (body4 > avg_body * 1.5) & \
                        (is_green3 | (body3 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h3 < h4) & (l3 > l4) & \
                        (is_green2 | (body2 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h2 < h4) & (l2 > l4) & \
                        is_green1 & (body1 < avg_body * consolidation_body_max_ratio_of_avg_body) & (h1 < h4) & (l1 > l4) & \
                        is_red & (body > avg_body * 1.5) & (c < c4) & (o < h1) # 当日阴线开盘低于前一整理K线高点 # MODIFIED: Corrected is_green1 condition
        patterns_calc.loc[falling_three, f'{kline_pattern_prefix}FALLINGTHREEMETHODS_{tf}'] = -1 # 使用获取到的前缀

    # 11. 其他三K线组合形态
    if len(df_calc_idx) >= 3: # 确保有足够数据
        # 向上跳空两只乌鸦 (Upside Gap Two Crows) - 看跌反转
        # 条件: 前2日强阳, 前1日向上跳空收小阴, 当日再收阴线开盘高于前阴实体但收盘低于前阴实体，形成吞噬。
        upside_gap_two_crows = is_green2 & (body2 > avg_body) & \
                              is_red1 & (body1 < avg_body * 0.7) & (o1 > h2) & \
                              is_red & (o > o1) & (c < c1) & (c < h2) # 当日阴线实体吞没前小阴，收盘低于第一天高点
        patterns_calc.loc[upside_gap_two_crows, f'{kline_pattern_prefix}UPSIDEGAPTWOCROWS_{tf}'] = -1 # 使用获取到的前缀

        # 跳空并列线 (Tasuki Gap) - 持续形态
        # 向上跳空并列阳线 (Upside Tasuki Gap) - 看涨持续
        # 条件: 前2日阳线, 前1日向上跳空收阳, 当日阴线开盘于前1日阳线实体内，收盘于缺口之内但未完全封闭缺口。
        upside_tasuki_gap = is_green2 & \
                           is_green1 & (o1 > h2) & \
                           is_red & (o > o1) & (o < c1) & (c < o1) & (c > h2)
        patterns_calc.loc[upside_tasuki_gap, f'{kline_pattern_prefix}UPSIDETASUKIGAP_{tf}'] = 1 # 使用获取到的前缀

        # 向下跳空并列阴线 (Downside Tasuki Gap) - 看跌持续
        # 条件: 前2日阴线, 前1日向下跳空收阴, 当日阳线开盘于前1日阴线实体内，收盘于缺口之内但未完全封闭缺口。
        downside_tasuki_gap = is_red2 & \
                             is_red1 & (o1 < l2) & \
                             is_green & (o < o1) & (o > c1) & (c > o1) & (c < l2)
        patterns_calc.loc[downside_tasuki_gap, f'{kline_pattern_prefix}DOWNSIDETASUKIGAP_{tf}'] = -1 # 使用获取到的前缀

    # 12. 其他两K线组合形态
    if len(df_calc_idx) >= 2: # 确保有足够数据
        # 分离线 (Separating Lines) - 持续形态
        # 条件：两根K线颜色相反，开盘价相同。
        same_open_cond = abs(o - o1) < avg_range * 0.02 # 开盘价几乎相同
        # 看涨分离线: 下降趋势中（前阴），前阴后阳，开盘相同。
        bullish_sep_lines = is_red1 & is_green & same_open_cond
        patterns_calc.loc[bullish_sep_lines, f'{kline_pattern_prefix}BULLISHSEPARATINGLINES_{tf}'] = 1 # 使用获取到的前缀
        # 看跌分离线: 上升趋势中（前阳），前阳后阴，开盘相同。
        bearish_sep_lines = is_green1 & is_red & same_open_cond
        patterns_calc.loc[bearish_sep_lines, f'{kline_pattern_prefix}BEARISHSEPARATINGLINES_{tf}'] = -1 # 使用获取到的前缀

        # 反击线 (Counterattack Lines) - 反转形态
        # 条件：两根K线颜色相反，收盘价相同，且第二根K线大幅跳空开盘。
        same_close_cond = abs(c - c1) < avg_range * 0.02 # 收盘价几乎相同
        # 看涨反击线: 下降趋势中（前长阴），当日长阳大幅跳空低开，但收盘价与前一日相同。
        bullish_counter = is_red1 & (body1 > avg_body) & is_green & (body > avg_body) & same_close_cond & (o < l1)
        patterns_calc.loc[bullish_counter, f'{kline_pattern_prefix}BULLISHCOUNTERATTACK_{tf}'] = 1 # 使用获取到的前缀
        # 看跌反击线: 上升趋势中（前长阳），当日长阴大幅跳空高开，但收盘价与前一日相同。
        bearish_counter = is_green1 & (body1 > avg_body) & is_red & (body > avg_body) & same_close_cond & (o > h1)
        patterns_calc.loc[bearish_counter, f'{kline_pattern_prefix}BEARISHCOUNTERATTACK_{tf}'] = -1 # 使用获取到的前缀

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

def calculate_rsi_score(rsi: pd.Series, params: Dict) -> pd.Series:
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
            lambda s: s.mean() + 2 * s.std() if s is not None and s.std() > 0 else (s.mean() + 0.01 * s.mean() if s is not None and s.mean() is not np.nan else 50.0), # upper 填充后，如果全 NaN 估算
            lambda s: s.mean() if s is not None and s.mean() is not np.nan else 50.0, # mid 填充后，如果全 NaN 估算
            lambda s: s.mean() - 2 * s.std() if s is not None and s.std() > 0 else (s.mean() - 0.01 * s.mean() if s is not None and s.mean() is not np.nan else 50.0)  # lower 填充后，如果全 NaN 估算
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

def calculate_cci_score(cci: pd.Series, params: Dict) -> pd.Series:
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

def calculate_mfi_score(mfi: pd.Series, params: Dict) -> pd.Series:
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

def calculate_roc_score(roc: pd.Series) -> pd.Series:
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

def calculate_sar_score(close: pd.Series, sar: pd.Series) -> pd.Series:
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

def calculate_stoch_score(k: pd.Series, d: pd.Series, params: Dict) -> pd.Series:
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

def calculate_ma_score(close: pd.Series, ma: pd.Series, params: Optional[Dict] = None) -> pd.Series:
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

def calculate_atr_score(atr: pd.Series) -> pd.Series:
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

def calculate_adl_score(adl: pd.Series) -> pd.Series:
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

def calculate_vwap_score(close: pd.Series, vwap: pd.Series) -> pd.Series:
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

def calculate_ichimoku_score(close: pd.Series, tenkan: pd.Series, kijun: pd.Series, senkou_a: pd.Series, senkou_b: pd.Series, chikou: pd.Series) -> pd.Series:
    """Ichimoku (一目均衡表) 评分 (0-100)。Simplified NaN handling."""
    # Ichimoku lines have inherent NaNs due to shifts. ffill/bfill is a simplification.
    # A more rigorous approach would respect these NaNs or use a sufficiently long data period.
    # Filling with close_s is a pragmatic choice if exact Ichimoku NaN propagation isn't critical.
    # MODIFIED: 使用 _safe_fillna_series 填充
    series_list = [close, tenkan, kijun, senkou_a, senkou_b, chikou]
    fill_values = [
        None, # close 优先 ffill/bfill
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0), # tenkan 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0), # kijun 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0), # senkou_a 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0), # senkou_b 填充后，如果全 NaN 使用均值
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0) # chikou 填充后，如果全 NaN 使用均值
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

def calculate_mom_score(mom: pd.Series) -> pd.Series:
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

def calculate_willr_score(willr: pd.Series) -> pd.Series:
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

def calculate_cmf_score(cmf: pd.Series) -> pd.Series:
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

def calculate_obv_score(obv: pd.Series, obv_ma: pd.Series = None, obv_ma_period: int = None) -> pd.Series:
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

def calculate_kc_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """KC (Keltner Channel) 评分 (0-100)。"""
    # Similar to BOLL, NaN handling for bands is key.
    # MODIFIED: 使用 _safe_fillna_series 填充
    close_s, upper_s, mid_s, lower_s = _safe_fillna_series(
        [close, upper, mid, lower],
        [
            None, # close 优先 ffill/bfill
             lambda s: s.mean() + 1.5 * (s.rolling(20, min_periods=1).max() - s.rolling(20, min_periods=1).min()).mean() if s is not None and s.mean() is not np.nan else (close.mean() + 1.5 * (close.rolling(20, min_periods=1).max() - close.rolling(20, min_periods=1).min()).mean() if close is not None and close.mean() is not np.nan else 50.0), # upper fallback
             lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0), # mid fallback
             lambda s: s.mean() - 1.5 * (s.rolling(20, min_periods=1).max() - s.rolling(20, min_periods=1).min()).mean() if s is not None and s.mean() is not np.nan else (close.mean() - 1.5 * (close.rolling(20, min_periods=1).max() - close.rolling(20, min_periods=1).min()).mean() if close is not None and close.mean() is not np.nan else 50.0) # lower fallback
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

def calculate_hv_score(hv: pd.Series) -> pd.Series:
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

def calculate_vroc_score(vroc: pd.Series) -> pd.Series:
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

def calculate_aroc_score(aroc: pd.Series) -> pd.Series:
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
        'R': {'series': r_series, 'base_score_breakout': 70, 'base_score_breakdown': None, 'level_multiplier': 5, 'is_resistance': True, 'prefix': 'R'}, # MODIFIED: Added prefix
        'S': {'series': s_series, 'base_score_breakout': None, 'base_score_breakdown': 30, 'level_multiplier': 5, 'is_resistance': False, 'prefix': 'S'}, # MODIFIED: Added prefix
        'F_R': {'series': fr_series, 'base_score_breakout': 75, 'base_score_breakdown': None, 'level_multiplier': 5, 'is_resistance': True, 'prefix': 'F_R'}, # 斐波那契阻力突破可能更强 # MODIFIED: Added prefix
        'F_S': {'series': fs_series, 'base_score_breakout': None, 'base_score_breakdown': 25, 'level_multiplier': 5, 'is_resistance': False, 'prefix': 'F_S'}, # 斐波那契支撑跌破可能更弱 # MODIFIED: Added prefix
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
    根据指标 key (策略内部标识), 内部 key (指标组成部分的标识，如 MACD 的 'macd_series', KDJ 的 'k'),
    参数列表 (计算该指标使用的参数值列表) 和时间框架后缀构建期望的 DataFrame 列名。

    此函数根据硬编码的指标命名规则来生成列名，这些规则应与 indicator_naming_conventions.json
    以及 IndicatorService 中实际生成指标列名的逻辑保持一致。

    Args:
        indicator_key (str): 策略内部用于标识指标的键 (例如 'macd', 'boll')。
        internal_key (str): 指标组成部分的内部键 (例如 MACD 的 'macd_series', BOLL 的 'upper')。
        params (List[Any]): 计算该指标时使用的参数值的列表 (例如 MACD 的 [12, 26, 9])。
        tf_suffix (str): 时间框架后缀 (例如 '30', 'D')。

    Returns:
        Optional[str]: 构建出的期望列名，如果无法识别指标或参数格式，则返回 None。
    """
    # 检查时间框架后缀是否有效
    if not tf_suffix:
         logger.error(f"构建列名失败: 时间框架后缀为空。 indicator_key={indicator_key}, internal_key={internal_key}")
         return None

    # 辅助函数：格式化参数为字符串，用于构建列名中的参数部分
    def format_param(p, key):
        if isinstance(p, float):
            # 根据 indicator_key 特殊处理浮点数参数的格式化
            if key == 'boll':
                 # BOLL 的标准差通常格式化为 .1f，例如 2.0 -> "2.0", 2.2 -> "2.2"
                 return f"{p:.1f}"
            if key == 'sar':
                 # SAR 的 af_step 和 max_af 格式化约定通常不同
                 # 根据 convention 文件，SAR_af_step_max_af_{timeframe}
                 # 例如 SAR_0.02_0.2_30
                 # af_step (params[0]) 格式化为 .2f，max_af (params[1]) 格式化为 .1f
                 # Need to know which parameter is which based on order in params list
                 # This is fragile. Relying on parameter names in a dict would be better.
                 # For now, assume order: [af_step, max_af]
                 if len(params) == 2:
                      if p == params[0]: return f"{p:.2f}" # af_step (期望是第一个参数)
                      if p == params[1]: return f"{p:.1f}" # max_af (期望是第二个参数)
                 # If cannot determine based on position, provide a default format (e.2f)
                 return f"{p:.2f}" # Default float format
            # Other float parameters convert directly to string
            return str(p)
        # Other types of parameters convert directly to string
        return str(p)

    # Build parameter string part from parameter list, joined by underscore
    param_str_parts = [format_param(p, indicator_key) for p in params] # MODIFIED: Pass indicator_key to format_param
    param_part = '_'.join(param_str_parts) if param_str_parts else ""
    # If parameter part is not empty, add an underscore as prefix
    param_suffix = f"_{param_part}" if param_part else ""


    # --- Build full column name based on indicator_key and internal_key ---

    # MACD Indicator
    if indicator_key == 'macd' and len(params) == 3: # Check parameter count matches MACD (fast_period, slow_period, signal_period)
        prefix_map = {'macd_series': 'MACD', 'macd_d': 'MACDs', 'macd_h': 'MACDh'} # Mapping from internal key to column prefix
        prefix = prefix_map.get(internal_key)
        if prefix:
            # Build column name: Prefix + Parameter Suffix + Timeframe Suffix
            # Example 'MACD_12_26_9_30'
            return f"{prefix}{param_suffix}_{tf_suffix}"

    # RSI Indicator
    elif indicator_key == 'rsi' and len(params) == 1: # Check parameter count matches RSI (period)
        if internal_key == 'rsi': # RSI score usually only needs the column corresponding to the RSI value itself
            # Build column name: 'RSI' + Parameter Suffix + Timeframe Suffix
            # Example 'RSI_14_30'
            return f"RSI{param_suffix}_{tf_suffix}"

    # KDJ Indicator (internal keys K, D, J)
    elif indicator_key == 'kdj' and len(params) == 3: # Check parameter count (K_period, D_period, Smooth_K_period)
        prefix_map = {'k': 'K', 'd': 'D', 'j': 'J'} # Mapping from internal key to column prefix
        prefix = prefix_map.get(internal_key)
        if prefix:
             # Build column name: Prefix + Parameter Suffix + Timeframe Suffix
             # Example 'K_9_3_3_30', 'D_9_3_3_30', 'J_9_3_3_30'
             return f"{prefix}{param_suffix}_{tf_suffix}"

    # BOLL Indicator
    elif indicator_key == 'boll' and len(params) == 2: # Check parameter count (period, std_dev)
        # MODIFIED: Corrected BOLL's prefix_map, added 'close'
        prefix_map = {'upper': 'BBU', 'mid': 'BBM', 'lower': 'BBL', 'close': 'close'} # Mapping from internal key to column prefix, includes close
        prefix = prefix_map.get(internal_key)
        if prefix:
             if internal_key == 'close':
                 # close column name has no parameter part, only 'close' + Timeframe Suffix
                 # Example 'close_30'
                 return f"{prefix}_{tf_suffix}"
             # BOLL Upper/Middle/Lower band column name: Prefix + Parameter Suffix + Timeframe Suffix
             # Parameter suffix consists of period and std_dev, e.g., '_20_2.0' or '_15_2.2'
             return f"{prefix}{param_suffix}_{tf_suffix}"

    # CCI Indicator
    elif indicator_key == 'cci' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'cci': # CCI score usually only needs the column corresponding to the CCI value itself
             # Build column name: 'CCI' + Parameter Suffix + Timeframe Suffix
             # Example 'CCI_14_30'
             return f"CCI{param_suffix}_{tf_suffix}"

    # MFI Indicator
    elif indicator_key == 'mfi' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'mfi': # MFI score usually only needs the column corresponding to the MFI value itself
             # Build column name: 'MFI' + Parameter Suffix + Timeframe Suffix
             # Example 'MFI_14_30'
             return f"MFI{param_suffix}_{tf_suffix}"

    # ROC Indicator (Price Rate of Change)
    elif indicator_key == 'roc' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'roc': # ROC score usually only needs the column corresponding to the ROC value itself
             # Build column name: 'ROC' + Parameter Suffix + Timeframe Suffix
             # Example 'ROC_12_30'
             return f"ROC{param_suffix}_{tf_suffix}"

    # DMI Indicator (+DI, -DI, ADX)
    elif indicator_key == 'dmi' and len(params) == 1: # Check parameter count (period)
        prefix_map = {'pdi': 'PDI', 'ndi': 'NDI', 'adx': 'ADX'} # Mapping from internal key to column prefix
        prefix = prefix_map.get(internal_key)
        if prefix:
             # Build column name: Prefix + Parameter Suffix + Timeframe Suffix
             # Parameter suffix consists of period, e.g., '_14'
             # Example 'PDI_14_30', 'NDI_14_30', 'ADX_14_30'
             return f"{prefix}{param_suffix}_{tf_suffix}"

    # SAR Indicator (Parabolic SAR)
    elif indicator_key == 'sar' and len(params) == 2: # Check parameter count (af_step, max_af)
        # MODIFIED: Corrected SAR's prefix_map, added 'close'
        prefix_map = {'sar': 'SAR', 'close': 'close'} # Mapping from internal key to column prefix, includes close
        prefix = prefix_map.get(internal_key)
        if prefix:
             if internal_key == 'close':
                  # close column name has no parameter part, only 'close' + Timeframe Suffix
                  return f"{prefix}_{tf_suffix}"
             # SAR column name format is special: 'SAR' + '_' + af_step (formatted) + '_' + max_af (formatted) + '_' + Timeframe Suffix
             # Example 'SAR_0.02_0.2_30'
             # Note: Parameter formatting is done directly below, not using the generic param_suffix
             if len(params) == 2:
                 param_str_sar = f"{format_param(params[0], indicator_key)}_{format_param(params[1], indicator_key)}" # SAR parameters' special formatting and joining
                 return f"{prefix}_{param_str_sar}_{tf_suffix}"
             else:
                 # Parameter count mismatch, log warning
                 logger.warning(f"构建 SAR 列名失败: 参数数量不正确 ({len(params)} != 2). internal_key='{internal_key}', params={params}, suffix='{tf_suffix}'")
                 return None

    # STOCH Indicator (Stochastic Oscillator %K and %D)
    elif indicator_key == 'stoch' and len(params) == 3: # Check parameter count (K_period, D_period, Smooth_K_period)
        prefix_map = {'k': 'STOCHk', 'd': 'STOCHd'} # Mapping from internal key to column prefix (using pandas_ta naming convention STOCHk, STOCHd)
        prefix = prefix_map.get(internal_key)
        if prefix:
             # Build column name: Prefix + Parameter Suffix + Timeframe Suffix
             # Parameter suffix consists of K_period, D_period, Smooth_K_period, e.g., '_14_3_3'
             # Example 'STOCHk_14_3_3_30', 'STOCHd_14_3_3_30'
             return f"{prefix}{param_suffix}_{tf_suffix}"

    # EMA / SMA Indicator (Generic Moving Average, distinguished by indicator_key)
    elif indicator_key in ['ema', 'sma'] and len(params) == 1: # Check parameter count (period)
        # MODIFIED: Corrected EMA/SMA's prefix_map, added 'close'
        prefix_map = {'ma': indicator_key.upper(), 'close': 'close'} # Mapping from internal key to column prefix, includes close
        prefix = prefix_map.get(internal_key)
        if prefix:
            if internal_key == 'close':
                 # close column name has no parameter part, only 'close' + Timeframe Suffix
                 return f"{prefix}_{tf_suffix}"
            # MA column name: Prefix + Parameter Suffix + Timeframe Suffix
            # Parameter suffix consists of period, e.g., '_20'
            # Example 'EMA_20_30', 'SMA_60_D'
            return f"{prefix}{param_suffix}_{tf_suffix}"

    # ATR Indicator (Average True Range)
    elif indicator_key == 'atr' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'atr': # ATR score usually only needs the column corresponding to the ATR value itself
             # Build column name: 'ATR' + Parameter Suffix + Timeframe Suffix
             # Example 'ATR_14_30'
             return f"ATR{param_suffix}_{tf_suffix}"

    # ADL Indicator (Accumulation/Distribution Line)
    elif indicator_key == 'adl' and not params: # Check parameter list is empty (ADL usually has no calculation parameters in the column name)
        if internal_key == 'adl': # ADL score usually only needs the column corresponding to the ADL value itself
             # Build column name: 'ADL' + '_' + Timeframe Suffix
             # Example 'ADL_30'
             return f"ADL_{tf_suffix}" # MODIFIED: Corrected ADL column name pattern to 'ADL_tf'

    # VWAP Indicator (Volume Weighted Average Price)
    elif indicator_key == 'vwap': # VWAP may or may not have an anchor parameter
        # VWAP column name can be 'VWAP_{timeframe}' or 'VWAP_{anchor}_{timeframe}'
        # According to IndicatorService generated column names, without anchor it's 'VWAP_{timeframe}'
        # With anchor it's 'VWAP_{anchor}_{timeframe}'
        prefix_map = {'vwap': 'VWAP', 'close': 'close'} # Mapping from internal key to column prefix
        prefix = prefix_map.get(internal_key)
        if prefix:
            if internal_key == 'close':
                # close column name has no parameter part
                 return f"{prefix}_{tf_suffix}"
            # VWAP column name needs to consider the anchor parameter (if params is not empty)
            if params:
                 # Assume the first parameter in the params list is the anchor
                 anchor = params[0]
                 # Build column name: 'VWAP' + '_' + anchor (if exists) + '_' + Timeframe Suffix
                 # Example 'VWAP_SESSION_D'
                 return f"VWAP_{anchor}_{tf_suffix}" # MODIFIED: Corrected VWAP column name pattern with anchor
            else:
                 # If params is empty, it means no anchor parameter
                 # Build column name: 'VWAP' + '_' + Timeframe Suffix
                 # Example 'VWAP_30'
                 return f"VWAP_{tf_suffix}" # MODIFIED: Corrected VWAP column name pattern without anchor


    # Ichimoku Indicator (Ichimoku Cloud)
    elif indicator_key == 'ichimoku' and len(params) == 3: # Check parameter count (tenkan, kijun, senkou_b periods)
         prefix_map = {
             'close': 'close', 'tenkan': 'TENKAN', 'kijun': 'KIJUN',
             'senkou_a': 'SENKOU_A', 'senkou_b': 'SENKOU_B', 'chikou': 'CHIKOU'
         }
         prefix = prefix_map.get(internal_key)
         if prefix:
             if internal_key == 'close':
                  # close column name has no parameter part
                  return f"{prefix}_{tf_suffix}"

             # Ichimoku lines have different column name patterns, need to build based on internal_key and corresponding parameters
             # params list should contain Tenkan, Kijun, Senkou B periods, e.g., [9, 26, 52]
             if len(params) >= 3: # Ensure enough parameters
                p_tenkan, p_kijun, p_senkou_b = params[:3] # Get main parameters

                if internal_key == 'tenkan': # Tenkan Sen: TENKAN_tenkan_period_tf
                    return f"TENKAN_{p_tenkan}_{tf_suffix}"
                if internal_key == 'kijun': # Kijun Sen: KIJUN_kijun_period_tf
                    return f"KIJUN_{p_kijun}_{tf_suffix}"
                if internal_key == 'chikou': # Chikou Span: CHIKOU_kijun_period_tf (uses Kijun's period)
                    return f"CHIKOU_{p_kijun}_{tf_suffix}"
                if internal_key == 'senkou_a': # Senkou Span A: SENKOU_A_tenkan_period_kijun_period_tf
                    return f"SENKOU_A_{p_tenkan}_{p_kijun}_{tf_suffix}"
                if internal_key == 'senkou_b': # Senkou Span B: SENKOU_B_senkou_b_period_tf (uses its own period)
                    return f"SENKOU_B_{p_senkou_b}_{tf_suffix}"
             else:
                 # Parameter count mismatch, log warning
                 logger.warning(f"构建 Ichimoku 列名失败: 参数数量不正确 ({len(params)} != 3). internal_key='{internal_key}', params={params}, suffix='{tf_suffix}'")
                 return None # Not enough parameters to build Ichimoku column name

    # MOM Indicator (Momentum)
    elif indicator_key == 'mom' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'mom': # MOM score usually only needs the column corresponding to the MOM value itself
             # Build column name: 'MOM' + Parameter Suffix + Timeframe Suffix
             # Example 'MOM_10_30'
             return f"MOM{param_suffix}_{tf_suffix}"

    # WILLR Indicator (Williams %R)
    elif indicator_key == 'willr' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'willr': # WILLR score usually only needs the column corresponding to the WILLR value itself
             # Build column name: 'WILLR' + Parameter Suffix + Timeframe Suffix
             # Example 'WILLR_14_30'
             return f"WILLR{param_suffix}_{tf_suffix}"

    # CMF Indicator (Chaikin Money Flow)
    elif indicator_key == 'cmf' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'cmf': # CMF score usually only needs the column corresponding to the CMF value itself
             # Build column name: 'CMF' + Parameter Suffix + Timeframe Suffix
             # Example 'CMF_20_30'
             return f"CMF{param_suffix}_{tf_suffix}"

    # OBV Indicator (On Balance Volume) and OBV_MA (Moving Average of OBV)
    elif indicator_key == 'obv':
        if internal_key == 'obv': # OBV column name has no parameter part
             # Build column name: 'OBV' + '_' + Timeframe Suffix
             # Example 'OBV_30'
             return f"OBV_{tf_suffix}"
        # OBV_MA needs to be built separately, parameter is the period of OBV_MA
        if internal_key == 'obv_ma' and len(params) == 1: # params should contain the period of OBV_MA [obv_ma_period]
             p_obv_ma = params[0] # Get the period of OBV_MA
             # Build column name: 'OBV_MA' + '_' + Period + '_' + Timeframe Suffix
             # Example 'OBV_MA_10_30'
             return f"OBV_MA_{p_obv_ma}_{tf_suffix}"

    # KC Indicator (Keltner Channels)
    elif indicator_key == 'kc' and len(params) == 2: # Check parameter count (EMA_period, ATR_period)
        # MODIFIED: Corrected KC's prefix_map, added 'close'
        prefix_map = {'upper': 'KCU', 'mid': 'KCM', 'lower': 'KCL', 'close': 'close'} # Mapping from internal key to column prefix, includes close
        prefix = prefix_map.get(internal_key)
        if prefix:
             if internal_key == 'close':
                 # close column name has no parameter part
                 return f"{prefix}_{tf_suffix}"
             # KC Upper/Middle/Lower band column name: Prefix + Parameter Suffix + Timeframe Suffix
             # Parameter suffix consists of EMA_period and ATR_period, e.g., '_20_10'
             return f"{prefix}{param_suffix}_{tf_suffix}"

    # HV Indicator (Historical Volatility)
    elif indicator_key == 'hv' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'hv': # HV score usually only needs the column corresponding to the HV value itself
             # Build column name: 'HV' + Parameter Suffix + Timeframe Suffix
             # Example 'HV_20_D'
             return f"HV{param_suffix}_{tf_suffix}"

    # VROC Indicator (Volume Rate of Change)
    elif indicator_key == 'vroc' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'vroc': # VROC score usually only needs the column corresponding to the VROC value itself
             # Build column name: 'VROC' + Parameter Suffix + Timeframe Suffix
             # Example 'VROC_12_30'
             return f"VROC{param_suffix}_{tf_suffix}"

    # AROC Indicator (Amount Rate of Change)
    elif indicator_key == 'aroc' and len(params) == 1: # Check parameter count (period)
        if internal_key == 'aroc': # AROC score usually only needs the column corresponding to the AROC value itself
             # Build column name: 'AROC' + Parameter Suffix + Timeframe Suffix
             # Example 'AROC_12_30'
             return f"AROC{param_suffix}_{tf_suffix}"

    # Pivot Indicator (Pivot Points)
    # Pivot levels (PP, R1, S1 etc.) are not built via this function, they are handled specially in calculate_all_indicator_scores
    # This function only builds the close column name needed for Pivot (if needed)
    elif indicator_key == 'pivot' and internal_key == 'close': # Check if it's the close column for Pivot
         # Pivot's close column name has no parameter part, only 'close' + Timeframe Suffix
         # Example 'close_D'
         return f"close_{tf_suffix}"
    # Pivot levels do not need to be built via build_expected_col_name individually, they are a list and handled uniformly in calculate_all_indicator_scores

    # If indicator_key or internal_key is unknown, or parameter count mismatch, log warning and return None
    logger.warning(f"无法为指标 '{indicator_key}' 构建列名，内部 key: '{internal_key}', 参数: {params}, 后缀: '{tf_suffix}'。配置或参数不匹配命名规则。")
    return None

def adjust_score_with_volume(
    preliminary_score: pd.Series,
    data: pd.DataFrame,
    vc_params: Dict,
    # 新增参数以支持调用时传递的额外参数，设置为可选参数
    vc_tf_list: Optional[List[str]] = None,
    vol_ma_period: Optional[int] = None, # <-- 接收成交量均线周期参数
    obv_ma_period: Optional[int] = None,
    naming_config: Optional[Dict] = None # 虽然不再用于格式化，但保留参数以兼容调用方
) -> pd.DataFrame:
    """
    使用量能相关指标（成交量、OBV、CMF等）对初步的策略评分 (0-100) 进行调整和确认。
    同时，此函数会计算并输出相关的量能分析信号。

    主要优化和深化点：
    1. 能够处理多个时间框架的量能分析。
    2. 更健壮的量价背离检测逻辑（启发式，寻找价格与OBV的趋势不一致）。
    3. 改进了分数调整机制，使其按比例向目标值（0, 50, 100）移动。
    4. 清晰的参数配置和默认值。
    5. 加强了数据对齐、缺失值处理的鲁棒性。
    6. 详细的中文注释。

    Args:
        preliminary_score (pd.Series): 经过基础指标计算得到的初步分数 (0-100)，索引为时间。
        data (pd.DataFrame): 包含OHLCV价格数据以及计算好的CMF、OBV等指标的DataFrame。
                             列名需要与配置中构建的名称一致。
        vc_params (Dict): 量能确认与调整相关的参数字典，包含：
            'enabled': bool, 是否启用基于量能的分数调整 (True/False)。
            'volume_analysis_enabled': bool, 是否启用量能分析信号的计算（即使不调整分数）。
            'tf': str or List[str], 进行量能分析所使用的数据时间级别 (例如 'D', '60', '15')。
                   如果 vc_tf_list 参数被传递，则优先使用 vc_tf_list。
            'boost_factor': float, 量能确认趋势时，分数向极值（0或100）移动的比例 (0.0-1.0)。
            'penalty_factor': float, 量能与趋势矛盾时，分数向中性值（50）移动的比例 (0.0-1.0)。
            'volume_spike_threshold': float, 成交量突增的判断阈值（当前量 / 均量的倍数）。
            'volume_analysis_lookback': int, 计算成交量均线、寻找量价背离等的回看窗口期。
            'cmf_period': int, CMF指标的计算周期 (应与指标服务中的计算一致)。
            'cmf_confirmation_threshold': float, CMF确认趋势的阈值 (例如 0.05)。
            'obv_ma_period': int, OBV移动平均线的计算周期 (应与指标服务中的计算一致)。
            'volume_ma_period': int, 成交量移动平均线的计算周期 (应与指标服务中的计算一致)。
            'vp_divergence_lookback': int, 量价背离检测的回看窗口期。
            'vp_divergence_peak_offset': int, 在量价背离中，比较当前极值点与之前第N个极值点。(此参数在当前实现中可能不直接使用，但保留)
            'vp_divergence_price_threshold': float, 价格创出新高/新低的最小幅度（相对于前一极值）。
            'vp_divergence_obv_threshold': float, OBV未能同步创出新高/新低的最小幅度（相对于前一极值对应的OBV）。
            'vp_divergence_penalty_factor': float, 量价背离惩罚分数调整的比例。
            'volume_spike_adj_factor': float, 成交量突增时调整分数的比例。
        vc_tf_list (Optional[List[str]]): 量能分析的时间框架列表，如果提供，则覆盖 vc_params['tf']。
        vol_ma_period (Optional[int]): 成交量移动平均线的周期，如果提供，则覆盖 vc_params['volume_ma_period']。
        obv_ma_period (Optional[int]): OBV移动平均线的周期，如果提供，则覆盖 vc_params['obv_ma_period']。
        naming_config (Optional[Dict]): 命名规范配置字典，如果提供，则覆盖默认配置。

    Returns:
        pd.DataFrame: 返回一个DataFrame，索引与 preliminary_score 一致，包含以下列:
            - 'ADJUSTED_SCORE': 经过量能调整后的最终分数 (0-100)。
            - 'VOL_CONFIRM_SIGNAL_{TF}': 量能确认信号 (1: 量能支持当前趋势, -1: 量能与当前趋势矛盾, 0: 中性). (为每个处理的时间框架生成)
            - 'VOL_SPIKE_SIGNAL_{TF}': 成交量突增信号 (1: 检测到突增, 0: 正常). (为每个处理的时间框架生成)
            - 'VOL_PRICE_DIV_SIGNAL_{TF}': 量价背离信号 (1: 看涨背离, -1: 看跌背离, 0: 无明显背离). (为每个处理的时间框架生成)
    """
    # --- 0. 初始化和参数准备 ---
    result_df = pd.DataFrame(index=preliminary_score.index)
    # 初始调整后分数等于原始分数，后续会修改
    result_df['ADJUSTED_SCORE'] = preliminary_score.copy().fillna(50.0) # 初始化时填充NaN为50

    # 使用传入的 naming_config，如果未传入则使用默认
    current_naming_config = naming_config if naming_config is not None else {}

    # 确定要处理的时间框架列表
    # 优先使用传入的 vc_tf_list 参数
    vol_tf_list_to_process = vc_tf_list
    if vol_tf_list_to_process is None or not isinstance(vol_tf_list_to_process, list) or not vol_tf_list_to_process:
        # 如果 vc_tf_list 无效或为空，则尝试从 vc_params['tf'] 获取
        tf_from_params = vc_params.get('tf', 'D') # 默认日线
        if isinstance(tf_from_params, str):
            vol_tf_list_to_process = [tf_from_params]
        elif isinstance(tf_from_params, list) and tf_from_params:
            vol_tf_list_to_process = tf_from_params
        else:
            # 如果 vc_params['tf'] 也无效，则使用默认日线并警告
            vol_tf_list_to_process = ['D']
            logger.warning("量能调整/分析模块：vc_tf_list 参数和 vc_params['tf'] 配置均无效或为空，使用默认时间框架: 'D'。")
    # 确保列表中的元素是字符串
    vol_tf_list_to_process = [str(tf) for tf in vol_tf_list_to_process]

    # 检查是否启用了量能分析或分数调整
    score_adjustment_enabled = vc_params.get('enabled', False)
    volume_analysis_enabled = vc_params.get('volume_analysis_enabled', False)
    if not score_adjustment_enabled and not volume_analysis_enabled:
        logger.info("量能分数调整和量能分析均未启用。")
        # 即使不启用，也返回包含原始分数和默认0信号列的DataFrame
        # 预先创建所有可能的信号列，并用0填充，确保它们总是存在于返回的DataFrame中
        # internal_cols_conf = current_naming_config.get('strategy_internal_columns', {}).get('output_columns', []) # 删除此行
        # signal_patterns = { ... } # 删除此字典创建
        for tf in vol_tf_list_to_process:
             # 直接使用字符串拼接构建信号列名 # 修改行
             result_df[f'VOL_CONFIRM_SIGNAL_{tf}'] = 0 # 修改行
             result_df[f'VOL_SPIKE_SIGNAL_{tf}'] = 0 # 修改行
             result_df[f'VOL_PRICE_DIV_SIGNAL_{tf}'] = 0 # 修改行

        return result_df # 返回包含默认0信号和原始（或填充后）分数的DataFrame

    # 提取各项配置参数
    boost_factor = vc_params.get('boost_factor', 0.20)
    penalty_factor = vc_params.get('penalty_factor', 0.30)
    vol_spike_threshold = vc_params.get('volume_spike_threshold', 2.5)
    vol_analysis_lookback = vc_params.get('volume_analysis_lookback', 20)
    cmf_period = vc_params.get('cmf_period', 20)
    cmf_confirm_thresh = vc_params.get('cmf_confirmation_threshold', 0.05)

    # 使用传入的周期参数，如果未传入则使用 vc_params 中的配置
    current_vol_ma_period = vol_ma_period if vol_ma_period is not None else vc_params.get('volume_ma_period', 20) # 优先使用传入参数
    current_obv_ma_period = obv_ma_period if obv_ma_period is not None else vc_params.get('obv_ma_period', 10) # 优先使用传入参数

    # 量价背离相关参数
    vp_div_lookback = vc_params.get('vp_divergence_lookback', 21)
    vp_price_thresh = vc_params.get('vp_divergence_price_threshold', 0.005)
    vp_obv_thresh = vc_params.get('vp_divergence_obv_threshold', 0.005)
    vp_div_penalty_factor = vc_params.get('vp_divergence_penalty_factor', 0.25)
    vol_spike_adj_factor = vc_params.get('volume_spike_adj_factor', 0.10)

    # 变量用于存储第一个成功处理的时间框架的信号，用于分数调整
    adjustment_confirm_signal = pd.Series(0, index=result_df.index)
    adjustment_spike_signal = pd.Series(0, index=result_df.index)
    adjustment_div_signal = pd.Series(0, index=result_df.index)
    first_tf_processed = False # 标记是否已成功处理至少一个时间框架

    # --- 循环处理每个时间框架 ---
    processed_tfs = [] # 记录成功处理的时间框架
    for current_vol_tf in vol_tf_list_to_process:
        print(f"\n--- DEBUG PRINT: 处理量能时间框架: {current_vol_tf} ---") # 保留调试输出

        # 1. 数据列名构建和有效性检查 (针对当前时间框架)
        # 使用 naming_config 查找列名模式并构建实际列名
        ohlcv_naming_conv = current_naming_config.get('ohlcv_naming_convention', {})
        indicator_naming_conv = current_naming_config.get('indicator_naming_conventions', {})
        timeframe_naming_conv = current_naming_config.get('timeframe_naming_convention', {})

        # Get possible suffixes for the current timeframe
        tf_score_str = str(current_vol_tf)
        possible_tf_suffixes_raw = timeframe_naming_conv.get('patterns', {}).get(tf_score_str.lower(), [tf_score_str])
        if not isinstance(possible_tf_suffixes_raw, list): possible_tf_suffixes = [str(possible_tf_suffixes_raw)]
        else: possible_tf_suffixes = [str(p) for p in possible_tf_suffixes_raw]
        if tf_score_str in possible_tf_suffixes: possible_tf_suffixes.remove(tf_score_str); possible_tf_suffixes.insert(0, tf_score_str)
        elif tf_score_str.upper() in possible_tf_suffixes: possible_tf_suffixes.remove(tf_score_str.upper()); possible_tf_suffixes.insert(0, tf_score_str.upper())
        else: possible_tf_suffixes.insert(0, tf_score_str)
        seen = set(); possible_tf_suffixes_unique = [];
        for suffix in possible_tf_suffixes:
            if suffix not in seen: seen.add(suffix); possible_tf_suffixes_unique.append(suffix)
        possible_tf_suffixes = possible_tf_suffixes_unique

        # Find OHLCV columns
        ohlcv_cols_found: Dict[str, str] = {}
        ohlcv_base_names = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_patterns = ohlcv_naming_conv.get('output_columns', [])
        if isinstance(ohlcv_patterns, list):
             for base_name in ohlcv_base_names:
                  pattern = next((p.get('name_pattern') for p in ohlcv_patterns if isinstance(p, dict) and p.get('name_pattern') == base_name), None)
                  if pattern:
                       found_col = None
                       for suffix in possible_tf_suffixes:
                            expected_col = f"{pattern}_{suffix}"
                            if expected_col in data.columns:
                                 found_col = expected_col
                                 break
                       if found_col: ohlcv_cols_found[base_name] = found_col
                  else:
                       logger.warning(f"OHLCV naming pattern for '{base_name}' not found.")

        # Find CMF, OBV, OBV_MA columns
        indicator_cols_found: Dict[str, str] = {}
        indicators_to_find = {'cmf': cmf_period, 'obv': None, 'obv_ma': current_obv_ma_period} # internal_key: period (or None)

        for internal_key, period_val in indicators_to_find.items():
             indi_naming_conf = indicator_naming_conv.get(internal_key.upper(), {})
             output_cols_patterns = indi_naming_conf.get('output_columns', [])
             ma_pattern = indi_naming_conf.get('ma_pattern') # For OBV_MA

             pattern = None
             if internal_key == 'obv_ma' and ma_pattern: pattern = ma_pattern
             elif isinstance(output_cols_patterns, list):
                  for col_conf in output_cols_patterns:
                       if isinstance(col_conf, dict) and col_conf.get('name_pattern', '').lower().startswith(internal_key):
                            pattern = col_conf.get('name_pattern')
                            break

             if pattern:
                  found_col = None
                  for suffix in possible_tf_suffixes:
                       expected_col = None
                       if period_val is not None: # Pattern with period
                            # Need to format period correctly, similar to build_expected_col_name
                            param_str = str(period_val) # Simplified param formatting
                            expected_col = f"{pattern}_{param_str}_{suffix}"
                            expected_col = expected_col.replace('__', '_').strip('_')
                       else: # Pattern without period (like OBV)
                            expected_col = f"{pattern}_{suffix}"
                            expected_col = expected_col.replace('__', '_').strip('_')

                       if expected_col and expected_col in data.columns:
                            found_col = expected_col
                            break
                  if found_col: indicator_cols_found[internal_key] = found_col
             else:
                  logger.warning(f"Indicator naming pattern for '{internal_key}' not found.")


        # Check if all required columns are found and have non-NaN data
        required_keys_for_tf = ['close', 'high', 'low', 'volume', 'cmf', 'obv'] # OBV_MA is optional for basic analysis
        all_required_found = True
        for key in required_keys_for_tf:
             col_name = ohlcv_cols_found.get(key) if key in ohlcv_base_names else indicator_cols_found.get(key)
             if col_name is None or col_name not in data.columns or data[col_name].isnull().all():
                  logger.warning(f"量能调整/分析模块：时间框架 '{current_vol_tf}' 缺少必需的数据列或数据全为 NaN: '{col_name}' (internal key: '{key}')。跳过此时间框架的分析。")
                  all_required_found = False
                  break

        if not all_required_found:
             # 预先创建当前时间框架的信号列，并用0填充，即使数据缺失也要确保列存在
             confirm_signal_col = f'VOL_CONFIRM_SIGNAL_{current_vol_tf}'
             spike_signal_col = f'VOL_SPIKE_SIGNAL_{current_vol_tf}'
             div_signal_col = f'VOL_PRICE_DIV_SIGNAL_{current_vol_tf}'
             result_df[confirm_signal_col] = 0
             result_df[spike_signal_col] = 0
             result_df[div_signal_col] = 0
             continue # Skip current timeframe, continue to next

        # 2. 数据提取, 对齐和填充 (针对当前时间框架)
        # 合并 preliminary_score 和所需数据列，确保索引对齐
        # 使用 outer join 避免丢失任一方的索引，然后 reindex 回 preliminary_score 的索引
        cols_to_merge = [close_col, high_col, low_col, volume_col, cmf_col_name, obv_col_name]
        if obv_ma_col_name: cols_to_merge.append(obv_ma_col_name)

        data_subset = data[cols_to_merge].copy() # 操作副本
        merged_df_tf = pd.concat([preliminary_score.rename("PRELIM_SCORE_TEMP_COL"), data_subset], axis=1, join='outer')
        merged_df_tf = merged_df_tf.reindex(preliminary_score.index) # 确保最终索引与输入分数一致

        # 对提取的序列进行填充
        close = merged_df_tf[close_col].ffill().bfill() # 价格用前后值填充
        high = merged_df_tf[high_col].ffill().bfill()
        low = merged_df_tf[low_col].ffill().bfill()
        volume = merged_df_tf[volume_col].fillna(0) # 成交量NaN填充为0
        cmf = merged_df_tf[cmf_col_name].fillna(0) # CMF NaN填充为0 (代表中性资金流)
        obv = merged_df_tf[obv_col_name].ffill().bfill() # OBV是累积值，用前后值填充可能更合理
        obv_ma = merged_df_tf[obv_ma_col_name].ffill().bfill() if obv_ma_col_name else pd.Series(np.nan, index=merged_df_tf.index) # OBV均线也用前后值填充，处理如果列未找到的情况

        # 再次检查关键数据是否在填充后仍然无效
        if close.isnull().all() or obv.isnull().all() or volume.isnull().all():
            logger.warning(f"量能调整/分析模块：时间框架 '{current_vol_tf}' 的关键数据 (收盘价/OBV/成交量) 在填充后仍无效。跳过此时间框架的分析。")
            continue # 跳过当前时间框架，继续处理下一个

        # --- 3. 计算量能分析信号 (针对当前时间框架) ---

        # 3.1. 量能确认信号 (VOL_CONFIRM_SIGNAL_{TF})
        # 定义：CMF 和 OBV 相对于其均线的方向是否支持当前价格趋势的初步判断
        # 看涨量能确认: CMF 为正且高于阈值，并且 OBV 在其均线上方
        bullish_volume_confirmation = (cmf > cmf_confirm_thresh) & (obv > obv_ma)
        # 看跌量能确认: CMF 为负且低于负阈值，并且 OBV 在其均线下方
        bearish_volume_confirmation = (cmf < -cmf_confirm_thresh) & (obv < obv_ma)

        # 将信号存储到 result_df 中对应的列
        confirm_signal_col = f'VOL_CONFIRM_SIGNAL_{current_vol_tf}'
        result_df[confirm_signal_col] = 0 # 在赋值前重置为0
        result_df.loc[bullish_volume_confirmation, confirm_signal_col] = 1
        result_df.loc[bearish_volume_confirmation, confirm_signal_col] = -1

        # 3.2. 成交量突增信号 (VOL_SPIKE_SIGNAL_{TF})
        # 定义：当前成交量显著高于其近期移动平均值
        spike_signal_col = f'VOL_SPIKE_SIGNAL_{current_vol_tf}'
        result_df[spike_signal_col] = 0 # 在赋值前重置为0
        if vol_analysis_lookback > 0 and not volume.empty:
            # 计算成交量均线，替换0值为NaN以避免除零错误
            # 使用 current_vol_ma_period 作为成交量均线周期
            vol_mean_period = current_vol_ma_period
            volume_mean = volume.rolling(window=vol_mean_period, min_periods=max(1, vol_mean_period // 2)).mean().replace(0, np.nan)
            # 仅在volume_mean有效时进行比较
            valid_mean_mask = volume_mean.notna()
            is_spike = pd.Series(False, index=volume.index) # 初始化为全False
            is_spike.loc[valid_mean_mask] = (volume.loc[valid_mean_mask] / volume_mean.loc[valid_mean_mask]) > vol_spike_threshold
            result_df[spike_signal_col] = is_spike.astype(int)

        # 3.3. 量价背离信号 (VOL_PRICE_DIV_SIGNAL_{TF}) - 深化版检测
        # 目标：检测价格创出新高/新低时，OBV是否未能同步创出新高/新低
        # 我们通过回溯窗口，寻找窗口内的最高价/最低价及其对应的OBV值，并与当前价/OBV进行比较。

        # 初始化背离信号 Series (针对当前时间框架)
        div_signal_col = f'VOL_PRICE_DIV_SIGNAL_{current_vol_tf}'
        result_df[div_signal_col] = 0 # 在赋值前重置为0
        divergence_signals = pd.Series(0, index=result_df.index)

        if vp_div_lookback > 1 and not obv.isnull().all(): # 需要至少2个点进行比较，且OBV数据有效
            # 遍历每个数据点，从可以形成完整回溯窗口的点开始
            # iloc 索引用于迭代，loc 索引用于基于日期/时间查找
            # 使用 merged_df_tf 来确保索引与 preliminary_score 对齐
            for i in range(vp_div_lookback - 1, len(merged_df_tf)): # 确保至少有 lookback 数量的数据可以回溯
                current_date_idx = merged_df_tf.index[i] # 当前日期索引

                # 获取回溯窗口数据 (不包含当前日期)
                lookback_slice = merged_df_tf.iloc[max(0, i - vp_div_lookback + 1): i].copy() # 过去 vp_div_lookback-1 天的数据

                if lookback_slice.empty:
                    continue # 窗口为空，跳过

                # 确保回溯窗口内有有效数据
                if lookback_slice[high_col].isnull().all() or lookback_slice[low_col].isnull().all() or lookback_slice[obv_col_name].isnull().all():
                    continue # 窗口内关键数据无效，跳过

                # --- 寻找回溯窗口内的价格极值点及其对应的OBV值 ---
                # 找到窗口内的最高价及其索引 (P1)
                p1_val = lookback_slice[high_col].max()
                p1_date_idx = lookback_slice[high_col].idxmax() # 最高价对应的日期索引

                # 找到窗口内的最低价及其索引 (T1)
                t1_val = lookback_slice[low_col].min()
                t1_date_idx = lookback_slice[low_col].idxmin() # 最低价对应的日期索引

                # 获取 P1 和 T1 日期对应的 OBV 值 (I1_at_P1, I1_at_T1)
                i1_obv_at_p1 = lookback_slice.loc[p1_date_idx, obv_col_name] if p1_date_idx in lookback_slice.index else np.nan # 确保索引存在
                i1_obv_at_t1 = lookback_slice.loc[t1_date_idx, obv_col_name] if t1_date_idx in lookback_slice.index else np.nan # 确保索引存在

                # --- 获取当前日期的价格和OBV值 (P2, I2) ---
                p2_val_high = high.iloc[i] # 当前最高价
                p2_val_low = low.iloc[i] # 当前最低价
                i2_obv = obv.iloc[i] # 当前OBV

                # 检查当前数据是否有效
                if pd.isna(p2_val_high) or pd.isna(p2_val_low) or pd.isna(i2_obv):
                    continue

                # --- 判断量价背离 ---
                # 看跌背离条件: 价格创出更高的高点 (P2 > P1) 且 OBV 未能创出更高的高点 (I2 <= I1_at_P1)
                # 同时考虑阈值，避免微小波动引起的误判
                if p2_val_high > p1_val * (1 + vp_price_thresh): # 价格明显创新高
                    # OBV 未能同步创新高：OBV 当前值低于或等于之前高点对应的OBV值 (考虑OBV阈值)
                    # 使用 OBV 的相对变化，避免 OBV 绝对值大小的影响
                    if not pd.isna(i1_obv_at_p1) and (i2_obv <= i1_obv_at_p1 or (i1_obv_at_p1 != 0 and (i2_obv - i1_obv_at_p1) / abs(i1_obv_at_p1) < -vp_obv_thresh)):
                        divergence_signals.loc[current_date_idx] = -1 # 标记为看跌背离

                # 看涨背离条件: 价格创出更低的低点 (P2 < T1) 且 OBV 未能创出更低的低点 (I2 >= I1_at_T1)
                # 同时考虑阈值，避免微小波动引起的误判
                # 只有当前点没有被标记为看跌背离时才检查看涨背离
                if divergence_signals.loc[current_date_idx] == 0 and p2_val_low < t1_val * (1 - vp_price_thresh): # 价格明显创新低
                    # OBV 未能同步创新低：OBV 当前值高于或等于之前低点对应的OBV值 (考虑OBV阈值)
                    if not pd.isna(i1_obv_at_t1) and (i2_obv >= i1_obv_at_t1 or (i1_obv_at_t1 != 0 and (i2_obv - i1_obv_at_t1) / abs(i1_obv_at_t1) > vp_obv_thresh)):
                        divergence_signals.loc[current_date_idx] = 1 # 标记为看涨背离

        result_df[div_signal_col] = divergence_signals.astype(int) # 确保信号为整数类型

        # 如果这是第一个成功处理的时间框架，则将其信号用于后续的分数调整
        if not first_tf_processed:
            adjustment_confirm_signal = result_df[confirm_signal_col].copy()
            adjustment_spike_signal = result_df[spike_signal_col].copy()
            adjustment_div_signal = result_df[div_signal_col].copy()
            first_tf_processed = True
            logger.debug(f"使用时间框架 '{current_vol_tf}' 的信号进行分数调整。")

        processed_tfs.append(current_vol_tf) # 记录成功处理的时间框架

    # --- 4. 应用量能调整到初步分数 (在循环结束后，仅当 score_adjustment_enabled 为 True 且至少处理了一个时间框架) ---
    if score_adjustment_enabled and first_tf_processed:
        logger.info("应用量能调整到初步分数...")
        current_score = result_df['ADJUSTED_SCORE'].copy() # 获取当前待调整的分数副本

        # 判断初步分数的趋势方向 (基于原始初步分数)
        is_bullish_prelim_score = preliminary_score.fillna(50.0) > 55 # 初步看涨 (可根据实际策略调整阈值)
        is_bearish_prelim_score = preliminary_score.fillna(50.0) < 45 # 初步看跌 (可根据实际策略调整阈值)
        is_neutral_prelim_score = (~is_bullish_prelim_score) & (~is_bearish_prelim_score) # 初步中性

        # 4.1. 基于量能确认信号调整 (使用第一个成功处理的时间框架的信号)
        # a) 初步看涨
        #    - 量能支持 (adjustment_confirm_signal == 1): 分数向100移动
        bull_confirmed_cond = is_bullish_prelim_score & (adjustment_confirm_signal == 1)
        current_score.loc[bull_confirmed_cond] = current_score + (100 - current_score) * boost_factor
        #    - 量能矛盾 (adjustment_confirm_signal == -1): 分数向50移动
        bull_contradict_cond = is_bullish_prelim_score & (adjustment_confirm_signal == -1)
        current_score.loc[bull_contradict_cond] = current_score - (current_score - 50) * penalty_factor

        # b) 初步看跌
        #    - 量能支持 (adjustment_confirm_signal == -1): 分数向0移动
        bear_confirmed_cond = is_bearish_prelim_score & (adjustment_confirm_signal == -1)
        current_score.loc[bear_confirmed_cond] = current_score - current_score * boost_factor
        #    - 量能矛盾 (adjustment_confirm_signal == 1): 分数向50移动
        bear_contradict_cond = is_bearish_prelim_score & (adjustment_confirm_signal == 1)
        current_score.loc[bear_contradict_cond] = current_score + (50 - current_score) * penalty_factor

        # c) 初步中性 (分数在45-55之间)
        #    - 量能看涨 (adjustment_confirm_signal == 1): 分数向看涨区域轻微移动 (例如60)
        neutral_to_bull_cond = is_neutral_prelim_score & (adjustment_confirm_signal == 1)
        current_score.loc[neutral_to_bull_cond] = current_score + (60 - current_score) * boost_factor * 0.5 # 调整幅度减半
        #    - 量能看跌 (adjustment_confirm_signal == -1): 分数向看跌区域轻微移动 (例如40)
        neutral_to_bear_cond = is_neutral_prelim_score & (adjustment_confirm_signal == -1)
        current_score.loc[neutral_to_bear_cond] = current_score - (current_score - 40) * boost_factor * 0.5 # 调整幅度减半

        # 4.2. 基于量价背离信号调整 (通常是惩罚性或警示性) (使用信号从第一个成功处理的时间框架)
        # a) 初步看涨，但出现看跌量价背离 (adjustment_div_signal == -1) -> 分数向50回调
        bull_score_with_bear_div = is_bullish_prelim_score & (adjustment_div_signal == -1)
        result_df.loc[bull_score_with_bear_div, 'ADJUSTED_SCORE'] = \
            result_df.loc[bull_score_with_bear_div, 'ADJUSTED_SCORE'] - \
            (result_df.loc[bull_score_with_bear_div, 'ADJUSTED_SCORE'] - 50) * vp_div_penalty_factor

        # b) 初步看跌，但出现看涨量价背离 (adjustment_div_signal == 1) -> 分数向50回调
        bear_score_with_bull_div = is_bearish_prelim_score & (adjustment_div_signal == 1)
        result_df.loc[bear_score_with_bull_div, 'ADJUSTED_SCORE'] = \
            result_df.loc[bear_score_with_bull_div, 'ADJUSTED_SCORE'] + \
            (50 - result_df.loc[bear_score_with_bull_div, 'ADJUSTED_SCORE']) * vp_div_penalty_factor

        # 4.3. 基于成交量突增信号调整 (可选，可能加强趋势或预示反转) (使用信号从第一个成功处理的时间框架)
        # 简化处理：如果趋势与突增方向一致，则轻微加强分数；如果矛盾，则轻微拉回。
        # a) 初步看涨 + 成交量突增: 轻微加强看涨分数
        bull_score_with_spike = is_bullish_prelim_score & (adjustment_spike_signal == 1)
        result_df.loc[bull_score_with_spike, 'ADJUSTED_SCORE'] = \
            result_df.loc[bull_score_with_spike, 'ADJUSTED_SCORE'] + \
            (100 - result_df.loc[bull_score_with_spike, 'ADJUSTED_SCORE']) * vol_spike_adj_factor

        # b) 初步看跌 + 成交量突增: 轻微加强看跌分数
        bear_score_with_spike = is_bearish_prelim_score & (adjustment_spike_signal == 1)
        result_df.loc[bear_score_with_spike, 'ADJUSTED_SCORE'] = \
            result_df.loc[bear_score_with_spike, 'ADJUSTED_SCORE'] - \
            result_df.loc[bear_score_with_spike, 'ADJUSTED_SCORE'] * vol_spike_adj_factor

        # 确保最终分数在0-100范围内
        result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].clip(0, 100)

    elif score_adjustment_enabled and not first_tf_processed:
         logger.warning("量能分数调整已启用，但没有成功处理任何时间框架的量能数据。分数将不会被调整。")


    # --- 5. 最终填充和返回 ---
    # 确保 ADJUSTED_SCORE 列的 NaN 被填充为中性值 50 (再次确认)
    result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].fillna(50.0)
    # 其他信号列的 NaN (理论上在初始化时已处理) 再次确保填充为0并转换为整数
    # 遍历所有以信号模式开头的列 # 修改行
    # 手动构建信号列名前缀列表 # 新增行
    signal_prefixes = ['VOL_CONFIRM_SIGNAL_', 'VOL_SPIKE_SIGNAL_', 'VOL_PRICE_DIV_SIGNAL_'] # 新增行
    all_signal_cols = [col for col in result_df.columns if any(col.startswith(prefix) for prefix in signal_prefixes)] # 修改行: Complete the list comprehension
    for col in all_signal_cols:
        result_df[col] = result_df[col].fillna(0).astype(int)

    logger.info(f"量能调整和分析模块处理完成。成功处理的时间框架: {processed_tfs}。")
    return result_df

# 注意：在修改后的回退查找逻辑中，此函数不再用于匹配模式，但保留以防其他地方使用或用于调试。
def parse_col_params(col_name: str, indicator_key: str, tf_suffix: str) -> List[Any] | None:
    """
    尝试从包含时间框架后缀的 DataFrame 列名中解析指标参数列表。
    此函数根据硬编码的指标命名规则来解析列名，提取参数。
    这些规则应与 indicator_naming_conventions.json 文件中定义的命名规范
    以及 IndicatorService 中实际生成指标列名的逻辑保持一致。
    Args:
        col_name (str): 完整的 DataFrame 列名 (例如 'BBU_20_2.0_30', 'ADX_14_30')。
        indicator_key (str): 策略内部用于标识指标的键 (例如 'macd', 'boll')。
        tf_suffix (str): 期望的时间框架后缀 (例如 '30', 'D')。
    Returns:
        List[Any] | None: 解析出的参数列表，如果后缀不匹配、列名模式不识别或参数转换失败，则返回 None。
                          对于没有参数但需要匹配的列名，返回空列表 []。
    """
    # 检查列名是否以期望的时间框架后缀结尾
    if not col_name.endswith(f"_{tf_suffix}"):
         # 如果后缀不匹配，记录调试信息并返回 None
         # print(f"列名 '{col_name}' 后缀与期望 '{tf_suffix}' 不匹配，无法解析参数。")
         return None
    # 移除时间框架后缀，得到基础名称和参数部分
    base_name_with_params = col_name[:-len(f"_{tf_suffix}")]
    # 按下划线分割，通常第一个部分是指标前缀，后面是参数
    parts = base_name_with_params.split('_')

    try:
        # 根据 indicator_key 识别指标类型并尝试解析参数
        if indicator_key == 'macd' and parts[0] in ['MACD', 'MACDh', 'MACDs']:
            # MACD 列名模式: MACD/MACDh/MACDs_fast_slow_signal
            if len(parts) >= 4: # 至少包含前缀和3个参数
                 # 尝试将参数部分转换为整数
                 return [int(parts[1]), int(parts[2]), int(parts[3])]

        elif indicator_key == 'rsi' and parts[0] == 'RSI':
            # RSI 列名模式: RSI_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                  # 尝试将参数部分转换为整数
                  return [int(parts[1])]

        elif indicator_key == 'kdj' and parts[0] in ['K', 'D', 'J']:
            # KDJ 列名模式: K/D/J_period_signal_period_smooth_k_period
             if len(parts) >= 4: # 至少包含前缀和3个参数
                  # 尝试将参数部分转换为整数
                  return [int(parts[1]), int(parts[2]), int(parts[3])]

        elif indicator_key == 'boll' and parts[0] in ['BBL', 'BBM', 'BBU']:
            # BOLL 列名模式: BBL/BBM/BBU_period_std_dev
             if len(parts) >= 3: # 至少包含前缀和2个参数
                  # 尝试将周期转换为整数，标准差转换为浮点数
                  return [int(parts[1]), float(parts[2])] # std_dev 是浮点数

        elif indicator_key == 'cci' and parts[0] == 'CCI':
            # CCI 列名模式: CCI_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                  # 尝试将参数部分转换为整数
                  return [int(parts[1])]

        elif indicator_key == 'mfi' and parts[0] == 'MFI':
            # MFI 列名模式: MFI_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                  # 尝试将参数部分转换为整数
                  return [int(parts[1])]

        elif indicator_key == 'roc' and parts[0] == 'ROC':
            # ROC 列名模式: ROC_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                  # 尝试将参数部分转换为整数
                  return [int(parts[1])]

        elif indicator_key == 'dmi' and parts[0] in ['PDI', 'NDI', 'ADX']:
            # DMI 列名模式: PDI/NDI/ADX_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                  # 尝试将参数部分转换为整数
                  return [int(parts[1])]

        elif indicator_key == 'sar' and parts[0] == 'SAR':
             # SAR 列名模式: SAR_af_step_max_af (参数是浮点数)
             if len(parts) >= 3: # 至少包含前缀和2个参数
                 # 尝试将参数部分转换为浮点数
                 return [float(parts[1]), float(parts[2])]

        elif indicator_key == 'stoch' and parts[0] in ['STOCHk', 'STOCHd']:
            # STOCH 列名模式: STOCHk/STOCHd_k_period_d_period_smooth_k_period
             if len(parts) >= 4: # 至少包含前缀和3个参数
                 # 尝试将参数部分转换为整数
                 return [int(parts[1]), int(parts[2]), int(parts[3])]

        elif indicator_key in ['ema', 'sma'] and parts[0] in ['EMA', 'SMA']:
            # MA 列名模式: EMA/SMA_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                 # 尝试将参数部分转换为整数
                 return [int(parts[1])]

        elif indicator_key == 'atr' and parts[0] == 'ATR':
             # ATR 列名模式: ATR_period
             if len(parts) >= 2: # 至少包含前缀和1个参数
                 # 尝试将参数部分转换为整数
                 return [int(parts[1])]

        elif indicator_key == 'adl' and parts[0] == 'ADL':
            # ADL 列名模式: ADL_{timeframe}, 无参数
            if len(parts) == 1: # 只包含前缀
                 return [] # 返回空列表表示没有参数

        elif indicator_key == 'vwap' and parts[0] == 'VWAP':
             # VWAP 列名模式: VWAP_{timeframe} 或 VWAP_{anchor}_{timeframe}
             # 检查 parts 的长度来区分是否有 anchor 参数
             if len(parts) == 1: # VWAP_{timeframe} 模式，无参数
                 return [] # 返回空列表表示没有参数
             elif len(parts) >= 2: # VWAP_{anchor}_{timeframe} 模式，anchor 作为参数
                 # 假设第一个参数是 anchor，作为字符串返回
                 return [parts[1]] # 返回包含 anchor 字符串的列表
             return None # 格式不匹配

        elif indicator_key == 'ichimoku' and parts[0] in ['TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU_A', 'SENKOU_B']:
            # Ichimoku 列名模式复杂，根据前缀和参数数量尝试解析
            # TENKAN_period, KIJUN_period, CHIKOU_period, SENKOU_A_tenkan_kijun, SENKOU_B_period
            if parts[0] in ['TENKAN', 'KIJUN', 'CHIKOU', 'SENKOU_B'] and len(parts) >= 2:
                 # These lines have only one period parameter
                 return [int(parts[1])] # Try to parse single integer period
            elif parts[0] == 'SENKOU_A' and len(parts) >= 3:
                 # Senkou Span A has two period parameters
                 return [int(parts[1]), int(parts[2])] # Try to parse two integer periods
            return None # Parameter format or count mismatch

        elif indicator_key == 'mom' and parts[0] == 'MOM':
            # MOM 列名模式: MOM_period
             if len(parts) >= 2: # At least prefix and 1 parameter
                 # Try to parse parameter part as integer
                 return [int(parts[1])]

        elif indicator_key == 'willr' and parts[0] == 'WILLR':
            # WILLR 列名模式: WILLR_period
             if len(parts) >= 2: # At least prefix and 1 parameter
                 # Try to parse parameter part as integer
                 return [int(parts[1])]

        elif indicator_key == 'cmf' and parts[0] == 'CMF':
            # CMF 列名模式: CMF_period
             if len(parts) >= 2: # At least prefix and 1 parameter
                 # Try to parse parameter part as integer
                 return [int(parts[1])]

        elif indicator_key == 'obv' and parts[0] == 'OBV':
             # OBV 列名模式: OBV_{timeframe}, no parameters
            if len(parts) == 1: # Only prefix
                 return [] # Return empty list for no parameters
        elif indicator_key == 'obv' and parts[0] == 'OBV_MA':
             # OBV_MA 列名模式: OBV_MA_period_{timeframe}
             if len(parts) >= 2: # At least prefix 'OBV_MA' and 1 parameter
                  # Try to parse parameter part as integer
                  return [int(parts[1])] # Return list containing OBV_MA period integer

        elif indicator_key == 'kc' and parts[0] in ['KCL', 'KCM', 'KCU']:
             # KC 列名模式: KCL/KCM/KCU_ema_period_atr_period
             if len(parts) >= 3: # At least prefix and 2 parameters
                  # Try to parse parameter parts as integers
                  return [int(parts[1]), int(parts[2])]

        elif indicator_key == 'hv' and parts[0] == 'HV':
            # HV 列名模式: HV_period
             if len(parts) >= 2: # At least prefix and 1 parameter
                 # Try to parse parameter part as integer
                 return [int(parts[1])]

        elif indicator_key == 'vroc' and parts[0] == 'VROC':
            # VROC 列名模式: VROC_period
             if len(parts) >= 2: # At least prefix and 1 parameter
                 # Try to parse parameter part as integer
                 return [int(parts[1])]

        elif indicator_key == 'aroc' and parts[0] == 'AROC':
            # AROC 列名模式: AROC_period
             if len(parts) >= 2: # At least prefix and 1 parameter
                 # Try to parse parameter part as integer
                 return [int(parts[1])]

        # Pivot column names (PP, R1, S1 etc.) do not contain parameters, only base name and suffix, no need to parse parameters via this function
        # Close column name also does not need parameter parsing via this function

        # If column name pattern is not recognized, log debug info and return None
        print(f"列名 '{col_name}' 不符合指标 '{indicator_key}' 期望的参数模式，或参数数量/类型不匹配 (suffix: {tf_suffix}).")
        return None
    except (ValueError, IndexError) as e:
        # If an error occurs during parameter conversion or index access, log debug info and return None
        print(f"从列名 '{col_name}' 解析参数失败 (indicator: {indicator_key}, suffix: {tf_suffix}). 错误: {e}", exc_info=True)
        return None # Parameter conversion failed or index out of bounds

def calculate_all_indicator_scores(data: pd.DataFrame, bs_params: Dict, indicator_configs: List[Dict], naming_config: Dict) -> pd.DataFrame:
    """
    根据配置计算所有指定指标在不同时间框架下的评分 (0-100)。
    函数会遍历 base_scoring 参数中指定的需要评分的指标和时间框架，
    根据 naming_config 和 indicator_configs 在输入的 DataFrame 中查找对应的指标数据列，
    然后调用相应的评分计算函数来得到评分。

    :param data: 包含所有原始 OHLCV 数据和已计算指标的 DataFrame。
                 列名应包含时间级别后缀和可能的计算参数，例如 'close_15', 'MACD_12_26_9_30'。
    :param bs_params: base_scoring 参数字典，包含 'score_indicators' (需要评分的指标键列表),
                      'timeframes' (需要计算评分的时间框架列表)，以及各指标的评分逻辑参数。
    :param indicator_configs: 由指标服务生成的配置列表，包含每个指标的计算参数和输出列名信息，
                              用于辅助查找 DataFrame 中的列。
    :param naming_config: 包含列命名规范的字典，用于正确构建和匹配 DataFrame 中的列名模式。
    :return: 返回一个 DataFrame，其列名为 SCORE_{指标名}_{时间级别} 的评分列。
             如果某个指标在某个时间框架的数据未找到或计算失败，对应的评分列将填充默认中性分 50.0。
    """
    # 初始化用于存储所有评分结果的 DataFrame，索引与输入数据相同
    scoring_results = pd.DataFrame(index=data.index)
    # 如果输入 DataFrame 是空的，记录警告并直接返回空结果 DataFrame
    if data.empty:
         logger.warning("输入 DataFrame 为空，无法计算指标评分。")
         print("DEBUG: 输入 DataFrame 为空，无法计算指标评分。")
         return scoring_results
    # 从 bs_params 中获取需要评分的指标键列表和时间框架列表
    # 这两行确保 score_indicators_keys 和 score_timeframes 变量被定义
    score_indicators_keys = bs_params.get('score_indicators', [])
    score_timeframes = bs_params.get('timeframes', [])
    # 如果未配置需要评分的指标或时间框架，记录警告并返回空结果 DataFrame
    if not score_indicators_keys or not score_timeframes:
        logger.warning("未配置需要评分的指标或时间框架 (base_scoring.score_indicators 或 base_scoring.timeframes)。")
        print("DEBUG: 未配置需要评分的指标或时间框架。")
        return scoring_results
    # 调试输出：开始计算指标评分的信息，显示需要评分的指标和时间框架
    print(f"DEBUG: 开始计算指标评分，指标: {score_indicators_keys}, 时间框架: {score_timeframes}")
    # 从 naming_config 中获取命名规范字典，包括指标、OHLCV 和时间框架的命名约定
    indicator_naming_conv = naming_config.get('indicator_naming_conventions', {})
    ohlcv_naming_conv = naming_config.get('ohlcv_naming_convention', {}) # 用于查找 close 列模式
    timeframe_naming_conv = naming_config.get('timeframe_naming_convention', {})
    # 增加类型检查，确保获取到的是字典，如果不是则初始化为空字典
    if not isinstance(indicator_naming_conv, dict): indicator_naming_conv = {}
    if not isinstance(ohlcv_naming_conv, dict): ohlcv_naming_conv = {}
    if not isinstance(timeframe_naming_conv, dict): timeframe_naming_conv = {}
    # --- 构建从 indicator_configs 到实际列名的映射 ---
    # 这个映射用于优先查找由 IndicatorService 生成的精确列名
    # 映射结构: { (indicator_key, internal_key, tf_str): actual_col_name }
    # 对于 Pivot levels，结构是 { (indicator_key, 'pivot_levels', tf_str): {level_key: actual_col_name} }
    config_to_actual_col_map: Dict[Tuple[str, str, str], Union[str, Dict[str, str]]] = {}
    # 遍历 indicator_configs 列表，这些配置包含了指标计算的详细信息和输出列名
    if isinstance(indicator_configs, list):
        print(f"DEBUG: 正在处理 indicator_configs ({len(indicator_configs)} 项)...")
        for config in indicator_configs:
            # 跳过非字典项的配置
            if not isinstance(config, dict): continue
            # 获取指标名称（转为小写以便比较）和时间框架列表
            indicator_name = config.get('name', '').lower()
            timeframes_list = config.get('timeframes', [])
            # 确保 timeframes_list 是列表，如果不是则尝试转换为列表
            if isinstance(timeframes_list, str): timeframes_list = [timeframes_list]
            # 如果 timeframes 不是列表，跳过此配置项
            if not isinstance(timeframes_list, list): continue
            # 从 naming_config 中获取当前指标的命名配置
            indi_naming_conf = indicator_naming_conv.get(indicator_name.upper(), {})
            # 获取当前指标的输出列模式列表
            output_cols_patterns = indi_naming_conf.get('output_columns', [])
            # 确保 output_cols_patterns 是列表
            if not isinstance(output_cols_patterns, list): output_cols_patterns = []
            # 特殊处理 Pivot levels 指标，它的数据结构不同于标准指标列
            if indicator_name == 'pivot':
                 # 假设 indicator_configs 中包含 'pivot_levels_data' 字段，存储了各时间框架的枢轴点级别列名
                 pivot_levels_data = config.get('pivot_levels_data')
                 if isinstance(pivot_levels_data, dict):
                      print(f"DEBUG: 处理 Pivot levels 配置数据: {pivot_levels_data.keys()}")
                      # 遍历配置中的时间框架
                      for tf_conf in timeframes_list:
                           tf_str = str(tf_conf)
                           # 如果 pivot_levels_data 中有当前时间框架的数据
                           if tf_str in pivot_levels_data:
                                level_data_for_tf = pivot_levels_data[tf_str]
                                # 如果该时间框架的数据是字典（level_key -> actual_col_name）
                                if isinstance(level_data_for_tf, dict):
                                     # 存储 level_key 到实际列名的字典映射到 config_to_actual_col_map
                                     config_to_actual_col_map[(indicator_name, 'pivot_levels', tf_str)] = level_data_for_tf
                                     # 调试输出：添加 Pivot levels 配置映射
                                     print(f"DEBUG: 添加 Pivot levels 配置映射，时间框架 {tf_str}，键: {level_data_for_tf.keys()}")
                                else:
                                     print(f"DEBUG: Pivot levels 数据 for tf {tf_str} 不是字典。")
                           else:
                                print(f"DEBUG: Pivot levels 数据中没有时间框架 {tf_str} 的条目。")
                 else:
                      print(f"DEBUG: Pivot levels 配置数据不是字典或不存在。")
                 # 处理完 Pivot levels 后跳到下一个配置项
                 continue
            # 处理其他标准指标
            # actual_output_columns 是 IndicatorService 生成的实际列名列表
            actual_output_columns = config.get('output_columns', [])
            # 确保 actual_output_columns 是列表
            if isinstance(actual_output_columns, str): actual_output_columns = [actual_output_columns]
            if not isinstance(actual_output_columns, list): continue # 如果 actual_output_columns 不是列表，跳过
            # 尝试根据 naming_config 和 parse_col_params 将实际列名映射回内部键
            for actual_col_name in actual_output_columns:
                 # 跳过非字符串的列名
                 if not isinstance(actual_col_name, str): continue
                 # 尝试从实际列名中解析时间框架后缀
                 found_tf_suffix = None
                 original_tf_str_matched = None # 记录匹配到的原始时间框架字符串
                 # 遍历配置中的时间框架列表，查找匹配的后缀
                 for tf_conf in timeframes_list:
                      tf_str = str(tf_conf)
                      # 获取当前时间框架可能的后缀模式列表
                      possible_suffixes = timeframe_naming_conv.get('patterns', {}).get(tf_str.lower(), [tf_str])
                      # 确保 possible_suffixes 是列表
                      if isinstance(possible_suffixes, str): possible_suffixes = [possible_suffixes]
                      if not isinstance(possible_suffixes, list): continue
                      # 将可能的后缀转换为字符串列表
                      possible_suffixes = [str(s) for s in possible_suffixes]
                      # 检查实际列名是否以任一可能的后缀结尾
                      for suffix in possible_suffixes:
                           if actual_col_name.endswith(f"_{suffix}"):
                                found_tf_suffix = suffix
                                original_tf_str_matched = tf_str # 记录匹配到的原始时间框架
                                break
                      # 如果找到后缀，跳出时间框架循环
                      if found_tf_suffix: break
                 # 如果基于配置的时间框架后缀未找到，尝试从列名末尾猜测
                 if not found_tf_suffix:
                      parts = actual_col_name.split('_')
                      if len(parts) > 1:
                           guessed_suffix = parts[-1]
                           # 检查猜测的后缀是否对应配置中的任一时间框架
                           is_valid_guessed_suffix = False
                           for tf_conf in timeframes_list:
                                tf_str = str(tf_conf)
                                possible_suffixes = timeframe_naming_conv.get('patterns', {}).get(tf_str.lower(), [tf_str])
                                if isinstance(possible_suffixes, str): possible_suffixes = [possible_suffixes]
                                if not isinstance(possible_suffixes, list): continue
                                possible_suffixes = [str(s) for s in possible_suffixes]
                                if guessed_suffix in possible_suffixes:
                                     is_valid_guessed_suffix = True
                                     found_tf_suffix = guessed_suffix
                                     original_tf_str_matched = tf_str # 记录匹配到的原始时间框架
                                     break
                           # 如果猜测的后缀无效，则重置 found_tf_suffix
                           if not is_valid_guessed_suffix:
                                found_tf_suffix = None
                                original_tf_str_matched = None
                 # 如果仍然无法确定时间框架后缀，记录警告并跳过此列
                 if not found_tf_suffix:
                      # 调试输出：无法确定时间框架后缀
                      print(f"WARNING: 无法从指标配置中确定列 '{actual_col_name}' 的时间框架后缀。")
                      continue # 没有有效后缀，无法映射此列
                 # 尝试从 naming_config 中查找与实际列名匹配的模式，并获取其 internal_key
                 matched_internal_key = None
                 # 遍历 naming_config 中当前指标的 output_columns 模式
                 for col_conf in output_cols_patterns:
                      # 确保配置项是字典且包含 'name_pattern' 和 'internal_key'
                      if isinstance(col_conf, dict) and 'name_pattern' in col_conf and 'internal_key' in col_conf:
                           pattern = col_conf['name_pattern']
                           internal_key_from_naming = col_conf['internal_key']
                           # 尝试使用 parse_col_params 和找到的后缀反向匹配模式
                           # parse_col_params 需要能够从 actual_col_name 解析出 pattern 中的参数值
                           # 例如：actual_col_name = 'MACD_12_26_9_5', pattern = 'MACD_{period_fast}_{period_slow}_{signal_period}'
                           # parse_col_params 应该返回 {'period_fast': 12, 'period_slow': 26, 'signal_period': 9}
                           # 然后用这些参数和 found_tf_suffix 格式化 pattern 看是否等于 actual_col_name
                           # 注意：这里的 parse_col_params 实现需要支持这种反向解析
                           try:
                                # 假设 parse_col_params 返回一个字典 {param_name: param_value} 或 None
                                params = parse_col_params(actual_col_name, indicator_name, found_tf_suffix, pattern) # 假设 parse_col_params 接受 pattern 作为参数
                                if params is not None:
                                     temp_format_params = params.copy()
                                     temp_format_params['timeframe'] = found_tf_suffix # 添加时间框架参数
                                     # 尝试格式化当前模式
                                     expected_col_from_pattern = pattern.format(**temp_format_params).replace('__', '_').strip('_')
                                     # 如果格式化后的模式等于实际列名，则认为匹配成功
                                     if expected_col_from_pattern == actual_col_name:
                                          matched_internal_key = internal_key_from_naming
                                          print(f"DEBUG: 列 '{actual_col_name}' 匹配 naming_config 模式 '{pattern}'，映射到内部键 '{matched_internal_key}'")
                                          break # 找到匹配的 internal_key，跳出模式循环
                                # else:
                                #      print(f"DEBUG: 列 '{actual_col_name}' 未能通过模式 '{pattern}' 反向解析匹配 (参数解析失败或不匹配)。")
                           except KeyError as e:
                                # 如果模式中包含 parse_col_params 未提供的参数，格式化会失败
                                # print(f"DEBUG: 反向匹配模式 '{pattern}' 时缺少参数 {e}")
                                pass # 继续尝试下一个模式
                           except Exception as e:
                                # 捕获其他可能的错误
                                # print(f"DEBUG: 反向匹配模式 '{pattern}' 时发生未知错误: {e}")
                                pass # 继续尝试下一个模式

                 # 如果找到匹配的内部键
                 if matched_internal_key:
                      # 存储映射: (indicator_key, internal_key, tf_str) -> actual_col_name
                      # 使用之前匹配到的原始时间框架字符串作为 tf_str，以与 score_timeframes 保持一致
                      if original_tf_str_matched:
                           config_to_actual_col_map[(indicator_name, matched_internal_key, original_tf_str_matched)] = actual_col_name
                           # 调试输出：映射配置列
                           print(f"DEBUG: 映射配置列: ({indicator_name}, {matched_internal_key}, {original_tf_str_matched}) -> '{actual_col_name}'")
                      else:
                           print(f"WARNING: 找到匹配的内部键 '{matched_internal_key}'，但无法确定原始时间框架字符串。跳过映射。")
                 # else:
                      # 调试输出：无法将实际列名映射回内部键 (如果不是必需的，这里可以忽略)
                      # print(f"DEBUG: 无法将实际列 '{actual_col_name}' 映射回指标 '{indicator_name}' 的内部键。")

    else:
        print("DEBUG: indicator_configs 不是列表或为空。")
    # --- 结束构建配置映射 ---
    # 遍历需要评分的每个指标键 (来自 bs_params)
    for indicator_key in score_indicators_keys:
        # 从 indicator_scoring_info 获取该指标的配置信息
        info = indicator_scoring_info.get(indicator_key)
        # 如果指标配置信息不存在，记录警告并跳过此指标的评分计算
        if not info:
             logger.warning(f"指标 '{indicator_key}' 未找到对应的评分函数定义或配置，跳过评分计算。")
             print(f"DEBUG: 指标 '{indicator_key}' 未找到评分配置，跳过。")
             continue
        # 获取评分函数、所需内部键、参数传递风格、参数映射、默认值和模式/参数映射
        score_func = info.get('func')
        required_score_keys = info.get('required_keys', [])
        param_passing_style = info.get('param_passing_style', 'dict')
        bs_param_key_to_score_func_arg = info.get('bs_param_key_to_score_func_arg', {})
        defaults = info.get('defaults', {})
        # 修改: 获取新的 key_patterns 结构
        key_patterns_info = info.get('key_patterns', {})
        # 如果评分函数不存在，记录警告并跳过此指标的评分计算
        if score_func is None:
             logger.warning(f"指标 '{indicator_key}' 没有关联的评分函数，跳过评分计算。")
             print(f"DEBUG: 指标 '{indicator_key}' 没有评分函数，跳过。")
             continue
        # 如果未配置所需的内部键，记录警告并跳过此指标的评分计算
        if not required_score_keys:
             logger.warning(f"指标 '{indicator_key}' 未配置所需的内部键 (required_keys)，跳过评分计算。")
             print(f"DEBUG: 指标 '{indicator_key}' 未配置 required_keys，跳过。")
             continue
        # 遍历需要计算评分的每个时间框架 (来自 bs_params)
        for tf_score in score_timeframes:
            # 初始化字典用于存储找到的内部键对应的实际列名或 Series 字典
            indicator_cols_for_score: Dict[str, str | Dict[str, pd.Series]] = {}
            # 标记是否成功找到当前指标和时间框架所需的所有数据列
            found = False
            # 确保时间框架是字符串
            tf_score_str = str(tf_score)
            print(f"\nDEBUG: 正在搜索指标 '{indicator_key}' 在时间框架 {tf_score_str} 的列...")
            # --- 优先尝试使用 config_to_actual_col_map 中提供的列名映射 ---
            temp_cols_from_config: Dict[str, str | Dict[str, pd.Series]] = {}
            print(f"DEBUG: 尝试通过 config_to_actual_col_map 查找所需键: {required_score_keys}")
            # 遍历评分所需的内部键
            for internal_key in required_score_keys:
                 # OBV_MA 和 Pivot levels 在配置映射阶段可能不会被所有 indicator_configs 包含
                 # 只有当 internal_key 是必需的（不在可选列表中）且不在 config_to_actual_col_map 中时，才需要标记配置查找失败
                 # 'obv_ma' 是可选键，'pivot_levels' 是特殊结构，它们的缺失不在此阶段标记为配置查找失败
                 if internal_key not in ['obv_ma', 'pivot_levels']:
                    config_key = (indicator_key, internal_key, tf_score_str)
                    print(f"DEBUG: 查找 config_to_actual_col_map 中的键: {config_key}")
                    # 检查 config_to_actual_col_map 中是否存在此组合
                    if config_key in config_to_actual_col_map:
                         actual_data_source = config_to_actual_col_map[config_key]
                         # 对于标准列，数据源是实际列名字符串
                         if isinstance(actual_data_source, str) and actual_data_source in data.columns:
                              temp_cols_from_config[internal_key] = actual_data_source
                              print(f"DEBUG: 通过配置找到列 ({indicator_key}, {internal_key}, {tf_score_str}): '{actual_data_source}'")
                         else:
                              print(f"DEBUG: 配置中的列 '{actual_data_source}' 未在数据中找到 ({indicator_key}, {internal_key}, {tf_score_str})。配置查找此键失败。")
                              # 如果必需的配置列未在数据中找到，此方法失败
                              temp_cols_from_config.pop(internal_key, None) # 移除无效的条目
                              # break # No need to break, let it collect all found and check later
                    else:
                         print(f"DEBUG: 必需的键 '{internal_key}' 未在配置映射中找到 ({indicator_key}, {tf_score_str})。配置查找此键失败。")
                         # Required key not found in config map, this method fails for this key
                         # break # No need to break
                 elif internal_key == 'pivot_levels': # 特殊处理 pivot_levels
                     config_key = (indicator_key, internal_key, tf_score_str)
                     print(f"DEBUG: 查找 config_to_actual_col_map 中的键: {config_key}")
                     if config_key in config_to_actual_col_map:
                          actual_data_source = config_to_actual_col_map[config_key]
                          if isinstance(actual_data_source, dict):
                              # 检查字典中的所有列名是否存在于 data DataFrame 中
                              all_pivot_cols_exist = all(col_name in data.columns for col_name in actual_data_source.values())
                              if all_pivot_cols_exist:
                                   temp_cols_from_config[internal_key] = {level_key: data[col_name] for level_key, col_name in actual_data_source.items()}
                                   print(f"DEBUG: 通过配置找到 Pivot levels，时间框架 {tf_score_str}。键: {actual_data_source.keys()}")
                              else:
                                   print(f"DEBUG: 配置中的 Pivot levels 列未在数据中找到，时间框架 {tf_score_str}。配置: {actual_data_source}. 配置查找此键失败。")
                          else:
                               print(f"DEBUG: 时间框架 {tf_score_str} 的 Pivot levels 配置映射不是字典。配置查找此键失败。")
                     else:
                          print(f"DEBUG: Pivot levels 键 '{internal_key}' 未在配置映射中找到 ({indicator_key}, {tf_score_str})。配置查找此键失败。")
                 elif internal_key == 'obv_ma': # 特殊处理可选的 obv_ma
                     config_key = (indicator_key, internal_key, tf_score_str)
                     print(f"DEBUG: 查找 config_to_actual_col_map 中的可选键: {config_key}")
                     if config_key in config_to_actual_col_map:
                          actual_data_source = config_to_actual_col_map[config_key]
                          if isinstance(actual_data_source, str) and actual_data_source in data.columns:
                               temp_cols_from_config[internal_key] = actual_data_source
                               print(f"DEBUG: 通过配置找到可选列 ({indicator_key}, {internal_key}, {tf_score_str}): '{actual_data_source}'")
                          else:
                               print(f"DEBUG: 配置中的可选列 '{actual_data_source}' 未在数据中找到 ({indicator_key}, {internal_key}, {tf_score_str})。")
                     else:
                          print(f"DEBUG: 可选键 '{internal_key}' 未在配置映射中找到 ({indicator_key}, {tf_score_str})。")


            # Check if all REQUIRED keys (excluding optional/special handled) were found via config map
            required_keys_base = [k for k in required_score_keys if k not in ['obv_ma', 'pivot_levels']] # 排除可选的 obv_ma 和特殊处理的 pivot_levels
            all_base_required_found = all(k in temp_cols_from_config for k in required_keys_base)

            pivot_levels_found_ok = True # Assume OK unless pivot_levels is required and not found as a valid dict
            if 'pivot_levels' in required_score_keys:
                 # If pivot_levels is required, check if it's found in temp_cols_from_config and is a valid non-empty dict
                 pivot_levels_found_ok = isinstance(temp_cols_from_config.get('pivot_levels'), dict) and bool(temp_cols_from_config.get('pivot_levels')) # 检查是否找到且是字典且非空

            if all_base_required_found and pivot_levels_found_ok:
                 indicator_cols_for_score = temp_cols_from_config
                 found = True
                 print(f"DEBUG: 通过配置成功找到指标 '{indicator_key}' 在时间框架 {tf_score_str} 的所有必需列和结构。") # 修改日志
            else:
                 print(f"DEBUG: 指标 '{indicator_key}' 在时间框架 {tf_score_str} 的配置查找失败 (必需键或必需特殊结构缺失)。尝试回退查找。") # 修改日志

            # --- 如果配置查找失败，尝试使用 indicator_scoring_info 的 key_patterns 构建列名进行回退查找 ---
            if not found:
                # 根据当前 tf_score 和命名规范定义可能的时框架后缀列表
                current_tf_possible_suffixes = []
                tf_score_str_lower = tf_score_str.lower()
                timeframe_patterns = timeframe_naming_conv.get('patterns', {})
                if isinstance(timeframe_patterns, dict):
                    patterns_for_tf = timeframe_patterns.get(tf_score_str_lower, [tf_score_str])
                    if isinstance(patterns_for_tf, str): patterns_for_tf = [patterns_for_tf]
                    if isinstance(patterns_for_tf, list):
                        current_tf_possible_suffixes = [str(p) for p in patterns_for_tf]
                # 确保原始的 tf_score_str 始终是回退检查的可能后缀之一
                if tf_score_str not in current_tf_possible_suffixes:
                     current_tf_possible_suffixes.append(tf_score_str)

                print(f"DEBUG: 尝试对时间框架 {tf_score_str} 进行回退查找，可能后缀: {current_tf_possible_suffixes}")

                # 遍历当前 tf_score 的可能后缀
                for tf_suffix in current_tf_possible_suffixes:
                    # 临时存储为此后缀找到的列
                    temp_cols_found: Dict[str, str | Dict[str, pd.Series]] = {}
                    # 标记是否为此后缀找到所有必需列
                    all_required_found_for_suffix = True
                    print(f"DEBUG: 回退查找，当前后缀: '{tf_suffix}'")

                    # 遍历评分所需的内部键
                    for internal_key in required_score_keys:
                        print(f"DEBUG: 回退查找，正在搜索内部键: '{internal_key}'")

                        # 特殊处理 'close' 列 (OHLCV 的一部分)
                        if internal_key == 'close':
                            close_pattern = None
                            ohlcv_output_cols_conf = ohlcv_naming_conv.get('output_columns', [])
                            if isinstance(ohlcv_output_cols_conf, list):
                                 # 在 OHLCV 配置中查找 internal_key 为 'close' 的模式
                                 for col_conf in ohlcv_output_cols_conf:
                                     if isinstance(col_conf, dict) and col_conf.get('internal_key') == 'close':
                                          close_pattern = col_conf.get('name_pattern')
                                          break
                            if close_pattern:
                                # 构建期望的 'close' 列名
                                expected_col_name = f"{close_pattern}_{tf_suffix}"
                                print(f"DEBUG: 回退查找: 查找 'close' 列，期望列名: '{expected_col_name}'")
                                if expected_col_name in data.columns:
                                    temp_cols_found[internal_key] = expected_col_name
                                    print(f"DEBUG: 回退查找: 找到 'close' 列: '{expected_col_name}'")
                                else:
                                    # 如果必需的 'close' 列未找到，标记此后缀查找失败
                                    all_required_found_for_suffix = False
                                    print(f"DEBUG: 必需的 'close' 列 '{expected_col_name}' 未找到，后缀 '{tf_suffix}'。")
                                    break
                            else:
                                # 如果未找到 'close' 的命名规范（即 naming_config 中 OHLCV output_columns 缺少 internal_key='close' 的配置）
                                logger.warning(f"回退查找: 未找到 'close' 的命名规范 (naming_config 中 OHLCV output_columns 缺少 internal_key='close' 的配置)。无法为时间框架 {tf_score} 找到 close 列。")
                                print(f"DEBUG: 回退查找: 未找到 'close' 的命名规范。")
                                all_required_found_for_suffix = False
                                break

                        # 特殊处理 'pivot_levels' (枢轴点)
                        elif indicator_key == 'pivot' and internal_key == 'pivot_levels':
                             # Pivot levels 是一个 Series 字典，而不是单个列
                             pivot_naming_convention = indicator_naming_conv.get('PIVOT', {})
                             # 获取枢轴点级别的基础名称列表
                             pivot_cols_base = pivot_naming_convention.get('levels', ["PP", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4", "F_R1", "F_R2", "F_R3", "F_S1", "F_S2", "F_S3"])
                             # 获取枢轴点级别的命名模式
                             pivot_level_pattern = pivot_naming_convention.get('pattern', "{level}_{timeframe}") # 默认模式
                             # 初始化字典用于存储找到的枢轴点级别 Series
                             pivot_levels_series_dict_for_score: Dict[str, pd.Series] = {}
                             all_pivot_levels_found_as_series = True
                             print(f"DEBUG: 回退查找: 查找 'pivot_levels'，基础级别: {pivot_cols_base}, 模式: '{pivot_level_pattern}'")

                             # 遍历枢轴点级别的基础名称
                             for p_base in pivot_cols_base:
                                 try:
                                     # 使用模式和当前后缀构建枢轴点级别列名
                                     col_name = pivot_level_pattern.format(level=p_base, timeframe=tf_suffix)
                                     col_name = col_name.replace('__', '_').strip('_')
                                 except KeyError as e:
                                     logger.warning(f"Pivot levels 模式 '{pivot_level_pattern}' 缺少格式化参数: {e}. 无法构建列名。")
                                     print(f"DEBUG: Pivot levels 模式 '{pivot_level_pattern}' 缺少格式化参数: {e}. 无法构建列名。")
                                     all_pivot_levels_found_as_series = False
                                     break
                                 except Exception as e:
                                     logger.warning(f"格式化 Pivot levels 模式 '{pivot_level_pattern}' 时发生未知错误: {e}. 无法构建列名。")
                                     print(f"DEBUG: 格式化 Pivot levels 模式 '{pivot_level_pattern}' 时发生未知错误: {e}. 无法构建列名。")
                                     all_pivot_levels_found_as_series = False
                                     break

                                 print(f"DEBUG: 回退查找: 查找枢轴点级别列: '{col_name}'")
                                 if col_name in data.columns:
                                     pivot_levels_series_dict_for_score[p_base] = data[col_name]
                                     print(f"DEBUG: 回退查找: 找到枢轴点级别列: '{col_name}'")
                                 else:
                                     # 如果任一必需的枢轴点级别缺失，标记查找失败
                                     all_pivot_levels_found_as_series = False
                                     print(f"DEBUG: 枢轴点级别列 '{col_name}' 未找到，后缀 '{tf_suffix}'。")
                                     break

                             # 如果找到所有枢轴点级别 Series
                             if all_pivot_levels_found_as_series:
                                 temp_cols_found[internal_key] = pivot_levels_series_dict_for_score
                                 print(f"DEBUG: 找到后缀 '{tf_suffix}' 的所有枢轴点级别。")
                             else:
                                 # 如果未完全找到枢轴点级别，标记此后缀查找失败
                                 all_required_found_for_suffix = False
                                 break

                        # 处理其他标准指标组件 (使用新的 key_patterns 结构进行查找)
                        else:
                             # 修改: 从 info['key_patterns'] 中获取模式和参数映射
                             key_pattern_info = key_patterns_info.get(internal_key)
                             # 如果在 indicator_scoring_info 的 key_patterns 中未找到此 internal_key 的配置
                             if not key_pattern_info or not isinstance(key_pattern_info, dict):
                                  # 检查是否为可选键 (如 obv_ma)，如果是则不标记失败
                                  if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                       logger.warning(f"回退查找: 未在 indicator_scoring_info['{indicator_key}']['key_patterns'] 中找到内部键 '{internal_key}' 的模式配置 (非可选键)。") # 修改日志输出
                                       print(f"DEBUG: 回退查找: 未在 key_patterns 中找到内部键 '{internal_key}' 的模式配置 (非可选键)。") # 修改日志输出
                                       all_required_found_for_suffix = False
                                       break # 必需键的模式配置缺失
                                  else:
                                       # 可选键 obv_ma 的模式配置缺失，不标记失败，继续查找其他必需键
                                       print(f"DEBUG: 回退查找: 可选键 '{internal_key}' 在 key_patterns 中没有模式配置，跳过此键的查找。")
                                       continue # 跳过此可选键，继续下一个 required_key

                             pattern = key_pattern_info.get('pattern')
                             params_map = key_pattern_info.get('params_map', {}) # 获取参数映射，默认为空字典

                             # 如果模式字符串不存在或不是字符串
                             if not pattern or not isinstance(pattern, str):
                                  # 检查是否为可选键 (如 obv_ma)，如果是则不标记失败
                                  if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                       logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 在 key_patterns 中的模式配置无效 (非可选键)。") # 修改日志输出
                                       print(f"DEBUG: 回退查找: 内部键 '{internal_key}' 的模式配置无效 (非可选键)。") # 修改日志输出
                                       all_required_found_for_suffix = False
                                       break # 必需键的模式配置无效
                                  else:
                                       # 可选键 obv_ma 的模式字符串无效，不标记失败
                                       print(f"DEBUG: 回退查找: 可选键 '{internal_key}' 的模式字符串无效，跳过此键的查找。")
                                       continue # 跳过此可选键，继续下一个 required_key


                             # --- 构建期望的列名并检查是否存在 (使用 key_patterns 中的信息) ---
                             format_params: Dict[str, Any] = {'timeframe': tf_suffix} # 总是包含时间框架参数
                             params_found_for_pattern = True
                             print(f"DEBUG: 回退查找: 构建期望列名，内部键 '{internal_key}'，模式: '{pattern}'，模式参数映射: {params_map}") # 修改日志输出

                             # 根据模式参数映射，从 bs_params 或 defaults 获取参数值
                             for pattern_param_name, bs_param_key in params_map.items():
                                  param_value = bs_params.get(bs_param_key, defaults.get(bs_param_key, None))
                                  if param_value is not None:
                                       # 确保浮点数使用正确的格式化，例如:.1f
                                       # 这里只存储值，格式化在 pattern.format 中完成
                                       format_params[pattern_param_name] = param_value
                                       print(f"DEBUG: 回退查找: 获取模式参数 '{pattern_param_name}' = {param_value} (来自 bs_key '{bs_param_key}')")
                                  else:
                                       # 如果 params_map 中列出了，且 pattern 中包含该占位符，则认为是必需的
                                       if '{' + pattern_param_name + '}' in pattern:
                                            logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 的模式 '{pattern}' 需要参数 '{pattern_param_name}' (对应 bs_key '{bs_param_key}')，但在 bs_params 和 defaults 中未找到。")
                                            print(f"DEBUG: 回退查找: 模式 '{pattern}' 缺少参数 '{pattern_param_name}' (bs_key '{bs_param_key}')。")
                                            params_found_for_pattern = False
                                            break
                                       else:
                                            # 如果 pattern 中没有这个占位符，那么这个参数不是必需的，跳过
                                            print(f"DEBUG: 模式 '{pattern}' 不需要参数 '{pattern_param_name}'，跳过获取。")


                             # 如果成功获取到格式化模式所需的所有参数
                             if params_found_for_pattern:
                                  expected_col_name = None
                                  try:
                                       # 使用收集到的参数格式化模式
                                       expected_col_name = pattern.format(**format_params)
                                       expected_col_name = expected_col_name.replace('__', '_').strip('_')
                                       print(f"DEBUG: 回退查找: 构建期望列名 -> '{expected_col_name}'")
                                  except KeyError as e:
                                       logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 的模式 '{pattern}' 格式化失败，缺少参数: {e}.")
                                       print(f"DEBUG: 回退查找: 模式 '{pattern}' 格式化失败，缺少参数: {e}.")
                                       all_required_found_for_suffix = False # 格式化失败，此后缀查找失败
                                       break # 格式化失败，跳出 internal_key 循环
                                  except ValueError as e:
                                       logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 的模式 '{pattern}' 格式化值错误: {e}. 参数: {format_params}.")
                                       print(f"DEBUG: 回退查找: 模式 '{pattern}' 格式化值错误: {e}. 参数: {format_params}.")
                                       all_required_found_for_suffix = False # 格式化失败，此后缀查找失败
                                       break # 格式化失败，跳出 internal_key 循环
                                  except Exception as e:
                                       logger.warning(f"回退查找: 格式化模式 '{pattern}' 时发生未知错误: {e}.")
                                       print(f"DEBUG: 回退查找: 格式化模式 '{pattern}' 时发生未知错误: {e}.")
                                       all_required_found_for_suffix = False # 格式化失败，此后缀查找失败
                                       break # 格式化失败，跳出 internal_key 循环

                                  # 如果期望列名成功构建且存在于 DataFrame 中
                                  if expected_col_name and expected_col_name in data.columns:
                                      temp_cols_found[internal_key] = expected_col_name
                                      print(f"DEBUG: 通过回退找到内部键 '{internal_key}' 的列，后缀 '{tf_suffix}': '{expected_col_name}'")
                                  else:
                                      # 如果期望列名未找到，标记此后缀查找失败
                                      # 检查是否为可选键 (如 obv_ma)，如果是则不标记失败
                                      if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                           all_required_found_for_suffix = False
                                           print(f"DEBUG: 通过回退未找到内部键 '{internal_key}' 的必需列，后缀 '{tf_suffix}'。期望列名: '{expected_col_name}'")
                                           break # 必需列未找到，此后缀查找失败
                                      else:
                                           # 可选键 obv_ma 未找到，不标记失败，继续查找其他必需键
                                           print(f"DEBUG: 通过回退未找到内部键 '{internal_key}' 的可选列，后缀 '{tf_suffix}'。期望列名: '{expected_col_name}'。")


                             else:
                                  # 如果缺少格式化模式所需的参数，标记此后缀查找失败
                                  # 检查是否为可选键 (如 obv_ma)，如果是则不标记失败
                                  if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                       all_required_found_for_suffix = False
                                       print(f"DEBUG: 回退查找: 缺少格式化模式所需的参数，后缀 '{tf_suffix}' 查找失败。")
                                       break # 缺少参数，跳出 internal_key 循环
                                  else:
                                       # 可选键 obv_ma 缺少参数，不标记失败
                                       print(f"DEBUG: 回退查找: 可选键 '{internal_key}' 缺少格式化模式所需参数，跳过此键的查找。")


                    # 如果为此后缀找到所有必需列（不包括可选的 obv_ma）
                    # 修正检查逻辑，只检查 required_score_keys 中除了特殊处理的键
                    required_keys_base = [k for k in required_score_keys if k not in ['obv_ma', 'pivot_levels']] # 排除可选的 obv_ma 和特殊处理的 pivot_levels
                    all_base_required_found = all(k in temp_cols_found for k in required_keys_base)

                    # 检查 pivot_levels 是否已找到（如果需要）
                    pivot_levels_found_ok = True
                    if 'pivot_levels' in required_score_keys:
                         pivot_levels_found_ok = isinstance(temp_cols_found.get('pivot_levels'), dict) and bool(temp_cols_found.get('pivot_levels'))

                    # obv_ma 是可选的，即使未找到也不会影响 all_required_found_for_suffix

                    if all_base_required_found and pivot_levels_found_ok:
                         # 如果所有必需的键和必需的特殊结构都通过此后缀找到了数据源
                         indicator_cols_for_score = temp_cols_found
                         found = True
                         print(f"DEBUG: 通过回退成功找到指标 '{indicator_key}' 在时间框架 {tf_score_str} 的所有必需列和结构，后缀 '{tf_suffix}'。")
                         break # 为此 tf_score 使用此后缀找到了匹配项，跳出后缀循环
                    else:
                         print(f"DEBUG: 回退查找，后缀 '{tf_suffix}' 未找到所有必需列或必需特殊结构。")


            # --- 尝试所有查找方法后，如果仍然未找到必需的数据列 ---
            if not found:
                # 记录警告，未能找到必要的数据列进行评分
                logger.warning(f"未能为指标 '{indicator_key}' 在时间框架 {tf_score} 找到所有必要的数据列进行评分。")
                # 记录尝试查找所需的内部键列表
                logger.info(f"尝试查找所需的内部键: {required_score_keys}.")
                # 记录当前时间框架尝试的实际后缀列表
                logger.info(f"尝试的后缀列表: {current_tf_possible_suffixes}.")

                # 尝试列出 DataFrame 中与此指标/时间框架相关的列以进行调试
                relevant_cols_for_tf = []
                all_prefixes = []

                # 从 indicator_scoring_info 的 key_patterns 中提取所有 pattern 的基础部分作为前缀
                if isinstance(key_patterns_info, dict):
                     for kp_info in key_patterns_info.values():
                          if isinstance(kp_info, dict) and 'pattern' in kp_info:
                               pattern = kp_info['pattern']
                               # 对于没有参数的模式，整个模式可能就是前缀，例如 'ADL'
                               pattern_base = pattern.split('_')[0] if '_' in pattern else pattern
                               if pattern_base and pattern_base not in all_prefixes:
                                    all_prefixes.append(pattern_base)

                # 从 scoring_info 的 prefixes 列表中添加前缀
                score_info_prefixes = info.get('prefixes', [])
                for p in score_info_prefixes:
                     if p and p not in all_prefixes: all_prefixes.append(p)

                # 如果需要 'close' 列，添加 'close' 的前缀 (查找 OHLCV naming)
                if 'close' in required_score_keys:
                     ohlcv_output_cols_conf = ohlcv_naming_conv.get('output_columns', [])
                     close_pattern_prefix = 'close' # 默认前缀
                     if isinstance(ohlcv_output_cols_conf, list):
                          for col_conf in ohlcv_output_cols_conf:
                               if isinstance(col_conf, dict) and col_conf.get('internal_key') == 'close':
                                    close_pattern_prefix = col_conf.get('name_pattern', 'close').split('_')[0]
                                    break
                     if close_pattern_prefix and close_pattern_prefix not in all_prefixes:
                          all_prefixes.append(close_pattern_prefix)

                # 如果是 'pivot' 指标，添加 Pivot level 的基础名称作为前缀
                if indicator_key == 'pivot':
                     pivot_naming_convention = indicator_naming_conv.get('PIVOT', {})
                     pivot_cols_base = pivot_naming_convention.get('levels', [])
                     for p_base in pivot_cols_base:
                         if p_base and p_base not in all_prefixes: all_prefixes.append(p_base)


                # 如果尚未包含，将 indicator_key 本身（大写）作为潜在前缀添加
                if indicator_key.upper() not in [p.upper() for p in all_prefixes]:
                    all_prefixes.append(indicator_key.upper())
                # 过滤掉 None 或空字符串的前缀
                all_prefixes = [p for p in all_prefixes if p]

                print(f"DEBUG: 未找到列，尝试列出相关列。使用的前缀: {all_prefixes}, 可能后缀: {current_tf_possible_suffixes}")

                # 根据前缀和可能的后缀过滤 DataFrame 列
                cols_to_check = []
                for col in data.columns:
                     if any(col.startswith(prefix) for prefix in all_prefixes) and any(col.endswith(f"_{s}") for s in current_tf_possible_suffixes):
                          cols_to_check.append(col)

                relevant_cols_for_tf = [c for c in cols_to_check if c in data.columns]

                logger.info(f"DataFrame 中与时间框架 {tf_score} 匹配的 '{indicator_key}' 相关列列表: {sorted(relevant_cols_for_tf)}.")
                print(f"DEBUG: DataFrame 中与时间框架 {tf_score} 匹配的 '{indicator_key}' 相关列列表: {sorted(relevant_cols_for_tf)}.")

                # 如果未找到必需列，为该指标/时间框架的评分列填充默认中性分 50.0
                score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                scoring_results[score_col_name] = 50.0
                print(f"DEBUG: 未找到必需列，列 '{score_col_name}' 填充默认评分 50.0。")

            # --- 调用评分函数并存储结果 ---
            if found:
                try:
                    # 准备评分函数的参数
                    positional_series_args: List[pd.Series] = []
                    keyword_score_func_args: Dict[str, Any] = {}

                    print(f"DEBUG: 准备调用评分函数，所需键: {required_score_keys}")
                    for internal_key in required_score_keys:
                         actual_data_source = indicator_cols_for_score.get(internal_key)
                         # 检查数据源是否已找到，如果未找到，记录错误并跳过此时间框架的评分计算
                         if actual_data_source is None:
                              # 这不应该发生如果 'found' 是 True，但作为双重检查
                              if not (indicator_key == 'obv' and internal_key == 'obv_ma'): # obv_ma 是可选的，可以为 None
                                   logger.error(f"内部错误: 指标 '{indicator_key}' 在时间框架 {tf_score} 必需的内部键 '{internal_key}' 没有找到对应的数据源 (在 found=True 的情况下)。")
                                   print(f"DEBUG: 内部错误: 指标 '{indicator_key}' 在时间框架 {tf_score} 必需的内部键 '{internal_key}' 没有找到对应的数据源 (在 found=True 的情况下)。")
                                   # 由于是内部错误，中断当前时间框架的评分计算，并填充默认值
                                   raise ValueError(f"Internal error: Required internal key '{internal_key}' data source not found for {indicator_key}@{tf_score_str} despite found=True")
                                   # break # Break from internal_key loop if a required key is missing

                         # 特殊处理作为关键字参数传递的数据结构 (例如 pivot_levels)
                         if indicator_key == 'pivot' and internal_key == 'pivot_levels':
                              # 确保 actual_data_source 是字典且非空
                              if isinstance(actual_data_source, dict) and bool(actual_data_source):
                                  # 确保字典中的所有值都是 Series
                                  if all(isinstance(s, pd.Series) for s in actual_data_source.values()):
                                       keyword_score_func_args['pivot_levels'] = actual_data_source
                                  else:
                                       logger.error(f"内部错误: 指标 'pivot' 在时间框架 {tf_score}: 'pivot_levels' 字典中包含非 Series 值 (内部 key: '{internal_key}')。")
                                       print(f"DEBUG: 内部错误: 指标 'pivot' 在时间框架 {tf_score}: 'pivot_levels' 字典中包含非 Series 值 (内部 key: '{internal_key}')。")
                                       raise TypeError(f"Pivot 'pivot_levels' dict must contain only Series for internal key '{internal_key}'")
                              elif 'pivot_levels' in required_score_keys: # 如果 pivot_levels 是必需的但未找到有效数据，则出错
                                   logger.error(f"内部错误: 指标 'pivot' 在时间框架 {tf_score}: 必需的 pivot_levels 数据未找到或格式无效。")
                                   print(f"DEBUG: 内部错误: 指标 'pivot' 在时间框架 {tf_score}: 必需的 pivot_levels 数据未找到或格式无效。")
                                   raise ValueError(f"Required pivot_levels data not found or invalid for {indicator_key}@{tf_score_str}")


                         # 处理其他标准列，作为位置参数或关键字参数传递
                         elif isinstance(actual_data_source, str) and actual_data_source in data.columns:
                             if param_passing_style == 'none':
                                  positional_series_args.append(data[actual_data_source])
                                  print(f"DEBUG: 将列 '{actual_data_source}' 作为位置参数添加到评分函数。")
                             elif param_passing_style == 'dict':
                                  keyword_score_func_args[internal_key] = data[actual_data_source]
                                  print(f"DEBUG: 将列 '{actual_data_source}' (内部键 '{internal_key}') 作为关键字参数添加到评分函数字典。")
                             elif param_passing_style == 'individual':
                                  keyword_score_func_args[internal_key] = data[actual_data_source]
                                  print(f"DEBUG: 将列 '{actual_data_source}' (内部键 '{internal_key}') 作为独立关键字参数添加到评分函数。")
                             else:
                                  logger.error(f"指标 '{indicator_key}' 配置了未知的参数传递风格: '{param_passing_style}'.")
                                  print(f"DEBUG: 指标 '{indicator_key}' 配置了未知的参数传递风格: '{param_passing_style}'.")
                                  raise ValueError(f"Unknown param_passing_style: {param_passing_style} for indicator {indicator_key}")
                         # 处理可选的 obv_ma，如果它被找到了
                         elif indicator_key == 'obv' and internal_key == 'obv_ma':
                              if actual_data_source and actual_data_source in data.columns:
                                   # obv_ma 作为可选参数传递，如果找到则添加到关键字参数中
                                   keyword_score_func_args['obv_ma'] = data[actual_data_source] # 注意这里的参数名 'obv_ma' 需要与评分函数接收的参数名一致
                                   print(f"DEBUG: 找到并添加可选列 '{actual_data_source}' (内部键 '{internal_key}') 作为关键字参数。")
                              else:
                                   # obv_ma 是可选的，未找到是允许的，不添加到参数列表
                                   print(f"DEBUG: 可选键 '{internal_key}' 未找到对应的数据源，跳过添加。")
                         else:
                              # 必需的 internal_key 未找到对应的数据源 (不应该发生，因为 found 标记已经检查过)
                              # 对于非可选、非特殊处理但又没有找到数据源的键，这里应该已经通过上面的 None 检查抛出错误
                              pass # Should not reach here for required keys if check above is correct


                    # 准备评分函数的参数 (从 bs_params 中获取评分函数需要的参数值)
                    score_func_params: Dict[str, Any] = {}
                    for bs_key, func_arg_name in bs_param_key_to_score_func_arg.items():
                         param_value = bs_params.get(bs_key, defaults.get(bs_key, None))
                         if param_value is not None:
                              score_func_params[func_arg_name] = param_value

                    print(f"DEBUG: 调用指标 '{indicator_key}' 在时间框架 {tf_score_str} 的评分函数 '{score_func.__name__}'...")
                    print(f"DEBUG: 位置参数数量: {len(positional_series_args)}, 关键字参数 (数据+参数): {keyword_score_func_args.keys()}, 仅参数: {score_func_params.keys()}")

                    final_keyword_args = keyword_score_func_args.copy()
                    if param_passing_style == 'dict':
                         final_keyword_args.update(score_func_params)
                         print(f"DEBUG: 合并后的关键字参数 (dict 风格): {final_keyword_args.keys()}")
                         score = score_func(**final_keyword_args)
                    elif param_passing_style == 'individual':
                         final_keyword_args.update(score_func_params)
                         print(f"DEBUG: 合并后的关键字参数 (individual 风格): {final_keyword_args.keys()}")
                         score = score_func(**final_keyword_args)
                    elif param_passing_style == 'none':
                         if score_func_params:
                              logger.warning(f"指标 '{indicator_key}' 配置为 'none' 参数传递风格，但 bs_param_key_to_score_func_arg 或 defaults 不为空。这些参数可能被忽略或导致错误。")
                              print(f"DEBUG: 指标 '{indicator_key}' 配置为 'none' 风格，但有额外参数，请检查。")
                         score = score_func(*positional_series_args)
                    else:
                         raise ValueError(f"Unhandled param_passing_style: {param_passing_style} for indicator {indicator_key}")

                    # 确保评分结果是 Series
                    if not isinstance(score, pd.Series):
                        logger.error(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的评分函数未返回 pandas Series.")
                        print(f"DEBUG: 指标 '{indicator_key}' 在时间框架 {tf_score} 的评分函数未返回 Series.")
                        # 如果评分函数返回非 Series，标记失败并使用默认分
                        score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                        scoring_results[score_col_name] = 50.0
                        logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 评分结果无效，列 '{score_col_name}' 填充默认评分 50.0。")
                        print(f"DEBUG: 评分结果无效，列 '{score_col_name}' 填充默认评分 50.0。")
                        continue # 跳到下一个时间框架或指标

                    # 确保评分结果索引与输入数据索引一致
                    if not score.index.equals(data.index):
                         logger.error(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的评分结果索引与输入数据不一致.")
                         print(f"DEBUG: 指标 '{indicator_key}' 在时间框架 {tf_score} 的评分结果索引与输入数据不一致.")
                         # 如果索引不一致，标记失败并使用默认分
                         score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                         scoring_results[score_col_name] = 50.0
                         logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 评分结果索引不一致，列 '{score_col_name}' 填充默认评分 50.0。")
                         print(f"DEBUG: 评分结果索引不一致，列 '{score_col_name}' 填充默认评分 50.0。")
                         continue # 跳到下一个时间框架或指标

                    # 将评分结果添加到 results DataFrame 中
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                    scoring_results[score_col_name] = score
                    # 调试输出：评分计算成功
                    print(f"DEBUG: 成功计算指标 '{indicator_key}' 在时间框架 {tf_score_str} 的评分，列名为 '{score_col_name}'。")

                except Exception as e:
                    # 如果计算评分过程中发生错误
                    logger.error(f"计算指标 '{indicator_key}' 在时间框架 {tf_score} 的评分时发生错误: {e}", exc_info=True)
                    print(f"DEBUG: 计算指标 '{indicator_key}' 在时间框架 {tf_score} 的评分时发生错误: {e}")
                    # 如果评分计算失败，为该指标/时间框架的评分列填充默认中性分 50.0
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                    scoring_results[score_col_name] = 50.0
                    logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 评分计算失败，列 '{score_col_name}' 填充默认评分 50.0。")
                    print(f"DEBUG: 评分计算失败，列 '{score_col_name}' 填充默认评分 50.0。")

            # 如果未找到数据列 (已在上面处理并填充默认值)
            # else:
                # pass # do nothing, default score already assigned if not found

    # 返回包含所有指标评分的 DataFrame
    return scoring_results

# 添加了 pattern_params_map 字段，用于描述内部键对应的列名模式中的参数如何映射到 bs_params 中的键
# 这个映射将用于回退查找逻辑中构建期望列名。
indicator_scoring_info: Dict[str, Dict[str, Any]] = {
    'macd': {
        'func': calculate_macd_score,
        'param_passing_style': 'none',
        'bs_param_key_to_score_func_arg': {},
        'defaults': {},
        'required_keys': ['macd_series', 'macd_d', 'macd_h'],
        'prefixes': ['MACD_', 'MACDh_', 'MACDs_'],
        # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
        'key_patterns': {
            'macd_series': {
                'pattern': 'MACD_{period_fast}_{period_slow}_{signal_period}', # 添加模式
                'params_map': {'period_fast': 'macd_fast', 'period_slow': 'macd_slow', 'signal_period': 'macd_signal'}
            },
            'macd_d': {
                'pattern': 'MACDs_{period_fast}_{period_slow}_{signal_period}', # 添加模式
                'params_map': {'period_fast': 'macd_fast', 'period_slow': 'macd_slow', 'signal_period': 'macd_signal'}
            },
            'macd_h': {
                 'pattern': 'MACDh_{period_fast}_{period_slow}_{signal_period}', # 添加模式
                 'params_map': {'period_fast': 'macd_fast', 'period_slow': 'macd_slow', 'signal_period': 'macd_signal'}
            }
        }
    },
    'rsi': {
        'func': calculate_rsi_score,
        'param_passing_style': 'dict',
        'bs_param_key_to_score_func_arg': {'rsi_period': 'period', 'rsi_oversold': 'oversold', 'rsi_overbought': 'overbought', 'rsi_extreme_oversold': 'extreme_oversold', 'rsi_extreme_overbought': 'extreme_overbought'},
        'defaults': {'period': 14, 'oversold': 30, 'overbought': 70, 'extreme_oversold': 20, 'extreme_overbought': 80},
        'required_keys': ['rsi'],
        'prefixes': ['RSI_'],
        # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
        'key_patterns': {
            'rsi': {
                'pattern': 'RSI_{period}', # 添加模式
                'params_map': {'period': 'rsi_period'}
            },
        }
    },
    'kdj': {
        'func': calculate_kdj_score,
        'param_passing_style': 'dict',
        'bs_param_key_to_score_func_arg': {'kdj_period_k': 'period', 'kdj_period_d': 'signal_period', 'kdj_period_j': 'smooth_k_period', 'kdj_oversold': 'oversold', 'kdj_overbought': 'overbought', 'kdj_extreme_oversold': 'extreme_oversold', 'kdj_extreme_overbought': 'extreme_overbought'}, # 修正 bs_param_key_to_score_func_arg 映射，根据 pattern 参数名调整
        'defaults': {'k_period': 9, 'd_period': 3, 'smooth_k_period': 3, 'oversold': 20, 'overbought': 80, 'extreme_oversold': 10, 'extreme_overbought': 90}, # 修正 defaults 键名，与 pattern 参数名一致
        'required_keys': ['k', 'd', 'j'],
        'prefixes': ['K_', 'D_', 'J_'],
        # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
        'key_patterns': {
            'k': {
                'pattern': 'K_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', # 添加模式，使用与 pattern 参数名一致的占位符
                'params_map': {'k_period': 'kdj_period_k', 'd_period': 'kdj_period_d', 'smooth_k_period': 'kdj_period_j'}, # 修正 params_map，从 bs_params 中取值
            },
            'd': {
                'pattern': 'D_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', # 添加模式
                 'params_map': {'k_period': 'kdj_period_k', 'd_period': 'kdj_period_d', 'smooth_k_period': 'kdj_period_j'},
            },
            'j': {
                'pattern': 'J_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', # 添加模式
                 'params_map': {'k_period': 'kdj_period_k', 'd_period': 'kdj_period_d', 'smooth_k_period': 'kdj_period_j'},
            },
        }
    },
    'boll': {
       'func': calculate_boll_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['close', 'upper', 'mid', 'lower'],
       'prefixes': ['BBL_', 'BBM_', 'BBU_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'upper': {
                'pattern': 'BBU_{period}_{std_dev:.1f}', # 添加模式
                'params_map': {'period': 'boll_period', 'std_dev': 'boll_std_dev'},
            },
            'mid': {
                'pattern': 'BBM_{period}_{std_dev:.1f}', # 添加模式
                'params_map': {'period': 'boll_period', 'std_dev': 'boll_std_dev'},
            },
            'lower': {
                'pattern': 'BBL_{period}_{std_dev:.1f}', # 添加模式
                'params_map': {'period': 'boll_period', 'std_dev': 'boll_std_dev'},
            },
       }
    },
    'cci': {
       'func': calculate_cci_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'cci_period': 'period', 'cci_threshold': 'threshold', 'cci_extreme_threshold': 'extreme_threshold'},
       'defaults': {'period': 14, 'threshold': 100, 'extreme_threshold': 200},
       'required_keys': ['cci'],
       'prefixes': ['CCI_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'cci': {
                'pattern': 'CCI_{period}', # 添加模式
                'params_map': {'period': 'cci_period'},
            }
       }
    },
    'mfi': {
       'func': calculate_mfi_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'mfi_period': 'period', 'mfi_oversold': 'oversold', 'mfi_overbought': 'overbought', 'mfi_extreme_oversold': 'extreme_oversold', 'mfi_extreme_overbought': 'extreme_overbought'},
       'defaults': {'period': 14, 'oversold': 20, 'overbought': 80, 'extreme_oversold': 10, 'extreme_overbought': 90},
       'required_keys': ['mfi'],
       'prefixes': ['MFI_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'mfi': {
                'pattern': 'MFI_{period}', # 添加模式
                'params_map': {'period': 'mfi_period'},
            }
       }
    },
    'roc': {
       'func': calculate_roc_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['roc'],
       'prefixes': ['ROC_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'roc': {
                'pattern': 'ROC_{period}', # 添加模式
                'params_map': {'period': 'roc_period'},
            }
       }
    },
    'dmi': {
       'func': calculate_dmi_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'dmi_period': 'period', 'adx_threshold': 'adx_threshold', 'adx_strong_threshold': 'adx_strong_threshold'},
       'defaults': {'period': 14, 'adx_threshold': 25, 'adx_strong_threshold': 40},
       'required_keys': ['pdi', 'ndi', 'adx'],
       'prefixes': ['PDI_', 'NDI_', 'ADX_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'pdi': {
                 'pattern': 'PDI_{period}', # 添加模式
                 'params_map': {'period': 'dmi_period'},
            },
            'ndi': {
                 'pattern': 'NDI_{period}', # 添加模式
                 'params_map': {'period': 'dmi_period'},
            },
            'adx': {
                 'pattern': 'ADX_{period}', # 添加模式
                 'params_map': {'period': 'dmi_period'},
            },
       }
    },
    'sar': {
       'func': calculate_sar_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['close', 'sar'],
       'prefixes': ['SAR_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'sar': {
                'pattern': 'SAR_{af_step}_{max_af:.2f}', # 添加模式，使用 .2f 格式化以匹配日志中的 0.2
                'params_map': {'af_step': 'sar_step', 'max_af': 'sar_max'},
            },
       }
    },
    'stoch': {
       'func': calculate_stoch_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'stoch_k': 'k_period', 'stoch_d': 'd_period', 'stoch_smooth_k': 'smooth_k_period', 'stoch_oversold': 'stoch_oversold', 'stoch_overbought': 'stoch_overbought', 'stoch_extreme_oversold': 'stoch_extreme_oversold', 'stoch_extreme_overbought': 'stoch_extreme_overbought'}, # 修正 bs_param_key_to_score_func_arg 映射
       'defaults': {'k_period': 14, 'd_period': 3, 'smooth_k_period': 3, 'stoch_oversold': 20, 'stoch_overbought': 80, 'extreme_oversold': 10, 'extreme_overbought': 90}, # 修正 defaults 键名
       'required_keys': ['k', 'd'],
       'prefixes': ['STOCHk_', 'STOCHd_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'k': {
                'pattern': 'STOCHk_{k_period}_{d_period}_{smooth_k_period}', # 添加模式
                'params_map': {'k_period': 'stoch_k', 'd_period': 'stoch_d', 'smooth_k_period': 'stoch_smooth_k'}, # 修正 params_map，从 bs_params 中取值
            },
            'd': {
                'pattern': 'STOCHd_{k_period}_{d_period}_{smooth_k_period}', # 添加模式
                'params_map': {'k_period': 'stoch_k', 'd_period': 'stoch_d', 'smooth_k_period': 'stoch_smooth_k'},
            },
       }
    },
    'ema': {
       'func': calculate_ma_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'ema_period': 'period'},
       'defaults': {'period': 20},
       'required_keys': ['close', 'ma'], # MA评分函数通常只需要MA线本身和close
       'prefixes': ['EMA_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'ma': {
                'pattern': 'EMA_{period}', # 添加模式
                'params_map': {'period': 'ema_period'},
            },
       }
    },
    'sma': {
       'func': calculate_ma_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'sma_period': 'period'},
       'defaults': {'period': 20},
       'required_keys': ['close', 'ma'], # MA评分函数通常只需要MA线本身和close
       'prefixes': ['SMA_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'ma': {
                'pattern': 'SMA_{period}', # 添加模式
                'params_map': {'period': 'sma_period'},
            },
       }
    },
    'atr': {
       'func': calculate_atr_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['atr'],
       'prefixes': ['ATR_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'atr': {
                'pattern': 'ATR_{period}', # 添加模式
                'params_map': {'period': 'atr_period'},
            },
       }
    },
    'adl': {
       'func': calculate_adl_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['adl'],
       'prefixes': ['ADL_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'adl': {
                'pattern': 'ADL', # ADL 通常没有参数
                'params_map': {},
            },
       }
    },
    'vwap': {
       'func': calculate_vwap_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['close', 'vwap'],
       'prefixes': ['VWAP_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'vwap': {
                'pattern': 'VWAP', # VWAP 可能有 anchor 参数，但日志中的列名 VWAP_5 表明模式可能只是 VWAP + 后缀
                'params_map': {}, # 如果 VWAP_day 有参数，这里需要定义
            },
       }
    },
    'ichimoku': {
       'func': calculate_ichimoku_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['close', 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'],
       'prefixes': ['TENKAN_', 'KIJUN_', 'CHIKOU_', 'SENKOU_A_', 'SENKOU_B_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'tenkan': {
                'pattern': 'TENKAN_{period}', # 添加模式
                'params_map': {'period': 'ichimoku_tenkan_period'},
            },
            'kijun': {
                'pattern': 'KIJUN_{period}', # 添加模式
                'params_map': {'period': 'ichimoku_kijun_period'},
            },
            'chikou': {
                'pattern': 'CHIKOU_{period}', # 添加模式
                'params_map': {'period': 'ichimoku_chikou_period'},
            },
            'senkou_a': {
                'pattern': 'SENKOU_A_{tenkan_period}_{kijun_period}', # 添加模式
                'params_map': {'tenkan_period': 'ichimoku_tenkan_period', 'kijun_period': 'ichimoku_kijun_period'},
            },
            'senkou_b': {
                'pattern': 'SENKOU_B_{period}', # 添加模式
                'params_map': {'period': 'ichimoku_senkou_b_period'},
            },
       }
    },
    'mom': {
       'func': calculate_mom_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['mom'],
       'prefixes': ['MOM_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'mom': {
                'pattern': 'MOM_{period}', # 添加模式
                'params_map': {'period': 'mom_period'},
            },
       }
    },
    'willr': {
       'func': calculate_willr_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['willr'],
       'prefixes': ['WILLR_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'willr': {
                'pattern': 'WILLR_{period}', # 添加模式
                'params_map': {'period': 'willr_period'},
            },
       }
    },
    'cmf': {
       'func': calculate_cmf_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['cmf'],
       'prefixes': ['CMF_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'cmf': {
                'pattern': 'CMF_{period}', # 添加模式
                'params_map': {'period': 'cmf_period'},
            },
       }
    },
    'obv': {
       'func': calculate_obv_score,
       'param_passing_style': 'individual',
       'bs_param_key_to_score_func_arg': {'obv_ma_period': 'obv_ma_period'},
       'defaults': {'obv_ma_period': 10},
       'required_keys': ['obv', 'obv_ma'], # OBV 评分函数需要 OBV 线和其均线 (可选)
       'prefixes': ['OBV_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'obv': {
                'pattern': 'OBV', # OBV 本身没有参数
                'params_map': {},
            },
            'obv_ma': {
                'pattern': 'OBV_MA_{period}', # OBV_MA 的模式参数映射
                'params_map': {'period': 'obv_ma_period'},
            },
       }
    },
    'kc': {
       'func': calculate_kc_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['close', 'upper', 'mid', 'lower'],
       'prefixes': ['KCL_', 'KCM_', 'KCU_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            'upper': {
                 'pattern': 'KCU_{ema_period}_{atr_period}', # 添加模式
                 'params_map': {'ema_period': 'kc_ema_period', 'atr_period': 'kc_atr_period'},
            },
            'mid': {
                 'pattern': 'KCM_{ema_period}_{atr_period}', # 添加模式
                 'params_map': {'ema_period': 'kc_ema_period', 'atr_period': 'kc_atr_period'},
            },
            'lower': {
                 'pattern': 'KCL_{ema_period}_{atr_period}', # 添加模式
                 'params_map': {'ema_period': 'kc_ema_period', 'atr_period': 'kc_atr_period'},
            },
       }
    },
    'hv': {
       'func': calculate_hv_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['hv'],
       'prefixes': ['HV_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'hv': {
                'pattern': 'HV_{period}', # 添加模式
                'params_map': {'period': 'hv_period'},
            },
       }
    },
    'vroc': {
       'func': calculate_vroc_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['vroc'],
       'prefixes': ['VROC_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'vroc': {
                'pattern': 'VROC_{period}', # 添加模式
                'params_map': {'period': 'vroc_period'},
            },
       }
    },
    'aroc': {
       'func': calculate_aroc_score,
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['aroc'],
       'prefixes': ['AROC_'],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            'aroc': {
                'pattern': 'AROC_{period}', # 添加模式
                'params_map': {'period': 'aroc_period'},
            },
       }
    },
    'pivot': {
       'func': calculate_pivot_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {},
       'required_keys': ['close', 'pivot_levels'],
       'prefixes': [],
       # 修改: 将 pattern_params_map 改为 key_patterns，并加入 pattern
       'key_patterns': {
            # 'close' 是 OHLCV，在回退查找时需要特殊处理，不在这里定义 pattern
            # 'pivot_levels' 是特殊结构，在回退查找时需要特殊处理，不在这里定义 pattern
       }
    }
}



