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
from scipy.signal import find_peaks, argrelextrema # 需要 scipy
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
        check_divergence_pairs(price_trough_matches, is_peak=False, div_type='hidden_bullish') # Corrected typo 'hidden_hidden_bullish' to 'hidden_bullish'
    return result_df.fillna(0).astype(int)

def detect_divergence(data: pd.DataFrame, dd_params: Dict, naming_config: Dict, indicator_scoring_info: Dict) -> pd.DataFrame:
    """
    检测价格与多个指定指标之间的常规和隐藏背离。
    优化：
    - 改进指标配置查找逻辑，使其更健壮。
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
    indicator_naming_conv = naming_config.get('indicator_naming_conventions', {})
    ohlcv_naming_conv = naming_config.get('ohlcv_naming_convention', {})
    timeframe_naming_conv = naming_config.get('timeframe_naming_convention', {})
    if not isinstance(indicator_naming_conv, dict): indicator_naming_conv = {}
    if not isinstance(ohlcv_naming_conv, dict): ohlcv_naming_conv = {}
    if not isinstance(timeframe_naming_conv, dict): timeframe_naming_conv = {}
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
    for indicator_key, enabled in indicators_to_check.items():
        if not enabled:
            continue
        # --- 改进主指标配置的查找逻辑 ---
        # 确定用于在 indicator_naming_conv 中查找指标配置的主键。
        # 原始逻辑 (indicator_key.upper()) 对于 'macd_hist' 会查找 'MACD_HIST'，这通常不存在。
        # 正确的逻辑是为 'macd_hist' 查找 'MACD' 的配置块。
        main_config_lookup_key = indicator_key.upper() # 默认情况，例如 'rsi' -> 'RSI'
        key_lower = indicator_key.lower()
        if key_lower == 'macd_hist':
            main_config_lookup_key = 'MACD' # 'macd_hist' 属于 'MACD' 指标配置
        elif key_lower.startswith('stoch_'): # 例如 'stoch_k', 'stoch_d'
            main_config_lookup_key = 'STOCH' # 这些属于 'STOCH' 指标配置
        elif key_lower.startswith('kdj_'): # 例如 'kdj_j'
            main_config_lookup_key = 'KDJ' # 属于 'KDJ' 指标配置
        elif key_lower.startswith('dmi_'): # 例如 'dmi_adx', 'dmi_pdi', 'dmi_ndi'
            main_config_lookup_key = 'DMI' # 属于 'DMI' 指标配置
        # 对于其他直接对应的指标 (如 'rsi', 'mfi', 'obv'), indicator_key.upper() 通常是正确的顶级键。
        indi_naming_conf = indicator_naming_conv.get(main_config_lookup_key) # 使用修正后的键查找配置
        if not isinstance(indi_naming_conf, dict):
            logger.warning(f"指标 '{indicator_key}' (尝试使用主配置键 '{main_config_lookup_key}') 在命名规范中未找到或配置无效，跳过背离检测。")
            continue
        # 假设 dd_params 中的键是小写 internal_key，或需要进一步映射到实际用于匹配模式的内部名称
        internal_key_for_div = indicator_key.lower()
        # 特殊处理，将 dd_params 中的 indicator_key (如 'macd_hist') 映射到
        # 在 output_columns 中 name_pattern 对应的具体部分 (如 'macdh')
        if indicator_key.lower() == 'macd_hist': internal_key_for_div = 'macdh'
        elif indicator_key.lower() == 'stoch_k': internal_key_for_div = 'stochk'
        elif indicator_key.lower() == 'stoch_d': internal_key_for_div = 'stochd'
        elif indicator_key.lower() == 'kdj_j': internal_key_for_div = 'j'
        elif indicator_key.lower() == 'dmi_adx': internal_key_for_div = 'adx'
        elif indicator_key.lower() == 'dmi_pdi': internal_key_for_div = 'pdi'
        elif indicator_key.lower() == 'dmi_ndi': internal_key_for_div = 'ndi'
        indicator_pattern = None
        output_cols_patterns = indi_naming_conf.get('output_columns', [])
        if isinstance(output_cols_patterns, list):
            for col_conf in output_cols_patterns:
                if isinstance(col_conf, dict):
                    current_pattern_name = col_conf.get('name_pattern', '')
                    # 尝试通过匹配模式的基础部分 (如 'MACDh' 中的 'macdh') 来查找
                    pattern_base_lower = current_pattern_name.split('_')[0].lower()
                    if pattern_base_lower == internal_key_for_div:
                         indicator_pattern = current_pattern_name
                         break
            # 如果上述精确匹配未成功，尝试原始的 startswith 逻辑作为回退
            if not indicator_pattern:
                for col_conf in output_cols_patterns:
                    if isinstance(col_conf, dict) and col_conf.get('name_pattern', '').lower().startswith(internal_key_for_div):
                        indicator_pattern = col_conf.get('name_pattern')
                        break
        # 特殊处理 OBV，其模式通常是固定的，不含参数占位符
        if indicator_key.lower() == 'obv':
            # 在 indicator_naming_conventions.json 中, OBV 的模式是 "OBV"
            obv_pattern_found = False
            if isinstance(output_cols_patterns, list):
                for col_conf in output_cols_patterns:
                    if isinstance(col_conf, dict) and col_conf.get('name_pattern', '').upper() == 'OBV':
                        indicator_pattern = col_conf.get('name_pattern')
                        obv_pattern_found = True
                        break
            if not obv_pattern_found: # 如果规范中没有明确的 "OBV"，则硬编码（但不推荐）
                 indicator_pattern = 'OBV' # 应该依赖命名规范文件
        if not indicator_pattern:
            logger.warning(f"指标 '{indicator_key}' (内部键 '{internal_key_for_div}') 在命名规范 '{main_config_lookup_key}' 的 output_columns 中未找到主要输出列模式，跳过背离检测。")
            continue
        scoring_info = indicator_scoring_info.get(indicator_key.lower())
        indicator_params = {}
        if scoring_info and isinstance(scoring_info.get('defaults'), dict):
             indicator_params = scoring_info['defaults'].copy()
             indi_params_from_dd = dd_params.get(f'{indicator_key.lower()}_params', {})
             if isinstance(indi_params_from_dd, dict):
                  indicator_params.update(indi_params_from_dd)
        # 参数处理部分 (保持不变，根据 indicator_key.lower() 进行)
        if indicator_key.lower() in ['dmi', 'dmi_adx', 'dmi_pdi', 'dmi_ndi']:
             dmi_period = dd_params.get('dmi_period', indicator_params.get('period', 14))
             indicator_params['period'] = dmi_period
        if indicator_key.lower() in ['macd', 'macd_hist']:
             macd_params_dd = dd_params.get('macd_params', {})
             indicator_params['period_fast'] = macd_params_dd.get('period_fast', indicator_params.get('period_fast', 12))
             indicator_params['period_slow'] = macd_params_dd.get('period_slow', indicator_params.get('period_slow', 26))
             indicator_params['signal_period'] = macd_params_dd.get('signal_period', indicator_params.get('signal_period', 9))
        if indicator_key.lower() in ['kdj', 'kdj_j']:
             kdj_params_dd = dd_params.get('kdj_params', {})
             indicator_params['period'] = kdj_params_dd.get('period', indicator_params.get('period', 9))
             indicator_params['signal_period'] = kdj_params_dd.get('signal_period', indicator_params.get('signal_period', 3))
             indicator_params['smooth_k_period'] = kdj_params_dd.get('smooth_k_period', indicator_params.get('smooth_k_period', 3))
        if indicator_key.lower() in ['stoch', 'stoch_k', 'stoch_d']:
             stoch_params_dd = dd_params.get('stoch_params', {})
             indicator_params['k_period'] = stoch_params_dd.get('k_period', indicator_params.get('k_period', 14))
             indicator_params['d_period'] = stoch_params_dd.get('d_period', indicator_params.get('d_period', 3))
             indicator_params['smooth_k_period'] = stoch_params_dd.get('smooth_k_period', indicator_params.get('smooth_k_period', 3))
        if indicator_key.lower() == 'boll':
             boll_params_dd = dd_params.get('boll_params', {})
             indicator_params['period'] = boll_params_dd.get('period', indicator_params.get('period', 20))
             indicator_params['std_dev'] = boll_params_dd.get('std_dev', indicator_params.get('std_dev', 2.0))
        if indicator_key.lower() == 'sar':
             sar_params_dd = dd_params.get('sar_params', {})
             indicator_params['af_step'] = sar_params_dd.get('af_step', indicator_params.get('af_step', 0.02))
             indicator_params['max_af'] = sar_params_dd.get('max_af', indicator_params.get('max_af', 0.2))
        if indicator_key.lower() == 'vwap':
             vwap_params_dd = dd_params.get('vwap_params', {})
             indicator_params['anchor'] = vwap_params_dd.get('anchor', indicator_params.get('anchor', None))
        if indicator_key.lower() == 'ichimoku':
             ichimoku_params_dd = dd_params.get('ichimoku_params', {})
             indicator_params['tenkan_period'] = ichimoku_params_dd.get('tenkan_period', indicator_params.get('tenkan_period', 9))
             indicator_params['kijun_period'] = ichimoku_params_dd.get('kijun_period', indicator_params.get('kijun_period', 26))
             indicator_params['senkou_period'] = ichimoku_params_dd.get('senkou_period', indicator_params.get('senkou_period', 52))
        if indicator_key.lower() == 'kc':
             kc_params_dd = dd_params.get('kc_params', {})
             indicator_params['ema_period'] = kc_params_dd.get('ema_period', indicator_params.get('ema_period', 20))
             indicator_params['atr_period'] = kc_params_dd.get('atr_period', indicator_params.get('atr_period', 10))
        if indicator_key.lower() == 'obv_ma':
             obv_ma_params_dd = dd_params.get('obv_ma_params', {})
             indicator_params['period'] = obv_ma_params_dd.get('period', indicator_params.get('period', 10))
        for tf_check in timeframes_to_check:
            tf_check_str = str(tf_check)
            possible_tf_suffixes_raw = timeframe_naming_conv.get('patterns', {}).get(tf_check_str.lower(), [tf_check_str])
            if not isinstance(possible_tf_suffixes_raw, list): possible_tf_suffixes = [str(possible_tf_suffixes_raw)]
            else: possible_tf_suffixes = [str(s) for s in possible_tf_suffixes_raw]
            if tf_check_str in possible_tf_suffixes: possible_tf_suffixes.remove(tf_check_str); possible_tf_suffixes.insert(0, tf_check_str)
            elif tf_check_str.upper() in possible_tf_suffixes: possible_tf_suffixes.remove(tf_check_str.upper()); possible_tf_suffixes.insert(0, tf_check_str.upper())
            else: possible_tf_suffixes.insert(0, tf_check_str)
            seen = set(); possible_tf_suffixes_unique = [];
            for suffix in possible_tf_suffixes:
                if suffix not in seen: seen.add(suffix); possible_tf_suffixes_unique.append(suffix)
            possible_tf_suffixes = possible_tf_suffixes_unique
            price_col = None
            # --- 确保在循环内部找到 price_col ---
            # 原代码 price_col 定义在循环外部，可能导致在特定时间框架找不到时使用上一个时间框架的 price_col
            current_price_col = None # 为当前时间框架查找价格列
            for suffix in possible_tf_suffixes:
                 potential_price_col = f'{price_pattern}_{suffix}'
                 if potential_price_col in data.columns:
                      current_price_col = potential_price_col
                      break
            if current_price_col is None or data[current_price_col].isnull().all(): # 使用 current_price_col
                logger.warning(f"价格列 '{price_type}' 在 TF {tf_check} ({possible_tf_suffixes}) 未找到或全为 NaN。跳过指标 '{indicator_key}' 在此 TF 的检测。")
                continue
            price_series = data[current_price_col] # 使用 current_price_col
            indicator_col = None
            # 尝试根据模式和参数构建列名并查找
            try:
                 def format_param_for_div(p, key_lower_for_format): # 修改参数名以避免与外部作用域冲突
                     if isinstance(p, float):
                         if key_lower_for_format == 'boll': return f"{p:.1f}"
                         if key_lower_for_format == 'sar':
                              if 'af_step' in indicator_params and p == indicator_params['af_step']: return f"{p:.2f}"
                              if 'max_af' in indicator_params and p == indicator_params['max_af']: return f"{p:.1f}" # SAR max_af 通常一位小数
                              return f"{p:.2f}"
                         return str(p)
                     return str(p)
                 param_str_parts = []
                 param_key_lower = indicator_key.lower() # 使用局部变量
                 if param_key_lower in ['macd', 'macd_hist'] and 'period_fast' in indicator_params and 'period_slow' in indicator_params and 'signal_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period_fast'], param_key_lower), format_param_for_div(indicator_params['period_slow'], param_key_lower), format_param_for_div(indicator_params['signal_period'], param_key_lower)]
                 elif param_key_lower in ['rsi', 'cci', 'mfi', 'roc', 'atr', 'mom', 'willr', 'vroc', 'aroc'] and 'period' in indicator_params: # 已包含 mfi
                      param_str_parts = [format_param_for_div(indicator_params['period'], param_key_lower)]
                 elif param_key_lower in ['kdj', 'kdj_j'] and 'period' in indicator_params and 'signal_period' in indicator_params and 'smooth_k_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], param_key_lower), format_param_for_div(indicator_params['signal_period'], param_key_lower), format_param_for_div(indicator_params['smooth_k_period'], param_key_lower)]
                 elif param_key_lower in ['stoch', 'stoch_k', 'stoch_d'] and 'k_period' in indicator_params and 'd_period' in indicator_params and 'smooth_k_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['k_period'], param_key_lower), format_param_for_div(indicator_params['d_period'], param_key_lower), format_param_for_div(indicator_params['smooth_k_period'], param_key_lower)]
                 elif param_key_lower in ['dmi', 'dmi_adx', 'dmi_pdi', 'dmi_ndi'] and 'period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], param_key_lower)]
                 elif param_key_lower == 'boll' and 'period' in indicator_params and 'std_dev' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], param_key_lower), format_param_for_div(indicator_params['std_dev'], param_key_lower)]
                 elif param_key_lower == 'sar' and 'af_step' in indicator_params and 'max_af' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['af_step'], param_key_lower), format_param_for_div(indicator_params['max_af'], param_key_lower)]
                 elif param_key_lower == 'vwap' and 'anchor' in indicator_params and indicator_params['anchor'] is not None:
                      param_str_parts = [str(indicator_params['anchor'])]
                 elif param_key_lower == 'ichimoku': # Ichimoku 参数构建复杂，通常参数已在模式名中
                      pass
                 elif param_key_lower == 'kc' and 'ema_period' in indicator_params and 'atr_period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['ema_period'], param_key_lower), format_param_for_div(indicator_params['atr_period'], param_key_lower)]
                 elif param_key_lower == 'obv_ma' and 'period' in indicator_params:
                      param_str_parts = [format_param_for_div(indicator_params['period'], param_key_lower)]
                 # 对于 OBV, indicator_pattern 是 'OBV', param_str_parts 为空
                 param_part_for_pattern = '_'.join(param_str_parts)
                 # 构建期望的列名
                 # indicator_pattern 可能是 "MACDh_{period_fast}_{period_slow}_{signal_period}"
                 # 我们需要用实际参数值替换占位符
                 expected_col_name_intermediate = indicator_pattern
                 # 替换占位符，注意要与 indicator_naming_conventions.json 中的占位符一致
                 # 例如: {period}, {std_dev:.1f}, {af_step:.2f}
                 # 这是简化的替换，实际中可能需要更复杂的模板引擎或格式化
                 # 例如，"MACDh_{period_fast}_{period_slow}_{signal_period}"
                 # 参数是 period_fast, period_slow, signal_period
                 if param_key_lower in ['macd', 'macd_hist']:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{period_fast}', param_str_parts[0])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{period_slow}', param_str_parts[1])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{signal_period}', param_str_parts[2])
                 elif param_key_lower in ['rsi', 'cci', 'mfi', 'roc', 'atr', 'mom', 'willr', 'vroc', 'aroc'] and param_str_parts:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{period}', param_str_parts[0])
                 elif param_key_lower in ['kdj', 'kdj_j'] and len(param_str_parts) == 3:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{period}', param_str_parts[0])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{signal_period}', param_str_parts[1])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{smooth_k_period}', param_str_parts[2])
                 elif param_key_lower in ['stoch', 'stoch_k', 'stoch_d'] and len(param_str_parts) == 3:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{k_period}', param_str_parts[0])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{d_period}', param_str_parts[1])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{smooth_k_period}', param_str_parts[2])
                 elif param_key_lower in ['dmi', 'dmi_adx', 'dmi_pdi', 'dmi_ndi'] and param_str_parts:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{period}', param_str_parts[0])
                 elif param_key_lower == 'boll' and len(param_str_parts) == 2:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{period}', param_str_parts[0])
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{std_dev:.1f}', param_str_parts[1]) # 注意格式匹配
                 elif param_key_lower == 'sar' and len(param_str_parts) == 2:
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{af_step:.2f}', param_str_parts[0]) # 注意格式匹配
                     expected_col_name_intermediate = expected_col_name_intermediate.replace('{max_af:.1f}', param_str_parts[1]) # 注意格式匹配
                 elif param_key_lower == 'vwap' and param_str_parts: # 带锚点的VWAP
                     if indicator_pattern == "VWAP_{anchor}": # 确保是带锚点的模式
                         expected_col_name_intermediate = expected_col_name_intermediate.replace('{anchor}', param_str_parts[0])
                 # Ichimoku 的模式参数替换类似处理，这里省略以保持简洁，因其模式较多
                 # OBV, ADL 等无参数指标, expected_col_name_intermediate 就是 indicator_pattern 本身
                 expected_col_name_final = f"{expected_col_name_intermediate}_{tf_check_str}"
                 expected_col_name_final = expected_col_name_final.replace('__', '_').strip('_')
                 if expected_col_name_final in data.columns:
                      indicator_col = expected_col_name_final
                 else: # Fallback if direct construction fails, try to find a column that starts with the pattern base and ends with params_tf
                    # This is a simpler match if the above formatting is too complex or slightly off
                    # Example: indicator_pattern "MACDh_{...}" -> pattern_base "MACDh"
                    # Look for "MACDh" + formatted_params + "_" + tf_suffix
                    pattern_base = indicator_pattern.split('_')[0] # e.g. MACDh, RSI
                    param_suffix_for_search = f"_{param_part_for_pattern}" if param_part_for_pattern else ""
                    potential_col_search = f"{pattern_base}{param_suffix_for_search}_{tf_check_str}".replace('__', '_').strip('_')
                    if potential_col_search in data.columns:
                        indicator_col = potential_col_search
                    # else:
                    #     logger.debug(f"Constructed name '{expected_col_name_final}' and fallback '{potential_col_search}' not found for {indicator_key} TF {tf_check_str}")
            except Exception as e:
                 logger.warning(f"尝试构建或查找指标 '{indicator_key}' (内部键: '{internal_key_for_div}') 在 TF {tf_check} 的列名时出错: {e}")
            # Fallback (原有的简单前缀后缀匹配，如果上面的构建逻辑都失败了)
            if indicator_col is None:
                 prefixes = indicator_scoring_info.get(indicator_key.lower(), {}).get('prefixes', [])
                 indi_naming_prefixes_conf = indi_naming_conf.get('prefixes', {})
                 if isinstance(indi_naming_prefixes_conf, dict):
                      prefix_from_config = indi_naming_prefixes_conf.get(internal_key_for_div)
                      if prefix_from_config and prefix_from_config not in prefixes:
                           prefixes.append(prefix_from_config)
                 elif isinstance(indi_naming_prefixes_conf, list):
                      for p_conf in indi_naming_prefixes_conf:
                           if p_conf not in prefixes: prefixes.append(p_conf)
                 
                 if prefixes: # 只有当有可用前缀时才进行此回退查找
                    potential_cols = [c for c in data.columns if any(c.startswith(p) for p in prefixes) and c.endswith(f"_{tf_check_str}")]
                    if potential_cols:
                        # 启发式选择：如果多个匹配，可能需要更复杂的逻辑
                        # 暂时选择第一个，或最长/最短的，或与 indicator_pattern 最相似的
                        # 为了简单，这里选择第一个找到的
                        indicator_col = potential_cols[0]
                        logger.debug(f"Fallback prefix-suffix lookup for '{indicator_key}' in TF {tf_check} found potential column: '{indicator_col}' using prefixes: {prefixes}")
            if indicator_col is None or data[indicator_col].isnull().all():
                logger.warning(f"指标 '{indicator_key}' (内部键: '{internal_key_for_div}') 在 TF {tf_check} 的列未找到或全为 NaN。尝试的列名构建可能为 '{expected_col_name_final if 'expected_col_name_final' in locals() else 'N/A'}'。跳过。")
                continue
            indicator_series = data[indicator_col]
            # 假设 get_find_peaks_params 和 find_divergence_for_indicator 函数已在外部定义
            # current_find_peaks_params = get_find_peaks_params(tf_check_str, safe_lookback)
            # current_find_peaks_params.update(base_find_peaks_params)
            # print(f"开始检测 TF {tf_check}: 价格 ('{current_price_col}') 与指标 ('{indicator_col}') 的背离 (lookback: {safe_lookback})...") # 使用 current_price_col
            # div_result = find_divergence_for_indicator(...)
            # --- 以下为模拟 div_result 以便代码能继续 ---
            _mock_index = price_series.index
            div_result = pd.DataFrame(index=_mock_index)
            if check_regular_bullish: div_result['regular_bullish'] = pd.Series(False, index=_mock_index)
            if check_regular_bearish: div_result['regular_bearish'] = pd.Series(False, index=_mock_index)
            if check_hidden_bullish: div_result['hidden_bullish'] = pd.Series(False, index=_mock_index)
            if check_hidden_bearish: div_result['hidden_bearish'] = pd.Series(False, index=_mock_index)
            # --- 模拟结束 ---
            for div_type_col_name in div_result.columns:
                parts = indicator_col.split('_')
                indi_name_part = parts[0]
                params_part = "_".join(parts[1:-1]) if len(parts) > 2 and parts[-1] == tf_check_str else "_".join(parts[1:]) # 尝试更灵活地提取参数部分
                if params_part.endswith(f"_{tf_check_str}"): # 如果参数部分错误地包含了时间框架后缀
                    params_part = params_part[:-len(f"_{tf_check_str}")]
                clean_div_type = "".join(word.capitalize() for word in div_type_col_name.split('_'))
                detailed_col_name = f'DIV_{indi_name_part}'
                if params_part:
                     detailed_col_name += f'_{params_part}'
                detailed_col_name += f'_{tf_check_str}_{clean_div_type}'
                all_divergence_signals[detailed_col_name.upper()] = div_result[div_type_col_name]
    bullish_cols = [col for col in all_divergence_signals.columns if 'BULLISH' in col.upper() and col not in ['HAS_BULLISH_DIVERGENCE', 'HAS_BEARISH_DIVERGENCE']]
    if bullish_cols:
        all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = (all_divergence_signals[bullish_cols].astype(float) > 0).any(axis=1) # 确保数值比较
    bearish_cols = [col for col in all_divergence_signals.columns if 'BEARISH' in col.upper() and col not in ['HAS_BULLISH_DIVERGENCE', 'HAS_BEARISH_DIVERGENCE']]
    if bearish_cols:
        all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = (all_divergence_signals[bearish_cols].astype(float) < 0).any(axis=1) # 确保数值比较
    for col in all_divergence_signals.columns:
        if all_divergence_signals[col].dtype == 'bool':
            all_divergence_signals[col] = all_divergence_signals[col].fillna(False)
        else: # 通常背离信号是数值 (-1, 1) 或 (0, 1, -1)
            all_divergence_signals[col] = all_divergence_signals[col].fillna(0)
    all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = all_divergence_signals['HAS_BULLISH_DIVERGENCE'].astype(bool)
    all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = all_divergence_signals['HAS_BEARISH_DIVERGENCE'].astype(bool)
    logger.info(f"背离检测完成。共生成 {len(all_divergence_signals.columns) - 2} 个详细信号列。")
    return all_divergence_signals

def detect_kline_patterns(df: pd.DataFrame, tf: str, naming_config: Dict) -> pd.DataFrame: # Added naming_config parameter
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
    :param naming_config: 包含列命名规范的字典。 #参数说明
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
    # Use naming_config to get pattern prefix if available, otherwise use default 'KAP_'
    kline_pattern_prefix = naming_config.get('kline_pattern_naming_convention', {}).get('prefix', 'KAP_') #：从命名规范获取K线形态前缀
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
                       is_green & (body > avg_body * 1.5) & (c > c4) & (o > l1) # 当日阳线开盘高于前一整理K线低点 # Corrected is_red1 condition
        patterns_calc.loc[rising_three, f'{kline_pattern_prefix}RISINGTHREEMETHODS_{tf}'] = 1 # 使用获取到的前缀
        # 下降三法 (Falling Three Methods) - 看跌持续
        # 条件: 前4日长阴, 中间三日小K线整理且在前4日长阴实体内, 当日长阴收盘低于前4日收盘。
        falling_three = is_red4 & (body4 > avg_body * 1.5) & \
                        (is_green3 | (body3 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h3 < h4) & (l3 > l4) & \
                        (is_green2 | (body2 < avg_body * consolidation_body_max_ratio_of_avg_body)) & (h2 < h4) & (l2 > l4) & \
                        is_green1 & (body1 < avg_body * consolidation_body_max_ratio_of_avg_body) & (h1 < h4) & (l1 > l4) & \
                        is_red & (body > avg_body * 1.5) & (c < c4) & (o < h1) # 当日阴线开盘低于前一整理K线高点 # Corrected is_green1 condition
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
def _safe_fillna_series(series_list: List[pd.Series], fill_values: Any) -> List[pd.Series]:
    """
    安全地填充 Series 中的 NaN 值，优先使用 ffill/bfill，然后使用指定值。
    确保所有 Series 具有相同的索引。
    """
    # 如果传入的是单个 Series，则转为列表
    if isinstance(series_list, pd.Series):
        # print("警告：传入的是单个 Series，已自动转为列表")
        series_list = [series_list]
    # 判断列表是否为空
    if len(series_list) == 0:
        return []
    # 如果 fill_values 不是列表或元组，则自动扩展为与 series_list 等长的列表
    if not isinstance(fill_values, (list, tuple)):
        fill_values = [fill_values] * len(series_list)
    # 确保所有 Series 索引一致，以第一个 Series 的索引为准
    base_index = series_list[0].index
    aligned_series = [s.reindex(base_index) for s in series_list]
    filled_series = []
    for i, s in enumerate(aligned_series):
        if s.isnull().all():
            fill_val = fill_values[i] if i < len(fill_values) else 50.0
            if fill_val is None:
                filled_series.append(s)
            elif isinstance(fill_val, (int, float)):
                filled_series.append(s.fillna(fill_val))
            elif hasattr(fill_val, '__call__'):
                try:
                    val = fill_val(s)
                    filled_series.append(s.fillna(val))
                except Exception:
                    filled_series.append(s.fillna(50.0))
                    logger.warning(f"调用填充函数失败，使用默认值 50.0 填充 Series {i}.")
            else:
                filled_series.append(s.fillna(fill_val))
        else:
            s_filled = s.ffill().bfill()
            fill_val = fill_values[i] if i < len(fill_values) else 50.0
            if fill_val is not None:
                if isinstance(fill_val, (int, float)):
                    s_filled = s_filled.fillna(fill_val)
                elif hasattr(fill_val, '__call__'):
                    try:
                        val = fill_val(s_filled)
                        s_filled = s_filled.fillna(val)
                    except Exception:
                        s_filled = s_filled.fillna(50.0)
                        logger.warning(f"调用填充函数失败，使用默认值 50.0 填充 Series {i}.")
                else:
                    s_filled = s_filled.fillna(fill_val)
            filled_series.append(s_filled)
    return filled_series

def calculate_macd_score(macd_series: pd.Series, macd_d: pd.Series, macd_h: pd.Series) -> pd.Series:
    """MACD 评分 (0-100)，深化规则，增加得分难度。"""
    # 确保索引一致并填充NaN
    macd_s, macd_d_s, macd_h_s = _safe_fillna_series(
        [macd_series, macd_d, macd_h],
        [0.0, 0.0, 0.0]
    )
    if macd_s.isnull().all() or macd_d_s.isnull().all() or macd_h_s.isnull().all():
        print("调试信息: 填充后仍存在全NaN序列，返回默认50分。")
        return pd.Series(50.0, index=macd_series.index).clip(0,100)
    # 定义MACD位置判断的阈值
    ZERO_THRESHOLD = 0.05 # 零轴附近范围调整得更小，判断更精确
    HIGH_MACD_ABS = 0.5   # MACD绝对值高位判断 (用于DEA和MACD线)
    LOW_MACD_ABS = -0.5  # MACD绝对值低位判断
    # 初始化分数
    score = pd.Series(50.0, index=macd_s.index)
    # 计算金叉和死叉
    # prev_macd_s = macd_s.shift(1) # 修改点: 提前计算shift，避免重复
    # prev_macd_d_s = macd_d_s.shift(1) # 修改点: 提前计算shift，避免重复
    # buy_cross = (prev_macd_s < prev_macd_d_s) & (macd_s >= macd_d_s)
    # sell_cross = (prev_macd_s > prev_macd_d_s) & (macd_s <= macd_d_s)
    # 修正金叉死叉的判断，确保shift(1)有有效值，否则可能导致首日误判
    # 为了更安全地处理边界情况，确保shift后的数据与原始数据对齐比较
    # 通常pandas的shift会自动处理索引对齐，但填充NaN后的首个有效数据点的shift(1)会是NaN
    # _safe_fillna_series 已经处理了NaN，所以这里的shift(1)在非首日应该是安全的
    # 但为了更严格，可以考虑在有足够数据点后才开始计算交叉
    # 简单处理：如果长度小于2，不计算交叉
    if len(macd_s) < 2:
        print("调试信息: 数据长度不足2，无法计算交叉，大部分评分将基于动量。")
        buy_cross = pd.Series(False, index=macd_s.index)
        sell_cross = pd.Series(False, index=macd_s.index)
    else:
        buy_cross = (macd_s.shift(1) < macd_d_s.shift(1)) & (macd_s >= macd_d_s)
        sell_cross = (macd_s.shift(1) > macd_d_s.shift(1)) & (macd_s <= macd_d_s)
        # 将交叉信号的第一个NaN（由于shift产生）填充为False
        buy_cross.iloc[0] = False
        sell_cross.iloc[0] = False

    # DEA线斜率 (趋势辅助)
    dea_slope = macd_d_s.diff().fillna(0) # DEA线的变化，正表示向上，负表示向下
    DEA_SLOPE_UP_THRESHOLD = 0.005 # DEA显著向上的阈值 (可调)
    DEA_SLOPE_DOWN_THRESHOLD = -0.005 # DEA显著向下的阈值 (可调)
    # MACDh 柱体确认 (交叉后的第一根柱子)
    # macdh_confirms_buy = buy_cross & (macd_h_s > 0) & (macd_h_s > macd_h_s.shift(1).fillna(0)) # 金叉后红柱放出且增长
    # macdh_confirms_sell = sell_cross & (macd_h_s < 0) & (macd_h_s < macd_h_s.shift(1).fillna(0)) # 死叉后绿柱放出且增长(绝对值)
    # 修正MACDh确认逻辑：金叉发生时，当前MACDh应为正；死叉发生时，当前MACDh应为负。
    # 确认强度可以看柱子是否比0大/小一个阈值，或者是否比前一期（交叉前）的柱子有明显改善
    macdh_strong_positive = macd_h_s > ZERO_THRESHOLD # MACDh为显著正值
    macdh_strong_negative = macd_h_s < -ZERO_THRESHOLD # MACDh为显著负值
    # --- 交叉点评分 ---
    # 条件列表 (优先级从高到低)
    conditions = []
    choices = []
    # 1. 极强信号: 低位金叉 + DEA向上 + MACDh确认 (红柱)
    cond_strongest_buy = buy_cross & (macd_s < LOW_MACD_ABS) & (macd_d_s < LOW_MACD_ABS) & \
                         (dea_slope > DEA_SLOPE_UP_THRESHOLD) & macdh_strong_positive
    conditions.append(cond_strongest_buy)
    choices.append(95.0)
    # 2. 极强信号: 高位死叉 + DEA向下 + MACDh确认 (绿柱)
    cond_strongest_sell = sell_cross & (macd_s > HIGH_MACD_ABS) & (macd_d_s > HIGH_MACD_ABS) & \
                          (dea_slope < DEA_SLOPE_DOWN_THRESHOLD) & macdh_strong_negative
    conditions.append(cond_strongest_sell)
    choices.append(5.0)
    # 3. 强信号: 零轴金叉 + DEA向上
    cond_strong_zero_buy = buy_cross & (macd_s.abs() < ZERO_THRESHOLD) & \
                           (dea_slope > DEA_SLOPE_UP_THRESHOLD) & macdh_strong_positive
    conditions.append(cond_strong_zero_buy)
    choices.append(88.0)
    # 4. 强信号: 零轴死叉 + DEA向下
    cond_strong_zero_sell = sell_cross & (macd_s.abs() < ZERO_THRESHOLD) & \
                            (dea_slope < DEA_SLOPE_DOWN_THRESHOLD) & macdh_strong_negative
    conditions.append(cond_strong_zero_sell)
    choices.append(12.0)
    # 5. 普通金叉 (细化)
    # 5.1 低位金叉 (无强DEA/MACDh确认，但仍是较好信号)
    cond_low_buy = buy_cross & (macd_s < LOW_MACD_ABS) & (macd_d_s < LOW_MACD_ABS) & ~cond_strongest_buy
    conditions.append(cond_low_buy)
    choices.append(80.0)
    # 5.2 零轴附近金叉 (无强DEA确认)
    cond_zero_area_buy = buy_cross & (macd_s.abs() < ZERO_THRESHOLD) & ~cond_strong_zero_buy & ~cond_strongest_buy
    conditions.append(cond_zero_area_buy)
    choices.append(75.0)
    # 5.3 DEA多头区金叉 (DEA > 0, MACD > 0)
    cond_dea_bull_buy = buy_cross & (macd_d_s > ZERO_THRESHOLD) & (macd_s > ZERO_THRESHOLD) & \
                        (dea_slope > 0) & ~cond_strongest_buy & ~cond_strong_zero_buy
    conditions.append(cond_dea_bull_buy)
    choices.append(70.0)
    # 5.4 高位金叉 (风险较高，得分保守)
    cond_high_buy = buy_cross & (macd_s > HIGH_MACD_ABS) & (macd_d_s > HIGH_MACD_ABS) & ~cond_strongest_buy
    conditions.append(cond_high_buy)
    choices.append(60.0) # 高位金叉，即使是金叉，也给相对低分以示风险
    # 6. 普通死叉 (细化)
    # 6.1 高位死叉 (无强DEA/MACDh确认)
    cond_high_sell = sell_cross & (macd_s > HIGH_MACD_ABS) & (macd_d_s > HIGH_MACD_ABS) & ~cond_strongest_sell
    conditions.append(cond_high_sell)
    choices.append(20.0)
    # 6.2 零轴附近死叉 (无强DEA确认)
    cond_zero_area_sell = sell_cross & (macd_s.abs() < ZERO_THRESHOLD) & ~cond_strong_zero_sell & ~cond_strongest_sell
    conditions.append(cond_zero_area_sell)
    choices.append(25.0)
    # 6.3 DEA空头区死叉 (DEA < 0, MACD < 0)
    cond_dea_bear_sell = sell_cross & (macd_d_s < -ZERO_THRESHOLD) & (macd_s < -ZERO_THRESHOLD) & \
                         (dea_slope < 0) & ~cond_strongest_sell & ~cond_strong_zero_sell
    conditions.append(cond_dea_bear_sell)
    choices.append(30.0)
    # 6.4 低位死叉 (有反弹可能，得分相对不那么低)
    cond_low_sell = sell_cross & (macd_s < LOW_MACD_ABS) & (macd_d_s < LOW_MACD_ABS) & ~cond_strongest_sell
    conditions.append(cond_low_sell)
    choices.append(40.0) # 低位死叉，即使是死叉，也给相对高分以示潜在反弹
    # 7. 其他一般性交叉 (未被上述更具体条件捕捉的)
    conditions.append(buy_cross)
    choices.append(65.0) # 一般金叉的基准分
    conditions.append(sell_cross)
    choices.append(35.0) # 一般死叉的基准分
    # 应用交叉点分数
    score = pd.Series(np.select(conditions, choices, default=score.values), index=score.index)
    # --- 非交叉点动量评分 (在交叉点分数基础上修正或对非交叉点赋值) ---
    not_cross_cond = ~buy_cross & ~sell_cross
    # MACDh 状态
    macd_h_positive = macd_h_s > 0
    macd_h_negative = macd_h_s < 0
    macd_h_growing = macd_h_s > macd_h_s.shift(1).fillna(0) # 红柱增长或绿柱缩短
    macd_h_shrinking = macd_h_s < macd_h_s.shift(1).fillna(0) # 红柱缩短或绿柱增长(绝对值)
    # 1. 强劲多头动量 (非交叉，MACD > DEA, DEA > 0且向上, MACDh红柱持续放大)
    strong_bullish_momentum = not_cross_cond & (macd_s > macd_d_s) & \
                              (macd_d_s > ZERO_THRESHOLD) & (dea_slope > DEA_SLOPE_UP_THRESHOLD) & \
                              macd_h_positive & macd_h_growing & (macd_h_s > macd_h_s.shift(1).fillna(0)) # 连续两期增长
    score.loc[strong_bullish_momentum] = np.maximum(score.loc[strong_bullish_momentum], 75.0)
    # 2. 强劲空头动量 (非交叉，MACD < DEA, DEA < 0且向下, MACDh绿柱持续放大)
    strong_bearish_momentum = not_cross_cond & (macd_s < macd_d_s) & \
                               (macd_d_s < -ZERO_THRESHOLD) & (dea_slope < DEA_SLOPE_DOWN_THRESHOLD) & \
                               macd_h_negative & macd_h_shrinking & (macd_h_s < macd_h_s.shift(1).fillna(0)) # 连续两期增长(绝对值)
    score.loc[strong_bearish_momentum] = np.minimum(score.loc[strong_bearish_momentum], 25.0)
    # 3. 一般多头动量 (MACDh红柱，且增长或在高位)
    bullish_momentum_cond = not_cross_cond & macd_h_positive & macd_h_growing
    score.loc[bullish_momentum_cond] = np.maximum(score.loc[bullish_momentum_cond], 60.0)
    # 4. 一般空头动量 (MACDh绿柱，且增长(绝对值)或在低位)
    bearish_momentum_cond = not_cross_cond & macd_h_negative & macd_h_shrinking
    score.loc[bearish_momentum_cond] = np.minimum(score.loc[bearish_momentum_cond], 40.0)
    # 5. 多头动能减弱 (MACDh红柱但缩短)
    bullish_weakening = not_cross_cond & macd_h_positive & macd_h_shrinking & (macd_s > macd_d_s) # 确保仍在金叉状态之上
    score.loc[bullish_weakening] = np.maximum(score.loc[bullish_weakening], 55.0) # 比中性略高，但低于动能增强
    score.loc[bullish_weakening & (score > 60.0)] = np.minimum(score.loc[bullish_weakening & (score > 60.0)], 60.0) # 如果之前分数很高，适当回调
    # 6. 空头动能减弱 (MACDh绿柱但缩短，即绝对值减小)
    bearish_weakening = not_cross_cond & macd_h_negative & macd_h_growing & (macd_s < macd_d_s) # 确保仍在死叉状态之下
    score.loc[bearish_weakening] = np.minimum(score.loc[bearish_weakening], 45.0) # 比中性略低，但高于动能增强
    score.loc[bearish_weakening & (score < 40.0)] = np.maximum(score.loc[bearish_weakening & (score < 40.0)], 40.0) # 如果之前分数很低，适当回调
    # 7. MACD/DEA 在零轴上方且 MACD > DEA，但动能不显著 (维持多头看法)
    above_zero_hold_bullish = not_cross_cond & (macd_s > macd_d_s) & (macd_d_s > -ZERO_THRESHOLD) & \
                                ~(bullish_momentum_cond | strong_bullish_momentum | bullish_weakening)
    score.loc[above_zero_hold_bullish] = np.maximum(score.loc[above_zero_hold_bullish], 58.0)
    # 8. MACD/DEA 在零轴下方且 MACD < DEA，但动能不显著 (维持空头看法)
    below_zero_hold_bearish = not_cross_cond & (macd_s < macd_d_s) & (macd_d_s < ZERO_THRESHOLD) & \
                                ~(bearish_momentum_cond | strong_bearish_momentum | bearish_weakening)
    score.loc[below_zero_hold_bearish] = np.minimum(score.loc[below_zero_hold_bearish], 42.0)
    return score.clip(0, 100)

def calculate_rsi_score(rsi: pd.Series, params: Dict) -> pd.Series:
    """
    RSI 评分 (0-100)。
    深化RSI指标的评分，考虑RSI的移动平均线金叉/死叉以及不同阈值。
    """
    # print("--- 开始计算 RSI 评分 ---") # 调试信息
    # print(f"原始 RSI 数据（前5行）:\n{rsi.head()}") # 调试信息
    # print(f"传入参数: {params}") # 调试信息
    # 使用 _safe_fillna_series 填充，RSI 中性50
    rsi_s, = _safe_fillna_series([rsi], [50.0]) # 修改点1: 确保 _safe_fillna_series 的调用和返回值处理正确
    # print(f"填充后的 RSI 数据（前5行）:\n{rsi_s.head()}") # 调试信息
    # 如果填充后仍然全部为NaN（理论上_safe_fillna_series会处理，但作为安全检查）
    if rsi_s.isnull().all():
        print("警告: RSI 数据填充后仍全部为 NaN，返回中性评分。") # 调试信息
        return pd.Series(50.0, index=rsi.index).clip(0, 100)
    score = pd.Series(50.0, index=rsi_s.index) # 初始化所有分数为中性50
    # print(f"初始评分（前5行）:\n{score.head()}") # 调试信息
    # 修改点2: 从 params 字典获取参数，使用 'rsi_' 前缀，并更新默认值以匹配用户提供
    os = params.get('rsi_oversold', 30)
    ob = params.get('rsi_overbought', 70)
    ext_os = params.get('rsi_extreme_oversold', 25)
    ext_ob = params.get('rsi_extreme_overbought', 75)
    ma_period = params.get('rsi_ma_period', 9)
    gc_low_threshold = params.get('rsi_gc_low_threshold', 40)
    gc_mid_upper_threshold = params.get('rsi_gc_mid_upper_threshold', 60)
    dc_high_threshold = params.get('rsi_dc_high_threshold', 60)
    dc_mid_lower_threshold = params.get('rsi_dc_mid_lower_threshold', 40)
    # print(f"解析参数: OS={os}, OB={ob}, ExtOS={ext_os}, ExtOB={ext_ob}, MA_Period={ma_period}") # 调试信息
    # print(f"GC_Low={gc_low_threshold}, GC_MidUpper={gc_mid_upper_threshold}, DC_High={dc_high_threshold}, DC_MidLower={dc_mid_lower_threshold}") # 调试信息
    # 计算 RSI 的移动平均线
    # 修改点3: 计算 RSI 的移动平均线，使用 min_periods=1 确保在数据开始时也能计算MA
    rsi_ma = rsi_s.rolling(window=ma_period, min_periods=1).mean()
    # print(f"RSI MA（前5行）:\n{rsi_ma.head()}") # 调试信息
    # 1. 极端超买/超卖区域评分
    # 修改点4: 极端区域评分，直接赋值，因为这是最强的信号
    score.loc[rsi_s < ext_os] = 95.0 # 极端超卖，强烈看涨
    score.loc[rsi_s > ext_ob] = 5.0  # 极端超买，强烈看跌
    # print(f"极端区域评分后（前5行）:\n{score.head()}") # 调试信息
    # 2. 超买/超卖区域评分 (非极端)
    # 修改点5: 超买/超卖区域评分，使用 np.maximum/minimum 确保不覆盖更强的极端信号
    score.loc[(rsi_s >= ext_os) & (rsi_s < os)] = np.maximum(score.loc[(rsi_s >= ext_os) & (rsi_s < os)], 85.0) # 超卖但非极端，看涨
    score.loc[(rsi_s <= ext_ob) & (rsi_s > ob)] = np.minimum(score.loc[(rsi_s <= ext_ob) & (rsi_s > ob)], 15.0) # 超买但非极端，看跌
    # print(f"超买/超卖区域评分后（前5行）:\n{score.head()}") # 调试信息
    # 3. RSI 穿越超买/超卖线信号
    # 修改点6: RSI 穿越信号，使用 np.maximum/minimum 确保不覆盖更强的信号
    buy_signal = (rsi_s.shift(1) < os) & (rsi_s >= os) # 从超卖区向上穿越超卖线
    sell_signal = (rsi_s.shift(1) > ob) & (rsi_s <= ob) # 从超买区向下穿越超买线
    score.loc[buy_signal] = np.maximum(score.loc[buy_signal], 75.0) # 穿越买入信号，看涨
    score.loc[sell_signal] = np.minimum(score.loc[sell_signal], 25.0) # 穿越卖出信号，看跌
    # print(f"RSI 穿越信号评分后（前5行）:\n{score.head()}") # 调试信息
    # 4. RSI 与其移动平均线的金叉/死叉信号
    # 修改点7: 金叉/死叉信号，根据强度赋予不同分数，并使用 np.maximum/minimum
    if not rsi_ma.isnull().all(): # 确保 RSI MA 有足够的数据点进行计算
        gc_signal = (rsi_s.shift(1) < rsi_ma.shift(1)) & (rsi_s >= rsi_ma) # RSI 上穿其MA
        dc_signal = (rsi_s.shift(1) > rsi_ma.shift(1)) & (rsi_s <= rsi_ma) # RSI 下穿其MA
        # 强金叉 (RSI 上穿MA 且 RSI 处于特定看涨区间)
        strong_gc = gc_signal & (rsi_s >= gc_low_threshold) & (rsi_s < gc_mid_upper_threshold)
        score.loc[strong_gc] = np.maximum(score.loc[strong_gc], 90.0) # 强金叉，非常看涨
        # print(f"强金叉信号评分后（前5行）:\n{score.head()}") # 调试信息
        # 强死叉 (RSI 下穿MA 且 RSI 处于特定看跌区间)
        strong_dc = dc_signal & (rsi_s <= dc_high_threshold) & (rsi_s > dc_mid_lower_threshold)
        score.loc[strong_dc] = np.minimum(score.loc[strong_dc], 10.0) # 强死叉，非常看跌
        # print(f"强死叉信号评分后（前5行）:\n{score.head()}") # 调试信息
        # 普通金叉/死叉 (RSI 上穿/下穿MA，但不在强信号区间)
        normal_gc = gc_signal & ~strong_gc # 排除强金叉的点
        normal_dc = dc_signal & ~strong_dc # 排除强死叉的点
        score.loc[normal_gc] = np.maximum(score.loc[normal_gc], 80.0) # 普通金叉，看涨
        score.loc[normal_dc] = np.minimum(score.loc[normal_dc], 20.0) # 普通死叉，看跌
        # print(f"普通金叉/死叉信号评分后（前5行）:\n{score.head()}") # 调试信息
    else:
        print("RSI MA 数据不足，跳过金叉/死叉计算。") # 调试信息
    # 5. 中性区域内的趋势判断 (在没有明确买卖信号时)
    # 修改点8: 中性区域趋势判断，确保不覆盖更强的信号
    # 排除已经被其他强信号（如穿越、金叉/死叉）覆盖的点
    not_strong_signal_cond = ~buy_signal & ~sell_signal
    if not rsi_ma.isnull().all():
        not_strong_signal_cond = not_strong_signal_cond & ~gc_signal & ~dc_signal
    neutral_zone_cond = (rsi_s >= os) & (rsi_s <= ob) & not_strong_signal_cond
    bullish_trend = neutral_zone_cond & (rsi_s > rsi_s.shift(1)) # RSI 在中性区上升
    bearish_trend = neutral_zone_cond & (rsi_s < rsi_s.shift(1)) # RSI 在中性区下降
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # 中性区看涨
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # 中性区看跌
    # print(f"中性区域趋势评分后（前5行）:\n{score.head()}") # 调试信息
    # 6. 最终分数裁剪到 0-100 范围
    final_score = score.clip(0, 100)
    # print(f"最终评分（前5行）:\n{final_score.head()}") # 调试信息
    # print("--- RSI 评分计算完成 ---") # 调试信息
    return final_score

def calculate_kdj_score(k: pd.Series, d: pd.Series, j: pd.Series, params: Dict) -> pd.Series:
    """
    KDJ 评分 (0-100)。
    根据 KDJ 指标的 K, D, J 值及其交叉、趋势、钝化等情况，计算一个综合评分。
    评分规则已进一步深化，增加了钝化后交叉、J值反转确认、三线趋势等条件，并调整了部分原有信号的评分以增加得分难度。
    参数:
        k (pd.Series): KDJ 指标的 K 值序列。
        d (pd.Series): KDJ 指标的 D 值序列。
        j (pd.Series): KDJ 指标的 J 值序列。
        params (Dict): 包含 KDJ 评分所需参数的字典。
                       预期参数包括: 'oversold', 'overbought', 'extreme_oversold', 'extreme_overbought',
                                   'kdj_passivation_period', 'kdj_reversal_confirmation_period'。
    返回:
        pd.Series: 对应每个时间点的 KDJ 综合评分序列。
    """
    # print("--- 开始计算 KDJ 评分 (深化版) ---") # 调试信息
    k_s, d_s, j_s = _safe_fillna_series([k, d, j], [50.0, 50.0, 50.0])
    if k_s.isnull().all():
        # print("KDJ Series 全为 NaN，返回默认评分 50。")
        return pd.Series(50.0, index=k.index).clip(0,100)
    score = pd.Series(50.0, index=k_s.index)
    os_val = params.get('oversold', 25)
    ob_val = params.get('overbought', 75)
    ext_os_val = params.get('extreme_oversold', 10)
    ext_ob_val = params.get('extreme_overbought', 90)
    passivation_period_orig = params.get('kdj_passivation_period', 3)
    passivation_period = max(1, passivation_period_orig) # 确保 passivation_period >= 1
    # if passivation_period_orig < 1:
        # print(f"警告: kdj_passivation_period ({passivation_period_orig}) 无效，已重置为 {passivation_period}。")
    reversal_confirmation_period_orig = params.get('kdj_reversal_confirmation_period', 2)
    reversal_confirmation_period = max(1, reversal_confirmation_period_orig) # J值反转确认周期至少为1，通常为2或更高
    # if reversal_confirmation_period_orig < 1 : # reversal_confirmation_period > 0 的判断在后面rolling里有，这里确保至少为1
        #  print(f"警告: kdj_reversal_confirmation_period ({reversal_confirmation_period_orig}) 无效，已重置为 {reversal_confirmation_period}。")
    # print(f"KDJ 参数: os={os_val}, ob={ob_val}, ext_os={ext_os_val}, ext_ob={ext_ob_val}, passivation_period={passivation_period}, reversal_confirmation_period={reversal_confirmation_period}")
    # --- 核心判断逻辑 ---
    # 0. 基础信号计算
    k_prev = k_s.shift(1)
    d_prev = d_s.shift(1)
    j_prev = j_s.shift(1)
    # 确保所有布尔条件序列在逻辑运算前都是纯 bool 类型
    buy_cross_intermediate = (k_prev < d_prev) & (k_s >= d_s)
    buy_cross = buy_cross_intermediate.fillna(False).astype(bool)
    sell_cross_intermediate = (k_prev > d_prev) & (k_s <= d_s)
    sell_cross = sell_cross_intermediate.fillna(False).astype(bool)
    not_cross_cond = ~buy_cross & ~sell_cross
    # 1. J 值极度超买超卖区域的评分
    score.loc[j_s < ext_os_val] = 95.0
    score.loc[j_s > ext_ob_val] = 5.0
    # print(f"步骤1 - J值极度超买超卖后 score (前5): {score.head().to_dict()}")
    # 2. KDJ 钝化判断及钝化后交叉
    # 确保 rolling().apply() 结果和 shift().fillna() 结果为纯 bool 类型
    j_low_passivation_intermediate = j_s.rolling(window=passivation_period).apply(lambda x: (x < os_val).all(), raw=True)
    j_low_passivation = j_low_passivation_intermediate.fillna(False).astype(bool)
    j_high_passivation_intermediate = j_s.rolling(window=passivation_period).apply(lambda x: (x > ob_val).all(), raw=True)
    j_high_passivation = j_high_passivation_intermediate.fillna(False).astype(bool)
    shifted_j_low_passivation = j_low_passivation.shift(1).fillna(False).astype(bool)
    low_passivation_then_buy_cross = shifted_j_low_passivation & buy_cross
    score.loc[low_passivation_then_buy_cross] = np.maximum(score.loc[low_passivation_then_buy_cross], 98.0)
    shifted_j_high_passivation = j_high_passivation.shift(1).fillna(False).astype(bool)
    high_passivation_then_sell_cross = shifted_j_high_passivation & sell_cross
    score.loc[high_passivation_then_sell_cross] = np.minimum(score.loc[high_passivation_then_sell_cross], 2.0)
    # print(f"步骤2 - 钝化后交叉后 score (前5): {score.head().to_dict()}")
    # 3. 常规金叉和死叉信号的评分
    buy_cross_os = buy_cross & (j_s < os_val) & (~low_passivation_then_buy_cross)
    buy_cross_ob = buy_cross & (j_s > ob_val)
    neutral_buy_cross = buy_cross & (j_s >= os_val) & (j_s <= ob_val)
    sell_cross_os = sell_cross & (j_s < os_val)
    sell_cross_ob = sell_cross & (j_s > ob_val) & (~high_passivation_then_sell_cross)
    neutral_sell_cross = sell_cross & (j_s >= os_val) & (j_s <= ob_val)
    score.loc[buy_cross_os] = np.maximum(score.loc[buy_cross_os], 88.0)
    score.loc[neutral_buy_cross] = np.maximum(score.loc[neutral_buy_cross], 72.0)
    score.loc[buy_cross_ob] = np.maximum(score.loc[buy_cross_ob], 58.0)
    score.loc[sell_cross_ob] = np.minimum(score.loc[sell_cross_ob], 22.0)
    score.loc[neutral_sell_cross] = np.minimum(score.loc[neutral_sell_cross], 28.0)
    score.loc[sell_cross_os] = np.minimum(score.loc[sell_cross_os], 42.0)
    # print(f"步骤3 - 常规交叉后 score (前5): {score.head().to_dict()}")
    # 4. K 或 D 在超买超卖区域的评分
    k_or_d_in_os_not_extreme_intermediate = (((k_s >= ext_os_val) & (k_s < os_val)) | ((d_s >= ext_os_val) & (d_s < os_val))) & not_cross_cond
    k_or_d_in_os_not_extreme = k_or_d_in_os_not_extreme_intermediate.fillna(False).astype(bool) # 修改行
    score.loc[k_or_d_in_os_not_extreme] = np.maximum(score.loc[k_or_d_in_os_not_extreme], 85.0)
    k_or_d_in_ob_not_extreme_intermediate = (((k_s <= ext_ob_val) & (k_s > ob_val)) | ((d_s <= ext_ob_val) & (d_s > ob_val))) & not_cross_cond
    k_or_d_in_ob_not_extreme = k_or_d_in_ob_not_extreme_intermediate.fillna(False).astype(bool) # 修改行
    score.loc[k_or_d_in_ob_not_extreme] = np.minimum(score.loc[k_or_d_in_ob_not_extreme], 15.0)
    # print(f"步骤4 - K/D在超买超卖区后 score (前5): {score.head().to_dict()}")
    # 5. J 值反转确认
    j_diff = j_s.diff() # 首行为 NaN
    # 确保 j_diff 比较结果和 rolling().apply() 结果为纯 bool 类型
    j_diff_gt_zero = (j_diff > 0).fillna(False).astype(bool)
    j_diff_lt_zero = (j_diff < 0).fillna(False).astype(bool)
    j_turning_up_confirmed_intermediate = j_diff_gt_zero.rolling(window=reversal_confirmation_period).apply(lambda x: x.all(), raw=True) if reversal_confirmation_period > 0 else pd.Series(False, index=j_s.index)
    j_turning_up_confirmed = j_turning_up_confirmed_intermediate.fillna(False).astype(bool)
    j_turning_down_confirmed_intermediate = j_diff_lt_zero.rolling(window=reversal_confirmation_period).apply(lambda x: x.all(), raw=True) if reversal_confirmation_period > 0 else pd.Series(False, index=j_s.index)
    j_turning_down_confirmed = j_turning_down_confirmed_intermediate.fillna(False).astype(bool)
    start_of_reversal_j = j_s.shift(reversal_confirmation_period - 1).fillna(method='bfill') if reversal_confirmation_period > 0 else j_s
    cond_start_j_lt_os = (start_of_reversal_j < os_val).fillna(False).astype(bool) # 修改行
    bullish_j_reversal_from_os_confirmed = j_turning_up_confirmed & cond_start_j_lt_os & not_cross_cond
    score.loc[bullish_j_reversal_from_os_confirmed] = np.maximum(score.loc[bullish_j_reversal_from_os_confirmed], 80.0)
    cond_start_j_gt_ob = (start_of_reversal_j > ob_val).fillna(False).astype(bool) # 修改行
    bearish_j_reversal_from_ob_confirmed = j_turning_down_confirmed & cond_start_j_gt_ob & not_cross_cond
    score.loc[bearish_j_reversal_from_ob_confirmed] = np.minimum(score.loc[bearish_j_reversal_from_ob_confirmed], 20.0)
    # print(f"步骤5 - J值反转确认后 score (前5): {score.head().to_dict()}")
    # 6. K, D, J 三线趋势共振
    # 确保趋势判断结果为纯 bool 类型
    k_trend_up = (k_s > k_prev).fillna(False).astype(bool)
    d_trend_up = (d_s > d_prev).fillna(False).astype(bool)
    j_trend_up = (j_s > j_prev).fillna(False).astype(bool)
    all_lines_bullish_momentum = k_trend_up & d_trend_up & j_trend_up & not_cross_cond & (~bullish_j_reversal_from_os_confirmed)
    all_lines_bearish_momentum = (~k_trend_up) & (~d_trend_up) & (~j_trend_up) & not_cross_cond & (~bearish_j_reversal_from_ob_confirmed)
    strong_bullish_kdj_config_neutral_intermediate = (j_s >= os_val) & (j_s <= ob_val) & (k_s > d_s) & (j_s > k_s) & all_lines_bullish_momentum
    strong_bullish_kdj_config_neutral = strong_bullish_kdj_config_neutral_intermediate.fillna(False).astype(bool) # 修改行
    score.loc[strong_bullish_kdj_config_neutral] = np.maximum(score.loc[strong_bullish_kdj_config_neutral], 65.0)
    strong_bearish_kdj_config_neutral_intermediate = (j_s >= os_val) & (j_s <= ob_val) & (k_s < d_s) & (j_s < k_s) & all_lines_bearish_momentum
    strong_bearish_kdj_config_neutral = strong_bearish_kdj_config_neutral_intermediate.fillna(False).astype(bool) # 修改行
    score.loc[strong_bearish_kdj_config_neutral] = np.minimum(score.loc[strong_bearish_kdj_config_neutral], 35.0)
    # print(f"步骤6 - 三线趋势共振后 score (前5): {score.head().to_dict()}")
    # 7. J 值普通趋势评分
    bullish_j_in_os_intermediate = (j_s < os_val) & j_trend_up & not_cross_cond & (~bullish_j_reversal_from_os_confirmed)
    bullish_j_in_os = bullish_j_in_os_intermediate.fillna(False).astype(bool) # 修改行
    score.loc[bullish_j_in_os] = np.maximum(score.loc[bullish_j_in_os], 68.0)
    bearish_j_in_ob_intermediate = (j_s > ob_val) & (~j_trend_up) & not_cross_cond & (~bearish_j_reversal_from_ob_confirmed)
    bearish_j_in_ob = bearish_j_in_ob_intermediate.fillna(False).astype(bool) # 修改行
    score.loc[bearish_j_in_ob] = np.minimum(score.loc[bearish_j_in_ob], 32.0)
    neutral_j_zone_cond_intermediate = (j_s >= os_val) & (j_s <= ob_val) & not_cross_cond & \
                                (~bullish_j_reversal_from_os_confirmed) & (~bearish_j_reversal_from_ob_confirmed) & \
                                (~strong_bullish_kdj_config_neutral) & (~strong_bearish_kdj_config_neutral)
    neutral_j_zone_cond = neutral_j_zone_cond_intermediate.fillna(False).astype(bool) # 修改行
    bullish_j_trend_neutral = neutral_j_zone_cond & j_trend_up
    score.loc[bullish_j_trend_neutral] = np.maximum(score.loc[bullish_j_trend_neutral], 53.0)
    bearish_j_trend_neutral = neutral_j_zone_cond & (~j_trend_up)
    score.loc[bearish_j_trend_neutral] = np.minimum(score.loc[bearish_j_trend_neutral], 47.0)
    # print(f"步骤7 - J值普通趋势后 score (前5): {score.head().to_dict()}")
    final_score = score.clip(0, 100)
    # print(f"最终评分 Series (前5): {final_score.head().to_dict()}")
    # print("--- KDJ 评分计算结束 (深化版) ---")
    return final_score

# 内部参考周期，用于滚动计算等，不作为函数参数
INTERNAL_PERIOD_REF = 20

def calculate_boll_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """
    BOLL 评分 (0-100)，规则深化版，增加得分难度。
    分数越高，代表买入信号越强（价格越倾向于被低估或处于潜在反弹点）。
    分数越低，代表卖出信号越强（价格越倾向于被高估或处于潜在回调点）。
    50分为中性。
    """
    # 确保输入序列的索引一致性检查
    if not (close.index.equals(upper.index) and close.index.equals(mid.index) and close.index.equals(lower.index)):
        logger.warning("BOLL 评分：输入 Series 索引不一致。可能导致错误。返回中性评分。")
        return pd.Series(50.0, index=close.index).clip(0,100)
    # 使用 _safe_fillna_series 填充，并提供后备填充逻辑
    def create_fallback_lambda(stat_func, default_val_on_nan_stat):
        def fallback(s_orig):
            if s_orig is None or s_orig.isnull().all():
                return default_val_on_nan_stat
            val = stat_func(s_orig.dropna())
            if pd.isna(val) or np.isinf(val):
                return default_val_on_nan_stat
            return val
        return fallback
    close_s, upper_s, mid_s, lower_s = _safe_fillna_series(
        [close, upper, mid, lower],
        [
            None, # close 优先 ffill/bfill，若全NaN则由_safe_fillna_series内部逻辑处理
            create_fallback_lambda(lambda s: s.mean() + 2 * s.std() if not s.empty and s.std() > 0 else (s.mean() * 1.01 if not s.empty else 50.0), 50.0),
            create_fallback_lambda(lambda s: s.mean() if not s.empty else 50.0, 50.0),
            create_fallback_lambda(lambda s: s.mean() - 2 * s.std() if not s.empty and s.std() > 0 else (s.mean() * 0.99 if not s.empty else 50.0), 50.0)
        ]
    )
    # 再次检查关键序列在填充后是否仍全为NaN
    if close_s.isnull().all() or upper_s.isnull().all() or mid_s.isnull().all() or lower_s.isnull().all():
         logger.warning("BOLL 评分：一个或多个关键序列在填充后仍全为NaN。返回中性评分。")
         return pd.Series(50.0, index=close.index).clip(0,100)
    # 初始化评分为50 (中性)
    score = pd.Series(50.0, index=close_s.index)
    # --- 计算辅助指标 ---
    epsilon = 1e-9 # 一个极小值，防止除以0
    # 1. %B (Percent B)
    band_range = upper_s - lower_s
    percent_b = (close_s - lower_s) / (band_range + epsilon)
    condition_band_is_zero = band_range.abs() < epsilon
    values_for_zero_band_case_array = np.select(
        [
            close_s > mid_s,
            close_s < mid_s
        ],
        [
            1.5, # 视为极度超买
            -0.5 # 视为极度超卖
        ],
        default=0.5
    )
    values_for_zero_band_case_series = pd.Series(values_for_zero_band_case_array, index=close_s.index)
    percent_b.loc[condition_band_is_zero] = values_for_zero_band_case_series[condition_band_is_zero]
    percent_b = percent_b.fillna(0.5)
    # 2. 布林带宽度 (BBW)
    bbw = (band_range / (mid_s + epsilon)) * 100
    bbw = bbw.fillna(method='ffill').fillna(method='bfill').fillna(bbw.mean())
    # 3. BBW 变化率 (用于判断带子是扩张还是收缩)
    bbw_change = bbw.diff()
    bbw_std = bbw.rolling(INTERNAL_PERIOD_REF, min_periods=max(1, INTERNAL_PERIOD_REF // 2)).std().fillna(method='ffill').fillna(method='bfill').fillna(0)
    is_expanding_bands = bbw_change > bbw_std * 0.5
    is_contracting_bands = bbw_change < -bbw_std * 0.5 # is_contracting_bands 已定义
    # 4. 挤牌状态 (Squeeze)
    bbw_squeeze_threshold = bbw.rolling(window=INTERNAL_PERIOD_REF, min_periods=max(1, INTERNAL_PERIOD_REF // 2)).quantile(0.15)
    bbw_squeeze_threshold = bbw_squeeze_threshold.fillna(method='ffill').fillna(method='bfill')
    if bbw_squeeze_threshold.isnull().any():
        bbw_squeeze_threshold.fillna(bbw.mean() * 0.5 if not pd.isna(bbw.mean()) else 0.1, inplace=True)
    is_squeeze = bbw < bbw_squeeze_threshold
    # 5. 趋势判断 (基于中轨方向和价格相对中轨位置)
    mid_band_slope = mid_s.diff(periods=3).fillna(0)
    is_price_above_mid_sustained = (close_s > mid_s).rolling(window=3, min_periods=2).sum() >= 2
    is_price_below_mid_sustained = (close_s < mid_s).rolling(window=3, min_periods=2).sum() >= 2
    is_uptrend = (mid_band_slope > 0) & is_price_above_mid_sustained
    is_downtrend = (mid_band_slope < 0) & is_price_below_mid_sustained
    # --- 评分规则 (优先级从高到低，更具体的规则覆盖更一般的规则) ---
    # 使用 np.minimum 和 np.maximum 来叠加影响，确保更强的信号能覆盖弱信号
    # 1. 假突破/快速回归 (最高优先级，强反转信号，增加得分难度)
    false_breakout_up = (percent_b.shift(1) > 1.0) & (percent_b <= 0.5)
    score.loc[false_breakout_up] = np.minimum(score.loc[false_breakout_up], 3.0)
    false_breakout_down = (percent_b.shift(1) < 0.0) & (percent_b >= 0.5)
    score.loc[false_breakout_down] = np.maximum(score.loc[false_breakout_down], 97.0)
    # 2. 挤牌后的突破 (强趋势信号，增加得分难度)
    squeeze_breakout_up = is_squeeze.shift(1) & (percent_b > 1.0) & is_expanding_bands
    score.loc[squeeze_breakout_up] = np.minimum(score.loc[squeeze_breakout_up], 8.0)
    squeeze_breakout_down = is_squeeze.shift(1) & (percent_b < 0.0) & is_expanding_bands
    score.loc[squeeze_breakout_down] = np.maximum(score.loc[squeeze_breakout_down], 92.0)
    # 3. 极端超买/超卖 (价格远超布林带，增加得分难度)
    extreme_overbought = percent_b > 1.3
    score.loc[extreme_overbought] = np.minimum(score.loc[extreme_overbought], 5.0)
    extreme_oversold = percent_b < -0.3
    score.loc[extreme_oversold] = np.maximum(score.loc[extreme_oversold], 95.0)
    # 4. 触及布林带边界 (常规突破或触碰，增加得分难度)
    touch_upper = (percent_b > 1.0) & (percent_b <= 1.3)
    score.loc[touch_upper] = np.minimum(score.loc[touch_upper], 12.0)
    touch_lower = (percent_b < 0.0) & (percent_b >= -0.3)
    score.loc[touch_lower] = np.maximum(score.loc[touch_lower], 88.0)
    # 5. 从布林带外侧回归 (非假突破，而是回到带内，但未穿过中轨)
    revert_from_upper = (percent_b.shift(1) > 1.0) & (percent_b > 0.5) & (percent_b <= 1.0)
    score.loc[revert_from_upper] = np.maximum(score.loc[revert_from_upper], 45.0)
    revert_from_lower = (percent_b.shift(1) < 0.0) & (percent_b >= 0.0) & (percent_b < 0.5)
    score.loc[revert_from_lower] = np.minimum(score.loc[revert_from_lower], 55.0)
    # 6. 中轨穿越 (趋势确认/反转信号，增加得分难度)
    buy_mid_cross = (percent_b.shift(1) < 0.5) & (percent_b >= 0.5)
    score.loc[buy_mid_cross & is_uptrend] = np.maximum(score.loc[buy_mid_cross & is_uptrend], 75.0)
    score.loc[buy_mid_cross & ~is_uptrend] = np.maximum(score.loc[buy_mid_cross & ~is_uptrend], 60.0)
    sell_mid_cross = (percent_b.shift(1) > 0.5) & (percent_b <= 0.5)
    score.loc[sell_mid_cross & is_downtrend] = np.minimum(score.loc[sell_mid_cross & is_downtrend], 25.0)
    score.loc[sell_mid_cross & ~is_downtrend] = np.minimum(score.loc[sell_mid_cross & ~is_downtrend], 40.0)
    # 7. "行走布林带" (趋势持续性信号，增加得分难度)
    N_walk_days = 3
    min_walk_periods = N_walk_days
    # 7.1 强劲行走 (布林带扩张)
    walking_up_strong = ((percent_b > 0.95) & is_expanding_bands).rolling(window=N_walk_days, min_periods=min_walk_periods).sum() >= N_walk_days
    score.loc[walking_up_strong] = np.minimum(score.loc[walking_up_strong], 18.0)
    walking_down_strong = ((percent_b < 0.05) & is_expanding_bands).rolling(window=N_walk_days, min_periods=min_walk_periods).sum() >= N_walk_days
    score.loc[walking_down_strong] = np.maximum(score.loc[walking_down_strong], 82.0)
    # 7.2 收缩行走 (布林带收缩) - 新增逻辑
    # 价格接近上轨且布林带收缩，趋势可能减弱
    walking_up_contracting = ((percent_b > 0.85) & (percent_b <= 0.95) & is_contracting_bands).rolling(window=N_walk_days, min_periods=min_walk_periods).sum() >= N_walk_days # 新增条件
    score.loc[walking_up_contracting] = np.minimum(score.loc[walking_up_contracting], 35.0) # 分数更接近中性
    # 价格接近下轨且布林带收缩，趋势可能减弱或筑底
    walking_down_contracting = ((percent_b < 0.15) & (percent_b >= 0.05) & is_contracting_bands).rolling(window=N_walk_days, min_periods=min_walk_periods).sum() >= N_walk_days # 新增条件
    score.loc[walking_down_contracting] = np.maximum(score.loc[walking_down_contracting], 65.0) # 分数更接近中性
    # 7.3 稳定行走 (布林带既不扩张也不收缩) - 调整逻辑
    # 确保不与强劲行走和收缩行走重叠
    is_stable_bands = ~is_expanding_bands & ~is_contracting_bands # 定义稳定带子条件
    walking_up_stable = ((percent_b > 0.85) & (percent_b <= 0.95) & is_stable_bands).rolling(window=N_walk_days, min_periods=min_walk_periods).sum() >= N_walk_days # 使用 is_stable_bands
    score.loc[walking_up_stable] = np.minimum(score.loc[walking_up_stable], 30.0)
    walking_down_stable = ((percent_b < 0.15) & (percent_b >= 0.05) & is_stable_bands).rolling(window=N_walk_days, min_periods=min_walk_periods).sum() >= N_walk_days # 使用 is_stable_bands
    score.loc[walking_down_stable] = np.maximum(score.loc[walking_down_stable], 70.0)
    # 8. 在布林带内部的位置 (非极端、非穿越中轨时，更平滑的评分)
    neutral_indices = score[score == 50.0].index
    cond_above_mid_neutral = (percent_b > 0.5) & (percent_b <= 1.0) & percent_b.index.isin(neutral_indices)
    if cond_above_mid_neutral.any():
        ratio_upper = (percent_b.loc[cond_above_mid_neutral] - 0.5) / 0.5
        score_val_upper = 50.0 - (ratio_upper ** 1.5) * 35.0
        score.loc[cond_above_mid_neutral] = score_val_upper
    cond_below_mid_neutral = (percent_b < 0.5) & (percent_b >= 0.0) & percent_b.index.isin(neutral_indices)
    if cond_below_mid_neutral.any():
        ratio_lower = (0.5 - percent_b.loc[cond_below_mid_neutral]) / 0.5
        score_val_lower = 50.0 + (ratio_lower ** 1.5) * 35.0
        score.loc[cond_below_mid_neutral] = score_val_lower
    # 9. 价格非常接近中轨，且无明确穿越信号，视为最中性
    very_near_mid = (percent_b >= 0.49) & (percent_b <= 0.51)
    score.loc[very_near_mid] = 50.0
    # 最终确保分数在0-100之间
    return score.clip(0, 100).round(2)

def calculate_cci_score(cci: pd.Series, params: Dict) -> pd.Series:
    """
    CCI 评分 (0-100)。深化指标的评分规则，丰富计算规则，使评分更具洞察力。
    """
    # 使用 _safe_fillna_series 填充，确保CCI序列没有NaN值
    # 确保cci_s被正确填充，如果原始cci全为NaN，则返回中性分数
    cci_s, = _safe_fillna_series([cci], [0.0]) # CCI 中性0
    # 如果填充后的CCI序列仍然全部是NaN（这不应该发生，但作为安全检查），则返回中性分数
    if cci_s.isnull().all():
        # 返回一个全为50的Series，并确保索引正确
        print("调试信息: 填充后的CCI序列仍然全部是NaN，返回中性分数。")
        return pd.Series(50.0, index=cci.index).clip(0,100)
    # 初始化评分序列，所有值默认为50（中性）
    score = pd.Series(50.0, index=cci_s.index)
    # 从 params 字典获取参数
    threshold = params.get('threshold', 100)
    ext_threshold = params.get('extreme_threshold', 200)
    # --- 1. 极端超买/超卖区域评分 (0-5, 95-100) ---
    # 这些是最高优先级，直接设置分数，并带有轻微的梯度，使分数在极端情况下更接近0或100。
    # 增加极端区域的梯度评分逻辑
    cond_extreme_bull = cci_s < -ext_threshold
    cond_extreme_bear = cci_s > ext_threshold
    # CCI远低于-ext_threshold时，分数从95向100趋近
    score.loc[cond_extreme_bull] = 95.0 + ((-ext_threshold - cci_s.loc[cond_extreme_bull]) / ext_threshold).clip(0, 1) * 5.0
    # CCI远高于ext_threshold时，分数从5向0趋近
    score.loc[cond_extreme_bear] = 5.0 - ((cci_s.loc[cond_extreme_bear] - ext_threshold) / ext_threshold).clip(0, 1) * 5.0
    # --- 2. 强超买/超卖区域评分 (5-15, 85-95) ---
    # 在极端阈值和普通阈值之间，分数呈线性梯度变化，提供更精细的超买超卖程度。
    # 增加强超买/超卖区域的线性梯度评分逻辑
    cond_strong_bull = (cci_s >= -ext_threshold) & (cci_s < -threshold)
    cond_strong_bear = (cci_s > threshold) & (cci_s <= ext_threshold)
    # 从-ext_threshold到-threshold，分数从95线性减少到85
    score.loc[cond_strong_bull] = 95.0 - ((cci_s.loc[cond_strong_bull] + ext_threshold) / (ext_threshold - threshold)) * 10.0
    # 从threshold到ext_threshold，分数从15线性减少到5
    score.loc[cond_strong_bear] = 15.0 - ((cci_s.loc[cond_strong_bear] - threshold) / (ext_threshold - threshold)) * 10.0
    # --- 3. 交叉信号评分 (更强的买卖信号) ---
    # 这些信号表示CCI穿越关键阈值，具有较强的指示意义，使用np.maximum/np.minimum确保分数只朝有利方向调整。
    # 优先处理从极端区域穿越的信号，赋予更高权重。
    # 增加从极端区域穿越的强信号
    buy_signal_from_extreme = (cci_s.shift(1) < -ext_threshold) & (cci_s >= -threshold)
    sell_signal_from_extreme = (cci_s.shift(1) > ext_threshold) & (cci_s <= threshold)
    # 调整原始买卖信号的条件和分数
    buy_signal_from_oversold = (cci_s.shift(1) < -threshold) & (cci_s >= -threshold)
    sell_signal_from_overbought = (cci_s.shift(1) > threshold) & (cci_s <= threshold)
    score.loc[buy_signal_from_extreme] = np.maximum(score.loc[buy_signal_from_extreme], 90.0)
    score.loc[sell_signal_from_extreme] = np.minimum(score.loc[sell_signal_from_extreme], 10.0)
    score.loc[buy_signal_from_oversold] = np.maximum(score.loc[buy_signal_from_oversold], 75.0)
    score.loc[sell_signal_from_overbought] = np.minimum(score.loc[sell_signal_from_overbought], 25.0)
    # --- 4. 中性区域评分 (40-60) ---
    # 在中性区域内，根据CCI的位置和动量进行更精细的评分。
    neutral_zone_cond = (cci_s >= -threshold) & (cci_s <= threshold)
    cci_diff = cci_s.diff()
    # 排除已经由强交叉信号处理过的点，避免重复或冲突的评分。
    # 定义一个排除所有强信号的条件
    not_strong_signal = ~buy_signal_from_extreme & ~sell_signal_from_extreme & \
                        ~buy_signal_from_oversold & ~sell_signal_from_overbought
    # 4.1. 中性区域内靠近阈值的评分 (45-55)
    # 越靠近-threshold越看涨，越靠近threshold越看跌，提供中性区域内的倾向性。
    # 增加中性区域内靠近阈值的评分逻辑
    cond_neutral_bullish_proximity = neutral_zone_cond & (cci_s < 0) & not_strong_signal
    cond_neutral_bearish_proximity = neutral_zone_cond & (cci_s > 0) & not_strong_signal
    # 从0到-threshold，分数从50线性增加到55
    score.loc[cond_neutral_bullish_proximity] = np.maximum(score.loc[cond_neutral_bullish_proximity],
                                                           50.0 + (abs(cci_s.loc[cond_neutral_bullish_proximity]) / threshold) * 5.0)
    # 从0到threshold，分数从50线性减少到45
    score.loc[cond_neutral_bearish_proximity] = np.minimum(score.loc[cond_neutral_bearish_proximity],
                                                           50.0 - (cci_s.loc[cond_neutral_bearish_proximity] / threshold) * 5.0)
    # 4.2. 中性区域内的动量评分 (40-60)
    # CCI在中性区域内上涨或下跌的幅度越大，趋势越明显，分数调整幅度越大。
    # 增加中性区域内的动量评分逻辑
    bullish_momentum_neutral = neutral_zone_cond & (cci_diff > 0) & not_strong_signal
    bearish_momentum_neutral = neutral_zone_cond & (cci_diff < 0) & not_strong_signal
    # 根据CCI变化幅度，分数在50基础上增加，最高到60
    score.loc[bullish_momentum_neutral] = np.maximum(score.loc[bullish_momentum_neutral],
                                                     50.0 + (cci_diff.loc[bullish_momentum_neutral] / threshold).clip(0, 1) * 10.0)
    # 根据CCI变化幅度，分数在50基础上减少，最低到40
    score.loc[bearish_momentum_neutral] = np.minimum(score.loc[bearish_momentum_neutral],
                                                     50.0 - (abs(cci_diff.loc[bearish_momentum_neutral]) / threshold).clip(0, 1) * 10.0)
    # 确保最终分数被裁剪到0-100范围
    return score.clip(0, 100)

def calculate_mfi_score(mfi: pd.Series, params: Dict) -> pd.Series:
    """
    MFI 评分 (0-100)，规则深化版。
    评分越高，代表看涨信号越强或买入时机越好。
    评分越低，代表看跌信号越强或卖出时机越好。
    50 为中性。
    """
    # 使用 _safe_fillna_series 填充，MFI 中性值为 50
    # mfi_s, = _safe_fillna_series([mfi], [50.0]) # 假设已实现
    # 为确保代码可独立运行，此处进行简单填充
    mfi_s = mfi.fillna(50.0)

    if mfi_s.isnull().all():
        return pd.Series(50.0, index=mfi.index).clip(0, 100)
    # 初始化评分为中性值 50
    score = pd.Series(50.0, index=mfi_s.index)
    # 从 params 字典获取参数，提供默认值
    os_thresh = params.get('oversold', 20)  # 超卖阈值
    ob_thresh = params.get('overbought', 80)  # 超买阈值
    ext_os_thresh = params.get('extreme_oversold', 10)  # 极度超卖阈值
    ext_ob_thresh = params.get('extreme_overbought', 90)  # 极度超买阈值
    # 预计算 MFI 的前一期和前两期值，用于趋势和加速度判断
    mfi_s_shifted1 = mfi_s.shift(1)
    mfi_s_shifted2 = mfi_s.shift(2)
    # 1. 极度超卖区 (MFI < ext_os_thresh) - 看涨信号强烈
    # 增强极度超卖区评分逻辑
    ext_os_cond = mfi_s < ext_os_thresh
    score.loc[ext_os_cond] = np.maximum(score.loc[ext_os_cond], 90.0) # 基础分90
    # 深化：如果MFI在极度超卖区进一步下跌（更超卖），信号更强
    deepening_ext_os_cond = ext_os_cond & (mfi_s < mfi_s_shifted1)
    score.loc[deepening_ext_os_cond] = np.maximum(score.loc[deepening_ext_os_cond], 95.0) # 深化分95
    # 持续性：如果MFI连续两期处于极度超卖区，信号更强
    persistent_ext_os_cond = ext_os_cond & (mfi_s_shifted1 < ext_os_thresh)
    score.loc[persistent_ext_os_cond] = np.maximum(score.loc[persistent_ext_os_cond], 97.0) # 持续分97
    # 持续且深化：最强信号
    persistent_deepening_ext_os_cond = persistent_ext_os_cond & deepening_ext_os_cond
    score.loc[persistent_deepening_ext_os_cond] = np.maximum(score.loc[persistent_deepening_ext_os_cond], 99.0) # 持续且深化分99
    
    # 2. 极度超买区 (MFI > ext_ob_thresh) - 看跌信号强烈
    # 增强极度超买区评分逻辑
    ext_ob_cond = mfi_s > ext_ob_thresh
    score.loc[ext_ob_cond] = np.minimum(score.loc[ext_ob_cond], 10.0) # 基础分10
    # 深化：如果MFI在极度超买区进一步上涨（更超买），信号更强
    deepening_ext_ob_cond = ext_ob_cond & (mfi_s > mfi_s_shifted1)
    score.loc[deepening_ext_ob_cond] = np.minimum(score.loc[deepening_ext_ob_cond], 5.0) # 深化分5
    # 持续性：如果MFI连续两期处于极度超买区，信号更强
    persistent_ext_ob_cond = ext_ob_cond & (mfi_s_shifted1 > ext_ob_thresh)
    score.loc[persistent_ext_ob_cond] = np.minimum(score.loc[persistent_ext_ob_cond], 3.0) # 持续分3
    # 持续且深化：最强信号
    persistent_deepening_ext_ob_cond = persistent_ext_ob_cond & deepening_ext_ob_cond
    score.loc[persistent_deepening_ext_ob_cond] = np.minimum(score.loc[persistent_deepening_ext_ob_cond], 1.0) # 持续且深化分1
    
    # 3. 超卖区 (ext_os_thresh <= MFI < os_thresh) - 看涨信号较强
    # 细化超卖区评分，使用线性插值
    os_zone_cond = (mfi_s >= ext_os_thresh) & (mfi_s < os_thresh)
    if os_thresh > ext_os_thresh: # 避免除以零
        # MFI 越接近 ext_os_thresh，分数越高 (逼近90分)
        # MFI 越接近 os_thresh，分数越低 (逼近70分)
        # (os_thresh - mfi_s) / (os_thresh - ext_os_thresh) 是一个从0到1的比例，当mfi_s=ext_os_thresh时为1，mfi_s=os_thresh时为0
        score_in_os_zone = 70.0 + ((os_thresh - mfi_s) / (os_thresh - ext_os_thresh)) * (89.0 - 70.0) # 分数范围 70-89
        score.loc[os_zone_cond] = np.maximum(score.loc[os_zone_cond], score_in_os_zone.loc[os_zone_cond])
    else: # 如果阈值相同，则按原逻辑或给一个固定值
        score.loc[os_zone_cond] = np.maximum(score.loc[os_zone_cond], 85.0) # fallback to original-like logic
    
    # 4. 超买区 (ob_thresh < MFI <= ext_ob_thresh) - 看跌信号较强
    # 细化超买区评分，使用线性插值
    ob_zone_cond = (mfi_s > ob_thresh) & (mfi_s <= ext_ob_thresh)
    if ext_ob_thresh > ob_thresh: # 避免除以零
        # MFI 越接近 ext_ob_thresh，分数越低 (逼近11分)
        # MFI 越接近 ob_thresh，分数越高 (逼近30分)
        # (mfi_s - ob_thresh) / (ext_ob_thresh - ob_thresh) 是一个从0到1的比例，当mfi_s=ob_thresh时为0，mfi_s=ext_ob_thresh时为1
        score_in_ob_zone = 30.0 - ((mfi_s - ob_thresh) / (ext_ob_thresh - ob_thresh)) * (30.0 - 11.0) # 分数范围 11-30
        score.loc[ob_zone_cond] = np.minimum(score.loc[ob_zone_cond], score_in_ob_zone.loc[ob_zone_cond])
    else: # 如果阈值相同
        score.loc[ob_zone_cond] = np.minimum(score.loc[ob_zone_cond], 15.0) # fallback
    
    # 5. 买入信号 (MFI 从下方上穿超卖线 os_thresh)
    # 强化交叉信号
    buy_signal_cond = (mfi_s_shifted1 < os_thresh) & (mfi_s >= os_thresh)
    score.loc[buy_signal_cond] = np.maximum(score.loc[buy_signal_cond], 75.0) # 基础交叉分75
    # 如果从极度超卖区上穿，信号更强
    buy_signal_from_ext_os_cond = buy_signal_cond & (mfi_s_shifted1 < ext_os_thresh)
    score.loc[buy_signal_from_ext_os_cond] = np.maximum(score.loc[buy_signal_from_ext_os_cond], 85.0) # 从极度超卖区交叉分85
    
    # 6. 卖出信号 (MFI 从上方下穿超买线 ob_thresh)
    # 强化交叉信号
    sell_signal_cond = (mfi_s_shifted1 > ob_thresh) & (mfi_s <= ob_thresh)
    score.loc[sell_signal_cond] = np.minimum(score.loc[sell_signal_cond], 25.0) # 基础交叉分25
    # 如果从极度超买区下穿，信号更强
    sell_signal_from_ext_ob_cond = sell_signal_cond & (mfi_s_shifted1 > ext_ob_thresh)
    score.loc[sell_signal_from_ext_ob_cond] = np.minimum(score.loc[sell_signal_from_ext_ob_cond], 15.0) # 从极度超买区交叉分15
    
    # 7. 中性区域 (os_thresh <= MFI <= ob_thresh 且非上述买卖信号点)
    # 丰富中性区评分逻辑，考虑趋势和加速度
    not_signal_cond = ~buy_signal_cond & ~sell_signal_cond # 非信号日
    neutral_zone_cond = (mfi_s >= os_thresh) & (mfi_s <= ob_thresh) & not_signal_cond
    mfi_change = mfi_s - mfi_s_shifted1 # MFI变化量
    mfi_acceleration = mfi_change - (mfi_s_shifted1 - mfi_s_shifted2) # MFI变化加速度
    # 中性区看涨趋势
    neutral_bullish_cond = neutral_zone_cond & (mfi_change > 0)
    # 基础分略高于50，根据变化量调整，但幅度不大
    score_neutral_bullish = 50.0 + np.clip(mfi_change * 0.5, 1, 5) # 51-55
    score.loc[neutral_bullish_cond] = np.maximum(score.loc[neutral_bullish_cond], score_neutral_bullish.loc[neutral_bullish_cond])
    # 如果加速上涨，则进一步提高分数
    neutral_bullish_accel_cond = neutral_bullish_cond & (mfi_acceleration > 0)
    score_neutral_bullish_accel = score.loc[neutral_bullish_accel_cond] + np.clip(mfi_acceleration * 0.3, 1, 5) # 再加1-5分
    score.loc[neutral_bullish_accel_cond] = np.maximum(score.loc[neutral_bullish_accel_cond], score_neutral_bullish_accel.loc[neutral_bullish_accel_cond])
    # 中性区看跌趋势
    neutral_bearish_cond = neutral_zone_cond & (mfi_change < 0)
    # 基础分略低于50，根据变化量调整
    score_neutral_bearish = 50.0 + np.clip(mfi_change * 0.5, -5, -1) # 45-49 (mfi_change是负数)
    score.loc[neutral_bearish_cond] = np.minimum(score.loc[neutral_bearish_cond], score_neutral_bearish.loc[neutral_bearish_cond])
    # 如果加速下跌，则进一步降低分数
    neutral_bearish_accel_cond = neutral_bearish_cond & (mfi_acceleration < 0)
    score_neutral_bearish_accel = score.loc[neutral_bearish_accel_cond] + np.clip(mfi_acceleration * 0.3, -5, -1) # 再减1-5分
    score.loc[neutral_bearish_accel_cond] = np.minimum(score.loc[neutral_bearish_accel_cond], score_neutral_bearish_accel.loc[neutral_bearish_accel_cond])
    
    # 确保所有 MFI 值本身接近50（例如48-52）且变化不大的情况下，分数也接近50
    # 上述逻辑已经通过clip和较小的调整因子来控制中性区得分的摆动幅度
    # 对于MFI值在50附近且几乎无变化的情况，mfi_change 和 mfi_acceleration 会很小，得分会保持在50附近
    # 最后，将分数裁剪到 0-100 范围
    return score.clip(0, 100)

def calculate_roc_score(roc: pd.Series) -> pd.Series:
    """
    计算ROC指标的评分 (0-100)。
    深化评分规则，增加得分难度，使评分更具洞察力，并对代码做效率优化。
    """
    # 使用 _safe_fillna_series 填充ROC中的NaN值，中性值为0
    roc_s, = _safe_fillna_series([roc], [0.0])
    # 如果所有ROC值都为NaN（经过填充后可能不会出现，但作为安全检查保留）
    if roc_s.isnull().all():
        return pd.Series(50.0, index=roc.index).clip(0, 100)
    # 初始化所有分数为中性值50.0
    score = pd.Series(50.0, index=roc_s.index, dtype=float)
    # 优化：预计算前一天的ROC值，避免重复计算
    roc_s_prev = roc_s.shift(1)
    # 1. 交叉信号 (优先级最高，直接设定分数)
    # 买入交叉：ROC从负值变为非负值
    buy_cross = (roc_s_prev < 0) & (roc_s >= 0)
    # 卖出交叉：ROC从正值变为非正值
    sell_cross = (roc_s_prev > 0) & (roc_s <= 0)
    # 赋予交叉信号更高的权重和更明确的指示性分数
    score.loc[buy_cross] = 80.0
    score.loc[sell_cross] = 20.0
    # 优化：预计算非交叉条件，减少后续重复计算
    # 预计算非交叉条件
    not_cross_cond = ~buy_cross & ~sell_cross
    # 2. ROC值幅度贡献 (在非交叉区域，根据ROC的绝对值强度调整分数)
    # 定义ROC幅度区间和对应的分数调整阈值，使评分更具区分度
    # 强劲上涨区域：ROC大于5
    roc_positive_strong = (roc_s > 5)
    # 温和上涨区域：ROC在(1, 5]之间
    roc_positive_moderate = (roc_s > 1) & (roc_s <= 5)
    # 强劲下跌区域：ROC小于-5
    roc_negative_strong = (roc_s < -5)
    # 温和下跌区域：ROC在[-5, -1)之间
    roc_negative_moderate = (roc_s < -1) & (roc_s >= -5)
    # 中性区域：ROC在[-1, 1]之间
    roc_neutral = (roc_s >= -1) & (roc_s <= 1)
    # 在非交叉区域，根据ROC幅度调整分数，使用np.maximum/minimum确保分数合理叠加
    # 强劲上涨，分数上限提高到90.0
    score.loc[not_cross_cond & roc_positive_strong] = np.maximum(score.loc[not_cross_cond & roc_positive_strong], 90.0)
    # 温和上涨，分数上限调整为70.0
    score.loc[not_cross_cond & roc_positive_moderate] = np.maximum(score.loc[not_cross_cond & roc_positive_moderate], 70.0)
    # 强劲下跌，分数下限降低到10.0
    score.loc[not_cross_cond & roc_negative_strong] = np.minimum(score.loc[not_cross_cond & roc_negative_strong], 10.0)
    # 温和下跌，分数下限调整为30.0
    score.loc[not_cross_cond & roc_negative_moderate] = np.minimum(score.loc[not_cross_cond & roc_negative_moderate], 30.0)
    # 中性区域，明确设置为50.0
    score.loc[not_cross_cond & roc_neutral] = 50.0
    # 3. 趋势强度贡献 (在非交叉区域，根据ROC的变化率调整分数)
    # 优化：计算ROC变化量
    # 计算ROC的变化量
    roc_change = roc_s - roc_s_prev
    # 强劲上涨趋势：ROC为正且显著增加 (变化量大于1.5)
    # 增加变化量阈值，使强劲趋势更难达成
    strong_bullish_trend = (roc_s > 0) & (roc_change > 1.5) & not_cross_cond
    # 温和上涨趋势：ROC为正且温和增加 (变化量在(0, 1.5]之间)
    # 定义温和上涨趋势的条件
    moderate_bullish_trend = (roc_s > 0) & (roc_change > 0) & (roc_change <= 1.5) & not_cross_cond
    # 强劲下跌趋势：ROC为负且显著减少 (变化量小于-1.5)
    # 增加变化量阈值，使强劲趋势更难达成
    strong_bearish_trend = (roc_s < 0) & (roc_change < -1.5) & not_cross_cond
    # 温和下跌趋势：ROC为负且温和减少 (变化量在[-1.5, 0)之间)
    # 定义温和下跌趋势的条件
    moderate_bearish_trend = (roc_s < 0) & (roc_change < 0) & (roc_change >= -1.5) & not_cross_cond
    # 趋势衰减：ROC方向与趋势相反，但未交叉
    # 牛市衰退：ROC为正但开始下降
    # 定义牛市衰退的条件
    bullish_waning = (roc_s > 0) & (roc_change < 0) & not_cross_cond
    # 熊市衰退：ROC为负但开始上升
    # 定义熊市衰退的条件
    bearish_waning = (roc_s < 0) & (roc_change > 0) & not_cross_cond
    # 应用趋势贡献，进一步调整分数，使用np.maximum/minimum进行叠加
    # 强劲上涨趋势，分数上限提高到95.0
    score.loc[strong_bullish_trend] = np.maximum(score.loc[strong_bullish_trend], 95.0)
    # 温和上涨趋势，分数上限调整为75.0
    score.loc[moderate_bullish_trend] = np.maximum(score.loc[moderate_bullish_trend], 75.0)
    # 强劲下跌趋势，分数下限降低到5.0
    score.loc[strong_bearish_trend] = np.minimum(score.loc[strong_bearish_trend], 5.0)
    # 温和下跌趋势，分数下限调整为25.0
    score.loc[moderate_bearish_trend] = np.minimum(score.loc[moderate_bearish_trend], 25.0)
    # 衰退趋势会使分数向中性靠拢
    # 牛市衰退，分数下限调整为60.0
    score.loc[bullish_waning] = np.minimum(score.loc[bullish_waning], 60.0)
    # 熊市衰退，分数上限调整为40.0
    score.loc[bearish_waning] = np.maximum(score.loc[bearish_waning], 40.0)
    # 确保最终分数在0-100之间
    # 对最终分数进行裁剪，确保在有效范围内
    final_score = score.clip(0, 100)
    return final_score

def calculate_dmi_score(pdi: pd.Series, ndi: pd.Series, adx: pd.Series, params: Dict) -> pd.Series:
    """
    DMI 评分 (0-100)。
    深化并优化DMI评分规则，使其更具洞察力，并对代码进行效率优化。
    """
    # 使用 _safe_fillna_series 填充，确保数据完整性，避免NaN值影响计算
    # DMI/ADX 中性0，表示无方向或无趋势
    pdi_s, ndi_s, adx_s = _safe_fillna_series([pdi, ndi, adx], [0.0, 0.0, 0.0]) # 确保所有输入Series都被安全填充
    # 如果PDI完全是NaN，则返回默认中性分数，表示无法计算
    if pdi_s.isnull().all():
        print("调试信息: PDI完全为NaN，返回默认中性分数。") # 添加调试信息
        return pd.Series(50.0, index=pdi.index).clip(0, 100)
    # 从 params 字典获取参数，设置默认值
    adx_th = params.get('adx_threshold', 25) # 趋势弱/无趋势与趋势开始的阈值
    adx_strong_th = params.get('adx_strong_threshold', 40) # 趋势中等与趋势强劲的阈值
    # 引入新的ADX阈值，用于识别极度无趋势市场，此时DMI信号通常不可靠
    adx_very_weak_th = params.get('adx_very_weak_threshold', 15)
    # 初始化分数Series为中性值，作为默认或无法判断时的基准
    score = pd.Series(50.0, index=pdi_s.index, dtype=float) # 明确指定dtype为float
    # --- 预计算所有必要的布尔条件，提高后续计算效率 ---
    # 交叉信号判断
    buy_cross = (pdi_s.shift(1) < ndi_s.shift(1)) & (pdi_s >= ndi_s)
    sell_cross = (ndi_s.shift(1) < pdi_s.shift(1)) & (ndi_s >= pdi_s)
    # ADX趋势方向判断
    adx_rising = adx_s > adx_s.shift(1)
    adx_falling = adx_s < adx_s.shift(1) # 增加ADX下降的判断，用于趋势衰竭的惩罚
    # ADX强度区域划分
    adx_weak = adx_s <= adx_th
    adx_moderate = (adx_s > adx_th) & (adx_s <= adx_strong_th)
    adx_strong = adx_s > adx_strong_th
    adx_very_weak = adx_s <= adx_very_weak_th # 增加极弱ADX区域判断
    # PDI/NDI相对位置判断
    pdi_gt_ndi = pdi_s > ndi_s
    ndi_gt_pdi = ndi_s > pdi_s
    # 非交叉条件，用于判断纯粹的趋势跟随
    not_cross_cond = ~buy_cross & ~sell_cross
    # 趋势判断（在非交叉情况下）
    is_bullish_trend = pdi_gt_ndi & not_cross_cond
    is_bearish_trend = ndi_gt_pdi & not_cross_cond
    # PDI和NDI之间的绝对差异，用于衡量趋势的“纯度”或强度
    pdi_ndi_diff_abs = (pdi_s - ndi_s).abs() # 计算PDI和NDI的绝对差异
    # --- 定义条件和对应的分数，使用 np.select 进行高效赋值 ---
    # 条件的顺序非常重要，更具体、优先级更高的条件应放在前面
    conditions = []
    choices = []
    # 1. 极度无趋势市场 (ADX非常低)，分数强制接近中性
    # 这是最高优先级的条件，因为在这种情况下，DMI信号通常不可靠，应避免给出强烈的方向性指示
    conditions.append(adx_very_weak)
    choices.append(50.0) # 极弱ADX时，分数强制为50，表示市场无明确方向
    # 2. 交叉信号 (优先级高于纯粹的趋势跟随)
    # 2.1. 强劲买入交叉 (PDI上穿NDI，ADX强劲且上升) - 最强的买入信号
    conditions.append(buy_cross & adx_strong & adx_rising)
    choices.append(90.0) # 强劲买入交叉，给予极高分
    # 2.2. 强劲卖出交叉 (NDI上穿PDI，ADX强劲且上升) - 最强的卖出信号
    conditions.append(sell_cross & adx_strong & adx_rising)
    choices.append(10.0) # 强劲卖出交叉，给予极低分
    # 2.3. 确认买入交叉 (PDI上穿NDI，ADX强劲但未上升) - 强买入信号，但动能可能未加速
    conditions.append(buy_cross & adx_strong)
    choices.append(80.0) # 确认买入交叉，次高分
    # 2.4. 确认卖出交叉 (NDI上穿PDI，ADX强劲但未上升) - 强卖出信号，但动能可能未加速
    conditions.append(sell_cross & adx_strong)
    choices.append(20.0) # 确认卖出交叉，次低分
    # 2.5. 发展中买入交叉 (PDI上穿NDI，ADX中等且上升) - 趋势正在形成或加速
    conditions.append(buy_cross & adx_moderate & adx_rising)
    choices.append(75.0) # 发展中买入交叉
    # 2.6. 发展中卖出交叉 (NDI上穿PDI，ADX中等且上升) - 趋势正在形成或加速
    conditions.append(sell_cross & adx_moderate & adx_rising)
    choices.append(25.0) # 发展中卖出交叉
    # 2.7. 一般买入交叉 (PDI上穿NDI，ADX中等或弱) - 信号强度一般，可能趋势不明确
    conditions.append(buy_cross & (adx_moderate | adx_weak))
    choices.append(65.0) # 一般买入交叉
    # 2.8. 一般卖出交叉 (NDI上穿PDI，ADX中等或弱) - 信号强度一般，可能趋势不明确
    conditions.append(sell_cross & (adx_moderate | adx_weak))
    choices.append(35.0) # 一般卖出交叉
    # 3. 趋势跟随信号 (在没有交叉的情况下，PDI/NDI已形成优势)
    # 3.1. 极强牛市趋势 (PDI > NDI，ADX强劲且上升) - 趋势非常健康且加速
    conditions.append(is_bullish_trend & adx_strong & adx_rising)
    choices.append(85.0) # 极强牛市趋势
    # 3.2. 极强熊市趋势 (NDI > PDI，ADX强劲且上升) - 趋势非常健康且加速
    conditions.append(is_bearish_trend & adx_strong & adx_rising)
    choices.append(15.0) # 极强熊市趋势
    # 3.3. 强牛市趋势 (PDI > NDI，ADX强劲但未上升) - 趋势强劲但动能可能放缓
    conditions.append(is_bullish_trend & adx_strong)
    choices.append(75.0) # 强牛市趋势
    # 3.4. 强熊市趋势 (NDI > PDI，ADX强劲但未上升) - 趋势强劲但动能可能放缓
    conditions.append(is_bearish_trend & adx_strong)
    choices.append(25.0) # 强熊市趋势
    # 3.5. 中等牛市趋势 (PDI > NDI，ADX中等且上升) - 趋势正在发展或巩固
    conditions.append(is_bullish_trend & adx_moderate & adx_rising)
    choices.append(70.0) # 中等牛市趋势
    # 3.6. 中等熊市趋势 (NDI > PDI，ADX中等且上升) - 趋势正在发展或巩固
    conditions.append(is_bearish_trend & adx_moderate & adx_rising)
    choices.append(30.0) # 中等熊市趋势
    # 3.7. 一般牛市趋势 (PDI > NDI，ADX中等但未上升) - 趋势存在但可能波动或减速
    conditions.append(is_bullish_trend & adx_moderate)
    choices.append(60.0) # 一般牛市趋势
    # 3.8. 一般熊市趋势 (NDI > PDI，ADX中等但未上升) - 趋势存在但可能波动或减速
    conditions.append(is_bearish_trend & adx_moderate)
    choices.append(40.0) # 一般熊市趋势
    # 3.9. 弱牛市趋势 (PDI > NDI，ADX弱) - 趋势非常弱，可能只是短期偏向
    conditions.append(is_bullish_trend & adx_weak)
    choices.append(55.0) # 弱牛市趋势
    # 3.10. 弱熊市趋势 (NDI > PDI，ADX弱) - 趋势非常弱，可能只是短期偏向
    conditions.append(is_bearish_trend & adx_weak)
    choices.append(45.0) # 弱熊市趋势
    # 4. 震荡或无明确趋势 (ADX弱，PDI/NDI无明显优势)
    # 如果ADX弱，且PDI和NDI接近（差异小于某个阈值），则分数趋向中性
    conditions.append(adx_weak & (pdi_ndi_diff_abs < 5)) # 增加震荡市场判断条件
    choices.append(50.0) # 震荡市场分数，强制为50
    # 使用 np.select 应用所有条件和分数，default=50.0 确保没有匹配的条件时，分数保持中性
    score_array = np.select(conditions, choices, default=50.0) # 使用np.select进行高效赋值
    # 将numpy数组转换回pandas Series，并保留原始索引
    score = pd.Series(score_array, index=pdi_s.index) # 将结果转换回pandas Series
    # --- 后处理：根据PDI/NDI差异和ADX下降进行微调，增加洞察力 ---
    # 差异因子：将PDI/NDI的绝对差异映射到0-1的范围，用于调整分数
    # 假设PDI/NDI差异最大可能为100，这里用一个经验值30进行归一化，避免过度调整
    diff_factor = (pdi_ndi_diff_abs / 30).clip(0, 1) # 计算差异因子，用于微调分数
    # 调整幅度：差异越大，调整幅度越大，但有上限
    adjustment_magnitude = 5 * diff_factor # 定义调整幅度，最大为5分
    # 仅对非交叉且有明确趋势的信号进行差异调整，交叉信号已在np.select中给予较高优先级
    # 对于看涨信号 (score > 50)，如果PDI远大于NDI，分数可以更高
    # 使用np.minimum确保分数不会超过99，留有余地
    bullish_adjust_cond = is_bullish_trend & (score > 50)
    score.loc[bullish_adjust_cond] = np.minimum(score.loc[bullish_adjust_cond] + adjustment_magnitude.loc[bullish_adjust_cond], 99.0) # 根据PDI/NDI差异微调看涨分数
    # 对于看跌信号 (score < 50)，如果NDI远大于PDI，分数可以更低
    # 使用np.maximum确保分数不会低于1，留有余地
    bearish_adjust_cond = is_bearish_trend & (score < 50)
    score.loc[bearish_adjust_cond] = np.maximum(score.loc[bearish_adjust_cond] - adjustment_magnitude.loc[bearish_adjust_cond], 1.0) # 根据PDI/NDI差异微调看跌分数
    # ADX下降时的惩罚 (趋势可能正在减弱或面临反转)
    # 如果ADX在强趋势区域下降，对分数进行轻微惩罚，表示趋势动能减弱
    penalty_magnitude = 5 # 定义ADX下降惩罚幅度
    # 牛市趋势中ADX下降：分数向中性靠拢，但不会变成看跌
    # 惩罚只适用于非交叉的趋势跟随信号，且ADX仍处于强劲区域
    bullish_adx_falling_penalty_cond = is_bullish_trend & adx_falling & adx_strong
    score.loc[bullish_adx_falling_penalty_cond] = np.maximum(score.loc[bullish_adx_falling_penalty_cond] - penalty_magnitude, 50.0) # 牛市趋势中ADX下降，分数降低
    # 熊市趋势中ADX下降：分数向中性靠拢，但不会变成看涨
    bearish_adx_falling_penalty_cond = is_bearish_trend & adx_falling & adx_strong
    score.loc[bearish_adx_falling_penalty_cond] = np.minimum(score.loc[bearish_adx_falling_penalty_cond] + penalty_magnitude, 50.0) # 熊市趋势中ADX下降，分数升高 (趋向中性)
    # 最终分数裁剪到0-100范围，确保有效性
    final_score = score.clip(0, 100) # 最终分数裁剪
    return final_score

def calculate_sar_score(close: pd.Series, sar: pd.Series) -> pd.Series:
    """
    SAR 评分 (0-100)。
    深化指标评分规则，丰富计算规则，增加得分难度，使评分更具洞察力，并对代码做效率优化。
    """
    # print("开始计算SAR评分...") # 调试信息
    # 复制原始Series，避免修改传入的Series，并进行NaN处理
    close_s = close.copy() # 复制原始Series，避免修改原始数据
    sar_s = sar.copy()     # 复制原始Series，避免修改原始数据
    # 优化NaN处理：对close_s和sar_s进行前向填充，再后向填充，以处理序列中间的NaN。
    # 如果序列开头或结尾有NaN，且没有数据可填充，则这些NaN会保留。
    close_s = close_s.ffill().bfill() # 对close_s进行ffill和bfill填充
    sar_s = sar_s.ffill().bfill()     # 对sar_s进行ffill和bfill填充
    # 如果填充后仍有NaN（例如，原始序列就是全NaN），则返回中性分数。
    if close_s.isnull().all() or sar_s.isnull().all(): # 原始逻辑：如果close_s或sar_s全NaN，返回50
        print("输入数据全为NaN，返回中性分数。") # 调试信息
        return pd.Series(50.0, index=close.index).clip(0, 100)
    # 初始化分数，所有点默认为50分
    score = pd.Series(50.0, index=close_s.index) # 原始逻辑：初始化分数为50
    # 计算买入和卖出信号
    # 买入信号：SAR前一天在Close上方，今天SAR在Close下方或等于Close (SAR向下突破Close)
    buy_signal = (sar_s.shift(1) > close_s.shift(1)) & (sar_s <= close_s) # 原始逻辑：买入信号
    # 卖出信号：SAR前一天在Close下方，今天SAR在Close上方或等于Close (SAR向上突破Close)
    sell_signal = (sar_s.shift(1) < close_s.shift(1)) & (sar_s >= close_s) # 原始逻辑：卖出信号
    # 应用信号分数，信号分数具有最高优先级
    score.loc[buy_signal] = 75.0 # 原始逻辑：买入信号得75分
    score.loc[sell_signal] = 25.0 # 原始逻辑：卖出信号得25分
    # 优化：计算非信号条件，避免重复计算
    not_signal_cond = ~(buy_signal | sell_signal) # 使用或操作符简化非信号条件判断
    # 增强评分规则：考虑SAR与Close的相对距离和趋势强度
    # 仅对非信号日进行更细致的评分
    # 趋势强度因子：SAR与Close的百分比距离
    # 使用 np.where 避免除以零，并确保类型一致。如果close_s为0，则diff_ratio也为0。
    diff_ratio = np.where(close_s != 0, (close_s - sar_s) / close_s, 0) # 计算SAR与Close的百分比距离，避免除以零
    # 趋势向上 (Close > SAR) 且非信号日
    bullish_trend_cond = (close_s > sar_s) & not_signal_cond # 定义看涨趋势条件
    if bullish_trend_cond.any(): # 优化：只有当条件为真时才进行计算
        # 基础分60，额外分数根据diff_ratio调整，最大额外分数30 (总分90)
        # 调整系数1000意味着每1%的diff_ratio增加10分，3%的diff_ratio达到最高额外分30。
        additional_score_bullish = np.minimum(30.0, diff_ratio[bullish_trend_cond] * 1000) # 根据diff_ratio计算额外分数，上限30
        # 确保分数至少为60，然后加上额外分数
        score.loc[bullish_trend_cond] = np.maximum(score.loc[bullish_trend_cond], 60.0 + additional_score_bullish) # 应用增强后的看涨分数
    # 趋势向下 (Close < SAR) 且非信号日
    bearish_trend_cond = (close_s < sar_s) & not_signal_cond # 定义看跌趋势条件
    if bearish_trend_cond.any(): # 优化：只有当条件为真时才进行计算
        # 基础分40，额外分数根据diff_ratio调整，最大额外分数30 (总分10)
        # diff_ratio此时为负值，我们取其绝对值来计算减分。
        additional_score_bearish = np.minimum(30.0, -diff_ratio[bearish_trend_cond] * 1000) # 根据diff_ratio计算额外分数，上限30
        # 确保分数至多为40，然后减去额外分数
        score.loc[bearish_trend_cond] = np.minimum(score.loc[bearish_trend_cond], 40.0 - additional_score_bearish) # 应用增强后的看跌分数
    # 考虑SAR与Close非常接近的情况 (震荡或盘整)
    # 如果SAR和Close的绝对百分比差异小于某个阈值 (例如0.5%)，则分数更接近50。
    # 这种情况下，表示市场处于盘整或不确定状态，优先级低于明确的趋势强度。
    proximity_threshold = 0.005 # 0.5% # 定义接近阈值
    neutral_cond = not_signal_cond & (np.abs(diff_ratio) < proximity_threshold) # 定义中性条件
    if neutral_cond.any(): # 优化：只有当条件为真时才进行计算
        score.loc[neutral_cond] = 50.0 # 中性条件得50分，覆盖弱趋势分数
    # print("SAR评分计算完成。") # 调试信息
    return score.clip(0, 100) # 原始逻辑：分数裁剪到0-100

def calculate_stoch_score(k: pd.Series, d: pd.Series, params: Dict) -> pd.Series:
    """
    随机指标 (STOCH) 评分 (0-100)。
    深化指标的评分规则，丰富计算规则，增加得分难度，使评分更具洞察力，并对代码做效率优化。
    """
    # 使用 _safe_fillna_series 填充K线和D线，STOCH 中性值为50
    k_s, d_s = _safe_fillna_series([k, d], [50.0, 50.0])
    # 如果K线全部为空，则返回一个填充50的Series
    if k_s.isnull().all():
        print("调试信息: K线全部为空，返回默认评分50。") # 调试信息
        return pd.Series(50.0, index=k.index).clip(0, 100)
    # 初始化评分Series，默认值为50
    score = pd.Series(50.0, index=k_s.index, dtype=float)
    # 从 params 字典获取参数
    os = params.get('stoch_oversold', 20) # 超卖区阈值
    ob = params.get('stoch_overbought', 80) # 超买区阈值
    ext_os = params.get('stoch_extreme_oversold', 10) # 极端超卖区阈值
    ext_ob = params.get('stoch_extreme_overbought', 90) # 极端超买区阈值
    # 计算K线和D线的斜率（变化率）
    k_slope = k_s.diff()
    d_slope = d_s.diff()
    # 计算K线和D线之间的差值
    kd_spread = k_s - d_s
    # 定义斜率和价差变化的阈值，用于判断动量强弱
    slope_threshold = 3.0 # K/D线变化超过此值视为强动量
    spread_change_threshold = 1.0 # K-D价差变化超过此值视为价差动量
    # --- 1. 极端超买/超卖区评分 (最高优先级) ---
    # K线或D线进入极端超卖区，评分设为95
    extreme_oversold_cond = (k_s < ext_os) | (d_s < ext_os)
    score.loc[extreme_oversold_cond] = 95.0 # 极端超卖区评分
    # K线或D线进入极端超买区，评分设为5
    extreme_overbought_cond = (k_s > ext_ob) | (d_s > ext_ob)
    score.loc[extreme_overbought_cond] = 5.0 # 极端超买区评分
    # --- 2. 超买/超卖区评分 (次高优先级，在极端区之上应用最大/最小限制) ---
    # K线或D线进入超卖区 (非极端超卖)，评分至少为85
    oversold_cond = ((k_s >= ext_os) & (k_s < os)) | ((d_s >= ext_os) & (d_s < os))
    score.loc[oversold_cond] = np.maximum(score.loc[oversold_cond], 85.0) # 超卖区评分
    # K线或D线进入超买区 (非极端超买)，评分至多为15
    overbought_cond = ((k_s <= ext_ob) & (k_s > ob)) | ((d_s <= ext_ob) & (d_s > ob))
    score.loc[overbought_cond] = np.minimum(score.loc[overbought_cond], 15.0) # 超买区评分
    # --- 3. "钩子"或反转信号 (从极端区反转，强信号) ---
    # K线从极端超卖区向上反转 (前一周期在极端超卖区，当前周期K线向上且回到非极端区)
    k_bull_hook = (k_s.shift(1) < ext_os) & (k_s > k_s.shift(1)) & (k_s >= ext_os) # K线向上钩子条件
    # D线从极端超卖区向上反转
    d_bull_hook = (d_s.shift(1) < ext_os) & (d_s > d_s.shift(1)) & (d_s >= ext_os) # D线向上钩子条件
    any_bull_hook = k_bull_hook | d_bull_hook # 组合K/D线向上钩子条件
    score.loc[any_bull_hook] = np.maximum(score.loc[any_bull_hook], 90.0) # 向上钩子评分
    # K线从极端超买区向下反转
    k_bear_hook = (k_s.shift(1) > ext_ob) & (k_s < k_s.shift(1)) & (k_s <= ext_ob) # K线向下钩子条件
    # D线从极端超买区向下反转
    d_bear_hook = (d_s.shift(1) > ext_ob) & (d_s < d_s.shift(1)) & (d_s <= ext_ob) # D线向下钩子条件
    any_bear_hook = k_bear_hook | d_bear_hook # 组合K/D线向下钩子条件
    score.loc[any_bear_hook] = np.minimum(score.loc[any_bear_hook], 10.0) # 向下钩子评分
    # --- 4. 动量/斜率分析 (K线和D线同时强劲上涨/下跌) ---
    # K线和D线同时强劲上涨
    strong_bull_momentum = (k_slope > slope_threshold) & (d_slope > slope_threshold) # 强劲上涨动量条件
    # 在现有评分基础上增加，但不超过100
    score.loc[strong_bull_momentum] = np.clip(score.loc[strong_bull_momentum] + 3, 0, 100) # 强劲上涨动量评分调整
    # K线和D线同时强劲下跌
    strong_bear_momentum = (k_slope < -slope_threshold) & (d_slope < -slope_threshold) # 强劲下跌动量条件
    # 在现有评分基础上减少，但不低于0
    score.loc[strong_bear_momentum] = np.clip(score.loc[strong_bear_momentum] - 3, 0, 100) # 强劲下跌动量评分调整
    # --- 5. K-D价差分析 (K线和D线之间距离的变化) ---
    # K线在D线之上且价差扩大 (多头动量增强)
    bullish_spread_strengthening = (k_s > d_s) & (kd_spread.diff() > spread_change_threshold) # 多头价差增强条件
    score.loc[bullish_spread_strengthening] = np.clip(score.loc[bullish_spread_strengthening] + 2, 0, 100) # 多头价差增强评分调整
    # K线在D线之下且价差扩大 (空头动量增强)
    bearish_spread_strengthening = (k_s < d_s) & (kd_spread.diff() < -spread_change_threshold) # 空头价差增强条件
    score.loc[bearish_spread_strengthening] = np.clip(score.loc[bearish_spread_strengthening] - 2, 0, 100) # 空头价差增强评分调整
    # --- 6. 金叉/死叉信号 ---
    # 计算金叉和死叉
    buy_cross = (k_s.shift(1) < d_s.shift(1)) & (k_s >= d_s)
    sell_cross = (k_s.shift(1) > d_s.shift(1)) & (k_s <= d_s)
    # 金叉发生在超卖区 (D线低于超卖阈值)
    buy_cross_os = buy_cross & (d_s < os)
    # 金叉发生在超买区 (D线高于超买阈值)
    buy_cross_ob = buy_cross & (d_s > ob)
    # 死叉发生在超卖区 (D线低于超卖阈值)
    sell_cross_os = sell_cross & (d_s < os)
    # 死叉发生在超买区 (D线高于超买阈值)
    sell_cross_ob = sell_cross & (d_s > ob)
    # 应用金叉评分
    score.loc[buy_cross_os] = np.maximum(score.loc[buy_cross_os], 80.0) # 超卖区金叉评分
    score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)] = np.maximum(score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)], 75.0) # 中性区金叉评分
    score.loc[buy_cross_ob] = np.maximum(score.loc[buy_cross_ob], 60.0) # 超买区金叉评分
    # 应用死叉评分
    score.loc[sell_cross_ob] = np.minimum(score.loc[sell_cross_ob], 20.0) # 超买区死叉评分
    score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)] = np.minimum(score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)], 25.0) # 中性区死叉评分
    score.loc[sell_cross_os] = np.minimum(score.loc[sell_cross_os], 40.0) # 超卖区死叉评分
    # --- 7. 中性区趋势判断 (无金叉/死叉时) ---
    # 既无金叉也无死叉的条件
    not_cross_cond = ~buy_cross & ~sell_cross
    # K线和D线都在中性区 (非超买非超卖) 且无交叉
    neutral_stoch_zone = (k_s >= os) & (k_s <= ob) & (d_s >= os) & (d_s <= ob) & not_cross_cond
    # 中性区内K线和D线同时向上 (看涨趋势)
    bullish_trend_neutral = neutral_stoch_zone & (k_s > k_s.shift(1)) & (d_s > d_s.shift(1))
    score.loc[bullish_trend_neutral] = np.maximum(score.loc[bullish_trend_neutral], 55.0) # 中性区看涨趋势评分
    # 中性区内K线和D线同时向下 (看跌趋势)
    bearish_trend_neutral = neutral_stoch_zone & (k_s < k_s.shift(1)) & (d_s < d_s.shift(1))
    score.loc[bearish_trend_neutral] = np.minimum(score.loc[bearish_trend_neutral], 45.0) # 中性区看跌趋势评分
    # 确保最终评分在0到100之间
    final_score = score.clip(0, 100)
    return final_score

def calculate_ma_score(close: pd.Series, ma: pd.Series, params: Optional[Dict] = None) -> pd.Series:
    """
    移动平均线 (MA) 评分 (0-100)。
    深化评分规则，增加得分难度，使评分更具层次感，更具洞察力，并对代码做效率优化。
    """
    # 敏感度参数，可根据需要调整。这些参数决定了距离和趋势对分数的影响程度。
    # K_DISTANCE_SENSITIVITY: 距离 MA 的敏感度。值越大，收盘价与 MA 的微小距离变化对分数影响越大。
    #   例如，K=500 意味着 1% 的距离差异会带来 5 分的变化 (50 + 0.01 * 500 = 55)。
    K_DISTANCE_SENSITIVITY = 500
    # M_SLOPE_ADJUSTMENT: MA 斜率的调整幅度。值越大，MA 趋势对分数影响越大。
    #   例如，M=5 意味着 MA 上升/下降会带来 +/- 5 分的调整。
    M_SLOPE_ADJUSTMENT = 5
    # 使用 _safe_fillna_series 填充数据，确保 close_s 和 ma_s 尽可能完整
    # 修改点1: 调用 _safe_fillna_series 获取处理后的 close_s 和 ma_s
    close_s, ma_s = _safe_fillna_series(
        [close, ma],
        [None, lambda s: s.rolling(20, min_periods=1).mean()] # ma 填充后，如果全 NaN 使用 close 的滚动平均
    )
    # 如果 close 或 MA 全为 NaN，则返回默认分数 50。这是最基础的异常处理。
    if close_s.isnull().all() or ma_s.isnull().all():
        # 确保返回 Series 的索引是原始 close 的索引
        return pd.Series(50.0, index=close.index).clip(0, 100)
    # 1. 计算 MA 相对收盘价的百分比差异 (距离因子)
    # 修改点2: 计算 diff_ratio，衡量收盘价与MA的相对距离。
    # 使用 .replace(0, np.nan) 避免除以零，然后用 .fillna(0) 将这些 NaN 视为无差异。
    diff_ratio = (close_s - ma_s) / ma_s.replace(0, np.nan)
    diff_ratio = diff_ratio.fillna(0)
    # 2. 计算 MA 的斜率 (趋势因子)
    # 修改点3: 计算 ma_slope，衡量MA的趋势方向。
    # .diff() 会在第一个元素处产生 NaN，用 .fillna(0) 填充。
    ma_slope = ma_s.diff().fillna(0)
    # 获取斜率方向 (-1: 下降, 0: 持平, 1: 上升)
    ma_slope_direction = np.sign(ma_slope)
    # 3. 初始化分数：基于距离的连续评分
    # 初始分数以 50 为基准，根据 diff_ratio 和 K_DISTANCE_SENSITIVITY 调整。
    # 距离 MA 越远，分数越偏离 50。
    # 修改点4: 初始化 score，结合距离敏感度 K_DISTANCE_SENSITIVITY
    score = 50 + diff_ratio * K_DISTANCE_SENSITIVITY
    # 4. 调整分数：基于 MA 趋势的微调
    # 根据 MA 的上升或下降趋势，对分数进行 M_SLOPE_ADJUSTMENT 幅度的微调。
    # 修改点5: 根据 MA 斜率方向 M_SLOPE_ADJUSTMENT 调整 score
    score += ma_slope_direction * M_SLOPE_ADJUSTMENT
    # 5. 识别买入和卖出交叉点 (事件因子)
    # 交叉点是明确的信号，其分数应具有最高优先级，覆盖之前的连续评分。
    # 修改点6: 优化交叉点判断逻辑，使用向量化操作
    prev_close = close_s.shift(1)
    prev_ma = ma_s.shift(1)
    # 买入交叉：前一周期收盘价低于MA，当前周期收盘价高于或等于MA
    buy_cross = (prev_close < prev_ma) & (close_s >= ma_s)
    # 卖出交叉：前一周期收盘价高于MA，当前周期收盘价低于或等于MA
    sell_cross = (prev_close > prev_ma) & (close_s <= ma_s)
    # 应用交叉点分数 (覆盖之前的连续评分)
    # 修改点7: 应用买入交叉点分数
    score.loc[buy_cross] = 70.0
    # 修改点8: 应用卖出交叉点分数
    score.loc[sell_cross] = 30.0
    # 6. 确保最终分数在 0-100 范围内
    # 修改点9: 使用 clip 确保分数在有效范围内
    final_score = score.clip(0, 100)
    return final_score

def calculate_atr_score(atr: pd.Series) -> pd.Series:
    """
    ATR 评分 (0-100)。
    深化指标的评分规则，丰富计算规则，增加得分难度，使评分更具层次感，更具洞察力，并对代码做效率优化。
    """
    # 使用 _safe_fillna_series 填充原始 ATR Series
    # 使用 _safe_fillna_series 确保 atr_s 有效数据
    atr_s, = _safe_fillna_series([atr], [lambda s: s.mean()]) # atr 填充后，如果全 NaN 使用平均值
    # 如果填充后仍然全 NaN 或平均值为 0，则返回默认分数 50
    if atr_s.isnull().all() or atr_s.mean() == 0:
        # 确保返回 Series 的索引是原始 atr 的索引
        return pd.Series(50.0, index=atr.index).clip(0,100)
    # 确保滚动窗口大小不超过 Series 长度
    rolling_window = min(len(atr_s), 20)
    min_periods_rolling = max(1, int(rolling_window * 0.5)) if rolling_window > 0 else 1
    # 计算 ATR 的滚动平均值和标准差
    # 优化 rolling().mean() 和 rolling().std() 的 fillna 逻辑
    atr_mean = atr_s.rolling(window=rolling_window, min_periods=min_periods_rolling).mean().fillna(atr_s.mean())
    # 如果 atr_std 计算结果全为 NaN (例如，min_periods 太大或数据不足)，则用 0 填充，否则用其自身的均值填充
    atr_std = atr_s.rolling(window=rolling_window, min_periods=min_periods_rolling).std().fillna(atr_s.std()).fillna(0)
    # 如果 atr_std 的平均值为 0，表示波动性极低或无波动，所有分数设为 50
    # 增加对 atr_std 为 0 的特殊处理，提高鲁棒性
    if atr_std.mean() == 0:
        return pd.Series(50.0, index=atr.index).clip(0,100)
    # 定义更精细的波动率区间阈值
    # 增加更多波动率区间，使评分更具层次感
    threshold_3_upper = atr_mean + 1.5 * atr_std # 极高波动率上限
    threshold_2_upper = atr_mean + 1.0 * atr_std # 较高波动率上限
    threshold_1_upper = atr_mean + 0.5 * atr_std # 中高波动率上限
    threshold_3_lower = atr_mean - 1.5 * atr_std # 极低波动率下限
    threshold_2_lower = atr_mean - 1.0 * atr_std # 较低波动率下限
    threshold_1_lower = atr_mean - 0.5 * atr_std # 中低波动率下限
    # 定义评分条件和对应的分数，使用 np.select 批量应用
    # 条件顺序从最极端到最不极端，确保正确匹配
    conditions = [
        atr_s > threshold_3_upper,  # 极高波动率
        atr_s > threshold_2_upper,  # 较高波动率
        atr_s > threshold_1_upper,  # 中高波动率
        atr_s < threshold_3_lower,  # 极低波动率
        atr_s < threshold_2_lower,  # 较低波动率
        atr_s < threshold_1_lower   # 中低波动率
    ]
    choices = [
        95.0,  # 极高波动率得分
        80.0,  # 较高波动率得分
        65.0,  # 中高波动率得分
        5.0,   # 极低波动率得分
        20.0,  # 较低波动率得分
        35.0   # 中低波动率得分
    ]
    # 使用 np.select 批量应用分数，默认值为 50.0 (中等波动率)
    score = pd.Series(np.select(conditions, choices, default=50.0), index=atr_s.index)
    # 增加 ATR 趋势的洞察力：如果 ATR 正在上升，略微提高分数；如果下降，略微降低分数
    # 引入 ATR 趋势作为评分调整因子
    # 计算 ATR 的短期动量 (例如，5周期均值变化)
    atr_momentum = atr_s.diff(periods=1).rolling(window=5, min_periods=1).mean().fillna(0) # 填充 NaN 为 0
    # 根据动量调整分数
    # 动量为正 (ATR 上升) 增加分数，动量为负 (ATR 下降) 减少分数
    score_adjustment = np.zeros_like(score, dtype=float) # 初始化调整值数组
    score_adjustment = np.where(atr_momentum > 0, 5.0, score_adjustment) # ATR 上升，加 5 分
    score_adjustment = np.where(atr_momentum < 0, -5.0, score_adjustment) # ATR 下降，减 5 分
    score += score_adjustment # 应用调整
    # 确保分数在 0 到 100 之间
    return score.clip(0, 100)

def calculate_adl_score(adl: pd.Series) -> pd.Series:
    """
    深化 ADL (Accumulation/Distribution Line) 评分 (0-100)。
    规则:
    - 基础分: 50 (中性)
    - ADL 变化方向 (adl_diff):
        - 上升 (adl_diff > 0): +10 (基础看涨至60)
        - 下降 (adl_diff < 0): -10 (基础看跌至40)
    - ADL 变化幅度 (与自身近期波动比较):
        - 显著上升 (adl_diff > 1.0 * rolling_std_adl_diff): 再 +15
        - 显著下降 (adl_diff < -1.0 * rolling_std_adl_diff): 再 -15
    - ADL 趋势确认 (ADL与其20期均线的关系):
        - 若 adl_diff > 0 且 ADL > ADL_SMA20: 再 +10
        - 若 adl_diff < 0 且 ADL < ADL_SMA20: 再 -10
    - ADL 短期动能 (ADL的5期均线方向):
        - 若 adl_diff > 0 且 ADL_SMA5 上升: 再 +5
        - 若 adl_diff < 0 且 ADL_SMA5 下降: 再 -5
    最终得分会clip在0-100之间。
    """
    # 使用 _safe_fillna_series 填充，ADL本身可正可负，其含义在于变化和趋势，
    # 此处假设 _safe_fillna_series 主要是为了处理NaN，而不是改变ADL的基准。
    # 如果ADL序列本身有意义的零点，填充0是合理的。如果ADL是累积值，可能需要bfill/ffill。
    # 鉴于原始代码使用0.0填充，我们继续沿用，但需注意ADL的特性。
    adl_s, = _safe_fillna_series([adl], [0.0]) # 假设 _safe_fillna_series 返回元组
    # 如果数据过少或全部为NaN，返回中性分50
    # 至少需要几个点才能进行diff和rolling计算，例如rolling(window=N)至少需要N个点才有第一个有效值
    # diff() 需要2个点，rolling(window=10, min_periods=1).std() 在diff后至少需要1个diff值
    # adl_s.rolling(window=20, min_periods=5) 需要至少5个点
    if adl_s.isnull().all() or len(adl_s) < 5: # 修改最小长度要求以适应更长的窗口期
        return pd.Series(50.0, index=adl.index).clip(0, 100)
    score = pd.Series(50.0, index=adl_s.index) # 初始化评分为50
    # 1. ADL 变化量
    adl_diff = adl_s.diff()
    # print(f"DEBUG: adl_diff tail:\n{adl_diff.tail()}")
    # 2. ADL 变化方向基础评分调整
    # 根据adl_diff调整分数
    score.loc[adl_diff > 0] += 10.0 # ADL 上升，基础分变为60
    score.loc[adl_diff < 0] -= 10.0 # ADL 下降，基础分变为40
    # print(f"DEBUG: Score after basic trend adjustment tail:\n{score.tail()}")
    # 3. ADL 变化幅度评分调整
    # 使用过去N期 ADL diff 的滚动标准差来衡量“显著”变化
    # min_periods=1 确保即使窗口未满也有值，但可能不够稳定，可适当调整
    rolling_std_adl_diff = adl_diff.rolling(window=10, min_periods=2).std()
    # 填充由于滚动窗口初期产生的NaN，使用向后填充，然后向前填充，最后用一个极小值避免除零
    rolling_std_adl_diff = rolling_std_adl_diff.fillna(method='bfill').fillna(method='ffill').fillna(1e-9)
    # print(f"DEBUG: rolling_std_adl_diff tail:\n{rolling_std_adl_diff.tail()}")
    significant_change_multiplier = 1.0 # 定义显著变化的乘数阈值
    # 定义显著上升条件
    is_significant_rise = (adl_diff > 0) & (adl_diff > (significant_change_multiplier * rolling_std_adl_diff))
    # 定义显著下降条件
    is_significant_fall = (adl_diff < 0) & (adl_diff < -(significant_change_multiplier * rolling_std_adl_diff))
    # 根据显著变化调整分数
    score.loc[is_significant_rise] += 15.0 # 显著上升，再加分
    score.loc[is_significant_fall] -= 15.0 # 显著下降，再减分
    # print(f"DEBUG: Score after magnitude adjustment tail:\n{score.tail()}")
    # 4. ADL 趋势确认 (ADL 与其长期均线的关系)
    adl_sma_long_window = 20
    # 计算ADL的长期均线，并填充NaN
    adl_sma_long = adl_s.rolling(window=adl_sma_long_window, min_periods=max(1, adl_sma_long_window // 2)).mean().fillna(method='bfill').fillna(method='ffill')
    # print(f"DEBUG: adl_s tail:\n{adl_s.tail()}")
    # print(f"DEBUG: adl_sma_long tail:\n{adl_sma_long.tail()}")
    # 定义看涨趋势确认条件
    condition_bullish_trend_confirm = (adl_diff > 0) & (adl_s > adl_sma_long)
    score.loc[condition_bullish_trend_confirm] += 10.0
    # 定义看跌趋势确认条件
    condition_bearish_trend_confirm = (adl_diff < 0) & (adl_s < adl_sma_long)
    score.loc[condition_bearish_trend_confirm] -= 10.0
    # print(f"DEBUG: Score after trend confirmation tail:\n{score.tail()}")
    # 5. ADL 短期动能 (ADL 的短期均线方向)
    adl_sma_short_window = 5
    # 计算ADL的短期均线，并填充NaN
    adl_sma_short = adl_s.rolling(window=adl_sma_short_window, min_periods=max(1, adl_sma_short_window // 2)).mean().fillna(method='bfill').fillna(method='ffill')
    adl_sma_short_diff = adl_sma_short.diff() # 短期均线的变化
    # print(f"DEBUG: adl_sma_short_diff tail:\n{adl_sma_short_diff.tail()}")
    # 定义看涨动能条件
    condition_bullish_momentum = (adl_diff > 0) & (adl_sma_short_diff > 0)
    score.loc[condition_bullish_momentum] += 5.0
    # 定义看跌动能条件
    condition_bearish_momentum = (adl_diff < 0) & (adl_sma_short_diff < 0)
    score.loc[condition_bearish_momentum] -= 5.0
    # print(f"DEBUG: Score after momentum adjustment tail:\n{score.tail()}")
    # 确保分数在0-100之间
    final_score = score.clip(0, 100)
    # print(f"DEBUG: Final score (clipped) tail:\n{final_score.tail()}")
    return final_score

def calculate_vwap_score(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """VWAP (Volume Weighted Average Price) 评分 (0-100)。
    深化指标的评分规则，丰富计算规则，增加得分难度，使评分更具层次感，更具洞察力，并对代码做效率优化。
    """
    # 使用 _safe_fillna_series 填充
    # vwap 填充后，如果全 NaN 使用均值，再不行用 close 均值
    # 确保 _safe_fillna_series 的调用和参数与描述一致
    close_s, vwap_s = _safe_fillna_series([close, vwap], [None, lambda s: s.mean() if s.mean() is not np.nan else close.mean()])
    # 优化全 NaN 检查，并确保返回 Series 的索引是原始 close 的索引
    if close_s.isnull().all() or vwap_s.isnull().all():
        return pd.Series(50.0, index=close.index).clip(0, 100)
    # 计算VWAP百分比差异，更精细化评分
    # 避免除以零：如果vwap_s为0，则diff_pct会是inf/-inf，这在后续比较中是可接受的。
    # 如果vwap_s为NaN，diff_pct也会是NaN，这需要后续处理。
    diff_pct = (close_s - vwap_s) / vwap_s
    # 定义评分阈值和对应的分数
    # 引入更细致的评分区间，增加层次感
    threshold_high_pos = 0.02  # 收盘价显著高于VWAP的百分比阈值 (例如 > 2%)
    threshold_mod_pos = 0.01   # 收盘价适度高于VWAP的百分比阈值 (例如 > 1% 且 <= 2%)
    threshold_slight_pos = 0.005 # 收盘价略高于VWAP的百分比阈值 (例如 > 0.5% 且 <= 1%)
    threshold_slight_neg = -0.005 # 收盘价略低于VWAP的百分比阈值 (例如 < -0.5% 且 >= -1%)
    threshold_mod_neg = -0.01  # 收盘价适度低于VWAP的百分比阈值 (例如 < -1% 且 >= -2%)
    threshold_high_neg = -0.02 # 收盘价显著低于VWAP的百分比阈值 (例如 <= -2%)
    # 定义条件列表
    conditions = [
        diff_pct > threshold_high_pos,
        (diff_pct > threshold_mod_pos) & (diff_pct <= threshold_high_pos),
        (diff_pct > threshold_slight_pos) & (diff_pct <= threshold_mod_pos),
        (diff_pct >= threshold_slight_neg) & (diff_pct <= threshold_slight_pos), # 中性区间 (例如 -0.5% <= diff_pct <= 0.5%)
        (diff_pct > threshold_mod_neg) & (diff_pct < threshold_slight_neg),
        (diff_pct > threshold_high_neg) & (diff_pct <= threshold_mod_neg),
        diff_pct <= threshold_high_neg
    ]
    # 定义与条件对应的分数列表
    scores = [90.0, 75.0, 60.0, 50.0, 40.0, 25.0, 10.0] # MODIFIED
    # 使用np.select进行矢量化计算，提高效率
    # default=np.nan 表示如果所有条件都不满足（例如diff_pct是NaN），则结果为NaN
    score_raw = np.select(conditions, scores, default=np.nan)
    # 将结果转换为Pandas Series，并保留原始索引
    score = pd.Series(score_raw, index=close_s.index)
    # 处理因diff_pct为NaN而导致的score为NaN的情况，将其填充为中性分数50
    score = score.fillna(50.0)
    # 确保分数在0到100之间
    return score.clip(0, 100)

def calculate_ichimoku_score(close: pd.Series, tenkan: pd.Series, kijun: pd.Series, senkou_a: pd.Series, senkou_b: pd.Series, chikou: pd.Series) -> pd.Series:
    """
    Ichimoku (一目均衡表) 评分 (0-100)。
    深化评分规则，增加得分难度，使评分更具层次感，更具洞察力，并对代码做效率优化。
    """
    # 使用 _safe_fillna_series 填充输入 Series
    series_list = [close, tenkan, kijun, senkou_a, senkou_b, chikou]
    # 填充策略：close 优先 ffill/bfill，其他线如果全 NaN 使用 close 的均值，再不行使用 50.0
    fill_values = [
        None, # close 优先 ffill/bfill，如果全 NaN，_safe_fillna_series 会尝试用均值
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0),
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0),
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0),
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0),
        lambda s: s.mean() if s is not None and s.mean() is not np.nan else (close.mean() if close is not None and close.mean() is not np.nan else 50.0)
        ]
    c, tk, kj, sa, sb, cs = _safe_fillna_series(series_list, fill_values)
    # 检查填充后是否有 Series 仍然是全 NaN
    if c.isnull().all() or tk.isnull().all() or kj.isnull().all() or sa.isnull().all() or sb.isnull().all() or cs.isnull().all():
        logger.warning("Ichimoku: One or more lines are all NaN after filling.")
        # 确保返回 Series 的索引是原始 close 的索引
        return pd.Series(50.0, index=close.index).clip(0,100)
    # 初始化得分，所有点都从 0 开始累加，最后加上 50 作为基准分
    score = pd.Series(0.0, index=c.index) # 初始分改为 0.0，方便累加正负分数
    # 定义每个指标类别的最大贡献分数，用于调整权重和最终得分范围
    # 调整权重为最大贡献分数，使评分更具层次感
    max_price_kijun_points = 25 # 价格与基准线
    max_tk_kj_cross_points = 20 # 转换线与基准线交叉
    max_price_cloud_points = 30 # 价格与云
    max_cloud_twist_points = 10 # 云扭曲
    max_chikou_price_points = 15 # 迟行线与价格
    # 预计算常用条件和辅助指标以提高效率
    # 预计算斜率和云边界，提高效率
    kj_diff = kj.diff() # 基准线斜率
    tk_diff = tk.diff() # 转换线斜率
    cloud_top = np.maximum(sa, sb)
    cloud_bottom = np.minimum(sa, sb)
    price_26_ago = c.shift(26) # 迟行线比较的26周期前的价格
    chikou_valid = price_26_ago.notna() # 迟行线有效性
    # 1. 价格与基准线 (Price vs Kijun)
    # 价格在基准线上方：看涨信号
    bullish_pk = (c > kj)
    # 价格在基准线下方：看跌信号
    bearish_pk = (c < kj)
    # 价格向上突破基准线：强看涨信号
    pk_up_cross = (c.shift(1) < kj.shift(1)) & (c >= kj)
    # 价格向下突破基准线：强看跌信号
    pk_dn_cross = (c.shift(1) > kj.shift(1)) & (c <= kj)
    # 细化价格与基准线的评分规则
    # 价格在基准线上方且基准线向上：强看涨，贡献大部分分数
    score.loc[bullish_pk & (kj_diff > 0)] += max_price_kijun_points * 0.8
    # 价格在基准线上方但基准线向下或平坦：弱看涨，贡献较少分数
    score.loc[bullish_pk & (kj_diff <= 0)] += max_price_kijun_points * 0.4
    # 价格向上突破基准线：额外加分，表示动能增强
    score.loc[pk_up_cross] += max_price_kijun_points * 0.5
    # 价格在基准线下方且基准线向下：强看跌，贡献大部分负分数
    score.loc[bearish_pk & (kj_diff < 0)] -= max_price_kijun_points * 0.8
    # 价格在基准线下方但基准线向上或平坦：弱看跌，贡献较少负分数
    score.loc[bearish_pk & (kj_diff >= 0)] -= max_price_kijun_points * 0.4
    # 价格向下突破基准线：额外减分，表示动能减弱
    score.loc[pk_dn_cross] -= max_price_kijun_points * 0.5
    # 2. 转换线与基准线交叉 (Tenkan/Kijun Cross)
    # 转换线向上突破基准线 (金叉)：强看涨信号
    tk_kj_up_cross = (tk.shift(1) < kj.shift(1)) & (tk >= kj)
    # 转换线向下突破基准线 (死叉)：强看跌信号
    tk_kj_dn_cross = (tk.shift(1) > kj.shift(1)) & (tk <= kj)
    # 细化转换线与基准线交叉的评分规则
    # 金叉：显著加分
    score.loc[tk_kj_up_cross] += max_tk_kj_cross_points * 1.0
    # 死叉：显著减分
    score.loc[tk_kj_dn_cross] -= max_tk_kj_cross_points * 1.0
    # 转换线在基准线上方：持续看涨信号
    score.loc[tk > kj] += max_tk_kj_cross_points * 0.3
    # 转换线在基准线下方：持续看跌信号
    score.loc[tk < kj] -= max_tk_kj_cross_points * 0.3
    # 3. 价格与云 (Price vs Cloud)
    # 价格在云上方：强看涨信号
    price_above_cloud = c > cloud_top
    # 价格在云下方：强看跌信号
    price_below_cloud = c < cloud_bottom
    # 价格在云中
    price_in_cloud = (c >= cloud_bottom) & (c <= cloud_top)
    # 细化价格与云的评分规则
    # 价格在云上方且云是看涨云 (Senkou A > Senkou B)：最强看涨，贡献全部分数
    score.loc[price_above_cloud & (sa > sb)] += max_price_cloud_points * 1.0
    # 价格在云上方但云是看跌云 (Senkou A < Senkou B)：看涨但有隐忧，贡献部分分数
    score.loc[price_above_cloud & (sa < sb)] += max_price_cloud_points * 0.6
    # 价格向上突破云：额外加分，表示趋势反转或加强
    price_up_break_cloud = (c.shift(1) <= cloud_top.shift(1)) & (c > cloud_top)
    score.loc[price_up_break_cloud] += max_price_cloud_points * 0.7
    # 价格在云下方且云是看跌云 (Senkou A < Senkou B)：最强看跌，贡献全部负分数
    score.loc[price_below_cloud & (sa < sb)] -= max_price_cloud_points * 1.0
    # 价格在云下方但云是看涨云 (Senkou A > Senkou B)：看跌但有支撑，贡献部分负分数
    score.loc[price_below_cloud & (sa > sb)] -= max_price_cloud_points * 0.6
    # 价格向下突破云：额外减分，表示趋势反转或加强
    price_dn_break_cloud = (c.shift(1) >= cloud_bottom.shift(1)) & (c < cloud_bottom)
    score.loc[price_dn_break_cloud] -= max_price_cloud_points * 0.7
    # 价格在云中：根据价格方向调整分数，表示震荡中的偏向
    score.loc[price_in_cloud & (c > c.shift(1))] += max_price_cloud_points * 0.2
    score.loc[price_in_cloud & (c < c.shift(1))] -= max_price_cloud_points * 0.2
    # 4. 云扭曲 (Cloud Twist - Senkou A vs Senkou B) - 未来信号
    # 领先线A向上突破领先线B (看涨云扭曲)：未来看涨信号
    cloud_twist_up = (sa.shift(1) < sb.shift(1)) & (sa >= sb)
    # 领先线A向下突破领先线B (看跌云扭曲)：未来看跌信号
    cloud_twist_dn = (sa.shift(1) > sb.shift(1)) & (sa <= sb)
    # 细化云扭曲的评分规则
    score.loc[cloud_twist_up] += max_cloud_twist_points * 0.8
    score.loc[cloud_twist_dn] -= max_cloud_twist_points * 0.8
    # 云的颜色和斜率：持续性信号，反映未来趋势的强度
    # 看涨云 (Senkou A > Senkou B)
    bullish_cloud = (sa > sb)
    # 看跌云 (Senkou A < Senkou B)
    bearish_cloud = (sa < sb)
    # 云的中心线斜率 (反映云的整体趋势)
    cloud_slope = (sa + sb).diff()
    # 增加云的颜色和斜率对评分的影响
    # 看涨云且云体向上倾斜：更强的未来看涨信号
    score.loc[bullish_cloud & (cloud_slope > 0)] += max_cloud_twist_points * 0.5
    # 看跌云且云体向下倾斜：更强的未来看跌信号
    score.loc[bearish_cloud & (cloud_slope < 0)] -= max_cloud_twist_points * 0.5
    # 5. 迟行线与价格 (Chikou Span vs Price)
    # 迟行线在26周期前的价格上方：看涨信号
    cs_above_price = cs > price_26_ago
    # 迟行线在26周期前的价格下方：看跌信号
    cs_below_price = cs < price_26_ago
    # 细化迟行线与价格的评分规则
    # 迟行线在26周期前的价格上方且有效：强看涨
    score.loc[cs_above_price & chikou_valid] += max_chikou_price_points * 1.0
    # 迟行线在26周期前的价格下方且有效：强看跌
    score.loc[cs_below_price & chikou_valid] -= max_chikou_price_points * 1.0
    # 迟行线与云的关系 (更强的趋势确认信号)
    #迟行线与云的关系，这是非常重要的确认信号
    # 迟行线在26周期前的云上方：更强看涨，表示价格已突破未来阻力
    cs_above_cloud_26_ago = (cs > cloud_top.shift(26)) & chikou_valid
    # 迟行线在26周期前的云下方：更强看跌，表示价格已跌破未来支撑
    cs_below_cloud_26_ago = (cs < cloud_bottom.shift(26)) & chikou_valid
    score.loc[cs_above_cloud_26_ago] += max_chikou_price_points * 0.5
    score.loc[cs_below_cloud_26_ago] -= max_chikou_price_points * 0.5
    # 最终得分 = 初始基准分 50 + 累加分数
    final_score = 50 + score # 最终分数计算
    return final_score.clip(0, 100) # 确保分数在 0-100 范围内

def calculate_mom_score(mom: pd.Series) -> pd.Series:
    """
    MOM (Momentum) 深度评分 (0-100)。
    规则更丰富，评分更具层次感和洞察力。
    """
    # 使用 _safe_fillna_series 填充，MOM 中性值为 0.0
    mom_s, = _safe_fillna_series([mom], [0.0])
    # 如果填充后所有值仍然是 NaN (理论上_safe_fillna_series会处理，但作为双重检查)
    # 或者如果 mom_s 只有一个值（无法计算 shift 和 diff）
    if mom_s.isnull().all() or len(mom_s) < 2:
        return pd.Series(50.0, index=mom.index).clip(0, 100)
    # 初始化分数为50
    score = pd.Series(50.0, index=mom_s.index)
    # --- 预计算和衍生变量 ---
    mom_shifted = mom_s.shift(1)
    mom_diff = mom_s.diff() # MOM的变化率，即动量的加速度
    # 对mom_diff也进行填充，其中性值为0 (无变化)
    mom_diff_filled, = _safe_fillna_series([mom_diff], [0.0])
    # --- 计算动态阈值所需的分位数 ---
    # 仅使用非零值计算分位数，以获得更有意义的阈值
    mom_s_positive = mom_s[mom_s > 1e-6] # 考虑一个小的epsilon避免浮点数问题
    mom_s_negative = mom_s[mom_s < -1e-6]
    # 正MOM的分位数
    q_mom_pos_25 = mom_s_positive.quantile(0.25) if not mom_s_positive.empty else 0
    q_mom_pos_50 = mom_s_positive.quantile(0.50) if not mom_s_positive.empty else 0
    q_mom_pos_75 = mom_s_positive.quantile(0.75) if not mom_s_positive.empty else 0
    q_mom_pos_90 = mom_s_positive.quantile(0.90) if not mom_s_positive.empty else 0
    max_mom_pos = mom_s_positive.max() if not mom_s_positive.empty else 0
    # 负MOM的分位数 (注意quantile对于负数，0.25代表更负的值)
    q_mom_neg_25 = mom_s_negative.quantile(0.25) if not mom_s_negative.empty else 0 # 更负的值
    q_mom_neg_50 = mom_s_negative.quantile(0.50) if not mom_s_negative.empty else 0
    q_mom_neg_75 = mom_s_negative.quantile(0.75) if not mom_s_negative.empty else 0 # 接近0的负值
    # 定义 q_mom_neg_10
    q_mom_neg_10 = mom_s_negative.quantile(0.10) if not mom_s_negative.empty else 0 # 极负值判断
    min_mom_neg = mom_s_negative.min() if not mom_s_negative.empty else 0
    # MOM变化率的分位数
    mom_diff_positive = mom_diff_filled[mom_diff_filled > 1e-6]
    mom_diff_negative = mom_diff_filled[mom_diff_filled < -1e-6]
    q_diff_pos_50 = mom_diff_positive.quantile(0.50) if not mom_diff_positive.empty else 0
    q_diff_pos_75 = mom_diff_positive.quantile(0.75) if not mom_diff_positive.empty else 0
    q_diff_neg_50 = mom_diff_negative.quantile(0.50) if not mom_diff_negative.empty else 0 # 绝对值较小的负变化
    q_diff_neg_25 = mom_diff_negative.quantile(0.25) if not mom_diff_negative.empty else 0 # 绝对值较大的负变化
    # --- 1. 穿越信号 (金叉/死叉) ---
    buy_cross = (mom_shifted < 0) & (mom_s >= 0)
    sell_cross = (mom_shifted > 0) & (mom_s <= 0)
    # 金叉基础分和增强
    score.loc[buy_cross] = 60.0
    if q_mom_pos_50 > 0:
        score.loc[buy_cross & (mom_s >= q_mom_pos_50)] = 65.0
    if q_mom_pos_75 > 0:
        score.loc[buy_cross & (mom_s >= q_mom_pos_75)] = 70.0
    # 死叉基础分和增强
    score.loc[sell_cross] = 40.0
    if q_mom_neg_50 < 0:
        score.loc[sell_cross & (mom_s <= q_mom_neg_50)] = 35.0
    if q_mom_neg_25 < 0:
        score.loc[sell_cross & (mom_s <= q_mom_neg_25)] = 30.0
    # --- 2. 趋势持续与强度 (非穿越时期) ---
    not_cross_cond = ~buy_cross & ~sell_cross
    # 2.1 多头趋势 (mom_s > 0 且非金叉点)
    bullish_trend_cond = not_cross_cond & (mom_s > 1e-6)
    if bullish_trend_cond.any():
        # 动态构建 xp 和 fp，包含 q_mom_pos_25
        xp_raw = [0, q_mom_pos_25, q_mom_pos_50, q_mom_pos_75, q_mom_pos_90, max_mom_pos]
        fp_raw = [50.0, 55.0, 60.0, 70.0, 80.0, 85.0] # 对应的分数
        xp = []
        fp = []
        # 始终添加第一个点
        xp.append(xp_raw[0])
        fp.append(fp_raw[0])
        # 遍历其余点，只在 x 严格递增时添加，如果 x 相同则取更高的 y
        for i in range(1, len(xp_raw)):
            if xp_raw[i] > xp[-1]:
                xp.append(xp_raw[i])
                fp.append(fp_raw[i])
            elif xp_raw[i] == xp[-1]: # 如果 x 相同，取更高的分数 (更看涨)
                fp[-1] = max(fp[-1], fp_raw[i])
        # 确保至少有两个点用于插值，否则使用简化版
        if len(xp) < 2:
            if max_mom_pos > 0:
                xp = [0, max_mom_pos]
                fp = [50.0, 75.0] # 简化范围
            else: # 所有正MOM值都为0或接近0
                xp = [0, 1e-6] # 小范围以保持分数在50
                fp = [50.0, 50.0]
        current_bullish_scores = np.interp(mom_s[bullish_trend_cond], xp, fp)
        score.loc[bullish_trend_cond] = np.maximum(score.loc[bullish_trend_cond], current_bullish_scores)
        # 多头趋势加速 (mom_diff > 0)
        bullish_accelerate = bullish_trend_cond & (mom_diff_filled > 1e-6)
        score.loc[bullish_accelerate] += 2.5
        if q_diff_pos_50 > 0:
            score.loc[bullish_accelerate & (mom_diff_filled >= q_diff_pos_50)] += 2.5
        if q_diff_pos_75 > 0:
            score.loc[bullish_accelerate & (mom_diff_filled >= q_diff_pos_75)] += 5.0
        # 多头趋势减速 (mom_diff < 0)
        bullish_decelerate = bullish_trend_cond & (mom_diff_filled < -1e-6)
        score.loc[bullish_decelerate] -= 2.5
        if q_diff_neg_50 < 0:
             score.loc[bullish_decelerate & (mom_diff_filled <= q_diff_neg_50)] -= 2.5
    # 2.2 空头趋势 (mom_s < 0 且非死叉点)
    bearish_trend_cond = not_cross_cond & (mom_s < -1e-6)
    if bearish_trend_cond.any():
        # 动态构建 xp 和 fp，包含 q_mom_neg_75
        # 注意: xp_raw 必须是单调递增的，所以从最负到0
        xp_raw = [min_mom_neg, q_mom_neg_25, q_mom_neg_50, q_mom_neg_75, 0]
        fp_raw = [15.0, 20.0, 30.0, 40.0, 50.0] # 对应的分数
        xp = []
        fp = []
        # 始终添加第一个点
        xp.append(xp_raw[0])
        fp.append(fp_raw[0])
        # 遍历其余点，只在 x 严格递增时添加，如果 x 相同则取更高的 y
        for i in range(1, len(xp_raw)):
            if xp_raw[i] > xp[-1]:
                xp.append(xp_raw[i])
                fp.append(fp_raw[i])
            elif xp_raw[i] == xp[-1]: # 如果 x 相同，取更高的分数 (更不看跌)
                fp[-1] = max(fp[-1], fp_raw[i])
        # 确保至少有两个点用于插值，否则使用简化版
        if len(xp) < 2:
            if min_mom_neg < 0:
                xp = [min_mom_neg, 0]
                fp = [25.0, 50.0] # 简化范围
            else: # 所有负MOM值都为0或接近0
                xp = [-1e-6, 0] # 小范围以保持分数在50
                fp = [50.0, 50.0]
        current_bearish_scores = np.interp(mom_s[bearish_trend_cond], xp, fp)
        score.loc[bearish_trend_cond] = np.minimum(score.loc[bearish_trend_cond], current_bearish_scores)
        # 空头趋势加速 (mom_diff < 0, 即更负)
        bearish_accelerate = bearish_trend_cond & (mom_diff_filled < -1e-6)
        score.loc[bearish_accelerate] -= 2.5
        if q_diff_neg_50 < 0:
            score.loc[bearish_accelerate & (mom_diff_filled <= q_diff_neg_50)] -= 2.5
        if q_diff_neg_25 < 0:
            score.loc[bearish_accelerate & (mom_diff_filled <= q_diff_neg_25)] -= 5.0
        # 空头趋势减速 (mom_diff > 0, 即负值减小，向0靠近)
        bearish_decelerate = bearish_trend_cond & (mom_diff_filled > 1e-6)
        score.loc[bearish_decelerate] += 2.5
        if q_diff_pos_50 > 0:
            score.loc[bearish_decelerate & (mom_diff_filled >= q_diff_pos_50)] += 2.5
    # --- 3. 极端MOM值调整 ---
    # 极强MOM (例如大于90分位数)，分数推向更高
    if q_mom_pos_90 > 0 and not mom_s_positive.empty:
        extreme_bullish_cond = mom_s >= q_mom_pos_90
        score.loc[extreme_bullish_cond] = np.maximum(score.loc[extreme_bullish_cond], 85.0)
        if max_mom_pos > q_mom_pos_90 :
            top_1_percentile_val = mom_s_positive.quantile(0.99) if not mom_s_positive.empty else max_mom_pos
            if top_1_percentile_val > q_mom_pos_90 :
                score.loc[mom_s >= top_1_percentile_val] = np.maximum(score.loc[mom_s >= top_1_percentile_val], 95.0)
    # 极弱MOM (例如小于10分位数)，分数推向更低
    # 使用已定义的 q_mom_neg_10
    if q_mom_neg_10 < 0 and not mom_s_negative.empty:
        extreme_bearish_cond = mom_s <= q_mom_neg_10
        score.loc[extreme_bearish_cond] = np.minimum(score.loc[extreme_bearish_cond], 15.0)
        # 使用已定义的 q_mom_neg_10
        if min_mom_neg < q_mom_neg_10:
            bottom_1_percentile_val = mom_s_negative.quantile(0.01) if not mom_s_negative.empty else min_mom_neg
            # 使用已定义的 q_mom_neg_10
            if bottom_1_percentile_val < q_mom_neg_10:
                 score.loc[mom_s <= bottom_1_percentile_val] = np.minimum(score.loc[mom_s <= bottom_1_percentile_val], 5.0)
    # --- 最终裁剪确保分数在0-100之间 ---
    final_score = score.clip(0, 100)
    return final_score

def calculate_willr_score(willr: pd.Series) -> pd.Series:
    """
    深化WILLR (%R) 评分 (0-100)，使其更具层次感和洞察力。
    WILLR本身是反向指标：越低越超卖（看涨），越高越超买（看跌）。
    评分是正向的：越高越看涨。
    """
    # 使用 _safe_fillna_series 填充，WILLR的中性值通常认为是-50
    willr_s, = _safe_fillna_series([willr], [-50.0])
    # 如果所有WILLR值都是NaN（填充后为-50），则返回中性分50
    if willr_s.isnull().all() or (willr_s == -50.0).all():
        return pd.Series(50.0, index=willr.index).clip(0, 100)
    # 初始化评分为50（中性）
    score = pd.Series(50.0, index=willr_s.index)
    # 定义更细致的WILLR阈值
    # 极端超卖区 (分数: 95-100)
    ext_os_hard = -98.0  # 极度超卖，接近-100，强烈看涨信号
    ext_os_soft = -90.0  # 显著超卖
    # 超卖区 (分数: 80-95)
    os_th = -80.0        # 标准超卖阈值
    # 中性偏多区 (分数: 60-80)
    neutral_bull_lower = -70.0 # 中性区，但偏向超卖
    neutral_bull_upper = -55.0 # 中性区，略微偏向超卖
    # 中性偏空区 (分数: 20-40)
    neutral_bear_lower = -45.0 # 中性区，略微偏向超买
    neutral_bear_upper = -30.0 # 中性区，但偏向超买
    # 超买区 (分数: 5-20)
    ob_th = -20.0        # 标准超买阈值
    # 极端超买区 (分数: 0-5)
    ext_ob_soft = -10.0  # 显著超买
    ext_ob_hard = -2.0   # 极度超买，接近0，强烈看跌信号
    # 1. 基础区域评分 (静态评分，基于WILLR当前值)
    # 规则应用顺序：从最极端到最普遍，确保极端情况优先匹配
    # 极端超卖
    score.loc[willr_s <= ext_os_hard] = 100.0
    score.loc[(willr_s > ext_os_hard) & (willr_s <= ext_os_soft)] = 95.0
    # 超卖
    score.loc[(willr_s > ext_os_soft) & (willr_s <= os_th)] = 85.0
    # 中性偏多
    score.loc[(willr_s > os_th) & (willr_s <= neutral_bull_lower)] = 75.0
    score.loc[(willr_s > neutral_bull_lower) & (willr_s <= neutral_bull_upper)] = 65.0
    # 极端超买
    score.loc[willr_s >= ext_ob_hard] = 0.0
    score.loc[(willr_s < ext_ob_hard) & (willr_s >= ext_ob_soft)] = 5.0
    # 超买
    score.loc[(willr_s < ext_ob_soft) & (willr_s >= ob_th)] = 15.0
    # 中性偏空
    score.loc[(willr_s < ob_th) & (willr_s >= neutral_bear_upper)] = 25.0
    score.loc[(willr_s < neutral_bear_upper) & (willr_s >= neutral_bear_lower)] = 35.0
    # 对于落在 (-55, -45) 之间的，保持初始的50分，或根据趋势调整
    # 2. 动态信号调整 (基于WILLR的变化和持续性)
    willr_prev = willr_s.shift(1)
    # 确保willr_prev的第一个值为有效值，以避免在比较时产生全NaN的Series
    if pd.notna(willr_s.iloc[0]) and pd.isna(willr_prev.iloc[0]):
        willr_prev.iloc[0] = willr_s.iloc[0]
    # 2.1 持续性信号 (在超买/超卖区停留)
    # 持续在深度超卖区 (如连续2期 <= -90)
    stay_deep_os = (willr_s <= ext_os_soft) & (willr_prev <= ext_os_soft) & willr_prev.notna()
    score.loc[stay_deep_os] = np.maximum(score.loc[stay_deep_os], 98.0)
    # 持续在超卖区 (如连续2期 <= -80 且不在深度超卖)
    stay_os = (willr_s <= os_th) & (willr_prev <= os_th) & (willr_s > ext_os_soft) & willr_prev.notna()
    score.loc[stay_os] = np.maximum(score.loc[stay_os], 90.0)
    # 持续在深度超买区 (如连续2期 >= -10)
    stay_deep_ob = (willr_s >= ext_ob_soft) & (willr_prev >= ext_ob_soft) & willr_prev.notna()
    score.loc[stay_deep_ob] = np.minimum(score.loc[stay_deep_ob], 2.0)
    # 持续在超买区 (如连续2期 >= -20 且不在深度超买)
    stay_ob = (willr_s >= ob_th) & (willr_prev >= ob_th) & (willr_s < ext_ob_soft) & willr_prev.notna()
    score.loc[stay_ob] = np.minimum(score.loc[stay_ob], 10.0)
    # 2.2 交叉信号 (上穿/下穿关键阈值)
    # 上穿超卖线 os_th (-80)，产生买入倾向信号
    buy_signal_os = (willr_prev < os_th) & (willr_s >= os_th) & willr_prev.notna()
    # 如果刚脱离超卖区，但仍在偏多区域
    score.loc[buy_signal_os & (willr_s <= neutral_bull_lower)] = np.maximum(score.loc[buy_signal_os & (willr_s <= neutral_bull_lower)], 80.0)
    score.loc[buy_signal_os & (willr_s > neutral_bull_lower)] = np.maximum(score.loc[buy_signal_os & (willr_s > neutral_bull_lower)], 70.0)
    # 下穿超买线 ob_th (-20)，产生卖出倾向信号
    sell_signal_ob = (willr_prev > ob_th) & (willr_s <= ob_th) & willr_prev.notna()
    # 如果刚脱离超买区，但仍在偏空区域
    score.loc[sell_signal_ob & (willr_s >= neutral_bear_upper)] = np.minimum(score.loc[sell_signal_ob & (willr_s >= neutral_bear_upper)], 20.0)
    score.loc[sell_signal_ob & (willr_s < neutral_bear_upper)] = np.minimum(score.loc[sell_signal_ob & (willr_s < neutral_bear_upper)], 30.0)
    # 2.3 中性区域内的趋势 (-55 < WILLR < -45)
    # 定义核心中性区条件，且当天不是上述的交叉信号日
    core_neutral_cond = (willr_s > neutral_bull_upper) & (willr_s < neutral_bear_lower) & \
                        ~buy_signal_os & ~sell_signal_ob & willr_prev.notna()
    # 在核心中性区，WILLR上升 (趋向超买，看跌趋势增强)
    trend_to_ob_neutral = core_neutral_cond & (willr_s > willr_prev)
    score.loc[trend_to_ob_neutral] = np.minimum(score.loc[trend_to_ob_neutral], 45.0)
    # 在核心中性区，WILLR下降 (趋向超卖，看涨趋势增强)
    trend_to_os_neutral = core_neutral_cond & (willr_s < willr_prev)
    score.loc[trend_to_os_neutral] = np.maximum(score.loc[trend_to_os_neutral], 55.0)
    # 2.4 考虑WILLR变化率 (ROC) 对中性区评分的微调 # 解锁并深化2.4节
    roc = willr_s.diff() # 计算WILLR的变化率
    # 定义ROC的阈值
    roc_sharp_fall_th = -10.0  # WILLR急剧下降的阈值 (看涨) # 定义ROC急剧下降阈值
    roc_very_sharp_fall_th = -20.0 # WILLR非常急剧下降的阈值 (更看涨) # 定义ROC非常急剧下降阈值
    roc_sharp_rise_th = 10.0   # WILLR急剧上升的阈值 (看跌) # 定义ROC急剧上升阈值
    roc_very_sharp_rise_th = 20.0  # WILLR非常急剧上升的阈值 (更看跌) # 定义ROC非常急剧上升阈值
    # 识别急剧变化条件，并确保在核心中性区内
    # 急剧下降 (看涨信号增强)
    sharp_fall_cond = core_neutral_cond & (roc < roc_sharp_fall_th) & roc.notna() # 识别急剧下降条件
    very_sharp_fall_cond = core_neutral_cond & (roc < roc_very_sharp_fall_th) & roc.notna() # 识别非常急剧下降条件
    # 急剧上升 (看跌信号增强)
    sharp_rise_cond = core_neutral_cond & (roc > roc_sharp_rise_th) & roc.notna() # 识别急剧上升条件
    very_sharp_rise_cond = core_neutral_cond & (roc > roc_very_sharp_rise_th) & roc.notna() # 识别非常急剧上升条件
    # 应用评分调整
    # 极度急剧下降：大幅增加看涨倾向
    score.loc[very_sharp_fall_cond] = np.maximum(score.loc[very_sharp_fall_cond], 65.0) # 极度急剧下降，分数提升至65
    # 急剧下降：增加看涨倾向
    # 确保不覆盖更强的“极度急剧下降”信号
    score.loc[sharp_fall_cond & ~very_sharp_fall_cond] = np.maximum(score.loc[sharp_fall_cond & ~very_sharp_fall_cond], 60.0) # 急剧下降，分数提升至60
    # 极度急剧上升：大幅增加看跌倾向
    score.loc[very_sharp_rise_cond] = np.minimum(score.loc[very_sharp_rise_cond], 35.0) # 极度急剧上升，分数降低至35
    # 急剧上升：增加看跌倾向
    # 确保不覆盖更强的“极度急剧上升”信号
    score.loc[sharp_rise_cond & ~very_sharp_rise_cond] = np.minimum(score.loc[sharp_rise_cond & ~very_sharp_rise_cond], 40.0) # 急剧上升，分数降低至40
    # 确保分数在0-100之间
    return score.clip(0, 100)

def calculate_cmf_score(cmf: pd.Series) -> pd.Series:
    """
    CMF (Chaikin Money Flow) 评分 (0-100)。
    深化指标的评分规则，丰富计算规则，增加得分难度，使评分更具层次感，更具洞察力，并对代码做效率优化。
    """
    # 使用 _safe_fillna_series 填充 CMF 中的 NaN 值，将中性 CMF 设为 0。
    # MODIFIED: 确保 cmf_s 是一个 Series，并处理 _safe_fillna_series 的返回。
    cmf_s, = _safe_fillna_series([cmf], [0.0])
    # 如果填充后的 CMF Series 中所有值都为 NaN（例如，原始 Series 为空或全 NaN），则返回中性分数。
    # MODIFIED: 检查填充后的 Series 是否全为 NaN。
    if cmf_s.isnull().all():
        # MODIFIED: 返回一个与原始 CMF 索引相同的 Series，值为 50.0。
        return pd.Series(50.0, index=cmf.index).clip(0, 100)
    # --- 第一层：基于 CMF 绝对值的评分 ---
    # 定义 CMF 值区间和对应的基础分数。CMF 范围通常在 -1 到 1 之间。
    # 分为七个等级，提供更细致的初始评分。
    cmf_conditions = [
        cmf_s <= -0.5,                               # 极度看跌：资金大幅流出
        (cmf_s > -0.5) & (cmf_s <= -0.2),            # 看跌：资金持续流出
        (cmf_s > -0.2) & (cmf_s <= -0.05),           # 略微看跌：资金小幅流出
        (cmf_s >= -0.05) & (cmf_s < 0.05),           # 中性：资金流出入平衡
        (cmf_s >= 0.05) & (cmf_s < 0.2),             # 略微看涨：资金小幅流入
        (cmf_s >= 0.2) & (cmf_s < 0.5),              # 看涨：资金持续流入
        cmf_s >= 0.5                                 # 极度看涨：资金大幅流入
    ]
    # 对应的基础分数，范围 0-100。
    cmf_choices = [
        10,  # 极度看跌
        25,  # 看跌
        40,  # 略微看跌
        50,  # 中性
        60,  # 略微看涨
        75,  # 看涨
        90   # 极度看涨
    ]
    # 使用 numpy.select 进行矢量化赋值，效率高，避免多次 .loc 操作。
    # MODIFIED: 使用 np.select 计算基础分数。
    base_score = pd.Series(np.select(cmf_conditions, cmf_choices, default=50), index=cmf_s.index)
    # --- 第二层：基于 CMF 趋势（变化率）的调整 ---
    # 计算 CMF 的日变化量，反映资金流动的加速或减速。
    cmf_diff = cmf_s.diff()
    # 定义 CMF 变化量的条件和对应的分数调整。
    # 趋势调整分为五个等级，对基础分数进行动态修正。
    diff_conditions = [
        cmf_diff > 0.1,                              # 强劲上涨趋势：资金流入加速
        (cmf_diff > 0.02) & (cmf_diff <= 0.1),       # 温和上涨趋势：资金流入温和加速
        (cmf_diff >= -0.02) & (cmf_diff <= 0.02),    # 横盘/无明显趋势：资金流向稳定
        (cmf_diff < -0.02) & (cmf_diff > -0.1),      # 温和下跌趋势：资金流出温和加速
        cmf_diff <= -0.1                             # 强劲下跌趋势：资金流出加速
    ]
    # 对应的分数调整值。
    diff_choices = [
        5,   # 强劲上涨，加分
        2,   # 温和上涨，加分
        0,   # 无明显趋势，不调整
        -2,  # 温和下跌，减分
        -5   # 强劲下跌，减分
    ]
    # 使用 numpy.select 计算趋势调整值。
    # MODIFIED: 使用 np.select 计算趋势调整值。
    trend_adjustment = pd.Series(np.select(diff_conditions, diff_choices, default=0), index=cmf_s.index)
    # CMF.diff() 的第一个值会是 NaN，导致 trend_adjustment 的第一个值也是 NaN。
    # 填充 NaN 值为 0，确保计算的连续性。
    # MODIFIED: 填充 trend_adjustment 中的 NaN 值。
    trend_adjustment = trend_adjustment.fillna(0)
    # 将趋势调整值加到基础分数上，形成初步的最终分数。
    # MODIFIED: 将趋势调整值加到基础分数上。
    final_score = base_score + trend_adjustment
    # --- 第三层：极端值惩罚/奖励 ---
    # 对 CMF 接近极端值（1 或 -1）的情况给予额外的奖励或惩罚，
    # 以进一步突出其强度，使评分更具洞察力。
    # MODIFIED: 增加极端值调整逻辑。
    extreme_adjustment = pd.Series(0.0, index=cmf_s.index)
    # 如果 CMF 极度看涨（接近1），且分数未达到最高，给予额外奖励。
    extreme_adjustment.loc[cmf_s > 0.9] = 5
    # 如果 CMF 极度看跌（接近-1），且分数未达到最低，给予额外惩罚。
    extreme_adjustment.loc[cmf_s < -0.9] = -5
    # 将极端值调整加到最终分数上。
    final_score += extreme_adjustment
    # 确保最终分数在 0 到 100 之间，防止越界。
    # MODIFIED: 确保最终分数在 0 到 100 之间。
    return final_score.clip(0, 100)

def calculate_obv_score(obv: pd.Series, obv_ma: pd.Series = None, obv_ma_period: int = None) -> pd.Series:
    """
    OBV (On Balance Volume) 评分 (0-100)。
    深化指标的评分规则，丰富计算规则，增加得分难度，使评分更具层次感，更具洞察力，并对代码做效率优化。
    参数:
        obv (pd.Series): OBV 指标序列。
        obv_ma (pd.Series, optional): OBV 的移动平均线序列。如果提供，则优先使用。默认为 None。
        obv_ma_period (int, optional): 如果未提供 obv_ma，则用于计算 OBV 简单移动平均线 (SMA) 的周期。
                                       必须大于 1 才能计算有意义的均线。默认为 None。
    返回:
        pd.Series: OBV 评分序列，范围在 0 到 100 之间。
    """
    # 使用 _safe_fillna_series 填充 OBV 序列。
    # 确保 obv_s 经过 ffill/bfill 后，再用均值填充剩余的 NaN。
    obv_s, = _safe_fillna_series([obv], [None])
    # 如果填充后 OBV 序列仍全为 NaN，则无法进行有效计算，直接返回 50 分。
    if obv_s.isnull().all():
        return pd.Series(50.0, index=obv.index).clip(0,100)
    # 如果填充后仍有 NaN（例如，序列开头或全部是 NaN），用均值填充。
    obv_s = obv_s.fillna(obv_s.mean())
    # 再次检查是否全为 NaN，以防均值也为 NaN（例如，序列为空或均值计算失败）。
    if obv_s.isnull().all():
        return pd.Series(50.0, index=obv.index).clip(0,100)
    # 初始化评分序列，所有点位默认为 50 分。
    score = pd.Series(50.0, index=obv_s.index, dtype=float)
    # 计算 OBV 的日变化量和其平均绝对值，用于后续的动量和强度归一化。
    # 计算 OBV 的日变化量 (OBV_t - OBV_t-1)。
    obv_diff = obv_s.diff()
    # 计算 OBV 日变化量的平均绝对值，用于归一化。
    obv_roc_norm_factor = obv_diff.abs().mean()
    # 如果归一化因子为零，将其替换为 1，防止除以零错误。
    if obv_roc_norm_factor == 0:
        obv_roc_norm_factor = 1.0
    # 处理 OBV 均线 (OBV_MA)
    obv_ma_s = None
    # 优先使用传入的 obv_ma series。
    if obv_ma is not None and not obv_ma.isnull().all():
        # 填充 OBV_MA，确保索引一致。
        obv_ma_s, = _safe_fillna_series([obv_ma], [lambda s: s.mean()])
        # 如果填充后 OBV_MA 仍全为 NaN，则视为无效，尝试根据 obv_ma_period 计算。
        if obv_ma_s.isnull().all():
            obv_ma_s = None
            logger.warning("OBV 评分：传入的 OBV_MA 序列在填充后仍全为NaN，尝试根据 obv_ma_period 计算。")
    # 如果没有有效的 obv_ma series，但提供了 obv_ma_period 且周期大于 1，则计算 SMA。
    if obv_ma_s is None and obv_ma_period is not None and obv_ma_period > 1:
        # 使用 pandas_ta 计算简单移动平均线 (SMA)。
        calculated_obv_ma = ta.sma(obv_s, length=obv_ma_period)
        # 填充计算出的均线。
        calculated_obv_ma, = _safe_fillna_series([calculated_obv_ma], [lambda s: s.mean()])
        # 如果计算出的均线有效，则使用它。
        if not calculated_obv_ma.isnull().all():
            obv_ma_s = calculated_obv_ma
        else:
            logger.warning(f"OBV 评分：根据 period={obv_ma_period} 计算的 OBV_MA 序列在填充后仍全为NaN，跳过 OBV_MA 相关评分逻辑。")
    # 确保 obv_ma_s 的索引与 obv_s 匹配，并对齐。
    if obv_ma_s is not None:
        obv_ma_s = obv_ma_s.reindex(obv_s.index)
        # 再次填充，以防 reindex 引入 NaN。
        obv_ma_s = obv_ma_s.fillna(obv_ma_s.mean())
        # 最终确认 obv_ma_s 是否有效。
        if obv_ma_s.isnull().all():
            obv_ma_s = None
    # 如果有有效的 OBV_MA series，进行交叉和位置判断。
    if obv_ma_s is not None:
        # 计算 OBV 与 MA 的差值及其平均绝对值，用于归一化。
        # 计算 OBV 与 MA 的差值。
        obv_ma_diff = obv_s - obv_ma_s
        # 计算 OBV 与 MA 差值的平均绝对值，用于归一化。
        obv_ma_diff_norm_factor = obv_ma_diff.abs().mean()
        # 如果归一化因子为零，将其替换为 1，防止除以零错误。
        if obv_ma_diff_norm_factor == 0:
            obv_ma_diff_norm_factor = 1.0
        # 计算 OBV 与 MA 差值的归一化强度，表示偏离均线的程度。
        position_strength = obv_ma_diff.abs() / obv_ma_diff_norm_factor
        # 交叉判断：这是最强的信号，直接设置分数。
        # 检测买入交叉信号 (OBV 从 MA 下方穿过 MA 上方)。
        buy_cross = (obv_s.shift(1) < obv_ma_s.shift(1)) & (obv_s >= obv_ma_s)
        # 检测卖出交叉信号 (OBV 从 MA 上方穿过 MA 下方)。
        sell_cross = (obv_s.shift(1) > obv_ma_s.shift(1)) & (obv_s <= obv_ma_s)
        # 评分规则：交叉信号优先级最高，且根据交叉强度调整分数。
        # 对买入交叉进行评分，基础分 70，根据交叉强度额外加分，最高可达 95。
        score.loc[buy_cross] = 70 + np.clip(position_strength.loc[buy_cross] * 10, 0, 25)
        # 对卖出交叉进行评分，基础分 30，根据交叉强度额外减分，最低可达 5。
        score.loc[sell_cross] = 30 - np.clip(position_strength.loc[sell_cross] * 10, 0, 25)
        # 在未交叉时，根据 OBV 相对于 MA 的位置和强度调整分数。
        # 找出既非买入交叉也非卖出交叉的日期。
        not_cross_cond = ~buy_cross & ~sell_cross
        # OBV 在 MA 上方且未交叉的条件。
        above_ma_cond = (obv_s > obv_ma_s) & not_cross_cond
        # OBV 在 MA 下方且未交叉的条件。
        below_ma_cond = (obv_s < obv_ma_s) & not_cross_cond
        # 评分规则：OBV 在 MA 上方/下方，根据偏离强度调整分数。
        # 对 OBV 在 MA 上方的情况进行评分，基础分 60，根据位置强度额外加分，最高可达 75。
        score.loc[above_ma_cond] = 60 + np.clip(position_strength.loc[above_ma_cond] * 5, 0, 15)
        # 对 OBV 在 MA 下方的情况进行评分，基础分 40，根据位置强度额外减分，最低可达 25。
        score.loc[below_ma_cond] = 40 - np.clip(position_strength.loc[below_ma_cond] * 5, 0, 15)
    # 使用 OBV 本身的趋势和动量进行评分，仅作用于尚未被强信号（MA 交叉或位置）覆盖的日期。
    # 找出当前评分仍为 50 的日期。这些日期没有被 OBV_MA 交叉或位置信号覆盖。
    # 这样确保趋势评分只影响那些没有强 MA 信号的日期，避免覆盖更强的信号。
    neutral_score_mask_for_trend = (score == 50.0)
    # 计算动量强度：OBV 日变化量相对于平均日变化量的强度。
    # 计算 OBV 动量强度，用于趋势评分。
    momentum_strength = obv_diff.abs() / obv_roc_norm_factor
    # 评分规则：根据 OBV 自身的趋势和动量调整分数。
    # OBV 呈上涨趋势且处于中性评分的条件。
    bullish_trend_cond = (obv_diff > 0) & neutral_score_mask_for_trend
    # OBV 呈下跌趋势且处于中性评分的条件。
    bearish_trend_cond = (obv_diff < 0) & neutral_score_mask_for_trend
    # 对上涨趋势进行评分，使用 np.maximum 确保只提高分数，基础分 55，根据动量强度额外加分，最高可达 70。
    trend_score_bullish = 55 + np.clip(momentum_strength.loc[bullish_trend_cond] * 5, 0, 15)
    score.loc[bullish_trend_cond] = np.maximum(score.loc[bullish_trend_cond], trend_score_bullish)
    # 对下跌趋势进行评分，使用 np.minimum 确保只降低分数，基础分 45，根据动量强度额外减分，最低可达 30。
    trend_score_bearish = 45 - np.clip(momentum_strength.loc[bearish_trend_cond] * 5, 0, 15)
    score.loc[bearish_trend_cond] = np.minimum(score.loc[bearish_trend_cond], trend_score_bearish)
    # 最终将分数裁剪到 0-100 范围，确保分数有效性。
    # 确保最终分数在有效范围内。
    return score.clip(0, 100)

def calculate_kc_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """KC (Keltner Channel) 评分 (0-100)。
    深化评分规则，结合价格在通道内的相对位置和关键事件（如穿越）进行分层评分。
    """
    # 使用 _safe_fillna_series 填充输入序列中的 NaN 值
    # 确保 fallback 函数在 s 为 None 或全 NaN 时能返回一个合理的值
    # 调整了 fallback lambda 表达式，增加了对空 Series 的检查，使其更健壮
    close_s, upper_s, mid_s, lower_s = _safe_fillna_series(
        [close, upper, mid, lower],
        [
            None, # close 优先 ffill/bfill
            lambda s: s.mean() + 1.5 * (s.rolling(20, min_periods=1).max() - s.rolling(20, min_periods=1).min()).mean() if s is not None and not s.empty and s.mean() is not np.nan else (close.mean() + 1.5 * (close.rolling(20, min_periods=1).max() - close.rolling(20, min_periods=1).min()).mean() if close is not None and not close.empty and close.mean() is not np.nan else 50.0), # upper fallback
            lambda s: s.mean() if s is not None and not s.empty and s.mean() is not np.nan else (close.mean() if close is not None and not close.empty and close.mean() is not np.nan else 50.0), # mid fallback
            lambda s: s.mean() - 1.5 * (s.rolling(20, min_periods=1).max() - s.rolling(20, min_periods=1).min()).mean() if s is not None and not s.empty and s.mean() is not np.nan else (close.mean() - 1.5 * (close.rolling(20, min_periods=1).max() - close.rolling(20, min_periods=1).min()).mean() if close is not None and not close.empty and close.mean() is not np.nan else 50.0) # lower fallback
        ]
    )
    # 检查填充后是否仍存在全NaN的情况，这通常意味着输入数据本身就非常异常
    # 移除了调试信息
    if close_s.isnull().all() or upper_s.isnull().all() or mid_s.isnull().all() or lower_s.isnull().all():
         # logger.warning("KC 评分：一个或多个关键序列在填充后仍全为NaN。")
         # 确保返回 Series 的索引是原始 close 的索引
         return pd.Series(50.0, index=close.index).clip(0,100)
    # 计算 KC 通道宽度，用于后续归一化价格位置
    channel_width = upper_s - lower_s
    # 初始化归一化价格位置，默认值为0.5（通道中线）
    # 初始化 normalized_pos 为 0.5，处理通道宽度为零或负数的情况
    normalized_pos = pd.Series(0.5, index=close_s.index)
    # 仅在通道宽度有效（大于0）的情况下计算归一化价格位置
    # 优化行: 使用布尔索引进行向量化计算，避免循环和条件判断
    valid_channel = channel_width > 0
    normalized_pos[valid_channel] = (close_s[valid_channel] - lower_s[valid_channel]) / channel_width[valid_channel]
    # 计算基础分数：价格在通道内的线性映射 (0-100)
    # 价格越靠近下轨，分数越高；越靠近上轨，分数越低
    # 引入 base_score 作为连续评分的基础
    base_score = 100 - (normalized_pos * 100)
    # 定义各种条件，用于 np.select 的优先级判断
    # 定义价格低于或等于下轨的条件
    cond_below_lower = close_s <= lower_s
    # 定义价格高于或等于上轨的条件
    cond_above_upper = close_s >= upper_s
    # 定义价格从下轨下方反弹至下轨上方（买入支撑）的条件
    cond_buy_support = (close_s.shift(1) < lower_s.shift(1)) & (close_s >= lower_s)
    # 定义价格从上轨上方回落至上轨下方（卖出压力）的条件
    cond_sell_pressure = (close_s.shift(1) > upper_s.shift(1)) & (close_s <= upper_s)
    # 定义价格从中轨下方穿越至中轨上方（买入中轨交叉）的条件
    cond_buy_mid_cross = (close_s.shift(1) < mid_s.shift(1)) & (close_s >= mid_s)
    # 定义价格从中轨上方穿越至中轨下方（卖出中轨交叉）的条件
    cond_sell_mid_cross = (close_s.shift(1) > mid_s.shift(1)) & (close_s <= mid_s)
    # 定义价格在中轨附近小幅震荡的条件（避免过度敏感，给予中性分数）
    # 引入 cond_near_mid，用于识别中轨附近的震荡区域
    cond_near_mid = (close_s > mid_s * 0.99) & (close_s < mid_s * 1.01) # 1% 容忍度
    # 使用 np.select 按照优先级应用评分规则
    # 优先级从高到低：强反转信号 -> 极端位置 -> 中轨穿越 -> 中轨附近震荡 -> 默认连续评分
    # 优化行: 使用 np.select 替代多个 score.loc，提高效率和可读性
    conditions = [
        cond_buy_support,   # 价格从下轨下方反弹，强买入信号
        cond_sell_pressure, # 价格从上轨上方回落，强卖出信号
        cond_below_lower,   # 价格持续在下轨下方，极度超卖
        cond_above_upper,   # 价格持续在上轨上方，极度超买
        cond_buy_mid_cross, # 价格突破中轨向上，看涨
        cond_sell_mid_cross,# 价格跌破中轨向下，看跌
        cond_near_mid       # 价格在中轨附近，中性
    ]
    # 对应的分数，与 conditions 列表一一对应
    # 这些分数是基于业务逻辑和经验设定的，可以根据实际需求调整，以增加层次感
    choices = [
        95.0, # 强买入信号，高于一般超卖
        5.0,  # 强卖出信号，低于一般超买
        90.0, # 极度超卖，买入信号
        10.0, # 极度超买，卖出信号
        70.0, # 价格突破中轨向上，看涨，高于中性
        30.0, # 价格跌破中轨向下，看跌，低于中性
        50.0  # 价格在中轨附近，中性
    ]
    # 默认分数：对于不满足上述任何条件的点，使用基于归一化位置的连续分数
    score = np.select(conditions, choices, default=base_score)
    # 将 numpy 数组转换为 pandas Series，并保留原始索引
    score = pd.Series(score, index=close_s.index)
    # 最终裁剪分数到 0-100 范围，确保结果的有效性
    return score.clip(0, 100)

def calculate_hv_score(hv: pd.Series) -> pd.Series:
    """HV (Historical Volatility) 评分 (0-100)，分层更细致，评分更具洞察力。"""
    # 使用 _safe_fillna_series 填充
    hv_s, = _safe_fillna_series([hv], [lambda s: s.mean()])  # hv 填充后，如果全 NaN 使用均值
    if hv_s.isnull().all() or hv_s.mean() == 0:
        # print("全部为NaN或均值为0，返回50分")  # 调试信息
        return pd.Series(50.0, index=hv.index).clip(0, 100)
    score = pd.Series(50.0, index=hv_s.index)
    rolling_window = min(len(hv_s), 20)
    min_periods_rolling = max(1, int(rolling_window * 0.5)) if rolling_window > 0 else 1
    # 计算滚动均值和标准差
    hv_mean = hv_s.rolling(window=rolling_window, min_periods=min_periods_rolling).mean().fillna(hv_s.mean())
    hv_std = hv_s.rolling(window=rolling_window, min_periods=min_periods_rolling).std().fillna(hv_s.std()).fillna(0)
    # 计算分位数
    q20 = hv_s.quantile(0.2)
    q40 = hv_s.quantile(0.4)
    q60 = hv_s.quantile(0.6)
    q80 = hv_s.quantile(0.8)
    # 极端高波动（>均值+1.5*std，或>80分位）
    extreme_high = (hv_s > (hv_mean + 1.5 * hv_std)) | (hv_s > q80)
    # 较高波动（>均值+0.5*std，或在60-80分位之间）
    high = ((hv_s > (hv_mean + 0.5 * hv_std)) & (hv_s <= (hv_mean + 1.5 * hv_std))) | ((hv_s > q60) & (hv_s <= q80))
    # 中性波动（40-60分位）
    neutral = (hv_s > q40) & (hv_s <= q60)
    # 较低波动（<均值-0.5*std，或在20-40分位之间）
    low = ((hv_s < (hv_mean - 0.5 * hv_std)) & (hv_s >= (hv_mean - 1.5 * hv_std))) | ((hv_s > q20) & (hv_s <= q40))
    # 极端低波动（<均值-1.5*std，或<20分位）
    extreme_low = (hv_s < (hv_mean - 1.5 * hv_std)) | (hv_s <= q20)
    # 分层赋分
    score.loc[extreme_high] = 80.0
    score.loc[high] = 65.0
    score.loc[neutral] = 50.0
    score.loc[low] = 35.0
    score.loc[extreme_low] = 20.0
    # 进一步细化：对极端高/低波动再做微调
    # score.loc[hv_s > (hv_mean + 2 * hv_std)] = 90.0  # 调试信息
    # score.loc[hv_s < (hv_mean - 2 * hv_std)] = 10.0  # 调试信息
    score.loc[hv_s > (hv_mean + 2 * hv_std)] = 90.0
    score.loc[hv_s < (hv_mean - 2 * hv_std)] = 10.0
    # print("分数分布：", score.value_counts())  # 调试信息
    return score.clip(0, 100)

def calculate_vroc_score(vroc: pd.Series) -> pd.Series:
    """VROC (Volume Rate of Change) 评分 (0-100)，层次更丰富，洞察力更强。"""
    # 使用 _safe_fillna_series 填充
    vroc_s, = _safe_fillna_series([vroc], [0.0])  # VROC 中性0
    if vroc_s.isnull().all():
        return pd.Series(50.0, index=vroc.index).clip(0, 100)
    score = pd.Series(50.0, index=vroc_s.index)  # 初始分数50
    # 1. 识别多空交叉
    buy_cross = (vroc_s.shift(1) < 0) & (vroc_s >= 0)
    sell_cross = (vroc_s.shift(1) > 0) & (vroc_s <= 0)
    score.loc[buy_cross] = 55.0  # 放量转正，略偏多
    score.loc[sell_cross] = 45.0  # 缩量转负，略偏空
    # 2. 趋势判断（非交叉时）
    not_cross_cond = ~buy_cross & ~sell_cross
    bullish_trend = (vroc_s > 0) & (vroc_s > vroc_s.shift(1)) & not_cross_cond
    bearish_trend = (vroc_s < 0) & (vroc_s < vroc_s.shift(1)) & not_cross_cond
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 52.0)  # 温和偏多
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 48.0)  # 温和偏空
    # 3. 极端放量/缩量
    extreme_bull = (vroc_s >= 20)
    extreme_bear = (vroc_s <= -20)
    score.loc[extreme_bull] = 65.0  # 极端放量，强烈偏多
    score.loc[extreme_bear] = 35.0  # 极端缩量，强烈偏空
    # 4. 连续放量/缩量（3日及以上）
    rolling_bull = (vroc_s.rolling(3).min() > 0)
    rolling_bear = (vroc_s.rolling(3).max() < 0)
    score.loc[rolling_bull] = np.maximum(score.loc[rolling_bull], 60.0)  # 连续放量，进一步偏多
    score.loc[rolling_bear] = np.minimum(score.loc[rolling_bear], 40.0)  # 连续缩量，进一步偏空
    # 5. VROC接近0，分数更中性
    near_zero = (vroc_s.abs() < 2)
    score.loc[near_zero] = 50.0  # 极度中性
    # 6. VROC波动剧烈（绝对值大于30），分数大幅调整
    very_extreme_bull = (vroc_s >= 30)
    very_extreme_bear = (vroc_s <= -30)
    score.loc[very_extreme_bull] = 75.0  # 极端放量，极度偏多
    score.loc[very_extreme_bear] = 25.0  # 极端缩量，极度偏空
    # 7. 轻微放量/缩量（2~5），分数微调
    mild_bull = (vroc_s >= 2) & (vroc_s < 5)
    mild_bear = (vroc_s <= -2) & (vroc_s > -5)
    score.loc[mild_bull] = np.maximum(score.loc[mild_bull], 53.0)
    score.loc[mild_bear] = np.minimum(score.loc[mild_bear], 47.0)
    # 8. 中等放量/缩量（5~20），分数进一步调整
    moderate_bull = (vroc_s >= 5) & (vroc_s < 20)
    moderate_bear = (vroc_s <= -5) & (vroc_s > -20)
    score.loc[moderate_bull] = np.maximum(score.loc[moderate_bull], 58.0)
    score.loc[moderate_bear] = np.minimum(score.loc[moderate_bear], 42.0)
    # print("VROC评分调试信息：", score.describe())  # 调试信息，实际部署时注释掉
    return score.clip(0, 100)

def calculate_aroc_score(aroc: pd.Series) -> pd.Series:
    """AROC (Absolute Rate of Change) 评分 (0-100)，层次更丰富，洞察力更强。"""
    # 使用 _safe_fillna_series 填充缺失值
    aroc_s, = _safe_fillna_series([aroc], [0.0])  # AROC 中性0
    # 全部为空，返回中性分
    if aroc_s.isnull().all():
        return pd.Series(50.0, index=aroc.index).clip(0, 100)
    # 初始化分数为中性
    score = pd.Series(50.0, index=aroc_s.index)
    # 计算穿越0线信号
    buy_cross = (aroc_s.shift(1) < 0) & (aroc_s >= 0)
    sell_cross = (aroc_s.shift(1) > 0) & (aroc_s <= 0)
    # 极端强势/弱势区间
    strong_bull = aroc_s >= 0.05
    strong_bear = aroc_s <= -0.05
    # 大幅度但未穿越0线
    mid_bull = (aroc_s >= 0.025) & (aroc_s < 0.05)
    mid_bear = (aroc_s <= -0.025) & (aroc_s > -0.05)
    # 震荡区间
    neutral_zone = (aroc_s.abs() < 0.005) & (aroc_s.diff().abs() < 0.002)
    # 趋势增强/减弱
    bullish_trend = (aroc_s > 0) & (aroc_s > aroc_s.shift(1)) & (~buy_cross) & (~strong_bull)
    bearish_trend = (aroc_s < 0) & (aroc_s < aroc_s.shift(1)) & (~sell_cross) & (~strong_bear)
    # 评分赋值，分层更细致
    score.loc[strong_bull] = 85.0  # 极端强势
    score.loc[strong_bear] = 15.0  # 极端弱势
    score.loc[mid_bull] = 70.0     # 明显强势
    score.loc[mid_bear] = 30.0     # 明显弱势
    score.loc[buy_cross] = 65.0    # 0轴上穿
    score.loc[sell_cross] = 35.0   # 0轴下穿
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 58.0)  # 趋势增强
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 42.0)  # 趋势减弱
    score.loc[neutral_zone] = 50.0  # 震荡区间
    # 进一步细化：连续多期强势/弱势，分数递增/递减
    # 统计连续正/负区间长度
    pos_run = (aroc_s > 0).astype(int).groupby((aroc_s <= 0).astype(int).cumsum()).cumsum()
    neg_run = (aroc_s < 0).astype(int).groupby((aroc_s >= 0).astype(int).cumsum()).cumsum()
    # 连续3期以上强势，分数上浮
    score += (pos_run >= 3) * 3
    # 连续3期以上弱势，分数下调
    score -= (neg_run >= 3) * 3
    # 分数限制在0-100
    score = score.clip(0, 100)
    # # 调试信息
    # print("aroc_s:", aroc_s.tail())
    # print("score:", score.tail())
    # print("strong_bull:", strong_bull.tail())
    # print("strong_bear:", strong_bear.tail())
    # print("pos_run:", pos_run.tail())
    # print("neg_run:", neg_run.tail())
    return score

def calculate_pivot_score(close: pd.Series, pivot_levels: Dict[str, pd.Series], tf: str, params: Optional[Dict] = None) -> pd.Series:
    """
    更具层次感和洞察力的 Pivot Points 评分 (0-100)。
    """
    score = pd.Series(50.0, index=close.index)
    close_filled = _safe_fillna_series(close)
    if close_filled.isnull().all():
        # print("Pivot Points 评分：收盘价序列在填充后仍全为 NaN。")
        return score.clip(0, 100)
    pp_series = pivot_levels.get('PP')
    r_series = {k: v for k, v in pivot_levels.items() if k.startswith('R') and k != 'R'}
    s_series = {k: v for k, v in pivot_levels.items() if k.startswith('S') and k != 'S'}
    fr_series = {k: v for k, v in pivot_levels.items() if k.startswith('F_R')}
    fs_series = {k: v for k, v in pivot_levels.items() if k.startswith('F_S')}
    # 1. 价格与 PP 的距离分级评分
    if pp_series is not None and pp_series.notna().any():
        pp_series_aligned = pp_series.reindex(close.index)
        valid_pp_mask = pp_series_aligned.notna()
        dist = (close_filled - pp_series_aligned).abs()
        # 以PP为中心，距离越远分数越极端，距离越近分数越中性
        # 0.5%以内变化极小，1%以内变化小，1-2%变化中，2%以上变化大
        pct_dist = dist / pp_series_aligned
        score.loc[valid_pp_mask & (close_filled > pp_series_aligned)] += (pct_dist * 40).clip(0, 20)
        score.loc[valid_pp_mask & (close_filled < pp_series_aligned)] -= (pct_dist * 40).clip(0, 20)
        # PP附近收敛
        score.loc[valid_pp_mask & (pct_dist < 0.005)] = 50
    # 2. 多级别支撑/阻力突破/跌破分层评分
    level_data = {
        'R': {'series': r_series, 'base_score_breakout': 65, 'base_score_breakdown': None, 'level_multiplier': 6, 'is_resistance': True, 'prefix': 'R', 'fib': False},
        'S': {'series': s_series, 'base_score_breakout': None, 'base_score_breakdown': 35, 'level_multiplier': 6, 'is_resistance': False, 'prefix': 'S', 'fib': False},
        'F_R': {'series': fr_series, 'base_score_breakout': 75, 'base_score_breakdown': None, 'level_multiplier': 8, 'is_resistance': True, 'prefix': 'F_R', 'fib': True},
        'F_S': {'series': fs_series, 'base_score_breakout': None, 'base_score_breakdown': 25, 'level_multiplier': 8, 'is_resistance': False, 'prefix': 'F_S', 'fib': True},
    }
    close_prev = close_filled.shift(1)
    for type_key, config in level_data.items():
        for level_key, level_series in config['series'].items():
            try:
                level_num_str = level_key.replace(config['prefix'], '')
                level_num = int(level_num_str)
            except (ValueError, TypeError):
                # print(f"Pivot Points 评分：无法从内部 key '{level_key}' 解析级别数字。跳过该级别。")
                continue
            if level_series is not None and level_series.notna().any():
                level_series_aligned = level_series.reindex(close.index)
                valid_level_mask = level_series_aligned.notna()
                level_prev = level_series_aligned.shift(1)
                # 斐波那契级别权重更高
                multiplier = config['level_multiplier'] * (1.2 if config['fib'] else 1.0)
                # 突破/跌破分数分层
                if config['is_resistance']:
                    breakout_cond = (close_prev < level_prev) & (close_filled >= level_series_aligned) & valid_level_mask
                    if breakout_cond.any():
                        breakout_score = config['base_score_breakout'] + level_num * multiplier
                        # 连续突破多级阻力，累计加分
                        score.loc[breakout_cond] += (breakout_score - 50) * (1 + 0.1 * (level_num - 1))
                    below_resistance_cond = (close_filled < level_series_aligned) & valid_level_mask
                    if below_resistance_cond.any():
                        penalty = level_num * 3 * (1.2 if config['fib'] else 1.0)
                        # 越接近阻力，分数越低，距离越近惩罚越大
                        dist = (level_series_aligned - close_filled).clip(lower=0)
                        pct_dist = dist / level_series_aligned
                        penalty_adj = penalty * (1 + (1 - pct_dist.clip(upper=1)))
                        score.loc[below_resistance_cond] -= penalty_adj.clip(0, 15)
                else:
                    breakdown_cond = (close_prev > level_prev) & (close_filled <= level_series_aligned) & valid_level_mask
                    if breakdown_cond.any():
                        breakdown_score = config['base_score_breakdown'] - level_num * multiplier
                        # 连续跌破多级支撑，累计扣分
                        score.loc[breakdown_cond] -= (50 - breakdown_score) * (1 + 0.1 * (level_num - 1))
                    above_support_cond = (close_filled > level_series_aligned) & valid_level_mask
                    if above_support_cond.any():
                        bonus = level_num * 3 * (1.2 if config['fib'] else 1.0)
                        dist = (close_filled - level_series_aligned).clip(lower=0)
                        pct_dist = dist / level_series_aligned
                        bonus_adj = bonus * (1 + (1 - pct_dist.clip(upper=1)))
                        score.loc[above_support_cond] += bonus_adj.clip(0, 15)
    # 3. 多重区间评分：价格在多重支撑/阻力区间内，分数更细腻
    # 统计价格位于多少个阻力/支撑区间内，分数微调
    all_levels = []
    for d in [r_series, s_series, fr_series, fs_series]:
        all_levels.extend([v.reindex(close.index) for v in d.values() if v is not None])
    if all_levels:
        stacked = pd.concat(all_levels, axis=1)
        # 统计价格在多少个区间内
        in_zone = ((close_filled.values[:, None] > stacked.min(axis=1).values[:, None]) &
                   (close_filled.values[:, None] < stacked.max(axis=1).values[:, None]))
        zone_count = in_zone.sum(axis=1)
        # 区间越多，分数越趋向中性
        score -= (zone_count * 2).clip(0, 10)
    # 4. 分数收敛与极端限制
    score = score.clip(0, 100)
    # print(f"Pivot Points 评分调试：最大分{score.max()}，最小分{score.min()}，均值{score.mean()}")
    return score

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
    param_str_parts = [format_param(p, indicator_key) for p in params] # Pass indicator_key to format_param
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
        # Corrected BOLL's prefix_map, added 'close'
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
        # Corrected SAR's prefix_map, added 'close'
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
        # Corrected EMA/SMA's prefix_map, added 'close'
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
             return f"ADL_{tf_suffix}" # Corrected ADL column name pattern to 'ADL_tf'
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
                 return f"VWAP_{anchor}_{tf_suffix}" # Corrected VWAP column name pattern with anchor
            else:
                 # If params is empty, it means no anchor parameter
                 # Build column name: 'VWAP' + '_' + Timeframe Suffix
                 # Example 'VWAP_30'
                 return f"VWAP_{tf_suffix}" # Corrected VWAP column name pattern without anchor
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
        # Corrected KC's prefix_map, added 'close'
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

def find_actual_col_name(base_name, suffixes, data_columns):
    """
    根据基础名和后缀列表，返回实际存在于data.columns的列名。
    优先带后缀，找不到再尝试不带后缀。
    """
    for suffix in suffixes:
        col = f"{base_name}_{suffix}".replace('__', '_').strip('_')
        if col in data_columns:
            return col
    if base_name in data_columns:
        return base_name
    return None

# adjust_score_with_volume.获取实际列名:
# close_col: close_15, high_col: high_15, low_col: low_15, volume_col: volume_15, cmf_col_name: CMF_20_15
# obv_col_name: OBV_15, obv_ma_col_name: None

def adjust_score_with_volume(
    preliminary_score: pd.Series,
    data: pd.DataFrame,
    vc_params: dict,
    vc_tf_list: Optional[List[str]] = None,
    vol_ma_period: Optional[int] = None,
    obv_ma_period: Optional[int] = None,
    naming_config: Optional[dict] = None
) -> pd.DataFrame:
    """
    量能分层评分，7层分级，分数调整分层递进，复杂共振权重加权。
    保持原始参数签名和信号输出格式。
    """
    import numpy as np
    import pandas as pd
    def get_resonance_weight(c, s, d):
        arr = np.array([c, s, d])
        signs = np.sign(arr)
        abs_arr = np.abs(arr)
        # 三信号极端共振
        if (np.all(signs == signs[0]) and signs[0] != 0 and np.all(abs_arr == 3)):
            return 2.0
        # 三信号强共振
        if (np.all(signs == signs[0]) and signs[0] != 0 and np.all(abs_arr >= 2)):
            return 1.6
        # 三信号中等共振
        if (np.all(signs == signs[0]) and signs[0] != 0 and np.all(abs_arr >= 1)):
            return 1.3
        # 两信号极端共振
        for i in range(3):
            idx = [j for j in range(3) if j != i]
            if (signs[idx[0]] == signs[idx[1]] != 0 and abs_arr[idx[0]] == 3 and abs_arr[idx[1]] == 3):
                return 1.5
        # 两信号强共振
        for i in range(3):
            idx = [j for j in range(3) if j != i]
            if (signs[idx[0]] == signs[idx[1]] != 0 and abs_arr[idx[0]] >= 2 and abs_arr[idx[1]] >= 2):
                return 1.25
        # 两信号中等共振
        for i in range(3):
            idx = [j for j in range(3) if j != i]
            if (signs[idx[0]] == signs[idx[1]] != 0 and abs_arr[idx[0]] >= 1 and abs_arr[idx[1]] >= 1):
                return 1.1
        # 三信号方向完全相反
        if (np.sum(signs) == 0 and np.count_nonzero(signs) == 3):
            return 0.6
        # 两信号方向相反且均为强
        for i in range(3):
            idx = [j for j in range(3) if j != i]
            if (signs[idx[0]] == -signs[idx[1]] and abs_arr[idx[0]] >= 2 and abs_arr[idx[1]] >= 2):
                return 0.8
        # 一正一负一中性
        if (np.count_nonzero(signs == 0) == 1 and np.count_nonzero(signs == 1) == 1 and np.count_nonzero(signs == -1) == 1):
            return 0.9
        # 其余情况
        return 1.0
    # --- 0. 初始化 ---
    result_df = pd.DataFrame(index=preliminary_score.index)
    result_df['ADJUSTED_SCORE'] = preliminary_score.copy().fillna(50.0)
    current_naming_config = naming_config if naming_config is not None else {}
    ohlcv_base_names = ['open', 'high', 'low', 'close', 'volume']
    # 时间框架处理
    vol_tf_list_to_process = vc_tf_list
    if vol_tf_list_to_process is None or not isinstance(vol_tf_list_to_process, list) or not vol_tf_list_to_process:
        tf_from_params = vc_params.get('timeframes', 'D')
        if isinstance(tf_from_params, str):
            vol_tf_list_to_process = [tf_from_params]
        elif isinstance(tf_from_params, list) and tf_from_params:
            vol_tf_list_to_process = tf_from_params
        else:
            vol_tf_list_to_process = ['D']
    vol_tf_list_to_process = [str(tf) for tf in vol_tf_list_to_process]
    # 只处理第一个时间框架用于分数调整，其余仅输出信号
    first_tf_processed = False
    processed_tfs = []
    for current_vol_tf in vol_tf_list_to_process:
        tf_score_str = str(current_vol_tf)
        # 列名查找
        def find_col(base, suffix):
            for s in [suffix, '']:
                col = f"{base}_{s}" if s else base
                if col in data.columns:
                    return col
            return None
        # 适配你的命名方式
        close_col = find_col('close', tf_score_str)
        high_col = find_col('high', tf_score_str)
        low_col = find_col('low', tf_score_str)
        volume_col = find_col('volume', tf_score_str)
        cmf_col_name = find_col(f"CMF_{vc_params.get('cmf_period', 20)}", tf_score_str)
        obv_col_name = find_col("OBV", tf_score_str)
        # OBV_MA不用
        # 检查必需列
        if not all([close_col, high_col, low_col, volume_col, cmf_col_name, obv_col_name]):
            result_df[f'VOL_CONFIRM_SIGNAL_{current_vol_tf}'] = 0
            result_df[f'VOL_SPIKE_SIGNAL_{current_vol_tf}'] = 0
            result_df[f'VOL_PRICE_DIV_SIGNAL_{current_vol_tf}'] = 0
            continue
        # 数据提取
        close = data[close_col].reindex(result_df.index).ffill().bfill()
        high = data[high_col].reindex(result_df.index).ffill().bfill()
        low = data[low_col].reindex(result_df.index).ffill().bfill()
        volume = data[volume_col].reindex(result_df.index).fillna(0)
        cmf = data[cmf_col_name].reindex(result_df.index).fillna(0)
        obv = data[obv_col_name].reindex(result_df.index).ffill().bfill()
        # --- 1. 信号分层 ---
        # 1.1 量能确认信号（CMF分层）
        confirm_signal = pd.Series(0, index=result_df.index)
        confirm_signal[cmf >= 0.10] = 3
        confirm_signal[(cmf >= 0.07) & (cmf < 0.10)] = 2
        confirm_signal[(cmf >= 0.04) & (cmf < 0.07)] = 1
        confirm_signal[(cmf > -0.04) & (cmf < 0.04)] = 0
        confirm_signal[(cmf > -0.07) & (cmf <= -0.04)] = -1
        confirm_signal[(cmf > -0.10) & (cmf <= -0.07)] = -2
        confirm_signal[cmf <= -0.10] = -3
        result_df[f'VOL_CONFIRM_SIGNAL_{current_vol_tf}'] = confirm_signal
        # 1.2 成交量突增信号（分层）
        ma_period = vol_ma_period if vol_ma_period is not None else vc_params.get('volume_ma_period', 20)
        vol_ma = volume.rolling(window=ma_period, min_periods=1).mean()
        vol_ratio = volume / (vol_ma + 1e-9)
        spike_signal = pd.Series(0, index=volume.index)
        spike_signal[vol_ratio >= 7] = 3
        spike_signal[(vol_ratio >= 5) & (vol_ratio < 7)] = 2
        spike_signal[(vol_ratio >= 3) & (vol_ratio < 5)] = 1
        spike_signal[(vol_ratio >= 0.7) & (vol_ratio < 3)] = 0
        spike_signal[(vol_ratio >= 0.5) & (vol_ratio < 0.7)] = -1
        spike_signal[(vol_ratio >= 0.3) & (vol_ratio < 0.5)] = -2
        spike_signal[vol_ratio < 0.3] = -3
        result_df[f'VOL_SPIKE_SIGNAL_{current_vol_tf}'] = spike_signal
        # 1.3 量价背离信号（分层）
        lookback = vc_params.get('vp_divergence_lookback', 21)
        price_thresh = vc_params.get('vp_divergence_price_threshold', 0.005)
        obv_thresh = vc_params.get('vp_divergence_obv_threshold', 0.005)
        temp_df = pd.DataFrame({'high': high, 'low': low, 'obv': obv}, index=result_df.index)
        div_value = pd.Series(0, index=temp_df.index)
        for i in range(lookback - 1, len(temp_df)):
            window = temp_df.iloc[i - lookback + 1: i + 1]
            if window['high'][:-1].isnull().all() or window['low'][:-1].isnull().all() or window['obv'][:-1].isnull().all():
                continue
            p1_val = window['high'][:-1].max()
            p1_idx = window['high'][:-1].idxmax()
            t1_val = window['low'][:-1].min()
            t1_idx = window['low'][:-1].idxmin()
            i1_obv_at_p1 = window['obv'][:-1].loc[p1_idx] if p1_idx in window.index else np.nan
            i1_obv_at_t1 = window['obv'][:-1].loc[t1_idx] if t1_idx in window.index else np.nan
            p2_high = window['high'].iloc[-1]
            p2_low = window['low'].iloc[-1]
            i2_obv = window['obv'].iloc[-1]
            # 看跌背离
            if p2_high > p1_val * (1 + price_thresh):
                if not pd.isna(i1_obv_at_p1) and (i2_obv <= i1_obv_at_p1 or (abs(i1_obv_at_p1) > 1e-9 and (i2_obv - i1_obv_at_p1) / abs(i1_obv_at_p1) < -obv_thresh)):
                    div_value.iloc[i] = (i2_obv - i1_obv_at_p1) / (abs(i1_obv_at_p1) + 1e-9)
            # 看涨背离
            elif p2_low < t1_val * (1 - price_thresh):
                if not pd.isna(i1_obv_at_t1) and (i2_obv >= i1_obv_at_t1 or (abs(i1_obv_at_t1) > 1e-9 and (i2_obv - i1_obv_at_t1) / abs(i1_obv_at_t1) > obv_thresh)):
                    div_value.iloc[i] = (i2_obv - i1_obv_at_t1) / (abs(i1_obv_at_t1) + 1e-9)
            else:
                div_value.iloc[i] = 0
        div_signal = pd.Series(0, index=div_value.index)
        div_signal[div_value >= 0.20] = 3
        div_signal[(div_value >= 0.12) & (div_value < 0.20)] = 2
        div_signal[(div_value >= 0.07) & (div_value < 0.12)] = 1
        div_signal[(div_value > -0.07) & (div_value < 0.07)] = 0
        div_signal[(div_value > -0.12) & (div_value <= -0.07)] = -1
        div_signal[(div_value > -0.20) & (div_value <= -0.12)] = -2
        div_signal[div_value <= -0.20] = -3
        result_df[f'VOL_PRICE_DIV_SIGNAL_{current_vol_tf}'] = div_signal
        # --- 2. 分数调整（仅第一个TF） ---
        if not first_tf_processed:
            level_adj = {3:0.30, 2:0.20, 1:0.12, 0:0, -1:-0.12, -2:-0.20, -3:-0.30}
            for idx in result_df.index:
                base_score = result_df.at[idx, 'ADJUSTED_SCORE']
                c = confirm_signal.get(idx, 0)
                s = spike_signal.get(idx, 0)
                d = div_signal.get(idx, 0)
                adj = 0
                if c != 0:
                    if c > 0:
                        adj += (100 - base_score) * level_adj[c]
                    else:
                        adj -= (base_score - 50) * abs(level_adj[c])
                if s != 0:
                    if s > 0:
                        adj += (100 - base_score) * level_adj[s] * 0.7
                    else:
                        adj -= (base_score - 50) * abs(level_adj[s]) * 0.7
                if d != 0:
                    if d > 0:
                        adj += (100 - base_score) * level_adj[d] * 0.5
                    else:
                        adj -= (base_score - 50) * abs(level_adj[d]) * 0.5
                # 共振加权
                weight = get_resonance_weight(c, s, d)
                adj *= weight
                result_df.at[idx, 'ADJUSTED_SCORE'] = np.clip(base_score + adj, 0, 100)
            first_tf_processed = True
        processed_tfs.append(current_vol_tf)
    # --- 3. 信号列填充 ---
    for col in result_df.columns:
        if col.startswith('VOL_'):
            result_df[col] = result_df[col].fillna(0).astype(int)
    result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].fillna(50.0)
    return result_df

# adjust_score_with_volume中废弃的查询列名的代码，在 # 2. 数据提取, 对齐和填充 之前：
# for current_vol_tf in vol_tf_list_to_process:
#     # print(f"\n--- DEBUG PRINT: 处理量能时间框架: {current_vol_tf} ---")

#     timeframe_naming_conv = current_naming_config.get('timeframe_naming_convention', {})
#     tf_score_str = str(current_vol_tf)
#     possible_tf_suffixes_raw = timeframe_naming_conv.get('patterns', {}).get(tf_score_str.lower(), [tf_score_str])
#     if not isinstance(possible_tf_suffixes_raw, list): possible_tf_suffixes = [str(possible_tf_suffixes_raw)]
#     else: possible_tf_suffixes = [str(p) for p in possible_tf_suffixes_raw]
#     # 确保原始tf字符串（如'15', 'D'）优先被尝试作为后缀
#     if tf_score_str not in possible_tf_suffixes: possible_tf_suffixes.insert(0, tf_score_str)
#     seen = set(); unique_suffixes = []
#     for suffix in possible_tf_suffixes:
#         if suffix not in seen: seen.add(suffix); unique_suffixes.append(suffix)
#     possible_tf_suffixes = unique_suffixes if unique_suffixes else [tf_score_str]

#     # --- OHLCV 列名查找逻辑 ---
#     ohlcv_cols_found: Dict[str, str] = {}
#     ohlcv_base_names = ['open', 'high', 'low', 'close', 'volume']
#     ohlcv_naming_conv_from_config = current_naming_config.get('ohlcv_naming_convention', {})
#     ohlcv_config_entries = ohlcv_naming_conv_from_config.get('output_columns', [])

#     for base_name_key in ohlcv_base_names:
#         pattern_entry = next((p for p in ohlcv_config_entries if isinstance(p, dict) and p.get('internal_key') == base_name_key), None)
#         actual_ohlcv_base_pattern = base_name_key # 默认使用内部键名
#         if pattern_entry and pattern_entry.get('name_pattern'):
#             actual_ohlcv_base_pattern = pattern_entry['name_pattern']

#         found_ohlcv_col = None
#         for suffix in possible_tf_suffixes:
#             expected_ohlcv_col = f"{actual_ohlcv_base_pattern}_{suffix}".replace('__', '_').strip('_')
#             if expected_ohlcv_col in data.columns:
#                 found_ohlcv_col = expected_ohlcv_col
#                 break
#         if not found_ohlcv_col and actual_ohlcv_base_pattern in data.columns: # 尝试不带后缀
#             found_ohlcv_col = actual_ohlcv_base_pattern
        
#         if found_ohlcv_col:
#             ohlcv_cols_found[base_name_key] = found_ohlcv_col

#     # --- 指标 (CMF, OBV, OBV_MA) 列名查找逻辑 ---
#     indicator_cols_found: Dict[str, str] = {}
#     indicator_naming_root = current_naming_config.get('indicator_naming_conventions', {})
#     derivative_naming_root = current_naming_config.get('derivative_feature_naming_conventions', {})

#     # 定义如何查找每个所需指标的模式
#     # 'path_to_pattern' 是一个列表，表示在获取到 'key_in_source' 对应的字典后，如何进一步导航到模式字符串
#     # 例如 ['output_columns', 0, 'name_pattern'] 表示 dict[key_in_source]['output_columns'][0]['name_pattern']
#     indicators_config_map = {
#         'cmf': {'period': cmf_period_param, 'source_dict_obj': indicator_naming_root, 'key_in_source': 'CMF', 'path_to_pattern': ['output_columns', 0, 'name_pattern']},
#         'obv': {'period': None, 'source_dict_obj': indicator_naming_root, 'key_in_source': 'OBV', 'path_to_pattern': ['output_columns', 0, 'name_pattern']},
#         'obv_ma': {'period': current_obv_ma_period_param, 'source_dict_obj': derivative_naming_root, 'key_in_source': 'OBV_MA', 'path_to_pattern': ['output_column_pattern']}
#     }

#     for internal_key, config_details in indicators_config_map.items():
#         period_value = config_details['period']
#         source_dict = config_details['source_dict_obj']
#         key_in_config_source = config_details['key_in_source']
#         path_to_pattern_list = config_details['path_to_pattern']

#         raw_pattern_str_from_config = None
#         if source_dict and key_in_config_source in source_dict:
#             current_level_dict = source_dict[key_in_config_source]
#             is_valid_path = True
#             for path_segment in path_to_pattern_list:
#                 if isinstance(current_level_dict, dict) and path_segment in current_level_dict:
#                     current_level_dict = current_level_dict[path_segment]
#                 elif isinstance(current_level_dict, list) and isinstance(path_segment, int) and 0 <= path_segment < len(current_level_dict):
#                     current_level_dict = current_level_dict[path_segment]
#                 else:
#                     is_valid_path = False
#                     break
#             if is_valid_path and isinstance(current_level_dict, str):
#                 raw_pattern_str_from_config = current_level_dict
        
#         # 根据是否从配置中获取到模式，确定基础列名（可能包含参数占位符）
#         base_pattern_for_col_name = internal_key # 默认使用内部键名
#         if raw_pattern_str_from_config:
#             # logger.debug(f"指标: 使用来自配置的原始模式 '{raw_pattern_str_from_config}' (内部键: '{internal_key}')。")
#             base_pattern_for_col_name = raw_pattern_str_from_config # 例如 "CMF_{period}" 或 "OBV_MA_{period}" 或 "OBV"
#             # 替换参数占位符
#             if period_value is not None and '{period}' in base_pattern_for_col_name:
#                 base_pattern_for_col_name = base_pattern_for_col_name.replace('{period}', str(period_value))
#             # 此时 base_pattern_for_col_name 类似于 "CMF_20", "OBV_MA_10", "OBV"
#         else:
#             # logger.debug(f"指标: 未在 naming_config 中找到内部键 '{internal_key}' 的模式。将使用默认规则 '{internal_key}' 或 '{internal_key}_{{period}}'。")
#             # 默认构建: 如果有周期，则为 internal_key_period，否则为 internal_key
#             if period_value is not None:
#                 base_pattern_for_col_name = f"{internal_key}_{str(period_value)}" # 例如 "cmf_20", "obv_ma_10"
#             # else base_pattern_for_col_name 保持为 "obv"

#         # 尝试组合基础模式和时间后缀来查找列
#         found_indicator_col = None
#         for suffix in possible_tf_suffixes:
#             expected_indicator_col = f"{base_pattern_for_col_name}_{suffix}".replace('__', '_').strip('_')
#             if expected_indicator_col in data.columns:
#                 found_indicator_col = expected_indicator_col
#                 break
        
#         # 如果带后缀未找到，尝试不带后缀的（可能数据未按时间框架区分，或后缀已在模式中）
#         if not found_indicator_col and base_pattern_for_col_name in data.columns:
#             found_indicator_col = base_pattern_for_col_name
        
#         if found_indicator_col:
#             indicator_cols_found[internal_key] = found_indicator_col


#     # 检查必需列是否都已找到
#     required_keys_for_tf = ['close', 'high', 'low', 'volume', 'cmf', 'obv'] # OBV_MA 是可选的
#     all_required_found = True
#     missing_details = []
#     for key in required_keys_for_tf:
#         col_name_to_check = ohlcv_cols_found.get(key) if key in ohlcv_base_names else indicator_cols_found.get(key)
#         if col_name_to_check is None:
#             all_required_found = False; missing_details.append(f"列名未找到 (internal key: '{key}')")
#         elif col_name_to_check not in data.columns:
#             all_required_found = False; missing_details.append(f"列 '{col_name_to_check}' 不在数据中 (internal key: '{key}')")
#         elif data[col_name_to_check].isnull().all():
#             all_required_found = False; missing_details.append(f"列 '{col_name_to_check}' 数据全为 NaN (internal key: '{key}')")
#         if not all_required_found: break

#     if not all_required_found:
#         logger.warning(f"量能调整/分析模块：时间框架 '{current_vol_tf}' 因以下原因跳过分析: {'; '.join(missing_details)}。")
#         result_df[f'VOL_CONFIRM_SIGNAL_{current_vol_tf}'] = 0
#         result_df[f'VOL_SPIKE_SIGNAL_{current_vol_tf}'] = 0
#         result_df[f'VOL_PRICE_DIV_SIGNAL_{current_vol_tf}'] = 0
#         continue

#     # 获取实际列名 (此时，必需的列名保证存在于 *_cols_found 且在 data.columns 中)
#     close_col = ohlcv_cols_found['close']
#     high_col = ohlcv_cols_found['high']
#     low_col = ohlcv_cols_found['low']
#     volume_col = ohlcv_cols_found['volume']
#     cmf_col_name = indicator_cols_found['cmf']
#     obv_col_name = indicator_cols_found['obv']
#     obv_ma_col_name = indicator_cols_found.get('obv_ma') # 可选
#     print("adjust_score_with_volume.获取实际列名:")
#     print(f"close_col: {close_col}, high_col: {high_col}, low_col: {low_col}, volume_col: {volume_col}, cmf_col_name: {cmf_col_name}")
#     print(f"obv_col_name: {obv_col_name}, obv_ma_col_name: {obv_ma_col_name}")

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
    scoring_results = pd.DataFrame(index=data.index)
    if data.empty:
         logger.warning("输入 DataFrame 为空，无法计算指标评分。")
         print("DEBUG: 输入 DataFrame 为空，无法计算指标评分。")
         return scoring_results
    score_indicators_keys = bs_params.get('score_indicators', [])
    score_timeframes = bs_params.get('timeframes', [])
    if not score_indicators_keys or not score_timeframes:
        logger.warning("未配置需要评分的指标或时间框架 (base_scoring.score_indicators 或 base_scoring.timeframes)。")
        print("DEBUG: 未配置需要评分的指标或时间框架。")
        return scoring_results
    indicator_naming_conv = naming_config.get('indicator_naming_conventions', {})
    ohlcv_naming_conv = naming_config.get('ohlcv_naming_convention', {})
    timeframe_naming_conv = naming_config.get('timeframe_naming_convention', {})
    if not isinstance(indicator_naming_conv, dict): indicator_naming_conv = {}
    if not isinstance(ohlcv_naming_conv, dict): ohlcv_naming_conv = {}
    if not isinstance(timeframe_naming_conv, dict): timeframe_naming_conv = {}
    # --- 构建从 indicator_configs 到实际列名的映射 ---
    config_to_actual_col_map: Dict[Tuple[str, str, str], Union[str, Dict[str, str]]] = {}
    # 缓存 timeframe_naming_conv.get('patterns', {}) 以减少重复查找
    timeframe_patterns_cache = timeframe_naming_conv.get('patterns', {})
    if not isinstance(timeframe_patterns_cache, dict): timeframe_patterns_cache = {}
    if isinstance(indicator_configs, list):
        for config in indicator_configs:
            if not isinstance(config, dict): continue
            indicator_name = config.get('name', '').lower()
            timeframes_list = config.get('timeframes', [])
            if isinstance(timeframes_list, str): timeframes_list = [timeframes_list]
            if not isinstance(timeframes_list, list): continue
            indi_naming_conf = indicator_naming_conv.get(indicator_name.upper(), {})
            output_cols_patterns = indi_naming_conf.get('output_columns', [])
            if not isinstance(output_cols_patterns, list): output_cols_patterns = []
            if indicator_name == 'pivot':
                 pivot_levels_data = config.get('pivot_levels_data')
                 if isinstance(pivot_levels_data, dict):
                      # print(f"DEBUG: 处理 Pivot levels 配置数据: {pivot_levels_data.keys()}") # DEBUG信息，可按需保留或移除
                      for tf_conf in timeframes_list:
                           tf_str = str(tf_conf)
                           if tf_str in pivot_levels_data:
                                level_data_for_tf = pivot_levels_data[tf_str]
                                if isinstance(level_data_for_tf, dict):
                                     config_to_actual_col_map[(indicator_name, 'pivot_levels', tf_str)] = level_data_for_tf
                                     print(f"DEBUG: 添加 Pivot levels 配置映射，时间框架 {tf_str}，键: {level_data_for_tf.keys()}") # DEBUG信息
                                else: # DEBUG信息
                                     print(f"DEBUG: Pivot levels 数据 for tf {tf_str} 不是字典。") # DEBUG信息
                           else: # DEBUG信息
                                print(f"DEBUG: Pivot levels 数据中没有时间框架 {tf_str} 的条目。") # DEBUG信息
                 else: # DEBUG信息
                     print(f"DEBUG: Pivot levels 配置数据不是字典或不存在。") # DEBUG信息
                 continue
            actual_output_columns = config.get('output_columns', [])
            if isinstance(actual_output_columns, str): actual_output_columns = [actual_output_columns]
            if not isinstance(actual_output_columns, list): continue
            for actual_col_name in actual_output_columns:
                 if not isinstance(actual_col_name, str): continue
                 found_tf_suffix = None
                 original_tf_str_matched = None
                 for tf_conf in timeframes_list:
                      tf_str = str(tf_conf)
                      # 使用缓存的 timeframe_patterns_cache
                      possible_suffixes = timeframe_patterns_cache.get(tf_str.lower(), [tf_str])
                      if isinstance(possible_suffixes, str): possible_suffixes = [possible_suffixes]
                      if not isinstance(possible_suffixes, list): continue
                      possible_suffixes = [str(s) for s in possible_suffixes]
                      for suffix in possible_suffixes:
                           if actual_col_name.endswith(f"_{suffix}"):
                                found_tf_suffix = suffix
                                original_tf_str_matched = tf_str
                                break
                      if found_tf_suffix: break
                 if not found_tf_suffix:
                      parts = actual_col_name.split('_')
                      if len(parts) > 1:
                           guessed_suffix = parts[-1]
                           is_valid_guessed_suffix = False
                           for tf_conf in timeframes_list:
                                tf_str = str(tf_conf)
                                # 使用缓存的 timeframe_patterns_cache
                                possible_suffixes = timeframe_patterns_cache.get(tf_str.lower(), [tf_str])
                                if isinstance(possible_suffixes, str): possible_suffixes = [possible_suffixes]
                                if not isinstance(possible_suffixes, list): continue
                                possible_suffixes = [str(s) for s in possible_suffixes]
                                if guessed_suffix in possible_suffixes:
                                     is_valid_guessed_suffix = True
                                     found_tf_suffix = guessed_suffix
                                     original_tf_str_matched = tf_str
                                     break
                           if not is_valid_guessed_suffix:
                                found_tf_suffix = None
                                original_tf_str_matched = None
                 if not found_tf_suffix:
                      # print(f"WARNING: 无法从指标配置中确定列 '{actual_col_name}' 的时间框架后缀。") # DEBUG信息
                      continue
                 matched_internal_key = None
                 # 初始化 params 为 None，以避免 NameError，并使后续 if params is not None: 的逻辑按预期（如果 parse_col_params 未实现则跳过）执行
                 params: Dict[str, Any] | None = None
                 for col_conf in output_cols_patterns:
                      if isinstance(col_conf, dict) and 'name_pattern' in col_conf and 'internal_key' in col_conf:
                           pattern = col_conf['name_pattern']
                           internal_key_from_naming = col_conf['internal_key']
                           # 原始代码中 parse_col_params 被注释掉了。如果它被调用，它应该填充 'params'。
                           # 为了让代码可运行且逻辑上等同于注释掉 parse_col_params 的情况，我们已在循环外将 params 初始化为 None。
                           # 如果 parse_col_params 应该被调用，需要取消注释并确保其正确实现。
                           # params = parse_col_params(actual_col_name, indicator_name, found_tf_suffix, pattern) # 原注释掉的行
                           if params is not None: # 由于 params 初始化为 None 并且 parse_col_params 被注释，此块通常不会执行
                                temp_format_params = params.copy()
                                temp_format_params['timeframe'] = found_tf_suffix
                                try:
                                     expected_col_from_pattern = pattern.format(**temp_format_params).replace('__', '_').strip('_')
                                     if expected_col_from_pattern == actual_col_name:
                                          matched_internal_key = internal_key_from_naming
                                          # print(f"DEBUG: 列 '{actual_col_name}' 匹配 naming_config 模式 '{pattern}'，映射到内部键 '{matched_internal_key}'") # DEBUG信息
                                          break
                                except KeyError as e:
                                     # print(f"DEBUG: 反向匹配模式 '{pattern}' 时缺少参数 {e}") # DEBUG信息
                                     pass
                                except Exception as e:
                                     # print(f"DEBUG: 反向匹配模式 '{pattern}' 时发生未知错误: {e}") # DEBUG信息
                                     pass
                           # else: # DEBUG信息
                                # print(f"DEBUG: 由于 params 为 None (parse_col_params 未调用或返回 None)，跳过列 '{actual_col_name}' 基于模式 '{pattern}' 的匹配尝试。")
                 if matched_internal_key:
                      if original_tf_str_matched:
                           config_to_actual_col_map[(indicator_name, matched_internal_key, original_tf_str_matched)] = actual_col_name
                           # print(f"DEBUG: 映射配置列: ({indicator_name}, {matched_internal_key}, {original_tf_str_matched}) -> '{actual_col_name}'") # DEBUG信息
                      # else: # DEBUG信息
                           # print(f"WARNING: 找到匹配的内部键 '{matched_internal_key}'，但无法确定原始时间框架字符串。跳过映射。") # DEBUG信息
    # else: # DEBUG信息
        # print("DEBUG: indicator_configs 不是列表或为空。") # DEBUG信息
    for indicator_key in score_indicators_keys:
        info = indicator_scoring_info.get(indicator_key)
        if not info:
             logger.warning(f"指标 '{indicator_key}' 未找到对应的评分函数定义或配置，跳过评分计算。")
             print(f"DEBUG: 指标 '{indicator_key}' 未找到评分配置，跳过。")
             continue
        score_func = info.get('func')
        required_score_keys = info.get('required_keys', [])
        param_passing_style = info.get('param_passing_style', 'dict')
        bs_param_key_to_score_func_arg = info.get('bs_param_key_to_score_func_arg', {})
        defaults = info.get('defaults', {})
        key_patterns_info = info.get('key_patterns', {})
        if score_func is None:
             logger.warning(f"指标 '{indicator_key}' 没有关联的评分函数，跳过评分计算。")
             print(f"DEBUG: 指标 '{indicator_key}' 没有评分函数，跳过。")
             continue
        if not required_score_keys:
             logger.warning(f"指标 '{indicator_key}' 未配置所需的内部键 (required_keys)，跳过评分计算。")
             print(f"DEBUG: 指标 '{indicator_key}' 未配置 required_keys，跳过。")
             continue
        for tf_score in score_timeframes:
            indicator_cols_for_score: Dict[str, Union[str, Dict[str, pd.Series]]] = {}
            found = False
            tf_score_str = str(tf_score)
            temp_cols_from_config: Dict[str, Union[str, Dict[str, pd.Series]]] = {}
            for internal_key in required_score_keys:
                 if internal_key not in ['obv_ma', 'pivot_levels']:
                    config_key = (indicator_key, internal_key, tf_score_str)
                    if config_key in config_to_actual_col_map:
                         actual_data_source = config_to_actual_col_map[config_key]
                         if isinstance(actual_data_source, str) and actual_data_source in data.columns:
                              temp_cols_from_config[internal_key] = actual_data_source
                         else:
                              temp_cols_from_config.pop(internal_key, None)
                 elif internal_key == 'pivot_levels':
                     config_key = (indicator_key, internal_key, tf_score_str) # 假设 indicator_key 对于 pivot 是 'pivot'
                     if config_key in config_to_actual_col_map:
                          actual_data_source = config_to_actual_col_map[config_key]
                          if isinstance(actual_data_source, dict):
                              all_pivot_cols_exist = all(col_name in data.columns for col_name in actual_data_source.values())
                              if all_pivot_cols_exist:
                                   temp_cols_from_config[internal_key] = {level_key: data[col_name] for level_key, col_name in actual_data_source.items()}
                          # else: # DEBUG信息
                            #    print(f"DEBUG: 时间框架 {tf_score_str} 的 Pivot levels 配置映射不是字典。配置查找此键失败。") # DEBUG信息
                     # else: # DEBUG信息
                        #  print(f"DEBUG: Pivot levels 键 '{internal_key}' 未在配置映射中找到 ({indicator_key}, {tf_score_str})。配置查找此键失败。") # DEBUG信息
                 elif internal_key == 'obv_ma':
                     config_key = (indicator_key, internal_key, tf_score_str)
                     if config_key in config_to_actual_col_map:
                          actual_data_source = config_to_actual_col_map[config_key]
                          if isinstance(actual_data_source, str) and actual_data_source in data.columns:
                               temp_cols_from_config[internal_key] = actual_data_source
            required_keys_base = [k for k in required_score_keys if k not in ['obv_ma', 'pivot_levels']]
            all_base_required_found = all(k in temp_cols_from_config for k in required_keys_base)
            pivot_levels_found_ok = True
            if 'pivot_levels' in required_score_keys:
                 pivot_levels_found_ok = isinstance(temp_cols_from_config.get('pivot_levels'), dict) and bool(temp_cols_from_config.get('pivot_levels'))
            if all_base_required_found and pivot_levels_found_ok:
                 indicator_cols_for_score = temp_cols_from_config
                 found = True
            current_tf_possible_suffixes: List[str] = [] # 在 fallback 之外声明，以用于后续的调试信息
            if not found:
                # 使用之前缓存的 timeframe_patterns_cache
                tf_score_str_lower = tf_score_str.lower()
                patterns_for_tf = timeframe_patterns_cache.get(tf_score_str_lower, [tf_score_str])
                if isinstance(patterns_for_tf, str): patterns_for_tf = [patterns_for_tf]
                if isinstance(patterns_for_tf, list):
                    current_tf_possible_suffixes = [str(p) for p in patterns_for_tf]
                else: # 防御性编程，确保 current_tf_possible_suffixes 是列表
                    current_tf_possible_suffixes = [tf_score_str]
                if tf_score_str not in current_tf_possible_suffixes:
                     current_tf_possible_suffixes.append(tf_score_str)
                # 将 ohlcv_output_cols_conf 的获取移到 tf_suffix 循环外，但在 internal_key 循环内按需获取
                # 将 pivot_naming_convention 的获取移到 tf_suffix 循环外，但在 internal_key 循环内按需获取
                for tf_suffix in current_tf_possible_suffixes:
                    temp_cols_found: Dict[str, Union[str, Dict[str, pd.Series]]] = {}
                    all_required_found_for_suffix = True
                    # 提前获取 OHLCV 和 Pivot 的命名配置，如果它们在循环中被多次使用
                    # （注意：这里保持在 internal_key 循环内按需获取，因为它们仅用于特定 internal_key）
                    for internal_key in required_score_keys:
                        if internal_key == 'close':
                            # ohlcv_output_cols_conf 在 internal_key=='close' 时获取一次
                            ohlcv_output_cols_conf = ohlcv_naming_conv.get('output_columns', [])
                            close_pattern = None
                            if isinstance(ohlcv_output_cols_conf, list):
                                 for col_conf in ohlcv_output_cols_conf:
                                     if isinstance(col_conf, dict) and col_conf.get('internal_key') == 'close':
                                          close_pattern = col_conf.get('name_pattern')
                                          break
                            if close_pattern:
                                expected_col_name = f"{close_pattern}_{tf_suffix}"
                                if expected_col_name in data.columns:
                                    temp_cols_found[internal_key] = expected_col_name
                                else:
                                    all_required_found_for_suffix = False
                                    # print(f"DEBUG: 必需的 'close' 列 '{expected_col_name}' 未找到，后缀 '{tf_suffix}'。") # DEBUG信息
                                    break
                            else:
                                logger.warning(f"回退查找: 未找到 'close' 的命名规范。无法为时间框架 {tf_score} 找到 close 列。")
                                all_required_found_for_suffix = False
                                break
                        elif indicator_key == 'pivot' and internal_key == 'pivot_levels':
                             # pivot_naming_convention 在 internal_key=='pivot_levels' 时获取一次
                             pivot_naming_convention = indicator_naming_conv.get('PIVOT', {}) # 假设指标键大写
                             pivot_cols_base = pivot_naming_convention.get('levels', ["PP", "S1", "S2", "S3", "S4", "R1", "R2", "R3", "R4", "F_R1", "F_R2", "F_R3", "F_S1", "F_S2", "F_S3"])
                             pivot_level_pattern = pivot_naming_convention.get('pattern', "{level}_{timeframe}")
                             pivot_levels_series_dict_for_score: Dict[str, pd.Series] = {}
                             all_pivot_levels_found_as_series = True
                             for p_base in pivot_cols_base:
                                 try:
                                     col_name = pivot_level_pattern.format(level=p_base, timeframe=tf_suffix)
                                     col_name = col_name.replace('__', '_').strip('_')
                                 except KeyError as e:
                                     logger.warning(f"Pivot levels 模式 '{pivot_level_pattern}' 缺少格式化参数: {e}。无法构建列名。")
                                     all_pivot_levels_found_as_series = False
                                     break
                                 except Exception as e: # pylint: disable=broad-except
                                     logger.warning(f"格式化 Pivot levels 模式 '{pivot_level_pattern}' 时发生未知错误: {e}。无法构建列名。")
                                     all_pivot_levels_found_as_series = False
                                     break
                                 if col_name in data.columns:
                                     pivot_levels_series_dict_for_score[p_base] = data[col_name]
                                 else:
                                     all_pivot_levels_found_as_series = False
                                     # print(f"DEBUG: 枢轴点级别列 '{col_name}' 未找到，后缀 '{tf_suffix}'。") # DEBUG信息
                                     break
                             if all_pivot_levels_found_as_series:
                                 temp_cols_found[internal_key] = pivot_levels_series_dict_for_score
                             else:
                                 all_required_found_for_suffix = False
                                 break
                        else:
                             key_pattern_info = key_patterns_info.get(internal_key)
                             if not key_pattern_info or not isinstance(key_pattern_info, dict):
                                  if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                       logger.warning(f"回退查找: 未在 indicator_scoring_info['{indicator_key}']['key_patterns'] 中找到内部键 '{internal_key}' 的模式配置 (非可选键)。")
                                       all_required_found_for_suffix = False
                                       break
                                  else:
                                       continue
                             pattern = key_pattern_info.get('pattern')
                             params_map = key_pattern_info.get('params_map', {})
                             if not pattern or not isinstance(pattern, str):
                                  if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                       logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 在 key_patterns 中的模式配置无效 (非可选键)。")
                                       all_required_found_for_suffix = False
                                       break
                                  else:
                                       continue
                             format_params: Dict[str, Any] = {'timeframe': tf_suffix}
                             params_found_for_pattern = True
                             for pattern_param_name, bs_param_key in params_map.items():
                                  param_value = bs_params.get(bs_param_key, defaults.get(bs_param_key, None))
                                  if param_value is not None:
                                       format_params[pattern_param_name] = param_value
                                  else:
                                       if '{' + pattern_param_name + '}' in pattern:
                                            logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 的模式 '{pattern}' 需要参数 '{pattern_param_name}' (对应 bs_key '{bs_param_key}')，但在 bs_params 和 defaults 中未找到。")
                                            params_found_for_pattern = False
                                            break
                                       # else: # DEBUG信息
                                            # print(f"DEBUG: 模式 '{pattern}' 不需要参数 '{pattern_param_name}'，跳过获取。") # DEBUG信息
                             if params_found_for_pattern:
                                  expected_col_name = None
                                  try:
                                       expected_col_name = pattern.format(**format_params)
                                       expected_col_name = expected_col_name.replace('__', '_').strip('_')
                                  except KeyError as e:
                                       logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 的模式 '{pattern}' 格式化失败，缺少参数: {e}.")
                                       all_required_found_for_suffix = False
                                       break
                                  except ValueError as e: # pylint: disable=broad-except
                                       logger.warning(f"回退查找: 指标 '{indicator_key}' 的内部键 '{internal_key}' 的模式 '{pattern}' 格式化值错误: {e}. 参数: {format_params}.")
                                       all_required_found_for_suffix = False
                                       break
                                  except Exception as e: # pylint: disable=broad-except
                                       logger.warning(f"回退查找: 格式化模式 '{pattern}' 时发生未知错误: {e}.")
                                       all_required_found_for_suffix = False
                                       break
                                  if expected_col_name and expected_col_name in data.columns:
                                      temp_cols_found[internal_key] = expected_col_name
                                  else:
                                      if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                           all_required_found_for_suffix = False
                                        #    print(f"DEBUG: 通过回退未找到内部键 '{internal_key}' 的必需列，后缀 '{tf_suffix}'。期望列名: '{expected_col_name}'") # DEBUG信息
                                           break
                                    #   else: # DEBUG信息
                                        #    print(f"DEBUG: 通过回退未找到内部键 '{internal_key}' 的可选列，后缀 '{tf_suffix}'。期望列名: '{expected_col_name}'。") # DEBUG信息
                             else:
                                  if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                       all_required_found_for_suffix = False
                                       break
                                #   else: # DEBUG信息
                                    #    print(f"DEBUG: 回退查找: 可选键 '{internal_key}' 缺少格式化模式所需参数，跳过此键的查找。") # DEBUG信息
                    required_keys_base_fallback = [k for k in required_score_keys if k not in ['obv_ma', 'pivot_levels']]
                    all_base_required_found_fallback = all(k in temp_cols_found for k in required_keys_base_fallback)
                    pivot_levels_found_ok_fallback = True
                    if 'pivot_levels' in required_score_keys:
                         pivot_levels_found_ok_fallback = isinstance(temp_cols_found.get('pivot_levels'), dict) and bool(temp_cols_found.get('pivot_levels'))
                    if all_base_required_found_fallback and pivot_levels_found_ok_fallback:
                         indicator_cols_for_score = temp_cols_found
                         found = True
                         break
                    # else: # DEBUG信息
                        #  print(f"DEBUG: 回退查找，后缀 '{tf_suffix}' 未找到所有必需列或必需特殊结构。") # DEBUG信息
            if not found:
                logger.warning(f"未能为指标 '{indicator_key}' 在时间框架 {tf_score} 找到所有必要的数据列进行评分。")
                logger.info(f"尝试查找所需的内部键: {required_score_keys}.")
                logger.info(f"尝试的后缀列表: {current_tf_possible_suffixes}.") # current_tf_possible_suffixes 在此作用域内有效
                relevant_cols_for_tf = []
                all_prefixes = []
                if isinstance(key_patterns_info, dict):
                     for kp_info in key_patterns_info.values():
                          if isinstance(kp_info, dict) and 'pattern' in kp_info:
                               pattern_val = kp_info['pattern']
                               pattern_base = pattern_val.split('_')[0] if '_' in pattern_val else pattern_val
                               if pattern_base and pattern_base not in all_prefixes:
                                    all_prefixes.append(pattern_base)
                score_info_prefixes = info.get('prefixes', [])
                for p in score_info_prefixes:
                     if p and p not in all_prefixes: all_prefixes.append(p)
                if 'close' in required_score_keys:
                     ohlcv_output_cols_conf_debug = ohlcv_naming_conv.get('output_columns', [])
                     close_pattern_prefix = 'close'
                     if isinstance(ohlcv_output_cols_conf_debug, list):
                          for col_conf_debug in ohlcv_output_cols_conf_debug:
                               if isinstance(col_conf_debug, dict) and col_conf_debug.get('internal_key') == 'close':
                                    close_pattern_prefix = col_conf_debug.get('name_pattern', 'close').split('_')[0]
                                    break
                     if close_pattern_prefix and close_pattern_prefix not in all_prefixes:
                          all_prefixes.append(close_pattern_prefix)
                if indicator_key == 'pivot': # 假设指标键是小写
                     pivot_naming_convention_debug = indicator_naming_conv.get('PIVOT', {}) # 假设配置中的键是大写
                     pivot_cols_base_debug = pivot_naming_convention_debug.get('levels', [])
                     for p_base_debug in pivot_cols_base_debug:
                         if p_base_debug and p_base_debug not in all_prefixes: all_prefixes.append(p_base_debug)
                if indicator_key.upper() not in [p.upper() for p in all_prefixes]:
                    all_prefixes.append(indicator_key.upper())
                all_prefixes = [p for p in all_prefixes if p]
                # print(f"DEBUG: 未找到列，尝试列出相关列。使用的前缀: {all_prefixes}, 可能后缀: {current_tf_possible_suffixes}") # DEBUG信息
                cols_to_check = []
                if data.columns.empty: # 增加对空列的检查
                    logger.info(f"DataFrame 为空，无法检查相关列。")
                else:
                    for col in data.columns:
                        col_str = str(col) # 确保列名是字符串
                        if any(col_str.startswith(str(prefix)) for prefix in all_prefixes) and \
                           any(col_str.endswith(f"_{str(s)}") for s in current_tf_possible_suffixes):
                            cols_to_check.append(col_str)
                relevant_cols_for_tf = [c for c in cols_to_check if c in data.columns]
                logger.info(f"DataFrame 中与时间框架 {tf_score} 匹配的 '{indicator_key}' 相关列列表: {sorted(relevant_cols_for_tf)}.")
                print(f"DEBUG: DataFrame 中与时间框架 {tf_score} 匹配的 '{indicator_key}' 相关列列表: {sorted(relevant_cols_for_tf)}.")
                score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                scoring_results[score_col_name] = 50.0
                print(f"DEBUG: 未找到必需列，列 '{score_col_name}' 填充默认评分 50.0。")
            if found:
                try:
                    positional_series_args: List[pd.Series] = []
                    keyword_score_func_args: Dict[str, Any] = {}
                    valid_call = True # 标记评分函数调用是否有效
                    for internal_key in required_score_keys:
                         actual_data_source = indicator_cols_for_score.get(internal_key)
                         if actual_data_source is None:
                              if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                                   logger.error(f"内部错误: 指标 '{indicator_key}' 在时间框架 {tf_score} 必需的内部键 '{internal_key}' 没有找到对应的数据源 (在 found=True 的情况下)。")
                                   valid_call = False; break # 中断并标记无效
                              else: # obv_ma 是可选的，可以为 None
                                  # print(f"DEBUG: 可选键 '{internal_key}' 未找到数据源，将作为 None (或不作为参数) 传递。") # DEBUG信息
                                  continue # 对于可选的 None，继续处理其他参数
                         if indicator_key == 'pivot' and internal_key == 'pivot_levels':
                              if isinstance(actual_data_source, dict) and bool(actual_data_source):
                                  if all(isinstance(s, pd.Series) for s in actual_data_source.values()):
                                       keyword_score_func_args['pivot_levels'] = actual_data_source
                                  else:
                                       logger.error(f"内部错误: 指标 'pivot' 在时间框架 {tf_score}: 'pivot_levels' 字典中包含非 Series 值。")
                                       valid_call = False; break
                              elif 'pivot_levels' in required_score_keys:
                                   logger.error(f"内部错误: 指标 'pivot' 在时间框架 {tf_score}: 必需的 pivot_levels 数据未找到或格式无效。")
                                   valid_call = False; break
                         elif isinstance(actual_data_source, str) and actual_data_source in data.columns:
                             series_data = data[actual_data_source]
                             if param_passing_style == 'none':
                                  positional_series_args.append(series_data)
                             elif param_passing_style == 'dict':
                                  keyword_score_func_args[internal_key] = series_data
                             elif param_passing_style == 'individual':
                                  keyword_score_func_args[internal_key] = series_data
                             else:
                                  logger.error(f"指标 '{indicator_key}' 配置了未知的参数传递风格: '{param_passing_style}'.")
                                  valid_call = False; break
                         elif indicator_key == 'obv' and internal_key == 'obv_ma': # actual_data_source 可能为 None
                              if actual_data_source and isinstance(actual_data_source, str) and actual_data_source in data.columns:
                                   keyword_score_func_args['obv_ma'] = data[actual_data_source]
                              # else: obv_ma 未找到或为 None，不添加到参数中，评分函数应能处理
                         # else: # 对于其他未找到数据源的情况，如果非可选，则 valid_call 应已为 False
                            # if not (indicator_key == 'obv' and internal_key == 'obv_ma'):
                            # logger.error(f"内部错误: 未处理的数据源类型或缺失。 Key: {internal_key}, Source: {actual_data_source}")
                            # valid_call = False; break
                    if not valid_call: # 如果参数准备失败
                        raise ValueError(f"准备指标 '{indicator_key}'@{tf_score_str} 的评分函数参数时失败。")
                    score_func_params: Dict[str, Any] = {}
                    for bs_key, func_arg_name in bs_param_key_to_score_func_arg.items():
                         param_value = bs_params.get(bs_key, defaults.get(bs_key, None))
                         if param_value is not None:
                              score_func_params[func_arg_name] = param_value
                    final_keyword_args = keyword_score_func_args.copy()
                    score: pd.Series | None = None # 初始化 score
                    if param_passing_style == 'dict':
                         final_keyword_args['params'] = score_func_params
                         score = score_func(**final_keyword_args)
                    elif param_passing_style == 'individual':
                         final_keyword_args.update(score_func_params)
                         score = score_func(**final_keyword_args)
                    elif param_passing_style == 'none':
                         if score_func_params:
                              logger.warning(f"指标 '{indicator_key}' 配置为 'none' 参数传递风格，但 bs_param_key_to_score_func_arg 或 defaults 不为空。这些参数可能被忽略。")
                         score = score_func(*positional_series_args)
                    else: # 已在参数准备阶段检查过，但作为防御
                         raise ValueError(f"未处理的 param_passing_style: {param_passing_style} for indicator {indicator_key}")
                    if not isinstance(score, pd.Series):
                        logger.error(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的评分函数未返回 pandas Series.")
                        raise TypeError("评分函数未返回 pandas Series") # 抛出错误由外部捕获
                    if not score.index.equals(data.index):
                         logger.error(f"指标 '{indicator_key}' 在时间框架 {tf_score} 的评分结果索引与输入数据不一致.")
                         raise ValueError("评分结果索引与输入数据不一致") # 抛出错误
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                    scoring_results[score_col_name] = score
                except Exception as e: # pylint: disable=broad-except
                    logger.error(f"计算指标 '{indicator_key}' 在时间框架 {tf_score} 的评分时发生错误: {e}", exc_info=True)
                    print(f"DEBUG: 计算指标 '{indicator_key}' 在时间框架 {tf_score} 的评分时发生错误: {e}")
                    score_col_name = f"SCORE_{indicator_key.upper()}_{tf_score_str}"
                    scoring_results[score_col_name] = 50.0
                    logger.warning(f"指标 '{indicator_key}' 在时间框架 {tf_score} 评分计算失败，列 '{score_col_name}' 填充默认评分 50.0。")
                    print(f"DEBUG: 评分计算失败，列 '{score_col_name}' 填充默认评分 50.0。")
    return scoring_results

# 添加了 pattern_params_map 字段，用于描述内部键对应的列名模式中的参数如何映射到 bs_params 中的键
# 这个映射将用于回退查找逻辑中构建期望列名。
indicator_scoring_info: Dict[str, Dict[str, Any]] = {
    'macd': {
        'func': calculate_macd_score,  # MACD评分函数引用
        'param_passing_style': 'none',  # 评分函数只接受Series，不接受额外配置参数
        'bs_param_key_to_score_func_arg': {},  # 无需从bs_params向评分函数传递参数
        'defaults': {  # bs_params中MACD参数的默认值，主要供key_patterns使用
            'macd_fast': 12, 
            'macd_slow': 26, 
            'macd_signal': 9
        },
        'required_keys': ['macd_series', 'macd_d', 'macd_h'],  # 评分函数必需的内部数据键
        'prefixes': ['MACD_', 'MACDh_', 'MACDs_'],  # 列查找失败时的调试前缀
        'key_patterns': {  # 如何从bs_params构建列名模式 (用于回退查找)
            'macd_series': {'pattern': 'MACD_{period_fast}_{period_slow}_{signal_period}_{timeframe}', 'params_map': {'period_fast': 'macd_fast', 'period_slow': 'macd_slow', 'signal_period': 'macd_signal'}},
            'macd_d': {'pattern': 'MACDs_{period_fast}_{period_slow}_{signal_period}_{timeframe}', 'params_map': {'period_fast': 'macd_fast', 'period_slow': 'macd_slow', 'signal_period': 'macd_signal'}},
            'macd_h': {'pattern': 'MACDh_{period_fast}_{period_slow}_{signal_period}_{timeframe}', 'params_map': {'period_fast': 'macd_fast', 'period_slow': 'macd_slow', 'signal_period': 'macd_signal'}}
        }
    },
    'rsi': {
        'func': calculate_rsi_score,  # RSI评分函数引用
        'param_passing_style': 'dict',  # 评分函数接受Series及来自bs_params的配置参数
        'bs_param_key_to_score_func_arg': {  # 将bs_params键映射到评分函数参数名
            'rsi_period': 'period',  # JSON 'rsi_period' -> 评分函数 'period'
            'rsi_oversold': 'oversold',
            'rsi_overbought': 'overbought',
            'rsi_extreme_oversold': 'extreme_oversold',
            'rsi_extreme_overbought': 'extreme_overbought'
        },
        'defaults': {  # bs_params中RSI参数的默认值
            'rsi_period': 14, 
            'rsi_oversold': 30, 
            'rsi_overbought': 70, 
            'rsi_extreme_oversold': 20, 
            'rsi_extreme_overbought': 80
        },
        'required_keys': ['rsi'],  # 必需的RSI数据列
        'prefixes': ['RSI_'],
        'key_patterns': {
            'rsi': {'pattern': 'RSI_{period}_{timeframe}', 'params_map': {'period': 'rsi_period'}}
        }
    },
    'kdj': {
        'func': calculate_kdj_score,  # KDJ评分函数引用
        'param_passing_style': 'dict',
        'bs_param_key_to_score_func_arg': {  # KDJ评分函数直接需要的参数 (通常是阈值)
            'kdj_oversold': 'oversold',
            'kdj_overbought': 'overbought',
            'kdj_extreme_oversold': 'extreme_oversold',
            'kdj_extreme_overbought': 'extreme_overbought'
            # KDJ周期参数 (kdj_period_k等) 用于key_patterns构建列名，不直接传给评分函数
        },
        'defaults': {
            'kdj_period_k': 9, 
            'kdj_period_d': 3, 
            'kdj_period_j': 3, 
            'kdj_oversold': 20, 
            'kdj_overbought': 80, 
            'kdj_extreme_oversold': 10, 
            'kdj_extreme_overbought': 90
        },
        'required_keys': ['k', 'd', 'j'],
        'prefixes': ['K_', 'D_', 'J_'],
        'key_patterns': {
            'k': {'pattern': 'K_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', 'params_map': {'k_period': 'kdj_period_k', 'd_period': 'kdj_period_d', 'smooth_k_period': 'kdj_period_j'}},
            'd': {'pattern': 'D_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', 'params_map': {'k_period': 'kdj_period_k', 'd_period': 'kdj_period_d', 'smooth_k_period': 'kdj_period_j'}},
            'j': {'pattern': 'J_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', 'params_map': {'k_period': 'kdj_period_k', 'd_period': 'kdj_period_d', 'smooth_k_period': 'kdj_period_j'}},
        }
    },
    'boll': {
       'func': calculate_boll_score,  # BOLL评分函数引用
       'param_passing_style': 'none', # 只接受Series
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'boll_period': 20, 'boll_std_dev': 2.0}, # 周期和标准差用于key_patterns
       'required_keys': ['close', 'upper', 'mid', 'lower'],
       'prefixes': ['BBL_', 'BBM_', 'BBU_'], # 布林带各线的前缀
       'key_patterns': { # 'close'列由OHLCV配置处理
            'upper': {'pattern': 'BBU_{period}_{std_dev:.1f}_{timeframe}', 'params_map': {'period': 'boll_period', 'std_dev': 'boll_std_dev'}},
            'mid': {'pattern': 'BBM_{period}_{std_dev:.1f}_{timeframe}', 'params_map': {'period': 'boll_period', 'std_dev': 'boll_std_dev'}},
            'lower': {'pattern': 'BBL_{period}_{std_dev:.1f}_{timeframe}', 'params_map': {'period': 'boll_period', 'std_dev': 'boll_std_dev'}},
       }
    },
    'cci': {
       'func': calculate_cci_score,  # CCI评分函数引用
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {
            'cci_period': 'period',
            'cci_threshold': 'threshold',
            'cci_extreme_threshold': 'extreme_threshold'
       },
       'defaults': {'cci_period': 14, 'cci_threshold': 100, 'cci_extreme_threshold': 200},
       'required_keys': ['cci'],
       'prefixes': ['CCI_'],
       'key_patterns': {
            'cci': {'pattern': 'CCI_{period}_{timeframe}', 'params_map': {'period': 'cci_period'}}
            }
    },
    'mfi': {
       'func': calculate_mfi_score,  # MFI评分函数引用
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {
            'mfi_period': 'period',
            'mfi_oversold': 'oversold',
            'mfi_overbought': 'overbought',
            'mfi_extreme_oversold': 'extreme_oversold',
            'mfi_extreme_overbought': 'extreme_overbought'
       },
       'defaults': {'mfi_period': 14, 'mfi_oversold': 20, 'mfi_overbought': 80, 'mfi_extreme_oversold': 10, 'mfi_extreme_overbought': 90},
       'required_keys': ['mfi'],
       'prefixes': ['MFI_'],
       'key_patterns': {
            'mfi': {'pattern': 'MFI_{period}_{timeframe}', 'params_map': {'period': 'mfi_period'}}
            }
    },
    'roc': {
       'func': calculate_roc_score,  # ROC评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'roc_period': 12}, # 周期用于key_patterns
       'required_keys': ['roc'],
       'prefixes': ['ROC_'],
       'key_patterns': {
            'roc': {'pattern': 'ROC_{period}_{timeframe}', 'params_map': {'period': 'roc_period'}}
            }
    },
    'dmi': {
       'func': calculate_dmi_score,  # DMI评分函数引用
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': { # DMI评分函数直接需要的参数
            'dmi_period': 'period', # 假设评分函数也用这个周期参数进行某些判断
            'adx_threshold': 'adx_threshold',
            'adx_strong_threshold': 'adx_strong_threshold'
       },
       'defaults': {'dmi_period': 14, 'adx_threshold': 25, 'adx_strong_threshold': 40},
       'required_keys': ['pdi', 'ndi', 'adx'], # PDI, NDI, ADX三条线
       'prefixes': ['PDI_', 'NDI_', 'ADX_'],
       'key_patterns': { # DMI各线的周期参数与dmi_period一致
            'pdi': {'pattern': 'PDI_{period}_{timeframe}', 'params_map': {'period': 'dmi_period'}},
            'ndi': {'pattern': 'NDI_{period}_{timeframe}', 'params_map': {'period': 'dmi_period'}},
            'adx': {'pattern': 'ADX_{period}_{timeframe}', 'params_map': {'period': 'dmi_period'}},
       }
    },
    'sar': {
       'func': calculate_sar_score,  # SAR评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'sar_step': 0.02, 'sar_max': 0.2}, # af_step和max_af用于key_patterns
       'required_keys': ['close', 'sar'],
       'prefixes': ['SAR_'],
       'key_patterns': { # 'close'列由OHLCV配置处理
            'sar': {'pattern': 'SAR_{af_step:.2f}_{max_af:.2f}_{timeframe}', 'params_map': {'af_step': 'sar_step', 'max_af': 'sar_max'}}
            }
    },
    'stoch': {
       'func': calculate_stoch_score,  # STOCH评分函数引用
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': { # STOCH评分函数直接需要的参数
            'stoch_k': 'k_period', # 假设评分函数也用这些周期参数
            'stoch_d': 'd_period',
            'stoch_smooth_k': 'smooth_k_period',
            'stoch_oversold': 'stoch_oversold',
            'stoch_overbought': 'stoch_overbought',
            'stoch_extreme_oversold': 'stoch_extreme_oversold',
            'stoch_extreme_overbought': 'stoch_extreme_overbought'
       },
       'defaults': {'stoch_k': 14, 'stoch_d': 3, 'stoch_smooth_k': 3, 'stoch_oversold': 20, 'stoch_overbought': 80, 'stoch_extreme_oversold': 10, 'stoch_extreme_overbought': 90},
       'required_keys': ['k', 'd'], # STOCH K线和D线
       'prefixes': ['STOCHk_', 'STOCHd_'],
       'key_patterns': {
            'k': {'pattern': 'STOCHk_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', 'params_map': {'k_period': 'stoch_k', 'd_period': 'stoch_d', 'smooth_k_period': 'stoch_smooth_k'}},
            'd': {'pattern': 'STOCHd_{k_period}_{d_period}_{smooth_k_period}_{timeframe}', 'params_map': {'k_period': 'stoch_k', 'd_period': 'stoch_d', 'smooth_k_period': 'stoch_smooth_k'}},
       }
    },
    'ema': { # 使用通用的MA评分函数
       'func': calculate_ma_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'ema_period': 'period'}, # 将JSON的'ema_period'作为'period'参数传给评分函数
       'defaults': {'ema_period': 20},
       'required_keys': ['close', 'ma'], # MA评分函数需要收盘价和MA线本身
       'prefixes': ['EMA_'],
       'key_patterns': { # 'close'由OHLCV处理
            'ma': {'pattern': 'EMA_{period}_{timeframe}', 'params_map': {'period': 'ema_period'}} # 'ma'这个内部键代表EMA线
            }
    },
    'sma': { # 使用通用的MA评分函数
       'func': calculate_ma_score,
       'param_passing_style': 'dict',
       'bs_param_key_to_score_func_arg': {'sma_period': 'period'}, # 将JSON的'sma_period'作为'period'参数传给评分函数
       'defaults': {'sma_period': 20},
       'required_keys': ['close', 'ma'],
       'prefixes': ['SMA_'],
       'key_patterns': { # 'close'由OHLCV处理
            'ma': {'pattern': 'SMA_{period}_{timeframe}', 'params_map': {'period': 'sma_period'}} # 'ma'这个内部键代表SMA线
            }
    },
    'atr': {
       'func': calculate_atr_score,  # ATR评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'atr_period':14}, # 周期用于key_patterns
       'required_keys': ['atr'],
       'prefixes':['ATR_'],
       'key_patterns':{'atr':{'pattern':'ATR_{period}_{timeframe}', 'params_map':{'period':'atr_period'}}}
    },
    'adl': {
       'func': calculate_adl_score,  # ADL评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {}, # ADL通常无参数
       'required_keys': ['adl'],
       'prefixes':['ADL'], # 注意：ADL 列名可能不包含下划线，若原始列名为ADL_5，则此前缀应为ADL_
       'key_patterns':{'adl':{'pattern':'ADL_{timeframe}', 'params_map':{}}} # 假设ADL列名为 ADL_{timeframe}
    },
    'vwap': {
       'func': calculate_vwap_score,  # VWAP评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {}, # VWAP计算参数（如anchor）在指标计算服务中处理
       'required_keys': ['close', 'vwap'],
       'prefixes':['VWAP_'],
       'key_patterns':{ # 'close'由OHLCV处理
           'vwap':{'pattern':'VWAP_{timeframe}', 'params_map':{}}} # 假设VWAP列名为 VWAP_{timeframe}
    },
    'ichimoku': {
       'func': calculate_ichimoku_score,  # Ichimoku评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': { # Ichimoku各线的周期参数，用于key_patterns
           'ichimoku_tenkan_period':9,
           'ichimoku_kijun_period':26,
           'ichimoku_senkou_b_period':52, # Senkou Span B 周期
           'ichimoku_chikou_period':26  # Chikou Span 的位移周期，也可能影响Senkou A的计算
           # senkou_a 通常由 tenkan 和 kijun 决定，其本身的独立周期参数较少见，但如果pandas_ta用不同方式，需调整
       },
       'required_keys': ['close', 'tenkan', 'kijun', 'senkou_a', 'senkou_b', 'chikou'],
       'prefixes':['TENKAN_', 'KIJUN_', 'CHIKOU_', 'SENKOU_A_', 'SENKOU_B_', 'ITS_', 'IKS_', 'ISA_', 'ISB_', 'ICS_'], # 包含可能的pandas_ta前缀
       'key_patterns': { # 'close'由OHLCV处理. 这些模式需要精确匹配指标计算服务生成的列名（在加_{timeframe}后缀之前的基础部分）
            'tenkan': {'pattern': 'ITS_{period}_{timeframe}', 'params_map': {'period': 'ichimoku_tenkan_period'}}, # pandas_ta默认: ITS_9
            'kijun': {'pattern': 'IKS_{period}_{timeframe}', 'params_map': {'period': 'ichimoku_kijun_period'}},   # pandas_ta默认: IKS_26
            'senkou_a': {'pattern': 'ISA_{tenkan_period}_{kijun_period}_{timeframe}', 'params_map': {'tenkan_period': 'ichimoku_tenkan_period', 'kijun_period': 'ichimoku_kijun_period'}}, # pandas_ta默认: ISA_9_26
            'senkou_b': {'pattern': 'ISB_{period}_{timeframe}', 'params_map': {'period': 'ichimoku_senkou_b_period'}}, # pandas_ta默认: ISB_52
            'chikou': {'pattern': 'ICS_{period}_{timeframe}', 'params_map': {'period': 'ichimoku_chikou_period'}},   # pandas_ta默认: ICS_26 (Chikou Span是收盘价向过去位移，其周期是位移量)
        }
    },
    'mom': {
       'func': calculate_mom_score,  # MOM评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'mom_period':10}, # 周期用于key_patterns
       'required_keys': ['mom'],
       'prefixes':['MOM_'],
       'key_patterns':{'mom':{'pattern':'MOM_{period}_{timeframe}', 'params_map':{'period':'mom_period'}}}
    },
    'willr': {
       'func': calculate_willr_score,  # WILLR评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'willr_period':14}, # 周期用于key_patterns
       'required_keys': ['willr'],
       'prefixes':['WILLR_'],
       'key_patterns':{'willr':{'pattern':'WILLR_{period}_{timeframe}', 'params_map':{'period':'willr_period'}}}
    },
    'cmf': {
       'func': calculate_cmf_score,  # CMF评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'cmf_period':20}, # 周期用于key_patterns
       'required_keys': ['cmf'],
       'prefixes':['CMF_'],
       'key_patterns':{'cmf':{'pattern':'CMF_{period}_{timeframe}', 'params_map':{'period':'cmf_period'}}}
    },
    'obv': {
       'func': calculate_obv_score,  # OBV评分函数引用
       'param_passing_style': 'individual', # OBV_MA_PERIOD是独立关键字参数
       'bs_param_key_to_score_func_arg': {'obv_ma_period': 'obv_ma_period'}, # 将JSON的'obv_ma_period'映射到评分函数同名参数
       'defaults': {'obv_ma_period': 10}, # OBV MA的周期
       'required_keys': ['obv', 'obv_ma'], # OBV线, OBV的移动平均线 (obv_ma是可选的)
       'prefixes': ['OBV_'],
       'key_patterns': {
            'obv': {'pattern': 'OBV_{timeframe}', 'params_map': {}}, # OBV本身通常无参数
            'obv_ma': {'pattern': 'OBV_MA_{period}_{timeframe}', 'params_map': {'period': 'obv_ma_period'}}, # OBV均线的模式
       }
    },
    'kc': {
       'func': calculate_kc_score,  # Keltner Channels评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'kc_ema_period':20, 'kc_atr_period':10, 'kc_atr_multiplier': 2.0}, # 参数用于key_patterns
       'required_keys': ['close', 'upper', 'mid', 'lower'], # Keltner三条线和收盘价
       'prefixes':['KCL_', 'KCM_', 'KCU_'], # Keltner各线的前缀
       'key_patterns': { # 'close'列由OHLCV配置处理
            # 注意：KC的列名通常包含EMA周期和ATR周期/乘数，模式需精确匹配
            'upper': {'pattern': 'KCU_{ema_period}_{atr_period}_{atr_multiplier:.1f}_{timeframe}', 'params_map': {'ema_period': 'kc_ema_period', 'atr_period': 'kc_atr_period', 'atr_multiplier': 'kc_atr_multiplier'}},
            'mid': {'pattern': 'KCM_{ema_period}_{atr_period}_{atr_multiplier:.1f}_{timeframe}', 'params_map': {'ema_period': 'kc_ema_period', 'atr_period': 'kc_atr_period', 'atr_multiplier': 'kc_atr_multiplier'}},
            'lower': {'pattern': 'KCL_{ema_period}_{atr_period}_{atr_multiplier:.1f}_{timeframe}', 'params_map': {'ema_period': 'kc_ema_period', 'atr_period': 'kc_atr_period', 'atr_multiplier': 'kc_atr_multiplier'}},
        }
    },
    'hv': {
       'func': calculate_hv_score,  # Historical Volatility评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'hv_period':20, 'hv_annual_factor': 252}, # 参数用于key_patterns
       'required_keys': ['hv'],
       'prefixes':['HV_'],
       'key_patterns':{'hv':{'pattern':'HV_{period}_{annual_factor}_{timeframe}', 'params_map':{'period':'hv_period', 'annual_factor': 'hv_annual_factor'}}}
    },
    'vroc': {
       'func': calculate_vroc_score,  # Volume ROC评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'vroc_period':10}, # 周期用于key_patterns
       'required_keys': ['vroc'],
       'prefixes':['VROC_'],
       'key_patterns':{'vroc':{'pattern':'VROC_{period}_{timeframe}', 'params_map':{'period':'vroc_period'}}}
    },
    'aroc': {
       'func': calculate_aroc_score,  # Amount ROC评分函数引用
       'param_passing_style': 'none',
       'bs_param_key_to_score_func_arg': {},
       'defaults': {'aroc_period':10}, # 周期用于key_patterns
       'required_keys': ['aroc'],
       'prefixes':['AROC_'],
       'key_patterns':{'aroc':{'pattern':'AROC_{period}_{timeframe}', 'params_map':{'period':'aroc_period'}}}
    },
    'pivot': {
       'func': calculate_pivot_score,  # Pivot Points评分函数引用
       'param_passing_style': 'dict', # 主要是为了pivot_levels (Dict) 和 close. 'tf' 由调用者处理.
       'bs_param_key_to_score_func_arg': {}, # 假设 'params: Optional[Dict]' 不从bs_params填充，或calculate_pivot_score不依赖它
       'defaults': {},
       'required_keys': ['close', 'pivot_levels'], # 'tf' 参数由调用逻辑特殊处理并传递，不在此处定义
       'prefixes': [], # Pivot levels列名在回退查找时有特殊处理逻辑，不依赖简单前缀
       'key_patterns': { # 'close'由OHLCV处理. 'pivot_levels'是特殊结构，回退查找时会查找naming_config中PIVOT的levels和pattern
            # 无需为 pivot_levels 的每个子键（R1, S1等）在此处定义 pattern，
            # 回退查找逻辑会使用 naming_config['indicator_naming_conventions']['PIVOT']['levels'] 和 ['pattern']
       }
    }
}


