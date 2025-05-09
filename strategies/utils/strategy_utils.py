# strategy_utils.py
from collections import defaultdict
import pandas as pd
import numpy as np
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

# --- 动态导入 pandas_ta ---
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    logger.warning("pandas_ta 库未安装或导入失败，依赖 pandas_ta 的功能将不可用。")

# --- 辅助函数区 ---

def _get_timeframe_in_minutes(tf_str: str) -> Optional[int]:
    """
    将时间级别字符串（如 '5', '15', 'D', 'W', 'M'）转换为近似的分钟数。
    此函数与 indicator_services 中的同名函数同步，确保一致性。
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

def get_find_peaks_params(time_level: str, base_lookback: int) -> Dict[str, Any]:
    """
    根据时间级别和基础回看期，返回适用于 find_peaks 的参数。
    短周期更敏感，长周期过滤更多噪音。
    """
    # 使用与 indicator_services 相似的分钟数转换以便比较
    tf_minutes = _get_timeframe_in_minutes(time_level)
    if tf_minutes is None:
        logger.warning(f"无法确定时间级别 {time_level} 的分钟数，使用默认 find_peaks 参数。")
        return {'distance': max(3, base_lookback // 3), 'prominence_factor': 0.3, 'width': max(1, base_lookback // 6)}

    # 根据分钟数调整基础参数
    # 例如，分钟数越小，distance 和 width 应该相对较小，prominence_factor 也可能需要调整
    # 这里提供一个更精细的映射，可以根据实际测试调整
    if tf_minutes <= 5: # 1-5分钟
        distance_factor = 4 # 距离较近
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
        distance_factor = 1.5 # 距离相对较大
        prominence_factor = 0.8 # 显著性要求更高
        width_factor = 1.5

    distance = max(2, int(base_lookback / distance_factor)) # 确保距离至少为2
    width = max(1, int(distance / width_factor)) # 宽度至少为1

    # 针对极短周期 (如1分钟)，确保参数不会过小
    if tf_minutes <= 5:
        distance = max(distance, 5) # 即使回看期很短，距离也至少5bar
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

    改进：
    1. 明确输入是 pd.Series。
    2. 优化峰谷匹配逻辑，确保匹配的指标点与价格点的时间顺序一致。
    3. 增加对输入 Series 有效性的更严格检查。
    4. 返回结果 DataFrame 只包含检测到的信号，0表示无信号。
    5. 信号标记在第二个价格极值点的位置。

    :param price_series: 价格序列 (通常是收盘价)。
    :param indicator_series: 指标序列。
    :param lookback: 查找峰值/谷值的回顾期。
    :param find_peaks_params: scipy.signal.find_peaks 的参数字典。
                              {'distance': int, 'prominence_factor': float, 'width': int}。
    :param check_regular_bullish: 是否检测常规看涨背离。
    :param check_regular_bearish: 是否检测常规看跌背离。
    :param check_hidden_bullish: 是否检测隐藏看涨背离。
    :param check_hidden_bearish: 是否检测隐藏看跌背ish。
    :return: 返回一个 DataFrame，包含四列 ('regular_bullish', 'regular_bearish',
             'hidden_bullish', 'hidden_bearish')。
             值为 1 表示检测到看涨背离，-1 表示检测到看跌背离，0 表示无。
             信号标记在第二个极值点的位置。
    """
    # 初始化结果 DataFrame，索引与输入 Series 一致
    result_df = pd.DataFrame({
        'regular_bullish': 0,
        'regular_bearish': 0,
        'hidden_bullish': 0,
        'hidden_bearish': 0
    }, index=price_series.index)

    # 数据有效性检查
    # 需要至少两个极值点才能形成背离，且需要考虑 lookback 窗口
    min_data_points = lookback * 2 if lookback > 0 else 30 # 至少需要一些数据点
    if price_series.isnull().all() or indicator_series.isnull().all() or len(price_series) < min_data_points:
        logger.debug(f"数据不足或全为 NaN (价格: {len(price_series)}, 指标: {len(indicator_series)}, 最小需要: {min_data_points})，无法检测背离。")
        return result_df

    # 确保输入是 Series 且索引一致
    if not isinstance(price_series, pd.Series) or not isinstance(indicator_series, pd.Series) or not price_series.index.equals(indicator_series.index):
         logger.error("价格序列或指标序列不是 pandas Series，或索引不一致。")
         return result_df

    # 填充指标序列中的 NaN 值，以便 find_peaks 可以处理
    # 注意：这里填充NaN可能会影响峰谷识别的准确性，特别是靠近数据边缘的部分。
    # 使用一个副本进行填充，避免修改原始 Series
    indicator_filled = indicator_series.ffill().bfill()
    if indicator_filled.isnull().all():
        logger.debug("填充后的指标序列全为 NaN，无法检测背离。")
        return result_df

    # --- 准备 find_peaks 参数 ---
    # 从传入的 params 中获取，并设置合理的默认值
    distance = find_peaks_params.get('distance', max(3, lookback // 3))
    width = find_peaks_params.get('width', max(1, distance // 2))
    prominence_factor = find_peaks_params.get('prominence_factor', 0.3)

    # 计算基于滚动标准差的最小显著性 (prominence)
    # 对价格和指标分别计算
    min_prominence_price_series = (price_series.rolling(lookback, min_periods=int(lookback*0.5)).std() * prominence_factor).fillna(0).replace([np.inf, -np.inf], 0)
    min_prominence_indicator_series = (indicator_filled.rolling(lookback, min_periods=int(lookback*0.5)).std() * prominence_factor).fillna(0).replace([np.inf, -np.inf], 0)

    # 确保 prominence 数组长度与序列一致，且为非负数，处理 NaN/Inf
    min_prominence_price = np.maximum(min_prominence_price_series.values, 1e-9) # 避免 prominence 为0
    min_prominence_indicator = np.maximum(min_prominence_indicator_series.values, 1e-9)


    # --- 查找价格和指标的峰值 (peaks) 和谷值 (troughs) ---
    # find_peaks 返回的是基于 Series 内部 numpy 数组的索引
    try:
        # 查找价格峰值
        price_peaks_indices, _ = find_peaks(price_series.values, distance=distance, prominence=min_prominence_price, width=width)
        # 查找指标峰值
        indicator_peaks_indices, _ = find_peaks(indicator_filled.values, distance=distance, prominence=min_prominence_indicator, width=width)

        # 查找价格谷值 (通过对序列取反实现)
        price_troughs_indices, _ = find_peaks(-price_series.values, distance=distance, prominence=min_prominence_price, width=width)
        # 查找指标谷值 (通过对序列取反实现)
        indicator_troughs_indices, _ = find_peaks(-indicator_filled.values, distance=distance, prominence=min_prominence_indicator, width=width)

    except Exception as fp_err:
        logger.warning(f"查找峰值/谷值时出错: {fp_err}。跳过此指标的背离检测。")
        return result_df

    # logger.debug(f"找到价格峰值: {len(price_peaks_indices)}, 谷值: {len(price_troughs_indices)}")
    # logger.debug(f"找到指标峰值: {len(indicator_peaks_indices)}, 谷值: {len(indicator_troughs_indices)}")


    # --- 背离检测逻辑 (改进的匹配和趋势判断) ---

    # 定义一个辅助函数来匹配价格极值点和指标极值点
    def find_matching_indicator_extremums(price_extremum_indices: np.ndarray, indicator_extremum_indices: np.ndarray, window: int) -> List[Tuple[int, Optional[int]]]:
        """
        为每个价格极值点，查找在指定窗口内时间上最接近的指标极值点。
        返回 (价格极值索引, 匹配的指标极值索引) 的列表。
        """
        matches = []
        for p_idx in price_extremum_indices:
            # 查找在 [p_idx - window, p_idx + window] 范围内的指标极值索引
            # 使用 np.searchsorted 会更高效，但需要指标索引是排序的
            # indicator_extremum_indices 是 find_peaks 返回的，已经是排序的
            lower_bound_idx = np.searchsorted(indicator_extremum_indices, p_idx - window, side='left')
            upper_bound_idx = np.searchsorted(indicator_extremum_indices, p_idx + window, side='right')
            nearby_indicator_indices = indicator_extremum_indices[lower_bound_idx:upper_bound_idx]

            if len(nearby_indicator_indices) > 0:
                # 找到窗口内距离最近的那个指标极值索引
                closest_i_idx = nearby_indicator_indices[np.abs(nearby_indicator_indices - p_idx).argmin()]
                matches.append((p_idx, closest_i_idx))
            else:
                matches.append((p_idx, None))
        return matches

    # 定义一个辅助函数来检查背离条件
    def check_divergence_pairs(price_matches: List[Tuple[int, Optional[int]]],
                               indicator_matches: List[Tuple[int, Optional[int]]], # Technically not needed, price_matches is enough
                               is_peak: bool, # True for peaks (bearish), False for troughs (bullish)
                               div_type: str # 'regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish'
                               ):
        """
        检查匹配对中的相邻两点是否构成指定类型的背离。
        """
        # 只需要价格匹配对，因为指标匹配对是基于价格点找到的
        # filtered_price_matches = [m for m in price_matches if m[1] is not None] # 只考虑有匹配指标的价顶点
        # 确保价格匹配对是按索引排序的
        sorted_price_matches = sorted([m for m in price_matches if m[1] is not None], key=lambda x: x[0])

        if len(sorted_price_matches) < 2:
            return # 需要至少两个匹配对来比较

        # 遍历相邻的匹配对 (p1_idx, i1_idx) 和 (p2_idx, i2_idx)
        # 其中 p1_idx < p2_idx
        for k in range(len(sorted_price_matches) - 1):
            p1_idx, i1_idx = sorted_price_matches[k]
            p2_idx, i2_idx = sorted_price_matches[k+1]

            # 必须确保指标点的顺序也一致 (i1_idx < i2_idx) 才能进行趋势比较
            # 且价格点和指标点必须都在 lookback 范围内 (相对于最后一个数据点)
            last_idx = len(price_series) - 1
            if i1_idx is None or i2_idx is None or i1_idx >= i2_idx or p2_idx < last_idx - lookback:
                continue # 跳过无效的匹配对或不在回看期内的对

            # 获取价格和指标值
            price1, price2 = price_series.iloc[p1_idx], price_series.iloc[p2_idx]
            indicator1, indicator2 = indicator_filled.iloc[i1_idx], indicator_filled.iloc[i2_idx]

            # 检查 NaN 值
            if pd.isna(price1) or pd.isna(price2) or pd.isna(indicator1) or pd.isna(indicator2):
                 continue

            # 根据 div_type 检查背离条件
            signal_value = 0
            if is_peak: # 看跌背离 (peaks)
                # 常规看跌: HH (价格) vs LH (指标)
                if div_type == 'regular_bearish' and price2 > price1 and indicator2 < indicator1:
                    signal_value = -1
                # 隐藏看跌: LH (价格) vs HH (指标)
                elif div_type == 'hidden_bearish' and price2 < price1 and indicator2 > indicator1:
                    signal_value = -1 # 隐藏背离信号值与常规相同，通过列名区分

            else: # 看涨背离 (troughs)
                # 常规看涨: LL (价格) vs HL (指标)
                if div_type == 'regular_bullish' and price2 < price1 and indicator2 > indicator1:
                    signal_value = 1
                # 隐藏看涨: HL (价格) vs LL (指标)
                elif div_type == 'hidden_bullish' and price2 > price1 and indicator2 < indicator1:
                    signal_value = 1 # 隐藏背离信号值与常规相同

            # 如果检测到信号，标记在第二个价格极值点的位置
            if signal_value != 0:
                 # 检查是否已经被更高优先级的信号标记
                 current_signal = result_df.loc[price_series.index[p2_idx], div_type]
                 if abs(signal_value) > abs(current_signal): # 如果当前信号优先级更高 (绝对值更大)
                      result_df.loc[price_series.index[p2_idx], div_type] = signal_value
                 elif current_signal == 0: # 如果当前位置没有信号
                      result_df.loc[price_series.index[p2_idx], div_type] = signal_value


    # 查找价格峰值对应的指标峰值
    price_peak_matches = find_matching_indicator_extremums(price_peaks_indices, indicator_peaks_indices, distance)
    # 查找价格谷值对应的指标谷值
    price_trough_matches = find_matching_indicator_extremums(price_troughs_indices, indicator_troughs_indices, distance)

    # 检查看跌背离 (峰值)
    if check_regular_bearish:
        check_divergence_pairs(price_peak_matches, indicator_peaks_indices, is_peak=True, div_type='regular_bearish')
    if check_hidden_bearish:
        check_divergence_pairs(price_peak_matches, indicator_peaks_indices, is_peak=True, div_type='hidden_bearish')

    # 检查看涨背离 (谷值)
    if check_regular_bullish:
        check_divergence_pairs(price_trough_matches, indicator_troughs_indices, is_peak=False, div_type='regular_bullish')
    if check_hidden_bullish:
        check_divergence_pairs(price_trough_matches, indicator_troughs_indices, is_peak=False, div_type='hidden_bullish')

    # 填充任何剩余的 NaN 值为 0
    result_df = result_df.fillna(0)
    return result_df.astype(int) # 确保返回整数

def detect_divergence(data: pd.DataFrame,
                      dd_params: Dict,
                      indicator_configs: List[Dict] # 传递在 indicator_services 中生成的指标配置列表
                      ) -> pd.DataFrame:
    """
    检测价格与多个指定指标之间的常规和隐藏背离。

    改进：
    1. 输入 DataFrame 已经包含所有时间级别和指标数据 (由 prepare_strategy_dataframe 生成)。
    2. 使用传递进来的 indicator_configs 来查找正确的指标列名。
    3. 允许在多个时间框架上检测背离 (根据 dd_params['tf'])。
    4. 聚合所有时间框架和指标的背离信号。

    :param data: 包含价格和指标列的 DataFrame (由 indicator_services.prepare_strategy_dataframe 生成)。
                   列名格式应为 {列名}_{时间级别}。
    :param dd_params: divergence_detection 参数字典, 包含:
                      'enabled': bool, 是否启用背离检测。
                      'timeframes': List[str], 用于检测的时间框架列表 (例如 ['15', '60'])。
                      'price_type': str, 用于比较的价格类型 ('close', 'high', 'low'), 默认为 'close'。
                      'lookback': int, 查找峰值/谷值的回顾期。
                      'find_peaks_params': dict, 传递给 find_peaks 的参数。
                      'check_regular_bullish', 'check_regular_bearish',
                      'check_hidden_bullish', 'check_hidden_bearish': bool, 控制检测类型。
                      'indicators': dict, 指定要检查的指标及其是否启用, 例如 {'macd_hist': True, 'rsi': True, 'mfi': False}。
    :param indicator_configs: 由 indicator_services.prepare_strategy_dataframe 生成的，
                              包含每个指标计算函数、参数、时间框架的列表。
    :return: 返回一个 DataFrame，包含详细的背离信号和聚合信号。
             详细信号列名示例: 'DIV_RSI_14_15_RegularBullish', 'DIV_MACDh_12_26_9_60_HiddenBearish'
             聚合信号列名: 'HAS_BULLISH_DIVERGENCE', 'HAS_BEARISH_DIVERGENCE' (布尔值)。
    """
    # 初始化结果 DataFrame，索引与输入数据一致
    all_divergence_signals = pd.DataFrame(index=data.index)
    # 默认没有信号
    all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = False
    all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = False

    # 检查是否启用背离检测
    if not dd_params.get('enabled', False):
        logger.info("参数中已禁用背离检测。")
        return all_divergence_signals

    # 获取配置参数
    timeframes_to_check = dd_params.get('timeframes', [])
    price_type = dd_params.get('price_type', 'close')
    lookback = dd_params.get('lookback', 14)
    # 获取通用 find_peaks 参数，并根据时间框架调整 (这里先获取通用参数，调整将在内部进行)
    base_find_peaks_params = dd_params.get('find_peaks_params', {})
    check_regular_bullish = dd_params.get('check_regular_bullish', True)
    check_regular_bearish = dd_params.get('check_regular_bearish', True)
    check_hidden_bullish = dd_params.get('check_hidden_bullish', True)
    check_hidden_bearish = dd_params.get('check_hidden_bearish', True)
    indicators_to_check = dd_params.get('indicators', {})

    if not timeframes_to_check:
        logger.warning("未指定用于背离检测的时间框架列表 (dd_params['timeframes'])。")
        return all_divergence_signals

    # 构建一个指标列名查找字典 { 'indicator_key': { 'tf': 'column_name', ... } }
    indicator_col_map = defaultdict(dict)
    for indi_conf in indicator_configs:
         base_name = indi_conf['name']
         for tf_conf in indi_conf['timeframes']:
              # 根据 base_name 和 tf_conf 查找对应的列名 (简单匹配，可能需要更精确的参数匹配)
              # 遍历 data.columns 查找以 base_name + 参数 + tf_conf 结尾的列
              # 这是一个简化的查找逻辑，假设列名格式固定
              # 更健壮的方式是从 indicator_services 获取完整的列名映射
              # 暂时使用简单的 endswith 匹配，并优先匹配参数最多的列
              matching_cols = [col for col in data.columns if col.startswith(base_name + '_') and col.endswith(f'_{tf_conf}')]
              if matching_cols:
                   # 如果有多个匹配（例如不同参数的EMA），选择最长名称的那个（通常参数越多列名越长）
                   best_match = max(matching_cols, key=len)
                   indicator_col_map[base_name][tf_conf] = best_match
                   # logger.debug(f"找到指标 '{base_name}' 在 TF '{tf_conf}' 的列名: '{best_match}'")
              # 特殊处理 MACDh，其 base_name 是 MACD，但列名是 MACDh_...
              elif base_name == 'MACD':
                   matching_macd_hist = [col for col in data.columns if col.startswith('MACDh_') and col.endswith(f'_{tf_conf}')]
                   if matching_macd_hist:
                        best_match_hist = max(matching_macd_hist, key=len)
                        indicator_col_map['macd_hist'][tf_conf] = best_match_hist
                        # logger.debug(f"找到指标 'macd_hist' 在 TF '{tf_conf}' 的列名: '{best_match_hist}'")


    # 遍历配置中指定要检查的指标和时间框架
    for indicator_key, enabled in indicators_to_check.items():
        if not enabled:
            continue # 跳过未启用的指标

        # 查找该 indicator_key 对应的实际指标名 (如果 key 和 name 不同，例如 'macd_hist' -> 'MACDh')
        # 这里的映射需要与 indicator_configs 或 calculate_* 函数的逻辑对应
        actual_indicator_name_prefix = None
        if indicator_key == 'macd_hist': actual_indicator_name_prefix = 'MACDh'
        elif indicator_key == 'rsi': actual_indicator_name_prefix = 'RSI'
        elif indicator_key == 'mfi': actual_indicator_name_prefix = 'MFI'
        elif indicator_key == 'obv': actual_indicator_name_prefix = 'OBV'
        elif indicator_key == 'cci': actual_indicator_name_prefix = 'CCI'
        elif indicator_key == 'cmf': actual_indicator_name_prefix = 'CMF'
        elif indicator_key == 'stoch_k': actual_indicator_name_prefix = 'STOCHk' # K 线
        elif indicator_key == 'stoch_d': actual_indicator_name_prefix = 'STOCHd' # D 线
        elif indicator_key == 'stoch_j': actual_indicator_name_prefix = 'J'      # J 线 (KDJ的一部分)
        elif indicator_key == 'roc': actual_indicator_name_prefix = 'ROC'
        elif indicator_key == 'adx': actual_indicator_name_prefix = 'ADX'
        elif indicator_key == 'pdi': actual_indicator_name_prefix = 'PDI' # DMI +
        elif indicator_key == 'ndi': actual_indicator_name_prefix = 'NDI' # DMI -
        elif indicator_key == 'bbp': actual_indicator_name_prefix = 'BBP' # %B
        # 添加其他需要检测背离的指标...
        else:
            logger.warning(f"未知的指标 key '{indicator_key}' 用于背离检测。")
            continue

        if actual_indicator_name_prefix is None:
             logger.warning(f"无法将指标 key '{indicator_key}' 映射到实际列名前缀。")
             continue

        for tf_check in timeframes_to_check:
            # 构建价格列名并检查是否存在
            price_col = f'{price_type}_{tf_check}' # 例如 'close_15'
            if price_col not in data.columns or data[price_col].isnull().all():
                logger.warning(f"用于背离检测的时间框架 {tf_check} 的价格列 '{price_col}' 不存在或全为 NaN。跳过此时间框架。")
                continue
            price_series = data[price_col]

            # 根据 actual_indicator_name_prefix 和 tf_check 查找具体的指标列名
            # 这里直接使用构建的 map 查找，更安全
            indicator_col = None
            # 尝试在 map 中查找精确匹配或以 actual_indicator_name_prefix 开头且以 tf_check 结尾的列
            # 例如，如果 key 是 'rsi'，prefix 是 'RSI'，tf 是 '15'，我们找 RSI_14_15
            # indicator_col_map[indicator_key][tf_check] 可能更直接，但需要确保 map 构建正确
            if indicator_key in indicator_col_map and tf_check in indicator_col_map[indicator_key]:
                 indicator_col = indicator_col_map[indicator_key][tf_check]
            elif actual_indicator_name_prefix in indicator_col_map and tf_check in indicator_col_map[actual_indicator_name_prefix]: # 尝试用 prefix 查找
                 indicator_col = indicator_col_map[actual_indicator_name_prefix][tf_check]
            else:
                # 尝试在 data.columns 中查找以 prefix 开头且以 tf_check 结尾的列
                potential_cols = [col for col in data.columns if col.startswith(actual_indicator_name_prefix + '_') and col.endswith(f'_{tf_check}')]
                if potential_cols:
                     # 如果有多个匹配，选择最长名称的那个（通常参数越多列名越长）
                    indicator_col = max(potential_cols, key=len)
                elif actual_indicator_name_prefix == 'OBV' and f'OBV_{tf_check}' in data.columns: # OBV可能没有参数
                    indicator_col = f'OBV_{tf_check}'

            if indicator_col is None or indicator_col not in data.columns or data[indicator_col].isnull().all():
                logger.warning(f"指标 '{indicator_key}' (前缀: {actual_indicator_name_prefix}) 在时间框架 {tf_check} 的列 '{indicator_col}' 不存在、全为 NaN 或未启用。跳过其背离检测。")
                continue

            indicator_series = data[indicator_col]

            # 获取针对当前时间框架调整过的 find_peaks 参数
            current_find_peaks_params = get_find_peaks_params(tf_check, lookback)
            # 用 dd_params 中指定的参数覆盖通用参数
            for pk, pv in base_find_peaks_params.items():
                 current_find_peaks_params[pk] = pv

            # --- 调用辅助函数进行单指标背离检测 ---
            logger.debug(f"开始检测 TF {tf_check}: 价格 ('{price_col}') 与指标 ('{indicator_col}') 的背离...")
            div_result = find_divergence_for_indicator(
                price_series=price_series,
                indicator_series=indicator_series,
                lookback=lookback,
                find_peaks_params=current_find_peaks_params,
                check_regular_bullish=check_regular_bullish,
                check_regular_bearish=check_regular_bearish,
                check_hidden_bullish=check_hidden_bullish,
                check_hidden_bearish=check_hidden_bearish
            )

            # 将检测结果合并到总的 DataFrame 中，并添加前缀和详细列名
            for div_type in div_result.columns:
                 # 详细列名示例: DIV_RSI_14_15_RegularBullish
                 # 从 indicator_col 中提取指标基础名和参数部分
                 col_parts = indicator_col.split('_')
                 # 假设列名格式是 NAME_PARAM1_PARAM2..._TF
                 indicator_base_name_from_col = col_parts[0]
                 indicator_params_from_col = "_".join(col_parts[1:-1]) # 参数部分
                 if indicator_params_from_col:
                      detailed_col_name = f'DIV_{indicator_base_name_from_col}_{indicator_params_from_col}_{tf_check}_{div_type.replace("_", "")}' # 例如: DIV_RSI_14_15_regularbullish
                 else: # OBV 等可能没有参数
                      detailed_col_name = f'DIV_{indicator_base_name_from_col}_{tf_check}_{div_type.replace("_", "")}' # 例如: DIV_OBV_15_regularbullish

                 all_divergence_signals[detailed_col_name.upper()] = div_result[div_type]

    # --- 聚合所有时间框架和指标的看涨和看跌信号 ---
    # 查找所有包含 'BULLISH' 且值 > 0 的列 (不区分大小写匹配)
    bullish_cols = [col for col in all_divergence_signals.columns if 'BULLISH' in col.upper()]
    if bullish_cols:
        # 使用 .any(axis=1) 会将整行的 True/False 合并，但我们希望保留信号强度
        # 改为查找任一列 > 0 即为 True
        all_divergence_signals['HAS_BULLISH_DIVERGENCE'] = (all_divergence_signals[bullish_cols] > 0).any(axis=1)

    # 查找所有包含 'BEARISH' 且值 < 0 的列 (不区分大小写匹配)
    bearish_cols = [col for col in all_divergence_signals.columns if 'BEARISH' in col.upper()]
    if bearish_cols:
        all_divergence_signals['HAS_BEARISH_DIVERGENCE'] = (all_divergence_signals[bearish_cols] < 0).any(axis=1)

    # 最终填充 NaN 为 False 或 0
    for col in all_divergence_signals.columns:
        if all_divergence_signals[col].dtype == 'bool':
            all_divergence_signals[col].fillna(False, inplace=True)
        else:
            all_divergence_signals[col].fillna(0, inplace=True)

    logger.info(f"背离检测完成。共生成 {len(all_divergence_signals.columns) - 2} 个详细信号列。")
    return all_divergence_signals

def detect_kline_patterns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    检测 K 线形态，使用指定时间框架的 OHLC 数据。

    信号值 (列名 KAP_{PatternName}_{TF}, 值为 1/-1 表示看涨/看跌):
     KAP_BullishEngulfing_{TF} (1)
     KAP_BearishEngulfing_{TF} (-1)
     KAP_Hammer_{TF} (1)
     KAP_HangingMan_{TF} (-1)
     KAP_MorningStar_{TF} (1)
     KAP_EveningStar_{TF} (-1)
     KAP_PiercingLine_{TF} (1)
     KAP_DarkCloudCover_{TF} (-1)
     KAP_Doji_{TF} (1/-1, 仅标记)
     KAP_ThreeWhiteSoldiers_{TF} (1)
     KAP_ThreeBlackCrows_{TF} (-1)
     KAP_BullishHarami_{TF} (1)
     KAP_BearishHarami_{TF} (-1)
     KAP_BullishHaramiCross_{TF} (1)
     KAP_BearishHaramiCross_{TF} (-1)
     KAP_TweezerBottom_{TF} (1)
     KAP_TweezerTop_{TF} (-1)
     KAP_BullishMarubozu_{TF} (1)
     KAP_BearishMarubozu_{TF} (-1)
     KAP_RisingThreeMethods_{TF} (1)
     KAP_FallingThreeMethods_{TF} (-1)
     KAP_UpsideGapTwoCrows_{TF} (-1)
     KAP_UpsideTasukiGap_{TF} (1)
     KAP_DownsideTasukiGap_{TF} (-1)
     KAP_BullishSeparatingLines_{TF} (1)
     KAP_BearishSeparatingLines_{TF} (-1)
     KAP_BullishCounterattack_{TF} (1)
     KAP_BearishCounterattack_{TF} (-1)

    返回 DataFrame，每列代表一个形态，值为 1/-1 表示检测到该形态，0 表示无。
    """
    required_cols_base = ['open', 'high', 'low', 'close', 'volume'] # Volume might be needed for filtering later
    required_cols_tf = [f"{col}_{tf}" for col in required_cols_base]

    if not all(col in df.columns for col in required_cols_tf):
        logger.warning(f"K-line pattern detection for TF {tf} requires columns: {required_cols_tf}. Skipping.")
        # 返回一个空的或者默认的 DataFrame
        result_df = pd.DataFrame(index=df.index)
        # 添加所有可能的形态列，并填充0
        pattern_names = [
            'BullishEngulfing', 'BearishEngulfing', 'Hammer', 'HangingMan',
            'MorningStar', 'EveningStar', 'PiercingLine', 'DarkCloudCover', 'Doji',
            'ThreeWhiteSoldiers', 'ThreeBlackCrows', 'BullishHarami', 'BearishHarami',
            'BullishHaramiCross', 'BearishHaramiCross', 'TweezerBottom', 'TweezerTop',
            'BullishMarubozu', 'BearishMarubozu', 'RisingThreeMethods', 'FallingThreeMethods',
            'UpsideGapTwoCrows', 'UpsideTasukiGap', 'DownsideTasukiGap',
            'BullishSeparatingLines', 'BearishSeparatingLines', 'BullishCounterattack', 'BearishCounterattack'
        ]
        for name in pattern_names:
             result_df[f'KAP_{name.upper()}_{tf}'] = 0
        return result_df


    # 使用指定时间框架的列
    o = df[f'open_{tf}']
    h = df[f'high_{tf}']
    l = df[f'low_{tf}']
    c = df[f'close_{tf}']
    v = df[f'volume_{tf}'] # 成交量，可能用于过滤低量 K 线

    # 过滤掉 OHLC 有 NaN 的行
    df_filtered = df[[f'open_{tf}', f'high_{tf}', f'low_{tf}', f'close_{tf}', f'volume_{tf}']].dropna()
    if len(df_filtered) < 5:
         logger.warning(f"TF {tf} 数据点不足 (<5)，无法检测所有 K 线形态。当前数据量: {len(df_filtered)}")

    # Re-assign series from filtered data
    o_f = df_filtered[f'open_{tf}']
    h_f = df_filtered[f'high_{tf}']
    l_f = df_filtered[f'low_{tf}']
    c_f = df_filtered[f'close_{tf}']
    v_f = df_filtered[f'volume_{tf}']

    # Shifted series (relative to the filtered data index)
    o1, h1, l1, c1, v1 = o_f.shift(1), h_f.shift(1), l_f.shift(1), c_f.shift(1), v_f.shift(1)
    o2, h2, l2, c2, v2 = o_f.shift(2), h_f.shift(2), l_f.shift(2), c_f.shift(2), v_f.shift(2)
    o3, h3, l3, c3 = o_f.shift(3), h_f.shift(3), l_f.shift(3), c_f.shift(3)
    o4, h4, l4, c4 = o_f.shift(4), h_f.shift(4), l_f.shift(4), c_f.shift(4)

    body = abs(c_f - o_f)
    body1 = abs(c1 - o1).fillna(0)
    body2 = abs(c2 - o2).fillna(0)
    body3 = abs(c3 - o3).fillna(0)
    body4 = abs(c4 - o4).fillna(0)

    full_range = (h_f - l_f).replace(0, 1e-9)
    full_range1 = (h1 - l1).fillna(0).replace(0, 1e-9)
    full_range2 = (h2 - l2).fillna(0).replace(0, 1e-9)
    full_range3 = (h3 - l3).fillna(0).replace(0, 1e-9)
    full_range4 = (h4 - l4).fillna(0).replace(0, 1e-9)

    upper_shadow = h_f - np.maximum(c_f, o_f)
    lower_shadow = np.minimum(o_f, c_f) - l_f
    upper_shadow1 = h1 - np.maximum(c1, o1)
    lower_shadow1 = np.minimum(o1, c1) - l1

    is_green = c_f > o_f
    is_red = c_f < o_f
    is_green1 = c1 > o1
    is_red1 = c1 < o1
    is_green2 = c2 > o2
    is_red2 = c2 < o2
    is_green3 = c3 > o3
    is_red3 = c3 < o3
    is_green4 = c4 > o4
    is_red4 = c4 < o4

    # 使用过滤后的数据计算平均实体和平均范围
    avg_body = body.rolling(min(len(body), 20), min_periods=max(1, int(len(body)*0.3))).mean().fillna(1e-9) # 至少需要30%数据计算平均
    avg_range = full_range.rolling(min(len(full_range), 20), min_periods=max(1, int(len(full_range)*0.3))).mean().fillna(1e-9)

    # 初始化结果 DataFrame，使用过滤后的索引，并填充0
    patterns_filtered = pd.DataFrame(0, index=df_filtered.index, columns=[
        f'KAP_BULLISHENGULFING_{tf}', f'KAP_BEARISHENGULFING_{tf}',
        f'KAP_HAMMER_{tf}', f'KAP_HANGINGMAN_{tf}',
        f'KAP_MORNINGSTAR_{tf}', f'KAP_EVENINGSTAR_{tf}',
        f'KAP_PIERCINGLINE_{tf}', f'KAP_DARKCLOUDCOVER_{tf}',
        f'KAP_DOJI_{tf}',
        f'KAP_THREEWHITESOLDIERS_{tf}', f'KAP_THREEBLACKCROWS_{tf}',
        f'KAP_BULLISHHARAMI_{tf}', f'KAP_BEARISHHARAMI_{tf}',
        f'KAP_BULLISHHARAMICROSS_{tf}', f'KAP_BEARISHHARAMICROSS_{tf}',
        f'KAP_TWEEZERBOTTOM_{tf}', f'KAP_TWEEZERTOP_{tf}',
        f'KAP_BULLISHMARUBOZU_{tf}', f'KAP_BEARISHMARUBOZU_{tf}',
        f'KAP_RISINGTHREEMETHODS_{tf}', f'KAP_FALLINGTHREEMETHODS_{tf}',
        f'KAP_UPSIDEGAPTWOCROWS_{tf}', f'KAP_UPSIDETASUKIGAP_{tf}',
        f'KAP_DOWNSIDETASUKIGAP_{tf}', f'KAP_BULLISHSEPARATINGLINES_{tf}',
        f'KAP_BEARISHSEPARATINGLINES_{tf}', f'KAP_BULLISHCOUNTERATTACK_{tf}',
        f'KAP_BEARISHCOUNTERATTACK_{tf}'
    ])

    # --- 十字星 (Doji) ---
    doji_threshold_factor = 0.1
    is_doji_f = (body <= avg_range * doji_threshold_factor) & (body > avg_body * 0.1) # 实体不能是零，且小于平均波幅的10%
    # 直接标记，不赋优先级，后面再根据优先级覆盖
    patterns_filtered.loc[is_doji_f[is_doji_f].index, f'KAP_DOJI_{tf}'] = np.where(is_green[is_doji_f], 1, -1) # 阳十字/阴十字区分一下？或者只标记为1，表示存在

    # --- 吞没 (Engulfing) ---
    # 看涨吞没: 前阴，后阳，后阳实体完全包住前阴实体 (影线不计)
    bull_engulf = is_red1 & is_green & (c_f > o1) & (o_f < c1) & (body > body1 * 1.01) # 后实体比前实体大1%
    patterns_filtered.loc[bull_engulf[bull_engulf].index, f'KAP_BULLISHENGULFING_{tf}'] = 1
    # 看跌吞没: 前阳，后阴，后阴实体完全包住前阳实体
    bear_engulf = is_green1 & is_red & (o_f > c1) & (c_f < o1) & (body > body1 * 1.01)
    patterns_filtered.loc[bear_engulf[bear_engulf].index, f'KAP_BEARISHENGULFING_{tf}'] = -1

    # --- 锤子/上吊 (Hammer/Hanging Man) ---
    # 小实体，长下影，几乎无上影
    small_body_threshold = avg_range * 0.3
    long_lower_shadow_cond = lower_shadow >= 2 * body
    short_upper_shadow_cond = upper_shadow < body * 0.5 # 上影线小于实体一半
    hammer_like = (body > 1e-9) & (body < small_body_threshold) & long_lower_shadow_cond & short_upper_shadow_cond
    # 锤子：出现在下跌趋势后
    # 上吊线：出现在上涨趋势后 (简单的用前一根K线颜色或收盘价关系判断)
    is_hammer = hammer_like & is_red1 # 前一根是阴线
    patterns_filtered.loc[is_hammer[is_hammer].index, f'KAP_HAMMER_{tf}'] = 1
    is_hanging = hammer_like & is_green1 # 前一根是阳线
    patterns_filtered.loc[is_hanging[is_hanging].index, f'KAP_HANGINGMAN_{tf}'] = -1

    # --- 星线 (Morning/Evening Star) ---
    star_body_threshold = avg_range * 0.3 # 中继实体不超过平均波幅30%
    is_star1_f = (body1 < star_body_threshold) & (body1 > 1e-9)
    # 第一根与第二根之间跳空
    gap1_down = (np.minimum(o1, c1) < c2) if is_red2.any() else False # 前前阴，前一根开收低于前前收
    gap1_up = (np.maximum(o1, c1) > c2) if is_green2.any() else False # 前前阳，前一根开收高于前前收
    # 第二根与第三根之间跳空 (可选，非必须)
    # gap2_down = np.minimum(o_f, c_f) < np.maximum(o1, c1)
    # gap2_up = np.maximum(o_f, c_f) > np.minimum(o1, c1)

    # 早晨之星: 前大阴 + 跳空小实体 + 后大阳 (后阳收盘深入前阴实体一半以上)
    morning_star = is_red2 & (body2 > avg_body) & is_star1_f & is_green & (body > body2 * 0.5) & (c_f > (o2 + c2) / 2) # & gap1_down
    patterns_filtered.loc[morning_star[morning_star].index, f'KAP_MORNINGSTAR_{tf}'] = 1
    # 黄昏之星: 前大阳 + 跳空小实体 + 后大阴 (后阴收盘深入前阳实体一半以下)
    evening_star = is_green2 & (body2 > avg_body) & is_star1_f & is_red & (body > body2 * 0.5) & (c_f < (o2 + c2) / 2) # & gap1_up
    patterns_filtered.loc[evening_star[evening_star].index, f'KAP_EVENINGSTAR_{tf}'] = -1

    # --- 刺透线/乌云盖顶 (Piercing/Dark Cloud) --- (两根 K 线形态)
    # 刺透线: 前大阴 + 后大阳跳空低开但收盘进入前阴实体一半以上
    piercing = is_red1 & (body1 > avg_body) & is_green & (o_f < l1) & (c_f > (o1 + c1) / 2) & (c_f < o1)
    patterns_filtered.loc[piercing[piercing].index, f'KAP_PIERCINGLINE_{tf}'] = 1
    # 乌云盖顶: 前大阳 + 后大阴跳空高开但收盘进入前阳实体一半以下
    dark_cloud = is_green1 & (body1 > avg_body) & is_red & (o_f > h1) & (c_f < (o1 + c1) / 2) & (c_f > o1)
    patterns_filtered.loc[dark_cloud[dark_cloud].index, f'KAP_DARKCLOUDCOVER_{tf}'] = -1

    # --- 三兵/三鸦 (Three Soldiers/Crows) --- (三根 K 线形态)
    # 红三兵: 连续三阳，逐步抬高，开盘在前实体内，收盘不断创新高，实体不能太小
    soldiers = is_green2 & (body2 > avg_body * 0.5) & \
               is_green1 & (body1 > avg_body * 0.5) & (c1 > c2) & (o1 < c2) & (o1 > o2) & \
               is_green & (body > avg_body * 0.5) & (c_f > c1) & (o_f < c1) & (o_f > o1)
    patterns_filtered.loc[soldiers[soldiers].index, f'KAP_THREEWHITESOLDIERS_{tf}'] = 1
    # 三只乌鸦: 连续三阴，逐步降低，开盘在前实体内，收盘不断创新低，实体不能太小
    crows = is_red2 & (body2 > avg_body * 0.5) & \
            is_red1 & (body1 > avg_body * 0.5) & (c1 < c2) & (o1 > c2) & (o1 < o2) & \
            is_red & (body > avg_body * 0.5) & (c_f < c1) & (o_f > c1) & (o_f < o1)
    patterns_filtered.loc[crows[crows].index, f'KAP_THREEBLACKCROWS_{tf}'] = -1

    # --- 孕线 (Harami) --- (两根 K 线形态)
    # 孕线体条件：后一根 K 线的实体完全被前一根 K 线实体包含
    is_harami_body = (np.maximum(o_f, c_f) < np.maximum(o1, c1)) & (np.minimum(o_f, c_f) > np.minimum(o1, c1))
    # 看涨孕线: 前大阴 + 后小实体 (颜色相反或十字)
    bullish_harami = is_red1 & (body1 > avg_body) & is_harami_body & (is_green | is_doji_f)
    patterns_filtered.loc[bullish_harami[bullish_harami].index, f'KAP_BULLISHHARAMI_{tf}'] = 1
    # 看跌孕线: 前大阳 + 后小实体 (颜色相反或十字)
    bearish_harami = is_green1 & (body1 > avg_body) & is_harami_body & (is_red | is_doji_f)
    patterns_filtered.loc[bearish_harami[bearish_harami].index, f'KAP_BEARISHHARAMI_{tf}'] = -1
    # --- 十字孕线 (Harami Cross) --- (孕线体的后一根是十字星)
    bullish_harami_cross = is_red1 & (body1 > avg_body) & is_harami_body & is_doji_f
    patterns_filtered.loc[bullish_harami_cross[bullish_harami_cross].index, f'KAP_BULLISHHARAMICROSS_{tf}'] = 1
    bearish_harami_cross = is_green1 & (body1 > avg_body) & is_harami_body & is_doji_f
    patterns_filtered.loc[bearish_harami_cross[bearish_harami_cross].index, f'KAP_BEARISHHARAMICROSS_{tf}'] = -1

    # --- 镊子顶/底 (Tweezers Top/Bottom) --- (两根 K 线形态)
    # 条件：最近两根 K 线的最高点或最低点几乎相同
    tweezer_tolerance = avg_range * 0.02 # 2% 的平均波幅容忍度
    # 镊子底: 两根 K 线的最低点接近，通常出现在下跌趋势后
    tweezer_bottom = (abs(l_f - l1) < tweezer_tolerance) # & (is_red1) # 可选：前一根是阴线
    patterns_filtered.loc[tweezer_bottom[tweezer_bottom].index, f'KAP_TWEEZERBOTTOM_{tf}'] = 1
    # 镊子顶: 两根 K 线的最高点接近，通常出现在上涨趋势后
    tweezer_top = (abs(h_f - h1) < tweezer_tolerance) # & (is_green1) # 可选：前一根是阳线
    patterns_filtered.loc[tweezer_top[tweezer_top].index, f'KAP_TWEEZERTOP_{tf}'] = -1

    # --- 光头光脚 (Marubozu) --- (单根 K 线形态)
    # 几乎没有影线，实体非常饱满
    shadow_threshold_factor = 0.05 # 影线小于实体或全范围的 5%
    no_upper_shadow_cond = upper_shadow < np.maximum(body, full_range * shadow_threshold_factor)
    no_lower_shadow_cond = lower_shadow < np.maximum(body, full_range * shadow_threshold_factor)
    # 确保实体占全范围的大部分
    is_marubozu_cond = (body > full_range * 0.9) & (body > avg_body * 1.5) # 实体大，占全范围比例高

    bull_marubozu = is_marubozu_cond & is_green & no_upper_shadow_cond & no_lower_shadow_cond
    patterns_filtered.loc[bull_marubozu[bull_marubozu].index, f'KAP_BULLISHMARUBOZU_{tf}'] = 1
    bear_marubozu = is_marubozu_cond & is_red & no_upper_shadow_cond & no_lower_shadow_cond
    patterns_filtered.loc[bear_marubozu[bear_marubozu].index, f'KAP_BEARISHMARUBOZU_{tf}'] = -1

    # --- 上升/下降三法 (Rising/Falling Three Methods) --- (五根 K 线形态)
    # 需要至少5根数据
    if len(df_filtered) >= 5:
        # 上升三法: 长阳(4) + 三个小阴/整理(1,2,3 在阳线4实体内) + 长阳(0 收盘高于4收盘)
        rising_three = is_green4 & (body4 > avg_body * 1.5) & \
                       (is_red3 | body3 < avg_body * 0.5) & (h3 < h4) & (l3 > l4) & \
                       (is_red2 | body2 < avg_body * 0.5) & (h2 < h4) & (l2 > l4) & \
                       (is_red1 | body1 < avg_body * 0.5) & (h1 < h4) & (l1 > l4) & \
                       is_green & (body > avg_body * 1.5) & (c_f > c4) & (o_f > l1) # 第五根阳线开盘高于前一根低点
        patterns_filtered.loc[rising_three[rising_three].index, f'KAP_RISINGTHREEMETHODS_{tf}'] = 1
        # 下降三法: 长阴(4) + 三个小阳/整理(1,2,3 在阴线4实体内) + 长阴(0 收盘低于4收盘)
        falling_three = is_red4 & (body4 > avg_body * 1.5) & \
                        (is_green3 | body3 < avg_body * 0.5) & (h3 < h4) & (l3 > l4) & \
                        (is_green2 | body2 < avg_body * 0.5) & (h2 < h4) & (l2 > l4) & \
                        (is_green1 | body1 < avg_body * 0.5) & (h1 < h4) & (l1 > l4) & \
                        is_red & (body > avg_body * 1.5) & (c_f < c4) & (o_f < h1) # 第五根阴线开盘低于前一根高点
        patterns_filtered.loc[falling_three[falling_three].index, f'KAP_FALLINGTHREEMETHODS_{tf}'] = -1

    # --- 向上跳空两只乌鸦 (Upside Gap Two Crows) --- (三根 K 线形态)
    # 需要至少3根数据
    if len(df_filtered) >= 3:
        # 条件: 强阳线(2) + 向上跳空的小阴线(1) + 更大的阴线(0 开盘高于前阴，收盘低于前阴，且吞没前阴)
        upside_gap_two_crows = is_green2 & (body2 > avg_body) & \
                              is_red1 & (body1 < avg_body * 0.7) & (o1 > h2) & \
                              is_red & (o_f > o1) & (c_f < c1) & (c_f < h2) # 实体吞没小阴线，收盘低于第一天高点
        patterns_filtered.loc[upside_gap_two_crows[upside_gap_two_crows].index, f'KAP_UPSIDEGAPTWOCROWS_{tf}'] = -1

        # --- 跳空并列线 (Tasuki Gap) --- (三根 K 线形态)
        # 向上跳空并列阳线: 阳线(2) + 向上跳空阳线(1) + 阴线(0 开盘在前阳实体，收盘在缺口内) - 看涨持续
        upside_tasuki_gap = is_green2 & \
                           is_green1 & (o1 > h2) & \
                           is_red & (o_f > o1) & (o_f < c1) & (c_f < o1) & (c_f > h2) # 收盘填补部分缺口
        patterns_filtered.loc[upside_tasuki_gap[upside_tasuki_gap].index, f'KAP_UPSIDETASUKIGAP_{tf}'] = 1
        # 向下跳空并列阴线: 阴线(2) + 向下跳空阴线(1) + 阳线(0 开盘在前阴实体，收盘在缺口内) - 看跌持续
        downside_tasuki_gap = is_red2 & \
                             is_red1 & (o1 < l2) & \
                             is_green & (o_f < o1) & (o_f > c1) & (c_f > o1) & (c_f < l2) # 收盘填补部分缺口
        patterns_filtered.loc[downside_tasuki_gap[downside_tasuki_gap].index, f'KAP_DOWNSIDETASUKIGAP_{tf}'] = -1

        # --- 分离线 (Separating Lines) --- (两根 K 线形态)
        # 条件：颜色相反，开盘价相同
        same_open_cond = abs(o_f - o1) < avg_range * 0.02 # 开盘价几乎相同
        # 看涨分离线: 下降趋势中(前阴)，前阴后阳，开盘相同
        bullish_sep_lines = is_red1 & is_green & same_open_cond
        patterns_filtered.loc[bullish_sep_lines[bullish_sep_lines].index, f'KAP_BULLISHSEPARATINGLINES_{tf}'] = 1
        # 看跌分离线: 上升趋势中(前阳)，前阳后阴，开盘相同
        bearish_sep_lines = is_green1 & is_red & same_open_cond
        patterns_filtered.loc[bearish_sep_lines[bearish_sep_lines].index, f'KAP_BEARISHSEPARATINGLINES_{tf}'] = -1

        # --- 反击线 (Counterattack Lines) --- (两根 K 线形态)
        # 条件：颜色相反，收盘价相同
        same_close_cond = abs(c_f - c1) < avg_range * 0.02 # 收盘价几乎相同
        # 看涨反击线: 下降趋势中(前长阴)，前长阴后长阳(大幅跳空低开)，收盘相同
        bullish_counter = is_red1 & (body1 > avg_body) & is_green & (body > avg_body) & same_close_cond & (o_f < l1) # 跳空低开
        patterns_filtered.loc[bullish_counter[bullish_counter].index, f'KAP_BULLISHCOUNTERATTACK_{tf}'] = 1
        # 看跌反击线: 上升趋势中(前长阳)，前长阳后长阴(大幅跳空高开)，收盘相同
        bearish_counter = is_green1 & (body1 > avg_body) & is_red & (body > avg_body) & same_close_cond & (o_f > h1) # 跳空高开
        patterns_filtered.loc[bearish_counter[bearish_counter].index, f'KAP_BEARISHCOUNTERATTACK_{tf}'] = -1


    # 将结果 DataFrame 重新对齐到原始输入的 df 索引，并用 0 填充 NaT/NaN
    patterns_aligned = pd.DataFrame(0, index=df.index, columns=patterns_filtered.columns)
    patterns_aligned.update(patterns_filtered) # 使用 update 方法将计算结果合并进来
    patterns_aligned = patterns_aligned.fillna(0).astype(int)

    logger.info(f"TF {tf} K 线形态检测完成。")
    return patterns_aligned

# --- 指标评分函数 (原 _get_xxx_score 改为公用) ---

def calculate_macd_score(macd_series: pd.Series, macd_d: pd.Series, macd_h: pd.Series) -> pd.Series:
    """
    MACD 评分 (0-100)。
    接收 MACD 线 (diff), MACD DEA (signal), MACD Hist (diff-dea) 三条序列。
    评分逻辑基于金叉死叉、零轴位置、柱状图变化等。

    Args:
        macd_series (pd.Series): MACD 线 (DIFF)。
        macd_d (pd.Series): MACD DEA 线 (SIGNAL)。
        macd_h (pd.Series): MACD 柱状图 (HIST)。

    Returns:
        pd.Series: 计算出的 MACD 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=macd_series.index)

    # 确保输入序列长度一致
    if not macd_series.index.equals(macd_d.index) or not macd_series.index.equals(macd_h.index):
        logger.warning("MACD 评分输入序列索引不一致，可能导致计算错误。")
        # 尝试重新索引，用 NaN 填充缺失值，然后填充 50.0
        index = macd_series.index.union(macd_d.index).union(macd_h.index)
        macd_series = macd_series.reindex(index).fillna(50.0)
        macd_d = macd_d.reindex(index).fillna(50.0)
        macd_h = macd_h.reindex(index).fillna(0.0) # MACDh 零轴是关键，NaN填充0
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    macd_series_filled = macd_series.fillna(50.0)
    macd_d_filled = macd_d.fillna(50.0)
    macd_h_filled = macd_h.fillna(0.0)

    # 金叉: MACD 线 (diff) 上穿 DEA 线
    # 使用 shift(1) 比较前一个周期
    buy_cross = (macd_series_filled.shift(1) < macd_d_filled.shift(1)) & (macd_series_filled >= macd_d_filled)
    # 死叉: MACD 线 (diff) 下穿 DEA 线
    sell_cross = (macd_series_filled.shift(1) > macd_d_filled.shift(1)) & (macd_series_filled <= macd_d_filled)

    # 零轴附近金叉死叉加强信号
    # 零轴上方金叉 (看涨信号加强)
    buy_cross_above_zero = buy_cross & (macd_series_filled > 0)
    # 零轴下方金叉 (看涨信号减弱，但仍是金叉)
    buy_cross_below_zero = buy_cross & (macd_series_filled <= 0) # 包含零轴上金叉

    # 零轴上方死叉 (看跌信号减弱，但仍是死叉)
    sell_cross_above_zero = sell_cross & (macd_series_filled >= 0) # 包含零轴上死叉
    # 零轴下方死叉 (看跌信号加强)
    sell_cross_below_zero = sell_cross & (macd_series_filled < 0)


    # 应用交叉信号评分 (优先级较高)
    score.loc[buy_cross_above_zero] = 80.0 # 零轴上方金叉更强
    score.loc[buy_cross_below_zero] = 70.0 # 零轴下方金叉稍弱
    score.loc[sell_cross_below_zero] = 20.0 # 零轴下方死叉更强
    score.loc[sell_cross_above_zero] = 30.0 # 零轴上方死叉稍弱

    # 根据 MACDh (Histogram) 的柱子变化判断动能 (非交叉时)
    # MACDh 上涨 (动能增加)
    bullish_momentum = (macd_h_filled > macd_h_filled.shift(1)) & (macd_h_filled > 0)
    # MACDh 下跌 (动能减少)
    bearish_momentum = (macd_h_filled < macd_h_filled.shift(1)) & (macd_h_filled < 0)

    # MACDh 零轴上方 (非动能增加)
    above_zero_no_momentum_increase = (macd_h_filled > 0) & (~bullish_momentum) & (~buy_cross) & (~sell_cross)
    # MACDh 零轴下方 (非动能减少)
    below_zero_no_momentum_decrease = (macd_h_filled < 0) & (~bearish_momentum) & (~buy_cross) & (~sell_cross)

    # 应用动能和零轴位置评分 (优先级低于交叉信号)
    # 使用 np.maximum/minimum 确保不覆盖优先级更高的交叉信号
    score.loc[bullish_momentum & (~buy_cross) & (~sell_cross)] = np.maximum(score.loc[bullish_momentum & (~buy_cross) & (~sell_cross)], 60.0)
    score.loc[bearish_momentum & (~buy_cross) & (~sell_cross)] = np.minimum(score.loc[bearish_momentum & (~buy_cross) & (~sell_cross)], 40.0)
    score.loc[above_zero_no_momentum_increase] = np.maximum(score.loc[above_zero_no_momentum_increase], 55.0)
    score.loc[below_zero_no_momentum_decrease] = np.minimum(score.loc[below_zero_no_momentum_decrease], 45.0)

    # 确保所有未被上述规则覆盖的区域默认为50
    # 这一步在初始化时已经完成，但为了保险可以再次执行
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_rsi_score(rsi: pd.Series, params: Dict) -> pd.Series:
    """
    RSI 评分 (0-100)。
    评分逻辑基于超买超卖区域、超买超卖线的突破以及趋势。

    Args:
        rsi (pd.Series): RSI 指标序列。
        params (Dict): 包含 RSI 超买超卖阈值的字典。
                       期望包含 'rsi_oversold', 'rsi_overbought',
                       'rsi_extreme_oversold', 'rsi_extreme_overbought'。

    Returns:
        pd.Series: 计算出的 RSI 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=rsi.index)

    # 填充 NaN 值，避免计算错误
    rsi_filled = rsi.fillna(50.0) # RSI 中性区域通常在 50 附近

    # 使用 get 并提供默认值，避免 KeyError
    os = params.get('rsi_oversold', 30)
    ob = params.get('rsi_overbought', 70)
    ext_os = params.get('rsi_extreme_oversold', 20)
    ext_ob = params.get('rsi_extreme_overbought', 80)

    # 极度超卖/超买区域 (优先级最高)
    score.loc[rsi_filled < ext_os] = 95.0 # 极度超卖 - 强烈看涨
    score.loc[rsi_filled > ext_ob] = 5.0  # 极度超买 - 强烈看跌

    # 普通超卖/超买区域 (优先级次高)
    # 确保不覆盖极度区域的评分
    score.loc[(rsi_filled >= ext_os) & (rsi_filled < os)] = np.maximum(score.loc[(rsi_filled >= ext_os) & (rsi_filled < os)], 85.0) # 普通超卖 - 较强看涨
    score.loc[(rsi_filled <= ext_ob) & (rsi_filled > ob)] = np.minimum(score.loc[(rsi_filled <= ext_ob) & (rsi_filled > ob)], 15.0) # 普通超买 - 较强看跌

    # 从超卖区向上突破超卖线 (买入信号)
    buy_signal = (rsi_filled.shift(1) < os) & (rsi_filled >= os)
    score.loc[buy_signal] = 75.0 # 通常是买入信号

    # 从超买区向下突破超买线 (卖出信号)
    sell_signal = (rsi_filled.shift(1) > ob) & (rsi_filled <= ob)
    score.loc[sell_signal] = 25.0 # 通常是卖出信号

    # 在中轴区域 (os 到 ob) 的趋势判断 (优先级低于突破信号)
    # 处于非超买超卖区域，且 RSI 上升
    bullish_trend = (rsi_filled >= os) & (rsi_filled <= ob) & (rsi_filled > rsi_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # 看涨趋势
    # 处于非超买超卖区域，且 RSI 下降
    bearish_trend = (rsi_filled >= os) & (rsi_filled <= ob) & (rsi_filled < rsi_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # 看跌趋势

    # 确保所有未被上述规则覆盖的区域默认为50
    # 这一行在上面的趋势判断后可能不再需要，或者用于填充剩余的无信号区域
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_kdj_score(k: pd.Series, d: pd.Series, j: pd.Series, params: Dict) -> pd.Series:
    """
    KDJ 评分 (0-100)。
    评分逻辑基于金叉死叉、超买超卖区域以及 J 值的变化。

    Args:
        k (pd.Series): K 线序列。
        d (pd.Series): D 线序列。
        j (pd.Series): J 线序列。
        params (Dict): 包含 KDJ 超买超卖阈值的字典。
                       期望包含 'kdj_oversold', 'kdj_overbought',
                       'kdj_extreme_oversold', 'kdj_extreme_overbought'。

    Returns:
        pd.Series: 计算出的 KDJ 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=k.index)

    # 确保输入序列长度一致
    if not k.index.equals(d.index) or not k.index.equals(j.index):
        logger.warning("KDJ 评分输入序列索引不一致，可能导致计算错误。")
        # 尝试重新索引，用 50.0 填充缺失值
        index = k.index.union(d.index).union(j.index)
        k = k.reindex(index).fillna(50.0)
        d = d.reindex(index).fillna(50.0)
        j = j.reindex(index).fillna(50.0)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    k_filled = k.fillna(50.0)
    d_filled = d.fillna(50.0)
    j_filled = j.fillna(50.0)

    # 使用 get 并提供默认值，避免 KeyError
    os = params.get('kdj_oversold', 20)
    ob = params.get('kdj_overbought', 80)
    ext_os = params.get('kdj_extreme_oversold', 10) # 增加极值区
    ext_ob = params.get('kdj_extreme_overbought', 90)

    # 极度超卖/超买 (主要看J值，优先级最高)
    score.loc[j_filled < ext_os] = 95.0 # 极度超卖 - 强烈看涨
    score.loc[j_filled > ext_ob] = 5.0  # 极度超买 - 强烈看跌

    # 普通超卖/超买 (主要看K,D值，优先级次高)
    # 确保不覆盖极度区域的评分
    score.loc[(k_filled < os) | (d_filled < os)] = np.maximum(score.loc[(k_filled < os) | (d_filled < os)], 85.0) # 普通超卖 - 较强看涨
    score.loc[(k_filled > ob) | (d_filled > ob)] = np.minimum(score.loc[(k_filled > ob) | (d_filled > ob)], 15.0) # 普通超买 - 较强看跌

    # 金叉: K 上穿 D
    buy_cross = (k_filled.shift(1) < d_filled.shift(1)) & (k_filled >= d_filled)
    # 死叉: K 下穿 D
    sell_cross = (k_filled.shift(1) > d_filled.shift(1)) & (k_filled <= d_filled)

    # 金叉死叉的区域加强 (优先级高于普通超买超卖区域，低于极度区域)
    buy_cross_os = buy_cross & (j_filled < os) # 超卖区金叉 (更强买入信号)
    buy_cross_ob = buy_cross & (j_filled > ob) # 超买区金叉 (假信号风险高)
    sell_cross_os = sell_cross & (j_filled < os) # 超卖区死叉 (假信号风险高)
    sell_cross_ob = sell_cross & (j_filled > ob) # 超买区死叉 (更强卖出信号)

    # 应用交叉信号评分
    score.loc[buy_cross_os] = 80.0 # 超卖区金叉更强
    score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)] = 75.0 # 非超买超卖区的金叉
    score.loc[buy_cross_ob] = 60.0 # 超买区金叉减弱

    score.loc[sell_cross_ob] = 20.0 # 超买区死叉更强
    score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)] = 25.0 # 非超买超卖区的死叉
    score.loc[sell_cross_os] = 40.0 # 超卖区死叉减弱


    # J 值的趋势和位置判断 (非交叉时，优先级最低)
    # J 值在超卖区向上运行
    bullish_j = (j_filled < os) & (j_filled > j_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bullish_j] = np.maximum(score.loc[bullish_j], 70.0)
    # J 值在超买区向下运行
    bearish_j = (j_filled > ob) & (j_filled < j_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bearish_j] = np.minimum(score.loc[bearish_j], 30.0)

    # J 值在中轴区域 (os 到 ob)
    neutral_j_zone = (j_filled >= os) & (j_filled <= ob) & (~buy_cross) & (~sell_cross)
    # 根据 J 值在中轴区的位置和趋势微调分数 (越高越看涨，越低越看跌)
    # 处于中轴区域，且 J 值上升
    bullish_j_trend_neutral_zone = neutral_j_zone & (j_filled > j_filled.shift(1))
    score.loc[bullish_j_trend_neutral_zone] = np.maximum(score.loc[bullish_j_trend_neutral_zone], 55.0)
    # 处于中轴区域，且 J 值下降
    bearish_j_trend_neutral_zone = neutral_j_zone & (j_filled < j_filled.shift(1))
    score.loc[bearish_j_trend_neutral_zone] = np.minimum(score.loc[bearish_j_trend_neutral_zone], 45.0)
    # 处于中轴区域，J 值横盘或波动不大
    neutral_j_trend_neutral_zone = neutral_j_zone & (~bullish_j_trend_neutral_zone) & (~bearish_j_trend_neutral_zone)
    score.loc[neutral_j_trend_neutral_zone] = 50.0 # 中轴横盘设为中性分

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_boll_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """
    BOLL 评分 (0-100)。
    评分逻辑基于价格与布林带上下轨和中轨的相对位置及突破。

    Args:
        close (pd.Series): 收盘价序列。
        upper (pd.Series): 布林带上轨序列。
        mid (pd.Series): 布林带中轨序列。
        lower (pd.Series): 布林带下轨序列。

    Returns:
        pd.Series: 计算出的 BOLL 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列长度一致
    if not close.index.equals(upper.index) or not close.index.equals(mid.index) or not close.index.equals(lower.index):
        logger.warning("BOLL 评分输入序列索引不一致，可能导致计算错误。")
        # 尝试重新索引，用 NaN 填充缺失值，然后填充合理默认值
        index = close.index.union(upper.index).union(mid.index).union(lower.index)
        close = close.reindex(index)
        upper = upper.reindex(index)
        mid = mid.reindex(index)
        lower = lower.reindex(index)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    # 对于价格和布林带，NaN 可能表示数据缺失，填充前向/后向再填充中轨或收盘价
    close_filled = close.ffill().bfill().fillna(mid.ffill().bfill().fillna(close.mean())) # 价格优先填充，再用中轨，最后用均值
    upper_filled = upper.ffill().bfill().fillna(mid.ffill().bfill().add(close_filled.std() * 2, fill_value=0)) # 上轨填充
    mid_filled = mid.ffill().bfill().fillna(close_filled.rolling(20).mean().fillna(close_filled.mean())) # 中轨填充
    lower_filled = lower.ffill().bfill().fillna(mid_filled.ffill().bfill().sub(close_filled.std() * 2, fill_value=0)) # 下轨填充

    # 触及或跌破下轨 (极度超卖区，强烈看涨)
    score.loc[close_filled <= lower_filled] = 90.0

    # 从下轨下方回到下轨上方 (买入信号)
    buy_support = (close_filled.shift(1) < lower_filled.shift(1)) & (close_filled >= lower_filled)
    score.loc[buy_support] = 80.0 # 下轨支撑确认

    # 触及或突破上轨 (极度超买区，强烈看跌)
    score.loc[close_filled >= upper_filled] = 10.0

    # 从上轨上方回到上轨下方 (卖出信号)
    sell_pressure = (close_filled.shift(1) > upper_filled.shift(1)) & (close_filled <= upper_filled)
    score.loc[sell_pressure] = 20.0 # 上轨压力确认

    # 向上突破中轨 (看涨信号)
    buy_mid_cross = (close_filled.shift(1) < mid_filled.shift(1)) & (close_filled >= mid_filled)
    score.loc[buy_mid_cross] = 65.0

    # 向下跌破中轨 (看跌信号)
    sell_mid_cross = (close_filled.shift(1) > mid_filled.shift(1)) & (close_filled <= mid_filled)
    score.loc[sell_mid_cross] = 35.0

    # 价格在中轨上方（未触及上轨且未向上突破中轨）(看涨趋势)
    is_above_mid = (close_filled > mid_filled) & (close_filled < upper_filled) & (~buy_mid_cross)
    score.loc[is_above_mid] = np.maximum(score.loc[is_above_mid], 55.0) # 确保不覆盖更高优先级信号

    # 价格在中轨下方（未触及下轨且未向下跌破中轨）(看跌趋势)
    is_below_mid = (close_filled < mid_filled) & (close_filled > lower_filled) & (~sell_mid_cross)
    score.loc[is_below_mid] = np.minimum(score.loc[is_below_mid], 45.0) # 确保不覆盖更高优先级信号

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_cci_score(cci: pd.Series, params: Dict) -> pd.Series:
    """
    CCI 评分 (0-100)。
    评分逻辑基于超买超卖区域 (+100, -100) 和极度超买超卖区域 (+200, -200) 的突破和趋势。

    Args:
        cci (pd.Series): CCI 指标序列。
        params (Dict): 包含 CCI 阈值的字典。
                       期望包含 'cci_threshold', 'cci_extreme_threshold'。

    Returns:
        pd.Series: 计算出的 CCI 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=cci.index)

    # 填充 NaN 值，避免计算错误
    cci_filled = cci.fillna(0.0) # CCI 中性区域通常在 0 附近

    # 使用 get 并提供默认值，避免 KeyError
    threshold = params.get('cci_threshold', 100)
    ext_threshold = params.get('cci_extreme_threshold', 200)

    # 极度超卖/超买区 (优先级最高)
    score.loc[cci_filled < -ext_threshold] = 95.0 # 极度超卖 - 强烈看涨
    score.loc[cci_filled > ext_threshold] = 5.0  # 极度超买 - 强烈看跌

    # 普通超卖/超买区 (优先级次高)
    # 确保不覆盖极度区域的评分
    score.loc[(cci_filled >= -ext_threshold) & (cci_filled < -threshold)] = np.maximum(score.loc[(cci_filled >= -ext_threshold) & (cci_filled < -threshold)], 85.0) # 普通超卖 - 较强看涨
    score.loc[(cci_filled <= ext_threshold) & (cci_filled > threshold)] = np.minimum(score.loc[(cci_filled <= ext_threshold) & (cci_filled > threshold)], 15.0) # 普通超买 - 较强看跌

    # 从超卖区向上突破 -100 线 (买入信号)
    buy_signal = (cci_filled.shift(1) < -threshold) & (cci_filled >= -threshold)
    score.loc[buy_signal] = 75.0

    # 从超买区向下突破 +100 线 (卖出信号)
    sell_signal = (cci_filled.shift(1) > threshold) & (cci_filled <= threshold)
    score.loc[sell_signal] = 25.0

    # 在中轴区域 (-threshold 到 +threshold) 的趋势判断 (优先级低于突破信号)
    # 处于中轴区域，且 CCI 上升
    bullish_trend = (cci_filled >= -threshold) & (cci_filled <= threshold) & (cci_filled > cci_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # 看涨趋势
    # 处于中轴区域，且 CCI 下降
    bearish_trend = (cci_filled >= -threshold) & (cci_filled <= threshold) & (cci_filled < cci_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # 看跌趋势

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_mfi_score(mfi: pd.Series, params: Dict) -> pd.Series:
    """
    MFI 评分 (0-100)。
    评分逻辑基于超买超卖区域以及超买超卖线的突破。

    Args:
        mfi (pd.Series): MFI 指标序列。
        params (Dict): 包含 MFI 超买超卖阈值的字典。
                       期望包含 'mfi_oversold', 'mfi_overbought',
                       'mfi_extreme_oversold', 'mfi_extreme_overbought'。

    Returns:
        pd.Series: 计算出的 MFI 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=mfi.index)

    # 填充 NaN 值，避免计算错误
    mfi_filled = mfi.fillna(50.0) # MFI 范围 0-100，中性区域通常在 50 附近

    # 使用 get 并提供默认值，避免 KeyError
    os = params.get('mfi_oversold', 20)
    ob = params.get('mfi_overbought', 80)
    ext_os = params.get('mfi_extreme_oversold', 10)
    ext_ob = params.get('mfi_extreme_overbought', 90)

    # 极度超卖/超买区 (优先级最高)
    score.loc[mfi_filled < ext_os] = 95.0 # 极度超卖 - 强烈看涨
    score.loc[mfi_filled > ext_ob] = 5.0  # 极度超买 - 强烈看跌

    # 普通超卖/超买区 (优先级次高)
    # 确保不覆盖极度区域的评分
    score.loc[(mfi_filled >= ext_os) & (mfi_filled < os)] = np.maximum(score.loc[(mfi_filled >= ext_os) & (mfi_filled < os)], 85.0) # 普通超卖 - 较强看涨
    score.loc[(mfi_filled <= ext_ob) & (mfi_filled > ob)] = np.minimum(score.loc[(mfi_filled <= ext_ob) & (mfi_filled > ob)], 15.0) # 普通超买 - 较强看跌

    # 从超卖区向上突破超卖线 (买入信号)
    buy_signal = (mfi_filled.shift(1) < os) & (mfi_filled >= os)
    score.loc[buy_signal] = 75.0

    # 从超买区向下突破超买线 (卖出信号)
    sell_signal = (mfi_filled.shift(1) > ob) & (mfi_filled <= ob)
    score.loc[sell_signal] = 25.0

    # 在中轴区域 (os 到 ob) 的趋势判断 (优先级低于突破信号)
    # 处于中轴区域，且 MFI 上升
    bullish_trend = (mfi_filled >= os) & (mfi_filled <= ob) & (mfi_filled > mfi_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # 看涨趋势
    # 处于中轴区域，且 MFI 下降
    bearish_trend = (mfi_filled >= os) & (mfi_filled <= ob) & (mfi_filled < mfi_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # 看跌趋势

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_roc_score(roc: pd.Series) -> pd.Series:
    """
    ROC 评分 (0-100)。
    评分逻辑基于 ROC 线与零轴的交叉和趋势。

    Args:
        roc (pd.Series): ROC 指标序列。

    Returns:
        pd.Series: 计算出的 ROC 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=roc.index)

    # 填充 NaN 值，避免计算错误
    roc_filled = roc.fillna(0.0) # ROC 中性区域是 0

    # 上穿 0 轴 (买入信号)
    buy_cross = (roc_filled.shift(1) < 0) & (roc_filled >= 0)
    score.loc[buy_cross] = 70.0

    # 下穿 0 轴 (卖出信号)
    sell_cross = (roc_filled.shift(1) > 0) & (roc_filled <= 0)
    score.loc[sell_cross] = 30.0

    # 在 0 轴上方且上升 (看涨趋势加强)
    bullish_trend = (roc_filled > 0) & (roc_filled > roc_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 60.0)

    # 在 0 轴下方且下降 (看跌趋势加强)
    bearish_trend = (roc_filled < 0) & (roc_filled < roc_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 40.0)

    # 在 0 轴上方且下降 (动能减弱，潜在回调风险)
    bullish_waning = (roc_filled > 0) & (roc_filled < roc_filled.shift(1)) & (~sell_cross)
    score.loc[bullish_waning] = np.minimum(score.loc[bullish_waning], 55.0) # 略微减弱看涨信号

    # 在 0 轴下方且上升 (动能减弱，潜在反弹风险)
    bearish_waning = (roc_filled < 0) & (roc_filled > roc_filled.shift(1)) & (~buy_cross)
    score.loc[bearish_waning] = np.maximum(score.loc[bearish_waning], 45.0) # 略微增强看跌信号


    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_dmi_score(pdi: pd.Series, ndi: pd.Series, adx: pd.Series, params: Dict) -> pd.Series:
    """
    DMI 评分 (0-100)。
    评分逻辑基于 PDI (+DI) 和 NDI (-DI) 的交叉以及 ADX 的趋势强度。

    Args:
        pdi (pd.Series): Positive Directional Indicator (+DI) 序列。
        ndi (pd.Series): Negative Directional Indicator (-DI) 序列。
        adx (pd.Series): Average Directional Index (ADX) 序列。
        params (Dict): 包含 ADX 阈值的字典。
                       期望包含 'adx_threshold', 'adx_strong_threshold'。

    Returns:
        pd.Series: 计算出的 DMI 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=pdi.index)

    # 确保输入序列长度一致
    if not pdi.index.equals(ndi.index) or not pdi.index.equals(adx.index):
        logger.warning("DMI 评分输入序列索引不一致，可能导致计算错误。")
        # 尝试重新索引，用 NaN 填充缺失值，然后填充合理默认值
        index = pdi.index.union(ndi.index).union(adx.index)
        pdi = pdi.reindex(index).fillna(0.0)
        ndi = ndi.reindex(index).fillna(0.0)
        adx = adx.reindex(index).fillna(0.0) # ADX 最小为 0
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    pdi_filled = pdi.fillna(0.0)
    ndi_filled = ndi.fillna(0.0)
    adx_filled = adx.fillna(0.0)

    # 使用 get 并提供默认值，避免 KeyError
    adx_th = params.get('adx_threshold', 25)
    adx_strong_th = params.get('adx_strong_threshold', 40)

    # 金叉: PDI 上穿 NDI (买入信号)
    buy_cross = (pdi_filled.shift(1) < ndi_filled.shift(1)) & (pdi_filled >= ndi_filled)
    score.loc[buy_cross] = 70.0 # 金叉基础分

    # 死叉: NDI 上穿 PDI (卖出信号)
    sell_cross = (ndi_filled.shift(1) < pdi_filled.shift(1)) & (ndi_filled >= pdi_filled)
    score.loc[sell_cross] = 30.0 # 死叉基础分

    # ADX 对趋势强度的确认 (加强金叉死叉信号)
    # ADX 趋势加强 (上升)
    adx_rising = adx_filled > adx_filled.shift(1)

    # 金叉且 ADX 确认 (ADX > 阈值)
    score.loc[buy_cross & (adx_filled > adx_th)] = 75.0
    # 金叉且 ADX 强趋势且趋势加强
    score.loc[buy_cross & (adx_filled > adx_strong_th) & adx_rising] = 85.0 # 强趋势且趋势加强的金叉

    # 死叉且 ADX 确认
    score.loc[sell_cross & (adx_filled > adx_th)] = 25.0
    # 死叉且 ADX 强趋势且趋势加强
    score.loc[sell_cross & (adx_filled > adx_strong_th) & adx_rising] = 15.0 # 强趋势且趋势加强的死叉

    # 非交叉时的趋势判断 (PDI 和 NDI 的相对位置) (优先级低于交叉信号)
    # 多头趋势: PDI > NDI
    is_bullish_trend = (pdi_filled > ndi_filled) & (~buy_cross) & (~sell_cross)
    # 空头趋势: NDI > PDI
    is_bearish_trend = (ndi_filled > pdi_filled) & (~buy_cross) & (~sell_cross)

    # 根据 ADX 强度细化趋势判断
    # 强多头趋势 (PDI > NDI 且 ADX > 强阈值)
    score.loc[is_bullish_trend & (adx_filled > adx_strong_th)] = np.maximum(score.loc[is_bullish_trend & (adx_filled > adx_strong_th)], 65.0)
    # 普通多头趋势 (PDI > NDI 且 ADX > 普通阈值)
    score.loc[is_bullish_trend & (adx_filled > adx_th) & (adx_filled <= adx_strong_th)] = np.maximum(score.loc[is_bullish_trend & (adx_filled > adx_th) & (adx_filled <= adx_strong_th)], 60.0)
    # 弱多头趋势或盘整中的多头占优 (PDI > NDI 且 ADX <= 普通阈值)
    score.loc[is_bullish_trend & (adx_filled <= adx_th)] = np.maximum(score.loc[is_bullish_trend & (adx_filled <= adx_th)], 55.0)

    # 强空头趋势 (NDI > PDI 且 ADX > 强阈值)
    score.loc[is_bearish_trend & (adx_filled > adx_strong_th)] = np.minimum(score.loc[is_bearish_trend & (adx_filled > adx_strong_th)], 35.0)
    # 普通空头趋势 (NDI > PDI 且 ADX > 普通阈值)
    score.loc[is_bearish_trend & (adx_filled > adx_th) & (adx_filled <= adx_strong_th)] = np.minimum(score.loc[is_bearish_trend & (adx_filled > adx_th) & (adx_filled <= adx_strong_th)], 40.0)
    # 弱空头趋势或盘整中的空头占优 (NDI > PDI 且 ADX <= 普通阈值)
    score.loc[is_bearish_trend & (adx_filled <= adx_th)] = np.minimum(score.loc[is_bearish_trend & (adx_filled <= adx_th)], 45.0)

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_sar_score(close: pd.Series, sar: pd.Series) -> pd.Series:
    """
    SAR 评分 (0-100)。
    评分逻辑基于价格与 SAR 点的相对位置以及 SAR 点的反转。

    Args:
        close (pd.Series): 收盘价序列。
        sar (pd.Series): SAR 指标序列。

    Returns:
        pd.Series: 计算出的 SAR 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列长度一致
    if not close.index.equals(sar.index):
        logger.warning("SAR 评分输入序列索引不一致，可能导致计算错误。")
        # 尝试重新索引，用 NaN 填充缺失值，然后填充合理默认值
        index = close.index.union(sar.index)
        close = close.reindex(index)
        sar = sar.reindex(index)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    # 对于 SAR，NaN 通常出现在序列开头，填充前向/后向可能不合适
    # 简单填充为收盘价，表示价格与SAR重合，视为中性
    close_filled = close.fillna(method='ffill').fillna(method='bfill').fillna(close.mean()) # 价格填充
    sar_filled = sar.fillna(close_filled) # SAR 填充为对应的收盘价

    # SAR 信号是反转点，当前 SAR 与前一日 SAR 的相对位置，以及 SAR 与价格的关系
    # SAR 从价格上方反转到价格下方 -> 买入信号 (SAR 向上反转)
    # 条件：前一日 SAR 在价格上方，今日 SAR 在价格下方或等于价格
    buy_signal = (sar_filled.shift(1) > close_filled.shift(1)) & (sar_filled <= close_filled)
    score.loc[buy_signal] = 75.0

    # SAR 从价格下方反转到价格上方 -> 卖出信号 (SAR 向下反转)
    # 条件：前一日 SAR 在价格下方，今日 SAR 在价格上方或等于价格
    sell_signal = (sar_filled.shift(1) < close_filled.shift(1)) & (sar_filled >= close_filled)
    score.loc[sell_signal] = 25.0

    # 价格在 SAR 上方 (多头趋势) (非反转点)
    score.loc[(close_filled > sar_filled) & (~buy_signal) & (~sell_signal)] = 60.0

    # 价格在 SAR 下方 (空头趋势) (非反转点)
    score.loc[(close_filled < sar_filled) & (~buy_signal) & (~sell_signal)] = 40.0

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

def calculate_stoch_score(k: pd.Series, d: pd.Series, params: Dict) -> pd.Series:
    """
    随机指标 (STOCH) 评分 (0-100)。
    评分逻辑基于 K 和 D 线的金叉死叉、超买超卖区域。

    Args:
        k (pd.Series): STOCH K 线序列。
        d (pd.Series): STOCH D 线序列。
        params (Dict): 包含 STOCH 超买超卖阈值的字典。
                       期望包含 'stoch_oversold', 'stoch_overbought',
                       'stoch_extreme_oversold', 'stoch_extreme_overbought'。

    Returns:
        pd.Series: 计算出的 STOCH 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=k.index)

    # 确保输入序列长度一致
    if not k.index.equals(d.index):
        logger.warning("STOCH 评分输入序列索引不一致，可能导致计算错误。")
        # 尝试重新索引，用 50.0 填充缺失值
        index = k.index.union(d.index)
        k = k.reindex(index).fillna(50.0)
        d = d.reindex(index).fillna(50.0)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    k_filled = k.fillna(50.0)
    d_filled = d.fillna(50.0)

    # 使用 get 并提供默认值，避免 KeyError
    os = params.get('stoch_oversold', 20)
    ob = params.get('stoch_overbought', 80)
    ext_os = params.get('stoch_extreme_oversold', 10) # 增加极值区
    ext_ob = params.get('stoch_extreme_overbought', 90)

    # 极度超卖/超买 (主要看 K, D 值，优先级最高)
    score.loc[(k_filled < ext_os) | (d_filled < ext_os)] = 95.0 # 极度超卖 - 强烈看涨
    score.loc[(k_filled > ext_ob) | (d_filled > ext_ob)] = 5.0  # 极度超买 - 强烈看跌

    # 普通超卖/超买 (主要看 K, D 值，优先级次高)
    # 确保不覆盖极度区域的评分
    score.loc[(k_filled >= ext_os) & (k_filled < os)] = np.maximum(score.loc[(k_filled >= ext_os) & (k_filled < os)], 85.0) # 普通超卖 - 较强看涨
    score.loc[(d_filled >= ext_os) & (d_filled < os)] = np.maximum(score.loc[(d_filled >= ext_os) & (d_filled < os)], 85.0)
    score.loc[(k_filled <= ext_ob) & (k_filled > ob)] = np.minimum(score.loc[(k_filled <= ext_ob) & (k_filled > ob)], 15.0) # 普通超买 - 较强看跌
    score.loc[(d_filled <= ext_ob) & (d_filled > ob)] = np.minimum(score.loc[(d_filled <= ext_ob) & (d_filled > ob)], 15.0)


    # 金叉: K 上穿 D
    buy_cross = (k_filled.shift(1) < d_filled.shift(1)) & (k_filled >= d_filled)
    # 死叉: K 下穿 D
    sell_cross = (k_filled.shift(1) > d_filled.shift(1)) & (k_filled <= d_filled)

    # 金叉死叉的区域加强 (优先级高于普通超买超卖区域，低于极度区域)
    buy_cross_os = buy_cross & (d_filled < os) # 超卖区金叉 (更强买入信号)
    buy_cross_ob = buy_cross & (d_filled > ob) # 超买区金叉 (假信号风险高)
    sell_cross_os = sell_cross & (d_filled < os) # 超卖区死叉 (假信号风险高)
    sell_cross_ob = sell_cross & (d_filled > ob) # 超买区死叉 (更强卖出信号)

    # 应用交叉信号评分
    score.loc[buy_cross_os] = 80.0
    score.loc[buy_cross & (~buy_cross_os) & (~buy_cross_ob)] = 75.0 # 非超买超卖区的金叉
    score.loc[buy_cross_ob] = 60.0

    score.loc[sell_cross_ob] = 20.0
    score.loc[sell_cross & (~sell_cross_os) & (~sell_cross_ob)] = 25.0 # 非超买超卖区的死叉
    score.loc[sell_cross_os] = 40.0

    # K, D 线在中轴区域 (os 到 ob) 的趋势判断 (优先级最低)
    neutral_stoch_zone = (k_filled >= os) & (k_filled <= ob) & (d_filled >= os) & (d_filled <= ob) & (~buy_cross) & (~sell_cross)
    # 根据 K, D 在中轴区的位置和趋势微调分数
    # 处于中轴区域，且 K, D 上升 (看涨趋势)
    bullish_trend_neutral_zone = neutral_stoch_zone & (k_filled > k_filled.shift(1)) & (d_filled > d_filled.shift(1))
    score.loc[bullish_trend_neutral_zone] = np.maximum(score.loc[bullish_trend_neutral_zone], 55.0)
    # 处于中轴区域，且 K, D 下降 (看跌趋势)
    bearish_trend_neutral_zone = neutral_stoch_zone & (k_filled < k_filled.shift(1)) & (d_filled < d_filled.shift(1))
    score.loc[bearish_trend_neutral_zone] = np.minimum(score.loc[bearish_trend_neutral_zone], 45.0)
    # 处于中轴区域，K, D 横盘或波动不大
    neutral_trend_neutral_zone = neutral_stoch_zone & (~bullish_trend_neutral_zone) & (~bearish_trend_neutral_zone)
    score.loc[neutral_trend_neutral_zone] = 50.0 # 中轴横盘设为中性分


    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_ma_score 函数用于 EMA/SMA 评分
def calculate_ma_score(close: pd.Series, ma: pd.Series, params: Dict) -> pd.Series:
    """
    移动平均线 (MA) 评分 (0-100)。
    评分逻辑基于价格与 MA 线的相对位置和交叉。

    Args:
        close (pd.Series): 收盘价序列。
        ma (pd.Series): 移动平均线序列 (EMA 或 SMA)。
        params (Dict): 包含 MA 类型 ('ma_type') 的字典。

    Returns:
        pd.Series: 计算出的 MA 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列长度一致
    if not close.index.equals(ma.index):
        logger.warning("MA 评分输入序列索引不一致，可能导致计算错误。")
        index = close.index.union(ma.index)
        close = close.reindex(index)
        ma = ma.reindex(index)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    close_filled = close.ffill().bfill().fillna(close.mean())
    ma_filled = ma.ffill().bfill().fillna(close_filled.rolling(20).mean().fillna(close_filled.mean())) # MA 填充为价格的均值

    # 价格上穿 MA 线 (买入信号)
    buy_cross = (close_filled.shift(1) < ma_filled.shift(1)) & (close_filled >= ma_filled)
    score.loc[buy_cross] = 70.0

    # 价格下穿 MA 线 (卖出信号)
    sell_cross = (close_filled.shift(1) > ma_filled.shift(1)) & (close_filled <= ma_filled)
    score.loc[sell_cross] = 30.0

    # 价格在 MA 线上方 (多头趋势) (非交叉时)
    score.loc[(close_filled > ma_filled) & (~buy_cross) & (~sell_cross)] = 60.0

    # 价格在 MA 线下方 (空头趋势) (非交叉时)
    score.loc[(close_filled < ma_filled) & (~buy_cross) & (~sell_cross)] = 40.0

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_atr_score 函数
def calculate_atr_score(atr: pd.Series) -> pd.Series:
    """
    ATR 评分 (0-100)。
    评分逻辑基于 ATR 的绝对值或相对值，反映波动率。
    高 ATR 可能表示趋势强劲或反转，低 ATR 表示盘整。
    这里简单基于 ATR 的相对大小评分。

    Args:
        atr (pd.Series): ATR 指标序列。

    Returns:
        pd.Series: 计算出的 ATR 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=atr.index)

    # 填充 NaN 值，避免计算错误
    atr_filled = atr.fillna(atr.mean()) # ATR 填充均值

    # 计算 ATR 的滚动平均，用于判断相对高低
    atr_mean = atr_filled.rolling(window=20, min_periods=max(1, len(atr_filled)//2)).mean().fillna(atr_filled.mean())
    # 计算 ATR 的滚动标准差
    atr_std = atr_filled.rolling(window=20, min_periods=max(1, len(atr_filled)//2)).std().fillna(atr_filled.std())

    # ATR 相对较高 (高于均值 + 0.5*std) - 波动率高，可能趋势强或反转
    high_volatility = atr_filled > (atr_mean + 0.5 * atr_std)
    # ATR 相对较低 (低于均值 - 0.5*std) - 波动率低，可能盘整
    low_volatility = atr_filled < (atr_mean - 0.5 * atr_std)

    # 评分：高波动率偏向趋势（看涨/看跌取决于其他指标），低波动率偏向中性
    # 这里 ATR 本身不直接指示方向，只指示强度/波动率
    # 简单评分：高 ATR 略微偏离中性（例如 60/40），低 ATR 靠近中性（例如 50）
    # 更复杂的评分需要结合价格趋势
    # 假设高 ATR 在趋势策略中是积极信号（趋势可能持续或启动）
    score.loc[high_volatility] = 60.0 # 高波动率，偏看涨（需要结合其他指标确认方向）
    score.loc[low_volatility] = 40.0 # 低波动率，偏看跌（趋势可能结束或盘整）

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_adl_score 函数
def calculate_adl_score(adl: pd.Series) -> pd.Series:
    """
    ADL (Accumulation/Distribution Line) 评分 (0-100)。
    评分逻辑基于 ADL 的趋势以及与价格的背离 (虽然背离检测有单独函数，这里可以做简单趋势判断)。
    ADL 上升表示累积（买入压力），下降表示派发（卖出压力）。

    Args:
        adl (pd.Series): ADL 指标序列。

    Returns:
        pd.Series: 计算出的 ADL 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=adl.index)

    # 填充 NaN 值，避免计算错误
    adl_filled = adl.fillna(method='ffill').fillna(method='bfill').fillna(0.0) # ADL 填充前向/后向再填充0

    # ADL 上升趋势 (看涨)
    bullish_trend = adl_filled > adl_filled.shift(1)
    score.loc[bullish_trend] = 60.0

    # ADL 下降趋势 (看跌)
    bearish_trend = adl_filled < adl_filled.shift(1)
    score.loc[bearish_trend] = 40.0

    # ADL 横盘 (中性)
    neutral_trend = adl_filled == adl_filled.shift(1)
    score.loc[neutral_trend] = 50.0

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_vwap_score 函数
def calculate_vwap_score(close: pd.Series, vwap: pd.Series) -> pd.Series:
    """
    VWAP (Volume Weighted Average Price) 评分 (0-100)。
    评分逻辑基于价格与 VWAP 的相对位置。
    价格在 VWAP 上方通常视为看涨，下方视为看跌。

    Args:
        close (pd.Series): 收盘价序列。
        vwap (pd.Series): VWAP 指标序列。

    Returns:
        pd.Series: 计算出的 VWAP 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列长度一致
    if not close.index.equals(vwap.index):
        logger.warning("VWAP 评分输入序列索引不一致，可能导致计算错误。")
        index = close.index.union(vwap.index)
        close = close.reindex(index)
        vwap = vwap.reindex(index)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    close_filled = close.ffill().bfill().fillna(close.mean())
    vwap_filled = vwap.ffill().bfill().fillna(close_filled) # VWAP 填充为对应的收盘价

    # 价格在 VWAP 上方 (看涨)
    score.loc[close_filled > vwap_filled] = 60.0

    # 价格在 VWAP 下方 (看跌)
    score.loc[close_filled < vwap_filled] = 40.0

    # 价格与 VWAP 重合 (中性)
    score.loc[close_filled == vwap_filled] = 50.0

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_ichimoku_score 函数
def calculate_ichimoku_score(close: pd.Series, tenkan: pd.Series, kijun: pd.Series, senkou_a: pd.Series, senkou_b: pd.Series, chikou: pd.Series) -> pd.Series:
    """
    Ichimoku (一目均衡表) 评分 (0-100)。
    评分逻辑基于价格与各线和云图的相对位置、线的交叉以及 Chikou Span 的位置。

    Args:
        close (pd.Series): 收盘价序列。
        tenkan (pd.Series): 转换线 (Tenkan-sen) 序列。
        kijun (pd.Series): 基准线 (Kijun-sen) 序列。
        senkou_a (pd.Series): 先行跨度 A (Senkou Span A) 序列。
        senkou_b (pd.Series): 先行跨度 B (Senkou Span B) 序列。
        chikou (pd.Series): 迟滞跨度 (Chikou Span) 序列。

    Returns:
        pd.Series: 计算出的 Ichimoku 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列长度一致 (这里简化检查，实际应用中需要更严格)
    # 填充 NaN 值，避免计算错误 (一目均衡表有很多滞后和先行指标，NaN处理复杂，这里简单填充)
    close_filled = close.ffill().bfill().fillna(close.mean())
    tenkan_filled = tenkan.ffill().bfill().fillna(close_filled)
    kijun_filled = kijun.ffill().bfill().fillna(close_filled)
    senkou_a_filled = senkou_a.ffill().bfill().fillna(close_filled)
    senkou_b_filled = senkou_b.ffill().bfill().fillna(close_filled)
    chikou_filled = chikou.ffill().bfill().fillna(close_filled) # Chikou 是当前收盘价后移

    # 计算云图边界 (Senkou Span A 和 B 构成云图)
    cloud_upper = np.maximum(senkou_a_filled, senkou_b_filled)
    cloud_lower = np.minimum(senkou_a_filled, senkou_b_filled)

    # 评分项权重 (示例，可以从参数中获取)
    w_price_kijun = 0.2 # 价格与基准线
    w_tenkan_kijun_cross = 0.2 # 转换线与基准线交叉
    w_price_cloud = 0.3 # 价格与云图
    w_cloud_twist = 0.1 # 云图扭转 (Senkou A 穿 Senkou B)
    w_chikou_price = 0.2 # 迟滞跨度与价格

    # 1. 价格与基准线 (Kijun-sen)
    # 价格在基准线上方 (看涨)
    price_above_kijun = close_filled > kijun_filled
    # 价格在基准线下方 (看跌)
    price_below_kijun = close_filled < kijun_filled
    # 价格上穿基准线 (买入信号)
    price_cross_kijun_up = (close_filled.shift(1) < kijun_filled.shift(1)) & (close_filled >= kijun_filled)
    # 价格下穿基准线 (卖出信号)
    price_cross_kijun_down = (close_filled.shift(1) > kijun_filled.shift(1)) & (close_filled <= kijun_filled)

    score.loc[price_cross_kijun_up] = np.maximum(score.loc[price_cross_kijun_up], 60 + w_price_kijun * 50) # 交叉信号加强
    score.loc[price_cross_kijun_down] = np.minimum(score.loc[price_cross_kijun_down], 40 - w_price_kijun * 50)
    score.loc[price_above_kijun & ~price_cross_kijun_up] = np.maximum(score.loc[price_above_kijun & ~price_cross_kijun_up], 50 + w_price_kijun * 25) # 上方趋势
    score.loc[price_below_kijun & ~price_cross_kijun_down] = np.minimum(score.loc[price_below_kijun & ~price_cross_kijun_down], 50 - w_price_kijun * 25) # 下方趋势


    # 2. 转换线 (Tenkan-sen) 与基准线 (Kijun-sen) 交叉
    # 转换线上穿基准线 (金叉，买入信号)
    tenkan_kijun_cross_up = (tenkan_filled.shift(1) < kijun_filled.shift(1)) & (tenkan_filled >= kijun_filled)
    # 转换线下穿基准线 (死叉，卖出信号)
    tenkan_kijun_cross_down = (tenkan_filled.shift(1) > kijun_filled.shift(1)) & (tenkan_filled <= kijun_filled)

    score.loc[tenkan_kijun_cross_up] = np.maximum(score.loc[tenkan_kijun_cross_up], 50 + w_tenkan_kijun_cross * 50)
    score.loc[tenkan_kijun_cross_down] = np.minimum(score.loc[tenkan_kijun_cross_down], 50 - w_tenkan_kijun_cross * 50)


    # 3. 价格与云图 (Kumo)
    # 价格在云图上方 (强烈看涨)
    price_above_cloud = close_filled > cloud_upper
    # 价格在云图下方 (强烈看跌)
    price_below_cloud = close_filled < cloud_lower
    # 价格在云图内部 (盘整或趋势不明)
    price_in_cloud = (close_filled >= cloud_lower) & (close_filled <= cloud_upper)

    score.loc[price_above_cloud] = np.maximum(score.loc[price_above_cloud], 50 + w_price_cloud * 50)
    score.loc[price_below_cloud] = np.minimum(score.loc[price_below_cloud], 50 - w_price_cloud * 50)
    # 价格在云图内，评分靠近 50
    # score.loc[price_in_cloud] = 50.0 # 可能会覆盖其他信号，这里不直接设为50
    # 可以根据价格在云图内的位置微调，靠近上沿偏多，靠近下沿偏空
    # 简单处理：在云图内且价格上涨偏多，价格下跌偏空
    price_rising_in_cloud = price_in_cloud & (close_filled > close_filled.shift(1))
    price_falling_in_cloud = price_in_cloud & (close_filled < close_filled.shift(1))
    score.loc[price_rising_in_cloud] = np.maximum(score.loc[price_rising_in_cloud], 55.0)
    score.loc[price_falling_in_cloud] = np.minimum(score.loc[price_falling_in_cloud], 45.0)


    # 4. 云图扭转 (Senkou Span A 穿 Senkou Span B) (先行指标，看未来趋势)
    # Senkou Span A 上穿 Senkou Span B (看涨扭转)
    cloud_twist_up = (senkou_a_filled.shift(1) < senkou_b_filled.shift(1)) & (senkou_a_filled >= senkou_b_filled)
    # Senkou Span A 下穿 Senkou Span B (看跌扭转)
    cloud_twist_down = (senkou_a_filled.shift(1) > senkou_b_filled.shift(1)) & (senkou_a_filled <= senkou_b_filled)

    # 云图扭转信号通常是先行指标，其评分可以影响未来几个周期的评分
    # 这里简单将扭转信号的评分应用到当前周期
    score.loc[cloud_twist_up] = np.maximum(score.loc[cloud_twist_up], 50 + w_cloud_twist * 50)
    score.loc[cloud_twist_down] = np.minimum(score.loc[cloud_twist_down], 50 - w_cloud_twist * 50)


    # 5. 迟滞跨度 (Chikou Span) 与价格
    # Chikou Span 在价格上方 (看涨)
    chikou_above_price = chikou_filled > close_filled.shift(26) # Chikou Span 是当前收盘价后移26周期
    # Chikou Span 在价格下方 (看跌)
    chikou_below_price = chikou_filled < close_filled.shift(26)

    score.loc[chikou_above_price] = np.maximum(score.loc[chikou_above_price], 50 + w_chikou_price * 50)
    score.loc[chikou_below_price] = np.minimum(score.loc[chikou_below_price], 50 - w_chikou_price * 50)


    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_mom_score 函数
def calculate_mom_score(mom: pd.Series) -> pd.Series:
    """
    MOM (Momentum) 评分 (0-100)。
    评分逻辑基于 MOM 线与零轴的交叉和趋势。
    MOM 上穿零轴表示动能向上，下穿零轴表示动能向下。

    Args:
        mom (pd.Series): MOM 指标序列。

    Returns:
        pd.Series: 计算出的 MOM 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=mom.index)

    # 填充 NaN 值，避免计算错误
    mom_filled = mom.fillna(0.0) # MOM 中性区域是 0

    # 上穿 0 轴 (买入信号)
    buy_cross = (mom_filled.shift(1) < 0) & (mom_filled >= 0)
    score.loc[buy_cross] = 65.0

    # 下穿 0 轴 (卖出信号)
    sell_cross = (mom_filled.shift(1) > 0) & (mom_filled <= 0)
    score.loc[sell_cross] = 35.0

    # 在 0 轴上方且上升 (看涨动能加强)
    bullish_trend = (mom_filled > 0) & (mom_filled > mom_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)

    # 在 0 轴下方且下降 (看跌动能加强)
    bearish_trend = (mom_filled < 0) & (mom_filled < mom_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_willr_score 函数
def calculate_willr_score(willr: pd.Series) -> pd.Series:
    """
    WILLR (%R) 评分 (0-100)。
    评分逻辑基于超买超卖区域 (-20, -80)。注意 %R 的范围通常是 0 到 -100。
    -20 以上为超买，-80 以下为超卖。

    Args:
        willr (pd.Series): Williams %R 指标序列。

    Returns:
        pd.Series: 计算出的 WILLR 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=willr.index)

    # 填充 NaN 值，避免计算错误
    willr_filled = willr.fillna(-50.0) # %R 中性区域通常在 -50 附近

    # 超买超卖阈值
    ob_th = -20
    os_th = -80
    # 极度超买超卖阈值 (示例)
    ext_ob_th = -10
    ext_os_th = -90

    # 极度超卖/超买区 (优先级最高)
    score.loc[willr_filled < ext_os_th] = 95.0 # 极度超卖 (-90以下) - 强烈看涨
    score.loc[willr_filled > ext_ob_th] = 5.0  # 极度超买 (-10以上) - 强烈看跌

    # 普通超卖/超买区 (优先级次高)
    # 确保不覆盖极度区域的评分
    score.loc[(willr_filled >= ext_os_th) & (willr_filled < os_th)] = np.maximum(score.loc[(willr_filled >= ext_os_th) & (willr_filled < os_th)], 85.0) # 普通超卖 (-80到-90) - 较强看涨
    score.loc[(willr_filled <= ext_ob_th) & (willr_filled > ob_th)] = np.minimum(score.loc[(willr_filled <= ext_ob_th) & (willr_filled > ob_th)], 15.0) # 普通超买 (-10到-20) - 较强看跌

    # 从超卖区向上突破超卖线 (-80) (买入信号)
    buy_signal = (willr_filled.shift(1) < os_th) & (willr_filled >= os_th)
    score.loc[buy_signal] = 75.0

    # 从超买区向下突破超买线 (-20) (卖出信号)
    sell_signal = (willr_filled.shift(1) > ob_th) & (willr_filled <= ob_th)
    score.loc[sell_signal] = 25.0

    # 在中轴区域 (-80 到 -20) 的趋势判断 (优先级低于突破信号)
    # 处于中轴区域，且 %R 上升 (偏看跌，因为越接近-20越超买)
    bearish_trend = (willr_filled >= os_th) & (willr_filled <= ob_th) & (willr_filled > willr_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0) # 偏看跌

    # 处于中轴区域，且 %R 下降 (偏看涨，因为越接近-80越超卖)
    bullish_trend = (willr_filled >= os_th) & (willr_filled <= ob_th) & (willr_filled < willr_filled.shift(1)) & (~buy_signal) & (~sell_signal)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0) # 偏看涨


    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_cmf_score 函数
def calculate_cmf_score(cmf: pd.Series) -> pd.Series:
    """
    CMF (Chaikin Money Flow) 评分 (0-100)。
    评分逻辑基于 CMF 线与零轴的相对位置和趋势。
    CMF > 0 表示资金流入（累积），CMF < 0 表示资金流出（派发）。

    Args:
        cmf (pd.Series): CMF 指标序列。

    Returns:
        pd.Series: 计算出的 CMF 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=cmf.index)

    # 填充 NaN 值，避免计算错误
    cmf_filled = cmf.fillna(0.0) # CMF 中性区域是 0

    # CMF > 0 (资金流入，看涨)
    score.loc[cmf_filled > 0] = 60.0

    # CMF < 0 (资金流出，看跌)
    score.loc[cmf_filled < 0] = 40.0

    # CMF 上升趋势 (资金流入增加，看涨加强)
    bullish_trend = cmf_filled > cmf_filled.shift(1)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)

    # CMF 下降趋势 (资金流出增加，看跌加强)
    bearish_trend = cmf_filled < cmf_filled.shift(1)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_obv_score 函数
def calculate_obv_score(obv: pd.Series) -> pd.Series:
    """
    OBV (On Balance Volume) 评分 (0-100)。
    评分逻辑基于 OBV 的趋势以及与价格的背离 (简单趋势判断)。
    OBV 上升表示买入量大于卖出量，下降表示卖出量大于买入量。

    Args:
        obv (pd.Series): OBV 指标序列。

    Returns:
        pd.Series: 计算出的 OBV 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=obv.index)

    # 填充 NaN 值，避免计算错误
    obv_filled = obv.fillna(method='ffill').fillna(method='bfill').fillna(obv.mean()) # OBV 填充前向/后向再填充均值

    # OBV 上升趋势 (看涨)
    bullish_trend = obv_filled > obv_filled.shift(1)
    score.loc[bullish_trend] = 60.0

    # OBV 下降趋势 (看跌)
    bearish_trend = obv_filled < obv_filled.shift(1)
    score.loc[bearish_trend] = 40.0

    # OBV 横盘 (中性)
    neutral_trend = obv_filled == obv_filled.shift(1)
    score.loc[neutral_trend] = 50.0

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_kc_score 函数
def calculate_kc_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """
    KC (Keltner Channel) 评分 (0-100)。
    评分逻辑基于价格与 KC 上下轨和中轨的相对位置及突破。
    类似于 BOLL 评分，但 KC 使用 ATR 而非标准差计算带宽。

    Args:
        close (pd.Series): 收盘价序列。
        upper (pd.Series): Keltner Channel 上轨序列。
        mid (pd.Series): Keltner Channel 中轨序列。
        lower (pd.Series): Keltner Channel 下轨序列。

    Returns:
        pd.Series: 计算出的 KC 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列长度一致 (这里简化检查)
    # 填充 NaN 值，避免计算错误
    close_filled = close.ffill().bfill().fillna(close.mean())
    mid_filled = mid.ffill().bfill().fillna(close_filled.rolling(20).mean().fillna(close_filled.mean())) # 中轨填充
    # 上下轨填充，简单用中轨加减一个固定值或均值ATR
    # 假设 ATR 列存在且已计算
    # atr_col_name = f'ATR_{params.get("atr_period", 10)}_{close.name.split("_")[-1]}' # 尝试构建 ATR 列名
    # atr_series = data.get(atr_col_name, pd.Series(0.0, index=close.index)).fillna(0.0) # 获取 ATR 或填充0
    # upper_filled = upper.ffill().bfill().fillna(mid_filled + atr_series * 2) # 填充上轨
    # lower_filled = lower.ffill().bfill().fillna(mid_filled - atr_series * 2) # 填充下轨
    # 如果 ATR 列不可用，简单填充为中轨加减一个比例
    upper_filled = upper.ffill().bfill().fillna(mid_filled * 1.05) # 填充上轨，假设上轨通常比中轨高5%
    lower_filled = lower.ffill().bfill().fillna(mid_filled * 0.95) # 填充下轨，假设下轨通常比中轨低5%


    # 触及或跌破下轨 (极度超卖区，强烈看涨)
    score.loc[close_filled <= lower_filled] = 90.0

    # 从下轨下方回到下轨上方 (买入信号)
    buy_support = (close_filled.shift(1) < lower_filled.shift(1)) & (close_filled >= lower_filled)
    score.loc[buy_support] = 80.0 # 下轨支撑确认

    # 触及或突破上轨 (极度超买区，强烈看跌)
    score.loc[close_filled >= upper_filled] = 10.0

    # 从上轨上方回到上轨下方 (卖出信号)
    sell_pressure = (close_filled.shift(1) > upper_filled.shift(1)) & (close_filled <= upper_filled)
    score.loc[sell_pressure] = 20.0 # 上轨压力确认

    # 向上突破中轨 (看涨信号)
    buy_mid_cross = (close_filled.shift(1) < mid_filled.shift(1)) & (close_filled >= mid_filled)
    score.loc[buy_mid_cross] = 65.0

    # 向下跌破中轨 (看跌信号)
    sell_mid_cross = (close_filled.shift(1) > mid_filled.shift(1)) & (close_filled <= mid_filled)
    score.loc[sell_mid_cross] = 35.0

    # 价格在中轨上方（未触及上轨且未向上突破中轨）(看涨趋势)
    is_above_mid = (close_filled > mid_filled) & (close_filled < upper_filled) & (~buy_mid_cross)
    score.loc[is_above_mid] = np.maximum(score.loc[is_above_mid], 55.0) # 确保不覆盖更高优先级信号

    # 价格在中轨下方（未触及下轨且未向下跌破中轨）(看跌趋势)
    is_below_mid = (close_filled < mid_filled) & (close_filled > lower_filled) & (~sell_mid_cross)
    score.loc[is_below_mid] = np.minimum(score.loc[is_below_mid], 45.0) # 确保不覆盖更高优先级信号

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_hv_score 函数
def calculate_hv_score(hv: pd.Series) -> pd.Series:
    """
    HV (Historical Volatility) 评分 (0-100)。
    评分逻辑基于历史波动率的绝对值或相对值。
    高 HV 表示波动剧烈，低 HV 表示波动平缓。
    与 ATR 类似，HV 本身不指示方向，只指示波动率。

    Args:
        hv (pd.Series): 历史波动率序列。

    Returns:
        pd.Series: 计算出的 HV 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=hv.index)

    # 填充 NaN 值，避免计算错误
    hv_filled = hv.fillna(hv.mean()) # HV 填充均值

    # 计算 HV 的滚动平均，用于判断相对高低
    hv_mean = hv_filled.rolling(window=20, min_periods=max(1, len(hv_filled)//2)).mean().fillna(hv_filled.mean())
    # 计算 HV 的滚动标准差
    hv_std = hv_filled.rolling(window=20, min_periods=max(1, len(hv_filled)//2)).std().fillna(hv_filled.std())

    # HV 相对较高 (高于均值 + 0.5*std) - 波动率高
    high_volatility = hv_filled > (hv_mean + 0.5 * hv_std)
    # HV 相对较低 (低于均值 - 0.5*std) - 波动率低
    low_volatility = hv_filled < (hv_mean - 0.5 * hv_std)

    # 评分：高波动率略微偏离中性，低波动率靠近中性
    # 假设高 HV 在趋势策略中是积极信号（趋势可能持续或启动）
    score.loc[high_volatility] = 60.0 # 高波动率，偏看涨（需要结合其他指标确认方向）
    score.loc[low_volatility] = 40.0 # 低波动率，偏看跌（趋势可能结束或盘整）

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_vroc_score 函数
def calculate_vroc_score(vroc: pd.Series) -> pd.Series:
    """
    VROC (Volume Rate of Change) 评分 (0-100)。
    评分逻辑基于 VROC 线与零轴的交叉和趋势。
    VROC 上穿零轴表示成交量增长加速，下穿零轴表示成交量增长减速或萎缩。

    Args:
        vroc (pd.Series): VROC 指标序列。

    Returns:
        pd.Series: 计算出的 VROC 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=vroc.index)

    # 填充 NaN 值，避免计算错误
    vroc_filled = vroc.fillna(0.0) # VROC 中性区域是 0

    # 上穿 0 轴 (成交量增长加速，可能支持当前趋势)
    buy_cross = (vroc_filled.shift(1) < 0) & (vroc_filled >= 0)
    score.loc[buy_cross] = 55.0 # 略微看涨

    # 下穿 0 轴 (成交量增长减速或萎缩，可能预示趋势减弱)
    sell_cross = (vroc_filled.shift(1) > 0) & (vroc_filled <= 0)
    score.loc[sell_cross] = 45.0 # 略微看跌

    # 在 0 轴上方且上升 (成交量持续加速增长，看涨加强)
    bullish_trend = (vroc_filled > 0) & (vroc_filled > vroc_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 52.0) # 微弱看涨

    # 在 0 轴下方且下降 (成交量持续萎缩，看跌加强)
    bearish_trend = (vroc_filled < 0) & (vroc_filled < vroc_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 48.0) # 微弱看跌

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_aroc_score 函数
def calculate_aroc_score(aroc: pd.Series) -> pd.Series:
    """
    AROC (Absolute Rate of Change) 评分 (0-100)。
    评分逻辑基于 AROC 线与零轴的交叉和趋势。
    AROC 上穿零轴表示价格增长加速，下穿零轴表示价格增长减速或下跌。

    Args:
        aroc (pd.Series): AROC 指标序列。

    Returns:
        pd.Series: 计算出的 AROC 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=aroc.index)

    # 填充 NaN 值，避免计算错误
    aroc_filled = aroc.fillna(0.0) # AROC 中性区域是 0

    # 上穿 0 轴 (价格增长加速，买入信号)
    buy_cross = (aroc_filled.shift(1) < 0) & (aroc_filled >= 0)
    score.loc[buy_cross] = 65.0

    # 下穿 0 轴 (价格增长减速或下跌，卖出信号)
    sell_cross = (aroc_filled.shift(1) > 0) & (aroc_filled <= 0)
    score.loc[sell_cross] = 35.0

    # 在 0 轴上方且上升 (价格持续加速增长，看涨加强)
    bullish_trend = (aroc_filled > 0) & (aroc_filled > aroc_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 55.0)

    # 在 0 轴下方且下降 (价格持续下跌，看跌加强)
    bearish_trend = (aroc_filled < 0) & (aroc_filled < aroc_filled.shift(1)) & (~buy_cross) & (~sell_cross)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 45.0)

    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)

# 假设存在 calculate_pivot_score 函数
def calculate_pivot_score(close: pd.Series, pivot_levels_df: pd.DataFrame, params: Dict = None) -> pd.Series:
    """
    Pivot Points 评分 (0-100)。
    评分逻辑基于收盘价相对于 Pivot Point (PP) 和各支撑/阻力水平的位置。
    价格在 PP 上方偏多，下方偏空。突破阻力看涨，跌破支撑看跌。

    Args:
        close (pd.Series): 收盘价序列 (通常是日线)。
        pivot_levels_df (pd.DataFrame): 包含 PP, S1-S4, R1-R4, F_S1-F_S3, F_R1-F_R3 等 Pivot 水平的 DataFrame。
                                        列名应与 calculate_all_indicator_scores 中查找的列名一致。
        params (Dict, optional): 评分函数可能需要的额外参数。目前未使用。

    Returns:
        pd.Series: 计算出的 Pivot Points 评分序列 (0-100)。
    """
    # 初始化评分序列，默认中性分 50.0
    score = pd.Series(50.0, index=close.index)

    # 确保输入序列/DataFrame 索引一致
    if not close.index.equals(pivot_levels_df.index):
        logger.warning("Pivot Points 评分输入序列/DataFrame 索引不一致，可能导致计算错误。")
        index = close.index.union(pivot_levels_df.index)
        close = close.reindex(index)
        pivot_levels_df = pivot_levels_df.reindex(index)
        score = pd.Series(50.0, index=index) # 重新初始化评分序列

    # 填充 NaN 值，避免计算错误
    close_filled = close.ffill().bfill().fillna(close.mean())
    # Pivot 水平的 NaN 填充可能需要更复杂的逻辑，这里简单填充前向/后向再填充收盘价
    pivot_levels_filled_df = pivot_levels_df.ffill().bfill().fillna(close_filled.to_frame()) # 填充为对应的收盘价

    # 获取 Pivot Point (PP) 列
    pp_col = next((col for col in pivot_levels_filled_df.columns if 'PP_' in col), None)
    if pp_col is None:
        logger.warning("未找到 Pivot Point (PP) 列，无法计算 Pivot Points 评分。")
        return score # 返回默认中性分

    # 2. 价格突破支撑/阻力水平 (在区域评分基础上进行调整)
    # 遍历所有支撑和阻力水平
    pp_series = pivot_levels_filled_df[pp_col]

    # 1. 价格与 Pivot Point (PP) 的相对位置
    # 价格在 PP 上方 (偏看涨)
    score.loc[close_filled > pp_series] = 55.0
    # 价格在 PP 下方 (偏看跌)
    score.loc[close_filled < pp_series] = 45.0

    for col in pivot_levels_filled_df.columns:
        # 检查是否是标准阻力水平 (R1, R2, ...)
        if col.startswith('R') and '_' in col: # 确保是带时间框架后缀的列
            # 价格向上突破阻力 (看涨信号)
            resistance_series = pivot_levels_filled_df[col]
            # 确保用于比较的 Series 没有 NaN
            resistance_series_filled = resistance_series.fillna(close_filled) # 填充为收盘价，避免NaN比较
            buy_breakout = (close_filled.shift(1) < resistance_series_filled.shift(1)) & (close_filled >= resistance_series_filled)

            # 突破级别越高，信号越强 (示例权重)
            try:
                # 从列名中提取级别，例如 'R1_D' -> 'R1' -> 1
                level_str = col.split('_')[0][1:]
                level = int(level_str)
            except (ValueError, IndexError):
                logger.warning(f"无法从列名 '{col}' 提取阻力级别，使用默认级别 1。")
                level = 1

            # 突破信号的评分值 (示例：基础分 70 + 级别加成)
            breakout_score_value = 70 + level * 5 # 级别越高，突破信号越强
            score.loc[buy_breakout] = np.maximum(score.loc[buy_breakout], breakout_score_value) # 突破信号优先级高于区域评分

        # 检查是否是标准支撑水平 (S1, S2, ...)
        elif col.startswith('S') and '_' in col: # 确保是带时间框架后缀的列
            # 价格向下跌破支撑 (看跌信号)
            support_series = pivot_levels_filled_df[col]
             # 确保用于比较的 Series 没有 NaN
            support_series_filled = support_series.fillna(close_filled) # 填充为收盘价，避免NaN比较
            sell_breakdown = (close_filled.shift(1) > support_series_filled.shift(1)) & (close_filled <= support_series_filled)

            # 跌破级别越高，信号越强 (示例权重)
            try:
                # 从列名中提取级别，例如 'S1_D' -> 'S1' -> 1
                level_str = col.split('_')[0][1:]
                level = int(level_str)
            except (ValueError, IndexError):
                logger.warning(f"无法从列名 '{col}' 提取支撑级别，使用默认级别 1。")
                level = 1

            # 跌破信号的评分值 (示例：基础分 30 - 级别加成)
            breakdown_score_value = 30 - level * 5 # 级别越高，跌破信号越强
            score.loc[sell_breakdown] = np.minimum(score.loc[sell_breakdown], breakdown_score_value) # 跌破信号优先级高于区域评分

        # --- 添加对 Fibonacci 阻力水平 (F_R1, F_R2, ...) 的处理 ---
        elif col.startswith('F_R') and '_' in col: # 确保是带时间框架后缀的列
            # 价格向上突破 Fibonacci 阻力 (看涨信号)
            fib_resistance_series = pivot_levels_filled_df[col]
            # 确保用于比较的 Series 没有 NaN
            fib_resistance_series_filled = fib_resistance_series.fillna(close_filled) # 填充为收盘价，避免NaN比较
            fib_buy_breakout = (close_filled.shift(1) < fib_resistance_series_filled.shift(1)) & (close_filled >= fib_resistance_series_filled)

            # 突破级别越高，信号越强 (示例权重)
            try:
                # 从列名中提取级别，例如 'F_R1_D' -> 'F_R1' -> 1
                level_str = col.split('_')[0][3:] # 从 'F_R' 后面开始取
                level = int(level_str)
            except (ValueError, IndexError):
                logger.warning(f"无法从列名 '{col}' 提取 Fibonacci 阻力级别，使用默认级别 1。")
                level = 1

            # 突破信号的评分值 (示例：基础分 75 + 级别加成，Fibonacci 级别可能更重要)
            fib_breakout_score_value = 75 + level * 5
            score.loc[fib_buy_breakout] = np.maximum(score.loc[fib_buy_breakout], fib_breakout_score_value)


        # --- 添加对 Fibonacci 支撑水平 (F_S1, F_S2, ...) 的处理 ---
        elif col.startswith('F_S') and '_' in col: # 确保是带时间框架后缀的列
            # 价格向下跌破 Fibonacci 支撑 (看跌信号)
            fib_support_series = pivot_levels_filled_df[col]
            # 确保用于比较的 Series 没有 NaN
            fib_support_series_filled = fib_support_series.fillna(close_filled) # 填充为收盘价，避免NaN比较
            fib_sell_breakdown = (close_filled.shift(1) > fib_support_series_filled.shift(1)) & (close_filled <= fib_support_series_filled)

            # 跌破级别越高，信号越强 (示例权重)
            try:
                # 从列名中提取级别，例如 'F_S1_D' -> 'F_S1' -> 1
                level_str = col.split('_')[0][3:] # 从 'F_S' 后面开始取
                level = int(level_str)
            except (ValueError, IndexError):
                logger.warning(f"无法从列名 '{col}' 提取 Fibonacci 支撑级别，使用默认级别 1。")
                level = 1

            # 跌破信号的评分值 (示例：基础分 25 - 级别加成)
            fib_breakdown_score_value = 25 - level * 5
            score.loc[fib_sell_breakdown] = np.minimum(score.loc[fib_sell_breakdown], fib_breakdown_score_value)


    # 确保所有未被上述规则覆盖的区域默认为50
    # score.loc[score == 50.0] = 50.0 # 显式保留未修改的默认值

    # 最终将评分限制在 0-100 范围内
    return score.clip(0, 100)


def adjust_score_with_volume(preliminary_score: pd.Series,
                             data: pd.DataFrame,
                             vc_params: Dict
                             ) -> pd.DataFrame:
    """
    使用量能指标调整初步的 0-100 分数。

    改进：
    1. 简化函数参数，仅接收 score Series, 完整 Data 和 vc_params。
    2. 使用 vc_params 获取所需的量能指标周期和时间框架。
    3. 调整逻辑更清晰。
    4. 返回包含调整后分数和量能分析中间结果的 DataFrame。

    Args:
        preliminary_score (pd.Series): 未经量能调整的基础分数 Series (0-100)。
        data (pd.DataFrame): 包含所需列的 DataFrame (价格, 量, CMF, OBV 等)。
        vc_params (Dict): volume_confirmation 参数字典，包含:
            'enabled': bool, 是否启用量能调整。
            'tf': str, 使用哪个时间框架的量能数据。
            'boost_factor': float, 量能确认时的增强因子 (>1)。
            'penalty_factor': float, 量能矛盾时的惩罚因子 (<1)。
            'volume_spike_threshold': float, 成交量突增的倍数阈值 (与均值比)。
            'volume_spike_window': int, 计算成交量均值的窗口期。
            'cmf_period', 'obv_ma_period', 'amount_ma_period': int, 相关指标周期。
             # divergence 参数已移至 detect_divergence
            'volume_analysis_lookback': int, 用于量能趋势和突增的回看期。
            'volume_analysis_enabled': bool, 是否启用量能分析（即使不用于分数调整，也计算并返回分析列）。

    Returns:
        pd.DataFrame: 返回一个 DataFrame，包含:
            - 'ADJUSTED_SCORE': 经过量能调整后的分数 (0-100)。
            - 'VOL_CONFIRM_SIGNAL_{TF}': 量能确认信号 (1: 支持, -1: 矛盾, 0: 中性)
            - 'VOL_SPIKE_SIGNAL_{TF}': 量能突增信号 (1: 突增, 0: 正常)
            - 'VOL_PRICE_DIV_SIGNAL_{TF}': 简单量价背离信号 (1: 底背离, -1: 顶背离, 0: 无)
    """
    # 初始化结果 DataFrame，确保索引与输入一致
    result_df = pd.DataFrame(index=preliminary_score.index)
    result_df['ADJUSTED_SCORE'] = preliminary_score.copy() # 初始时调整分数等于原始分数

    # 检查是否启用量能分析或调整
    analysis_enabled = vc_params.get('enabled', False) or vc_params.get('volume_analysis_enabled', False)
    if not analysis_enabled:
        logger.debug("参数中已禁用量能调整和分析，直接返回原始分数。")
        result_df['VOL_CONFIRM_SIGNAL_N/A'] = 0 # 添加默认列，防止下游出错
        result_df['VOL_SPIKE_SIGNAL_N/A'] = 0
        result_df['VOL_PRICE_DIV_SIGNAL_N/A'] = 0
        return result_df

    # 提取配置参数
    vol_tf = vc_params.get('tf', '15')
    boost = vc_params.get('boost_factor', 1.15)
    penalty = vc_params.get('penalty_factor', 0.85)
    volume_spike_threshold = vc_params.get('volume_spike_threshold', 2.0)
    volume_analysis_lookback = vc_params.get('volume_analysis_lookback', 20) # 新增参数用于分析窗口

    # 构建所需列名
    close_col = f'close_{vol_tf}'
    high_col = f'high_{vol_tf}'
    low_col = f'low_{vol_tf}'
    volume_col = f'volume_{vol_tf}'
    # 使用 config 中的周期，如果vc_params中没有指定，则使用服务的默认值（此处使用估算值）
    cmf_period = vc_params.get('cmf_period', 20)
    obv_ma_period = vc_params.get('obv_ma_period', 10)
    amount_ma_period = vc_params.get('amount_ma_period', 20)

    amt_ma_col = f'AMT_MA_{amount_ma_period}_{vol_tf}'
    cmf_col = f'CMF_{cmf_period}_{vol_tf}'
    obv_col = f'OBV_{vol_tf}'
    obv_ma_col = f'OBV_MA_{obv_ma_period}_{vol_tf}'

    # 检查所需列是否存在且数据有效
    required_cols = [close_col, high_col, low_col, volume_col, amt_ma_col, cmf_col, obv_col, obv_ma_col]
    missing_cols = [col for col in required_cols if col not in data.columns or data[col].isnull().all()]

    if missing_cols:
        logger.warning(f"量能调整/分析缺少必需的数据列: {missing_cols} (时间框架: {vol_tf})，跳过量能调整和详细分析。")
        # 添加默认分析列，填充0
        result_df[f'VOL_CONFIRM_SIGNAL_{vol_tf}'] = 0
        result_df[f'VOL_SPIKE_SIGNAL_{vol_tf}'] = 0
        result_df[f'VOL_PRICE_DIV_SIGNAL_{vol_tf}'] = 0
        return result_df


    # 获取数据序列，确保索引对齐
    data_aligned = data.reindex(preliminary_score.index, method='ffill') # 使用ffill填充缺失的时间点
    close = data_aligned[close_col]
    high = data_aligned[high_col]
    low = data_aligned[low_col]
    volume = data_aligned[volume_col]
    amount_ma = data_aligned[amt_ma_col]
    cmf = data_aligned[cmf_col].fillna(0) # CMF可能计算结果为NaN，填充0表示无流动方向
    obv = data_aligned[obv_col]
    obv_ma = data_aligned[obv_ma_col]

    # --- 量能确认信号 ---
    # CMF > 0.1 且 OBV 在 OBV_MA 上方 视为买入量能确认
    # CMF < -0.1 且 OBV 在 OBV_MA 下方 视为卖出量能确认
    cmf_threshold = vc_params.get('cmf_confirmation_threshold', 0.1) # CMF确认阈值
    is_volume_supportive = (cmf > cmf_threshold) & (obv > obv_ma)
    is_volume_contradictory = (cmf < -cmf_threshold) & (obv < obv_ma)
    volume_confirmation_signal = pd.Series(0, index=result_df.index)
    volume_confirmation_signal.loc[is_volume_supportive] = 1
    volume_confirmation_signal.loc[is_volume_contradictory] = -1
    result_df[f'VOL_CONFIRM_SIGNAL_{vol_tf}'] = volume_confirmation_signal

    # --- 成交量突增信号 ---
    # 当前成交量超过其滚动平均的 threshold 倍
    volume_ma = volume.rolling(window=volume_analysis_lookback, min_periods=max(1, volume_analysis_lookback // 2)).mean().replace(0, np.nan) # 避免除以0
    is_volume_spike = (volume / volume_ma > volume_spike_threshold).fillna(False)
    volume_spike_signal = is_volume_spike.astype(int)
    result_df[f'VOL_SPIKE_SIGNAL_{vol_tf}'] = volume_spike_signal

    # --- 简化版量价背离信号 (基于高低点和量能趋势) ---
    # 这不是完整的背离检测，只是一个简单的启发式检查
    volume_price_div_signal = pd.Series(0, index=result_df.index)
    div_lookback = vc_params.get('volume_div_lookback', 20) # 量价背离检查回看期
    if len(data_aligned) >= div_lookback:
        # 检查最近 div_lookback 期间的高低点
        price_high_rolling = high.rolling(window=div_lookback, min_periods=1).max()
        price_low_rolling = low.rolling(window=div_lookback, min_periods=1).min()
        obv_rolling = obv.rolling(window=div_lookback, min_periods=1)
        cmf_rolling = cmf.rolling(window=div_lookback, min_periods=1)

        # 检查当前是否是最近 div_lookback 期间的最高价
        is_local_high = (high >= price_high_rolling.shift(1))
        # 检查当前是否是最近 div_lookback 期间的最低价
        is_local_low = (low <= price_low_rolling.shift(1))

        # 检查量能指标趋势 (简化：OBV 在当前位置的斜率或与 MA 的关系，CMF 的正负)
        # OBV 趋势下降 (最近 diff_period 内)
        obv_trend_down = obv.diff(periods=vc_params.get('volume_div_diff_period', 5)).fillna(0) < 0 # 检查最近5天的OBV变化
        # CMF 负值
        cmf_negative = cmf < -cmf_threshold # 使用与量能确认相同的阈值

        # 可能的顶背离：价格创新高，但 OBV 下降或 CMF 负值
        bearish_div_cond = is_local_high & (obv_trend_down | cmf_negative)

        # OBV 趋势上升
        obv_trend_up = obv.diff(periods=vc_params.get('volume_div_diff_period', 5)).fillna(0) > 0
        # CMF 正值
        cmf_positive = cmf > cmf_threshold

        # 可能的底背离：价格创新低，但 OBV 上升或 CMF 正值
        bullish_div_cond = is_local_low & (obv_trend_up | cmf_positive)

        # 信号标记
        volume_price_div_signal.loc[bearish_div_cond.fillna(False)] = -1
        volume_price_div_signal.loc[bullish_div_cond.fillna(False)] = 1

    result_df[f'VOL_PRICE_DIV_SIGNAL_{vol_tf}'] = volume_price_div_signal

    # --- 应用调整到分数 (仅当 enabled 为 True 时进行分数调整) ---
    if vc_params.get('enabled', False):
         is_bullish_score = result_df['ADJUSTED_SCORE'] > 55
         is_bearish_score = result_df['ADJUSTED_SCORE'] < 45

         # 量能确认调整
         # 看涨分数 & 量能支持 -> 分数增加 (远离50)
         confirm_bull_cond = is_bullish_score & (volume_confirmation_signal == 1)
         result_df.loc[confirm_bull_cond, 'ADJUSTED_SCORE'] = result_df.loc[confirm_bull_cond, 'ADJUSTED_SCORE'] * boost # 使分数更接近100

         # 看涨分数 & 量能矛盾 -> 分数减少 (靠近50)
         contradict_bull_cond = is_bullish_score & (volume_confirmation_signal == -1)
         result_df.loc[contradict_bull_cond, 'ADJUSTED_SCORE'] = 50 + (result_df.loc[contradict_bull_cond, 'ADJUSTED_SCORE'] - 50) * penalty # 使分数更接近50

         # 看跌分数 & 量能支持 (空头) -> 分数减少 (远离50)
         confirm_bear_cond = is_bearish_score & (volume_confirmation_signal == -1)
         result_df.loc[confirm_bear_cond, 'ADJUSTED_SCORE'] = 50 - (50 - result_df.loc[confirm_bear_cond, 'ADJUSTED_SCORE']) * boost # 使分数更接近0

         # 看跌分数 & 量能矛盾 (多头) -> 分数增加 (靠近50)
         contradict_bear_cond = is_bearish_score & (volume_confirmation_signal == 1)
         result_df.loc[contradict_bear_cond, 'ADJUSTED_SCORE'] = 50 - (50 - result_df.loc[contradict_bear_cond, 'ADJUSTED_SCORE']) * penalty # 使分数更接近50


         # 简单量价背离调整 (如果启用了这个简单的背离检测)
         divergence_penalty_factor = vc_params.get('volume_div_penalty_factor', 0.85)
         # 看涨分数 & 量价顶背离 -> 分数减少
         div_bear_cond = is_bullish_score & (volume_price_div_signal == -1)
         result_df.loc[div_bear_cond, 'ADJUSTED_SCORE'] = 50 + (result_df.loc[div_bear_cond, 'ADJUSTED_SCORE'] - 50) * divergence_penalty_factor

         # 看跌分数 & 量价底背离 -> 分数增加
         div_bull_cond = is_bearish_score & (volume_price_div_signal == 1)
         result_df.loc[div_bull_cond, 'ADJUSTED_SCORE'] = 50 - (50 - result_df.loc[div_bull_cond, 'ADJUSTED_SCORE']) * divergence_penalty_factor

         # 成交量突增调整 (可选，可能加强当前趋势或预示反转，这里简单处理为加强当前趋势)
         spike_factor = vc_params.get('volume_spike_boost_factor', 0.05) # 调整比例
         # 看涨分数 & 成交量突增 -> 分数微增
         spike_bull_cond = is_bullish_score & (volume_spike_signal == 1)
         result_df.loc[spike_bull_cond, 'ADJUSTED_SCORE'] += (100 - result_df.loc[spike_bull_cond, 'ADJUSTED_SCORE']) * spike_factor

         # 看跌分数 & 成交量突增 -> 分数微减
         spike_bear_cond = is_bearish_score & (volume_spike_signal == 1)
         result_df.loc[spike_bear_cond, 'ADJUSTED_SCORE'] -= result_df.loc[spike_bear_cond, 'ADJUSTED_SCORE'] * spike_factor

    # 确保分数在 0-100 范围内
    result_df['ADJUSTED_SCORE'] = result_df['ADJUSTED_SCORE'].clip(0, 100)
    result_df['ADJUSTED_SCORE'].fillna(50.0, inplace=True) # 如果有NaN，填充为50

    logger.info(f"时间框架 {vol_tf} 的量能调整和分析完成。")
    return result_df.fillna(0) # 确保分析信号列没有NaN

