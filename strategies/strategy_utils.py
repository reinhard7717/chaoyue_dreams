# strategy_utils.py
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

def get_find_peaks_params(time_level: str, base_lookback: int) -> Dict[str, Any]:
    """
    根据时间级别和基础回看期，返回适用于 find_peaks 的参数。
    短周期更敏感，长周期过滤更多噪音。
    """
    distance_factor = 2 # 默认峰/谷之间至少距离 lookback / 2
    prominence_factor = 0.5 # 默认显著性为 0.5 倍滚动标准差

    level_map = {
        '1': (3, 0.2),   # 1分钟: 距离更近，显著性要求更低
        '5': (2, 0.3),   # 5分钟
        '15': (2, 0.4),  # 15分钟
        '30': (1.5, 0.5),# 30分钟
        '60': (1.5, 0.6),# 60分钟
        'D': (1, 0.8),   # 日线
        'W': (1, 1.2),   # 周线
        'M': (1, 1.5),   # 月线
    }

    level_key_str = str(time_level).upper() # 统一转为大写字符串处理

    if level_key_str.isdigit(): # 处理 '1', '5', '15' 等数字级别
        level_key = level_key_str
    elif level_key_str in ['D', 'W', 'M']: # 处理 'D', 'W', 'M'
        level_key = level_key_str
    else: # 其他或未知级别使用默认值
        level_key = '15' # 假设未知级别接近15分钟

    distance_factor, prominence_factor = level_map.get(level_key, (2, 0.5))

    params = {
        'distance': max(1, base_lookback // distance_factor), # 确保 distance >= 1
        'prominence_factor': prominence_factor
    }
    return params

def find_divergence_for_indicator(price: pd.Series,
                                  indicator: pd.Series,
                                  lookback: int,
                                  find_peaks_params: Dict[str, Any],
                                  check_regular_bullish: bool,
                                  check_regular_bearish: bool,
                                  check_hidden_bullish: bool,
                                  check_hidden_bearish: bool
                                  ) -> pd.DataFrame:
    """
    辅助函数：检测单个价格序列与指标序列之间的背离。

    :param price: 价格序列 (通常是收盘价)。
    :param indicator: 指标序列。
    :param lookback: 查找峰值/谷值的回顾期。
    :param find_peaks_params: scipy.signal.find_peaks 的参数字典, 例如:
                              {'distance': 7, 'prominence_factor': 0.3, 'width': 3}。
                              'distance': 峰/谷之间的最小距离。
                              'prominence_factor': 峰/谷的最小显著性因子 (乘以滚动标准差)。
                              'width': 峰/谷的最小宽度。
    :param check_regular_bullish: 是否检测常规看涨背离。
    :param check_regular_bearish: 是否检测常规看跌背离。
    :param check_hidden_bullish: 是否检测隐藏看涨背离。
    :param check_hidden_bearish: 是否检测隐藏看跌背离。
    :return: 返回一个 DataFrame，包含四列 ('regular_bullish', 'regular_bearish',
             'hidden_bullish', 'hidden_bearish')。
             值为 1 表示检测到看涨背离，-1 表示检测到看跌背离，0 表示无。
             信号标记在第二个峰值/谷值的位置。
    """
    # 初始化结果 DataFrame
    result_df = pd.DataFrame({
        'regular_bullish': 0,
        'regular_bearish': 0,
        'hidden_bullish': 0,
        'hidden_bearish': 0
    }, index=price.index)

    # 数据有效性检查
    if price.isnull().all() or indicator.isnull().all() or len(price) < lookback * 2:
        logger.debug("价格或指标数据不足或全为 NaN，无法检测背离。")
        return result_df

    # --- 准备 find_peaks 参数 ---
    distance = find_peaks_params.get('distance', max(3, lookback // 3)) # 最小距离，至少为3
    width = find_peaks_params.get('width', max(1, distance // 2))       # 最小宽度
    prominence_factor = find_peaks_params.get('prominence_factor', 0.3) # 显著性因子

    # 计算基于滚动标准差的最小显著性 (prominence)
    min_prominence_price_series = (price.rolling(lookback).std() * prominence_factor).fillna(0)
    min_prominence_indicator_series = (indicator.rolling(lookback).std() * prominence_factor).fillna(0)

    # 填充指标序列中的 NaN 值，以便 find_peaks 可以处理
    indicator_filled = indicator.ffill().bfill()
    if indicator_filled.isnull().all(): # 如果填充后仍然全为 NaN
        logger.debug("填充后的指标序列全为 NaN，无法检测背离。")
        return result_df

    # --- 查找价格和指标的峰值 (peaks) 和谷值 (troughs) ---
    try:
        # 确保 prominence 为非负数，并处理 NaN/Inf
        min_prominence_price = np.maximum(min_prominence_price_series.values, 0)
        min_prominence_indicator = np.maximum(min_prominence_indicator_series.values, 0)
        min_prominence_price[~np.isfinite(min_prominence_price)] = 1e-9 # 用一个极小值代替 Inf/NaN
        min_prominence_indicator[~np.isfinite(min_prominence_indicator)] = 1e-9

        # 查找峰值 (高点)
        price_peaks_indices, _ = find_peaks(price.values, distance=distance, prominence=min_prominence_price, width=width)
        indicator_peaks_indices, _ = find_peaks(indicator_filled.values, distance=distance, prominence=min_prominence_indicator, width=width)

        # 查找谷值 (低点)，通过对序列取反实现
        price_troughs_indices, _ = find_peaks(-price.values, distance=distance, prominence=min_prominence_price, width=width)
        indicator_troughs_indices, _ = find_peaks(-indicator_filled.values, distance=distance, prominence=min_prominence_indicator, width=width)

    except Exception as fp_err:
        logger.warning(f"查找峰值/谷值时出错: {fp_err}。跳过此指标的背离检测。")
        return result_df

    # --- 检测背离逻辑 ---

    # -- 看跌背离 (比较峰值) --
    if len(price_peaks_indices) >= 2 and len(indicator_peaks_indices) >= 2:
        # 获取最近的两个价格峰值索引
        p_peak1_idx, p_peak2_idx = price_peaks_indices[-2], price_peaks_indices[-1]
        # 查找与这两个价格峰值时间上最接近的指标峰值索引
        window = distance # 使用 distance 作为查找窗口
        # 找到距离 p_peak1_idx 最近的 indicator_peak_idx
        indicator_peaks_near_p1 = indicator_peaks_indices[np.abs(indicator_peaks_indices - p_peak1_idx) <= window]
        # 找到距离 p_peak2_idx 最近的 indicator_peak_idx
        indicator_peaks_near_p2 = indicator_peaks_indices[np.abs(indicator_peaks_indices - p_peak2_idx) <= window]

        if len(indicator_peaks_near_p1) > 0 and len(indicator_peaks_near_p2) > 0:
            # 选择窗口内最接近的那个（或者选择窗口内最后一个）
            i_peak1_match_idx = indicator_peaks_near_p1[np.abs(indicator_peaks_near_p1 - p_peak1_idx).argmin()]
            i_peak2_match_idx = indicator_peaks_near_p2[np.abs(indicator_peaks_near_p2 - p_peak2_idx).argmin()]

            # 确保索引有效且值非 NaN
            if pd.notna(price.iloc[p_peak1_idx]) and pd.notna(price.iloc[p_peak2_idx]) and \
               pd.notna(indicator_filled.iloc[i_peak1_match_idx]) and pd.notna(indicator_filled.iloc[i_peak2_match_idx]):

                price_peak1, price_peak2 = price.iloc[p_peak1_idx], price.iloc[p_peak2_idx]
                indicator_peak1, indicator_peak2 = indicator_filled.iloc[i_peak1_match_idx], indicator_filled.iloc[i_peak2_match_idx]

                # 常规看跌背离: 价格创更高的高点 (HH), 指标创更低的高点 (LH)
                if check_regular_bearish and price_peak2 > price_peak1 and indicator_peak2 < indicator_peak1:
                    result_df.loc[price.index[p_peak2_idx], 'regular_bearish'] = -1

                # 隐藏看跌背离: 价格创更低的高点 (LH), 指标创更高的高点 (HH)
                elif check_hidden_bearish and price_peak2 < price_peak1 and indicator_peak2 > indicator_peak1:
                    # 隐藏信号优先级通常低于常规信号，如果已有常规信号则不覆盖
                    if result_df.loc[price.index[p_peak2_idx], 'regular_bearish'] == 0:
                        result_df.loc[price.index[p_peak2_idx], 'hidden_bearish'] = -1

    # -- 看涨背离 (比较谷值) --
    if len(price_troughs_indices) >= 2 and len(indicator_troughs_indices) >= 2:
        # 获取最近的两个价格谷值索引
        p_trough1_idx, p_trough2_idx = price_troughs_indices[-2], price_troughs_indices[-1]
        # 查找与这两个价格谷值时间上最接近的指标谷值索引
        window = distance
        indicator_troughs_near_p1 = indicator_troughs_indices[np.abs(indicator_troughs_indices - p_trough1_idx) <= window]
        indicator_troughs_near_p2 = indicator_troughs_indices[np.abs(indicator_troughs_indices - p_trough2_idx) <= window]

        if len(indicator_troughs_near_p1) > 0 and len(indicator_troughs_near_p2) > 0:
            i_trough1_match_idx = indicator_troughs_near_p1[np.abs(indicator_troughs_near_p1 - p_trough1_idx).argmin()]
            i_trough2_match_idx = indicator_troughs_near_p2[np.abs(indicator_troughs_near_p2 - p_trough2_idx).argmin()]

            # 确保索引有效且值非 NaN
            if pd.notna(price.iloc[p_trough1_idx]) and pd.notna(price.iloc[p_trough2_idx]) and \
               pd.notna(indicator_filled.iloc[i_trough1_match_idx]) and pd.notna(indicator_filled.iloc[i_trough2_match_idx]):

                price_trough1, price_trough2 = price.iloc[p_trough1_idx], price.iloc[p_trough2_idx]
                indicator_trough1, indicator_trough2 = indicator_filled.iloc[i_trough1_match_idx], indicator_filled.iloc[i_trough2_match_idx]

                # 常规看涨背离: 价格创更低的低点 (LL), 指标创更高的低点 (HL)
                if check_regular_bullish and price_trough2 < price_trough1 and indicator_trough2 > indicator_trough1:
                    result_df.loc[price.index[p_trough2_idx], 'regular_bullish'] = 1

                # 隐藏看涨背离: 价格创更高的低点 (HL), 指标创更低的低点 (LL)
                elif check_hidden_bullish and price_trough2 > price_trough1 and indicator_trough2 < indicator_trough1:
                    # 隐藏信号优先级通常低于常规信号
                    if result_df.loc[price.index[p_trough2_idx], 'regular_bullish'] == 0:
                        result_df.loc[price.index[p_trough2_idx], 'hidden_bullish'] = 1

    # 填充 NaN 值为 0
    result_df = result_df.fillna(0)
    return result_df.astype(int) # 确保返回整数

def detect_divergence(data: pd.DataFrame,
                      dd_params: Dict,
                      bs_params: Dict,
                      vc_params: Optional[Dict] = None) -> pd.DataFrame: # vc_params 可选，如果用到 CMF
    """
    检测价格与多个指定指标之间的常规和隐藏背离 (使用 find_divergence_for_indicator)。

    :param data: 包含价格和指标列的 DataFrame。列名格式需符合策略约定 (例如 'close_15', 'RSI_9_15')。
    :param dd_params: divergence_detection 参数字典, 包含:
                      'enabled': bool, 是否启用背离检测。
                      'tf': str, 用于检测的时间框架 (例如 '15')。
                      'price_type': str, 用于比较的价格类型 ('close', 'high', 'low'), 默认为 'close'。
                      'lookback': int, 查找峰值/谷值的回顾期。
                      'find_peaks_params': dict, 传递给 find_peaks 的参数。
                      'check_regular_bullish', 'check_regular_bearish',
                      'check_hidden_bullish', 'check_hidden_bearish': bool, 控制检测类型。
                      'indicators': dict, 指定要检查的指标及其是否启用, 例如 {'macd_hist': True, 'rsi': True}。
    :param bs_params: base_scoring 参数字典 (用于获取指标的周期等信息以构建列名)。
    :param vc_params: volume_confirmation 参数字典 (可选, 仅当需要检测 CMF 背离时提供)。
    :return: 返回一个 DataFrame，包含详细的背离信号和聚合信号。
             例如: 'div_macd_hist_regular_bullish', 'div_rsi_hidden_bearish',
                   'has_bullish_divergence', 'has_bearish_divergence' (布尔值)。
    """
    # 初始化结果 DataFrame
    all_divergence_signals = pd.DataFrame(index=data.index)
    all_divergence_signals['has_bullish_divergence'] = False
    all_divergence_signals['has_bearish_divergence'] = False

    # 检查是否启用背离检测
    if not dd_params.get('enabled', False):
        logger.info("参数中已禁用背离检测。")
        return all_divergence_signals

    # 获取配置参数
    tf = dd_params.get('tf', '15')                 # 时间框架
    price_type = dd_params.get('price_type', 'close') # 价格类型
    lookback = dd_params.get('lookback', 14)
    find_peaks_params = dd_params.get('find_peaks_params', {'distance': 7, 'prominence_factor': 0.3, 'width': 3})
    check_regular_bullish = dd_params.get('check_regular_bullish', True)
    check_regular_bearish = dd_params.get('check_regular_bearish', True)
    check_hidden_bullish = dd_params.get('check_hidden_bullish', True)
    check_hidden_bearish = dd_params.get('check_hidden_bearish', True)
    indicators_to_check = dd_params.get('indicators', {})

    # 构建价格列名并检查是否存在
    price_col = f'{price_type}_{tf}' # 例如 'close_15'
    if price_col not in data.columns or data[price_col].isnull().all():
        logger.warning(f"用于背离检测的价格列 '{price_col}' 不存在或全为 NaN。无法执行背离检测。")
        return all_divergence_signals
    price_series = data[price_col]

    # 遍历配置中指定的指标
    for indicator_key, enabled in indicators_to_check.items():
        if not enabled:
            continue # 跳过未启用的指标

        indicator_col = None
        indicator_series = None

        # --- 根据指标 key 构建指标列名 ---
        # 注意：这里的列名构建逻辑需要与你的数据准备过程严格对应
        try:
            if indicator_key == 'macd_hist':
                indicator_col = f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
            elif indicator_key == 'rsi':
                indicator_col = f'RSI_{bs_params["rsi_period"]}_{tf}'
            elif indicator_key == 'mfi':
                indicator_col = f'MFI_{bs_params["mfi_period"]}_{tf}'
            elif indicator_key == 'obv':
                indicator_col = f'OBV_{tf}' # OBV 通常不带参数
            elif indicator_key == 'cci':
                indicator_col = f'CCI_{bs_params["cci_period"]}_{tf}'
            elif indicator_key == 'cmf':
                if vc_params: # 需要 vc_params 来获取 CMF 周期
                    indicator_col = f'CMF_{vc_params.get("cmf_period", 20)}_{tf}'
                else:
                    logger.warning(f"检测 CMF 背离需要 vc_params，但未提供。跳过 CMF。")
                    continue
            # --- 在此添加其他需要检测背离的指标 ---
            else:
                logger.warning(f"未知的指标 key '{indicator_key}' 用于背离检测。")
                continue
        except KeyError as e:
             logger.warning(f"构建指标 '{indicator_key}' 列名时缺少参数: {e}。跳过此指标。")
             continue

        # 检查指标列是否存在且有效
        if indicator_col and indicator_col in data and not data[indicator_col].isnull().all():
            indicator_series = data[indicator_col]
        else:
            logger.warning(f"指标 '{indicator_key}' 的列 '{indicator_col}' 在时间框架 {tf} 不存在、全为 NaN 或未启用。跳过其背离检测。")
            continue

    # --- 调用辅助函数进行单指标背离检测 ---
        logger.debug(f"开始检测价格 ('{price_col}') 与指标 ('{indicator_col}') 的背离...")
        div_result = find_divergence_for_indicator(
            price=price_series,
            indicator=indicator_series,
            lookback=lookback,
            find_peaks_params=find_peaks_params,
            check_regular_bullish=check_regular_bullish,
            check_regular_bearish=check_regular_bearish,
            check_hidden_bullish=check_hidden_bullish,
            check_hidden_bearish=check_hidden_bearish
        )

        # 将检测结果合并到总的 DataFrame 中，并添加前缀 'div_'
        for div_type in div_result.columns:
            # 结果列名示例: 'div_rsi_regular_bullish'
            all_divergence_signals[f'div_{indicator_key}_{div_type}'] = div_result[div_type]

    # --- 聚合所有指标的看涨和看跌信号 ---
    # 查找所有包含 'bullish' 且值 > 0 的列
    bullish_cols = [col for col in all_divergence_signals.columns if 'bullish' in col]
    if bullish_cols:
        all_divergence_signals['has_bullish_divergence'] = (all_divergence_signals[bullish_cols] > 0).any(axis=1)

    # 查找所有包含 'bearish' 且值 < 0 的列
    bearish_cols = [col for col in all_divergence_signals.columns if 'bearish' in col]
    if bearish_cols:
        all_divergence_signals['has_bearish_divergence'] = (all_divergence_signals[bearish_cols] < 0).any(axis=1)

    logger.debug(f"时间框架 {tf} 的背离检测完成。")
    return all_divergence_signals

def detect_kline_patterns(df: pd.DataFrame) -> pd.Series:
    """
    检测 K 线形态。
    信号值:
     1: 看涨吞没 (Bullish Engulfing)
    -1: 看跌吞没 (Bearish Engulfing)
     2: 锤子线 (Hammer)
    -2: 上吊线 (Hanging Man)
     3: 早晨之星 (Morning Star)
    -3: 黄昏之星 (Evening Star)
     4: 刺透线 (Piercing Line)
    -4: 乌云盖顶 (Dark Cloud Cover)
     5: 十字星 (Doji) - 优先级较低
     6: 红三兵 (Three White Soldiers)
    -6: 三只乌鸦 (Three Black Crows)
     7: 看涨孕线 (Bullish Harami)
    -7: 看跌孕线 (Bearish Harami)
     8: 看涨十字孕线 (Bullish Harami Cross) - 优先级高于普通孕线
    -8: 看跌十字孕线 (Bearish Harami Cross) - 优先级高于普通孕线
     9: 镊子底 (Tweezer Bottom) - 优先级较低
    -9: 镊子顶 (Tweezer Top) - 优先级较低
    10: 光头阳线 (Bullish Marubozu) - 优先级高
    -10: 光头阴线 (Bearish Marubozu) - 优先级高
    11: 上升三法 (Rising Three Methods)
    -11: 下降三法 (Falling Three Methods)
    -12: 向上跳空两只乌鸦 (Upside Gap Two Crows) - 看跌反转
    13: 向上跳空并列阳线 (Upside Tasuki Gap)
    -13: 向下跳空并列阴线 (Downside Tasuki Gap)
    14: 看涨分离线 (Bullish Separating Lines)
    -14: 看跌分离线 (Bearish Separating Lines)
    15: 看涨反击线 (Bullish Counterattack)
    -15: 看跌反击线 (Bearish Counterattack)
     0: 无显著形态
    """
    patterns = pd.Series(0, index=df.index)
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.warning("K-line detection requires OHLC columns.")
        return patterns
    df_ohlc = df[required_cols].copy()
    for col in required_cols:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
    df_ohlc = df_ohlc.dropna()
    if len(df_ohlc) < 5: # 某些新形态需要至少5根K线
        logger.warning("数据点不足 (<5)，无法检测所有K线形态。")
        # 仍然尝试计算需要较少K线的形态

    o, h, l, c = df_ohlc['open'], df_ohlc['high'], df_ohlc['low'], df_ohlc['close']
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, h2, l2, c2 = o.shift(2), h.shift(2), l.shift(2), c.shift(2)
    o3, h3, l3, c3 = o.shift(3), h.shift(3), l.shift(3), c.shift(3)
    o4, h4, l4, c4 = o.shift(4), h.shift(4), l.shift(4), c.shift(4) # 为三法准备

    body = abs(c - o)
    body1 = abs(c1 - o1).fillna(0)
    body2 = abs(c2 - o2).fillna(0)
    body3 = abs(c3 - o3).fillna(0)
    body4 = abs(c4 - o4).fillna(0) # 为三法准备

    full_range = (h - l).replace(0, 1e-6)
    full_range1 = (h1 - l1).fillna(0).replace(0, 1e-6)
    full_range2 = (h2 - l2).fillna(0).replace(0, 1e-6)
    full_range3 = (h3 - l3).fillna(0).replace(0, 1e-6) # 为三法准备
    full_range4 = (h4 - l4).fillna(0).replace(0, 1e-6) # 为三法准备

    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(o, c) - l
    upper_shadow1 = h1 - np.maximum(c1, o1)
    lower_shadow1 = np.minimum(o1, c1) - l1

    is_green = c > o
    is_red = c < o
    is_green1 = c1 > o1
    is_red1 = c1 < o1
    is_green2 = c2 > o2
    is_red2 = c2 < o2
    is_green3 = c3 > o3
    is_red3 = c3 < o3
    is_green4 = c4 > o4 # 为三法准备
    is_red4 = c4 < o4 # 为三法准备

    avg_body = body.rolling(10).mean().fillna(1e-6) # 近期平均实体大小
    avg_range = full_range.rolling(10).mean().fillna(1e-6) # 近期平均波动范围

    # --- 十字星 (Doji) ---
    doji_threshold = avg_range * 0.1 # 实体小于平均波幅的10%
    is_doji = (body <= doji_threshold) & (body > 1e-6)
    is_doji1 = (body1 <= avg_range * 0.1) & (body1 > 1e-6) # 前一根是十字星

    # --- 吞没 (Engulfing) ---
    bull_engulf = is_red1 & is_green & (c > o1) & (o < c1) & (body > body1 * 1.01)
    patterns.loc[bull_engulf[bull_engulf].index] = 1
    bear_engulf = is_green1 & is_red & (o > c1) & (c < o1) & (body > body1 * 1.01)
    patterns.loc[bear_engulf[bear_engulf].index] = -1

    # --- 锤子/上吊 (Hammer/Hanging Man) ---
    small_body_threshold = avg_range * 0.3
    long_lower_shadow = lower_shadow >= 2 * body
    short_upper_shadow = upper_shadow < body * 0.5 # 上影线小于实体一半
    hammer_like = (body > 1e-6) & (body < small_body_threshold) & long_lower_shadow & short_upper_shadow
    is_hammer = hammer_like & (is_red1 | (c1 < c2))
    patterns.loc[is_hammer[is_hammer].index] = 2
    is_hanging = hammer_like & (is_green1 | (c1 > c2))
    patterns.loc[is_hanging[is_hanging].index] = -2

    # --- 星线 (Morning/Evening Star) ---
    star_body_threshold = avg_range * 0.3 # 中继实体不超过平均波幅30%
    is_star1 = (body1 < star_body_threshold) & (body1 > 1e-6)
    gap_down1 = np.where(is_red2, np.maximum(o1, c1) < c2, np.maximum(o1, c1) < o2)
    gap_up1 = np.where(is_green2, np.minimum(o1, c1) > c2, np.minimum(o1, c1) > o2)
    gap_down2 = np.minimum(o, c) < np.maximum(o1, c1)
    gap_up2 = np.maximum(o, c) > np.minimum(o1, c1)
    morning_star = is_red2 & (body2 > avg_body) & is_star1 & gap_down1 & is_green & (body > body2 * 0.5) & gap_up2 & (c > (o2 + c2) / 2)
    patterns.loc[morning_star[morning_star].index] = 3
    evening_star = is_green2 & (body2 > avg_body) & is_star1 & gap_up1 & is_red & (body > body2 * 0.5) & gap_down2 & (c < (o2 + c2) / 2)
    patterns.loc[evening_star[evening_star].index] = -3

    # --- 刺透线/乌云盖顶 (Piercing/Dark Cloud) ---
    piercing = is_red1 & (body1 > avg_body) & is_green & (o < l1) & (c > (o1 + c1) / 2) & (c < o1)
    patterns.loc[piercing[piercing].index] = 4
    dark_cloud = is_green1 & (body1 > avg_body) & is_red & (o > h1) & (c < (o1 + c1) / 2) & (c > o1)
    patterns.loc[dark_cloud[dark_cloud].index] = -4

    # --- 十字星 (Doji) - 赋予较低优先级 ---
    patterns.loc[is_doji & (patterns == 0)] = 5

    # --- 三兵/三鸦 (Three Soldiers/Crows) ---
    # 红三兵: 连续三阳，逐步抬高，开盘在前实体，收盘创新高，实体不能太小
    soldiers = is_green & (body > avg_body * 0.7) & (c > c1) & (o < c1) & (o > o1) & \
               is_green1 & (body1 > avg_body * 0.7) & (c1 > c2) & (o1 < c2) & (o1 > o2) & \
               is_green2 & (body2 > avg_body * 0.7) & (c2 > c3 if len(df_ohlc)>3 else True) # 检查趋势背景
    patterns.loc[soldiers[soldiers].index] = 6
    # 三只乌鸦: 连续三阴，逐步降低，开盘在前实体，收盘创新低，实体不能太小
    crows = is_red & (body > avg_body * 0.7) & (c < c1) & (o > c1) & (o < o1) & \
            is_red1 & (body1 > avg_body * 0.7) & (c1 < c2) & (o1 > c2) & (o1 < o2) & \
            is_red2 & (body2 > avg_body * 0.7) & (c2 < c3 if len(df_ohlc)>3 else True) # 检查趋势背景
    patterns.loc[crows[crows].index] = -6

    # --- 孕线 (Harami) ---
    is_harami_body = (np.maximum(o, c) < np.maximum(o1, c1)) & (np.minimum(o, c) > np.minimum(o1, c1))
    bullish_harami = is_red1 & (body1 > avg_body) & is_harami_body & (is_green | is_doji)
    patterns.loc[bullish_harami[bullish_harami].index] = 7
    bearish_harami = is_green1 & (body1 > avg_body) & is_harami_body & (is_red | is_doji)
    patterns.loc[bearish_harami[bearish_harami].index] = -7
    # --- 十字孕线 (Harami Cross) ---
    bullish_harami_cross = is_red1 & (body1 > avg_body) & is_harami_body & is_doji
    patterns.loc[bullish_harami_cross[bullish_harami_cross].index] = 8 # 覆盖普通孕线
    bearish_harami_cross = is_green1 & (body1 > avg_body) & is_harami_body & is_doji
    patterns.loc[bearish_harami_cross[bearish_harami_cross].index] = -8 # 覆盖普通孕线
    # --- 镊子顶/底 (Tweezers Top/Bottom) ---
    tweezer_tolerance = avg_range * 0.05
    tweezer_bottom = abs(l - l1) < tweezer_tolerance
    tweezer_top = abs(h - h1) < tweezer_tolerance
    patterns.loc[tweezer_bottom & (is_red1 | (c1 < o1)) & (patterns == 0)] = 9
    patterns.loc[tweezer_top & (is_green1 | (c1 > o1)) & (patterns == 0)] = -9
    # --- 上升/下降三法 (Rising/Falling Three Methods) --- 需要5根K线
    # 上升三法: 长阳 + 三个小阴(或混合色)回调(在第一根长阳范围内) + 长阳(收盘超第一根)
    rising_three = is_green4 & (body4 > avg_body * 1.5) & \
                   (is_red3 | body3 < avg_body * 0.5) & (h3 < h4) & (l3 > l4) & \
                   (is_red2 | body2 < avg_body * 0.5) & (h2 < h4) & (l2 > l4) & \
                   (is_red1 | body1 < avg_body * 0.5) & (h1 < h4) & (l1 > l4) & \
                   is_green & (body > avg_body * 1.5) & (c > c4) & (o > l1) # 第五根阳线突破
    patterns.loc[rising_three[rising_three].index] = 11
    # 下降三法: 长阴 + 三个小阳(或混合色)反弹(在第一根长阴范围内) + 长阴(收盘低于第一根)
    falling_three = is_red4 & (body4 > avg_body * 1.5) & \
                    (is_green3 | body3 < avg_body * 0.5) & (h3 < h4) & (l3 > l4) & \
                    (is_green2 | body2 < avg_body * 0.5) & (h2 < h4) & (l2 > l4) & \
                    (is_green1 | body1 < avg_body * 0.5) & (h1 < h4) & (l1 > l4) & \
                    is_red & (body > avg_body * 1.5) & (c < c4) & (o < h1) # 第五根阴线突破
    patterns.loc[falling_three[falling_three].index] = -11
    # --- 向上跳空两只乌鸦 (Upside Gap Two Crows) --- 看跌反转，需要3根K线
    # 条件: 强阳线 + 向上跳空的小阴线 + 更大的阴线(开盘高于前阴，收盘低于前阴，且吞没前阴)
    upside_gap_two_crows = is_green2 & (body2 > avg_body) & \
                          is_red1 & (body1 < avg_body * 0.7) & (o1 > h2) & \
                          is_red & (o > o1) & (c < c1) & (c < h2) # 实体吞没小阴线，收盘低于第一天高点
    patterns.loc[upside_gap_two_crows[upside_gap_two_crows].index] = -12
    # --- 跳空并列线 (Tasuki Gap) --- 需要3根K线
    # 向上跳空并列阳线: 阳线 + 向上跳空阳线 + 阴线(开盘在前阳实体，收盘在缺口内) - 看涨持续
    upside_tasuki_gap = is_green2 & \
                        is_green1 & (o1 > h2) & \
                        is_red & (o > o1) & (o < c1) & (c < o1) & (c > h2) # 收盘填补部分缺口
    patterns.loc[upside_tasuki_gap[upside_tasuki_gap].index] = 13
    # 向下跳空并列阴线: 阴线 + 向下跳空阴线 + 阳线(开盘在前阴实体，收盘在缺口内) - 看跌持续
    downside_tasuki_gap = is_red2 & \
                         is_red1 & (o1 < l2) & \
                         is_green & (o < o1) & (o > c1) & (c > o1) & (c < l2) # 收盘填补部分缺口
    patterns.loc[downside_tasuki_gap[downside_tasuki_gap].index] = -13
    # --- 分离线 (Separating Lines) ---
    # 条件：颜色相反，开盘价相同
    same_open = abs(o - o1) < avg_range * 0.02 # 开盘价几乎相同
    # 看涨分离线: 下降趋势中，前阴后阳，开盘相同
    bullish_sep_lines = is_red1 & is_green & same_open & (c1 < o1) # 简单趋势判断
    patterns.loc[bullish_sep_lines[bullish_sep_lines].index] = 14
    # 看跌分离线: 上升趋势中，前阳后阴，开盘相同
    bearish_sep_lines = is_green1 & is_red & same_open & (c1 > o1) # 简单趋势判断
    patterns.loc[bearish_sep_lines[bearish_sep_lines].index] = -14
    # --- 反击线 (Counterattack Lines) ---
    # 条件：颜色相反，收盘价相同
    same_close = abs(c - c1) < avg_range * 0.02 # 收盘价几乎相同
    # 看涨反击线: 下降趋势中，前长阴后长阳(大幅跳空低开)，收盘相同
    bullish_counter = is_red1 & (body1 > avg_body) & is_green & (body > avg_body) & same_close & (o < l1) # 跳空低开
    patterns.loc[bullish_counter[bullish_counter].index] = 15
    # 看跌反击线: 上升趋势中，前长阳后长阴(大幅跳空高开)，收盘相同
    bearish_counter = is_green1 & (body1 > avg_body) & is_red & (body > avg_body) & same_close & (o > h1) # 跳空高开
    patterns.loc[bearish_counter[bearish_counter].index] = -15
    # --- 光头光脚 (Marubozu) - 优先级最高 ---
    shadow_threshold_factor = 0.05
    no_upper_shadow = upper_shadow < body * shadow_threshold_factor
    no_lower_shadow = lower_shadow < body * shadow_threshold_factor
    is_marubozu = no_upper_shadow & no_lower_shadow & (body > full_range * 0.95)
    bull_marubozu = is_marubozu & is_green
    patterns.loc[bull_marubozu[bull_marubozu].index] = 10 # 覆盖其他形态
    bear_marubozu = is_marubozu & is_red
    patterns.loc[bear_marubozu[bear_marubozu].index] = -10 # 覆盖其他形态
    # 重新对齐索引并填充 NaN
    patterns_aligned = pd.Series(0, index=df.index)
    patterns_aligned.update(patterns)
    return patterns_aligned.astype(int)

# --- 指标评分函数 (原 _get_xxx_score 改为公用) ---

def calculate_macd_score(diff: pd.Series, dea: pd.Series, macd: pd.Series) -> pd.Series:
    """MACD 评分 (0-100)"""
    score = pd.Series(50.0, index=diff.index)
    buy_cross = (macd.shift(1) < 0) & (macd > 0)
    score.loc[buy_cross] = 75.0
    sell_cross = (macd.shift(1) > 0) & (macd < 0)
    score.loc[sell_cross] = 25.0
    # 可根据需要添加更多条件，例如零轴上方/下方
    # 趋势加强信号
    bullish_trend = (macd > 0) & (macd > macd.shift(1)) & (diff > dea)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 65.0) # 若非金叉，则设为65
    bearish_trend = (macd < 0) & (macd < macd.shift(1)) & (diff < dea)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 35.0) # 若非死叉，则设为35
    return score

def calculate_rsi_score(rsi: pd.Series, params: Dict) -> pd.Series:
    """RSI 评分 (0-100)"""
    score = pd.Series(50.0, index=rsi.index)
    p = params # 使用传入的参数字典
    os, ob = p.get('rsi_oversold', 30), p.get('rsi_overbought', 70)
    ext_os, ext_ob = p.get('rsi_extreme_oversold', 20), p.get('rsi_extreme_overbought', 80)

    score.loc[rsi < ext_os] = 95.0
    score.loc[(rsi >= ext_os) & (rsi < os)] = 85.0
    buy_signal = (rsi.shift(1) < os) & (rsi >= os)
    score.loc[buy_signal] = 75.0

    score.loc[rsi > ext_ob] = 5.0
    score.loc[(rsi <= ext_ob) & (rsi > ob)] = 15.0
    sell_signal = (rsi.shift(1) > ob) & (rsi <= ob)
    score.loc[sell_signal] = 25.0

    neutral_zone = (rsi >= os) & (rsi <= ob) & (~buy_signal) & (~sell_signal)
    score.loc[neutral_zone] = 50.0
    return score

def calculate_kdj_score(k: pd.Series, d: pd.Series, j: pd.Series, params: Dict) -> pd.Series:
    """KDJ 评分 (0-100)"""
    score = pd.Series(50.0, index=k.index)
    p = params # 使用传入的参数字典
    os, ob = p.get('kdj_oversold', 20), p.get('kdj_overbought', 80)

    score.loc[j < os] = 85.0
    score.loc[j < 10] = 95.0 # 极度超卖
    buy_cross = (k.shift(1) < d.shift(1)) & (k > d) & (j < ob) # 金叉发生在非超买区

    score.loc[j > ob] = 15.0
    score.loc[j > 90] = 5.0 # 极度超买
    sell_cross = (k.shift(1) > d.shift(1)) & (k < d) & (j > os) # 死叉发生在非超卖区

    # 交叉信号优先
    score.loc[buy_cross] = 75.0
    score.loc[sell_cross] = 25.0
    return score

def calculate_boll_score(close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
    """BOLL 评分 (0-100)"""
    score = pd.Series(50.0, index=close.index)
    score.loc[close < lower] = 90.0 # 触及下轨 - 超卖区
    buy_support = (close.shift(1) < lower.shift(1)) & (close >= lower) # 突破下轨后收回
    score.loc[buy_support] = 80.0

    score.loc[close > upper] = 10.0 # 触及上轨 - 超买区
    sell_pressure = (close.shift(1) > upper.shift(1)) & (close <= upper) # 突破上轨后收回
    score.loc[sell_pressure] = 20.0

    buy_mid_cross = (close.shift(1) < mid.shift(1)) & (close > mid) # 向上突破中轨
    score.loc[buy_mid_cross] = 65.0
    sell_mid_cross = (close.shift(1) > mid.shift(1)) & (close < mid) # 向下跌破中轨
    score.loc[sell_mid_cross] = 35.0

    # 在轨道内的评分
    is_signal = buy_support | sell_pressure | buy_mid_cross | sell_mid_cross
    score.loc[(~is_signal) & (close > mid) & (close < upper)] = 55.0 # 中轨上方
    score.loc[(~is_signal) & (close < mid) & (close > lower)] = 45.0 # 中轨下方
    return score

def calculate_cci_score(cci: pd.Series, params: Dict) -> pd.Series:
    """CCI 评分 (0-100)"""
    score = pd.Series(50.0, index=cci.index)
    p = params # 使用传入的参数字典
    threshold, ext_threshold = p.get('cci_threshold', 100), p.get('cci_extreme_threshold', 200)

    score.loc[cci < -ext_threshold] = 95.0
    score.loc[(cci >= -ext_threshold) & (cci < -threshold)] = 85.0
    buy_signal = (cci.shift(1) < -threshold) & (cci >= -threshold)
    score.loc[buy_signal] = 75.0

    score.loc[cci > ext_threshold] = 5.0
    score.loc[(cci <= ext_threshold) & (cci > threshold)] = 15.0
    sell_signal = (cci.shift(1) > threshold) & (cci <= threshold)
    score.loc[sell_signal] = 25.0

    neutral_zone = (cci >= -threshold) & (cci <= threshold) & (~buy_signal) & (~sell_signal)
    score.loc[neutral_zone] = 50.0
    return score

def calculate_mfi_score(mfi: pd.Series, params: Dict) -> pd.Series:
    """MFI 评分 (0-100)"""
    score = pd.Series(50.0, index=mfi.index)
    p = params # 使用传入的参数字典
    os, ob = p.get('mfi_oversold', 20), p.get('mfi_overbought', 80)
    ext_os, ext_ob = p.get('mfi_extreme_oversold', 10), p.get('mfi_extreme_overbought', 90)

    score.loc[mfi < ext_os] = 95.0
    score.loc[(mfi >= ext_os) & (mfi < os)] = 85.0
    buy_signal = (mfi.shift(1) < os) & (mfi >= os)
    score.loc[buy_signal] = 75.0

    score.loc[mfi > ext_ob] = 5.0
    score.loc[(mfi <= ext_ob) & (mfi > ob)] = 15.0
    sell_signal = (mfi.shift(1) > ob) & (mfi <= ob)
    score.loc[sell_signal] = 25.0

    neutral_zone = (mfi >= os) & (mfi <= ob) & (~buy_signal) & (~sell_signal)
    score.loc[neutral_zone] = 50.0
    return score

def calculate_roc_score(roc: pd.Series) -> pd.Series:
    """ROC 评分 (0-100)"""
    score = pd.Series(50.0, index=roc.index)
    buy_signal = (roc.shift(1) < 0) & (roc > 0) # 上穿 0 轴
    score.loc[buy_signal] = 70.0
    sell_signal = (roc.shift(1) > 0) & (roc < 0) # 下穿 0 轴
    score.loc[sell_signal] = 30.0

    # 趋势加强信号 (在0轴同侧且持续)
    bullish_trend = (roc > 0) & (roc > roc.shift(1)) & (~buy_signal)
    score.loc[bullish_trend] = np.maximum(score.loc[bullish_trend], 60.0) # 保持在 60 或更高
    bearish_trend = (roc < 0) & (roc < roc.shift(1)) & (~sell_signal)
    score.loc[bearish_trend] = np.minimum(score.loc[bearish_trend], 40.0) # 保持在 40 或更低
    return score

def calculate_dmi_score(pdi: pd.Series, mdi: pd.Series, adx: pd.Series, params: Dict) -> pd.Series:
    """DMI 评分 (0-100)"""
    score = pd.Series(50.0, index=pdi.index)
    p = params # 使用传入的参数字典
    adx_th, adx_strong_th = p.get('adx_threshold', 25), p.get('adx_strong_threshold', 40)

    buy_cross = (pdi.shift(1) < mdi.shift(1)) & (pdi > mdi) # PDI 上穿 MDI
    sell_cross = (mdi.shift(1) < pdi.shift(1)) & (mdi > pdi) # MDI 上穿 PDI

    # 金叉评分
    score.loc[buy_cross & (adx > adx_th)] = 75.0
    score.loc[buy_cross & (adx > adx_strong_th)] = 85.0 # 强趋势下的金叉
    # 死叉评分
    score.loc[sell_cross & (adx > adx_th)] = 25.0
    score.loc[sell_cross & (adx > adx_strong_th)] = 15.0 # 强趋势下的死叉

    # 非交叉时的趋势评分
    is_bullish = (pdi > mdi) & (~buy_cross)
    score.loc[is_bullish & (adx > adx_strong_th)] = 65.0 # 强多头趋势
    score.loc[is_bullish & (adx <= adx_strong_th) & (adx > adx_th)] = 60.0 # 普通多头趋势
    score.loc[is_bullish & (adx <= adx_th)] = 55.0 # 弱多头趋势或无趋势

    is_bearish = (mdi > pdi) & (~sell_cross)
    score.loc[is_bearish & (adx > adx_strong_th)] = 35.0 # 强空头趋势
    score.loc[is_bearish & (adx <= adx_strong_th) & (adx > adx_th)] = 40.0 # 普通空头趋势
    score.loc[is_bearish & (adx <= adx_th)] = 45.0 # 弱空头趋势或无趋势
    return score

def calculate_sar_score(close: pd.Series, sar: pd.Series) -> pd.Series:
    """SAR 评分 (0-100)"""
    score = pd.Series(50.0, index=close.index)
    buy_signal = (sar.shift(1) > close.shift(1)) & (sar < close) # 向上反转
    score.loc[buy_signal] = 75.0
    sell_signal = (sar.shift(1) < close.shift(1)) & (sar > close) # 向下反转
    score.loc[sell_signal] = 25.0

    # 持续状态
    score.loc[(close > sar) & (~buy_signal)] = 60.0 # 价格在 SAR 上方
    score.loc[(close < sar) & (~sell_signal)] = 40.0 # 价格在 SAR 下方
    return score

def adjust_score_with_volume(preliminary_score: pd.Series,
                             data: pd.DataFrame,
                             vc_params: Dict,
                             dd_params: Dict,
                             bs_params: Dict,
                             return_analysis: bool = True
                             ) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    使用量能指标调整初步的 0-100 分数，并选择性返回量能分析的中间结果。

    调整逻辑:
    1. **量能确认**: 检查 CMF 和 OBV 趋势是否支持当前分数代表的趋势方向。
       - 支持（例如，看涨分数且 CMF>0, OBV>OBV_MA）-> 增强分数（更接近0或100）。
       - 矛盾（例如，看涨分数但 CMF<0, OBV<OBV_MA）-> 削弱分数（向50回归）。
    2. **成交量突增**: 检测成交量是否显著高于近期平均水平。
       - 信号：1 表示放量，0 表示正常。
    3. **量价背离(简化版)**: 检测价格创新高/低，但量能指标（如 CMF, OBV）未跟上。
       - 信号：-1 表示可能的顶背离，1 表示可能的底背离，0 表示无。

    Args:
        preliminary_score (pd.Series): 未经量能调整的基础分数 Series (0-100)。
        data (pd.DataFrame): 包含所需列的 DataFrame (价格, 量, CMF, OBV, MFI 等)。
        vc_params (Dict): volume_confirmation 参数字典，包含:
            'enabled': bool, 是否启用量能调整。
            'tf': str, 使用哪个时间框架的量能数据。
            'boost_factor': float, 量能确认时的增强因子 (>1)。
            'penalty_factor': float, 量能矛盾时的惩罚因子 (<1)。
            'volume_spike_threshold': float, 成交量突增的倍数阈值 (与均值比)。
            'volume_spike_window': int, 计算成交量均值的窗口期。
            'cmf_period', 'obv_ma_period', 'amount_ma_period': int, 相关指标周期。
        dd_params (Dict): divergence_detection 参数字典，用于量价背离检查，包含:
            'enabled': bool, 是否启用背离检测 (影响此处的量价背离)。
            'price_period': int, 用于查找价格高/低点的回顾期。
            'divergence_penalty_factor': float, 量价背离的惩罚因子 (<1)。
            'indicators': dict, (可选) 如果 MFI 用于量价背离检查。
        bs_params (Dict): base_scoring 参数字典，用于获取 MFI 等指标的周期。
        return_analysis (bool): 是否返回包含量能分析中间结果的 DataFrame。

    Returns:
        Tuple[pd.Series, Optional[pd.DataFrame]]: 返回一个元组:
            - pd.Series: 经过量能调整后的分数 (0-100)。
            - Optional[pd.DataFrame]: 如果 return_analysis=True，则包含量能分析结果
              (列: 'volume_confirmation_signal', 'volume_spike_signal',
              'volume_bearish_divergence_signal', 'volume_bullish_divergence_signal')；
              否则返回 None。
    """
    # 初始化调整后的分数，确保索引与输入一致
    adjusted_score = preliminary_score.copy()
    # 初始化分析结果 DataFrame，确保索引一致
    volume_analysis_df = pd.DataFrame(index=preliminary_score.index)

    # 检查是否启用量能调整，若未启用则直接返回默认值
    if not vc_params.get('enabled', False):
        logger.debug("参数中已禁用量能调整，直接返回原始分数。")
        if return_analysis:
            volume_analysis_df['volume_confirmation_signal'] = 0
            volume_analysis_df['volume_spike_signal'] = 0
            volume_analysis_df['volume_bearish_divergence_signal'] = 0
            volume_analysis_df['volume_bullish_divergence_signal'] = 0
            return adjusted_score, volume_analysis_df
        return adjusted_score, None

    # 提取配置参数，减少字典访问次数以优化性能
    vol_tf = vc_params.get('tf', '15')
    boost = vc_params.get('boost_factor', 1.15)
    penalty = vc_params.get('penalty_factor', 0.85)
    volume_spike_threshold = vc_params.get('volume_spike_threshold', 2.0)
    volume_spike_window = vc_params.get('volume_spike_window', 10)

    # 构建所需列名，集中处理以提高可读性
    close_col = f'close_{vol_tf}'
    high_col = f'high_{vol_tf}'
    low_col = f'low_{vol_tf}'
    volume_col = f'volume_{vol_tf}'
    amt_ma_col = f'AMT_MA_{vc_params.get("amount_ma_period", 20)}_{vol_tf}'
    cmf_col = f'CMF_{vc_params.get("cmf_period", 20)}_{vol_tf}'
    obv_col = f'OBV_{vol_tf}'
    obv_ma_col = f'OBV_MA_{vc_params.get("obv_ma_period", 10)}_{vol_tf}'
    mfi_col = f'MFI_{bs_params.get("mfi_period", 14)}_{vol_tf}'

    # 检查所需列是否存在，提前定义必需列以避免重复计算
    required_cols = [close_col, high_col, low_col, volume_col, amt_ma_col, cmf_col, obv_col, obv_ma_col]
    if dd_params.get('enabled', False) and dd_params.get('indicators', {}).get('mfi', False):
        required_cols.append(mfi_col)

    missing_cols = [col for col in required_cols if col not in data.columns or data[col].isnull().all()]
    if missing_cols:
        logger.warning(f"量能调整缺少必需的数据列: {missing_cols} (时间框架: {vol_tf})，跳过量能调整。")
        if return_analysis:
            volume_analysis_df['volume_confirmation_signal'] = 0
            volume_analysis_df['volume_spike_signal'] = 0
            volume_analysis_df['volume_bearish_divergence_signal'] = 0
            volume_analysis_df['volume_bullish_divergence_signal'] = 0
            return adjusted_score, volume_analysis_df
        return adjusted_score, None

    # 获取数据序列并确保索引对齐，集中处理以优化性能
    data_aligned = data.reindex(preliminary_score.index, fill_value=np.nan)
    close = data_aligned[close_col]
    high = data_aligned[high_col]
    low = data_aligned[low_col]
    volume = data_aligned[volume_col]
    amount_ma = data_aligned[amt_ma_col]
    cmf = data_aligned[cmf_col].fillna(0)
    obv = data_aligned[obv_col]
    obv_ma = data_aligned[obv_ma_col]
    mfi = data_aligned.get(mfi_col, pd.Series(np.nan, index=preliminary_score.index))

    # 计算量能确认信号，逻辑清晰分块处理
    is_volume_supportive = (cmf > 0.05) & (obv > obv_ma)
    is_volume_contradictory = (cmf < -0.05) & (obv < obv_ma)
    volume_confirmation_signal = pd.Series(0, index=adjusted_score.index)
    volume_confirmation_signal.loc[is_volume_supportive] = 1
    volume_confirmation_signal.loc[is_volume_contradictory] = -1
    if return_analysis:
        volume_analysis_df['volume_confirmation_signal'] = volume_confirmation_signal

    # 计算成交量突增信号，优化 rolling 操作
    volume_ma = volume.rolling(window=volume_spike_window, min_periods=max(1, volume_spike_window // 2)).mean()
    is_volume_spike = (volume > volume_ma * volume_spike_threshold).fillna(False)
    volume_spike_signal = is_volume_spike.astype(int)
    if return_analysis:
        volume_analysis_df['volume_spike_signal'] = volume_spike_signal

    # 计算量价背离信号，逻辑分块以提高可读性
    volume_bearish_divergence_signal = pd.Series(0, index=adjusted_score.index)
    volume_bullish_divergence_signal = pd.Series(0, index=adjusted_score.index)
    if dd_params.get('enabled', False) and len(close) >= 10:
        lookback_div = dd_params.get('price_period', 14)
        price_high_rolling = high.rolling(window=lookback_div).max()
        price_low_rolling = low.rolling(window=lookback_div).min()
        is_new_high = high >= price_high_rolling.shift(1)
        is_new_low = low <= price_low_rolling.shift(1)
        obv_trend_up = obv.diff().fillna(0) > 0
        obv_trend_down = obv.diff().fillna(0) < 0
        cmf_positive = cmf > 0.05
        cmf_negative = cmf < -0.05
        bearish_div_cond = is_new_high & (~obv_trend_up | cmf_negative)
        volume_bearish_divergence_signal.loc[bearish_div_cond.fillna(False)] = -1
        bullish_div_cond = is_new_low & (~obv_trend_down | cmf_positive)
        volume_bullish_divergence_signal.loc[bullish_div_cond.fillna(False)] = 1
    if return_analysis:
        volume_analysis_df['volume_bearish_divergence_signal'] = volume_bearish_divergence_signal
        volume_analysis_df['volume_bullish_divergence_signal'] = volume_bullish_divergence_signal

    # 应用调整到分数，逻辑分组以提高可读性
    is_bullish_score = adjusted_score > 55
    is_bearish_score = adjusted_score < 45

    # 量能确认调整
    adjusted_score.loc[is_bullish_score & (volume_confirmation_signal == 1)] *= boost
    adjusted_score.loc[is_bullish_score & (volume_confirmation_signal == -1)] *= penalty
    adjusted_score.loc[is_bearish_score & (volume_confirmation_signal == -1)] = 50 - (50 - adjusted_score.loc[is_bearish_score & (volume_confirmation_signal == -1)]) * boost
    adjusted_score.loc[is_bearish_score & (volume_confirmation_signal == 1)] = 50 - (50 - adjusted_score.loc[is_bearish_score & (volume_confirmation_signal == 1)]) * penalty

    # 量价背离调整
    divergence_penalty = dd_params.get('divergence_penalty_factor', 0.85)
    adjusted_score.loc[is_bullish_score & (volume_bearish_divergence_signal == -1)] *= divergence_penalty
    adjusted_score.loc[is_bearish_score & (volume_bullish_divergence_signal == 1)] = 50 - (50 - adjusted_score.loc[is_bearish_score & (volume_bullish_divergence_signal == 1)]) * penalty

    # 成交量突增调整
    spike_factor = 0.05
    adjusted_score.loc[is_bullish_score & (volume_spike_signal == 1)] += (100 - adjusted_score.loc[is_bullish_score & (volume_spike_signal == 1)]) * spike_factor
    adjusted_score.loc[is_bearish_score & (volume_spike_signal == 1)] -= adjusted_score.loc[is_bearish_score & (volume_spike_signal == 1)] * spike_factor

    # 确保分数在 0-100 范围内
    adjusted_score = adjusted_score.clip(0, 100)
    logger.debug(f"时间框架 {vol_tf} 的量能调整完成。")

    # 确保 adjusted_score 的索引与 preliminary_score 一致
    adjusted_score = adjusted_score.reindex(preliminary_score.index, fill_value=preliminary_score)

    # 根据 return_analysis 返回结果
    if return_analysis:
        return adjusted_score, volume_analysis_df
    return adjusted_score, None

