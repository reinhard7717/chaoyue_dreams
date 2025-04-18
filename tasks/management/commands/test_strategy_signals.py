# tasks/management/commands/test_strategy_signals.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from django.core.management.base import BaseCommand
import logging
from django.utils import timezone
from decimal import Decimal, InvalidOperation
from asgiref.sync import sync_to_async
from django.core.cache import cache
from scipy.signal import find_peaks # 确保已安装: pip install scipy

from dao_manager.daos.stock_basic_dao import StockBasicDAO
from dao_manager.daos.stock_realtime_dao import StockRealtimeDAO
from stock_models.stock_analytics import StockScoreAnalysis # 确保导入更新后的模型
from stock_models.stock_basic import StockInfo
from stock_models.stock_realtime import StockRealtimeData
from utils.cache_manager import CacheManager # 用于处理时区

# --- 导入时区处理库 ---
try:
    import tzlocal
    from zoneinfo import ZoneInfo
except ImportError:
    tzlocal = None
    ZoneInfo = None
    print("警告：无法导入 'tzlocal' 或 'zoneinfo'。")

# --- 导入 pandas_ta ---
try:
    import pandas_ta as ta
except ImportError:
    ta = None
    print("警告：无法导入 'pandas_ta'。")

# --- 导入项目模块 ---
from services.indicator_services import IndicatorService # 依赖这个服务来准备数据
from strategies.macd_rsi_kdj_boll_strategy import MacdRsiKdjBollEnhancedStrategy

logger = logging.getLogger(__name__)

# --- 辅助函数：清理潜在的无效数值 ---
def clean_value(value, default=None):
    """
    将 NaN, inf, -inf, NaT, None 等无效值转换为指定的默认值 (通常是 None)。
    处理 numpy 类型和 Decimal 的特殊情况。

    Args:
        value: 输入值。
        default: 当输入值为无效值时返回的默认值。

    Returns:
        清理后的值或默认值。
    """
    # 检查 pandas 的 NA 或 Python 的 None
    if pd.isna(value):
        return default
    # 检查浮点数的无穷大
    if isinstance(value, float) and not np.isfinite(value):
        return default
    # 检查 Decimal 的特殊值
    if isinstance(value, Decimal) and (value.is_nan() or value.is_infinite()):
        return default
    # 检查 numpy 数值类型的 NaN 或 Inf
    if isinstance(value, (np.number)) and not np.isfinite(value):
         return default

    # 如果是 numpy 类型，尝试转换为 Python 内建类型
    if hasattr(value, 'item'):
        try:
            item_value = value.item()
            # 再次检查转换后的值是否有效
            if isinstance(item_value, float) and not np.isfinite(item_value):
                 return default
            return item_value
        except (ValueError, TypeError):
             # 转换失败则返回默认值
             return default

    # 其他情况，假设值是有效的，直接返回
    return value

# --- 辅助函数：获取 find_peaks 参数 ---
def get_find_peaks_params(time_level: str, base_lookback: int) -> Dict[str, Any]:
    """
    根据时间级别和基础回看期，返回适用于 find_peaks 的参数。
    短周期更敏感，长周期过滤更多噪音。

    Args:
        time_level (str): 分析的时间级别 ('1', '5', '15', '30', '60', 'D', 'W', 'M', 'unknown').
        base_lookback (int): 用于计算参数的基础回看期。

    Returns:
        Dict[str, Any]: 包含 'distance' 和 'prominence_factor' 的字典。
                       'prominence_factor' 用于乘以滚动标准差来确定最小显著性。
    """
    distance_factor = 2 # 默认峰/谷之间至少距离 lookback / 2
    prominence_factor = 0.5 # 默认显著性为 0.5 倍滚动标准差

    level_map = {
        '1': (4, 0.3),   # 1分钟: 距离更近，显著性要求更低
        '5': (3, 0.4),   # 5分钟
        '15': (2, 0.5),  # 15分钟 (基准)
        '30': (2, 0.6),  # 30分钟
        '60': (2, 0.7),  # 60分钟
        'D': (2, 1.0),   # 日线: 距离不变，显著性要求更高
        'W': (2, 1.5),   # 周线
        'M': (2, 2.0),   # 月线
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
    # logger.debug(f"Time level '{time_level}', find_peaks params: {params}")
    return params

# --- 辅助函数：增强版背离检测 (包含隐藏背离) ---
def detect_divergence(price: pd.Series,
                      indicator: pd.Series,
                      lookback: int = 14,
                      find_peaks_params: Dict[str, Any] = {'distance': 7, 'prominence_factor': 0.5}
                      ) -> pd.Series:
    """
    检测价格与指标之间的常规背离和隐藏背离。
    信号值: 1 (常规牛), -1 (常规熊), 2 (隐藏牛), -2 (隐藏熊), 0 (无)。

    Args:
        price (pd.Series): 价格序列。
        indicator (pd.Series): 指标序列。
        lookback (int): 回看窗口。
        find_peaks_params (Dict[str, Any]): find_peaks 的参数 ('distance', 'prominence_factor')。

    Returns:
        pd.Series: 背离信号。
    """
    divergence_signal = pd.Series(0, index=price.index)
    if price.isnull().all() or indicator.isnull().all() or len(price) < lookback * 2:
        return divergence_signal

    # --- 准备 find_peaks 参数 ---
    distance = find_peaks_params.get('distance', max(1, lookback // 2))
    prominence_factor = find_peaks_params.get('prominence_factor', 0.5)

    # 计算价格和指标的最小显著性 (基于滚动标准差)
    # 使用 fillna(0) 处理滚动窗口初期的 NaN
    min_prominence_price = (price.rolling(lookback).std() * prominence_factor).fillna(0).values
    min_prominence_indicator = (indicator.rolling(lookback).std() * prominence_factor).fillna(0).values

    # 填充指标中的 NaN，以避免 find_peaks 出错
    indicator_filled = indicator.fillna(method='ffill').fillna(method='bfill')
    if indicator_filled.isnull().all(): # 如果填充后仍然全是 NaN
        return divergence_signal

    # --- 查找峰值和谷值 ---
    try:
        # 确保 prominence 是非负数
        min_prominence_price = np.maximum(min_prominence_price, 0)
        min_prominence_indicator = np.maximum(min_prominence_indicator, 0)

        price_peaks, _ = find_peaks(price, distance=distance, prominence=min_prominence_price)
        price_troughs, _ = find_peaks(-price, distance=distance, prominence=min_prominence_price)
        indicator_peaks, _ = find_peaks(indicator_filled, distance=distance, prominence=min_prominence_indicator)
        indicator_troughs, _ = find_peaks(-indicator_filled, distance=distance, prominence=min_prominence_indicator)
    except Exception as fp_err:
        logger.warning(f"find_peaks encountered an error: {fp_err}. Skipping divergence detection.")
        return divergence_signal


    # --- 检测常规看跌背离 (顶背离) ---
    # 价格创新高 (HH)，指标未创新高 (LH)
    if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
        p_peak1_idx, p_peak2_idx = price_peaks[-2], price_peaks[-1]
        if price.iloc[p_peak2_idx] > price.iloc[p_peak1_idx]:
            # 查找对应的指标峰值 (简单取最近的)
            ind_peaks_near_p1 = indicator_peaks[(indicator_peaks >= p_peak1_idx - distance//2) & (indicator_peaks <= p_peak1_idx + distance//2)]
            ind_peaks_near_p2 = indicator_peaks[(indicator_peaks >= p_peak2_idx - distance//2) & (indicator_peaks <= p_peak2_idx + distance//2)]
            if len(ind_peaks_near_p1) > 0 and len(ind_peaks_near_p2) > 0:
                i_peak1_idx, i_peak2_idx = ind_peaks_near_p1[-1], ind_peaks_near_p2[-1]
                # 确保比较的指标值有效
                if pd.notna(indicator_filled.iloc[i_peak2_idx]) and pd.notna(indicator_filled.iloc[i_peak1_idx]):
                    if indicator_filled.iloc[i_peak2_idx] < indicator_filled.iloc[i_peak1_idx]:
                        divergence_signal.iloc[p_peak2_idx] = -1 # 常规熊

    # --- 检测常规看涨背离 (底背离) ---
    # 价格创新低 (LL)，指标未创新低 (HL)
    if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
        p_trough1_idx, p_trough2_idx = price_troughs[-2], price_troughs[-1]
        if price.iloc[p_trough2_idx] < price.iloc[p_trough1_idx]:
            ind_troughs_near_p1 = indicator_troughs[(indicator_troughs >= p_trough1_idx - distance//2) & (indicator_troughs <= p_trough1_idx + distance//2)]
            ind_troughs_near_p2 = indicator_troughs[(indicator_troughs >= p_trough2_idx - distance//2) & (indicator_troughs <= p_trough2_idx + distance//2)]
            if len(ind_troughs_near_p1) > 0 and len(ind_troughs_near_p2) > 0:
                i_trough1_idx, i_trough2_idx = ind_troughs_near_p1[-1], ind_troughs_near_p2[-1]
                if pd.notna(indicator_filled.iloc[i_trough2_idx]) and pd.notna(indicator_filled.iloc[i_trough1_idx]):
                    if indicator_filled.iloc[i_trough2_idx] > indicator_filled.iloc[i_trough1_idx]:
                        divergence_signal.iloc[p_trough2_idx] = 1 # 常规牛

    # --- 检测隐藏看跌背离 ---
    # 价格未创新高 (LH)，指标创新高 (HH)
    if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
        p_peak1_idx, p_peak2_idx = price_peaks[-2], price_peaks[-1]
        if price.iloc[p_peak2_idx] < price.iloc[p_peak1_idx]:
            ind_peaks_near_p1 = indicator_peaks[(indicator_peaks >= p_peak1_idx - distance//2) & (indicator_peaks <= p_peak1_idx + distance//2)]
            ind_peaks_near_p2 = indicator_peaks[(indicator_peaks >= p_peak2_idx - distance//2) & (indicator_peaks <= p_peak2_idx + distance//2)]
            if len(ind_peaks_near_p1) > 0 and len(ind_peaks_near_p2) > 0:
                i_peak1_idx, i_peak2_idx = ind_peaks_near_p1[-1], ind_peaks_near_p2[-1]
                if pd.notna(indicator_filled.iloc[i_peak2_idx]) and pd.notna(indicator_filled.iloc[i_peak1_idx]):
                    if indicator_filled.iloc[i_peak2_idx] > indicator_filled.iloc[i_peak1_idx]:
                        # 仅在最近的峰值处标记，且之前没有常规熊背离
                        if divergence_signal.iloc[p_peak2_idx] == 0:
                             divergence_signal.iloc[p_peak2_idx] = -2 # 隐藏熊

    # --- 检测隐藏看涨背离 ---
    # 价格未创新低 (HL)，指标创新低 (LL)
    if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
        p_trough1_idx, p_trough2_idx = price_troughs[-2], price_troughs[-1]
        if price.iloc[p_trough2_idx] > price.iloc[p_trough1_idx]:
            ind_troughs_near_p1 = indicator_troughs[(indicator_troughs >= p_trough1_idx - distance//2) & (indicator_troughs <= p_trough1_idx + distance//2)]
            ind_troughs_near_p2 = indicator_troughs[(indicator_troughs >= p_trough2_idx - distance//2) & (indicator_troughs <= p_trough2_idx + distance//2)]
            if len(ind_troughs_near_p1) > 0 and len(ind_troughs_near_p2) > 0:
                i_trough1_idx, i_trough2_idx = ind_troughs_near_p1[-1], ind_troughs_near_p2[-1]
                if pd.notna(indicator_filled.iloc[i_trough2_idx]) and pd.notna(indicator_filled.iloc[i_trough1_idx]):
                    if indicator_filled.iloc[i_trough2_idx] < indicator_filled.iloc[i_trough1_idx]:
                         # 仅在最近的谷值处标记，且之前没有常规牛背离
                         if divergence_signal.iloc[p_trough2_idx] == 0:
                              divergence_signal.iloc[p_trough2_idx] = 2 # 隐藏牛

    return divergence_signal.astype(int)


# --- 辅助函数：K线形态检测 ---
def detect_kline_patterns(df: pd.DataFrame) -> pd.Series:
    """
    检测基本的 K 线形态。
    信号值:
        1: 看涨吞没 (Bullish Engulfing)
       -1: 看跌吞没 (Bearish Engulfing)
        2: 锤子线 (Hammer - 需结合前期下跌看)
       -2: 上吊线 (Hanging Man - 需结合前期上涨看)
        3: 早晨之星 (Morning Star - 简化版)
       -3: 黄昏之星 (Evening Star - 简化版)
        5: 十字星 (Doji - 中性，可能反转)
       10: 光头光脚阳线 (Bullish Marubozu)
      -10: 光头光脚阴线 (Bearish Marubozu)
        0: 无显著形态
    """
    patterns = pd.Series(0, index=df.index)
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.warning("K-line detection requires OHLC columns.")
        return patterns
    # 确保数据类型正确，并处理潜在的 NaN
    df_ohlc = df[required_cols].copy()
    for col in required_cols:
        df_ohlc[col] = pd.to_numeric(df_ohlc[col], errors='coerce')
    df_ohlc = df_ohlc.dropna() # 删除包含 NaN 的行，避免计算错误
    if df_ohlc.empty:
        return patterns # 如果没有有效数据，返回空 Series

    o, h, l, c = df_ohlc['open'], df_ohlc['high'], df_ohlc['low'], df_ohlc['close']
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1) # 前一根 K 线
    o2, c2 = o.shift(2), c.shift(2) # 再前一根 K 线

    # 计算实体和影线 (确保处理 NaN)
    body = abs(c - o)
    body1 = abs(c1 - o1).fillna(0) # 前一根 K 线可能不存在
    body2 = abs(c2 - o2).fillna(0) # 再前一根 K 线可能不存在
    full_range = (h - l).replace(0, 1e-6) # 防止除以零
    full_range1 = (h1 - l1).fillna(0).replace(0, 1e-6)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(o, c) - l
    is_green = c > o
    is_red = c < o
    is_green1 = c1 > o1
    is_red1 = c1 < o1
    is_green2 = c2 > o2
    is_red2 = c2 < o2

    # --- 吞没形态 ---
    # 看涨吞没: 前阴线，当前阳线实体吞没前阴线实体
    bull_engulf = is_red1 & is_green & (c > o1) & (o < c1) & (body > body1 * 1.01) # 当前实体略大于前实体
    patterns.loc[bull_engulf[bull_engulf].index] = 1 # 使用 .loc 和布尔索引赋值
    # 看跌吞没: 前阳线，当前阴线实体吞没前阳线实体
    bear_engulf = is_green1 & is_red & (o > c1) & (c < o1) & (body > body1 * 1.01)
    patterns.loc[bear_engulf[bear_engulf].index] = -1

    # --- 锤子线 / 上吊线 ---
    # 特征: 实体小，下影线长 (>= 2*实体)，上影线短 (< 0.5*实体)
    small_body_threshold = full_range * 0.2 # 实体小于全长的 20%
    long_lower_shadow = lower_shadow >= 2 * body
    short_upper_shadow = upper_shadow < 0.5 * body
    hammer_like = (body > 1e-6) & (body < small_body_threshold) & long_lower_shadow & short_upper_shadow

    # 锤子线: 出现在下跌后 (简化: 前两根 K 线至少一根是阴线)
    is_hammer = hammer_like & (is_red1 | is_red2)
    patterns.loc[is_hammer[is_hammer].index] = 2
    # 上吊线: 出现在上涨后 (简化: 前两根 K 线至少一根是阳线)
    is_hanging = hammer_like & (is_green1 | is_green2)
    patterns.loc[is_hanging[is_hanging].index] = -2 # 注意：上吊线信号可能覆盖锤子线

    # --- 早晨/黄昏之星 (简化版三 K 线模式) ---
    star_body_threshold = full_range1 * 0.3 # 星线的实体小于自身全长的 30%
    is_star = (body1 < star_body_threshold) & (body1 > 1e-6)
    # 星线跳空 (简化: 星线实体与前后实体无重叠)
    gap_down1 = np.minimum(o1, c1) < np.minimum(o2, c2) # 第一根与第二根向下跳空
    gap_up1 = np.minimum(o1, c1) > np.maximum(o2, c2) # 第一根与第二根向上跳空
    gap_down2 = np.minimum(o, c) < np.minimum(o1, c1) # 第二根与第三根向下跳空
    gap_up2 = np.minimum(o, c) > np.maximum(o1, c1) # 第二根与第三根向上跳空

    # 早晨之星: 1.长阴线 -> 2.星(向下跳空) -> 3.长阳线(向上跳空, 收盘>第一根中点)
    morning_star = is_red2 & (body2 > body * 1.5) & is_star & gap_down1 & is_green & (body > body2 * 0.5) & gap_up2 & (c > (o2 + c2) / 2)
    patterns.loc[morning_star[morning_star].index] = 3

    # 黄昏之星: 1.长阳线 -> 2.星(向上跳空) -> 3.长阴线(向下跳空, 收盘<第一根中点)
    evening_star = is_green2 & (body2 > body * 1.5) & is_star & gap_up1 & is_red & (body > body2 * 0.5) & gap_down2 & (c < (o2 + c2) / 2)
    patterns.loc[evening_star[evening_star].index] = -3

    # --- 十字星 (Doji) ---
    # 实体非常小 (例如小于全长的 5%)
    doji_threshold = full_range * 0.05
    is_doji = (body <= doji_threshold) & (body > 1e-6) # 实体非常小但存在
    patterns.loc[is_doji[is_doji].index] = 5

    # --- 光头光脚线 (Marubozu) ---
    # 实体接近全长，影线非常短 (例如上下影线都小于实体的 5%)
    shadow_threshold_factor = 0.05
    no_upper_shadow = upper_shadow < body * shadow_threshold_factor
    no_lower_shadow = lower_shadow < body * shadow_threshold_factor
    is_marubozu = no_upper_shadow & no_lower_shadow & (body > full_range * 0.9) # 实体占 90% 以上

    bull_marubozu = is_marubozu & is_green
    patterns.loc[bull_marubozu[bull_marubozu].index] = 10
    bear_marubozu = is_marubozu & is_red
    patterns.loc[bear_marubozu[bear_marubozu].index] = -10

    # 将结果合并回原始索引，填充未计算行为 0
    patterns_aligned = pd.Series(0, index=df.index)
    patterns_aligned.update(patterns)

    return patterns_aligned.astype(int)


# --- analyze_score_trend 函数 (完整版) ---
async def analyze_score_trend(stock_code: str,
    score_price_vwap_df: pd.DataFrame,
    t0_params: Optional[Dict[str, Any]] = None,
    save_to_db: bool = True,
    cache_results: bool = True,
    analysis_time_level: Optional[str] = None,
    analysis_params: Optional[Dict[str, Any]] = None
    ) -> Optional[pd.DataFrame]:
    """
    增强版分析函数：
    - 集成隐藏背离检测
    - 实现基础 K 线形态识别
    - 使用加权方式计算确认反转信号
    - find_peaks 参数根据时间级别自适应
    """
    # --- 0. 参数处理与初始化 ---
    default_t0_params = {
        'enabled': True,
        'buy_dev_threshold': -0.003,
        'sell_dev_threshold': 0.005,
        'use_long_term_filter': True
    }
    if t0_params is None: t0_params = default_t0_params
    else: t0_params = {**default_t0_params, **t0_params}

    # 分析参数默认值 (加入权重和 K 线相关)
    default_analysis_params = {
        'macd_fast': 10, 'macd_slow': 26, 'macd_signal': 9,
        'rsi_period': 12, 'rsi_ob': 70, 'rsi_os': 30, 'rsi_extreme_ob': 80, 'rsi_extreme_os': 20,
        'stoch_k': 9, 'stoch_d': 3, 'stoch_smooth_k': 3, 'stoch_ob': 80, 'stoch_os': 20,
        'cci_period': 14, 'cci_ob': 100, 'cci_os': -100, 'cci_extreme_ob': 200, 'cci_extreme_os': -200,
        'mfi_period': 14, 'mfi_ob': 80, 'mfi_os': 20,
        'obv_ma_period': 10,
        'boll_period': 20, 'boll_std': 2,
        'volume_ma_period': 20, 'volume_spike_factor': 2.0,
        'divergence_lookback': 14,
        'confirmation_weighted_threshold': 3.0, # 加权确认信号的阈值
        'signal_weights': { # 定义各信号权重
            'alignment_reversal': 1.0, # 基于 EMA 排列的反转
            'macd_regular_div': 1.5,   # MACD 常规背离
            'macd_hidden_div': 0.8,    # MACD 隐藏背离 (趋势持续信号，权重稍低)
            'rsi_regular_div': 1.2,
            'rsi_hidden_div': 0.7,
            'mfi_regular_div': 1.0,
            'mfi_hidden_div': 0.6,
            'obv_regular_div': 0.8,    # OBV 背离权重可根据经验调整
            'obv_hidden_div': 0.5,
            'rsi_obos_reversal': 1.0,
            'stoch_obos_reversal': 0.9,
            'cci_obos_reversal': 0.8,
            'bb_reversal': 0.7,
            'volume_spike_confirm': 0.5, # 放量确认（仅在已有其他信号时加分）
            'kline_engulfing': 1.0,
            'kline_hammer_hanging': 1.2, # 锤子/上吊权重较高
            'kline_star': 1.5,           # 星线组合权重最高
            'kline_marubozu': 0.5,       # 光头光脚线 (趋势加强信号)
            'kline_doji': 0.3,           # Doji (犹豫信号，权重低)
        }
    }
    if analysis_params is None: analysis_params = default_analysis_params
    else:
        # 合并字典，特别是嵌套的 weights 字典
        merged_params = default_analysis_params.copy()
        merged_params.update(analysis_params)
        if 'signal_weights' in analysis_params: # 如果传入了 weights，需要合并
             merged_params['signal_weights'] = {**default_analysis_params['signal_weights'], **analysis_params['signal_weights']}
        analysis_params = merged_params


    # --- 1. 输入数据验证 ---
    if ta is None:
        logger.warning(f"[{stock_code}] pandas_ta 未安装，部分指标计算可能受限。")
    if score_price_vwap_df is None or score_price_vwap_df.empty:
        logger.warning(f"[{stock_code}] 输入的 DataFrame 为空，无法分析。")
        return None

    # 检查必需列 (核心 + OHLCV + T+0)
    required_cols = ['score', 'open', 'high', 'low', 'close', 'volume']
    if t0_params['enabled']: required_cols.append('vwap')
    missing_cols = [col for col in required_cols if col not in score_price_vwap_df.columns]
    if missing_cols:
        logger.warning(f"[{stock_code}] 输入 DataFrame 缺少进行完整分析所需的基础列: {missing_cols}。部分信号可能无法计算。")
        # 如果缺少核心列，则无法继续
        if 'score' in missing_cols or 'close' in missing_cols:
            logger.error(f"[{stock_code}] 缺少 'score' 或 'close' 列，无法进行核心分析。")
            return None
        if 'vwap' in missing_cols and t0_params['enabled']:
            logger.warning(f"[{stock_code}] 缺少 'vwap' 列，T+0 信号功能已禁用。")
            t0_params['enabled'] = False

    # 检查索引类型
    if not isinstance(score_price_vwap_df.index, pd.DatetimeIndex):
        logger.warning(f"[{stock_code}] 输入 DataFrame 的索引不是 DatetimeIndex，无法进行时间序列分析。")
        return None

    # 检查数据量
    LONG_TERM_EMA_PERIOD = 233
    min_required_data = LONG_TERM_EMA_PERIOD + 55 # 确保 EMA 计算稳定
    if len(score_price_vwap_df) < min_required_data:
        logger.warning(f"[{stock_code}] 数据点数量 ({len(score_price_vwap_df)}) 不足 {min_required_data}，可能无法完成所有分析（特别是 233 EMA 和长周期背离）。")


    # --- 2. 确定分析级别 ---
    if analysis_time_level is None:
        price_col_base = 'close'
        potential_price_cols = [col for col in score_price_vwap_df.columns if col.startswith(f'{price_col_base}_')]
        if potential_price_cols:
             try:
                 level_str = potential_price_cols[0].split('_')[-1]
                 if level_str.isdigit() or level_str.upper() in ['D', 'W', 'M']:
                     analysis_time_level = str(level_str).upper() if not level_str.isdigit() else str(level_str)
                 else:
                     analysis_time_level = 'unknown'
             except (IndexError, ValueError):
                 analysis_time_level = 'unknown'
        elif price_col_base in score_price_vwap_df.columns:
             logger.warning(f"[{stock_code}] 仅找到 '{price_col_base}' 列，无法确定精确时间级别，将使用 'unknown'。")
             analysis_time_level = 'unknown'
        else:
             logger.warning(f"[{stock_code}] 无法自动推断分析时间级别，将使用 'unknown'。数据库保存可能受影响。")
             analysis_time_level = 'unknown'

    logger.info(f"[{stock_code}] 开始分析 (级别: {analysis_time_level}, DB保存: {save_to_db}, Redis缓存: {cache_results})...")
    analysis_df = score_price_vwap_df.copy()
    realtime_dao = StockRealtimeDAO()

    # --- 3. 标准化时区 ---
    target_tz = None
    try:
        if ZoneInfo:
            target_tz = ZoneInfo("Asia/Shanghai")
            # 如果索引是幼稚型 (naive)，本地化为目标时区
            if analysis_df.index.tz is None:
                try:
                    # 尝试直接本地化
                    analysis_df.index = analysis_df.index.tz_localize(target_tz)
                except Exception:
                    # 如果直接本地化失败 (可能因为夏令时等歧义)，先设为 UTC 再转换
                    analysis_df.index = analysis_df.index.tz_localize('UTC').tz_convert(target_tz)
            # 如果索引已有其他时区，转换到目标时区
            elif analysis_df.index.tz != target_tz:
                analysis_df.index = analysis_df.index.tz_convert(target_tz)
        else:
            logger.warning(f"[{stock_code}] 'zoneinfo' 库不可用，无法进行精确的时区处理。")
    except Exception as e:
        logger.error(f"[{stock_code}] 处理 DataFrame 索引时区时出错: {e}", exc_info=True)
        return None # 时区处理失败则无法继续
    # 确保目标时区已成功设置
    if target_tz is None and ZoneInfo:
        logger.error(f"[{stock_code}] 未能成功设置目标时区 'Asia/Shanghai'。")
        return None


    # --- 3.5 检查并计算缺失的关键指标 (使用 pandas_ta) ---
    if ta: # 只有在 pandas_ta 可用时才尝试计算
        logger.debug(f"[{stock_code}] 检查并计算缺失的技术指标...")
        required_indicators = {
            'macd': {'fast': analysis_params['macd_fast'], 'slow': analysis_params['macd_slow'], 'signal': analysis_params['macd_signal']},
            'rsi': {'length': analysis_params['rsi_period']},
            'stoch': {'k': analysis_params['stoch_k'], 'd': analysis_params['stoch_d'], 'smooth_k': analysis_params['stoch_smooth_k']},
            'cci': {'length': analysis_params['cci_period']},
            'mfi': {'length': analysis_params['mfi_period']},
            'obv': {},
            'bbands': {'length': analysis_params['boll_period'], 'std': analysis_params['boll_std']}
        }
        # 检查 OHLCV 是否存在
        has_ohlcv = all(col in analysis_df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

        if has_ohlcv:
            # 使用 ta.Strategy 来批量计算，如果列不存在的话
            ta_strategy = ta.Strategy(
                name="AnalysisIndicators",
                description="Calculate missing indicators for analysis",
                ta=[]
            )
            for name, params in required_indicators.items():
                # 检查指标列是否已存在 (适配 pandas_ta 默认名称)
                col_to_check = ""
                if name == 'macd': col_to_check = f"MACDh_{params['fast']}_{params['slow']}_{params['signal']}"
                elif name == 'rsi': col_to_check = f"RSI_{params['length']}"
                elif name == 'stoch': col_to_check = f"STOCHk_{params['k']}_{params['d']}_{params['smooth_k']}"
                elif name == 'cci': col_to_check = f"CCI_{params['length']}_0.015" # pandas_ta 默认会加常数
                elif name == 'mfi': col_to_check = f"MFI_{params['length']}"
                elif name == 'obv': col_to_check = "OBV"
                elif name == 'bbands': col_to_check = f"BBL_{params['length']}_{params['std']}"

                if col_to_check and col_to_check not in analysis_df.columns:
                    logger.info(f"[{stock_code}] 指标列 '{col_to_check}' 不存在，将使用 pandas_ta 计算 {name}。")
                    ta_strategy.ta.append({"kind": name, **params})
                elif not col_to_check and name == 'obv' and 'OBV' not in analysis_df.columns: # 特殊处理 OBV
                     logger.info(f"[{stock_code}] 指标列 'OBV' 不存在，将使用 pandas_ta 计算 {name}。")
                     ta_strategy.ta.append({"kind": name, **params})

            if ta_strategy.ta: # 如果有需要计算的指标
                try:
                    # 确保使用正确的列名
                    analysis_df.ta.strategy(ta_strategy, open='open', high='high', low='low', close='close', volume='volume')
                    logger.info(f"[{stock_code}] 使用 pandas_ta 计算了缺失的指标。")
                except Exception as calc_err:
                    logger.error(f"[{stock_code}] 使用 pandas_ta 计算指标时出错: {calc_err}", exc_info=True)
        else:
            logger.warning(f"[{stock_code}] 缺少 OHLCV 数据，无法使用 pandas_ta 计算缺失的技术指标。")


    # --- 4. 计算评分 EMA 指标 ---
    all_ema_periods = [5, 13, 21, 55, LONG_TERM_EMA_PERIOD] # 定义所有需要计算的 EMA 周期
    try:
        for period in all_ema_periods:
            # 使用 pandas_ta 计算指数移动平均线
            analysis_df[f'ema_score_{period}'] = ta.ema(analysis_df['score'], length=period)
    except Exception as e:
        logger.error(f"[{stock_code}] 计算评分 EMA 时出错: {e}", exc_info=True)
        # EMA 计算失败也可能继续，但后续依赖 EMA 的计算会产生 NaN

    # --- 5. 计算评分趋势排列信号 ---
    # 基于短期 EMA (5, 13, 21, 55) 的相对大小关系判断趋势方向和强度
    # 信号值范围: -3 (完全空头) 到 +3 (完全多头)
    signal_5_13 = np.where(analysis_df['ema_score_5'] > analysis_df['ema_score_13'], 1, np.where(analysis_df['ema_score_5'] < analysis_df['ema_score_13'], -1, 0))
    signal_13_21 = np.where(analysis_df['ema_score_13'] > analysis_df['ema_score_21'], 1, np.where(analysis_df['ema_score_13'] < analysis_df['ema_score_21'], -1, 0))
    signal_21_55 = np.where(analysis_df['ema_score_21'] > analysis_df['ema_score_55'], 1, np.where(analysis_df['ema_score_21'] < analysis_df['ema_score_55'], -1, 0))
    analysis_df['alignment_signal'] = signal_5_13 + signal_13_21 + signal_21_55
    # 如果任何一个用于计算的 EMA 是 NaN，则排列信号也设为 NaN
    ema_cols_for_alignment = [f'ema_score_{p}' for p in [5, 13, 21, 55]]
    analysis_df.loc[analysis_df[ema_cols_for_alignment].isna().any(axis=1), 'alignment_signal'] = np.nan

    # --- 6. 计算评分趋势强度 ---
    # 使用 EMA13 和 EMA55 的差值衡量中期趋势强度
    analysis_df['ema_strength_13_55'] = analysis_df['ema_score_13'] - analysis_df['ema_score_55']

    # --- 7. 计算评分动能 ---
    # 计算评分的单周期变化量
    analysis_df['score_momentum'] = analysis_df['score'].diff()

    # --- 8. 计算评分波动性 ---
    # 计算评分在指定窗口期内的标准差
    volatility_window = 10 # 定义波动率计算窗口
    analysis_df['score_volatility'] = analysis_df['score'].rolling(window=volatility_window).std()

    # --- 9. 长期趋势背景分析 ---
    # 基于当前评分与 233 周期 EMA 的关系判断长期趋势
    # 1: 偏多 (评分 > 233 EMA), -1: 偏空 (评分 < 233 EMA), 0: 中性 (评分 ≈ 233 EMA)
    analysis_df['long_term_context'] = np.nan # 初始化为 NaN
    ema_233_col = f'ema_score_{LONG_TERM_EMA_PERIOD}'
    if ema_233_col in analysis_df.columns:
        analysis_df['long_term_context'] = np.where(
            analysis_df['score'] > analysis_df[ema_233_col], 1,
            np.where(analysis_df['score'] < analysis_df[ema_233_col], -1, 0)
        )
        # 如果 233 EMA 本身是 NaN，则 context 也是 NaN
        analysis_df.loc[analysis_df[ema_233_col].isna(), 'long_term_context'] = np.nan
    else:
        logger.warning(f"[{stock_code}] 未找到 '{ema_233_col}' 列，无法判断长期趋势背景。")

    # --- 10. 趋势反转信号检测 (基于排列信号变化 - 原始方法) ---
    analysis_df['reversal_signal'] = 0 # 初始化为 0
    if len(analysis_df) > 1 and 'alignment_signal' in analysis_df.columns:
        prev_alignment = analysis_df['alignment_signal'].shift(1) # 上一期的排列信号
        current_alignment = analysis_df['alignment_signal'] # 当前排列信号
        # 潜在顶部反转条件：之前是多头/偏多头 (>=1)，现在变为中性或空头 (<=0)
        top_reversal_condition = ((prev_alignment >= 1) & (current_alignment <= 0))
        # 潜在底部反转条件：之前是空头/偏空头 (<=-1)，现在变为中性或多头 (>=0)
        bottom_reversal_condition = ((prev_alignment <= -1) & (current_alignment >= 0))
        # 应用反转信号
        analysis_df.loc[top_reversal_condition, 'reversal_signal'] = -1
        analysis_df.loc[bottom_reversal_condition, 'reversal_signal'] = 1
        # 清理因 NaN 产生的无效信号
        analysis_df.loc[prev_alignment.isna() | current_alignment.isna(), 'reversal_signal'] = 0


    # --- 11. 计算增强的技术指标反转信号 ---
    logger.debug(f"[{stock_code}] 开始计算增强的技术指标反转信号...")

    # --- 11.1 增强背离信号检测 ---
    analysis_df['macd_hist_divergence'] = 0
    analysis_df['rsi_divergence'] = 0
    analysis_df['mfi_divergence'] = 0
    analysis_df['obv_divergence'] = 0

    macd_h_col = f"MACDh_{analysis_params['macd_fast']}_{analysis_params['macd_slow']}_{analysis_params['macd_signal']}"
    rsi_col = f"RSI_{analysis_params['rsi_period']}"
    mfi_col = f"MFI_{analysis_params['mfi_period']}"
    obv_col = "OBV"
    price_col = 'close'
    div_lookback = analysis_params['divergence_lookback']
    # 获取自适应的 find_peaks 参数
    fp_params = get_find_peaks_params(analysis_time_level, div_lookback)

    # 计算 MACD Histogram 背离
    if macd_h_col in analysis_df.columns and price_col in analysis_df.columns:
        try:
            analysis_df['macd_hist_divergence'] = detect_divergence(
                analysis_df[price_col], analysis_df[macd_h_col], lookback=div_lookback, find_peaks_params=fp_params
            )
        except Exception as e: logger.warning(f"[{stock_code}] 计算 MACD Hist 背离出错: {e}")
    else: logger.warning(f"[{stock_code}] 缺少列无法计算 MACD 背离。")

    # 计算 RSI 背离
    if rsi_col in analysis_df.columns and price_col in analysis_df.columns:
        try:
            analysis_df['rsi_divergence'] = detect_divergence(
                analysis_df[price_col], analysis_df[rsi_col], lookback=div_lookback, find_peaks_params=fp_params
            )
        except Exception as e: logger.warning(f"[{stock_code}] 计算 RSI 背离出错: {e}")
    else: logger.warning(f"[{stock_code}] 缺少列无法计算 RSI 背离。")

    # 计算 MFI 背离
    if mfi_col in analysis_df.columns and price_col in analysis_df.columns:
        try:
            analysis_df['mfi_divergence'] = detect_divergence(
                analysis_df[price_col], analysis_df[mfi_col], lookback=div_lookback, find_peaks_params=fp_params
            )
        except Exception as e: logger.warning(f"[{stock_code}] 计算 MFI 背离出错: {e}")
    else: logger.warning(f"[{stock_code}] 缺少列无法计算 MFI 背离。")

    # 计算 OBV 背离
    if obv_col in analysis_df.columns and price_col in analysis_df.columns:
        try:
            analysis_df['obv_divergence'] = detect_divergence(
                analysis_df[price_col], analysis_df[obv_col], lookback=div_lookback, find_peaks_params=fp_params
            )
        except Exception as e: logger.warning(f"[{stock_code}] 计算 OBV 背离出错: {e}")
    else: logger.warning(f"[{stock_code}] 缺少列无法计算 OBV 背离。")

    # --- 11.2 超买超卖区反转信号 ---
    analysis_df['rsi_ob_os_reversal'] = 0
    analysis_df['stoch_ob_os_reversal'] = 0
    analysis_df['cci_ob_os_reversal'] = 0

    # RSI OB/OS Reversal
    if rsi_col in analysis_df.columns:
        rsi = analysis_df[rsi_col]
        rsi_ob = analysis_params['rsi_ob']
        rsi_os = analysis_params['rsi_os']
        rsi_extreme_ob = analysis_params['rsi_extreme_ob']
        rsi_extreme_os = analysis_params['rsi_extreme_os']
        # 从超买区下穿
        sell_cond = ((rsi.shift(1) > rsi_ob) & (rsi <= rsi_ob)) | ((rsi.shift(1) > rsi_extreme_ob) & (rsi <= rsi_extreme_ob))
        # 从超卖区上穿
        buy_cond = ((rsi.shift(1) < rsi_os) & (rsi >= rsi_os)) | ((rsi.shift(1) < rsi_extreme_os) & (rsi >= rsi_extreme_os))
        analysis_df.loc[sell_cond, 'rsi_ob_os_reversal'] = -1
        analysis_df.loc[buy_cond, 'rsi_ob_os_reversal'] = 1
    else: logger.warning(f"[{stock_code}] 缺少 '{rsi_col}' 列，无法计算 RSI OB/OS 反转。")

    # Stochastic OB/OS Reversal (使用 %K 线)
    stoch_k_col = f"STOCHk_{analysis_params['stoch_k']}_{analysis_params['stoch_d']}_{analysis_params['stoch_smooth_k']}"
    if stoch_k_col in analysis_df.columns:
        stoch_k = analysis_df[stoch_k_col]
        stoch_ob = analysis_params['stoch_ob']
        stoch_os = analysis_params['stoch_os']
        sell_cond = (stoch_k.shift(1) > stoch_ob) & (stoch_k <= stoch_ob)
        buy_cond = (stoch_k.shift(1) < stoch_os) & (stoch_k >= stoch_os)
        analysis_df.loc[sell_cond, 'stoch_ob_os_reversal'] = -1
        analysis_df.loc[buy_cond, 'stoch_ob_os_reversal'] = 1
    else: logger.warning(f"[{stock_code}] 缺少 '{stoch_k_col}' 列，无法计算 Stochastic OB/OS 反转。")

    # CCI OB/OS Reversal
    cci_col = f"CCI_{analysis_params['cci_period']}_0.015"
    if cci_col in analysis_df.columns:
        cci = analysis_df[cci_col]
        cci_ob = analysis_params['cci_ob']
        cci_os = analysis_params['cci_os']
        cci_extreme_ob = analysis_params['cci_extreme_ob']
        cci_extreme_os = analysis_params['cci_extreme_os']
        sell_cond = ((cci.shift(1) > cci_ob) & (cci <= cci_ob)) | ((cci.shift(1) > cci_extreme_ob) & (cci <= cci_extreme_ob))
        buy_cond = ((cci.shift(1) < cci_os) & (cci >= cci_os)) | ((cci.shift(1) < cci_extreme_os) & (cci >= cci_extreme_os))
        analysis_df.loc[sell_cond, 'cci_ob_os_reversal'] = -1
        analysis_df.loc[buy_cond, 'cci_ob_os_reversal'] = 1
    else: logger.warning(f"[{stock_code}] 缺少 '{cci_col}' 列，无法计算 CCI OB/OS 反转。")

    # --- 11.3 成交量信号 ---
    analysis_df['volume_signal'] = 0
    volume_col = 'volume'
    if volume_col in analysis_df.columns:
        vol = analysis_df[volume_col]
        vol_ma_period = analysis_params['volume_ma_period']
        vol_spike_factor = analysis_params['volume_spike_factor']
        if len(vol.dropna()) > vol_ma_period:
            vol_ma = ta.sma(vol, length=vol_ma_period)
            analysis_df['volume_ma'] = vol_ma # 可以选择性保留均线值
            # 简单判断：当前成交量显著高于均线为放量 (1)
            analysis_df.loc[vol > vol_ma * vol_spike_factor, 'volume_signal'] = 1
            # 缩量判断可以更复杂，例如结合价格下跌判断
            # analysis_df.loc[vol < vol_ma * 0.5, 'volume_signal'] = -1 # 示例：低于均线一半为缩量
        else: logger.warning(f"[{stock_code}] 成交量数据不足 ({len(vol.dropna())})，无法计算 {vol_ma_period} 周期均线。")
    else: logger.warning(f"[{stock_code}] 缺少 '{volume_col}' 列，无法计算成交量信号。")

    # --- 11.4 布林带反转信号 ---
    analysis_df['bb_reversal_signal'] = 0
    bbl_col = f"BBL_{analysis_params['boll_period']}_{analysis_params['boll_std']}"
    bbu_col = f"BBU_{analysis_params['boll_period']}_{analysis_params['boll_std']}"
    if bbl_col in analysis_df.columns and bbu_col in analysis_df.columns and 'close' in analysis_df.columns:
        close_price = analysis_df['close']
        bbl = analysis_df[bbl_col]
        bbu = analysis_df[bbu_col]
        # 从上轨外侧跌回内部
        sell_cond = (close_price.shift(1) > bbu.shift(1)) & (close_price <= bbu)
        # 从下轨外侧涨回内部
        buy_cond = (close_price.shift(1) < bbl.shift(1)) & (close_price >= bbl)
        analysis_df.loc[sell_cond, 'bb_reversal_signal'] = -1
        analysis_df.loc[buy_cond, 'bb_reversal_signal'] = 1
    else: logger.warning(f"[{stock_code}] 缺少布林带列 ('{bbl_col}', '{bbu_col}') 或 'close' 列，无法计算布林带反转信号。")

    # --- 11.5 K 线形态信号 ---
    logger.debug(f"[{stock_code}] 计算 K 线形态信号...")
    try:
        analysis_df['kline_pattern'] = detect_kline_patterns(analysis_df)
    except Exception as e:
        logger.warning(f"[{stock_code}] 计算 K 线形态时出错: {e}")
        analysis_df['kline_pattern'] = 0

    # --- 11.6 计算加权确认的反转信号 ---
    analysis_df['confirmed_reversal_signal'] = 0
    weights = analysis_params['signal_weights']
    confirmation_threshold = analysis_params['confirmation_weighted_threshold']

    # 初始化加权分数
    bullish_score = pd.Series(0.0, index=analysis_df.index)
    bearish_score = pd.Series(0.0, index=analysis_df.index)

    # 累加看涨信号权重
    bullish_score += (analysis_df['reversal_signal'] == 1) * weights.get('alignment_reversal', 1.0)
    bullish_score += (analysis_df['macd_hist_divergence'] == 1) * weights.get('macd_regular_div', 1.0)
    bullish_score += (analysis_df['macd_hist_divergence'] == 2) * weights.get('macd_hidden_div', 0.5) # 隐藏看涨
    bullish_score += (analysis_df['rsi_divergence'] == 1) * weights.get('rsi_regular_div', 1.0)
    bullish_score += (analysis_df['rsi_divergence'] == 2) * weights.get('rsi_hidden_div', 0.5)
    bullish_score += (analysis_df['mfi_divergence'] == 1) * weights.get('mfi_regular_div', 1.0)
    bullish_score += (analysis_df['mfi_divergence'] == 2) * weights.get('mfi_hidden_div', 0.5)
    bullish_score += (analysis_df['obv_divergence'] == 1) * weights.get('obv_regular_div', 0.5)
    bullish_score += (analysis_df['obv_divergence'] == 2) * weights.get('obv_hidden_div', 0.3)
    bullish_score += (analysis_df['rsi_ob_os_reversal'] == 1) * weights.get('rsi_obos_reversal', 1.0)
    bullish_score += (analysis_df['stoch_ob_os_reversal'] == 1) * weights.get('stoch_obos_reversal', 1.0)
    bullish_score += (analysis_df['cci_ob_os_reversal'] == 1) * weights.get('cci_obos_reversal', 1.0)
    bullish_score += (analysis_df['bb_reversal_signal'] == 1) * weights.get('bb_reversal', 0.5)
    # K 线形态看涨信号
    bullish_score += (analysis_df['kline_pattern'] == 1) * weights.get('kline_engulfing', 1.0) # 看涨吞没
    bullish_score += (analysis_df['kline_pattern'] == 2) * weights.get('kline_hammer_hanging', 1.0) # 锤子线
    bullish_score += (analysis_df['kline_pattern'] == 3) * weights.get('kline_star', 1.5) # 早晨之星
    bullish_score += (analysis_df['kline_pattern'] == 10) * weights.get('kline_marubozu', 0.5) # 看涨 Marubozu (趋势加强)
    bullish_score += (analysis_df['kline_pattern'] == 5) * weights.get('kline_doji', 0.3) # Doji (可能反转)
    # 放量确认 (只有在已有其他看涨信号时才加分)
    bullish_score += ((analysis_df['volume_signal'] == 1) & (bullish_score > 0)) * weights.get('volume_spike_confirm', 0.5)

    # 累加看跌信号权重
    bearish_score += (analysis_df['reversal_signal'] == -1) * weights.get('alignment_reversal', 1.0)
    bearish_score += (analysis_df['macd_hist_divergence'] == -1) * weights.get('macd_regular_div', 1.0)
    bearish_score += (analysis_df['macd_hist_divergence'] == -2) * weights.get('macd_hidden_div', 0.5) # 隐藏看跌
    bearish_score += (analysis_df['rsi_divergence'] == -1) * weights.get('rsi_regular_div', 1.0)
    bearish_score += (analysis_df['rsi_divergence'] == -2) * weights.get('rsi_hidden_div', 0.5)
    bearish_score += (analysis_df['mfi_divergence'] == -1) * weights.get('mfi_regular_div', 1.0)
    bearish_score += (analysis_df['mfi_divergence'] == -2) * weights.get('mfi_hidden_div', 0.5)
    bearish_score += (analysis_df['obv_divergence'] == -1) * weights.get('obv_regular_div', 0.5)
    bearish_score += (analysis_df['obv_divergence'] == -2) * weights.get('obv_hidden_div', 0.3)
    bearish_score += (analysis_df['rsi_ob_os_reversal'] == -1) * weights.get('rsi_obos_reversal', 1.0)
    bearish_score += (analysis_df['stoch_ob_os_reversal'] == -1) * weights.get('stoch_obos_reversal', 1.0)
    bearish_score += (analysis_df['cci_ob_os_reversal'] == -1) * weights.get('cci_obos_reversal', 1.0)
    bearish_score += (analysis_df['bb_reversal_signal'] == -1) * weights.get('bb_reversal', 0.5)
    # K 线形态看跌信号
    bearish_score += (analysis_df['kline_pattern'] == -1) * weights.get('kline_engulfing', 1.0) # 看跌吞没
    bearish_score += (analysis_df['kline_pattern'] == -2) * weights.get('kline_hammer_hanging', 1.0) # 上吊线
    bearish_score += (analysis_df['kline_pattern'] == -3) * weights.get('kline_star', 1.5) # 黄昏之星
    bearish_score += (analysis_df['kline_pattern'] == -10) * weights.get('kline_marubozu', 0.5) # 看跌 Marubozu
    bearish_score += (analysis_df['kline_pattern'] == 5) * weights.get('kline_doji', 0.3) # Doji
    # 放量确认
    bearish_score += ((analysis_df['volume_signal'] == 1) & (bearish_score > 0)) * weights.get('volume_spike_confirm', 0.5)

    # 判断确认信号
    analysis_df.loc[bullish_score >= confirmation_threshold, 'confirmed_reversal_signal'] = 1
    analysis_df.loc[bearish_score >= confirmation_threshold, 'confirmed_reversal_signal'] = -1 # 注意：如果同时满足，看跌优先覆盖看涨 (可调整)

    # 可选：保存加权分数用于调试或进一步分析
    # analysis_df['bullish_reversal_score'] = bullish_score
    # analysis_df['bearish_reversal_score'] = bearish_score

    logger.debug(f"[{stock_code}] 增强的技术指标反转信号计算完成。")
    # --- 结束信号计算 ---


    # --- 12. T+0 相关计算 (历史数据部分) ---
    analysis_df['t0_signal'] = 0 # 初始化历史 T+0 信号列
    analysis_df['price_vwap_deviation'] = np.nan # 初始化偏离度列
    if t0_params['enabled']:
        logger.debug(f"[{stock_code}] 计算历史 T+0 指标 (基于 VWAP)...")
        buy_dev_threshold = t0_params['buy_dev_threshold']
        sell_dev_threshold = t0_params['sell_dev_threshold']
        use_long_term_filter = t0_params['use_long_term_filter']
        if 'vwap' in analysis_df.columns:
            # 计算价格偏离度 = (收盘价 - VWAP) / VWAP
            analysis_df['price_vwap_deviation'] = np.where(
                analysis_df['vwap'].isna() | (analysis_df['vwap'] == 0), np.nan, # 处理 VWAP 无效或为零的情况
                (analysis_df['close'] - analysis_df['vwap']) / analysis_df['vwap']
            )
            # 判断历史 T+0 信号条件
            is_score_uptrend = analysis_df['alignment_signal'] >= 1 # 短期趋势向好
            is_score_downtrend = analysis_df['alignment_signal'] <= -1 # 短期趋势向差
            is_price_below_vwap = analysis_df['price_vwap_deviation'] < buy_dev_threshold # 价格低于买入阈值
            is_price_above_vwap = analysis_df['price_vwap_deviation'] > sell_dev_threshold # 价格高于卖出阈值
            # 应用长期趋势过滤条件 (如果启用)
            long_term_buy_ok = True
            long_term_sell_ok = True
            if use_long_term_filter and 'long_term_context' in analysis_df.columns:
                # 允许买入：长期趋势为正(1)或中性(0) 或 未知(NaN)
                long_term_buy_ok = (analysis_df['long_term_context'] >= 0) | analysis_df['long_term_context'].isna()
                # 允许卖出：长期趋势为负(-1)或中性(0) 或 未知(NaN)
                long_term_sell_ok = (analysis_df['long_term_context'] <= 0) | analysis_df['long_term_context'].isna()
            elif use_long_term_filter:
                 logger.warning(f"[{stock_code}] T+0 配置了长期趋势过滤，但 'long_term_context' 列不存在或计算失败，过滤未生效。")
            # 合并所有条件生成历史 T+0 信号
            buy_condition = is_score_uptrend & is_price_below_vwap & long_term_buy_ok
            sell_condition = is_score_downtrend & is_price_above_vwap & long_term_sell_ok
            analysis_df.loc[buy_condition, 't0_signal'] = 1
            analysis_df.loc[sell_condition, 't0_signal'] = -1
            logger.debug(f"[{stock_code}] 历史 T+0 信号计算完成 (长期过滤: {use_long_term_filter})。")
        else:
            # 如果缺少 vwap 列，则禁用 T+0
            logger.warning(f"[{stock_code}] 'vwap' 列不存在于历史数据中，无法计算历史 T+0 指标。")
            t0_params['enabled'] = False # 禁用 T+0 功能


    # --- 13. 综合分析与输出 ---
    summary = "" # 初始化分析摘要文本
    latest_historical_data_cache = {} # 初始化用于缓存的最新历史数据字典
    realtime_data_cache = {} # 初始化用于缓存的实时数据字典
    if not analysis_df.empty:
        # 获取最新一条历史分析数据
        latest_data_row = analysis_df.iloc[-1] # Pandas Series
        latest_hist_time = analysis_df.index[-1] # Pandas Timestamp

        # --- 13.1 获取实时数据并计算实时 T+0 信号 ---
        latest_realtime: Optional[StockRealtimeData] = None # 最新实时行情对象
        latest_price: Optional[float] = None # 最新实时价格
        realtime_fetch_error: bool = False # 实时数据获取是否出错标志
        current_t0_signal: int = 0 # 当前计算出的 T+0 信号 (-1, 0, 1)
        current_deviation: float = np.nan # 当前实时价格相对最新历史 VWAP 的偏离度
        latest_vwap_for_rt = latest_data_row.get('vwap', np.nan) # 获取用于实时比较的最新历史 VWAP
        if t0_params['enabled']: # 仅在 T+0 功能启用时执行
            try:
                logger.debug(f"[{stock_code}] 正在获取最新实时数据...")
                latest_realtime = await realtime_dao.get_latest_realtime_data(stock_code)
                # 检查是否成功获取到有效的实时价格
                if latest_realtime and latest_realtime.current_price is not None:
                    latest_price = float(latest_realtime.current_price)
                    logger.debug(f"[{stock_code}] 获取到实时价格: {latest_price} at {latest_realtime.trade_time}")
                    # 计算实时偏离度和 T+0 信号
                    if not pd.isna(latest_vwap_for_rt) and latest_vwap_for_rt != 0:
                        current_deviation = (latest_price - latest_vwap_for_rt) / latest_vwap_for_rt
                        # 获取最新的历史信号用于判断当前 T+0
                        latest_alignment = latest_data_row.get('alignment_signal')
                        latest_long_term_ctx = latest_data_row.get('long_term_context', np.nan)
                        use_filter = t0_params.get('use_long_term_filter', False)
                        # 只有在短期排列信号有效时才判断 T+0
                        if not pd.isna(latest_alignment):
                            buy_threshold = t0_params['buy_dev_threshold']
                            sell_threshold = t0_params['sell_dev_threshold']
                            # 判断潜在买卖条件
                            potential_buy = latest_alignment >= 1 and current_deviation < buy_threshold
                            potential_sell = latest_alignment <= -1 and current_deviation > sell_threshold
                            # 判断长期过滤条件是否通过
                            buy_filter_passed = not use_filter or (not pd.isna(latest_long_term_ctx) and latest_long_term_ctx >= 0)
                            sell_filter_passed = not use_filter or (not pd.isna(latest_long_term_ctx) and latest_long_term_ctx <= 0)
                            # 生成最终实时 T+0 信号
                            if potential_buy and buy_filter_passed: current_t0_signal = 1
                            elif potential_sell and sell_filter_passed: current_t0_signal = -1
                            else: current_t0_signal = 0 # 其他情况为无信号或观望
                    else:
                        logger.warning(f"[{stock_code}] 最新的历史 VWAP 无效 ({latest_vwap_for_rt})，无法计算实时 T+0 信号。")
                    # --- 准备实时数据缓存字典 ---
                    realtime_data_cache = {
                        'fetch_time': timezone.now().isoformat(), # 记录获取缓存的时间 (ISO 格式)
                        'realtime_price': clean_value(latest_price), # 清理后的实时价格
                        'realtime_trade_time': latest_realtime.trade_time.isoformat() if latest_realtime and latest_realtime.trade_time else None, # 实时行情时间
                        'latest_vwap_used': clean_value(latest_vwap_for_rt), # 用于比较的 VWAP
                        'current_deviation': clean_value(current_deviation), # 清理后的实时偏离度
                        'current_t0_signal': clean_value(current_t0_signal, 0), # 清理后的实时 T+0 信号 (默认 0)
                        'fetch_error': False # 获取成功
                    }
                else:
                    # 未能获取到有效的实时价格
                    logger.warning(f"[{stock_code}] 未能获取到有效的最新实时价格。")
                    realtime_fetch_error = True # 标记获取错误
                    realtime_data_cache = {'fetch_error': True, 'fetch_time': timezone.now().isoformat()}
            except Exception as e:
                # 获取实时数据过程中发生异常
                logger.error(f"[{stock_code}] 获取实时数据时出错: {e}", exc_info=True)
                realtime_fetch_error = True # 标记获取错误
                realtime_data_cache = {'fetch_error': True, 'fetch_time': timezone.now().isoformat()}
        # ------------------------ (实时数据处理结束) ------------------------

        # --- 13.2 生成中文分析摘要 (加入新信号解读) ---
        summary = f"[{stock_code}] 最新评分与价格趋势分析 (历史数据截至: {latest_hist_time.strftime('%Y-%m-%d %H:%M:%S %Z')}):\n"
        summary += f"  - 最新历史评分: {clean_value(latest_data_row.get('score'), 'N/A'):.2f}, 收盘价: {clean_value(latest_data_row.get('close'), 'N/A'):.2f}, VWAP: {clean_value(latest_data_row.get('vwap'), 'N/A'):.2f}\n"
        summary += f"  - 评分 EMA: 5={clean_value(latest_data_row.get('ema_score_5'), 'N/A'):.2f}, 13={clean_value(latest_data_row.get('ema_score_13'), 'N/A'):.2f}, 21={clean_value(latest_data_row.get('ema_score_21'), 'N/A'):.2f}, 55={clean_value(latest_data_row.get('ema_score_55'), 'N/A'):.2f}\n"
        summary += f"  - 评分 EMA 233: {clean_value(latest_data_row.get(ema_233_col), 'N/A'):.2f}\n"
        # 长期趋势背景文本
        long_term_ctx = latest_data_row.get('long_term_context', np.nan)
        ctx_text = "未知 (NaN)"
        if long_term_ctx == 1: ctx_text = "偏多 (评分 > 233 EMA)"
        elif long_term_ctx == -1: ctx_text = "偏空 (评分 < 233 EMA)"
        elif long_term_ctx == 0: ctx_text = "中性 (评分 ≈ 233 EMA)"
        summary += f"  - 长期趋势背景: {ctx_text}\n"
        # 短期趋势排列文本
        alignment = latest_data_row.get('alignment_signal')
        align_text = "信号不足 (NaN)"
        if isinstance(alignment, (int, float)) and not pd.isna(alignment):
            alignment = int(alignment) # 转为整数方便比较
            if alignment == 3: align_text = "完全多头 (+3)"
            elif alignment == -3: align_text = "完全空头 (-3)"
            elif alignment > 0: align_text = f"偏多头 ({alignment})"
            elif alignment < 0: align_text = f"偏空头 ({alignment})"
            else: align_text = "混合/粘合 (0)"
        summary += f"  - 短期趋势排列 (5/13/21/55 EMA): {align_text}\n"
        # 评分动能文本
        momentum = latest_data_row.get('score_momentum')
        mom_text = "NaN"
        if not pd.isna(momentum):
            mom_text = f"{momentum:.2f} "
            if momentum > 0.5: mom_text += "(显著上升)"
            elif momentum > 0: mom_text += "(上升)"
            elif momentum < -0.5: mom_text += "(显著下降)"
            elif momentum < 0: mom_text += "(下降)"
            else: mom_text += "(持平)"
        summary += f"  - 评分动能 (单期变化): {mom_text}\n"
        # 评分波动性文本
        volatility = latest_data_row.get('score_volatility')
        vol_text = "NaN"
        vol_level_text = "" # 初始化波动性水平描述
        if not pd.isna(volatility):
             vol_text = f"{volatility:.2f} "
             try:
                 # 与历史分位数比较判断波动性高低
                 if len(analysis_df['score_volatility'].dropna()) > volatility_window * 2:
                     q75 = analysis_df['score_volatility'].quantile(0.75)
                     q25 = analysis_df['score_volatility'].quantile(0.25)
                     if pd.notna(q75) and pd.notna(q25): # 确保分位数有效
                         if volatility > q75: vol_level_text = "(偏高)"
                         elif volatility < q25: vol_level_text = "(偏低)"
                         else: vol_level_text = "(适中)"
                     else: vol_level_text = "(历史分位数计算失败)"
                 else: vol_level_text = "(历史数据不足无法判断高低)"
             except Exception: vol_level_text = "(无法计算历史分位数)"
             vol_text += vol_level_text
        summary += f"  - 评分波动性 ({volatility_window}期 std): {vol_text}\n"

        # --- 技术指标反转信号摘要 ---
        summary += f"--- 技术指标反转信号 ---\n"
        # 排列反转
        reversal = latest_data_row.get('reversal_signal', 0)
        reversal_text = "无"
        if reversal == 1: reversal_text = "底部(排列)"
        elif reversal == -1: reversal_text = "顶部(排列)"
        summary += f"  - 排列信号反转: {reversal_text}\n"

        # 背离信号 (更详细)
        div_signals = []
        div_map = {1: "常规牛", -1: "常规熊", 2: "隐藏牛", -2: "隐藏熊"}
        for ind in ['macd_hist', 'rsi', 'mfi', 'obv']:
            sig_col = f'{ind}_divergence'
            sig = latest_data_row.get(sig_col, 0)
            if sig != 0: div_signals.append(f"{ind.upper()}{div_map.get(sig, '')}")
        summary += f"  - 背离信号: {', '.join(div_signals) if div_signals else '无'}\n"

        # OB/OS 反转信号
        obos_signals = []
        if latest_data_row.get('rsi_ob_os_reversal') == 1: obos_signals.append("RSI超卖反转")
        if latest_data_row.get('rsi_ob_os_reversal') == -1: obos_signals.append("RSI超买反转")
        if latest_data_row.get('stoch_ob_os_reversal') == 1: obos_signals.append("KDJ超卖反转")
        if latest_data_row.get('stoch_ob_os_reversal') == -1: obos_signals.append("KDJ超买反转")
        if latest_data_row.get('cci_ob_os_reversal') == 1: obos_signals.append("CCI超卖反转")
        if latest_data_row.get('cci_ob_os_reversal') == -1: obos_signals.append("CCI超买反转")
        summary += f"  - 超买/卖反转: {', '.join(obos_signals) if obos_signals else '无'}\n"

        # K线形态信号
        kline_sig = latest_data_row.get('kline_pattern', 0)
        kline_map = {1: "看涨吞没", -1: "看跌吞没", 2: "锤子线", -2: "上吊线", 3: "早晨之星",
                     -3: "黄昏之星", 5: "十字星", 10: "看涨Marubozu", -10: "看跌Marubozu"}
        kline_text = kline_map.get(kline_sig, "无")
        summary += f"  - K线形态: {kline_text}\n"

        # 其他信号 (BB, Volume)
        other_rev_signals = []
        if latest_data_row.get('bb_reversal_signal') == 1: other_rev_signals.append("BB下轨反转")
        if latest_data_row.get('bb_reversal_signal') == -1: other_rev_signals.append("BB上轨反转")
        vol_sig = latest_data_row.get('volume_signal')
        if vol_sig == 1: other_rev_signals.append("放量")
        # elif vol_sig == -1: other_rev_signals.append("缩量") # 缩量信号意义需结合场景
        summary += f"  - 其他信号: {', '.join(other_rev_signals) if other_rev_signals else '无'}\n"

        # 确认反转信号 (加权)
        confirmed_reversal = latest_data_row.get('confirmed_reversal_signal', 0)
        confirmed_text = "无"
        conf_threshold = analysis_params.get('confirmation_weighted_threshold', 3.0) # 获取阈值
        if confirmed_reversal == 1: confirmed_text = f"**确认底部反转 (加权分数 >= {conf_threshold:.1f})**"
        elif confirmed_reversal == -1: confirmed_text = f"**确认顶部反转 (加权分数 >= {conf_threshold:.1f})**"
        summary += f"  - 确认反转信号: {confirmed_text}\n"
        # --- 结束反转信号摘要 ---

        # --- T+0 信号摘要 (基于实时数据) ---
        summary += f"--- 日内 T+0 交易信号 (基于实时价格 vs 最新历史 VWAP, 长期过滤: {'启用' if t0_params.get('use_long_term_filter', False) else '禁用'}) ---\n"
        if t0_params['enabled']:
             # 使用 realtime_data_cache 中的信息生成摘要
             if realtime_data_cache.get('fetch_error'):
                 summary += "  - 实时状态: 获取实时数据失败\n  - T+0 信号: 无法判断\n"
             elif realtime_data_cache.get('realtime_price') is None:
                 summary += "  - 实时状态: 未获取到有效实时价格\n  - T+0 信号: 无法判断\n"
             elif pd.isna(realtime_data_cache.get('latest_vwap_used')) or realtime_data_cache.get('latest_vwap_used') == 0:
                 summary += f"  - 实时价格: {clean_value(realtime_data_cache.get('realtime_price'), 'N/A'):.2f}\n"
                 summary += f"  - 最新历史 VWAP: 无效或为零\n"
                 summary += "  - T+0 信号: 无法判断 (VWAP无效)\n"
             else:
                 # 显示实时信息
                 summary += f"  - 实时价格: {realtime_data_cache['realtime_price']:.2f} (时间: {realtime_data_cache.get('realtime_trade_time', 'N/A')})\n"
                 summary += f"  - 最新历史 VWAP: {realtime_data_cache['latest_vwap_used']:.2f}\n"
                 summary += f"  - 当前价格相对 VWAP 偏离度: {clean_value(realtime_data_cache.get('current_deviation'), 'N/A'):.2%}\n"
                 # 显示实时 T+0 信号
                 rt_t0_signal = realtime_data_cache.get('current_t0_signal', 0)
                 if rt_t0_signal == 1:
                     summary += f"  - T+0 信号: **潜在买入点** (短期趋势向好, 价<VWAP阈值 {t0_params['buy_dev_threshold']:.2%}, 长期趋势允许)\n"
                 elif rt_t0_signal == -1:
                     summary += f"  - T+0 信号: **潜在卖出点** (短期趋势向差, 价>VWAP阈值 {t0_params['sell_dev_threshold']:.2%}, 长期趋势允许)\n"
                 else: # rt_t0_signal == 0
                     summary += "  - T+0 信号: 无或观望\n"
                     # 可以添加更详细的无信号原因解释 (可选)
                     # ...
        else:
            # T+0 功能未启用
            summary += "--- 日内 T+0 交易信号: 未启用或因数据缺失无法计算 ---\n"
        # ------------------------ (摘要生成结束) ------------------------

        # --- 13.3 打印摘要 ---
        print("\n" + "="*30 + " 评分与价格趋势分析摘要 " + "="*30)
        print(summary)
        print("="*78)

        # --- 13.4 准备用于缓存的最新历史数据字典 (加入新信号) ---
        latest_historical_data_cache = {
            'trade_time': latest_hist_time.isoformat(), # 时间使用 ISO 格式字符串
            'score': clean_value(latest_data_row.get('score')),
            'close_price': clean_value(latest_data_row.get('close')),
            'vwap': clean_value(latest_data_row.get('vwap')),
            'ema_score_5': clean_value(latest_data_row.get('ema_score_5')),
            'ema_score_13': clean_value(latest_data_row.get('ema_score_13')),
            'ema_score_21': clean_value(latest_data_row.get('ema_score_21')),
            'ema_score_55': clean_value(latest_data_row.get('ema_score_55')),
            'ema_score_233': clean_value(latest_data_row.get(ema_233_col)),
            'alignment_signal': clean_value(latest_data_row.get('alignment_signal')),
            'ema_strength_13_55': clean_value(latest_data_row.get('ema_strength_13_55')),
            'score_momentum': clean_value(latest_data_row.get('score_momentum')),
            'score_volatility': clean_value(latest_data_row.get('score_volatility')),
            'long_term_context': clean_value(latest_data_row.get('long_term_context')),
            'reversal_signal': clean_value(latest_data_row.get('reversal_signal')), # 原排列反转
            'price_vwap_deviation': clean_value(latest_data_row.get('price_vwap_deviation')),
            't0_signal_hist': clean_value(latest_data_row.get('t0_signal')), # 保存的是基于历史数据的 T+0 信号
            # --- 更新/新增缓存字段 ---
            'macd_hist_divergence': clean_value(latest_data_row.get('macd_hist_divergence')),
            'rsi_divergence': clean_value(latest_data_row.get('rsi_divergence')),
            'mfi_divergence': clean_value(latest_data_row.get('mfi_divergence')),
            'obv_divergence': clean_value(latest_data_row.get('obv_divergence')),
            'rsi_ob_os_reversal': clean_value(latest_data_row.get('rsi_ob_os_reversal')),
            'stoch_ob_os_reversal': clean_value(latest_data_row.get('stoch_ob_os_reversal')),
            'cci_ob_os_reversal': clean_value(latest_data_row.get('cci_ob_os_reversal')),
            'volume_signal': clean_value(latest_data_row.get('volume_signal')),
            'bb_reversal_signal': clean_value(latest_data_row.get('bb_reversal_signal')),
            'kline_pattern': clean_value(latest_data_row.get('kline_pattern')), # 新增 K 线
            'confirmed_reversal_signal': clean_value(latest_data_row.get('confirmed_reversal_signal')),
            # --- 结束更新/新增缓存字段 ---
        }

        # --- 14. 保存结果到 MySQL 数据库 (更新 defaults) ---
        if save_to_db and StockScoreAnalysis is not None and StockInfo is not None and analysis_time_level != 'unknown':
            logger.info(f"[{stock_code}] 开始保存分析结果到数据库 (级别: {analysis_time_level})...")
            try:
                # 异步获取关联的 StockInfo 对象
                try:
                    stock_instance = await StockInfo.objects.aget(stock_code=stock_code)
                except AttributeError: # Django 版本低于 4.1
                    stock_instance = await sync_to_async(StockInfo.objects.get)(stock_code=stock_code)

                # 定义一个内部异步函数来执行数据库保存操作
                @sync_to_async
                def save_analysis_data_orm(data_row, level):
                    """使用 Django ORM 的 update_or_create 保存单行数据"""
                    trade_time_aware = data_row.name # data_row.name 是带时区的 Timestamp
                    # 再次检查时区，以防万一
                    if not isinstance(trade_time_aware, pd.Timestamp) or trade_time_aware.tzinfo is None:
                         logger.warning(f"[{stock_code}] 保存数据库时时间戳无效或无时区: {trade_time_aware}, 跳过此条记录。")
                         return False # 返回 False 表示未成功保存
                    # 准备要写入或更新的数据字典 (使用 clean_value 清理)
                    defaults = {
                        'score': clean_value(data_row.get('score')),
                        'close_price': clean_value(data_row.get('close')),
                        'vwap': clean_value(data_row.get('vwap')),
                        'ema_score_5': clean_value(data_row.get('ema_score_5')),
                        'ema_score_13': clean_value(data_row.get('ema_score_13')),
                        'ema_score_21': clean_value(data_row.get('ema_score_21')),
                        'ema_score_55': clean_value(data_row.get('ema_score_55')),
                        'ema_score_233': clean_value(data_row.get(ema_233_col)),
                        'alignment_signal': clean_value(data_row.get('alignment_signal')),
                        'ema_strength_13_55': clean_value(data_row.get('ema_strength_13_55')),
                        'score_momentum': clean_value(data_row.get('score_momentum')),
                        'score_volatility': clean_value(data_row.get('score_volatility')),
                        'long_term_context': clean_value(data_row.get('long_term_context')),
                        'reversal_signal': clean_value(data_row.get('reversal_signal')),
                        'price_vwap_deviation': clean_value(data_row.get('price_vwap_deviation')),
                        't0_signal': clean_value(data_row.get('t0_signal')), # 保存历史 T+0 信号
                        # --- 更新/新增字段赋值 ---
                        'macd_hist_divergence': clean_value(data_row.get('macd_hist_divergence'), 0), # 默认0
                        'rsi_divergence': clean_value(data_row.get('rsi_divergence'), 0),
                        'mfi_divergence': clean_value(data_row.get('mfi_divergence'), 0),
                        'obv_divergence': clean_value(data_row.get('obv_divergence'), 0),
                        'rsi_ob_os_reversal': clean_value(data_row.get('rsi_ob_os_reversal'), 0),
                        'stoch_ob_os_reversal': clean_value(data_row.get('stoch_ob_os_reversal'), 0),
                        'cci_ob_os_reversal': clean_value(data_row.get('cci_ob_os_reversal'), 0),
                        'volume_signal': clean_value(data_row.get('volume_signal'), 0),
                        'bb_reversal_signal': clean_value(data_row.get('bb_reversal_signal'), 0),
                        'kline_pattern': clean_value(data_row.get('kline_pattern'), 0), # 新增 K 线
                        'confirmed_reversal_signal': clean_value(data_row.get('confirmed_reversal_signal'), 0),
                        # --- 结束更新/新增字段赋值 ---
                    }
                    try:
                        # 使用 update_or_create: 如果记录存在则更新，不存在则创建
                        # 查找条件是 stock, trade_time, time_level (联合唯一键)
                        obj, created = StockScoreAnalysis.objects.update_or_create(
                            stock=stock_instance,
                            trade_time=trade_time_aware,
                            time_level=level,
                            defaults=defaults # 要更新或创建的字段值
                        )
                        return created # 返回 True 表示新创建，False 表示更新
                    except Exception as db_err:
                         # 捕获数据库操作可能出现的错误
                         logger.error(f"[{stock_code}] 保存数据到数据库时出错 (时间: {trade_time_aware}, 级别: {level}): {db_err}", exc_info=True)
                         return False # 返回 False 表示保存失败

                # --- 选择保存策略：保存最近 N 条有效记录 ---
                N = 10 # 定义要保存的最近记录数量
                # 筛选出 alignment_signal 不是 NaN 的行，这些行通常包含有效的分析结果
                valid_rows = analysis_df.dropna(subset=['alignment_signal'])
                # 从有效行中选取最后 N 行
                rows_to_save = valid_rows.iloc[-N:] if len(valid_rows) > N else valid_rows
                logger.info(f"[{stock_code}] 准备保存 {len(rows_to_save)} 条最近的有效分析记录到数据库...")
                # 创建异步任务列表
                tasks = [save_analysis_data_orm(row, analysis_time_level) for _, row in rows_to_save.iterrows()]
                # 并发执行所有保存任务
                results = await asyncio.gather(*tasks, return_exceptions=True) # return_exceptions=True 捕获任务中的异常
                # 统计保存结果
                failed_count = sum(1 for r in results if isinstance(r, Exception) or r is None or r is False) # False 也算失败或跳过
                successful_saves = len(results) - failed_count
                created_count = sum(1 for r in results if r is True) # 统计新创建的数量
                logger.info(f"[{stock_code}] 数据库保存完成: 成功 {successful_saves} 条 (新增 {created_count} 条), 失败/跳过 {failed_count} 条。")

            except StockInfo.DoesNotExist:
                 # 如果数据库中找不到对应的股票基础信息
                 logger.error(f"[{stock_code}] 无法在数据库中找到股票代码 {stock_code} 的基础信息，无法保存分析结果。")
            except Exception as e:
                # 捕获数据库操作过程中的其他未知错误
                logger.error(f"[{stock_code}] 保存分析结果到数据库时发生意外错误: {e}", exc_info=True)
        elif save_to_db and (StockScoreAnalysis is None or StockInfo is None):
             logger.warning(f"[{stock_code}] 数据库模型未加载，跳过数据库保存。")
        elif save_to_db and analysis_time_level == 'unknown':
             logger.warning(f"[{stock_code}] 分析时间级别未知，跳过数据库保存以确保数据完整性。")

        # --- 15. 结构化缓存到 Redis (使用 CacheManager - 更新缓存结构) ---
        if cache_results:
            if CacheManager: # 检查 CacheManager 是否成功导入
                logger.info(f"[{stock_code}] 开始缓存最新分析状态到 Redis (使用 CacheManager)...")
                try:
                    cache_manager = CacheManager() # 实例化 CacheManager
                    # CacheManager 内部方法会自动初始化连接
                    # 生成缓存键 (类型: analysis, 子类: latest, 标识: stock_code)
                    cache_key = cache_manager.generate_key('analysis', 'latest', stock_code)
                    # 准备要缓存的完整数据结构 (已在步骤 13.4 中更新)
                    cache_data = {
                        'stock_code': stock_code,
                        'analysis_time_level': analysis_time_level,
                        'latest_historical_data': latest_historical_data_cache, # 包含新信号
                        'realtime_data': realtime_data_cache,
                        'summary_text': summary, # 包含新信号的摘要
                        'cache_timestamp': timezone.now().isoformat() # 缓存生成时间戳
                    }
                    # 调用 CacheManager 的 set 方法进行缓存 (内部处理序列化)
                    # 设置缓存过期时间为 1 小时 (3600 秒)
                    success = await cache_manager.set(cache_key, cache_data, timeout=3600)
                    if success:
                        logger.info(f"[{stock_code}] 最新分析状态已通过 CacheManager 缓存至 Redis (Key: {cache_key})。")
                    else:
                        logger.warning(f"[{stock_code}] CacheManager 缓存最新分析状态至 Redis 返回 False (Key: {cache_key})，请检查 CacheManager 日志。")
                except ConnectionError as e:
                     logger.error(f"[{stock_code}] 缓存时 Redis 连接错误: {e}")
                except ValueError as e: # 捕获序列化错误
                     logger.error(f"[{stock_code}] 缓存时序列化错误: {e}")
                except Exception as e:
                    logger.error(f"[{stock_code}] 使用 CacheManager 缓存分析结果至 Redis 时发生未知错误: {e}", exc_info=True)
            else:
                 logger.warning(f"[{stock_code}] CacheManager 未加载，跳过 Redis 缓存。")
        # --- 结束缓存处理 ---

    else: # analysis_df 为空
        logger.warning(f"[{stock_code}] 分析 DataFrame 为空，无法进行后续处理。")
        return None

    # --- 16. 返回分析结果 DataFrame ---
    # 返回包含所有计算列的 DataFrame，调用者可以根据需要使用
    return analysis_df
# --- 结束 analyze_score_trend 函数 ---


# --- test_strategy_scores 函数 (完整版) ---
async def test_strategy_scores(stock_code: str, time_level_for_analysis: str = '5'):
    """
    测试指定股票代码的策略评分生成过程，并进行增强的趋势和 T+0 分析。
    """
    # --- 获取本地时区 ---
    local_tz = None
    local_tz_name = "系统默认"
    if tzlocal and ZoneInfo:
        try:
            local_tz = tzlocal.get_localzone()
            local_tz_name = str(local_tz)
            logger.info(f"检测到本地时区: {local_tz_name}")
        except Exception as tz_e:
            logger.warning(f"获取本地时区时出错: {tz_e}. 时间将不会转换。")
            local_tz = None
    else:
        logger.warning("'tzlocal' 或 'zoneinfo' 不可用，时间将不会转换为本地时区。")

    # 1. 初始化服务和 DAO 实例
    indicator_service = IndicatorService()
    stock_basic_dao = StockBasicDAO()

    # 2. 定义策略参数 (与之前文件一致)
    strategy_params: Dict[str, Any] = {
        'rsi_period': 12, 'rsi_oversold': 30, 'rsi_overbought': 70, 'rsi_extreme_oversold': 20, 'rsi_extreme_overbought': 80,
        'kdj_period_k': 9, 'kdj_period_d': 3, 'kdj_period_j': 3, 'kdj_oversold': 20, 'kdj_overbought': 80,
        'boll_period': 20, 'boll_std_dev': 2,
        'macd_fast': 10, 'macd_slow': 26, 'macd_signal': 9,
        'cci_period': 14, 'cci_threshold': 100, 'cci_extreme_threshold': 200,
        'mfi_period': 14, 'mfi_oversold': 20, 'mfi_overbought': 80, 'mfi_extreme_oversold': 10, 'mfi_extreme_overbought': 90,
        'roc_period': 12,
        'dmi_period': 14, 'adx_threshold': 20, 'adx_strong_threshold': 30,
        'sar_step': 0.02, 'sar_max': 0.2,
        'amount_ma_period': 20, 'obv_ma_period': 10, 'cmf_period': 20,
        'ema_period': 13,
        'weights': {'5': 0.1, '15': 0.4, '30': 0.3, '60': 0.2},
        'volume_tf': '15', # VWAP 和 Volume 使用 15 分钟周期
        'volume_confirmation': True, 'volume_confirm_boost': 1.1, 'volume_fail_penalty': 0.8, 'divergence_penalty': 0.3,
        'check_bearish_divergence': True, 'divergence_price_period': 5,
        'divergence_threshold_cmf': -0.05, 'divergence_threshold_mfi': 40,
    }
    strategy_instance = MacdRsiKdjBollEnhancedStrategy(params=strategy_params)

    # 3. 确定策略和分析所需的所有时间周期
    strategy_timeframes = strategy_instance.timeframes
    all_required_timeframes = set(strategy_timeframes)
    all_required_timeframes.add(time_level_for_analysis) # 分析用的时间级别
    volume_tf = strategy_params['volume_tf'] # 获取 VWAP/Volume 的时间级别
    all_required_timeframes.add(volume_tf)
    timeframes_list = sorted(list(all_required_timeframes), key=int) # 假设都是数字分钟级别

    # 获取股票基础信息
    stock = await stock_basic_dao.get_stock_by_code(stock_code)
    if not stock:
        logger.error(f"无法找到股票信息: {stock_code}")
        return

    # 4. 准备统一的策略数据帧 (调用 IndicatorService)
    logger.info(f"[{stock}] 正在准备统一策略数据 (周期: {timeframes_list})...")
    # --- 增加数据量以满足 233 EMA 和长周期背离计算 ---
    limit_count = 2000 # 增加请求的数据量
    strategy_df: Optional[pd.DataFrame] = await indicator_service.prepare_strategy_dataframe(
        stock_code=stock_code,
        timeframes=timeframes_list,
        strategy_params=strategy_params,
        limit_per_tf=limit_count,
        include_ohlcv=True # 确保请求 OHLCV 数据
    )

    # 5. 检查数据准备是否成功
    if strategy_df is None or strategy_df.empty:
        logger.error(f"[{stock}] 统一策略数据准备失败或为空，无法继续。")
        return
    logger.info(f"[{stock}] 统一策略数据准备完成，形状: {strategy_df.shape}")
    # logger.debug(f"[{stock}] strategy_df columns: {strategy_df.columns.tolist()}") # 调试时取消注释

    # 6. 生成评分
    logger.info(f"[{stock}] 正在生成策略评分 (0-100)...")
    scores: Optional[pd.Series] = None
    intermediate_data: Optional[pd.DataFrame] = None
    try:
        scores = strategy_instance.run(strategy_df)
        intermediate_data = strategy_instance.get_intermediate_data() # 获取策略内部计算的指标
        logger.info(f"[{stock}] 策略评分生成完成。")

        # 7. 查看评分结果和进行分析
        if scores is not None and not scores.empty:
            scores_display = scores.copy()
            # 尝试将索引转换为本地时区进行显示
            if local_tz and isinstance(scores_display.index, pd.DatetimeIndex):
                 try:
                     if scores_display.index.tz is None:
                         # 假设原始数据是 UTC 或无时区，先本地化为 UTC 再转本地
                         scores_display.index = scores_display.index.tz_localize('UTC').tz_convert(local_tz)
                     else:
                         # 如果已有其他时区，直接转换
                         scores_display.index = scores_display.index.tz_convert(local_tz)
                 except Exception as tz_convert_e:
                     logger.warning(f"转换评分时间到本地时区失败: {tz_convert_e}")

            print(f"\n[{stock}] 最新的评分 (最后10条，时间：{local_tz_name}):")
            print(scores_display.tail(10).round(2))
            print("\n评分统计描述:")
            print(scores.describe().round(2))
            nan_count = scores.isna().sum()
            if nan_count > 0: print(f"\n警告: 生成的评分中包含 {nan_count} 个 NaN 值。")

            # --- 开始进行趋势和 T+0 分析 ---
            logger.info(f"[{stock}] 开始准备分析输入 (使用已获取的数据)...")

            # --- 准备分析输入 DataFrame ---
            # 目标是创建一个包含 score, close, vwap, open, high, low, volume 及所需指标的 DataFrame
            # 这些数据应该都在 strategy_df 或 intermediate_data 中

            # a. 确定分析所需的列名
            # 使用 time_level_for_analysis 对应的 OHLCV
            ohlcv_cols = {
                'open': f'open_{time_level_for_analysis}',
                'high': f'high_{time_level_for_analysis}',
                'low': f'low_{time_level_for_analysis}',
                'close': f'close_{time_level_for_analysis}',
                'volume': f'volume_{time_level_for_analysis}' # 通常成交量也用分析级别
            }
            vwap_col = f'vwap_{volume_tf}' # VWAP 使用 volume_tf 指定的周期

            # b. 从 strategy_df 和 intermediate_data 提取数据
            analysis_input_data = {'score': scores}
            missing_input_cols = []

            # 添加 OHLCV
            for key, col_name in ohlcv_cols.items():
                if col_name in strategy_df.columns:
                    analysis_input_data[key] = strategy_df[col_name]
                else:
                    missing_input_cols.append(col_name)
                    analysis_input_data[key] = pd.Series(dtype=float) # 添加空列以防出错

            # 添加 VWAP
            if vwap_col in strategy_df.columns:
                analysis_input_data['vwap'] = strategy_df[vwap_col]
            else:
                missing_input_cols.append(vwap_col)
                analysis_input_data['vwap'] = pd.Series(dtype=float)

            # 添加策略中间计算的指标 (如果存在且分析需要)
            # 注意：intermediate_data 的列名可能与 pandas_ta 不同，需要适配
            if intermediate_data is not None:
                # 假设 intermediate_data 包含 'MACDh', 'RSI', 'MFI', 'OBV', 'CCI', 'STOCHk', 'STOCHd', 'BBL', 'BBU' 等
                # 将这些列合并到 analysis_input_data，注意索引对齐
                for col in intermediate_data.columns:
                    if col not in analysis_input_data: # 避免覆盖 score 等核心列
                         # 尝试直接合并，如果索引不完全匹配，reindex
                         if intermediate_data[col].index.equals(scores.index):
                             analysis_input_data[col] = intermediate_data[col]
                         else:
                             try:
                                 # 使用 reindex 并指定填充方法可能更好
                                 analysis_input_data[col] = intermediate_data[col].reindex(scores.index, method='ffill')
                             except Exception as reindex_err:
                                 logger.warning(f"[{stock}] Reindexing intermediate column '{col}' failed: {reindex_err}")


            if missing_input_cols:
                 logger.warning(f"[{stock}] 准备分析输入时缺少列: {missing_input_cols}。分析可能不完整。")


            # c. 创建用于分析的 DataFrame
            score_price_vwap_df = pd.DataFrame() # 初始化为空
            try:
                # 使用 score 的索引作为基准
                common_index = scores.index
                aligned_data = {}
                for key, series in analysis_input_data.items():
                     if isinstance(series, pd.Series):
                         if not series.index.equals(common_index):
                             logger.debug(f"Aligning index for column: {key}")
                             # 使用 reindex 并填充 NaN，避免丢失数据或引入未来数据
                             aligned_data[key] = series.reindex(common_index)
                         else:
                             aligned_data[key] = series
                     else: # 处理非 Series 数据 (理论上不应发生)
                         aligned_data[key] = pd.Series(series, index=common_index)


                score_price_vwap_df = pd.DataFrame(aligned_data)
                # 删除 score 或 close 为 NaN 的行，这些是分析的基础
                score_price_vwap_df.dropna(subset=['score', 'close'], how='any', inplace=True)

            except Exception as concat_err:
                 logger.error(f"[{stock}] 创建分析 DataFrame 时出错: {concat_err}", exc_info=True)
                 # score_price_vwap_df 保持为空


            if not score_price_vwap_df.empty:
                logger.info(f"[{stock}] 分析输入数据准备完成 (数据条数: {len(score_price_vwap_df)})，开始调用 analyze_score_trend...")
                # d. 定义 T+0 参数
                t0_settings = {
                    'enabled': True,
                    'buy_dev_threshold': -0.003,
                    'sell_dev_threshold': 0.005,
                    'use_long_term_filter': True
                }
                # e. 定义分析参数 (可以从配置读取，这里使用默认值或从 strategy_params 映射)
                analysis_settings = {
                    # 从 strategy_params 映射指标参数
                    'macd_fast': strategy_params['macd_fast'],
                    'macd_slow': strategy_params['macd_slow'],
                    'macd_signal': strategy_params['macd_signal'],
                    'rsi_period': strategy_params['rsi_period'],
                    'rsi_ob': strategy_params['rsi_overbought'],
                    'rsi_os': strategy_params['rsi_oversold'],
                    'stoch_k': strategy_params['kdj_period_k'],
                    'stoch_d': strategy_params['kdj_period_d'],
                    'stoch_smooth_k': strategy_params['kdj_period_j'], # KDJ 的 J 线参数通常用于平滑 K
                    'stoch_ob': strategy_params['kdj_overbought'],
                    'stoch_os': strategy_params['kdj_oversold'],
                    'cci_period': strategy_params['cci_period'],
                    'cci_ob': strategy_params['cci_threshold'],
                    'cci_os': -strategy_params['cci_threshold'],
                    'mfi_period': strategy_params['mfi_period'],
                    'mfi_ob': strategy_params['mfi_overbought'],
                    'mfi_os': strategy_params['mfi_oversold'],
                    'boll_period': strategy_params['boll_period'],
                    'boll_std': strategy_params['boll_std_dev'],
                    'volume_ma_period': strategy_params['amount_ma_period'], # 使用成交额均线周期
                    # 可以覆盖默认权重或阈值
                    # 'confirmation_weighted_threshold': 3.5,
                    # 'signal_weights': { 'rsi_regular_div': 1.5 } # 示例：提高 RSI 背离权重
                }
                # 合并默认权重和自定义权重 (如果 analysis_settings 中定义了 signal_weights)
                # 获取 analyze_score_trend 中的默认参数
                default_analysis_params_for_merge = {
                    'confirmation_weighted_threshold': 3.0,
                    'signal_weights': {
                        'alignment_reversal': 1.0, 'macd_regular_div': 1.5, 'macd_hidden_div': 0.8,
                        'rsi_regular_div': 1.2, 'rsi_hidden_div': 0.7, 'mfi_regular_div': 1.0,
                        'mfi_hidden_div': 0.6, 'obv_regular_div': 0.8, 'obv_hidden_div': 0.5,
                        'rsi_obos_reversal': 1.0, 'stoch_obos_reversal': 0.9, 'cci_obos_reversal': 0.8,
                        'bb_reversal': 0.7, 'volume_spike_confirm': 0.5, 'kline_engulfing': 1.0,
                        'kline_hammer_hanging': 1.2, 'kline_star': 1.5, 'kline_marubozu': 0.5,
                        'kline_doji': 0.3,
                    }
                }
                final_analysis_params = default_analysis_params_for_merge.copy() # Start with defaults from analyze_score_trend
                final_analysis_params.update(analysis_settings) # Update with specific settings from test_strategy_scores
                if 'signal_weights' in analysis_settings: # Merge weights if provided in test_strategy_scores
                    final_analysis_params['signal_weights'] = {
                        **default_analysis_params_for_merge['signal_weights'],
                        **analysis_settings['signal_weights']
                    }


                # f. 调用更新后的分析函数
                analysis_result_df = await analyze_score_trend(
                    stock_code=str(stock_code),
                    score_price_vwap_df=score_price_vwap_df, # 传入包含所有数据的 DF
                    t0_params=t0_settings,
                    analysis_time_level=time_level_for_analysis, # 明确传递分析级别
                    analysis_params=final_analysis_params, # 传递最终合并的参数
                    save_to_db=True, # 启用数据库保存
                    cache_results=True # 启用 Redis 缓存
                )
                # analysis_result_df 可以在这里进一步使用或检查
                if analysis_result_df is not None:
                    logger.info(f"[{stock}] analyze_score_trend 完成，返回 DataFrame 形状: {analysis_result_df.shape}")
                    # print(analysis_result_df.tail()) # 打印最后几行结果用于调试
                else:
                    logger.error(f"[{stock}] analyze_score_trend 返回 None，分析失败。")

            else:
                logger.warning(f"[{stock}] 准备用于分析的数据为空或失败，跳过趋势分析。")

        else:
            logger.error(f"\n[{stock}] 未能获取有效的评分结果 (scores is None or empty)。")

    except ValueError as ve:
        logger.error(f"[{stock}] 生成评分或分析时发生值错误: {ve}", exc_info=True)
    except Exception as e:
        logger.error(f"[{stock}] 生成评分或分析时发生未知错误: {e}", exc_info=True)


# --- Django Management Command 类 ---
class Command(BaseCommand):
    help = '测试策略评分、趋势分析及基于 VWAP 的 T+0 信号 (包含多种技术指标反转检测和 K 线形态)'

    def add_arguments(self, parser):
        parser.add_argument('stock_code', type=str, help='要测试的股票代码 (例如: 000001)')
        parser.add_argument(
            '--level',
            type=str,
            default='5',
            help='用于分析和选取 OHLCV 列的时间级别 (例如: 1, 5, 15, 30, 60, D)'
        )

    def handle(self, *args, **options):
        stock_code_to_test = options['stock_code']
        time_level = options['level']

        self.stdout.write(self.style.SUCCESS(f'开始测试策略评分及增强趋势分析 for {stock_code_to_test} (分析级别: {time_level})...'))

        try:
            asyncio.run(test_strategy_scores(
                stock_code=stock_code_to_test,
                time_level_for_analysis=time_level
            ))
            self.stdout.write(self.style.SUCCESS(f'测试完成 for {stock_code_to_test}.'))
        except Exception as e:
            logger.error(f"命令执行期间发生错误 for {stock_code_to_test}: {e}", exc_info=True)
            self.stderr.write(self.style.ERROR(f'测试过程中发生错误: {e}'))


# --- 更新：读取并展示缓存数据的函数 (使用 CacheManager) ---
async def display_cached_analysis(stock_code: str):
    """
    从 Redis 读取指定股票的最新分析缓存数据 (使用 CacheManager) 并格式化输出。
    """
    if CacheManager: # 检查 CacheManager 是否可用
        cache_manager = CacheManager()
        # 使用与存储时相同的逻辑生成键
        cache_key = cache_manager.generate_key('analysis', 'latest', stock_code)
        logger.info(f"尝试从 Redis 读取缓存数据 (使用 CacheManager, Key: {cache_key})...")

        try:
            # 调用 CacheManager 的 get 方法，它处理反序列化 (umsgpack)
            cached_data = await cache_manager.get(cache_key)

            if cached_data:
                # get 方法成功时返回反序列化后的 Python 对象 (通常是字典)
                if isinstance(cached_data, dict):
                    logger.info(f"成功读取并反序列化缓存数据 (缓存时间: {cached_data.get('cache_timestamp', 'N/A')})。")

                    # --- 输出方式一：直接打印缓存的摘要文本 ---
                    print("\n" + "="*30 + f" {stock_code} 缓存分析摘要 (来自 Redis - CacheManager) " + "="*30)
                    print(cached_data.get('summary_text', '缓存中未找到摘要文本。'))
                    print("="* (62 + len(stock_code) + len(" 缓存分析摘要 (来自 Redis - CacheManager) ")))

                    # --- 输出方式二：根据结构化数据动态生成格式化输出 (可选调试) ---
                    # print("\n" + "="*30 + f" {stock_code} 结构化缓存数据详情 " + "="*30)
                    # hist_data = cached_data.get('latest_historical_data', {})
                    # rt_data = cached_data.get('realtime_data', {})
                    # print(f"分析级别: {cached_data.get('analysis_time_level', 'N/A')}")
                    # import json
                    # print("--- 最新历史数据 ---")
                    # print(json.dumps(hist_data, indent=2, ensure_ascii=False, default=str)) # 使用 default=str 处理无法序列化的类型
                    # print("--- 实时数据 ---")
                    # print(json.dumps(rt_data, indent=2, ensure_ascii=False, default=str))
                    # print("="* (62 + len(stock_code) + len(" 结构化缓存数据详情 ")))
                else:
                    logger.error(f"[{stock_code}] 从 CacheManager 获取的数据类型不是预期的字典: type={type(cached_data)}")
                    print(f"\n错误：股票 {stock_code} 的缓存数据格式不正确。")

            else:
                # get 方法返回 None 表示缓存未命中
                logger.warning(f"[{stock_code}] 在 Redis 中未找到缓存数据 (使用 CacheManager, Key: {cache_key})。")
                print(f"\n未找到股票 {stock_code} 的缓存分析数据。")

        except ConnectionError as e:
             logger.error(f"[{stock_code}] 读取缓存时 Redis 连接错误: {e}")
             print(f"\n错误：读取股票 {stock_code} 的缓存时连接 Redis 失败。")
        except Exception as e:
             logger.error(f"[{stock_code}] 处理 Redis 缓存数据时发生错误: {e}", exc_info=True)
             print(f"\n错误：处理股票 {stock_code} 的缓存数据时发生错误。")

    else:
        # CacheManager 不可用
        logger.warning(f"[{stock_code}] CacheManager 未加载，无法读取缓存。")
        print(f"\n错误：CacheManager 未加载，无法读取股票 {stock_code} 的缓存。")

