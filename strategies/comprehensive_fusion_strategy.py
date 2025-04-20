import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Tuple
import logging
from scipy.signal import find_peaks # 需要 scipy
import warnings

# 假设 BaseStrategy 和常量在 .base 或 core.constants
from .base import BaseStrategy
# from core.constants import TimeLevel # 如果需要

logger = logging.getLogger("strategy")

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

# --- 辅助函数区 (来自 test_strategy_signals.py) ---
def get_find_peaks_params(time_level: str, base_lookback: int) -> Dict[str, Any]:
    """
    根据时间级别和基础回看期，返回适用于 find_peaks 的参数。
    短周期更敏感，长周期过滤更多噪音。
    (与原 test_strategy_signals.py 中的实现相同)
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

def detect_divergence(price: pd.Series,
                      indicator: pd.Series,
                      lookback: int = 14,
                      find_peaks_params: Dict[str, Any] = {'distance': 7, 'prominence_factor': 0.5},
                      check_regular_bullish: bool = True,
                      check_regular_bearish: bool = True,
                      check_hidden_bullish: bool = True,
                      check_hidden_bearish: bool = True
                      ) -> pd.Series:
    """
    检测价格与指标之间的常规背离和隐藏背离。
    信号值: 1 (常规牛), -1 (常规熊), 2 (隐藏牛), -2 (隐藏熊), 0 (无)。
    (与原 test_strategy_signals.py 中的实现类似，增加了启用/禁用选项)
    """
    divergence_signal = pd.Series(0, index=price.index)
    if price.isnull().all() or indicator.isnull().all() or len(price) < lookback * 2:
        return divergence_signal

    # --- 准备 find_peaks 参数 ---
    distance = find_peaks_params.get('distance', max(1, lookback // 2))
    prominence_factor = find_peaks_params.get('prominence_factor', 0.5)

    # 计算最小显著性
    min_prominence_price = (price.rolling(lookback).std() * prominence_factor).fillna(0).values
    min_prominence_indicator = (indicator.rolling(lookback).std() * prominence_factor).fillna(0).values

    indicator_filled = indicator.ffill().bfill()
    if indicator_filled.isnull().all():
        return divergence_signal

    # --- 查找峰值和谷值 ---
    try:
        min_prominence_price = np.maximum(min_prominence_price, 0)
        min_prominence_indicator = np.maximum(min_prominence_indicator, 0)

        price_peaks, _ = find_peaks(price, distance=distance, prominence=min_prominence_price)
        price_troughs, _ = find_peaks(-price, distance=distance, prominence=min_prominence_price)
        indicator_peaks, _ = find_peaks(indicator_filled, distance=distance, prominence=min_prominence_indicator)
        indicator_troughs, _ = find_peaks(-indicator_filled, distance=distance, prominence=min_prominence_indicator)
    except Exception as fp_err:
        logger.warning(f"find_peaks encountered an error: {fp_err}. Skipping divergence.")
        return divergence_signal

    # --- 检测背离 ---
    if len(price_peaks) >= 2 and len(indicator_peaks) >= 2:
        p_peak1_idx, p_peak2_idx = price_peaks[-2], price_peaks[-1]
        ind_peaks_near_p1 = indicator_peaks[(indicator_peaks >= p_peak1_idx - distance//2) & (indicator_peaks <= p_peak1_idx + distance//2)]
        ind_peaks_near_p2 = indicator_peaks[(indicator_peaks >= p_peak2_idx - distance//2) & (indicator_peaks <= p_peak2_idx + distance//2)]
        if len(ind_peaks_near_p1) > 0 and len(ind_peaks_near_p2) > 0:
            i_peak1_idx, i_peak2_idx = ind_peaks_near_p1[-1], ind_peaks_near_p2[-1]
            if pd.notna(indicator_filled.iloc[i_peak2_idx]) and pd.notna(indicator_filled.iloc[i_peak1_idx]):
                # 常规看跌
                if check_regular_bearish and price.iloc[p_peak2_idx] > price.iloc[p_peak1_idx] and indicator_filled.iloc[i_peak2_idx] < indicator_filled.iloc[i_peak1_idx]:
                    divergence_signal.iloc[p_peak2_idx] = -1
                # 隐藏看跌
                elif check_hidden_bearish and price.iloc[p_peak2_idx] < price.iloc[p_peak1_idx] and indicator_filled.iloc[i_peak2_idx] > indicator_filled.iloc[i_peak1_idx]:
                    if divergence_signal.iloc[p_peak2_idx] == 0: # 避免覆盖常规信号
                         divergence_signal.iloc[p_peak2_idx] = -2

    if len(price_troughs) >= 2 and len(indicator_troughs) >= 2:
        p_trough1_idx, p_trough2_idx = price_troughs[-2], price_troughs[-1]
        ind_troughs_near_p1 = indicator_troughs[(indicator_troughs >= p_trough1_idx - distance//2) & (indicator_troughs <= p_trough1_idx + distance//2)]
        ind_troughs_near_p2 = indicator_troughs[(indicator_troughs >= p_trough2_idx - distance//2) & (indicator_troughs <= p_trough2_idx + distance//2)]
        if len(ind_troughs_near_p1) > 0 and len(ind_troughs_near_p2) > 0:
            i_trough1_idx, i_trough2_idx = ind_troughs_near_p1[-1], ind_troughs_near_p2[-1]
            if pd.notna(indicator_filled.iloc[i_trough2_idx]) and pd.notna(indicator_filled.iloc[i_trough1_idx]):
                # 常规看涨
                if check_regular_bullish and price.iloc[p_trough2_idx] < price.iloc[p_trough1_idx] and indicator_filled.iloc[i_trough2_idx] > indicator_filled.iloc[i_trough1_idx]:
                    divergence_signal.iloc[p_trough2_idx] = 1
                # 隐藏看涨
                elif check_hidden_bullish and price.iloc[p_trough2_idx] > price.iloc[p_trough1_idx] and indicator_filled.iloc[i_trough2_idx] < indicator_filled.iloc[i_trough1_idx]:
                     if divergence_signal.iloc[p_trough2_idx] == 0: # 避免覆盖常规信号
                          divergence_signal.iloc[p_trough2_idx] = 2

    return divergence_signal.astype(int)

def detect_kline_patterns(df: pd.DataFrame) -> pd.Series:
    """
    检测基本的 K 线形态。
    (与原 test_strategy_signals.py 中的实现基本相同)
    信号值: 1 (看涨吞没), -1 (看跌吞没), 2 (锤子线), -2 (上吊线),
            3 (早晨星), -3 (黄昏星), 5 (十字星), 10 (光头阳), -10 (光头阴), 0 (无)
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
    if df_ohlc.empty:
        return patterns

    o, h, l, c = df_ohlc['open'], df_ohlc['high'], df_ohlc['low'], df_ohlc['close']
    o1, h1, l1, c1 = o.shift(1), h.shift(1), l.shift(1), c.shift(1)
    o2, c2 = o.shift(2), c.shift(2)

    body = abs(c - o)
    body1 = abs(c1 - o1).fillna(0)
    body2 = abs(c2 - o2).fillna(0)
    full_range = (h - l).replace(0, 1e-6)
    full_range1 = (h1 - l1).fillna(0).replace(0, 1e-6)
    upper_shadow = h - np.maximum(c, o)
    lower_shadow = np.minimum(o, c) - l
    is_green = c > o
    is_red = c < o
    is_green1 = c1 > o1
    is_red1 = c1 < o1
    is_green2 = c2 > o2
    is_red2 = c2 < o2

    # --- 吞没 ---
    bull_engulf = is_red1 & is_green & (c > o1) & (o < c1) & (body > body1 * 1.01)
    patterns.loc[bull_engulf[bull_engulf].index] = 1
    bear_engulf = is_green1 & is_red & (o > c1) & (c < o1) & (body > body1 * 1.01)
    patterns.loc[bear_engulf[bear_engulf].index] = -1

    # --- 锤子/上吊 ---
    small_body_threshold = full_range * 0.2
    long_lower_shadow = lower_shadow >= 2 * body
    short_upper_shadow = upper_shadow < 0.5 * body
    hammer_like = (body > 1e-6) & (body < small_body_threshold) & long_lower_shadow & short_upper_shadow
    is_hammer = hammer_like & (is_red1 | is_red2)
    patterns.loc[is_hammer[is_hammer].index] = 2
    is_hanging = hammer_like & (is_green1 | is_green2)
    patterns.loc[is_hanging[is_hanging].index] = -2

    # --- 星线 ---
    star_body_threshold = full_range1 * 0.3
    is_star = (body1 < star_body_threshold) & (body1 > 1e-6)
    gap_down1 = np.minimum(o1, c1) < np.minimum(o2, c2)
    gap_up1 = np.maximum(o1, c1) > np.maximum(o2, c2)
    gap_down2 = np.minimum(o, c) < np.minimum(o1, c1)
    gap_up2 = np.minimum(o, c) > np.maximum(o1, c1)
    morning_star = is_red2 & (body2 > body * 1.5) & is_star & gap_down1 & is_green & (body > body2 * 0.5) & gap_up2 & (c > (o2 + c2) / 2)
    patterns.loc[morning_star[morning_star].index] = 3
    evening_star = is_green2 & (body2 > body * 1.5) & is_star & gap_up1 & is_red & (body > body2 * 0.5) & gap_down2 & (c < (o2 + c2) / 2)
    patterns.loc[evening_star[evening_star].index] = -3

    # --- 十字星 ---
    doji_threshold = full_range * 0.05
    is_doji = (body <= doji_threshold) & (body > 1e-6)
    patterns.loc[is_doji[is_doji].index] = 5

    # --- 光头光脚 ---
    shadow_threshold_factor = 0.05
    no_upper_shadow = upper_shadow < body * shadow_threshold_factor
    no_lower_shadow = lower_shadow < body * shadow_threshold_factor
    is_marubozu = no_upper_shadow & no_lower_shadow & (body > full_range * 0.9)
    bull_marubozu = is_marubozu & is_green
    patterns.loc[bull_marubozu[bull_marubozu].index] = 10
    bear_marubozu = is_marubozu & is_red
    patterns.loc[bear_marubozu[bear_marubozu].index] = -10

    patterns_aligned = pd.Series(0, index=df.index)
    patterns_aligned.update(patterns)
    return patterns_aligned.astype(int)

# --- 主策略类 ---
class ComprehensiveFusionStrategy(BaseStrategy):
    """
    融合 MACD/RSI/KDJ/BOLL 多时间框架评分与趋势/背离/K线形态分析的综合策略。
    策略流程:
    1. 加载外部 JSON 配置参数。
    2. 接收由 IndicatorService 准备好的包含基础指标的 DataFrame。
    3. 计算多时间框架加权基础评分 (0-100)。
    4. 对基础评分应用量能确认调整。
    5. 进行趋势分析 (EMA 排列, 长期背景)。
    6. 进行背离检测 (常规和隐藏)。
    7. 进行 K 线形态检测。
    8. 计算技术指标反转信号 (超买超卖, 布林带)。
    9. 计算综合确认信号强度 (加权求和)。
    10. 生成最终信号 (可能包括 T+0 建议)。
    """
    strategy_name = "ComprehensiveFusionStrategy" # 会被 JSON 覆盖

    def __init__(self, params_file: str = "strategies/indicator_parameters.json"):
        """
        初始化策略，从 JSON 文件加载参数。
        :param params_file: 参数 JSON 文件的路径。
        """
        self.params_file = params_file
        self.params = self._load_params()
        self.strategy_name = self.params.get('strategy_name', self.strategy_name) # 从 JSON 更新名称
        # 定义一些常量或从参数获取
        self.timeframes = self.params['base_scoring']['timeframes']
        self.score_indicators = self.params['base_scoring']['score_indicators']
        self._num_score_indicators = len(self.score_indicators)
        self.intermediate_data: Optional[pd.DataFrame] = None
        self.analysis_results: Optional[pd.DataFrame] = None # 存储分析步骤的结果

        if ta is None:
            logger.error(f"[{self.strategy_name}] pandas_ta 未加载，策略无法运行。")
            # 在实际应用中，可能需要更健壮的错误处理
            raise ImportError("pandas_ta is required but not installed or loaded.")

        super().__init__(self.params) # 调用基类初始化，会触发 _validate_params

    def _load_params(self) -> Dict[str, Any]:
        """从 JSON 文件加载参数"""
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"参数文件未找到: {self.params_file}")
        try:
            with open(self.params_file, 'r', encoding='utf-8') as f:
                params = json.load(f)
            logger.info(f"成功从 {self.params_file} 加载策略参数。")
            return params
        except json.JSONDecodeError as e:
            logger.error(f"解析参数文件 {self.params_file} 失败: {e}")
            raise ValueError(f"无法解析参数文件: {e}")
        except Exception as e:
            logger.error(f"加载参数文件 {self.params_file} 时发生未知错误: {e}")
            raise

    def _validate_params(self):
        """验证从 JSON 加载的参数"""
        super()._validate_params() # 调用基类的验证 (如果需要)

        # 验证 base_scoring
        bs_params = self.params.get('base_scoring')
        if not bs_params or not isinstance(bs_params, dict):
            raise ValueError("参数 'base_scoring' 缺失或格式错误")
        if not bs_params.get('timeframes') or not isinstance(bs_params['timeframes'], list):
            raise ValueError("'base_scoring.timeframes' 必须是一个非空列表")
        if not bs_params.get('score_indicators') or not isinstance(bs_params['score_indicators'], list):
             raise ValueError("'base_scoring.score_indicators' 必须是一个列表")
        if 'weights' not in bs_params or not isinstance(bs_params['weights'], dict) or \
           set(bs_params['weights'].keys()) != set(bs_params['timeframes']) or \
           abs(sum(bs_params['weights'].values()) - 1.0) > 1e-6:
            raise ValueError("'base_scoring.weights' 必须是包含所有 timeframes 且总和为 1.0 的字典")
        if self._num_score_indicators == 0:
             logger.warning("参数 'score_indicators' 为空列表，基础评分将始终为 50。") # 改为警告

        # 验证 volume_confirmation
        vc_params = self.params.get('volume_confirmation')
        if not vc_params or not isinstance(vc_params, dict):
             raise ValueError("参数 'volume_confirmation' 缺失或格式错误")
        if vc_params['enabled'] and vc_params['tf'] not in bs_params['timeframes']:
             raise ValueError(f"参数 'volume_confirmation.tf' ({vc_params['tf']}) 必须是 base_scoring.timeframes 中的一个")

        # 验证 divergence_detection
        dd_params = self.params.get('divergence_detection')
        if not dd_params or not isinstance(dd_params, dict):
             raise ValueError("参数 'divergence_detection' 缺失或格式错误")
        if dd_params['enabled'] and dd_params['tf'] not in bs_params['timeframes']:
            raise ValueError(f"参数 'divergence_detection.tf' ({dd_params['tf']}) 必须是 base_scoring.timeframes 中的一个")
        if 'indicators' not in dd_params or not isinstance(dd_params['indicators'], dict):
            raise ValueError("'divergence_detection.indicators' 必须是一个字典")

        # 验证 kline_pattern_detection
        kpd_params = self.params.get('kline_pattern_detection')
        if not kpd_params or not isinstance(kpd_params, dict):
             raise ValueError("参数 'kline_pattern_detection' 缺失或格式错误")
        if kpd_params['enabled'] and kpd_params['tf'] not in bs_params['timeframes']:
            raise ValueError(f"参数 'kline_pattern_detection.tf' ({kpd_params['tf']}) 必须是 base_scoring.timeframes 中的一个")

        # 验证 trend_analysis
        ta_params = self.params.get('trend_analysis')
        if not ta_params or not isinstance(ta_params, dict):
             raise ValueError("参数 'trend_analysis' 缺失或格式错误")
        if not ta_params.get('ema_periods') or not isinstance(ta_params['ema_periods'], list):
             raise ValueError("'trend_analysis.ema_periods' 必须是一个列表")
        if ta_params.get('long_term_ema_period') not in ta_params['ema_periods']:
             raise ValueError("'trend_analysis.long_term_ema_period' 必须在 'ema_periods' 列表中")

        # 验证 final_signal_generation
        fsg_params = self.params.get('final_signal_generation')
        if not fsg_params or not isinstance(fsg_params, dict):
             raise ValueError("参数 'final_signal_generation' 缺失或格式错误")
        if 'signal_weights' not in fsg_params or not isinstance(fsg_params['signal_weights'], dict):
             raise ValueError("'final_signal_generation.signal_weights' 必须是一个字典")

        logger.info("策略参数验证通过。")

    def get_required_columns(self) -> List[str]:
        """
        返回策略运行所需的 DataFrame 列名。
        假设 IndicatorService 会根据 JSON 参数提供这些列。
        """
        required = set()
        bs_params = self.params['base_scoring']
        vc_params = self.params['volume_confirmation']
        dd_params = self.params['divergence_detection']
        kpd_params = self.params['kline_pattern_detection']
        ia_params = self.params['indicator_analysis_params']
        t0_params = self.params['t_plus_0_signals']

        # 基础 OHLCV (假定 K 线形态检测的时间框需要这些)
        kline_tf = kpd_params['tf']
        required.update([f'open_{kline_tf}', f'high_{kline_tf}', f'low_{kline_tf}', f'close_{kline_tf}', f'volume_{kline_tf}'])

        # 基础评分指标
        for tf in self.timeframes:
            if 'macd' in self.score_indicators:
                required.update([f'MACD_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}',
                                 f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}',
                                 f'MACDs_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'])
            if 'rsi' in self.score_indicators:
                required.add(f'RSI_{bs_params["rsi_period"]}_{tf}')
            if 'kdj' in self.score_indicators:
                required.update([f'K_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{tf}', # 调整以匹配 IndicatorService 输出 Key 格式 (K_period)
                                 f'D_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{tf}', # 调整以匹配 IndicatorService 输出 Key 格式 (D_period)
                                 f'J_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{tf}'])# 调整以匹配 IndicatorService 输出 Key 格式 (J_period)
                                 #f'K_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{bs_params["kdj_period_j"]}_{tf}', # pandas_ta 命名
                                 #f'D_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{bs_params["kdj_period_j"]}_{tf}',
                                 #f'J_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{bs_params["kdj_period_j"]}_{tf}'])
            if 'boll' in self.score_indicators:
                 # IndicatorService 输出为 BB_UPPER, BB_MIDDLE, BB_LOWER (无周期)
                 required.update([f'BB_UPPER_{tf}', f'BB_MIDDLE_{tf}', f'BB_LOWER_{tf}'])
                 # pandas_ta bbands 命名
                 # required.update([f'BBU_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{tf}',
                 #                  f'BBM_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{tf}',
                 #                  f'BBL_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{tf}'])
            if 'cci' in self.score_indicators:
                # IndicatorService 输出为 CCI_period
                required.add(f'CCI_{bs_params["cci_period"]}_{tf}')
                # pandas_ta cci 命名
                # required.add(f'CCI_{bs_params["cci_period"]}_0.015_{tf}')
            if 'mfi' in self.score_indicators:
                required.add(f'MFI_{bs_params["mfi_period"]}_{tf}')
            if 'roc' in self.score_indicators:
                required.add(f'ROC_{bs_params["roc_period"]}_{tf}')
            if 'dmi' in self.score_indicators:
                 # IndicatorService 输出为 +DI_period, -DI_period, ADX_period
                 required.update([f'+DI_{bs_params["dmi_period"]}_{tf}', f'-DI_{bs_params["dmi_period"]}_{tf}', f'ADX_{bs_params["dmi_period"]}_{tf}'])
                 # pandas_ta adx 命名
                 # required.update([f'DMP_{bs_params["dmi_period"]}_{tf}', f'DMN_{bs_params["dmi_period"]}_{tf}', f'ADX_{bs_params["dmi_period"]}_{tf}'])
            if 'sar' in self.score_indicators:
                required.add(f'SAR_{tf}') # IndicatorService 输出为 SAR (无参数)

            # 需要相应时间周期的收盘价用于比较 (如 BOLL, SAR)
            required.add(f'close_{tf}')

        # 量能确认指标
        if vc_params['enabled']:
            vol_tf = vc_params['tf']
            required.add(f'close_{vol_tf}') # 需要价格
            required.add(f'high_{vol_tf}') # 可能用于 CMF/MFI 背离
            required.add(f'low_{vol_tf}')  # 可能用于 CMF/MFI 背离
            required.add(f'amount_{vol_tf}') # 需要原始成交额
            required.add(f'AMT_MA_{vc_params["amount_ma_period"]}_{vol_tf}') # IndicatorService 输出为 AMT_MA_period
            required.add(f'CMF_{vc_params["cmf_period"]}_{vol_tf}') # IndicatorService 输出为 CMF_period
            required.add(f'OBV_{vol_tf}') # IndicatorService 输出为 OBV (无周期)
            # 假设 OBV MA 是单独计算的或在 IndicatorService 准备阶段完成
            required.add(f'OBV_MA_{vc_params["obv_ma_period"]}_{vol_tf}') # 需要 OBV 的移动平均

        # 背离检测指标
        if dd_params['enabled']:
            div_tf = dd_params['tf']
            required.add(f'close_{div_tf}') # 背离比较的价格基准
            if dd_params['indicators'].get('macd_hist'):
                required.add(f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{div_tf}')
            if dd_params['indicators'].get('rsi'):
                required.add(f'RSI_{bs_params["rsi_period"]}_{div_tf}')
            if dd_params['indicators'].get('mfi'):
                required.add(f'MFI_{bs_params["mfi_period"]}_{div_tf}')
            if dd_params['indicators'].get('obv'):
                required.add(f'OBV_{div_tf}')

        # 技术指标反转信号所需指标 (假定使用分析时间框)
        analysis_tf = vc_params.get('tf', '15') # 默认用量能确认的时间框
        required.add(f'RSI_{bs_params["rsi_period"]}_{analysis_tf}')
        required.add(f'CCI_{bs_params["cci_period"]}_{analysis_tf}')
        # required.add(f'STOCHk_{ia_params["stoch_k"]}_{ia_params["stoch_d"]}_{ia_params["stoch_smooth_k"]}_{analysis_tf}') # Stoch 可能需要独立计算或由 Service 提供
        required.add(f'STOCH_K_{ia_params["stoch_k"]}_{analysis_tf}') # IndicatorService 输出为 STOCH_K_period
        required.add(f'BB_UPPER_{analysis_tf}') # IndicatorService 输出
        required.add(f'BB_LOWER_{analysis_tf}') # IndicatorService 输出
        required.add(f'close_{analysis_tf}')
        required.add(f'volume_{analysis_tf}')
        # required.add(f'volume_ma_{analysis_tf}') # 需要成交量均线
        required.add(f'VOL_MA_{ia_params["volume_ma_period"]}_{analysis_tf}') # IndicatorService 输出为 VOL_MA_period


        # T+0 信号所需
        if t0_params['enabled']:
            # 假设 IndicatorService 提供了 VWAP (通常是日内才有意义)
            required.add(f'vwap_{analysis_tf}') # 或者直接是 'vwap' (如果 VWAP 不分时间框)

        return list(required)

    # --- 内部评分和分析方法 ---

    def _get_macd_score(self, diff: pd.Series, dea: pd.Series, macd: pd.Series) -> pd.Series:
        """MACD 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=diff.index)
        buy_cross = (macd.shift(1) < 0) & (macd > 0)
        score.loc[buy_cross] = 75.0
        sell_cross = (macd.shift(1) > 0) & (macd < 0)
        score.loc[sell_cross] = 25.0
        # 可根据需要添加更多条件，例如零轴上方/下方
        return score

    def _get_rsi_score(self, rsi: pd.Series) -> pd.Series:
        """RSI 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=rsi.index)
        p = self.params['base_scoring']
        os, ob = p['rsi_oversold'], p['rsi_overbought']
        ext_os, ext_ob = p['rsi_extreme_oversold'], p['rsi_extreme_overbought']

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

    def _get_kdj_score(self, k: pd.Series, d: pd.Series, j: pd.Series) -> pd.Series:
        """KDJ 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=k.index)
        p = self.params['base_scoring']
        os, ob = p['kdj_oversold'], p['kdj_overbought']

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

    def _get_boll_score(self, close: pd.Series, upper: pd.Series, mid: pd.Series, lower: pd.Series) -> pd.Series:
        """BOLL 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=close.index)
        score.loc[close < lower] = 90.0
        buy_support = (close.shift(1) < lower.shift(1)) & (close >= lower)
        score.loc[buy_support] = 80.0

        score.loc[close > upper] = 10.0
        sell_pressure = (close.shift(1) > upper.shift(1)) & (close <= upper)
        score.loc[sell_pressure] = 20.0

        buy_mid_cross = (close.shift(1) < mid.shift(1)) & (close > mid)
        score.loc[buy_mid_cross] = 65.0
        sell_mid_cross = (close.shift(1) > mid.shift(1)) & (close < mid)
        score.loc[sell_mid_cross] = 35.0

        is_signal = buy_support | sell_pressure | buy_mid_cross | sell_mid_cross
        score.loc[(~is_signal) & (close > mid) & (close < upper)] = 55.0
        score.loc[(~is_signal) & (close < mid) & (close > lower)] = 45.0
        return score

    def _get_cci_score(self, cci: pd.Series) -> pd.Series:
        """CCI 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=cci.index)
        p = self.params['base_scoring']
        threshold, ext_threshold = p['cci_threshold'], p['cci_extreme_threshold']

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

    def _get_mfi_score(self, mfi: pd.Series) -> pd.Series:
        """MFI 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=mfi.index)
        p = self.params['base_scoring']
        os, ob = p['mfi_oversold'], p['mfi_overbought']
        ext_os, ext_ob = p['mfi_extreme_oversold'], p['mfi_extreme_overbought']

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

    def _get_roc_score(self, roc: pd.Series) -> pd.Series:
        """ROC 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=roc.index)
        buy_signal = (roc.shift(1) < 0) & (roc > 0)
        score.loc[buy_signal] = 70.0
        sell_signal = (roc.shift(1) > 0) & (roc < 0)
        score.loc[sell_signal] = 30.0

        score.loc[(roc > 0) & (roc > roc.shift(1)) & (~buy_signal)] = 60.0
        score.loc[(roc < 0) & (roc < roc.shift(1)) & (~sell_signal)] = 40.0
        return score

    def _get_dmi_score(self, pdi: pd.Series, mdi: pd.Series, adx: pd.Series) -> pd.Series:
        """DMI 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=pdi.index)
        p = self.params['base_scoring']
        adx_th, adx_strong_th = p['adx_threshold'], p['adx_strong_threshold']

        buy_cross = (pdi.shift(1) < mdi.shift(1)) & (pdi > mdi)
        sell_cross = (mdi.shift(1) < pdi.shift(1)) & (mdi > pdi)

        score.loc[buy_cross & (adx > adx_th)] = 75.0
        score.loc[buy_cross & (adx > adx_strong_th)] = 85.0
        score.loc[sell_cross & (adx > adx_th)] = 25.0
        score.loc[sell_cross & (adx > adx_strong_th)] = 15.0

        is_bullish = (pdi > mdi) & (~buy_cross)
        score.loc[is_bullish & (adx > adx_strong_th)] = 65.0
        score.loc[is_bullish & (adx <= adx_strong_th) & (adx > adx_th)] = 60.0
        score.loc[is_bullish & (adx <= adx_th)] = 55.0

        is_bearish = (mdi > pdi) & (~sell_cross)
        score.loc[is_bearish & (adx > adx_strong_th)] = 35.0
        score.loc[is_bearish & (adx <= adx_strong_th) & (adx > adx_th)] = 40.0
        score.loc[is_bearish & (adx <= adx_th)] = 45.0
        return score

    def _get_sar_score(self, close: pd.Series, sar: pd.Series) -> pd.Series:
        """SAR 评分 (0-100) - 来自 MacdRsiKdjBollEnhancedStrategy"""
        score = pd.Series(50.0, index=close.index)
        buy_signal = (sar.shift(1) > close.shift(1)) & (sar < close)
        score.loc[buy_signal] = 75.0
        sell_signal = (sar.shift(1) < close.shift(1)) & (sar > close)
        score.loc[sell_signal] = 25.0

        score.loc[(close > sar) & (~buy_signal)] = 60.0
        score.loc[(close < sar) & (~sell_signal)] = 40.0
        return score

    def _adjust_score_with_volume(self, preliminary_score: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        使用量能指标调整初步的 0-100 分数。
        包括量能确认和顶背离惩罚。
        """
        vc_params = self.params['volume_confirmation']
        dd_params = self.params['divergence_detection']
        bs_params = self.params['base_scoring'] # 需要 MACD 参数用于获取列名

        if not vc_params['enabled']:
            return preliminary_score

        vol_tf = vc_params['tf']
        adjusted_score = preliminary_score.copy()

        # --- 列名准备 ---
        close_col = f'close_{vol_tf}'
        high_col = f'high_{vol_tf}'
        # low_col = f'low_{vol_tf}' # 如果需要 MFI 背离中的价格低点
        amount_col = f'amount_{vol_tf}'
        amt_ma_col = f'AMT_MA_{vc_params["amount_ma_period"]}_{vol_tf}'
        cmf_col = f'CMF_{vc_params["cmf_period"]}_{vol_tf}'
        obv_col = f'OBV_{vol_tf}'
        obv_ma_col = f'OBV_MA_{vc_params["obv_ma_period"]}_{vol_tf}'
        mfi_col = f'MFI_{bs_params["mfi_period"]}_{vol_tf}' # MFI 用于背离检查

        # 检查所需列
        required_cols = [close_col, high_col, amount_col, amt_ma_col, cmf_col, obv_col, obv_ma_col, mfi_col]
        missing_cols = [col for col in required_cols if col not in data.columns or data[col].isnull().all()]
        if missing_cols:
            logger.warning(f"量能调整缺少数据列: {missing_cols} (时间周期 {vol_tf})。跳过量能调整。")
            return preliminary_score

        # 获取数据 Series
        close = data[close_col]
        high = data[high_col]
        amount = data[amount_col]
        amount_ma = data[amt_ma_col]
        cmf = data[cmf_col]
        obv = data[obv_col]
        obv_ma = data[obv_ma_col]
        mfi = data[mfi_col] # MFI for divergence

        # 1. 量能确认/不确认调整
        boost = vc_params['boost_factor']
        penalty = vc_params['penalty_factor']

        # 买入量能确认条件 (至少两个条件满足)
        buy_volume_confirm = ((amount > amount_ma) & (cmf > 0)) | \
                            ((amount > amount_ma) & (obv > obv_ma)) | \
                            ((cmf > 0) & (obv > obv_ma))
        buy_volume_confirm = buy_volume_confirm.fillna(False)

        is_bullish_score = adjusted_score > 50
        adjusted_score.loc[is_bullish_score & buy_volume_confirm] *= boost * 1.2  # 增加正向影响
        adjusted_score.loc[is_bullish_score & (~buy_volume_confirm)] *= penalty * 0.9  # 加大惩罚力度

        # 2. 检查量能顶背离并惩罚分数 (基于价格和量能指标)
        if dd_params['enabled'] and dd_params.get('check_regular_bearish', True): # 只有在启用背离检测时才执行
            price_period = dd_params['price_period']
            cmf_threshold = dd_params['thresholds']['cmf']
            mfi_threshold = dd_params['thresholds']['mfi']
            divergence_penalty_factor = dd_params['divergence_penalty_factor']

            if len(high) >= price_period:
                is_price_high = high == high.rolling(window=price_period, min_periods=price_period).max()
            else:
                is_price_high = pd.Series(False, index=high.index)

            is_price_rising = close > close.shift(1)
            # 量能指标弱势条件
            is_cmf_weak = cmf < cmf_threshold
            is_obv_weak = obv < obv_ma
            is_mfi_weak = mfi < mfi_threshold

            # 量能顶背离: 价格新高 + 价格上涨 + (CMF弱 或 OBV弱 或 MFI弱)
            volume_bearish_divergence = is_price_high & is_price_rising & (is_cmf_weak | is_obv_weak | is_mfi_weak)
            volume_bearish_divergence = volume_bearish_divergence.fillna(False)

            # 对检测到顶背离且分数>50的情况应用惩罚
            adjusted_score.loc[volume_bearish_divergence & is_bullish_score] *= divergence_penalty_factor

        # 3. 确保分数在 0-100
        adjusted_score = adjusted_score.clip(0, 100)
        return adjusted_score

    def _calculate_base_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算多时间框架加权的基础评分 (0-100)。
        """
        bs_params = self.params['base_scoring']
        weights = bs_params['weights']
        scores = pd.DataFrame(index=data.index)
        scores['total_weighted_score'] = 0.0
        for tf in self.timeframes:
            tf_score_sum = pd.Series(0.0, index=data.index)
            indicator_count_in_tf = 0
            close_price_col = f'close_{tf}'
            if close_price_col not in data.columns:
                logger.warning(f"缺少价格列 {close_price_col}，涉及价格比较的指标 ({tf}周期) 评分可能不准确。")
                close_price = pd.Series(np.nan, index=data.index) # 设为 NaN
            else:
                close_price = data[close_price_col]
            # --- MACD ---
            if 'macd' in self.score_indicators:
                macd_col = f'MACD_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
                macdh_col = f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
                macds_col = f'MACDs_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{tf}'
                if all(c in data for c in [macd_col, macdh_col, macds_col]):
                    # 注意：_get_macd_score 使用的是 macd(柱状图), diff, dea。需确认列名
                    # 这里假设 _get_macd_score 使用 macdh (柱状图), macd (快线-慢线), macds (信号线)
                    macd_score = self._get_macd_score(data[macd_col], data[macds_col], data[macdh_col])
                    tf_score_sum += macd_score.fillna(50.0)
                    scores[f'macd_score_{tf}'] = macd_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 MACD 列 for tf={tf}")
            # --- RSI ---
            if 'rsi' in self.score_indicators:
                rsi_col = f'RSI_{bs_params["rsi_period"]}_{tf}'
                if rsi_col in data:
                    rsi_score = self._get_rsi_score(data[rsi_col])
                    tf_score_sum += rsi_score.fillna(50.0)
                    scores[f'rsi_score_{tf}'] = rsi_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 RSI 列 for tf={tf}")
            # --- KDJ ---
            if 'kdj' in self.score_indicators:
                 # 调整为匹配 IndicatorService 的 key
                 k_col = f'K_{bs_params["kdj_period_k"]}_{tf}'
                 d_col = f'D_{bs_params["kdj_period_k"]}_{tf}'
                 j_col = f'J_{bs_params["kdj_period_k"]}_{tf}'
                 # k_col = f'K_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{bs_params["kdj_period_j"]}_{tf}'
                 # d_col = f'D_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{bs_params["kdj_period_j"]}_{tf}'
                 # j_col = f'J_{bs_params["kdj_period_k"]}_{bs_params["kdj_period_d"]}_{bs_params["kdj_period_j"]}_{tf}'
                 if all(c in data for c in [k_col, d_col, j_col]):
                     kdj_score = self._get_kdj_score(data[k_col], data[d_col], data[j_col])
                     tf_score_sum += kdj_score.fillna(50.0)
                     scores[f'kdj_score_{tf}'] = kdj_score
                     indicator_count_in_tf += 1
                 # else: logger.debug(f"缺少 KDJ 列 for tf={tf}")
            # --- BOLL ---
            if 'boll' in self.score_indicators:
                # 匹配 IndicatorService 的 key
                upper_col, mid_col, lower_col = f'BB_UPPER_{tf}', f'BB_MIDDLE_{tf}', f'BB_LOWER_{tf}'
                # upper_col = f'BBU_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{tf}'
                # mid_col = f'BBM_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{tf}'
                # lower_col = f'BBL_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{tf}'
                if all(c in data for c in [upper_col, mid_col, lower_col]) and not close_price.isnull().all():
                    boll_score = self._get_boll_score(close_price, data[upper_col], data[mid_col], data[lower_col])
                    tf_score_sum += boll_score.fillna(50.0)
                    scores[f'boll_score_{tf}'] = boll_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 BOLL 列或价格列 for tf={tf}")
            # --- CCI ---
            if 'cci' in self.score_indicators:
                # 匹配 IndicatorService 的 key
                cci_col = f'CCI_{bs_params["cci_period"]}_{tf}'
                # cci_col = f'CCI_{bs_params["cci_period"]}_0.015_{tf}'
                if cci_col in data:
                    cci_score = self._get_cci_score(data[cci_col])
                    tf_score_sum += cci_score.fillna(50.0)
                    scores[f'cci_score_{tf}'] = cci_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 CCI 列 for tf={tf}")
            # --- MFI ---
            if 'mfi' in self.score_indicators:
                mfi_col = f'MFI_{bs_params["mfi_period"]}_{tf}'
                if mfi_col in data:
                    mfi_score = self._get_mfi_score(data[mfi_col])
                    tf_score_sum += mfi_score.fillna(50.0)
                    scores[f'mfi_score_{tf}'] = mfi_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 MFI 列 for tf={tf}")
            # --- ROC ---
            if 'roc' in self.score_indicators:
                roc_col = f'ROC_{bs_params["roc_period"]}_{tf}'
                if roc_col in data:
                    roc_score = self._get_roc_score(data[roc_col])
                    tf_score_sum += roc_score.fillna(50.0)
                    scores[f'roc_score_{tf}'] = roc_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 ROC 列 for tf={tf}")
            # --- DMI ---
            if 'dmi' in self.score_indicators:
                # 匹配 IndicatorService 的 key
                pdi_col, mdi_col, adx_col = f'+DI_{bs_params["dmi_period"]}_{tf}', f'-DI_{bs_params["dmi_period"]}_{tf}', f'ADX_{bs_params["dmi_period"]}_{tf}'
                # pdi_col = f'DMP_{bs_params["dmi_period"]}_{tf}'
                # mdi_col = f'DMN_{bs_params["dmi_period"]}_{tf}'
                # adx_col = f'ADX_{bs_params["dmi_period"]}_{tf}'
                if all(c in data for c in [pdi_col, mdi_col, adx_col]):
                    dmi_score = self._get_dmi_score(data[pdi_col], data[mdi_col], data[adx_col])
                    tf_score_sum += dmi_score.fillna(50.0)
                    scores[f'dmi_score_{tf}'] = dmi_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 DMI 列 for tf={tf}")
            # --- SAR ---
            if 'sar' in self.score_indicators:
                # 匹配 IndicatorService 的 key
                sar_col = f'SAR_{tf}'
                if sar_col in data and not close_price.isnull().all():
                    sar_score = self._get_sar_score(close_price, data[sar_col])
                    tf_score_sum += sar_score.fillna(50.0)
                    scores[f'sar_score_{tf}'] = sar_score
                    indicator_count_in_tf += 1
                # else: logger.debug(f"缺少 SAR 列或价格列 for tf={tf}")
            # 计算加权平均分
            if indicator_count_in_tf > 0:
                avg_tf_score = tf_score_sum / indicator_count_in_tf
                scores['total_weighted_score'] += avg_tf_score * weights[tf]
            else:
                scores['total_weighted_score'] += 50.0 * weights[tf] # 无指标则贡献中性分
        # 裁剪到 0-100
        scores['base_score_raw'] = scores['total_weighted_score'].clip(0, 100)
        return scores

    def _perform_trend_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行趋势和反转分析 (EMA排列, 长期背景, 动量, 波动率)。
        """
        analysis_df = pd.DataFrame(index=data.index)
        ta_params = self.params['trend_analysis']
        base_score_col = 'base_score_volume_adjusted' # 使用量能调整后的基础分
        if base_score_col not in data.columns or data[base_score_col].isnull().all():
             logger.warning(f"缺少或无效的 '{base_score_col}' 列，无法执行趋势分析。")
             return analysis_df # 返回空的分析结果
        score_series = data[base_score_col]
        # 1. 计算 EMA
        all_ema_periods = ta_params['ema_periods']
        for period in all_ema_periods:
            try:
                analysis_df[f'ema_score_{period}'] = ta.ema(score_series, length=period)
            except Exception as e:
                logger.error(f"计算 EMA Score {period} 时出错: {e}")
                analysis_df[f'ema_score_{period}'] = np.nan
        # 2. 计算 EMA 排列信号 (5, 13, 21, 55)
        ema_cols_align = [f'ema_score_{p}' for p in [5, 13, 21, 55]]
        if all(col in analysis_df for col in ema_cols_align):
             signal_5_13 = np.where(analysis_df['ema_score_5'] > analysis_df['ema_score_13'], 1, np.where(analysis_df['ema_score_5'] < analysis_df['ema_score_13'], -1, 0))
             signal_13_21 = np.where(analysis_df['ema_score_13'] > analysis_df['ema_score_21'], 1, np.where(analysis_df['ema_score_13'] < analysis_df['ema_score_21'], -1, 0))
             signal_21_55 = np.where(analysis_df['ema_score_21'] > analysis_df['ema_score_55'], 1, np.where(analysis_df['ema_score_21'] < analysis_df['ema_score_55'], -1, 0))
             analysis_df['alignment_signal'] = signal_5_13 + signal_13_21 + signal_21_55
             # 处理 NaN
             analysis_df.loc[analysis_df[ema_cols_align].isna().any(axis=1), 'alignment_signal'] = np.nan
        else:
             analysis_df['alignment_signal'] = np.nan
        # 3. 计算排列反转信号
        analysis_df['alignment_reversal'] = 0
        if 'alignment_signal' in analysis_df and analysis_df['alignment_signal'].notna().any():
             prev_alignment = analysis_df['alignment_signal'].shift(1)
             current_alignment = analysis_df['alignment_signal']
             top_reversal = ((prev_alignment >= 1) & (current_alignment <= 0))
             bottom_reversal = ((prev_alignment <= -1) & (current_alignment >= 0))
             analysis_df.loc[top_reversal, 'alignment_reversal'] = -1
             analysis_df.loc[bottom_reversal, 'alignment_reversal'] = 1
             analysis_df.loc[prev_alignment.isna() | current_alignment.isna(), 'alignment_reversal'] = 0

        # 4. 计算 EMA 强度 (13-55)
        if 'ema_score_13' in analysis_df and 'ema_score_55' in analysis_df:
             analysis_df['ema_strength_13_55'] = analysis_df['ema_score_13'] - analysis_df['ema_score_55']
        else:
             analysis_df['ema_strength_13_55'] = np.nan
        # 5. 计算得分动量
        analysis_df['score_momentum'] = score_series.diff()
        # 6. 计算得分波动率
        volatility_window = ta_params['volatility_window']
        analysis_df['score_volatility'] = score_series.rolling(window=volatility_window).std()
        # 7. 长期趋势背景
        long_term_ema_col = f'ema_score_{ta_params["long_term_ema_period"]}'
        if long_term_ema_col in analysis_df:
             analysis_df['long_term_context'] = np.where(
                 score_series > analysis_df[long_term_ema_col], 1,
                 np.where(score_series < analysis_df[long_term_ema_col], -1, 0)
             )
             analysis_df.loc[score_series.isna() | analysis_df[long_term_ema_col].isna(), 'long_term_context'] = np.nan
        else:
             analysis_df['long_term_context'] = np.nan
        return analysis_df
    
    def _perform_signal_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        执行背离、K线、指标反转等信号分析。
        """
        analysis_df = pd.DataFrame(index=data.index)
        dd_params = self.params['divergence_detection']
        kpd_params = self.params['kline_pattern_detection']
        bs_params = self.params['base_scoring']
        ia_params = self.params['indicator_analysis_params']
        fsg_params = self.params['final_signal_generation']
        # 确定用于分析的时间框架 (优先使用背离检测的 tf，否则用 K 线，最后用量能)
        analysis_tf = dd_params.get('tf') if dd_params['enabled'] else \
                      kpd_params.get('tf') if kpd_params['enabled'] else \
                      self.params['volume_confirmation'].get('tf', '30')
        logger.info(f"信号分析使用时间框架: {analysis_tf}")
        # --- 1. 背离检测 ---
        if dd_params['enabled']:
            price_col = f'close_{analysis_tf}'
            div_lookback = dd_params['lookback']
            fp_params = get_find_peaks_params(analysis_tf, div_lookback) # 获取 find_peaks 参数
            if price_col not in data.columns or data[price_col].isnull().all():
                logger.warning(f"缺少价格列 {price_col} 或全为 NaN，无法进行背离检测。")
            else:
                price_series = data[price_col]
                for indicator_key, enabled in dd_params['indicators'].items():
                    if not enabled: continue
                    div_signal_col = f'{indicator_key}_divergence'
                    indicator_col = None
                    if indicator_key == 'macd_hist':
                         indicator_col = f'MACDh_{bs_params["macd_fast"]}_{bs_params["macd_slow"]}_{bs_params["macd_signal"]}_{analysis_tf}'
                    elif indicator_key == 'rsi':
                         indicator_col = f'RSI_{bs_params["rsi_period"]}_{analysis_tf}'
                    elif indicator_key == 'mfi':
                         indicator_col = f'MFI_{bs_params["mfi_period"]}_{analysis_tf}'
                    elif indicator_key == 'obv':
                         indicator_col = f'OBV_{analysis_tf}'

                    if indicator_col and indicator_col in data and data[indicator_col].notna().any():
                        analysis_df[div_signal_col] = detect_divergence(
                            price_series, data[indicator_col],
                            lookback=div_lookback,
                            find_peaks_params=fp_params,
                            check_regular_bullish=dd_params['check_regular_bullish'],
                            check_regular_bearish=dd_params['check_regular_bearish'],
                            check_hidden_bullish=dd_params['check_hidden_bullish'],
                            check_hidden_bearish=dd_params['check_hidden_bearish']
                        )
                    else:
                        logger.warning(f"缺少指标列 {indicator_col} 或全为 NaN，无法计算 {indicator_key} 背离。")
                        analysis_df[div_signal_col] = 0
        else:
            # 如果禁用，则添加全 0 列以保持一致性
            for indicator_key in dd_params['indicators'].keys():
                analysis_df[f'{indicator_key}_divergence'] = 0
        # --- 2. K 线形态检测 ---
        if kpd_params['enabled']:
            kline_tf = kpd_params['tf']
            ohlc_cols = [f'open_{kline_tf}', f'high_{kline_tf}', f'low_{kline_tf}', f'close_{kline_tf}']
            if all(col in data for col in ohlc_cols):
                 # 重命名列以便 detect_kline_patterns 函数使用
                 temp_df = data[ohlc_cols].rename(columns={
                     f'open_{kline_tf}': 'open', f'high_{kline_tf}': 'high',
                     f'low_{kline_tf}': 'low', f'close_{kline_tf}': 'close'
                 })
                 analysis_df['kline_pattern'] = detect_kline_patterns(temp_df)
            else:
                 logger.warning(f"缺少 OHLC 列 for tf={kline_tf}，无法进行 K 线形态检测。")
                 analysis_df['kline_pattern'] = 0
        else:
            analysis_df['kline_pattern'] = 0
        # --- 3. 技术指标反转信号 (OB/OS, BB) ---
        analysis_df['rsi_obos_reversal'] = 0
        analysis_df['stoch_obos_reversal'] = 0
        analysis_df['cci_obos_reversal'] = 0
        analysis_df['bb_reversal'] = 0
        analysis_df['volume_spike'] = 0
        # 需要价格数据
        price_col = f'close_{analysis_tf}'
        if price_col not in data or data[price_col].isnull().all():
            logger.warning(f"缺少价格列 {price_col} 或全为 NaN，无法计算指标反转信号。")
        else:
            close_series = data[price_col]
            # RSI OB/OS Reversal
            rsi_col = f'RSI_{bs_params["rsi_period"]}_{analysis_tf}'
            if rsi_col in data and data[rsi_col].notna().any():
                rsi = data[rsi_col]
                ob, os = bs_params['rsi_overbought'], bs_params['rsi_oversold']
                ext_ob, ext_os = bs_params['rsi_extreme_overbought'], bs_params['rsi_extreme_oversold']
                sell_cond = ((rsi.shift(1) > ob) & (rsi <= ob)) | ((rsi.shift(1) > ext_ob) & (rsi <= ext_ob))
                buy_cond = ((rsi.shift(1) < os) & (rsi >= os)) | ((rsi.shift(1) < ext_os) & (rsi >= ext_os))
                analysis_df.loc[sell_cond, 'rsi_obos_reversal'] = -1
                analysis_df.loc[buy_cond, 'rsi_obos_reversal'] = 1
            # Stoch OB/OS Reversal
            # stoch_k_col = f'STOCHk_{ia_params["stoch_k"]}_{ia_params["stoch_d"]}_{ia_params["stoch_smooth_k"]}_{analysis_tf}'
            stoch_k_col = f'STOCH_K_{ia_params["stoch_k"]}_{analysis_tf}' # 匹配 IndicatorService
            if stoch_k_col in data and data[stoch_k_col].notna().any():
                stoch_k = data[stoch_k_col]
                ob, os = ia_params['stoch_ob'], ia_params['stoch_os']
                sell_cond = (stoch_k.shift(1) > ob) & (stoch_k <= ob)
                buy_cond = (stoch_k.shift(1) < os) & (stoch_k >= os)
                analysis_df.loc[sell_cond, 'stoch_obos_reversal'] = -1
                analysis_df.loc[buy_cond, 'stoch_obos_reversal'] = 1
            # CCI OB/OS Reversal
            # cci_col = f'CCI_{bs_params["cci_period"]}_0.015_{analysis_tf}'
            cci_col = f'CCI_{bs_params["cci_period"]}_{analysis_tf}' # 匹配 IndicatorService
            if cci_col in data and data[cci_col].notna().any():
                cci = data[cci_col]
                ob, os = bs_params['cci_threshold'], -bs_params['cci_threshold']
                ext_ob, ext_os = bs_params['cci_extreme_threshold'], -bs_params['cci_extreme_threshold']
                sell_cond = ((cci.shift(1) > ob) & (cci <= ob)) | ((cci.shift(1) > ext_ob) & (cci <= ext_ob))
                buy_cond = ((cci.shift(1) < os) & (cci >= os)) | ((cci.shift(1) < ext_os) & (cci >= ext_os))
                analysis_df.loc[sell_cond, 'cci_obos_reversal'] = -1
                analysis_df.loc[buy_cond, 'cci_obos_reversal'] = 1
            # Bollinger Band Reversal
            # bb_upper_col = f'BBU_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{analysis_tf}'
            # bb_lower_col = f'BBL_{bs_params["boll_period"]}_{bs_params["boll_std_dev"]}_{analysis_tf}'
            bb_upper_col = f'BB_UPPER_{analysis_tf}' # 匹配 IndicatorService
            bb_lower_col = f'BB_LOWER_{analysis_tf}' # 匹配 IndicatorService
            if bb_upper_col in data and bb_lower_col in data and data[bb_upper_col].notna().any() and data[bb_lower_col].notna().any():
                 upper = data[bb_upper_col]
                 lower = data[bb_lower_col]
                 sell_cond = (close_series.shift(1) > upper.shift(1)) & (close_series <= upper)
                 buy_cond = (close_series.shift(1) < lower.shift(1)) & (close_series >= lower)
                 analysis_df.loc[sell_cond, 'bb_reversal'] = -1
                 analysis_df.loc[buy_cond, 'bb_reversal'] = 1
            # Volume Spike
            volume_col = f'volume_{analysis_tf}'
            # vol_ma_col = f'volume_ma_{analysis_tf}' # 假设已存在
            vol_ma_col = f'VOL_MA_{ia_params["volume_ma_period"]}_{analysis_tf}' # 匹配 IndicatorService
            if volume_col in data and vol_ma_col in data and data[volume_col].notna().any() and data[vol_ma_col].notna().any():
                 vol = data[volume_col]
                 vol_ma = data[vol_ma_col]
                 vol_spike_factor = ia_params['volume_spike_factor']
                 analysis_df.loc[vol > vol_ma * vol_spike_factor, 'volume_spike'] = 1
        # --- 4. 计算加权确认信号 ---
        analysis_df['confirmation_signal'] = 0.0
        signal_weights = fsg_params['signal_weights']
        # 辅助函数：安全地获取信号值
        def get_signal_value(col_name):
            return analysis_df[col_name] if col_name in analysis_df else pd.Series(0.0, index=analysis_df.index)
        # EMA 排列反转
        analysis_df['confirmation_signal'] += get_signal_value('alignment_reversal') * signal_weights.get('alignment_reversal', 0)
        # 背离信号 (区分常规和隐藏)
        for indicator_key in dd_params['indicators'].keys():
            if dd_params['indicators'].get(indicator_key):
                 div_col = f'{indicator_key}_divergence'
                 div_series = get_signal_value(div_col)
                 # 常规牛 (+1)
                 analysis_df['confirmation_signal'] += (div_series == 1) * signal_weights.get(f'{indicator_key}_regular_div', 0)
                 # 常规熊 (-1)
                 analysis_df['confirmation_signal'] -= (div_series == -1) * signal_weights.get(f'{indicator_key}_regular_div', 0)
                 # 隐藏牛 (+2 -> +1 * weight)
                 analysis_df['confirmation_signal'] += (div_series == 2) * signal_weights.get(f'{indicator_key}_hidden_div', 0)
                 # 隐藏熊 (-2 -> -1 * weight)
                 analysis_df['confirmation_signal'] -= (div_series == -2) * signal_weights.get(f'{indicator_key}_hidden_div', 0)
        # 指标 OB/OS 反转
        analysis_df['confirmation_signal'] += get_signal_value('rsi_obos_reversal') * signal_weights.get('rsi_obos_reversal', 0)
        analysis_df['confirmation_signal'] += get_signal_value('stoch_obos_reversal') * signal_weights.get('stoch_obos_reversal', 0)
        analysis_df['confirmation_signal'] += get_signal_value('cci_obos_reversal') * signal_weights.get('cci_obos_reversal', 0)
        analysis_df['confirmation_signal'] += get_signal_value('bb_reversal') * signal_weights.get('bb_reversal', 0)
        # K 线形态信号 (区分不同形态)
        kline_series = get_signal_value('kline_pattern')
        analysis_df['confirmation_signal'] += (kline_series == 1) * signal_weights.get('kline_engulfing', 0) # Bull Engulf
        analysis_df['confirmation_signal'] -= (kline_series == -1) * signal_weights.get('kline_engulfing', 0) # Bear Engulf
        analysis_df['confirmation_signal'] += (kline_series == 2) * signal_weights.get('kline_hammer_hanging', 0) # Hammer
        analysis_df['confirmation_signal'] -= (kline_series == -2) * signal_weights.get('kline_hammer_hanging', 0) # Hanging Man
        analysis_df['confirmation_signal'] += (kline_series == 3) * signal_weights.get('kline_star', 0) # Morning Star
        analysis_df['confirmation_signal'] -= (kline_series == -3) * signal_weights.get('kline_star', 0) # Evening Star
        analysis_df['confirmation_signal'] += (kline_series == 10) * signal_weights.get('kline_marubozu', 0) # Bull Marubozu
        analysis_df['confirmation_signal'] -= (kline_series == -10) * signal_weights.get('kline_marubozu', 0) # Bear Marubozu
        # Doji (5) - 权重较低，可能表示犹豫
        analysis_df['confirmation_signal'] += (kline_series == 5) * signal_weights.get('kline_doji', 0) # Doji - 通常是中性或反转前兆，给较低权重
        # 放量确认 (仅当存在其他买入信号时加强)
        volume_spike_series = get_signal_value('volume_spike')
        # 仅在 confirmation_signal > 0 且放量时增加权重
        analysis_df.loc[(analysis_df['confirmation_signal'] > 0) & (volume_spike_series == 1), 'confirmation_signal'] += signal_weights.get('volume_spike_confirm', 0)
        # 触发强确认信号
        threshold = fsg_params['confirmation_weighted_threshold']
        analysis_df['strong_confirmation'] = 0
        analysis_df.loc[analysis_df['confirmation_signal'] >= threshold, 'strong_confirmation'] = 1
        analysis_df.loc[analysis_df['confirmation_signal'] <= -threshold, 'strong_confirmation'] = -1

        return analysis_df

    def _generate_t0_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成 T+0 交易建议信号"""
        signals = pd.DataFrame(index=data.index)
        signals['t0_signal'] = 0 # 0: No signal, 1: Buy, -1: Sell
        t0_params = self.params['t_plus_0_signals']

        if not t0_params['enabled']:
            return signals
        # 确定分析时间框 (与指标反转信号一致)
        analysis_tf = self.params['divergence_detection'].get('tf') or \
                      self.params['kline_pattern_detection'].get('tf') or \
                      self.params['volume_confirmation'].get('tf', '15')

        vwap_col = f'vwap_{analysis_tf}' # 或者 'vwap'
        close_col = f'close_{analysis_tf}'
        long_term_context_col = 'long_term_context' # 来自趋势分析
        if vwap_col not in data or close_col not in data:
             logger.warning(f"缺少 VWAP ({vwap_col}) 或 Close ({close_col}) 列，无法生成 T+0 信号。")
             return signals
        if t0_params['use_long_term_filter'] and long_term_context_col not in data:
             logger.warning(f"缺少 {long_term_context_col} 列，无法应用 T+0 长期趋势过滤。")
             # 可以选择禁用过滤或返回
             # t0_params['use_long_term_filter'] = False
        vwap = data[vwap_col]
        close = data[close_col]
        if vwap.isnull().all() or close.isnull().all():
            logger.warning(f"VWAP ({vwap_col}) 或 Close ({close_col}) 列全为 NaN，无法生成 T+0 信号。")
            return signals
        # 计算价格与 VWAP 的偏离度
        deviation = (close - vwap) / vwap
        deviation = deviation.fillna(0) # 处理可能的 NaN
        # 买入条件：价格低于 VWAP 达到阈值
        buy_condition = deviation <= t0_params['buy_dev_threshold']
        # 卖出条件：价格高于 VWAP 达到阈值
        sell_condition = deviation >= t0_params['sell_dev_threshold']
        # 应用长期趋势过滤
        if t0_params['use_long_term_filter'] and long_term_context_col in data:
             long_term_context = data[long_term_context_col].fillna(0) # 填充 NaN 为中性
             # 只有在长期趋势向上 (>=0) 时才考虑买入信号
             buy_condition &= (long_term_context >= 0)
             # 只有在长期趋势向下 (<=0) 时才考虑卖出信号
             sell_condition &= (long_term_context <= 0)
        signals.loc[buy_condition, 't0_signal'] = 1
        signals.loc[sell_condition, 't0_signal'] = -1
        return signals

    # --- 主信号生成方法 ---
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        核心方法：根据输入的包含多时间周期指标的 DataFrame 生成最终的综合评分或信号。

        :param data: 输入的 DataFrame，应包含 get_required_columns() 返回的所有列。
                     通常由 IndicatorService.prepare_strategy_dataframe 提供。
        :return: 包含最终评分或信号的 Pandas Series。
                 'final_score': 0-100 的综合评分
                 可能包含其他列如 'confirmation_signal', 'strong_confirmation', 't0_signal'
        """
        logger.info(f"开始执行策略: {self.strategy_name}")
        if data is None or data.empty:
            logger.warning("输入数据为空，无法生成信号。")
            return pd.Series(dtype=float) # 返回空 Series

        # --- 检查必需列 ---
        required_cols = self.get_required_columns()
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"输入数据缺少必需列: {missing_cols}。策略无法运行。")
            # 在实际部署中，可能需要更详细的日志记录或异常处理
            # 检查是否是关键指标缺失导致无法计算基础分
            core_missing = any(tf in col for tf in self.timeframes for col in missing_cols)
            if core_missing:
                logger.error("缺少核心时间框架指标，无法计算基础分。")
                return pd.Series(50.0, index=data.index) # 返回中性分
            else:
                logger.warning("缺少部分辅助列，部分信号可能受影响，但尝试继续。")


        # --- 步骤 1 & 2: 计算基础评分并应用量能调整 ---
        base_scores_df = self._calculate_base_score(data)
        base_score_adjusted = self._adjust_score_with_volume(base_scores_df['base_score_raw'], data)
        base_scores_df['base_score_volume_adjusted'] = base_score_adjusted

        # --- 步骤 3: 执行趋势分析 ---
        trend_analysis_df = self._perform_trend_analysis(base_scores_df) # 使用量能调整后的分数进行趋势分析

        # --- 步骤 4: 执行信号分析 (背离, K线, 指标反转) ---
        # 合并基础数据和趋势分析结果，传递给信号分析
        combined_data_for_signals = pd.concat([data, base_scores_df, trend_analysis_df], axis=1)
        signal_analysis_df = self._perform_signal_analysis(combined_data_for_signals)

        # --- 步骤 5: 生成 T+0 信号 ---
        # 合并基础数据和趋势分析结果，传递给 T+0 分析
        combined_data_for_t0 = pd.concat([data, trend_analysis_df], axis=1) # T+0 可能需要长期趋势
        t0_signals_df = self._generate_t0_signals(combined_data_for_t0)

        # --- 步骤 6: 组合最终结果 ---
        # 可以定义不同的输出模式：
        # 模式A: 返回一个综合分数 (例如: 基础分 + 确认信号强度)
        # 模式B: 返回包含所有中间结果和最终信号的 DataFrame

        # 模式A: 计算最终综合分数
        final_score = base_scores_df['base_score_volume_adjusted'].copy()
        # 可以根据 confirmation_signal 对基础分进行微调
        # 例如：强确认买入信号，略微提高分数；强确认卖出信号，略微降低分数
        # 这里的调整逻辑需要根据实际回测效果确定，下面是一个示例：
        confirmation_strength = signal_analysis_df.get('confirmation_signal', pd.Series(0.0, index=data.index)).fillna(0)
        # 优化缩放逻辑：分段缩放确认信号强度
        def scale_confirmation(strength):
            if strength > 5:
                return 10 * (1 - 1 / (1 + strength - 5))  # 非线性缩放，限制正向影响
            elif strength < -5:
                return -10 * (1 - 1 / (1 + abs(strength) - 5))  # 非线性缩放，限制负向影响
            else:
                return strength * 2  # 线性缩放，保持敏感性
        scaled_confirmation = confirmation_strength.apply(scale_confirmation)
        # 针对A股市场 T+1 制度，对买入信号（正向确认）加轻微惩罚
        scaled_confirmation = scaled_confirmation.apply(lambda x: x * 0.95 if x > 0 else x)

        final_score += scaled_confirmation
        final_score = final_score.clip(0, 100).round(2) # 限制在 0-100

        # 将所有中间和最终结果合并到一个 DataFrame 中
        self.intermediate_data = pd.concat([
            base_scores_df,
            trend_analysis_df,
            signal_analysis_df,
            t0_signals_df,
            pd.DataFrame({'final_score': final_score}, index=data.index)
        ], axis=1)

        logger.info(f"{self.strategy_name}: 信号生成完毕。")
        # logger.debug(f"Sample final scores:\n{final_score.tail()}")

        # --- 步骤 7: 对信号进行分析 ---
        self.analyze_signals()

        # 返回最终评分 Series
        return final_score

    def get_intermediate_data(self) -> Optional[pd.DataFrame]:
        """返回包含所有中间计算结果的 DataFrame，用于分析和调试。"""
        return self.intermediate_data

    # 分析信号
    def analyze_signals(self) -> pd.DataFrame:
        """
        对生成的信号和中间数据进行详细统计分析，返回分析结果。
        分析内容包括信号分布、强确认信号比例、背离信号统计、K线形态分布、趋势分析、指标反转信号及趋势反转信号等。
        同时对信号进行详细判断，生成结构化数据供数据库存储。
        中文解读以详细形式直接输出到日志，不存储在结果中，先输出最重要观点，再输出详细说明。
        :return: 包含分析结果的 DataFrame（仅结构化数据）
        """
        if self.intermediate_data is None or self.intermediate_data.empty:
            logger.warning("中间数据为空，无法进行信号分析。")
            return pd.DataFrame()

        analysis_results = {}
        data = self.intermediate_data

        # 1. 最终信号分布统计
        if 'final_score' in data:
            final_score = data['final_score'].dropna()
            if not final_score.empty:
                analysis_results['final_score_mean'] = final_score.mean()
                analysis_results['final_score_std'] = final_score.std()
                analysis_results['final_score_min'] = final_score.min()
                analysis_results['final_score_max'] = final_score.max()
                analysis_results['final_score_bullish_ratio'] = (final_score > 50).mean()
                analysis_results['final_score_strong_bullish_ratio'] = (final_score >= 70).mean()
                analysis_results['final_score_weak_bullish_ratio'] = ((final_score >= 55) & (final_score < 70)).mean()
                analysis_results['final_score_bearish_ratio'] = (final_score < 50).mean()
                analysis_results['final_score_strong_bearish_ratio'] = (final_score <= 30).mean()
                analysis_results['final_score_weak_bearish_ratio'] = ((final_score <= 45) & (final_score > 30)).mean()
                analysis_results['final_score_neutral_ratio'] = (final_score == 50).mean()

        # 2. 强确认信号比例
        if 'strong_confirmation' in data:
            strong_conf = data['strong_confirmation'].dropna()
            if not strong_conf.empty:
                analysis_results['strong_confirmation_bullish_ratio'] = (strong_conf == 1).mean()
                analysis_results['strong_confirmation_bearish_ratio'] = (strong_conf == -1).mean()
                analysis_results['strong_confirmation_neutral_ratio'] = (strong_conf == 0).mean()
                analysis_results['strong_confirmation_bullish_count'] = (strong_conf == 1).sum()
                analysis_results['strong_confirmation_bearish_count'] = (strong_conf == -1).sum()

        # 3. 背离信号统计
        dd_params = self.params['divergence_detection']
        if dd_params['enabled']:
            for indicator_key in dd_params['indicators'].keys():
                if dd_params['indicators'].get(indicator_key):
                    div_col = f'{indicator_key}_divergence'
                    if div_col in data:
                        div_series = data[div_col].dropna()
                        if not div_series.empty:
                            analysis_results[f'{indicator_key}_regular_bullish_count'] = (div_series == 1).sum()
                            analysis_results[f'{indicator_key}_regular_bearish_count'] = (div_series == -1).sum()
                            analysis_results[f'{indicator_key}_hidden_bullish_count'] = (div_series == 2).sum()
                            analysis_results[f'{indicator_key}_hidden_bearish_count'] = (div_series == -2).sum()
                            analysis_results[f'{indicator_key}_total_divergence_count'] = (
                                (div_series == 1) | (div_series == -1) | (div_series == 2) | (div_series == -2)
                            ).sum()

        # 4. K线形态分布
        if 'kline_pattern' in data:
            kline_series = data['kline_pattern'].dropna()
            if not kline_series.empty:
                analysis_results['kline_bullish_engulfing_count'] = (kline_series == 1).sum()
                analysis_results['kline_bearish_engulfing_count'] = (kline_series == -1).sum()
                analysis_results['kline_hammer_count'] = (kline_series == 2).sum()
                analysis_results['kline_hanging_man_count'] = (kline_series == -2).sum()
                analysis_results['kline_morning_star_count'] = (kline_series == 3).sum()
                analysis_results['kline_evening_star_count'] = (kline_series == -3).sum()
                analysis_results['kline_doji_count'] = (kline_series == 5).sum()
                analysis_results['kline_bullish_marubozu_count'] = (kline_series == 10).sum()
                analysis_results['kline_bearish_marubozu_count'] = (kline_series == -10).sum()
                analysis_results['kline_total_bullish_count'] = (
                    (kline_series == 1) | (kline_series == 2) | (kline_series == 3) | (kline_series == 10)
                ).sum()
                analysis_results['kline_total_bearish_count'] = (
                    (kline_series == -1) | (kline_series == -2) | (kline_series == -3) | (kline_series == -10)
                ).sum()

        # 5. T+0 信号分布
        if 't0_signal' in data:
            t0_series = data['t0_signal'].dropna()
            if not t0_series.empty:
                analysis_results['t0_buy_signal_count'] = (t0_series == 1).sum()
                analysis_results['t0_sell_signal_count'] = (t0_series == -1).sum()
                analysis_results['t0_no_signal_ratio'] = (t0_series == 0).mean()

        # 6. 趋势分析统计
        if 'alignment_signal' in data:
            alignment_series = data['alignment_signal'].dropna()
            if not alignment_series.empty:
                analysis_results['alignment_bullish_ratio'] = (alignment_series > 0).mean()
                analysis_results['alignment_bearish_ratio'] = (alignment_series < 0).mean()
                analysis_results['alignment_neutral_ratio'] = (alignment_series == 0).mean()
                analysis_results['alignment_strong_bullish_count'] = (alignment_series == 3).sum()
                analysis_results['alignment_strong_bearish_count'] = (alignment_series == -3).sum()

        if 'long_term_context' in data:
            long_term_series = data['long_term_context'].dropna()
            if not long_term_series.empty:
                analysis_results['long_term_bullish_ratio'] = (long_term_series == 1).mean()
                analysis_results['long_term_bearish_ratio'] = (long_term_series == -1).mean()
                analysis_results['long_term_neutral_ratio'] = (long_term_series == 0).mean()

        if 'score_momentum' in data:
            momentum_series = data['score_momentum'].dropna()
            if not momentum_series.empty:
                analysis_results['momentum_positive_ratio'] = (momentum_series > 0).mean()
                analysis_results['momentum_negative_ratio'] = (momentum_series < 0).mean()
                analysis_results['momentum_mean'] = momentum_series.mean()

        if 'score_volatility' in data:
            volatility_series = data['score_volatility'].dropna()
            if not volatility_series.empty:
                analysis_results['volatility_mean'] = volatility_series.mean()
                analysis_results['volatility_std'] = volatility_series.std()

        # 7. 趋势反转信号统计
        if 'alignment_reversal' in data:
            reversal_series = data['alignment_reversal'].dropna()
            if not reversal_series.empty:
                analysis_results['reversal_bullish_count'] = (reversal_series == 1).sum()
                analysis_results['reversal_bearish_count'] = (reversal_series == -1).sum()
                analysis_results['reversal_bullish_ratio'] = (reversal_series == 1).mean()
                analysis_results['reversal_bearish_ratio'] = (reversal_series == -1).mean()
                analysis_results['reversal_total_count'] = ((reversal_series == 1) | (reversal_series == -1)).sum()

        # 8. 指标反转信号统计
        if 'rsi_obos_reversal' in data:
            rsi_reversal = data['rsi_obos_reversal'].dropna()
            if not rsi_reversal.empty:
                analysis_results['rsi_bullish_reversal_count'] = (rsi_reversal == 1).sum()
                analysis_results['rsi_bearish_reversal_count'] = (rsi_reversal == -1).sum()

        if 'stoch_obos_reversal' in data:
            stoch_reversal = data['stoch_obos_reversal'].dropna()
            if not stoch_reversal.empty:
                analysis_results['stoch_bullish_reversal_count'] = (stoch_reversal == 1).sum()
                analysis_results['stoch_bearish_reversal_count'] = (stoch_reversal == -1).sum()

        if 'cci_obos_reversal' in data:
            cci_reversal = data['cci_obos_reversal'].dropna()
            if not cci_reversal.empty:
                analysis_results['cci_bullish_reversal_count'] = (cci_reversal == 1).sum()
                analysis_results['cci_bearish_reversal_count'] = (cci_reversal == -1).sum()

        if 'bb_reversal' in data:
            bb_reversal = data['bb_reversal'].dropna()
            if not bb_reversal.empty:
                analysis_results['bb_bullish_reversal_count'] = (bb_reversal == 1).sum()
                analysis_results['bb_bearish_reversal_count'] = (bb_reversal == -1).sum()

        if 'volume_spike' in data:
            vol_spike = data['volume_spike'].dropna()
            if not vol_spike.empty:
                analysis_results['volume_spike_count'] = (vol_spike == 1).sum()
                analysis_results['volume_spike_ratio'] = (vol_spike == 1).mean()

        # 9. 信号判断逻辑
        signal_judgment = {}
        latest_data = data.iloc[-1] if not data.empty else None
        if latest_data is not None:
            # 9.1 最终评分判断
            if 'final_score' in latest_data:
                final_score_latest = latest_data['final_score']
                if final_score_latest >= 70:
                    signal_judgment['score_strength'] = "强买入信号"
                    signal_judgment['score_suggestion'] = "建议买入或加仓，市场可能有较强的上涨动能"
                    signal_judgment['score_confidence'] = "高"
                elif final_score_latest >= 55:
                    signal_judgment['score_strength'] = "弱买入信号"
                    signal_judgment['score_suggestion'] = "可考虑小仓位买入，市场可能有温和上涨趋势"
                    signal_judgment['score_confidence'] = "中等"
                elif final_score_latest <= 30:
                    signal_judgment['score_strength'] = "强卖出信号"
                    signal_judgment['score_suggestion'] = "建议卖出或减仓，市场可能有较强的下跌风险"
                    signal_judgment['score_confidence'] = "高"
                elif final_score_latest <= 45:
                    signal_judgment['score_strength'] = "弱卖出信号"
                    signal_judgment['score_suggestion'] = "可考虑小仓位卖出，市场可能有温和下跌趋势"
                    signal_judgment['score_confidence'] = "中等"
                else:
                    signal_judgment['score_strength'] = "中性信号"
                    signal_judgment['score_suggestion'] = "观望为主，市场方向不明，建议等待更明确信号"
                    signal_judgment['score_confidence'] = "低"

            # 9.2 强确认信号判断
            if 'strong_confirmation' in latest_data:
                strong_conf_latest = latest_data['strong_confirmation']
                if strong_conf_latest == 1:
                    signal_judgment['confirmation_strength'] = "强确认买入"
                    signal_judgment['confirmation_suggestion'] = "信号可靠性较高，建议积极操作，市场可能即将启动上涨"
                elif strong_conf_latest == -1:
                    signal_judgment['confirmation_strength'] = "强确认卖出"
                    signal_judgment['confirmation_suggestion'] = "信号可靠性较高，建议积极操作，市场可能即将下跌"
                else:
                    signal_judgment['confirmation_strength'] = "无强确认"
                    signal_judgment['confirmation_suggestion'] = "信号可靠性一般，需结合其他指标或形态确认"

            # 9.3 长期趋势背景判断
            if 'long_term_context' in latest_data:
                long_term_context = latest_data['long_term_context']
                if long_term_context == 1:
                    signal_judgment['trend_context'] = "长期看涨"
                    signal_judgment['trend_suggestion'] = "适合做多，买入信号更可靠，市场整体处于上升趋势"
                elif long_term_context == -1:
                    signal_judgment['trend_context'] = "长期看跌"
                    signal_judgment['trend_suggestion'] = "适合做空，卖出信号更可靠，市场整体处于下降趋势"
                else:
                    signal_judgment['trend_context'] = "长期趋势不明"
                    signal_judgment['trend_suggestion'] = "趋势不明，操作需谨慎，建议关注趋势变化"

            # 9.4 EMA排列信号判断
            if 'alignment_signal' in latest_data:
                alignment_latest = latest_data['alignment_signal']
                if alignment_latest == 3:
                    signal_judgment['alignment_status'] = "完全多头排列"
                    signal_judgment['alignment_suggestion'] = "短期、中期、长期均看涨，上涨趋势非常明确"
                elif alignment_latest > 0:
                    signal_judgment['alignment_status'] = "部分多头排列"
                    signal_judgment['alignment_suggestion'] = "部分时间框架看涨，上涨趋势正在形成"
                elif alignment_latest == -3:
                    signal_judgment['alignment_status'] = "完全空头排列"
                    signal_judgment['alignment_suggestion'] = "短期、中期、长期均看跌，下跌趋势非常明确"
                elif alignment_latest < 0:
                    signal_judgment['alignment_status'] = "部分空头排列"
                    signal_judgment['alignment_suggestion'] = "部分时间框架看跌，下跌趋势正在形成"
                else:
                    signal_judgment['alignment_status'] = "排列混乱"
                    signal_judgment['alignment_suggestion'] = "多空方向不一致，趋势不明，建议观望"

            # 9.5 趋势反转信号判断
            if 'alignment_reversal' in latest_data:
                reversal_latest = latest_data['alignment_reversal']
                if reversal_latest == 1:
                    signal_judgment['reversal_status'] = "底部反转信号"
                    signal_judgment['reversal_suggestion'] = "趋势可能从下跌转为上涨，建议关注买入机会，市场可能即将反转"
                elif reversal_latest == -1:
                    signal_judgment['reversal_status'] = "顶部反转信号"
                    signal_judgment['reversal_suggestion'] = "趋势可能从上涨转为下跌，建议关注卖出机会，市场可能即将反转"
                else:
                    signal_judgment['reversal_status'] = "无趋势反转信号"
                    signal_judgment['reversal_suggestion'] = "当前无明显趋势反转迹象，市场可能继续原有趋势或震荡"

            # 9.6 动量和波动率判断
            if 'score_momentum' in latest_data:
                momentum_latest = latest_data['score_momentum']
                if momentum_latest > 0:
                    signal_judgment['momentum_status'] = "得分动量向上"
                    signal_judgment['momentum_suggestion'] = "市场情绪可能正在转好，买入信号可能更有效"
                elif momentum_latest < 0:
                    signal_judgment['momentum_status'] = "得分动量向下"
                    signal_judgment['momentum_suggestion'] = "市场情绪可能正在转差，卖出信号可能更有效"
                else:
                    signal_judgment['momentum_status'] = "得分动量平稳"
                    signal_judgment['momentum_suggestion'] = "市场情绪稳定，信号方向性不强"

            if 'score_volatility' in latest_data:
                volatility_latest = latest_data['score_volatility']
                volatility_mean = analysis_results.get('volatility_mean', 0)
                if volatility_latest > volatility_mean * 1.2:
                    signal_judgment['volatility_status'] = "高波动率"
                    signal_judgment['volatility_suggestion'] = "市场波动较大，操作风险较高，建议缩小仓位"
                elif volatility_latest < volatility_mean * 0.8:
                    signal_judgment['volatility_status'] = "低波动率"
                    signal_judgment['volatility_suggestion'] = "市场波动较小，趋势可能不明显，信号可靠性较低"
                else:
                    signal_judgment['volatility_status'] = "正常波动率"
                    signal_judgment['volatility_suggestion'] = "市场波动正常，信号可按常规操作"

            # 9.7 T+0 信号判断
            if 't0_signal' in latest_data:
                t0_signal_latest = latest_data['t0_signal']
                if t0_signal_latest == 1:
                    signal_judgment['t0_signal'] = "T+0 买入信号"
                    signal_judgment['t0_suggestion'] = "适合日内短线买入，价格可能低于均值，存在反弹机会"
                elif t0_signal_latest == -1:
                    signal_judgment['t0_signal'] = "T+0 卖出信号"
                    signal_judgment['t0_suggestion'] = "适合日内短线卖出，价格可能高于均值，存在回落风险"
                else:
                    signal_judgment['t0_signal'] = "无 T+0 信号"
                    signal_judgment['t0_suggestion'] = "不适合日内交易，价格波动可能较小或无明显机会"

            # 9.8 量能确认判断
            if 'volume_spike' in latest_data:
                vol_spike_latest = latest_data['volume_spike']
                if vol_spike_latest == 1:
                    signal_judgment['volume_status'] = "放量确认"
                    signal_judgment['volume_suggestion'] = "成交量显著放大，信号可靠性增强，尤其是买入信号"
                else:
                    signal_judgment['volume_status'] = "无放量确认"
                    signal_judgment['volume_suggestion'] = "成交量未见异常，信号可靠性一般，需其他指标确认"

        # 10. 生成详细中文解读并直接输出到日志
        if signal_judgment:
            chinese_interpretation = {
                '信号强度': f"当前信号强度：{signal_judgment.get('score_strength', '未知')}",
                '操作建议': f"操作建议：{signal_judgment.get('score_suggestion', '暂无建议')}",
                '信号置信度': f"信号置信度：{signal_judgment.get('score_confidence', '未知')}",
                '确认强度': f"确认信号强度：{signal_judgment.get('confirmation_strength', '未知')}",
                '确认建议': f"确认信号建议：{signal_judgment.get('confirmation_suggestion', '暂无建议')}",
                '趋势背景': f"长期趋势背景：{signal_judgment.get('trend_context', '未知')}",
                '趋势建议': f"趋势背景建议：{signal_judgment.get('trend_suggestion', '暂无建议')}",
                '排列状态': f"EMA排列状态：{signal_judgment.get('alignment_status', '未知')}",
                '排列建议': f"EMA排列建议：{signal_judgment.get('alignment_suggestion', '暂无建议')}",
                '反转状态': f"趋势反转状态：{signal_judgment.get('reversal_status', '未知')}",
                '反转建议': f"趋势反转建议：{signal_judgment.get('reversal_suggestion', '暂无建议')}",
                '动量状态': f"得分动量状态：{signal_judgment.get('momentum_status', '未知')}",
                '动量建议': f"动量建议：{signal_judgment.get('momentum_suggestion', '暂无建议')}",
                '波动率状态': f"波动率状态：{signal_judgment.get('volatility_status', '未知')}",
                '波动率建议': f"波动率建议：{signal_judgment.get('volatility_suggestion', '暂无建议')}",
                '量能状态': f"量能状态：{signal_judgment.get('volume_status', '未知')}",
                '量能建议': f"量能建议：{signal_judgment.get('volume_suggestion', '暂无建议')}",
                'T+0信号': f"T+0 交易信号：{signal_judgment.get('t0_signal', '未知')}",
                'T+0建议': f"T+0 操作建议：{signal_judgment.get('t0_suggestion', '暂无建议')}"
            }

            # 核心观点 - 先输出最重要的结论
            core_summary = (
                f"【核心信号观点】\n"
                f"1. 信号强度与操作建议：\n"
                f"   - 信号强度：{chinese_interpretation['信号强度']}\n"
                f"   - 操作建议：{chinese_interpretation['操作建议']}\n"
                f"   - 信号置信度：{chinese_interpretation['信号置信度']}\n"
                f"2. 确认信号分析：\n"
                f"   - 确认强度：{chinese_interpretation['确认强度']}\n"
                f"   - 确认建议：{chinese_interpretation['确认建议']}\n"
                f"3. 趋势与反转关键点：\n"
                f"   - 长期趋势：{chinese_interpretation['趋势背景']}\n"
                f"   - 趋势建议：{chinese_interpretation['趋势建议']}\n"
                f"   - 反转状态：{chinese_interpretation['反转状态']}\n"
                f"   - 反转建议：{chinese_interpretation['反转建议']}\n"
            )

            # 详细解读 - 其他辅助分析
            detailed_interpretation = (
                f"【详细信号解读】\n"
                f"1. 趋势排列分析：\n"
                f"   - 排列状态：{chinese_interpretation['排列状态']}\n"
                f"   - 排列建议：{chinese_interpretation['排列建议']}\n"
                f"2. 动量与波动率分析：\n"
                f"   - 动量状态：{chinese_interpretation['动量状态']}\n"
                f"   - 动量建议：{chinese_interpretation['动量建议']}\n"
                f"   - 波动率状态：{chinese_interpretation['波动率状态']}\n"
                f"   - 波动率建议：{chinese_interpretation['波动率建议']}\n"
                f"3. 量能确认分析：\n"
                f"   - 量能状态：{chinese_interpretation['量能状态']}\n"
                f"   - 量能建议：{chinese_interpretation['量能建议']}\n"
                f"4. T+0 交易信号：\n"
                f"   - T+0 信号：{chinese_interpretation['T+0信号']}\n"
                f"   - T+0 建议：{chinese_interpretation['T+0建议']}\n"
            )

            # 统计数据补充说明
            stats_summary = (
                f"【信号统计数据补充说明】\n"
                f"1. 最终评分分布：\n"
                f"   - 平均分：{analysis_results.get('final_score_mean', '未知'):.2f}\n"
                f"   - 标准差：{analysis_results.get('final_score_std', '未知'):.2f}\n"
                f"   - 看涨比例：{analysis_results.get('final_score_bullish_ratio', '未知')*100:.2f}%\n"
                f"   - 看跌比例：{analysis_results.get('final_score_bearish_ratio', '未知')*100:.2f}%\n"
                f"2. 强确认信号统计：\n"
                f"   - 强确认买入比例：{analysis_results.get('strong_confirmation_bullish_ratio', '未知')*100:.2f}%\n"
                f"   - 强确认卖出比例：{analysis_results.get('strong_confirmation_bearish_ratio', '未知')*100:.2f}%\n"
            )

            # 背离信号补充说明
            divergence_summary = "3. 背离信号统计：\n"
            for indicator_key in dd_params['indicators'].keys():
                if dd_params['indicators'].get(indicator_key):
                    divergence_summary += (
                        f"   - {indicator_key.upper()} 指标背离：\n"
                        f"     * 常规看涨背离：{analysis_results.get(f'{indicator_key}_regular_bullish_count', 0)} 次\n"
                        f"     * 常规看跌背离：{analysis_results.get(f'{indicator_key}_regular_bearish_count', 0)} 次\n"
                        f"     * 隐藏看涨背离：{analysis_results.get(f'{indicator_key}_hidden_bullish_count', 0)} 次\n"
                        f"     * 隐藏看跌背离：{analysis_results.get(f'{indicator_key}_hidden_bearish_count', 0)} 次\n"
                    )

            # K线形态补充说明
            kline_summary = (
                f"4. K线形态统计：\n"
                f"   - 看涨形态：\n"
                f"     * 看涨吞没：{analysis_results.get('kline_bullish_engulfing_count', 0)} 次\n"
                f"     * 锤子线：{analysis_results.get('kline_hammer_count', 0)} 次\n"
                f"     * 早晨星：{analysis_results.get('kline_morning_star_count', 0)} 次\n"
                f"     * 光头阳线：{analysis_results.get('kline_bullish_marubozu_count', 0)} 次\n"
                f"   - 看跌形态：\n"
                f"     * 看跌吞没：{analysis_results.get('kline_bearish_engulfing_count', 0)} 次\n"
                f"     * 上吊线：{analysis_results.get('kline_hanging_man_count', 0)} 次\n"
                f"     * 黄昏星：{analysis_results.get('kline_evening_star_count', 0)} 次\n"
                f"     * 光头阴线：{analysis_results.get('kline_bearish_marubozu_count', 0)} 次\n"
                f"   - 中性形态：\n"
                f"     * 十字星：{analysis_results.get('kline_doji_count', 0)} 次\n"
            )

            # 指标反转信号补充说明
            reversal_summary = (
                f"5. 指标反转信号统计：\n"
                f"   - RSI 反转：看涨 {analysis_results.get('rsi_bullish_reversal_count', 0)} 次，看跌 {analysis_results.get('rsi_bearish_reversal_count', 0)} 次\n"
                f"   - Stoch 反转：看涨 {analysis_results.get('stoch_bullish_reversal_count', 0)} 次，看跌 {analysis_results.get('stoch_bearish_reversal_count', 0)} 次\n"
                f"   - CCI 反转：看涨 {analysis_results.get('cci_bullish_reversal_count', 0)} 次，看跌 {analysis_results.get('cci_bearish_reversal_count', 0)} 次\n"
                f"   - 布林带反转：看涨 {analysis_results.get('bb_bullish_reversal_count', 0)} 次，看跌 {analysis_results.get('bb_bearish_reversal_count', 0)} 次\n"
            )

            # 趋势反转信号补充说明
            trend_reversal_summary = (
                f"6. 趋势反转信号统计：\n"
                f"   - 底部反转信号：{analysis_results.get('reversal_bullish_count', 0)} 次，占比 {analysis_results.get('reversal_bullish_ratio', 0)*100:.2f}%\n"
                f"   - 顶部反转信号：{analysis_results.get('reversal_bearish_count', 0)} 次，占比 {analysis_results.get('reversal_bearish_ratio', 0)*100:.2f}%\n"
                f"   - 总反转信号：{analysis_results.get('reversal_total_count', 0)} 次\n"
            )

            # 合并所有中文解读内容并输出到日志，先核心观点，再详细解读和统计数据
            full_interpretation = (
                f"{core_summary}\n"
                f"{detailed_interpretation}\n"
                f"{stats_summary}\n"
                f"{divergence_summary}\n"
                f"{kline_summary}\n"
                f"{reversal_summary}\n"
                f"{trend_reversal_summary}"
            )
            logger.info(full_interpretation)

        # 合并统计结果和信号判断（不包含中文解读）
        analysis_results.update(signal_judgment)

        # 转换为 DataFrame
        analysis_df = pd.DataFrame([analysis_results])
        self.analysis_results = analysis_df
        logger.info("信号详细分析及判断完成，结构化数据已存储到 analysis_results，详细中文解读已输出到日志。")
        return analysis_df











