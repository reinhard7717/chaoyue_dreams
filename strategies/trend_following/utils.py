# 文件: strategies/trend_following/utils.py
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any, Dict, Tuple
import gc

# 这个文件包含所有层级都可能用到的通用辅助函数

def get_param_value(param: Any, default: Any = None) -> Any:
    if isinstance(param, dict) and 'value' in param:
        return param['value']
    if param is not None:
        return param
    return default

def get_params_block(strategy_instance, block_name: str, default_return: Any = None) -> dict:
    """
    【V1.1 健壮版】
    - 智能判断传入的 strategy_instance 是对象还是字典，并从中安全地提取配置。
    """
    if default_return is None:
        default_return = {}

    config = {}
    if isinstance(strategy_instance, dict):
        # 如果传入的是字典，直接用 .get() 方法获取
        config = strategy_instance.get('unified_config', {})
    else:
        # 如果传入的是对象实例，用 getattr() 安全地获取属性
        config = getattr(strategy_instance, 'unified_config', {})

    # 后续逻辑都基于上面提取出的 config 字典进行操作，不再依赖于 strategy_instance 的类型
    trend_follow_params = config.get('strategy_params', {}).get('trend_follow', {})
    params = trend_follow_params.get(block_name)
    if params is None:
        # 如果第一层没找到，从配置的根目录再找一次
        params = config.get(block_name)
    
    if params is not None:
        return params
    return default_return

def ensure_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    converted_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            first_valid_item = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_valid_item, Decimal):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                converted_cols.append(col)
    # if not converted_cols:
        # print("      -> 所有数值列类型正常，无需转换。")
    return df

# def fuse_multi_level_scores(atomic_states: Dict[str, pd.Series], df_index: pd.Index, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
#     """
#     【新增辅助函数】融合S/A/B等多层置信度分数。
#     - 逻辑: 根据给定的权重，将 'SCORE_..._S', 'SCORE_..._A', 'SCORE_..._B' 等分数
#             加权融合成一个单一的综合分数。
#     - :param atomic_states: 包含所有原子状态的字典。
#     - :param df_index: DataFrame的索引，用于创建Series。
#     - :param base_name: 分数的基础名称 (例如 'MA_BULLISH_RESONANCE').
#     - :param weights: 一个字典，定义了 'S', 'A', 'B' 等级的权重。
#     - :return: 融合后的分数 (pd.Series).
#     """
#     if weights is None:
#         weights = {'S': 1.0, 'A': 0.6, 'B': 0.3}
#     total_score = pd.Series(0.0, index=df_index)
#     total_weight = 0.0
#     # 动态地获取并加权S/A/B等级的分数
#     for level, weight in weights.items():
#         score_name = f"SCORE_{base_name}_{level}"
#         if score_name in atomic_states:
#             score_series = atomic_states[score_name]
#             total_score += score_series * weight
#             total_weight += weight
#     # 如果没有找到任何等级的分数，返回一个中性分数
#     if total_weight == 0:
#         # 尝试获取没有等级的单一分数
#         single_score_name = f"SCORE_{base_name}"
#         if single_score_name in atomic_states:
#             return atomic_states[single_score_name]
#         return pd.Series(0.5, index=df_index)
#     # 归一化处理
#     return (total_score / total_weight).clip(0, 1)

def get_unified_score(atomic_states: Dict[str, pd.Series], df_index: pd.Index, base_name: str) -> pd.Series:
    """
    【V1.0 · 净化版】获取唯一的、归一化的终极信号分数。
    - 核心职责: 根据基础名称 (如 'CHIP_BULLISH_RESONANCE')，直接查找并返回唯一的终极信号
                  (如 'SCORE_CHIP_BULLISH_RESONANCE')。
    - 替代目标: 完全取代过时的、复杂的 fuse_multi_level_scores 函数。
    """
    # 直接构建唯一的、简化的信号全名
    signal_name = f"SCORE_{base_name}"
    
    # 直接从原子状态中获取这个唯一的信号，如果找不到，则返回一个包含默认值0.0的Series
    score_series = atomic_states.get(signal_name, pd.Series(0.0, index=df_index))
    
    # 确保返回的Series索引正确并填充缺失值
    return score_series.reindex(df_index).fillna(0.0).astype(np.float32)

def format_debug_dates(signal_series: pd.Series, display_limit: int = 10) -> str:
    if not isinstance(signal_series, pd.Series) or signal_series.dtype != bool:
        return ""
    active_dates = signal_series.index[signal_series]
    count = len(active_dates)
    if count == 0:
        return ""
    date_strings = [d.strftime('%Y-%m-%d') for d in active_dates]
    if count > display_limit:
        return f" -> 日期: [...{date_strings[-display_limit:]}] (共 {count} 天)"
    else:
        return f" -> 日期: {date_strings}"

def create_persistent_state(df: pd.DataFrame, entry_event_series: pd.Series, persistence_days: int, break_condition_series: pd.Series, state_name: str) -> pd.Series:
    """
    【V1.1 · 赫尔墨斯之翼优化版】创建持久化状态
    - 性能优化: 1. 预先计算并筛选出所有可能的进入点和中断点索引，减少主循环内的重复计算。
                  2. 这是一个典型的路径依赖问题，完全向量化极其复杂。当前优化是在保持逻辑绝对正确的前提下，
                     对循环内部操作的极致精简。对于超大规模数据，未来可考虑使用Numba进行JIT编译加速。
    - 核心逻辑: 从一个进入事件开始，状态持续`persistence_days`天，除非被`break_condition`提前中断。
    """
    persistent_series = pd.Series(False, index=df.index, dtype=bool)
    
    # 预先计算并获取所有可能为True的索引，将Series操作移出循环
    entry_indices = entry_event_series.index[entry_event_series]
    if entry_indices.empty:
        return persistent_series.astype(np.int8) # 返回int8以节省内存

    # 预先筛选出所有可能的中断点索引，避免在循环中对整个Series进行掩码操作
    break_indices = df.index[break_condition_series]

    for entry_idx in entry_indices:
        window_end_date = entry_idx + pd.Timedelta(days=persistence_days)
        
        # 在预筛选的中断点索引上进行范围查找，比在整个Series上应用掩码更快
        # 使用searchsorted可以进一步优化，但对于通用DatetimeIndex，直接切片已足够高效
        possible_break_points = break_indices[(break_indices >= entry_idx) & (break_indices <= window_end_date)]
        
        if not possible_break_points.empty:
            # 如果在窗口内找到中断点，状态在第一个中断点处结束
            end_date = possible_break_points[0]
        else:
            # 如果没有中断点，状态持续到窗口结束
            # 获取df.index在窗口内的最后一个有效索引
            window_indices = df.index[(df.index >= entry_idx) & (df.index <= window_end_date)]
            end_date = window_indices[-1] if not window_indices.empty else entry_idx

        # 使用.loc进行高效的区间赋值
        persistent_series.loc[entry_idx:end_date] = True
        
    return persistent_series.astype(np.int8) # 使用int8类型，内存效率最高

def optimize_df_memory(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    【内存优化工具】
    遍历DataFrame的所有列，并尽可能地将其数据类型转换为占用内存更小的类型。
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category' and 'datetime' not in col_type.name:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # 对于浮点数，优先使用float32，精度足够且内存减半
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'    -> [内存优化] DataFrame内存从 {start_mem:.2f} MB 优化至 {end_mem:.2f} MB (减少了 {(start_mem - end_mem) / start_mem * 100:.1f}%)')
    return df

def normalize_score(series: pd.Series, target_index: pd.Index, window: int, ascending: bool = True, default_value=0.5) -> pd.Series:
    """
    【V1.1 · 赫尔墨斯之翼注释升级版】通用归一化引擎 (万物标尺)
    - 战略意义: 作为系统中最基础的“标尺”，负责将任意一个指标的数值，转化为在[0, 1]区间内、具有历史相对意义的分数。
                它是系统进行跨指标比较和融合的基石。
    - 核心逻辑: 使用滚动窗口的百分比排名 (rank(pct=True)) 来实现归一化。
    - :param series: 原始数据序列。
    - :param target_index: 目标DataFrame的索引，用于确保输出对齐。
    - :param window: 滚动窗口，定义了“近期历史”的范围。
    - :param ascending: True表示值越大分数越高，False反之。
    - :param default_value: 默认填充值。
    - :return: 在[0, 1]区间归一化的pd.Series。
    """
    if series is None or series.isnull().all() or series.empty:
        return pd.Series(default_value, index=target_index, dtype=np.float32)

    # 确保series的索引与目标索引对齐，避免后续操作因索引不匹配产生问题
    series = series.reindex(target_index)

    # min_periods确保在窗口数据不足时也能计算，增加了早期数据的可用性
    min_periods = max(1, int(window * 0.2))
    
    rank = series.rolling(
        window=window, 
        min_periods=min_periods
    ).rank(
        pct=True, 
        ascending=ascending
    )
    
    # 再次使用reindex确保最终输出的索引是完整的，并填充可能因滚动产生的NaN
    return rank.reindex(target_index).fillna(default_value).astype(np.float32)

def calculate_context_scores(df: pd.DataFrame, atomic_states: Dict) -> Tuple[pd.Series, pd.Series]:
    """
    【V8.3 · 得墨忒耳收获协议版】(逻辑保持不变, 仅为展示完整性)
    - 核心逻辑: 最终底部上下文分 = max(盖亚基石分, 受深度过滤器约束的常规底部组件分)。
    - 协同作战: 此函数接收由“宙斯权杖协议”计算出的、更智能的gaia_bedrock_support_score。
    """
    if isinstance(df, dict):
        df = df.get('df_indicators', pd.DataFrame())
    close_col, high_col, low_col = 'close_D', 'high_D', 'low_D'
    if close_col not in df.columns:
        print(f"      -> [calculate_context_scores] 警告: 输入的DataFrame缺少'{close_col}'列，无法计算。")
        empty_series = pd.Series(0.5, index=df.index if not df.empty else None, dtype=np.float32)
        return empty_series, empty_series
    strategy_instance_ref = atomic_states.get('strategy_instance_ref') or getattr(df, 'strategy', None)
    p_synthesis = get_params_block(strategy_instance_ref, 'ultimate_signal_synthesis_params', {}) if strategy_instance_ref else {}
    depth_threshold = get_param_value(p_synthesis.get('deep_bearish_threshold'), 0.05)
    ma55_lifeline = df.get('EMA_55_D', df[close_col])
    is_deep_bearish_zone = (df[close_col] < ma55_lifeline * (1 - depth_threshold)).astype(float)
    # --- 底部上下文分数计算 ---
    ma55_slope = ma55_lifeline.diff(3).fillna(0)
    slope_moderator = (0.5 + 0.5 * np.tanh(ma55_slope * 100)).fillna(0.5)
    distance_from_ma55 = (df[close_col] - ma55_lifeline) / ma55_lifeline.replace(0, np.nan)
    lifeline_support_score_raw = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2).fillna(0.0)
    lifeline_support_score = lifeline_support_score_raw * slope_moderator
    price_pos_yearly = normalize_score(df[close_col], df.index, window=250, ascending=True, default_value=0.5)
    absolute_value_zone_score = 1.0 - price_pos_yearly
    p_fib_support = get_params_block(strategy_instance_ref, 'fibonacci_support_params', {}) if strategy_instance_ref else {}
    dynamic_support_score = pd.Series(0.0, index=df.index, dtype=np.float32)
    if get_param_value(p_fib_support.get('enabled'), False):
        fib_periods = get_param_value(p_fib_support.get('periods'), [34, 55, 89, 144, 233])
        tolerance_pct = get_param_value(p_fib_support.get('tolerance_pct'), 0.01)
        level_scores = get_param_value(p_fib_support.get('level_scores'), {})
        trigger_mask = df[close_col] < ma55_lifeline
        for period in fib_periods:
            period_str = str(period)
            if period_str not in level_scores: continue
            historical_low = df[close_col].rolling(window=period, min_periods=max(1, int(period*0.8))).min().shift(1)
            support_line = historical_low * (1 - tolerance_pct)
            is_defended = df[low_col] >= support_line
            dynamic_support_score.loc[trigger_mask & is_defended] = level_scores[period_str]
    gaia_params = get_param_value(p_synthesis.get('gaia_bedrock_params'), {})
    gaia_bedrock_support_score = _calculate_gaia_bedrock_support(df, gaia_params)
    deep_bottom_context_score_values = np.maximum.reduce([
        lifeline_support_score.values,
        absolute_value_zone_score.values,
        dynamic_support_score.values
    ])
    deep_bottom_context_score = pd.Series(deep_bottom_context_score_values, index=df.index, dtype=np.float32)
    rsi_w_col = 'RSI_13_W'
    rsi_w_oversold_score = normalize_score(df.get(rsi_w_col, pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
    cycle_phase = atomic_states.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index)).fillna(0.0)
    cycle_trough_score = (1 - cycle_phase) / 2.0
    context_weights = get_param_value(p_synthesis.get('bottom_context_weights'), {'price_pos': 0.5, 'rsi_w': 0.3, 'cycle': 0.2})
    score_components = {'price_pos': deep_bottom_context_score, 'rsi_w': rsi_w_oversold_score, 'cycle': cycle_trough_score}
    valid_scores, valid_weights = [], []
    for name, weight in context_weights.items():
        if name in score_components and weight > 0:
            valid_scores.append(score_components[name].values)
            valid_weights.append(weight)
    if not valid_scores:
        bottom_context_score_raw = pd.Series(0.5, index=df.index, dtype=np.float32)
    else:
        weights_array = np.array(valid_weights)
        total_weight = weights_array.sum()
        normalized_weights = weights_array / total_weight if total_weight > 0 else np.full_like(weights_array, 1.0 / len(weights_array))
        stacked_scores = np.stack(valid_scores, axis=0)
        safe_scores = np.maximum(stacked_scores, 1e-9)
        weighted_log_sum = np.sum(np.log(safe_scores) * normalized_weights[:, np.newaxis], axis=0)
        bottom_context_score_raw = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
    conventional_bottom_score = bottom_context_score_raw * is_deep_bearish_zone
    bottom_context_score = np.maximum(conventional_bottom_score, gaia_bedrock_support_score).astype(np.float32)
    # --- 顶部上下文分数计算 (逻辑不变) ---
    ma55 = df.get('EMA_55_D', df[close_col])
    rolling_high_55d = df[high_col].rolling(window=55, min_periods=21).max()
    wave_channel_height = (rolling_high_55d - ma55).replace(0, 1e-9)
    stretch_score = ((df[close_col] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
    ma_periods = [5, 13, 21, 55]
    short_ma_cols = [f'EMA_{p}_D' for p in ma_periods[:-1]]
    long_ma_cols = [f'EMA_{p}_D' for p in ma_periods[1:]]
    if all(col in df for col in short_ma_cols + long_ma_cols):
        short_mas = df[short_ma_cols].values
        long_mas = df[long_ma_cols].values
        misalignment_matrix = (short_mas < long_mas).astype(np.float32)
        misalignment_score_values = np.mean(misalignment_matrix, axis=1)
        misalignment_score = pd.Series(misalignment_score_values, index=df.index)
    else:
        misalignment_score = pd.Series(0.5, index=df.index)
    bias_col = 'BIAS_21_D'
    bias_abs = df.get(bias_col, pd.Series(0, index=df.index)).abs()
    bias_params = get_param_value(p_synthesis.get('bias_overheat_params'), {})
    warning_threshold = get_param_value(bias_params.get('warning_threshold'), 0.15)
    danger_threshold = get_param_value(bias_params.get('danger_threshold'), 0.25)
    denominator = danger_threshold - warning_threshold
    if denominator <= 1e-6:
        overheat_score = (bias_abs > danger_threshold).astype(float)
    else:
        overheat_score = ((bias_abs - warning_threshold) / denominator).clip(0, 1)
    overheat_score = overheat_score.fillna(0.0)
    top_context_score = (stretch_score * misalignment_score * overheat_score)**(1/3)
    top_context_score = top_context_score.astype(np.float32)
    return bottom_context_score, top_context_score

def normalize_to_bipolar(series: pd.Series, target_index: pd.Index, window: int, sensitivity: float = 1.0, default_value: float = 0.0) -> pd.Series:
    """
    【V2.1 · 赫尔墨斯之翼注释升级版】双极归一化引擎 (力学罗盘)
    - 战略意义: 用于将一个指标的变化率或原始值，转化为一个同时蕴含【方向】和【强度】的标准化分数。
                它是构建“速度 vs 加速度”、“推力 vs 阻力”等力学分析模型的核心工具。
                +1 代表极强的正向偏离，-1 代表极强的负向偏离，0 代表符合近期常态。
    - 核心逻辑: 采用滚动Z-score并使用tanh函数进行平滑压缩，完美适用于四象限分析。
    - :param sensitivity: 敏感度因子。值越小，Z-score的绝对值越大，得分越快地趋近于±1。
                          战术上，调小此参数可放大微小变化的信号强度，用于捕捉早期拐点。
    - :return: 归一化到(-1, 1)区间的pd.Series。
    """
    if series is None or series.isnull().all() or series.empty:
        return pd.Series(default_value, index=target_index, dtype=np.float32)
    series = series.reindex(target_index)
    min_periods = max(1, int(window * 0.2))
    
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    
    # 避免除以零。如果窗口内值无波动，标准差为0，Z-score应为0（无偏离）。
    rolling_std = rolling_std.replace(0, np.nan)
    
    # 计算Z-score，sensitivity作为分母，调节敏感度
    z_score = (series - rolling_mean) / (rolling_std * sensitivity)
    
    # 使用tanh函数进行平滑压缩到(-1, 1)，相比clip，tanh能保留更多的中间值信息
    bipolar_score = np.tanh(z_score)
    
    return bipolar_score.reindex(target_index).fillna(default_value).astype(np.float32)

def calculate_holographic_dynamics(df: pd.DataFrame, base_name: str, norm_window: int) -> Tuple[pd.Series, pd.Series]:
    """
    【V2.3 · 赫尔墨斯之翼优化版】全息动态计算引擎
    - 战略意义: 捕捉一个指标的“动态画像”，不仅看它的值，更看它的“速度变化”(加速度)和“力量变化”(加加速度/Jerk)。
                这使得系统能区分“匀速上涨”和“加速上涨”，从而识别趋势的真实动能。
    - 性能优化: 1. 使用`np.sqrt`替代`**0.5`，提高代码可读性。
                  2. 确保所有返回的Series都为float32类型。
    """
    # 创建一个默认的Series，用于在df.get找不到列时返回，构建双重保险
    default_series = pd.Series(0.0, index=df.index, dtype=np.float32)

    # 维度一：速度变化 (加速度) - 衡量趋势斜率的变化趋势
    slope_1 = df.get(f'SLOPE_1_{base_name}', default_series)
    slope_5 = df.get(f'SLOPE_5_{base_name}', default_series)
    slope_diff = slope_1 - slope_5
    velocity_accel_score = normalize_score(slope_diff, df.index, norm_window, ascending=True)
    velocity_decel_score = normalize_score(slope_diff, df.index, norm_window, ascending=False)

    # 维度二：力量变化 (加加速度 / Jerk) - 衡量趋势加速度的变化趋势
    accel_1 = df.get(f'ACCEL_1_{base_name}', default_series)
    accel_5 = df.get(f'ACCEL_5_{base_name}', default_series)
    accel_diff = accel_1 - accel_5
    jerk_accel_score = normalize_score(accel_diff, df.index, norm_window, ascending=True)
    jerk_decel_score = normalize_score(accel_diff, df.index, norm_window, ascending=False)
    
    # 融合：两大维度必须共振，形成合力
    # 使用np.sqrt，意图更清晰
    bullish_holographic_score = np.sqrt(velocity_accel_score * jerk_accel_score).astype(np.float32)
    bearish_holographic_score = np.sqrt(velocity_decel_score * jerk_decel_score).astype(np.float32)
    
    return bullish_holographic_score, bearish_holographic_score

def transmute_health_to_ultimate_signals(
    df: pd.DataFrame,
    atomic_states: Dict,
    overall_health: Dict,
    params: Dict,
    domain_prefix: str
) -> Dict[str, pd.Series]:
    """
    【V5.0 · 赫尔墨斯之靴协议版】终极信号中央合成引擎
    - 核心革命: 签署“赫尔墨斯之靴协议”，建立“战略反转”与“战术反转”的双轨制。
    - 新核心逻辑:
      1. 【战略反转】: 维持“雅典娜之盾”逻辑，并增加 (1-趋势确认) 因子，确保其只在趋势混沌或下跌时活跃。
      2. 【战术反转】: 新增独立的计算路径，融合“宏观趋势许可”、“动态反转加速度”和“微观动能”，用于捕捉上升趋势中的回调买点。
    """
    states = {}
    # --- 1. 获取通用参数和上下文信号 ---
    resonance_tf_weights = get_param_value(params.get('resonance_tf_weights'), {'short': 0.2, 'medium': 0.5, 'long': 0.3})
    reversal_tf_weights = get_param_value(params.get('reversal_tf_weights'), {'short': 0.6, 'medium': 0.3, 'long': 0.1})
    periods = get_param_value(params.get('periods'), [1, 5, 13, 21, 55])
    norm_window = get_param_value(params.get('norm_window'), 55)
    bottom_context_bonus_factor = get_param_value(params.get('bottom_context_bonus_factor'), 0.5)
    exponent = get_param_value(params.get('final_score_exponent'), 1.0)
    atomic_states['strategy_instance_ref'] = df.strategy if hasattr(df, 'strategy') else {}
    bottom_context_score, top_context_score = calculate_context_scores(df, atomic_states)
    if 'strategy_instance_ref' in atomic_states:
        del atomic_states['strategy_instance_ref']
    recent_reversal_context = atomic_states.get('SCORE_CONTEXT_RECENT_REVERSAL', pd.Series(0.0, index=df.index))
    default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
    # --- 2. 上下文计算 ---
    new_high_params = get_param_value(params.get('new_high_context_params'), {})
    new_high_context_score = _calculate_new_high_context(df, new_high_params)
    atomic_states['CONTEXT_NEW_HIGH_STRENGTH'] = new_high_context_score
    memory_retention_factor = 1.0 - new_high_context_score
    recent_reversal_context_modulated = recent_reversal_context * memory_retention_factor
    trend_confirmation_params = get_param_value(params.get('trend_confirmation_context_params'), {})
    # [代码修改] 调用公共函数
    trend_confirmation_context = calculate_trend_confirmation_context(df, trend_confirmation_params, norm_window)
    atomic_states['CONTEXT_TREND_CONFIRMED'] = trend_confirmation_context
    # [代码新增] 为“赫尔墨斯之靴”计算专属上下文
    tactical_params = get_param_value(params.get('tactical_reversal_params'), {})
    macro_window = get_param_value(tactical_params.get('macro_trend_window'), 3)
    macro_trend_permit_context = trend_confirmation_context.rolling(window=macro_window, min_periods=1).mean()
    atomic_states['CONTEXT_MACRO_TREND_PERMIT'] = macro_trend_permit_context
    dynamic_reversal_params = get_param_value(params.get('dynamic_reversal_context_params'), {})
    dynamic_reversal_context = _calculate_dynamic_reversal_context(df, dynamic_reversal_params, norm_window)
    atomic_states['CONTEXT_DYNAMIC_REVERSAL'] = dynamic_reversal_context
    # --- 3. 信号计算 ---
    # 战略底部反转 (Strategic Bottom Reversal)
    bullish_reversal_health = {p: overall_health['s_bull'].get(p, default_series) * (1 - overall_health['s_bear'].get(p, default_series)) for p in periods}
    bullish_short_force_rev = (bullish_reversal_health.get(1, default_series) * bullish_reversal_health.get(5, default_series))**0.5
    bullish_medium_trend_rev = (bullish_reversal_health.get(13, default_series) * bullish_reversal_health.get(21, default_series))**0.5
    bullish_long_inertia_rev = bullish_reversal_health.get(55, default_series)
    overall_bullish_reversal_trigger = ((bullish_short_force_rev ** reversal_tf_weights['short']) * (bullish_medium_trend_rev ** reversal_tf_weights['medium']) * (bullish_long_inertia_rev ** reversal_tf_weights['long']))
    raw_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + recent_reversal_context_modulated * bottom_context_bonus_factor)).clip(0, 1)
    # [代码修改] 战略反转只在趋势未确认时最强，与战术反转形成互补
    final_bottom_reversal_score = raw_bottom_reversal_score * bottom_context_score * (1 - trend_confirmation_context)
    # [代码新增] 战术回调反转 (Tactical Pullback Reversal)
    final_tactical_reversal_score = calculate_tactical_reversal_score(df, atomic_states, overall_health, tactical_params, norm_window)
    # 看涨共振 (Bullish Resonance)
    bullish_resonance_health = {p: overall_health['s_bull'].get(p, default_series) * overall_health['d_intensity'].get(p, default_series) for p in periods}
    bullish_short_force_res = (bullish_resonance_health.get(1, default_series) * bullish_resonance_health.get(5, default_series))**0.5
    bullish_medium_trend_res = (bullish_resonance_health.get(13, default_series) * bullish_resonance_health.get(21, default_series))**0.5
    bullish_long_inertia_res = bullish_resonance_health.get(55, default_series)
    overall_bullish_resonance = ((bullish_short_force_res ** resonance_tf_weights['short']) * (bullish_medium_trend_res ** resonance_tf_weights['medium']) * (bullish_long_inertia_res ** resonance_tf_weights['long']))
    # 看跌共振 (Bearish Resonance)
    bearish_resonance_health = {p: overall_health['s_bear'].get(p, default_series) * (1 - overall_health['d_intensity'].get(p, default_series)) for p in periods}
    bearish_short_force_res = (bearish_resonance_health.get(1, default_series) * bearish_resonance_health.get(5, default_series))**0.5
    bearish_medium_trend_res = (bearish_resonance_health.get(13, default_series) * bearish_resonance_health.get(21, default_series))**0.5
    bearish_long_inertia_res = bearish_resonance_health.get(55, default_series)
    overall_bearish_resonance = ((bearish_short_force_res ** resonance_tf_weights['short']) * (bearish_medium_trend_res ** resonance_tf_weights['medium']) * (bearish_long_inertia_res ** resonance_tf_weights['long']))
    # 顶部反转 (Top Reversal)
    bearish_reversal_health = {p: overall_health['s_bear'].get(p, default_series) * (1 - overall_health['s_bull'].get(p, default_series)) for p in periods}
    bearish_short_force_rev = (bearish_reversal_health.get(1, default_series) * bearish_reversal_health.get(5, default_series))**0.5
    bearish_medium_trend_rev = (bearish_reversal_health.get(13, default_series) * bearish_reversal_health.get(21, default_series))**0.5
    bearish_long_inertia_rev = bearish_reversal_health.get(55, default_series)
    overall_bearish_reversal_trigger = ((bearish_short_force_rev ** reversal_tf_weights['short']) * (bearish_medium_trend_rev ** reversal_tf_weights['medium']) * (bearish_long_inertia_rev ** reversal_tf_weights['long']))
    final_top_reversal_score = (overall_bearish_reversal_trigger * (1 + top_context_score * bottom_context_bonus_factor)).clip(0, 1)
    # --- 4. 组装并返回最终信号字典 ---
    final_signal_map = {
        f'SCORE_{domain_prefix}_BULLISH_RESONANCE': (overall_bullish_resonance ** exponent),
        f'SCORE_{domain_prefix}_BOTTOM_REVERSAL': (final_bottom_reversal_score ** exponent),
        f'SCORE_{domain_prefix}_TACTICAL_REVERSAL': (final_tactical_reversal_score ** exponent), # [代码新增]
        f'SCORE_{domain_prefix}_BEARISH_RESONANCE': (overall_bearish_resonance ** exponent),
        f'SCORE_{domain_prefix}_TOP_REVERSAL': (final_top_reversal_score ** exponent)
    }
    for signal_name, score in final_signal_map.items():
        states[signal_name] = score.astype(np.float32)
    return states

def _calculate_new_high_context(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V2.1 · 赫尔墨斯之翼优化版】多维新高上下文分数计算器
    - 性能优化: 废除了“先创建Series列表再求和”的低效模式，改为在预初始化的Numpy数组上进行
                  原地累加，显著减少了内存分配和计算开销。
    - 核心逻辑: 保持“价格突破、均线斜率、乖离健康度”三位一体的评估逻辑不变。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    periods = get_param_value(params.get('periods'), [5, 13, 21, 55])
    period_weights = get_param_value(params.get('period_weights'), {})
    bias_thresholds = get_param_value(params.get('bias_thresholds'), {})
    fusion_weights = get_param_value(params.get('fusion_weights'), {})
    
    # 预先初始化一个Numpy数组用于累加分数，避免创建Series列表
    final_scores_np = np.zeros(len(df.index), dtype=np.float32)
    
    for p in periods:
        period_weight = period_weights.get(str(p), 0)
        if period_weight == 0:
            continue

        # 维度一: 价格突破
        rolling_high = df['high_D'].rolling(window=p, min_periods=1).max().shift(1)
        is_new_high_score = (df['high_D'] > rolling_high).astype(np.float32)

        # 维度二: 趋势确认 (均线斜率)
        ma_slope_col = f'SLOPE_{p}_EMA_{p}_D'
        if ma_slope_col not in df.columns: ma_slope_col = f'SLOPE_{p}_close_D'
        ma_slope = df.get(ma_slope_col, pd.Series(0, index=df.index))
        ma_slope_score = normalize_score(ma_slope, df.index, window=p*2, ascending=True)

        # 维度三: 乖离健康度
        bias_period = 21 if p <= 21 else 55
        bias_col = f'BIAS_{bias_period}_D'
        bias_threshold = bias_thresholds.get(str(bias_period), 0.2)
        bias_value = df.get(bias_col, pd.Series(0, index=df.index)).abs()
        bias_health_score = (1 - (bias_value / bias_threshold)).clip(0, 1)

        # 三位一体融合，得到该周期的综合新高分
        period_new_high_score = (
            is_new_high_score * fusion_weights.get('new_high', 0.4) +
            ma_slope_score * fusion_weights.get('ma_slope', 0.3) +
            bias_health_score * fusion_weights.get('bias_health', 0.3)
        )
        
        # 直接在Numpy数组上进行加权累加
        final_scores_np += period_new_high_score.values * period_weight

    total_weight = sum(w for p, w in period_weights.items() if int(p) in periods)
    if total_weight == 0:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
        
    # 一次性完成归一化和Series创建
    final_score_values = final_scores_np / total_weight
    return pd.Series(final_score_values, index=df.index).clip(0, 1).astype(np.float32)

def calculate_trend_confirmation_context(df: pd.DataFrame, params: Dict, norm_window: int) -> pd.Series:
    """
    【V3.0 · 伊卡洛斯协议版】趋势确认上下文计算器 (波塞冬三叉戟)
    - 核心革命: 签署“伊卡洛斯协议”，将“趋势健康度(BIAS)”从确认逻辑中剥离，避免因趋势过热导致确认失效的“伊卡洛斯问题”。
    - 新核心逻辑: 趋势确认分 = 趋势强度分 * 趋势方向分。一个纯粹的、只判断趋势是否存在及方向的强大引擎。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    adx_threshold = get_param_value(params.get('adx_threshold'), 20)
    # 叉戟一: 趋势强度 (ADX) - 浪潮有多高？
    adx = df.get('ADX_14_D', pd.Series(0, index=df.index))
    is_trending = (adx > adx_threshold).astype(np.float32)
    strength_score = normalize_score(adx, df.index, window=norm_window, ascending=True) * is_trending
    # 叉戟二: 趋势方向 (PDI/NDI) - 浪潮往哪边涌？
    pdi = df.get('PDI_14_D', pd.Series(0, index=df.index))
    ndi = df.get('NDI_14_D', pd.Series(0, index=df.index))
    direction_score = (pdi > ndi).astype(np.float32)
    # [代码修改] 最终融合：现在是二叉戟合一，更纯粹、更强大
    trend_confirmation_score = (strength_score * direction_score)
    return trend_confirmation_score.clip(0, 1).astype(np.float32)

def calculate_tactical_reversal_score(
    df: pd.DataFrame,
    atomic_states: Dict,
    overall_health: Dict,
    params: Dict,
    norm_window: int
) -> pd.Series:
    """
    【V1.0 · 新增】战术反转信号计算器 (赫尔墨斯的飞翼鞋)
    - 战略意义: 捕捉上升趋势中的健康回调买点。它模拟了飞行员的操作：先获得飞行许可，再等待有利气流，最后点燃引擎。
    - 核心公式: 战术反转分 = 飞行许可 * 有利气流 * 引擎推力
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    momentum_weight = get_param_value(params.get('momentum_weight'), 0.5)
    relational_power_weight = get_param_value(params.get('relational_power_weight'), 0.5)
    
    # 准入证: 飞行许可 (Permit to Fly) - 宏观趋势是否允许进行看涨操作？
    trend_permission_score = atomic_states.get('CONTEXT_MACRO_TREND_PERMIT', pd.Series(0.0, index=df.index))
    # 核心驱动: 有利气流 (Favorable Wind) - 短期结构是否已形成反转加速度？
    dynamic_reversal_context_score = atomic_states.get('CONTEXT_DYNAMIC_REVERSAL', pd.Series(0.0, index=df.index))
    # 最终推力: 引擎推力 (Engine Thrust) - 此刻是否有多周期动能和关系力量的共振？
    relational_power = atomic_states.get('SCORE_ATOMIC_RELATIONAL_DYNAMICS', pd.Series(0.5, index=df.index))
    short_term_momentum = overall_health.get('d_intensity', {}).get(1, pd.Series(0.5, index=df.index))
    reversal_momentum_score = (
        relational_power * relational_power_weight +
        short_term_momentum * momentum_weight
    )
    # 最终融合
    tactical_reversal_score = (
        trend_permission_score *
        dynamic_reversal_context_score *
        reversal_momentum_score
    )
    
    return tactical_reversal_score.clip(0, 1).astype(np.float32)

def _calculate_dynamic_reversal_context(df: pd.DataFrame, params: Dict, norm_window: int) -> pd.Series:
    """
    【V1.0 · 新增】动态反转上下文计算器 (二阶求导引擎)
    - 战略意义: 捕捉趋势的“拐点”，即变化率的变化率达到最大的时刻。这使得系统能在反转的最早期、
                最具爆发力的时刻介入，而非等待趋势形成后再追赶。
    - 核心逻辑: 通过对“均线距离”和“均线斜率”进行二阶求导，量化反转的“加速度”。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    short_ma_period = get_param_value(params.get('short_ma_period'), 5)
    mid_ma_period = get_param_value(params.get('mid_ma_period'), 21)
    slope_period = get_param_value(params.get('slope_period'), 3)
    weights = get_param_value(params.get('fusion_weights'), {'distance_accel': 0.5, 'slope_accel': 0.5})
    
    short_ma_col = f'EMA_{short_ma_period}_D'
    mid_ma_col = f'EMA_{mid_ma_period}_D'
    short_ma_slope_col = f'SLOPE_{slope_period}_EMA_{short_ma_period}_D'
    if not all(c in df.columns for c in [short_ma_col, mid_ma_col, short_ma_slope_col]):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    # 维度一: 均线距离的收敛加速度 (Convergence Acceleration)
    # 衡量短期均线从下方“追赶”中期均线的加速度。
    ma_distance = df[short_ma_col] - df[mid_ma_col]
    ma_distance_slope = ma_distance.diff(slope_period).fillna(0)
    distance_accel_score = normalize_score(ma_distance_slope, df.index, window=norm_window, ascending=True)
    # 维度二: 短期均线斜率的扭转加速度 (Slope Turn Acceleration)
    # 衡量短期均线自身从“向下”扭转为“向上”的加速度。
    short_ma_slope = df[short_ma_slope_col]
    short_ma_slope_accel = short_ma_slope.diff(slope_period).fillna(0)
    slope_accel_score = normalize_score(short_ma_slope_accel, df.index, window=norm_window, ascending=True)
    # 最终融合: 两个维度的加速度加权求和，形成最终的动态反转分数
    dynamic_reversal_score = (
        distance_accel_score * weights.get('distance_accel', 0.5) +
        slope_accel_score * weights.get('slope_accel', 0.5)
    )
    
    return dynamic_reversal_score.clip(0, 1).astype(np.float32)

def _calculate_gaia_bedrock_support(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V2.0 · 宙斯权杖协议版】“盖亚基石”支撑分计算引擎
    - 核心革命: 引入“代理总指挥”(Acting Lifeline)概念。对每一天，动态寻找低于收盘价的、最强的那条长期均线作为支撑基准。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    support_levels = get_param_value(params.get('support_levels'), [55, 89, 144, 233])
    confirmation_window = get_param_value(params.get('confirmation_window'), 3)
    defense_score = get_param_value(params.get('defense_score'), 0.6)
    confirmation_score = get_param_value(params.get('confirmation_score'), 1.0)
    close_col, low_col = 'close', 'low'
    # 1. 收集所有防线指挥官
    ma_cols = [f'EMA_{p}_D' for p in support_levels if f'EMA_{p}_D' in df.columns]
    if not ma_cols:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    ma_df = df[ma_cols]
    # 2. 任命“代理总指挥”：对每一天，找到低于收盘价的最高指挥官
    #    首先，将所有高于收盘价的均线值设为NaN，表示它们不是支撑
    ma_df_below_price = ma_df.where(ma_df.le(df[close_col], axis=0))
    #    然后，从剩下的有效支撑中，选出最大的那一个作为“代理总指挥”
    acting_lifeline = ma_df_below_price.max(axis=1)
    # 3. 基于“代理总指挥”进行战况评估
    #    如果某天价格低于所有支撑线，acting_lifeline会是NaN，后续判断自动为False
    is_defended = (df[low_col] <= acting_lifeline) & (df[close_col] >= acting_lifeline)
    #    注意：这里的确认逻辑也应该基于动态的acting_lifeline，但rolling无法直接作用于动态基准。
    #    我们简化处理：只要连续N天收盘价都高于其各自的代理总指挥，就视为确认。
    is_above_acting_lifeline = df[close_col] > acting_lifeline
    is_confirmed = is_above_acting_lifeline.rolling(window=confirmation_window).sum() >= confirmation_window
    # 4. 计分
    gaia_score = pd.Series(0.0, index=df.index, dtype=np.float32)
    gaia_score[is_defended] = defense_score
    gaia_score[is_confirmed] = confirmation_score # 确认信号覆盖防守信号
    return gaia_score.astype(np.float32)















