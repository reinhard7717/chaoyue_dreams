# 文件: strategies/trend_following/utils.py
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any, Dict, Tuple, Optional, Union, List
import gc
import scipy.stats
import numba

# 这个文件包含所有层级都可能用到的通用辅助函数

def get_param_value(param: Any, default: Any = None) -> Any:
    # 恢复 get_param_value 到其原始的、简单的逻辑
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

def is_limit_up(df_row: pd.Series, tolerance: float = 0.005) -> bool:
    """
    【V1.1 · 精确制导修正版】判断给定行（代表一天）的K线是否为涨停。
    - 核心修正: 使用带有 '_D' 后缀的列名，与数据管道的最终输出保持一致。
    """
    # 使用 'close_D'
    close_price = df_row.get('close_D')
    if pd.isna(close_price):
        return False
    # 优先使用 'up_limit_D'
    up_limit_price = df_row.get('up_limit_D')
    if pd.notna(up_limit_price) and up_limit_price > 0:
        return close_price >= up_limit_price * (1 - tolerance)
    # 回退方案使用 'pre_close_D'
    pre_close = df_row.get('pre_close_D')
    if pd.isna(pre_close):
        return False
    stock_code = df_row.get('stock_code', '')
    if stock_code.startswith('688') or stock_code.startswith('300'):
        limit_percent = 0.20
    elif stock_code.startswith('8'):
        limit_percent = 0.30
    else:
        limit_percent = 0.10
    stock_name = df_row.get('stock_name', '')
    if 'ST' in stock_name:
        limit_percent = 0.05
    estimated_up_limit = round(pre_close * (1 + limit_percent), 2)
    return close_price >= estimated_up_limit * (1 - tolerance)

def is_limit_down(df_row: pd.Series, tolerance: float = 0.005) -> bool:
    """
    【V1.1 · 精确制导修正版】判断给定行（代表一天）的K线是否为跌停。
    - 核心修正: 使用带有 '_D' 后缀的列名，与数据管道的最终输出保持一致。
    """
    # 使用 'close_D'
    close_price = df_row.get('close_D')
    if pd.isna(close_price):
        return False
    # 优先使用 'down_limit_D'
    down_limit_price = df_row.get('down_limit_D')
    if pd.notna(down_limit_price) and down_limit_price > 0:
        return close_price <= down_limit_price * (1 + tolerance)
    # 回退方案使用 'pre_close_D'
    pre_close = df_row.get('pre_close_D')
    if pd.isna(pre_close):
        return False
    stock_code = df_row.get('stock_code', '')
    if stock_code.startswith('688') or stock_code.startswith('300'):
        limit_percent = 0.20
    elif stock_code.startswith('8'):
        limit_percent = 0.30
    else:
        limit_percent = 0.10
    stock_name = df_row.get('stock_name', '')
    if 'ST' in stock_name:
        limit_percent = 0.05
    estimated_down_limit = round(pre_close * (1 - limit_percent), 2)
    return close_price <= estimated_down_limit * (1 + tolerance)

def ensure_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    【V1.2 · 健壮列类型处理版】
    遍历DataFrame的所有列，并尽可能地将其数据类型转换为占用内存更小的类型。
    - 核心修复: 更改遍历方式为 `df.items()`，确保每次迭代都能获取到单个 `Series` 对象及其标签，
                  从而避免 `df[col]` 返回 `DataFrame` 或 `df.dtypes[col]` 返回 `Series` 导致的 `ValueError`。
                  同时，将数值类型优化逻辑整合到此循环中。
    """
    start_mem = df.memory_usage().sum() / 1024**2
    converted_cols = []
    # 遍历 DataFrame 的每一列，col_label 是列名，series_data 是对应的 Series
    for col_label, series_data in df.items():
        # 1. 处理 object 类型，尝试转换为数值
        if series_data.dtype == 'object':
            # 检查 Series 中是否有 Decimal 对象，如果有则尝试转换为数值
            first_valid_item = series_data.dropna().iloc[0] if not series_data.dropna().empty else None
            if isinstance(first_valid_item, Decimal):
                df[col_label] = pd.to_numeric(series_data, errors='coerce')
                converted_cols.append(col_label)
        # 2. 优化非 object 类型的数值列的内存使用
        elif series_data.dtype != 'category' and 'datetime' not in str(series_data.dtype):
            c_min = series_data.min()
            c_max = series_data.max()
            if str(series_data.dtype)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col_label] = series_data.astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col_label] = series_data.astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col_label] = series_data.astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col_label] = series_data.astype(np.int64)
            else: # float types
                # 对于浮点数，优先使用float32，精度足够且内存减半
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col_label] = series_data.astype(np.float32)
                else:
                    df[col_label] = series_data.astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'    -> [内存优化] (ensure_numeric_types) DataFrame内存从 {start_mem:.2f} MB 优化至 {end_mem:.2f} MB (减少了 {(start_mem - end_mem) / start_mem * 100:.1f}%)')
    return df

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
        print(f'    -> [内存优化] (optimize_df_memory) DataFrame内存从 {start_mem:.2f} MB 优化至 {end_mem:.2f} MB (减少了 {(start_mem - end_mem) / start_mem * 100:.1f}%)')
    return df

def calculate_context_scores(df: pd.DataFrame, atomic_states: Dict, strategy_instance) -> Tuple[pd.Series, pd.Series]:
    """
    【V11.7 · 签名修复与参数直传版】计算全局的底部和顶部上下文分数
    - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名，避免参数错位和未定义参数的错误。
    """
    if isinstance(df, dict):
        df = df.get('df_indicators', pd.DataFrame())
    close_col, high_col, low_col = 'close_D', 'high_D', 'low_D'
    if close_col not in df.columns:
        print(f"      -> [calculate_context_scores] 警告: 输入的DataFrame缺少'{close_col}'列，无法计算。")
        empty_series = pd.Series(0.5, index=df.index if not df.empty else None, dtype=np.float32)
        return empty_series, empty_series
    p_synthesis = get_params_block(strategy_instance, 'ultimate_signal_synthesis_params', {})
    norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
    depth_threshold = get_param_value(p_synthesis.get('deep_bearish_threshold'), 0.05)
    ma55_lifeline = df.get('MA_55_D', df[close_col])
    is_deep_bearish_zone = (df[close_col] < ma55_lifeline * (1 - depth_threshold)).astype(float)
    gaia_params = get_param_value(p_synthesis.get('gaia_bedrock_params'), {})
    support_levels = get_param_value(gaia_params.get('support_levels'), [55, 89, 144, 233])
    ma_cols = [f'MA_{p}_D' for p in support_levels if f'MA_{p}_D' in df.columns]
    if ma_cols:
        ma_df = df[ma_cols]
        ma_df_below_price = ma_df.where(ma_df.le(df[close_col], axis=0))
        last_stand_line = ma_df_below_price.max(axis=1).ffill()
        atomic_states['LAST_STAND_LINE'] = last_stand_line
    ma55_slope = ma55_lifeline.diff(3).fillna(0)
    slope_moderator = (0.5 + 0.5 * np.tanh(ma55_slope * 100)).fillna(0.5)
    distance_from_ma55 = (df[close_col] - ma55_lifeline) / ma55_lifeline.replace(0, np.nan)
    lifeline_support_score_raw = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2).fillna(0.0)
    lifeline_support_score = lifeline_support_score_raw * slope_moderator
    price_pos_yearly = normalize_score(df[close_col], df.index, 250, ascending=True)
    absolute_value_zone_score = 1.0 - price_pos_yearly
    deep_bottom_context_score_values = np.maximum.reduce([
        lifeline_support_score.values,
        absolute_value_zone_score.values
    ])
    deep_bottom_context_score = pd.Series(deep_bottom_context_score_values, index=df.index, dtype=np.float32)
    rsi_w_col = 'RSI_13_W'
    rsi_w_oversold_score = normalize_score(df.get(rsi_w_col, pd.Series(50, index=df.index)), df.index, 52, ascending=False)
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
    p_meta = get_param_value(p_synthesis.get('meta_dynamics_context_params'), {})
    if get_param_value(p_meta.get('enabled'), False):
        long_ma_p = get_param_value(p_meta.get('long_ma_period'), 55)
        short_slope_p = get_param_value(p_meta.get('short_slope_period'), 5)
        bonus_factor = get_param_value(p_meta.get('bonus_factor'), 0.3)
        meta_dynamics_col = f'SLOPE_{short_slope_p}_EMA_{long_ma_p}_D'
        if meta_dynamics_col in df.columns:
            deceleration_score = normalize_score(df[meta_dynamics_col], df.index, norm_window, ascending=True)
            meta_dynamics_bonus = (deceleration_score * is_deep_bearish_zone * bonus_factor)
            bottom_context_score_raw = (bottom_context_score_raw + meta_dynamics_bonus).clip(0, 1)
    conventional_bottom_score = bottom_context_score_raw * is_deep_bearish_zone
    gaia_bedrock_support_score = _calculate_gaia_bedrock_support(df, gaia_params, atomic_states)
    p_fib_support = get_param_value(p_synthesis.get('fibonacci_support_params'), {})
    historical_low_support_score = _calculate_historical_low_support(df, p_fib_support)
    gaia_confirmation_score = atomic_states.get('SCORE_FOUNDATION_BOTTOM_CONFIRMED', pd.Series(0.0, index=df.index))
    fused_confirmation_score = np.maximum(gaia_confirmation_score, historical_low_support_score)
    atomic_states['SCORE_FOUNDATION_BOTTOM_CONFIRMED'] = fused_confirmation_score.astype(np.float32)
    structural_support_score = np.maximum(gaia_bedrock_support_score, historical_low_support_score).astype(np.float32)
    bottom_context_score = np.maximum(conventional_bottom_score, structural_support_score).astype(np.float32)
    ma55 = df.get('MA_55_D', df[close_col])
    rolling_high_55d = df[high_col].rolling(window=55, min_periods=21).max()
    wave_channel_height = (rolling_high_55d - ma55).replace(0, 1e-9)
    stretch_score = ((df[close_col] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
    ma_periods = [5, 13, 21, 55]
    short_ma_cols = [f'MA_{p}_D' for p in ma_periods[:-1]]
    long_ma_cols = [f'MA_{p}_D' for p in ma_periods[1:]]
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
    conventional_top_score = (stretch_score * misalignment_score * overheat_score)**(1/3)
    uranus_params = get_param_value(p_synthesis.get('uranus_ceiling_params'), {})
    uranus_ceiling_resistance_score = _calculate_uranus_ceiling_resistance(df, uranus_params)
    p_fib_resistance = get_param_value(p_synthesis.get('fibonacci_resistance_params'), {})
    historical_high_resistance_score = _calculate_historical_high_resistance(df, p_fib_resistance, uranus_params)
    structural_resistance_score = np.maximum(uranus_ceiling_resistance_score, historical_high_resistance_score).astype(np.float32)
    top_context_score = np.maximum(conventional_top_score, structural_resistance_score).astype(np.float32)
    return bottom_context_score, top_context_score

def get_robust_bipolar_normalized_score(series: pd.Series, target_index: pd.Index, window: int, sensitivity: float = 2.0, default_value: float = 0.0, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> pd.Series:
    """
    【V1.4 · 健壮类型处理版 & 调试增强版】双极归一化的高阶进化版 (力学罗盘)
    - 核心修复: 在函数入口处增加对 `series` 参数的健壮性检查和类型转换。
                  确保 `series` 始终是数值型的 `pandas.Series`，避免因类型不一致导致的 `ValueError`。
    - 调试增强: 增加了 `debug_info` 参数，用于在调试模式下输出详细信息。
    """
    is_debug_enabled, probe_ts, signal_name = debug_info if debug_info else (False, None, "Unknown")
    if is_debug_enabled and probe_ts:
        print(f"         [探针] {signal_name} - get_robust_bipolar_normalized_score 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
        if probe_ts in series.index:
            print(f"           - 原始值: {series.loc[probe_ts]:.4f}")
    if isinstance(series, pd.DataFrame):
        if len(series.columns) == 1:
            series = series.iloc[:, 0]
        else:
            if is_debug_enabled and probe_ts:
                print(f"           - 警告: 接收到多列DataFrame '{series.columns.tolist()}'，无法处理。返回默认值。")
            return pd.Series(default_value, index=target_index, dtype=np.float32)
    series = pd.to_numeric(series, errors='coerce').fillna(0)
    if series.empty:
        if is_debug_enabled and probe_ts:
            print(f"           - 警告: 原始Series为空，返回默认值。")
        return pd.Series(default_value, index=target_index, dtype=np.float32)
    # 直接调用优化后的 normalize_to_bipolar
    final_score = normalize_to_bipolar(series, target_index, window, sensitivity, default_value, debug_info)
    if is_debug_enabled and probe_ts:
        print(f"         [探针] {signal_name} - get_robust_bipolar_normalized_score 结果 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
    return final_score

def calculate_holographic_dynamics(df: pd.DataFrame, base_name: str, norm_window: int) -> Tuple[pd.Series, pd.Series]:
    """
    【V2.4 · 归一化函数调用修复版】全息动态计算引擎
    - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
    """
    default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
    slope_1 = df.get(f'SLOPE_1_{base_name}', default_series)
    slope_5 = df.get(f'SLOPE_5_{base_name}', default_series)
    slope_diff = slope_1 - slope_5
    velocity_accel_score = normalize_score(slope_diff, df.index, norm_window, ascending=True)
    velocity_decel_score = normalize_score(slope_diff, df.index, norm_window, ascending=False)
    accel_1 = df.get(f'ACCEL_1_{base_name}', default_series)
    accel_5 = df.get(f'ACCEL_5_{base_name}', default_series)
    accel_diff = accel_1 - accel_5
    jerk_accel_score = normalize_score(accel_diff, df.index, norm_window, ascending=True)
    jerk_decel_score = normalize_score(accel_diff, df.index, norm_window, ascending=False)
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
    【V5.9 · 终极净化版】终极信号中央合成引擎
    - 核心重构: 彻底移除了函数内部关于 `neutral_zone_threshold` 的定义和使用。
                  现在，单双极性转换完全依赖于已修复的、无阈值的 `bipolar_to_exclusive_unipolar` 函数，
                  从根本上解决了弱信号在最后合成阶段被错误归零的问题。
    """
    states = {}
    resonance_tf_weights = get_param_value(params.get('resonance_tf_weights'), {'short': 0.2, 'medium': 0.5, 'long': 0.3})
    reversal_tf_weights = get_param_value(params.get('reversal_tf_weights'), {'short': 0.6, 'medium': 0.3, 'long': 0.1})
    periods = get_param_value(params.get('periods'), [1, 5, 13, 21, 55])
    norm_window = get_param_value(params.get('norm_window'), 55)
    bottom_context_bonus_factor = get_param_value(params.get('bottom_context_bonus_factor'), 0.5)
    exponent = get_param_value(params.get('final_score_exponent'), 1.0)
    # 彻底移除了 neutral_zone_threshold 的获取
    atomic_states['strategy_instance_ref'] = df.strategy if hasattr(df, 'strategy') else {}
    bottom_context_score, top_context_score = calculate_context_scores(df, atomic_states)
    if 'strategy_instance_ref' in atomic_states:
        del atomic_states['strategy_instance_ref']
    recent_reversal_context = atomic_states.get('SCORE_CONTEXT_RECENT_REVERSAL', pd.Series(0.0, index=df.index))
    default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
    new_high_params = get_param_value(params.get('new_high_context_params'), {})
    new_high_context_score = _calculate_new_high_context(df, new_high_params)
    atomic_states['CONTEXT_NEW_HIGH_STRENGTH'] = new_high_context_score
    memory_retention_factor = 1.0 - new_high_context_score
    recent_reversal_context_modulated = recent_reversal_context * memory_retention_factor
    trend_confirmation_params = get_param_value(params.get('trend_confirmation_context_params'), {})
    trend_confirmation_context = calculate_trend_confirmation_context(df, trend_confirmation_params, norm_window)
    atomic_states['CONTEXT_TREND_CONFIRMED'] = trend_confirmation_context
    tactical_params = get_param_value(params.get('tactical_reversal_params'), {})
    macro_window = get_param_value(tactical_params.get('macro_trend_window'), 3)
    macro_trend_permit_context = trend_confirmation_context.rolling(window=macro_window, min_periods=1).mean()
    atomic_states['CONTEXT_MACRO_TREND_PERMIT'] = macro_trend_permit_context
    dynamic_reversal_params = get_param_value(params.get('dynamic_reversal_context_params'), {})
    dynamic_reversal_context = _calculate_dynamic_reversal_context(df, dynamic_reversal_params, norm_window)
    atomic_states['CONTEXT_DYNAMIC_REVERSAL'] = dynamic_reversal_context
    period_groups = {
        'short': [p for p in periods if p <= 5],
        'medium': [p for p in periods if 5 < p <= 21],
        'long': [p for p in periods if p > 21]
    }
    def fuse_bipolar_health(health_dict: Dict, weights: Dict) -> pd.Series:
        final_score = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_weight = sum(weights.values())
        if total_weight > 0:
            for tf_name, weight in weights.items():
                group_periods = period_groups.get(tf_name, [])
                group_scores = [health_dict.get(p, default_series) for p in group_periods]
                if group_scores:
                    avg_group_score = sum(group_scores) / len(group_scores)
                    final_score += avg_group_score * (weight / total_weight)
        return final_score.clip(-1, 1).astype(np.float32)
    bipolar_health = {}
    s_bull_dict = overall_health.get('s_bull', {})
    s_bear_dict = overall_health.get('s_bear', {})
    available_periods = s_bull_dict.keys() | s_bear_dict.keys()
    for p in available_periods:
        s_bull = s_bull_dict.get(p, default_series)
        s_bear = s_bear_dict.get(p, default_series)
        bipolar_health[p] = s_bull - s_bear
    final_bipolar_resonance = fuse_bipolar_health(bipolar_health, resonance_tf_weights)
    final_bipolar_reversal = fuse_bipolar_health(bipolar_health, reversal_tf_weights)
    # 调用无阈值的 bipolar_to_exclusive_unipolar 函数，确保信号的完整性
    final_bullish_resonance, final_bearish_resonance = bipolar_to_exclusive_unipolar(final_bipolar_resonance)
    final_bottom_reversal_trigger, final_top_reversal_trigger = bipolar_to_exclusive_unipolar(final_bipolar_reversal)
    raw_bottom_reversal_score = (final_bottom_reversal_trigger * (1 + recent_reversal_context_modulated * bottom_context_bonus_factor)).clip(0, 1)
    final_bottom_reversal_score = raw_bottom_reversal_score * bottom_context_score * (1 - trend_confirmation_context)
    final_top_reversal_score = (final_top_reversal_trigger * (1 + top_context_score * bottom_context_bonus_factor)).clip(0, 1)
    final_tactical_reversal_score = calculate_tactical_reversal_score(df, atomic_states, overall_health, tactical_params, norm_window)
    # 在这里，final_bullish_resonance 将直接用于生成最终信号，不再有任何中间的阈值过滤
    final_signal_map = {
        f'SCORE_{domain_prefix}_BULLISH_RESONANCE': (final_bullish_resonance ** exponent),
        f'SCORE_{domain_prefix}_BOTTOM_REVERSAL': (final_bottom_reversal_score ** exponent),
        f'SCORE_{domain_prefix}_TACTICAL_REVERSAL': (final_tactical_reversal_score ** exponent),
        f'SCORE_{domain_prefix}_BEARISH_RESONANCE': (final_bearish_resonance ** exponent),
        f'SCORE_{domain_prefix}_TOP_REVERSAL': (final_top_reversal_score ** exponent)
    }
    for signal_name, score in final_signal_map.items():
        states[signal_name] = score.astype(np.float32)
    return states

def _calculate_new_high_context(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V2.3 · 归一化函数调用修复版】多维新高上下文分数计算器
    - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    periods = get_param_value(params.get('periods'), [5, 13, 21, 55])
    period_weights = get_param_value(params.get('period_weights'), {})
    bias_thresholds = get_param_value(params.get('bias_thresholds'), {})
    fusion_weights = get_param_value(params.get('fusion_weights'), {})
    final_scores_np = np.zeros(len(df.index), dtype=np.float32)
    for p in periods:
        period_weight = period_weights.get(str(p), 0)
        if period_weight == 0:
            continue
        rolling_high = df['high_D'].rolling(window=p, min_periods=1).max().shift(1)
        is_new_high_score = (df['high_D'] > rolling_high).astype(np.float32)
        ma_slope_col = f'SLOPE_{p}_MA_{p}_D'
        if ma_slope_col not in df.columns: ma_slope_col = f'SLOPE_{p}_close_D'
        ma_slope = df.get(ma_slope_col, pd.Series(0, index=df.index))
        ma_slope_score = normalize_score(ma_slope, df.index, p*2, ascending=True)
        bias_period = 21 if p <= 21 else 55
        bias_col = f'BIAS_{bias_period}_D'
        bias_threshold = bias_thresholds.get(str(bias_period), 0.2)
        bias_value = df.get(bias_col, pd.Series(0, index=df.index)).abs()
        bias_health_score = (1 - (bias_value / bias_threshold)).clip(0, 1)
        period_new_high_score = (
            is_new_high_score * fusion_weights.get('new_high', 0.4) +
            ma_slope_score * fusion_weights.get('ma_slope', 0.3) +
            bias_health_score * fusion_weights.get('bias_health', 0.3)
        )
        final_scores_np += period_new_high_score.values * period_weight
    total_weight = sum(w for p, w in period_weights.items() if int(p) in periods)
    if total_weight == 0:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    final_score_values = final_scores_np / total_weight
    return pd.Series(final_score_values, index=df.index).clip(0, 1).astype(np.float32)

def calculate_trend_confirmation_context(df: pd.DataFrame, params: Dict, norm_window: int) -> pd.Series:
    """
    【V3.1 · 归一化函数调用修复版】趋势确认上下文计算器 (波塞冬三叉戟)
    - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    adx_threshold = get_param_value(params.get('adx_threshold'), 20)
    adx = df.get('ADX_14_D', pd.Series(0, index=df.index))
    is_trending = (adx > adx_threshold).astype(np.float32)
    strength_score = normalize_score(adx, df.index, norm_window, ascending=True) * is_trending
    pdi = df.get('PDI_14_D', pd.Series(0, index=df.index))
    ndi = df.get('NDI_14_D', pd.Series(0, index=df.index))
    direction_score = (pdi > ndi).astype(np.float32)
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
    【V1.0】战术反转信号计算器 (赫尔墨斯的飞翼鞋)
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
    【V1.2 · 归一化函数调用修复版】动态反转上下文计算器 (二阶求导引擎)
    - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    short_ma_period = get_param_value(params.get('short_ma_period'), 5)
    mid_ma_period = get_param_value(params.get('mid_ma_period'), 21)
    slope_period = get_param_value(params.get('slope_period'), 3)
    weights = get_param_value(params.get('fusion_weights'), {'distance_accel': 0.5, 'slope_accel': 0.5})
    short_ma_col = f'MA_{short_ma_period}_D'
    mid_ma_col = f'MA_{mid_ma_period}_D'
    short_ma_slope_col = f'SLOPE_{slope_period}_MA_{short_ma_period}_D'
    if not all(c in df.columns for c in [short_ma_col, mid_ma_col, short_ma_slope_col]):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    ma_distance = df[short_ma_col] - df[mid_ma_col]
    ma_distance_slope = ma_distance.diff(slope_period).fillna(0)
    distance_accel_score = normalize_score(ma_distance_slope, df.index, norm_window, ascending=True)
    short_ma_slope = df[short_ma_slope_col]
    short_ma_slope_accel = short_ma_slope.diff(slope_period).fillna(0)
    slope_accel_score = normalize_score(short_ma_slope_accel, df.index, norm_window, ascending=True)
    dynamic_reversal_score = (
        distance_accel_score * weights.get('distance_accel', 0.5) +
        slope_accel_score * weights.get('slope_accel', 0.5)
    )
    return dynamic_reversal_score.clip(0, 1).astype(np.float32)

def _calculate_gaia_bedrock_support(df: pd.DataFrame, params: Dict, atomic_states: Dict) -> pd.Series: # 增加 atomic_states 参数
    """
    【V23.11 · 冷却期向量化优化版 & 日期类型修复版 & 调试探针版】“盖亚基石”支撑分计算引擎
    - 核心修复: 修正了上下影线的计算逻辑，使用 np.maximum/minimum(open, close) 作为实体边界，
                  确保在阴阳线上计算的绝对准确性。
    - 性能优化: 将冷却期逻辑从显式 `for` 循环重构为向量化操作，显著提升效率。
    - 日期类型修复: 解决了 `df.index - filled_dates` 运算中 `Timestamp` 与 `float` 混淆的 `TypeError`，
                    确保 `filled_dates` 始终为 `datetime64[ns]` 类型。
    - Timedelta 访问修复: 修正了 `time_diff.days` 为 `time_diff.dt.days`，以正确访问 Timedelta Series 的天数属性。
    - .dt accessor 错误修复: 确保 `time_diff` 是 `Timedelta` Series，以便正确使用 `.dt` 访问器。
    - 强制类型转换移除: 移除了多余的 `astype('datetime64[ns]')`，避免与时区信息冲突。
    - 调试探针: 添加了打印语句，用于在错误发生时输出 `df.index.to_series()` 和 `filled_dates` 的详细信息。
    - 关键修复: 确保 `confirmation_dates` 初始化时与 `df.index` 具有相同的 `dtype` (包括时区)，避免类型冲突。
    - 显式处理NaT: 在日期减法前显式过滤掉 `NaT` 值，避免 `Timestamp` 与 `float` 运算错误。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    support_levels = get_param_value(params.get('support_levels'), [55, 89, 144, 233, 377])
    confirmation_window = get_param_value(params.get('confirmation_window'), 3)
    aegis_lookback_window = get_param_value(params.get('aegis_lookback_window'), 5)
    confirmation_cooldown_period = get_param_value(params.get('confirmation_cooldown_period'), 10)
    influence_zone_pct = get_param_value(params.get('influence_zone_pct'), 0.03)
    defense_base_score = get_param_value(params.get('defense_base_score'), 0.4)
    defense_yang_line_weight = get_param_value(params.get('defense_yang_line_weight'), 0.1)
    defense_dominance_weight = get_param_value(params.get('defense_dominance_weight'), 0.2)
    defense_volume_weight = get_param_value(params.get('defense_volume_weight'), 0.3)
    confirmation_score = get_param_value(params.get('confirmation_score'), 0.8)
    aegis_quality_bonus_factor = get_param_value(params.get('aegis_quality_bonus_factor'), 0.25)
    cooldown_reset_volume_ma_period = get_param_value(params.get('cooldown_reset_volume_ma_period'), 55)
    close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
    ares_vol_ma_col = 'VOL_MA_5_D'
    cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
    ma_cols = [f'MA_{p}_D' for p in support_levels if f'MA_{p}_D' in df.columns]
    required_cols = [close_col, open_col, low_col, high_col, vol_col, ares_vol_ma_col, cooldown_vol_ma_col] + ma_cols
    if not all(col in df.columns for col in required_cols):
        print(f"盖亚基石模块缺少必要列，将返回0分。缺失列: {[c for c in required_cols if c not in df.columns]}")
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    ma_df = df[ma_cols]
    ma_df_below_price = ma_df.where(ma_df.le(df[close_col], axis=0))
    acting_lifeline = ma_df_below_price.max(axis=1).ffill()
    valid_indices = acting_lifeline.dropna().index
    if valid_indices.empty:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    is_in_influence_zone = pd.Series(False, index=df.index)
    upper_bound = acting_lifeline[valid_indices] * (1 + influence_zone_pct)
    is_in_influence_zone.loc[valid_indices] = df.loc[valid_indices, close_col].between(acting_lifeline[valid_indices], upper_bound)
    defense_quality_score = pd.Series(0.0, index=df.index, dtype=np.float32)
    base_defense_condition = (df[low_col] < acting_lifeline) & is_in_influence_zone & (df[close_col] > df[low_col])
    defense_quality_score.loc[base_defense_condition] = defense_base_score
    is_yang_line = df[close_col] > df[open_col]
    upper_shadow = df[high_col] - np.maximum(df[open_col], df[close_col])
    lower_shadow = np.minimum(df[open_col], df[close_col]) - df[low_col]
    has_dominance = lower_shadow > upper_shadow
    has_volume_spike = df[vol_col] > df[ares_vol_ma_col]
    defense_quality_score.loc[base_defense_condition & is_yang_line] += defense_yang_line_weight
    defense_quality_score.loc[base_defense_condition & has_dominance] += defense_dominance_weight
    defense_quality_score.loc[base_defense_condition & has_dominance & has_volume_spike] += defense_volume_weight
    is_cassandra_warning = (upper_shadow > lower_shadow) & has_volume_spike
    defense_quality_score.loc[is_in_influence_zone & is_cassandra_warning] = 0.0
    defense_quality_score = defense_quality_score.clip(0, 1.0)
    max_recent_defense_quality = defense_quality_score.rolling(window=aegis_lookback_window, min_periods=1).max()
    is_standing_firm_in_zone = (df[close_col] > acting_lifeline) & is_in_influence_zone
    is_confirmed_base = is_standing_firm_in_zone.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
    is_cooldown_reset_signal = (upper_shadow > lower_shadow) & (df[vol_col] > df[cooldown_vol_ma_col])
    # --- 冷却期逻辑向量化优化 ---
    # 1. 计算确认事件的原始分数
    raw_confirmation_scores = pd.Series(0.0, index=df.index, dtype=np.float32)
    confirmed_base_with_bonus_mask = is_confirmed_base & (max_recent_defense_quality > 0)
    raw_confirmation_scores.loc[confirmed_base_with_bonus_mask] = (confirmation_score + max_recent_defense_quality.loc[confirmed_base_with_bonus_mask] * aegis_quality_bonus_factor).clip(0, 1.0)
    confirmed_base_no_bonus_mask = is_confirmed_base & (max_recent_defense_quality <= 0)
    raw_confirmation_scores.loc[confirmed_base_no_bonus_mask] = confirmation_score
    # 2. 标记冷却期重置点
    #    创建一个组ID，每次遇到重置信号时递增，这样ffill就不会跨越重置点
    group_ids = is_cooldown_reset_signal.astype(int).cumsum()
    # 3. 跟踪每个确认事件的日期
    # 确保 confirmation_dates 的 dtype 与 df.index 保持一致，包括时区信息
    confirmation_dates = pd.Series(pd.NaT, index=df.index, dtype=df.index.dtype) 
    confirmation_dates.loc[is_confirmed_base] = df.index[is_confirmed_base]
    # 4. 在每个组内，前向填充确认分数和确认日期
    filled_scores = raw_confirmation_scores.groupby(group_ids).ffill()
    filled_dates = confirmation_dates.groupby(group_ids).ffill()
    # 5. 检查当前日期是否在冷却期内
    # 关键修复：显式处理 NaT，避免 Pandas 内部将 NaT 转换为 float 导致 TypeError
    time_diff = pd.Series(pd.NaT, index=df.index, dtype='timedelta64[ns]') # 初始化为 Timedelta Series
    # 找出 filled_dates 中非 NaT 的位置
    non_nat_filled_dates_mask = filled_dates.notna()
    if non_nat_filled_dates_mask.any():
        # 只对非 NaT 的部分进行减法运算，确保 Timestamp - Timestamp
        time_diff.loc[non_nat_filled_dates_mask] = df.index.to_series().loc[non_nat_filled_dates_mask] - filled_dates.loc[non_nat_filled_dates_mask]
    # 过滤掉 time_diff 中的 NaT 值，只对有效日期进行计算
    valid_time_diff_mask = time_diff.notna()
    is_within_cooldown_period = pd.Series(False, index=df.index)
    if valid_time_diff_mask.any():
        is_within_cooldown_period.loc[valid_time_diff_mask] = time_diff.dt.days.loc[valid_time_diff_mask] < confirmation_cooldown_period
    # 6. 最终的确认分数：只有在冷却期内且有填充分数的地方才有效
    confirmation_score_series = filled_scores.where(is_within_cooldown_period, 0.0).fillna(0.0)
    # --- 冷却期逻辑向量化优化结束 ---
    # 将确认分数作为一个独立的信号存入 atomic_states
    atomic_states['SCORE_FOUNDATION_BOTTOM_CONFIRMED'] = confirmation_score_series.astype(np.float32)
    gaia_score = np.maximum(defense_quality_score, confirmation_score_series)
    return gaia_score.astype(np.float32)

def _calculate_uranus_ceiling_resistance(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V3.6 · 冷却期向量化优化版 & 日期类型修复版 & 调试探针版】“乌拉诺斯穹顶”阻力分计算引擎
    - 核心升级: 不再包含拒绝质量评估逻辑，而是调用通用的 _calculate_rejection_quality_score 函数。
    - 性能优化: 将冷却期逻辑从显式 `for` 循环重构为向量化操作，显著提升效率。
    - 强制类型转换移除: 移除了多余的 `astype('datetime64[ns]')`，避免与时区信息冲突。
    - 调试探针: 添加了打印语句，用于在错误发生时输出 `df.index.to_series()` 和 `filled_dates` 的详细信息。
    - 关键修复: 确保 `confirmation_dates` 初始化时与 `df.index` 具有相同的 `dtype` (包括时区)，避免类型冲突。
    - 显式处理NaT: 在日期减法前显式过滤掉 `NaT` 值，避免 `Timestamp` 与 `float` 运算错误。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    resistance_levels = get_param_value(params.get('resistance_levels'), [55, 89, 144, 233, 377])
    confirmation_window = get_param_value(params.get('confirmation_window'), 3)
    confirmation_cooldown_period = get_param_value(params.get('confirmation_cooldown_period'), 10)
    confirmation_score = get_param_value(params.get('confirmation_score'), 0.8)
    rejection_quality_bonus_factor = get_param_value(params.get('rejection_quality_bonus_factor'), 0.25)
    cooldown_reset_volume_ma_period = get_param_value(params.get('cooldown_reset_volume_ma_period'), 55)
    rejection_lookback_window = get_param_value(params.get('rejection_lookback_window'), 5)
    close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
    cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
    ma_cols = [f'MA_{p}_D' for p in resistance_levels if f'MA_{p}_D' in df.columns]
    if not all(col in df.columns for col in [close_col, open_col, low_col, high_col, vol_col, cooldown_vol_ma_col] + ma_cols):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    # 1. 寻找代理天花板 (acting_ceiling)
    ma_df = df[ma_cols]
    ma_df_above_price = ma_df.where(ma_df.ge(df[close_col], axis=0))
    acting_ceiling = ma_df_above_price.min(axis=1).ffill()
    # 2. 调用通用函数计算拒绝质量分
    rejection_quality_score = _calculate_rejection_quality_score(df, params, acting_ceiling)
    # 3. 计算确认压制分
    max_recent_rejection_quality = rejection_quality_score.rolling(window=rejection_lookback_window, min_periods=1).max()
    is_in_influence_zone = pd.Series(False, index=df.index)
    valid_indices = acting_ceiling.dropna().index
    if not valid_indices.empty:
        influence_zone_pct = get_param_value(params.get('influence_zone_pct'), 0.03)
        lower_bound = acting_ceiling[valid_indices] * (1 - influence_zone_pct)
        is_in_influence_zone.loc[valid_indices] = df.loc[valid_indices, close_col].between(lower_bound, acting_ceiling[valid_indices])
    is_failing_to_break = (df[close_col] < acting_ceiling) & is_in_influence_zone
    is_confirmed_rejection = is_failing_to_break.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
    upper_shadow = df[high_col] - np.maximum(df[open_col], df[close_col])
    lower_shadow = np.minimum(df[open_col], df[close_col]) - df[low_col]
    is_cooldown_reset_signal = (lower_shadow > upper_shadow) & (df[vol_col] > df[cooldown_vol_ma_col])
    # --- 冷却期逻辑向量化优化 ---
    # 1. 计算确认事件的原始分数
    raw_confirmation_scores = pd.Series(0.0, index=df.index, dtype=np.float32)
    confirmed_rejection_with_bonus_mask = is_confirmed_rejection & (max_recent_rejection_quality > 0)
    raw_confirmation_scores.loc[confirmed_rejection_with_bonus_mask] = (confirmation_score + max_recent_rejection_quality.loc[confirmed_rejection_with_bonus_mask] * rejection_quality_bonus_factor).clip(0, 1.0)
    confirmed_rejection_no_bonus_mask = is_confirmed_rejection & (max_recent_rejection_quality <= 0)
    raw_confirmation_scores.loc[confirmed_rejection_no_bonus_mask] = confirmation_score
    # 2. 标记冷却期重置点
    group_ids = is_cooldown_reset_signal.astype(int).cumsum()
    # 3. 跟踪每个确认事件的日期
    # 确保 confirmation_dates 的 dtype 与 df.index 保持一致，包括时区信息
    confirmation_dates = pd.Series(pd.NaT, index=df.index, dtype=df.index.dtype) 
    confirmation_dates.loc[is_confirmed_rejection] = df.index[is_confirmed_rejection]
    # 4. 在每个组内，前向填充确认分数和确认日期
    filled_scores = raw_confirmation_scores.groupby(group_ids).ffill()
    filled_dates = confirmation_dates.groupby(group_ids).ffill()
    # --- 调试探针开始 ---
    print(f"    -> [DEBUG: _calculate_uranus_ceiling_resistance] df.index.to_series().dtype: {df.index.to_series().dtype}")
    print(f"    -> [DEBUG: _calculate_uranus_ceiling_resistance] filled_dates.dtype (after ffill): {filled_dates.dtype}")
    print(f"    -> [DEBUG: _calculate_calculate_uranus_ceiling_resistance] df.index.to_series() head:\n{df.index.to_series().head()}")
    print(f"    -> [DEBUG: _calculate_uranus_ceiling_resistance] filled_dates head:\n{filled_dates.head()}")
    print(f"    -> [DEBUG: _calculate_uranus_ceiling_resistance] filled_dates contains NaT: {filled_dates.isna().any()}")
    # --- 调试探针结束 ---
    # 5. 检查当前日期是否在冷却期内
    # 关键修复：显式处理 NaT，避免 Pandas 内部将 NaT 转换为 float 导致 TypeError
    time_diff = pd.Series(pd.NaT, index=df.index, dtype='timedelta64[ns]') # 初始化为 Timedelta Series
    # 找出 filled_dates 中非 NaT 的位置
    non_nat_filled_dates_mask = filled_dates.notna()
    if non_nat_filled_dates_mask.any():
        # 只对非 NaT 的部分进行减法运算，确保 Timestamp - Timestamp
        time_diff.loc[non_nat_filled_dates_mask] = df.index.to_series().loc[non_nat_filled_dates_mask] - filled_dates.loc[non_nat_filled_dates_mask]
    # 过滤掉 time_diff 中的 NaT 值，只对有效日期进行计算
    valid_time_diff_mask = time_diff.notna()
    is_within_cooldown_period = pd.Series(False, index=df.index)
    if valid_time_diff_mask.any():
        is_within_cooldown_period.loc[valid_time_diff_mask] = time_diff.dt.days.loc[valid_time_diff_mask] < confirmation_cooldown_period
    # 6. 最终的确认分数：只有在冷却期内且有填充分数的地方才有效
    confirmation_score_series = filled_scores.where(is_within_cooldown_period, 0.0).fillna(0.0)
    # --- 冷却期逻辑向量化优化结束 ---
    # 4. 最终融合
    uranus_score = np.maximum(rejection_quality_score, confirmation_score_series)
    return uranus_score.astype(np.float32)

def _calculate_historical_low_support(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V1.2 · MA基准版】“历史低点”支撑分计算引擎
    - 将触发掩码的均线基准从 EMA 替换为 MA。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    close_col, low_col = 'close_D', 'low_D'
    ma55_lifeline = df.get('MA_55_D', df[close_col])
    fib_periods = get_param_value(params.get('periods'), [34, 55, 89, 144, 233])
    tolerance_pct = get_param_value(params.get('tolerance_pct'), 0.01)
    level_scores = get_param_value(params.get('level_scores'), {})
    dynamic_support_score = pd.Series(0.0, index=df.index, dtype=np.float32)
    trigger_mask = df[close_col] < ma55_lifeline
    for period in fib_periods:
        period_str = str(period)
        if period_str not in level_scores: continue
        historical_low = df[close_col].rolling(window=period, min_periods=max(1, int(period*0.8))).min().shift(1)
        support_line_upper = historical_low * (1 + tolerance_pct)
        support_line_lower = historical_low * (1 - tolerance_pct)
        is_defended = (df[low_col] <= support_line_upper) & (df[close_col] >= support_line_lower)
        score_mask = trigger_mask & is_defended
        dynamic_support_score.loc[score_mask] = np.maximum(dynamic_support_score.loc[score_mask], level_scores[period_str])
    return dynamic_support_score.astype(np.float32)

def _calculate_historical_high_resistance(df: pd.DataFrame, params: Dict, quality_params: Dict) -> pd.Series:
    """
    【V2.0 · 神盾协议版】“历史高点”阻力分计算引擎
    - 核心升级: 调用通用的 _calculate_rejection_quality_score 函数来评估拒绝质量。
    - 融合逻辑: 最终得分 = 拒绝质量分 * 阻力位战略重要性分。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    close_col, high_col = 'close_D', 'high_D'
    ma55_lifeline = df.get('MA_55_D', df[close_col])
    fib_periods = get_param_value(params.get('periods'), [34, 55, 89, 144, 233])
    level_scores = get_param_value(params.get('level_scores'), {})
    dynamic_resistance_score = pd.Series(0.0, index=df.index, dtype=np.float32)
    trigger_mask = df[close_col] > ma55_lifeline
    for period in fib_periods:
        period_str = str(period)
        if period_str not in level_scores: continue
        historical_high = df[high_col].rolling(window=period, min_periods=max(1, int(period*0.8))).max().shift(1)
        # 调用通用函数评估拒绝质量
        rejection_quality = _calculate_rejection_quality_score(df, quality_params, historical_high)
        # 融合逻辑：战术质量 * 战略重要性
        strategic_importance = level_scores[period_str]
        period_score = rejection_quality * strategic_importance
        # 应用触发掩码，并更新最终分数
        final_period_score = period_score.where(trigger_mask, 0)
        dynamic_resistance_score = np.maximum(dynamic_resistance_score, final_period_score)
    return dynamic_resistance_score.astype(np.float32)

def _calculate_uranus_ceiling_resistance(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V3.7 · 冷却期向量化优化版 & 日期类型修复版 & 调试探针版】“乌拉诺斯穹顶”阻力分计算引擎
    - 核心升级: 不再包含拒绝质量评估逻辑，而是调用通用的 _calculate_rejection_quality_score 函数。
    - 性能优化: 将冷却期逻辑从显式 `for` 循环重构为向量化操作，显著提升效率。
    - 强制类型转换移除: 移除了多余的 `astype('datetime64[ns]')`，避免与时区信息冲突。
    - 调试探针: 添加了打印语句，用于在错误发生时输出 `df.index.to_series()` 和 `filled_dates` 的详细信息。
    - 关键修复: 确保 `confirmation_dates` 初始化时与 `df.index` 具有相同的 `dtype` (包括时区)，避免类型冲突。
    - 显式处理NaT: 在日期减法前显式过滤掉 `NaT` 值，避免 `Timestamp` 与 `float` 运算错误。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    # ... [获取参数的代码保持不变] ...
    resistance_levels = get_param_value(params.get('resistance_levels'), [55, 89, 144, 233, 377])
    confirmation_window = get_param_value(params.get('confirmation_window'), 3)
    confirmation_cooldown_period = get_param_value(params.get('confirmation_cooldown_period'), 10)
    confirmation_score = get_param_value(params.get('confirmation_score'), 0.8)
    rejection_quality_bonus_factor = get_param_value(params.get('rejection_quality_bonus_factor'), 0.25)
    cooldown_reset_volume_ma_period = get_param_value(params.get('cooldown_reset_volume_ma_period'), 55)
    rejection_lookback_window = get_param_value(params.get('rejection_lookback_window'), 5)
    close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
    cooldown_vol_ma_col = f'VOL_MA_{cooldown_reset_volume_ma_period}_D'
    ma_cols = [f'MA_{p}_D' for p in resistance_levels if f'MA_{p}_D' in df.columns]
    if not all(col in df.columns for col in [close_col, open_col, low_col, high_col, vol_col, cooldown_vol_ma_col] + ma_cols):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    # 1. 寻找代理天花板 (acting_ceiling)
    ma_df = df[ma_cols]
    ma_df_above_price = ma_df.where(ma_df.ge(df[close_col], axis=0))
    acting_ceiling = ma_df_above_price.min(axis=1).ffill()
    # 2. 调用通用函数计算拒绝质量分
    rejection_quality_score = _calculate_rejection_quality_score(df, params, acting_ceiling)
    # 3. 计算确认压制分
    max_recent_rejection_quality = rejection_quality_score.rolling(window=rejection_lookback_window, min_periods=1).max()
    is_in_influence_zone = pd.Series(False, index=df.index)
    valid_indices = acting_ceiling.dropna().index
    if not valid_indices.empty:
        influence_zone_pct = get_param_value(params.get('influence_zone_pct'), 0.03)
        lower_bound = acting_ceiling[valid_indices] * (1 - influence_zone_pct)
        is_in_influence_zone.loc[valid_indices] = df.loc[valid_indices, close_col].between(lower_bound, acting_ceiling[valid_indices])
    is_failing_to_break = (df[close_col] < acting_ceiling) & is_in_influence_zone
    is_confirmed_rejection = is_failing_to_break.rolling(window=confirmation_window, min_periods=confirmation_window).sum() >= confirmation_window
    upper_shadow = df[high_col] - np.maximum(df[open_col], df[close_col])
    lower_shadow = np.minimum(df[open_col], df[close_col]) - df[low_col]
    is_cooldown_reset_signal = (lower_shadow > upper_shadow) & (df[vol_col] > df[cooldown_vol_ma_col])
    # --- 冷却期逻辑向量化优化 ---
    # 1. 计算确认事件的原始分数
    raw_confirmation_scores = pd.Series(0.0, index=df.index, dtype=np.float32)
    confirmed_rejection_with_bonus_mask = is_confirmed_rejection & (max_recent_rejection_quality > 0)
    raw_confirmation_scores.loc[confirmed_rejection_with_bonus_mask] = (confirmation_score + max_recent_rejection_quality.loc[confirmed_rejection_with_bonus_mask] * rejection_quality_bonus_factor).clip(0, 1.0)
    confirmed_rejection_no_bonus_mask = is_confirmed_rejection & (max_recent_rejection_quality <= 0)
    raw_confirmation_scores.loc[confirmed_rejection_no_bonus_mask] = confirmation_score
    # 2. 标记冷却期重置点
    group_ids = is_cooldown_reset_signal.astype(int).cumsum()
    # 3. 跟踪每个确认事件的日期
    # 确保 confirmation_dates 的 dtype 与 df.index 保持一致，包括时区信息
    confirmation_dates = pd.Series(pd.NaT, index=df.index, dtype=df.index.dtype) 
    confirmation_dates.loc[is_confirmed_rejection] = df.index[is_confirmed_rejection]
    # 4. 在每个组内，前向填充确认分数和确认日期
    filled_scores = raw_confirmation_scores.groupby(group_ids).ffill()
    filled_dates = confirmation_dates.groupby(group_ids).ffill()
    # 5. 检查当前日期是否在冷却期内
    # 关键修复：显式处理 NaT，避免 Pandas 内部将 NaT 转换为 float 导致 TypeError
    time_diff = pd.Series(pd.NaT, index=df.index, dtype='timedelta64[ns]') # 初始化为 Timedelta Series
    # 找出 filled_dates 中非 NaT 的位置
    non_nat_filled_dates_mask = filled_dates.notna()
    if non_nat_filled_dates_mask.any():
        # 只对非 NaT 的部分进行减法运算，确保 Timestamp - Timestamp
        time_diff.loc[non_nat_filled_dates_mask] = df.index.to_series().loc[non_nat_filled_dates_mask] - filled_dates.loc[non_nat_filled_dates_mask]
    # 过滤掉 time_diff 中的 NaT 值，只对有效日期进行计算
    valid_time_diff_mask = time_diff.notna()
    is_within_cooldown_period = pd.Series(False, index=df.index)
    if valid_time_diff_mask.any():
        is_within_cooldown_period.loc[valid_time_diff_mask] = time_diff.dt.days.loc[valid_time_diff_mask] < confirmation_cooldown_period
    # 6. 最终的确认分数：只有在冷却期内且有填充分数的地方才有效
    confirmation_score_series = filled_scores.where(is_within_cooldown_period, 0.0).fillna(0.0)
    # --- 冷却期逻辑向量化优化结束 ---
    # 4. 最终融合
    uranus_score = np.maximum(rejection_quality_score, confirmation_score_series)
    return uranus_score.astype(np.float32)

def _calculate_rejection_quality_score(df: pd.DataFrame, params: Dict, resistance_line: pd.Series) -> pd.Series:
    """
    【V1.3 · 归一化函数调用修复版】通用拒绝质量评估引擎 (阿波罗之箭)
    - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
    """
    influence_zone_pct = get_param_value(params.get('influence_zone_pct'), 0.03)
    rejection_base_score = get_param_value(params.get('rejection_base_score'), 0.4)
    rejection_yin_line_weight = get_param_value(params.get('rejection_yin_line_weight'), 0.1)
    rejection_dominance_weight = get_param_value(params.get('rejection_dominance_weight'), 0.2)
    rejection_volume_weight = get_param_value(params.get('rejection_volume_weight'), 0.3)
    min_shadow_ratio = get_param_value(params.get('min_shadow_ratio'), 0.15)
    # 修正参数名，从 'icarus_fall_bonus' 改为 'icarus_fall_base_score' 以匹配配置文件
    icarus_fall_bonus = get_param_value(params.get('icarus_fall_base_score'), 0.5)
    cooldown_reset_volume_ma_period = get_param_value(params.get('cooldown_reset_volume_ma_period'), 55)
    close_col, open_col, low_col, high_col, vol_col = 'close_D', 'open_D', 'low_D', 'high_D', 'volume_D'
    ares_vol_ma_col = 'VOL_MA_5_D'
    required_cols = [close_col, open_col, low_col, high_col, vol_col, ares_vol_ma_col, 'up_limit_D']
    if not all(col in df.columns for col in required_cols):
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    valid_indices = resistance_line.dropna().index
    if valid_indices.empty:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
    is_in_influence_zone = pd.Series(False, index=df.index)
    lower_bound = resistance_line[valid_indices] * (1 - influence_zone_pct)
    is_in_influence_zone.loc[valid_indices] = df.loc[valid_indices, close_col].between(lower_bound, resistance_line[valid_indices])
    base_rejection_condition = (df[high_col] > resistance_line) & is_in_influence_zone & (df[close_col] < df[high_col])
    rejection_quality_score = pd.Series(0.0, index=df.index, dtype=np.float32)
    rejection_quality_score.loc[base_rejection_condition] = rejection_base_score
    is_yin_line = df[close_col] < df[open_col]
    upper_shadow = df[high_col] - np.maximum(df[open_col], df[close_col])
    lower_shadow = np.minimum(df[open_col], df[close_col]) - df[low_col]
    kline_range = (df[high_col] - df[low_col]).replace(0, np.nan)
    upper_shadow_ratio = upper_shadow / kline_range
    is_upper_shadow_significant = upper_shadow_ratio > min_shadow_ratio
    has_dominance = (upper_shadow > lower_shadow) & is_upper_shadow_significant
    has_volume_spike = df[vol_col] > df[ares_vol_ma_col]
    rejection_quality_score.loc[base_rejection_condition & is_yin_line] += rejection_yin_line_weight
    rejection_quality_score.loc[base_rejection_condition & has_dominance] += rejection_dominance_weight
    volume_ratio = df[vol_col] / df[ares_vol_ma_col].replace(0, np.nan)
    proportional_volume_score = normalize_score(volume_ratio, df.index, cooldown_reset_volume_ma_period, ascending=True)
    dynamic_volume_contribution = rejection_volume_weight * proportional_volume_score
    volume_mask = base_rejection_condition & has_dominance & has_volume_spike
    rejection_quality_score.loc[volume_mask] += dynamic_volume_contribution.loc[volume_mask]
    limit_up_price = df['up_limit_D']
    is_icarus_fall = (df[high_col] >= limit_up_price * 0.995) & (df[close_col] < df[high_col] * 0.98)
    rejection_quality_score.loc[is_icarus_fall] += icarus_fall_bonus
    is_apollo_absorption = (lower_shadow > upper_shadow) & has_volume_spike
    rejection_quality_score.loc[is_in_influence_zone & is_apollo_absorption] = 0.0
    return rejection_quality_score.clip(0, 1.0)

def bipolar_to_exclusive_unipolar(bipolar_score: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    【V3.1 · 类型安全版】将双极性分数转换为互斥的单极性分数。
    - 核心升级: 在返回前，将结果明确转换为 np.float32 类型，确保数据流中的类型一致性。
    """
    s_bull = bipolar_score.clip(lower=0)
    s_bear = bipolar_score.clip(upper=0).abs()
    return s_bull.astype(np.float32), s_bear.astype(np.float32)

def _parse_tf_weights(tf_weights: dict) -> Tuple[List[int], List[Tuple[int, float]]]:
    valid_windows_weights = []
    windows_list = []
    for window_str, weight in tf_weights.items():
        try:
            window = int(window_str)
        except ValueError:
            print(f"警告: 无法将MTF权重配置中的周期 '{window_str}' 转换为整数。跳过此项。")
            continue
        if window > 0 and weight > 0:
            valid_windows_weights.append((window, weight))
            windows_list.append(window)
    return windows_list, valid_windows_weights

def get_adaptive_mtf_normalized_score(series: pd.Series, target_index: pd.Index, tf_weights: dict, ascending: bool = True, default_value: float = 0.0, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None, _parsed_tf_data: Optional[Tuple[List[int], List[Tuple[int, float]]]] = None, force_zero_if_original_zero: bool = True) -> pd.Series:
    is_debug_enabled, probe_ts, signal_name = debug_info if debug_info else (False, None, "Unknown")
    if is_debug_enabled and probe_ts:
        print(f"       [探针] {signal_name} - get_adaptive_mtf_normalized_score 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
        if probe_ts in series.index:
            print(f"         - 原始值: {series.loc[probe_ts]:.4f}")
    if not isinstance(series, pd.Series) or series.empty:
        if is_debug_enabled and probe_ts:
            print(f"         - 警告: 原始Series为空，返回0。")
        return pd.Series(default_value, index=target_index)
    if _parsed_tf_data is None:
        windows_list, valid_windows_weights = _parse_tf_weights(tf_weights)
    else:
        windows_list, valid_windows_weights = _parsed_tf_data
    if not valid_windows_weights:
        if is_debug_enabled and probe_ts:
            print(f"         - 警告: 没有有效的窗口权重，返回0。")
        return pd.Series(0.0, index=target_index)
    # 调用优化后的 normalize_score，一次性获取所有窗口的归一化分数
    normalized_scores_df = normalize_score(series, target_index, windows_list, ascending, default_value, debug_info, force_zero_if_original_zero=force_zero_if_original_zero)
    final_scores = pd.Series(0.0, index=target_index, dtype=np.float32)
    total_weight = 0.0
    for window, weight in valid_windows_weights:
        if window in normalized_scores_df.columns:
            final_scores += normalized_scores_df[window] * weight
            total_weight += weight
    if total_weight > 0:
        final_scores /= total_weight
    else:
        final_scores = pd.Series(0.0, index=target_index)
    final_score = final_scores.astype(np.float32)
    if is_debug_enabled and probe_ts:
        print(f"       [探针] {signal_name} - get_adaptive_mtf_normalized_score 结果 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
    return final_score

def get_adaptive_mtf_normalized_bipolar_score(series: pd.Series, target_index: pd.Index, tf_weights: dict, sensitivity: float = 1.0, default_value: float = 0.0, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None, _parsed_tf_data: Optional[Tuple[List[int], List[Tuple[int, float]]]] = None) -> pd.Series:
    is_debug_enabled, probe_ts, signal_name = debug_info if debug_info else (False, None, "Unknown")
    if is_debug_enabled and probe_ts:
        print(f"       [探针] {signal_name} - get_adaptive_mtf_normalized_bipolar_score 启动 @ {probe_ts.strftime('%Y-%m-%d')}")
        if probe_ts in series.index:
            print(f"         - 原始值: {series.loc[probe_ts]:.4f}")
    if not isinstance(series, pd.Series) or series.empty:
        if is_debug_enabled and probe_ts:
            print(f"         - 警告: 原始Series为空，返回0。")
        return pd.Series(default_value, index=target_index)
    if _parsed_tf_data is None:
        windows_list, valid_windows_weights = _parse_tf_weights(tf_weights)
    else:
        windows_list, valid_windows_weights = _parsed_tf_data
    if not valid_windows_weights:
        if is_debug_enabled and probe_ts:
            print(f"         - 警告: 没有有效的窗口权重，返回0。")
        return pd.Series(default_value, index=target_index)
    # 调用优化后的 normalize_to_bipolar，一次性获取所有窗口的归一化分数
    normalized_scores_df = normalize_to_bipolar(series, target_index, windows_list, sensitivity, default_value, debug_info)
    final_scores = pd.Series(0.0, index=target_index, dtype=np.float32)
    total_weight = 0.0
    for window, weight in valid_windows_weights:
        if window in normalized_scores_df.columns:
            final_scores += normalized_scores_df[window] * weight
            total_weight += weight
    if total_weight > 0:
        final_scores /= total_weight
    else:
        final_scores = pd.Series(0.0, index=target_index)
    final_score = final_scores.astype(np.float32)
    if is_debug_enabled and probe_ts:
        print(f"       [探针] {signal_name} - get_adaptive_mtf_normalized_bipolar_score 结果 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
    return final_score

def normalize_score(series: pd.Series, target_index: pd.Index, windows: Union[int, List[int]], ascending: bool = True, default_value: float = 0.0, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None, force_zero_if_original_zero: bool = True) -> Union[pd.Series, pd.DataFrame]:
    is_debug_enabled, probe_ts, signal_name = debug_info if debug_info else (False, None, "Unknown")
    if not isinstance(series, pd.Series) or series.empty:
        if isinstance(windows, int):
            return pd.Series(default_value, index=target_index, dtype=np.float32)
        else:
            return pd.DataFrame({w: default_value for w in windows}, index=target_index, dtype=np.float32)
    series_aligned = series.reindex(target_index)
    is_original_zero = (series_aligned.abs() < 1e-9)
    padded_series = series_aligned.ffill().bfill()
    padded_series = pd.to_numeric(padded_series, errors='coerce').fillna(0)
    if isinstance(windows, int):
        windows_list = [windows]
    else:
        windows_list = windows
    series_values = padded_series.values.astype(np.float32)
    is_original_zero_values = is_original_zero.values
    results_array = _numba_normalize_score_multi_window_core(
        series_values,
        windows_list,
        ascending,
        default_value,
        is_original_zero_values
    )
    results_df = pd.DataFrame(results_array, index=target_index, columns=windows_list, dtype=np.float32)
    if is_debug_enabled and probe_ts:
        for window in windows_list:
            if probe_ts in results_df.index:
                print(f"           [探针] {signal_name} - normalize_score (window={window}) 结果 @ {probe_ts.strftime('%Y-%m-%d')}: {results_df[window].loc[probe_ts]:.4f}")
    if isinstance(windows, int):
        return results_df[windows].astype(np.float32)
    else:
        return results_df.astype(np.float32)

def normalize_to_bipolar(series: pd.Series, target_index: pd.Index, windows: Union[int, List[int]], sensitivity: Union[float, pd.Series] = 1.0, default_value: float = 0.0, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> Union[pd.Series, pd.DataFrame]:
    is_debug_enabled, probe_ts, signal_name = debug_info if debug_info else (False, None, "Unknown")
    if not isinstance(series, pd.Series) or series.empty:
        if isinstance(windows, int):
            return pd.Series(default_value, index=target_index, dtype=np.float32)
        else:
            return pd.DataFrame({w: default_value for w in windows}, index=target_index, dtype=np.float32)
    padded_series = series.reindex(target_index).ffill().bfill()
    padded_series = pd.to_numeric(padded_series, errors='coerce').fillna(0)
    is_original_zero = (padded_series.abs() < 1e-9)
    if isinstance(windows, int):
        windows_list = [windows]
    else:
        windows_list = windows
    series_values = padded_series.values.astype(np.float32)
    is_original_zero_values = is_original_zero.values
    # 将 sensitivity 转换为 NumPy 数组
    if isinstance(sensitivity, pd.Series):
        sensitivity_values = sensitivity.reindex(target_index).fillna(1.0).values.astype(np.float32)
    else:
        sensitivity_values = np.full(len(target_index), sensitivity, dtype=np.float32)
    results_array = _numba_normalize_to_bipolar_multi_window_core(
        series_values,
        windows_list,
        sensitivity_values, # 传递 NumPy 数组
        default_value,
        is_original_zero_values
    )
    results_df = pd.DataFrame(results_array, index=target_index, columns=windows_list, dtype=np.float32)
    if is_debug_enabled and probe_ts:
        for window in windows_list:
            if probe_ts in results_df.index:
                print(f"           [探针] {signal_name} - normalize_to_bipolar (window={window}) 结果 @ {probe_ts.strftime('%Y-%m-%d')}: {results_df[window].loc[probe_ts]:.4f}")
    if isinstance(windows, int):
        return results_df[windows].astype(np.float32)
    else:
        return results_df.astype(np.float32)

def _normalize_single_window_energy_score(series: pd.Series, target_index: pd.Index, window: int, ascending: bool = True, default_value: float = 0.5) -> pd.Series:
    """
    【内部辅助函数】对单个窗口进行归一化，并严格处理全零窗口。
    - 当滚动窗口内的所有有效数据点都为零时，强制其归一化分数为 0.0。
    - 否则，执行标准的 Min-Max 归一化。
    """
    if series.empty:
        return pd.Series(default_value, index=target_index, dtype=np.float32)
    series_aligned = series.reindex(target_index).fillna(method='ffill').fillna(method='bfill').fillna(0)
    series_values = series_aligned.values.astype(np.float32)
    normalized_values = _numba_normalize_single_window_energy_score_core(
        series_values,
        window,
        ascending,
        default_value
    )
    return pd.Series(normalized_values, index=target_index, dtype=np.float32)

def get_adaptive_mtf_normalized_energy_score(series: pd.Series, target_index: pd.Index, tf_weights: Dict[str, float], ascending: bool = True, default_value: float = 0.5) -> pd.Series:
    """
    【V1.0 · 能量指标专用】自适应多时间维度归一化分数计算器
    - 核心功能: 专为能量型指标设计，内部调用 `_normalize_single_window_energy_score`，
                  确保当多时间维度窗口内所有值都为零时，归一化分数严格为 0.0。
    """
    if series.empty:
        return pd.Series(default_value, index=target_index, dtype=np.float32)
    weighted_scores = pd.Series(0.0, index=target_index, dtype=np.float32)
    total_weight = 0.0
    for period_str, weight in tf_weights.items():
        period = int(period_str)
        if weight > 0:
            score = _normalize_single_window_energy_score(series, target_index, period, ascending=ascending, default_value=default_value)
            weighted_scores += score * weight
            total_weight += weight
    if total_weight > 0:
        return (weighted_scores / total_weight).clip(0, 1).astype(np.float32)
    else:
        return pd.Series(default_value, index=target_index, dtype=np.float32)

def load_external_json_config(file_path: str, default_return: Any = None) -> dict:
    import json
    import os
    if default_return is None:
        default_return = {}
    if not os.path.exists(file_path):
        print(f"    -> [配置加载警告] 文件 '{file_path}' 不存在，返回默认配置。")
        return default_return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"    -> [配置加载错误] 解析文件 '{file_path}' 失败: {e}，返回默认配置。")
        return default_return
    except Exception as e:
        print(f"    -> [配置加载错误] 读取文件 '{file_path}' 发生未知错误: {e}，返回默认配置。")
        return default_return

def _robust_geometric_mean(scores_dict: Dict[str, pd.Series], weights_dict: Dict[str, Any], df_index: pd.Index) -> pd.Series:
    """
    【V2.0 · NumPy优化版】计算健壮的加权几何平均分数。
    - 核心优化: 将内部计算从创建多个中间 Pandas DataFrame 转换为直接在 NumPy 数组上进行操作，
                  显著减少内存分配和 Python 解释器开销，提高计算效率。
    - 逻辑不变: 保持原始业务逻辑不变，即值为0（或接近0）的分数将被视为缺失，并从计算中排除，
                  同时调整总权重。如果某个日期所有组件都为0，则结果为0。
    """
    # 初始化用于累积加权对数和有效权重的NumPy数组
    # Initialize NumPy arrays for accumulating weighted log sums and effective weights
    log_sum_weighted_log_scores = np.zeros(len(df_index), dtype=np.float32)
    sum_of_effective_weights = np.zeros(len(df_index), dtype=np.float32)
    # 遍历每个分数序列及其对应的权重
    # Iterate through each score series and its corresponding weight
    for name, score_series in scores_dict.items():
        # 确保分数序列与目标索引对齐，填充缺失值，并转换为NumPy数组
        # Ensure score series is aligned with target index, fill NaNs, and convert to NumPy array
        aligned_score_values = score_series.reindex(df_index).fillna(0.0).values
        # 获取权重值，如果权重是Series，则进行对齐和填充；否则，创建常数NumPy数组
        # Get weight value; if it's a Series, align and fill; otherwise, create a constant NumPy array
        weight_val = weights_dict.get(name, 0.0)
        if isinstance(weight_val, pd.Series):
            aligned_weight_values = weight_val.reindex(df_index).fillna(0.0).values
        else:
            aligned_weight_values = np.full_like(aligned_score_values, weight_val, dtype=np.float32)
        # 识别有效的分数（大于一个很小的正数，避免log(0)）
        # Identify valid scores (greater than a small positive number to avoid log(0))
        is_valid_mask = (aligned_score_values > 1e-9)
        # 计算有效分数的对数
        # Calculate logarithm of valid scores
        log_score_col = np.full_like(aligned_score_values, np.nan, dtype=np.float32)
        log_score_col[is_valid_mask] = np.log(aligned_score_values[is_valid_mask])
        # 累加加权对数和，处理NaN值（np.nansum会忽略NaN）
        # Accumulate weighted log sums, handling NaN values (np.nansum ignores NaNs)
        # 使用 np.nansum 确保在有 NaN 的情况下也能正确累加
        log_sum_weighted_log_scores = np.nansum([log_sum_weighted_log_scores, log_score_col * aligned_weight_values], axis=0)
        # 累加有效权重
        # Accumulate effective weights
        sum_of_effective_weights[is_valid_mask] += aligned_weight_values[is_valid_mask]
    # 处理总有效权重为零或接近零的情况，避免除以零
    # Handle cases where total effective weights are zero or close to zero, to avoid division by zero
    sum_of_effective_weights_safe = np.where(np.isclose(sum_of_effective_weights, 0), np.nan, sum_of_effective_weights)
    # 计算指数部分
    # Calculate the exponent part
    exponent = log_sum_weighted_log_scores / sum_of_effective_weights_safe
    # 计算最终结果，并填充NaN为0.0
    # Calculate the final result, and fill NaNs with 0.0
    result_values = np.exp(exponent)
    result_values[np.isnan(result_values)] = 0.0 
    # 将NumPy数组转换为Pandas Series并返回
    # Convert NumPy array to Pandas Series and return
    return pd.Series(result_values, index=df_index, dtype=np.float32)

@numba.njit(cache=True)
def _numba_rank_window_average(arr_window, min_periods_for_rank):
    """
    【Numba优化版】用于 Pandas rolling().apply() 的自定义排名函数。
    精确模拟 scipy.stats.rankdata(arr_window, method='average')[-1] / len(arr_window) 的行为。
    - arr_window: 滚动窗口中的数据 (NumPy 数组)。
    - min_periods_for_rank: 计算排名所需的最小有效周期数。
    """
    n = len(arr_window)
    # 检查有效元素数量是否满足最小周期数条件
    valid_elements_count = np.sum(~np.isnan(arr_window))
    if valid_elements_count < min_periods_for_rank:
        return np.nan
    # 步骤1: 分离 NaN 和非 NaN 元素及其索引
    is_nan = np.isnan(arr_window)
    non_nan_indices = np.where(~is_nan)[0]
    nan_indices = np.where(is_nan)[0]
    # 步骤2: 对非 NaN 元素进行排序，获取其在非 NaN 数组中的排序索引
    # np.argsort 默认将 NaN 放在末尾，但我们这里只对非 NaN 部分排序
    sorted_non_nan_values_indices = np.argsort(arr_window[non_nan_indices])
    # 步骤3: 构建完整的排序索引，NaN 元素在前，然后是非 NaN 元素
    # 这样可以确保 NaN 获得最低的排名，与 scipy.stats.rankdata 默认行为一致
    # full_sorted_indices 存储的是原始 arr_window 中的索引
    full_sorted_indices = np.empty(n, dtype=np.intp) # np.intp 是平台相关的整数类型，适合索引
    full_sorted_indices[:len(nan_indices)] = nan_indices
    full_sorted_indices[len(nan_indices):] = non_nan_indices[sorted_non_nan_values_indices]
    # 步骤4: 分配排名 (平均排名法)
    ranks = np.empty(n, dtype=np.float64)
    current_rank = 1.0
    i = 0
    while i < n:
        j = i
        # 处理 NaN 元素
        if np.isnan(arr_window[full_sorted_indices[i]]):
            while j < n and np.isnan(arr_window[full_sorted_indices[j]]):
                j += 1
        else:
            # 处理非 NaN 元素
            while j < n and not np.isnan(arr_window[full_sorted_indices[j]]) and arr_window[full_sorted_indices[j]] == arr_window[full_sorted_indices[i]]:
                j += 1
        # 计算并分配平均排名
        avg_rank = (current_rank + (current_rank + (j - i) - 1)) / 2.0
        for k in range(i, j):
            ranks[full_sorted_indices[k]] = avg_rank
        current_rank += (j - i)
        i = j
    # 原始代码取 arr_window 中最后一个元素的排名，并除以窗口总长度
    rank_of_last_element = ranks[n - 1] # n-1 是 arr_window 中最后一个元素的索引
    return rank_of_last_element / n # 归一化排名 (除以窗口总长度)

@numba.njit(cache=True)
def _numba_rolling_rank_core(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算滚动窗口内的排名归一化分数。
    arr: 输入的NumPy数组。
    window: 滚动窗口大小。
    min_periods: 计算所需的最小非NaN周期数。
    返回: 滚动排名归一化分数数组。
    """
    n = len(arr)
    ranked_scores = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        current_window_data = arr[start_idx : i + 1]
        valid_data = current_window_data[~np.isnan(current_window_data)]
        if len(valid_data) < min_periods:
            continue
        last_element = arr[i]
        if np.isnan(last_element):
            continue
        less_than = 0
        equal_to = 0
        for val in valid_data:
            if val < last_element:
                less_than += 1
            elif val == last_element:
                equal_to += 1
        if len(valid_data) > 0:
            rank = (less_than + (equal_to - 1) / 2.0 + 1) / len(valid_data)
            ranked_scores[i] = rank
    return ranked_scores

@numba.njit(cache=True)
def _numba_normalize_score_multi_window_core(
    series_values: np.ndarray,
    windows_list: List[int],
    ascending: bool,
    default_value: float,
    is_original_zero_values: np.ndarray,
    min_periods_ratio: float = 0.2
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算多个窗口的滚动排名归一化分数。
    直接操作NumPy数组。
    """
    num_dates = len(series_values)
    num_windows = len(windows_list)
    results_array = np.full((num_dates, num_windows), default_value, dtype=np.float32)
    for w_idx in range(num_windows):
        window = windows_list[w_idx]
        if window <= 0:
            continue
        min_periods_for_rank = max(1, int(window * min_periods_ratio))
        ranked_series_values = _numba_rolling_rank_core(series_values, window, min_periods_for_rank)
        normalized_series_window = np.full(num_dates, default_value, dtype=np.float32)
        for i in range(num_dates):
            if np.isnan(ranked_series_values[i]):
                normalized_series_window[i] = default_value
            else:
                if not ascending:
                    normalized_series_window[i] = 1.0 - ranked_series_values[i]
                else:
                    normalized_series_window[i] = ranked_series_values[i]
            # 替换 np.clip 为 Numba 兼容的标量裁剪逻辑
            normalized_series_window[i] = max(np.float32(0.0), min(normalized_series_window[i], np.float32(1.0)))
            if is_original_zero_values[i]:
                normalized_series_window[i] = 0.0
        results_array[:, w_idx] = normalized_series_window
    return results_array

@numba.njit(cache=True)
def _numba_rolling_mean_std_core(arr: np.ndarray, window: int, min_periods: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba优化后的核心函数，用于计算滚动窗口内的均值和标准差。
    """
    n = len(arr)
    rolling_mean = np.full(n, np.nan, dtype=np.float32)
    rolling_std = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        start_idx = max(0, i - window + 1)
        current_window_data = arr[start_idx : i + 1]
        valid_data = current_window_data[~np.isnan(current_window_data)]
        if len(valid_data) >= min_periods:
            rolling_mean[i] = np.mean(valid_data)
            rolling_std[i] = np.std(valid_data) # np.std in Numba is population std by default
    return rolling_mean, rolling_std

@numba.njit(cache=True)
def _numba_normalize_to_bipolar_multi_window_core(
    series_values: np.ndarray,
    windows_list: List[int],
    sensitivity_values: np.ndarray, # 修改为 np.ndarray
    default_value: float,
    is_original_zero_values: np.ndarray,
    min_periods_ratio: float = 0.2
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算多个窗口的双极归一化分数。
    直接操作NumPy数组。
    """
    num_dates = len(series_values)
    num_windows = len(windows_list)
    results_array = np.full((num_dates, num_windows), default_value, dtype=np.float32)
    for w_idx in range(num_windows):
        window = windows_list[w_idx]
        if window <= 0:
            continue
        min_periods_for_rolling = max(1, int(window * min_periods_ratio))
        rolling_mean, rolling_std = _numba_rolling_mean_std_core(series_values, window, min_periods_for_rolling)
        bipolar_score_window = np.full(num_dates, default_value, dtype=np.float32)
        for i in range(num_dates):
            if np.isnan(series_values[i]):
                continue
            current_sensitivity = sensitivity_values[i] # 获取当前时间点的敏感度
            if not np.isnan(rolling_std[i]) and rolling_std[i] > 1e-9 and current_sensitivity > 1e-9: # 避免除以零
                z_score = (series_values[i] - rolling_mean[i]) / (rolling_std[i] * current_sensitivity)
                bipolar_score_window[i] = np.tanh(z_score)
            elif not np.isnan(rolling_mean[i]):
                bipolar_score_window[i] = np.sign(series_values[i] - rolling_mean[i])
            else:
                bipolar_score_window[i] = np.sign(series_values[i])
            if is_original_zero_values[i]:
                bipolar_score_window[i] = 0.0
            bipolar_score_window[i] = max(np.float32(-1.0), min(bipolar_score_window[i], np.float32(1.0)))
            if np.isnan(bipolar_score_window[i]):
                bipolar_score_window[i] = default_value
        results_array[:, w_idx] = bipolar_score_window
    return results_array

@numba.njit(cache=True)
def _numba_normalize_single_window_energy_score_core(
    series_values: np.ndarray,
    window: int,
    ascending: bool,
    default_value: float,
    min_periods_ratio: float = 0.2
) -> np.ndarray:
    """
    Numba优化后的核心函数，对单个窗口进行能量指标归一化。
    """
    n = len(series_values)
    normalized_scores = np.full(n, default_value, dtype=np.float32)
    min_periods = max(1, int(window * min_periods_ratio))
    for i in range(n):
        start_idx = max(0, i - window + 1)
        current_window_data = series_values[start_idx : i + 1]
        valid_data = current_window_data[~np.isnan(current_window_data)]
        if len(valid_data) < min_periods:
            continue
        rolling_min = np.min(valid_data)
        rolling_max = np.max(valid_data)
        if rolling_min == 0 and rolling_max == 0:
            normalized_scores[i] = 0.0
        elif rolling_max - rolling_min > 1e-9:
            if ascending:
                normalized_scores[i] = (series_values[i] - rolling_min) / (rolling_max - rolling_min)
            else:
                normalized_scores[i] = (rolling_max - series_values[i]) / (rolling_max - rolling_min)
        else:
            normalized_scores[i] = np.float32(1.0) if series_values[i] == rolling_max else np.float32(0.0)
        # 替换 np.clip 为 Numba 兼容的标量裁剪逻辑
        normalized_scores[i] = max(np.float32(0.0), min(normalized_scores[i], np.float32(1.0)))
        if np.isnan(normalized_scores[i]):
            normalized_scores[i] = default_value
    return normalized_scores

# 文件: strategies/trend_following/utils.py

@numba.njit(cache=True)
def _numba_calculate_zscore_core(
    series_values: np.ndarray,
    window: int,
    min_periods: int,
    default_value: float
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算滚动窗口内的Z-score。
    """
    n = len(series_values)
    zscores = np.full(n, default_value, dtype=np.float32)
    rolling_mean, rolling_std = _numba_rolling_mean_std_core(series_values, window, min_periods)
    for i in range(n):
        if not np.isnan(series_values[i]) and not np.isnan(rolling_mean[i]) and not np.isnan(rolling_std[i]):
            if rolling_std[i] > 1e-9: # 避免除以零
                zscores[i] = (series_values[i] - rolling_mean[i]) / rolling_std[i]
            else:
                # 如果标准差为0，表示窗口内所有值相同，zscore为0
                zscores[i] = 0.0
        else:
            zscores[i] = default_value
    return zscores



