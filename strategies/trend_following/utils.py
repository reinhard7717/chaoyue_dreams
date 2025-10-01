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
    persistent_series = pd.Series(False, index=df.index)
    entry_indices = df.index[entry_event_series]
    if entry_indices.empty:
        return persistent_series
    # print(f"          -> [状态机引擎] 正在为 '{state_name}' 创建持续状态窗口 (共 {len(entry_indices)} 个进入点)...")
    for entry_idx in entry_indices:
        window_end_date = entry_idx + pd.Timedelta(days=persistence_days)
        actual_window_mask = (df.index >= entry_idx) & (df.index <= window_end_date)
        break_points = df.index[actual_window_mask & break_condition_series]
        if not break_points.empty:
            end_date = break_points[0]
        else:
            end_date = df.index[actual_window_mask][-1] if actual_window_mask.any() else entry_idx
        persistent_series.loc[entry_idx:end_date] = True
    return persistent_series

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
    【新增公共函数】计算一个系列在滚动窗口内的归一化得分 (0-1)。
    - 核心逻辑: 使用滚动窗口的百分比排名 (rank(pct=True)) 来实现归一化。
    - 健壮性: 处理空Series，并使用 target_index 确保返回的Series长度和索引正确。
    """
    if series is None or series.isnull().all() or series.empty:
        return pd.Series(default_value, index=target_index, dtype=np.float32)

    # 确保series的索引与目标索引对齐，避免后续操作因索引不匹配产生问题
    series = series.reindex(target_index)

    min_periods = max(1, int(window * 0.2))
    
    rank = series.rolling(
        window=window, 
        min_periods=min_periods
    ).rank(
        pct=True, 
        ascending=ascending
    )
    
    # 使用 reindex 再次确保最终输出的索引是完整的 target_index
    return rank.reindex(target_index).fillna(default_value).astype(np.float32)

def calculate_context_scores(df: pd.DataFrame, atomic_states: Dict) -> Tuple[pd.Series, pd.Series]:
    """
    【V2.1 · 哲人石归位版】计算全局的底部和顶部上下文分数。
    - 核心加固: 增加对输入 df 类型的检查。如果传入的是一个字典（旧的错误调用方式），
                  则尝试从中提取 'df_indicators'，为数据链路提供“双重保险”。
    """
    # 增加防御性编程，处理错误的字典输入
    if isinstance(df, dict):
        df = df.get('df_indicators', pd.DataFrame())
    if 'close_D' not in df.columns:
        # 如果关键列仍然缺失，返回默认值以避免崩溃
        print("      -> [calculate_context_scores] 警告: 输入的DataFrame缺少'close_D'列，无法计算上下文分数。")
        empty_series = pd.Series(0.5, index=df.index if not df.empty else None, dtype=np.float32)
        return empty_series, empty_series
    # --- 底部上下文分数计算 (保持不变) ---
    price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
    deep_bottom_context_score = 1.0 - price_pos_yearly
    rsi_w_oversold_score = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
    cycle_phase = atomic_states.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index)).fillna(0.0)
    cycle_trough_score = (1 - cycle_phase) / 2.0
    bottom_context_score_values = np.maximum.reduce([
        deep_bottom_context_score.values,
        rsi_w_oversold_score.values,
        cycle_trough_score.values
    ])
    bottom_context_score = pd.Series(bottom_context_score_values, index=df.index, dtype=np.float32)
    # --- 顶部上下文分数计算 ---
    # 特征一：波段伸展度 (Overextension) - 价格大幅远离MA55生命线
    ma55 = df.get('EMA_55_D', df['close_D'])
    rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
    wave_channel_height = (rolling_high_55d - ma55).replace(0, 1e-9)
    stretch_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
    # 特征二：均线排列恶化 (MA Misalignment) - 短期均线开始掉头向下
    ma_periods = [5, 13, 21, 55]
    misalignment_scores = []
    for i in range(len(ma_periods) - 1):
        short_ma = df.get(f'EMA_{ma_periods[i]}_D', df['close_D'])
        long_ma = df.get(f'EMA_{ma_periods[i+1]}_D', df['close_D'])
        # 当短期均线下穿长期均线时，分数变高
        misalignment_scores.append((short_ma < long_ma).astype(float))
    if misalignment_scores:
        misalignment_score = pd.DataFrame(misalignment_scores).mean()
    else:
        misalignment_score = pd.Series(0.5, index=df.index)
    # 融合两大特征，得到最终的顶部上下文分数
    top_context_score = (stretch_score * misalignment_score).astype(np.float32)
    return bottom_context_score, top_context_score

def normalize_to_bipolar(series: pd.Series, target_index: pd.Index, window: int, sensitivity: float = 1.0, default_value: float = 0.0) -> pd.Series:
    """
    【V2.0.1 增强文档版】将序列归一化到 -1 到 1 的双极区间。
    - 核心逻辑: 采用滚动Z-score并使用tanh函数进行平滑压缩，完美适用于四象限分析。
    - 战略意义: 用于将一个指标的变化率或原始值，转化为一个同时蕴含【方向】和【强度】的标准化分数。
                +1 代表极强的正向偏离，-1 代表极强的负向偏离，0 代表符合近期常态。
                这对于构建“动量A vs 推力B”的力学模型至关重要。
    - :param series: 原始数据序列。
    - :param target_index: 目标DataFrame的索引。
    - :param window: 滚动窗口大小，用于计算均值和标准差的“常态”区间。
    - :param sensitivity: 敏感度因子。值越小，Z-score的绝对值越大，得分越快地趋近于±1，反应越灵敏。默认为1.0，代表标准Z-score。
    - :param default_value: 默认填充值。
    - :return: 归一化到(-1, 1)区间的pd.Series。
    """
    if series is None or series.isnull().all() or series.empty:
        return pd.Series(default_value, index=target_index, dtype=np.float32)
    series = series.reindex(target_index)
    min_periods = max(1, int(window * 0.2))
    # 计算滚动均值和标准差
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()
    # 避免除以零，如果窗口内值无波动，标准差为0，视为无偏离
    rolling_std = rolling_std.replace(0, np.nan)
    # 计算Z-score，sensitivity作为分母，调节敏感度
    z_score = (series - rolling_mean) / (rolling_std * sensitivity)
    # 使用tanh函数进行平滑压缩到(-1, 1)
    bipolar_score = np.tanh(z_score)
    return bipolar_score.reindex(target_index).fillna(default_value).astype(np.float32)

def calculate_holographic_dynamics(df: pd.DataFrame, base_name: str, norm_window: int) -> Tuple[pd.Series, pd.Series]:
    """
    【V2.2 · 防御性加固版】全息动态计算引擎
    - 核心逻辑: 融合“速度变化”(加速度)和“力量变化”(加加速度/Jerk)。
    - 本次加固: 无论上游数据是否存在，都确保本函数内部处理的是Series，彻底杜绝类型错误。
    """
    # 创建一个默认的Series，用于在df.get找不到列时返回，确保类型安全
    default_series = pd.Series(0.0, index=df.index)

    # 维度一：速度变化 (加速度)
    # 使用 default_series 作为 df.get 的备用返回值
    slope_1 = df.get(f'SLOPE_1_{base_name}', default_series)
    slope_5 = df.get(f'SLOPE_5_{base_name}', default_series)
    slope_diff = slope_1 - slope_5
    velocity_accel_score = normalize_score(slope_diff, df.index, norm_window, ascending=True)
    velocity_decel_score = normalize_score(slope_diff, df.index, norm_window, ascending=False)

    # 维度二：力量变化 (加加速度 / Jerk)
    # 使用 default_series 作为 df.get 的备用返回值
    accel_1 = df.get(f'ACCEL_1_{base_name}', default_series)
    accel_5 = df.get(f'ACCEL_5_{base_name}', default_series)
    accel_diff = accel_1 - accel_5
    jerk_accel_score = normalize_score(accel_diff, df.index, norm_window, ascending=True)
    jerk_decel_score = normalize_score(accel_diff, df.index, norm_window, ascending=False)
    
    # 融合：两大维度必须共振
    bullish_holographic_score = (velocity_accel_score * jerk_accel_score)**0.5
    bearish_holographic_score = (velocity_decel_score * jerk_decel_score)**0.5
    
    return bullish_holographic_score, bearish_holographic_score

def transmute_health_to_ultimate_signals(
    df: pd.DataFrame,
    atomic_states: Dict,
    overall_health: Dict,
    params: Dict,
    domain_prefix: str
) -> Dict[str, pd.Series]:
    """
    【V1.7 · 赫淮斯托斯之砧版】终极信号中央合成引擎
    - 核心升级: 新增“宏观趋势上下文”，为战术反转信号提供稳定的“行动许可”，解决其无法激活的问题。
    - 核心逻辑: 1. 计算实时、敏感的“趋势确认分”，用于抑制战略底部信号。
                2. 基于实时分数，生成一个平滑、稳定的“宏观趋势许可分”。
                3. “战术反转”信号的激活，将由这个稳定的宏观分数授权。
    """
    states = {}
    # --- 1. 获取通用参数和上下文信号 ---
    resonance_tf_weights = get_param_value(params.get('resonance_tf_weights'), {'short': 0.2, 'medium': 0.5, 'long': 0.3})
    reversal_tf_weights = get_param_value(params.get('reversal_tf_weights'), {'short': 0.6, 'medium': 0.3, 'long': 0.1})
    periods = get_param_value(params.get('periods'), [1, 5, 13, 21, 55])
    norm_window = get_param_value(params.get('norm_window'), 55)
    bottom_context_bonus_factor = get_param_value(params.get('bottom_context_bonus_factor'), 0.5)
    exponent = get_param_value(params.get('final_score_exponent'), 1.0)
    bottom_context_score, top_context_score = calculate_context_scores(df, atomic_states)
    recent_reversal_context = atomic_states.get('SCORE_CONTEXT_RECENT_REVERSAL', pd.Series(0.0, index=df.index))
    relational_dynamics_power = atomic_states.get('SCORE_ATOMIC_RELATIONAL_DYNAMICS', pd.Series(0.5, index=df.index))
    default_series = pd.Series(0.5, index=df.index, dtype=np.float32)

    # --- 上下文计算 ---
    # 阿波罗日冕：计算新高强度分，用于渐进式遗忘 (逻辑不变)
    new_high_params = get_param_value(params.get('new_high_context_params'), {})
    new_high_context_score = _calculate_new_high_context(df, new_high_params)
    atomic_states['CONTEXT_NEW_HIGH_STRENGTH'] = new_high_context_score
    memory_retention_factor = 1.0 - new_high_context_score
    recent_reversal_context_modulated = recent_reversal_context * memory_retention_factor

    # 升级趋势上下文的计算流程
    # 1. 计算实时的、敏感的趋势确认分 (波塞冬三叉戟)
    trend_confirmation_params = get_param_value(params.get('trend_confirmation_context_params'), {})
    trend_confirmation_context = _calculate_trend_confirmation_context(df, trend_confirmation_params, norm_window)
    atomic_states['CONTEXT_TREND_CONFIRMED'] = trend_confirmation_context

    # 2. 基于实时分数，计算平滑的、稳定的宏观趋势许可分 (赫淮斯托斯之砧)
    tactical_params = get_param_value(params.get('tactical_reversal_params'), {})
    macro_window = get_param_value(tactical_params.get('macro_trend_window'), 3)
    macro_trend_permit_context = trend_confirmation_context.rolling(window=macro_window, min_periods=1).mean()
    atomic_states['CONTEXT_MACRO_TREND_PERMIT'] = macro_trend_permit_context
    
    # 计算全新的“动态反转上下文”，为战术信号提供核心驱动
    dynamic_reversal_params = get_param_value(params.get('dynamic_reversal_context_params'), {})
    dynamic_reversal_context = _calculate_dynamic_reversal_context(df, dynamic_reversal_params, norm_window)
    atomic_states['CONTEXT_DYNAMIC_REVERSAL'] = dynamic_reversal_context

    # --- 信号计算 ---
    
    # 战略底部反转 (Strategic Bottom Reversal) - 使用敏感的实时分数进行抑制
    bullish_reversal_health = {p: recent_reversal_context_modulated * relational_dynamics_power * overall_health['d_intensity'].get(p, default_series) for p in periods}
    bullish_short_force_rev = (bullish_reversal_health.get(1, default_series) * bullish_reversal_health.get(5, default_series))**0.5
    bullish_medium_trend_rev = (bullish_reversal_health.get(13, default_series) * bullish_reversal_health.get(21, default_series))**0.5
    bullish_long_inertia_rev = bullish_reversal_health.get(55, default_series)
    overall_bullish_reversal_trigger = ((bullish_short_force_rev ** reversal_tf_weights['short']) * (bullish_medium_trend_rev ** reversal_tf_weights['medium']) * (bullish_long_inertia_rev ** reversal_tf_weights['long']))
    raw_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + recent_reversal_context_modulated * bottom_context_bonus_factor)).clip(0, 1)
    final_bottom_reversal_score = raw_bottom_reversal_score * (1 - trend_confirmation_context)

    # 战术回调反转 (Tactical Pullback Reversal) - 现在由新的宏观许可分授权
    final_tactical_reversal_score = calculate_tactical_reversal_score(df, atomic_states, overall_health, tactical_params, norm_window)

    # 看涨共振 (Bullish Resonance) - 逻辑不变
    bullish_resonance_health = {p: np.maximum(overall_health['s_bull'].get(p, default_series), relational_dynamics_power) * overall_health['d_intensity'].get(p, default_series) for p in periods}
    bullish_short_force_res = (bullish_resonance_health.get(1, default_series) * bullish_resonance_health.get(5, default_series))**0.5
    bullish_medium_trend_res = (bullish_resonance_health.get(13, default_series) * bullish_resonance_health.get(21, default_series))**0.5
    bullish_long_inertia_res = bullish_resonance_health.get(55, default_series)
    overall_bullish_resonance = ((bullish_short_force_res ** resonance_tf_weights['short']) * (bullish_medium_trend_res ** resonance_tf_weights['medium']) * (bullish_long_inertia_res ** resonance_tf_weights['long']))
    
    # 看跌共振 (Bearish Resonance) - 逻辑不变
    bearish_resonance_health = {p: overall_health['s_bear'].get(p, default_series) * overall_health['d_intensity'].get(p, default_series) for p in periods}
    bearish_short_force_res = (bearish_resonance_health.get(1, default_series) * bearish_resonance_health.get(5, default_series))**0.5
    bearish_medium_trend_res = (bearish_resonance_health.get(13, default_series) * bearish_resonance_health.get(21, default_series))**0.5
    bearish_long_inertia_res = bearish_resonance_health.get(55, default_series)
    overall_bearish_resonance = ((bearish_short_force_res ** resonance_tf_weights['short']) * (bearish_medium_trend_res ** resonance_tf_weights['medium']) * (bearish_long_inertia_res ** resonance_tf_weights['long']))
    
    # 顶部反转 (Top Reversal) - 逻辑不变
    bearish_reversal_health = {p: overall_health['s_bear'].get(p, default_series) * overall_health['d_intensity'].get(p, default_series) for p in periods}
    bearish_short_force_rev = (bearish_reversal_health.get(1, default_series) * bearish_reversal_health.get(5, default_series))**0.5
    bearish_medium_trend_rev = (bearish_reversal_health.get(13, default_series) * bearish_reversal_health.get(21, default_series))**0.5
    bearish_long_inertia_rev = bearish_reversal_health.get(55, default_series)
    overall_bearish_reversal_trigger = ((bearish_short_force_rev ** reversal_tf_weights['short']) * (bearish_medium_trend_rev ** reversal_tf_weights['medium']) * (bearish_long_inertia_rev ** reversal_tf_weights['long']))
    final_top_reversal_score = (overall_bearish_reversal_trigger * top_context_score).clip(0, 1)
    
    # --- 6. 组装并返回最终信号字典 ---
    final_signal_map = {
        f'SCORE_{domain_prefix}_BULLISH_RESONANCE': (overall_bullish_resonance ** exponent),
        f'SCORE_{domain_prefix}_BOTTOM_REVERSAL': (final_bottom_reversal_score ** exponent),
        f'SCORE_{domain_prefix}_TACTICAL_REVERSAL': (final_tactical_reversal_score ** exponent),
        f'SCORE_{domain_prefix}_BEARISH_RESONANCE': (overall_bearish_resonance ** exponent),
        f'SCORE_{domain_prefix}_TOP_REVERSAL': (final_top_reversal_score ** exponent)
    }
    for signal_name, score in final_signal_map.items():
        states[signal_name] = score.astype(np.float32)
    return states

def _calculate_new_high_context(df: pd.DataFrame, params: Dict) -> pd.Series:
    """
    【V2.0 · 阿波罗日冕版】多维新高上下文分数计算器
    - 核心革命: 不再只看价格，而是融合“价格突破”、“均线斜率”和“乖离健康度”三位一体评估新高。
    - 核心逻辑:
      1. 价格突破: 价格是否创出P周期新高。
      2. 趋势确认: 对应均线的斜率是否为正。
      3. 乖离健康度: BIAS是否在健康阈值内。
      - 最终分数 = (价格分 * w1 + 斜率分 * w2 + 乖离分 * w3)
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    # 从配置中获取更丰富的参数
    periods = get_param_value(params.get('periods'), [5, 13, 21, 55])
    period_weights = get_param_value(params.get('period_weights'), {})
    bias_thresholds = get_param_value(params.get('bias_thresholds'), {})
    fusion_weights = get_param_value(params.get('fusion_weights'), {})
    
    final_period_scores = []
    
    for p in periods:
        period_weight = period_weights.get(str(p), 0)
        if period_weight == 0:
            continue

        # 维度一: 价格突破 (Price Breakout)
        rolling_high = df['high_D'].rolling(window=p, min_periods=1).max().shift(1)
        is_new_high_score = (df['high_D'] > rolling_high).astype(float)

        # 维度二: 趋势确认 (Trend Confirmation via MA Slope)
        ma_slope_col = f'SLOPE_{p}_EMA_{p}_D'
        # 如果没有对应周期的斜率，则使用收盘价斜率作为备用
        if ma_slope_col not in df.columns:
            ma_slope_col = f'SLOPE_{p}_close_D'
        
        ma_slope = df.get(ma_slope_col, pd.Series(0, index=df.index))
        # 将斜率归一化到0-1，斜率越大分数越高
        ma_slope_score = normalize_score(ma_slope, df.index, window=p*2, ascending=True)

        # 维度三: 乖离健康度 (BIAS Health)
        bias_period = 21 if p <= 21 else 55 # 短中期用BIAS21，长期用BIAS55
        bias_col = f'BIAS_{bias_period}_D'
        bias_threshold = bias_thresholds.get(str(bias_period), 0.2)
        
        bias_value = df.get(bias_col, pd.Series(0, index=df.index)).abs()
        # BIAS越小越健康，分数越高。当BIAS超过阈值时，健康度急剧下降。
        bias_health_score = (1 - (bias_value / bias_threshold)).clip(0, 1)

        # 三位一体融合，得到该周期的综合新高分
        period_new_high_score = (
            is_new_high_score * fusion_weights.get('new_high', 0.4) +
            ma_slope_score * fusion_weights.get('ma_slope', 0.3) +
            bias_health_score * fusion_weights.get('bias_health', 0.3)
        )
        
        # 对该周期的分数应用其在最终融合中的权重
        final_period_scores.append(period_new_high_score * period_weight)

    if not final_period_scores:
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    # 将所有加权的周期分数相加，得到最终的“新高强度分”
    total_weight = sum(w for p, w in period_weights.items() if int(p) in periods)
    if total_weight == 0:
        return pd.Series(0.0, index=df.index, dtype=np.float32)
        
    new_high_context_score = sum(final_period_scores) / total_weight
    return new_high_context_score.clip(0, 1).astype(np.float32)

def _calculate_trend_confirmation_context(df: pd.DataFrame, params: Dict, norm_window: int) -> pd.Series:
    """
    【V2.0 · 波塞冬三叉戟版】趋势确认上下文计算器
    - 核心革命: 引入ADX(强度)、PDI/NDI(方向)、BIAS(健康度)三位一体评估趋势。
    - 核心逻辑: 趋势确认分 = 强度分 * 方向分 * (1 - 乖离分)
                 这使得在健康回调（BIAS回归）时，抑制作用会减弱，从而能捕捉小波段底部。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    # 获取新的三叉戟参数
    adx_threshold = get_param_value(params.get('adx_threshold'), 20)
    fusion_weights = get_param_value(params.get('fusion_weights'), {})
    
    # 叉戟一: 趋势强度 (Trend Strength)
    adx = df.get('ADX_14_D', pd.Series(0, index=df.index))
    is_trending = (adx > adx_threshold).astype(float)
    strength_score = normalize_score(adx, df.index, window=norm_window, ascending=True) * is_trending

    # 叉戟二: 趋势方向 (Trend Direction)
    pdi = df.get('PDI_14_D', pd.Series(0, index=df.index))
    ndi = df.get('NDI_14_D', pd.Series(0, index=df.index))
    direction_score = (pdi > ndi).astype(float)

    # 叉戟三: 趋势健康度 (Trend Health via BIAS)
    # 我们关心的是乖离的绝对值，乖离越大越不健康
    bias_abs = df.get('BIAS_21_D', pd.Series(0, index=df.index)).abs()
    # 乖离越大，分数越高，代表越“不健康”
    unhealthiness_score = normalize_score(bias_abs, df.index, window=norm_window, ascending=True)
    
    # 最终融合
    # 健康回调时，bias_abs减小 -> unhealthiness_score减小 -> (1 - unhealthiness_score)增大
    # -> 最终分数减小 -> 对底部反转的抑制减弱！
    trend_confirmation_score = (
        strength_score * fusion_weights.get('strength', 0.4) +
        direction_score * fusion_weights.get('direction', 0.4) +
        (1 - unhealthiness_score) * fusion_weights.get('health', 0.2) # 注意这里是 (1 - score)
    )
    
    # 修正：上面的加权和逻辑是错误的，应该使用乘法来体现“与”逻辑
    # 一个确认的上升趋势 = 趋势性强 AND 方向向上 AND 乖离健康
    # 健康度分数应该是 bias 越小分数越高
    health_score = 1 - unhealthiness_score
    
    # 正确的融合逻辑
    trend_confirmation_score = (strength_score * direction_score * health_score)

    return trend_confirmation_score.clip(0, 1).astype(np.float32)

def calculate_tactical_reversal_score(
    df: pd.DataFrame,
    atomic_states: Dict,
    overall_health: Dict,
    params: Dict,
    norm_window: int
) -> pd.Series:
    """
    【V1.2 · 动态反转版】战术反转信号计算器 (赫尔墨斯的飞翼鞋)
    - 核心升级: 废除简陋的BIAS回调逻辑，使用全新的、基于二阶求导的“动态反转上下文”作为核心驱动。
    - 新激活公式: 宏观趋势许可分 * 动态反转上下文分 * 反转动能分
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    # --- 1. 获取参数和基础信号 ---
    momentum_weight = get_param_value(params.get('momentum_weight'), 0.5)
    relational_power_weight = get_param_value(params.get('relational_power_weight'), 0.5)
    
    # 准入证: 宏观趋势许可 (逻辑不变)
    trend_permission_score = atomic_states.get('CONTEXT_MACRO_TREND_PERMIT', pd.Series(0.0, index=df.index))

    # 核心驱动力升级为“动态反转上下文”
    # 不再计算BIAS回调深度，而是直接获取由新引擎计算的分数
    dynamic_reversal_context_score = atomic_states.get('CONTEXT_DYNAMIC_REVERSAL', pd.Series(0.0, index=df.index))

    # --- 3. 计算反转动能分 (Reversal Momentum Score) - 逻辑不变 ---
    relational_power = atomic_states.get('SCORE_ATOMIC_RELATIONAL_DYNAMICS', pd.Series(0.5, index=df.index))
    short_term_momentum = overall_health.get('d_intensity', {}).get(1, pd.Series(0.5, index=df.index))
    reversal_momentum_score = (
        relational_power * relational_power_weight +
        short_term_momentum * momentum_weight
    )

    # --- 4. 最终融合 ---
    tactical_reversal_score = (
        trend_permission_score *
        dynamic_reversal_context_score * # 使用新的核心驱动
        reversal_momentum_score
    )
    
    return tactical_reversal_score.clip(0, 1).astype(np.float32)

def _calculate_dynamic_reversal_context(df: pd.DataFrame, params: Dict, norm_window: int) -> pd.Series:
    """
    【V1.0 · 新增】动态反转上下文计算器 (二阶求导引擎)
    - 核心职责: 借鉴ProcessIntelligence的二阶求导思想，捕捉短期均线距离和斜率的“反转加速度”。
    - 输出: 一个在 [0, 1] 区间的Series，分数越高，代表战术性反转的动能越强。
    """
    if not get_param_value(params.get('enabled'), False):
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    # --- 1. 获取参数 ---
    short_ma_period = get_param_value(params.get('short_ma_period'), 5)
    mid_ma_period = get_param_value(params.get('mid_ma_period'), 21)
    slope_period = get_param_value(params.get('slope_period'), 3)
    weights = get_param_value(params.get('fusion_weights'), {'distance_accel': 0.5, 'slope_accel': 0.5})
    
    short_ma_col = f'EMA_{short_ma_period}_D'
    mid_ma_col = f'EMA_{mid_ma_period}_D'
    short_ma_slope_col = f'SLOPE_{slope_period}_EMA_{short_ma_period}_D'

    if not all(c in df.columns for c in [short_ma_col, mid_ma_col, short_ma_slope_col]):
        return pd.Series(0.0, index=df.index, dtype=np.float32)

    # --- 2. 维度一: 均线距离的收敛加速度 ---
    ma_distance = df[short_ma_col] - df[mid_ma_col]
    # 我们关心的是距离从负值（回调）向0（收敛）变化的速度，所以对距离的斜率求导
    ma_distance_slope = ma_distance.diff(slope_period).fillna(0)
    # 距离收敛的加速度，越高越好
    distance_accel_score = normalize_score(ma_distance_slope, df.index, window=norm_window, ascending=True)

    # --- 3. 维度二: 短期均线斜率的扭转加速度 ---
    short_ma_slope = df[short_ma_slope_col]
    # 斜率从负值（向下）向正值（向上）变化的加速度，越高越好
    short_ma_slope_accel = short_ma_slope.diff(slope_period).fillna(0)
    slope_accel_score = normalize_score(short_ma_slope_accel, df.index, window=norm_window, ascending=True)

    # --- 4. 最终融合 ---
    dynamic_reversal_score = (
        distance_accel_score * weights.get('distance_accel', 0.5) +
        slope_accel_score * weights.get('slope_accel', 0.5)
    )
    
    return dynamic_reversal_score.clip(0, 1).astype(np.float32)
















