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
    【V2.0 · 顶部识别升级版】计算全局的底部和顶部上下文分数。
    - 核心升级: 新增了对“顶部上下文”的精确识别。
    - 顶部识别算法: 融合了“波段伸展度”（价格远离生命线MA55的程度）和“均线排列恶化”（短期均线开始下穿长期均线）两大特征。
    """
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

