# 文件: strategies/trend_following/utils.py
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any
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
    - 核心修改: 智能判断传入的 strategy_instance 是对象还是字典，并从中安全地提取配置。
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
    if not converted_cols:
        print("      -> 所有数值列类型正常，无需转换。")
    return df

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









