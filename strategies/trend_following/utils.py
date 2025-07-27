# 文件: strategies/trend_following/utils.py
import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Any

# 这个文件包含所有层级都可能用到的通用辅助函数

def get_param_value(param: Any, default: Any = None) -> Any:
    if isinstance(param, dict) and 'value' in param:
        return param['value']
    if param is not None:
        return param
    return default

def get_params_block(strategy_instance, block_name: str, default_return: Any = None) -> dict:
    if default_return is None:
        default_return = {}
    trend_follow_params = strategy_instance.unified_config.get('strategy_params', {}).get('trend_follow', {})
    params = trend_follow_params.get(block_name)
    if params is None:
        params = strategy_instance.unified_config.get(block_name)
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
    print(f"          -> [状态机引擎] 正在为 '{state_name}' 创建持续状态窗口 (共 {len(entry_indices)} 个进入点)...")
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
