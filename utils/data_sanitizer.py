# 文件: utils/data_sanitizer.py

import numpy as np
import pandas as pd
from decimal import Decimal
from datetime import date, datetime

def sanitize_for_json(data):
    """
    【核心工具函数】递归地将数据结构中的非JSON兼容类型转换为Python原生类型。
    处理包括：
    - NumPy 的整数、浮点数、布尔值
    - Pandas 的 Timestamp, NaT, NA
    - Python 的 Decimal, date, datetime
    - 递归处理字典和列表
    """
    # 检查是否是Pandas的特殊空值
    if pd.isna(data):
        return None
        
    # 检查 NumPy 的整数类型
    if isinstance(data, np.integer):
        return int(data)
    # 检查 NumPy 的浮点数类型
    if isinstance(data, np.floating):
        # 处理无穷大和NaN (Not a Number)
        if np.isinf(data) or np.isnan(data):
            return None
        return float(data)
    # 检查 NumPy 的布尔类型
    if isinstance(data, np.bool_):
        return bool(data)
        
    # 检查 Python 的 Decimal 类型 (常用于 models.DecimalField)
    if isinstance(data, Decimal):
        # 转换为 float 可能有精度损失，但对于JSON快照通常足够
        # 如果需要绝对精度，应转换为字符串: str(data)
        return float(data)
        
    # 检查 Python 的 datetime 或 date 类型
    if isinstance(data, (datetime, date, pd.Timestamp)):
        # 转换为标准的 ISO 8601 格式字符串
        return data.isoformat()
    # --- 递归处理容器类型 ---
    # 必须在处理完原子类型之后
    # 如果是字典，递归处理它的每一个值
    if isinstance(data, dict):
        return {sanitize_for_json(key): sanitize_for_json(value) for key, value in data.items()}
        
    # 如果是列表或元组，递归处理它的每一个元素
    if isinstance(data, (list, tuple)):
        return [sanitize_for_json(element) for element in data]
    # 如果以上都不是，假定数据类型是安全的，直接返回
    return data

