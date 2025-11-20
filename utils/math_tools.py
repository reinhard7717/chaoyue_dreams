# 文件: utils/math_tools.py
import numpy as np
from hurst import compute_Hc

def hurst_exponent(series: np.ndarray) -> float:
    # 1. 检查传入的是否是有效的numpy数组
    if not isinstance(series, np.ndarray) or series.ndim != 1:
        return np.nan
    # 2. 检查长度
    if len(series) < 100:
        return np.nan
    # 3. 【关键】检查数据是否包含NaN或inf
    if np.any(np.isnan(series)) or np.any(np.isinf(series)):
        return np.nan
    # 4. 【关键】检查数据是否为常数（导致标准差为0）
    if np.std(series) < 1e-9: # 如果标准差极小，视为常数
        return 0.5 # 常数序列或白噪声的H值理论上是0.5
    try:
        H, c, data = compute_Hc(series, kind='price', simplified=True)
        return H
    except Exception as e:
        # 打印更详细的错误，便于调试
        print(f"[DEBUG] hurst_exponent failed with error: {e} for series: {series[:5]}...")
        return np.nan