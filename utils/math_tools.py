# 文件: utils/math_tools.py
import numpy as np
from hurst import compute_Hc

def hurst_exponent(series: np.ndarray) -> float:
    """
    一个包装函数，用于计算赫斯特指数，使其能被 rolling().apply() 调用。
    
    Args:
        series (np.ndarray): 一个时间序列（例如价格）。
    
    Returns:
        float: 赫斯特指数 (H)。
    """
    # compute_Hc 需要一个list或numpy array，并且不能有NaN
    # rolling().apply() 传递过来的已经是处理好的numpy array
    if len(series) < 20: # 赫斯特指数需要足够的数据点才有意义
        return np.nan
    
    try:
        # kind='price' 表示这是一个价格序列
        H, c, data = compute_Hc(series, kind='price', simplified=True)
        return H
    except Exception:
        # 在计算过程中可能出现各种数学错误，返回NaN是安全的
        return np.nan
