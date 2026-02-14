# 新增文件: services/feature_engineering_service.py

import asyncio
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import pandas_ta as ta
import numba
from numba import objmode, prange
from utils.math_tools import hurst_exponent
from strategies.trend_following.utils import _numba_nonlinear_fusion_core
logger = logging.getLogger("services")

@numba.njit(parallel=True, cache=True)
def _numba_rolling_slope(data: np.ndarray, window: int) -> np.ndarray:
    """
    【V1.1 · Numba并行优化版】计算滚动线性回归斜率。
    - 优化: 启用 parallel=True 和 prange，利用多核并行计算每个窗口的斜率。
    - 性能: 在多核 CPU 上提升显著。
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    # 预计算 x 的相关项 (x 为 0 到 window-1)
    sum_x = (window - 1) * window / 2.0
    sum_x_sq = (window - 1) * window * (2 * window - 1) / 6.0
    denominator = window * sum_x_sq - sum_x * sum_x
    if denominator == 0:
        return result
        
    # 并行循环
    for i in prange(window - 1, n):
        # 提取当前窗口 (注意：切片在 Numba 中通常不产生拷贝，但在并行中是安全的)
        y_slice = data[i - window + 1 : i + 1]
        # 检查 NaN (手动循环检查比 np.isnan(slice).any() 更快且兼容性更好)
        has_nan = False
        for k in range(window):
            if np.isnan(y_slice[k]):
                has_nan = True
                break
        if has_nan:
            continue
        sum_y = 0.0
        sum_xy = 0.0
        for j in range(window):
            val = y_slice[j]
            sum_y += val
            sum_xy += j * val
        slope = (window * sum_xy - sum_x * sum_y) / denominator
        result[i] = slope
    return result

@numba.njit(cache=True)
def _numba_sample_entropy_core(x: np.ndarray, m: int, r: float) -> float:
    """
    【Numba优化】样本熵计算核心逻辑。
    """
    n = len(x)
    if n < m + 1:
        return np.nan

    # 统计匹配模板的数量
    # count_m: 长度为 m 的匹配数
    # count_m_plus_1: 长度为 m+1 的匹配数
    count_m = 0
    count_m_plus_1 = 0
    # 优化：避免重复切片，直接比较
    # 但为保持逻辑清晰，使用循环比较
    # A: count of vector pairs of length m+1 having d < r
    # B: count of vector pairs of length m having d < r
    # 简单的 O(N^2) 实现
    for i in range(n - m):
        for j in range(i + 1, n - m):
            # 检查长度 m
            dist_m = 0.0
            for k in range(m):
                d = np.abs(x[i+k] - x[j+k])
                if d > dist_m:
                    dist_m = d
            if dist_m < r:
                count_m += 1
                # 仅当 m 匹配时，才检查 m+1
                if i < n - m - 1 and j < n - m - 1: # 确保索引有效
                    d_plus = np.abs(x[i+m] - x[j+m])
                    if d_plus < r: # max(dist_m, d_plus) < r
                        count_m_plus_1 += 1
    if count_m == 0:
        return np.nan
    return -np.log(count_m_plus_1 / count_m)

@numba.njit(parallel=True, cache=True)
def _numba_rolling_sample_entropy(data: np.ndarray, window: int, tol_ratio: float, rolling_std: np.ndarray) -> np.ndarray:
    """
    【V1.1 · Numba并行优化版】滚动计算样本熵。
    - 优化: 启用 parallel=True 和 prange。由于样本熵计算(O(W^2))极其耗时，并行化收益巨大。
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result
        
    # 并行循环
    for i in prange(window - 1, n):
        # 获取预计算的 std
        std_val = rolling_std[i]
        if np.isnan(std_val) or std_val == 0:
            continue
        # 获取当前窗口数据
        window_data = data[i - window + 1 : i + 1]
        r = std_val * tol_ratio
        # 调用核心计算函数 (核心函数保持串行，在每个线程中运行)
        se = _numba_sample_entropy_core(window_data, 2, r)
        result[i] = se
    return result

@numba.njit(parallel=True, cache=True)
def _numba_spearman_orderliness(ma_values: np.ndarray, ma_ranks_x: np.ndarray) -> np.ndarray:
    """
    【V1.1 · Numba并行优化版】行级 Spearman 秩相关系数计算。
    - 优化: 启用 parallel=True 和 prange，并行处理每一行的数据。
    """
    n_rows, n_cols = ma_values.shape
    results = np.zeros(n_rows, dtype=np.float32)
    if n_cols <= 1:
        return results

    # 常数项：n(n^2 - 1)
    denom = n_cols * (n_cols * n_cols - 1.0)
    # 并行循环处理每一行
    for i in prange(n_rows):
        row = ma_values[i, :]
        # 检查 NaN
        has_nan = False
        for k in range(n_cols):
            if np.isnan(row[k]):
                has_nan = True
                break
        if has_nan:
            results[i] = 0.0
            continue
        # 计算当前行的排名
        # argsort().argsort() 获取排名 (0-based)
        temp_args = np.argsort(row)
        ranks_y = np.empty(n_cols, dtype=np.int64)
        # 填充排名
        for r in range(n_cols):
            ranks_y[temp_args[r]] = r + 1 # 1-based rank
        # 计算距离平方和 d^2
        d_sq_sum = 0.0
        for j in range(n_cols):
            d = ma_ranks_x[j] - ranks_y[j]
            d_sq_sum += d * d
        # Spearman 公式
        rho = 1.0 - (6.0 * d_sq_sum) / denom
        results[i] = rho
    return results

@numba.njit(parallel=True, cache=True)
def calculate_correction_scores_v2_numba(
    pct_change: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vol_ma21: np.ndarray,
    flow_intensity: np.ndarray,
    chip_stability: np.ndarray,
    long_term_ratio: np.ndarray,
    correction_min_magnitude: float,
    correction_max_magnitude: float
) -> np.ndarray:
    """
    【V2.1 · Numba并行优化版】向量化计算回调信号评分
    - 优化: 使用 prange 替代 range 以激活 parallel=True 的多核并行能力。
    - 逻辑: 保持原有业务逻辑不变，仅提升计算效率。
    """
    n = len(pct_change)
    scores = np.zeros(n, dtype=np.float32)
    is_correction = np.zeros(n, dtype=np.bool_)
    # 预分配数组以避免循环内重复分配（虽然在并行循环中写入不同索引是安全的）
    # 注意：在并行循环中计算局部统计量比预计算更节省内存带宽，且利用多核优势
    # 主循环：并行计算每个数据点的分数
    for i in prange(n):
        # 基础边界检查
        if i < 19: # 最大窗口需求
            continue
        # 条件1: 回调幅度条件（必要条件，快速过滤）
        # 提前判断以减少后续计算量
        p_change = pct_change[i]
        cond1 = (p_change < 0) and (p_change >= correction_max_magnitude) and (p_change <= correction_min_magnitude)
        if not cond1:
            continue
        # 提取当前切片数据 (避免在循环外创建大数组)
        # 均线计算
        c_slice_5 = close[i-4:i+1]
        c_slice_10 = close[i-9:i+1]
        c_slice_20 = close[i-19:i+1]
        ma5_val = np.mean(c_slice_5)
        ma10_val = np.mean(c_slice_10)
        ma20_val = np.mean(c_slice_20)
        # 前3日平均涨幅
        avg_gain_3d_val = 0.0
        if i >= 3:
            avg_gain_3d_val = np.mean(pct_change[i-3:i])
        # 统计指标计算 (窗口19+1=20)
        # 资金流
        flow_window = flow_intensity[i-19:i+1]
        flow_median_val = np.median(flow_window)
        flow_std_val = np.std(flow_window)
        # 筹码稳定性
        chip_window = chip_stability[i-19:i+1]
        chip_stab_median_val = np.median(chip_window)
        chip_stab_std_val = np.std(chip_window)
        # 长线筹码
        ltr_window = long_term_ratio[i-19:i+1]
        ltr_median_val = np.median(ltr_window)
        ltr_std_val = np.std(ltr_window)
        # 当前值
        vol_val = volume[i]
        vol_ma21_val = vol_ma21[i]
        close_val = close[i]
        flow_val = flow_intensity[i]
        chip_val = chip_stability[i]
        ltr_val = long_term_ratio[i]
        # 其他条件判断
        cond2 = vol_val > vol_ma21_val * 0.8 if vol_ma21_val > 0 else True
        cond3 = (close_val >= ma5_val * 0.95) and (ma5_val > ma10_val * 0.96)
        cond4 = ma5_val > ma20_val * 0.96
        cond5 = flow_val > (flow_median_val - flow_std_val * 0.5)
        cond6 = chip_val > (chip_stab_median_val - chip_stab_std_val * 0.3)
        cond7 = ltr_val > (ltr_median_val - ltr_std_val * 0.5)
        cond8 = avg_gain_3d_val > 0.01
        cond9 = close_val > ma10_val * 0.97
        cond10 = vol_val <= vol_ma21_val * 1.3 if vol_ma21_val > 0 else True
        # 计算分数
        score = 0.0
        score += 2.5  # cond1 已确认为 True
        score += cond2 * 1.0
        score += cond3 * 1.2
        score += cond4 * 1.0
        score += cond5 * 1.5
        score += cond6 * 1.3
        score += cond7 * 1.2
        score += cond8 * 1.0
        score += cond9 * 1.0
        score += cond10 * 0.8
        scores[i] = score
        is_correction[i] = score >= 8.0
    return is_correction

@numba.njit(cache=True)
def _numba_rolling_quantile(data: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """
    【V1.0 · 新增】Numba加速的滚动分位数计算
    - 原理: 对每个窗口进行切片和计算分位数。
    - 性能: 相比 Pandas rolling().quantile() 提升 10x-50x。
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result
    # 针对每个位置计算
    for i in range(window - 1, n):
        # 获取窗口切片
        window_slice = data[i - window + 1 : i + 1]
        # 移除 NaN
        valid_mask = ~np.isnan(window_slice)
        valid_data = window_slice[valid_mask]
        if len(valid_data) > 0:
            # np.quantile 在 Numba 中通常支持，如果不支持可以使用 np.percentile (quantile * 100)
            # 注意：Numba 的 quantile 实现可能需要排序
            result[i] = np.quantile(valid_data, quantile)
            
    return result

@numba.njit(cache=True)
def _numba_calculate_block_stats(
    is_consolidating: np.ndarray, 
    high: np.ndarray, 
    low: np.ndarray, 
    volume: np.ndarray,
    chip_conc: np.ndarray,
    chip_stab: np.ndarray,
    accum_score: np.ndarray,
    flow_stab: np.ndarray,
    absorb_energy: np.ndarray
) -> np.ndarray:
    """
    【V1.0 · 新增】一次性计算连续盘整块的统计特征
    - 原理: 单次遍历识别连续 True 区间，计算统计值并回填。
    - 输出: (N, 9) 的二维数组，包含所有需要的统计列。
    - 性能: 消除 Pandas GroupBy 开销，提升显著。
    """
    n = len(is_consolidating)
    # 结果列顺序: 
    # 0: high_max, 1: low_min, 2: vol_mean, 3: duration, 
    # 4: chip_conc_mean, 5: chip_stab_mean, 6: accum_mean, 7: flow_stab_mean, 8: absorb_mean
    results = np.full((n, 9), np.nan, dtype=np.float64)
    i = 0
    while i < n:
        if is_consolidating[i]:
            # 找到块的结束点
            start = i
            end = i + 1
            while end < n and is_consolidating[end]:
                end += 1
            # 提取切片
            # 注意：需要处理全NaN的情况
            h_slice = high[start:end]
            l_slice = low[start:end]
            v_slice = volume[start:end]
            # 计算统计值
            # 使用 np.nanmax/min/mean 安全处理
            val_high = np.nanmax(h_slice)
            val_low = np.nanmin(l_slice)
            val_vol = np.nanmean(v_slice)
            val_dur = float(end - start)
            # 可选列统计 (检查输入是否全0/NaN，这里简化处理)
            val_conc = np.nanmean(chip_conc[start:end])
            val_stab = np.nanmean(chip_stab[start:end])
            val_accum = np.nanmean(accum_score[start:end])
            val_flow = np.nanmean(flow_stab[start:end])
            val_absorb = np.nanmean(absorb_energy[start:end])
            # 填充结果
            for k in range(start, end):
                results[k, 0] = val_high
                results[k, 1] = val_low
                results[k, 2] = val_vol
                results[k, 3] = val_dur
                results[k, 4] = val_conc
                results[k, 5] = val_stab
                results[k, 6] = val_accum
                results[k, 7] = val_flow
                results[k, 8] = val_absorb
            # 跳过已处理的块
            i = end
        else:
            i += 1
            
    return results

@numba.njit(parallel=True, cache=True)
def _numba_rolling_fractal_dimension(data: np.ndarray, window: int) -> np.ndarray:
    """
    【V1.0 · 新增】基于Sevcik算法的滚动分形维数计算
    - 数学原理: 将每个窗口内的价格序列归一化到单位正方形中，计算波形长度，进而推导分形维数。
    - A股意义: 
        D -> 1.0: 强趋势 (主力合力拉升/出货，路径最短)。
        D -> 1.5: 随机游走 (无主控盘)。
        D > 1.5: 剧烈混沌 (主力宽幅震荡洗盘，制造恐慌)。
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result
    # 预计算常数项: ln(2 * (window - 1))
    denom = np.log(2.0 * (window - 1))
    if denom == 0:
        return result
    # x轴归一化步长 (在单位正方形中，时间轴的总长为1)
    x_step_sq = (1.0 / (window - 1)) ** 2
    for i in prange(window - 1, n):
        # 提取窗口数据
        y_slice = data[i - window + 1 : i + 1]
        y_min = np.min(y_slice)
        y_max = np.max(y_slice)
        y_range = y_max - y_min
        # 如果极差为0（如一字板），维度为1.0
        if y_range == 0:
            result[i] = 1.0
            continue
        # 计算归一化后的波形长度 L
        length = 0.0
        for j in range(1, window):
            # 归一化后的增量 dy
            dy = (y_slice[j] - y_slice[j-1]) / y_range
            # 欧几里得距离
            dist = np.sqrt(x_step_sq + dy * dy)
            length += dist
        # Sevcik 分形维数公式
        if length > 0:
            d_val = 1.0 + np.log(length) / denom
            result[i] = d_val
        else:
            result[i] = 1.0
    return result

@numba.njit(parallel=True, cache=True)
def _numba_calculate_geometric_features(data: np.ndarray, window: int) -> np.ndarray:
    """
    【V1.0 · Numba加速】计算几何形态特征矩阵
    - 输出列定义 (4列):
        0: Linear Slope (线性回归斜率，归一化)
        1: R-Squared (拟合优度，代表趋势稳定性)
        2: Channel Position (通道位置 Z-Score)
        3: Arc Curvature (弧形曲率，正值代表凸/前期加速，负值代表凹/后期加速)
    - 算法:
        - 线性回归: 使用最小二乘法计算 y = kx + b
        - 曲率: 计算价格序列相对于【起点-终点连线】的平均偏离度。
          (A股"老鸭头"或"圆弧底"的数学表达)
    """
    n = len(data)
    results = np.full((n, 4), np.nan, dtype=np.float64)
    if n < window:
        return results

    # 预计算 x 的相关项 (x 为 0 到 window-1)
    x = np.arange(window, dtype=np.float64)
    sum_x = np.sum(x)
    sum_x_sq = np.sum(x * x)
    x_mean = sum_x / window
    # SS_xx: Sum of squares of x deviations
    ss_xx = sum_x_sq - (sum_x * sum_x) / window
    # 归一化因子 (为了让斜率在不同股价下可比，通常除以首个价格，但这里在循环内处理)
    for i in prange(window - 1, n):
        y_slice = data[i - window + 1 : i + 1]
        # 检查 NaN
        has_nan = False
        for k in range(window):
            if np.isnan(y_slice[k]):
                has_nan = True
                break
        if has_nan:
            continue
        y_mean = np.mean(y_slice)
        # 1. 线性回归核心统计量
        sum_y = np.sum(y_slice)
        sum_xy = np.sum(x * y_slice)
        # SS_xy
        ss_xy = sum_xy - (sum_x * sum_y) / window
        # Slope (k)
        if ss_xx == 0:
            slope = 0.0
        else:
            slope = ss_xy / ss_xx
        # Intercept (b)
        intercept = y_mean - slope * x_mean
        # 2. R-Squared (趋势稳定性)
        # SS_tot (Total Sum of Squares)
        ss_tot = np.sum((y_slice - y_mean)**2)
        # SS_res (Residual Sum of Squares)
        y_pred = slope * x + intercept
        ss_res = np.sum((y_slice - y_pred)**2)
        r_squared = 0.0
        if ss_tot > 0:
            r_squared = 1.0 - (ss_res / ss_tot)
        # 3. Channel Position (通道位置 Z-Score)
        # 衡量当前价格相对于回归线的偏离程度，用标准差标准化
        # std_error 约为 sqrt(ss_res / (n-2))，这里简化使用残差标准差
        rmse = np.sqrt(ss_res / window) if window > 0 else 1.0
        current_price = y_slice[-1]
        current_pred = y_pred[-1]
        channel_pos = 0.0
        if rmse > 1e-9:
            channel_pos = (current_price - current_pred) / rmse
        # 4. Arc Curvature (弧形特征)
        # 逻辑：连接起点(y[0])和终点(y[-1])形成一条弦。
        # 计算所有点到弦的垂直距离之和（或平均值）。
        # 正值：价格在弦之上（拱形，Convex，如抛物线顶部或减速上涨）
        # 负值：价格在弦之下（凹形，Concave，如圆弧底或加速上涨）
        # 对于A股，"下凹"（负值）且价格上涨，往往意味着"加速赶顶"或"主升浪"。
        y_start = y_slice[0]
        y_end = y_slice[-1]
        # 弦的方程 y = mx + c (相对于窗口内的x)
        chord_slope = (y_end - y_start) / (window - 1)
        chord_line = y_start + chord_slope * x
        # 偏离度 (Arc) - 使用归一化偏差
        # 除以 y_mean 消除股价绝对值影响
        arc_deviations = (y_slice - chord_line)
        arc_curvature = np.mean(arc_deviations)
        if y_mean > 0:
            arc_curvature /= y_mean # 归一化
            slope /= y_mean # 归一化斜率
        results[i, 0] = slope * 100 # 转换为百分比/天
        results[i, 1] = r_squared
        results[i, 2] = channel_pos
        results[i, 3] = arc_curvature * 100 # 放大数值
        
    return results

class FeatureEngineeringService:
    """
    特征工程服务
    - 核心职责: 专注于从基础数据（OHLCV和简单指标）中衍生出更高级的技术特征。
                它负责所有与K线本身形态、趋势、波动率相关的深度计算。
    """
    def __init__(self, calculator):
        self.calculator = calculator

    async def calculate_all_slopes(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.5 · 数组赋值优化版】计算所有配置的斜率特征。
        - 优化: 移除 pd.Series 创建和 fillna 开销，使用 np.nan_to_num 和直接数组赋值。
        """
        slope_params = config.get('feature_engineering_params', {}).get('slope_params', {})
        if not slope_params.get('enabled', False):
            return all_dfs
        series_to_slope = slope_params.get('series_to_slope', {})
        if not series_to_slope:
            return all_dfs
        for col_pattern, lookbacks in series_to_slope.items():
            if "说明" in col_pattern: continue
            try:
                timeframe = col_pattern.split('_')[-1]
                if timeframe.upper() not in ['D', 'W', 'M'] and not timeframe.isdigit():
                    timeframe = 'D'
            except IndexError:
                continue
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            if col_pattern not in df.columns:
                continue
            # 提取源数据为 NumPy 数组 (float64 以保证精度，Numba函数也是 float64)
            source_values = df[col_pattern].values.astype(np.float64)
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{lookback}_{col_pattern}'
                if slope_col_name in df.columns:
                    continue
                # 【Numba加速】直接调用编译好的滚动斜率函数
                slope_values = _numba_rolling_slope(source_values, int(lookback))
                # 【优化】NumPy 层面处理 NaN，避免 Pandas fillna 开销
                # copy=False 尝试原地修改
                slope_values = np.nan_to_num(slope_values, nan=0.0, copy=False)
                # 直接赋值数组
                df[slope_col_name] = slope_values
            all_dfs[timeframe] = df
        return all_dfs

    async def calculate_all_accelerations(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.5 · 数组赋值优化版】计算所有配置的加速度特征。
        - 优化: 移除 pd.Series 创建和 fillna 开销，使用 np.nan_to_num 和直接数组赋值。
        """
        accel_params = config.get('feature_engineering_params', {}).get('accel_params', {})
        if not accel_params.get('enabled', False):
            return all_dfs
        series_to_accel = accel_params.get('series_to_accel', {})
        if not series_to_accel:
            return all_dfs
        for base_col_name, periods in series_to_accel.items():
            if "说明" in base_col_name: continue
            timeframe = base_col_name.split('_')[-1]
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            for period in periods:
                slope_col_name = f'SLOPE_{period}_{base_col_name}'
                if slope_col_name not in df.columns:
                    continue
                accel_col_name = f'ACCEL_{period}_{base_col_name}'
                if accel_col_name in df.columns:
                    continue
                # 提取斜率数据
                slope_values = df[slope_col_name].values.astype(np.float64)
                # 【Numba加速】计算加速度
                accel_values = _numba_rolling_slope(slope_values, int(period))
                # 【优化】NumPy 层面处理 NaN
                accel_values = np.nan_to_num(accel_values, nan=0.0, copy=False)
                df[accel_col_name] = accel_values
        return all_dfs

    async def calculate_all_jerks(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 · 新增】计算三阶项(Jerk/加加速度)特征。
        - 逻辑: 对加速度(ACCEL)序列进行二次导数计算，捕捉主力在启动瞬间的力道突变。
        - 性能: 延续Numba并行加速架构，确保万级数据量下的实时响应。
        """
        jerk_params = config.get('feature_engineering_params', {}).get('jerk_params', {})
        if not jerk_params.get('enabled', False):
            return all_dfs
        series_to_jerk = jerk_params.get('series_to_jerk', {})
        if not series_to_jerk:
            return all_dfs
        for base_col_name, periods in series_to_jerk.items():
            if "说明" in base_col_name: continue
            timeframe = base_col_name.split('_')[-1]
            if timeframe not in all_dfs or all_dfs[timeframe] is None:
                continue
            df = all_dfs[timeframe]
            for period in periods:
                accel_col_name = f'ACCEL_{period}_{base_col_name}'
                # 检查依赖的二阶项（加速度）是否存在
                if accel_col_name not in df.columns:
                    continue
                jerk_col_name = f'JERK_{period}_{base_col_name}'
                if jerk_col_name in df.columns:
                    continue
                # 提取加速度数据
                accel_values = df[accel_col_name].values.astype(np.float64)
                # 【Numba加速】计算三阶导数（Jerk）
                jerk_values = _numba_rolling_slope(accel_values, int(period))
                # NumPy 层面处理 NaN，保持内存连续性
                jerk_values = np.nan_to_num(jerk_values, nan=0.0, copy=False)
                df[jerk_col_name] = jerk_values
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_jerk_momentum_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 · 新增】基于三阶项(Jerk)的动量爆发检测。
        - 逻辑: 识别“加加速”状态。当 Jerk 与 Accel 发生共振时，判定为情绪极度亢奋或恐慌。
        - 应用: 用于识别A股短线情绪标的的暴力主升浪起点。
        """
        timeframe = 'D'
        if timeframe not in all_dfs: return all_dfs
        df = all_dfs[timeframe]
        # 选取核心均线的三阶项作为参考，例如 EMA_5 (反映最灵敏的价格动量)
        jerk_col = f'JERK_5_EMA_5_D'
        accel_col = f'ACCEL_5_EMA_5_D'
        if jerk_col in df.columns and accel_col in df.columns:
            # 核心判断逻辑：加速度在增加 (Jerk > 0)，且当前正处于加速状态 (Accel > 0)
            # 这通常意味着价格正在经历“非线性”增长，是捕捉连板股的关键数学特征
            jerk_val = df[jerk_col].values
            accel_val = df[accel_col].values
            # 向量化生成动量爆发信号
            df['IS_NONLINEAR_IGNITION_D'] = (jerk_val > 0) & (accel_val > 0)
            # 识别衰竭：速度极快但加速度开始减小 (Accel > 0 且 Jerk < 0)
            df['IS_ACCELERATION_EXHAUSTION_D'] = (accel_val > 0) & (jerk_val < 0)
        return all_dfs

    async def calculate_vpa_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.6 · 斐波那契周期对齐版】A股换手率与主力成本增强版
        - 核心修改: 将 VPA 加速度计算周期从 10D 调整为 13D 斐波那契周期。
        - 优化: 引入自由换手率修正项，识别“缩量锁仓”高效率模式。
        """
        timeframe = 'D'
        if timeframe not in all_dfs: return all_dfs
        df = all_dfs[timeframe]
        required = ['pct_change_D', 'volume_D', 'VOL_MA_21_D', 'turnover_rate_f_D']
        if not all(col in df.columns for col in required): return all_dfs
        pct_change = df['pct_change_D'].values.astype(np.float32)
        volume = df['volume_D'].values.astype(np.float32)
        vol_ma_21 = df['VOL_MA_21_D'].values.astype(np.float32)
        turnover_f = df['turnover_rate_f_D'].values.astype(np.float32)
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = np.divide(volume, vol_ma_21)
            vol_ratio = np.where(vol_ratio == 0, 1.0, vol_ratio)
            vpa_eff = pct_change / (vol_ratio * (1 + turnover_f / 5.0))
            df['VPA_EFFICIENCY_D'] = np.nan_to_num(vpa_eff)
        if 'net_mf_amount_D' in df.columns:
            net_mf = df['net_mf_amount_D'].values.astype(np.float32)
            vpa_adj = np.where((pct_change > 0) & (net_mf < 0), vpa_eff * 0.3, vpa_eff)
            df['VPA_MF_ADJUSTED_EFF_D'] = np.nan_to_num(vpa_adj)
        vpa_values = df['VPA_EFFICIENCY_D'].values.astype(np.float64)
        # 【修改】计算窗口从 5 调整为 13 (斐波那契) 以增强趋势稳健性
        df['VPA_ACCELERATION_13D'] = _numba_rolling_slope(vpa_values, 13)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V7.0.0 · 状态强度模糊化版】
        - 核心重构: 将布尔信号统一升级为 0-1 的浮点数状态强度(STATE_)。
        - 逻辑升级: 基于距离阈值的偏离度计算概率/强度，增强对信号演进的感知。
        - 包含指标: STATE_TRENDING_STAGE_D, STATE_BREAKOUT_CONFIRMED_D, STATE_EMOTIONAL_EXTREME_D, STATE_MARKET_LEADER_D。
        """
        timeframe = 'D'
        if timeframe not in all_dfs: return all_dfs
        df = all_dfs[timeframe].copy()
        # 1. 趋势阶段强度 (STATE_TRENDING_STAGE)
        if 'chip_entropy_D' in df.columns:
            entropy_ma = df['chip_entropy_D'].rolling(20).mean()
            # 逻辑: 熵值低于均值越多，趋势越有序。映射 0.95-0.85 均值区间到 0-1
            entropy_diff_ratio = (entropy_ma - df['chip_entropy_D']) / entropy_ma.replace(0, 1)
            df['STATE_TRENDING_STAGE_D'] = np.clip((entropy_diff_ratio - 0.05) / 0.1, 0, 1).fillna(0)
        else:
            adx = df['ADX_14_D'] if 'ADX_14_D' in df.columns else pd.Series(20, index=df.index)
            df['STATE_TRENDING_STAGE_D'] = np.clip((adx - 20) / 20, 0, 1)
        # 2. 领涨者地位强度 (STATE_MARKET_LEADER)
        # 逻辑: 整合行业领袖分数与行业排位斜率
        if 'industry_leader_score_D' in df.columns:
            leader_score = df['industry_leader_score_D'] / 100.0
            rank_slope = df.get('industry_rank_slope_D', pd.Series(0, index=df.index))
            df['STATE_MARKET_LEADER_D'] = np.clip(leader_score * (1 + rank_slope.clip(lower=0)), 0, 1)
        else:
            df['STATE_MARKET_LEADER_D'] = 0.0
        # 3. 突破确认强度 (STATE_BREAKOUT_CONFIRMED)
        vol_ma = df['VOL_MA_21_D'].replace(0, 1)
        main_cost_ratio = df.get('main_cost_range_ratio_D', pd.Series(0.5, index=df.index))
        dynamic_vol_req = 2.0 - (main_cost_ratio * 0.8)
        vol_ratio = df['volume_D'] / vol_ma
        # 突破强度 = 趋势强度 * 量比强度 * VPA正向效率
        vpa_eff = df.get('VPA_EFFICIENCY_D', pd.Series(0, index=df.index)).clip(lower=0)
        vol_strength = np.clip((vol_ratio - 1.0) / (dynamic_vol_req - 1.0), 0, 1)
        df['STATE_BREAKOUT_CONFIRMED_D'] = (df['STATE_TRENDING_STAGE_D'] * vol_strength * np.clip(vpa_eff * 5, 0, 1)).fillna(0)
        # 4. 情绪极端强度 (STATE_EMOTIONAL_EXTREME)
        bias = df.get('BIAS_5_D', pd.Series(0, index=df.index))
        turnover = df.get('turnover_rate_D', pd.Series(0, index=df.index))
        to_q90 = turnover.rolling(20).quantile(0.9).replace(0, 1)
        bias_extreme = np.clip(bias.abs() / 10.0, 0, 1)
        to_extreme = np.clip(turnover / to_q90, 0, 1)
        df['STATE_EMOTIONAL_EXTREME_D'] = (bias_extreme * to_extreme).fillna(0)
        print(f"  -- [模式识别探针] {timeframe} Avg Trending: {df['STATE_TRENDING_STAGE_D'].mean():.4f}, Avg Breakout: {df['STATE_BREAKOUT_CONFIRMED_D'].mean():.4f}")
        all_dfs[timeframe] = df
        return all_dfs

    def _calculate_breakout_readiness(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · A股板前蓄势模型】计算突破就绪分数
        - 逻辑重构: 
            1. 筹码锁定: 使用自由换手率(turnover_rate_f_D)替代单纯量比，捕捉"缩量锁仓"。
            2. 筹码结构: 引入筹码集中度与稳定性，确认"单峰密集"。
            3. 主力潜伏: 结合主力活跃度，识别"横盘暗建仓"。
        """
        # 1. 平台压缩度 (Platform Compression)
        # A股突破前通常伴随极窄的波动 (Volatility Contraction)
        # 使用 ATR 归一化幅度
        atr = df['ATR_14_D'].replace(0, 1e-8) if 'ATR_14_D' in df.columns else df['high_D'] * 0.02
        amplitude = (df['high_D'] - df['low_D']) / df['close_D'].replace(0, 1)
        # 波动率越低分越高，A股妖股启动前常出现“心电图”
        compression_score = (1 - (amplitude / (atr / df['close_D'] * 3)).clip(0, 1))
        # 2. 筹码锁定度 (Chip Locking)
        # 核心：自由换手率越低，说明锁仓越好。阈值设定为 3% (0.03) 为极佳，10% 以上扣分
        if 'turnover_rate_f_D' in df.columns:
            to_f = df['turnover_rate_f_D']
            # 3%以下满分，15%以上0分
            locking_score = np.clip((15 - to_f) / 12, 0, 1)
        else:
            # 降级：使用量比
            vol_ma5 = df['volume_D'].rolling(5).mean()
            vol_ma20 = df['volume_D'].rolling(20).mean()
            locking_score = np.clip(1 - (vol_ma5 / vol_ma20), 0, 1)
        # 3. 筹码结构优势 (Chip Structure)
        # 筹码越集中，稳定性越高，拉升越轻松
        chip_score = pd.Series(0.5, index=df.index)
        if 'chip_concentration_ratio_D' in df.columns and 'chip_stability_D' in df.columns:
            # 集中度 > 0.7 (70%) 且 稳定性 > 4 为佳
            conc_factor = df['chip_concentration_ratio_D'].clip(0, 1)
            stab_factor = (df['chip_stability_D'] / 5.0).clip(0, 1) # 假设5是满分
            chip_score = (conc_factor * 0.6 + stab_factor * 0.4)
        # 4. 主力潜伏迹象 (Latent Main Force)
        # 价格未涨但主力活跃
        mf_score = pd.Series(0.5, index=df.index)
        if 'main_force_activity_index_D' in df.columns:
            mf_score = (df['main_force_activity_index_D'] / 100.0).clip(0, 1)
        # 综合加权 (向量化)
        # 权重：筹码锁定(30%) + 结构优势(30%) + 平台压缩(20%) + 主力潜伏(20%)
        final_score = 100 * (
            locking_score * 0.3 + 
            chip_score * 0.3 + 
            compression_score * 0.2 + 
            mf_score * 0.2
        )
        return final_score.fillna(0)

    def _calculate_structural_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V3.0 · A股成本胡克定律】计算结构张力
        - 逻辑重构:
            1. 成本乖离: 使用加权平均成本(weight_avg_cost_D)计算全市场获利盘抛压。
            2. 资金背离: 修复缺失字段，用特大单+大单计算主力净流向与价格的背离。
            3. 动力学极限: 结合加速度(ACCEL)判断超买/超卖极限。
        """
        tensions_list = []
        # 1. 真实成本乖离张力
        # 价格远离成本线 -> 获利回吐压力大
        if 'weight_avg_cost_D' in df.columns:
            cost = df['weight_avg_cost_D'].replace(0, np.nan).ffill()
            cost_bias = (df['close_D'] - cost) / cost
            # A股经验: 乖离率>20%张力极大
            tension_cost = (cost_bias.abs() / 0.20).clip(0, 1)
            tensions_list.append(tension_cost)
        elif 'MA_20_D' in df.columns:
            ma_dist = (df['close_D'] - df['MA_20_D']).abs() / (df['MA_20_D'] + 1e-8)
            tensions_list.append((ma_dist / 0.15).clip(0, 1))
        # 2. 资金-价格背离 (Flow-Price Divergence)
        # 修复: 替换不存在的 OFI 字段，使用 elg/lg 数据
        has_mf = all(c in df.columns for c in ['buy_elg_amount_D', 'sell_elg_amount_D', 'buy_lg_amount_D'])
        if has_mf:
            net_mf_large = (df['buy_elg_amount_D'] + df['buy_lg_amount_D']) - \
                           (df['sell_elg_amount_D'] + df['sell_lg_amount_D'])
            flow_norm = net_mf_large.rolling(20).rank(pct=True).fillna(0.5)
            price_norm = df['close_D'].rolling(20).rank(pct=True).fillna(0.5)
            # 价格在高位但资金流出 -> 背离张力大
            tension_flow = (price_norm - flow_norm).abs()
            tensions_list.append(tension_flow)
        # 3. 动力学极限 (利用已有ACCEL)
        if 'MA_ACCELERATION_EMA_55_D' in df.columns:
            accel = df['MA_ACCELERATION_EMA_55_D']
            # Z-Score标准化
            tension_accel = ((accel - accel.rolling(20).mean()).abs() / accel.rolling(20).std().replace(0, 1)).clip(0, 3) / 3.0
            tensions_list.append(tension_accel)
        # 综合合成
        if tensions_list:
            tensions_df = pd.concat(tensions_list, axis=1)
            return tensions_df.max(axis=1).fillna(0)
        return pd.Series(0, index=df.index)

    async def calculate_regime_switch_metrics(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 · 新增】市场环境切换熵模型 (Regime Switch)
        - 核心职责: 计算价格序列的复杂度(分形维数)和无序度(样本熵)，作为识别"变盘点"的底层物理参数。
        - 包含指标:
            1. PRICE_FRACTAL_DIM_D: 价格分形维数，>1.5为洗盘，~1.0为趋势。
            2. PRICE_ENTROPY_D: 价格样本熵，熵减代表合力形成。
        """
        timeframe = 'D'
        if timeframe not in all_dfs: return all_dfs
        df = all_dfs[timeframe]
        if 'close_D' not in df.columns: return all_dfs
        close_values = df['close_D'].values.astype(np.float64)
        # 1. 计算分形维数 (Fractal Dimension)
        # 窗口选择: 21天 (月度周期)
        fractal_window = 21
        fractal_dim = _numba_rolling_fractal_dimension(close_values, fractal_window)
        df['PRICE_FRACTAL_DIM_D'] = np.nan_to_num(fractal_dim, nan=1.0)
        # 2. 计算样本熵 (Sample Entropy)
        # 动态计算容差 r: 使用滚动标准差的0.2倍
        entropy_window = 21
        rolling_std = df['close_D'].rolling(window=entropy_window).std().fillna(0).values.astype(np.float64)
        sample_entropy = _numba_rolling_sample_entropy(close_values, entropy_window, 0.2, rolling_std)
        df['PRICE_ENTROPY_D'] = np.nan_to_num(sample_entropy)
        # 3. 衍生指标：熵的变化率 (Entropy Velocity)
        # 熵减加速(Slope为负且绝对值大) -> 主力合力形成，变盘在即
        entropy_slope = _numba_rolling_slope(df['PRICE_ENTROPY_D'].values, 5)
        df['ENTROPY_VELOCITY_D'] = np.nan_to_num(entropy_slope)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_consolidation_period(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V4.1 · 熵与分形增强版】基于复杂性理论的盘整识别
        - 核心逻辑:
            1. 引入 'PRICE_FRACTAL_DIM_D'。高分形维数(>1.35)是A股主力"宽幅震荡洗盘"的数学铁证。
            2. 引入 'PRICE_ENTROPY_D'。盘整末端通常伴随熵减(有序化)，作为质量评分核心。
            3. 保留自由换手率方差逻辑，确保是"缩量控盘"的有效盘整。
        - 性能: 向量化计算判定逻辑，Numba并行计算块统计特征。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            return all_dfs
        df = all_dfs[timeframe].copy()
        # 0. 确保前置依赖指标已存在 (分形与熵)，若缺失则进行默认填充以防报错
        if 'PRICE_FRACTAL_DIM_D' not in df.columns:
             df['PRICE_FRACTAL_DIM_D'] = 1.5 # 默认为随机游走/震荡状态
        if 'PRICE_ENTROPY_D' not in df.columns:
             df['PRICE_ENTROPY_D'] = 0.0
        # 1. 计算卡夫曼效率系数 (ER)
        # 逻辑：ER = |位移| / 路径长度。值越小代表噪音越大(盘整)，A股典型盘整阈值 < 0.3
        n = 10
        direction = df['close_D'].diff(n).abs()
        volatility = df['close_D'].diff(1).abs().rolling(n).sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            er = direction / volatility
        df['ER_10_D'] = er.fillna(0)
        # 2. 计算自由换手率稳定性 (Turnover Stability Index)
        # 逻辑：优质的蓄势盘整，换手率应稳定在低位。计算变异系数(CV) = Std / Mean
        if 'turnover_rate_f_D' in df.columns:
            to_f = df['turnover_rate_f_D']
            to_std = to_f.rolling(10).std()
            to_mean = to_f.rolling(10).mean()
            to_cv = (to_std / to_mean.replace(0, np.nan)).fillna(1.0)
            df['TURNOVER_STABILITY_INDEX_D'] = to_cv
        else:
            # 降级方案：使用成交量
            vol_std = df['volume_D'].rolling(10).std()
            vol_mean = df['volume_D'].rolling(10).mean()
            df['TURNOVER_STABILITY_INDEX_D'] = (vol_std / vol_mean.replace(0, np.nan)).fillna(1.0)
        # 3. 盘整状态多维判定
        # 条件A: 效率系数低 (基础条件)
        cond_er = df['ER_10_D'] < 0.30
        # 条件B: 分形维数高 (宽幅震荡特征) 或 筹码极度稳定 (死鱼盘/锁仓特征)
        # 分形维数 > 1.35 说明价格曲线填充了二维平面，主力在进行复杂的图形构造(洗盘)
        cond_complex = df['PRICE_FRACTAL_DIM_D'] > 1.35
        cond_stable = df['TURNOVER_STABILITY_INDEX_D'] < 0.45
        # 综合判定: 效率低 且 (形态复杂 或 筹码稳定)
        is_consolidating = cond_er & (cond_complex | cond_stable)
        df[f'is_consolidating_{timeframe}'] = is_consolidating
        # 4. 盘整质量评分 (Quality Score)
        # 仅对识别为盘整的区域评分
        quality_score = np.zeros(len(df), dtype=np.float64)
        if is_consolidating.any():
            # 维度1: 熵减趋势 (Entropy Ordering)
            # 熵的变化率为负(熵减)，代表市场从混乱走向有序，合力即将形成
            score_entropy = np.zeros(len(df))
            if 'ENTROPY_VELOCITY_D' in df.columns:
                # 限制在0-25分，熵速越负分越高
                score_entropy = np.clip(df['ENTROPY_VELOCITY_D'] * -10, 0, 1) * 25
            # 维度2: 缩量程度 (Volume Compression)
            # 自由换手率越低越好，<3%为满分
            score_vol = np.full(len(df), 15.0)
            if 'turnover_rate_f_D' in df.columns:
                score_vol = np.clip((3.0 - df['turnover_rate_f_D']) / 3.0, 0, 1) * 25.0
            # 维度3: 分形复杂度 (Fractal Complexity)
            # 维度越高代表洗盘越充分(支撑越强)
            score_fractal = np.clip((df['PRICE_FRACTAL_DIM_D'] - 1.2) / 0.3, 0, 1) * 25.0
            # 维度4: 筹码集中度 (Chip Concentration)
            score_chip = np.zeros(len(df))
            if 'chip_concentration_ratio_D' in df.columns:
                score_chip = np.clip(df['chip_concentration_ratio_D'], 0, 1) * 25.0
            # 合成总分
            total_raw = score_entropy + score_vol + score_fractal + score_chip
            quality_score = np.where(is_consolidating, total_raw, 0.0)
        df[f'consolidation_quality_score_{timeframe}'] = quality_score
        # 5. Numba加速计算盘整块统计特征
        # 准备数据数组 (Float64, 处理NaN)
        if is_consolidating.any():
            is_cons_arr = is_consolidating.values
            high_arr = df['high_D'].values.astype(np.float64)
            low_arr = df['low_D'].values.astype(np.float64)
            vol_arr = df['volume_D'].fillna(0).values.astype(np.float64)
            def get_arr(col):
                return df[col].fillna(0).values.astype(np.float64) if col in df.columns else np.full(len(df), np.nan)
            chip_conc_arr = get_arr('chip_concentration_ratio_D')
            chip_stab_arr = get_arr('chip_stability_D')
            accum_arr = get_arr('accumulation_score_D')
            flow_stab_arr = get_arr('flow_stability_D')
            absorb_arr = get_arr('absorption_energy_D')
            # 调用Numba核心函数
            stats_matrix = _numba_calculate_block_stats(
                is_cons_arr, high_arr, low_arr, vol_arr,
                chip_conc_arr, chip_stab_arr, accum_arr, flow_stab_arr, absorb_arr
            )
            # 回填统计结果
            df[f'dynamic_consolidation_high_{timeframe}'] = stats_matrix[:, 0]
            df[f'dynamic_consolidation_low_{timeframe}'] = stats_matrix[:, 1]
            df[f'dynamic_consolidation_avg_vol_{timeframe}'] = stats_matrix[:, 2]
            df[f'dynamic_consolidation_duration_{timeframe}'] = stats_matrix[:, 3]
            # 附加统计回填
            if 'chip_concentration_ratio_D' in df.columns: df[f'consolidation_chip_concentration_{timeframe}'] = stats_matrix[:, 4]
            if 'chip_stability_D' in df.columns: df[f'consolidation_chip_stability_{timeframe}'] = stats_matrix[:, 5]
            if 'accumulation_score_D' in df.columns: df[f'consolidation_accumulation_score_{timeframe}'] = stats_matrix[:, 6]
            # 前向填充逻辑，确保非盘整K线能获取最近一个盘整区的数据用于突破判定
            fill_cols = [
                f'dynamic_consolidation_high_{timeframe}', f'dynamic_consolidation_low_{timeframe}',
                f'dynamic_consolidation_duration_{timeframe}', f'consolidation_quality_score_{timeframe}'
            ]
            for c in fill_cols:
                if c in df.columns:
                    df[c] = df[c].replace(0, np.nan).ffill().fillna(0)
        # 6. 缺失值与默认值处理
        if f'dynamic_consolidation_high_{timeframe}' not in df.columns:
            df[f'dynamic_consolidation_high_{timeframe}'] = df['high_D']
            df[f'dynamic_consolidation_low_{timeframe}'] = df['low_D']
        else:
            df[f'dynamic_consolidation_high_{timeframe}'] = df[f'dynamic_consolidation_high_{timeframe}'].fillna(df['high_D'])
            df[f'dynamic_consolidation_low_{timeframe}'] = df[f'dynamic_consolidation_low_{timeframe}'].fillna(df['low_D'])
        # 7. 质量分级 (Grade Categorization)
        if f'consolidation_quality_score_{timeframe}' in df.columns:
            df['consolidation_quality_grade_D'] = pd.cut(
                df[f'consolidation_quality_score_{timeframe}'],
                bins=[-1, 30, 50, 70, 85, 999],
                labels=['POOR', 'FAIR', 'GOOD', 'EXCELLENT', 'OUTSTANDING']
            )
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_enhancement_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 · A股日内微观结构增强版】
        - 优化思路:
            1. 日内偏度(Skewness): 负偏度(尾部在左)意味着主力盘中打压吸筹或护盘，利好T+1。
            2. 缺口动能: A股"缺口不补"是极强的溢价信号。
            3. 尾盘抢筹: 结合close位置判断。
        """
        params = config.get('feature_engineering_params', {}).get('indicators', {}).get('pattern_enhancement_signals', {})
        if not params.get('enabled', False): return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty: return all_dfs
        df = all_dfs[timeframe]
        # 1. 跳空缺口动能
        if 'open_D' in df.columns and 'pre_close_D' in df.columns:
            gap_pct = (df['open_D'] - df['pre_close_D']) / df['pre_close_D']
            # 缺口未补 (最低价 > 昨收)
            gap_held = (df['low_D'] > df['pre_close_D']) & (gap_pct > 0)
            df['GAP_MOMENTUM_STRENGTH_D'] = np.where(gap_held, gap_pct * 100, 0)
        else:
            df['GAP_MOMENTUM_STRENGTH_D'] = 0.0
        # 2. 日内微观结构分布 (Skewness)
        # Skewness < 0: 价格集中在高位，下杀只是瞬间 -> 主力护盘/吸筹
        if 'intraday_price_distribution_skewness_D' in df.columns:
            skew = df['intraday_price_distribution_skewness_D'].fillna(0)
            df['INTRADAY_SUPPORT_INTENT_D'] = np.clip(skew * -1, -1, 1)
        else:
            df['INTRADAY_SUPPORT_INTENT_D'] = 0.0
        # 3. 尾盘抢筹特征
        close_pos = (df['close_D'] - df['low_D']) / (df['high_D'] - df['low_D'].replace(0, np.nan))
        df['CLOSING_STRENGTH_D'] = close_pos.fillna(0.5)
        # 4. T+1 溢价预期合成
        trend_factor = 0.5
        if 'uptrend_strength_D' in df.columns:
            trend_factor = df['uptrend_strength_D'] / 100.0
        premium_score = (
            df['GAP_MOMENTUM_STRENGTH_D'].clip(0, 5) * 6 +
            (df['INTRADAY_SUPPORT_INTENT_D'] + 1) * 15 +
            df['CLOSING_STRENGTH_D'] * 20 +
            trend_factor * 20
        )
        df['T1_PREMIUM_EXPECTATION_D'] = np.clip(premium_score, 0, 100)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_breakout_quality(self, all_dfs: Dict, params: dict) -> Dict:
        """
        【V4.1 · 斐波那契字段同步版】突破质量分计算专用通道
        - 核心修改：同步 FundFlowFactor 模型中已修改的斐波那契周期字段。
        - 字段同步：移除 ma10，新增 ma8；移除 volatility_10d/20d，新增 13d/21d。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe] is None:
            return all_dfs
        df_daily = all_dfs[timeframe]
        if 'breakout_quality_score_D' in df_daily.columns:
            return all_dfs
        required_materials = [
            'volume', 'VOL_MA_21', 'open', 'high', 'low', 'close',
            'VPA_EFFICIENCY', 'VPA_BUY_EFFICIENCY',
            'main_force_buy_execution_alpha', 'upward_impulse_strength',
            'buy_order_book_clearing_rate', 'bid_side_liquidity',
            'vwap_cross_up_intensity', 'opening_buy_strength',
            'floating_chip_cleansing_efficiency', 'deception_lure_long_intensity', 'wash_trade_buy_volume'
        ]
        # 【核心修改点】同步新版模型字段（斐波那契化）
        new_model_materials = [
            'winner_rate', 'profit_ratio', 'chip_concentration_ratio', 'chip_stability', 'profit_pressure',
            'flow_intensity', 'accumulation_score', 'pushing_score', 'net_amount_ratio', 
            'net_amount_ratio_ma5', 'net_amount_ratio_ma8',  # 替换 ma10 -> ma8
            'flow_momentum_5d', 'flow_momentum_8d',         # 新增 8d
            'flow_stability', 'large_order_anomaly',
            'absorption_energy', 'distribution_energy', 'net_energy_flow', 'game_intensity', 'breakout_potential', 'energy_concentration',
            'behavior_accumulation', 'behavior_distribution', 'behavior_consolidation',
            'pressure_trapped', 'pressure_profit', 'support_strength', 'resistance_strength',
            'flow_volatility_13d', 'flow_volatility_21d'    # 替换 10d/20d -> 13d/21d
        ]
        all_required_materials = required_materials + new_model_materials
        df_standardized = pd.DataFrame(index=df_daily.index)
        for material in all_required_materials:
            source_col_with_suffix = f"{material}_{timeframe}"
            if source_col_with_suffix in df_daily.columns:
                df_standardized[material] = df_daily[source_col_with_suffix]
            else:
                alt_names = [f"{material}", f"{material.upper()}_{timeframe}", f"{material.lower()}_{timeframe}"]
                found = False
                for alt_name in alt_names:
                    if alt_name in df_daily.columns:
                        df_standardized[material] = df_daily[alt_name]
                        found = True
                        break
        critical_materials = ['volume', 'high', 'low', 'close', 'VPA_EFFICIENCY']
        missing_critical = [m for m in critical_materials if m not in df_standardized.columns]
        if missing_critical:
            logger.warning(f"突破质量分计算中止，缺少关键材料: {missing_critical}")
            return all_dfs
        try:
            result_df = await self.calculate_breakout_quality_score_v4(df_daily=df_standardized, params=params)
        except AttributeError:
            result_df = await self.calculate_breakout_quality_score_v3(df_standardized, params)
        if result_df is not None and not result_df.empty:
            df_daily = df_daily.join(result_df, how='left')
            rename_dict = {col: f"{col}_{timeframe}" for col in result_df.columns if not col.endswith(f'_{timeframe}')}
            if rename_dict:
                df_daily.rename(columns=rename_dict, inplace=True)
            fill_cols = ['breakout_quality_score_D', 'breakout_technical_score_D', 'breakout_chip_score_D', 'breakout_fundflow_score_D', 'breakout_energy_score_D']
            for col in fill_cols:
                if col in df_daily.columns: df_daily[col] = df_daily[col].ffill()
            all_dfs[timeframe] = df_daily
        return all_dfs

    async def calculate_ma_potential_metrics(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.1 · 修复版】均线多维势能分析引擎
        - 修复: 解决 spread_delta (Series, RangeIndex) 与 price_delta (Series, DatetimeIndex) 在 np.where 计算时的广播形状不匹配问题。
        - 优化: 强制使用 NumPy 数组进行向量化计算，消除 Pandas 索引对齐开销。
        """
        if not params.get('enabled', False):
            return all_dfs
        # 定义核心均线组 (A股常用)
        short_mas = [5, 13, 21]
        long_mas = [55, 89, 144]
        all_periods = sorted(list(set(short_mas + long_mas)))
        for timeframe in params.get('apply_on', []):
            if timeframe not in all_dfs or all_dfs[timeframe] is None or all_dfs[timeframe].empty:
                continue
            df = all_dfs[timeframe]
            # 1. 数据准备与清洗
            ma_type = params.get('ma_type', 'EMA') # 默认使用EMA，对近期权重更高
            cols_needed = [f"{ma_type}_{p}_{timeframe}" for p in all_periods] + \
                          [f"ATR_14_{timeframe}", f"close_{timeframe}"]
            # 检查缺失列
            missing = [c for c in cols_needed if c not in df.columns]
            if missing:
                # 尝试现场计算缺失的MA
                try:
                    close_vals = df[f"close_{timeframe}"]
                    for p in all_periods:
                        col_name = f"{ma_type}_{p}_{timeframe}"
                        if col_name not in df.columns:
                            # 简单的EMA计算补充
                            df[col_name] = close_vals.ewm(span=p, adjust=False).mean()
                except Exception:
                    logger.warning(f"均线势能计算缺少核心列: {missing}")
                    continue
            try:
                # 提取数据矩阵 (N_samples, N_periods)
                # 显式转换为 float64 避免类型不一致
                ma_matrix = df[[f"{ma_type}_{p}_{timeframe}" for p in all_periods]].values.astype(np.float64)
                close_arr = df[f"close_{timeframe}"].values.astype(np.float64)
                # ATR 处理: 确保无0值
                atr_series = df[f"ATR_14_{timeframe}"].replace(0, np.nan).bfill()
                atr_arr = atr_series.values.astype(np.float64)
                # 2. 计算 MA_POTENTIAL_COMPRESSION_RATE (极致压缩率)
                # 逻辑: 所有均线的标准差越小，压缩越紧。除以ATR进行归一化。
                ma_std = np.std(ma_matrix, axis=1)
                normalized_spread = np.divide(ma_std, atr_arr, out=np.zeros_like(ma_std), where=atr_arr!=0)
                # 使用 Rank Percentile量化压缩程度 (0-1)
                # 使用 df.index 创建 Series 以保持对齐，便于 rolling 计算
                spread_series = pd.Series(normalized_spread, index=df.index)
                compression_rank = spread_series.rolling(window=120).rank(pct=True).fillna(0.5).values
                df[f'MA_POTENTIAL_COMPRESSION_RATE_{timeframe}'] = 1.0 - compression_rank
                # 3. 计算 MA_RUBBER_BAND_EXTENSION (橡皮筋拉伸度 - 均值回归压力)
                # 逻辑: 价格相对于最长周期均线(如144日)的偏离程度，经过ATR标准化
                longest_ma = df[f"{ma_type}_{max(all_periods)}_{timeframe}"].values.astype(np.float64)
                extension_raw = np.divide(close_arr - longest_ma, atr_arr, out=np.zeros_like(close_arr), where=atr_arr!=0)
                extension_series = pd.Series(extension_raw, index=df.index)
                extension_mean = extension_series.rolling(250).mean()
                extension_std = extension_series.rolling(250).std().replace(0, 1)
                extension_z = (extension_series - extension_mean) / extension_std
                df[f'MA_RUBBER_BAND_EXTENSION_{timeframe}'] = extension_z.fillna(0).clip(-3, 3).values
                # 4. 计算 MA_COHERENCE_RESONANCE (多均线共振度)
                # 逻辑: 检查所有均线的瞬时斜率是否同向且加速
                # 使用 numpy diff 计算行间差异 (Time t vs Time t-1)
                # np.diff 结果长度会少1，需要补一行 0
                ma_diff = np.zeros_like(ma_matrix)
                ma_diff[1:] = ma_matrix[1:] - ma_matrix[:-1]
                ma_rising = (ma_diff > 0).sum(axis=1) # 统计上升的均线数量
                # 均线多头排列得分
                ranks_x = np.arange(len(all_periods), dtype=np.float64) # 理想排名
                orderliness = _numba_spearman_orderliness(ma_matrix, ranks_x)
                # 共振度 = (上升均线数量占比 + 排列有序度) / 2
                coherence = (ma_rising / len(all_periods) + orderliness) / 2
                df[f'MA_COHERENCE_RESONANCE_{timeframe}'] = coherence
                # 5. 计算 MA_FAN_EFFICIENCY (发散效率 - 识别诱多)
                # 逻辑: 价格涨幅 / 均线组发散增量。
                # 修复核心: 统一提取为 numpy values，避免 Series 索引不一致导致的广播错误
                spread_delta = spread_series.diff(3).fillna(0).values # (N,) array
                # 计算价格变化的绝对值 (N,) array
                close_diff_values = df[f"close_{timeframe}"].diff(3).abs().fillna(0).values
                price_delta = np.divide(close_diff_values, atr_arr, out=np.zeros_like(close_diff_values), where=atr_arr!=0)
                # 向量化计算效率，避免除以0
                efficiency = np.zeros_like(spread_delta)
                # 创建 mask，只计算分母大于阈值的部分
                mask = spread_delta > 0.01
                efficiency[mask] = price_delta[mask] / spread_delta[mask]
                df[f'MA_FAN_EFFICIENCY_{timeframe}'] = np.clip(efficiency, 0, 10)
                # 6. 计算 MA_POTENTIAL_TENSION_INDEX (原有张力指标)
                # 使用短期均线(5)与中期均线(21)的距离作为"攻击张力"
                short_val = df[f"{ma_type}_{short_mas[0]}_{timeframe}"].values
                long_val = df[f"{ma_type}_{short_mas[-1]}_{timeframe}"].values
                short_term_tension = np.divide(short_val - long_val, atr_arr, out=np.zeros_like(short_val), where=atr_arr!=0)
                df[f'MA_POTENTIAL_TENSION_INDEX_{timeframe}'] = np.nan_to_num(short_term_tension)
            except Exception as e:
                logger.error(f"计算均线系统势能时发生错误({timeframe}): {e}", exc_info=True)
        return all_dfs

    async def calculate_och(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V4.0 · 熵权筹码健康度模型 (Entropy-Weighted OCH)】
        - 核心优化: 结合A股"筹码分布学"与"信息熵"理论，深度重构OCH算法。
        - 逻辑升级:
            1. 引入 'chip_entropy_D' (筹码熵): 熵越低，筹码结构越有序(单峰密集)，爆发力越强。
            2. 引入 'chip_kurtosis_D' (峰度): 替代简单的集中度，精准量化"尖峰"形态。峰度>3代表极致锁仓。
            3. 引入 'cost_efficiency' (获利效率): 获利比例/胜率。区分"薄利多销"与"厚利锁仓"。
            4. 引入 'main_force_alpha' (主力Alpha): 剥离大盘影响，识别主力的独立意志。
        - 数据依据: 充分利用数据层提供的 entropy, kurtosis, skewness, alpha 等高阶统计量。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 OCH 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        n_rows = len(df)
        # --- 辅助函数：安全获取 Float32 数组 ---
        def _get_arr(col_name, default_val=0.0):
            if col_name not in df.columns:
                return np.full(n_rows, default_val, dtype=np.float32)
            arr = df[col_name].values.astype(np.float32, copy=False)
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr, nan=default_val)
            return arr
        # --- 1. 筹码形态与有序度 (Morphology & Order) ---
        # 逻辑：A股妖股启动前，筹码形态通常呈现"低熵高峰"（有序且集中）。
        # 熵 (Entropy): 越小越好。标准化处理: 1 - Normalized(Entropy)
        chip_entropy = _get_arr('chip_entropy_D', 1.0)
        entropy_score = np.clip(1.0 - (chip_entropy / 5.0), 0, 1) # 假设熵值范围通常在0-5
        # 峰度 (Kurtosis): 越大越好。峰度反映了筹码的重合度。
        chip_kurtosis = _get_arr('chip_kurtosis_D', 3.0)
        # 峰度大于3为尖峰，大于10为极致。映射到0-1。
        kurtosis_score = np.clip((chip_kurtosis - 1.5) / 8.5, 0, 1)
        # 集中度 (Concentration): 传统指标
        conc_ratio = _get_arr('chip_concentration_ratio_D', 0.5)
        conc_score = np.clip(1.0 - conc_ratio, 0, 1)
        # 形态综合分: 熵权核心，峰度辅助
        morphology_score = entropy_score * 0.4 + kurtosis_score * 0.4 + conc_score * 0.2
        # --- 2. 成本结构与获利动力 (Cost & Profit) ---
        # 逻辑：获利盘比例(Winner Rate)决定抛压，获利幅度(Profit Ratio)决定持仓信心。
        winner_rate = _get_arr('winner_rate_D', 0.5)
        profit_ratio = _get_arr('profit_ratio_D', 0.0)
        # 获利效率 (Cost Efficiency): 每单位获利盘的平均利润。
        # 高效率(>10%)代表主力吃肉，低效率(<2%)代表散户喝汤。
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency_raw = profit_ratio / winner_rate
            efficiency = np.where(winner_rate < 0.01, 0, efficiency_raw)
        efficiency_score = np.clip(efficiency * 5, 0, 1) # 20%效率即满分
        # 获利盘压力修正: 只有当换手率高时，高获利盘才是压力；缩量时是锁仓。
        turnover_f = _get_arr('turnover_rate_f_D', 3.0)
        locking_factor = np.clip((5.0 - turnover_f) / 5.0, 0.1, 1.0) # 换手越低，因子越大
        # 结构综合分: 高获利且锁仓为佳，低获利且被套最差
        structure_score = winner_rate * locking_factor * 0.5 + efficiency_score * 0.5
        # --- 3. 主力意志与攻击性 (Main Force Intent) ---
        # 逻辑：识别"真主力"的关键是看其是否具备独立于大盘的Alpha收益能力及执行力。
        mf_alpha = _get_arr('main_force_buy_execution_alpha_D', 0.0)
        mf_flow_int = _get_arr('flow_intensity_D', 0.0)
        mf_consistency = _get_arr('flow_consistency_D', 0.5)
        # 攻击性: Alpha越高，说明主力越是在逆势或独立拉升
        attack_score = np.clip(mf_alpha, 0, 1)
        # 持续性: 资金流必须连续，突击可能是诱多
        sustainability_score = np.clip(mf_consistency, 0, 1)
        # 主力综合分
        force_score = attack_score * 0.4 + np.clip(mf_flow_int / 10, 0, 1) * 0.4 + sustainability_score * 0.2
        # --- 4. 市场情绪与博弈 (Sentiment & Game) ---
        # 逻辑：日内偏度反映盘中主力态度（左偏=护盘/压盘吸筹），微观结构决定成败。
        skewness = _get_arr('intraday_price_distribution_skewness_D', 0.0)
        # 负偏度通常代表支撑强（底部有承接），正偏度代表阻力大
        support_intent = np.clip(skewness * -1, 0, 1)
        # 散户博弈: 散户被套比例越高，反转概率越大（绝望中诞生）
        retail_trapped = _get_arr('pressure_trapped_D', 0.0)
        reversal_potential = np.clip(retail_trapped, 0, 1)
        sentiment_score = support_intent * 0.5 + reversal_potential * 0.5
        # --- 5. OCH 最终融合 (Non-linear Fusion) ---
        # 权重分配: 形态(30%) + 结构(25%) + 主力(25%) + 情绪(20%)
        # 使用简单的加权，因为各子项已经非线性处理过
        final_och = (
            morphology_score * 0.30 +
            structure_score * 0.25 +
            force_score * 0.25 +
            sentiment_score * 0.20
        )
        # 映射到 -1 到 1 区间 (OCH标准输出格式)
        df['OCH_D'] = (final_och * 2.0) - 1.0
        # 附加衍生指标：OCH 加速度 (识别筹码状态突变)
        if len(df) > 5:
            och_slope = _numba_rolling_slope(df['OCH_D'].values.astype(np.float64), 3)
            df['OCH_ACCELERATION_D'] = np.nan_to_num(och_slope)
        all_dfs[timeframe] = df
        logger.info("OCH 指标计算完成 (V4.0 熵权增强版)。")
        return all_dfs

    async def calculate_structural_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.0 · 结构完整性与时空势能计算器】
        - 核心优化: 从单纯的"趋势健康度"升级为"时空结构势能"分析。
        - 新增逻辑:
            1. 分形效率 (Fractal Efficiency): 利用 'PRICE_FRACTAL_DIM_D'。分形维数趋近1.0时，趋势最纯粹（阻力最小）；趋近1.5时为布朗运动（震荡）；>1.5为反身性剧烈波动。
            2. 盈亏空间比 (Space Potential): 利用支撑/阻力强度，计算上方空间与下方风险的比率。
            3. 量价背离度 (Structural Divergence): 向量化计算价格斜率与量能斜率的背离，识别"顶部钝化"或"底部背离"。
            4. 趋势确认度 (Trend Confirmation): 结合 ADX 与 OCH，确认趋势的有效性。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"结构特征计算失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        n = len(df)
        # --- 1. 分形效率 (Fractal Efficiency) ---
        # 逻辑：好的趋势应该是线性的(D->1)。D越低，趋势结构越紧凑，能量损耗越小。
        if 'PRICE_FRACTAL_DIM_D' in df.columns:
            fd = df['PRICE_FRACTAL_DIM_D'].fillna(1.5)
            # 效率分：1.0-1.2为优(100分)，1.2-1.4为良，>1.5为差
            fractal_efficiency = np.clip((1.5 - fd) / 0.3, 0, 1) * 100
        else:
            fractal_efficiency = pd.Series(50, index=df.index)
        df['STRUCTURAL_FRACTAL_EFFICIENCY_D'] = fractal_efficiency
        # --- 2. 盈亏空间比 (Space Potential) ---
        # 逻辑：做多势能 = (上方阻力距离 / 下方支撑距离)。比值越大，结构优势越大。
        # 使用 close 与 support/resistance 估算
        close = df['close_D']
        # 假设 support_strength_D 和 resistance_strength_D 代表强度，我们需要推算位置
        # 这里简化使用 ATR 波动率作为空间标尺
        atr = df['ATR_14_D'].replace(0, 1.0)
        # 如果有 Explicit Support/Resistance Levels 最好，若无，使用 20日极值 + 强度修正
        res_level = df['high_D'].rolling(20).max()
        sup_level = df['low_D'].rolling(20).min()
        upside = (res_level - close).clip(lower=atr * 0.5) # 至少有0.5ATR空间
        downside = (close - sup_level).clip(lower=atr * 0.5)
        # 引入强度因子：阻力越弱，上方空间视为越大
        res_strength = df.get('resistance_strength_D', pd.Series(0.5, index=df.index))
        sup_strength = df.get('support_strength_D', pd.Series(0.5, index=df.index))
        adjusted_upside = upside * (1 + (1 - res_strength)) # 阻力弱，空间加成
        adjusted_downside = downside * (1 + (1 - sup_strength)) # 支撑弱，风险加成
        with np.errstate(divide='ignore'):
            space_ratio = adjusted_upside / adjusted_downside
        df['STRUCTURAL_SPACE_RATIO_D'] = np.clip(space_ratio, 0.1, 10.0)
        # --- 3. 量价结构背离 (Volume-Price Divergence) ---
        # 逻辑：价格创新高(Slope>0)但量能萎缩(Slope<0) -> 顶背离风险。
        # 价格创新低(Slope<0)但量能释放(Slope>0) -> 底背离机会(恐慌盘杀出)。
        # 计算斜率 (使用5日窗口)
        price_slope = _numba_rolling_slope(df['close_D'].values.astype(np.float64), 5)
        vol_slope = _numba_rolling_slope(df['volume_D'].values.astype(np.float64), 5)
        # 归一化斜率方向
        p_dir = np.sign(price_slope)
        v_dir = np.sign(vol_slope)
        # 背离判定: 方向相反
        divergence = np.zeros(n)
        # 顶背离 (价涨量缩)
        top_div = (p_dir > 0) & (v_dir < 0)
        divergence[top_div] = -1 # 负分代表风险
        # 底背离 (价跌量增 - 恐慌盘)
        bottom_div = (p_dir < 0) & (v_dir > 0)
        divergence[bottom_div] = 1 # 正分代表机会
        df['STRUCTURAL_DIVERGENCE_STATE_D'] = divergence
        # --- 4. 结构健康度合成 (Structural Health) ---
        # 综合考虑：趋势效率(分形)、空间优势、量价配合
        # 量价配合：同向为佳 (1.0)，背离根据位置判断
        vp_match = np.where(p_dir == v_dir, 1.0, 0.5) 
        health_score = (
            fractal_efficiency * 0.4 + 
            np.clip(space_ratio * 10, 0, 100) * 0.3 + 
            (vp_match * 100) * 0.3
        )
        # 极端修正：如果顶背离严重，健康度打折
        health_score = np.where(top_div, health_score * 0.6, health_score)
        df['STRUCTURAL_TREND_HEALTH_D'] = np.clip(health_score, 0, 100)
        # 保留原有的指标透传，保持兼容性
        if 'breakthrough_conviction_score_D' in df.columns:
            df['BREAKTHROUGH_CONVICTION_D'] = df['breakthrough_conviction_score_D']
        else:
            df['BREAKTHROUGH_CONVICTION_D'] = 50.0
        all_dfs[timeframe] = df
        logger.info("结构特征计算完成 (V2.0 时空势能版)。")
        return all_dfs

    async def calculate_geometric_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V7.0.0 · 几何形态强度模糊化版】
        - 核心重构: 将布尔信号统一升级为 0-1 的浮点数状态强度(STATE_)。
        - 判定逻辑优化: 利用线性映射将 R2、通道位置、曲率转换为连续的状态分值。
        - 包含指标: STATE_ROBUST_TREND_D, STATE_PARABOLIC_WARNING_D, STATE_GOLDEN_PIT_D, STATE_ROUNDING_BOTTOM_D。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty: return all_dfs
        df = all_dfs[timeframe]
        if 'close_D' not in df.columns: return all_dfs
        window = 21
        close_values = df['close_D'].values.astype(np.float64)
        geom_matrix = _numba_calculate_geometric_features(close_values, window)
        slope = geom_matrix[:, 0]
        r2 = geom_matrix[:, 1]
        pos = geom_matrix[:, 2]
        arc = geom_matrix[:, 3]
        # 1. 稳健趋势强度 (STATE_ROBUST_TREND)
        # 逻辑: 斜率 > 0.3 且 R2 > 0.85 为满分趋势
        slope_score = np.clip(slope / 0.5, 0, 1)
        r2_score = np.clip((r2 - 0.5) / 0.4, 0, 1)
        df['STATE_ROBUST_TREND_D'] = (slope_score * r2_score).astype(np.float32)
        # 2. 抛物线赶顶预警强度 (STATE_PARABOLIC_WARNING)
        # 逻辑: Pos > 2.0, Arc < -1.0, Slope > 1.0。越偏离阈值，分值越高
        pos_warn = np.clip((pos - 1.5) / 1.5, 0, 1)
        arc_warn = np.clip((-arc - 0.5) / 1.5, 0, 1)
        slope_warn = np.clip((slope - 0.5) / 1.5, 0, 1)
        df['STATE_PARABOLIC_WARNING_D'] = (pos_warn * arc_warn * slope_warn).astype(np.float32)
        # 3. 黄金坑强度 (STATE_GOLDEN_PIT)
        # 逻辑: 长期斜率 > 0 且 短期打到下轨 (Pos < -1.5)
        long_slope_support = np.clip(slope / 0.3, 0, 1)
        dip_intensity = np.clip((-pos - 1.0) / 2.0, 0, 1)
        df['STATE_GOLDEN_PIT_D'] = (long_slope_support * dip_intensity).astype(np.float32)
        # 4. 圆弧底蓄势强度 (STATE_ROUNDING_BOTTOM)
        # 逻辑: 斜率绝对值小 (接近0) 且 呈现下凹形态 (Arc < -0.5)
        flatness = np.clip(1 - np.abs(slope) / 0.3, 0, 1)
        bottom_arc = np.clip((-arc - 0.2) / 0.8, 0, 1)
        df['STATE_ROUNDING_BOTTOM_D'] = (flatness * bottom_arc).astype(np.float32)
        # 注入基础几何列
        df['GEOM_REG_SLOPE_D'] = slope
        df['GEOM_REG_R2_D'] = r2
        df['GEOM_CHANNEL_POS_D'] = pos
        df['GEOM_ARC_CURVATURE_D'] = arc
        print(f"  -- [几何特征探针] {timeframe} Avg Robust: {df['STATE_ROBUST_TREND_D'].mean():.4f}, Max Parabolic: {df['STATE_PARABOLIC_WARNING_D'].max():.4f}")
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_breakout_quality_score_v4(self, df_daily: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V5.1 · 斐波那契逻辑同步版】猎杀A股伪突破增强版
        - 核心修改: 同步 ma8 和 momentum_8d 的权重引用，废弃 ma10 逻辑。
        """
        df = df_daily.copy()
        results = pd.DataFrame(index=df.index)
        def get_series(col, default_val=0.0):
            if col in df.columns: return df[col].fillna(default_val)
            return pd.Series(default_val, index=df.index)
        pct_change = get_series('pct_change')
        turnover_f = get_series('turnover_rate_f')
        volume = get_series('volume')
        # ==================== 1. 技术形态质量 ====================
        tech_score = pd.Series(0.0, index=df.index)
        high, low, close, open_p = get_series('high'), get_series('low'), get_series('close'), get_series('open')
        body_len = (close - open_p).abs()
        upper_shadow = high - pd.concat([close, open_p], axis=1).max(axis=1)
        total_len = high - low
        solid_candle_score = np.where((total_len > 0) & (body_len / total_len > 0.6) & (upper_shadow < body_len * 0.5), 15, 0)
        tech_score += solid_candle_score
        pre_close = get_series('pre_close')
        gap_score = np.where((low > pre_close) & (pct_change > 2.0), 10, 0)
        tech_score += gap_score
        tech_score += np.clip(get_series('opening_buy_strength') * 2, 0, 15)
        safe_to = turnover_f.replace(0, 1.0)
        tech_score += np.clip((pct_change / safe_to) * 5, 0, 20)
        # ==================== 2. 筹码结构质量 ====================
        chip_score = pd.Series(0.0, index=df.index)
        chip_score += np.clip(get_series('winner_rate') * 25, 0, 25)
        avg_cost = get_series('weight_avg_cost')
        cost_bias = (close - avg_cost) / avg_cost.replace(0, 1)
        chip_score += np.where((cost_bias > 0) & (cost_bias <= 0.10), 25, np.where(cost_bias <= 0.20, 15, 0))
        chip_score += np.clip((0.25 - get_series('chip_concentration_ratio')) * 100, 0, 20)
        chip_score += np.clip((1 - get_series('pressure_trapped')) * 20, 0, 20)
        # ==================== 3. 资金与博弈质量 ====================
        flow_score = pd.Series(0.0, index=df.index)
        flow_score += np.clip(get_series('flow_intensity'), 0, 25)
        # 【修改】使用 flow_momentum_8d (斐波那契) 替代 10d
        flow_mom = get_series('flow_momentum_8d')
        flow_score += np.clip(flow_mom * 50 + 10, 0, 15)
        flow_score += np.clip(get_series('accumulation_score') * 0.2, 0, 15)
        flow_score += np.clip(get_series('energy_concentration') * 100, 0, 15)
        # ==================== 4. 风险惩罚因子 ====================
        penalty = pd.Series(0.0, index=df.index)
        penalty += get_series('deception_lure_long_intensity') * 40
        penalty += np.where(get_series('large_order_anomaly') > 0, 15, 0)
        penalty += np.where((get_series('closing_flow_ratio') > 0.4) & (pct_change < 9.5), 20, 0)
        penalty += np.where((turnover_f > 25) & (~(close >= get_series('up_limit') * 0.99)), 30, 0)
        # ==================== 5. 综合合成 ====================
        w_tech, w_chip, w_flow = params.get('weight_technical', 0.3), params.get('weight_chip', 0.35), params.get('weight_fundflow', 0.35)
        raw_score = (tech_score * w_tech) + (chip_score * w_chip) + (flow_score * w_flow)
        final_score = np.clip(raw_score - penalty, 0, 100)
        results['breakout_quality_score'] = final_score
        results['breakout_technical_score'] = np.clip(tech_score, 0, 100)
        results['breakout_chip_score'] = np.clip(chip_score, 0, 100)
        results['breakout_fundflow_score'] = np.clip(flow_score, 0, 100)
        results['breakout_penalty_score'] = penalty
        return results

    async def calculate_breakout_quality_score_v3(self, df_daily: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V3.0】突破质量分计算降级版本 - 兼容旧逻辑
        当新模型字段不全时使用
        """
        df = df_daily.copy()
        results = pd.DataFrame(index=df.index)
        # 简化版本，只计算技术维度
        breakout_score = pd.Series(50, index=df.index, dtype=float)
        # 基础技术指标
        if all(col in df.columns for col in ['volume', 'VOL_MA_21']):
            volume_ratio = df['volume'] / df['VOL_MA_21'].replace(0, 1)
            volume_score = np.clip((volume_ratio - 1) * 25, 0, 30)
            breakout_score += volume_score
        if 'VPA_EFFICIENCY' in df.columns:
            vpa_score = np.clip(df['VPA_EFFICIENCY'] * 15, 0, 25)
            breakout_score += vpa_score
        # 价格突破
        if all(col in df.columns for col in ['high', 'close']):
            high_20_max = df['high'].rolling(20).max()
            price_break = ((df['high'] - high_20_max) / df['close'] * 100).fillna(0)
            price_score = np.clip(price_break * 2, 0, 25)
            breakout_score += price_score
        results['breakout_quality_score'] = np.clip(breakout_score, 0, 100)
        return results




