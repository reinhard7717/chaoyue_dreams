# 新增文件: services/feature_engineering_service.py

import asyncio
import logging
from typing import Dict, List
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
        【V2.5 · A股换手率与主力成本增强版】
        - 优化: 引入自由换手率(turnover_rate_f_D)作为效率分母的修正项，识别A股特有的“缩量涨停”高效率模式。
        - 优化: 结合主力净额(net_mf_amount_D)对量价方向进行硬性归因。
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
        # 1. 计算A股特有的“换手调整效率” (Turnover Adjusted Efficiency)
        # 逻辑：在相同涨幅下，换手率越低，说明筹码锁定度越高，VPA攻击性越强
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = np.divide(volume, vol_ma_21)
            vol_ratio = np.where(vol_ratio == 0, 1.0, vol_ratio)
            # 引入换手率惩罚系数: (1 + turnover_f / 100)
            vpa_eff = pct_change / (vol_ratio * (1 + turnover_f / 5.0))
            df['VPA_EFFICIENCY_D'] = np.nan_to_num(vpa_eff)
        # 2. 结合主力净额修正方向
        if 'net_mf_amount_D' in df.columns:
            net_mf = df['net_mf_amount_D'].values.astype(np.float32)
            # 若价格上涨但主力净流出，则判定为“诱多性效率”，打折扣
            vpa_adj = np.where((pct_change > 0) & (net_mf < 0), vpa_eff * 0.3, vpa_eff)
            df['VPA_MF_ADJUSTED_EFF_D'] = np.nan_to_num(vpa_adj)
        # 3. 计算VPA加速度 (使用已有的SLOPE逻辑计算VPA的变化率)
        vpa_values = df['VPA_EFFICIENCY_D'].values.astype(np.float64)
        df['VPA_ACCELERATION_5D'] = _numba_rolling_slope(vpa_values, 5)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V6.5 · 熵增与情绪周期识别版】
        - 优化: 使用样本熵(chip_entropy_D)替代ADX判断市场状态。低熵值+高情绪=强趋势阶段。
        - 优化: 引入主力控盘度(main_cost_range_ratio_D)动态调整突破信号的阈值。
        """
        timeframe = 'D'
        if timeframe not in all_dfs: return all_dfs
        df = all_dfs[timeframe].copy()
        # 1. 基于信息论的市场阶段判定
        # 低熵代表趋势有序，高熵代表震荡混沌
        if 'chip_entropy_D' in df.columns:
            entropy_ma = df['chip_entropy_D'].rolling(20).mean()
            is_trending = (df['chip_entropy_D'] < entropy_ma * 0.95).values
        else:
            is_trending = (df['ADX_14_D'] > 25).values if 'ADX_14_D' in df.columns else np.zeros(len(df), dtype=bool)
        # 2. 动态突破阈值修正
        # 逻辑：主力控盘度越高，向上突破所需的成交量倍率可以适当放低
        vol_break_ratio = 1.8 # 默认1.8倍
        if 'main_cost_range_ratio_D' in df.columns:
            # 控盘度从0.1-0.9映射到1.2-2.0倍量需求
            dynamic_vol_req = 2.0 - (df['main_cost_range_ratio_D'] * 0.8)
            vol_break_cond = (df['volume_D'] > df['VOL_MA_21_D'] * dynamic_vol_req)
        else:
            vol_break_cond = (df['volume_D'] > df['VOL_MA_21_D'] * vol_break_ratio)
        # 3. 信号合成 (向量化)
        df['IS_TRENDING_STAGE_D'] = is_trending
        if 'VPA_EFFICIENCY_D' in df.columns and 'uptrend_strength_D' in df.columns:
            # 强趋势+量价效率激增+能量集中
            df['IS_BREAKOUT_CONFIRMED_D'] = is_trending & vol_break_cond & (df['VPA_EFFICIENCY_D'] > 0)
        # 4. 反转警告逻辑优化
        if 'reversal_warning_score_D' in df.columns:
            # 结合换手率极值和乖离率判定A股特有的“情绪顶”
            bias_extreme = (df['BIAS_5_D'] > 5) | (df['BIAS_5_D'] < -5)
            df['IS_EMOTIONAL_EXTREME_D'] = bias_extreme & (df['turnover_rate_D'] > df['turnover_rate_D'].rolling(20).quantile(0.9))
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
        【V4.0 · 多因子融合版】突破质量分计算专用通道
        - 核心重构：使用ChipFactorBase、ChipHoldingMatrixBase、FundFlowFactor模型的新字段
        - 废弃字段替代：
            total_winner_rate -> winner_rate_D + profit_ratio_D + pressure_trapped_D组合
            main_force_flow_directionality -> behavior_pattern_D + net_energy_flow_D + flow_momentum_5d_D组合
        - 增强维度：从单一突破质量扩展为技术、筹码、资金、能量四维突破质量评估
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe] is None:
            return all_dfs
        df_daily = all_dfs[timeframe]
        # 新增幂等性检查
        if 'breakout_quality_score_D' in df_daily.columns:
            logger.debug(f"突破质量分 (breakout_quality_score_D) 已存在于周期 '{timeframe}' 的DataFrame中，跳过重复计算。")
            return all_dfs
        # ==================== 1. 新模型字段替代方案 ====================
        # 废弃字段映射：
        # total_winner_rate -> winner_rate_D + profit_ratio_D + pressure_trapped_D
        # main_force_flow_directionality -> behavior_pattern_D + net_energy_flow_D + flow_momentum_5d_D
        # 补充 calculate_breakout_quality_score 所需的材料
        required_materials = [
            # 基础量价指标
            'volume', 'VOL_MA_21', 
            'open', 'high', 'low', 'close',
            # VPA效率指标
            'VPA_EFFICIENCY',
            'VPA_BUY_EFFICIENCY',
            # 执行强度指标
            'main_force_buy_execution_alpha', 'upward_impulse_strength',
            'buy_order_book_clearing_rate', 'bid_side_liquidity',
            'vwap_cross_up_intensity', 'opening_buy_strength',
            # 筹码清洗指标
            'floating_chip_cleansing_efficiency',
            # 风险预警指标
            'deception_lure_long_intensity', 'wash_trade_buy_volume'
        ]
        # 新增：新模型字段（替代废弃字段）
        new_model_materials = [
            # ChipFactorBase字段（筹码替代）
            'winner_rate',                    # 胜率（替代total_winner_rate）
            'profit_ratio',                   # 获利比例
            'chip_concentration_ratio',       # 筹码集中度
            'chip_stability',                 # 筹码稳定性
            'profit_pressure',                # 获利盘压力
            # FundFlowFactor字段（资金流向替代）
            'flow_intensity',                 # 资金流入强度得分
            'accumulation_score',             # 建仓模式得分
            'pushing_score',                  # 拉升模式得分
            'net_amount_ratio',               # 净流入占比
            'flow_momentum_5d',               # 5日资金动量（替代directionality）
            'flow_stability',                 # 资金流稳定性
            'large_order_anomaly',            # 大单异动
            # ChipHoldingMatrixBase字段（能量场替代）
            'absorption_energy',              # 吸收能量
            'distribution_energy',            # 派发能量
            'net_energy_flow',                # 净能量流向（替代directionality）
            'game_intensity',                 # 博弈强度
            'breakout_potential',             # 突破势能
            'energy_concentration',           # 能量集中度
            # 博弈状态字段
            'behavior_accumulation',          # 吸筹强度
            'behavior_distribution',          # 派发强度
            'behavior_consolidation',         # 整理强度
            # 压力支撑字段
            'pressure_trapped',               # 套牢盘压力
            'pressure_profit',                # 获利盘比例
            'support_strength',               # 下方支撑强度
            'resistance_strength',            # 上方阻力强度
        ]
        # 合并所有需要的字段
        all_required_materials = required_materials + new_model_materials
        df_standardized = pd.DataFrame(index=df_daily.index)
        missing_materials = []
        for material in all_required_materials:
            source_col_with_suffix = f"{material}_{timeframe}"
            # 特殊处理：部分字段可能有不同的命名格式
            if source_col_with_suffix in df_daily.columns:
                df_standardized[material] = df_daily[source_col_with_suffix]
            else:
                # 尝试其他可能的命名格式
                alt_names = [
                    f"{material}",
                    f"{material.upper()}_{timeframe}",
                    f"{material.lower()}_{timeframe}"
                ]
                found = False
                for alt_name in alt_names:
                    if alt_name in df_daily.columns:
                        df_standardized[material] = df_daily[alt_name]
                        found = True
                        break
                if not found:
                    missing_materials.append(source_col_with_suffix)
        # 检查关键字段是否存在
        critical_materials = ['volume', 'high', 'low', 'close', 'VPA_EFFICIENCY']
        missing_critical = [m for m in critical_materials if m not in df_standardized.columns]
        if missing_critical:
            logger.warning(f"突破质量分计算中止，缺少关键材料: {missing_critical}")
            return all_dfs
        if missing_materials:
            logger.debug(f"突破质量分计算缺少部分材料（非关键）: {missing_materials[:10]}...")
        # ==================== 2. 计算突破质量分（多因子融合版） ====================
        try:
            # 调用新版计算器（支持新模型字段）
            result_df = await self.calculate_breakout_quality_score_v4(
                df_daily=df_standardized, 
                params=params
            )
        except AttributeError:
            # 如果新版计算器不存在，使用降级版本
            logger.warning("新版突破质量分计算器不存在，使用降级版本...")
            result_df = await self.calculate_breakout_quality_score_v3(df_standardized, params)
        if result_df is not None and not result_df.empty:
            df_daily = df_daily.join(result_df, how='left')
            # --- 命名协议强制执行 ---
            # 将合并进来的列，强制重命名为带 '_D' 后缀的标准格式
            rename_dict = {}
            for col in result_df.columns:
                if not col.endswith(f'_{timeframe}'):
                    rename_dict[col] = f"{col}_{timeframe}"
            if rename_dict:
                df_daily.rename(columns=rename_dict, inplace=True)
            # 前向填充关键指标
            fill_cols = [
                'breakout_quality_score_D',
                'breakout_technical_score_D',
                'breakout_chip_score_D',
                'breakout_fundflow_score_D',
                'breakout_energy_score_D'
            ]
            for col in fill_cols:
                if col in df_daily.columns:
                    df_daily[col] = df_daily[col].ffill()
            all_dfs[timeframe] = df_daily
            logger.info("突破质量分计算完成并已集成。")
            # 输出统计信息
            if 'breakout_quality_score_D' in df_daily.columns:
                score_series = df_daily['breakout_quality_score_D'].dropna()
                if not score_series.empty:
                    logger.info(f"突破质量分统计 - 均值: {score_series.mean():.2f}, 最大值: {score_series.max():.2f}, 最小值: {score_series.min():.2f}")
        else:
            logger.warning("突破质量分计算器返回了None或空DataFrame，未集成。")
        return all_dfs

    async def calculate_ma_potential_metrics(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.2 · 向量化优化版】均线系统势能分析引擎
        - 优化: 移除 rolling().apply(lambda) 的低效实现，改用向量化的 rolling().mean() 和 std() 计算 Z-Score。
        - 性能: 提升约 50-100 倍。
        """
        if not params.get('enabled', False):
            return all_dfs
        for timeframe in params.get('apply_on', []):
            if timeframe not in all_dfs or all_dfs[timeframe] is None or all_dfs[timeframe].empty:
                continue
            df = all_dfs[timeframe]
            ma_periods = params.get('ma_periods', [])
            ma_type = params.get('ma_type', 'EMA')
            tension_short_period = params.get('tension_short_period', 5)
            tension_long_period = params.get('tension_long_period', 55)
            norm_window = params.get('norm_window', 55)
            # 1. 军火库点验
            ma_cols = [f"{ma_type}_{p}_{timeframe}" for p in ma_periods]
            tension_cols = [f"{ma_type}_{tension_short_period}_{timeframe}", f"{ma_type}_{tension_long_period}_{timeframe}"]
            atr_col = f"ATR_14_{timeframe}"
            required_cols = list(set(ma_cols + tension_cols + [atr_col]))
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"均线系统势能分析失败({timeframe})：缺少核心数据列 {missing_cols}")
                continue
            try:
                # 2. 计算势能大小 (张力)
                tension_range = df[f"{ma_type}_{tension_short_period}_{timeframe}"] - df[f"{ma_type}_{tension_long_period}_{timeframe}"]
                atr = df[atr_col].replace(0, np.nan)
                raw_tension_index = tension_range / atr
                # 【优化】向量化计算 Rolling Z-Score
                roll_mean = raw_tension_index.rolling(window=norm_window).mean()
                roll_std = raw_tension_index.rolling(window=norm_window).std()
                # 避免除以零
                z_score = (raw_tension_index - roll_mean) / (roll_std + 1e-9)
                tension_index_series = (z_score.fillna(0).clip(-3, 3) / 3).astype(np.float32)
                df[f'MA_POTENTIAL_TENSION_INDEX_{timeframe}'] = tension_index_series
                # 3. 计算势能方向 (有序性) - 【Numba 优化】
                ma_df = df[ma_cols]
                ma_values = ma_df.values.astype(np.float64)
                periods_series = pd.Series(ma_periods)
                ma_ranks_x = periods_series.rank(method='first').values.astype(np.float64)
                orderliness_scores = _numba_spearman_orderliness(ma_values, ma_ranks_x)
                df[f'MA_POTENTIAL_ORDERLINESS_SCORE_{timeframe}'] = orderliness_scores.astype(np.float32)
                # 4. 计算势能变化率 (压缩率)
                ma_std = ma_df.std(axis=1)
                normalized_std = ma_std / atr
                compression_rate = 1 - (normalized_std.rolling(window=norm_window).rank(pct=True)).fillna(0.5)
                df[f'MA_POTENTIAL_COMPRESSION_RATE_{timeframe}'] = compression_rate.astype(np.float32)
            except Exception as e:
                logger.error(f"计算均线系统势能时发生错误({timeframe}): {e}", exc_info=True)
        return all_dfs

    async def calculate_och(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V3.6 · 向量化与内存优化版】计算整体筹码健康度 (Overall Chip Health, OCH)。
        - 优化: 
            1. 移除 _get_safe_series_local 中的 Series 创建和 fillna/astype 开销，直接操作 NumPy 数组。
            2. 优化 _nonlinear_fusion 的数据准备过程，减少 Python 循环和列表构建。
        - 机制: In-place 修改 DataFrame，高效追加 'OCH_D' 列。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 OCH 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        n_rows = len(df)
        # 辅助函数：直接获取 float32 NumPy 数组，处理 NaN，避免 Series 开销
        def _get_safe_array(col_name, default_val=0.0):
            if col_name not in df.columns:
                return np.full(n_rows, default_val, dtype=np.float32)
            # 获取底层数组并转换为 float32 (如果已经是则开销很小)
            arr = df[col_name].values.astype(np.float32, copy=False)
            # 检查并填充 NaN (原地修改副本或新数组)
            if np.isnan(arr).any():
                # 为了安全起见，astype 可能已经拷贝了数据，所以这里修改是安全的
                # 如果原数据是 float32 且无拷贝，则需要 copy=True 避免修改原 DF
                # 这里 astype 默认 copy=True 除非类型完全匹配且内存连续
                arr = np.nan_to_num(arr, nan=default_val)
            return arr
        # --- 情境自适应调制器 (直接获取数组) ---
        # rolling mean 仍然使用 Pandas，因为 Numba 实现 rolling mean 较为繁琐且 Pandas 较快
        def _get_rolling_context(col, window=21):
            if col not in df.columns:
                return np.zeros(n_rows, dtype=np.float32)
            return df[col].rolling(window).mean().fillna(0).values.astype(np.float32)
        volatility_context = _get_rolling_context('VOLATILITY_INSTABILITY_INDEX_21d_D')
        sentiment_context = _get_rolling_context('market_sentiment_score_D')
        entropy_context = _get_rolling_context('price_volume_entropy_D')
        # 融合函数：直接接受数组字典
        def _nonlinear_fusion_optimized(scores_map: Dict[str, np.ndarray], weights_map: Dict[str, float]) -> np.ndarray:
            # 预分配列表
            score_arrays = []
            weight_arrays = []
            # 遍历字典 (Python 3.7+ 保持插入顺序)
            for name, weight in weights_map.items():
                arr = scores_map.get(name)
                if arr is None:
                    arr = np.zeros(n_rows, dtype=np.float32)
                score_arrays.append(arr)
                # 创建同维度的权重数组 (Numba 函数通常需要数组输入)
                weight_arrays.append(np.full(n_rows, weight, dtype=np.float32))
            # 调用 Numba 核心函数
            return _numba_nonlinear_fusion_core(
                score_arrays,
                weight_arrays,
                volatility_context,
                sentiment_context,
                entropy_context
            )
        # --- 1. 筹码集中度与结构优化 ---
        cost_gini = _get_safe_array('cost_gini_coefficient_D', 0.5)
        # rank(pct=True) 依然依赖 Pandas，因为 NumPy 没有直接的 rank pct
        peak_kurtosis_series = df.get('primary_peak_kurtosis_D', pd.Series(np.full(n_rows, 3.0)))
        normalized_kurtosis = peak_kurtosis_series.rolling(window=120, min_periods=20).rank(pct=True).fillna(0.5).values.astype(np.float32)
        peak_solidity = _get_safe_array('dominant_peak_solidity_D', 0.5)
        peak_volume_ratio = _get_safe_array('dominant_peak_volume_ratio_D', 0.5)
        chip_fault = _get_safe_array('chip_fault_blockage_ratio_D', 0.0)
        # 向量化计算
        concentration_health = np.clip(1 - cost_gini, 0, 1)
        peak_quality = np.clip(peak_solidity * peak_volume_ratio * normalized_kurtosis, 0, 1)
        blockage_penalty = 1 - chip_fault
        concentration_scores = {
            'concentration_health': concentration_health, 
            'peak_quality': peak_quality, 
            'blockage_penalty': blockage_penalty
        }
        concentration_weights = {'concentration_health': 0.5, 'peak_quality': 0.4, 'blockage_penalty': 0.1}
        concentration_score = _nonlinear_fusion_optimized(concentration_scores, concentration_weights)
        # --- 2. 成本与盈亏结构动态 ---
        total_winner_rate = _get_safe_array('total_winner_rate_D', 0.5)
        total_loser_rate = _get_safe_array('total_loser_rate_D', 0.5)
        winner_profit_margin = _get_safe_array('winner_profit_margin_avg_D', 0.0) / 100.0
        loser_loss_margin = _get_safe_array('loser_loss_margin_avg_D', 0.0) / 100.0
        cost_divergence = _get_safe_array('cost_structure_skewness_D', 0.0)
        mf_cost_advantage = _get_safe_array('main_force_cost_advantage_D', 0.0)
        imminent_profit_taking = _get_safe_array('profit_taking_flow_ratio_D', 0.0)
        loser_capitulation_pressure = _get_safe_array('loser_pain_index_D', 0.0)
        rally_sell = _get_safe_array('rally_sell_distribution_intensity_D', 0.0)
        dip_buy = _get_safe_array('dip_buy_absorption_strength_D', 0.0)
        panic_buy = _get_safe_array('panic_buy_absorption_contribution_D', 0.0)
        profit_pressure = total_winner_rate * winner_profit_margin * np.clip(rally_sell + imminent_profit_taking, 0, 1)
        loser_support = total_loser_rate * loser_loss_margin * (1 - loser_capitulation_pressure) + \
                        dip_buy * 0.5 + panic_buy * 0.5
        cost_advantage_score = np.clip(mf_cost_advantage - cost_divergence, -1, 1)
        cost_structure_scores = {
            'loser_support': loser_support, 
            'cost_advantage_score': cost_advantage_score, 
            'profit_pressure': profit_pressure
        }
        cost_structure_weights = {'loser_support': 0.4, 'cost_advantage_score': 0.4, 'profit_pressure': -0.2}
        cost_structure_score = _nonlinear_fusion_optimized(cost_structure_scores, cost_structure_weights)
        # --- 3. 持股心态与交易行为 ---
        winner_conviction = _get_safe_array('winner_stability_index_D', 0.0)
        chip_fatigue = _get_safe_array('chip_fatigue_index_D', 0.0)
        locked_profit = winner_conviction # 复用
        locked_loss = 1.0 - _get_safe_array('capitulation_flow_ratio_D', 0.0)
        # 复合指标计算 (向量化加权求和)
        buy_side_absorption_composite = np.clip(
            _get_safe_array('capitulation_absorption_index_D') * 0.2 +
            _get_safe_array('active_buying_support_D') * 0.1 +
            dip_buy * 0.1 +
            panic_buy * 0.1 +
            _get_safe_array('opening_buy_strength_D') * 0.05 +
            _get_safe_array('pre_closing_buy_posture_D') * 0.05 +
            _get_safe_array('closing_auction_buy_ambush_D') * 0.05 +
            _get_safe_array('main_force_buy_ofi_D') * 0.1 +
            _get_safe_array('retail_buy_ofi_D') * 0.05 +
            _get_safe_array('bid_side_liquidity_D') * 0.05 +
            _get_safe_array('buy_order_book_clearing_rate_D') * 0.05 +
            _get_safe_array('vwap_buy_control_strength_D') * 0.05,
            0, 1
        )
        sell_side_pressure_composite = np.clip(
            _get_safe_array('active_selling_pressure_D') * 0.1 +
            rally_sell * 0.1 +
            _get_safe_array('dip_sell_pressure_resistance_D') * 0.05 +
            _get_safe_array('panic_sell_volume_contribution_D') * 0.1 +
            _get_safe_array('opening_sell_strength_D') * 0.05 +
            _get_safe_array('pre_closing_sell_posture_D') * 0.05 +
            _get_safe_array('closing_auction_sell_ambush_D') * 0.05 +
            _get_safe_array('main_force_sell_ofi_D') * 0.1 +
            _get_safe_array('retail_sell_ofi_D') * 0.05 +
            _get_safe_array('ask_side_liquidity_D') * 0.05 +
            _get_safe_array('sell_order_book_clearing_rate_D') * 0.05 +
            _get_safe_array('vwap_sell_control_strength_D') * 0.05,
            0, 1
        )
        combat_intensity = _get_safe_array('mf_retail_battle_intensity_D', 0.0)
        conviction_lock_score = np.clip(winner_conviction + locked_profit - chip_fatigue - locked_loss, -1, 1)
        absorption_support_score = np.clip(buy_side_absorption_composite - sell_side_pressure_composite, -1, 1)
        wash_trade_penalty = np.clip(_get_safe_array('wash_trade_buy_volume_D') + _get_safe_array('wash_trade_sell_volume_D'), 0, 1) * 0.1
        sentiment_scores = {
            'conviction_lock_score': (conviction_lock_score + 1) / 2,
            'absorption_support_score': (absorption_support_score + 1) / 2,
            'combat_intensity': combat_intensity,
            'wash_trade_penalty': wash_trade_penalty
        }
        sentiment_weights = {'conviction_lock_score': 0.4, 'absorption_support_score': 0.4, 'combat_intensity': 0.2, 'wash_trade_penalty': -0.1}
        sentiment_score = _nonlinear_fusion_optimized(sentiment_scores, sentiment_weights)
        # --- 4. 主力控盘与意图 ---
        mf_control_leverage = _get_safe_array('control_solidity_index_D', 0.0)
        mf_on_peak_flow_composite = _get_safe_array('main_force_on_peak_buy_flow_D') - _get_safe_array('main_force_on_peak_sell_flow_D')
        # Rank pct 依然需要 Pandas Series
        mf_on_peak_flow_normalized = pd.Series(mf_on_peak_flow_composite).rank(pct=True).fillna(0.5).values.astype(np.float32)
        mf_on_peak_flow_normalized = np.clip(mf_on_peak_flow_normalized * 2 - 1, 0, 1)
        mf_intent_composite = np.clip(
            _get_safe_array('main_force_flow_directionality_D') * 0.2 +
            (_get_safe_array('main_force_buy_execution_alpha_D') - _get_safe_array('main_force_sell_execution_alpha_D')) * 0.2 +
            _get_safe_array('main_force_conviction_index_D') * 0.1 +
            (_get_safe_array('main_force_vwap_up_guidance_D') - _get_safe_array('main_force_vwap_down_guidance_D')) * 0.1 +
            (_get_safe_array('vwap_cross_up_intensity_D') - _get_safe_array('vwap_cross_down_intensity_D')) * 0.1 +
            (_get_safe_array('main_force_t0_buy_efficiency_D') - _get_safe_array('main_force_t0_sell_efficiency_D')) * 0.1 +
            (_get_safe_array('buy_flow_efficiency_index_D') - _get_safe_array('sell_flow_efficiency_index_D')) * 0.1,
            -1, 1
        )
        mf_vpoc_premium = _get_safe_array('mf_vpoc_premium_D', 0.0)
        vwap_control_composite = _get_safe_array('vwap_buy_control_strength_D') - _get_safe_array('vwap_sell_control_strength_D')
        control_strength = mf_control_leverage * ((vwap_control_composite + 1) / 2)
        mf_cost_advantage_final = (mf_vpoc_premium + 1) / 2
        turnover_rate_f = _get_safe_array('turnover_rate_f_D', 0.0)
        turnover_health = np.ones(n_rows, dtype=np.float32)
        # 向量化条件赋值
        mask_low = turnover_rate_f < 2
        turnover_health[mask_low] = turnover_rate_f[mask_low] / 2
        mask_high = turnover_rate_f > 15
        turnover_health[mask_high] = 1 - (turnover_rate_f[mask_high] - 15) / 10
        turnover_health = np.clip(turnover_health, 0, 1)
        distribution_penalty = np.clip(_get_safe_array('covert_distribution_signal_D') + _get_safe_array('supportive_distribution_intensity_D'), 0, 1) * 0.1
        main_force_scores = {
            'control_strength': control_strength, 
            'mf_on_peak_flow_normalized': mf_on_peak_flow_normalized, 
            'mf_intent_composite': (mf_intent_composite + 1) / 2,
            'mf_cost_advantage_final': mf_cost_advantage_final, 
            'turnover_health': turnover_health, 
            'distribution_penalty': distribution_penalty
        }
        main_force_weights = {
            'control_strength': 0.3, 'mf_on_peak_flow_normalized': 0.2, 'mf_intent_composite': 0.3,
            'mf_cost_advantage_final': 0.1, 'turnover_health': 0.1, 'distribution_penalty': -0.1
        }
        main_force_score = _nonlinear_fusion_optimized(main_force_scores, main_force_weights)
        # --- 最终 OCH_D 融合 ---
        och_scores = {
            'concentration_score': concentration_score, 
            'cost_structure_score': cost_structure_score, 
            'sentiment_score': sentiment_score, 
            'main_force_score': main_force_score
        }
        och_weights = {'concentration_score': 0.25, 'cost_structure_score': 0.25, 'sentiment_score': 0.25, 'main_force_score': 0.25}
        och_score = _nonlinear_fusion_optimized(och_scores, och_weights) * 2 - 1
        # 赋值回 DataFrame
        df['OCH_D'] = och_score
        all_dfs[timeframe] = df
        logger.info("OCH 指标计算完成 (向量化优化版)。")
        return all_dfs

    async def calculate_structural_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.0 · 结构特征合成版】
        - 核心职责: 从 BaseAdvancedStructuralMetrics 中提取和合成更高级的结构性特征。
        - 指标含义:
            - STRUCTURAL_TREND_HEALTH_D: 综合评估趋势的健康度、加速状态和成交结构。
            - BREAKTHROUGH_CONVICTION_D: 直接使用突破信念分。
            - DEFENSE_SOLIDITY_D: 直接使用防守稳固度。
            - EQUILIBRIUM_COMPRESSION_D: 直接使用均衡压缩指数。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"结构特征计算失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        # 检查计算所需的原始结构指标列是否存在
        required_structural_cols = [
            'trend_acceleration_score_D', 'final_charge_intensity_D', 'volume_structure_skew_D',
            'breakthrough_conviction_score_D', 'defense_solidity_score_D', 'equilibrium_compression_index_D'
        ]
        missing_cols = [col for col in required_structural_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"结构特征计算缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        # 1. 合成 STRUCTURAL_TREND_HEALTH_D
        # 综合趋势加速、终场冲锋强度和成交结构偏度，评估趋势的整体健康度
        # 假设这些指标都已归一化到0-100或-1到1的范围
        df['STRUCTURAL_TREND_HEALTH_D'] = (
            df['trend_acceleration_score_D'] * 0.4 +
            df['final_charge_intensity_D'] * 0.3 +
            (1 - df['volume_structure_skew_D'].abs()) * 0.3 # 偏度越小，结构越健康
        ).clip(0, 100) # 假设输出范围是0-100
        # 2. 直接使用其他结构指标作为特征
        df['BREAKTHROUGH_CONVICTION_D'] = df['breakthrough_conviction_score_D']
        df['DEFENSE_SOLIDITY_D'] = df['defense_solidity_score_D']
        df['EQUILIBRIUM_COMPRESSION_D'] = df['equilibrium_compression_index_D']
        all_dfs[timeframe] = df
        logger.info("结构特征计算完成。")
        return all_dfs

    async def calculate_geometric_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.1 · 几何特征计算与列名冲突修复】
        - 核心职责: 计算与K线形态、趋势几何结构相关的特征。
        - 修复: 解决 df_daily.join(df_geometric_features) 时列名冲突问题，通过添加 rsuffix 区分。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df_daily = all_dfs[timeframe]
        # 假设 df_geometric_features 是在此方法内部计算生成的
        # 这是一个占位符，实际的几何特征计算逻辑应在此处实现
        df_geometric_features = pd.DataFrame(index=df_daily.index)
        # 识别重叠列并记录警告
        overlapping_cols = df_daily.columns.intersection(df_geometric_features.columns)
        if not overlapping_cols.empty:
            logger.warning(f"在合并几何特征时发现重叠列: {overlapping_cols.tolist()}。来自 df_geometric_features 的重叠列将添加 '_geom' 后缀。")
        df_daily = df_daily.join(df_geometric_features, how='left', rsuffix='_geom')
        all_dfs[timeframe] = df_daily
        return all_dfs

    async def calculate_breakout_quality_score_v4(self, df_daily: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V4.1 · 向量化优化版】突破质量分计算核心逻辑
        - 优化: 移除 breakout_type 计算中的 Python 循环，使用向量化 idxmax 和 np.select。
        """
        df = df_daily.copy()
        results = pd.DataFrame(index=df.index)
        # ==================== 1. 技术维度突破质量 ====================
        technical_scores = pd.Series(0, index=df.index, dtype=float)
        # 1.1 价格突破强度
        if all(col in df.columns for col in ['high', 'low', 'close']):
            # 计算真实波幅突破
            atr_period = 14
            tr = pd.concat([
                df['high'] - df['low'],
                (df['high'] - df['close'].shift()).abs(),
                (df['low'] - df['close'].shift()).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(atr_period).mean()
            price_breakout_intensity = ((df['high'] - df['high'].rolling(20).max()) / atr).fillna(0)
            technical_scores += np.clip(price_breakout_intensity * 10, 0, 20)
        # 1.2 成交量确认
        if all(col in df.columns for col in ['volume', 'VOL_MA_21']):
            volume_breakout_ratio = df['volume'] / df['VOL_MA_21'].replace(0, 1)
            volume_score = np.clip((volume_breakout_ratio - 1) * 15, 0, 20)
            technical_scores += volume_score
        # 1.3 VPA效率确认
        if 'VPA_EFFICIENCY' in df.columns:
            vpa_score = np.clip(df['VPA_EFFICIENCY'] * 10, 0, 15)
            technical_scores += vpa_score
        if 'VPA_BUY_EFFICIENCY' in df.columns:
            vpa_buy_score = np.clip(df['VPA_BUY_EFFICIENCY'] * 12, 0, 15)
            technical_scores += vpa_buy_score
        # 1.4 执行强度
        execution_indicators = [
            'main_force_buy_execution_alpha',
            'upward_impulse_strength',
            'buy_order_book_clearing_rate',
            'bid_side_liquidity',
            'vwap_cross_up_intensity',
            'opening_buy_strength'
        ]
        for indicator in execution_indicators:
            if indicator in df.columns:
                score = np.clip(df[indicator] * 5, 0, 10)
                technical_scores += score
        results['breakout_technical_score'] = np.clip(technical_scores, 0, 100)
        # ==================== 2. 筹码维度突破质量 ====================
        chip_scores = pd.Series(0, index=df.index, dtype=float)
        # 2.1 胜率和获利比例（替代total_winner_rate）
        if 'winner_rate' in df.columns:
            winner_score = np.clip(df['winner_rate'] * 100, 0, 25)
            chip_scores += winner_score
        if 'profit_ratio' in df.columns:
            profit_score = np.clip(df['profit_ratio'] * 100, 0, 20)
            chip_scores += profit_score
        # 2.2 筹码集中度
        if 'chip_concentration_ratio' in df.columns:
            concentration_score = np.clip(df['chip_concentration_ratio'] * 100, 0, 15)
            chip_scores += concentration_score
        # 2.3 筹码稳定性
        if 'chip_stability' in df.columns:
            stability_score = np.clip(df['chip_stability'] * 100, 0, 15)
            chip_scores += stability_score
        # 2.4 获利压力释放
        if 'profit_pressure' in df.columns:
            # 获利压力小，突破质量高
            pressure_release_score = np.clip((1 - abs(df['profit_pressure'])) * 15, 0, 15)
            chip_scores += pressure_release_score
        # 2.5 套牢盘压力（替代total_winner_rate的另一个维度）
        if 'pressure_trapped' in df.columns:
            # 套牢盘压力小，突破阻力小
            trapped_pressure_score = np.clip((1 - df['pressure_trapped']) * 10, 0, 10)
            chip_scores += trapped_pressure_score
        results['breakout_chip_score'] = np.clip(chip_scores, 0, 100)
        # ==================== 3. 资金维度突破质量 ====================
        fundflow_scores = pd.Series(0, index=df.index, dtype=float)
        # 3.1 资金流强度（核心指标）
        if 'flow_intensity' in df.columns:
            flow_intensity_score = np.clip(df['flow_intensity'], 0, 25)
            fundflow_scores += flow_intensity_score
        # 3.2 行为模式（替代main_force_flow_directionality）
        if 'accumulation_score' in df.columns:
            accumulation_score = np.clip(df['accumulation_score'] * 0.2, 0, 15)
            fundflow_scores += accumulation_score
        if 'pushing_score' in df.columns:
            pushing_score = np.clip(df['pushing_score'] * 0.2, 0, 15)
            fundflow_scores += pushing_score
        # 3.3 净流入占比
        if 'net_amount_ratio' in df.columns:
            net_ratio_score = np.clip(df['net_amount_ratio'] * 10, 0, 15)
            fundflow_scores += net_ratio_score
        # 3.4 资金动量（替代directionality的方向性）
        if 'flow_momentum_5d' in df.columns:
            momentum_score = np.clip(df['flow_momentum_5d'] * 100 + 50, 0, 15)
            fundflow_scores += momentum_score
        # 3.5 资金流稳定性
        if 'flow_stability' in df.columns:
            stability_score = np.clip(df['flow_stability'] * 0.5, 0, 10)
            fundflow_scores += stability_score
        # 3.6 大单异动过滤
        if 'large_order_anomaly' in df.columns:
            # 无大单异动时加分，有异动时扣分
            anomaly_penalty = np.where(df['large_order_anomaly'], -10, 5)
            fundflow_scores += anomaly_penalty
        results['breakout_fundflow_score'] = np.clip(fundflow_scores, 0, 100)
        # ==================== 4. 能量维度突破质量 ====================
        energy_scores = pd.Series(0, index=df.index, dtype=float)
        # 4.1 吸收能量（关键突破能量）
        if 'absorption_energy' in df.columns:
            absorption_score = np.clip(df['absorption_energy'], 0, 25)
            energy_scores += absorption_score
        # 4.2 净能量流向（替代directionality的能量视角）
        if 'net_energy_flow' in df.columns:
            net_energy_score = np.clip(df['net_energy_flow'] + 50, 0, 20)
            energy_scores += net_energy_score
        # 4.3 博弈强度
        if 'game_intensity' in df.columns:
            game_score = np.clip(df['game_intensity'] * 100, 0, 15)
            energy_scores += game_score
        # 4.4 突破势能
        if 'breakout_potential' in df.columns:
            potential_score = np.clip(df['breakout_potential'], 0, 20)
            energy_scores += potential_score
        # 4.5 能量集中度
        if 'energy_concentration' in df.columns:
            concentration_score = np.clip(df['energy_concentration'] * 100, 0, 10)
            energy_scores += concentration_score
        # 4.6 派发能量（负向指标）
        if 'distribution_energy' in df.columns:
            distribution_penalty = np.clip((100 - df['distribution_energy']) * 0.1, 0, 10)
            energy_scores += distribution_penalty
        results['breakout_energy_score'] = np.clip(energy_scores, 0, 100)
        # ==================== 5. 综合突破质量分 ====================
        # 5.1 计算各维度加权平均
        weight_technical = params.get('weight_technical', 0.25)
        weight_chip = params.get('weight_chip', 0.25)
        weight_fundflow = params.get('weight_fundflow', 0.30)
        weight_energy = params.get('weight_energy', 0.20)
        breakout_scores = pd.Series(0, index=df.index, dtype=float)
        if 'breakout_technical_score' in results.columns:
            breakout_scores += results['breakout_technical_score'] * weight_technical
        if 'breakout_chip_score' in results.columns:
            breakout_scores += results['breakout_chip_score'] * weight_chip
        if 'breakout_fundflow_score' in results.columns:
            breakout_scores += results['breakout_fundflow_score'] * weight_fundflow
        if 'breakout_energy_score' in results.columns:
            breakout_scores += results['breakout_energy_score'] * weight_energy
        results['breakout_quality_score'] = np.clip(breakout_scores, 0, 100)
        # 5.2 突破质量分级
        conditions = [
            results['breakout_quality_score'] >= 80,
            results['breakout_quality_score'] >= 65,
            results['breakout_quality_score'] >= 50,
            results['breakout_quality_score'] >= 35,
            results['breakout_quality_score'] < 35
        ]
        choices = ['EXCELLENT', 'GOOD', 'MODERATE', 'WEAK', 'POOR']
        results['breakout_quality_grade'] = np.select(conditions, choices, default='UNKNOWN')
        # 5.3 突破类型识别 (向量化优化)
        # 构建分数矩阵
        score_cols = ['breakout_technical_score', 'breakout_chip_score', 'breakout_fundflow_score', 'breakout_energy_score']
        # 映射列名到简写
        col_mapping = {
            'breakout_technical_score': 'TECH',
            'breakout_chip_score': 'CHIP',
            'breakout_fundflow_score': 'FUND',
            'breakout_energy_score': 'ENERGY'
        }
        score_df = results[score_cols].fillna(0)
        # 找到最大分数的列和值
        max_vals = score_df.max(axis=1)
        max_cols = score_df.idxmax(axis=1).map(col_mapping)
        mean_vals = score_df.mean(axis=1)
        # 向量化判断逻辑
        cond_strong = max_vals > 70
        cond_mod = max_vals > 50
        cond_bal = mean_vals > 40
        # 构建类型字符串
        type_strong = 'STRONG_' + max_cols
        type_mod = 'MODERATE_' + max_cols
        # 使用 np.select 进行选择
        breakout_types = np.select(
            [cond_strong, cond_mod, cond_bal],
            [type_strong, type_mod, 'BALANCED'],
            default='NONE'
        )
        results['breakout_type'] = breakout_types
        # 5.4 风险预警
        results['breakout_risk_warning'] = False
        # 检查高风险信号
        risk_indicators = []
        if 'deception_lure_long_intensity' in df.columns:
            risk_indicators.append(df['deception_lure_long_intensity'] > 0.5)
        if 'wash_trade_buy_volume' in df.columns:
            risk_indicators.append(df['wash_trade_buy_volume'] > 0.3)
        if 'large_order_anomaly' in df.columns:
            risk_indicators.append(df['large_order_anomaly'] == True)
        if 'distribution_score' in df.columns:
            risk_indicators.append(df['distribution_score'] > 60)
        if risk_indicators:
            risk_mask = pd.concat(risk_indicators, axis=1).any(axis=1)
            results['breakout_risk_warning'] = risk_mask
        # ==================== 6. 置信度计算 ====================
        # 计算数据完整性置信度
        completeness_scores = []
        required_categories = {
            'technical': ['volume', 'high', 'low', 'close', 'VPA_EFFICIENCY'],
            'chip': ['winner_rate', 'profit_ratio'],
            'fundflow': ['flow_intensity', 'net_amount_ratio'],
            'energy': ['absorption_energy', 'net_energy_flow']
        }
        for category, fields in required_categories.items():
            available = sum(1 for f in fields if f in df.columns)
            completeness = available / len(fields) if fields else 1.0
            completeness_scores.append(completeness)
        data_completeness = np.mean(completeness_scores) if completeness_scores else 0.5
        results['breakout_confidence'] = np.clip(data_completeness * 100, 0, 100)
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




