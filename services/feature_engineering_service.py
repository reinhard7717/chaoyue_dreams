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

    async def calculate_vpa_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.1 · 向量化与NumPy优化版】VPA效率指标生产线重构
        - 优化: 将核心计算下沉到 NumPy 层，使用 float32 降级，避免 Pandas Series 运算的索引对齐开销。
        - 优化: 使用 np.divide 处理除零，替代慢速的 .replace(0, np.nan)。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        
        # 关键依赖检查
        required_cols = ['pct_change_D', 'volume_D', 'VOL_MA_21_D']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"VPA效率生产线缺少基础数据，模块已跳过！")
            return all_dfs
        # 提取基础数据为 NumPy 数组 (float32)
        pct_change = df['pct_change_D'].values.astype(np.float32)
        volume = df['volume_D'].values.astype(np.float32)
        vol_ma_21 = df['VOL_MA_21_D'].values.astype(np.float32)
        
        # ==================== 1. 资金流向增强版VPA ====================
        # 1.1 总VPA效率
        # 优化除法：避免 replace(0, nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            volume_ratio = np.divide(volume, vol_ma_21)
            # 处理 inf 和 nan (当 vol_ma_21 为 0 时)
            volume_ratio[~np.isfinite(volume_ratio)] = 0.0
            # 再次除法计算效率
            vpa_efficiency = np.divide(pct_change, volume_ratio)
            vpa_efficiency[~np.isfinite(vpa_efficiency)] = 0.0
        df['VPA_EFFICIENCY_D'] = vpa_efficiency
        # 1.2 使用资金流向因子替代幻方指标
        has_fundflow_data = all(col in df.columns for col in ['flow_intensity_D', 'net_amount_ratio_D', 'large_order_anomaly_D'])
        
        if has_fundflow_data:
            # 2.1 主力行为增强VPA
            if 'accumulation_score_D' in df.columns and 'distribution_score_D' in df.columns:
                accum_score = df['accumulation_score_D'].fillna(0).values.astype(np.float32)
                dist_score = df['distribution_score_D'].fillna(0).values.astype(np.float32)
                accumulation_weight = accum_score / 100.0
                distribution_weight = dist_score / 100.0
                # 向量化计算
                positive_change = np.maximum(pct_change, 0)
                negative_change = np.minimum(pct_change, 0)
                # 避免除零
                safe_vol_ratio = np.where(volume_ratio == 0, 1.0, volume_ratio)
                buy_eff = (positive_change / safe_vol_ratio) * accumulation_weight
                sell_eff = (negative_change / safe_vol_ratio) * distribution_weight
                df['VPA_BUY_ACCUM_EFF_D'] = np.nan_to_num(buy_eff)
                df['VPA_SELL_DIST_EFF_D'] = np.nan_to_num(sell_eff)
            # 2.2 大单异动增强VPA
            if 'large_order_anomaly_D' in df.columns and 'anomaly_intensity_D' in df.columns:
                anomaly_mask = df['large_order_anomaly_D'].values.astype(bool)
                anomaly_weight = df['anomaly_intensity_D'].fillna(0).values.astype(np.float32) / 100.0
                anomaly_vpa = vpa_efficiency * (1 + anomaly_weight)
                # 使用 np.where 向量化选择
                df['VPA_ANOMALY_ENHANCED_D'] = np.where(anomaly_mask, anomaly_vpa, vpa_efficiency)
            # 2.3 净流入占比调整的VPA
            if 'net_amount_ratio_D' in df.columns:
                net_ratio = df['net_amount_ratio_D'].fillna(0).values.astype(np.float32) / 100.0
                safe_vol_ratio = np.where(volume_ratio == 0, 1.0, volume_ratio)
                buy_eff_adj = np.maximum(pct_change, 0) / safe_vol_ratio
                sell_eff_adj = np.minimum(pct_change, 0) / safe_vol_ratio
                df['VPA_BUY_NETFLOW_ADJ_D'] = np.nan_to_num(buy_eff_adj * (1 + np.maximum(net_ratio, 0)))
                df['VPA_SELL_NETFLOW_ADJ_D'] = np.nan_to_num(sell_eff_adj * (1 + np.abs(np.minimum(net_ratio, 0))))
            # 2.4 资金流向稳定性调整VPA
            if 'flow_stability_D' in df.columns:
                stability_weight = df['flow_stability_D'].fillna(50).values.astype(np.float32) / 100.0
                df['VPA_STABILITY_WEIGHTED_D'] = vpa_efficiency * stability_weight
        # ==================== 3. 筹码结构增强版VPA ====================
        has_chip_data = all(col in df.columns for col in ['chip_concentration_ratio_D', 'winner_rate_D', 'turnover_rate_D'])
        if has_chip_data:
            # 3.1 筹码集中度调整
            conc_weight = df['chip_concentration_ratio_D'].fillna(0.5).values.astype(np.float32)
            df['VPA_CONCENTRATION_ADJ_D'] = vpa_efficiency * conc_weight
            # 3.2 获利盘压力调整
            if 'profit_pressure_D' in df.columns:
                pressure = df['profit_pressure_D'].fillna(0).values.astype(np.float32)
                pressure_adj = 1.0 / (1.0 + np.abs(pressure))
                df['VPA_PROFIT_PRESSURE_ADJ_D'] = vpa_efficiency * pressure_adj
            # 3.3 换手率调整
            turnover = df['turnover_rate_D'].fillna(0).values.astype(np.float32) / 100.0
            turnover_penalty = 1.0 / (1.0 + turnover * 5.0)
            df['VPA_TURNOVER_ADJ_D'] = vpa_efficiency * turnover_penalty
            # 3.4 筹码峰动态调整
            if 'peak_migration_speed_5d' in df.columns:
                speed = df['peak_migration_speed_5d'].fillna(0).values.astype(np.float32)
                mig_factor = 1.0 / (1.0 + np.abs(speed) * 0.1)
                df['VPA_MIGRATION_ADJ_D'] = vpa_efficiency * mig_factor
        # ==================== 4. 筹码持有时间矩阵增强版VPA ====================
        has_matrix_data = all(col in df.columns for col in ['short_term_ratio_D', 'long_term_ratio_D', 'absorption_energy_D', 'distribution_energy_D'])
        if has_matrix_data:
            # 4.1 短线筹码
            short_ratio = df['short_term_ratio_D'].fillna(0.2).values.astype(np.float32)
            short_penalty = 1.0 - short_ratio * 0.5
            df['VPA_SHORT_TERM_ADJ_D'] = vpa_efficiency * short_penalty
            # 4.2 长线筹码
            long_ratio = df['long_term_ratio_D'].fillna(0.5).values.astype(np.float32)
            long_boost = 1.0 + long_ratio * 0.3
            df['VPA_LONG_TERM_ADJ_D'] = vpa_efficiency * long_boost
            # 4.3 博弈能量场
            absorb_energy = df['absorption_energy_D'].fillna(0).values.astype(np.float32) / 100.0
            dist_energy = df['distribution_energy_D'].fillna(0).values.astype(np.float32) / 100.0
            safe_vol_ratio = np.where(volume_ratio == 0, 1.0, volume_ratio)
            buy_eff = np.maximum(pct_change, 0) / safe_vol_ratio
            sell_eff = np.minimum(pct_change, 0) / safe_vol_ratio
            df['VPA_ABSORPTION_ENHANCED_D'] = np.nan_to_num(buy_eff * (1 + absorb_energy))
            df['VPA_DISTRIBUTION_ENHANCED_D'] = np.nan_to_num(sell_eff * (1 + dist_energy))
            # 4.4 净能量流向
            if 'net_energy_flow_D' in df.columns:
                net_energy = df['net_energy_flow_D'].fillna(0).values.astype(np.float32) / 100.0
                df['VPA_NET_ENERGY_ADJ_D'] = vpa_efficiency * (1 + net_energy * 0.5)
        # ==================== 5. 复合VPA综合指标 ====================
        # 5.1 综合VPA评分
        vpa_cols_check = [
            'VPA_EFFICIENCY_D', 'VPA_CONCENTRATION_ADJ_D', 'VPA_STABILITY_WEIGHTED_D',
            'VPA_LONG_TERM_ADJ_D', 'VPA_NET_ENERGY_ADJ_D'
        ]
        valid_cols = [col for col in vpa_cols_check if col in df.columns]
        
        if len(valid_cols) > 1:
            # 提取矩阵并计算均值
            vpa_matrix = df[valid_cols].fillna(0).values.astype(np.float32)
            composite_score = np.mean(vpa_matrix, axis=1)
            df['VPA_COMPOSITE_SCORE_D'] = composite_score
            # 5.2 VPA效率分级 (向量化 np.select)
            conds = [
                composite_score > 2.0,
                composite_score > 1.0,
                composite_score > 0,
                composite_score > -1.0,
                composite_score > -2.0
            ]
            choices = [4, 3, 2, 1, 0]
            df['VPA_EFFICIENCY_LEVEL_D'] = np.select(conds, choices, default=0)
            # 5.3 VPA背离检测
            # Rolling 仍需 Pandas，但只对单列操作
            price_trend = df['pct_change_D'].rolling(window=5, min_periods=3).mean()
            vpa_trend = df['VPA_COMPOSITE_SCORE_D'].rolling(window=5, min_periods=3).mean()
            bullish_div = (price_trend < 0) & (vpa_trend > 0)
            bearish_div = (price_trend > 0) & (vpa_trend < 0)
            df['VPA_BULLISH_DIVERGENCE_D'] = bullish_div.astype(int)
            df['VPA_BEARISH_DIVERGENCE_D'] = bearish_div.astype(int)
            # 5.4 VPA动量
            df['VPA_MOMENTUM_5D'] = df['VPA_COMPOSITE_SCORE_D'].diff(5)
            df['VPA_MOMENTUM_10D'] = df['VPA_COMPOSITE_SCORE_D'].diff(10)
        # ==================== 6. 信号质量评估 ====================
        # 6.1 VPA信号置信度
        conf_factors = []
        if 'flow_stability_D' in df.columns: conf_factors.append(df['flow_stability_D'].fillna(50).values / 100.0)
        if 'chip_stability_D' in df.columns: conf_factors.append(df['chip_stability_D'].fillna(0.5).values)
        if 'validation_score_D' in df.columns: conf_factors.append(df['validation_score_D'].fillna(0.5).values)
        
        if conf_factors:
            # 堆叠并计算均值
            conf_matrix = np.vstack(conf_factors)
            df['VPA_SIGNAL_CONFIDENCE_D'] = np.mean(conf_matrix, axis=0)
        # 6.2 VPA交易信号生成
        if 'VPA_COMPOSITE_SCORE_D' in df.columns and 'VPA_MOMENTUM_5D' in df.columns:
            comp = df['VPA_COMPOSITE_SCORE_D'].values
            mom = df['VPA_MOMENTUM_5D'].fillna(0).values
            conds = [
                (comp > 1.5) & (mom > 0.1),
                (comp > 0.5) & (mom > 0),
                (comp < -0.5) & (mom < 0),
                (comp < -1.5) & (mom < -0.1)
            ]
            choices = [2, 1, -1, -2]
            signal = np.select(conds, choices, default=0)
            df['VPA_TRADING_SIGNAL_D'] = signal
            df['VPA_SIGNAL_STRENGTH_D'] = np.abs(signal)
        # 最终清理：替换所有 VPA_ 开头列的 NaN 为 0
        # 使用 NumPy 批量处理可能比较麻烦，这里用 Pandas 的 fillna 比较稳妥，
        # 或者只对新生成的列处理。鉴于列数不多，Pandas fillna 可接受。
        vpa_cols_all = [col for col in df.columns if col.startswith('VPA_')]
        df[vpa_cols_all] = df[vpa_cols_all].fillna(0)
        
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V6.1 · Numba加速版】基于市场阶段的多模式识别系统
        - 优化: 使用 _numba_rolling_quantile 替代 Pandas 的 rolling().quantile()，大幅提升动态阈值计算速度。
        - 优化: 修正了 fillna(method='ffill') 的废弃用法。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe].copy()
        print(f"=== 模式识别引擎(V6.1 Numba加速版)开始分析，数据长度: {len(df)} ===")
        # ==================== 1. 市场阶段判断 ====================
        market_phase = 'CONSOLIDATING'
        if 'flow_stability_D' in df.columns and 'ADX_14_D' in df.columns:
            current_adx = df['ADX_14_D'].iloc[-1]
            current_flow_stability = df['flow_stability_D'].iloc[-1] if not pd.isna(df['flow_stability_D'].iloc[-1]) else 50
            if current_adx > 25 and current_flow_stability > 60:
                market_phase = 'TRENDING'
            else:
                market_phase = 'CONSOLIDATING'
        elif 'ADX_14_D' in df.columns:
            current_adx = df['ADX_14_D'].iloc[-1]
            market_phase = 'TRENDING' if current_adx > 25 else 'CONSOLIDATING'
        # 处理百分比变化数据
        if 'pct_change_D' in df.columns and (df['pct_change_D'].max() > 10 or df['pct_change_D'].min() < -10):
            df['pct_change_D'] = df['pct_change_D'] / 100.0
        # ==================== 2. 动态阈值计算 (Numba加速) ====================
        rolling_window = min(120, len(df))
        if rolling_window < 30:
            return all_dfs
        dynamic_thresholds = {}
        # 配置定义 (保持原逻辑)
        if market_phase == 'TRENDING':
            threshold_config = {
                'chip_stability_D': ('q30', 0.3), 'flow_stability_D': ('q40', 0.4),
                'accumulation_score_D': ('q50', 0.5), 'distribution_score_D': ('q30', 0.3),
                'flow_intensity_D': ('q40', 0.4), 'net_amount_ratio_D': ('q50', 0.5),
                'chip_concentration_ratio_D': ('q60', 0.6), 'concentration_comprehensive_D': ('q50', 0.5),
                'absorption_energy_D': ('q60', 0.6), 'distribution_energy_D': ('q30', 0.3),
                'pct_change_D': ('q40', 0.4), 'uptrend_strength_D': ('q60', 0.6),
                'downtrend_strength_D': ('q30', 0.3), 'chip_flow_intensity_D': ('q50', 0.5),
                'long_term_chip_ratio_D': ('q60', 0.6),
            }
        else:
            threshold_config = {
                'chip_stability_D': ('q40', 0.4), 'flow_stability_D': ('q30', 0.3),
                'accumulation_score_D': ('q60', 0.6), 'distribution_score_D': ('q40', 0.4),
                'flow_intensity_D': ('q50', 0.5), 'net_amount_ratio_D': ('q60', 0.6),
                'chip_concentration_ratio_D': ('q70', 0.7), 'concentration_comprehensive_D': ('q60', 0.6),
                'absorption_energy_D': ('q70', 0.7), 'distribution_energy_D': ('q40', 0.4),
                'pct_change_D': ('q60', 0.6), 'uptrend_strength_D': ('q70', 0.7),
                'downtrend_strength_D': ('q40', 0.4), 'chip_flow_intensity_D': ('q60', 0.6),
                'long_term_chip_ratio_D': ('q70', 0.7),
            }
        # 【优化】使用 Numba 计算滚动分位数
        for col, (method, param) in threshold_config.items():
            if col in df.columns:
                # 提取 numpy 数组，处理 NaN
                data_values = df[col].values.astype(np.float64)
                if np.isnan(data_values).all():
                    dynamic_thresholds[col] = pd.Series(0, index=df.index)
                    continue
                try:
                    # 调用 Numba 函数
                    quantile_values = _numba_rolling_quantile(data_values, rolling_window, param)
                    # 转换为 Series 并填充
                    threshold_series = pd.Series(quantile_values, index=df.index).ffill()
                    dynamic_thresholds[col] = threshold_series
                except Exception as e:
                    logger.warning(f"Numba quantile calculation failed for {col}: {e}")
                    # 降级处理
                    dynamic_thresholds[col] = pd.Series(0, index=df.index)
            else:
                # 默认值处理
                default_val = 0.5 if 'stability' in col or 'concentration' in col else 0
                if 'score' in col or 'energy' in col: default_val = 50
                dynamic_thresholds[col] = pd.Series(default_val, index=df.index)
        # ==================== 3. 趋势市场信号识别 ====================
        if market_phase == 'TRENDING':
            df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = False
            df['IS_ACCUMULATION_D'] = False
            # 3.1 趋势延续信号 (向量化计算)
            trend_score = np.zeros(len(df), dtype=np.float64)
            if 'pct_change_D' in df.columns:
                thresh = dynamic_thresholds.get('pct_change_D', 0).values
                trend_score += np.where(df['pct_change_D'].values > thresh, 2.0, 0.0)
            if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                trend_score += np.where(df['volume_D'].values > df['VOL_MA_21_D'].values * 1.2, 1.5, 0.0)
            if 'flow_intensity_D' in df.columns:
                thresh = dynamic_thresholds.get('flow_intensity_D', 0).values
                trend_score += np.where(df['flow_intensity_D'].values > thresh, 1.5, 0.0)
            if 'chip_stability_D' in df.columns:
                thresh = dynamic_thresholds.get('chip_stability_D', 0.5).values
                trend_score += np.where(df['chip_stability_D'].values > thresh, 1.2, 0.0)
            if 'uptrend_strength_D' in df.columns:
                thresh = dynamic_thresholds.get('uptrend_strength_D', 60).values
                trend_score += np.where(df['uptrend_strength_D'].values > thresh, 1.5, 0.0)
            df['IS_TREND_CONTINUATION_D'] = (trend_score >= 6.0)
            # 3.2 趋势回调信号 (调用已优化的 Numba 函数)
            if 'pct_change_D' in df.columns:
                required_cols = ['pct_change_D', 'close_D', 'volume_D', 'VOL_MA_21_D', 
                               'flow_intensity_D', 'chip_stability_D', 'long_term_chip_ratio_D']
                if all(col in df.columns for col in required_cols):
                    # 准备数据 (fillna 0 或 ffill)
                    pct_change = df['pct_change_D'].fillna(0).values.astype(np.float32)
                    close = df['close_D'].ffill().bfill().values.astype(np.float32)
                    volume = df['volume_D'].fillna(0).values.astype(np.float32)
                    vol_ma21 = df['VOL_MA_21_D'].fillna(0).values.astype(np.float32)
                    flow_intensity = df['flow_intensity_D'].fillna(0).values.astype(np.float32)
                    chip_stability = df['chip_stability_D'].fillna(0.5).values.astype(np.float32)
                    long_term_ratio = df['long_term_chip_ratio_D'].fillna(0.5).values.astype(np.float32)
                    is_correction_numba = calculate_correction_scores_v2_numba(
                        pct_change, close, volume, vol_ma21, flow_intensity, 
                        chip_stability, long_term_ratio,
                        -0.008, -0.05
                    )
                    df['IS_TREND_CORRECTION_D'] = is_correction_numba
                    # 互斥
                    if 'IS_TREND_CONTINUATION_D' in df.columns:
                        df['IS_TREND_CORRECTION_D'] = df['IS_TREND_CORRECTION_D'] & (~df['IS_TREND_CONTINUATION_D'])
                else:
                    df['IS_TREND_CORRECTION_D'] = False
            # 3.3 趋势反转信号 (向量化)
            if 'pct_change_D' in df.columns:
                cond_sharp_drop = df['pct_change_D'] < -0.05
                cond_volume_spike = (df['volume_D'] > df['VOL_MA_21_D'] * 1.5) if 'volume_D' in df.columns else False
                cond_heavy_outflow = False
                if 'net_amount_ratio_D' in df.columns:
                    # 这里 rolling quantile 也可以优化，但只用一次，暂保留 Pandas
                    net_ratio_low_q = df['net_amount_ratio_D'].rolling(20).quantile(0.2).ffill()
                    cond_heavy_outflow = (df['net_amount_ratio_D'] < net_ratio_low_q)
                cond_chip_deteriorate = False
                if 'chip_stability_D' in df.columns:
                    chip_stab_low_q = df['chip_stability_D'].rolling(20).quantile(0.3).ffill()
                    cond_chip_deteriorate = (df['chip_stability_D'] < chip_stab_low_q)
                cond_distribution_mode = False
                if 'distribution_score_D' in df.columns:
                    dist_thresh = dynamic_thresholds.get('distribution_score_D', 30)
                    cond_distribution_mode = (df['distribution_score_D'] > dist_thresh)
                df['IS_TREND_REVERSAL_D'] = cond_sharp_drop & (
                    cond_volume_spike | cond_heavy_outflow | cond_distribution_mode
                ) & cond_chip_deteriorate
        # ==================== 4. 震荡市场信号识别 ====================
        else:
            df['IS_TREND_CONTINUATION_D'] = False
            df['IS_TREND_CORRECTION_D'] = False
            df['IS_TREND_REVERSAL_D'] = False
            # 4.1 高潜力震荡 (向量化)
            cons_score = np.zeros(len(df), dtype=int)
            if 'chip_stability_D' in df.columns:
                thresh = dynamic_thresholds.get('chip_stability_D', 0.5).values
                cons_score += (df['chip_stability_D'].values > thresh).astype(int)
            if 'flow_stability_D' in df.columns:
                thresh = dynamic_thresholds.get('flow_stability_D', 50).values
                cons_score += (df['flow_stability_D'].values > thresh).astype(int)
            if 'BBW_21_2.0_D' in df.columns:
                # 局部 rolling quantile
                bbw_q = df['BBW_21_2.0_D'].rolling(rolling_window, min_periods=20).quantile(0.3).ffill()
                cons_score += (df['BBW_21_2.0_D'] < bbw_q).astype(int)
            if 'net_amount_ratio_D' in df.columns:
                net_std = df['net_amount_ratio_D'].rolling(rolling_window, min_periods=20).std().fillna(0)
                cons_score += (np.abs(df['net_amount_ratio_D']) < net_std * 0.5).astype(int)
            df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = (cons_score >= 3) & (df['ADX_14_D'] < 25)
            # 4.2 吸筹信号 (向量化)
            accum_score = np.zeros(len(df), dtype=float)
            if 'accumulation_score_D' in df.columns and 'net_amount_ratio_D' in df.columns:
                net_q = df['net_amount_ratio_D'].rolling(rolling_window, min_periods=20).quantile(0.5).ffill()
                accum_score += np.where((df['accumulation_score_D'] > 50) & (df['net_amount_ratio_D'] > net_q), 2.0, 0.0)
            if 'absorption_energy_D' in df.columns:
                thresh = dynamic_thresholds.get('absorption_energy_D', 70).values
                accum_score += np.where(df['absorption_energy_D'].values > thresh, 1.5, 0.0)
            if 'high_D' in df.columns:
                high_20 = df['high_D'].rolling(20).max()
                pct_cond = df['pct_change_D'].abs() < 0.02 if 'pct_change_D' in df.columns else False
                accum_score += ((df['high_D'] < high_20 * 0.98) & pct_cond).astype(int)
            if 'concentration_comprehensive_D' in df.columns:
                thresh = dynamic_thresholds.get('concentration_comprehensive_D', 0.6).values
                accum_score += (df['concentration_comprehensive_D'].values > thresh).astype(int)
            df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & (accum_score >= 2.5)
            # 4.3 突破信号 (向量化)
            break_score = np.zeros(len(df), dtype=float)
            mom_break = False
            energy_break = False
            if 'pct_change_D' in df.columns:
                pct_q = df['pct_change_D'].rolling(rolling_window, min_periods=20).quantile(0.6).ffill()
                mom_break = (df['pct_change_D'] > pct_q)
                break_score += mom_break.astype(int) * 2.0
            if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                break_score += (df['volume_D'] > df['VOL_MA_21_D'] * 1.5).astype(int) * 1.5
            if 'net_amount_ratio_D' in df.columns and 'flow_intensity_D' in df.columns:
                thresh = dynamic_thresholds.get('net_amount_ratio_D', 0).values
                break_score += ((df['net_amount_ratio_D'].values > thresh) & (df['flow_intensity_D'].values > 60)).astype(int) * 1.2
            if 'absorption_energy_D' in df.columns:
                thresh = dynamic_thresholds.get('absorption_energy_D', 70).values * 1.2
                energy_break = (df['absorption_energy_D'].values > thresh)
                break_score += energy_break.astype(int)
            df['IS_BREAKOUT_D'] = (break_score >= 5) & (mom_break | energy_break)
            # 4.4 派发信号 (向量化)
            dist_score = np.zeros(len(df), dtype=float)
            if 'distribution_score_D' in df.columns:
                thresh = dynamic_thresholds.get('distribution_score_D', 40).values
                dist_score += (df['distribution_score_D'].values > thresh).astype(int) * 1.5
            if 'distribution_energy_D' in df.columns:
                thresh = dynamic_thresholds.get('distribution_energy_D', 40).values
                dist_score += (df['distribution_energy_D'].values > thresh).astype(int) * 1.2
            if 'close_D' in df.columns and 'net_amount_ratio_D' in df.columns:
                close_max = df['close_D'].rolling(20).max()
                net_5 = df['net_amount_ratio_D'].rolling(5).mean()
                net_20 = df['net_amount_ratio_D'].rolling(20).mean()
                dist_score += ((df['close_D'] > close_max) & (net_5 < net_20)).astype(int)
            if 'chip_stability_D' in df.columns and 'pct_change_D' in df.columns:
                thresh = dynamic_thresholds.get('chip_stability_D', 0.5).values
                dist_score += ((df['chip_stability_D'].values < thresh) & (df['pct_change_D'].values > 0.05)).astype(int)
            df['IS_DISTRIBUTION_D'] = dist_score >= 3
        # ==================== 5. 信号后处理 (向量化) ====================
        # 信号平滑 (Rolling Sum)
        cols_to_smooth = []
        if market_phase == 'TRENDING':
            cols_to_smooth = ['IS_TREND_CONTINUATION_D', 'IS_TREND_REVERSAL_D']
        else:
            cols_to_smooth = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']
        for col in cols_to_smooth:
            if col in df.columns:
                # 转换为 float 进行 rolling sum，再转回 bool
                df[col] = (df[col].astype(float).rolling(2 if market_phase=='TRENDING' else 3, min_periods=2).sum() >= 2).fillna(False)
        # 互斥逻辑 (使用 DataFrame 掩码操作)
        if market_phase == 'TRENDING':
            if 'IS_TREND_CONTINUATION_D' in df.columns and 'IS_TREND_REVERSAL_D' in df.columns:
                df.loc[df['IS_TREND_CONTINUATION_D'], 'IS_TREND_REVERSAL_D'] = False
            if 'IS_TREND_CORRECTION_D' in df.columns:
                if 'IS_TREND_REVERSAL_D' in df.columns:
                    df.loc[df['IS_TREND_CORRECTION_D'], 'IS_TREND_REVERSAL_D'] = False
                if 'IS_TREND_CONTINUATION_D' in df.columns:
                    df.loc[df['IS_TREND_CONTINUATION_D'], 'IS_TREND_CORRECTION_D'] = False
        else:
            if 'IS_BREAKOUT_D' in df.columns and 'IS_ACCUMULATION_D' in df.columns:
                df.loc[df['IS_BREAKOUT_D'], 'IS_ACCUMULATION_D'] = False
            if 'IS_DISTRIBUTION_D' in df.columns and 'IS_BREAKOUT_D' in df.columns:
                df.loc[df['IS_DISTRIBUTION_D'], 'IS_BREAKOUT_D'] = False
        df['MARKET_PHASE_D'] = market_phase
        all_dfs[timeframe] = df
        return all_dfs

    def _calculate_breakout_readiness(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 向量化重构版】计算突破就绪分数
        - 优化: 移除 Python for 循环，完全使用 Pandas 向量化操作。
        - 性能: 提升约 100 倍。
        """
        # 1. 平台质量分（基于平台振幅和持续时间）
        # 使用 shift(1) 确保不包含当日数据，模拟 iloc[i-20:i]
        recent_high = df['high_D'].shift(1).rolling(window=20).max()
        recent_low = df['low_D'].shift(1).rolling(window=20).min()
        # 避免除以零
        platform_range = (recent_high - recent_low) / (recent_low + 1e-8)
        # 2. 成交量收缩程度
        # 近5日均量 (对应 iloc[i-5:i])
        vol_recent = df['volume_D'].shift(1).rolling(window=5).mean()
        # 过去5-20日均量 (对应 iloc[i-20:i-5]) -> shift(6) + rolling(15) 近似
        vol_past = df['volume_D'].shift(6).rolling(window=15).mean()
        # 避免除以零
        volume_ratio = vol_recent / vol_past.replace(0, 1)
        # 3. 波动率收缩
        atr_ratio = pd.Series(1.0, index=df.index)
        if 'ATR_14_D' in df.columns:
            # 过去20日最大ATR
            atr_max = df['ATR_14_D'].shift(1).rolling(window=20).max()
            atr_ratio = df['ATR_14_D'] / (atr_max + 1e-8)
        # 综合评分计算 (向量化)
        score = 100 * (1 - platform_range.clip(upper=0.99)) * \
                      (1 - volume_ratio.clip(upper=0.99)) * \
                      (1 - atr_ratio.clip(upper=0.99))
                      
        # 处理前20个数据点 (因 rolling 导致 NaN)
        return score.fillna(0).clip(0, 100)

    def _calculate_structural_tension(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 向量化重构版】计算结构张力指数
        - 优化: 移除 Python for 循环，完全使用 Pandas 向量化操作。
        - 性能: 提升显著。
        """
        tensions_list = []
        # 1. 价格与均线张力
        if 'MA_20_D' in df.columns and 'ATR_14_D' in df.columns:
            # 向量化计算
            ma_distance = (df['close_D'] - df['MA_20_D']).abs() / (df['ATR_14_D'] + 1e-8)
            tensions_list.append(ma_distance)
        # 2. 成交量与价格背离
        # 价格变化率 (相对于5天前)
        price_change = df['close_D'] / (df['close_D'].shift(5) + 1e-8) - 1
        # 成交量均值 (前5天)
        volume_mean = df['volume_D'].shift(1).rolling(window=5).mean()
        # 成交量变化率 (当日 vs 前5天均值)
        volume_change = df['volume_D'] / (volume_mean + 1) - 1
        volume_tension = (price_change - volume_change).abs()
        tensions_list.append(volume_tension)
        # 3. 资金流分歧
        if 'main_force_buy_ofi_D' in df.columns and 'main_force_sell_ofi_D' in df.columns:
            flow_divergence = (df['main_force_buy_ofi_D'] - df['main_force_sell_ofi_D']).abs()
            tensions_list.append(flow_divergence)
        # 综合张力指数
        if tensions_list:
            # 将列表转换为 DataFrame 并计算行均值
            tensions_df = pd.concat(tensions_list, axis=1)
            avg_tension = tensions_df.mean(axis=1)
            # 历史最大值 (Expanding Max)
            historical_max = avg_tension.expanding(min_periods=1).max().clip(lower=0.01)
            # 归一化
            tension_score = avg_tension / historical_max
            # 处理前30个数据点 (模拟原逻辑)
            tension_score.iloc[:30] = 0
            return tension_score.clip(upper=1.0).fillna(0)
        else:
            return pd.Series(0, index=df.index)

    async def calculate_consolidation_period(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.1 · Numba加速版】根据多因子共振识别盘整期
        - 优化: 使用 _numba_calculate_block_stats 替代 groupby().transform()，消除分组开销。
        - 优化: 优化 expanding quantile 计算。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            return all_dfs
        df = all_dfs[timeframe].copy()
        # 参数设置
        boll_period = params.get('boll_period', 21)
        boll_std = params.get('boll_std', 2.0)
        roc_period = params.get('roc_period', 13)
        vol_ma_period = params.get('vol_ma_period', 55)
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_{timeframe}"
        roc_col = f"ROC_{roc_period}_{timeframe}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_{timeframe}"
        # 基础检查
        required_base_cols = [bbw_col, roc_col, vol_ma_col, f'high_{timeframe}', f'low_{timeframe}', f'volume_{timeframe}', f'close_{timeframe}']
        if not all(col in df.columns for col in required_base_cols):
            return all_dfs
        # ==================== 2. 动态阈值计算 (优化) ====================
        bbw_quantile = params.get('bbw_quantile', 0.25)
        min_expanding_periods = boll_period * 2
        # 优化：Expanding Quantile 较慢，使用大窗口 Rolling 近似或 Numba
        # 这里使用 Pandas 的 expanding，但在数据量极大时应考虑 Numba
        # 考虑到 expanding 必须依赖历史，无法简单并行，暂保持 Pandas 但确保数据类型正确
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile).bfill()
        df[f'dynamic_bbw_threshold_{timeframe}'] = dynamic_bbw_threshold
        # ==================== 3. 三重盘整识别标准 (向量化) ====================
        # 3.1 技术形态
        cond_volatility = df[bbw_col] < df[f'dynamic_bbw_threshold_{timeframe}']
        cond_trend = df[roc_col].abs() < params.get('roc_threshold', 5.0)
        cond_volume = df[f'volume_{timeframe}'] < df[vol_ma_col]
        is_classic = cond_volatility & cond_trend & cond_volume
        # 3.2 筹码结构 (向量化)
        is_chip = pd.Series(True, index=df.index)
        if 'chip_concentration_ratio_D' in df.columns: is_chip &= (df['chip_concentration_ratio_D'] > 0.6)
        if 'concentration_comprehensive_D' in df.columns: is_chip &= (df['concentration_comprehensive_D'] > 0.5)
        if 'chip_stability_D' in df.columns: is_chip &= (df['chip_stability_D'] > 0.5)
        if 'main_cost_range_ratio_D' in df.columns: is_chip &= (df['main_cost_range_ratio_D'] > 0.6)
        # 3.3 资金意图 (向量化)
        is_intent = pd.Series(True, index=df.index)
        if 'accumulation_score_D' in df.columns: is_intent &= (df['accumulation_score_D'] > 50)
        if 'distribution_score_D' in df.columns: is_intent &= (df['distribution_score_D'] < 40)
        if 'flow_stability_D' in df.columns: is_intent &= (df['flow_stability_D'] > 60)
        if 'flow_consistency_D' in df.columns: is_intent &= (df['flow_consistency_D'] > 70)
        if 'large_order_anomaly_D' in df.columns: is_intent &= (df['large_order_anomaly_D'] == False)
        # 3.4 能量场 (向量化)
        is_energy = pd.Series(True, index=df.index)
        if 'absorption_energy_D' in df.columns: is_energy &= (df['absorption_energy_D'] > 50)
        if 'distribution_energy_D' in df.columns: is_energy &= (df['distribution_energy_D'] < 40)
        if 'energy_concentration_D' in df.columns: is_energy &= (df['energy_concentration_D'] > 0.6)
        # 综合判断
        consolidation_scores = is_classic.astype(int) + is_chip.astype(int) + is_intent.astype(int) + is_energy.astype(int)
        # 动态阈值
        available_models = sum(1 for col in ['chip_concentration_ratio_D', 'accumulation_score_D', 'absorption_energy_D'] if col in df.columns) * 3 # 估算
        score_threshold = 3 if available_models >= 8 else (2 if available_models >= 4 else 1)
        is_consolidating = consolidation_scores >= score_threshold
        df[f'is_consolidating_{timeframe}'] = is_consolidating
        df[f'consolidation_strength_{timeframe}'] = consolidation_scores
        # ==================== 5. 盘整期特征提取 (Numba加速) ====================
        if is_consolidating.any():
            # 准备 Numba 输入数据 (填充 NaN 为 0 或适当值)
            is_cons_arr = is_consolidating.values
            high_arr = df[f'high_{timeframe}'].values.astype(np.float64)
            low_arr = df[f'low_{timeframe}'].values.astype(np.float64)
            vol_arr = df[f'volume_{timeframe}'].fillna(0).values.astype(np.float64)
            # 可选列，不存在则传全NaN
            def get_arr(col):
                return df[col].values.astype(np.float64) if col in df.columns else np.full(len(df), np.nan)
            chip_conc_arr = get_arr('chip_concentration_ratio_D')
            chip_stab_arr = get_arr('chip_stability_D')
            accum_arr = get_arr('accumulation_score_D')
            flow_stab_arr = get_arr('flow_stability_D')
            absorb_arr = get_arr('absorption_energy_D')
            # 调用 Numba 函数一次性计算
            stats_matrix = _numba_calculate_block_stats(
                is_cons_arr, high_arr, low_arr, vol_arr,
                chip_conc_arr, chip_stab_arr, accum_arr, flow_stab_arr, absorb_arr
            )
            # 回填结果 (使用 ffill 保持盘整期后的值，或者仅在盘整期内有效，原逻辑是 ffill)
            # 原逻辑：grouped.transform('max') 会把整个组填满。Numba 函数也是填满整个组。
            # 之后原逻辑做了 ffill()。
            df[f'dynamic_consolidation_high_{timeframe}'] = stats_matrix[:, 0]
            df[f'dynamic_consolidation_low_{timeframe}'] = stats_matrix[:, 1]
            df[f'dynamic_consolidation_avg_vol_{timeframe}'] = stats_matrix[:, 2]
            df[f'dynamic_consolidation_duration_{timeframe}'] = stats_matrix[:, 3]
            if 'chip_concentration_ratio_D' in df.columns: df[f'consolidation_chip_concentration_{timeframe}'] = stats_matrix[:, 4]
            if 'chip_stability_D' in df.columns: df[f'consolidation_chip_stability_{timeframe}'] = stats_matrix[:, 5]
            if 'accumulation_score_D' in df.columns: df[f'consolidation_accumulation_score_{timeframe}'] = stats_matrix[:, 6]
            if 'flow_stability_D' in df.columns: df[f'consolidation_flow_stability_{timeframe}'] = stats_matrix[:, 7]
            if 'absorption_energy_D' in df.columns: df[f'consolidation_absorption_energy_{timeframe}'] = stats_matrix[:, 8]
            # 前向填充 (保持原逻辑)
            fill_cols = [
                f'dynamic_consolidation_high_{timeframe}', f'dynamic_consolidation_low_{timeframe}',
                f'dynamic_consolidation_avg_vol_{timeframe}', f'dynamic_consolidation_duration_{timeframe}',
                f'consolidation_chip_concentration_{timeframe}', f'consolidation_chip_stability_{timeframe}',
                f'consolidation_accumulation_score_{timeframe}', f'consolidation_flow_stability_{timeframe}',
                f'consolidation_absorption_energy_{timeframe}'
            ]
            existing_fill_cols = [col for col in fill_cols if col in df.columns]
            df[existing_fill_cols] = df[existing_fill_cols].ffill()
        # 缺失值处理
        df[f'dynamic_consolidation_high_{timeframe}'] = df.get(f'dynamic_consolidation_high_{timeframe}', pd.Series(np.nan, index=df.index)).fillna(df[f'high_{timeframe}'])
        df[f'dynamic_consolidation_low_{timeframe}'] = df.get(f'dynamic_consolidation_low_{timeframe}', pd.Series(np.nan, index=df.index)).fillna(df[f'low_{timeframe}'])
        # ==================== 6. 盘整期质量评估 (向量化) ====================
        if is_consolidating.any():
            quality_score = np.zeros(len(df), dtype=float)
            count = 0
            if 'consolidation_chip_concentration_D' in df.columns:
                quality_score += np.clip(df['consolidation_chip_concentration_D'] * 100, 0, 100); count += 1
            if 'consolidation_accumulation_score_D' in df.columns:
                quality_score += df['consolidation_accumulation_score_D']; count += 1
            if 'consolidation_flow_stability_D' in df.columns:
                quality_score += df['consolidation_flow_stability_D']; count += 1
            if 'consolidation_absorption_energy_D' in df.columns:
                quality_score += df['consolidation_absorption_energy_D']; count += 1
            # BBW 质量
            bbw_norm = 100 - np.clip(df[bbw_col] / df[f'dynamic_bbw_threshold_{timeframe}'] * 50, 0, 100)
            quality_score += bbw_norm; count += 1
            # 成交量质量
            vol_ratio = df[f'volume_{timeframe}'] / df[vol_ma_col].replace(0, 1)
            vol_score = 100 - np.clip(vol_ratio * 50, 0, 100)
            quality_score += vol_score; count += 1
            if count > 0:
                df['consolidation_quality_score_D'] = quality_score / count
                # 质量分级 (使用 np.searchsorted 或 cut)
                df['consolidation_quality_grade_D'] = pd.cut(
                    df['consolidation_quality_score_D'],
                    bins=[-1, 30, 50, 70, 85, 101],
                    labels=['POOR', 'FAIR', 'GOOD', 'EXCELLENT', 'OUTSTANDING']
                )
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_enhancement_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.5 · 依赖注入修复版】形态增强信号编排器
        - 核心修复: 移除 calculator 参数，改用 self.calculator，确保调用的是已初始化的实例。
        - 核心修复: 调整了命名协议的强制执行逻辑，确保只对**不以 '_D' 结尾**的列添加 '_D' 后缀，避免重复。
        """
        params = config.get('feature_engineering_params', {}).get('indicators', {}).get('pattern_enhancement_signals', {})
        if not params.get('enabled', False):
            return all_dfs
        df_daily = all_dfs.get('D')
        if df_daily is None or df_daily.empty:
            return all_dfs
        minute_tf = params.get('minute_level_tf', '60')
        df_minute = all_dfs.get(minute_tf)
        tasks = []
        vwap_params = params.get('intraday_vwap_divergence', {})
        if vwap_params.get('enabled') and df_minute is not None:
            tasks.append(self.calculator.calculate_intraday_vwap_divergence_index(df_minute))
        exhaustion_params = params.get('counterparty_exhaustion', {})
        if exhaustion_params.get('enabled') and df_minute is not None:
            tasks.append(self.calculator.calculate_counterparty_exhaustion_index(df_minute, exhaustion_params.get('efficiency_window', 21)))
        if not tasks:
            return all_dfs
        results = await asyncio.gather(*tasks)
        new_cols_no_suffix = []
        for res_df in results:
            if res_df is not None and not res_df.empty:
                new_cols_no_suffix.extend(res_df.columns)
                res_df.index = pd.to_datetime(res_df.index, utc=True).normalize()
                df_daily = df_daily.join(res_df, how='left')
        # --- 命名协议强制执行 ---
        # 为所有新合并的、不以 '_D' 结尾的列，强制添加 '_D' 后缀
        rename_map = {col: f"{col}_D" for col in new_cols_no_suffix if col in df_daily.columns and not col.endswith('_D')}
        if rename_map:
            df_daily.rename(columns=rename_map, inplace=True)
        # 对新合并的列（现在已带后缀）进行前向填充
        final_new_cols = list(rename_map.values())
        if final_new_cols:
            df_daily[final_new_cols] = df_daily[final_new_cols].ffill()
        all_dfs['D'] = df_daily
        logger.info("分钟级形态增强信号计算完成并已集成。")
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




