# 新增文件: services/feature_engineering_service.py

import asyncio
import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import pandas_ta as ta
import numba
from numba import objmode
from utils.math_tools import hurst_exponent
from strategies.trend_following.utils import _numba_nonlinear_fusion_core
logger = logging.getLogger("services")

@numba.njit(cache=True)
def _numba_rolling_fft_energy_ratio_core(
    data: np.ndarray,
    window: int,
    low_freq_cutoff_ratio: float
) -> np.ndarray:
    """
    【Numba优化版】计算滚动窗口内的FFT能量比。
    将原 _fft_energy_ratio 的逻辑内联到此滚动计算核心中。
    """
    n = len(data)
    results = np.full(n, np.nan, dtype=np.float64) # FFT结果通常为float64
    for i in range(n):
        if i < window - 1:
            continue
        window_data = data[i - window + 1 : i + 1]
        # 内联 _fft_energy_ratio 的逻辑
        N_window = len(window_data)
        if N_window < 2:
            results[i] = np.nan
            continue
        # Numba 的 np.fft.fft 需要浮点数输入，输出为复数
        # 确保输入是 float64，以避免类型转换问题
        # 使用 objmode 暂时退出 Numba 模式，调用 Python 的 np.fft.fft
        with objmode(yf='complex128[:]'): # 声明 yf 的类型为一维复数数组
            yf = np.fft.fft(window_data.astype(np.float64))
        yf_abs = np.abs(yf[:N_window // 2]) # 取正频率部分
        total_energy = np.sum(yf_abs**2)
        if total_energy == 0:
            results[i] = np.nan
            continue
        low_freq_idx = int(N_window * low_freq_cutoff_ratio)
        low_freq_energy = np.sum(yf_abs[:low_freq_idx]**2)
        results[i] = low_freq_energy / total_energy
        
    return results

@numba.njit(cache=True)
def _numba_rolling_slope(data: np.ndarray, window: int) -> np.ndarray:
    """
    【Numba优化】计算滚动线性回归斜率。
    原理: 最小二乘法 (OLS)
    Slope = (N * Σ(xy) - Σx * Σy) / (N * Σ(x^2) - (Σx)^2)
    其中 x 为时间序列 0, 1, ..., N-1，Σx 和 Σ(x^2) 对于固定窗口是常数。
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return result

    # 预计算 x 的相关项 (x 为 0 到 window-1)
    sum_x = (window - 1) * window / 2.0
    sum_x_sq = (window - 1) * window * (2 * window - 1) / 6.0
    denominator = window * sum_x_sq - sum_x * sum_x
    
    # 如果分母为0（窗口为1），则无法计算
    if denominator == 0:
        return result

    for i in range(window - 1, n):
        y_slice = data[i - window + 1 : i + 1]
        
        # 检查 NaN
        has_nan = False
        for val in y_slice:
            if np.isnan(val):
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
def _numba_higuchi_fd(x: np.ndarray, k_max: int) -> float:
    """
    【Numba优化】Higuchi分形维数计算核心。
    提取自原 calculate_meta_features 方法。
    """
    x_len = len(x)
    if x_len < 2:
        return np.nan
    
    L = np.empty(k_max, dtype=np.float64)
    L[:] = np.nan # 手动填充 NaN
    
    for k in range(1, k_max + 1):
        Lk_sum = 0.0
        count = 0
        for m in range(k):
            # 计算子序列长度
            # n_k = floor((N - m - 1) / k) + 1
            n_k = (x_len - m - 1) // k + 1
            if n_k < 2:
                continue
            sum_diff = 0.0
            for i in range(1, n_k):
                sum_diff += np.abs(x[m + i * k] - x[m + (i - 1) * k])
                
            norm_factor = (x_len - 1) / (n_k * k) # 这里简化处理，原文可能有细微差异，保持标准Higuchi
            # 原代码逻辑: denominator = ((x_len - m) // k * k) -> 实际上近似 n_k * k
            # 保持原代码逻辑的等价实现
            denominator = ((x_len - m) // k) * k
            if denominator == 0:
                continue
                
            Lk_sum += sum_diff * (x_len - 1) / denominator
            count += 1
            
        if count > 0 and Lk_sum > 0:
            L[k-1] = np.log(Lk_sum / count / k) # Log(L(k))
    
    # 线性回归计算斜率
    # x轴: log(1/k) 或 -log(k)。Higuchi定义 FD = -slope of log(L(k)) vs log(k)
    # 原代码用 log(k) 和 log(L)，slope应为负，取绝对值
    valid_mask = ~np.isnan(L)
    valid_L = L[valid_mask]
    if len(valid_L) < 2:
        return np.nan
        
    k_values = np.arange(1, k_max + 1, dtype=np.float64)
    valid_k_log = np.log(k_values[valid_mask])
    
    N = len(valid_L)
    sum_x = np.sum(valid_k_log)
    sum_y = np.sum(valid_L)
    sum_xy = np.sum(valid_k_log * valid_L)
    sum_x2 = np.sum(valid_k_log * valid_k_log)
    
    denom = N * sum_x2 - sum_x * sum_x
    if denom == 0:
        return np.nan
        
    slope = (N * sum_xy - sum_x * sum_y) / denom
    fd = np.abs(slope)
    
    # 截断结果
    if fd < 1.0: return 1.0
    if fd > 2.0: return 2.0
    return fd

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

@numba.njit(cache=True)
def _numba_rolling_sample_entropy(data: np.ndarray, window: int, tol_ratio: float, rolling_std: np.ndarray) -> np.ndarray:
    """
    【Numba优化】滚动计算样本熵。
    将原本在 Python 中的 for 循环移入 Numba，消除循环开销。
    """
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    
    if n < window:
        return result
        
    for i in range(window - 1, n):
        # 获取当前窗口数据
        window_data = data[i - window + 1 : i + 1]
        
        # 获取预计算的 std (注意索引对齐)
        std_val = rolling_std[i]
        if np.isnan(std_val) or std_val == 0:
            continue
            
        r = std_val * tol_ratio
        
        # 调用核心计算函数
        se = _numba_sample_entropy_core(window_data, 2, r)
        result[i] = se
        
    return result

@numba.njit(cache=True)
def _numba_spearman_orderliness(ma_values: np.ndarray, ma_ranks_x: np.ndarray) -> np.ndarray:
    """
    【Numba优化】行级 Spearman 秩相关系数计算。
    用于衡量均线排列的有序度。
    ma_values: (Rows, Cols) 每一行是不同周期的均线值
    ma_ranks_x: (Cols,) 均线周期的理论排名 (例如 1, 2, 3...)
    """
    n_rows, n_cols = ma_values.shape
    results = np.zeros(n_rows, dtype=np.float32)
    
    if n_cols <= 1:
        return results

    # 常数项：n(n^2 - 1)
    denom = n_cols * (n_cols * n_cols - 1.0)
    
    for i in range(n_rows):
        row = ma_values[i, :]
        
        # 检查 NaN
        has_nan = False
        for val in row:
            if np.isnan(val):
                has_nan = True
                break
        if has_nan:
            results[i] = 0.0
            continue
            
        # 计算当前行的排名 (Ordinal Rank)
        # Numba 中没有直接的 rank，使用 argsort().argsort()
        # 第一次 argsort 获取排序后的索引
        # 第二次 argsort 获取原位置对应的排名 (0-based)
        # 注意：这里不处理平局 (ties)，对于均线这种连续浮点数，完全相等的概率极低
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
            
        # Spearman 公式: 1 - 6 * Σd^2 / (n(n^2-1))
        rho = 1.0 - (6.0 * d_sq_sum) / denom
        results[i] = rho
        
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
        【V3.4 · Numba加速版】计算所有配置的斜率特征。
        - 核心逻辑: 使用 Numba 优化的 _numba_rolling_slope 替代 pandas_ta.linreg。
        - 性能提升: 避免了循环调用 pandas_ta 时的 DataFrame 构建开销，直接在 NumPy 数组上进行极速运算。
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
                logger.warning(f"SLOPE计算跳过: 周期 '{timeframe}' 的源列 '{col_pattern}' 不存在。")
                continue
                
            # 提取源数据为 NumPy 数组，确保 float64 类型
            source_values = df[col_pattern].astype(float).values
            for lookback in lookbacks:
                slope_col_name = f'SLOPE_{lookback}_{col_pattern}'
                if slope_col_name in df.columns:
                    continue
                # 【Numba加速】直接调用编译好的滚动斜率函数
                slope_values = _numba_rolling_slope(source_values, int(lookback))
                # 填充结果，保持与原 DataFrame 索引一致
                df[slope_col_name] = pd.Series(slope_values, index=df.index).fillna(0)
                
            all_dfs[timeframe] = df
            
        return all_dfs

    async def calculate_all_accelerations(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.4 · Numba加速版】计算所有配置的加速度特征。
        - 核心逻辑: 同样使用 _numba_rolling_slope 对斜率列再次求导。
        - 性能提升: 向量化运算，消除 Python 循环开销。
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
                    logger.warning(f"ACCEL计算跳过: 周期 '{timeframe}' 的依赖斜率列 '{slope_col_name}' 不存在。")
                    continue
                    
                accel_col_name = f'ACCEL_{period}_{base_col_name}'
                if accel_col_name in df.columns:
                    continue
                # 提取斜率数据
                slope_values = df[slope_col_name].values.astype(float)
                # 【Numba加速】计算加速度（斜率的斜率）
                accel_values = _numba_rolling_slope(slope_values, int(period))
                df[accel_col_name] = pd.Series(accel_values, index=df.index).fillna(0)
                
        return all_dfs

    async def calculate_vpa_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.2 · 向量化加速版】VPA效率指标生产线
        - 优化: 使用 NumPy 的向量化函数 (np.maximum, np.minimum) 替换 Pandas 的 apply(lambda) 操作。
        - 性能提升: 将计算下沉至 C 层，大幅提高大规模数据下的计算速度。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        
        required_cols = ['pct_change_D', 'volume_D', 'VOL_MA_21_D', 'main_force_buy_ofi_D', 'main_force_sell_ofi_D']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"VPA效率生产线缺少关键数据: {missing}，模块已跳过！")
            return all_dfs
        
        # 1. 计算总VPA效率
        volume_ratio = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        vpa_efficiency = df['pct_change_D'] / volume_ratio.replace(0, np.nan)
        df['VPA_EFFICIENCY_D'] = vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 2. 计算买方VPA效率 (VPA_BUY_EFFICIENCY_D)
        mf_buy_ofi_ma_21 = df['main_force_buy_ofi_D'].rolling(window=21, min_periods=1).mean()
        buy_flow_ratio = df['main_force_buy_ofi_D'] / mf_buy_ofi_ma_21.replace(0, np.nan)
        
        # 【向量化优化】使用 np.maximum 替代 apply(lambda x: max(0, x))
        positive_pct_change = np.maximum(df['pct_change_D'].values, 0)
        buy_vpa_efficiency = positive_pct_change / buy_flow_ratio.replace(0, np.nan)
        df['VPA_BUY_EFFICIENCY_D'] = buy_vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 3. 计算卖方VPA效率 (VPA_SELL_EFFICIENCY_D)
        mf_sell_ofi_ma_21 = df['main_force_sell_ofi_D'].rolling(window=21, min_periods=1).mean()
        sell_flow_ratio = df['main_force_sell_ofi_D'] / mf_sell_ofi_ma_21.replace(0, np.nan)
        
        # 【向量化优化】使用 np.minimum 替代 apply(lambda x: min(0, x))
        negative_pct_change = np.minimum(df['pct_change_D'].values, 0)
        sell_vpa_efficiency = negative_pct_change / sell_flow_ratio.replace(0, np.nan)
        df['VPA_SELL_EFFICIENCY_D'] = sell_vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)

        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_meta_features(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """【V3.6 · Numba重构版】元特征计算车间 - 动态阈值版"""
        timeframe = 'D'
        if timeframe not in all_dfs:
            print(f"calculate_meta_features: {timeframe}不在all_dfs中")
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        params = config.get('feature_engineering_params', {}).get('meta_feature_params', {})
        if not params.get('enabled', False):
            print("calculate_meta_features: 元特征计算未启用")
            return all_dfs
        print(f"calculate_meta_features: 输入数据形状 {df.shape}")
        source_series_configs = [
            {'col': f'close{suffix}', 'prefix': ''},
            {'col': f'main_force_buy_ofi{suffix}', 'prefix': 'MF_BUY_OFI_'},
            {'col': f'bid_side_liquidity{suffix}', 'prefix': 'BID_LIQUIDITY_'}
        ]
        print(f"calculate_meta_features: 源列配置 {source_series_configs}")
        for src_config in source_series_configs:
            source_col = src_config['col']
            prefix = src_config['prefix']
            print(f"计算元特征: 处理源列 {source_col}, 前缀 {prefix}")
            if source_col not in df.columns:
                print(f"警告: 源列 {source_col} 不存在, 可用列: {list(df.columns)[:10]}...")
                logger.warning(f"元特征计算缺少核心列 '{source_col}'，跳过其元特征计算。")
                continue
            print(f"源列 {source_col} 存在, 非空值数量: {df[source_col].notna().sum()}")
            current_series = df[source_col]
            if isinstance(current_series, pd.DataFrame):
                current_series = current_series.iloc[:, 0]
            clean_series = current_series.dropna()
            clean_values = clean_series.values.astype(np.float64)
            print(f"清理后数据长度: {len(clean_values)}")
            hurst_window = params.get('hurst_window', 144)
            hurst_col = f'{prefix}HURST_{hurst_window}d{suffix}'
            print(f"准备计算赫斯特指数, 目标列: {hurst_col}, 窗口: {hurst_window}")
            if hurst_col not in df.columns:
                try:
                    if len(clean_values) >= hurst_window:
                        print(f"开始计算赫斯特指数 {hurst_col}")
                        df[hurst_col] = current_series.rolling(window=hurst_window, min_periods=hurst_window).apply(
                            lambda x: hurst_exponent(x.dropna().values) if len(x.dropna()) >= hurst_window else np.nan, raw=False
                        )
                        print(f"赫斯特指数计算完成, 非空值数量: {df[hurst_col].notna().sum()}, 前5个值: {df[hurst_col].head().tolist()}")
                    else:
                        print(f"数据长度 {len(clean_values)} 小于赫斯特窗口 {hurst_window}, 创建空列")
                        df[hurst_col] = np.nan
                except Exception as e:
                    print(f"赫斯特指数计算失败: {e}")
                    logger.error(f"赫斯特指数计算失败: {e}")
                    df[hurst_col] = np.nan
            else:
                print(f"赫斯特指数列 {hurst_col} 已存在")
            fd_window = params.get('fractal_dimension_window', 89)
            fd_col = f'{prefix}FRACTAL_DIMENSION_{fd_window}d{suffix}'
            print(f"准备计算分形维度, 目标列: {fd_col}, 窗口: {fd_window}")
            if fd_col not in df.columns:
                try:
                    if len(clean_values) >= fd_window:
                        k_max = int(np.sqrt(fd_window))
                        print(f"分形维度计算: k_max={k_max}")
                        df[fd_col] = current_series.rolling(window=fd_window, min_periods=fd_window).apply(
                            lambda x: _numba_higuchi_fd(x, k_max), raw=True
                        )
                        print(f"分形维度计算完成, 非空值数量: {df[fd_col].notna().sum()}, 前5个值: {df[fd_col].head().tolist()}")
                    else:
                        print(f"数据长度 {len(clean_values)} 小于分形维度窗口 {fd_window}, 创建空列")
                        df[fd_col] = np.nan
                except Exception as e:
                    print(f"分形维度计算失败: {e}")
                    logger.error(f"分形维度计算失败: {e}")
                    df[fd_col] = np.nan
            else:
                print(f"分形维度列 {fd_col} 已存在")
            se_window = params.get('sample_entropy_window', 13)
            se_tol_ratio = params.get('sample_entropy_tolerance_ratio', 0.2)
            se_col = f'{prefix}SAMPLE_ENTROPY_{se_window}d{suffix}'
            print(f"准备计算样本熵, 目标列: {se_col}, 窗口: {se_window}, 容差比率: {se_tol_ratio}")
            if se_col not in df.columns:
                try:
                    if len(clean_values) >= se_window + 1:
                        print(f"开始计算样本熵, 使用Numba加速")
                        rolling_std = clean_series.rolling(window=se_window, min_periods=se_window).std().values
                        entropy_values = _numba_rolling_sample_entropy(clean_values, se_window, se_tol_ratio, rolling_std)
                        print(f"样本熵Numba计算完成, 结果长度: {len(entropy_values)}")
                        df[se_col] = pd.Series(entropy_values, index=clean_series.index).reindex(df.index)
                        print(f"样本熵列创建完成, 非空值数量: {df[se_col].notna().sum()}, 前5个值: {df[se_col].head().tolist()}")
                    else:
                        print(f"数据长度 {len(clean_values)} 小于样本熵窗口 {se_window}+1, 创建空列")
                        df[se_col] = np.nan
                except Exception as e:
                    print(f"样本熵计算失败: {e}")
                    logger.error(f"样本熵计算失败: {e}")
                    df[se_col] = np.nan
            else:
                print(f"样本熵列 {se_col} 已存在")
            nolds_sampen_window = params.get('approximate_entropy_window', 21)
            nolds_sampen_tol_ratio = params.get('approximate_entropy_tolerance_ratio', 0.2)
            nolds_sampen_col = f'{prefix}NOLDS_SAMPLE_ENTROPY_{nolds_sampen_window}d{suffix}'
            print(f"准备计算NOLDS样本熵, 目标列: {nolds_sampen_col}, 窗口: {nolds_sampen_window}")
            if nolds_sampen_col not in df.columns:
                try:
                    if len(clean_values) >= nolds_sampen_window:
                        print(f"开始异步计算NOLDS样本熵")
                        df[nolds_sampen_col] = await self.calculator.calculate_nolds_sample_entropy(
                            df=df, period=nolds_sampen_window, column=source_col, tolerance_ratio=nolds_sampen_tol_ratio
                        )
                        print(f"NOLDS样本熵计算完成, 非空值数量: {df[nolds_sampen_col].notna().sum()}")
                    else:
                        print(f"数据长度 {len(clean_values)} 小于NOLDS样本熵窗口 {nolds_sampen_window}, 创建空列")
                        df[nolds_sampen_col] = np.nan
                except Exception as e:
                    print(f"NOLDS样本熵计算失败: {e}")
                    logger.error(f"NOLDS样本熵计算失败: {e}")
                    df[nolds_sampen_col] = np.nan
            else:
                print(f"NOLDS样本熵列 {nolds_sampen_col} 已存在")
            fft_window = params.get('fft_energy_ratio_window', 34)
            fft_col = f'{prefix}FFT_ENERGY_RATIO_{fft_window}d{suffix}'
            print(f"准备计算FFT能量比, 目标列: {fft_col}, 窗口: {fft_window}")
            if fft_col not in df.columns:
                try:
                    if len(clean_values) >= fft_window:
                        print(f"开始计算FFT能量比")
                        fft_energy_ratios_values = _numba_rolling_fft_energy_ratio_core(
                            clean_values,
                            fft_window,
                            low_freq_cutoff_ratio=0.1
                        )
                        print(f"FFT能量比计算完成, 结果长度: {len(fft_energy_ratios_values)}")
                        df[fft_col] = pd.Series(fft_energy_ratios_values, index=clean_series.index).reindex(df.index)
                        print(f"FFT能量比列创建完成, 非空值数量: {df[fft_col].notna().sum()}")
                    else:
                        print(f"数据长度 {len(clean_values)} 小于FFT窗口 {fft_window}, 创建空列")
                        df[fft_col] = np.nan
                except Exception as e:
                    print(f"FFT能量比计算失败: {e}")
                    logger.error(f"FFT能量比计算失败: {e}")
                    df[fft_col] = np.nan
            else:
                print(f"FFT能量比列 {fft_col} 已存在")
            vi_window = params.get('volatility_instability_window', 21)
            vi_col = f'VOLATILITY_INSTABILITY_INDEX_{vi_window}d{suffix}'
            atr_col = f'ATR_14{suffix}'
            print(f"检查波动率不稳定性计算, ATR列: {atr_col}, 目标列: {vi_col}")
            if atr_col in df.columns and vi_col not in df.columns:
                try:
                    if df[atr_col].notna().sum() >= vi_window:
                        print(f"开始计算波动率不稳定性")
                        df[vi_col] = df[atr_col].rolling(window=vi_window, min_periods=vi_window).std()
                        print(f"波动率不稳定性计算完成, 非空值数量: {df[vi_col].notna().sum()}")
                    else:
                        print(f"ATR数据不足 {df[atr_col].notna().sum()} 小于窗口 {vi_window}, 创建空列")
                        df[vi_col] = np.nan
                except Exception as e:
                    print(f"波动率不稳定性计算失败: {e}")
                    df[vi_col] = np.nan
            elif vi_col in df.columns:
                print(f"波动率不稳定性列 {vi_col} 已存在")
            else:
                print(f"ATR列 {atr_col} 不存在, 跳过波动率不稳定性计算")
        print(f"元特征计算完成, 新增列列表: {[col for col in df.columns if any(x in col for x in ['HURST', 'FRACTAL', 'ENTROPY', 'FFT', 'VOLATILITY'])]}")
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V4.5 · 数据格式修复版】基于多维度证据链的量化模式识别系统
        - 修复pct_change_D数据格式：将百分比转换为小数
        - 增加探针输出：在方法开头输出关键原料数据
        - 优化数据格式检测：自动检测并转换异常数据格式
        - 修复霸占信号输出错误
        - 调整ADX条件：从固定阈值25改为动态阈值（ADX分位数）
        - 优化证据权重：增加关键证据的权重
        - 调整阈值逻辑：使用分位数而非绝对阈值
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe].copy()
        print(f"=== 模式识别引擎开始分析，数据长度: {len(df)} ===")
        print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"=== 关键原料数据检查（最新值）===")
        print(f"基础行情:")
        print(f"  收盘价: {df['close_D'].iloc[-1]:.2f}")
        print(f"  涨跌幅原始值: {df['pct_change_D'].iloc[-1]}")
        print(f"  成交量: {df['volume_D'].iloc[-1]:.0f}")
        print(f"  ADX: {df['ADX_14_D'].iloc[-1]:.1f}")
        print(f"  波动率指标:")
        print(f"    BBW: {df['BBW_21_2.0_D'].iloc[-1]:.4f}")
        print(f"    ATR: {df['ATR_14_D'].iloc[-1]:.4f}")
        if 'chip_health_score_D' in df.columns:
            print(f"筹码指标:")
            print(f"  筹码健康度: {df['chip_health_score_D'].iloc[-1]:.1f}")
            print(f"  主峰坚固度: {df['dominant_peak_solidity_D'].iloc[-1]:.4f}")
            print(f"  获利盘稳定性: {df['winner_stability_index_D'].iloc[-1]:.4f}")
        if 'main_force_net_flow_calibrated_D' in df.columns:
            print(f"资金流指标:")
            print(f"  主力净流入: {df['main_force_net_flow_calibrated_D'].iloc[-1]:.0f}")
            print(f"  主力买入执行Alpha: {df['main_force_buy_execution_alpha_D'].iloc[-1]:.4f}")
            print(f"  主力卖出执行Alpha: {df['main_force_sell_execution_alpha_D'].iloc[-1]:.4f}")
        if 'structural_tension_index_D' in df.columns:
            print(f"结构指标:")
            print(f"  结构张力指数: {df['structural_tension_index_D'].iloc[-1]:.4f}")
            print(f"  趋势加速分: {df['trend_acceleration_score_D'].iloc[-1] if 'trend_acceleration_score_D' in df.columns else 'N/A'}")
        if 'platform_conviction_score_D' in df.columns:
            print(f"几何特征:")
            print(f"  平台信念分: {df['platform_conviction_score_D'].iloc[-1] if not pd.isna(df['platform_conviction_score_D'].iloc[-1]) else 'N/A'}")
            print(f"  趋势信念分: {df['trend_conviction_score_D'].iloc[-1] if 'trend_conviction_score_D' in df.columns and not pd.isna(df['trend_conviction_score_D'].iloc[-1]) else 'N/A'}")
            print(f"  突破就绪分: {df['breakout_readiness_score_D'].iloc[-1] if 'breakout_readiness_score_D' in df.columns and not pd.isna(df['breakout_readiness_score_D'].iloc[-1]) else 'N/A'}")
        if 'retail_panic_surrender_index_D' in df.columns:
            print(f"博弈指标:")
            print(f"  散户恐慌指数: {df['retail_panic_surrender_index_D'].iloc[-1]:.4f}")
            print(f"  散户FOMO指数: {df['retail_fomo_premium_index_D'].iloc[-1] if 'retail_fomo_premium_index_D' in df.columns else 'N/A':.4f}")
            print(f"  诱空欺骗强度: {df['deception_lure_short_intensity_D'].iloc[-1] if 'deception_lure_short_intensity_D' in df.columns else 'N/A':.4f}")
        print(f"=== 数据格式检查与修复 ===")
        pct_change_series = df['pct_change_D']
        print(f"涨跌幅原始值统计: 最小值={pct_change_series.min():.2f}, 最大值={pct_change_series.max():.2f}, 均值={pct_change_series.mean():.2f}")
        if pct_change_series.max() > 10 or pct_change_series.min() < -10:
            print(f"检测到涨跌幅数据为百分比格式（如10.00表示10%），正在转换为小数格式...")
            df['pct_change_D'] = df['pct_change_D'] / 100.0
            print(f"转换后最新涨跌幅: {df['pct_change_D'].iloc[-1]:.2%}")
        else:
            print(f"涨跌幅数据已经是小数格式，无需转换")
            print(f"最新涨跌幅: {df['pct_change_D'].iloc[-1]:.2%}")
        print(f"=== 数据修复完成，开始分析 ===")
        required_cols = [
            'open_D', 'high_D', 'low_D', 'close_D', 'volume_D', 'amount_D', 'pct_change_D',
            'VOL_MA_21_D', 'BBW_21_2.0_D', 'ATR_14_D', 'ADX_14_D',
            'chip_health_score_D', 'winner_stability_index_D', 'dominant_peak_solidity_D',
            'cost_structure_skewness_D', 'total_winner_rate_D', 'profit_taking_flow_ratio_D',
            'main_force_net_flow_calibrated_D', 'main_force_buy_execution_alpha_D',
            'main_force_sell_execution_alpha_D', 'main_force_flow_directionality_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'structural_tension_index_D', 'trend_acceleration_score_D',
            'final_charge_intensity_D', 'breakthrough_conviction_score_D',
            'equilibrium_compression_index_D', 'price_volume_entropy_D',
            'platform_conviction_score_D', 'quality_score_D', 'trendline_validity_score_D',
            'trend_conviction_score_D', 'breakout_readiness_score_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"警告：缺少列: {missing_cols[:5]}...，共{len(missing_cols)}个")
            if 'breakout_readiness_score_D' not in df.columns and 'platform_conviction_score_D' in df.columns:
                df['breakout_readiness_score_D'] = self._calculate_breakout_readiness(df)
                print("已计算breakout_readiness_score_D替代值")
            if 'structural_tension_index_D' not in df.columns:
                df['structural_tension_index_D'] = self._calculate_structural_tension(df)
                print("已计算structural_tension_index_D替代值")
        rolling_window = min(120, len(df))
        if rolling_window < 30:
            print("数据长度不足，退出分析")
            return all_dfs
        print(f"使用滚动窗口: {rolling_window}天")
        dynamic_thresholds = {}
        threshold_config = {
            'chip_health_score_D': ('q30', 0.3),
            'dominant_peak_solidity_D': ('q40', 0.4),
            'main_force_net_flow_calibrated_D': ('q60', 0.6),
            'structural_tension_index_D': ('q40', 0.4),
            'breakout_readiness_score_D': ('q60', 0.6),
            'platform_conviction_score_D': ('q60', 0.6),
            'trend_conviction_score_D': ('q70', 0.7),
            'ADX_14_D': ('q70', 0.7),
        }
        for col, (method, param) in threshold_config.items():
            if col in df.columns:
                try:
                    threshold_series = df[col].rolling(rolling_window, min_periods=20).quantile(param)
                    dynamic_thresholds[col] = threshold_series.fillna(method='ffill')
                except Exception as e:
                    print(f"阈值计算失败 {col}: {e}")
                    if col == 'chip_health_score_D':
                        dynamic_thresholds[col] = pd.Series(40, index=df.index)
                    elif col == 'dominant_peak_solidity_D':
                        dynamic_thresholds[col] = pd.Series(0.4, index=df.index)
                    elif col == 'ADX_14_D':
                        dynamic_thresholds[col] = pd.Series(40, index=df.index)
                    else:
                        dynamic_thresholds[col] = pd.Series(0, index=df.index)
        print(f"=== 动态阈值计算完成，共{len(dynamic_thresholds)}个指标 ===")
        for col, thresh_series in list(dynamic_thresholds.items())[:8]:
            current_val = df[col].iloc[-1] if col in df.columns else 'N/A'
            threshold_val = thresh_series.iloc[-1]
            if isinstance(current_val, (int, float)):
                if col == 'pct_change_D':
                    print(f"  {col}: 阈值={threshold_val:.2%}, 当前值={current_val:.2%}, 达标={current_val > threshold_val}")
                else:
                    print(f"  {col}: 阈值={threshold_val:.2f}, 当前值={current_val:.2f}, 达标={current_val > threshold_val}")
            else:
                print(f"  {col}: 阈值={threshold_val:.2f}, 当前值={current_val}")
        chip_stability_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'chip_health_score_D' in df.columns and 'winner_stability_index_D' in df.columns:
            chip_thresh = dynamic_thresholds.get('chip_health_score_D', pd.Series(40, index=df.index))
            winner_std_5 = df['winner_stability_index_D'].rolling(5).std()
            winner_std_20 = df['winner_stability_index_D'].rolling(20).std()
            chip_stability_evidence = ((df['chip_health_score_D'] > chip_thresh) & (winner_std_5 < winner_std_20 * 0.9)).astype(int)
            print(f"筹码稳定性证据: 触发次数={chip_stability_evidence.sum()}, 触发率={chip_stability_evidence.mean():.2%}")
            print(f"  筹码健康度: 当前={df['chip_health_score_D'].iloc[-1]:.1f}, 阈值={chip_thresh.iloc[-1]:.1f}, 达标={df['chip_health_score_D'].iloc[-1] > chip_thresh.iloc[-1]}")
        volatility_compression_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'BBW_21_2.0_D' in df.columns and 'ATR_14_D' in df.columns:
            bbw_quantile = df['BBW_21_2.0_D'].rolling(rolling_window, min_periods=20).quantile(0.4).fillna(method='ffill')
            atr_quantile = df['ATR_14_D'].rolling(rolling_window, min_periods=20).quantile(0.4).fillna(method='ffill')
            volatility_compression_evidence = ((df['BBW_21_2.0_D'] < bbw_quantile) & (df['ATR_14_D'] < atr_quantile)).astype(int)
            print(f"波动率收缩证据: 触发次数={volatility_compression_evidence.sum()}, 触发率={volatility_compression_evidence.mean():.2%}")
            print(f"  BBW: 当前={df['BBW_21_2.0_D'].iloc[-1]:.4f}, 阈值={bbw_quantile.iloc[-1]:.4f}, 达标={df['BBW_21_2.0_D'].iloc[-1] < bbw_quantile.iloc[-1]}")
            print(f"  ATR: 当前={df['ATR_14_D'].iloc[-1]:.4f}, 阈值={atr_quantile.iloc[-1]:.4f}, 达标={df['ATR_14_D'].iloc[-1] < atr_quantile.iloc[-1]}")
        fund_flow_balance_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'main_force_net_flow_calibrated_D' in df.columns and 'main_force_buy_ofi_D' in df.columns and 'main_force_sell_ofi_D' in df.columns:
            mf_std = df['main_force_net_flow_calibrated_D'].rolling(rolling_window, min_periods=20).std().fillna(0)
            mf_buy_ofi_ma = df['main_force_buy_ofi_D'].rolling(3).mean().abs()
            mf_sell_ofi_ma = df['main_force_sell_ofi_D'].rolling(3).mean().abs()
            fund_flow_balance_evidence = ((abs(df['main_force_net_flow_calibrated_D']) < mf_std * 0.8) & (mf_buy_ofi_ma < mf_sell_ofi_ma * 2.0)).astype(int)
            print(f"资金流平衡证据: 触发次数={fund_flow_balance_evidence.sum()}, 触发率={fund_flow_balance_evidence.mean():.2%}")
            print(f"  主力净流绝对值: {abs(df['main_force_net_flow_calibrated_D'].iloc[-1]):.0f}, 阈值={mf_std.iloc[-1]*0.8:.0f}")
        structure_tension_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'structural_tension_index_D' in df.columns:
            st_threshold = dynamic_thresholds.get('structural_tension_index_D', pd.Series(0.4, index=df.index))
            structure_tension_evidence = (df['structural_tension_index_D'] < st_threshold).astype(int)
            print(f"结构张力证据: 触发次数={structure_tension_evidence.sum()}, 触发率={structure_tension_evidence.mean():.2%}")
            print(f"  结构张力: 当前={df['structural_tension_index_D'].iloc[-1]:.4f}, 阈值={st_threshold.iloc[-1]:.4f}")
        consolidation_score = (chip_stability_evidence + volatility_compression_evidence + fund_flow_balance_evidence + structure_tension_evidence)
        print(f"盘整总分分布: 0分={int((consolidation_score == 0).sum())}, 1分={int((consolidation_score == 1).sum())}, 2分={int((consolidation_score == 2).sum())}, 3分={int((consolidation_score == 3).sum())}, 4分={int((consolidation_score == 4).sum())}")
        adx_threshold = dynamic_thresholds.get('ADX_14_D', pd.Series(40, index=df.index))
        df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = (consolidation_score >= 3) & (df['ADX_14_D'] < adx_threshold)
        print(f"盘整信号: 总触发={df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].sum()}, 触发率={df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].mean():.2%}")
        print(f"  ADX条件: 当前ADX={df['ADX_14_D'].iloc[-1]:.1f}, 动态阈值={adx_threshold.iloc[-1]:.1f}, 达标={df['ADX_14_D'].iloc[-1] < adx_threshold.iloc[-1]}")
        hidden_buy_evidence = pd.Series(0, index=df.index, dtype=int)
        if all(col in df.columns for col in ['main_force_buy_execution_alpha_D', 'main_force_net_flow_calibrated_D', 'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D']):
            mf_quantile = df['main_force_net_flow_calibrated_D'].rolling(rolling_window, min_periods=20).quantile(0.5).fillna(method='ffill')
            wash_ratio = df['wash_trade_buy_volume_D'] / (df['wash_trade_sell_volume_D'] + 1e-8)
            hidden_buy_evidence = ((df['main_force_buy_execution_alpha_D'] > -0.1) & (df['main_force_net_flow_calibrated_D'] > mf_quantile) & (wash_ratio > 1.1)).astype(int)
            print(f"主力隐蔽买入证据: 触发次数={hidden_buy_evidence.sum()}, 触发率={hidden_buy_evidence.mean():.2%}")
            print(f"  主力买入执行Alpha: {df['main_force_buy_execution_alpha_D'].iloc[-1]:.4f}")
            print(f"  主力净流分位数阈值: {mf_quantile.iloc[-1]:.0f}")
        retail_panic_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'retail_panic_surrender_index_D' in df.columns:
            rp_quantile = df['retail_panic_surrender_index_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
            if 'total_winner_rate_D' in df.columns:
                retail_panic_evidence = ((df['retail_panic_surrender_index_D'] > rp_quantile) & (df['total_winner_rate_D'] < 0.4)).astype(int)
            else:
                retail_panic_evidence = ((df['retail_panic_surrender_index_D'] > rp_quantile) & (df['close_D'] < df['close_D'].rolling(20).mean() * 0.97)).astype(int)
            print(f"散户恐慌证据: 触发次数={retail_panic_evidence.sum()}, 触发率={retail_panic_evidence.mean():.2%}")
            print(f"  散户恐慌指数: 当前={df['retail_panic_surrender_index_D'].iloc[-1]:.4f}, 阈值={rp_quantile.iloc[-1]:.4f}")
        price_suppression_evidence = pd.Series(0, index=df.index, dtype=int)
        high_20_max = df['high_D'].rolling(20).max()
        price_suppression_evidence = ((df['high_D'] < high_20_max * 0.99) & (df['pct_change_D'].abs() < 0.04) & (df['close_D'] < df['open_D'] * 1.02)).astype(int)
        print(f"价格压制证据: 触发次数={price_suppression_evidence.sum()}, 触发率={price_suppression_evidence.mean():.2%}")
        print(f"  价格压制详情: 最高价={df['high_D'].iloc[-1]:.2f}, 20日最高={high_20_max.iloc[-1]:.2f}, 涨跌幅={df['pct_change_D'].iloc[-1]:.2%}")
        chip_concentration_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'dominant_peak_solidity_D' in df.columns:
            dp_threshold = dynamic_thresholds.get('dominant_peak_solidity_D', pd.Series(0.4, index=df.index))
            if 'cost_structure_skewness_D' in df.columns:
                chip_concentration_evidence = ((df['dominant_peak_solidity_D'] > dp_threshold) & (df['cost_structure_skewness_D'].abs() > 0.3)).astype(int)
            else:
                chip_concentration_evidence = (df['dominant_peak_solidity_D'] > dp_threshold).astype(int)
            print(f"筹码集中证据: 触发次数={chip_concentration_evidence.sum()}, 触发率={chip_concentration_evidence.mean():.2%}")
            print(f"  主峰坚固度: 当前={df['dominant_peak_solidity_D'].iloc[-1]:.4f}, 阈值={dp_threshold.iloc[-1]:.4f}")
        accumulation_score = (hidden_buy_evidence * 2.0 + retail_panic_evidence * 1.5 + price_suppression_evidence + chip_concentration_evidence)
        print(f"吸筹总分分布: 0-1分={int(((accumulation_score < 2).astype(int).sum()))}, 2-2.5分={int(((accumulation_score >= 2) & (accumulation_score < 2.5)).astype(int).sum())}, 2.5-3分={int(((accumulation_score >= 2.5) & (accumulation_score < 3)).astype(int).sum())}, 3分以上={int(((accumulation_score >= 3).astype(int).sum()))}")
        df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & (accumulation_score >= 2.0)
        print(f"吸筹信号: 总触发={df['IS_ACCUMULATION_D'].sum()}, 触发率={df['IS_ACCUMULATION_D'].mean():.2%}")
        momentum_break_evidence = pd.Series(0, index=df.index, dtype=int)
        pct_quantile = df['pct_change_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
        if 'trend_acceleration_score_D' in df.columns:
            momentum_break_evidence = ((df['pct_change_D'] > pct_quantile) & (df['trend_acceleration_score_D'] > -0.1)).astype(int)
        else:
            momentum_break_evidence = ((df['pct_change_D'] > pct_quantile) & (df['close_D'] > df['close_D'].rolling(20).max() * 0.98)).astype(int)
        print(f"动量突破证据: 触发次数={momentum_break_evidence.sum()}, 触发率={momentum_break_evidence.mean():.2%}")
        print(f"  动量突破详情: 涨跌幅={df['pct_change_D'].iloc[-1]:.2%}, 阈值={pct_quantile.iloc[-1]:.2%}")
        volume_break_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'amount_D' in df.columns:
            amount_20_ma = df['amount_D'].rolling(20).mean()
            volume_break_evidence = ((df['volume_D'] > df['VOL_MA_21_D'] * 1.3) & (df['amount_D'] > amount_20_ma * 1.5)).astype(int)
        else:
            volume_break_evidence = (df['volume_D'] > df['VOL_MA_21_D'] * 1.3).astype(int)
        print(f"成交量突破证据: 触发次数={volume_break_evidence.sum()}, 触发率={volume_break_evidence.mean():.2%}")
        print(f"  成交量突破详情: 成交量={df['volume_D'].iloc[-1]:.0f}, 成交量均线={df['VOL_MA_21_D'].iloc[-1]:.0f}, 倍数={df['volume_D'].iloc[-1]/df['VOL_MA_21_D'].iloc[-1]:.2f}")
        fund_flow_break_evidence = pd.Series(0, index=df.index, dtype=int)
        if all(col in df.columns for col in ['main_force_net_flow_calibrated_D', 'main_force_flow_directionality_D', 'main_force_buy_ofi_D', 'main_force_sell_ofi_D']):
            mf_threshold = dynamic_thresholds.get('main_force_net_flow_calibrated_D', pd.Series(0, index=df.index))
            ofi_ratio = df['main_force_buy_ofi_D'] / (df['main_force_sell_ofi_D'].abs() + 1e-8)
            fund_flow_break_evidence = ((df['main_force_net_flow_calibrated_D'] > mf_threshold) & (df['main_force_flow_directionality_D'] > 0.5) & (ofi_ratio > 1.5)).astype(int)
            print(f"资金流突破证据: 触发次数={fund_flow_break_evidence.sum()}, 触发率={fund_flow_break_evidence.mean():.2%}")
            print(f"  资金流突破详情: 主力净流={df['main_force_net_flow_calibrated_D'].iloc[-1]:.0f}, 方向性={df['main_force_flow_directionality_D'].iloc[-1]:.4f}")
        structure_break_evidence = pd.Series(0, index=df.index, dtype=int)
        if all(col in df.columns for col in ['platform_conviction_score_D', 'trend_conviction_score_D']):
            pc_threshold = dynamic_thresholds.get('platform_conviction_score_D', pd.Series(60, index=df.index))
            tc_threshold = dynamic_thresholds.get('trend_conviction_score_D', pd.Series(70, index=df.index))
            if 'breakthrough_conviction_score_D' in df.columns:
                structure_break_evidence = ((df['platform_conviction_score_D'] > pc_threshold) & (df['trend_conviction_score_D'] > tc_threshold) & (df['breakthrough_conviction_score_D'] > 60)).astype(int)
            else:
                structure_break_evidence = ((df['platform_conviction_score_D'] > pc_threshold) & (df['trend_conviction_score_D'] > tc_threshold)).astype(int)
            print(f"结构突破证据: 触发次数={structure_break_evidence.sum()}, 触发率={structure_break_evidence.mean():.2%}")
            print(f"  结构突破详情: 平台信念={df['platform_conviction_score_D'].iloc[-1] if not pd.isna(df['platform_conviction_score_D'].iloc[-1]) else 'N/A'}, 趋势信念={df['trend_conviction_score_D'].iloc[-1] if 'trend_conviction_score_D' in df.columns and not pd.isna(df['trend_conviction_score_D'].iloc[-1]) else 'N/A'}")
        breakout_score = (momentum_break_evidence * 2.0 + volume_break_evidence * 1.5 + fund_flow_break_evidence * 1.2 + structure_break_evidence * 1.5)
        print(f"突破总分分布: 0-2分={int(((breakout_score < 3).astype(int).sum()))}, 3-4分={int(((breakout_score >= 3) & (breakout_score < 5)).astype(int).sum())}, 5-6分={int(((breakout_score >= 5) & (breakout_score < 7)).astype(int).sum())}, 7分以上={int(((breakout_score >= 7).astype(int).sum()))}")
        df['IS_BREAKOUT_D'] = (breakout_score >= 4) & (momentum_break_evidence.astype(bool) | structure_break_evidence.astype(bool) | volume_break_evidence.astype(bool))
        print(f"突破信号: 总触发={df['IS_BREAKOUT_D'].sum()}, 触发率={df['IS_BREAKOUT_D'].mean():.2%}")
        main_force_dist_evidence = pd.Series(0, index=df.index, dtype=int)
        if all(col in df.columns for col in ['main_force_net_flow_calibrated_D', 'main_force_sell_execution_alpha_D', 'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D']):
            mf_low_quantile = df['main_force_net_flow_calibrated_D'].rolling(rolling_window, min_periods=20).quantile(0.4).fillna(method='ffill')
            wash_sell_ratio = df['wash_trade_sell_volume_D'] / (df['wash_trade_buy_volume_D'] + 1e-8)
            main_force_dist_evidence = ((df['main_force_net_flow_calibrated_D'] < mf_low_quantile) & (df['main_force_sell_execution_alpha_D'] > -0.1) & (wash_sell_ratio > 1.3)).astype(int)
            print(f"主力派发证据: 触发次数={main_force_dist_evidence.sum()}, 触发率={main_force_dist_evidence.mean():.2%}")
            print(f"  主力派发详情: 主力净流={df['main_force_net_flow_calibrated_D'].iloc[-1]:.0f}, 阈值={mf_low_quantile.iloc[-1]:.0f}")
        retail_fomo_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'retail_fomo_premium_index_D' in df.columns:
            rf_quantile = df['retail_fomo_premium_index_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
            if 'total_winner_rate_D' in df.columns:
                retail_fomo_evidence = ((df['retail_fomo_premium_index_D'] > rf_quantile) & (df['total_winner_rate_D'] > 0.7)).astype(int)
            else:
                retail_fomo_evidence = ((df['retail_fomo_premium_index_D'] > rf_quantile) & (df['close_D'] > df['close_D'].rolling(20).max() * 1.03)).astype(int)
            print(f"散户追高证据: 触发次数={retail_fomo_evidence.sum()}, 触发率={retail_fomo_evidence.mean():.2%}")
            print(f"  散户追高详情: FOMO指数={df['retail_fomo_premium_index_D'].iloc[-1]:.4f}, 阈值={rf_quantile.iloc[-1]:.4f}")
        price_divergence_evidence = pd.Series(0, index=df.index, dtype=int)
        close_20_max = df['close_D'].rolling(20).max()
        mf_5ma = df['main_force_net_flow_calibrated_D'].rolling(5).mean()
        mf_20ma = df['main_force_net_flow_calibrated_D'].rolling(20).mean()
        price_divergence_evidence = ((df['close_D'] > close_20_max * 0.98) & (mf_5ma < mf_20ma * 0.9)).astype(int)
        print(f"价格背离证据: 触发次数={price_divergence_evidence.sum()}, 触发率={price_divergence_evidence.mean():.2%}")
        print(f"  价格背离详情: 收盘价={df['close_D'].iloc[-1]:.2f}, 20日最高={close_20_max.iloc[-1]:.2f}")
        chip_dispersion_evidence = pd.Series(0, index=df.index, dtype=int)
        if 'chip_health_score_D' in df.columns:
            ch_threshold = dynamic_thresholds.get('chip_health_score_D', pd.Series(40, index=df.index))
            if 'profit_taking_flow_ratio_D' in df.columns:
                chip_dispersion_evidence = ((df['chip_health_score_D'] < ch_threshold) & (df['profit_taking_flow_ratio_D'] > 0.6)).astype(int)
            else:
                chip_dispersion_evidence = ((df['chip_health_score_D'] < ch_threshold) & (df['pct_change_D'] > 0.03)).astype(int)
            print(f"筹码分散证据: 触发次数={chip_dispersion_evidence.sum()}, 触发率={chip_dispersion_evidence.mean():.2%}")
            print(f"  筹码分散详情: 筹码健康度={df['chip_health_score_D'].iloc[-1]:.1f}, 阈值={ch_threshold.iloc[-1]:.1f}")
        distribution_score = (main_force_dist_evidence * 2.0 + retail_fomo_evidence * 1.5 + price_divergence_evidence * 1.2 + chip_dispersion_evidence)
        print(f"派发总分分布: 0-1分={int(((distribution_score < 2).astype(int).sum()))}, 2-2.5分={int(((distribution_score >= 2) & (distribution_score < 3)).astype(int).sum())}, 3分以上={int(((distribution_score >= 3).astype(int).sum()))}")
        df['IS_DISTRIBUTION_D'] = distribution_score >= 2.5
        print(f"派发信号: 总触发={df['IS_DISTRIBUTION_D'].sum()}, 触发率={df['IS_DISTRIBUTION_D'].mean():.2%}")
        df['IS_BEAR_TRAP_WASHOUT_D'] = False
        if len(df) > 60:
            try:
                normal_drop_mask = (df['pct_change_D'] < -0.015) & (df['volume_D'] > df['VOL_MA_21_D'] * 0.8)
                if normal_drop_mask.sum() > 10:
                    normal_drop_chip = df.loc[normal_drop_mask, 'chip_health_score_D'].mean()
                    normal_drop_mf = df.loc[normal_drop_mask, 'main_force_net_flow_calibrated_D'].mean()
                    dl_quantile = df['deception_lure_short_intensity_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
                    bear_trap_conditions = ((df['pct_change_D'] < -0.025) & (df['chip_health_score_D'] > normal_drop_chip * 1.1) & (df['main_force_net_flow_calibrated_D'] > normal_drop_mf * 0.8) & (df['deception_lure_short_intensity_D'] > dl_quantile))
                    df['IS_BEAR_TRAP_WASHOUT_D'] = bear_trap_conditions.fillna(False)
                    print(f"诱空洗盘信号: 总触发={df['IS_BEAR_TRAP_WASHOUT_D'].sum()}, 触发率={df['IS_BEAR_TRAP_WASHOUT_D'].mean():.2%}")
            except Exception as e:
                print(f"计算诱空洗盘模式失败: {e}")
        df['IS_BULL_TRAP_LURE_D'] = False
        if 'IS_BREAKOUT_D' in df.columns and df['IS_BREAKOUT_D'].sum() > 5:
            try:
                fake_breakout_mask = (df['IS_BREAKOUT_D'] & (df['main_force_net_flow_calibrated_D'].rolling(3).mean().shift(-2) < df['main_force_net_flow_calibrated_D'] * 0.8))
                df['IS_BULL_TRAP_LURE_D'] = fake_breakout_mask.fillna(False)
                print(f"假突破诱多信号: 总触发={df['IS_BULL_TRAP_LURE_D'].sum()}, 触发率={df['IS_BULL_TRAP_LURE_D'].mean():.2%}")
            except Exception as e:
                print(f"计算假突破诱多模式失败: {e}")
        df['IS_BAZHAN_D'] = False
        if 'amount_D' in df.columns and len(df) > 50:
            try:
                amount_ma5 = df['amount_D'].rolling(5).mean()
                amount_ma20 = df['amount_D'].rolling(20).mean()
                ff_ratio = amount_ma5 / (amount_ma20.shift(1) + 1e-8)
                position_window = min(120, len(df))
                llv_c = df['close_D'].rolling(position_window).min()
                hhv_c = df['close_D'].rolling(position_window).max()
                position = (df['close_D'] - llv_c) / (hhv_c - llv_c + 1e-8) * 100
                cond_amount_break = ff_ratio >= 1.8
                cond_price_low = position < 45
                cond_price_mid = position < 75
                cond_consolidation = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] | df['IS_ACCUMULATION_D']
                bazhan_v1 = cond_amount_break & cond_price_low & cond_consolidation
                bazhan_v2 = cond_amount_break & cond_price_mid & (df['volume_D'] > df['VOL_MA_21_D'] * 1.8)
                df['IS_BAZHAN_D'] = (bazhan_v1 | bazhan_v2).fillna(False)
                print(f"霸占信号: 总触发={df['IS_BAZHAN_D'].sum()}, 触发率={df['IS_BAZHAN_D'].mean():.2%}")
            except Exception as e:
                print(f"计算霸占模式失败: {e}")
        df['IS_WW1_D'] = False
        if 'open_D' in df.columns:
            try:
                cond_open_gap = df['open_D'] < df['low_D'].shift(1) * 0.99
                cond_close_recovery = df['close_D'] > (df['open_D'] + df['low_D'].shift(1)) / 2.1
                cond_volume_confirmation = df['volume_D'] > df['VOL_MA_21_D'] * 1.3
                cond_chip_support = df['chip_health_score_D'] > 35 if 'chip_health_score_D' in df.columns else True
                df['IS_WW1_D'] = (cond_open_gap & cond_close_recovery & cond_volume_confirmation & cond_chip_support).fillna(False)
                print(f"WW1信号: 总触发={df['IS_WW1_D'].sum()}, 触发率={df['IS_WW1_D'].mean():.2%}")
            except Exception as e:
                print(f"计算WW1模式失败: {e}")
        pattern_cols = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D', 'IS_BAZHAN_D', 'IS_WW1_D', 'IS_BEAR_TRAP_WASHOUT_D', 'IS_BULL_TRAP_LURE_D']
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        for signal_col in ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']:
            if signal_col in df.columns:
                try:
                    signal_series = df[signal_col].astype(float)
                    df[signal_col] = (signal_series.rolling(3, min_periods=2).sum() >= 2).fillna(False).astype(bool)
                except Exception as e:
                    print(f"信号去抖动失败: {signal_col}, 错误: {e}")
        try:
            if 'IS_BREAKOUT_D' in df.columns and 'IS_ACCUMULATION_D' in df.columns:
                breakout_mask = df['IS_BREAKOUT_D'].astype(bool)
                if breakout_mask.any():
                    df.loc[breakout_mask, 'IS_ACCUMULATION_D'] = False
                    print(f"互斥处理: 突破信号时关闭吸筹信号，影响{breakout_mask.sum()}个数据点")
            if 'IS_DISTRIBUTION_D' in df.columns and 'IS_BREAKOUT_D' in df.columns:
                distribution_mask = df['IS_DISTRIBUTION_D'].astype(bool)
                if distribution_mask.any():
                    df.loc[distribution_mask, 'IS_BREAKOUT_D'] = False
                    print(f"互斥处理: 派发信号时关闭突破信号，影响{distribution_mask.sum()}个数据点")
        except Exception as e:
            print(f"模式互斥性处理失败: {e}")
        try:
            df['_PROBE_CONSOLIDATION_SCORE_D'] = consolidation_score
            df['_PROBE_ACCUMULATION_SCORE_D'] = accumulation_score
            df['_PROBE_BREAKOUT_SCORE_D'] = breakout_score
            df['_PROBE_DISTRIBUTION_SCORE_D'] = distribution_score
            print(f"探针数据存储完成: 盘整分均值={consolidation_score.mean():.2f}, 吸筹分均值={accumulation_score.mean():.2f}, 突破分均值={breakout_score.mean():.2f}, 派发分均值={distribution_score.mean():.2f}")
            print(f"信号触发统计: 盘整={df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].sum()}, 吸筹={df['IS_ACCUMULATION_D'].sum()}, 突破={df['IS_BREAKOUT_D'].sum()}, 派发={df['IS_DISTRIBUTION_D'].sum()}")
        except Exception as e:
            print(f"探针数据存储失败: {e}")
        all_dfs[timeframe] = df
        print("=== 高级模式识别引擎(V4.5 数据格式修复版)分析完成 ===")
        return all_dfs

    def _calculate_breakout_readiness(self, df: pd.DataFrame) -> pd.Series:
        """计算突破就绪分数"""
        scores = pd.Series(0, index=df.index, dtype=float)
        for i in range(len(df)):
            if i < 20:
                scores.iloc[i] = 0
                continue
            
            try:
                # 1. 平台质量分（基于平台振幅和持续时间）
                recent_high = df['high_D'].iloc[i-20:i].max()
                recent_low = df['low_D'].iloc[i-20:i].min()
                platform_range = (recent_high - recent_low) / (recent_low + 1e-8)
                
                # 2. 成交量收缩程度
                if i >= 20:
                    volume_ratio = df['volume_D'].iloc[i-5:i].mean() / max(df['volume_D'].iloc[i-20:i-5].mean(), 1)
                else:
                    volume_ratio = 1
                
                # 3. 波动率收缩
                if 'ATR_14_D' in df.columns and i >= 20:
                    atr_max = df['ATR_14_D'].iloc[i-20:i].max()
                    atr_ratio = df['ATR_14_D'].iloc[i] / max(atr_max, 1e-8)
                else:
                    atr_ratio = 1
                
                # 综合评分
                score = 100 * (1 - min(platform_range, 0.99)) * (1 - min(volume_ratio, 0.99)) * (1 - min(atr_ratio, 0.99))
                scores.iloc[i] = min(score, 100)
            except:
                scores.iloc[i] = 0
        
        return scores

    def _calculate_structural_tension(self, df: pd.DataFrame) -> pd.Series:
        """计算结构张力指数"""
        tension_scores = pd.Series(0, index=df.index, dtype=float)
        for i in range(len(df)):
            if i < 30:
                tension_scores.iloc[i] = 0
                continue
            
            try:
                tensions = []
                
                # 1. 价格与均线张力
                if 'MA_20_D' in df.columns and 'ATR_14_D' in df.columns and df['ATR_14_D'].iloc[i] > 0:
                    ma_distance = abs(df['close_D'].iloc[i] - df['MA_20_D'].iloc[i]) / df['ATR_14_D'].iloc[i]
                    tensions.append(ma_distance)
                
                # 2. 成交量与价格背离
                if i >= 5:
                    price_change = df['close_D'].iloc[i] / max(df['close_D'].iloc[i-5], 1e-8) - 1
                    volume_mean = df['volume_D'].iloc[i-5:i].mean()
                    volume_change = df['volume_D'].iloc[i] / max(volume_mean, 1) - 1
                    volume_tension = abs(price_change - volume_change)
                    tensions.append(volume_tension)
                
                # 3. 资金流分歧
                if 'main_force_buy_ofi_D' in df.columns and 'main_force_sell_ofi_D' in df.columns:
                    flow_divergence = abs(df['main_force_buy_ofi_D'].iloc[i] - df['main_force_sell_ofi_D'].iloc[i])
                    tensions.append(flow_divergence)
                
                # 综合张力指数
                if tensions:
                    avg_tension = np.mean(tensions)
                    if i > 60:
                        historical_max = max(np.max(tensions) if len(tensions) > 0 else 0, 0.01)
                        tension_score = avg_tension / historical_max
                    else:
                        tension_score = avg_tension
                else:
                    tension_score = 0
                
                tension_scores.iloc[i] = min(tension_score, 1.0)
            except:
                tension_scores.iloc[i] = 0
        
        return tension_scores

    async def calculate_aaa_indicator(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V1.0】计算通达信 AAA 指标。
        - 核心逻辑: AAA:=ABS((2*CLOSE+HIGH+LOW)/4-MA(CLOSE,N))/MA(CLOSE,N); N:=30;
        - 作为 DMA 的平滑因子。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 AAA 指标失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = ['close_D', 'high_D', 'low_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"计算 AAA 指标缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        n_period_for_aaa = 30
        if len(df) < n_period_for_aaa:
            logger.warning(f"计算 AAA 指标失败：数据行数 ({len(df)}) 不足以计算周期为 {n_period_for_aaa} 的 MA。")
            df['AAA_D'] = 0.0
            all_dfs[timeframe] = df
            return all_dfs
        ma_close_n = df['close_D'].rolling(window=n_period_for_aaa, min_periods=1).mean()
        weighted_price = (2 * df['close_D'] + df['high_D'] + df['low_D']) / 4
        aaa_series = (weighted_price - ma_close_n).abs() / ma_close_n.replace(0, np.nan)
        df['AAA_D'] = aaa_series.fillna(0)
        all_dfs[timeframe] = df
        logger.info("AAA 指标计算完成。")
        return all_dfs

    async def calculate_consolidation_period(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V2.2 · 细粒度意图与斐波那契周期升级版】根据多因子共振识别盘整期。
        - 核心升级: 判定逻辑从单一的“几何形态”升级为“几何形态 或 主力意图”的双重标准。
                      即使K线形态不完美，只要筹码高度锁定且主力展现出强控盘意图，也将其识别为盘整构筑期。
                      增强主力意图判断，引入细粒度买卖方意图和欺骗指数。
                      周期参数调整为斐波那契数列。
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            return all_dfs
        df = all_dfs[timeframe]
        # 修改代码行: 调整默认周期为斐波那契数
        boll_period = params.get('boll_period', 21) # 21是斐波那契数，保持
        # 新增代码行: 从params中获取 boll_std，如果不存在则默认为 2.0
        boll_std = params.get('boll_std', 2.0)
        roc_period = params.get('roc_period', 13)   # 从12改为13
        vol_ma_period = params.get('vol_ma_period', 55) # 55是斐波那契数，保持
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_{timeframe}"
        roc_col = f"ROC_{roc_period}_{timeframe}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_{timeframe}"
        required_cols = [
            bbw_col, roc_col, vol_ma_col, f'high_{timeframe}', f'low_{timeframe}', f'volume_{timeframe}',
            'dominant_peak_solidity_D', 'control_solidity_index_D',
            'mf_cost_zone_buy_intent_D', 'mf_cost_zone_sell_intent_D',
            'deception_lure_long_intensity_D'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.warning(f"盘整期计算跳过，依赖的列 '{', '.join(missing)}' 不存在。")
            return all_dfs
        bbw_quantile = params.get('bbw_quantile', 0.25)
        roc_threshold = params.get('roc_threshold', 5.0)
        min_expanding_periods = boll_period * 2
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile).bfill()
        df[f'dynamic_bbw_threshold_{timeframe}'] = dynamic_bbw_threshold
        cond_volatility = df[bbw_col] < df[f'dynamic_bbw_threshold_{timeframe}']
        cond_trend = df[roc_col].abs() < roc_threshold
        cond_volume = df[f'volume_{timeframe}'] < df[vol_ma_col]
        is_classic_consolidation = cond_volatility & cond_trend & cond_volume
        cond_chips_locked = df['dominant_peak_solidity_D'] > 0.7
        cond_main_force_control = df['control_solidity_index_D'] > 0.6
        cond_mf_buy_intent_strong = df['mf_cost_zone_buy_intent_D'] > 0.5
        cond_mf_sell_intent_weak = df['mf_cost_zone_sell_intent_D'] < 0.3
        cond_no_deception_lure_long = df['deception_lure_long_intensity_D'] < 0.2
        is_intent_based_consolidation = cond_chips_locked & cond_main_force_control & \
                                         cond_mf_buy_intent_strong & cond_mf_sell_intent_weak & \
                                         cond_no_deception_lure_long
        is_consolidating = is_classic_consolidation | is_intent_based_consolidation
        df[f'is_consolidating_{timeframe}'] = is_consolidating
        if is_consolidating.any():
            consolidation_blocks = (is_consolidating != is_consolidating.shift()).cumsum()
            consolidating_df = df[is_consolidating].copy()
            grouped = consolidating_df.groupby(consolidation_blocks[is_consolidating])
            df[f'dynamic_consolidation_high_{timeframe}'] = grouped[f'high_{timeframe}'].transform('max')
            df[f'dynamic_consolidation_low_{timeframe}'] = grouped[f'low_{timeframe}'].transform('min')
            df[f'dynamic_consolidation_avg_vol_{timeframe}'] = grouped[f'volume_{timeframe}'].transform('mean')
            df[f'dynamic_consolidation_duration_{timeframe}'] = grouped[f'high_{timeframe}'].transform('size')
            fill_cols = [f'dynamic_consolidation_high_{timeframe}', f'dynamic_consolidation_low_{timeframe}', f'dynamic_consolidation_avg_vol_{timeframe}', f'dynamic_consolidation_duration_{timeframe}']
            df[fill_cols] = df[fill_cols].ffill()
        df[f'dynamic_consolidation_high_{timeframe}'] = df.get(f'dynamic_consolidation_high_{timeframe}', pd.Series(index=df.index)).fillna(df[f'high_{timeframe}'])
        df[f'dynamic_consolidation_low_{timeframe}'] = df.get(f'dynamic_consolidation_low_{timeframe}', pd.Series(index=df.index)).fillna(df[f'low_{timeframe}'])
        df[f'dynamic_consolidation_avg_vol_{timeframe}'] = df.get(f'dynamic_consolidation_avg_vol_{timeframe}', pd.Series(index=df.index)).fillna(0)
        df[f'dynamic_consolidation_duration_{timeframe}'] = df.get(f'dynamic_consolidation_duration_{timeframe}', pd.Series(index=df.index)).fillna(0)
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
        【V3.4 · 依赖注入修复版】突破质量分计算专用通道
        - 核心修复: 移除 calculator 参数，改用 self.calculator。
        - 核心修复: 增加幂等性检查。在计算前判断 `breakout_quality_score_D` 列是否已存在，
                  如果存在则跳过计算，避免重复处理和潜在的数据覆盖。
        - 核心修复: 协同 IndicatorCalculator V2.5 修复了接口契约。本方法现在能正确接收不带后缀的
                  'breakout_quality_score'，并执行重命名与填充，确保数据流的绝对标准化和健壮性。
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
        # 修改代码行: 补充 calculate_breakout_quality_score 所需的所有列
        required_materials = [
            'volume', 'VOL_MA_21', 'main_force_flow_directionality',
            'open', 'high', 'low', 'close',
            'total_winner_rate', 'dominant_peak_solidity', 'VPA_EFFICIENCY',
            'main_force_buy_execution_alpha', 'upward_impulse_strength',
            'buy_order_book_clearing_rate', 'bid_side_liquidity',
            'vwap_cross_up_intensity', 'opening_buy_strength',
            'floating_chip_cleansing_efficiency',
            'VPA_BUY_EFFICIENCY',
            'deception_lure_long_intensity', 'wash_trade_buy_volume'
        ]
        df_standardized = pd.DataFrame(index=df_daily.index)
        missing_materials = []
        for material in required_materials:
            source_col_with_suffix = f"{material}_{timeframe}"
            if source_col_with_suffix in df_daily.columns:
                df_standardized[material] = df_daily[source_col_with_suffix]
            else:
                missing_materials.append(source_col_with_suffix)
        if missing_materials:
            logger.warning(f"突破质量分计算中止，缺少标准化原材料: {missing_materials}")
            return all_dfs
        result_df = await self.calculator.calculate_breakout_quality_score(df_daily=df_standardized, params=params)
        if result_df is not None and not result_df.empty:
            df_daily = df_daily.join(result_df, how='left')
            # --- 命名协议强制执行 ---
            # 将合并进来的、不带后缀的列，强制重命名为带 '_D' 后缀的标准格式
            if 'breakout_quality_score' in df_daily.columns:
                df_daily.rename(columns={'breakout_quality_score': 'breakout_quality_score_D'}, inplace=True)
                df_daily['breakout_quality_score_D'] = df_daily['breakout_quality_score_D'].ffill()
            all_dfs[timeframe] = df_daily
            logger.info("突破质量分计算完成并已集成。")
        else:
            logger.warning("突破质量分计算器返回了None或空DataFrame，未集成。")
        return all_dfs

    async def calculate_ma_potential_metrics(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V1.1 · Numba加速版】均线系统势能分析引擎
        - 核心优化: 使用 Numba 优化的 _numba_spearman_orderliness 替代 df.apply(spearman_corr, axis=1)。
        - 性能提升: 避免了按行遍历 DataFrame 造成的巨大开销，将相关性计算并行化/向量化。
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
                tension_index_series = raw_tension_index.rolling(window=norm_window).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-9) if len(x) > 1 else 0, raw=False).fillna(0).clip(-3, 3) / 3
                df[f'MA_POTENTIAL_TENSION_INDEX_{timeframe}'] = tension_index_series.astype(np.float32)
                # 3. 计算势能方向 (有序性) - 【Numba 优化】
                ma_df = df[ma_cols]
                # 准备 Numba 需要的数据
                ma_values = ma_df.values.astype(np.float64)
                # 计算理论排名的 Rank 值 (1-based)
                # 例如 ma_periods=[5, 10, 20]，则 rank 为 [1, 2, 3]
                periods_series = pd.Series(ma_periods)
                # 使用 ordinal rank 配合 Numba 逻辑 (从小到大)
                # 修复: pandas 2.x 不支持 method='ordinal'，改为 'first'
                ma_ranks_x = periods_series.rank(method='first').values.astype(np.float64)
                # 调用 Numba 函数
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

    async def calculate_nmfnf(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V1.1 · 细粒度NMFNF升级版】计算标准化主力净流量 (Normalized Main Force Net Flow, NMFNF)。
        - 核心逻辑: NMFNF = main_force_net_flow_calibrated_D / total_market_value_D。
        - 新增逻辑: NMFNF_BUY_D = main_force_buy_ofi_D / total_market_value_D。
        - 新增逻辑: NMFNF_SELL_D = main_force_sell_ofi_D / total_market_value_D。
        - 目的: 将主力净流量标准化，使其在不同市值和活跃度的股票间可比，并细化买卖方贡献。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 NMFNF 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        required_cols = ['main_force_net_flow_calibrated_D', 'total_market_value_D', 'main_force_buy_ofi_D', 'main_force_sell_ofi_D']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"计算 NMFNF 缺少关键数据: {missing_cols}，模块已跳过！")
            return all_dfs
        # 避免除以零，将总市值中的零替换为 NaN，然后整个表达式会变为 NaN，最后fillna(0)
        total_market_value_safe = df['total_market_value_D'].replace(0, np.nan)
        # 计算总NMFNF
        nmfnf_series = df['main_force_net_flow_calibrated_D'] / total_market_value_safe
        df['NMFNF_D'] = nmfnf_series.fillna(0)
        # 计算买方NMFNF (NMFNF_BUY_D)
        nmfnf_buy_series = df['main_force_buy_ofi_D'] / total_market_value_safe
        df['NMFNF_BUY_D'] = nmfnf_buy_series.fillna(0)
        # 计算卖方NMFNF (NMFNF_SELL_D)
        nmfnf_sell_series = df['main_force_sell_ofi_D'] / total_market_value_safe
        df['NMFNF_SELL_D'] = nmfnf_sell_series.fillna(0)
        all_dfs[timeframe] = df
        logger.info("NMFNF 指标计算完成。")
        return all_dfs

    async def calculate_och(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V3.4 · OCH非线性融合Numba优化版】计算整体筹码健康度 (Overall Chip Health, OCH)。
        - 核心升级: 维度一“筹码集中度与结构优化”的计算逻辑重构，引入 cost_gini_coefficient 和 primary_peak_kurtosis 两个核心指标，
                    替代旧的 winner/loser_concentration 指标，实现对筹码结构更精准、更深刻的量化评估。
                    全面替换旧的聚合指标，引入新的买卖双方细粒度指标，并重新设计部分融合逻辑，以更精确地反映市场博弈的真实情况。
        - 【新增】引入非线性融合（tanh函数）和情境自适应（基于波动率和市场情绪）来增强OCH的鲁棒性和敏感度。
        - 【新增】`_nonlinear_fusion` 逻辑已通过 Numba 优化。
        - 目的: 在数据层提前计算一个综合的筹码健康度指标，供后续斜率计算和情报层使用。
        - 数据源: 直接从 df 中获取原始筹码相关列。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 OCH 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        df_index = df.index
        # 定义所有需要的原始筹码相关列
        # 更新必需列的列表，以匹配V33.0版本的新指标体系和细粒度数据
        required_cols = [
            # 维度一: 筹码集中度与结构优化
            'cost_gini_coefficient_D',
            'primary_peak_kurtosis_D',
            'dominant_peak_solidity_D', 'dominant_peak_volume_ratio_D',
            'chip_fault_blockage_ratio_D',
            # 维度二: 成本与盈亏结构动态
            'total_winner_rate_D', 'total_loser_rate_D',
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D',
            'cost_structure_skewness_D',
            'main_force_cost_advantage_D',
            'profit_taking_flow_ratio_D',
            'loser_pain_index_D',
            'rally_sell_distribution_intensity_D',
            'dip_buy_absorption_strength_D',
            'panic_buy_absorption_contribution_D',
            # 维度三: 持股心态与交易行为
            'winner_stability_index_D',
            'chip_fatigue_index_D',
            'capitulation_flow_ratio_D',
            'capitulation_absorption_index_D',
            'active_buying_support_D', 'active_selling_pressure_D',
            'mf_retail_battle_intensity_D',
            'dip_buy_absorption_strength_D',
            'dip_sell_pressure_resistance_D',
            'rally_sell_distribution_intensity_D',
            'rally_buy_support_weakness_D',
            'panic_sell_volume_contribution_D',
            'panic_buy_absorption_contribution_D',
            'opening_buy_strength_D',
            'opening_sell_strength_D',
            'pre_closing_buy_posture_D',
            'pre_closing_sell_posture_D',
            'closing_auction_buy_ambush_D',
            'closing_auction_sell_ambush_D',
            'main_force_buy_ofi_D',
            'main_force_sell_ofi_D',
            'retail_buy_ofi_D',
            'retail_sell_ofi_D',
            'wash_trade_buy_volume_D',
            'wash_trade_sell_volume_D',
            'buy_order_book_clearing_rate_D',
            'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D',
            'vwap_sell_control_strength_D',
            'bid_side_liquidity_D',
            'ask_side_liquidity_D',
            # 维度四: 主力控盘与意图
            'control_solidity_index_D',
            'main_force_on_peak_buy_flow_D',
            'main_force_on_peak_sell_flow_D',
            'main_force_flow_directionality_D',
            'main_force_buy_execution_alpha_D',
            'main_force_sell_execution_alpha_D',
            'main_force_conviction_index_D', 'mf_vpoc_premium_D',
            'vwap_buy_control_strength_D',
            'vwap_sell_control_strength_D',
            'main_force_vwap_up_guidance_D',
            'main_force_vwap_down_guidance_D',
            'vwap_cross_up_intensity_D',
            'vwap_cross_down_intensity_D',
            'main_force_t0_buy_efficiency_D',
            'main_force_t0_sell_efficiency_D',
            'buy_flow_efficiency_index_D',
            'sell_flow_efficiency_index_D',
            'covert_distribution_signal_D',
            'supportive_distribution_intensity_D',
            'turnover_rate_f_D',
            # 【新增代码行】情境自适应相关指标
            'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'market_sentiment_score_D',
            'price_volume_entropy_D'
        ]
        # 使用 _get_safe_series 确保所有数据都存在，并用默认值填充缺失值
        def _get_safe_series_local(col_name, default_val=0.0):
            if col_name not in df.columns:
                print(f"调试信息: OCH计算缺少列: {col_name}，使用默认值 {default_val}。")
                return pd.Series(default_val, index=df_index, dtype=np.float32) # 确保返回 float32
            return df[col_name].fillna(default_val).astype(np.float32) # 确保返回 float32
        # --- 情境自适应调制器 ---
        # 波动率情境：高波动率时，筹码健康度可能更不稳定，需要更谨慎的评估
        volatility_context = _get_safe_series_local('VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0).rolling(21).mean().fillna(0).astype(np.float32)
        # 市场情绪情境：极端情绪下，筹码健康度可能被扭曲
        sentiment_context = _get_safe_series_local('market_sentiment_score_D', 0.0).rolling(21).mean().fillna(0).astype(np.float32)
        # 市场信息复杂度情境：高复杂度时，信号可能更模糊
        entropy_context = _get_safe_series_local('price_volume_entropy_D', 0.0).rolling(21).mean().fillna(0).astype(np.float32)
        # 融合函数：使用 tanh 激活函数进行非线性融合，并考虑情境调制
        def _nonlinear_fusion(scores_dict: Dict[str, pd.Series], weights_dict: Dict[str, float], volatility_mod: pd.Series, sentiment_mod: pd.Series, entropy_mod: pd.Series) -> pd.Series:
            # 准备 Numba 函数所需的 NumPy 数组
            score_arrays = []
            weight_arrays = []
            for score_name, weight in weights_dict.items():
                score_series = scores_dict.get(score_name, pd.Series(0.0, index=df_index, dtype=np.float32))
                score_arrays.append(score_series.values)
                weight_arrays.append(np.full_like(score_series.values, weight, dtype=np.float32))
            # 调用 Numba 优化函数
            fused_score_values = _numba_nonlinear_fusion_core(
                score_arrays,
                weight_arrays,
                volatility_mod.values,
                sentiment_mod.values,
                entropy_mod.values
            )
            return pd.Series(fused_score_values, index=df_index, dtype=np.float32)
        # --- 1. 筹码集中度与结构优化 (Concentration & Structure Optimization Score) ---
        cost_gini = _get_safe_series_local('cost_gini_coefficient_D', 0.5)
        peak_kurtosis = _get_safe_series_local('primary_peak_kurtosis_D', 3.0)
        peak_solidity = _get_safe_series_local('dominant_peak_solidity_D', 0.5)
        peak_volume_ratio = _get_safe_series_local('dominant_peak_volume_ratio_D', 0.5)
        chip_fault = _get_safe_series_local('chip_fault_blockage_ratio_D', 0.0)
        concentration_health = (1 - cost_gini).clip(0, 1)
        # 修复点1: 显式转换为 np.float32
        normalized_kurtosis = peak_kurtosis.rolling(window=120, min_periods=20).rank(pct=True).fillna(0.5).astype(np.float32)
        peak_quality = (peak_solidity * peak_volume_ratio * normalized_kurtosis).clip(0, 1)
        blockage_penalty = (1 - chip_fault)
        concentration_scores = {
            'concentration_health': concentration_health,
            'peak_quality': peak_quality,
            'blockage_penalty': blockage_penalty
        }
        concentration_weights = {'concentration_health': 0.5, 'peak_quality': 0.4, 'blockage_penalty': 0.1}
        concentration_score = _nonlinear_fusion(concentration_scores, concentration_weights, volatility_context, sentiment_context, entropy_context)
        # --- 2. 成本与盈亏结构动态 (Cost & P/L Structure Dynamics Score) ---
        total_winner_rate = _get_safe_series_local('total_winner_rate_D', 0.5)
        total_loser_rate = _get_safe_series_local('total_loser_rate_D', 0.5)
        winner_profit_margin = _get_safe_series_local('winner_profit_margin_avg_D', 0.0) / 100
        loser_loss_margin = _get_safe_series_local('loser_loss_margin_avg_D', 0.0) / 100
        cost_divergence = _get_safe_series_local('cost_structure_skewness_D', 0.0)
        mf_cost_advantage = _get_safe_series_local('main_force_cost_advantage_D', 0.0)
        imminent_profit_taking = _get_safe_series_local('profit_taking_flow_ratio_D', 0.0)
        loser_capitulation_pressure = _get_safe_series_local('loser_pain_index_D', 0.0)
        profit_pressure = total_winner_rate * winner_profit_margin * (_get_safe_series_local('rally_sell_distribution_intensity_D', 0.0) + imminent_profit_taking).clip(0,1)
        loser_support = total_loser_rate * loser_loss_margin * (1 - loser_capitulation_pressure) + \
                        _get_safe_series_local('dip_buy_absorption_strength_D', 0.0) * 0.5 + \
                        _get_safe_series_local('panic_buy_absorption_contribution_D', 0.0) * 0.5
        cost_advantage_score = (mf_cost_advantage - cost_divergence).clip(-1, 1)
        cost_structure_scores = {
            'loser_support': loser_support,
            'cost_advantage_score': cost_advantage_score,
            'profit_pressure': profit_pressure
        }
        cost_structure_weights = {'loser_support': 0.4, 'cost_advantage_score': 0.4, 'profit_pressure': -0.2} # 利润压力是负向权重
        cost_structure_score = _nonlinear_fusion(cost_structure_scores, cost_structure_weights, volatility_context, sentiment_context, entropy_context)
        # --- 3. 持股心态与交易行为 (Holder Sentiment & Behavior Score) ---
        winner_conviction = _get_safe_series_local('winner_stability_index_D', 0.0)
        chip_fatigue = _get_safe_series_local('chip_fatigue_index_D', 0.0)
        locked_profit = _get_safe_series_local('winner_stability_index_D', 0.0)
        locked_loss = 1.0 - _get_safe_series_local('capitulation_flow_ratio_D', 0.0)
        buy_side_absorption_composite = (
            _get_safe_series_local('capitulation_absorption_index_D', 0.0) * 0.2 +
            _get_safe_series_local('active_buying_support_D', 0.0) * 0.1 +
            _get_safe_series_local('dip_buy_absorption_strength_D', 0.0) * 0.1 +
            _get_safe_series_local('panic_buy_absorption_contribution_D', 0.0) * 0.1 +
            _get_safe_series_local('opening_buy_strength_D', 0.0) * 0.05 +
            _get_safe_series_local('pre_closing_buy_posture_D', 0.0) * 0.05 +
            _get_safe_series_local('closing_auction_buy_ambush_D', 0.0) * 0.05 +
            _get_safe_series_local('main_force_buy_ofi_D', 0.0) * 0.1 +
            _get_safe_series_local('retail_buy_ofi_D', 0.0) * 0.05 +
            _get_safe_series_local('bid_side_liquidity_D', 0.0) * 0.05 +
            _get_safe_series_local('buy_order_book_clearing_rate_D', 0.0) * 0.05 +
            _get_safe_series_local('vwap_buy_control_strength_D', 0.0) * 0.05
        ).clip(0, 1)
        sell_side_pressure_composite = (
            _get_safe_series_local('active_selling_pressure_D', 0.0) * 0.1 +
            _get_safe_series_local('rally_sell_distribution_intensity_D', 0.0) * 0.1 +
            _get_safe_series_local('dip_sell_pressure_resistance_D', 0.0) * 0.05 +
            _get_safe_series_local('panic_sell_volume_contribution_D', 0.0) * 0.1 +
            _get_safe_series_local('opening_sell_strength_D', 0.0) * 0.05 +
            _get_safe_series_local('pre_closing_sell_posture_D', 0.0) * 0.05 +
            _get_safe_series_local('closing_auction_sell_ambush_D', 0.0) * 0.05 +
            _get_safe_series_local('main_force_sell_ofi_D', 0.0) * 0.1 +
            _get_safe_series_local('retail_sell_ofi_D', 0.0) * 0.05 +
            _get_safe_series_local('ask_side_liquidity_D', 0.0) * 0.05 +
            _get_safe_series_local('sell_order_book_clearing_rate_D', 0.0) * 0.05 +
            _get_safe_series_local('vwap_sell_control_strength_D', 0.0) * 0.05
        ).clip(0, 1)
        combat_intensity = _get_safe_series_local('mf_retail_battle_intensity_D', 0.0)
        conviction_lock_score = (winner_conviction + locked_profit - chip_fatigue - locked_loss).clip(-1, 1)
        absorption_support_score = (buy_side_absorption_composite - sell_side_pressure_composite).clip(-1, 1)
        wash_trade_penalty = (_get_safe_series_local('wash_trade_buy_volume_D', 0.0) + _get_safe_series_local('wash_trade_sell_volume_D', 0.0)).clip(0, 1) * 0.1
        sentiment_scores = {
            'conviction_lock_score': (conviction_lock_score + 1) / 2, # 归一化到 [0, 1]
            'absorption_support_score': (absorption_support_score + 1) / 2, # 归一化到 [0, 1]
            'combat_intensity': combat_intensity,
            'wash_trade_penalty': wash_trade_penalty
        }
        sentiment_weights = {'conviction_lock_score': 0.4, 'absorption_support_score': 0.4, 'combat_intensity': 0.2, 'wash_trade_penalty': -0.1}
        sentiment_score = _nonlinear_fusion(sentiment_scores, sentiment_weights, volatility_context, sentiment_context, entropy_context)
        # --- 4. 主力控盘与意图 (Main Force Control & Intent Score) ---
        mf_control_leverage = _get_safe_series_local('control_solidity_index_D', 0.0)
        mf_on_peak_flow_composite = (_get_safe_series_local('main_force_on_peak_buy_flow_D', 0.0) - _get_safe_series_local('main_force_on_peak_sell_flow_D', 0.0))
        # 修复点2: 显式转换为 np.float32
        mf_on_peak_flow_normalized = (mf_on_peak_flow_composite.rank(pct=True) * 2 - 1).clip(0, 1).astype(np.float32)
        mf_intent_composite = (
            _get_safe_series_local('main_force_flow_directionality_D', 0.0) * 0.2 +
            (_get_safe_series_local('main_force_buy_execution_alpha_D', 0.0) - _get_safe_series_local('main_force_sell_execution_alpha_D', 0.0)) * 0.2 +
            _get_safe_series_local('main_force_conviction_index_D', 0.0) * 0.1 +
            (_get_safe_series_local('main_force_vwap_up_guidance_D', 0.0) - _get_safe_series_local('main_force_vwap_down_guidance_D', 0.0)) * 0.1 +
            (_get_safe_series_local('vwap_cross_up_intensity_D', 0.0) - _get_safe_series_local('vwap_cross_down_intensity_D', 0.0)) * 0.1 +
            (_get_safe_series_local('main_force_t0_buy_efficiency_D', 0.0) - _get_safe_series_local('main_force_t0_sell_efficiency_D', 0.0)) * 0.1 +
            (_get_safe_series_local('buy_flow_efficiency_index_D', 0.0) - _get_safe_series_local('sell_flow_efficiency_index_D', 0.0)) * 0.1
        ).clip(-1, 1)
        mf_vpoc_premium = _get_safe_series_local('mf_vpoc_premium_D', 0.0)
        vwap_control_composite = (_get_safe_series_local('vwap_buy_control_strength_D', 0.0) - _get_safe_series_local('vwap_sell_control_strength_D', 0.0))
        control_strength = mf_control_leverage * ((vwap_control_composite + 1) / 2)
        mf_cost_advantage_final = (mf_vpoc_premium + 1) / 2
        turnover_rate_f = _get_safe_series_local('turnover_rate_f_D', 0.0)
        turnover_health = pd.Series(1.0, index=df_index, dtype=np.float32)
        turnover_health[turnover_rate_f < 2] = turnover_rate_f[turnover_rate_f < 2] / 2
        turnover_health[turnover_rate_f > 15] = 1 - (turnover_rate_f[turnover_rate_f > 15] - 15) / 10
        turnover_health = turnover_health.clip(0, 1)
        distribution_penalty = (_get_safe_series_local('covert_distribution_signal_D', 0.0) + _get_safe_series_local('supportive_distribution_intensity_D', 0.0)).clip(0, 1) * 0.1
        main_force_scores = {
            'control_strength': control_strength, 'mf_on_peak_flow_normalized': mf_on_peak_flow_normalized, 'mf_intent_composite': (mf_intent_composite + 1) / 2,
            'mf_cost_advantage_final': mf_cost_advantage_final, 'turnover_health': turnover_health, 'distribution_penalty': distribution_penalty
        }
        main_force_weights = {
            'control_strength': 0.3, 'mf_on_peak_flow_normalized': 0.2, 'mf_intent_composite': 0.3,
            'mf_cost_advantage_final': 0.1, 'turnover_health': 0.1, 'distribution_penalty': -0.1
        }
        main_force_score = _nonlinear_fusion(main_force_scores, main_force_weights, volatility_context, sentiment_context, entropy_context)
        # --- 最终 OCH_D 融合 ---
        och_scores = {
            'concentration_score': concentration_score,
            'cost_structure_score': cost_structure_score,
            'sentiment_score': sentiment_score,
            'main_force_score': main_force_score
        }
        och_weights = {
            'concentration_score': 0.25, 'cost_structure_score': 0.25,
            'sentiment_score': 0.25, 'main_force_score': 0.25
        }
        # 最终OCH也使用非线性融合，并考虑情境自适应
        och_score = _nonlinear_fusion(och_scores, och_weights, volatility_context, sentiment_context, entropy_context) * 2 - 1 # 映射回 [-1, 1]
        df['OCH_D'] = och_score.astype(np.float32)
        all_dfs[timeframe] = df
        logger.info("OCH 指标计算完成。")
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





