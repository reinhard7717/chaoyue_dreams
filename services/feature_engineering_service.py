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
        """【V3.8·优化数据阈值】元特征计算车间-降低数据阈值要求并增强数据质量监控"""
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        params = config.get('feature_engineering_params', {}).get('meta_feature_params', {})
        if not params.get('enabled', False):
            return all_dfs
        logger.info(f"元特征计算开始，总数据行数:{len(df)}")
        source_series_configs = [
            {'col': f'close{suffix}', 'prefix': '', 'log_prefix': 'CLOSE', 'min_valid_length': 20},
            {'col': f'main_force_buy_ofi{suffix}', 'prefix': 'MF_BUY_OFI_', 'log_prefix': 'MF_BUY_OFI', 'min_valid_length': 30},
            {'col': f'bid_side_liquidity{suffix}', 'prefix': 'BID_LIQUIDITY_', 'log_prefix': 'BID_LIQUIDITY', 'min_valid_length': 30}
        ]
        data_quality_report = []
        for src_config in source_series_configs:
            source_col = src_config['col']
            prefix = src_config['prefix']
            log_prefix = src_config['log_prefix']
            min_valid_length = src_config['min_valid_length']
            if source_col not in df.columns:
                logger.warning(f"元特征计算缺少核心列'{source_col}'，跳过其元特征计算。")
                data_quality_report.append(f"{log_prefix}: 列不存在")
                continue
            current_series = df[source_col]
            if isinstance(current_series, pd.DataFrame):
                current_series = current_series.iloc[:, 0]
            total_count = len(current_series)
            non_null_count = current_series.notna().sum()
            null_percentage = (total_count - non_null_count) / total_count * 100 if total_count > 0 else 100
            data_quality_report.append(f"{log_prefix}: 总数{total_count}, 非空{non_null_count}, 缺失率{null_percentage:.2f}%")
            logger.debug(f"{log_prefix}数据质量: 总数{total_count}, 非空{non_null_count}, 缺失率{null_percentage:.2f}%")
            clean_series = current_series.dropna()
            clean_values = clean_series.values.astype(np.float64)
            if len(clean_values) < min_valid_length:
                logger.warning(f"{log_prefix}有效数据{len(clean_values)}个, 低于阈值{min_valid_length}, 跳过元特征计算")
                continue
            logger.info(f"{log_prefix}有效数据长度: {len(clean_values)}, 开始计算元特征")
            hurst_window = params.get('hurst_window', 144)
            hurst_col = f'{prefix}HURST_{hurst_window}d{suffix}'
            if hurst_col not in df.columns:
                try:
                    actual_hurst_window = min(hurst_window, len(clean_values) // 2)
                    logger.debug(f"计算{log_prefix}赫斯特指数, 窗口:{actual_hurst_window}(原始:{hurst_window})")
                    df[hurst_col] = current_series.rolling(window=actual_hurst_window, min_periods=actual_hurst_window).apply(
                        lambda x: hurst_exponent(x.dropna().values) if len(x.dropna()) >= actual_hurst_window else np.nan, raw=False
                    )
                    logger.debug(f"{log_prefix}赫斯特指数计算完成, 非空值:{df[hurst_col].notna().sum()}")
                except Exception as e:
                    logger.error(f"{log_prefix}赫斯特指数计算失败: {e}")
                    df[hurst_col] = np.nan
            else:
                logger.debug(f"{log_prefix}赫斯特指数列已存在")
            fd_window = params.get('fractal_dimension_window', 89)
            fd_col = f'{prefix}FRACTAL_DIMENSION_{fd_window}d{suffix}'
            if fd_col not in df.columns:
                try:
                    actual_fd_window = min(fd_window, len(clean_values))
                    logger.debug(f"计算{log_prefix}分形维度, 窗口:{actual_fd_window}(原始:{fd_window})")
                    k_max = max(2, int(np.sqrt(actual_fd_window)))
                    df[fd_col] = current_series.rolling(window=actual_fd_window, min_periods=actual_fd_window).apply(
                        lambda x: _numba_higuchi_fd(x, k_max) if len(x) >= actual_fd_window else np.nan, raw=True
                    )
                    logger.debug(f"{log_prefix}分形维度计算完成, 非空值:{df[fd_col].notna().sum()}")
                except Exception as e:
                    logger.error(f"{log_prefix}分形维度计算失败: {e}")
                    df[fd_col] = np.nan
            else:
                logger.debug(f"{log_prefix}分形维度列已存在")
            se_window = params.get('sample_entropy_window', 13)
            se_tol_ratio = params.get('sample_entropy_tolerance_ratio', 0.2)
            se_col = f'{prefix}SAMPLE_ENTROPY_{se_window}d{suffix}'
            if se_col not in df.columns:
                try:
                    if len(clean_values) < se_window + 1:
                        df[se_col] = np.nan
                        logger.warning(f"{log_prefix}样本熵计算数据不足, 需求:{se_window+1}, 实际:{len(clean_values)}")
                    else:
                        logger.debug(f"计算{log_prefix}样本熵, 窗口:{se_window}")
                        rolling_std = clean_series.rolling(window=se_window, min_periods=se_window).std().values
                        entropy_values = _numba_rolling_sample_entropy(clean_values, se_window, se_tol_ratio, rolling_std)
                        df[se_col] = pd.Series(entropy_values, index=clean_series.index).reindex(df.index)
                        logger.debug(f"{log_prefix}样本熵计算完成, 非空值:{df[se_col].notna().sum()}")
                except Exception as e:
                    logger.error(f"{log_prefix}样本熵计算失败: {e}")
                    df[se_col] = np.nan
            else:
                logger.debug(f"{log_prefix}样本熵列已存在")
            nolds_sampen_window = params.get('approximate_entropy_window', 21)
            nolds_sampen_tol_ratio = params.get('approximate_entropy_tolerance_ratio', 0.2)
            nolds_sampen_col = f'{prefix}NOLDS_SAMPLE_ENTROPY_{nolds_sampen_window}d{suffix}'
            if nolds_sampen_col not in df.columns:
                try:
                    logger.debug(f"计算{log_prefix}NOLDS样本熵, 窗口:{nolds_sampen_window}")
                    df[nolds_sampen_col] = await self.calculator.calculate_nolds_sample_entropy(
                        df=df, period=nolds_sampen_window, column=source_col, tolerance_ratio=nolds_sampen_tol_ratio
                    )
                    logger.debug(f"{log_prefix}NOLDS样本熵计算完成, 非空值:{df[nolds_sampen_col].notna().sum()}")
                except Exception as e:
                    logger.error(f"{log_prefix}NOLDS样本熵计算失败: {e}")
                    df[nolds_sampen_col] = np.nan
            else:
                logger.debug(f"{log_prefix}NOLDS样本熵列已存在")
            fft_window = params.get('fft_energy_ratio_window', 34)
            fft_col = f'{prefix}FFT_ENERGY_RATIO_{fft_window}d{suffix}'
            if fft_col not in df.columns:
                try:
                    if len(clean_values) < fft_window:
                        df[fft_col] = np.nan
                        logger.warning(f"{log_prefix}FFT能量比计算数据不足, 需求:{fft_window}, 实际:{len(clean_values)}")
                    else:
                        logger.debug(f"计算{log_prefix}FFT能量比, 窗口:{fft_window}")
                        fft_energy_ratios_values = _numba_rolling_fft_energy_ratio_core(
                            clean_values,
                            fft_window,
                            low_freq_cutoff_ratio=0.1
                        )
                        df[fft_col] = pd.Series(fft_energy_ratios_values, index=clean_series.index).reindex(df.index)
                        logger.debug(f"{log_prefix}FFT能量比计算完成, 非空值:{df[fft_col].notna().sum()}")
                except Exception as e:
                    logger.error(f"{log_prefix}FFT能量比计算失败: {e}")
                    df[fft_col] = np.nan
            else:
                logger.debug(f"{log_prefix}FFT能量比列已存在")
        logger.info(f"元特征计算数据质量报告: {', '.join(data_quality_report)}")
        atr_col = f'ATR_14{suffix}'
        vi_window = params.get('volatility_instability_window', 21)
        vi_col = f'VOLATILITY_INSTABILITY_INDEX_{vi_window}d{suffix}'
        if atr_col in df.columns and vi_col not in df.columns:
            logger.debug(f"计算波动率不稳定性指数, 窗口:{vi_window}")
            df[vi_col] = df[atr_col].rolling(window=vi_window, min_periods=vi_window).std()
            logger.debug(f"波动率不稳定性指数计算完成, 非空值:{df[vi_col].notna().sum()}")
        generated_cols = [col for col in df.columns if any(prefix in col for prefix in ['HURST', 'FRACTAL_DIMENSION', 'SAMPLE_ENTROPY', 'FFT_ENERGY_RATIO'])]
        logger.info(f"元特征计算完成, 生成元特征列{len(generated_cols)}个, 包括: {generated_cols[:5]}{'...' if len(generated_cols) > 5 else ''}")
        all_dfs[timeframe] = df
        return all_dfs

    async def probe_source_data_quality(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, any]:
        """深度探针: 检测源数据质量, 定位数据缺失问题"""
        timeframe = 'D'
        if timeframe not in all_dfs:
            return {"error": f"时间周期{timeframe}不存在"}
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        probe_columns = [
            f'close{suffix}',
            f'main_force_buy_ofi{suffix}',
            f'bid_side_liquidity{suffix}',
            f'ATR_14{suffix}'
        ]
        results = {
            'timeframe': timeframe,
            'total_rows': len(df),
            'columns_analysis': {},
            'missing_patterns': {},
            'data_summary_stats': {}
        }
        for col in probe_columns:
            if col not in df.columns:
                results['columns_analysis'][col] = {'exists': False, 'message': '列不存在'}
                continue
            series = df[col]
            total = len(series)
            non_null = series.notna().sum()
            null_count = total - non_null
            null_percentage = null_count / total * 100 if total > 0 else 100
            if non_null > 0:
                mean_val = series.mean()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
            else:
                mean_val = std_val = min_val = max_val = None
            results['columns_analysis'][col] = {
                'exists': True,
                'total_values': total,
                'non_null_values': non_null,
                'null_percentage': f"{null_percentage:.2f}%",
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
            if null_count > 0:
                null_mask = series.isna()
                null_indices = np.where(null_mask)[0]
                null_gaps = []
                if len(null_indices) > 0:
                    gap_start = null_indices[0]
                    gap_length = 1
                    for i in range(1, len(null_indices)):
                        if null_indices[i] == null_indices[i-1] + 1:
                            gap_length += 1
                        else:
                            null_gaps.append({'start_index': gap_start, 'length': gap_length})
                            gap_start = null_indices[i]
                            gap_length = 1
                    null_gaps.append({'start_index': gap_start, 'length': gap_length})
                results['missing_patterns'][col] = {
                    'total_null_gaps': len(null_gaps),
                    'largest_gap': max([g['length'] for g in null_gaps]) if null_gaps else 0,
                    'null_gaps_summary': null_gaps[:5]
                }
            if non_null > 0:
                numeric_series = series.dropna()
                results['data_summary_stats'][col] = {
                    'q1': np.percentile(numeric_series, 25),
                    'median': np.median(numeric_series),
                    'q3': np.percentile(numeric_series, 75),
                    'zero_count': (numeric_series == 0).sum(),
                    'negative_count': (numeric_series < 0).sum()
                }
        critical_columns = [f'main_force_buy_ofi{suffix}', f'bid_side_liquidity{suffix}']
        for col in critical_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                results['columns_analysis'][col]['suggestion'] = f"有效数据{non_null_count}个, 建议最小窗口: {max(10, non_null_count//3)}"
        logger.info(f"数据质量探针结果: 总行数{results['total_rows']}, 关键列分析完成")
        return results

    def validate_meta_feature_calculation(self, df: pd.DataFrame) -> Dict[str, any]:
        """验证元特征计算结果的完整性和正确性"""
        validation_results = {
            'total_rows': len(df),
            'missing_source_columns': [],
            'generated_meta_features': [],
            'missing_expected_features': [],
            'coverage_stats': {}
        }
        expected_source_cols = ['close_D', 'main_force_buy_ofi_D', 'bid_side_liquidity_D']
        for col in expected_source_cols:
            if col not in df.columns:
                validation_results['missing_source_columns'].append(col)
        meta_feature_patterns = [
            r'.*HURST_144d_D$',
            r'.*FRACTAL_DIMENSION_89d_D$',
            r'.*SAMPLE_ENTROPY_13d_D$',
            r'.*FFT_ENERGY_RATIO_34d_D$',
            r'VOLATILITY_INSTABILITY_INDEX_21d_D$'
        ]
        for pattern in meta_feature_patterns:
            matching_cols = [col for col in df.columns if re.match(pattern, col)]
            validation_results['generated_meta_features'].extend(matching_cols)
            if not matching_cols:
                validation_results['missing_expected_features'].append(pattern)
        for col in validation_results['generated_meta_features']:
            non_na_count = df[col].notna().sum()
            validation_results['coverage_stats'][col] = {
                'non_na_count': non_na_count,
                'coverage_ratio': non_na_count / len(df) if len(df) > 0 else 0,
                'mean': df[col].mean() if non_na_count > 0 else None,
                'std': df[col].std() if non_na_count > 0 else None
            }
        logger.info(f"元特征验证结果: 源数据列缺失{len(validation_results['missing_source_columns'])}个, "
                    f"生成元特征{len(validation_results['generated_meta_features'])}个, "
                    f"缺失预期特征{len(validation_results['missing_expected_features'])}个")
        return validation_results

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.9 · 几何特征列名修复版】高级模式识别信号生产线
        - 核心修复: 将 `required_cols` 列表中的 `validity_score_D` 替换为 `trendline_validity_score_D`，
                  以匹配 `_process_supplemental_df` 中对 `TrendlineFeature` 数据的命名约定。
        - 核心升级: 全面引入新一代的结构、博弈和风险指标，将市场状态的识别精度从战术层面提升至战略层面。
                      - 盘整识别: 引入“结构张力指数”，要求筹码结构内部应力低。
                      - 吸筹识别: 引入“浮筹清洗效率”，验证主力吸筹的质量。
                      - 突破识别: 引入“突破就绪分”，作为突破信号的强确认。
                      - 派发识别: 引入“衰竭风险指数”，预警趋势顶部的系统性风险。
                      - 细粒度数据集成: 将旧的聚合指标替换为新的买卖双方细粒度指标，提升判断精度。
        - 【新增】引入“诡道博弈”模式识别，如诱空洗盘、假突破诱多等。
        - 【新增】引入情境自适应机制，根据市场波动率和趋势强度调整信号敏感度。
        """
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        # 基于可用数据列，替换缺失的特征
        required_cols = [
            'high_D', 'low_D', 'close_D', 'volume_D', 'pct_change_D', 'VOL_MA_21_D', 'BBW_21_2.0_D', 'ATR_14_D', 'ADX_14_D',
            'open_D', 'amount_D',
            'chip_health_score_D',
            'hidden_accumulation_intensity_D',
            'rally_sell_distribution_intensity_D',
            'rally_buy_support_weakness_D',
            'winner_stability_index_D',
            'cost_structure_skewness_D',
            'dominant_peak_solidity_D',
            'main_force_net_flow_calibrated_D',
            'main_force_buy_execution_alpha_D',
            'main_force_sell_execution_alpha_D',
            'dip_buy_absorption_strength_D',
            'dip_sell_pressure_resistance_D',
            'main_force_on_peak_buy_flow_D',
            'buy_flow_efficiency_index_D',
            'main_force_flow_directionality_D',
            'MA_POTENTIAL_TENSION_INDEX_D',
            'MA_POTENTIAL_ORDERLINESS_SCORE_D',
            'MA_POTENTIAL_COMPRESSION_RATE_D',
            'VPA_EFFICIENCY_D',
            'structural_tension_index_D',
            'floating_chip_cleansing_efficiency_D',
            'breakout_readiness_score_D',
            'exhaustion_risk_index_D',
            # 【新增代码行】诡道博弈相关指标
            'deception_lure_long_intensity_D',
            'deception_lure_short_intensity_D',
            'wash_trade_buy_volume_D',
            'wash_trade_sell_volume_D',
            'price_volume_entropy_D', # 用于情境自适应
            'VOLATILITY_INSTABILITY_INDEX_21d_D', # 用于情境自适应
            'trend_acceleration_score_D', # 来自 AdvancedStructuralMetrics
            'final_charge_intensity_D', # 来自 AdvancedStructuralMetrics
            'platform_conviction_score_D', # 来自 PlatformFeature
            'trend_conviction_score_D', # 来自 MultiTimeframeTrendline
            'quality_score_D', # 来自 PlatformFeature
            'trendline_validity_score_D' # 将 validity_score_D 替换为 trendline_validity_score_D
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"高级模式识别引擎缺少关键数据: {missing_cols}，模块已跳过！")
            # print("当前可用的列名包括:")
            # print(df.columns.tolist())
            return all_dfs
        # --- 1. 【情境自适应调制器】 ---
        # 根据市场波动率和趋势强度动态调整信号敏感度
        volatility_instability = df['VOLATILITY_INSTABILITY_INDEX_21d_D'].fillna(0).rolling(21).mean()
        trend_strength = df['ADX_14_D'].fillna(0).rolling(21).mean()
        # 波动率越高，信号越需要谨慎，趋势越强，信号越需要确认
        # 假设一个简单的调制函数：高波动率降低敏感度，强趋势提高确认度
        # 调制因子在 0.5 到 1.5 之间
        volatility_modulator = (1 - (volatility_instability.rolling(21).rank(pct=True) - 0.5) * 0.5).clip(0.5, 1.5)
        trend_modulator = (1 + (trend_strength.rolling(21).rank(pct=True) - 0.5) * 0.5).clip(0.5, 1.5)
        # --- 2. 【战场状态定义】: 基于“状态+势能”的多因子共振 ---
        cond_high_tension = df['MA_POTENTIAL_TENSION_INDEX_D'] < df['MA_POTENTIAL_TENSION_INDEX_D'].rolling(60).quantile(0.20)
        cond_low_orderliness = df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].abs() < 0.5
        cond_struct_healthy = (df['chip_health_score_D'] > 50) & (df['dominant_peak_solidity_D'] > 0.5)
        cond_low_volatility = df['BBW_21_2.0_D'] < df['BBW_21_2.0_D'].rolling(60).quantile(0.25)
        cond_weak_trend = df['ADX_14_D'] < 25
        cond_low_structural_tension = df['structural_tension_index_D'] < df['structural_tension_index_D'].rolling(60).quantile(0.30)
        df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = cond_high_tension & cond_low_orderliness & cond_struct_healthy & cond_low_volatility & cond_weak_trend & cond_low_structural_tension
        cond_compressing = df['MA_POTENTIAL_COMPRESSION_RATE_D'] < 0
        cond_chip_cleansing = df['floating_chip_cleansing_efficiency_D'] > 0.5
        cond_main_force_accum = (df['hidden_accumulation_intensity_D'] > 0) | \
                                ((df['dip_buy_absorption_strength_D'] > 0.5) & (df['dip_sell_pressure_resistance_D'] < 0.3)) | \
                                cond_chip_cleansing
        cond_peak_flow_positive = df['main_force_on_peak_buy_flow_D'].rolling(3).mean() > 0
        cond_vpa_efficient_accum = df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.5)
        df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & cond_compressing & (cond_main_force_accum | cond_peak_flow_positive) & cond_vpa_efficient_accum
        cond_was_consolidating = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].shift(1).fillna(False)
        cond_orderliness_turn_up = (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'] > 0.8) & (df['MA_POTENTIAL_ORDERLINESS_SCORE_D'].diff() > 0.3)
        cond_main_force_ignition = (df['main_force_net_flow_calibrated_D'] > 0) & \
                                   (df['main_force_buy_execution_alpha_D'] > 0) & \
                                   (df['main_force_flow_directionality_D'] > 0.6)
        cond_price_volume_confirm = (df['pct_change_D'] > 0.01) & \
                                    (df['volume_D'] > df['VOL_MA_21_D'] * 1.2) & \
                                    (df['VPA_EFFICIENCY_D'] > df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.9))
        cond_breakout_ready = df['breakout_readiness_score_D'] > 60
        # 【新增代码行】结合结构与形态指标，增强突破信号
        cond_platform_breakout_potential = (df['platform_conviction_score_D'] > 70) & (df['quality_score_D'] > 0.7) # 高质量平台
        cond_trendline_breakout_confirm = (df['trend_conviction_score_D'] > 80) & (df['trendline_validity_score_D'] > 0.8) # 强趋势线突破
        df['IS_BREAKOUT_D'] = (cond_was_consolidating & cond_orderliness_turn_up & cond_main_force_ignition & cond_price_volume_confirm & cond_breakout_ready) | \
                              (cond_platform_breakout_potential & cond_trendline_breakout_confirm) # 非线性融合
        cond_rally_dist = (df['pct_change_D'] > 0) & (df['rally_sell_distribution_intensity_D'] > 0.5) & (df['rally_buy_support_weakness_D'] > 0.5)
        cond_main_force_outflow = (df['main_force_net_flow_calibrated_D'].rolling(3).sum() < 0) & \
                                  (df['main_force_flow_directionality_D'] < -0.3) & \
                                  (df['main_force_sell_execution_alpha_D'] > 0)
        cond_winner_conviction_drop = df['winner_stability_index_D'].diff() < 0
        cond_resilience_drop = df['dominant_peak_solidity_D'].diff() < 0
        cond_vpa_inefficient_dist = (df['pct_change_D'] > 0) & (df['VPA_EFFICIENCY_D'] < df['VPA_EFFICIENCY_D'].rolling(21).quantile(0.2))
        cond_exhaustion_risk = df['exhaustion_risk_index_D'] > 70
        df['IS_DISTRIBUTION_D'] = cond_rally_dist | (cond_main_force_outflow & cond_winner_conviction_drop & cond_resilience_drop & cond_vpa_inefficient_dist) | cond_exhaustion_risk
        # --- 3. 【诡道博弈模式识别】 ---
        # 诱空洗盘 (Bear Trap Washout)
        # 逻辑：价格快速下跌（pct_change_D < -0.03），伴随洗盘交易量增加 (wash_trade_sell_volume_D > 0.5)，
        # 但主力资金流向（main_force_net_flow_calibrated_D）并未大幅流出，且筹码健康度（chip_health_score_D）保持稳定或小幅下降后迅速回升。
        # 结合情境自适应：在低波动率、弱趋势情境下，诱空洗盘的信号更可信。
        cond_sharp_drop = df['pct_change_D'] < -0.03
        cond_wash_sell_volume = df['wash_trade_sell_volume_D'] > 0.5
        cond_mf_not_outflow = df['main_force_net_flow_calibrated_D'].rolling(3).mean() > -0.01 # 主力净流出不明显
        cond_chip_resilient = (df['chip_health_score_D'].diff() > -10) | (df['chip_health_score_D'].shift(1) < df['chip_health_score_D']) # 筹码健康度未大幅恶化或开始回升
        df['IS_BEAR_TRAP_WASHOUT_D'] = (cond_sharp_drop & cond_wash_sell_volume & cond_mf_not_outflow & cond_chip_resilient) * volatility_modulator.fillna(1) # 乘以调制因子
        # 假突破诱多 (Bull Trap Lure)
        # 逻辑：价格突破（IS_BREAKOUT_D），但伴随主力资金流向（main_force_net_flow_calibrated_D）负向背离，
        # 且诱多欺骗强度（deception_lure_long_intensity_D）高，筹码派发迹象（rally_sell_distribution_intensity_D）明显。
        # 结合情境自适应：在高波动率、强趋势情境下，假突破的风险更高。
        cond_breakout_signal = df['IS_BREAKOUT_D']
        cond_mf_divergence = df['main_force_net_flow_calibrated_D'].rolling(5).mean().diff() < 0 # 主力资金动能减弱
        cond_lure_long = df['deception_lure_long_intensity_D'] > 0.6
        cond_dist_sign = df['rally_sell_distribution_intensity_D'] > 0.5
        df['IS_BULL_TRAP_LURE_D'] = (cond_breakout_signal & cond_mf_divergence & cond_lure_long & cond_dist_sign) * trend_modulator.fillna(1) # 乘以调制因子
        # 高位对倒出货 (High-Level Wash Trade Distribution)
        # 逻辑：股价处于高位（例如高于过去60日最高价的80%），成交量异常放大（volume_D > VOL_MA_21_D * 2），
        # 对倒买卖量（wash_trade_buy_volume_D + wash_trade_sell_volume_D）高，但价格涨幅有限（pct_change_D < 0.01），
        # 且主力资金净流出（main_force_net_flow_calibrated_D < 0）。
        cond_high_price = df['close_D'] > df['high_D'].rolling(60).max() * 0.8
        cond_volume_spike = df['volume_D'] > df['VOL_MA_21_D'] * 2
        cond_high_wash_trade = (df['wash_trade_buy_volume_D'] + df['wash_trade_sell_volume_D']) > 1.0 # 假设对倒量归一化后大于1
        cond_limited_gain = df['pct_change_D'] < 0.01
        cond_mf_outflow = df['main_force_net_flow_calibrated_D'] < 0
        df['IS_HIGH_LEVEL_WASH_DISTRIBUTION_D'] = cond_high_price & cond_volume_spike & cond_high_wash_trade & cond_limited_gain & cond_mf_outflow
        # --- 4. 【通达信模式集成】 ---
        if 'amount_D' in df.columns:
            ema_amount_5 = df['amount_D'].ewm(span=5, adjust=False).mean()
            ff_ratio = ema_amount_5 / ema_amount_5.shift(1)
            llv_c_120 = df['close_D'].rolling(window=120, min_periods=1).min()
            hhv_c_120 = df['close_D'].rolling(window=120, min_periods=1).max()
            ww_position = (df['close_D'] - llv_c_120) / (hhv_c_120 - llv_c_120).replace(0, np.nan) * 100
            cond_ff_high = (ff_ratio >= 2)
            cond_ww_low = (ww_position < 35)
            cond_ww_any = (ww_position < 100)
            cond_bars_gt_30 = (df.index.to_series().diff().dt.days.cumsum().fillna(0) > 30)
            cond_bars_lt_50 = (df.index.to_series().diff().dt.days.cumsum().fillna(0) < 50)
            is_bazhan = (cond_ff_high & cond_ww_low & cond_bars_gt_30) | (cond_ff_high & cond_ww_any & cond_bars_lt_50)
            df['IS_BAZHAN_D'] = is_bazhan.fillna(False)
        else:
            df['IS_BAZHAN_D'] = False
            logger.warning("高级模式识别引擎缺少 'amount_D' 列，无法计算 '霸占' 模式。")
        if 'open_D' in df.columns and 'volume_D' in df.columns:
            cond_open_below_prev_low = (df['open_D'] < df['low_D'].shift(1))
            cond_close_above_open = (df['close_D'] > df['open_D'])
            cond_volume_increase = (df['volume_D'] > df['volume_D'].shift(1))
            is_ww1 = cond_open_below_prev_low & cond_close_above_open & cond_volume_increase
            df['IS_WW1_D'] = is_ww1.fillna(False)
        else:
            df['IS_WW1_D'] = False
            logger.warning("高级模式识别引擎缺少 'open_D' 或 'volume_D' 列，无法计算 'WW1' 模式。")
        # --- 5. 【信号整合与输出】 ---
        pattern_cols = [
            'IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D',
            'IS_BAZHAN_D', 'IS_WW1_D',
            'IS_BEAR_TRAP_WASHOUT_D', 'IS_BULL_TRAP_LURE_D', 'IS_HIGH_LEVEL_WASH_DISTRIBUTION_D' # 新增诡道博弈信号
        ]
        for col in pattern_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)
        all_dfs[timeframe] = df
        logger.info("高级模式识别引擎(V3.9 几何特征列名修复版)分析完成。")
        return all_dfs

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
        print(f"DEBUG: 对手盘衰竭指数参数配置: {exhaustion_params}, 启用状态: {exhaustion_params.get('enabled')}, df_minute 形状: {df_minute.shape if df_minute is not None else 'None'}")
        if exhaustion_params.get('enabled') and df_minute is not None:
            print(f"DEBUG: 准备调用对手盘衰竭指数计算，分钟数据形状: {df_minute.shape}")
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





