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

@numba.njit(parallel=True, cache=True)
def _numba_rolling_fractal_dimension(data: np.ndarray, window: int, k_max: int) -> np.ndarray:
    """Numba并行化滚动分形维度计算"""
    n = len(data)
    result = np.full(n, np.nan, dtype=np.float64)
    k_values = np.arange(1, k_max + 1)
    for i in prange(window - 1, n):
        window_data = data[i - window + 1:i + 1]
        l_values = np.zeros(len(k_values))
        for j, k in enumerate(k_values):
            lk = 0.0
            max_m = (window - 1) // k
            if max_m < 1:
                l_values[j] = 0.0
                continue
            for m in range(max_m):
                idx1 = m * k
                idx2 = (m + 1) * k
                if idx2 < len(window_data):
                    lk += abs(window_data[idx2] - window_data[idx1])
            lk = lk * (window - 1) / (max_m * k * k)
            l_values[j] = lk
        # 线性回归拟合log(1/k) vs log(L(k))
        valid_mask = l_values > 0
        if np.sum(valid_mask) < 2:
            result[i] = np.nan
            continue
        x = np.log(1.0 / k_values[valid_mask])
        y = np.log(l_values[valid_mask])
        n_valid = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)
        denom = n_valid * sum_x2 - sum_x * sum_x
        if denom == 0:
            result[i] = np.nan
            continue
        slope = (n_valid * sum_xy - sum_x * sum_y) / denom
        result[i] = slope
    return result[window-1:]

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
        """【V4.2 · 形状匹配修正版】元特征计算车间 - 修复数组形状错误"""
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe]
        suffix = f"_{timeframe}"
        params = config.get('feature_engineering_params', {}).get('meta_feature_params', {})
        if not params.get('enabled', False):
            return all_dfs
        source_series_configs = [
            {'col': f'close{suffix}', 'prefix': ''},
            {'col': f'main_force_buy_ofi{suffix}', 'prefix': 'MF_BUY_OFI_'},
            {'col': f'bid_side_liquidity{suffix}', 'prefix': 'BID_LIQUIDITY_'}
        ]
        for src_config in source_series_configs:
            source_col = src_config['col']
            prefix = src_config['prefix']
            if source_col not in df.columns:
                logger.warning(f"元特征计算缺少核心列 '{source_col}'，跳过其元特征计算。")
                continue
            series_values = df[source_col].values.astype(np.float64)
            valid_mask = ~np.isnan(series_values)
            clean_values = series_values[valid_mask]
            if len(clean_values) == 0:
                continue
            # 1. 赫斯特指数
            hurst_window = params.get('hurst_window', 144)
            hurst_col = f'{prefix}HURST_{hurst_window}d{suffix}'
            if hurst_col not in df.columns:
                try:
                    if len(clean_values) >= hurst_window:
                        df[hurst_col] = df[source_col].rolling(window=hurst_window, min_periods=hurst_window).apply(
                            lambda x: hurst_exponent(x.dropna().values) if len(x.dropna()) >= hurst_window else np.nan, raw=False
                        )
                    else:
                        df[hurst_col] = np.nan
                except Exception as e:
                    logger.error(f"赫斯特指数计算失败: {e}")
                    df[hurst_col] = np.nan
            # 2. 分形维度
            fd_window = params.get('fractal_dimension_window', 89)
            fd_col = f'{prefix}FRACTAL_DIMENSION_{fd_window}d{suffix}'
            if fd_col not in df.columns:
                try:
                    if len(clean_values) >= fd_window:
                        from numba import njit
                        @njit(cache=True)
                        def numba_higuchi_fd_windowed(data, window, k_max):
                            n = len(data)
                            result = np.full(n, np.nan, dtype=np.float64)
                            for i in range(window - 1, n):
                                window_data = data[i - window + 1:i + 1]
                                k_values = np.arange(1, k_max + 1)
                                l_values = np.zeros(len(k_values))
                                for j, k in enumerate(k_values):
                                    lk = 0.0
                                    max_m = (window - 1) // k
                                    if max_m < 1:
                                        l_values[j] = 0.0
                                        continue
                                    for m in range(max_m):
                                        idx1 = m * k
                                        idx2 = (m + 1) * k
                                        if idx2 < len(window_data):
                                            lk += abs(window_data[idx2] - window_data[idx1])
                                    lk = lk * (window - 1) / (max_m * k * k)
                                    l_values[j] = lk
                                valid_mask = l_values > 0
                                if np.sum(valid_mask) < 2:
                                    result[i] = np.nan
                                    continue
                                x = np.log(1.0 / k_values[valid_mask])
                                y = np.log(l_values[valid_mask])
                                n_valid = len(x)
                                sum_x = np.sum(x)
                                sum_y = np.sum(y)
                                sum_xy = np.sum(x * y)
                                sum_x2 = np.sum(x * x)
                                denom = n_valid * sum_x2 - sum_x * sum_x
                                if denom == 0:
                                    result[i] = np.nan
                                    continue
                                slope = (n_valid * sum_xy - sum_x * sum_y) / denom
                                result[i] = slope
                            return result[window-1:]
                        k_max = int(np.sqrt(fd_window))
                        fd_window_data = numba_higuchi_fd_windowed(clean_values, fd_window, k_max)
                        fd_values = np.full(len(series_values), np.nan, dtype=np.float64)
                        valid_indices = np.where(valid_mask)[0]
                        fd_values[valid_indices[fd_window-1:]] = fd_window_data
                        df[fd_col] = fd_values
                    else:
                        df[fd_col] = np.nan
                except Exception as e:
                    logger.error(f"分形维度计算失败: {e}")
                    df[fd_col] = np.nan
            # 3. 样本熵 (修复形状不匹配)
            se_window = params.get('sample_entropy_window', 13)
            se_tol_ratio = params.get('sample_entropy_tolerance_ratio', 0.2)
            se_col = f'{prefix}SAMPLE_ENTROPY_{se_window}d{suffix}'
            if se_col not in df.columns:
                try:
                    if len(clean_values) >= se_window + 1:
                        rolling_std = np.full(len(clean_values), np.nan, dtype=np.float64)
                        for i in range(se_window - 1, len(clean_values)):
                            window_data = clean_values[i - se_window + 1:i + 1]
                            rolling_std[i] = np.std(window_data)
                        entropy_values = _numba_rolling_sample_entropy(clean_values, se_window, se_tol_ratio, rolling_std)
                        se_result = np.full(len(series_values), np.nan, dtype=np.float64)
                        valid_indices = np.where(valid_mask)[0]
                        se_result[valid_indices[se_window-1:]] = entropy_values[se_window-1:]
                        df[se_col] = se_result
                    else:
                        df[se_col] = np.nan
                except Exception as e:
                    logger.error(f"样本熵计算失败: {e}")
                    df[se_col] = np.nan
            # 4. NOLDS样本熵
            nolds_sampen_window = params.get('approximate_entropy_window', 21)
            nolds_sampen_tol_ratio = params.get('approximate_entropy_tolerance_ratio', 0.2)
            nolds_sampen_col = f'{prefix}NOLDS_SAMPLE_ENTROPY_{nolds_sampen_window}d{suffix}'
            if nolds_sampen_col not in df.columns:
                try:
                    if len(clean_values) >= nolds_sampen_window:
                        df[nolds_sampen_col] = await self.calculator.calculate_nolds_sample_entropy(
                            df=df, period=nolds_sampen_window, column=source_col, tolerance_ratio=nolds_sampen_tol_ratio
                        )
                    else:
                        df[nolds_sampen_col] = np.nan
                except Exception as e:
                    logger.error(f"NOLDS样本熵计算失败: {e}")
                    df[nolds_sampen_col] = np.nan
            # 5. FFT能量比
            fft_window = params.get('fft_energy_ratio_window', 34)
            fft_col = f'{prefix}FFT_ENERGY_RATIO_{fft_window}d{suffix}'
            if fft_col not in df.columns:
                try:
                    if len(clean_values) >= fft_window:
                        fft_values = np.full(len(series_values), np.nan, dtype=np.float64)
                        for i in range(fft_window - 1, len(clean_values)):
                            window_data = clean_values[i - fft_window + 1:i + 1]
                            fft_result = np.fft.fft(window_data - np.mean(window_data))
                            freqs = np.fft.fftfreq(fft_window)
                            power_spectrum = np.abs(fft_result) ** 2
                            low_freq_mask = np.abs(freqs) <= 0.1
                            high_freq_mask = np.abs(freqs) > 0.1
                            low_freq_energy = np.sum(power_spectrum[low_freq_mask])
                            high_freq_energy = np.sum(power_spectrum[high_freq_mask])
                            if high_freq_energy == 0:
                                fft_values[valid_mask][i] = np.nan
                            else:
                                fft_values[valid_mask][i] = low_freq_energy / high_freq_energy
                        df[fft_col] = fft_values
                    else:
                        df[fft_col] = np.nan
                except Exception as e:
                    logger.error(f"FFT能量比计算失败: {e}")
                    df[fft_col] = np.nan
            # 6. 波动率不稳定性
            vi_window = params.get('volatility_instability_window', 21)
            vi_col = f'VOLATILITY_INSTABILITY_INDEX_{vi_window}d{suffix}'
            atr_col = f'ATR_14{suffix}'
            if atr_col in df.columns and vi_col not in df.columns:
                try:
                    atr_values = df[atr_col].values.astype(np.float64)
                    atr_valid_mask = ~np.isnan(atr_values)
                    if np.sum(atr_valid_mask) >= vi_window:
                        vi_values = np.full(len(atr_values), np.nan, dtype=np.float64)
                        atr_valid_values = atr_values[atr_valid_mask]
                        for i in range(vi_window - 1, len(atr_valid_values)):
                            vi_values[np.where(atr_valid_mask)[0][i]] = np.std(atr_valid_values[i - vi_window + 1:i + 1])
                        df[vi_col] = vi_values
                    else:
                        df[vi_col] = np.nan
                except Exception as e:
                    df[vi_col] = np.nan
        all_dfs[timeframe] = df
        return all_dfs

    def _vectorized_hurst_exponent(self, data: np.ndarray, window: int) -> float:
        """向量化赫斯特指数计算"""
        n = len(data)
        if n < window:
            return np.nan
        lags = np.arange(2, min(20, n // 4))
        tau = np.zeros_like(lags, dtype=np.float64)
        for i, lag in enumerate(lags):
            diffs = np.abs(data[lag:] - data[:-lag])
            tau[i] = np.mean(diffs)
        if len(lags) < 2:
            return np.nan
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]

    def _vectorized_rolling_std(self, data: np.ndarray, window: int) -> np.ndarray:
        """向量化滚动标准差计算"""
        n = len(data)
        if n < window:
            return np.full(n, np.nan, dtype=np.float64)
        result = np.full(n, np.nan, dtype=np.float64)
        # 使用累积和计算滚动标准差
        cumsum = np.cumsum(data)
        cumsum_sq = np.cumsum(data ** 2)
        for i in range(window - 1, n):
            if i == window - 1:
                sum_ = cumsum[i]
                sum_sq = cumsum_sq[i]
            else:
                sum_ = cumsum[i] - cumsum[i - window]
                sum_sq = cumsum_sq[i] - cumsum_sq[i - window]
            mean = sum_ / window
            variance = (sum_sq - window * mean ** 2) / (window - 1)
            result[i] = np.sqrt(variance) if variance > 0 else 0
        return result[window-1:]

    def _vectorized_fft_energy_ratio(self, data: np.ndarray, window: int) -> float:
        """向量化FFT能量比计算"""
        n = len(data)
        if n < window:
            return np.nan
        fft_result = np.fft.fft(data - np.mean(data))
        freqs = np.fft.fftfreq(n)
        power_spectrum = np.abs(fft_result) ** 2
        low_freq_mask = np.abs(freqs) <= 0.1
        high_freq_mask = np.abs(freqs) > 0.1
        low_freq_energy = np.sum(power_spectrum[low_freq_mask])
        high_freq_energy = np.sum(power_spectrum[high_freq_mask])
        if high_freq_energy == 0:
            return np.nan
        return low_freq_energy / high_freq_energy

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe].copy()
        print(f"=== 模式识别引擎(V5.3)开始分析，数据长度: {len(df)} ===")
        print(f"数据时间范围: {df.index[0]} 到 {df.index[-1]}")
        if 'ADX_14_D' not in df.columns:
            print("错误：缺少ADX数据，无法判断市场阶段")
            return all_dfs
        current_adx = df['ADX_14_D'].iloc[-1]
        market_phase = 'TRENDING' if current_adx > 25 else 'CONSOLIDATING'
        print(f"=== 市场阶段判断 ===")
        print(f"当前ADX: {current_adx:.1f}, 市场阶段: {market_phase}")
        print(f"当前价格: {df['close_D'].iloc[-1]:.2f}, 涨跌幅: {df['pct_change_D'].iloc[-1]:.2%}")
        if df['pct_change_D'].max() > 10 or df['pct_change_D'].min() < -10:
            df['pct_change_D'] = df['pct_change_D'] / 100.0
            print(f"涨跌幅数据已转换为小数格式，最新值: {df['pct_change_D'].iloc[-1]:.2%}")
        rolling_window = min(120, len(df))
        if rolling_window < 30:
            print("数据长度不足，退出分析")
            return all_dfs
        print(f"使用滚动窗口: {rolling_window}天")
        print(f"=== 阶段自适应阈值计算 ===")
        dynamic_thresholds = {}
        threshold_config = {}
        if market_phase == 'TRENDING':
            print("趋势市场阈值配置（相对宽松）:")
            threshold_config = {
                'chip_health_score_D': ('q25', 0.25),
                'dominant_peak_solidity_D': ('q40', 0.4),
                'main_force_net_flow_calibrated_D': ('q50', 0.5),
                'structural_tension_index_D': ('q50', 0.5),
                'breakout_readiness_score_D': ('q60', 0.6),
                'platform_conviction_score_D': ('q60', 0.6),
                'trend_conviction_score_D': ('q60', 0.6),
                'pct_change_D': ('q40', 0.4),
            }
        else:
            print("盘整市场阈值配置（相对严格）:")
            threshold_config = {
                'chip_health_score_D': ('q30', 0.3),
                'dominant_peak_solidity_D': ('q30', 0.3),
                'main_force_net_flow_calibrated_D': ('q60', 0.6),
                'structural_tension_index_D': ('q30', 0.3),
                'breakout_readiness_score_D': ('q70', 0.7),
                'platform_conviction_score_D': ('q70', 0.7),
                'trend_conviction_score_D': ('q70', 0.7),
                'pct_change_D': ('q60', 0.6),
            }
        for col, (method, param) in threshold_config.items():
            if col in df.columns and not df[col].isna().all():
                try:
                    threshold_series = df[col].rolling(rolling_window, min_periods=20).quantile(param)
                    dynamic_thresholds[col] = threshold_series.fillna(method='ffill')
                    if col in ['chip_health_score_D', 'pct_change_D', 'ADX_14_D']:
                        print(f"  {col}: 阈值={threshold_series.iloc[-1]:.2f}, 当前值={df[col].iloc[-1]:.2f}")
                except Exception as e:
                    print(f"阈值计算失败 {col}: {e}")
                    if col == 'chip_health_score_D':
                        dynamic_thresholds[col] = pd.Series(40, index=df.index)
                    elif col == 'dominant_peak_solidity_D':
                        dynamic_thresholds[col] = pd.Series(0.4, index=df.index)
                    else:
                        dynamic_thresholds[col] = pd.Series(0, index=df.index)
        print(f"=== 市场阶段自适应计算 ===")
        if market_phase == 'TRENDING':
            print("执行趋势市场逻辑...")
            df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = False
            df['IS_ACCUMULATION_D'] = False
            trend_continuation_score = pd.Series(0, index=df.index, dtype=float)
            trend_reversal_score = pd.Series(0, index=df.index, dtype=float)
            trend_correction_score = pd.Series(0, index=df.index, dtype=float)
            if 'pct_change_D' in df.columns:
                pct_threshold = dynamic_thresholds.get('pct_change_D', pd.Series(0, index=df.index))
                trend_continuation_score = trend_continuation_score + ((df['pct_change_D'] > pct_threshold) * 2.0).fillna(0)
            if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                trend_continuation_score = trend_continuation_score + ((df['volume_D'] > df['VOL_MA_21_D'] * 1.2) * 1.5).fillna(0)
            if 'main_force_net_flow_calibrated_D' in df.columns:
                mf_threshold = dynamic_thresholds.get('main_force_net_flow_calibrated_D', pd.Series(0, index=df.index))
                trend_continuation_score = trend_continuation_score + ((df['main_force_net_flow_calibrated_D'] > mf_threshold) * 1.2).fillna(0)
            if 'trend_acceleration_score_D' in df.columns:
                trend_continuation_score = trend_continuation_score + ((df['trend_acceleration_score_D'] > 0) * 1.0).fillna(0)
            df['IS_TREND_CONTINUATION_D'] = (trend_continuation_score >= 4.0)
            if 'pct_change_D' in df.columns:
                print("=== 回调信号动态阈值计算 ===")
                print("[回调信号探针] 回调幅度条件重构...")
                # 回调信号核心逻辑简化重构
                # 回调幅度在-5%到-0.5%之间（即-0.05到-0.005）
                correction_min_magnitude = -0.005  # 最小回调幅度（绝对值）
                correction_max_magnitude = -0.05   # 最大回调幅度（绝对值）
                print(f"回调幅度允许范围: [{correction_max_magnitude:.4f}, {correction_min_magnitude:.4f}]")
                
                # 回调幅度条件：回调幅度在合理范围内且为负值
                cond_small_correction = (df['pct_change_D'] < 0) & \
                                       (df['pct_change_D'] >= correction_max_magnitude) & \
                                       (df['pct_change_D'] <= correction_min_magnitude)
                
                # 新增：回调应该出现在上涨趋势中（前5个交易日中有至少3天上涨）
                if 'pct_change_D' in df.columns:
                    # 计算前5个交易日的上涨天数
                    positive_days = (df['pct_change_D'].shift(1).rolling(5).sum() > 0)
                    cond_recent_uptrend = (positive_days >= 3)
                else:
                    cond_recent_uptrend = pd.Series(True, index=df.index)
                
                # 新增：回调日不应该跌破重要支撑（20日均线）
                if 'close_D' in df.columns:
                    ma20 = df['close_D'].rolling(20).mean()
                    cond_not_break_ma20 = (df['close_D'] > ma20 * 0.95)  # 收盘价不低于20日线5%
                else:
                    cond_not_break_ma20 = pd.Series(True, index=df.index)
                
                # 探针：打印关键日期的回调判断
                test_dates = ['2025-12-30', '2025-12-23', '2025-12-29', '2025-12-24', '2025-12-26']
                for test_date in test_dates:
                    if test_date in df.index:
                        idx = df.index.get_loc(test_date)
                        print(f"[回调信号探针] 日期: {test_date}")
                        print(f"  涨跌幅: {df['pct_change_D'].iloc[idx]:.4f} ({df['pct_change_D'].iloc[idx]*100:.2f}%)")
                        print(f"  回调幅度条件: {cond_small_correction.iloc[idx]}")
                
                cond_volume_support = (df['volume_D'] > df['VOL_MA_21_D'] * 0.8) if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns else pd.Series(True, index=df.index)
                
                # 主力资金条件：放宽但有限度
                if 'main_force_net_flow_calibrated_D' in df.columns:
                    # 使用更严格的动态阈值
                    mf_median = df['main_force_net_flow_calibrated_D'].rolling(20).median().fillna(method='ffill')
                    mf_std = df['main_force_net_flow_calibrated_D'].rolling(20).std().fillna(0)
                    # 主力资金条件：不低于中位数减去1个标准差
                    cond_mf_not_outflow = (df['main_force_net_flow_calibrated_D'] > mf_median - mf_std * 1.0)
                else:
                    cond_mf_not_outflow = pd.Series(True, index=df.index)
                
                # 筹码健康条件：使用动态阈值
                if 'chip_health_score_D' in df.columns:
                    chip_median = df['chip_health_score_D'].rolling(20).median().fillna(method='ffill')
                    chip_std = df['chip_health_score_D'].rolling(20).std().fillna(0)
                    # 筹码健康条件：不低于中位数减去0.5个标准差
                    cond_chip_stable = (df['chip_health_score_D'] > chip_median - chip_std * 0.5)
                else:
                    cond_chip_stable = pd.Series(True, index=df.index)
                
                if 'close_D' in df.columns:
                    ma5 = df['close_D'].rolling(5).mean()
                    ma10 = df['close_D'].rolling(10).mean()
                    # 回调日收盘价应在5日线附近（不超过8%的偏离）
                    cond_trend_support = (df['close_D'] >= ma5 * 0.92) & (ma5 > ma10 * 0.94)
                else:
                    cond_trend_support = pd.Series(True, index=df.index)
                
                if 'close_D' in df.columns:
                    ma_short = df['close_D'].rolling(5).mean()
                    ma_long = df['close_D'].rolling(20).mean()
                    cond_trend_direction = (ma_short > ma_long * 0.94)  # 允许短期均线略低于长期均线
                else:
                    cond_trend_direction = pd.Series(True, index=df.index)
                
                if 'pct_change_D' in df.columns:
                    recent_volatility = df['pct_change_D'].abs().rolling(10).mean()
                    cond_slow_correction = (df['pct_change_D'].abs() < recent_volatility * 2.5)  # 放宽到2.5倍波动率
                else:
                    cond_slow_correction = pd.Series(True, index=df.index)
                
                # 回调信号权重评分系统
                correction_score = pd.Series(0, index=df.index, dtype=float)
                correction_score = correction_score + (cond_small_correction * 2.5).fillna(0)  # 回调幅度条件最重要
                correction_score = correction_score + (cond_volume_support * 1.5).fillna(0)  # 成交量支持重要
                correction_score = correction_score + (cond_trend_support * 1.5).fillna(0)  # 趋势支撑重要
                correction_score = correction_score + (cond_trend_direction * 1.2).fillna(0)  # 趋势方向重要
                correction_score = correction_score + (cond_slow_correction * 1.0).fillna(0)  # 回调速度重要
                correction_score = correction_score + (cond_recent_uptrend * 1.0).fillna(0)  # 近期上涨趋势重要
                correction_score = correction_score + (cond_not_break_ma20 * 1.0).fillna(0)  # 不破20日线重要
                correction_score = correction_score + (cond_mf_not_outflow * 0.8).fillna(0)  # 主力资金条件
                correction_score = correction_score + (cond_chip_stable * 0.8).fillna(0)  # 筹码健康条件
                
                # 综合回调信号：总分达到7分即可，且回调幅度条件必须满足
                df['IS_TREND_CORRECTION_D'] = (correction_score >= 7.0) & cond_small_correction
                
                # 新增：回调信号与趋势延续信号互斥
                if 'IS_TREND_CONTINUATION_D' in df.columns:
                    df['IS_TREND_CORRECTION_D'] = df['IS_TREND_CORRECTION_D'] & (~df['IS_TREND_CONTINUATION_D'])
                
                # 原始版本：严格要求所有条件
                df['IS_TREND_CORRECTION_RAW_D'] = cond_small_correction & cond_volume_support & cond_mf_not_outflow & cond_chip_stable
                
                # 探针：检查最终回调信号
                if '2025-12-30' in df.index:
                    idx = df.index.get_loc('2025-12-30')
                    print(f"[最终回调信号探针] 日期: 2025-12-30")
                    print(f"  回调幅度条件: {cond_small_correction.iloc[idx]}")
                    print(f"  成交量条件: {cond_volume_support.iloc[idx]}")
                    print(f"  趋势支撑条件: {cond_trend_support.iloc[idx]}")
                    print(f"  趋势方向条件: {cond_trend_direction.iloc[idx]}")
                    print(f"  回调速度条件: {cond_slow_correction.iloc[idx]}")
                    print(f"  近期上涨趋势条件: {cond_recent_uptrend.iloc[idx] if 'cond_recent_uptrend' in locals() else 'N/A'}")
                    print(f"  不破20日线条件: {cond_not_break_ma20.iloc[idx] if 'cond_not_break_ma20' in locals() else 'N/A'}")
                    print(f"  主力资金条件: {cond_mf_not_outflow.iloc[idx]}")
                    print(f"  筹码健康条件: {cond_chip_stable.iloc[idx]}")
                    print(f"  回调评分: {correction_score.iloc[idx]:.1f}")
                    print(f"  趋势延续信号: {df['IS_TREND_CONTINUATION_D'].iloc[idx] if 'IS_TREND_CONTINUATION_D' in df.columns else 'N/A'}")
                    print(f"  最终回调信号: {df['IS_TREND_CORRECTION_D'].iloc[idx]}")
            if 'pct_change_D' in df.columns:
                cond_sharp_drop = df['pct_change_D'] < -0.05  # 修改为-5%，与回调信号区分
                cond_volume_spike = (df['volume_D'] > df['VOL_MA_21_D'] * 1.5) if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns else pd.Series(False, index=df.index)
                if 'main_force_net_flow_calibrated_D' in df.columns:
                    mf_low_quantile = df['main_force_net_flow_calibrated_D'].rolling(20).quantile(0.2).fillna(method='ffill')
                    cond_mf_heavy_outflow = (df['main_force_net_flow_calibrated_D'] < mf_low_quantile)
                else:
                    cond_mf_heavy_outflow = pd.Series(False, index=df.index)
                if 'chip_health_score_D' in df.columns:
                    chip_low_quantile = df['chip_health_score_D'].rolling(20).quantile(0.3).fillna(method='ffill')
                    cond_chip_deteriorate = (df['chip_health_score_D'] < chip_low_quantile)
                else:
                    cond_chip_deteriorate = pd.Series(False, index=df.index)
                df['IS_TREND_REVERSAL_D'] = cond_sharp_drop & (cond_volume_spike | cond_mf_heavy_outflow) & cond_chip_deteriorate
            print(f"趋势市场信号统计:")
            print(f"  趋势延续信号原始触发: {df['IS_TREND_CONTINUATION_D'].sum()}次")
            print(f"  趋势回调信号原始触发: {df['IS_TREND_CORRECTION_D'].sum() if 'IS_TREND_CORRECTION_D' in df.columns else 0}次")
            print(f"  趋势反转信号原始触发: {df['IS_TREND_REVERSAL_D'].sum() if 'IS_TREND_REVERSAL_D' in df.columns else 0}次")
            if 'IS_TREND_CORRECTION_D' in df.columns and 'IS_TREND_CORRECTION_RAW_D' in df.columns:
                print(f"  趋势回调信号(增强版): {df['IS_TREND_CORRECTION_D'].sum()}次")
                print(f"  趋势回调信号(原始版): {df['IS_TREND_CORRECTION_RAW_D'].sum()}次")
                recent_days = 5
                recent_df = df.tail(recent_days)
                for idx, date in enumerate(recent_df.index):
                    date_str = date.strftime('%Y-%m-%d')
                    print(f"    日期 {date_str}: 增强版={recent_df['IS_TREND_CORRECTION_D'].iloc[idx]}, 原始版={recent_df['IS_TREND_CORRECTION_RAW_D'].iloc[idx]}, 涨跌幅={recent_df['pct_change_D'].iloc[idx]:.2%}")
        else:
            print("执行盘整市场逻辑...")
            df['IS_TREND_CONTINUATION_D'] = False
            df['IS_TREND_CORRECTION_D'] = False
            df['IS_TREND_REVERSAL_D'] = False
            chip_stability_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'chip_health_score_D' in df.columns:
                chip_thresh = dynamic_thresholds.get('chip_health_score_D', pd.Series(40, index=df.index))
                chip_stability_evidence = (df['chip_health_score_D'] > chip_thresh).astype(int)
                print(f"筹码稳定性证据: 触发{chip_stability_evidence.sum()}次")
            volatility_compression_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'BBW_21_2.0_D' in df.columns:
                bbw_quantile = df['BBW_21_2.0_D'].rolling(rolling_window, min_periods=20).quantile(0.3).fillna(method='ffill')
                volatility_compression_evidence = (df['BBW_21_2.0_D'] < bbw_quantile).astype(int)
                print(f"波动率收缩证据: 触发{volatility_compression_evidence.sum()}次")
            fund_flow_balance_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'main_force_net_flow_calibrated_D' in df.columns:
                mf_std = df['main_force_net_flow_calibrated_D'].rolling(rolling_window, min_periods=20).std().fillna(0)
                fund_flow_balance_evidence = (abs(df['main_force_net_flow_calibrated_D']) < mf_std * 0.5).astype(int)
                print(f"资金流平衡证据: 触发{fund_flow_balance_evidence.sum()}次")
            structure_tension_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'structural_tension_index_D' in df.columns:
                st_threshold = dynamic_thresholds.get('structural_tension_index_D', pd.Series(0.3, index=df.index))
                structure_tension_evidence = (df['structural_tension_index_D'] < st_threshold).astype(int)
                print(f"结构张力证据: 触发{structure_tension_evidence.sum()}次")
            consolidation_score = (chip_stability_evidence + volatility_compression_evidence + fund_flow_balance_evidence + structure_tension_evidence)
            df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = (consolidation_score >= 3) & (df['ADX_14_D'] < 25)
            print(f"盘整信号: {df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'].sum()}次")
            hidden_buy_evidence = pd.Series(0, index=df.index, dtype=int)
            if all(col in df.columns for col in ['main_force_buy_execution_alpha_D', 'main_force_net_flow_calibrated_D']):
                mf_quantile = df['main_force_net_flow_calibrated_D'].rolling(rolling_window, min_periods=20).quantile(0.5).fillna(method='ffill')
                hidden_buy_evidence = ((df['main_force_buy_execution_alpha_D'] > 0) & (df['main_force_net_flow_calibrated_D'] > mf_quantile)).astype(int)
            retail_panic_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'retail_panic_surrender_index_D' in df.columns:
                rp_quantile = df['retail_panic_surrender_index_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
                retail_panic_evidence = (df['retail_panic_surrender_index_D'] > rp_quantile).astype(int)
            price_suppression_evidence = pd.Series(0, index=df.index, dtype=int)
            high_20_max = df['high_D'].rolling(20).max()
            price_suppression_evidence = ((df['high_D'] < high_20_max * 0.98) & (df['pct_change_D'].abs() < 0.02)).astype(int)
            chip_concentration_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'dominant_peak_solidity_D' in df.columns:
                dp_threshold = dynamic_thresholds.get('dominant_peak_solidity_D', pd.Series(0.3, index=df.index))
                chip_concentration_evidence = (df['dominant_peak_solidity_D'] > dp_threshold).astype(int)
            accumulation_score = (hidden_buy_evidence * 2.0 + retail_panic_evidence * 1.5 + price_suppression_evidence + chip_concentration_evidence)
            df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & (accumulation_score >= 2.5)
            print(f"吸筹信号: {df['IS_ACCUMULATION_D'].sum()}次")
            momentum_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'pct_change_D' in df.columns:
                pct_quantile = df['pct_change_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
                momentum_break_evidence = (df['pct_change_D'] > pct_quantile).astype(int)
            volume_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                volume_break_evidence = (df['volume_D'] > df['VOL_MA_21_D'] * 1.5).astype(int)
            fund_flow_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if all(col in df.columns for col in ['main_force_net_flow_calibrated_D', 'main_force_flow_directionality_D']):
                mf_threshold = dynamic_thresholds.get('main_force_net_flow_calibrated_D', pd.Series(0, index=df.index))
                fund_flow_break_evidence = ((df['main_force_net_flow_calibrated_D'] > mf_threshold) & (df['main_force_flow_directionality_D'] > 0.7)).astype(int)
            structure_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if all(col in df.columns for col in ['platform_conviction_score_D', 'trend_conviction_score_D']):
                pc_threshold = dynamic_thresholds.get('platform_conviction_score_D', pd.Series(70, index=df.index))
                tc_threshold = dynamic_thresholds.get('trend_conviction_score_D', pd.Series(70, index=df.index))
                structure_break_evidence = ((df['platform_conviction_score_D'] > pc_threshold) & (df['trend_conviction_score_D'] > tc_threshold)).astype(int)
            breakout_score = (momentum_break_evidence * 2.0 + volume_break_evidence * 1.5 + fund_flow_break_evidence * 1.2 + structure_break_evidence)
            df['IS_BREAKOUT_D'] = (breakout_score >= 5) & (momentum_break_evidence.astype(bool) | structure_break_evidence.astype(bool))
            print(f"突破信号: {df['IS_BREAKOUT_D'].sum()}次")
            main_force_dist_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'main_force_net_flow_calibrated_D' in df.columns:
                mf_low_quantile = df['main_force_net_flow_calibrated_D'].rolling(rolling_window, min_periods=20).quantile(0.3).fillna(method='ffill')
                main_force_dist_evidence = (df['main_force_net_flow_calibrated_D'] < mf_low_quantile).astype(int)
            retail_fomo_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'retail_fomo_premium_index_D' in df.columns:
                rf_quantile = df['retail_fomo_premium_index_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
                retail_fomo_evidence = (df['retail_fomo_premium_index_D'] > rf_quantile).astype(int)
            price_divergence_evidence = pd.Series(0, index=df.index, dtype=int)
            close_20_max = df['close_D'].rolling(20).max()
            mf_5ma = df['main_force_net_flow_calibrated_D'].rolling(5).mean()
            mf_20ma = df['main_force_net_flow_calibrated_D'].rolling(20).mean()
            price_divergence_evidence = ((df['close_D'] > close_20_max) & (mf_5ma < mf_20ma)).astype(int)
            chip_dispersion_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'chip_health_score_D' in df.columns:
                ch_threshold = dynamic_thresholds.get('chip_health_score_D', pd.Series(40, index=df.index))
                chip_dispersion_evidence = ((df['chip_health_score_D'] < ch_threshold) & (df['pct_change_D'] > 0.05)).astype(int)
            distribution_score = (main_force_dist_evidence * 1.5 + retail_fomo_evidence * 1.2 + price_divergence_evidence + chip_dispersion_evidence)
            df['IS_DISTRIBUTION_D'] = distribution_score >= 3
            print(f"派发信号: {df['IS_DISTRIBUTION_D'].sum()}次")
        print(f"=== 信号后处理 ===")
        def print_signal_dates(signal_name, signal_series, n=5):
            signal_dates = signal_series[signal_series].index
            if len(signal_dates) > 0:
                print(f"  {signal_name}触发日期({len(signal_dates)}次):")
                for i, date in enumerate(signal_dates[-n:]):
                    print(f"    {i+1}. {date.strftime('%Y-%m-%d')}")
                if len(signal_dates) > n:
                    print(f"    ... 还有{len(signal_dates)-n}个更早的触发")
            else:
                print(f"  {signal_name}从未触发")
        if market_phase == 'TRENDING':
            print("趋势市场信号去抖动...")
            print("原始信号触发日期:")
            print_signal_dates("IS_TREND_CONTINUATION_D(原始)", df['IS_TREND_CONTINUATION_D'])
            print_signal_dates("IS_TREND_CORRECTION_D(原始)", df['IS_TREND_CORRECTION_D'])
            print_signal_dates("IS_TREND_REVERSAL_D(原始)", df['IS_TREND_REVERSAL_D'])
            
            # 趋势延续信号去抖动：使用3日窗口，需要至少有2天触发
            if 'IS_TREND_CONTINUATION_D' in df.columns:
                df['IS_TREND_CONTINUATION_D'] = df['IS_TREND_CONTINUATION_D'].fillna(False).astype(bool)
                try:
                    signal_series = df['IS_TREND_CONTINUATION_D'].astype(float)
                    df['IS_TREND_CONTINUATION_D'] = (signal_series.rolling(3, min_periods=2).sum() >= 2).fillna(False).astype(bool)
                except Exception as e:
                    print(f"趋势延续信号去抖动失败: {e}")
            
            # 趋势回调信号去抖动：使用2日窗口，需要至少有1天触发，且避免信号扩散
            if 'IS_TREND_CORRECTION_D' in df.columns:
                df['IS_TREND_CORRECTION_D'] = df['IS_TREND_CORRECTION_D'].fillna(False).astype(bool)
                try:
                    signal_series = df['IS_TREND_CORRECTION_D'].astype(float)
                    # 使用2日窗口，至少1天触发，同时避免连续多日触发导致信号扩散
                    df['IS_TREND_CORRECTION_D'] = (signal_series.rolling(2, min_periods=1).sum() >= 1).fillna(False).astype(bool)
                    # 如果连续多日触发，只保留第一天
                    correction_mask = df['IS_TREND_CORRECTION_D']
                    continuous_correction = (correction_mask & correction_mask.shift(1))
                    df.loc[continuous_correction, 'IS_TREND_CORRECTION_D'] = False
                    print(f"趋势回调信号去抖动：去抖动后{df['IS_TREND_CORRECTION_D'].sum()}次")
                except Exception as e:
                    print(f"趋势回调信号去抖动失败: {e}")
            
            # 趋势反转信号去抖动：使用3日窗口，需要至少有2天触发
            if 'IS_TREND_REVERSAL_D' in df.columns:
                df['IS_TREND_REVERSAL_D'] = df['IS_TREND_REVERSAL_D'].fillna(False).astype(bool)
                try:
                    signal_series = df['IS_TREND_REVERSAL_D'].astype(float)
                    df['IS_TREND_REVERSAL_D'] = (signal_series.rolling(3, min_periods=2).sum() >= 2).fillna(False).astype(bool)
                except Exception as e:
                    print(f"趋势反转信号去抖动失败: {e}")
            
            print("去抖动后信号触发日期:")
            print_signal_dates("IS_TREND_CONTINUATION_D(去抖动后)", df['IS_TREND_CONTINUATION_D'])
            print_signal_dates("IS_TREND_CORRECTION_D(去抖动后)", df['IS_TREND_CORRECTION_D'])
            print_signal_dates("IS_TREND_REVERSAL_D(去抖动后)", df['IS_TREND_REVERSAL_D'])
        else:
            print("盘整市场信号去抖动...")
            signal_cols = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']
            for col in signal_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(False).astype(bool)
                    try:
                        signal_series = df[col].astype(float)
                        df[col] = (signal_series.rolling(3, min_periods=2).sum() >= 2).fillna(False).astype(bool)
                    except Exception as e:
                        print(f"信号去抖动失败: {col}, 错误: {e}")
        print(f"=== 信号互斥处理 ===")
        try:
            if market_phase == 'TRENDING':
                # 趋势延续时关闭趋势反转信号
                if 'IS_TREND_CONTINUATION_D' in df.columns and 'IS_TREND_REVERSAL_D' in df.columns:
                    continuation_mask = df['IS_TREND_CONTINUATION_D'].astype(bool)
                    if continuation_mask.any():
                        df.loc[continuation_mask, 'IS_TREND_REVERSAL_D'] = False
                        print(f"趋势延续时关闭趋势反转信号，影响{continuation_mask.sum()}个数据点")
                
                # 趋势回调时关闭趋势反转信号
                if 'IS_TREND_CORRECTION_D' in df.columns and 'IS_TREND_REVERSAL_D' in df.columns:
                    correction_mask = df['IS_TREND_CORRECTION_D'].astype(bool)
                    if correction_mask.any():
                        df.loc[correction_mask, 'IS_TREND_REVERSAL_D'] = False
                        print(f"趋势回调时关闭趋势反转信号，影响{correction_mask.sum()}个数据点")
                
                # 趋势反转时关闭趋势回调信号
                if 'IS_TREND_REVERSAL_D' in df.columns and 'IS_TREND_CORRECTION_D' in df.columns:
                    reversal_mask = df['IS_TREND_REVERSAL_D'].astype(bool)
                    if reversal_mask.any():
                        df.loc[reversal_mask, 'IS_TREND_CORRECTION_D'] = False
                        print(f"趋势反转时关闭趋势回调信号，影响{reversal_mask.sum()}个数据点")
            else:
                if 'IS_BREAKOUT_D' in df.columns and 'IS_ACCUMULATION_D' in df.columns:
                    breakout_mask = df['IS_BREAKOUT_D'].astype(bool)
                    if breakout_mask.any():
                        df.loc[breakout_mask, 'IS_ACCUMULATION_D'] = False
                        print(f"突破时关闭吸筹信号，影响{breakout_mask.sum()}个数据点")
                if 'IS_DISTRIBUTION_D' in df.columns and 'IS_BREAKOUT_D' in df.columns:
                    distribution_mask = df['IS_DISTRIBUTION_D'].astype(bool)
                    if distribution_mask.any():
                        df.loc[distribution_mask, 'IS_BREAKOUT_D'] = False
                        print(f"派发时关闭突破信号，影响{distribution_mask.sum()}个数据点")
        except Exception as e:
            print(f"模式互斥性处理失败: {e}")
        df['MARKET_PHASE_D'] = market_phase
        print(f"=== 分析完成 ===")
        print(f"市场阶段: {market_phase}")
        if market_phase == 'TRENDING':
            signal_cols = ['IS_TREND_CONTINUATION_D', 'IS_TREND_CORRECTION_D', 'IS_TREND_REVERSAL_D']
            for col in signal_cols:
                if col in df.columns:
                    print(f"{col}: 总触发={df[col].sum()}, 最新信号={df[col].iloc[-1]}, 最新日期={df.index[-1].strftime('%Y-%m-%d')}")
            if 'IS_TREND_CORRECTION_RAW_D' in df.columns:
                print(f"IS_TREND_CORRECTION_RAW_D: 总触发={df['IS_TREND_CORRECTION_RAW_D'].sum()}, 最新信号={df['IS_TREND_CORRECTION_RAW_D'].iloc[-1]}")
        else:
            signal_cols = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']
            for col in signal_cols:
                if col in df.columns:
                    print(f"{col}: 总触发={df[col].sum()}, 最新信号={df[col].iloc[-1]}, 最新日期={df.index[-1].strftime('%Y-%m-%d')}")
        all_dfs[timeframe] = df
        print("=== 高级模式识别引擎(V5.3 信号探针增强版)分析完成 ===")
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





