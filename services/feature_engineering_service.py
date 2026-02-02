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
    """V2版向量化计算回调信号评分 - 使用新模型字段"""
    n = len(pct_change)
    scores = np.zeros(n, dtype=np.float32)
    is_correction = np.zeros(n, dtype=np.bool_)
    
    # 计算技术指标（向量化）
    # 均线计算
    ma5 = np.zeros(n, dtype=np.float32)
    ma10 = np.zeros(n, dtype=np.float32)
    ma20 = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i >= 4:
            ma5[i] = np.mean(close[max(0, i-4):i+1])
        if i >= 9:
            ma10[i] = np.mean(close[max(0, i-9):i+1])
        if i >= 19:
            ma20[i] = np.mean(close[max(0, i-19):i+1])
    
    # 前3日平均涨幅
    avg_gain_3d = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        if i >= 3:
            avg_gain_3d[i] = np.mean(pct_change[max(0, i-3):i])
    
    # 资金流统计
    flow_median = np.zeros(n, dtype=np.float32)
    flow_std = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i >= 19:
            window = flow_intensity[max(0, i-19):i+1]
            flow_median[i] = np.median(window)
            flow_std[i] = np.std(window)
    
    # 筹码稳定性统计
    chip_stab_median = np.zeros(n, dtype=np.float32)
    chip_stab_std = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i >= 19:
            window = chip_stability[max(0, i-19):i+1]
            chip_stab_median[i] = np.median(window)
            chip_stab_std[i] = np.std(window)
    
    # 长线筹码比例统计
    ltr_median = np.zeros(n, dtype=np.float32)
    ltr_std = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if i >= 19:
            window = long_term_ratio[max(0, i-19):i+1]
            ltr_median[i] = np.median(window)
            ltr_std[i] = np.std(window)
    
    # 主循环：计算每个数据点的分数
    for i in range(n):
        # 条件1: 回调幅度条件（必要条件）
        cond1 = (pct_change[i] < 0) and (pct_change[i] >= correction_max_magnitude) and (pct_change[i] <= correction_min_magnitude)
        if not cond1:
            continue
        # 其他条件（使用新模型字段）
        cond2 = volume[i] > vol_ma21[i] * 0.8 if vol_ma21[i] > 0 else True  # 成交量支持
        cond3 = (close[i] >= ma5[i] * 0.95) and (ma5[i] > ma10[i] * 0.96) if i >= 9 else True  # 趋势支撑
        cond4 = ma5[i] > ma20[i] * 0.96 if i >= 19 else True  # 趋势方向
        cond5 = flow_intensity[i] > (flow_median[i] - flow_std[i] * 0.5) if i >= 19 else True  # 资金流条件
        cond6 = chip_stability[i] > (chip_stab_median[i] - chip_stab_std[i] * 0.3) if i >= 19 else True  # 筹码稳定性条件
        cond7 = long_term_ratio[i] > (ltr_median[i] - ltr_std[i] * 0.5) if i >= 19 else True  # 长线筹码条件
        cond8 = avg_gain_3d[i] > 0.01 if i >= 3 else True  # 前期上涨趋势
        cond9 = close[i] > ma10[i] * 0.97 if i >= 9 else True  # 不破10日线
        cond10 = volume[i] <= vol_ma21[i] * 1.3 if vol_ma21[i] > 0 else True  # 正常成交量
        # 计算分数（新权重分配）
        score = 0.0
        score += cond1 * 2.5  # 回调幅度条件最重要
        score += cond2 * 1.0  # 成交量支持
        score += cond3 * 1.2  # 趋势支撑
        score += cond4 * 1.0  # 趋势方向
        score += cond5 * 1.5  # 资金流条件（新）
        score += cond6 * 1.3  # 筹码稳定性条件（新）
        score += cond7 * 1.2  # 长线筹码条件（新）
        score += cond8 * 1.0  # 前期上涨趋势
        score += cond9 * 1.0  # 不破10日线
        score += cond10 * 0.8  # 正常成交量
        scores[i] = score
        is_correction[i] = score >= 8.0 and cond1
    
    return is_correction

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
        【V2.0 · 多因子融合版】VPA效率指标生产线重构
        使用FundFlowFactor、ChipFactorBase、ChipHoldingMatrixBase的先进指标替代废弃的幻方指标
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
        # ==================== 1. 资金流向增强版VPA ====================
        # 1.1 总VPA效率（保留经典算法）
        volume_ratio = df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)
        vpa_efficiency = df['pct_change_D'] / volume_ratio.replace(0, np.nan)
        df['VPA_EFFICIENCY_D'] = vpa_efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
        # 1.2 使用资金流向因子替代幻方指标
        # 检查是否有资金流向因子数据
        has_fundflow_data = all(col in df.columns for col in [
            'flow_intensity_D', 'net_amount_ratio_D', 'large_order_anomaly_D'
        ])
        if has_fundflow_data:
            # ==================== 2. 多维度VPA效率计算 ====================
            # 2.1 主力行为增强VPA
            # 使用行为模式得分构建多空效率
            if all(col in df.columns for col in ['accumulation_score_D', 'distribution_score_D']):
                # 吸筹模式下的买入效率
                accumulation_weight = df['accumulation_score_D'].fillna(0) / 100.0
                positive_change_accum = np.maximum(df['pct_change_D'].values, 0)
                buy_efficiency_accum = positive_change_accum / volume_ratio.replace(0, np.nan)
                df['VPA_BUY_ACCUM_EFF_D'] = buy_efficiency_accum * accumulation_weight
                # 派发模式下的卖出效率
                distribution_weight = df['distribution_score_D'].fillna(0) / 100.0
                negative_change_dist = np.minimum(df['pct_change_D'].values, 0)
                sell_efficiency_dist = negative_change_dist / volume_ratio.replace(0, np.nan)
                df['VPA_SELL_DIST_EFF_D'] = sell_efficiency_dist * distribution_weight
            # 2.2 大单异动增强VPA
            if 'large_order_anomaly_D' in df.columns and 'anomaly_intensity_D' in df.columns:
                anomaly_mask = df['large_order_anomaly_D'] == True
                anomaly_weight = df['anomaly_intensity_D'].fillna(0) / 100.0
                # 大单异动期间的VPA效率（更具信号意义）
                anomaly_vpa = df['VPA_EFFICIENCY_D'].copy()
                anomaly_vpa = anomaly_vpa * (1 + anomaly_weight)
                df['VPA_ANOMALY_ENHANCED_D'] = np.where(anomaly_mask, anomaly_vpa, df['VPA_EFFICIENCY_D'])
            # 2.3 净流入占比调整的VPA
            if 'net_amount_ratio_D' in df.columns:
                # 资金流入强度调整的VPA
                net_ratio_adjustment = df['net_amount_ratio_D'].fillna(0) / 100.0
                # 正净流入增强买入效率，负净流入增强卖出效率
                buy_efficiency_adj = np.maximum(df['pct_change_D'].values, 0) / volume_ratio.replace(0, np.nan)
                sell_efficiency_adj = np.minimum(df['pct_change_D'].values, 0) / volume_ratio.replace(0, np.nan)
                df['VPA_BUY_NETFLOW_ADJ_D'] = buy_efficiency_adj * (1 + np.maximum(net_ratio_adjustment.values, 0))
                df['VPA_SELL_NETFLOW_ADJ_D'] = sell_efficiency_adj * (1 + np.abs(np.minimum(net_ratio_adjustment.values, 0)))
            # 2.4 资金流向稳定性调整VPA
            if 'flow_stability_D' in df.columns:
                stability_weight = df['flow_stability_D'].fillna(50) / 100.0
                # 稳定性越高，VPA信号越可靠
                df['VPA_STABILITY_WEIGHTED_D'] = df['VPA_EFFICIENCY_D'] * stability_weight
        # ==================== 3. 筹码结构增强版VPA ====================
        # 检查是否有筹码因子数据
        has_chip_data = all(col in df.columns for col in [
            'chip_concentration_ratio_D', 'winner_rate_D', 'turnover_rate_D'
        ])
        if has_chip_data:
            # 3.1 筹码集中度调整的VPA
            # 集中度越高，VPA效率信号越强
            concentration_weight = df['chip_concentration_ratio_D'].fillna(0.5)
            df['VPA_CONCENTRATION_ADJ_D'] = df['VPA_EFFICIENCY_D'] * concentration_weight
            # 3.2 获利盘压力调整的VPA
            if 'profit_pressure_D' in df.columns:
                # 获利压力越大，上涨需要的成交量越大
                pressure_adjustment = 1 / (1 + df['profit_pressure_D'].fillna(0).abs())
                df['VPA_PROFIT_PRESSURE_ADJ_D'] = df['VPA_EFFICIENCY_D'] * pressure_adjustment
            # 3.3 换手率调整的VPA
            # 换手率过高可能降低VPA效率信号质量
            turnover_rate_norm = df['turnover_rate_D'].fillna(0) / 100.0
            turnover_penalty = 1 / (1 + turnover_rate_norm * 5)  # 换手率过高时惩罚
            df['VPA_TURNOVER_ADJ_D'] = df['VPA_EFFICIENCY_D'] * turnover_penalty
            # 3.4 筹码峰动态调整
            if 'peak_migration_speed_5d' in df.columns:
                # 筹码峰快速迁移时，VPA效率需要调整
                migration_speed = df['peak_migration_speed_5d'].fillna(0)
                migration_factor = 1 / (1 + np.abs(migration_speed) * 0.1)
                df['VPA_MIGRATION_ADJ_D'] = df['VPA_EFFICIENCY_D'] * migration_factor
        # ==================== 4. 筹码持有时间矩阵增强版VPA ====================
        # 检查是否有筹码矩阵数据
        has_matrix_data = all(col in df.columns for col in [
            'short_term_ratio_D', 'long_term_ratio_D', 'absorption_energy_D', 'distribution_energy_D'
        ])
        if has_matrix_data:
            # 4.1 短线筹码比例调整
            # 短线筹码比例高时，VPA效率可能被高估
            short_term_ratio = df['short_term_ratio_D'].fillna(0.2)
            short_term_penalty = 1 - short_term_ratio * 0.5  # 短线筹码比例每增加10%，效率降低5%
            df['VPA_SHORT_TERM_ADJ_D'] = df['VPA_EFFICIENCY_D'] * short_term_penalty
            # 4.2 长线筹码锁定调整
            # 长线筹码比例高时，VPA效率更可靠
            long_term_ratio = df['long_term_ratio_D'].fillna(0.5)
            long_term_boost = 1 + long_term_ratio * 0.3  # 长线筹码比例每增加10%，效率可信度提高3%
            df['VPA_LONG_TERM_ADJ_D'] = df['VPA_EFFICIENCY_D'] * long_term_boost
            # 4.3 博弈能量场调整VPA
            absorption_energy = df['absorption_energy_D'].fillna(0) / 100.0
            distribution_energy = df['distribution_energy_D'].fillna(0) / 100.0
            # 吸收能量增强买入效率，派发能量增强卖出效率
            buy_efficiency = np.maximum(df['pct_change_D'].values, 0) / volume_ratio.replace(0, np.nan)
            sell_efficiency = np.minimum(df['pct_change_D'].values, 0) / volume_ratio.replace(0, np.nan)
            df['VPA_ABSORPTION_ENHANCED_D'] = buy_efficiency * (1 + absorption_energy)
            df['VPA_DISTRIBUTION_ENHANCED_D'] = sell_efficiency * (1 + distribution_energy)
            # 4.4 净能量流向调整
            if 'net_energy_flow' in df.columns:
                net_energy = df['net_energy_flow_D'].fillna(0) / 100.0
                df['VPA_NET_ENERGY_ADJ_D'] = df['VPA_EFFICIENCY_D'] * (1 + net_energy * 0.5)
        # ==================== 5. 复合VPA综合指标 ====================
        # 5.1 综合VPA评分（融合所有调整因子）
        vpa_columns = [
            'VPA_EFFICIENCY_D',
            'VPA_CONCENTRATION_ADJ_D' if 'VPA_CONCENTRATION_ADJ_D' in df.columns else None,
            'VPA_STABILITY_WEIGHTED_D' if 'VPA_STABILITY_WEIGHTED_D' in df.columns else None,
            'VPA_LONG_TERM_ADJ_D' if 'VPA_LONG_TERM_ADJ_D' in df.columns else None,
            'VPA_NET_ENERGY_ADJ_D' if 'VPA_NET_ENERGY_ADJ_D' in df.columns else None,
        ]
        vpa_columns = [col for col in vpa_columns if col is not None]
        if len(vpa_columns) > 1:
            # 标准化处理
            vpa_matrix = df[vpa_columns].fillna(0).values
            # 计算综合评分（等权平均）
            df['VPA_COMPOSITE_SCORE_D'] = np.mean(vpa_matrix, axis=1)
            # 5.2 VPA效率分级
            conditions = [
                df['VPA_COMPOSITE_SCORE_D'] > 2.0,
                df['VPA_COMPOSITE_SCORE_D'] > 1.0,
                df['VPA_COMPOSITE_SCORE_D'] > 0,
                df['VPA_COMPOSITE_SCORE_D'] > -1.0,
                df['VPA_COMPOSITE_SCORE_D'] > -2.0,
            ]
            choices = [4, 3, 2, 1, 0]  # 4:极高效率, 3:高效率, 2:中等效率, 1:低效率, 0:负效率
            df['VPA_EFFICIENCY_LEVEL_D'] = np.select(conditions, choices, default=0)
            # 5.3 VPA背离检测
            # 价格与VPA效率的背离
            price_trend = df['pct_change_D'].rolling(window=5, min_periods=3).mean()
            vpa_trend = df['VPA_COMPOSITE_SCORE_D'].rolling(window=5, min_periods=3).mean()
            # 看多背离：价格下跌但VPA效率上升
            bullish_divergence = (price_trend < 0) & (vpa_trend > 0)
            # 看空背离：价格上涨但VPA效率下降
            bearish_divergence = (price_trend > 0) & (vpa_trend < 0)
            df['VPA_BULLISH_DIVERGENCE_D'] = bullish_divergence.astype(int)
            df['VPA_BEARISH_DIVERGENCE_D'] = bearish_divergence.astype(int)
            # 5.4 VPA动量指标
            df['VPA_MOMENTUM_5D'] = df['VPA_COMPOSITE_SCORE_D'].diff(5)
            df['VPA_MOMENTUM_10D'] = df['VPA_COMPOSITE_SCORE_D'].diff(10)
        # ==================== 6. 信号质量评估 ====================
        # 6.1 VPA信号置信度
        confidence_factors = []
        if 'flow_stability_D' in df.columns:
            confidence_factors.append(df['flow_stability_D'].fillna(50) / 100.0)
        if 'chip_stability_D' in df.columns:
            confidence_factors.append(df['chip_stability_D'].fillna(0.5))
        if 'validation_score_D' in df.columns:
            confidence_factors.append(df['validation_score_D'].fillna(0.5))
        if confidence_factors:
            df['VPA_SIGNAL_CONFIDENCE_D'] = np.mean(confidence_factors, axis=0)
        # 6.2 VPA交易信号生成
        if 'VPA_COMPOSITE_SCORE_D' in df.columns and 'VPA_MOMENTUM_5D' in df.columns:
            conditions = [
                (df['VPA_COMPOSITE_SCORE_D'] > 1.5) & (df['VPA_MOMENTUM_5D'] > 0.1),
                (df['VPA_COMPOSITE_SCORE_D'] > 0.5) & (df['VPA_MOMENTUM_5D'] > 0),
                (df['VPA_COMPOSITE_SCORE_D'] < -0.5) & (df['VPA_MOMENTUM_5D'] < 0),
                (df['VPA_COMPOSITE_SCORE_D'] < -1.5) & (df['VPA_MOMENTUM_5D'] < -0.1),
            ]
            choices = [2, 1, -1, -2]  # 2:强买入, 1:买入, -1:卖出, -2:强卖出
            df['VPA_TRADING_SIGNAL_D'] = np.select(conditions, choices, default=0)
            # 信号强度
            df['VPA_SIGNAL_STRENGTH_D'] = np.abs(df['VPA_TRADING_SIGNAL_D'])
        # 替换缺失值为0
        vpa_columns_all = [col for col in df.columns if col.startswith('VPA_')]
        df[vpa_columns_all] = df[vpa_columns_all].fillna(0)
        all_dfs[timeframe] = df
        return all_dfs

    async def calculate_pattern_recognition_signals(self, all_dfs: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
        """【V6.0 · 多因子融合版】基于市场阶段的多模式识别系统 - 使用新模型字段重构"""
        timeframe = 'D'
        if timeframe not in all_dfs:
            return all_dfs
        df = all_dfs[timeframe].copy()
        print(f"=== 模式识别引擎(V6.0 多因子融合版)开始分析，数据长度: {len(df)} ===")
        # ==================== 1. 市场阶段判断 ====================
        # 使用资金流稳定性判断市场阶段
        market_phase = 'CONSOLIDATING'  # 默认
        if 'flow_stability_D' in df.columns and 'ADX_14_D' in df.columns:
            # 双重确认市场阶段
            current_adx = df['ADX_14_D'].iloc[-1]
            current_flow_stability = df['flow_stability_D'].iloc[-1] if not pd.isna(df['flow_stability_D'].iloc[-1]) else 50
            # 资金流稳定且ADX>25：趋势市场
            # 资金流不稳定或ADX<25：震荡市场
            if current_adx > 25 and current_flow_stability > 60:
                market_phase = 'TRENDING'
                print(f"市场阶段: TRENDING (ADX={current_adx:.1f}, 资金流稳定性={current_flow_stability:.1f})")
            else:
                market_phase = 'CONSOLIDATING'
                print(f"市场阶段: CONSOLIDATING (ADX={current_adx:.1f}, 资金流稳定性={current_flow_stability:.1f})")
        elif 'ADX_14_D' in df.columns:
            current_adx = df['ADX_14_D'].iloc[-1]
            market_phase = 'TRENDING' if current_adx > 25 else 'CONSOLIDATING'
            print(f"市场阶段: {market_phase}, ADX: {current_adx:.1f}")
        else:
            print("警告：缺少ADX数据，使用默认震荡市场阶段")
        # 处理百分比变化数据
        if 'pct_change_D' in df.columns and (df['pct_change_D'].max() > 10 or df['pct_change_D'].min() < -10):
            df['pct_change_D'] = df['pct_change_D'] / 100.0
        # ==================== 2. 动态阈值计算 ====================
        rolling_window = min(120, len(df))
        if rolling_window < 30:
            return all_dfs
        dynamic_thresholds = {}
        threshold_config = {}
        if market_phase == 'TRENDING':
            # 趋势市场：更严格的阈值，侧重趋势延续和回调识别
            threshold_config = {
                # 筹码健康度替代指标：筹码稳定性 + 资金流稳定性
                'chip_stability_D': ('q30', 0.3),
                'flow_stability_D': ('q40', 0.4),
                # 主力行为替代指标：使用行为模式得分
                'accumulation_score_D': ('q50', 0.5),
                'distribution_score_D': ('q30', 0.3),
                # 资金流强度替代指标
                'flow_intensity_D': ('q40', 0.4),
                'net_amount_ratio_D': ('q50', 0.5),
                # 筹码集中度替代指标
                'chip_concentration_ratio_D': ('q60', 0.6),
                'concentration_comprehensive_D': ('q50', 0.5),
                # 博弈能量场指标
                'absorption_energy_D': ('q60', 0.6),
                'distribution_energy_D': ('q30', 0.3),
                # 基础价格指标
                'pct_change_D': ('q40', 0.4),
                # 新增：趋势强度指标
                'uptrend_strength_D': ('q60', 0.6),
                'downtrend_strength_D': ('q30', 0.3),
                # 新增：筹码流动指标
                'chip_flow_intensity_D': ('q50', 0.5),
                'long_term_chip_ratio_D': ('q60', 0.6),
            }
        else:
            # 震荡市场：更宽松的阈值，侧重突破和吸筹识别
            threshold_config = {
                'chip_stability_D': ('q40', 0.4),
                'flow_stability_D': ('q30', 0.3),
                'accumulation_score_D': ('q60', 0.6),
                'distribution_score_D': ('q40', 0.4),
                'flow_intensity_D': ('q50', 0.5),
                'net_amount_ratio_D': ('q60', 0.6),
                'chip_concentration_ratio_D': ('q70', 0.7),
                'concentration_comprehensive_D': ('q60', 0.6),
                'absorption_energy_D': ('q70', 0.7),
                'distribution_energy_D': ('q40', 0.4),
                'pct_change_D': ('q60', 0.6),
                'uptrend_strength_D': ('q70', 0.7),
                'downtrend_strength_D': ('q40', 0.4),
                'chip_flow_intensity_D': ('q60', 0.6),
                'long_term_chip_ratio_D': ('q70', 0.7),
            }
        # 计算动态阈值
        for col, (method, param) in threshold_config.items():
            if col in df.columns and not df[col].isna().all():
                try:
                    threshold_series = df[col].rolling(rolling_window, min_periods=20).quantile(param)
                    dynamic_thresholds[col] = threshold_series.fillna(method='ffill')
                except:
                    # 设置默认值
                    if col in ['chip_stability_D', 'flow_stability_D']:
                        dynamic_thresholds[col] = pd.Series(0.5, index=df.index)
                    elif col in ['accumulation_score_D', 'distribution_score_D']:
                        dynamic_thresholds[col] = pd.Series(50, index=df.index)
                    elif col in ['flow_intensity_D', 'net_amount_ratio_D']:
                        dynamic_thresholds[col] = pd.Series(0, index=df.index)
                    elif 'concentration' in col:
                        dynamic_thresholds[col] = pd.Series(0.5, index=df.index)
                    elif 'energy' in col:
                        dynamic_thresholds[col] = pd.Series(50, index=df.index)
                    else:
                        dynamic_thresholds[col] = pd.Series(0, index=df.index)
        # ==================== 3. 趋势市场信号识别 ====================
        if market_phase == 'TRENDING':
            df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = False
            df['IS_ACCUMULATION_D'] = False
            # 3.1 趋势延续信号（使用多因子融合）
            trend_continuation_score = pd.Series(0, index=df.index, dtype=float)
            # 价格动量证据
            if 'pct_change_D' in df.columns:
                pct_threshold = dynamic_thresholds.get('pct_change_D', pd.Series(0, index=df.index))
                trend_continuation_score = trend_continuation_score + ((df['pct_change_D'] > pct_threshold) * 2.0).fillna(0)
            # 成交量证据
            if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                trend_continuation_score = trend_continuation_score + ((df['volume_D'] > df['VOL_MA_21_D'] * 1.2) * 1.5).fillna(0)
            # 资金流强度证据
            if 'flow_intensity_D' in df.columns:
                flow_threshold = dynamic_thresholds.get('flow_intensity_D', pd.Series(0, index=df.index))
                trend_continuation_score = trend_continuation_score + ((df['flow_intensity_D'] > flow_threshold) * 1.5).fillna(0)
            # 筹码稳定性证据
            if 'chip_stability_D' in df.columns:
                chip_stab_threshold = dynamic_thresholds.get('chip_stability_D', pd.Series(0.5, index=df.index))
                trend_continuation_score = trend_continuation_score + ((df['chip_stability_D'] > chip_stab_threshold) * 1.2).fillna(0)
            # 趋势强度证据
            if 'uptrend_strength_D' in df.columns:
                uptrend_threshold = dynamic_thresholds.get('uptrend_strength_D', pd.Series(60, index=df.index))
                trend_continuation_score = trend_continuation_score + ((df['uptrend_strength_D'] > uptrend_threshold) * 1.5).fillna(0)
            df['IS_TREND_CONTINUATION_D'] = (trend_continuation_score >= 6.0)
            # 3.2 趋势回调信号（使用多因子识别健康回调）
            if 'pct_change_D' in df.columns:
                print("=== 回调信号计算（多因子融合版） ===")
                correction_min_magnitude = -0.008
                correction_max_magnitude = -0.05
                print(f"回调幅度允许范围: [{correction_max_magnitude:.4f}, {correction_min_magnitude:.4f}]")
                # 准备数据 - 使用新模型字段
                required_cols = ['pct_change_D', 'close_D', 'volume_D', 'VOL_MA_21_D', 
                               'flow_intensity_D', 'chip_stability_D', 'long_term_chip_ratio_D']
                if all(col in df.columns for col in required_cols):
                    # 准备数据，填充NaN值
                    pct_change = df['pct_change_D'].fillna(0).values.astype(np.float32)
                    close = df['close_D'].fillna(method='ffill').fillna(method='bfill').values.astype(np.float32)
                    volume = df['volume_D'].fillna(0).values.astype(np.float32)
                    vol_ma21 = df['VOL_MA_21_D'].fillna(0).values.astype(np.float32)
                    flow_intensity = df['flow_intensity_D'].fillna(0).values.astype(np.float32)
                    chip_stability = df['chip_stability_D'].fillna(0.5).values.astype(np.float32)
                    long_term_ratio = df['long_term_chip_ratio_D'].fillna(0.5).values.astype(np.float32)
                    
                    # 使用新版本的numba函数
                    is_correction_numba = calculate_correction_scores_v2_numba(
                        pct_change, close, volume, vol_ma21, flow_intensity, 
                        chip_stability, long_term_ratio,
                        correction_min_magnitude, correction_max_magnitude
                    )
                    
                    df['IS_TREND_CORRECTION_D'] = pd.Series(is_correction_numba, index=df.index)
                    
                    # 与趋势延续信号互斥
                    if 'IS_TREND_CONTINUATION_D' in df.columns:
                        df['IS_TREND_CORRECTION_D'] = df['IS_TREND_CORRECTION_D'] & (~df['IS_TREND_CONTINUATION_D'])
                    
                    # 探针：检查最近的结果
                    if len(df) > 5:
                        last_5 = df.index[-5:]
                        for idx in last_5:
                            if idx in df.index:
                                pos = df.index.get_loc(idx)
                                print(f"回调信号 {idx}: {df['IS_TREND_CORRECTION_D'].iloc[pos]}")
                else:
                    df['IS_TREND_CORRECTION_D'] = False
            # 3.3 趋势反转信号（使用多因子识别危险信号）
            if 'pct_change_D' in df.columns:
                # 价格暴跌证据
                cond_sharp_drop = df['pct_change_D'] < -0.05
                # 成交量异常证据
                cond_volume_spike = (df['volume_D'] > df['VOL_MA_21_D'] * 1.5) if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns else pd.Series(False, index=df.index)
                # 资金大幅流出证据
                if 'net_amount_ratio_D' in df.columns:
                    net_ratio_low_q = df['net_amount_ratio_D'].rolling(20).quantile(0.2).fillna(method='ffill')
                    cond_heavy_outflow = (df['net_amount_ratio_D'] < net_ratio_low_q)
                else:
                    cond_heavy_outflow = pd.Series(False, index=df.index)
                # 筹码恶化证据
                if 'chip_stability_D' in df.columns:
                    chip_stab_low_q = df['chip_stability_D'].rolling(20).quantile(0.3).fillna(method='ffill')
                    cond_chip_deteriorate = (df['chip_stability_D'] < chip_stab_low_q)
                else:
                    cond_chip_deteriorate = pd.Series(False, index=df.index)
                # 派发模式证据
                if 'distribution_score_D' in df.columns:
                    dist_score_threshold = dynamic_thresholds.get('distribution_score_D', pd.Series(30, index=df.index))
                    cond_distribution_mode = (df['distribution_score_D'] > dist_score_threshold)
                else:
                    cond_distribution_mode = pd.Series(False, index=df.index)
                # 综合反转信号
                df['IS_TREND_REVERSAL_D'] = cond_sharp_drop & (
                    cond_volume_spike | cond_heavy_outflow | cond_distribution_mode
                ) & cond_chip_deteriorate
            print(f"趋势市场信号统计:")
            print(f"  趋势延续信号: {df['IS_TREND_CONTINUATION_D'].sum()}次")
            print(f"  趋势回调信号: {df['IS_TREND_CORRECTION_D'].sum() if 'IS_TREND_CORRECTION_D' in df.columns else 0}次")
            print(f"  趋势反转信号: {df['IS_TREND_REVERSAL_D'].sum() if 'IS_TREND_REVERSAL_D' in df.columns else 0}次")
        # ==================== 4. 震荡市场信号识别 ====================
        else:
            df['IS_TREND_CONTINUATION_D'] = False
            df['IS_TREND_CORRECTION_D'] = False
            df['IS_TREND_REVERSAL_D'] = False
            # 4.1 高潜力震荡识别（使用筹码和资金流稳定性）
            chip_stability_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'chip_stability_D' in df.columns:
                chip_stab_thresh = dynamic_thresholds.get('chip_stability_D', pd.Series(0.5, index=df.index))
                chip_stability_evidence = (df['chip_stability_D'] > chip_stab_thresh).astype(int)
            flow_stability_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'flow_stability_D' in df.columns:
                flow_stab_thresh = dynamic_thresholds.get('flow_stability_D', pd.Series(50, index=df.index))
                flow_stability_evidence = (df['flow_stability_D'] > flow_stab_thresh).astype(int)
            volatility_compression_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'BBW_21_2.0_D' in df.columns:
                bbw_quantile = df['BBW_21_2.0_D'].rolling(rolling_window, min_periods=20).quantile(0.3).fillna(method='ffill')
                volatility_compression_evidence = (df['BBW_21_2.0_D'] < bbw_quantile).astype(int)
            fund_flow_balance_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'net_amount_ratio_D' in df.columns:
                net_ratio_std = df['net_amount_ratio_D'].rolling(rolling_window, min_periods=20).std().fillna(0)
                fund_flow_balance_evidence = (abs(df['net_amount_ratio_D']) < net_ratio_std * 0.5).astype(int)
            consolidation_score = (chip_stability_evidence + flow_stability_evidence + 
                                  volatility_compression_evidence + fund_flow_balance_evidence)
            df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] = (consolidation_score >= 3) & (df['ADX_14_D'] < 25)
            # 4.2 吸筹信号识别（使用多因子融合）
            hidden_buy_evidence = pd.Series(0, index=df.index, dtype=int)
            if all(col in df.columns for col in ['accumulation_score_D', 'net_amount_ratio_D']):
                net_ratio_quantile = df['net_amount_ratio_D'].rolling(rolling_window, min_periods=20).quantile(0.5).fillna(method='ffill')
                hidden_buy_evidence = ((df['accumulation_score_D'] > 50) & (df['net_amount_ratio_D'] > net_ratio_quantile)).astype(int)
            absorption_energy_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'absorption_energy_D' in df.columns:
                absorb_threshold = dynamic_thresholds.get('absorption_energy_D', pd.Series(70, index=df.index))
                absorption_energy_evidence = (df['absorption_energy_D'] > absorb_threshold).astype(int)
            price_suppression_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'high_D' in df.columns:
                high_20_max = df['high_D'].rolling(20).max()
                price_suppression_evidence = ((df['high_D'] < high_20_max * 0.98) & 
                                              (df['pct_change_D'].abs() < 0.02) if 'pct_change_D' in df.columns else pd.Series(False, index=df.index)).astype(int)
            chip_concentration_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'concentration_comprehensive_D' in df.columns:
                conc_threshold = dynamic_thresholds.get('concentration_comprehensive_D', pd.Series(0.6, index=df.index))
                chip_concentration_evidence = (df['concentration_comprehensive_D'] > conc_threshold).astype(int)
            accumulation_score = (hidden_buy_evidence * 2.0 + absorption_energy_evidence * 1.5 + 
                                 price_suppression_evidence + chip_concentration_evidence)
            df['IS_ACCUMULATION_D'] = df['IS_HIGH_POTENTIAL_CONSOLIDATION_D'] & (accumulation_score >= 2.5)
            # 4.3 突破信号识别
            momentum_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'pct_change_D' in df.columns:
                pct_quantile = df['pct_change_D'].rolling(rolling_window, min_periods=20).quantile(0.6).fillna(method='ffill')
                momentum_break_evidence = (df['pct_change_D'] > pct_quantile).astype(int)
            volume_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                volume_break_evidence = (df['volume_D'] > df['VOL_MA_21_D'] * 1.5).astype(int)
            fund_flow_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if all(col in df.columns for col in ['net_amount_ratio_D', 'flow_intensity_D']):
                net_ratio_threshold = dynamic_thresholds.get('net_amount_ratio_D', pd.Series(0, index=df.index))
                fund_flow_break_evidence = ((df['net_amount_ratio_D'] > net_ratio_threshold) & 
                                           (df['flow_intensity_D'] > 60)).astype(int)
            energy_break_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'absorption_energy_D' in df.columns:
                absorb_break_threshold = dynamic_thresholds.get('absorption_energy_D', pd.Series(70, index=df.index)) * 1.2
                energy_break_evidence = (df['absorption_energy_D'] > absorb_break_threshold).astype(int)
            breakout_score = (momentum_break_evidence * 2.0 + volume_break_evidence * 1.5 + 
                             fund_flow_break_evidence * 1.2 + energy_break_evidence)
            df['IS_BREAKOUT_D'] = (breakout_score >= 5) & (momentum_break_evidence.astype(bool) | energy_break_evidence.astype(bool))
            # 4.4 派发信号识别
            distribution_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'distribution_score_D' in df.columns:
                dist_threshold = dynamic_thresholds.get('distribution_score_D', pd.Series(40, index=df.index))
                distribution_evidence = (df['distribution_score_D'] > dist_threshold).astype(int)
            distribution_energy_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'distribution_energy_D' in df.columns:
                dist_energy_threshold = dynamic_thresholds.get('distribution_energy_D', pd.Series(40, index=df.index))
                distribution_energy_evidence = (df['distribution_energy_D'] > dist_energy_threshold).astype(int)
            price_divergence_evidence = pd.Series(0, index=df.index, dtype=int)
            if all(col in df.columns for col in ['close_D', 'net_amount_ratio_D']):
                close_20_max = df['close_D'].rolling(20).max()
                net_ratio_5ma = df['net_amount_ratio_D'].rolling(5).mean()
                net_ratio_20ma = df['net_amount_ratio_D'].rolling(20).mean()
                price_divergence_evidence = ((df['close_D'] > close_20_max) & (net_ratio_5ma < net_ratio_20ma)).astype(int)
            chip_dispersion_evidence = pd.Series(0, index=df.index, dtype=int)
            if 'chip_stability_D' in df.columns and 'pct_change_D' in df.columns:
                chip_stab_threshold = dynamic_thresholds.get('chip_stability_D', pd.Series(0.5, index=df.index))
                chip_dispersion_evidence = ((df['chip_stability_D'] < chip_stab_threshold) & 
                                           (df['pct_change_D'] > 0.05)).astype(int)
            distribution_score = (distribution_evidence * 1.5 + distribution_energy_evidence * 1.2 + 
                                price_divergence_evidence + chip_dispersion_evidence)
            df['IS_DISTRIBUTION_D'] = distribution_score >= 3
        # ==================== 5. 信号后处理与互斥逻辑 ====================
        # 信号平滑处理
        if market_phase == 'TRENDING':
            trend_signals = ['IS_TREND_CONTINUATION_D', 'IS_TREND_REVERSAL_D']
            for col in trend_signals:
                if col in df.columns:
                    df[col] = df[col].fillna(False).astype(bool)
                    try:
                        signal_series = df[col].astype(float)
                        df[col] = (signal_series.rolling(2, min_periods=2).sum() >= 2).fillna(False).astype(bool)
                    except:
                        pass
            if 'IS_TREND_CORRECTION_D' in df.columns:
                df['IS_TREND_CORRECTION_D'] = df['IS_TREND_CORRECTION_D'].fillna(False).astype(bool)
                try:
                    signal_series = df['IS_TREND_CORRECTION_D'].astype(float)
                    df['IS_TREND_CORRECTION_D'] = (signal_series.rolling(2, min_periods=1).max() >= 1).fillna(False).astype(bool)
                    # 避免连续回调信号
                    correction_mask = df['IS_TREND_CORRECTION_D']
                    continuous_correction = (correction_mask & correction_mask.shift(1))
                    df.loc[continuous_correction, 'IS_TREND_CORRECTION_D'] = False
                except:
                    pass
        else:
            consolidation_signals = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 
                                   'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']
            for col in consolidation_signals:
                if col in df.columns:
                    df[col] = df[col].fillna(False).astype(bool)
                    try:
                        signal_series = df[col].astype(float)
                        df[col] = (signal_series.rolling(3, min_periods=2).sum() >= 2).fillna(False).astype(bool)
                    except:
                        pass
        # 信号互斥逻辑
        if market_phase == 'TRENDING':
            # 趋势延续与反转互斥
            if 'IS_TREND_CONTINUATION_D' in df.columns and 'IS_TREND_REVERSAL_D' in df.columns:
                continuation_mask = df['IS_TREND_CONTINUATION_D'].astype(bool)
                if continuation_mask.any():
                    df.loc[continuation_mask, 'IS_TREND_REVERSAL_D'] = False
            # 回调与反转互斥
            if 'IS_TREND_CORRECTION_D' in df.columns and 'IS_TREND_REVERSAL_D' in df.columns:
                correction_mask = df['IS_TREND_CORRECTION_D'].astype(bool)
                if correction_mask.any():
                    df.loc[correction_mask, 'IS_TREND_REVERSAL_D'] = False
            # 回调与延续互斥
            if 'IS_TREND_CONTINUATION_D' in df.columns and 'IS_TREND_CORRECTION_D' in df.columns:
                continuation_mask = df['IS_TREND_CONTINUATION_D'].astype(bool)
                if continuation_mask.any():
                    df.loc[continuation_mask, 'IS_TREND_CORRECTION_D'] = False
        else:
            # 突破与吸筹互斥
            if 'IS_BREAKOUT_D' in df.columns and 'IS_ACCUMULATION_D' in df.columns:
                breakout_mask = df['IS_BREAKOUT_D'].astype(bool)
                if breakout_mask.any():
                    df.loc[breakout_mask, 'IS_ACCUMULATION_D'] = False
            # 派发与突破互斥
            if 'IS_DISTRIBUTION_D' in df.columns and 'IS_BREAKOUT_D' in df.columns:
                distribution_mask = df['IS_DISTRIBUTION_D'].astype(bool)
                if distribution_mask.any():
                    df.loc[distribution_mask, 'IS_BREAKOUT_D'] = False
        # ==================== 6. 结果输出 ====================
        df['MARKET_PHASE_D'] = market_phase
        print(f"=== 分析完成 ===")
        print(f"市场阶段: {market_phase}")
        if market_phase == 'TRENDING':
            signal_cols = ['IS_TREND_CONTINUATION_D', 'IS_TREND_CORRECTION_D', 'IS_TREND_REVERSAL_D']
            for col in signal_cols:
                if col in df.columns:
                    print(f"{col}: 总触发={df[col].sum()}, 最新信号={df[col].iloc[-1]}")
        else:
            signal_cols = ['IS_HIGH_POTENTIAL_CONSOLIDATION_D', 'IS_ACCUMULATION_D', 
                          'IS_BREAKOUT_D', 'IS_DISTRIBUTION_D']
            for col in signal_cols:
                if col in df.columns:
                    print(f"{col}: 总触发={df[col].sum()}, 最新信号={df[col].iloc[-1]}")
        all_dfs[timeframe] = df
        print("=== 高级模式识别引擎(V6.0 多因子融合版)分析完成 ===")
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

    async def calculate_consolidation_period(self, all_dfs: Dict[str, pd.DataFrame], params: dict) -> Dict[str, pd.DataFrame]:
        """
        【V3.0 · 多因子融合版】根据多因子共振识别盘整期
        - 核心升级：使用ChipFactorBase、ChipHoldingMatrixBase、FundFlowFactor三大模型的新字段
        - 三重标准：技术形态 + 筹码结构 + 资金意图，全方位识别高质量盘整期
        - 周期参数：斐波那契数列优化，结合A股市场特性
        """
        if not params.get('enabled', False):
            return all_dfs
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            return all_dfs
        df = all_dfs[timeframe].copy()
        # 调整默认周期为斐波那契数
        boll_period = params.get('boll_period', 21)  # 21是斐波那契数
        boll_std = params.get('boll_std', 2.0)
        roc_period = params.get('roc_period', 13)    # 从12改为13
        vol_ma_period = params.get('vol_ma_period', 55)  # 55是斐波那契数
        bbw_col = f"BBW_{boll_period}_{float(boll_std)}_{timeframe}"
        roc_col = f"ROC_{roc_period}_{timeframe}"
        vol_ma_col = f"VOL_MA_{vol_ma_period}_{timeframe}"
        # ==================== 1. 新模型字段替代方案 ====================
        # 废弃字段映射：
        # dominant_peak_solidity_D -> chip_concentration_ratio_D + concentration_comprehensive_D
        # control_solidity_index_D -> chip_stability_D + flow_stability_D
        # mf_cost_zone_buy_intent_D -> accumulation_score_D + absorption_energy_D
        # deception_lure_long_intensity_D -> flow_consistency_D + large_order_anomaly_D
        # 基础必备字段
        required_base_cols = [
            bbw_col, roc_col, vol_ma_col, 
            f'high_{timeframe}', f'low_{timeframe}', f'volume_{timeframe}',
            f'close_{timeframe}'
        ]
        # 新模型字段 - 按重要性排序
        required_model_cols = [
            # 筹码结构字段 (ChipFactorBase)
            'chip_concentration_ratio_D',        # 筹码集中度
            'chip_stability_D',                  # 筹码稳定性
            'main_cost_range_ratio_D',           # 主力成本区间锁定比例
            # 资金流向字段 (FundFlowFactor)
            'accumulation_score_D',              # 建仓模式得分
            'distribution_score_D',              # 派发模式得分
            'flow_stability_D',                  # 资金流稳定性
            'flow_consistency_D',                # 分档资金一致性
            'large_order_anomaly_D',             # 大单异动
            # 博弈能量场字段 (ChipHoldingMatrixBase)
            'concentration_comprehensive_D',     # 综合集中度
            'absorption_energy_D',               # 吸收能量
            'distribution_energy_D',             # 派发能量
            'energy_concentration_D',            # 能量集中度
        ]
        # 检查字段存在性
        available_base_cols = [col for col in required_base_cols if col in df.columns]
        available_model_cols = [col for col in required_model_cols if col in df.columns]
        if len(available_base_cols) < len(required_base_cols):
            missing = [col for col in required_base_cols if col not in df.columns]
            logger.warning(f"盘整期计算跳过，基础列 '{', '.join(missing)}' 不存在。")
            return all_dfs
        print(f"=== 盘整期识别引擎(V3.0)开始分析 ===")
        print(f"可用的新模型字段: {len(available_model_cols)}/{len(required_model_cols)}")
        # ==================== 2. 动态阈值计算 ====================
        bbw_quantile = params.get('bbw_quantile', 0.25)
        roc_threshold = params.get('roc_threshold', 5.0)
        min_expanding_periods = boll_period * 2
        # 布林带宽度动态阈值
        dynamic_bbw_threshold = df[bbw_col].expanding(min_periods=min_expanding_periods).quantile(bbw_quantile).bfill()
        df[f'dynamic_bbw_threshold_{timeframe}'] = dynamic_bbw_threshold
        # ==================== 3. 三重盘整识别标准 ====================
        # 3.1 技术形态盘整（经典标准）
        cond_volatility = df[bbw_col] < df[f'dynamic_bbw_threshold_{timeframe}']
        cond_trend = df[roc_col].abs() < roc_threshold
        cond_volume = df[f'volume_{timeframe}'] < df[vol_ma_col]
        is_classic_consolidation = cond_volatility & cond_trend & cond_volume
        # 3.2 筹码结构盘整（使用ChipFactorBase和ChipHoldingMatrixBase）
        is_chip_based_consolidation = pd.Series(True, index=df.index)
        # 筹码集中度条件
        if 'chip_concentration_ratio_D' in df.columns:
            # 高集中度表明筹码锁定良好，适合盘整
            cond_chip_concentration = df['chip_concentration_ratio_D'] > 0.6
            is_chip_based_consolidation = is_chip_based_consolidation & cond_chip_concentration
        if 'concentration_comprehensive_D' in df.columns:
            # 综合集中度确认
            cond_comprehensive_concentration = df['concentration_comprehensive_D'] > 0.5
            is_chip_based_consolidation = is_chip_based_consolidation & cond_comprehensive_concentration
        # 筹码稳定性条件
        if 'chip_stability_D' in df.columns:
            cond_chip_stability = df['chip_stability_D'] > 0.5
            is_chip_based_consolidation = is_chip_based_consolidation & cond_chip_stability
        # 主力成本锁定条件
        if 'main_cost_range_ratio_D' in df.columns:
            cond_main_cost_locked = df['main_cost_range_ratio_D'] > 0.6
            is_chip_based_consolidation = is_chip_based_consolidation & cond_main_cost_locked
        # 长线筹码条件
        if 'long_term_chip_ratio_D' in df.columns:
            cond_long_term_chip = df['long_term_chip_ratio_D'] > 0.4
            is_chip_based_consolidation = is_chip_based_consolidation & cond_long_term_chip
        # 3.3 资金意图盘整（使用FundFlowFactor）
        is_intent_based_consolidation = pd.Series(True, index=df.index)
        # 建仓模式条件
        if 'accumulation_score_D' in df.columns:
            # 高建仓得分表明主力在建仓，适合盘整期
            cond_accumulation_score = df['accumulation_score_D'] > 50
            is_intent_based_consolidation = is_intent_based_consolidation & cond_accumulation_score
        # 低派发模式条件
        if 'distribution_score_D' in df.columns:
            cond_low_distribution = df['distribution_score_D'] < 40
            is_intent_based_consolidation = is_intent_based_consolidation & cond_low_distribution
        # 资金流稳定性条件
        if 'flow_stability_D' in df.columns:
            cond_flow_stability = df['flow_stability_D'] > 60
            is_intent_based_consolidation = is_intent_based_consolidation & cond_flow_stability
        # 资金一致性条件（替代欺骗指数）
        if 'flow_consistency_D' in df.columns:
            cond_flow_consistency = df['flow_consistency_D'] > 70
            is_intent_based_consolidation = is_intent_based_consolidation & cond_flow_consistency
        # 无大单异动条件
        if 'large_order_anomaly_D' in df.columns:
            cond_no_anomaly = df['large_order_anomaly_D'] == False
            is_intent_based_consolidation = is_intent_based_consolidation & cond_no_anomaly
        # 3.4 博弈能量场确认（使用ChipHoldingMatrixBase）
        is_energy_based_consolidation = pd.Series(True, index=df.index)
        # 吸收能量条件
        if 'absorption_energy_D' in df.columns:
            cond_absorption_energy = df['absorption_energy_D'] > 50
            is_energy_based_consolidation = is_energy_based_consolidation & cond_absorption_energy
        # 低派发能量条件
        if 'distribution_energy_D' in df.columns:
            cond_low_distribution_energy = df['distribution_energy_D'] < 40
            is_energy_based_consolidation = is_energy_based_consolidation & cond_low_distribution_energy
        # 能量集中度条件
        if 'energy_concentration_D' in df.columns:
            cond_energy_concentration = df['energy_concentration_D'] > 0.6
            is_energy_based_consolidation = is_energy_based_consolidation & cond_energy_concentration
        # ==================== 4. 综合盘整判断逻辑 ====================
        # 三重共振：至少满足两种盘整条件
        consolidation_scores = pd.Series(0, index=df.index)
        # 计分系统
        consolidation_scores += is_classic_consolidation.astype(int)
        consolidation_scores += is_chip_based_consolidation.astype(int)
        consolidation_scores += is_intent_based_consolidation.astype(int)
        consolidation_scores += is_energy_based_consolidation.astype(int)
        # 根据可用模型字段数量调整阈值
        available_models = len(available_model_cols)
        if available_models >= 8:
            # 字段充足时，要求三重共振
            score_threshold = 3
        elif available_models >= 4:
            # 字段中等时，要求双重共振
            score_threshold = 2
        else:
            # 字段不足时，主要依赖技术形态
            score_threshold = 1
        is_consolidating = consolidation_scores >= score_threshold
        # 强度分级
        df[f'consolidation_strength_{timeframe}'] = consolidation_scores
        df[f'consolidation_type_{timeframe}'] = pd.cut(
            consolidation_scores,
            bins=[-1, 0.5, 1.5, 2.5, 3.5, 4.5],
            labels=['NONE', 'WEAK', 'MODERATE', 'STRONG', 'VERY_STRONG']
        )
        # 高级盘整信号：高质量盘整（满足所有条件）
        is_high_quality_consolidation = consolidation_scores == 4
        df[f'is_high_quality_consolidation_{timeframe}'] = is_high_quality_consolidation
        # 主力盘整信号：以筹码和资金意图为主
        is_main_force_consolidation = is_chip_based_consolidation & is_intent_based_consolidation
        df[f'is_main_force_consolidation_{timeframe}'] = is_main_force_consolidation
        df[f'is_consolidating_{timeframe}'] = is_consolidating
        # ==================== 5. 盘整期特征提取 ====================
        if is_consolidating.any():
            consolidation_blocks = (is_consolidating != is_consolidating.shift()).cumsum()
            consolidating_df = df[is_consolidating].copy()
            grouped = consolidating_df.groupby(consolidation_blocks[is_consolidating])
            # 基础特征
            df[f'dynamic_consolidation_high_{timeframe}'] = grouped[f'high_{timeframe}'].transform('max')
            df[f'dynamic_consolidation_low_{timeframe}'] = grouped[f'low_{timeframe}'].transform('min')
            df[f'dynamic_consolidation_avg_vol_{timeframe}'] = grouped[f'volume_{timeframe}'].transform('mean')
            df[f'dynamic_consolidation_duration_{timeframe}'] = grouped[f'high_{timeframe}'].transform('size')
            # 新增：盘整期筹码特征
            if 'chip_concentration_ratio_D' in df.columns:
                df[f'consolidation_chip_concentration_{timeframe}'] = grouped['chip_concentration_ratio_D'].transform('mean')
            if 'chip_stability_D' in df.columns:
                df[f'consolidation_chip_stability_{timeframe}'] = grouped['chip_stability_D'].transform('mean')
            # 新增：盘整期资金特征
            if 'accumulation_score_D' in df.columns:
                df[f'consolidation_accumulation_score_{timeframe}'] = grouped['accumulation_score_D'].transform('mean')
            if 'flow_stability_D' in df.columns:
                df[f'consolidation_flow_stability_{timeframe}'] = grouped['flow_stability_D'].transform('mean')
            # 新增：盘整期能量特征
            if 'absorption_energy_D' in df.columns:
                df[f'consolidation_absorption_energy_{timeframe}'] = grouped['absorption_energy_D'].transform('mean')
            # 前向填充
            fill_cols = [
                f'dynamic_consolidation_high_{timeframe}',
                f'dynamic_consolidation_low_{timeframe}',
                f'dynamic_consolidation_avg_vol_{timeframe}',
                f'dynamic_consolidation_duration_{timeframe}',
                f'consolidation_chip_concentration_{timeframe}',
                f'consolidation_chip_stability_{timeframe}',
                f'consolidation_accumulation_score_{timeframe}',
                f'consolidation_flow_stability_{timeframe}',
                f'consolidation_absorption_energy_{timeframe}'
            ]
            existing_fill_cols = [col for col in fill_cols if col in df.columns]
            df[existing_fill_cols] = df[existing_fill_cols].ffill()
        # 处理缺失值
        df[f'dynamic_consolidation_high_{timeframe}'] = df.get(f'dynamic_consolidation_high_{timeframe}', pd.Series(index=df.index)).fillna(df[f'high_{timeframe}'])
        df[f'dynamic_consolidation_low_{timeframe}'] = df.get(f'dynamic_consolidation_low_{timeframe}', pd.Series(index=df.index)).fillna(df[f'low_{timeframe}'])
        df[f'dynamic_consolidation_avg_vol_{timeframe}'] = df.get(f'dynamic_consolidation_avg_vol_{timeframe}', pd.Series(index=df.index)).fillna(0)
        df[f'dynamic_consolidation_duration_{timeframe}'] = df.get(f'dynamic_consolidation_duration_{timeframe}', pd.Series(index=df.index)).fillna(0)
        # ==================== 6. 盘整期质量评估 ====================
        if 'is_consolidating_D' in df.columns and 'consolidation_strength_D' in df.columns:
            consolidating_mask = df['is_consolidating_D'] == True
            if consolidating_mask.any():
                # 计算盘整期质量评分
                quality_factors = []
                # 筹码集中度质量
                if 'consolidation_chip_concentration_D' in df.columns:
                    chip_conc_score = np.clip(df['consolidation_chip_concentration_D'] * 100, 0, 100)
                    quality_factors.append(chip_conc_score)
                # 资金建仓质量
                if 'consolidation_accumulation_score_D' in df.columns:
                    accum_score = df['consolidation_accumulation_score_D']
                    quality_factors.append(accum_score)
                # 资金稳定性质量
                if 'consolidation_flow_stability_D' in df.columns:
                    flow_stab_score = df['consolidation_flow_stability_D']
                    quality_factors.append(flow_stab_score)
                # 能量吸收质量
                if 'consolidation_absorption_energy_D' in df.columns:
                    energy_score = df['consolidation_absorption_energy_D']
                    quality_factors.append(energy_score)
                # 价格压缩质量（布林带宽度越小越好）
                bbw_norm = 100 - np.clip(df[bbw_col] / df[f'dynamic_bbw_threshold_{timeframe}'] * 50, 0, 100)
                quality_factors.append(bbw_norm)
                # 成交量萎缩质量
                if f'volume_{timeframe}' in df.columns and vol_ma_col in df.columns:
                    vol_ratio = df[f'volume_{timeframe}'] / df[vol_ma_col].replace(0, 1)
                    vol_score = 100 - np.clip(vol_ratio * 50, 0, 100)
                    quality_factors.append(vol_score)
                if quality_factors:
                    quality_matrix = pd.concat(quality_factors, axis=1)
                    df['consolidation_quality_score_D'] = quality_matrix.mean(axis=1)
                    
                    # 质量分级
                    df['consolidation_quality_grade_D'] = pd.cut(
                        df['consolidation_quality_score_D'],
                        bins=[-1, 30, 50, 70, 85, 101],
                        labels=['POOR', 'FAIR', 'GOOD', 'EXCELLENT', 'OUTSTANDING']
                    )
        # ==================== 7. 盘整期突破预警 ====================
        # 监测盘整末期特征
        if 'is_consolidating_D' in df.columns:
            # 寻找盘整结束点
            consolidation_end = (df['is_consolidating_D'] == True) & (df['is_consolidating_D'].shift(-1) == False)
            if consolidation_end.any():
                # 突破预警信号
                if 'volume_D' in df.columns and 'VOL_MA_21_D' in df.columns:
                    volume_breakout = df['volume_D'] > df['VOL_MA_21_D'] * 1.5
                    df['consolidation_volume_breakout_D'] = consolidation_end & volume_breakout
                if 'flow_intensity_D' in df.columns:
                    flow_intensity_surge = df['flow_intensity_D'] > 70
                    df['consolidation_flow_breakout_D'] = consolidation_end & flow_intensity_surge
        # ==================== 8. 结果统计与输出 ====================
        consolidation_count = df[f'is_consolidating_{timeframe}'].sum()
        high_quality_count = df[f'is_high_quality_consolidation_{timeframe}'].sum() if f'is_high_quality_consolidation_{timeframe}' in df.columns else 0
        print(f"盘整期识别统计:")
        print(f"  总盘整期数: {consolidation_count}")
        print(f"  高质量盘整期数: {high_quality_count}")
        print(f"  盘整强度分布: {df[f'consolidation_strength_{timeframe}'].value_counts().to_dict()}")
        if 'consolidation_quality_grade_D' in df.columns:
            quality_dist = df[df[f'is_consolidating_{timeframe}']][f'consolidation_quality_grade_D'].value_counts()
            print(f"  盘整质量分布: {quality_dist.to_dict()}")
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

    async def calculate_och(self, all_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        【V3.5 · OCH非线性融合版】计算整体筹码健康度 (Overall Chip Health, OCH)。
        - 逻辑确认: 此方法在原 DataFrame 上追加 'OCH_D' 列，严格保留包括 'close_D' 在内的所有原始列。
        - 机制: 使用 in-place 修改或列追加模式，防止列丢失。
        """
        timeframe = 'D'
        if timeframe not in all_dfs or all_dfs[timeframe].empty:
            logger.warning(f"计算 OCH 失败：缺少日线数据。")
            return all_dfs
        df = all_dfs[timeframe]
        df_index = df.index
        # 辅助函数：安全获取 Series，缺失补默认值，确保 float32
        def _get_safe_series_local(col_name, default_val=0.0):
            if col_name not in df.columns:
                # 仅在开发调试时开启此日志，避免刷屏
                # print(f"调试信息: OCH计算缺少列: {col_name}，使用默认值 {default_val}。")
                return pd.Series(default_val, index=df_index, dtype=np.float32)
            return df[col_name].fillna(default_val).astype(np.float32)
        # --- 情境自适应调制器 ---
        volatility_context = _get_safe_series_local('VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0).rolling(21).mean().fillna(0).astype(np.float32)
        sentiment_context = _get_safe_series_local('market_sentiment_score_D', 0.0).rolling(21).mean().fillna(0).astype(np.float32)
        entropy_context = _get_safe_series_local('price_volume_entropy_D', 0.0).rolling(21).mean().fillna(0).astype(np.float32)
        # 融合函数：非线性融合 + Numba 优化
        def _nonlinear_fusion(scores_dict: Dict[str, pd.Series], weights_dict: Dict[str, float], volatility_mod: pd.Series, sentiment_mod: pd.Series, entropy_mod: pd.Series) -> pd.Series:
            score_arrays = []
            weight_arrays = []
            for score_name, weight in weights_dict.items():
                score_series = scores_dict.get(score_name, pd.Series(0.0, index=df_index, dtype=np.float32))
                score_arrays.append(score_series.values)
                weight_arrays.append(np.full_like(score_series.values, weight, dtype=np.float32))
            # 调用 Numba 优化核心函数 (确保该函数已在文件头部导入或定义)
            fused_score_values = _numba_nonlinear_fusion_core(
                score_arrays,
                weight_arrays,
                volatility_mod.values,
                sentiment_mod.values,
                entropy_mod.values
            )
            return pd.Series(fused_score_values, index=df_index, dtype=np.float32)
        # --- 1. 筹码集中度与结构优化 ---
        cost_gini = _get_safe_series_local('cost_gini_coefficient_D', 0.5)
        peak_kurtosis = _get_safe_series_local('primary_peak_kurtosis_D', 3.0)
        peak_solidity = _get_safe_series_local('dominant_peak_solidity_D', 0.5)
        peak_volume_ratio = _get_safe_series_local('dominant_peak_volume_ratio_D', 0.5)
        chip_fault = _get_safe_series_local('chip_fault_blockage_ratio_D', 0.0)
        concentration_health = (1 - cost_gini).clip(0, 1)
        normalized_kurtosis = peak_kurtosis.rolling(window=120, min_periods=20).rank(pct=True).fillna(0.5).astype(np.float32)
        peak_quality = (peak_solidity * peak_volume_ratio * normalized_kurtosis).clip(0, 1)
        blockage_penalty = (1 - chip_fault)
        concentration_scores = {'concentration_health': concentration_health, 'peak_quality': peak_quality, 'blockage_penalty': blockage_penalty}
        concentration_weights = {'concentration_health': 0.5, 'peak_quality': 0.4, 'blockage_penalty': 0.1}
        concentration_score = _nonlinear_fusion(concentration_scores, concentration_weights, volatility_context, sentiment_context, entropy_context)
        # --- 2. 成本与盈亏结构动态 ---
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
        cost_structure_scores = {'loser_support': loser_support, 'cost_advantage_score': cost_advantage_score, 'profit_pressure': profit_pressure}
        cost_structure_weights = {'loser_support': 0.4, 'cost_advantage_score': 0.4, 'profit_pressure': -0.2}
        cost_structure_score = _nonlinear_fusion(cost_structure_scores, cost_structure_weights, volatility_context, sentiment_context, entropy_context)
        # --- 3. 持股心态与交易行为 ---
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
            'conviction_lock_score': (conviction_lock_score + 1) / 2,
            'absorption_support_score': (absorption_support_score + 1) / 2,
            'combat_intensity': combat_intensity,
            'wash_trade_penalty': wash_trade_penalty
        }
        sentiment_weights = {'conviction_lock_score': 0.4, 'absorption_support_score': 0.4, 'combat_intensity': 0.2, 'wash_trade_penalty': -0.1}
        sentiment_score = _nonlinear_fusion(sentiment_scores, sentiment_weights, volatility_context, sentiment_context, entropy_context)
        # --- 4. 主力控盘与意图 ---
        mf_control_leverage = _get_safe_series_local('control_solidity_index_D', 0.0)
        mf_on_peak_flow_composite = (_get_safe_series_local('main_force_on_peak_buy_flow_D', 0.0) - _get_safe_series_local('main_force_on_peak_sell_flow_D', 0.0))
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
        och_scores = {'concentration_score': concentration_score, 'cost_structure_score': cost_structure_score, 'sentiment_score': sentiment_score, 'main_force_score': main_force_score}
        och_weights = {'concentration_score': 0.25, 'cost_structure_score': 0.25, 'sentiment_score': 0.25, 'main_force_score': 0.25}
        och_score = _nonlinear_fusion(och_scores, och_weights, volatility_context, sentiment_context, entropy_context) * 2 - 1
        # 【核心逻辑】直接在原 DataFrame 上赋值，确保不会丢失其他列 (如 close_D)
        df['OCH_D'] = och_score.astype(np.float32)
        # 调试输出（可选）
        if 'close_D' not in df.columns:
            logger.error("严重错误：计算 OCH 后 close_D 丢失！请检查是否有其他过滤器。")
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

    async def calculate_breakout_quality_score_v4(self, df_daily: pd.DataFrame, params: dict) -> pd.DataFrame:
        """
        【V4.0】突破质量分计算核心逻辑 - 多因子融合版
        使用新模型字段全面评估突破质量
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
        # 5.3 突破类型识别
        breakout_types = pd.Series('NONE', index=df.index)
        # 根据各维度分数判断突破类型
        for idx in df.index:
            tech_score = results.loc[idx, 'breakout_technical_score'] if 'breakout_technical_score' in results.columns else 0
            chip_score = results.loc[idx, 'breakout_chip_score'] if 'breakout_chip_score' in results.columns else 0
            fund_score = results.loc[idx, 'breakout_fundflow_score'] if 'breakout_fundflow_score' in results.columns else 0
            energy_score = results.loc[idx, 'breakout_energy_score'] if 'breakout_energy_score' in results.columns else 0
            # 确定突破主导力量
            scores = {'TECH': tech_score, 'CHIP': chip_score, 'FUND': fund_score, 'ENERGY': energy_score}
            max_type = max(scores, key=scores.get)
            if max(scores.values()) > 70:
                breakout_types.loc[idx] = f"STRONG_{max_type}"
            elif max(scores.values()) > 50:
                breakout_types.loc[idx] = f"MODERATE_{max_type}"
            elif sum(scores.values()) / 4 > 40:
                breakout_types.loc[idx] = "BALANCED"
            else:
                breakout_types.loc[idx] = "NONE"
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




