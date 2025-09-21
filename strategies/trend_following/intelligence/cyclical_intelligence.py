# 文件: strategies/trend_following/intelligence/cyclical_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict
from scipy.fft import rfft, rfftfreq
from strategies.trend_following.utils import get_params_block, get_param_value

class CyclicalIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化周期情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def run_cyclical_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】周期情报分析总指挥
        - 核心职责: 使用快速傅里叶变换(FFT)分析价格序列，提取市场的周期性特征。
        - 产出: 生成描述市场“趋势性”与“周期性”的全新原子信号。
        """
        # print("      -> [周期情报分析总指挥 V1.0] 启动...")
        p = get_params_block(self.strategy, 'cyclical_analysis_params')
        if not get_param_value(p.get('enabled'), True):
            print("      -> [周期情报分析] 已在配置中禁用，跳过。")
            return {}
        fft_states = self.diagnose_market_cycles_with_fft(df, p)
        # print(f"      -> [周期情报分析总指挥 V1.0] 分析完毕，共生成 {len(fft_states)} 个周期信号。")
        return fft_states

    def diagnose_market_cycles_with_fft(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V1.2 向量化性能重构版】使用FFT诊断市场周期
        - 核心优化 (本次修改):
          - [性能重构] 彻底移除了原有的Python for循环，采用NumPy的`as_strided`技术创建滚动窗口的视图(view)，实现了完全向量化的FFT计算。
          - [效率提升] 所有操作（去趋势、加窗、FFT、频谱分析）现在都在整个2D数组上并行执行，避免了上千次的函数调用和数据切片开销，计算效率提升了几个数量级。
          - [内存优化] `as_strided`创建数据视图而不复制数据，显著降低了内存峰值占用。
        - 业务逻辑: 保持与V1.1版本完全一致的去趋势、加窗、频谱分析逻辑，仅重构实现方式。
        """
        states = {}
        # --- 1. 获取参数 ---
        fft_window = get_param_value(params.get('fft_window'), 128) # FFT窗口长度
        top_n_cycles = get_param_value(params.get('top_n_cycles'), 3) # 分析最强的N个周期
        detrend_method = get_param_value(params.get('detrend_method'), 'linear') # 去趋势方法
        # --- 2. 准备数据与检查 ---
        close_prices = df['close_D']
        if len(close_prices) < fft_window:
            print(f"          -> [警告] 日线FFT数据长度({len(close_prices)})不足窗口({fft_window})，跳过计算。")
            # 依然返回带有默认值的字典，以避免下游模块出错
            default_series_spec = {
                'SCORE_TRENDING_REGIME_FFT': 0.5, 'SCORE_CYCLICAL_REGIME': 0.0,
                'DOMINANT_CYCLE_PERIOD': np.nan, 'DOMINANT_CYCLE_POWER': 0.0,
                'DOMINANT_CYCLE_PHASE': np.nan
            }
            for name, val in default_series_spec.items():
                states[name] = pd.Series(val, index=df.index, dtype=np.float32)
            return states
        # --- 3. 向量化滚动窗口FFT分析 ---
        # 代码新增：使用as_strided创建滚动窗口视图，避免循环
        close_values = close_prices.to_numpy()
        shape = (len(close_values) - fft_window + 1, fft_window)
        strides = (close_values.strides[0], close_values.strides[0])
        rolling_windows = np.lib.stride_tricks.as_strided(close_values, shape=shape, strides=strides)
        # 代码新增：向量化去趋势
        if detrend_method == 'linear':
            linspace_matrix = np.linspace(rolling_windows[:, 0], rolling_windows[:, -1], fft_window).T
            detrended_windows = rolling_windows - linspace_matrix
        else: # 默认线性去趋势
            linspace_matrix = np.linspace(rolling_windows[:, 0], rolling_windows[:, -1], fft_window).T
            detrended_windows = rolling_windows - linspace_matrix
        # 代码新增：向量化加窗
        hanning_window = np.hanning(fft_window)
        windowed_segments_cycle = detrended_windows * hanning_window
        windowed_segments_raw = rolling_windows * hanning_window
        # 代码新增：向量化FFT计算
        yf_cycle = rfft(windowed_segments_cycle, axis=1)
        yf_raw = rfft(windowed_segments_raw, axis=1)
        xf = rfftfreq(fft_window, 1)
        # 代码新增：向量化频谱计算
        power_spectrum_cycle = np.abs(yf_cycle)**2
        power_spectrum_cycle[:, 0] = 0  # 忽略所有窗口的直流分量
        power_spectrum_raw = np.abs(yf_raw)**2
        # --- 4. 向量化频谱分析与信号生成 ---
        # 4.1 计算FFT版趋势分
        total_power_raw = np.sum(power_spectrum_raw[:, 1:], axis=1)
        trend_freq_cutoff = 1.0 / (fft_window / 3.0)
        trend_indices = np.where(xf < trend_freq_cutoff)[0][1:]
        trend_power = np.sum(power_spectrum_raw[:, trend_indices], axis=1)
        # 代码新增：使用np.divide安全地处理除以0的情况
        trending_score_arr = np.divide(trend_power, total_power_raw, out=np.full_like(total_power_raw, 0.5), where=total_power_raw!=0)
        # 4.2 寻找主导周期
        valid_indices = np.where((xf > 1.0/(fft_window/2)) & (xf < 1.0/4))[0]
        if len(valid_indices) > 0:
            power_in_valid_range = power_spectrum_cycle[:, valid_indices]
            # 寻找每个窗口的最强周期
            dominant_indices_in_valid = np.argmax(power_in_valid_range, axis=1)
            dominant_global_indices = valid_indices[dominant_indices_in_valid]
            dominant_period_arr = 1.0 / xf[dominant_global_indices]
            # 计算周期性强度
            total_power_cycle = np.sum(power_spectrum_cycle, axis=1)
            dominant_power_arr = np.divide(np.choose(dominant_indices_in_valid, power_in_valid_range.T), total_power_cycle, out=np.zeros_like(total_power_cycle), where=total_power_cycle!=0)
            # 计算周期性总分
            top_indices_in_valid = np.argsort(power_in_valid_range, axis=1)[:, -top_n_cycles:]
            top_powers = np.take_along_axis(power_in_valid_range, top_indices_in_valid, axis=1)
            cyclical_score_arr = np.divide(np.sum(top_powers, axis=1), total_power_cycle, out=np.zeros_like(total_power_cycle), where=total_power_cycle!=0)
            # 4.3 计算相位
            dominant_phase_rad = np.angle(np.choose(dominant_global_indices, yf_cycle.T))
            phase_angle = ((fft_window - 1) * 2 * np.pi / dominant_period_arr + dominant_phase_rad) % (2 * np.pi)
            cycle_phase_arr = np.cos(phase_angle)
        else: # 如果没有有效周期，则生成默认值数组
            num_results = len(rolling_windows)
            dominant_period_arr = np.full(num_results, np.nan)
            dominant_power_arr = np.zeros(num_results)
            cyclical_score_arr = np.zeros(num_results)
            cycle_phase_arr = np.full(num_results, np.nan)
        # --- 5. 存储信号 ---
        # 代码新增：将计算结果数组填充回完整长度的Pandas Series
        def to_full_series(arr, fill_value):
            full_arr = np.full(len(df), fill_value, dtype=np.float32)
            full_arr[fft_window-1:] = arr
            return pd.Series(full_arr, index=df.index, dtype=np.float32)
        states['SCORE_TRENDING_REGIME_FFT'] = to_full_series(trending_score_arr, 0.5)
        states['SCORE_CYCLICAL_REGIME'] = to_full_series(cyclical_score_arr, 0.0)
        states['DOMINANT_CYCLE_PERIOD'] = to_full_series(dominant_period_arr, np.nan)
        states['DOMINANT_CYCLE_POWER'] = to_full_series(dominant_power_arr, 0.0)
        states['DOMINANT_CYCLE_PHASE'] = to_full_series(cycle_phase_arr, np.nan)
        return states
