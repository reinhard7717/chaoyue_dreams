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
        【V1.3 np.choose修复版】使用FFT诊断市场周期
        - 核心修复 (本次修改):
          - [BUG修复] 修复了V1.2版中使用 `np.choose` 导致的 `ValueError: Need at most 32 array objects` 的严重错误。
          - [性能保持] 使用了功能更强大且无限制的NumPy高级索引（`array[rows, cols]`）来替代 `np.choose`，在修复BUG的同时，保持了完全向量化的高性能。
        - 业务逻辑: 保持与V1.2版本完全一致，仅修复实现层面的BUG。
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
            default_series_spec = {
                'SCORE_TRENDING_REGIME_FFT': 0.5, 'SCORE_CYCLICAL_REGIME': 0.0,
                'DOMINANT_CYCLE_PERIOD': np.nan, 'DOMINANT_CYCLE_POWER': 0.0,
                'DOMINANT_CYCLE_PHASE': np.nan
            }
            for name, val in default_series_spec.items():
                states[name] = pd.Series(val, index=df.index, dtype=np.float32)
            return states
        # --- 3. 向量化滚动窗口FFT分析 ---
        close_values = close_prices.to_numpy()
        shape = (len(close_values) - fft_window + 1, fft_window)
        strides = (close_values.strides[0], close_values.strides[0])
        rolling_windows = np.lib.stride_tricks.as_strided(close_values, shape=shape, strides=strides)
        if detrend_method == 'linear':
            linspace_matrix = np.linspace(rolling_windows[:, 0], rolling_windows[:, -1], fft_window).T
            detrended_windows = rolling_windows - linspace_matrix
        else:
            linspace_matrix = np.linspace(rolling_windows[:, 0], rolling_windows[:, -1], fft_window).T
            detrended_windows = rolling_windows - linspace_matrix
        hanning_window = np.hanning(fft_window)
        windowed_segments_cycle = detrended_windows * hanning_window
        windowed_segments_raw = rolling_windows * hanning_window
        yf_cycle = rfft(windowed_segments_cycle, axis=1)
        yf_raw = rfft(windowed_segments_raw, axis=1)
        xf = rfftfreq(fft_window, 1)
        power_spectrum_cycle = np.abs(yf_cycle)**2
        power_spectrum_cycle[:, 0] = 0
        power_spectrum_raw = np.abs(yf_raw)**2
        # --- 4. 向量化频谱分析与信号生成 ---
        # 4.1 计算FFT版趋势分
        total_power_raw = np.sum(power_spectrum_raw[:, 1:], axis=1)
        trend_freq_cutoff = 1.0 / (fft_window / 3.0)
        trend_indices = np.where(xf < trend_freq_cutoff)[0][1:]
        trend_power = np.sum(power_spectrum_raw[:, trend_indices], axis=1)
        trending_score_arr = np.divide(trend_power, total_power_raw, out=np.full_like(total_power_raw, 0.5), where=total_power_raw!=0)
        # 4.2 寻找主导周期
        valid_indices = np.where((xf > 1.0/(fft_window/2)) & (xf < 1.0/4))[0]
        num_results = len(rolling_windows) # 获取结果数组的长度
        if len(valid_indices) > 0:
            power_in_valid_range = power_spectrum_cycle[:, valid_indices]
            dominant_indices_in_valid = np.argmax(power_in_valid_range, axis=1)
            dominant_global_indices = valid_indices[dominant_indices_in_valid]
            dominant_period_arr = 1.0 / xf[dominant_global_indices]
            total_power_cycle = np.sum(power_spectrum_cycle, axis=1)
            # 使用高级索引替代np.choose，修复BUG
            row_indices = np.arange(num_results)
            selected_powers = power_in_valid_range[row_indices, dominant_indices_in_valid]
            dominant_power_arr = np.divide(selected_powers, total_power_cycle, out=np.zeros_like(total_power_cycle), where=total_power_cycle!=0)
            top_indices_in_valid = np.argsort(power_in_valid_range, axis=1)[:, -top_n_cycles:]
            top_powers = np.take_along_axis(power_in_valid_range, top_indices_in_valid, axis=1)
            cyclical_score_arr = np.divide(np.sum(top_powers, axis=1), total_power_cycle, out=np.zeros_like(total_power_cycle), where=total_power_cycle!=0)
            # 4.3 计算相位
            # 使用高级索引替代np.choose，修复BUG
            selected_yf_cycle = yf_cycle[row_indices, dominant_global_indices]
            dominant_phase_rad = np.angle(selected_yf_cycle)
            phase_angle = ((fft_window - 1) * 2 * np.pi / dominant_period_arr + dominant_phase_rad) % (2 * np.pi)
            cycle_phase_arr = np.cos(phase_angle)
        else:
            dominant_period_arr = np.full(num_results, np.nan)
            dominant_power_arr = np.zeros(num_results)
            cyclical_score_arr = np.zeros(num_results)
            cycle_phase_arr = np.full(num_results, np.nan)
        # --- 5. 存储信号 ---
        def to_full_series(arr, fill_value):
            full_arr = np.full(len(df), fill_value, dtype=np.float32)
            # 修正数组切片，应为 fft_window - 1 到结尾
            full_arr[fft_window-1:] = arr
            return pd.Series(full_arr, index=df.index, dtype=np.float32)
        states['SCORE_TRENDING_REGIME_FFT'] = to_full_series(trending_score_arr, 0.5)
        states['SCORE_CYCLICAL_REGIME'] = to_full_series(cyclical_score_arr, 0.0)
        states['DOMINANT_CYCLE_PERIOD'] = to_full_series(dominant_period_arr, np.nan)
        states['DOMINANT_CYCLE_POWER'] = to_full_series(dominant_power_arr, 0.0)
        states['DOMINANT_CYCLE_PHASE'] = to_full_series(cycle_phase_arr, np.nan)
        return states
