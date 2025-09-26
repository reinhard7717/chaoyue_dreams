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
        【V1.4 · 性能优化版】使用FFT诊断市场周期
        - 核心修复: 修复了V1.2版中使用 `np.choose` 导致的 `ValueError`。
        - 本次优化:
          - [效率/内存] 重构了最终信号的存储方式。不再为每个信号单独创建全尺寸数组，
                        而是创建一个共享的、预填充NaN的DataFrame，然后将所有计算结果
                        一次性填充进去，显著减少了内存分配和重复操作。
        """
        states = {}
        # --- 1. 获取参数 ---
        fft_window = get_param_value(params.get('fft_window'), 128)
        top_n_cycles = get_param_value(params.get('top_n_cycles'), 3)
        detrend_method = get_param_value(params.get('detrend_method'), 'linear')
        
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
        total_power_raw = np.sum(power_spectrum_raw[:, 1:], axis=1)
        trend_freq_cutoff = 1.0 / (fft_window / 3.0)
        trend_indices = np.where(xf < trend_freq_cutoff)[0][1:]
        trend_power = np.sum(power_spectrum_raw[:, trend_indices], axis=1)
        trending_score_arr = np.divide(trend_power, total_power_raw, out=np.full_like(total_power_raw, 0.5), where=total_power_raw!=0)
        
        valid_indices = np.where((xf > 1.0/(fft_window/2)) & (xf < 1.0/4))[0]
        num_results = len(rolling_windows)
        
        if len(valid_indices) > 0:
            power_in_valid_range = power_spectrum_cycle[:, valid_indices]
            dominant_indices_in_valid = np.argmax(power_in_valid_range, axis=1)
            dominant_global_indices = valid_indices[dominant_indices_in_valid]
            dominant_period_arr = 1.0 / xf[dominant_global_indices]
            total_power_cycle = np.sum(power_spectrum_cycle, axis=1)
            
            row_indices = np.arange(num_results)
            selected_powers = power_in_valid_range[row_indices, dominant_indices_in_valid]
            dominant_power_arr = np.divide(selected_powers, total_power_cycle, out=np.zeros_like(total_power_cycle), where=total_power_cycle!=0)
            
            top_indices_in_valid = np.argsort(power_in_valid_range, axis=1)[:, -top_n_cycles:]
            top_powers = np.take_along_axis(power_in_valid_range, top_indices_in_valid, axis=1)
            cyclical_score_arr = np.divide(np.sum(top_powers, axis=1), total_power_cycle, out=np.zeros_like(total_power_cycle), where=total_power_cycle!=0)
            
            selected_yf_cycle = yf_cycle[row_indices, dominant_global_indices]
            dominant_phase_rad = np.angle(selected_yf_cycle)
            phase_angle = ((fft_window - 1) * 2 * np.pi / dominant_period_arr + dominant_phase_rad) % (2 * np.pi)
            cycle_phase_arr = np.cos(phase_angle)
        else:
            dominant_period_arr = np.full(num_results, np.nan)
            dominant_power_arr = np.zeros(num_results)
            cyclical_score_arr = np.zeros(num_results)
            cycle_phase_arr = np.full(num_results, np.nan)
            
        # --- 5. 存储信号 (高效版) ---
        # 定义所有需要生成的信号及其默认值
        signal_specs = {
            'SCORE_TRENDING_REGIME_FFT': (trending_score_arr, 0.5),
            'SCORE_CYCLICAL_REGIME': (cyclical_score_arr, 0.0),
            'DOMINANT_CYCLE_PERIOD': (dominant_period_arr, np.nan),
            'DOMINANT_CYCLE_POWER': (dominant_power_arr, 0.0),
            'DOMINANT_CYCLE_PHASE': (cycle_phase_arr, np.nan)
        }
        
        # 创建一个预填充的DataFrame，一次性分配所有内存
        results_df = pd.DataFrame(index=df.index, columns=signal_specs.keys(), dtype=np.float32)
        
        # 计算结果应该被填充到的起始行索引
        start_index = fft_window - 1
        
        # 一次性遍历所有计算结果，并填充到DataFrame中
        for name, (arr, fill_value) in signal_specs.items():
            results_df[name].iloc[:start_index] = fill_value
            results_df[name].iloc[start_index:] = arr
        
        # 将DataFrame的列转换为一个字典的Series，符合方法签名
        states = {col: results_df[col] for col in results_df.columns}
        
        return states
