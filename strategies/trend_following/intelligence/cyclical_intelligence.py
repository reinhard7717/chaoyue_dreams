# 文件: strategies/trend_following/intelligence/cyclical_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from scipy.fft import rfft, rfftfreq
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

class CyclicalIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化周期情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [周期情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]
    def run_cyclical_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 双星系统版】周期情报分析总指挥
        - 核心重构: 建立FFT(频谱结构)与Hurst(序列记忆)的“双星系统”架构。
        - 核心流程:
          1. 诊断公理一 (FFT): 提取市场的周期性特征。
          2. 诊断公理二 (Hurst): 提取市场的记忆性特征（趋势/均值回归）。
          3. 诊断周期顶风险 (COGNITIVE_RISK_CYCLICAL_TOP)。
          4. 融合裁决: 综合FFT和Hurst的诊断，输出一个更可靠的“趋势政权”评分。
        """
        all_cyclical_states = {}
        p = get_params_block(self.strategy, 'cyclical_analysis_params')
        if not get_param_value(p.get('enabled'), True):
            print("周期情报分析已在配置中禁用，跳过。")
            return {}
        # --- 公理一: 频谱结构 (FFT) ---
        fft_states = self.diagnose_market_cycles_with_fft(df, p)
        all_cyclical_states.update(fft_states)
        # --- 公理二: 序列记忆 (Hurst) ---
        hurst_states = self.diagnose_market_memory_with_hurst(df, p)
        all_cyclical_states.update(hurst_states)
        # [代码新增开始]
        # --- 诊断周期顶风险 ---
        cyclical_top_risk = self._diagnose_cyclical_top_risk(df, fft_states)
        all_cyclical_states.update(cyclical_top_risk)
        # [代码新增结束]
        # --- 融合裁决: 趋势政权 ---
        fft_trend_score = fft_states.get('SCORE_CYCLICAL_FFT_TREND_REGIME', pd.Series(0.5, index=df.index))
        hurst_trend_score = hurst_states.get('SCORE_CYCLICAL_HURST_TREND_REGIME', pd.Series(0.5, index=df.index))
        # 只有当两个独立维度都指向趋势时，才认为趋势政权成立
        final_trend_regime_score = (fft_trend_score * hurst_trend_score).pow(0.5)
        all_cyclical_states['SCORE_TRENDING_REGIME'] = final_trend_regime_score.astype(np.float32)
        return all_cyclical_states
    def _diagnose_cyclical_top_risk(self, df: pd.DataFrame, fft_states: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        【V1.0】诊断认知风险信号：周期顶风险 (COGNITIVE_RISK_CYCLICAL_TOP)
        - 核心逻辑: 融合主导周期强度和当前相位，当市场处于一个强周期的波峰位置时，此风险分会显著提高。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        # 证据1: 主导周期强度 (DOMINANT_CYCLE_POWER)
        # 从 fft_states 中获取，而不是从 df 中获取
        dominant_power = fft_states.get('DOMINANT_CYCLE_POWER', pd.Series(0.0, index=df.index))
        # 证据2: 当前相位 (DOMINANT_CYCLE_PHASE)
        # 相位接近 +1 (波峰) 时风险最高
        # 从 fft_states 中获取，而不是从 df 中获取
        dominant_phase = fft_states.get('DOMINANT_CYCLE_PHASE', pd.Series(0.0, index=df.index))
        # 将相位 [-1, 1] 映射到 [0, 1]，并强调接近 1 的值
        # 例如，使用 (phase + 1) / 2 映射到 [0, 1]，然后平方或指数化以强调波峰
        phase_contribution = ((dominant_phase + 1) / 2).pow(2) # 平方强调波峰
        # 融合：周期强度 * 相位贡献
        # 只有当周期强度高且处于波峰时，风险才高
        cyclical_top_risk = (dominant_power * phase_contribution).fillna(0.0)
        return {'COGNITIVE_RISK_CYCLICAL_TOP': cyclical_top_risk.astype(np.float32)}
    def diagnose_market_cycles_with_fft(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 命名净化版】使用FFT诊断市场周期 (公理一)
        - 核心升级: 净化所有输出信号的命名，明确其来源为FFT。
        - 核心修复: 增加对 'close_D' 数据的存在性检查。
        """
        states = {}
        # --- 1. 获取参数 ---
        fft_window = get_param_value(params.get('fft_window'), 128)
        top_n_cycles = get_param_value(params.get('top_n_cycles'), 3)
        # --- 2. 准备数据与检查 ---
        close_prices = self._get_safe_series(df, 'close_D', method_name="diagnose_market_cycles_with_fft")
        if len(close_prices) < fft_window:
            print(f"日线FFT数据长度({len(close_prices)})不足窗口({fft_window})，跳过计算。")
            default_series_spec = {
                'SCORE_CYCLICAL_FFT_TREND_REGIME': 0.5, 'SCORE_CYCLICAL_FFT_CYCLE_REGIME': 0.0,
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
        signal_specs = {
            'SCORE_CYCLICAL_FFT_TREND_REGIME': (trending_score_arr, 0.5),
            'SCORE_CYCLICAL_FFT_CYCLE_REGIME': (cyclical_score_arr, 0.0),
            'DOMINANT_CYCLE_PERIOD': (dominant_period_arr, np.nan),
            'DOMINANT_CYCLE_POWER': (dominant_power_arr, 0.0),
            'DOMINANT_CYCLE_PHASE': (cycle_phase_arr, np.nan)
        }
        results_df = pd.DataFrame(index=df.index, columns=signal_specs.keys(), dtype=np.float32)
        start_index = fft_window - 1
        for name, (arr, fill_value) in signal_specs.items():
            results_df[name].iloc[:start_index] = fill_value
            results_df[name].iloc[start_index:] = arr
        states = {col: results_df[col] for col in results_df.columns}
        return states
    def diagnose_market_memory_with_hurst(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V2.1 · 双极性重构与多时间维度归一化版】使用Hurst指数诊断市场记忆性 (公理二)
        - 核心重构: 废除线性映射，改用 normalize_to_bipolar 对 (Hurst - 0.5) 进行自适应归一化。
        - 核心逻辑:
          - Hurst > 0.5 (正值输入): 市场具有趋势性，输出正分。
          - Hurst < 0.5 (负值输入): 市场具有均值回归性，输出负分。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `hurst_memory_score` 的归一化方式改为多时间维度自适应归一化。
        """
        states = {}
        hurst_period = get_param_value(params.get('hurst_period'), 120)
        hurst_signal_name = f'hurst_{hurst_period}d_D'
        hurst_series = self._get_safe_series(df, hurst_signal_name, 0.5, method_name="diagnose_market_memory_with_hurst").fillna(0.5)
        if hurst_series.isnull().all(): # 如果获取到的Series全是NaN，说明数据确实不存在
            print(f"Hurst指数 '{hurst_signal_name}' 不存在或全为NaN，跳过公理二诊断。")
            states['SCORE_CYCLICAL_HURST_MEMORY'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            states['SCORE_CYCLICAL_HURST_TREND_REGIME'] = pd.Series(0.5, index=df.index, dtype=np.float32)
            states['SCORE_CYCLICAL_HURST_REVERSION_REGIME'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        # 构造核心双极性序列 (Hurst - 0.5)
        raw_bipolar_series = hurst_series - 0.5
        # 使用双极归一化引擎进行最终裁决，输出一个[-1, 1]的记忆性分数
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        hurst_memory_score = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df.index, default_weights, sensitivity=0.1)
        states['SCORE_CYCLICAL_HURST_MEMORY'] = hurst_memory_score
        # 将双极性分数分解为互斥的单极性“政权”分
        trend_regime_score, reversion_regime_score = bipolar_to_exclusive_unipolar(hurst_memory_score)
        states['SCORE_CYCLICAL_HURST_TREND_REGIME'] = trend_regime_score
        states['SCORE_CYCLICAL_HURST_REVERSION_REGIME'] = reversion_regime_score
        return states

