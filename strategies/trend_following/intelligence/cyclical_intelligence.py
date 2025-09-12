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
        【V1.1 信号处理强化版】使用FFT诊断市场周期
        - 核心升级 (本次修改):
          - [去趋势强化] 明确了“线性去趋势”是FFT分析金融时间序列的关键前置步骤，
                        用于剥离长期趋势，聚焦于周期性成分。
          - [加窗优化] 明确了使用汉宁窗(Hanning Window)是为了平滑窗口两端的突变，
                       减少频谱泄露(Spectral Leakage)，使周期识别更纯净。
          - [趋势分优化] 优化了趋势分的计算逻辑，使其更准确地反映原始信号中的长周期趋势能量。
        - 收益: 确保日线级别的FFT分析能够准确捕捉短期周期性波动，而不是被长期趋势干扰。
        """
        states = {}
        # --- 1. 获取参数 ---
        fft_window = get_param_value(params.get('fft_window'), 128) # FFT窗口长度，最好是2的幂
        top_n_cycles = get_param_value(params.get('top_n_cycles'), 3) # 分析最强的N个周期
        detrend_method = get_param_value(params.get('detrend_method'), 'linear') # 去趋势方法
        # --- 2. 准备数据 ---
        close_prices = df['close_D']
        if len(close_prices) < fft_window:
            print(f"          -> [警告] 日线FFT数据长度({len(close_prices)})不足窗口({fft_window})，跳过计算。")
            # 返回空的或者带有默认值的字典，以避免后续错误
            states['SCORE_TRENDING_REGIME_FFT'] = pd.Series(0.5, index=df.index, dtype=np.float32)
            states['SCORE_CYCLICAL_REGIME'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            states['DOMINANT_CYCLE_PERIOD'] = pd.Series(np.nan, index=df.index, dtype=np.float32)
            states['DOMINANT_CYCLE_POWER'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            states['DOMINANT_CYCLE_PHASE'] = pd.Series(np.nan, index=df.index, dtype=np.float32)
            return states
        # 初始化结果Series
        dominant_period = pd.Series(np.nan, index=df.index)
        dominant_power = pd.Series(0.0, index=df.index)
        cycle_phase = pd.Series(np.nan, index=df.index)
        trending_score = pd.Series(0.5, index=df.index)
        cyclical_score = pd.Series(0.0, index=df.index)
        # --- 3. 滚动执行FFT分析 ---
        for i in range(fft_window, len(df)):
            # 截取窗口数据
            price_segment = close_prices.iloc[i - fft_window : i].values
            # --- 关键预处理步骤 1: 去趋势 (Detrending) ---
            # 目的: 移除窗口内的长期趋势，使FFT能专注于分析波动周期。
            detrended_segment = None
            if detrend_method == 'linear':
                # 方法: 减去从起点到终点的线性趋势线。
                detrended_segment = price_segment - np.linspace(price_segment[0], price_segment[-1], fft_window)
            else: # 默认使用线性去趋势
                detrended_segment = price_segment - np.linspace(price_segment[0], price_segment[-1], fft_window)
            # --- 关键预处理步骤 2: 加窗 (Windowing) ---
            # 目的: 平滑数据窗口两端的边界，防止因数据截断产生的“频谱泄露”。
            windowed_segment = detrended_segment * np.hanning(fft_window)
            # 对去趋势、加窗后的信号执行FFT，用于分析周期性成分
            yf_cycle = rfft(windowed_segment)
            xf = rfftfreq(fft_window, 1) # 频率 (周期/天)
            power_spectrum_cycle = np.abs(yf_cycle)**2
            power_spectrum_cycle[0] = 0 # 忽略直流分量
            total_power_cycle = np.sum(power_spectrum_cycle)
            if total_power_cycle == 0: continue
            # --- 4. 分析频谱，生成信号 ---
            # 4.1 计算FFT版趋势分 (在原始信号上计算)
            # 为了准确评估趋势强度，我们对原始(但加窗)的信号进行FFT
            raw_windowed_segment = price_segment * np.hanning(fft_window)
            yf_raw = rfft(raw_windowed_segment)
            power_spectrum_raw = np.abs(yf_raw)**2
            # 忽略直流分量(索引0)进行总能量计算
            total_power_raw = np.sum(power_spectrum_raw[1:])
            if total_power_raw > 0:
                # 趋势分 = 低频分量的功率 / 总功率
                # 定义长周期(低频)为周期 > fft_window/3 的部分
                trend_freq_cutoff = 1.0 / (fft_window / 3.0)
                # 从索引1开始计算，以排除直流分量
                trend_power = np.sum(power_spectrum_raw[np.where(xf < trend_freq_cutoff)[0][1:]])
                trending_score.iloc[i] = trend_power / total_power_raw
            # 4.2 寻找主导周期 (在去趋势后的信号上寻找)
            # 排除极低频（趋势）和极高频（噪音），例如周期在4天到fft_window/2天之间
            valid_indices = np.where((xf > 1.0/(fft_window/2)) & (xf < 1.0/4))
            if len(valid_indices[0]) > 0:
                # 寻找最强的N个周期
                top_indices = np.argsort(power_spectrum_cycle[valid_indices])[-top_n_cycles:]
                dominant_idx = valid_indices[0][top_indices[-1]]
                # 周期 = 1 / 频率
                current_dominant_period = 1.0 / xf[dominant_idx]
                dominant_period.iloc[i] = current_dominant_period
                # 周期性强度 = 主导周期功率 / 总功率
                dominant_power.iloc[i] = power_spectrum_cycle[dominant_idx] / total_power_cycle
                # 周期性总分 = TopN个周期的功率之和 / 总功率
                cyclical_score.iloc[i] = np.sum(power_spectrum_cycle[valid_indices[0][top_indices]]) / total_power_cycle
                # 4.3 计算当前相位 (在去趋势后的信号上计算)
                dominant_phase_rad = np.angle(yf_cycle[dominant_idx])
                cycle_len = int(round(current_dominant_period))
                if i > cycle_len:
                    # (t * 2 * pi / T + phi) % (2 * pi)
                    # 我们只关心最后一个点，它的相位位置可以近似为 t = fft_window - 1
                    phase_angle = ((fft_window - 1) * 2 * np.pi / current_dominant_period + dominant_phase_rad) % (2 * np.pi)
                    # 将 [0, 2*pi] 的角度映射到 [-1, 1] 的相位分数 (cos)
                    cycle_phase.iloc[i] = np.cos(phase_angle)
        # --- 5. 存储信号 ---
        states['SCORE_TRENDING_REGIME_FFT'] = trending_score.astype(np.float32)
        states['SCORE_CYCLICAL_REGIME'] = cyclical_score.astype(np.float32)
        states['DOMINANT_CYCLE_PERIOD'] = dominant_period.astype(np.float32)
        states['DOMINANT_CYCLE_POWER'] = dominant_power.astype(np.float32)
        states['DOMINANT_CYCLE_PHASE'] = cycle_phase.astype(np.float32) # -1:波谷, +1:波峰
        return states
