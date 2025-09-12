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
        print("      -> [周期情报分析总指挥 V1.0] 启动...")
        
        p = get_params_block(self.strategy, 'cyclical_analysis_params')
        if not get_param_value(p.get('enabled'), True):
            print("      -> [周期情报分析] 已在配置中禁用，跳过。")
            return {}
            
        fft_states = self.diagnose_market_cycles_with_fft(df, p)

        print(f"      -> [周期情报分析总指挥 V1.0] 分析完毕，共生成 {len(fft_states)} 个周期信号。")
        return fft_states

    def diagnose_market_cycles_with_fft(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V1.0 新增】使用FFT诊断市场周期
        - 核心逻辑:
          1. 对价格序列进行FFT，得到频谱。
          2. 分析频谱特征，提取主导周期、周期强度、趋势强度等信息。
          3. 计算当前价格在主导周期中所处的相位。
        - 收益: 为策略提供了全新的频域视角，能够量化市场的“节律”。
        """
        states = {}
        
        # --- 1. 获取参数 ---
        fft_window = get_param_value(params.get('fft_window'), 128) # FFT窗口长度，最好是2的幂
        top_n_cycles = get_param_value(params.get('top_n_cycles'), 3) # 分析最强的N个周期
        low_freq_threshold = get_param_value(params.get('low_freq_threshold'), 0.1) # 低于此频率的视为“趋势”分量

        # --- 2. 准备数据 ---
        close_prices = df['close_D']
        if len(close_prices) < fft_window:
            print(f"          -> [警告] 数据长度({len(close_prices)})不足FFT窗口({fft_window})，跳过计算。")
            return {}

        # 初始化结果Series
        dominant_period = pd.Series(np.nan, index=df.index)
        dominant_power = pd.Series(0.0, index=df.index)
        cycle_phase = pd.Series(np.nan, index=df.index)
        trending_score = pd.Series(0.5, index=df.index) # FFT版趋势分
        cyclical_score = pd.Series(0.0, index=df.index) # 周期性强度分

        # --- 3. 滚动执行FFT分析 ---
        # 这是一个计算密集型操作，实际应用中可能需要优化
        for i in range(fft_window, len(df)):
            # 截取窗口数据
            price_segment = close_prices.iloc[i - fft_window : i].values
            
            # 去趋势并应用窗函数，这是FFT分析金融时间序列的关键步骤
            detrended_price = price_segment - np.linspace(price_segment[0], price_segment[-1], fft_window)
            windowed_price = detrended_price * np.hanning(fft_window)
            
            # 执行FFT
            yf = rfft(windowed_price)
            xf = rfftfreq(fft_window, 1) # 假设每天一个数据点
            
            # 计算功率谱
            power_spectrum = np.abs(yf)**2
            
            # 忽略直流分量(索引0)
            power_spectrum[0] = 0
            
            # --- 4. 分析频谱，生成信号 ---
            total_power = np.sum(power_spectrum)
            if total_power == 0: continue

            # 4.1 计算FFT版趋势分
            # 趋势分 = 低频分量的功率 / 总功率
            low_freq_power = np.sum(power_spectrum[xf < (1.0 / (fft_window * low_freq_threshold))])
            current_trending_score = low_freq_power / total_power
            trending_score.iloc[i] = current_trending_score

            # 4.2 寻找主导周期
            # 排除极低频（趋势）和极高频（噪音）
            valid_indices = np.where((xf > 1.0/fft_window) & (xf < 0.25)) # 周期在4天到fft_window天之间
            if len(valid_indices[0]) == 0: continue
                
            top_indices = np.argsort(power_spectrum[valid_indices])[-top_n_cycles:]
            dominant_idx = valid_indices[0][top_indices[-1]]
            
            # 周期 = 1 / 频率
            current_dominant_period = 1.0 / xf[dominant_idx]
            dominant_period.iloc[i] = current_dominant_period
            
            # 周期性强度 = 主导周期功率 / 总功率
            current_dominant_power = power_spectrum[dominant_idx] / total_power
            dominant_power.iloc[i] = current_dominant_power
            
            # 周期性总分 = TopN个周期的功率之和 / 总功率
            cyclical_score.iloc[i] = np.sum(power_spectrum[valid_indices[0][top_indices]]) / total_power

            # 4.3 计算当前相位
            # 相位 = arctan(虚部 / 实部)
            dominant_phase_rad = np.angle(yf[dominant_idx])
            # 将相位映射到[-1, 1]区间，-1代表波谷，+1代表波峰
            # 这里使用简化的线性映射，实际应用中可用更复杂的模型
            # 一个完整的周期是 2*pi，我们计算当前点在周期中的位置
            # (t * 2 * pi / T + phi) % (2 * pi)
            # 为了简化，我们直接使用反正弦来近似相位位置
            # 注意：这是一个简化的示例，精确的相位计算更复杂
            # 我们用一个简单的方法：将最近一小段数据的走势与理想正弦波做相关性
            cycle_len = int(round(current_dominant_period))
            if i > cycle_len:
                segment_for_phase = detrended_price[-cycle_len:]
                t = np.arange(cycle_len)
                # 创建一个与主导周期和相位匹配的理想正弦波
                ideal_wave = np.cos(2 * np.pi * t / current_dominant_period + dominant_phase_rad)
                # 计算当前价格在理想波形中的位置，这里用一个简化的投影
                # 我们只关心最后一个点，它的相位位置可以近似为 t = cycle_len - 1
                phase_angle = ( (cycle_len - 1) * 2 * np.pi / current_dominant_period + dominant_phase_rad) % (2 * np.pi)
                # 将 [0, 2*pi] 的角度映射到 [-1, 1] 的相位分数
                cycle_phase.iloc[i] = np.cos(phase_angle)

        # --- 5. 存储信号 ---
        states['SCORE_TRENDING_REGIME_FFT'] = trending_score.astype(np.float32)
        states['SCORE_CYCLICAL_REGIME'] = cyclical_score.astype(np.float32)
        states['DOMINANT_CYCLE_PERIOD'] = dominant_period.astype(np.float32)
        states['DOMINANT_CYCLE_POWER'] = dominant_power.astype(np.float32)
        states['DOMINANT_CYCLE_PHASE'] = cycle_phase.astype(np.float32) # -1:波谷, +1:波峰
        
        return states
