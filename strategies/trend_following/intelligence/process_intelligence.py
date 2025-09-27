# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
from typing import Dict, List
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class ProcessIntelligence:
    """
    【V1.1 · 动能增强版】
    - 核心升级: 遵从指挥官指令，引入“加速度”概念，将诊断依据从单一的“趋势”
                  升级为“趋势”与“加速度”融合的“过程动能”，极大提升了诊断的精确度。
    - 版本: 1.1
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
        self.lookback_window = get_param_value(self.params.get('lookback_window'), 13)
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])

    def run_process_diagnostics(self) -> Dict[str, pd.Series]:
        print("      -> [过程情报引擎 V1.1 · 动能增强版] 启动...") # [代码修改] 更新版本号
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print("      -> [过程情报引擎 V1.1] 警告: 数据DataFrame为空，跳过诊断。")
            return {}
        
        for config in self.diagnostics_config:
            signal_name = config.get('name')
            signal_type = config.get('type')
            if not signal_name or not signal_type:
                continue
            
            if signal_type == 'divergence':
                state = self._diagnose_divergence(df, config)
                if state:
                    all_process_states.update(state)
            elif signal_type == 'resonance':
                state = self._diagnose_resonance(df, config)
                if state:
                    all_process_states.update(state)
            
        print(f"      -> [过程情报引擎 V1.1] 分析完毕，共生成 {len(all_process_states)} 个高维度过程元状态。")
        return all_process_states

    def _calculate_raw_process_trend(self, series: pd.Series) -> pd.Series:
        return series.rolling(window=self.lookback_window).apply(
            lambda x: np.polyfit(range(len(x)), x.dropna(), 1)[0] if len(x.dropna()) > 1 else 0, raw=False
        ).fillna(0)

    
    def _calculate_raw_process_accel(self, trend_series: pd.Series) -> pd.Series:
        """
        【核心工具 V1.1】计算原始趋势的加速度（即趋势的斜率）。
        :param trend_series: 原始趋势（斜率）的 pd.Series。
        :return: 加速度的 pd.Series。
        """
        # 加速度是趋势的变化率，所以我们对趋势序列本身再求一次斜率
        return self._calculate_raw_process_trend(trend_series)
    

    def _diagnose_divergence(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        series_a = df.get(signal_a_name)
        series_b = df.get(signal_b_name)
        if series_a is None or series_b is None:
            print(f"        -> [背离诊断] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'，跳过 '{config['name']}' 的计算。")
            return {}
            
        trend_a_raw = self._calculate_raw_process_trend(series_a)
        trend_b_raw = self._calculate_raw_process_trend(series_b)
        
        # 引入加速度计算
        accel_a_raw = self._calculate_raw_process_accel(trend_a_raw)
        accel_b_raw = self._calculate_raw_process_accel(trend_b_raw)
        

        # 条件判断保持不变，只判断趋势方向
        is_divergence = (trend_a_raw < 0) & (trend_b_raw > 0)
        
        # 强度由“过程动能”（趋势+加速度）决定
        strength_trend_norm = normalize_score(trend_b_raw, df.index, self.norm_window)
        strength_accel_norm = normalize_score(accel_b_raw, df.index, self.norm_window)
        # 使用几何平均融合趋势和加速度的强度
        strength_momentum_score = (strength_trend_norm * strength_accel_norm)**0.5
        
        
        final_score = (is_divergence.astype(float) * strength_momentum_score).astype(np.float32)
        return {config['name']: final_score}

    def _diagnose_resonance(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        series_a = df.get(signal_a_name)
        series_b = df.get(signal_b_name)
        if series_a is None or series_b is None:
            print(f"        -> [共振诊断] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'，跳过 '{config['name']}' 的计算。")
            return {}
            
        trend_a_raw = self._calculate_raw_process_trend(series_a)
        trend_b_raw = self._calculate_raw_process_trend(series_b)

        # 引入加速度计算
        accel_a_raw = self._calculate_raw_process_accel(trend_a_raw)
        accel_b_raw = self._calculate_raw_process_accel(trend_b_raw)
        

        # 条件判断保持不变，只判断趋势方向
        is_resonance = (trend_a_raw > 0) & (trend_b_raw > 0)
        
        # 强度由两个信号的“过程动能”共同决定
        momentum_a_norm = (normalize_score(trend_a_raw, df.index, self.norm_window) * normalize_score(accel_a_raw, df.index, self.norm_window))**0.5
        momentum_b_norm = (normalize_score(trend_b_raw, df.index, self.norm_window) * normalize_score(accel_b_raw, df.index, self.norm_window))**0.5
        strength_momentum_score = (momentum_a_norm * momentum_b_norm)**0.5
        
        
        final_score = (is_resonance.astype(float) * strength_momentum_score).astype(np.float32)
        return {config['name']: final_score}








