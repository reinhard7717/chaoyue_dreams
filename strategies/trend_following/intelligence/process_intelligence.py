# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict

from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class ProcessIntelligence:
    """
    【V1.8.1 · API修正版】
    - 核心修正: 修复了因错误调用 pandas-ta API 导致的 AttributeError。
                将 'series.ta.indicator()' 的错误形式修正为 'ta.indicator(series)' 的正确函数式调用。
    - 哲学与功能: 保持 V1.8 的通用元分析引擎设计不变。
    - 版本: 1.8.1
    """
    def __init__(self, strategy_instance):
        """
        初始化通用元分析引擎。
        """
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
        # 从配置中读取通用参数
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        self.std_window = get_param_value(self.params.get('std_window'), 21)
        self.meta_window = get_param_value(self.params.get('meta_window'), 5)
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])

    def run_process_diagnostics(self) -> Dict[str, pd.Series]:
        """
        运行所有在配置中定义的元分析诊断任务。
        """
        print("      -> [过程情报引擎 V1.8.1 · API修正版] 启动...") # 更新版本号
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print("      -> [过程情报引擎 V1.8.1] 警告: 数据量不足，跳过诊断。")
            return {}

        # 遍历所有诊断配置，执行元分析
        for config in self.diagnostics_config:
            signal_name = config.get('name')
            signal_type = config.get('type')
            if not signal_name or signal_type != 'meta_analysis':
                continue
            
            meta_states = self._diagnose_meta_relationship(df, config)
            if meta_states:
                all_process_states.update(meta_states)
            
        print(f"      -> [过程情报引擎 V1.8.1] 分析完毕，共生成 {len(all_process_states)} 个高维度过程元状态。")
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.8.1 核心工具 - 第一维】基于“量化力学模型”计算任意两个信号的“瞬时关系分”。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        
        signal_a = df.get(signal_a_name)
        signal_b = df.get(signal_b_name)

        if signal_a is None or signal_b is None:
            print(f"        -> [元分析] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)

        # 使用 ta.percent_return(Series) 的函数式调用
        change_a = ta.percent_return(signal_a, length=1).fillna(0)
        change_b = ta.percent_return(signal_b, length=1).fillna(0)
        
        # 步骤1: 标准化
        # 使用 ta.stdev(Series) 的函数式调用
        change_a_std = ta.stdev(change_a, length=self.std_window).replace(0, np.nan).fillna(method='bfill').fillna(1)
        change_b_std = ta.stdev(change_b, length=self.std_window).replace(0, np.nan).fillna(method='bfill').fillna(1)
        
        # 步骤2: 计算标准化动量
        momentum_a = (change_a / change_a_std).clip(-3, 3)
        thrust_b = (change_b / change_b_std).clip(-3, 3)
        
        # 步骤3: 应用量化力学公式
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        relationship_score = momentum_a * (1 + signal_b_factor_k * thrust_b)
        
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        
        return relationship_score

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.8.1 核心诊断 - 第二维】对“关系分”进行元分析。
        """
        signal_name = config.get('name')
        
        # --- 步骤1: 获取第一维的“瞬时关系分”序列 ---
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
            
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{config.get('signal_A')}_VS_{config.get('signal_B')}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        
        # --- 步骤2: 对“关系分”序列本身，进行趋势和加速度分析 ---
        # 使用 ta.linreg(Series) 的函数式调用
        relationship_trend = ta.linreg(relationship_score, length=self.meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=self.meta_window).fillna(0)
        
        # --- 步骤3: 融合趋势和加速度，形成最终的“元信号” ---
        trend_strength = normalize_score(relationship_trend, df.index, self.norm_window, ascending=True)
        accel_strength = normalize_score(relationship_accel, df.index, self.norm_window, ascending=True)
        
        meta_score = (trend_strength * accel_strength).astype(np.float32)
        
        return {signal_name: meta_score}
