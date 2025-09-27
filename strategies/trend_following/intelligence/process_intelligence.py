# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict

from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class ProcessIntelligence:
    """
    【V1.8 · 通用元分析引擎】
    - 核心重构: 引擎已完全泛化，不再局限于量价。通过配置可分析价格与任何维度（资金、筹码等）
                  的关系，并对该“关系”本身进行元分析，寻找拐点。
    - 哲学: 万物皆可关系。通过“量化力学模型”将任意两个信号的关系转化为一张可分析的“走势图”，
            再通过元分析寻找这张“关系走势图”的拐点。
    - 优化: 全面采用 pandas-ta 库进行核心数学计算，性能与优雅性兼备。
    - 版本: 1.8
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
        print("      -> [过程情报引擎 V1.8 · 通用元分析引擎] 启动...") # [代码修改] 更新版本号
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print("      -> [过程情报引擎 V1.8] 警告: 数据量不足，跳过诊断。")
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
            
        print(f"      -> [过程情报引擎 V1.8] 分析完毕，共生成 {len(all_process_states)} 个高维度过程元状态。")
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.8 核心工具 - 第一维】基于“量化力学模型”计算任意两个信号的“瞬时关系分”。
        :param df: 指标DataFrame。
        :param config: 当前诊断任务的配置。
        :return: “瞬时关系分”序列。
        """
        signal_a_name = config.get('signal_A') # 通常是 'close_D'
        signal_b_name = config.get('signal_B') # 另一个维度，如 'volume_D', 'net_flow_consensus_D'
        
        signal_a = df.get(signal_a_name)
        signal_b = df.get(signal_b_name)

        if signal_a is None or signal_b is None:
            print(f"        -> [元分析] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)

        # 使用 pandas-ta 计算百分比变化率
        change_a = signal_a.ta.percent_return(length=1, append=False).fillna(0)
        change_b = signal_b.ta.percent_return(length=1, append=False).fillna(0)
        
        # 步骤1: 标准化 - 计算各自的滚动标准差
        change_a_std = change_a.ta.stdev(length=self.std_window, append=False).replace(0, np.nan).fillna(method='bfill').fillna(1)
        change_b_std = change_b.ta.stdev(length=self.std_window, append=False).replace(0, np.nan).fillna(method='bfill').fillna(1)
        
        # 步骤2: 计算标准化动量
        momentum_a = (change_a / change_a_std).clip(-3, 3)
        thrust_b = (change_b / change_b_std).clip(-3, 3)
        
        # 步骤3: 应用量化力学公式
        # 从配置中获取信号B的影响因子，提供默认值
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        relationship_score = momentum_a * (1 + signal_b_factor_k * thrust_b)
        
        # 将重要的中间结果存入信号总线，供探针使用 (使用动态名称)
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        
        return relationship_score

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.8 核心诊断 - 第二维】对“关系分”进行元分析。
        :param df: 指标DataFrame。
        :param config: 当前诊断任务的配置。
        :return: 最终的元信号分字典。
        """
        signal_name = config.get('name')
        
        # --- 步骤1: 获取第一维的“瞬时关系分”序列 ---
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
            
        # 使用动态名称存储关系分，以便探针和未来扩展
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{config.get('signal_A')}_VS_{config.get('signal_B')}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        
        # --- 步骤2: 对“关系分”序列本身，进行趋势和加速度分析 ---
        relationship_trend = relationship_score.ta.linreg(length=self.meta_window, append=False).fillna(0)
        relationship_accel = relationship_trend.ta.linreg(length=self.meta_window, append=False).fillna(0)
        
        # --- 步骤3: 融合趋势和加速度，形成最终的“元信号” ---
        trend_strength = normalize_score(relationship_trend, df.index, self.norm_window, ascending=True)
        accel_strength = normalize_score(relationship_accel, df.index, self.norm_window, ascending=True)
        
        meta_score = (trend_strength * accel_strength).astype(np.float32)
        
        return {signal_name: meta_score}
