# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List

## 修改开始: 导入新的归一化工具 ##
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar
## 修改结束 ##

class ProcessIntelligence:
    """
    【V2.0.0 · 全息四象限引擎】
    - 核心升级: 最终输出分数 meta_score 已升级为 [-1, 1] 的双极区间，完美对齐四象限逻辑。
                +1 代表极强的看涨拐点信号，-1 代表极强的看跌拐点信号。
    - 实现方式: 1. 使用 normalize_to_bipolar 替换 normalize_score 对趋势和加速度进行归一化。
                2. 使用加权平均法替换乘法来融合趋势和加速度，避免负负得正的逻辑错误。
    - 版本: 2.0.0
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
        self.bipolar_sensitivity = get_param_value(self.params.get('bipolar_sensitivity'), 1.0)
        ## 修改开始: 增加新的可配置参数，用于最终分数的融合 ##
        self.meta_score_weights = get_param_value(self.params.get('meta_score_weights'), [0.6, 0.4]) # 新增：趋势与加速度的融合权重
        ## 修改结束 ##
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])

    def run_process_diagnostics(self) -> Dict[str, pd.Series]:
        """
        运行所有在配置中定义的元分析诊断任务。
        """
        print("      -> [过程情报引擎 V2.0.0 · 全息四象限引擎] 启动...") # 更新版本号
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print("      -> [过程情报引擎 V2.0.0] 警告: 数据量不足，跳过诊断。")
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
        print(f"      -> [过程情报引擎 V2.0.0] 分析完毕，共生成 {len(all_process_states)} 个高维度过程元状态。")
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0.0 核心工具 - 第一维】基于“量化力学模型”计算任意两个信号的“瞬时关系分”。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index
        signal_a = df.get(signal_a_name)
        signal_b = df.get(signal_b_name)
        if signal_a is None or signal_b is None:
            print(f"        -> [元分析] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)
        change_a = ta.percent_return(signal_a, length=1).fillna(0)
        change_b = ta.percent_return(signal_b, length=1).fillna(0)
        momentum_a = normalize_to_bipolar(
            series=change_a,
            target_index=df_index,
            window=self.std_window,
            sensitivity=self.bipolar_sensitivity
        )
        thrust_b = normalize_to_bipolar(
            series=change_b,
            target_index=df_index,
            window=self.std_window,
            sensitivity=self.bipolar_sensitivity
        )
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        relationship_score = momentum_a * (1 + signal_b_factor_k * thrust_b)
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.0.0 核心诊断 - 第二维】对“关系分”进行元分析，输出[-1, 1]双极分数。
        """
        signal_name = config.get('name')
        df_index = df.index
        # --- 步骤1: 获取第一维的“瞬时关系分”序列 ---
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{config.get('signal_A')}_VS_{config.get('signal_B')}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        # --- 步骤2: 对“关系分”序列本身，进行趋势和加速度分析 ---
        relationship_trend = ta.linreg(relationship_score, length=self.meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=self.meta_window).fillna(0)
        
        ## 修改开始: 使用双极归一化和加权平均法，生成[-1, 1]的最终分数 ##
        # --- 步骤3: 将趋势和加速度归一化到[-1, 1]区间 ---
        bipolar_trend_strength = normalize_to_bipolar(
            series=relationship_trend,
            target_index=df_index,
            window=self.norm_window,
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_accel_strength = normalize_to_bipolar(
            series=relationship_accel,
            target_index=df_index,
            window=self.norm_window,
            sensitivity=self.bipolar_sensitivity
        )
        
        # --- 步骤4: 使用加权平均法融合，确保最终分数在[-1, 1]区间 ---
        trend_weight = self.meta_score_weights[0]
        accel_weight = self.meta_score_weights[1]
        
        meta_score = (bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight)
        meta_score = meta_score.clip(-1, 1).astype(np.float32) # clip作为最后的保险
        ## 修改结束 ##
        
        return {signal_name: meta_score}
