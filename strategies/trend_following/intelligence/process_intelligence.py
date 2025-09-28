# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional

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

    def run_process_diagnostics(self, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]: # 增加可选参数 task_type_filter
        """
        运行所有在配置中定义的元分析诊断任务。
        - 新增参数 task_type_filter: 可选 'base' 或 'strategy'，用于执行特定类型的任务。
        """
        # 更新版本号和日志，以反映当前执行模式
        print(f"      -> [过程情报引擎 V2.1.0 · 自适应版] 启动 (模式: {task_type_filter or 'all'})...")
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print(f"      -> [过程情报引擎 V2.1.0] 警告: 数据量不足，跳过诊断 (模式: {task_type_filter or 'all'})。")
            return {}
            
        for config in self.diagnostics_config:
            # [代码新增] 根据过滤器执行任务。如果设置了过滤器，但任务类型不匹配，则跳过。
            if task_type_filter and config.get('task_type') != task_type_filter:
                continue

            signal_name = config.get('name')
            signal_type = config.get('type')
            if not signal_name or signal_type != 'meta_analysis':
                continue
            meta_states = self._diagnose_meta_relationship(df, config)
            if meta_states:
                all_process_states.update(meta_states)
        # 更新版本号和日志
        print(f"      -> [过程情报引擎 V2.1.0] 分析完毕 (模式: {task_type_filter or 'all'})，共生成 {len(all_process_states)} 个过程元状态。")
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1.0 核心工具 - 自适应版】
        - 新增: 根据配置中的 'source' 字段，智能选择数据源 (df_indicators 或 atomic_states)。
        - 新增: 根据配置中的 'change_type' 字段，智能选择变化率计算方法 ('pct' 或 'diff')。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index

        # [代码新增] 智能数据源选择
        def get_signal_series(signal_name: str, source_type: str) -> Optional[pd.Series]:
            if source_type == 'atomic_states':
                # 从 self.strategy.atomic_states 获取高阶信号
                return self.strategy.atomic_states.get(signal_name)
            # 默认从 df_indicators 获取原始指标
            return df.get(signal_name)

        signal_a = get_signal_series(signal_a_name, config.get('source_A', 'df'))
        signal_b = get_signal_series(signal_b_name, config.get('source_B', 'df'))
        
        if signal_a is None or signal_b is None:
            print(f"        -> [元分析] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)

        # [代码新增] 自适应物理模型：根据 change_type 选择计算方法
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if change_type == 'diff':
                # 对归一化分数使用差值
                return series.diff(1).fillna(0)
            # 默认对原始数据使用百分比变化
            return ta.percent_return(series, length=1).fillna(0)

        change_a = get_change_series(signal_a, config.get('change_type_A', 'pct'))
        change_b = get_change_series(signal_b, config.get('change_type_B', 'pct'))
        
        momentum_a = normalize_to_bipolar(
            series=change_a, target_index=df_index, window=self.std_window, sensitivity=self.bipolar_sensitivity
        )
        thrust_b = normalize_to_bipolar(
            series=change_b, target_index=df_index, window=self.std_window, sensitivity=self.bipolar_sensitivity
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

        return {signal_name: meta_score}









