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
        【V3.1.0 · 法典统一版】
        - 核心升级: 注入四种全新的、深刻的“关系对”诊断任务，从底层重塑系统对市场博弈的理解。
        - 核心修复: 强制加载并持有 score_type_map，确保所有诊断都遵循唯一的“信号法典”。
        """
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
        # [代码新增] 加载唯一的、权威的信号元数据字典
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        self.std_window = get_param_value(self.params.get('std_window'), 21)
        self.meta_window = get_param_value(self.params.get('meta_window'), 5)
        self.bipolar_sensitivity = get_param_value(self.params.get('bipolar_sensitivity'), 1.0)
        self.meta_score_weights = get_param_value(self.params.get('meta_score_weights'), [0.6, 0.4])
        genesis_diagnostics = [
            {
                "name": "PROCESS_META_POWER_TRANSFER", # 权力转移 (主力 vs 散户)
                "type": "meta_analysis", "task_type": "base",
                "signal_A": "main_force_net_flow_consensus_D", "source_A": "df", "change_type_A": "diff",
                "signal_B": "retail_net_flow_consensus_D", "source_B": "df", "change_type_B": "diff",
                "signal_b_factor_k": -1.0, # 关键：主力流入 vs 散户流出，捕捉背离
                "description": "主力从散户手中夺取控制权的过程。"
            },
            {
                "name": "PROCESS_META_STEALTH_ACCUMULATION", # 隐秘吸筹 (筹码 vs 价格)
                "type": "meta_analysis", "task_type": "base",
                "signal_A": "concentration_90pct_D", "source_A": "df", "change_type_A": "diff",
                "signal_B": "pct_change_D", "source_B": "df", "change_type_B": "diff",
                "signal_b_factor_k": -0.5, # 关键：奖励筹码集中时价格的稳定或下跌
                "description": "主力在不拉高价格的情况下悄悄收集筹码的过程。"
            },
            {
                "name": "PROCESS_META_WINNER_CONVICTION", # 赢家信念 (利润 vs 换手)
                "type": "meta_analysis", "task_type": "base",
                "signal_A": "winner_profit_margin_D", "source_A": "df", "change_type_A": "diff",
                "signal_B": "turnover_from_winners_ratio_D", "source_B": "df", "change_type_B": "diff",
                "signal_b_factor_k": -1.0, # 关键：利润增加但换手减少，代表信念坚定
                "description": "获利盘拒绝出售，预期更高价格的过程。"
            },
            {
                "name": "PROCESS_META_LOSER_CAPITULATION", # 投降仪式 (套牢 vs 换手)
                "type": "meta_analysis", "task_type": "base",
                "signal_A": "turnover_from_losers_ratio_D", "source_A": "df", "change_type_A": "diff",
                "signal_B": "total_loser_rate_D", "source_B": "df", "change_type_B": "diff",
                "signal_b_factor_k": 1.0, # 关键：套牢盘比例高位时，其换手率的加速
                "description": "套牢盘最终放弃希望，不计成本卖出的过程。"
            }
        ]
        default_diagnostics = get_param_value(self.params.get('diagnostics'), [])
        self.diagnostics_config = default_diagnostics + genesis_diagnostics

    def run_process_diagnostics(self, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V2.2.0 · 架构升级版】运行所有在配置中定义的元分析诊断任务。
        - 核心升级: 新增对 'strategy_sync' 任务类型的支持，调用专属的计算引擎。
        """
        # print(f"      -> [过程情报引擎 V2.2.0 · 架构升级版] 启动 (模式: {task_type_filter or 'all'})...")
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            print(f"      -> [过程情报引擎 V2.2.0] 警告: 数据量不足，跳过诊断 (模式: {task_type_filter or 'all'})。")
            return {}
            
        for config in self.diagnostics_config:
            if task_type_filter and config.get('task_type') != task_type_filter:
                continue

            signal_name = config.get('name')
            signal_type = config.get('type')
            if not signal_name:
                continue

            # 增加对新任务类型的路由，为 'strategy_sync' 调用专属诊断方法
            if signal_type == 'meta_analysis':
                meta_states = self._diagnose_meta_relationship(df, config)
                if meta_states:
                    all_process_states.update(meta_states)
            elif signal_type == 'strategy_sync':
                sync_states = self._diagnose_strategy_sync(df, config)
                if sync_states:
                    all_process_states.update(sync_states)
        
        # print(f"      -> [过程情报引擎 V2.2.0] 分析完毕 (模式: {task_type_filter or 'all'})，共生成 {len(all_process_states)} 个过程元状态。")
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.2.0 · 范围约束版】
        - 核心修复: 对最终的 relationship_score 增加 .clip(-1, 1) 约束，彻底杜绝范围溢出问题。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index

        def get_signal_series(signal_name: str, source_type: str) -> Optional[pd.Series]:
            if source_type == 'atomic_states':
                return self.strategy.atomic_states.get(signal_name)
            return df.get(signal_name)

        signal_a = get_signal_series(signal_a_name, config.get('source_A', 'df'))
        signal_b = get_signal_series(signal_b_name, config.get('source_B', 'df'))
        
        if signal_a is None or signal_b is None:
            print(f"        -> [元分析] 警告: 缺少原始信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)

        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)

        change_a = get_change_series(signal_a, config.get('change_type_A', 'pct'))
        change_b = get_change_series(signal_b, config.get('change_type_B', 'pct'))
        
        momentum_a = normalize_to_bipolar(change_a, df_index, self.std_window, self.bipolar_sensitivity)
        thrust_b = normalize_to_bipolar(change_b, df_index, self.std_window, self.bipolar_sensitivity)
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        relationship_score = momentum_a * (1 + signal_b_factor_k * thrust_b)
        
        # 增加范围约束，防止数学溢出
        relationship_score = relationship_score.clip(-1, 1)

        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.2.0 · 法典统一版】对“关系分”进行元分析，输出分数。
        - 核心修复: 不再从计算配置中错误地读取 scoring_mode，而是从唯一的、权威的
                      self.score_type_map (信号法典) 中获取，确保语义正确。
        - 收益: 彻底修复了因错误惩罚“隐秘吸筹”等单极性事件而导致在关键拐点分数过低的致命BUG。
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
        # 从权威的 self.score_type_map 获取信号元数据和计分模式
        signal_meta = self.score_type_map.get(signal_name, {})
        scoring_mode = signal_meta.get('scoring_mode', 'bipolar')
        if scoring_mode == 'unipolar':
            meta_score = meta_score.clip(lower=0)
        meta_score = meta_score.clip(-1, 1).astype(np.float32) # clip作为最后的保险
        return {signal_name: meta_score}

    def _diagnose_strategy_sync(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】诊断高阶战略信号之间的同步性。
        - 核心逻辑: 直接将 [0, 1] 的分数映射为 [-1, 1] 的动量，分析其关系。
        """
        signal_name = config.get('name')
        df_index = df.index

        # --- 步骤1: 获取第一维的“瞬时关系分”序列 ---
        relationship_score = self._calculate_strategy_sync_relationship(df, config)
        if relationship_score.empty:
            return {}
        
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{config.get('signal_A')}_VS_{config.get('signal_B')}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)

        # --- 步骤2: 对“关系分”序列本身，进行趋势和加速度分析 ---
        relationship_trend = ta.linreg(relationship_score, length=self.meta_window).fillna(0)
        relationship_accel = ta.linreg(relationship_trend, length=self.meta_window).fillna(0)

        # --- 步骤3: 将趋势和加速度归一化到[-1, 1]区间 ---
        bipolar_trend_strength = normalize_to_bipolar(
            series=relationship_trend, target_index=df_index,
            window=self.norm_window, sensitivity=self.bipolar_sensitivity
        )
        bipolar_accel_strength = normalize_to_bipolar(
            series=relationship_accel, target_index=df_index,
            window=self.norm_window, sensitivity=self.bipolar_sensitivity
        )

        # --- 步骤4: 使用加权平均法融合 ---
        trend_weight = self.meta_score_weights[0]
        accel_weight = self.meta_score_weights[1]
        meta_score = (bipolar_trend_strength * trend_weight + bipolar_accel_strength * accel_weight)
        meta_score = meta_score.clip(-1, 1).astype(np.float32)

        return {signal_name: meta_score}

    def _calculate_strategy_sync_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 范围约束版】
        - 核心修复: 对最终的 relationship_score 增加 .clip(-1, 1) 约束，彻底杜绝范围溢出问题。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')

        signal_a = self.strategy.atomic_states.get(signal_a_name)
        signal_b = self.strategy.atomic_states.get(signal_b_name)
        
        if signal_a is None or signal_b is None:
            print(f"        -> [战略同步] 警告: 缺少战略信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)

        momentum_a = (signal_a - 0.5) * 2
        thrust_b = (signal_b - 0.5) * 2
        
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        relationship_score = momentum_a * (1 + signal_b_factor_k * thrust_b)
        
        # 增加范围约束，防止数学溢出
        relationship_score = relationship_score.clip(-1, 1)
        
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        
        return relationship_score








