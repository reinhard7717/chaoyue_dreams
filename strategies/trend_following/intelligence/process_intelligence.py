# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional

## 导入新的归一化工具 ##
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar
# ##

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
        【V3.2.0 · 单一来源版】
        - 核心修复: 彻底移除在代码中硬编码的 `genesis_diagnostics` 列表。
        - 核心升级: 确保 `process_intelligence_params.diagnostics` 配置是诊断任务的唯一真相来源，
                      消除了重复执行的严重BUG，并遵循了“配置即代码”的最佳实践。
        """
        self.strategy = strategy_instance
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
        # 加载唯一的、权威的信号元数据字典
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        self.std_window = get_param_value(self.params.get('std_window'), 21)
        self.meta_window = get_param_value(self.params.get('meta_window'), 5)
        self.bipolar_sensitivity = get_param_value(self.params.get('bipolar_sensitivity'), 1.0)
        self.meta_score_weights = get_param_value(self.params.get('meta_score_weights'), [0.6, 0.4])
        # 移除硬编码的 'genesis_diagnostics' 列表
        # 直接从配置文件加载所有诊断任务，确保其为唯一真相来源
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])

    def run_process_diagnostics(self, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V3.1.0 · 衰减分析版】运行所有在配置中定义的元分析诊断任务。
        - 核心升级: 新增对 'decay_analysis' 任务类型的支持，调用专属的信号衰减计算引擎。
        """
        all_process_states = {}
        df = self.strategy.df_indicators
        if df.empty:
            return {}
        for config in self.diagnostics_config:
            if task_type_filter and config.get('task_type') != task_type_filter:
                continue
            
            signal_name = config.get('name')
            signal_type = config.get('type')
            
            if not signal_name:
                continue
            # [代码修改开始]
            # 统一路由：无论是基础元分析还是策略同步，都使用同一个诊断引擎
            if signal_type in ['meta_analysis', 'strategy_sync']:
                custom_signal_type = config.get('signal_type')
                if custom_signal_type == 'split_meta_analysis':
                    split_states = self._diagnose_split_meta_relationship(df, config)
                    if split_states:
                        all_process_states.update(split_states)
                # 新增路由到衰减分析器
                elif custom_signal_type == 'decay_analysis':
                    decay_states = self._diagnose_signal_decay(df, config)
                    if decay_states:
                        all_process_states.update(decay_states)
                else:
                    meta_states = self._diagnose_meta_relationship(df, config)
                    if meta_states:
                        all_process_states.update(meta_states)
            # [代码修改结束]
        return all_process_states

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4.0 · 赫拉敕令版】
        - 核心革命: 签署“赫拉的敕令”，废除单一计算模型。引入 `relationship_type` 配置，
                      为 "consensus" (共识) 和 "divergence" (背离) 两种关系类型提供专属的、
                      不可混淆的计算公式，彻底解决了公式错配的根本性问题。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index
        # 新增 relationship_type 的读取
        relationship_type = config.get('relationship_type', 'consensus') # 默认为共识
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
        # 根据关系类型执行不同的计算法则
        if relationship_type == 'divergence':
            # 背离法则：衡量B动量在多大程度上“战胜”了A动量
            relationship_score = (signal_b_factor_k * thrust_b - momentum_a) / (signal_b_factor_k + 1)
        else: # 默认为 'consensus'
            # 共识法则：计算A和B动量的加权平均值
            relationship_score = (momentum_a + signal_b_factor_k * thrust_b) / (1 + signal_b_factor_k)
        relationship_score = relationship_score.clip(-1, 1)
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.0.0 · 希格斯场分析法】对“关系分”进行元分析，输出分数。
        - 核心革命: 废除基于线性回归(linreg)的“趋势/加速度”模型，引入全新的“关系位移/关系动量”模型。
                      这使得引擎能更灵敏地捕捉关系的非线性变化和“势”的拐点，更符合A股特性。
        - 核心修复: 再次强调并修正“加速度”计算的致命逻辑错误。加速度是速度(trend)的一阶导数，
                      必须使用 relationship_trend.diff(1) 进行计算。
        """
        signal_name = config.get('name')
        df_index = df.index
        if signal_name == 'PROCESS_META_WINNER_CONVICTION' and 'antidote_signal' in config:
            relationship_score = self._calculate_winner_conviction_relationship(df, config)
        else:
            relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{config.get('signal_A')}_VS_{config.get('signal_B')}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        diagnosis_mode = config.get('diagnosis_mode', 'meta_analysis')
        if diagnosis_mode == 'direct_confirmation':
            meta_score = relationship_score
        else:
            # --- “希格斯场”分析法核心实现 ---
            # 1. 计算“关系位移”(Displacement)，取代旧的“趋势”(Trend)
            # 它衡量关系分在meta_window周期内的净变化量，更真实地反映短期变化。
            relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
            
            # 2. 计算“关系动量”(Momentum)，取代旧的“加速度”(Acceleration)
            # 它是“关系位移”的一阶导数，衡量关系变化本身的速度，即“势”的变化。
            relationship_momentum = relationship_displacement.diff(1).fillna(0)
            # 3. 将“位移”和“动量”归一化为双极性强度分
            bipolar_displacement_strength = normalize_to_bipolar(
                series=relationship_displacement,
                target_index=df_index,
                window=self.norm_window,
                sensitivity=self.bipolar_sensitivity
            )
            bipolar_momentum_strength = normalize_to_bipolar(
                series=relationship_momentum,
                target_index=df_index,
                window=self.norm_window,
                sensitivity=self.bipolar_sensitivity
            )
            
            # 4. 融合“位移”与“动量”，得到最终的元分析分数
            displacement_weight = self.meta_score_weights[0]
            momentum_weight = self.meta_score_weights[1]
            meta_score = (bipolar_displacement_strength * displacement_weight + bipolar_momentum_strength * momentum_weight)
        # --- 情境门控逻辑 (保持不变) ---
        if diagnosis_mode == 'gated_meta_analysis':
            gate_condition_config = config.get('gate_condition', {})
            gate_type = gate_condition_config.get('type')
            gate_is_open = pd.Series(True, index=df_index) # 默认门是打开的
            if gate_type == 'price_vs_ma':
                ma_period = gate_condition_config.get('ma_period', 5)
                ma_series = df.get(f'EMA_{ma_period}_D')
                if ma_series is not None:
                    # 门控条件：仅当收盘价低于指定均线时，门才打开
                    gate_is_open = df['close_D'] < ma_series
            # 应用门控：只有当门打开时，信号才能通过
            meta_score = meta_score * gate_is_open.astype(float)
        signal_meta = self.score_type_map.get(signal_name, {})
        scoring_mode = signal_meta.get('scoring_mode', 'bipolar')
        if scoring_mode == 'unipolar':
            meta_score = meta_score.clip(lower=0)
        meta_score = meta_score.clip(-1, 1).astype(np.float32)
        return {signal_name: meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 希格斯场分析法】分裂型元关系诊断器
        - 核心升级: 同步采用全新的“关系位移/关系动量”模型进行核心计算。
        """
        states = {}
        output_names = config.get('output_names', {})
        opportunity_signal_name = output_names.get('opportunity')
        risk_signal_name = output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name:
            print(f"        -> [分裂元分析] 警告: 缺少 'output_names' 配置，无法进行信号分裂。")
            return {}
        # 步骤1: 计算瞬时关系分
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        # 步骤2: 对关系分进行动态元分析，得到最终的双极性 meta_score
        # --- “希格斯场”分析法核心实现 ---
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0)
        bipolar_displacement_strength = normalize_to_bipolar(
            series=relationship_displacement,
            target_index=df.index,
            window=self.norm_window,
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_momentum_strength = normalize_to_bipolar(
            series=relationship_momentum,
            target_index=df.index,
            window=self.norm_window,
            sensitivity=self.bipolar_sensitivity
        )
        displacement_weight = self.meta_score_weights[0]
        momentum_weight = self.meta_score_weights[1]
        meta_score = (bipolar_displacement_strength * displacement_weight + bipolar_momentum_strength * momentum_weight)
        meta_score = meta_score.clip(-1, 1)
        # 步骤3: 信号分裂
        # 机会信号: 只取正向部分，代表机会强度
        opportunity_part = meta_score.clip(lower=0)
        states[opportunity_signal_name] = opportunity_part.astype(np.float32)
        # 风险信号: 只取负向部分，并转为正值，代表风险强度
        risk_part = meta_score.clip(upper=0).abs()
        states[risk_signal_name] = risk_part.astype(np.float32)
        return states

    def _calculate_winner_conviction_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 真理探针植入版】“赢家信念”专属关系计算引擎
        - 新增功能: 植入“真理探针”，打印计算过程中的所有中间变量。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        antidote_signal_name = config.get('antidote_signal')
        df_index = df.index
        def get_signal_series(signal_name: str) -> Optional[pd.Series]:
            return df.get(signal_name)
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if series is None: return pd.Series(dtype=np.float32)
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)
        signal_a = get_signal_series(signal_a_name)
        signal_b = get_signal_series(signal_b_name)
        signal_antidote = get_signal_series(antidote_signal_name)
        if signal_a is None or signal_b is None or signal_antidote is None:
            print(f"        -> [赢家信念] 警告: 缺少原始信号 '{signal_a_name}', '{signal_b_name}' 或 '{antidote_signal_name}'。")
            return pd.Series(dtype=np.float32)
        momentum_a = normalize_to_bipolar(get_change_series(signal_a, config.get('change_type_A')), df_index, self.std_window, self.bipolar_sensitivity)
        momentum_b_raw = normalize_to_bipolar(get_change_series(signal_b, config.get('change_type_B')), df_index, self.std_window, self.bipolar_sensitivity)
        momentum_antidote = normalize_to_bipolar(get_change_series(signal_antidote, config.get('antidote_change_type')), df_index, self.std_window, self.bipolar_sensitivity)
        antidote_k = config.get('antidote_k', 1.0)
        momentum_b_corrected = momentum_b_raw + antidote_k * momentum_antidote
        k = config.get('signal_b_factor_k', 1.0)
        relationship_score = (k * momentum_b_corrected - momentum_a) / (k + 1)
        return relationship_score.clip(-1, 1)

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】信号衰减诊断器
        - 核心职责: 专门用于计算单个信号的负向变化（衰减）强度。
        - 数学逻辑: 1. 计算信号的一阶差分。 2. 只保留负值（代表衰减）。 3. 取绝对值。 4. 归一化。
        - 收益: 提供了计算“衰减”的正确且健壮的数学模型，取代了错误的关系诊断模型。
        """
        # [代码新增开始]
        signal_name = config.get('name')
        source_signal_name = config.get('source_signal')
        source_type = config.get('source_type', 'df')
        df_index = df.index
        if not source_signal_name:
            print(f"        -> [衰减分析] 警告: 缺少 'source_signal' 配置。")
            return {}
        # 获取源信号
        if source_type == 'atomic_states':
            source_series = self.strategy.atomic_states.get(source_signal_name)
        else:
            source_series = df.get(source_signal_name)
        if source_series is None:
            print(f"        -> [衰减分析] 警告: 缺少源信号 '{source_signal_name}'。")
            return {}
        # 1. 计算信号的一阶差分（变化）
        signal_change = source_series.diff(1).fillna(0)
        # 2. 只保留负值（衰减），并取绝对值
        decay_magnitude = signal_change.clip(upper=0).abs()
        # 3. 归一化衰减幅度
        decay_score = normalize_score(decay_magnitude, df_index, window=self.norm_window, ascending=True)
        return {signal_name: decay_score.astype(np.float32)}
        # [代码新增结束]







