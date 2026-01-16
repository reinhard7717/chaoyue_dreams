# 文件: strategies/trend_following/intelligence/process_intelligence.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score, _robust_geometric_mean
)
from strategies.trend_following.intelligence.process.helper import ProcessIntelligenceHelper
from strategies.trend_following.intelligence.process.calculate_main_force_rally_intent import CalculateMainForceRallyIntent
from strategies.trend_following.intelligence.process.calculate_split_order_accumulation import CalculateSplitOrderAccumulation
from strategies.trend_following.intelligence.process.calculate_price_volume_dynamics import CalculatePriceVolumeDynamics
from strategies.trend_following.intelligence.process.calculate_process_covert_accumulation import CalculateProcessCovertAccumulation
from strategies.trend_following.intelligence.process.calculate_price_momentum_divergence import CalculatePriceMomentumDivergence
from strategies.trend_following.intelligence.process.calculate_winner_conviction_decay import CalculateWinnerConvictionDecay
from strategies.trend_following.intelligence.process.calculate_storm_eye_calm import CalculateStormEyeCalm
from strategies.trend_following.intelligence.process.calculate_winner_conviction_relationship import CalculateWinnerConvictionRelationship
from strategies.trend_following.intelligence.process.calculate_cost_advantage_trend_relationship import CalculateCostAdvantageTrendRelationship
from strategies.trend_following.intelligence.process.calculate_upthrust_washout import CalculateUpthrustWashoutRelationship
from strategies.trend_following.intelligence.process.calculate_main_force_control import CalculateMainForceControlRelationship

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
        self.strategy = strategy_instance
        self.helper = ProcessIntelligenceHelper(strategy_instance)
        self.calculate_main_force_rally_intent_processor = CalculateMainForceRallyIntent(strategy_instance, self.helper)
        self.calculate_split_order_accumulation_processor = CalculateSplitOrderAccumulation(strategy_instance, self.helper)
        self.calculate_price_volume_dynamics_processor = CalculatePriceVolumeDynamics(strategy_instance, self.helper)
        self.calculate_process_covert_accumulation_processor = CalculateProcessCovertAccumulation(strategy_instance, self.helper)
        self.calculate_price_momentum_divergence_processor = CalculatePriceMomentumDivergence(strategy_instance, self.helper)
        self.calculate_winner_conviction_decay_processor = CalculateWinnerConvictionDecay(strategy_instance, self.helper)
        self.calculate_storm_eye_calm_processor = CalculateStormEyeCalm(strategy_instance, self.helper)
        self.calculate_winner_conviction_relationship_processor = CalculateWinnerConvictionRelationship(strategy_instance, self.helper)
        self.calculate_cost_advantage_trend_relationship_processor = CalculateCostAdvantageTrendRelationship(strategy_instance, self.helper)
        self.calculate_upthrust_washout_processor = CalculateUpthrustWashoutRelationship(strategy_instance, self.helper)
        self.calculate_main_force_control_processor = CalculateMainForceControlRelationship(strategy_instance, self.helper) # 新增此行
        self.params = self.helper.params
        self.score_type_map = self.helper.score_type_map
        self.norm_window = self.helper.norm_window
        self.std_window = self.helper.std_window
        self.meta_window = self.helper.meta_window
        self.bipolar_sensitivity = self.helper.bipolar_sensitivity
        self.meta_score_weights = self.helper.meta_score_weights
        self.diagnostics_config = self.helper.diagnostics_config
        self.debug_params = self.helper.debug_params
        self.probe_dates = self.helper.probe_dates
        if not hasattr(self.strategy, 'atomic_states'):
            self.strategy.atomic_states = {}

    def _print_debug_output(self, debug_output: Dict):
        """
        统一打印调试信息。
        """
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = 0.0, method_name: str = "") -> pd.Series:
        """
        安全地从DataFrame中获取Series，如果列不存在或值为NaN，则填充默认值。
        V11.0: 针对特定信号，当其值为NaN时，提供更具业务含义的默认值（例如0.5表示中性），
               以适配normalize_score和normalize_to_bipolar的归一化逻辑。
               neutral_nan_defaults 字典已提取到配置中。
        """
        # 从配置中获取 neutral_nan_defaults 字典
        # 假设 self 是 Strategy 实例，或者 self.strategy 是 Strategy 实例
        # get_params_block 函数需要一个 strategy_instance 参数
        # 如果 _get_safe_series 是 Strategy 类的方法，那么 self 就是 strategy_instance
        # 如果 _get_safe_series 是一个辅助类的方法，且该辅助类持有 Strategy 实例的引用 self.strategy
        # 那么这里应该传入 self.strategy
        # 根据上下文，_calculate_main_force_rally_intent 是 Strategy 类的方法，它调用 self._get_safe_series
        # 所以 _get_safe_series 也是 Strategy 类的方法，可以直接传入 self
        process_params = get_params_block(self, 'process_intelligence_params', {})
        neutral_nan_defaults = process_params.get('neutral_nan_defaults', {})
        # 检查是否为需要特殊默认值的信号
        current_default_value = neutral_nan_defaults.get(col_name, default_value)
        if col_name not in df.columns:
            # print(f"  [警告] {method_name}: 列 '{col_name}' 不存在，使用默认值 {current_default_value}")
            return pd.Series(current_default_value, index=df.index, dtype=np.float32)
        series = df[col_name].astype(np.float32)
        # 填充NaN值
        return series.fillna(current_default_value)

    def _get_mtf_slope_accel_score(self, df: pd.DataFrame, base_signal_name: str, mtf_weights_config: Dict, df_index: pd.Index, method_name: str, ascending: bool = True, bipolar: bool = False) -> pd.Series:
        """
        【V1.1 · 统一归一化调用版】计算多时间框架斜率和加速度的融合分数。
        - 核心修正: 统一调用 `_normalize_series` 进行归一化，利用其多时间框架加权能力。
        """
        slope_periods_weights = get_param_value(mtf_weights_config.get('slope_periods'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        accel_periods_weights = get_param_value(mtf_weights_config.get('accel_periods'), {"5": 0.6, "13": 0.4})
        all_scores_components = []
        total_combined_weight = 0.0
        # 处理斜率
        for period_str, weight in slope_periods_weights.items():
            period = int(period_str)
            slope_col = f'SLOPE_{period}_{base_signal_name}'
            slope_raw = self._get_safe_series(df, slope_col, np.nan, method_name=method_name)
            if slope_raw.isnull().all():
                continue
            # 使用 _normalize_series 进行归一化，它会处理多时间框架的加权
            norm_score = self._normalize_series(slope_raw, df_index, bipolar=bipolar, ascending=ascending)
            all_scores_components.append(norm_score * weight)
            total_combined_weight += weight
        # 处理加速度
        for period_str, weight in accel_periods_weights.items():
            period = int(period_str)
            accel_col = f'ACCEL_{period}_{base_signal_name}'
            accel_raw = self._get_safe_series(df, accel_col, np.nan, method_name=method_name)
            if accel_raw.isnull().all():
                continue
            # 使用 _normalize_series 进行归一化
            norm_score = self._normalize_series(accel_raw, df_index, bipolar=bipolar, ascending=ascending)
            all_scores_components.append(norm_score * weight)
            total_combined_weight += weight
        if not all_scores_components or total_combined_weight == 0:
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        fused_score = sum(all_scores_components) / total_combined_weight
        return fused_score.clip(0, 1) if not bipolar else fused_score.clip(-1, 1)

    def _get_mtf_cohesion_score(self, df: pd.DataFrame, base_signal_names: List[str], mtf_weights_config: Dict, df_index: pd.Index, method_name: str) -> pd.Series:
        """
        【V1.2 · 修复Rolling.std()的axis参数错误】计算多时间框架信号的协同性分数。
        此方法将对多个基础信号计算其MTF斜率和加速度融合分数，然后评估这些融合分数之间的离散度，
        离散度越低（即越协同），分数越高。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            base_signal_names (List[str]): 基础信号名称列表，例如 ['close_D', 'volume_D']。
            mtf_weights_config (Dict): 包含 'slope_periods' 和 'accel_periods' 权重的配置字典。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            pd.Series: 融合后的MTF协同性分数。
        """
        all_fused_mtf_scores = {}
        for base_signal_name in base_signal_names:
            # 调用已有的 _get_mtf_slope_accel_score 来获取每个信号的融合MTF分数
            # Cohesion score should be unipolar, so bipolar=False
            fused_score = self._get_mtf_slope_accel_score(df, base_signal_name, mtf_weights_config, df_index, method_name, ascending=True, bipolar=False)
            all_fused_mtf_scores[base_signal_name] = fused_score
        if not all_fused_mtf_scores:
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 将所有融合分数转换为DataFrame
        fused_scores_df = pd.DataFrame(all_fused_mtf_scores, index=df_index)
        # 直接计算每个时间点上（axis=1）不同信号之间的标准差
        # 然后对这个标准差进行滚动平均，以平滑协同性度量
        min_periods_std = max(1, int(self.meta_window * 0.5))
        # 计算每个时间点上，不同信号之间的标准差
        instant_std = fused_scores_df.std(axis=1)
        # 对这个标准差进行滚动平均，以获得更平滑的协同性度量
        smoothed_std = instant_std.rolling(window=self.meta_window, min_periods=min_periods_std).mean()
        # 将标准差转换为协同性分数：标准差越小，分数越高
        # 确保 smoothed_std 不为0，避免除以零。填充NaN为均值，避免极端值
        smoothed_std_safe = smoothed_std.replace(0, np.nan).fillna(smoothed_std.mean())
        cohesion_score = self._normalize_series(smoothed_std_safe, df_index, ascending=False) # 标准差越小，分数越高
        return cohesion_score.clip(0, 1)

    def _normalize_series(self, series: pd.Series, target_index: pd.Index, bipolar: bool = False, ascending: bool = True) -> pd.Series:
        """
        【V1.1 · 统一归一化引擎】
        - 核心职责: 为类内部提供一个统一的、基于多时间框架自适应归一化的方法。
        - 核心逻辑: 根据 bipolar 参数，调用 get_adaptive_mtf_normalized_score (单极) 或
                     get_adaptive_mtf_normalized_bipolar_score (双极) 进行归一化。
        """
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        if bipolar:
            return get_adaptive_mtf_normalized_bipolar_score(
                series=series,
                target_index=target_index,
                tf_weights=actual_mtf_weights,
                sensitivity=self.bipolar_sensitivity
            )
        else:
            return get_adaptive_mtf_normalized_score(
                series=series,
                target_index=target_index,
                ascending=ascending,
                tf_weights=actual_mtf_weights
            )

    def _get_atomic_score(self, df: pd.DataFrame, score_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0 · 原子信号访问器】
        - 核心职责: 提供一个标准的、安全的方法来从 self.strategy.atomic_states 中获取预先计算好的原子信号。
        - 核心逻辑: 尝试从 atomic_states 字典中获取指定的信号 Series。如果不存在，则打印警告并
                     返回一个与 df 索引对齐的、填充了默认值的 Series，以保证数据流的健壮性。
        - 修复: 解决了 'ProcessIntelligence' object has no attribute '_get_atomic_score' 的 AttributeError。
        """
        #  实现了安全的原子信号访问逻辑
        score_series = self.strategy.atomic_states.get(score_name)
        if score_series is None:
            print(f"    -> [过程情报警告] 依赖的原子信号 '{score_name}' 不存在，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return score_series

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: List[str], method_name: str) -> bool:
        """
        【V1.0 · 新增】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        - 核心职责: 作为“投料前校验”的安全阀，确保所有计算都有可靠的数据基础。
        """
        missing_signals = []
        for signal in required_signals:
            if signal not in df.columns and signal not in self.strategy.atomic_states:
                missing_signals.append(signal)
        if missing_signals:
            print(f"    -> [过程情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def _extract_and_validate_config_signals(self, df: pd.DataFrame, config: Dict, method_name: str) -> bool:
        """
        【V1.0 · 新增】蓝图审查官。解析诊断配置，提取所有信号依赖，并进行统一校验。
        - 核心职责: 作为配置驱动逻辑的安全阀，确保“情报蓝图”中的所有“原料”都真实存在。
        """
        required_signals = []
        # 提取元关系分析中的信号
        if config.get('signal_A'):
            required_signals.append(config['signal_A'])
        if config.get('signal_B'):
            required_signals.append(config['signal_B'])
        # 提取赢家信念中的特殊信号
        if config.get('antidote_signal'):
            required_signals.append(config['antidote_signal'])
        # 提取信号衰减分析中的信号
        if config.get('source_signal'):
            required_signals.append(config['source_signal'])
        # 提取领域反转分析中的公理信号
        if config.get('axioms'):
            for axiom_config in config.get('axioms', []):
                if axiom_config.get('name'):
                    required_signals.append(axiom_config['name'])
        # 如果没有需要校验的信号，则直接通过
        if not required_signals:
            return True
        # 调用通用的校验器进行检查
        return self._validate_required_signals(df, required_signals, method_name)

    def _get_mtf_slope_score(self, df: pd.DataFrame, base_signal_name: str, mtf_weights: Dict, df_index: pd.Index, method_name: str, bipolar: bool = True) -> pd.Series:
        """
        【V1.1 · 健壮周期解析版】计算多时间框架斜率的融合分数。
        - 核心修复: 增加对 `mtf_weights` 键的类型检查，确保只有数字周期才被用于构建信号名称。
        """
        fused_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        total_weight = 0.0
        for period_str, weight in mtf_weights.items():
            try:
                period = int(period_str)
            except ValueError:
                continue
            slope_col = f'SLOPE_{period}_{base_signal_name}'
            slope_raw = self._get_safe_series(df, slope_col, np.nan, method_name=method_name)
            if slope_raw.isnull().all():
                continue
            score = self._normalize_series(slope_raw, df_index, bipolar=bipolar)
            fused_score += score * weight
            total_weight += weight
        return (fused_score / total_weight) if total_weight > 0 else pd.Series(0.0, index=df_index, dtype=np.float32)

    def run_process_diagnostics(self, df: pd.DataFrame, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V5.7 · 基石信号增强版】过程情报分析总指挥
        - 核心升级: 扩展“基石信号”清单，将 `PROCESS_META_COVERT_ACCUMULATION` 纳入优先计算，
                      确保其在被其他信号依赖时已就绪。
        """
        print("启动【V5.7 · 基石信号增强版】过程情报分析...")
        all_process_states = {}
        # 直接使用 self.params，因为它已在 __init__ 中加载了 process_intelligence_params
        p_conf = self.params
        diagnostics = get_param_value(p_conf.get('diagnostics'), [])
        # 定义需要优先计算的“基石信号”清单，新增 PROCESS_META_COVERT_ACCUMULATION
        priority_signals = [
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_COVERT_ACCUMULATION'
        ]
        # 依赖前置处理逻辑，遍历基石信号清单
        processed_priority_signals = set()
        for signal_name in priority_signals:
            config = next((d for d in diagnostics if d.get('name') == signal_name), None)
            if config:
                # 使用 _run_meta_analysis 进行计算，以复用其内部逻辑
                score_dict = self._run_meta_analysis(df, config)
                if score_dict:
                    all_process_states.update(score_dict)
                    self.strategy.atomic_states.update(score_dict)
                    latest_score = next(iter(score_dict.values())).iloc[-1]
                    processed_priority_signals.add(signal_name)
        # 主循环处理剩余的诊断任务
        remaining_diagnostics = [d for d in diagnostics if d.get('name') not in processed_priority_signals]
        for diag_config in remaining_diagnostics:
            if task_type_filter:
                if diag_config.get('task_type') != task_type_filter:
                    continue
            diag_name = diag_config.get('name')
            if not diag_name:
                continue
            score = self._run_meta_analysis(df, diag_config)
            if isinstance(score, pd.Series):
                all_process_states[diag_name] = score
                self.strategy.atomic_states[diag_name] = score
            elif isinstance(score, dict):
                all_process_states.update(score)
                self.strategy.atomic_states.update(score)
        print(f"【V5.7 · 基石信号增强版】分析完成，生成 {len(all_process_states)} 个过程元信号。")
        return all_process_states

    def _run_meta_analysis(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1 · 蓝图审查版】元分析调度中心
        - 核心升级: 在调度任何任务前，首先调用 `_extract_and_validate_config_signals`
                      进行“蓝图审查”，确保配置文件中定义的所有依赖信号都真实存在。
        """
        signal_name = config.get('name', '未知信号')
        # “蓝图审查”协议，校验配置文件中声明的所有信号
        if not self._extract_and_validate_config_signals(df, config, f"_run_meta_analysis (for {signal_name})"):
            return {}
        diagnosis_type = config.get('diagnosis_type', 'meta_relationship')
        if diagnosis_type == 'meta_relationship':
            return self._diagnose_meta_relationship(df, config)
        elif diagnosis_type == 'split_meta_relationship':
            return self._diagnose_split_meta_relationship(df, config)
        elif diagnosis_type == 'signal_decay':
            return self._diagnose_signal_decay(df, config)
        elif diagnosis_type == 'domain_reversal':
            return self._diagnose_domain_reversal(df, config)
        else:
            print(f"    -> [过程情报警告] 未知的元分析诊断类型: '{diagnosis_type}'，跳过信号 '{config.get('name')}' 的计算。")
            return {}

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        signal_name = config.get('name', '未知信号')
        if not self._extract_and_validate_config_signals(df, config, f"_run_meta_analysis (for {signal_name})"):
            return {}
        diagnosis_type = config.get('diagnosis_type', 'meta_relationship')
        if diagnosis_type == 'meta_relationship':
            return self._diagnose_meta_relationship_internal(df, config)
        elif diagnosis_type == 'split_meta_relationship':
            return self._diagnose_split_meta_relationship(df, config)
        elif diagnosis_type == 'signal_decay':
            return self._diagnose_signal_decay(df, config)
        elif diagnosis_type == 'domain_reversal':
            return self._diagnose_domain_reversal(df, config)
        else:
            print(f"    -> [过程情报警告] 未知的元分析诊断类型: '{diagnosis_type}'，跳过信号 '{config.get('name')}' 的计算。")
            return {}

    def _diagnose_meta_relationship_internal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        signal_name = config.get('name')
        df_index = df.index
        meta_score = pd.Series(dtype=np.float32)
        if signal_name == 'PROCESS_STRATEGY_FF_VS_STRUCTURE_LEAD':
            relationship_score = self._calculate_ff_vs_structure_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_STRATEGY_DYN_VS_CHIP_DECAY':
            relationship_score = self._calculate_dyn_vs_chip_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_PRICE_VS_RETAIL_CAPITULATION':
            relationship_score = self._calculate_price_vs_capitulation_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_PD_DIVERGENCE_CONFIRM':
            relationship_score = self._calculate_pd_divergence_relationship(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_PROFIT_VS_FLOW':
            relationship_score = self._calculate_profit_vs_flow_relationship(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_PF_REL_BULLISH_TURN':
            meta_score = self._calculate_pf_relationship(df, config)
        elif signal_name == 'PROCESS_META_PC_REL_BULLISH_TURN':
            meta_score = self._calculate_pc_relationship(df, config)
        elif signal_name == 'PROCESS_META_PRICE_VOLUME_DYNAMICS':
            meta_score = self.calculate_price_volume_dynamics_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_RALLY_INTENT':
            meta_score = self.calculate_main_force_rally_intent_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_WINNER_CONVICTION':
            relationship_score = self.calculate_winner_conviction_relationship_processor.calculate(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_LOSER_CAPITULATION':
            meta_score = self._calculate_loser_capitulation(df, config)
        elif signal_name == 'PROCESS_META_COST_ADVANTAGE_TREND':
            meta_score = self.calculate_cost_advantage_trend_relationship_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_CONTROL':
            meta_score = self.calculate_main_force_control_processor.calculate(df, config) # 修改此行
        elif signal_name == 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION':
            meta_score = self._calculate_panic_washout_accumulation(df, config)
        elif signal_name == 'PROCESS_META_DECEPTIVE_ACCUMULATION':
            meta_score = self._calculate_deceptive_accumulation(df, config)
        elif signal_name == 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY':
            meta_score = self.calculate_split_order_accumulation_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_UPTHRUST_WASHOUT':
            meta_score = self.calculate_upthrust_washout_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_ACCUMULATION_INFLECTION':
            meta_score = self._calculate_accumulation_inflection(df, config)
        elif signal_name == 'PROCESS_META_BREAKOUT_ACCELERATION':
            meta_score = self._calculate_breakout_acceleration(df, config)
        elif signal_name == 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT':
            meta_score = self._calculate_fund_flow_accumulation_inflection(df, config)
        elif signal_name == 'PROCESS_META_STOCK_SECTOR_SYNC':
            relationship_score = self._calculate_stock_sector_sync(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_HOT_SECTOR_COOLING':
            relationship_score = self._calculate_hot_sector_cooling(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE':
            meta_score = self.calculate_price_momentum_divergence_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_STORM_EYE_CALM':
            meta_score = self.calculate_storm_eye_calm_processor.calculate(df, config)
        elif signal_name == 'PROCESS_META_WASH_OUT_REBOUND':
            offensive_absorption_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
            meta_score = self._calculate_process_wash_out_rebound(df, offensive_absorption_intent, config)
        elif signal_name == 'PROCESS_META_COVERT_ACCUMULATION':
            meta_score = self.calculate_process_covert_accumulation_processor.calculate(df, config)
        else:
            relationship_score = self._calculate_instantaneous_relationship(df, config)
            if relationship_score.empty:
                return {}
            self.strategy.atomic_states[f"PROCESS_ATOMIC_REL_SCORE_{signal_name}"] = relationship_score.astype(np.float32)
            diagnosis_mode = config.get('diagnosis_mode', 'direct_confirmation')
            if diagnosis_mode == 'direct_confirmation':
                meta_score = relationship_score
            else:
                meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        if meta_score.empty:
            return {}
        return {signal_name: meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.6 · 参数名修正版】分裂型元关系诊断器
        - 核心升级: 增加对 `enable_probe` 配置项的检查，实现探针输出的可配置化管理。
        """
        states = {}
        output_names = config.get('output_names', {})
        opportunity_signal_name = output_names.get('opportunity')
        risk_signal_name = output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name:
            print(f"        -> [分裂元分析] 警告: 缺少 'output_names' 配置，无法进行信号分裂。")
            return {}
        relationship_score = self._calculate_price_efficiency_relationship(df, config)
        if relationship_score.empty:
            return {}
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0)
        bipolar_displacement_strength = self._normalize_series(relationship_displacement, df.index, bipolar=True)
        bipolar_momentum_strength = self._normalize_series(relationship_momentum, df.index, bipolar=True)
        displacement_weight = self.meta_score_weights[0]
        momentum_weight = self.meta_score_weights[1]
        meta_score = (bipolar_displacement_strength * displacement_weight + bipolar_momentum_strength * momentum_weight)
        meta_score = meta_score.clip(-1, 1)
        opportunity_part = meta_score.clip(lower=0)
        states[opportunity_signal_name] = opportunity_part.astype(np.float32)
        risk_part = meta_score.clip(upper=0).abs()
        states[risk_signal_name] = risk_part.astype(np.float32)
        return states

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【生产版】计算“权力转移”信号。
        - 核心逻辑: 融合“主力信念”、“战场清晰度”（由对倒和欺骗构成）来计算资金转移的真实性，
                      并对最终结果进行非线性放大，以捕捉市场的极端博弈。
        """
        required_signals = [
            'net_sh_amount_calibrated_D', 'net_md_amount_calibrated_D', 'net_lg_amount_calibrated_D',
            'net_xl_amount_calibrated_D', 'main_force_conviction_index_D', 'wash_trade_intensity_D',
            'deception_index_D', 'pct_change_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_power_transfer"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        net_sm_amount = self._get_safe_series(df, 'net_sh_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        net_md_amount = self._get_safe_series(df, 'net_md_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        net_lg_amount = self._get_safe_series(df, 'net_lg_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        net_elg_amount = self._get_safe_series(df, 'net_xl_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        main_force_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_power_transfer")
        wash_trade_intensity = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_power_transfer")
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_calculate_power_transfer")
        wash_trade_norm = self._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        deception_norm = self._normalize_series(deception_index, df_index, bipolar=True)
        conviction_norm = self._normalize_series(main_force_conviction, df_index, bipolar=True)
        clarity_from_noise = (1 - wash_trade_norm) * 0.4
        clarity_from_deception = (1 + deception_norm) / 2 * 0.6
        clarity_factor = (clarity_from_noise + clarity_from_deception).clip(0, 1)
        transfer_authenticity_factor = (conviction_norm * clarity_factor).clip(-1, 1)
        md_to_main_force = net_md_amount * transfer_authenticity_factor
        sm_to_main_force = net_sm_amount * transfer_authenticity_factor
        effective_main_force_flow = net_lg_amount + net_elg_amount + md_to_main_force + sm_to_main_force
        effective_retail_flow = (net_sm_amount - sm_to_main_force) + (net_md_amount - md_to_main_force)
        power_transfer_raw = effective_main_force_flow.diff(1) - effective_retail_flow.diff(1)
        normalized_score = self._normalize_series(power_transfer_raw.fillna(0), df_index, bipolar=True)
        final_score = np.sign(normalized_score) * normalized_score.abs().pow(1.2)
        final_score = final_score.clip(-1, 1)
        return final_score.astype(np.float32)

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 承接验证版】计算“价格与散户投降”的专属瞬时关系分。
        - 核心升级: 引入 `active_buying_support_D` (主动承接强度) 作为“真实性放大器”。
                      只有当“价跌慌不增”的表象伴随主力真实承接时，信号才会被放大，
                      以此区分“死寂”与“黄金坑”。
        """
        method_name = "_calculate_price_vs_capitulation_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价格与散户投降关系..."] = ""
        required_signals = ['close_D', 'retail_panic_surrender_index_D', 'active_buying_support_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        _temp_debug_values["基础背离分数"] = {
            "base_divergence_score": base_divergence_score
        }
        # 引入主动承接作为真实性放大器
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        active_buying_norm = self._normalize_series(active_buying_support, df_index, bipolar=False)
        authenticity_amplifier = 1 + active_buying_norm
        final_score = (base_divergence_score * authenticity_amplifier).clip(-1, 1)
        _temp_debug_values["主动承接放大器"] = {
            "active_buying_support": active_buying_support,
            "active_buying_norm": active_buying_norm,
            "authenticity_amplifier": authenticity_amplifier,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础背离分数 ---"] = ""
            for key, series in _temp_debug_values["基础背离分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主动承接放大器 ---"] = ""
            for key, series in _temp_debug_values["主动承接放大器"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价格与散户投降关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_score

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 信念校准版】计算“价格效率”的专属瞬时关系分。
        - 核心升级: 引入由“主力信念”和“对倒强度”构成的“品质因子”，对原始的价效共识分
                      进行“血统校准”，优先采纳由高信念、低噪音资金驱动的行为。
        """
        method_name = "_calculate_price_efficiency_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价格效率关系..."] = ""
        required_signals = ['close_D', 'VPA_EFFICIENCY_D', 'main_force_conviction_index_D', 'wash_trade_intensity_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 计算基础共识分数
        base_consensus_score = self._calculate_instantaneous_relationship(df, config)
        _temp_debug_values["基础共识分数"] = {
            "base_consensus_score": base_consensus_score
        }
        # 引入品质因子进行校准
        main_force_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        wash_trade_intensity = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        conviction_norm = self._normalize_series(main_force_conviction, df_index, bipolar=True)
        wash_trade_norm = self._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        quality_factor = (conviction_norm.clip(lower=0) * (1 - wash_trade_norm)).clip(0, 1)
        final_score = (base_consensus_score * quality_factor).clip(-1, 1)
        _temp_debug_values["品质因子校准"] = {
            "main_force_conviction_index_D": main_force_conviction,
            "wash_trade_intensity_D": wash_trade_intensity,
            "conviction_norm": conviction_norm,
            "wash_trade_norm": wash_trade_norm,
            "quality_factor": quality_factor,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础共识分数 ---"] = ""
        #     for key, series in _temp_debug_values["基础共识分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 品质因子校准 ---"] = ""
        #     for key, series in _temp_debug_values["品质因子校准"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价格效率关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return final_score

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 战场纵深版】计算“博弈背离”的专属瞬时关系分。
        - 核心升级: 引入基于 `main_force_vpoc_D` 的“战场纵深因子”，为博弈背离信号
                      赋予战略坐标。当背离发生在主力成本线之上时，信号将被放大，反之则被抑制。
        """
        method_name = "_calculate_pd_divergence_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算博弈背离关系..."] = ""
        required_signals = ['close_D', 'mf_retail_battle_intensity_D', 'main_force_vpoc_D']
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        _temp_debug_values["基础背离分数"] = {
            "base_divergence_score": base_divergence_score
        }
        # 引入战场纵深因子
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        mf_vpoc = self._get_safe_series(df, 'main_force_vpoc_D', 0.0, method_name=method_name)
        mf_vpoc_safe = mf_vpoc.replace(0, np.nan) # 防止除以零
        battlefield_context_factor = (1 + (close_price - mf_vpoc_safe) / mf_vpoc_safe).fillna(1).clip(0, 2)
        final_score = (base_divergence_score * battlefield_context_factor).clip(-1, 1)
        _temp_debug_values["战场纵深因子"] = {
            "close_D": close_price,
            "main_force_vpoc_D": mf_vpoc,
            "battlefield_context_factor": battlefield_context_factor,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础背离分数 ---"] = ""
            for key, series in _temp_debug_values["基础背离分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 战场纵深因子 ---"] = ""
            for key, series in _temp_debug_values["战场纵深因子"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 博弈背离关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_score

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.5 · 信念侵蚀版】信号衰减诊断器
        - 核心升级: 为“赢家信念衰减”信号分派专属计算引擎，执行全新的“信念侵蚀”逻辑。
        """
        method_name = "_diagnose_signal_decay"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断信号衰减..."] = ""
        signal_name = config.get('name')
        if signal_name == 'PROCESS_META_WINNER_CONVICTION_DECAY':
            # 调用 CalculateWinnerConvictionDecay 处理器
            decay_score = self.calculate_winner_conviction_decay_processor.calculate(df, config)
            # --- 统一输出调试信息 ---
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 信号衰减诊断完成，最终分值: {decay_score.loc[probe_ts]:.4f}"] = ""
                for key, value in debug_output.items():
                    if value:
                        print(f"{key}: {value}")
                    else:
                        print(key)
            return {signal_name: decay_score.astype(np.float32)}
        source_signal_name = config.get('source_signal')
        source_type = config.get('source_type', 'df')
        df_index = df.index
        if not source_signal_name:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"        -> [衰减分析] 警告: 缺少 'source_signal' 配置。"] = ""
                self._print_debug_output(debug_output)
            return {}
        source_series = None
        if source_type == 'atomic_states':
            source_series = self.strategy.atomic_states.get(source_signal_name)
        else:
            source_series = self._get_safe_series(df, source_signal_name, method_name=method_name)
        if source_series is None:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"        -> [衰减分析] 警告: 缺少源信号 '{source_signal_name}'。"] = ""
                self._print_debug_output(debug_output)
            return {}
        _temp_debug_values["原始信号值"] = {
            source_signal_name: source_series
        }
        signal_change = source_series.diff(1).fillna(0)
        decay_magnitude = signal_change.clip(upper=0).abs()
        decay_score = self._normalize_series(decay_magnitude, df_index, ascending=True)
        _temp_debug_values["衰减计算"] = {
            "signal_change": signal_change,
            "decay_magnitude": decay_magnitude,
            "decay_score": decay_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 衰减计算 ---"] = ""
            for key, series in _temp_debug_values["衰减计算"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 信号衰减诊断完成，最终分值: {decay_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return {signal_name: decay_score.astype(np.float32)}

    def _diagnose_domain_reversal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 神谕调度版】通用领域反转诊断调度中心
        - 核心升级: 剥离旧的、有缺陷的计算逻辑，转变为一个纯粹的“调度中心”。
        - 核心职责: 1. 计算各领域的“领域健康度(bipolar_domain_health)”。
                      2. 将健康度数据呈送给新建的 `_judge_domain_reversal` 方法进行最终审判。
        """
        domain_name = config.get('domain_name')
        axiom_configs = config.get('axioms', [])
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        if not domain_name or not axiom_configs or not output_bottom_name or not output_top_name:
            print(f"        -> [领域反转诊断] 警告: 配置不完整，跳过领域 '{domain_name}' 的反转诊断。")
            return {}
        df_index = df.index
        domain_health_components = []
        total_weight = 0.0
        for axiom_config in axiom_configs:
            axiom_name = axiom_config.get('name')
            axiom_weight = axiom_config.get('weight', 0.0)
            # 增加对公理信号是否存在的防御性检查
            if axiom_name not in self.strategy.atomic_states:
                print(f"    -> [过程情报警告] 领域 '{domain_name}' 依赖的公理信号 '{axiom_name}' 不存在，跳过此公理。")
                continue
            axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
            domain_health_components.append(axiom_score * axiom_weight)
            total_weight += abs(axiom_weight) # 使用绝对值权重总和，以正确处理负权重
        if total_weight == 0:
            print(f"        -> [领域反转诊断] 警告: 领域 '{domain_name}' 的公理权重总和为0，无法计算健康度。")
            return {}
        # 计算该领域的双极性健康度
        bipolar_domain_health = (sum(domain_health_components) / total_weight).clip(-1, 1)
        # 将健康度呈送给新的“神谕审判”方法进行最终裁决
        return self._judge_domain_reversal(bipolar_domain_health, config)

    def _judge_domain_reversal(self, bipolar_domain_health: pd.Series, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1 · 神谕审判生产版】领域反转信号的核心审判庭
        - 核心模型: 创立基于“情境感知”的“神谕审判”模型，取代旧的、情境盲目的归一化逻辑。
        - 底部反转神谕: `底部反转分 = 健康度正向变化量 × (1 - 昨日健康度)`
        - 顶部反转神谕: `顶部反转分 = 健康度负向变化量(取绝对值) × (1 + 昨日健康度)`
        """
        domain_name = config.get('domain_name', '未知领域')
        output_bottom_name = config.get('output_bottom_reversal_name')
        output_top_name = config.get('output_top_reversal_name')
        # 计算核心变量
        health_yesterday = bipolar_domain_health.shift(1).fillna(0)
        health_change = bipolar_domain_health.diff(1).fillna(0)
        # 底部反转神谕
        bottom_context_factor = (1 - health_yesterday).clip(0, 2) # 情境调节器：昨日越差，反转越有价值
        bottom_reversal_raw = health_change.clip(lower=0) * bottom_context_factor
        bottom_reversal_score = bottom_reversal_raw.clip(0, 1) # 直接裁剪，其值已具意义
        # 顶部反转神谕
        top_context_factor = (1 + health_yesterday).clip(0, 2) # 情境调节器：昨日越好，反转越危险
        top_reversal_raw = health_change.clip(upper=0).abs() * top_context_factor
        top_reversal_score = top_reversal_raw.clip(0, 1) # 直接裁剪
        # [删除] 移除所有“究极探针”调试代码
        return {
            output_bottom_name: bottom_reversal_score.astype(np.float32),
            output_top_name: top_reversal_score.astype(np.float32)
        }

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.7 · 战果审判与全息资金流验证强化版】计算“恐慌洗盘吸筹”的专属信号。
        - 核心升级: 创立“战果优先”原则。在基础分之上，额外引入由“修复分”驱动的“战果调节器”，
                      对最终得分进行二次审判与加权，旨在奖赏战役结果优异（筹码结构优化）的吸筹行为。
        - 【强化】引入主力资金净流入和资金流可信度作为判断吸筹真实性的核心证据，并对主力资金流出进行惩罚。
        - 【强化】将恐慌、吸收、修复信号升级为多时间维度（MTF）斜率/加速度融合，增强信号鲁棒性。
        - 【强化】增强 `washout_candidate_mask` 的多时间维度感知。
        - 【新增】引入整体 MTF 价格趋势作为上下文，区分“超跌反弹”和“下跌中继”中的洗盘吸筹。
        - 【新增】增加“价格企稳”的判断，确保洗盘后有筑底迹象。
        """
        method_name = "_calculate_panic_washout_accumulation"
        df_index = df.index
        historical_potential = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        potential_gate = config.get('historical_potential_gate', 0.0)
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'pct_change_D', 'close_D', 'retail_panic_surrender_index_D', 'loser_pain_index_D',
            'main_force_net_flow_calibrated_D', 'active_buying_support_D',
            'SLOPE_1_winner_concentration_90pct_D', 'main_force_cost_advantage_D',
            'structural_leverage_D', 'flow_credibility_index_D',
            'volume_D', 'volume_burstiness_index_D', 'lower_shadow_absorption_strength_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['retail_panic_surrender_index_D', 'loser_pain_index_D', 'active_buying_support_D',
                         'winner_concentration_90pct_D', 'main_force_cost_advantage_D', 'main_force_net_flow_calibrated_D',
                         'lower_shadow_absorption_strength_D', 'volume_D', 'pct_change_D', 'close_D']: # 增加close_D的MTF趋势
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        required_atomic_signals = [
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 'SCORE_BEHAVIOR_VOLUME_BURST',
            'PROCESS_META_POWER_TRANSFER'
        ]
        all_required_signals = required_signals + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 原始数据获取 ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        lower_shadow_strength = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        volume_burst = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_BURST', 0.0)
        retail_panic_index = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        loser_pain_index = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        power_transfer_score = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        concentration_slope = self._get_safe_series(df, f'SLOPE_1_winner_concentration_90pct_D', 0.0, method_name=method_name)
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name)
        structural_leverage_raw = self._get_safe_series(df, 'structural_leverage_D', 0.0, method_name=method_name)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        volume_raw = self._get_safe_series(df, 'volume_D', 0.0, method_name=method_name)
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        # 恐慌分：使用MTF斜率/加速度融合
        mtf_retail_panic = self._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_pain = self._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        panic_score_instant = (mtf_retail_panic * 0.7 + mtf_loser_pain * 0.3).clip(0, 1)
        panic_score = panic_score_instant.rolling(3, min_periods=1).mean()
        # 吸收分：使用MTF斜率/加速度融合
        mtf_active_buying_support = self._get_mtf_slope_accel_score(df, 'active_buying_support_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_lower_shadow_absorption = self._get_mtf_slope_accel_score(df, 'lower_shadow_absorption_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        absorption_score_instant = (
            power_transfer_score.clip(lower=0) * 0.5 +
            mtf_lower_shadow_absorption * 0.25 +
            mtf_active_buying_support * 0.25
        ).clip(0, 1)
        absorption_score = absorption_score_instant.rolling(3, min_periods=1).mean()
        # 修复分：使用MTF斜率/加速度融合
        mtf_concentration_slope = self._get_mtf_slope_accel_score(df, 'winner_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True) # 筹码集中度斜率应为双极
        mtf_cost_advantage_slope = self._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True) # 成本优势斜率应为双极
        mtf_mf_net_flow_slope = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True) # 主力净流斜率应为双极
        structural_leverage_norm = get_adaptive_mtf_normalized_score(structural_leverage_raw, df_index, actual_mtf_weights, ascending=True)
        # 修复分应更侧重于积极的修复信号
        original_repair_score = (mtf_concentration_slope.clip(lower=0) * 0.4 + mtf_cost_advantage_slope.clip(lower=0) * 0.3 + mtf_mf_net_flow_slope.clip(lower=0) * 0.3).clip(0, 1)
        repair_score = (original_repair_score * 0.5 + structural_leverage_norm * 0.5).clip(0, 1)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        # --- 4. 场景识别 (Washout Candidate Mask) - 增强多时间维度感知 ---
        # 整体MTF价格趋势
        mtf_price_trend = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 价格下跌深度：使用MTF价格趋势的负向部分
        is_significant_drop_cumulative = (mtf_price_trend < -0.3) # 整体MTF价格趋势显著为负
        # 成交量爆发：使用MTF成交量爆发
        mtf_volume_burst = self._get_mtf_slope_accel_score(df, 'volume_burstiness_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        is_blitz_washout = (is_significant_drop_cumulative) & (mtf_volume_burst > 0.5) # 闪电洗盘
        is_high_panic = panic_score > 0.4
        is_high_absorption = absorption_score > 0.15
        # 价格企稳判断：短期MTF价格趋势接近0或略正，且波动率下降
        # 假设我们有一个短期MTF价格趋势信号，例如5日斜率
        short_term_price_slope = self._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name=method_name)
        norm_short_term_price_slope = self._normalize_series(short_term_price_slope, df_index, bipolar=True)
        price_volatility = close_price.rolling(window=5).std() / close_price.rolling(window=5).mean() # 5日价格波动率
        norm_price_volatility = self._normalize_series(price_volatility, df_index, bipolar=False, ascending=False) # 波动率越低越好
        is_price_stabilizing = (norm_short_term_price_slope > -0.2) & (norm_short_term_price_slope < 0.2) & (norm_price_volatility > 0.5) # 短期趋势平稳且波动率低
        is_protracted_washout = is_high_panic & is_high_absorption & is_price_stabilizing # 价格企稳后的长期洗盘
        pct_change_3d = close_price.pct_change(3).fillna(0)
        is_moderate_rise = (pct_change_3d > 0) & (pct_change_3d < 0.10)
        is_mid_air_refueling = is_high_panic & is_high_absorption & is_moderate_rise
        washout_candidate_mask = is_blitz_washout | is_protracted_washout | is_mid_air_refueling
        # --- 5. 基础分计算 ---
        base_score = (panic_score * absorption_score * repair_score).pow(1/3)
        # --- 6. 战果调节器 (Battle Outcome Modulator) ---
        battle_outcome_modulator = 1 + repair_score
        # --- 7. 主力资金净流入验证 (Main Force Net Flow Validation) ---
        mf_inflow_validation = main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm
        # --- 8. 最终审判 ---
        judged_base_score = (base_score * battle_outcome_modulator * mf_inflow_validation).clip(0, 1)
        # --- 9. 主力资金流出惩罚 (Main Force Outflow Penalty) ---
        mf_outflow_penalty = main_force_net_flow_norm.clip(upper=0).abs()
        final_score = judged_base_score * (1 - mf_outflow_penalty)
        # --- 10. 势能门控与整体趋势过滤 ---
        potential_gate_mask = historical_potential > potential_gate
        # 整体趋势过滤：如果MTF价格趋势强烈负向，则大幅惩罚或归零
        trend_penalty_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        trend_penalty_factor = trend_penalty_factor.mask(mtf_price_trend < -0.5, 0.0) # 强烈下跌趋势直接归零
        trend_penalty_factor = trend_penalty_factor.mask((mtf_price_trend < -0.2) & (mtf_price_trend >= -0.5), 0.5) # 中等下跌趋势惩罚50%
        final_score = final_score.where(washout_candidate_mask & potential_gate_mask, 0.0).fillna(0.0)
        final_score = final_score * trend_penalty_factor # 应用趋势过滤
        self.strategy.atomic_states["_DEBUG_washout_panic_score"] = panic_score
        self.strategy.atomic_states["_DEBUG_washout_absorption_score"] = absorption_score
        self.strategy.atomic_states["_DEBUG_washout_repair_score"] = repair_score
        self.strategy.atomic_states["_DEBUG_washout_judged_base_score"] = judged_base_score
        self.strategy.atomic_states["_DEBUG_washout_mf_inflow_validation"] = mf_inflow_validation
        self.strategy.atomic_states["_DEBUG_washout_mf_outflow_penalty"] = mf_outflow_penalty
        return final_score.astype(np.float32)

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.8 · 价格筹码背离与多维诡道融合版】计算“诡道吸筹”信号。
        - 核心重构: 创立“价格筹码背离共振”原则。明确捕捉“价格下跌”与“筹码加速集中”的矛盾共振，
                      这是主力“明修栈道，暗度陈仓”的终极诡道。
        - 证据升级: 引入 `winner_concentration_90pct_D` 的 MTF 斜率/加速度，作为筹码加速集中的核心证据。
        - 【强化】将 `price_trend_norm` 和 `deception_evidence` 升级为 MTF 融合信号。
        - 【重要修正】重构 `deceptive_context_score`，使其直接体现“价格下跌”与“筹码集中”的背离共振。
        - 【调整】调整 `price_trend_adjustment_factor`，使其在价格下跌时，更合理地调节最终分数。
        - 【新增】重新定义 `disguise_score`，使其能捕捉“价格下跌但主力资金流入”和“价格下跌但整体权力转移为正”
                      这两种更广泛的“诡道”矛盾，从而更全面地识别诡道吸筹。
        """
        method_name = "_calculate_deceptive_accumulation"
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'hidden_accumulation_intensity_D', 'deception_index_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_COHERENT_DRIVE', 'main_force_net_flow_calibrated_D',
            'winner_concentration_90pct_D' # 新增筹码集中度
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['close_D', 'deception_index_D', 'winner_concentration_90pct_D', 'main_force_net_flow_calibrated_D']: # 增加筹码集中度MTF和主力净流MTF
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # --- 原始数据获取 ---
        split_order_accum_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        power_transfer_score = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        coherent_drive_score = self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0)
        main_force_net_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        core_action_score = self._normalize_series(split_order_accum_raw, df_index, bipolar=False)
        # 欺诈证据：使用MTF融合信号
        mtf_deception_evidence = self._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        deception_evidence = mtf_deception_evidence.clip(lower=0) # 确保是正向证据
        # 价格趋势：使用MTF融合信号
        mtf_price_trend_norm = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 筹码集中度趋势：使用MTF融合信号
        mtf_winner_concentration_slope = self._get_mtf_slope_accel_score(df, 'winner_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 主力资金净流：使用MTF融合信号
        mtf_mf_net_flow = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # --- 1. 伪装分 (Disguise Score) ---
        # 捕捉“价格下跌但主力资金流入”和“价格下跌但整体权力转移为正”这两种诡道矛盾
        price_down_strength = mtf_price_trend_norm.clip(upper=0).abs()
        # 诡道矛盾1: 价格下跌但主力资金流入
        disguise_score_price_mf_flow = (price_down_strength * mtf_mf_net_flow.clip(lower=0)).pow(0.5)
        # 诡道矛盾2: 价格下跌但整体权力转移为正
        disguise_score_price_power_transfer = (price_down_strength * power_transfer_score.clip(lower=0)).pow(0.5)
        # 融合两种诡道矛盾
        disguise_score = (disguise_score_price_mf_flow * 0.5 + disguise_score_price_power_transfer * 0.5).clip(0, 1)
        # --- 2. 价格筹码背离共振 (Price-Chip Divergence Resonance) ---
        # 价格下跌的强度 (负向) - 已在 disguise_score 中计算
        # 筹码集中度上升的强度 (正向)
        chip_concentration_up_strength = mtf_winner_concentration_slope.clip(lower=0)
        # 价格下跌与筹码集中的背离共振
        price_chip_divergence_resonance = (price_down_strength * chip_concentration_up_strength).pow(0.5)
        # --- 3. 诡道氛围 (Deceptive Context) ---
        # 结合背离共振、欺诈证据和伪装分
        deceptive_context_score = (
            price_chip_divergence_resonance * 0.4 + # 核心背离共振，权重略降，因为disguise_score更全面
            deception_evidence * 0.3 + # 欺诈指数
            disguise_score * 0.3 # 多维诡道矛盾
        ).clip(0, 1)
        # --- 4. 价格趋势调节因子 (Price Trend Adjustment Factor) ---
        # 在诡道吸筹的语境下，价格下跌是其特征，不应过度惩罚。
        # 但如果价格跌幅过大，也应体现风险。
        # 调整为：当价格趋势为负时，因子为1；当价格趋势为正时，因子为 (1 - mtf_price_trend_norm)
        price_trend_adjustment_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        price_trend_adjustment_factor = price_trend_adjustment_factor.mask(mtf_price_trend_norm > 0, (1 - mtf_price_trend_norm).clip(0, 1))
        # --- 5. 协同惩罚 (Coherence Penalty) ---
        coherence_penalty_factor = (1 - coherent_drive_score.clip(upper=0).abs()).clip(0, 1)
        # --- 6. 最终分数 ---
        final_score = (core_action_score * deceptive_context_score * coherence_penalty_factor * price_trend_adjustment_factor).fillna(0.0)
        return final_score.clip(0, 1).astype(np.float32)

    def _calculate_upthrust_washout(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1 · 强证优先与形态门控版】识别主力利用“上冲回落”阴线进行的洗盘行为。
        - 核心重构: 创立“强证优先”原则。废除对多种承接证据的加权平均，改为采用 max() 函数，
                      直接取“主动买盘”、“下影线强度”、“权力转移”三者中的最强者作为最终承接证据，
                      旨在识别任何一种足以扭转战局的决定性吸收力量。
        - 【新增】严格限制 K 线形态，只有当 K 线为“上冲回落”形态时才激活信号。
        - 【新增】引入主力资金净流向作为判断洗盘真实性的关键证据。
        """
        method_name = "_calculate_upthrust_washout"
        required_signals = [
            'SCORE_STRUCT_AXIOM_TREND_FORM', 'BIAS_21_D', 'pct_change_D',
            'upward_impulse_purity_D', 'upper_shadow_selling_pressure_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 'active_buying_support_D',
            'PROCESS_META_POWER_TRANSFER', 'open_D', 'high_D', 'close_D', 'low_D',
            'main_force_net_flow_calibrated_D' # 新增主力资金净流向
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # --- 原始数据获取 ---
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        bias_21 = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name=method_name)
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name)
        lower_shadow_strength = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        power_transfer = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        open_price = self._get_safe_series(df, 'open_D', 0.0, method_name=method_name)
        high_price = self._get_safe_series(df, 'high_D', 0.0, method_name=method_name)
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        low_price = self._get_safe_series(df, 'low_D', 0.0, method_name=method_name)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        upward_purity_norm = self._normalize_series(upward_purity_raw, df_index, bipolar=False)
        upper_shadow_pressure_norm = self._normalize_series(upper_shadow_pressure_raw, df_index, bipolar=False)
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        power_transfer_norm = power_transfer.clip(lower=0)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        # --- 1. 市场上下文 (Context Mask) ---
        # 趋势向上，乖离率不过高，上涨纯度良好
        context_mask = (trend_form_score > 0.2) & (bias_21 < 0.2) & (upward_purity_norm.rolling(3).mean() > 0.3)
        # --- 2. K线形态门控 (K-line Pattern Gate) ---
        # 严格定义“上冲回落”的 K 线形态：高开低走阴线，或长上影线
        total_range = high_price - low_price
        total_range_safe = total_range.replace(0, 1e-9)
        upper_shadow = high_price - np.maximum(open_price, close_price)
        upper_shadow_ratio = (upper_shadow / total_range_safe).fillna(0)
        # 条件1: 高开低走阴线 (开盘价高于收盘价，且当日下跌)
        is_high_open_low_close_yin = (open_price > close_price) & (pct_change < 0)
        # 条件2: 长上影线 (上影线占比超过一定阈值，且收盘价低于开盘价)
        is_long_upper_shadow_yin = (upper_shadow_ratio > 0.4) & (close_price < open_price)
        # 只有满足这两种形态之一，才认为是“上冲回落”的 K 线
        is_upthrust_kline = is_high_open_low_close_yin | is_long_upper_shadow_yin
        # --- 3. 卖压审判分 (Selling Pressure Score) ---
        is_down_day = (pct_change < 0).astype(float)
        selling_pressure_score = (upper_shadow_pressure_norm * 0.7 + is_down_day * 0.3).clip(0, 1)
        # --- 4. 承接审判分 (Absorption Rebuttal Score) ---
        # 采用“强证优先”原则，取最强的承接证据
        absorption_rebuttal_score = pd.concat([
            active_buying_norm,
            lower_shadow_strength,
            power_transfer_norm
        ], axis=1).max(axis=1)
        # --- 5. 净洗盘意图 (Net Washout Intent) ---
        # 只有当承接力量大于卖压时，才认为是净洗盘意图
        net_washout_intent = (absorption_rebuttal_score - selling_pressure_score).clip(0, 1)
        # --- 6. 主力资金净流入门控 (Main Force Net Inflow Gate) ---
        # 只有当主力资金净流入为正时，才认为洗盘真实
        mf_inflow_gate = main_force_net_flow_norm.clip(lower=0)
        # --- 7. 最终分数 ---
        # 结合市场上下文、K线形态门控和主力资金净流入门控
        final_score = net_washout_intent.where(context_mask & is_upthrust_kline & (mf_inflow_gate > 0.1), 0.0).fillna(0.0) # 0.1为门槛，可调
        self.strategy.atomic_states["_DEBUG_washout_context_mask"] = context_mask
        self.strategy.atomic_states["_DEBUG_washout_is_upthrust_kline"] = is_upthrust_kline
        self.strategy.atomic_states["_DEBUG_washout_mf_inflow_gate"] = mf_inflow_gate
        self.strategy.atomic_states["_DEBUG_washout_selling_pressure_score"] = selling_pressure_score
        self.strategy.atomic_states["_DEBUG_washout_absorption_rebuttal_score"] = absorption_rebuttal_score
        self.strategy.atomic_states["_DEBUG_washout_net_washout_intent"] = net_washout_intent
        return final_score.astype(np.float32)

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4 · 势能衰减与多维融合版】识别多日累积吸筹后，即将由“量变”引发“质变”的拉升拐点。
        - 核心升级: 引入“势能衰减”机制。将累积势能的计算方法从简单的滚动求和(rolling.sum)
                      升级为指数加权移动平均(ewm.mean)，赋予近期吸筹行为更高的权重，
                      更精准地度量具备时效性的“爆发势能”。
        - 【强化】重构 `daily_accumulation_strength`，采用加权几何平均融合多种吸筹信号，
                      更精细地评估综合吸筹强度。
        - 【增强】增强 `ignition_intent_score`，引入更多维度信号来判断“质变”的意图。
        """
        method_name = "_calculate_accumulation_inflection"
        required_signals = [
            'PROCESS_META_STEALTH_ACCUMULATION', 'PROCESS_META_DECEPTIVE_ACCUMULATION',
            'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY',
            'PROCESS_META_POWER_TRANSFER', 'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_PD_DIVERGENCE_CONFIRM',
            'SCORE_CHIP_COHERENT_DRIVE', 'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        accumulation_window = config.get('accumulation_window', 21)
        # 获取吸筹信号
        stealth_accum = self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0)
        deceptive_accum = self._get_atomic_score(df, 'PROCESS_META_DECEPTIVE_ACCUMULATION', 0.0)
        panic_washout_accum = self._get_atomic_score(df, 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 0.0)
        split_order_accum = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0)
        power_transfer_accum = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0) # 权力转移只取正向
        # 获取点火意图信号
        rally_intent = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0)
        divergence_confirm = self._get_atomic_score(df, 'PROCESS_META_PD_DIVERGENCE_CONFIRM', 0.0)
        coherent_drive = self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0)
        capital_signature = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 0.0)
        upward_efficiency = self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.0)
        # --- 1. 重构 `daily_accumulation_strength` (日度吸筹强度) ---
        # 采用加权几何平均融合多种吸筹信号
        accumulation_components = {
            "stealth_accum": stealth_accum,
            "deceptive_accum": deceptive_accum,
            "panic_washout_accum": panic_washout_accum,
            "split_order_accum": split_order_accum,
            "power_transfer_accum": power_transfer_accum
        }
        # 定义吸筹信号的权重 (可配置)
        accumulation_weights = config.get('accumulation_weights', {
            "stealth_accum": 0.25,
            "deceptive_accum": 0.25,
            "panic_washout_accum": 0.2,
            "split_order_accum": 0.15,
            "power_transfer_accum": 0.15
        })
        daily_accumulation_strength = _robust_geometric_mean(accumulation_components, accumulation_weights, df_index)
        # 采用指数加权移动平均(ewm)计算势能，引入时间衰减
        potential_energy_raw = daily_accumulation_strength.ewm(span=accumulation_window, adjust=False, min_periods=5).mean()
        potential_energy_score = self._normalize_series(potential_energy_raw, df_index, bipolar=False) # 确保归一化到 [0, 1]
        # --- 2. 增强 `ignition_intent_score` (点火意图分数) ---
        # 融合主力拉升意图、背离确认、筹码协同驱动、资金属性和上涨效率
        ignition_components = {
            "rally_intent": rally_intent.clip(lower=0), # 只取正向拉升意图
            "divergence_confirm": divergence_confirm.clip(lower=0), # 只取正向背离确认
            "coherent_drive": coherent_drive.clip(lower=0), # 只取正向筹码协同
            "capital_signature": capital_signature.clip(lower=0), # 只取正向资金属性
            "upward_efficiency": upward_efficiency.clip(lower=0) # 只取正向上涨效率
        }
        # 定义点火意图信号的权重 (可配置)
        ignition_weights = config.get('ignition_weights', {
            "rally_intent": 0.3,
            "divergence_confirm": 0.25,
            "coherent_drive": 0.2,
            "capital_signature": 0.15,
            "upward_efficiency": 0.1
        })
        ignition_intent_score = _robust_geometric_mean(ignition_components, ignition_weights, df_index)
        # --- 3. 最终分数 ---
        # 势能与意图的乘积，体现“量变到质变”
        final_score = (potential_energy_score * ignition_intent_score).fillna(0.0)
        # --- 调试信息 ---
        probe_dates = self.probe_dates
        return final_score.clip(0, 1).astype(np.float32)

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 战场扩展版】计算“套牢盘投降”信号。
        - 核心重构: 扩展“战场”定义。战场不再仅限于收盘下跌日，而是扩展为“收盘下跌 或 出现强力下影线吸收”，
                      旨在捕捉经典的“金针探底”反转形态。
        """
        method_name = "_calculate_loser_capitulation"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算套牢盘投降..."] = ""
        required_signals = [
            'pct_change_D', 'capitulation_flow_ratio_D', 'active_buying_support_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        capitulation_flow_raw = self._get_safe_series(df, 'capitulation_flow_ratio_D', 0.0, method_name=method_name)
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        lower_shadow_absorption = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        _temp_debug_values["原始信号值"] = {
            "pct_change_D": pct_change,
            "capitulation_flow_ratio_D": capitulation_flow_raw,
            "active_buying_support_D": active_buying_raw,
            "SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION": lower_shadow_absorption
        }
        # 战场上下文：扩展为“下跌日”或“有强力下影线吸收”
        context_mask = (pct_change < 0) | (lower_shadow_absorption > 0.5)
        _temp_debug_values["战场上下文"] = {
            "context_mask": context_mask
        }
        # 恐慌分：衡量抛售的烈度
        panic_score = self._normalize_series(capitulation_flow_raw, df_index, bipolar=False)
        _temp_debug_values["恐慌分"] = {
            "panic_score": panic_score
        }
        # 吸收分：采用“强证优先”原则，取最强的承接证据
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        absorption_score = pd.concat([active_buying_norm, lower_shadow_absorption], axis=1).max(axis=1)
        _temp_debug_values["吸收分"] = {
            "active_buying_norm": active_buying_norm,
            "absorption_score": absorption_score
        }
        # 最终审判：恐慌与吸收的乘积
        final_score = (panic_score * absorption_score).where(context_mask, 0.0).fillna(0.0)
        _temp_debug_values["最终审判"] = {
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        #     for sig_name, series in _temp_debug_values["原始信号值"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 战场上下文 ---"] = ""
        #     for key, series in _temp_debug_values["战场上下文"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val}"] = "" # Boolean series
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 恐慌分 ---"] = ""
        #     for key, series in _temp_debug_values["恐慌分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 吸收分 ---"] = ""
        #     for key, series in _temp_debug_values["吸收分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终审判 ---"] = ""
        #     for key, series in _temp_debug_values["最终审判"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 套牢盘投降诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return final_score.astype(np.float32)

    def _calculate_cost_advantage_trend_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.4 · 象限审判与全息资金流验证版】计算成本优势趋势。
        - 核心升级: 为各象限的确认环节配备专属的、最高保真度的战术信号，实现“精准审判”。
                      - Q2(派发下跌)引入“利润兑现流量”作为核心证据。
                      - Q4(牛市陷阱)引入“买盘虚弱度”作为核心惩罚项。
        - 【强化】增强“牛市陷阱”识别，引入主力资金净流出和资金流可信度。
        - 【强化】调整 Q1 和 Q3 的确认，引入资金流可信度。
        - 【强化】引入“前置下跌”上下文，在 Q3（黄金坑）中增加其权重。
        - 【新增】将价格变化、成本优势变化升级为多时间维度（MTF）斜率/加速度融合。
        - 【新增】增强 Q1、Q3、Q4 的确认，引入更多 MTF 融合信号。
        """
        method_name = "_calculate_cost_advantage_trend_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算成本优势趋势关系..."] = ""
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'pct_change_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'upward_impulse_purity_D', 'suppressive_accumulation_intensity_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 'distribution_at_peak_intensity_D',
            'active_selling_pressure_D', 'profit_taking_flow_ratio_D', 'active_buying_support_D',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['close_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
                         'upward_impulse_purity_D', 'distribution_at_peak_intensity_D', 'active_selling_pressure_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # --- 原始数据获取 ---
        # 价格变化和成本优势变化改为MTF融合信号
        mtf_price_change = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_ca_change = self._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        main_force_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        upward_impulse_purity = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        suppressive_accum = self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name)
        lower_shadow_absorb = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        distribution_intensity = self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name=method_name)
        active_selling = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name=method_name)
        profit_taking_flow = self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name=method_name)
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name) # 用于计算前置下跌
        _temp_debug_values["原始信号值"] = {
            "mtf_price_change": mtf_price_change,
            "mtf_ca_change": mtf_ca_change,
            "main_force_conviction_index_D": main_force_conviction,
            "upward_impulse_purity_D": upward_impulse_purity,
            "suppressive_accumulation_intensity_D": suppressive_accum,
            "SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION": lower_shadow_absorb,
            "distribution_at_peak_intensity_D": distribution_intensity,
            "active_selling_pressure_D": active_selling,
            "profit_taking_flow_ratio_D": profit_taking_flow,
            "active_buying_support_D": active_buying_support,
            "main_force_net_flow_calibrated_D": main_force_net_flow,
            "flow_credibility_index_D": flow_credibility,
            "close_D": close_price
        }
        # --- 归一化处理 ---
        mtf_main_force_conviction = self._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_upward_purity = self._get_mtf_slope_accel_score(df, 'upward_impulse_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        suppressive_accum_norm = self._normalize_series(suppressive_accum, df_index, bipolar=False)
        mtf_distribution_intensity = self._get_mtf_slope_accel_score(df, 'distribution_at_peak_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_active_selling = self._get_mtf_slope_accel_score(df, 'active_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        profit_taking_flow_norm = self._normalize_series(profit_taking_flow, df_index, bipolar=False)
        active_buying_support_norm = self._normalize_series(active_buying_support, df_index, bipolar=False)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "mtf_main_force_conviction": mtf_main_force_conviction,
            "mtf_upward_purity": mtf_upward_purity,
            "suppressive_accum_norm": suppressive_accum_norm,
            "mtf_distribution_intensity": mtf_distribution_intensity,
            "mtf_active_selling": mtf_active_selling,
            "profit_taking_flow_norm": profit_taking_flow_norm,
            "active_buying_support_norm": active_buying_support_norm,
            "main_force_net_flow_norm": main_force_net_flow_norm,
            "flow_credibility_norm": flow_credibility_norm
        }
        # --- 1. Q1: 价涨 & 优扩 (健康上涨) ---
        Q1_base = (mtf_price_change.clip(lower=0) * mtf_ca_change.clip(lower=0)).pow(0.5)
        # 确认：主力信念、上涨纯度、资金流可信度
        Q1_confirm = (mtf_main_force_conviction.clip(lower=0) * mtf_upward_purity * flow_credibility_norm).pow(1/3)
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        _temp_debug_values["Q1: 价涨 & 优扩"] = {
            "Q1_base": Q1_base,
            "Q1_confirm": Q1_confirm,
            "Q1_final": Q1_final
        }
        # --- 2. Q2: 价跌 & 优缩 (派发下跌) ---
        Q2_base = (mtf_price_change.clip(upper=0).abs() * mtf_ca_change.clip(upper=0).abs()).pow(0.5)
        # 确认：利润兑现流量、主动卖压、行为派发意图 (使用MTF信号)
        Q2_distribution_evidence = (profit_taking_flow_norm * 0.4 + mtf_active_selling * 0.3 + mtf_distribution_intensity * 0.3).clip(0, 1)
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        _temp_debug_values["Q2: 价跌 & 优缩"] = {
            "Q2_base": Q2_base,
            "Q2_distribution_evidence": Q2_distribution_evidence,
            "Q2_final": Q2_final
        }
        # --- 3. Q3: 价跌 & 优扩 (黄金坑) ---
        Q3_base = (mtf_price_change.clip(upper=0).abs() * mtf_ca_change.clip(lower=0)).pow(0.5)
        # 确认：隐蔽吸筹、下影线吸收、资金流可信度
        Q3_confirm = (suppressive_accum_norm * lower_shadow_absorb * flow_credibility_norm).pow(1/3)
        # 前置下跌上下文，如果前几日有深跌，则增加黄金坑的权重
        pre_5day_pct_change = close_price.pct_change(periods=5).shift(1).fillna(0)
        norm_pre_drop_5d = self._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        pre_drop_context_bonus = norm_pre_drop_5d * 0.5
        Q3_final = (Q3_base * Q3_confirm * (1 + pre_drop_context_bonus)).clip(0, 1)
        _temp_debug_values["Q3: 价跌 & 优扩"] = {
            "Q3_base": Q3_base,
            "Q3_confirm": Q3_confirm,
            "pre_5day_pct_change": pre_5day_pct_change,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "pre_drop_context_bonus": pre_drop_context_bonus,
            "Q3_final": Q3_final
        }
        # --- 4. Q4: 价涨 & 优缩 (牛市陷阱) ---
        Q4_base = (mtf_price_change.clip(lower=0) * mtf_ca_change.clip(upper=0).abs()).pow(0.5)
        # 确认：派发强度、买盘虚弱度、主力资金净流出 (使用MTF信号)
        mf_outflow_risk = main_force_net_flow_norm.clip(upper=0).abs() # 主力资金净流出
        Q4_trap_evidence = (mtf_distribution_intensity * 0.4 + (1 - active_buying_support_norm) * 0.3 + mf_outflow_risk * 0.3).clip(0, 1)
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        _temp_debug_values["Q4: 价涨 & 优缩"] = {
            "Q4_base": Q4_base,
            "mf_outflow_risk": mf_outflow_risk,
            "Q4_trap_evidence": Q4_trap_evidence,
            "Q4_final": Q4_final
        }
        # --- 最终融合 ---
        final_score = (Q1_final + Q2_final + Q3_final + Q4_final).clip(-1, 1)
        _temp_debug_values["最终融合"] = {
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                if isinstance(value, dict):
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_series in value.items():
                        val = sub_series.loc[probe_ts] if probe_ts in sub_series.index else np.nan
                        debug_output[f"          {sub_key}: {val:.4f}"] = ""
                else:
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        '{key}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- Q1: 价涨 & 优扩 ---"] = ""
            for key, series in _temp_debug_values["Q1: 价涨 & 优扩"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- Q2: 价跌 & 优缩 ---"] = ""
            for key, series in _temp_debug_values["Q2: 价跌 & 优缩"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- Q3: 价跌 & 优扩 ---"] = ""
            for key, series in _temp_debug_values["Q3: 价跌 & 优扩"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- Q4: 价涨 & 优缩 ---"] = ""
            for key, series in _temp_debug_values["Q4: 价涨 & 优缩"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
            for key, series in _temp_debug_values["最终融合"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 成本优势趋势关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_score.astype(np.float32)

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 共振审判与突破门控版】诊断“突破加速抢筹”战术。
        - 核心重构: 创立“突破即共振”模型。废除硬阈值门槛和几何平均，改为对四大核心证据
                      （突破、意图、资金、结构）进行加权融合，以更鲁棒的“共振分”审判突破质量。
        - 信号修正: 修正了对“拉升意图”信号的引用，确保使用最终权威信号。
        - 【强化】引入主力资金净流入和资金流可信度作为共振分和相对强度调节器的重要组成部分。
        - 【重要修改】强化“突破”的门控作用，当形态层没有确认突破时，大幅惩罚共振分。
        """
        method_name = "_calculate_breakout_acceleration"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算突破加速抢筹..."] = ""
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'SCORE_PATTERN_AXIOM_BREAKOUT', 'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_POWER_TRANSFER', 'SCORE_STRUCT_AXIOM_TREND_FORM',
            'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH',
            'main_force_net_flow_calibrated_D', 'flow_credibility_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        relative_strength = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        # --- 原始数据获取 ---
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "SCORE_PATTERN_AXIOM_BREAKOUT": self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0),
            "PROCESS_META_MAIN_FORCE_RALLY_INTENT": self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0),
            "PROCESS_META_POWER_TRANSFER": self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0),
            "SCORE_STRUCT_AXIOM_TREND_FORM": self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0),
            "SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH": relative_strength,
            "main_force_net_flow_calibrated_D": main_force_net_flow,
            "flow_credibility_index_D": flow_credibility
        }
        # --- 归一化处理 ---
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "main_force_net_flow_norm": main_force_net_flow_norm,
            "flow_credibility_norm": flow_credibility_norm
        }
        # --- 定义四大核心证据 ---
        breakout_evidence = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0)
        intent_evidence = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0)
        flow_evidence = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0)
        structure_evidence = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(lower=0)
        _temp_debug_values["核心证据"] = {
            "breakout_evidence": breakout_evidence,
            "intent_evidence": intent_evidence,
            "flow_evidence": flow_evidence,
            "structure_evidence": structure_evidence
        }
        # --- 增强“共振分”的构成 ---
        mf_flow_validation = main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm
        weights = {'breakout': 0.3, 'intent': 0.25, 'structure': 0.2, 'flow': 0.15, 'mf_flow_validation': 0.1}
        resonance_score = (
            breakout_evidence * weights['breakout'] +
            intent_evidence * weights['intent'] +
            structure_evidence * weights['structure'] +
            flow_evidence * weights['flow'] +
            mf_flow_validation * weights['mf_flow_validation']
        ).clip(0, 1)
        _temp_debug_values["共振分"] = {
            "mf_flow_validation": mf_flow_validation,
            "resonance_score": resonance_score
        }
        # --- 强化“突破”的门控作用 ---
        # 如果突破证据很弱，则大幅惩罚共振分
        breakout_gate_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        breakout_gate_factor = breakout_gate_factor.mask(breakout_evidence < 0.2, breakout_evidence * 0.5) # 突破证据低于0.2时，惩罚50%
        breakout_gate_factor = breakout_gate_factor.mask(breakout_evidence < 0.05, 0.0) # 突破证据极弱时，直接归零
        resonance_score_gated = resonance_score * breakout_gate_factor
        _temp_debug_values["突破门控"] = {
            "breakout_gate_factor": breakout_gate_factor,
            "resonance_score_gated": resonance_score_gated
        }
        # --- 相对强度调节器 ---
        rs_modulator_base = (1 + relative_strength * rs_amplifier)
        mf_flow_modulator = (1 + mf_flow_validation * 0.5)
        rs_modulator = rs_modulator_base * mf_flow_modulator
        final_score = (resonance_score_gated * rs_modulator).clip(0, 1).fillna(0.0)
        _temp_debug_values["相对强度调节器与最终分数"] = {
            "rs_modulator_base": rs_modulator_base,
            "mf_flow_modulator": mf_flow_modulator,
            "rs_modulator": rs_modulator,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 核心证据 ---"] = ""
            for key, series in _temp_debug_values["核心证据"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 共振分 ---"] = ""
            for key, series in _temp_debug_values["共振分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 突破门控 ---"] = ""
            for key, series in _temp_debug_values["突破门控"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 相对强度调节器与最终分数 ---"] = ""
            for key, series in _temp_debug_values["相对强度调节器与最终分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 突破加速抢筹诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_score.astype(np.float32)

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.7 · 战术升级与全息资金流验证强化版】识别主力从隐蔽吸筹转向公开强攻的转折信号。
        - 核心重构: 废除僵化的“AND”门槛，创立“战术评分”模型。最终分 = 前奏吸筹分 * 强攻分。
        - 证据升级: “前奏分”通过归一化消除尺度问题；“强攻分”对核心证据进行加权，更具实战性。
        - 【强化】引入资金流可信度，确保强攻的资金基础是可靠的。
        - 【强化】调整“前奏分”的计算，引入主力资金净流入的趋势，确保前奏吸筹的持续性。
        - 【强化】将 `hidden_accumulation_intensity_D`、`buy_quote_exhaustion_rate_D` 和 `large_order_pressure_D` 升级为 MTF 融合信号。
        - 【重要修正】重新定义 `pressure_exhaustion_synergy`，衡量“大单压力与买盘枯竭的协同性”，并增加其惩罚权重。
        - 【新增】引入 `active_buying_support_D` 的 MTF 融合版本，作为强攻的直接证据。
        - 【新增】将 `main_force_flow_momentum` 升级为 MTF 融合版本。
        - 【修正】调整 `attack_score` 中 `pressure_exhaustion_synergy` 的惩罚项，使其更符合逻辑。
        """
        method_name = "_calculate_fund_flow_accumulation_inflection"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算资金流吸筹拐点意图..."] = ""
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'hidden_accumulation_intensity_D', 'main_force_net_flow_calibrated_D',
            'buy_quote_exhaustion_rate_D', 'large_order_pressure_D',
            'flow_credibility_index_D', 'active_buying_support_D' # 新增 active_buying_support_D
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['hidden_accumulation_intensity_D', 'buy_quote_exhaustion_rate_D', 
                         'large_order_pressure_D', 'main_force_net_flow_calibrated_D',
                         'active_buying_support_D']: # 增加 active_buying_support_D 的MTF
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        # 获取原料
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "hidden_accumulation_intensity_D": self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name),
            "main_force_net_flow_calibrated_D": self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name),
            "buy_quote_exhaustion_rate_D": self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name=method_name),
            "large_order_pressure_D": self._get_safe_series(df, 'large_order_pressure_D', 0.0, method_name=method_name),
            "flow_credibility_index_D": flow_credibility_raw,
            "active_buying_support_D": self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        }
        # --- 归一化处理 ---
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "flow_credibility_norm": flow_credibility_norm
        }
        # --- 1. 重铸“前奏分”，消除尺度问题并引入主力资金净流入趋势 ---
        mtf_hidden_accumulation = self._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_net_flow_slope = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        prelude_score_base = self._normalize_series(mtf_hidden_accumulation.rolling(5).mean(), df_index, bipolar=False)
        # 确保主力资金净流入趋势为正向才贡献
        prelude_score = (prelude_score_base * mtf_mf_net_flow_slope.clip(lower=0)).pow(0.5)
        _temp_debug_values["前奏分"] = {
            "mtf_hidden_accumulation": mtf_hidden_accumulation,
            "mtf_mf_net_flow_slope": mtf_mf_net_flow_slope,
            "prelude_score_base": prelude_score_base,
            "prelude_score": prelude_score
        }
        # --- 2. 重铸“强攻分”，采用加权模型并引入资金流可信度 ---
        mtf_buy_exhaustion = self._get_mtf_slope_accel_score(df, 'buy_quote_exhaustion_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_large_pressure = self._get_mtf_slope_accel_score(df, 'large_order_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_active_buying_support = self._get_mtf_slope_accel_score(df, 'active_buying_support_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False) # 新增
        # 重新定义 pressure_exhaustion_synergy：大单压力与买盘枯竭的协同性，两者都高时应惩罚
        # 当大单压力和买盘枯竭都高时，协同性高，意味着强攻阻力大，应惩罚
        pressure_exhaustion_synergy = (mtf_large_pressure * mtf_buy_exhaustion).pow(0.5)
        # 主力资金流向动量：使用MTF融合信号
        mtf_main_force_flow_momentum = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        attack_score = (
            (1 - mtf_buy_exhaustion) * 0.2 + # 买盘韧性，枯竭度越低越好
            mtf_main_force_flow_momentum.clip(lower=0) * 0.25 + # 主力资金流入加速
            (1 - mtf_large_pressure) * 0.2 + # 压力清除，大单压力越低越好
            flow_credibility_norm * 0.15 + # 资金流可信度
            mtf_active_buying_support * 0.2 - # 主动买盘支持
            pressure_exhaustion_synergy * 0.2 # 惩罚大单压力与买盘枯竭的协同性，权重增加
        ).clip(0, 1) # 确保分数在 [0, 1] 之间
        _temp_debug_values["强攻分"] = {
            "mtf_buy_exhaustion": mtf_buy_exhaustion,
            "mtf_large_pressure": mtf_large_pressure,
            "mtf_main_force_flow": mtf_main_force_flow,
            "mtf_active_buying_support": mtf_active_buying_support,
            "pressure_exhaustion_synergy": pressure_exhaustion_synergy,
            "mtf_main_force_flow_momentum": mtf_main_force_flow_momentum,
            "attack_score": attack_score
        }
        # --- 3. 最终审判 ---
        final_score = (prelude_score * attack_score).fillna(0.0)
        _temp_debug_values["最终审判"] = {
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        #     for key, value in _temp_debug_values["原始信号值"].items():
        #         val = value.loc[probe_ts] if probe_ts in value.index else np.nan
        #         debug_output[f"        '{key}': {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
        #     for key, series in _temp_debug_values["归一化处理"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 前奏分 ---"] = ""
        #     for key, series in _temp_debug_values["前奏分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 强攻分 ---"] = ""
        #     for key, series in _temp_debug_values["强攻分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终审判 ---"] = ""
        #     for key, series in _temp_debug_values["最终审判"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流吸筹拐点意图诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return final_score.astype(np.float32)

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.3 · 战场态势与多维压力版】“利润与流向”专属关系计算引擎
        - 核心重构: 创立“战场态势”审判模型，从比较“动量”升维为比较力量的“当前水平”。
        - 信号升级: 将核心“压力”信号从“T0效率”升级为更精准的“利润兑现流量占比”。
        - 【新增】增强“派发压力”的判断，引入赢家平均利润率和行为派发意图。
        - 【新增】调整“建仓动力”的判断，引入资金流可信度。
        """
        method_name = "_calculate_profit_vs_flow_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算主力盈亏vs流量关系..."] = ""
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        drive_signal_name = 'main_force_net_flow_calibrated_D'
        winner_profit_margin_name = 'winner_profit_margin_avg_D' # 新增
        distribution_intent_name = 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT' # 新增
        flow_credibility_name = 'flow_credibility_index_D' # 新增
        required_signals = [pressure_signal_name, drive_signal_name, winner_profit_margin_name, flow_credibility_name]
        required_atomic_signals = [distribution_intent_name]
        all_required_signals = required_signals + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(dtype=np.float32)
        df_index = df.index
        # 获取原始数据
        profit_taking_flow_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name=method_name)
        main_force_net_flow_raw = self._get_safe_series(df, drive_signal_name, 0.0, method_name=method_name)
        winner_profit_margin_raw = self._get_safe_series(df, winner_profit_margin_name, 0.0, method_name=method_name)
        distribution_intent_score = self._get_atomic_score(df, distribution_intent_name, 0.0)
        flow_credibility_raw = self._get_safe_series(df, flow_credibility_name, 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "profit_taking_flow_ratio_D": profit_taking_flow_raw,
            "main_force_net_flow_calibrated_D": main_force_net_flow_raw,
            "winner_profit_margin_avg_D": winner_profit_margin_raw,
            "SCORE_BEHAVIOR_DISTRIBUTION_INTENT": distribution_intent_score,
            "flow_credibility_index_D": flow_credibility_raw
        }
        # --- 归一化处理 ---
        profit_taking_flow_norm = self._normalize_series(profit_taking_flow_raw, df_index, bipolar=False)
        winner_profit_margin_norm = self._normalize_series(winner_profit_margin_raw, df_index, bipolar=False)
        distribution_intent_norm = self._normalize_series(distribution_intent_score, df_index, bipolar=False)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow_raw, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "profit_taking_flow_norm": profit_taking_flow_norm,
            "winner_profit_margin_norm": winner_profit_margin_norm,
            "distribution_intent_norm": distribution_intent_norm,
            "main_force_net_flow_norm": main_force_net_flow_norm,
            "flow_credibility_norm": flow_credibility_norm
        }
        # --- 1. 派发压力分 (Pressure Score) ---
        # 结合利润兑现流量、赢家平均利润率和行为派发意图
        pressure_score = (
            profit_taking_flow_norm * 0.4 +
            winner_profit_margin_norm * 0.3 +
            distribution_intent_norm * 0.3
        ).clip(0, 1)
        _temp_debug_values["派发压力分"] = {
            "pressure_score": pressure_score
        }
        # --- 2. 建仓动力分 (Drive Score) ---
        # 结合主力资金净流向和资金流可信度
        drive_score = (main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm).clip(0, 1)
        _temp_debug_values["建仓动力分"] = {
            "drive_score": drive_score
        }
        # --- 3. 核心逻辑：战场态势对抗 ---
        relationship_score = drive_score - pressure_score
        final_score = relationship_score.clip(-1, 1)
        _temp_debug_values["最终分数"] = {
            "relationship_score": relationship_score,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        #     for key, value in _temp_debug_values["原始信号值"].items():
        #         val = value.loc[probe_ts] if probe_ts in value.index else np.nan
        #         debug_output[f"        '{key}': {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
        #     for key, series in _temp_debug_values["归一化处理"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派发压力分 ---"] = ""
        #     for key, series in _temp_debug_values["派发压力分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 建仓动力分 ---"] = ""
        #     for key, series in _temp_debug_values["建仓动力分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终分数 ---"] = ""
        #     for key, series in _temp_debug_values["最终分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力盈亏vs流量关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return final_score.astype(np.float32)

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 协同共振与全息资金流验证强化版】“个股板块协同共振”专属关系计算引擎
        - 核心重构: 创立“协同共振模型”，明确区分看涨和看跌协同，并融入板块动量。
        - 信号升级: 个股强度由 `pct_change_D` 直接衡量，板块强度由 `industry_strength_rank_D` 衡量，
                      并新增 `SLOPE_5_industry_strength_rank_D` 捕捉板块动量。
        - 【强化】引入主力资金净流向和资金流可信度作为判断看涨协同真实性的关键证据。
        - 【重要修改】强化板块动量的门控作用，只有当板块动量为正时，才允许个股的看涨协同被放大。
        - 【新增】引入 MTF 板块动量和 MTF 行业强度排名，增强信号鲁棒性。
        """
        method_name = "_calculate_stock_sector_sync"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算个股板块协同共振..."] = ""
        stock_signal_name = 'pct_change_D'
        sector_rank_name = 'industry_strength_rank_D'
        sector_momentum_name = 'SLOPE_5_industry_strength_rank_D'
        main_force_net_flow_name = 'main_force_net_flow_calibrated_D'
        flow_credibility_name = 'flow_credibility_index_D'
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [stock_signal_name, sector_rank_name, sector_momentum_name, main_force_net_flow_name, flow_credibility_name]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['industry_strength_rank_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 获取原始数据
        stock_signal_raw = self._get_safe_series(df, stock_signal_name, 0.0, method_name=method_name)
        sector_rank_raw = self._get_safe_series(df, sector_rank_name, 0.0, method_name=method_name)
        sector_momentum_raw = self._get_safe_series(df, sector_momentum_name, 0.0, method_name=method_name)
        main_force_net_flow_raw = self._get_safe_series(df, main_force_net_flow_name, 0.0, method_name=method_name)
        flow_credibility_raw = self._get_safe_series(df, flow_credibility_name, 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "pct_change_D": stock_signal_raw,
            "industry_strength_rank_D": sector_rank_raw,
            "SLOPE_5_industry_strength_rank_D": sector_momentum_raw,
            "main_force_net_flow_calibrated_D": main_force_net_flow_raw,
            "flow_credibility_index_D": flow_credibility_raw
        }
        # 归一化当前状态和动量
        stock_strength_score = self._normalize_series(stock_signal_raw, target_index=df_index, bipolar=True)
        # 行业强度排名：使用MTF融合信号
        mtf_sector_rank_score = self._get_mtf_slope_accel_score(df, 'industry_strength_rank_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 板块动量：使用MTF融合信号
        mtf_sector_momentum_score = self._get_mtf_slope_accel_score(df, 'industry_strength_rank_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow_raw, target_index=df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, target_index=df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "stock_strength_score": stock_strength_score,
            "mtf_sector_rank_score": mtf_sector_rank_score,
            "mtf_sector_momentum_score": mtf_sector_momentum_score,
            "main_force_net_flow_norm": main_force_net_flow_norm,
            "flow_credibility_norm": flow_credibility_norm
        }
        # --- 看涨协同部分 (Bullish Synchronicity) ---
        # 1. 提取个股正向运动
        bullish_stock_movement = stock_strength_score.clip(lower=0)
        # 2. 提取板块正向强度和正向动量
        sector_strength_context_bullish = mtf_sector_rank_score # 使用MTF行业强度排名
        sector_momentum_context_bullish = mtf_sector_momentum_score.clip(lower=0)
        # 3. 计算板块看涨共振因子 (几何平均，要求两者都为正才高)
        bullish_sector_resonance = (sector_strength_context_bullish * sector_momentum_context_bullish).pow(0.5).fillna(0.0)
        # 4. 引入主力资金净流入和资金流可信度作为看涨放大因子
        mf_inflow_factor = main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm
        # 强化板块动量的门控作用：只有当板块动量为正且达到一定阈值时，才允许个股的看涨协同被放大
        bullish_amplification_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        bullish_amplification_factor = bullish_amplification_factor.mask(sector_momentum_context_bullish > 0.1, 1 + bullish_sector_resonance * mf_inflow_factor)
        bullish_amplification_factor = bullish_amplification_factor.mask(sector_momentum_context_bullish <= 0.1, 0.0) # 板块动量不足时，直接归零
        # 5. 最终看涨协同分数
        final_bullish_score = bullish_stock_movement * bullish_amplification_factor
        _temp_debug_values["看涨协同部分"] = {
            "bullish_stock_movement": bullish_stock_movement,
            "sector_strength_context_bullish": sector_strength_context_bullish,
            "sector_momentum_context_bullish": sector_momentum_context_bullish,
            "bullish_sector_resonance": bullish_sector_resonance,
            "mf_inflow_factor": mf_inflow_factor,
            "bullish_amplification_factor": bullish_amplification_factor,
            "final_bullish_score": final_bullish_score
        }
        # --- 看跌协同部分 (Bearish Synchronicity) ---
        # 1. 提取个股负向运动的绝对值
        bearish_stock_movement = stock_strength_score.clip(upper=0).abs()
        # 2. 提取板块负向强度和负向动量
        sector_weakness_context_bearish = (1 - mtf_sector_rank_score) # 使用MTF行业强度排名
        sector_negative_momentum_context_bearish = mtf_sector_momentum_score.clip(upper=0).abs()
        # 3. 计算板块看跌共振因子 (几何平均)
        bearish_sector_resonance = (sector_weakness_context_bearish * sector_negative_momentum_context_bearish).pow(0.5).fillna(0.0)
        # 4. 计算看跌放大因子
        bearish_amplification_factor = 1 + bearish_sector_resonance
        # 5. 最终看跌协同分数 (转换为负分)
        final_bearish_score = bearish_stock_movement * bearish_amplification_factor * -1
        _temp_debug_values["看跌协同部分"] = {
            "bearish_stock_movement": bearish_stock_movement,
            "sector_weakness_context_bearish": sector_weakness_context_bearish,
            "sector_negative_momentum_context_bearish": sector_negative_momentum_context_bearish,
            "bearish_sector_resonance": bearish_sector_resonance,
            "bearish_amplification_factor": bearish_amplification_factor,
            "final_bearish_score": final_bearish_score
        }
        # --- 最终融合 ---
        relationship_score = final_bullish_score + final_bearish_score
        final_score = relationship_score.clip(-1, 1)
        _temp_debug_values["最终融合"] = {
            "relationship_score": relationship_score,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        #     for key, value in _temp_debug_values["原始信号值"].items():
        #         val = value.loc[probe_ts] if probe_ts in value.index else np.nan
        #         debug_output[f"        '{key}': {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
        #     for key, series in _temp_debug_values["归一化处理"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看涨协同部分 ---"] = ""
        #     for key, series in _temp_debug_values["看涨协同部分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看跌协同部分 ---"] = ""
        #     for key, series in _temp_debug_values["看跌协同部分"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
        #     for key, series in _temp_debug_values["最终融合"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 个股板块协同共振诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return final_score.astype(np.float32)

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1 · 军令直达版】“热门板块冷却”专属关系计算引擎
        - 核心重构: 创立“状态与方向”乘积模型，审判“高位下的资金背叛”。
        - 信号升级: 资金信号升级为更具意图的 `main_force_net_flow_calibrated_D`。
        - 核心逻辑: 瞬时关系分 = 板块热度(状态分) * 主力出逃(方向分)。
        """
        method_name = "_calculate_hot_sector_cooling"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算热门板块冷却..."] = ""
        hotness_signal_name = 'THEME_HOTNESS_SCORE_D'
        flow_signal_name = 'main_force_net_flow_calibrated_D'
        required_signals = [hotness_signal_name, flow_signal_name]
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(dtype=np.float32)
        df_index = df.index
        hotness_signal_raw = self._get_safe_series(df, hotness_signal_name, 0.0, method_name=method_name)
        flow_signal_raw = self._get_safe_series(df, flow_signal_name, 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "THEME_HOTNESS_SCORE_D": hotness_signal_raw,
            "main_force_net_flow_calibrated_D": flow_signal_raw
        }
        # 归一化状态与方向
        hotness_state_score = self._normalize_series(hotness_signal_raw, df_index, bipolar=False)
        flow_direction_score = self._normalize_series(flow_signal_raw, df_index, bipolar=True)
        # 只关注主力出逃的部分
        outflow_score = flow_direction_score.clip(upper=0).abs()
        _temp_debug_values["归一化处理"] = {
            "hotness_state_score": hotness_state_score,
            "flow_direction_score": flow_direction_score,
            "outflow_score": outflow_score
        }
        # 核心逻辑：寒潮来袭模型
        relationship_score = hotness_state_score * outflow_score
        final_score = relationship_score.clip(0, 1) # 这是一个单极风险信号
        _temp_debug_values["最终分数"] = {
            "relationship_score": relationship_score,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        #     for key, series in _temp_debug_values["原始信号值"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        '{key}': {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
        #     for key, series in _temp_debug_values["归一化处理"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终分数 ---"] = ""
        #     for key, series in _temp_debug_values["最终分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 热门板块冷却诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 调用修正版】计算“价资关系”的专属方法。
        - 核心修复: 修正了对 `_perform_meta_analysis_on_score` 的调用，确保传递了完整的 `df` 和 `df.index`。
        """
        method_name = "_calculate_pf_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价资关系..."] = ""
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 关系分数为空，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        _temp_debug_values["关系分数"] = {
            "relationship_score": relationship_score
        }
        # 修正调用参数，同时传递 df 和 df.index
        meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df.index)
        _temp_debug_values["元分析分数"] = {
            "meta_score": meta_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 关系分数 ---"] = ""
        #     for key, series in _temp_debug_values["关系分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 元分析分数 ---"] = ""
        #     for key, series in _temp_debug_values["元分析分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价资关系诊断完成，最终分值: {meta_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
        return meta_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 调用修正版】计算“价筹关系”的专属方法。
        - 核心修复: 修正了对 `_perform_meta_analysis_on_score` 的调用，确保传递了完整的 `df` 和 `df.index`。
        """
        method_name = "_calculate_pc_relationship"
        # --- 调试信息构建 ---
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_output = {}
        _temp_debug_values = {} # 临时存储所有中间计算结果的原始值 (无条件收集)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价筹关系..."] = ""
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 关系分数为空，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        _temp_debug_values["关系分数"] = {
            "relationship_score": relationship_score
        }
        # 修正调用参数，同时传递 df 和 df.index
        meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df.index)
        _temp_debug_values["元分析分数"] = {
            "meta_score": meta_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 关系分数 ---"] = ""
            for key, series in _temp_debug_values["关系分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 元分析分数 ---"] = ""
            for key, series in _temp_debug_values["元分析分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价筹关系诊断完成，最终分值: {meta_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return meta_score

    def _perform_meta_analysis_on_score(self, relationship_score: pd.Series, config: Dict, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """
        【V1.4 · 参数名修正版】可复用的元分析核心引擎。
        - 核心升级: 新增 `df` 参数，接收完整的DataFrame。
        - 核心修复: 修正了“门控元分析”逻辑，使其从 `df` 而非临时的 `relationship_score.to_frame()`
                      中获取 `close_D` 和均线数据，彻底解决了数据缺失的警告。
        """
        signal_name = config.get('name')
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0)
        bipolar_displacement_strength = self._normalize_series(relationship_displacement, df_index, bipolar=True)
        bipolar_momentum_strength = self._normalize_series(relationship_momentum, df_index, bipolar=True)
        instant_score_normalized = (relationship_score + 1) / 2
        weight_momentum = (1 - instant_score_normalized).clip(0, 1)
        weight_displacement = 1 - weight_momentum
        meta_score = (bipolar_displacement_strength * weight_displacement + bipolar_momentum_strength * weight_momentum)
        diagnosis_mode = config.get('diagnosis_mode', 'meta_analysis')
        if diagnosis_mode == 'gated_meta_analysis':
            gate_condition_config = config.get('gate_condition', {})
            gate_type = gate_condition_config.get('type')
            gate_is_open = pd.Series(True, index=df_index)
            if gate_type == 'price_vs_ma':
                ma_period = gate_condition_config.get('ma_period', 5)
                ma_series = self._get_safe_series(df, f'EMA_{ma_period}_D', method_name="_perform_meta_analysis_on_score")
                if ma_series is not None:
                    close_series = self._get_safe_series(df, 'close_D', method_name="_perform_meta_analysis_on_score")
                    gate_is_open = close_series < ma_series
            meta_score = meta_score * gate_is_open.astype(float)
        signal_meta = self.score_type_map.get(signal_name, {})
        scoring_mode = signal_meta.get('scoring_mode', 'unipolar')
        if scoring_mode == 'unipolar':
            meta_score = meta_score.clip(lower=0)
        return meta_score.clip(-1, 1).astype(np.float32)

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4 · 参数名修正版】计算通用的瞬时关系分数。
        - 核心升级: 增加对 `enable_probe` 配置项的检查，实现探针输出的可配置化。
        """
        signal_name = config.get('name')
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        df_index = df.index
        relationship_type = config.get('relationship_type', 'consensus')
        def get_signal_series(signal_name: str, source_type: str) -> Optional[pd.Series]:
            series = None
            if source_type == 'atomic_states':
                series = self.strategy.atomic_states.get(signal_name)
            else:
                series = self._get_safe_series(df, signal_name, method_name="_calculate_instantaneous_relationship")
            if series is None:
                print(f"        -> [过程层警告] 依赖信号 '{signal_name}' (来源: {source_type}) 不存在，无法计算关系。")
            return series
        signal_a = get_signal_series(config.get('signal_A'), config.get('source_A', 'df'))
        signal_b = get_signal_series(config.get('signal_B'), config.get('source_B', 'df'))
        if signal_a is None or signal_b is None:
            return pd.Series(dtype=np.float32)
        def get_change_series(series: pd.Series, change_type: str) -> pd.Series:
            if change_type == 'diff':
                return series.diff(1).fillna(0)
            return ta.percent_return(series, length=1).fillna(0)
        change_a = get_change_series(signal_a, config.get('change_type_A', 'pct'))
        change_b = get_change_series(signal_b, config.get('change_type_B', 'pct'))
        momentum_a = self._normalize_series(change_a, df_index, bipolar=True)
        thrust_b = self._normalize_series(change_b, df_index, bipolar=True)
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        if relationship_type == 'divergence':
            relationship_score = (signal_b_factor_k * thrust_b - momentum_a) / (signal_b_factor_k + 1)
        else:
            force_vector_sum = momentum_a + signal_b_factor_k * thrust_b
            magnitude = (momentum_a.abs() * thrust_b.abs()).pow(0.5)
            relationship_score = np.sign(force_vector_sum) * magnitude
        relationship_score = relationship_score.clip(-1, 1).fillna(0.0)
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _calculate_ff_vs_structure_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 势能加权版】计算“资金与结构”的专属瞬时关系分。
        - 核心升级: 引入由 `SCORE_STRUCT_AXIOM_TREND_FORM` 绝对值构成的“战略态势放大器”，
                      对基础背离分进行加权，使得在强趋势背景下的背离信号更具影响力。
        """
        required_signals = ['SCORE_STRUCT_AXIOM_TREND_FORM', 'SCORE_FF_AXIOM_CONSENSUS']
        if not self._validate_required_signals(df, required_signals, "_calculate_ff_vs_structure_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        # 引入战略态势放大器
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        strategic_context_amplifier = 1 + trend_form_score.abs()
        final_score = (base_divergence_score * strategic_context_amplifier).clip(-1, 1)
        return final_score

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 派发审判版】计算“动能与筹码”的专属瞬时关系分。
        - 核心升级: 引入基于 `winner_profit_margin_avg_D` 的“派发压力因子”，对负向共识分
                      进行“动机审判”。当人心溃散伴随高额浮盈时，看跌信号将被加重。
        """
        required_signals = ['SCORE_DYN_AXIOM_MOMENTUM', 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'winner_profit_margin_avg_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_dyn_vs_chip_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 计算基础共识分数
        base_consensus_score = self._calculate_instantaneous_relationship(df, config)
        # 引入派发压力因子进行动机审判
        profit_margin = self._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name="_calculate_dyn_vs_chip_relationship")
        profit_margin_norm = self._normalize_series(profit_margin, df.index, bipolar=False)
        distribution_pressure_factor = 1 + profit_margin_norm
        # 仅当基础分为负（内部分裂）时，才进行动机审判
        final_score = base_consensus_score.where(
            base_consensus_score >= 0,
            base_consensus_score * distribution_pressure_factor
        ).clip(-1, 1)
        return final_score

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series, config: Dict) -> pd.Series:
        """
        【V2.5 · 深度情境与多维洗盘反弹版】识别主力利用洗盘后进行反弹的信号。
        - 核心升级: 优化 `deception_context_score` 中的欺诈信号融合，更侧重于诱空欺诈。
        - 【强化】调整 `panic_depth_weights` 中恐慌加速的权重，以更强调恐慌加速的重要性。
        - 【调整】优化 `rebound_quality_weights` 中的吸收信号，避免重复并更清晰地表达吸收意图和强度。
        """
        method_name = "_calculate_process_wash_out_rebound"
        df_index = df.index
        # 直接使用 self.params，因为它已在 __init__ 中加载了 process_intelligence_params
        p_conf = self.params
        params = get_param_value(p_conf.get('wash_out_rebound_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"deception_context": 0.3, "panic_depth": 0.3, "rebound_quality": 0.4})
        # 修改代码：新增更多权重配置
        deception_context_weights = get_param_value(params.get('deception_context_weights'), {"wash_trade": 0.2, "active_selling": 0.15, "deception_lure_short": 0.2, "behavior_deception_index": 0.2, "stealth_ops": 0.15, "wash_trade_slope": 0.05, "active_selling_slope": 0.05})
        panic_depth_weights = get_param_value(params.get('panic_depth_weights'), {"panic_cascade": 0.2, "retail_surrender": 0.2, "loser_pain": 0.2, "holder_sentiment_inverted": 0.15, "sentiment_pendulum_negative": 0.15, "retail_surrender_slope": 0.075, "loser_pain_slope": 0.075}) # 提高权重
        rebound_quality_weights = get_param_value(params.get('rebound_quality_weights'), {"closing_strength": 0.15, "upward_purity": 0.15, "absorption_strength": 0.3, "offensive_absorption": 0.25, "mf_buy_execution_alpha": 0.05, "buy_sweep_intensity": 0.1})
        context_amplification_weights = get_param_value(params.get('context_amplification_weights'), {"trend_form": 0.4, "stability": 0.3, "tension": 0.15, "mtf_cohesion": 0.15})
        max_context_bonus_factor = get_param_value(params.get('max_context_bonus_factor'), 0.5)
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 从 config 中获取 mtf_slope_accel_weights
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        # --- 2. 获取所有原始数据和原子信号 ---
        required_signals = [
            'wash_trade_intensity_D', 'deception_index_D', 'active_selling_pressure_D',
            'panic_selling_cascade_D', 'retail_panic_surrender_index_D', 'loser_pain_index_D',
            'closing_strength_index_D', 'upward_impulse_purity_D',
            'SCORE_STRUCT_AXIOM_TREND_FORM', 'SCORE_STRUCT_AXIOM_STABILITY',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            # 修改代码：新增依赖信号
            'SCORE_BEHAVIOR_DECEPTION_INDEX', 'SCORE_MICRO_STRATEGY_STEALTH_OPS',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT',
            'main_force_buy_execution_alpha_D', 'buy_sweep_intensity_D',
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_STRUCT_AXIOM_MTF_COHESION'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['wash_trade_intensity_D', 'active_selling_pressure_D',
                         'retail_panic_surrender_index_D', 'loser_pain_index_D',
                         'deception_lure_short_intensity_D']: # 新增诱空欺诈的MTF
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] _calculate_process_wash_out_rebound 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        # Raw data
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        active_selling_raw = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name=method_name)
        panic_cascade_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name=method_name)
        retail_surrender_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        closing_strength_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name=method_name)
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name)
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        stability_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        # 修改代码：新增获取信号
        behavior_deception_index = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0)
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        holder_sentiment_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        absorption_strength_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 0.0)
        offensive_absorption_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
        mf_buy_execution_alpha_raw = self._get_safe_series(df, 'main_force_buy_execution_alpha_D', 0.0, method_name=method_name)
        buy_sweep_intensity_raw = self._get_safe_series(df, 'buy_sweep_intensity_D', 0.0, method_name=method_name)
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        mtf_cohesion_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0)
        # --- 3. 维度一：洗盘诱空背景 (Wash-out Deception Context) ---
        mtf_wash_trade_score = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_active_selling_score = self._get_mtf_slope_accel_score(df, 'active_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        # 直接使用 deception_lure_short_intensity_D 的MTF融合版本作为诱空欺诈证据
        mtf_deception_lure_short_score = self._get_mtf_slope_accel_score(df, 'deception_lure_short_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        behavior_deception_score_negative = behavior_deception_index.clip(upper=0).abs() # 负向欺骗
        stealth_ops_normalized = self._normalize_series(stealth_ops_score, df_index, bipolar=False)
        mtf_wash_trade_slope_score = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False) # 增加的洗盘强度斜率
        mtf_active_selling_slope_score = self._get_mtf_slope_accel_score(df, 'active_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False) # 增加的主动卖压斜率
        deception_context_score = (
            (mtf_wash_trade_score).pow(deception_context_weights.get('wash_trade', 0.2)) *
            (mtf_active_selling_score).pow(deception_context_weights.get('active_selling', 0.15)) *
            (mtf_deception_lure_short_score).pow(deception_context_weights.get('deception_lure_short', 0.2)) * # 修正为直接使用诱空欺诈
            (behavior_deception_score_negative).pow(deception_context_weights.get('behavior_deception_index', 0.2)) *
            (stealth_ops_normalized).pow(deception_context_weights.get('stealth_ops', 0.15)) *
            (mtf_wash_trade_slope_score).pow(deception_context_weights.get('wash_trade_slope', 0.05)) *
            (mtf_active_selling_slope_score).pow(deception_context_weights.get('active_selling_slope', 0.05))
        ).pow(1/sum(deception_context_weights.values())).fillna(0.0)
        # --- 4. 维度二：恐慌割肉深度 (Panic Capitulation Depth) ---
        panic_cascade_score = self._normalize_series(panic_cascade_raw, df_index, bipolar=False)
        mtf_retail_surrender_score = self._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_loser_pain_score = self._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        holder_sentiment_inverted_score = (1 - holder_sentiment_score).clip(0, 1) # 低信念韧性
        sentiment_pendulum_negative_score = sentiment_pendulum_score.clip(upper=0).abs() # 负向情绪
        mtf_retail_surrender_slope_score = self._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False) # 恐慌加速
        mtf_loser_pain_slope_score = self._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False) # 亏损痛苦加速
        panic_depth_score = (
            (panic_cascade_score).pow(panic_depth_weights.get('panic_cascade', 0.2)) *
            (mtf_retail_surrender_score).pow(panic_depth_weights.get('retail_surrender', 0.2)) *
            (mtf_loser_pain_score).pow(panic_depth_weights.get('loser_pain', 0.2)) *
            (holder_sentiment_inverted_score).pow(panic_depth_weights.get('holder_sentiment_inverted', 0.15)) *
            (sentiment_pendulum_negative_score).pow(panic_depth_weights.get('sentiment_pendulum_negative', 0.15)) *
            (mtf_retail_surrender_slope_score).pow(panic_depth_weights.get('retail_surrender_slope', 0.075)) * # 提高权重
            (mtf_loser_pain_slope_score).pow(panic_depth_weights.get('loser_pain_slope', 0.075)) # 提高权重
        ).pow(1/sum(panic_depth_weights.values())).fillna(0.0)
        # --- 5. 维度三：承接反弹品质 (Absorption Rebound Quality) ---
        # 移除 offensive_absorption_intent，将其权重分配给 offensive_absorption_normalized 和 absorption_strength_normalized
        closing_strength_score = normalize_score(closing_strength_raw, df_index, windows=55)
        upward_purity_score = self._normalize_series(upward_purity_raw, df_index, bipolar=False)
        absorption_strength_normalized = self._normalize_series(absorption_strength_score, df_index, bipolar=False)
        offensive_absorption_normalized = self._normalize_series(offensive_absorption_score, df_index, bipolar=False)
        mf_buy_execution_alpha_score = self._normalize_series(mf_buy_execution_alpha_raw, df_index, bipolar=False)
        buy_sweep_intensity_score = self._normalize_series(buy_sweep_intensity_raw, df_index, bipolar=False)
        rebound_quality_score = (
            (closing_strength_score).pow(rebound_quality_weights.get('closing_strength', 0.15)) *
            (upward_purity_score).pow(rebound_quality_weights.get('upward_purity', 0.15)) *
            (absorption_strength_normalized).pow(rebound_quality_weights.get('absorption_strength', 0.3)) *
            (offensive_absorption_normalized).pow(rebound_quality_weights.get('offensive_absorption', 0.25)) *
            (mf_buy_execution_alpha_score).pow(rebound_quality_weights.get('mf_buy_execution_alpha', 0.05)) *
            (buy_sweep_intensity_score).pow(rebound_quality_weights.get('buy_sweep_intensity', 0.1))
        ).pow(1/sum(rebound_quality_weights.values())).fillna(0.0)
        # --- 6. 最终合成：三维融合 ---
        wash_out_rebound_score_base = (
            (deception_context_score).pow(fusion_weights.get('deception_context', 0.3)) *
            (panic_depth_score).pow(fusion_weights.get('panic_depth', 0.3)) *
            (rebound_quality_score).pow(fusion_weights.get('rebound_quality', 0.4))
        ).pow(1/(fusion_weights.get('deception_context', 0.3) + fusion_weights.get('panic_depth', 0.3) + fusion_weights.get('rebound_quality', 0.4))).fillna(0.0)
        # 新增代码：情境放大器
        trend_form_norm = trend_form_score.clip(lower=0)
        stability_norm = stability_score
        # 修改代码：新增情境放大器信号
        tension_norm = tension_score.clip(lower=0)
        mtf_cohesion_norm = mtf_cohesion_score.clip(lower=0)
        structural_context_amplifier = (
            (trend_form_norm * context_amplification_weights.get('trend_form', 0.4)) +
            (stability_norm * context_amplification_weights.get('stability', 0.3)) +
            (tension_norm * context_amplification_weights.get('tension', 0.15)) +
            (mtf_cohesion_norm * context_amplification_weights.get('mtf_cohesion', 0.15))
        ).clip(0, 1)
        final_amplifier = 1 + (structural_context_amplifier * max_context_bonus_factor)
        final_wash_out_rebound_score = (wash_out_rebound_score_base * final_amplifier).clip(0, 1)
        return final_wash_out_rebound_score.clip(0, 1).astype(np.float32)






