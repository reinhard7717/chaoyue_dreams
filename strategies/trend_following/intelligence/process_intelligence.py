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
        【V1.2 · 蓝图审查增强版】元分析调度中心
        - 核心升级: 增加对 'PROCESS_META_POWER_TRANSFER' 的特殊处理，豁免其配置信号校验，
                      因为该信号已升级为内置硬编码依赖 L2 底层数据，不再依赖配置文件中的 signal_A/B。
        """
        signal_name = config.get('name', '未知信号')
        # [升级] 豁免 PROCESS_META_POWER_TRANSFER 的配置校验，防止因旧配置导致启动失败
        if signal_name != 'PROCESS_META_POWER_TRANSFER':
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
        """
        【V2.1 · 校验协议对齐版】元关系诊断分发器。
        - 核心修复: 修正了校验拦截逻辑，确保其与 `_run_meta_analysis` 保持一致，
                      对内置硬编码 L2 依赖的 'PROCESS_META_POWER_TRANSFER' 信号予以校验豁免。
        """
        signal_name = config.get('name', '未知信号')
        # 对内置硬编码 L2 数据依赖的信号实施校验豁免，防止 Blueprint Inspector 误报
        if signal_name != 'PROCESS_META_POWER_TRANSFER':
            if not self._extract_and_validate_config_signals(df, config, f"_diagnose_meta_relationship (for {signal_name})"):
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

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.2 · 军械库镜像对冲版】计算"主力-散户"权力交接评分。
        - 核心逻辑: 严格基于军械库 L2 逐单统计数据构建“资金剪刀差”模型。
        - 信号映射:
          1. 主力流: (buy_elg + buy_lg) - (sell_elg + sell_lg) [Source 1 & 3]
          2. 散户流: (buy_md + buy_sm) - (sell_md + sell_sm) [Source 2 & 3]
          3. 结构增强: 结合 chip_concentration_ratio_D [Source 2] 与 turnover_rate_D [Source 3]。
        """
        method_name = "_calculate_power_transfer"
        # 声明军械库 L2 核心依赖信号
        required_signals = [
            'buy_elg_amount_D', 'sell_elg_amount_D', 'buy_lg_amount_D', 'sell_lg_amount_D',
            'buy_md_amount_D', 'sell_md_amount_D', 'buy_sm_amount_D', 'sell_sm_amount_D',
            'amount_D', 'chip_concentration_ratio_D', 'chip_stability_D', 'turnover_rate_D',
            'main_force_activity_index_D', 'downtrend_strength_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 1. 提取 L2 资金流分项 (买入端 Source 1/2, 卖出端 Source 3)
        buy_elg = self._get_safe_series(df, 'buy_elg_amount_D', 0.0, method_name)
        sell_elg = self._get_safe_series(df, 'sell_elg_amount_D', 0.0, method_name)
        buy_lg = self._get_safe_series(df, 'buy_lg_amount_D', 0.0, method_name)
        sell_lg = self._get_safe_series(df, 'sell_lg_amount_D', 0.0, method_name)
        buy_md = self._get_safe_series(df, 'buy_md_amount_D', 0.0, method_name)
        sell_md = self._get_safe_series(df, 'sell_md_amount_D', 0.0, method_name)
        buy_sm = self._get_safe_series(df, 'buy_sm_amount_D', 0.0, method_name)
        sell_sm = self._get_safe_series(df, 'sell_sm_amount_D', 0.0, method_name)
        amount = self._get_safe_series(df, 'amount_D', 1.0, method_name).replace(0, 1.0)
        # 2. 提取结构与行为因子
        chip_conc = self._get_safe_series(df, 'chip_concentration_ratio_D', 0.0, method_name)
        chip_stab = self._get_safe_series(df, 'chip_stability_D', 0.5, method_name)
        turnover = self._get_safe_series(df, 'turnover_rate_D', 0.0, method_name)
        mf_activity = self._get_safe_series(df, 'main_force_activity_index_D', 50.0, method_name)
        downtrend = self._get_safe_series(df, 'downtrend_strength_D', 0.0, method_name)
        # 3. 计算对冲博弈差 (Mirror Hedging Spread)
        # 主力净额 vs 散户净额
        main_force_net = (buy_elg - sell_elg) + (buy_lg - sell_lg)
        retail_net = (buy_md - sell_md) + (buy_sm - sell_sm)
        # 权力转移系数：主力抢筹且散户割肉时最高
        fund_game_spread = (main_force_net - retail_net) / amount
        # 4. 计算筹码穿透增益 (Chip Penetration Gain)
        conc_diff = chip_conc.diff().fillna(0)
        # 换手率越高，集中度变化的信号置信度越高
        turnover_weight = turnover.clip(0, 0.2) * 5.0 
        chip_penetration = conc_diff * chip_stab * (1 + turnover_weight)
        # 5. 综合评分合成
        # 权重配比：资金对冲 50% + 筹码穿透 30% + 活跃度因子 20%
        score_raw = (
            (fund_game_spread * 5.0).clip(-1, 1) * 0.5 +
            (chip_penetration * 20.0).clip(-1, 1) * 0.3 +
            ((mf_activity / 100.0) * 2 - 1).clip(-1, 1) * 0.2
        )
        # 6. 下跌趋势折价过滤
        # 在极强下跌趋势中，由于市场惯性，单纯资金流入的有效性需打折
        trend_discount = pd.Series(1.0, index=df_index)
        trend_discount = trend_discount.mask(downtrend > 0.8, 0.6)
        final_score = (score_raw * trend_discount).clip(-1, 1)
        # 7. 调试状态反馈
        if hasattr(self.strategy, 'atomic_states'):
            self.strategy.atomic_states["_DEBUG_power_transfer_spread"] = fund_game_spread
            self.strategy.atomic_states["_DEBUG_power_transfer_penetration"] = chip_penetration
        return final_score.astype(np.float32)

    def _diagnose_meta_relationship_internal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        signal_name = config.get('name')
        df_index = df.index
        meta_score = pd.Series(dtype=np.float32)
        if signal_name == 'PROCESS_META_COST_ADVANTAGE_TREND':
            meta_score = self.calculate_cost_advantage_trend_relationship_processor.calculate(df, config)
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
        elif signal_name == 'PROCESS_STRATEGY_FF_VS_STRUCTURE_LEAD':
            relationship_score = self._calculate_ff_vs_structure_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_CONTROL':
            meta_score = self.calculate_main_force_control_processor.calculate(df, config)
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
        elif signal_name == 'PROCESS_META_POWER_TRANSFER':
            # [新增] 显式调用新的权力交接计算逻辑
            meta_score = self._calculate_power_transfer(df, config)
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

    def _calculate_price_vs_capitulation_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 军械库直连版】计算“价格与散户投降”的专属瞬时关系分。
        - 信号来源:
            1. 基础背离: `pressure_trapped_D` [Source 3] (由Config驱动)
            2. 放大器: `INTRADAY_SUPPORT_INTENT_D` [Source 1] (日内承接意图)
        """
        method_name = "_calculate_price_vs_capitulation_relationship"
        # 辅助信号: 日内承接意图 [Source 1]
        support_signal = 'INTRADAY_SUPPORT_INTENT_D'
        if support_signal not in df.columns:
             # 如果缺承接信号，仅返回基础分
             return self._calculate_instantaneous_relationship(df, config).clip(-1, 1)
        df_index = df.index
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        # 引入主动承接作为真实性放大器
        active_buying_support = self._get_safe_series(df, support_signal, 0.0, method_name=method_name)
        active_buying_norm = self._normalize_series(active_buying_support, df_index, bipolar=False)
        authenticity_amplifier = 1 + active_buying_norm
        final_score = (base_divergence_score * authenticity_amplifier).clip(-1, 1)
        return final_score

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 军械库直连版】计算“价格效率”的专属瞬时关系分。
        - 信号来源:
            1. 效率: `VPA_EFFICIENCY_D` [Source 1] (Config驱动)
            2. 信念(品质): `net_mf_amount_D` [Source 3] (主力净额)
            3. 噪音(惩罚): `shakeout_score_D` [Source 3] (洗盘/震仓评分)
        """
        method_name = "_calculate_price_efficiency_relationship"
        # 军械库信号映射
        conviction_signal = 'net_mf_amount_D'
        wash_signal = 'shakeout_score_D'
        required_signals = [conviction_signal, wash_signal]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        df_index = df.index
        # 计算基础共识分数
        base_consensus_score = self._calculate_instantaneous_relationship(df, config)
        # 引入品质因子
        main_force_conviction = self._get_safe_series(df, conviction_signal, 0.0, method_name=method_name)
        wash_trade_intensity = self._get_safe_series(df, wash_signal, 0.0, method_name=method_name)
        # 归一化：主力净额为双极
        conviction_norm = self._normalize_series(main_force_conviction, df_index, bipolar=True)
        # 洗盘评分为单极
        wash_trade_norm = self._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        # 资金越正向，洗盘越少，效率越“纯”
        quality_factor = (conviction_norm.clip(lower=0) * (1 - wash_trade_norm)).clip(0, 1)
        final_score = (base_consensus_score * quality_factor).clip(-1, 1)
        return final_score

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 军械库直连版】计算“博弈背离”的专属瞬时关系分。
        - 信号来源:
            1. 背离: `game_intensity_D` [Source 2] (Config驱动)
            2. 战场重心: `weight_avg_cost_D` [Source 4] (加权平均成本)
        """
        method_name = "_calculate_pd_divergence_relationship"
        cost_signal = 'weight_avg_cost_D' 
        if cost_signal not in df.columns:
            return self._calculate_instantaneous_relationship(df, config).clip(-1, 1)
            
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        # 引入战场纵深因子
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name)
        mf_cost = self._get_safe_series(df, cost_signal, 0.0, method_name=method_name)
        mf_cost_safe = mf_cost.replace(0, np.nan) 
        # 价格 > 平均成本，多头占优，信号放大
        battlefield_context_factor = (1 + (close_price - mf_cost_safe) / mf_cost_safe).fillna(1).clip(0, 2)
        final_score = (base_divergence_score * battlefield_context_factor).clip(-1, 1)
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
        【V5.1 · 军械库全适配重构版】计算“恐慌洗盘吸筹”的专属信号。
        - 信号替换:
          1. 恐慌/痛苦: `pressure_trapped_D` [Source 3]
          2. 吸收: `absorption_energy_D` [Source 1]
          3. 结构支撑: `uptrend_strength_D` [Source 3] (代理杠杆)
          4. 放量: `tick_abnormal_volume_ratio_D` [Source 3] (代理爆发现数)
          5. 势能: `chip_stability_D` [Source 2] (代理历史势能)
        """
        method_name = "_calculate_panic_washout_accumulation"
        df_index = df.index
        potential_sig = 'chip_stability_D'
        historical_potential = self._get_safe_series(df, potential_sig, 0.5, method_name)
        potential_gate = config.get('historical_potential_gate', 0.0)
        pain_sig = 'pressure_trapped_D'
        buy_support_sig = 'INTRADAY_SUPPORT_INTENT_D'
        absorption_sig = 'absorption_energy_D'
        conc_sig = 'chip_concentration_ratio_D'
        cost_adv_sig = 'profit_ratio_D'
        mf_flow_sig = 'net_mf_amount_D'
        flow_cred_sig = 'flow_consistency_D'
        struct_sig = 'uptrend_strength_D'
        burst_sig = 'tick_abnormal_volume_ratio_D'
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = ['close_D', pain_sig, mf_flow_sig, buy_support_sig, conc_sig, cost_adv_sig, struct_sig, flow_cred_sig, burst_sig, absorption_sig]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        mtf_retail_panic = self._get_mtf_slope_accel_score(df, pain_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        panic_score = mtf_retail_panic.rolling(3, min_periods=1).mean()
        power_transfer_score = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        mtf_active_buying_support = self._get_mtf_slope_accel_score(df, buy_support_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_absorption = self._get_mtf_slope_accel_score(df, absorption_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        absorption_score = (power_transfer_score.clip(lower=0) * 0.5 + mtf_absorption * 0.25 + mtf_active_buying_support * 0.25).clip(0, 1).rolling(3, min_periods=1).mean()
        mtf_concentration_slope = self._get_mtf_slope_accel_score(df, conc_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_cost_adv_slope = self._get_mtf_slope_accel_score(df, cost_adv_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mf_net_flow_slope = self._get_mtf_slope_accel_score(df, mf_flow_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        struct_norm = self._normalize_series(self._get_safe_series(df, struct_sig, 0.0, method_name), df_index, ascending=True)
        repair_score = ((mtf_concentration_slope.clip(lower=0) * 0.4 + mtf_cost_adv_slope.clip(lower=0) * 0.3 + mtf_mf_net_flow_slope.clip(lower=0) * 0.3).clip(0, 1) * 0.5 + struct_norm * 0.5).clip(0, 1)
        mtf_price_trend = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_volume_burst = self._get_mtf_slope_accel_score(df, burst_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        is_blitz_washout = (mtf_price_trend < -0.3) & (mtf_volume_burst > 0.5)
        washout_candidate_mask = is_blitz_washout | (panic_score > 0.4 & absorption_score > 0.15)
        base_score = (panic_score * absorption_score * repair_score).pow(1/3)
        mf_flow_norm = self._normalize_series(self._get_safe_series(df, mf_flow_sig, 0.0, method_name), df_index, bipolar=True)
        flow_cred_norm = self._normalize_series(self._get_safe_series(df, flow_cred_sig, 0.0, method_name), df_index, bipolar=False)
        judged_base_score = (base_score * (1 + repair_score) * (mf_flow_norm.clip(lower=0) * flow_cred_norm)).clip(0, 1)
        final_score = judged_base_score * (1 - mf_flow_norm.clip(upper=0).abs())
        potential_gate_mask = historical_potential > potential_gate
        final_score = final_score.where(washout_candidate_mask & potential_gate_mask, 0.0).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.12 · 军械库全适配重构版】计算“诡道吸筹”信号。
        - 信号替换:
          1. 核心动作(拆单): `stealth_flow_ratio_D` [Source 3] (作为隐蔽强度的物理代理)
          2. 欺诈指标: `stealth_flow_ratio_D` (复用其多维含义)
          3. 协同驱动: `chip_flow_intensity_D` [Source 2] (替代原子公理)
          4. 集中度: `chip_concentration_ratio_D` [Source 2]
        """
        method_name = "_calculate_deceptive_accumulation"
        deception_sig = 'stealth_flow_ratio_D'
        conc_sig = 'chip_concentration_ratio_D'
        mf_flow_sig = 'net_mf_amount_D'
        coherent_sig = 'chip_flow_intensity_D'
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [deception_sig, conc_sig, mf_flow_sig, coherent_sig, 'close_D', 'PROCESS_META_POWER_TRANSFER']
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        core_action_raw = self._get_safe_series(df, deception_sig, 0.0, method_name)
        core_action_score = self._normalize_series(core_action_raw, df_index, bipolar=False)
        mtf_deception_evidence = self._get_mtf_slope_accel_score(df, deception_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_trend = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_conc_slope = self._get_mtf_slope_accel_score(df, conc_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mf_flow = self._get_mtf_slope_accel_score(df, mf_flow_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        power_transfer = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        price_down_strength = mtf_price_trend.clip(upper=0).abs()
        disguise_score = ((price_down_strength * mtf_mf_flow.clip(lower=0)).pow(0.5) * 0.5 + (price_down_strength * power_transfer.clip(lower=0)).pow(0.5) * 0.5).clip(0, 1)
        divergence_resonance = (price_down_strength * mtf_conc_slope.clip(lower=0)).pow(0.5)
        deceptive_context = (divergence_resonance * 0.4 + mtf_deception_evidence * 0.3 + disguise_score * 0.3).clip(0, 1)
        price_adj = pd.Series(1.0, index=df_index).mask(mtf_price_trend > 0, (1 - mtf_price_trend).clip(0, 1))
        coherent_drive = self._get_safe_series(df, coherent_sig, 0.0, method_name)
        coherence_penalty = (1 - self._normalize_series(coherent_drive, df_index, bipolar=True).clip(upper=0).abs()).clip(0, 1)
        final_score = (core_action_score * deceptive_context * coherence_penalty * price_adj).fillna(0.0)
        return final_score.clip(0, 1).astype(np.float32)

    def _calculate_upthrust_washout(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4 · 军械库直连版】识别主力利用“上冲回落”阴线进行的洗盘行为。
        - 信号映射:
          1. 纯度: `pushing_score_D` [Source 3] (推升评分，替代纯度)
          2. 卖压: `pressure_release_index_D` [Source 3] (抛压释放)
          3. 承接: `INTRADAY_SUPPORT_INTENT_D` [Source 1]
          4. 资金: `net_mf_amount_D` [Source 3]
        """
        method_name = "_calculate_upthrust_washout"
        purity_sig = 'pushing_score_D'
        upper_pressure_sig = 'pressure_release_index_D'
        buy_support_sig = 'INTRADAY_SUPPORT_INTENT_D'
        mf_flow_sig = 'net_mf_amount_D'
        required_signals = [
            'SCORE_STRUCT_AXIOM_TREND_FORM', 'BIAS_21_D', 'pct_change_D',
            purity_sig, upper_pressure_sig,
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', buy_support_sig,
            'PROCESS_META_POWER_TRANSFER', 'open_D', 'high_D', 'close_D', 'low_D',
            mf_flow_sig
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        df_index = df.index
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        bias_21 = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name)
        upward_purity_raw = self._get_safe_series(df, purity_sig, 0.0, method_name)
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name)
        upper_shadow_pressure_raw = self._get_safe_series(df, upper_pressure_sig, 0.0, method_name)
        lower_shadow_strength = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        active_buying_raw = self._get_safe_series(df, buy_support_sig, 0.0, method_name)
        power_transfer = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        open_price = self._get_safe_series(df, 'open_D', 0.0, method_name)
        high_price = self._get_safe_series(df, 'high_D', 0.0, method_name)
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name)
        low_price = self._get_safe_series(df, 'low_D', 0.0, method_name)
        main_force_net_flow = self._get_safe_series(df, mf_flow_sig, 0.0, method_name)
        upward_purity_norm = self._normalize_series(upward_purity_raw, df_index, bipolar=False)
        upper_shadow_pressure_norm = self._normalize_series(upper_shadow_pressure_raw, df_index, bipolar=False)
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        power_transfer_norm = power_transfer.clip(lower=0)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        context_mask = (trend_form_score > 0.2) & (bias_21 < 0.2) & (upward_purity_norm.rolling(3).mean() > 0.3)
        total_range = high_price - low_price
        total_range_safe = total_range.replace(0, 1e-9)
        upper_shadow = high_price - np.maximum(open_price, close_price)
        upper_shadow_ratio = (upper_shadow / total_range_safe).fillna(0)
        is_high_open_low_close_yin = (open_price > close_price) & (pct_change < 0)
        is_long_upper_shadow_yin = (upper_shadow_ratio > 0.4) & (close_price < open_price)
        is_upthrust_kline = is_high_open_low_close_yin | is_long_upper_shadow_yin
        is_down_day = (pct_change < 0).astype(float)
        selling_pressure_score = (upper_shadow_pressure_norm * 0.7 + is_down_day * 0.3).clip(0, 1)
        absorption_rebuttal_score = pd.concat([
            active_buying_norm,
            lower_shadow_strength,
            power_transfer_norm
        ], axis=1).max(axis=1)
        net_washout_intent = (absorption_rebuttal_score - selling_pressure_score).clip(0, 1)
        mf_inflow_gate = main_force_net_flow_norm.clip(lower=0)
        final_score = net_washout_intent.where(context_mask & is_upthrust_kline & (mf_inflow_gate > 0.1), 0.0).fillna(0.0)
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
        【V3.4 · 军械库直连版】计算“套牢盘投降”信号。
        - 信号映射:
          1. 投降(恐慌): `pressure_trapped_D` [Source 3] (套牢压力)
          2. 承接: `INTRADAY_SUPPORT_INTENT_D` [Source 1]
          3. 备用吸收: `absorption_energy_D` [Source 1]
        """
        method_name = "_calculate_loser_capitulation"
        capitulation_sig = 'pressure_trapped_D'
        active_buy_sig = 'INTRADAY_SUPPORT_INTENT_D'
        lower_shadow_sig = 'absorption_energy_D' 
        required_signals = [
            'pct_change_D', capitulation_sig, active_buy_sig
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        df_index = df.index
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name)
        capitulation_flow_raw = self._get_safe_series(df, capitulation_sig, 0.0, method_name)
        active_buying_raw = self._get_safe_series(df, active_buy_sig, 0.0, method_name)
        # 优先使用原子信号，若无则使用原始信号替代
        lower_shadow_absorption = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        if lower_shadow_absorption.sum() == 0: 
             lower_shadow_raw = self._get_safe_series(df, lower_shadow_sig, 0.0, method_name)
             lower_shadow_absorption = self._normalize_series(lower_shadow_raw, df_index, bipolar=False)
        context_mask = (pct_change < 0) | (lower_shadow_absorption > 0.5)
        panic_score = self._normalize_series(capitulation_flow_raw, df_index, bipolar=False)
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        absorption_score = pd.concat([active_buying_norm, lower_shadow_absorption], axis=1).max(axis=1)
        final_score = (panic_score * absorption_score).where(context_mask, 0.0).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_cost_advantage_trend_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.7 · 军械库直连版】计算成本优势趋势。
        - 信号映射:
          1. 成本优势: `profit_ratio_D` [Source 3] (获利比例)
          2. 信念: `net_mf_amount_D` [Source 3] (作为信念代理)
          3. 纯度: `pushing_score_D` [Source 3]
          4. 隐蔽: `consolidation_accumulation_score_D` [Source 2]
          5. 派发: `distribution_score_D` [Source 2]
          6. 卖压: `intraday_distribution_confidence_D` [Source 2]
          7. 兑现: `profit_pressure_D` [Source 3]
          8. 承接: `INTRADAY_SUPPORT_INTENT_D` [Source 1]
          9. 净流: `net_mf_amount_D` [Source 3]
          10. 信用: `flow_consistency_D` [Source 2]
        """
        method_name = "_calculate_cost_advantage_trend_relationship"
        ca_sig = 'profit_ratio_D' 
        conviction_sig = 'net_mf_amount_D' # 使用净额作为信念代理
        purity_sig = 'pushing_score_D'
        suppress_sig = 'consolidation_accumulation_score_D'
        dist_sig = 'distribution_score_D'
        active_sell_sig = 'intraday_distribution_confidence_D'
        profit_sig = 'profit_pressure_D'
        buy_support_sig = 'INTRADAY_SUPPORT_INTENT_D'
        mf_flow_sig = 'net_mf_amount_D'
        cred_sig = 'flow_consistency_D'
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'pct_change_D', ca_sig, conviction_sig,
            purity_sig, suppress_sig,
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', dist_sig,
            active_sell_sig, profit_sig, buy_support_sig,
            mf_flow_sig, cred_sig
        ]
        for base_sig in ['close_D', ca_sig, conviction_sig, purity_sig, dist_sig, active_sell_sig]:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
                
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        df_index = df.index
        mtf_price_change = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_ca_change = self._get_mtf_slope_accel_score(df, ca_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_conviction = self._get_mtf_slope_accel_score(df, conviction_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_upward_purity = self._get_mtf_slope_accel_score(df, purity_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_distribution_intensity = self._get_mtf_slope_accel_score(df, dist_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_active_selling = self._get_mtf_slope_accel_score(df, active_sell_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        suppressive_accum_norm = self._normalize_series(self._get_safe_series(df, suppress_sig, 0.0, method_name), df_index, bipolar=False)
        lower_shadow_absorb = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        profit_taking_flow_norm = self._normalize_series(self._get_safe_series(df, profit_sig, 0.0, method_name), df_index, bipolar=False)
        active_buying_support_norm = self._normalize_series(self._get_safe_series(df, buy_support_sig, 0.0, method_name), df_index, bipolar=False)
        main_force_net_flow_norm = self._normalize_series(self._get_safe_series(df, mf_flow_sig, 0.0, method_name), df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(self._get_safe_series(df, cred_sig, 0.0, method_name), df_index, bipolar=False)
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name)
        # Q1: 价涨 & 优扩
        Q1_base = (mtf_price_change.clip(lower=0) * mtf_ca_change.clip(lower=0)).pow(0.5)
        Q1_confirm = (mtf_main_force_conviction.clip(lower=0) * mtf_upward_purity * flow_credibility_norm).pow(1/3)
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        # Q2: 价跌 & 优缩
        Q2_base = (mtf_price_change.clip(upper=0).abs() * mtf_ca_change.clip(upper=0).abs()).pow(0.5)
        Q2_distribution_evidence = (profit_taking_flow_norm * 0.4 + mtf_active_selling * 0.3 + mtf_distribution_intensity * 0.3).clip(0, 1)
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        # Q3: 价跌 & 优扩
        Q3_base = (mtf_price_change.clip(upper=0).abs() * mtf_ca_change.clip(lower=0)).pow(0.5)
        Q3_confirm = (suppressive_accum_norm * lower_shadow_absorb * flow_credibility_norm).pow(1/3)
        pre_5day_pct_change = close_price.pct_change(periods=5).shift(1).fillna(0)
        norm_pre_drop_5d = self._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        pre_drop_context_bonus = norm_pre_drop_5d * 0.5
        Q3_final = (Q3_base * Q3_confirm * (1 + pre_drop_context_bonus)).clip(0, 1)
        # Q4: 价涨 & 优缩
        Q4_base = (mtf_price_change.clip(lower=0) * mtf_ca_change.clip(upper=0).abs()).pow(0.5)
        mf_outflow_risk = main_force_net_flow_norm.clip(upper=0).abs()
        Q4_trap_evidence = (mtf_distribution_intensity * 0.4 + (1 - active_buying_support_norm) * 0.3 + mf_outflow_risk * 0.3).clip(0, 1)
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        final_score = (Q1_final + Q2_final + Q3_final + Q4_final).clip(-1, 1)
        return final_score.astype(np.float32)

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.7 · 军械库全适配版】诊断“突破加速抢筹”战术。
        - 核心修复: 
            1. 突破证据: `breakout_confidence_D` [Source 1] 替代旧公理。
            2. 结构证据: `uptrend_strength_D` [Source 3] 替代趋势形态公理。
            3. 相对强度: `industry_strength_rank_D` [Source 2] 替代相对强度公理。
        """
        method_name = "_calculate_breakout_acceleration"
        mf_flow_sig = 'net_mf_amount_D'
        cred_sig = 'flow_consistency_D'
        breakout_sig = 'breakout_confidence_D'
        trend_sig = 'uptrend_strength_D'
        rs_sig = 'industry_strength_rank_D'
        required_signals = [
            breakout_sig, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_POWER_TRANSFER', trend_sig, rs_sig,
            mf_flow_sig, cred_sig
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        relative_strength_raw = self._get_safe_series(df, rs_sig, 0.0, method_name)
        relative_strength = self._normalize_series(relative_strength_raw, df_index, bipolar=False)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        main_force_net_flow = self._get_safe_series(df, mf_flow_sig, 0.0, method_name)
        flow_credibility = self._get_safe_series(df, cred_sig, 0.0, method_name)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        breakout_evidence = self._normalize_series(self._get_safe_series(df, breakout_sig, 0.0, method_name), df_index, bipolar=False)
        intent_evidence = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0)
        flow_evidence = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0)
        structure_evidence = self._normalize_series(self._get_safe_series(df, trend_sig, 0.0, method_name), df_index, bipolar=False)
        mf_flow_validation = main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm
        weights = {'breakout': 0.3, 'intent': 0.25, 'structure': 0.2, 'flow': 0.15, 'mf_flow_validation': 0.1}
        resonance_score = (breakout_evidence * weights['breakout'] + intent_evidence * weights['intent'] + structure_evidence * weights['structure'] + flow_evidence * weights['flow'] + mf_flow_validation * weights['mf_flow_validation']).clip(0, 1)
        breakout_gate_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        breakout_gate_factor = breakout_gate_factor.mask(breakout_evidence < 0.2, breakout_evidence * 0.5)
        breakout_gate_factor = breakout_gate_factor.mask(breakout_evidence < 0.05, 0.0)
        resonance_score_gated = resonance_score * breakout_gate_factor
        rs_modulator_base = (1 + relative_strength * rs_amplifier)
        mf_flow_modulator = (1 + mf_flow_validation * 0.5)
        rs_modulator = rs_modulator_base * mf_flow_modulator
        final_score = (resonance_score_gated * rs_modulator).clip(0, 1).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 军械库全适配版】识别主力从吸筹转向公开强攻的转折信号。
        - 核心修复: 
            1. 基石指标: `accumulation_signal_score_D` [Source 1] 替代旧隐蔽吸筹强度。
            2. 其他信号保持军械库映射: `net_mf_amount_D` [Source 3], `flow_efficiency_D` [Source 2], `tick_large_order_net_D` [Source 3]。
        """
        method_name = "_calculate_fund_flow_accumulation_inflection"
        acc_sig = 'accumulation_signal_score_D'
        mf_flow_sig = 'net_mf_amount_D'
        buy_exhaust_sig = 'flow_efficiency_D'
        large_pressure_sig = 'tick_large_order_net_D'
        cred_sig = 'flow_consistency_D'
        support_sig = 'INTRADAY_SUPPORT_INTENT_D'
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [acc_sig, mf_flow_sig, buy_exhaust_sig, large_pressure_sig, cred_sig, support_sig]
        for base_sig in [acc_sig, buy_exhaust_sig, large_pressure_sig, mf_flow_sig, support_sig]:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        flow_credibility_raw = self._get_safe_series(df, cred_sig, 0.0, method_name)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        mtf_hidden_accumulation = self._get_mtf_slope_accel_score(df, acc_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_net_flow_slope = self._get_mtf_slope_accel_score(df, mf_flow_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        prelude_score_base = self._normalize_series(mtf_hidden_accumulation.rolling(5).mean(), df_index, bipolar=False)
        prelude_score = (prelude_score_base * mtf_mf_net_flow_slope.clip(lower=0)).pow(0.5)
        mtf_buy_efficiency = self._get_mtf_slope_accel_score(df, buy_exhaust_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        buy_resilience = mtf_buy_efficiency 
        large_order_net = self._get_safe_series(df, large_pressure_sig, 0.0, method_name)
        large_pressure_raw = -large_order_net 
        mtf_large_pressure = self._normalize_series(large_pressure_raw, df_index, bipolar=False)
        mtf_active_buying_support = self._get_mtf_slope_accel_score(df, support_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        pressure_exhaustion_synergy = (mtf_large_pressure * (1 - buy_resilience)).pow(0.5)
        mtf_main_force_flow_momentum = self._get_mtf_slope_accel_score(df, mf_flow_sig, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        attack_score = (buy_resilience * 0.2 + mtf_main_force_flow_momentum.clip(lower=0) * 0.25 + (1 - mtf_large_pressure) * 0.2 + flow_credibility_norm * 0.15 + mtf_active_buying_support * 0.2 - pressure_exhaustion_synergy * 0.2).clip(0, 1)
        final_score = (prelude_score * attack_score).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.7 · 军械库全适配版】“利润与流向”专属关系计算引擎。
        - 核心重构: 废除所有旧 SCORE_* 依赖，直连军械库底层信号。
        - 信号映射:
          1. 派发压力: `profit_pressure_D` [Source 3]
          2. 净流动力: `net_mf_amount_D` [Source 3]
          3. 获利广度: `profit_ratio_D` [Source 3]
          4. 派发信心: `intraday_distribution_confidence_D` [Source 2] (替代缺失的原子意图信号)
          5. 资金一致: `flow_consistency_D` [Source 2]
        """
        method_name = "_calculate_profit_vs_flow_relationship"
        # 定义军械库信号名称
        pressure_sig = 'profit_pressure_D'
        drive_sig = 'net_mf_amount_D'
        profit_ratio_sig = 'profit_ratio_D'
        dist_confidence_sig = 'intraday_distribution_confidence_D'
        consistency_sig = 'flow_consistency_D'
        # 执行严格信号校验
        required_signals = [pressure_sig, drive_sig, profit_ratio_sig, dist_confidence_sig, consistency_sig]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 1. 原始数据安全获取与预处理
        profit_pressure_raw = self._get_safe_series(df, pressure_sig, 0.0, method_name)
        net_mf_amount_raw = self._get_safe_series(df, drive_sig, 0.0, method_name)
        profit_ratio_raw = self._get_safe_series(df, profit_ratio_sig, 0.0, method_name)
        dist_confidence_raw = self._get_safe_series(df, dist_confidence_sig, 0.0, method_name)
        consistency_raw = self._get_safe_series(df, consistency_sig, 0.0, method_name)
        # 2. 维度归一化处理
        # 派发压力、获利比例及派发信心均为单极信号
        profit_pressure_norm = self._normalize_series(profit_pressure_raw, df_index, bipolar=False)
        profit_ratio_norm = self._normalize_series(profit_ratio_raw, df_index, bipolar=False)
        dist_conf_norm = self._normalize_series(dist_confidence_raw, df_index, bipolar=False)
        # 主力净额为双极信号，资金一致性为单极信号
        net_mf_norm = self._normalize_series(net_mf_amount_raw, df_index, bipolar=True)
        consistency_norm = self._normalize_series(consistency_raw, df_index, bipolar=False)
        # 3. 计算派发压力综合分 (Pressure Score)
        # 权重分配：获利压力(40%) + 获利广度(30%) + 派发意图/信心(30%)
        pressure_score = (
            profit_pressure_norm * 0.4 +
            profit_ratio_norm * 0.3 +
            dist_conf_norm * 0.3
        ).clip(0, 1)
        # 4. 计算建仓动力综合分 (Drive Score)
        # 只有当净流为正时，一致性才作为增强因子；若为负，则视为离场驱动
        drive_score = (net_mf_norm.clip(lower=0) * consistency_norm).clip(0, 1)
        # 5. 态势对冲计算
        # 最终分数 = 进场动力 - 离场压力；正值代表建仓占优，负值代表派发占优
        relationship_score = drive_score - pressure_score
        final_score = relationship_score.clip(-1, 1)
        # 调试信息记录
        if hasattr(self.strategy, 'atomic_states'):
            self.strategy.atomic_states["_DEBUG_profit_vs_flow_drive"] = drive_score
            self.strategy.atomic_states["_DEBUG_profit_vs_flow_pressure"] = pressure_score
        return final_score.astype(np.float32)

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.6 · 军械库直连版】“个股板块协同共振”专属关系计算引擎
        - 信号映射:
          1. 资金: `net_mf_amount_D` [Source 3]
          2. 信用: `flow_consistency_D` [Source 2]
        """
        method_name = "_calculate_stock_sector_sync"
        stock_signal_name = 'pct_change_D'
        sector_rank_name = 'industry_strength_rank_D'
        sector_momentum_name = 'SLOPE_5_industry_strength_rank_D'
        main_force_net_flow_name = 'net_mf_amount_D'
        flow_credibility_name = 'flow_consistency_D'
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [stock_signal_name, sector_rank_name, sector_momentum_name, main_force_net_flow_name, flow_credibility_name]
        for base_sig in ['industry_strength_rank_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
                
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        df_index = df.index
        stock_signal_raw = self._get_safe_series(df, stock_signal_name, 0.0, method_name)
        main_force_net_flow_raw = self._get_safe_series(df, main_force_net_flow_name, 0.0, method_name)
        flow_credibility_raw = self._get_safe_series(df, flow_credibility_name, 0.0, method_name)
        stock_strength_score = self._normalize_series(stock_signal_raw, target_index=df_index, bipolar=True)
        mtf_sector_rank_score = self._get_mtf_slope_accel_score(df, 'industry_strength_rank_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_sector_momentum_score = self._get_mtf_slope_accel_score(df, 'industry_strength_rank_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow_raw, target_index=df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, target_index=df_index, bipolar=False)
        # 看涨协同
        bullish_stock_movement = stock_strength_score.clip(lower=0)
        sector_strength_context_bullish = mtf_sector_rank_score
        sector_momentum_context_bullish = mtf_sector_momentum_score.clip(lower=0)
        bullish_sector_resonance = (sector_strength_context_bullish * sector_momentum_context_bullish).pow(0.5).fillna(0.0)
        mf_inflow_factor = main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm
        bullish_amplification_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        bullish_amplification_factor = bullish_amplification_factor.mask(sector_momentum_context_bullish > 0.1, 1 + bullish_sector_resonance * mf_inflow_factor)
        bullish_amplification_factor = bullish_amplification_factor.mask(sector_momentum_context_bullish <= 0.1, 0.0) 
        final_bullish_score = bullish_stock_movement * bullish_amplification_factor
        # 看跌协同
        bearish_stock_movement = stock_strength_score.clip(upper=0).abs()
        sector_weakness_context_bearish = (1 - mtf_sector_rank_score)
        sector_negative_momentum_context_bearish = mtf_sector_momentum_score.clip(upper=0).abs()
        bearish_sector_resonance = (sector_weakness_context_bearish * sector_negative_momentum_context_bearish).pow(0.5).fillna(0.0)
        bearish_amplification_factor = 1 + bearish_sector_resonance
        final_bearish_score = bearish_stock_movement * bearish_amplification_factor * -1
        relationship_score = final_bullish_score + final_bearish_score
        final_score = relationship_score.clip(-1, 1)
        return final_score.astype(np.float32)

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.4 · 军械库直连版】“热门板块冷却”专属关系计算引擎
        - 信号映射:
          1. 资金: `net_mf_amount_D` [Source 3]
        """
        method_name = "_calculate_hot_sector_cooling"
        hotness_signal_name = 'THEME_HOTNESS_SCORE_D'
        flow_signal_name = 'net_mf_amount_D'
        required_signals = [hotness_signal_name, flow_signal_name]
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(dtype=np.float32)
            
        df_index = df.index
        hotness_signal_raw = self._get_safe_series(df, hotness_signal_name, 0.0, method_name)
        flow_signal_raw = self._get_safe_series(df, flow_signal_name, 0.0, method_name)
        hotness_state_score = self._normalize_series(hotness_signal_raw, df_index, bipolar=False)
        flow_direction_score = self._normalize_series(flow_signal_raw, df_index, bipolar=True)
        outflow_score = flow_direction_score.clip(upper=0).abs()
        relationship_score = hotness_state_score * outflow_score
        final_score = relationship_score.clip(0, 1)
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.4 · 军械库直连版】计算“价资关系”的专属方法。
        - 核心逻辑: 依赖 process.json 配置的 `net_mf_amount_D` [Source 3]。
        """
        method_name = "_calculate_pf_relationship"
        # 直接调用通用方法，校验逻辑下沉到 _validate_required_signals
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df.index)
        return meta_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.4 · 军械库直连版】计算“价筹关系”的专属方法。
        - 核心逻辑: 依赖 process.json 配置的 `peak_concentration_D` [Source 3]。
        """
        method_name = "_calculate_pc_relationship"
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df.index)
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
        【V3.1 · 军械库适配版】计算“资金与结构”的专属瞬时关系分。
        - 核心修复:
            1. 结构: `uptrend_strength_D` [Source 3] (替代趋势形态评分)
            2. 资金: `flow_consistency_D` [Source 2] (替代资金共识评分)
        """
        # 军械库信号
        structure_signal = 'uptrend_strength_D'
        flow_signal = 'flow_consistency_D'
        
        required_signals = [structure_signal, flow_signal]
        # 使用通用校验
        if not self._validate_required_signals(df, required_signals, "_calculate_ff_vs_structure_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        # 计算基础背离分数 (依赖 Config 中的 signal_A/B 配置)
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        
        # 引入战略态势放大器
        # 使用 trend strength 的绝对值作为放大器，趋势越强，背离越重要
        trend_strength = self._get_safe_series(df, structure_signal, 0.0, method_name="_calculate_ff_vs_structure_relationship")
        
        # uptrend_strength_D 通常是 0-100 或 0-1，这里假设已归一化或直接使用
        # 为了安全，先归一化
        trend_strength_norm = self._normalize_series(trend_strength, df.index, bipolar=False)
        
        strategic_context_amplifier = 1 + trend_strength_norm
        final_score = (base_divergence_score * strategic_context_amplifier).clip(-1, 1)
        
        return final_score

    def _calculate_dyn_vs_chip_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 军械库适配版】计算“动能与筹码”的专属瞬时关系分。
        - 核心修复:
            1. 动能: `ROC_13_D` [Source 1] (替代动量公理)
            2. 心态: `winner_rate_D` [Source 4] (替代持仓心态公理)
            3. 压力: `profit_ratio_D` [Source 3] (替代获利幅度)
        """
        # 军械库信号
        momentum_signal = 'ROC_13_D'
        sentiment_signal = 'winner_rate_D'
        profit_signal = 'profit_ratio_D'
        
        required_signals = [momentum_signal, sentiment_signal, profit_signal]
        if not self._validate_required_signals(df, required_signals, "_calculate_dyn_vs_chip_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        # 计算基础共识分数 (依赖 Config)
        base_consensus_score = self._calculate_instantaneous_relationship(df, config)
        
        # 引入派发压力因子进行动机审判
        profit_ratio = self._get_safe_series(df, profit_signal, 0.0, method_name="_calculate_dyn_vs_chip_relationship")
        profit_ratio_norm = self._normalize_series(profit_ratio, df.index, bipolar=False)
        
        distribution_pressure_factor = 1 + profit_ratio_norm
        
        # 仅当基础分为负（内部分裂/同步下跌）时，才进行动机审判
        # 如果动能和心态都在变差（负分），且获利比例很高，说明是派发导致的下跌，信号加重
        final_score = base_consensus_score.where(
            base_consensus_score >= 0,
            base_consensus_score * distribution_pressure_factor
        ).clip(-1, 1)
        
        return final_score

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series, config: Dict) -> pd.Series:
        """
        【V2.8 · 军械库直连版】识别主力利用洗盘后进行反弹的信号。
        - 信号映射:
          1. 洗盘: `shakeout_score_D` [Source 3]
          2. 欺诈: `stealth_flow_ratio_D` [Source 3]
          3. 卖压: `intraday_distribution_confidence_D` [Source 2]
          4. 恐慌: `pressure_trapped_D` [Source 3]
          5. 纯度: `pushing_score_D` [Source 3]
        """
        method_name = "_calculate_process_wash_out_rebound"
        df_index = df.index
        p_conf = self.params
        params = get_param_value(p_conf.get('wash_out_rebound_params'), {})
        wash_sig = 'shakeout_score_D'
        active_sell_sig = 'intraday_distribution_confidence_D'
        panic_sig = 'pressure_trapped_D' 
        purity_sig = 'pushing_score_D'
        deception_sig = 'stealth_flow_ratio_D'
        required_signals = [
            wash_sig, deception_sig, active_sell_sig,
            panic_sig, 
            'closing_strength_index_D', purity_sig, # closing_strength_index_D 假设存在或使用默认
            'SCORE_STRUCT_AXIOM_TREND_FORM', 'SCORE_STRUCT_AXIOM_STABILITY',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'SCORE_BEHAVIOR_DECEPTION_INDEX', 'SCORE_MICRO_STRATEGY_STEALTH_OPS',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT',
            'main_force_buy_execution_alpha_D', 'buy_sweep_intensity_D',
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_STRUCT_AXIOM_MTF_COHESION'
        ]
        fusion_weights = get_param_value(params.get('fusion_weights'), {"deception_context": 0.3, "panic_depth": 0.3, "rebound_quality": 0.4})
        deception_context_weights = get_param_value(params.get('deception_context_weights'), {})
        panic_depth_weights = get_param_value(params.get('panic_depth_weights'), {})
        rebound_quality_weights = get_param_value(params.get('rebound_quality_weights'), {})
        context_amplification_weights = get_param_value(params.get('context_amplification_weights'), {})
        max_context_bonus_factor = get_param_value(params.get('max_context_bonus_factor'), 0.5)
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        for base_sig in [wash_sig, active_sell_sig, panic_sig, 'deception_lure_short_intensity_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df.index)
        wash_trade_raw = self._get_safe_series(df, wash_sig, 0.0, method_name)
        active_selling_raw = self._get_safe_series(df, active_sell_sig, 0.0, method_name)
        panic_cascade_raw = self._get_safe_series(df, panic_sig, 0.0, method_name) 
        closing_strength_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name)
        upward_purity_raw = self._get_safe_series(df, purity_sig, 0.0, method_name)
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        stability_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        behavior_deception_index = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0)
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        holder_sentiment_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        absorption_strength_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 0.0)
        offensive_absorption_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
        mf_buy_execution_alpha_raw = self._get_safe_series(df, 'main_force_buy_execution_alpha_D', 0.0, method_name)
        buy_sweep_intensity_raw = self._get_safe_series(df, 'buy_sweep_intensity_D', 0.0, method_name)
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        mtf_cohesion_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0)
        # Context Scores
        mtf_wash_trade_score = self._get_mtf_slope_accel_score(df, wash_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_active_selling_score = self._get_mtf_slope_accel_score(df, active_sell_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_deception_lure_short_score = self._get_mtf_slope_accel_score(df, 'deception_lure_short_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        behavior_deception_score_negative = behavior_deception_index.clip(upper=0).abs()
        stealth_ops_normalized = self._normalize_series(stealth_ops_score, df_index, bipolar=False)
        mtf_wash_trade_slope_score = self._get_mtf_slope_accel_score(df, wash_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_active_selling_slope_score = self._get_mtf_slope_accel_score(df, active_sell_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        deception_context_score = (
            (mtf_wash_trade_score).pow(deception_context_weights.get('wash_trade', 0.2)) *
            (mtf_active_selling_score).pow(deception_context_weights.get('active_selling', 0.15)) *
            (mtf_deception_lure_short_score).pow(deception_context_weights.get('deception_lure_short', 0.2)) *
            (behavior_deception_score_negative).pow(deception_context_weights.get('behavior_deception_index', 0.2)) *
            (stealth_ops_normalized).pow(deception_context_weights.get('stealth_ops', 0.15)) *
            (mtf_wash_trade_slope_score).pow(deception_context_weights.get('wash_trade_slope', 0.05)) *
            (mtf_active_selling_slope_score).pow(deception_context_weights.get('active_selling_slope', 0.05))
        ).pow(1/sum(deception_context_weights.values())).fillna(0.0)
        # Panic Scores
        panic_cascade_score = self._normalize_series(panic_cascade_raw, df_index, bipolar=False)
        mtf_retail_surrender_score = self._get_mtf_slope_accel_score(df, panic_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_loser_pain_score = self._get_mtf_slope_accel_score(df, panic_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False) # 使用相同信号代替
        holder_sentiment_inverted_score = (1 - holder_sentiment_score).clip(0, 1)
        sentiment_pendulum_negative_score = sentiment_pendulum_score.clip(upper=0).abs()
        mtf_retail_surrender_slope_score = self._get_mtf_slope_accel_score(df, panic_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_loser_pain_slope_score = self._get_mtf_slope_accel_score(df, panic_sig, mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        panic_depth_score = (
            (panic_cascade_score).pow(panic_depth_weights.get('panic_cascade', 0.2)) *
            (mtf_retail_surrender_score).pow(panic_depth_weights.get('retail_surrender', 0.2)) *
            (mtf_loser_pain_score).pow(panic_depth_weights.get('loser_pain', 0.2)) *
            (holder_sentiment_inverted_score).pow(panic_depth_weights.get('holder_sentiment_inverted', 0.15)) *
            (sentiment_pendulum_negative_score).pow(panic_depth_weights.get('sentiment_pendulum_negative', 0.15)) *
            (mtf_retail_surrender_slope_score).pow(panic_depth_weights.get('retail_surrender_slope', 0.075)) *
            (mtf_loser_pain_slope_score).pow(panic_depth_weights.get('loser_pain_slope', 0.075))
        ).pow(1/sum(panic_depth_weights.values())).fillna(0.0)
        # Rebound Quality
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
        # Final Synthesis
        wash_out_rebound_score_base = (
            (deception_context_score).pow(fusion_weights.get('deception_context', 0.3)) *
            (panic_depth_score).pow(fusion_weights.get('panic_depth', 0.3)) *
            (rebound_quality_score).pow(fusion_weights.get('rebound_quality', 0.4))
        ).pow(1/(fusion_weights.get('deception_context', 0.3) + fusion_weights.get('panic_depth', 0.3) + fusion_weights.get('rebound_quality', 0.4))).fillna(0.0)
        trend_form_norm = trend_form_score.clip(lower=0)
        stability_norm = stability_score
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





