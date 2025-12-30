# 文件: strategies/trend_following/intelligence/process_intelligence.py
import json
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    get_adaptive_mtf_normalized_score, is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score, _robust_geometric_mean
)


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
        【V3.4 · 探针优化版】
        - 核心修复: 彻底移除在代码中硬编码的 `genesis_diagnostics` 列表。
        - 核心升级: 确保 `process_intelligence_params.diagnostics` 配置是诊断任务的唯一真相来源，
                      消除了重复执行的严重BUG，并遵循了“配置即代码”的最佳实践。
        - 支持生成原子情报领域的反转信号。
        - 优化: 统一在构造函数中获取探针配置，避免在各方法中重复读取。
        """
        self.strategy = strategy_instance
        # 直接从新文件加载 process_intelligence_params
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..'))
        process_config_path = os.path.join(project_root, 'config', 'intelligence', 'process.json')
        try:
            with open(process_config_path, 'r', encoding='utf-8') as f:
                self.params = json.load(f).get('process_intelligence_params', {})
        except FileNotFoundError:
            print(f"警告: 未找到过程情报配置文件 {process_config_path}，使用默认空配置。")
            self.params = {}
        except json.JSONDecodeError:
            print(f"警告: 过程情报配置文件 {process_config_path} 解析失败，使用默认空配置。")
            self.params = {}
        # score_type_map 和 debug_params 仍通过 get_params_block 获取，假设其能正确处理
        self.score_type_map = get_params_block(self.strategy, 'score_type_map', {})
        self.norm_window = get_param_value(self.params.get('norm_window'), 55)
        self.std_window = get_param_value(self.params.get('std_window'), 21)
        self.meta_window = get_param_value(self.params.get('meta_window'), 5)
        self.bipolar_sensitivity = get_param_value(self.params.get('bipolar_sensitivity'), 1.0)
        self.meta_score_weights = get_param_value(self.params.get('meta_score_weights'), [0.6, 0.4])
        self.diagnostics_config = get_param_value(self.params.get('diagnostics'), [])
        # 统一获取调试参数和探针日期，提高代码效率和健壮性
        self.debug_params = get_params_block(self.strategy, 'debug_params', {})
        self.probe_dates = get_param_value(self.debug_params.get('probe_dates'), [])

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        根据最新要求，此方法不再对获取到的Series进行fillna(default_value)操作，以暴露原始数据中的NaN问题。
        """
        if column_name not in df.columns:
            print(f"    -> [过程情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        # 移除 .fillna(default_value)
        return df[column_name]

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
        # [修改] 定义需要优先计算的“基石信号”清单，新增 PROCESS_META_COVERT_ACCUMULATION
        priority_signals = [
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_COVERT_ACCUMULATION'
        ]
        # [修改] 依赖前置处理逻辑，遍历基石信号清单
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
                    print(f"    -> [过程层] {signal_name} (基石信号) 计算完成，最新分值: {latest_score:.4f}")
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

    def _calculate_power_transfer(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【生产版】计算“权力转移”信号。
        - 核心逻辑: 融合“主力信念”、“战场清晰度”（由对倒和欺骗构成）来计算资金转移的真实性，
                      并对最终结果进行非线性放大，以捕捉市场的极端博弈。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_POWER_TRANSFER...")
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

    def _calculate_main_force_rally_intent(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.8 · 全息趋势与深度情境风险审判版】计算“主力拉升意图”的专属关系分数。
        - 核心升级: 将核心信号（价格变化、主力净流）升级为多时间维度（MTF）斜率/加速度融合，
                      更鲁棒地捕捉趋势和动能。
        - 风险审判强化: 增强风险审判的多时间维度感知，引入更长期的下跌趋势和资金流出背离。
        - 涨停日处理优化: 移除涨停日无条件奖励，改为根据 MTF 融合后的风险信号进行动态调整。
        - 【重要修正】`pre_drop_risk_factor` 深度情境感知：不再仅仅判断前一日是否下跌，
                      而是检查更长的周期，判断是否处于“上升波浪结束后开始下跌”的趋势中，
                      以正确分辨超跌反弹和下跌中继。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_RALLY_INTENT (V5.8 · 全息趋势与深度情境风险审判版)...")
        method_name = "_calculate_main_force_rally_intent"
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = [
            'pct_change_D', 'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D',
            'upward_impulse_purity_D', 'volume_ratio_D', 'control_solidity_index_D',
            'main_force_cost_advantage_D', 'SLOPE_5_winner_concentration_90pct_D',
            'dominant_peak_solidity_D', 'active_buying_support_D', 'pressure_rejection_strength_D',
            'profit_realization_quality_D', 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH',
            'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 'distribution_at_peak_intensity_D',
            'upper_shadow_selling_pressure_D', 'flow_credibility_index_D', 'chip_health_score_D',
            'retail_fomo_premium_index_D',
            'SLOPE_21_close_D', 'ACCEL_21_close_D', # 新增中长期趋势信号
            'SLOPE_34_close_D', 'ACCEL_34_close_D' # 进一步增加中长期趋势信号
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['close_D', 'main_force_net_flow_calibrated_D', 'upper_shadow_selling_pressure_D', 'retail_fomo_premium_index_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        relative_strength = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        capital_signature = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 0.0)
        cs_modulator_weight = config.get('capital_signature_modulator_weight', 0.0)
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        # --- 原始数据获取 ---
        mtf_price_trend = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mf_net_flow = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        prev_day_pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name).shift(1).fillna(0)
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name) # 用于计算从高点回落
        # 新增中长期趋势信号
        slope_21_close = self._get_safe_series(df, 'SLOPE_21_close_D', 0.0, method_name=method_name)
        accel_21_close = self._get_safe_series(df, 'ACCEL_21_close_D', 0.0, method_name=method_name)
        slope_34_close = self._get_safe_series(df, 'SLOPE_34_close_D', 0.0, method_name=method_name)
        accel_34_close = self._get_safe_series(df, 'ACCEL_34_close_D', 0.0, method_name=method_name)
        main_force_slippage = self._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name)
        upward_impulse_purity = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name=method_name)
        control_solidity = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name)
        concentration_slope = self._get_safe_series(df, f'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name=method_name)
        peak_solidity = self._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name=method_name)
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        pressure_rejection = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name=method_name)
        profit_realization_quality = self._get_safe_series(df, 'profit_realization_quality_D', 0.5, method_name=method_name)
        distribution_at_peak_intensity = self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name=method_name)
        upper_shadow_selling_pressure = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name)
        flow_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        chip_health = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name)
        retail_fomo = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        price_impact_norm = self._normalize_series(main_force_slippage, df_index, bipolar=True)
        impulse_purity_norm = self._normalize_series(upward_impulse_purity, df_index, bipolar=True)
        volume_ratio_norm = self._normalize_series(volume_ratio - 1.0, df_index, bipolar=True)
        control_solidity_norm = self._normalize_series(control_solidity, df_index, bipolar=True)
        cost_advantage_norm = self._normalize_series(cost_advantage, df_index, bipolar=True)
        concentration_slope_norm = self._normalize_series(concentration_slope, df_index, bipolar=True)
        peak_solidity_norm = self._normalize_series(peak_solidity, df_index, bipolar=True)
        buying_support_norm = self._normalize_series(active_buying_support, df_index, bipolar=True)
        pressure_rejection_norm = self._normalize_series(pressure_rejection, df_index, bipolar=True)
        profit_absorption_norm = self._normalize_series((1 - profit_realization_quality) - 0.5, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        chip_health_norm = self._normalize_series(chip_health, df_index, bipolar=False)
        retail_fomo_norm = self._normalize_series(retail_fomo, df_index, bipolar=False)
        # --- 1. 攻击性 (Aggressiveness) ---
        aggressiveness_score = (
            mtf_price_trend.clip(lower=0) * 0.20 +
            mtf_mf_net_flow.clip(lower=0) * 0.20 +
            price_impact_norm.clip(lower=0) * 0.15 +
            impulse_purity_norm.clip(lower=0) * 0.15 +
            volume_ratio_norm.clip(lower=0) * 0.10 +
            flow_credibility_norm * 0.10 +
            chip_health_norm * 0.10
        ).clip(0, 1)
        # --- 2. 控制力 (Control) ---
        control_score = (
            control_solidity_norm.clip(lower=0) * 0.35 +
            cost_advantage_norm.clip(lower=0) * 0.25 +
            concentration_slope_norm.clip(lower=0) * 0.20 +
            peak_solidity_norm.clip(lower=0) * 0.20
        ).clip(0, 1)
        # --- 3. 障碍清除 (Obstacle Clearance) ---
        obstacle_clearance_score = (
            buying_support_norm.clip(lower=0) * 0.40 +
            pressure_rejection_norm.clip(lower=0) * 0.30 +
            profit_absorption_norm.clip(lower=0) * 0.30
        ).clip(0, 1)
        # --- 4. 基础看涨意图 (Bullish Intent) ---
        bullish_intent = (aggressiveness_score * control_score * obstacle_clearance_score).pow(1/3)
        # --- 5. 看跌意图 (Bearish Intent) ---
        bearish_mask = (mtf_price_trend < 0) | (mtf_mf_net_flow < 0)
        bearish_score = (mtf_price_trend.clip(upper=0).abs() * 0.5 + mtf_mf_net_flow.clip(upper=0).abs() * 0.5).clip(0, 1) * -1
        # --- 6. 风险审判模块 (Risk Adjudication) ---
        # 6.1. 派发风险 (Distribution Risk) - 引入MTF斜率
        mtf_upper_shadow_pressure = self._get_mtf_slope_accel_score(df, 'upper_shadow_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_retail_fomo = self._get_mtf_slope_accel_score(df, 'retail_fomo_premium_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        distribution_intensity_norm = self._normalize_series(distribution_at_peak_intensity, df_index, bipolar=False)
        mf_outflow_divergence = mtf_mf_net_flow.clip(upper=0).abs()
        distribution_risk_score = (
            distribution_intensity_norm * 0.25 +
            mtf_upper_shadow_pressure * 0.25 +
            mf_outflow_divergence * 0.25 +
            mtf_retail_fomo * 0.25
        ).clip(0, 1)
        # 6.2. 前置下跌风险 (Pre-Drop Risk) - 深度情境感知
        # 短期累计跌幅
        pre_5day_pct_change = df['close_D'].pct_change(periods=5).shift(1).fillna(0)
        pre_13day_pct_change = df['close_D'].pct_change(periods=13).shift(1).fillna(0)
        norm_pre_drop_5d = self._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        norm_pre_drop_13d = self._normalize_series(pre_13day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        # 前一日的下跌风险
        single_day_drop_risk = self._normalize_series(prev_day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        # 中长期趋势反转/下跌确认
        # 21日和34日斜率和加速度的负向强度
        norm_slope_21_neg = self._normalize_series(slope_21_close.clip(upper=0).abs(), df_index, bipolar=False)
        norm_accel_21_neg = self._normalize_series(accel_21_close.clip(upper=0).abs(), df_index, bipolar=False)
        norm_slope_34_neg = self._normalize_series(slope_34_close.clip(upper=0).abs(), df_index, bipolar=False)
        norm_accel_34_neg = self._normalize_series(accel_34_close.clip(upper=0).abs(), df_index, bipolar=False)
        medium_term_downtrend_strength = (norm_slope_21_neg * 0.3 + norm_accel_21_neg * 0.2 + 
                                          norm_slope_34_neg * 0.3 + norm_accel_34_neg * 0.2).clip(0, 1)
        # 从近期高点回落幅度
        # 21日内最高价
        high_21d = close_price.rolling(window=21).max()
        # 从21日高点回落的百分比，并归一化
        fall_from_peak_21d = (1 - close_price / high_21d).clip(lower=0).fillna(0)
        norm_fall_from_peak_21d = self._normalize_series(fall_from_peak_21d, df_index, bipolar=False)
        # 综合前置下跌风险
        pre_drop_risk_factor = (
            single_day_drop_risk * 0.2 + # 短期下跌
            norm_pre_drop_5d * 0.2 + # 5日累计下跌
            norm_pre_drop_13d * 0.1 + # 13日累计下跌
            medium_term_downtrend_strength * 0.3 + # 中长期下跌趋势确认
            norm_fall_from_peak_21d * 0.2 # 从高点回落幅度
        ).clip(0, 1) * 0.7 # 整体风险因子权重
        # 6.3. 综合风险惩罚因子
        total_risk_penalty = (distribution_risk_score * 0.5 + pre_drop_risk_factor * 0.5).clip(0, 1)
        # --- 7. 最终意图合成 ---
        penalized_bullish_part = bullish_intent * (1 - total_risk_penalty)
        final_rally_intent = (penalized_bullish_part + bearish_score).clip(-1, 1)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (total_risk_penalty > 0.5), final_rally_intent * (1 - total_risk_penalty))
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0)
        # --- 8. 相对强度和资本属性调节 ---
        rs_modulator = (1 + relative_strength * rs_amplifier)
        capital_modulator = (1 + capital_signature * cs_modulator_weight)
        final_rally_intent = (final_rally_intent * rs_modulator * capital_modulator).clip(-1, 1)
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score
        self.strategy.atomic_states["_DEBUG_rally_distribution_risk"] = distribution_risk_score
        self.strategy.atomic_states["_DEBUG_rally_pre_drop_risk_factor"] = pre_drop_risk_factor
        self.strategy.atomic_states["_DEBUG_rally_total_risk_penalty"] = total_risk_penalty
        return final_rally_intent.astype(np.float32)

    def _calculate_main_force_control_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.3 · 控盘杠杆与全息资金流验证强化版】计算“主力控盘”的专属关系分数。
        - 核心重构: 创立“控盘即杠杆”模型。将“控盘度”作为调节“资金流向”影响力的核心杠杆。
                      最终分 = 主力净流入分 * (1 + 融合控盘分)。
        - 证据升级: 融合传统的均线控盘度与更现代的“控盘稳固度”，形成更立体的控盘评分。
        - 【强化】引入主力资金净流向和资金流可信度作为控盘杠杆的调节因子，确保控盘的积极性。
        - 【重要修改】修正 `control_leverage` 逻辑，当控盘不强时，对资金流入进行更强的惩罚。
        - 【新增】引入 MTF 控盘信号和 MTF 主力资金净流，增强信号鲁棒性。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_CONTROL (V2.3 · 控盘杠杆与全息资金流验证强化版)...")
        method_name = "_calculate_main_force_control_relationship"
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        required_signals = ['close_D', 'main_force_net_flow_calibrated_D', 'control_solidity_index_D', 'flow_credibility_index_D']
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['control_solidity_index_D', 'main_force_net_flow_calibrated_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # --- 原始数据获取 ---
        close_price = self._get_safe_series(df, 'close_D', method_name=method_name)
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        # --- 1. 传统控盘度计算 ---
        ema13 = ta.ema(close=close_price, length=13, append=False)
        if ema13 is None:
            print(f"    -> [过程情报警告] {method_name} EMA_13 计算失败，返回默认值。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        varn1 = ta.ema(close=ema13, length=13, append=False)
        if varn1 is None:
            print(f"    -> [过程情报警告] {method_name} VARN1 计算失败，返回默认值。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        prev_varn1 = varn1.shift(1).replace(0, np.nan)
        kongpan_raw = (varn1 - prev_varn1) / prev_varn1 * 1000
        # --- 2. 归一化处理 ---
        traditional_control_score = self._normalize_series(kongpan_raw, df_index, bipolar=True)
        # 结构控盘度：使用MTF融合信号
        mtf_structural_control_score = self._get_mtf_slope_accel_score(df, 'control_solidity_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 主力资金流分：使用MTF融合信号
        mtf_main_force_flow_score = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        # --- 3. 融合控盘分 ---
        fused_control_score = (traditional_control_score * 0.4 + mtf_structural_control_score * 0.6).clip(-1, 1)
        # --- 4. 控盘杠杆模型 ---
        # 杠杆效应：当控盘为正时，放大资金流入；当控盘为负时，抑制资金流入，甚至反向惩罚
        mf_inflow_validation = mtf_main_force_flow_score.clip(lower=0) * flow_credibility_norm
        control_leverage = pd.Series(1.0, index=df_index, dtype=np.float32)
        # 控盘强 (fused_control_score > 0)，则放大资金流入
        control_leverage = control_leverage.mask(fused_control_score > 0, 1 + fused_control_score * mf_inflow_validation)
        # 控盘弱或负 (fused_control_score <= 0)，则更强地抑制资金流入，甚至反向惩罚
        # 惩罚因子可以是非线性的，例如 (1 + fused_control_score) * (1 - mf_inflow_validation)
        control_leverage = control_leverage.mask(fused_control_score <= 0, (1 + fused_control_score) * (1 - mf_inflow_validation * 0.5)) # 惩罚力度增强
        control_leverage = control_leverage.clip(0, 2) # 限制杠杆范围，避免过大或过小的负值
        # --- 5. 最终控盘分数 ---
        final_control_score = (mtf_main_force_flow_score * control_leverage).clip(-1, 1)
        return final_control_score.astype(np.float32)

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.16 · 价势背离升级版】对“关系分”进行元分析，输出分数。
        - 核心升级: 新增对“价势背离”信号的专属路由，执行其诊断逻辑。
        """
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
        elif signal_name == 'PROCESS_META_PV_REL_BULLISH_TURN':
            relationship_score = self._calculate_price_volume_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_RALLY_INTENT':
            meta_score = self._calculate_main_force_rally_intent(df, config)
        elif signal_name == 'PROCESS_META_WINNER_CONVICTION':
            relationship_score = self._calculate_winner_conviction_relationship(df, config)
            meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df_index)
        elif signal_name == 'PROCESS_META_LOSER_CAPITULATION':
            meta_score = self._calculate_loser_capitulation(df, config)
        elif signal_name == 'PROCESS_META_COST_ADVANTAGE_TREND':
            meta_score = self._calculate_cost_advantage_trend_relationship(df, config)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_CONTROL':
            meta_score = self._calculate_main_force_control_relationship(df, config)
        elif signal_name == 'PROCESS_META_STEALTH_ACCUMULATION':
            meta_score = self._calculate_stealth_accumulation(df, config)
        elif signal_name == 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION':
            meta_score = self._calculate_panic_washout_accumulation(df, config)
        elif signal_name == 'PROCESS_META_DECEPTIVE_ACCUMULATION':
            meta_score = self._calculate_deceptive_accumulation(df, config)
        elif signal_name == 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY':
            meta_score = self._calculate_split_order_accumulation(df, config)
        elif signal_name == 'PROCESS_META_UPTHRUST_WASHOUT':
            meta_score = self._calculate_upthrust_washout(df, config)
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
            meta_score = self._calculate_price_momentum_divergence(df, config)
        elif signal_name == 'PROCESS_META_STORM_EYE_CALM':
            meta_score = self._calculate_storm_eye_calm(df, config)
        elif signal_name == 'PROCESS_META_WASH_OUT_REBOUND':
            offensive_absorption_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
            meta_score = self._calculate_process_wash_out_rebound(df, offensive_absorption_intent, config) # 传递 config
        elif signal_name == 'PROCESS_META_COVERT_ACCUMULATION':
            meta_score = self._calculate_process_covert_accumulation(df, config) # <--- 修正此处，传递 config
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
        print(f"    -> [过程层] {signal_name} 计算完成，最新分值: {meta_score.iloc[-1]:.4f}")
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
        【V3.0 · 承接验证版】计算“价格与散户投降”的专属瞬时关系分。
        - 核心升级: 引入 `active_buying_support_D` (主动承接强度) 作为“真实性放大器”。
                      只有当“价跌慌不增”的表象伴随主力真实承接时，信号才会被放大，
                      以此区分“死寂”与“黄金坑”。
        """
        required_signals = ['close_D', 'retail_panic_surrender_index_D', 'active_buying_support_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_price_vs_capitulation_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        # 引入主动承接作为真实性放大器
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_price_vs_capitulation_relationship")
        active_buying_norm = self._normalize_series(active_buying_support, df.index, bipolar=False)
        authenticity_amplifier = 1 + active_buying_norm
        final_score = (base_divergence_score * authenticity_amplifier).clip(-1, 1)
        return final_score

    def _calculate_price_efficiency_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 信念校准版】计算“价格效率”的专属瞬时关系分。
        - 核心升级: 引入由“主力信念”和“对倒强度”构成的“品质因子”，对原始的价效共识分
                      进行“血统校准”，优先采纳由高信念、低噪音资金驱动的行为。
        """
        required_signals = ['close_D', 'VPA_EFFICIENCY_D', 'main_force_conviction_index_D', 'wash_trade_intensity_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_price_efficiency_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 计算基础共识分数
        base_consensus_score = self._calculate_instantaneous_relationship(df, config)
        # 引入品质因子进行校准
        main_force_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_price_efficiency_relationship")
        wash_trade_intensity = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_price_efficiency_relationship")
        conviction_norm = self._normalize_series(main_force_conviction, df.index, bipolar=True)
        wash_trade_norm = self._normalize_series(wash_trade_intensity, df.index, bipolar=False)
        quality_factor = (conviction_norm.clip(lower=0) * (1 - wash_trade_norm)).clip(0, 1)
        final_score = (base_consensus_score * quality_factor).clip(-1, 1)
        return final_score

    def _calculate_pd_divergence_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 战场纵深版】计算“博弈背离”的专属瞬时关系分。
        - 核心升级: 引入基于 `main_force_vpoc_D` 的“战场纵深因子”，为博弈背离信号
                      赋予战略坐标。当背离发生在主力成本线之上时，信号将被放大，反之则被抑制。
        """
        required_signals = ['close_D', 'mf_retail_battle_intensity_D', 'main_force_vpoc_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_pd_divergence_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 计算基础背离分数
        base_divergence_score = self._calculate_instantaneous_relationship(df, config)
        # 引入战场纵深因子
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name="_calculate_pd_divergence_relationship")
        mf_vpoc = self._get_safe_series(df, 'main_force_vpoc_D', 0.0, method_name="_calculate_pd_divergence_relationship")
        mf_vpoc_safe = mf_vpoc.replace(0, np.nan) # 防止除以零
        battlefield_context_factor = (1 + (close_price - mf_vpoc_safe) / mf_vpoc_safe).fillna(1).clip(0, 2)
        final_score = (base_divergence_score * battlefield_context_factor).clip(-1, 1)
        return final_score

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.5 · 信念侵蚀版】信号衰减诊断器
        - 核心升级: 为“赢家信念衰减”信号分派专属计算引擎，执行全新的“信念侵蚀”逻辑。
        """
        signal_name = config.get('name')
        if signal_name == 'PROCESS_META_WINNER_CONVICTION_DECAY':
            decay_score = self._calculate_winner_conviction_decay(df, config)
            return {signal_name: decay_score.astype(np.float32)}
        source_signal_name = config.get('source_signal')
        source_type = config.get('source_type', 'df')
        df_index = df.index
        if not source_signal_name:
            print(f"        -> [衰减分析] 警告: 缺少 'source_signal' 配置。")
            return {}
        source_series = None
        if source_type == 'atomic_states':
            source_series = self.strategy.atomic_states.get(source_signal_name)
        else:
            source_series = self._get_safe_series(df, source_signal_name, method_name="_diagnose_signal_decay")
        if source_series is None:
            print(f"        -> [衰减分析] 警告: 缺少源信号 '{source_signal_name}'。")
            return {}
        signal_change = source_series.diff(1).fillna(0)
        decay_magnitude = signal_change.clip(upper=0).abs()
        decay_score = self._normalize_series(decay_magnitude, df_index, ascending=True)
        return {signal_name: decay_score.astype(np.float32)}

    def _calculate_price_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        print("    -> [过程层] 正在计算 PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE (V4.1 · 深度情境与动态权重版)...")
        df_index = df.index
        params = get_param_value(config.get('price_momentum_divergence_params'), {})
        # --- 1. 获取配置参数 ---
        price_components_weights = get_param_value(params.get('price_components_weights'), {"close_D": 0.6, "upward_efficiency": 0.2, "price_momentum_quality": 0.2})
        momentum_components_weights = get_param_value(params.get('momentum_components_weights'), {"MACDh_13_34_8_D": 0.5, "RSI_13_D": 0.3, "ROC_13_D": 0.2, "momentum_quality": 0.2})
        mtf_slope_weights = get_param_value(params.get('mtf_slope_weights'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        mtf_accel_weights = get_param_value(params.get('mtf_accel_weights'), {"5": 0.6, "13": 0.4})
        volume_confirmation_weights = get_param_value(params.get('volume_confirmation_weights'), {"volume_slope": 0.5, "volume_burst": 0.2, "volume_atrophy": 0.3, "constructive_turnover": 0.1, "volume_structure_skew_inverted": 0.1})
        dynamic_volume_confirmation_modulators = get_param_value(params.get('dynamic_volume_confirmation_modulators'), {"enabled": False})
        main_force_confirmation_weights = get_param_value(params.get('main_force_confirmation_weights'), {"mf_net_flow_slope": 0.4, "deception_index": 0.2, "distribution_intent": 0.2, "covert_accumulation": 0.1, "chip_divergence": 0.1, "main_force_conviction": 0.1, "chip_health": 0.1})
        dynamic_main_force_confirmation_modulators = get_param_value(params.get('dynamic_main_force_confirmation_modulators'), {"enabled": False})
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {"volatility_inverse": 0.3, "trend_strength_inverse": 0.2, "sentiment_neutrality": 0.2, "liquidity_tide_calm": 0.15, "market_constitution_neutrality": 0.15})
        divergence_quality_weights = get_param_value(params.get('divergence_quality_weights'), {"duration": 0.4, "depth": 0.3, "stability": 0.15, "chip_potential": 0.15})
        final_fusion_exponent = get_param_value(params.get('final_fusion_exponent'), 1.5)
        synergy_threshold = get_param_value(params.get('synergy_threshold'), 0.6)
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.1)
        conflict_penalty_factor = get_param_value(params.get('conflict_penalty_factor'), 0.15)
        dynamic_fusion_weights_params = get_param_value(params.get('dynamic_fusion_weights_params'), {"enabled": False})
        # --- 2. 校验所有必需的信号 ---
        valid_mtf_periods = [p_str for p_str in mtf_slope_weights.keys() if p_str.isdigit()]
        required_signals = [
            *[f'SLOPE_{p}_close_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_MACDh_13_34_8_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_RSI_13_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_ROC_13_D' for p in valid_mtf_periods],
            *[f'SLOPE_{p}_volume_D' for p in valid_mtf_periods],
            'volume_burstiness_index_D', 'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            *[f'SLOPE_{p}_main_force_net_flow_calibrated_D' for p in valid_mtf_periods],
            'deception_index_D', 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 'SCORE_CHIP_AXIOM_DIVERGENCE',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', 'market_sentiment_score_D',
            'PROCESS_META_COVERT_ACCUMULATION',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM',
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
            'SCORE_DYN_AXIOM_MOMENTUM',
            'constructive_turnover_ratio_D',
            'volume_structure_skew_D',
            'main_force_conviction_index_D',
            'chip_health_score_D',
            'SCORE_DYN_AXIOM_STABILITY',
            'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL',
            'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE',
            'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION',
            'SCORE_FOUNDATION_AXIOM_MARKET_TENSION'
        ]
        for p_str in mtf_accel_weights.keys():
            p = int(p_str)
            required_signals.append(f'ACCEL_{p}_close_D')
            required_signals.append(f'ACCEL_{p}_MACDh_13_34_8_D')
            required_signals.append(f'ACCEL_{p}_RSI_13_D')
            required_signals.append(f'ACCEL_{p}_ROC_13_D')
            required_signals.append(f'ACCEL_{p}_volume_D')
            required_signals.append(f'ACCEL_{p}_main_force_net_flow_calibrated_D')
        if not self._validate_required_signals(df, required_signals, "_calculate_price_momentum_divergence"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 3. 获取原始数据 ---
        price_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_close_D', 0.0, method_name="_calculate_price_momentum_divergence") for p in valid_mtf_periods}
        macdh_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_MACDh_13_34_8_D', 0.0, method_name="_calculate_price_momentum_divergence") for p in valid_mtf_periods}
        rsi_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_RSI_13_D', 0.0, method_name="_calculate_price_momentum_divergence") for p in valid_mtf_periods}
        roc_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_ROC_13_D', 0.0, method_name="_calculate_price_momentum_divergence") for p in valid_mtf_periods}
        volume_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_volume_D', 0.0, method_name="_calculate_price_momentum_divergence") for p in valid_mtf_periods}
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name="_calculate_price_momentum_divergence")
        volume_atrophy_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        mf_net_flow_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_price_momentum_divergence") for p in valid_mtf_periods}
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_calculate_price_momentum_divergence")
        distribution_intent_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        covert_accumulation_score = self._get_atomic_score(df, 'PROCESS_META_COVERT_ACCUMULATION', 0.0)
        chip_divergence_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', 0.0)
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_calculate_price_momentum_divergence")
        adx_raw = self._get_safe_series(df, 'ADX_14_D', 0.0, method_name="_calculate_price_momentum_divergence")
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name="_calculate_price_momentum_divergence")
        upward_efficiency_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.0)
        price_upward_momentum_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM', 0.0)
        price_downward_momentum_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0)
        momentum_quality_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_MOMENTUM', 0.0)
        constructive_turnover_raw = self._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name="_calculate_price_momentum_divergence")
        volume_structure_skew_raw = self._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name="_calculate_price_momentum_divergence")
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_price_momentum_divergence")
        chip_health_raw = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name="_calculate_price_momentum_divergence")
        stability_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        chip_historical_potential_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        liquidity_tide_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.0)
        market_constitution_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION', 0.0)
        market_tension_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0)
        # --- 4. 计算各维度分数 ---
        # 4.1. Fused Price Direction (MTF Slope Fusion)
        fused_price_direction_base = self._get_mtf_slope_score(df, 'close_D', mtf_slope_weights, df_index, "_calculate_price_momentum_divergence", bipolar=True)
        price_momentum_quality_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        bullish_price_momentum_quality = (price_upward_momentum_score * upward_efficiency_score).pow(0.5)
        bearish_price_momentum_quality = price_downward_momentum_score
        price_momentum_quality_score = bullish_price_momentum_quality.where(fused_price_direction_base > 0, -bearish_price_momentum_quality)
        fused_price_direction_components = {
            "close_D": fused_price_direction_base,
            "upward_efficiency": upward_efficiency_score,
            "price_momentum_quality": price_momentum_quality_score
        }
        fused_price_direction = _robust_geometric_mean(fused_price_direction_components, price_components_weights, df_index)
        # 4.2. Fused Momentum Direction (Multi-Indicator & MTF Fusion)
        fused_macdh_direction = self._get_mtf_slope_score(df, 'MACDh_13_34_8_D', mtf_slope_weights, df_index, "_calculate_price_momentum_divergence", bipolar=True)
        fused_rsi_direction = self._get_mtf_slope_score(df, 'RSI_13_D', mtf_slope_weights, df_index, "_calculate_price_momentum_divergence", bipolar=True)
        fused_roc_direction = self._get_mtf_slope_score(df, 'ROC_13_D', mtf_slope_weights, df_index, "_calculate_price_momentum_divergence", bipolar=True)
        fused_momentum_direction_components = {
            "MACDh_13_34_8_D": fused_macdh_direction,
            "RSI_13_D": fused_rsi_direction,
            "ROC_13_D": fused_roc_direction,
            "momentum_quality": momentum_quality_score
        }
        momentum_components_weights_extended = momentum_components_weights.copy()
        momentum_components_weights_extended["momentum_quality"] = get_param_value(params.get('momentum_components_weights', {}).get("momentum_quality"), 0.2)
        fused_momentum_direction = _robust_geometric_mean(fused_momentum_direction_components, momentum_components_weights_extended, df_index)
        # 4.3. Base Divergence Score
        base_divergence_score = (fused_price_direction - fused_momentum_direction).clip(-1, 1)
        # 4.4. Volume Confirmation Score
        fused_volume_slope = self._get_mtf_slope_score(df, 'volume_D', mtf_slope_weights, df_index, "_calculate_price_momentum_divergence", bipolar=True)
        volume_burst_norm = self._normalize_series(volume_burstiness_raw, df_index, ascending=True)
        volume_atrophy_norm = self._normalize_series(volume_atrophy_score, df_index, ascending=True)
        constructive_turnover_norm = self._normalize_series(constructive_turnover_raw, df_index, ascending=True)
        volume_structure_skew_inverted_norm = self._normalize_series(volume_structure_skew_raw.abs(), df_index, ascending=False)
        # 动态调节量能确认权重
        current_volume_confirmation_weights = volume_confirmation_weights.copy()
        if get_param_value(dynamic_volume_confirmation_modulators.get('enabled'), False):
            modulator_signal_raw = self._get_atomic_score(df, dynamic_volume_confirmation_modulators['modulator_signal'], 0.0)
            modulator_signal = self._normalize_series(modulator_signal_raw, df_index, bipolar=True)
            sensitivity = dynamic_volume_confirmation_modulators['sensitivity']
            min_factor = dynamic_volume_confirmation_modulators['min_factor']
            max_factor = dynamic_volume_confirmation_modulators['max_factor']
            modulator_factor = (1 + modulator_signal * sensitivity).clip(min_factor, max_factor)
            for k in current_volume_confirmation_weights:
                current_volume_confirmation_weights[k] = current_volume_confirmation_weights[k] * modulator_factor
        top_vol_conf_components = {
            "volume_slope_negative": fused_volume_slope.clip(upper=0).abs(),
            "volume_burst": volume_burst_norm,
            "constructive_turnover": constructive_turnover_norm,
            "volume_structure_skew_inverted": volume_structure_skew_inverted_norm
        }
        top_vol_conf = _robust_geometric_mean(top_vol_conf_components, current_volume_confirmation_weights, df_index)
        bottom_vol_conf_components = {
            "volume_slope_positive": fused_volume_slope.clip(lower=0),
            "volume_atrophy": volume_atrophy_norm,
            "constructive_turnover": constructive_turnover_norm,
            "volume_structure_skew_inverted": volume_structure_skew_inverted_norm
        }
        bottom_vol_conf = _robust_geometric_mean(bottom_vol_conf_components, current_volume_confirmation_weights, df_index)
        volume_confirmation_score = pd.Series([
            top_vol_conf.loc[idx] if x > 0 else (-bottom_vol_conf.loc[idx] if x < 0 else 0)
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        # 4.5. Main Force/Chip Confirmation Score
        fused_mf_net_flow_slope = self._get_mtf_slope_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_weights, df_index, "_calculate_price_momentum_divergence", bipolar=True)
        deception_index_norm = self._normalize_series(deception_index_raw, df_index, bipolar=True)
        distribution_intent_norm = self._normalize_series(distribution_intent_score, df_index, ascending=True)
        covert_accumulation_norm = self._normalize_series(covert_accumulation_score, df_index, ascending=True)
        chip_divergence_norm = self._normalize_series(chip_divergence_score, df_index, bipolar=True)
        main_force_conviction_norm = self._normalize_series(main_force_conviction_raw, df_index, bipolar=True)
        chip_health_norm = self._normalize_series(chip_health_raw, df_index, bipolar=False)
        # 动态调节主力/筹码确认权重
        current_main_force_confirmation_weights = main_force_confirmation_weights.copy()
        if get_param_value(dynamic_main_force_confirmation_modulators.get('enabled'), False):
            modulator_signal_raw = self._get_atomic_score(df, dynamic_main_force_confirmation_modulators['modulator_signal'], 0.0)
            modulator_signal = self._normalize_series(modulator_signal_raw, df_index, bipolar=True)
            sensitivity = dynamic_main_force_confirmation_modulators['sensitivity']
            min_factor = dynamic_main_force_confirmation_modulators['min_factor']
            max_factor = dynamic_main_force_confirmation_modulators['max_factor']
            modulator_factor = (1 + modulator_signal * sensitivity).clip(min_factor, max_factor)
            for k in current_main_force_confirmation_weights:
                current_main_force_confirmation_weights[k] = current_main_force_confirmation_weights[k] * modulator_factor
        top_mf_conf_components = {
            "mf_net_flow_slope_negative": fused_mf_net_flow_slope.clip(upper=0).abs(),
            "deception_index_positive": deception_index_norm.clip(lower=0),
            "distribution_intent": distribution_intent_norm,
            "chip_divergence_positive": chip_divergence_norm.clip(lower=0),
            "main_force_conviction": main_force_conviction_norm.clip(lower=0),
            "chip_health": chip_health_norm
        }
        top_mf_conf = _robust_geometric_mean(top_mf_conf_components, current_main_force_confirmation_weights, df_index)
        bottom_mf_conf_components = {
            "mf_net_flow_slope_positive": fused_mf_net_flow_slope.clip(lower=0),
            "deception_index_negative": deception_index_norm.clip(upper=0).abs(),
            "covert_accumulation": covert_accumulation_norm,
            "chip_divergence_negative": chip_divergence_norm.clip(upper=0).abs(),
            "main_force_conviction": main_force_conviction_norm.clip(upper=0).abs(),
            "chip_health": chip_health_norm
        }
        bottom_mf_conf = _robust_geometric_mean(bottom_mf_conf_components, current_main_force_confirmation_weights, df_index)
        main_force_confirmation_score = pd.Series([
            top_mf_conf.loc[idx] if x > 0 else (-bottom_mf_conf.loc[idx] if x < 0 else 0)
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        # 4.6. Divergence Quality Score
        is_top_divergence_bool = (base_divergence_score > 0.1)
        is_bottom_divergence_bool = (base_divergence_score < -0.1)
        top_divergence_duration = is_top_divergence_bool.astype(int).rolling(window=5, min_periods=1).sum()
        bottom_divergence_duration = is_bottom_divergence_bool.astype(int).rolling(window=5, min_periods=1).sum()
        top_divergence_duration_norm = (top_divergence_duration / 5).clip(0,1)
        bottom_divergence_duration_norm = (bottom_divergence_duration / 5).clip(0,1)
        divergence_depth_norm = base_divergence_score.abs()
        stability_norm = self._normalize_series(stability_score, df_index, bipolar=False)
        chip_potential_norm = self._normalize_series(chip_historical_potential_score, df_index, bipolar=False)
        divergence_quality_score = pd.Series([
            (_robust_geometric_mean(
                {"duration": pd.Series(top_divergence_duration_norm.loc[idx], index=[idx]),
                 "depth": pd.Series(divergence_depth_norm.loc[idx], index=[idx]),
                 "stability": pd.Series(stability_norm.loc[idx], index=[idx]),
                 "chip_potential": pd.Series(chip_potential_norm.loc[idx], index=[idx])},
                divergence_quality_weights,
                pd.Index([idx])
            ).iloc[0] if x > 0 else
             (_robust_geometric_mean(
                 {"duration": pd.Series(bottom_divergence_duration_norm.loc[idx], index=[idx]),
                  "depth": pd.Series(divergence_depth_norm.loc[idx], index=[idx]),
                  "stability": pd.Series(stability_norm.loc[idx], index=[idx]),
                  "chip_potential": pd.Series(chip_potential_norm.loc[idx], index=[idx])},
                 divergence_quality_weights,
                 pd.Index([idx])
             ).iloc[0] if x < 0 else 0))
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        # 4.7. Context Modulator
        volatility_instability_norm_inverted = self._normalize_series(volatility_instability_raw, df_index, ascending=False)
        adx_norm_inverted = self._normalize_series(adx_raw, df_index, ascending=False)
        market_sentiment_norm_bipolar = self._normalize_series(market_sentiment_raw, df_index, bipolar=True)
        liquidity_tide_calm_norm = self._normalize_series(liquidity_tide_score.abs(), df_index, ascending=False)
        market_constitution_neutrality_norm = 1 - self._normalize_series(market_constitution_score.abs(), df_index, ascending=True)
        context_modulator_components = {
            "volatility_inverse": volatility_instability_norm_inverted,
            "trend_strength_inverse": adx_norm_inverted,
            "sentiment_neutrality": 1 - market_sentiment_norm_bipolar.abs(),
            "liquidity_tide_calm": liquidity_tide_calm_norm,
            "market_constitution_neutrality": market_constitution_neutrality_norm
        }
        context_modulator = _robust_geometric_mean(context_modulator_components, context_modulator_weights, df_index)
        # --- 5. 最终融合 ---
        # Use a weighted geometric mean for robust fusion
        final_components = {
            "base_divergence": base_divergence_score.abs(),
            "volume_confirmation": volume_confirmation_score.abs(),
            "main_force_confirmation": main_force_confirmation_score.abs(),
            "divergence_quality": divergence_quality_score,
            "context_modulator": context_modulator
        }
        # Define fusion weights for the final geometric mean (these should be in config)
        final_fusion_weights_dict = get_param_value(params.get('dynamic_fusion_weights_params', {}).get('base_weights'), {
            "base_divergence": 0.3,
            "volume_confirmation": 0.2,
            "main_force_confirmation": 0.25,
            "divergence_quality": 0.15,
            "context_modulator": 0.1
        })
        # 动态调整最终融合权重
        if get_param_value(dynamic_fusion_weights_params.get('enabled'), False):
            modulator_signal_1_raw = self._get_atomic_score(df, dynamic_fusion_weights_params['modulator_signal_1'], 0.0)
            modulator_signal_2_raw = self._get_atomic_score(df, dynamic_fusion_weights_params['modulator_signal_2'], 0.0)
            modulator_signal_1 = self._normalize_series(modulator_signal_1_raw, df_index, bipolar=True) # 市场张力
            modulator_signal_2 = self._normalize_series(modulator_signal_2_raw, df_index, bipolar=True) # 流动性潮汐
            sensitivity_tension = dynamic_fusion_weights_params['sensitivity_tension']
            sensitivity_liquidity = dynamic_fusion_weights_params['sensitivity_liquidity']
            tension_impact_weights = dynamic_fusion_weights_params['tension_impact_weights']
            liquidity_impact_weights = dynamic_fusion_weights_params['liquidity_impact_weights']
            # 创建一个临时的 Series 来存储动态调整后的权重，确保所有操作都是 Series 级别的
            adjusted_weights_series = pd.DataFrame(final_fusion_weights_dict, index=df_index)
            for k in final_fusion_weights_dict:
                # 市场张力越高，base_divergence和divergence_quality权重增加，context_modulator权重减少
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_1 * tension_impact_weights.get(k, 0.0) * sensitivity_tension)
                # 流动性潮汐越高，volume_confirmation和main_force_confirmation权重增加
                adjusted_weights_series[k] = adjusted_weights_series[k] + (modulator_signal_2 * liquidity_impact_weights.get(k, 0.0) * sensitivity_liquidity)
            # 确保权重和为1，并进行归一化
            total_dynamic_weight = adjusted_weights_series.sum(axis=1)
            if (total_dynamic_weight > 0).all():
                final_fusion_weights_dict = (adjusted_weights_series.div(total_dynamic_weight, axis=0)).to_dict('series')
            else:
                final_fusion_weights_dict = get_param_value(params.get('dynamic_fusion_weights_params', {}).get('base_weights'), {
                    "base_divergence": 0.3, "volume_confirmation": 0.2, "main_force_confirmation": 0.25, "divergence_quality": 0.15, "context_modulator": 0.1
                })
        # Calculate the raw fused score (unipolar)
        raw_fused_score = _robust_geometric_mean(final_components, final_fusion_weights_dict, df_index)
        # Apply synergy/conflict logic
        synergy_conflict_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        sign_base = np.sign(base_divergence_score.replace(0, 1e-9))
        sign_vol = np.sign(volume_confirmation_score.replace(0, 1e-9))
        sign_mf = np.sign(main_force_confirmation_score.replace(0, 1e-9))
        sign_price_momentum_quality = np.sign(price_momentum_quality_score.replace(0, 1e-9))
        aligned_count = (sign_base == sign_vol).astype(int) + \
                        (sign_base == sign_mf).astype(int) + \
                        (sign_base == sign_price_momentum_quality).astype(int)
        is_synergistic = (aligned_count >= 3) & (base_divergence_score.abs() > synergy_threshold)
        is_conflicting = (aligned_count < 1) & (base_divergence_score.abs() > synergy_threshold)
        synergy_conflict_factor.loc[is_synergistic] = 1 + synergy_bonus_factor
        synergy_conflict_factor.loc[is_conflicting] = 1 - conflict_penalty_factor
        raw_fused_score_modulated = raw_fused_score * synergy_conflict_factor
        final_score = raw_fused_score_modulated * base_divergence_score.apply(np.sign)
        final_score = np.sign(final_score) * (final_score.abs().pow(final_fusion_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_winner_conviction_decay(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.1 · 全息动态审判版】“赢家信念衰减”专属计算引擎
        - 核心重构: 引入更多维度信号，深化对信念衰减、利润压力、派发确认和情境调制的感知。
        - 核心升级: 引入“买盘抵抗瓦解”证据，强化“诡道派发”识别，扩展情境调制器。
        - 核心优化: 引入“动态融合指数”，根据市场波动率和情绪动态调整最终融合的非线性指数。
        - 核心逻辑: 最终衰减分 = (核心衰减分 * (1 + 情境调制器))^动态非线性指数。
        """
        print(f"    -> [过程层] 正在计算 {config.get('name')} (V4.1 · 全息动态审判版)...")
        signal_name = config.get('name')
        belief_signal_name = 'winner_stability_index_D'
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        # 获取配置参数
        decay_params = get_param_value(self.params.get('winner_conviction_decay_params'), {})
        mtf_slope_accel_weights = get_param_value(decay_params.get('mtf_slope_accel_weights'), {"slope_periods": {"5": 0.4, "13": 0.3}, "accel_periods": {"5": 0.6}})
        belief_decay_components_weights = get_param_value(decay_params.get('belief_decay_components_weights'), {
            "winner_stability_mtf": 0.4, "winner_profit_margin_avg_inverted": 0.2,
            "total_winner_rate_inverted": 0.2, "chip_fatigue": 0.2
        })
        profit_pressure_components_weights = get_param_value(decay_params.get('profit_pressure_components_weights'), {
            "profit_taking_flow_mtf": 0.3, "active_selling_pressure": 0.2,
            "rally_sell_distribution_intensity": 0.2, "main_force_t0_sell_efficiency": 0.15,
            "main_force_on_peak_sell_flow": 0.15
        })
        distribution_confirmation_components_weights = get_param_value(decay_params.get('distribution_confirmation_components_weights'), {
            "distribution_intent": 0.3, "chip_distribution_whisper": 0.2,
            "upper_shadow_selling_pressure": 0.2, "deception_lure_long": 0.15,
            "wash_trade_intensity": 0.15
        })
        buying_resistance_collapse_weights = get_param_value(decay_params.get('buying_resistance_collapse_weights'), {
            "pressure_rejection_strength_inverted": 0.25, "rally_buy_support_weakness": 0.25,
            "buy_quote_exhaustion": 0.2, "bid_side_liquidity_inverted": 0.15,
            "main_force_slippage": 0.15
        })
        contextual_modulator_weights = get_param_value(decay_params.get('contextual_modulator_weights'), {
            "price_overextension_composite": 0.2, "retail_fomo": 0.15,
            "market_tension": 0.15, "sentiment_pendulum_negative": 0.1,
            "structural_tension": 0.1, "volatility_expansion": 0.1,
            "chip_health_inverted": 0.1, "market_impact_cost": 0.05,
            "buying_resistance_collapse": 0.05
        })
        dynamic_fusion_exponent_params = get_param_value(decay_params.get('dynamic_fusion_exponent_params'), {"enabled": False, "base_exponent": 1.5})
        price_overextension_composite_weights = get_param_value(decay_params.get('price_overextension_composite_weights'), {"bias_13": 0.3, "bias_21": 0.2, "rsi_13": 0.3, "bbp_21": 0.2})
        # 更新所有必需的DF列和原子信号
        required_df_columns = [
            belief_signal_name, pressure_signal_name,
            'upper_shadow_selling_pressure_D', 'retail_fomo_premium_index_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'BIAS_13_D', 'BIAS_21_D', 'RSI_13_D', 'BBP_21_2.0_D',
            'winner_profit_margin_avg_D', 'total_winner_rate_D', 'chip_fatigue_index_D',
            'active_selling_pressure_D', 'rally_sell_distribution_intensity_D',
            'main_force_t0_sell_efficiency_D', 'main_force_on_peak_sell_flow_D',
            'deception_lure_long_intensity_D', 'wash_trade_intensity_D',
            'pressure_rejection_strength_D', 'rally_buy_support_weakness_D',
            'buy_quote_exhaustion_rate_D', 'bid_side_liquidity_D', 'main_force_slippage_index_D',
            'structural_tension_index_D', 'volatility_expansion_ratio_D',
            'chip_health_score_D', 'market_impact_cost_D'
        ]
        # 动态添加MTF斜率和加速度信号
        for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
            required_df_columns.append(f'SLOPE_{period_str}_{belief_signal_name}')
            required_df_columns.append(f'SLOPE_{period_str}_{pressure_signal_name}')
            required_df_columns.append(f'SLOPE_{period_str}_winner_profit_margin_avg_D')
            required_df_columns.append(f'SLOPE_{period_str}_total_winner_rate_D')
            required_df_columns.append(f'SLOPE_{period_str}_chip_fatigue_index_D')
            required_df_columns.append(f'SLOPE_{period_str}_active_selling_pressure_D')
            required_df_columns.append(f'SLOPE_{period_str}_rally_sell_distribution_intensity_D')
            required_df_columns.append(f'SLOPE_{period_str}_main_force_t0_sell_efficiency_D')
            required_df_columns.append(f'SLOPE_{period_str}_main_force_on_peak_sell_flow_D')
            required_df_columns.append(f'SLOPE_{period_str}_deception_lure_long_intensity_D')
            required_df_columns.append(f'SLOPE_{period_str}_wash_trade_intensity_D')
            required_df_columns.append(f'SLOPE_{period_str}_pressure_rejection_strength_D')
            required_df_columns.append(f'SLOPE_{period_str}_rally_buy_support_weakness_D')
            required_df_columns.append(f'SLOPE_{period_str}_buy_quote_exhaustion_rate_D')
            required_df_columns.append(f'SLOPE_{period_str}_bid_side_liquidity_D')
            required_df_columns.append(f'SLOPE_{period_str}_main_force_slippage_index_D')
            required_df_columns.append(f'SLOPE_{period_str}_structural_tension_index_D')
            required_df_columns.append(f'SLOPE_{period_str}_volatility_expansion_ratio_D')
            required_df_columns.append(f'SLOPE_{period_str}_chip_health_score_D')
            required_df_columns.append(f'SLOPE_{period_str}_market_impact_cost_D')
        for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
            required_df_columns.append(f'ACCEL_{period_str}_{belief_signal_name}')
            required_df_columns.append(f'ACCEL_{period_str}_{pressure_signal_name}')
            required_df_columns.append(f'ACCEL_{period_str}_winner_profit_margin_avg_D')
            required_df_columns.append(f'ACCEL_{period_str}_total_winner_rate_D')
            required_df_columns.append(f'ACCEL_{period_str}_chip_fatigue_index_D')
            required_df_columns.append(f'ACCEL_{period_str}_active_selling_pressure_D')
            required_df_columns.append(f'ACCEL_{period_str}_rally_sell_distribution_intensity_D')
            required_df_columns.append(f'ACCEL_{period_str}_main_force_t0_sell_efficiency_D')
            required_df_columns.append(f'ACCEL_{period_str}_main_force_on_peak_sell_flow_D')
            required_df_columns.append(f'ACCEL_{period_str}_deception_lure_long_intensity_D')
            required_df_columns.append(f'ACCEL_{period_str}_wash_trade_intensity_D')
            required_df_columns.append(f'ACCEL_{period_str}_pressure_rejection_strength_D')
            required_df_columns.append(f'ACCEL_{period_str}_rally_buy_support_weakness_D')
            required_df_columns.append(f'ACCEL_{period_str}_buy_quote_exhaustion_rate_D')
            required_df_columns.append(f'ACCEL_{period_str}_bid_side_liquidity_D')
            required_df_columns.append(f'ACCEL_{period_str}_main_force_slippage_index_D')
            required_df_columns.append(f'ACCEL_{period_str}_structural_tension_index_D')
            required_df_columns.append(f'ACCEL_{period_str}_volatility_expansion_ratio_D')
            required_df_columns.append(f'ACCEL_{period_str}_chip_health_score_D')
            required_df_columns.append(f'ACCEL_{period_str}_market_impact_cost_D')
        required_atomic_signals = [
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER',
            'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'
        ]
        all_required_signals = required_df_columns + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, "_calculate_winner_conviction_decay"):
            print(f"    -> [过程情报警告] _calculate_winner_conviction_decay 缺少核心信号，返回默认值。")
            return pd.Series(dtype=np.float32)
        print(f"    -> [DEBUG] _calculate_winner_conviction_decay: 信号校验通过。")
        df_index = df.index
        # 原始输入信号
        belief_signal_raw = self._get_safe_series(df, belief_signal_name, 0.0, method_name="_calculate_winner_conviction_decay")
        pressure_signal_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name="_calculate_winner_conviction_decay")
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_calculate_winner_conviction_decay")
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name="_calculate_winner_conviction_decay")
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name="_calculate_winner_conviction_decay")
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_calculate_winner_conviction_decay")
        bias_13_raw = self._get_safe_series(df, 'BIAS_13_D', 0.0, method_name="_calculate_winner_conviction_decay")
        bias_21_raw = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name="_calculate_winner_conviction_decay")
        rsi_13_raw = self._get_safe_series(df, 'RSI_13_D', 0.0, method_name="_calculate_winner_conviction_decay")
        bbp_21_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.0, method_name="_calculate_winner_conviction_decay")
        distribution_intent_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        chip_distribution_whisper_score = self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0)
        market_tension_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        winner_profit_margin_avg_raw = self._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name="_calculate_winner_conviction_decay")
        total_winner_rate_raw = self._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name="_calculate_winner_conviction_decay")
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_calculate_winner_conviction_decay")
        active_selling_pressure_raw = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_calculate_winner_conviction_decay")
        rally_sell_distribution_intensity_raw = self._get_safe_series(df, 'rally_sell_distribution_intensity_D', 0.0, method_name="_calculate_winner_conviction_decay")
        main_force_t0_sell_efficiency_raw = self._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name="_calculate_winner_conviction_decay")
        main_force_on_peak_sell_flow_raw = self._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name="_calculate_winner_conviction_decay")
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name="_calculate_winner_conviction_decay")
        wash_trade_intensity_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_winner_conviction_decay")
        pressure_rejection_strength_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_calculate_winner_conviction_decay")
        rally_buy_support_weakness_raw = self._get_safe_series(df, 'rally_buy_support_weakness_D', 0.0, method_name="_calculate_winner_conviction_decay")
        buy_quote_exhaustion_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_calculate_winner_conviction_decay")
        bid_side_liquidity_raw = self._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name="_calculate_winner_conviction_decay")
        main_force_slippage_raw = self._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name="_calculate_winner_conviction_decay")
        structural_tension_raw = self._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name="_calculate_winner_conviction_decay")
        volatility_expansion_raw = self._get_safe_series(df, 'volatility_expansion_ratio_D', 0.0, method_name="_calculate_winner_conviction_decay")
        chip_health_raw = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name="_calculate_winner_conviction_decay")
        market_impact_cost_raw = self._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name="_calculate_winner_conviction_decay")
        # --- 1. 信念衰减分 (MTF Belief Decay Score) ---
        mtf_winner_stability_score = self._get_mtf_slope_accel_score(df, belief_signal_name, mtf_slope_accel_weights, df_index, "_calculate_winner_conviction_decay", ascending=False, bipolar=False)
        winner_profit_margin_avg_inverted = self._normalize_series(winner_profit_margin_avg_raw, df_index, ascending=False)
        total_winner_rate_inverted = self._normalize_series(total_winner_rate_raw, df_index, ascending=False)
        chip_fatigue_norm = self._normalize_series(chip_fatigue_raw, df_index, ascending=True)
        belief_decay_components = {
            "winner_stability_mtf": mtf_winner_stability_score,
            "winner_profit_margin_avg_inverted": winner_profit_margin_avg_inverted,
            "total_winner_rate_inverted": total_winner_rate_inverted,
            "chip_fatigue": chip_fatigue_norm
        }
        mtf_decay_score = _robust_geometric_mean(belief_decay_components, belief_decay_components_weights, df_index)
        # --- 2. 利润压力分 (MTF Profit Pressure Score) ---
        mtf_profit_taking_flow_score = self._get_mtf_slope_accel_score(df, pressure_signal_name, mtf_slope_accel_weights, df_index, "_calculate_winner_conviction_decay", ascending=True, bipolar=False)
        active_selling_pressure_norm = self._normalize_series(active_selling_pressure_raw, df_index, ascending=True)
        rally_sell_distribution_intensity_norm = self._normalize_series(rally_sell_distribution_intensity_raw, df_index, ascending=True)
        main_force_t0_sell_efficiency_norm = self._normalize_series(main_force_t0_sell_efficiency_raw, df_index, ascending=True)
        main_force_on_peak_sell_flow_norm = self._normalize_series(main_force_on_peak_sell_flow_raw, df_index, ascending=True)
        profit_pressure_components = {
            "profit_taking_flow_mtf": mtf_profit_taking_flow_score,
            "active_selling_pressure": active_selling_pressure_norm,
            "rally_sell_distribution_intensity": rally_sell_distribution_intensity_norm,
            "main_force_t0_sell_efficiency": main_force_t0_sell_efficiency_norm,
            "main_force_on_peak_sell_flow": main_force_on_peak_sell_flow_norm
        }
        mtf_pressure_score = _robust_geometric_mean(profit_pressure_components, profit_pressure_components_weights, df_index)
        # --- 3. 派发确认分 (Distribution Confirmation Score) ---
        upper_shadow_pressure_norm = self._normalize_series(upper_shadow_pressure_raw, df_index, bipolar=False)
        deception_lure_long_norm = self._normalize_series(deception_lure_long_raw, df_index, ascending=True)
        wash_trade_intensity_norm = self._normalize_series(wash_trade_intensity_raw, df_index, ascending=True)
        distribution_confirmation_components = {
            "distribution_intent": distribution_intent_score,
            "chip_distribution_whisper": chip_distribution_whisper_score,
            "upper_shadow_selling_pressure": upper_shadow_pressure_norm,
            "deception_lure_long": deception_lure_long_norm,
            "wash_trade_intensity": wash_trade_intensity_norm
        }
        distribution_confirmation_score = _robust_geometric_mean(distribution_confirmation_components, distribution_confirmation_components_weights, df_index)
        # --- 4. 买盘抵抗瓦解分 (Buying Resistance Collapse Score) ---
        pressure_rejection_strength_inverted = self._normalize_series(pressure_rejection_strength_raw, df_index, ascending=False)
        rally_buy_support_weakness_norm = self._normalize_series(rally_buy_support_weakness_raw, df_index, ascending=True)
        buy_quote_exhaustion_norm = self._normalize_series(buy_quote_exhaustion_raw, df_index, ascending=True)
        bid_side_liquidity_inverted = self._normalize_series(bid_side_liquidity_raw, df_index, ascending=False)
        main_force_slippage_norm = self._normalize_series(main_force_slippage_raw, df_index, ascending=True)
        buying_resistance_collapse_components = {
            "pressure_rejection_strength_inverted": pressure_rejection_strength_inverted,
            "rally_buy_support_weakness": rally_buy_support_weakness_norm,
            "buy_quote_exhaustion": buy_quote_exhaustion_norm,
            "bid_side_liquidity_inverted": bid_side_liquidity_inverted,
            "main_force_slippage": main_force_slippage_norm
        }
        buying_resistance_collapse_score = _robust_geometric_mean(buying_resistance_collapse_components, buying_resistance_collapse_weights, df_index)
        # --- 5. 情境调制器 (Contextual Modulator) ---
        price_overextension_composite_components = {
            "bias_13": self._normalize_series(bias_13_raw.clip(lower=0), df_index, bipolar=False),
            "bias_21": self._normalize_series(bias_21_raw.clip(lower=0), df_index, bipolar=False),
            "rsi_13": self._normalize_series((rsi_13_raw - 70).clip(lower=0), df_index, bipolar=False),
            "bbp_21": self._normalize_series((bbp_21_raw - 0.8).clip(lower=0), df_index, bipolar=False)
        }
        price_overextension_composite_score = _robust_geometric_mean(price_overextension_composite_components, price_overextension_composite_weights, df_index)
        retail_fomo_norm = self._normalize_series(retail_fomo_raw, df_index, bipolar=False)
        market_tension_norm = self._normalize_series(market_tension_score, df_index, bipolar=False)
        sentiment_pendulum_negative_norm = self._normalize_series(sentiment_pendulum_score, df_index, bipolar=True).clip(lower=0)
        structural_tension_norm = self._normalize_series(structural_tension_raw, df_index, ascending=True)
        volatility_expansion_norm = self._normalize_series(volatility_expansion_raw, df_index, ascending=True)
        chip_health_inverted = self._normalize_series(chip_health_raw, df_index, ascending=False)
        market_impact_cost_norm = self._normalize_series(market_impact_cost_raw, df_index, ascending=True)
        contextual_modulator_components = {
            "price_overextension_composite": price_overextension_composite_score,
            "retail_fomo": retail_fomo_norm,
            "market_tension": market_tension_norm,
            "sentiment_pendulum_negative": sentiment_pendulum_negative_norm,
            "structural_tension": structural_tension_norm,
            "volatility_expansion": volatility_expansion_norm,
            "chip_health_inverted": chip_health_inverted,
            "market_impact_cost": market_impact_cost_norm,
            "buying_resistance_collapse": buying_resistance_collapse_score
        }
        contextual_modulator = _robust_geometric_mean(contextual_modulator_components, contextual_modulator_weights, df_index)
        # --- 6. 核心衰减分 (Core Decay Score) ---
        core_decay_components = {
            "mtf_decay": mtf_decay_score,
            "mtf_pressure": mtf_pressure_score,
            "distribution_confirmation": distribution_confirmation_score
        }
        core_decay_score = _robust_geometric_mean(core_decay_components, {"mtf_decay": 1/3, "mtf_pressure": 1/3, "distribution_confirmation": 1/3}, df_index)
        # --- 7. 动态融合指数 (Dynamic Fusion Exponent) ---
        dynamic_final_fusion_exponent = pd.Series(get_param_value(dynamic_fusion_exponent_params.get('base_exponent'), 1.5), index=df_index, dtype=np.float32)
        if get_param_value(dynamic_fusion_exponent_params.get('enabled'), False):
            volatility_signal_raw = self._get_safe_series(df, dynamic_fusion_exponent_params['volatility_signal'], 0.0, method_name="_calculate_winner_conviction_decay")
            sentiment_signal_raw = self._get_safe_series(df, dynamic_fusion_exponent_params['sentiment_signal'], 0.0, method_name="_calculate_winner_conviction_decay")
            volatility_norm = self._normalize_series(volatility_signal_raw, df_index, ascending=True)
            sentiment_norm_bipolar = self._normalize_series(sentiment_signal_raw, df_index, bipolar=True)
            volatility_sensitivity = dynamic_fusion_exponent_params.get('volatility_sensitivity', 0.5)
            sentiment_sensitivity = dynamic_fusion_exponent_params.get('sentiment_sensitivity', 0.3)
            min_exponent = dynamic_fusion_exponent_params.get('min_exponent', 1.0)
            max_exponent = dynamic_fusion_exponent_params.get('max_exponent', 2.0)
            exponent_modulator = (volatility_norm * volatility_sensitivity + sentiment_norm_bipolar.abs() * sentiment_sensitivity) / (volatility_sensitivity + sentiment_sensitivity + 1e-9)
            dynamic_final_fusion_exponent = (dynamic_final_fusion_exponent + exponent_modulator * (max_exponent - min_exponent)).clip(min_exponent, max_exponent)
        # --- 8. 最终信念衰减分 (Final Winner Conviction Decay Score) ---
        final_score = (core_decay_score * (1 + contextual_modulator)).pow(dynamic_final_fusion_exponent)
        final_score = final_score.clip(0, 1).fillna(0.0)
        return final_score.astype(np.float32)

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
            # [修改] 增加对公理信号是否存在的防御性检查
            if axiom_name not in self.strategy.atomic_states:
                print(f"    -> [过程情报警告] 领域 '{domain_name}' 依赖的公理信号 '{axiom_name}' 不存在，跳过此公理。")
                continue
            axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
            domain_health_components.append(axiom_score * axiom_weight)
            total_weight += abs(axiom_weight) # [修改] 使用绝对值权重总和，以正确处理负权重
        if total_weight == 0:
            print(f"        -> [领域反转诊断] 警告: 领域 '{domain_name}' 的公理权重总和为0，无法计算健康度。")
            return {}
        # 计算该领域的双极性健康度
        bipolar_domain_health = (sum(domain_health_components) / total_weight).clip(-1, 1)
        # [修改] 将健康度呈送给新的“神谕审判”方法进行最终裁决
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

    def _calculate_stealth_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.2 · 全息融合与多维趋势感知版】计算“隐蔽吸筹”的专属关系分数。
        - 核心升级: 将“横盘”与“温和推升”场景的证据融合方式，从“几何平均”升级为“全息证据加权融合”，
                      允许核心证据的强势弥补次要证据的不足，以勘破“明修栈道，暗度陈仓”的诡道。
        - 【强化】将价格趋势、筹码集中度趋势、峰值稳固度趋势和成本优势趋势全部升级为多时间维度（MTF）融合信号，
                      增强信号的鲁棒性和趋势感知能力。
        - 【调整】优化 `suppressive_score` 和 `consolidative_score` 的构成，使其更精准地捕捉隐蔽吸筹的特征。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_STEALTH_ACCUMULATION (V5.2 · 全息融合与多维趋势感知版)...")
        method_name = "_calculate_stealth_accumulation"
        df_index = df.index
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        historical_potential = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        potential_gate = config.get('historical_potential_gate', 0.0)
        potential_amplifier = config.get('historical_potential_amplifier', 0.0)
        required_signals = [
            'winner_concentration_90pct_D', 'dominant_peak_solidity_D',
            'main_force_cost_advantage_D', 'upward_impulse_purity_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['close_D', 'winner_concentration_90pct_D', 'dominant_peak_solidity_D', 'main_force_cost_advantage_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        required_atomic_signals = [
            'SCORE_DYN_AXIOM_STABILITY', 'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            'PROCESS_META_POWER_TRANSFER', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY'
        ]
        all_required_signals = required_signals + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 原始数据获取 ---
        stability_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        volume_atrophy_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        power_transfer_score = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        split_order_accumulation_score = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0)
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        upward_purity = self._normalize_series(upward_purity_raw, df_index, bipolar=False)
        # 价格趋势：使用MTF融合信号
        mtf_price_trend_norm = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 筹码集中度趋势：使用MTF融合信号
        mtf_concentration_trend_norm = self._get_mtf_slope_accel_score(df, 'winner_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 峰值稳固度趋势：使用MTF融合信号
        mtf_peak_solidity_trend_norm = self._get_mtf_slope_accel_score(df, 'dominant_peak_solidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 成本优势趋势：使用MTF融合信号
        mtf_cost_advantage_norm = self._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # --- 1. 压制吸筹场景 (Suppressive Accumulation) ---
        # 价格趋势为负或接近零，且成交量萎缩，权力转移为正，筹码集中度上升，成本优势上升
        suppressive_mask = mtf_price_trend_norm <= 0.1 # 价格压制或横盘
        evidence1_suppressive = volume_atrophy_score.clip(lower=0) # 成交量萎缩
        evidence2_suppressive = power_transfer_score.clip(lower=0) # 权力转移为正
        evidence3_suppressive = mtf_concentration_trend_norm.clip(lower=0) # 筹码集中度上升
        evidence4_suppressive = mtf_cost_advantage_norm.clip(lower=0) # 成本优势上升
        suppressive_score = (
            evidence1_suppressive * 0.25 +
            evidence2_suppressive * 0.25 +
            evidence3_suppressive * 0.25 +
            evidence4_suppressive * 0.25
        ).clip(0, 1)
        suppressive_score = suppressive_score.where(suppressive_mask, 0.0)
        # --- 2. 盘整吸筹场景 (Consolidative Accumulation) ---
        # 价格稳定，成交量萎缩，权力转移为正，峰值稳固度上升，拆单吸筹强度高
        consolidative_mask = stability_score > 0.2 # 价格稳定
        evidence1_consolidative = volume_atrophy_score.clip(lower=0) # 成交量萎缩
        evidence2_consolidative = power_transfer_score.clip(lower=0) # 权力转移为正
        evidence3_consolidative = mtf_peak_solidity_trend_norm.clip(lower=0) # 峰值稳固度上升
        evidence4_consolidative = split_order_accumulation_score.clip(lower=0) # 拆单吸筹强度高
        consolidative_score = (
            evidence1_consolidative * 0.25 +
            evidence2_consolidative * 0.25 +
            evidence3_consolidative * 0.25 +
            evidence4_consolidative * 0.25
        ).clip(0, 1)
        consolidative_score = consolidative_score.where(consolidative_mask, 0.0)
        # --- 3. 温和推升吸筹场景 (Gentle Push Accumulation) ---
        # 价格温和上涨，上涨纯度高，权力转移为正，拆单吸筹强度高
        gentle_push_mask = (mtf_price_trend_norm > 0.1) & (mtf_price_trend_norm < 0.5) # 价格温和上涨
        evidence1_gentle = upward_purity.clip(lower=0) # 上涨纯度高
        evidence2_gentle = power_transfer_score.clip(lower=0) # 权力转移为正
        evidence3_gentle = split_order_accumulation_score.clip(lower=0) # 拆单吸筹强度高
        gentle_push_score = (
            evidence1_gentle * 0.33 +
            evidence2_gentle * 0.33 +
            evidence3_gentle * 0.34
        ).clip(0, 1)
        gentle_push_score = gentle_push_score.where(gentle_push_mask, 0.0)
        # --- 4. 基础分：取三种场景中的最大值 ---
        base_score = pd.concat([suppressive_score, consolidative_score, gentle_push_score], axis=1).max(axis=1).fillna(0.0)
        # --- 5. 历史势能门控与调节 ---
        potential_gate_mask = historical_potential > potential_gate
        potential_modulator = (1 + historical_potential * potential_amplifier)
        final_score = (base_score * potential_modulator).where(potential_gate_mask, 0.0)
        self.strategy.atomic_states["_DEBUG_accum_suppressive_score"] = suppressive_score
        self.strategy.atomic_states["_DEBUG_accum_consolidative_score"] = consolidative_score
        self.strategy.atomic_states["_DEBUG_accum_gentle_push_score"] = gentle_push_score
        return final_score.clip(0, 1).astype(np.float32)

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
        print("    -> [过程层] 正在计算 PROCESS_META_PANIC_WASHOUT_ACCUMULATION (V4.7 · 战果审判与全息资金流验证强化版)...")
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
        print("    -> [过程层] 正在计算 PROCESS_META_DECEPTIVE_ACCUMULATION (V3.8 · 价格筹码背离与多维诡道融合版)...")
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
        print("    -> [过程层] 正在计算 PROCESS_META_UPTHRUST_WASHOUT (V2.1 · 强证优先与形态门控版)...")
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
        print("    -> [过程层] 正在计算 PROCESS_META_ACCUMULATION_INFLECTION (V2.4 · 势能衰减与多维融合版)...")
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

    def _calculate_winner_conviction_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0 · 韧性博弈与多维时空版】“赢家信念”专属关系计算引擎
        - 核心重构: 彻底废弃旧的“状态对抗”逻辑。引入“信念强度 × 压力韧性 × 诡道过滤 × 情境调制”的全新四维诊断框架。
        - 核心升级:
            1.  **多时间维度斜率与加速度融合：** 对“赢家稳定性”和“利润兑现压力”进行多时间维度（5, 13, 21, 34, 55日）斜率和加速度的融合，评估其趋势和动能。
            2.  **共振与背离判断：** 评估“赢家稳定性”和“利润兑现压力”在多时间维度上的共振（同向增强/减弱）或背离（一强一弱）。
            3.  **历史相对位置：** 引入信号相对于其历史区间的百分位，判断其是处于高位还是低位。
            4.  **诡道博弈特性：** 引入欺骗指数、对倒强度等信号，对虚假的信念增强或减弱进行惩罚。
            5.  **情境调制：** 引入市场情绪、波动率等情境因子进行动态调整。
            6.  **非线性融合：** 使用 _robust_geometric_mean 对所有强度/幅度组件进行融合，并结合整体方向。
        - 目标: 提供一个双极性分数，正值代表赢家信念坚定，负值代表信念动摇或面临风险。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_WINNER_CONVICTION (V4.0 · 韧性博弈与多维时空版)...")
        method_name = "_calculate_winner_conviction_relationship"
        df_index = df.index
        # --- 调试信息构建 ---
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False)
        probe_dates_list = self.probe_dates 
        probe_ts_for_debug = None
        if is_debug_enabled and probe_dates_list:
            probe_timestamps = pd.to_datetime(probe_dates_list).tz_localize(df.index.tz if df.index.tz else None)
            valid_probe_dates_in_df = [d for d in probe_timestamps if d in df.index]
            if valid_probe_dates_in_df:
                probe_ts_for_debug = valid_probe_dates_in_df[-1] # 使用最新的有效探针日期
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        # 获取配置参数
        params = get_param_value(config.get('winner_conviction_params'), {})
        relative_position_weights = get_param_value(params.get('relative_position_weights'), {"winner_stability_high": 0.6, "profit_taking_flow_low": 0.4})
        # 更新 context_modulator_weights 的默认值键名
        context_modulator_weights = get_param_value(params.get('context_modulator_weights'), {"market_sentiment": 0.4, "volatility_stability": 0.3, "trend_vitality": 0.3})
        final_exponent = get_param_value(params.get('final_exponent'), 1.5)
        final_fusion_gm_weights = get_param_value(params.get('final_fusion_gm_weights'), {
            "conviction_magnitude": 0.3,
            "pressure_magnitude": 0.2,
            "synergy_factor": 0.2,
            "deception_filter": 0.15,
            "context_modulator": 0.15
        })
        # 核心信号
        belief_signal_name = 'winner_stability_index_D'
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        # 额外依赖信号
        required_signals = [
            belief_signal_name, pressure_signal_name,
            'deception_index_D', 'wash_trade_intensity_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'trend_vitality_index_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in [belief_signal_name, pressure_signal_name, 'deception_index_D', 'wash_trade_intensity_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 原始数据获取 ---
        winner_stability_raw = self._get_safe_series(df, belief_signal_name, 0.0, method_name=method_name)
        profit_taking_flow_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name=method_name)
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        wash_trade_intensity_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        trend_vitality_raw = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        # --- 调试探针：原始数据 ---
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - winner_stability_raw", winner_stability_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - profit_taking_flow_raw", profit_taking_flow_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - deception_index_raw", deception_index_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - wash_trade_intensity_raw", wash_trade_intensity_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - market_sentiment_raw", market_sentiment_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - volatility_instability_raw", volatility_instability_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "原始数据 - trend_vitality_raw", trend_vitality_raw)
        # --- 1. 信念强度 (Conviction Strength) ---
        # 使用MTF融合赢家稳定性及其斜率和加速度 (双极性)
        mtf_winner_stability = self._get_mtf_slope_accel_score(df, belief_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 赢家稳定性相对于历史区间的百分位 (越高越好，映射到 [0, 1])
        winner_stability_percentile = winner_stability_raw.rank(pct=True).fillna(0.5)
        # 综合信念强度：MTF趋势 + 历史相对位置
        conviction_strength_score = (mtf_winner_stability * relative_position_weights.get("winner_stability_high", 0.6) + 
                                     (winner_stability_percentile * 2 - 1) * (1 - relative_position_weights.get("winner_stability_high", 0.6))).clip(-1, 1)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - conviction_strength_score", conviction_strength_score)
        # --- 2. 压力韧性 (Pressure Resilience) ---
        # 使用MTF融合利润兑现流量及其斜率和加速度 (双极性，负值代表压力大，韧性差)
        # 利润兑现流量是负向指标，所以其MTF分数越低（负值越大）代表压力越大，韧性越差
        mtf_profit_taking_flow = self._get_mtf_slope_accel_score(df, pressure_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 利润兑现流量相对于历史区间的百分位 (越低越好，映射到 [0, 1])
        profit_taking_flow_percentile = (1 - profit_taking_flow_raw.rank(pct=True)).fillna(0.5)
        # 综合压力韧性：(1 - MTF利润兑现流量) + 历史相对位置
        # 将mtf_profit_taking_flow反向，使其正值代表韧性强
        pressure_resilience_score = ((mtf_profit_taking_flow * -1) * relative_position_weights.get("profit_taking_flow_low", 0.4) + 
                                     (profit_taking_flow_percentile * 2 - 1) * (1 - relative_position_weights.get("profit_taking_flow_low", 0.4))).clip(-1, 1)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - pressure_resilience_score", pressure_resilience_score)
        # --- 3. 共振与背离因子 (Synergy Factor) ---
        # 评估赢家稳定性与利润兑现压力之间的共振或背离
        # 当两者同向（信念增强且压力减弱，或信念减弱且压力增强）时，共振因子高
        # 当两者背离（信念增强但压力也增强，或信念减弱但压力也减弱）时，共振因子低
        # 将信念强度和压力韧性映射到 [0, 1]
        norm_conviction = (conviction_strength_score + 1) / 2
        norm_resilience = (pressure_resilience_score + 1) / 2
        # 协同因子：当两者都高或都低时，协同性高
        synergy_factor = (norm_conviction * norm_resilience + (1 - norm_conviction) * (1 - norm_resilience)).clip(0, 1)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - synergy_factor", synergy_factor)
        # --- 4. 诡道过滤 (Deception Filter) ---
        # 欺骗指数和对倒强度越高，对信念的真实性惩罚越大
        mtf_deception_index = self._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_wash_trade_intensity = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        deception_penalty = (mtf_deception_index * 0.6 + mtf_wash_trade_intensity * 0.4).clip(0, 1)
        deception_filter = (1 - deception_penalty).clip(0, 1) # 惩罚因子，0表示完全过滤，1表示无影响
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - deception_penalty", deception_penalty)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - deception_filter", deception_filter)
        # --- 5. 情境调制 (Contextual Modulation) ---
        # 市场情绪、波动率、趋势活力等对信念的影响
        norm_market_sentiment = self._normalize_series(market_sentiment_raw, df_index, bipolar=True)
        # 将 volatility_instability_raw 视为负向指标，即值越小越好，因此对其进行反向处理后进行正向归一化。
        # 这样，低不稳定性（高稳定性）将得到高分。
        # 明确提供 windows 参数，使用 21 作为窗口，因为 VOLATILITY_INSTABILITY_INDEX_21d_D 是一个21天的指标。
        # 同时传递 debug_info。
        volatility_stability_raw = 1 - normalize_score(
            volatility_instability_raw, 
            df_index, 
            21, # 明确指定 windows 参数
            ascending=True,
            debug_info=(is_debug_enabled, probe_ts_for_debug, method_name + "_volatility_stability_raw_norm")
        ) # 将不稳定性转换为稳定性，并归一化到 [0, 1]
        # 探针：volatility_stability_raw 的值
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - volatility_stability_raw (after 1-norm)", volatility_stability_raw)
        norm_volatility_stability = self._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self._normalize_series(trend_vitality_raw, df_index, bipolar=False) # 趋势活力越高越好
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability, # 使用修正后的稳定性
            "trend_vitality": norm_trend_vitality
        }
        # 调试探针：context_modulator_components 的输入
        for k, v in context_modulator_components.items():
            self._debug_probe(df_index, probe_ts_for_debug, method_name, f"GM输入 - context_modulator_components[{k}]", v)
        # 使用几何平均融合情境调制器，确保只有当多个情境同时有利时才高
        context_modulator_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()}, # 确保输入为正
            context_modulator_weights,
            df_index
        )
        # 将情境调制器映射到 [0.5, 1.5] 范围，以实现放大或抑制
        context_modulator = 0.5 + context_modulator_score # 0.5 + [0,1] -> [0.5, 1.5]
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "中间分 - context_modulator", context_modulator)
        # --- 6. 最终融合 ---
        # 1. 确定整体方向：由信念强度和压力韧性的加权和决定
        # 权重可以从配置中获取，这里使用默认值
        direction_weight_conviction = get_param_value(params.get('direction_weights', {}).get('conviction', 0.6), 0.6)
        direction_weight_pressure = get_param_value(params.get('direction_weights', {}).get('pressure', 0.4), 0.4)
        overall_direction_raw = (conviction_strength_score * direction_weight_conviction + pressure_resilience_score * direction_weight_pressure)
        overall_direction = np.sign(overall_direction_raw)
        overall_direction = overall_direction.replace(0, 1) # 如果和为0，则视为正向，让幅度决定
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "最终融合 - overall_direction_raw", overall_direction_raw)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "最终融合 - overall_direction", overall_direction)
        # 2. 准备所有组件的“强度/幅度”版本，映射到 [0, 1] 或 [0.5, 1.5]
        # conviction_strength_score 和 pressure_resilience_score 是双极性 [-1, 1]
        # 它们的绝对值代表强度，映射到 [0, 1]
        conviction_magnitude = (conviction_strength_score.abs() + 1) / 2
        pressure_magnitude = (pressure_resilience_score.abs() + 1) / 2
        # synergy_factor 是 [0, 1]
        # deception_filter 是 [0, 1]
        # context_modulator 是 [0.5, 1.5]
        fusion_components_for_gm = {
            "conviction_magnitude": conviction_magnitude,
            "pressure_magnitude": pressure_magnitude,
            "synergy_factor": synergy_factor,
            "deception_filter": deception_filter,
            "context_modulator": context_modulator
        }
        # 调试探针：fusion_components_for_gm 的输入
        for k, v in fusion_components_for_gm.items():
            self._debug_probe(df_index, probe_ts_for_debug, method_name, f"GM输入 - fusion_components_for_gm[{k}]", v)
        # 3. 使用 _robust_geometric_mean 融合所有强度/幅度组件
        fused_magnitude = _robust_geometric_mean(
            fusion_components_for_gm,
            final_fusion_gm_weights,
            df_index
        )
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "最终融合 - fused_magnitude", fused_magnitude)
        # 4. 结合整体方向和融合后的幅度
        final_score = fused_magnitude * overall_direction
        # 5. 应用非线性指数
        final_score = np.sign(final_score) * (final_score.abs().pow(final_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        self._debug_probe(df_index, probe_ts_for_debug, method_name, "最终分 - final_score", final_score)
        return final_score.astype(np.float32)

    def _debug_probe(self, index: pd.Index, probe_ts: Optional[pd.Timestamp], method_name: str, signal_name: str, series: pd.Series):
        """
        调试探针辅助方法，用于在指定日期输出信号的详细值。
        """
        # # 直接访问 self.debug_params
        # if not get_param_value(self.debug_params.get('enabled'), False) or probe_ts is None:
        #     return
        # if probe_ts in index:
        #     value = series.loc[probe_ts]
        #     print(f"    -> [探针] 方法 '{method_name}' @ {probe_ts.strftime('%Y-%m-%d')} - '{signal_name}': {value:.4f}")

    def _calculate_loser_capitulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 战场扩展版】计算“套牢盘投降”信号。
        - 核心重构: 扩展“战场”定义。战场不再仅限于收盘下跌日，而是扩展为“收盘下跌 或 出现强力下影线吸收”，
                      旨在捕捉经典的“金针探底”反转形态。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_LOSER_CAPITULATION (V3.1 · 战场扩展版)...")
        required_signals = [
            'pct_change_D', 'capitulation_flow_ratio_D', 'active_buying_support_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_loser_capitulation"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_loser_capitulation")
        capitulation_flow_raw = self._get_safe_series(df, 'capitulation_flow_ratio_D', 0.0, method_name="_calculate_loser_capitulation")
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_loser_capitulation")
        lower_shadow_absorption = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        # 战场上下文：扩展为“下跌日”或“有强力下影线吸收”
        context_mask = (pct_change < 0) | (lower_shadow_absorption > 0.5)
        # 恐慌分：衡量抛售的烈度
        panic_score = self._normalize_series(capitulation_flow_raw, df_index, bipolar=False)
        # 吸收分：采用“强证优先”原则，取最强的承接证据
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        absorption_score = pd.concat([active_buying_norm, lower_shadow_absorption], axis=1).max(axis=1)
        # 最终审判：恐慌与吸收的乘积
        final_score = (panic_score * absorption_score).where(context_mask, 0.0).fillna(0.0)
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
        print("    -> [过程层] 正在计算 PROCESS_META_COST_ADVANTAGE_TREND (V4.4 · 象限审判与全息资金流验证版)...")
        method_name = "_calculate_cost_advantage_trend_relationship"
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
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
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
        # --- 归一化处理 ---
        # P_change = self._normalize_series(price_change, df_index, bipolar=True) # 替换为MTF
        # CA_change = self._normalize_series(main_force_cost_advantage.diff(1).fillna(0), df_index, bipolar=True) # 替换为MTF
        mtf_main_force_conviction = self._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_upward_purity = self._get_mtf_slope_accel_score(df, 'upward_impulse_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        suppressive_accum_norm = self._normalize_series(suppressive_accum, df_index, bipolar=False)
        mtf_distribution_intensity = self._get_mtf_slope_accel_score(df, 'distribution_at_peak_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_active_selling = self._get_mtf_slope_accel_score(df, 'active_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        profit_taking_flow_norm = self._normalize_series(profit_taking_flow, df_index, bipolar=False)
        active_buying_support_norm = self._normalize_series(active_buying_support, df_index, bipolar=False)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        # --- 1. Q1: 价涨 & 优扩 (健康上涨) ---
        Q1_base = (mtf_price_change.clip(lower=0) * mtf_ca_change.clip(lower=0)).pow(0.5)
        # 确认：主力信念、上涨纯度、资金流可信度
        Q1_confirm = (mtf_main_force_conviction.clip(lower=0) * mtf_upward_purity * flow_credibility_norm).pow(1/3)
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        # --- 2. Q2: 价跌 & 优缩 (派发下跌) ---
        Q2_base = (mtf_price_change.clip(upper=0).abs() * mtf_ca_change.clip(upper=0).abs()).pow(0.5)
        # 确认：利润兑现流量、主动卖压、行为派发意图 (使用MTF信号)
        Q2_distribution_evidence = (profit_taking_flow_norm * 0.4 + mtf_active_selling * 0.3 + mtf_distribution_intensity * 0.3).clip(0, 1)
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        # --- 3. Q3: 价跌 & 优扩 (黄金坑) ---
        Q3_base = (mtf_price_change.clip(upper=0).abs() * mtf_ca_change.clip(lower=0)).pow(0.5)
        # 确认：隐蔽吸筹、下影线吸收、资金流可信度
        Q3_confirm = (suppressive_accum_norm * lower_shadow_absorb * flow_credibility_norm).pow(1/3)
        # 前置下跌上下文，如果前几日有深跌，则增加黄金坑的权重
        pre_5day_pct_change = close_price.pct_change(periods=5).shift(1).fillna(0)
        norm_pre_drop_5d = self._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        pre_drop_context_bonus = norm_pre_drop_5d * 0.5
        Q3_final = (Q3_base * Q3_confirm * (1 + pre_drop_context_bonus)).clip(0, 1)
        # --- 4. Q4: 价涨 & 优缩 (牛市陷阱) ---
        Q4_base = (mtf_price_change.clip(lower=0) * mtf_ca_change.clip(upper=0).abs()).pow(0.5)
        # 确认：派发强度、买盘虚弱度、主力资金净流出 (使用MTF信号)
        mf_outflow_risk = main_force_net_flow_norm.clip(upper=0).abs() # 主力资金净流出
        Q4_trap_evidence = (mtf_distribution_intensity * 0.4 + (1 - active_buying_support_norm) * 0.3 + mf_outflow_risk * 0.3).clip(0, 1)
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        # --- 最终融合 ---
        final_score = (Q1_final + Q2_final + Q3_final + Q4_final).clip(-1, 1)
        return final_score.astype(np.float32)

    def _calculate_split_order_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.2 · 质效校准版】计算“拆单吸筹强度”的专属信号。
        - 核心升级: 引入“效率基准线”(efficiency_baseline)概念。在计算“质效调节指数”前，
                      先对“全息验证综合分”进行校准。这使得任何低于基准线的战果（即使为正）
                      都会被视为负向贡献，从而受到惩罚性抑制，为模型注入了赏罚分明的“主帅”逻辑。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY (V3.2 · 质效校准版)...")
        required_signals = [
            'hidden_accumulation_intensity_D', 'SLOPE_5_close_D', 'deception_index_D',
            'upward_impulse_purity_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_DYN_AXIOM_STABILITY'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_split_order_accumulation"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        efficiency_baseline = config.get('efficiency_baseline', 0.15)
        raw_intensity = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_split_order_accumulation")
        price_trend_raw = self._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name="_calculate_split_order_accumulation")
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_calculate_split_order_accumulation")
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_split_order_accumulation"), df_index, bipolar=False)
        flow_outcome = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        structure_outcome = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        potential_outcome = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        normalized_score = (raw_intensity / 100).clip(0, 1)
        price_trend_norm = self._normalize_series(price_trend_raw, df_index, bipolar=True)
        price_suppression_factor = (1 - price_trend_norm.clip(lower=0) * (1 - upward_purity)).clip(0, 1)
        deception_norm = self._normalize_series(deception_index, df_index, bipolar=True)
        strategic_context_factor = (potential_outcome * 0.5 + deception_norm.clip(lower=0) * 0.5).clip(0, 1)
        preliminary_score = (normalized_score * price_suppression_factor * strategic_context_factor).pow(1/3).fillna(0.0)
        tactical_momentum_score = self._normalize_series(preliminary_score.diff(1).fillna(0), df_index, bipolar=False)
        dynamic_preliminary_score = (preliminary_score * 0.7 + tactical_momentum_score * 0.3).clip(0, 1)
        stability_score = potential_outcome
        weight_flow = 1 - stability_score
        weight_structure = stability_score
        total_weight = weight_flow + weight_structure + 0.2
        w_f = weight_flow / total_weight
        w_s = weight_structure / total_weight
        w_p = 0.2 / total_weight
        holographic_state_score = (flow_outcome * w_f + structure_outcome * w_s + potential_outcome * w_p)
        flow_trend = self._normalize_series(flow_outcome.diff(3).fillna(0), df_index, bipolar=True)
        structure_trend = self._normalize_series(structure_outcome.diff(3).fillna(0), df_index, bipolar=True)
        potential_trend = self._normalize_series(potential_outcome.diff(3).fillna(0), df_index, bipolar=True)
        holographic_trend_score = (flow_trend * w_f + structure_trend * w_s + potential_trend * w_p)
        holographic_validation_score = (holographic_state_score * 0.6 + holographic_trend_score * 0.4).clip(-1, 1)
        calibrated_holographic_score = holographic_validation_score - efficiency_baseline
        quality_efficiency_modulator = (1 - calibrated_holographic_score).clip(0.1, 2.0)
        final_score = dynamic_preliminary_score.pow(quality_efficiency_modulator).clip(0, 1)
        return final_score.astype(np.float32)

    def _calculate_price_volume_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【生产版】计算价量关系的专属分数。
        - 核心逻辑: 基于价量四象限博弈模型，并引入“共振催化剂”（由形态、流向、心理、主动承接构成）
                      作为关键场景的乘法确认项，核心权重倾向于体现“主力意志”的主动承接行为。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_PV_REL_BULLISH_TURN...")
        required_signals = [
            'close_D', 'volume_D', 'main_force_conviction_index_D', 'wash_trade_intensity_D',
            'suppressive_accumulation_intensity_D', 'retail_panic_surrender_index_D',
            'upward_impulse_purity_D', 'PROCESS_META_POWER_TRANSFER', 'active_buying_support_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_price_volume_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        price = self._get_safe_series(df, 'close_D', method_name="_calculate_price_volume_relationship")
        volume = self._get_safe_series(df, 'volume_D', method_name="_calculate_price_volume_relationship")
        main_force_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_price_volume_relationship")
        wash_trade_penalty = self._normalize_series(self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index)
        volume_atrophy_quality = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        suppressive_accum = self._normalize_series(self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index)
        panic_evidence = self._normalize_series(self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index)
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index)
        reversal_confirmation_shape = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        reversal_confirmation_flow = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        reversal_confirmation_psyche = self._normalize_series(main_force_conviction.diff(1).fillna(0), df_index, bipolar=True)
        active_buying_confirm = self._normalize_series(self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index)
        accel_shape = self._normalize_series(reversal_confirmation_shape.diff(2).fillna(0), df_index)
        accel_flow = self._normalize_series(reversal_confirmation_flow.diff(2).fillna(0), df_index)
        accel_psyche = self._normalize_series(reversal_confirmation_psyche.diff(2).fillna(0), df_index)
        acceleration_bonus = (accel_shape * 0.3 + accel_flow * 0.4 + accel_psyche * 0.3).clip(0, 1)
        base_resonance_score = (
            reversal_confirmation_shape * 0.2 +
            reversal_confirmation_flow.clip(lower=0) * 0.25 +
            reversal_confirmation_psyche.clip(lower=0) * 0.15 +
            active_buying_confirm * 0.4
        ).clip(0, 1)
        weight_accel = (1 - base_resonance_score)
        weight_base = 1 - weight_accel
        fused_resonance_score = (base_resonance_score * weight_base + acceleration_bonus * weight_accel)
        resonance_components = pd.concat([reversal_confirmation_shape, reversal_confirmation_flow, reversal_confirmation_psyche, active_buying_confirm], axis=1)
        harmony_degree = (1 - (resonance_components.max(axis=1) - resonance_components.min(axis=1))).clip(0, 1)
        resonance_confirmation_factor = (fused_resonance_score * (1 + harmony_degree)).clip(0, 1)
        p_mom = self._normalize_series(price.pct_change().fillna(0), df_index, bipolar=True)
        v_mom = self._normalize_series(volume.pct_change().fillna(0), df_index, bipolar=True)
        final_score = pd.Series(0.0, index=df_index)
        quality_factor = (self._normalize_series(main_force_conviction, df_index, bipolar=True).clip(lower=0) * (1 - wash_trade_penalty)).pow(0.5)
        score1 = (p_mom * v_mom).pow(0.5) * quality_factor
        intent_factor = (volume_atrophy_quality * chip_posture.clip(lower=0) * upward_purity).pow(1/3)
        score2 = (p_mom - v_mom) / 2 * intent_factor
        base_score3 = -((p_mom.abs() * v_mom).pow(0.5))
        score3 = base_score3 * (1 - suppressive_accum)
        recent_panic_context = panic_evidence.rolling(window=3, min_periods=1).max()
        exhaustion_degree = (1 + v_mom.clip(upper=0)).clip(0, 1)
        narrative_factor_4 = (recent_panic_context * exhaustion_degree).clip(0, 1)
        base_score4 = (v_mom.abs() - p_mom.abs()) / 2
        score4 = base_score4 * narrative_factor_4 * resonance_confirmation_factor
        mask1 = (p_mom > 0) & (v_mom > 0)
        mask2 = (p_mom > 0) & (v_mom <= 0)
        mask3 = (p_mom <= 0) & (v_mom > 0)
        mask4 = (p_mom <= 0) & (v_mom <= 0)
        if mask1.any(): final_score.loc[mask1] = score1.loc[mask1]
        if mask2.any(): final_score.loc[mask2] = score2.loc[mask2]
        if mask3.any(): final_score.loc[mask3] = score3.loc[mask3]
        if mask4.any(): final_score.loc[mask4] = score4.loc[mask4]
        final_score = final_score.clip(-1, 1)
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
        print("    -> [过程层] 正在计算 PROCESS_META_BREAKOUT_ACCELERATION (V3.3 · 共振审判与突破门控版)...")
        method_name = "_calculate_breakout_acceleration"
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
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        relative_strength = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        # --- 原始数据获取 ---
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility, df_index, bipolar=False)
        # --- 定义四大核心证据 ---
        breakout_evidence = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0)
        intent_evidence = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0)
        flow_evidence = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0)
        structure_evidence = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(lower=0)
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
        # --- 强化“突破”的门控作用 ---
        # 如果突破证据很弱，则大幅惩罚共振分
        breakout_gate_factor = pd.Series(1.0, index=df_index, dtype=np.float32)
        breakout_gate_factor = breakout_gate_factor.mask(breakout_evidence < 0.2, breakout_evidence * 0.5) # 突破证据低于0.2时，惩罚50%
        breakout_gate_factor = breakout_gate_factor.mask(breakout_evidence < 0.05, 0.0) # 突破证据极弱时，直接归零
        resonance_score_gated = resonance_score * breakout_gate_factor
        # --- 相对强度调节器 ---
        rs_modulator_base = (1 + relative_strength * rs_amplifier)
        mf_flow_modulator = (1 + mf_flow_validation * 0.5)
        rs_modulator = rs_modulator_base * mf_flow_modulator
        final_score = (resonance_score_gated * rs_modulator).clip(0, 1).fillna(0.0)
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
        print("    -> [过程层] 正在计算 PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT (V2.7 · 战术升级与全息资金流验证强化版)...")
        method_name = "_calculate_fund_flow_accumulation_inflection"
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
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        # 获取原料
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        # --- 归一化处理 ---
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        # --- 1. 重铸“前奏分”，消除尺度问题并引入主力资金净流入趋势 ---
        mtf_hidden_accumulation = self._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_net_flow_slope = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        prelude_score_base = self._normalize_series(mtf_hidden_accumulation.rolling(5).mean(), df_index, bipolar=False)
        # 确保主力资金净流入趋势为正向才贡献
        prelude_score = (prelude_score_base * mtf_mf_net_flow_slope.clip(lower=0)).pow(0.5)
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
        # --- 3. 最终审判 ---
        final_score = (prelude_score * attack_score).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.3 · 战场态势与多维压力版】“利润与流向”专属关系计算引擎
        - 核心重构: 创立“战场态势”审判模型，从比较“动量”升维为比较力量的“当前水平”。
        - 信号升级: 将核心“压力”信号从“T0效率”升级为更精准的“利润兑现流量占比”。
        - 【新增】增强“派发压力”的判断，引入赢家平均利润率和行为派发意图。
        - 【新增】调整“建仓动力”的判断，引入资金流可信度。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_PROFIT_VS_FLOW (V4.3 · 战场态势与多维压力版)...")
        method_name = "_calculate_profit_vs_flow_relationship"
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        drive_signal_name = 'main_force_net_flow_calibrated_D'
        winner_profit_margin_name = 'winner_profit_margin_avg_D' # 新增
        distribution_intent_name = 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT' # 新增
        flow_credibility_name = 'flow_credibility_index_D' # 新增
        required_signals = [pressure_signal_name, drive_signal_name, winner_profit_margin_name, flow_credibility_name]
        required_atomic_signals = [distribution_intent_name]
        all_required_signals = required_signals + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(dtype=np.float32)
        df_index = df.index
        # 获取原始数据
        profit_taking_flow_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name=method_name)
        main_force_net_flow_raw = self._get_safe_series(df, drive_signal_name, 0.0, method_name=method_name)
        winner_profit_margin_raw = self._get_safe_series(df, winner_profit_margin_name, 0.0, method_name=method_name)
        distribution_intent_score = self._get_atomic_score(df, distribution_intent_name, 0.0)
        flow_credibility_raw = self._get_safe_series(df, flow_credibility_name, 0.0, method_name=method_name)
        # --- 归一化处理 ---
        profit_taking_flow_norm = self._normalize_series(profit_taking_flow_raw, df_index, bipolar=False)
        winner_profit_margin_norm = self._normalize_series(winner_profit_margin_raw, df_index, bipolar=False)
        distribution_intent_norm = self._normalize_series(distribution_intent_score, df_index, bipolar=False)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow_raw, df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        # --- 1. 派发压力分 (Pressure Score) ---
        # 结合利润兑现流量、赢家平均利润率和行为派发意图
        pressure_score = (
            profit_taking_flow_norm * 0.4 +
            winner_profit_margin_norm * 0.3 +
            distribution_intent_norm * 0.3
        ).clip(0, 1)
        # --- 2. 建仓动力分 (Drive Score) ---
        # 结合主力资金净流向和资金流可信度
        drive_score = (main_force_net_flow_norm.clip(lower=0) * flow_credibility_norm).clip(0, 1)
        # --- 3. 核心逻辑：战场态势对抗 ---
        relationship_score = drive_score - pressure_score
        final_score = relationship_score.clip(-1, 1)
        # --- 探针输出 ---
        probe_dates = self.probe_dates
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
        print("    -> [过程层] 正在计算 PROCESS_META_STOCK_SECTOR_SYNC (V3.3 · 协同共振与全息资金流验证强化版)...")
        method_name = "_calculate_stock_sector_sync"
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
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 获取原始数据
        stock_signal_raw = self._get_safe_series(df, stock_signal_name, 0.0, method_name=method_name)
        sector_rank_raw = self._get_safe_series(df, sector_rank_name, 0.0, method_name=method_name)
        sector_momentum_raw = self._get_safe_series(df, sector_momentum_name, 0.0, method_name=method_name)
        main_force_net_flow_raw = self._get_safe_series(df, main_force_net_flow_name, 0.0, method_name=method_name)
        flow_credibility_raw = self._get_safe_series(df, flow_credibility_name, 0.0, method_name=method_name)
        # 归一化当前状态和动量
        stock_strength_score = self._normalize_series(stock_signal_raw, target_index=df_index, bipolar=True)
        # 行业强度排名：使用MTF融合信号
        mtf_sector_rank_score = self._get_mtf_slope_accel_score(df, 'industry_strength_rank_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 板块动量：使用MTF融合信号
        mtf_sector_momentum_score = self._get_mtf_slope_accel_score(df, 'industry_strength_rank_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        main_force_net_flow_norm = self._normalize_series(main_force_net_flow_raw, target_index=df_index, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, target_index=df_index, bipolar=False)
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
        # --- 最终融合 ---
        relationship_score = final_bullish_score + final_bearish_score
        final_score = relationship_score.clip(-1, 1)
        return final_score.astype(np.float32)

    def _calculate_hot_sector_cooling(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1 · 军令直达版】“热门板块冷却”专属关系计算引擎
        - 核心重构: 创立“状态与方向”乘积模型，审判“高位下的资金背叛”。
        - 信号升级: 资金信号升级为更具意图的 `main_force_net_flow_calibrated_D`。
        - 核心逻辑: 瞬时关系分 = 板块热度(状态分) * 主力出逃(方向分)。
        """
        hotness_signal_name = 'THEME_HOTNESS_SCORE_D'
        flow_signal_name = 'main_force_net_flow_calibrated_D'
        required_signals = [hotness_signal_name, flow_signal_name]
        if not self._validate_required_signals(df, required_signals, "_calculate_hot_sector_cooling"):
            return pd.Series(dtype=np.float32)
        df_index = df.index
        hotness_signal_raw = self._get_safe_series(df, hotness_signal_name, 0.0, method_name="_calculate_hot_sector_cooling")
        flow_signal_raw = self._get_safe_series(df, flow_signal_name, 0.0, method_name="_calculate_hot_sector_cooling")
        # 归一化状态与方向
        hotness_state_score = self._normalize_series(hotness_signal_raw, df_index, bipolar=False)
        flow_direction_score = self._normalize_series(flow_signal_raw, df_index, bipolar=True)
        # 只关注主力出逃的部分
        outflow_score = flow_direction_score.clip(upper=0).abs()
        # 核心逻辑：寒潮来袭模型
        relationship_score = hotness_state_score * outflow_score
        final_score = relationship_score.clip(0, 1) # 这是一个单极风险信号
        return final_score

    def _calculate_pf_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 调用修正版】计算“价资关系”的专属方法。
        - 核心修复: 修正了对 `_perform_meta_analysis_on_score` 的调用，确保传递了完整的 `df` 和 `df.index`。
        """
        print(f"    -> [过程层] 正在计算 {config.get('name')} (V3.0 · 信念驱动版)...")
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # [修改] 修正调用参数，同时传递 df 和 df.index
        meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df.index)
        return meta_score

    def _calculate_pc_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 调用修正版】计算“价筹关系”的专属方法。
        - 核心修复: 修正了对 `_perform_meta_analysis_on_score` 的调用，确保传递了完整的 `df` 和 `df.index`。
        """
        print(f"    -> [过程层] 正在计算 {config.get('name')} (V3.0 · 结构验证版)...")
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # [修改] 修正调用参数，同时传递 df 和 df.index
        meta_score = self._perform_meta_analysis_on_score(relationship_score, config, df, df.index)
        return meta_score

    def _calculate_storm_eye_calm(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        print("    -> [过程层] 正在计算 PROCESS_META_STORM_EYE_CALM (V10.1 · 情绪与主力意图精细化感知版)...")
        method_name = "_calculate_storm_eye_calm"
        df_index = df.index
        params = get_param_value(config.get('storm_eye_calm_params'), {})
        # --- 1. 获取配置参数 ---
        energy_compression_weights = get_param_value(params.get('energy_compression_weights'), {"tension": 0.2, "bbw_inverted": 0.15, "vol_instability_inverted": 0.15, "equilibrium_compression": 0.15, "bbw_slope_inverted": 0.1, "vol_instability_slope_inverted": 0.1, "dyn_stability": 0.05, "market_tension": 0.05, "price_sample_entropy_inverted": 0.025, "price_volume_entropy_inverted": 0.025, "price_fractal_dimension_calm": 0.025, "volume_structure_skew_inverted": 0.025, "volume_profile_entropy_inverted": 0.025})
        volume_exhaustion_weights = get_param_value(params.get('volume_exhaustion_weights'), {"volume_atrophy": 0.15, "turnover_rate_inverted": 0.1, "counterparty_exhaustion": 0.1, "order_book_liquidity_inverted": 0.1, "buy_quote_exhaustion": 0.1, "sell_quote_exhaustion": 0.05, "turnover_rate_slope_inverted": 0.05, "order_book_imbalance_inverted": 0.05, "micro_price_impact_asymmetry_inverted": 0.05, "bid_side_liquidity_inverted": 0.05, "ask_side_liquidity_inverted": 0.05, "vpin_score_inverted": 0.025, "bid_liquidity_sample_entropy_inverted": 0.025, "volume_structure_skew_inverted": 0.05, "volume_profile_entropy_inverted": 0.05, "turnover_rate_raw_inverted": 0.05})
        main_force_covert_intent_weights = get_param_value(params.get('main_force_covert_intent_weights'), {"stealth_ops": 0.1, "split_order_accum": 0.1, "mf_net_flow_positive": 0.1, "mf_cost_advantage_positive": 0.08, "mf_buy_ofi_positive": 0.08, "mf_t0_buy_efficiency_positive": 0.08, "mf_net_flow_slope_positive": 0.08, "order_book_imbalance_positive": 0.05, "micro_price_impact_asymmetry_positive": 0.05, "mf_vwap_guidance_neutrality": 0.05, "vwap_control_neutrality": 0.05, "observed_large_order_size_avg_inverted": 0.025, "market_impact_cost_inverted": 0.025, "main_force_net_flow_volatility_inverted": 0.05, "main_force_flow_ambiguity": 0.15})
        subdued_market_sentiment_weights = get_param_value(params.get('subdued_market_sentiment_weights'), {"sentiment_pendulum_negative": 0.075, "market_sentiment_inverted": 0.075, "retail_panic_inverted": 0.05, "retail_fomo_inverted": 0.05, "loser_pain_positive": 0.05, "liquidity_tide_calm": 0.075, "hurst_calm": 0.075, "sentiment_neutrality": 0.1, "sentiment_pendulum_neutrality": 0.1, "sentiment_volatility_inverted": 0.075, "sentiment_pendulum_volatility_inverted": 0.05, "long_term_sentiment_subdued": 0.05, "market_sentiment_not_extreme": 0.1, "sentiment_pendulum_not_extreme": 0.075, "market_sentiment_boring_score": 0.05, "price_reversion_velocity_inverted": 0.05, "structural_entropy_change_inverted": 0.05})
        breakout_readiness_weights = get_param_value(params.get('breakout_readiness_weights'), {"struct_breakout_readiness": 0.3, "struct_platform_foundation": 0.25, "goodness_of_fit": 0.25, "platform_conviction": 0.2})
        mtf_cohesion_weights = get_param_value(params.get('mtf_cohesion_weights'), {"cohesion_score": 1.0})
        final_fusion_weights = get_param_value(params.get('final_fusion_weights'), {"energy_compression": 0.2, "volume_exhaustion": 0.2, "main_force_covert_intent": 0.2, "subdued_market_sentiment": 0.15, "breakout_readiness": 0.15, "mtf_cohesion": 0.1})
        price_calmness_modulator_params = get_param_value(params.get('price_calmness_modulator_params'), {"slope_period": 5, "pct_change_threshold": 0.005, "modulator_factor": 0.5})
        main_force_control_adjudicator_params = get_param_value(params.get('main_force_control_adjudicator'), {"control_signal": "control_solidity_index_D", "activity_signal": "main_force_activity_ratio_D", "veto_threshold": -0.2, "amplifier_factor": 0.5})
        mtf_slope_accel_weights = get_param_value(params.get('mtf_slope_accel_weights'), {})
        regime_modulator_params = get_param_value(params.get('regime_modulator_params'), {})
        mtf_cohesion_base_signals = get_param_value(params.get('mtf_cohesion_base_signals'), ['close_D', 'volume_D', 'main_force_net_flow_calibrated_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'turnover_rate_f_D', 'BBW_21_2.0_D'])
        sentiment_volatility_window = get_param_value(params.get('sentiment_volatility_window'), 21)
        long_term_sentiment_window = get_param_value(params.get('long_term_sentiment_window'), 55)
        main_force_flow_volatility_window = get_param_value(params.get('main_force_flow_volatility_window'), 21)
        sentiment_neutral_range = get_param_value(params.get('sentiment_neutral_range'), 1.0)
        sentiment_pendulum_neutral_range = get_param_value(params.get('sentiment_pendulum_neutral_range'), 0.2)
        ambiguity_components_weights = get_param_value(params.get('ambiguity_components_weights'), {"directionality_neutrality": 0.15, "net_flow_near_zero": 0.15, "deception_score": 0.15, "wash_trade_score": 0.15, "mf_conviction_neutrality": 0.1, "deception_lure_neutrality": 0.1, "covert_action_score": 0.1, "main_force_slippage_inverted": 0.05, "main_force_flow_gini_inverted": 0.05, "micro_impact_elasticity_positive": 0.05, "order_flow_imbalance_neutrality": 0.05, "liquidity_authenticity_positive": 0.05})
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 2. 校验所有必需的信号 ---
        required_signals = [
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'control_solidity_index_D',
            'BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'turnover_rate_f_D',
            'counterparty_exhaustion_index_D', 'main_force_conviction_index_D',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY',
            'main_force_net_flow_calibrated_D', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'market_sentiment_score_D', 'SLOPE_5_close_D', 'pct_change_D',
            'equilibrium_compression_index_D',
            'order_book_liquidity_supply_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'main_force_cost_advantage_D', 'main_force_buy_ofi_D', 'main_force_t0_buy_efficiency_D',
            'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D', 'loser_pain_index_D',
            'SCORE_STRUCT_BREAKOUT_READINESS', 'SCORE_STRUCT_PLATFORM_FOUNDATION',
            # 'goodness_of_fit_score_D', 'platform_conviction_score_D', # 已移除，因为它们并非每天都存在
            'main_force_activity_ratio_D', 'order_book_imbalance_D', 'micro_price_impact_asymmetry_D', 'ADX_14_D',
            # 新增信号
            'SCORE_DYN_AXIOM_STABILITY', 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION',
            'SAMPLE_ENTROPY_13d_D', 'price_volume_entropy_D', 'FRACTAL_DIMENSION_89d_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D', 'vpin_score_D', 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D',
            'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D', 'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'observed_large_order_size_avg_D', 'market_impact_cost_D', 'main_force_flow_directionality_D',
            'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 'HURST_144d_D', 'turnover_rate_D',
            'volume_structure_skew_D', 'volume_profile_entropy_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'covert_accumulation_signal_D', 'covert_distribution_signal_D',
            'main_force_slippage_index_D', 'main_force_flow_gini_D',
            'price_reversion_velocity_D', 'structural_entropy_change_D',
            'micro_impact_elasticity_D', 'order_flow_imbalance_score_D', 'liquidity_authenticity_score_D',
            'mean_reversion_frequency_D', 'trend_alignment_index_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in mtf_cohesion_base_signals:
            for period_str in get_param_value(mtf_slope_accel_weights.get('slope_periods'), {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in get_param_value(mtf_slope_accel_weights.get('accel_periods'), {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, method_name):
            print(f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 3. 获取原始数据 (包括新增的) ---
        # Energy Compression
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', np.nan)
        bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', np.nan, method_name=method_name)
        vol_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', np.nan, method_name=method_name)
        equilibrium_compression_raw = self._get_safe_series(df, 'equilibrium_compression_index_D', np.nan, method_name=method_name)
        dyn_stability_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', np.nan)
        market_tension_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', np.nan)
        price_sample_entropy_raw = self._get_safe_series(df, 'SAMPLE_ENTROPY_13d_D', np.nan, method_name=method_name)
        price_volume_entropy_raw = self._get_safe_series(df, 'price_volume_entropy_D', np.nan, method_name=method_name)
        price_fractal_dimension_raw = self._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', np.nan, method_name=method_name)
        volume_structure_skew_raw = self._get_safe_series(df, 'volume_structure_skew_D', np.nan, method_name=method_name)
        volume_profile_entropy_raw = self._get_safe_series(df, 'volume_profile_entropy_D', np.nan, method_name=method_name)
        # Volume Exhaustion
        atrophy_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', np.nan)
        turnover_rate_f_raw = self._get_safe_series(df, 'turnover_rate_f_D', np.nan, method_name=method_name)
        turnover_rate_raw = self._get_safe_series(df, 'turnover_rate_D', np.nan, method_name=method_name)
        counterparty_exhaustion_raw = self._get_safe_series(df, 'counterparty_exhaustion_index_D', np.nan, method_name=method_name)
        order_book_liquidity_raw = self._get_safe_series(df, 'order_book_liquidity_supply_D', np.nan, method_name=method_name)
        buy_quote_exhaustion_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', np.nan, method_name=method_name)
        sell_quote_exhaustion_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', np.nan, method_name=method_name)
        order_book_imbalance_raw = self._get_safe_series(df, 'order_book_imbalance_D', np.nan, method_name=method_name)
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, 'micro_price_impact_asymmetry_D', np.nan, method_name=method_name)
        bid_side_liquidity_raw = self._get_safe_series(df, 'bid_side_liquidity_D', np.nan, method_name=method_name)
        ask_side_liquidity_raw = self._get_safe_series(df, 'ask_side_liquidity_D', np.nan, method_name=method_name)
        vpin_score_raw = self._get_safe_series(df, 'vpin_score_D', np.nan, method_name=method_name)
        bid_liquidity_sample_entropy_raw = self._get_safe_series(df, 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D', np.nan, method_name=method_name)
        # Main Force Covert Intent
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', np.nan)
        split_order_accum_score = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', np.nan)
        mf_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', np.nan, method_name=method_name)
        mf_net_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', np.nan, method_name=method_name)
        mf_cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', np.nan, method_name=method_name)
        mf_buy_ofi_raw = self._get_safe_series(df, 'main_force_buy_ofi_D', np.nan, method_name=method_name)
        mf_t0_buy_efficiency_raw = self._get_safe_series(df, 'main_force_t0_buy_efficiency_D', np.nan, method_name=method_name)
        mf_vwap_up_guidance_raw = self._get_safe_series(df, 'main_force_vwap_up_guidance_D', np.nan, method_name=method_name)
        mf_vwap_down_guidance_raw = self._get_safe_series(df, 'main_force_vwap_down_guidance_D', np.nan, method_name=method_name)
        vwap_buy_control_raw = self._get_safe_series(df, 'vwap_buy_control_strength_D', np.nan, method_name=method_name)
        vwap_sell_control_raw = self._get_safe_series(df, 'vwap_sell_control_strength_D', np.nan, method_name=method_name)
        observed_large_order_size_avg_raw = self._get_safe_series(df, 'observed_large_order_size_avg_D', np.nan, method_name=method_name)
        market_impact_cost_raw = self._get_safe_series(df, 'market_impact_cost_D', np.nan, method_name=method_name)
        main_force_flow_directionality_raw = self._get_safe_series(df, 'main_force_flow_directionality_D', np.nan, method_name=method_name)
        mf_net_flow_std_raw = mf_net_flow_raw.rolling(window=main_force_flow_volatility_window, min_periods=1).std()
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', np.nan, method_name=method_name)
        wash_trade_intensity_raw = self._get_safe_series(df, 'wash_trade_intensity_D', np.nan, method_name=method_name)
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', np.nan, method_name=method_name)
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', np.nan, method_name=method_name)
        covert_accumulation_raw = self._get_safe_series(df, 'covert_accumulation_signal_D', np.nan, method_name=method_name)
        covert_distribution_raw = self._get_safe_series(df, 'covert_distribution_signal_D', np.nan, method_name=method_name)
        main_force_slippage_raw = self._get_safe_series(df, 'main_force_slippage_index_D', np.nan, method_name=method_name)
        main_force_flow_gini_raw = self._get_safe_series(df, 'main_force_flow_gini_D', np.nan, method_name=method_name)
        # 新增主力意图相关原始数据
        micro_impact_elasticity_raw = self._get_safe_series(df, 'micro_impact_elasticity_D', np.nan, method_name=method_name)
        order_flow_imbalance_score_raw = self._get_safe_series(df, 'order_flow_imbalance_score_D', np.nan, method_name=method_name)
        liquidity_authenticity_score_raw = self._get_safe_series(df, 'liquidity_authenticity_score_D', np.nan, method_name=method_name)
        # Subdued Market Sentiment
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', np.nan)
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', np.nan, method_name=method_name)
        retail_panic_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', np.nan, method_name=method_name)
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', np.nan, method_name=method_name)
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', np.nan, method_name=method_name)
        liquidity_tide_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', np.nan)
        hurst_raw = self._get_safe_series(df, 'HURST_144d_D', np.nan, method_name=method_name)
        market_sentiment_std_raw = market_sentiment_raw.rolling(window=sentiment_volatility_window, min_periods=1).std()
        sentiment_pendulum_std_raw = sentiment_pendulum_score.rolling(window=sentiment_volatility_window, min_periods=1).std()
        market_sentiment_long_term_mean = market_sentiment_raw.rolling(window=long_term_sentiment_window, min_periods=1).mean()
        price_reversion_velocity_raw = self._get_safe_series(df, 'price_reversion_velocity_D', np.nan, method_name=method_name)
        structural_entropy_change_raw = self._get_safe_series(df, 'structural_entropy_change_D', np.nan, method_name=method_name)
        # 新增情绪相关原始数据
        mean_reversion_frequency_raw = self._get_safe_series(df, 'mean_reversion_frequency_D', np.nan, method_name=method_name)
        trend_alignment_index_raw = self._get_safe_series(df, 'trend_alignment_index_D', np.nan, method_name=method_name)
        # Breakout Readiness
        struct_breakout_readiness_score = self._get_atomic_score(df, 'SCORE_STRUCT_BREAKOUT_READINESS', np.nan)
        struct_platform_foundation_score = self._get_atomic_score(df, 'SCORE_STRUCT_PLATFORM_FOUNDATION', np.nan)
        # 对于 goodness_of_fit_score_D 和 platform_conviction_score_D，即使它们不在df中，
        # _get_safe_series也会返回一个填充了np.nan的Series，
        # 随后_normalize_series会将其转换为0.0，这符合“没有平台信息时贡献为0”的逻辑。
        # 因此，无需将它们列为required_signals。
        goodness_of_fit_raw = self._get_safe_series(df, 'goodness_of_fit_score_D', np.nan, method_name=method_name)
        platform_conviction_raw = self._get_safe_series(df, 'platform_conviction_score_D', np.nan, method_name=method_name)
        # Modulators
        price_slope_raw = self._get_safe_series(df, f'SLOPE_{price_calmness_modulator_params.get("slope_period", 5)}_close_D', np.nan, method_name=method_name)
        pct_change_raw = self._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        control_solidity_raw = self._get_safe_series(df, main_force_control_adjudicator_params.get('control_signal', 'control_solidity_index_D'), np.nan, method_name=method_name)
        mf_activity_ratio_raw = self._get_safe_series(df, main_force_control_adjudicator_params.get('activity_signal', 'main_force_activity_ratio_D'), np.nan, method_name=method_name)
        volatility_regime_raw = self._get_safe_series(df, regime_modulator_params.get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D'), np.nan, method_name=method_name)
        trend_regime_raw = self._get_safe_series(df, regime_modulator_params.get('trend_signal', 'ADX_14_D'), np.nan, method_name=method_name)
        # --- 4. 计算MTF斜率/加速度分数 ---
        bbw_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'BBW_21_2.0_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        vol_instability_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        turnover_rate_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mf_net_flow_slope_positive = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_cohesion_score = self._get_mtf_cohesion_score(df, mtf_cohesion_base_signals, mtf_slope_accel_weights, df_index, method_name)
        # --- 5. 归一化和计算各维度分数 ---
        # Energy Compression
        bbw_inverted_score = self._normalize_series(bbw_raw, target_index=df_index, ascending=False)
        vol_instability_inverted_score = self._normalize_series(vol_instability_raw, target_index=df_index, ascending=False)
        equilibrium_compression_score = self._normalize_series(equilibrium_compression_raw, target_index=df_index, ascending=True)
        dyn_stability_norm = self._normalize_series(dyn_stability_score, target_index=df_index, bipolar=False)
        market_tension_norm = self._normalize_series(market_tension_score, target_index=df_index, bipolar=False)
        price_sample_entropy_inverted = self._normalize_series(price_sample_entropy_raw, target_index=df_index, ascending=False)
        price_volume_entropy_inverted = self._normalize_series(price_volume_entropy_raw, target_index=df_index, ascending=False)
        price_fractal_dimension_calm = (1 - (price_fractal_dimension_raw - 1.5).abs() / 0.5).clip(0, 1)
        volume_structure_skew_inverted = self._normalize_series(volume_structure_skew_raw.abs(), target_index=df_index, ascending=False)
        volume_profile_entropy_inverted = self._normalize_series(volume_profile_entropy_raw, target_index=df_index, ascending=False)
        energy_compression_scores_dict = {
            'tension': tension_score, 'bbw_inverted': bbw_inverted_score, 'vol_instability_inverted': vol_instability_inverted_score,
            'equilibrium_compression': equilibrium_compression_score, 'bbw_slope_inverted': bbw_slope_inverted_score,
            'vol_instability_slope_inverted': vol_instability_slope_inverted_score,
            'dyn_stability': dyn_stability_norm, 'market_tension': market_tension_norm,
            'price_sample_entropy_inverted': price_sample_entropy_inverted, 'price_volume_entropy_inverted': price_volume_entropy_inverted,
            'price_fractal_dimension_calm': price_fractal_dimension_calm,
            'volume_structure_skew_inverted': volume_structure_skew_inverted, 'volume_profile_entropy_inverted': volume_profile_entropy_inverted
        }
        energy_compression_score = _robust_geometric_mean(energy_compression_scores_dict, energy_compression_weights, df_index)
        # Volume Exhaustion
        turnover_rate_inverted_score = self._normalize_series(turnover_rate_f_raw, target_index=df_index, ascending=False)
        turnover_rate_raw_inverted = self._normalize_series(turnover_rate_raw, target_index=df_index, ascending=False)
        counterparty_exhaustion_score = self._normalize_series(counterparty_exhaustion_raw, target_index=df_index, ascending=True)
        order_book_liquidity_inverted_score = self._normalize_series(order_book_liquidity_raw, target_index=df_index, ascending=False)
        buy_quote_exhaustion_score = self._normalize_series(buy_quote_exhaustion_raw, target_index=df_index, ascending=True)
        sell_quote_exhaustion_score = self._normalize_series(sell_quote_exhaustion_raw, target_index=df_index, ascending=True)
        order_book_imbalance_inverted = self._normalize_series(order_book_imbalance_raw.abs(), target_index=df_index, ascending=False)
        micro_price_impact_asymmetry_inverted = self._normalize_series(micro_price_impact_asymmetry_raw.abs(), target_index=df_index, ascending=False)
        bid_side_liquidity_inverted = self._normalize_series(bid_side_liquidity_raw, target_index=df_index, ascending=False)
        ask_side_liquidity_inverted = self._normalize_series(ask_side_liquidity_raw, target_index=df_index, ascending=False)
        vpin_score_inverted = self._normalize_series(vpin_score_raw, target_index=df_index, ascending=False)
        bid_liquidity_sample_entropy_inverted = self._normalize_series(bid_liquidity_sample_entropy_raw, target_index=df_index, ascending=False)
        volume_exhaustion_scores_dict = {
            'volume_atrophy': atrophy_score, 'turnover_rate_inverted': turnover_rate_inverted_score,
            'counterparty_exhaustion': counterparty_exhaustion_score, 'order_book_liquidity_inverted': order_book_liquidity_inverted_score,
            'buy_quote_exhaustion': buy_quote_exhaustion_score, 'sell_quote_exhaustion': sell_quote_exhaustion_score,
            'turnover_rate_slope_inverted': turnover_rate_slope_inverted_score,
            'order_book_imbalance_inverted': order_book_imbalance_inverted,
            'micro_price_impact_asymmetry_inverted': micro_price_impact_asymmetry_inverted,
            'bid_side_liquidity_inverted': bid_side_liquidity_inverted, 'ask_side_liquidity_inverted': ask_side_liquidity_inverted,
            'vpin_score_inverted': vpin_score_inverted, 'bid_liquidity_sample_entropy_inverted': bid_liquidity_sample_entropy_inverted,
            'volume_structure_skew_inverted': volume_structure_skew_inverted, 'volume_profile_entropy_inverted': volume_profile_entropy_inverted,
            'turnover_rate_raw_inverted': turnover_rate_raw_inverted
        }
        volume_exhaustion_score = _robust_geometric_mean(volume_exhaustion_scores_dict, volume_exhaustion_weights, df_index)
        # Main Force Covert Intent
        stealth_ops_normalized = self._normalize_series(stealth_ops_score, target_index=df_index, ascending=True)
        split_order_accum_normalized = self._normalize_series(split_order_accum_score, target_index=df_index, ascending=True)
        mf_conviction_positive = self._normalize_series(mf_conviction_raw, target_index=df_index, bipolar=True).clip(lower=0)
        mf_net_flow_positive = self._normalize_series(mf_net_flow_raw, target_index=df_index, bipolar=True).clip(lower=0)
        mf_cost_advantage_positive = self._normalize_series(mf_cost_advantage_raw, target_index=df_index, bipolar=True).clip(lower=0)
        mf_buy_ofi_positive = self._normalize_series(mf_buy_ofi_raw, target_index=df_index, ascending=True)
        mf_t0_buy_efficiency_positive = self._normalize_series(mf_t0_buy_efficiency_raw, target_index=df_index, ascending=True)
        order_book_imbalance_positive = self._normalize_series(order_book_imbalance_raw.clip(lower=0), target_index=df_index, ascending=True)
        micro_price_impact_asymmetry_positive = self._normalize_series(micro_price_impact_asymmetry_raw.clip(lower=0), target_index=df_index, ascending=True)
        mf_vwap_guidance_neutrality = 1 - self._normalize_series((mf_vwap_up_guidance_raw - mf_vwap_down_guidance_raw).abs(), target_index=df_index, ascending=True)
        vwap_control_neutrality = 1 - self._normalize_series((vwap_buy_control_raw - vwap_sell_control_raw).abs(), target_index=df_index, ascending=True)
        observed_large_order_size_avg_inverted = self._normalize_series(observed_large_order_size_avg_raw, target_index=df_index, ascending=False)
        market_impact_cost_inverted = self._normalize_series(market_impact_cost_raw, target_index=df_index, ascending=False)
        main_force_net_flow_volatility_inverted = self._normalize_series(mf_net_flow_std_raw, target_index=df_index, ascending=False)
        # 主力资金流模糊性 (main_force_flow_ambiguity) 的组成部分
        main_force_flow_directionality_neutrality = 1 - self._normalize_series(main_force_flow_directionality_raw.abs(), target_index=df_index, ascending=True)
        mf_net_flow_near_zero = 1 - self._normalize_series(mf_net_flow_raw.abs(), target_index=df_index, ascending=True)
        deception_score = self._normalize_series(deception_index_raw, target_index=df_index, ascending=True)
        wash_trade_score = self._normalize_series(wash_trade_intensity_raw, target_index=df_index, ascending=True)
        mf_conviction_neutrality = 1 - self._normalize_series(mf_conviction_raw.abs(), target_index=df_index, ascending=True)
        deception_lure_neutrality = 1 - self._normalize_series(deception_lure_long_raw.abs() + deception_lure_short_raw.abs(), target_index=df_index, ascending=True)
        # 修正 covert_action_score 的计算逻辑
        covert_action_score = (1 - (self._normalize_series(covert_accumulation_raw, target_index=df_index, ascending=True) + self._normalize_series(covert_distribution_raw, target_index=df_index, ascending=True)).clip(0,1))
        main_force_slippage_inverted = self._normalize_series(main_force_slippage_raw, target_index=df_index, ascending=False)
        main_force_flow_gini_inverted = self._normalize_series(main_force_flow_gini_raw, target_index=df_index, ascending=False)
        # 新增模糊性组件
        micro_impact_elasticity_positive = self._normalize_series(micro_impact_elasticity_raw, target_index=df_index, ascending=True)
        order_flow_imbalance_neutrality = 1 - self._normalize_series(order_flow_imbalance_score_raw.abs(), target_index=df_index, ascending=True)
        liquidity_authenticity_positive = self._normalize_series(liquidity_authenticity_score_raw, target_index=df_index, ascending=True)
        ambiguity_components = {
            'directionality_neutrality': main_force_flow_directionality_neutrality,
            'net_flow_near_zero': mf_net_flow_near_zero,
            'deception_score': deception_score,
            'wash_trade_score': wash_trade_score,
            'mf_conviction_neutrality': mf_conviction_neutrality,
            'deception_lure_neutrality': deception_lure_neutrality,
            'covert_action_score': covert_action_score,
            'main_force_slippage_inverted': main_force_slippage_inverted,
            'main_force_flow_gini_inverted': main_force_flow_gini_inverted,
            'micro_impact_elasticity_positive': micro_impact_elasticity_positive,
            'order_flow_imbalance_neutrality': order_flow_imbalance_neutrality,
            'liquidity_authenticity_positive': liquidity_authenticity_positive
        }
        main_force_flow_ambiguity = _robust_geometric_mean(ambiguity_components, ambiguity_components_weights, df_index)
        main_force_covert_intent_scores_dict = {
            'stealth_ops': stealth_ops_normalized, 'split_order_accum': split_order_accum_normalized,
            'mf_net_flow_positive': mf_net_flow_positive,
            'mf_cost_advantage_positive': mf_cost_advantage_positive, 'mf_buy_ofi_positive': mf_buy_ofi_positive,
            'mf_t0_buy_efficiency_positive': mf_t0_buy_efficiency_positive, 'mf_net_flow_slope_positive': mf_net_flow_slope_positive,
            'order_book_imbalance_positive': order_book_imbalance_positive,
            'micro_price_impact_asymmetry_positive': micro_price_impact_asymmetry_positive,
            'mf_vwap_guidance_neutrality': mf_vwap_guidance_neutrality, 'vwap_control_neutrality': vwap_control_neutrality,
            'observed_large_order_size_avg_inverted': observed_large_order_size_avg_inverted, 'market_impact_cost_inverted': market_impact_cost_inverted,
            'main_force_net_flow_volatility_inverted': main_force_net_flow_volatility_inverted,
            'main_force_flow_ambiguity': main_force_flow_ambiguity
        }
        main_force_covert_intent_score = _robust_geometric_mean(main_force_covert_intent_scores_dict, main_force_covert_intent_weights, df_index)
        # Subdued Market Sentiment
        sentiment_pendulum_negative = self._normalize_series(sentiment_pendulum_score, target_index=df_index, bipolar=True).clip(upper=0).abs()
        market_sentiment_inverted = self._normalize_series(market_sentiment_raw, target_index=df_index, ascending=False)
        retail_panic_inverted = self._normalize_series(retail_panic_raw, target_index=df_index, ascending=False)
        retail_fomo_inverted = self._normalize_series(retail_fomo_raw, target_index=df_index, ascending=False)
        loser_pain_positive = self._normalize_series(loser_pain_raw, target_index=df_index, ascending=True)
        liquidity_tide_calm = self._normalize_series(liquidity_tide_score.abs(), target_index=df_index, ascending=False)
        hurst_calm = (1 - (hurst_raw - 0.5).abs() / 0.5).clip(0, 1)
        sentiment_neutrality = 1 - self._normalize_series(market_sentiment_raw.abs(), target_index=df_index, ascending=True)
        sentiment_pendulum_neutrality = 1 - self._normalize_series(sentiment_pendulum_score.abs(), target_index=df_index, bipolar=True).abs()
        sentiment_volatility_inverted = self._normalize_series(market_sentiment_std_raw, target_index=df_index, ascending=False)
        sentiment_pendulum_volatility_inverted = self._normalize_series(sentiment_pendulum_std_raw, target_index=df_index, ascending=False)
        long_term_sentiment_subdued = self._normalize_series(market_sentiment_long_term_mean - market_sentiment_raw, target_index=df_index, ascending=True)
        # 修正 sentiment_pendulum_not_extreme 和 market_sentiment_not_extreme
        sentiment_pendulum_not_extreme = (1 - (sentiment_pendulum_score.abs() - sentiment_pendulum_neutral_range).clip(lower=0) / (sentiment_pendulum_score.abs().max() - sentiment_pendulum_neutral_range + 1e-9)).clip(0, 1)
        market_sentiment_not_extreme = (1 - (market_sentiment_raw.abs() - sentiment_neutral_range).clip(lower=0) / (market_sentiment_raw.abs().max() - sentiment_neutral_range + 1e-9)).clip(0, 1)
        market_sentiment_boring_score = _robust_geometric_mean({'volatility_inverted': sentiment_volatility_inverted, 'not_extreme': market_sentiment_not_extreme}, {'volatility_inverted': 0.5, 'not_extreme': 0.5}, df_index)
        price_reversion_velocity_inverted = self._normalize_series(price_reversion_velocity_raw, target_index=df_index, ascending=False)
        structural_entropy_change_inverted = self._normalize_series(structural_entropy_change_raw.abs(), target_index=df_index, ascending=False)
        # 新增情绪组件
        mean_reversion_frequency_inverted = self._normalize_series(mean_reversion_frequency_raw, target_index=df_index, ascending=False)
        trend_alignment_positive = self._normalize_series(trend_alignment_index_raw, target_index=df_index, ascending=True)
        subdued_market_sentiment_scores_dict = {
            'sentiment_pendulum_negative': sentiment_pendulum_negative, 'market_sentiment_inverted': market_sentiment_inverted,
            'retail_panic_inverted': retail_panic_inverted, 'retail_fomo_inverted': retail_fomo_inverted,
            'loser_pain_positive': loser_pain_positive,
            'liquidity_tide_calm': liquidity_tide_calm, 'hurst_calm': hurst_calm,
            'sentiment_neutrality': sentiment_neutrality, 'sentiment_pendulum_neutrality': sentiment_pendulum_neutrality,
            'sentiment_volatility_inverted': sentiment_volatility_inverted,
            'sentiment_pendulum_volatility_inverted': sentiment_pendulum_volatility_inverted,
            'long_term_sentiment_subdued': long_term_sentiment_subdued,
            'market_sentiment_not_extreme': market_sentiment_not_extreme,
            'sentiment_pendulum_not_extreme': sentiment_pendulum_not_extreme,
            'market_sentiment_boring_score': market_sentiment_boring_score,
            'price_reversion_velocity_inverted': price_reversion_velocity_inverted,
            'structural_entropy_change_inverted': structural_entropy_change_inverted,
            'mean_reversion_frequency_inverted': mean_reversion_frequency_inverted,
            'trend_alignment_positive': trend_alignment_positive
        }
        subdued_market_sentiment_score = _robust_geometric_mean(subdued_market_sentiment_scores_dict, subdued_market_sentiment_weights, df_index)
        # Breakout Readiness
        goodness_of_fit_score = self._normalize_series(goodness_of_fit_raw, target_index=df_index, ascending=True)
        platform_conviction_score = self._normalize_series(platform_conviction_raw, target_index=df_index, ascending=True)
        breakout_readiness_scores_dict = {
            'struct_breakout_readiness': struct_breakout_readiness_score,
            'struct_platform_foundation': struct_platform_foundation_score,
            'goodness_of_fit': goodness_of_fit_score,
            'platform_conviction': platform_conviction_score
        }
        breakout_readiness_score = _robust_geometric_mean(breakout_readiness_scores_dict, breakout_readiness_weights, df_index)
        # --- 6. 市场情境动态调节器 ---
        market_regime_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        if get_param_value(regime_modulator_params.get('enabled'), False):
            volatility_sensitivity = get_param_value(regime_modulator_params.get('volatility_sensitivity'), 0.5)
            trend_sensitivity = get_param_value(regime_modulator_params.get('trend_sensitivity'), 0.5)
            base_modulator_factor = get_param_value(regime_modulator_params.get('base_modulator_factor'), 1.0)
            min_modulator = get_param_value(regime_modulator_params.get('min_modulator'), 0.8)
            max_modulator = get_param_value(regime_modulator_params.get('max_modulator'), 1.2)
            volatility_norm = self._normalize_series(volatility_regime_raw, target_index=df_index, ascending=False)
            trend_norm = self._normalize_series(trend_regime_raw, target_index=df_index, ascending=False)
            market_regime_modulator = (
                base_modulator_factor +
                (volatility_norm * volatility_sensitivity + trend_norm * trend_sensitivity) / (volatility_sensitivity + trend_sensitivity + 1e-9)
            ).clip(min_modulator, max_modulator)
        # --- 7. 最终融合 ---
        adjusted_final_fusion_weights = {k: v * market_regime_modulator for k, v in final_fusion_weights.items()}
        base_calm_scores_dict = {
            'energy_compression': energy_compression_score,
            'volume_exhaustion': volume_exhaustion_score,
            'main_force_covert_intent': main_force_covert_intent_score,
            'subdued_market_sentiment': subdued_market_sentiment_score,
            'breakout_readiness': breakout_readiness_score,
            'mtf_cohesion': mtf_cohesion_score
        }
        base_calm_score = _robust_geometric_mean(base_calm_scores_dict, adjusted_final_fusion_weights, df_index)
        price_slope_norm_bipolar = self._normalize_series(price_slope_raw, target_index=df_index, bipolar=True)
        pct_change_abs_norm_inverted = self._normalize_series(pct_change_raw.abs(), target_index=df_index, ascending=False)
        # 修正 price_calmness_modulator 的参数名
        price_calmness_modulator = (price_calmness_modulator_params.get('modulator_factor', 0.5) * (1 - price_slope_norm_bipolar.abs()) + (1 - price_calmness_modulator_params.get('modulator_factor', 0.5)) * pct_change_abs_norm_inverted).clip(0,1)
        price_calmness_amplifier = 1 + (price_calmness_modulator * price_calmness_modulator_params.get('modulator_factor', 0.5))
        control_solidity_score = self._normalize_series(control_solidity_raw, target_index=df_index, bipolar=True)
        mf_activity_ratio_score = self._normalize_series(mf_activity_ratio_raw, target_index=df_index, ascending=True)
        veto_threshold = main_force_control_adjudicator_params.get('veto_threshold', -0.2)
        amplifier_factor = main_force_control_adjudicator_params.get('amplifier_factor', 0.5)
        final_score = base_calm_score * price_calmness_amplifier
        combined_control_score = (control_solidity_score * 0.7 + mf_activity_ratio_score * 0.3).clip(-1, 1)
        final_score = final_score.mask(combined_control_score < veto_threshold, 0.0)
        main_force_amplifier = 1 + (combined_control_score * amplifier_factor)
        final_score = (final_score * main_force_amplifier).clip(0, 1).fillna(0.0)
        # --- 调试信息 ---
        probe_dates = self.probe_dates
        return final_score.astype(np.float32)

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
        print("    -> [过程层] 正在计算 PROCESS_META_WASH_OUT_REBOUND (V2.5 · 深度情境与多维洗盘反弹版)...")
        method_name = "_calculate_process_wash_out_rebound"
        df_index = df.index
        # 直接使用 self.params，因为它已在 __init__ 中加载了 process_intelligence_params
        p_conf = self.params
        params = get_param_value(p_conf.get('wash_out_rebound_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"deception_context": 0.3, "panic_depth": 0.3, "rebound_quality": 0.4})
        # 修改代码：新增更多权重配置
        deception_context_weights = get_param_value(params.get('deception_context_weights'), {"wash_trade": 0.2, "active_selling": 0.15, "deception_lure_short": 0.2, "behavior_deception_index": 0.2, "stealth_ops": 0.15, "wash_trade_slope": 0.05, "active_selling_slope": 0.05})
        panic_depth_weights = get_param_value(params.get('panic_depth_weights'), {"panic_cascade": 0.2, "retail_surrender": 0.2, "loser_pain": 0.2, "holder_sentiment_inverted": 0.15, "sentiment_pendulum_negative": 0.15, "retail_surrender_slope": 0.075, "loser_pain_slope": 0.075}) # 提高权重
        rebound_quality_weights = get_param_value(params.get('rebound_quality_weights'), {"closing_strength": 0.15, "upward_purity": 0.15, "absorption_strength": 0.3, "offensive_absorption": 0.25, "mf_buy_execution_alpha": 0.05, "buy_sweep_intensity": 0.1}) # 调整权重
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
            (absorption_strength_normalized).pow(rebound_quality_weights.get('absorption_strength', 0.3)) * # 调整权重
            (offensive_absorption_normalized).pow(rebound_quality_weights.get('offensive_absorption', 0.25)) * # 调整权重
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

    def _calculate_process_covert_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.5 · 深度情境与多维隐蔽行动版】计算“隐蔽吸筹”的专属信号。
        - 核心升级: 优化 `market_context_score` 中的价格弱势判断，直接奖励价格弱势。
        - 【强化】优化 `covert_action_score` 中的欺诈信号融合，更侧重于正向的诱多欺诈。
        - 【调整】调整 `covert_action_weights` 中拆单吸筹的权重，使用原始指标的MTF融合版本。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_COVERT_ACCUMULATION (V2.5 · 深度情境与多维隐蔽行动版)...")
        method_name = "_calculate_process_covert_accumulation"
        # 直接使用 self.params，因为它已在 __init__ 中加载了 process_intelligence_params
        p_conf = self.params
        params = get_param_value(p_conf.get('covert_accumulation_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"market_context": 0.3, "covert_action": 0.4, "chip_optimization": 0.3})
        # 修改代码：新增更多权重配置
        market_context_weights = get_param_value(params.get('market_context_weights'), {"retail_panic": 0.2, "price_weakness": 0.2, "low_volatility": 0.2, "sentiment_pendulum_inverted": 0.15, "tension_inverted": 0.1, "market_sentiment_inverted": 0.1, "volatility_instability_inverted": 0.05})
        covert_action_weights = get_param_value(params.get('covert_action_weights'), {"suppressive_accum": 0.15, "main_force_flow": 0.15, "deception_lure_long": 0.15, "stealth_ops": 0.15, "hidden_accumulation_intensity": 0.1, "chip_historical_potential": 0.1, "mf_buy_ofi": 0.05, "mf_cost_advantage": 0.05, "mf_flow_slope": 0.05, "suppressive_accum_slope": 0.05})
        chip_optimization_weights = get_param_value(params.get('chip_optimization_weights'), {"chip_fatigue": 0.25, "loser_pain": 0.25, "holder_sentiment_inverted": 0.2, "turnover_purity_cost_opt": 0.15, "floating_chip_cleansing": 0.1, "total_loser_rate": 0.05})
        price_weakness_slope_window = get_param_value(params.get('price_weakness_slope_window'), 5)
        low_volatility_bbw_window = get_param_value(params.get('low_volatility_bbw_window'), 21)
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        # --- 2. 获取所有原始数据 ---
        required_df_columns = [
            'retail_panic_surrender_index_D', f'SLOPE_{price_weakness_slope_window}_close_D', f'BBW_{low_volatility_bbw_window}_2.0_D',
            'suppressive_accumulation_intensity_D', 'main_force_net_flow_calibrated_D', 'deception_index_D',
            'chip_fatigue_index_D', 'loser_pain_index_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            # 新增原始数据依赖
            'hidden_accumulation_intensity_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'main_force_buy_ofi_D', 'main_force_cost_advantage_D',
            'floating_chip_cleansing_efficiency_D', 'total_loser_rate_D', 'loser_concentration_90pct_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_df_columns
        for base_sig in ['main_force_net_flow_calibrated_D', 'suppressive_accumulation_intensity_D',
                         'deception_lure_long_intensity_D', 'hidden_accumulation_intensity_D',
                         'main_force_buy_ofi_D', 'main_force_cost_advantage_D',
                         'retail_panic_surrender_index_D', 'BBW_21_2.0_D', 'market_sentiment_score_D',
                         'VOLATILITY_INSTABILITY_INDEX_21d_D', 'chip_fatigue_index_D', 'loser_pain_index_D',
                         'floating_chip_cleansing_efficiency_D', 'total_loser_rate_D']:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_df_columns.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_df_columns.append(f'ACCEL_{period_str}_{base_sig}')
        required_atomic_signals = [
            'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 'SCORE_STRUCT_AXIOM_TENSION',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS',
            'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION'
        ]
        all_required_signals = required_df_columns + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, method_name):
            print(f"    -> [过程情报警告] _calculate_process_covert_accumulation 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        retail_panic_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        price_weakness_slope_raw = self._get_safe_series(df, f'SLOPE_{price_weakness_slope_window}_close_D', 0.0, method_name=method_name)
        bbw_raw = self._get_safe_series(df, f'BBW_{low_volatility_bbw_window}_2.0_D', 0.0, method_name=method_name)
        suppressive_accum_raw = self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name)
        main_force_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name)
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name)
        hidden_accumulation_intensity_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        chip_historical_potential_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        mf_buy_ofi_raw = self._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name=method_name)
        mf_cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name)
        holder_sentiment_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        turnover_purity_cost_opt_score = self._get_atomic_score(df, 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION', 0.0)
        floating_chip_cleansing_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name=method_name)
        total_loser_rate_raw = self._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name)
        # --- 3. 维度一：市场背景 (Market Context) ---
        retail_panic_score = self._normalize_series(retail_panic_raw, df_index, bipolar=False)
        # 直接奖励价格弱势
        mtf_price_weakness_score = self._get_mtf_slope_accel_score(df, f'close_D', mtf_slope_accel_weights, df_index, method_name, ascending=False, bipolar=False)
        low_volatility_score = self._normalize_series(bbw_raw, df_index, ascending=False)
        sentiment_pendulum_inverted_score = (1 - sentiment_pendulum_score.clip(lower=0)) # 情绪低迷
        tension_inverted_score = (1 - tension_score.clip(lower=0)) # 低张力
        market_sentiment_inverted_score = self._normalize_series(market_sentiment_raw, df_index, ascending=False)
        volatility_instability_inverted_score = self._normalize_series(volatility_instability_raw, df_index, ascending=False)
        market_context_score = (
            (retail_panic_score).pow(market_context_weights.get('retail_panic', 0.2)) *
            (mtf_price_weakness_score).pow(market_context_weights.get('price_weakness', 0.2)) * # 修正为直接奖励价格弱势
            (low_volatility_score).pow(market_context_weights.get('low_volatility', 0.2)) *
            (sentiment_pendulum_inverted_score).pow(market_context_weights.get('sentiment_pendulum_inverted', 0.15)) *
            (tension_inverted_score).pow(market_context_weights.get('tension_inverted', 0.1)) *
            (market_sentiment_inverted_score).pow(market_context_weights.get('market_sentiment_inverted', 0.1)) *
            (volatility_instability_inverted_score).pow(market_context_weights.get('volatility_instability_inverted', 0.05))
        ).pow(1/sum(market_context_weights.values())).fillna(0.0)
        # --- 4. 维度二：隐蔽行动 (Covert Action) ---
        mtf_suppressive_accum_score = self._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow_score = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 使用 deception_lure_long_intensity_D 的MTF融合版本作为欺诈证据
        mtf_deception_lure_long_score = self._get_mtf_slope_accel_score(df, 'deception_lure_long_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        stealth_ops_normalized = self._normalize_series(stealth_ops_score, df_index, bipolar=False)
        # 使用 hidden_accumulation_intensity_D 的MTF融合版本作为拆单吸筹证据
        mtf_hidden_accumulation_intensity = self._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        chip_historical_potential_normalized = self._normalize_series(chip_historical_potential_score.clip(lower=0), df_index, bipolar=False)
        mtf_mf_buy_ofi_normalized = self._get_mtf_slope_accel_score(df, 'main_force_buy_ofi_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_cost_advantage_normalized = self._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_flow_slope_normalized = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True).clip(lower=0) # 只取正向斜率
        mtf_suppressive_accum_slope_normalized = self._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True).clip(lower=0) # 只取正向斜率
        covert_action_score = (
            (mtf_suppressive_accum_score).pow(covert_action_weights.get('suppressive_accum', 0.15)) *
            (mtf_main_force_flow_score).pow(covert_action_weights.get('main_force_flow', 0.15)) *
            (mtf_deception_lure_long_score).pow(covert_action_weights.get('deception_lure_long', 0.15)) * # 修正为直接使用诱多欺诈
            (stealth_ops_normalized).pow(covert_action_weights.get('stealth_ops', 0.15)) *
            (mtf_hidden_accumulation_intensity).pow(covert_action_weights.get('hidden_accumulation_intensity', 0.1)) * # 修正为使用原始拆单吸筹的MTF
            (chip_historical_potential_normalized).pow(covert_action_weights.get('chip_historical_potential', 0.1)) *
            (mtf_mf_buy_ofi_normalized).pow(covert_action_weights.get('mf_buy_ofi', 0.05)) *
            (mtf_mf_cost_advantage_normalized).pow(covert_action_weights.get('mf_cost_advantage', 0.05)) *
            (mtf_mf_flow_slope_normalized).pow(covert_action_weights.get('mf_flow_slope', 0.05)) *
            (mtf_suppressive_accum_slope_normalized).pow(covert_action_weights.get('suppressive_accum_slope', 0.05))
        ).pow(1/sum(covert_action_weights.values())).fillna(0.0)
        # --- 5. 维度三：筹码优化 (Chip Optimization) ---
        mtf_chip_fatigue_score = self._get_mtf_slope_accel_score(df, 'chip_fatigue_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_pain_score = self._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        holder_sentiment_inverted_score = (1 - holder_sentiment_score).clip(0, 1)
        turnover_purity_cost_opt_normalized = self._normalize_series(turnover_purity_cost_opt_score.clip(lower=0), df_index, bipolar=False)
        mtf_floating_chip_cleansing_normalized = self._get_mtf_slope_accel_score(df, 'floating_chip_cleansing_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_loser_rate_normalized = self._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        chip_optimization_score = (
            (mtf_chip_fatigue_score).pow(chip_optimization_weights.get('chip_fatigue', 0.25)) *
            (mtf_loser_pain_score).pow(chip_optimization_weights.get('loser_pain', 0.25)) *
            (holder_sentiment_inverted_score).pow(chip_optimization_weights.get('holder_sentiment_inverted', 0.2)) *
            (turnover_purity_cost_opt_normalized).pow(chip_optimization_weights.get('turnover_purity_cost_opt', 0.15)) *
            (mtf_floating_chip_cleansing_normalized).pow(chip_optimization_weights.get('floating_chip_cleansing', 0.1)) *
            (mtf_total_loser_rate_normalized).pow(chip_optimization_weights.get('total_loser_rate', 0.05))
        ).pow(1/sum(chip_optimization_weights.values())).fillna(0.0)
        # --- 6. 最终合成：三维融合 ---
        covert_accumulation_score = (
            (market_context_score).pow(fusion_weights.get('market_context', 0.3)) *
            (covert_action_score).pow(fusion_weights.get('covert_action', 0.4)) *
            (chip_optimization_score).pow(fusion_weights.get('chip_optimization', 0.3))
        ).pow(1/(sum(fusion_weights.values()))).fillna(0.0)
        return covert_accumulation_score.clip(0, 1).astype(np.float32)






