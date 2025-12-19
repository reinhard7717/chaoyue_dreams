# 文件: strategies/trend_following/intelligence/process_intelligence.py
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
        self.params = get_params_block(self.strategy, 'process_intelligence_params', {})
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
        【V1.0 · 新增】计算多时间框架斜率和加速度的融合分数。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            base_signal_name (str): 基础信号的名称，例如 'BBW_21_2.0_D'。
            mtf_weights_config (Dict): 包含 'slope_periods' 和 'accel_periods' 权重的配置字典。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            ascending (bool): 归一化时是否升序。
            bipolar (bool): 归一化时是否使用双极性。
        返回:
            pd.Series: 融合后的MTF斜率和加速度分数。
        """
        slope_periods = get_param_value(mtf_weights_config.get('slope_periods'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        accel_periods = get_param_value(mtf_weights_config.get('accel_periods'), {"5": 0.6, "13": 0.4})
        all_scores = []
        total_weight = 0.0
        # 融合斜率
        for period_str, weight in slope_periods.items():
            period = int(period_str)
            slope_col = f'SLOPE_{period}_{base_signal_name}'
            slope_raw = self._get_safe_series(df, slope_col, np.nan, method_name=method_name)
            if slope_raw.isnull().all():
                continue
            score = self._normalize_series(slope_raw, df_index, bipolar=bipolar, ascending=ascending)
            all_scores.append(score * weight)
            total_weight += weight
        # 融合加速度
        for period_str, weight in accel_periods.items():
            period = int(period_str)
            accel_col = f'ACCEL_{period}_{base_signal_name}'
            accel_raw = self._get_safe_series(df, accel_col, np.nan, method_name=method_name)
            if accel_raw.isnull().all():
                continue
            score = self._normalize_series(accel_raw, df_index, bipolar=bipolar, ascending=ascending)
            all_scores.append(score * weight)
            total_weight += weight
        if not all_scores or total_weight == 0:
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        fused_score = sum(all_scores) / total_weight
        return fused_score.clip(0, 1) if not bipolar else fused_score.clip(-1, 1)

    def _normalize_series(self, series: pd.Series, target_index: pd.Index, bipolar: bool = False, ascending: bool = True) -> pd.Series: # 新增 ascending 参数
        """
        【V1.0 · 统一归一化引擎】
        - 核心职责: 为类内部提供一个统一的、基于多时间框架自适应归一化的方法。
        - 核心逻辑: 根据 bipolar 参数，调用 get_adaptive_mtf_normalized_score (单极) 或
                     get_adaptive_mtf_normalized_bipolar_score (双极) 进行归一化。
        - 修复: 解决了 'ProcessIntelligence' object has no attribute '_normalize_series' 的 AttributeError。
        """
        #  实现了统一的归一化逻辑
        # 获取MTF权重配置
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        if bipolar:
            return get_adaptive_mtf_normalized_bipolar_score(
                series=series,
                index=target_index, # 将 target_index 改为 index
                tf_weights=actual_mtf_weights, # 使用修正后的权重字典
                sensitivity=self.bipolar_sensitivity
            )
        else:
            return get_adaptive_mtf_normalized_score(
                series=series,
                index=target_index, # 将 target_index 改为 index
                ascending=ascending, # 传递 ascending 参数
                tf_weights=actual_mtf_weights # 使用修正后的权重字典
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

    def run_process_diagnostics(self, df: pd.DataFrame, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V5.6 · 帅帐中军版】过程情报分析总指挥
        - 核心升级: 建立“帅帐中军”调度机制。通过定义一个“基石信号”清单，按依赖顺序优先计算
                      如“权力转移”、“主力拉升意图”等被广泛依赖的核心信号，确保无论配置文件顺序如何，
                      依赖关系都能被满足，彻底解决因计算顺序错误导致的崩溃问题。
        """
        print("启动【V5.6 · 帅帐中军版】过程情报分析...")
        all_process_states = {}
        p_conf = get_params_block(self.strategy, 'process_intelligence_params', {})
        diagnostics = get_param_value(p_conf.get('diagnostics'), [])
        # [新增] 定义需要优先计算的“基石信号”清单
        priority_signals = [
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_MAIN_FORCE_RALLY_INTENT'
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
        print(f"【V5.6 · 帅帐中军版】分析完成，生成 {len(all_process_states)} 个过程元信号。")
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
        【V5.1 · 风险审判修正版】计算“主力拉升意图”的专属关系分数。
        - 核心升级: 引入“风险审判”机制。在评估拉升“动力”的基础上，额外构建一个由“顶部派发强度”、
                      “上影线抛压”等信号组成的“派发风险分”，并用其对原始拉升意图进行惩罚性调节。
                      旨在穿透上涨表象，精准区分“真突破”与“拉高出货的陷阱”。
        - 核心修正: 修正风险审判逻辑，确保“派发风险分”只惩罚看涨意图部分，而不会错误地削弱
                      已有的看跌意图信号，解决了负负得正的逻辑漏洞。
        - 新增功能: 植入“真理探针”，用于在指定日期输出风险调节前后的分数变化。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_RALLY_INTENT (V5.1 · 风险审判修正版)...")
        # 引入新的风险审判信号依赖
        required_signals = [
            'pct_change_D', 'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D',
            'upward_impulse_purity_D', 'volume_ratio_D', 'control_solidity_index_D',
            'main_force_cost_advantage_D', 'SLOPE_5_winner_concentration_90pct_D',
            'dominant_peak_solidity_D', 'active_buying_support_D', 'pressure_rejection_strength_D',
            'profit_realization_quality_D', 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH',
            'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 'distribution_at_peak_intensity_D',
            'upper_shadow_selling_pressure_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_main_force_rally_intent"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        relative_strength = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        capital_signature = self._get_atomic_score(df, 'SCORE_FF_AXIOM_CAPITAL_SIGNATURE', 0.0)
        cs_modulator_weight = config.get('capital_signature_modulator_weight', 0.0)
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_main_force_rally_intent")
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_main_force_rally_intent")
        price_impact_ratio = self._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name="_calculate_main_force_rally_intent")
        upward_impulse_purity = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_main_force_rally_intent")
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 1.0, method_name="_calculate_main_force_rally_intent")
        control_solidity = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name="_calculate_main_force_rally_intent")
        cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_main_force_rally_intent")
        concentration_slope = self._get_safe_series(df, f'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name="_calculate_main_force_rally_intent")
        peak_solidity = self._get_safe_series(df, 'dominant_peak_solidity_D', 0.0, method_name="_calculate_main_force_rally_intent")
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_main_force_rally_intent")
        pressure_rejection = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name="_calculate_main_force_rally_intent")
        profit_realization_quality = self._get_safe_series(df, 'profit_realization_quality_D', 0.5, method_name="_calculate_main_force_rally_intent")
        # 使用修正后的权重字典
        price_change_norm = get_adaptive_mtf_normalized_bipolar_score(price_change, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        net_flow_norm = get_adaptive_mtf_normalized_bipolar_score(main_force_net_flow, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        price_impact_norm = get_adaptive_mtf_normalized_bipolar_score(price_impact_ratio, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        impulse_purity_norm = get_adaptive_mtf_normalized_bipolar_score(upward_impulse_purity, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        volume_ratio_norm = get_adaptive_mtf_normalized_bipolar_score(volume_ratio - 1.0, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        control_solidity_norm = get_adaptive_mtf_normalized_bipolar_score(control_solidity, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        cost_advantage_norm = get_adaptive_mtf_normalized_bipolar_score(cost_advantage, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        concentration_slope_norm = get_adaptive_mtf_normalized_bipolar_score(concentration_slope, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        peak_solidity_norm = get_adaptive_mtf_normalized_bipolar_score(peak_solidity, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        buying_support_norm = get_adaptive_mtf_normalized_bipolar_score(active_buying_support, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        pressure_rejection_norm = get_adaptive_mtf_normalized_bipolar_score(pressure_rejection, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        profit_absorption_norm = get_adaptive_mtf_normalized_bipolar_score((1 - profit_realization_quality) - 0.5, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        aggressiveness_score = (
            price_change_norm.clip(lower=0) * 0.30 +
            net_flow_norm.clip(lower=0) * 0.30 +
            price_impact_norm.clip(lower=0) * 0.15 +
            impulse_purity_norm.clip(lower=0) * 0.15 +
            volume_ratio_norm.clip(lower=0) * 0.10
        ).clip(0, 1)
        control_score = (
            control_solidity_norm.clip(lower=0) * 0.35 +
            cost_advantage_norm.clip(lower=0) * 0.25 +
            concentration_slope_norm.clip(lower=0) * 0.20 +
            peak_solidity_norm.clip(lower=0) * 0.20
        ).clip(0, 1)
        obstacle_clearance_score = (
            buying_support_norm.clip(lower=0) * 0.40 +
            pressure_rejection_norm.clip(lower=0) * 0.30 +
            profit_absorption_norm.clip(lower=0) * 0.30
        ).clip(0, 1)
        bullish_intent = (aggressiveness_score * control_score * obstacle_clearance_score).pow(1/3)
        bearish_mask = (price_change_norm < 0) | (net_flow_norm < 0)
        bearish_score = (price_change_norm.clip(upper=0).abs() * 0.5 + net_flow_norm.clip(upper=0).abs() * 0.5).clip(0, 1) * -1
        base_rally_intent = bullish_intent.mask(bearish_mask, bearish_score)
        rs_modulator = (1 + relative_strength * rs_amplifier)
        capital_modulator = (1 + capital_signature * cs_modulator_weight)
        modulated_rally_intent = (base_rally_intent * rs_modulator * capital_modulator).clip(-1, 1)
        # 风险审判模块
        distribution_intensity = self._normalize_series(self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name="_calculate_main_force_rally_intent"), df_index, bipolar=False)
        upper_shadow_pressure = self._normalize_series(self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_calculate_main_force_rally_intent"), df_index, bipolar=False)
        distribution_risk_score = (distribution_intensity * 0.6 + upper_shadow_pressure * 0.4).clip(0, 1)
        # 修正风险审判逻辑，只惩罚看涨部分，避免削弱看跌信号
        bullish_part = modulated_rally_intent.clip(lower=0)
        bearish_part = modulated_rally_intent.clip(upper=0)
        penalized_bullish_part = bullish_part * (1 - distribution_risk_score)
        final_rally_intent = (penalized_bullish_part + bearish_part).clip(-1, 1)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day, (final_rally_intent + 0.35)).clip(-1, 1)
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score
        self.strategy.atomic_states["_DEBUG_rally_distribution_risk"] = distribution_risk_score # 新增调试信号
        return final_rally_intent.astype(np.float32)

    def _calculate_main_force_control_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0 · 控盘杠杆版】计算“主力控盘”的专属关系分数。
        - 核心重构: 创立“控盘即杠杆”模型。将“控盘度”作为调节“资金流向”影响力的核心杠杆。
                      最终分 = 主力净流入分 * (1 + 融合控盘分)。
        - 证据升级: 融合传统的均线控盘度与更现代的“控盘稳固度”，形成更立体的控盘评分。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_CONTROL (V2.0 · 控盘杠杆版)...")
        required_signals = ['close_D', 'main_force_net_flow_calibrated_D', 'control_solidity_index_D']
        if not self._validate_required_signals(df, required_signals, "_calculate_main_force_control_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 传统控盘度计算
        ema13 = ta.ema(close=self._get_safe_series(df, 'close_D', method_name="_calculate_main_force_control_relationship"), length=13, append=False)
        if ema13 is None: return pd.Series(0.0, index=df_index)
        varn1 = ta.ema(close=ema13, length=13, append=False)
        if varn1 is None: return pd.Series(0.0, index=df_index)
        prev_varn1 = varn1.shift(1).replace(0, np.nan)
        kongpan_raw = (varn1 - prev_varn1) / prev_varn1 * 1000
        # 结构控盘度
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name="_calculate_main_force_control_relationship")
        # 归一化
        traditional_control_score = self._normalize_series(kongpan_raw, df_index, bipolar=True)
        structural_control_score = self._normalize_series(control_solidity_raw, df_index, bipolar=True)
        # 融合控盘分
        fused_control_score = (traditional_control_score * 0.4 + structural_control_score * 0.6).clip(-1, 1)
        # 主力资金流
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_main_force_control_relationship")
        main_force_flow_score = self._normalize_series(main_force_net_flow, df_index, bipolar=True)
        # 核心逻辑：控盘杠杆模型
        control_leverage = 1 + fused_control_score.clip(lower=0) # 杠杆效应只在控盘为正时生效
        final_control_score = (main_force_flow_score * control_leverage).clip(-1, 1)
        return final_control_score.astype(np.float32)

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.15 · 诡道反击版】对“关系分”进行元分析，输出分数。
        - 核心升级: 新增对“隐蔽吸筹”信号的专属路由，执行其诊断逻辑。
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
            relationship_score = self._calculate_price_momentum_divergence(df, config)
            meta_score = relationship_score
        elif signal_name == 'PROCESS_META_STORM_EYE_CALM':
            meta_score = self._calculate_storm_eye_calm(df, config) # 直接传递 config
        elif signal_name == 'PROCESS_META_WASH_OUT_REBOUND':
            offensive_absorption_intent = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
            meta_score = self._calculate_process_wash_out_rebound(df, offensive_absorption_intent)
        elif signal_name == 'PROCESS_META_COVERT_ACCUMULATION': # 新增隐蔽吸筹信号路由
            meta_score = self._calculate_process_covert_accumulation(df)
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
        【V2.4 · 探针可控版】分裂型元关系诊断器
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
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        bipolar_displacement_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_displacement,
            index=df.index, # 将 target_index 改为 index
            tf_weights=actual_mtf_weights, # 使用修正后的权重字典
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_momentum_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_momentum,
            index=df.index, # 将 target_index 改为 index
            tf_weights=actual_mtf_weights, # 使用修正后的权重字典
            sensitivity=self.bipolar_sensitivity
        )
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
        【V1.4 · 信念侵蚀版】信号衰减诊断器
        - 核心升级: 为“赢家信念衰减”信号分派专属计算引擎，执行全新的“信念侵蚀”逻辑。
        """
        signal_name = config.get('name')
        # [新增] 为“赢家信念衰减”信号增加专属路由
        if signal_name == 'PROCESS_META_WINNER_CONVICTION_DECAY':
            decay_score = self._calculate_winner_belief_erosion(df, config)
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
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        decay_score = get_adaptive_mtf_normalized_score(decay_magnitude, df_index, actual_mtf_weights, ascending=True)
        return {signal_name: decay_score.astype(np.float32)}

    def _calculate_price_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1 · 阴阳易位版】“价势背离”专属关系计算引擎
        - 核心重构: 创立“方向对抗”模型，回归背离本质。
        - 核心修正: 将核心公式修正为“价格方向分 - 动能方向分”，使顶背离（风险）输出正分。
        - 核心逻辑: 瞬时关系分 = 价格方向分(归一化) - 动能方向分(归一化)。
        """
        price_slope_signal = 'SLOPE_5_close_D'
        momentum_slope_signal = 'SLOPE_5_MACDh_13_34_8_D'
        required_signals = [price_slope_signal, momentum_slope_signal]
        if not self._validate_required_signals(df, required_signals, "_calculate_price_momentum_divergence"):
            return pd.Series(dtype=np.float32)
        df_index = df.index
        price_slope_raw = self._get_safe_series(df, price_slope_signal, 0.0, method_name="_calculate_price_momentum_divergence")
        momentum_slope_raw = self._get_safe_series(df, momentum_slope_signal, 0.0, method_name="_calculate_price_momentum_divergence")
        # 归一化当前方向（斜率值）
        price_direction_score = self._normalize_series(price_slope_raw, df_index, bipolar=True)
        momentum_direction_score = self._normalize_series(momentum_slope_raw, df_index, bipolar=True)
        # 核心逻辑：方向对抗模型 (阴阳易位修正)
        # 顶背离: (正的价格分) - (负的动能分) = 显著正分 (风险)
        # 底背离: (负的价格分) - (正的动能分) = 显著负分 (机会)
        relationship_score = (price_direction_score - momentum_direction_score).clip(-1, 1)
        return relationship_score

    def _calculate_winner_belief_erosion(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0 · 信念侵蚀版】“赢家信念衰减”专属计算引擎
        - 核心重构: 创立“压力放大”模型，审判信念衰减与派发压力的共振。
        - 信号升级: 引入 `profit_taking_flow_ratio_D` 作为核心压力信号。
        - 核心逻辑: 侵蚀分 = 基础衰减分 * (1 + 派发压力分)。
        """
        belief_signal_name = 'winner_stability_index_D'
        pressure_signal_name = 'profit_taking_flow_ratio_D'
        required_signals = [belief_signal_name, pressure_signal_name]
        if not self._validate_required_signals(df, required_signals, "_calculate_winner_belief_erosion"):
            return pd.Series(dtype=np.float32)
        df_index = df.index
        belief_signal_raw = self._get_safe_series(df, belief_signal_name, 0.0, method_name="_calculate_winner_belief_erosion")
        pressure_signal_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name="_calculate_winner_belief_erosion")
        # 1. 计算基础衰减分
        belief_change = belief_signal_raw.diff(1).fillna(0)
        decay_magnitude = belief_change.clip(upper=0).abs()
        base_decay_score = self._normalize_series(decay_magnitude, df_index, bipolar=False)
        # 2. 计算派发压力分
        pressure_score = self._normalize_series(pressure_signal_raw, df_index, bipolar=False)
        # 3. 核心逻辑：压力放大模型
        pressure_amplifier = 1 + pressure_score
        erosion_score = (base_decay_score * pressure_amplifier).clip(0, 1)
        return erosion_score

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
        【V5.1 · 全息融合版】计算“隐蔽吸筹”的专属关系分数。
        - 核心升级: 将“横盘”与“温和推升”场景的证据融合方式，从“几何平均”升级为“全息证据加权融合”，
                      允许核心证据的强势弥补次要证据的不足，以勘破“明修栈道，暗度陈仓”的诡道。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_STEALTH_ACCUMULATION (V5.1 · 全息融合版)...")
        df_index = df.index
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        historical_potential = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        potential_gate = config.get('historical_potential_gate', 0.0)
        potential_amplifier = config.get('historical_potential_amplifier', 0.0)
        price_trend_raw = self._get_safe_series(df, f'SLOPE_5_close_D', 0.0, method_name="_calculate_stealth_accumulation")
        stability_score = self.strategy.atomic_states.get('SCORE_DYN_AXIOM_STABILITY', pd.Series(0.0, index=df_index))
        volume_atrophy_score = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_ATROPHY', pd.Series(0.0, index=df_index))
        concentration_trend_raw = self._get_safe_series(df, f'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name="_calculate_stealth_accumulation")
        peak_solidity_trend_raw = self._get_safe_series(df, f'SLOPE_5_dominant_peak_solidity_D', 0.0, method_name="_calculate_stealth_accumulation")
        cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_stealth_accumulation")
        power_transfer_score = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        split_order_accumulation_score = self.strategy.atomic_states.get('PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', pd.Series(0.0, index=df_index))
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_stealth_accumulation")
        upward_purity = self._normalize_series(upward_purity_raw, df_index, bipolar=False)
        # 使用修正后的权重字典
        price_trend_norm = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        concentration_trend_norm = get_adaptive_mtf_normalized_bipolar_score(concentration_trend_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        peak_solidity_trend_norm = get_adaptive_mtf_normalized_bipolar_score(peak_solidity_trend_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        cost_advantage_norm = get_adaptive_mtf_normalized_bipolar_score(cost_advantage_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        suppressive_mask = price_trend_norm <= 0.1
        evidence1_suppressive = volume_atrophy_score.clip(lower=0)
        evidence2_suppressive = power_transfer_score.clip(lower=0)
        evidence3_suppressive = concentration_trend_norm.clip(lower=0)
        evidence4_suppressive = cost_advantage_norm.clip(lower=0)
        suppressive_score = (evidence1_suppressive * evidence2_suppressive * evidence3_suppressive * evidence4_suppressive).pow(1/4)
        suppressive_score = suppressive_score.where(suppressive_mask, 0.0)
        consolidative_mask = stability_score > 0.2
        evidence1_consolidative = volume_atrophy_score.clip(lower=0)
        evidence2_consolidative = power_transfer_score.clip(lower=0)
        evidence3_consolidative = peak_solidity_trend_norm.clip(lower=0)
        evidence4_consolidative = split_order_accumulation_score.clip(lower=0)
        consolidative_score = (
            evidence1_consolidative * 0.2 +
            evidence2_consolidative * 0.2 +
            evidence3_consolidative * 0.2 +
            evidence4_consolidative * 0.4
        )
        consolidative_score = consolidative_score.where(consolidative_mask, 0.0)
        gentle_push_mask = (price_trend_norm > 0.1) & (price_trend_norm < 0.5)
        evidence1_gentle = upward_purity.clip(lower=0)
        evidence2_gentle = power_transfer_score.clip(lower=0)
        evidence3_gentle = split_order_accumulation_score.clip(lower=0)
        gentle_push_score = (
            evidence3_gentle * 0.5 +
            evidence1_gentle * 0.3 +
            evidence2_gentle * 0.2
        )
        gentle_push_score = gentle_push_score.where(gentle_push_mask, 0.0)
        base_score = pd.concat([suppressive_score, consolidative_score, gentle_push_score], axis=1).max(axis=1).fillna(0.0)
        potential_gate_mask = historical_potential > potential_gate
        potential_modulator = (1 + historical_potential * potential_amplifier)
        final_score = (base_score * potential_modulator).where(potential_gate_mask, 0.0)
        self.strategy.atomic_states["_DEBUG_accum_suppressive_score"] = suppressive_score
        self.strategy.atomic_states["_DEBUG_accum_consolidative_score"] = consolidative_score
        self.strategy.atomic_states["_DEBUG_accum_gentle_push_score"] = gentle_push_score
        return final_score.clip(0, 1).astype(np.float32)

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.4 · 战果审判版】计算“恐慌洗盘吸筹”的专属信号。
        - 核心升级: 创立“战果优先”原则。在基础分之上，额外引入由“修复分”驱动的“战果调节器”，
                      对最终得分进行二次审判与加权，旨在奖赏战役结果优异（筹码结构优化）的吸筹行为。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_PANIC_WASHOUT_ACCUMULATION (V4.4 · 战果审判版)...")
        df_index = df.index
        historical_potential = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        potential_gate = config.get('historical_potential_gate', 0.0)
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        lower_shadow_strength = self.strategy.atomic_states.get('SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', pd.Series(0.0, index=df_index))
        volume_burst = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_BURST', pd.Series(0.0, index=df_index))
        retail_panic_index = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        loser_pain_index = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        power_transfer_score = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        concentration_slope = self._get_safe_series(df, f'SLOPE_1_winner_concentration_90pct_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        structural_leverage_raw = self._get_safe_series(df, 'structural_leverage_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        cost_advantage_slope = main_force_cost_advantage.diff(1).fillna(0)
        # 使用修正后的权重字典
        retail_panic_norm = get_adaptive_mtf_normalized_score(retail_panic_index, df_index, actual_mtf_weights, ascending=True)
        loser_pain_norm = get_adaptive_mtf_normalized_score(loser_pain_index, df_index, actual_mtf_weights, ascending=True)
        active_buying_support_norm = get_adaptive_mtf_normalized_score(active_buying_support, df_index, actual_mtf_weights, ascending=True)
        concentration_slope_norm = get_adaptive_mtf_normalized_score(concentration_slope, df_index, actual_mtf_weights, ascending=True)
        cost_advantage_slope_norm = get_adaptive_mtf_normalized_score(cost_advantage_slope, df_index, actual_mtf_weights, ascending=True)
        structural_leverage_norm = get_adaptive_mtf_normalized_score(structural_leverage_raw, df_index, actual_mtf_weights, ascending=True)
        panic_score_instant = (retail_panic_norm * 0.7 + loser_pain_norm * 0.3).clip(0, 1)
        absorption_score_instant = (power_transfer_score.clip(lower=0) * 0.5 + lower_shadow_strength * 0.25 + active_buying_support_norm * 0.25).clip(0, 1)
        panic_score = panic_score_instant.rolling(3, min_periods=1).mean()
        absorption_score = absorption_score_instant.rolling(3, min_periods=1).mean()
        original_repair_score = (concentration_slope_norm * 0.6 + cost_advantage_slope_norm * 0.4).clip(0, 1)
        repair_score = (original_repair_score * 0.5 + structural_leverage_norm * 0.5).clip(0, 1)
        is_significant_drop_daily = (pct_change < -0.03) | (lower_shadow_strength > 0.6)
        is_significant_drop_cumulative = (close_price.pct_change(3).fillna(0) < -0.07)
        is_blitz_washout = (is_significant_drop_daily | is_significant_drop_cumulative) & (volume_burst > 0.5)
        is_high_panic = panic_score > 0.4
        is_high_absorption = absorption_score > 0.15
        is_price_stalemate = close_price.pct_change(3).fillna(0).abs() < 0.05
        is_protracted_washout = is_high_panic & is_high_absorption & is_price_stalemate
        pct_change_3d = close_price.pct_change(3).fillna(0)
        is_moderate_rise = (pct_change_3d > 0) & (pct_change_3d < 0.10)
        is_mid_air_refueling = is_high_panic & is_high_absorption & is_moderate_rise
        washout_candidate_mask = is_blitz_washout | is_protracted_washout | is_mid_air_refueling
        base_score = (panic_score * absorption_score * repair_score).pow(1/3)
        battle_outcome_modulator = 1 + repair_score
        judged_base_score = (base_score * battle_outcome_modulator).clip(0, 1)
        potential_gate_mask = historical_potential > potential_gate
        final_score = judged_base_score.where(washout_candidate_mask & potential_gate_mask, 0.0).fillna(0.0)
        self.strategy.atomic_states["_DEBUG_washout_panic_score"] = panic_score
        self.strategy.atomic_states["_DEBUG_washout_absorption_score"] = absorption_score
        self.strategy.atomic_states["_DEBUG_washout_repair_score"] = repair_score
        self.strategy.atomic_states["_DEBUG_washout_judged_base_score"] = judged_base_score
        return final_score.astype(np.float32)

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 矛盾博弈版】计算“诡道吸筹”信号。
        - 核心重构: 创立“矛盾即证据”原则。通过构建“伪装分”来奖励“拆单吸筹”与“权力流出”
                      同时发生的矛盾行为，旨在勘破“明修栈道，暗度陈仓”的终极诡道。
        - 证据升级: 以“拆单吸筹强度”为核心行动，以“伪装分”和“欺诈指数”共同构建“诡道氛围”。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_DECEPTIVE_ACCUMULATION (V3.3 · 矛盾博弈版)...")
        required_signals = [
            'hidden_accumulation_intensity_D', 'deception_index_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_COHERENT_DRIVE', 'SLOPE_5_close_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_deceptive_accumulation"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        split_order_accum_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_deceptive_accumulation")
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_calculate_deceptive_accumulation")
        power_transfer_score = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        coherent_drive_score = self._get_atomic_score(df, 'SCORE_CHIP_COHERENT_DRIVE', 0.0)
        price_trend_raw = self._get_safe_series(df, f'SLOPE_5_close_D', 0.0, method_name="_calculate_deceptive_accumulation")
        core_action_score = self._normalize_series(split_order_accum_raw, df_index, bipolar=False)
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 使用修正后的权重字典
        deception_evidence = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity).clip(lower=0)
        # 创立“伪装分”，奖励权力转移为负的矛盾行为
        disguise_score = (1 - power_transfer_score) / 2
        # 重构“诡道氛围”，融合“伪装分”与“欺诈证据”
        deceptive_context_score = (disguise_score * 0.6 + deception_evidence * 0.4).clip(0, 1)
        # 使用修正后的权重字典
        price_trend_norm = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        price_gating_score = (1 - price_trend_norm.clip(lower=0.1)).clip(0, 1)
        coherence_penalty_factor = (1 - coherent_drive_score.clip(upper=0).abs()).clip(0, 1)
        final_score = (core_action_score * deceptive_context_score * coherence_penalty_factor * price_gating_score).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_upthrust_washout(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0 · 强证优先版】识别主力利用“上冲回落”阴线进行的洗盘行为。
        - 核心重构: 创立“强证优先”原则。废除对多种承接证据的加权平均，改为采用 max() 函数，
                      直接取“主动买盘”、“下影线强度”、“权力转移”三者中的最强者作为最终承接证据，
                      旨在识别任何一种足以扭转战局的决定性吸收力量。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_UPTHRUST_WASHOUT (V2.0 · 强证优先版)...")
        required_signals = [
            'SCORE_STRUCT_AXIOM_TREND_FORM', 'BIAS_21_D', 'pct_change_D',
            'upward_impulse_purity_D', 'upper_shadow_selling_pressure_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 'active_buying_support_D',
            'PROCESS_META_POWER_TRANSFER'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_upthrust_washout"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        bias_21 = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name="_calculate_upthrust_washout")
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_upthrust_washout")
        upward_purity_norm = self._normalize_series(upward_purity_raw, df_index, bipolar=False)
        context_mask = (trend_form_score > 0.2) & (bias_21 < 0.2) & (upward_purity_norm.rolling(3).mean() > 0.3)
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_upthrust_washout")
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_calculate_upthrust_washout")
        lower_shadow_strength = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        active_buying_raw = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_upthrust_washout")
        power_transfer = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        is_down_day = (pct_change < 0).astype(float)
        upper_shadow_pressure_norm = self._normalize_series(upper_shadow_pressure_raw, df_index, bipolar=False)
        selling_pressure_score = (upper_shadow_pressure_norm * 0.7 + is_down_day * 0.3).clip(0, 1)
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        power_transfer_norm = power_transfer.clip(lower=0)
        # 重构承接审判分，采用“强证优先”原则
        absorption_rebuttal_score = pd.concat([
            active_buying_norm,
            lower_shadow_strength,
            power_transfer_norm
        ], axis=1).max(axis=1)
        net_washout_intent = (absorption_rebuttal_score - selling_pressure_score).clip(0, 1)
        final_score = net_washout_intent.where(context_mask, 0.0).fillna(0.0)
        # [删除] 移除所有探针及调试信号存储代码
        return final_score.astype(np.float32)

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.3 · 势能衰减版】识别多日累积吸筹后，即将由“量变”引发“质变”的拉升拐点。
        - 核心升级: 引入“势能衰减”机制。将累积势能的计算方法从简单的滚动求和(rolling.sum)
                      升级为指数加权移动平均(ewm.mean)，赋予近期吸筹行为更高的权重，
                      更精准地度量具备时效性的“爆发势能”。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_ACCUMULATION_INFLECTION (V2.3 · 势能衰减版)...")
        required_signals = [
            'PROCESS_META_STEALTH_ACCUMULATION', 'PROCESS_META_DECEPTIVE_ACCUMULATION',
            'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY',
            'PROCESS_META_POWER_TRANSFER', 'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_PD_DIVERGENCE_CONFIRM'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_accumulation_inflection"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        accumulation_window = config.get('accumulation_window', 21)
        stealth_accum = self._get_atomic_score(df, 'PROCESS_META_STEALTH_ACCUMULATION', 0.0)
        deceptive_accum = self._get_atomic_score(df, 'PROCESS_META_DECEPTIVE_ACCUMULATION', 0.0)
        panic_washout_accum = self._get_atomic_score(df, 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 0.0)
        split_order_accum = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0)
        power_transfer_accum = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0)
        daily_accumulation_strength = pd.concat([stealth_accum, deceptive_accum, panic_washout_accum, split_order_accum, power_transfer_accum], axis=1).max(axis=1)
        # 采用指数加权移动平均(ewm)计算势能，引入时间衰减
        potential_energy_raw = daily_accumulation_strength.ewm(span=accumulation_window, adjust=False, min_periods=5).mean()
        potential_energy_score = normalize_score(potential_energy_raw, df_index, window=accumulation_window, ascending=True).clip(0, 1)
        rally_intent = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0)
        divergence_confirm = self._get_atomic_score(df, 'PROCESS_META_PD_DIVERGENCE_CONFIRM', 0.0)
        ignition_intent_score = (
            rally_intent.clip(lower=0) * 0.6 +
            divergence_confirm.clip(lower=0) * 0.4
        ).clip(0, 1)
        final_score = (potential_energy_score * ignition_intent_score).fillna(0.0)
        # [删除] 移除所有探针及调试信号存储代码
        return final_score.astype(np.float32)

    def _calculate_winner_conviction_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.2 · 状态对抗版】“赢家信念”专属关系计算引擎
        - 核心重构: 从“动量背离”升维至“状态对抗”。不再比较信号的变化率，而是直接对比
                      “赢家稳定性”和“利润兑现压力”的绝对强度状态，更直观地反映信念与压力的对抗格局。
        - 旨在解决因过度关注二阶“动量”而错失一阶“状态”强对抗信号的问题。
        """
        signal_a_name = 'profit_taking_flow_ratio_D'  # 压力方
        signal_b_name = 'winner_stability_index_D'    # 信念方
        config['signal_A'] = signal_a_name
        config['signal_B'] = signal_b_name
        df_index = df.index
        def get_signal_series(signal_name: str) -> Optional[pd.Series]:
            return self._get_safe_series(df, signal_name, method_name="_calculate_winner_conviction_relationship")
        pressure_signal_raw = get_signal_series(signal_a_name)
        conviction_signal_raw = get_signal_series(signal_b_name)
        if pressure_signal_raw is None or conviction_signal_raw is None:
            print(f"        -> [赢家信念] 警告: 缺少核心信号 '{signal_a_name}' 或 '{signal_b_name}'。")
            return pd.Series(dtype=np.float32)
        # 从计算动量改为计算状态分
        pressure_state_score = self._normalize_series(pressure_signal_raw, df_index, bipolar=False)
        conviction_state_score = self._normalize_series(conviction_signal_raw, df_index, bipolar=False)
        k = config.get('signal_b_factor_k', 1.0)
        # 核心逻辑变为状态对抗：信念状态分 - 压力状态分
        relationship_score = (k * conviction_state_score - pressure_state_score) / (k + 1)
        return relationship_score.clip(-1, 1)

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
        【V4.1 · 象限审判版】计算成本优势趋势。
        - 核心升级: 为各象限的确认环节配备专属的、最高保真度的战术信号，实现“精准审判”。
                      - Q2(派发下跌)引入“利润兑现流量”作为核心证据。
                      - Q4(牛市陷阱)引入“买盘虚弱度”作为核心惩罚项。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_COST_ADVANTAGE_TREND (V4.1 · 象限审判版)...")
        required_signals = [
            'pct_change_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'upward_impulse_purity_D', 'suppressive_accumulation_intensity_D',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 'distribution_at_peak_intensity_D',
            'active_selling_pressure_D', 'profit_taking_flow_ratio_D', 'active_buying_support_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_cost_advantage_trend_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        price_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship")
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship")
        P_change = self._normalize_series(price_change, df_index, bipolar=True)
        CA_change = self._normalize_series(main_force_cost_advantage.diff(1).fillna(0), df_index, bipolar=True)
        main_force_conviction = self._normalize_series(self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=True)
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        suppressive_accum = self._normalize_series(self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        lower_shadow_absorb = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        distribution_intensity = self._normalize_series(self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        active_selling = self._normalize_series(self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        profit_taking_flow = self._normalize_series(self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        active_buying_support = self._normalize_series(self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        # Q1: 价涨 & 优扩 (健康上涨)
        Q1_base = (P_change.clip(lower=0) * CA_change.clip(lower=0)).pow(0.5)
        Q1_confirm = (main_force_conviction.clip(lower=0) * upward_purity).pow(0.5)
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        # Q2: 价跌 & 优缩 (派发下跌)
        Q2_base = (P_change.clip(upper=0).abs() * CA_change.clip(upper=0).abs()).pow(0.5)
        Q2_distribution_evidence = (profit_taking_flow * 0.6 + active_selling * 0.4).clip(0, 1)
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        # Q3: 价跌 & 优扩 (黄金坑)
        Q3_base = (P_change.clip(upper=0).abs() * CA_change.clip(lower=0)).pow(0.5)
        Q3_confirm = (suppressive_accum * lower_shadow_absorb).pow(0.5)
        Q3_final = (Q3_base * Q3_confirm).clip(0, 1)
        # Q4: 价涨 & 优缩 (牛市陷阱)
        Q4_base = (P_change.clip(lower=0) * CA_change.clip(upper=0).abs()).pow(0.5)
        Q4_trap_evidence = (distribution_intensity * (1 - active_buying_support)).clip(0, 1)
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        final_score = (Q1_final + Q2_final + Q3_final + Q4_final).clip(-1, 1)
        return final_score.astype(np.float32)

    def _calculate_split_order_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 质效校准版】计算“拆单吸筹强度”的专属信号。
        - 核心升级: 引入“效率基准线”(efficiency_baseline)概念。在计算“质效调节指数”前，
                      先对“全息验证综合分”进行校准。这使得任何低于基准线的战果（即使为正）
                      都会被视为负向贡献，从而受到惩罚性抑制，为模型注入了赏罚分明的“主帅”逻辑。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY (V3.1 · 质效校准版)...")
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
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 使用修正后的权重字典
        price_trend_norm = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, actual_mtf_weights, self.bipolar_sensitivity)
        price_suppression_factor = (1 - price_trend_norm.clip(lower=0) * (1 - upward_purity)).clip(0, 1)
        # 使用修正后的权重字典
        deception_norm = get_adaptive_mtf_normalized_bipolar_score(deception_index, df_index, actual_mtf_weights, self.bipolar_sensitivity)
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
        # 使用修正后的权重字典
        flow_trend = get_adaptive_mtf_normalized_bipolar_score(flow_outcome.diff(3).fillna(0), df_index, actual_mtf_weights, self.bipolar_sensitivity)
        # 使用修正后的权重字典
        structure_trend = get_adaptive_mtf_normalized_bipolar_score(structure_outcome.diff(3).fillna(0), df_index, actual_mtf_weights, self.bipolar_sensitivity)
        # 使用修正后的权重字典
        potential_trend = get_adaptive_mtf_normalized_bipolar_score(potential_outcome.diff(3).fillna(0), df_index, actual_mtf_weights, self.bipolar_sensitivity)
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
        # MODIFIED: 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # MODIFIED: 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # MODIFIED: 移除 bipolar=False 参数
        wash_trade_penalty = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, actual_mtf_weights)
        volume_atrophy_quality = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        # MODIFIED: 移除 bipolar=False 参数
        suppressive_accum = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, actual_mtf_weights)
        # MODIFIED: 移除 bipolar=False 参数
        panic_evidence = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, actual_mtf_weights)
        # MODIFIED: 移除 bipolar=False 参数
        upward_purity = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, actual_mtf_weights)
        reversal_confirmation_shape = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        reversal_confirmation_flow = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        # MODIFIED: 使用修正后的权重字典
        reversal_confirmation_psyche = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction.diff(1).fillna(0), df_index, actual_mtf_weights, self.bipolar_sensitivity)
        # MODIFIED: 移除 bipolar=False 参数
        active_buying_confirm = get_adaptive_mtf_normalized_score(self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, actual_mtf_weights)
        # MODIFIED: 移除 bipolar=False 参数
        accel_shape = get_adaptive_mtf_normalized_score(reversal_confirmation_shape.diff(2).fillna(0), df_index, actual_mtf_weights)
        # MODIFIED: 移除 bipolar=False 参数
        accel_flow = get_adaptive_mtf_normalized_score(reversal_confirmation_flow.diff(2).fillna(0), df_index, actual_mtf_weights)
        # MODIFIED: 移除 bipolar=False 参数
        accel_psyche = get_adaptive_mtf_normalized_score(reversal_confirmation_psyche.diff(2).fillna(0), df_index, actual_mtf_weights)
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
        # MODIFIED: 使用修正后的权重字典
        p_mom = get_adaptive_mtf_normalized_bipolar_score(price.pct_change().fillna(0), df_index, actual_mtf_weights, self.bipolar_sensitivity)
        # MODIFIED: 使用修正后的权重字典
        v_mom = get_adaptive_mtf_normalized_bipolar_score(volume.pct_change().fillna(0), df_index, actual_mtf_weights, self.bipolar_sensitivity)
        final_score = pd.Series(0.0, index=df_index)
        # MODIFIED: 使用修正后的权重字典
        quality_factor = (get_adaptive_mtf_normalized_bipolar_score(main_force_conviction, df_index, actual_mtf_weights, self.bipolar_sensitivity).clip(lower=0) * (1 - wash_trade_penalty)).pow(0.5)
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
        【V3.0 · 共振审判版】诊断“突破加速抢筹”战术。
        - 核心重构: 创立“突破即共振”模型。废除硬阈值门槛和几何平均，改为对四大核心证据
                      （突破、意图、资金、结构）进行加权融合，以更鲁棒的“共振分”审判突破质量。
        - 信号修正: 修正了对“拉升意图”信号的引用，确保使用最终权威信号。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_BREAKOUT_ACCELERATION (V3.0 · 共振审判版)...")
        required_signals = [
            'SCORE_PATTERN_AXIOM_BREAKOUT', 'PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_POWER_TRANSFER', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_breakout_acceleration"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        relative_strength = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        # 定义四大核心证据
        breakout_evidence = self._get_atomic_score(df, 'SCORE_PATTERN_AXIOM_BREAKOUT', 0.0)
        intent_evidence = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_RALLY_INTENT', 0.0).clip(lower=0)
        flow_evidence = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0).clip(lower=0)
        structure_evidence = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0).clip(lower=0)
        # 废除硬门槛，改为加权共振模型
        weights = {'breakout': 0.4, 'intent': 0.3, 'structure': 0.2, 'flow': 0.1}
        resonance_score = (
            breakout_evidence * weights['breakout'] +
            intent_evidence * weights['intent'] +
            structure_evidence * weights['structure'] +
            flow_evidence * weights['flow']
        ).clip(0, 1)
        # 相对强度调节器
        rs_modulator = (1 + relative_strength.clip(lower=0) * rs_amplifier)
        final_score = (resonance_score * rs_modulator).clip(0, 1).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.0 · 战术升级版】识别主力从隐蔽吸筹转向公开强攻的转折信号。
        - 核心重构: 废除僵化的“AND”门槛，创立“战术评分”模型。最终分 = 前奏吸筹分 * 强攻分。
        - 证据升级: “前奏分”通过归一化消除尺度问题；“强攻分”对核心证据进行加权，更具实战性。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT (V2.0 · 战术升级版)...")
        required_signals = [
            'hidden_accumulation_intensity_D', 'main_force_net_flow_calibrated_D',
            'buy_quote_exhaustion_rate_D', 'large_order_pressure_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_fund_flow_accumulation_inflection"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        # 获取原料
        prelude_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        main_force_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        buy_exhaustion_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        large_pressure_raw = self._get_safe_series(df, 'large_order_pressure_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        # 1. 重铸“前奏分”，消除尺度问题
        prelude_score = self._normalize_series(prelude_raw.rolling(5).mean(), df_index, bipolar=False)
        # 2. 重铸“强攻分”，采用加权模型
        buy_exhaustion_norm = self._normalize_series(buy_exhaustion_raw, df_index, bipolar=False)
        main_force_flow_momentum = self._normalize_series(main_force_flow_raw.diff(1).fillna(0), df_index, bipolar=True)
        pressure_clearance_norm = 1 - self._normalize_series(large_pressure_raw, df_index, bipolar=False)
        attack_score = (
            buy_exhaustion_norm * 0.6 +
            main_force_flow_momentum.clip(lower=0) * 0.3 +
            pressure_clearance_norm.clip(lower=0) * 0.1
        ).clip(0, 1)
        # 3. 最终审判
        final_score = (prelude_score * attack_score).fillna(0.0)
        return final_score.astype(np.float32)

    def _calculate_profit_vs_flow_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.1 · 战场态势版】“利润与流向”专属关系计算引擎
        - 核心重构: 创立“战场态势”审判模型，从比较“动量”升维为比较力量的“当前水平”。
        - 信号升级: 将核心“压力”信号从“T0效率”升级为更精准的“利润兑现流量占比”。
        - 核心逻辑: 瞬时关系分 = 建仓动力分(归一化) - 派发压力分(归一化)。
        """
        pressure_signal_name = 'profit_taking_flow_ratio_D'    # 压力方
        drive_signal_name = 'main_force_net_flow_calibrated_D' # 动力方
        required_signals = [pressure_signal_name, drive_signal_name]
        if not self._validate_required_signals(df, required_signals, "_calculate_profit_vs_flow_relationship"):
            return pd.Series(dtype=np.float32)
        df_index = df.index
        pressure_signal_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name="_calculate_profit_vs_flow_relationship")
        drive_signal_raw = self._get_safe_series(df, drive_signal_name, 0.0, method_name="_calculate_profit_vs_flow_relationship")
        # 归一化信号的当前值，代表战场态势
        pressure_score = self._normalize_series(pressure_signal_raw, df_index, bipolar=False) # 压力是无方向的，只看大小
        drive_score = self._normalize_series(drive_signal_raw, df_index, bipolar=True)      # 动力有正负方向
        # 核心逻辑：战场态势对抗
        relationship_score = drive_score - pressure_score
        final_score = relationship_score.clip(-1, 1)
        return final_score

    def _calculate_stock_sector_sync(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.0 · 协同共振版】“个股板块协同共振”专属关系计算引擎
        - 核心重构: 创立“协同共振模型”，明确区分看涨和看跌协同，并融入板块动量。
        - 信号升级: 个股强度由 `pct_change_D` 直接衡量，板块强度由 `industry_strength_rank_D` 衡量，
                      并新增 `SLOPE_5_industry_strength_rank_D` 捕捉板块动量。
        - 核心逻辑: 融合个股表现、板块强度和板块动量，通过非线性放大机制，量化看涨和看跌的协同共振。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_STOCK_SECTOR_SYNC (V3.0 · 协同共振版)...")
        stock_signal_name = 'pct_change_D'
        sector_rank_name = 'industry_strength_rank_D'
        sector_momentum_name = 'SLOPE_5_industry_strength_rank_D' # 新增依赖
        required_signals = [stock_signal_name, sector_rank_name, sector_momentum_name]
        if not self._validate_required_signals(df, required_signals, "_calculate_stock_sector_sync"):
            print(f"    -> [过程情报警告] _calculate_stock_sector_sync 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 获取原始数据
        stock_signal_raw = self._get_safe_series(df, stock_signal_name, 0.0, method_name="_calculate_stock_sector_sync")
        sector_rank_raw = self._get_safe_series(df, sector_rank_name, 0.0, method_name="_calculate_stock_sector_sync")
        sector_momentum_raw = self._get_safe_series(df, sector_momentum_name, 0.0, method_name="_calculate_stock_sector_sync")
        # 归一化当前状态和动量
        stock_strength_score = self._normalize_series(stock_signal_raw, df_index, bipolar=True)
        sector_rank_score = self._normalize_series(sector_rank_raw, df_index, bipolar=False) # 板块排名是[0,1]的单极信号
        sector_momentum_score = self._normalize_series(sector_momentum_raw, df_index, bipolar=True) # 板块动量是双极信号
        # --- 看涨协同部分 (Bullish Synchronicity) ---
        # 1. 提取个股正向运动
        bullish_stock_movement = stock_strength_score.clip(lower=0)
        # 2. 提取板块正向强度和正向动量
        sector_strength_context_bullish = sector_rank_score
        sector_momentum_context_bullish = sector_momentum_score.clip(lower=0)
        # 3. 计算板块看涨共振因子 (几何平均，要求两者都为正才高)
        bullish_sector_resonance = (sector_strength_context_bullish * sector_momentum_context_bullish).pow(0.5).fillna(0.0)
        # 4. 计算看涨放大因子
        bullish_amplification_factor = 1 + bullish_sector_resonance
        # 5. 最终看涨协同分数
        final_bullish_score = bullish_stock_movement * bullish_amplification_factor
        # --- 看跌协同部分 (Bearish Synchronicity) ---
        # 1. 提取个股负向运动的绝对值
        bearish_stock_movement = stock_strength_score.clip(upper=0).abs()
        # 2. 提取板块负向强度和负向动量
        sector_weakness_context_bearish = (1 - sector_rank_score) # 板块排名越低，弱势越强
        sector_negative_momentum_context_bearish = sector_momentum_score.clip(upper=0).abs()
        # 3. 计算板块看跌共振因子 (几何平均)
        bearish_sector_resonance = (sector_weakness_context_bearish * sector_negative_momentum_context_bearish).pow(0.5).fillna(0.0)
        # 4. 计算看跌放大因子
        bearish_amplification_factor = 1 + bearish_sector_resonance
        # 5. 最终看跌协同分数 (转换为负分)
        final_bearish_score = bearish_stock_movement * bearish_amplification_factor * -1
        # --- 最终融合 ---
        relationship_score = final_bullish_score + final_bearish_score
        final_score = relationship_score.clip(-1, 1)
        # --- 探针输出 ---
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            last_date_index = -1
            print(f"\n--- [PROCESS_META_STOCK_SECTOR_SYNC 探针: {df.index[last_date_index].strftime('%Y-%m-%d')}] ---")
            print("  [输入原料 (原始值)]: ")
            print(f"    - {stock_signal_name}: {stock_signal_raw.iloc[last_date_index]:.4f}")
            print(f"    - {sector_rank_name}: {sector_rank_raw.iloc[last_date_index]:.4f}")
            print(f"    - {sector_momentum_name}: {sector_momentum_raw.iloc[last_date_index]:.4f}")
            print("  [关键计算 (归一化/中间分)]: ")
            print(f"    - stock_strength_score (bipolar): {stock_strength_score.iloc[last_date_index]:.4f}")
            print(f"    - sector_rank_score (unipolar): {sector_rank_score.iloc[last_date_index]:.4f}")
            print(f"    - sector_momentum_score (bipolar): {sector_momentum_score.iloc[last_date_index]:.4f}")
            print("  [看涨协同部分]: ")
            print(f"    - bullish_stock_movement: {bullish_stock_movement.iloc[last_date_index]:.4f}")
            print(f"    - sector_strength_context_bullish: {sector_strength_context_bullish.iloc[last_date_index]:.4f}")
            print(f"    - sector_momentum_context_bullish: {sector_momentum_context_bullish.iloc[last_date_index]:.4f}")
            print(f"    - bullish_sector_resonance: {bullish_sector_resonance.iloc[last_date_index]:.4f}")
            print(f"    - bullish_amplification_factor: {bullish_amplification_factor.iloc[last_date_index]:.4f}")
            print(f"    - final_bullish_score: {final_bullish_score.iloc[last_date_index]:.4f}")
            print("  [看跌协同部分]: ")
            print(f"    - bearish_stock_movement: {bearish_stock_movement.iloc[last_date_index]:.4f}")
            print(f"    - sector_weakness_context_bearish: {sector_weakness_context_bearish.iloc[last_date_index]:.4f}")
            print(f"    - sector_negative_momentum_context_bearish: {sector_negative_momentum_context_bearish.iloc[last_date_index]:.4f}")
            print(f"    - bearish_sector_resonance: {bearish_sector_resonance.iloc[last_date_index]:.4f}")
            print(f"    - bearish_amplification_factor: {bearish_amplification_factor.iloc[last_date_index]:.4f}")
            print(f"    - final_bearish_score: {final_bearish_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]: ")
            print(f"    - final_score: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        print("    -> [过程层] 正在计算 PROCESS_META_STORM_EYE_CALM (V4.0 · 全息共振版)...")
        df_index = df.index
        params = get_param_value(config.get('storm_eye_calm_params'), {})
        energy_compression_weights = get_param_value(params.get('energy_compression_weights'), {"tension": 0.2, "bbw_inverted": 0.15, "vol_instability_inverted": 0.15, "equilibrium_compression": 0.2, "bbw_slope_inverted": 0.15, "vol_instability_slope_inverted": 0.15})
        volume_exhaustion_weights = get_param_value(params.get('volume_exhaustion_weights'), {"volume_atrophy": 0.2, "turnover_rate_inverted": 0.15, "counterparty_exhaustion": 0.15, "order_book_liquidity_inverted": 0.15, "buy_quote_exhaustion": 0.15, "sell_quote_exhaustion": 0.1, "turnover_rate_slope_inverted": 0.1})
        main_force_covert_intent_weights = get_param_value(params.get('main_force_covert_intent_weights'), {"stealth_ops": 0.15, "split_order_accum": 0.15, "mf_conviction_positive": 0.15, "mf_net_flow_positive": 0.15, "mf_cost_advantage_positive": 0.1, "mf_buy_ofi_positive": 0.1, "mf_t0_buy_efficiency_positive": 0.1, "mf_net_flow_slope_positive": 0.1})
        subdued_market_sentiment_weights = get_param_value(params.get('subdued_market_sentiment_weights'), {"sentiment_pendulum_negative": 0.25, "market_sentiment_inverted": 0.25, "retail_panic_inverted": 0.15, "retail_fomo_inverted": 0.15, "loser_pain_positive": 0.2})
        breakout_readiness_weights = get_param_value(params.get('breakout_readiness_weights'), {"struct_breakout_readiness": 0.3, "struct_platform_foundation": 0.25, "goodness_of_fit": 0.25, "platform_conviction": 0.2})
        final_fusion_weights = get_param_value(params.get('final_fusion_weights'), {"energy_compression": 0.2, "volume_exhaustion": 0.2, "main_force_covert_intent": 0.25, "subdued_market_sentiment": 0.15, "breakout_readiness": 0.2})
        price_calmness_modulator_params = get_param_value(params.get('price_calmness_modulator_params'), {"slope_period": 5, "pct_change_threshold": 0.005, "modulator_factor": 0.5})
        main_force_control_adjudicator_params = get_param_value(params.get('main_force_control_adjudicator'), {"control_signal": "control_solidity_index_D", "activity_signal": "main_force_activity_ratio_D", "veto_threshold": -0.2, "amplifier_factor": 0.5})
        mtf_slope_accel_weights = get_param_value(params.get('mtf_slope_accel_weights'), {}) # 新增MTF斜率加速度权重
        regime_modulator_params = get_param_value(params.get('regime_modulator_params'), {}) # 新增市场情境调节器参数
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        required_signals = [
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'control_solidity_index_D',
            'BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'turnover_rate_f_D',
            'counterparty_exhaustion_index_D', 'main_force_conviction_index_D',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY',
            'main_force_net_flow_calibrated_D', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'market_sentiment_score_D', 'SLOPE_5_close_D', 'pct_change_D',
            'equilibrium_compression_index_D', 'SLOPE_5_BBW_21_2.0_D', 'SLOPE_5_VOLATILITY_INSTABILITY_INDEX_21d_D',
            'order_book_liquidity_supply_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D', 'SLOPE_5_turnover_rate_f_D',
            'main_force_cost_advantage_D', 'main_force_buy_ofi_D', 'main_force_t0_buy_efficiency_D', 'SLOPE_5_main_force_net_flow_calibrated_D',
            'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D', 'loser_pain_index_D',
            'SCORE_STRUCT_BREAKOUT_READINESS', 'SCORE_STRUCT_PLATFORM_FOUNDATION', 'goodness_of_fit_score_D', 'platform_conviction_score_D',
            'main_force_activity_ratio_D',
            # 新增MTF斜率/加速度和微观结构指标
            'order_book_imbalance_D', 'micro_price_impact_asymmetry_D', 'ADX_14_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        for base_sig in ['BBW_21_2.0_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'turnover_rate_f_D', 'main_force_net_flow_calibrated_D']:
            for period_str in get_param_value(mtf_slope_accel_weights.get('slope_periods'), {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig}')
            for period_str in get_param_value(mtf_slope_accel_weights.get('accel_periods'), {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig}')
        if not self._validate_required_signals(df, required_signals, "_calculate_storm_eye_calm"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', np.nan) # 默认值改为np.nan
        bbw_raw = self._get_safe_series(df, 'BBW_21_2.0_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        vol_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        equilibrium_compression_raw = self._get_safe_series(df, 'equilibrium_compression_index_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        # 使用_get_mtf_slope_accel_score计算MTF斜率/加速度分数
        bbw_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'BBW_21_2.0_D', mtf_slope_accel_weights, df_index, "_calculate_storm_eye_calm", ascending=True, bipolar=False)
        vol_instability_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_slope_accel_weights, df_index, "_calculate_storm_eye_calm", ascending=True, bipolar=False)
        atrophy_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', np.nan) # 默认值改为np.nan
        turnover_rate_raw = self._get_safe_series(df, 'turnover_rate_f_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        counterparty_exhaustion_raw = self._get_safe_series(df, 'counterparty_exhaustion_index_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        order_book_liquidity_raw = self._get_safe_series(df, 'order_book_liquidity_supply_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        buy_quote_exhaustion_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        sell_quote_exhaustion_raw = self._get_safe_series(df, 'sell_quote_exhaustion_rate_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        # 使用_get_mtf_slope_accel_score计算MTF斜率/加速度分数
        turnover_rate_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, "_calculate_storm_eye_calm", ascending=True, bipolar=False)
        # 新增微观结构指标
        order_book_imbalance_raw = self._get_safe_series(df, 'order_book_imbalance_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, 'micro_price_impact_asymmetry_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', np.nan) # 默认值改为np.nan
        split_order_accum_score = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', np.nan) # 默认值改为np.nan
        mf_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        mf_net_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        mf_cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        mf_buy_ofi_raw = self._get_safe_series(df, 'main_force_buy_ofi_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        mf_t0_buy_efficiency_raw = self._get_safe_series(df, 'main_force_t0_buy_efficiency_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        # 使用_get_mtf_slope_accel_score计算MTF斜率/加速度分数
        mf_net_flow_slope_positive = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, "_calculate_storm_eye_calm", ascending=True, bipolar=False)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', np.nan) # 默认值改为np.nan
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        retail_panic_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        struct_breakout_readiness_score = self._get_atomic_score(df, 'SCORE_STRUCT_BREAKOUT_READINESS', np.nan) # 默认值改为np.nan
        struct_platform_foundation_score = self._get_atomic_score(df, 'SCORE_STRUCT_PLATFORM_FOUNDATION', np.nan) # 默认值改为np.nan
        goodness_of_fit_raw = self._get_safe_series(df, 'goodness_of_fit_score_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        platform_conviction_raw = self._get_safe_series(df, 'platform_conviction_score_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        price_slope_raw = self._get_safe_series(df, f'SLOPE_{price_calmness_modulator_params.get("slope_period", 5)}_close_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        pct_change_raw = self._get_safe_series(df, 'pct_change_D', np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        control_solidity_raw = self._get_safe_series(df, main_force_control_adjudicator_params.get('control_signal', 'control_solidity_index_D'), np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        mf_activity_ratio_raw = self._get_safe_series(df, main_force_control_adjudicator_params.get('activity_signal', 'main_force_activity_ratio_D'), np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        # 新增市场情境调节器信号
        volatility_regime_raw = self._get_safe_series(df, regime_modulator_params.get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D'), np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        trend_regime_raw = self._get_safe_series(df, regime_modulator_params.get('trend_signal', 'ADX_14_D'), np.nan, method_name="_calculate_storm_eye_calm") # 默认值改为np.nan
        bbw_inverted_score = self._normalize_series(bbw_raw, df_index, ascending=False)
        vol_instability_inverted_score = self._normalize_series(vol_instability_raw, df_index, ascending=False)
        equilibrium_compression_score = self._normalize_series(equilibrium_compression_raw, df_index, ascending=True)
        energy_compression_scores_dict = {
            'tension': tension_score,
            'bbw_inverted': bbw_inverted_score,
            'vol_instability_inverted': vol_instability_inverted_score,
            'equilibrium_compression': equilibrium_compression_score,
            'bbw_slope_inverted': bbw_slope_inverted_score,
            'vol_instability_slope_inverted': vol_instability_slope_inverted_score
        }
        energy_compression_score = _robust_geometric_mean(energy_compression_scores_dict, energy_compression_weights, df_index)
        turnover_rate_inverted_score = self._normalize_series(turnover_rate_raw, df_index, ascending=False)
        counterparty_exhaustion_score = self._normalize_series(counterparty_exhaustion_raw, df_index, ascending=True)
        order_book_liquidity_inverted_score = self._normalize_series(order_book_liquidity_raw, df_index, ascending=False)
        buy_quote_exhaustion_score = self._normalize_series(buy_quote_exhaustion_raw, df_index, ascending=True)
        sell_quote_exhaustion_score = self._normalize_series(sell_quote_exhaustion_raw, df_index, ascending=True)
        # 新增微观结构指标归一化
        order_book_imbalance_inverted = self._normalize_series(order_book_imbalance_raw.abs(), df_index, ascending=False) # 绝对值越小，越平衡，越平静
        micro_price_impact_asymmetry_inverted = self._normalize_series(micro_price_impact_asymmetry_raw.abs(), df_index, ascending=False) # 绝对值越小，冲击越对称，越平静
        volume_exhaustion_scores_dict = {
            'volume_atrophy': atrophy_score,
            'turnover_rate_inverted': turnover_rate_inverted_score,
            'counterparty_exhaustion': counterparty_exhaustion_score,
            'order_book_liquidity_inverted': order_book_liquidity_inverted_score,
            'buy_quote_exhaustion': buy_quote_exhaustion_score,
            'sell_quote_exhaustion': sell_quote_exhaustion_score,
            'turnover_rate_slope_inverted': turnover_rate_slope_inverted_score,
            'order_book_imbalance_inverted': order_book_imbalance_inverted, # 新增微观结构指标
            'micro_price_impact_asymmetry_inverted': micro_price_impact_asymmetry_inverted # 新增微观结构指标
        }
        # 更新volume_exhaustion_weights以包含新指标
        volume_exhaustion_weights.update({"order_book_imbalance_inverted": 0.05, "micro_price_impact_asymmetry_inverted": 0.05})
        volume_exhaustion_score = _robust_geometric_mean(volume_exhaustion_scores_dict, volume_exhaustion_weights, df_index)
        stealth_ops_normalized = self._normalize_series(stealth_ops_score, df_index, ascending=True)
        split_order_accum_normalized = self._normalize_series(split_order_accum_score, df_index, ascending=True)
        mf_conviction_positive = self._normalize_series(mf_conviction_raw, df_index, bipolar=True).clip(lower=0)
        mf_net_flow_positive = self._normalize_series(mf_net_flow_raw, df_index, bipolar=True).clip(lower=0)
        mf_cost_advantage_positive = self._normalize_series(mf_cost_advantage_raw, df_index, bipolar=True).clip(lower=0)
        mf_buy_ofi_positive = self._normalize_series(mf_buy_ofi_raw, df_index, ascending=True)
        mf_t0_buy_efficiency_positive = self._normalize_series(mf_t0_buy_efficiency_raw, df_index, ascending=True)
        # 新增微观结构指标归一化
        order_book_imbalance_positive = self._normalize_series(order_book_imbalance_raw.clip(lower=0), df_index, ascending=True) # 关注买方不平衡
        micro_price_impact_asymmetry_positive = self._normalize_series(micro_price_impact_asymmetry_raw.clip(lower=0), df_index, ascending=True) # 关注买方冲击优势
        main_force_covert_intent_scores_dict = {
            'stealth_ops': stealth_ops_normalized,
            'split_order_accum': split_order_accum_normalized,
            'mf_conviction_positive': mf_conviction_positive,
            'mf_net_flow_positive': mf_net_flow_positive,
            'mf_cost_advantage_positive': mf_cost_advantage_positive,
            'mf_buy_ofi_positive': mf_buy_ofi_positive,
            'mf_t0_buy_efficiency_positive': mf_t0_buy_efficiency_positive,
            'mf_net_flow_slope_positive': mf_net_flow_slope_positive,
            'order_book_imbalance_positive': order_book_imbalance_positive, # 新增微观结构指标
            'micro_price_impact_asymmetry_positive': micro_price_impact_asymmetry_positive # 新增微观结构指标
        }
        # 更新main_force_covert_intent_weights以包含新指标
        main_force_covert_intent_weights.update({"order_book_imbalance_positive": 0.05, "micro_price_impact_asymmetry_positive": 0.05})
        main_force_covert_intent_score = _robust_geometric_mean(main_force_covert_intent_scores_dict, main_force_covert_intent_weights, df_index)
        sentiment_pendulum_negative = self._normalize_series(sentiment_pendulum_score, df_index, bipolar=True).clip(upper=0).abs()
        market_sentiment_inverted = self._normalize_series(market_sentiment_raw, df_index, ascending=False)
        retail_panic_inverted = self._normalize_series(retail_panic_raw, df_index, ascending=False)
        retail_fomo_inverted = self._normalize_series(retail_fomo_raw, df_index, ascending=False)
        loser_pain_positive = self._normalize_series(loser_pain_raw, df_index, ascending=True)
        subdued_market_sentiment_scores_dict = {
            'sentiment_pendulum_negative': sentiment_pendulum_negative,
            'market_sentiment_inverted': market_sentiment_inverted,
            'retail_panic_inverted': retail_panic_inverted,
            'retail_fomo_inverted': retail_fomo_inverted,
            'loser_pain_positive': loser_pain_positive
        }
        subdued_market_sentiment_score = _robust_geometric_mean(subdued_market_sentiment_scores_dict, subdued_market_sentiment_weights, df_index)
        goodness_of_fit_score = self._normalize_series(goodness_of_fit_raw, df_index, ascending=True)
        platform_conviction_score = self._normalize_series(platform_conviction_raw, df_index, ascending=True)
        breakout_readiness_scores_dict = {
            'struct_breakout_readiness': struct_breakout_readiness_score,
            'struct_platform_foundation': struct_platform_foundation_score,
            'goodness_of_fit': goodness_of_fit_score,
            'platform_conviction': platform_conviction_score
        }
        breakout_readiness_score = _robust_geometric_mean(breakout_readiness_scores_dict, breakout_readiness_weights, df_index)
        # 市场情境动态调节器
        market_regime_modulator = pd.Series(1.0, index=df_index, dtype=np.float32)
        if get_param_value(regime_modulator_params.get('enabled'), False):
            volatility_sensitivity = get_param_value(regime_modulator_params.get('volatility_sensitivity'), 0.5)
            trend_sensitivity = get_param_value(regime_modulator_params.get('trend_sensitivity'), 0.5)
            base_modulator_factor = get_param_value(regime_modulator_params.get('base_modulator_factor'), 1.0)
            min_modulator = get_param_value(regime_modulator_params.get('min_modulator'), 0.8)
            max_modulator = get_param_value(regime_modulator_params.get('max_modulator'), 1.2)
            volatility_norm = self._normalize_series(volatility_regime_raw, df_index, ascending=False) # 波动率越低，越平静，调节器越高
            trend_norm = self._normalize_series(trend_regime_raw, df_index, ascending=False) # 趋势越弱，越平静，调节器越高
            # 融合波动率和趋势对调节器的影响
            market_regime_modulator = (
                base_modulator_factor +
                (volatility_norm * volatility_sensitivity + trend_norm * trend_sensitivity) / (volatility_sensitivity + trend_sensitivity + 1e-9)
            ).clip(min_modulator, max_modulator)
        # 动态调整final_fusion_weights
        adjusted_final_fusion_weights = {k: v * market_regime_modulator for k, v in final_fusion_weights.items()}
        # 使用调整后的权重计算base_calm_score
        base_calm_scores_dict = {
            'energy_compression': energy_compression_score,
            'volume_exhaustion': volume_exhaustion_score,
            'main_force_covert_intent': main_force_covert_intent_score,
            'subdued_market_sentiment': subdued_market_sentiment_score,
            'breakout_readiness': breakout_readiness_score
        }
        base_calm_score = _robust_geometric_mean(base_calm_scores_dict, adjusted_final_fusion_weights, df_index) # 使用调整后的权重
        price_slope_norm_bipolar = self._normalize_series(price_slope_raw, df_index, bipolar=True)
        pct_change_abs_norm_inverted = self._normalize_series(pct_change_raw.abs(), df_index, ascending=False)
        price_calmness_modulator = (price_calmness_modulator_params.get('modulator_factor', 0.5) * (1 - price_slope_norm_bipolar.abs()) + (1 - price_calmness_modulator_params.get('modulator_factor', 0.5)) * pct_change_abs_norm_inverted).clip(0,1)
        price_calmness_amplifier = 1 + (price_calmness_modulator * price_calmness_modulator_params.get('modulator_factor', 0.5))
        control_solidity_score = self._normalize_series(control_solidity_raw, df_index, bipolar=True)
        mf_activity_ratio_score = self._normalize_series(mf_activity_ratio_raw, df_index, ascending=True)
        veto_threshold = main_force_control_adjudicator_params.get('veto_threshold', -0.2)
        amplifier_factor = main_force_control_adjudicator_params.get('amplifier_factor', 0.5)
        final_score = base_calm_score * price_calmness_amplifier
        combined_control_score = (control_solidity_score * 0.7 + mf_activity_ratio_score * 0.3).clip(-1, 1)
        final_score = final_score.mask(combined_control_score < veto_threshold, 0.0)
        main_force_amplifier = 1 + (combined_control_score * amplifier_factor)
        final_score = (final_score * main_force_amplifier).clip(0, 1).fillna(0.0)
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            last_date_index = -1
            print(f"\n--- [PROCESS_META_STORM_EYE_CALM 探针: {df.index[last_date_index].strftime('%Y-%m-%d')}] ---")
            print("  [输入原料 (原始值)]: ")
            print(f"    - SCORE_STRUCT_AXIOM_TENSION: {tension_score.iloc[last_date_index]:.4f}")
            print(f"    - SCORE_BEHAVIOR_VOLUME_ATROPHY: {atrophy_score.iloc[last_date_index]:.4f}")
            print(f"    - control_solidity_index_D: {control_solidity_raw.iloc[last_date_index]:.4f}")
            print(f"    - BBW_21_2.0_D: {bbw_raw.iloc[last_date_index]:.4f}")
            print(f"    - VOLATILITY_INSTABILITY_INDEX_21d_D: {vol_instability_raw.iloc[last_date_index]:.4f}")
            print(f"    - equilibrium_compression_index_D: {equilibrium_compression_raw.iloc[last_date_index]:.4f}")
            # 输出MTF斜率/加速度的原始数据
            for period_str in get_param_value(mtf_slope_accel_weights.get('slope_periods'), {}).keys():
                print(f"    - SLOPE_{period_str}_BBW_21_2.0_D: {self._get_safe_series(df, f'SLOPE_{period_str}_BBW_21_2.0_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
                print(f"    - SLOPE_{period_str}_VOLATILITY_INSTABILITY_INDEX_21d_D: {self._get_safe_series(df, f'SLOPE_{period_str}_VOLATILITY_INSTABILITY_INDEX_21d_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
                print(f"    - SLOPE_{period_str}_turnover_rate_f_D: {self._get_safe_series(df, f'SLOPE_{period_str}_turnover_rate_f_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
                print(f"    - SLOPE_{period_str}_main_force_net_flow_calibrated_D: {self._get_safe_series(df, f'SLOPE_{period_str}_main_force_net_flow_calibrated_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
            for period_str in get_param_value(mtf_slope_accel_weights.get('accel_periods'), {}).keys():
                print(f"    - ACCEL_{period_str}_BBW_21_2.0_D: {self._get_safe_series(df, f'ACCEL_{period_str}_BBW_21_2.0_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
                print(f"    - ACCEL_{period_str}_VOLATILITY_INSTABILITY_INDEX_21d_D: {self._get_safe_series(df, f'ACCEL_{period_str}_VOLATILITY_INSTABILITY_INDEX_21d_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
                print(f"    - ACCEL_{period_str}_turnover_rate_f_D: {self._get_safe_series(df, f'ACCEL_{period_str}_turnover_rate_f_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
                print(f"    - ACCEL_{period_str}_main_force_net_flow_calibrated_D: {self._get_safe_series(df, f'ACCEL_{period_str}_main_force_net_flow_calibrated_D', np.nan, method_name='_calculate_storm_eye_calm').iloc[last_date_index]:.4f}")
            print(f"    - turnover_rate_f_D: {turnover_rate_raw.iloc[last_date_index]:.4f}")
            print(f"    - counterparty_exhaustion_index_D: {counterparty_exhaustion_raw.iloc[last_date_index]:.4f}")
            print(f"    - order_book_liquidity_supply_D: {order_book_liquidity_raw.iloc[last_date_index]:.4f}")
            print(f"    - buy_quote_exhaustion_rate_D: {buy_quote_exhaustion_raw.iloc[last_date_index]:.4f}")
            print(f"    - sell_quote_exhaustion_rate_D: {sell_quote_exhaustion_raw.iloc[last_date_index]:.4f}")
            # 输出新增的微观结构指标原始值
            print(f"    - order_book_imbalance_D: {order_book_imbalance_raw.iloc[last_date_index]:.4f}")
            print(f"    - micro_price_impact_asymmetry_D: {micro_price_impact_asymmetry_raw.iloc[last_date_index]:.4f}")
            print(f"    - SCORE_MICRO_STRATEGY_STEALTH_OPS: {stealth_ops_score.iloc[last_date_index]:.4f}")
            print(f"    - PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY: {split_order_accum_score.iloc[last_date_index]:.4f}")
            print(f"    - main_force_conviction_index_D: {mf_conviction_raw.iloc[last_date_index]:.4f}")
            print(f"    - main_force_net_flow_calibrated_D: {mf_net_flow_raw.iloc[last_date_index]:.4f}")
            print(f"    - main_force_cost_advantage_D: {mf_cost_advantage_raw.iloc[last_date_index]:.4f}")
            print(f"    - main_force_buy_ofi_D: {mf_buy_ofi_raw.iloc[last_date_index]:.4f}")
            print(f"    - main_force_t0_buy_efficiency_D: {mf_t0_buy_efficiency_raw.iloc[last_date_index]:.4f}")
            print(f"    - SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM: {sentiment_pendulum_score.iloc[last_date_index]:.4f}")
            print(f"    - market_sentiment_score_D: {market_sentiment_raw.iloc[last_date_index]:.4f}")
            print(f"    - retail_panic_surrender_index_D: {retail_panic_raw.iloc[last_date_index]:.4f}")
            print(f"    - retail_fomo_premium_index_D: {retail_fomo_raw.iloc[last_date_index]:.4f}")
            print(f"    - loser_pain_index_D: {loser_pain_raw.iloc[last_date_index]:.4f}")
            print(f"    - SCORE_STRUCT_BREAKOUT_READINESS: {struct_breakout_readiness_score.iloc[last_date_index]:.4f}")
            print(f"    - SCORE_STRUCT_PLATFORM_FOUNDATION: {struct_platform_foundation_score.iloc[last_date_index]:.4f}")
            print(f"    - goodness_of_fit_score_D: {goodness_of_fit_raw.iloc[last_date_index]:.4f}")
            print(f"    - platform_conviction_score_D: {platform_conviction_raw.iloc[last_date_index]:.4f}")
            print(f"    - SLOPE_5_close_D: {price_slope_raw.iloc[last_date_index]:.4f}")
            print(f"    - pct_change_D: {pct_change_raw.iloc[last_date_index]:.4f}")
            print(f"    - main_force_activity_ratio_D: {mf_activity_ratio_raw.iloc[last_date_index]:.4f}")
            # 输出市场情境调节器原始信号
            print(f"    - {regime_modulator_params.get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D')}: {volatility_regime_raw.iloc[last_date_index]:.4f}")
            print(f"    - {regime_modulator_params.get('trend_signal', 'ADX_14_D')}: {trend_regime_raw.iloc[last_date_index]:.4f}")
            print("  [关键计算 (归一化/中间分)]: ")
            print(f"    - bbw_inverted_score: {bbw_inverted_score.iloc[last_date_index]:.4f}")
            print(f"    - vol_instability_inverted_score: {vol_instability_inverted_score.iloc[last_date_index]:.4f}")
            print(f"    - equilibrium_compression_score: {equilibrium_compression_score.iloc[last_date_index]:.4f}")
            print(f"    - bbw_slope_inverted_score (MTF): {bbw_slope_inverted_score.iloc[last_date_index]:.4f}") # 注明MTF
            print(f"    - vol_instability_slope_inverted_score (MTF): {vol_instability_slope_inverted_score.iloc[last_date_index]:.4f}") # 注明MTF
            print(f"    - energy_compression_score: {energy_compression_score.iloc[last_date_index]:.4f}")
            print(f"    - turnover_rate_inverted_score: {turnover_rate_inverted_score.iloc[last_date_index]:.4f}")
            print(f"    - counterparty_exhaustion_score: {counterparty_exhaustion_score.iloc[last_date_index]:.4f}")
            print(f"    - order_book_liquidity_inverted_score: {order_book_liquidity_inverted_score.iloc[last_date_index]:.4f}")
            print(f"    - buy_quote_exhaustion_score: {buy_quote_exhaustion_score.iloc[last_date_index]:.4f}")
            print(f"    - sell_quote_exhaustion_score: {sell_quote_exhaustion_score.iloc[last_date_index]:.4f}")
            print(f"    - turnover_rate_slope_inverted_score (MTF): {turnover_rate_slope_inverted_score.iloc[last_date_index]:.4f}") # 注明MTF
            # 输出新增的微观结构指标归一化分数
            print(f"    - order_book_imbalance_inverted: {order_book_imbalance_inverted.iloc[last_date_index]:.4f}")
            print(f"    - micro_price_impact_asymmetry_inverted: {micro_price_impact_asymmetry_inverted.iloc[last_date_index]:.4f}")
            print(f"    - volume_exhaustion_score: {volume_exhaustion_score.iloc[last_date_index]:.4f}")
            print(f"    - stealth_ops_normalized: {stealth_ops_normalized.iloc[last_date_index]:.4f}")
            print(f"    - split_order_accum_normalized: {split_order_accum_normalized.iloc[last_date_index]:.4f}")
            print(f"    - mf_conviction_positive: {mf_conviction_positive.iloc[last_date_index]:.4f}")
            print(f"    - mf_net_flow_positive: {mf_net_flow_positive.iloc[last_date_index]:.4f}")
            print(f"    - mf_cost_advantage_positive: {mf_cost_advantage_positive.iloc[last_date_index]:.4f}")
            print(f"    - mf_buy_ofi_positive: {mf_buy_ofi_positive.iloc[last_date_index]:.4f}")
            print(f"    - mf_t0_buy_efficiency_positive: {mf_t0_buy_efficiency_positive.iloc[last_date_index]:.4f}")
            print(f"    - mf_net_flow_slope_positive (MTF): {mf_net_flow_slope_positive.iloc[last_date_index]:.4f}") # 注明MTF
            # 输出新增的微观结构指标归一化分数
            print(f"    - order_book_imbalance_positive: {order_book_imbalance_positive.iloc[last_date_index]:.4f}")
            print(f"    - micro_price_impact_asymmetry_positive: {micro_price_impact_asymmetry_positive.iloc[last_date_index]:.4f}")
            print(f"    - main_force_covert_intent_score: {main_force_covert_intent_score.iloc[last_date_index]:.4f}")
            print(f"    - sentiment_pendulum_negative: {sentiment_pendulum_negative.iloc[last_date_index]:.4f}")
            print(f"    - market_sentiment_inverted: {market_sentiment_inverted.iloc[last_date_index]:.4f}")
            print(f"    - retail_panic_inverted: {retail_panic_inverted.iloc[last_date_index]:.4f}")
            print(f"    - retail_fomo_inverted: {retail_fomo_inverted.iloc[last_date_index]:.4f}")
            print(f"    - loser_pain_positive: {loser_pain_positive.iloc[last_date_index]:.4f}")
            print(f"    - subdued_market_sentiment_score: {subdued_market_sentiment_score.iloc[last_date_index]:.4f}")
            print(f"    - breakout_readiness_score: {breakout_readiness_score.iloc[last_date_index]:.4f}")
            print(f"    - goodness_of_fit_score: {goodness_of_fit_score.iloc[last_date_index]:.4f}")
            print(f"    - platform_conviction_score: {platform_conviction_score.iloc[last_date_index]:.4f}")
            # 输出市场情境调节器相关分数
            print(f"    - volatility_regime_norm: {self._normalize_series(volatility_regime_raw, df_index, ascending=False).iloc[last_date_index]:.4f}")
            print(f"    - trend_regime_norm: {self._normalize_series(trend_regime_raw, df_index, ascending=False).iloc[last_date_index]:.4f}")
            print(f"    - market_regime_modulator: {market_regime_modulator.iloc[last_date_index]:.4f}")
            print(f"    - base_calm_score: {base_calm_score.iloc[last_date_index]:.4f}")
            print(f"    - price_calmness_modulator: {price_calmness_modulator.iloc[last_date_index]:.4f}")
            print(f"    - price_calmness_amplifier: {price_calmness_amplifier.iloc[last_date_index]:.4f}")
            print(f"    - control_solidity_score: {control_solidity_score.iloc[last_date_index]:.4f}")
            print(f"    - mf_activity_ratio_score: {mf_activity_ratio_score.iloc[last_date_index]:.4f}")
            print(f"    - combined_control_score: {combined_control_score.iloc[last_date_index]:.4f}")
            print(f"    - main_force_amplifier: {main_force_amplifier.iloc[last_date_index]:.4f}")
            print("  [最终结果]: ")
            print(f"    - final_storm_eye_calm_score: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

    def _perform_meta_analysis_on_score(self, relationship_score: pd.Series, config: Dict, df: pd.DataFrame, df_index: pd.Index) -> pd.Series:
        """
        【V1.2 · 数据脉络贯通版】可复用的元分析核心引擎。
        - 核心升级: 新增 `df` 参数，接收完整的DataFrame。
        - 核心修复: 修正了“门控元分析”逻辑，使其从 `df` 而非临时的 `relationship_score.to_frame()`
                      中获取 `close_D` 和均线数据，彻底解决了数据缺失的警告。
        """
        signal_name = config.get('name')
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0)
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        bipolar_displacement_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_displacement,
            index=df_index, # 将 target_index 改为 index
            tf_weights=actual_mtf_weights, # 使用修正后的权重字典
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_momentum_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_momentum,
            index=df_index, # 将 target_index 改为 index
            tf_weights=actual_mtf_weights, # 使用修正后的权重字典
            sensitivity=self.bipolar_sensitivity
        )
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
                # [修改] 从完整的df中获取均线和收盘价数据
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
        【V2.2 · 探针可控版】计算通用的瞬时关系分数。
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
        # 修正 MTF 权重配置的获取路径，从 structural_ultimate_params 中获取
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        # 修正 get_param_value 的默认值，避免嵌套的 'weights' 键
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        momentum_a = get_adaptive_mtf_normalized_bipolar_score(change_a, index=df_index, tf_weights=actual_mtf_weights, sensitivity=self.bipolar_sensitivity)
        thrust_b = get_adaptive_mtf_normalized_bipolar_score(change_b, index=df_index, tf_weights=actual_mtf_weights, sensitivity=self.bipolar_sensitivity)
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

    def _calculate_process_wash_out_rebound(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series) -> pd.Series:
        print("    -> [过程层] 正在计算 PROCESS_META_WASH_OUT_REBOUND (V2.4 · 深度情境感知版)...") # 修改代码行
        df_index = df.index
        p_conf = get_params_block(self.strategy, 'process_intelligence_params', {})
        params = get_param_value(p_conf.get('wash_out_rebound_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"deception_context": 0.3, "panic_depth": 0.3, "rebound_quality": 0.4})
        # 修改代码：新增更多权重配置
        deception_context_weights = get_param_value(params.get('deception_context_weights'), {"wash_trade": 0.2, "active_selling": 0.15, "fused_deception": 0.2, "behavior_deception_index": 0.2, "stealth_ops": 0.15, "wash_trade_slope": 0.05, "active_selling_slope": 0.05})
        panic_depth_weights = get_param_value(params.get('panic_depth_weights'), {"panic_cascade": 0.2, "retail_surrender": 0.2, "loser_pain": 0.2, "holder_sentiment_inverted": 0.15, "sentiment_pendulum_negative": 0.15, "retail_surrender_slope": 0.05, "loser_pain_slope": 0.05})
        rebound_quality_weights = get_param_value(params.get('rebound_quality_weights'), {"offensive_absorption_intent": 0.2, "closing_strength": 0.15, "upward_purity": 0.15, "absorption_strength": 0.2, "offensive_absorption": 0.15, "mf_buy_execution_alpha": 0.05, "buy_sweep_intensity": 0.1})
        context_amplification_weights = get_param_value(params.get('context_amplification_weights'), {"trend_form": 0.4, "stability": 0.3, "tension": 0.15, "mtf_cohesion": 0.15}) # 修改代码行
        max_context_bonus_factor = get_param_value(params.get('max_context_bonus_factor'), 0.5)
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 2. 获取所有原始数据和原子信号 ---
        required_signals = [
            'wash_trade_intensity_D', 'deception_index_D', 'active_selling_pressure_D',
            'panic_selling_cascade_D', 'retail_panic_surrender_index_D', 'loser_pain_index_D',
            'closing_strength_index_D', 'upward_impulse_purity_D',
            'SCORE_STRUCT_AXIOM_TREND_FORM', 'SCORE_STRUCT_AXIOM_STABILITY',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            # 修改代码：新增依赖信号
            'SCORE_BEHAVIOR_DECEPTION_INDEX', 'SCORE_MICRO_STRATEGY_STEALTH_OPS',
            'SLOPE_5_wash_trade_intensity_D', 'SLOPE_5_active_selling_pressure_D',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM',
            'SLOPE_5_retail_panic_surrender_index_D', 'SLOPE_5_loser_pain_index_D',
            'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT',
            'main_force_buy_execution_alpha_D', 'buy_sweep_intensity_D',
            'SCORE_STRUCT_AXIOM_TENSION', 'SCORE_STRUCT_AXIOM_MTF_COHESION'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_process_wash_out_rebound"):
            print(f"    -> [过程情报警告] _calculate_process_wash_out_rebound 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        # Raw data
        wash_trade_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        active_selling_raw = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        panic_cascade_raw = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        retail_surrender_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        closing_strength_raw = self._get_safe_series(df, 'closing_strength_index_D', 0.5, method_name="_calculate_process_wash_out_rebound")
        upward_purity_raw = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        trend_form_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        stability_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_STABILITY', 0.0)
        # 修改代码：新增获取信号
        behavior_deception_index = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DECEPTION_INDEX', 0.0)
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        wash_trade_slope_raw = self._get_safe_series(df, 'SLOPE_5_wash_trade_intensity_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        active_selling_slope_raw = self._get_safe_series(df, 'SLOPE_5_active_selling_pressure_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        holder_sentiment_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        retail_surrender_slope_raw = self._get_safe_series(df, 'SLOPE_5_retail_panic_surrender_index_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        loser_pain_slope_raw = self._get_safe_series(df, 'SLOPE_5_loser_pain_index_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        absorption_strength_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 0.0)
        offensive_absorption_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 0.0)
        mf_buy_execution_alpha_raw = self._get_safe_series(df, 'main_force_buy_execution_alpha_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        buy_sweep_intensity_raw = self._get_safe_series(df, 'buy_sweep_intensity_D', 0.0, method_name="_calculate_process_wash_out_rebound")
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        mtf_cohesion_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_MTF_COHESION', 0.0)
        # --- 3. 维度一：洗盘诱空背景 (Wash-out Deception Context) ---
        wash_trade_score = get_adaptive_mtf_normalized_score(wash_trade_raw, df_index, actual_mtf_weights, ascending=True)
        active_selling_score = get_adaptive_mtf_normalized_score(active_selling_raw, df_index, actual_mtf_weights, ascending=True)
        deception_positive_lure_long_score = get_adaptive_mtf_normalized_score(deception_lure_long_raw, df_index, actual_mtf_weights, ascending=True)
        deception_lure_short_score = get_adaptive_mtf_normalized_score(deception_lure_short_raw, df_index, actual_mtf_weights, ascending=True)
        fused_deception_score = (
            deception_positive_lure_long_score * deception_context_weights.get('deception_positive_lure_long', 0.2) +
            deception_lure_short_score * deception_context_weights.get('deception_lure_short', 0.2)
        ) / (deception_context_weights.get('deception_positive_lure_long', 0.2) + deception_context_weights.get('deception_lure_short', 0.2) + 1e-9)
        # 修改代码：新增归一化和融合
        behavior_deception_score_negative = behavior_deception_index.clip(upper=0).abs() # 负向欺骗
        stealth_ops_normalized = get_adaptive_mtf_normalized_score(stealth_ops_score, df_index, actual_mtf_weights, ascending=True)
        wash_trade_slope_score = get_adaptive_mtf_normalized_score(wash_trade_slope_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True) # 增加的洗盘强度斜率
        active_selling_slope_score = get_adaptive_mtf_normalized_score(active_selling_slope_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True) # 增加的主动卖压斜率
        deception_context_score = (
            (wash_trade_score).pow(deception_context_weights.get('wash_trade', 0.2)) *
            (active_selling_score).pow(deception_context_weights.get('active_selling', 0.15)) *
            (fused_deception_score).pow(deception_context_weights.get('fused_deception', 0.2)) *
            (behavior_deception_score_negative).pow(deception_context_weights.get('behavior_deception_index', 0.2)) *
            (stealth_ops_normalized).pow(deception_context_weights.get('stealth_ops', 0.15)) *
            (wash_trade_slope_score).pow(deception_context_weights.get('wash_trade_slope', 0.05)) *
            (active_selling_slope_score).pow(deception_context_weights.get('active_selling_slope', 0.05))
        ).pow(1/sum(deception_context_weights.values())).fillna(0.0)
        # --- 4. 维度二：恐慌割肉深度 (Panic Capitulation Depth) ---
        panic_cascade_score = get_adaptive_mtf_normalized_score(panic_cascade_raw, df_index, actual_mtf_weights, ascending=True)
        retail_surrender_score = get_adaptive_mtf_normalized_score(retail_surrender_raw, df_index, actual_mtf_weights, ascending=True)
        loser_pain_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, actual_mtf_weights, ascending=True)
        # 修改代码：新增归一化和融合
        holder_sentiment_inverted_score = (1 - holder_sentiment_score).clip(0, 1) # 低信念韧性
        sentiment_pendulum_negative_score = sentiment_pendulum_score.clip(upper=0).abs() # 负向情绪
        retail_surrender_slope_score = get_adaptive_mtf_normalized_score(retail_surrender_slope_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True) # 恐慌加速
        loser_pain_slope_score = get_adaptive_mtf_normalized_score(loser_pain_slope_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True) # 亏损痛苦加速
        panic_depth_score = (
            (panic_cascade_score).pow(panic_depth_weights.get('panic_cascade', 0.2)) *
            (retail_surrender_score).pow(panic_depth_weights.get('retail_surrender', 0.2)) *
            (loser_pain_score).pow(panic_depth_weights.get('loser_pain', 0.2)) *
            (holder_sentiment_inverted_score).pow(panic_depth_weights.get('holder_sentiment_inverted', 0.15)) *
            (sentiment_pendulum_negative_score).pow(panic_depth_weights.get('sentiment_pendulum_negative', 0.15)) *
            (retail_surrender_slope_score).pow(panic_depth_weights.get('retail_surrender_slope', 0.05)) *
            (loser_pain_slope_score).pow(panic_depth_weights.get('loser_pain_slope', 0.05))
        ).pow(1/sum(panic_depth_weights.values())).fillna(0.0)
        # --- 5. 维度三：承接反弹品质 (Absorption Rebound Quality) ---
        absorption_intent_score = offensive_absorption_intent
        closing_strength_score = normalize_score(closing_strength_raw, df_index, 55)
        upward_purity_score = get_adaptive_mtf_normalized_score(upward_purity_raw, df_index, actual_mtf_weights, ascending=True)
        # 修改代码：新增归一化和融合
        absorption_strength_normalized = get_adaptive_mtf_normalized_score(absorption_strength_score, df_index, actual_mtf_weights, ascending=True)
        offensive_absorption_normalized = get_adaptive_mtf_normalized_score(offensive_absorption_score, df_index, actual_mtf_weights, ascending=True)
        mf_buy_execution_alpha_score = get_adaptive_mtf_normalized_score(mf_buy_execution_alpha_raw, df_index, actual_mtf_weights, ascending=True)
        buy_sweep_intensity_score = get_adaptive_mtf_normalized_score(buy_sweep_intensity_raw, df_index, actual_mtf_weights, ascending=True)
        rebound_quality_score = (
            (absorption_intent_score).pow(rebound_quality_weights.get('offensive_absorption_intent', 0.2)) *
            (closing_strength_score).pow(rebound_quality_weights.get('closing_strength', 0.15)) *
            (upward_purity_score).pow(rebound_quality_weights.get('upward_purity', 0.15)) *
            (absorption_strength_normalized).pow(rebound_quality_weights.get('absorption_strength', 0.2)) *
            (offensive_absorption_normalized).pow(rebound_quality_weights.get('offensive_absorption', 0.15)) *
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

    def _calculate_process_covert_accumulation(self, df: pd.DataFrame) -> pd.Series:
        print("    -> [过程层] 正在计算 PROCESS_META_COVERT_ACCUMULATION (V2.4 · 深度情境感知版)...") # 修改代码行
        p_conf = get_params_block(self.strategy, 'process_intelligence_params', {})
        params = get_param_value(p_conf.get('covert_accumulation_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"market_context": 0.3, "covert_action": 0.4, "chip_optimization": 0.3})
        # 修改代码：新增更多权重配置
        market_context_weights = get_param_value(params.get('market_context_weights'), {"retail_panic": 0.2, "price_weakness_inverted": 0.2, "low_volatility": 0.2, "sentiment_pendulum_inverted": 0.15, "tension_inverted": 0.1, "market_sentiment_inverted": 0.1, "volatility_instability_inverted": 0.05})
        covert_action_weights = get_param_value(params.get('covert_action_weights'), {"suppressive_accum": 0.15, "main_force_flow": 0.15, "fused_deception": 0.15, "stealth_ops": 0.15, "split_order_accum": 0.1, "chip_historical_potential": 0.1, "mf_buy_ofi": 0.05, "mf_cost_advantage": 0.05, "mf_flow_slope": 0.05, "suppressive_accum_slope": 0.05})
        chip_optimization_weights = get_param_value(params.get('chip_optimization_weights'), {"chip_fatigue": 0.25, "loser_pain": 0.25, "holder_sentiment_inverted": 0.2, "turnover_purity_cost_opt": 0.15, "floating_chip_cleansing": 0.1, "total_loser_rate": 0.05})
        price_weakness_slope_window = get_param_value(params.get('price_weakness_slope_window'), 5)
        low_volatility_bbw_window = get_param_value(params.get('low_volatility_bbw_window'), 21)
        p_conf_structural_ultimate = get_params_block(self.strategy, 'structural_ultimate_params', {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # --- 2. 获取所有原始数据 ---
        required_signals = [
            'retail_panic_surrender_index_D', f'SLOPE_{price_weakness_slope_window}_close_D', f'BBW_{low_volatility_bbw_window}_2.0_D',
            'suppressive_accumulation_intensity_D', 'main_force_net_flow_calibrated_D', 'deception_index_D',
            'chip_fatigue_index_D', 'loser_pain_index_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            # 修改代码：新增依赖信号
            'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 'SCORE_STRUCT_AXIOM_TENSION',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'SCORE_MICRO_STRATEGY_STEALTH_OPS', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY',
            'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 'main_force_buy_ofi_D', 'main_force_cost_advantage_D',
            'SLOPE_5_main_force_net_flow_calibrated_D', 'SLOPE_5_suppressive_accumulation_intensity_D',
            'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION',
            'floating_chip_cleansing_efficiency_D', 'total_loser_rate_D', 'loser_concentration_90pct_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_process_covert_accumulation"):
            print(f"    -> [过程情报警告] _calculate_process_covert_accumulation 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        retail_panic_raw = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_process_covert_accumulation")
        price_weakness_slope_raw = self._get_safe_series(df, f'SLOPE_{price_weakness_slope_window}_close_D', 0.0, method_name="_calculate_process_covert_accumulation")
        bbw_raw = self._get_safe_series(df, f'BBW_{low_volatility_bbw_window}_2.0_D', 0.0, method_name="_calculate_process_covert_accumulation")
        suppressive_accum_raw = self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_process_covert_accumulation")
        main_force_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_process_covert_accumulation")
        deception_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name="_calculate_process_covert_accumulation")
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name="_calculate_process_covert_accumulation")
        loser_pain_raw = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_process_covert_accumulation")
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name="_calculate_process_covert_accumulation")
        deception_lure_short_raw = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name="_calculate_process_covert_accumulation")
        # 修改代码：新增获取信号
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        tension_score = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TENSION', 0.0)
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name="_calculate_process_covert_accumulation")
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_calculate_process_covert_accumulation")
        stealth_ops_score = self._get_atomic_score(df, 'SCORE_MICRO_STRATEGY_STEALTH_OPS', 0.0)
        split_order_accum_score = self._get_atomic_score(df, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0)
        chip_historical_potential_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        mf_buy_ofi_raw = self._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name="_calculate_process_covert_accumulation")
        mf_cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_process_covert_accumulation")
        mf_flow_slope_raw = self._get_safe_series(df, 'SLOPE_5_main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_process_covert_accumulation")
        suppressive_accum_slope_raw = self._get_safe_series(df, 'SLOPE_5_suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_process_covert_accumulation")
        holder_sentiment_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT', 0.0)
        turnover_purity_cost_opt_score = self._get_atomic_score(df, 'SCORE_CHIP_TURNOVER_PURITY_COST_OPTIMIZATION', 0.0)
        floating_chip_cleansing_raw = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name="_calculate_process_covert_accumulation")
        total_loser_rate_raw = self._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name="_calculate_process_covert_accumulation")
        # --- 3. 维度一：市场背景 (Market Context) ---
        retail_panic_score = get_adaptive_mtf_normalized_score(retail_panic_raw, df_index, actual_mtf_weights, ascending=True)
        price_weakness_score_inverted = (1 - get_adaptive_mtf_normalized_score(price_weakness_slope_raw.clip(upper=0).abs(), df_index, actual_mtf_weights, ascending=True))
        low_volatility_score = get_adaptive_mtf_normalized_score(bbw_raw, df_index, actual_mtf_weights, ascending=False)
        # 修改代码：新增归一化和融合
        sentiment_pendulum_inverted_score = (1 - sentiment_pendulum_score.clip(lower=0)) # 情绪低迷
        tension_inverted_score = (1 - tension_score.clip(lower=0)) # 低张力
        market_sentiment_inverted_score = (1 - get_adaptive_mtf_normalized_score(market_sentiment_raw, df_index, actual_mtf_weights, ascending=True))
        volatility_instability_inverted_score = (1 - get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, actual_mtf_weights, ascending=True))
        market_context_score = (
            (retail_panic_score).pow(market_context_weights.get('retail_panic', 0.2)) *
            (price_weakness_score_inverted).pow(market_context_weights.get('price_weakness_inverted', 0.2)) *
            (low_volatility_score).pow(market_context_weights.get('low_volatility', 0.2)) *
            (sentiment_pendulum_inverted_score).pow(market_context_weights.get('sentiment_pendulum_inverted', 0.15)) *
            (tension_inverted_score).pow(market_context_weights.get('tension_inverted', 0.1)) *
            (market_sentiment_inverted_score).pow(market_context_weights.get('market_sentiment_inverted', 0.1)) *
            (volatility_instability_inverted_score).pow(market_context_weights.get('volatility_instability_inverted', 0.05))
        ).pow(1/sum(market_context_weights.values())).fillna(0.0)
        # --- 4. 维度二：隐蔽行动 (Covert Action) ---
        suppressive_accum_score = get_adaptive_mtf_normalized_score(suppressive_accum_raw, df_index, actual_mtf_weights, ascending=True)
        main_force_flow_score = get_adaptive_mtf_normalized_score(main_force_flow_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True)
        deception_positive_lure_long_score = get_adaptive_mtf_normalized_score(deception_lure_long_raw, df_index, actual_mtf_weights, ascending=True)
        deception_lure_short_score = get_adaptive_mtf_normalized_score(deception_lure_short_raw, df_index, actual_mtf_weights, ascending=True)
        fused_deception_score = (
            deception_positive_lure_long_score * covert_action_weights.get('deception_positive_lure_long', 0.2) +
            deception_lure_short_score * covert_action_weights.get('deception_lure_short', 0.2)
        ) / (covert_action_weights.get('deception_positive_lure_long', 0.2) + covert_action_weights.get('deception_lure_short', 0.2) + 1e-9)
        # 修改代码：新增归一化和融合
        stealth_ops_normalized = get_adaptive_mtf_normalized_score(stealth_ops_score, df_index, actual_mtf_weights, ascending=True)
        split_order_accum_normalized = get_adaptive_mtf_normalized_score(split_order_accum_score, df_index, actual_mtf_weights, ascending=True)
        chip_historical_potential_normalized = get_adaptive_mtf_normalized_score(chip_historical_potential_score.clip(lower=0), df_index, actual_mtf_weights, ascending=True)
        mf_buy_ofi_normalized = get_adaptive_mtf_normalized_score(mf_buy_ofi_raw, df_index, actual_mtf_weights, ascending=True)
        mf_cost_advantage_normalized = get_adaptive_mtf_normalized_score(mf_cost_advantage_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True)
        mf_flow_slope_normalized = get_adaptive_mtf_normalized_score(mf_flow_slope_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True)
        suppressive_accum_slope_normalized = get_adaptive_mtf_normalized_score(suppressive_accum_slope_raw.clip(lower=0), df_index, actual_mtf_weights, ascending=True)
        covert_action_score = (
            (suppressive_accum_score).pow(covert_action_weights.get('suppressive_accum', 0.15)) *
            (main_force_flow_score).pow(covert_action_weights.get('main_force_flow', 0.15)) *
            (fused_deception_score).pow(covert_action_weights.get('fused_deception', 0.15)) *
            (stealth_ops_normalized).pow(covert_action_weights.get('stealth_ops', 0.15)) *
            (split_order_accum_normalized).pow(covert_action_weights.get('split_order_accum', 0.1)) *
            (chip_historical_potential_normalized).pow(covert_action_weights.get('chip_historical_potential', 0.1)) *
            (mf_buy_ofi_normalized).pow(covert_action_weights.get('mf_buy_ofi', 0.05)) *
            (mf_cost_advantage_normalized).pow(covert_action_weights.get('mf_cost_advantage', 0.05)) *
            (mf_flow_slope_normalized).pow(covert_action_weights.get('mf_flow_slope', 0.05)) *
            (suppressive_accum_slope_normalized).pow(covert_action_weights.get('suppressive_accum_slope', 0.05))
        ).pow(1/sum(covert_action_weights.values())).fillna(0.0)
        # --- 5. 维度三：筹码优化 (Chip Optimization) ---
        chip_fatigue_score = get_adaptive_mtf_normalized_score(chip_fatigue_raw, df_index, actual_mtf_weights, ascending=True)
        loser_pain_score = get_adaptive_mtf_normalized_score(loser_pain_raw, df_index, actual_mtf_weights, ascending=True)
        # 修改代码：新增归一化和融合
        holder_sentiment_inverted_score = (1 - holder_sentiment_score).clip(0, 1)
        turnover_purity_cost_opt_normalized = get_adaptive_mtf_normalized_score(turnover_purity_cost_opt_score.clip(lower=0), df_index, actual_mtf_weights, ascending=True)
        floating_chip_cleansing_normalized = get_adaptive_mtf_normalized_score(floating_chip_cleansing_raw, df_index, actual_mtf_weights, ascending=True)
        total_loser_rate_normalized = get_adaptive_mtf_normalized_score(total_loser_rate_raw, df_index, actual_mtf_weights, ascending=True)
        chip_optimization_score = (
            (chip_fatigue_score).pow(chip_optimization_weights.get('chip_fatigue', 0.25)) *
            (loser_pain_score).pow(chip_optimization_weights.get('loser_pain', 0.25)) *
            (holder_sentiment_inverted_score).pow(chip_optimization_weights.get('holder_sentiment_inverted', 0.2)) *
            (turnover_purity_cost_opt_normalized).pow(chip_optimization_weights.get('turnover_purity_cost_opt', 0.15)) *
            (floating_chip_cleansing_normalized).pow(chip_optimization_weights.get('floating_chip_cleansing', 0.1)) *
            (total_loser_rate_normalized).pow(chip_optimization_weights.get('total_loser_rate', 0.05))
        ).pow(1/sum(chip_optimization_weights.values())).fillna(0.0)
        # --- 6. 最终合成：三维融合 ---
        covert_accumulation_score = (
            (market_context_score).pow(fusion_weights.get('market_context', 0.3)) *
            (covert_action_score).pow(fusion_weights.get('covert_action', 0.4)) *
            (chip_optimization_score).pow(fusion_weights.get('chip_optimization', 0.3))
        ).pow(1/(fusion_weights.get('market_context', 0.3) + fusion_weights.get('covert_action', 0.4) + fusion_weights.get('chip_optimization', 0.3))).fillna(0.0)
        return covert_accumulation_score.clip(0, 1).astype(np.float32)






