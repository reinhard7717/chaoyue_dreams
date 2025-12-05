# 文件: strategies/trend_following/intelligence/process_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Any

from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, 
    get_adaptive_mtf_normalized_score, is_limit_up, get_adaptive_mtf_normalized_bipolar_score, 
    normalize_score
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
        """
        if column_name not in df.columns:
            print(f"    -> [过程情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _normalize_series(self, series: pd.Series, target_index: pd.Index, bipolar: bool = False) -> pd.Series:
        """
        【V1.0 · 统一归一化引擎】
        - 核心职责: 为类内部提供一个统一的、基于多时间框架自适应归一化的方法。
        - 核心逻辑: 根据 bipolar 参数，调用 get_adaptive_mtf_normalized_score (单极) 或
                     get_adaptive_mtf_normalized_bipolar_score (双极) 进行归一化。
        - 修复: 解决了 'ProcessIntelligence' object has no attribute '_normalize_series' 的 AttributeError。
        """
        #  实现了统一的归一化逻辑
        # 获取MTF权重配置
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        if bipolar:
            return get_adaptive_mtf_normalized_bipolar_score(
                series=series,
                target_index=target_index,
                tf_weights=default_weights,
                sensitivity=self.bipolar_sensitivity
            )
        else:
            return get_adaptive_mtf_normalized_score(
                series=series,
                target_index=target_index,
                ascending=True,
                tf_weights=default_weights
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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
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
        price_change_norm = get_adaptive_mtf_normalized_bipolar_score(price_change, df_index, default_weights, self.bipolar_sensitivity)
        net_flow_norm = get_adaptive_mtf_normalized_bipolar_score(main_force_net_flow, df_index, default_weights, self.bipolar_sensitivity)
        price_impact_norm = get_adaptive_mtf_normalized_bipolar_score(price_impact_ratio, df_index, default_weights, self.bipolar_sensitivity)
        impulse_purity_norm = get_adaptive_mtf_normalized_bipolar_score(upward_impulse_purity, df_index, default_weights, self.bipolar_sensitivity)
        volume_ratio_norm = get_adaptive_mtf_normalized_bipolar_score(volume_ratio - 1.0, df_index, default_weights, self.bipolar_sensitivity)
        control_solidity_norm = get_adaptive_mtf_normalized_bipolar_score(control_solidity, df_index, default_weights, self.bipolar_sensitivity)
        cost_advantage_norm = get_adaptive_mtf_normalized_bipolar_score(cost_advantage, df_index, default_weights, self.bipolar_sensitivity)
        concentration_slope_norm = get_adaptive_mtf_normalized_bipolar_score(concentration_slope, df_index, default_weights, self.bipolar_sensitivity)
        peak_solidity_norm = get_adaptive_mtf_normalized_bipolar_score(peak_solidity, df_index, default_weights, self.bipolar_sensitivity)
        buying_support_norm = get_adaptive_mtf_normalized_bipolar_score(active_buying_support, df_index, default_weights, self.bipolar_sensitivity)
        pressure_rejection_norm = get_adaptive_mtf_normalized_bipolar_score(pressure_rejection, df_index, default_weights, self.bipolar_sensitivity)
        profit_absorption_norm = get_adaptive_mtf_normalized_bipolar_score((1 - profit_realization_quality) - 0.5, df_index, default_weights, self.bipolar_sensitivity)
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
        # 优化探针日期获取逻辑并更新探针内容
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [主力拉升意图探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 顶部派发强度(原始): {df['distribution_at_peak_intensity_D'].iloc[last_date_index]:.4f}")
            print(f"    - 上影线抛压(原始): {df['upper_shadow_selling_pressure_D'].iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 基础拉升意图: {base_rally_intent.iloc[last_date_index]:.4f}")
            print(f"    - 调节后拉升意图(未审判): {modulated_rally_intent.iloc[last_date_index]:.4f}")
            print(f"    - 派发风险分: {distribution_risk_score.iloc[last_date_index]:.4f}")
            print(f"    - 调节后意图(看涨部分): {bullish_part.iloc[last_date_index]:.4f}")
            print(f"    - 调节后意图(看跌部分): {bearish_part.iloc[last_date_index]:.4f}")
            print(f"    - 惩罚后看涨部分: {penalized_bullish_part.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 风险审判后最终分: {final_rally_intent.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score
        self.strategy.atomic_states["_DEBUG_rally_distribution_risk"] = distribution_risk_score # 新增调试信号
        return final_rally_intent.astype(np.float32)

    def _calculate_main_force_control_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.4 · 探针植入版】计算“主力控盘”的专属关系分数。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_CONTROL (V1.4 · 探针植入版)...")
        df_index = df.index
        std_window = self.std_window
        bipolar_sensitivity = self.bipolar_sensitivity
        ema13 = ta.ema(close=self._get_safe_series(df, 'close_D', method_name="_calculate_main_force_control_relationship"), length=13, append=False)
        if ema13 is None:
            print(f"    -> [过程层警告] '主力控盘'计算失败：数据长度不足以计算13周期EMA。")
            return pd.Series(0.0, index=df_index)
        varn1 = ta.ema(close=ema13, length=13, append=False)
        if varn1 is None:
            print(f"    -> [过程层警告] '主力控盘'计算失败：数据长度不足以计算双重13周期EMA。")
            return pd.Series(0.0, index=df_index)
        prev_varn1 = varn1.shift(1).replace(0, np.nan)
        kongpan_raw = (varn1 - prev_varn1) / prev_varn1 * 1000
        youzhuang_kongpan = (kongpan_raw > kongpan_raw.shift(1)) & (kongpan_raw > 0)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_calculate_main_force_control_relationship")
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        kongpan_score = get_adaptive_mtf_normalized_bipolar_score(kongpan_raw, df_index, default_weights, bipolar_sensitivity)
        main_force_flow_score = get_adaptive_mtf_normalized_bipolar_score(main_force_net_flow, df_index, default_weights, bipolar_sensitivity)
        final_control_score = pd.Series(0.0, index=df_index)
        youzhuang_kongpan_float = youzhuang_kongpan.astype(float)
        final_control_score = (kongpan_score.clip(lower=0) * main_force_flow_score.clip(lower=0) * youzhuang_kongpan_float).pow(1/3)
        final_control_score = final_control_score.mask(kongpan_score < 0, kongpan_score.clip(upper=0))
        final_control_score = final_control_score.mask(main_force_flow_score < 0, main_force_flow_score.clip(upper=0))
        final_control_score = final_control_score.clip(-1, 1)
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [主力控盘探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 控盘度(原始): {kongpan_raw.iloc[last_date_index]:.4f}")
            print(f"    - 主力净流入(原始): {main_force_net_flow.iloc[last_date_index]:.2f}")
            print("  [关键计算]:")
            print(f"    - 有庄控盘(布尔): {youzhuang_kongpan.iloc[last_date_index]}")
            print(f"    - 控盘度(归一化): {kongpan_score.iloc[last_date_index]:.4f}")
            print(f"    - 主力净流入(归一化): {main_force_flow_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 主力控盘最终分: {final_control_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states[f"_DEBUG_kongpan_raw"] = kongpan_raw
        self.strategy.atomic_states[f"_DEBUG_youzhuang_kongpan"] = youzhuang_kongpan
        self.strategy.atomic_states[f"_DEBUG_kongpan_score"] = kongpan_score
        self.strategy.atomic_states[f"_DEBUG_main_force_flow_score"] = main_force_flow_score
        return final_control_score.astype(np.float32)

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V5.6 · 军令同步升级版】对“关系分”进行元分析，输出分数。
        - 核心升级: 为 PROCESS_META_LOSER_CAPITULATION 信号分派专属计算引擎，
                      确保其“恐慌吸收”的诡道逻辑得以执行。
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
        # [新增] 为“套牢盘投降”信号增加专属路由
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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        bipolar_displacement_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_displacement,
            target_index=df.index,
            tf_weights=default_weights,
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_momentum_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_momentum,
            target_index=df.index,
            tf_weights=default_weights,
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
        probe_dates = self.probe_dates
        # [修改] 增加对 enable_probe 配置的检查
        enable_probe = config.get('enable_probe', True)
        if enable_probe and not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [分裂元分析探针: {config.get('name')}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 瞬时关系分(信念校准后): {relationship_score.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 关系位移强度(归一化): {bipolar_displacement_strength.iloc[last_date_index]:.4f}")
            print(f"    - 关系动量强度(归一化): {bipolar_momentum_strength.iloc[last_date_index]:.4f}")
            print(f"    - 元分析总分(分裂前): {meta_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 机会部分({opportunity_signal_name}): {opportunity_part.iloc[last_date_index]:.4f}")
            print(f"    - 风险部分({risk_signal_name}): {risk_part.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        probe_dates = self.probe_dates
        enable_probe = config.get('enable_probe', True)
        if enable_probe and not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [瞬时关系探针(承接验证版): {config.get('name')}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 主动承接强度(原始): {active_buying_support.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 基础背离分: {base_divergence_score.iloc[last_date_index]:.4f}")
            print(f"    - 主动承接(归一化): {active_buying_norm.iloc[last_date_index]:.4f}")
            print(f"    - 真实性放大器: {authenticity_amplifier.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 瞬时关系分(承接验证后): {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        probe_dates = self.probe_dates
        enable_probe = config.get('enable_probe', True)
        if enable_probe and not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [瞬时关系探针(信念校准版): {config.get('name')}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 主力信念(原始): {main_force_conviction.iloc[last_date_index]:.4f}")
            print(f"    - 对倒强度(原始): {wash_trade_intensity.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 基础共识分: {base_consensus_score.iloc[last_date_index]:.4f}")
            print(f"    - 品质因子: {quality_factor.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 瞬时关系分(信念校准后): {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        probe_dates = self.probe_dates
        enable_probe = config.get('enable_probe', True)
        if enable_probe and not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [瞬时关系探针(战场纵深版): {config.get('name')}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 收盘价: {close_price.iloc[last_date_index]:.2f}")
            print(f"    - 主力VPOC: {mf_vpoc.iloc[last_date_index]:.2f}")
            print("  [关键计算]:")
            print(f"    - 基础背离分: {base_divergence_score.iloc[last_date_index]:.4f}")
            print(f"    - 战场纵深因子: {battlefield_context_factor.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 瞬时关系分(战场纵深校准后): {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.3 · 探针植入版】信号衰减诊断器
        - 核心清理: 移除了为兼容旧信号 `winner_conviction_index_D` 而设置的临时补丁，
                      因为配置文件已完成信号同步，代码逻辑恢复纯粹。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        signal_name = config.get('name')
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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        decay_score = get_adaptive_mtf_normalized_score(decay_magnitude, df_index, ascending=True, tf_weights=default_weights)
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [信号衰减探针: {signal_name}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 源信号 ({source_signal_name}): {source_series.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 信号变化量: {signal_change.iloc[last_date_index]:.4f}")
            print(f"    - 衰减幅度(原始): {decay_magnitude.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 衰减分数(归一化): {decay_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return {signal_name: decay_score.astype(np.float32)}

    def _diagnose_domain_reversal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.2 · 探针植入版】通用领域反转诊断器
        - 核心职责: 接收一个原子情报领域的公理信号列表和权重，计算该领域的双极性健康度，
                      然后从健康度的变化中派生底部反转和顶部反转信号。
        - 命名规范: 输出信号为 PROCESS_META_DOMAIN_BOTTOM_REVERSAL 和 PROCESS_META_DOMAIN_TOP_REVERSAL。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `bottom_reversal_score` 和 `top_reversal_score` 的归一化方式改为多时间维度自适应归一化。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
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
            axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
            domain_health_components.append(axiom_score * axiom_weight)
            total_weight += axiom_weight
        if total_weight == 0:
            print(f"        -> [领域反转诊断] 警告: 领域 '{domain_name}' 的公理权重总和为0，无法计算健康度。")
            return {}
        # 计算该领域的双极性健康度
        bipolar_domain_health = (sum(domain_health_components) / total_weight).clip(-1, 1)
        # 从健康度派生反转信号
        # 底部反转信号：当健康度从负值区域开始向上改善时
        bottom_reversal_raw = (bipolar_domain_health.diff(1).clip(lower=0) * (1 - bipolar_domain_health.clip(lower=0))).fillna(0)
        # 获取MTF权重配置
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        bottom_reversal_score = get_adaptive_mtf_normalized_score(bottom_reversal_raw, df_index, ascending=True, tf_weights=default_weights)
        # 顶部反转信号：当健康度从正值区域开始向下恶化时
        top_reversal_raw = (bipolar_domain_health.diff(1).clip(upper=0).abs() * (1 + bipolar_domain_health.clip(upper=0))).fillna(0)
        top_reversal_score = get_adaptive_mtf_normalized_score(top_reversal_raw, df_index, ascending=True, tf_weights=default_weights)
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [领域反转探针: {domain_name}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            for axiom_config in axiom_configs:
                axiom_name = axiom_config.get('name')
                axiom_score = self.strategy.atomic_states.get(axiom_name, pd.Series(0.0, index=df_index))
                print(f"    - 公理 ({axiom_name}): {axiom_score.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 领域健康度(双极性): {bipolar_domain_health.iloc[last_date_index]:.4f}")
            print(f"    - 底部反转(原始): {bottom_reversal_raw.iloc[last_date_index]:.4f}")
            print(f"    - 顶部反转(原始): {top_reversal_raw.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 底部反转分({output_bottom_name}): {bottom_reversal_score.iloc[last_date_index]:.4f}")
            print(f"    - 顶部反转分({output_top_name}): {top_reversal_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
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
        price_trend_norm = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, default_weights, self.bipolar_sensitivity)
        concentration_trend_norm = get_adaptive_mtf_normalized_bipolar_score(concentration_trend_raw, df_index, default_weights, self.bipolar_sensitivity)
        peak_solidity_trend_norm = get_adaptive_mtf_normalized_bipolar_score(peak_solidity_trend_raw, df_index, default_weights, self.bipolar_sensitivity)
        cost_advantage_norm = get_adaptive_mtf_normalized_bipolar_score(cost_advantage_raw, df_index, default_weights, self.bipolar_sensitivity)
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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
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
        retail_panic_norm = get_adaptive_mtf_normalized_score(retail_panic_index, df_index, ascending=True, tf_weights=default_weights)
        loser_pain_norm = get_adaptive_mtf_normalized_score(loser_pain_index, df_index, ascending=True, tf_weights=default_weights)
        active_buying_support_norm = get_adaptive_mtf_normalized_score(active_buying_support, df_index, ascending=True, tf_weights=default_weights)
        concentration_slope_norm = get_adaptive_mtf_normalized_score(concentration_slope, df_index, ascending=True, tf_weights=default_weights)
        cost_advantage_slope_norm = get_adaptive_mtf_normalized_score(cost_advantage_slope, df_index, ascending=True, tf_weights=default_weights)
        structural_leverage_norm = get_adaptive_mtf_normalized_score(structural_leverage_raw, df_index, ascending=True, tf_weights=default_weights)
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
        deception_evidence = self._normalize_series(deception_index_raw, df_index, bipolar=True).clip(lower=0)
        # 创立“伪装分”，奖励权力转移为负的矛盾行为
        disguise_score = (1 - power_transfer_score) / 2
        # 重构“诡道氛围”，融合“伪装分”与“欺诈证据”
        deceptive_context_score = (disguise_score * 0.6 + deception_evidence * 0.4).clip(0, 1)
        price_trend_norm = self._normalize_series(price_trend_raw, df_index, bipolar=True)
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
        【V3.0 · 恐慌吸收版】计算“套牢盘投降”信号。
        - 核心重构: 创立“恐慌与吸收”二元对抗模型。废除通用关系分析，转而审判在下跌日中，
                      “恐慌抛售的烈度”与“主力主动吸收的强度”的乘积，旨在精准捕捉洗盘终点的信号。
        - 新增功能: 植入详尽的“真理探针”，全面暴露新的“恐慌吸收”模型。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_LOSER_CAPITULATION (V3.0 · 恐慌吸收版)...")
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
        # 战场上下文：只在下跌日激活
        context_mask = (pct_change < 0)
        # 恐慌分：衡量抛售的烈度
        panic_score = self._normalize_series(capitulation_flow_raw, df_index, bipolar=False)
        # 吸收分：采用“强证优先”原则，取最强的承接证据
        active_buying_norm = self._normalize_series(active_buying_raw, df_index, bipolar=False)
        absorption_score = pd.concat([active_buying_norm, lower_shadow_absorption], axis=1).max(axis=1)
        # 最终审判：恐慌与吸收的乘积
        final_score = (panic_score * absorption_score).where(context_mask, 0.0).fillna(0.0)
        # 植入探针
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [套牢盘投降探针(恐慌吸收版)] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 当日涨跌幅: {pct_change.iloc[last_date_index]:.4f}")
            print(f"    - 恐慌抛售流量(原始): {capitulation_flow_raw.iloc[last_date_index]:.4f}")
            print(f"    - 主动买盘支撑(原始): {active_buying_raw.iloc[last_date_index]:.4f}")
            print(f"    - 下影线吸收强度: {lower_shadow_absorption.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 战场上下文(下跌日): {context_mask.iloc[last_date_index]}")
            print(f"    - 恐慌分(归一化): {panic_score.iloc[last_date_index]:.4f}")
            print(f"    - 吸收分(强证优先): {absorption_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 套牢盘投降最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

    def _calculate_cost_advantage_trend_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.1 · 象限审判版】计算成本优势趋势。
        - 核心升级: 为各象限的确认环节配备专属的、最高保真度的战术信号，实现“精准审判”。
                      - Q2(派发下跌)引入“利润兑现流量”作为核心证据。
                      - Q4(牛市陷阱)引入“买盘虚弱度”作为核心惩罚项。
        - 新增功能: 植入“真理探针”，用于在指定日期输出各象限的计算结果。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_COST_ADVANTAGE_TREND (V4.1 · 象限审判版)...")
        # [修改] 引入新的、更精准的战术信号依赖
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
        # 获取战术信号并归一化
        main_force_conviction = self._normalize_series(self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=True)
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        suppressive_accum = self._normalize_series(self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        lower_shadow_absorb = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        distribution_intensity = self._normalize_series(self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        active_selling = self._normalize_series(self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        # [新增] 获取Q2和Q4象限审判所需的新信号
        profit_taking_flow = self._normalize_series(self._get_safe_series(df, 'profit_taking_flow_ratio_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        active_buying_support = self._normalize_series(self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        # Q1: 价涨 & 优扩 (健康上涨) - 逻辑不变
        Q1_base = (P_change.clip(lower=0) * CA_change.clip(lower=0)).pow(0.5)
        Q1_confirm = (main_force_conviction.clip(lower=0) * upward_purity).pow(0.5)
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        # [修改] Q2: 价跌 & 优缩 (派发下跌) - 引入利润兑现作为核心证据
        Q2_base = (P_change.clip(upper=0).abs() * CA_change.clip(upper=0).abs()).pow(0.5)
        Q2_distribution_evidence = (profit_taking_flow * 0.6 + active_selling * 0.4).clip(0, 1)
        Q2_final = (Q2_base * Q2_distribution_evidence * -1).clip(-1, 0)
        # Q3: 价跌 & 优扩 (黄金坑) - 逻辑不变
        Q3_base = (P_change.clip(upper=0).abs() * CA_change.clip(lower=0)).pow(0.5)
        Q3_confirm = (suppressive_accum * lower_shadow_absorb).pow(0.5)
        Q3_final = (Q3_base * Q3_confirm).clip(0, 1)
        # [修改] Q4: 价涨 & 优缩 (牛市陷阱) - 引入买盘虚弱度作为惩罚
        Q4_base = (P_change.clip(lower=0) * CA_change.clip(upper=0).abs()).pow(0.5)
        Q4_trap_evidence = (distribution_intensity * (1 - active_buying_support)).clip(0, 1)
        Q4_final = (Q4_base * Q4_trap_evidence * -1).clip(-1, 0)
        final_score = (Q1_final + Q2_final + Q3_final + Q4_final).clip(-1, 1)
        # [修改] 升级探针以反映新逻辑
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [成本优势趋势探针(象限审判版)] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 价格变化(归一化): {P_change.iloc[last_date_index]:.4f}")
            print(f"    - 成本优势变化(归一化): {CA_change.iloc[last_date_index]:.4f}")
            print(f"    - 利润兑现流量(归一化): {profit_taking_flow.iloc[last_date_index]:.4f}")
            print(f"    - 主动买盘支撑(归一化): {active_buying_support.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - Q1 (健康上涨)得分: {Q1_final.iloc[last_date_index]:.4f}")
            print(f"    - Q2 (派发下跌)证据分: {Q2_distribution_evidence.iloc[last_date_index]:.4f} -> 得分: {Q2_final.iloc[last_date_index]:.4f}")
            print(f"    - Q3 (黄金坑)得分: {Q3_final.iloc[last_date_index]:.4f}")
            print(f"    - Q4 (牛市陷阱)证据分: {Q4_trap_evidence.iloc[last_date_index]:.4f} -> 得分: {Q4_final.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 成本优势趋势最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        wash_trade_penalty = self._normalize_series(self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, bipolar=False)
        volume_atrophy_quality = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        chip_posture = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        suppressive_accum = self._normalize_series(self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, bipolar=False)
        panic_evidence = self._normalize_series(self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, bipolar=False)
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, bipolar=False)
        reversal_confirmation_shape = self._get_atomic_score(df, 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', 0.0)
        reversal_confirmation_flow = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        reversal_confirmation_psyche = self._normalize_series(main_force_conviction.diff(1).fillna(0), df_index, bipolar=True)
        active_buying_confirm = self._normalize_series(self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_price_volume_relationship"), df_index, bipolar=False)
        accel_shape = self._normalize_series(reversal_confirmation_shape.diff(2).fillna(0), df_index, bipolar=False)
        accel_flow = self._normalize_series(reversal_confirmation_flow.diff(2).fillna(0), df_index, bipolar=False)
        accel_psyche = self._normalize_series(reversal_confirmation_psyche.diff(2).fillna(0), df_index, bipolar=False)
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
        【V2.1 · 探针植入版】诊断“突破加速抢筹”战术。
        - 核心升级: 引入“相对强度”公理作为环境调节器，放大“领军者”的突破信号，确认其龙头地位。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_BREAKOUT_ACCELERATION (V2.1 · 探针植入版)...")
        # 将新公理加入依赖校验
        required_signals = [
            'SCORE_PATTERN_AXIOM_BREAKOUT', 'PROCESS_ATOMIC_REL_SCORE_PROCESS_META_MAIN_FORCE_RALLY_INTENT',
            'PROCESS_META_POWER_TRANSFER', 'SCORE_STRUCT_AXIOM_TREND_FORM', 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_breakout_acceleration"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # 获取新公理及配置参数
        relative_strength = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH', 0.0)
        rs_amplifier = config.get('relative_strength_amplifier', 0.0)
        breakout_signal = self.strategy.atomic_states.get('SCORE_PATTERN_AXIOM_BREAKOUT', pd.Series(0.0, index=df_index))
        rally_intent_signal_name = 'PROCESS_ATOMIC_REL_SCORE_PROCESS_META_MAIN_FORCE_RALLY_INTENT'
        rally_intent = self.strategy.atomic_states.get(rally_intent_signal_name, pd.Series(0.0, index=df_index))
        power_transfer = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        trend_form = self.strategy.atomic_states.get('SCORE_STRUCT_AXIOM_TREND_FORM', pd.Series(0.0, index=df_index))
        breakout_trigger_mask = breakout_signal.rolling(window=3, min_periods=1).max() > 0.5
        driver_evidence = rally_intent.clip(lower=0)
        transfer_evidence = power_transfer.clip(lower=0)
        confirmation_evidence = trend_form.clip(lower=0)
        base_score = (driver_evidence * transfer_evidence * confirmation_evidence).pow(1/3)
        # 融合相对强度
        rs_modulator = (1 + relative_strength.clip(lower=0) * rs_amplifier)
        final_score = (base_score * rs_modulator).clip(0, 1)
        final_score = final_score.where(breakout_trigger_mask, 0.0).fillna(0.0)
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [突破加速抢筹探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 突破信号: {breakout_signal.iloc[last_date_index]:.4f}")
            print(f"    - 拉升意图: {rally_intent.iloc[last_date_index]:.4f}")
            print(f"    - 权力转移: {power_transfer.iloc[last_date_index]:.4f}")
            print(f"    - 趋势形态: {trend_form.iloc[last_date_index]:.4f}")
            print(f"    - 相对强度: {relative_strength.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 突破触发掩码: {breakout_trigger_mask.iloc[last_date_index]}")
            print(f"    - 基础分(三证据融合): {base_score.iloc[last_date_index]:.4f}")
            print(f"    - 相对强度调节器: {rs_modulator.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 突破加速抢筹最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

    def _calculate_fund_flow_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.2 · 探针植入版】识别主力从隐蔽吸筹转向公开强攻的转折信号。
        - 核心修复: 修正了数据访问逻辑，确保所有依赖的底层原子信号（如高频强度、校准后资金流等）
                      都从正确的数据源 `df` 中获取，而非从 `atomic_states` 中错误查找，
                      彻底解决了“依赖信号不存在”的警告。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT (V1.2 · 探针植入版)...")
        required_signals = [
            'SCORE_FF_AXIOM_FLOW_MOMENTUM', 'hidden_accumulation_intensity_D',
            'main_force_net_flow_calibrated_D', 'buy_quote_exhaustion_rate_D', 'large_order_pressure_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_fund_flow_accumulation_inflection"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        tf_weights_inflection = get_param_value(p_mtf.get('short_term_weights'), {'weights': {3: 0.5, 5: 0.3, 8: 0.2}})
        # 修正数据源，从 df 获取原子原料，从 atomic_states 获取情报产物
        flow_momentum = self._get_atomic_score(df, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0)
        psai = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        main_force_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        buy_exhaustion_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        large_pressure_raw = self._get_safe_series(df, 'large_order_pressure_D', 0.0, method_name="_calculate_fund_flow_accumulation_inflection")
        # 从config中读取参数
        psai_high_threshold = config.get('psai_high_threshold', 0.5)
        mf_flow_positive_threshold = config.get('mf_flow_positive_threshold', 0.0)
        buy_exhaustion_threshold = config.get('buy_exhaustion_threshold', 0.7)
        large_pressure_low_threshold = config.get('large_pressure_low_threshold', 0.3)
        buy_exhaustion_score = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_inflection)
        large_pressure_score = get_adaptive_mtf_normalized_score(large_pressure_raw, df_index, ascending=True, tf_weights=tf_weights_inflection)
        cond_prelude_accumulation = (psai.rolling(window=5).mean() > psai_high_threshold)
        cond_overt_attack = (
            (main_force_flow > mf_flow_positive_threshold) &
            (buy_exhaustion_score > buy_exhaustion_threshold) &
            (large_pressure_score < large_pressure_low_threshold)
        )
        inflection_intent_mask = cond_prelude_accumulation & cond_overt_attack
        inflection_intent_score = (flow_momentum.clip(lower=0) * 0.5 + buy_exhaustion_score * 0.5)
        inflection_intent_score = inflection_intent_score.where(inflection_intent_mask, 0.0)
        final_score = get_adaptive_mtf_normalized_score(inflection_intent_score, df_index, ascending=True, tf_weights=tf_weights_inflection).clip(0, 1)
        probe_dates = self.probe_dates
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [资金流吸筹拐点探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 资金流纯度与动能: {flow_momentum.iloc[last_date_index]:.4f}")
            print(f"    - 隐蔽吸筹强度(5日均): {psai.rolling(window=5).mean().iloc[last_date_index]:.4f}")
            print(f"    - 主力净流入: {main_force_flow.iloc[last_date_index]:.2f}")
            print(f"    - 买盘消耗率(归一化): {buy_exhaustion_score.iloc[last_date_index]:.4f}")
            print(f"    - 大单压力(归一化): {large_pressure_score.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 条件1(前奏吸筹)满足: {cond_prelude_accumulation.iloc[last_date_index]}")
            print(f"    - 条件2(公开强攻)满足: {cond_overt_attack.iloc[last_date_index]}")
            print(f"    - 拐点意图掩码: {inflection_intent_mask.iloc[last_date_index]}")
            print(f"    - 拐点意图得分(原始): {inflection_intent_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 资金流吸筹拐点最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        bipolar_displacement_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_displacement,
            target_index=df_index,
            tf_weights=default_weights,
            sensitivity=self.bipolar_sensitivity
        )
        bipolar_momentum_strength = get_adaptive_mtf_normalized_bipolar_score(
            series=relationship_momentum,
            target_index=df_index,
            tf_weights=default_weights,
            sensitivity=self.bipolar_sensitivity
        )
        instant_score_normalized = (relationship_score + 1) / 2
        weight_momentum = (1 - instant_score_normalized).clip(0, 1)
        weight_displacement = 1 - weight_momentum
        meta_score = (bipolar_displacement_strength * weight_displacement + bipolar_momentum_strength * weight_momentum)
        probe_dates = self.probe_dates
        enable_probe = config.get('enable_probe', True)
        if enable_probe and not relationship_score.empty and relationship_score.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [关系元分析探针: {signal_name}] ---")
            last_date_index = -1
            print(f"日期: {relationship_score.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 瞬时关系分: {relationship_score.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 关系位移强度(归一化): {bipolar_displacement_strength.iloc[last_date_index]:.4f}")
            print(f"    - 关系动量强度(归一化): {bipolar_momentum_strength.iloc[last_date_index]:.4f}")
            print(f"    - 动态权重(位移/动量): {weight_displacement.iloc[last_date_index]:.2f}/{weight_momentum.iloc[last_date_index]:.2f}")
            print("  [最终结果]:")
            print(f"    - 元分析最终分: {meta_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        momentum_a = get_adaptive_mtf_normalized_bipolar_score(change_a, df_index, default_weights, self.bipolar_sensitivity)
        thrust_b = get_adaptive_mtf_normalized_bipolar_score(change_b, df_index, default_weights, self.bipolar_sensitivity)
        signal_b_factor_k = config.get('signal_b_factor_k', 1.0)
        if relationship_type == 'divergence':
            relationship_score = (signal_b_factor_k * thrust_b - momentum_a) / (signal_b_factor_k + 1)
        else:
            force_vector_sum = momentum_a + signal_b_factor_k * thrust_b
            magnitude = (momentum_a.abs() * thrust_b.abs()).pow(0.5)
            relationship_score = np.sign(force_vector_sum) * magnitude
        relationship_score = relationship_score.clip(-1, 1).fillna(0.0)
        probe_dates = self.probe_dates
        # [修改] 增加对 enable_probe 配置的检查
        enable_probe = config.get('enable_probe', True)
        if enable_probe and not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [瞬时关系探针: {signal_name}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 信号A ({signal_a_name}): {signal_a.iloc[last_date_index]:.4f}")
            print(f"    - 信号B ({signal_b_name}): {signal_b.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 信号A动量(归一化): {momentum_a.iloc[last_date_index]:.4f}")
            print(f"    - 信号B推力(归一化): {thrust_b.iloc[last_date_index]:.4f}")
            print(f"    - 关系类型: {relationship_type}")
            if relationship_type == 'consensus':
                print(f"    - 合力方向向量: {force_vector_sum.iloc[last_date_index]:.4f}")
                print(f"    - 协同强度(几何平均): {magnitude.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 瞬时关系分: {relationship_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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










