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
        【V3.3.0 · 领域反转生成版】
        - 核心修复: 彻底移除在代码中硬编码的 `genesis_diagnostics` 列表。
        - 核心升级: 确保 `process_intelligence_params.diagnostics` 配置是诊断任务的唯一真相来源，
                      消除了重复执行的严重BUG，并遵循了“配置即代码”的最佳实践。
        - 支持生成原子情报领域的反转信号。
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

    def run_process_diagnostics(self, df: pd.DataFrame, task_type_filter: Optional[str] = None) -> Dict[str, pd.Series]:
        """
        【V5.3 · 调度逻辑修复版】过程情报分析总指挥
        - 核心修复 (V5.3): 移除了对 '_run_custom_analysis' 的错误调用。统一所有元分析任务
                           入口至 '_run_meta_analysis'，解决了 AttributeError 并简化了调度逻辑。
        - 核心修复 (V5.2): 增加了对 task_type_filter 参数的支持。
        - 核心升级 (V5.1): 对 PROCESS_META_POWER_TRANSFER 的计算进行重构。
        """
        print("启动【V5.3 · 调度逻辑修复版】过程情报分析...") # 更新版本信息
        all_process_states = {}
        p_conf = get_params_block(self.strategy, 'process_intelligence_params', {})
        diagnostics = get_param_value(p_conf.get('diagnostics'), [])
        for diag_config in diagnostics:
            if task_type_filter:
                if diag_config.get('task_type') != task_type_filter:
                    continue
            diag_name = diag_config.get('name')
            if not diag_name:
                continue
            # 拦截并使用新方法计算权力转移
            if diag_name == 'PROCESS_META_POWER_TRANSFER':
                power_transfer_score = self._calculate_power_transfer(df, diag_config)
                all_process_states[diag_name] = power_transfer_score
                self.strategy.atomic_states[diag_name] = power_transfer_score
                continue # 计算完后跳过通用逻辑
            # 移除了 'custom' 和 'meta' 的错误分支，统一由 _run_meta_analysis 处理
            score = self._run_meta_analysis(df, diag_config)
            if isinstance(score, pd.Series):
                all_process_states[diag_name] = score
                self.strategy.atomic_states[diag_name] = score
            elif isinstance(score, dict):
                all_process_states.update(score)
                self.strategy.atomic_states.update(score)
        print(f"【V5.3 · 调度逻辑修复版】分析完成，生成 {len(all_process_states)} 个过程元信号。") # 更新版本信息
        return all_process_states

    def _run_meta_analysis(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.0 · 元分析调度中心】
        - 核心职责: 作为所有非自定义元分析任务的中央调度器。
        - 核心逻辑: 根据诊断配置中的 'diagnosis_type' 字段，将任务分派给具体的诊断方法。
        - 修复: 解决了 'ProcessIntelligence' object has no attribute '_run_meta_analysis' 的 AttributeError。
        """
        #  实现了元分析的调度逻辑
        diagnosis_type = config.get('diagnosis_type', 'meta_relationship') # 默认为最常见的元关系分析
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
        【V3.3 · 指挥链静默版】计算“权力转移”信号。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_POWER_TRANSFER (V3.3 · 指挥链静默版)...")
        df_index = df.index
        net_sm_amount = self._get_safe_series(df, 'net_sh_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        net_md_amount = self._get_safe_series(df, 'net_md_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        net_lg_amount = self._get_safe_series(df, 'net_lg_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        net_elg_amount = self._get_safe_series(df, 'net_xl_amount_calibrated_D', 0.0, method_name="_calculate_power_transfer")
        trade_count = self._get_safe_series(df, 'trade_count_D', 0.0, method_name="_calculate_power_transfer")
        main_force_ofi = self._get_safe_series(df, 'main_force_ofi_D', 0.0, method_name="_calculate_power_transfer")
        wash_trade_intensity = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name="_calculate_power_transfer")
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_power_transfer")
        suppression_mask = (pct_change <= 0.01)
        rally_mask = (pct_change > 0.01)
        ofi_evidence = self._normalize_series(main_force_ofi, df_index, bipolar=True)
        wash_trade_evidence = self._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        micro_confirmation_factor = (ofi_evidence * (1 - wash_trade_evidence)).clip(-1, 1)
        final_allegiance_factor = pd.Series(0.0, index=df_index)
        if suppression_mask.any():
            allegiance_factor = pd.Series(0.0, index=df_index)
            corroboration_factor = pd.Series(0.0, index=df_index)
            md_condition_mask = suppression_mask & (net_md_amount > 0)
            if md_condition_mask.any():
                absorption_evidence = self._normalize_series(net_md_amount, df_index)
                chip_posture_axiom = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0).clip(lower=0)
                allegiance_score = (absorption_evidence * 0.6 + chip_posture_axiom * 0.4).clip(0, 1)
                allegiance_factor.loc[md_condition_mask] = allegiance_score.loc[md_condition_mask]
            sm_condition_mask = suppression_mask & (net_sm_amount > 0)
            if sm_condition_mask.any():
                sm_buy_evidence = self._normalize_series(net_sm_amount, df_index)
                trade_count_roc = trade_count.pct_change(21).fillna(0)
                trade_count_evidence = self._normalize_series(trade_count_roc, df_index)
                corroboration_score = (sm_buy_evidence * 0.7 + trade_count_evidence * 0.3).clip(0, 1)
                corroboration_factor.loc[sm_condition_mask] = corroboration_score.loc[sm_condition_mask]
            suppression_allegiance = (
                allegiance_factor * 0.2 +
                corroboration_factor * 0.2 +
                micro_confirmation_factor.clip(lower=0) * 0.6
            ).clip(0, 1)
            final_allegiance_factor.loc[suppression_mask] = suppression_allegiance.loc[suppression_mask]
        if rally_mask.any():
            retail_profit_taking = self._normalize_series(net_sm_amount.clip(upper=0).abs() + net_md_amount.clip(upper=0).abs(), df_index)
            main_force_chasing = self._normalize_series(net_lg_amount.clip(lower=0) + net_elg_amount.clip(lower=0), df_index)
            macro_evidence = (retail_profit_taking * main_force_chasing).pow(0.5)
            rally_allegiance = (
                macro_evidence * 0.4 +
                micro_confirmation_factor.clip(lower=0) * 0.6
            ).clip(0, 1)
            final_allegiance_factor.loc[rally_mask] = rally_allegiance.loc[rally_mask]
        md_to_main_force = net_md_amount * final_allegiance_factor
        sm_to_main_force = net_sm_amount * final_allegiance_factor
        effective_main_force_flow = net_lg_amount + net_elg_amount + md_to_main_force + sm_to_main_force
        effective_retail_flow = (net_sm_amount - sm_to_main_force) + (net_md_amount - md_to_main_force)
        power_transfer_raw = effective_main_force_flow.diff(1) - effective_retail_flow.diff(1)
        final_score = self._normalize_series(power_transfer_raw.fillna(0), df_index, bipolar=True)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_POWER_TRANSFER (V3.3) @ {probe_date.date()} ---")
                    triggered_scenario = "打压/横盘" if suppression_mask.loc[probe_date] else "拉升" if rally_mask.loc[probe_date] else "无"
                    print(f"      - 触发场景: {triggered_scenario} (涨幅: {pct_change.loc[probe_date]:.2%})")
                    print(f"      --- 微观确证 (通用) ---")
                    print(f"        - 主力OFI: {main_force_ofi.loc[probe_date]:.2f} -> OFI证据分: {ofi_evidence.loc[probe_date]:.4f}")
                    print(f"        - 对倒强度: {wash_trade_intensity.loc[probe_date]:.4f} -> 对倒证据分: {wash_trade_evidence.loc[probe_date]:.4f}")
                    print(f"        - 微观确证综合分: {micro_confirmation_factor.loc[probe_date]:.4f}")
                    if triggered_scenario == "打压/横盘":
                        print(f"      --- 场景一: 打压/横盘吸筹 ---")
                        print(f"        - 宏观推断(中/小单): (计算细节略)")
                    elif triggered_scenario == "拉升":
                        print(f"      --- 场景二: 拉升换手 ---")
                        print(f"        - 宏观证据(散户了结*主力追逐): (计算细节略)")
                    print(f"      --- 最终裁决 ---")
                    print(f"        - 最终归属因子 (Final Allegiance): {final_allegiance_factor.loc[probe_date]:.4f}")
                    print(f"        - 划归主力的(中单/小单)金额: {md_to_main_force.loc[probe_date]:.2f} / {sm_to_main_force.loc[probe_date]:.2f}")
                    print(f"      - 有效主力净额: {effective_main_force_flow.loc[probe_date]:.2f}")
                    print(f"      - 有效散户净额: {effective_retail_flow.loc[probe_date]:.2f}")
                    print(f"      - 最终权力转移分: {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        return final_score.astype(np.float32)

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.5 · 指挥链静默版】计算“诡道吸筹”信号。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_DECEPTIVE_ACCUMULATION (V2.5 · 指挥链静默版)...")
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        hidden_accumulation_raw = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_deceptive_accumulation")
        power_transfer_score = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        coherent_drive_score = self.strategy.atomic_states.get('SCORE_CHIP_COHERENT_DRIVE', pd.Series(0.0, index=df_index))
        price_trend_raw = self._get_safe_series(df, f'SLOPE_5_close_D', 0.0, method_name="_calculate_deceptive_accumulation")
        tactic_evidence = (hidden_accumulation_raw / 100).clip(0, 1)
        transfer_evidence = power_transfer_score.clip(lower=0)
        price_trend_norm = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, default_weights, self.bipolar_sensitivity)
        gating_score = (1 - (price_trend_norm - 0.1) / 0.4).clip(0, 1)
        bullish_evidence = (tactic_evidence * transfer_evidence).pow(0.5)
        penalty_factor = (1 + coherent_drive_score).clip(0, 1)
        final_score = (bullish_evidence * penalty_factor * gating_score).fillna(0.0)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_DECEPTIVE_ACCUMULATION @ {probe_date.date()} ---")
                    print(f"      --- 场景触发条件 (软门控) ---")
                    print(f"        - 门控得分 (Gating Score): {gating_score.loc[probe_date]:.4f} (源于价格趋势分: {price_trend_norm.loc[probe_date]:.4f})")
                    print(f"      --- 核心证据链 ---")
                    print(f"        - 核心战术 (Tactic Evidence): {tactic_evidence.loc[probe_date]:.4f}")
                    print(f"        - 权力交割 (Transfer Evidence): {transfer_evidence.loc[probe_date]:.4f}")
                    print(f"        - 结构优化 (Bipolar Input): {coherent_drive_score.loc[probe_date]:.4f} -> 惩罚因子: {penalty_factor.loc[probe_date]:.4f}")
                    print(f"      --- 最终裁决 ---")
                    print(f"        - 最终得分: {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        return final_score.astype(np.float32)

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 指挥权统一版】计算通用的瞬时关系分数。
        - 核心重构: 移除了所有针对特定信号的硬编码 `if` 判断（如 COST_ADVANTAGE_TREND）。
                     此方法现在是一个纯粹的、通用的关系计算引擎。
        - 指挥权统一: 所有特殊信号的调度逻辑已完全上移至 `_diagnose_meta_relationship`，
                       彻底解决了“重复指令”和“指挥链精神分裂”的BUG。
        """
        signal_name = config.get('name')
        # [删除] 移除所有针对特定信号的硬编码 if 判断
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
            relationship_score = (momentum_a + signal_b_factor_k * thrust_b) / (1 + signal_b_factor_k)
        relationship_score = relationship_score.clip(-1, 1)
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _calculate_cost_advantage_trend_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.2 · 指挥链静默版】计算成本优势趋势。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_COST_ADVANTAGE_TREND (V3.2 · 指挥链静默版)...") # 修改: 更新版本信息
        df_index = df.index
        std_window = self.std_window
        bipolar_sensitivity = self.bipolar_sensitivity
        price_change = self._get_safe_series(df, 'pct_change_D', pd.Series(0.0, index=df_index), method_name="_calculate_cost_advantage_trend_relationship")
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', pd.Series(0.0, index=df_index), method_name="_calculate_cost_advantage_trend_relationship")
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        P_change = get_adaptive_mtf_normalized_bipolar_score(price_change, df_index, default_weights, bipolar_sensitivity)
        CA_change = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage.diff(1).fillna(0), df_index, default_weights, bipolar_sensitivity)
        MF_flow = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_calculate_cost_advantage_trend_relationship"), df_index, default_weights, bipolar_sensitivity)
        Chip_posture = self.strategy.atomic_states.get('SCORE_CHIP_STRATEGIC_POSTURE', pd.Series(0.0, index=df_index))
        Micro_decep = self.strategy.atomic_states.get('SCORE_MICRO_AXIOM_DECEPTION', pd.Series(0.0, index=df_index))
        Up_eff_unipolar = self.strategy.atomic_states.get('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', pd.Series(0.5, index=df_index))
        Up_eff_bipolar = (Up_eff_unipolar * 2 - 1).clip(-1, 1)
        Vol_apathy_unipolar = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_ATROPHY', pd.Series(0.5, index=df_index))
        Vol_apathy_bipolar = (Vol_apathy_unipolar * 2 - 1).clip(-1, 1)
        Q1_base = (P_change.clip(lower=0) + CA_change.clip(lower=0)) / 2
        Q1_confirm = (MF_flow.clip(lower=0) + Chip_posture.clip(lower=0) + Up_eff_bipolar.clip(lower=0)) / 3
        Q1_final = Q1_base * Q1_confirm
        Q2_base = (P_change.clip(upper=0).abs() + CA_change.clip(upper=0).abs()) / 2
        MF_flow_bearish = MF_flow.clip(upper=0).abs()
        Chip_posture_bearish = Chip_posture.clip(upper=0).abs()
        Down_eff_bearish = Up_eff_bipolar.clip(upper=0).abs()
        Q2_confirm = (MF_flow_bearish + Chip_posture_bearish + Down_eff_bearish) / 3
        Q2_final = Q2_base * Q2_confirm * -1
        Q3_base = (P_change.clip(upper=0).abs() + CA_change.clip(lower=0)) / 2
        Q3_confirm = (MF_flow.clip(lower=0) + Chip_posture.clip(lower=0) + Micro_decep.clip(lower=0) + Vol_apathy_bipolar.clip(lower=0)) / 4
        Q3_final = Q3_base * Q3_confirm
        Q4_base = (P_change.clip(lower=0) + CA_change.clip(upper=0).abs()) / 2
        MF_flow_bearish_Q4 = MF_flow.clip(upper=0).abs()
        Chip_posture_bearish_Q4 = Chip_posture.clip(upper=0).abs()
        Micro_decep_bearish_Q4 = Micro_decep.clip(upper=0).abs()
        Up_eff_bearish_Q4 = Up_eff_bipolar.clip(upper=0).abs()
        Q4_confirm = (MF_flow_bearish_Q4 + Chip_posture_bearish_Q4 + Micro_decep_bearish_Q4 + Up_eff_bearish_Q4) / 4
        Q4_final = Q4_base * Q4_confirm * -1
        final_score = (Q1_final * 0.4 + Q2_final * 0.3 + Q3_final * 0.2 + Q4_final * 0.1)
        final_score = final_score.clip(-1, 1)
        self.strategy.atomic_states[f"_DEBUG_P_change"] = P_change
        self.strategy.atomic_states[f"_DEBUG_CA_change"] = CA_change
        self.strategy.atomic_states[f"_DEBUG_MF_flow"] = MF_flow
        self.strategy.atomic_states[f"_DEBUG_Chip_posture"] = Chip_posture
        self.strategy.atomic_states[f"_DEBUG_Micro_decep"] = Micro_decep
        self.strategy.atomic_states[f"_DEBUG_Up_eff_bipolar"] = Up_eff_bipolar
        self.strategy.atomic_states[f"_DEBUG_Vol_apathy_bipolar"] = Vol_apathy_bipolar
        self.strategy.atomic_states[f"_DEBUG_Q1_final"] = Q1_final
        self.strategy.atomic_states[f"_DEBUG_Q2_final"] = Q2_final
        self.strategy.atomic_states[f"_DEBUG_Q3_final"] = Q3_final
        self.strategy.atomic_states[f"_DEBUG_Q4_final"] = Q4_final
        # [删除] 移除此处的战报发布
        return final_score.astype(np.float32)

    def _calculate_main_force_rally_intent(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 指挥链静默版】计算“主力拉升意图”的专属关系分数。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_RALLY_INTENT (V3.1 · 指挥链静默版)...")
        df_index = df.index
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        price_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_main_force_rally_intent")
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_calculate_main_force_rally_intent")
        price_impact_ratio = self._get_safe_series(df, 'main_force_price_impact_ratio_D', 0.0, method_name="_calculate_main_force_rally_intent")
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
        final_rally_intent = bullish_intent.mask(bearish_mask, bearish_score)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day, (final_rally_intent + 0.35)).clip(-1, 1)
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score
        return final_rally_intent.astype(np.float32)

    def _calculate_main_force_control_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.3 · 指挥链静默版】计算“主力控盘”的专属关系分数。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_CONTROL (V1.3 · 指挥链静默版)...")
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
        self.strategy.atomic_states[f"_DEBUG_kongpan_raw"] = kongpan_raw
        self.strategy.atomic_states[f"_DEBUG_youzhuang_kongpan"] = youzhuang_kongpan
        self.strategy.atomic_states[f"_DEBUG_kongpan_score"] = kongpan_score
        self.strategy.atomic_states[f"_DEBUG_main_force_flow_score"] = main_force_flow_score
        return final_control_score.astype(np.float32)

    def _diagnose_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V4.4 · 指挥权统一版】对“关系分”进行元分析，输出分数。
        - 核心修复: 补全了对 `PROCESS_META_MAIN_FORCE_URGENCY` 信号的专属调度分支，
                     确保所有定制化计算都由本调度中心统一指挥，完成了指挥权的最终统一。
        """
        signal_name = config.get('name')
        df_index = df.index
        if signal_name == 'PROCESS_META_MAIN_FORCE_RALLY_INTENT':
            relationship_score = self._calculate_main_force_rally_intent(df, config)
        # [新增] 补全对“主力紧迫度”信号的专属调度
        elif signal_name == 'PROCESS_META_MAIN_FORCE_URGENCY':
            relationship_score = self._calculate_main_force_urgency_relationship(df, config)
        elif signal_name == 'PROCESS_META_WINNER_CONVICTION' and 'antidote_signal' in config:
            relationship_score = self._calculate_winner_conviction_relationship(df, config)
        elif signal_name == 'PROCESS_META_COST_ADVANTAGE_TREND':
            relationship_score = self._calculate_cost_advantage_trend_relationship(df, config)
        elif signal_name == 'PROCESS_META_MAIN_FORCE_CONTROL':
            relationship_score = self._calculate_main_force_control_relationship(df, config)
        elif signal_name == 'PROCESS_META_STEALTH_ACCUMULATION':
            relationship_score = self._calculate_stealth_accumulation(df, config)
        elif signal_name == 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION':
            relationship_score = self._calculate_panic_washout_accumulation(df, config)
        elif signal_name == 'PROCESS_META_DECEPTIVE_ACCUMULATION': 
            relationship_score = self._calculate_deceptive_accumulation(df, config)
        elif signal_name == 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY':
            relationship_score = self._calculate_split_order_accumulation(df, config)
        elif signal_name == 'PROCESS_META_UPTHRUST_WASHOUT': 
            relationship_score = self._calculate_upthrust_washout(df, config) 
        elif signal_name == 'PROCESS_META_ACCUMULATION_INFLECTION': 
            relationship_score = self._calculate_accumulation_inflection(df, config)
        elif signal_name == 'PROCESS_META_BREAKOUT_ACCELERATION':
            relationship_score = self._calculate_breakout_acceleration(df, config)
        else:
            relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        self.strategy.atomic_states[signal_name] = relationship_score.astype(np.float32)
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{signal_name}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        diagnosis_mode = config.get('diagnosis_mode', 'meta_analysis')
        if signal_name in ['PROCESS_META_STEALTH_ACCUMULATION', 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 'PROCESS_META_DECEPTIVE_ACCUMULATION', 'PROCESS_META_UPTHRUST_WASHOUT', 'PROCESS_META_ACCUMULATION_INFLECTION', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 'PROCESS_META_BREAKOUT_ACCELERATION']:
            diagnosis_mode = 'direct_confirmation'
        if diagnosis_mode == 'direct_confirmation':
            meta_score = relationship_score
        else:
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
            displacement_weight = self.meta_score_weights[0]
            momentum_weight = self.meta_score_weights[1]
            meta_score = (bipolar_displacement_strength * displacement_weight + bipolar_momentum_strength * momentum_weight)
        if diagnosis_mode == 'gated_meta_analysis':
            gate_condition_config = config.get('gate_condition', {})
            gate_type = gate_condition_config.get('type')
            gate_is_open = pd.Series(True, index=df_index)
            if gate_type == 'price_vs_ma':
                ma_period = gate_condition_config.get('ma_period', 5)
                ma_series = df.get(f'EMA_{ma_period}_D')
                if ma_series is not None:
                    gate_is_open = df['close_D'] < ma_series
            meta_score = meta_score * gate_is_open.astype(float)
        signal_meta = self.score_type_map.get(signal_name, {})
        scoring_mode = signal_meta.get('scoring_mode', 'unipolar')
        if scoring_mode == 'unipolar':
            meta_score = meta_score.clip(lower=0)
        meta_score = meta_score.clip(-1, 1).astype(np.float32)
        print(f"    -> [过程层] {signal_name} 计算完成，最新分值: {meta_score.iloc[-1]:.4f}")
        return {signal_name: meta_score}

    def _diagnose_split_meta_relationship(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V2.1 · 希格斯场分析法与多时间维度归一化版】分裂型元关系诊断器
        - 核心升级: 同步采用全新的“关系位移/关系动量”模型进行核心计算。
        - 【优化】将 `bipolar_displacement_strength` 和 `bipolar_momentum_strength` 的归一化方式改为多时间维度自适应归一化。
        """
        states = {}
        output_names = config.get('output_names', {})
        opportunity_signal_name = output_names.get('opportunity')
        risk_signal_name = output_names.get('risk')
        if not opportunity_signal_name or not risk_signal_name:
            print(f"        -> [分裂元分析] 警告: 缺少 'output_names' 配置，无法进行信号分裂。")
            return {}
        relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        relationship_displacement = relationship_score.diff(self.meta_window).fillna(0)
        relationship_momentum = relationship_displacement.diff(1).fillna(0)
        # 获取MTF权重配置
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
        return states

    def _calculate_winner_conviction_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.2 · 真理探针植入与多时间维度归一化版】“赢家信念”专属关系计算引擎
        - 新增功能: 植入“真理探针”，打印计算过程中的所有中间变量。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将所有组成信号的归一化方式改为多时间维度自适应归一化。
        """
        signal_a_name = config.get('signal_A')
        signal_b_name = config.get('signal_B')
        antidote_signal_name = config.get('antidote_signal')
        df_index = df.index
        def get_signal_series(signal_name: str) -> Optional[pd.Series]:
            return self._get_safe_series(df, signal_name, method_name="_calculate_winner_conviction_relationship")
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
        # 获取MTF权重配置
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        momentum_a = get_adaptive_mtf_normalized_bipolar_score(get_change_series(signal_a, config.get('change_type_A')), df_index, default_weights, self.bipolar_sensitivity)
        momentum_b_raw = get_adaptive_mtf_normalized_bipolar_score(get_change_series(signal_b, config.get('change_type_B')), df_index, default_weights, self.bipolar_sensitivity)
        momentum_antidote = get_adaptive_mtf_normalized_bipolar_score(get_change_series(signal_antidote, config.get('antidote_change_type')), df_index, default_weights, self.bipolar_sensitivity)
        antidote_k = config.get('antidote_k', 1.0)
        momentum_b_corrected = momentum_b_raw + antidote_k * momentum_antidote
        k = config.get('signal_b_factor_k', 1.0)
        relationship_score = (k * momentum_b_corrected - momentum_a) / (k + 1)
        return relationship_score.clip(-1, 1)

    def _diagnose_signal_decay(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1 · 多时间维度归一化版】信号衰减诊断器
        - 核心职责: 专门用于计算单个信号的负向变化（衰减）强度。
        - 数学逻辑: 1. 计算信号的一阶差分。 2. 只保留负值（代表衰减）。 3. 取绝对值。 4. 归一化。
        - 收益: 提供了计算“衰减”的正确且健壮的数学模型，取代了错误的关系诊断模型。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `decay_score` 的归一化方式改为多时间维度自适应归一化。
        """
        signal_name = config.get('name')
        source_signal_name = config.get('source_signal')
        # 临时补丁，将已废弃的 winner_conviction_index_D 替换为 winner_stability_index_D
        if source_signal_name == 'winner_conviction_index_D':
            original_signal_name = source_signal_name
            source_signal_name = 'winner_stability_index_D'
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
        # 获取MTF权重配置
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        decay_score = get_adaptive_mtf_normalized_score(decay_magnitude, df_index, ascending=True, tf_weights=default_weights)
        return {signal_name: decay_score.astype(np.float32)}

    def _diagnose_domain_reversal(self, df: pd.DataFrame, config: Dict) -> Dict[str, pd.Series]:
        """
        【V1.1 · 多时间维度归一化版】通用领域反转诊断器
        - 核心职责: 接收一个原子情报领域的公理信号列表和权重，计算该领域的双极性健康度，
                      然后从健康度的变化中派生底部反转和顶部反转信号。
        - 命名规范: 输出信号为 PROCESS_META_DOMAIN_BOTTOM_REVERSAL 和 PROCESS_META_DOMAIN_TOP_REVERSAL。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `bottom_reversal_score` 和 `top_reversal_score` 的归一化方式改为多时间维度自适应归一化。
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
        return {
            output_bottom_name: bottom_reversal_score.astype(np.float32),
            output_top_name: top_reversal_score.astype(np.float32)
        }

    def _calculate_stealth_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.3 · 指挥链重组版】计算“隐蔽吸筹”的专属关系分数。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心
                     `_diagnose_meta_relationship`，以确保战报的绝对真实性。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_STEALTH_ACCUMULATION (V3.3 · 指挥链重组版)...")
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        price_trend_raw = self._get_safe_series(df, f'SLOPE_5_close_D', 0.0, method_name="_calculate_stealth_accumulation")
        stability_score = self.strategy.atomic_states.get('SCORE_DYN_AXIOM_STABILITY', pd.Series(0.0, index=df_index))
        volume_atrophy_score = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_ATROPHY', pd.Series(0.0, index=df_index))
        concentration_trend_raw = self._get_safe_series(df, f'SLOPE_5_winner_concentration_90pct_D', 0.0, method_name="_calculate_stealth_accumulation")
        peak_solidity_trend_raw = self._get_safe_series(df, f'SLOPE_5_dominant_peak_solidity_D', 0.0, method_name="_calculate_stealth_accumulation")
        cost_advantage_raw = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_stealth_accumulation")
        power_transfer_score = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        split_order_accumulation_score = self.strategy.atomic_states.get('PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', pd.Series(0.0, index=df_index))
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
        consolidative_score = (evidence1_consolidative * evidence2_consolidative * evidence3_consolidative * evidence4_consolidative).pow(1/4)
        consolidative_score = consolidative_score.where(consolidative_mask, 0.0)
        final_score = pd.concat([suppressive_score, consolidative_score], axis=1).max(axis=1).fillna(0.0)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_STEALTH_ACCUMULATION @ {probe_date.date()} ---")
                    print(f"      --- 场景一: 打压式吸筹 ---")
                    print(f"        - 场景触发条件 (价格趋势 <= 0.1): {suppressive_mask.loc[probe_date]} (价格趋势分: {price_trend_norm.loc[probe_date]:.4f})")
                    print(f"        - 证据1 (成交量萎缩): {evidence1_suppressive.loc[probe_date]:.4f}")
                    print(f"        - 证据2 (权力转移): {evidence2_suppressive.loc[probe_date]:.4f}")
                    print(f"        - 证据3 (筹码集中趋势): {evidence3_suppressive.loc[probe_date]:.4f}")
                    print(f"        - 证据4 (成本优势): {evidence4_suppressive.loc[probe_date]:.4f}")
                    print(f"        - 场景一得分: {suppressive_score.loc[probe_date]:.4f}")
                    print(f"      --- 场景二: 横盘式吸筹 ---")
                    print(f"        - 场景触发条件 (稳定性 > 0.2): {consolidative_mask.loc[probe_date]} (稳定性分: {stability_score.loc[probe_date]:.4f})")
                    print(f"        - 证据1 (成交量萎缩): {evidence1_consolidative.loc[probe_date]:.4f}")
                    print(f"        - 证据2 (权力转移): {evidence2_consolidative.loc[probe_date]:.4f}")
                    print(f"        - 证据3 (筹码固化趋势): {evidence3_consolidative.loc[probe_date]:.4f}")
                    print(f"        - 证据4 (拆单吸筹): {evidence4_consolidative.loc[probe_date]:.4f}")
                    print(f"        - 场景二得分: {consolidative_score.loc[probe_date]:.4f}")
                    print(f"      --- 最终裁决 ---")
                    print(f"        - 最终得分 (max(场景一, 场景二)): {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        self.strategy.atomic_states["_DEBUG_accum_suppressive_score"] = suppressive_score
        self.strategy.atomic_states["_DEBUG_accum_consolidative_score"] = consolidative_score
        return final_score.astype(np.float32)

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.3 · 指挥链静默版】计算“恐慌洗盘吸筹”的专属信号。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_PANIC_WASHOUT_ACCUMULATION (V2.3 · 指挥链静默版)...")
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        lower_shadow_strength = self.strategy.atomic_states.get('SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', pd.Series(0.0, index=df_index))
        volume_burst = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_BURST', pd.Series(0.0, index=df_index))
        retail_panic_index = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        loser_pain_index = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        power_transfer_score = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        concentration_slope = self._get_safe_series(df, f'SLOPE_1_winner_concentration_90pct_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_panic_washout_accumulation")
        cost_advantage_slope = main_force_cost_advantage.diff(1).fillna(0)
        retail_panic_norm = get_adaptive_mtf_normalized_score(retail_panic_index, df_index, ascending=True, tf_weights=default_weights)
        loser_pain_norm = get_adaptive_mtf_normalized_score(loser_pain_index, df_index, ascending=True, tf_weights=default_weights)
        active_buying_support_norm = get_adaptive_mtf_normalized_score(active_buying_support, df_index, ascending=True, tf_weights=default_weights)
        concentration_slope_norm = get_adaptive_mtf_normalized_score(concentration_slope, df_index, ascending=True, tf_weights=default_weights)
        cost_advantage_slope_norm = get_adaptive_mtf_normalized_score(cost_advantage_slope, df_index, ascending=True, tf_weights=default_weights)
        panic_score = (retail_panic_norm * 0.7 + loser_pain_norm * 0.3).clip(0, 1)
        absorption_score = (power_transfer_score.clip(lower=0) * 0.5 + lower_shadow_strength * 0.25 + active_buying_support_norm * 0.25).clip(0, 1)
        repair_score = (concentration_slope_norm * 0.6 + cost_advantage_slope_norm * 0.4).clip(0, 1)
        is_significant_drop = (pct_change < -0.03) | (lower_shadow_strength > 0.6)
        is_volume_spike = volume_burst > 0.5
        washout_candidate_mask = is_significant_drop & is_volume_spike
        final_score = (panic_score * absorption_score * repair_score).pow(1/3)
        final_score = final_score.where(washout_candidate_mask, 0.0).fillna(0.0)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_PANIC_WASHOUT_ACCUMULATION @ {probe_date.date()} ---")
                    print(f"      --- 场景触发条件 (Washout Candidate Mask) ---")
                    print(f"        - 显著下跌或长下影 (is_significant_drop): {is_significant_drop.loc[probe_date]} (跌幅: {pct_change.loc[probe_date]:.2%}, 下影线强度: {lower_shadow_strength.loc[probe_date]:.4f})")
                    print(f"        - 成交量爆发 (is_volume_spike): {is_volume_spike.loc[probe_date]} (成交量爆发分: {volume_burst.loc[probe_date]:.4f})")
                    print(f"        - 最终掩码 (washout_candidate_mask): {washout_candidate_mask.loc[probe_date]}")
                    print(f"      --- 三维核心分数 ---")
                    print(f"        - 恐慌度 (Panic Score): {panic_score.loc[probe_date]:.4f}")
                    print(f"          - 散户恐慌分: {retail_panic_norm.loc[probe_date]:.4f} | 输家痛苦分: {loser_pain_norm.loc[probe_date]:.4f}")
                    print(f"        - 吸收度 (Absorption Score): {absorption_score.loc[probe_date]:.4f}")
                    print(f"          - 权力转移分: {power_transfer_score.clip(lower=0).loc[probe_date]:.4f} | 下影线强度: {lower_shadow_strength.loc[probe_date]:.4f} | 主动买盘支撑分: {active_buying_support_norm.loc[probe_date]:.4f}")
                    print(f"        - 修复度 (Repair Score): {repair_score.loc[probe_date]:.4f}")
                    print(f"          - 筹码集中斜率分: {concentration_slope_norm.loc[probe_date]:.4f} | 成本优势斜率分: {cost_advantage_slope_norm.loc[probe_date]:.4f}")
                    print(f"      --- 最终裁决 ---")
                    print(f"        - 最终得分: {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        self.strategy.atomic_states["_DEBUG_washout_panic_score"] = panic_score
        self.strategy.atomic_states["_DEBUG_washout_absorption_score"] = absorption_score
        self.strategy.atomic_states["_DEBUG_washout_repair_score"] = repair_score
        return final_score.astype(np.float32)

    def _calculate_upthrust_washout(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.5 · 指挥链静默版】识别主力在拉升初期利用“高开低走”阴线进行的洗盘行为。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_UPTHRUST_WASHOUT (V1.5 · 指挥链静默版)...")
        df_index = df.index
        trend_form_score = self.strategy.atomic_states.get('SCORE_STRUCT_AXIOM_TREND_FORM', pd.Series(0.0, index=df_index))
        bias_21 = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name="_calculate_upthrust_washout")
        open_price = self._get_safe_series(df, 'open_D', 0.0, method_name="_calculate_upthrust_washout")
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name="_calculate_upthrust_washout")
        prev_close = close_price.shift(1)
        volume_burst = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_BURST', pd.Series(0.0, index=df_index))
        power_transfer = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        lower_shadow_strength = self.strategy.atomic_states.get('SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', pd.Series(0.0, index=df_index))
        concentration_slope = self._get_safe_series(df, f'SLOPE_1_winner_concentration_90pct_D', 0.0, method_name="_calculate_upthrust_washout")
        split_order_accumulation = self.strategy.atomic_states.get('PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', pd.Series(0.0, index=df_index))
        chip_strategic_posture = self.strategy.atomic_states.get('SCORE_CHIP_STRATEGIC_POSTURE', pd.Series(0.0, index=df_index))
        context_score = ((trend_form_score > 0.2) & (bias_21 < 0.2)).astype(float)
        is_high_open_low_close = (open_price > prev_close) & (close_price < open_price)
        action_score = (is_high_open_low_close & (volume_burst > 0.3)).astype(float)
        internals_score = (
            lower_shadow_strength * 0.7 +
            (concentration_slope > 0).astype(float) * 0.3
        ).clip(0, 1)
        bullish_evidence = (
            split_order_accumulation * 0.5 +
            power_transfer.clip(lower=0) * 0.3 +
            chip_strategic_posture.clip(lower=0) * 0.2
        ).clip(0, 1)
        bearish_evidence = (
            power_transfer.clip(upper=0).abs() * 0.5 +
            chip_strategic_posture.clip(upper=0).abs() * 0.5
        ).clip(0, 1)
        washout_authenticity_score = (bullish_evidence - bearish_evidence).clip(0, 1)
        final_score = (context_score * internals_score * washout_authenticity_score)
        final_score = final_score.where(action_score > 0, 0.0).fillna(0.0)
        self.strategy.atomic_states["_DEBUG_washout_context_score"] = context_score
        self.strategy.atomic_states["_DEBUG_washout_action_score"] = action_score
        self.strategy.atomic_states["_DEBUG_washout_internals_score"] = internals_score
        self.strategy.atomic_states["_DEBUG_washout_authenticity_score"] = washout_authenticity_score
        self.strategy.atomic_states["_DEBUG_washout_auth_bull_evidence"] = bullish_evidence
        self.strategy.atomic_states["_DEBUG_washout_auth_bear_evidence"] = bearish_evidence
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_UPTHRUST_WASHOUT @ {probe_date.date()} ---")
                    print(f"      - 环境分 (Context): {context_score.loc[probe_date]:.4f}")
                    print(f"      - 动作分 (Action): {action_score.loc[probe_date]:.4f}")
                    print(f"      - 内核分 (Internals): {internals_score.loc[probe_date]:.4f}")
                    print(f"      - 真实性评分 (Authenticity): {washout_authenticity_score.loc[probe_date]:.4f}")
                    print(f"        - 看涨证据 (Bullish Evidence): {bullish_evidence.loc[probe_date]:.4f}")
                    print(f"          - 战略态势(Bipolar Input): {chip_strategic_posture.loc[probe_date]:.4f} -> 正向贡献: {chip_strategic_posture.clip(lower=0).loc[probe_date]:.4f}")
                    print(f"        - 看跌证据 (Bearish Evidence): {bearish_evidence.loc[probe_date]:.4f}")
                    print(f"          - 权力转移(负向): {power_transfer.clip(upper=0).abs().loc[probe_date]:.4f}")
                    print(f"          - 战略态势(负向): {chip_strategic_posture.clip(upper=0).abs().loc[probe_date]:.4f}")
                    print(f"      - 最终得分 (Final Score): {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        return final_score.astype(np.float32)

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.1 · 指挥链静默版】识别多日累积吸筹后，即将由“量变”引发“质变”的拉升拐点。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_ACCUMULATION_INFLECTION (V1.1 · 指挥链静默版)...")
        df_index = df.index
        accumulation_window = config.get('accumulation_window', 21)
        stealth_accum = self.strategy.atomic_states.get('PROCESS_META_STEALTH_ACCUMULATION', pd.Series(0.0, index=df_index))
        deceptive_accum = self.strategy.atomic_states.get('PROCESS_META_DECEPTIVE_ACCUMULATION', pd.Series(0.0, index=df_index))
        panic_washout_accum = self.strategy.atomic_states.get('PROCESS_META_PANIC_WASHOUT_ACCUMULATION', pd.Series(0.0, index=df_index))
        daily_accumulation_strength = pd.concat([stealth_accum, deceptive_accum, panic_washout_accum], axis=1).max(axis=1)
        potential_energy_raw = daily_accumulation_strength.rolling(window=accumulation_window, min_periods=5).sum()
        potential_energy_score = normalize_score(potential_energy_raw, df_index, window=accumulation_window, ascending=True).clip(0, 1)
        price_slope_1d = self._get_safe_series(df, f'SLOPE_1_close_D', 0.0, method_name="_calculate_accumulation_inflection")
        volume_burst = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_BURST', pd.Series(0.0, index=df_index))
        closing_position = self._get_safe_series(df, 'closing_price_deviation_score_D', 0.0, method_name="_calculate_accumulation_inflection")
        price_trigger = (price_slope_1d > 0).astype(float)
        volume_trigger = (volume_burst > 0.1).astype(float)
        kline_trigger = ((closing_position / 100).clip(0, 1))
        kinetic_trigger_score = (price_trigger * 0.4 + volume_trigger * 0.3 + kline_trigger * 0.3).clip(0, 1)
        final_score = (potential_energy_score * kinetic_trigger_score).fillna(0.0)
        self.strategy.atomic_states["_DEBUG_inflection_potential_energy"] = potential_energy_score
        self.strategy.atomic_states["_DEBUG_inflection_kinetic_trigger"] = kinetic_trigger_score
        return final_score.astype(np.float32)

    def _calculate_split_order_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.3 · 指挥链静默版】计算“拆单吸筹强度”的专属信号。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY (V1.3 · 指挥链静默版)...")
        df_index = df.index
        raw_intensity = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_split_order_accumulation")
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_split_order_accumulation")
        scene_mask = pct_change <= 0.02
        normalized_score = (raw_intensity / 100).clip(0, 1)
        final_score = normalized_score.where(scene_mask, 0.0).fillna(0.0)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY @ {probe_date.date()} ---")
                    print(f"      - 原始数据 (hidden_accumulation_intensity_D): {raw_intensity.loc[probe_date]:.4f}")
                    print(f"      - 场景约束 (涨幅 <= 2%): {scene_mask.loc[probe_date]} (当日涨幅: {pct_change.loc[probe_date]:.2%})")
                    print(f"      - 归一化得分 (原始值/100): {normalized_score.loc[probe_date]:.4f}")
                    print(f"      - 最终得分 (应用场景约束后): {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        return final_score.astype(np.float32)

    def _calculate_breakout_acceleration(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.2 · 指挥链静默版】诊断“突破加速抢筹”战术。
        - 核心重构: 移除了此处的最终日志输出。战报发布权已上移至调度中心。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_BREAKOUT_ACCELERATION (V1.2 · 指挥链静默版)...")
        df_index = df.index
        breakout_signal = self.strategy.atomic_states.get('SCORE_PATTERN_AXIOM_BREAKOUT', pd.Series(0.0, index=df_index))
        rally_intent_signal_name = 'PROCESS_ATOMIC_REL_SCORE_PROCESS_META_MAIN_FORCE_RALLY_INTENT'
        rally_intent = self.strategy.atomic_states.get(rally_intent_signal_name, pd.Series(0.0, index=df_index))
        power_transfer = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        trend_form = self.strategy.atomic_states.get('SCORE_STRUCT_AXIOM_TREND_FORM', pd.Series(0.0, index=df_index))
        breakout_trigger_mask = breakout_signal.rolling(window=3, min_periods=1).max() > 0.5
        driver_evidence = rally_intent.clip(lower=0)
        transfer_evidence = power_transfer.clip(lower=0)
        confirmation_evidence = trend_form.clip(lower=0)
        final_score = (driver_evidence * transfer_evidence * confirmation_evidence).pow(1/3)
        final_score = final_score.where(breakout_trigger_mask, 0.0).fillna(0.0)
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_dates = [pd.to_datetime(d).tz_localize(df_index.tz if df_index.tz else None) for d in probe_dates_str]
            for probe_date in probe_dates:
                if probe_date in df_index:
                    print(f"    -> [探针] --- PROCESS_META_BREAKOUT_ACCELERATION @ {probe_date.date()} ---")
                    print(f"      --- 战术前置 (Trigger) ---")
                    print(f"        - 近期突破事件: {breakout_trigger_mask.loc[probe_date]}")
                    print(f"      --- 核心证据链 ---")
                    print(f"        - 驱动证据 (原始主力拉升意图): {driver_evidence.loc[probe_date]:.4f}")
                    print(f"        - 交割证据 (权力转移): {transfer_evidence.loc[probe_date]:.4f}")
                    print(f"        - 确认证据 (趋势形态): {confirmation_evidence.loc[probe_date]:.4f}")
                    print(f"      --- 最终裁决 ---")
                    print(f"        - 最终得分: {final_score.loc[probe_date]:.4f}")
                    print("    -> [探针] ----------------------------------------------------")
        return final_score.astype(np.float32)




