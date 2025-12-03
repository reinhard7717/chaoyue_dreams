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
        【V5.4 · 探针清理版】过程情报分析总指挥
        - 核心清理: 更新了版本信息和日志输出，以反映探针清理后的状态。
                     为特殊调度的信号 `PROCESS_META_POWER_TRANSFER` 补上了统一的最终战报发布。
        """
        print("启动【V5.4 · 探针清理版】过程情报分析...")
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
            if diag_name == 'PROCESS_META_POWER_TRANSFER':
                power_transfer_score = self._calculate_power_transfer(df, diag_config)
                all_process_states[diag_name] = power_transfer_score
                self.strategy.atomic_states[diag_name] = power_transfer_score
                # 为特殊调度的信号补上统一的最终战报发布
                print(f"    -> [过程层] {diag_name} 计算完成，最新分值: {power_transfer_score.iloc[-1]:.4f}")
                continue
            score = self._run_meta_analysis(df, diag_config)
            if isinstance(score, pd.Series):
                all_process_states[diag_name] = score
                self.strategy.atomic_states[diag_name] = score
            elif isinstance(score, dict):
                all_process_states.update(score)
                self.strategy.atomic_states.update(score)
        print(f"【V5.4 · 探针清理版】分析完成，生成 {len(all_process_states)} 个过程元信号。")
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
        【V4.0 · 诡道甄别版】计算“权力转移”信号。
        - 核心升级: 废除基于推断的 `allegiance_factor`，引入基于直接诡道证据的 `transfer_authenticity_factor`。
                      该因子融合了“对倒强度”与“欺骗指数”，旨在甄别筹码转移的真实意图，过滤掉主力通过
                      诱多或诱空制造的虚假权力交割，从而更精确地量化真实的控盘权转移。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_POWER_TRANSFER (V4.0 · 诡道甄别版)...")
        # [修改] 引入新的诡道博弈信号依赖
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
        # [新增] 引入“转移真实性”因子，替代旧的“忠诚度”因子
        # 1. 对倒强度归一化，值越高，交易越不真实
        wash_trade_norm = self._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        # 2. 欺骗指数归一化，正分代表打压吸筹，负分代表拉高出货
        deception_norm = self._normalize_series(deception_index, df_index, bipolar=True)
        # 3. 主力信念归一化，作为真实意图的基石
        conviction_norm = self._normalize_series(main_force_conviction, df_index, bipolar=True)
        # 转移真实性 = (1 - 对倒惩罚) * (主力信念 + 欺骗修正)
        # 当主力信念强(conviction>0)且存在打压行为(deception>0)时，真实性最高
        # 当存在对倒行为(wash_trade>0)或拉高出货行为(deception<0)时，真实性降低
        transfer_authenticity_factor = ((1 - wash_trade_norm) * (conviction_norm + deception_norm.clip(lower=0))).clip(0, 1)
        # [修改] 使用新的“转移真实性”因子来计算有效转移的资金
        md_to_main_force = net_md_amount * transfer_authenticity_factor
        sm_to_main_force = net_sm_amount * transfer_authenticity_factor
        effective_main_force_flow = net_lg_amount + net_elg_amount + md_to_main_force + sm_to_main_force
        effective_retail_flow = (net_sm_amount - sm_to_main_force) + (net_md_amount - md_to_main_force)
        power_transfer_raw = effective_main_force_flow.diff(1) - effective_retail_flow.diff(1)
        final_score = self._normalize_series(power_transfer_raw.fillna(0), df_index, bipolar=True)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [权力转移探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 中单净额: {net_md_amount.iloc[last_date_index]:.2f}")
            print(f"    - 散户净额: {net_sm_amount.iloc[last_date_index]:.2f}")
            print(f"    - 对倒强度: {wash_trade_intensity.iloc[last_date_index]:.4f}")
            print(f"    - 欺骗指数: {deception_index.iloc[last_date_index]:.4f}")
            print(f"    - 主力信念: {main_force_conviction.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 对倒强度(归一化): {wash_trade_norm.iloc[last_date_index]:.4f}")
            print(f"    - 欺骗指数(归一化): {deception_norm.iloc[last_date_index]:.4f}")
            print(f"    - 主力信念(归一化): {conviction_norm.iloc[last_date_index]:.4f}")
            print(f"    - 转移真实性因子: {transfer_authenticity_factor.iloc[last_date_index]:.4f}")
            print(f"    - 中单->主力有效转移: {md_to_main_force.iloc[last_date_index]:.2f}")
            print(f"    - 散户->主力有效转移: {sm_to_main_force.iloc[last_date_index]:.2f}")
            print(f"    - 权力转移原始分: {power_transfer_raw.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 权力转移最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

    def _calculate_deceptive_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 探针植入版】计算“诡道吸筹”信号。
        - 核心重构: 彻底重构了信号的识别逻辑。现在，此信号的核心驱动力是我们新锻造的、
                      基于微观证据的 `suppressive_accumulation_intensity_D` (打压吸筹强度) 指标。
                      这使其能精准捕捉主力“打压吸筹”这一高阶战术，而不再被日线级别的“假发散”所迷惑。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_DECEPTIVE_ACCUMULATION (V3.1 · 探针植入版)...")
        # 更新依赖，引入新的核心驱动信号
        required_signals = [
            'suppressive_accumulation_intensity_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_COHERENT_DRIVE', 'SLOPE_5_close_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_deceptive_accumulation"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 使用新的“打压吸筹强度”作为核心战术证据
        suppressive_accum_raw = self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_deceptive_accumulation")
        power_transfer_score = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index))
        coherent_drive_score = self.strategy.atomic_states.get('SCORE_CHIP_COHERENT_DRIVE', pd.Series(0.0, index=df_index))
        price_trend_raw = self._get_safe_series(df, f'SLOPE_5_close_D', 0.0, method_name="_calculate_deceptive_accumulation")
        # 归一化新的核心证据
        tactic_evidence = (suppressive_accum_raw / 100).clip(0, 1)
        transfer_evidence = power_transfer_score.clip(lower=0)
        price_trend_norm = get_adaptive_mtf_normalized_bipolar_score(price_trend_raw, df_index, default_weights, self.bipolar_sensitivity)
        # 门控条件：价格趋势不能过强，否则不符合“诡道”
        gating_score = (1 - (price_trend_norm - 0.1) / 0.4).clip(0, 1)
        bullish_evidence = (tactic_evidence * transfer_evidence).pow(0.5)
        # 惩罚项：如果筹码同调性差，则降低信号分值
        penalty_factor = (1 + coherent_drive_score).clip(0, 1)
        final_score = (bullish_evidence * penalty_factor * gating_score).fillna(0.0)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [诡道吸筹探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 打压吸筹强度(原始): {suppressive_accum_raw.iloc[last_date_index]:.4f}")
            print(f"    - 权力转移分: {power_transfer_score.iloc[last_date_index]:.4f}")
            print(f"    - 筹码同调驱动力: {coherent_drive_score.iloc[last_date_index]:.4f}")
            print(f"    - 价格趋势(原始): {price_trend_raw.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 战术证据(归一化): {tactic_evidence.iloc[last_date_index]:.4f}")
            print(f"    - 权力转移证据: {transfer_evidence.iloc[last_date_index]:.4f}")
            print(f"    - 价格趋势(归一化): {price_trend_norm.iloc[last_date_index]:.4f}")
            print(f"    - 价格门控分: {gating_score.iloc[last_date_index]:.4f}")
            print(f"    - 看涨证据融合: {bullish_evidence.iloc[last_date_index]:.4f}")
            print(f"    - 筹码同调惩罚因子: {penalty_factor.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 诡道吸筹最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

    def _calculate_instantaneous_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.2 · 探针植入版】计算通用的瞬时关系分数。
        - 核心重构: 移除了所有针对特定信号的硬编码 `if` 判断（如 COST_ADVANTAGE_TREND）。
                     此方法现在是一个纯粹的、通用的关系计算引擎。
        - 指挥权统一: 所有特殊信号的调度逻辑已完全上移至 `_diagnose_meta_relationship`，
                       彻底解决了“重复指令”和“指挥链精神分裂”的BUG。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
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
            relationship_score = (momentum_a + signal_b_factor_k * thrust_b) / (1 + signal_b_factor_k)
        relationship_score = relationship_score.clip(-1, 1)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [瞬时关系探针: {signal_name}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 信号A ({signal_a_name}): {signal_a.iloc[last_date_index]:.4f}")
            print(f"    - 信号B ({signal_b_name}): {signal_b.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 信号A变化率: {change_a.iloc[last_date_index]:.4f}")
            print(f"    - 信号B变化率: {change_b.iloc[last_date_index]:.4f}")
            print(f"    - 信号A动量(归一化): {momentum_a.iloc[last_date_index]:.4f}")
            print(f"    - 信号B推力(归一化): {thrust_b.iloc[last_date_index]:.4f}")
            print(f"    - 关系类型: {relationship_type}")
            print("  [最终结果]:")
            print(f"    - 瞬时关系分: {relationship_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states[f"_DEBUG_momentum_{signal_a_name}"] = momentum_a
        self.strategy.atomic_states[f"_DEBUG_thrust_{signal_b_name}"] = thrust_b
        return relationship_score

    def _calculate_cost_advantage_trend_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.0 · 全息战场版】计算成本优势趋势。
        - 核心升级: 将四象限模型的分析基石从“间接推断”升级为“直接战术行为”。
                      使用数据层最新的战术级信号（如打压吸筹强度、顶部派发强度等）直接驱动
                      各象限的逻辑判断，极大提升了对战场真实博弈意图的识别精度。
        - 新增功能: 植入“真理探针”，用于在指定日期输出各象限的计算结果。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_COST_ADVANTAGE_TREND (V4.0 · 全息战场版)...")
        # [修改] 引入新的战术级信号依赖
        required_signals = [
            'pct_change_D', 'main_force_cost_advantage_D', 'main_force_conviction_index_D',
            'upward_impulse_purity_D', 'suppressive_accumulation_intensity_D',
            'lower_shadow_absorption_strength_D', 'distribution_at_peak_intensity_D',
            'upper_shadow_selling_pressure_D', 'active_selling_pressure_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_calculate_cost_advantage_trend_relationship"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        price_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship")
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship")
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        P_change = self._normalize_series(price_change, df_index, bipolar=True)
        CA_change = self._normalize_series(main_force_cost_advantage.diff(1).fillna(0), df_index, bipolar=True)
        # [新增] 获取新的战术信号并归一化
        main_force_conviction = self._normalize_series(self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=True)
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        suppressive_accum = self._normalize_series(self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        lower_shadow_absorb = self.strategy.atomic_states.get('SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION', pd.Series(0.0, index=df_index))
        distribution_intensity = self._normalize_series(self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        upper_shadow_pressure = self._normalize_series(self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        active_selling = self._normalize_series(self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name="_calculate_cost_advantage_trend_relationship"), df_index, bipolar=False)
        # [修改] 使用新的战术信号重构四象限逻辑
        # Q1: 价涨 & 优扩 (健康上涨)
        Q1_base = (P_change.clip(lower=0) * CA_change.clip(lower=0)).pow(0.5)
        Q1_confirm = (main_force_conviction.clip(lower=0) * upward_purity).pow(0.5)
        Q1_final = (Q1_base * Q1_confirm).clip(0, 1)
        # Q2: 价跌 & 优缩 (健康下跌/派发)
        Q2_base = (P_change.clip(upper=0).abs() * CA_change.clip(upper=0).abs()).pow(0.5)
        Q2_confirm = (active_selling * main_force_conviction.clip(upper=0).abs()).pow(0.5)
        Q2_final = (Q2_base * Q2_confirm * -1).clip(-1, 0)
        # Q3: 价跌 & 优扩 (诡道吸筹 - 黄金坑)
        Q3_base = (P_change.clip(upper=0).abs() * CA_change.clip(lower=0)).pow(0.5)
        Q3_confirm = (suppressive_accum * lower_shadow_absorb).pow(0.5)
        Q3_final = (Q3_base * Q3_confirm).clip(0, 1)
        # Q4: 价涨 & 优缩 (拉高出货 - 牛市陷阱)
        Q4_base = (P_change.clip(lower=0) * CA_change.clip(upper=0).abs()).pow(0.5)
        Q4_confirm = (distribution_intensity * upper_shadow_pressure).pow(0.5)
        Q4_final = (Q4_base * Q4_confirm * -1).clip(-1, 0)
        final_score = (Q1_final + Q2_final + Q3_final + Q4_final).clip(-1, 1)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [成本优势趋势探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 价格变化(归一化): {P_change.iloc[last_date_index]:.4f}")
            print(f"    - 成本优势变化(归一化): {CA_change.iloc[last_date_index]:.4f}")
            print(f"    - 打压吸筹强度(归一化): {suppressive_accum.iloc[last_date_index]:.4f}")
            print(f"    - 顶部派发强度(归一化): {distribution_intensity.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - Q1 (健康上涨)得分: {Q1_final.iloc[last_date_index]:.4f}")
            print(f"    - Q2 (派发下跌)得分: {Q2_final.iloc[last_date_index]:.4f}")
            print(f"    - Q3 (黄金坑)得分: {Q3_final.iloc[last_date_index]:.4f}")
            print(f"    - Q4 (牛市陷阱)得分: {Q4_final.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 成本优势趋势最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return final_score.astype(np.float32)

    def _calculate_main_force_rally_intent(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.0 · 风险审判版】计算“主力拉升意图”的专属关系分数。
        - 核心升级: 引入“风险审判”机制。在评估拉升“动力”的基础上，额外构建一个由“顶部派发强度”、
                      “上影线抛压”等信号组成的“派发风险分”，并用其对原始拉升意图进行惩罚性调节。
                      旨在穿透上涨表象，精准区分“真突破”与“拉高出货的陷阱”。
        - 新增功能: 植入“真理探针”，用于在指定日期输出风险调节前后的分数变化。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_MAIN_FORCE_RALLY_INTENT (V5.0 · 风险审判版)...")
        # [修改] 引入新的风险审判信号依赖
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
        # [新增] 风险审判模块
        distribution_intensity = self._normalize_series(self._get_safe_series(df, 'distribution_at_peak_intensity_D', 0.0, method_name="_calculate_main_force_rally_intent"), df_index, bipolar=False)
        upper_shadow_pressure = self._normalize_series(self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name="_calculate_main_force_rally_intent"), df_index, bipolar=False)
        distribution_risk_score = (distribution_intensity * 0.6 + upper_shadow_pressure * 0.4).clip(0, 1)
        # [修改] 应用风险审判调节器
        final_rally_intent = (modulated_rally_intent * (1 - distribution_risk_score)).clip(-1, 1)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day, (final_rally_intent + 0.35)).clip(-1, 1)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
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
        【V4.6 · 探针植入版】对“关系分”进行元分析，输出分数。
        - 核心扩充: 正式接管“资金流吸筹拐点意图”的计算职责，为其新增专属调度分支，
                     确保了情报架构的职责清晰与信息流的正确性。
        - 新增功能: 植入“真理探针”，用于在指定日期输出元分析过程。
        """
        signal_name = config.get('name')
        df_index = df.index
        if signal_name == 'PROCESS_META_MAIN_FORCE_RALLY_INTENT':
            relationship_score = self._calculate_main_force_rally_intent(df, config)
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
        # 为“资金流吸筹拐点意图”添加专属调度分支
        elif signal_name == 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT':
            relationship_score = self._calculate_fund_flow_accumulation_inflection(df, config)
        else:
            relationship_score = self._calculate_instantaneous_relationship(df, config)
        if relationship_score.empty:
            return {}
        self.strategy.atomic_states[signal_name] = relationship_score.astype(np.float32)
        intermediate_signal_name = f"PROCESS_ATOMIC_REL_SCORE_{signal_name}"
        self.strategy.atomic_states[intermediate_signal_name] = relationship_score.astype(np.float32)
        diagnosis_mode = config.get('diagnosis_mode', 'meta_analysis')
        if signal_name in ['PROCESS_META_STEALTH_ACCUMULATION', 'PROCESS_META_PANIC_WASHOUT_ACCUMULATION', 'PROCESS_META_DECEPTIVE_ACCUMULATION', 'PROCESS_META_UPTHRUST_WASHOUT', 'PROCESS_META_ACCUMULATION_INFLECTION', 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 'PROCESS_META_BREAKOUT_ACCELERATION', 'PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT']:
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
            # [修改] 修正探针日期获取逻辑
            debug_params = get_params_block(self.strategy, 'debug_params', {})
            probe_dates = get_param_value(debug_params.get('probe_dates'), [])
            if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
                print(f"\n--- [关系元分析探针: {signal_name}] ---")
                last_date_index = -1
                print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
                print("  [输入原料]:")
                print(f"    - 瞬时关系分: {relationship_score.iloc[last_date_index]:.4f}")
                print("  [关键计算]:")
                print(f"    - 关系位移(原始): {relationship_displacement.iloc[last_date_index]:.4f}")
                print(f"    - 关系动量(原始): {relationship_momentum.iloc[last_date_index]:.4f}")
                print(f"    - 关系位移强度(归一化): {bipolar_displacement_strength.iloc[last_date_index]:.4f}")
                print(f"    - 关系动量强度(归一化): {bipolar_momentum_strength.iloc[last_date_index]:.4f}")
                print("  [最终结果]:")
                print(f"    - 元分析最终分: {meta_score.iloc[last_date_index]:.4f}")
                print("--- [探针结束] ---\n")
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
        【V2.2 · 探针植入版】分裂型元关系诊断器
        - 核心升级: 同步采用全新的“关系位移/关系动量”模型进行核心计算。
        - 【优化】将 `bipolar_displacement_strength` 和 `bipolar_momentum_strength` 的归一化方式改为多时间维度自适应归一化。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print(f"\n--- [分裂元分析探针: {config.get('name')}] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 瞬时关系分: {relationship_score.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 关系位移强度(归一化): {bipolar_displacement_strength.iloc[last_date_index]:.4f}")
            print(f"    - 关系动量强度(归一化): {bipolar_momentum_strength.iloc[last_date_index]:.4f}")
            print(f"    - 元分析总分(分裂前): {meta_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 机会部分({opportunity_signal_name}): {opportunity_part.iloc[last_date_index]:.4f}")
            print(f"    - 风险部分({risk_signal_name}): {risk_part.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return states

    def _calculate_winner_conviction_relationship(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.3 · 探针增强版】“赢家信念”专属关系计算引擎
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [赢家信念探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 信号A ({signal_a_name}): {signal_a.iloc[last_date_index]:.4f}")
            print(f"    - 信号B ({signal_b_name}): {signal_b.iloc[last_date_index]:.4f}")
            print(f"    - 解毒剂 ({antidote_signal_name}): {signal_antidote.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 动量A(归一化): {momentum_a.iloc[last_date_index]:.4f}")
            print(f"    - 动量B(原始, 归一化): {momentum_b_raw.iloc[last_date_index]:.4f}")
            print(f"    - 动量解毒剂(归一化): {momentum_antidote.iloc[last_date_index]:.4f}")
            print(f"    - 动量B(修正后): {momentum_b_corrected.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 赢家信念关系分: {relationship_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        return relationship_score.clip(-1, 1)

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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
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
        【V4.1 · 探针植入版】计算“隐蔽吸筹”的专属关系分数。
        - 核心升级: 引入“筹码势能”公理作为背景过滤器和结果放大器，优先识别具备长期潜力的吸筹行为。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_STEALTH_ACCUMULATION (V4.1 · 探针植入版)...")
        df_index = df.index
        p_conf_behavioral = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_mtf = get_param_value(p_conf_behavioral.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        # 获取新公理及配置参数
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
        base_score = pd.concat([suppressive_score, consolidative_score], axis=1).max(axis=1).fillna(0.0)
        # 融合筹码势能
        potential_gate_mask = historical_potential > potential_gate
        potential_modulator = (1 + historical_potential * potential_amplifier)
        final_score = (base_score * potential_modulator).where(potential_gate_mask, 0.0)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [隐秘吸筹探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 筹码势能: {historical_potential.iloc[last_date_index]:.4f}")
            print(f"    - 价格趋势(归一化): {price_trend_norm.iloc[last_date_index]:.4f}")
            print(f"    - 稳定性: {stability_score.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 打压场景得分: {suppressive_score.iloc[last_date_index]:.4f}")
            print(f"    - 横盘场景得分: {consolidative_score.iloc[last_date_index]:.4f}")
            print(f"    - 基础得分(场景最大值): {base_score.iloc[last_date_index]:.4f}")
            print(f"    - 势能门控是否通过: {potential_gate_mask.iloc[last_date_index]}")
            print(f"    - 势能调节器: {potential_modulator.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 隐秘吸筹最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states["_DEBUG_accum_suppressive_score"] = suppressive_score
        self.strategy.atomic_states["_DEBUG_accum_consolidative_score"] = consolidative_score
        return final_score.clip(0, 1).astype(np.float32)

    def _calculate_panic_washout_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.1 · 探针植入版】计算“恐慌洗盘吸筹”的专属信号。
        - 核心升级: 引入“筹码势能”公理作为强门控条件，确保此高风险战术只在主力根基深厚时被识别。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_PANIC_WASHOUT_ACCUMULATION (V3.1 · 探针植入版)...")
        df_index = df.index
        # 获取新公理及配置参数
        historical_potential = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        potential_gate = config.get('historical_potential_gate', 0.0)
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
        base_score = (panic_score * absorption_score * repair_score).pow(1/3)
        # 融合筹码势能作为强门控
        potential_gate_mask = historical_potential > potential_gate
        final_score = base_score.where(washout_candidate_mask & potential_gate_mask, 0.0).fillna(0.0)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [恐慌洗盘吸筹探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 筹码势能: {historical_potential.iloc[last_date_index]:.4f}")
            print(f"    - 跌幅: {pct_change.iloc[last_date_index]:.4f}")
            print(f"    - 量能爆发: {volume_burst.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 恐慌分: {panic_score.iloc[last_date_index]:.4f}")
            print(f"    - 承接分: {absorption_score.iloc[last_date_index]:.4f}")
            print(f"    - 修复分: {repair_score.iloc[last_date_index]:.4f}")
            print(f"    - 基础分(三者融合): {base_score.iloc[last_date_index]:.4f}")
            print(f"    - 场景候选掩码: {washout_candidate_mask.iloc[last_date_index]}")
            print(f"    - 势能门控掩码: {potential_gate_mask.iloc[last_date_index]}")
            print("  [最终结果]:")
            print(f"    - 恐慌洗盘吸筹最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states["_DEBUG_washout_panic_score"] = panic_score
        self.strategy.atomic_states["_DEBUG_washout_absorption_score"] = absorption_score
        self.strategy.atomic_states["_DEBUG_washout_repair_score"] = repair_score
        return final_score.astype(np.float32)

    def _calculate_upthrust_washout(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.7 · 探针植入版】识别主力在拉升初期利用“高开低走”阴线进行的洗盘行为。
        - 核心清理: 移除了方法内的调试探针逻辑，净化日志输出。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_UPTHRUST_WASHOUT (V1.7 · 探针植入版)...")
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [上冲回落洗盘探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 趋势形态分: {trend_form_score.iloc[last_date_index]:.4f}")
            print(f"    - 21日乖离率: {bias_21.iloc[last_date_index]:.4f}")
            print(f"    - 量能爆发分: {volume_burst.iloc[last_date_index]:.4f}")
            print(f"    - 下影线强度: {lower_shadow_strength.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 上下文得分: {context_score.iloc[last_date_index]:.4f}")
            print(f"    - 行为得分(高开低走放量): {action_score.iloc[last_date_index]:.4f}")
            print(f"    - 内部结构得分: {internals_score.iloc[last_date_index]:.4f}")
            print(f"    - 洗盘真实性得分: {washout_authenticity_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 上冲回落洗盘最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states["_DEBUG_washout_context_score"] = context_score
        self.strategy.atomic_states["_DEBUG_washout_action_score"] = action_score
        self.strategy.atomic_states["_DEBUG_washout_internals_score"] = internals_score
        self.strategy.atomic_states["_DEBUG_washout_authenticity_score"] = washout_authenticity_score
        self.strategy.atomic_states["_DEBUG_washout_auth_bull_evidence"] = bullish_evidence
        self.strategy.atomic_states["_DEBUG_washout_auth_bear_evidence"] = bearish_evidence
        return final_score.astype(np.float32)

    def _calculate_accumulation_inflection(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.1 · 探针植入版】识别多日累积吸筹后，即将由“量变”引发“质变”的拉升拐点。
        - 核心升级: 彻底重构了“潜在势能”的计算逻辑。不再仅仅依赖于三种“已定性”的吸筹战术，
                      而是融合了包括“拆单吸筹强度”、“权力转移”在内的五种“全息证据”，
                      解决了因战术模板过于严苛而导致势能被低估的“大拐点悖论”，极大提升了信号的
                      灵敏度和准确性。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_ACCUMULATION_INFLECTION (V2.1 · 探针植入版)...")
        df_index = df.index
        accumulation_window = config.get('accumulation_window', 21)
        stealth_accum = self.strategy.atomic_states.get('PROCESS_META_STEALTH_ACCUMULATION', pd.Series(0.0, index=df_index))
        deceptive_accum = self.strategy.atomic_states.get('PROCESS_META_DECEPTIVE_ACCUMULATION', pd.Series(0.0, index=df_index))
        panic_washout_accum = self.strategy.atomic_states.get('PROCESS_META_PANIC_WASHOUT_ACCUMULATION', pd.Series(0.0, index=df_index))
        # 引入更全面的“全息证据”
        split_order_accum = self.strategy.atomic_states.get('PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', pd.Series(0.0, index=df_index))
        power_transfer_accum = self.strategy.atomic_states.get('PROCESS_META_POWER_TRANSFER', pd.Series(0.0, index=df_index)).clip(lower=0)
        # 融合五大“全息证据”来计算每日的吸筹强度
        daily_accumulation_strength = pd.concat([stealth_accum, deceptive_accum, panic_washout_accum, split_order_accum, power_transfer_accum], axis=1).max(axis=1)
        potential_energy_raw = daily_accumulation_strength.rolling(window=accumulation_window, min_periods=5).sum()
        potential_energy_score = normalize_score(potential_energy_raw, df_index, window=accumulation_window, ascending=True).clip(0, 1)
        price_slope_1d = self._get_safe_series(df, f'SLOPE_1_close_D', 0.0, method_name="_calculate_accumulation_inflection")
        volume_burst = self.strategy.atomic_states.get('SCORE_BEHAVIOR_VOLUME_BURST', pd.Series(0.0, index=df_index))
        closing_position = self._get_safe_series(df, 'closing_strength_index_D', 0.0, method_name="_calculate_accumulation_inflection")
        price_trigger = (price_slope_1d > 0).astype(float)
        volume_trigger = (volume_burst > 0.1).astype(float)
        kline_trigger = ((closing_position / 100).clip(0, 1))
        kinetic_trigger_score = (price_trigger * 0.4 + volume_trigger * 0.3 + kline_trigger * 0.3).clip(0, 1)
        final_score = (potential_energy_score * kinetic_trigger_score).fillna(0.0)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [吸筹末端拐点探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 当日综合吸筹强度: {daily_accumulation_strength.iloc[last_date_index]:.4f}")
            print(f"    - 价格斜率(1日): {price_slope_1d.iloc[last_date_index]:.4f}")
            print(f"    - 量能爆发分: {volume_burst.iloc[last_date_index]:.4f}")
            print(f"    - 收盘位置: {closing_position.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 累积势能(原始): {potential_energy_raw.iloc[last_date_index]:.4f}")
            print(f"    - 势能得分(归一化): {potential_energy_score.iloc[last_date_index]:.4f}")
            print(f"    - 动能触发得分: {kinetic_trigger_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 吸筹末端拐点最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
        self.strategy.atomic_states["_DEBUG_inflection_potential_energy"] = potential_energy_score
        self.strategy.atomic_states["_DEBUG_inflection_kinetic_trigger"] = kinetic_trigger_score
        return final_score.astype(np.float32)

    def _calculate_split_order_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V1.5 · 探针植入版】计算“拆单吸筹强度”的专属信号。
        - 核心清理: 移除了方法内的调试探针逻辑，净化日志输出。
        - 新增功能: 植入“真理探针”，用于在指定日期输出关键计算过程。
        """
        print("    -> [过程层] 正在计算 PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY (V1.5 · 探针植入版)...")
        df_index = df.index
        raw_intensity = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name="_calculate_split_order_accumulation")
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name="_calculate_split_order_accumulation")
        scene_mask = pct_change <= 0.02
        normalized_score = (raw_intensity / 100).clip(0, 1)
        final_score = normalized_score.where(scene_mask, 0.0).fillna(0.0)
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
        if not df.empty and df.index[-1].strftime('%Y-%m-%d') in probe_dates:
            print("\n--- [拆单吸筹强度探针] ---")
            last_date_index = -1
            print(f"日期: {df.index[last_date_index].strftime('%Y-%m-%d')}")
            print("  [输入原料]:")
            print(f"    - 原始强度值: {raw_intensity.iloc[last_date_index]:.4f}")
            print(f"    - 当日涨跌幅: {pct_change.iloc[last_date_index]:.4f}")
            print("  [关键计算]:")
            print(f"    - 场景掩码(涨幅<=2%): {scene_mask.iloc[last_date_index]}")
            print(f"    - 强度归一化得分: {normalized_score.iloc[last_date_index]:.4f}")
            print("  [最终结果]:")
            print(f"    - 拆单吸筹强度最终分: {final_score.iloc[last_date_index]:.4f}")
            print("--- [探针结束] ---\n")
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
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
        # [修改] 修正探针日期获取逻辑
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates = get_param_value(debug_params.get('probe_dates'), [])
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



