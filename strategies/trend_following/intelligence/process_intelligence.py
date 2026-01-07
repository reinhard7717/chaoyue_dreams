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
        method_name = "_calculate_main_force_control_relationship"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算主力控盘关系..."] = ""
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
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        # --- 原始数据获取 ---
        close_price = self._get_safe_series(df, 'close_D', method_name=method_name)
        control_solidity_raw = self._get_safe_series(df, 'control_solidity_index_D', 0.0, method_name=method_name)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "close_D": close_price,
            "control_solidity_index_D": control_solidity_raw,
            "main_force_net_flow_calibrated_D": main_force_net_flow,
            "flow_credibility_index_D": flow_credibility_raw
        }
        # --- 1. 传统控盘度计算 ---
        ema13 = ta.ema(close=close_price, length=13, append=False)
        if ema13 is None:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} EMA_13 计算失败，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        varn1 = ta.ema(close=ema13, length=13, append=False)
        if varn1 is None:
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} VARN1 计算失败，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        prev_varn1 = varn1.shift(1).replace(0, np.nan)
        kongpan_raw = (varn1 - prev_varn1) / prev_varn1 * 1000
        _temp_debug_values["传统控盘度计算"] = {
            "ema13": ema13,
            "varn1": varn1,
            "prev_varn1": prev_varn1,
            "kongpan_raw": kongpan_raw
        }
        # --- 2. 归一化处理 ---
        traditional_control_score = self._normalize_series(kongpan_raw, df_index, bipolar=True)
        # 结构控盘度：使用MTF融合信号
        mtf_structural_control_score = self._get_mtf_slope_accel_score(df, 'control_solidity_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 主力资金流分：使用MTF融合信号
        mtf_main_force_flow_score = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        flow_credibility_norm = self._normalize_series(flow_credibility_raw, df_index, bipolar=False)
        _temp_debug_values["归一化处理"] = {
            "traditional_control_score": traditional_control_score,
            "mtf_structural_control_score": mtf_structural_control_score,
            "mtf_main_force_flow_score": mtf_main_force_flow_score,
            "flow_credibility_norm": flow_credibility_norm
        }
        # --- 3. 融合控盘分 ---
        fused_control_score = (traditional_control_score * 0.4 + mtf_structural_control_score * 0.6).clip(-1, 1)
        _temp_debug_values["融合控盘分"] = {
            "fused_control_score": fused_control_score
        }
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
        _temp_debug_values["控盘杠杆模型"] = {
            "mf_inflow_validation": mf_inflow_validation,
            "control_leverage": control_leverage
        }
        # --- 5. 最终控盘分数 ---
        final_control_score = (mtf_main_force_flow_score * control_leverage).clip(-1, 1)
        _temp_debug_values["最终控盘分数"] = {
            "final_control_score": final_control_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 传统控盘度计算 ---"] = ""
            for key, series in _temp_debug_values["传统控盘度计算"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合控盘分 ---"] = ""
            for key, series in _temp_debug_values["融合控盘分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 控盘杠杆模型 ---"] = ""
            for key, series in _temp_debug_values["控盘杠杆模型"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终控盘分数 ---"] = ""
            for key, series in _temp_debug_values["最终控盘分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力控盘关系诊断完成，最终分值: {final_control_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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
        elif signal_name == 'PROCESS_META_PRICE_VOLUME_DYNAMICS': # <--- 修改此处
            meta_score = self._calculate_price_volume_dynamics(df, config) # <--- 修改此处
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
            meta_score = self._calculate_process_wash_out_rebound(df, offensive_absorption_intent, config)
        elif signal_name == 'PROCESS_META_COVERT_ACCUMULATION':
            meta_score = self._calculate_process_covert_accumulation(df, config)
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

    def _calculate_main_force_rally_intent(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V11.3 · 承接强化与风险情境化版】计算“主力拉升意图”的专属关系分数。
        - 核心强化: 显著增强“派发吸收强度”的积极作用，使其更直接地贡献于看涨意图，并提高其在攻击性中的权重。
        - 核心优化: 进一步情境化“派发风险”，提高“派发吸收强度反向”在派发风险中的权重，以更有效地对冲派发烈度。
        - 核心调整: 进一步调整风险惩罚的非线性函数参数，使其在面对中等风险时更加温和。
        - 核心升级: 引入“派发吸收强度”指标，对冲“派发烈度”带来的负面影响，更全面反映市场承接能力。
        - 核心优化: 引入“派发情境衰减器”，根据当日涨幅动态削弱“派发烈度”对看跌意图的贡献。
        - 核心升级: 严格限制仅使用数据层提供的原始指标，移除所有情报层生成的SCORE_FOUNDATION_AXIOM_*信号。
        - 动态权重机制精细化: 综合市场稳定性、市场情绪（基于原始零售/主力行为）、流动性（基于原始订单簿/资金流）等原始情境因子，自适应调整各维度权重。
        - 非线性融合优化: 基础看涨意图的加权幂平均幂次p根据原始情境因子动态调整。
        - 风险审判深化: 引入更多微观风险信号，并优化非线性风险惩罚函数。
        - 攻击性、控制力、障碍清除维度进一步补充微观结构和行为信号。
        - 情境调节器扩展，融入更多行为心理和市场情绪因子（基于原始数据）。
        - 新增历史记忆与上下文机制：引入长期主力资金累计记忆、筹码集中度稳定性记忆和长期趋势强度上下文，以更全面地评估市场状态。
        - 调整长期主力资金累计记忆和长期趋势强度的周期为21日，以更好地捕捉中短期趋势。
        - 探针优化: 增加详细的调试输出，覆盖所有新增的原始数据、归一化过程、中间计算节点和最终结果。
        """
        method_name = "_calculate_main_force_rally_intent"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算主力拉升意图..."] = ""
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_param_value(self.strategy.config.get('structural_ultimate_params'), {})
        p_mtf = get_param_value(p_conf_structural_ultimate.get('mtf_normalization_weights'), {})
        actual_mtf_weights = get_param_value(p_mtf.get('default'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_slope_accel_weights = config.get('mtf_slope_accel_weights', {"slope_periods": {"5": 0.6, "13": 0.4}, "accel_periods": {"5": 0.7, "13": 0.3}})
        # 获取历史上下文参数
        historical_context_params = config.get('historical_context_params', {})
        hc_enabled = get_param_value(historical_context_params.get('enabled'), False)
        cumulative_mf_flow_window = get_param_value(historical_context_params.get('cumulative_mf_flow_window'), 21) # 调整为21
        cumulative_mf_flow_modulator_factor = get_param_value(historical_context_params.get('cumulative_mf_flow_modulator_factor'), 0.1)
        chip_concentration_stability_window = get_param_value(historical_context_params.get('chip_concentration_stability_window'), 55)
        chip_concentration_stability_modulator_factor = get_param_value(historical_context_params.get('chip_concentration_stability_modulator_factor'), 0.05)
        long_term_trend_slope_period = get_param_value(historical_context_params.get('long_term_trend_slope_period'), 21) # 调整为21
        hc_mtf_weights_long = get_param_value(historical_context_params.get('mtf_weights_long'), {"55": 0.5, "89": 0.3, "144": 0.2})
        hc_mtf_weights_medium = get_param_value(historical_context_params.get('mtf_weights_medium'), {"21": 0.4, "34": 0.3, "55": 0.3})
        long_term_trend_modulator_factor = get_param_value(historical_context_params.get('long_term_trend_modulator_factor'), 0.15)
        # 获取主力拉升意图合成参数块
        rally_intent_synthesis_params = config.get('rally_intent_synthesis_params', {})
        # 获取风险惩罚参数
        risk_sensitivity = get_param_value(rally_intent_synthesis_params.get('risk_sensitivity'), 5.0)
        sigmoid_center = get_param_value(rally_intent_synthesis_params.get('sigmoid_center'), 0.3)

        # --- V10.0 原始信号列表 (严格限制为数据层提供) ---
        required_signals = [
            'pct_change_D', 'main_force_net_flow_calibrated_D', 'main_force_slippage_index_D',
            'upward_impulse_purity_D', 'volume_ratio_D', 'control_solidity_index_D',
            'main_force_cost_advantage_D', 'SLOPE_5_winner_concentration_90pct_D',
            'dominant_peak_solidity_D', 'active_buying_support_D', 'pressure_rejection_strength_D',
            'profit_realization_quality_D', 
            'distribution_at_peak_intensity_D', 'upper_shadow_selling_pressure_D', 
            'flow_credibility_index_D', 'chip_health_score_D', 'retail_fomo_premium_index_D',
            'SLOPE_21_close_D', 'ACCEL_21_close_D', 'SLOPE_34_close_D', 'ACCEL_34_close_D',
            'buy_sweep_intensity_D', 'main_force_buy_ofi_D', 'main_force_t0_buy_efficiency_D',
            'order_book_imbalance_D', 'micro_price_impact_asymmetry_D', 'constructive_turnover_ratio_D',
            'upward_impulse_strength_D', 'vwap_buy_control_strength_D', 'mf_cost_zone_buy_intent_D',
            'chip_fault_blockage_ratio_D', 'vacuum_traversal_efficiency_D', 'vacuum_zone_magnitude_D',
            'dip_buy_absorption_strength_D', 'rally_buy_support_weakness_D', 'covert_distribution_signal_D',
            'deception_lure_short_intensity_D', 'rally_distribution_pressure_D', 'exhaustion_risk_index_D',
            'asymmetric_friction_index_D', 'volatility_expansion_ratio_D', 'market_sentiment_score_D',
            'structural_tension_index_D', 'trend_vitality_index_D', 'liquidity_authenticity_score_D',
            'order_book_clearing_rate_D', 'sell_sweep_intensity_D', 'main_force_flow_gini_D',
            'microstructure_efficiency_index_D', 'imbalance_effectiveness_D', 'auction_showdown_score_D',
            'closing_conviction_score_D', 'intraday_energy_density_D', 'intraday_thrust_purity_D',
            'price_thrust_divergence_D', 'trend_efficiency_ratio_D', 'loser_concentration_90pct_D',
            'winner_loser_momentum_D', 'cost_structure_skewness_D', 'cost_gini_coefficient_D',
            'mf_vpoc_premium_D', 'character_score_D', 'signal_conviction_score_D',
            'touch_conviction_score_D', 'gathering_by_chasing_D', 'gathering_by_support_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'ADX_14_D', # 用于动态权重
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D', # 订单流效率
            'auction_closing_position_D', 'auction_impact_score_D', 'auction_intent_signal_D', # 集合竞价
            'order_book_liquidity_supply_D', 'liquidity_slope_D', # 流动性深度
            'peak_mass_transfer_rate_D', # 筹码转移
            'mf_cost_zone_defense_intent_D', # 成本区防守
            'bid_side_liquidity_D', 'ask_side_liquidity_D', # 订单簿流动性
            'retail_panic_surrender_index_D', # 零售恐慌投降
            'main_force_activity_ratio_D', 'main_force_conviction_index_D', # 主力活跃度与信念
            'main_force_execution_alpha_D', 'main_force_flow_directionality_D', # 主力执行效率与流向
            'main_force_on_peak_buy_flow_D', 'main_force_on_peak_sell_flow_D', # 主力峰值买卖流
            'main_force_t0_efficiency_D', 'main_force_t0_sell_efficiency_D', # 主力T0效率
            'main_force_vwap_down_guidance_D', 'main_force_vwap_up_guidance_D', # 主力VWAP引导
            'market_impact_cost_D', # 市场冲击成本
            'opening_buy_strength_D', 'opening_sell_strength_D', # 开盘强度
            'closing_strength_index_D', # 收盘强度
            'total_buy_amount_calibrated_D', 'total_sell_amount_calibrated_D', # 总买卖金额
            'wash_trade_intensity_D', # 洗盘强度
            'winner_profit_margin_avg_D', 'loser_loss_margin_avg_D', # 赢家利润，输家亏损
            'total_winner_rate_D', 'total_loser_rate_D', # 赢家输家比例
            'impulse_quality_ratio_D', 'thrust_efficiency_score_D', # 脉冲质量与推力效率
            'platform_conviction_score_D', 'platform_high_D', 'platform_low_D', # 平台信念与高低
            'breakout_quality_score_D', 'breakout_readiness_score_D', 'breakthrough_conviction_score_D', # 突破相关
            'defense_solidity_score_D', 'support_validation_strength_D', # 防守与支撑
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D', # 隐蔽吸筹
            'deception_index_D', 'deception_lure_long_intensity_D', # 欺骗
            'equilibrium_compression_index_D', 'final_charge_intensity_D', # 均衡压缩与最终冲刺
            'floating_chip_cleansing_efficiency_D', # 浮筹清洗效率
            'hidden_accumulation_intensity_D', 'internal_accumulation_intensity_D', # 隐藏吸筹
            'intraday_posture_score_D', 'opening_gap_defense_strength_D', # 日内姿态
            'panic_buy_absorption_contribution_D', 'panic_sell_volume_contribution_D', 'panic_selling_cascade_D', # 恐慌
            'peak_control_transfer_D', 'peak_separation_ratio_D', # 峰值控制
            'price_reversion_velocity_D', 'pullback_depth_ratio_D', # 价格回归与回调
            'quality_score_D', 'reversal_conviction_rate_D', 'reversal_power_index_D', 'reversal_recovery_rate_D', # 反转
            'risk_reward_profile_D', 'shock_conviction_score_D', # 风险回报与冲击
            'strategic_phase_score_D', 'structural_entropy_change_D', 'structural_leverage_D', 'structural_node_count_D', 'structural_potential_score_D', # 结构
            'support_validation_score_D', 'supportive_distribution_intensity_D', # 支撑与派发
            'trend_acceleration_score_D', 'trend_alignment_index_D', 'trend_asymmetry_index_D', 'trend_conviction_score_D', # 趋势
            'value_area_migration_D', 'value_area_overlap_pct_D', # 价值区域
            'volatility_asymmetry_index_D', 'volume_burstiness_index_D', 'volume_structure_skew_D', # 波动率与成交量
            'vpin_score_D', 'vwap_control_strength_D', 'vwap_cross_down_intensity_D', 'vwap_cross_up_intensity_D', 'vwap_crossing_intensity_D', 'vwap_mean_reversion_corr_D', 'vwap_sell_control_strength_D', # VWAP
            'winner_stability_index_D', # 赢家稳定性
            'winner_concentration_90pct_D', # 用于筹码集中度稳定性
            f'SLOPE_{long_term_trend_slope_period}_close_D', # 长期趋势斜率，现在是21日
            'absorption_of_distribution_intensity_D' # 新增：派发吸收强度
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        # 遍历所有原始信号，为其添加MTF斜率和加速度版本
        base_signals_for_mtf = [s.replace('_D', '') for s in required_signals if not s.startswith(('SLOPE_', 'ACCEL_')) and s.endswith('_D')]
        for base_sig_name in base_signals_for_mtf:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                required_signals.append(f'SLOPE_{period_str}_{base_sig_name}_D')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                required_signals.append(f'ACCEL_{period_str}_{base_sig_name}_D')
        # 确保所有需要的信号都在df中
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        is_limit_up_day = df.apply(lambda row: is_limit_up(row), axis=1)
        # --- 原始数据获取 ---
        pct_change = self._get_safe_series(df, 'pct_change_D', 0.0, method_name=method_name)
        main_force_net_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
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
        close_price = self._get_safe_series(df, 'close_D', 0.0, method_name=method_name) # 用于计算从高点回落
        prev_day_pct_change = pct_change.shift(1).fillna(0)
        slope_21_close = self._get_safe_series(df, 'SLOPE_21_close_D', 0.0, method_name=method_name)
        accel_21_close = self._get_safe_series(df, 'ACCEL_21_close_D', 0.0, method_name=method_name)
        slope_34_close = self._get_safe_series(df, 'SLOPE_34_close_D', 0.0, method_name=method_name)
        accel_34_close = self._get_safe_series(df, 'ACCEL_34_close_D', 0.0, method_name=method_name)
        # V10.0 新增原始信号
        buy_sweep_intensity = self._get_safe_series(df, 'buy_sweep_intensity_D', 0.0, method_name=method_name)
        main_force_buy_ofi = self._get_safe_series(df, 'main_force_buy_ofi_D', 0.0, method_name=method_name)
        main_force_t0_buy_efficiency = self._get_safe_series(df, 'main_force_t0_buy_efficiency_D', 0.0, method_name=method_name)
        order_book_imbalance = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name)
        micro_price_impact_asymmetry = self._get_safe_series(df, 'micro_price_impact_asymmetry_D', 0.0, method_name=method_name)
        constructive_turnover = self._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name)
        upward_impulse_strength = self._get_safe_series(df, 'upward_impulse_strength_D', 0.0, method_name=method_name)
        vwap_buy_control_strength = self._get_safe_series(df, 'vwap_buy_control_strength_D', 0.0, method_name=method_name)
        mf_cost_zone_buy_intent = self._get_safe_series(df, 'mf_cost_zone_buy_intent_D', 0.0, method_name=method_name)
        chip_fault_blockage_ratio = self._get_safe_series(df, 'chip_fault_blockage_ratio_D', 0.0, method_name=method_name)
        vacuum_traversal_efficiency = self._get_safe_series(df, 'vacuum_traversal_efficiency_D', 0.0, method_name=method_name)
        vacuum_zone_magnitude = self._get_safe_series(df, 'vacuum_zone_magnitude_D', 0.0, method_name=method_name)
        dip_buy_absorption_strength = self._get_safe_series(df, 'dip_buy_absorption_strength_D', 0.0, method_name=method_name)
        rally_buy_support_weakness = self._get_safe_series(df, 'rally_buy_support_weakness_D', 0.0, method_name=method_name)
        covert_distribution = self._get_safe_series(df, 'covert_distribution_signal_D', 0.0, method_name=method_name)
        deception_lure_short = self._get_safe_series(df, 'deception_lure_short_intensity_D', 0.0, method_name=method_name)
        rally_distribution_pressure = self._get_safe_series(df, 'rally_distribution_pressure_D', 0.0, method_name=method_name)
        exhaustion_risk = self._get_safe_series(df, 'exhaustion_risk_index_D', 0.0, method_name=method_name)
        asymmetric_friction = self._get_safe_series(df, 'asymmetric_friction_index_D', 0.0, method_name=method_name)
        volatility_expansion = self._get_safe_series(df, 'volatility_expansion_ratio_D', 0.0, method_name=method_name)
        market_sentiment = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        structural_tension = self._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name)
        trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        liquidity_authenticity = self._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name)
        order_book_clearing_rate = self._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name=method_name)
        sell_sweep_intensity = self._get_safe_series(df, 'sell_sweep_intensity_D', 0.0, method_name=method_name)
        main_force_flow_gini = self._get_safe_series(df, 'main_force_flow_gini_D', 0.0, method_name=method_name)
        microstructure_efficiency = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name)
        imbalance_effectiveness = self._get_safe_series(df, 'imbalance_effectiveness_D', 0.0, method_name=method_name)
        auction_showdown = self._get_safe_series(df, 'auction_showdown_score_D', 0.0, method_name=method_name)
        closing_conviction = self._get_safe_series(df, 'closing_conviction_score_D', 0.0, method_name=method_name)
        intraday_energy_density = self._get_safe_series(df, 'intraday_energy_density_D', 0.0, method_name=method_name)
        intraday_thrust_purity = self._get_safe_series(df, 'intraday_thrust_purity_D', 0.0, method_name=method_name)
        price_thrust_divergence = self._get_safe_series(df, 'price_thrust_divergence_D', 0.0, method_name=method_name)
        trend_efficiency_ratio = self._get_safe_series(df, 'trend_efficiency_ratio_D', 0.0, method_name=method_name)
        loser_concentration_90pct = self._get_safe_series(df, 'loser_concentration_90pct_D', 0.0, method_name=method_name)
        winner_loser_momentum = self._get_safe_series(df, 'winner_loser_momentum_D', 0.0, method_name=method_name)
        cost_structure_skewness = self._get_safe_series(df, 'cost_structure_skewness_D', 0.0, method_name=method_name)
        cost_gini_coefficient = self._get_safe_series(df, 'cost_gini_coefficient_D', 0.0, method_name=method_name)
        mf_vpoc_premium = self._get_safe_series(df, 'mf_vpoc_premium_D', 0.0, method_name=method_name)
        character_score = self._get_safe_series(df, 'character_score_D', 0.0, method_name=method_name)
        signal_conviction_score = self._get_safe_series(df, 'signal_conviction_score_D', 0.0, method_name=method_name)
        touch_conviction_score = self._get_safe_series(df, 'touch_conviction_score_D', 0.0, method_name=method_name)
        gathering_by_chasing = self._get_safe_series(df, 'gathering_by_chasing_D', 0.0, method_name=method_name)
        gathering_by_support = self._get_safe_series(df, 'gathering_by_support_D', 0.0, method_name=method_name)
        volatility_instability = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        adx = self._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        buy_flow_efficiency = self._get_safe_series(df, 'buy_flow_efficiency_index_D', 0.0, method_name=method_name)
        sell_flow_efficiency = self._get_safe_series(df, 'sell_flow_efficiency_index_D', 0.0, method_name=method_name)
        auction_closing_position = self._get_safe_series(df, 'auction_closing_position_D', 0.0, method_name=method_name)
        auction_impact_score = self._get_safe_series(df, 'auction_impact_score_D', 0.0, method_name=method_name)
        auction_intent_signal = self._get_safe_series(df, 'auction_intent_signal_D', 0.0, method_name=method_name)
        order_book_liquidity_supply = self._get_safe_series(df, 'order_book_liquidity_supply_D', 0.0, method_name=method_name)
        liquidity_slope = self._get_safe_series(df, 'liquidity_slope_D', 0.0, method_name=method_name)
        peak_mass_transfer_rate = self._get_safe_series(df, 'peak_mass_transfer_rate_D', 0.0, method_name=method_name)
        mf_cost_zone_defense_intent = self._get_safe_series(df, 'mf_cost_zone_defense_intent_D', 0.0, method_name=method_name)
        bid_side_liquidity = self._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name=method_name)
        ask_side_liquidity = self._get_safe_series(df, 'ask_side_liquidity_D', 0.0, method_name=method_name)
        retail_panic_surrender = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        main_force_activity_ratio = self._get_safe_series(df, 'main_force_activity_ratio_D', 0.0, method_name=method_name)
        main_force_conviction_index = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        main_force_execution_alpha = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name=method_name)
        main_force_flow_directionality = self._get_safe_series(df, 'main_force_flow_directionality_D', 0.0, method_name=method_name)
        main_force_on_peak_buy_flow = self._get_safe_series(df, 'main_force_on_peak_buy_flow_D', 0.0, method_name=method_name)
        main_force_on_peak_sell_flow = self._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name=method_name)
        main_force_t0_efficiency = self._get_safe_series(df, 'main_force_t0_efficiency_D', 0.0, method_name=method_name)
        main_force_t0_sell_efficiency = self._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name=method_name)
        main_force_vwap_down_guidance = self._get_safe_series(df, 'main_force_vwap_down_guidance_D', 0.0, method_name=method_name)
        main_force_vwap_up_guidance = self._get_safe_series(df, 'main_force_vwap_up_guidance_D', 0.0, method_name=method_name)
        market_impact_cost = self._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name=method_name)
        opening_buy_strength = self._get_safe_series(df, 'opening_buy_strength_D', 0.0, method_name=method_name)
        opening_sell_strength = self._get_safe_series(df, 'opening_sell_strength_D', 0.0, method_name=method_name)
        closing_strength_index = self._get_safe_series(df, 'closing_strength_index_D', 0.0, method_name=method_name)
        total_buy_amount_calibrated = self._get_safe_series(df, 'total_buy_amount_calibrated_D', 0.0, method_name=method_name)
        total_sell_amount_calibrated = self._get_safe_series(df, 'total_sell_amount_calibrated_D', 0.0, method_name=method_name)
        wash_trade_intensity = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        winner_profit_margin_avg = self._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name=method_name)
        loser_loss_margin_avg = self._get_safe_series(df, 'loser_loss_margin_avg_D', 0.0, method_name=method_name)
        total_winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name=method_name)
        total_loser_rate = self._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name)
        impulse_quality_ratio = self._get_safe_series(df, 'impulse_quality_ratio_D', 0.0, method_name=method_name)
        thrust_efficiency_score = self._get_safe_series(df, 'thrust_efficiency_score_D', 0.0, method_name=method_name)
        platform_conviction_score = self._get_safe_series(df, 'platform_conviction_score_D', 0.0, method_name=method_name)
        platform_high = self._get_safe_series(df, 'platform_high_D', 0.0, method_name=method_name)
        platform_low = self._get_safe_series(df, 'platform_low_D', 0.0, method_name=method_name)
        breakout_quality_score = self._get_safe_series(df, 'breakout_quality_score_D', 0.0, method_name=method_name)
        breakout_readiness_score = self._get_safe_series(df, 'breakout_readiness_score_D', 0.0, method_name=method_name)
        breakthrough_conviction_score = self._get_safe_series(df, 'breakthrough_conviction_score_D', 0.0, method_name=method_name)
        defense_solidity_score = self._get_safe_series(df, 'defense_solidity_score_D', 0.0, method_name=method_name)
        support_validation_strength = self._get_safe_series(df, 'support_validation_strength_D', 0.0, method_name=method_name)
        covert_accumulation_signal = self._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name=method_name)
        suppressive_accumulation_intensity = self._get_safe_series(df, 'suppressive_accumulation_intensity_D', 0.0, method_name=method_name)
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        deception_lure_long_intensity = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        equilibrium_compression_index = self._get_safe_series(df, 'equilibrium_compression_index_D', 0.0, method_name=method_name)
        final_charge_intensity = self._get_safe_series(df, 'final_charge_intensity_D', 0.0, method_name=method_name)
        floating_chip_cleansing_efficiency = self._get_safe_series(df, 'floating_chip_cleansing_efficiency_D', 0.0, method_name=method_name)
        hidden_accumulation_intensity = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        internal_accumulation_intensity = self._get_safe_series(df, 'internal_accumulation_intensity_D', 0.0, method_name=method_name)
        intraday_posture_score = self._get_safe_series(df, 'intraday_posture_score_D', 0.0, method_name=method_name)
        opening_gap_defense_strength = self._get_safe_series(df, 'opening_gap_defense_strength_D', 0.0, method_name=method_name)
        panic_buy_absorption_contribution = self._get_safe_series(df, 'panic_buy_absorption_contribution_D', 0.0, method_name=method_name)
        panic_sell_volume_contribution = self._get_safe_series(df, 'panic_sell_volume_contribution_D', 0.0, method_name=method_name)
        panic_selling_cascade = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name=method_name)
        peak_control_transfer = self._get_safe_series(df, 'peak_control_transfer_D', 0.0, method_name=method_name)
        peak_separation_ratio = self._get_safe_series(df, 'peak_separation_ratio_D', 0.0, method_name=method_name)
        price_reversion_velocity = self._get_safe_series(df, 'price_reversion_velocity_D', 0.0, method_name=method_name)
        pullback_depth_ratio = self._get_safe_series(df, 'pullback_depth_ratio_D', 0.0, method_name=method_name)
        quality_score = self._get_safe_series(df, 'quality_score_D', 0.0, method_name=method_name)
        reversal_conviction_rate = self._get_safe_series(df, 'reversal_conviction_rate_D', 0.0, method_name=method_name)
        reversal_power_index = self._get_safe_series(df, 'reversal_power_index_D', 0.0, method_name=method_name)
        reversal_recovery_rate = self._get_safe_series(df, 'reversal_recovery_rate_D', 0.0, method_name=method_name)
        risk_reward_profile = self._get_safe_series(df, 'risk_reward_profile_D', 0.0, method_name=method_name)
        shock_conviction_score = self._get_safe_series(df, 'shock_conviction_score_D', 0.0, method_name=method_name)
        strategic_phase_score = self._get_safe_series(df, 'strategic_phase_score_D', 0.0, method_name=method_name)
        structural_entropy_change = self._get_safe_series(df, 'structural_entropy_change_D', 0.0, method_name=method_name)
        structural_leverage = self._get_safe_series(df, 'structural_leverage_D', 0.0, method_name=method_name)
        structural_node_count = self._get_safe_series(df, 'structural_node_count_D', 0.0, method_name=method_name)
        structural_potential_score = self._get_safe_series(df, 'structural_potential_score_D', 0.0, method_name=method_name)
        support_validation_score = self._get_safe_series(df, 'support_validation_score_D', 0.0, method_name=method_name)
        supportive_distribution_intensity = self._get_safe_series(df, 'supportive_distribution_intensity_D', 0.0, method_name=method_name)
        trend_acceleration_score = self._get_safe_series(df, 'trend_acceleration_score_D', 0.0, method_name=method_name)
        trend_alignment_index = self._get_safe_series(df, 'trend_alignment_index_D', 0.0, method_name=method_name)
        trend_asymmetry_index = self._get_safe_series(df, 'trend_asymmetry_index_D', 0.0, method_name=method_name)
        trend_conviction_score = self._get_safe_series(df, 'trend_conviction_score_D', 0.0, method_name=method_name)
        value_area_migration = self._get_safe_series(df, 'value_area_migration_D', 0.0, method_name=method_name)
        value_area_overlap_pct = self._get_safe_series(df, 'value_area_overlap_pct_D', 0.0, method_name=method_name)
        volatility_asymmetry_index = self._get_safe_series(df, 'volatility_asymmetry_index_D', 0.0, method_name=method_name)
        volume_burstiness_index = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name)
        volume_structure_skew = self._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name=method_name)
        vpin_score = self._get_safe_series(df, 'vpin_score_D', 0.0, method_name=method_name)
        vwap_control_strength = self._get_safe_series(df, 'vwap_control_strength_D', 0.0, method_name=method_name)
        vwap_cross_down_intensity = self._get_safe_series(df, 'vwap_cross_down_intensity_D', 0.0, method_name=method_name)
        vwap_cross_up_intensity = self._get_safe_series(df, 'vwap_cross_up_intensity_D', 0.0, method_name=method_name)
        vwap_crossing_intensity = self._get_safe_series(df, 'vwap_crossing_intensity_D', 0.0, method_name=method_name)
        vwap_mean_reversion_corr = self._get_safe_series(df, 'vwap_mean_reversion_corr_D', 0.0, method_name=method_name)
        vwap_sell_control_strength = self._get_safe_series(df, 'vwap_sell_control_strength_D', 0.0, method_name=method_name)
        winner_stability_index = self._get_safe_series(df, 'winner_stability_index_D', 0.0, method_name=method_name)
        # V11.0 新增原始信号
        winner_concentration_90pct = self._get_safe_series(df, 'winner_concentration_90pct_D', 0.0, method_name=method_name)
        long_term_trend_slope = self._get_safe_series(df, f'SLOPE_{long_term_trend_slope_period}_close_D', 0.0, method_name=method_name) # 使用新的周期
        # V11.2 新增原始信号
        absorption_of_distribution_intensity = self._get_safe_series(df, 'absorption_of_distribution_intensity_D', 0.0, method_name=method_name)

        _temp_debug_values["原始信号值"] = {
            "pct_change_D": pct_change,
            "main_force_net_flow_calibrated_D": main_force_net_flow,
            "main_force_slippage_index_D": main_force_slippage,
            "upward_impulse_purity_D": upward_impulse_purity,
            "volume_ratio_D": volume_ratio,
            "control_solidity_index_D": control_solidity,
            "main_force_cost_advantage_D": cost_advantage,
            "SLOPE_5_winner_concentration_90pct_D": concentration_slope,
            "dominant_peak_solidity_D": peak_solidity,
            "active_buying_support_D": active_buying_support,
            "pressure_rejection_strength_D": pressure_rejection,
            "profit_realization_quality_D": profit_realization_quality,
            "distribution_at_peak_intensity_D": distribution_at_peak_intensity,
            "upper_shadow_selling_pressure_D": upper_shadow_selling_pressure,
            "flow_credibility_index_D": flow_credibility,
            "chip_health_score_D": chip_health,
            "retail_fomo_premium_index_D": retail_fomo,
            "SLOPE_21_close_D": slope_21_close,
            "ACCEL_21_close_D": accel_21_close,
            "SLOPE_34_close_D": slope_34_close,
            "ACCEL_34_close_D": accel_34_close,
            # 移除情报层信号
            # "SCORE_FOUNDATION_AXIOM_RELATIVE_STRENGTH": relative_strength,
            # "SCORE_FF_AXIOM_CAPITAL_SIGNATURE": capital_signature,
            # 新增原始信号调试
            "buy_sweep_intensity_D": buy_sweep_intensity,
            "main_force_buy_ofi_D": main_force_buy_ofi,
            "main_force_t0_buy_efficiency_D": main_force_t0_buy_efficiency,
            "order_book_imbalance_D": order_book_imbalance,
            "micro_price_impact_asymmetry_D": micro_price_impact_asymmetry,
            "constructive_turnover_ratio_D": constructive_turnover,
            "upward_impulse_strength_D": upward_impulse_strength,
            "vwap_buy_control_strength_D": vwap_buy_control_strength,
            "mf_cost_zone_buy_intent_D": mf_cost_zone_buy_intent,
            "chip_fault_blockage_ratio_D": chip_fault_blockage_ratio,
            "vacuum_traversal_efficiency_D": vacuum_traversal_efficiency,
            "vacuum_zone_magnitude_D": vacuum_zone_magnitude,
            "dip_buy_absorption_strength_D": dip_buy_absorption_strength,
            "rally_buy_support_weakness_D": rally_buy_support_weakness,
            "covert_distribution_signal_D": covert_distribution,
            "deception_lure_short_intensity_D": deception_lure_short,
            "rally_distribution_pressure_D": rally_distribution_pressure,
            "exhaustion_risk_index_D": exhaustion_risk,
            "asymmetric_friction_index_D": asymmetric_friction,
            "volatility_expansion_ratio_D": volatility_expansion,
            "market_sentiment_score_D": market_sentiment,
            "structural_tension_index_D": structural_tension,
            "trend_vitality_index_D": trend_vitality,
            "liquidity_authenticity_score_D": liquidity_authenticity,
            "order_book_clearing_rate_D": order_book_clearing_rate,
            "sell_sweep_intensity_D": sell_sweep_intensity,
            "main_force_flow_gini_D": main_force_flow_gini,
            "microstructure_efficiency_index_D": microstructure_efficiency,
            "imbalance_effectiveness_D": imbalance_effectiveness,
            "auction_showdown_score_D": auction_showdown,
            "closing_conviction_score_D": closing_conviction,
            "intraday_energy_density_D": intraday_energy_density,
            "intraday_thrust_purity_D": intraday_thrust_purity,
            "price_thrust_divergence_D": price_thrust_divergence,
            "trend_efficiency_ratio_D": trend_efficiency_ratio,
            "loser_concentration_90pct_D": loser_concentration_90pct,
            "winner_loser_momentum_D": winner_loser_momentum,
            "cost_structure_skewness_D": cost_structure_skewness,
            "cost_gini_coefficient_D": cost_gini_coefficient,
            "mf_vpoc_premium_D": mf_vpoc_premium,
            "character_score_D": character_score,
            "signal_conviction_score_D": signal_conviction_score,
            "touch_conviction_score_D": touch_conviction_score,
            "gathering_by_chasing_D": gathering_by_chasing,
            "gathering_by_support_D": gathering_by_support,
            "VOLATILITY_INSTABILITY_INDEX_21d_D": volatility_instability,
            "ADX_14_D": adx,
            "buy_flow_efficiency_index_D": buy_flow_efficiency,
            "sell_flow_efficiency_index_D": sell_flow_efficiency,
            "auction_closing_position_D": auction_closing_position,
            "auction_impact_score_D": auction_impact_score,
            "auction_intent_signal_D": auction_intent_signal,
            "order_book_liquidity_supply_D": order_book_liquidity_supply,
            "liquidity_slope_D": liquidity_slope,
            "peak_mass_transfer_rate_D": peak_mass_transfer_rate,
            "mf_cost_zone_defense_intent_D": mf_cost_zone_defense_intent,
            "bid_side_liquidity_D": bid_side_liquidity,
            "ask_side_liquidity_D": ask_side_liquidity,
            "retail_panic_surrender_index_D": retail_panic_surrender,
            "main_force_activity_ratio_D": main_force_activity_ratio,
            "main_force_conviction_index_D": main_force_conviction_index,
            "main_force_execution_alpha_D": main_force_execution_alpha,
            "main_force_flow_directionality_D": main_force_flow_directionality,
            "main_force_on_peak_buy_flow_D": main_force_on_peak_buy_flow,
            "main_force_on_peak_sell_flow_D": main_force_on_peak_sell_flow,
            "main_force_t0_efficiency_D": main_force_t0_efficiency,
            "main_force_t0_sell_efficiency_D": main_force_t0_sell_efficiency,
            "main_force_vwap_down_guidance_D": main_force_vwap_down_guidance,
            "main_force_vwap_up_guidance_D": main_force_vwap_up_guidance,
            "market_impact_cost_D": market_impact_cost,
            "opening_buy_strength_D": opening_buy_strength,
            "opening_sell_strength_D": opening_sell_strength,
            "closing_strength_index_D": closing_strength_index,
            "total_buy_amount_calibrated_D": total_buy_amount_calibrated,
            "total_sell_amount_calibrated_D": total_sell_amount_calibrated,
            "wash_trade_intensity_D": wash_trade_intensity,
            "winner_profit_margin_avg_D": winner_profit_margin_avg,
            "loser_loss_margin_avg_D": loser_loss_margin_avg,
            "total_winner_rate_D": total_winner_rate,
            "total_loser_rate_D": total_loser_rate,
            "impulse_quality_ratio_D": impulse_quality_ratio,
            "thrust_efficiency_score_D": thrust_efficiency_score,
            "platform_conviction_score_D": platform_conviction_score,
            "platform_high_D": platform_high,
            "platform_low_D": platform_low,
            "breakout_quality_score_D": breakout_quality_score,
            "breakout_readiness_score_D": breakout_readiness_score,
            "breakthrough_conviction_score_D": breakthrough_conviction_score,
            "defense_solidity_score_D": defense_solidity_score,
            "support_validation_strength_D": support_validation_strength,
            "covert_accumulation_signal_D": covert_accumulation_signal,
            "suppressive_accumulation_intensity_D": suppressive_accumulation_intensity,
            "deception_index_D": deception_index,
            "deception_lure_long_intensity_D": deception_lure_long_intensity,
            "equilibrium_compression_index_D": equilibrium_compression_index,
            "final_charge_intensity_D": final_charge_intensity,
            "floating_chip_cleansing_efficiency_D": floating_chip_cleansing_efficiency,
            "hidden_accumulation_intensity_D": hidden_accumulation_intensity,
            "internal_accumulation_intensity_D": internal_accumulation_intensity,
            "intraday_posture_score_D": intraday_posture_score,
            "opening_gap_defense_strength_D": opening_gap_defense_strength,
            "panic_buy_absorption_contribution_D": panic_buy_absorption_contribution,
            "panic_sell_volume_contribution_D": panic_sell_volume_contribution,
            "panic_selling_cascade_D": panic_selling_cascade,
            "peak_control_transfer_D": peak_control_transfer,
            "peak_separation_ratio_D": peak_separation_ratio,
            "price_reversion_velocity_D": price_reversion_velocity,
            "pullback_depth_ratio_D": pullback_depth_ratio,
            "quality_score_D": quality_score,
            "reversal_conviction_rate_D": reversal_conviction_rate,
            "reversal_power_index_D": reversal_power_index,
            "reversal_recovery_rate_D": reversal_recovery_rate,
            "risk_reward_profile_D": risk_reward_profile,
            "shock_conviction_score_D": shock_conviction_score,
            "strategic_phase_score_D": strategic_phase_score,
            "structural_entropy_change_D": structural_entropy_change,
            "structural_leverage_D": structural_leverage,
            "structural_node_count_D": structural_node_count,
            "structural_potential_score_D": structural_potential_score,
            "support_validation_score_D": support_validation_score,
            "supportive_distribution_intensity_D": supportive_distribution_intensity,
            "trend_acceleration_score_D": trend_acceleration_score,
            "trend_alignment_index_D": trend_alignment_index,
            "trend_asymmetry_index_D": trend_asymmetry_index,
            "trend_conviction_score_D": trend_conviction_score,
            "value_area_migration_D": value_area_migration,
            "value_area_overlap_pct_D": value_area_overlap_pct,
            "volatility_asymmetry_index_D": volatility_asymmetry_index,
            "volume_burstiness_index_D": volume_burstiness_index,
            "volume_structure_skew_D": volume_structure_skew,
            "vpin_score_D": vpin_score,
            "vwap_control_strength_D": vwap_control_strength,
            "vwap_cross_down_intensity_D": vwap_cross_down_intensity,
            "vwap_cross_up_intensity_D": vwap_cross_up_intensity,
            "vwap_crossing_intensity_D": vwap_crossing_intensity,
            "vwap_mean_reversion_corr_D": vwap_mean_reversion_corr,
            "vwap_sell_control_strength_D": vwap_sell_control_strength,
            "winner_stability_index_D": winner_stability_index,
            "winner_concentration_90pct_D": winner_concentration_90pct, # V11.0
            f"SLOPE_{long_term_trend_slope_period}_close_D": long_term_trend_slope, # V11.0
            "absorption_of_distribution_intensity_D": absorption_of_distribution_intensity # V11.2
        }
        _temp_debug_values["派生信号值"] = {
            "is_limit_up_day": is_limit_up_day
        }
        # --- MTF融合信号 ---
        mtf_price_trend = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mf_net_flow = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_upper_shadow_pressure = self._get_mtf_slope_accel_score(df, 'upper_shadow_selling_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_retail_fomo = self._get_mtf_slope_accel_score(df, 'retail_fomo_premium_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # V10.0 新增MTF融合信号
        mtf_buy_sweep_intensity = self._get_mtf_slope_accel_score(df, 'buy_sweep_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_buy_ofi = self._get_mtf_slope_accel_score(df, 'main_force_buy_ofi_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_t0_buy_efficiency = self._get_mtf_slope_accel_score(df, 'main_force_t0_buy_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_upward_impulse_strength = self._get_mtf_slope_accel_score(df, 'upward_impulse_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vwap_buy_control_strength = self._get_mtf_slope_accel_score(df, 'vwap_buy_control_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_cost_zone_buy_intent = self._get_mtf_slope_accel_score(df, 'mf_cost_zone_buy_intent_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_chip_fault_blockage_ratio = self._get_mtf_slope_accel_score(df, 'chip_fault_blockage_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vacuum_traversal_efficiency = self._get_mtf_slope_accel_score(df, 'vacuum_traversal_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_dip_buy_absorption_strength = self._get_mtf_slope_accel_score(df, 'dip_buy_absorption_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_rally_buy_support_weakness = self._get_mtf_slope_accel_score(df, 'rally_buy_support_weakness_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_covert_distribution = self._get_mtf_slope_accel_score(df, 'covert_distribution_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_deception_lure_short = self._get_mtf_slope_accel_score(df, 'deception_lure_short_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_rally_distribution_pressure = self._get_mtf_slope_accel_score(df, 'rally_distribution_pressure_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_exhaustion_risk = self._get_mtf_slope_accel_score(df, 'exhaustion_risk_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_asymmetric_friction = self._get_mtf_slope_accel_score(df, 'asymmetric_friction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_volatility_expansion = self._get_mtf_slope_accel_score(df, 'volatility_expansion_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_market_sentiment = self._get_mtf_slope_accel_score(df, 'market_sentiment_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_structural_tension = self._get_mtf_slope_accel_score(df, 'structural_tension_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_trend_vitality = self._get_mtf_slope_accel_score(df, 'trend_vitality_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_liquidity_authenticity = self._get_mtf_slope_accel_score(df, 'liquidity_authenticity_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_order_book_clearing_rate = self._get_mtf_slope_accel_score(df, 'order_book_clearing_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_sell_sweep_intensity = self._get_mtf_slope_accel_score(df, 'sell_sweep_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow_gini = self._get_mtf_slope_accel_score(df, 'main_force_flow_gini_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_microstructure_efficiency = self._get_mtf_slope_accel_score(df, 'microstructure_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_imbalance_effectiveness = self._get_mtf_slope_accel_score(df, 'imbalance_effectiveness_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_auction_showdown = self._get_mtf_slope_accel_score(df, 'auction_showdown_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_closing_conviction = self._get_mtf_slope_accel_score(df, 'closing_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_intraday_energy_density = self._get_mtf_slope_accel_score(df, 'intraday_energy_density_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_intraday_thrust_purity = self._get_mtf_slope_accel_score(df, 'intraday_thrust_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_thrust_divergence = self._get_mtf_slope_accel_score(df, 'price_thrust_divergence_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_trend_efficiency_ratio = self._get_mtf_slope_accel_score(df, 'trend_efficiency_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_concentration_90pct = self._get_mtf_slope_accel_score(df, 'loser_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_winner_loser_momentum = self._get_mtf_slope_accel_score(df, 'winner_loser_momentum_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_cost_structure_skewness = self._get_mtf_slope_accel_score(df, 'cost_structure_skewness_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_cost_gini_coefficient = self._get_mtf_slope_accel_score(df, 'cost_gini_coefficient_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_vpoc_premium = self._get_mtf_slope_accel_score(df, 'mf_vpoc_premium_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_character_score = self._get_mtf_slope_accel_score(df, 'character_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_signal_conviction_score = self._get_mtf_slope_accel_score(df, 'signal_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_touch_conviction_score = self._get_mtf_slope_accel_score(df, 'touch_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_gathering_by_chasing = self._get_mtf_slope_accel_score(df, 'gathering_by_chasing_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_gathering_by_support = self._get_mtf_slope_accel_score(df, 'gathering_by_support_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_buy_flow_efficiency = self._get_mtf_slope_accel_score(df, 'buy_flow_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_sell_flow_efficiency = self._get_mtf_slope_accel_score(df, 'sell_flow_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_auction_closing_position = self._get_mtf_slope_accel_score(df, 'auction_closing_position_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_auction_impact_score = self._get_mtf_slope_accel_score(df, 'auction_impact_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_auction_intent_signal = self._get_mtf_slope_accel_score(df, 'auction_intent_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_order_book_liquidity_supply = self._get_mtf_slope_accel_score(df, 'order_book_liquidity_supply_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_liquidity_slope = self._get_mtf_slope_accel_score(df, 'liquidity_slope_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_peak_mass_transfer_rate = self._get_mtf_slope_accel_score(df, 'peak_mass_transfer_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_mf_cost_zone_defense_intent = self._get_mtf_slope_accel_score(df, 'mf_cost_zone_defense_intent_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_bid_side_liquidity = self._get_mtf_slope_accel_score(df, 'bid_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_ask_side_liquidity = self._get_mtf_slope_accel_score(df, 'ask_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_retail_panic_surrender = self._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_activity_ratio = self._get_mtf_slope_accel_score(df, 'main_force_activity_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_conviction_index = self._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_execution_alpha = self._get_mtf_slope_accel_score(df, 'main_force_execution_alpha_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_flow_directionality = self._get_mtf_slope_accel_score(df, 'main_force_flow_directionality_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_on_peak_buy_flow = self._get_mtf_slope_accel_score(df, 'main_force_on_peak_buy_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_on_peak_sell_flow = self._get_mtf_slope_accel_score(df, 'main_force_on_peak_sell_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_t0_efficiency = self._get_mtf_slope_accel_score(df, 'main_force_t0_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_t0_sell_efficiency = self._get_mtf_slope_accel_score(df, 'main_force_t0_sell_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_vwap_down_guidance = self._get_mtf_slope_accel_score(df, 'main_force_vwap_down_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_vwap_up_guidance = self._get_mtf_slope_accel_score(df, 'main_force_vwap_up_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_market_impact_cost = self._get_mtf_slope_accel_score(df, 'market_impact_cost_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_opening_buy_strength = self._get_mtf_slope_accel_score(df, 'opening_buy_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_opening_sell_strength = self._get_mtf_slope_accel_score(df, 'opening_sell_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_closing_strength_index = self._get_mtf_slope_accel_score(df, 'closing_strength_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_buy_amount_calibrated = self._get_mtf_slope_accel_score(df, 'total_buy_amount_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_sell_amount_calibrated = self._get_mtf_slope_accel_score(df, 'total_sell_amount_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_wash_trade_intensity = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_winner_profit_margin_avg = self._get_mtf_slope_accel_score(df, 'winner_profit_margin_avg_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_loser_loss_margin_avg = self._get_mtf_slope_accel_score(df, 'loser_loss_margin_avg_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_total_winner_rate = self._get_mtf_slope_accel_score(df, 'total_winner_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_loser_rate = self._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_impulse_quality_ratio = self._get_mtf_slope_accel_score(df, 'impulse_quality_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_thrust_efficiency_score = self._get_mtf_slope_accel_score(df, 'thrust_efficiency_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_platform_conviction_score = self._get_mtf_slope_accel_score(df, 'platform_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_platform_high = self._get_mtf_slope_accel_score(df, 'platform_high_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_platform_low = self._get_mtf_slope_accel_score(df, 'platform_low_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_breakout_quality_score = self._get_mtf_slope_accel_score(df, 'breakout_quality_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_breakout_readiness_score = self._get_mtf_slope_accel_score(df, 'breakout_readiness_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_breakthrough_conviction_score = self._get_mtf_slope_accel_score(df, 'breakthrough_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_defense_solidity_score = self._get_mtf_slope_accel_score(df, 'defense_solidity_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_support_validation_strength = self._get_mtf_slope_accel_score(df, 'support_validation_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_covert_accumulation_signal = self._get_mtf_slope_accel_score(df, 'covert_accumulation_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_suppressive_accumulation_intensity = self._get_mtf_slope_accel_score(df, 'suppressive_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_deception_index = self._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_deception_lure_long_intensity = self._get_mtf_slope_accel_score(df, 'deception_lure_long_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_equilibrium_compression_index = self._get_mtf_slope_accel_score(df, 'equilibrium_compression_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_final_charge_intensity = self._get_mtf_slope_accel_score(df, 'final_charge_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_floating_chip_cleansing_efficiency = self._get_mtf_slope_accel_score(df, 'floating_chip_cleansing_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_hidden_accumulation_intensity = self._get_mtf_slope_accel_score(df, 'hidden_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_internal_accumulation_intensity = self._get_mtf_slope_accel_score(df, 'internal_accumulation_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_intraday_posture_score = self._get_mtf_slope_accel_score(df, 'intraday_posture_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_opening_gap_defense_strength = self._get_mtf_slope_accel_score(df, 'opening_gap_defense_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_panic_buy_absorption_contribution = self._get_mtf_slope_accel_score(df, 'panic_buy_absorption_contribution_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_panic_sell_volume_contribution = self._get_mtf_slope_accel_score(df, 'panic_sell_volume_contribution_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_panic_selling_cascade = self._get_mtf_slope_accel_score(df, 'panic_selling_cascade_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_peak_control_transfer = self._get_mtf_slope_accel_score(df, 'peak_control_transfer_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_peak_separation_ratio = self._get_mtf_slope_accel_score(df, 'peak_separation_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_reversion_velocity = self._get_mtf_slope_accel_score(df, 'price_reversion_velocity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_pullback_depth_ratio = self._get_mtf_slope_accel_score(df, 'pullback_depth_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_quality_score = self._get_mtf_slope_accel_score(df, 'quality_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_reversal_conviction_rate = self._get_mtf_slope_accel_score(df, 'reversal_conviction_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_reversal_power_index = self._get_mtf_slope_accel_score(df, 'reversal_power_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_reversal_recovery_rate = self._get_mtf_slope_accel_score(df, 'reversal_recovery_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_risk_reward_profile = self._get_mtf_slope_accel_score(df, 'risk_reward_profile_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_shock_conviction_score = self._get_mtf_slope_accel_score(df, 'shock_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_strategic_phase_score = self._get_mtf_slope_accel_score(df, 'strategic_phase_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_structural_entropy_change = self._get_mtf_slope_accel_score(df, 'structural_entropy_change_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_structural_leverage = self._get_mtf_slope_accel_score(df, 'structural_leverage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_structural_node_count = self._get_mtf_slope_accel_score(df, 'structural_node_count_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_structural_potential_score = self._get_mtf_slope_accel_score(df, 'structural_potential_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_support_validation_score = self._get_mtf_slope_accel_score(df, 'support_validation_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_supportive_distribution_intensity = self._get_mtf_slope_accel_score(df, 'supportive_distribution_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_trend_acceleration_score = self._get_mtf_slope_accel_score(df, 'trend_acceleration_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_trend_alignment_index = self._get_mtf_slope_accel_score(df, 'trend_alignment_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_trend_asymmetry_index = self._get_mtf_slope_accel_score(df, 'trend_asymmetry_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_trend_conviction_score = self._get_mtf_slope_accel_score(df, 'trend_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_value_area_migration = self._get_mtf_slope_accel_score(df, 'value_area_migration_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_value_area_overlap_pct = self._get_mtf_slope_accel_score(df, 'value_area_overlap_pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_volatility_asymmetry_index = self._get_mtf_slope_accel_score(df, 'volatility_asymmetry_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_volume_burstiness_index = self._get_mtf_slope_accel_score(df, 'volume_burstiness_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_volume_structure_skew = self._get_mtf_slope_accel_score(df, 'volume_structure_skew_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_vpin_score = self._get_mtf_slope_accel_score(df, 'vpin_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vwap_control_strength = self._get_mtf_slope_accel_score(df, 'vwap_control_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vwap_cross_down_intensity = self._get_mtf_slope_accel_score(df, 'vwap_cross_down_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vwap_cross_up_intensity = self._get_mtf_slope_accel_score(df, 'vwap_cross_up_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vwap_crossing_intensity = self._get_mtf_slope_accel_score(df, 'vwap_crossing_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vwap_mean_reversion_corr = self._get_mtf_slope_accel_score(df, 'vwap_mean_reversion_corr_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_vwap_sell_control_strength = self._get_mtf_slope_accel_score(df, 'vwap_sell_control_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_winner_stability_index = self._get_mtf_slope_accel_score(df, 'winner_stability_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # V11.2 新增MTF融合信号
        mtf_absorption_of_distribution_intensity = self._get_mtf_slope_accel_score(df, 'absorption_of_distribution_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)

        _temp_debug_values["MTF融合信号"] = {
            "mtf_price_trend": mtf_price_trend,
            "mtf_mf_net_flow": mtf_mf_net_flow,
            "mtf_upper_shadow_pressure": mtf_upper_shadow_pressure,
            "mtf_retail_fomo": mtf_retail_fomo,
            "mtf_buy_sweep_intensity": mtf_buy_sweep_intensity,
            "mtf_main_force_buy_ofi": mtf_main_force_buy_ofi,
            "mtf_main_force_t0_buy_efficiency": mtf_main_force_t0_buy_efficiency,
            "mtf_upward_impulse_strength": mtf_upward_impulse_strength,
            "mtf_vwap_buy_control_strength": mtf_vwap_buy_control_strength,
            "mtf_mf_cost_zone_buy_intent": mtf_mf_cost_zone_buy_intent,
            "mtf_chip_fault_blockage_ratio": mtf_chip_fault_blockage_ratio,
            "mtf_vacuum_traversal_efficiency": mtf_vacuum_traversal_efficiency,
            "mtf_dip_buy_absorption_strength": mtf_dip_buy_absorption_strength,
            "mtf_rally_buy_support_weakness": mtf_rally_buy_support_weakness,
            "mtf_covert_distribution": mtf_covert_distribution,
            "mtf_deception_lure_short": mtf_deception_lure_short,
            "mtf_rally_distribution_pressure": mtf_rally_distribution_pressure,
            "mtf_exhaustion_risk": mtf_exhaustion_risk,
            "mtf_asymmetric_friction": mtf_asymmetric_friction,
            "mtf_volatility_expansion": mtf_volatility_expansion,
            "mtf_market_sentiment": mtf_market_sentiment,
            "mtf_structural_tension": mtf_structural_tension,
            "mtf_trend_vitality": mtf_trend_vitality,
            "mtf_liquidity_authenticity": mtf_liquidity_authenticity,
            "mtf_order_book_clearing_rate": mtf_order_book_clearing_rate,
            "mtf_sell_sweep_intensity": mtf_sell_sweep_intensity,
            "mtf_main_force_flow_gini": mtf_main_force_flow_gini,
            "mtf_microstructure_efficiency": mtf_microstructure_efficiency,
            "mtf_imbalance_effectiveness": mtf_imbalance_effectiveness,
            "mtf_auction_showdown": mtf_auction_showdown,
            "mtf_closing_conviction": mtf_closing_conviction,
            "mtf_intraday_energy_density": mtf_intraday_energy_density,
            "mtf_intraday_thrust_purity": mtf_intraday_thrust_purity,
            "mtf_price_thrust_divergence": mtf_price_thrust_divergence,
            "mtf_trend_efficiency_ratio": mtf_trend_efficiency_ratio,
            "mtf_loser_concentration_90pct": mtf_loser_concentration_90pct,
            "mtf_winner_loser_momentum": mtf_winner_loser_momentum,
            "mtf_cost_structure_skewness": mtf_cost_structure_skewness,
            "mtf_cost_gini_coefficient": mtf_cost_gini_coefficient,
            "mtf_mf_vpoc_premium": mtf_mf_vpoc_premium,
            "mtf_character_score": mtf_character_score,
            "mtf_signal_conviction_score": mtf_signal_conviction_score,
            "mtf_touch_conviction_score": mtf_touch_conviction_score,
            "mtf_gathering_by_chasing": mtf_gathering_by_chasing,
            "mtf_gathering_by_support": mtf_gathering_by_support,
            "mtf_buy_flow_efficiency": mtf_buy_flow_efficiency,
            "mtf_sell_flow_efficiency": mtf_sell_flow_efficiency,
            "mtf_auction_closing_position": mtf_auction_closing_position,
            "mtf_auction_impact_score": mtf_auction_impact_score,
            "mtf_auction_intent_signal": mtf_auction_intent_signal,
            "mtf_order_book_liquidity_supply": mtf_order_book_liquidity_supply,
            "mtf_liquidity_slope": mtf_liquidity_slope,
            "mtf_peak_mass_transfer_rate": mtf_peak_mass_transfer_rate,
            "mtf_mf_cost_zone_defense_intent": mtf_mf_cost_zone_defense_intent,
            "mtf_bid_side_liquidity": mtf_bid_side_liquidity,
            "mtf_ask_side_liquidity": mtf_ask_side_liquidity,
            "mtf_retail_panic_surrender": mtf_retail_panic_surrender,
            "mtf_main_force_activity_ratio": mtf_main_force_activity_ratio,
            "mtf_main_force_conviction_index": mtf_main_force_conviction_index,
            "mtf_main_force_execution_alpha": mtf_main_force_execution_alpha,
            "mtf_main_force_flow_directionality": mtf_main_force_flow_directionality,
            "mtf_main_force_on_peak_buy_flow": mtf_main_force_on_peak_buy_flow,
            "mtf_main_force_on_peak_sell_flow": mtf_main_force_on_peak_sell_flow,
            "mtf_main_force_t0_efficiency": mtf_main_force_t0_efficiency,
            "mtf_main_force_t0_sell_efficiency": mtf_main_force_t0_sell_efficiency,
            "mtf_main_force_vwap_down_guidance": mtf_main_force_vwap_down_guidance,
            "mtf_main_force_vwap_up_guidance": mtf_main_force_vwap_up_guidance,
            "mtf_market_impact_cost": mtf_market_impact_cost,
            "mtf_opening_buy_strength": mtf_opening_buy_strength,
            "mtf_opening_sell_strength": mtf_opening_sell_strength,
            "mtf_closing_strength_index": mtf_closing_strength_index,
            "mtf_total_buy_amount_calibrated": mtf_total_buy_amount_calibrated,
            "mtf_total_sell_amount_calibrated": mtf_total_sell_amount_calibrated,
            "mtf_wash_trade_intensity": mtf_wash_trade_intensity,
            "mtf_winner_profit_margin_avg": mtf_winner_profit_margin_avg,
            "mtf_loser_loss_margin_avg": mtf_loser_loss_margin_avg,
            "mtf_total_winner_rate": mtf_total_winner_rate,
            "mtf_total_loser_rate": mtf_total_loser_rate,
            "mtf_impulse_quality_ratio": mtf_impulse_quality_ratio,
            "mtf_thrust_efficiency_score": mtf_thrust_efficiency_score,
            "mtf_platform_conviction_score": mtf_platform_conviction_score,
            "mtf_platform_high": mtf_platform_high,
            "mtf_platform_low": mtf_platform_low,
            "mtf_breakout_quality_score": mtf_breakout_quality_score,
            "mtf_breakout_readiness_score": mtf_breakout_readiness_score,
            "mtf_breakthrough_conviction_score": mtf_breakthrough_conviction_score,
            "mtf_defense_solidity_score": mtf_defense_solidity_score,
            "mtf_support_validation_strength": mtf_support_validation_strength,
            "mtf_covert_accumulation_signal": mtf_covert_accumulation_signal,
            "mtf_suppressive_accumulation_intensity": mtf_suppressive_accumulation_intensity,
            "mtf_deception_index": mtf_deception_index,
            "mtf_deception_lure_long_intensity": mtf_deception_lure_long_intensity,
            "mtf_equilibrium_compression_index": mtf_equilibrium_compression_index,
            "mtf_final_charge_intensity": mtf_final_charge_intensity,
            "mtf_floating_chip_cleansing_efficiency": mtf_floating_chip_cleansing_efficiency,
            "mtf_hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "mtf_internal_accumulation_intensity": mtf_internal_accumulation_intensity,
            "mtf_intraday_posture_score": mtf_intraday_posture_score,
            "mtf_opening_gap_defense_strength": mtf_opening_gap_defense_strength,
            "mtf_panic_buy_absorption_contribution": mtf_panic_buy_absorption_contribution,
            "mtf_panic_sell_volume_contribution": mtf_panic_sell_volume_contribution,
            "mtf_panic_selling_cascade": mtf_panic_selling_cascade,
            "mtf_peak_control_transfer": mtf_peak_control_transfer,
            "mtf_peak_separation_ratio": mtf_peak_separation_ratio,
            "mtf_price_reversion_velocity": mtf_price_reversion_velocity,
            "mtf_pullback_depth_ratio": mtf_pullback_depth_ratio,
            "mtf_quality_score": mtf_quality_score,
            "mtf_reversal_conviction_rate": mtf_reversal_conviction_rate,
            "mtf_reversal_power_index": mtf_reversal_power_index,
            "mtf_reversal_recovery_rate": mtf_reversal_recovery_rate,
            "mtf_risk_reward_profile": mtf_risk_reward_profile,
            "mtf_shock_conviction_score": mtf_shock_conviction_score,
            "mtf_strategic_phase_score": mtf_strategic_phase_score,
            "mtf_structural_entropy_change": mtf_structural_entropy_change,
            "mtf_structural_leverage": mtf_structural_leverage,
            "mtf_structural_node_count": mtf_structural_node_count,
            "mtf_structural_potential_score": mtf_structural_potential_score,
            "mtf_support_validation_score": mtf_support_validation_score,
            "mtf_supportive_distribution_intensity": mtf_supportive_distribution_intensity,
            "mtf_trend_acceleration_score": mtf_trend_acceleration_score,
            "mtf_trend_alignment_index": mtf_trend_alignment_index,
            "mtf_trend_asymmetry_index": mtf_trend_asymmetry_index,
            "mtf_trend_conviction_score": mtf_trend_conviction_score,
            "mtf_value_area_migration": mtf_value_area_migration,
            "mtf_value_area_overlap_pct": mtf_value_area_overlap_pct,
            "mtf_volatility_asymmetry_index": mtf_volatility_asymmetry_index,
            "mtf_volume_burstiness_index": mtf_volume_burstiness_index,
            "mtf_volume_structure_skew": mtf_volume_structure_skew,
            "mtf_vpin_score": mtf_vpin_score,
            "mtf_vwap_control_strength": mtf_vwap_control_strength,
            "mtf_vwap_cross_down_intensity": mtf_vwap_cross_down_intensity,
            "mtf_vwap_cross_up_intensity": mtf_vwap_cross_up_intensity,
            "mtf_vwap_crossing_intensity": mtf_vwap_crossing_intensity,
            "mtf_vwap_mean_reversion_corr": mtf_vwap_mean_reversion_corr,
            "mtf_vwap_sell_control_strength": mtf_vwap_sell_control_strength,
            "mtf_winner_stability_index": mtf_winner_stability_index,
            "mtf_absorption_of_distribution_intensity": mtf_absorption_of_distribution_intensity # V11.2
        }
        # --- V11.1: 引入历史记忆与上下文机制 ---
        # 确保这些变量在任何情况下都被初始化
        mtf_cumulative_mf_flow = pd.Series(0.0, index=df.index, dtype=np.float32)
        mtf_chip_concentration_stability = pd.Series(0.5, index=df.index, dtype=np.float32) # 0.5表示中性，不增强也不削弱
        mtf_long_term_trend_strength = pd.Series(0.0, index=df.index, dtype=np.float32)

        if hc_enabled:
            # 1. 主力资金累计记忆 (Cumulative Main Force Flow Memory)
            # 计算长期主力资金累计净流入
            # min_periods 确保在数据不足时不会产生NaN，而是从有足够数据开始计算
            cumulative_mf_flow_long = main_force_net_flow.rolling(window=cumulative_mf_flow_window, min_periods=int(cumulative_mf_flow_window * 0.5)).sum()
            # MTF归一化，使用中短期MTF权重
            mtf_cumulative_mf_flow = get_adaptive_mtf_normalized_score(
                cumulative_mf_flow_long, df.index, hc_mtf_weights_medium, ascending=True,
                debug_info=(is_debug_enabled_for_method, probe_ts, "mtf_cumulative_mf_flow")
            )
            _temp_debug_values["MTF融合信号"]["mtf_cumulative_mf_flow"] = mtf_cumulative_mf_flow

            # 2. 筹码集中度稳定性 (Chip Concentration Stability)
            # 计算赢家筹码集中度长期波动性的倒数（波动性越低，稳定性越高）
            # 使用 winner_concentration_90pct 作为基础信号
            rolling_std_winner_concentration = winner_concentration_90pct.rolling(window=chip_concentration_stability_window, min_periods=int(chip_concentration_stability_window * 0.5)).std().replace(0, np.nan)
            chip_concentration_stability_raw = (1 / rolling_std_winner_concentration).fillna(0)
            # MTF归一化，使用中短期MTF权重
            mtf_chip_concentration_stability = get_adaptive_mtf_normalized_score(
                chip_concentration_stability_raw, df.index, hc_mtf_weights_medium, ascending=True,
                debug_info=(is_debug_enabled_for_method, probe_ts, "mtf_chip_concentration_stability")
            )
            _temp_debug_values["MTF融合信号"]["mtf_chip_concentration_stability"] = mtf_chip_concentration_stability

            # 3. 长期趋势强度上下文 (Long-Term Trend Strength Context)
            # 获取长期趋势斜率，现在是21日周期
            # MTF归一化，使用中短期MTF权重
            mtf_long_term_trend_strength = get_adaptive_mtf_normalized_score(
                long_term_trend_slope, df.index, hc_mtf_weights_medium, ascending=True,
                debug_info=(is_debug_enabled_for_method, probe_ts, "mtf_long_term_trend_strength")
            )
            _temp_debug_values["MTF融合信号"]["mtf_long_term_trend_strength"] = mtf_long_term_trend_strength
        # else 块已在初始化时处理，确保变量始终有值

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
        # V10.0 新增归一化信号
        order_book_imbalance_positive_norm = self._normalize_series(order_book_imbalance.clip(lower=0), df_index, bipolar=False)
        micro_price_impact_asymmetry_positive_norm = self._normalize_series(micro_price_impact_asymmetry.clip(lower=0), df_index, bipolar=False)
        constructive_turnover_norm = self._normalize_series(constructive_turnover, df_index, bipolar=False)
        chip_fault_blockage_ratio_inverted_norm = self._normalize_series(chip_fault_blockage_ratio, df_index, bipolar=False, ascending=False)
        vacuum_zone_magnitude_norm = self._normalize_series(vacuum_zone_magnitude, df_index, bipolar=False)
        rally_buy_support_weakness_inverted_norm = self._normalize_series(rally_buy_support_weakness, df_index, bipolar=False, ascending=False)
        order_book_clearing_rate_norm = self._normalize_series(order_book_clearing_rate, df_index, bipolar=False)
        sell_sweep_intensity_inverted_norm = self._normalize_series(sell_sweep_intensity, df_index, bipolar=False, ascending=False)
        main_force_flow_gini_inverted_norm = self._normalize_series(main_force_flow_gini, df_index, bipolar=False, ascending=False)
        microstructure_efficiency_norm = self._normalize_series(microstructure_efficiency, df_index, bipolar=False)
        imbalance_effectiveness_norm = self._normalize_series(imbalance_effectiveness, df_index, bipolar=False)
        auction_showdown_norm = self._normalize_series(auction_showdown, df_index, bipolar=False)
        closing_conviction_norm = self._normalize_series(closing_conviction, df_index, bipolar=False)
        intraday_energy_density_norm = self._normalize_series(intraday_energy_density, df_index, bipolar=False)
        intraday_thrust_purity_norm = self._normalize_series(intraday_thrust_purity, df_index, bipolar=False)
        price_thrust_divergence_norm = self._normalize_series(price_thrust_divergence, df_index, bipolar=True)
        trend_efficiency_ratio_norm = self._normalize_series(trend_efficiency_ratio, df_index, bipolar=False)
        loser_concentration_90pct_norm = self._normalize_series(loser_concentration_90pct, df_index, bipolar=False)
        winner_loser_momentum_norm = self._normalize_series(winner_loser_momentum, df_index, bipolar=True)
        cost_structure_skewness_norm = self._normalize_series(cost_structure_skewness, df_index, bipolar=True)
        cost_gini_coefficient_norm = self._normalize_series(cost_gini_coefficient, df_index, bipolar=False)
        mf_vpoc_premium_norm = self._normalize_series(mf_vpoc_premium, df_index, bipolar=True)
        character_score_norm = self._normalize_series(character_score, df_index, bipolar=False)
        signal_conviction_score_norm = self._normalize_series(signal_conviction_score, df_index, bipolar=False)
        touch_conviction_score_norm = self._normalize_series(touch_conviction_score, df_index, bipolar=False)
        gathering_by_chasing_norm = self._normalize_series(gathering_by_chasing, df_index, bipolar=False)
        gathering_by_support_norm = self._normalize_series(gathering_by_support, df_index, bipolar=False)
        volatility_instability_norm = self._normalize_series(volatility_instability, df_index, bipolar=False)
        adx_norm = self._normalize_series(adx, df_index, bipolar=False)
        buy_flow_efficiency_norm = self._normalize_series(buy_flow_efficiency, df_index, bipolar=False)
        sell_flow_efficiency_norm = self._normalize_series(sell_flow_efficiency, df_index, bipolar=False)
        auction_closing_position_norm = self._normalize_series(auction_closing_position, df_index, bipolar=True)
        auction_impact_score_norm = self._normalize_series(auction_impact_score, df_index, bipolar=False)
        auction_intent_signal_norm = self._normalize_series(auction_intent_signal, df_index, bipolar=False)
        order_book_liquidity_supply_norm = self._normalize_series(order_book_liquidity_supply, df_index, bipolar=False)
        liquidity_slope_norm = self._normalize_series(liquidity_slope, df_index, bipolar=True)
        peak_mass_transfer_rate_norm = self._normalize_series(peak_mass_transfer_rate, df_index, bipolar=False)
        mf_cost_zone_defense_intent_norm = self._normalize_series(mf_cost_zone_defense_intent, df_index, bipolar=False)
        bid_side_liquidity_norm = self._normalize_series(bid_side_liquidity, df_index, bipolar=False)
        ask_side_liquidity_norm = self._normalize_series(ask_side_liquidity, df_index, bipolar=False)
        retail_panic_surrender_norm = self._normalize_series(retail_panic_surrender, df_index, bipolar=False)
        main_force_activity_ratio_norm = self._normalize_series(main_force_activity_ratio, df_index, bipolar=False)
        main_force_conviction_index_norm = self._normalize_series(main_force_conviction_index, df_index, bipolar=False)
        main_force_execution_alpha_norm = self._normalize_series(main_force_execution_alpha, df_index, bipolar=True)
        main_force_flow_directionality_norm = self._normalize_series(main_force_flow_directionality, df_index, bipolar=True)
        main_force_on_peak_buy_flow_norm = self._normalize_series(main_force_on_peak_buy_flow, df_index, bipolar=False)
        main_force_on_peak_sell_flow_norm = self._normalize_series(main_force_on_peak_sell_flow, df_index, bipolar=False)
        main_force_t0_efficiency_norm = self._normalize_series(main_force_t0_efficiency, df_index, bipolar=False)
        main_force_t0_sell_efficiency_norm = self._normalize_series(main_force_t0_sell_efficiency, df_index, bipolar=False)
        main_force_vwap_down_guidance_norm = self._normalize_series(main_force_vwap_down_guidance, df_index, bipolar=True)
        main_force_vwap_up_guidance_norm = self._normalize_series(main_force_vwap_up_guidance, df_index, bipolar=True)
        market_impact_cost_norm = self._normalize_series(market_impact_cost, df_index, bipolar=False)
        opening_buy_strength_norm = self._normalize_series(opening_buy_strength, df_index, bipolar=False)
        opening_sell_strength_norm = self._normalize_series(opening_sell_strength, df_index, bipolar=False)
        closing_strength_index_norm = self._normalize_series(closing_strength_index, df_index, bipolar=False)
        total_buy_amount_calibrated_norm = self._normalize_series(total_buy_amount_calibrated, df_index, bipolar=False)
        total_sell_amount_calibrated_norm = self._normalize_series(total_sell_amount_calibrated, df_index, bipolar=False)
        wash_trade_intensity_norm = self._normalize_series(wash_trade_intensity, df_index, bipolar=False)
        winner_profit_margin_avg_norm = self._normalize_series(winner_profit_margin_avg, df_index, bipolar=True)
        loser_loss_margin_avg_norm = self._normalize_series(loser_loss_margin_avg, df_index, bipolar=True)
        total_winner_rate_norm = self._normalize_series(total_winner_rate, df_index, bipolar=False)
        total_loser_rate_norm = self._normalize_series(total_loser_rate, df_index, bipolar=False)
        impulse_quality_ratio_norm = self._normalize_series(impulse_quality_ratio, df_index, bipolar=False)
        thrust_efficiency_score_norm = self._normalize_series(thrust_efficiency_score, df_index, bipolar=False)
        platform_conviction_score_norm = self._normalize_series(platform_conviction_score, df_index, bipolar=False)
        platform_high_norm = self._normalize_series(platform_high, df_index, bipolar=False)
        platform_low_norm = self._normalize_series(platform_low, df_index, bipolar=False)
        breakout_quality_score_norm = self._normalize_series(breakout_quality_score, df_index, bipolar=False)
        breakout_readiness_score_norm = self._normalize_series(breakout_readiness_score, df_index, bipolar=False)
        breakthrough_conviction_score_norm = self._normalize_series(breakthrough_conviction_score, df_index, bipolar=False)
        defense_solidity_score_norm = self._normalize_series(defense_solidity_score, df_index, bipolar=False)
        support_validation_strength_norm = self._normalize_series(support_validation_strength, df_index, bipolar=False)
        covert_accumulation_signal_norm = self._normalize_series(covert_accumulation_signal, df_index, bipolar=False)
        suppressive_accumulation_intensity_norm = self._normalize_series(suppressive_accumulation_intensity, df_index, bipolar=False)
        deception_index_norm = self._normalize_series(deception_index, df_index, bipolar=False)
        deception_lure_long_intensity_norm = self._normalize_series(deception_lure_long_intensity, df_index, bipolar=False)
        equilibrium_compression_index_norm = self._normalize_series(equilibrium_compression_index, df_index, bipolar=False)
        final_charge_intensity_norm = self._normalize_series(final_charge_intensity, df_index, bipolar=False)
        floating_chip_cleansing_efficiency_norm = self._normalize_series(floating_chip_cleansing_efficiency, df_index, bipolar=False)
        hidden_accumulation_intensity_norm = self._normalize_series(hidden_accumulation_intensity, df_index, bipolar=False)
        internal_accumulation_intensity_norm = self._normalize_series(internal_accumulation_intensity, df_index, bipolar=False)
        intraday_posture_score_norm = self._normalize_series(intraday_posture_score, df_index, bipolar=False)
        opening_gap_defense_strength_norm = self._normalize_series(opening_gap_defense_strength, df_index, bipolar=False)
        panic_buy_absorption_contribution_norm = self._normalize_series(panic_buy_absorption_contribution, df_index, bipolar=False)
        panic_sell_volume_contribution_norm = self._normalize_series(panic_sell_volume_contribution, df_index, bipolar=False)
        panic_selling_cascade_norm = self._normalize_series(panic_selling_cascade, df_index, bipolar=False)
        peak_control_transfer_norm = self._normalize_series(peak_control_transfer, df_index, bipolar=False)
        peak_separation_ratio_norm = self._normalize_series(peak_separation_ratio, df_index, bipolar=False)
        price_reversion_velocity_norm = self._normalize_series(price_reversion_velocity, df_index, bipolar=True)
        pullback_depth_ratio_norm = self._normalize_series(pullback_depth_ratio, df_index, bipolar=False)
        quality_score_norm = self._normalize_series(quality_score, df_index, bipolar=False)
        reversal_conviction_rate_norm = self._normalize_series(reversal_conviction_rate, df_index, bipolar=False)
        reversal_power_index_norm = self._normalize_series(reversal_power_index, df_index, bipolar=False)
        reversal_recovery_rate_norm = self._normalize_series(reversal_recovery_rate, df_index, bipolar=False)
        risk_reward_profile_norm = self._normalize_series(risk_reward_profile, df_index, bipolar=True)
        shock_conviction_score_norm = self._normalize_series(shock_conviction_score, df_index, bipolar=False)
        strategic_phase_score_norm = self._normalize_series(strategic_phase_score, df_index, bipolar=False)
        structural_entropy_change_norm = self._normalize_series(structural_entropy_change, df_index, bipolar=True)
        structural_leverage_norm = self._normalize_series(structural_leverage, df_index, bipolar=False)
        structural_node_count_norm = self._normalize_series(structural_node_count, df_index, bipolar=False)
        structural_potential_score_norm = self._normalize_series(structural_potential_score, df_index, bipolar=False)
        support_validation_score_norm = self._normalize_series(support_validation_score, df_index, bipolar=False)
        supportive_distribution_intensity_norm = self._normalize_series(supportive_distribution_intensity, df_index, bipolar=False)
        trend_acceleration_score_norm = self._normalize_series(trend_acceleration_score, df_index, bipolar=False)
        trend_alignment_index_norm = self._normalize_series(trend_alignment_index, df_index, bipolar=False)
        trend_asymmetry_index_norm = self._normalize_series(trend_asymmetry_index, df_index, bipolar=True)
        trend_conviction_score_norm = self._normalize_series(trend_conviction_score, df_index, bipolar=False)
        value_area_migration_norm = self._normalize_series(value_area_migration, df_index, bipolar=True)
        value_area_overlap_pct_norm = self._normalize_series(value_area_overlap_pct, df_index, bipolar=False)
        volatility_asymmetry_index_norm = self._normalize_series(volatility_asymmetry_index, df_index, bipolar=True)
        volume_burstiness_index_norm = self._normalize_series(volume_burstiness_index, df_index, bipolar=False)
        volume_structure_skew_norm = self._normalize_series(volume_structure_skew, df_index, bipolar=True)
        vpin_score_norm = self._normalize_series(vpin_score, df_index, bipolar=False)
        vwap_control_strength_norm = self._normalize_series(vwap_control_strength, df_index, bipolar=False)
        vwap_cross_down_intensity_norm = self._normalize_series(vwap_cross_down_intensity, df_index, bipolar=False)
        vwap_cross_up_intensity_norm = self._normalize_series(vwap_cross_up_intensity, df_index, bipolar=False)
        vwap_crossing_intensity_norm = self._normalize_series(vwap_crossing_intensity, df_index, bipolar=False)
        vwap_mean_reversion_corr_norm = self._normalize_series(vwap_mean_reversion_corr, df_index, bipolar=True)
        vwap_sell_control_strength_norm = self._normalize_series(vwap_sell_control_strength, df_index, bipolar=False)
        winner_stability_index_norm = self._normalize_series(winner_stability_index, df_index, bipolar=False)
        # V11.2 新增归一化信号
        absorption_of_distribution_intensity_norm = self._normalize_series(absorption_of_distribution_intensity, df_index, bipolar=False)

        _temp_debug_values["归一化处理"] = {
            "price_impact_norm": price_impact_norm,
            "impulse_purity_norm": impulse_purity_norm,
            "volume_ratio_norm": volume_ratio_norm,
            "control_solidity_norm": control_solidity_norm,
            "cost_advantage_norm": cost_advantage_norm,
            "concentration_slope_norm": concentration_slope_norm,
            "peak_solidity_norm": peak_solidity_norm,
            "buying_support_norm": buying_support_norm,
            "pressure_rejection_norm": pressure_rejection_norm,
            "profit_absorption_norm": profit_absorption_norm,
            "flow_credibility_norm": flow_credibility_norm,
            "chip_health_norm": chip_health_norm,
            "retail_fomo_norm": retail_fomo_norm,
            # 新增归一化信号调试
            "order_book_imbalance_positive_norm": order_book_imbalance_positive_norm,
            "micro_price_impact_asymmetry_positive_norm": micro_price_impact_asymmetry_positive_norm,
            "constructive_turnover_norm": constructive_turnover_norm,
            "chip_fault_blockage_ratio_inverted_norm": chip_fault_blockage_ratio_inverted_norm,
            "vacuum_zone_magnitude_norm": vacuum_zone_magnitude_norm,
            "rally_buy_support_weakness_inverted_norm": rally_buy_support_weakness_inverted_norm,
            "order_book_clearing_rate_norm": order_book_clearing_rate_norm,
            "sell_sweep_intensity_inverted_norm": sell_sweep_intensity_inverted_norm,
            "main_force_flow_gini_inverted_norm": main_force_flow_gini_inverted_norm,
            "microstructure_efficiency_norm": microstructure_efficiency_norm,
            "imbalance_effectiveness_norm": imbalance_effectiveness_norm,
            "auction_showdown_norm": auction_showdown_norm,
            "closing_conviction_norm": closing_conviction_norm,
            "intraday_energy_density_norm": intraday_energy_density_norm,
            "intraday_thrust_purity_norm": intraday_thrust_purity_norm,
            "price_thrust_divergence_norm": price_thrust_divergence_norm,
            "trend_efficiency_ratio_norm": trend_efficiency_ratio_norm,
            "loser_concentration_90pct_norm": loser_concentration_90pct_norm,
            "winner_loser_momentum_norm": winner_loser_momentum_norm,
            "cost_structure_skewness_norm": cost_structure_skewness_norm,
            "cost_gini_coefficient_norm": cost_gini_coefficient_norm,
            "mf_vpoc_premium_norm": mf_vpoc_premium_norm,
            "character_score_norm": character_score_norm,
            "signal_conviction_score_norm": signal_conviction_score_norm,
            "touch_conviction_score_norm": touch_conviction_score_norm,
            "gathering_by_chasing_norm": gathering_by_chasing_norm,
            "gathering_by_support_norm": gathering_by_support_norm,
            "volatility_instability_norm": volatility_instability_norm,
            "adx_norm": adx_norm,
            "buy_flow_efficiency_norm": buy_flow_efficiency_norm,
            "sell_flow_efficiency_norm": sell_flow_efficiency_norm,
            "auction_closing_position_norm": auction_closing_position_norm,
            "auction_impact_score_norm": auction_impact_score_norm,
            "auction_intent_signal_norm": auction_intent_signal_norm,
            "order_book_liquidity_supply_norm": order_book_liquidity_supply_norm,
            "liquidity_slope_norm": liquidity_slope_norm,
            "peak_mass_transfer_rate_norm": peak_mass_transfer_rate_norm,
            "mf_cost_zone_defense_intent_norm": mf_cost_zone_defense_intent_norm,
            "bid_side_liquidity_norm": bid_side_liquidity_norm,
            "ask_side_liquidity_norm": ask_side_liquidity_norm,
            "retail_panic_surrender_norm": retail_panic_surrender_norm,
            "main_force_activity_ratio_norm": main_force_activity_ratio_norm,
            "main_force_conviction_index_norm": main_force_conviction_index_norm,
            "main_force_execution_alpha_norm": main_force_execution_alpha_norm,
            "main_force_flow_directionality_norm": main_force_flow_directionality_norm,
            "main_force_on_peak_buy_flow_norm": main_force_on_peak_buy_flow_norm,
            "main_force_on_peak_sell_flow_norm": main_force_on_peak_sell_flow_norm,
            "main_force_t0_efficiency_norm": main_force_t0_efficiency_norm,
            "main_force_t0_sell_efficiency_norm": main_force_t0_sell_efficiency_norm,
            "main_force_vwap_down_guidance_norm": main_force_vwap_down_guidance_norm,
            "main_force_vwap_up_guidance_norm": main_force_vwap_up_guidance_norm,
            "market_impact_cost_norm": market_impact_cost_norm,
            "opening_buy_strength_norm": opening_buy_strength_norm,
            "opening_sell_strength_norm": opening_sell_strength_norm,
            "closing_strength_index_norm": closing_strength_index_norm,
            "total_buy_amount_calibrated_norm": total_buy_amount_calibrated_norm,
            "total_sell_amount_calibrated_norm": total_sell_amount_calibrated_norm,
            "wash_trade_intensity_norm": wash_trade_intensity_norm,
            "winner_profit_margin_avg_norm": winner_profit_margin_avg_norm,
            "loser_loss_margin_avg_norm": loser_loss_margin_avg_norm,
            "total_loser_rate_norm": total_loser_rate_norm,
            "impulse_quality_ratio_norm": impulse_quality_ratio_norm,
            "thrust_efficiency_score_norm": thrust_efficiency_score_norm,
            "platform_conviction_score_norm": platform_conviction_score_norm,
            "platform_high_norm": platform_high_norm,
            "platform_low_norm": platform_low_norm,
            "breakout_quality_score_norm": breakout_quality_score_norm,
            "breakout_readiness_score_norm": breakout_readiness_score_norm,
            "breakthrough_conviction_score_norm": breakthrough_conviction_score_norm,
            "defense_solidity_score_norm": defense_solidity_score_norm,
            "support_validation_strength_norm": support_validation_strength_norm,
            "covert_accumulation_signal_norm": covert_accumulation_signal_norm,
            "suppressive_accumulation_intensity_norm": suppressive_accumulation_intensity_norm,
            "deception_index_norm": deception_index_norm,
            "deception_lure_long_intensity_norm": deception_lure_long_intensity_norm,
            "equilibrium_compression_index_norm": equilibrium_compression_index_norm,
            "final_charge_intensity_norm": final_charge_intensity_norm,
            "floating_chip_cleansing_efficiency_norm": floating_chip_cleansing_efficiency_norm,
            "hidden_accumulation_intensity_norm": hidden_accumulation_intensity_norm,
            "internal_accumulation_intensity_norm": internal_accumulation_intensity_norm,
            "intraday_posture_score_norm": intraday_posture_score_norm,
            "opening_gap_defense_strength_norm": opening_gap_defense_strength_norm,
            "panic_buy_absorption_contribution_norm": panic_buy_absorption_contribution_norm,
            "panic_sell_volume_contribution_norm": panic_sell_volume_contribution_norm,
            "panic_selling_cascade_norm": panic_selling_cascade_norm,
            "peak_control_transfer_norm": peak_control_transfer_norm,
            "peak_separation_ratio_norm": peak_separation_ratio_norm,
            "price_reversion_velocity_norm": price_reversion_velocity_norm,
            "pullback_depth_ratio_norm": pullback_depth_ratio_norm,
            "quality_score_norm": quality_score_norm,
            "reversal_conviction_rate_norm": reversal_conviction_rate_norm,
            "reversal_power_index_norm": reversal_power_index_norm,
            "reversal_recovery_rate_norm": reversal_recovery_rate_norm,
            "risk_reward_profile_norm": risk_reward_profile_norm,
            "shock_conviction_score_norm": shock_conviction_score_norm,
            "strategic_phase_score_norm": strategic_phase_score_norm,
            "structural_entropy_change_norm": structural_entropy_change_norm,
            "structural_leverage_norm": structural_leverage_norm,
            "structural_node_count_norm": structural_node_count_norm,
            "structural_potential_score_norm": structural_potential_score_norm,
            "support_validation_score_norm": support_validation_score_norm,
            "supportive_distribution_intensity_norm": supportive_distribution_intensity_norm,
            "trend_acceleration_score_norm": trend_acceleration_score_norm,
            "trend_alignment_index_norm": trend_alignment_index_norm,
            "trend_asymmetry_index_norm": trend_asymmetry_index_norm,
            "trend_conviction_score_norm": trend_conviction_score_norm,
            "value_area_migration_norm": value_area_migration_norm,
            "value_area_overlap_pct_norm": value_area_overlap_pct_norm,
            "volatility_asymmetry_index_norm": volatility_asymmetry_index_norm,
            "volume_burstiness_index_norm": volume_burstiness_index_norm,
            "volume_structure_skew_norm": volume_structure_skew_norm,
            "vpin_score_norm": vpin_score_norm,
            "vwap_control_strength_norm": vwap_control_strength_norm,
            "vwap_cross_down_intensity_norm": vwap_cross_down_intensity_norm,
            "vwap_cross_up_intensity_norm": vwap_cross_up_intensity_norm,
            "vwap_crossing_intensity_norm": vwap_crossing_intensity_norm,
            "vwap_mean_reversion_corr_norm": vwap_mean_reversion_corr_norm,
            "vwap_sell_control_strength_norm": vwap_sell_control_strength_norm,
            "winner_stability_index_norm": winner_stability_index_norm,
            "absorption_of_distribution_intensity_norm": absorption_of_distribution_intensity_norm # V11.2
        }
        # --- V10.0 代理信号构建 (替代情报层信号) ---
        # 1. 相对强度代理 (rs_modulator_proxy)
        # 综合价格趋势、趋势活力、突破质量等
        rs_modulator_proxy = (mtf_price_trend.clip(lower=0) * 0.4 + mtf_trend_vitality * 0.3 + mtf_breakout_quality_score * 0.3).clip(0,1)
        rs_modulator = (1 + rs_modulator_proxy * config.get('relative_strength_amplifier', 0.0))
        # 2. 资本属性代理 (capital_modulator_proxy)
        # 综合主力净流向、主力资金集中度、主力买入意愿、主力活跃度、主力信念等
        capital_modulator_proxy = (mtf_mf_net_flow.clip(lower=0) * 0.2 + main_force_flow_gini_inverted_norm * 0.15 + mtf_main_force_buy_ofi * 0.15 +
                                   mtf_main_force_t0_buy_efficiency * 0.1 + main_force_activity_ratio_norm * 0.1 + main_force_conviction_index_norm * 0.1 +
                                   mtf_main_force_execution_alpha.clip(lower=0) * 0.1 + mtf_main_force_flow_directionality.clip(lower=0) * 0.1).clip(0,1)
        capital_modulator = (1 + capital_modulator_proxy * config.get('capital_signature_modulator_weight', 0.0))
        # 3. 市场情绪代理 (market_sentiment_proxy)
        # 综合市场情绪分、散户Fomo、散户恐慌投降（反向）
        market_sentiment_proxy = (mtf_market_sentiment.clip(lower=0) * 0.4 + mtf_retail_fomo * 0.3 + (1 - mtf_retail_panic_surrender) * 0.3).clip(0,1)
        # 4. 流动性潮汐代理 (liquidity_tide_proxy)
        # 综合订单簿流动性供给、流动性斜率、买盘流动性、卖盘流动性（反向）
        liquidity_tide_proxy = (mtf_order_book_liquidity_supply * 0.3 + mtf_liquidity_slope.clip(lower=0) * 0.3 +
                                mtf_bid_side_liquidity * 0.2 + (1 - mtf_ask_side_liquidity) * 0.2).clip(0,1)
        _temp_debug_values["代理信号"] = {
            "rs_modulator_proxy": rs_modulator_proxy,
            "capital_modulator_proxy": capital_modulator_proxy,
            "market_sentiment_proxy": market_sentiment_proxy,
            "liquidity_tide_proxy": liquidity_tide_proxy
        }
        # --- 动态权重计算 (V10.0 升级) ---
        # 市场情境因子：波动率、趋势强度、市场情绪、流动性潮汐
        # 市场稳定性得分：低波动率 + 强趋势
        market_stability_score = (1 - volatility_instability_norm) * adx_norm
        market_stability_norm = self._normalize_series(market_stability_score, df_index, bipolar=False)
        # 市场情绪得分 (0-1范围)
        market_sentiment_norm_unipolar = market_sentiment_proxy
        # 流动性潮汐得分 (0-1范围)
        liquidity_tide_norm_unipolar = liquidity_tide_proxy
        # 综合情境得分 (0-1范围)
        overall_context_score = (market_stability_norm * 0.4 + market_sentiment_norm_unipolar * 0.3 + liquidity_tide_norm_unipolar * 0.3).clip(0,1)
        # 定义基础权重 (可配置)
        base_weights = {
            "aggressiveness": 0.3,
            "control": 0.3,
            "obstacle_clearance": 0.2,
            "risk": 0.2
        }
        # 根据综合情境得分动态调整权重
        dynamic_weights = {}
        for key, base_w in base_weights.items():
            if key in ["aggressiveness", "control"]:
                # 综合情境越好，权重增加越多
                dynamic_weights[key] = base_w * (1 + overall_context_score * 0.4 - (1 - overall_context_score) * 0.1)
            elif key == "obstacle_clearance":
                # 综合情境越好，权重略增
                dynamic_weights[key] = base_w * (1 + overall_context_score * 0.1 - (1 - overall_context_score) * 0.05)
            elif key == "risk":
                # 综合情境越好，风险权重降低越多
                dynamic_weights[key] = base_w * (1 - overall_context_score * 0.4 + (1 - overall_context_score) * 0.1)
            dynamic_weights[key] = dynamic_weights[key].clip(0.05, 0.5) # 限制权重范围
        # 归一化动态权重，确保和为1
        total_dynamic_weight = pd.Series(0.0, index=df_index, dtype=np.float32)
        for key in dynamic_weights:
            total_dynamic_weight += dynamic_weights[key]
        for key in dynamic_weights:
            dynamic_weights[key] = dynamic_weights[key] / total_dynamic_weight
        _temp_debug_values["动态权重"] = {
            "market_stability_score": market_stability_score,
            "market_stability_norm": market_stability_norm,
            "market_sentiment_norm_unipolar": market_sentiment_norm_unipolar,
            "liquidity_tide_norm_unipolar": liquidity_tide_norm_unipolar,
            "overall_context_score": overall_context_score,
            "dynamic_weights_aggressiveness": dynamic_weights["aggressiveness"],
            "dynamic_weights_control": dynamic_weights["control"],
            "dynamic_weights_obstacle_clearance": dynamic_weights["obstacle_clearance"],
            "dynamic_weights_risk": dynamic_weights["risk"]
        }
        # --- 1. 攻击性 (Aggressiveness) ---
        aggressiveness_components = {
            "mtf_price_trend": mtf_price_trend.clip(lower=0),
            "mtf_mf_net_flow": mtf_mf_net_flow.clip(lower=0),
            "price_impact_norm": price_impact_norm.clip(lower=0),
            "impulse_purity_norm": impulse_purity_norm.clip(lower=0),
            "volume_ratio_norm": volume_ratio_norm.clip(lower=0),
            "flow_credibility_norm": flow_credibility_norm,
            "chip_health_norm": chip_health_norm,
            "mtf_buy_sweep_intensity": mtf_buy_sweep_intensity,
            "mtf_main_force_buy_ofi": mtf_main_force_buy_ofi,
            "mtf_main_force_t0_buy_efficiency": mtf_main_force_t0_buy_efficiency,
            "order_book_imbalance_positive_norm": order_book_imbalance_positive_norm,
            "micro_price_impact_asymmetry_positive_norm": micro_price_impact_asymmetry_positive_norm,
            "constructive_turnover_norm": constructive_turnover_norm,
            "mtf_upward_impulse_strength": mtf_upward_impulse_strength,
            "mtf_order_book_clearing_rate": mtf_order_book_clearing_rate,
            "sell_sweep_intensity_inverted_norm": sell_sweep_intensity_inverted_norm,
            "microstructure_efficiency_norm": microstructure_efficiency_norm,
            "imbalance_effectiveness_norm": imbalance_effectiveness_norm,
            "mtf_auction_showdown": mtf_auction_showdown,
            "mtf_closing_conviction": mtf_closing_conviction,
            "mtf_intraday_energy_density": mtf_intraday_energy_density,
            "mtf_intraday_thrust_purity": mtf_intraday_thrust_purity,
            "mtf_buy_flow_efficiency": mtf_buy_flow_efficiency,
            "mtf_sell_flow_efficiency_inverted": (1 - mtf_sell_flow_efficiency), # 卖方效率低，买方攻击性强
            "mtf_auction_closing_position_positive": mtf_auction_closing_position.clip(lower=0), # 集合竞价收盘位置偏高
            "mtf_auction_impact_score": mtf_auction_impact_score,
            "mtf_auction_intent_signal": mtf_auction_intent_signal,
            "mtf_opening_buy_strength": mtf_opening_buy_strength,
            "mtf_closing_strength_index": mtf_closing_strength_index,
            "mtf_total_buy_amount_calibrated": mtf_total_buy_amount_calibrated,
            "mtf_main_force_vwap_up_guidance": mtf_main_force_vwap_up_guidance.clip(lower=0),
            "mtf_impulse_quality_ratio": mtf_impulse_quality_ratio,
            "mtf_thrust_efficiency_score": mtf_thrust_efficiency_score,
            "mtf_breakout_quality_score": mtf_breakout_quality_score,
            "mtf_breakout_readiness_score": mtf_breakout_readiness_score,
            "mtf_breakthrough_conviction_score": mtf_breakthrough_conviction_score,
            "mtf_final_charge_intensity": mtf_final_charge_intensity,
            "mtf_hidden_accumulation_intensity": mtf_hidden_accumulation_intensity,
            "mtf_internal_accumulation_intensity": mtf_internal_accumulation_intensity,
            "mtf_intraday_posture_score": mtf_intraday_posture_score,
            "mtf_panic_buy_absorption_contribution": mtf_panic_buy_absorption_contribution,
            "mtf_reversal_power_index": mtf_reversal_power_index,
            "mtf_trend_acceleration_score": mtf_trend_acceleration_score,
            "mtf_trend_conviction_score": mtf_trend_conviction_score,
            "mtf_vwap_cross_up_intensity": mtf_vwap_cross_up_intensity,
            "mtf_absorption_of_distribution_intensity": mtf_absorption_of_distribution_intensity # 新增：派发吸收强度作为攻击性的一部分
        }
        aggressiveness_weights = {
            "mtf_price_trend": 0.04, "mtf_mf_net_flow": 0.04, "price_impact_norm": 0.03,
            "impulse_purity_norm": 0.03, "volume_ratio_norm": 0.02, "flow_credibility_norm": 0.02,
            "chip_health_norm": 0.02, "mtf_buy_sweep_intensity": 0.05, "mtf_main_force_buy_ofi": 0.05,
            "mtf_main_force_t0_buy_efficiency": 0.05, "order_book_imbalance_positive_norm": 0.03,
            "micro_price_impact_asymmetry_positive_norm": 0.03, "constructive_turnover_norm": 0.03,
            "mtf_upward_impulse_strength": 0.03, "mtf_order_book_clearing_rate": 0.02,
            "sell_sweep_intensity_inverted_norm": 0.02, "microstructure_efficiency_norm": 0.02,
            "imbalance_effectiveness_norm": 0.02, "mtf_auction_showdown": 0.01,
            "mtf_closing_conviction": 0.01, "mtf_intraday_energy_density": 0.01,
            "mtf_intraday_thrust_purity": 0.01, "mtf_buy_flow_efficiency": 0.02,
            "mtf_sell_flow_efficiency_inverted": 0.01, "mtf_auction_closing_position_positive": 0.01,
            "mtf_auction_impact_score": 0.01, "mtf_auction_intent_signal": 0.01,
            "mtf_opening_buy_strength": 0.02, "mtf_closing_strength_index": 0.02,
            "mtf_total_buy_amount_calibrated": 0.02, "mtf_main_force_vwap_up_guidance": 0.02,
            "mtf_impulse_quality_ratio": 0.02, "mtf_thrust_efficiency_score": 0.02,
            "mtf_breakout_quality_score": 0.02, "mtf_breakout_readiness_score": 0.02,
            "mtf_breakthrough_conviction_score": 0.02, "mtf_final_charge_intensity": 0.02,
            "mtf_hidden_accumulation_intensity": 0.02, "mtf_internal_accumulation_intensity": 0.02,
            "mtf_intraday_posture_score": 0.02, "mtf_panic_buy_absorption_contribution": 0.02,
            "mtf_reversal_power_index": 0.02, "mtf_trend_acceleration_score": 0.02,
            "mtf_trend_conviction_score": 0.02, "mtf_vwap_cross_up_intensity": 0.02,
            "mtf_absorption_of_distribution_intensity": 0.05 # 提高权重，从0.03调整为0.05
        }
        aggressiveness_score = _robust_geometric_mean(aggressiveness_components, aggressiveness_weights, df_index).clip(0, 1)
        _temp_debug_values["攻击性"] = {
            "aggressiveness_score": aggressiveness_score
        }
        # --- 2. 控制力 (Control) ---
        control_components = {
            "control_solidity_norm": control_solidity_norm.clip(lower=0),
            "cost_advantage_norm": cost_advantage_norm.clip(lower=0),
            "concentration_slope_norm": concentration_slope_norm.clip(lower=0),
            "peak_solidity_norm": peak_solidity_norm.clip(lower=0),
            "mtf_vwap_buy_control_strength": mtf_vwap_buy_control_strength,
            "mtf_mf_cost_zone_buy_intent": mtf_mf_cost_zone_buy_intent,
            "chip_fault_blockage_ratio_inverted_norm": chip_fault_blockage_ratio_inverted_norm,
            "main_force_flow_gini_inverted_norm": main_force_flow_gini_inverted_norm,
            "mtf_cost_structure_skewness_positive": mtf_cost_structure_skewness.clip(lower=0), # 成本结构偏度正向
            "cost_gini_coefficient_norm": cost_gini_coefficient_norm,
            "mtf_mf_vpoc_premium_positive": mtf_mf_vpoc_premium.clip(lower=0), # VPOC溢价正向
            "mtf_mf_cost_zone_defense_intent_inverted": (1 - mtf_mf_cost_zone_defense_intent), # 成本区防守意图低，控盘强
            "mtf_main_force_activity_ratio": mtf_main_force_activity_ratio,
            "mtf_main_force_conviction_index": mtf_main_force_conviction_index,
            "mtf_main_force_execution_alpha": mtf_main_force_execution_alpha.clip(lower=0),
            "mtf_main_force_flow_directionality": mtf_main_force_flow_directionality.clip(lower=0),
            "mtf_main_force_on_peak_buy_flow": mtf_main_force_on_peak_buy_flow,
            "mtf_main_force_t0_efficiency": mtf_main_force_t0_efficiency,
            "mtf_main_force_vwap_up_guidance": mtf_main_force_vwap_up_guidance.clip(lower=0),
            "mtf_platform_conviction_score": mtf_platform_conviction_score,
            "mtf_peak_control_transfer": mtf_peak_control_transfer,
            "mtf_vwap_control_strength": mtf_vwap_control_strength,
            "mtf_winner_stability_index": mtf_winner_stability_index
        }
        control_weights = {
            "control_solidity_norm": 0.08, "cost_advantage_norm": 0.07, "concentration_slope_norm": 0.07,
            "peak_solidity_norm": 0.06, "mtf_vwap_buy_control_strength": 0.06,
            "mtf_mf_cost_zone_buy_intent": 0.06, "chip_fault_blockage_ratio_inverted_norm": 0.04,
            "main_force_flow_gini_inverted_norm": 0.04, "mtf_cost_structure_skewness_positive": 0.04,
            "cost_gini_coefficient_norm": 0.04, "mtf_mf_vpoc_premium_positive": 0.04,
            "mtf_mf_cost_zone_defense_intent_inverted": 0.05, "mtf_main_force_activity_ratio": 0.04,
            "mtf_main_force_conviction_index": 0.04, "mtf_main_force_execution_alpha": 0.04,
            "mtf_main_force_flow_directionality": 0.04, "mtf_main_force_on_peak_buy_flow": 0.03,
            "mtf_main_force_t0_efficiency": 0.03, "mtf_main_force_vwap_up_guidance": 0.03,
            "mtf_platform_conviction_score": 0.03, "mtf_peak_control_transfer": 0.03,
            "mtf_vwap_control_strength": 0.03, "mtf_winner_stability_index": 0.03
        }
        control_score = _robust_geometric_mean(control_components, control_weights, df_index).clip(0, 1)
        # V11.1: 应用筹码集中度稳定性调节器
        if hc_enabled:
            control_score = (control_score * (1 + mtf_chip_concentration_stability * chip_concentration_stability_modulator_factor)).clip(0, 1)
        _temp_debug_values["控制力"] = {
            "control_score": control_score
        }
        # --- 3. 障碍清除 (Obstacle Clearance) ---
        obstacle_clearance_components = {
            "buying_support_norm": buying_support_norm.clip(lower=0),
            "pressure_rejection_norm": pressure_rejection_norm.clip(lower=0),
            "profit_absorption_norm": profit_absorption_norm.clip(lower=0),
            "mtf_vacuum_traversal_efficiency": mtf_vacuum_traversal_efficiency,
            "vacuum_zone_magnitude_norm": vacuum_zone_magnitude_norm,
            "mtf_dip_buy_absorption_strength": mtf_dip_buy_absorption_strength,
            "rally_buy_support_weakness_inverted_norm": rally_buy_support_weakness_inverted_norm,
            "mtf_price_thrust_divergence_positive": mtf_price_thrust_divergence.clip(lower=0), # 价格推力正向
            "mtf_trend_efficiency_ratio": mtf_trend_efficiency_ratio,
            "mtf_order_book_liquidity_supply": mtf_order_book_liquidity_supply,
            "mtf_liquidity_slope_positive": mtf_liquidity_slope.clip(lower=0), # 流动性斜率正向
            "bid_side_liquidity_norm": bid_side_liquidity_norm,
            "ask_side_liquidity_inverted_norm": (1 - ask_side_liquidity_norm),
            "mtf_defense_solidity_score": mtf_defense_solidity_score,
            "mtf_support_validation_strength": mtf_support_validation_strength,
            "mtf_floating_chip_cleansing_efficiency": mtf_floating_chip_cleansing_efficiency,
            "mtf_opening_gap_defense_strength": mtf_opening_gap_defense_strength,
            "mtf_peak_separation_ratio": mtf_peak_separation_ratio,
            "mtf_pullback_depth_ratio_inverted": (1 - mtf_pullback_depth_ratio),
            "mtf_support_validation_score": mtf_support_validation_score,
            "mtf_value_area_migration_positive": mtf_value_area_migration.clip(lower=0),
            "mtf_value_area_overlap_pct": mtf_value_area_overlap_pct
        }
        obstacle_clearance_weights = {
            "buying_support_norm": 0.08, "pressure_rejection_norm": 0.07, "profit_absorption_norm": 0.07,
            "mtf_vacuum_traversal_efficiency": 0.06, "vacuum_zone_magnitude_norm": 0.03,
            "mtf_dip_buy_absorption_strength": 0.08, "rally_buy_support_weakness_inverted_norm": 0.03,
            "mtf_price_thrust_divergence_positive": 0.03, "mtf_trend_efficiency_ratio": 0.06,
            "mtf_order_book_liquidity_supply": 0.05, "mtf_liquidity_slope_positive": 0.04,
            "bid_side_liquidity_norm": 0.04, "ask_side_liquidity_inverted_norm": 0.03,
            "mtf_defense_solidity_score": 0.04, "mtf_support_validation_strength": 0.04,
            "mtf_floating_chip_cleansing_efficiency": 0.03, "mtf_opening_gap_defense_strength": 0.03,
            "mtf_peak_separation_ratio": 0.03, "mtf_pullback_depth_ratio_inverted": 0.03,
            "mtf_support_validation_score": 0.03, "mtf_value_area_migration_positive": 0.03,
            "mtf_value_area_overlap_pct": 0.03
        }
        obstacle_clearance_score = _robust_geometric_mean(obstacle_clearance_components, obstacle_clearance_weights, df_index).clip(0, 1)
        _temp_debug_values["障碍清除"] = {
            "obstacle_clearance_score": obstacle_clearance_score
        }
        # --- 4. 基础看涨意图合成 ---
        # V11.1: 引入长期趋势强度上下文调节器
        long_term_trend_strength_modulator = (1 + mtf_long_term_trend_strength * long_term_trend_modulator_factor) # 长期趋势越强，看涨意图越受加成

        # 攻击性、控制力、障碍清除的加权平均
        bullish_intent_base = (
            (aggressiveness_score * dynamic_weights["aggressiveness"] +
             control_score * dynamic_weights["control"] +
             obstacle_clearance_score * dynamic_weights["obstacle_clearance"]) /
            (dynamic_weights["aggressiveness"] + dynamic_weights["control"] + dynamic_weights["obstacle_clearance"])
        )
        # V11.3: 强化 mtf_absorption_of_distribution_intensity 的积极作用，直接贡献于看涨意图
        # 承接强度越高，看涨意图越强，这里使用一个较小的系数，避免过度放大
        bullish_intent = (bullish_intent_base + mtf_absorption_of_distribution_intensity * 0.1).clip(0, 1)

        # V11.1: 应用长期趋势强度调节
        bullish_intent = (bullish_intent * long_term_trend_strength_modulator).clip(0, 1)

        # 幂平均，放大高分，抑制低分
        power_mean_exponent = get_param_value(rally_intent_synthesis_params.get('power_mean_exponent'), 2.0)
        bullish_intent = bullish_intent.pow(power_mean_exponent)
        _temp_debug_values["基础看涨意图"] = {
            "power_mean_exponent": power_mean_exponent,
            "bullish_intent_base": bullish_intent_base, # 增加基础看涨意图的调试输出
            "bullish_intent": bullish_intent
        }
        # --- 5. 看跌意图 ---
        # V11.1: 引入主力资金累计记忆调节器
        # 历史累计资金流入强劲 (mtf_cumulative_mf_flow > 0)，削弱看跌意图
        mf_flow_memory_anti_bearish_modulator = (1 - mtf_cumulative_mf_flow.clip(lower=0) * cumulative_mf_flow_modulator_factor).clip(0, 1)

        # 看跌意图的计算
        distribution_intensity_norm = self._normalize_series(distribution_at_peak_intensity, df_index, bipolar=False)
        upper_shadow_selling_pressure_norm = self._normalize_series(upper_shadow_selling_pressure, df_index, bipolar=False)
        flow_credibility_norm_inverted = (1 - flow_credibility_norm) # 信用度低，看跌
        
        # V11.2: 引入派发情境衰减器 (Distribution Context Dampener)
        # 当日涨幅越大，对派发强度的看跌解读越弱
        # 使用 tanh 函数将涨幅映射到 [0, 1] 范围内的衰减因子
        # 例如，pct_change = 7% -> tanh(0.07 * 10) = tanh(0.7) = 0.6，衰减因子为 1 - 0.6 = 0.4
        # pct_change = 0% -> tanh(0) = 0，衰减因子为 1
        # pct_change = -5% -> tanh(-0.5) = -0.46，衰减因子为 1 - (-0.46) = 1.46 (负涨幅反而增强派发风险)
        # 确保衰减因子在合理范围，例如 [0.1, 1.5]
        distribution_dampener = (1 - np.tanh(pct_change / 100 * 10)).clip(0.1, 1.5) # 涨幅越大，dampener越小，削弱派发影响

        # 修正 bearish_score 的计算逻辑：它应该是一个正值，表示看跌意图的强度，然后乘以 -1 转换为双极性。
        # 并且 mf_flow_memory_anti_bearish_modulator 应该削弱看跌意图，而不是使其变得更负。
        bearish_score_raw = (
            (distribution_intensity_norm * distribution_dampener * 0.4 + # 应用衰减器
             upper_shadow_selling_pressure_norm * 0.3 +
             flow_credibility_norm_inverted * 0.3)
        ).clip(0, 1) # 确保原始看跌分数在 [0, 1] 之间

        # V11.1: 应用主力资金累计记忆调节，削弱看跌意图
        bearish_score_modulated = (bearish_score_raw * mf_flow_memory_anti_bearish_modulator).clip(0, 1)
        
        # 转换为负值，表示看跌
        bearish_score = -bearish_score_modulated
        _temp_debug_values["看跌意图"] = {
            "distribution_dampener": distribution_dampener, # 增加衰减器调试输出
            "bearish_score_raw": bearish_score_raw, # 增加原始看跌分数的调试输出
            "mf_flow_memory_anti_bearish_modulator": mf_flow_memory_anti_bearish_modulator, # 增加调节器的调试输出
            "bearish_score_modulated": bearish_score_modulated, # 增加调节后的看跌分数的调试输出
            "bearish_score": bearish_score
        }
        # --- 6. 风险审判模块 (Risk Adjudication) ---
        # 6.1. 派发风险 (Distribution Risk)
        mf_outflow_divergence = mtf_mf_net_flow.clip(upper=0).abs()
        distribution_risk_components = {
            "distribution_intensity_norm": distribution_intensity_norm,
            "mtf_upper_shadow_pressure": mtf_upper_shadow_pressure,
            "mf_outflow_divergence": mf_outflow_divergence,
            "mtf_retail_fomo": mtf_retail_fomo,
            "mtf_covert_distribution": mtf_covert_distribution,
            "mtf_deception_lure_short": mtf_deception_lure_short,
            "mtf_rally_distribution_pressure": mtf_rally_distribution_pressure,
            "mtf_exhaustion_risk": mtf_exhaustion_risk,
            "mtf_asymmetric_friction": mtf_asymmetric_friction,
            "mtf_volatility_expansion": mtf_volatility_expansion,
            "mtf_loser_concentration_90pct": mtf_loser_concentration_90pct,
            "mtf_winner_loser_momentum_negative": mtf_winner_loser_momentum.clip(upper=0).abs(), # 赢家动量减弱，输家动量增强
            "mtf_sell_flow_efficiency": mtf_sell_flow_efficiency, # 卖方效率高，派发风险高
            "mtf_sell_sweep_intensity": mtf_sell_sweep_intensity, # 卖出扫单强度高，派发风险高
            "mtf_peak_mass_transfer_rate": mtf_peak_mass_transfer_rate, # 筹码转移率高，派发风险高
            "mtf_main_force_on_peak_sell_flow": mtf_main_force_on_peak_sell_flow,
            "mtf_main_force_t0_sell_efficiency": mtf_main_force_t0_sell_efficiency,
            "mtf_main_force_vwap_down_guidance": mtf_main_force_vwap_down_guidance.clip(upper=0).abs(),
            "mtf_market_impact_cost": mtf_market_impact_cost,
            "mtf_opening_sell_strength": mtf_opening_sell_strength,
            "mtf_total_sell_amount_calibrated": mtf_total_sell_amount_calibrated,
            "mtf_wash_trade_intensity": mtf_wash_trade_intensity,
            "mtf_winner_profit_margin_avg_negative": mtf_winner_profit_margin_avg.clip(upper=0).abs(),
            "mtf_loser_loss_margin_avg_positive": mtf_loser_loss_margin_avg.clip(lower=0),
            "mtf_total_winner_rate_inverted": (1 - mtf_total_winner_rate),
            "mtf_total_loser_rate": mtf_total_loser_rate,
            "mtf_deception_index": mtf_deception_index,
            "mtf_deception_lure_long_intensity": mtf_deception_lure_long_intensity,
            "mtf_panic_sell_volume_contribution": mtf_panic_sell_volume_contribution,
            "mtf_panic_selling_cascade": mtf_panic_selling_cascade,
            "mtf_price_reversion_velocity_negative": mtf_price_reversion_velocity.clip(upper=0).abs(),
            "mtf_pullback_depth_ratio": mtf_pullback_depth_ratio,
            "mtf_risk_reward_profile_negative": mtf_risk_reward_profile.clip(upper=0).abs(),
            "mtf_shock_conviction_score": mtf_shock_conviction_score,
            "mtf_structural_entropy_change_positive": mtf_structural_entropy_change.clip(lower=0),
            "mtf_structural_leverage": mtf_structural_leverage,
            "mtf_structural_node_count": mtf_structural_node_count,
            "mtf_structural_potential_score": mtf_structural_potential_score,
            "mtf_supportive_distribution_intensity": mtf_supportive_distribution_intensity,
            "mtf_trend_asymmetry_index_negative": mtf_trend_asymmetry_index.clip(upper=0).abs(),
            "mtf_value_area_migration_negative": mtf_value_area_migration.clip(upper=0).abs(),
            "mtf_volatility_asymmetry_index_positive": mtf_volatility_asymmetry_index.clip(lower=0),
            "mtf_volume_burstiness_index": mtf_volume_burstiness_index,
            "mtf_volume_structure_skew_positive": mtf_volume_structure_skew.clip(lower=0),
            "mtf_vpin_score": mtf_vpin_score,
            "mtf_vwap_cross_down_intensity": mtf_vwap_cross_down_intensity,
            "mtf_vwap_sell_control_strength": mtf_vwap_sell_control_strength,
            "mtf_absorption_of_distribution_intensity_inverted": (1 - mtf_absorption_of_distribution_intensity) # 新增：派发吸收强度反向，承接越强，风险越低
        }
        distribution_risk_weights = {
            "distribution_intensity_norm": 0.03, "mtf_upper_shadow_pressure": 0.03,
            "mf_outflow_divergence": 0.03, "mtf_retail_fomo": 0.02,
            "mtf_covert_distribution": 0.03, "mtf_deception_lure_short": 0.03,
            "mtf_rally_distribution_pressure": 0.03, "mtf_exhaustion_risk": 0.02,
            "mtf_asymmetric_friction": 0.02, "mtf_volatility_expansion": 0.02,
            "mtf_loser_concentration_90pct": 0.02, "mtf_winner_loser_momentum_negative": 0.02,
            "mtf_sell_flow_efficiency": 0.02, "mtf_sell_sweep_intensity": 0.02,
            "mtf_peak_mass_transfer_rate": 0.02, "mtf_main_force_on_peak_sell_flow": 0.02,
            "mtf_main_force_t0_sell_efficiency": 0.02, "mtf_main_force_vwap_down_guidance": 0.02,
            "mtf_market_impact_cost": 0.02, "mtf_opening_sell_strength": 0.02,
            "mtf_total_sell_amount_calibrated": 0.02, "mtf_wash_trade_intensity": 0.02,
            "mtf_winner_profit_margin_avg_negative": 0.02, "mtf_loser_loss_margin_avg_positive": 0.02,
            "mtf_total_winner_rate_inverted": 0.02, "mtf_total_loser_rate": 0.02,
            "mtf_deception_index": 0.02, "mtf_deception_lure_long_intensity": 0.02,
            "mtf_panic_sell_volume_contribution": 0.02, "mtf_panic_selling_cascade": 0.02,
            "mtf_price_reversion_velocity_negative": 0.02, "mtf_pullback_depth_ratio": 0.02,
            "mtf_risk_reward_profile_negative": 0.02, "mtf_shock_conviction_score": 0.02,
            "mtf_structural_entropy_change_positive": 0.02, "mtf_structural_leverage": 0.02,
            "mtf_structural_node_count": 0.02, "mtf_structural_potential_score": 0.02,
            "mtf_supportive_distribution_intensity": 0.02, "mtf_trend_asymmetry_index_negative": 0.02,
            "mtf_value_area_migration_negative": 0.02, "mtf_volatility_asymmetry_index_positive": 0.02,
            "mtf_volume_burstiness_index": 0.02, "mtf_volume_structure_skew_positive": 0.02,
            "mtf_vpin_score": 0.02, "mtf_vwap_cross_down_intensity": 0.02,
            "mtf_vwap_sell_control_strength": 0.02,
            "mtf_absorption_of_distribution_intensity_inverted": 0.05 # 提高权重，从0.03调整为0.05
        }
        distribution_risk_score = _robust_geometric_mean(distribution_risk_components, distribution_risk_weights, df_index).clip(0, 1)
        _temp_debug_values["派发风险"] = {
            "distribution_risk_score": distribution_risk_score
        }
        # 6.2. 前置下跌风险 (Pre-Drop Risk) - 深度情境感知
        pre_5day_pct_change = close_price.pct_change(periods=5).shift(1).fillna(0)
        pre_13day_pct_change = close_price.pct_change(periods=13).shift(1).fillna(0)
        norm_pre_drop_5d = self._normalize_series(pre_5day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        norm_pre_drop_13d = self._normalize_series(pre_13day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        single_day_drop_risk = self._normalize_series(prev_day_pct_change.clip(upper=0).abs(), df_index, bipolar=False)
        norm_slope_21_neg = self._normalize_series(slope_21_close.clip(upper=0).abs(), df_index, bipolar=False)
        norm_accel_21_neg = self._normalize_series(accel_21_close.clip(upper=0).abs(), df_index, bipolar=False)
        norm_slope_34_neg = self._normalize_series(slope_34_close.clip(upper=0).abs(), df_index, bipolar=False)
        norm_accel_34_neg = self._normalize_series(accel_34_close.clip(upper=0).abs(), df_index, bipolar=False)
        medium_term_downtrend_strength = (norm_slope_21_neg * 0.3 + norm_accel_21_neg * 0.2 + 
                                          norm_slope_34_neg * 0.3 + norm_accel_34_neg * 0.2).clip(0, 1)
        high_21d = close_price.rolling(window=21).max()
        fall_from_peak_21d = (1 - close_price / high_21d).clip(lower=0).fillna(0)
        norm_fall_from_peak_21d = self._normalize_series(fall_from_peak_21d, df_index, bipolar=False)
        pre_drop_risk_components = {
            "single_day_drop_risk": single_day_drop_risk,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "norm_pre_drop_13d": norm_pre_drop_13d,
            "medium_term_downtrend_strength": medium_term_downtrend_strength,
            "norm_fall_from_peak_21d": norm_fall_from_peak_21d,
            "mtf_price_thrust_divergence_negative": mtf_price_thrust_divergence.clip(upper=0).abs(), # 价格推力负向
            "mtf_trend_efficiency_ratio_inverted": (1 - mtf_trend_efficiency_ratio), # 趋势效率低，风险高
            "mtf_loser_concentration_90pct": mtf_loser_concentration_90pct, # 输家集中度高，风险高
            "mtf_main_force_vwap_down_guidance_negative": mtf_main_force_vwap_down_guidance.clip(upper=0).abs(),
            "mtf_platform_high": mtf_platform_high,
            "mtf_platform_low_inverted": (1 - mtf_platform_low),
            "mtf_pullback_depth_ratio": mtf_pullback_depth_ratio,
            "mtf_reversal_conviction_rate_inverted": (1 - mtf_reversal_conviction_rate),
            "mtf_reversal_recovery_rate_inverted": (1 - mtf_reversal_recovery_rate),
            "mtf_risk_reward_profile_negative": mtf_risk_reward_profile.clip(upper=0).abs(),
            "mtf_shock_conviction_score": mtf_shock_conviction_score,
            "mtf_structural_entropy_change_negative": mtf_structural_entropy_change.clip(upper=0).abs(),
            "mtf_structural_leverage": mtf_structural_leverage,
            "mtf_structural_node_count": mtf_structural_node_count,
            "mtf_structural_potential_score": mtf_structural_potential_score,
            "mtf_trend_alignment_index_inverted": (1 - mtf_trend_alignment_index),
            "mtf_trend_asymmetry_index_negative": mtf_trend_asymmetry_index.clip(upper=0).abs(),
            "mtf_vwap_cross_down_intensity": mtf_vwap_cross_down_intensity,
            "mtf_vwap_mean_reversion_corr_negative": mtf_vwap_mean_reversion_corr.clip(upper=0).abs()
        }
        pre_drop_risk_weights = {
            "single_day_drop_risk": 0.08, "norm_pre_drop_5d": 0.08, "norm_pre_drop_13d": 0.05,
            "medium_term_downtrend_strength": 0.15, "norm_fall_from_peak_21d": 0.06,
            "mtf_price_thrust_divergence_negative": 0.06, "mtf_trend_efficiency_ratio_inverted": 0.05,
            "mtf_loser_concentration_90pct": 0.05, "mtf_main_force_vwap_down_guidance_negative": 0.05,
            "mtf_platform_high": 0.04, "mtf_platform_low_inverted": 0.04,
            "mtf_pullback_depth_ratio": 0.04, "mtf_reversal_conviction_rate_inverted": 0.04,
            "mtf_reversal_recovery_rate_inverted": 0.04, "mtf_risk_reward_profile_negative": 0.04,
            "mtf_shock_conviction_score": 0.04, # 保持冲击信念的风险贡献，但后续会调整其在上涨行情中的解读
            "mtf_structural_entropy_change_negative": 0.04, "mtf_structural_leverage": 0.04,
            "mtf_structural_node_count": 0.04, "mtf_structural_potential_score": 0.04,
            "mtf_trend_alignment_index_inverted": 0.04, "mtf_trend_asymmetry_index_negative": 0.04,
            "mtf_vwap_cross_down_intensity": 0.04,
            "mtf_vwap_mean_reversion_corr_negative": 0.04
        }
        pre_drop_risk_factor = _robust_geometric_mean(pre_drop_risk_components, pre_drop_risk_weights, df_index).clip(0, 1) * 0.7 # 整体风险因子权重
        _temp_debug_values["前置下跌风险"] = {
            "pre_5day_pct_change": pre_5day_pct_change,
            "pre_13day_pct_change": pre_13day_pct_change,
            "norm_pre_drop_5d": norm_pre_drop_5d,
            "norm_pre_drop_13d": norm_pre_drop_13d,
            "single_day_drop_risk": single_day_drop_risk,
            "norm_slope_21_neg": norm_slope_21_neg,
            "norm_accel_21_neg": norm_accel_21_neg,
            "norm_slope_34_neg": norm_slope_34_neg,
            "norm_accel_34_neg": norm_accel_34_neg,
            "medium_term_downtrend_strength": medium_term_downtrend_strength,
            "high_21d": high_21d,
            "fall_from_peak_21d": fall_from_peak_21d,
            "norm_fall_from_peak_21d": norm_fall_from_peak_21d,
            "pre_drop_risk_factor": pre_drop_risk_factor
        }
        # 6.3. 综合风险惩罚因子 - V10.0 非线性惩罚优化
        # 风险权重根据动态权重调整
        total_risk_penalty_raw = (distribution_risk_score * dynamic_weights["risk"] + pre_drop_risk_factor * (1 - dynamic_weights["risk"])).clip(0, 1)
        # 应用Sigmoid函数进行非线性惩罚
        # risk_sensitivity 和 sigmoid_center 从 config 中获取
        # V11.3: 进一步调整 risk_sensitivity 和 sigmoid_center
        total_risk_penalty = 1 / (1 + np.exp(risk_sensitivity * (total_risk_penalty_raw - sigmoid_center)))
        # 归一化为惩罚因子，高风险对应高惩罚 (1-sigmoid_output)
        total_risk_penalty = (1 - total_risk_penalty).clip(0, 1)
        _temp_debug_values["综合风险惩罚因子"] = {
            "total_risk_penalty_raw": total_risk_penalty_raw,
            "total_risk_penalty": total_risk_penalty
        }
        # --- 7. 最终意图合成 ---
        penalized_bullish_part = bullish_intent * (1 - total_risk_penalty)
        final_rally_intent = (penalized_bullish_part + bearish_score).clip(-1, 1)
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (total_risk_penalty > 0.5), final_rally_intent * (1 - total_risk_penalty))
        final_rally_intent = final_rally_intent.mask(is_limit_up_day & (final_rally_intent < 0), 0.0)
        _temp_debug_values["最终意图合成"] = {
            "penalized_bullish_part": penalized_bullish_part,
            "final_rally_intent_before_mod": final_rally_intent
        }
        # --- 8. 相对强度和资本属性调节 (V10.0 使用代理信号) ---
        # rs_modulator 和 capital_modulator 已在上方使用代理信号计算
        # V10.0 情境调节器 (基于原始MTF信号)
        market_sentiment_modulator = (1 + mtf_market_sentiment * 0.1) # 市场情绪越好，意图越强
        structural_tension_modulator = (1 - mtf_structural_tension * 0.1) # 结构张力越大，意图越弱
        trend_vitality_modulator = (1 + mtf_trend_vitality * 0.1) # 趋势活力越强，意图越强
        liquidity_authenticity_modulator = (1 + mtf_liquidity_authenticity * 0.05) # 流动性越真实，意图越强
        # V10.0 行为心理调节器 (基于原始MTF信号)
        character_score_modulator = (1 + mtf_character_score * 0.05)
        signal_conviction_modulator = (1 + mtf_signal_conviction_score * 0.05)
        touch_conviction_modulator = (1 + mtf_touch_conviction_score * 0.05)
        gathering_by_chasing_modulator = (1 + mtf_gathering_by_chasing * 0.05)
        gathering_by_support_modulator = (1 + mtf_gathering_by_support * 0.05)
        # V10.0 微观结构调节器 (基于原始MTF信号)
        microstructure_efficiency_modulator = (1 + mtf_microstructure_efficiency * 0.05)
        imbalance_effectiveness_modulator = (1 + mtf_imbalance_effectiveness * 0.05)
        auction_intent_modulator = (1 + mtf_auction_intent_signal * 0.05) # 集合竞价意图
        final_rally_intent = (final_rally_intent * rs_modulator * capital_modulator *
                              market_sentiment_modulator * structural_tension_modulator *
                              trend_vitality_modulator * liquidity_authenticity_modulator *
                              character_score_modulator * signal_conviction_modulator *
                              touch_conviction_modulator * gathering_by_chasing_modulator *
                              gathering_by_support_modulator * microstructure_efficiency_modulator *
                              imbalance_effectiveness_modulator * auction_intent_modulator).clip(-1, 1)
        _temp_debug_values["相对强度和资本属性调节"] = {
            "rs_modulator": rs_modulator,
            "capital_modulator": capital_modulator,
            "market_sentiment_modulator": market_sentiment_modulator,
            "structural_tension_modulator": structural_tension_modulator,
            "trend_vitality_modulator": trend_vitality_modulator,
            "liquidity_authenticity_modulator": liquidity_authenticity_modulator,
            "character_score_modulator": character_score_modulator,
            "signal_conviction_modulator": signal_conviction_modulator,
            "touch_conviction_modulator": touch_conviction_modulator,
            "gathering_by_chasing_modulator": gathering_by_chasing_modulator,
            "gathering_by_support_modulator": gathering_by_support_modulator,
            "microstructure_efficiency_modulator": microstructure_efficiency_modulator,
            "imbalance_effectiveness_modulator": imbalance_effectiveness_modulator,
            "auction_intent_modulator": auction_intent_modulator,
            "final_rally_intent": final_rally_intent
        }
        self.strategy.atomic_states["_DEBUG_rally_aggressiveness"] = aggressiveness_score
        self.strategy.atomic_states["_DEBUG_rally_control"] = control_score
        self.strategy.atomic_states["_DEBUG_rally_obstacle_clearance"] = obstacle_clearance_score
        self.strategy.atomic_states["_DEBUG_rally_bullish_intent"] = bullish_intent
        self.strategy.atomic_states["_DEBUG_rally_bearish_score"] = bearish_score
        self.strategy.atomic_states["_DEBUG_rally_distribution_risk"] = distribution_risk_score
        self.strategy.atomic_states["_DEBUG_rally_pre_drop_risk_factor"] = pre_drop_risk_factor
        self.strategy.atomic_states["_DEBUG_rally_total_risk_penalty"] = total_risk_penalty
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派生信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["派生信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF融合信号 ---"] = ""
            for key, series in _temp_debug_values["MTF融合信号"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 代理信号 ---"] = ""
            for key, series in _temp_debug_values["代理信号"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态权重 ---"] = ""
            for key, series in _temp_debug_values["动态权重"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 攻击性 ---"] = ""
            for key, series in _temp_debug_values["攻击性"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 控制力 ---"] = ""
            for key, series in _temp_debug_values["控制力"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 障碍清除 ---"] = ""
            for key, series in _temp_debug_values["障碍清除"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础看涨意图 ---"] = ""
            for key, value in _temp_debug_values["基础看涨意图"].items(): # 这里的value可能是float
                if isinstance(value, pd.Series):
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else: # 如果是标量，直接输出
                    debug_output[f"        {key}: {value:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看跌意图 ---"] = ""
            for key, series in _temp_debug_values["看跌意图"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派发风险 ---"] = ""
            # 增加对 distribution_risk_components 的调试输出
            if "distribution_risk_components_debug" in _temp_debug_values:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派发风险组件 ---"] = ""
                for comp_name, comp_series in _temp_debug_values["distribution_risk_components_debug"].items():
                    val = comp_series.loc[probe_ts] if probe_ts in comp_series.index else np.nan
                    debug_output[f"        {comp_name}: {val:.4f}"] = ""
            for key, series in _temp_debug_values["派发风险"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 前置下跌风险 ---"] = ""
            for key, series in _temp_debug_values["前置下跌风险"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 综合风险惩罚因子 ---"] = ""
            for key, series in _temp_debug_values["综合风险惩罚因子"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终意图合成 ---"] = ""
            for key, series in _temp_debug_values["最终意图合成"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 相对强度和资本属性调节 ---"] = ""
            for key, series in _temp_debug_values["相对强度和资本属性调节"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力拉升意图诊断完成，最终分值: {final_rally_intent.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_rally_intent.astype(np.float32)

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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价格效率关系..."] = ""
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础共识分数 ---"] = ""
            for key, series in _temp_debug_values["基础共识分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 品质因子校准 ---"] = ""
            for key, series in _temp_debug_values["品质因子校准"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价格效率关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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
            decay_score = self._calculate_winner_conviction_decay(df, config)
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

    def _calculate_price_momentum_divergence(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "_calculate_price_momentum_divergence"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价势背离..."] = ""
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
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 3. 获取原始数据 ---
        price_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_close_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        macdh_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_MACDh_13_34_8_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        rsi_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_RSI_13_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        roc_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_ROC_13_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        volume_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_volume_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        volume_burstiness_raw = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name)
        volume_atrophy_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_VOLUME_ATROPHY', 0.0)
        mf_net_flow_slopes_raw = {p: self._get_safe_series(df, f'SLOPE_{p}_main_force_net_flow_calibrated_D', 0.0, method_name=method_name) for p in valid_mtf_periods}
        deception_index_raw = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        distribution_intent_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        covert_accumulation_score = self._get_atomic_score(df, 'PROCESS_META_COVERT_ACCUMULATION', 0.0)
        chip_divergence_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_DIVERGENCE', 0.0)
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        adx_raw = self._get_safe_series(df, 'ADX_14_D', 0.0, method_name=method_name)
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        upward_efficiency_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 0.0)
        price_upward_momentum_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM', 0.0)
        price_downward_momentum_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0)
        momentum_quality_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_MOMENTUM', 0.0)
        constructive_turnover_raw = self._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name)
        volume_structure_skew_raw = self._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name=method_name)
        main_force_conviction_raw = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        chip_health_raw = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name)
        stability_score = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        chip_historical_potential_score = self._get_atomic_score(df, 'SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL', 0.0)
        liquidity_tide_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE', 0.0)
        market_constitution_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION', 0.0)
        market_tension_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0)
        _temp_debug_values["原始信号值"] = {
            "price_slopes_raw": price_slopes_raw,
            "macdh_slopes_raw": macdh_slopes_raw,
            "rsi_slopes_raw": rsi_slopes_raw,
            "roc_slopes_raw": roc_slopes_raw,
            "volume_slopes_raw": volume_slopes_raw,
            "volume_burstiness_index_D": volume_burstiness_raw,
            "SCORE_BEHAVIOR_VOLUME_ATROPHY": volume_atrophy_score,
            "mf_net_flow_slopes_raw": mf_net_flow_slopes_raw,
            "deception_index_D": deception_index_raw,
            "SCORE_BEHAVIOR_DISTRIBUTION_INTENT": distribution_intent_score,
            "PROCESS_META_COVERT_ACCUMULATION": covert_accumulation_score,
            "SCORE_CHIP_AXIOM_DIVERGENCE": chip_divergence_score,
            "VOLATILITY_INSTABILITY_INDEX_21d_D": volatility_instability_raw,
            "ADX_14_D": adx_raw,
            "market_sentiment_score_D": market_sentiment_raw,
            "SCORE_BEHAVIOR_UPWARD_EFFICIENCY": upward_efficiency_score,
            "SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM": price_upward_momentum_score,
            "SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM": price_downward_momentum_score,
            "SCORE_DYN_AXIOM_MOMENTUM": momentum_quality_score,
            "constructive_turnover_ratio_D": constructive_turnover_raw,
            "volume_structure_skew_D": volume_structure_skew_raw,
            "main_force_conviction_index_D": main_force_conviction_raw,
            "chip_health_score_D": chip_health_raw,
            "SCORE_DYN_AXIOM_STABILITY": stability_score,
            "SCORE_CHIP_AXIOM_HISTORICAL_POTENTIAL": chip_historical_potential_score,
            "SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE": liquidity_tide_score,
            "SCORE_FOUNDATION_AXIOM_MARKET_CONSTITUTION": market_constitution_score,
            "SCORE_FOUNDATION_AXIOM_MARKET_TENSION": market_tension_score
        }
        # --- 4. 计算各维度分数 ---
        # 4.1. Fused Price Direction (MTF Slope Fusion)
        fused_price_direction_base = self._get_mtf_slope_score(df, 'close_D', mtf_slope_weights, df_index, method_name, bipolar=True)
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
        _temp_debug_values["融合价格方向"] = {
            "fused_price_direction_base": fused_price_direction_base,
            "price_momentum_quality_score": price_momentum_quality_score,
            "fused_price_direction": fused_price_direction
        }
        # 4.2. Fused Momentum Direction (Multi-Indicator & MTF Fusion)
        fused_macdh_direction = self._get_mtf_slope_score(df, 'MACDh_13_34_8_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        fused_rsi_direction = self._get_mtf_slope_score(df, 'RSI_13_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        fused_roc_direction = self._get_mtf_slope_score(df, 'ROC_13_D', mtf_slope_weights, df_index, method_name, bipolar=True)
        fused_momentum_direction_components = {
            "MACDh_13_34_8_D": fused_macdh_direction,
            "RSI_13_D": fused_rsi_direction,
            "ROC_13_D": fused_roc_direction,
            "momentum_quality": momentum_quality_score
        }
        momentum_components_weights_extended = momentum_components_weights.copy()
        momentum_components_weights_extended["momentum_quality"] = get_param_value(params.get('momentum_components_weights', {}).get("momentum_quality"), 0.2)
        fused_momentum_direction = _robust_geometric_mean(fused_momentum_direction_components, momentum_components_weights_extended, df_index)
        _temp_debug_values["融合动量方向"] = {
            "fused_macdh_direction": fused_macdh_direction,
            "fused_rsi_direction": fused_rsi_direction,
            "fused_roc_direction": fused_roc_direction,
            "fused_momentum_direction": fused_momentum_direction
        }
        # 4.3. Base Divergence Score
        base_divergence_score = (fused_price_direction - fused_momentum_direction).clip(-1, 1)
        _temp_debug_values["基础背离分数"] = {
            "base_divergence_score": base_divergence_score
        }
        # 4.4. Volume Confirmation Score
        fused_volume_slope = self._get_mtf_slope_score(df, 'volume_D', mtf_slope_weights, df_index, method_name, bipolar=True)
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
        _temp_debug_values["量能确认分数"] = {
            "fused_volume_slope": fused_volume_slope,
            "volume_burst_norm": volume_burst_norm,
            "volume_atrophy_norm": volume_atrophy_norm,
            "constructive_turnover_norm": constructive_turnover_norm,
            "volume_structure_skew_inverted_norm": volume_structure_skew_inverted_norm,
            "top_vol_conf": top_vol_conf,
            "bottom_vol_conf": bottom_vol_conf,
            "volume_confirmation_score": volume_confirmation_score
        }
        # 4.5. Main Force/Chip Confirmation Score
        fused_mf_net_flow_slope = self._get_mtf_slope_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_weights, df_index, method_name, bipolar=True)
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
        bottom_mf_conf_components = {
            "mf_net_flow_slope_positive": fused_mf_net_flow_slope.clip(lower=0),
            "deception_index_negative": deception_index_norm.clip(upper=0).abs(),
            "covert_accumulation": covert_accumulation_norm,
            "chip_divergence_negative": chip_divergence_norm.clip(upper=0).abs(),
            "main_force_conviction": main_force_conviction_norm.clip(upper=0).abs(),
            "chip_health": chip_health_norm
        }
        top_mf_conf = _robust_geometric_mean(top_mf_conf_components, current_main_force_confirmation_weights, df_index)
        bottom_mf_conf = _robust_geometric_mean(bottom_mf_conf_components, current_main_force_confirmation_weights, df_index)
        main_force_confirmation_score = pd.Series([
            top_mf_conf.loc[idx] if x > 0 else (-bottom_mf_conf.loc[idx] if x < 0 else 0)
            for idx, x in base_divergence_score.items()
        ], index=df_index, dtype=np.float32)
        _temp_debug_values["主力/筹码确认分数"] = {
            "fused_mf_net_flow_slope": fused_mf_net_flow_slope,
            "deception_index_norm": deception_index_norm,
            "distribution_intent_norm": distribution_intent_norm,
            "covert_accumulation_norm": covert_accumulation_norm,
            "chip_divergence_norm": chip_divergence_norm,
            "main_force_conviction_norm": main_force_conviction_norm,
            "chip_health_norm": chip_health_norm,
            "top_mf_conf": top_mf_conf,
            "bottom_mf_conf": bottom_mf_conf,
            "main_force_confirmation_score": main_force_confirmation_score
        }
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
        _temp_debug_values["背离质量分数"] = {
            "divergence_quality_score": divergence_quality_score
        }
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
        _temp_debug_values["情境调制器"] = {
            "volatility_instability_norm_inverted": volatility_instability_norm_inverted,
            "adx_norm_inverted": adx_norm_inverted,
            "market_sentiment_norm_bipolar": market_sentiment_norm_bipolar,
            "liquidity_tide_calm_norm": liquidity_tide_calm_norm,
            "market_constitution_neutrality_norm": market_constitution_neutrality_norm,
            "context_modulator": context_modulator
        }
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
        _temp_debug_values["最终融合组件"] = {
            "base_divergence_abs": base_divergence_score.abs(),
            "volume_confirmation_abs": volume_confirmation_score.abs(),
            "main_force_confirmation_abs": main_force_confirmation_score.abs(),
            "divergence_quality": divergence_quality_score,
            "context_modulator": context_modulator
        }
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
            _temp_debug_values["动态融合权重调整"] = {
                "modulator_signal_1": modulator_signal_1,
                "modulator_signal_2": modulator_signal_2,
                "adjusted_weights_series": adjusted_weights_series,
                "final_fusion_weights_dict_dynamic": final_fusion_weights_dict
            }
        # Calculate the raw fused score (unipolar)
        raw_fused_score = _robust_geometric_mean(final_components, final_fusion_weights_dict, df_index)
        _temp_debug_values["原始融合分数"] = {
            "raw_fused_score": raw_fused_score
        }
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
        _temp_debug_values["协同/冲突与最终分数"] = {
            "sign_base": sign_base,
            "sign_vol": sign_vol,
            "sign_mf": sign_mf,
            "sign_price_momentum_quality": sign_price_momentum_quality,
            "aligned_count": aligned_count,
            "is_synergistic": is_synergistic,
            "is_conflicting": is_conflicting,
            "synergy_conflict_factor": synergy_conflict_factor,
            "raw_fused_score_modulated": raw_fused_score_modulated,
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合价格方向 ---"] = ""
            for key, series in _temp_debug_values["融合价格方向"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合动量方向 ---"] = ""
            for key, series in _temp_debug_values["融合动量方向"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础背离分数 ---"] = ""
            for key, series in _temp_debug_values["基础背离分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {probe_ts.strftime('%Y-%m-%d')}: --- 量能确认分数 ---"] = ""
            for key, series in _temp_debug_values["量能确认分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力/筹码确认分数 ---"] = ""
            for key, series in _temp_debug_values["主力/筹码确认分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 背离质量分数 ---"] = ""
            for key, series in _temp_debug_values["背离质量分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制器 ---"] = ""
            for key, series in _temp_debug_values["情境调制器"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合组件 ---"] = ""
            for key, series in _temp_debug_values["最终融合组件"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            if "动态融合权重调整" in _temp_debug_values:
                debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态融合权重调整 ---"] = ""
                for key, series in _temp_debug_values["动态融合权重调整"].items():
                    if isinstance(series, pd.Series):
                        val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                        debug_output[f"        {key}: {val:.4f}"] = ""
                    else:
                        debug_output[f"        {key}: {series}"] = "" # For non-Series values like dict
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始融合分数 ---"] = ""
            for key, series in _temp_debug_values["原始融合分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 协同/冲突与最终分数 ---"] = ""
            for key, series in _temp_debug_values["协同/冲突与最终分数"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = "" # For boolean series
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价势背离诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_score.astype(np.float32)

    def _calculate_winner_conviction_decay(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V4.1 · 全息动态审判版】“赢家信念衰减”专属计算引擎
        - 核心重构: 引入更多维度信号，深化对信念衰减、利润压力、派发确认和情境调制的感知。
        - 核心升级: 引入“买盘抵抗瓦解”证据，强化“诡道派发”识别，扩展情境调制器。
        - 核心优化: 引入“动态融合指数”，根据市场波动率和情绪动态调整最终融合的非线性指数。
        - 核心逻辑: 最终衰减分 = (核心衰减分 * (1 + 情境调制器))^动态非线性指数。
        """
        method_name = "_calculate_winner_conviction_decay"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算赢家信念衰减..."] = ""
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
        # --- 修复: 添加 relative_position_weights 的获取 ---
        relative_position_weights = get_param_value(decay_params.get('relative_position_weights'), {"winner_stability_high": 0.6, "profit_taking_flow_low": 0.4})
        # --- 修复: 添加 final_fusion_gm_weights 和 final_exponent 的获取 ---
        final_fusion_gm_weights = get_param_value(decay_params.get('final_fusion_gm_weights'), {
            "conviction_magnitude": 0.3, "pressure_magnitude": 0.25,
            "synergy_factor": 0.2, "deception_filter": 0.15,
            "context_modulator": 0.1
        })
        final_exponent = get_param_value(decay_params.get('final_exponent'), 1.5)
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
            'chip_health_score_D', 'market_impact_cost_D',
            'trend_vitality_index_D' # --- 修复: 添加 trend_vitality_index_D ---
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
            required_df_columns.append(f'SLOPE_{period_str}_trend_vitality_index_D') # 添加MTF斜率
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
            required_df_columns.append(f'ACCEL_{period_str}_trend_vitality_index_D') # 添加MTF加速度
        required_atomic_signals = [
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER',
            'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM'
        ]
        all_required_signals = required_df_columns + required_atomic_signals
        if not self._validate_required_signals(df, all_required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(dtype=np.float32)
        df_index = df.index
        # 原始输入信号
        belief_signal_raw = self._get_safe_series(df, belief_signal_name, 0.0, method_name=method_name)
        pressure_signal_raw = self._get_safe_series(df, pressure_signal_name, 0.0, method_name=method_name)
        upper_shadow_pressure_raw = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', 0.0, method_name=method_name)
        retail_fomo_raw = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name)
        market_sentiment_raw = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_raw = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        bias_13_raw = self._get_safe_series(df, 'BIAS_13_D', 0.0, method_name=method_name)
        bias_21_raw = self._get_safe_series(df, 'BIAS_21_D', 0.0, method_name=method_name)
        rsi_13_raw = self._get_safe_series(df, 'RSI_13_D', 0.0, method_name=method_name)
        bbp_21_raw = self._get_safe_series(df, 'BBP_21_2.0_D', 0.0, method_name=method_name)
        distribution_intent_score = self._get_atomic_score(df, 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 0.0)
        chip_distribution_whisper_score = self._get_atomic_score(df, 'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER', 0.0)
        market_tension_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_MARKET_TENSION', 0.0)
        sentiment_pendulum_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM', 0.0)
        winner_profit_margin_avg_raw = self._get_safe_series(df, 'winner_profit_margin_avg_D', 0.0, method_name=method_name)
        total_winner_rate_raw = self._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name=method_name)
        chip_fatigue_raw = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name)
        active_selling_pressure_raw = self._get_safe_series(df, 'active_selling_pressure_D', 0.0, method_name=method_name)
        rally_sell_distribution_intensity_raw = self._get_safe_series(df, 'rally_sell_distribution_intensity_D', 0.0, method_name=method_name)
        main_force_t0_sell_efficiency_raw = self._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name=method_name)
        main_force_on_peak_sell_flow_raw = self._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name=method_name)
        deception_lure_long_raw = self._get_safe_series(df, 'deception_lure_long_intensity_D', 0.0, method_name=method_name)
        wash_trade_intensity_raw = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        pressure_rejection_strength_raw = self._get_safe_series(df, 'pressure_rejection_strength_D', 0.0, method_name=method_name)
        rally_buy_support_weakness_raw = self._get_safe_series(df, 'rally_buy_support_weakness_D', 0.0, method_name=method_name)
        buy_quote_exhaustion_raw = self._get_safe_series(df, 'buy_quote_exhaustion_rate_D', 0.0, method_name=method_name)
        bid_side_liquidity_raw = self._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name=method_name)
        main_force_slippage_raw = self._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name)
        structural_tension_raw = self._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name)
        volatility_expansion_raw = self._get_safe_series(df, 'volatility_expansion_ratio_D', 0.0, method_name=method_name)
        chip_health_raw = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name)
        market_impact_cost_raw = self._get_safe_series(df, 'market_impact_cost_D', 0.0, method_name=method_name)
        # --- 修复: 获取 trend_vitality_raw ---
        trend_vitality_raw = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "winner_stability_index_D": belief_signal_raw, # 使用已赋值的变量
            "profit_taking_flow_ratio_D": pressure_signal_raw, # 使用已赋值的变量
            "upper_shadow_selling_pressure_D": upper_shadow_pressure_raw,
            "retail_fomo_premium_index_D": retail_fomo_raw,
            "market_sentiment_score_D": market_sentiment_raw,
            "VOLATILITY_INSTABILITY_INDEX_21d_D": volatility_instability_raw,
            "BIAS_13_D": bias_13_raw,
            "BIAS_21_D": bias_21_raw,
            "RSI_13_D": rsi_13_raw,
            "BBP_21_2.0_D": bbp_21_raw,
            "SCORE_BEHAVIOR_DISTRIBUTION_INTENT": distribution_intent_score,
            "SCORE_CHIP_RISK_DISTRIBUTION_WHISPER": chip_distribution_whisper_score,
            "SCORE_FOUNDATION_AXIOM_MARKET_TENSION": market_tension_score,
            "SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM": sentiment_pendulum_score,
            "winner_profit_margin_avg_D": winner_profit_margin_avg_raw,
            "total_winner_rate_D": total_winner_rate_raw,
            "chip_fatigue_index_D": chip_fatigue_raw,
            "active_selling_pressure_D": active_selling_pressure_raw,
            "rally_sell_distribution_intensity_D": rally_sell_distribution_intensity_raw,
            "main_force_t0_sell_efficiency_D": main_force_t0_sell_efficiency_raw,
            "main_force_on_peak_sell_flow_D": main_force_on_peak_sell_flow_raw,
            "deception_lure_long_intensity_D": deception_lure_long_raw,
            "wash_trade_intensity_D": wash_trade_intensity_raw,
            "pressure_rejection_strength_D": pressure_rejection_strength_raw,
            "rally_buy_support_weakness_D": rally_buy_support_weakness_raw,
            "buy_quote_exhaustion_rate_D": buy_quote_exhaustion_raw,
            "bid_side_liquidity_D": bid_side_liquidity_raw,
            "main_force_slippage_index_D": main_force_slippage_raw,
            "structural_tension_index_D": structural_tension_raw,
            "volatility_expansion_ratio_D": volatility_expansion_raw,
            "chip_health_score_D": chip_health_raw,
            "market_impact_cost_D": market_impact_cost_raw,
            "trend_vitality_index_D": trend_vitality_raw # --- 修复: 添加 trend_vitality_raw ---
        }
        # --- 1. 信念强度 (Conviction Strength) ---
        # 使用MTF融合赢家稳定性及其斜率和加速度 (双极性)
        mtf_winner_stability = self._get_mtf_slope_accel_score(df, belief_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 赢家稳定性相对于历史区间的百分位 (越高越好，映射到 [0, 1])
        winner_stability_percentile = belief_signal_raw.rank(pct=True).fillna(0.5) # 使用 belief_signal_raw
        # 综合信念强度：MTF趋势 + 历史相对位置
        conviction_strength_score = (mtf_winner_stability * relative_position_weights.get("winner_stability_high", 0.6) + 
                                     (winner_stability_percentile * 2 - 1) * (1 - relative_position_weights.get("winner_stability_high", 0.6))).clip(-1, 1)
        _temp_debug_values["信念强度"] = {
            "mtf_winner_stability": mtf_winner_stability,
            "winner_stability_percentile": winner_stability_percentile,
            "conviction_strength_score": conviction_strength_score
        }
        # --- 2. 压力韧性 (Pressure Resilience) ---
        # 使用MTF融合利润兑现流量及其斜率和加速度 (双极性，负值代表压力大，韧性差)
        # 利润兑现流量是负向指标，所以其MTF分数越低（负值越大）代表压力越大，韧性越差
        mtf_profit_taking_flow = self._get_mtf_slope_accel_score(df, pressure_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 利润兑现流量相对于历史区间的百分位 (越低越好，映射到 [0, 1])
        profit_taking_flow_percentile = (1 - pressure_signal_raw.rank(pct=True)).fillna(0.5) # 使用 pressure_signal_raw
        # 综合压力韧性：(1 - MTF利润兑现流量) + 历史相对位置
        # 将mtf_profit_taking_flow反向，使其正值代表韧性强
        pressure_resilience_score = ((mtf_profit_taking_flow * -1) * relative_position_weights.get("profit_taking_flow_low", 0.4) + 
                                     (profit_taking_flow_percentile * 2 - 1) * (1 - relative_position_weights.get("profit_taking_flow_low", 0.4))).clip(-1, 1)
        _temp_debug_values["压力韧性"] = {
            "mtf_profit_taking_flow": mtf_profit_taking_flow,
            "profit_taking_flow_percentile": profit_taking_flow_percentile,
            "pressure_resilience_score": pressure_resilience_score
        }
        # --- 3. 共振与背离因子 (Synergy Factor) ---
        # 评估赢家稳定性与利润兑现压力之间的共振或背离
        # 当两者同向（信念增强且压力减弱，或信念减弱且压力增强）时，共振因子高
        # 当两者背离（信念增强但压力也增强，或信念减弱但压力也减弱）时，共振因子低
        # 将信念强度和压力韧性映射到 [0, 1]
        norm_conviction = (conviction_strength_score + 1) / 2
        norm_resilience = (pressure_resilience_score + 1) / 2
        # 协同因子：当两者都高或都低时，协同性高
        synergy_factor = (norm_conviction * norm_resilience + (1 - norm_conviction) * (1 - norm_resilience)).clip(0, 1)
        _temp_debug_values["共振与背离因子"] = {
            "norm_conviction": norm_conviction,
            "norm_resilience": norm_resilience,
            "synergy_factor": synergy_factor
        }
        # --- 4. 诡道过滤 (Deception Filter) ---
        # 欺骗指数和对倒强度越高，对信念的真实性惩罚越大
        mtf_deception_index = self._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_wash_trade_intensity = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        deception_penalty = (mtf_deception_index * 0.6 + mtf_wash_trade_intensity * 0.4).clip(0, 1)
        deception_filter = (1 - deception_penalty).clip(0, 1) # 惩罚因子，0表示完全过滤，1表示无影响
        _temp_debug_values["诡道过滤"] = {
            "mtf_deception_index": mtf_deception_index,
            "mtf_wash_trade_intensity": mtf_wash_trade_intensity,
            "deception_penalty": deception_penalty,
            "deception_filter": deception_filter
        }
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
            debug_info=False
        ) # 将不稳定性转换为稳定性，并归一化到 [0, 1]
        # 探针：volatility_stability_raw 的值
        norm_volatility_stability = self._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self._normalize_series(trend_vitality_raw, df_index, bipolar=False) # 趋势活力越高越好
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability, # 使用修正后的稳定性
            "trend_vitality": norm_trend_vitality
        }
        # 使用几何平均融合情境调制器，确保只有当多个情境同时有利时才高
        context_modulator_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()}, # 确保输入为正
            contextual_modulator_weights, # --- 修复: 使用正确的变量名 ---
            df_index
        )
        # 将情境调制器映射到 [0.5, 1.5] 范围，以实现放大或抑制
        context_modulator = 0.5 + context_modulator_score # 0.5 + [0,1] -> [0.5, 1.5]
        _temp_debug_values["情境调制"] = {
            "norm_market_sentiment": norm_market_sentiment,
            "volatility_stability_raw": volatility_stability_raw,
            "norm_volatility_stability": norm_volatility_stability,
            "norm_trend_vitality": norm_trend_vitality,
            "context_modulator_score": context_modulator_score,
            "context_modulator": context_modulator
        }
        # --- 6. 最终融合 ---
        # 1. 确定整体方向：由信念强度和压力韧性的加权和决定
        # 权重可以从配置中获取，这里使用默认值
        direction_weight_conviction = get_param_value(decay_params.get('direction_weights', {}).get('conviction', 0.6), 0.6)
        direction_weight_pressure = get_param_value(decay_params.get('direction_weights', {}).get('pressure', 0.4), 0.4)
        overall_direction_raw = (conviction_strength_score * direction_weight_conviction + pressure_resilience_score * direction_weight_pressure)
        overall_direction = np.sign(overall_direction_raw)
        overall_direction = overall_direction.replace(0, 1) # 如果和为0，则视为正向，让幅度决定
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
        # 3. 使用 _robust_geometric_mean 融合所有强度/幅度组件
        fused_magnitude = _robust_geometric_mean(
            fusion_components_for_gm,
            final_fusion_gm_weights,
            df_index
        )
        # 4. 结合整体方向和融合后的幅度
        final_score = fused_magnitude * overall_direction
        # 5. 应用非线性指数
        final_score = np.sign(final_score) * (final_score.abs().pow(final_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        _temp_debug_values["最终融合"] = {
            "direction_weight_conviction": direction_weight_conviction,
            "direction_weight_pressure": direction_weight_pressure,
            "overall_direction_raw": overall_direction_raw,
            "overall_direction": overall_direction,
            "conviction_magnitude": conviction_magnitude,
            "pressure_magnitude": pressure_magnitude,
            "fused_magnitude": fused_magnitude,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                if isinstance(value, pd.Series):
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        '{key}': {val:.4f}"] = ""
                elif isinstance(value, dict): # Handle dicts within _temp_debug_values["原始信号值"]
                    debug_output[f"        '{key}':"] = ""
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.Series):
                            val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                            debug_output[f"          {sub_key}: {val:.4f}"] = ""
                        else:
                            debug_output[f"          {sub_key}: {sub_value}"] = ""
                else:
                    debug_output[f"        '{key}': {value}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 信念强度 ---"] = ""
            for key, series in _temp_debug_values["信念强度"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 压力韧性 ---"] = ""
            for key, series in _temp_debug_values["压力韧性"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 共振与背离因子 ---"] = ""
            for key, series in _temp_debug_values["共振与背离因子"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道过滤 ---"] = ""
            for key, series in _temp_debug_values["诡道过滤"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制 ---"] = ""
            for key, series in _temp_debug_values["情境调制"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
            for key, series in _temp_debug_values["最终融合"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                elif isinstance(series, dict): # Handle dicts within _temp_debug_values["最终融合"]
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_value in series.items():
                        if isinstance(sub_value, pd.Series):
                            val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                            debug_output[f"          {sub_key}: {val:.4f}"] = ""
                        else:
                            debug_output[f"          {sub_key}: {sub_value}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 赢家信念衰减诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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

    def _calculate_stealth_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V5.2 · 全息融合与多维趋势感知版】计算“隐蔽吸筹”的专属关系分数。
        - 核心升级: 将“横盘”与“温和推升”场景的证据融合方式，从“几何平均”升级为“全息证据加权融合”，
                      允许核心证据的强势弥补次要证据的不足，以勘破“明修栈道，暗度陈仓”的诡道。
        - 【强化】将价格趋势、筹码集中度趋势、峰值稳固度趋势和成本优势趋势全部升级为多时间维度（MTF）融合信号，
                      增强信号的鲁棒性和趋势感知能力。
        - 【调整】优化 `suppressive_score` 和 `consolidative_score` 的构成，使其更精准地捕捉隐蔽吸筹的特征。
        """
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
        # --- 1. 信念强度 (Conviction Strength) ---
        # 使用MTF融合赢家稳定性及其斜率和加速度 (双极性)
        mtf_winner_stability = self._get_mtf_slope_accel_score(df, belief_signal_name, mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 赢家稳定性相对于历史区间的百分位 (越高越好，映射到 [0, 1])
        winner_stability_percentile = winner_stability_raw.rank(pct=True).fillna(0.5)
        # 综合信念强度：MTF趋势 + 历史相对位置
        conviction_strength_score = (mtf_winner_stability * relative_position_weights.get("winner_stability_high", 0.6) + 
                                     (winner_stability_percentile * 2 - 1) * (1 - relative_position_weights.get("winner_stability_high", 0.6))).clip(-1, 1)
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
        # --- 3. 共振与背离因子 (Synergy Factor) ---
        # 评估赢家稳定性与利润兑现压力之间的共振或背离
        # 当两者同向（信念增强且压力减弱，或信念减弱且压力增强）时，共振因子高
        # 当两者背离（信念增强但压力也增强，或信念减弱但压力也减弱）时，共振因子低
        # 将信念强度和压力韧性映射到 [0, 1]
        norm_conviction = (conviction_strength_score + 1) / 2
        norm_resilience = (pressure_resilience_score + 1) / 2
        # 协同因子：当两者都高或都低时，协同性高
        synergy_factor = (norm_conviction * norm_resilience + (1 - norm_conviction) * (1 - norm_resilience)).clip(0, 1)
        # --- 4. 诡道过滤 (Deception Filter) ---
        # 欺骗指数和对倒强度越高，对信念的真实性惩罚越大
        mtf_deception_index = self._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_wash_trade_intensity = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        deception_penalty = (mtf_deception_index * 0.6 + mtf_wash_trade_intensity * 0.4).clip(0, 1)
        deception_filter = (1 - deception_penalty).clip(0, 1) # 惩罚因子，0表示完全过滤，1表示无影响
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
            debug_info=False
        ) # 将不稳定性转换为稳定性，并归一化到 [0, 1]
        # 探针：volatility_stability_raw 的值
        norm_volatility_stability = self._normalize_series(volatility_stability_raw, df_index, bipolar=False, ascending=True)
        norm_trend_vitality = self._normalize_series(trend_vitality_raw, df_index, bipolar=False) # 趋势活力越高越好
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_stability": norm_volatility_stability, # 使用修正后的稳定性
            "trend_vitality": norm_trend_vitality
        }
        # 调试探针：context_modulator_components 的输入
        # 使用几何平均融合情境调制器，确保只有当多个情境同时有利时才高
        context_modulator_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()}, # 确保输入为正
            context_modulator_weights,
            df_index
        )
        # 将情境调制器映射到 [0.5, 1.5] 范围，以实现放大或抑制
        context_modulator = 0.5 + context_modulator_score # 0.5 + [0,1] -> [0.5, 1.5]
        # --- 6. 最终融合 ---
        # 1. 确定整体方向：由信念强度和压力韧性的加权和决定
        # 权重可以从配置中获取，这里使用默认值
        direction_weight_conviction = get_param_value(params.get('direction_weights', {}).get('conviction', 0.6), 0.6)
        direction_weight_pressure = get_param_value(params.get('direction_weights', {}).get('pressure', 0.4), 0.4)
        overall_direction_raw = (conviction_strength_score * direction_weight_conviction + pressure_resilience_score * direction_weight_pressure)
        overall_direction = np.sign(overall_direction_raw)
        overall_direction = overall_direction.replace(0, 1) # 如果和为0，则视为正向，让幅度决定
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
        # 3. 使用 _robust_geometric_mean 融合所有强度/幅度组件
        fused_magnitude = _robust_geometric_mean(
            fusion_components_for_gm,
            final_fusion_gm_weights,
            df_index
        )
        # 4. 结合整体方向和融合后的幅度
        final_score = fused_magnitude * overall_direction
        # 5. 应用非线性指数
        final_score = np.sign(final_score) * (final_score.abs().pow(final_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        return final_score.astype(np.float32)

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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算套牢盘投降..."] = ""
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 战场上下文 ---"] = ""
            for key, series in _temp_debug_values["战场上下文"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val}"] = "" # Boolean series
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 恐慌分 ---"] = ""
            for key, series in _temp_debug_values["恐慌分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 吸收分 ---"] = ""
            for key, series in _temp_debug_values["吸收分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终审判 ---"] = ""
            for key, series in _temp_debug_values["最终审判"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 套牢盘投降诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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

    def _calculate_split_order_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V3.2 · 质效校准版】计算“拆单吸筹强度”的专属信号。
        - 核心升级: 引入“效率基准线”(efficiency_baseline)概念。在计算“质效调节指数”前，
                      先对“全息验证综合分”进行校准。这使得任何低于基准线的战果（即使为正）
                      都会被视为负向贡献，从而受到惩罚性抑制，为模型注入了赏罚分明的“主帅”逻辑。
        """
        method_name = "_calculate_split_order_accumulation"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算拆单吸筹强度..."] = ""
        required_signals = [
            'hidden_accumulation_intensity_D', 'SLOPE_5_close_D', 'deception_index_D',
            'upward_impulse_purity_D', 'PROCESS_META_POWER_TRANSFER',
            'SCORE_CHIP_STRATEGIC_POSTURE', 'SCORE_DYN_AXIOM_STABILITY'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        efficiency_baseline = config.get('efficiency_baseline', 0.15)
        raw_intensity = self._get_safe_series(df, 'hidden_accumulation_intensity_D', 0.0, method_name=method_name)
        price_trend_raw = self._get_safe_series(df, 'SLOPE_5_close_D', 0.0, method_name=method_name)
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        upward_purity = self._normalize_series(self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name), df_index, bipolar=False)
        flow_outcome = self._get_atomic_score(df, 'PROCESS_META_POWER_TRANSFER', 0.0)
        structure_outcome = self._get_atomic_score(df, 'SCORE_CHIP_STRATEGIC_POSTURE', 0.0)
        potential_outcome = self._get_atomic_score(df, 'SCORE_DYN_AXIOM_STABILITY', 0.0)
        _temp_debug_values["原始信号值"] = {
            "hidden_accumulation_intensity_D": raw_intensity,
            "SLOPE_5_close_D": price_trend_raw,
            "deception_index_D": deception_index,
            "upward_impulse_purity_D": upward_purity,
            "PROCESS_META_POWER_TRANSFER": flow_outcome,
            "SCORE_CHIP_STRATEGIC_POSTURE": structure_outcome,
            "SCORE_DYN_AXIOM_STABILITY": potential_outcome
        }
        normalized_score = (raw_intensity / 100).clip(0, 1)
        price_trend_norm = self._normalize_series(price_trend_raw, df_index, bipolar=True)
        price_suppression_factor = (1 - price_trend_norm.clip(lower=0) * (1 - upward_purity)).clip(0, 1)
        deception_norm = self._normalize_series(deception_index, df_index, bipolar=True)
        strategic_context_factor = (potential_outcome * 0.5 + deception_norm.clip(lower=0) * 0.5).clip(0, 1)
        preliminary_score = (normalized_score * price_suppression_factor * strategic_context_factor).pow(1/3).fillna(0.0)
        tactical_momentum_score = self._normalize_series(preliminary_score.diff(1).fillna(0), df_index, bipolar=False)
        dynamic_preliminary_score = (preliminary_score * 0.7 + tactical_momentum_score * 0.3).clip(0, 1)
        _temp_debug_values["初步计算"] = {
            "normalized_score": normalized_score,
            "price_trend_norm": price_trend_norm,
            "price_suppression_factor": price_suppression_factor,
            "deception_norm": deception_norm,
            "strategic_context_factor": strategic_context_factor,
            "preliminary_score": preliminary_score,
            "tactical_momentum_score": tactical_momentum_score,
            "dynamic_preliminary_score": dynamic_preliminary_score
        }
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
        _temp_debug_values["全息验证"] = {
            "stability_score": stability_score,
            "weight_flow": weight_flow,
            "weight_structure": weight_structure,
            "total_weight": total_weight,
            "w_f": w_f,
            "w_s": w_s,
            "w_p": w_p,
            "holographic_state_score": holographic_state_score,
            "flow_trend": flow_trend,
            "structure_trend": structure_trend,
            "potential_trend": potential_trend,
            "holographic_trend_score": holographic_trend_score,
            "holographic_validation_score": holographic_validation_score
        }
        calibrated_holographic_score = holographic_validation_score - efficiency_baseline
        quality_efficiency_modulator = (1 - calibrated_holographic_score).clip(0.1, 2.0)
        final_score = dynamic_preliminary_score.pow(quality_efficiency_modulator).clip(0, 1)
        _temp_debug_values["最终分数"] = {
            "calibrated_holographic_score": calibrated_holographic_score,
            "quality_efficiency_modulator": quality_efficiency_modulator,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 初步计算 ---"] = ""
            for key, series in _temp_debug_values["初步计算"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 全息验证 ---"] = ""
            for key, series in _temp_debug_values["全息验证"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终分数 ---"] = ""
            for key, series in _temp_debug_values["最终分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 拆单吸筹强度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
        return final_score.astype(np.float32)

    def _calculate_price_volume_dynamics(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V17.0 · 信号强度自适应放大与情境敏感度增强】计算价量动态的专属分数。
        - 核心升级: 优化非线性指数变换，使其在有利情境下放大信号，并增强情境对动态权重的影响力，确保最终分数更准确反映市场行为。
        - 核心重构: 优化四象限的数学模型，特别是Q2和Q4，使其更符合价量分析的业务逻辑。
        - 引入多维共振因子和动态权重，增强信号的鲁棒性和情境感知能力。
        - 全面探针调试，输出所有关键计算节点。
        - 新增原始数据：is_consolidating_D, dynamic_consolidation_duration_D, breakout_readiness_score_D,
                        trend_acceleration_score_D, trend_conviction_score_D, covert_accumulation_signal_D,
                        covert_distribution_signal_D, holistic_cmf_D, reversal_power_index_D,
                        reversal_recovery_rate_D, volatility_asymmetry_index_D, mean_reversion_frequency_D。
        - 优化判定思路：引入动态象限边界，强化多层级共振因子，细化象限内智能逻辑，动态调整非线性指数。
        - 修复：MTF信号名称生成逻辑，确保正确处理包含'_D'子串的原始信号名称。
        - 修复：'norm_trend_vitality'等情境因子在被使用前未赋值的错误。
        """
        method_name = "_calculate_price_volume_dynamics"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价量动态..."] = ""
        df_index = df.index
        params = config.get('price_volume_dynamics_params', {})
        quadrant_weights = get_param_value(params.get('quadrant_weights'), {"Q1_healthy_rally": 0.3, "Q2_bearish_divergence": 0.2, "Q3_panic_distribution": 0.2, "Q4_selling_exhaustion": 0.3})
        multi_level_resonance_weights = get_param_value(params.get('multi_level_resonance_weights'), {"price_volume_resonance": 0.25, "main_chip_resonance": 0.25, "sentiment_liquidity_resonance": 0.2, "micro_structure_resonance": 0.15, "quality_efficiency_resonance": 0.15})
        price_volume_resonance_components = get_param_value(params.get('price_volume_resonance_components'), {"lower_shadow_absorption": 0.2, "active_buying_support": 0.2, "volume_burstiness": 0.15, "VPA_EFFICIENCY": 0.15, "volume_profile_entropy_inverted": 0.1, "volume_ratio_positive": 0.1, "upward_impulse_strength": 0.1})
        main_chip_resonance_components = get_param_value(params.get('main_chip_resonance_components'), {"power_transfer_positive": 0.2, "main_force_conviction_positive": 0.2, "main_force_flow_directionality_positive": 0.15, "chip_strategic_posture": 0.15, "chip_fault_blockage_ratio_inverted": 0.1, "main_force_cost_advantage_positive": 0.1, "SMART_MONEY_HM_COORDINATED_ATTACK": 0.1})
        sentiment_liquidity_resonance_components = get_param_value(params.get('sentiment_liquidity_resonance_components'), {"market_sentiment_positive": 0.25, "retail_panic_surrender_inverted": 0.15, "bid_side_liquidity": 0.15, "liquidity_slope_positive": 0.15, "order_flow_imbalance_positive": 0.15, "loser_pain_index_inverted": 0.15})
        micro_structure_resonance_components = get_param_value(params.get('micro_structure_resonance_components'), {"order_book_imbalance_positive": 0.2, "micro_price_impact_asymmetry_positive": 0.2, "intraday_energy_density": 0.15, "vpin_score_inverted": 0.15, "micro_impact_elasticity_positive": 0.1, "order_book_clearing_rate": 0.1, "closing_acceptance_type_positive": 0.1})
        quality_efficiency_resonance_components = get_param_value(params.get('quality_efficiency_resonance_components'), {"upward_impulse_purity": 0.25, "flow_credibility_index": 0.25, "profit_realization_quality": 0.2, "active_volume_price_efficiency": 0.15, "constructive_turnover_ratio": 0.15})
        Q1_reward_weights = get_param_value(params.get('Q1_reward_weights'), {"VPA_BUY_EFFICIENCY": 0.2, "main_force_execution_alpha": 0.2, "order_flow_imbalance_positive": 0.15, "main_force_vwap_up_guidance_positive": 0.15, "upward_impulse_strength": 0.15, "flow_credibility_index": 0.15})
        Q1_penalty_weights = get_param_value(params.get('Q1_penalty_weights'), {"wash_trade": 0.3, "deception_index": 0.2, "main_force_t0_sell_efficiency": 0.2, "ask_side_liquidity_high": 0.15, "profit_realization_quality_low": 0.15})
        Q2_divergence_penalty_weights = get_param_value(params.get('Q2_divergence_penalty_weights'), {"retail_fomo": 0.2, "wash_trade": 0.2, "deception_index": 0.15, "vpin_score_high": 0.15, "winner_loser_momentum_negative": 0.15, "price_thrust_divergence_negative": 0.15})
        Q3_reward_weights = get_param_value(params.get('Q3_reward_weights'), {"lower_shadow_absorption": 0.25, "active_buying_support": 0.25, "main_force_flow_directionality_positive": 0.15, "main_force_t0_buy_efficiency": 0.15, "capitulation_absorption_index": 0.2})
        Q3_penalty_weights = get_param_value(params.get('Q3_penalty_weights'), {"loser_loss_margin_avg_expanding": 0.25, "panic_selling_cascade": 0.2, "chip_fault_blockage_ratio": 0.2, "downward_impulse_strength": 0.2, "structural_tension_index": 0.15})
        Q4_reward_weights = get_param_value(params.get('Q4_reward_weights'), {"volume_atrophy": 0.2, "volume_profile_entropy_inverted": 0.2, "FRACTAL_DIMENSION_calm": 0.15, "bid_side_liquidity": 0.15, "vpin_score_low": 0.1, "loser_pain_index_high": 0.1, "equilibrium_compression_index": 0.1})
        Q4_penalty_weights = get_param_value(params.get('Q4_penalty_weights'), {"price_reversion_velocity_negative": 0.3, "structural_entropy_change_positive": 0.25, "main_force_vwap_down_guidance_positive": 0.25, "chip_fatigue_index_low": 0.2})
        dynamic_context_modulator_weights = get_param_value(params.get('dynamic_context_modulator_weights'), {"market_sentiment": 0.2, "volatility_inverse": 0.2, "trend_vitality": 0.2, "liquidity_authenticity_score": 0.2, "market_regime_strength": 0.2})
        dynamic_exponent_modulator_weights = get_param_value(params.get('dynamic_exponent_modulator_weights'), {"volatility_inverse": 0.2, "trend_vitality": 0.2, "market_sentiment": 0.15, "liquidity_slope_positive": 0.15, "microstructure_efficiency_index": 0.15, "reversal_potential_score": 0.15})
        historical_context_weights = get_param_value(params.get('historical_context_weights'), {"quadrant_persistence_Q1_Q4": 0.3, "quadrant_persistence_Q2_Q3": 0.2, "phase_transition_Q4_to_Q1": 0.2, "cumulative_flow_balance": 0.15, "market_regime_strength": 0.15})
        context_impact_modulators = get_param_value(params.get('context_impact_modulators'), {
            "deception_impact_reduction_factor": 0.5,
            "trend_reward_enhancement_factor": 0.2,
            "divergence_penalty_enhancement_factor": 0.3,
            "panic_impact_reduction_factor": 0.4,
            "absorption_reward_enhancement_factor": 0.3,
            "blockage_penalty_enhancement_factor": 0.3,
            "exhaustion_reward_enhancement_factor": 0.3,
            "false_bottom_penalty_reduction_factor": 0.4
        })
        final_exponent_base = get_param_value(params.get('final_exponent_base'), 1.0) # 调整基准指数为1.0
        exponent_context_sensitivity = get_param_value(params.get('exponent_context_sensitivity'), 0.8) # 新增参数
        dynamic_weight_sensitivity = get_param_value(params.get('dynamic_weight_sensitivity'), 0.3) # 新增参数
        dynamic_threshold_sensitivity = get_param_value(params.get('dynamic_threshold_sensitivity'), 0.05)
        mtf_slope_accel_weights = get_param_value(params.get('mtf_slope_accel_weights'), {"slope_periods": {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1}, "accel_periods": {"5": 0.6, "13": 0.4}})
        historical_window_short = get_param_value(params.get('historical_window_short'), 5)
        historical_window_long = get_param_value(params.get('historical_window_long'), 21)
        # --- 1. 原始数据获取 (仅数据层信号) ---
        # 价格和成交量
        close_price = self._get_safe_series(df, 'close_D', method_name=method_name)
        volume = self._get_safe_series(df, 'volume_D', method_name=method_name)
        open_price = self._get_safe_series(df, 'open_D', method_name=method_name)
        high_price = self._get_safe_series(df, 'high_D', method_name=method_name)
        low_price = self._get_safe_series(df, 'low_D', method_name=method_name)
        # 主力资金与行为
        main_force_conviction = self._get_safe_series(df, 'main_force_conviction_index_D', 0.0, method_name=method_name)
        wash_trade_intensity = self._get_safe_series(df, 'wash_trade_intensity_D', 0.0, method_name=method_name)
        retail_panic_surrender = self._get_safe_series(df, 'retail_panic_surrender_index_D', 0.0, method_name=method_name)
        upward_impulse_purity = self._get_safe_series(df, 'upward_impulse_purity_D', 0.0, method_name=method_name)
        active_buying_support = self._get_safe_series(df, 'active_buying_support_D', 0.0, method_name=method_name)
        deception_index = self._get_safe_series(df, 'deception_index_D', 0.0, method_name=method_name)
        retail_fomo_premium_index = self._get_safe_series(df, 'retail_fomo_premium_index_D', 0.0, method_name=method_name)
        # 权力转移的原始组件
        net_sh_amount = self._get_safe_series(df, 'net_sh_amount_calibrated_D', 0.0, method_name=method_name)
        net_md_amount = self._get_safe_series(df, 'net_md_amount_calibrated_D', 0.0, method_name=method_name)
        net_lg_amount = self._get_safe_series(df, 'net_lg_amount_calibrated_D', 0.0, method_name=method_name)
        net_elg_amount = self._get_safe_series(df, 'net_xl_amount_calibrated_D', 0.0, method_name=method_name)
        # 筹码与健康度
        winner_concentration_90pct = self._get_safe_series(df, 'winner_concentration_90pct_D', 0.0, method_name=method_name)
        loser_concentration_90pct = self._get_safe_series(df, 'loser_concentration_90pct_D', 0.0, method_name=method_name)
        chip_health_score = self._get_safe_series(df, 'chip_health_score_D', 0.0, method_name=method_name)
        mf_vpoc_premium = self._get_safe_series(df, 'mf_vpoc_premium_D', 0.0, method_name=method_name)
        # 市场情境
        market_sentiment_score = self._get_safe_series(df, 'market_sentiment_score_D', 0.0, method_name=method_name)
        volatility_instability_index = self._get_safe_series(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name=method_name)
        trend_vitality_index = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name=method_name)
        # V12.0 新增原始数据
        volume_burstiness_index = self._get_safe_series(df, 'volume_burstiness_index_D', 0.0, method_name=method_name)
        main_force_flow_directionality = self._get_safe_series(df, 'main_force_flow_directionality_D', 0.0, method_name=method_name)
        order_book_imbalance = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name)
        micro_price_impact_asymmetry = self._get_safe_series(df, 'micro_price_impact_asymmetry_D', 0.0, method_name=method_name)
        bid_side_liquidity = self._get_safe_series(df, 'bid_side_liquidity_D', 0.0, method_name=method_name)
        ask_side_liquidity = self._get_safe_series(df, 'ask_side_liquidity_D', 0.0, method_name=method_name)
        vpin_score = self._get_safe_series(df, 'vpin_score_D', 0.0, method_name=method_name)
        loser_loss_margin_avg = self._get_safe_series(df, 'loser_loss_margin_avg_D', 0.0, method_name=method_name)
        total_winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 0.0, method_name=method_name)
        total_loser_rate = self._get_safe_series(df, 'total_loser_rate_D', 0.0, method_name=method_name)
        panic_selling_cascade = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name=method_name)
        intraday_energy_density = self._get_safe_series(df, 'intraday_energy_density_D', 0.0, method_name=method_name)
        price_reversion_velocity = self._get_safe_series(df, 'price_reversion_velocity_D', 0.0, method_name=method_name)
        # V13.0 新增原始数据
        volume_profile_entropy = self._get_safe_series(df, 'volume_profile_entropy_D', 0.0, method_name=method_name)
        volume_structure_skew = self._get_safe_series(df, 'volume_structure_skew_D', 0.0, method_name=method_name)
        vpa_efficiency = self._get_safe_series(df, 'VPA_EFFICIENCY_D', 0.0, method_name=method_name)
        vpa_buy_efficiency = self._get_safe_series(df, 'VPA_BUY_EFFICIENCY_D', 0.0, method_name=method_name)
        vpa_sell_efficiency = self._get_safe_series(df, 'VPA_SELL_EFFICIENCY_D', 0.0, method_name=method_name)
        turnover_rate_f = self._get_safe_series(df, 'turnover_rate_f_D', 0.0, method_name=method_name)
        main_force_flow_gini = self._get_safe_series(df, 'main_force_flow_gini_D', 0.0, method_name=method_name)
        main_force_slippage_index = self._get_safe_series(df, 'main_force_slippage_index_D', 0.0, method_name=method_name)
        main_force_execution_alpha = self._get_safe_series(df, 'main_force_execution_alpha_D', 0.0, method_name=method_name)
        main_force_t0_buy_efficiency = self._get_safe_series(df, 'main_force_t0_buy_efficiency_D', 0.0, method_name=method_name)
        main_force_t0_sell_efficiency = self._get_safe_series(df, 'main_force_t0_sell_efficiency_D', 0.0, method_name=method_name)
        main_force_vwap_up_guidance = self._get_safe_series(df, 'main_force_vwap_up_guidance_D', 0.0, method_name=method_name)
        main_force_vwap_down_guidance = self._get_safe_series(df, 'main_force_vwap_down_guidance_D', 0.0, method_name=method_name)
        order_flow_imbalance_score = self._get_safe_series(df, 'order_flow_imbalance_score_D', 0.0, method_name=method_name)
        liquidity_slope = self._get_safe_series(df, 'liquidity_slope_D', 0.0, method_name=method_name)
        order_book_clearing_rate = self._get_safe_series(df, 'order_book_clearing_rate_D', 0.0, method_name=method_name)
        chip_fault_blockage_ratio = self._get_safe_series(df, 'chip_fault_blockage_ratio_D', 0.0, method_name=method_name)
        winner_loser_momentum = self._get_safe_series(df, 'winner_loser_momentum_D', 0.0, method_name=method_name)
        fractal_dimension = self._get_safe_series(df, 'FRACTAL_DIMENSION_89d_D', 0.0, method_name=method_name)
        sample_entropy = self._get_safe_series(df, 'SAMPLE_ENTROPY_13d_D', 0.0, method_name=method_name)
        micro_impact_elasticity = self._get_safe_series(df, 'micro_impact_elasticity_D', 0.0, method_name=method_name)
        structural_entropy_change = self._get_safe_series(df, 'structural_entropy_change_D', 0.0, method_name=method_name) # 用于Q4惩罚
        # V14.0 新增原始数据
        volume_ratio = self._get_safe_series(df, 'volume_ratio_D', 0.0, method_name=method_name)
        active_volume_price_efficiency = self._get_safe_series(df, 'active_volume_price_efficiency_D', 0.0, method_name=method_name)
        constructive_turnover_ratio = self._get_safe_series(df, 'constructive_turnover_ratio_D', 0.0, method_name=method_name)
        main_force_cost_advantage = self._get_safe_series(df, 'main_force_cost_advantage_D', 0.0, method_name=method_name)
        main_force_activity_ratio = self._get_safe_series(df, 'main_force_activity_ratio_D', 0.0, method_name=method_name)
        main_force_posture_index = self._get_safe_series(df, 'main_force_posture_index_D', 0.0, method_name=method_name)
        main_force_on_peak_buy_flow = self._get_safe_series(df, 'main_force_on_peak_buy_flow_D', 0.0, method_name=method_name)
        main_force_on_peak_sell_flow = self._get_safe_series(df, 'main_force_on_peak_sell_flow_D', 0.0, method_name=method_name)
        smart_money_coordinated_attack = self._get_safe_series(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', 0.0, method_name=method_name)
        order_book_liquidity_supply = self._get_safe_series(df, 'order_book_liquidity_supply_D', 0.0, method_name=method_name)
        closing_acceptance_type = self._get_safe_series(df, 'closing_acceptance_type_D', 0.0, method_name=method_name)
        auction_showdown_score = self._get_safe_series(df, 'auction_showdown_score_D', 0.0, method_name=method_name)
        chip_fatigue_index = self._get_safe_series(df, 'chip_fatigue_index_D', 0.0, method_name=method_name)
        loser_pain_index = self._get_safe_series(df, 'loser_pain_index_D', 0.0, method_name=method_name)
        profit_realization_quality = self._get_safe_series(df, 'profit_realization_quality_D', 0.0, method_name=method_name)
        structural_tension_index = self._get_safe_series(df, 'structural_tension_index_D', 0.0, method_name=method_name)
        upward_impulse_strength = self._get_safe_series(df, 'upward_impulse_strength_D', 0.0, method_name=method_name)
        price_thrust_divergence = self._get_safe_series(df, 'price_thrust_divergence_D', 0.0, method_name=method_name)
        equilibrium_compression_index = self._get_safe_series(df, 'equilibrium_compression_index_D', 0.0, method_name=method_name)
        microstructure_efficiency_index = self._get_safe_series(df, 'microstructure_efficiency_index_D', 0.0, method_name=method_name)
        liquidity_authenticity_score = self._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name)
        flow_credibility_index = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name)
        # V15.0 新增原始数据
        is_consolidating = self._get_safe_series(df, 'is_consolidating_D', 0.0, method_name=method_name)
        dynamic_consolidation_duration = self._get_safe_series(df, 'dynamic_consolidation_duration_D', 0.0, method_name=method_name)
        breakout_readiness_score = self._get_safe_series(df, 'breakout_readiness_score_D', 0.0, method_name=method_name)
        trend_acceleration_score = self._get_safe_series(df, 'trend_acceleration_score_D', 0.0, method_name=method_name)
        trend_conviction_score = self._get_safe_series(df, 'trend_conviction_score_D', 0.0, method_name=method_name)
        covert_accumulation_signal = self._get_safe_series(df, 'covert_accumulation_signal_D', 0.0, method_name=method_name)
        covert_distribution_signal = self._get_safe_series(df, 'covert_distribution_signal_D', 0.0, method_name=method_name)
        holistic_cmf = self._get_safe_series(df, 'holistic_cmf_D', 0.0, method_name=method_name)
        main_force_net_flow_calibrated = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name=method_name)
        reversal_power_index = self._get_safe_series(df, 'reversal_power_index_D', 0.0, method_name=method_name)
        reversal_recovery_rate = self._get_safe_series(df, 'reversal_recovery_rate_D', 0.0, method_name=method_name)
        volatility_asymmetry_index = self._get_safe_series(df, 'volatility_asymmetry_index_D', 0.0, method_name=method_name)
        mean_reversion_frequency = self._get_safe_series(df, 'mean_reversion_frequency_D', 0.0, method_name=method_name)
        # 确保所有需要的信号都在df中
        required_signals = [
            'close_D', 'volume_D', 'open_D', 'high_D', 'low_D',
            'main_force_conviction_index_D', 'wash_trade_intensity_D', 'retail_panic_surrender_index_D',
            'upward_impulse_purity_D', 'active_buying_support_D', 'deception_index_D', 'retail_fomo_premium_index_D',
            'net_sh_amount_calibrated_D', 'net_md_amount_calibrated_D', 'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D',
            'winner_concentration_90pct_D', 'loser_concentration_90pct_D', 'chip_health_score_D', 'mf_vpoc_premium_D',
            'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'trend_vitality_index_D',
            'volume_burstiness_index_D', 'main_force_flow_directionality_D', 'order_book_imbalance_D',
            'micro_price_impact_asymmetry_D', 'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'vpin_score_D', 'loser_loss_margin_avg_D', 'total_winner_rate_D', 'total_loser_rate_D',
            'panic_selling_cascade_D', 'intraday_energy_density_D', 'price_reversion_velocity_D',
            'VOL_MA_21_D', # 用于量能萎缩代理
            'volume_profile_entropy_D', 'volume_structure_skew_D', 'VPA_EFFICIENCY_D', 'VPA_BUY_EFFICIENCY_D', 'VPA_SELL_EFFICIENCY_D',
            'turnover_rate_f_D', 'main_force_flow_gini_D', 'main_force_slippage_index_D', 'main_force_execution_alpha_D',
            'main_force_t0_buy_efficiency_D', 'main_force_t0_sell_efficiency_D', 'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D',
            'order_flow_imbalance_score_D', 'liquidity_slope_D', 'order_book_clearing_rate_D', 'chip_fault_blockage_ratio_D',
            'winner_loser_momentum_D', 'FRACTAL_DIMENSION_89d_D', 'SAMPLE_ENTROPY_13d_D', 'micro_impact_elasticity_D',
            'structural_entropy_change_D',
            'volume_ratio_D', 'active_volume_price_efficiency_D', 'constructive_turnover_ratio_D',
            'main_force_cost_advantage_D', 'main_force_activity_ratio_D', 'main_force_posture_index_D',
            'main_force_on_peak_buy_flow_D', 'main_force_on_peak_sell_flow_D', 'SMART_MONEY_HM_COORDINATED_ATTACK_D',
            'order_book_liquidity_supply_D', 'closing_acceptance_type_D', 'auction_showdown_score_D',
            'chip_fatigue_index_D', 'loser_pain_index_D', 'profit_realization_quality_D',
            'structural_tension_index_D', 'upward_impulse_strength_D', 'price_thrust_divergence_D',
            'equilibrium_compression_index_D', 'microstructure_efficiency_index_D', 'liquidity_authenticity_score_D',
            'flow_credibility_index_D',
            'is_consolidating_D', 'dynamic_consolidation_duration_D', 'breakout_readiness_score_D',
            'trend_acceleration_score_D', 'trend_conviction_score_D', 'covert_accumulation_signal_D',
            'covert_distribution_signal_D', 'holistic_cmf_D', 'main_force_net_flow_calibrated_D',
            'reversal_power_index_D', 'reversal_recovery_rate_D', 'volatility_asymmetry_index_D',
            'mean_reversion_frequency_D'
        ]
        # 动态添加MTF斜率和加速度信号到required_signals
        base_signals_for_mtf_raw = []
        for s in required_signals:
            # 排除已经包含SLOPE_或ACCEL_的信号，并且确保是日线信号
            if not s.startswith(('SLOPE_', 'ACCEL_')) and s.endswith('_D'):
                # 修正：提取不带_D后缀的原始信号名，使用rsplit确保只移除最后一个_D
                base_signals_for_mtf_raw.append(s.rsplit('_', 1)[0]) # e.g., 'FRACTAL_DIMENSION_89d_D' -> 'FRACTAL_DIMENSION_89d'
        for base_sig_name in base_signals_for_mtf_raw:
            for period_str in mtf_slope_accel_weights.get('slope_periods', {}).keys():
                # 构造MTF斜率信号名：SLOPE_PERIOD_ORIGINAL_NAME_D
                required_signals.append(f'SLOPE_{period_str}_{base_sig_name}_D')
            for period_str in mtf_slope_accel_weights.get('accel_periods', {}).keys():
                # 构造MTF加速度信号名：ACCEL_PERIOD_ORIGINAL_NAME_D
                required_signals.append(f'ACCEL_{period_str}_{base_sig_name}_D')
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        _temp_debug_values["原始信号值"] = {
            "close_D": close_price, "volume_D": volume, "open_D": open_price, "high_D": high_price, "low_D": low_price,
            "main_force_conviction_index_D": main_force_conviction, "wash_trade_intensity_D": wash_trade_intensity,
            "retail_panic_surrender_index_D": retail_panic_surrender, "upward_impulse_purity_D": upward_impulse_purity,
            "active_buying_support_D": active_buying_support, "deception_index_D": deception_index,
            "retail_fomo_premium_index_D": retail_fomo_premium_index,
            "net_sh_amount_calibrated_D": net_sh_amount, "net_md_amount_calibrated_D": net_md_amount,
            "net_lg_amount_calibrated_D": net_lg_amount, "net_elg_amount_calibrated_D": net_elg_amount,
            "winner_concentration_90pct_D": winner_concentration_90pct, "loser_concentration_90pct_D": loser_concentration_90pct,
            "chip_health_score_D": chip_health_score, "mf_vpoc_premium_D": mf_vpoc_premium,
            "market_sentiment_score_D": market_sentiment_score, "VOLATILITY_INSTABILITY_INDEX_21d_D": volatility_instability_index,
            "trend_vitality_index_D": trend_vitality_index,
            "volume_burstiness_index_D": volume_burstiness_index, "main_force_flow_directionality_D": main_force_flow_directionality,
            "order_book_imbalance_D": order_book_imbalance, "micro_price_impact_asymmetry_D": micro_price_impact_asymmetry,
            "bid_side_liquidity_D": bid_side_liquidity, "ask_side_liquidity_D": ask_side_liquidity,
            "vpin_score_D": vpin_score, "loser_loss_margin_avg_D": loser_loss_margin_avg,
            "total_winner_rate_D": total_winner_rate, "total_loser_rate_D": total_loser_rate,
            "panic_selling_cascade_D": panic_selling_cascade, "intraday_energy_density_D": intraday_energy_density,
            "price_reversion_velocity_D": price_reversion_velocity,
            "volume_profile_entropy_D": volume_profile_entropy, "volume_structure_skew_D": volume_structure_skew,
            "VPA_EFFICIENCY_D": vpa_efficiency, "VPA_BUY_EFFICIENCY_D": vpa_buy_efficiency, "VPA_SELL_EFFICIENCY_D": vpa_sell_efficiency,
            "turnover_rate_f_D": turnover_rate_f, "main_force_flow_gini_D": main_force_flow_gini,
            "main_force_slippage_index_D": main_force_slippage_index, "main_force_execution_alpha_D": main_force_execution_alpha,
            "main_force_t0_buy_efficiency_D": main_force_t0_buy_efficiency, "main_force_t0_sell_efficiency_D": main_force_t0_sell_efficiency,
            "main_force_vwap_up_guidance_D": main_force_vwap_up_guidance, "main_force_vwap_down_guidance_D": main_force_vwap_down_guidance,
            "order_flow_imbalance_score_D": order_flow_imbalance_score, "liquidity_slope_D": liquidity_slope,
            "order_book_clearing_rate_D": order_book_clearing_rate, "chip_fault_blockage_ratio_D": chip_fault_blockage_ratio,
            "winner_loser_momentum_D": winner_loser_momentum, "FRACTAL_DIMENSION_89d_D": fractal_dimension,
            "SAMPLE_ENTROPY_13d_D": sample_entropy, "micro_impact_elasticity_D": micro_impact_elasticity,
            "structural_entropy_change_D": structural_entropy_change,
            "volume_ratio_D": volume_ratio, "active_volume_price_efficiency_D": active_volume_price_efficiency,
            "constructive_turnover_ratio_D": constructive_turnover_ratio, "main_force_cost_advantage_D": main_force_cost_advantage,
            "main_force_activity_ratio_D": main_force_activity_ratio, "main_force_posture_index_D": main_force_posture_index,
            "main_force_on_peak_buy_flow_D": main_force_on_peak_buy_flow, "main_force_on_peak_sell_flow_D": main_force_on_peak_sell_flow,
            "SMART_MONEY_HM_COORDINATED_ATTACK_D": smart_money_coordinated_attack, "order_book_liquidity_supply_D": order_book_liquidity_supply,
            "closing_acceptance_type_D": closing_acceptance_type, "auction_showdown_score_D": auction_showdown_score,
            "chip_fatigue_index_D": chip_fatigue_index, "loser_pain_index_D": loser_pain_index,
            "profit_realization_quality_D": profit_realization_quality, "structural_tension_index_D": structural_tension_index,
            "upward_impulse_strength_D": upward_impulse_strength, "price_thrust_divergence_D": price_thrust_divergence,
            "equilibrium_compression_index_D": equilibrium_compression_index, "microstructure_efficiency_index_D": microstructure_efficiency_index,
            "liquidity_authenticity_score_D": liquidity_authenticity_score,
            "flow_credibility_index_D": flow_credibility_index,
            "is_consolidating_D": is_consolidating, "dynamic_consolidation_duration_D": dynamic_consolidation_duration,
            "breakout_readiness_score_D": breakout_readiness_score, "trend_acceleration_score_D": trend_acceleration_score,
            "trend_conviction_score_D": trend_conviction_score, "covert_accumulation_signal_D": covert_accumulation_signal,
            "covert_distribution_signal_D": covert_distribution_signal, "holistic_cmf_D": holistic_cmf,
            "main_force_net_flow_calibrated_D": main_force_net_flow_calibrated, "reversal_power_index_D": reversal_power_index,
            "reversal_recovery_rate_D": reversal_recovery_rate, "volatility_asymmetry_index_D": volatility_asymmetry_index,
            "mean_reversion_frequency_D": mean_reversion_frequency
        }
        # --- 2. MTF融合与归一化处理 ---
        # 价格和成交量动量
        mtf_price_momentum = self._get_mtf_slope_accel_score(df, 'close_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_volume_momentum = self._get_mtf_slope_accel_score(df, 'volume_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 主力资金与行为
        mtf_main_force_conviction = self._get_mtf_slope_accel_score(df, 'main_force_conviction_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_wash_trade_intensity = self._get_mtf_slope_accel_score(df, 'wash_trade_intensity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_retail_panic_surrender = self._get_mtf_slope_accel_score(df, 'retail_panic_surrender_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_upward_impulse_purity = self._get_mtf_slope_accel_score(df, 'upward_impulse_purity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_active_buying_support = self._get_mtf_slope_accel_score(df, 'active_buying_support_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_deception_index = self._get_mtf_slope_accel_score(df, 'deception_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_retail_fomo_premium_index = self._get_mtf_slope_accel_score(df, 'retail_fomo_premium_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 权力转移代理 (PROCESS_META_POWER_TRANSFER 的原始数据实现)
        effective_main_force_flow_proxy = (net_lg_amount + net_elg_amount).diff(1).fillna(0)
        effective_retail_flow_proxy = (net_sh_amount + net_md_amount).diff(1).fillna(0)
        power_transfer_raw_proxy = effective_main_force_flow_proxy - effective_retail_flow_proxy
        power_transfer_raw_proxy.name = 'power_transfer_raw_proxy_D' # 赋予临时列名
        mtf_power_transfer = self._get_mtf_slope_accel_score(df.assign(power_transfer_raw_proxy_D=power_transfer_raw_proxy), 'power_transfer_raw_proxy_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # 下影线吸收代理 (SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION 的原始数据实现)
        total_range = (high_price - low_price).replace(0, 1e-9)
        lower_shadow = np.minimum(open_price, close_price) - low_price
        lower_shadow_ratio = (lower_shadow / total_range).fillna(0)
        lower_shadow_absorption_raw = (lower_shadow_ratio > 0.3).astype(float) * (close_price > open_price * 0.99).astype(float)
        lower_shadow_absorption_raw.name = 'lower_shadow_absorption_raw_D' # 赋予临时列名
        mtf_lower_shadow_absorption = self._get_mtf_slope_accel_score(df.assign(lower_shadow_absorption_raw_D=lower_shadow_absorption_raw), 'lower_shadow_absorption_raw_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 量能萎缩代理 (SCORE_BEHAVIOR_VOLUME_ATROPHY 的原始数据实现)
        vol_ma_21 = self._get_safe_series(df, 'VOL_MA_21_D', 0.0, method_name=method_name)
        volume_atrophy_raw = (1 - (volume / vol_ma_21)).clip(0, 1) # 量能低于均线越多，萎缩越严重
        volume_atrophy_raw.name = 'volume_atrophy_raw_D' # 赋予临时列名
        mtf_volume_atrophy = self._get_mtf_slope_accel_score(df.assign(volume_atrophy_raw_D=volume_atrophy_raw), 'volume_atrophy_raw_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # 筹码战略态势代理 (SCORE_CHIP_STRATEGIC_POSTURE 的原始数据实现)
        mtf_winner_concentration = self._get_mtf_slope_accel_score(df, 'winner_concentration_90pct_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_chip_health = self._get_mtf_slope_accel_score(df, 'chip_health_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mf_vpoc_premium = self._get_mtf_slope_accel_score(df, 'mf_vpoc_premium_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_chip_strategic_posture = (mtf_winner_concentration * 0.4 + mtf_chip_health * 0.3 + mtf_mf_vpoc_premium * 0.3).clip(-1, 1)
        # V12.0 新增MTF融合信号
        mtf_volume_burstiness = self._get_mtf_slope_accel_score(df, 'volume_burstiness_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow_directionality = self._get_mtf_slope_accel_score(df, 'main_force_flow_directionality_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_order_book_imbalance = self._get_mtf_slope_accel_score(df, 'order_book_imbalance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_micro_price_impact_asymmetry = self._get_mtf_slope_accel_score(df, 'micro_price_impact_asymmetry_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_bid_side_liquidity = self._get_mtf_slope_accel_score(df, 'bid_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_ask_side_liquidity = self._get_mtf_slope_accel_score(df, 'ask_side_liquidity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vpin_score = self._get_mtf_slope_accel_score(df, 'vpin_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_loss_margin_avg = self._get_mtf_slope_accel_score(df, 'loser_loss_margin_avg_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_total_winner_rate = self._get_mtf_slope_accel_score(df, 'total_winner_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_total_loser_rate = self._get_mtf_slope_accel_score(df, 'total_loser_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_panic_selling_cascade = self._get_mtf_slope_accel_score(df, 'panic_selling_cascade_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_intraday_energy_density = self._get_mtf_slope_accel_score(df, 'intraday_energy_density_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_reversion_velocity = self._get_mtf_slope_accel_score(df, 'price_reversion_velocity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # V13.0 新增MTF融合信号
        mtf_volume_profile_entropy = self._get_mtf_slope_accel_score(df, 'volume_profile_entropy_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_volume_structure_skew = self._get_mtf_slope_accel_score(df, 'volume_structure_skew_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_vpa_efficiency = self._get_mtf_slope_accel_score(df, 'VPA_EFFICIENCY_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_vpa_buy_efficiency = self._get_mtf_slope_accel_score(df, 'VPA_BUY_EFFICIENCY_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_vpa_sell_efficiency = self._get_mtf_slope_accel_score(df, 'VPA_SELL_EFFICIENCY_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_turnover_rate_f = self._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_flow_gini = self._get_mtf_slope_accel_score(df, 'main_force_flow_gini_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_slippage_index = self._get_mtf_slope_accel_score(df, 'main_force_slippage_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_execution_alpha = self._get_mtf_slope_accel_score(df, 'main_force_execution_alpha_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_t0_buy_efficiency = self._get_mtf_slope_accel_score(df, 'main_force_t0_buy_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_t0_sell_efficiency = self._get_mtf_slope_accel_score(df, 'main_force_t0_sell_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_vwap_up_guidance = self._get_mtf_slope_accel_score(df, 'main_force_vwap_up_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_vwap_down_guidance = self._get_mtf_slope_accel_score(df, 'main_force_vwap_down_guidance_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_order_flow_imbalance_score = self._get_mtf_slope_accel_score(df, 'order_flow_imbalance_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_liquidity_slope = self._get_mtf_slope_accel_score(df, 'liquidity_slope_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_order_book_clearing_rate = self._get_mtf_slope_accel_score(df, 'order_book_clearing_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_chip_fault_blockage_ratio = self._get_mtf_slope_accel_score(df, 'chip_fault_blockage_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_winner_loser_momentum = self._get_mtf_slope_accel_score(df, 'winner_loser_momentum_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_fractal_dimension = self._get_mtf_slope_accel_score(df, 'FRACTAL_DIMENSION_89d_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_sample_entropy = self._get_mtf_slope_accel_score(df, 'SAMPLE_ENTROPY_13d_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_micro_impact_elasticity = self._get_mtf_slope_accel_score(df, 'micro_impact_elasticity_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_structural_entropy_change = self._get_mtf_slope_accel_score(df, 'structural_entropy_change_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        # V14.0 新增MTF融合信号
        mtf_volume_ratio = self._get_mtf_slope_accel_score(df, 'volume_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_active_volume_price_efficiency = self._get_mtf_slope_accel_score(df, 'active_volume_price_efficiency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_constructive_turnover_ratio = self._get_mtf_slope_accel_score(df, 'constructive_turnover_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_cost_advantage = self._get_mtf_slope_accel_score(df, 'main_force_cost_advantage_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_activity_ratio = self._get_mtf_slope_accel_score(df, 'main_force_activity_ratio_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_posture_index = self._get_mtf_slope_accel_score(df, 'main_force_posture_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_on_peak_buy_flow = self._get_mtf_slope_accel_score(df, 'main_force_on_peak_buy_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_main_force_on_peak_sell_flow = self._get_mtf_slope_accel_score(df, 'main_force_on_peak_sell_flow_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_smart_money_coordinated_attack = self._get_mtf_slope_accel_score(df, 'SMART_MONEY_HM_COORDINATED_ATTACK_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_order_book_liquidity_supply = self._get_mtf_slope_accel_score(df, 'order_book_liquidity_supply_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_closing_acceptance_type = self._get_mtf_slope_accel_score(df, 'closing_acceptance_type_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_auction_showdown_score = self._get_mtf_slope_accel_score(df, 'auction_showdown_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_chip_fatigue_index = self._get_mtf_slope_accel_score(df, 'chip_fatigue_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_loser_pain_index = self._get_mtf_slope_accel_score(df, 'loser_pain_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_profit_realization_quality = self._get_mtf_slope_accel_score(df, 'profit_realization_quality_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_structural_tension_index = self._get_mtf_slope_accel_score(df, 'structural_tension_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_upward_impulse_strength = self._get_mtf_slope_accel_score(df, 'upward_impulse_strength_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_price_thrust_divergence = self._get_mtf_slope_accel_score(df, 'price_thrust_divergence_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_equilibrium_compression_index = self._get_mtf_slope_accel_score(df, 'equilibrium_compression_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_microstructure_efficiency_index = self._get_mtf_slope_accel_score(df, 'microstructure_efficiency_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_liquidity_authenticity_score = self._get_mtf_slope_accel_score(df, 'liquidity_authenticity_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_flow_credibility_index = self._get_mtf_slope_accel_score(df, 'flow_credibility_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        # V15.0 新增MTF融合信号
        mtf_is_consolidating = self._get_mtf_slope_accel_score(df, 'is_consolidating_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_dynamic_consolidation_duration = self._get_mtf_slope_accel_score(df, 'dynamic_consolidation_duration_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_breakout_readiness_score = self._get_mtf_slope_accel_score(df, 'breakout_readiness_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_trend_acceleration_score = self._get_mtf_slope_accel_score(df, 'trend_acceleration_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_trend_conviction_score = self._get_mtf_slope_accel_score(df, 'trend_conviction_score_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_covert_accumulation_signal = self._get_mtf_slope_accel_score(df, 'covert_accumulation_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_covert_distribution_signal = self._get_mtf_slope_accel_score(df, 'covert_distribution_signal_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_holistic_cmf = self._get_mtf_slope_accel_score(df, 'holistic_cmf_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_main_force_net_flow_calibrated = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_reversal_power_index = self._get_mtf_slope_accel_score(df, 'reversal_power_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_reversal_recovery_rate = self._get_mtf_slope_accel_score(df, 'reversal_recovery_rate_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        mtf_volatility_asymmetry_index = self._get_mtf_slope_accel_score(df, 'volatility_asymmetry_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=True)
        mtf_mean_reversion_frequency = self._get_mtf_slope_accel_score(df, 'mean_reversion_frequency_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False)
        _temp_debug_values["MTF融合信号"] = {
            "mtf_price_momentum": mtf_price_momentum,
            "mtf_volume_momentum": mtf_volume_momentum,
            "mtf_main_force_conviction": mtf_main_force_conviction,
            "mtf_wash_trade_intensity": mtf_wash_trade_intensity,
            "mtf_retail_panic_surrender": mtf_retail_panic_surrender,
            "mtf_upward_impulse_purity": mtf_upward_impulse_purity,
            "mtf_active_buying_support": mtf_active_buying_support,
            "mtf_deception_index": mtf_deception_index,
            "mtf_retail_fomo_premium_index": mtf_retail_fomo_premium_index,
            "mtf_power_transfer": mtf_power_transfer,
            "mtf_lower_shadow_absorption": mtf_lower_shadow_absorption,
            "mtf_volume_atrophy": mtf_volume_atrophy,
            "mtf_chip_strategic_posture": mtf_chip_strategic_posture,
            "mtf_volume_burstiness": mtf_volume_burstiness,
            "mtf_main_force_flow_directionality": mtf_main_force_flow_directionality,
            "mtf_order_book_imbalance": mtf_order_book_imbalance,
            "mtf_micro_price_impact_asymmetry": mtf_micro_price_impact_asymmetry,
            "mtf_bid_side_liquidity": mtf_bid_side_liquidity,
            "mtf_ask_side_liquidity": mtf_ask_side_liquidity,
            "mtf_vpin_score": mtf_vpin_score,
            "mtf_loser_loss_margin_avg": mtf_loser_loss_margin_avg,
            "mtf_total_winner_rate": mtf_total_winner_rate,
            "mtf_total_loser_rate": mtf_total_loser_rate,
            "mtf_panic_selling_cascade": mtf_panic_selling_cascade,
            "mtf_intraday_energy_density": mtf_intraday_energy_density,
            "mtf_price_reversion_velocity": mtf_price_reversion_velocity,
            "mtf_volume_profile_entropy": mtf_volume_profile_entropy,
            "mtf_volume_structure_skew": mtf_volume_structure_skew,
            "mtf_vpa_efficiency": mtf_vpa_efficiency,
            "mtf_vpa_buy_efficiency": mtf_vpa_buy_efficiency,
            "mtf_vpa_sell_efficiency": mtf_vpa_sell_efficiency,
            "mtf_turnover_rate_f": mtf_turnover_rate_f,
            "mtf_main_force_flow_gini": mtf_main_force_flow_gini,
            "mtf_main_force_slippage_index": mtf_main_force_slippage_index,
            "mtf_main_force_execution_alpha": mtf_main_force_execution_alpha,
            "mtf_main_force_t0_buy_efficiency": mtf_main_force_t0_buy_efficiency,
            "mtf_main_force_t0_sell_efficiency": mtf_main_force_t0_sell_efficiency,
            "mtf_main_force_vwap_up_guidance": mtf_main_force_vwap_up_guidance,
            "mtf_main_force_vwap_down_guidance": mtf_main_force_vwap_down_guidance,
            "mtf_order_flow_imbalance_score": mtf_order_flow_imbalance_score,
            "mtf_liquidity_slope": mtf_liquidity_slope,
            "mtf_order_book_clearing_rate": mtf_order_book_clearing_rate,
            "mtf_chip_fault_blockage_ratio": mtf_chip_fault_blockage_ratio,
            "mtf_winner_loser_momentum": mtf_winner_loser_momentum,
            "mtf_fractal_dimension": mtf_fractal_dimension,
            "mtf_sample_entropy": mtf_sample_entropy,
            "mtf_micro_impact_elasticity": mtf_micro_impact_elasticity,
            "mtf_structural_entropy_change": mtf_structural_entropy_change,
            "mtf_volume_ratio": mtf_volume_ratio,
            "mtf_active_volume_price_efficiency": mtf_active_volume_price_efficiency,
            "mtf_constructive_turnover_ratio": mtf_constructive_turnover_ratio,
            "mtf_main_force_cost_advantage": mtf_main_force_cost_advantage,
            "mtf_main_force_activity_ratio": mtf_main_force_activity_ratio,
            "mtf_main_force_posture_index": mtf_main_force_posture_index,
            "mtf_main_force_on_peak_buy_flow": mtf_main_force_on_peak_buy_flow,
            "mtf_main_force_on_peak_sell_flow": mtf_main_force_on_peak_sell_flow,
            "mtf_smart_money_coordinated_attack": mtf_smart_money_coordinated_attack,
            "mtf_order_book_liquidity_supply": mtf_order_book_liquidity_supply,
            "mtf_closing_acceptance_type": mtf_closing_acceptance_type,
            "mtf_auction_showdown_score": mtf_auction_showdown_score,
            "mtf_chip_fatigue_index": mtf_chip_fatigue_index,
            "mtf_loser_pain_index": mtf_loser_pain_index,
            "mtf_profit_realization_quality": mtf_profit_realization_quality,
            "mtf_structural_tension_index": mtf_structural_tension_index,
            "mtf_upward_impulse_strength": mtf_upward_impulse_strength,
            "mtf_price_thrust_divergence": mtf_price_thrust_divergence,
            "mtf_equilibrium_compression_index": mtf_equilibrium_compression_index,
            "mtf_microstructure_efficiency_index": mtf_microstructure_efficiency_index,
            "mtf_liquidity_authenticity_score": mtf_liquidity_authenticity_score,
            "mtf_flow_credibility_index": mtf_flow_credibility_index,
            "mtf_is_consolidating": mtf_is_consolidating,
            "mtf_dynamic_consolidation_duration": mtf_dynamic_consolidation_duration,
            "mtf_breakout_readiness_score": mtf_breakout_readiness_score,
            "mtf_trend_acceleration_score": mtf_trend_acceleration_score,
            "mtf_trend_conviction_score": mtf_trend_conviction_score,
            "mtf_covert_accumulation_signal": mtf_covert_accumulation_signal,
            "mtf_covert_distribution_signal": mtf_covert_distribution_signal,
            "mtf_holistic_cmf": mtf_holistic_cmf,
            "mtf_main_force_net_flow_calibrated": mtf_main_force_net_flow_calibrated,
            "mtf_reversal_power_index": mtf_reversal_power_index,
            "mtf_reversal_recovery_rate": mtf_reversal_recovery_rate,
            "mtf_volatility_asymmetry_index": mtf_volatility_asymmetry_index,
            "mtf_mean_reversion_frequency": mtf_mean_reversion_frequency
        }
        # --- 辅助函数：动态权重调整 ---
        def _get_dynamic_weights(base_weights: Dict[str, float], context_modulator_score: pd.Series, dynamic_weight_sensitivity: float) -> Dict[str, pd.Series]:
            dynamic_weights = {}
            for key, base_w in base_weights.items():
                # 简单示例：根据市场情绪调整权重，情绪越好，某些积极信号权重越高
                # 实际应用中可以根据具体信号和情境设计更复杂的调整逻辑
                # 这里使用一个简化的逻辑，积极信号在情境分数高时权重增加，消极信号在情境分数低时权重增加
                # context_modulator_score 范围 [0, 1]
                if "positive" in key or "inverted" in key or "high" in key or "calm" in key or "efficiency" in key or "purity" in key or "strength" in key or "quality" in key or "attack" in key or "readiness" in key or "recovery" in key or "accumulation" in key:
                    dynamic_weights[key] = base_w * (1 + context_modulator_score * dynamic_weight_sensitivity) # 积极信号在好情境下权重增加
                elif "negative" in key or "low" in key or "fatigue" in key or "blockage" in key or "penalty" in key or "deception" in key or "wash_trade" in key or "slippage" in key or "pain" in key or "tension" in key or "distribution" in key:
                    dynamic_weights[key] = base_w * (1 + (1 - context_modulator_score) * dynamic_weight_sensitivity) # 消极信号在坏情境下权重增加
                else:
                    dynamic_weights[key] = pd.Series(base_w, index=df_index)
            # 归一化动态权重
            total_dynamic_weight = pd.Series(0.0, index=df_index, dtype=np.float32)
            for key in dynamic_weights:
                total_dynamic_weight += dynamic_weights[key]
            # 避免除以零
            total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
            for key in dynamic_weights:
                dynamic_weights[key] = dynamic_weights[key] / total_dynamic_weight
            return dynamic_weights
        # --- 市场情境因子 (用于动态权重调整) ---
        norm_market_sentiment = self._normalize_series(market_sentiment_score, df_index, bipolar=True)
        norm_volatility_inverse = self._normalize_series(volatility_instability_index, df_index, ascending=False)
        norm_trend_vitality = self._normalize_series(trend_vitality_index, df_index, bipolar=False)
        norm_liquidity_authenticity = self._normalize_series(liquidity_authenticity_score, df_index, bipolar=False)
        # --- 历史情境感知层 (V15.0 新增) ---
        # 象限持续性分数 (QPS)
        # 假设我们已经有了象限的初步判断，这里先用MTF动量作为代理
        # 实际应用中，Q1_score, Q2_score, Q3_score, Q4_score 应该来自上一个迭代的计算结果或更复杂的分类器
        # 为了避免循环依赖，这里先用MTF动量作为代理，后续可以优化为基于历史最终分数的象限归属
        # 临时代理：
        temp_q1_proxy = (mtf_price_momentum.clip(lower=0) * mtf_volume_momentum.clip(lower=0)).clip(0,1)
        temp_q2_proxy = (mtf_price_momentum.clip(lower=0) * mtf_volume_momentum.clip(upper=0).abs()).clip(0,1)
        temp_q3_proxy = (mtf_price_momentum.clip(upper=0).abs() * mtf_volume_momentum.clip(lower=0)).clip(0,1)
        temp_q4_proxy = (mtf_price_momentum.clip(upper=0).abs() * mtf_volume_momentum.clip(upper=0).abs()).clip(0,1)
        quadrant_persistence_Q1_Q4 = (temp_q1_proxy.rolling(window=historical_window_short).mean() + temp_q4_proxy.rolling(window=historical_window_short).mean()) / 2
        quadrant_persistence_Q2_Q3 = (temp_q2_proxy.rolling(window=historical_window_short).mean() + temp_q3_proxy.rolling(window=historical_window_short).mean()) / 2
        quadrant_persistence_Q1_Q4 = self._normalize_series(quadrant_persistence_Q1_Q4, df_index, bipolar=False)
        quadrant_persistence_Q2_Q3 = self._normalize_series(quadrant_persistence_Q2_Q3, df_index, bipolar=False)
        # 阶段转换指示器 (PTS)
        # 示例：从Q4到Q1的转换，或从Q1到Q2的警告
        phase_transition_Q4_to_Q1 = ((temp_q4_proxy.shift(1) > 0.5) & (temp_q1_proxy > 0.5)).astype(float)
        phase_transition_Q4_to_Q1 = self._normalize_series(phase_transition_Q4_to_Q1.rolling(window=historical_window_short).mean(), df_index, bipolar=False)
        # 累积资金流平衡 (CFB)
        cumulative_flow_balance_raw = (mtf_covert_accumulation_signal - mtf_covert_distribution_signal + mtf_holistic_cmf + mtf_main_force_net_flow_calibrated).rolling(window=historical_window_long).sum()
        cumulative_flow_balance = self._normalize_series(cumulative_flow_balance_raw, df_index, bipolar=True)
        # 市场阶段强度 (MRS)
        market_regime_strength_raw = (1 - mtf_is_consolidating) * mtf_trend_conviction_score + mtf_breakout_readiness_score + mtf_trend_acceleration_score
        market_regime_strength = self._normalize_series(market_regime_strength_raw, df_index, bipolar=False)
        # 反转潜力分数 (RPS) - 用于动态指数
        reversal_potential_score = self._normalize_series(mtf_reversal_power_index + mtf_reversal_recovery_rate, df_index, bipolar=False)
        _temp_debug_values["历史情境感知层"] = {
            "quadrant_persistence_Q1_Q4": quadrant_persistence_Q1_Q4,
            "quadrant_persistence_Q2_Q3": quadrant_persistence_Q2_Q3,
            "phase_transition_Q4_to_Q1": phase_transition_Q4_to_Q1,
            "cumulative_flow_balance": cumulative_flow_balance,
            "market_regime_strength": market_regime_strength,
            "reversal_potential_score": reversal_potential_score
        }
        # --- V16.0 情境调制器生成 ---
        # 将 bipolar 的 cumulative_flow_balance 映射到 [0, 1]
        accumulation_strength_modulator = (cumulative_flow_balance + 1) / 2
        trend_strength_modulator = market_regime_strength
        bullish_persistence_modulator = quadrant_persistence_Q1_Q4
        bearish_persistence_modulator = quadrant_persistence_Q2_Q3
        # V17.0 新增：整体看涨/看跌情境强度，用于最终指数调整
        overall_bullish_context_strength = (accumulation_strength_modulator + trend_strength_modulator + bullish_persistence_modulator) / 3
        overall_bearish_context_strength = ((1 - accumulation_strength_modulator) + (1 - trend_strength_modulator) + bearish_persistence_modulator) / 3
        _temp_debug_values["情境调制器"] = {
            "accumulation_strength_modulator": accumulation_strength_modulator,
            "trend_strength_modulator": trend_strength_modulator,
            "bullish_persistence_modulator": bullish_persistence_modulator,
            "bearish_persistence_modulator": bearish_persistence_modulator,
            "overall_bullish_context_strength": overall_bullish_context_strength,
            "overall_bearish_context_strength": overall_bearish_context_strength
        }
        context_modulator_components_for_weights = {
            "market_sentiment": norm_market_sentiment,
            "volatility_inverse": norm_volatility_inverse,
            "trend_vitality": norm_trend_vitality,
            "liquidity_authenticity_score": norm_liquidity_authenticity,
            "market_regime_strength": market_regime_strength # 新增
        }
        context_modulator_score_for_weights = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components_for_weights.items()},
            dynamic_context_modulator_weights,
            df_index
        )
        # --- 3. 多层级共振引擎 ---
        # 3.1 价格-成交量共振
        dynamic_pv_resonance_weights = _get_dynamic_weights(price_volume_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        price_volume_resonance_components_dict = {
            "lower_shadow_absorption": mtf_lower_shadow_absorption,
            "active_buying_support": mtf_active_buying_support,
            "volume_burstiness": mtf_volume_burstiness,
            "VPA_EFFICIENCY": mtf_vpa_efficiency.clip(lower=0), # 只取正向效率
            "volume_profile_entropy_inverted": (1 - mtf_volume_profile_entropy), # 熵越低越好
            "volume_ratio_positive": mtf_volume_ratio.clip(lower=0), # 量比越高越好
            "upward_impulse_strength": mtf_upward_impulse_strength # 上涨脉冲强度
        }
        price_volume_resonance = _robust_geometric_mean(price_volume_resonance_components_dict, dynamic_pv_resonance_weights, df_index).clip(0, 1)
        # 3.2 主力-筹码共振
        dynamic_mc_resonance_weights = _get_dynamic_weights(main_chip_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        main_chip_resonance_components_dict = {
            "power_transfer_positive": mtf_power_transfer.clip(lower=0),
            "main_force_conviction_positive": mtf_main_force_conviction.clip(lower=0),
            "main_force_flow_directionality_positive": mtf_main_force_flow_directionality.clip(lower=0),
            "chip_strategic_posture": mtf_chip_strategic_posture.clip(lower=0),
            "chip_fault_blockage_ratio_inverted": (1 - mtf_chip_fault_blockage_ratio), # 堵塞比越低越好
            "main_force_cost_advantage_positive": mtf_main_force_cost_advantage.clip(lower=0), # 主力成本优势
            "SMART_MONEY_HM_COORDINATED_ATTACK": mtf_smart_money_coordinated_attack # 智能资金协同攻击
        }
        main_chip_resonance = _robust_geometric_mean(main_chip_resonance_components_dict, dynamic_mc_resonance_weights, df_index).clip(0, 1)
        # 3.3 市场情绪-流动性共振
        dynamic_sl_resonance_weights = _get_dynamic_weights(sentiment_liquidity_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        sentiment_liquidity_resonance_components_dict = {
            "market_sentiment_positive": market_sentiment_score.clip(lower=0), # 使用原始情绪分数，MTF情绪可能过于平滑
            "retail_panic_surrender_inverted": (1 - mtf_retail_panic_surrender), # 散户恐慌越低越好
            "bid_side_liquidity": mtf_bid_side_liquidity,
            "liquidity_slope_positive": mtf_liquidity_slope.clip(lower=0),
            "order_flow_imbalance_positive": mtf_order_flow_imbalance_score.clip(lower=0),
            "loser_pain_index_inverted": (1 - mtf_loser_pain_index) # 输家痛苦指数越低越好 (反向指标)
        }
        sentiment_liquidity_resonance = _robust_geometric_mean(sentiment_liquidity_resonance_components_dict, dynamic_sl_resonance_weights, df_index).clip(0, 1)
        # 3.4 微观结构共振
        dynamic_ms_resonance_weights = _get_dynamic_weights(micro_structure_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        micro_structure_resonance_components_dict = {
            "order_book_imbalance_positive": mtf_order_book_imbalance.clip(lower=0),
            "micro_price_impact_asymmetry_positive": mtf_micro_price_impact_asymmetry.clip(lower=0),
            "intraday_energy_density": mtf_intraday_energy_density,
            "vpin_score_inverted": (1 - mtf_vpin_score), # VPIN越低越好
            "micro_impact_elasticity_positive": mtf_micro_impact_elasticity.clip(lower=0),
            "order_book_clearing_rate": mtf_order_book_clearing_rate,
            "closing_acceptance_type_positive": mtf_closing_acceptance_type.clip(lower=0) # 收盘接受类型积极
        }
        micro_structure_resonance = _robust_geometric_mean(micro_structure_resonance_components_dict, dynamic_ms_resonance_weights, df_index).clip(0, 1)
        # 3.5 质量与效率共振 (新增层级)
        dynamic_qe_resonance_weights = _get_dynamic_weights(quality_efficiency_resonance_components, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        quality_efficiency_resonance_components_dict = {
            "upward_impulse_purity": mtf_upward_impulse_purity,
            "flow_credibility_index": mtf_flow_credibility_index,
            "profit_realization_quality": mtf_profit_realization_quality,
            "active_volume_price_efficiency": mtf_active_volume_price_efficiency.clip(lower=0),
            "constructive_turnover_ratio": mtf_constructive_turnover_ratio
        }
        quality_efficiency_resonance = _robust_geometric_mean(quality_efficiency_resonance_components_dict, dynamic_qe_resonance_weights, df_index).clip(0, 1)
        # 3.6 融合所有多层级共振因子
        multi_level_resonance_factor_dict = {
            "price_volume_resonance": price_volume_resonance,
            "main_chip_resonance": main_chip_resonance,
            "sentiment_liquidity_resonance": sentiment_liquidity_resonance,
            "micro_structure_resonance": micro_structure_resonance,
            "quality_efficiency_resonance": quality_efficiency_resonance
        }
        multi_level_resonance_factor = _robust_geometric_mean(multi_level_resonance_factor_dict, multi_level_resonance_weights, df_index).clip(0, 1)
        _temp_debug_values["多层级共振引擎"] = {
            "price_volume_resonance": price_volume_resonance,
            "main_chip_resonance": main_chip_resonance,
            "sentiment_liquidity_resonance": sentiment_liquidity_resonance,
            "micro_structure_resonance": micro_structure_resonance,
            "quality_efficiency_resonance": quality_efficiency_resonance,
            "multi_level_resonance_factor": multi_level_resonance_factor
        }
        # --- 4. 四象限分数计算 ---
        p_mom = mtf_price_momentum
        v_mom = mtf_volume_momentum
        final_score = pd.Series(0.0, index=df_index)
        # 市场情境因子：波动率、趋势强度、市场情绪 (已在上方计算)
        # norm_market_sentiment, norm_volatility_inverse, norm_trend_vitality, norm_liquidity_authenticity
        # 动态阈值：基于价格动量和成交量动量的历史标准差，并考虑市场情境
        price_volatility_norm = self._normalize_series(close_price.pct_change().rolling(self.std_window).std(), df_index, bipolar=False)
        volume_volatility_norm = self._normalize_series(volume.pct_change().rolling(self.std_window).std(), df_index, bipolar=False)
        # 动态阈值进一步考虑市场趋势强度、情绪和流动性真实性
        dynamic_threshold_modulator = (norm_trend_vitality * 0.3 + (1 - norm_market_sentiment.abs()) * 0.3 + norm_liquidity_authenticity * 0.4).clip(0.5, 1.5) # 趋势强、情绪中性或流动性真实时，阈值更严格
        dynamic_price_threshold = price_volatility_norm * dynamic_threshold_sensitivity * dynamic_threshold_modulator
        dynamic_volume_threshold = volume_volatility_norm * dynamic_threshold_sensitivity * dynamic_threshold_modulator
        _temp_debug_values["动态阈值"] = {
            "price_volatility_norm": price_volatility_norm,
            "volume_volatility_norm": volume_volatility_norm,
            "dynamic_threshold_modulator": dynamic_threshold_modulator,
            "dynamic_price_threshold": dynamic_price_threshold,
            "dynamic_volume_threshold": dynamic_volume_threshold
        }
        # Q1: 价涨量增 (健康上涨)
        # V16.0 信号影响力调制
        deception_impact_reduction = accumulation_strength_modulator * context_impact_modulators.get("deception_impact_reduction_factor", 0.5)
        trend_reward_enhancement = trend_strength_modulator * context_impact_modulators.get("trend_reward_enhancement_factor", 0.2)
        mtf_wash_trade_intensity_adjusted = mtf_wash_trade_intensity * (1 - deception_impact_reduction)
        mtf_deception_index_adjusted = mtf_deception_index * (1 - deception_impact_reduction)
        mtf_main_force_t0_sell_efficiency_adjusted = mtf_main_force_t0_sell_efficiency * (1 - deception_impact_reduction)
        mtf_ask_side_liquidity_adjusted = mtf_ask_side_liquidity * (1 - trend_reward_enhancement) # 强趋势下，卖盘流动性高不一定是坏事
        mtf_profit_realization_quality_low_adjusted = (1 - mtf_profit_realization_quality) * (1 - trend_reward_enhancement) # 强趋势下，获利了结质量低可能只是正常调整
        dynamic_Q1_reward_weights = _get_dynamic_weights(Q1_reward_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q1_reward_components_dict = {
            "p_mom": p_mom.clip(lower=0),
            "v_mom": v_mom.clip(lower=0),
            "upward_purity": mtf_upward_impulse_purity * (1 + trend_reward_enhancement), # 增强奖励
            "main_force_conviction": mtf_main_force_conviction.clip(lower=0) * (1 + trend_reward_enhancement), # 增强奖励
            "main_force_flow_directionality_positive": mtf_main_force_flow_directionality.clip(lower=0) * (1 + trend_reward_enhancement), # 增强奖励
            "VPA_BUY_EFFICIENCY": mtf_vpa_buy_efficiency * (1 + trend_reward_enhancement), # 奖励买入效率
            "main_force_execution_alpha": mtf_main_force_execution_alpha.clip(lower=0) * (1 + trend_reward_enhancement), # 奖励主力执行Alpha
            "order_flow_imbalance_positive": mtf_order_flow_imbalance_score.clip(lower=0) * (1 + trend_reward_enhancement), # 奖励订单流买方不平衡
            "main_force_vwap_up_guidance_positive": mtf_main_force_vwap_up_guidance.clip(lower=0) * (1 + trend_reward_enhancement), # 奖励主力VWAP向上引导
            "upward_impulse_strength": mtf_upward_impulse_strength * (1 + trend_reward_enhancement), # 上涨脉冲强度
            "flow_credibility_index": mtf_flow_credibility_index * (1 + trend_reward_enhancement) # 资金流可信度
        }
        score1_reward = _robust_geometric_mean(Q1_reward_components_dict, dynamic_Q1_reward_weights, df_index).clip(0, 1)
        # 惩罚虚假上涨：对倒、欺骗过高时惩罚，主力T0卖出效率高，卖盘流动性过高，获利了结质量低
        dynamic_Q1_penalty_weights = _get_dynamic_weights(Q1_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q1_penalty_components_dict = {
            "wash_trade": mtf_wash_trade_intensity_adjusted,
            "deception_index": mtf_deception_index_adjusted,
            "main_force_t0_sell_efficiency": mtf_main_force_t0_sell_efficiency_adjusted,
            "ask_side_liquidity_high": mtf_ask_side_liquidity_adjusted, # 卖盘流动性高，阻力大
            "profit_realization_quality_low": mtf_profit_realization_quality_low_adjusted # 获利了结质量低
        }
        false_rally_penalty = _robust_geometric_mean(Q1_penalty_components_dict, dynamic_Q1_penalty_weights, df_index).clip(0, 1)
        score1 = score1_reward * (1 - false_rally_penalty)
        # Q2: 价涨量缩 (上涨乏力/背离 - 负向信号)
        # V16.0 信号影响力调制
        divergence_penalty_enhancement = bullish_persistence_modulator * context_impact_modulators.get("divergence_penalty_enhancement_factor", 0.3)
        deception_impact_reduction_Q2 = accumulation_strength_modulator * context_impact_modulators.get("deception_impact_reduction_factor", 0.5)
        mtf_retail_fomo_adjusted = mtf_retail_fomo_premium_index * (1 - deception_impact_reduction_Q2)
        mtf_wash_trade_adjusted_Q2 = mtf_wash_trade_intensity * (1 - deception_impact_reduction_Q2)
        mtf_deception_index_adjusted_Q2 = mtf_deception_index * (1 - deception_impact_reduction_Q2)
        mtf_vpin_score_high_adjusted = mtf_vpin_score * (1 + divergence_penalty_enhancement)
        mtf_winner_loser_momentum_negative_adjusted = mtf_winner_loser_momentum.clip(upper=0).abs() * (1 + divergence_penalty_enhancement)
        mtf_price_thrust_divergence_negative_adjusted = mtf_price_thrust_divergence.clip(upper=0).abs() * (1 + divergence_penalty_enhancement)
        # 显式背离证据：价格上涨，成交量未有效放大，且主力流向偏负
        price_up_volume_not_up = (p_mom > dynamic_price_threshold) & (v_mom < dynamic_volume_threshold) & (mtf_main_force_flow_directionality < 0)
        dynamic_Q2_divergence_penalty_weights = _get_dynamic_weights(Q2_divergence_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q2_divergence_penalty_components_dict = {
            "retail_fomo": mtf_retail_fomo_adjusted, # 散户追涨，风险高
            "wash_trade": mtf_wash_trade_adjusted_Q2, # 对倒，虚假上涨
            "deception_index": mtf_deception_index_adjusted_Q2, # 欺骗，风险高
            "vpin_score_high": mtf_vpin_score_high_adjusted, # VPIN高，可能预示反转
            "winner_loser_momentum_negative": mtf_winner_loser_momentum_negative_adjusted, # 赢家动量减弱，输家动量增强
            "price_thrust_divergence_negative": mtf_price_thrust_divergence_negative_adjusted # 价格推力背离
        }
        divergence_penalty_factor = _robust_geometric_mean(Q2_divergence_penalty_components_dict, dynamic_Q2_divergence_penalty_weights, df_index).clip(0, 1)
        score2_magnitude = _robust_geometric_mean(
            {"p_mom_positive": p_mom.clip(lower=0), "v_mom_negative_abs": v_mom.clip(upper=0).abs()},
            {"p_mom_positive": 0.5, "v_mom_negative_abs": 0.5},
            df_index
        )
        score2 = -(score2_magnitude * (1 + divergence_penalty_factor)).clip(0, 1) # 惩罚因子越高，负向信号越强
        score2 = score2.where(price_up_volume_not_up, 0.0) # 只有在价涨量缩背离时才激活
        # 历史情境增强惩罚：如果Q1持续性强，但当前出现Q2，则视为更严重的趋势反转预警
        score2 -= score2.abs() * bullish_persistence_modulator * 0.5 # Q1持续性越强，Q2惩罚越大
        # Q3: 价跌量增 (放量下跌/恐慌 - 负向信号)
        # V16.0 信号影响力调制
        panic_impact_reduction = accumulation_strength_modulator * context_impact_modulators.get("panic_impact_reduction_factor", 0.4)
        absorption_reward_enhancement = cumulative_flow_balance.clip(lower=0) * context_impact_modulators.get("absorption_reward_enhancement_factor", 0.3)
        blockage_penalty_enhancement = bearish_persistence_modulator * context_impact_modulators.get("blockage_penalty_enhancement_factor", 0.3)
        mtf_retail_panic_surrender_adjusted = mtf_retail_panic_surrender * (1 - panic_impact_reduction)
        mtf_panic_selling_cascade_adjusted = mtf_panic_selling_cascade * (1 - panic_impact_reduction)
        mtf_chip_fault_blockage_ratio_adjusted = mtf_chip_fault_blockage_ratio * (1 + blockage_penalty_enhancement)
        mtf_structural_tension_index_adjusted = mtf_structural_tension_index * (1 + blockage_penalty_enhancement)
        Q3_panic_evidence_components = {
            "retail_panic_surrender": mtf_retail_panic_surrender_adjusted,
            "chip_strategic_posture_negative": mtf_chip_strategic_posture.clip(upper=0).abs(), # 筹码态势恶化
            "loser_loss_margin_avg_positive": mtf_loser_loss_margin_avg.clip(lower=0), # 输家亏损加剧
            "total_loser_rate_positive": mtf_total_loser_rate, # 输家比例增加
            "panic_selling_cascade": mtf_panic_selling_cascade_adjusted # 恐慌抛售级联
        }
        Q3_panic_evidence_weights_internal = {"retail_panic_surrender": 0.25, "chip_strategic_posture_negative": 0.2, "loser_loss_margin_avg_positive": 0.2, "total_loser_rate_positive": 0.15, "panic_selling_cascade": 0.2}
        panic_evidence_factor = _robust_geometric_mean(Q3_panic_evidence_components, Q3_panic_evidence_weights_internal, df_index).clip(0, 1)
        score3_magnitude = _robust_geometric_mean(
            {"p_mom_negative_abs": p_mom.clip(upper=0).abs(), "v_mom_positive": v_mom.clip(lower=0)},
            {"p_mom_negative_abs": 0.5, "v_mom_positive": 0.5},
            df_index
        )
        score3_base = -(score3_magnitude * (1 + panic_evidence_factor)).clip(0, 1) # 恐慌证据越高，负向信号越强
        # 奖励逆势承接：主动买盘支持、下影线吸收、主力逆势吸筹、投降式吸收
        dynamic_Q3_reward_weights = _get_dynamic_weights(Q3_reward_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q3_reward_components_dict = {
            "lower_shadow_absorption": mtf_lower_shadow_absorption * (1 + absorption_reward_enhancement),
            "active_buying_support": mtf_active_buying_support * (1 + absorption_reward_enhancement),
            "main_force_flow_directionality_positive": mtf_main_force_flow_directionality.clip(lower=0) * (1 + absorption_reward_enhancement),
            "main_force_t0_buy_efficiency": mtf_main_force_t0_buy_efficiency * (1 + absorption_reward_enhancement),
            "capitulation_absorption_index": self._get_mtf_slope_accel_score(df, 'capitulation_absorption_index_D', mtf_slope_accel_weights, df_index, method_name, bipolar=False) * (1 + absorption_reward_enhancement) # 投降式吸收
        }
        absorption_reward = _robust_geometric_mean(Q3_reward_components_dict, dynamic_Q3_reward_weights, df_index).clip(0, 1)
        # 惩罚筹码堵塞：输家亏损扩大，恐慌抛售级联，筹码断层堵塞比高，下跌脉冲强度高，结构张力高
        dynamic_Q3_penalty_weights = _get_dynamic_weights(Q3_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q3_penalty_components_dict = {
            "loser_loss_margin_avg_expanding": mtf_loser_loss_margin_avg.clip(lower=0), # 输家亏损扩大
            "panic_selling_cascade": mtf_panic_selling_cascade_adjusted,
            "chip_fault_blockage_ratio": mtf_chip_fault_blockage_ratio_adjusted, # 筹码断层堵塞比高
            "downward_impulse_strength": mtf_upward_impulse_strength.clip(upper=0).abs(), # 下跌脉冲强度 (使用upward_impulse_strength的负向部分作为代理)
            "structural_tension_index": mtf_structural_tension_index_adjusted # 结构张力
        }
        blockage_penalty = _robust_geometric_mean(Q3_penalty_components_dict, dynamic_Q3_penalty_weights, df_index).clip(0, 1)
        score3 = score3_base * (1 - absorption_reward) * (1 + blockage_penalty) # 承接越强，负分越小；堵塞越强，负分越大
        # 历史情境增强奖励/惩罚：如果累积资金流开始转正，减少负分；如果Q3持续性强，增强惩罚
        score3 -= score3.abs() * bearish_persistence_modulator * 0.5 # Q2/Q3持续性越强，Q3惩罚越大
        # Q4: 价跌量缩 (卖压枯竭/底部 - 双向信号)
        # V16.0 信号影响力调制
        exhaustion_reward_enhancement = bearish_persistence_modulator * context_impact_modulators.get("exhaustion_reward_enhancement_factor", 0.3)
        false_bottom_penalty_reduction = accumulation_strength_modulator * context_impact_modulators.get("false_bottom_penalty_reduction_factor", 0.4)
        mtf_volume_atrophy_adjusted = mtf_volume_atrophy * (1 + exhaustion_reward_enhancement)
        mtf_retail_panic_surrender_adjusted_Q4 = mtf_retail_panic_surrender * (1 + exhaustion_reward_enhancement)
        mtf_loser_pain_index_high_adjusted = mtf_loser_pain_index.clip(lower=0) * (1 + exhaustion_reward_enhancement)
        mtf_price_reversion_velocity_negative_adjusted = mtf_price_reversion_velocity.clip(upper=0).abs() * (1 - false_bottom_penalty_reduction)
        mtf_structural_entropy_change_positive_adjusted = mtf_structural_entropy_change.clip(lower=0) * (1 - false_bottom_penalty_reduction)
        mtf_main_force_vwap_down_guidance_positive_adjusted = mtf_main_force_vwap_down_guidance.clip(lower=0) * (1 - false_bottom_penalty_reduction)
        mtf_chip_fatigue_index_low_adjusted = (1 - mtf_chip_fatigue_index) * (1 - false_bottom_penalty_reduction)
        dynamic_Q4_reward_weights = _get_dynamic_weights(Q4_reward_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q4_exhaustion_evidence_components_dict = {
            "volume_atrophy": mtf_volume_atrophy_adjusted,
            "retail_panic_surrender": mtf_retail_panic_surrender_adjusted_Q4, # 散户恐慌投降，可能接近底部
            "lower_shadow_absorption": mtf_lower_shadow_absorption,
            "chip_health": mtf_chip_strategic_posture.clip(lower=0), # 筹码健康度改善
            "bid_side_liquidity": mtf_bid_side_liquidity, # 买盘流动性增加
            "vpin_score_low": (1 - mtf_vpin_score), # VPIN低，卖压小
            "volume_profile_entropy_inverted": (1 - mtf_volume_profile_entropy), # 熵越低越好
            "FRACTAL_DIMENSION_calm": (1 - (mtf_fractal_dimension - 1.5).abs() / 0.5).clip(0, 1), # 分形维数接近1.5
            "loser_pain_index_high": mtf_loser_pain_index_high_adjusted, # 输家痛苦指数高
            "equilibrium_compression_index": mtf_equilibrium_compression_index # 均衡压缩指数
        }
        exhaustion_evidence_factor = _robust_geometric_mean(Q4_exhaustion_evidence_components_dict, dynamic_Q4_reward_weights, df_index).clip(0, 1)
        score4_magnitude = _robust_geometric_mean(
            {"p_mom_negative_abs": p_mom.clip(upper=0).abs(), "v_mom_negative_abs": v_mom.clip(upper=0).abs()},
            {"p_mom_negative_abs": 0.5, "v_mom_negative_abs": 0.5},
            df_index
        )
        # 如果卖压枯竭证据强，则为正向信号；否则为负向信号（持续弱势）
        score4_base = (score4_magnitude * exhaustion_evidence_factor - score4_magnitude * (1 - exhaustion_evidence_factor)).clip(-1, 1)
        # 惩罚虚假底部：价格回归速度过快（向下），结构熵增加，主力VWAP向下引导，筹码疲劳指数低
        dynamic_Q4_penalty_weights = _get_dynamic_weights(Q4_penalty_weights, context_modulator_score_for_weights, dynamic_weight_sensitivity)
        Q4_penalty_components_dict = {
            "price_reversion_velocity_negative": mtf_price_reversion_velocity_negative_adjusted,
            "structural_entropy_change_positive": mtf_structural_entropy_change_positive_adjusted,
            "main_force_vwap_down_guidance_positive": mtf_main_force_vwap_down_guidance_positive_adjusted, # 主力VWAP向下引导
            "chip_fatigue_index_low": mtf_chip_fatigue_index_low_adjusted # 筹码疲劳指数低
        }
        false_bottom_penalty = _robust_geometric_mean(Q4_penalty_components_dict, dynamic_Q4_penalty_weights, df_index).clip(0, 1)
        score4 = score4_base * (1 - false_bottom_penalty) # 惩罚越强，分数越低
        # 历史情境增强奖励/惩罚：如果Q3持续性强，且识别到Q3->Q4转换，增强奖励；如果市场阶段指示盘整，且Q4持续性强，增强奖励
        score4 += phase_transition_Q4_to_Q1 * 0.3 # Q4到Q1转换，增强奖励
        score4 += bullish_persistence_modulator * 0.2 # Q4持续性强，增强奖励
        _temp_debug_values["四象限分数"] = {
            "p_mom": p_mom,
            "v_mom": v_mom,
            "score1": score1,
            "score2": score2,
            "score3": score3,
            "score4": score4
        }
        # --- 5. 动态权重与最终融合 ---
        # 市场情境因子：波动率、趋势强度、市场情绪 (已在上方计算)
        # norm_market_sentiment, norm_volatility_inverse, norm_trend_vitality, norm_liquidity_authenticity
        context_modulator_components = {
            "market_sentiment": norm_market_sentiment,
            "volatility_inverse": norm_volatility_inverse,
            "trend_vitality": norm_trend_vitality,
            "liquidity_authenticity_score": norm_liquidity_authenticity,
            "market_regime_strength": market_regime_strength # 新增
        }
        # 确保输入为正，然后进行几何平均
        context_modulator_score = _robust_geometric_mean(
            {k: (v + 1) / 2 if v.min() < 0 else v for k, v in context_modulator_components.items()},
            dynamic_context_modulator_weights,
            df_index
        )
        # 将情境调制器映射到 [0.5, 1.5] 范围，以实现放大或抑制
        context_modulator = 0.5 + context_modulator_score # 0.5 + [0,1] -> [0.5, 1.5]
        # 动态调整象限权重
        dynamic_quadrant_weights = {}
        for q_name, base_w in quadrant_weights.items():
            if "Q1" in q_name or "Q4" in q_name: # 看涨象限，市场情境越好，权重越高
                dynamic_quadrant_weights[q_name] = base_w * (1 + context_modulator_score * dynamic_weight_sensitivity)
            elif "Q2" in q_name or "Q3" in q_name: # 看跌象限，市场情境越差，权重越高
                dynamic_quadrant_weights[q_name] = base_w * (1 + (1 - context_modulator_score) * dynamic_weight_sensitivity)
            dynamic_quadrant_weights[q_name] = dynamic_quadrant_weights[q_name].clip(0.05, 0.5) # 限制权重范围
        # 归一化动态权重
        total_dynamic_weight = pd.Series(0.0, index=df_index, dtype=np.float32)
        for key in dynamic_quadrant_weights:
            total_dynamic_weight += dynamic_quadrant_weights[key]
        # 避免除以零
        total_dynamic_weight = total_dynamic_weight.replace(0, 1e-9)
        for key in dynamic_quadrant_weights:
            dynamic_quadrant_weights[key] = dynamic_quadrant_weights[key] / total_dynamic_weight
        # 动态调整 final_exponent (V17.0 核心修改)
        dynamic_final_exponent_components = {
            "volatility_inverse": norm_volatility_inverse,
            "trend_vitality": norm_trend_vitality,
            "market_sentiment": norm_market_sentiment.clip(lower=0), # 情绪积极时放大
            "liquidity_slope_positive": mtf_liquidity_slope.clip(lower=0), # 流动性斜率积极时放大
            "microstructure_efficiency_index": mtf_microstructure_efficiency_index, # 微观结构效率
            "reversal_potential_score": reversal_potential_score # 反转潜力分数
        }
        dynamic_exponent_modulator = _robust_geometric_mean(dynamic_final_exponent_components, dynamic_exponent_modulator_weights, df_index)
        # 新的指数计算逻辑：当情境越积极 (dynamic_exponent_modulator 越高)，指数越小 (越趋近于0)，从而放大分数
        adjusted_final_exponent = final_exponent_base * (1 - dynamic_exponent_modulator * exponent_context_sensitivity)
        # 确保指数不会过小或过大，例如限制在 [0.1, 2.0] 之间
        adjusted_final_exponent = adjusted_final_exponent.clip(0.1, 2.0)
        _temp_debug_values["动态权重与情境调制"] = {
            "norm_market_sentiment": norm_market_sentiment,
            "norm_volatility_inverse": norm_volatility_inverse,
            "norm_trend_vitality": norm_trend_vitality,
            "norm_liquidity_authenticity": norm_liquidity_authenticity,
            "context_modulator_score_for_weights": context_modulator_score_for_weights,
            "context_modulator": context_modulator,
            "dynamic_quadrant_weights": dynamic_quadrant_weights,
            "dynamic_exponent_modulator": dynamic_exponent_modulator,
            "adjusted_final_exponent": adjusted_final_exponent
        }
        # 最终融合：加权平均
        final_score_raw = (
            score1 * dynamic_quadrant_weights["Q1_healthy_rally"] +
            score2 * dynamic_quadrant_weights["Q2_bearish_divergence"] +
            score3 * dynamic_quadrant_weights["Q3_panic_distribution"] +
            score4 * dynamic_quadrant_weights["Q4_selling_exhaustion"]
        )
        # 应用多层级共振因子和非线性指数
        final_score = final_score_raw * (1 + multi_level_resonance_factor * 0.5) # 共振因子放大
        final_score = np.sign(final_score) * (final_score.abs().pow(adjusted_final_exponent))
        final_score = final_score.clip(-1, 1).fillna(0.0)
        _temp_debug_values["最终融合分数"] = {
            "final_score_raw": final_score_raw,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        # if is_debug_enabled_for_method and probe_ts:
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
        #     for sig_name, series in _temp_debug_values["原始信号值"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF融合信号 ---"] = ""
        #     for key, series in _temp_debug_values["MTF融合信号"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 历史情境感知层 ---"] = ""
        #     for key, series in _temp_debug_values["历史情境感知层"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制器 ---"] = ""
        #     for key, series in _temp_debug_values["情境调制器"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 多层级共振引擎 ---"] = ""
        #     for key, series in _temp_debug_values["多层级共振引擎"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态阈值 ---"] = ""
        #     for key, series in _temp_debug_values["动态阈值"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 四象限分数 ---"] = ""
        #     for key, series in _temp_debug_values["四象限分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态权重与情境调制 ---"] = ""
        #     for key, series in _temp_debug_values["动态权重与情境调制"].items():
        #         if isinstance(series, pd.Series):
        #             val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #             debug_output[f"        {key}: {val:.4f}"] = ""
        #         else: # For dicts in dynamic_quadrant_weights
        #             debug_output[f"        {key}: {series}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合分数 ---"] = ""
        #     for key, series in _temp_debug_values["最终融合分数"].items():
        #         val = series.loc[probe_ts] if probe_ts in series.index else np.nan
        #         debug_output[f"        {key}: {val:.4f}"] = ""
        #     debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价量动态诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
        #     for key, value in debug_output.items():
        #         if value:
        #             print(f"{key}: {value}")
        #         else:
        #             print(key)
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算资金流吸筹拐点意图..."] = ""
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 前奏分 ---"] = ""
            for key, series in _temp_debug_values["前奏分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 强攻分 ---"] = ""
            for key, series in _temp_debug_values["强攻分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终审判 ---"] = ""
            for key, series in _temp_debug_values["最终审判"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流吸筹拐点意图诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算主力盈亏vs流量关系..."] = ""
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 派发压力分 ---"] = ""
            for key, series in _temp_debug_values["派发压力分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 建仓动力分 ---"] = ""
            for key, series in _temp_debug_values["建仓动力分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终分数 ---"] = ""
            for key, series in _temp_debug_values["最终分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力盈亏vs流量关系诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算个股板块协同共振..."] = ""
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看涨协同部分 ---"] = ""
            for key, series in _temp_debug_values["看涨协同部分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看跌协同部分 ---"] = ""
            for key, series in _temp_debug_values["看跌协同部分"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
            for key, series in _temp_debug_values["最终融合"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 个股板块协同共振诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算热门板块冷却..."] = ""
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, series in _temp_debug_values["原始信号值"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        '{key}': {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化处理 ---"] = ""
            for key, series in _temp_debug_values["归一化处理"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终分数 ---"] = ""
            for key, series in _temp_debug_values["最终分数"].items():
                val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                debug_output[f"        {key}: {val:.4f}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 热门板块冷却诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算价资关系..."] = ""
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 价资关系诊断完成，最终分值: {meta_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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

    def _calculate_storm_eye_calm(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        method_name = "_calculate_storm_eye_calm"
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
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算风暴眼中的寂静..."] = ""
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
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"    -> [过程情报警告] {method_name} 缺少核心信号，返回默认值。"] = ""
                self._print_debug_output(debug_output)
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
        goodness_of_fit_raw = self._get_safe_series(df, 'goodness_of_fit_score_D', np.nan, method_name=method_name)
        platform_conviction_raw = self._get_safe_series(df, 'platform_conviction_score_D', np.nan, method_name=method_name)
        # Modulators
        price_slope_raw = self._get_safe_series(df, f'SLOPE_{price_calmness_modulator_params.get("slope_period", 5)}_close_D', np.nan, method_name=method_name)
        pct_change_raw = self._get_safe_series(df, 'pct_change_D', np.nan, method_name=method_name)
        control_solidity_raw = self._get_safe_series(df, main_force_control_adjudicator_params.get('control_signal', 'control_solidity_index_D'), np.nan, method_name=method_name)
        mf_activity_ratio_raw = self._get_safe_series(df, main_force_control_adjudicator_params.get('activity_signal', 'main_force_activity_ratio_D'), np.nan, method_name=method_name)
        volatility_regime_raw = self._get_safe_series(df, regime_modulator_params.get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D'), np.nan, method_name=method_name)
        trend_regime_raw = self._get_safe_series(df, regime_modulator_params.get('trend_signal', 'ADX_14_D'), np.nan, method_name=method_name)
        _temp_debug_values["原始信号值"] = {
            "SCORE_STRUCT_AXIOM_TENSION": tension_score,
            "BBW_21_2.0_D": bbw_raw,
            "VOLATILITY_INSTABILITY_INDEX_21d_D": vol_instability_raw,
            "equilibrium_compression_index_D": equilibrium_compression_raw,
            "SCORE_DYN_AXIOM_STABILITY": dyn_stability_score,
            "SCORE_FOUNDATION_AXIOM_MARKET_TENSION": market_tension_score,
            "SAMPLE_ENTROPY_13d_D": price_sample_entropy_raw,
            "price_volume_entropy_D": price_volume_entropy_raw,
            "FRACTAL_DIMENSION_89d_D": price_fractal_dimension_raw,
            "bid_side_liquidity_D": bid_side_liquidity_raw,
            "ask_side_liquidity_D": ask_side_liquidity_raw,
            "vpin_score_D": vpin_score_raw,
            "BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D": bid_liquidity_sample_entropy_raw,
            "main_force_vwap_up_guidance_D": mf_vwap_up_guidance_raw,
            "main_force_vwap_down_guidance_D": mf_vwap_down_guidance_raw,
            "vwap_buy_control_strength_D": vwap_buy_control_raw,
            "vwap_sell_control_strength_D": vwap_sell_control_raw,
            "observed_large_order_size_avg_D": observed_large_order_size_avg_raw,
            "market_impact_cost_D": market_impact_cost_raw,
            "main_force_flow_directionality_D": main_force_flow_directionality_raw,
            "SCORE_FOUNDATION_AXIOM_LIQUIDITY_TIDE": liquidity_tide_score,
            "HURST_144d_D": hurst_raw,
            "turnover_rate_D": turnover_rate_raw,
            "volume_structure_skew_D": volume_structure_skew_raw,
            "volume_profile_entropy_D": volume_profile_entropy_raw,
            "deception_index_D": deception_index_raw,
            "wash_trade_intensity_D": wash_trade_intensity_raw,
            "deception_lure_long_intensity_D": deception_lure_long_raw,
            "deception_lure_short_intensity_D": deception_lure_short_raw,
            "covert_accumulation_signal_D": covert_accumulation_raw,
            "covert_distribution_signal_D": covert_distribution_raw,
            "main_force_slippage_index_D": main_force_slippage_raw,
            "main_force_flow_gini_D": main_force_flow_gini_raw,
            "price_reversion_velocity_D": price_reversion_velocity_raw,
            "structural_entropy_change_D": structural_entropy_change_raw,
            "micro_impact_elasticity_D": micro_impact_elasticity_raw,
            "order_flow_imbalance_score_D": order_flow_imbalance_score_raw,
            "liquidity_authenticity_score_D": liquidity_authenticity_score_raw,
            "mean_reversion_frequency_D": mean_reversion_frequency_raw,
            "trend_alignment_index_D": trend_alignment_index_raw,
            "SCORE_BEHAVIOR_VOLUME_ATROPHY": atrophy_score,
            "turnover_rate_f_D": turnover_rate_f_raw,
            "counterparty_exhaustion_index_D": counterparty_exhaustion_raw,
            "order_book_liquidity_supply_D": order_book_liquidity_raw,
            "buy_quote_exhaustion_rate_D": buy_quote_exhaustion_raw,
            "sell_quote_exhaustion_rate_D": sell_quote_exhaustion_raw,
            "SCORE_MICRO_STRATEGY_STEALTH_OPS": stealth_ops_score,
            "PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY": split_order_accum_score,
            "SCORE_FOUNDATION_AXIOM_SENTIMENT_PENDULUM": sentiment_pendulum_score,
            "market_sentiment_score_D": market_sentiment_raw,
            "retail_panic_surrender_index_D": retail_panic_raw,
            "retail_fomo_premium_index_D": retail_fomo_raw,
            "loser_pain_index_D": loser_pain_raw,
            "SCORE_STRUCT_BREAKOUT_READINESS": struct_breakout_readiness_score,
            "SCORE_STRUCT_PLATFORM_FOUNDATION": struct_platform_foundation_score,
            "SLOPE_5_close_D": price_slope_raw,
            "pct_change_D": pct_change_raw,
            "control_solidity_index_D": control_solidity_raw,
            "main_force_activity_ratio_D": mf_activity_ratio_raw,
            "ADX_14_D": trend_regime_raw
        }
        # --- 4. 计算MTF斜率/加速度分数 ---
        bbw_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'BBW_21_2.0_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        vol_instability_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        turnover_rate_slope_inverted_score = self._get_mtf_slope_accel_score(df, 'turnover_rate_f_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mf_net_flow_slope_positive = self._get_mtf_slope_accel_score(df, 'main_force_net_flow_calibrated_D', mtf_slope_accel_weights, df_index, method_name, ascending=True, bipolar=False)
        mtf_cohesion_score = self._get_mtf_cohesion_score(df, mtf_cohesion_base_signals, mtf_slope_accel_weights, df_index, method_name)
        _temp_debug_values["MTF斜率/加速度分数"] = {
            "bbw_slope_inverted_score": bbw_slope_inverted_score,
            "vol_instability_slope_inverted_score": vol_instability_slope_inverted_score,
            "turnover_rate_slope_inverted_score": turnover_rate_slope_inverted_score,
            "mf_net_flow_slope_positive": mf_net_flow_slope_positive,
            "mtf_cohesion_score": mtf_cohesion_score
        }
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
        _temp_debug_values["能量压缩"] = {
            "energy_compression_score": energy_compression_score
        }
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
        _temp_debug_values["量能枯竭"] = {
            "volume_exhaustion_score": volume_exhaustion_score
        }
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
        _temp_debug_values["主力隐蔽意图"] = {
            "stealth_ops": stealth_ops_normalized,
            "split_order_accum": split_order_accum_normalized,
            "mf_conviction_positive": mf_conviction_positive,
            "mf_net_flow_positive": mf_net_flow_positive,
            "mf_cost_advantage_positive": mf_cost_advantage_positive,
            "mf_buy_ofi_positive": mf_buy_ofi_positive,
            "mf_t0_buy_efficiency_positive": mf_t0_buy_efficiency_positive,
            "mf_net_flow_slope_positive": mf_net_flow_slope_positive,
            "order_book_imbalance_positive": order_book_imbalance_positive,
            "micro_price_impact_asymmetry_positive": micro_price_impact_asymmetry_positive,
            "mf_vwap_guidance_neutrality": mf_vwap_guidance_neutrality,
            "vwap_control_neutrality": vwap_control_neutrality,
            "observed_large_order_size_avg_inverted": observed_large_order_size_avg_inverted,
            "market_impact_cost_inverted": market_impact_cost_inverted,
            "main_force_net_flow_volatility_inverted": main_force_net_flow_volatility_inverted,
            "main_force_flow_ambiguity": main_force_flow_ambiguity
        }
        # 定义 main_force_covert_intent_scores_dict 局部变量
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
        _temp_debug_values["主力隐蔽意图融合"] = {
            "main_force_covert_intent_score": main_force_covert_intent_score
        }
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
        _temp_debug_values["市场情绪低迷融合"] = {
            "subdued_market_sentiment_score": subdued_market_sentiment_score
        }
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
        _temp_debug_values["突破准备度融合"] = {
            "breakout_readiness_score": breakout_readiness_score
        }
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
        _temp_debug_values["市场情境动态调节器"] = {
            "market_regime_modulator": market_regime_modulator
        }
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
        _temp_debug_values["最终融合"] = {
            "adjusted_final_fusion_weights": adjusted_final_fusion_weights,
            "base_calm_score": base_calm_score,
            "price_calmness_modulator": price_calmness_modulator,
            "price_calmness_amplifier": price_calmness_amplifier,
            "control_solidity_score": control_solidity_score,
            "mf_activity_ratio_score": mf_activity_ratio_score,
            "combined_control_score": combined_control_score,
            "main_force_amplifier": main_force_amplifier,
            "final_score": final_score
        }
        # --- 统一输出调试信息 ---
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for key, value in _temp_debug_values["原始信号值"].items():
                if isinstance(value, pd.Series):
                    val = value.loc[probe_ts] if probe_ts in value.index else np.nan
                    debug_output[f"        '{key}': {val:.4f}"] = ""
                else: # Handle non-Series values like dicts or raw numbers
                    debug_output[f"        '{key}': {value}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF斜率/加速度分数 ---"] = ""
            for key, series in _temp_debug_values["MTF斜率/加速度分数"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 能量压缩 ---"] = ""
            for key, series in _temp_debug_values["能量压缩"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 量能枯竭 ---"] = ""
            for key, series in _temp_debug_values["量能枯竭"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力隐蔽意图 ---"] = ""
            for key, series in _temp_debug_values["主力隐蔽意图"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 主力隐蔽意图融合 ---"] = ""
            for key, series in _temp_debug_values["主力隐蔽意图融合"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场情绪低迷融合 ---"] = ""
            for key, series in _temp_debug_values["市场情绪低迷融合"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 突破准备度融合 ---"] = ""
            for key, series in _temp_debug_values["突破准备度融合"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 市场情境动态调节器 ---"] = ""
            for key, series in _temp_debug_values["市场情境动态调节器"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合 ---"] = ""
            for key, series in _temp_debug_values["最终融合"].items():
                if isinstance(series, pd.Series):
                    val = series.loc[probe_ts] if probe_ts in series.index else np.nan
                    debug_output[f"        {key}: {val:.4f}"] = ""
                elif isinstance(series, dict): # Handle dicts within _temp_debug_values["最终融合"]
                    debug_output[f"        {key}:"] = ""
                    for sub_key, sub_value in series.items():
                        if isinstance(sub_value, pd.Series):
                            val = sub_value.loc[probe_ts] if probe_ts in sub_value.index else np.nan
                            debug_output[f"          {sub_key}: {val:.4f}"] = ""
                        else:
                            debug_output[f"          {sub_key}: {sub_value}"] = ""
                else:
                    debug_output[f"        {key}: {series}"] = ""
            debug_output[f"  -- [过程情报调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 风暴眼中的寂静诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            for key, value in debug_output.items():
                if value:
                    print(f"{key}: {value}")
                else:
                    print(key)
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

    def _calculate_process_covert_accumulation(self, df: pd.DataFrame, config: Dict) -> pd.Series:
        """
        【V2.5 · 深度情境与多维隐蔽行动版】计算“隐蔽吸筹”的专属信号。
        - 核心升级: 优化 `market_context_score` 中的价格弱势判断，直接奖励价格弱势。
        - 【强化】优化 `covert_action_score` 中的欺诈信号融合，更侧重于正向的诱多欺诈。
        - 【调整】调整 `covert_action_weights` 中拆单吸筹的权重，使用原始指标的MTF融合版本。
        """
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






