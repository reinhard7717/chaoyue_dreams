# 文件: strategies/trend_following/intelligence/process/process_intelligence_helper.py
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

class ProcessIntelligenceHelper:
    """
    【V1.0 · 过程情报辅助工具集】
    - 核心职责: 封装 ProcessIntelligence 类中多个方法共享的辅助逻辑，提高代码模块化和复用性。
    """
    def __init__(self, strategy_instance):
        """
        初始化 ProcessIntelligenceHelper。
        参数:
            strategy_instance: 策略实例，用于访问全局配置和原子状态。
        """
        self.strategy = strategy_instance
        # 直接从新文件加载 process_intelligence_params
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..', '..', '..'))
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

    def _get_safe_series(self, df: pd.DataFrame, col_name: str, default_value: float = np.nan, method_name: str = "") -> pd.Series:
        """
        安全地从DataFrame中获取Series。如果列不存在，则返回一个填充了np.nan的Series。
        如果列存在但包含NaN值，则这些NaN值将保留，除非在调用方明确处理。
        """
        if col_name not in df.columns:
            print(f"  [警告] {method_name}: 核心列 '{col_name}' 不存在，返回填充np.nan的Series。")
            return pd.Series(np.nan, index=df.index, dtype=np.float32)
        series = df[col_name].astype(np.float32)
        # 不再在此处填充NaN，让NaN值自然传播，以暴露问题
        return series

    def _get_mtf_slope_accel_score(self, df: pd.DataFrame, base_signal_name: str, mtf_weights_config: Dict, df_index: pd.Index, method_name: str, ascending: bool = True, bipolar: bool = False, periods: Optional[List[int]] = None) -> pd.Series:
        """
        【V1.2 · 周期过滤与统一归一化调用版】计算多时间框架斜率和加速度的融合分数。
        - 核心修正: 增加 `periods` 参数，允许指定要计算的特定周期，从而支持更灵活的MTF信号构建。
        - 核心修正: 统一调用 `_normalize_series` 进行归一化，利用其多时间框架加权能力。
        - 健壮性增强: 确保即使没有有效组件分数，也能返回一个填充了np.nan的Series，以暴露问题。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            base_signal_name (str): 基础信号的名称，例如 'close_D'。
            mtf_weights_config (Dict): 包含 'slope_periods' 和 'accel_periods' 权重的配置字典。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
            ascending (bool): 归一化时是否升序处理 (True表示值越大分数越高)。
            bipolar (bool): 归一化是否为双极性 (-1到1)。
            periods (Optional[List[int]]): 可选参数，如果提供，则只计算这些周期内的斜率和加速度。
                                          例如 [13] 表示只计算13周期的斜率和加速度。
        返回:
            pd.Series: 融合后的MTF斜率和加速度分数。
        """
        slope_periods_weights = get_param_value(mtf_weights_config.get('slope_periods'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.1})
        accel_periods_weights = get_param_value(mtf_weights_config.get('accel_periods'), {"5": 0.6, "13": 0.4})
        all_scores_components = []
        total_combined_weight = 0.0
        # 过滤周期
        filtered_slope_periods_weights = {p: w for p, w in slope_periods_weights.items() if periods is None or int(p) in periods}
        filtered_accel_periods_weights = {p: w for p, w in accel_periods_weights.items() if periods is None or int(p) in periods}
        # 处理斜率
        for period_str, weight in filtered_slope_periods_weights.items():
            try:
                period = int(period_str)
            except ValueError:
                continue
            slope_col = f'SLOPE_{period}_{base_signal_name}'
            slope_raw = self._get_safe_series(df, slope_col, np.nan, method_name=method_name)
            if slope_raw.isnull().all(): # 如果原始斜率数据全为NaN，则跳过此组件
                continue
            # 使用 _normalize_series 进行归一化
            norm_score = self._normalize_series(slope_raw, df_index, bipolar=bipolar, ascending=ascending)
            all_scores_components.append(norm_score * weight)
            total_combined_weight += weight
        # 处理加速度
        for period_str, weight in filtered_accel_periods_weights.items():
            try:
                period = int(period_str)
            except ValueError:
                continue
            accel_col = f'ACCEL_{period}_{base_signal_name}'
            accel_raw = self._get_safe_series(df, accel_col, np.nan, method_name=method_name)
            if accel_raw.isnull().all(): # 如果原始加速度数据全为NaN，则跳过此组件
                continue
            # 使用 _normalize_series 进行归一化
            norm_score = self._normalize_series(accel_raw, df_index, bipolar=bipolar, ascending=ascending)
            all_scores_components.append(norm_score * weight)
            total_combined_weight += weight
        if not all_scores_components or total_combined_weight == 0:
            # 如果没有任何有效组件分数，返回一个填充了np.nan的Series，以暴露问题
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        fused_score = sum(all_scores_components) / total_combined_weight
        # 根据 bipolar 参数进行裁剪
        return fused_score.clip(-1, 1) if bipolar else fused_score.clip(0, 1)

    def _get_mtf_resonance_score_from_config(self, df: pd.DataFrame, contextual_mtf_config: Dict, mtf_weights_config: Dict, df_index: pd.Index, method_name: str) -> pd.Series:
        """
        【V1.5 · 情境配置驱动版】计算多个信号在多时间框架上的共振分数。
        此方法将对多个基础信号计算其MTF斜率和加速度融合分数（双极性），
        然后评估这些融合分数之间的方向一致性和强度，生成一个双极性共振分数。
        - 核心逻辑:
            1. 遍历 contextual_mtf_config，为每个信号确定其 MTF 融合时的 bipolar 和 ascending 参数。
            2. 调用 _get_mtf_slope_accel_score 获取每个信号的双极性MTF斜率/加速度融合分数。
            3. 计算这些融合分数的平均值，代表整体方向和强度。
            4. 计算这些融合分数的标准差，代表离散度（不一致性）。
            5. 将离散度转换为一致性强度（1-归一化标准差）。
            6. 最终共振分数 = 整体方向和强度 * 一致性强度。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            contextual_mtf_config (Dict): 包含情境MTF信号配置的字典，用于获取 base_signal_name, bipolar, inverted_for_stability, inverted_for_decay。
            mtf_weights_config (Dict): 包含 'slope_periods' 和 'accel_periods' 权重的配置字典。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            pd.Series: 融合后的MTF共振分数 (范围 [-1, 1])。
        """
        all_fused_mtf_scores = {}
        for config_key, config_val in contextual_mtf_config.items():
            if isinstance(config_val, dict):
                base_signal_name = config_val.get('base_signal_name')
                if base_signal_name is None:
                    continue
                is_bipolar_mtf = config_val.get('bipolar', True)
                is_inverted_for_stability = config_val.get('inverted_for_stability', False)
                is_inverted_for_decay = config_val.get('inverted_for_decay', False)
                # 确定传递给 _get_mtf_slope_accel_score 的 ascending 参数
                # 如果是双极性信号且需要反转衰减，则 ascending=False (高值 -> 低分)
                # 如果是稳定性信号且需要反转，则 ascending=False (不稳定性高 -> 稳定性低)
                ascending_param_for_mtf = True
                if is_inverted_for_decay or is_inverted_for_stability:
                    ascending_param_for_mtf = False
                # 获取每个信号的双极性MTF斜率/加速度融合分数
                fused_score = self._get_mtf_slope_accel_score(
                    df,
                    base_signal_name,
                    mtf_weights_config,
                    df_index,
                    method_name,
                    bipolar=is_bipolar_mtf,
                    ascending=ascending_param_for_mtf
                )
                # 只有当分数不是全NaN或全0时才加入，避免影响共振计算
                if not fused_score.isnull().all() and not (fused_score == 0).all():
                    all_fused_mtf_scores[config_key] = fused_score # 使用 config_key 作为字典键，保持一致性
        if not all_fused_mtf_scores or len(all_fused_mtf_scores) < 2:
            print(f"    -> [过程情报警告] {method_name}: 计算MTF共振分数至少需要2个有效信号，当前只有 {len(all_fused_mtf_scores)} 个。返回np.nan。")
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        fused_scores_df = pd.DataFrame(all_fused_mtf_scores, index=df_index)
        # 计算每个时间点上（axis=1）不同信号的平均值 (代表整体方向和强度)
        mean_scores = fused_scores_df.mean(axis=1)
        # 计算每个时间点上（axis=1）不同信号之间的标准差 (离散度)
        std_scores = fused_scores_df.std(axis=1).fillna(np.nan) # 修正：fillna(np.nan)
        # 将标准差归一化到 [0, 1] 范围，并转换为一致性强度
        max_possible_std = fused_scores_df.max(axis=1) - fused_scores_df.min(axis=1)
        max_possible_std = max_possible_std.replace(0, 1)
        normalized_std = (std_scores / max_possible_std).clip(0, 1)
        consistency_strength = (1 - normalized_std).fillna(np.nan) # 修正：fillna(np.nan)
        # 最终共振分数 = 整体方向和强度 * 一致性强度
        resonance_score = mean_scores * consistency_strength
        return resonance_score.clip(-1, 1).astype(np.float32)

    def _get_mtf_cohesion_score(self, df: pd.DataFrame, base_signal_names: List[str], mtf_weights_config: Dict, df_index: pd.Index, method_name: str) -> pd.Series:
        """
        【V1.3 · 协同性双极化增强版】计算多时间框架信号的协同性分数。
        此方法将对多个基础信号计算其MTF斜率和加速度融合分数，然后评估这些融合分数之间的离散度，
        离散度越低（即越协同），分数越高。
        - 核心增强: 协同性分数现在是双极性的 (-1到1)。当多个信号在多时间框架上协同向上时，分数趋近于1；
                    协同向下时，分数趋近于-1；不协同或无明显方向时，分数趋近于0。
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
            # 调用 _get_mtf_slope_accel_score 来获取每个信号的融合MTF分数
            # 协同性分数需要反映方向，所以这里 bipolar=True
            fused_score = self._get_mtf_slope_accel_score(df, base_signal_name, mtf_weights_config, df_index, method_name, ascending=True, bipolar=True)
            all_fused_mtf_scores[base_signal_name] = fused_score
        if not all_fused_mtf_scores:
            # 如果没有任何有效的MTF分数，返回一个填充了np.nan的Series，以暴露问题
            return pd.Series(np.nan, index=df_index, dtype=np.float32)
        # 将所有融合分数转换为DataFrame
        fused_scores_df = pd.DataFrame(all_fused_mtf_scores, index=df_index)
        # 计算每个时间点上（axis=1）不同信号之间的标准差 (离散度)
        # 标准差越小，协同性越高
        min_periods_std = max(1, int(self.meta_window * 0.5))
        instant_std = fused_scores_df.std(axis=1)
        # 对标准差进行平滑处理
        smoothed_std = instant_std.rolling(window=self.meta_window, min_periods=min_periods_std).mean().fillna(np.nan) # 修正：fillna(np.nan)
        # 计算每个时间点上，不同信号的平均值 (代表整体方向)
        instant_mean = fused_scores_df.mean(axis=1)
        # 将标准差转换为协同性强度 (0到1，标准差越小，强度越大)
        # 避免除以零，并处理全为零的情况
        max_std = smoothed_std.max()
        cohesion_magnitude = pd.Series(np.nan, index=df_index, dtype=np.float32) # 修正：默认值改为np.nan
        if max_std > 0:
            cohesion_magnitude = (1 - (smoothed_std / max_std)).clip(0, 1)
        # 结合方向和强度，生成双极性协同性分数
        # 使用 np.sign() 获取方向，并乘以强度
        bipolar_cohesion_score = cohesion_magnitude * np.sign(instant_mean)
        # 最终归一化到 -1 到 1 范围
        return self._normalize_series(bipolar_cohesion_score, df_index, bipolar=True)

    def _normalize_series(self, series: pd.Series, target_index: pd.Index, bipolar: bool = False, ascending: bool = True) -> pd.Series:
        """
        【V1.1 · 统一归一化引擎】
        - 核心职责: 为类内部提供一个统一的、基于多时间框架自适应归一化的方法。
        - 核心逻辑: 根据 bipolar 参数，调用 get_adaptive_mtf_normalized_score (单极) 或
                     get_adaptive_mtf_normalized_bipolar_score (双极) 进行归一化。
        """
        # 获取MTF权重配置
        p_conf_structural_ultimate = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
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
                ascending=ascending, # 确保 ascending 参数被传递
                tf_weights=actual_mtf_weights
            )

    def _get_atomic_score(self, df: pd.DataFrame, score_name: str, default_value: float = np.nan) -> pd.Series:
        """
        【V1.0 · 原子信号访问器】
        - 核心职责: 提供一个标准的、安全的方法来从 self.strategy.atomic_states 中获取预先计算好的原子信号。
        - 核心逻辑: 尝试从 atomic_states 字典中获取指定的信号 Series。如果不存在，则打印警告并
                     返回一个与 df 索引对齐的、填充了np.nan的Series，以暴露问题。
        """
        score_series = self.strategy.atomic_states.get(score_name)
        if score_series is None:
            print(f"    -> [过程情报警告] 依赖的原子信号 '{score_name}' 不存在，返回填充np.nan的Series。")
            return pd.Series(np.nan, index=df.index)
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

    def _calculate_slope_series(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算给定Series的斜率。
        参数:
            series (pd.Series): 输入Series。
            period (int): 计算斜率的周期。
        返回:
            pd.Series: 斜率Series。
        """
        if period <= 1:
            return pd.Series(0.0, index=series.index, dtype=np.float32)
        # 使用线性回归计算斜率
        # 这里的实现可以简化为 diff(period) / period，或者更复杂的线性回归
        # 为了保持一致性，我们假设斜率是预计算的，或者这里使用一个简单的近似
        # 考虑到我们已经有SLOPE_X_signal_D，这里可以模拟其行为
        # 简单实现：(当前值 - period周期前的值) / period
        return (series - series.shift(period)).fillna(0.0) / period

    def _calculate_accel_series(self, series: pd.Series, period: int) -> pd.Series:
        """
        计算给定Series的加速度。
        加速度是斜率的斜率。
        参数:
            series (pd.Series): 输入Series。
            period (int): 计算加速度的周期。
        返回:
            pd.Series: 加速度Series。
        """
        if period <= 2: # 加速度至少需要3个点 (2个斜率)
            return pd.Series(0.0, index=series.index, dtype=np.float32)
        # 计算period周期的斜率
        slope = self._calculate_slope_series(series, period)
        # 计算斜率的period周期斜率作为加速度
        return self._calculate_slope_series(slope, period)

    def _get_mtf_resonance_score(self, df: pd.DataFrame, base_signal_names: List[str], mtf_weights_config: Dict, df_index: pd.Index, method_name: str) -> pd.Series:
        """
        【V1.4 · 新增】计算多个信号在多时间框架上的共振分数。
        该方法将对多个基础信号计算其MTF斜率和加速度融合分数（双极性），
        然后评估这些融合分数之间的方向一致性和强度，生成一个双极性共振分数。
        - 核心逻辑:
            1. 获取每个基础信号的双极性MTF斜率/加速度融合分数。
            2. 计算这些融合分数的平均值，代表整体方向和强度。
            3. 计算这些融合分数的标准差，代表离散度（不一致性）。
            4. 将离散度转换为一致性强度（1-归一化标准差）。
            5. 最终共振分数 = 整体方向和强度 * 一致性强度。
        参数:
            df (pd.DataFrame): 包含所有原始数据的DataFrame。
            base_signal_names (List[str]): 基础信号名称列表。
            mtf_weights_config (Dict): 包含 'slope_periods' 和 'accel_periods' 权重的配置字典。
            df_index (pd.Index): DataFrame的索引。
            method_name (str): 调用此方法的名称，用于日志输出。
        返回:
            pd.Series: 融合后的MTF共振分数 (范围 [-1, 1])。
        """
        if not base_signal_names or len(base_signal_names) < 2:
            # 至少需要两个信号才能计算共振
            print(f"    -> [过程情报警告] {method_name}: 计算MTF共振分数至少需要2个信号，当前提供 {len(base_signal_names)} 个。返回0.0。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        all_fused_mtf_scores = {}
        for base_signal_name in base_signal_names:
            # 获取每个信号的双极性MTF斜率/加速度融合分数
            fused_score = self._get_mtf_slope_accel_score(df, base_signal_name, mtf_weights_config, df_index, method_name, bipolar=True)
            all_fused_mtf_scores[base_signal_name] = fused_score
        # 将所有融合分数转换为DataFrame
        fused_scores_df = pd.DataFrame(all_fused_mtf_scores, index=df_index)
        # 计算每个时间点上（axis=1）不同信号的平均值 (代表整体方向和强度)
        mean_scores = fused_scores_df.mean(axis=1)
        # 计算每个时间点上（axis=1）不同信号之间的标准差 (离散度)
        # 标准差越小，一致性越高
        std_scores = fused_scores_df.std(axis=1).fillna(0.0)
        # 将标准差归一化到 [0, 1] 范围，并转换为一致性强度
        # 理论上，对于 [-1, 1] 范围的N个信号，最大标准差发生在 N/2 个 -1 和 N/2 个 1 的情况下
        # 例如，对于2个信号，最大标准差是 std([-1, 1]) = 1
        # 对于3个信号，std([-1, -1, 1]) = sqrt(8/9)约0.94，std([-1, 0, 1]) = 1
        # 简单起见，我们可以将标准差裁剪到 [0, 1] 范围，并用 1 减去它来表示一致性
        max_possible_std = fused_scores_df.max(axis=1) - fused_scores_df.min(axis=1) # 实际最大可能范围
        max_possible_std = max_possible_std.replace(0, 1) # 避免除以0
        normalized_std = (std_scores / max_possible_std).clip(0, 1)
        consistency_strength = (1 - normalized_std).fillna(0.0)
        # 最终共振分数 = 整体方向和强度 * 一致性强度
        # 结果范围 [-1, 1]
        resonance_score = mean_scores * consistency_strength
        return resonance_score.clip(-1, 1).astype(np.float32)


