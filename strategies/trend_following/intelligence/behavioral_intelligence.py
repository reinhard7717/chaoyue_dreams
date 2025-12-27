# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Tuple, Optional, List, Any, final
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_score, get_adaptive_mtf_normalized_energy_score,
    is_limit_up, get_adaptive_mtf_normalized_bipolar_score, get_robust_bipolar_normalized_score,
    normalize_score,  load_external_json_config, normalize_to_bipolar
)

class BehavioralIntelligence:
    """
    【V28.0 · 结构升维版】
    - 核心升级: 废弃了旧的 _calculate_price_health, _calculate_volume_health, _calculate_kline_pattern_health 方法。
                所有健康度计算已统一由全新的 _calculate_structural_behavior_health 引擎负责。
    """
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.pattern_recognizer = strategy_instance.pattern_recognizer
        # 加载外部配置文件
        self.config_params = load_external_json_config('config/intelligence/behavioral.json').get('behavioral_divergence_params', {})
        if not self.config_params:
            print("    -> [行为情报初始化警告] 未能加载 behavioral.json 或 behavioral_divergence_params 为空。")
        # 调试参数将在 run_behavioral_analysis_command 中动态设置和传递
        self.is_debug_enabled = False
        self.probe_ts = None

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法", is_debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 警告：缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str, is_debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0, is_debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        【V1.0】安全地从原子状态库或主数据帧中获取分数。
        - 核心职责: 统一信号获取路径，优先从 self.strategy.atomic_states 获取，
                      若无则从主数据帧 df 获取，最后提供默认值，确保数据流的稳定性。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in df.columns:
            return df[name]
        else:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - _get_atomic_score] 警告：信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=df.index)

    def _get_signal(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0, is_debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        【V1.0】信号获取哨兵方法
        - 核心职责: 安全地从DataFrame获取信号。
        - 预警机制: 如果信号不存在，打印明确的警告信息，并返回一个包含默认值的Series，以防止程序崩溃。
        """
        if signal_name not in df.columns:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - _get_signal] 依赖信号 '{signal_name}' 在数据帧中不存在，将使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[signal_name]

    def run_behavioral_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V30.2 · 依赖感知型背离品质重构版】行为情报模块总指挥
        - 核心重构: 修复了因信号生成顺序导致的依赖缺失问题。
                    现在，所有依赖信号（特别是 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT）
                    都会在被使用之前生成并合并到主数据帧中，确保情报流的完整性。
        - 【修正】确保df在整个调用链中正确更新。
        - 【清理】移除用于检查 breakout_quality_score_D 原始输入值的临时探针。
        - 【新增】根据用户要求，启用详细调试探针，输出原始数据、关键计算节点和结果。
        """
        all_behavioral_states = {}
        # --- 调试探针设置 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        is_debug_enabled = get_param_value(debug_params.get('enabled'), False)
        probe_dates_list = get_param_value(debug_params.get('probe_dates'), [])
        probe_ts = None
        if is_debug_enabled and probe_dates_list:
            # Convert probe_dates_list to a set of datetime objects for efficient lookup
            probe_dates_set = {pd.to_datetime(d).date() for d in probe_dates_list}
            # Find the latest probe_date that is present in the DataFrame index
            for date_idx in reversed(df.index):
                if date_idx.date() in probe_dates_set:
                    probe_ts = date_idx
                    break
        # Pass this debug_info_tuple to all subsequent methods
        # debug_info_tuple = (is_debug_enabled, probe_ts, "run_behavioral_analysis_command") # Not needed, pass flags directly
        # 调整执行顺序：首先生成 SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT
        # 因为它是 _diagnose_divergence_quality 的依赖项，而 _diagnose_divergence_quality
        # 在 _diagnose_behavioral_axioms 内部被调用。
        micro_intent_signals = self._diagnose_microstructure_intent(df, is_debug_enabled, probe_ts)
        self.strategy.atomic_states.update(micro_intent_signals)
        all_behavioral_states.update(micro_intent_signals)
        # 立即将微观意图信号合并到df中，供后续方法使用 _get_safe_series 获取
        for k, v in micro_intent_signals.items():
            if k not in df.columns:
                df[k] = v
        # --- 移除临时探针：检查 breakout_quality_score_D 的原始值 ---
        # 接着生成核心公理信号，此时 _diagnose_divergence_quality 可以访问到微观意图信号
        atomic_signals = self._diagnose_behavioral_axioms(df, is_debug_enabled, probe_ts)
        # 如果核心公理诊断失败，则提前返回，防止后续错误
        if not atomic_signals:
            # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            #     print(f"      [探针 - run_behavioral_analysis_command] 核心公理诊断失败，行为分析中止。")
            return {}
        self.strategy.atomic_states.update(atomic_signals)
        all_behavioral_states.update(atomic_signals)
        # 将原子信号合并到df中，供后续的 _calculate_signal_dynamics 方法使用
        for k, v in atomic_signals.items():
            if k not in df.columns:
                df[k] = v
        # 生成上下文新高强度信号
        context_new_high_strength = self._diagnose_context_new_high_strength(df, is_debug_enabled, probe_ts)
        self.strategy.atomic_states.update(context_new_high_strength)
        # 修正NameError: context_new_high_high_strength -> context_new_high_strength
        all_behavioral_states.update(context_new_high_strength) 
        # 将上下文信号合并到df中
        for k, v in context_new_high_strength.items():
            if k not in df.columns:
                df[k] = v
        # 将_calculate_signal_dynamics返回的新的DataFrame重新赋值给df
        df = self._calculate_signal_dynamics(df, is_debug_enabled, probe_ts) 
        dynamic_cols = [c for c in df.columns if c.startswith(('MOMENTUM_', 'POTENTIAL_', 'THRUST_', 'RESONANCE_'))] # 从更新后的df中获取列
        self.strategy.atomic_states.update(df[dynamic_cols])
        all_behavioral_states.update(df[dynamic_cols])
        return all_behavioral_states

    def _generate_all_atomic_signals(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Dict[str, pd.Series]:
        """
        【V3.0 · 职责净化版】原子信号中心
        - 核心升级: 遵循“三层金字塔”架构，本方法不再计算跨领域的“趋势健康度”和“绝望度”。
                      这些高级融合逻辑已迁移至 FusionIntelligence。
                      新增对纯净版“行为K线质量分”的计算和发布。
        """
        atomic_signals = {}
        atomic_signals.update(self._diagnose_behavioral_axioms(df, is_debug_enabled, probe_ts))
        day_quality_score = self._calculate_behavioral_day_quality(df, is_debug_enabled, probe_ts)
        atomic_signals['BIPOLAR_BEHAVIORAL_DAY_QUALITY'] = day_quality_score
        battlefield_momentum = day_quality_score.ewm(span=5, adjust=False).mean()
        atomic_signals['SCORE_BEHAVIORAL_BATTLEFIELD_MOMENTUM'] = battlefield_momentum.astype(np.float32)
        self.strategy.atomic_states.update(atomic_signals)
        atomic_signals.update(self._diagnose_upper_shadow_intent(df, is_debug_enabled, probe_ts))
        return atomic_signals

    def _get_mtf_fused_indicator_score(self, df: pd.DataFrame, base_name: str, mtf_slope_accel_weights: Dict, mtf_indicator_component_weights: Dict, is_negative_indicator: bool = False, ascending: bool = True, debug_info: Optional[Tuple[bool, pd.Timestamp, str]] = None) -> pd.Series:
        method_name = debug_info[2] if debug_info else "Unknown"
        component_weights = mtf_indicator_component_weights.get(base_name, mtf_indicator_component_weights.get('default', {"raw_weight": 0.4, "slope_weight": 0.3, "accel_weight": 0.3}))
        raw_w = component_weights.get('raw_weight', 0.4)
        slope_w = component_weights.get('slope_weight', 0.3)
        accel_w = component_weights.get('accel_weight', 0.3)
        total_component_weight = raw_w + slope_w + accel_w
        if total_component_weight == 0:
            total_component_weight = 1.0
        scores = []
        total_mtf_weight = sum(mtf_slope_accel_weights.values())
        if total_mtf_weight == 0:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        raw_val = self._get_safe_series(df, f'{base_name}_D', 0.0, method_name=method_name)
        # 收集所有需要归一化的窗口
        windows_to_normalize = [int(p_str) * 2 for p_str in mtf_slope_accel_weights.keys() if mtf_slope_accel_weights.get(p_str, 0) > 0]
        windows_to_normalize = list(set(windows_to_normalize)) # 去重
        # 一次性计算所有窗口的归一化分数
        if is_negative_indicator:
            norm_raw_scores_df = normalize_score(raw_val.abs(), df.index, windows=windows_to_normalize, ascending=True)
        else:
            norm_raw_scores_df = normalize_score(raw_val, df.index, windows=windows_to_normalize, ascending=ascending)
        for period_str, weight in mtf_slope_accel_weights.items():
            period = int(period_str)
            if weight == 0:
                continue
            slope_col = f'SLOPE_{period}_{base_name}_D'
            slope_val = self._get_safe_series(df, slope_col, 0.0, method_name=method_name)
            accel_col = f'ACCEL_{period}_{base_name}_D'
            accel_val = self._get_safe_series(df, accel_col, 0.0, method_name=method_name)
            norm_window = period * 2 # 归一化窗口通常取周期的两倍
            # 从一次性计算的结果中获取对应窗口的分数
            norm_raw = norm_raw_scores_df[norm_window] if norm_window in norm_raw_scores_df.columns else pd.Series(0.0, index=df.index)
            if is_negative_indicator:
                norm_slope = normalize_score(slope_val.clip(upper=0).abs(), df.index, windows=norm_window, ascending=True)
                norm_accel = normalize_score(accel_val.clip(upper=0).abs(), df.index, windows=norm_window, ascending=True)
            else:
                norm_slope = normalize_score(slope_val, df.index, windows=norm_window, ascending=ascending)
                norm_accel = normalize_score(accel_val, df.index, windows=norm_window, ascending=ascending)
            fused_period_score = (
                (norm_raw + 1e-9).pow(raw_w) *
                (norm_slope + 1e-9).pow(slope_w) *
                (norm_accel + 1e-9).pow(accel_w)
            ).pow(1 / total_component_weight).fillna(0.0).clip(0,1)
            scores.append(fused_period_score * weight)
        if not scores:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        fused_score = sum(scores) / total_mtf_weight
        return fused_score.astype(np.float32)

    def _calculate_series_dynamics(self, series: pd.Series, periods: List[int], df_index: pd.Index, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp], base_name: str) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]:
        slopes = {}
        accels = {}
        series = series.astype(np.float32)
        if len(series) < max(periods) + 1:
            for p in periods:
                slopes[p] = pd.Series(0.0, index=df_index, dtype=np.float32)
                if p <= 34:
                    accels[p] = pd.Series(0.0, index=df_index, dtype=np.float32)
            return slopes, accels
        for p in periods:
            slope_series = series.diff(p).fillna(0.0).astype(np.float32)
            slopes[p] = slope_series
            if p <= 34:
                accel_series = slope_series.diff(1).fillna(0.0).astype(np.float32)
                accels[p] = accel_series
        return slopes, accels

    def _calculate_mtf_dynamic_score(self, df: pd.DataFrame, base_indicator_name: str, mtf_periods: List[int], mtf_weights: Dict[str, float], dynamic_type: str, is_positive_trend: bool, method_name: str, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V1.0 · 新增】计算多时间框架动态指标分数（斜率或加速度）。
        - 核心职责: 根据给定的基础指标名称、时间框架周期和权重，计算其在不同时间框架下的斜率或加速度，并进行加权融合。
        - 参数:
            - df: 包含指标数据的DataFrame。
            - base_indicator_name: 基础指标的名称（例如 'RSI_13', 'MACDh_13_34_8'）。
            - mtf_periods: 多时间框架周期列表（例如 [5, 13, 21]）。
            - mtf_weights: 各时间框架的权重字典（例如 {'5': 0.4, '13': 0.3, '21': 0.2}）。
            - dynamic_type: 'SLOPE' 或 'ACCEL'，表示计算斜率或加速度。
            - is_positive_trend: 如果为True，则只考虑正向趋势（斜率>0或加速度>0），负向趋势归零；否则相反。
            - method_name: 调用此方法的父方法名称，用于调试信息。
            - is_debug_enabled: 是否启用调试模式。
            - probe_ts: 调试的目标时间戳。
        - 返回: 融合后的多时间框架动态分数。
        """
        scores_list = []
        total_weight = 0.0
        debug_info = (is_debug_enabled, probe_ts, method_name)
        for p in mtf_periods:
            col_name = f"{dynamic_type}_{p}_{base_indicator_name}_D"
            if col_name in df.columns:
                raw_series = df[col_name]
                
                # 根据 is_positive_trend 过滤
                if is_positive_trend:
                    filtered_series = raw_series.clip(lower=0)
                else:
                    filtered_series = raw_series.clip(upper=0).abs() # 负向趋势取绝对值
                
                # 归一化
                norm_score = normalize_score(filtered_series, df.index, windows=p * 2, ascending=True, debug_info=False)
                
                weight = mtf_weights.get(str(p), 0.0)
                if weight > 0:
                    scores_list.append(norm_score * weight)
                    total_weight += weight
                
                if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                    print(f"        [探针 - {method_name}] {dynamic_type} {base_indicator_name} {p}d @ {probe_ts.strftime('%Y-%m-%d')}:")
                    print(f"          - 原始值: {raw_series.loc[probe_ts]:.4f}")
                    print(f"          - 过滤后值: {filtered_series.loc[probe_ts]:.4f}")
                    print(f"          - 归一化分数: {norm_score.loc[probe_ts]:.4f}")
                    print(f"          - 权重: {weight:.4f}")
        if not scores_list or total_weight == 0:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        fused_score = sum(scores_list) / total_weight
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"        [探针 - {method_name}] 融合 {dynamic_type} {base_indicator_name} 最终分数 @ {probe_ts.strftime('%Y-%m-%d')}: {fused_score.loc[probe_ts]:.4f}")
        return fused_score.clip(0, 1).astype(np.float32)

    def _calculate_signal_dynamics(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.DataFrame:
        """
        【V4.3 · 上涨衰竭动态增强与多时间维度归一化版】信号动态计算引擎
        - 核心错误修复: 彻底剥离了对其他情报层终极共振信号的依赖，解决了因执行时序错乱导致的信号获取失败问题。
        - 核心逻辑重构: 遵循“职责分离”原则，本方法现在只聚焦于为【本模块生产的】纯粹行为原子信号注入动态因子（动量、潜力、推力）。
                        不再计算跨领域的 RESONANCE_HEALTH_D 等信号。
        - 【修改】移除对 `SCORE_BEHAVIOR_RISK_UPPER_SHADOW_PRESSURE` 的动态增强。
        - 【优化】将 `momentum`, `potential`, `thrust` 的归一化方式改为多时间维度自适应归一化。
        - 【新增】将 SCORE_RISK_UNRESOLVED_PRESSURE 和 SCORE_OPPORTUNITY_PRESSURE_ABSORPTION 加入动态因子计算。
        - 【新增】在调试模式下，打印增强后的信号值。
        """
        method_name = "_calculate_signal_dynamics"
        p_dyn = get_param_value(self.config_params.get('signal_dynamics_params'), {})
        momentum_span = get_param_value(p_dyn.get('momentum_span'), 5)
        potential_window = get_param_value(p_dyn.get('potential_window'), 120)
        dynamics_df = pd.DataFrame(index=df.index)
        atomic_signals_to_enhance = [
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM',
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
            'SCORE_BEHAVIOR_VOLUME_BURST',
            'SCORE_BEHAVIOR_VOLUME_ATROPHY',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'SCORE_OPPORTUNITY_LOCKUP_RALLY',
            'SCORE_OPPORTUNITY_SELLING_EXHAUSTION',
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW', # 新增上涨衰竭原始分的动态增强
            'SCORE_RISK_LIQUIDITY_DRAIN',
            'SCORE_RISK_UNRESOLVED_PRESSURE', # 新增未解决压力风险
            'SCORE_OPPORTUNITY_PRESSURE_ABSORPTION' # 新增压力吸收机会
        ]
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        debug_info = (is_debug_enabled, probe_ts, method_name)
        for signal_name in atomic_signals_to_enhance:
            if signal_name in self.strategy.atomic_states:
                signal_series = self.strategy.atomic_states[signal_name]
                signal_series = signal_series.astype(np.float32)
                
                momentum = signal_series.diff(momentum_span).fillna(0.0).astype(np.float32)
                norm_momentum = get_adaptive_mtf_normalized_score(momentum, df.index, ascending=True, tf_weights=default_weights, debug_info=False)
                dynamics_df[f'MOMENTUM_{signal_name}'] = norm_momentum
                
                potential = signal_series.rolling(window=potential_window).mean().fillna(signal_series).astype(np.float32)
                norm_potential = get_adaptive_mtf_normalized_score(potential, df.index, ascending=True, tf_weights=default_weights, debug_info=False)
                dynamics_df[f'POTENTIAL_{signal_name}'] = norm_potential
                
                thrust = momentum.diff(1).fillna(0.0).astype(np.float32)
                norm_thrust = get_adaptive_mtf_normalized_score(thrust, df.index, ascending=True, tf_weights=default_weights, debug_info=False)
                dynamics_df[f'THRUST_{signal_name}'] = norm_thrust
                if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                    print(f"      [探针 - {method_name}] 增强信号 '{signal_name}' @ {probe_ts.strftime('%Y-%m-%d')}:")
                    print(f"        - 原始值: {signal_series.loc[probe_ts]:.4f}")
                    print(f"        - 动量 (MOMENTUM_{signal_name}): {norm_momentum.loc[probe_ts]:.4f}")
                    print(f"        - 潜力 (POTENTIAL_{signal_name}): {norm_potential.loc[probe_ts]:.4f}")
                    print(f"        - 推力 (THRUST_{signal_name}): {norm_thrust.loc[probe_ts]:.4f}")
            else:
                if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                    print(f"      [探针 - {method_name}] 警告：信号 '{signal_name}' 在原子状态库中不存在，跳过动态因子计算。")
        final_df = pd.concat([df, dynamics_df], axis=1)
        return final_df

    def _robust_generalized_mean(self, scores_dict: Dict[str, pd.Series], weights_dict: Dict[str, Any], df_index: pd.Index, power_p: Any = 0.0, is_debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None, fusion_level_name: str = "未知融合层") -> pd.Series:
        """
        【V5.1 · 广义平均融合器 - 动态幂参数版】
        计算给定分数序列的加权广义平均 (Weighted Generalized Mean / Power Mean)。
        - 当 power_p 接近 0 时，行为类似于加权几何平均。
        - 当 power_p 为 1 时，行为类似于加权算术平均。
        - 确保输入分数在 [0, 1] 范围内，并处理零值以避免数学错误。
        - 新增: power_p 参数现在可以是标量或 pd.Series，以支持动态调整融合幂。
        - 修复: 兼容 weights_dict 中的权重可以是标量或 pd.Series。
        - 【新增】在调试模式下，打印融合过程中的关键信息。
        """
        if not scores_dict:
            if is_debug_enabled and probe_ts and not df_index.empty and probe_ts == df_index[-1]:
                print(f"      [探针 - _robust_generalized_mean] {fusion_level_name}：scores_dict 为空，返回0分。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 准备分数 DataFrame，确保所有分数都与 df_index 对齐，填充 NaN，并裁剪到 [0, 1]
        scores_df = pd.DataFrame(index=df_index)
        for name, score_series in scores_dict.items():
            scores_df[name] = score_series.reindex(df_index).fillna(0.0).clip(0, 1).astype(np.float32)
        # 准备权重 DataFrame，处理标量和 Series 权重，并确保与 df_index 对齐
        dynamic_weights_df = pd.DataFrame(index=df_index)
        for name in scores_df.columns:
            weight_val = weights_dict.get(name, 0.0)
            if isinstance(weight_val, pd.Series):
                dynamic_weights_df[name] = weight_val.reindex(df_index).fillna(0.0).astype(np.float32)
            else:
                dynamic_weights_df[name] = pd.Series(weight_val, index=df_index, dtype=np.float32)
        # 过滤掉权重为零或负数的项（向量化操作）
        # 使用一个小的 epsilon 来判断权重是否有效，避免浮点数比较问题
        valid_weight_mask = (dynamic_weights_df > 1e-9)
        # 将无效权重的分数和权重本身设为0，使其不参与后续计算
        filtered_scores_df = scores_df.where(valid_weight_mask, 0.0)
        filtered_weights_df = dynamic_weights_df.where(valid_weight_mask, 0.0)
        # 计算每个时间点的总权重
        total_weight = filtered_weights_df.sum(axis=1)
        # 处理总权重为零的情况：这些时间点的最终分数应为0
        zero_total_weight_mask = (total_weight.abs() < 1e-9)
        # 归一化权重：对于总权重为零的行，归一化权重设为0，避免除以零
        normalized_weights_df = filtered_weights_df.div(total_weight.where(~zero_total_weight_mask, 1.0), axis=0)
        normalized_weights_df = normalized_weights_df.where(~zero_total_weight_mask, 0.0) # 确保总权重为0的行，归一化权重也为0
        # 确保分数略大于零，以避免对数和幂运算中的数学错误
        safe_scores_df = filtered_scores_df.replace(0.0, 1e-9)
        # 初始化结果 Series
        result_values = np.zeros(len(df_index), dtype=np.float32)
        # 确保 power_p 是一个与 df_index 对齐的 Series
        if isinstance(power_p, pd.Series):
            power_p_aligned = power_p.reindex(df_index).fillna(0.0).astype(np.float32)
        else:
            power_p_aligned = pd.Series(power_p, index=df_index, dtype=np.float32)
        epsilon = 1e-9 # 用于判断 power_p 是否接近0
        # 几何平均部分 (power_p 接近 0)
        # 仅对总权重不为零且 power_p 接近 0 的行进行计算
        geometric_mean_mask = (power_p_aligned.abs() < epsilon) & (~zero_total_weight_mask)
        if geometric_mean_mask.any():
            # 元素级乘法：log(safe_scores_df) * normalized_weights_df，然后按行求和
            weighted_log_sum = (np.log(safe_scores_df) * normalized_weights_df).sum(axis=1)
            result_values[geometric_mean_mask.values] = np.exp(weighted_log_sum[geometric_mean_mask].values)
        # 广义平均部分 (power_p 不接近 0)
        # 仅对总权重不为零且 power_p 不接近 0 的行进行计算
        power_mean_mask = (~geometric_mean_mask) & (~zero_total_weight_mask)
        if power_mean_mask.any():
            # 避免 power_p 恰好为 0 导致 0**0 或 1/0 错误
            power_p_for_power_mean = power_p_aligned.copy()
            power_p_for_power_mean[power_mean_mask & (power_p_for_power_mean.abs() < epsilon)] = epsilon * np.sign(power_p_for_power_mean[power_mean_mask & (power_p_for_power_mean.abs() < epsilon)])
            # 元素级乘法：safe_scores_df.pow(power_p_for_power_mean, axis=0) * normalized_weights_df，然后按行求和
            # axis=0 确保 Series power_p_for_power_mean 与 DataFrame safe_scores_df 的行对齐
            weighted_power_sum = (safe_scores_df.pow(power_p_for_power_mean, axis=0) * normalized_weights_df).sum(axis=1)
            # 处理 weighted_power_sum 接近 0 的情况，避免 0 的负数幂或 0 的 1/0 幂
            temp_result = np.where(
                weighted_power_sum[power_mean_mask].values < epsilon,
                0.0,
                weighted_power_sum[power_mean_mask].values**(1/power_p_for_power_mean[power_mean_mask].values)
            )
            result_values[power_mean_mask.values] = temp_result
        # 对于总权重为零的行，最终结果应为 0.0
        result_values[zero_total_weight_mask.values] = 0.0
        final_fused_score = pd.Series(result_values, index=df_index, dtype=np.float32).clip(0, 1)
        if is_debug_enabled and probe_ts and not df_index.empty and probe_ts == df_index[-1]:
            print(f"        [探针 - _robust_generalized_mean] {fusion_level_name} 融合详情 @ {probe_ts.strftime('%Y-%m-%d')}:")
            for name in scores_dict.keys():
                if name in filtered_scores_df.columns:
                    print(f"          - {name}: 原始分数={scores_df[name].loc[probe_ts]:.4f}, 归一化权重={normalized_weights_df[name].loc[probe_ts]:.4f}")
            print(f"          - 动态幂参数 (power_p): {power_p_aligned.loc[probe_ts]:.4f}")
            print(f"          - 最终融合分数: {final_fused_score.loc[probe_ts]:.4f}")
        return final_fused_score

    def _calculate_price_momentum_resonance(self, df: pd.DataFrame, tf_weights_config: Dict, default_tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        计算价格动能共振分数。
        融合多时间维度 (5, 13, 21, 34, 55) 的价格斜率和加速度。
        """
        method_name = "_calculate_price_momentum_resonance"
        momentum_periods = [5, 13, 21, 34, 55]
        period_scores_list = []
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # 预取所有斜率和加速度的原始数据
        slope_raw_data = {p: self._get_safe_series(df, f'SLOPE_{p}_close_D', 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts) for p in momentum_periods}
        accel_raw_data = {p: self._get_safe_series(df, f'ACCEL_{p}_close_D', 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts) for p in momentum_periods}
        for p in momentum_periods:
            norm_window = p * 2 # 归一化窗口通常取周期的两倍
            # 为每个周期单独计算双极归一化分数
            slope_score_bipolar = get_robust_bipolar_normalized_score(slope_raw_data[p], df.index, window=norm_window, sensitivity=2.0, default_value=0.0, debug_info=False)
            accel_score_bipolar = get_robust_bipolar_normalized_score(accel_raw_data[p], df.index, window=norm_window, sensitivity=2.0, default_value=0.0, debug_info=False)
            # 转换为单极性 [0, 1]，并确保为 float32
            slope_score = ((slope_score_bipolar + 1) / 2).astype(np.float32)
            accel_score = ((accel_score_bipolar + 1) / 2).astype(np.float32)
            period_momentum_score = self._robust_generalized_mean(
                {"slope": slope_score, "acceleration": accel_score},
                {"slope": 0.5, "acceleration": 0.5},
                df.index,
                power_p=0.0,
                is_debug_enabled=is_debug_enabled,
                probe_ts=probe_ts,
                fusion_level_name=f"{p}d 动能子融合"
            )
            period_scores_list.append(period_momentum_score)
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"        [探针 - {method_name}] {p}d 周期动量 @ {probe_ts.strftime('%Y-%m-%d')}:")
                print(f"          - 原始斜率: {slope_raw_data[p].loc[probe_ts]:.4f}, 归一化斜率: {slope_score.loc[probe_ts]:.4f}")
                print(f"          - 原始加速度: {accel_raw_data[p].loc[probe_ts]:.4f}, 归一化加速度: {accel_score.loc[probe_ts]:.4f}")
                print(f"          - 周期动量子分数: {period_momentum_score.loc[probe_ts]:.4f}")
        momentum_fusion_weights = get_param_value(tf_weights_config.get('price_momentum_resonance'), default_tf_weights)
        if not period_scores_list:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        fused_score_components = []
        total_weight = 0.0
        for i, p in enumerate(momentum_periods):
            weight = momentum_fusion_weights.get(str(p), 0.0)
            if weight > 0:
                fused_score_components.append(period_scores_list[i] * weight)
                total_weight += weight
        if not fused_score_components or total_weight == 0:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        final_momentum_resonance_score = sum(fused_score_components) / total_weight
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 最终价格动能共振分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_momentum_resonance_score.loc[probe_ts]:.4f}")
        return final_momentum_resonance_score.clip(0, 1).astype(np.float32)

    def _calculate_structural_health(self, df: pd.DataFrame, tf_weights_config: Dict, default_tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        计算结构健康分数。
        融合流动性真实性、订单簿不平衡和资金流可信度。
        """
        method_name = "_calculate_structural_health"
        debug_info = (is_debug_enabled, probe_ts, method_name)
        liquidity_authenticity_raw = self._get_safe_series(df, 'liquidity_authenticity_score_D', 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts)
        order_book_imbalance_raw = self._get_safe_series(df, 'order_book_imbalance_D', 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts)
        flow_credibility_raw = self._get_safe_series(df, 'flow_credibility_index_D', 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts)
        # 收集所有需要归一化的窗口
        windows_liquidity = [int(p) for p in get_param_value(tf_weights_config.get('liquidity_authenticity'), default_tf_weights).keys()]
        windows_flow_credibility = [int(p) for p in get_param_value(tf_weights_config.get('flow_credibility'), default_tf_weights).keys()]
        # 一次性计算所有窗口的归一化分数
        liquidity_authenticity_score = get_adaptive_mtf_normalized_score(liquidity_authenticity_raw, df.index, tf_weights=get_param_value(tf_weights_config.get('liquidity_authenticity'), default_tf_weights), ascending=True, debug_info=False)
        # 修正：直接传入整数21作为window参数
        order_book_imbalance_score_bipolar = get_robust_bipolar_normalized_score(order_book_imbalance_raw, df.index, window=21, sensitivity=2.0, default_value=0.0, debug_info=False)
        flow_credibility_score = get_adaptive_mtf_normalized_score(flow_credibility_raw, df.index, tf_weights=get_param_value(tf_weights_config.get('flow_credibility'), default_tf_weights), ascending=True, debug_info=False)
        # 从一次性计算的结果中获取对应窗口的分数
        # 修正：order_book_imbalance_score_bipolar 现在已经是Series，直接使用
        order_book_imbalance_score = (order_book_imbalance_score_bipolar + 1) / 2
        # 几何平均融合
        structural_health_score = self._robust_generalized_mean(
            {
                "liquidity_authenticity": liquidity_authenticity_score,
                "order_book_imbalance": order_book_imbalance_score,
                "flow_credibility": flow_credibility_score
            },
            {"liquidity_authenticity": 0.33, "order_book_imbalance": 0.33, "flow_credibility": 0.34},
            df.index,
            power_p=0.0,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="结构健康子融合"
        )
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - liquidity_authenticity_score_D: {liquidity_authenticity_raw.loc[probe_ts]:.4f}")
            print(f"        - order_book_imbalance_D: {order_book_imbalance_raw.loc[probe_ts]:.4f}")
            print(f"        - flow_credibility_index_D: {flow_credibility_raw.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 归一化流动性真实性: {liquidity_authenticity_score.loc[probe_ts]:.4f}")
            print(f"        - 归一化订单簿不平衡: {order_book_imbalance_score.loc[probe_ts]:.4f}")
            print(f"        - 归一化资金流可信度: {flow_credibility_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终结构健康分数 @ {probe_ts.strftime('%Y-%m-%d')}: {structural_health_score.loc[probe_ts]:.4f}")
        return structural_health_score.clip(0, 1)

    def _calculate_behavioral_day_quality(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        method_name = "_calculate_behavioral_day_quality"
        debug_info = (is_debug_enabled, probe_ts, method_name)
        params = get_param_value(self.config_params.get('day_quality_protocol_params'), {})
        outcome_weights = get_param_value(params.get('outcome_weights'), {"intraday_posture": 0.25, "closing_strength": 0.2, "pct_change": 0.2, "auction_closing_position": 0.1, "closing_conviction": 0.1, "closing_acceptance": 0.05, "auction_impact": 0.1})
        process_weights = get_param_value(params.get('process_weights'), {"microstructure_efficiency": 0.25, "impulse_quality": 0.2, "vwap_control": 0.2, "main_force_activity": 0.1, "intraday_energy_density": 0.1, "flow_credibility": 0.05, "control_solidity": 0.1})
        narrative_weights = get_param_value(params.get('narrative_weights'), {"closing_auction_ambush_inverse": 0.2, "deception_lure_long_inverse": 0.15, "deception_lure_short_positive": 0.15, "wash_trade_intensity_inverse": 0.15, "main_force_slippage_inverse": 0.1, "deception_index_inverse": 0.1, "structural_tension_inverse": 0.05, "panic_selling_cascade_inverse": 0.1})
        behavioral_cohesion_weights = get_param_value(params.get('behavioral_cohesion_weights'), {"trend_conviction": 0.25, "trend_efficiency": 0.25, "trend_asymmetry": 0.2, "main_force_conviction": 0.15, "covert_accumulation": 0.1, "covert_distribution_inverse": 0.05})
        trend_momentum_resonance_weights = get_param_value(params.get('trend_momentum_resonance_weights'), {"price_momentum_resonance": 0.6, "structural_health": 0.4})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"outcome_assessment": 0.2, "process_quality": 0.2, "narrative_integrity": 0.2, "behavioral_cohesion": 0.2, "trend_momentum_resonance": 0.2})
        context_modulator_params = get_param_value(params.get('context_modulator_params'), {})
        final_exponent = get_param_value(params.get('final_exponent'), 1.2)
        fusion_power_p = get_param_value(params.get('fusion_power_p'), 0.1)
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_tf_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        tf_weights_config = get_param_value(params.get('tf_weights'), {}) # This is a dict of tf_weights for specific signals
        # --- 1. 获取所有原始数据 ---
        required_signals = [
            'intraday_posture_score_D', 'microstructure_efficiency_index_D', 'impulse_quality_ratio_D',
            'closing_strength_index_D', 'pct_change_D', 'vwap_control_strength_D',
            'closing_auction_ambush_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'market_sentiment_score_D',
            'auction_closing_position_D', 'closing_conviction_score_D', 'main_force_activity_ratio_D',
            'intraday_energy_density_D', 'wash_trade_intensity_D', 'main_force_slippage_index_D',
            'trend_conviction_score_D', 'trend_efficiency_ratio_D', 'trend_asymmetry_index_D',
            'closing_acceptance_type_D', 'auction_impact_score_D',
            'flow_credibility_index_D', 'control_solidity_index_D',
            'deception_index_D', 'structural_tension_index_D', 'panic_selling_cascade_D',
            'main_force_conviction_index_D', 'covert_accumulation_signal_D', 'covert_distribution_signal_D',
            'SLOPE_5_close_D', 'ACCEL_5_close_D',
            'SLOPE_13_close_D', 'ACCEL_13_close_D',
            'SLOPE_21_close_D', 'ACCEL_21_close_D',
            'SLOPE_34_close_D', 'ACCEL_34_close_D',
            'SLOPE_55_close_D', 'ACCEL_55_close_D',
            'liquidity_authenticity_score_D', 'order_book_imbalance_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回0分。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的信号，减少重复的字典查找和方法调用
        signals_data = {sig: df[sig] for sig in required_signals}
        # Cache for normalized scores
        normalized_scores_cache = {}
        # Helper to get cached score or compute it
        def get_cached_norm_score(series_raw, tf_w, asc, is_energy=False, is_bipolar=False, window=None):
            # Create a unique key for the cache based on series ID, tf_weights (as tuple), ascending, and type
            # Using id(series_raw) is crucial here because the same raw data might be clipped or transformed
            # before being passed to the normalizer, creating a new Series object.
            # So, we need to ensure the key reflects the *exact* Series object being normalized.
            # For normalize_score (single window), window is part of the key
            key_components = [id(series_raw), id(frozenset(tf_w.items())) if tf_w else None, asc, is_energy, is_bipolar, window]
            key = tuple(key_components)
            if key not in normalized_scores_cache:
                if is_energy:
                    normalized_scores_cache[key] = get_adaptive_mtf_normalized_energy_score(series_raw, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
                elif is_bipolar:
                    normalized_scores_cache[key] = get_adaptive_mtf_normalized_bipolar_score(series_raw, df.index, tf_weights=tf_w, debug_info=False)
                elif window is not None: # For normalize_score (single window)
                    normalized_scores_cache[key] = normalize_score(series_raw, df.index, windows=window, ascending=asc, debug_info=False)
                else: # For get_adaptive_mtf_normalized_score
                    normalized_scores_cache[key] = get_adaptive_mtf_normalized_score(series_raw, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
            return normalized_scores_cache[key]
        # --- 2. 战役结果评估 (Outcome Assessment) ---
        intraday_posture_score = (signals_data['intraday_posture_score_D'].clip(-1, 1) + 1) / 2
        closing_strength_score = get_cached_norm_score(signals_data['closing_strength_index_D'], get_param_value(tf_weights_config.get('closing_strength'), default_tf_weights), True)
        pct_change_score = get_cached_norm_score(signals_data['pct_change_D'].clip(lower=0), get_param_value(tf_weights_config.get('pct_change'), default_tf_weights), True)
        auction_closing_position_score = get_cached_norm_score(signals_data['auction_closing_position_D'], get_param_value(tf_weights_config.get('auction_closing_position'), default_tf_weights), True)
        closing_conviction_score = get_cached_norm_score(signals_data['closing_conviction_score_D'], get_param_value(tf_weights_config.get('closing_conviction'), default_tf_weights), True)
        closing_acceptance_score = get_cached_norm_score(signals_data['closing_acceptance_type_D'], get_param_value(tf_weights_config.get('closing_acceptance'), default_tf_weights), True)
        auction_impact_score = get_cached_norm_score(signals_data['auction_impact_score_D'], get_param_value(tf_weights_config.get('auction_impact'), default_tf_weights), True)
        outcome_assessment_score = self._robust_generalized_mean(
            {
                "intraday_posture": intraday_posture_score,
                "closing_strength": closing_strength_score,
                "pct_change": pct_change_score,
                "auction_closing_position": auction_closing_position_score,
                "closing_conviction": closing_conviction_score,
                "closing_acceptance": closing_acceptance_score,
                "auction_impact": auction_impact_score
            },
            outcome_weights,
            df.index,
            power_p=0.0,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="战役结果评估"
        )
        # --- 3. 战役过程质量评估 (Process Quality) ---
        micro_efficiency_score = get_cached_norm_score(signals_data['microstructure_efficiency_index_D'], default_tf_weights, True)
        impulse_quality_score = get_cached_norm_score(signals_data['impulse_quality_ratio_D'], default_tf_weights, True)
        vwap_control_score = (signals_data['vwap_control_strength_D'].clip(-1, 1) + 1) / 2
        main_force_activity_score = get_cached_norm_score(signals_data['main_force_activity_ratio_D'], get_param_value(tf_weights_config.get('main_force_activity'), default_tf_weights), True)
        intraday_energy_density_score = get_cached_norm_score(signals_data['intraday_energy_density_D'], get_param_value(tf_weights_config.get('intraday_energy_density'), default_tf_weights), True, is_energy=True)
        flow_credibility_score = get_cached_norm_score(signals_data['flow_credibility_index_D'], get_param_value(tf_weights_config.get('flow_credibility'), default_tf_weights), True)
        control_solidity_score = get_cached_norm_score(signals_data['control_solidity_index_D'], get_param_value(tf_weights_config.get('control_solidity'), default_tf_weights), True)
        process_quality_score = self._robust_generalized_mean(
            {
                "microstructure_efficiency": micro_efficiency_score,
                "impulse_quality": impulse_quality_score,
                "vwap_control": vwap_control_score,
                "main_force_activity": main_force_activity_score,
                "intraday_energy_density": intraday_energy_density_score,
                "flow_credibility": flow_credibility_score,
                "control_solidity": control_solidity_score
            },
            process_weights,
            df.index,
            power_p=0.0,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="战役过程质量评估"
        )
        # --- 4. 战役叙事诚信度 (Narrative Integrity) ---
        closing_auction_ambush_inverse = (1 - get_cached_norm_score(signals_data['closing_auction_ambush_D'], get_param_value(tf_weights_config.get('closing_auction_ambush'), default_tf_weights), True)).clip(0, 1)
        deception_lure_long_inverse = (1 - get_cached_norm_score(signals_data['deception_lure_long_intensity_D'], get_param_value(tf_weights_config.get('deception_lure_long'), default_tf_weights), True)).clip(0, 1)
        deception_lure_short_positive = get_cached_norm_score(signals_data['deception_lure_short_intensity_D'], get_param_value(tf_weights_config.get('deception_lure_short'), default_tf_weights), True)
        wash_trade_intensity_inverse = (1 - get_cached_norm_score(signals_data['wash_trade_intensity_D'], get_param_value(tf_weights_config.get('wash_trade_intensity'), default_tf_weights), True)).clip(0, 1)
        main_force_slippage_inverse = (1 - get_cached_norm_score(signals_data['main_force_slippage_index_D'], get_param_value(tf_weights_config.get('main_force_slippage'), default_tf_weights), True)).clip(0, 1)
        deception_index_inverse = (1 - get_cached_norm_score(signals_data['deception_index_D'], get_param_value(tf_weights_config.get('deception_index'), default_tf_weights), True)).clip(0, 1)
        structural_tension_inverse = (1 - get_cached_norm_score(signals_data['structural_tension_index_D'], get_param_value(tf_weights_config.get('structural_tension'), default_tf_weights), True)).clip(0, 1)
        panic_selling_cascade_inverse = (1 - get_cached_norm_score(signals_data['panic_selling_cascade_D'], get_param_value(tf_weights_config.get('panic_selling_cascade'), default_tf_weights), True)).clip(0, 1)
        narrative_integrity_score = self._robust_generalized_mean(
            {
                "closing_auction_ambush_inverse": closing_auction_ambush_inverse,
                "deception_lure_long_inverse": deception_lure_long_inverse,
                "deception_lure_short_positive": deception_lure_short_positive,
                "wash_trade_intensity_inverse": wash_trade_intensity_inverse,
                "main_force_slippage_inverse": main_force_slippage_inverse,
                "deception_index_inverse": deception_index_inverse,
                "structural_tension_inverse": structural_tension_inverse,
                "panic_selling_cascade_inverse": panic_selling_cascade_inverse
            },
            narrative_weights,
            df.index,
            power_p=0.0,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="战役叙事诚信度"
        )
        # --- 5. 行为协同 (Behavioral Cohesion) ---
        trend_conviction_score = get_cached_norm_score(signals_data['trend_conviction_score_D'], get_param_value(tf_weights_config.get('trend_conviction'), default_tf_weights), True)
        trend_efficiency_score = get_cached_norm_score(signals_data['trend_efficiency_ratio_D'], get_param_value(tf_weights_config.get('trend_efficiency'), default_tf_weights), True)
        trend_asymmetry_score = get_cached_norm_score(signals_data['trend_asymmetry_index_D'], get_param_value(tf_weights_config.get('trend_asymmetry'), default_tf_weights), True)
        main_force_conviction_score = get_cached_norm_score(signals_data['main_force_conviction_index_D'], get_param_value(tf_weights_config.get('main_force_conviction'), default_tf_weights), True)
        covert_accumulation_score = get_cached_norm_score(signals_data['covert_accumulation_signal_D'], get_param_value(tf_weights_config.get('covert_accumulation'), default_tf_weights), True)
        covert_distribution_inverse = (1 - get_cached_norm_score(signals_data['covert_distribution_signal_D'], get_param_value(tf_weights_config.get('covert_distribution'), default_tf_weights), True)).clip(0, 1)
        behavioral_cohesion_score = self._robust_generalized_mean(
            {
                "trend_conviction": trend_conviction_score,
                "trend_efficiency": trend_efficiency_score,
                "trend_asymmetry": trend_asymmetry_score,
                "main_force_conviction": main_force_conviction_score,
                "covert_accumulation": covert_accumulation_score,
                "covert_distribution_inverse": covert_distribution_inverse
            },
            behavioral_cohesion_weights,
            df.index,
            power_p=0.0,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="行为协同"
        )
        # --- 6. 趋势动能与结构共振 (Trend Momentum & Structural Resonance) ---
        price_momentum_resonance_score = self._calculate_price_momentum_resonance(df, tf_weights_config, default_tf_weights, is_debug_enabled, probe_ts)
        structural_health_score = self._calculate_structural_health(df, tf_weights_config, default_tf_weights, is_debug_enabled, probe_ts)
        trend_momentum_resonance_score = self._robust_generalized_mean(
            {
                "price_momentum_resonance": price_momentum_resonance_score,
                "structural_health": structural_health_score
            },
            trend_momentum_resonance_weights,
            df.index,
            power_p=0.0,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="趋势动能与结构共振"
        )
        # --- 7. 顶层融合 (加权广义平均) ---
        dynamic_fusion_weights = fusion_weights.copy()
        if context_modulator_params.get('dynamic_fusion_weights_enabled', False):
            volatility_signal_name = context_modulator_params.get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D')
            sentiment_signal_name = context_modulator_params.get('sentiment_signal', 'market_sentiment_score_D')
            norm_volatility = get_cached_norm_score(signals_data[volatility_signal_name], get_param_value(tf_weights_config.get('volatility_instability'), default_tf_weights), True)
            norm_sentiment = get_cached_norm_score(signals_data[sentiment_signal_name], get_param_value(tf_weights_config.get('market_sentiment'), default_tf_weights), True)
            base_weights = context_modulator_params.get('dynamic_fusion_weights_base', fusion_weights)
            volatility_impact = context_modulator_params.get('volatility_impact_weights', {})
            sentiment_impact = context_modulator_params.get('sentiment_impact_weights', {})
            for dim in dynamic_fusion_weights.keys():
                current_weight = base_weights.get(dim, 0.0)
                v_impact = volatility_impact.get(dim, 0.0)
                s_impact = sentiment_impact.get(dim, 0.0)
                dynamic_fusion_weights[dim] = current_weight + \
                                              norm_volatility * v_impact + \
                                              norm_sentiment * s_impact
            total_dynamic_weight = sum(dynamic_fusion_weights.values())
            zero_sum_mask = (total_dynamic_weight.abs() < 1e-9)
            normalized_dynamic_weights = {}
            for dim in dynamic_fusion_weights.keys():
                normalized_dim_weight = dynamic_fusion_weights[dim] / total_dynamic_weight.where(~zero_sum_mask, 1.0)
                normalized_dynamic_weights[dim] = normalized_dim_weight.where(
                    ~zero_sum_mask,
                    pd.Series(fusion_weights.get(dim, 0.0), index=df.index) # Ensure it's a Series for consistent operations
                )
            dynamic_fusion_weights = normalized_dynamic_weights
        day_quality_base_score = self._robust_generalized_mean(
            {
                "outcome_assessment": outcome_assessment_score,
                "process_quality": process_quality_score,
                "narrative_integrity": narrative_integrity_score,
                "behavioral_cohesion": behavioral_cohesion_score,
                "trend_momentum_resonance": trend_momentum_resonance_score
            },
            dynamic_fusion_weights,
            df.index,
            power_p=fusion_power_p,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="顶层融合基础分数"
        )
        # --- 8. 情境调制 (Contextual Modulation) ---
        dynamic_modulator_factor = pd.Series(1.0, index=df.index)
        if context_modulator_params.get('enabled', False):
            volatility_signal_name = context_modulator_params.get('volatility_signal', 'VOLATILITY_INSTABILITY_INDEX_21d_D')
            sentiment_signal_name = context_modulator_params.get('sentiment_signal', 'market_sentiment_score_D')
            norm_volatility = get_cached_norm_score(signals_data[volatility_signal_name], get_param_value(tf_weights_config.get('volatility_instability'), default_tf_weights), True)
            norm_sentiment = get_cached_norm_score(signals_data[sentiment_signal_name], get_param_value(tf_weights_config.get('market_sentiment'), default_tf_weights), True)
            norm_sentiment_neutrality = get_cached_norm_score(signals_data['market_sentiment_score_D'].abs(), get_param_value(tf_weights_config.get('market_sentiment'), default_tf_weights), True) # For neutrality
            volatility_sensitivity = context_modulator_params.get('volatility_sensitivity', 0.3)
            sentiment_sensitivity = context_modulator_params.get('sentiment_sensitivity', 0.2)
            base_modulator_factor = context_modulator_params.get('base_modulator_factor', 1.0)
            min_modulator = context_modulator_params.get('min_modulator', 0.8)
            max_modulator = context_modulator_params.get('max_modulator', 1.2)
            norm_volatility_inverse = (1 - norm_volatility).clip(0, 1)
            dynamic_modulator_factor = base_modulator_factor + \
                                       norm_volatility_inverse * volatility_sensitivity + \
                                       norm_sentiment_neutrality * sentiment_sensitivity
            dynamic_modulator_factor = dynamic_modulator_factor.clip(min_modulator, max_modulator)
        final_score_modulated = (day_quality_base_score * dynamic_modulator_factor).clip(0, 1)
        # --- 9. 最终非线性变换并映射到 [-1, 1] ---
        final_day_quality_score = (final_score_modulated.pow(final_exponent) * 2 - 1).clip(-1, 1)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            for sig_name in required_signals:
                print(f"        - {sig_name}: {signals_data[sig_name].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战役结果评估分数: {outcome_assessment_score.loc[probe_ts]:.4f}")
            print(f"        - 战役过程质量评估分数: {process_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 战役叙事诚信度分数: {narrative_integrity_score.loc[probe_ts]:.4f}")
            print(f"        - 行为协同分数: {behavioral_cohesion_score.loc[probe_ts]:.4f}")
            print(f"        - 趋势动能与结构共振分数: {trend_momentum_resonance_score.loc[probe_ts]:.4f}")
            print(f"        - 顶层融合基础分数: {day_quality_base_score.loc[probe_ts]:.4f}")
            print(f"        - 动态调制因子: {dynamic_modulator_factor.loc[probe_ts]:.4f}")
            print(f"        - 调制后分数: {final_score_modulated.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终行为K线质量分 @ {probe_ts.strftime('%Y-%m-%d')}: {final_day_quality_score.loc[probe_ts]:.4f}")
        return final_day_quality_score.astype(np.float32)

    def _diagnose_behavioral_axioms(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Dict[str, pd.Series]:
        method_name = "_diagnose_behavioral_axioms"
        p_behavioral_div_conf = self.config_params
        mtf_slopes_params_from_config = p_behavioral_div_conf.get('multi_timeframe_slopes')
        default_mtf_slopes_config = {"enabled": True, "periods": [5, 13], "weights": {"5": 0.7, "13": 0.3}}
        if mtf_slopes_params_from_config is None:
            mtf_slopes_params = default_mtf_slopes_config
        else:
            mtf_slopes_params = {**default_mtf_slopes_config, **mtf_slopes_params_from_config}
            if 'weights' in mtf_slopes_params_from_config and isinstance(mtf_slopes_params_from_config['weights'], dict):
                mtf_slopes_params['weights'] = {**default_mtf_slopes_config['weights'], **mtf_slopes_params_from_config['weights']}
            elif 'weights' not in mtf_slopes_params_from_config:
                mtf_slopes_params['weights'] = default_mtf_slopes_config['weights']
        mtf_periods = mtf_slopes_params.get('periods', [5])
        multi_level_resonance_params = get_param_value(p_behavioral_div_conf.get('multi_level_resonance_params'), {"enabled": True, "long_term_period": 21, "resonance_bonus": 0.2})
        long_term_period = multi_level_resonance_params.get('long_term_period', 21)
        pattern_sequence_params = get_param_value(p_behavioral_div_conf.get('pattern_sequence_params'), {"enabled": True, "lookback_window": 3, "volume_drying_up_ratio": 0.8, "volume_climax_ratio": 1.5, "reversal_pct_change_threshold": 0.01, "sequence_bonus": 0.2})
        pattern_lookback_window = pattern_sequence_params.get('lookback_window', 3)
        accel_period = mtf_periods[0]
        required_signals = [
            'close_D', 'high_D', 'low_D', 'open_D', 'volume_D', 'amount_D', 'pct_change_D',
            'volume_ratio_D', 'turnover_rate_f_D', 'main_force_net_flow_calibrated_D',
            'retail_net_flow_calibrated_D', 'net_mf_amount_D', 'buy_elg_amount_D', 'buy_lg_amount_D',
            'dip_absorption_power_D', 'lower_shadow_absorption_strength_D',
            'rally_distribution_pressure_D', 'upper_shadow_selling_pressure_D',
            'profit_taking_flow_ratio_D', 'main_force_execution_alpha_D',
            'SLOPE_5_main_force_conviction_index_D', 'breakout_quality_score_D',
            'SLOPE_5_breakout_quality_score_D', 'total_winner_rate_D', 'winner_stability_index_D',
            'control_solidity_index_D', 'trend_vitality_index_D', 'BIAS_21_D', 'RSI_13_D',
            'ACCEL_5_pct_change_D', 'closing_strength_index_D', 'active_selling_pressure_D',
            'chip_fatigue_index_D', 'main_force_ofi_D', 'retail_ofi_D', 'buy_quote_exhaustion_rate_D',
            'sell_quote_exhaustion_rate_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'microstructure_efficiency_index_D', 'upward_impulse_purity_D', 'vacuum_traversal_efficiency_D',
            'support_validation_strength_D', 'impulse_quality_ratio_D', 'floating_chip_cleansing_efficiency_D',
            'panic_selling_cascade_D', 'capitulation_absorption_index_D', 'covert_accumulation_signal_D',
            'VOL_MA_5_D', 'VOL_MA_13_D', 'VOL_MA_21_D', 'loser_pain_index_D',
            'wash_trade_intensity_D', 'closing_auction_ambush_D', 'mf_retail_battle_intensity_D',
            'main_force_conviction_index_D', 'SLOPE_5_loser_pain_index_D',
            'pressure_rejection_strength_D', 'active_buying_support_D', 'vwap_control_strength_D',
            'SLOPE_5_winner_stability_index_D', 'retail_fomo_premium_index_D', 'BBP_21_2.0_D', 'BIAS_5_D',
            'ATR_14_D', 'BBW_21_2.0_D', 'ADX_14_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', 'intraday_posture_score_D',
            'volume_structure_skew_D', 'SLOPE_55_close_D', 'market_sentiment_score_D',
            'SLOPE_55_ADX_14_D', 'order_book_imbalance_D', 'micro_price_impact_asymmetry_D',
            'sell_sweep_intensity_D', 'panic_sell_volume_contribution_D', 'ask_side_liquidity_D', 'bid_side_liquidity_D', 'liquidity_slope_D',
            'market_impact_cost_D', 'order_book_clearing_rate_D', 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D', 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d_D',
            'price_volume_entropy_D', 'volatility_expansion_ratio_D', 'trend_acceleration_score_D', 'constructive_turnover_ratio_D',
            'buy_sweep_intensity_D', 'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT'
        ]
        # 使用 set 进行去重，然后转换回 list
        required_signals = list(set(required_signals))
        indicators_for_robust_slopes = [
            'close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'BBW_21_2.0', 'pct_change',
            'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry',
            'breakout_quality_score', 'upward_impulse_purity', 'trend_acceleration_score',
            'volume_burstiness_index', 'constructive_turnover_ratio', 'buy_sweep_intensity',
            'upper_shadow_selling_pressure', 'market_sentiment_score'
        ]
        for indicator in indicators_for_robust_slopes:
            for period in mtf_periods:
                required_signals.append(f'SLOPE_{period}_{indicator}_D')
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'ADX_14']:
            required_signals.append(f'SLOPE_{long_term_period}_{indicator}_D')
        for indicator in ['close', 'volume']:
            required_signals.append(f'SLOPE_{pattern_lookback_window}_{indicator}_D')
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume',
                         'breakout_quality_score', 'upward_impulse_purity', 'trend_acceleration_score',
                         'volume_burstiness_index', 'constructive_turnover_ratio', 'buy_sweep_intensity',
                         'upper_shadow_selling_pressure', 'market_sentiment_score']:
            required_signals.append(f'ACCEL_{accel_period}_{indicator}_D')
        liquidity_drain_mtf_periods = get_param_value(p_behavioral_div_conf.get('liquidity_drain_params', {}).get('mtf_slope_accel_weights'), {}).keys()
        liquidity_drain_mtf_periods = [int(p) for p in liquidity_drain_mtf_periods]
        indicators_for_mtf_dynamics = [
            'pct_change', 'panic_selling_cascade', 'active_selling_pressure', 'retail_panic_surrender_index',
            'main_force_net_flow_calibrated', 'sell_sweep_intensity', 'loser_pain_index',
            'active_buying_support', 'vwap_control_strength', 'buy_quote_exhaustion_rate',
            'support_validation_strength', 'chip_fatigue_index', 'sell_quote_exhaustion_rate',
            'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry',
            'ask_side_liquidity', 'bid_side_liquidity', 'market_impact_cost',
            'BID_LIQUIDITY_SAMPLE_ENTROPY_13d', 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d',
            'price_volume_entropy', 'volatility_expansion_ratio'
        ]
        for period in liquidity_drain_mtf_periods:
            for indicator in indicators_for_mtf_dynamics:
                required_signals.append(f'SLOPE_{period}_{indicator}_D')
                required_signals.append(f'ACCEL_{period}_{indicator}_D')
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 核心公理诊断失败，缺少必要原始信号，行为分析中止。")
            return {}
        states = {}
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        long_term_weights = get_param_value(p_mtf.get('long_term_stability'), {'21': 0.5, '55': 0.3, '89': 0.2})
        # 集中提取所有必需的原始信号，减少重复的字典查找和方法调用
        raw_df_signals = {sig: df[sig] for sig in required_signals}
        pct_change = raw_df_signals['pct_change_D']
        price_accel = raw_df_signals['ACCEL_5_pct_change_D']
        robust_slopes = {}
        all_slope_cols_to_extract = []
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'BBW_21_2.0', 'pct_change', 'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry',
                         'breakout_quality_score', 'upward_impulse_purity', 'trend_acceleration_score', 'volume_burstiness_index', 'constructive_turnover_ratio', 'buy_sweep_intensity', 'upper_shadow_selling_pressure', 'market_sentiment_score']:
            for period in mtf_periods:
                col_name = f'SLOPE_{period}_{indicator}_D'
                if col_name in raw_df_signals:
                    all_slope_cols_to_extract.append(col_name)
        if all_slope_cols_to_extract:
            slopes_df_extracted = pd.DataFrame({col: raw_df_signals[col] for col in all_slope_cols_to_extract})
        else:
            slopes_df_extracted = pd.DataFrame(index=df.index)
        for indicator in ['close', 'RSI_13', 'MACDh_13_34_8', 'volume', 'BBW_21_2.0', 'pct_change', 'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry',
                         'breakout_quality_score', 'upward_impulse_purity', 'trend_acceleration_score', 'volume_burstiness_index', 'constructive_turnover_ratio', 'buy_sweep_intensity', 'upper_shadow_selling_pressure', 'market_sentiment_score']:
            weighted_slope = pd.Series(0.0, index=df.index, dtype=np.float32)
            total_weight = 0.0
            indicator_slopes_cols = []
            current_weights = []
            for period in mtf_periods:
                col_name = f'SLOPE_{period}_{indicator}_D'
                if col_name in slopes_df_extracted.columns:
                    indicator_slopes_cols.append(col_name)
                    current_weights.append(mtf_slopes_params['weights'].get(str(period), 0.0))
            if indicator_slopes_cols and sum(current_weights) > 0:
                weighted_slope = (slopes_df_extracted[indicator_slopes_cols] * current_weights).sum(axis=1)
                total_weight = sum(current_weights)
                robust_slopes[indicator] = weighted_slope / total_weight
            else:
                robust_slopes[indicator] = raw_df_signals.get(f'SLOPE_{mtf_periods[0]}_{indicator}_D', pd.Series(0.0, index=df.index, dtype=np.float32)) if mtf_periods else pd.Series(0.0, index=df.index, dtype=np.float32)
            df[f'robust_{indicator}_slope'] = robust_slopes[indicator]
            states[f'robust_{indicator}_slope'] = robust_slopes[indicator]
        long_term_period = multi_level_resonance_params.get('long_term_period', 21)
        long_term_close_slope = raw_df_signals[f'SLOPE_{long_term_period}_close_D']
        long_term_rsi_slope = raw_df_signals[f'SLOPE_{long_term_period}_RSI_13_D']
        long_term_macd_slope = raw_df_signals[f'SLOPE_{long_term_period}_MACDh_13_34_8_D']
        long_term_volume_slope = raw_df_signals[f'SLOPE_{long_term_period}_volume_D']
        long_term_adx_slope = raw_df_signals[f'SLOPE_{long_term_period}_ADX_14_D']
        df['long_term_close_slope'] = long_term_close_slope
        states['long_term_close_slope'] = long_term_close_slope
        df['long_term_RSI_13_slope'] = long_term_rsi_slope
        states['long_term_RSI_13_slope'] = long_term_rsi_slope
        df['long_term_MACDh_13_34_8_slope'] = long_term_macd_slope
        states['long_term_MACDh_13_34_8_slope'] = long_term_macd_slope
        df['long_term_volume_slope'] = long_term_volume_slope
        states['long_term_volume_slope'] = long_term_volume_slope
        df['long_term_adx_slope'] = long_term_adx_slope
        states['long_term_adx_slope'] = long_term_adx_slope
        pattern_lookback_window = pattern_sequence_params.get('lookback_window', 3)
        pattern_close_slope = raw_df_signals[f'SLOPE_{pattern_lookback_window}_close_D']
        pattern_volume_slope = raw_df_signals[f'SLOPE_{pattern_lookback_window}_volume_D']
        df['pattern_close_slope'] = pattern_close_slope
        states['pattern_close_slope'] = pattern_close_slope
        df['pattern_volume_slope'] = pattern_volume_slope
        states['pattern_volume_slope'] = pattern_volume_slope
        upward_momentum_score = self._diagnose_upward_momentum(df, default_weights, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = upward_momentum_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM' @ {probe_ts.strftime('%Y-%m-%d')}: {upward_momentum_score.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'] = upward_momentum_score.astype(np.float32)
        downward_momentum_score = self._diagnose_downward_momentum(df, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = downward_momentum_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM' @ {probe_ts.strftime('%Y-%m-%d')}: {downward_momentum_score.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM'] = downward_momentum_score.astype(np.float32)
        upward_efficiency_score = self._diagnose_upward_efficiency(df, default_weights, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY' @ {probe_ts.strftime('%Y-%m-%d')}: {upward_efficiency_score.loc[probe_ts]:.4f}")
        self.strategy.atomic_states['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        df['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'] = upward_efficiency_score.astype(np.float32)
        downward_resistance_score = self._diagnose_downward_resistance(df, default_weights, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE' @ {probe_ts.strftime('%Y-%m-%d')}: {downward_resistance_score.loc[probe_ts]:.4f}")
        self.strategy.atomic_states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        df['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] = downward_resistance_score.astype(np.float32)
        intraday_bull_control_score = self._diagnose_intraday_bull_control(df, default_weights, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL' @ {probe_ts.strftime('%Y-%m-%d')}: {intraday_bull_control_score.loc[probe_ts]:.4f}")
        self.strategy.atomic_states['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        df['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'] = intraday_bull_control_score.astype(np.float32)
        final_overextension_score = self._calculate_behavioral_price_overextension(df, default_weights, long_term_weights, is_debug_enabled, probe_ts)
        states['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = final_overextension_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW' @ {probe_ts.strftime('%Y-%m-%d')}: {final_overextension_score.loc[probe_ts]:.4f}")
        df['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'] = final_overextension_score.astype(np.float32)
        stagnation_evidence = self._calculate_behavioral_stagnation_evidence(df, default_weights, is_debug_enabled, probe_ts)
        states['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = stagnation_evidence.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW' @ {probe_ts.strftime('%Y-%m-%d')}: {stagnation_evidence.loc[probe_ts]:.4f}")
        df['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'] = stagnation_evidence.astype(np.float32)
        lower_shadow_quality = self._diagnose_lower_shadow_quality(df, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION' @ {probe_ts.strftime('%Y-%m-%d')}: {lower_shadow_quality.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] = lower_shadow_quality
        deception_index = self._diagnose_deception_index(df, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_DECEPTION_INDEX'] = deception_index
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_DECEPTION_INDEX' @ {probe_ts.strftime('%Y-%m-%d')}: {deception_index.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_DECEPTION_INDEX'] = deception_index
        distribution_intent = self._diagnose_distribution_intent(df, default_weights, final_overextension_score, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'] = distribution_intent
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT' @ {probe_ts.strftime('%Y-%m-%d')}: {distribution_intent.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'] = distribution_intent
        offensive_absorption_intent = self._diagnose_offensive_absorption_intent(df, lower_shadow_quality, distribution_intent, is_debug_enabled, probe_ts)
        states['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] = offensive_absorption_intent
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT' @ {probe_ts.strftime('%Y-%m-%d')}: {offensive_absorption_intent.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] = offensive_absorption_intent
        states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = self._diagnose_ambush_counterattack(df, offensive_absorption_intent, is_debug_enabled, probe_ts)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'].loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK'] = states['SCORE_BEHAVIOR_AMBUSH_COUNTERATTACK']
        states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = self._diagnose_breakout_failure_risk(
            df,
            distribution_intent,
            final_overextension_score,
            deception_index,
            is_debug_enabled,
            probe_ts
        )
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_RISK_BREAKOUT_FAILURE_CASCADE' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'].loc[probe_ts]:.4f}")
        df['SCORE_RISK_BREAKOUT_FAILURE_CASCADE'] = states['SCORE_RISK_BREAKOUT_FAILURE_CASCADE']
        states['SCORE_BEHAVIOR_VOLUME_BURST'] = self._calculate_volume_burst_quality(df, default_weights, is_debug_enabled, probe_ts)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_VOLUME_BURST' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_BEHAVIOR_VOLUME_BURST'].loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_VOLUME_BURST'] = states['SCORE_BEHAVIOR_VOLUME_BURST']
        states['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = self._calculate_volume_atrophy(df, default_weights, is_debug_enabled, probe_ts)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_VOLUME_ATROPHY' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_BEHAVIOR_VOLUME_ATROPHY'].loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_VOLUME_ATROPHY'] = states['SCORE_BEHAVIOR_VOLUME_ATROPHY']
        states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'] = self._calculate_absorption_strength(df, default_weights, is_debug_enabled, probe_ts)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_ABSORPTION_STRENGTH' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'].loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'] = states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH']
        states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'] = self._diagnose_shakeout_confirmation(
            df,
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'],
            is_debug_enabled, probe_ts
        )
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'].loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION'] = states['SCORE_BEHAVIOR_SHAKEOUT_CONFIRMATION']
        bullish_pure_div, bearish_pure_div = self._diagnose_pure_behavioral_divergence(
            df,
            default_weights,
            is_debug_enabled,
            probe_ts
        )
        states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_pure_div
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE' @ {probe_ts.strftime('%Y-%m-%d')}: {bullish_pure_div.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_pure_div
        states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_pure_div
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_BEARISH_DIVERGENCE' @ {probe_ts.strftime('%Y-%m-%d')}: {bearish_pure_div.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_pure_div
        microstructure_intent_score = raw_df_signals['SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT']
        bullish_divergence_quality, bearish_divergence_quality = self._diagnose_divergence_quality(
            df,
            states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH'],
            states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'],
            states, # 传递 states
            is_debug_enabled,
            probe_ts # 传递 probe_ts
        )
        states['SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'] = bullish_divergence_quality
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY' @ {probe_ts.strftime('%Y-%m-%d')}: {bullish_divergence_quality.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_BULLISH_DIVERGENCE_QUALITY'] = bullish_divergence_quality
        states['SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'] = bearish_divergence_quality
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY' @ {probe_ts.strftime('%Y-%m-%d')}: {bearish_divergence_quality.loc[probe_ts]:.4f}")
        df['SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY'] = bearish_divergence_quality
        lockup_rally_score = self._calculate_lockup_rally_opportunity(df, states, default_weights, is_debug_enabled, probe_ts)
        states['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = lockup_rally_score
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_OPPORTUNITY_LOCKUP_RALLY' @ {probe_ts.strftime('%Y-%m-%d')}: {lockup_rally_score.loc[probe_ts]:.4f}")
        df['SCORE_OPPORTUNITY_LOCKUP_RALLY'] = lockup_rally_score
        selling_exhaustion_score = self._diagnose_selling_exhaustion_opportunity(df, states, default_weights, is_debug_enabled, probe_ts)
        states['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = selling_exhaustion_score
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_OPPORTUNITY_SELLING_EXHAUSTION' @ {probe_ts.strftime('%Y-%m-%d')}: {selling_exhaustion_score.loc[probe_ts]:.4f}")
        df['SCORE_OPPORTUNITY_SELLING_EXHAUSTION'] = selling_exhaustion_score
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = self._diagnose_liquidity_drain_risk(df, states, default_weights, is_debug_enabled, probe_ts)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_RISK_LIQUIDITY_DRAIN' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_RISK_LIQUIDITY_DRAIN'].loc[probe_ts]:.4f}")
        df['SCORE_RISK_LIQUIDITY_DRAIN'] = states['SCORE_RISK_LIQUIDITY_DRAIN']
        pressure_absorption_signals = self._resolve_pressure_absorption_dynamics(df, self._calculate_raw_selling_pressure(df, default_weights, is_debug_enabled, probe_ts), states, is_debug_enabled, probe_ts)
        states.update(pressure_absorption_signals)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_RISK_UNRESOLVED_PRESSURE' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_RISK_UNRESOLVED_PRESSURE'].loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 发布 'SCORE_OPPORTUNITY_PRESSURE_ABSORPTION' @ {probe_ts.strftime('%Y-%m-%d')}: {states['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION'].loc[probe_ts]:.4f}")
        df['SCORE_RISK_UNRESOLVED_PRESSURE'] = states['SCORE_RISK_UNRESOLVED_PRESSURE']
        df['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION'] = states['SCORE_OPPORTUNITY_PRESSURE_ABSORPTION']
        return states

    def _resolve_pressure_absorption_dynamics(self, df: pd.DataFrame, raw_selling_pressure: pd.Series, states: Dict[str, pd.Series], is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Dict[str, pd.Series]:
        """
        【V2.0 · 生产就绪版】解析压力吸收动态。
        - 核心重构: 废弃V1.0“静态快照谬误”模型，引入“压力 × 吸收 × 意图”的全新三维动态博弈框架。
        - 诊断三维度:
          1. 压力强度 (Pressure Magnitude): 评估原始卖压的强度。
          2. 吸收能力 (Absorption Capacity): 评估市场承接卖压的能力。
          3. 意图确认 (Intent Confirmation): 评估主力吸收或派发的真实意图。
        - 数学模型:
          - 未解决压力风险 = 压力强度 * (1 - 吸收能力) * 派发意图
          - 压力吸收机会 = 吸收能力 * (1 - 压力强度) * (1 - 派发意图)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_resolve_pressure_absorption_dynamics"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在解析 '压力吸收动态' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('pressure_absorption_params'), {})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_state_signals = [
            'SCORE_BEHAVIOR_ABSORPTION_STRENGTH', 'SCORE_BEHAVIOR_DISTRIBUTION_INTENT'
        ]
        missing_state_signals = [s for s in required_state_signals if s not in states]
        if missing_state_signals:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心状态信号：{missing_state_signals}，返回0分。")
            return {
                'SCORE_RISK_UNRESOLVED_PRESSURE': pd.Series(0.0, index=df.index, dtype=np.float32),
                'SCORE_OPPORTUNITY_PRESSURE_ABSORPTION': pd.Series(0.0, index=df.index, dtype=np.float32)
            }
        absorption_strength = states['SCORE_BEHAVIOR_ABSORPTION_STRENGTH']
        distribution_intent = states['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'raw_selling_pressure': (raw_selling_pressure, default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        pressure_magnitude_score = normalized_mtf_scores['raw_selling_pressure']
        absorption_capacity_score = absorption_strength
        intent_confirmation_score = distribution_intent
        # 未解决压力风险
        unresolved_pressure_risk = (
            pressure_magnitude_score *
            (1 - absorption_capacity_score) *
            intent_confirmation_score
        ).clip(0, 1).fillna(0.0)
        # 压力吸收机会
        pressure_absorption_opportunity = (
            absorption_capacity_score *
            (1 - pressure_magnitude_score) *
            (1 - intent_confirmation_score)
        ).clip(0, 1).fillna(0.0)
        final_unresolved_pressure_risk = unresolved_pressure_risk.astype(np.float32)
        final_pressure_absorption_opportunity = pressure_absorption_opportunity.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - raw_selling_pressure (来自外部): {raw_selling_pressure.loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_ABSORPTION_STRENGTH (from states): {absorption_strength.loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_DISTRIBUTION_INTENT (from states): {distribution_intent.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 压力强度分数: {pressure_magnitude_score.loc[probe_ts]:.4f}")
            print(f"        - 吸收能力分数: {absorption_capacity_score.loc[probe_ts]:.4f}")
            print(f"        - 意图确认分数: {intent_confirmation_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '未解决压力风险'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_unresolved_pressure_risk.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '压力吸收机会'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_pressure_absorption_opportunity.loc[probe_ts]:.4f}")
        return {
            'SCORE_RISK_UNRESOLVED_PRESSURE': final_unresolved_pressure_risk,
            'SCORE_OPPORTUNITY_PRESSURE_ABSORPTION': final_pressure_absorption_opportunity
        }

    def _calculate_raw_selling_pressure(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V2.0 · 生产就绪版】计算纯粹基于行为类原始数据的原始卖压。
        - 核心重构: 废弃V1.0“单一指标谬误”模型，引入“战术抛压 × 战略脆弱性”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术抛压 (Tactical Selling Pressure): 评估主动卖盘的强度和效率。
          2. 战略脆弱性 (Strategic Vulnerability): 评估市场对卖压的抵抗能力。
        - 数学模型: 原始卖压 = (战术抛压分 * 战略脆弱性分) ^ 0.5
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_raw_selling_pressure"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '原始卖压' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('raw_selling_pressure_params'), {})
        tactical_weights = get_param_value(params.get('tactical_weights'), {"active_selling": 0.4, "upper_shadow": 0.3, "sell_sweep": 0.3})
        strategic_weights = get_param_value(params.get('strategic_weights'), {"downward_resistance_inverse": 0.5, "support_validation_inverse": 0.5})
        required_signals = [
            'active_selling_pressure_D', 'upper_shadow_selling_pressure_D',
            'sell_sweep_intensity_D', 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'support_validation_strength_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'active_selling_pressure_D': (signals_data['active_selling_pressure_D'], tf_weights, True),
            'upper_shadow_selling_pressure_D': (signals_data['upper_shadow_selling_pressure_D'], tf_weights, True),
            'sell_sweep_intensity_D': (signals_data['sell_sweep_intensity_D'], tf_weights, True),
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE': (signals_data['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'], tf_weights, True),
            'support_validation_strength_D': (signals_data['support_validation_strength_D'], tf_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        active_selling_score = normalized_mtf_scores['active_selling_pressure_D']
        upper_shadow_score = normalized_mtf_scores['upper_shadow_selling_pressure_D']
        sell_sweep_score = normalized_mtf_scores['sell_sweep_intensity_D']
        tactical_selling_pressure_score = (
            active_selling_score * tactical_weights.get('active_selling', 0.4) +
            upper_shadow_score * tactical_weights.get('upper_shadow', 0.3) +
            sell_sweep_score * tactical_weights.get('sell_sweep', 0.3)
        ).clip(0, 1)
        downward_resistance_inverse_score = (1 - normalized_mtf_scores['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']).clip(0, 1)
        support_validation_inverse_score = (1 - normalized_mtf_scores['support_validation_strength_D']).clip(0, 1)
        strategic_vulnerability_score = (
            downward_resistance_inverse_score * strategic_weights.get('downward_resistance_inverse', 0.5) +
            support_validation_inverse_score * strategic_weights.get('support_validation_inverse', 0.5)
        ).clip(0, 1)
        # --- 4. 最终合成 ---
        raw_selling_pressure = (tactical_selling_pressure_score * strategic_vulnerability_score).pow(0.5).fillna(0.0)
        final_score = raw_selling_pressure.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - active_selling_pressure_D: {signals_data['active_selling_pressure_D'].loc[probe_ts]:.4f}")
            print(f"        - upper_shadow_selling_pressure_D: {signals_data['upper_shadow_selling_pressure_D'].loc[probe_ts]:.4f}")
            print(f"        - sell_sweep_intensity_D: {signals_data['sell_sweep_intensity_D'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_DOWNWARD_RESISTANCE: {signals_data['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'].loc[probe_ts]:.4f}")
            print(f"        - support_validation_strength_D: {signals_data['support_validation_strength_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战术抛压分数: {tactical_selling_pressure_score.loc[probe_ts]:.4f}")
            print(f"        - 战略脆弱性分数: {strategic_vulnerability_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '原始卖压'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_dynamic_threshold(self, df: pd.DataFrame, params: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V1.0 · 动态阈值计算器】根据市场波动率动态调整价格跌幅阈值。
        - 核心逻辑: 阈值 = 基础阈值 + ATR * 乘数因子
        - 目标: 使流动性枯竭风险的触发条件更具适应性，避免在市场剧烈波动时过于敏感，或在市场平静时反应迟钝。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_dynamic_threshold"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '动态阈值' @ {probe_ts.strftime('%Y-%m-%d')}")
        base_threshold = get_param_value(params.get('base_threshold', 0.01))
        atr_multiplier = get_param_value(params.get('atr_multiplier', 0.005))
        min_threshold = get_param_value(params.get('min_threshold', 0.005))
        max_threshold = get_param_value(params.get('max_threshold', 0.02))
        required_signals = ['ATR_14_D']
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回默认阈值。")
            return pd.Series(base_threshold, index=df.index, dtype=np.float32)
        atr_val = df['ATR_14_D']
        dynamic_threshold = base_threshold + atr_val * atr_multiplier
        dynamic_threshold = dynamic_threshold.clip(min_threshold, max_threshold).fillna(base_threshold)
        final_score = dynamic_threshold.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - ATR_14_D: {atr_val.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 基础阈值: {base_threshold:.4f}")
            print(f"        - ATR乘数: {atr_multiplier:.4f}")
            print(f"        - 最小阈值: {min_threshold:.4f}")
            print(f"        - 最大阈值: {max_threshold:.4f}")
            print(f"      [探针 - {method_name}] 最终 '动态阈值' @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_upward_momentum(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V2.1 · 生产版】诊断高品质上涨动能。
        - 核心重构: 废弃了基于“表观强度幻觉”的 V2.0 模型。引入基于“闪电战三要素”
                      （攻击力度-战略指挥-后勤支撑）的全新品质诊断模型。
        - 闪电战三要素:
          1. 攻击力度 (Offensive Force): 审判攻击的纯净度与效率。采用 `upward_impulse_purity_D`
                                         和 `impulse_quality_ratio_D`。
          2. 战略指挥 (Strategic Command): 审判攻击背后的主力真实信念。采用 `main_force_conviction_index_D`。
          3. 后勤支撑 (Sustainability): 审判动能的可持续性，即内部获利盘的稳固程度。
                                        采用 `winner_stability_index_D`。
        - 数学模型: 动能分 = (攻击力度分 * 战略指挥分 * 后勤支撑分) ^ (1/3)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_upward_momentum"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '上涨动能' @ {probe_ts.strftime('%Y-%m-%d')}")
        required_signals = [
            'upward_impulse_purity_D', 'impulse_quality_ratio_D',
            'main_force_conviction_index_D', 'winner_stability_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'upward_impulse_purity_D': (signals_data['upward_impulse_purity_D'], tf_weights, True),
            'impulse_quality_ratio_D': (signals_data['impulse_quality_ratio_D'], tf_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), tf_weights, True),
            'winner_stability_index_D': (signals_data['winner_stability_index_D'], tf_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 2. 计算各要素得分 ---
        purity_score = normalized_mtf_scores['upward_impulse_purity_D']
        quality_score = normalized_mtf_scores['impulse_quality_ratio_D']
        offensive_force_score = (purity_score * quality_score).pow(0.5)
        strategic_command_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        sustainability_score = normalized_mtf_scores['winner_stability_index_D']
        # --- 3. “闪电战”三要素合成 ---
        upward_momentum_score = (
            (offensive_force_score + 1e-9) *
            (strategic_command_score + 1e-9) *
            (sustainability_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        final_score = upward_momentum_score.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - upward_impulse_purity_D: {signals_data['upward_impulse_purity_D'].loc[probe_ts]:.4f}")
            print(f"        - impulse_quality_ratio_D: {signals_data['impulse_quality_ratio_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - winner_stability_index_D: {signals_data['winner_stability_index_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 攻击力度 (Offensive Force): {offensive_force_score.loc[probe_ts]:.4f}")
            print(f"        - 战略指挥 (Strategic Command): {strategic_command_score.loc[probe_ts]:.4f}")
            print(f"        - 后勤支撑 (Sustainability): {sustainability_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '上涨动能'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_downward_momentum(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断价格下跌动能。
        - 核心重构: 废弃V2.0“单点否决”逻辑，引入基于“多头防御体系系统性崩溃”的全新诊断模型。
        - 诊断框架 (三要素):
          1. 前沿阵地失守 (The Frontline Breach): 审判攻击的纯净度与效率。采用 `upward_impulse_purity_D`
                                         和 `impulse_quality_ratio_D`。
          2. 防御工事崩塌 (The Fortress Crumbling): 审判抵抗的缺席 (1 - 防御力量分)。
          3. 指挥系统溃败 (The Command Collapse): 审判主力信心的崩塌 (1 - 主力正面信念分)。
        - 数学模型: 动能分 = (破坏力 * 防御真空 * 信念真空) ^ (1/3)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_downward_momentum"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '下跌动能' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('scorched_earth_params'), {})
        weights = get_param_value(params.get('fusion_weights'), {'breach_force': 0.4, 'defense_vacuum': 0.3, 'command_vacuum': 0.3})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'pct_change_D', 'vacuum_traversal_efficiency_D', 'dip_absorption_power_D', 'active_buying_support_D',
            'main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'pct_change_D_clip_abs': (signals_data['pct_change_D'].clip(upper=0).abs(), default_weights, True),
            'vacuum_traversal_efficiency_D': (signals_data['vacuum_traversal_efficiency_D'], default_weights, True),
            'dip_absorption_power_D': (signals_data['dip_absorption_power_D'], default_weights, True),
            'active_buying_support_D': (signals_data['active_buying_support_D'], default_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算核心组件 ---
        drop_score = normalized_mtf_scores['pct_change_D_clip_abs']
        efficiency_score = normalized_mtf_scores['vacuum_traversal_efficiency_D']
        breach_force_score = (drop_score * efficiency_score).pow(0.5)
        dip_absorption_score = normalized_mtf_scores['dip_absorption_power_D']
        active_buying_score = normalized_mtf_scores['active_buying_support_D']
        defense_power_score = (dip_absorption_score * 0.5 + active_buying_score * 0.5)
        defense_vacuum_score = (1 - defense_power_score).clip(0, 1)
        positive_conviction_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        command_vacuum_score = (1 - positive_conviction_score).clip(0, 1)
        # --- 4. 最终合成 ---
        downward_momentum_score = (
            (breach_force_score + 1e-9).pow(weights.get('breach_force', 0.4)) *
            (defense_vacuum_score + 1e-9).pow(weights.get('defense_vacuum', 0.3)) *
            (command_vacuum_score + 1e-9).pow(weights.get('command_vacuum', 0.3))
        ).fillna(0.0)
        final_score = downward_momentum_score.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - pct_change_D: {signals_data['pct_change_D'].loc[probe_ts]:.4f}")
            print(f"        - vacuum_traversal_efficiency_D: {signals_data['vacuum_traversal_efficiency_D'].loc[probe_ts]:.4f}")
            print(f"        - dip_absorption_power_D: {signals_data['dip_absorption_power_D'].loc[probe_ts]:.4f}")
            print(f"        - active_buying_support_D: {signals_data['active_buying_support_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 前沿阵地失守 (Breach Force): {breach_force_score.loc[probe_ts]:.4f}")
            print(f"        - 防御工事崩塌 (Defense Vacuum): {defense_vacuum_score.loc[probe_ts]:.4f}")
            print(f"        - 指挥系统溃败 (Command Collapse): {command_vacuum_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '下跌动能'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_offensive_absorption_intent(self, df: pd.DataFrame, lower_shadow_quality: pd.Series, distribution_intent: pd.Series, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V4.0 · Production Ready版】诊断进攻性承接意图。
        - 核心重构: 废弃了对“下影线”形态的路径依赖，引入基于“战役过程”的全新诊断模型。
        - 诊断框架:
          1. 战略前提 (Strategic Prerequisite): 主力无派发意图 (`distribution_intent`)。
          2. 战役背景 (The Crisis): 审判战场抛压的烈度 (`panic_selling_cascade_D`)。
          3. 核心行动 (The Response): 审判多头的反攻力量 (融合 `dip_absorption_power_D` 和 `active_buying_support_D`)。
          4. 司令部意志 (Commander's Will): 审判主力真实信念 (`main_force_conviction_index_D`)。
        - 数学模型: 意图分 = 战略前提 * (背景分 * 行动分 * 意志分) ^ (1/3)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_offensive_absorption_intent"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '进攻性承接意图' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('offensive_absorption_params'), {})
        weights = get_param_value(params.get('fusion_weights'), {'crisis_context': 0.3, 'counter_offensive_force': 0.4, 'commanders_will': 0.3})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'panic_selling_cascade_D', 'dip_absorption_power_D',
            'active_buying_support_D', 'main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'panic_selling_cascade_D': (signals_data['panic_selling_cascade_D'], default_weights, True),
            'dip_absorption_power_D': (signals_data['dip_absorption_power_D'], default_weights, True),
            'active_buying_support_D': (signals_data['active_buying_support_D'], default_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各组件得分 ---
        strategic_prerequisite_score = (1 - distribution_intent).clip(0, 1)
        crisis_context_score = normalized_mtf_scores['panic_selling_cascade_D']
        dip_absorption_score = normalized_mtf_scores['dip_absorption_power_D']
        active_buying_score = normalized_mtf_scores['active_buying_support_D']
        counter_offensive_force_score = (dip_absorption_score * 0.5 + active_buying_score * 0.5)
        commanders_will_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        # --- 4. 最终合成 ---
        base_quality_score = (
            (crisis_context_score + 1e-9).pow(weights.get('crisis_context', 0.3)) *
            (counter_offensive_force_score + 1e-9).pow(weights.get('counter_offensive_force', 0.4)) *
            (commanders_will_score + 1e-9).pow(weights.get('commanders_will', 0.3))
        ).fillna(0.0)
        final_offensive_absorption_intent = (base_quality_score * strategic_prerequisite_score).clip(0, 1)
        final_score = final_offensive_absorption_intent.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - panic_selling_cascade_D: {signals_data['panic_selling_cascade_D'].loc[probe_ts]:.4f}")
            print(f"        - dip_absorption_power_D: {signals_data['dip_absorption_power_D'].loc[probe_ts]:.4f}")
            print(f"        - active_buying_support_D: {signals_data['active_buying_support_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - distribution_intent (来自外部): {distribution_intent.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战略前提 (无派发意图): {strategic_prerequisite_score.loc[probe_ts]:.4f}")
            print(f"        - 战役背景 (Crisis Context): {crisis_context_score.loc[probe_ts]:.4f}")
            print(f"        - 核心行动 (Counter Offensive Force): {counter_offensive_force_score.loc[probe_ts]:.4f}")
            print(f"        - 司令部意志 (Commander's Will): {commanders_will_score.loc[probe_ts]:.4f}")
            print(f"        - 基础品质分: {base_quality_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '进攻性承接意图'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_intraday_bull_control(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V6.3 · 生产就绪版】诊断“日内多头控制力”
        - 核心重构: 废弃V5.1“最后一分钟谎言谬误”模型，引入“战果×过程×叙事”的全新三维诊断框架。
        - 诊断三维度:
          1. 战略位置 (Strategic Position): 评估最终战果，即收盘价相对VWAP的位置。
          2. 过程品质 (Process Quality): 评估全天攻防动作的综合效率与信念。
          3. 叙事诚信度 (Narrative Integrity): 审判结局与过程是否一致，惩罚“尾盘偷袭”等欺骗行为。
        - 数学模型: 最终控制力 = 战略位置分 * (过程品质分 * 叙事诚信度分) ^ 0.5
        - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
        - 【清理】移除所有调试探针代码，恢复生产状态。
        - 【新增】在调试模式下，输出详细的计算过程和中间结果。
        - 【修正】调整最终融合逻辑，避免战略位置中性时分数强制归零。
        """
        method_name = "_diagnose_intraday_bull_control"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '日内多头控制力' @ {probe_ts.strftime('%Y-%m-%d')}")
        debug_info = (is_debug_enabled, probe_ts, method_name)
        params = get_param_value(self.config_params.get('chronos_protocol_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {'process_quality': 0.5, 'narrative_integrity': 0.5})
        top_level_fusion_weights = get_param_value(params.get('top_level_fusion_weights'), {"strategic_position": 0.5, "quality_modulator": 0.5})
        required_signals = [
            'vwap_control_strength_D', 'upward_impulse_purity_D', 'pressure_rejection_strength_D', 'main_force_conviction_index_D',
            'intraday_posture_score_D', 'closing_auction_ambush_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'upward_impulse_purity_D': (signals_data['upward_impulse_purity_D'], tf_weights, True, False), # (series_obj, tf_weights, ascending, is_bipolar)
            'pressure_rejection_strength_D': (signals_data['pressure_rejection_strength_D'], tf_weights, True, False),
            'main_force_conviction_index_D': (signals_data['main_force_conviction_index_D'], tf_weights, True, True), # is_bipolar=True
            'closing_auction_ambush_D': (signals_data['closing_auction_ambush_D'], tf_weights, True, False)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc, is_bipolar_flag) in series_for_mtf_norm_config.items():
            if is_bipolar_flag:
                normalized_mtf_scores[key] = get_adaptive_mtf_normalized_bipolar_score(series_obj, df.index, tf_weights=tf_w, debug_info=False)
            else:
                normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        strategic_position_score = signals_data['vwap_control_strength_D'].clip(-1, 1)
        strategic_position_score_mapped = (strategic_position_score + 1) / 2
        purity_score = normalized_mtf_scores['upward_impulse_purity_D']
        resistance_score = normalized_mtf_scores['pressure_rejection_strength_D']
        conviction_score = normalized_mtf_scores['main_force_conviction_index_D'] # 已经是双极归一化
        process_quality_score = ((purity_score + resistance_score) / 2 * (conviction_score.clip(0,1) + 1) / 2).clip(0, 1)
        narrative_integrity_score = pd.Series(1.0, index=df.index)
        posture_raw = signals_data['intraday_posture_score_D']
        ambush_score = normalized_mtf_scores['closing_auction_ambush_D']
        narrative_deception_score = (ambush_score * (1 - posture_raw.clip(0,1))).clip(0, 1)
        narrative_integrity_score = (1 - narrative_deception_score)
        # --- 4. “时序裁决”三维合成 ---
        quality_modulator = (
            process_quality_score * fusion_weights.get('process_quality', 0.5) +
            narrative_integrity_score * fusion_weights.get('narrative_integrity', 0.5)
        )
        total_top_level_weight = sum(top_level_fusion_weights.values())
        if total_top_level_weight == 0:
            total_top_level_weight = 1.0
        final_control_score = (
            strategic_position_score_mapped * top_level_fusion_weights.get('strategic_position', 0.5) +
            quality_modulator * top_level_fusion_weights.get('quality_modulator', 0.5)
        ) / total_top_level_weight
        final_control_score = final_control_score.clip(0, 1)
        final_score = final_control_score.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            for sig_name in required_signals:
                print(f"        - {sig_name}: {signals_data[sig_name].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战略位置分数 (mapped): {strategic_position_score_mapped.loc[probe_ts]:.4f}")
            print(f"        - 过程品质分数: {process_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 叙事诚信度分数: {narrative_integrity_score.loc[probe_ts]:.4f}")
            print(f"        - 品质调制器: {quality_modulator.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '日内多头控制力'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_deception_index(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V2.6 · 欺骗情境感知与价格上涨中的主力意图背离 - 强化版】诊断博弈欺骗指数
        - 核心重构: 废弃V1.2“结果导向”模型，引入“认知失调 × 欺骗工具强度 × 情境调制”的全新诊断模型。
        - 核心逻辑: 欺骗分 = (主力真实意图向量 - K线表象剧本向量) * 欺骗工具强度 * 情境放大器
                      直接量化“意图”与“表象”的背离程度，并结合欺骗工具的活跃度和市场情境，能识别更高明的欺骗形态。
        - 诊断三要素:
          1. 舞台剧本 (The Apparent Narrative): K线收盘位置讲述的故事 (`closing_strength_index_D`)。
          2. 幕后黑手 (The Hidden Intent): 主力真实的订单流意图 (`main_force_ofi_D`)。
          3. 作案工具 (The Deceptive Tools): 对倒、欺骗等行为 (`deception_lure_long_intensity_D`, `deception_lure_short_intensity_D`, `wash_trade_intensity_D`)，作为放大器。
        - 【新增】情境调制器：根据散户狂热和市场波动性，动态调整欺骗指数。
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        - 核心修复: 修正了 `normalize_to_bipolar` 函数的调用方式，使其符合新的参数签名。
        - **【第四次修改】进一步强化欺骗工具权重，在欺骗强度基础分中直接引入散户狂热，并调整欺骗工具强度的计算逻辑。**
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_deception_index"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '博弈欺骗指数' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('puppeteers_gambit_params'), {})
        k_amplifier = params.get('evidence_amplifier_k', 0.5)
        deception_tool_weights = get_param_value(params.get('deception_tool_weights'), {"lure_long": 0.4, "wash_trade": 0.3, "lure_short": 0.3}) # 新增权重
        context_modulator_params = get_param_value(params.get('context_modulator_params'), {"enabled": True, "fomo_sensitivity": 0.3, "volatility_sensitivity": 0.2, "base_modulator": 1.0}) # 新增情境调制参数
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'closing_strength_index_D', 'main_force_ofi_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'wash_trade_intensity_D', 'retail_fomo_premium_index_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D', # 新增情境信号
            'pct_change_D' # 用于判断是否上涨
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行归一化的 Series 的配置 ---
        series_for_norm_config = {
            'closing_strength_index_D': (signals_data['closing_strength_index_D'], 'bipolar', 55, 1.0, None), # (series_obj, type, window, sensitivity, tf_weights)
            'main_force_ofi_D': (signals_data['main_force_ofi_D'], 'mtf_bipolar', None, None, default_weights),
            'deception_lure_long_intensity_D': (signals_data['deception_lure_long_intensity_D'], 'mtf_norm', None, None, default_weights), # 独立归一化
            'deception_lure_short_intensity_D': (signals_data['deception_lure_short_intensity_D'], 'mtf_norm', None, None, default_weights), # 独立归一化
            'wash_trade_intensity_D': (signals_data['wash_trade_intensity_D'], 'mtf_norm', None, None, default_weights), # 独立归一化
            'retail_fomo_premium_index_D': (signals_data['retail_fomo_premium_index_D'], 'mtf_norm', None, None, default_weights), # 情境调制
            'VOLATILITY_INSTABILITY_INDEX_21d_D': (signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'], 'mtf_norm', None, None, default_weights) # 情境调制
        }
        # 批量计算所有归一化分数
        normalized_scores = {}
        for key, (series_obj, norm_type, window, sensitivity, tf_w) in series_for_norm_config.items():
            if norm_type == 'bipolar':
                normalized_scores[key] = normalize_to_bipolar(series_obj, df.index, windows=window, sensitivity=sensitivity, debug_info=False)
            elif norm_type == 'mtf_bipolar':
                normalized_scores[key] = get_adaptive_mtf_normalized_bipolar_score(series_obj, df.index, tf_weights=tf_w, debug_info=False)
            elif norm_type == 'mtf_norm':
                normalized_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=True, debug_info=False)
        # --- 3. 计算核心组件 ---
        narrative_vector = normalized_scores['closing_strength_index_D']
        intent_vector = normalized_scores['main_force_ofi_D']
        # 认知失调向量：直接体现意图与表象的背离程度
        cognitive_dissonance_vector = (intent_vector - narrative_vector) # 范围 [-1, 1]
        # 欺骗工具强度：加权融合诱多、对倒、诱空强度
        norm_lure_long = normalized_scores['deception_lure_long_intensity_D']
        norm_wash_trade = normalized_scores['wash_trade_intensity_D']
        norm_lure_short = normalized_scores['deception_lure_short_intensity_D']
        deception_tool_strength = pd.Series(0.0, index=df.index, dtype=np.float32)
        # 如果是正向认知失调（意图 > 表象），则关注诱多和对倒
        deception_tool_strength = deception_tool_strength.mask(cognitive_dissonance_vector > 0,
            (norm_lure_long * deception_tool_weights.get('lure_long', 0.4) +
             norm_wash_trade * deception_tool_weights.get('wash_trade', 0.3)) / (deception_tool_weights.get('lure_long', 0.4) + deception_tool_weights.get('wash_trade', 0.3))
        )
        # 如果是负向认知失调（意图 < 表象），则关注诱空和对倒
        deception_tool_strength = deception_tool_strength.mask(cognitive_dissonance_vector < 0,
            (norm_lure_short * deception_tool_weights.get('lure_short', 0.3) +
             norm_wash_trade * deception_tool_weights.get('wash_trade', 0.3)) / (deception_tool_weights.get('lure_short', 0.3) + deception_tool_weights.get('wash_trade', 0.3))
        )
        # 如果认知失调接近0，则只考虑对倒，并给予更高的权重，因为对倒本身就是欺骗
        deception_tool_strength = deception_tool_strength.mask(cognitive_dissonance_vector.abs() < 1e-9, norm_wash_trade * 0.8) # 提高对倒在无明显失调时的权重
        deception_tool_strength = deception_tool_strength.clip(0, 1)
        # 情境调制器：散户狂热和波动性
        context_modulator_factor = pd.Series(context_modulator_params.get('base_modulator', 1.0), index=df.index, dtype=np.float32)
        if context_modulator_params.get('enabled', False):
            norm_fomo = normalized_scores['retail_fomo_premium_index_D']
            norm_volatility = normalized_scores['VOLATILITY_INSTABILITY_INDEX_21d_D']
            # 散户狂热时，欺骗更容易得逞，放大欺骗指数
            fomo_modulator = 1 + norm_fomo * context_modulator_params.get('fomo_sensitivity', 0.3)
            # 波动性低时，欺骗更隐蔽，放大欺骗指数
            volatility_modulator = 1 + (1 - norm_volatility) * context_modulator_params.get('volatility_sensitivity', 0.2)
            context_modulator_factor = context_modulator_factor * fomo_modulator * volatility_modulator
            context_modulator_factor = context_modulator_factor.clip(0.5, 2.0) # 限制调制范围
        # --- 4. 价格上涨中的主力意图背离作为欺骗的直接证据 ---
        is_price_rising = (signals_data['pct_change_D'] > 0.005).astype(float) # 价格显著上涨
        # 主力意图低于K线表象，且价格上涨，这可能是派发欺骗
        price_rising_deception_evidence = pd.Series(0.0, index=df.index, dtype=np.float32)
        price_rising_deception_evidence = price_rising_deception_evidence.mask(
            (is_price_rising > 0.5) & (cognitive_dissonance_vector < 0),
            cognitive_dissonance_vector.abs() * (1 + norm_wash_trade) # 放大对倒强度
        ).clip(0, 1)
        # --- 5. 最终合成：认知失调的绝对值 * 欺骗工具强度 * 情境调制器，并保留方向 ---
        # 欺骗强度基础分：认知失调的绝对值和欺骗工具强度进行加权平均，并加入价格上涨中的主力意图背离和散户狂热
        deception_magnitude_score = (
            cognitive_dissonance_vector.abs() * 0.3 + # 降低认知失调的直接权重
            deception_tool_strength * 0.4 + # 提高欺骗工具的权重
            price_rising_deception_evidence * 0.2 + # 价格上涨中的主力意图背离
            normalized_scores['retail_fomo_premium_index_D'] * 0.1 # 直接引入散户狂热作为欺骗证据
        ).clip(0, 1)
        # 最终欺骗指数 = 欺骗强度基础分 * 情境调制器 * 认知失调的方向
        final_deception_index = (deception_magnitude_score * context_modulator_factor * np.sign(cognitive_dissonance_vector)).clip(-1, 1)
        # 确保在价格上涨时，如果认知失调为负（主力意图低于K线），则视为派发欺骗
        # 这一步在上面的融合中已经通过 np.sign(cognitive_dissonance_vector) 实现了方向性
        # 但为了强调“诱多”的风险，可以进一步强化负向欺骗的绝对值
        final_deception_index = pd.Series(np.where(
            (is_price_rising > 0.5) & (cognitive_dissonance_vector < 0),
            final_deception_index.abs() * -1 * (1 + normalized_scores['retail_fomo_premium_index_D'] * 0.5), # 价格上涨且主力意图背离，放大负向欺骗，尤其在散户狂热时
            final_deception_index
        ), index=df.index).astype(np.float32)
        final_score = final_deception_index.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - closing_strength_index_D: {signals_data['closing_strength_index_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_ofi_D: {signals_data['main_force_ofi_D'].loc[probe_ts]:.4f}")
            print(f"        - deception_lure_long_intensity_D: {signals_data['deception_lure_long_intensity_D'].loc[probe_ts]:.4f}")
            print(f"        - deception_lure_short_intensity_D: {signals_data['deception_lure_short_intensity_D'].loc[probe_ts]:.4f}")
            print(f"        - wash_trade_intensity_D: {signals_data['wash_trade_intensity_D'].loc[probe_ts]:.4f}")
            print(f"        - retail_fomo_premium_index_D: {signals_data['retail_fomo_premium_index_D'].loc[probe_ts]:.4f}")
            print(f"        - VOLATILITY_INSTABILITY_INDEX_21d_D: {signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'].loc[probe_ts]:.4f}")
            print(f"        - pct_change_D: {signals_data['pct_change_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ 2025-12-10:")
            print(f"        - 归一化收盘强度 (Narrative Vector): {narrative_vector.loc[probe_ts]:.4f}")
            print(f"        - 归一化主力OFII (Intent Vector): {intent_vector.loc[probe_ts]:.4f}")
            print(f"        - 认知失调向量: {cognitive_dissonance_vector.loc[probe_ts]:.4f}")
            print(f"        - 欺骗工具强度: {deception_tool_strength.loc[probe_ts]:.4f}")
            print(f"        - 情境调制因子: {context_modulator_factor.loc[probe_ts]:.4f}")
            print(f"        - 价格上涨中的欺骗证据: {price_rising_deception_evidence.loc[probe_ts]:.4f}")
            print(f"        - 欺骗强度基础分: {deception_magnitude_score.loc[probe_ts]:.4f}")
            print(f"        - 是否上涨: {is_price_rising.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '博弈欺骗指数'分数 @ 2025-12-10: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_divergence_quality(self, df: pd.DataFrame, absorption_strength: pd.Series, distribution_intent: pd.Series, states: Dict[str, pd.Series], is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, pd.Series]:
        """
        【V5.2 · Production Ready版 - 背离品质鲁棒融合】诊断高品质价量/价资背离
        - 核心重构: 废弃V4.0“宏观趋势分析”模型，引入“背离深度与广度 × 战略位置 × 双重确认 × 欺骗叙事确认”的全新四维诊断框架。
        - 诊断四维度:
          1. 背离深度与广度 (Divergence Depth & Breadth): 使用斜率更鲁棒地检测价格趋势和主力信念趋势。
          2. 战略位置 (Strategic Location): 评估背离是否发生在绝望区或获利盘不稳定区。
          3. 双重确认 (Dual Confirmation): 由“主力承接/派发”和“微观意图”进行双重印证。
          4. 欺骗叙事确认 (Deceptive Narrative Confirmation): 引入欺骗指数的负向部分，捕捉诱多本质。
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        - **【修正】直接使用 `SCORE_BEHAVIOR_BULLISH_DIVERGENCE` 和 `SCORE_BEHAVIOR_BEARISH_DIVERGENCE` 作为背离深度与广度的输入，避免重复计算和逻辑错误。**
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_divergence_quality"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质价量/价资背离' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('deceptive_divergence_protocol_params'), {})
        # bullish_magnitude_params = get_param_value(params.get('bullish_magnitude_params'), {"price_downtrend_slope_window": 5, "conviction_uptrend_slope_window": 5}) # 废弃
        # bearish_magnitude_params = get_param_value(params.get('bearish_magnitude_params'), {"price_slope_window": 5, "conviction_downtrend_slope_window": 5}) # 废弃
        fusion_weights = get_param_value(params.get('fusion_weights'), {"magnitude": 0.4, "location": 0.3, "bullish_absorption_confirmation": 0.2, "bearish_distribution_confirmation": 0.2, "micro_intent_confirmation": 0.1, "deceptive_narrative_confirmation": 0.1})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'close_D', 'main_force_conviction_index_D', 'loser_pain_index_D', 'winner_stability_index_D',
            'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT', 'SCORE_BEHAVIOR_DECEPTION_INDEX'
        ]
        required_state_signals = [ # 新增对纯行为背离信号的依赖
            'SCORE_BEHAVIOR_BULLISH_DIVERGENCE', 'SCORE_BEHAVIOR_BEARISH_DIVERGENCE'
        ]
        missing_df_signals = [s for s in required_signals if s not in df.columns]
        missing_state_signals = [s for s in required_state_signals if s not in states]
        if missing_df_signals or missing_state_signals:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号。")
                if missing_df_signals: print(f"         - DataFrame中缺失: {missing_df_signals}")
                if missing_state_signals: print(f"         - States中缺失: {missing_state_signals}")
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        state_signals_data = {sig: states[sig] for sig in required_state_signals} # 获取纯行为背离信号
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 2. 获取所有原始数据 ---
        price = signals_data['close_D']
        conviction_raw = signals_data['main_force_conviction_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        winner_stability_raw = signals_data['winner_stability_index_D']
        micro_intent_raw = signals_data['SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT']
        deception_raw = signals_data['SCORE_BEHAVIOR_DECEPTION_INDEX']
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        winner_instability_raw = (1 - winner_stability_raw)
        micro_intent_raw_clip_lower_0 = micro_intent_raw.clip(lower=0)
        micro_intent_raw_clip_upper_0_abs = micro_intent_raw.clip(upper=0).abs()
        deception_raw_clip_upper_0_abs = deception_raw.clip(upper=0).abs() # 只有负向欺骗才作为欺骗叙事确认
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'loser_pain_index_D': (loser_pain_raw, default_weights, True),
            'winner_instability_raw': (winner_instability_raw, default_weights, True), # winner_instability_raw
            'micro_intent_raw_clip_lower_0': (micro_intent_raw_clip_lower_0, default_weights, True), # bullish_micro_intent_confirmation_score
            'micro_intent_raw_clip_upper_0_abs': (micro_intent_raw_clip_upper_0_abs, default_weights, True), # bearish_micro_intent_confirmation_score
            'deception_raw_clip_upper_0_abs': (deception_raw_clip_upper_0_abs, default_weights, True) # deceptive_narrative_confirmation_score
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 牛市背离品质 (Bullish Divergence Quality) ---
        # 直接使用 SCORE_BEHAVIOR_BULLISH_DIVERGENCE 作为背离深度与广度
        bullish_magnitude_score = state_signals_data['SCORE_BEHAVIOR_BULLISH_DIVERGENCE']
        bullish_location_score = normalized_mtf_scores['loser_pain_index_D']
        bullish_absorption_confirmation_score = absorption_strength
        bullish_micro_intent_confirmation_score = normalized_mtf_scores['micro_intent_raw_clip_lower_0']
        bullish_divergence_quality = self._robust_generalized_mean(
            {
                "magnitude": bullish_magnitude_score,
                "location": bullish_location_score,
                "absorption_confirmation": bullish_absorption_confirmation_score,
                "micro_intent_confirmation": bullish_micro_intent_confirmation_score
            },
            {
                "magnitude": fusion_weights.get('magnitude', 0.4),
                "location": fusion_weights.get('location', 0.3),
                "absorption_confirmation": fusion_weights.get('bullish_absorption_confirmation', 0.2),
                "micro_intent_confirmation": fusion_weights.get('micro_intent_confirmation', 0.1)
            },
            df.index,
            power_p=0.0, # 几何平均
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="牛市背离品质融合"
        ).fillna(0.0)
        # --- 4. 熊市背离品质 (Bearish Divergence Quality) ---
        # 直接使用 SCORE_BEHAVIOR_BEARISH_DIVERGENCE 作为背离深度与广度
        bearish_magnitude_score = state_signals_data['SCORE_BEHAVIOR_BEARISH_DIVERGENCE']
        bearish_location_score = normalized_mtf_scores['winner_instability_raw']
        bearish_distribution_confirmation_score = distribution_intent
        bearish_micro_intent_confirmation_score = normalized_mtf_scores['micro_intent_raw_clip_upper_0_abs']
        deceptive_narrative_confirmation_score = normalized_mtf_scores['deception_raw_clip_upper_0_abs']
        bearish_divergence_quality = self._robust_generalized_mean(
            {
                "magnitude": bearish_magnitude_score,
                "location": bearish_location_score,
                "distribution_confirmation": bearish_distribution_confirmation_score,
                "micro_intent_confirmation": bearish_micro_intent_confirmation_score,
                "deceptive_narrative_confirmation": deceptive_narrative_confirmation_score
            },
            {
                "magnitude": fusion_weights.get('magnitude', 0.4),
                "location": fusion_weights.get('location', 0.3),
                "distribution_confirmation": fusion_weights.get('bearish_distribution_confirmation', 0.2),
                "micro_intent_confirmation": fusion_weights.get('micro_intent_confirmation', 0.1),
                "deceptive_narrative_confirmation": fusion_weights.get('deceptive_narrative_confirmation', 0.1)
            },
            df.index,
            power_p=0.0, # 几何平均
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name="熊市背离品质融合"
        ).fillna(0.0)
        bullish_divergence_quality_final_score = bullish_divergence_quality.clip(0, 1).astype(np.float32)
        bearish_divergence_quality_final_score = bearish_divergence_quality.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - close_D: {signals_data['close_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - loser_pain_index_D: {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
            print(f"        - winner_stability_index_D: {signals_data['winner_stability_index_D'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT: {signals_data['SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_DECEPTION_INDEX: {signals_data['SCORE_BEHAVIOR_DECEPTION_INDEX'].loc[probe_ts]:.4f}")
            print(f"        - absorption_strength (来自外部): {absorption_strength.loc[probe_ts]:.4f}")
            print(f"        - distribution_intent (来自外部): {distribution_intent.loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_BULLISH_DIVERGENCE (from states): {state_signals_data['SCORE_BEHAVIOR_BULLISH_DIVERGENCE'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_BEARISH_DIVERGENCE (from states): {state_signals_data['SCORE_BEHAVIOR_BEARISH_DIVERGENCE'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 牛市背离幅度分数: {bullish_magnitude_score.loc[probe_ts]:.4f}")
            print(f"        - 牛市背离位置分数: {bullish_location_score.loc[probe_ts]:.4f}")
            print(f"        - 牛市吸收确认分数: {bullish_absorption_confirmation_score.loc[probe_ts]:.4f}")
            print(f"        - 牛市微观意图确认分数: {bullish_micro_intent_confirmation_score.loc[probe_ts]:.4f}")
            print(f"        - 熊市背离幅度分数: {bearish_magnitude_score.loc[probe_ts]:.4f}")
            print(f"        - 熊市背离位置分数: {bearish_location_score.loc[probe_ts]:.4f}")
            print(f"        - 熊市派发确认分数: {bearish_distribution_confirmation_score.loc[probe_ts]:.4f}")
            print(f"        - 熊市微观意图确认分数: {bearish_micro_intent_confirmation_score.loc[probe_ts]:.4f}")
            print(f"        - 欺骗叙事确认分数: {deceptive_narrative_confirmation_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '牛市背离品质'分数 @ 2025-12-10: {bullish_divergence_quality_final_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '熊市背离品质'分数 @ 2025-12-10: {bearish_divergence_quality_final_score.loc[probe_ts]:.4f}")
        return bullish_divergence_quality_final_score, bearish_divergence_quality_final_score

    def _diagnose_price_overextension(self, df: pd.DataFrame, tf_weights: Dict, long_term_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V2.1 · 生产版】诊断价格过热风险。
        - 核心重构: 废弃了基于“静态热度谬误”和“粗暴音量陷阱”的 V1.0 模型。引入基于
                      “泡沫脆弱度”思想的全新对抗性诊断模型。
        - 核心博弈: 脆弱度 = 内部压力 (市场狂热) / 结构完整性 (主力信念与控制力)
          1. 内部压力 (Internal Pressure): 审判市场狂热的加速度。由 `total_winner_rate_D`,
                                           `ACCEL_5_pct_change_D`, `turnover_rate_f_D` 构成。
          2. 结构完整性 (Structural Integrity): 审判泡沫壁的坚固度。由 `winner_stability_index_D`,
                                                `control_solidity_index_D`, `main_force_conviction_index_D` 构成。
        - 数学模型: 脆弱度分 = 内部压力分 / (结构完整性分 + ε)，并废弃成交量放大器。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_price_overextension"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '价格过热风险' @ {probe_ts.strftime('%Y-%m-%d')}")
        required_signals = [
            'total_winner_rate_D', 'ACCEL_5_pct_change_D', 'turnover_rate_f_D',
            'winner_stability_index_D', 'control_solidity_index_D', 'main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'total_winner_rate_D': (signals_data['total_winner_rate_D'], tf_weights, True),
            'ACCEL_5_pct_change_D': (signals_data['ACCEL_5_pct_change_D'], tf_weights, True),
            'turnover_rate_f_D': (signals_data['turnover_rate_f_D'], tf_weights, True),
            'winner_stability_index_D': (signals_data['winner_stability_index_D'], long_term_weights, True),
            'control_solidity_index_D': (signals_data['control_solidity_index_D'], long_term_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), long_term_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 2. 计算各维度得分 ---
        winner_rate_score = normalized_mtf_scores['total_winner_rate_D']
        price_accel_score = normalized_mtf_scores['ACCEL_5_pct_change_D']
        turnover_score = normalized_mtf_scores['turnover_rate_f_D']
        internal_pressure_score = (winner_rate_score * price_accel_score * turnover_score).pow(1/3)
        winner_stability_score = normalized_mtf_scores['winner_stability_index_D']
        control_solidity_score = normalized_mtf_scores['control_solidity_index_D']
        conviction_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        structural_integrity_score = (winner_stability_score * control_solidity_score * conviction_score).pow(1/3)
        # --- 3. “泡沫脆弱度”合成 ---
        bubble_fragility_score = (internal_pressure_score / (structural_integrity_score + 1e-9)).fillna(0.0)
        final_overextension_score = np.tanh(bubble_fragility_score * 0.5)
        final_score = final_overextension_score.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - total_winner_rate_D: {signals_data['total_winner_rate_D'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_pct_change_D: {signals_data['ACCEL_5_pct_change_D'].loc[probe_ts]:.4f}")
            print(f"        - turnover_rate_f_D: {signals_data['turnover_rate_f_D'].loc[probe_ts]:.4f}")
            print(f"        - winner_stability_index_D: {signals_data['winner_stability_index_D'].loc[probe_ts]:.4f}")
            print(f"        - control_solidity_index_D: {signals_data['control_solidity_index_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 内部压力分数: {internal_pressure_score.loc[probe_ts]:.4f}")
            print(f"        - 结构完整性分数: {structural_integrity_score.loc[probe_ts]:.4f}")
            print(f"        - 泡沫脆弱度分数: {bubble_fragility_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '价格过热风险'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_upward_efficiency(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断高品质上涨效率。
        - 核心重构: 废弃V2.1“皮洛士胜利谬误”模型，引入“战术品质 × 战略地形”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术强攻品质 (The Spearhead's Edge): 沿用V2.1逻辑，评估“矛头”的锋利度。
          2. 战略环境地形 (The Battlefield Terrain): 新增战略评估，审判前方“地形”的阻力。
        - 数学模型: 最终效率分 = 战术品质分 * (1 - 战略阻力分)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_upward_efficiency"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质上涨效率' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('pathfinder_protocol_params'), {})
        resistance_weights = get_param_value(params.get('resistance_weights'), {'chip_fatigue': 0.6, 'loser_pain': 0.4})
        required_signals = [
            'upward_impulse_purity_D', 'impulse_quality_ratio_D',
            'pressure_rejection_strength_D', 'chip_fatigue_index_D',
            'loser_pain_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'upward_impulse_purity_D': (signals_data['upward_impulse_purity_D'], tf_weights, True),
            'impulse_quality_ratio_D': (signals_data['impulse_quality_ratio_D'], tf_weights, True),
            'pressure_rejection_strength_D': (signals_data['pressure_rejection_strength_D'], tf_weights, True),
            'chip_fatigue_index_D': (signals_data['chip_fatigue_index_D'], tf_weights, True),
            'loser_pain_index_D': (signals_data['loser_pain_index_D'], tf_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        purity_score = normalized_mtf_scores['upward_impulse_purity_D']
        offensive_efficiency_score = normalized_mtf_scores['impulse_quality_ratio_D']
        suppression_score = normalized_mtf_scores['pressure_rejection_strength_D']
        tactical_assault_score = (
            (purity_score + 1e-9).pow(0.4) *
            (offensive_efficiency_score + 1e-9).pow(0.3) *
            (suppression_score + 1e-9).pow(0.3)
        ).fillna(0.0)
        chip_fatigue_score = normalized_mtf_scores['chip_fatigue_index_D']
        loser_pain_score = normalized_mtf_scores['loser_pain_index_D']
        strategic_resistance_score = (
            chip_fatigue_score * resistance_weights.get('chip_fatigue', 0.6) +
            loser_pain_score * resistance_weights.get('loser_pain', 0.4)
        ).clip(0, 1)
        strategic_environment_score = (1 - strategic_resistance_score)
        # --- 4. 最终合成：战术品质 × 战略环境 ---
        final_upward_efficiency = (tactical_assault_score * strategic_environment_score).clip(0, 1)
        final_score = final_upward_efficiency.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - upward_impulse_purity_D: {signals_data['upward_impulse_purity_D'].loc[probe_ts]:.4f}")
            print(f"        - impulse_quality_ratio_D: {signals_data['impulse_quality_ratio_D'].loc[probe_ts]:.4f}")
            print(f"        - pressure_rejection_strength_D: {signals_data['pressure_rejection_strength_D'].loc[probe_ts]:.4f}")
            print(f"        - chip_fatigue_index_D: {signals_data['chip_fatigue_index_D'].loc[probe_ts]:.4f}")
            print(f"        - loser_pain_index_D: {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战术强攻品质分数: {tactical_assault_score.loc[probe_ts]:.4f}")
            print(f"        - 战略阻力分数: {strategic_resistance_score.loc[probe_ts]:.4f}")
            print(f"        - 战略环境分数: {strategic_environment_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '高品质上涨效率'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_downward_resistance(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断高品质下跌抵抗。
        - 核心重构: 废弃V2.1“空城计谬误”模型，引入“战术应对 × 战略意图”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术应对能力 (The Tactical Response): 沿用V2.1逻辑，评估防线的坚固度。
          2. 战略欺诈意图 (The Strategic Feint): 新增战略评估，审判抵抗的真实目的。
        - 数学模型: 最终抵抗分 = 战术应对分 * 战略意图分
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_downward_resistance"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质下跌抵抗' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('elastic_defense_params'), {})
        intent_weights = get_param_value(params.get('intent_weights'), {'conviction': 0.6, 'cleansing': 0.4})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'dip_absorption_power_D', 'support_validation_strength_D',
            'active_buying_support_D', 'main_force_conviction_index_D',
            'floating_chip_cleansing_efficiency_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'dip_absorption_power_D': (signals_data['dip_absorption_power_D'], default_weights, True),
            'support_validation_strength_D': (signals_data['support_validation_strength_D'], default_weights, True),
            'active_buying_support_D': (signals_data['active_buying_support_D'], default_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), default_weights, True),
            'floating_chip_cleansing_efficiency_D': (signals_data['floating_chip_cleansing_efficiency_D'], default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        passive_absorption_score = normalized_mtf_scores['dip_absorption_power_D']
        active_defense_score = normalized_mtf_scores['support_validation_strength_D']
        counter_attack_score = normalized_mtf_scores['active_buying_support_D']
        tactical_response_score = (
            (passive_absorption_score + 1e-9).pow(0.2) *
            (active_defense_score + 1e-9).pow(0.4) *
            (counter_attack_score + 1e-9).pow(0.4)
        ).fillna(0.0)
        conviction_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        cleansing_score = normalized_mtf_scores['floating_chip_cleansing_efficiency_D']
        strategic_intent_score = (
            conviction_score * intent_weights.get('conviction', 0.6) +
            cleansing_score * intent_weights.get('cleansing', 0.4)
        ).clip(0, 1)
        # --- 4. 最终合成：战术应对 × 战略意图 ---
        final_downward_resistance = (tactical_response_score * strategic_intent_score).clip(0, 1)
        final_score = final_downward_resistance.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - dip_absorption_power_D: {signals_data['dip_absorption_power_D'].loc[probe_ts]:.4f}")
            print(f"        - support_validation_strength_D: {signals_data['support_validation_strength_D'].loc[probe_ts]:.4f}")
            print(f"        - active_buying_support_D: {signals_data['active_buying_support_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - floating_chip_cleansing_efficiency_D: {signals_data['floating_chip_cleansing_efficiency_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战术应对分数: {tactical_response_score.loc[probe_ts]:.4f}")
            print(f"        - 战略意图分数: {strategic_intent_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '高品质下跌抵抗'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_context_new_high_strength(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Dict[str, pd.Series]:
        """
        【V2.0 · 行为共振增强版】诊断内部上下文信号：新高强度 (CONTEXT_NEW_HIGH_STRENGTH)
        - 核心重构: 废弃V1.2“三要素简单融合”模型，引入“价格动量与品质 × 量能与流动性确认 × 阻力与过热抑制 × 日内控制与情绪共振”的全新四维诊断框架。
        - 核心逻辑: 融合价格行为、量能、市场结构和情绪等多维度行为数据，评估新高的综合质量和可持续性。
        - 【优化】所有组成信号的归一化方式均采用多时间维度自适应归一化。
        - 【探针】加入详细探针，输出原料数据、关键计算节点、结果的值，以便于检查和调试。
        - 【修正】优化 breakout_quality_score_D 的 nan 处理，将其视为质量缺失，赋予0分。
        """
        method_name = "_diagnose_context_new_high_strength"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '新高强度' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('new_high_strength_params'), {})
        debug_info = (is_debug_enabled, probe_ts, method_name)
        fusion_weights = get_param_value(params.get('fusion_weights'), {"price_momentum_quality": 0.3, "volume_liquidity_confirmation": 0.25, "resistance_overextension": 0.25, "intraday_control_sentiment": 0.2})
        price_momentum_quality_weights = get_param_value(params.get('price_momentum_quality_weights'), {"pct_change": 0.25, "ma_slope": 0.25, "breakout_quality": 0.2, "upward_impulse_purity": 0.15, "trend_acceleration": 0.15})
        volume_liquidity_confirmation_weights = get_param_value(params.get('volume_liquidity_confirmation_weights'), {"volume_burstiness": 0.4, "constructive_turnover": 0.3, "buy_sweep_intensity": 0.3})
        resistance_overextension_weights = get_param_value(params.get('resistance_overextension_weights'), {"bias_health": 0.4, "upper_shadow_selling_pressure_inverse": 0.3, "volatility_instability_inverse": 0.3})
        intraday_control_sentiment_weights = get_param_value(params.get('intraday_control_sentiment_weights'), {"intraday_bull_control": 0.5, "market_sentiment": 0.5})
        final_exponent = get_param_value(params.get('final_exponent'), 1.2)
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_tf_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # 获取所有 tf_weights 配置，如果未配置则使用 default_tf_weights
        pct_change_tf_weights = get_param_value(params.get('pct_change_tf_weights'), default_tf_weights)
        ma_slope_tf_weights = get_param_value(params.get('ma_slope_tf_weights'), default_tf_weights)
        bias_health_tf_weights = get_param_value(params.get('bias_health_tf_weights'), default_tf_weights)
        breakout_quality_tf_weights = get_param_value(params.get('breakout_quality_tf_weights'), default_tf_weights)
        upward_impulse_purity_tf_weights = get_param_value(params.get('upward_impulse_purity_tf_weights'), default_tf_weights)
        trend_acceleration_tf_weights = get_param_value(params.get('trend_acceleration_tf_weights'), default_tf_weights)
        volume_burstiness_tf_weights = get_param_value(params.get('volume_burstiness_tf_weights'), default_tf_weights)
        constructive_turnover_tf_weights = get_param_value(params.get('constructive_turnover_tf_weights'), default_tf_weights)
        buy_sweep_intensity_tf_weights = get_param_value(params.get('buy_sweep_intensity_tf_weights'), default_tf_weights)
        upper_shadow_selling_pressure_tf_weights = get_param_value(params.get('upper_shadow_selling_pressure_tf_weights'), default_tf_weights)
        volatility_instability_tf_weights = get_param_value(params.get('volatility_instability_tf_weights'), default_tf_weights)
        intraday_bull_control_tf_weights = get_param_value(params.get('intraday_bull_control_tf_weights'), default_tf_weights)
        market_sentiment_tf_weights = get_param_value(params.get('market_sentiment_tf_weights'), default_tf_weights)
        required_signals = [
            'pct_change_D', 'SLOPE_5_EMA_55_D', 'BIAS_55_D', 'breakout_quality_score_D',
            'upward_impulse_purity_D', 'trend_acceleration_score_D', 'volume_burstiness_index_D',
            'constructive_turnover_ratio_D', 'buy_sweep_intensity_D', 'upper_shadow_selling_pressure_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', 'market_sentiment_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回0分。")
            return {'CONTEXT_NEW_HIGH_STRENGTH': pd.Series(0.0, index=df.index, dtype=np.float32)}
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'pct_change_D_clip': (signals_data['pct_change_D'].clip(lower=0), pct_change_tf_weights, True),
            'SLOPE_5_EMA_55_D_clip': (signals_data['SLOPE_5_EMA_55_D'].clip(lower=0), ma_slope_tf_weights, True),
            'BIAS_55_D_abs': (signals_data['BIAS_55_D'].abs(), bias_health_tf_weights, True), # for bias_health_score
            'breakout_quality_score_D_fillna': (signals_data['breakout_quality_score_D'].fillna(0.0), breakout_quality_tf_weights, True),
            'upward_impulse_purity_D': (signals_data['upward_impulse_purity_D'], upward_impulse_purity_tf_weights, True),
            'trend_acceleration_score_D': (signals_data['trend_acceleration_score_D'], trend_acceleration_tf_weights, True),
            'volume_burstiness_index_D': (signals_data['volume_burstiness_index_D'], volume_burstiness_tf_weights, True),
            'constructive_turnover_ratio_D': (signals_data['constructive_turnover_ratio_D'], constructive_turnover_tf_weights, True),
            'buy_sweep_intensity_D': (signals_data['buy_sweep_intensity_D'], buy_sweep_intensity_tf_weights, True),
            'upper_shadow_selling_pressure_D': (signals_data['upper_shadow_selling_pressure_D'], upper_shadow_selling_pressure_tf_weights, True),
            'VOLATILITY_INSTABILITY_INDEX_21d_D': (signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'], volatility_instability_tf_weights, True),
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL': (signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'], intraday_bull_control_tf_weights, True),
            'market_sentiment_score_D': (signals_data['market_sentiment_score_D'], market_sentiment_tf_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        price_breakthrough_score = normalized_mtf_scores['pct_change_D_clip']
        ma_slope_score = normalized_mtf_scores['SLOPE_5_EMA_55_D_clip']
        breakout_quality_score = normalized_mtf_scores['breakout_quality_score_D_fillna']
        upward_impulse_purity_score = normalized_mtf_scores['upward_impulse_purity_D']
        trend_acceleration_score = normalized_mtf_scores['trend_acceleration_score_D']
        price_momentum_quality_score = (
            (price_breakthrough_score + 1e-9).pow(price_momentum_quality_weights.get('pct_change', 0.25)) *
            (ma_slope_score + 1e-9).pow(price_momentum_quality_weights.get('ma_slope', 0.25)) *
            (breakout_quality_score + 1e-9).pow(price_momentum_quality_weights.get('breakout_quality', 0.2)) *
            (upward_impulse_purity_score + 1e-9).pow(price_momentum_quality_weights.get('upward_impulse_purity', 0.15)) *
            (trend_acceleration_score + 1e-9).pow(price_momentum_quality_weights.get('trend_acceleration', 0.15))
        ).pow(1 / sum(price_momentum_quality_weights.values())).fillna(0.0).clip(0, 1)
        volume_burstiness_score = normalized_mtf_scores['volume_burstiness_index_D']
        constructive_turnover_score = normalized_mtf_scores['constructive_turnover_ratio_D']
        buy_sweep_intensity_score = normalized_mtf_scores['buy_sweep_intensity_D']
        volume_liquidity_confirmation_score = (
            (volume_burstiness_score + 1e-9).pow(volume_liquidity_confirmation_weights.get('volume_burstiness', 0.4)) *
            (constructive_turnover_score + 1e-9).pow(volume_liquidity_confirmation_weights.get('constructive_turnover', 0.3)) *
            (buy_sweep_intensity_score + 1e-9).pow(volume_liquidity_confirmation_weights.get('buy_sweep_intensity', 0.3))
        ).pow(1 / sum(volume_liquidity_confirmation_weights.values())).fillna(0.0).clip(0, 1)
        bias_health_score = (1 - normalized_mtf_scores['BIAS_55_D_abs']).clip(0, 1)
        upper_shadow_selling_pressure_inverse_score = (1 - normalized_mtf_scores['upper_shadow_selling_pressure_D']).clip(0, 1)
        volatility_instability_inverse_score = (1 - normalized_mtf_scores['VOLATILITY_INSTABILITY_INDEX_21d_D']).clip(0, 1)
        resistance_overextension_score = (
            (bias_health_score + 1e-9).pow(resistance_overextension_weights.get('bias_health', 0.4)) *
            (upper_shadow_selling_pressure_inverse_score + 1e-9).pow(resistance_overextension_weights.get('upper_shadow_selling_pressure_inverse', 0.3)) *
            (volatility_instability_inverse_score + 1e-9).pow(resistance_overextension_weights.get('volatility_instability_inverse', 0.3))
        ).pow(1 / sum(resistance_overextension_weights.values())).fillna(0.0).clip(0, 1)
        intraday_bull_control_score = normalized_mtf_scores['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL']
        market_sentiment_score = normalized_mtf_scores['market_sentiment_score_D']
        intraday_control_sentiment_score = (
            (intraday_bull_control_score + 1e-9).pow(intraday_control_sentiment_weights.get('intraday_bull_control', 0.5)) *
            (market_sentiment_score + 1e-9).pow(intraday_control_sentiment_weights.get('market_sentiment', 0.5))
        ).pow(1 / sum(intraday_control_sentiment_weights.values())).fillna(0.0).clip(0, 1)
        # --- 4. 最终融合 (加权几何平均) ---
        new_high_strength = (
            (price_momentum_quality_score + 1e-9).pow(fusion_weights.get('price_momentum_quality', 0.3)) *
            (volume_liquidity_confirmation_score + 1e-9).pow(fusion_weights.get('volume_liquidity_confirmation', 0.25)) *
            (resistance_overextension_score + 1e-9).pow(fusion_weights.get('resistance_overextension', 0.25)) *
            (intraday_control_sentiment_score + 1e-9).pow(fusion_weights.get('intraday_control_sentiment', 0.2))
        ).pow(1 / sum(fusion_weights.values())).fillna(0.0).clip(0, 1)
        final_new_high_strength = new_high_strength.pow(final_exponent)
        final_score = final_new_high_strength.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            for sig_name in required_signals:
                print(f"        - {sig_name}: {signals_data[sig_name].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 价格动量品质分数: {price_momentum_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 量能流动性确认分数: {volume_liquidity_confirmation_score.loc[probe_ts]:.4f}")
            print(f"        - 阻力过热抑制分数: {resistance_overextension_score.loc[probe_ts]:.4f}")
            print(f"        - 日内控制情绪分数: {intraday_control_sentiment_score.loc[probe_ts]:.4f}")
            print(f"        - 基础新高强度: {new_high_strength.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '新高强度'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return {'CONTEXT_NEW_HIGH_STRENGTH': final_score}

    def _diagnose_microstructure_intent(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Dict[str, pd.Series]:
        """
        【V2.1 · Production Ready版】微观结构意图诊断引擎
        - 核心重构: 废弃V1.3“静态快照”模型，引入“核心意图强度 × 环境适应性 × 行为一致性”的全新三维诊断框架。
        - 诊断三维度:
          1. 核心意图强度 (Core Intent Magnitude): 诊断主力订单流和挂单枯竭的原始意图。
          2. 环境适应性 (Environmental Adaptability): 根据市场波动率和流动性校准意图的显著性。
          3. 行为一致性 (Behavioral Coherence): 通过价格脉冲纯度、欺诈指数印证意图的真实性。
        - 数学模型: 最终微观意图 = 核心意图强度 × 环境适应性因子 × 行为一致性因子
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_microstructure_intent"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '微观结构意图' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('fog_of_war_protocol_params'), {})
        core_intent_weights = get_param_value(params.get('core_intent_weights'), {"ofi": 0.6, "quote_exhaustion": 0.4})
        env_adapt_weights = get_param_value(params.get('environmental_adaptability_weights'), {"volatility_sensitivity": 0.5, "liquidity_sensitivity": 0.5})
        behavior_coherence_weights = get_param_value(params.get('behavioral_coherence_weights'), {"impulse_purity": 0.7, "deception_penalty": 0.3})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        # --- 2. 维度一：核心意图强度 (Core Intent Magnitude) ---
        required_signals_core = ['main_force_ofi_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D']
        required_signals_env = ['ATR_14_D', 'volume_ratio_D']
        required_signals_behavior = ['upward_impulse_purity_D', 'vacuum_traversal_efficiency_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D']
        required_signals = required_signals_core + required_signals_env + required_signals_behavior
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回默认值。")
            return {'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT': pd.Series(0.0, index=df.index)}
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        deception_lure_combined_raw = pd.concat([signals_data['deception_lure_long_intensity_D'], signals_data['deception_lure_short_intensity_D']], axis=1).max(axis=1)
        # --- 收集所有需要进行多时间框架归一化的 Series ---
        series_for_mtf_norm_map = {
            id(signals_data['main_force_ofi_D']): (signals_data['main_force_ofi_D'], default_weights, True, True), # (series_obj, tf_weights, ascending, is_bipolar)
            id(signals_data['buy_quote_exhaustion_rate_D']): (signals_data['buy_quote_exhaustion_rate_D'], default_weights, True, False),
            id(signals_data['sell_quote_exhaustion_rate_D']): (signals_data['sell_quote_exhaustion_rate_D'], default_weights, True, False),
            id(signals_data['ATR_14_D']): (signals_data['ATR_14_D'], default_weights, True, False),
            id(signals_data['volume_ratio_D']): (signals_data['volume_ratio_D'], default_weights, True, False),
            id(signals_data['upward_impulse_purity_D']): (signals_data['upward_impulse_purity_D'], default_weights, True, False),
            id(signals_data['vacuum_traversal_efficiency_D']): (signals_data['vacuum_traversal_efficiency_D'], default_weights, True, False),
            id(deception_lure_combined_raw): (deception_lure_combined_raw, default_weights, True, False) # 使用预计算的 Series
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for series_id, (series_obj, tf_w, asc, is_bipolar_flag) in series_for_mtf_norm_map.items():
            if is_bipolar_flag:
                normalized_mtf_scores[series_id] = get_adaptive_mtf_normalized_bipolar_score(series_obj, df.index, tf_weights=tf_w, debug_info=False)
            else:
                normalized_mtf_scores[series_id] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 维度一：核心意图强度 (Core Intent Magnitude) ---
        ofi_score = normalized_mtf_scores[id(signals_data['main_force_ofi_D'])]
        buy_sweep_score = normalized_mtf_scores[id(signals_data['buy_quote_exhaustion_rate_D'])]
        sell_sweep_score = normalized_mtf_scores[id(signals_data['sell_quote_exhaustion_rate_D'])]
        bullish_core_intent = (ofi_score.clip(lower=0) * core_intent_weights.get('ofi', 0.6) + buy_sweep_score * core_intent_weights.get('quote_exhaustion', 0.4))
        bearish_core_intent = (ofi_score.clip(upper=0).abs() * core_intent_weights.get('ofi', 0.6) + sell_sweep_score * core_intent_weights.get('quote_exhaustion', 0.4))
        core_intent_magnitude = (bullish_core_intent - bearish_core_intent).clip(-1, 1)
        # --- 4. 维度二：环境适应性 (Environmental Adaptability) ---
        environmental_adaptability_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if all(s in signals_data for s in required_signals_env):
            volatility_score = normalized_mtf_scores[id(signals_data['ATR_14_D'])]
            liquidity_score = (1 - normalized_mtf_scores[id(signals_data['volume_ratio_D'])]) # 流动性越低，适应性越强
            environmental_adaptability_factor_raw = (
                volatility_score * env_adapt_weights.get('volatility_sensitivity', 0.5) +
                liquidity_score * env_adapt_weights.get('liquidity_sensitivity', 0.5)
            ).clip(0, 1)
            environmental_adaptability_factor = 0.5 + environmental_adaptability_factor_raw * 0.5 # 映射到 [0.5, 1.0]
        # --- 5. 维度三：行为一致性 (Behavioral Coherence) ---
        behavioral_coherence_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if all(s in signals_data for s in required_signals_behavior):
            upward_purity_score = normalized_mtf_scores[id(signals_data['upward_impulse_purity_D'])]
            downward_purity_score = normalized_mtf_scores[id(signals_data['vacuum_traversal_efficiency_D'])]
            # 使用预计算的 Series ID 进行检索
            deception_score = normalized_mtf_scores[id(deception_lure_combined_raw)]
            purity_coherence = pd.Series(np.where(
                core_intent_magnitude > 0, # 看涨意图
                upward_purity_score * (1 - downward_purity_score), # 上涨纯度高，下跌纯度低
                np.where(
                    core_intent_magnitude < 0, # 看跌意图
                    downward_purity_score * (1 - upward_purity_score), # 下跌纯度高，上涨纯度低
                    0.5 # 中性
                )
            ), index=df.index)
            deception_penalty_factor = (1 - deception_score * behavior_coherence_weights.get('deception_penalty', 0.3)).clip(0, 1)
            behavioral_coherence_factor_raw = (
                purity_coherence * behavior_coherence_weights.get('impulse_purity', 0.7) +
                deception_penalty_factor * (1 - behavior_coherence_weights.get('impulse_purity', 0.7))
            ).clip(0, 1)
            behavioral_coherence_factor = 0.5 + behavioral_coherence_factor_raw * 0.5 # 映射到 [0.5, 1.0]
        # --- 6. 最终合成：三维融合 ---
        final_micro_intent = (core_intent_magnitude * environmental_adaptability_factor * behavioral_coherence_factor).clip(-1, 1)
        final_score = final_micro_intent.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            for sig_name in required_signals:
                print(f"        - {sig_name}: {signals_data[sig_name].loc[probe_ts]:.4f}")
            print(f"        - deception_lure_combined_raw: {deception_lure_combined_raw.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 看涨核心意图: {bullish_core_intent.loc[probe_ts]:.4f}")
            print(f"        - 看跌核心意图: {bearish_core_intent.loc[probe_ts]:.4f}")
            print(f"        - 核心意图强度: {core_intent_magnitude.loc[probe_ts]:.4f}")
            print(f"        - 环境适应性因子: {environmental_adaptability_factor.loc[probe_ts]:.4f}")
            print(f"        - 行为一致性因子: {behavioral_coherence_factor.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '微观结构意图'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        states = {'SCORE_BEHAVIOR_MICROSTRUCTURE_INTENT': final_score}
        return states

    def _diagnose_stagnation_evidence(self, df: pd.DataFrame, upward_efficiency: pd.Series, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V4.1 · 生产版】诊断内部行为信号：滞涨证据
        - 核心重构: 废弃了基于“战术僵化”的 V3.9 模型。引入基于“信念危机”思想的全新
                      双维度诊断模型，旨在区分“良性蓄势”与“恶性派发”的滞涨。
        - 信念危机双维度:
          1. 微观战局僵持 (Micro-Battlefield Stalemate): 审判前线战况的胶着程度。
          2. 宏观信念动摇 (Macro-Conviction Erosion): 审判主力司令部的真实意图与筹码结构的稳定性。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_stagnation_evidence"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '滞涨证据' @ {probe_ts.strftime('%Y-%m-%d')}")
        df_index = df.index
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        p_thresholds = get_param_value(self.config_params.get('neutral_zone_thresholds'), {})
        alpha_threshold = get_param_value(p_thresholds.get('main_force_execution_alpha_D'), 0.0)
        required_signals = [
            'pct_change_D', 'ACCEL_5_pct_change_D', 'chip_fatigue_index_D',
            'rally_distribution_pressure_D', 'upper_shadow_selling_pressure_D',
            'main_force_execution_alpha_D', 'total_winner_rate_D',
            'SLOPE_5_main_force_conviction_index_D', 'SLOPE_5_winner_stability_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 1. 获取原始数据 ---
        pct_change = signals_data['pct_change_D']
        price_accel = signals_data['ACCEL_5_pct_change_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        rally_pressure_raw = signals_data['rally_distribution_pressure_D'].clip(lower=0)
        upper_shadow_pressure_raw = signals_data['upper_shadow_selling_pressure_D']
        mf_alpha_raw = signals_data['main_force_execution_alpha_D']
        winner_rate_raw = signals_data['total_winner_rate_D']
        conviction_slope_raw = signals_data['SLOPE_5_main_force_conviction_index_D']
        winner_stability_slope_raw = signals_data['SLOPE_5_winner_stability_index_D']
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'price_accel_clip_abs': (price_accel.clip(upper=0).abs(), default_weights, True),
            'chip_fatigue_index_D': (chip_fatigue_raw, default_weights, True),
            'rally_distribution_pressure_D_clip': (rally_pressure_raw, default_weights, True),
            'upper_shadow_selling_pressure_D': (upper_shadow_pressure_raw, default_weights, True),
            'mf_alpha_filtered_clip_abs': (self._apply_neutral_zone_filter(mf_alpha_raw, alpha_threshold).clip(upper=0).abs(), default_weights, True),
            'total_winner_rate_D': (winner_rate_raw, default_weights, True),
            'conviction_slope_raw_clip_abs': (conviction_slope_raw.clip(upper=0).abs(), default_weights, True),
            'winner_stability_slope_raw_clip_abs': (winner_stability_slope_raw.clip(upper=0).abs(), default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 2. 维度一：微观战局僵持 (Micro-Battlefield Stalemate) ---
        inefficiency_score = (1 - upward_efficiency).clip(0, 1)
        momentum_decay_score = normalized_mtf_scores['price_accel_clip_abs']
        chip_fatigue_score = normalized_mtf_scores['chip_fatigue_index_D']
        bullish_exhaustion_score = (inefficiency_score * momentum_decay_score * chip_fatigue_score).pow(1/3).fillna(0.0)
        rally_pressure_score = normalized_mtf_scores['rally_distribution_pressure_D_clip']
        upper_shadow_score = normalized_mtf_scores['upper_shadow_selling_pressure_D']
        mf_distribution_evidence = normalized_mtf_scores['mf_alpha_filtered_clip_abs']
        bearish_ambush_score = (rally_pressure_score * upper_shadow_score * mf_distribution_evidence).pow(1/3).fillna(0.0)
        total_energy = (bullish_exhaustion_score + bearish_ambush_score) / 2
        balance_factor = 1 - (bullish_exhaustion_score - bearish_ambush_score).abs()
        micro_stalemate_score = (total_energy * balance_factor).fillna(0.0)
        # --- 3. 维度二：宏观信念动摇 (Macro-Conviction Erosion) ---
        profit_pressure_score = normalized_mtf_scores['total_winner_rate_D']
        conviction_decay_score = normalized_mtf_scores['conviction_slope_raw_clip_abs']
        instability_score = normalized_mtf_scores['winner_stability_slope_raw_clip_abs']
        macro_erosion_score = (profit_pressure_score * conviction_decay_score * instability_score).pow(1/3).fillna(0.0)
        # --- 4. 最终合成 ---
        stagnation_evidence = (micro_stalemate_score * 0.6 + macro_erosion_score * 0.4)
        is_rising_or_flat = (pct_change >= -0.005).astype(float)
        final_stagnation_evidence = (stagnation_evidence * is_rising_or_flat).clip(0, 1)
        final_score = final_stagnation_evidence.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - pct_change_D: {signals_data['pct_change_D'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_pct_change_D: {signals_data['ACCEL_5_pct_change_D'].loc[probe_ts]:.4f}")
            print(f"        - chip_fatigue_index_D: {signals_data['chip_fatigue_index_D'].loc[probe_ts]:.4f}")
            print(f"        - rally_distribution_pressure_D: {signals_data['rally_distribution_pressure_D'].loc[probe_ts]:.4f}")
            print(f"        - upper_shadow_selling_pressure_D: {signals_data['upper_shadow_selling_pressure_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_execution_alpha_D: {signals_data['main_force_execution_alpha_D'].loc[probe_ts]:.4f}")
            print(f"        - total_winner_rate_D: {signals_data['total_winner_rate_D'].loc[probe_ts]:.4f}")
            print(f"        - SLOPE_5_main_force_conviction_index_D: {signals_data['SLOPE_5_main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - SLOPE_5_winner_stability_index_D: {signals_data['SLOPE_5_winner_stability_index_D'].loc[probe_ts]:.4f}")
            print(f"        - upward_efficiency (来自外部): {upward_efficiency.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 微观战局僵持分数: {micro_stalemate_score.loc[probe_ts]:.4f}")
            print(f"        - 宏观信念动摇分数: {macro_erosion_score.loc[probe_ts]:.4f}")
            print(f"        - 价格上涨或平盘: {is_rising_or_flat.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '滞涨证据'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_lower_shadow_quality(self, df: pd.DataFrame, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V13.0 · Production Ready版】诊断下影线承接品质。
        - 核心重构: 废弃V12.1“战地记者”模型，引入“剧本×表演×意图”的全新三幕式诊断框架。
        - 诊断三幕剧:
          1. 剧本 (The Script): 审判“危机”的真实性与烈度 (`panic_selling_cascade_D`)。
          2. 表演 (The Performance): 审判“主角”救场的完成度 (融合 `active_buying_support_D` 等)。
          3. 意图 (The Intent): 审判“导演”的真实内心独白 (融合 `main_force_conviction_index_D` 等)。
        - 数学模型: 品质分 = (剧本品质 * 表演品质) ^ 0.5 * 导演意图分
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_lower_shadow_quality"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '下影线承接品质' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('directors_cut_params'), {})
        intent_weights = get_param_value(params.get('intent_weights'), {'conviction': 0.7, 'covert_ops': 0.3})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'panic_selling_cascade_D', 'active_buying_support_D',
            'dip_absorption_power_D', 'main_force_conviction_index_D',
            'covert_accumulation_signal_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'panic_selling_cascade_D': (signals_data['panic_selling_cascade_D'], default_weights, True),
            'active_buying_support_D': (signals_data['active_buying_support_D'], default_weights, True),
            'dip_absorption_power_D': (signals_data['dip_absorption_power_D'], default_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), default_weights, True),
            'covert_accumulation_signal_D': (signals_data['covert_accumulation_signal_D'], default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各幕得分 ---
        script_quality_score = normalized_mtf_scores['panic_selling_cascade_D']
        performance_active_score = normalized_mtf_scores['active_buying_support_D']
        performance_dip_score = normalized_mtf_scores['dip_absorption_power_D']
        performance_quality_score = (performance_active_score * 0.6 + performance_dip_score * 0.4)
        intent_conviction_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        intent_covert_ops_score = normalized_mtf_scores['covert_accumulation_signal_D']
        directors_intent_score = (
            intent_conviction_score * intent_weights.get('conviction', 0.7) +
            intent_covert_ops_score * intent_weights.get('covert_ops', 0.3)
        ).clip(0, 1)
        # --- 4. 最终合成 ---
        base_drama_quality = (script_quality_score * performance_quality_score).pow(0.5).fillna(0.0)
        final_lower_shadow_quality = (base_drama_quality * directors_intent_score).clip(0, 1)
        final_score = final_lower_shadow_quality.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - panic_selling_cascade_D: {signals_data['panic_selling_cascade_D'].loc[probe_ts]:.4f}")
            print(f"        - active_buying_support_D: {signals_data['active_buying_support_D'].loc[probe_ts]:.4f}")
            print(f"        - dip_absorption_power_D: {signals_data['dip_absorption_power_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - covert_accumulation_signal_D: {signals_data['covert_accumulation_signal_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 剧本品质分数: {script_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 表演品质分数: {performance_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 导演意图分数: {directors_intent_score.loc[probe_ts]:.4f}")
            print(f"        - 基础戏剧品质: {base_drama_quality.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '下影线承接品质'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_distribution_intent(self, df: pd.DataFrame, tf_weights: Dict, overextension_raw: pd.Series, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V7.0 · Production Ready版】诊断派发意图。
        - 核心重构: 废弃V6.0“战术绝对主义”乘法模型，引入“双轨独立审判”框架。
        - 诊断双轨:
          1. 战术风险 (Tactical Risk): 评估主动派发动作，即“风暴”强度。
          2. 战略风险 (Strategic Risk): 评估战场环境恶化，即“大气压”读数。
        - 数学模型: 最终风险 = max(战术风险, 战略风险) * (1 + 协同奖励)
        - 升级说明: 增加了详细探针，用于调试和检查每一步计算。
        """
        method_name = "_diagnose_distribution_intent"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '派发意图' @ {probe_ts.strftime('%Y-%m-%d')}")
        # 从 self.config_params 获取 atmospheric_pressure_params
        params = get_param_value(self.config_params.get('atmospheric_pressure_params'), {})
        synergy_bonus = get_param_value(params.get('synergy_bonus_factor'), 0.2)
        # 修正：定义 weights 变量
        weights = get_param_value(params.get('fusion_weights'), {'motive': 0.2, 'weapon': 0.4, 'fingerprint': 0.4}) # 确保这里获取的是正确的融合权重
        # 从 self.config_params 获取 judgment_day_protocol_params
        env_params = get_param_value(self.config_params.get('judgment_day_protocol_params'), {})
        env_weights = get_param_value(env_params.get('environment_weights'), {'fatigue': 0.4, 'decay': 0.3, 'betrayal': 0.3})
        # 从 self.config_params 获取 mtf_normalization_params
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 2. 轨道一：战术风险评估 (风暴强度) ---
        required_signals = [
            'profit_taking_flow_ratio_D', 'rally_distribution_pressure_D',
            'upper_shadow_selling_pressure_D', 'main_force_execution_alpha_D',
            'trend_vitality_index_D', 'winner_stability_index_D',
            'control_solidity_index_D', 'SLOPE_5_main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'profit_taking_flow_ratio_D': (signals_data['profit_taking_flow_ratio_D'], default_weights, True),
            'rally_distribution_pressure_D': (signals_data['rally_distribution_pressure_D'], default_weights, True),
            'upper_shadow_selling_pressure_D': (signals_data['upper_shadow_selling_pressure_D'], default_weights, True),
            'main_force_execution_alpha_D_clip_abs': (signals_data['main_force_execution_alpha_D'].clip(upper=0).abs(), default_weights, True),
            'trend_vitality_index_D': (signals_data['trend_vitality_index_D'], default_weights, True),
            'winner_stability_index_D': (signals_data['winner_stability_index_D'], default_weights, True),
            'control_solidity_index_D': (signals_data['control_solidity_index_D'], default_weights, True),
            'SLOPE_5_main_force_conviction_index_D_clip_abs': (signals_data['SLOPE_5_main_force_conviction_index_D'].clip(upper=0).abs(), default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        motive_score = normalized_mtf_scores['profit_taking_flow_ratio_D']
        rally_pressure_score = normalized_mtf_scores['rally_distribution_pressure_D']
        upper_shadow_score = normalized_mtf_scores['upper_shadow_selling_pressure_D']
        weapon_score = (rally_pressure_score * 0.5 + upper_shadow_score * 0.5)
        fingerprint_score = normalized_mtf_scores['main_force_execution_alpha_D_clip_abs']
        tactical_risk_score = (
            (motive_score + 1e-9).pow(weights.get('motive', 0.2)) *
            (weapon_score + 1e-9).pow(weights.get('weapon', 0.4)) *
            (fingerprint_score + 1e-9).pow(weights.get('fingerprint', 0.4))
        ).fillna(0.0)
        # --- 3. 轨道二：战略风险评估 (大气压读数) ---
        vitality_score = normalized_mtf_scores['trend_vitality_index_D']
        bullish_fatigue_score = ((1 - vitality_score) * overextension_raw).pow(0.5)
        stability_score = normalized_mtf_scores['winner_stability_index_D']
        solidity_score = normalized_mtf_scores['control_solidity_index_D']
        fortress_decay_score = ((1 - stability_score) * (1 - solidity_score)).pow(0.5)
        commanders_betrayal_score = normalized_mtf_scores['SLOPE_5_main_force_conviction_index_D_clip_abs']
        strategic_risk_score = (
            bullish_fatigue_score * env_weights.get('fatigue', 0.4) +
            fortress_decay_score * env_weights.get('decay', 0.3) +
            commanders_betrayal_score * env_weights.get('betrayal', 0.3)
        ).clip(0, 1)
        # --- 4. 最终合成：双轨独立审判 ---
        base_risk = pd.concat([tactical_risk_score, strategic_risk_score], axis=1).max(axis=1)
        synergy_amplifier = 1 + (tactical_risk_score * strategic_risk_score).pow(0.5) * synergy_bonus
        final_distribution_intent = (base_risk * synergy_amplifier).clip(0, 1)
        final_score = final_distribution_intent.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - profit_taking_flow_ratio_D: {signals_data['profit_taking_flow_ratio_D'].loc[probe_ts]:.4f}")
            print(f"        - rally_distribution_pressure_D: {signals_data['rally_distribution_pressure_D'].loc[probe_ts]:.4f}")
            print(f"        - upper_shadow_selling_pressure_D: {signals_data['upper_shadow_selling_pressure_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_execution_alpha_D: {signals_data['main_force_execution_alpha_D'].loc[probe_ts]:.4f}")
            print(f"        - trend_vitality_index_D: {signals_data['trend_vitality_index_D'].loc[probe_ts]:.4f}")
            print(f"        - winner_stability_index_D: {signals_data['winner_stability_index_D'].loc[probe_ts]:.4f}")
            print(f"        - control_solidity_index_D: {signals_data['control_solidity_index_D'].loc[probe_ts]:.4f}")
            print(f"        - SLOPE_5_main_force_conviction_index_D: {signals_data['SLOPE_5_main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - overextension_raw (来自外部): {overextension_raw.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战术风险分数: {tactical_risk_score.loc[probe_ts]:.4f}")
            print(f"        - 战略风险分数: {strategic_risk_score.loc[probe_ts]:.4f}")
            print(f"        - 基础风险 (max): {base_risk.loc[probe_ts]:.4f}")
            print(f"        - 协同放大器: {synergy_amplifier.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '派发意图'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_ambush_counterattack(self, df: pd.DataFrame, offensive_absorption_intent: pd.Series, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V5.2 · 归一化函数调用修复版】诊断伏击式反攻信号。
        - 核心重构: 引入“脆弱战场 × 幽灵诡计 × 突袭品质”的全新三维诊断框架。
        - 诊断三维度:
          1. 脆弱战场 (Vulnerable Battlefield): 评估市场先前的脆弱性（恐慌、短期下跌趋势/价格停滞、亏损盘痛苦）。
          2. 幽灵诡计 (Phantom Trick): 评估主力承接的强度及其看涨欺骗性（制造弱势假象）。
          3. 突袭品质 (Strike Quality): 评估反攻的有效性与纯粹性（收盘强度、上涨脉冲纯度）。
        - 数学模型: 伏击反攻分 = (脆弱战场分^W1 * 幽灵诡计分^W2 * 突袭品质分^W3)
        - 【调优】原始指标deception_index被拆分为deception_lure_long_intensity、deception_lure_short_intensity，本方法已更新以利用这两个更精细的指标。
        - 核心修复: 修正了 `normalize_score` 函数的调用方式，使其符合新的参数签名。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_ambush_counterattack"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '伏击式反攻' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('ambush_counterattack_params'), {})
        fusion_weights = get_param_value(params.get('fusion_weights'), {"context": 0.3, "action": 0.4, "quality": 0.3})
        context_weights = get_param_value(params.get('context_weights'), {"panic": 0.3, "prior_weakness_slope": 0.4, "loser_pain": 0.2, "price_stagnation": 0.1})
        prior_weakness_slope_window = get_param_value(params.get('prior_weakness_slope_window'), 5)
        price_stagnation_params = get_param_value(params.get('price_stagnation_params'), {"slope_window": 5, "max_abs_slope_threshold": 0.005, "max_bbw_score_threshold": 0.3})
        action_weights = get_param_value(params.get('action_weights'), {"absorption": 0.6, "deception_positive": 0.4})
        quality_weights = get_param_value(params.get('quality_weights'), {"closing_strength": 0.6, "upward_purity": 0.4})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'panic_selling_cascade_D', f'SLOPE_{prior_weakness_slope_window}_close_D', 'loser_pain_index_D',
            'closing_strength_index_D', 'upward_impulse_purity_D',
            f'SLOPE_{price_stagnation_params.get("slope_window", 5)}_close_D', 'BBW_21_2.0_D',
            'SCORE_BEHAVIOR_DECEPTION_INDEX' # 从 states 获取，但为了校验也列出
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        stagnation_slope_raw = signals_data[f'SLOPE_{price_stagnation_params.get("slope_window", 5)}_close_D']
        bbw_score_raw = signals_data['BBW_21_2.0_D']
        deception_index_clip_abs_raw = signals_data['SCORE_BEHAVIOR_DECEPTION_INDEX'].clip(upper=0).abs()
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'panic_selling_cascade_D': (signals_data['panic_selling_cascade_D'], default_weights, True),
            f'SLOPE_{prior_weakness_slope_window}_close_D_clip_abs': (signals_data[f'SLOPE_{prior_weakness_slope_window}_close_D'].clip(upper=0).abs(), default_weights, True),
            'loser_pain_index_D': (signals_data['loser_pain_index_D'], default_weights, True),
            'BBW_21_2.0_D': (bbw_score_raw, default_weights, False), # For bbw_score
            'SCORE_BEHAVIOR_DECEPTION_INDEX_clip_abs': (deception_index_clip_abs_raw, default_weights, True), # For deceptive_narrative_score
            'upward_impulse_purity_D': (signals_data['upward_impulse_purity_D'], default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 维度一：脆弱战场 (Vulnerable Battlefield) ---
        panic_score = normalized_mtf_scores['panic_selling_cascade_D']
        prior_weakness_score = normalized_mtf_scores[f'SLOPE_{prior_weakness_slope_window}_close_D_clip_abs']
        loser_pain_score = normalized_mtf_scores['loser_pain_index_D']
        bbw_score = normalized_mtf_scores['BBW_21_2.0_D']
        price_stagnation_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        price_stagnation_condition = (stagnation_slope_raw.abs() < price_stagnation_params.get("max_abs_slope_threshold", 0.005)) & \
                                     (bbw_score > (1 - price_stagnation_params.get("max_bbw_score_threshold", 0.3)))
        price_stagnation_score[price_stagnation_condition] = (bbw_score[price_stagnation_condition] * (1 - stagnation_slope_raw.abs()[price_stagnation_condition] / price_stagnation_params.get("max_abs_slope_threshold", 0.005))).clip(0,1)
        ambush_context_score = (
            (panic_score + 1e-9).pow(context_weights.get('panic', 0.3)) *
            (prior_weakness_score + 1e-9).pow(context_weights.get('prior_weakness_slope', 0.4)) *
            (loser_pain_score + 1e-9).pow(context_weights.get('loser_pain', 0.2)) *
            (price_stagnation_score + 1e-9).pow(context_weights.get('price_stagnation', 0.1))
        ).pow(1/(context_weights.get('panic', 0.3) + context_weights.get('prior_weakness_slope', 0.4) + context_weights.get('loser_pain', 0.2) + context_weights.get('price_stagnation', 0.1))).fillna(0.0)
        # --- 4. 维度二：幽灵诡计 (Phantom Trick) ---
        absorption_score = offensive_absorption_intent
        deceptive_narrative_score = normalized_mtf_scores['SCORE_BEHAVIOR_DECEPTION_INDEX_clip_abs']
        deceptive_action_score = (
            (absorption_score + 1e-9).pow(action_weights.get('absorption', 0.6)) *
            (deceptive_narrative_score + 1e-9).pow(action_weights.get('deception_positive', 0.4))
        ).pow(1/(action_weights.get('absorption', 0.6) + action_weights.get('deception_positive', 0.4))).fillna(0.0)
        # --- 5. 维度三：突袭品质 (Strike Quality) ---
        closing_strength_raw = signals_data['closing_strength_index_D']
        closing_strength_score = normalize_score(closing_strength_raw, df.index, 55, default_value=0.5, debug_info=False) # normalize_score uses window
        upward_purity_score = normalized_mtf_scores['upward_impulse_purity_D']
        counterattack_quality_score = (
            (closing_strength_score + 1e-9).pow(quality_weights.get('closing_strength', 0.6)) *
            (upward_purity_score + 1e-9).pow(quality_weights.get('upward_purity', 0.4))
        ).pow(1/(quality_weights.get('closing_strength', 0.6) + quality_weights.get('upward_purity', 0.4))).fillna(0.0)
        # --- 6. 最终合成：三维融合 ---
        ambush_counterattack_score = (
            (ambush_context_score + 1e-9).pow(fusion_weights.get('context', 0.3)) *
            (deceptive_action_score + 1e-9).pow(fusion_weights.get('action', 0.4)) *
            (counterattack_quality_score + 1e-9).pow(fusion_weights.get('quality', 0.3))
        ).pow(1/(fusion_weights.get('context', 0.3) + fusion_weights.get('action', 0.4) + fusion_weights.get('quality', 0.3))).fillna(0.0)
        final_score = ambush_counterattack_score.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - panic_selling_cascade_D: {signals_data['panic_selling_cascade_D'].loc[probe_ts]:.4f}")
            print(f"        - SLOPE_{prior_weakness_slope_window}_close_D: {signals_data[f'SLOPE_{prior_weakness_slope_window}_close_D'].loc[probe_ts]:.4f}")
            print(f"        - loser_pain_index_D: {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
            print(f"        - closing_strength_index_D: {signals_data['closing_strength_index_D'].loc[probe_ts]:.4f}")
            print(f"        - upward_impulse_purity_D: {signals_data['upward_impulse_purity_D'].loc[probe_ts]:.4f}")
            print(f"        - BBW_21_2.0_D: {signals_data['BBW_21_2.0_D'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_DECEPTION_INDEX: {signals_data['SCORE_BEHAVIOR_DECEPTION_INDEX'].loc[probe_ts]:.4f}")
            print(f"        - offensive_absorption_intent (来自外部): {offensive_absorption_intent.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 脆弱战场分数: {ambush_context_score.loc[probe_ts]:.4f}")
            print(f"        - 幽灵诡计分数: {deceptive_action_score.loc[probe_ts]:.4f}")
            print(f"        - 突袭品质分数: {counterattack_quality_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '伏击式反攻'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_breakout_failure_risk(self, df: pd.DataFrame, distribution_intent: pd.Series, overextension_score_series: pd.Series, deception_index_series: pd.Series, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V5.3 · 行为模式精微化版】诊断突破失败级联风险
        - 核心重构: 废弃了基于简单价格比较的“机械式突破谬误”模型。引入基于“诱多-伏击-情境”
                      诡道剧本的全新三维诊断模型，旨在精确识别高迷惑性的“牛市陷阱”。
        - 行为情报核心聚焦: 严格遵循“只分析行为类原始数据”的原则。本模块不再分析原始筹码/资金流信号，
                              也不依赖其他情报层的高阶融合信号。原有的“套牢盘痛苦度”和“主力背弃度”维度
                              因其本质依赖筹码/资金流数据，已从本方法中移除，以确保行为情报的纯粹性。
                              本信号现在专注于纯粹的市场行为模式。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_breakout_failure_risk"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '突破失败级联风险' @ {probe_ts.strftime('%Y-%m-%d')}")
        required_signals = [
            'breakout_quality_score_D', 'retail_fomo_premium_index_D', 'trend_vitality_index_D',
            'active_buying_support_D', 'upward_impulse_purity_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'RSI_13_D', 'SLOPE_5_RSI_13_D', 'BIAS_55_D', 'SLOPE_5_close_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        p_behavioral_div_conf = self.config_params
        p_mtf = get_param_value(p_behavioral_div_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        breakout_params = get_param_value(p_behavioral_div_conf.get('breakout_failure_risk_params'), {})
        core_risk_weights = get_param_value(breakout_params.get('core_risk_weights'), {"ambush": 1.0})
        context_amplifier_weights = get_param_value(breakout_params.get('context_amplifier_weights'), {"overextension": 0.3, "positive_deception": 0.2, "retail_fomo": 0.2, "behavioral_sentiment_extreme": 0.3})
        max_amplification_factor = get_param_value(breakout_params.get('max_amplification_factor'), 0.5)
        lure_weakness_multiplier = get_param_value(breakout_params.get('lure_weakness_multiplier'), 0.5)
        ambush_fusion_weights = get_param_value(breakout_params.get('ambush_fusion_weights'), {"distribution_intent": 0.7, "covert_ambush_intent": 0.2, "behavioral_momentum_divergence": 0.1})
        covert_ambush_intent_weights = get_param_value(breakout_params.get('covert_ambush_intent_weights'), {"weak_buying_support": 0.6, "declining_impulse_purity": 0.4})
        deceptive_calm_weights = get_param_value(breakout_params.get('deceptive_calm_weights'), {"overextension_inverse": 0.3, "positive_deception_inverse": 0.3, "retail_fomo_inverse": 0.4})
        deceptive_calm_multiplier = get_param_value(breakout_params.get('deceptive_calm_multiplier'), 0.2)
        deceptive_calm_threshold = get_param_value(breakout_params.get('deceptive_calm_threshold'), 0.5)
        base_dynamic_risk_weight_exponent = get_param_value(breakout_params.get('base_dynamic_risk_weight_exponent'), 1.5)
        volatility_exponent_multiplier = get_param_value(breakout_params.get('volatility_exponent_multiplier'), 0.5)
        trend_vitality_exponent_multiplier = get_param_value(breakout_params.get('trend_vitality_exponent_multiplier'), 0.5)
        core_risk_synergy_exponent = get_param_value(breakout_params.get('core_risk_synergy_exponent'), 2.0)
        core_risk_high_end_stretch_power = get_param_value(breakout_params.get('core_risk_high_end_stretch_power'), 2.0)
        risk_ema_span = get_param_value(breakout_params.get('risk_ema_span'), 5)
        risk_trend_slope_window = get_param_value(breakout_params.get('risk_trend_slope_window'), 5)
        risk_momentum_diff_window = get_param_value(breakout_params.get('risk_momentum_diff_window'), 1)
        risk_trend_mod_multiplier = get_param_value(breakout_params.get('risk_trend_mod_multiplier'), 0.2)
        risk_momentum_mod_multiplier = get_param_value(breakout_params.get('risk_momentum_mod_multiplier'), 0.1)
        behavioral_momentum_divergence_weights = get_param_value(breakout_params.get('behavioral_momentum_divergence_weights'), {"price_slope_weight": 0.5, "rsi_slope_weight": 0.5})
        behavioral_sentiment_extreme_weights = get_param_value(breakout_params.get('behavioral_sentiment_extreme_weights'), {"retail_fomo": 0.4, "rsi_extreme": 0.3, "bias_extreme": 0.3})
        # --- 1. 获取核心战术要素的原始数据 ---
        breakout_quality_raw = signals_data['breakout_quality_score_D']
        overextension_score = overextension_score_series
        deception_raw = deception_index_series
        retail_fomo_raw = signals_data['retail_fomo_premium_index_D']
        trend_vitality_raw = signals_data['trend_vitality_index_D']
        active_buying_raw = signals_data['active_buying_support_D']
        upward_purity_raw = signals_data['upward_impulse_purity_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        close_slope_raw = signals_data['SLOPE_5_close_D']
        rsi_raw = signals_data['RSI_13_D']
        rsi_slope_raw = signals_data['SLOPE_5_RSI_13_D']
        bias_raw = signals_data['BIAS_55_D']
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        behavioral_momentum_divergence_raw = pd.Series(0.0, index=df.index, dtype=np.float32)
        bullish_divergence_mask = (close_slope_raw > 0) & (rsi_slope_raw < 0)
        behavioral_momentum_divergence_raw.loc[bullish_divergence_mask] = \
            close_slope_raw.loc[bullish_divergence_mask].abs() * behavioral_momentum_divergence_weights.get('price_slope_weight', 0.5) + \
            rsi_slope_raw.loc[bullish_divergence_mask].abs() * behavioral_momentum_divergence_weights.get('rsi_slope_weight', 0.5)
        rsi_raw_clip_70_100 = rsi_raw.clip(70, 100)
        bias_raw_clip_0_1 = bias_raw.clip(0.1, 1.0)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'breakout_quality_score_D': (breakout_quality_raw, default_weights, True),
            'trend_vitality_index_D': (trend_vitality_raw, default_weights, True),
            'active_buying_support_D': (active_buying_raw, default_weights, True),
            'upward_impulse_purity_D': (upward_purity_raw, default_weights, True),
            'VOLATILITY_INSTABILITY_INDEX_21d_D': (volatility_instability_raw, default_weights, True),
            'retail_fomo_premium_index_D_clip': (retail_fomo_raw.clip(lower=0), default_weights, True),
            'RSI_13_D_clip_70_100': (rsi_raw_clip_70_100, default_weights, True), # rsi_extreme
            'BIAS_55_D_clip_0_1': (bias_raw_clip_0_1, default_weights, True), # bias_extreme
            'behavioral_momentum_divergence_raw': (behavioral_momentum_divergence_raw, default_weights, True) # For behavioral_momentum_divergence_score
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 2. 计算各要素得分 ---
        lure_score = normalized_mtf_scores['breakout_quality_score_D']
        market_weakness_score = (1 - normalized_mtf_scores['trend_vitality_index_D']).clip(0, 1)
        lure_score_modulated = (lure_score * (1 + market_weakness_score * lure_weakness_multiplier)).clip(0, 1)
        weak_buying_support_score = (1 - normalized_mtf_scores['active_buying_support_D']).clip(0,1)
        declining_impulse_purity_score = (1 - normalized_mtf_scores['upward_impulse_purity_D']).clip(0,1)
        covert_ambush_intent_score = (
            weak_buying_support_score * covert_ambush_intent_weights.get('weak_buying_support', 0.6) +
            declining_impulse_purity_score * covert_ambush_intent_weights.get('declining_impulse_purity', 0.4)
        ).clip(0,1)
        behavioral_momentum_divergence_score = normalized_mtf_scores['behavioral_momentum_divergence_raw']
        ambush_fusion_weights['behavioral_momentum_divergence'] = ambush_fusion_weights.get('behavioral_momentum_divergence', 0.1)
        ambush_score = (
            distribution_intent * ambush_fusion_weights.get('distribution_intent', 0.7) +
            covert_ambush_intent_score * ambush_fusion_weights.get('covert_ambush_intent', 0.2) +
            behavioral_momentum_divergence_score * ambush_fusion_weights.get('behavioral_momentum_divergence', 0.1)
        ).clip(0,1)
        positive_deception_score = deception_index_series.clip(lower=0) # deception_raw 是 Series，不需要再从 normalized_mtf_scores 获取
        retail_fomo_score = normalized_mtf_scores['retail_fomo_premium_index_D_clip']
        norm_retail_fomo_extreme = normalized_mtf_scores['retail_fomo_premium_index_D_clip']
        norm_rsi_extreme = normalized_mtf_scores['RSI_13_D_clip_70_100']
        norm_bias_extreme = normalized_mtf_scores['BIAS_55_D_clip_0_1']
        behavioral_sentiment_extreme_score = (
            norm_retail_fomo_extreme * behavioral_sentiment_extreme_weights.get('retail_fomo', 0.4) +
            norm_rsi_extreme * behavioral_sentiment_extreme_weights.get('rsi_extreme', 0.3) +
            norm_bias_extreme * behavioral_sentiment_extreme_weights.get('bias_extreme', 0.3)
        ).clip(0,1)
        context_amplifier_weights['behavioral_sentiment_extreme'] = context_amplifier_weights.get('behavioral_sentiment_extreme', 0.3)
        context_amplifier_factor = (
            overextension_score * context_amplifier_weights.get('overextension', 0.3) +
            positive_deception_score * context_amplifier_weights.get('positive_deception', 0.2) +
            retail_fomo_score * context_amplifier_weights.get('retail_fomo', 0.2) +
            behavioral_sentiment_extreme_score * context_amplifier_weights.get('behavioral_sentiment_extreme', 0.3)
        ).clip(0, 1)
        # --- 3. 核心风险基准分合成 ---
        normalized_volatility = normalized_mtf_scores['VOLATILITY_INSTABILITY_INDEX_21d_D']
        normalized_inverse_trend_vitality = (1 - normalized_mtf_scores['trend_vitality_index_D']).clip(0,1)
        adaptive_dynamic_risk_weight_exponent = (
            base_dynamic_risk_weight_exponent +
            normalized_volatility * volatility_exponent_multiplier +
            normalized_inverse_trend_vitality * trend_vitality_exponent_multiplier
        ).clip(1.0, 3.0)
        dynamic_ambush_contribution = (ambush_score.pow(adaptive_dynamic_risk_weight_exponent)) * core_risk_weights.get('ambush', 1.0)
        dynamic_ambush_weight = pd.Series(1.0, index=df.index, dtype=np.float32) # This seems unused, consider removing if not used later
        ambush_score_pow = (ambush_score + 1e-9).pow(core_risk_synergy_exponent)
        weighted_avg_risk = (dynamic_ambush_weight * ambush_score_pow).pow(1 / core_risk_synergy_exponent).fillna(0.0)
        stretched_weighted_avg_risk = (1 - (1 - weighted_avg_risk).pow(core_risk_high_end_stretch_power)).clip(0,1)
        core_risk_base_initial = (lure_score_modulated * stretched_weighted_avg_risk).clip(0,1).fillna(0.0)
        deceptive_calm_weights['overextension_inverse'] = deceptive_calm_weights.get('overextension_inverse', 0.3)
        deceptive_calm_weights['positive_deception_inverse'] = deceptive_calm_weights.get('positive_deception_inverse', 0.3)
        deceptive_calm_weights['retail_fomo_inverse'] = deceptive_calm_weights.get('retail_fomo_inverse', 0.4)
        deceptive_calm_score = (
            (1 - overextension_score) * deceptive_calm_weights.get('overextension_inverse', 0.3) +
            (1 - positive_deception_score) * deceptive_calm_weights.get('positive_deception_inverse', 0.3) +
            (1 - retail_fomo_score) * deceptive_calm_weights.get('retail_fomo_inverse', 0.4)
        ).clip(0,1)
        deceptive_calm_effect = deceptive_calm_score * deceptive_calm_multiplier * (core_risk_base_initial > deceptive_calm_threshold).astype(float)
        final_amplifier = 1 + (context_amplifier_factor * max_amplification_factor) + deceptive_calm_effect
        if len(core_risk_base_initial) < max(risk_ema_span, risk_trend_slope_window, risk_momentum_diff_window) + 1:
            risk_dynamic_modulator = pd.Series(1.0, index=df.index, dtype=np.float32)
        else:
            risk_ema = core_risk_base_initial.ewm(span=risk_ema_span, adjust=False).mean()
            risk_trend = risk_ema.diff(risk_trend_slope_window).fillna(0.0)
            risk_momentum = risk_trend.diff(risk_momentum_diff_window).fillna(0.0)
            # --- 收集所有需要进行双极归一化的 Series 的配置 ---
            risk_series_for_bipolar_norm_config = {
                'risk_trend': (risk_trend, default_weights, True),
                'risk_momentum': (risk_momentum, default_weights, True)
            }
            normalized_risk_bipolar_scores = {}
            for key, (series_obj, tf_w, asc) in risk_series_for_bipolar_norm_config.items():
                normalized_risk_bipolar_scores[key] = get_adaptive_mtf_normalized_bipolar_score(series_obj, df.index, tf_weights=tf_w, debug_info=False)
            risk_trend_score = normalized_risk_bipolar_scores['risk_trend']
            risk_momentum_score = normalized_risk_bipolar_scores['risk_momentum']
            risk_dynamic_modulator = (
                1 +
                (risk_trend_score * risk_trend_mod_multiplier) +
                (risk_momentum_score * risk_momentum_mod_multiplier)
            ).clip(0.5, 1.5)
        # --- 4. 最终风险合成 ---
        breakout_failure_risk = (core_risk_base_initial * final_amplifier * risk_dynamic_modulator).clip(0, 1)
        final_score = breakout_failure_risk.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - breakout_quality_score_D: {signals_data['breakout_quality_score_D'].loc[probe_ts]:.4f}")
            print(f"        - retail_fomo_premium_index_D: {signals_data['retail_fomo_premium_index_D'].loc[probe_ts]:.4f}")
            print(f"        - trend_vitality_index_D: {signals_data['trend_vitality_index_D'].loc[probe_ts]:.4f}")
            print(f"        - active_buying_support_D: {signals_data['active_buying_support_D'].loc[probe_ts]:.4f}")
            print(f"        - upward_impulse_purity_D: {signals_data['upward_impulse_purity_D'].loc[probe_ts]:.4f}")
            print(f"        - VOLATILITY_INSTABILITY_INDEX_21d_D: {signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'].loc[probe_ts]:.4f}")
            print(f"        - RSI_13_D: {signals_data['RSI_13_D'].loc[probe_ts]:.4f}")
            print(f"        - SLOPE_5_RSI_13_D: {signals_data['SLOPE_5_RSI_13_D'].loc[probe_ts]:.4f}")
            print(f"        - BIAS_55_D: {signals_data['BIAS_55_D'].loc[probe_ts]:.4f}")
            print(f"        - SLOPE_5_close_D: {signals_data['SLOPE_5_close_D'].loc[probe_ts]:.4f}")
            print(f"        - distribution_intent (来自外部): {distribution_intent.loc[probe_ts]:.4f}")
            print(f"        - overextension_score_series (来自外部): {overextension_score_series.loc[probe_ts]:.4f}")
            print(f"        - deception_index_series (来自外部): {deception_index_series.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 诱饵分数 (Lure Score): {lure_score.loc[probe_ts]:.4f}")
            print(f"        - 市场弱势分数 (Market Weakness Score): {market_weakness_score.loc[probe_ts]:.4f}")
            print(f"        - 调制后的诱饵分数: {lure_score_modulated.loc[probe_ts]:.4f}")
            print(f"        - 隐蔽伏击意图分数: {covert_ambush_intent_score.loc[probe_ts]:.4f}")
            print(f"        - 行为动量背离分数: {behavioral_momentum_divergence_score.loc[probe_ts]:.4f}")
            print(f"        - 伏击分数 (Ambush Score): {ambush_score.loc[probe_ts]:.4f}")
            print(f"        - 积极欺骗分数: {positive_deception_score.loc[probe_ts]:.4f}")
            print(f"        - 散户FOMO分数: {retail_fomo_score.loc[probe_ts]:.4f}")
            print(f"        - 行为情绪极端分数: {behavioral_sentiment_extreme_score.loc[probe_ts]:.4f}")
            print(f"        - 情境放大因子: {context_amplifier_factor.loc[probe_ts]:.4f}")
            print(f"        - 核心风险基础初始分数: {core_risk_base_initial.loc[probe_ts]:.4f}")
            print(f"        - 欺骗性平静效应: {deceptive_calm_effect.loc[probe_ts]:.4f}")
            print(f"        - 最终放大器: {final_amplifier.loc[probe_ts]:.4f}")
            print(f"        - 风险动态调制器: {risk_dynamic_modulator.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '突破失败级联风险'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_volume_burst_quality(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】计算高品质看涨量能爆发信号。
        - 核心重构: 废弃V2.1“战术近视眼”模型，引入“战术品质 × 战略环境”的全新双维诊断框架。
        - 诊断双维度:
          1. 战术强攻品质 (Tactical Assault Quality): 保留V2.1四维模型(幅度、信念、效率、战果)，评估登陆部队战斗力。
          2. 战略环境评估 (Strategic Environment Assessment): 新增“滩头阵地阻力指数”，评估登陆点上方的套牢盘压力。
        - 数学模型: 品质分 = 战术品质分 * (1 - 战略阻力分)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_volume_burst_quality"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质量能爆发' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('beachhead_protocol_params'), {})
        strategic_weights = get_param_value(params.get('strategic_weights'), {'chip_fatigue': 0.6, 'loser_pain': 0.4})
        required_signals = [
            'volume_ratio_D', 'main_force_conviction_index_D',
            'impulse_quality_ratio_D', 'closing_strength_index_D',
            'chip_fatigue_index_D', 'loser_pain_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'volume_ratio_D': (signals_data['volume_ratio_D'], tf_weights, True, False),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), tf_weights, True, False),
            'impulse_quality_ratio_D': (signals_data['impulse_quality_ratio_D'], tf_weights, True, False),
            'closing_strength_index_D': (signals_data['closing_strength_index_D'], 55, True, False), # normalize_score uses window
            'chip_fatigue_index_D': (signals_data['chip_fatigue_index_D'], tf_weights, True, False),
            'loser_pain_index_D': (signals_data['loser_pain_index_D'], tf_weights, True, False)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w_or_window, asc, is_bipolar_flag) in series_for_mtf_norm_config.items():
            if isinstance(tf_w_or_window, dict): # get_adaptive_mtf_normalized_score
                normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w_or_window, ascending=asc, debug_info=False)
            else: # normalize_score
                normalized_mtf_scores[key] = normalize_score(series_obj, df.index, windows=tf_w_or_window, ascending=asc, debug_info=False)
        # --- 2. 维度一：战术强攻品质评估 (沿用V2.1逻辑) ---
        magnitude_score = normalized_mtf_scores['volume_ratio_D']
        conviction_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        efficiency_score = normalized_mtf_scores['impulse_quality_ratio_D']
        result_score = normalized_mtf_scores['closing_strength_index_D']
        tactical_assault_quality_score = (
            (magnitude_score + 1e-9) * (conviction_score + 1e-9) *
            (efficiency_score + 1e-9) * (result_score + 1e-9)
        ).pow(1/4).fillna(0.0)
        # --- 3. 维度二：战略环境评估 ---
        chip_fatigue_score = normalized_mtf_scores['chip_fatigue_index_D']
        loser_pain_score = normalized_mtf_scores['loser_pain_index_D']
        beachhead_resistance_score = (
            chip_fatigue_score * strategic_weights.get('chip_fatigue', 0.6) +
            loser_pain_score * strategic_weights.get('loser_pain', 0.4)
        ).clip(0, 1)
        strategic_environment_score = (1 - beachhead_resistance_score)
        # --- 4. 最终合成：战术品质 × 战略环境 ---
        final_burst_quality = (tactical_assault_quality_score * strategic_environment_score).clip(0, 1)
        final_score = final_burst_quality.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - volume_ratio_D: {signals_data['volume_ratio_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"        - impulse_quality_ratio_D: {signals_data['impulse_quality_ratio_D'].loc[probe_ts]:.4f}")
            print(f"        - closing_strength_index_D: {signals_data['closing_strength_index_D'].loc[probe_ts]:.4f}")
            print(f"        - chip_fatigue_index_D: {signals_data['chip_fatigue_index_D'].loc[probe_ts]:.4f}")
            print(f"        - loser_pain_index_D: {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战术强攻品质分数: {tactical_assault_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 滩头阵地阻力分数: {beachhead_resistance_score.loc[probe_ts]:.4f}")
            print(f"        - 战略环境分数: {strategic_environment_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '高品质量能爆发'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_volume_atrophy(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】计算高品质成交量萎缩信号。
        - 核心重构: 废弃V2.1“静态快照谬误”模型，引入“战略环境×静态筹码×动态过程”的全新三维诊断框架。
        - 诊断三维度:
          1. 战略环境门控 (The Furnace Check): 审判多头是否掌控日内主导权，作为点火前提。
          2. 筹码纯度诊断 (The Purity Test): 沿用V2.1逻辑，评估“炉料”品质（获利盘、套牢盘、浮筹）。
          3. 过程稳定性封印 (The Stability Seal): 新增动态诊断，审判“淬炼”过程是否平稳（低波动率）。
        - 数学模型: 品质分 = 战略门控 * 基础萎缩分 * (纯度分 * 稳定分) ^ 0.5
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_volume_atrophy"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质成交量萎缩' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('crucible_protocol_params'), {})
        stability_window = get_param_value(params.get('stability_window'), 5)
        quality_weights = get_param_value(params.get('quality_weights'), {'purity_score': 0.6, 'stability_score': 0.4})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'volume_ratio_D', 'winner_stability_index_D',
            'loser_pain_index_D', 'floating_chip_cleansing_efficiency_D',
            'vwap_control_strength_D', 'close_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        close_pct_change_rolling_std = signals_data['close_D'].pct_change().rolling(window=stability_window).std().fillna(0)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'volume_ratio_D': (signals_data['volume_ratio_D'], tf_weights, True, False),
            'winner_stability_index_D': (signals_data['winner_stability_index_D'], tf_weights, True, False),
            'loser_pain_index_D': (signals_data['loser_pain_index_D'], tf_weights, True, False),
            'floating_chip_cleansing_efficiency_D': (signals_data['floating_chip_cleansing_efficiency_D'], tf_weights, True, False),
            'vwap_control_strength_D': (signals_data['vwap_control_strength_D'], 55, True, False), # normalize_score uses window
            'close_pct_change_rolling_std': (close_pct_change_rolling_std, tf_weights, True, False) # normalized_volatility
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w_or_window, asc, is_bipolar_flag) in series_for_mtf_norm_config.items():
            if isinstance(tf_w_or_window, dict): # get_adaptive_mtf_normalized_score
                normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w_or_window, ascending=asc, debug_info=False)
            else: # normalize_score
                normalized_mtf_scores[key] = normalize_score(series_obj, df.index, windows=tf_w_or_window, ascending=asc, debug_info=False)
        # --- 3. 计算核心组件 ---
        strategic_context_gate = normalized_mtf_scores['vwap_control_strength_D']
        base_atrophy_score = 1 - normalized_mtf_scores['volume_ratio_D']
        lockup_score = normalized_mtf_scores['winner_stability_index_D']
        exhaustion_score = normalized_mtf_scores['loser_pain_index_D']
        cleansing_score = normalized_mtf_scores['floating_chip_cleansing_efficiency_D']
        purity_score = ((lockup_score + 1e-9) * (exhaustion_score + 1e-9) * (cleansing_score + 1e-9)).pow(1/3).fillna(0.0)
        normalized_volatility = normalized_mtf_scores['close_pct_change_rolling_std']
        stability_score = (1 - normalized_volatility).clip(0, 1)
        # --- 4. 最终品质合成 ---
        quality_modulator = (
            (purity_score).pow(quality_weights.get('purity_score', 0.6)) *
            (stability_score).pow(quality_weights.get('stability_score', 0.4))
        ).fillna(0.0)
        final_atrophy_quality = (strategic_context_gate * base_atrophy_score * quality_modulator).clip(0, 1)
        final_score = final_atrophy_quality.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - volume_ratio_D: {signals_data['volume_ratio_D'].loc[probe_ts]:.4f}")
            print(f"        - winner_stability_index_D: {signals_data['winner_stability_index_D'].loc[probe_ts]:.4f}")
            print(f"        - loser_pain_index_D: {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
            print(f"        - floating_chip_cleansing_efficiency_D: {signals_data['floating_chip_cleansing_efficiency_D'].loc[probe_ts]:.4f}")
            print(f"        - vwap_control_strength_D: {signals_data['vwap_control_strength_D'].loc[probe_ts]:.4f}")
            print(f"        - close_D: {signals_data['close_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战略环境门控分数: {strategic_context_gate.loc[probe_ts]:.4f}")
            print(f"        - 基础萎缩分数: {base_atrophy_score.loc[probe_ts]:.4f}")
            print(f"        - 筹码纯度分数: {purity_score.loc[probe_ts]:.4f}")
            print(f"        - 过程稳定性分数: {stability_score.loc[probe_ts]:.4f}")
            print(f"        - 品质调制器: {quality_modulator.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '高品质成交量萎缩'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_absorption_strength(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】计算高品质承接强度信号。
        - 核心重构: 废弃V2.1“孤城谬误”模型，引入“地基×行动×意图”的全新三维诊断框架。
        - 诊断三维度:
          1. 地基勘探 (The Foundation Survey): 审判承接是否发生在经过验证的坚固支撑位上。
          2. 构筑行动 (The Construction Action): 审判承接过程本身的强度与主动性。
          3. 总督意志 (The Governor's Will): 审判承接行为背后的主力真实信念。
        - 数学模型: 强度分 = (地基品质分 * 构筑行动分 * 总督意志分) ^ (1/3)
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_absorption_strength"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质承接强度' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('citadel_protocol_params'), {})
        action_weights = get_param_value(params.get('action_weights'), {'dip_absorption': 0.6, 'active_buying': 0.4})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'support_validation_strength_D', 'dip_absorption_power_D',
            'active_buying_support_D', 'main_force_conviction_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'support_validation_strength_D': (signals_data['support_validation_strength_D'], default_weights, True),
            'dip_absorption_power_D': (signals_data['dip_absorption_power_D'], default_weights, True),
            'active_buying_support_D': (signals_data['active_buying_support_D'], default_weights, True),
            'main_force_conviction_index_D_clip': (signals_data['main_force_conviction_index_D'].clip(lower=0), default_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        foundation_score = normalized_mtf_scores['support_validation_strength_D']
        action_dip_score = normalized_mtf_scores['dip_absorption_power_D']
        action_active_score = normalized_mtf_scores['active_buying_support_D']
        construction_action_score = (
            action_dip_score * action_weights.get('dip_absorption', 0.6) +
            action_active_score * action_weights.get('active_buying', 0.4)
        )
        governors_will_score = normalized_mtf_scores['main_force_conviction_index_D_clip']
        # --- 4. “堡垒协议”三维合成 ---
        absorption_strength = (
            (foundation_score + 1e-9) *
            (construction_action_score + 1e-9) *
            (governors_will_score + 1e-9)
        ).pow(1/3).fillna(0.0)
        final_score = absorption_strength.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - support_validation_strength_D: {signals_data['support_validation_strength_D'].loc[probe_ts]:.4f}")
            print(f"        - dip_absorption_power_D: {signals_data['dip_absorption_power_D'].loc[probe_ts]:.4f}")
            print(f"        - active_buying_support_D: {signals_data['active_buying_support_D'].loc[probe_ts]:.4f}")
            print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 地基品质分数: {foundation_score.loc[probe_ts]:.4f}")
            print(f"        - 构筑行动分数: {construction_action_score.loc[probe_ts]:.4f}")
            print(f"        - 总督意志分数: {governors_will_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '高品质承接强度'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_behavioral_price_overextension(self, df: pd.DataFrame, tf_weights: Dict, long_term_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V4.4 · 生产就绪版 - 行为过热深度感知与鲁棒融合】计算纯粹基于行为类原始数据的价格超买亢奋原始分。
        - 核心升级: 引入“行为惯性”、“散户狂热”和“动态阈值”概念，使亢奋诊断更具情境感知和前瞻性。
        - 优化点:
          1. 行为惯性: 考虑价格和成交量斜率的加速度，作为动量过热的补充证据。
          2. 动态阈值: 根据市场波动率（ATR）动态调整BIAS和BBP的超买阈值。
          3. 成交量极端细化: 区分放量滞涨和放量加速上涨，避免误判。
          4. 引入散户狂热：将 `retail_fomo_premium_index_D` 纳入过热评估。
          5. **融合函数优化: 从加权几何平均改为加权算术平均，避免零值问题，并调整权重。**
          6. **引入亢奋加速度因子：直接将价格和动量指标的加速度作为亢奋的独立证据。**
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_behavioral_price_overextension"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '高品质价格超买亢奋' @ {probe_ts.strftime('%Y-%m-%d')}")
        overextension_params = get_param_value(self.config_params.get('price_overextension_params'), {
            "enabled": True, "rsi_overbought_threshold": 70, "bias_overbought_threshold": 0.05,
            "bbp_overbought_threshold": 0.95, "volume_climax_multiplier": 1.8,
            "upward_efficiency_decay_penalty": 0.1, "intraday_control_decay_penalty": 0.1,
            "dynamic_bias_bbp_atr_multiplier": 0.005,
            "momentum_accel_bonus": 0.1,
            "fomo_weight": 0.2, # 新增散户狂热权重
            "price_speed_weight": 0.15, # 价格速度权重
            "price_accel_weight": 0.15, # 价格加速度权重
            "fusion_weights": { # 新增融合权重
                "price_deviation": 0.3,
                "momentum_overheat": 0.25,
                "volume_extremity": 0.15,
                "intraday_extremity": 0.3
            }
        })
        if not overextension_params.get('enabled', False):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 信号未启用，返回0分。")
            return pd.Series(0.0, index=df.index)
        fusion_weights = overextension_params.get('fusion_weights', {})
        required_signals = [
            'close_D', 'RSI_13_D', 'MACDh_13_34_8_D', 'volume_D', 'VOL_MA_21_D',
            'BIAS_5_D', 'BBP_21_2.0_D', 'ATR_14_D',
            'ACCEL_5_close_D', 'ACCEL_5_RSI_13_D',
            'ACCEL_5_MACDh_13_34_8_D', 'ACCEL_5_volume_D',
            'robust_pct_change_slope', 'robust_volume_slope',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'retail_fomo_premium_index_D', # 新增散户狂热指标
            'pct_change_D' # 用于计算价格速度
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回0分。")
            return pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 1. 价格偏离度 (Price Deviation) ---
        rsi_val = signals_data['RSI_13_D']
        bias_val = signals_data['BIAS_5_D']
        bbp_val = signals_data['BBP_21_2.0_D']
        atr_val = signals_data['ATR_14_D']
        rsi_overbought_threshold = overextension_params.get('rsi_overbought_threshold', 70)
        bias_overbought_threshold = overextension_params.get('bias_overbought_threshold', 0.05)
        bbp_overbought_threshold = overextension_params.get('bbp_overbought_threshold', 0.95)
        dynamic_bias_bbp_atr_multiplier = overextension_params.get('dynamic_bias_bbp_atr_multiplier', 0.005)
        # 动态调整BIAS和BBP阈值
        dynamic_bias_threshold = bias_overbought_threshold + atr_val * dynamic_bias_bbp_atr_multiplier
        dynamic_bbp_threshold = bbp_overbought_threshold + atr_val * dynamic_bias_bbp_atr_multiplier * 5 # BBP通常在0-1之间，乘数可调整
        norm_rsi_overbought = (rsi_val - rsi_overbought_threshold).clip(lower=0) / (100 - rsi_overbought_threshold)
        norm_rsi_overbought = norm_rsi_overbought.fillna(0).clip(0, 1)
        bias_denominator = (bias_val.max() - dynamic_bias_threshold)
        norm_bias_overbought = (bias_val - dynamic_bias_threshold).clip(lower=0) / bias_denominator.where(bias_denominator > 1e-9, 1e-9)
        norm_bias_overbought = norm_bias_overbought.fillna(0).clip(0, 1)
        bbp_denominator = (1 - dynamic_bbp_threshold)
        norm_bbp_overbought = (bbp_val - dynamic_bbp_threshold).clip(lower=0) / bbp_denominator.where(bbp_denominator > 1e-9, 1e-9)
        norm_bbp_overbought = norm_bbp_overbought.fillna(0).clip(0, 1)
        # 价格偏离度分数
        price_deviation_score = (norm_rsi_overbought + norm_bias_overbought + norm_bbp_overbought) / 3
        # --- 2. 动量过热 (Momentum Overheating) ---
        pct_change_val = signals_data['pct_change_D']
        accel_rsi = signals_data['ACCEL_5_RSI_13_D']
        accel_macd = signals_data['ACCEL_5_MACDh_13_34_8_D']
        robust_pct_change_slope = signals_data['robust_pct_change_slope']
        robust_volume_slope = signals_data['robust_volume_slope']
        accel_close = signals_data['ACCEL_5_close_D']
        accel_volume = signals_data['ACCEL_5_volume_D']
        momentum_accel_factor = pd.Series(0.0, index=df.index, dtype=np.float32)
        momentum_accel_factor = momentum_accel_factor.mask((accel_rsi > 0) & (accel_macd > 0), overextension_params.get('momentum_accel_bonus', 0.1))
        momentum_accel_factor = momentum_accel_factor.mask(((accel_rsi > 0) | (accel_macd > 0)) & (momentum_accel_factor == 0), overextension_params.get('momentum_accel_bonus', 0.1) / 2)
        behavioral_inertia_bonus = pd.Series(0.0, index=df.index, dtype=np.float32)
        is_price_accelerating = (robust_pct_change_slope > 0) & (accel_close > 0)
        is_volume_accelerating = (robust_volume_slope > 0) & (accel_volume > 0)
        behavioral_inertia_bonus = behavioral_inertia_bonus.mask(is_price_accelerating & is_volume_accelerating, overextension_params.get('momentum_accel_bonus', 0.1))
        behavioral_inertia_bonus = behavioral_inertia_bonus.mask((is_price_accelerating | is_volume_accelerating) & (behavioral_inertia_bonus == 0), overextension_params.get('momentum_accel_bonus', 0.1) / 2)
        # 价格速度和加速度作为过热信号
        price_speed_score = normalize_score(pct_change_val.clip(lower=0), df.index, windows=5, ascending=True, debug_info=False)
        price_accel_score = normalize_score(accel_close.clip(lower=0), df.index, windows=5, ascending=True, debug_info=False)
        momentum_overheat_score = (
            price_speed_score * overextension_params.get('price_speed_weight', 0.15) +
            price_accel_score * overextension_params.get('price_accel_weight', 0.15) +
            momentum_accel_factor * 0.35 + # 动量指标加速度
            behavioral_inertia_bonus * 0.35 # 行为惯性
        ).clip(0, 1)
        # --- 3. 成交量极端 (Volume Extremity) ---
        close_price = signals_data['close_D']
        current_volume = signals_data['volume_D']
        volume_avg = signals_data['VOL_MA_21_D']
        volume_climax_multiplier = overextension_params.get('volume_climax_multiplier', 1.8)
        is_price_rising_for_volume = (close_price > close_price.shift(1))
        is_volume_climax = (current_volume > volume_avg * volume_climax_multiplier) & is_price_rising_for_volume
        volume_extremity_score = is_volume_climax.astype(float) * (current_volume / volume_avg).clip(1, 2)
        # --- 4. 日内行为极端 (Intraday Behavioral Extremity) ---
        upward_efficiency = signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY']
        intraday_bull_control = signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL']
        retail_fomo_premium = signals_data['retail_fomo_premium_index_D'] # 新增散户狂热
        upward_efficiency_decay_penalty = overextension_params.get('upward_efficiency_decay_penalty', 0.1)
        intraday_control_decay_penalty = overextension_params.get('intraday_control_decay_penalty', 0.1)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'upward_efficiency': (upward_efficiency, tf_weights, True), # For norm_upward_efficiency_decay
            'intraday_bull_control': (intraday_bull_control, tf_weights, True), # For norm_intraday_control_decay
            'retail_fomo_premium_index_D': (retail_fomo_premium, tf_weights, True) # For retail_fomo_score
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores_for_decay = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores_for_decay[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        norm_upward_efficiency_decay = (1 - normalized_mtf_scores_for_decay['upward_efficiency']).clip(0, 1) * upward_efficiency_decay_penalty
        norm_intraday_control_decay = (1 - normalized_mtf_scores_for_decay['intraday_bull_control']).clip(0, 1) * intraday_control_decay_penalty
        retail_fomo_score = normalized_mtf_scores_for_decay['retail_fomo_premium_index_D'] # 归一化后的散户狂热
        intraday_extremity_score = (
            norm_upward_efficiency_decay * 0.3 +
            norm_intraday_control_decay * 0.3 +
            retail_fomo_score * overextension_params.get('fomo_weight', 0.4) # 散户狂热权重
        ).clip(0, 1)
        # --- 5. 最终融合 (加权算术平均) ---
        overextension_score = (
            price_deviation_score * fusion_weights.get("price_deviation", 0.3) +
            momentum_overheat_score * fusion_weights.get("momentum_overheat", 0.25) +
            volume_extremity_score * fusion_weights.get("volume_extremity", 0.15) +
            intraday_extremity_score * fusion_weights.get("intraday_extremity", 0.3)
        ) / sum(fusion_weights.values()) # 归一化权重
        final_score = overextension_score.clip(0, 1).astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - RSI_13_D: {signals_data['RSI_13_D'].loc[probe_ts]:.4f}")
            print(f"        - BIAS_5_D: {signals_data['BIAS_5_D'].loc[probe_ts]:.4f}")
            print(f"        - BBP_21_2.0_D: {signals_data['BBP_21_2.0_D'].loc[probe_ts]:.4f}")
            print(f"        - ATR_14_D: {signals_data['ATR_14_D'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_RSI_13_D: {signals_data['ACCEL_5_RSI_13_D'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_MACDh_13_34_8_D: {signals_data['ACCEL_5_MACDh_13_34_8_D'].loc[probe_ts]:.4f}")
            print(f"        - robust_pct_change_slope: {signals_data['robust_pct_change_slope'].loc[probe_ts]:.4f}")
            print(f"        - robust_volume_slope: {signals_data['robust_volume_slope'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_close_D: {signals_data['ACCEL_5_close_D'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_volume_D: {signals_data['ACCEL_5_volume_D'].loc[probe_ts]:.4f}")
            print(f"        - volume_D: {signals_data['volume_D'].loc[probe_ts]:.4f}")
            print(f"        - VOL_MA_21_D: {signals_data['VOL_MA_21_D'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_UPWARD_EFFICIENCY: {signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL: {signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'].loc[probe_ts]:.4f}")
            print(f"        - retail_fomo_premium_index_D: {signals_data['retail_fomo_premium_index_D'].loc[probe_ts]:.4f}")
            print(f"        - pct_change_D: {signals_data['pct_change_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 动态BIAS阈值: {dynamic_bias_threshold.loc[probe_ts]:.4f}")
            print(f"        - 动态BBP阈值: {dynamic_bbp_threshold.loc[probe_ts]:.4f}")
            print(f"        - 归一化RSI超买: {norm_rsi_overbought.loc[probe_ts]:.4f}")
            print(f"        - 归一化BIAS超买: {norm_bias_overbought.loc[probe_ts]:.4f}")
            print(f"        - 归一化BBP超买: {norm_bbp_overbought.loc[probe_ts]:.4f}")
            print(f"        - 价格偏离度分数: {price_deviation_score.loc[probe_ts]:.4f}")
            print(f"        - 价格速度分数: {price_speed_score.loc[probe_ts]:.4f}")
            print(f"        - 价格加速度分数: {price_accel_score.loc[probe_ts]:.4f}")
            print(f"        - 动量过热分数: {momentum_overheat_score.loc[probe_ts]:.4f}")
            print(f"        - 成交量极端分数: {volume_extremity_score.loc[probe_ts]:.4f}")
            print(f"        - 散户狂热分数: {retail_fomo_score.loc[probe_ts]:.4f}")
            print(f"        - 日内行为极端分数: {intraday_extremity_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '价格超买亢奋'分数 @ 2025-12-10: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_behavioral_stagnation_evidence(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V4.0 · 行为纯化版】计算纯粹基于行为类原始数据的滞涨证据原始分。
        - 核心升级: 引入“行为惯性”和“动态阈值”概念，使滞涨诊断更具情境感知和前瞻性。
        - 优化点:
          1. 行为惯性: 考虑价格和成交量斜率的减速或负向加速度，作为滞涨的补充证据。
          2. 动态阈值: 根据市场波动率（ATR）动态调整K线形态（长上影线、小实体）的阈值。
          3. 动量背离细化: 价格上涨但动量指标下降，且下降速度加快，则滞涨证据更强。
          4. 成交量异常细化: 区分放量滞涨和缩量上涨，后者在某些情境下也可能是滞涨证据。
          5. 融合函数优化: 调整融合权重，并引入一个“滞涨加速度”因子。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_behavioral_stagnation_evidence"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '纯粹基于行为类原始数据的滞涨证据原始分' @ {probe_ts.strftime('%Y-%m-%d')}")
        stagnation_params = get_param_value(self.config_params.get('stagnation_evidence_params'), {
            "enabled": True, "upper_shadow_ratio_threshold": 0.4, "body_ratio_threshold": 0.3,
            "volume_stagnation_multiplier": 1.2, "momentum_divergence_penalty": 0.15,
            "upward_efficiency_decay_bonus": 0.1, "intraday_control_decay_bonus": 0.1,
            "dynamic_kline_atr_multiplier": 0.005,
            "momentum_deceleration_bonus": 0.1,
            "volume_drying_up_multiplier": 0.8
        })
        if not stagnation_params.get('enabled', False):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 信号未启用，返回0分。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        required_signals = [
            'close_D', 'open_D', 'high_D', 'low_D', 'volume_D', 'VOL_MA_21_D', 'ATR_14_D',
            'robust_close_slope', 'robust_RSI_13_slope', 'robust_MACDh_13_34_8_slope', 'robust_volume_slope', # 修正此处
            'ACCEL_5_close_D', 'ACCEL_5_RSI_13_D', 'ACCEL_5_MACDh_13_34_8_D', 'ACCEL_5_volume_D',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'pct_change_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回0分。")
            return pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- K线形态分析 ---
        open_price = signals_data['open_D']
        high_price = signals_data['high_D']
        low_price = signals_data['low_D']
        close_price = signals_data['close_D']
        current_volume = signals_data['volume_D']
        volume_avg = signals_data['VOL_MA_21_D']
        pct_change_val = signals_data['pct_change_D']
        atr_val = signals_data['ATR_14_D']
        robust_close_slope = signals_data['robust_close_slope']
        robust_rsi_slope = signals_data['robust_RSI_13_slope']
        robust_macd_slope = signals_data['robust_MACDh_13_34_8_slope'] # 修正此处
        accel_rsi = signals_data['ACCEL_5_RSI_13_D']
        accel_macd = signals_data['ACCEL_5_MACDh_13_34_8_D']
        upward_efficiency = signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY']
        intraday_bull_control = signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL']
        total_range = high_price - low_price
        total_range_safe = total_range.replace(0, 1e-9)
        body_range = (close_price - open_price).abs()
        upper_shadow = high_price - np.maximum(open_price, close_price)
        lower_shadow = np.minimum(open_price, close_price) - low_price
        upper_shadow_ratio = (upper_shadow / total_range_safe).clip(0, 1)
        body_ratio = (body_range / total_range_safe).clip(0, 1)
        upper_shadow_ratio_threshold = stagnation_params.get('upper_shadow_ratio_threshold', 0.4)
        body_ratio_threshold = stagnation_params.get('body_ratio_threshold', 0.3)
        volume_stagnation_multiplier = stagnation_params.get('volume_stagnation_multiplier', 1.2)
        momentum_divergence_penalty = stagnation_params.get('momentum_divergence_penalty', 0.15)
        dynamic_kline_atr_multiplier = stagnation_params.get('dynamic_kline_atr_multiplier', 0.005)
        momentum_deceleration_bonus = stagnation_params.get('momentum_deceleration_bonus', 0.1)
        volume_drying_up_multiplier = stagnation_params.get('volume_drying_up_multiplier', 0.8)
        dynamic_upper_shadow_threshold = upper_shadow_ratio_threshold + atr_val * dynamic_kline_atr_multiplier
        dynamic_body_ratio_threshold = body_ratio_threshold - atr_val * dynamic_kline_atr_multiplier
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        pct_change_abs_raw = signals_data['pct_change_D'].abs()
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'pct_change_abs_raw': (pct_change_abs_raw, tf_weights, True), # For volume_anomaly_score
            'upward_efficiency': (upward_efficiency, tf_weights, True), # For norm_upward_efficiency_decay
            'intraday_bull_control': (intraday_bull_control, tf_weights, True) # For norm_intraday_control_decay
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # 1. 价格行为疲软 (Price Action Weakness)
        is_long_upper_shadow = (upper_shadow_ratio > dynamic_upper_shadow_threshold)
        is_small_body = (body_ratio < dynamic_body_ratio_threshold)
        is_high_open_low_close = (open_price > close_price) & (pct_change_val < 0)
        price_weakness_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        price_weakness_score = price_weakness_score.mask(is_long_upper_shadow & is_small_body, 0.3)
        price_weakness_score = price_weakness_score.mask(is_high_open_low_close, price_weakness_score + 0.4)
        # 2. 动量背离 (Momentum Divergence)
        is_price_rising = (robust_close_slope > 0)
        is_rsi_momentum_decay = (robust_rsi_slope < 0)
        is_macd_momentum_decay = (robust_macd_slope < 0)
        momentum_divergence_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        rsi_deceleration_bonus = (is_rsi_momentum_decay & (accel_rsi < 0)).astype(int) * momentum_deceleration_bonus
        macd_deceleration_bonus = (is_macd_momentum_decay & (accel_macd < 0)).astype(int) * momentum_deceleration_bonus
        momentum_divergence_score = momentum_divergence_score.mask(
            is_price_rising & (is_rsi_momentum_decay | is_macd_momentum_decay),
            momentum_divergence_penalty * (is_rsi_momentum_decay.astype(int) + is_macd_momentum_decay.astype(int)) + \
            rsi_deceleration_bonus + macd_deceleration_bonus
        )
        momentum_divergence_score = momentum_divergence_score.clip(0, 0.3)
        # 3. 成交量异常 (Volume Anomaly)
        norm_pct_change_abs = normalized_mtf_scores['pct_change_abs_raw']
        is_volume_stagnation = (norm_pct_change_abs < 0.01) & (current_volume > volume_avg * volume_stagnation_multiplier) & (robust_close_slope > 0)
        volume_extremity_score = is_volume_stagnation.astype(float) * (current_volume / volume_avg).clip(1, 2)
        is_volume_drying_up = (pct_change_val > 0) & (current_volume < volume_avg * volume_drying_up_multiplier) & (robust_close_slope > 0)
        volume_drying_up_score = is_volume_drying_up.astype(float) * (1 - (current_volume / volume_avg)).clip(0, 1)
        volume_anomaly_score = (volume_extremity_score + volume_drying_up_score).clip(0, 1)
        # 4. 日内控制力减弱 (Intraday Control Weakness)
        norm_upward_efficiency_decay = (1 - normalized_mtf_scores['upward_efficiency']).clip(0, 1) * stagnation_params.get('upward_efficiency_decay_bonus', 0.1)
        norm_intraday_control_decay = (1 - normalized_mtf_scores['intraday_bull_control']).clip(0, 1) * stagnation_params.get('intraday_control_decay_bonus', 0.1)
        intraday_control_weakness_score = (norm_upward_efficiency_decay + norm_intraday_control_decay).clip(0, 1)
        stagnation_score = (
            (price_weakness_score + 1e-9).pow(0.3) *
            (momentum_divergence_score + 1e-9).pow(0.3) *
            (volume_anomaly_score + 1e-9).pow(0.2) *
            (intraday_control_weakness_score + 1e-9).pow(0.2)
        ).pow(1/1.0).fillna(0.0).clip(0, 1)
        final_score = stagnation_score.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - close_D: {signals_data['close_D'].loc[probe_ts]:.4f}")
            print(f"        - open_D: {signals_data['open_D'].loc[probe_ts]:.4f}")
            print(f"        - high_D: {signals_data['high_D'].loc[probe_ts]:.4f}")
            print(f"        - low_D: {signals_data['low_D'].loc[probe_ts]:.4f}")
            print(f"        - volume_D: {signals_data['volume_D'].loc[probe_ts]:.4f}")
            print(f"        - VOL_MA_21_D: {signals_data['VOL_MA_21_D'].loc[probe_ts]:.4f}")
            print(f"        - ATR_14_D: {signals_data['ATR_14_D'].loc[probe_ts]:.4f}")
            print(f"        - robust_close_slope: {signals_data['robust_close_slope'].loc[probe_ts]:.4f}")
            print(f"        - robust_RSI_13_slope: {signals_data['robust_RSI_13_slope'].loc[probe_ts]:.4f}")
            print(f"        - robust_MACDh_13_34_8_slope: {signals_data['robust_MACDh_13_34_8_slope'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_RSI_13_D: {signals_data['ACCEL_5_RSI_13_D'].loc[probe_ts]:.4f}")
            print(f"        - ACCEL_5_MACDh_13_34_8_D: {signals_data['ACCEL_5_MACDh_13_34_8_D'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_UPWARD_EFFICIENCY: {signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'].loc[probe_ts]:.4f}")
            print(f"        - SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL: {signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'].loc[probe_ts]:.4f}")
            print(f"        - pct_change_D: {signals_data['pct_change_D'].loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 价格行为疲软分数: {price_weakness_score.loc[probe_ts]:.4f}")
            print(f"        - 动量背离分数: {momentum_divergence_score.loc[probe_ts]:.4f}")
            print(f"        - 成交量异常分数: {volume_anomaly_score.loc[probe_ts]:.4f}")
            print(f"        - 日内控制力减弱分数: {intraday_control_weakness_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '滞涨证据'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_shakeout_confirmation(self, df: pd.DataFrame, absorption_strength: pd.Series, distribution_intent: pd.Series, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · Production Ready版】诊断震荡洗盘确认信号。
        - 核心重构: 废弃V2.1“事件审计员”模型，引入“意图×行动×品质”的全新三维诊断框架。
        - 诊断三维度:
          1. 战略意图 (Strategic Intent): 审判动机，必须满足“无派发意图”。
          2. 战术行动 (Tactical Action): 评估核心承接力量 (`absorption_strength`)。
          3. 执行品质 (Execution Quality): 评估洗盘技艺（效率、控制力、决断力）。
        - 数学模型: 确认分 = 战略意图 * (战术行动 * 执行品质) ^ 0.5
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_shakeout_confirmation"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '震荡洗盘确认信号' @ {probe_ts.strftime('%Y-%m-%d')}")
        params = get_param_value(self.config_params.get('grandmasters_protocol_params'), {})
        quality_weights = get_param_value(params.get('quality_weights'), {'efficiency': 0.4, 'control': 0.4, 'decisiveness': 0.2})
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        required_signals = [
            'floating_chip_cleansing_efficiency_D', 'vwap_control_strength_D',
            'high_D', 'low_D', 'close_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'floating_chip_cleansing_efficiency_D': (signals_data['floating_chip_cleansing_efficiency_D'], default_weights, True, False),
            'vwap_control_strength_D': (signals_data['vwap_control_strength_D'], 55, True, False) # normalize_score uses window
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w_or_window, asc, is_bipolar_flag) in series_for_mtf_norm_config.items():
            if isinstance(tf_w_or_window, dict): # get_adaptive_mtf_normalized_score
                normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w_or_window, ascending=asc, debug_info=False)
            else: # normalize_score
                normalized_mtf_scores[key] = normalize_score(series_obj, df.index, windows=tf_w_or_window, ascending=asc, debug_info=False)
        # --- 3. 计算各维度得分 ---
        strategic_intent_score = (1 - distribution_intent).clip(0, 1)
        tactical_action_score = absorption_strength
        efficiency_score = normalized_mtf_scores['floating_chip_cleansing_efficiency_D']
        control_score = normalized_mtf_scores['vwap_control_strength_D']
        high = signals_data['high_D']
        low = signals_data['low_D']
        close = signals_data['close_D']
        decisiveness_score = ((close - low) / (high - low + 1e-9)).fillna(0.5).clip(0, 1)
        execution_quality_score = (
            efficiency_score * quality_weights.get('efficiency', 0.4) +
            control_score * quality_weights.get('control', 0.4) +
            decisiveness_score * quality_weights.get('decisiveness', 0.2)
        ).clip(0, 1)
        # --- 4. “大国工匠协议”三维合成 ---
        base_confirmation = (tactical_action_score * execution_quality_score).pow(0.5).fillna(0.0)
        shakeout_confirmation_score = (strategic_intent_score * base_confirmation).clip(0, 1)
        final_score = shakeout_confirmation_score.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - floating_chip_cleansing_efficiency_D: {signals_data['floating_chip_cleansing_efficiency_D'].loc[probe_ts]:.4f}")
            print(f"        - vwap_control_strength_D: {signals_data['vwap_control_strength_D'].loc[probe_ts]:.4f}")
            print(f"        - high_D: {signals_data['high_D'].loc[probe_ts]:.4f}")
            print(f"        - low_D: {signals_data['low_D'].loc[probe_ts]:.4f}")
            print(f"        - close_D: {signals_data['close_D'].loc[probe_ts]:.4f}")
            print(f"        - absorption_strength (来自外部): {absorption_strength.loc[probe_ts]:.4f}")
            print(f"        - distribution_intent (来自外部): {distribution_intent.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 战略意图分数: {strategic_intent_score.loc[probe_ts]:.4f}")
            print(f"        - 战术行动分数: {tactical_action_score.loc[probe_ts]:.4f}")
            print(f"        - 执行品质分数: {execution_quality_score.loc[probe_ts]:.4f}")
            print(f"        - 基础确认分数: {base_confirmation.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '震荡洗盘确认信号'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _diagnose_pure_behavioral_divergence(self, df: pd.DataFrame, tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> Tuple[pd.Series, pd.Series]:
        """
        【V8.1 · 行为背离强度惯性与自适应引擎版】诊断纯粹基于行为类原始数据的看涨/看跌背离信号。
        - 核心重构: 提取看涨/看跌背离的公共计算逻辑到辅助方法，减少代码冗余，提高可读性和维护性。
        - 优化效率: 集中获取参数和信号，避免重复计算。
        - 清理探针: 移除所有调试打印，使代码更简洁。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_pure_behavioral_divergence"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '纯粹基于行为类原始数据的看涨/看跌背离信号' @ {probe_ts.strftime('%Y-%m-%d')}")
        # 1. 获取所有配置参数
        p_conf = self.config_params
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights_from_config = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
        params = { # 将所有参数打包成一个字典
            'mtf_slopes_params': get_param_value(p_conf.get('multi_timeframe_slopes'), {"enabled": True, "periods": [5, 13], "weights": {"5": 0.7, "13": 0.3}}),
            'multi_level_resonance_params': get_param_value(p_conf.get('multi_level_resonance_params'), {"enabled": True, "long_term_period": 21, "resonance_bonus": 0.2}),
            'persistence_params': get_param_value(p_conf.get('persistence_params'), {"enabled": True, "min_duration": 2, "max_duration_window": 5, "quality_decay_factor": 0.05}),
            'adaptive_thresholds_params': get_param_value(p_conf.get('adaptive_thresholds'), {"enabled": True, "min_slope_diff_atr_multiplier": 0.005, "min_slope_diff_base": 0.005, "rsi_oversold_trend_adjust_factor": 5, "rsi_overbought_trend_adjust_factor": 5}),
            'synergy_weights_params': get_param_value(p_conf.get('synergy_weights'), {"enabled": True, "two_indicators_synergy_bonus": 0.1, "three_indicators_synergy_bonus": 0.2}),
            'structural_context_weights_params': get_param_value(p_conf.get('structural_context_weights'), {"enabled": True, "bbw_slope_penalty_factor": 0.1}),
            'price_action_confirmation_params': get_param_value(p_conf.get('price_action_confirmation_params'), {"enabled": True, "lower_shadow_ratio_threshold": 0.4, "upper_shadow_ratio_threshold": 0.4, "body_ratio_threshold": 0.3, "confirmation_bonus": 0.15, "engulfing_pattern_bonus": 0.1, "hammer_shooting_star_bonus": 0.1}),
            'volume_price_structure_params': get_param_value(p_conf.get('volume_price_structure_params'), {"enabled": True, "volume_climax_threshold_multiplier": 2.0, "volume_drying_up_threshold_multiplier": 0.5, "narrow_range_body_ratio": 0.2, "wide_range_body_ratio": 0.7, "structure_bonus": 0.15}),
            'purity_assessment_params': get_param_value(p_conf.get('purity_assessment_params'), {"enabled": True, "slope_std_dev_threshold": 0.5, "whipsaw_penalty_factor": 0.1}),
            'market_regime_params': get_param_value(p_conf.get('market_regime_params'), {"enabled": True, "adx_trend_threshold": 25, "adx_ranging_threshold": 20, "adx_div_weight_max_adjust": 0.3, "adx_conf_weight_max_adjust": 0.3, "atr_conf_weight_max_adjust": 0.2}),
            'market_context_params': get_param_value(p_conf.get('market_context_params'), {"enabled": True, "price_momentum_window": 5, "favorable_sentiment_slope_threshold": 0.001, "unfavorable_sentiment_slope_threshold": -0.001, "slope_stability_threshold": 0.7, "adx_strength_threshold": 25, "favorable_context_bonus": 0.1, "unfavorable_context_penalty": 0.1}),
            'signal_freshness_params': get_param_value(p_conf.get('signal_freshness_params'), {"enabled": True, "freshness_bonus": 0.1}),
            'behavioral_consistency_params': get_param_value(p_conf.get('behavioral_consistency_params'), {"enabled": True, "min_consistent_indicators_for_bonus": 2, "conflict_penalty_threshold": 1, "consistency_bonus": 0.1, "conflict_penalty": 0.15}),
            'pattern_sequence_params': get_param_value(p_conf.get('pattern_sequence_params'), {"enabled": True, "lookback_window": 3, "volume_drying_up_ratio": 0.8, "volume_climax_ratio": 1.5, "reversal_pct_change_threshold": 0.01, "sequence_bonus": 0.2}),
            'behavioral_strength_params': get_param_value(p_conf.get('behavioral_strength_params'), {"enabled": True, "acceleration_bonus": 0.05}),
            'behavioral_inertia_params': get_param_value(p_conf.get('behavioral_inertia_params'), {"enabled": True, "long_term_adx_threshold": 30, "long_term_slope_stability_threshold": 0.8, "high_inertia_penalty": 0.1, "low_inertia_bonus": 0.05}),
            'adaptive_fusion_weights_params': get_param_value(p_conf.get('adaptive_fusion_weights_params'), {"enabled": True, "trend_strong_penalty_factor": 0.1, "ranging_bonus_factor": 0.1, "volatility_high_penalty_factor": 0.05}),
            'bullish_div_weights': get_param_value(p_conf.get('bullish_divergence_weights'), {"price_rsi": 0.4, "price_macd": 0.3, "price_volume": 0.3}),
            'bearish_div_weights': get_param_value(p_conf.get('bearish_divergence_weights'), {"price_rsi": 0.4, "price_macd": 0.3, "price_volume": 0.3}),
            'bullish_conf_weights': get_param_value(p_conf.get('bullish_confirmation_weights'), {"rsi_oversold": 0.3, "volume_increase": 0.3, "buying_support": 0.2, "volatility_high": 0.2}),
            'bearish_conf_weights': get_param_value(p_conf.get('bearish_confirmation_weights'), {"rsi_overbought": 0.3, "volume_decrease": 0.3, "selling_pressure": 0.2, "volatility_high": 0.2}),
            'rsi_oversold_threshold_base': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('rsi_oversold_threshold'), 30),
            'rsi_overbought_threshold_base': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('rsi_overbought_threshold'), 70),
            'min_divergence_slope_diff_base': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('min_slope_diff_base'), 0.005),
            'min_slope_diff_atr_multiplier': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('min_slope_diff_atr_multiplier'), 0.005),
            'rsi_oversold_trend_adjust_factor': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('rsi_oversold_trend_adjust_factor'), 5),
            'rsi_overbought_trend_adjust_factor': get_param_value(get_param_value(p_conf.get('adaptive_thresholds'), {}).get('rsi_overbought_trend_adjust_factor'), 5),
            'long_term_period': get_param_value(get_param_value(p_conf.get('multi_level_resonance_params'), {}).get('long_term_period'), 21)
        }
        # 2. 获取所需原始数据和派生信号
        mtf_periods = params['mtf_slopes_params'].get('periods', [5])
        accel_period = mtf_periods[0]
        required_signals = [
            'close_D', 'RSI_13_D', 'MACDh_13_34_8_D', 'volume_D', 'ATR_14_D', 'BBW_21_2.0_D',
            'active_buying_support_D', 'active_selling_pressure_D', 'trend_vitality_index_D',
            'open_D', 'high_D', 'low_D', 'ADX_14_D', 'VOL_MA_21_D', 'pct_change_D',
            'BIAS_5_D', 'BBP_21_2.0_D',
            'ACCEL_5_pct_change_D',
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW', 'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW',
            'robust_close_slope', 'robust_RSI_13_slope', 'robust_MACDh_13_34_8_slope', 'robust_volume_slope',
            'robust_BBW_21_2.0_slope', 'robust_pct_change_slope',
            'long_term_close_slope', 'long_term_RSI_13_slope', 'long_term_MACDh_13_34_8_slope', 'long_term_volume_slope',
            'long_term_adx_slope',
            'pattern_close_slope', 'pattern_volume_slope',
            f'ACCEL_{accel_period}_close_D',
            f'ACCEL_{accel_period}_RSI_13_D',
            f'ACCEL_{accel_period}_MACDh_13_34_8_D',
            f'ACCEL_{accel_period}_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name, is_debug_enabled, probe_ts):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号，返回默认值。")
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        # For purity_factor
        # 修正：从 params 中获取 mtf_slopes_params
        mtf_slopes_params_local = params['mtf_slopes_params']
        slope_std_dev_raw = signals_data['robust_close_slope'].rolling(window=mtf_slopes_params_local.get("periods", [5])[0]).std().fillna(0)
        # For inertia_factor
        long_term_close_slopes_series = signals_data['long_term_close_slope'] # Already a Series
        long_term_slope_std_dev_raw = long_term_close_slopes_series.rolling(window=params['long_term_period']).std().fillna(0)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_norm_config = {
            'robust_close_slope_for_purity': (signals_data['robust_close_slope'], 55, True), # For purity_factor
            'robust_pct_change_slope': (signals_data['robust_pct_change_slope'], 55, True), # For market_context_params
            'ATR_14_D': (signals_data['ATR_14_D'], tf_weights, True), # For norm_atr
            'ADX_14_D': (signals_data['ADX_14_D'], 55, True), # For market_regime_params
            'RSI_13_D': (signals_data['RSI_13_D'], 55, True), # For rsi_conf
            'robust_volume_slope_for_conf': (signals_data['robust_volume_slope'], tf_weights, True), # For volume_change_conf
            'slope_std_dev_raw': (slope_std_dev_raw, tf_weights, True), # For purity_factor
            'long_term_slope_std_dev_raw': (long_term_slope_std_dev_raw, tf_weights, True), # For inertia_factor
            'trend_vitality_index_D': (signals_data['trend_vitality_index_D'], tf_weights, True), # 已经添加
            'active_buying_support_D': (signals_data['active_buying_support_D'], tf_weights, True), # 新增此处
            'active_selling_pressure_D': (signals_data['active_selling_pressure_D'], tf_weights, True) # 新增此处
        }
        # 批量计算所有多时间框架归一化分数
        normalized_scores = {}
        for key, (series_obj, tf_w_or_window, asc) in series_for_norm_config.items():
            if isinstance(tf_w_or_window, dict): # get_adaptive_mtf_normalized_score
                normalized_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w_or_window, ascending=asc, debug_info=False)
            else: # normalize_score
                normalized_scores[key] = normalize_score(series_obj, df.index, windows=tf_w_or_window, ascending=asc, debug_info=False)
        # 集中获取所有信号
        robust_close_slope = signals_data['robust_close_slope']
        robust_rsi_slope = signals_data['robust_RSI_13_slope']
        robust_macd_slope = signals_data['robust_MACDh_13_34_8_slope']
        robust_volume_slope = signals_data['robust_volume_slope']
        robust_bbw_slope = signals_data['robust_BBW_21_2.0_slope']
        robust_pct_change_slope = signals_data['robust_pct_change_slope']
        long_term_close_slope = signals_data['long_term_close_slope']
        long_term_rsi_slope = signals_data['long_term_RSI_13_slope']
        long_term_macd_slope = signals_data['long_term_MACDh_13_34_8_slope']
        long_term_volume_slope = signals_data['long_term_volume_slope']
        long_term_adx_slope = signals_data['long_term_adx_slope']
        pattern_close_slope = signals_data['pattern_close_slope']
        pattern_volume_slope = signals_data['pattern_volume_slope']
        accel_close = signals_data[f'ACCEL_{accel_period}_close_D']
        accel_rsi = signals_data[f'ACCEL_{accel_period}_RSI_13_D']
        accel_macd = signals_data[f'ACCEL_{accel_period}_MACDh_13_34_8_D']
        accel_volume = signals_data[f'ACCEL_{accel_period}_volume_D']
        rsi_val = signals_data['RSI_13_D']
        atr_val = signals_data['ATR_14_D']
        active_buying = signals_data['active_buying_support_D']
        active_selling = signals_data['active_selling_pressure_D']
        raw_trend_vitality = signals_data['trend_vitality_index_D']
        trend_vitality = normalized_scores['trend_vitality_index_D']
        open_price = signals_data['open_D']
        high_price = signals_data['high_D']
        low_price = signals_data['low_D']
        close_price = signals_data['close_D']
        adx_val = signals_data['ADX_14_D']
        current_volume = signals_data['volume_D']
        volume_avg = signals_data['VOL_MA_21_D']
        pct_change_val = signals_data['pct_change_D']
        upward_efficiency_val = signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY']
        intraday_bull_control_val = signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL']
        internal_price_overextension_raw = signals_data['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW']
        internal_stagnation_evidence_raw = signals_data['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW']
        # 归一化确认因子 (0到1)
        norm_active_buying = normalized_scores['active_buying_support_D']
        norm_active_selling = normalized_scores['active_selling_pressure_D']
        norm_atr = normalized_scores['ATR_14_D']
        # 动态阈值计算
        dynamic_min_divergence_slope_diff = params['min_divergence_slope_diff_base'] + atr_val * params['min_slope_diff_atr_multiplier']
        rsi_oversold_threshold_dynamic = params['rsi_oversold_threshold_base'] - (trend_vitality - 0.5) * params['rsi_oversold_trend_adjust_factor']
        rsi_overbought_threshold_dynamic = params['rsi_overbought_threshold_base'] + (trend_vitality - 0.5) * params['rsi_overbought_trend_adjust_factor']
        # 3. 调用辅助方法计算看涨和看跌背离
        bullish_divergence_score = self._calculate_single_divergence_type(
            df, True, params, tf_weights, signals_data, # 传递 signals_data
            robust_close_slope, robust_rsi_slope, robust_macd_slope, robust_volume_slope, robust_bbw_slope, robust_pct_change_slope,
            long_term_close_slope, long_term_rsi_slope, long_term_macd_slope, long_term_volume_slope, long_term_adx_slope,
            pattern_close_slope, pattern_volume_slope,
            accel_close, accel_rsi, accel_macd, accel_volume,
            rsi_val, atr_val, active_buying, active_selling, trend_vitality,
            open_price, high_price, low_price, close_price, adx_val, current_volume, volume_avg, pct_change_val,
            upward_efficiency_val, intraday_bull_control_val,
            internal_price_overextension_raw, internal_stagnation_evidence_raw,
            norm_active_buying, norm_active_selling, norm_atr,
            dynamic_min_divergence_slope_diff, rsi_oversold_threshold_dynamic, rsi_overbought_threshold_dynamic,
            is_debug_enabled, probe_ts # 传递 debug_info_tuple
        )
        bearish_divergence_score = self._calculate_single_divergence_type(
            df, False, params, tf_weights, signals_data, # 传递 signals_data
            robust_close_slope, robust_rsi_slope, robust_macd_slope, robust_volume_slope, robust_bbw_slope, robust_pct_change_slope,
            long_term_close_slope, long_term_rsi_slope, long_term_macd_slope, long_term_volume_slope, long_term_adx_slope,
            pattern_close_slope, pattern_volume_slope,
            accel_close, accel_rsi, accel_macd, accel_volume,
            rsi_val, atr_val, active_buying, active_selling, trend_vitality,
            open_price, high_price, low_price, close_price, adx_val, current_volume, volume_avg, pct_change_val,
            upward_efficiency_val, intraday_bull_control_val,
            internal_price_overextension_raw, internal_stagnation_evidence_raw,
            norm_active_buying, norm_active_selling, norm_atr,
            dynamic_min_divergence_slope_diff, rsi_oversold_threshold_dynamic, rsi_overbought_threshold_dynamic,
            is_debug_enabled, probe_ts # 传递 debug_info_tuple
        )
        final_bullish_score = bullish_divergence_score.astype(np.float32)
        final_bearish_score = bearish_divergence_score.astype(np.float32)
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
            # 遍历 signals_data 字典并打印其内容
            for sig_name, sig_series in signals_data.items():
                if probe_ts in sig_series.index:
                    print(f"        - {sig_name}: {sig_series.loc[probe_ts]:.4f}")
                else:
                    print(f"        - {sig_name}: N/A (probe_ts not in index)")
            print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
            print(f"        - 动态最小背离斜率差异: {dynamic_min_divergence_slope_diff.loc[probe_ts]:.4f}")
            print(f"        - 动态RSI超卖阈值: {rsi_oversold_threshold_dynamic.loc[probe_ts]:.4f}")
            print(f"        - 动态RSI超买阈值: {rsi_overbought_threshold_dynamic.loc[probe_ts]:.4f}")
            print(f"        - 归一化活跃买盘: {norm_active_buying.loc[probe_ts]:.4f}")
            print(f"        - 归一化活跃卖盘: {norm_active_selling.loc[probe_ts]:.4f}")
            print(f"        - 归一化ATR: {norm_atr.loc[probe_ts]:.4f}")
            print(f"        - 趋势活力指数 (归一化): {trend_vitality.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '看涨背离'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_bullish_score.loc[probe_ts]:.4f}")
            print(f"      [探针 - {method_name}] 最终 '看跌背离'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_bearish_score.loc[probe_ts]:.4f}")
        return final_bullish_score, final_bearish_score

    def _calculate_single_divergence_type(self, df: pd.DataFrame, is_bullish: bool, params: Dict, tf_weights: Dict, signals_data: Dict[str, pd.Series],
                                          robust_close_slope, robust_rsi_slope, robust_macd_slope, robust_volume_slope, robust_bbw_slope, robust_pct_change_slope,
                                          long_term_close_slope, long_term_rsi_slope, long_term_macd_slope, long_term_volume_slope, long_term_adx_slope,
                                          pattern_close_slope, pattern_volume_slope,
                                          accel_close, accel_rsi, accel_macd, accel_volume,
                                          rsi_val, atr_val, active_buying, active_selling, trend_vitality,
                                          open_price, high_price, low_price, close_price, adx_val, current_volume, volume_avg, pct_change_val,
                                          upward_efficiency_val, intraday_bull_control_val,
                                          internal_price_overextension_raw, internal_stagnation_evidence_raw,
                                          norm_active_buying, norm_active_selling, norm_atr, # 这些参数已经传入，无需再次从 normalized_scores 获取
                                          dynamic_min_divergence_slope_diff, rsi_oversold_threshold_dynamic, rsi_overbought_threshold_dynamic,
                                          is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        method_name = "_calculate_single_divergence_type"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 {'看涨' if is_bullish else '看跌'} 背离类型 @ {probe_ts.strftime('%Y-%m-%d')}")
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # 修正：从 params 字典中获取 mtf_slopes_params
        mtf_slopes_params = params['mtf_slopes_params']
        multi_level_resonance_params = params['multi_level_resonance_params']
        persistence_params = params['persistence_params']
        adaptive_thresholds_params = params['adaptive_thresholds_params']
        synergy_weights_params = params['synergy_weights_params']
        structural_context_weights_params = params['structural_context_weights_params']
        price_action_confirmation_params = params['price_action_confirmation_params']
        volume_price_structure_params = params['volume_price_structure_params']
        purity_assessment_params = params['purity_assessment_params']
        market_regime_params = params['market_regime_params']
        market_context_params = params['market_context_params']
        signal_freshness_params = params['signal_freshness_params']
        behavioral_consistency_params = params['behavioral_consistency_params']
        pattern_sequence_params = params['pattern_sequence_params']
        behavioral_strength_params = params['behavioral_strength_params']
        behavioral_inertia_params = params['behavioral_inertia_params']
        adaptive_fusion_weights_params = params['adaptive_fusion_weights_params']
        bullish_div_weights = params['bullish_div_weights']
        bearish_div_weights = params['bearish_div_weights']
        bullish_conf_weights = params['bullish_conf_weights']
        bearish_conf_weights = params['bearish_conf_weights']
        long_term_period = params['long_term_period']
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        # For purity_factor
        # 修正：从 params 中获取 mtf_slopes_params
        mtf_slopes_params_local = params['mtf_slopes_params']
        slope_std_dev_raw = robust_close_slope.rolling(window=mtf_slopes_params_local.get("periods", [5])[0]).std().fillna(0)
        # For inertia_factor
        long_term_close_slopes_series = long_term_close_slope # Already a Series
        long_term_slope_std_dev_raw = long_term_close_slopes_series.rolling(window=long_term_period).std().fillna(0)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_norm_config = {
            'robust_close_slope_for_purity': (robust_close_slope, 55, True), # For purity_factor
            'robust_pct_change_slope': (robust_pct_change_slope, 55, True), # For market_context_params
            # 'ATR_14_D': (atr_val, tf_weights, True), # For norm_atr - 移除，因为 norm_atr 已作为参数传入
            'ADX_14_D': (adx_val, 55, True), # For market_regime_params
            'RSI_13_D': (rsi_val, 55, True), # For rsi_conf
            'robust_volume_slope_for_conf': (robust_volume_slope, tf_weights, True), # For volume_change_conf
            'slope_std_dev_raw': (slope_std_dev_raw, tf_weights, True), # For purity_factor
            'long_term_slope_std_dev_raw': (long_term_slope_std_dev_raw, tf_weights, True) # For inertia_factor
            # 已经传入的参数不再重复添加
        }
        # 批量计算所有多时间框架归一化分数
        normalized_scores = {}
        for key, (series_obj, tf_w_or_window, asc) in series_for_norm_config.items():
            if isinstance(tf_w_or_window, dict): # get_adaptive_mtf_normalized_score
                normalized_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w_or_window, ascending=asc, debug_info=False)
            else: # normalize_score
                normalized_scores[key] = normalize_score(series_obj, df.index, windows=tf_w_or_window, ascending=asc, debug_info=False)
        # 集中获取所有信号 (这些信号已经作为参数传入，这里只是为了调试打印方便)
        # robust_close_slope = signals_data['robust_close_slope'] # 已经作为参数传入
        # ... (其他已传入的信号，此处省略)
        # 归一化确认因子 (0到1) - 这些值已作为参数传入，无需再次从 normalized_scores 获取
        # norm_active_buying = normalized_scores['active_buying_support_D'] # 移除此行
        # norm_active_selling = normalized_scores['active_selling_pressure_D'] # 移除此行
        # norm_atr = normalized_scores['ATR_14_D'] # 移除此行
        # 动态阈值计算
        dynamic_min_divergence_slope_diff = params['min_divergence_slope_diff_base'] + atr_val * params['min_slope_diff_atr_multiplier']
        rsi_oversold_threshold_dynamic = params['rsi_oversold_threshold_base'] - (trend_vitality - 0.5) * params['rsi_oversold_trend_adjust_factor']
        rsi_overbought_threshold_dynamic = params['rsi_overbought_threshold_base'] + (trend_vitality - 0.5) * params['rsi_overbought_trend_adjust_factor']
        price_trend_condition = (robust_close_slope < 0) if is_bullish else (robust_close_slope > 0)
        rsi_indicator_trend = (robust_rsi_slope > 0) if is_bullish else (robust_rsi_slope < 0)
        macd_indicator_trend = (robust_macd_slope > 0) if is_bullish else (robust_macd_slope < 0)
        volume_indicator_trend = (robust_volume_slope > 0) if is_bullish else (robust_volume_slope < 0)
        div_condition_raw = price_trend_condition & (rsi_indicator_trend | macd_indicator_trend | volume_indicator_trend)
        accelerated_strength = pd.Series(0.0, index=df.index, dtype=np.float32)
        if behavioral_strength_params.get('enabled'):
            acceleration_bonus = behavioral_strength_params.get('acceleration_bonus', 0.05)
            accel_period = mtf_slopes_params.get("periods", [5])[0]
            accel_rsi_condition = (accel_rsi > 0) if is_bullish else (accel_rsi < 0)
            accel_macd_condition = (accel_macd > 0) if is_bullish else (accel_macd < 0)
            accel_volume_condition = (accel_volume > 0) if is_bullish else (accel_volume < 0)
            accel_strength_rsi = (rsi_indicator_trend & accel_rsi_condition).astype(int) * acceleration_bonus
            accel_strength_macd = (macd_indicator_trend & accel_macd_condition).astype(int) * acceleration_bonus
            accel_strength_volume = (volume_indicator_trend & accel_volume_condition).astype(int) * acceleration_bonus
            accelerated_strength = (accel_strength_rsi + accel_strength_macd + accel_strength_volume).clip(0, 0.15)
        # --- 修正背离强度计算逻辑 ---
        # 背离强度应该体现价格斜率的绝对值与指标斜率的绝对值之和，当背离条件满足时
        div_strength_rsi = pd.Series(0.0, index=df.index, dtype=np.float32)
        div_strength_macd = pd.Series(0.0, index=df.index, dtype=np.float32)
        div_strength_volume = pd.Series(0.0, index=df.index, dtype=np.float32)
        # 只有当价格趋势条件和指标趋势条件都满足时，才计算背离强度
        div_strength_rsi = div_strength_rsi.mask(
            price_trend_condition & rsi_indicator_trend,
            robust_close_slope.abs() + robust_rsi_slope.abs()
        )
        div_strength_macd = div_strength_macd.mask(
            price_trend_condition & macd_indicator_trend,
            robust_close_slope.abs() + robust_macd_slope.abs()
        )
        div_strength_volume = div_strength_volume.mask(
            price_trend_condition & volume_indicator_trend,
            robust_close_slope.abs() + robust_volume_slope.abs()
        )
        div_weights = bullish_div_weights if is_bullish else bearish_div_weights
        total_div_strength = (
            div_strength_rsi * div_weights.get('price_rsi', 0.4) +
            div_strength_macd * div_weights.get('price_macd', 0.3) +
            div_strength_volume * div_weights.get('price_volume', 0.3)
        )
        total_div_strength = total_div_strength.where(total_div_strength > dynamic_min_divergence_slope_diff, 0.0)
        norm_total_div_strength = normalize_score(total_div_strength, df.index, windows=55, ascending=True, debug_info=False) # Use normalize_score
        final_strength_factor = norm_total_div_strength * (1 + accelerated_strength)
        final_strength_factor = final_strength_factor.clip(0, 1.5)
        conf_weights = bullish_conf_weights if is_bullish else bearish_conf_weights
        if is_bullish:
            rsi_conf = normalize_score((rsi_oversold_threshold_dynamic - rsi_val).clip(lower=0), df.index, windows=55, ascending=True, debug_info=False) # Use normalize_score
            volume_change_conf = normalize_score(robust_volume_slope.clip(lower=0), df.index, windows=55, ascending=True, debug_info=False) # Use normalize_score
            active_flow_conf = norm_active_buying # 直接使用传入的参数
        else:
            rsi_conf = normalize_score((rsi_val - rsi_overbought_threshold_dynamic).clip(lower=0), df.index, windows=55, ascending=True, debug_info=False) # Use normalize_score
            volume_change_conf = normalize_score(robust_volume_slope.clip(upper=0).abs(), df.index, windows=55, ascending=True, debug_info=False) # Use normalize_score
            active_flow_conf = norm_active_selling # 直接使用传入的参数
        total_conf_factor = (
            rsi_conf * conf_weights.get('rsi_oversold' if is_bullish else 'rsi_overbought', 0.3) +
            volume_change_conf * conf_weights.get('volume_increase' if is_bullish else 'volume_decrease', 0.3) +
            active_flow_conf * conf_weights.get('buying_support' if is_bullish else 'selling_pressure', 0.2) +
            norm_atr * conf_weights.get('volatility_high', 0.2) # 直接使用传入的 norm_atr 参数
        )
        norm_total_conf_factor = normalize_score(total_conf_factor, df.index, windows=55, ascending=True, debug_info=False) # Use normalize_score
        persistence_factor = pd.Series(0.0, index=df.index, dtype=np.float32)
        if persistence_params.get('enabled'):
            min_persistence_duration = persistence_params.get('min_duration', 2)
            max_persistence_window = persistence_params.get('max_duration_window', 5)
            quality_decay_factor = persistence_params.get('quality_decay_factor', 0.05)
            accel_close_condition = (accel_close > 0) if is_bullish else (accel_close < 0)
            accel_volume_condition = (accel_volume > 0) if is_bullish else (accel_volume < 0) # Volume acceleration for bearish is also negative
            persistence_quality = (accel_close_condition.astype(int) + accel_volume_condition.astype(int)).clip(0, 2)
            persistence_count = div_condition_raw.astype(int).rolling(window=max_persistence_window, min_periods=1).sum().fillna(0)
            persistence_factor = (persistence_count / max_persistence_window) * (1 + persistence_quality * quality_decay_factor * persistence_count)
            persistence_factor = persistence_factor.where(persistence_count >= min_persistence_duration, 0.0)
            persistence_factor = persistence_factor.clip(0, 1.5)
        num_indicators_diverging = rsi_indicator_trend.astype(int) + macd_indicator_trend.astype(int) + volume_indicator_trend.astype(int)
        synergy_bonus = pd.Series(0.0, index=df.index, dtype=np.float32)
        if synergy_weights_params.get('enabled'):
            synergy_bonus = synergy_bonus.mask(num_indicators_diverging == 2, synergy_weights_params.get('two_indicators_synergy_bonus', 0.1))
            synergy_bonus = synergy_bonus.mask(num_indicators_diverging >= 3, synergy_weights_params.get('three_indicators_synergy_bonus', 0.2))
        synergy_factor = (1 + synergy_bonus).clip(1, 1.5)
        bbw_slope_penalty = pd.Series(0.0, index=df.index, dtype=np.float32)
        if structural_context_weights_params.get('enabled'):
            bbw_slope_penalty = robust_bbw_slope.clip(lower=0) * structural_context_weights_params.get('bbw_slope_penalty_factor', 0.1)
        structural_context_factor = (1 - normalize_score(bbw_slope_penalty, df.index, windows=55, ascending=True, debug_info=False)).clip(0.5, 1) # Use normalize_score
        resonance_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if multi_level_resonance_params.get('enabled'):
            long_term_price_trend = (long_term_close_slope < 0) if is_bullish else (long_term_close_slope > 0)
            long_term_rsi_trend = (long_term_rsi_slope > 0) if is_bullish else (long_term_rsi_slope < 0)
            long_term_macd_trend = (long_term_macd_slope > 0) if is_bullish else (long_term_macd_slope < 0)
            long_term_volume_trend = (long_term_volume_slope > 0) if is_bullish else (long_term_volume_slope < 0)
            long_term_div_condition = long_term_price_trend & (long_term_rsi_trend | long_term_macd_trend | long_term_volume_trend)
            resonance_factor = resonance_factor.mask(div_condition_raw & long_term_div_condition, 1 + multi_level_resonance_params.get('resonance_bonus', 0.2))
        price_action_conf = pd.Series(0.0, index=df.index, dtype=np.float32)
        if price_action_confirmation_params.get('enabled'):
            body_range = (close_price - open_price).abs()
            total_range = high_price - low_price
            total_range_safe = total_range.replace(0, 1e-9)
            lower_shadow = np.minimum(open_price, close_price) - low_price
            upper_shadow = high_price - np.maximum(open_price, close_price)
            # 显式计算并定义这些比率
            lower_shadow_ratio = (lower_shadow / total_range_safe).clip(0, 1)
            upper_shadow_ratio = (upper_shadow / total_range_safe).clip(0, 1)
            body_ratio = (body_range / total_range_safe).clip(0, 1)
            is_long_lower_shadow = (lower_shadow_ratio > price_action_confirmation_params.get('lower_shadow_ratio_threshold', 0.4))
            is_long_upper_shadow = (upper_shadow_ratio > price_action_confirmation_params.get('upper_shadow_ratio_threshold', 0.4))
            is_small_body = (body_ratio < price_action_confirmation_params.get('body_ratio_threshold', 0.3))
            if is_bullish:
                is_engulfing = (
                    (close_price > open_price) &
                    (df['close_D'].shift(1) < df['open_D'].shift(1)) &
                    (close_price > df['open_D'].shift(1)) &
                    (open_price < df['close_D'].shift(1))
                )
                is_reversal_candle = (
                    (close_price > open_price) &
                    (is_long_lower_shadow) &
                    (is_small_body) &
                    (upper_shadow_ratio < 0.1)
                )
            else:
                is_engulfing = (
                    (close_price < open_price) &
                    (df['close_D'].shift(1) > df['open_D'].shift(1)) &
                    (close_price < df['open_D'].shift(1)) &
                    (open_price > df['close_D'].shift(1))
                )
                is_reversal_candle = (
                    (close_price < open_price) &
                    (is_long_upper_shadow) &
                    (is_small_body) &
                    (lower_shadow_ratio < 0.1) # 现在 lower_shadow_ratio 已被定义
                )
            price_action_conf = price_action_conf.mask((is_long_lower_shadow if is_bullish else is_long_upper_shadow) & is_small_body, price_action_confirmation_params.get('confirmation_bonus', 0.15))
            price_action_conf = price_action_conf.mask(is_engulfing, price_action_confirmation_params.get('engulfing_pattern_bonus', 0.1))
            price_action_conf = price_action_conf.mask(is_reversal_candle, price_action_confirmation_params.get('hammer_shooting_star_bonus', 0.1))
        price_action_factor = (1 + price_action_conf).clip(1, 1.5)
        volume_price_structure_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if volume_price_structure_params.get('enabled'):
            vol_climax_mult = volume_price_structure_params.get('volume_climax_threshold_multiplier', 2.0)
            vol_drying_mult = volume_price_structure_params.get('volume_drying_up_threshold_multiplier', 0.5)
            narrow_body_ratio = volume_price_structure_params.get('narrow_range_body_ratio', 0.2)
            wide_body_ratio = volume_price_structure_params.get('wide_range_body_ratio', 0.7)
            structure_bonus = volume_price_structure_params.get('structure_bonus', 0.15)
            is_volume_drying_up = (current_volume < volume_avg * vol_drying_mult)
            is_volume_climax = (current_volume > volume_avg * vol_climax_mult)
            total_range_safe = (high_price - low_price).replace(0, 1e-9)
            body_ratio = ((close_price - open_price).abs() / total_range_safe).clip(0, 1)
            is_narrow_range_bar = (body_ratio < narrow_body_ratio)
            is_wide_range_bar = (body_ratio > wide_body_ratio)
            volume_price_structure_conf = pd.Series(0.0, index=df.index, dtype=np.float32)
            if is_bullish:
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    is_volume_drying_up & is_narrow_range_bar, structure_bonus
                )
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    price_trend_condition & is_volume_climax & (close_price > open_price), structure_bonus
                )
            else:
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    is_volume_drying_up & is_wide_range_bar, structure_bonus
                )
                volume_price_structure_conf = volume_price_structure_conf.mask(
                    price_trend_condition & is_volume_climax & (close_price < open_price), structure_bonus
                )
            volume_price_structure_factor = (1 + volume_price_structure_conf).clip(1, 1.5)
        purity_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if purity_assessment_params.get('enabled'):
            # short_term_close_slopes = robust_close_slope # Already calculated
            # slope_std_dev = short_term_close_slopes.rolling(window=mtf_slopes_params.get("periods", [5])[0]).std().fillna(0)
            norm_slope_std_dev = normalized_scores['slope_std_dev_raw'] # Use normalized_scores
            purity_penalty = pd.Series(0.0, index=df.index, dtype=np.float32)
            purity_penalty = purity_penalty.mask(
                price_trend_condition & (norm_slope_std_dev > purity_assessment_params.get('slope_std_dev_threshold', 0.5)),
                norm_slope_std_dev * purity_assessment_params.get('whipsaw_penalty_factor', 0.1)
            )
            purity_factor = (1 - purity_penalty).clip(0.5, 1)
        dynamic_div_weight_multiplier = pd.Series(1.0, index=df.index, dtype=np.float32)
        dynamic_conf_weight_multiplier = pd.Series(1.0, index=df.index, dtype=np.float32)
        if market_regime_params.get('enabled'):
            adx_trend_threshold = market_regime_params.get('adx_trend_threshold', 25)
            adx_ranging_threshold = market_regime_params.get('adx_ranging_threshold', 20)
            adx_div_max_adjust = market_regime_params.get('adx_div_weight_max_adjust', 0.3)
            adx_conf_max_adjust = market_regime_params.get('adx_conf_weight_max_adjust', 0.3)
            norm_adx = normalized_scores['ADX_14_D'] # Use normalized_scores
            dynamic_div_weight_multiplier = 1 + norm_adx * adx_div_max_adjust
            dynamic_conf_weight_multiplier = dynamic_conf_weight_multiplier.mask(
                adx_val < adx_ranging_threshold, 1 + (1 - norm_adx) * adx_conf_max_adjust
            )
            dynamic_conf_weight_multiplier = dynamic_conf_weight_multiplier * (1 - norm_atr * market_regime_params.get('atr_conf_weight_max_adjust', 0.2))
            dynamic_conf_weight_multiplier = dynamic_conf_weight_multiplier.clip(0.5, 1.5)
        market_context_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if market_context_params.get('enabled'):
            favorable_sentiment_slope_threshold = market_context_params.get('favorable_sentiment_slope_threshold', 0.001)
            unfavorable_sentiment_slope_threshold = market_context_params.get('unfavorable_sentiment_slope_threshold', -0.001)
            slope_stability_threshold = market_context_params.get('slope_stability_threshold', 0.7)
            adx_strength_threshold = market_context_params.get('adx_strength_threshold', 25)
            favorable_context_bonus = market_context_params.get('favorable_context_bonus', 0.1)
            unfavorable_context_penalty = market_context_params.get('unfavorable_context_penalty', 0.1)
            is_favorable_sentiment = (robust_pct_change_slope > favorable_sentiment_slope_threshold)
            is_unfavorable_sentiment = (robust_pct_change_slope < unfavorable_sentiment_slope_threshold)
            is_healthy_trend = (adx_val > adx_strength_threshold) & ((1 - normalized_scores['slope_std_dev_raw']) > slope_stability_threshold) # Use normalized_scores
            market_context_bonus_penalty = pd.Series(0.0, index=df.index, dtype=np.float32)
            if is_bullish:
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_favorable_sentiment & is_healthy_trend, favorable_context_bonus
                )
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_unfavorable_sentiment & ~is_healthy_trend, -unfavorable_context_penalty
                )
            else:
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_unfavorable_sentiment & is_healthy_trend, favorable_context_bonus
                )
                market_context_bonus_penalty = market_context_bonus_penalty.mask(
                    is_favorable_sentiment & ~is_healthy_trend, -unfavorable_context_penalty
                )
            market_context_factor = (1 + market_context_bonus_penalty).clip(0.5, 1.5)
        freshness_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if signal_freshness_params.get('enabled'):
            freshness_bonus = signal_freshness_params.get('freshness_bonus', 0.1)
            persistence_count = div_condition_raw.astype(int).rolling(window=persistence_params.get('max_duration_window', 5), min_periods=1).sum().fillna(0)
            freshness_factor = freshness_factor.mask(
                persistence_count == 1, 1 + freshness_bonus
            )
        freshness_factor = freshness_factor.clip(1, 1.5)
        consistency_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if behavioral_consistency_params.get('enabled'):
            min_consistent_indicators = behavioral_consistency_params.get('min_consistent_indicators_for_bonus', 2)
            conflict_penalty_threshold = behavioral_consistency_params.get('conflict_penalty_threshold', 1)
            consistency_bonus = behavioral_consistency_params.get('consistency_bonus', 0.1)
            conflict_penalty = behavioral_consistency_params.get('conflict_penalty', 0.15)
            diverging_indicators_count = rsi_indicator_trend.astype(int) + macd_indicator_trend.astype(int) + volume_indicator_trend.astype(int)
            conflicting_indicators_count = pd.Series(0, index=df.index, dtype=np.int32)
            if is_bullish:
                conflicting_indicators_count += (~rsi_indicator_trend & (robust_rsi_slope < 0)).astype(int)
                conflicting_indicators_count += (~macd_indicator_trend & (robust_macd_slope < 0)).astype(int)
                conflicting_indicators_count += (~volume_indicator_trend & (robust_volume_slope < 0)).astype(int)
            else:
                conflicting_indicators_count += (~rsi_indicator_trend & (robust_rsi_slope > 0)).astype(int)
                conflicting_indicators_count += (~macd_indicator_trend & (robust_macd_slope > 0)).astype(int)
                conflicting_indicators_count += (~volume_indicator_trend & (robust_volume_slope > 0)).astype(int)
            consistency_factor = consistency_factor.mask(
                diverging_indicators_count >= min_consistent_indicators, 1 + consistency_bonus
            )
            consistency_factor = consistency_factor.mask(
                (diverging_indicators_count < min_consistent_indicators) & (conflicting_indicators_count >= conflict_penalty_threshold),
                1 - conflict_penalty
            )
        consistency_factor = consistency_factor.clip(0.5, 1.5)
        pattern_sequence_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if pattern_sequence_params.get('enabled'):
            lookback_window = pattern_sequence_params.get('lookback_window', 3)
            volume_drying_up_ratio = pattern_sequence_params.get('volume_drying_up_ratio', 0.8)
            volume_climax_ratio = pattern_sequence_params.get('volume_climax_ratio', 1.5)
            reversal_pct_change_threshold = pattern_sequence_params.get('reversal_pct_change_threshold', 0.01)
            sequence_bonus = pattern_sequence_params.get('sequence_bonus', 0.2)
            price_trend_in_window = (pattern_close_slope < 0) if is_bullish else (pattern_close_slope > 0)
            volume_drying_up_in_window = (pattern_volume_slope < 0) & (current_volume < volume_avg * volume_drying_up_ratio)
            if is_bullish:
                current_bar_reversal = (pct_change_val > reversal_pct_change_threshold) & (current_volume > volume_avg * volume_climax_ratio) & (close_price > open_price)
            else:
                current_bar_reversal = (pct_change_val < -reversal_pct_change_threshold) & (current_volume > volume_avg * volume_climax_ratio) & (close_price < open_price)
            is_pattern_sequence = price_trend_in_window & volume_drying_up_in_window & current_bar_reversal
            pattern_sequence_factor = pattern_sequence_factor.mask(is_pattern_sequence, 1 + sequence_bonus)
        pattern_sequence_factor = pattern_sequence_factor.clip(1, 1.5)
        inertia_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if behavioral_inertia_params.get('enabled'):
            long_term_adx_threshold = behavioral_inertia_params.get('long_term_adx_threshold', 30)
            long_term_slope_stability_threshold = behavioral_inertia_params.get('long_term_slope_stability_threshold', 0.8)
            high_inertia_penalty = behavioral_inertia_params.get('high_inertia_penalty', 0.1)
            low_inertia_bonus = behavioral_inertia_params.get('low_inertia_bonus', 0.05)
            long_term_adx_mean = adx_val.rolling(long_term_period).mean()
            is_strong_long_term_trend = (long_term_adx_mean > long_term_adx_threshold)
            # long_term_close_slopes_series = signals_data[f'SLOPE_{long_term_period}_close_D'] # Already defined
            # long_term_slope_std_dev = long_term_close_slopes_series.rolling(window=long_term_period).std().fillna(0)
            norm_long_term_slope_std_dev = normalized_scores['long_term_slope_std_dev_raw'] # Use normalized_scores
            is_stable_long_term_slope = (norm_long_term_slope_std_dev > long_term_slope_stability_threshold)
            is_high_inertia_market = is_strong_long_term_trend & is_stable_long_term_slope
            is_low_inertia_market = ~is_strong_long_term_trend | ~is_stable_long_term_slope
            inertia_adjustment = pd.Series(0.0, index=df.index, dtype=np.float32)
            inertia_adjustment = inertia_adjustment.mask(is_high_inertia_market, -high_inertia_penalty)
            inertia_adjustment = inertia_adjustment.mask(is_low_inertia_market, low_inertia_bonus)
            inertia_factor = (1 + inertia_adjustment).clip(0.5, 1.5)
        adaptive_fusion_weight_multiplier = pd.Series(1.0, index=df.index, dtype=np.float32)
        if behavioral_inertia_params.get('enabled'):
            trend_strong_penalty_factor = adaptive_fusion_weights_params.get('trend_strong_penalty_factor', 0.1)
            ranging_bonus_factor = adaptive_fusion_weights_params.get('ranging_bonus_factor', 0.1)
            volatility_high_penalty_factor = adaptive_fusion_weights_params.get('volatility_high_penalty_factor', 0.05)
            is_strong_trend = (adx_val > market_regime_params.get('adx_trend_threshold', 25))
            adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.mask(
                is_strong_trend, adaptive_fusion_weight_multiplier * (1 - trend_strong_penalty_factor)
            )
            is_ranging_market = (adx_val < market_regime_params.get('adx_ranging_threshold', 20))
            adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.mask(
                is_ranging_market, adaptive_fusion_weight_multiplier * (1 + ranging_bonus_factor)
            )
            is_high_volatility = (norm_atr > 0.8) # 直接使用传入的 norm_atr 参数
            adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.mask(
                is_high_volatility, adaptive_fusion_weight_multiplier * (1 - volatility_high_penalty_factor)
            )
        adaptive_fusion_weight_multiplier = adaptive_fusion_weight_multiplier.clip(0.5, 1.5)
        divergence_score = (
            (final_strength_factor + 1e-9).pow(0.4 * dynamic_div_weight_multiplier) *
            (persistence_factor + 1e-9).pow(0.2) *
            (norm_total_conf_factor + 1e-9).pow(0.3 * dynamic_conf_weight_multiplier) *
            (synergy_factor + 1e-9).pow(0.1) *
            (structural_context_factor + 1e-9).pow(0.1) *
            (resonance_factor + 1e-9).pow(0.1) *
            (price_action_factor + 1e-9).pow(0.1) *
            (volume_price_structure_factor + 1e-9).pow(0.1) *
            (purity_factor + 1e-9).pow(0.1) *
            (market_context_factor + 1e-9).pow(0.1) *
            (freshness_factor + 1e-9).pow(0.1) *
            (consistency_factor + 1e-9).pow(0.1) *
            (pattern_sequence_factor + 1e-9).pow(0.1) *
            (inertia_factor + 1e-9).pow(0.1)
        ).pow(1 / (2.2 * adaptive_fusion_weight_multiplier)).fillna(0.0).clip(0, 1)
        divergence_score = divergence_score.where(div_condition_raw, 0.0)
        final_score = divergence_score.astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     # 遍历 signals_data 字典并打印其内容
        #     for sig_name, sig_series in signals_data.items():
        #         if probe_ts in sig_series.index:
        #             print(f"        - {sig_name}: {sig_series.loc[probe_ts]:.4f}")
        #         else:
        #             print(f"        - {sig_name}: N/A (probe_ts not in index)")
        #     print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - 价格趋势条件: {price_trend_condition.loc[probe_ts]}")
        #     print(f"        - RSI指标趋势: {rsi_indicator_trend.loc[probe_ts]}")
        #     print(f"        - MACD指标趋势: {macd_indicator_trend.loc[probe_ts]}")
        #     print(f"        - 成交量指标趋势: {volume_indicator_trend.loc[probe_ts]}")
        #     print(f"        - 背离原始条件: {div_condition_raw.loc[probe_ts]}")
        #     print(f"        - 加速强度: {accelerated_strength.loc[probe_ts]:.4f}")
        #     print(f"        - RSI背离强度: {div_strength_rsi.loc[probe_ts]:.4f}")
        #     print(f"        - MACD背离强度: {div_strength_macd.loc[probe_ts]:.4f}")
        #     print(f"        - 成交量背离强度: {div_strength_volume.loc[probe_ts]:.4f}")
        #     print(f"        - 总背离强度: {total_div_strength.loc[probe_ts]:.4f}")
        #     print(f"        - 最终强度因子: {final_strength_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 总确认因子: {norm_total_conf_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 持续性因子: {persistence_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 协同因子: {synergy_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 结构上下文因子: {structural_context_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 共振因子: {resonance_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 价格行为因子: {price_action_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 量价结构因子: {volume_price_structure_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 纯度因子: {purity_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 市场情境因子: {market_context_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 新鲜度因子: {freshness_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 一致性因子: {consistency_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 模式序列因子: {pattern_sequence_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 惯性因子: {inertia_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 自适应融合权重乘数: {adaptive_fusion_weight_multiplier.loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 最终 '背离信号'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_score.loc[probe_ts]:.4f}")
        return final_score

    def _calculate_lockup_rally_opportunity(self, df: pd.DataFrame, states: Dict[str, pd.Series], default_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.2 · 锁仓拉升深度洞察与多维情境感知版 - 生产就绪】计算锁仓拉升机会信号。
        - 核心升级: 将“锁仓拉升”解构为“上涨纯度”、“供应枯竭”、“主力控盘意图”和“情境共振”四大核心维度，
                      并引入多时间维度斜率与加速度、筹码结构、市场情绪等新原始数据，深化判断。
        - 目标: 识别在多头趋势中，主力资金高度控盘，市场抛压枯竭，且上涨动能纯粹、效率高，同时市场环境有利的锁仓拉升机会。
        - 【新增】在方法开始时加入对所有原料数据的存在性检查。
        - 【清理】移除所有调试探针代码，恢复生产状态。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_calculate_lockup_rally_opportunity"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '锁仓拉升机会信号' @ {probe_ts.strftime('%Y-%m-%d')}")
        p_behavioral_div_conf = self.config_params
        lockup_rally_params = get_param_value(p_behavioral_div_conf.get('lockup_rally_params'), {})
        lockup_rally_enabled = get_param_value(lockup_rally_params.get('enabled'), False)
        if not lockup_rally_enabled:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 信号未启用，返回0分。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        lockup_rally_fusion_weights = get_param_value(lockup_rally_params.get('fusion_weights'), {
            "rally_purity": 0.25, "supply_exhaustion": 0.25, "main_force_intent": 0.25, "context_resonance": 0.25
        })
        lockup_rally_rally_purity_weights = get_param_value(lockup_rally_params.get('rally_purity_weights'), {
            "upward_momentum": 0.3, "upward_efficiency": 0.3, "price_momentum_coherence": 0.4
        })
        lockup_rally_supply_exhaustion_weights = get_param_value(lockup_rally_params.get('supply_exhaustion_weights'), {
            "volume_atrophy": 0.3, "low_turnover": 0.2, "low_volatility_instability": 0.2, "supply_pressure_relief": 0.3
        })
        lockup_rally_main_force_intent_weights = get_param_value(lockup_rally_params.get('main_force_intent_weights'), {
            "intraday_bull_control": 0.3, "no_distribution_intent": 0.2, "low_deception_lure_short": 0.2, "mf_accumulation_conviction": 0.3
        })
        lockup_rally_context_resonance_weights = get_param_value(lockup_rally_params.get('context_modulator_weights'), { # Corrected key
            "low_overextension": 0.3, "low_stagnation_evidence": 0.2, "trend_alignment": 0.3, "market_sentiment_health": 0.2, "low_volatility_instability": 0.0 # Added low_volatility_instability
        })
        lockup_rally_final_exponent = get_param_value(lockup_rally_params.get('final_exponent'), 1.0)
        # 获取所有 tf_weights 配置，如果未配置则使用 default_weights
        turnover_rate_tf_weights = get_param_value(lockup_rally_params.get('turnover_rate_tf_weights'), default_weights)
        volatility_instability_tf_weights = get_param_value(lockup_rally_params.get('volatility_instability_tf_weights'), default_weights)
        deception_lure_short_tf_weights = get_param_value(lockup_rally_params.get('deception_lure_short_tf_weights'), default_weights)
        overextension_tf_weights = get_param_value(lockup_rally_params.get('overextension_tf_weights'), default_weights)
        stagnation_evidence_tf_weights = get_param_value(lockup_rally_params.get('stagnation_evidence_tf_weights'), default_weights)
        retail_fomo_tf_weights = get_param_value(lockup_rally_params.get('retail_fomo_tf_weights'), default_weights) # Added retail_fomo_tf_weights
        required_signals = [
            'pct_change_D', 'turnover_rate_f_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'deception_lure_short_intensity_D', 'chip_fatigue_index_D', 'loser_pain_index_D',
            'floating_chip_cleansing_efficiency_D', 'main_force_conviction_index_D',
            'covert_accumulation_signal_D', 'ADX_14_D', 'retail_fomo_premium_index_D',
            'robust_close_slope', 'robust_pct_change_slope', 'robust_RSI_13_slope',
            'robust_MACDh_13_34_8_slope', # Corrected from _slope to _D
            'robust_volume_slope',
            'long_term_close_slope', 'long_term_adx_slope',
            'ACCEL_5_close_D', 'ACCEL_5_RSI_13_D', 'ACCEL_5_MACDh_13_34_8_D', 'ACCEL_5_volume_D',
            'SLOPE_5_main_force_conviction_index_D'
        ]
        required_state_signals = [
            'SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM', 'SCORE_BEHAVIOR_UPWARD_EFFICIENCY',
            'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL',
            'SCORE_BEHAVIOR_DISTRIBUTION_INTENT', 'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW',
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'
        ]
        missing_df_signals = [s for s in required_signals if s not in df.columns]
        missing_state_signals = [s for s in required_state_signals if s not in states]
        if missing_df_signals or missing_state_signals:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号。")
                if missing_df_signals:
                    print(f"         - DataFrame中缺失: {missing_df_signals}")
                if missing_state_signals:
                    print(f"         - States中缺失: {missing_state_signals}")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_signals}
        # 集中提取所有必需的原子状态信号
        state_signals_data = {sig: states[sig] for sig in required_state_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        pct_change = signals_data['pct_change_D']
        is_rising = (pct_change > 0).astype(float)
        upward_momentum_score = state_signals_data['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM']
        upward_efficiency_score = state_signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY']
        volume_atrophy_score = state_signals_data['SCORE_BEHAVIOR_VOLUME_ATROPHY']
        intraday_bull_control_score = state_signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL']
        distribution_intent_score = state_signals_data['SCORE_BEHAVIOR_DISTRIBUTION_INTENT']
        price_overextension_raw = state_signals_data['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW']
        stagnation_evidence_raw = state_signals_data['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW']
        turnover_rate_raw = signals_data['turnover_rate_f_D']
        volatility_instability_raw = signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D']
        deception_lure_short_raw = signals_data['deception_lure_short_intensity_D']
        chip_fatigue_raw = signals_data['chip_fatigue_index_D']
        loser_pain_raw = signals_data['loser_pain_index_D']
        cleansing_efficiency_raw = signals_data['floating_chip_cleansing_efficiency_D']
        main_force_conviction_raw = signals_data['main_force_conviction_index_D']
        covert_accumulation_raw = signals_data['covert_accumulation_signal_D']
        adx_raw = signals_data['ADX_14_D']
        retail_fomo_raw = signals_data['retail_fomo_premium_index_D']
        robust_close_slope = signals_data['robust_close_slope']
        robust_pct_change_slope = signals_data['robust_pct_change_slope']
        robust_rsi_slope = signals_data['robust_RSI_13_slope']
        robust_macd_slope = signals_data['robust_MACDh_13_34_8_slope']
        robust_volume_slope = signals_data['robust_volume_slope']
        long_term_close_slope = signals_data['long_term_close_slope']
        long_term_adx_slope = signals_data['long_term_adx_slope']
        accel_close = signals_data['ACCEL_5_close_D']
        accel_rsi = signals_data['ACCEL_5_RSI_13_D']
        accel_macd = signals_data['ACCEL_5_MACDh_13_34_8_D']
        accel_volume = signals_data['ACCEL_5_volume_D']
        mf_conviction_slope_raw = signals_data['SLOPE_5_main_force_conviction_index_D']
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        robust_close_slope_clip_lower_0 = robust_close_slope.clip(lower=0)
        robust_pct_change_slope_clip_lower_0 = robust_pct_change_slope.clip(lower=0)
        robust_rsi_slope_clip_lower_0 = robust_rsi_slope.clip(lower=0)
        robust_macd_slope_clip_lower_0 = robust_macd_slope.clip(lower=0)
        long_term_close_slope_clip_lower_0 = long_term_close_slope.clip(lower=0)
        long_term_adx_slope_clip_lower_0 = long_term_adx_slope.clip(lower=0)
        accel_close_clip_lower_0 = accel_close.clip(lower=0)
        accel_rsi_clip_lower_0 = accel_rsi.clip(lower=0)
        accel_macd_clip_lower_0 = accel_macd.clip(lower=0)
        main_force_conviction_raw_clip_lower_0 = main_force_conviction_raw.clip(lower=0)
        mf_conviction_slope_raw_clip_lower_0 = mf_conviction_slope_raw.clip(lower=0)
        retail_fomo_raw_clip_lower_0 = retail_fomo_raw.clip(lower=0)
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'robust_close_slope_clip_lower_0': (robust_close_slope_clip_lower_0, default_weights, True),
            'robust_pct_change_slope_clip_lower_0': (robust_pct_change_slope_clip_lower_0, default_weights, True),
            'robust_rsi_slope_clip_lower_0': (robust_rsi_slope_clip_lower_0, default_weights, True),
            'robust_macd_slope_clip_lower_0': (robust_macd_slope_clip_lower_0, default_weights, True),
            'long_term_close_slope_clip_lower_0': (long_term_close_slope_clip_lower_0, default_weights, True),
            'long_term_adx_slope_clip_lower_0': (long_term_adx_slope_clip_lower_0, default_weights, True),
            'accel_close_clip_lower_0': (accel_close_clip_lower_0, default_weights, True),
            'accel_rsi_clip_lower_0': (accel_rsi_clip_lower_0, default_weights, True),
            'accel_macd_clip_lower_0': (accel_macd_clip_lower_0, default_weights, True),
            'chip_fatigue_index_D': (chip_fatigue_raw, default_weights, True),
            'loser_pain_index_D': (loser_pain_raw, default_weights, True),
            'floating_chip_cleansing_efficiency_D': (cleansing_efficiency_raw, default_weights, True),
            'turnover_rate_f_D': (turnover_rate_raw, turnover_rate_tf_weights, True),
            'VOLATILITY_INSTABILITY_INDEX_21d_D': (volatility_instability_raw, volatility_instability_tf_weights, True),
            'main_force_conviction_index_D_clip_lower_0': (main_force_conviction_raw_clip_lower_0, default_weights, True),
            'covert_accumulation_signal_D': (covert_accumulation_raw, default_weights, True),
            'SLOPE_5_main_force_conviction_index_D_clip_lower_0': (mf_conviction_slope_raw_clip_lower_0, default_weights, True),
            'ADX_14_D': (adx_raw, default_weights, True),
            'retail_fomo_premium_index_D_clip_lower_0': (retail_fomo_raw_clip_lower_0, retail_fomo_tf_weights, True),
            'INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW': (price_overextension_raw, overextension_tf_weights, True),
            'INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW': (stagnation_evidence_raw, stagnation_evidence_tf_weights, True),
            'deception_lure_short_intensity_D': (deception_lure_short_raw, deception_lure_short_tf_weights, True)
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        # --- 2. 计算四大维度分数 ---
        # Rally Purity
        price_momentum_coherence_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        norm_robust_close_slope = normalized_mtf_scores['robust_close_slope_clip_lower_0']
        norm_robust_pct_change_slope = normalized_mtf_scores['robust_pct_change_slope_clip_lower_0']
        norm_robust_rsi_slope = normalized_mtf_scores['robust_rsi_slope_clip_lower_0']
        norm_robust_macd_slope = normalized_mtf_scores['robust_macd_slope_clip_lower_0']
        norm_long_term_close_slope = normalized_mtf_scores['long_term_close_slope_clip_lower_0']
        norm_long_term_adx_slope = normalized_mtf_scores['long_term_adx_slope_clip_lower_0']
        norm_accel_close = normalized_mtf_scores['accel_close_clip_lower_0']
        norm_accel_rsi = normalized_mtf_scores['accel_rsi_clip_lower_0']
        norm_accel_macd = normalized_mtf_scores['accel_macd_clip_lower_0']
        price_momentum_coherence_score = (
            (norm_robust_close_slope + norm_robust_pct_change_slope + norm_robust_rsi_slope + norm_robust_macd_slope) / 4 *
            (norm_long_term_close_slope + norm_long_term_adx_slope) / 2 *
            (norm_accel_close + norm_accel_rsi + norm_accel_macd) / 3
        ).pow(1/3).clip(0,1)
        rally_purity_score = (
            (upward_momentum_score + 1e-9).pow(lockup_rally_rally_purity_weights.get('upward_momentum', 0.3)) *
            (upward_efficiency_score + 1e-9).pow(lockup_rally_rally_purity_weights.get('upward_efficiency', 0.3)) *
            (price_momentum_coherence_score + 1e-9).pow(lockup_rally_rally_purity_weights.get('price_momentum_coherence', 0.4))
        ).pow(1 / sum(lockup_rally_rally_purity_weights.values())).fillna(0.0)
        # Supply Exhaustion
        norm_chip_fatigue = normalized_mtf_scores['chip_fatigue_index_D']
        norm_loser_pain = normalized_mtf_scores['loser_pain_index_D']
        norm_cleansing_efficiency = normalized_mtf_scores['floating_chip_cleansing_efficiency_D']
        supply_pressure_relief_score = (
            (norm_chip_fatigue + 1e-9).pow(0.4) *
            (norm_loser_pain + 1e-9).pow(0.3) *
            (norm_cleansing_efficiency + 1e-9).pow(0.3)
        ).pow(1/3).clip(0,1)
        norm_low_turnover = (1 - normalized_mtf_scores['turnover_rate_f_D']).clip(0, 1)
        norm_low_volatility_instability = (1 - normalized_mtf_scores['VOLATILITY_INSTABILITY_INDEX_21d_D']).clip(0, 1)
        supply_exhaustion_score = (
            (volume_atrophy_score + 1e-9).pow(lockup_rally_supply_exhaustion_weights.get('volume_atrophy', 0.3)) *
            (norm_low_turnover + 1e-9).pow(lockup_rally_supply_exhaustion_weights.get('low_turnover', 0.2)) *
            (norm_low_volatility_instability + 1e-9).pow(lockup_rally_supply_exhaustion_weights.get('low_volatility_instability', 0.2)) *
            (supply_pressure_relief_score + 1e-9).pow(lockup_rally_supply_exhaustion_weights.get('supply_pressure_relief', 0.3))
        ).pow(1 / sum(lockup_rally_supply_exhaustion_weights.values())).fillna(0.0)
        # Main Force Intent
        norm_main_force_conviction = normalized_mtf_scores['main_force_conviction_index_D_clip_lower_0']
        norm_covert_accumulation = normalized_mtf_scores['covert_accumulation_signal_D']
        norm_mf_conviction_slope = normalized_mtf_scores['SLOPE_5_main_force_conviction_index_D_clip_lower_0']
        mf_accumulation_conviction_score = (
            (norm_main_force_conviction + 1e-9).pow(0.4) *
            (norm_covert_accumulation + 1e-9).pow(0.3) *
            (norm_mf_conviction_slope + 1e-9).pow(0.3)
        ).pow(1/3).clip(0,1)
        norm_no_distribution_intent = (1 - distribution_intent_score).clip(0, 1)
        norm_low_deception_lure_short = (1 - normalized_mtf_scores['deception_lure_short_intensity_D']).clip(0, 1)
        main_force_intent_score = (
            (intraday_bull_control_score + 1e-9).pow(lockup_rally_main_force_intent_weights.get('intraday_bull_control', 0.3)) *
            (norm_no_distribution_intent + 1e-9).pow(lockup_rally_main_force_intent_weights.get('no_distribution_intent', 0.2)) *
            (norm_low_deception_lure_short + 1e-9).pow(lockup_rally_main_force_intent_weights.get('low_deception_lure_short', 0.2)) *
            (mf_accumulation_conviction_score + 1e-9).pow(lockup_rally_main_force_intent_weights.get('mf_accumulation_conviction', 0.3))
        ).pow(1 / sum(lockup_rally_main_force_intent_weights.values())).fillna(0.0)
        # Context Resonance
        norm_long_term_close_slope_positive = normalized_mtf_scores['long_term_close_slope_clip_lower_0']
        norm_adx_strength = normalized_mtf_scores['ADX_14_D']
        trend_alignment_score = (norm_long_term_close_slope_positive * norm_adx_strength).pow(0.5).clip(0,1)
        norm_low_retail_fomo = (1 - normalized_mtf_scores['retail_fomo_premium_index_D_clip_lower_0']).clip(0,1)
        market_sentiment_health_score = norm_low_retail_fomo
        norm_low_overextension = (1 - normalized_mtf_scores['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW']).clip(0, 1)
        norm_low_stagnation_evidence = (1 - normalized_mtf_scores['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW']).clip(0, 1)
        norm_low_volatility_instability_for_context = (1 - normalized_mtf_scores['VOLATILITY_INSTABILITY_INDEX_21d_D']).clip(0, 1)
        context_resonance_score = (
            (norm_low_overextension + 1e-9).pow(lockup_rally_context_resonance_weights.get('low_overextension', 0.3)) *
            (norm_low_stagnation_evidence + 1e-9).pow(lockup_rally_context_resonance_weights.get('low_stagnation_evidence', 0.2)) *
            (trend_alignment_score + 1e-9).pow(lockup_rally_context_resonance_weights.get('trend_alignment', 0.3)) *
            (market_sentiment_health_score + 1e-9).pow(lockup_rally_context_resonance_weights.get('market_sentiment_health', 0.2)) *
            (norm_low_volatility_instability_for_context + 1e-9).pow(lockup_rally_context_resonance_weights.get('low_volatility_instability', 0.0))
        ).pow(1 / sum(lockup_rally_context_resonance_weights.values())).fillna(0.0).clip(0, 1)
        # --- 3. 核心融合 (加权几何平均) ---
        lockup_rally_score = (
            (rally_purity_score + 1e-9).pow(lockup_rally_fusion_weights.get('rally_purity', 0.25)) *
            (supply_exhaustion_score + 1e-9).pow(lockup_rally_fusion_weights.get('supply_exhaustion', 0.25)) *
            (main_force_intent_score + 1e-9).pow(lockup_rally_fusion_weights.get('main_force_intent', 0.25)) *
            (context_resonance_score + 1e-9).pow(lockup_rally_fusion_weights.get('context_resonance', 0.25))
        ).pow(1 / sum(lockup_rally_fusion_weights.values()))
        lockup_rally_score = lockup_rally_score.where(is_rising > 0, 0.0).clip(0, 1).astype(np.float32)
        final_lockup_rally_score = lockup_rally_score.pow(lockup_rally_final_exponent)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - pct_change_D: {signals_data['pct_change_D'].loc[probe_ts]:.4f}")
        #     print(f"        - turnover_rate_f_D: {signals_data['turnover_rate_f_D'].loc[probe_ts]:.4f}")
        #     print(f"        - VOLATILITY_INSTABILITY_INDEX_21d_D: {signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'].loc[probe_ts]:.4f}")
        #     print(f"        - deception_lure_short_intensity_D: {signals_data['deception_lure_short_intensity_D'].loc[probe_ts]:.4f}")
        #     print(f"        - chip_fatigue_index_D: {signals_data['chip_fatigue_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - loser_pain_index_D: {signals_data['loser_pain_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - floating_chip_cleansing_efficiency_D: {signals_data['floating_chip_cleansing_efficiency_D'].loc[probe_ts]:.4f}")
        #     print(f"        - main_force_conviction_index_D: {signals_data['main_force_conviction_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - covert_accumulation_signal_D: {signals_data['covert_accumulation_signal_D'].loc[probe_ts]:.4f}")
        #     print(f"        - ADX_14_D: {signals_data['ADX_14_D'].loc[probe_ts]:.4f}")
        #     print(f"        - retail_fomo_premium_index_D: {signals_data['retail_fomo_premium_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - robust_close_slope: {signals_data['robust_close_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - robust_pct_change_slope: {signals_data['robust_pct_change_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - robust_RSI_13_slope: {signals_data['robust_RSI_13_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - robust_MACDh_13_34_8_slope: {signals_data['robust_MACDh_13_34_8_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - robust_volume_slope: {signals_data['robust_volume_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - long_term_close_slope: {signals_data['long_term_close_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - long_term_adx_slope: {signals_data['long_term_adx_slope'].loc[probe_ts]:.4f}")
        #     print(f"        - ACCEL_5_close_D: {signals_data['ACCEL_5_close_D'].loc[probe_ts]:.4f}")
        #     print(f"        - ACCEL_5_RSI_13_D: {signals_data['ACCEL_5_RSI_13_D'].loc[probe_ts]:.4f}")
        #     print(f"        - ACCEL_5_MACDh_13_34_8_D: {signals_data['ACCEL_5_MACDh_13_34_8_D'].loc[probe_ts]:.4f}")
        #     print(f"        - ACCEL_5_volume_D: {signals_data['ACCEL_5_volume_D'].loc[probe_ts]:.4f}")
        #     print(f"        - SLOPE_5_main_force_conviction_index_D: {signals_data['SLOPE_5_main_force_conviction_index_D'].loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM (from states): {state_signals_data['SCORE_BEHAVIOR_PRICE_UPWARD_MOMENTUM'].loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_UPWARD_EFFICIENCY (from states): {state_signals_data['SCORE_BEHAVIOR_UPWARD_EFFICIENCY'].loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_VOLUME_ATROPHY (from states): {state_signals_data['SCORE_BEHAVIOR_VOLUME_ATROPHY'].loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL (from states): {state_signals_data['SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL'].loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_DISTRIBUTION_INTENT (from states): {state_signals_data['SCORE_BEHAVIOR_DISTRIBUTION_INTENT'].loc[probe_ts]:.4f}")
        #     print(f"        - INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW (from states): {state_signals_data['INTERNAL_BEHAVIOR_PRICE_OVEREXTENSION_RAW'].loc[probe_ts]:.4f}")
        #     print(f"        - INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW (from states): {state_signals_data['INTERNAL_BEHAVIOR_STAGNATION_EVIDENCE_RAW'].loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - 上涨纯度分数: {rally_purity_score.loc[probe_ts]:.4f}")
        #     print(f"        - 供应枯竭分数: {supply_exhaustion_score.loc[probe_ts]:.4f}")
        #     print(f"        - 主力意图分数: {main_force_intent_score.loc[probe_ts]:.4f}")
        #     print(f"        - 情境共振分数: {context_resonance_score.loc[probe_ts]:.4f}")
        #     print(f"        - 是否上涨: {is_rising.loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 最终 '锁仓拉升机会信号'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_lockup_rally_score.loc[probe_ts]:.4f}")
        return final_lockup_rally_score

    def _apply_neutral_zone_filter(self, series: pd.Series, threshold: float, is_debug_enabled: bool = False, probe_ts: Optional[pd.Timestamp] = None) -> pd.Series:
        """
        【V1.0 · 新增】应用中性“死区”过滤器。
        - 核心职责: 将信号中绝对值小于阈值的“噪声”强制归零，以符合业务逻辑。
        """
        if threshold > 0:
            filtered_series = series.where(series.abs() > threshold, 0.0)
            # if is_debug_enabled and probe_ts and not series.empty and probe_ts == series.index[-1]:
            #     if series.loc[probe_ts].abs() <= threshold and series.loc[probe_ts] != 0.0:
            #         print(f"      [探针 - _apply_neutral_zone_filter] 信号值 {series.loc[probe_ts]:.4f} 小于阈值 {threshold:.4f}，被过滤为0。")
            return filtered_series
        return series

    def _probe_raw_material_diagnostics(self, df: pd.DataFrame, probe_ts: pd.Timestamp):
        """
        【V1.0 · 新增】原料数据深度探针。
        - 核心职责: 打印出导致关键信号归零的最底层、最原始的输入数据，
                      用于终极的根源验证。
        """
        probe_date_str = probe_ts.strftime('%Y-%m-%d')
        print(f"      [原料探针] 关键输入数据 @ {probe_date_str}")
        # --- 追溯“意图分”和“驱动分”的根源 ---
        raw_mf_flow = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        raw_amount = self._get_safe_series(df, 'amount_D', 1.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        flow_ratio = (raw_mf_flow / raw_amount) if raw_amount > 0 else 0
        print(f"        - 主力资金流 (根源): raw_mf_flow={raw_mf_flow:.2f}, amount={raw_amount:.2f} -> flow_ratio={flow_ratio:.6f}")
        # --- 追溯“恐慌承接度”的根源 ---
        raw_panic = self._get_safe_series(df, 'panic_selling_cascade_D', 0.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        raw_capitulation = self._get_safe_series(df, 'capitulation_absorption_index_D', 0.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        print(f"        - 恐慌承接度 (根源): panic_cascade={raw_panic:.2f}, capitulation_absorption={raw_capitulation:.2f}")
        # --- 追溯“滞涨证据”中“宏观风险”的根源 ---
        raw_winner_rate = self._get_safe_series(df, 'total_winner_rate_D', 50.0, method_name="_probe_raw_material_diagnostics").loc[probe_ts]
        raw_trend_vitality = self._get_safe_series(df, 'trend_vitality_index_D', 0.0, method_name="_probe_raw_material_diagnostics")
        vitality_diff = raw_trend_vitality.diff(3).clip(upper=0).abs().loc[probe_ts]
        print(f"        - 宏观风险 (根源): winner_rate={raw_winner_rate:.2f}, trend_vitality_decay={vitality_diff:.2f}")

    def _diagnose_selling_exhaustion_opportunity(self, df: pd.DataFrame, states: Dict[str, pd.Series], default_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.0 · 凤凰涅槃协议 - 多时间维度动态增强版】诊断卖盘衰竭机会信号。
        - 核心升级: 将卖盘衰竭视为一个多阶段、多维度、非线性演化的过程。
                    解构为四大核心维度：下降与减速、净化与枯竭、吸收与意图、情境就绪。
                    通过加权几何平均融合四大维度，并引入动态情境调制器调整融合权重。
                    **新增多时间维度斜率与加速度分析，深度捕捉趋势动态和反转信号。**
        - 目标: 识别由多重行为证据确认的、具备高可靠性的卖盘衰竭反转机会。
        - 【行为层纯化】该信号仅针对行为类原始数据进行分析，不引用其他情报层的信号。
        - 【探针增强】输出所有原始数据、关键计算节点和最终结果，以便调试和问题暴露。
        """
        method_name = "_diagnose_selling_exhaustion_opportunity"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '卖盘衰竭机会信号' @ {probe_ts.strftime('%Y-%m-%d')}")
        p_behavioral_div_conf = self.config_params
        selling_exhaustion_params = get_param_value(p_behavioral_div_conf.get('selling_exhaustion_params'), {})
        if not selling_exhaustion_params.get('enabled', False):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 信号未启用，返回0分。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        fusion_weights = get_param_value(selling_exhaustion_params.get('fusion_weights'), {
            "descent_deceleration": 0.25, "purification_exhaustion": 0.25, "absorption_intent": 0.3, "contextual_readiness": 0.2
        })
        descent_deceleration_weights = get_param_value(selling_exhaustion_params.get('descent_deceleration_weights'), {
            "price_deceleration": 0.4, "reversal_momentum_accel": 0.3, "rsi_slope_divergence": 0.3
        })
        purification_exhaustion_weights = get_param_value(selling_exhaustion_params.get('purification_exhaustion_weights'), {
            "volume_atrophy": 0.3, "volatility_contraction": 0.2, "low_turnover": 0.15, "sell_exhaustion": 0.15, "low_active_selling": 0.1, "loser_pain": 0.1, "exhaustion_trend_cohesion": 0.1
        })
        absorption_intent_weights = get_param_value(selling_exhaustion_params.get('absorption_intent_weights'), {
            "capitulation_absorption": 0.3, "downward_resistance": 0.2, "offensive_absorption_intent": 0.2, "lower_shadow_absorption": 0.15, "covert_accumulation": 0.1, "main_force_conviction_positive": 0.05, "absorption_intent_accel": 0.05
        })
        contextual_readiness_weights = get_param_value(selling_exhaustion_params.get('contextual_readiness_weights'), {
            "bearish_divergence_inverse": 0.3, "bullish_divergence": 0.2, "behavioral_sentiment_context": 0.3, "behavioral_weak_hand_exhaustion": 0.2, "mtf_trend_alignment": 0.1
        })
        dynamic_modulator_params = get_param_value(selling_exhaustion_params.get('dynamic_modulator_params'), {})
        strong_uptrend_gate_params = get_param_value(selling_exhaustion_params.get('strong_uptrend_gate_params'), {})
        final_exponent = get_param_value(selling_exhaustion_params.get('final_exponent'), 1.5)
        mtf_periods = get_param_value(selling_exhaustion_params.get('mtf_periods'), [5, 13, 21, 34, 55])
        mtf_slope_weights = get_param_value(selling_exhaustion_params.get('mtf_slope_weights'), {'5': 0.4, '13': 0.3, '21': 0.2, '34': 0.05, '55': 0.05})
        mtf_accel_weights = get_param_value(selling_exhaustion_params.get('mtf_accel_weights'), {'5': 0.5, '13': 0.3, '21': 0.15, '34': 0.05})
        # --- 1. 获取所有原始数据和已计算的原子信号 ---
        required_df_signals = [
            'pct_change_D', 'ACCEL_5_pct_change_D', 'SLOPE_5_RSI_13_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'BBW_21_2.0_D', 'turnover_rate_f_D',
            'sell_quote_exhaustion_rate_D', 'active_selling_pressure_D', 'loser_pain_index_D',
            'capitulation_absorption_index_D', 'covert_accumulation_signal_D', 'main_force_conviction_index_D',
            'SLOPE_55_close_D', 'market_sentiment_score_D',
            'retail_fomo_premium_index_D', 'panic_selling_cascade_D', 'chip_fatigue_index_D',
            'SLOPE_55_ADX_14_D'
        ]
        # 收集所有需要进行 _get_mtf_fused_indicator_score 计算的 base_indicator_name
        indicators_for_mtf_fused_score = [
            'pct_change', 'panic_selling_cascade', 'active_selling_pressure', 'retail_panic_surrender_index',
            'main_force_net_flow_calibrated', 'sell_sweep_intensity', 'loser_pain_index',
            'active_buying_support', 'vwap_control_strength', 'buy_quote_exhaustion_rate',
            'support_validation_strength', 'chip_fatigue_index', 'sell_quote_exhaustion_rate',
            'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry',
            'ask_side_liquidity', 'bid_side_liquidity', 'market_impact_cost',
            'BID_LIQUIDITY_SAMPLE_ENTROPY_13d', 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d',
            'price_volume_entropy', 'volatility_expansion_ratio',
            'capitulation_absorption_index', 'covert_accumulation_signal', 'main_force_conviction_index',
            'volume', 'turnover_rate_f' # For exhaustion_trend_cohesion
        ]
        # 动态添加MTF斜率和加速度信号到 required_df_signals
        for period_str in mtf_slope_weights.keys(): # Use mtf_slope_weights for periods
            period = int(period_str)
            for indicator in indicators_for_mtf_fused_score:
                required_df_signals.append(f'SLOPE_{period}_{indicator}_D')
                if period <= 34: # Accel only for shorter periods
                    required_df_signals.append(f'ACCEL_{period}_{indicator}_D')
        required_state_signals = [
            'SCORE_BEHAVIOR_VOLUME_ATROPHY', 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE',
            'SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT', 'SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION',
            'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY', 'SCORE_BEHAVIOR_BULLISH_DIVERGENCE'
        ]
        missing_df_signals = [s for s in required_df_signals if s not in df.columns]
        missing_state_signals = [s for s in required_state_signals if s not in states]
        if missing_df_signals or missing_state_signals:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号。")
                if missing_df_signals: print(f"         - DataFrame中缺失: {missing_df_signals}")
                if missing_state_signals: print(f"         - States中缺失: {missing_state_signals}")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_df_signals}
        state_signals_data = {sig: states[sig] for sig in required_state_signals}
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # 获取信号
        pct_change = signals_data['pct_change_D']
        is_falling = (pct_change < 0).astype(float) # 简化为0，因为min_price_drop_pct_threshold在_diagnose_liquidity_drain_risk中计算
        # --- 预先计算组合 Series，确保 id() 一致性 ---
        accel_5_pct_change_clip_abs = signals_data['ACCEL_5_pct_change_D'].clip(upper=0).abs()
        slope_5_rsi_13_clip_lower_0 = signals_data['SLOPE_5_RSI_13_D'].clip(lower=0)
        volatility_instability_inverse = (1 - signals_data['VOLATILITY_INSTABILITY_INDEX_21d_D'])
        bbw_21_2_0_inverse = (1 - signals_data['BBW_21_2.0_D'])
        turnover_rate_f_inverse = (1 - signals_data['turnover_rate_f_D'])
        main_force_conviction_index_clip_lower_0 = signals_data['main_force_conviction_index_D'].clip(lower=0)
        retail_fomo_premium_index_inverse = (1 - signals_data['retail_fomo_premium_index_D'])
        panic_selling_cascade_raw = signals_data['panic_selling_cascade_D']
        chip_fatigue_index_raw = signals_data['chip_fatigue_index_D']
        loser_pain_index_raw = signals_data['loser_pain_index_D']
        slope_55_close_clip_lower_0 = signals_data['SLOPE_55_close_D'].clip(lower=0)
        slope_55_adx_14_clip_lower_0 = signals_data['SLOPE_55_ADX_14_D'].clip(lower=0)
        market_sentiment_score_clip_upper_0_abs = signals_data['market_sentiment_score_D'].clip(upper=0).abs()
        # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
        series_for_mtf_norm_config = {
            'ACCEL_5_pct_change_D_clip_abs': (accel_5_pct_change_clip_abs, default_weights, True), # price_deceleration_score
            'SLOPE_5_RSI_13_D_clip_lower_0': (slope_5_rsi_13_clip_lower_0, default_weights, True), # rsi_slope_divergence
            'VOLATILITY_INSTABILITY_INDEX_21d_D_inverse': (volatility_instability_inverse, default_weights, True), # volatility_contraction
            'BBW_21_2.0_D_inverse': (bbw_21_2_0_inverse, default_weights, True), # volatility_contraction
            'turnover_rate_f_D_inverse': (turnover_rate_f_inverse, default_weights, True), # low_turnover
            'sell_quote_exhaustion_rate_D': (signals_data['sell_quote_exhaustion_rate_D'], default_weights, True), # sell_exhaustion
            'active_selling_pressure_D': (signals_data['active_selling_pressure_D'], default_weights, True), # low_active_selling
            'loser_pain_index_D': (loser_pain_index_raw, default_weights, True), # norm_loser_pain
            'capitulation_absorption_index_D': (signals_data['capitulation_absorption_index_D'], default_weights, True), # capitulation_confirm_score
            'covert_accumulation_signal_D': (signals_data['covert_accumulation_signal_D'], default_weights, True), # norm_covert_accumulation
            'main_force_conviction_index_D_clip_lower_0': (main_force_conviction_index_clip_lower_0, default_weights, True), # norm_main_force_conviction_positive
            'retail_fomo_premium_index_D_inverse': (retail_fomo_premium_index_inverse, default_weights, True), # norm_retail_fomo_inverse
            'panic_selling_cascade_D': (panic_selling_cascade_raw, default_weights, True), # norm_panic_cascade
            'chip_fatigue_index_D': (chip_fatigue_index_raw, default_weights, True), # norm_chip_fatigue_proxy
            'SLOPE_55_close_D_clip_lower_0': (slope_55_close_clip_lower_0, default_weights, True), # norm_long_term_close_slope_positive
            'SLOPE_55_ADX_14_D_clip_lower_0': (slope_55_adx_14_clip_lower_0, default_weights, True), # norm_long_term_adx_slope_positive
            'market_sentiment_score_D_clip_upper_0_abs': (market_sentiment_score_clip_upper_0_abs, default_weights, True) # For dynamic_modulator_params
        }
        # 批量计算所有多时间框架归一化分数
        normalized_mtf_scores = {}
        for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
            normalized_mtf_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
        current_fusion_power_p = get_param_value(selling_exhaustion_params.get('fusion_power_p'), 0.0)
        if get_param_value(dynamic_modulator_params.get('enabled'), False):
            modulator_signal_1 = signals_data[dynamic_modulator_params.get('modulator_signal_1')]
            modulator_signal_2 = signals_data[dynamic_modulator_params.get('modulator_signal_2')]
            sensitivity_volatility = dynamic_modulator_params.get('sensitivity_volatility', 0.4)
            sensitivity_sentiment = dynamic_modulator_params.get('sensitivity_sentiment', 0.3)
            base_modulator_factor = dynamic_modulator_params.get('base_modulator_factor', 1.0)
            min_modulator = dynamic_modulator_params.get('min_modulator', 0.5)
            max_modulator = dynamic_modulator_params.get('max_modulator', 1.5)
            # --- 预先计算组合 Series，确保 id() 一致性 ---
            modulator_signal_2_clip_upper_0_abs = modulator_signal_2.clip(upper=0).abs()
            # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
            series_for_modulator_norm_config = {
                'modulator_signal_1': (modulator_signal_1, default_weights, True),
                'modulator_signal_2_clip_upper_0_abs': (modulator_signal_2_clip_upper_0_abs, default_weights, True)
            }
            normalized_modulator_scores = {}
            for key, (series_obj, tf_w, asc) in series_for_modulator_norm_config.items():
                normalized_modulator_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
            norm_volatility_inverse = (1 - normalized_modulator_scores['modulator_signal_1']).clip(0, 1)
            norm_sentiment_negative = normalized_modulator_scores['modulator_signal_2_clip_upper_0_abs']
            dynamic_modulator_factor = base_modulator_factor + \
                                       norm_volatility_inverse * sensitivity_volatility + \
                                       norm_sentiment_negative * sensitivity_sentiment
            dynamic_modulator_factor = dynamic_modulator_factor.clip(min_modulator, max_modulator)
        # --- 2. 计算四大维度分数 ---
        # Descent Deceleration
        price_deceleration_score = normalized_mtf_scores['ACCEL_5_pct_change_D_clip_abs']
        rsi_slope_divergence = normalized_mtf_scores['SLOPE_5_RSI_13_D_clip_lower_0']
        # reversal_momentum_accel 的计算需要 _calculate_mtf_dynamic_score，这部分保持不变
        rsi_accel_score = self._calculate_mtf_dynamic_score(df, 'RSI_13', mtf_periods, mtf_slope_weights, 'SLOPE', True, method_name, is_debug_enabled, probe_ts) * \
                          self._calculate_mtf_dynamic_score(df, 'RSI_13', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', True, method_name, is_debug_enabled, probe_ts)
        macd_accel_score = self._calculate_mtf_dynamic_score(df, 'MACDh_13_34_8', mtf_periods, mtf_slope_weights, 'SLOPE', True, method_name, is_debug_enabled, probe_ts) * \
                           self._calculate_mtf_dynamic_score(df, 'MACDh_13_34_8', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', True, method_name, is_debug_enabled, probe_ts)
        reversal_momentum_accel = (rsi_accel_score + macd_accel_score) / 2
        descent_deceleration_score = (
            (price_deceleration_score + 1e-9).pow(descent_deceleration_weights.get('price_deceleration', 0.4)) *
            (reversal_momentum_accel + 1e-9).pow(descent_deceleration_weights.get('reversal_momentum_accel', 0.3)) *
            (rsi_slope_divergence + 1e-9).pow(descent_deceleration_weights.get('rsi_slope_divergence', 0.3))
        ).pow(1 / sum(descent_deceleration_weights.values())).fillna(0.0).clip(0, 1)
        # Purification Exhaustion
        volatility_contraction = normalized_mtf_scores['VOLATILITY_INSTABILITY_INDEX_21d_D_inverse'] * \
                                 normalized_mtf_scores['BBW_21_2.0_D_inverse']
        low_turnover = normalized_mtf_scores['turnover_rate_f_D_inverse']
        sell_exhaustion = normalized_mtf_scores['sell_quote_exhaustion_rate_D']
        low_active_selling = normalized_mtf_scores['active_selling_pressure_D']
        norm_loser_pain = normalized_mtf_scores['loser_pain_index_D']
        # exhaustion_trend_cohesion 的计算需要 _calculate_mtf_dynamic_score，这部分保持不变
        volume_exhaustion_trend = self._calculate_mtf_dynamic_score(df, 'volume', mtf_periods, mtf_slope_weights, 'SLOPE', False, method_name, is_debug_enabled, probe_ts) * \
                                  self._calculate_mtf_dynamic_score(df, 'volume', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', False, method_name, is_debug_enabled, probe_ts)
        turnover_exhaustion_trend = self._calculate_mtf_dynamic_score(df, 'turnover_rate_f', mtf_periods, mtf_slope_weights, 'SLOPE', False, method_name, is_debug_enabled, probe_ts) * \
                                    self._calculate_mtf_dynamic_score(df, 'turnover_rate_f', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', False, method_name, is_debug_enabled, probe_ts)
        sell_pressure_exhaustion_trend = self._calculate_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate', mtf_periods, mtf_slope_weights, 'SLOPE', False, method_name, is_debug_enabled, probe_ts) * \
                                         self._calculate_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', False, method_name, is_debug_enabled, probe_ts)
        exhaustion_trend_cohesion = (volume_exhaustion_trend + turnover_exhaustion_trend + sell_pressure_exhaustion_trend) / 3
        purification_exhaustion_score = (
            (state_signals_data['SCORE_BEHAVIOR_VOLUME_ATROPHY'] + 1e-9).pow(purification_exhaustion_weights.get('volume_atrophy', 0.3)) *
            (volatility_contraction + 1e-9).pow(purification_exhaustion_weights.get('volatility_contraction', 0.2)) *
            (low_turnover + 1e-9).pow(purification_exhaustion_weights.get('low_turnover', 0.15)) *
            (sell_exhaustion + 1e-9).pow(purification_exhaustion_weights.get('sell_exhaustion', 0.15)) *
            (low_active_selling + 1e-9).pow(purification_exhaustion_weights.get('low_active_selling', 0.1)) *
            (norm_loser_pain + 1e-9).pow(purification_exhaustion_weights.get('loser_pain', 0.1)) *
            (exhaustion_trend_cohesion + 1e-9).pow(purification_exhaustion_weights.get('exhaustion_trend_cohesion', 0.1))
        ).pow(1 / sum(purification_exhaustion_weights.values())).fillna(0.0).clip(0, 1)
        # Absorption Intent
        capitulation_confirm_score = normalized_mtf_scores['capitulation_absorption_index_D']
        norm_covert_accumulation = normalized_mtf_scores['covert_accumulation_signal_D']
        norm_main_force_conviction_positive = normalized_mtf_scores['main_force_conviction_index_D_clip_lower_0']
        # absorption_intent_accel 的计算需要 _calculate_mtf_dynamic_score，这部分保持不变
        capitulation_accel_score = self._calculate_mtf_dynamic_score(df, 'capitulation_absorption_index', mtf_periods, mtf_slope_weights, 'SLOPE', True, method_name, is_debug_enabled, probe_ts) * \
                                   self._calculate_mtf_dynamic_score(df, 'capitulation_absorption_index', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', True, method_name, is_debug_enabled, probe_ts)
        covert_accum_accel_score = self._calculate_mtf_dynamic_score(df, 'covert_accumulation_signal', mtf_periods, mtf_slope_weights, 'SLOPE', True, method_name, is_debug_enabled, probe_ts) * \
                                   self._calculate_mtf_dynamic_score(df, 'covert_accumulation_signal', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', True, method_name, is_debug_enabled, probe_ts)
        mf_conviction_accel_score = self._calculate_mtf_dynamic_score(df, 'main_force_conviction_index', mtf_periods, mtf_slope_weights, 'SLOPE', True, method_name, is_debug_enabled, probe_ts) * \
                                    self._calculate_mtf_dynamic_score(df, 'main_force_conviction_index', [p for p in mtf_periods if p <= 34], mtf_accel_weights, 'ACCEL', True, method_name, is_debug_enabled, probe_ts)
        absorption_intent_accel = (capitulation_accel_score + covert_accum_accel_score + mf_conviction_accel_score) / 3
        absorption_intent_score = (
            (capitulation_confirm_score + 1e-9).pow(absorption_intent_weights.get('capitulation_absorption', 0.3)) *
            (state_signals_data['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE'] + 1e-9).pow(absorption_intent_weights.get('downward_resistance', 0.2)) *
            (state_signals_data['SCORE_BEHAVIOR_OFFENSIVE_ABSORPTION_INTENT'] + 1e-9).pow(absorption_intent_weights.get('offensive_absorption_intent', 0.2)) *
            (state_signals_data['SCORE_BEHAVIOR_LOWER_SHADOW_ABSORPTION'] + 1e-9).pow(absorption_intent_weights.get('lower_shadow_absorption', 0.15)) *
            (norm_covert_accumulation + 1e-9).pow(absorption_intent_weights.get('covert_accumulation', 0.1)) *
            (norm_main_force_conviction_positive + 1e-9).pow(absorption_intent_weights.get('main_force_conviction_positive', 0.05)) *
            (absorption_intent_accel + 1e-9).pow(absorption_intent_weights.get('absorption_intent_accel', 0.05))
        ).pow(1 / sum(absorption_intent_weights.values())).fillna(0.0).clip(0, 1)
        # Contextual Readiness
        bearish_divergence_inverse = (1 - state_signals_data['SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY']).clip(0, 1)
        bullish_divergence = state_signals_data['SCORE_BEHAVIOR_BULLISH_DIVERGENCE']
        norm_retail_fomo_inverse = normalized_mtf_scores['retail_fomo_premium_index_D_inverse']
        norm_panic_cascade = normalized_mtf_scores['panic_selling_cascade_D']
        norm_price_deceleration_context = normalized_mtf_scores['ACCEL_5_pct_change_D_clip_abs']
        behavioral_sentiment_context = (
            (norm_retail_fomo_inverse + 1e-9).pow(0.3) *
            (norm_panic_cascade + 1e-9).pow(0.4) *
            (norm_price_deceleration_context + 1e-9).pow(0.3)
        ).pow(1/1.0).fillna(0.0).clip(0,1)
        norm_loser_pain_proxy = normalized_mtf_scores['loser_pain_index_D']
        norm_chip_fatigue_proxy = normalized_mtf_scores['chip_fatigue_index_D']
        norm_low_turnover_proxy = normalized_mtf_scores['turnover_rate_f_D_inverse']
        behavioral_weak_hand_exhaustion = (
            (norm_loser_pain_proxy + 1e-9).pow(0.4) *
            (norm_chip_fatigue_proxy + 1e-9).pow(0.3) *
            (norm_low_turnover_proxy + 1e-9).pow(0.3)
        ).pow(1/1.0).fillna(0.0).clip(0,1)
        norm_long_term_close_slope_positive = normalized_mtf_scores['SLOPE_55_close_D_clip_lower_0']
        norm_long_term_adx_slope_positive = normalized_mtf_scores['SLOPE_55_ADX_14_D_clip_lower_0']
        mtf_trend_alignment = (norm_long_term_close_slope_positive * norm_long_term_adx_slope_positive).pow(0.5)
        contextual_readiness_score = (
            (bearish_divergence_inverse + 1e-9).pow(contextual_readiness_weights.get('bearish_divergence_inverse', 0.3)) *
            (bullish_divergence + 1e-9).pow(contextual_readiness_weights.get('bullish_divergence', 0.2)) *
            (behavioral_sentiment_context + 1e-9).pow(contextual_readiness_weights.get('behavioral_sentiment_context', 0.3)) *
            (behavioral_weak_hand_exhaustion + 1e-9).pow(contextual_readiness_weights.get('behavioral_weak_hand_exhaustion', 0.2)) *
            (mtf_trend_alignment + 1e-9).pow(contextual_readiness_weights.get('mtf_trend_alignment', 0.1))
        ).pow(1 / sum(contextual_readiness_weights.values())).fillna(0.0).clip(0, 1)
        # --- 3. 核心融合 (加权几何平均) ---
        selling_exhaustion_base_score = self._robust_generalized_mean(
            {
                "descent_deceleration": descent_deceleration_score,
                "purification_exhaustion": purification_exhaustion_score,
                "absorption_intent": absorption_intent_score,
                "contextual_readiness": contextual_readiness_score
            },
            fusion_weights,
            df.index,
            power_p=current_fusion_power_p,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name=f"{method_name}_selling_exhaustion_base_score"
        ).fillna(0.0).clip(0, 1)
        # --- 4. 动态情境调制 ---
        dynamic_modulator_factor = pd.Series(1.0, index=df.index, dtype=np.float32)
        if dynamic_modulator_params.get('enabled', False):
            modulator_signal_1 = signals_data[dynamic_modulator_params.get('modulator_signal_1')]
            modulator_signal_2 = signals_data[dynamic_modulator_params.get('modulator_signal_2')]
            sensitivity_volatility = dynamic_modulator_params.get('sensitivity_volatility', 0.4)
            sensitivity_sentiment = dynamic_modulator_params.get('sensitivity_sentiment', 0.3)
            base_modulator_factor = dynamic_modulator_params.get('base_modulator_factor', 1.0)
            min_modulator = dynamic_modulator_params.get('min_modulator', 0.5)
            max_modulator = dynamic_modulator_params.get('max_modulator', 1.5)
            # --- 预先计算组合 Series，确保 id() 一致性 ---
            modulator_signal_2_clip_upper_0_abs = modulator_signal_2.clip(upper=0).abs()
            # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
            series_for_modulator_norm_config = {
                'modulator_signal_1': (modulator_signal_1, default_weights, True),
                'modulator_signal_2_clip_upper_0_abs': (modulator_signal_2_clip_upper_0_abs, default_weights, True)
            }
            normalized_modulator_scores = {}
            for key, (series_obj, tf_w, asc) in series_for_modulator_norm_config.items():
                normalized_modulator_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
            norm_volatility_inverse = (1 - normalized_modulator_scores['modulator_signal_1']).clip(0, 1)
            norm_sentiment_negative = normalized_modulator_scores['modulator_signal_2_clip_upper_0_abs']
            dynamic_modulator_factor = base_modulator_factor + \
                                       norm_volatility_inverse * sensitivity_volatility + \
                                       norm_sentiment_negative * sensitivity_sentiment
            dynamic_modulator_factor = dynamic_modulator_factor.clip(min_modulator, max_modulator)
        final_score_modulated = (selling_exhaustion_base_score * dynamic_modulator_factor).clip(0, 1)
        # --- 5. 门控条件: 仅在价格下跌时激活信号 ---
        final_score_gated = final_score_modulated * is_falling
        if strong_uptrend_gate_params.get('enabled', False):
            long_term_slope_signal = signals_data[strong_uptrend_gate_params.get('long_term_slope_signal')]
            slope_threshold = strong_uptrend_gate_params.get('slope_threshold', 0.005)
            gate_penalty_factor = strong_uptrend_gate_params.get('gate_penalty_factor', 0.5)
            is_strong_uptrend = (long_term_slope_signal > slope_threshold).astype(float)
            final_score_gated = final_score_gated * (1 - is_strong_uptrend * gate_penalty_factor)
        # --- 6. 最终非线性变换 ---
        final_selling_exhaustion_score = final_score_gated.pow(final_exponent).astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     for sig_name in required_df_signals:
        #         print(f"        - {sig_name}: {signals_data[sig_name].loc[probe_ts]:.4f}")
        #     for sig_name in required_state_signals:
        #         print(f"        - {sig_name} (from states): {state_signals_data[sig_name].loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - 下降与减速分数: {descent_deceleration_score.loc[probe_ts]:.4f}")
        #     print(f"        - 净化与枯竭分数: {purification_exhaustion_score.loc[probe_ts]:.4f}")
        #     print(f"        - 吸收与意图分数: {absorption_intent_score.loc[probe_ts]:.4f}")
        #     print(f"        - 情境就绪分数: {contextual_readiness_score.loc[probe_ts]:.4f}")
        #     print(f"        - 基础卖盘衰竭分数: {selling_exhaustion_base_score.loc[probe_ts]:.4f}")
        #     print(f"        - 动态调制因子: {dynamic_modulator_factor.loc[probe_ts]:.4f}")
        #     print(f"        - 调制后分数: {final_score_modulated.loc[probe_ts]:.4f}")
        #     print(f"        - 是否下跌: {is_falling.loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 最终 '卖盘衰竭机会信号'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_selling_exhaustion_score.loc[probe_ts]:.4f}")
        return final_selling_exhaustion_score

    def _diagnose_liquidity_drain_risk(self, df: pd.DataFrame, states: Dict[str, pd.Series], tf_weights: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.1 · 混沌深渊协议 - 动态权重版】诊断流动性枯竭风险。
        - 核心升级: 引入可配置的MTF指标内部组件权重（原始值、斜率、加速度），实现更精细的风险因子调控。
        - 核心重构: 在“恐慌瀑布协议”基础上深度进化，旨在更早期、更精准地识别由恐慌抛售、买盘抵抗瓦解、
                    市场流动性枯竭以及市场结构混沌脆弱共同驱动的系统性风险。
        - 【新增】在调试模式下，打印原始输入、中间计算结果和最终分数。
        """
        method_name = "_diagnose_liquidity_drain_risk"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '流动性枯竭风险' @ {probe_ts.strftime('%Y-%m-%d')}")
        p_behavioral_div_conf = self.config_params
        liquidity_drain_params = get_param_value(p_behavioral_div_conf.get('liquidity_drain_params'), {})
        if not liquidity_drain_params.get('enabled', False):
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 信号未启用，返回0分。")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        fusion_weights = get_param_value(liquidity_drain_params.get('fusion_weights'), {
            "panic_selling_intensity": 0.3, "resistance_collapse": 0.3, "liquidity_exhaustion_evidence": 0.25, "chaos_fragility": 0.15
        })
        panic_selling_intensity_weights = get_param_value(liquidity_drain_params.get('panic_selling_intensity_weights'), {
            "panic_cascade": 0.2, "active_selling_pressure": 0.2, "retail_panic_surrender": 0.15, "price_drop_magnitude": 0.15,
            "main_force_net_flow_negative": 0.15, "sell_sweep_intensity": 0.1, "loser_pain_index": 0.05
        })
        resistance_collapse_weights = get_param_value(liquidity_drain_params.get('resistance_collapse_weights'), {
            "downward_resistance_inverse": 0.25, "active_buying_support_inverse": 0.2, "vwap_control_negative": 0.15,
            "buy_quote_exhaustion_inverse": 0.15, "support_validation_inverse": 0.15, "chip_fatigue_index": 0.1
        })
        liquidity_exhaustion_evidence_weights = get_param_value(liquidity_drain_params.get('liquidity_exhaustion_evidence_weights'), {
            "sell_quote_exhaustion": 0.2, "order_book_imbalance_negative": 0.2, "volume_structure_skew_negative": 0.15,
            "micro_price_impact_asymmetry_negative": 0.15, "ask_side_liquidity_inverse": 0.1, "bid_side_liquidity_inverse": 0.1,
            "market_impact_cost": 0.1
        })
        chaos_fragility_weights = get_param_value(liquidity_drain_params.get('chaos_fragility_weights'), {
            "bid_liquidity_sample_entropy_inverse": 0.3, "bid_liquidity_fractal_dimension_inverse": 0.3,
            "price_volume_entropy": 0.2, "volatility_expansion_ratio": 0.2
        })
        mtf_slope_accel_weights = get_param_value(liquidity_drain_params.get('mtf_slope_accel_weights'), {
            "5": 0.4, "13": 0.3, "21": 0.2, "34": 0.05, "55": 0.05
        })
        mtf_indicator_component_weights = get_param_value(liquidity_drain_params.get('mtf_indicator_component_weights'), {})
        final_exponent = get_param_value(liquidity_drain_params.get('final_exponent'), 1.8)
        dynamic_threshold_params = get_param_value(liquidity_drain_params.get('dynamic_threshold_params'), {})
        dynamic_fusion_power_p_params = get_param_value(liquidity_drain_params.get('dynamic_fusion_power_p_params'), {})
        # --- 1. 计算动态价格跌幅阈值 ---
        min_price_drop_pct_threshold = self._calculate_dynamic_threshold(df, dynamic_threshold_params, is_debug_enabled, probe_ts)
        # --- 2. 获取所有原始数据和已计算的原子信号 ---
        required_df_signals = [
            'pct_change_D', 'panic_selling_cascade_D', 'active_selling_pressure_D',
            'retail_panic_surrender_index_D', 'main_force_net_flow_calibrated_D', 'sell_sweep_intensity_D',
            'loser_pain_index_D', 'active_buying_support_D', 'vwap_control_strength_D',
            'buy_quote_exhaustion_rate_D', 'support_validation_strength_D', 'chip_fatigue_index_D',
            'sell_quote_exhaustion_rate_D', 'order_book_imbalance_D', 'volume_structure_skew_D',
            'micro_price_impact_asymmetry_D', 'ask_side_liquidity_D', 'bid_side_liquidity_D',
            'liquidity_slope_D', 'market_impact_cost_D', 'order_book_clearing_rate_D',
            'BID_LIQUIDITY_SAMPLE_ENTROPY_13d_D', 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d_D',
            'price_volume_entropy_D', 'volatility_expansion_ratio_D'
        ]
        required_state_signals = ['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']
        # 收集所有需要进行 _get_mtf_fused_indicator_score 计算的 base_indicator_name
        indicators_for_mtf_fused_score = [
            'pct_change', 'panic_selling_cascade', 'active_selling_pressure', 'retail_panic_surrender_index',
            'main_force_net_flow_calibrated', 'sell_sweep_intensity', 'loser_pain_index',
            'active_buying_support', 'vwap_control_strength', 'buy_quote_exhaustion_rate',
            'support_validation_strength', 'chip_fatigue_index', 'sell_quote_exhaustion_rate',
            'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry',
            'ask_side_liquidity', 'bid_side_liquidity', 'market_impact_cost',
            'BID_LIQUIDITY_SAMPLE_ENTROPY_13d', 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d',
            'price_volume_entropy', 'volatility_expansion_ratio'
        ]
        # 动态添加MTF斜率和加速度信号到 required_df_signals
        for period_str in mtf_slope_accel_weights.keys():
            period = int(period_str)
            for indicator in indicators_for_mtf_fused_score:
                required_df_signals.append(f'SLOPE_{period}_{indicator}_D')
                required_df_signals.append(f'ACCEL_{period}_{indicator}_D')
        missing_df_signals = [s for s in required_df_signals if s not in df.columns]
        missing_state_signals = [s for s in required_state_signals if s not in states]
        if missing_df_signals or missing_state_signals:
            if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                print(f"      [探针 - {method_name}] 缺少核心信号。")
                if missing_df_signals: print(f"         - DataFrame中缺失: {missing_df_signals}")
                if missing_state_signals: print(f"         - States中缺失: {missing_state_signals}")
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中提取所有必需的原始信号
        signals_data = {sig: df[sig] for sig in required_df_signals}
        state_signals_data = {sig: states[sig] for sig in required_state_signals}
        # 获取信号
        pct_change = signals_data['pct_change_D']
        is_falling = (pct_change < -min_price_drop_pct_threshold).astype(float)
        debug_info = (is_debug_enabled, probe_ts, method_name)
        current_fusion_power_p = get_param_value(liquidity_drain_params.get('fusion_power_p'), 0.0)
        if get_param_value(dynamic_fusion_power_p_params.get('enabled'), False):
            modulator_signal_1 = self._get_safe_series(df, dynamic_fusion_power_p_params.get('volatility_signal'), 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts)
            modulator_signal_2 = self._get_safe_series(df, dynamic_fusion_power_p_params.get('sentiment_signal'), 0.0, method_name=method_name, is_debug_enabled=is_debug_enabled, probe_ts=probe_ts)
            p_mtf = self.config_params.get('mtf_normalization_params', {})
            default_tf_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1})
            # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
            series_for_mtf_norm_config = {
                'modulator_signal_1': (modulator_signal_1, default_tf_weights, True),
                'modulator_signal_2': (modulator_signal_2, default_tf_weights, True)
            }
            # 批量计算所有多时间框架归一化分数
            normalized_mtf_scores_for_modulator = {}
            for key, (series_obj, tf_w, asc) in series_for_mtf_norm_config.items():
                normalized_mtf_scores_for_modulator[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
            norm_volatility = normalized_mtf_scores_for_modulator['modulator_signal_1']
            norm_sentiment = normalized_mtf_scores_for_modulator['modulator_signal_2']
            volatility_impact = (1 - norm_volatility) * dynamic_fusion_power_p_params.get('volatility_sensitivity', 0.5)
            sentiment_impact = (1 - norm_sentiment) * dynamic_fusion_power_p_params.get('sentiment_sensitivity', 0.5)
            current_fusion_power_p = current_fusion_power_p + volatility_impact + sentiment_impact
            current_fusion_power_p = pd.Series(current_fusion_power_p, index=df.index).clip(dynamic_fusion_power_p_params.get('min_power_p', -0.5), dynamic_fusion_power_p_params.get('max_power_p', 0.5))
        # --- 3. 计算恐慌抛售烈度 (Panic Selling Intensity, PSI) ---
        # 批量计算所有 _get_mtf_fused_indicator_score
        fused_indicator_scores = {}
        for indicator_base_name in indicators_for_mtf_fused_score:
            is_negative = False
            ascending = True
            if indicator_base_name in ['pct_change', 'main_force_net_flow_calibrated', 'order_book_imbalance', 'volume_structure_skew', 'micro_price_impact_asymmetry']:
                is_negative = True
            if indicator_base_name in ['ask_side_liquidity', 'bid_side_liquidity', 'BID_LIQUIDITY_SAMPLE_ENTROPY_13d', 'BID_LIQUIDITY_FRACTAL_DIMENSION_89d']:
                ascending = False # 这些指标的inverse是ascending
            fused_indicator_scores[indicator_base_name] = self._get_mtf_fused_indicator_score(
                df, indicator_base_name, mtf_slope_accel_weights, mtf_indicator_component_weights,
                is_negative_indicator=is_negative, ascending=ascending, debug_info=False
            )
        norm_price_drop_magnitude = fused_indicator_scores['pct_change']
        norm_panic_cascade = fused_indicator_scores['panic_selling_cascade']
        norm_active_selling = fused_indicator_scores['active_selling_pressure']
        norm_retail_panic = fused_indicator_scores['retail_panic_surrender_index']
        norm_main_force_net_flow_negative = fused_indicator_scores['main_force_net_flow_calibrated']
        norm_sell_sweep_intensity = fused_indicator_scores['sell_sweep_intensity']
        norm_loser_pain_index = fused_indicator_scores['loser_pain_index']
        psi_score = (
            (norm_panic_cascade + 1e-9).pow(panic_selling_intensity_weights.get('panic_cascade', 0.2)) *
            (norm_active_selling + 1e-9).pow(panic_selling_intensity_weights.get('active_selling_pressure', 0.2)) *
            (norm_retail_panic + 1e-9).pow(panic_selling_intensity_weights.get('retail_panic_surrender', 0.15)) *
            (norm_price_drop_magnitude + 1e-9).pow(panic_selling_intensity_weights.get('price_drop_magnitude', 0.15)) *
            (norm_main_force_net_flow_negative + 1e-9).pow(panic_selling_intensity_weights.get('main_force_net_flow_negative', 0.15)) *
            (norm_sell_sweep_intensity + 1e-9).pow(panic_selling_intensity_weights.get('sell_sweep_intensity', 0.1)) *
            (norm_loser_pain_index + 1e-9).pow(panic_selling_intensity_weights.get('loser_pain_index', 0.05))
        ).pow(1 / sum(panic_selling_intensity_weights.values())).fillna(0.0).clip(0, 1)
        # --- 4. 计算抵抗瓦解度 (Resistance Collapse, RC) ---
        norm_downward_resistance_inverse = (1 - state_signals_data['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']).clip(0, 1)
        norm_active_buying_inverse = (1 - fused_indicator_scores['active_buying_support']).clip(0, 1)
        norm_vwap_control_negative = fused_indicator_scores['vwap_control_strength']
        norm_buy_quote_exhaustion_inverse = (1 - fused_indicator_scores['buy_quote_exhaustion_rate']).clip(0, 1)
        norm_support_validation_inverse = (1 - fused_indicator_scores['support_validation_strength']).clip(0, 1)
        norm_chip_fatigue_index = fused_indicator_scores['chip_fatigue_index']
        rc_score = (
            (norm_downward_resistance_inverse + 1e-9).pow(resistance_collapse_weights.get('downward_resistance_inverse', 0.25)) *
            (norm_active_buying_inverse + 1e-9).pow(resistance_collapse_weights.get('active_buying_support_inverse', 0.2)) *
            (norm_vwap_control_negative + 1e-9).pow(resistance_collapse_weights.get('vwap_control_negative', 0.15)) *
            (norm_buy_quote_exhaustion_inverse + 1e-9).pow(resistance_collapse_weights.get('buy_quote_exhaustion_inverse', 0.15)) *
            (norm_support_validation_inverse + 1e-9).pow(resistance_collapse_weights.get('support_validation_inverse', 0.15)) *
            (norm_chip_fatigue_index + 1e-9).pow(resistance_collapse_weights.get('chip_fatigue_index', 0.1))
        ).pow(1 / sum(resistance_collapse_weights.values())).fillna(0.0).clip(0, 1)
        # --- 5. 计算流动性枯竭证据 (Liquidity Exhaustion Evidence, LEE) ---
        norm_sell_quote_exhaustion = fused_indicator_scores['sell_quote_exhaustion_rate']
        norm_order_book_imbalance_negative = fused_indicator_scores['order_book_imbalance']
        norm_volume_structure_skew_negative = fused_indicator_scores['volume_structure_skew']
        norm_micro_price_impact_asymmetry_negative = fused_indicator_scores['micro_price_impact_asymmetry']
        norm_ask_side_liquidity_inverse = (1 - fused_indicator_scores['ask_side_liquidity']).clip(0, 1)
        norm_bid_side_liquidity_inverse = (1 - fused_indicator_scores['bid_side_liquidity']).clip(0, 1)
        norm_market_impact_cost = fused_indicator_scores['market_impact_cost']
        lee_score = (
            (norm_sell_quote_exhaustion + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('sell_quote_exhaustion', 0.2)) *
            (norm_order_book_imbalance_negative + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('order_book_imbalance_negative', 0.2)) *
            (norm_volume_structure_skew_negative + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('volume_structure_skew_negative', 0.15)) *
            (norm_micro_price_impact_asymmetry_negative + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('micro_price_impact_asymmetry_negative', 0.15)) *
            (norm_ask_side_liquidity_inverse + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('ask_side_liquidity_inverse', 0.1)) *
            (norm_bid_side_liquidity_inverse + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('bid_side_liquidity_inverse', 0.1)) *
            (norm_market_impact_cost + 1e-9).pow(liquidity_exhaustion_evidence_weights.get('market_impact_cost', 0.1))
        ).pow(1 / sum(liquidity_exhaustion_evidence_weights.values())).fillna(0.0).clip(0, 1)
        # --- 6. 计算混沌与脆弱性 (Chaos & Fragility, CF) ---
        norm_bid_liquidity_sample_entropy_inverse = (1 - fused_indicator_scores['BID_LIQUIDITY_SAMPLE_ENTROPY_13d']).clip(0, 1)
        norm_bid_liquidity_fractal_dimension_inverse = (1 - fused_indicator_scores['BID_LIQUIDITY_FRACTAL_DIMENSION_89d']).clip(0, 1)
        norm_price_volume_entropy = fused_indicator_scores['price_volume_entropy']
        norm_volatility_expansion_ratio = fused_indicator_scores['volatility_expansion_ratio']
        cf_score = (
            (norm_bid_liquidity_sample_entropy_inverse + 1e-9).pow(chaos_fragility_weights.get('bid_liquidity_sample_entropy_inverse', 0.3)) *
            (norm_bid_liquidity_fractal_dimension_inverse + 1e-9).pow(chaos_fragility_weights.get('bid_liquidity_fractal_dimension_inverse', 0.3)) *
            (norm_price_volume_entropy + 1e-9).pow(chaos_fragility_weights.get('price_volume_entropy', 0.2)) *
            (norm_volatility_expansion_ratio + 1e-9).pow(chaos_fragility_weights.get('volatility_expansion_ratio', 0.2))
        ).pow(1 / sum(chaos_fragility_weights.values())).fillna(0.0).clip(0, 1)
        # --- 7. 最终融合 (加权几何平均) ---
        liquidity_drain_base_score = self._robust_generalized_mean(
            {
                "panic_selling_intensity": psi_score,
                "resistance_collapse": rc_score,
                "liquidity_exhaustion_evidence": lee_score,
                "chaos_fragility": cf_score
            },
            fusion_weights,
            df.index,
            power_p=current_fusion_power_p,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name=f"{method_name}_liquidity_drain_base_score"
        ).fillna(0.0).clip(0, 1)
        # --- 8. 门控条件: 仅在价格下跌时激活信号 ---
        final_score_gated = liquidity_drain_base_score * is_falling
        # --- 9. 最终非线性变换 ---
        final_liquidity_drain_score = final_score_gated.pow(final_exponent).astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     for sig_name in required_df_signals:
        #         print(f"        - {sig_name}: {signals_data[sig_name].loc[probe_ts]:.4f}")
        #     for sig_name in required_state_signals:
        #         print(f"        - {sig_name} (from states): {state_signals_data[sig_name].loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - 动态价格跌幅阈值: {min_price_drop_pct_threshold.loc[probe_ts]:.4f}")
        #     print(f"        - 恐慌抛售烈度分数: {psi_score.loc[probe_ts]:.4f}")
        #     print(f"        - 抵抗瓦解度分数: {rc_score.loc[probe_ts]:.4f}")
        #     print(f"        - 流动性枯竭证据分数: {lee_score.loc[probe_ts]:.4f}")
        #     print(f"        - 混沌与脆弱性分数: {cf_score.loc[probe_ts]:.4f}")
        #     print(f"        - 基础流动性枯竭分数: {liquidity_drain_base_score.loc[probe_ts]:.4f}")
        #     print(f"        - 是否下跌: {is_falling.loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 最终 '流动性枯竭风险'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_liquidity_drain_score.loc[probe_ts]:.4f}")
        return final_liquidity_drain_score

    def _calculate_battlefield_momentum(self, df: pd.DataFrame, day_quality_score: pd.Series, params: Dict, is_debug_enabled: bool, probe_ts: Optional[pd.Timestamp]) -> pd.Series:
        """
        【V3.1 · 多时间框架共振增强版】战场动量信号参数。
        - 核心升级: 1) 自适应指数移动平均(AEMA)：根据日内行为K线质量分自身的波动率动态调整平滑因子，提高信号响应性。
                      2) 动量加速度：引入动量变化率，识别动量增强或减弱的早期迹象。
                      3) 行为情境调制：使用如'上涨效率'等行为类信号作为乘数因子，在有利情境下放大动量，在不利情境下抑制动量。
                      4) 【新增】多时间框架方向一致性：显式量化不同时间框架斜率的方向一致性，增强对趋势共振的捕捉。
        - 目标: 提供一个更具前瞻性和情境感知的短期行为趋势信号。
        - 【探针增强】输出所有原始数据、关键计算节点和最终结果，以便调试和问题暴露。
        """
        method_name = "_calculate_battlefield_momentum"
        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
            print(f"    -> [探针 - {method_name}] 正在计算 '战场动量信号' @ {probe_ts.strftime('%Y-%m-%d')}")
        # 1. 获取参数
        # 从 self.config_params 获取 battlefield_momentum_params
        params = get_param_value(self.config_params.get('battlefield_momentum_params'), {})
        mtf_periods = get_param_value(params.get('mtf_periods'), [5, 13, 21, 34, 55])
        mtf_slope_weights = get_param_value(params.get('mtf_slope_weights'), {'5': 0.4, '13': 0.3, '21': 0.2, '34': 0.05, '55': 0.05})
        mtf_accel_weights = get_param_value(params.get('mtf_accel_weights'), {'5': 0.5, '13': 0.3, '21': 0.15, '34': 0.05})
        momentum_accel_fusion_power_p = get_param_value(params.get('momentum_accel_fusion_power_p'), 0.0) # 0.0 for geometric mean
        contextual_modulator_enabled = get_param_value(params.get('contextual_modulator_enabled'), True)
        context_signals_weights = get_param_value(params.get('context_signals_weights'), {
            'SCORE_BEHAVIOR_UPWARD_EFFICIENCY': 0.4,
            'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE_INVERSE': 0.3,
            'SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL': 0.3
        })
        mtf_directional_consistency_weight = get_param_value(params.get('mtf_directional_consistency_weight'), 0.2)
        final_fusion_power_p = get_param_value(params.get('final_fusion_power_p'), 0.5) # Between geometric and arithmetic mean
        final_exponent = get_param_value(params.get('final_exponent'), 1.5)
        # 从 self.config_params 获取 mtf_normalization_params
        p_mtf = get_param_value(self.config_params.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default'), {'5': 0.4, '13': 0.3, '21': 0.2, '55': 0.1}) # 新增：定义 default_weights
        # 确保 day_quality_score 为 float32 类型，并处理 NaN
        day_quality_score = pd.to_numeric(day_quality_score, errors='coerce').fillna(0).astype(np.float32)
        debug_info = (is_debug_enabled, probe_ts, method_name)
        # 2. 动态计算 day_quality_score 的斜率和加速度
        bq_slopes, bq_accels = self._calculate_series_dynamics(day_quality_score, mtf_periods, df.index, is_debug_enabled, probe_ts, 'BIPOLAR_BEHAVIORAL_DAY_QUALITY')
        # --- 收集所有需要进行双极归一化的 Series 的配置 ---
        series_for_bipolar_norm_config = {}
        for p_str, weight in mtf_slope_weights.items():
            p = int(p_str)
            if p in bq_slopes and weight > 0:
                series_for_bipolar_norm_config[f'bq_slope_{p}'] = (bq_slopes[p], p * 2, 1.0) # (series_obj, window, sensitivity)
        for p_str, weight in mtf_accel_weights.items():
            p = int(p_str)
            if p in bq_accels and weight > 0:
                series_for_bipolar_norm_config[f'bq_accel_{p}'] = (bq_accels[p], p * 2, 1.0) # (series_obj, window, sensitivity)
        # 批量计算所有双极归一化分数
        normalized_bipolar_scores = {}
        for key, (series_obj, window, sensitivity) in series_for_bipolar_norm_config.items():
            normalized_bipolar_scores[key] = normalize_to_bipolar(series_obj, df.index, windows=window, sensitivity=sensitivity, debug_info=False)
        # 3. 多时间维度动量 (MTF Momentum)
        mtf_momentum_scores_raw = {}
        mtf_momentum_scores_weighted = []
        total_slope_weight = sum(mtf_slope_weights.values())
        for p_str, weight in mtf_slope_weights.items():
            p = int(p_str)
            if p in bq_slopes and weight > 0:
                norm_slope = normalized_bipolar_scores[f'bq_slope_{p}']
                mtf_momentum_scores_raw[p] = norm_slope
                mtf_momentum_scores_weighted.append(norm_slope * weight)
        if not mtf_momentum_scores_weighted or total_slope_weight == 0:
            mtf_momentum_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        else:
            mtf_momentum_score = sum(mtf_momentum_scores_weighted) / total_slope_weight
        # 4. 多时间维度加速度 (MTF Acceleration)
        mtf_acceleration_scores_weighted = []
        total_accel_weight = sum(mtf_accel_weights.values())
        for p_str, weight in mtf_accel_weights.items():
            p = int(p_str)
            if p in bq_accels and weight > 0:
                norm_accel = normalized_bipolar_scores[f'bq_accel_{p}']
                mtf_acceleration_scores_weighted.append(norm_accel * weight)
        if not mtf_acceleration_scores_weighted or total_accel_weight == 0:
            mtf_acceleration_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        else:
            mtf_acceleration_score = sum(mtf_acceleration_scores_weighted) / total_accel_weight
        # 5. 多时间框架方向一致性 (MTF Directional Consistency)
        mtf_directional_consistency = pd.Series(0.0, index=df.index, dtype=np.float32)
        if mtf_momentum_scores_raw and mtf_directional_consistency_weight > 0:
            slope_signs_df = pd.DataFrame({p: np.sign(s) for p, s in mtf_momentum_scores_raw.items()})
            mtf_directional_consistency = slope_signs_df.mean(axis=1).fillna(0)
        # 6. 融合MTF动量和加速度为双极性分数
        fusion_components_momentum_accel = {
            "momentum": mtf_momentum_score,
            "acceleration": mtf_acceleration_score,
        }
        fusion_weights_momentum_accel = {
            "momentum": 0.7,
            "acceleration": 0.3,
        }
        directional_momentum_raw = self._robust_generalized_mean(
            fusion_components_momentum_accel,
            fusion_weights_momentum_accel,
            df.index,
            power_p=momentum_accel_fusion_power_p,
            is_debug_enabled=is_debug_enabled,
            probe_ts=probe_ts,
            fusion_level_name=f"{method_name}_MTF动量加速度融合"
        )
        base_directional_momentum = (directional_momentum_raw + 1) / 2
        # 7. 行为情境健康度调制
        behavioral_context_health = pd.Series(1.0, index=df.index, dtype=np.float32)
        if contextual_modulator_enabled:
            context_scores = {}
            valid_context_weights = {}
            # --- 收集所有需要进行多时间框架归一化的 Series 的配置 ---
            series_for_context_norm_config = {}
            for sig_name, weight in context_signals_weights.items():
                if sig_name == 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE_INVERSE':
                    # 确保 SCORE_BEHAVIOR_DOWNWARD_RESISTANCE 存在于 states
                    if 'SCORE_BEHAVIOR_DOWNWARD_RESISTANCE' in self.strategy.atomic_states:
                        series_for_context_norm_config[sig_name] = (
                            (1 - self.strategy.atomic_states['SCORE_BEHAVIOR_DOWNWARD_RESISTANCE']).clip(0, 1),
                            default_weights, True
                        )
                        valid_context_weights[sig_name] = weight
                    else:
                        if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                            print(f"      [探针 - {method_name}] 警告：缺少上下文信号 '{sig_name}'。")
                elif sig_name in self.strategy.atomic_states:
                    series_for_context_norm_config[sig_name] = (
                        self.strategy.atomic_states[sig_name],
                        default_weights, True
                    )
                    valid_context_weights[sig_name] = weight
                else:
                    if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
                        print(f"      [探针 - {method_name}] 警告：缺少上下文信号 '{sig_name}'。")
            # 批量计算所有多时间框架归一化分数
            normalized_context_scores = {}
            for key, (series_obj, tf_w, asc) in series_for_context_norm_config.items():
                normalized_context_scores[key] = get_adaptive_mtf_normalized_score(series_obj, df.index, tf_weights=tf_w, ascending=asc, debug_info=False)
            # 将归一化后的分数赋值给 context_scores
            for key in normalized_context_scores:
                context_scores[key] = normalized_context_scores[key]
            if context_scores:
                behavioral_context_health = self._robust_generalized_mean(
                    context_scores,
                    valid_context_weights,
                    df.index,
                    power_p=0.0, # Geometric mean for health score
                    is_debug_enabled=is_debug_enabled,
                    probe_ts=probe_ts,
                    fusion_level_name=f"{method_name}_行为情境健康度融合"
                )
        # 8. 最终融合
        final_fusion_components = {
            "base_directional_momentum": base_directional_momentum,
            "behavioral_context_health": behavioral_context_health,
            "mtf_directional_consistency": (mtf_directional_consistency + 1) / 2 # Map to [0,1]
        }
        final_fusion_weights = {
            "base_directional_momentum": 1.0 - mtf_directional_consistency_weight,
            "behavioral_context_health": 1.0, # This weight is applied multiplicatively later
            "mtf_directional_consistency": mtf_directional_consistency_weight
        }
        # 调整 mtf_directional_consistency 的权重，使其与 base_directional_momentum 融合
        # 这里的融合逻辑需要重新考虑，因为 mtf_directional_consistency 是一个 [-1, 1] 的值，
        # 而 base_directional_momentum 是 [0, 1] 的值。
        # 简单的加权平均可能不合适。
        # 考虑将 mtf_directional_consistency 作为一个乘数因子，或者在融合前将其映射到 [0, 1]
        # 重新设计融合逻辑：
        # 1. 将 mtf_directional_consistency 映射到 [0, 1]
        norm_mtf_directional_consistency = (mtf_directional_consistency + 1) / 2
        # 2. 融合 base_directional_momentum 和 norm_mtf_directional_consistency
        # 采用加权平均，权重由 final_fusion_weights 决定
        weighted_momentum_consistency = (
            base_directional_momentum * final_fusion_weights.get("base_directional_momentum", 1.0 - mtf_directional_consistency_weight) +
            norm_mtf_directional_consistency * final_fusion_weights.get("mtf_directional_consistency", mtf_directional_consistency_weight)
        )
        # 3. 将融合结果与 behavioral_context_health 相乘
        battlefield_momentum_score = (weighted_momentum_consistency * behavioral_context_health).clip(0, 1)
        final_battlefield_momentum = battlefield_momentum_score.pow(final_exponent).astype(np.float32)
        # if is_debug_enabled and probe_ts and not df.empty and probe_ts == df.index[-1]:
        #     print(f"      [探针 - {method_name}] 原始输入 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - day_quality_score: {day_quality_score.loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_UPWARD_EFFICIENCY (from states): {self.strategy.atomic_states.get('SCORE_BEHAVIOR_UPWARD_EFFICIENCY', pd.Series(0.0, index=df.index)).loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_DOWNWARD_RESISTANCE (from states): {self.strategy.atomic_states.get('SCORE_BEHAVIOR_DOWNWARD_RESISTANCE', pd.Series(0.0, index=df.index)).loc[probe_ts]:.4f}")
        #     print(f"        - SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL (from states): {self.strategy.atomic_states.get('SCORE_BEHAVIOR_INTRADAY_BULL_CONTROL', pd.Series(0.0, index=df.index)).loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 中间计算 @ {probe_ts.strftime('%Y-%m-%d')}:")
        #     print(f"        - MTF动量分数: {mtf_momentum_score.loc[probe_ts]:.4f}")
        #     print(f"        - MTF加速度分数: {mtf_acceleration_score.loc[probe_ts]:.4f}")
        #     print(f"        - MTF方向一致性: {mtf_directional_consistency.loc[probe_ts]:.4f}")
        #     print(f"        - 基础方向动量: {base_directional_momentum.loc[probe_ts]:.4f}")
        #     print(f"        - 行为情境健康度: {behavioral_context_health.loc[probe_ts]:.4f}")
        #     print(f"        - 加权动量一致性: {weighted_momentum_consistency.loc[probe_ts]:.4f}")
        #     print(f"      [探针 - {method_name}] 最终 '战场动量'分数 @ {probe_ts.strftime('%Y-%m-%d')}: {final_battlefield_momentum.loc[probe_ts]:.4f}")
        return final_battlefield_momentum







