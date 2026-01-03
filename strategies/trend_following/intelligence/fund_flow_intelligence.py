# strategies\trend_following\intelligence\fund_flow_intelligence.py
import os
import pandas as pd
import numpy as np
import numba as nb
from typing import Dict, List, Tuple, Any, Union, Optional
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, 
    get_adaptive_mtf_normalized_score, load_external_json_config, _robust_geometric_mean
)

@nb.njit(cache=True)
def _numba_calculate_deception_risk_core(
    norm_wash_trade_values: np.ndarray,
    norm_deception_values: np.ndarray,
    norm_conviction_values: np.ndarray,
    norm_flow_credibility_values: np.ndarray,
    norm_market_sentiment_values: np.ndarray,
    norm_deception_lure_long_values: np.ndarray,
    norm_deception_lure_short_values: np.ndarray, # 虽然在风险计算中不直接使用，但作为参数传入保持一致性
    wash_trade_penalty_sensitivity: float,
    deception_penalty_sensitivity: float,
    deception_lure_long_penalty_sensitivity: float,
    deception_context_sensitivity: float,
    flow_credibility_threshold: float,
    deception_cohesion_mod_values: np.ndarray,
    wash_trade_cohesion_mod_values: np.ndarray
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算资金流诡道博弈风险。
    直接操作NumPy数组，避免Pandas Series的内部开销。
    """
    num_dates = len(norm_wash_trade_values)
    deception_risk_score_values = np.zeros(num_dates, dtype=np.float32)
    for i in range(num_dates):
        # 情境调制因子：市场情绪对风险的放大或抑制
        sentiment_mod_factor = (1 + np.abs(norm_market_sentiment_values[i]) * deception_context_sensitivity * np.sign(norm_market_sentiment_values[i]))
        # 修正：对标量进行裁剪，使用 np.maximum 和 np.minimum
        sentiment_mod_factor = np.maximum(0.5, np.minimum(sentiment_mod_factor, 1.5)) # 裁剪到合理范围
        # 风险来源1：对倒强度
        risk_from_wash_trade = norm_wash_trade_values[i] * wash_trade_penalty_sensitivity * sentiment_mod_factor * wash_trade_cohesion_mod_values[i]
        # 风险来源2：诱多欺骗（正向欺骗指数）
        risk_from_bull_trap = np.maximum(0.0, norm_deception_values[i]) * deception_penalty_sensitivity * sentiment_mod_factor * deception_cohesion_mod_values[i]
        # 风险来源3：诱多强度（deception_lure_long_intensity）
        risk_from_lure_long = norm_deception_lure_long_values[i] * deception_lure_long_penalty_sensitivity
        # 风险来源4：弱信念下的诱空（负向欺骗指数，且主力信念弱）
        # (1 - norm_conviction.clip(lower=0)) 转换为 (1 - 正向信念)
        risk_from_bear_trap_weak_conviction = np.maximum(0.0, -norm_deception_values[i]) * (1 - np.maximum(0.0, norm_conviction_values[i])) * deception_penalty_sensitivity * deception_cohesion_mod_values[i]
        # 风险来源5：低资金流可信度
        # 只有当可信度低于阈值时才产生风险，且风险程度与低于阈值的程度成正比
        risk_from_low_credibility = np.maximum(0.0, (1 - norm_flow_credibility_values[i]) - (1 - flow_credibility_threshold))
        # 修正：对标量进行裁剪，使用 np.maximum 和 np.minimum
        risk_from_low_credibility = np.maximum(0.0, np.minimum(risk_from_low_credibility, 1.0)) # 确保在0到1之间
        # 累加所有风险成分
        total_risk = risk_from_wash_trade + risk_from_bull_trap + risk_from_lure_long + risk_from_bear_trap_weak_conviction + risk_from_low_credibility
        # 最终风险分数裁剪到 [0, 1] 范围
        # 修正：对标量进行裁剪，使用 np.maximum 和 np.minimum
        deception_risk_score_values[i] = np.maximum(0.0, np.minimum(total_risk, 1.0))

    return deception_risk_score_values

@nb.njit(cache=True)
def _numba_mtf_cohesion_divergence_core(
    avg_short_slope_values: np.ndarray,
    avg_short_accel_values: np.ndarray,
    avg_long_slope_values: np.ndarray,
    avg_long_accel_values: np.ndarray,
    epsilon_sign: float,
    persistence_window: int
) -> np.ndarray:
    """
    Numba优化后的核心函数，用于计算多时间框架的共振/背离因子。
    直接操作NumPy数组，避免Pandas Series的内部开销。
    """
    num_dates = len(avg_short_slope_values)
    
    # 1. 方向一致性 (Direction Cohesion)
    direction_cohesion_values = np.clip(avg_short_slope_values * avg_long_slope_values, -1.0, 1.0)

    # 2. 强度一致性 (Magnitude Cohesion)
    # 相对差异：短期和长期强度越接近，一致性越高
    relative_strength_cohesion_slope_values = 1 - np.abs(avg_short_slope_values - avg_long_slope_values) / (np.abs(avg_short_slope_values) + np.abs(avg_long_slope_values) + epsilon_sign)
    relative_strength_cohesion_accel_values = 1 - np.abs(avg_short_accel_values - avg_long_accel_values) / (np.abs(avg_short_accel_values) + np.abs(avg_long_accel_values) + epsilon_sign)
    
    # 共同强度：短期和长期强度都高，则共同强度高
    overall_magnitude_slope_values = (np.abs(avg_short_slope_values) + np.abs(avg_long_slope_values)) / 2
    overall_magnitude_accel_values = (np.abs(avg_short_accel_values) + np.abs(avg_long_accel_values)) / 2
    
    # 融合相对差异和共同强度
    strength_cohesion_slope_values = np.clip((relative_strength_cohesion_slope_values + overall_magnitude_slope_values) / 2, 0.0, 1.0)
    strength_cohesion_accel_values = np.clip((relative_strength_cohesion_accel_values + overall_magnitude_accel_values) / 2, 0.0, 1.0)
    
    # 综合强度一致性
    strength_cohesion_values = (strength_cohesion_slope_values + strength_cohesion_accel_values) / 2

    # 3. 趋势质量/持久性调制器 (Trend Quality/Persistence Modulator)
    short_slope_sign_values = np.where(np.abs(avg_short_slope_values) > epsilon_sign, np.sign(avg_short_slope_values), 0.0)
    long_slope_sign_values = np.where(np.abs(avg_long_slope_values) > epsilon_sign, np.sign(avg_long_slope_values), 0.0)
    
    directional_consistency_values = np.zeros(num_dates, dtype=np.float32)
    for i in range(num_dates):
        if short_slope_sign_values[i] != 0.0 and short_slope_sign_values[i] == long_slope_sign_values[i]:
            directional_consistency_values[i] = 1.0
    
    # Rolling mean for trend_persistence (manual Numba implementation)
    trend_persistence_values = np.zeros(num_dates, dtype=np.float32)
    for i in range(num_dates):
        start_idx = max(0, i - persistence_window + 1)
        window_sum = 0.0
        window_count = 0
        for j in range(start_idx, i + 1):
            window_sum += directional_consistency_values[j]
            window_count += 1
        if window_count > 0:
            trend_persistence_values[i] = window_sum / window_count
    
    trend_quality_modulator_values = np.clip(0.5 + trend_persistence_values, 0.5, 1.5)

    # 4. 最终双极性共振分数：方向 * 强度 * 趋势质量
    mtf_resonance_score_values = direction_cohesion_values * strength_cohesion_values * trend_quality_modulator_values
    
    return np.clip(mtf_resonance_score_values, -1.0, 1.0)

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        external_config = load_external_json_config("config/intelligence/fund_flow.json", {})
        # 直接从加载的配置中获取 fund_flow_ultimate_params 块，而不是通过 get_params_block
        self.p_conf_ff = external_config.get('fund_flow_ultimate_params', {})
        # 获取策略实例的 debug_params
        self.debug_params = get_params_block(self.strategy, 'debug_params', {})
        self.probe_dates = get_param_value(self.debug_params.get('probe_dates'), [])
        self.tf_weights_ff = get_param_value(self.p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})

    def _get_safe_series(self, df: pd.DataFrame, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        【V1.1 · 上下文修复版】安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        df_index = df.index
        series = None
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                series = pd.Series(default_value, index=df_index)
            else:
                series = data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                series = pd.Series(default_value, index=df_index)
            else:
                raw_data = data_source[column_name]
                if isinstance(raw_data, pd.Series):
                    series = raw_data.reindex(df_index, fill_value=default_value)
                else:
                    series = pd.Series(raw_data, index=df_index)
        else:
            print(f"    -> [资金流情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            series = pd.Series(default_value, index=df_index)
        return series

    def _get_mtf_dynamic_score(self, df: pd.DataFrame, signal_base_name: str, periods_list: list, weights_dict: dict, is_bipolar: bool, is_accel: bool = False, method_name: str = "未知方法", pre_fetched_data: Optional[Dict[str, pd.Series]] = None, tf_weights: Optional[Dict] = None) -> pd.Series:
        """
        【V1.5 · 效率优化与调试增强修正版】计算多时间框架的动态得分。
        - 核心修正: 确保 `get_adaptive_mtf_normalized_score` 和 `get_adaptive_mtf_normalized_bipolar_score`
                    被正确调用，以利用其多窗口并行计算能力。
        - 调试增强: 增加了 `debug_info` 参数的传递，使底层归一化函数也能输出探针信息。
        """
        # tf_weights 参数在这里是 _get_mtf_dynamic_score 内部用于加权 periods_list 的权重
        # 而传递给 get_adaptive_mtf_normalized_score 的 tf_weights 应该是全局的 self.tf_weights_ff
        # 因为 get_adaptive_mtf_normalized_score 内部会用这些权重对单个 Series 进行多时间框架归一化
        if tf_weights is None:
            tf_weights = self.tf_weights_ff # 使用默认权重，但这个 tf_weights 是 _get_mtf_dynamic_score 内部的权重
        numeric_weights = {k: v for k, v in weights_dict.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        if total_weight == 0:
            print(f"    -> [资金流情报警告] 方法 '{method_name}' 中权重总和为0，返回全0 Series。")
            return pd.Series(0.0, index=df.index)
        # 构造 debug_info
        is_debug_enabled = get_param_value(self.debug_params.get('should_probe'), False) and get_param_value(self.debug_params.get('enabled'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            # 找到 df.index 中最接近 probe_dates_dt 的日期
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, f"{method_name}_{signal_base_name}_{'ACCEL' if is_accel else 'SLOPE'}")
        mtf_scores = []
        for period_str, weight in numeric_weights.items():
            period = int(period_str)
            prefix = 'ACCEL' if is_accel else 'SLOPE'
            col_name = f'{prefix}_{period}_{signal_base_name}'
            # 优先从预取数据中获取
            if pre_fetched_data and col_name in pre_fetched_data:
                raw_data = pre_fetched_data[col_name]
            else:
                raw_data = self._get_safe_series(df, df, col_name, 0.0, method_name=method_name)
            # 关键修正：这里传入的 tf_weights 应该是 self.tf_weights_ff，
            # 因为 get_adaptive_mtf_normalized_score 内部会用这些权重对 raw_data 进行多时间框架归一化
            if is_bipolar:
                norm_score = get_adaptive_mtf_normalized_bipolar_score(raw_data, df.index, tf_weights=self.tf_weights_ff)
            else:
                norm_score = get_adaptive_mtf_normalized_score(raw_data, df.index, tf_weights=self.tf_weights_ff, ascending=True)
            mtf_scores.append(norm_score * weight)
        if not mtf_scores:
            return pd.Series(0.0, index=df.index)
        return sum(mtf_scores) / total_weight

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: List[str], method_name: str, atomic_states: Optional[Dict[str, pd.Series]] = None) -> bool:
        """
        【V1.1 · 扩展校验源版】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        - 核心升级: 增加对 atomic_states 的校验，允许信号来自 df.columns 或 atomic_states。
        """
        missing_signals = []
        for s in required_signals:
            if s not in df.columns:
                # 如果信号不在df的列中，则检查是否在atomic_states中
                if atomic_states is None or s not in atomic_states:
                    missing_signals.append(s)
        if missing_signals:
            print(f"    -> [资金流情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def _print_debug_output(self, debug_output: Dict[str, str]):
        """
        辅助方法：统一打印调试信息。
        """
        for key, value in debug_output.items():
            if value:
                print(f"{key}: {value}")
            else:
                print(key)

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V29.2 · 诱多陷阱感知版 & 调试信息统一输出版】资金流情报分析总指挥
        - 核心升级: 优化了 `axiom_divergence` 的计算，使其能更有效识别“诱多陷阱”。
        - 信号原理: 基于微积分思想，对顶层“战略态势”信号进行二阶求导。只有当态势的“速度”与“加速度”
                      同时为正时，才确认为一次高置信度的V型反转拐点。旨在捕捉趋势“破晓”的关键瞬间。
        - 【新增】计算并存储 SCORE_FF_DECEPTION_RISK 信号。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        # 直接使用在 __init__ 中加载的配置
        p_conf = self.p_conf_ff
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df.index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        # 如果没有找到探针日期，则禁用当前方法的调试输出
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, "diagnose_fund_flow_states")
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- diagnose_fund_flow_states 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"FundFlowIntelligence self.debug_params: {self.debug_params}"] = ""
            debug_output["启动【V29.2 · 诱多陷阱感知版】资金流情报分析..."] = ""
        if not get_param_value(p_conf.get('enabled'), True):
            if is_debug_enabled and probe_ts:
                debug_output["-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。"] = ""
                self._print_debug_output(debug_output)
            return {}
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 1. 计算所有原子公理 ---
        # 调整调用顺序：
        # 1. 先计算 SCORE_FF_DECEPTION_RISK，因为它被 axiom_divergence 使用。
        # 2. 接着计算 axiom_divergence，因为它被 axiom_intent_purity 和 _diagnose_fund_flow_divergence_signals 使用。
        # 3. 然后计算其他原子公理。
        score_ff_deception_risk = self._diagnose_deception_risk(df, debug_info_tuple)
        self.strategy.atomic_states['SCORE_FF_DECEPTION_RISK'] = score_ff_deception_risk # 存储诡道风险信号
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        self.strategy.atomic_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence # 存储分歧公理，供后续使用
        axiom_capital_signature = self._diagnose_axiom_capital_signature(df, norm_window)
        axiom_flow_structure_health = self._diagnose_axiom_flow_structure_health(df, norm_window)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_flow_momentum = self._diagnose_axiom_flow_momentum(df, norm_window)
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        axiom_intent_purity = self._diagnose_axiom_intent_purity(df, norm_window) # 现在可以安全调用，因为 axiom_divergence 已存储
        # --- 2. 战略态势的向量合成 (V3.1 · 脆弱性感知版) ---
        fusion_weights = get_param_value(p_conf.get('posture_fusion_weights'), {})
        attack_group = fusion_weights.get('attack_group', {})
        defense_group = fusion_weights.get('defense_group', {})
        harmony_group = fusion_weights.get('harmony_group', {})
        context_group = fusion_weights.get('context_group', {})
        # 2.1 攻击力量 (矛)
        attack_base = (axiom_conviction * attack_group.get('conviction', 0.6) +
                       axiom_flow_momentum * attack_group.get('flow_momentum', 0.4))
        attack_dissonance = abs(axiom_conviction - axiom_flow_momentum)
        attack_dissonance_penalty_sensitivity = get_param_value(attack_group.get('dissonance_penalty_sensitivity'), 0.5)
        attack_dissonance_penalty = np.tanh(attack_dissonance * attack_dissonance_penalty_sensitivity)
        attack_score = attack_base * (1 - attack_dissonance_penalty)
        # 2.2 防御力量 (盾)
        defense_base = (axiom_consensus * defense_group.get('consensus', 0.6) +
                        axiom_flow_structure_health * defense_group.get('flow_health', 0.4))
        defense_dissonance = abs(axiom_consensus - axiom_flow_structure_health)
        defense_dissonance_penalty_sensitivity = get_param_value(defense_group.get('dissonance_penalty_sensitivity'), 0.5)
        defense_dissonance_penalty = np.tanh(defense_dissonance * defense_dissonance_penalty_sensitivity)
        defense_score = defense_base * (1 - defense_dissonance_penalty)
        # 2.3 内部协同度调制器 (Internal Harmony Modulator)
        imbalance = abs(attack_score - defense_score)
        imbalance_penalty_sensitivity = get_param_value(harmony_group.get('imbalance_penalty_sensitivity'), 0.8)
        internal_harmony_modulator = 1 - np.tanh(imbalance * imbalance_penalty_sensitivity)
        # 2.4 情境调节器 (Context Modulator)
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="diagnose_fund_flow_states")
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="diagnose_fund_flow_states")
        norm_capital_signature = (axiom_capital_signature + 1) / 2
        norm_divergence = (axiom_divergence + 1) / 2
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df.index, self.tf_weights_ff)
        norm_volatility_instability_inverse = 1 - get_adaptive_mtf_normalized_score(volatility_instability_raw, df.index, self.tf_weights_ff, ascending=True)
        context_base = (norm_capital_signature * context_group.get('capital_signature', 0.2) +
                        norm_divergence * context_group.get('divergence', 0.2) +
                        norm_flow_credibility * context_group.get('flow_credibility', 0.3) +
                        norm_volatility_instability_inverse * context_group.get('volatility_instability_inverse', 0.3))
        amplification_sensitivity = get_param_value(context_group.get('amplification_sensitivity'), 1.0)
        context_modulator_final = 1 + np.tanh(context_base * amplification_sensitivity)
        # 2.5 脆弱性放大机制 (Vulnerability Amplification)
        extreme_negative_defense_threshold = get_param_value(defense_group.get('extreme_negative_defense_threshold'), -0.7)
        vulnerability_amplification_sensitivity = get_param_value(defense_group.get('vulnerability_amplification_sensitivity'), 1.5)
        defense_vulnerability = (extreme_negative_defense_threshold - defense_score).clip(lower=0)
        vulnerability_penalty = np.tanh(defense_vulnerability * vulnerability_amplification_sensitivity)
        # 2.6 最终战略态势融合
        posture_core = attack_score * (1 + defense_score) / 2
        strategic_posture_score = (posture_core * internal_harmony_modulator * context_modulator_final * (1 - vulnerability_penalty)).clip(-1, 1)
        # --- 3. 和谐拐点计算 ---
        posture_velocity = strategic_posture_score.diff().fillna(0)
        posture_acceleration = posture_velocity.diff().fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_score(posture_velocity, df.index, ascending=True, tf_weights={3:1.0})
        norm_acceleration = get_adaptive_mtf_normalized_score(posture_acceleration, df.index, ascending=True, tf_weights={3:1.0})
        harmony_inflection_score = (norm_velocity.clip(lower=0) * norm_acceleration.clip(lower=0)).pow(0.5)
        # --- 4. 资金流看涨/看跌背离信号 ---
        # 将所有原子公理存储到 self.strategy.atomic_states，以便 _diagnose_fund_flow_divergence_signals 可以获取
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        self.strategy.atomic_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        self.strategy.atomic_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        self.strategy.atomic_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        self.strategy.atomic_states['SCORE_FF_AXIOM_INTENT_PURITY'] = axiom_intent_purity
        self.strategy.atomic_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score
        bullish_divergence, bearish_divergence = self._diagnose_fund_flow_divergence_signals(df, norm_window, axiom_divergence)
        # --- 5. 状态赋值 ---
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        all_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        all_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        all_states['SCORE_FF_AXIOM_INTENT_PURITY'] = axiom_intent_purity.astype(np.float32)
        all_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score.astype(np.float32)
        all_states['SCORE_FF_HARMONY_INFLECTION'] = harmony_inflection_score.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        all_states['SCORE_FF_DECEPTION_RISK'] = score_ff_deception_risk.astype(np.float32)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] diagnose_fund_flow_states @ {probe_ts.strftime('%Y-%m-%d')}: --- 战略态势合成 ---"] = ""
            debug_output[f"        攻击力量 (attack_score): {attack_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        防御力量 (defense_score): {defense_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        内部协同度调制器 (internal_harmony_modulator): {internal_harmony_modulator.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        情境调节器 (context_modulator_final): {context_modulator_final.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        脆弱性惩罚 (vulnerability_penalty): {vulnerability_penalty.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        最终战略态势分数 (strategic_posture_score): {strategic_posture_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"      [资金流层调试] diagnose_fund_flow_states @ {probe_ts.strftime('%Y-%m-%d')}: --- 和谐拐点计算 ---"] = ""
            debug_output[f"        态势速度 (posture_velocity): {posture_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        态势加速度 (posture_acceleration): {posture_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        和谐拐点分数 (harmony_inflection_score): {harmony_inflection_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"      [资金流层调试] diagnose_fund_flow_states @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终原子及融合信号 ---"] = ""
            for key, series in all_states.items():
                debug_output[f"        {key}: {series.loc[probe_ts]:.4f}"] = ""
            debug_output[f"【V29.2 · 诱多陷阱感知版】分析完成，生成 {len(all_states)} 个资金流原子及融合信号。"] = ""
            self._print_debug_output(debug_output)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.4 · 诱多陷阱感知版 & 调试信息统一输出版】资金流公理四：诊断“资金流内部分歧与意图张力”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`，减少重复数据查找。
        - 核心修正: 调整 `core_divergence_score` 计算逻辑，以更准确地捕捉资金流与主力信念之间的看涨/看跌背离。
        - 核心增强: 引入诡道意图张力对结构性张力的调制，使其能更有效识别“诱多陷阱”。
        - 核心增强: 诡道意图张力直接整合 SCORE_FF_DECEPTION_RISK，更全面量化欺骗风险。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_divergence"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        # 如果没有找到探针日期，则禁用当前方法的调试输出
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断资金流内部分歧与意图张力..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        ad_params = get_param_value(p_conf_ff.get('axiom_divergence_params'), {})
        core_divergence_logic = get_param_value(ad_params.get('core_divergence_logic'), 'flow_minus_conviction')
        structural_tension_deception_impact_factor = get_param_value(ad_params.get('structural_tension_deception_impact_factor'), 0.8)
        deception_risk_weight_in_tension = get_param_value(ad_params.get('deception_risk_weight_in_tension'), 1.0)
        divergence_slope_periods = get_param_value(ad_params.get('divergence_slope_periods'), [5, 13, 21, 34, 55])
        raw_divergence_slope_weights = get_param_value(ad_params.get('divergence_slope_weights'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.05, "55": 0.05})
        divergence_slope_weights = {k: v for k, v in raw_divergence_slope_weights.items() if isinstance(v, (int, float))}
        divergence_accel_periods = get_param_value(ad_params.get('divergence_accel_periods'), [5, 13, 21, 34, 55])
        raw_divergence_accel_weights = get_param_value(ad_params.get('divergence_accel_weights'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.05, "55": 0.05})
        divergence_accel_weights = {k: v for k, v in raw_divergence_accel_weights.items() if isinstance(v, (int, float))}
        slope_accel_fusion_weights = get_param_value(ad_params.get('slope_accel_fusion_weights'), {"slope": 0.6, "accel": 0.4})
        energy_injection_signals_weights = get_param_value(ad_params.get('energy_injection_signals'), {'main_force_activity_ratio_D': 0.4, 'main_force_ofi_D': 0.3, 'micro_impact_elasticity_D': 0.3})
        energy_injection_context_modulators = get_param_value(ad_params.get('energy_injection_context_modulators'), {})
        persistence_window = get_param_value(ad_params.get('persistence_window'), 13)
        amplification_factor = get_param_value(ad_params.get('amplification_factor'), 1.5)
        retail_sentiment_mod_sensitivity = get_param_value(ad_params.get('retail_sentiment_mod_sensitivity'), 0.2)
        deception_mod_sensitivity = get_param_value(ad_params.get('deception_mod_sensitivity'), 0.3)
        raw_divergence_component_weights = get_param_value(ad_params.get('divergence_component_weights'), {'core_divergence': 0.3, 'structural_tension': 0.25, 'deceptive_tension': 0.25, 'micro_intent_tension': 0.2})
        divergence_component_weights = {k: v for k, v in raw_divergence_component_weights.items() if isinstance(v, (int, float))}
        micro_intent_tension_signals_weights = get_param_value(ad_params.get('micro_intent_tension_signals'), {'order_book_imbalance_D': 0.5, 'buy_quote_exhaustion_rate_D': 0.25, 'sell_quote_exhaustion_rate_D': 0.25})
        micro_intent_dynamic_signals = get_param_value(ad_params.get('micro_intent_dynamic_signals'), {})
        non_linear_fusion_exponent = get_param_value(ad_params.get('non_linear_fusion_exponent'), 0.8)
        adaptive_weight_modulator_signal_1_name = get_param_value(ad_params.get('adaptive_weight_modulator_signal_1'), 'flow_credibility_index_D')
        adaptive_weight_modulator_signal_2_name = get_param_value(ad_params.get('adaptive_weight_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        adaptive_weight_modulator_signal_3_name = get_param_value(ad_params.get('adaptive_weight_modulator_signal_3'), 'market_sentiment_score_D')
        adaptive_weight_sensitivity_credibility = get_param_value(ad_params.get('adaptive_weight_sensitivity_credibility'), 0.2)
        adaptive_weight_sensitivity_volatility = get_param_value(ad_params.get('adaptive_weight_sensitivity_volatility'), 0.1)
        adaptive_weight_sensitivity_sentiment = get_param_value(ad_params.get('adaptive_weight_sensitivity_sentiment'), 0.1)
        smoothing_ema_span = get_param_value(ad_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ad_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_1_name = get_param_value(ad_params.get('dynamic_evolution_context_modulator_1_name'), 'main_force_conviction_index_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(ad_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        required_signals = []
        for p in divergence_slope_periods:
            required_signals.extend([
                f'SLOPE_{p}_NMFNF_D',
                f'SLOPE_{p}_main_force_conviction_index_D',
                f'SLOPE_{p}_net_lg_amount_calibrated_D',
                f'SLOPE_{p}_retail_net_flow_calibrated_D',
                f'SLOPE_{p}_deception_index_D',
                f'SLOPE_{p}_wash_trade_intensity_D',
                f'SLOPE_{p}_order_book_imbalance_D',
                f'SLOPE_{p}_buy_quote_exhaustion_rate_D',
                f'SLOPE_{p}_sell_quote_exhaustion_rate_D'
            ])
        for p in divergence_accel_periods:
            required_signals.extend([
                f'ACCEL_{p}_NMFNF_D',
                f'ACCEL_{p}_main_force_conviction_index_D',
                f'ACCEL_{p}_net_lg_amount_calibrated_D',
                f'ACCEL_{p}_retail_net_flow_calibrated_D',
                f'ACCEL_{p}_deception_index_D',
                f'ACCEL_{p}_wash_trade_intensity_D',
                f'ACCEL_{p}_order_book_imbalance_D',
                f'ACCEL_{p}_buy_quote_exhaustion_rate_D',
                f'ACCEL_{p}_sell_quote_exhaustion_rate_D'
            ])
        required_signals.extend([
            'retail_fomo_premium_index_D',
            'retail_panic_surrender_index_D',
            'flow_credibility_index_D',
            'wash_trade_intensity_D',
            'main_force_activity_ratio_D',
            'main_force_ofi_D',
            'micro_impact_elasticity_D',
            'order_book_imbalance_D',
            'buy_quote_exhaustion_rate_D',
            'sell_quote_exhaustion_rate_D',
            adaptive_weight_modulator_signal_1_name,
            adaptive_weight_modulator_signal_2_name,
            adaptive_weight_modulator_signal_3_name,
            dynamic_evolution_context_modulator_1_name
        ])
        for mod_name, mod_params in energy_injection_context_modulators.items():
            if isinstance(mod_params, dict) and 'signal' in mod_params:
                required_signals.append(mod_params['signal'])
        required_signals.append('SCORE_FF_DECEPTION_RISK') # 确保 SCORE_FF_DECEPTION_RISK 存在于 atomic_states
        if not self._validate_required_signals(df, required_signals, method_name, atomic_states=self.strategy.atomic_states):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        signal_bases_to_prefetch = [
            'NMFNF_D', 'main_force_conviction_index_D', 'net_lg_amount_calibrated_D',
            'retail_net_flow_calibrated_D', 'deception_index_D', 'wash_trade_intensity_D',
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D'
        ]
        for signal_base in signal_bases_to_prefetch:
            for p in divergence_slope_periods:
                col_name = f'SLOPE_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name=method_name)
            for p in divergence_accel_periods:
                col_name = f'ACCEL_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name=method_name)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name in all_pre_fetched_slopes_accels:
                raw_data_cache[signal_name] = all_pre_fetched_slopes_accels[signal_name]
            else:
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        retail_fomo_premium_raw = raw_data_cache['retail_fomo_premium_index_D']
        retail_panic_surrender_raw = raw_data_cache['retail_panic_surrender_index_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        order_book_imbalance_raw = raw_data_cache['order_book_imbalance_D']
        buy_exhaustion_raw = raw_data_cache['buy_quote_exhaustion_rate_D']
        sell_exhaustion_raw = raw_data_cache['sell_quote_exhaustion_rate_D']
        mf_activity_ratio_raw = raw_data_cache['main_force_activity_ratio_D']
        mf_ofi_raw = raw_data_cache['main_force_ofi_D']
        micro_impact_elasticity_raw = raw_data_cache['micro_impact_elasticity_D']
        energy_modulator_signals = {}
        for mod_name, mod_params in energy_injection_context_modulators.items():
            if isinstance(mod_params, dict) and 'signal' in mod_params:
                energy_modulator_signals[mod_name] = self._get_safe_series(df, df, mod_params['signal'], 0.0, method_name=method_name)
        adaptive_weight_modulator_1_raw = raw_data_cache[adaptive_weight_modulator_signal_1_name]
        adaptive_weight_modulator_2_raw = raw_data_cache[adaptive_weight_modulator_signal_2_name]
        adaptive_weight_modulator_3_raw = raw_data_cache[adaptive_weight_modulator_signal_3_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_1_name]
        score_ff_deception_risk = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_DECEPTION_RISK', 0.0, method_name=method_name)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始数据 ---"] = ""
            debug_output[f"        retail_fomo_premium_raw: {retail_fomo_premium_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        retail_panic_surrender_raw: {retail_panic_surrender_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        flow_credibility_raw: {flow_credibility_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        order_book_imbalance_raw: {order_book_imbalance_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        buy_exhaustion_raw: {buy_exhaustion_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        sell_exhaustion_raw: {sell_exhaustion_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        mf_activity_ratio_raw: {mf_activity_ratio_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        mf_ofi_raw: {mf_ofi_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        micro_impact_elasticity_raw: {micro_impact_elasticity_raw.loc[probe_ts]:.4f}"] = ""
            for mod_name, signal_series in energy_modulator_signals.items():
                debug_output[f"        energy_modulator_signals[{mod_name}]: {signal_series.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        adaptive_weight_modulator_1_raw: {adaptive_weight_modulator_1_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        adaptive_weight_modulator_2_raw: {adaptive_weight_modulator_2_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        adaptive_weight_modulator_3_raw: {adaptive_weight_modulator_3_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_evolution_context_modulator_1_raw: {dynamic_evolution_context_modulator_1_raw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        SCORE_FF_DECEPTION_RISK: {score_ff_deception_risk.loc[probe_ts]:.4f}"] = ""
        # --- 1. 核心分歧向量 (Core Divergence Vector) ---
        norm_nmfnf_slope_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_nmfnf_accel_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_mf_conviction_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_mf_conviction_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        nmfnf_dynamic_score = (norm_nmfnf_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_nmfnf_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        mf_conviction_dynamic_score = (norm_mf_conviction_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_mf_conviction_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        if core_divergence_logic == 'flow_conviction_tension':
            conditions = [
                (nmfnf_dynamic_score > 0) & (mf_conviction_dynamic_score > 0),
                (nmfnf_dynamic_score < 0) & (mf_conviction_dynamic_score < 0),
                (nmfnf_dynamic_score < 0) & (mf_conviction_dynamic_score > 0),
                (nmfnf_dynamic_score > 0) & (mf_conviction_dynamic_score < 0)
            ]
            choices = [
                (nmfnf_dynamic_score + mf_conviction_dynamic_score) / 2,
                (nmfnf_dynamic_score + mf_conviction_dynamic_score) / 2,
                (mf_conviction_dynamic_score - nmfnf_dynamic_score) / 2,
                (mf_conviction_dynamic_score - nmfnf_dynamic_score) / 2
            ]
            core_divergence_score = np.select(conditions, choices, default=(nmfnf_dynamic_score + mf_conviction_dynamic_score) / 2)
            core_divergence_score = pd.Series(core_divergence_score, index=df_index).clip(-1, 1)
        else: # Original logic: flow_minus_conviction
            core_divergence_score = (nmfnf_dynamic_score - mf_conviction_dynamic_score).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 核心分歧向量 ---"] = ""
            debug_output[f"        nmfnf_dynamic_score: {nmfnf_dynamic_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        mf_conviction_dynamic_score: {mf_conviction_dynamic_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        core_divergence_score ({core_divergence_logic}): {core_divergence_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 诡道意图张力 (Deceptive Intent Tension) ---
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        deceptive_tension_score = (norm_flow_credibility - score_ff_deception_risk * deception_risk_weight_in_tension).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道意图张力 ---"] = ""
            debug_output[f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        SCORE_FF_DECEPTION_RISK: {score_ff_deception_risk.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        deception_risk_weight_in_tension: {deception_risk_weight_in_tension:.4f}"] = ""
            debug_output[f"        deceptive_tension_score: {deceptive_tension_score.loc[probe_ts]:.4f}"] = ""
        # --- 3. 结构性张力 (Structural Tension) ---
        norm_lg_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_lg_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_retail_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_retail_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        lg_flow_dynamic_score = (norm_lg_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_lg_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        retail_flow_dynamic_score = (norm_retail_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_retail_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        structural_divergence_base = (lg_flow_dynamic_score - retail_flow_dynamic_score)
        retail_modulator = (1 - norm_retail_fomo * retail_sentiment_mod_sensitivity) + (norm_retail_panic * retail_sentiment_mod_sensitivity)
        deception_modulator_for_structural_tension = (1 + deceptive_tension_score * structural_tension_deception_impact_factor).clip(0.1, 2.0)
        structural_tension_score = (structural_divergence_base * retail_modulator * deception_modulator_for_structural_tension).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 结构性张力 ---"] = ""
            debug_output[f"        norm_lg_flow_slope_mtf: {norm_lg_flow_slope_mtf.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_lg_flow_accel_mtf: {norm_lg_flow_accel_mtf.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_flow_slope_mtf: {norm_retail_flow_slope_mtf.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_flow_accel_mtf: {norm_retail_flow_accel_mtf.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        lg_flow_dynamic_score: {lg_flow_dynamic_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        retail_flow_dynamic_score: {retail_flow_dynamic_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_fomo: {norm_retail_fomo.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_panic: {norm_retail_panic.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        structural_divergence_base: {structural_divergence_base.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        retail_modulator: {retail_modulator.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        deception_modulator_for_structural_tension: {deception_modulator_for_structural_tension.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        structural_tension_score: {structural_tension_score.loc[probe_ts]:.4f}"] = ""
        # --- 4. 微观意图张力 (Micro-Flow Intent Tension) ---
        norm_order_book_imbalance = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights=self.tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=self.tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        micro_exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        obi_dynamic_params = micro_intent_dynamic_signals.get('order_book_imbalance_D', {"slope": 0.6, "accel": 0.4, "weight": 0.2})
        norm_obi_slope_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_obi_accel_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        obi_dynamic_pulse = (norm_obi_slope_mtf * obi_dynamic_params.get('slope', 0.6) + norm_obi_accel_mtf * obi_dynamic_params.get('accel', 0.4)).clip(-1, 1)
        buy_exh_dynamic_params = micro_intent_dynamic_signals.get('buy_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        norm_buy_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_buy_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        buy_exh_dynamic_pulse = (norm_buy_exh_slope_mtf * buy_exh_dynamic_params.get('slope', 0.5) + norm_buy_exh_accel_mtf * buy_exh_dynamic_params.get('accel', 0.5)).clip(0, 1)
        sell_exh_dynamic_params = micro_intent_dynamic_signals.get('sell_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        norm_sell_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_sell_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name=method_name, pre_fetched_data=all_pre_fetched_slopes_accels)
        sell_exh_dynamic_pulse = (norm_sell_exh_slope_mtf * sell_exh_dynamic_params.get('slope', 0.5) + norm_sell_exh_accel_mtf * sell_exh_dynamic_params.get('accel', 0.5)).clip(0, 1)
        micro_dynamic_exhaustion_score = (sell_exh_dynamic_pulse - buy_exh_dynamic_pulse).clip(-1, 1)
        micro_intent_tension_score = (
            norm_order_book_imbalance * micro_intent_tension_signals_weights.get('order_book_imbalance_D', 0.5) +
            micro_exhaustion_score * (micro_intent_tension_signals_weights.get('buy_quote_exhaustion_rate_D', 0.25) + micro_intent_tension_signals_weights.get('sell_quote_exhaustion_rate_D', 0.25)) +
            obi_dynamic_pulse * obi_dynamic_params.get('weight', 0.2) +
            micro_dynamic_exhaustion_score * (buy_exh_dynamic_params.get('weight', 0.15) + sell_exh_dynamic_params.get('weight', 0.15))
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 微观意图张力 ---"] = ""
            debug_output[f"        norm_order_book_imbalance: {norm_order_book_imbalance.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_buy_exhaustion: {norm_buy_exhaustion.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sell_exhaustion: {norm_sell_exhaustion.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        micro_exhaustion_score: {micro_exhaustion_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        obi_dynamic_pulse: {obi_dynamic_pulse.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        buy_exh_dynamic_pulse: {buy_exh_dynamic_pulse.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        sell_exh_dynamic_pulse: {sell_exh_dynamic_pulse.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        micro_dynamic_exhaustion_score: {micro_dynamic_exhaustion_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        micro_intent_tension_score: {micro_intent_tension_score.loc[probe_ts]:.4f}"] = ""
        # --- 5. 能量注入与持续性 (Energy Injection & Persistence) ---
        norm_mf_activity = get_adaptive_mtf_normalized_score(mf_activity_ratio_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_mf_ofi = get_adaptive_mtf_normalized_score(mf_ofi_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_micro_impact_elasticity = get_adaptive_mtf_normalized_score(micro_impact_elasticity_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        energy_injection_base = (
            norm_mf_activity * energy_injection_signals_weights.get('main_force_activity_ratio_D', 0.4) +
            norm_mf_ofi * energy_injection_signals_weights.get('main_force_ofi_D', 0.3) +
            norm_micro_impact_elasticity * energy_injection_signals_weights.get('micro_impact_elasticity_D', 0.3)
        ).clip(0, 1)
        energy_injection_modulator = pd.Series(1.0, index=df_index)
        for mod_name, mod_params in energy_injection_context_modulators.items():
            if isinstance(mod_params, dict) and 'signal' in mod_params:
                mod_signal = energy_modulator_signals[mod_name]
                norm_mod_signal = get_adaptive_mtf_normalized_score(mod_signal, df_index, ascending=mod_params.get('ascending', True), tf_weights=self.tf_weights_ff)
                energy_injection_modulator *= (1 + (norm_mod_signal - 0.5) * mod_params.get('sensitivity', 0.0))
        energy_injection = (energy_injection_base * energy_injection_modulator).clip(0, 1)
        combined_divergence_abs = (
            core_divergence_score.abs() * divergence_component_weights.get('core_divergence', 0.3) +
            structural_tension_score.abs() * divergence_component_weights.get('structural_tension', 0.25) +
            deceptive_tension_score.abs() * divergence_component_weights.get('deceptive_tension', 0.25) +
            micro_intent_tension_score.abs() * divergence_component_weights.get('micro_intent_tension', 0.2)
        )
        persistence = combined_divergence_abs.rolling(window=persistence_window, min_periods=max(1, int(persistence_window * 0.2))).std().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        eip_score = (energy_injection * norm_persistence).pow(0.5).clip(0, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 能量注入与持续性 ---"] = ""
            debug_output[f"        norm_mf_activity: {norm_mf_activity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_mf_ofi: {norm_mf_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_micro_impact_elasticity: {norm_micro_impact_elasticity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        energy_injection_base: {energy_injection_base.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        energy_injection_modulator: {energy_injection_modulator.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        energy_injection: {energy_injection.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        combined_divergence_abs: {combined_divergence_abs.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        persistence: {persistence.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_persistence: {norm_persistence.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        eip_score: {eip_score.loc[probe_ts]:.4f}"] = ""
        # --- 6. 非线性融合与情境自适应权重 ---
        norm_adaptive_weight_modulator_1 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_1_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_adaptive_weight_modulator_2 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_2_raw, df_index, ascending=False, tf_weights=self.tf_weights_ff)
        norm_adaptive_weight_modulator_3 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_3_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        total_base_weight = sum(divergence_component_weights.values())
        if total_base_weight == 0:
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 警告：divergence_component_weights总和为0，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        dynamic_core_divergence_weight = divergence_component_weights.get('core_divergence', 0.3) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility - norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility + norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        dynamic_structural_tension_weight = divergence_component_weights.get('structural_tension', 0.25) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility + norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        dynamic_deceptive_tension_weight = divergence_component_weights.get('deceptive_tension', 0.25) * (1 - norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility - norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        dynamic_micro_intent_tension_weight = divergence_component_weights.get('micro_intent_tension', 0.2) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility + norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        sum_dynamic_weights = dynamic_core_divergence_weight + dynamic_structural_tension_weight + dynamic_deceptive_tension_weight + dynamic_micro_intent_tension_weight
        sum_dynamic_weights = sum_dynamic_weights.replace(0, 1.0) # 避免除以零
        dynamic_core_divergence_weight = dynamic_core_divergence_weight / sum_dynamic_weights
        dynamic_structural_tension_weight = dynamic_structural_tension_weight / sum_dynamic_weights
        dynamic_deceptive_tension_weight = dynamic_deceptive_tension_weight / sum_dynamic_weights
        dynamic_micro_intent_tension_weight = dynamic_micro_intent_tension_weight / sum_dynamic_weights
        fused_divergence_base = (
            (core_divergence_score.add(1)/2).pow(dynamic_core_divergence_weight) *
            (structural_tension_score.add(1)/2).pow(dynamic_structural_tension_weight) *
            (deceptive_tension_score.add(1)/2).pow(dynamic_deceptive_tension_weight) *
            (micro_intent_tension_score.add(1)/2).pow(dynamic_micro_intent_tension_weight)
        ).pow(1 / (dynamic_core_divergence_weight + dynamic_structural_tension_weight + dynamic_deceptive_tension_weight + dynamic_micro_intent_tension_weight)) * 2 - 1
        base_tension_score = (fused_divergence_base * (1 + eip_score * amplification_factor)).clip(-1, 1)
        base_tension_score = np.tanh(base_tension_score * non_linear_fusion_exponent)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 非线性融合 ---"] = ""
            debug_output[f"        norm_adaptive_weight_modulator_1: {norm_adaptive_weight_modulator_1.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_adaptive_weight_modulator_2: {norm_adaptive_weight_modulator_2.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_adaptive_weight_modulator_3: {norm_adaptive_weight_modulator_3.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_core_divergence_weight: {dynamic_core_divergence_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_structural_tension_weight: {dynamic_structural_tension_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_deceptive_tension_weight: {dynamic_deceptive_tension_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_micro_intent_tension_weight: {dynamic_micro_intent_tension_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        fused_divergence_base: {fused_divergence_base.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        base_tension_score (before tanh): {fused_divergence_base.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        base_tension_score (after tanh): {base_tension_score.loc[probe_ts]:.4f}"] = ""
        # --- 7. 张力演化趋势与预警 (Tension Evolution & Early Warning) ---
        smoothed_base_score = base_tension_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=self.tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=self.tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_1_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1.0)
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_tension_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 演化趋势 ---"] = ""
            debug_output[f"        smoothed_base_score: {smoothed_base_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        velocity: {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        acceleration: {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_velocity: {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_acceleration: {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_dynamic_evolution_context_1: {norm_dynamic_evolution_context_1.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_base_score_weight: {dynamic_base_score_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_velocity_weight: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        dynamic_acceleration_weight: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流内部分歧与意图张力诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self._print_debug_output(debug_output) # 统一输出调试信息
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V6.3 · 诡道风险分离版 & 调试信息统一输出版】资金流公理一：诊断“战场控制权”
        - 核心升级1: 诡道风险分离：将诡道博弈调制逻辑分离为独立的 `_diagnose_deception_risk` 方法，输出 `SCORE_FF_DECEPTION_RISK`。
        - 核心升级2: 战场控制权引用诡道风险：`SCORE_FF_DECEPTION_RISK` 作为负向调制器应用于 `base_battlefield_control_score`，使其更直接地反映诡道风险对主力控制力的削弱。
        - 核心升级3: 宏观资金流质量深化：引入主力资金流方向性、基尼系数等，更精细评估宏观流向品质。
        - 核心升级4: 微观控制力增强：整合市场冲击成本、流动性斜率、扫单强度等，捕捉更细致的微观影响力。
        - 核心升级5: 非线性微观-宏观交互：改变最终融合方式，允许微观信号非线性地放大或抑制宏观信号。
        - 核心升级6: 动态权重情境扩展：增加趋势活力、结构张力作为动态权重调制器，适应更广泛市场情境。
        - 核心升级7: 详细探针输出：增加print语句，方便调试和理解计算过程。
        - 核心升级8: 引入多时间框架共振因子：评估关键资金流信号在不同时间框架下的协同或背离，作为重要的调制器。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_consensus"
        df_index = df.index
        p_conf_ff = self.p_conf_ff
        ac_params = get_param_value(p_conf_ff.get('axiom_consensus_params'), {})
        is_debug_enabled_for_method = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled_for_method and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        # 如果没有找到探针日期，则禁用当前方法的调试输出
        if probe_ts is None:
            is_debug_enabled_for_method = False
        debug_info_tuple = (is_debug_enabled_for_method, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        dynamic_weight_mod_enabled = get_param_value(ac_params.get('dynamic_weight_mod_enabled'), True)
        macro_flow_base_weight = get_param_value(ac_params.get('macro_flow_base_weight'), 0.4)
        micro_control_base_weight = get_param_value(ac_params.get('micro_control_base_weight'), 0.6)
        dynamic_weight_modulator_signal_1_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_2'), 'SLOPE_5_NMFNF_D')
        dynamic_weight_modulator_signal_3_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_3'), 'market_sentiment_score_D')
        dynamic_weight_modulator_signal_4_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_4'), 'trend_vitality_index_D')
        dynamic_weight_modulator_signal_5_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_5'), 'structural_tension_index_D')
        dynamic_weight_sensitivity_volatility = get_param_value(ac_params.get('dynamic_weight_sensitivity_volatility'), 0.3)
        dynamic_weight_sensitivity_flow_slope = get_param_value(ac_params.get('dynamic_weight_sensitivity_flow_slope'), 0.2)
        dynamic_weight_sensitivity_sentiment = get_param_value(ac_params.get('dynamic_weight_sensitivity_sentiment'), 0.1)
        dynamic_weight_sensitivity_trend_vitality = get_param_value(ac_params.get('dynamic_weight_sensitivity_trend_vitality'), 0.1)
        dynamic_weight_sensitivity_structural_tension = get_param_value(ac_params.get('dynamic_weight_sensitivity_structural_tension'), 0.1)
        asymmetric_micro_control_enabled = get_param_value(ac_params.get('asymmetric_micro_control_enabled'), True)
        exhaustion_boost_factor = get_param_value(ac_params.get('exhaustion_boost_factor'), 0.2)
        exhaustion_penalty_factor = get_param_value(ac_params.get('exhaustion_penalty_factor'), 0.3)
        micro_intent_fusion_weights = get_param_value(ac_params.get('micro_intent_fusion_weights'), {'imbalance': 0.4, 'efficiency': 0.3, 'exhaustion': 0.3})
        micro_buy_power_weights = get_param_value(ac_params.get('micro_buy_power_weights'), {})
        micro_sell_power_weights = get_param_value(ac_params.get('micro_sell_power_weights'), {})
        macro_flow_quality_weights = get_param_value(ac_params.get('macro_flow_quality_weights'), {})
        micro_control_quality_weights = get_param_value(ac_params.get('micro_control_quality_weights'), {})
        micro_macro_interaction_exponent = get_param_value(ac_params.get('micro_macro_interaction_exponent'), 1.2)
        smoothing_ema_span = get_param_value(ac_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ac_params.get('dynamic_evolution_base_weights'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_name = get_param_value(ac_params.get('dynamic_evolution_context_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_evolution_context_sensitivity = get_param_value(ac_params.get('dynamic_evolution_context_sensitivity'), 0.2)
        # MTF共振因子参数
        mtf_cohesion_params = get_param_value(ac_params.get('mtf_cohesion_params'), {})
        mtf_cohesion_enabled = get_param_value(mtf_cohesion_params.get('enabled'), True)
        mtf_cohesion_short_periods = get_param_value(mtf_cohesion_params.get('short_periods'), [5, 13])
        mtf_cohesion_long_periods = get_param_value(mtf_cohesion_params.get('long_periods'), [21, 55])
        mtf_cohesion_modulator_sensitivity = get_param_value(mtf_cohesion_params.get('modulator_sensitivity'), 0.5)
        mtf_cohesion_macro_flow_weights = get_param_value(mtf_cohesion_params.get('macro_flow_weights'), {"main_force_flow_directionality": 0.5, "nmfnf": 0.5})
        mtf_cohesion_micro_control_weights = get_param_value(mtf_cohesion_params.get('micro_control_weights'), {"order_book_imbalance": 0.5, "microstructure_efficiency": 0.5})
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'ATR_14_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D',
            'buy_order_book_clearing_rate_D', 'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'main_force_flow_gini_D', 'order_book_imbalance_D', 'flow_credibility_index_D',
            'microstructure_efficiency_index_D', 'wash_trade_intensity_D',
            'deception_index_D', 'main_force_conviction_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name,
            dynamic_weight_modulator_signal_3_name, dynamic_weight_modulator_signal_4_name, dynamic_weight_modulator_signal_5_name,
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            dynamic_evolution_context_modulator_signal_name,
            'dip_buy_absorption_strength_D', 'dip_sell_pressure_resistance_D',
            'panic_sell_volume_contribution_D', 'panic_buy_absorption_contribution_D',
            'opening_buy_strength_D', 'opening_sell_strength_D',
            'pre_closing_buy_posture_D', 'pre_closing_sell_posture_D',
            'closing_auction_buy_ambush_D', 'closing_auction_sell_ambush_D',
            'main_force_t0_buy_efficiency_D', 'main_force_t0_sell_efficiency_D',
            'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D',
            'vwap_cross_up_intensity_D', 'vwap_cross_down_intensity_D',
            'main_force_on_peak_sell_flow_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D',
            'main_force_flow_directionality_D', 'NMFNF_D',
            'market_impact_cost_D', 'liquidity_slope_D', 'liquidity_authenticity_score_D',
            'buy_sweep_intensity_D', 'sell_sweep_intensity_D', 'order_flow_imbalance_score_D',
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D'
        ]
        required_signals = list(set(required_signals))
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        all_mtf_periods = list(set(mtf_cohesion_short_periods + mtf_cohesion_long_periods))
        signal_bases_for_mtf_cohesion = [
            'main_force_flow_directionality_D', 'NMFNF_D', 'order_book_imbalance_D',
            'microstructure_efficiency_index_D', 'deception_index_D', 'wash_trade_intensity_D'
        ]
        for signal_base in signal_bases_for_mtf_cohesion:
            for p in all_mtf_periods:
                all_pre_fetched_slopes_accels[f'SLOPE_{p}_{signal_base}'] = self._get_safe_series(df, df, f'SLOPE_{p}_{signal_base}', 0.0, method_name=method_name)
                all_pre_fetched_slopes_accels[f'ACCEL_{p}_{signal_base}'] = self._get_safe_series(df, df, f'ACCEL_{p}_{signal_base}', 0.0, method_name=method_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 集中所有原始数据获取
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name in all_pre_fetched_slopes_accels:
                raw_data_cache[signal_name] = all_pre_fetched_slopes_accels[signal_name]
            else:
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        # 提取常用信号到局部变量
        main_force_flow_raw = raw_data_cache['main_force_net_flow_calibrated_D']
        retail_flow_raw = raw_data_cache['retail_net_flow_calibrated_D']
        order_book_imbalance_raw = raw_data_cache['order_book_imbalance_D']
        ofi_impact_raw = raw_data_cache['microstructure_efficiency_index_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        volatility_instability_raw = raw_data_cache[dynamic_weight_modulator_signal_1_name]
        flow_slope_raw = raw_data_cache[dynamic_weight_modulator_signal_2_name]
        market_sentiment_raw = raw_data_cache[dynamic_weight_modulator_signal_3_name]
        trend_vitality_raw = raw_data_cache[dynamic_weight_modulator_signal_4_name]
        structural_tension_raw = raw_data_cache[dynamic_weight_modulator_signal_5_name]
        buy_exhaustion_raw = raw_data_cache['buy_quote_exhaustion_rate_D']
        sell_exhaustion_raw = raw_data_cache['sell_quote_exhaustion_rate_D']
        dynamic_evolution_context_modulator_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_name]
        main_force_flow_directionality_raw = raw_data_cache['main_force_flow_directionality_D']
        main_force_flow_gini_raw = raw_data_cache['main_force_flow_gini_D']
        nmfnf_raw = raw_data_cache['NMFNF_D']
        market_impact_cost_raw = raw_data_cache['market_impact_cost_D']
        liquidity_slope_raw = raw_data_cache['liquidity_slope_D']
        liquidity_authenticity_raw = raw_data_cache['liquidity_authenticity_score_D']
        buy_sweep_intensity_raw = raw_data_cache['buy_sweep_intensity_D']
        sell_sweep_intensity_raw = raw_data_cache['sell_sweep_intensity_D']
        order_flow_imbalance_score_raw = raw_data_cache['order_flow_imbalance_score_D']
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        # --- MTF共振因子计算 ---
        macro_flow_directionality_cohesion = pd.Series(1.0, index=df_index)
        nmfnf_cohesion = pd.Series(1.0, index=df_index)
        micro_imbalance_cohesion = pd.Series(1.0, index=df_index)
        micro_efficiency_cohesion = pd.Series(1.0, index=df_index)
        if mtf_cohesion_enabled:
            macro_flow_directionality_cohesion = self._calculate_mtf_cohesion_divergence(df, 'main_force_flow_directionality_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            nmfnf_cohesion = self._calculate_mtf_cohesion_divergence(df, 'NMFNF_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            micro_imbalance_cohesion = self._calculate_mtf_cohesion_divergence(df, 'order_book_imbalance_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            micro_efficiency_cohesion = self._calculate_mtf_cohesion_divergence(df, 'microstructure_efficiency_index_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            if is_debug_enabled_for_method and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF共振因子计算 ---"] = ""
                debug_output[f"        macro_flow_directionality_cohesion: {macro_flow_directionality_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        nmfnf_cohesion: {nmfnf_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        micro_imbalance_cohesion: {micro_imbalance_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        micro_efficiency_cohesion: {micro_efficiency_cohesion.loc[probe_ts]:.4f}"] = ""
        # --- 1. 宏观资金流质量 (Enhanced Macro Fund Flow Quality) ---
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights_ff)
        norm_main_force_flow_gini_inverted = 1 - get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_nmfnf_net_flow = get_adaptive_mtf_normalized_bipolar_score(nmfnf_raw, df_index, tf_weights_ff)
        macro_flow_quality_score = (
            norm_main_force_flow_directionality * mtf_cohesion_macro_flow_weights.get('main_force_flow_directionality', 0.5) * (1 + macro_flow_directionality_cohesion * mtf_cohesion_macro_flow_weights.get('main_force_flow_directionality', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_main_force_flow_gini_inverted * macro_flow_quality_weights.get('main_force_flow_gini_inverted', 0.2) +
            norm_nmfnf_net_flow * mtf_cohesion_macro_flow_weights.get('nmfnf', 0.5) * (1 + nmfnf_cohesion * mtf_cohesion_macro_flow_weights.get('nmfnf', 0.5) * mtf_cohesion_modulator_sensitivity)
        ).clip(-1, 1)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 宏观资金流质量计算 ---"] = ""
            debug_output[f"        主力资金流方向性归一化: {norm_main_force_flow_directionality.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        主力资金流基尼系数反向归一化: {norm_main_force_flow_gini_inverted.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        NMFNF净流量归一化: {norm_nmfnf_net_flow.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        宏观资金流质量分数: {macro_flow_quality_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 微观盘口意图推断 (Enhanced Micro Order Book Intent Inference) ---
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff)
        impact_score = get_adaptive_mtf_normalized_bipolar_score(ofi_impact_raw, df_index, tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        norm_market_impact_cost_inverted = 1 - get_adaptive_mtf_normalized_score(market_impact_cost_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_liquidity_slope = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_raw, df_index, tf_weights_ff)
        norm_liquidity_authenticity = get_adaptive_mtf_normalized_score(liquidity_authenticity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_sweep_intensity = get_adaptive_mtf_normalized_score(buy_sweep_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_sweep_intensity_inverted = 1 - get_adaptive_mtf_normalized_score(sell_sweep_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_order_flow_imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_flow_imbalance_score_raw, df_index, tf_weights=tf_weights_ff)
        micro_control_quality_score = (
            norm_market_impact_cost_inverted * micro_control_quality_weights.get('market_impact_cost_inverted', 0.2) +
            norm_liquidity_slope * micro_control_quality_weights.get('liquidity_slope', 0.2) +
            norm_liquidity_authenticity * micro_control_quality_weights.get('liquidity_authenticity', 0.1) +
            norm_buy_sweep_intensity * micro_control_quality_weights.get('buy_sweep_intensity', 0.2) +
            norm_sell_sweep_intensity_inverted * micro_control_quality_weights.get('sell_sweep_intensity_inverted', 0.2) +
            norm_order_flow_imbalance_score * micro_control_quality_weights.get('order_flow_imbalance', 0.1)
        ).clip(-1, 1)
        norm_dip_buy_absorption_strength = get_adaptive_mtf_normalized_score(raw_data_cache['dip_buy_absorption_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_panic_buy_absorption_contribution = get_adaptive_mtf_normalized_score(raw_data_cache['panic_buy_absorption_contribution_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_opening_buy_strength = get_adaptive_mtf_normalized_score(raw_data_cache['opening_buy_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_pre_closing_buy_posture = get_adaptive_mtf_normalized_score(raw_data_cache['pre_closing_buy_posture_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_closing_auction_buy_ambush = get_adaptive_mtf_normalized_score(raw_data_cache['closing_auction_buy_ambush_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_bid_side_liquidity = get_adaptive_mtf_normalized_score(raw_data_cache['bid_side_liquidity_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_t0_buy_efficiency = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_t0_buy_efficiency_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_flow_efficiency_index = get_adaptive_mtf_normalized_score(raw_data_cache['buy_flow_efficiency_index_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_order_book_clearing_rate = get_adaptive_mtf_normalized_score(raw_data_cache['buy_order_book_clearing_rate_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_buy_control_strength = get_adaptive_mtf_normalized_score(raw_data_cache['vwap_buy_control_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_vwap_up_guidance = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_vwap_up_guidance_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_cross_up_intensity = get_adaptive_mtf_normalized_score(raw_data_cache['vwap_cross_up_intensity_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(raw_data_cache['wash_trade_buy_volume_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_dip_sell_pressure_resistance = get_adaptive_mtf_normalized_score(raw_data_cache['dip_sell_pressure_resistance_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_panic_sell_volume_contribution = get_adaptive_mtf_normalized_score(raw_data_cache['panic_sell_volume_contribution_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_opening_sell_strength = get_adaptive_mtf_normalized_score(raw_data_cache['opening_sell_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_pre_closing_sell_posture = get_adaptive_mtf_normalized_score(raw_data_cache['pre_closing_sell_posture_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_closing_auction_sell_ambush = get_adaptive_mtf_normalized_score(raw_data_cache['closing_auction_sell_ambush_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_ask_side_liquidity = get_adaptive_mtf_normalized_score(raw_data_cache['ask_side_liquidity_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_on_peak_sell_flow = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_on_peak_sell_flow_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_t0_sell_efficiency = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_t0_sell_efficiency_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_flow_efficiency_index = get_adaptive_mtf_normalized_score(raw_data_cache['sell_flow_efficiency_index_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_order_book_clearing_rate = get_adaptive_mtf_normalized_score(raw_data_cache['sell_order_book_clearing_rate_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_sell_control_strength = get_adaptive_mtf_normalized_score(raw_data_cache['vwap_sell_control_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_vwap_down_guidance = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_vwap_down_guidance_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_cross_down_intensity = get_adaptive_mtf_normalized_score(raw_data_cache['vwap_cross_down_intensity_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(raw_data_cache['wash_trade_sell_volume_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        total_buy_power = (
            norm_dip_buy_absorption_strength * micro_buy_power_weights.get('dip_buy_absorption_strength', 0.1) +
            norm_panic_buy_absorption_contribution * micro_buy_power_weights.get('panic_buy_absorption_contribution', 0.1) +
            norm_opening_buy_strength * micro_buy_power_weights.get('opening_buy_strength', 0.1) +
            norm_pre_closing_buy_posture * micro_buy_power_weights.get('pre_closing_buy_posture', 0.1) +
            norm_closing_auction_buy_ambush * micro_buy_power_weights.get('closing_auction_buy_ambush', 0.1) +
            norm_bid_side_liquidity * micro_buy_power_weights.get('bid_side_liquidity', 0.05) +
            norm_main_force_t0_buy_efficiency * micro_buy_power_weights.get('main_force_t0_buy_efficiency', 0.05) +
            norm_buy_flow_efficiency_index * micro_buy_power_weights.get('buy_flow_efficiency_index', 0.05) +
            norm_buy_order_book_clearing_rate * micro_buy_power_weights.get('buy_order_book_clearing_rate', 0.05) +
            norm_vwap_buy_control_strength * micro_buy_power_weights.get('vwap_buy_control_strength', 0.05) +
            norm_main_force_vwap_up_guidance * micro_buy_power_weights.get('main_force_vwap_up_guidance', 0.05) +
            norm_vwap_cross_up_intensity * micro_buy_power_weights.get('vwap_cross_up_intensity', 0.05) +
            norm_wash_trade_buy_volume * micro_buy_power_weights.get('wash_trade_buy_volume', -0.05)
        ).clip(0, 1)
        total_sell_power = (
            norm_dip_sell_pressure_resistance * micro_sell_power_weights.get('dip_sell_pressure_resistance', 0.1) +
            norm_panic_sell_volume_contribution * micro_sell_power_weights.get('panic_sell_volume_contribution', 0.1) +
            norm_opening_sell_strength * micro_sell_power_weights.get('opening_sell_strength', 0.1) +
            norm_pre_closing_sell_posture * micro_sell_power_weights.get('pre_closing_sell_posture', 0.1) +
            norm_closing_auction_sell_ambush * micro_sell_power_weights.get('closing_auction_sell_ambush', 0.1) +
            norm_ask_side_liquidity * micro_sell_power_weights.get('ask_side_liquidity', 0.05) +
            norm_main_force_on_peak_sell_flow * micro_sell_power_weights.get('main_force_on_peak_sell_flow', 0.05) +
            norm_main_force_t0_sell_efficiency * micro_sell_power_weights.get('main_force_t0_sell_efficiency', 0.05) +
            norm_sell_flow_efficiency_index * micro_sell_power_weights.get('sell_flow_efficiency_index', 0.05) +
            norm_sell_order_book_clearing_rate * micro_sell_power_weights.get('sell_order_book_clearing_rate', 0.05) +
            norm_vwap_sell_control_strength * micro_sell_power_weights.get('vwap_sell_control_strength', 0.05) +
            norm_main_force_vwap_down_guidance * micro_sell_power_weights.get('main_force_vwap_down_guidance', 0.05) +
            norm_vwap_cross_down_intensity * micro_sell_power_weights.get('vwap_cross_down_intensity', 0.05) +
            norm_wash_trade_sell_volume * micro_sell_power_weights.get('wash_trade_sell_volume', -0.05)
        ).clip(0, 1)
        micro_control_score_v5_1 = (total_buy_power - total_sell_power).clip(-1, 1)
        micro_intent_score = (
            imbalance_score * micro_intent_fusion_weights.get('imbalance', 0.4) * (1 + micro_imbalance_cohesion * mtf_cohesion_micro_control_weights.get('order_book_imbalance', 0.5) * mtf_cohesion_modulator_sensitivity) +
            impact_score * micro_intent_fusion_weights.get('efficiency', 0.3) * (1 + micro_efficiency_cohesion * mtf_cohesion_micro_control_weights.get('microstructure_efficiency', 0.5) * mtf_cohesion_modulator_sensitivity) +
            exhaustion_score * micro_intent_fusion_weights.get('exhaustion', 0.3) +
            micro_control_score_v5_1 * 0.5 +
            micro_control_quality_score * 0.5
        ).clip(-1, 1)
        micro_control_modulator = pd.Series(1.0, index=df_index)
        if asymmetric_micro_control_enabled:
            boost_mask = (norm_buy_exhaustion > 0.5) & (norm_sell_exhaustion > 0.5)
            micro_control_modulator.loc[boost_mask] = 1 + (norm_buy_exhaustion.loc[boost_mask] * norm_sell_exhaustion.loc[boost_mask]) * exhaustion_boost_factor
            penalty_mask = (norm_buy_exhaustion < 0.5) & (norm_sell_exhaustion < 0.5)
            micro_control_modulator.loc[penalty_mask] = 1 - (norm_buy_exhaustion.loc[penalty_mask] * norm_sell_exhaustion.loc[penalty_mask]) * exhaustion_penalty_factor
            micro_control_modulator = micro_control_modulator.clip(0.5, 1.5)
        micro_control_score = micro_intent_score * micro_control_modulator
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 微观盘口意图推断计算 ---"] = ""
            debug_output[f"        订单簿不平衡归一化: {imbalance_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        OFI冲击归一化: {impact_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        买方枯竭归一化: {norm_buy_exhaustion.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        卖方枯竭归一化: {norm_sell_exhaustion.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        枯竭得分: {exhaustion_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        市场冲击成本反向归一化: {norm_market_impact_cost_inverted.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        流动性斜率归一化: {norm_liquidity_slope.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        流动性真实性归一化: {norm_liquidity_authenticity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        买方扫单强度归一化: {norm_buy_sweep_intensity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        卖方扫单强度反向归一化: {norm_sell_sweep_intensity_inverted.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        订单流不平衡归一化: {norm_order_flow_imbalance_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观控制质量分数: {micro_control_quality_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        总买方力量: {total_buy_power.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        总卖方力量: {total_sell_power.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观控制分数V5.1: {micro_control_score_v5_1.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观意图分数 (融合后): {micro_intent_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观控制调制器: {micro_control_modulator.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观控制分数 (调制后): {micro_control_score.loc[probe_ts]:.4f}"] = ""
        # --- 3. 诡道博弈深度情境感知与调制 (已分离为独立信号 SCORE_FF_DECEPTION_RISK) ---
        score_ff_deception_risk = self._diagnose_deception_risk(df, debug_info_tuple)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道风险信号获取 ---"] = ""
            debug_output[f"        资金流诡道风险 (SCORE_FF_DECEPTION_RISK): {score_ff_deception_risk.loc[probe_ts]:.4f}"] = ""
        # --- 4. 多维度情境自适应权重 (Enhanced Adaptive Macro-Micro Weighting) ---
        dynamic_macro_weight = pd.Series(macro_flow_base_weight, index=df_index)
        dynamic_micro_weight = pd.Series(micro_control_base_weight, index=df_index)
        if dynamic_weight_mod_enabled:
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_flow_slope = get_adaptive_mtf_normalized_bipolar_score(flow_slope_raw, df_index, tf_weights=tf_weights_ff)
            norm_market_sentiment_dw = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_structural_tension = get_adaptive_mtf_normalized_score(structural_tension_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) + \
                         (norm_flow_slope.abs() * dynamic_weight_sensitivity_flow_slope * np.sign(norm_flow_slope)) + \
                         (norm_market_sentiment_dw * dynamic_weight_sensitivity_sentiment) + \
                         (norm_trend_vitality * dynamic_weight_sensitivity_trend_vitality) + \
                         (norm_structural_tension * dynamic_weight_sensitivity_structural_tension)
            dynamic_macro_weight = dynamic_macro_weight * (1 + mod_factor)
            dynamic_micro_weight = dynamic_micro_weight * (1 - mod_factor)
            sum_dynamic_weights = dynamic_macro_weight + dynamic_micro_weight
            sum_dynamic_weights = sum_dynamic_weights.replace(0, 1e-9) # Avoid division by zero
            dynamic_macro_weight = dynamic_macro_weight / sum_dynamic_weights
            dynamic_micro_weight = dynamic_micro_weight / sum_dynamic_weights
            dynamic_macro_weight = dynamic_macro_weight.clip(0.1, 0.9)
            dynamic_micro_weight = dynamic_micro_weight.clip(0.1, 0.9)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态权重自适应计算 ---"] = ""
            debug_output[f"        波动不稳定性归一化: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        资金流斜率归一化: {norm_flow_slope.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        市场情绪归一化 (动态权重): {norm_market_sentiment_dw.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        趋势活力归一化: {norm_trend_vitality.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        结构张力归一化: {norm_structural_tension.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        调制因子: {mod_factor.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态宏观权重: {dynamic_macro_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态微观权重: {dynamic_micro_weight.loc[probe_ts]:.4f}"] = ""
        # --- 5. 融合基础战场控制权 (V6.0 非线性微观-宏观交互) ---
        macro_score_unipolar = (macro_flow_quality_score + 1) / 2
        micro_score_unipolar = (micro_control_score + 1) / 2
        macro_score_unipolar_safe = macro_score_unipolar.clip(lower=1e-9)
        micro_score_unipolar_safe = micro_score_unipolar.clip(lower=1e-9)
        weighted_log_sum = (
            np.log(macro_score_unipolar_safe) * dynamic_macro_weight +
            np.log(micro_score_unipolar_safe) * dynamic_micro_weight
        )
        total_effective_weight = dynamic_macro_weight + dynamic_micro_weight
        total_effective_weight_safe = total_effective_weight.replace(0, 1e-9)
        geometric_mean_unipolar = np.exp(weighted_log_sum / total_effective_weight_safe)
        base_battlefield_control_score = (geometric_mean_unipolar * 2 - 1).clip(-1, 1)
        # 应用诡道风险作为负向调制器
        deception_penalty_factor = (1 - score_ff_deception_risk)
        base_battlefield_control_score = base_battlefield_control_score * deception_penalty_factor
        # 应用非线性交互指数
        base_battlefield_control_score = np.tanh(base_battlefield_control_score * micro_macro_interaction_exponent)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础战场控制权融合 ---"] = ""
            debug_output[f"        宏观流向质量分数 (单极): {macro_score_unipolar.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观控制分数 (单极): {micro_score_unipolar.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加权对数和: {weighted_log_sum.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        总有效权重: {total_effective_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        几何平均 (单极): {geometric_mean_unipolar.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        基础战场控制分数 (融合前): {base_battlefield_control_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        诡道风险惩罚因子: {deception_penalty_factor.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        基础战场控制分数 (诡道风险调制后): {base_battlefield_control_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        非线性交互指数: {micro_macro_interaction_exponent:.4f}"] = ""
            debug_output[f"        基础战场控制分数 (最终): {base_battlefield_control_score.loc[probe_ts]:.4f}"] = ""
        # --- 6. 战场控制权动态演化与前瞻性增强 (Dynamic Evolution & Foresight Enhancement) ---
        smoothed_base_score = base_battlefield_control_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_name], df_index, ascending=False, tf_weights=tf_weights_ff)
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.2) * (1 + norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        dynamic_base_weight = dynamic_evolution_base_weights.get('base_score', 0.6) * (1 - norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        total_dynamic_weights = dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1e-9)
        dynamic_base_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score_components = {
            'base_score': (base_battlefield_control_score + 1) / 2,
            'velocity': (norm_velocity + 1) / 2,
            'acceleration': (norm_acceleration + 1) / 2
        }
        final_score_weights = {
            'base_score': dynamic_base_weight,
            'velocity': dynamic_velocity_weight,
            'acceleration': dynamic_acceleration_weight
        }
        final_score_unipolar = _robust_geometric_mean(final_score_components, final_score_weights, df_index)
        final_score = (final_score_unipolar * 2 - 1).clip(-1, 1)
        if is_debug_enabled_for_method and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态演化与前瞻性增强 ---"] = ""
            debug_output[f"        平滑基础得分: {smoothed_base_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        速度: {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加速度: {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度: {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度: {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态演化上下文归一化: {norm_dynamic_evolution_context.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态基础权重: {dynamic_base_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态速度权重: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态加速度权重: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        最终战场控制分数: {final_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 战场控制权诊断完成。"] = ""
            self._print_debug_output(debug_output) # 统一输出调试信息
        return final_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.4 · 效率优化与MTF共振强化版 & 调试信息统一输出版】资金流公理二：诊断“信念韧性”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 核心升级: 引入多时间框架共振因子，评估关键信念信号在不同时间框架下的协同或背离，作为重要的调制器。
        - 核心细化: 将资金流效率从 `flow_efficiency_index_D` 细化为 `buy_flow_efficiency_index_D` 和 `sell_flow_efficiency_index_D`。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_conviction"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断信念韧性..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ac_params = get_param_value(p_conf_ff.get('axiom_conviction_params'), {})
        core_conviction_weights = get_param_value(ac_params.get('core_conviction_weights'), {'main_force_conviction_slope_5': 0.2, 'main_force_conviction_slope_13': 0.2, 'main_force_conviction_slope_21': 0.1, 'smart_money_net_buy_slope_5': 0.15, 'smart_money_net_buy_slope_13': 0.15, 'flow_credibility': 0.1, 'intraday_large_order_flow': 0.1})
        deceptive_resilience_mod_enabled = get_param_value(ac_params.get('deceptive_resilience_mod_enabled'), True)
        deception_penalty_factor = get_param_value(ac_params.get('deception_penalty_factor'), 0.5)
        wash_trade_penalty_factor = get_param_value(ac_params.get('wash_trade_penalty_factor'), 0.3)
        resilience_context_modulator_signal_1_name = get_param_value(ac_params.get('resilience_context_modulator_signal_1'), 'market_sentiment_score_D')
        resilience_context_modulator_signal_2_name = get_param_value(ac_params.get('resilience_context_modulator_signal_2'), 'main_force_cost_advantage_D')
        resilience_context_modulator_signal_3_name = get_param_value(ac_params.get('resilience_context_modulator_signal_3'), 'order_book_liquidity_supply_D')
        resilience_context_sensitivity_sentiment = get_param_value(ac_params.get('resilience_context_sensitivity_sentiment'), 0.3)
        resilience_context_sensitivity_cost_advantage = get_param_value(ac_params.get('resilience_context_sensitivity_cost_advantage'), 0.2)
        resilience_context_sensitivity_liquidity = get_param_value(ac_params.get('resilience_context_sensitivity_liquidity'), 0.2)
        deception_slope_weights = get_param_value(ac_params.get('deception_slope_weights'), {'slope_5': 0.5, 'slope_13': 0.3, 'slope_21': 0.2})
        wash_trade_slope_weights = get_param_value(ac_params.get('wash_trade_slope_weights'), {'slope_5': 0.5, 'slope_13': 0.3, 'slope_21': 0.2})
        conviction_feedback_sensitivity = get_param_value(ac_params.get('conviction_feedback_sensitivity'), 0.2)
        transmission_efficiency_weights = get_param_value(ac_params.get('transmission_efficiency_weights'), {
            'main_force_execution_alpha_slope_5': 0.2, 'main_force_execution_alpha_slope_13': 0.1,
            'buy_flow_efficiency_slope_5': 0.15, 'buy_flow_efficiency_slope_13': 0.05,
            'sell_flow_efficiency_slope_5': -0.15, 'sell_flow_efficiency_slope_13': -0.05,
            'intraday_price_impact': 0.2, 'large_order_pressure': 0.1, 'intraday_vwap_deviation': 0.1
        })
        dynamic_weight_mod_enabled = get_param_value(ac_params.get('dynamic_weight_mod_enabled'), True)
        core_conviction_base_weight = get_param_value(ac_params.get('core_conviction_base_weight'), 0.4)
        deceptive_resilience_base_weight = get_param_value(ac_params.get('deceptive_resilience_base_weight'), 0.3)
        transmission_efficiency_base_weight = get_param_value(ac_params.get('transmission_efficiency_base_weight'), 0.3)
        dynamic_weight_modulator_signal_1_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_2'), 'market_sentiment_score_D')
        dynamic_weight_modulator_signal_3_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_3'), 'order_book_liquidity_supply_D')
        dynamic_weight_modulator_signal_4_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_4'), 'trend_vitality_index_D')
        dynamic_weight_sensitivity_volatility = get_param_value(ac_params.get('dynamic_weight_sensitivity_volatility'), 0.4)
        dynamic_weight_sensitivity_sentiment = get_param_value(ac_params.get('dynamic_weight_sensitivity_sentiment'), 0.3)
        dynamic_weight_sensitivity_liquidity = get_param_value(ac_params.get('dynamic_weight_sensitivity_liquidity'), 0.2)
        dynamic_weight_sensitivity_trend_vitality = get_param_value(ac_params.get('dynamic_weight_sensitivity_trend_vitality'), 0.1)
        smoothing_ema_span = get_param_value(ac_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ac_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(ac_params.get('dynamic_evolution_context_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(ac_params.get('dynamic_evolution_context_sensitivity'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(ac_params.get('dynamic_evolution_context_modulator_signal_2'), 'trend_vitality_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(ac_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        core_conviction_weights_v4_1 = get_param_value(ac_params.get('core_conviction_weights_v4_1'), {
            'rally_sell_distribution_intensity': -0.1, 'rally_buy_support_weakness': -0.1,
            'main_force_buy_ofi': 0.1, 'main_force_sell_ofi': -0.1,
            'retail_buy_ofi': -0.05, 'retail_sell_ofi': 0.05,
            'wash_trade_buy_volume': -0.05, 'wash_trade_sell_volume': -0.05
        })
        # MTF共振因子参数
        mtf_cohesion_params = get_param_value(ac_params.get('mtf_cohesion_params'), {})
        mtf_cohesion_enabled = get_param_value(mtf_cohesion_params.get('enabled'), True)
        mtf_cohesion_short_periods = get_param_value(mtf_cohesion_params.get('short_periods'), [5, 13])
        mtf_cohesion_long_periods = get_param_value(mtf_cohesion_params.get('long_periods'), [21, 55])
        mtf_cohesion_modulator_sensitivity = get_param_value(mtf_cohesion_params.get('modulator_sensitivity'), 0.5)
        mtf_cohesion_conviction_weights = get_param_value(mtf_cohesion_params.get('conviction_weights'), {"main_force_conviction": 0.5, "smart_money_net_buy": 0.5})
        mtf_cohesion_deception_weights = get_param_value(mtf_cohesion_params.get('deception_weights'), {"deception_index": 0.5, "wash_trade_intensity": 0.5})
        mtf_cohesion_efficiency_weights = get_param_value(mtf_cohesion_params.get('efficiency_weights'), {
            "main_force_execution_alpha": 0.5,
            "buy_flow_efficiency": 0.25,
            "sell_flow_efficiency": 0.25
        })
        required_signals = [
            'SLOPE_5_main_force_conviction_index_D', 'SLOPE_13_main_force_conviction_index_D', 'SLOPE_21_main_force_conviction_index_D',
            'SLOPE_5_SMART_MONEY_HM_NET_BUY_D', 'SLOPE_13_SMART_MONEY_HM_NET_BUY_D',
            'flow_credibility_index_D',
            'buy_lg_amount_D', 'buy_elg_amount_D', 'sell_lg_amount_D', 'sell_elg_amount_D',
            'peak_exchange_purity_D',
            'SLOPE_5_deception_index_D', 'SLOPE_13_deception_index_D', 'SLOPE_21_deception_index_D',
            'SLOPE_5_wash_trade_intensity_D', 'SLOPE_13_wash_trade_intensity_D', 'SLOPE_21_wash_trade_intensity_D',
            'main_force_cost_advantage_D', resilience_context_modulator_signal_1_name, resilience_context_modulator_signal_3_name,
            'SLOPE_5_main_force_execution_alpha_D', 'SLOPE_13_main_force_execution_alpha_D',
            'SLOPE_5_buy_flow_efficiency_index_D', 'SLOPE_13_buy_flow_efficiency_index_D',
            'SLOPE_5_sell_flow_efficiency_index_D', 'SLOPE_13_sell_flow_efficiency_index_D',
            'micro_price_impact_asymmetry_D', 'large_order_pressure_D', 'intraday_vwap_div_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name, dynamic_weight_modulator_signal_3_name, dynamic_weight_modulator_signal_4_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name,
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D',
            'SLOPE_55_main_force_conviction_index_D', 'ACCEL_55_main_force_conviction_index_D',
            'SLOPE_55_SMART_MONEY_HM_NET_BUY_D', 'ACCEL_55_SMART_MONEY_HM_NET_BUY_D',
            'SLOPE_55_deception_index_D', 'ACCEL_55_deception_index_D',
            'SLOPE_55_wash_trade_intensity_D', 'ACCEL_55_wash_trade_intensity_D',
            'SLOPE_55_main_force_execution_alpha_D', 'ACCEL_55_main_force_execution_alpha_D',
            'SLOPE_55_buy_flow_efficiency_index_D', 'ACCEL_55_buy_flow_efficiency_index_D',
            'SLOPE_55_sell_flow_efficiency_index_D', 'ACCEL_55_sell_flow_efficiency_index_D'
        ]
        required_signals = list(set(required_signals))
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        slope_periods = [5, 13, 21, 55]
        accel_periods = [5, 13, 21, 55]
        all_mtf_periods = list(set(slope_periods + accel_periods + mtf_cohesion_short_periods + mtf_cohesion_long_periods))
        signal_bases_to_prefetch_slope = [
            'main_force_conviction_index_D', 'SMART_MONEY_HM_NET_BUY_D', 'deception_index_D',
            'wash_trade_intensity_D', 'main_force_execution_alpha_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D'
        ]
        for signal_base in signal_bases_to_prefetch_slope:
            for p in all_mtf_periods:
                all_pre_fetched_slopes_accels[f'SLOPE_{p}_{signal_base}'] = self._get_safe_series(df, df, f'SLOPE_{p}_{signal_base}', 0.0, method_name=method_name)
        signal_bases_to_prefetch_accel = [
            'main_force_conviction_index_D', 'SMART_MONEY_HM_NET_BUY_D', 'deception_index_D',
            'wash_trade_intensity_D', 'main_force_execution_alpha_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D'
        ]
        for signal_base in signal_bases_to_prefetch_accel:
            for p in all_mtf_periods:
                all_pre_fetched_slopes_accels[f'ACCEL_{p}_{signal_base}'] = self._get_safe_series(df, df, f'ACCEL_{p}_{signal_base}', 0.0, method_name=method_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name in all_pre_fetched_slopes_accels:
                raw_data_cache[signal_name] = all_pre_fetched_slopes_accels[signal_name]
            else:
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        mf_conviction_slope_5_raw = raw_data_cache.get('SLOPE_5_main_force_conviction_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_main_force_conviction_index_D'))
        mf_conviction_slope_13_raw = raw_data_cache.get('SLOPE_13_main_force_conviction_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_main_force_conviction_index_D'))
        mf_conviction_slope_21_raw = raw_data_cache.get('SLOPE_21_main_force_conviction_index_D', all_pre_fetched_slopes_accels.get('SLOPE_21_main_force_conviction_index_D'))
        sm_net_buy_slope_5_raw = raw_data_cache.get('SLOPE_5_SMART_MONEY_HM_NET_BUY_D', all_pre_fetched_slopes_accels.get('SLOPE_5_SMART_MONEY_HM_NET_BUY_D'))
        sm_net_buy_slope_13_raw = raw_data_cache.get('SLOPE_13_SMART_MONEY_HM_NET_BUY_D', all_pre_fetched_slopes_accels.get('SLOPE_13_SMART_MONEY_HM_NET_BUY_D'))
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        buy_lg_amount_raw = raw_data_cache['buy_lg_amount_D']
        buy_elg_amount_raw = raw_data_cache['buy_elg_amount_D']
        sell_lg_amount_raw = raw_data_cache['sell_lg_amount_D']
        sell_elg_amount_raw = raw_data_cache['sell_elg_amount_D']
        intraday_large_order_flow_synthesized = (buy_lg_amount_raw + buy_elg_amount_raw) - (sell_lg_amount_raw + sell_elg_amount_raw)
        main_force_flow_purity_raw = raw_data_cache['peak_exchange_purity_D']
        deception_slope_5_raw = raw_data_cache.get('SLOPE_5_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_deception_index_D'))
        deception_slope_13_raw = raw_data_cache.get('SLOPE_13_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_deception_index_D'))
        deception_slope_21_raw = raw_data_cache.get('SLOPE_21_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_21_deception_index_D'))
        wash_trade_slope_5_raw = raw_data_cache.get('SLOPE_5_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_5_wash_trade_intensity_D'))
        wash_trade_slope_13_raw = raw_data_cache.get('SLOPE_13_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_13_wash_trade_intensity_D'))
        wash_trade_slope_21_raw = raw_data_cache.get('SLOPE_21_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_21_wash_trade_intensity_D'))
        main_force_cost_advantage_raw = raw_data_cache[resilience_context_modulator_signal_2_name]
        market_sentiment_raw = raw_data_cache[resilience_context_modulator_signal_1_name]
        market_liquidity_raw = raw_data_cache[resilience_context_modulator_signal_3_name]
        mf_exec_alpha_slope_5_raw = raw_data_cache.get('SLOPE_5_main_force_execution_alpha_D', all_pre_fetched_slopes_accels.get('SLOPE_5_main_force_execution_alpha_D'))
        mf_exec_alpha_slope_13_raw = raw_data_cache.get('SLOPE_13_main_force_execution_alpha_D', all_pre_fetched_slopes_accels.get('SLOPE_13_main_force_execution_alpha_D'))
        buy_flow_efficiency_slope_5_raw = raw_data_cache.get('SLOPE_5_buy_flow_efficiency_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_buy_flow_efficiency_index_D'))
        buy_flow_efficiency_slope_13_raw = raw_data_cache.get('SLOPE_13_buy_flow_efficiency_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_buy_flow_efficiency_index_D'))
        sell_flow_efficiency_slope_5_raw = raw_data_cache.get('SLOPE_5_sell_flow_efficiency_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_sell_flow_efficiency_index_D'))
        sell_flow_efficiency_slope_13_raw = raw_data_cache.get('SLOPE_13_sell_flow_efficiency_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_sell_flow_efficiency_index_D'))
        intraday_price_impact_raw = raw_data_cache['micro_price_impact_asymmetry_D']
        large_order_pressure_raw = raw_data_cache['large_order_pressure_D']
        intraday_vwap_deviation_raw = raw_data_cache['intraday_vwap_div_index_D']
        volatility_instability_raw = raw_data_cache[dynamic_weight_modulator_signal_1_name]
        market_sentiment_dw_raw = raw_data_cache[dynamic_weight_modulator_signal_2_name]
        market_liquidity_dw_raw = raw_data_cache[dynamic_weight_modulator_signal_3_name]
        trend_vitality_dw_raw = raw_data_cache[dynamic_weight_modulator_signal_4_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        rally_sell_distribution_intensity_raw = raw_data_cache['rally_sell_distribution_intensity_D']
        rally_buy_support_weakness_raw = raw_data_cache['rally_buy_support_weakness_D']
        main_force_buy_ofi_raw = raw_data_cache['main_force_buy_ofi_D']
        main_force_sell_ofi_raw = raw_data_cache['main_force_sell_ofi_D']
        retail_buy_ofi_raw = raw_data_cache['retail_buy_ofi_D']
        retail_sell_ofi_raw = raw_data_cache['retail_sell_ofi_D']
        wash_trade_buy_volume_raw = raw_data_cache['wash_trade_buy_volume_D']
        wash_trade_sell_volume_raw = raw_data_cache['wash_trade_sell_volume_D']
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        # --- MTF共振因子计算 ---
        mf_conviction_cohesion = pd.Series(1.0, index=df_index)
        sm_net_buy_cohesion = pd.Series(1.0, index=df_index)
        deception_cohesion = pd.Series(1.0, index=df_index)
        wash_trade_cohesion = pd.Series(1.0, index=df_index)
        mf_exec_alpha_cohesion = pd.Series(1.0, index=df_index)
        buy_flow_efficiency_cohesion = pd.Series(1.0, index=df_index)
        sell_flow_efficiency_cohesion = pd.Series(1.0, index=df_index)
        if mtf_cohesion_enabled:
            mf_conviction_cohesion = self._calculate_mtf_cohesion_divergence(df, 'main_force_conviction_index_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            sm_net_buy_cohesion = self._calculate_mtf_cohesion_divergence(df, 'SMART_MONEY_HM_NET_BUY_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            deception_cohesion = self._calculate_mtf_cohesion_divergence(df, 'deception_index_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            wash_trade_cohesion = self._calculate_mtf_cohesion_divergence(df, 'wash_trade_intensity_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            mf_exec_alpha_cohesion = self._calculate_mtf_cohesion_divergence(df, 'main_force_execution_alpha_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            buy_flow_efficiency_cohesion = self._calculate_mtf_cohesion_divergence(df, 'buy_flow_efficiency_index_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            sell_flow_efficiency_cohesion = self._calculate_mtf_cohesion_divergence(df, 'sell_flow_efficiency_index_D', mtf_cohesion_short_periods, mtf_cohesion_long_periods, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- MTF共振因子计算 ---"] = ""
                debug_output[f"        mf_conviction_cohesion: {mf_conviction_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        sm_net_buy_cohesion: {sm_net_buy_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        deception_cohesion: {deception_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        wash_trade_cohesion: {wash_trade_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        mf_exec_alpha_cohesion: {mf_exec_alpha_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        buy_flow_efficiency_cohesion: {buy_flow_efficiency_cohesion.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        sell_flow_efficiency_cohesion: {sell_flow_efficiency_cohesion.loc[probe_ts]:.4f}"] = ""
        # --- 1. 核心信念强度 (Core Conviction Strength) ---
        norm_mf_conviction_slope_5 = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slope_5_raw, df_index, tf_weights_ff)
        norm_mf_conviction_slope_13 = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slope_13_raw, df_index, tf_weights_ff)
        norm_mf_conviction_slope_21 = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slope_21_raw, df_index, tf_weights_ff)
        norm_sm_net_buy_slope_5 = get_adaptive_mtf_normalized_bipolar_score(sm_net_buy_slope_5_raw, df_index, tf_weights_ff)
        norm_sm_net_buy_slope_13 = get_adaptive_mtf_normalized_bipolar_score(sm_net_buy_slope_13_raw, df_index, tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_intraday_large_order_flow = get_adaptive_mtf_normalized_bipolar_score(intraday_large_order_flow_synthesized, df_index, tf_weights=tf_weights_ff)
        norm_main_force_flow_purity = get_adaptive_mtf_normalized_score(main_force_flow_purity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        core_conviction_score = (
            norm_mf_conviction_slope_5 * core_conviction_weights.get('main_force_conviction_slope_5', 0.2) * (1 + mf_conviction_cohesion * mtf_cohesion_conviction_weights.get('main_force_conviction', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_mf_conviction_slope_13 * core_conviction_weights.get('main_force_conviction_slope_13', 0.2) * (1 + mf_conviction_cohesion * mtf_cohesion_conviction_weights.get('main_force_conviction', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_mf_conviction_slope_21 * core_conviction_weights.get('main_force_conviction_slope_21', 0.1) * (1 + mf_conviction_cohesion * mtf_cohesion_conviction_weights.get('main_force_conviction', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_sm_net_buy_slope_5 * core_conviction_weights.get('smart_money_net_buy_slope_5', 0.15) * (1 + sm_net_buy_cohesion * mtf_cohesion_conviction_weights.get('smart_money_net_buy', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_sm_net_buy_slope_13 * core_conviction_weights.get('smart_money_net_buy_slope_13', 0.15) * (1 + sm_net_buy_cohesion * mtf_cohesion_conviction_weights.get('smart_money_net_buy', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_flow_credibility * core_conviction_weights.get('flow_credibility', 0.1) +
            norm_intraday_large_order_flow * core_conviction_weights.get('intraday_large_order_flow', 0.1)
        ).clip(-1, 1)
        norm_rally_sell_distribution_intensity = get_adaptive_mtf_normalized_score(raw_data_cache['rally_sell_distribution_intensity_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_rally_buy_support_weakness = get_adaptive_mtf_normalized_score(raw_data_cache['rally_buy_support_weakness_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_buy_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_sell_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['retail_buy_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['retail_sell_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(raw_data_cache['wash_trade_buy_volume_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(raw_data_cache['wash_trade_sell_volume_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        core_conviction_score = core_conviction_score + \
                                (norm_main_force_buy_ofi * core_conviction_weights_v4_1.get('main_force_buy_ofi', 0.1)) + \
                                (norm_retail_sell_ofi * core_conviction_weights_v4_1.get('retail_sell_ofi', 0.05)) - \
                                (norm_rally_sell_distribution_intensity * core_conviction_weights_v4_1.get('rally_sell_distribution_intensity', -0.1)) - \
                                (norm_rally_buy_support_weakness * core_conviction_weights_v4_1.get('rally_buy_support_weakness', -0.1)) - \
                                (norm_main_force_sell_ofi * core_conviction_weights_v4_1.get('main_force_sell_ofi', -0.1)) - \
                                (norm_retail_buy_ofi * core_conviction_weights_v4_1.get('retail_buy_ofi', -0.05)) - \
                                (norm_wash_trade_buy_volume * core_conviction_weights_v4_1.get('wash_trade_buy_volume', -0.05)) - \
                                (norm_wash_trade_sell_volume * core_conviction_weights_v4_1.get('wash_trade_sell_volume', -0.05))
        core_conviction_score = core_conviction_score.clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 核心信念强度 ---"] = ""
            debug_output[f"        norm_mf_conviction_slope_5: {norm_mf_conviction_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_mf_conviction_slope_13: {norm_mf_conviction_slope_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_mf_conviction_slope_21: {norm_mf_conviction_slope_21.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sm_net_buy_slope_5: {norm_sm_net_buy_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sm_net_buy_slope_13: {norm_sm_net_buy_slope_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_intraday_large_order_flow: {norm_intraday_large_order_flow.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_flow_purity: {norm_main_force_flow_purity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_rally_sell_distribution_intensity: {norm_rally_sell_distribution_intensity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_rally_buy_support_weakness: {norm_rally_buy_support_weakness.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_buy_ofi: {norm_main_force_buy_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_sell_ofi: {norm_main_force_sell_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_buy_ofi: {norm_retail_buy_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_sell_ofi: {norm_retail_sell_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_wash_trade_buy_volume: {norm_wash_trade_buy_volume.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_wash_trade_sell_volume: {norm_wash_trade_sell_volume.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        核心信念分数 (core_conviction_score): {core_conviction_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 诡道博弈韧性调制 (Deceptive Resilience Modulation) ---
        deceptive_resilience_modulator = pd.Series(1.0, index=df_index)
        if deceptive_resilience_mod_enabled:
            norm_deception_slope_5 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_5_raw, df_index, tf_weights_ff)
            norm_deception_slope_13 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_13_raw, df_index, tf_weights_ff)
            norm_deception_slope_21 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_21_raw, df_index, tf_weights_ff)
            norm_deception_multi_tf = (
                norm_deception_slope_5 * deception_slope_weights.get('slope_5', 0.5) +
                norm_deception_slope_13 * deception_slope_weights.get('slope_13', 0.3) +
                norm_deception_slope_21 * deception_slope_weights.get('slope_21', 0.2)
            ).clip(-1, 1)
            norm_wash_trade_slope_5 = get_adaptive_mtf_normalized_score(wash_trade_slope_5_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_wash_trade_slope_13 = get_adaptive_mtf_normalized_score(wash_trade_slope_13_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_wash_trade_slope_21 = get_adaptive_mtf_normalized_score(wash_trade_slope_21_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_wash_trade_multi_tf = (
                norm_wash_trade_slope_5 * wash_trade_slope_weights.get('slope_5', 0.5) +
                norm_wash_trade_slope_13 * wash_trade_slope_weights.get('slope_13', 0.3) +
                norm_wash_trade_slope_21 * wash_trade_slope_weights.get('slope_21', 0.2)
            ).clip(0, 1)
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            norm_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights_ff)
            norm_market_liquidity = get_adaptive_mtf_normalized_score(market_liquidity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            sentiment_mod = (1 + norm_market_sentiment.abs() * resilience_context_sensitivity_sentiment * np.sign(norm_market_sentiment))
            cost_advantage_mod = (1 + norm_cost_advantage.abs() * resilience_context_sensitivity_cost_advantage * np.sign(norm_cost_advantage))
            liquidity_mod = (1 + (norm_market_liquidity - 0.5) * resilience_context_sensitivity_liquidity)
            conviction_feedback_mod = (1 - core_conviction_score.abs() * conviction_feedback_sensitivity * np.sign(core_conviction_score))
            deception_cohesion_mod = (1 + deception_cohesion * mtf_cohesion_deception_weights.get('deception_index', 0.5) * mtf_cohesion_modulator_sensitivity)
            wash_trade_cohesion_mod = (1 + wash_trade_cohesion * mtf_cohesion_deception_weights.get('wash_trade_intensity', 0.5) * mtf_cohesion_modulator_sensitivity)
            deceptive_resilience_modulator = deceptive_resilience_modulator * (1 - norm_wash_trade_multi_tf * wash_trade_penalty_factor * conviction_feedback_mod.clip(0.5, 1.5) * sentiment_mod.clip(0.5, 1.5) * liquidity_mod.clip(0.5, 1.5) * wash_trade_cohesion_mod)
            bull_trap_mask = (norm_deception_multi_tf > 0)
            deceptive_resilience_modulator.loc[bull_trap_mask] = deceptive_resilience_modulator.loc[bull_trap_mask] * (1 - norm_deception_multi_tf.loc[bull_trap_mask] * deception_penalty_factor * conviction_feedback_mod.loc[bull_trap_mask].clip(0.5, 1.5) * sentiment_mod.loc[bull_trap_mask].clip(0.5, 1.5) * liquidity_mod.loc[bull_trap_mask].clip(0.5, 1.5) * deception_cohesion_mod.loc[bull_trap_mask])
            bear_trap_resilience_mask = (norm_deception_multi_tf < 0) & (norm_cost_advantage > 0.5) & (norm_market_sentiment < -0.5) & (norm_market_liquidity < 0.5) & (core_conviction_score > 0.2)
            deceptive_resilience_modulator.loc[bear_trap_resilience_mask] = deceptive_resilience_modulator.loc[bear_trap_resilience_mask] * (1 + norm_deception_multi_tf.loc[bear_trap_resilience_mask].abs() * deception_penalty_factor * cost_advantage_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5) * (1 - liquidity_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5)) * deception_cohesion_mod.loc[bear_trap_resilience_mask])
            deceptive_resilience_modulator = deceptive_resilience_modulator.clip(0.01, 2.0)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道博弈韧性调制 ---"] = ""
                debug_output[f"        norm_deception_multi_tf: {norm_deception_multi_tf.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_wash_trade_multi_tf: {norm_wash_trade_multi_tf.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_market_sentiment: {norm_market_sentiment.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_cost_advantage: {norm_cost_advantage.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_market_liquidity: {norm_market_liquidity.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        sentiment_mod: {sentiment_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        cost_advantage_mod: {cost_advantage_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        liquidity_mod: {liquidity_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        conviction_feedback_mod: {conviction_feedback_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        deception_cohesion_mod: {deception_cohesion_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        wash_trade_cohesion_mod: {wash_trade_cohesion_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        诡道韧性调制器 (deceptive_resilience_modulator): {deceptive_resilience_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 3. 信念传导效率 (Conviction Transmission Efficiency) ---
        norm_mf_exec_alpha_slope_5 = get_adaptive_mtf_normalized_bipolar_score(mf_exec_alpha_slope_5_raw, df_index, tf_weights_ff)
        norm_mf_exec_alpha_slope_13 = get_adaptive_mtf_normalized_bipolar_score(mf_exec_alpha_slope_13_raw, df_index, tf_weights_ff)
        norm_buy_flow_efficiency_slope_5 = get_adaptive_mtf_normalized_bipolar_score(buy_flow_efficiency_slope_5_raw, df_index, tf_weights_ff)
        norm_buy_flow_efficiency_slope_13 = get_adaptive_mtf_normalized_bipolar_score(buy_flow_efficiency_slope_13_raw, df_index, tf_weights_ff)
        norm_sell_flow_efficiency_slope_5 = get_adaptive_mtf_normalized_bipolar_score(sell_flow_efficiency_slope_5_raw, df_index, tf_weights_ff)
        norm_sell_flow_efficiency_slope_13 = get_adaptive_mtf_normalized_bipolar_score(sell_flow_efficiency_slope_13_raw, df_index, tf_weights_ff)
        norm_intraday_price_impact = get_adaptive_mtf_normalized_bipolar_score(intraday_price_impact_raw, df_index, tf_weights=tf_weights_ff)
        norm_large_order_pressure = get_adaptive_mtf_normalized_score(large_order_pressure_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_intraday_vwap_deviation = get_adaptive_mtf_normalized_score(intraday_vwap_deviation_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        efficiency_ratio = (norm_intraday_price_impact + (1 - norm_intraday_vwap_deviation)) / (norm_large_order_pressure + 1e-9)
        norm_efficiency_ratio = get_adaptive_mtf_normalized_score(efficiency_ratio, df_index, ascending=True, tf_weights=tf_weights_ff)
        transmission_efficiency_score = (
            norm_mf_exec_alpha_slope_5 * transmission_efficiency_weights.get('main_force_execution_alpha_slope_5', 0.2) * (1 + mf_exec_alpha_cohesion * mtf_cohesion_efficiency_weights.get('main_force_execution_alpha', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_mf_exec_alpha_slope_13 * transmission_efficiency_weights.get('main_force_execution_alpha_slope_13', 0.1) * (1 + mf_exec_alpha_cohesion * mtf_cohesion_efficiency_weights.get('main_force_execution_alpha', 0.5) * mtf_cohesion_modulator_sensitivity) +
            norm_buy_flow_efficiency_slope_5 * transmission_efficiency_weights.get('buy_flow_efficiency_slope_5', 0.15) * (1 + buy_flow_efficiency_cohesion * mtf_cohesion_efficiency_weights.get('buy_flow_efficiency', 0.25) * mtf_cohesion_modulator_sensitivity) +
            norm_buy_flow_efficiency_slope_13 * transmission_efficiency_weights.get('buy_flow_efficiency_slope_13', 0.05) * (1 + buy_flow_efficiency_cohesion * mtf_cohesion_efficiency_weights.get('buy_flow_efficiency', 0.25) * mtf_cohesion_modulator_sensitivity) +
            norm_sell_flow_efficiency_slope_5 * transmission_efficiency_weights.get('sell_flow_efficiency_slope_5', -0.15) * (1 + sell_flow_efficiency_cohesion * mtf_cohesion_efficiency_weights.get('sell_flow_efficiency', 0.25) * mtf_cohesion_modulator_sensitivity) +
            norm_sell_flow_efficiency_slope_13 * transmission_efficiency_weights.get('sell_flow_efficiency_slope_13', -0.05) * (1 + sell_flow_efficiency_cohesion * mtf_cohesion_efficiency_weights.get('sell_flow_efficiency', 0.25) * mtf_cohesion_modulator_sensitivity) +
            norm_intraday_price_impact * transmission_efficiency_weights.get('intraday_price_impact', 0.2) +
            norm_large_order_pressure * transmission_efficiency_weights.get('large_order_pressure', 0.1) +
            norm_intraday_vwap_deviation * transmission_efficiency_weights.get('intraday_vwap_deviation', 0.1)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 信念传导效率 ---"] = ""
            debug_output[f"        norm_mf_exec_alpha_slope_5: {norm_mf_exec_alpha_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_mf_exec_alpha_slope_13: {norm_mf_exec_alpha_slope_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_buy_flow_efficiency_slope_5: {norm_buy_flow_efficiency_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_buy_flow_efficiency_slope_13: {norm_buy_flow_efficiency_slope_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sell_flow_efficiency_slope_5: {norm_sell_flow_efficiency_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sell_flow_efficiency_slope_13: {norm_sell_flow_efficiency_slope_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_intraday_price_impact: {norm_intraday_price_impact.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_large_order_pressure: {norm_large_order_pressure.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_intraday_vwap_deviation: {norm_intraday_vwap_deviation.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        efficiency_ratio: {efficiency_ratio.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_efficiency_ratio: {norm_efficiency_ratio.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        信念传导效率分数 (transmission_efficiency_score): {transmission_efficiency_score.loc[probe_ts]:.4f}"] = ""
        # --- 4. 动态情境自适应权重 (Dynamic Contextual Weighting) ---
        dynamic_core_conviction_weight = pd.Series(core_conviction_base_weight, index=df_index)
        dynamic_deceptive_resilience_weight = pd.Series(deceptive_resilience_base_weight, index=df_index)
        dynamic_transmission_efficiency_weight = pd.Series(transmission_efficiency_base_weight, index=df_index)
        if dynamic_weight_mod_enabled:
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_market_sentiment_dw = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_dw_raw, df_index, tf_weights=tf_weights_ff)
            norm_market_liquidity_dw = get_adaptive_mtf_normalized_score(market_liquidity_dw_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_trend_vitality_dw = get_adaptive_mtf_normalized_score(trend_vitality_dw_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            mod_factor_volatility = (norm_volatility_instability - 0.5) * dynamic_weight_sensitivity_volatility
            mod_factor_sentiment = norm_market_sentiment_dw * dynamic_weight_sensitivity_sentiment
            mod_factor_liquidity = (0.5 - norm_market_liquidity_dw) * dynamic_weight_sensitivity_liquidity
            mod_factor_trend_vitality = (norm_trend_vitality_dw - 0.5) * dynamic_weight_sensitivity_trend_vitality
            dynamic_core_conviction_weight = dynamic_core_conviction_weight * (1 + mod_factor_sentiment - mod_factor_volatility + mod_factor_trend_vitality)
            dynamic_deceptive_resilience_weight = dynamic_deceptive_resilience_weight * (1 + mod_factor_volatility + mod_factor_liquidity - mod_factor_sentiment - mod_factor_trend_vitality)
            dynamic_transmission_efficiency_weight = dynamic_transmission_efficiency_weight * (1 + mod_factor_sentiment - mod_factor_liquidity + mod_factor_trend_vitality)
            sum_dynamic_weights = dynamic_core_conviction_weight + dynamic_deceptive_resilience_weight + dynamic_transmission_efficiency_weight
            sum_dynamic_weights = sum_dynamic_weights.replace(0, 1e-9)
            dynamic_core_conviction_weight = dynamic_core_conviction_weight / sum_dynamic_weights
            dynamic_deceptive_resilience_weight = dynamic_deceptive_resilience_weight / sum_dynamic_weights
            dynamic_transmission_efficiency_weight = dynamic_transmission_efficiency_weight / sum_dynamic_weights
            dynamic_core_conviction_weight = dynamic_core_conviction_weight.clip(0.1, 0.8)
            dynamic_deceptive_resilience_weight = dynamic_deceptive_resilience_weight.clip(0.1, 0.8)
            dynamic_transmission_efficiency_weight = dynamic_transmission_efficiency_weight.clip(0.1, 0.8)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 动态情境自适应权重 ---"] = ""
                debug_output[f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_market_sentiment_dw: {norm_market_sentiment_dw.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_market_liquidity_dw: {norm_market_liquidity_dw.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_trend_vitality_dw: {norm_trend_vitality_dw.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        mod_factor_volatility: {mod_factor_volatility.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        mod_factor_sentiment: {mod_factor_sentiment.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        mod_factor_liquidity: {mod_factor_liquidity.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        mod_factor_trend_vitality: {mod_factor_trend_vitality.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        动态核心信念权重: {dynamic_core_conviction_weight.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        动态诡道韧性权重: {dynamic_deceptive_resilience_weight.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        动态传导效率权重: {dynamic_transmission_efficiency_weight.loc[probe_ts]:.4f}"] = ""
        # --- 5. 融合基础信念分数 (V4.0 非线性建模) ---
        base_conviction_score = np.tanh(
            core_conviction_score * dynamic_core_conviction_weight * deceptive_resilience_modulator +
            transmission_efficiency_score * dynamic_transmission_efficiency_weight
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合基础信念分数 ---"] = ""
            debug_output[f"        基础信念分数 (base_conviction_score): {base_conviction_score.loc[probe_ts]:.4f}"] = ""
        # --- 6. 信念演化趋势与前瞻性增强 (Conviction Evolution & Foresight Enhancement) ---
        smoothed_base_score = base_conviction_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_1_name], df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_2_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1e-9)
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_conviction_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 信念演化趋势与前瞻性增强 ---"] = ""
            debug_output[f"        平滑基础分数 (smoothed_base_score): {smoothed_base_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        速度 (velocity): {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加速度 (acceleration): {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        组合演化上下文调制 (combined_evolution_context_mod): {combined_evolution_context_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态基础权重: {dynamic_base_score_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态速度权重: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态加速度权重: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 信念韧性诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self._print_debug_output(debug_output)
        return final_score.astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V6.4 · 原始数据缓存修复版 & 调试信息统一输出版】资金流公理三：诊断“资金流纯度与动能”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 【修复】修复了 `raw_data_cache` 在调试模式下可能因缺少预取信号而引发 `KeyError` 的问题。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_flow_momentum"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断资金流纯度与动能..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        fm_params = get_param_value(p_conf_ff.get('axiom_flow_momentum_params'), {})
        base_momentum_weights = get_param_value(fm_params.get('base_momentum_weights'), {'nmfnf_slope_5': 0.2, 'nmfnf_slope_13': 0.15, 'nmfnf_slope_21': 0.1, 'nmfnf_slope_55': 0.05, 'nmfnf_accel_5': 0.2, 'nmfnf_accel_13': 0.15, 'nmfnf_accel_21': 0.1, 'nmfnf_accel_55': 0.05})
        smart_money_momentum_weights = get_param_value(fm_params.get('smart_money_momentum_weights'), {'sm_net_buy_slope_5': 0.6, 'sm_net_buy_accel_5': 0.4})
        purity_filter_enabled = get_param_value(fm_params.get('purity_filter_enabled'), True)
        wash_trade_slope_weights = get_param_value(fm_params.get('wash_trade_slope_weights'), {'slope_5': 0.5, 'slope_13': 0.3, 'slope_21': 0.2})
        deception_slope_weights = get_param_value(fm_params.get('deception_slope_weights'), {'slope_5': 0.5, 'slope_13': 0.3, 'slope_21': 0.2})
        purity_context_modulator_signal_1_name = get_param_value(fm_params.get('purity_context_modulator_signal_1'), 'main_force_conviction_index_D')
        purity_context_modulator_signal_2_name = get_param_value(fm_params.get('purity_context_modulator_signal_2'), 'flow_credibility_index_D')
        purity_context_modulator_signal_3_name = get_param_value(fm_params.get('purity_context_modulator_signal_3'), 'main_force_flow_gini_D')
        purity_context_modulator_signal_4_name = get_param_value(fm_params.get('purity_context_modulator_signal_4'), 'retail_fomo_premium_index_D')
        purity_context_sensitivity_conviction = get_param_value(fm_params.get('purity_context_sensitivity_conviction'), 0.3)
        purity_context_sensitivity_credibility = get_param_value(fm_params.get('purity_context_sensitivity_credibility'), 0.2)
        purity_context_sensitivity_gini = get_param_value(fm_params.get('purity_context_sensitivity_gini'), 0.2)
        purity_context_sensitivity_fomo = get_param_value(fm_params.get('purity_context_sensitivity_fomo'), 0.1)
        purity_penalty_factor_wash_trade = get_param_value(fm_params.get('purity_penalty_factor_wash_trade'), 0.5)
        purity_penalty_factor_deception = get_param_value(fm_params.get('purity_penalty_factor_deception'), 0.7)
        purity_mitigation_factor = get_param_value(fm_params.get('purity_mitigation_factor'), 0.2)
        purity_auxiliary_signal_name = get_param_value(fm_params.get('purity_auxiliary_signal'), 'main_force_t0_efficiency_D')
        contextual_modulator_enabled = get_param_value(fm_params.get('contextual_modulator_enabled'), True)
        liquidity_slope_weights = get_param_value(fm_params.get('liquidity_slope_weights'), {'slope_5': 0.6, 'slope_13': 0.4})
        liquidity_impact_signal_name = get_param_value(fm_params.get('liquidity_impact_signal'), 'micro_impact_elasticity_D')
        environment_context_signal_1_name = get_param_value(fm_params.get('environment_context_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        environment_context_signal_2_name = get_param_value(fm_params.get('environment_context_signal_2'), 'trend_vitality_index_D')
        environment_context_signal_3_name = get_param_value(fm_params.get('environment_context_signal_3'), 'price_volume_entropy_D')
        liquidity_mod_sensitivity_level = get_param_value(fm_params.get('liquidity_mod_sensitivity_level'), 0.5)
        liquidity_mod_sensitivity_slope = get_param_value(fm_params.get('liquidity_mod_sensitivity_slope'), 0.3)
        environment_mod_sensitivity_volatility = get_param_value(fm_params.get('environment_mod_sensitivity_volatility'), 0.3)
        environment_mod_sensitivity_trend_vitality = get_param_value(fm_params.get('environment_mod_sensitivity_trend_vitality'), 0.2)
        environment_mod_sensitivity_entropy = get_param_value(fm_params.get('environment_mod_sensitivity_entropy'), 0.1)
        structural_momentum_weights = get_param_value(fm_params.get('structural_momentum_weights'), {'large_order_flow_slope_5': 0.2, 'large_order_flow_accel_5': 0.15, 'main_force_flow_directionality': 0.2, 'main_force_flow_gini': 0.15, 'retail_flow_slope_5': -0.1, 'retail_flow_accel_5': -0.05, 'retail_flow_dominance': -0.15})
        structural_momentum_weights_v6_1 = get_param_value(fm_params.get('structural_momentum_weights_v6_1'), {
            'rally_sell_distribution_intensity': -0.1, 'rally_buy_support_weakness': -0.1,
            'main_force_buy_ofi': 0.1, 'main_force_sell_ofi': -0.1,
            'retail_buy_ofi': -0.05, 'retail_sell_ofi': 0.05,
            'wash_trade_buy_volume': -0.05, 'wash_trade_sell_volume': -0.05
        })
        flow_quality_signal_name = get_param_value(fm_params.get('flow_quality_signal'), 'main_force_flow_gini_D')
        retail_dominance_signal_name = get_param_value(fm_params.get('retail_dominance_signal'), 'retail_flow_dominance_index_D')
        smoothing_ema_span = get_param_value(fm_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(fm_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(fm_params.get('dynamic_evolution_context_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(fm_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(fm_params.get('dynamic_evolution_context_modulator_signal_2'), 'trend_vitality_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(fm_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        dynamic_evolution_context_modulator_signal_3_name = get_param_value(fm_params.get('dynamic_evolution_context_modulator_signal_3'), 'market_sentiment_score_D')
        dynamic_evolution_context_sensitivity_3 = get_param_value(fm_params.get('dynamic_evolution_context_sensitivity_3'), 0.1)
        dynamic_evolution_context_modulator_signal_4_name = get_param_value(fm_params.get('dynamic_evolution_context_modulator_signal_4'), 'flow_credibility_index_D')
        dynamic_evolution_context_sensitivity_4 = get_param_value(fm_params.get('dynamic_evolution_context_sensitivity_4'), 0.05)
        required_signals = [
            'SLOPE_5_NMFNF_D', 'SLOPE_13_NMFNF_D', 'SLOPE_21_NMFNF_D', 'SLOPE_55_NMFNF_D',
            'ACCEL_5_NMFNF_D', 'ACCEL_13_NMFNF_D', 'ACCEL_21_NMFNF_D', 'ACCEL_55_NMFNF_D',
            'SLOPE_5_SMART_MONEY_HM_NET_BUY_D', 'ACCEL_5_SMART_MONEY_HM_NET_BUY_D',
            'SLOPE_5_wash_trade_intensity_D', 'SLOPE_13_wash_trade_intensity_D', 'SLOPE_21_wash_trade_intensity_D',
            'SLOPE_5_deception_index_D', 'SLOPE_13_deception_index_D', 'SLOPE_21_deception_index_D',
            purity_context_modulator_signal_1_name, purity_context_modulator_signal_2_name,
            purity_context_modulator_signal_3_name, purity_context_modulator_signal_4_name,
            purity_auxiliary_signal_name,
            'SLOPE_5_order_book_liquidity_supply_D', 'SLOPE_13_order_book_liquidity_supply_D',
            'order_book_liquidity_supply_D', liquidity_impact_signal_name,
            environment_context_signal_1_name, environment_context_signal_2_name, environment_context_signal_3_name,
            'SLOPE_5_net_lg_amount_calibrated_D', 'ACCEL_5_net_lg_amount_calibrated_D',
            'SLOPE_5_net_xl_amount_calibrated_D', 'ACCEL_5_net_xl_amount_calibrated_D',
            'SLOPE_5_retail_net_flow_calibrated_D', 'ACCEL_5_retail_net_flow_calibrated_D',
            'main_force_flow_directionality_D', flow_quality_signal_name, retail_dominance_signal_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name,
            dynamic_evolution_context_modulator_signal_3_name, dynamic_evolution_context_modulator_signal_4_name,
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        slope_periods_all = list(set([5, 13, 21, 55]))
        accel_periods_all = list(set([5, 13, 21, 55]))
        signal_bases_to_prefetch_slope = [
            'NMFNF_D', 'SMART_MONEY_HM_NET_BUY_D', 'wash_trade_intensity_D', 'deception_index_D',
            'order_book_liquidity_supply_D', 'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D',
            'retail_net_flow_calibrated_D'
        ]
        for signal_base in signal_bases_to_prefetch_slope:
            for p in slope_periods_all:
                col_name = f'SLOPE_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name=method_name)
        signal_bases_to_prefetch_accel = [
            'NMFNF_D', 'SMART_MONEY_HM_NET_BUY_D', 'net_lg_amount_calibrated_D',
            'net_xl_amount_calibrated_D', 'retail_net_flow_calibrated_D'
        ]
        for signal_base in signal_bases_to_prefetch_accel:
            for p in accel_periods_all:
                col_name = f'ACCEL_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name=method_name)
        # --- 原始数据获取 (用于探针和计算) ---
        # 修复：确保 raw_data_cache 包含所有 required_signals，包括预取的斜率和加速度
        raw_data_cache = all_pre_fetched_slopes_accels.copy() # 从预取数据开始
        for signal_name in required_signals:
            if signal_name not in raw_data_cache: # 如果信号不在预取数据中，则按需获取
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        nmfnf_slope_5_raw = raw_data_cache['SLOPE_5_NMFNF_D'] # 现在可以直接访问，因为已确保存在
        nmfnf_slope_13_raw = raw_data_cache['SLOPE_13_NMFNF_D']
        nmfnf_slope_21_raw = raw_data_cache['SLOPE_21_NMFNF_D']
        nmfnf_slope_55_raw = raw_data_cache['SLOPE_55_NMFNF_D']
        nmfnf_accel_5_raw = raw_data_cache['ACCEL_5_NMFNF_D']
        nmfnf_accel_13_raw = raw_data_cache['ACCEL_13_NMFNF_D']
        nmfnf_accel_21_raw = raw_data_cache['ACCEL_21_NMFNF_D']
        nmfnf_accel_55_raw = raw_data_cache['ACCEL_55_NMFNF_D']
        sm_net_buy_slope_5_raw = raw_data_cache['SLOPE_5_SMART_MONEY_HM_NET_BUY_D']
        sm_net_buy_accel_5_raw = raw_data_cache['ACCEL_5_SMART_MONEY_HM_NET_BUY_D']
        wash_trade_slope_5_raw = raw_data_cache['SLOPE_5_wash_trade_intensity_D']
        wash_trade_slope_13_raw = raw_data_cache['SLOPE_13_wash_trade_intensity_D']
        wash_trade_slope_21_raw = raw_data_cache['SLOPE_21_wash_trade_intensity_D']
        deception_slope_5_raw = raw_data_cache['SLOPE_5_deception_index_D']
        deception_slope_13_raw = raw_data_cache['SLOPE_13_deception_index_D']
        deception_slope_21_raw = raw_data_cache['SLOPE_21_deception_index_D']
        main_force_conviction_raw = raw_data_cache[purity_context_modulator_signal_1_name]
        flow_credibility_raw = raw_data_cache[purity_context_modulator_signal_2_name]
        main_force_flow_gini_raw = raw_data_cache[purity_context_modulator_signal_3_name]
        retail_fomo_premium_raw = raw_data_cache[purity_context_modulator_signal_4_name]
        purity_auxiliary_raw = raw_data_cache[purity_auxiliary_signal_name]
        liquidity_supply_raw = raw_data_cache['order_book_liquidity_supply_D']
        liquidity_slope_5_raw = raw_data_cache['SLOPE_5_order_book_liquidity_supply_D']
        liquidity_slope_13_raw = raw_data_cache['SLOPE_13_order_book_liquidity_supply_D']
        liquidity_impact_raw = raw_data_cache[liquidity_impact_signal_name]
        volatility_instability_raw = raw_data_cache[environment_context_signal_1_name]
        trend_vitality_raw = raw_data_cache[environment_context_signal_2_name]
        price_volume_entropy_raw = raw_data_cache[environment_context_signal_3_name]
        lg_flow_slope_5_raw = raw_data_cache['SLOPE_5_net_lg_amount_calibrated_D']
        lg_flow_accel_5_raw = raw_data_cache['ACCEL_5_net_lg_amount_calibrated_D']
        xl_flow_slope_5_raw = raw_data_cache['SLOPE_5_net_xl_amount_calibrated_D']
        xl_flow_accel_5_raw = raw_data_cache['ACCEL_5_net_xl_amount_calibrated_D']
        retail_flow_slope_5_raw = raw_data_cache['SLOPE_5_retail_net_flow_calibrated_D']
        retail_flow_accel_5_raw = raw_data_cache['ACCEL_5_retail_net_flow_calibrated_D']
        main_force_flow_directionality_raw = raw_data_cache['main_force_flow_directionality_D']
        flow_quality_raw = raw_data_cache[flow_quality_signal_name]
        retail_dominance_raw = raw_data_cache[retail_dominance_signal_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        dynamic_evolution_context_modulator_3_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_3_name]
        dynamic_evolution_context_modulator_4_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_4_name]
        rally_sell_distribution_intensity_raw = raw_data_cache['rally_sell_distribution_intensity_D']
        rally_buy_support_weakness_raw = raw_data_cache['rally_buy_support_weakness_D']
        main_force_buy_ofi_raw = raw_data_cache['main_force_buy_ofi_D']
        main_force_sell_ofi_raw = raw_data_cache['main_force_sell_ofi_D']
        retail_buy_ofi_raw = raw_data_cache['retail_buy_ofi_D']
        retail_sell_ofi_raw = raw_data_cache['retail_sell_ofi_D']
        wash_trade_buy_volume_raw = raw_data_cache['wash_trade_buy_volume_D']
        wash_trade_sell_volume_raw = raw_data_cache['wash_trade_sell_volume_D']
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        # --- 1. 基础动能深化 (Enhanced Base Momentum) ---
        norm_nmfnf_slope_5 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_slope_5_raw, df_index, tf_weights_ff)
        norm_nmfnf_slope_13 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_slope_13_raw, df_index, tf_weights_ff)
        norm_nmfnf_slope_21 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_slope_21_raw, df_index, tf_weights_ff)
        norm_nmfnf_slope_55 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_slope_55_raw, df_index, tf_weights_ff)
        norm_nmfnf_accel_5 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_accel_5_raw, df_index, tf_weights_ff)
        norm_nmfnf_accel_13 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_accel_13_raw, df_index, tf_weights_ff)
        norm_nmfnf_accel_21 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_accel_21_raw, df_index, tf_weights_ff)
        norm_nmfnf_accel_55 = get_adaptive_mtf_normalized_bipolar_score(nmfnf_accel_55_raw, df_index, tf_weights_ff)
        norm_sm_net_buy_slope_5 = get_adaptive_mtf_normalized_bipolar_score(sm_net_buy_slope_5_raw, df_index, tf_weights_ff)
        norm_sm_net_buy_accel_5 = get_adaptive_mtf_normalized_bipolar_score(sm_net_buy_accel_5_raw, df_index, tf_weights_ff)
        base_momentum_score = (
            norm_nmfnf_slope_5 * base_momentum_weights.get('nmfnf_slope_5', 0.2) +
            norm_nmfnf_slope_13 * base_momentum_weights.get('nmfnf_slope_13', 0.15) +
            norm_nmfnf_slope_21 * base_momentum_weights.get('nmfnf_slope_21', 0.1) +
            norm_nmfnf_slope_55 * base_momentum_weights.get('nmfnf_slope_55', 0.05) +
            norm_nmfnf_accel_5 * base_momentum_weights.get('nmfnf_accel_5', 0.2) +
            norm_nmfnf_accel_13 * base_momentum_weights.get('nmfnf_accel_13', 0.15) +
            norm_nmfnf_accel_21 * base_momentum_weights.get('nmfnf_accel_21', 0.1) +
            norm_nmfnf_accel_55 * base_momentum_weights.get('nmfnf_accel_55', 0.05) +
            norm_sm_net_buy_slope_5 * smart_money_momentum_weights.get('sm_net_buy_slope_5', 0.6) +
            norm_sm_net_buy_accel_5 * smart_money_momentum_weights.get('sm_net_buy_accel_5', 0.4)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 基础动能深化 ---"] = ""
            debug_output[f"        norm_nmfnf_slope_5: {norm_nmfnf_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_slope_13: {norm_nmfnf_slope_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_slope_21: {norm_nmfnf_slope_21.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_slope_55: {norm_nmfnf_slope_55.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_accel_5: {norm_nmfnf_accel_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_accel_13: {norm_nmfnf_accel_13.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_accel_21: {norm_nmfnf_accel_21.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_nmfnf_accel_55: {norm_nmfnf_accel_55.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sm_net_buy_slope_5: {norm_sm_net_buy_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_sm_net_buy_accel_5: {norm_sm_net_buy_accel_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        基础动能分数 (base_momentum_score): {base_momentum_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 诡道纯度精修 (Deceptive Purity Refinement) ---
        purity_modulator = pd.Series(1.0, index=df_index)
        if purity_filter_enabled:
            norm_wash_trade_slope_5 = get_adaptive_mtf_normalized_score(wash_trade_slope_5_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_wash_trade_slope_13 = get_adaptive_mtf_normalized_score(wash_trade_slope_13_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_wash_trade_slope_21 = get_adaptive_mtf_normalized_score(wash_trade_slope_21_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_wash_trade_multi_tf = (
                norm_wash_trade_slope_5 * wash_trade_slope_weights.get('slope_5', 0.5) +
                norm_wash_trade_slope_13 * wash_trade_slope_weights.get('slope_13', 0.3) +
                norm_wash_trade_slope_21 * wash_trade_slope_weights.get('slope_21', 0.2)
            ).clip(0, 1)
            norm_deception_slope_5 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_5_raw, df_index, tf_weights=tf_weights_ff)
            norm_deception_slope_13 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_13_raw, df_index, tf_weights=tf_weights_ff)
            norm_deception_slope_21 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_21_raw, df_index, tf_weights=tf_weights_ff)
            norm_deception_multi_tf = (
                norm_deception_slope_5 * deception_slope_weights.get('slope_5', 0.5) +
                norm_deception_slope_13 * deception_slope_weights.get('slope_13', 0.3) +
                norm_deception_slope_21 * deception_slope_weights.get('slope_21', 0.2)
            ).clip(-1, 1)
            norm_mf_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_mf_flow_gini_inverted = 1 - get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_purity_auxiliary = get_adaptive_mtf_normalized_score(purity_auxiliary_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            context_mod_factor = (
                norm_mf_conviction * purity_context_sensitivity_conviction +
                norm_flow_credibility * purity_context_sensitivity_credibility +
                norm_mf_flow_gini_inverted * purity_context_sensitivity_gini +
                (1 - norm_retail_fomo) * purity_context_sensitivity_fomo
            ).clip(0, 1)
            purity_penalty = (
                norm_wash_trade_multi_tf * purity_penalty_factor_wash_trade +
                norm_deception_multi_tf.abs() * purity_penalty_factor_deception
            ) * (1 - context_mod_factor)
            purity_modulator = (1 - purity_penalty + norm_purity_auxiliary * purity_mitigation_factor).clip(0.01, 1.5)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道纯度精修 ---"] = ""
                debug_output[f"        norm_wash_trade_multi_tf: {norm_wash_trade_multi_tf.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_deception_multi_tf: {norm_deception_multi_tf.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_mf_conviction: {norm_mf_conviction.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_mf_flow_gini_inverted: {norm_mf_flow_gini_inverted.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_retail_fomo: {norm_retail_fomo.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_purity_auxiliary: {norm_purity_auxiliary.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        context_mod_factor: {context_mod_factor.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        purity_penalty: {purity_penalty.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        纯度调制器 (purity_modulator): {purity_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 3. 环境感知增强 (Environmental Awareness Enhancement) ---
        environment_modulator = pd.Series(1.0, index=df_index)
        if contextual_modulator_enabled:
            norm_liquidity_supply = get_adaptive_mtf_normalized_score(liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_liquidity_slope_5 = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_5_raw, df_index, tf_weights=tf_weights_ff)
            norm_liquidity_slope_13 = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_13_raw, df_index, tf_weights=tf_weights_ff)
            norm_liquidity_slope_mtf = (
                norm_liquidity_slope_5 * liquidity_slope_weights.get('slope_5', 0.6) +
                norm_liquidity_slope_13 * liquidity_slope_weights.get('slope_13', 0.4)
            ).clip(-1, 1)
            norm_liquidity_impact = get_adaptive_mtf_normalized_score(liquidity_impact_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
            liquidity_mod = (
                (norm_liquidity_supply * liquidity_mod_sensitivity_level) +
                (norm_liquidity_slope_mtf.abs() * liquidity_mod_sensitivity_slope) +
                (norm_liquidity_impact * liquidity_mod_sensitivity_level)
            ).clip(0, 1)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_price_volume_entropy = get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
            environment_mod = (
                (1 - norm_volatility_instability) * environment_mod_sensitivity_volatility +
                norm_trend_vitality * environment_mod_sensitivity_trend_vitality +
                norm_price_volume_entropy * environment_mod_sensitivity_entropy
            ).clip(0, 1)
            environment_modulator = (1 + liquidity_mod * environment_mod).clip(0.5, 1.5)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 环境感知增强 ---"] = ""
                debug_output[f"        norm_liquidity_supply: {norm_liquidity_supply.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_liquidity_slope_mtf: {norm_liquidity_slope_mtf.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_liquidity_impact: {norm_liquidity_impact.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        liquidity_mod: {liquidity_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_trend_vitality: {norm_trend_vitality.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_price_volume_entropy: {norm_price_volume_entropy.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        environment_mod: {environment_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        环境调制器 (environment_modulator): {environment_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 4. 结构洞察升级 (Structural Insight Upgrade) ---
        norm_lg_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(lg_flow_slope_5_raw, df_index, tf_weights_ff)
        norm_lg_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(lg_flow_accel_5_raw, df_index, tf_weights_ff)
        norm_xl_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(xl_flow_slope_5_raw, df_index, tf_weights_ff)
        norm_xl_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(xl_flow_accel_5_raw, df_index, tf_weights_ff)
        norm_retail_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(retail_flow_slope_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_retail_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(retail_flow_accel_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights=tf_weights_ff)
        norm_flow_quality = get_adaptive_mtf_normalized_score(flow_quality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_dominance = get_adaptive_mtf_normalized_score(retail_dominance_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        structural_momentum_score = (
            norm_lg_flow_slope_5 * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) +
            norm_lg_flow_accel_5 * structural_momentum_weights.get('large_order_flow_accel_5', 0.15) +
            norm_xl_flow_slope_5 * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) + # 假设xl和lg使用相同权重
            norm_xl_flow_accel_5 * structural_momentum_weights.get('large_order_flow_accel_5', 0.15) + # 假设xl和lg使用相同权重
            norm_main_force_flow_directionality * structural_momentum_weights.get('main_force_flow_directionality', 0.2) +
            norm_flow_quality * structural_momentum_weights.get('main_force_flow_gini', 0.15) + # flow_quality_signal_name 对应 main_force_flow_gini_D
            norm_retail_flow_slope_5 * structural_momentum_weights.get('retail_flow_slope_5', -0.1) +
            norm_retail_flow_accel_5 * structural_momentum_weights.get('retail_flow_accel_5', -0.05) +
            (1 - norm_retail_dominance) * structural_momentum_weights.get('retail_flow_dominance', -0.15) # 零售主导越低，结构动能越好
        ).clip(-1, 1)
        norm_rally_sell_distribution_intensity = get_adaptive_mtf_normalized_score(raw_data_cache['rally_sell_distribution_intensity_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_rally_buy_support_weakness = get_adaptive_mtf_normalized_score(raw_data_cache['rally_buy_support_weakness_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_buy_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_sell_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['retail_buy_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['retail_sell_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(raw_data_cache['wash_trade_buy_volume_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(raw_data_cache['wash_trade_sell_volume_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        structural_momentum_score = structural_momentum_score + \
                                    (norm_main_force_buy_ofi * structural_momentum_weights_v6_1.get('main_force_buy_ofi', 0.1)) + \
                                    (norm_retail_sell_ofi * structural_momentum_weights_v6_1.get('retail_sell_ofi', 0.05)) - \
                                    (norm_rally_sell_distribution_intensity * structural_momentum_weights_v6_1.get('rally_sell_distribution_intensity', -0.1)) - \
                                    (norm_rally_buy_support_weakness * structural_momentum_weights_v6_1.get('rally_buy_support_weakness', -0.1)) - \
                                    (norm_main_force_sell_ofi * structural_momentum_weights_v6_1.get('main_force_sell_ofi', -0.1)) - \
                                    (norm_retail_buy_ofi * structural_momentum_weights_v6_1.get('retail_buy_ofi', -0.05)) - \
                                    (norm_wash_trade_buy_volume * structural_momentum_weights_v6_1.get('wash_trade_buy_volume', -0.05)) - \
                                    (norm_wash_trade_sell_volume * structural_momentum_weights_v6_1.get('wash_trade_sell_volume', -0.05))
        structural_momentum_score = structural_momentum_score.clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 结构洞察升级 ---"] = ""
            debug_output[f"        norm_lg_flow_slope_5: {norm_lg_flow_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_lg_flow_accel_5: {norm_lg_flow_accel_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_xl_flow_slope_5: {norm_xl_flow_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_xl_flow_accel_5: {norm_xl_flow_accel_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_flow_slope_5: {norm_retail_flow_slope_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_flow_accel_5: {norm_retail_flow_accel_5.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_flow_directionality: {norm_main_force_flow_directionality.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_flow_quality: {norm_flow_quality.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_dominance: {norm_retail_dominance.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_rally_sell_distribution_intensity: {norm_rally_sell_distribution_intensity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_rally_buy_support_weakness: {norm_rally_buy_support_weakness.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_buy_ofi: {norm_main_force_buy_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_sell_ofi: {norm_main_force_sell_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_buy_ofi: {norm_retail_buy_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_retail_sell_ofi: {norm_retail_sell_ofi.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_wash_trade_buy_volume: {norm_wash_trade_buy_volume.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_wash_trade_sell_volume: {norm_wash_trade_sell_volume.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        结构动能分数 (structural_momentum_score): {structural_momentum_score.loc[probe_ts]:.4f}"] = ""
        # --- 5. 最终融合与演化趋势 (Final Fusion & Evolution Trend) ---
        # 融合基础动能、纯度调制和结构洞察
        fused_momentum_score = (base_momentum_score * purity_modulator * environment_modulator + structural_momentum_score).clip(-1, 1)
        smoothed_fused_momentum = fused_momentum_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_fused_momentum.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_1_name], df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_2_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_3 = get_adaptive_mtf_normalized_bipolar_score(raw_data_cache[dynamic_evolution_context_modulator_signal_3_name], df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_4 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_4_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2 +
            norm_dynamic_evolution_context_3.abs() * dynamic_evolution_context_sensitivity_3 +
            norm_dynamic_evolution_context_4 * dynamic_evolution_context_sensitivity_4
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1e-9)
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (fused_momentum_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 最终融合与演化趋势 ---"] = ""
            debug_output[f"        融合动能分数 (fused_momentum_score): {fused_momentum_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        平滑融合动能 (smoothed_fused_momentum): {smoothed_fused_momentum.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        速度 (velocity): {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加速度 (acceleration): {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        组合演化上下文调制 (combined_evolution_context_mod): {combined_evolution_context_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态基础权重: {dynamic_base_score_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态速度权重: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态加速度权重: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流纯度与动能诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self._print_debug_output(debug_output)
        return final_score.astype(np.float32)

    def _diagnose_axiom_capital_signature(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.3 · 信号替换与效率优化版 & 调试信息统一输出版】资金流公理五：诊断“主力资金的成本与效率特征”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 【修复】将缺失的 `main_force_vwap_deviation_D` 替换为 `intraday_vwap_div_index_D`。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_capital_signature"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断主力资金的成本与效率特征..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        cs_params = get_param_value(p_conf_ff.get('axiom_capital_signature_params'), {})
        # 替换 main_force_vwap_deviation 为 intraday_vwap_div_index
        cost_advantage_weights = get_param_value(cs_params.get('cost_advantage_weights'), {'main_force_cost_advantage': 0.6, 'intraday_vwap_div_index_D': 0.4})
        execution_efficiency_weights = get_param_value(cs_params.get('execution_efficiency_weights'), {'main_force_execution_alpha': 0.5, 'main_force_t0_efficiency': 0.5})
        capital_flow_quality_weights = get_param_value(cs_params.get('capital_flow_quality_weights'), {'main_force_flow_gini_inverted': 0.5, 'flow_credibility': 0.5})
        contextual_modulator_enabled = get_param_value(cs_params.get('contextual_modulator_enabled'), True)
        context_modulator_signal_1_name = get_param_value(cs_params.get('context_modulator_signal_1'), 'market_sentiment_score_D')
        context_modulator_signal_2_name = get_param_value(cs_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_signal_3_name = get_param_value(cs_params.get('context_modulator_signal_3'), 'order_book_liquidity_supply_D')
        context_sensitivity_sentiment = get_param_value(cs_params.get('context_sensitivity_sentiment'), 0.3)
        context_sensitivity_volatility = get_param_value(cs_params.get('context_sensitivity_volatility'), 0.2)
        context_sensitivity_liquidity = get_param_value(cs_params.get('context_sensitivity_liquidity'), 0.1)
        smoothing_ema_span = get_param_value(cs_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(cs_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(cs_params.get('dynamic_evolution_context_modulator_signal_1'), 'main_force_conviction_index_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(cs_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(cs_params.get('dynamic_evolution_context_modulator_signal_2'), 'trend_vitality_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(cs_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        required_signals = [
            'main_force_cost_advantage_D', 'intraday_vwap_div_index_D', # 替换为 intraday_vwap_div_index_D
            'main_force_execution_alpha_D', 'main_force_t0_efficiency_D',
            'main_force_flow_gini_D', 'flow_credibility_index_D',
            context_modulator_signal_1_name, context_modulator_signal_2_name, context_modulator_signal_3_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name
        ]
        required_signals = list(set(required_signals))
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        main_force_cost_advantage_raw = raw_data_cache['main_force_cost_advantage_D']
        main_force_vwap_deviation_raw = raw_data_cache['intraday_vwap_div_index_D'] # 使用替换后的信号
        main_force_execution_alpha_raw = raw_data_cache['main_force_execution_alpha_D']
        main_force_t0_efficiency_raw = raw_data_cache['main_force_t0_efficiency_D']
        main_force_flow_gini_raw = raw_data_cache['main_force_flow_gini_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        market_sentiment_raw = raw_data_cache[context_modulator_signal_1_name]
        volatility_instability_raw = raw_data_cache[context_modulator_signal_2_name]
        liquidity_supply_raw = raw_data_cache[context_modulator_signal_3_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        # --- 1. 成本优势 (Cost Advantage) ---
        norm_main_force_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights_ff)
        # 使用 intraday_vwap_div_index_D，它通常是双极性的
        norm_main_force_vwap_deviation = get_adaptive_mtf_normalized_bipolar_score(main_force_vwap_deviation_raw, df_index, tf_weights=tf_weights_ff)
        cost_advantage_score = (
            norm_main_force_cost_advantage * cost_advantage_weights.get('main_force_cost_advantage', 0.6) +
            norm_main_force_vwap_deviation * cost_advantage_weights.get('intraday_vwap_div_index_D', 0.4) # 更新权重键名
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 成本优势 ---"] = ""
            debug_output[f"        norm_main_force_cost_advantage: {norm_main_force_cost_advantage.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_vwap_deviation (intraday_vwap_div_index_D): {norm_main_force_vwap_deviation.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        成本优势分数 (cost_advantage_score): {cost_advantage_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 执行效率 (Execution Efficiency) ---
        norm_main_force_execution_alpha = get_adaptive_mtf_normalized_bipolar_score(main_force_execution_alpha_raw, df_index, tf_weights=tf_weights_ff)
        norm_main_force_t0_efficiency = get_adaptive_mtf_normalized_score(main_force_t0_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        execution_efficiency_score = (
            norm_main_force_execution_alpha * execution_efficiency_weights.get('main_force_execution_alpha', 0.5) +
            norm_main_force_t0_efficiency * execution_efficiency_weights.get('main_force_t0_efficiency', 0.5)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 执行效率 ---"] = ""
            debug_output[f"        norm_main_force_execution_alpha: {norm_main_force_execution_alpha.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_main_force_t0_efficiency: {norm_main_force_t0_efficiency.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        执行效率分数 (execution_efficiency_score): {execution_efficiency_score.loc[probe_ts]:.4f}"] = ""
        # --- 3. 资金流质量 (Capital Flow Quality) ---
        norm_main_force_flow_gini_inverted = 1 - get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        capital_flow_quality_score = (
            norm_main_force_flow_gini_inverted * capital_flow_quality_weights.get('main_force_flow_gini_inverted', 0.5) +
            norm_flow_credibility * capital_flow_quality_weights.get('flow_credibility', 0.5)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 资金流质量 ---"] = ""
            debug_output[f"        norm_main_force_flow_gini_inverted: {norm_main_force_flow_gini_inverted.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        资金流质量分数 (capital_flow_quality_score): {capital_flow_quality_score.loc[probe_ts]:.4f}"] = ""
        # --- 4. 情境调制器 (Contextual Modulator) ---
        context_modulator = pd.Series(1.0, index=df_index)
        if contextual_modulator_enabled:
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_liquidity_supply = get_adaptive_mtf_normalized_score(liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            
            sentiment_mod = (1 + norm_market_sentiment.abs() * context_sensitivity_sentiment * np.sign(norm_market_sentiment))
            volatility_mod = (1 - norm_volatility_instability * context_sensitivity_volatility)
            liquidity_mod = (1 + norm_liquidity_supply * context_sensitivity_liquidity)
            
            context_modulator = (sentiment_mod * volatility_mod * liquidity_mod).clip(0.5, 1.5)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制器 ---"] = ""
                debug_output[f"        norm_market_sentiment: {norm_market_sentiment.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_liquidity_supply: {norm_liquidity_supply.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        sentiment_mod: {sentiment_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        volatility_mod: {volatility_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        liquidity_mod: {liquidity_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        情境调制器 (context_modulator): {context_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 5. 融合基础分数 (Fusion Base Score) ---
        base_capital_signature_score = (
            cost_advantage_score * 0.4 +
            execution_efficiency_score * 0.3 +
            capital_flow_quality_score * 0.3
        ) * context_modulator
        base_capital_signature_score = base_capital_signature_score.clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合基础分数 ---"] = ""
            debug_output[f"        基础资金特征分数 (base_capital_signature_score): {base_capital_signature_score.loc[probe_ts]:.4f}"] = ""
        # --- 6. 演化趋势与前瞻性增强 (Evolution Trend & Foresight Enhancement) ---
        smoothed_base_score = base_capital_signature_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_bipolar_score(raw_data_cache[dynamic_evolution_context_modulator_signal_1_name], df_index, tf_weights=tf_weights_ff) # main_force_conviction_index_D 是双极性
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_2_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1e-9)
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_capital_signature_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 演化趋势与前瞻性增强 ---"] = ""
            debug_output[f"        平滑基础分数 (smoothed_base_score): {smoothed_base_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        速度 (velocity): {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加速度 (acceleration): {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        组合演化上下文调制 (combined_evolution_context_mod): {combined_evolution_context_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态基础权重: {dynamic_base_score_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态速度权重: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态加速度权重: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 主力资金的成本与效率特征诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self._print_debug_output(debug_output)
        return final_score.astype(np.float32)

    def _diagnose_axiom_flow_structure_health(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.2 · 效率优化版 & 调试信息统一输出版】资金流公理六：诊断“资金流结构健康度”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_flow_structure_health"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断资金流结构健康度..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        fsh_params = get_param_value(p_conf_ff.get('axiom_flow_structure_health_params'), {})
        microstructure_efficiency_weights = get_param_value(fsh_params.get('microstructure_efficiency_weights'), {'microstructure_efficiency_index': 0.6, 'order_book_imbalance': 0.4})
        liquidity_authenticity_weights = get_param_value(fsh_params.get('liquidity_authenticity_weights'), {'liquidity_authenticity_score': 0.7, 'wash_trade_intensity_inverted': 0.3})
        flow_quality_weights = get_param_value(fsh_params.get('flow_quality_weights'), {'main_force_flow_gini_inverted': 0.5, 'flow_credibility': 0.5})
        contextual_modulator_enabled = get_param_value(fsh_params.get('contextual_modulator_enabled'), True)
        context_modulator_signal_1_name = get_param_value(fsh_params.get('context_modulator_signal_1'), 'market_sentiment_score_D')
        context_modulator_signal_2_name = get_param_value(fsh_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_signal_3_name = get_param_value(fsh_params.get('context_modulator_signal_3'), 'trend_vitality_index_D')
        context_sensitivity_sentiment = get_param_value(fsh_params.get('context_sensitivity_sentiment'), 0.3)
        context_sensitivity_volatility = get_param_value(fsh_params.get('context_sensitivity_volatility'), 0.2)
        context_sensitivity_trend_vitality = get_param_value(fsh_params.get('context_sensitivity_trend_vitality'), 0.1)
        smoothing_ema_span = get_param_value(fsh_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(fsh_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(fsh_params.get('dynamic_evolution_context_modulator_signal_1'), 'main_force_conviction_index_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(fsh_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(fsh_params.get('dynamic_evolution_context_modulator_signal_2'), 'flow_credibility_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(fsh_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        required_signals = [
            'microstructure_efficiency_index_D', 'order_book_imbalance_D',
            'liquidity_authenticity_score_D', 'wash_trade_intensity_D',
            'main_force_flow_gini_D', 'flow_credibility_index_D',
            context_modulator_signal_1_name, context_modulator_signal_2_name, context_modulator_signal_3_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name
        ]
        required_signals = list(set(required_signals))
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        microstructure_efficiency_raw = raw_data_cache['microstructure_efficiency_index_D']
        order_book_imbalance_raw = raw_data_cache['order_book_imbalance_D']
        liquidity_authenticity_raw = raw_data_cache['liquidity_authenticity_score_D']
        wash_trade_intensity_raw = raw_data_cache['wash_trade_intensity_D']
        main_force_flow_gini_raw = raw_data_cache['main_force_flow_gini_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        market_sentiment_raw = raw_data_cache[context_modulator_signal_1_name]
        volatility_instability_raw = raw_data_cache[context_modulator_signal_2_name]
        trend_vitality_raw = raw_data_cache[context_modulator_signal_3_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
        # --- 1. 微观结构效率 (Microstructure Efficiency) ---
        norm_microstructure_efficiency = get_adaptive_mtf_normalized_score(microstructure_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_order_book_imbalance = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights=tf_weights_ff)
        microstructure_efficiency_score = (
            norm_microstructure_efficiency * microstructure_efficiency_weights.get('microstructure_efficiency_index', 0.6) +
            norm_order_book_imbalance * microstructure_efficiency_weights.get('order_book_imbalance', 0.4)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 微观结构效率 ---"] = ""
            debug_output[f"        norm_microstructure_efficiency: {norm_microstructure_efficiency.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_order_book_imbalance: {norm_order_book_imbalance.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        微观结构效率分数 (microstructure_efficiency_score): {microstructure_efficiency_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 流动性真实性 (Liquidity Authenticity) ---
        norm_liquidity_authenticity = get_adaptive_mtf_normalized_score(liquidity_authenticity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_intensity_inverted = 1 - get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        liquidity_authenticity_score = (
            norm_liquidity_authenticity * liquidity_authenticity_weights.get('liquidity_authenticity_score', 0.7) +
            norm_wash_trade_intensity_inverted * liquidity_authenticity_weights.get('wash_trade_intensity_inverted', 0.3)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 流动性真实性 ---"] = ""
            debug_output[f"        norm_liquidity_authenticity: {norm_liquidity_authenticity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_wash_trade_intensity_inverted: {norm_wash_trade_intensity_inverted.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        流动性真实性分数 (liquidity_authenticity_score): {liquidity_authenticity_score.loc[probe_ts]:.4f}"] = ""
        # --- 3. 资金流质量 (Flow Quality) ---
        norm_main_force_flow_gini_inverted = 1 - get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        flow_quality_score = (
            norm_main_force_flow_gini_inverted * flow_quality_weights.get('main_force_flow_gini_inverted', 0.5) +
            norm_flow_credibility * flow_quality_weights.get('flow_credibility', 0.5)
        ).clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 资金流质量 ---"] = ""
            debug_output[f"        norm_main_force_flow_gini_inverted: {norm_main_force_flow_gini_inverted.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        资金流质量分数 (flow_quality_score): {flow_quality_score.loc[probe_ts]:.4f}"] = ""
        # --- 4. 情境调制器 (Contextual Modulator) ---
        context_modulator = pd.Series(1.0, index=df_index)
        if contextual_modulator_enabled:
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            sentiment_mod = (1 + norm_market_sentiment.abs() * context_sensitivity_sentiment * np.sign(norm_market_sentiment))
            volatility_mod = (1 - norm_volatility_instability * context_sensitivity_volatility)
            trend_vitality_mod = (1 + norm_trend_vitality * context_sensitivity_trend_vitality)
            context_modulator = (sentiment_mod * volatility_mod * trend_vitality_mod).clip(0.5, 1.5)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制器 ---"] = ""
                debug_output[f"        norm_market_sentiment: {norm_market_sentiment.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_trend_vitality: {norm_trend_vitality.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        sentiment_mod: {sentiment_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        volatility_mod: {volatility_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        trend_vitality_mod: {trend_vitality_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        情境调制器 (context_modulator): {context_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 5. 融合基础分数 (Fusion Base Score) ---
        base_flow_structure_health_score = (
            microstructure_efficiency_score * 0.4 +
            liquidity_authenticity_score * 0.3 +
            flow_quality_score * 0.3
        ) * context_modulator
        base_flow_structure_health_score = base_flow_structure_health_score.clip(-1, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合基础分数 ---"] = ""
            debug_output[f"        基础资金流结构健康分数 (base_flow_structure_health_score): {base_flow_structure_health_score.loc[probe_ts]:.4f}"] = ""
        # --- 6. 演化趋势与前瞻性增强 (Evolution Trend & Foresight Enhancement) ---
        smoothed_base_score = base_flow_structure_health_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_1_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_2_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1e-9)
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_flow_structure_health_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 演化趋势与前瞻性增强 ---"] = ""
            debug_output[f"        平滑基础分数 (smoothed_base_score): {smoothed_base_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        速度 (velocity): {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加速度 (acceleration): {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        组合演化上下文调制 (combined_evolution_context_mod): {combined_evolution_context_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态基础权重: {dynamic_base_score_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态速度权重: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态加速度权重: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流结构健康度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self._print_debug_output(debug_output)
        return final_score.astype(np.float32)

    def _calculate_mtf_cohesion_divergence(self, df: pd.DataFrame, signal_base_name: str, short_periods: List[int], long_periods: List[int], is_bipolar: bool, tf_weights: Dict, pre_fetched_data: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """
        【V4.4 升级 · Numba优化与共振因子精修版 & 趋势质量强化版 & 调试信息统一输出版】计算双极性多时间框架的共振/背离因子。
        - 核心优化: 将核心数值计算逻辑迁移至Numba加速的辅助函数 `_numba_mtf_cohesion_divergence_core`。
        - 效率优化: 增加了 `pre_fetched_data` 参数，允许预先传入数据，避免重复调用 `_get_safe_series`。
        - 新增: `tf_weights` 参数，允许传入不同的时间框架权重。
        - 共振因子精修: 优化了方向一致性和强度一致性的计算，使其在斜率接近零时也能更合理地反映共振状态。
        - 趋势质量强化: 引入了趋势质量调制器，评估短期和长期趋势方向的持续性，以增强共振信号的可靠性。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name_str = "_calculate_mtf_cohesion_divergence"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name_str)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name_str} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name_str} @ {probe_ts.strftime('%Y-%m-%d')}: 正在计算MTF共振/背离因子 for {signal_base_name}..."] = ""
        # 1. 获取短期和长期斜率/加速度的归一化分数
        short_slope_scores = []
        short_accel_scores = []
        long_slope_scores = []
        long_accel_scores = []
        # Helper to get normalized score
        def _get_norm_score(raw_series, is_bipolar_flag):
            if is_bipolar_flag:
                return get_adaptive_mtf_normalized_bipolar_score(raw_series, df_index, tf_weights)
            else:
                return get_adaptive_mtf_normalized_score(raw_series, df_index, ascending=True, tf_weights=tf_weights)
        # Process short periods
        for p in short_periods:
            slope_col = f'SLOPE_{p}_{signal_base_name}'
            accel_col = f'ACCEL_{p}_{signal_base_name}'
            slope_raw = pre_fetched_data.get(slope_col) if pre_fetched_data else self._get_safe_series(df, df, slope_col, 0.0, method_name=method_name_str)
            accel_raw = pre_fetched_data.get(accel_col) if pre_fetched_data else self._get_safe_series(df, df, accel_col, 0.0, method_name=method_name_str)
            short_slope_scores.append(_get_norm_score(slope_raw, is_bipolar))
            short_accel_scores.append(_get_norm_score(accel_raw, is_bipolar))
        # Process long periods
        for p in long_periods:
            slope_col = f'SLOPE_{p}_{signal_base_name}'
            accel_col = f'ACCEL_{p}_{signal_base_name}'
            slope_raw = pre_fetched_data.get(slope_col) if pre_fetched_data else self._get_safe_series(df, df, slope_col, 0.0, method_name=method_name_str)
            accel_raw = pre_fetched_data.get(accel_col) if pre_fetched_data else self._get_safe_series(df, df, accel_col, 0.0, method_name=method_name_str)
            long_slope_scores.append(_get_norm_score(slope_raw, is_bipolar))
            long_accel_scores.append(_get_norm_score(accel_raw, is_bipolar))
        # Calculate average scores, handling empty lists
        avg_short_slope = sum(short_slope_scores) / len(short_slope_scores) if short_slope_scores else pd.Series(0.0, index=df_index)
        avg_short_accel = sum(short_accel_scores) / len(short_accel_scores) if short_accel_scores else pd.Series(0.0, index=df_index)
        avg_long_slope = sum(long_slope_scores) / len(long_slope_scores) if long_slope_scores else pd.Series(0.0, index=df_index)
        avg_long_accel = sum(long_accel_scores) / len(long_accel_scores) if long_accel_scores else pd.Series(0.0, index=df_index)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name_str} @ {probe_ts.strftime('%Y-%m-%d')}: --- 平均归一化斜率/加速度 ---"] = ""
            debug_output[f"        avg_short_slope: {avg_short_slope.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        avg_short_accel: {avg_short_accel.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        avg_long_slope: {avg_long_slope.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        avg_long_accel: {avg_long_accel.loc[probe_ts]:.4f}"] = ""
        # 2. 准备NumPy数组，传递给Numba函数
        epsilon_sign = 1e-9 # Small constant to avoid division by zero
        persistence_window = 5 # Default, can be made configurable
        mtf_resonance_score_values = _numba_mtf_cohesion_divergence_core(
            avg_short_slope.values,
            avg_short_accel.values,
            avg_long_slope.values,
            avg_long_accel.values,
            epsilon_sign,
            persistence_window
        )
        # 3. 将Numba函数返回的NumPy数组转换回Pandas Series
        mtf_resonance_score = pd.Series(mtf_resonance_score_values, index=df_index, dtype=np.float32)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name_str} @ {probe_ts.strftime('%Y-%m-%d')}: --- Numba核心计算结果 ---"] = ""
            debug_output[f"        mtf_resonance_score: {mtf_resonance_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name_str} @ {probe_ts.strftime('%Y-%m-%d')}: MTF共振/背离因子计算完成。"] = ""
            self._print_debug_output(debug_output)
        return mtf_resonance_score.astype(np.float32)

    def _diagnose_fund_flow_divergence_signals(self, df: pd.DataFrame, norm_window: int, axiom_divergence: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V3.1 · 效率优化版 & 调试信息统一输出版】诊断资金流看涨/看跌背离信号。
        - 核心优化: 集中所有原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_fund_flow_divergence_signals"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断资金流看涨/看跌背离信号..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        div_params = get_param_value(p_conf_ff.get('fund_flow_divergence_params'), {})
        divergence_threshold = get_param_value(div_params.get('divergence_threshold'), 0.5)
        conviction_threshold = get_param_value(div_params.get('conviction_threshold'), 0.3)
        flow_credibility_threshold = get_param_value(div_params.get('flow_credibility_threshold'), 0.6)
        momentum_threshold = get_param_value(div_params.get('momentum_threshold'), 0.4)
        sentiment_mod_sensitivity = get_param_value(div_params.get('sentiment_mod_sensitivity'), 0.2)
        volatility_mod_sensitivity = get_param_value(div_params.get('volatility_mod_sensitivity'), 0.1)
        smoothing_ema_span = get_param_value(div_params.get('smoothing_ema_span'), 5)
        required_signals = [
            'main_force_conviction_index_D', 'main_force_flow_directionality_D',
            'flow_credibility_index_D', 'market_sentiment_score_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D'
        ]
        required_signals = list(set(required_signals))
        if not self._validate_required_signals(df, required_signals, method_name, atomic_states=self.strategy.atomic_states):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32), pd.Series(0.0, index=df_index, dtype=np.float32)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        main_force_conviction_raw = raw_data_cache['main_force_conviction_index_D']
        main_force_flow_directionality_raw = raw_data_cache['main_force_flow_directionality_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        market_sentiment_raw = raw_data_cache['market_sentiment_score_D']
        volatility_instability_raw = raw_data_cache['VOLATILITY_INSTABILITY_INDEX_21d_D']
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"        axiom_divergence: {axiom_divergence.loc[probe_ts]:.4f}"] = ""
        # --- 1. 归一化关键信号 ---
        norm_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
        norm_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights=tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 归一化关键信号 ---"] = ""
            debug_output[f"        norm_conviction: {norm_conviction.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_flow_directionality: {norm_flow_directionality.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_flow_credibility: {norm_flow_credibility.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_market_sentiment: {norm_market_sentiment.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
        # --- 2. 情境调制器 ---
        sentiment_mod = (1 + norm_market_sentiment.abs() * sentiment_mod_sensitivity * np.sign(norm_market_sentiment))
        volatility_mod = (1 - norm_volatility_instability * volatility_mod_sensitivity)
        context_modulator = (sentiment_mod * volatility_mod).clip(0.5, 1.5)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制器 ---"] = ""
            debug_output[f"        sentiment_mod: {sentiment_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        volatility_mod: {volatility_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        context_modulator: {context_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 3. 看涨背离信号 (Bullish Divergence) ---
        # 资金流分歧为正 (看涨)，主力信念为正 (看涨)，资金流方向为正 (看涨)，且可信度高
        bullish_divergence_base = (
            (axiom_divergence > divergence_threshold) &
            (norm_conviction > conviction_threshold) &
            (norm_flow_directionality > momentum_threshold) &
            (norm_flow_credibility > flow_credibility_threshold)
        ).astype(np.float32)
        # 强度调制：分歧越大、信念越强、方向性越强、可信度越高，信号越强
        bullish_divergence_strength = (
            (axiom_divergence.clip(lower=divergence_threshold) - divergence_threshold) +
            (norm_conviction.clip(lower=conviction_threshold) - conviction_threshold) +
            (norm_flow_directionality.clip(lower=momentum_threshold) - momentum_threshold) +
            (norm_flow_credibility.clip(lower=flow_credibility_threshold) - flow_credibility_threshold)
        ).clip(0, 1)
        bullish_divergence = bullish_divergence_base * bullish_divergence_strength * context_modulator
        bullish_divergence = bullish_divergence.ewm(span=smoothing_ema_span, adjust=False).mean().clip(0, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看涨背离信号 ---"] = ""
            debug_output[f"        bullish_divergence_base: {bullish_divergence_base.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        bullish_divergence_strength: {bullish_divergence_strength.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        看涨背离信号 (bullish_divergence): {bullish_divergence.loc[probe_ts]:.4f}"] = ""
        # --- 4. 看跌背离信号 (Bearish Divergence) ---
        # 资金流分歧为负 (看跌)，主力信念为负 (看跌)，资金流方向为负 (看跌)，且可信度高
        bearish_divergence_base = (
            (axiom_divergence < -divergence_threshold) &
            (norm_conviction < -conviction_threshold) &
            (norm_flow_directionality < -momentum_threshold) &
            (norm_flow_credibility > flow_credibility_threshold)
        ).astype(np.float32)
        # 强度调制：分歧越大、信念越强、方向性越强、可信度越高，信号越强
        bearish_divergence_strength = (
            (axiom_divergence.clip(upper=-divergence_threshold) + divergence_threshold).abs() +
            (norm_conviction.clip(upper=-conviction_threshold) + conviction_threshold).abs() +
            (norm_flow_directionality.clip(upper=-momentum_threshold) + momentum_threshold).abs() +
            (norm_flow_credibility.clip(lower=flow_credibility_threshold) - flow_credibility_threshold)
        ).clip(0, 1)
        bearish_divergence = bearish_divergence_base * bearish_divergence_strength * context_modulator
        bearish_divergence = bearish_divergence.ewm(span=smoothing_ema_span, adjust=False).mean().clip(0, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 看跌背离信号 ---"] = ""
            debug_output[f"        bearish_divergence_base: {bearish_divergence_base.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        bearish_divergence_strength: {bearish_divergence_strength.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        看跌背离信号 (bearish_divergence): {bearish_divergence.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流看涨/看跌背离信号诊断完成。"] = ""
            self._print_debug_output(debug_output)
        return bullish_divergence, bearish_divergence

    def _diagnose_axiom_intent_purity(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.2 · 效率优化版 & 调试信息统一输出版】资金流公理七：诊断“资金流意图纯度”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 【新增】所有调试信息统一在方法末尾输出。
        """
        method_name = "_diagnose_axiom_intent_purity"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('enabled'), False) and get_param_value(self.debug_params.get('should_probe'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        if probe_ts is None:
            is_debug_enabled = False
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 调试信息收集字典
        debug_output = {}
        if is_debug_enabled and probe_ts:
            debug_output[f"--- {method_name} 诊断详情 @ {probe_ts.strftime('%Y-%m-%d')} ---"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 正在诊断资金流意图纯度..."] = ""
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ip_params = get_param_value(p_conf_ff.get('axiom_intent_purity_params'), {})
        purity_components_weights = get_param_value(ip_params.get('purity_components_weights'), {
            'main_force_flow_purity': 0.4, 'order_flow_imbalance_purity': 0.3, 'liquidity_authenticity': 0.3
        })
        deception_risk_penalty_factor = get_param_value(ip_params.get('deception_risk_penalty_factor'), 0.8)
        divergence_mod_sensitivity = get_param_value(ip_params.get('divergence_mod_sensitivity'), 0.5)
        contextual_modulator_enabled = get_param_value(ip_params.get('contextual_modulator_enabled'), True)
        context_modulator_signal_1_name = get_param_value(ip_params.get('context_modulator_signal_1'), 'market_sentiment_score_D')
        context_modulator_signal_2_name = get_param_value(ip_params.get('context_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        context_modulator_signal_3_name = get_param_value(ip_params.get('context_modulator_signal_3'), 'trend_vitality_index_D')
        context_sensitivity_sentiment = get_param_value(ip_params.get('context_sensitivity_sentiment'), 0.3)
        context_sensitivity_volatility = get_param_value(ip_params.get('context_sensitivity_volatility'), 0.2)
        context_sensitivity_trend_vitality = get_param_value(ip_params.get('context_sensitivity_trend_vitality'), 0.1)
        smoothing_ema_span = get_param_value(ip_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ip_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(ip_params.get('dynamic_evolution_context_modulator_signal_1'), 'main_force_conviction_index_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(ip_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(ip_params.get('dynamic_evolution_context_modulator_signal_2'), 'flow_credibility_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(ip_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        required_signals = [
            'peak_exchange_purity_D', 'order_flow_imbalance_score_D', 'liquidity_authenticity_score_D',
            'wash_trade_intensity_D', 'deception_index_D',
            context_modulator_signal_1_name, context_modulator_signal_2_name, context_modulator_signal_3_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name
        ]
        required_signals = list(set(required_signals))
        # 确保 axiom_divergence 和 SCORE_FF_DECEPTION_RISK 存在于 atomic_states
        if 'SCORE_FF_AXIOM_DIVERGENCE' not in self.strategy.atomic_states:
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号 SCORE_FF_AXIOM_DIVERGENCE，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        if 'SCORE_FF_DECEPTION_RISK' not in self.strategy.atomic_states:
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号 SCORE_FF_DECEPTION_RISK，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        axiom_divergence = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_DIVERGENCE', 0.0, method_name=method_name)
        score_ff_deception_risk = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_DECEPTION_RISK', 0.0, method_name=method_name)
        if not self._validate_required_signals(df, required_signals, method_name):
            if is_debug_enabled and probe_ts:
                debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 缺少必要信号，返回0。"] = ""
                self._print_debug_output(debug_output)
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        main_force_flow_purity_raw = raw_data_cache['peak_exchange_purity_D']
        order_flow_imbalance_score_raw = raw_data_cache['order_flow_imbalance_score_D']
        liquidity_authenticity_raw = raw_data_cache['liquidity_authenticity_score_D']
        wash_trade_intensity_raw = raw_data_cache['wash_trade_intensity_D']
        deception_index_raw = raw_data_cache['deception_index_D']
        market_sentiment_raw = raw_data_cache[context_modulator_signal_1_name]
        volatility_instability_raw = raw_data_cache[context_modulator_signal_2_name]
        trend_vitality_raw = raw_data_cache[context_modulator_signal_3_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 原始信号值 ---"] = ""
            for sig_name in required_signals:
                val = raw_data_cache[sig_name].loc[probe_ts] if probe_ts in raw_data_cache[sig_name].index else np.nan
                debug_output[f"        '{sig_name}': {val:.4f}"] = ""
            debug_output[f"        axiom_divergence: {axiom_divergence.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        SCORE_FF_DECEPTION_RISK: {score_ff_deception_risk.loc[probe_ts]:.4f}"] = ""
        # --- 1. 意图纯度核心组件 (Core Intent Purity Components) ---
        norm_main_force_flow_purity = get_adaptive_mtf_normalized_score(main_force_flow_purity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_order_flow_imbalance_score = get_adaptive_mtf_normalized_score(order_flow_imbalance_score_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_liquidity_authenticity = get_adaptive_mtf_normalized_score(liquidity_authenticity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        core_purity_score = (
            norm_main_force_flow_purity * purity_components_weights.get('main_force_flow_purity', 0.4) +
            norm_order_flow_imbalance_score * purity_components_weights.get('order_flow_imbalance_purity', 0.3) +
            norm_liquidity_authenticity * purity_components_weights.get('liquidity_authenticity', 0.3)
        ).clip(0, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 意图纯度核心组件 ---"] = ""
            debug_output[f"        norm_main_force_flow_purity: {norm_main_force_flow_purity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_order_flow_imbalance_score: {norm_order_flow_imbalance_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        norm_liquidity_authenticity: {norm_liquidity_authenticity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        核心纯度分数 (core_purity_score): {core_purity_score.loc[probe_ts]:.4f}"] = ""
        # --- 2. 诡道风险与分歧调制 (Deception Risk & Divergence Modulation) ---
        # 诡道风险越高，纯度越低
        deception_penalty = score_ff_deception_risk * deception_risk_penalty_factor
        # 资金流分歧越大，意图纯度越受影响
        divergence_mod = (1 - axiom_divergence.abs() * divergence_mod_sensitivity).clip(0, 1)
        purity_modulated_by_risk_divergence = core_purity_score * (1 - deception_penalty) * divergence_mod
        purity_modulated_by_risk_divergence = purity_modulated_by_risk_divergence.clip(0, 1)
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 诡道风险与分歧调制 ---"] = ""
            debug_output[f"        deception_penalty: {deception_penalty.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        divergence_mod: {divergence_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        风险分歧调制后的纯度 (purity_modulated_by_risk_divergence): {purity_modulated_by_risk_divergence.loc[probe_ts]:.4f}"] = ""
        # --- 3. 情境调制器 (Contextual Modulator) ---
        context_modulator = pd.Series(1.0, index=df_index)
        if contextual_modulator_enabled:
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            sentiment_mod = (1 + norm_market_sentiment.abs() * context_sensitivity_sentiment * np.sign(norm_market_sentiment))
            volatility_mod = (1 - norm_volatility_instability * context_sensitivity_volatility)
            trend_vitality_mod = (1 + norm_trend_vitality * context_sensitivity_trend_vitality)
            context_modulator = (sentiment_mod * volatility_mod * trend_vitality_mod).clip(0.5, 1.5)
            if is_debug_enabled and probe_ts:
                debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 情境调制器 ---"] = ""
                debug_output[f"        norm_market_sentiment: {norm_market_sentiment.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_volatility_instability: {norm_volatility_instability.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        norm_trend_vitality: {norm_trend_vitality.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        sentiment_mod: {sentiment_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        volatility_mod: {volatility_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        trend_vitality_mod: {trend_vitality_mod.loc[probe_ts]:.4f}"] = ""
                debug_output[f"        情境调制器 (context_modulator): {context_modulator.loc[probe_ts]:.4f}"] = ""
        # --- 4. 融合基础分数 (Fusion Base Score) ---
        base_intent_purity_score = purity_modulated_by_risk_divergence * context_modulator
        base_intent_purity_score = base_intent_purity_score.clip(0, 1) # 纯度分数通常是0到1
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 融合基础分数 ---"] = ""
            debug_output[f"        基础意图纯度分数 (base_intent_purity_score): {base_intent_purity_score.loc[probe_ts]:.4f}"] = ""
        # --- 5. 演化趋势与前瞻性增强 (Evolution Trend & Foresight Enhancement) ---
        smoothed_base_score = base_intent_purity_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_1_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(raw_data_cache[dynamic_evolution_context_modulator_signal_2_name], df_index, ascending=True, tf_weights=tf_weights_ff)
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        total_dynamic_weights = total_dynamic_weights.replace(0, 1e-9)
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        # 意图纯度分数通常是0到1，所以这里不需要 (score + 1) / 2 的转换
        final_score = (
            base_intent_purity_score.pow(dynamic_base_score_weight) *
            ((norm_velocity + 1) / 2).pow(dynamic_velocity_weight) * # 速度和加速度仍是双极性
            ((norm_acceleration + 1) / 2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight))
        final_score = final_score.clip(0, 1) # 最终分数也应在0到1之间
        if is_debug_enabled and probe_ts:
            debug_output[f"      [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: --- 演化趋势与前瞻性增强 ---"] = ""
            debug_output[f"        平滑基础分数 (smoothed_base_score): {smoothed_base_score.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        速度 (velocity): {velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        加速度 (acceleration): {acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化速度 (norm_velocity): {norm_velocity.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        归一化加速度 (norm_acceleration): {norm_acceleration.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        组合演化上下文调制 (combined_evolution_context_mod): {combined_evolution_context_mod.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态基础权重: {dynamic_base_score_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态速度权重: {dynamic_velocity_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"        动态加速度权重: {dynamic_acceleration_weight.loc[probe_ts]:.4f}"] = ""
            debug_output[f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 资金流意图纯度诊断完成，最终分值: {final_score.loc[probe_ts]:.4f}"] = ""
            self._print_debug_output(debug_output)
        return final_score.astype(np.float32)

    def _diagnose_deception_risk(self, df: pd.DataFrame, debug_info_tuple: tuple) -> pd.Series:
        """
        【V1.1 · Numba优化版】诊断资金流层面的诡道博弈风险。
        融合了欺骗指数、对倒强度、诱多/诱空强度以及资金流可信度等因素，并根据市场情境进行动态调制。
        高分代表资金流层面存在显著的诡道风险，如主力诱多、虚假对倒等，预示潜在的陷阱或风险。
        - 核心优化: 将最终的风险聚合计算逻辑迁移至Numba加速的辅助函数 `_numba_calculate_deception_risk_core`。
        """
        method_name = "_diagnose_deception_risk"
        df_index = df.index
        p_conf_ff = self.p_conf_ff
        ac_params = get_param_value(p_conf_ff.get('axiom_consensus_params'), {})
        is_debug_enabled_for_method, probe_ts, _ = debug_info_tuple
        deception_mod_enabled = get_param_value(ac_params.get('deception_mod_enabled'), True)
        # 如果模块被禁用，则提前返回0分
        if not deception_mod_enabled:
            # if is_debug_enabled_for_method and probe_ts and probe_ts in df_index:
            #     print(f"  -- [资金流层调试] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}: 诡道风险模块被禁用，返回0。")
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 获取配置参数
        wash_trade_penalty_sensitivity = get_param_value(ac_params.get('wash_trade_penalty_sensitivity'), 0.4)
        deception_penalty_sensitivity = get_param_value(ac_params.get('deception_penalty_sensitivity'), 0.6)
        deception_lure_long_penalty_sensitivity = get_param_value(ac_params.get('deception_lure_long_penalty_sensitivity'), 0.2)
        deception_context_sensitivity = get_param_value(ac_params.get('deception_context_sensitivity'), 0.3)
        flow_credibility_threshold = get_param_value(ac_params.get('flow_credibility_threshold'), 0.5)
        deception_context_modulator_signal_name = get_param_value(ac_params.get('deception_context_modulator_signal'), 'market_sentiment_score_D')
        # 预取所有斜率和加速度数据（从 _diagnose_axiom_consensus 复制过来，确保独立性）
        all_mtf_periods = list(set(get_param_value(ac_params.get('mtf_cohesion_params', {}).get('short_periods'), [5, 13]) + get_param_value(ac_params.get('mtf_cohesion_params', {}).get('long_periods'), [21, 55])))
        signal_bases_for_mtf_cohesion = [
            'main_force_flow_directionality_D', 'NMFNF_D', 'order_book_imbalance_D',
            'microstructure_efficiency_index_D', 'deception_index_D', 'wash_trade_intensity_D',
            'main_force_conviction_index_D', 'flow_credibility_index_D', deception_context_modulator_signal_name,
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D'
        ]
        all_pre_fetched_slopes_accels = {}
        for signal_base in signal_bases_for_mtf_cohesion:
            for p in all_mtf_periods:
                all_pre_fetched_slopes_accels[f'SLOPE_{p}_{signal_base}'] = self._get_safe_series(df, df, f'SLOPE_{p}_{signal_base}', 0.0, method_name=method_name)
                all_pre_fetched_slopes_accels[f'ACCEL_{p}_{signal_base}'] = self._get_safe_series(df, df, f'ACCEL_{p}_{signal_base}', 0.0, method_name=method_name)
        # 校验所需信号是否存在
        required_signals = [
            'wash_trade_intensity_D', 'deception_index_D', 'main_force_conviction_index_D',
            'flow_credibility_index_D', deception_context_modulator_signal_name,
            'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D'
        ]
        required_signals = list(set(required_signals)) # 去重
        if not self._validate_required_signals(df, required_signals, method_name):
            return pd.Series(0.0, index=df_index, dtype=np.float32)
        # 集中获取原始数据
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name in all_pre_fetched_slopes_accels:
                raw_data_cache[signal_name] = all_pre_fetched_slopes_accels[signal_name]
            else:
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        wash_trade_intensity_raw = raw_data_cache['wash_trade_intensity_D']
        deception_index_raw = raw_data_cache['deception_index_D']
        main_force_conviction_raw = raw_data_cache['main_force_conviction_index_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        deception_context_modulator_raw = raw_data_cache[deception_context_modulator_signal_name]
        deception_lure_long_raw = raw_data_cache['deception_lure_long_intensity_D']
        deception_lure_short_raw = raw_data_cache['deception_lure_short_intensity_D']
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        mtf_cohesion_params = get_param_value(ac_params.get('mtf_cohesion_params'), {})
        mtf_cohesion_enabled = get_param_value(mtf_cohesion_params.get('enabled'), True)
        mtf_cohesion_modulator_sensitivity = get_param_value(mtf_cohesion_params.get('modulator_sensitivity'), 0.5)
        mtf_cohesion_deception_weights = get_param_value(mtf_cohesion_params.get('deception_weights'), {"deception_index": 0.5, "wash_trade_intensity": 0.5})
        # MTF共振因子计算 (仅与诡道相关的)
        deception_cohesion = pd.Series(1.0, index=df_index)
        wash_trade_cohesion = pd.Series(1.0, index=df_index)
        if mtf_cohesion_enabled:
            deception_cohesion = self._calculate_mtf_cohesion_divergence(df, 'deception_index_D', all_mtf_periods[:2], all_mtf_periods[2:], True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
            wash_trade_cohesion = self._calculate_mtf_cohesion_divergence(df, 'wash_trade_intensity_D', all_mtf_periods[:2], all_mtf_periods[2:], False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        # 归一化所有原始信号
        norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights_ff)
        norm_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(deception_context_modulator_raw, df_index, tf_weights=tf_weights_ff)
        norm_deception_lure_long = get_adaptive_mtf_normalized_score(deception_lure_long_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_deception_lure_short = get_adaptive_mtf_normalized_score(deception_lure_short_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 准备NumPy数组，传递给Numba函数
        norm_wash_trade_values = norm_wash_trade.values
        norm_deception_values = norm_deception.values
        norm_conviction_values = norm_conviction.values
        norm_flow_credibility_values = norm_flow_credibility.values
        norm_market_sentiment_values = norm_market_sentiment.values
        norm_deception_lure_long_values = norm_deception_lure_long.values
        norm_deception_lure_short_values = norm_deception_lure_short.values # 传入但可能不直接用于风险计算
        deception_cohesion_mod_values = (1 + deception_cohesion * mtf_cohesion_deception_weights.get('deception_index', 0.5) * mtf_cohesion_modulator_sensitivity).values
        wash_trade_cohesion_mod_values = (1 + wash_trade_cohesion * mtf_cohesion_deception_weights.get('wash_trade_intensity', 0.5) * mtf_cohesion_modulator_sensitivity).values
        # 调用Numba优化后的核心函数
        deception_risk_score_values = _numba_calculate_deception_risk_core(
            norm_wash_trade_values,
            norm_deception_values,
            norm_conviction_values,
            norm_flow_credibility_values,
            norm_market_sentiment_values,
            norm_deception_lure_long_values,
            norm_deception_lure_short_values,
            wash_trade_penalty_sensitivity,
            deception_penalty_sensitivity,
            deception_lure_long_penalty_sensitivity,
            deception_context_sensitivity,
            flow_credibility_threshold,
            deception_cohesion_mod_values,
            wash_trade_cohesion_mod_values
        )
        # 将Numba函数返回的NumPy数组转换回Pandas Series
        deception_risk_score = pd.Series(deception_risk_score_values, index=df_index, dtype=np.float32)
        # if is_debug_enabled_for_method and probe_ts and probe_ts in df_index:
        #     print(f"  -- [资金流层调试] {method_ts.strftime('%Y-%m-%d')}: 资金流诡道风险诊断完成，最终分值: {deception_risk_score.loc[probe_ts]:.4f}")
        return deception_risk_score.astype(np.float32)









