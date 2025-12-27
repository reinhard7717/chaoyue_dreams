# strategies\trend_following\intelligence\fund_flow_intelligence.py
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
from strategies.trend_following.utils import (
    get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, 
    get_adaptive_mtf_normalized_score, load_external_json_config, _robust_geometric_mean
)

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

    def diagnose_fund_flow_states(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V29.0 · 拐点洞察版】资金流情报分析总指挥
        - 核心升级: 新增终极机会信号 SCORE_FF_HARMONY_INFLECTION (资金流和谐拐点)。
        - 信号原理: 基于微积分思想，对顶层“战略态势”信号进行二阶求导。只有当态势的“速度”与“加速度”
                      同时为正时，才确认为一次高置信度的V型反转拐点。旨在捕捉趋势“破晓”的关键瞬间。
        """
        # 直接使用在 __init__ 中加载的配置
        p_conf = self.p_conf_ff
        print(f"FundFlowIntelligence self.debug_params: {self.debug_params}")
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。")
            return {}
        print("启动【V29.0 · 拐点洞察版】资金流情报分析...")
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 1. 计算所有原子公理 ---
        axiom_capital_signature = self._diagnose_axiom_capital_signature(df, norm_window)
        axiom_flow_structure_health = self._diagnose_axiom_flow_structure_health(df, norm_window)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_flow_momentum = self._diagnose_axiom_flow_momentum(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        # 修正：在调用依赖它的方法之前，将 axiom_divergence 存储到 atomic_states
        self.strategy.atomic_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        # 新增：意图纯度公理
        axiom_intent_purity = self._diagnose_axiom_intent_purity(df, norm_window)
        # --- 2. 战略态势的向量合成 (V3.1 · 脆弱性感知版) ---
        print("    -> [资金流层] 正在计算“资金流战略态势 (V3.1 · 脆弱性感知版)”...")
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
        print(f"        [探针] 攻击力量 (矛) 原始分: {attack_base.iloc[-1]:.4f}, 不谐和惩罚: {attack_dissonance_penalty.iloc[-1]:.4f}, 最终攻击分: {attack_score.iloc[-1]:.4f}")
        # 2.2 防御力量 (盾)
        defense_base = (axiom_consensus * defense_group.get('consensus', 0.6) +
                        axiom_flow_structure_health * defense_group.get('flow_health', 0.4))
        defense_dissonance = abs(axiom_consensus - axiom_flow_structure_health)
        defense_dissonance_penalty_sensitivity = get_param_value(defense_group.get('dissonance_penalty_sensitivity'), 0.5)
        defense_dissonance_penalty = np.tanh(defense_dissonance * defense_dissonance_penalty_sensitivity)
        defense_score = defense_base * (1 - defense_dissonance_penalty)
        print(f"        [探针] 防御力量 (盾) 原始分: {defense_base.iloc[-1]:.4f}, 不谐和惩罚: {defense_dissonance_penalty.iloc[-1]:.4f}, 最终防御分: {defense_score.iloc[-1]:.4f}")
        # 2.3 内部协同度调制器 (Internal Harmony Modulator)
        imbalance = abs(attack_score - defense_score)
        imbalance_penalty_sensitivity = get_param_value(harmony_group.get('imbalance_penalty_sensitivity'), 0.8)
        internal_harmony_modulator = 1 - np.tanh(imbalance * imbalance_penalty_sensitivity)
        print(f"        [探针] 矛盾不平衡度: {imbalance.iloc[-1]:.4f}, 内部协同度调制器: {internal_harmony_modulator.iloc[-1]:.4f}")
        # 2.4 情境调节器 (Context Modulator)
        # 确保 flow_credibility_index_D 和 VOLATILITY_INSTABILITY_INDEX_21d_D 存在
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
        print(f"        [探针] 情境调节器基础分: {context_base.iloc[-1]:.4f}, 最终情境调节器: {context_modulator_final.iloc[-1]:.4f}")
        # 2.5 脆弱性放大机制 (Vulnerability Amplification)
        extreme_negative_defense_threshold = get_param_value(defense_group.get('extreme_negative_defense_threshold'), -0.7)
        vulnerability_amplification_sensitivity = get_param_value(defense_group.get('vulnerability_amplification_sensitivity'), 1.5)
        # 计算防御力量低于阈值的程度，clip(lower=0) 确保只有低于阈值时才产生正值
        defense_vulnerability = (extreme_negative_defense_threshold - defense_score).clip(lower=0)
        # 使用 tanh 函数将脆弱性程度映射为 [0, 1) 的惩罚因子
        vulnerability_penalty = np.tanh(defense_vulnerability * vulnerability_amplification_sensitivity)
        print(f"        [探针] 防御脆弱性原始值: {defense_vulnerability.iloc[-1]:.4f}, 脆弱性惩罚因子: {vulnerability_penalty.iloc[-1]:.4f}")
        # 2.6 最终战略态势融合
        posture_core = attack_score * (1 + defense_score) / 2
        strategic_posture_score = (posture_core * internal_harmony_modulator * context_modulator_final * (1 - vulnerability_penalty)).clip(-1, 1)
        print(f"        [探针] 矛盾核心分: {posture_core.iloc[-1]:.4f}, 最终战略态势分: {strategic_posture_score.iloc[-1]:.4f}")
        # --- 3. 和谐拐点计算 ---
        posture_velocity = strategic_posture_score.diff().fillna(0)
        posture_acceleration = posture_velocity.diff().fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_score(posture_velocity, df.index, ascending=True, tf_weights={3:1.0})
        norm_acceleration = get_adaptive_mtf_normalized_score(posture_acceleration, df.index, ascending=True, tf_weights={3:1.0})
        # 核心裁决：速度和加速度必须同时为正
        harmony_inflection_score = (norm_velocity.clip(lower=0) * norm_acceleration.clip(lower=0)).pow(0.5)
        # --- 4. 资金流看涨/看跌背离信号 ---
        # 将所有原子公理存储到 self.strategy.atomic_states，以便 _diagnose_fund_flow_divergence_signals 可以获取
        # axiom_divergence 已经在上面存储，这里不再重复
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        print(f"    -> [资金流情报校验] 计算“资金流内部分歧与意图张力(SCORE_FF_AXIOM_CONVICTION)” 分数：{axiom_conviction.mean():.4f}")
        self.strategy.atomic_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        self.strategy.atomic_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        self.strategy.atomic_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        self.strategy.atomic_states['SCORE_FF_AXIOM_INTENT_PURITY'] = axiom_intent_purity # 新增意图纯度公理
        self.strategy.atomic_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score
        bullish_divergence, bearish_divergence = self._diagnose_fund_flow_divergence_signals(df, norm_window, axiom_divergence)
        # --- 5. 状态赋值 ---
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        all_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        all_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        all_states['SCORE_FF_AXIOM_INTENT_PURITY'] = axiom_intent_purity.astype(np.float32) # 新增意图纯度公理
        all_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score.astype(np.float32)
        all_states['SCORE_FF_HARMONY_INFLECTION'] = harmony_inflection_score.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        print(f"【V29.0 · 拐点洞察版】分析完成，生成 {len(all_states)} 个资金流原子及融合信号。")
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.1 · 效率优化版】资金流公理四：诊断“资金流内部分歧与意图张力”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`，减少重复数据查找。
        """
        print("    -> [资金流层] 正在诊断“资金流内部分歧与意图张力 (V5.1 · 效率优化版)”公理...")
        df_index = df.index
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        ad_params = get_param_value(p_conf_ff.get('axiom_divergence_params'), {})
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
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        # 收集所有需要预取的信号基础名称
        signal_bases_to_prefetch = [
            'NMFNF_D', 'main_force_conviction_index_D', 'net_lg_amount_calibrated_D',
            'retail_net_flow_calibrated_D', 'deception_index_D', 'wash_trade_intensity_D',
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D'
        ]
        for signal_base in signal_bases_to_prefetch:
            for p in divergence_slope_periods:
                col_name = f'SLOPE_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_divergence")
            for p in divergence_accel_periods:
                col_name = f'ACCEL_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_divergence")
        # --- 原始数据获取 (用于探针和计算) ---
        retail_fomo_premium_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        retail_panic_surrender_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_divergence")
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence")
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence")
        mf_activity_ratio_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', 0.0, method_name="_diagnose_axiom_divergence")
        mf_ofi_raw = self._get_safe_series(df, df, 'main_force_ofi_D', 0.0, method_name="_diagnose_axiom_divergence")
        micro_impact_elasticity_raw = self._get_safe_series(df, df, 'micro_impact_elasticity_D', 0.0, method_name="_diagnose_axiom_divergence")
        energy_modulator_signals = {}
        for mod_name, mod_params in energy_injection_context_modulators.items():
            if isinstance(mod_params, dict) and 'signal' in mod_params:
                energy_modulator_signals[mod_name] = self._get_safe_series(df, df, mod_params['signal'], 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_1_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_2_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_3_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_divergence")
        dynamic_evolution_context_modulator_1_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_1_name, 0.0, method_name="_diagnose_axiom_divergence")
        # --- 1. 核心分歧向量 (Core Divergence Vector) ---
        norm_nmfnf_slope_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_nmfnf_accel_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_mf_conviction_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_mf_conviction_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        nmfnf_dynamic_score = (norm_nmfnf_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_nmfnf_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        mf_conviction_dynamic_score = (norm_mf_conviction_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_mf_conviction_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        core_divergence_score = (nmfnf_dynamic_score - mf_conviction_dynamic_score).clip(-1, 1)
        # --- 2. 结构性张力 (Structural Tension) ---
        norm_lg_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_lg_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_retail_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_retail_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        lg_flow_dynamic_score = (norm_lg_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_lg_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        retail_flow_dynamic_score = (norm_retail_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_retail_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        structural_divergence_base = (lg_flow_dynamic_score - retail_flow_dynamic_score)
        retail_modulator = (1 - norm_retail_fomo * retail_sentiment_mod_sensitivity) + (norm_retail_panic * retail_sentiment_mod_sensitivity)
        structural_tension_score = (structural_divergence_base * retail_modulator).clip(-1, 1)
        # --- 3. 诡道意图张力 (Deceptive Intent Tension) ---
        norm_deception_slope_mtf = self._get_mtf_dynamic_score(df, 'deception_index_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_deception_accel_mtf = self._get_mtf_dynamic_score(df, 'deception_index_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_wash_trade_slope_mtf = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_wash_trade_accel_mtf = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        deception_dynamic_score = (norm_deception_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_deception_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        wash_trade_dynamic_score = (norm_wash_trade_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_wash_trade_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(0, 1)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        deceptive_divergence_base = (norm_flow_credibility - deception_dynamic_score.abs() * np.sign(deception_dynamic_score))
        wash_trade_modulator = (1 - wash_trade_dynamic_score * deception_mod_sensitivity)
        deceptive_tension_score = (deceptive_divergence_base * wash_trade_modulator).clip(-1, 1)
        # --- 4. 微观意图张力 (Micro-Flow Intent Tension) ---
        norm_order_book_imbalance = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights=self.tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=self.tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        micro_exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        # 微观动态脉冲
        obi_dynamic_params = micro_intent_dynamic_signals.get('order_book_imbalance_D', {"slope": 0.6, "accel": 0.4, "weight": 0.2})
        # 传递预取数据
        norm_obi_slope_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_obi_accel_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        obi_dynamic_pulse = (norm_obi_slope_mtf * obi_dynamic_params.get('slope', 0.6) + norm_obi_accel_mtf * obi_dynamic_params.get('accel', 0.4)).clip(-1, 1)
        buy_exh_dynamic_params = micro_intent_dynamic_signals.get('buy_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        # 传递预取数据
        norm_buy_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_buy_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        buy_exh_dynamic_pulse = (norm_buy_exh_slope_mtf * buy_exh_dynamic_params.get('slope', 0.5) + norm_buy_exh_accel_mtf * buy_exh_dynamic_params.get('accel', 0.5)).clip(0, 1)
        sell_exh_dynamic_params = micro_intent_dynamic_signals.get('sell_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        # 传递预取数据
        norm_sell_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_sell_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name="_diagnose_axiom_divergence", pre_fetched_data=all_pre_fetched_slopes_accels)
        sell_exh_dynamic_pulse = (norm_sell_exh_slope_mtf * sell_exh_dynamic_params.get('slope', 0.5) + norm_sell_exh_accel_mtf * sell_exh_dynamic_params.get('accel', 0.5)).clip(0, 1)
        micro_dynamic_exhaustion_score = (sell_exh_dynamic_pulse - buy_exh_dynamic_pulse).clip(-1, 1)
        micro_intent_tension_score = (
            norm_order_book_imbalance * micro_intent_tension_signals_weights.get('order_book_imbalance_D', 0.5) +
            micro_exhaustion_score * (micro_intent_tension_signals_weights.get('buy_quote_exhaustion_rate_D', 0.25) + micro_intent_tension_signals_weights.get('sell_quote_exhaustion_rate_D', 0.25)) +
            obi_dynamic_pulse * obi_dynamic_params.get('weight', 0.2) +
            micro_dynamic_exhaustion_score * (buy_exh_dynamic_params.get('weight', 0.15) + sell_exh_dynamic_params.get('weight', 0.15))
        ).clip(-1, 1)
        # --- 5. 能量注入与持续性 (Energy Injection & Persistence) ---
        norm_mf_activity = get_adaptive_mtf_normalized_score(mf_activity_ratio_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_mf_ofi = get_adaptive_mtf_normalized_score(mf_ofi_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_micro_impact_elasticity = get_adaptive_mtf_normalized_score(micro_impact_elasticity_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        energy_injection_base = (
            norm_mf_activity * energy_injection_signals_weights.get('main_force_activity_ratio_D', 0.4) +
            norm_mf_ofi * energy_injection_signals_weights.get('main_force_ofi_D', 0.3) +
            norm_micro_impact_elasticity * energy_injection_signals_weights.get('micro_impact_elasticity_D', 0.3)
        ).clip(0, 1)
        # 情境自适应调制能量注入
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
        # --- 6. 非线性融合与情境自适应权重 ---
        norm_adaptive_weight_modulator_1 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_1_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_adaptive_weight_modulator_2 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_2_raw, df_index, ascending=False, tf_weights=self.tf_weights_ff)
        norm_adaptive_weight_modulator_3 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_3_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        total_base_weight = sum(divergence_component_weights.values())
        if total_base_weight == 0:
            return pd.Series(0.0, index=df.index)
        dynamic_core_divergence_weight = divergence_component_weights.get('core_divergence', 0.3) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility - norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility + norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        dynamic_structural_tension_weight = divergence_component_weights.get('structural_tension', 0.25) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility + norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        dynamic_deceptive_tension_weight = divergence_component_weights.get('deceptive_tension', 0.25) * (1 - norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility - norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        dynamic_micro_intent_tension_weight = divergence_component_weights.get('micro_intent_tension', 0.2) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility + norm_adaptive_weight_modulator_3 * adaptive_weight_sensitivity_sentiment)
        sum_dynamic_weights = dynamic_core_divergence_weight + dynamic_structural_tension_weight + dynamic_deceptive_tension_weight + dynamic_micro_intent_tension_weight
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
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_tension_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        print(f"    -> [资金流层] 资金流内部分歧与意图张力 (final_score): {final_score.mean():.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V6.0 · 意图推断与情境预测版】资金流公理一：诊断“战场控制权”
        - 核心升级1: 宏观资金流质量深化：引入主力资金流方向性、基尼系数等，更精细评估宏观流向品质。
        - 核心升级2: 微观控制力增强：整合市场冲击成本、流动性斜率、扫单强度等，捕捉更细致的微观影响力。
        - 核心升级3: 诡道博弈调制升级：引入诱多/诱空欺骗强度，并实现非对称奖励/惩罚，更智能应对主力诡道。
        - 核心升级4: 非线性微观-宏观交互：改变最终融合方式，允许微观信号非线性地放大或抑制宏观信号。
        - 核心升级5: 动态权重情境扩展：增加趋势活力、结构张力作为动态权重调制器，适应更广泛市场情境。
        - 核心升级6: 详细探针输出：增加print语句，方便调试和理解计算过程。
        """
        print(f"    -> [资金流层] 正在诊断“战场控制权 (V6.0 · 意图推断与情境预测版)”公理...")
        df_index = df.index
        p_conf_ff = self.p_conf_ff
        ac_params = get_param_value(p_conf_ff.get('axiom_consensus_params'), {})
        probe_enabled = get_param_value(ac_params.get('probe_enabled'), False)
        current_probe_date = None
        if probe_enabled: # 只有当probe_enabled为True时才打印
            if current_probe_date:
                print(f"        [探针] 战场控制权诊断启动。探针日期: {current_probe_date.strftime('%Y-%m-%d')}")
            else:
                print(f"        [探针] 战场控制权诊断启动。probe_enabled为True，但当前DataFrame不包含任何指定探针日期。")
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        deception_mod_enabled = get_param_value(ac_params.get('deception_mod_enabled'), True)
        deception_penalty_sensitivity = get_param_value(ac_params.get('deception_penalty_sensitivity'), 0.6)
        wash_trade_penalty_sensitivity = get_param_value(ac_params.get('wash_trade_penalty_sensitivity'), 0.4)
        conviction_threshold_deception = get_param_value(ac_params.get('conviction_threshold_deception'), 0.2)
        flow_credibility_threshold = get_param_value(ac_params.get('flow_credibility_threshold'), 0.5)
        deception_context_modulator_signal_name = get_param_value(ac_params.get('deception_context_modulator_signal'), 'market_sentiment_score_D')
        deception_context_sensitivity = get_param_value(ac_params.get('deception_context_sensitivity'), 0.3)
        deception_lure_long_penalty_sensitivity = get_param_value(ac_params.get('deception_lure_long_penalty_sensitivity'), 0.2)
        deception_lure_short_bonus_sensitivity = get_param_value(ac_params.get('deception_lure_short_bonus_sensitivity'), 0.1)
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
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'order_book_imbalance_D', 'microstructure_efficiency_index_D', 'wash_trade_intensity_D',
            'deception_index_D', 'main_force_conviction_index_D', 'flow_credibility_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name,
            dynamic_weight_modulator_signal_3_name, dynamic_weight_modulator_signal_4_name, dynamic_weight_modulator_signal_5_name,
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            deception_context_modulator_signal_name,
            dynamic_evolution_context_modulator_signal_name,
            'dip_buy_absorption_strength_D', 'dip_sell_pressure_resistance_D',
            'panic_sell_volume_contribution_D', 'panic_buy_absorption_contribution_D',
            'opening_buy_strength_D', 'opening_sell_strength_D',
            'pre_closing_buy_posture_D', 'pre_closing_sell_posture_D',
            'closing_auction_buy_ambush_D', 'closing_auction_sell_ambush_D',
            'main_force_t0_buy_efficiency_D', 'main_force_t0_sell_efficiency_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D',
            'buy_order_book_clearing_rate_D', 'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D',
            'vwap_cross_up_intensity_D', 'vwap_cross_down_intensity_D',
            'main_force_on_peak_sell_flow_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'main_force_flow_directionality_D', 'main_force_flow_gini_D', 'NMFNF_D',
            'market_impact_cost_D', 'liquidity_slope_D', 'liquidity_authenticity_score_D',
            'buy_sweep_intensity_D', 'sell_sweep_intensity_D', 'order_flow_imbalance_score_D',
            'deception_lure_long_intensity_D',             'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_consensus"):
            return pd.Series(0.0, index=df.index)
        # 集中所有原始数据获取
        raw_data_cache = {}
        for signal_name in required_signals:
            raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_consensus")
        # 提取常用信号到局部变量
        main_force_flow_raw = raw_data_cache['main_force_net_flow_calibrated_D']
        retail_flow_raw = raw_data_cache['retail_net_flow_calibrated_D']
        order_book_imbalance_raw = raw_data_cache['order_book_imbalance_D']
        ofi_impact_raw = raw_data_cache['microstructure_efficiency_index_D']
        wash_trade_intensity_raw = raw_data_cache['wash_trade_intensity_D']
        deception_index_raw = raw_data_cache['deception_index_D']
        main_force_conviction_raw = raw_data_cache['main_force_conviction_index_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        volatility_instability_raw = raw_data_cache[dynamic_weight_modulator_signal_1_name]
        flow_slope_raw = raw_data_cache[dynamic_weight_modulator_signal_2_name]
        market_sentiment_raw = raw_data_cache[dynamic_weight_modulator_signal_3_name]
        trend_vitality_raw = raw_data_cache[dynamic_weight_modulator_signal_4_name]
        structural_tension_raw = raw_data_cache[dynamic_weight_modulator_signal_5_name]
        buy_exhaustion_raw = raw_data_cache['buy_quote_exhaustion_rate_D']
        sell_exhaustion_raw = raw_data_cache['sell_quote_exhaustion_rate_D']
        deception_context_modulator_raw = raw_data_cache[deception_context_modulator_signal_name]
        dynamic_evolution_context_modulator_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_name]
        # 新增的原始数据
        main_force_flow_directionality_raw = raw_data_cache['main_force_flow_directionality_D']
        main_force_flow_gini_raw = raw_data_cache['main_force_flow_gini_D']
        nmfnf_raw = raw_data_cache['NMFNF_D']
        market_impact_cost_raw = raw_data_cache['market_impact_cost_D']
        liquidity_slope_raw = raw_data_cache['liquidity_slope_D']
        liquidity_authenticity_raw = raw_data_cache['liquidity_authenticity_score_D']
        buy_sweep_intensity_raw = raw_data_cache['buy_sweep_intensity_D']
        sell_sweep_intensity_raw = raw_data_cache['sell_sweep_intensity_D']
        order_flow_imbalance_score_raw = raw_data_cache['order_flow_imbalance_score_D']
        deception_lure_long_raw = raw_data_cache['deception_lure_long_intensity_D']
        deception_lure_short_raw = raw_data_cache['deception_lure_short_intensity_D']
        if probe_enabled and current_probe_date:
            print(f"        [探针] 原始数据获取完成。")
            print(f"          - main_force_flow_directionality_D: {main_force_flow_directionality_raw.loc[current_probe_date]:.4f}")
            print(f"          - main_force_flow_gini_D: {main_force_flow_gini_raw.loc[current_probe_date]:.4f}")
            print(f"          - NMFNF_D: {nmfnf_raw.loc[current_probe_date]:.4f}")
            print(f"          - market_impact_cost_D: {market_impact_cost_raw.loc[current_probe_date]:.4f}")
            print(f"          - liquidity_slope_D: {liquidity_slope_raw.loc[current_probe_date]:.4f}")
            print(f"          - buy_sweep_intensity_D: {buy_sweep_intensity_raw.loc[current_probe_date]:.4f}")
            print(f"          - deception_lure_long_intensity_D: {deception_lure_long_raw.loc[current_probe_date]:.4f}")
        # --- 1. 宏观资金流质量 (Enhanced Macro Fund Flow Quality) ---
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights_ff)
        norm_main_force_flow_gini_inverted = 1 - get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 基尼系数越低越好，所以反向归一化
        norm_nmfnf_net_flow = get_adaptive_mtf_normalized_bipolar_score(nmfnf_raw, df_index, tf_weights_ff)
        macro_flow_quality_score = (
            norm_main_force_flow_directionality * macro_flow_quality_weights.get('main_force_flow_directionality', 0.3) +
            norm_main_force_flow_gini_inverted * macro_flow_quality_weights.get('main_force_flow_gini_inverted', 0.2) +
            norm_nmfnf_net_flow * macro_flow_quality_weights.get('nmfnf_net_flow', 0.5)
        ).clip(-1, 1)
        if probe_enabled and current_probe_date:
            print(f"        [探针] 宏观资金流质量分数: {macro_flow_quality_score.loc[current_probe_date]:.4f}")
        # --- 2. 微观盘口意图推断 (Enhanced Micro Order Book Intent Inference) ---
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff)
        impact_score = get_adaptive_mtf_normalized_bipolar_score(ofi_impact_raw, df_index, tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        # V6.0 整合新增微观控制质量信号
        norm_market_impact_cost_inverted = 1 - get_adaptive_mtf_normalized_score(market_impact_cost_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 冲击成本越低越好
        norm_liquidity_slope = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_raw, df_index, tf_weights_ff)
        norm_liquidity_authenticity = get_adaptive_mtf_normalized_score(liquidity_authenticity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_sweep_intensity = get_adaptive_mtf_normalized_score(buy_sweep_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_sweep_intensity_inverted = 1 - get_adaptive_mtf_normalized_score(sell_sweep_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 卖方扫单强度越低越好
        norm_order_flow_imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_flow_imbalance_score_raw, df_index, tf_weights_ff)
        micro_control_quality_score = (
            norm_market_impact_cost_inverted * micro_control_quality_weights.get('market_impact_cost_inverted', 0.2) +
            norm_liquidity_slope * micro_control_quality_weights.get('liquidity_slope', 0.2) +
            norm_liquidity_authenticity * micro_control_quality_weights.get('liquidity_authenticity', 0.1) +
            norm_buy_sweep_intensity * micro_control_quality_weights.get('buy_sweep_intensity', 0.2) +
            norm_sell_sweep_intensity_inverted * micro_control_quality_weights.get('sell_sweep_intensity_inverted', 0.2) +
            norm_order_flow_imbalance_score * micro_control_quality_weights.get('order_flow_imbalance', 0.1)
        ).clip(-1, 1)
        # V5.1 整合新增资金指标到微观盘口意图推断 (保持原有逻辑，但现在 micro_control_score_v5_1 融合了更多信号)
        norm_dip_buy_absorption_strength = get_adaptive_mtf_normalized_score(raw_data_cache['dip_buy_absorption_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_panic_buy_absorption_contribution = get_adaptive_mtf_normalized_score(raw_data_cache['panic_buy_absorption_contribution_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_opening_buy_strength = get_adaptive_mtf_normalized_score(raw_data_cache['opening_buy_strength_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_pre_closing_buy_posture = get_adaptive_mtf_normalized_score(raw_data_cache['pre_closing_buy_posture_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_closing_auction_buy_ambush = get_adaptive_mtf_normalized_score(raw_data_cache['closing_auction_buy_ambush_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_buy_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['retail_buy_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
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
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['main_force_sell_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(raw_data_cache['retail_sell_ofi_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
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
        # 融合微观意图和微观控制质量分数
        micro_intent_score = (
            imbalance_score * micro_intent_fusion_weights.get('imbalance', 0.4) +
            impact_score * micro_intent_fusion_weights.get('efficiency', 0.3) +
            exhaustion_score * micro_intent_fusion_weights.get('exhaustion', 0.3) +
            micro_control_score_v5_1 * 0.5 + # 保持原有微观力量融合
            micro_control_quality_score * 0.5 # 新增微观控制质量分数
        ).clip(-1, 1)
        micro_control_modulator = pd.Series(1.0, index=df_index)
        if asymmetric_micro_control_enabled:
            boost_mask = (norm_buy_exhaustion > 0.5) & (norm_sell_exhaustion > 0.5)
            micro_control_modulator.loc[boost_mask] = 1 + (norm_buy_exhaustion.loc[boost_mask] * norm_sell_exhaustion.loc[boost_mask]) * exhaustion_boost_factor
            penalty_mask = (norm_buy_exhaustion < 0.5) & (norm_sell_exhaustion < 0.5)
            micro_control_modulator.loc[penalty_mask] = 1 - (norm_buy_exhaustion.loc[penalty_mask] * norm_sell_exhaustion.loc[penalty_mask]) * exhaustion_penalty_factor
            micro_control_modulator = micro_control_modulator.clip(0.5, 1.5)
        micro_control_score = micro_intent_score * micro_control_modulator
        if probe_enabled and current_probe_date:
            print(f"        [探针] 微观控制质量分数: {micro_control_quality_score.loc[current_probe_date]:.4f}")
            print(f"        [探针] 微观盘口意图分数 (融合后): {micro_intent_score.loc[current_probe_date]:.4f}")
            print(f"        [探针] 微观控制分数 (调制后): {micro_control_score.loc[current_probe_date]:.4f}")
        # --- 3. 诡道博弈深度情境感知与调制 (Enhanced Deceptive Game Integration & Contextual Modulation) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_mod_enabled:
            norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights_ff)
            norm_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(deception_context_modulator_raw, df_index, tf_weights=tf_weights_ff)
            norm_deception_lure_long = get_adaptive_mtf_normalized_score(deception_lure_long_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_deception_lure_short = get_adaptive_mtf_normalized_score(deception_lure_short_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            sentiment_mod_factor = (1 + norm_market_sentiment.abs() * deception_context_sensitivity * np.sign(norm_market_sentiment))
            deception_modulator = deception_modulator * (1 - norm_wash_trade * wash_trade_penalty_sensitivity * sentiment_mod_factor.clip(0.5, 1.5))
            # 诱多惩罚
            bull_trap_mask = (norm_deception > 0)
            deception_modulator.loc[bull_trap_mask] = deception_modulator.loc[bull_trap_mask] * (1 - norm_deception.loc[bull_trap_mask] * deception_penalty_sensitivity * sentiment_mod_factor.loc[bull_trap_mask].clip(0.5, 1.5))
            deception_modulator = deception_modulator * (1 - norm_deception_lure_long * deception_lure_long_penalty_sensitivity)
            # 诱空奖励 (如果主力信念也强，则视为洗盘吸筹)
            bear_trap_mitigation_mask = (norm_deception < 0) & (norm_conviction > conviction_threshold_deception) & (norm_flow_credibility > flow_credibility_threshold)
            deception_modulator.loc[bear_trap_mitigation_mask] = deception_modulator.loc[bear_trap_mitigation_mask] * (1 + norm_deception.loc[bear_trap_mitigation_mask].abs() * deception_penalty_sensitivity * 0.5 * sentiment_mod_factor.loc[bear_trap_mitigation_mask].clip(0.5, 1.5))
            # 诱空奖励，如果主力信念为正，则增强
            deception_modulator = deception_modulator * (1 + norm_deception_lure_short * deception_lure_short_bonus_sensitivity * norm_conviction.clip(lower=0))
            deception_modulator = deception_modulator * (1 + (norm_flow_credibility - 0.5) * 0.5)
            low_credibility_mask = (norm_flow_credibility < flow_credibility_threshold)
            deception_modulator.loc[low_credibility_mask] = deception_modulator.loc[low_credibility_mask] * (norm_flow_credibility.loc[low_credibility_mask] / flow_credibility_threshold).clip(0.1, 1.0)
            deception_modulator = deception_modulator.clip(0.01, 2.0)
        if probe_enabled and current_probe_date:
            print(f"        [探针] 诡道调制器: {deception_modulator.loc[current_probe_date]:.4f}")
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
            dynamic_macro_weight = dynamic_macro_weight / sum_dynamic_weights
            dynamic_micro_weight = dynamic_micro_weight / sum_dynamic_weights
            dynamic_macro_weight = dynamic_macro_weight.clip(0.1, 0.9)
            dynamic_micro_weight = dynamic_micro_weight.clip(0.1, 0.9)
        if probe_enabled and current_probe_date:
            print(f"        [探针] 动态宏观权重: {dynamic_macro_weight.loc[current_probe_date]:.4f}")
            print(f"        [探针] 动态微观权重: {dynamic_micro_weight.loc[current_probe_date]:.4f}")
        # --- 5. 融合基础战场控制权 (V6.0 非线性微观-宏观交互) ---
        # 将宏观流向分数和微观控制分数转换为 [0, 1] 范围，以便进行几何平均
        macro_score_unipolar = (macro_flow_quality_score + 1) / 2
        micro_score_unipolar = (micro_control_score + 1) / 2
        # 健壮的加权几何平均
        # 避免 log(0) 错误，将接近0的值替换为一个小正数
        macro_score_unipolar_safe = macro_score_unipolar.clip(lower=1e-9)
        micro_score_unipolar_safe = micro_score_unipolar.clip(lower=1e-9)
        # 计算加权对数和
        weighted_log_sum = (
            np.log(macro_score_unipolar_safe) * dynamic_macro_weight +
            np.log(micro_score_unipolar_safe) * dynamic_micro_weight
        )
        # 计算总有效权重
        total_effective_weight = dynamic_macro_weight + dynamic_micro_weight
        # 避免除以零
        total_effective_weight_safe = total_effective_weight.replace(0, 1e-9)
        # 计算几何平均，并转换回 [-1, 1] 范围
        geometric_mean_unipolar = np.exp(weighted_log_sum / total_effective_weight_safe)
        base_battlefield_control_score = (geometric_mean_unipolar * 2 - 1).clip(-1, 1)
        # 应用诡道调制器
        base_battlefield_control_score = base_battlefield_control_score * deception_modulator
        # 应用非线性交互指数
        base_battlefield_control_score = np.tanh(base_battlefield_control_score * micro_macro_interaction_exponent)
        if probe_enabled and current_probe_date:
            print(f"        [探针] 宏观流向质量分数: {macro_flow_quality_score.loc[current_probe_date]:.4f}")
            print(f"        [探针] 微观控制分数: {micro_control_score.loc[current_probe_date]:.4f}")
            print(f"        [探针] 基础战场控制分数 (融合前): {base_battlefield_control_score.loc[current_probe_date]:.4f}")
            print(f"        [探针] 诡道调制器: {deception_modulator.loc[current_probe_date]:.4f}")
            print(f"        [探针] 基础战场控制分数 (调制后): {base_battlefield_control_score.loc[current_probe_date]:.4f}")
        # --- 6. 战场控制权动态演化与前瞻性增强 (Dynamic Evolution & Foresight Enhancement) ---
        smoothed_base_score = base_battlefield_control_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.2) * (1 + norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        dynamic_base_weight = dynamic_evolution_base_weights.get('base_score', 0.6) * (1 - norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        total_dynamic_weights = dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        # 再次使用健壮的加权几何平均进行最终融合
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
        final_score = _robust_geometric_mean(final_score_components, final_score_weights, df_index)
        final_score = (final_score * 2 - 1).clip(-1, 1)
        if probe_enabled and current_probe_date:
            print(f"        [探针] 最终战场控制分数: {final_score.loc[current_probe_date]:.4f}")
            print(f"        [探针] 战场控制权诊断完成。")
        return final_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.2 · 效率优化版】资金流公理二：诊断“信念韧性”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        """
        print(f"    -> [资金流层] 正在诊断 资金流公理二：诊断“信念韧性”")
        df_index = df.index
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
        transmission_efficiency_weights = get_param_value(ac_params.get('transmission_efficiency_weights'), {'main_force_execution_alpha_slope_5': 0.2, 'main_force_execution_alpha_slope_13': 0.1, 'flow_efficiency_slope_5': 0.2, 'flow_efficiency_slope_13': 0.1, 'intraday_price_impact': 0.2, 'large_order_pressure': 0.1, 'intraday_vwap_deviation': 0.1})
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
            'SLOPE_5_flow_efficiency_index_D', 'SLOPE_13_flow_efficiency_index_D',
            'micro_price_impact_asymmetry_D', 'large_order_pressure_D', 'intraday_vwap_div_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name, dynamic_weight_modulator_signal_3_name, dynamic_weight_modulator_signal_4_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name,
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_conviction"):
            return pd.Series(0.0, index=df.index)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        # 收集所有需要预取的信号基础名称和周期
        slope_periods = [5, 13, 21] # 假设这些是所有斜率信号的最大周期
        accel_periods = [5, 13, 21] # 假设这些是所有加速度信号的最大周期
        signal_bases_to_prefetch_slope = [
            'main_force_conviction_index_D', 'SMART_MONEY_HM_NET_BUY_D', 'deception_index_D',
            'wash_trade_intensity_D', 'main_force_execution_alpha_D', 'flow_efficiency_index_D'
        ]
        for signal_base in signal_bases_to_prefetch_slope:
            for p in slope_periods:
                col_name = f'SLOPE_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_conviction")
        # 针对 _get_mtf_dynamic_score 内部调用的 ACCEL 信号，虽然这里没有直接使用，但为了完整性，可以预取
        # 实际上, _get_mtf_dynamic_score 内部会根据 is_accel 参数构建列名，这里只需要确保原始数据存在即可
        # 但在这个方法中，_get_mtf_dynamic_score 并没有被调用来获取 ACCEL 信号，所以这里不需要预取 ACCEL
        # _get_mtf_dynamic_score 内部会根据 is_accel 参数构建列名，所以这里需要预取 ACCEL 信号
        signal_bases_to_prefetch_accel = [
            'main_force_t0_efficiency_D', 'main_force_slippage_index_D', 'main_force_execution_alpha_D'
        ]
        for signal_base in signal_bases_to_prefetch_accel:
            for p in accel_periods:
                col_name = f'ACCEL_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_conviction")
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            # 避免重复获取已在 all_pre_fetched_slopes_accels 中的斜率/加速度数据
            if signal_name not in all_pre_fetched_slopes_accels:
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_conviction")
        # Core Conviction
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
        # Deceptive Resilience
        deception_slope_5_raw = raw_data_cache.get('SLOPE_5_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_deception_index_D'))
        deception_slope_13_raw = raw_data_cache.get('SLOPE_13_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_deception_index_D'))
        deception_slope_21_raw = raw_data_cache.get('SLOPE_21_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_21_deception_index_D'))
        wash_trade_slope_5_raw = raw_data_cache.get('SLOPE_5_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_5_wash_trade_intensity_D'))
        wash_trade_slope_13_raw = raw_data_cache.get('SLOPE_13_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_13_wash_trade_intensity_D'))
        wash_trade_slope_21_raw = raw_data_cache.get('SLOPE_21_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_21_wash_trade_intensity_D'))
        main_force_cost_advantage_raw = raw_data_cache[resilience_context_modulator_signal_2_name]
        market_sentiment_raw = raw_data_cache[resilience_context_modulator_signal_1_name]
        market_liquidity_raw = raw_data_cache[resilience_context_modulator_signal_3_name]
        # Transmission Efficiency
        mf_exec_alpha_slope_5_raw = raw_data_cache.get('SLOPE_5_main_force_execution_alpha_D', all_pre_fetched_slopes_accels.get('SLOPE_5_main_force_execution_alpha_D'))
        mf_exec_alpha_slope_13_raw = raw_data_cache.get('SLOPE_13_main_force_execution_alpha_D', all_pre_fetched_slopes_accels.get('SLOPE_13_main_force_execution_alpha_D'))
        flow_efficiency_slope_5_raw = raw_data_cache.get('SLOPE_5_flow_efficiency_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_flow_efficiency_index_D'))
        flow_efficiency_slope_13_raw = raw_data_cache.get('SLOPE_13_flow_efficiency_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_flow_efficiency_index_D'))
        intraday_price_impact_raw = raw_data_cache['micro_price_impact_asymmetry_D']
        large_order_pressure_raw = raw_data_cache['large_order_pressure_D']
        intraday_vwap_deviation_raw = raw_data_cache['intraday_vwap_div_index_D']
        # Dynamic Weighting & Evolution Context
        volatility_instability_raw = raw_data_cache[dynamic_weight_modulator_signal_1_name]
        market_sentiment_dw_raw = raw_data_cache[dynamic_weight_modulator_signal_2_name]
        market_liquidity_dw_raw = raw_data_cache[dynamic_weight_modulator_signal_3_name]
        trend_vitality_dw_raw = raw_data_cache[dynamic_weight_modulator_signal_4_name]
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        # V4.1 获取新增资金指标
        rally_sell_distribution_intensity_raw = raw_data_cache['rally_sell_distribution_intensity_D']
        rally_buy_support_weakness_raw = raw_data_cache['rally_buy_support_weakness_D']
        main_force_buy_ofi_raw = raw_data_cache['main_force_buy_ofi_D']
        main_force_sell_ofi_raw = raw_data_cache['main_force_sell_ofi_D']
        retail_buy_ofi_raw = raw_data_cache['retail_buy_ofi_D']
        retail_sell_ofi_raw = raw_data_cache['retail_sell_ofi_D']
        wash_trade_buy_volume_raw = raw_data_cache['wash_trade_buy_volume_D']
        wash_trade_sell_volume_raw = raw_data_cache['wash_trade_sell_volume_D']
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
            norm_mf_conviction_slope_5 * core_conviction_weights.get('main_force_conviction_slope_5', 0.2) +
            norm_mf_conviction_slope_13 * core_conviction_weights.get('main_force_conviction_slope_13', 0.2) +
            norm_mf_conviction_slope_21 * core_conviction_weights.get('main_force_conviction_slope_21', 0.1) +
            norm_sm_net_buy_slope_5 * core_conviction_weights.get('smart_money_net_buy_slope_5', 0.15) +
            norm_sm_net_buy_slope_13 * core_conviction_weights.get('smart_money_net_buy_slope_13', 0.15) +
            norm_flow_credibility * core_conviction_weights.get('flow_credibility', 0.1) +
            norm_intraday_large_order_flow * core_conviction_weights.get('intraday_large_order_flow', 0.1)
        ).clip(-1, 1)
        norm_rally_sell_distribution_intensity = get_adaptive_mtf_normalized_score(rally_sell_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_rally_buy_support_weakness = get_adaptive_mtf_normalized_score(rally_buy_support_weakness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(main_force_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(main_force_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(retail_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(retail_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(wash_trade_buy_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(wash_trade_sell_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
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
            deceptive_resilience_modulator = deceptive_resilience_modulator * (1 - norm_wash_trade_multi_tf * wash_trade_penalty_factor * conviction_feedback_mod.clip(0.5, 1.5) * sentiment_mod.clip(0.5, 1.5) * liquidity_mod.clip(0.5, 1.5))
            bull_trap_mask = (norm_deception_multi_tf > 0)
            deceptive_resilience_modulator.loc[bull_trap_mask] = deceptive_resilience_modulator.loc[bull_trap_mask] * (1 - norm_deception_multi_tf.loc[bull_trap_mask] * deception_penalty_factor * conviction_feedback_mod.loc[bull_trap_mask].clip(0.5, 1.5) * sentiment_mod.loc[bull_trap_mask].clip(0.5, 1.5) * liquidity_mod.loc[bull_trap_mask].clip(0.5, 1.5))
            bear_trap_resilience_mask = (norm_deception_multi_tf < 0) & (norm_cost_advantage > 0.5) & (norm_market_sentiment < -0.5) & (norm_market_liquidity < 0.5) & (core_conviction_score > 0.2)
            deceptive_resilience_modulator.loc[bear_trap_resilience_mask] = deceptive_resilience_modulator.loc[bear_trap_resilience_mask] * (1 + norm_deception_multi_tf.loc[bear_trap_resilience_mask].abs() * deception_penalty_factor * cost_advantage_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5) * (1 - liquidity_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5)))
            deceptive_resilience_modulator = deceptive_resilience_modulator.clip(0.01, 2.0)
        # --- 3. 信念传导效率 (Conviction Transmission Efficiency) ---
        norm_mf_exec_alpha_slope_5 = get_adaptive_mtf_normalized_bipolar_score(mf_exec_alpha_slope_5_raw, df_index, tf_weights_ff)
        norm_mf_exec_alpha_slope_13 = get_adaptive_mtf_normalized_bipolar_score(mf_exec_alpha_slope_13_raw, df_index, tf_weights_ff)
        norm_flow_efficiency_slope_5 = get_adaptive_mtf_normalized_bipolar_score(flow_efficiency_slope_5_raw, df_index, tf_weights_ff)
        norm_flow_efficiency_slope_13 = get_adaptive_mtf_normalized_bipolar_score(flow_efficiency_slope_13_raw, df_index, tf_weights_ff)
        norm_intraday_price_impact = get_adaptive_mtf_normalized_bipolar_score(intraday_price_impact_raw, df_index, tf_weights=tf_weights_ff)
        norm_large_order_pressure = get_adaptive_mtf_normalized_score(large_order_pressure_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_intraday_vwap_deviation = get_adaptive_mtf_normalized_score(intraday_vwap_deviation_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        efficiency_ratio = (norm_intraday_price_impact + (1 - norm_intraday_vwap_deviation)) / (norm_large_order_pressure + 1e-9)
        norm_efficiency_ratio = get_adaptive_mtf_normalized_score(efficiency_ratio, df_index, ascending=True, tf_weights=tf_weights_ff)
        transmission_efficiency_score = (
            norm_mf_exec_alpha_slope_5 * transmission_efficiency_weights.get('main_force_execution_alpha_slope_5', 0.2) +
            norm_mf_exec_alpha_slope_13 * transmission_efficiency_weights.get('main_force_execution_alpha_slope_13', 0.1) +
            norm_flow_efficiency_slope_5 * transmission_efficiency_weights.get('flow_efficiency_slope_5', 0.2) +
            norm_flow_efficiency_slope_13 * transmission_efficiency_weights.get('flow_efficiency_slope_13', 0.1) +
            norm_intraday_price_impact * transmission_efficiency_weights.get('intraday_price_impact', 0.2) +
            norm_large_order_pressure * transmission_efficiency_weights.get('large_order_pressure', 0.1) +
            norm_intraday_vwap_deviation * transmission_efficiency_weights.get('intraday_vwap_deviation', 0.1)
        ).clip(-1, 1)
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
            dynamic_core_conviction_weight = dynamic_core_conviction_weight / sum_dynamic_weights
            dynamic_deceptive_resilience_weight = dynamic_deceptive_resilience_weight / sum_dynamic_weights
            dynamic_transmission_efficiency_weight = dynamic_transmission_efficiency_weight / sum_dynamic_weights
            dynamic_core_conviction_weight = dynamic_core_conviction_weight.clip(0.1, 0.8)
            dynamic_deceptive_resilience_weight = dynamic_deceptive_resilience_weight.clip(0.1, 0.8)
            dynamic_transmission_efficiency_weight = dynamic_transmission_efficiency_weight.clip(0.1, 0.8)
        # --- 5. 融合基础信念分数 (V4.0 非线性建模) ---
        base_conviction_score = np.tanh(
            core_conviction_score * dynamic_core_conviction_weight * deceptive_resilience_modulator +
            transmission_efficiency_score * dynamic_transmission_efficiency_weight
        ).clip(-1, 1)
        # --- 6. 信念演化趋势与前瞻性增强 (Conviction Evolution & Foresight Enhancement) ---
        smoothed_base_score = base_conviction_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_1_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_2_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_conviction_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        print(f"    -> [资金流层] 信念韧性 (final_score): {final_score.mean():.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V6.2 · 效率优化版】资金流公理三：诊断“资金流纯度与动能”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        """
        print(f"    -> [资金流层] 正在诊断 资金流公理三：诊断“资金流纯度与动能”")
        df_index = df.index
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
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_momentum"):
            return pd.Series(0.0, index=df.index)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        # 收集所有需要预取的信号基础名称和周期
        slope_periods_all = list(set([5, 13, 21, 55])) # 包含所有可能用到的斜率周期
        accel_periods_all = list(set([5, 13, 21, 55])) # 包含所有可能用到的加速度周期
        signal_bases_to_prefetch_slope = [
            'NMFNF_D', 'SMART_MONEY_HM_NET_BUY_D', 'wash_trade_intensity_D', 'deception_index_D',
            'order_book_liquidity_supply_D', 'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D',
            'retail_net_flow_calibrated_D'
        ]
        for signal_base in signal_bases_to_prefetch_slope:
            for p in slope_periods_all:
                col_name = f'SLOPE_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        signal_bases_to_prefetch_accel = [
            'NMFNF_D', 'SMART_MONEY_HM_NET_BUY_D', 'net_lg_amount_calibrated_D',
            'net_xl_amount_calibrated_D', 'retail_net_flow_calibrated_D'
        ]
        for signal_base in signal_bases_to_prefetch_accel:
            for p in accel_periods_all:
                col_name = f'ACCEL_{p}_{signal_base}'
                all_pre_fetched_slopes_accels[col_name] = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name not in all_pre_fetched_slopes_accels: # 避免重复获取
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        # 基础动能深化
        nmfnf_slope_5_raw = raw_data_cache.get('SLOPE_5_NMFNF_D', all_pre_fetched_slopes_accels.get('SLOPE_5_NMFNF_D'))
        nmfnf_slope_13_raw = raw_data_cache.get('SLOPE_13_NMFNF_D', all_pre_fetched_slopes_accels.get('SLOPE_13_NMFNF_D'))
        nmfnf_slope_21_raw = raw_data_cache.get('SLOPE_21_NMFNF_D', all_pre_fetched_slopes_accels.get('SLOPE_21_NMFNF_D'))
        nmfnf_slope_55_raw = raw_data_cache.get('SLOPE_55_NMFNF_D', all_pre_fetched_slopes_accels.get('SLOPE_55_NMFNF_D'))
        nmfnf_accel_5_raw = raw_data_cache.get('ACCEL_5_NMFNF_D', all_pre_fetched_slopes_accels.get('ACCEL_5_NMFNF_D'))
        nmfnf_accel_13_raw = raw_data_cache.get('ACCEL_13_NMFNF_D', all_pre_fetched_slopes_accels.get('ACCEL_13_NMFNF_D'))
        nmfnf_accel_21_raw = raw_data_cache.get('ACCEL_21_NMFNF_D', all_pre_fetched_slopes_accels.get('ACCEL_21_NMFNF_D'))
        nmfnf_accel_55_raw = raw_data_cache.get('ACCEL_55_NMFNF_D', all_pre_fetched_slopes_accels.get('ACCEL_55_NMFNF_D'))
        sm_net_buy_slope_5_raw = raw_data_cache.get('SLOPE_5_SMART_MONEY_HM_NET_BUY_D', all_pre_fetched_slopes_accels.get('SLOPE_5_SMART_MONEY_HM_NET_BUY_D'))
        sm_net_buy_accel_5_raw = raw_data_cache.get('ACCEL_5_SMART_MONEY_HM_NET_BUY_D', all_pre_fetched_slopes_accels.get('ACCEL_5_SMART_MONEY_HM_NET_BUY_D'))
        # 诡道纯度精修
        wash_trade_slope_5_raw = raw_data_cache.get('SLOPE_5_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_5_wash_trade_intensity_D'))
        wash_trade_slope_13_raw = raw_data_cache.get('SLOPE_13_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_13_wash_trade_intensity_D'))
        wash_trade_slope_21_raw = raw_data_cache.get('SLOPE_21_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_21_wash_trade_intensity_D'))
        deception_slope_5_raw = raw_data_cache.get('SLOPE_5_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_deception_index_D'))
        deception_slope_13_raw = raw_data_cache.get('SLOPE_13_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_deception_index_D'))
        deception_slope_21_raw = raw_data_cache.get('SLOPE_21_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_21_deception_index_D'))
        main_force_conviction_raw = raw_data_cache[purity_context_modulator_signal_1_name]
        flow_credibility_raw = raw_data_cache[purity_context_modulator_signal_2_name]
        main_force_flow_gini_raw = raw_data_cache[purity_context_modulator_signal_3_name]
        retail_fomo_premium_raw = raw_data_cache[purity_context_modulator_signal_4_name]
        purity_auxiliary_raw = raw_data_cache[purity_auxiliary_signal_name]
        # 环境感知增强
        liquidity_supply_raw = raw_data_cache['order_book_liquidity_supply_D']
        liquidity_slope_5_raw = raw_data_cache.get('SLOPE_5_order_book_liquidity_supply_D', all_pre_fetched_slopes_accels.get('SLOPE_5_order_book_liquidity_supply_D'))
        liquidity_slope_13_raw = raw_data_cache.get('SLOPE_13_order_book_liquidity_supply_D', all_pre_fetched_slopes_accels.get('SLOPE_13_order_book_liquidity_supply_D'))
        liquidity_impact_raw = raw_data_cache[liquidity_impact_signal_name]
        volatility_instability_raw = raw_data_cache[environment_context_signal_1_name]
        trend_vitality_raw = raw_data_cache[environment_context_signal_2_name]
        price_volume_entropy_raw = raw_data_cache[environment_context_signal_3_name]
        # 结构洞察升级
        lg_flow_slope_5_raw = raw_data_cache.get('SLOPE_5_net_lg_amount_calibrated_D', all_pre_fetched_slopes_accels.get('SLOPE_5_net_lg_amount_calibrated_D'))
        lg_flow_accel_5_raw = raw_data_cache.get('ACCEL_5_net_lg_amount_calibrated_D', all_pre_fetched_slopes_accels.get('ACCEL_5_net_lg_amount_calibrated_D'))
        xl_flow_slope_5_raw = raw_data_cache.get('SLOPE_5_net_xl_amount_calibrated_D', all_pre_fetched_slopes_accels.get('SLOPE_5_net_xl_amount_calibrated_D'))
        xl_flow_accel_5_raw = raw_data_cache.get('ACCEL_5_net_xl_amount_calibrated_D', all_pre_fetched_slopes_accels.get('ACCEL_5_net_xl_amount_calibrated_D'))
        retail_flow_slope_5_raw = raw_data_cache.get('SLOPE_5_retail_net_flow_calibrated_D', all_pre_fetched_slopes_accels.get('SLOPE_5_retail_net_flow_calibrated_D'))
        retail_flow_accel_5_raw = raw_data_cache.get('ACCEL_5_retail_net_flow_calibrated_D', all_pre_fetched_slopes_accels.get('ACCEL_5_retail_net_flow_calibrated_D'))
        main_force_flow_directionality_raw = raw_data_cache['main_force_flow_directionality_D']
        flow_quality_raw = raw_data_cache[flow_quality_signal_name]
        retail_dominance_raw = raw_data_cache[retail_dominance_signal_name]
        # 动态融合优化
        dynamic_evolution_context_modulator_1_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_1_name]
        dynamic_evolution_context_modulator_2_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_2_name]
        dynamic_evolution_context_modulator_3_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_3_name]
        dynamic_evolution_context_modulator_4_raw = raw_data_cache[dynamic_evolution_context_modulator_signal_4_name]
        # V6.1 获取新增资金指标
        rally_sell_distribution_intensity_raw = raw_data_cache['rally_sell_distribution_intensity_D']
        rally_buy_support_weakness_raw = raw_data_cache['rally_buy_support_weakness_D']
        main_force_buy_ofi_raw = raw_data_cache['main_force_buy_ofi_D']
        main_force_sell_ofi_raw = raw_data_cache['main_force_sell_ofi_D']
        retail_buy_ofi_raw = raw_data_cache['retail_buy_ofi_D']
        retail_sell_ofi_raw = raw_data_cache['retail_sell_ofi_D']
        wash_trade_buy_volume_raw = raw_data_cache['wash_trade_buy_volume_D']
        wash_trade_sell_volume_raw = raw_data_cache['wash_trade_sell_volume_D']
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
        # --- 2. 诡道纯度精修 (Refined Purity Filter) ---
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
            norm_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_main_force_flow_gini = get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
            norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_purity_auxiliary = get_adaptive_mtf_normalized_score(purity_auxiliary_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            conviction_mod = (1 + norm_main_force_conviction.abs() * purity_context_sensitivity_conviction * np.sign(norm_main_force_conviction))
            credibility_mod = (1 + (norm_flow_credibility - 0.5) * purity_context_sensitivity_credibility)
            gini_mod = (1 + (norm_main_force_flow_gini - 0.5) * purity_context_sensitivity_gini)
            fomo_mod = (1 - norm_retail_fomo_premium * purity_context_sensitivity_fomo)
            purity_modulator = purity_modulator * (1 - norm_wash_trade_multi_tf * purity_penalty_factor_wash_trade * conviction_mod.clip(0.5, 1.5) * credibility_mod.clip(0.5, 1.5) * gini_mod.clip(0.5, 1.5) * fomo_mod.clip(0.5, 1.5))
            bull_trap_mask = (norm_deception_multi_tf > 0)
            purity_modulator.loc[bull_trap_mask] = purity_modulator.loc[bull_trap_mask] * (1 - norm_deception_multi_tf.loc[bull_trap_mask] * purity_penalty_factor_deception * conviction_mod.loc[bull_trap_mask].clip(0.5, 1.5) * credibility_mod.loc[bull_trap_mask].clip(0.5, 1.5) * gini_mod.loc[bull_trap_mask].clip(0.5, 1.5) * fomo_mod.loc[bull_trap_mask].clip(0.5, 1.5))
            benign_mask = (norm_wash_trade_multi_tf < 0.5) & (norm_deception_multi_tf < 0.5) & (norm_main_force_conviction > 0.5) & (norm_flow_credibility > 0.5) & (norm_main_force_flow_gini < 0.5) & (norm_purity_auxiliary > 0.5)
            purity_modulator.loc[benign_mask] = purity_modulator.loc[benign_mask] * (1 + purity_mitigation_factor * norm_purity_auxiliary.loc[benign_mask].clip(0.5, 1.5))
            purity_modulator = purity_modulator.clip(0.01, 2.0)
        # --- 3. 环境感知增强 (Enhanced Contextual Modulator) ---
        context_modulator = pd.Series(1.0, index=df_index)
        if contextual_modulator_enabled:
            norm_liquidity_supply = get_adaptive_mtf_normalized_score(liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_liquidity_slope_5 = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_5_raw, df_index, tf_weights=tf_weights_ff)
            norm_liquidity_slope_13 = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_13_raw, df_index, tf_weights=tf_weights_ff)
            norm_liquidity_slope_multi_tf = (
                norm_liquidity_slope_5 * liquidity_slope_weights.get('slope_5', 0.6) +
                norm_liquidity_slope_13 * liquidity_slope_weights.get('slope_13', 0.4)
            ).clip(-1, 1)
            norm_liquidity_impact = get_adaptive_mtf_normalized_bipolar_score(liquidity_impact_raw, df_index, tf_weights=tf_weights_ff)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_price_volume_entropy = get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
            level_mod = (1 + (norm_liquidity_supply - 0.5) * liquidity_mod_sensitivity_level)
            slope_mod = (1 + norm_liquidity_slope_multi_tf * liquidity_mod_sensitivity_slope)
            impact_mod = (1 + norm_liquidity_impact)
            volatility_mod = (1 - norm_volatility_instability * environment_mod_sensitivity_volatility)
            trend_mod = (1 + norm_trend_vitality * environment_mod_sensitivity_trend_vitality)
            entropy_mod = (1 + norm_price_volume_entropy * environment_mod_sensitivity_entropy)
            context_modulator = level_mod * slope_mod * impact_mod * volatility_mod * trend_mod * entropy_mod
            context_modulator = context_modulator.clip(0.5, 2.0)
        # --- 4. 结构洞察升级 (Upgraded Structural Momentum) ---
        norm_lg_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(lg_flow_slope_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_lg_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(lg_flow_accel_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_xl_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(xl_flow_slope_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_xl_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(xl_flow_accel_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_retail_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(retail_flow_slope_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_retail_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(retail_flow_accel_5_raw, df_index, tf_weights=tf_weights_ff)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights=tf_weights_ff)
        norm_flow_quality = get_adaptive_mtf_normalized_score(flow_quality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_dominance = get_adaptive_mtf_normalized_score(retail_dominance_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        large_order_flow_momentum = (
            norm_lg_flow_slope_5 * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) +
            norm_lg_flow_accel_5 * structural_momentum_weights.get('large_order_flow_accel_5', 0.15) +
            norm_xl_flow_slope_5 * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) +
            norm_xl_flow_accel_5 * structural_momentum_weights.get('large_order_flow_accel_5', 0.15)
        ).clip(-1, 1)
        retail_flow_momentum = (
            norm_retail_flow_slope_5 * structural_momentum_weights.get('retail_flow_slope_5', -0.1) +
            norm_retail_flow_accel_5 * structural_momentum_weights.get('retail_flow_accel_5', -0.05)
        ).clip(-1, 1)
        structural_momentum_score = (
            large_order_flow_momentum * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) +
            norm_main_force_flow_directionality * structural_momentum_weights.get('main_force_flow_directionality', 0.2) +
            norm_flow_quality * structural_momentum_weights.get('main_force_flow_gini', 0.15) +
            retail_flow_momentum * structural_momentum_weights.get('retail_flow_slope_5', -0.1) +
            norm_retail_dominance * structural_momentum_weights.get('retail_flow_dominance', -0.15)
        ).clip(-1, 1)
        norm_rally_sell_distribution_intensity = get_adaptive_mtf_normalized_score(rally_sell_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_rally_buy_support_weakness = get_adaptive_mtf_normalized_score(rally_buy_support_weakness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(main_force_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(main_force_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(retail_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(retail_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(wash_trade_buy_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(wash_trade_sell_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
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
        # --- 5. 融合基础动能、纯度、环境和结构性动能 ---
        base_flow_momentum_score = (
            (base_momentum_score.add(1)/2).pow(0.3) *
            (purity_modulator.add(1)/2).pow(0.2) *
            (context_modulator.add(1)/2).pow(0.2) *
            (structural_momentum_score.add(1)/2).pow(0.3)
        ).pow(1 / (0.3 + 0.2 + 0.2 + 0.3)) * 2 - 1
        # --- 6. 动能演化趋势与前瞻性增强 (Momentum Evolution & Foresight Enhancement) ---
        smoothed_base_score = base_flow_momentum_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_1_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_2_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_3 = get_adaptive_mtf_normalized_bipolar_score(dynamic_evolution_context_modulator_3_raw, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_4 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_4_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
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
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score = (
            (base_flow_momentum_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        print(f"    -> [资金流层] 资金流纯度与动能 (final_score): {final_score.mean():.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_capital_signature(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.5 · 命名修正与效率优化版】资金流公理五：诊断“资本属性”
        - 核心优化: 预先获取所有斜率和加速度数据，并通过 `pre_fetched_data` 参数传递给 `_get_mtf_dynamic_score`。
                    集中所有其他原始数据获取操作，减少重复的 `_get_safe_series` 调用。
        - 错误修复: 修正了 `SLOPE_5_SLOPE_5_net_lg_amount_calibrated_D` 命名错误，统一使用 `NMFNF_D` 及其衍生信号。
                    修复了 `retail_fomo_premium_index_D` 和 `retail_panic_surrender_index_D` 信号未被正确缓存导致的 KeyError。
        """
        print(f"    -> [资金流层] 正在诊断 资金流公理五：诊断“资本属性”")
        df_index = df.index
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        acs_params = get_param_value(p_conf_ff.get('axiom_capital_signature_params'), {})
        patient_capital_weights = get_param_value(acs_params.get('patient_capital_weights'), {"mtf_flow_persistence": 0.25, "cost_efficiency": 0.2, "covertness_anti_recon": 0.2, "chip_structure_control": 0.2, "flow_structure_resilience": 0.15})
        agile_capital_weights = get_param_value(acs_params.get('agile_capital_weights'), {"ofi_impact_directionality": 0.25, "emotion_driven_risk_appetite": 0.2, "game_efficiency_utilization": 0.2, "short_term_explosiveness": 0.2, "theme_chasing": 0.15})
        mtf_periods_patient_flow = get_param_value(acs_params.get('mtf_periods_patient_flow'), {"short": [5, 13], "long": [21, 55]})
        mtf_periods_agile_ofi = get_param_value(acs_params.get('mtf_periods_agile_ofi'), {"short": [5, 13], "long": [21]})
        capital_context_modulator_sensitivity = get_param_value(acs_params.get('capital_context_modulator_sensitivity'), {"liquidity": 0.2, "volatility": 0.3, "sentiment": 0.1, "trend_vitality": 0.2, "market_phase": 0.15, "risk_appetite": 0.1})
        fusion_exponent = get_param_value(acs_params.get('fusion_exponent'), 1.0)
        dynamic_fusion_weights = get_param_value(acs_params.get('dynamic_fusion_weights'), {"patient_base": 0.5, "agile_base": 0.5, "trend_vitality_mod": 0.2, "volatility_mod": 0.1, "market_phase_mod": 0.15, "risk_appetite_mod": 0.1})
        inter_capital_game_weights = get_param_value(acs_params.get('inter_capital_game_weights'), {"mf_retail_battle_intensity": 0.6, "mf_retail_liquidity_swap_corr": 0.4})
        covertness_anti_recon_weights = get_param_value(acs_params.get('covertness_anti_recon_weights'), {"covert_accumulation_slope": 0.4, "suppressive_accumulation_slope": 0.3, "deception_wash_trade_inverse": 0.3})
        chip_structure_control_weights = get_param_value(acs_params.get('chip_structure_control_weights'), {"cost_structure_skewness": 0.4, "chip_fatigue": 0.3, "winner_loser_momentum": 0.3})
        emotion_driven_risk_appetite_weights = get_param_value(acs_params.get('emotion_driven_risk_appetite_weights'), {"retail_fomo_slope": 0.4, "retail_panic_slope": 0.3, "market_sentiment_slope": 0.3})
        game_efficiency_utilization_weights = get_param_value(acs_params.get('game_efficiency_utilization_weights'), {"main_force_t0_efficiency_accel": 0.4, "main_force_slippage_inverse_accel": 0.3, "main_force_execution_alpha_accel": 0.3})
        patient_capital_weights_v3_1 = get_param_value(acs_params.get('patient_capital_weights_v3_1'), {
            'main_force_buy_ofi': 0.1, 'main_force_sell_ofi': 0.1,
            'wash_trade_buy_volume': 0.05, 'wash_trade_sell_volume': 0.05
        })
        agile_capital_weights_v3_1 = get_param_value(acs_params.get('agile_capital_weights_v3_1'), {
            'rally_sell_distribution_intensity': 0.1, 'rally_buy_support_weakness': 0.1,
            'retail_buy_ofi': 0.05, 'retail_sell_ofi': 0.05
        })
        # --- 信号依赖校验 ---
        required_signals = [
            'net_lg_amount_calibrated_D', 'NMFNF_D', 'main_force_slippage_index_D',
            'SLOPE_5_NMFNF_D', 'ACCEL_5_NMFNF_D',
            'SLOPE_13_net_lg_amount_calibrated_D', 'ACCEL_13_net_lg_amount_calibrated_D',
            'SLOPE_21_net_lg_amount_calibrated_D', 'ACCEL_21_net_lg_amount_calibrated_D',
            'SLOPE_55_net_lg_amount_calibrated_D', 'ACCEL_55_net_lg_amount_calibrated_D',
            'main_force_cost_advantage_D', 'main_force_vwap_guidance_D', 'main_force_execution_alpha_D',
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D', 'flow_credibility_index_D',
            'chip_health_score_D', 'control_solidity_index_D',
            'SLOPE_5_covert_accumulation_signal_D', 'ACCEL_5_covert_accumulation_signal_D',
            'SLOPE_5_suppressive_accumulation_intensity_D', 'ACCEL_5_suppressive_accumulation_intensity_D',
            'deception_index_D', 'wash_trade_intensity_D',
            'SLOPE_5_deception_index_D', 'SLOPE_5_wash_trade_intensity_D',
            'cost_structure_skewness_D', 'chip_fatigue_index_D', 'winner_loser_momentum_D',
            'structural_leverage_D', 'structural_node_count_D',
            'main_force_ofi_D', 'main_force_t0_efficiency_D',
            'SLOPE_5_main_force_ofi_D', 'ACCEL_5_main_force_ofi_D',
            'SLOPE_13_main_force_ofi_D', 'ACCEL_13_main_force_ofi_D',
            'SLOPE_21_main_force_ofi_D', 'ACCEL_21_main_force_ofi_D',
            'micro_price_impact_asymmetry_D', 'THEME_HOTNESS_SCORE_D',
            'retail_fomo_premium_index_D', 'retail_panic_surrender_index_D',
            'SLOPE_5_retail_fomo_premium_index_D', 'SLOPE_5_retail_panic_surrender_index_D',
            'SLOPE_5_market_sentiment_score_D',
            'ACCEL_5_main_force_t0_efficiency_D', 'ACCEL_5_main_force_slippage_index_D', 'ACCEL_5_main_force_execution_alpha_D',
            'mf_retail_battle_intensity_D', 'mf_retail_liquidity_swap_corr_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'order_book_liquidity_supply_D', 'order_book_clearing_rate_D',
            'market_sentiment_score_D', 'trend_vitality_index_D', 'strategic_phase_score_D', 'risk_reward_profile_D',
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_capital_signature"):
            return pd.Series(0.0, index=df.index)
        # --- 预取所有斜率和加速度数据到单个字典 ---
        all_pre_fetched_slopes_accels = {}
        # 收集所有需要预取的信号基础名称和周期
        # patient flow periods
        patient_flow_periods = list(set(mtf_periods_patient_flow.get('short', []) + mtf_periods_patient_flow.get('long', [])))
        for p in patient_flow_periods:
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_net_lg_amount_calibrated_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_net_lg_amount_calibrated_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # agile ofi periods
        agile_ofi_periods = list(set(mtf_periods_agile_ofi.get('short', []) + mtf_periods_agile_ofi.get('long', [])))
        for p in agile_ofi_periods:
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_main_force_ofi_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_main_force_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_main_force_ofi_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_main_force_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # covertness_anti_recon signals
        for p in [5]: # covert_accumulation_signal_D, suppressive_accumulation_intensity_D, deception_index_D, wash_trade_intensity_D
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_covert_accumulation_signal_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_covert_accumulation_signal_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_covert_accumulation_signal_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_covert_accumulation_signal_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_suppressive_accumulation_intensity_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_suppressive_accumulation_intensity_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_deception_index_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_deception_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_wash_trade_intensity_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # emotion_driven_risk_appetite signals
        for p in [5]: # retail_fomo_premium_index_D, retail_panic_surrender_index_D, market_sentiment_score_D
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_retail_fomo_premium_index_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_retail_panic_surrender_index_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_market_sentiment_score_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_market_sentiment_score_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # game_efficiency_utilization signals
        for p in [5]: # main_force_t0_efficiency_D, main_force_slippage_index_D, main_force_execution_alpha_D
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_main_force_t0_efficiency_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_main_force_t0_efficiency_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_main_force_slippage_index_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_main_force_slippage_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_main_force_execution_alpha_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_main_force_execution_alpha_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # short_term_explosiveness signals
        for p in [5]: # NMFNF_D
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_NMFNF_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'ACCEL_{p}_NMFNF_D'] = self._get_safe_series(df, df, f'ACCEL_{p}_NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature")
            all_pre_fetched_slopes_accels[f'SLOPE_{p}_SLOPE_{p}_NMFNF_D'] = self._get_safe_series(df, df, f'SLOPE_{p}_SLOPE_{p}_NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name not in all_pre_fetched_slopes_accels: # 避免重复获取
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_capital_signature")
        # 耐心资本相关
        main_force_cost_advantage_raw = raw_data_cache['main_force_cost_advantage_D']
        main_force_vwap_guidance_raw = raw_data_cache['main_force_vwap_guidance_D']
        main_force_execution_alpha_raw = raw_data_cache['main_force_execution_alpha_D']
        covert_accumulation_signal_raw = raw_data_cache['covert_accumulation_signal_D']
        suppressive_accumulation_intensity_raw = raw_data_cache['suppressive_accumulation_intensity_D']
        deception_index_raw = raw_data_cache['deception_index_D']
        wash_trade_intensity_raw = raw_data_cache['wash_trade_intensity_D']
        cost_structure_skewness_raw = raw_data_cache['cost_structure_skewness_D']
        chip_fatigue_index_raw = raw_data_cache['chip_fatigue_index_D']
        winner_loser_momentum_raw = raw_data_cache['winner_loser_momentum_D']
        structural_leverage_raw = raw_data_cache['structural_leverage_D']
        structural_node_count_raw = raw_data_cache['structural_node_count_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        # 敏捷资本相关
        main_force_ofi_raw = raw_data_cache['main_force_ofi_D']
        micro_price_impact_asymmetry_raw = raw_data_cache['micro_price_impact_asymmetry_D']
        retail_fomo_premium_index_raw = raw_data_cache['retail_fomo_premium_index_D']
        retail_panic_surrender_index_raw = raw_data_cache['retail_panic_surrender_index_D'] # 修复点：现在 retail_panic_surrender_index_D 应该在 raw_data_cache 中
        market_sentiment_score_raw = raw_data_cache['market_sentiment_score_D']
        main_force_t0_efficiency_raw = raw_data_cache['main_force_t0_efficiency_D']
        main_force_slippage_index_raw = raw_data_cache['main_force_slippage_index_D']
        nmfnf_raw = raw_data_cache['NMFNF_D']
        theme_hotness_raw = raw_data_cache['THEME_HOTNESS_SCORE_D']
        # 资本间意图博弈相关
        mf_retail_battle_intensity_raw = raw_data_cache['mf_retail_battle_intensity_D']
        mf_retail_liquidity_swap_corr_raw = raw_data_cache['mf_retail_liquidity_swap_corr_D']
        # 情境调制相关
        volatility_instability_raw = raw_data_cache['VOLATILITY_INSTABILITY_INDEX_21d_D']
        order_book_liquidity_supply_raw = raw_data_cache['order_book_liquidity_supply_D']
        order_book_clearing_rate_raw = raw_data_cache['order_book_clearing_rate_D']
        market_sentiment_raw_context = raw_data_cache['market_sentiment_score_D'] # 区分与情绪驱动的
        trend_vitality_raw = raw_data_cache['trend_vitality_index_D']
        strategic_phase_raw = raw_data_cache['strategic_phase_score_D']
        risk_reward_profile_raw = raw_data_cache['risk_reward_profile_D']
        rally_sell_distribution_intensity_raw = raw_data_cache['rally_sell_distribution_intensity_D']
        rally_buy_support_weakness_raw = raw_data_cache['rally_buy_support_weakness_D']
        main_force_buy_ofi_raw = raw_data_cache['main_force_buy_ofi_D']
        main_force_sell_ofi_raw = raw_data_cache['main_force_sell_ofi_D']
        retail_buy_ofi_raw = raw_data_cache['retail_buy_ofi_D']
        retail_sell_ofi_raw = raw_data_cache['retail_sell_ofi_D']
        wash_trade_buy_volume_raw = raw_data_cache['wash_trade_buy_volume_D']
        wash_trade_sell_volume_raw = raw_data_cache['wash_trade_sell_volume_D']
        # --- 1. 耐心资本 (Patient Capital) - 意图博弈与结构演化 ---
        # 1.1 多时间框架净流入持久性
        patient_flow_slope_weights_short = {str(p): 1/len(mtf_periods_patient_flow.get('short', [1])) for p in mtf_periods_patient_flow.get('short', [1])}
        patient_flow_accel_weights_short = {str(p): 1/len(mtf_periods_patient_flow.get('short', [1])) for p in mtf_periods_patient_flow.get('short', [1])}
        patient_flow_slope_weights_long = {str(p): 1/len(mtf_periods_patient_flow.get('long', [1])) for p in mtf_periods_patient_flow.get('long', [1])}
        norm_institutional_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('short', []), patient_flow_slope_weights_short, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_institutional_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('short', []), patient_flow_accel_weights_short, True, True, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_institutional_flow_long_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('long', []), patient_flow_slope_weights_long, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        flow_persistence = (
            norm_institutional_flow_slope_mtf * 0.4 +
            norm_institutional_flow_accel_mtf * 0.3 +
            norm_institutional_flow_long_slope_mtf * 0.3
        ).clip(-1, 1)
        # 1.2 成本控制与效率
        norm_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights_ff)
        norm_vwap_guidance = get_adaptive_mtf_normalized_bipolar_score(main_force_vwap_guidance_raw, df_index, tf_weights=tf_weights_ff)
        norm_execution_alpha = get_adaptive_mtf_normalized_bipolar_score(main_force_execution_alpha_raw, df_index, tf_weights=tf_weights_ff)
        cost_efficiency = (norm_cost_advantage * 0.4 + norm_vwap_guidance * 0.3 + norm_execution_alpha * 0.3).clip(-1, 1)
        # 1.3 隐蔽性与反侦察能力 (V3.0 新增)
        covert_acc_slope_weights = {"5": 1.0}
        suppressive_acc_slope_weights = {"5": 1.0}
        deception_slope_weights = {"5": 1.0}
        wash_trade_slope_weights = {"5": 1.0}
        norm_covert_accumulation_slope = self._get_mtf_dynamic_score(df, 'covert_accumulation_signal_D', [5], covert_acc_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_suppressive_accumulation_slope = self._get_mtf_dynamic_score(df, 'suppressive_accumulation_intensity_D', [5], suppressive_acc_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_deception_slope = self._get_mtf_dynamic_score(df, 'deception_index_D', [5], deception_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_wash_trade_slope = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', [5], wash_trade_slope_weights, False, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        deception_wash_trade_inverse = (1 - norm_deception_slope.abs() - norm_wash_trade_slope).clip(0, 1)
        covertness_anti_recon = (
            norm_covert_accumulation_slope * covertness_anti_recon_weights.get('covert_accumulation_slope', 0.4) +
            norm_suppressive_accumulation_slope * covertness_anti_recon_weights.get('suppressive_accumulation_slope', 0.3) +
            deception_wash_trade_inverse * covertness_anti_recon_weights.get('deception_wash_trade_inverse', 0.3)
        ).clip(0, 1)
        # 1.4 成本结构与筹码分布 (V3.0 新增)
        norm_cost_structure_skewness = get_adaptive_mtf_normalized_bipolar_score(cost_structure_skewness_raw, df_index, tf_weights=tf_weights_ff)
        norm_chip_fatigue = get_adaptive_mtf_normalized_score(chip_fatigue_index_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_bipolar_score(winner_loser_momentum_raw, df_index, tf_weights=tf_weights_ff)
        chip_structure_control = (
            norm_cost_structure_skewness * chip_structure_control_weights.get('cost_structure_skewness', 0.4) +
            norm_chip_fatigue * chip_structure_control_weights.get('chip_fatigue', 0.3) +
            norm_winner_loser_momentum * chip_structure_control_weights.get('winner_loser_momentum', 0.3)
        ).clip(-1, 1)
        # 1.5 资金流结构韧性 (V3.0 新增)
        norm_structural_leverage = get_adaptive_mtf_normalized_score(structural_leverage_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_structural_node_count = get_adaptive_mtf_normalized_score(structural_node_count_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        flow_structure_resilience = (norm_structural_leverage * 0.5 + norm_structural_node_count * 0.5).clip(0, 1)
        # 融合耐心资本得分
        patient_capital_score = (
            flow_persistence * patient_capital_weights.get('mtf_flow_persistence', 0.25) +
            cost_efficiency * patient_capital_weights.get('cost_efficiency', 0.2) +
            covertness_anti_recon * patient_capital_weights.get('covertness_anti_recon', 0.2) +
            chip_structure_control * patient_capital_weights.get('chip_structure_control', 0.2) +
            flow_structure_resilience * patient_capital_weights.get('flow_structure_resilience', 0.15)
        )
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(main_force_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(main_force_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(wash_trade_buy_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(wash_trade_sell_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        patient_capital_score = patient_capital_score + \
                                (norm_main_force_buy_ofi * patient_capital_weights_v3_1.get('main_force_buy_ofi', 0.1)) - \
                                (norm_main_force_sell_ofi * patient_capital_weights_v3_1.get('main_force_sell_ofi', 0.1)) - \
                                (norm_wash_trade_buy_volume * patient_capital_weights_v3_1.get('wash_trade_buy_volume', 0.05)) - \
                                (norm_wash_trade_sell_volume * patient_capital_weights_v3_1.get('wash_trade_sell_volume', 0.05))
        patient_capital_score = patient_capital_score.clip(-1, 1)
        # --- 2. 敏捷资本 (Agile Capital) - 情绪驱动与博弈效率 ---
        # 2.1 高频冲击力与方向性
        agile_ofi_slope_weights_short = {str(p): 1/len(mtf_periods_agile_ofi.get('short', [1])) for p in mtf_periods_agile_ofi.get('short', [1])}
        agile_ofi_accel_weights_short = {str(p): 1/len(mtf_periods_agile_ofi.get('short', [1])) for p in mtf_periods_agile_ofi.get('short', [1])}
        norm_ofi_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_ofi_D', mtf_periods_agile_ofi.get('short', []), agile_ofi_slope_weights_short, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_ofi_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_ofi_D', mtf_periods_agile_ofi.get('short', []), agile_ofi_accel_weights_short, True, True, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_price_impact_asymmetry = get_adaptive_mtf_normalized_bipolar_score(micro_price_impact_asymmetry_raw, df_index, tf_weights=tf_weights_ff)
        ofi_impact_directionality = (norm_ofi_slope_mtf * 0.4 + norm_ofi_accel_mtf * 0.3 + norm_price_impact_asymmetry * 0.3).clip(-1, 1)
        # 2.2 情绪驱动与风险偏好 (V3.0 新增)
        retail_fomo_slope_weights = {"5": 1.0}
        retail_panic_slope_weights = {"5": 1.0}
        market_sentiment_slope_weights = {"5": 1.0}
        norm_retail_fomo_slope = self._get_mtf_dynamic_score(df, 'retail_fomo_premium_index_D', [5], retail_fomo_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_retail_panic_slope = self._get_mtf_dynamic_score(df, 'retail_panic_surrender_index_D', [5], retail_panic_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_market_sentiment_slope = self._get_mtf_dynamic_score(df, 'market_sentiment_score_D', [5], market_sentiment_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        emotion_driven_risk_appetite = (
            norm_retail_fomo_slope * emotion_driven_risk_appetite_weights.get('retail_fomo_slope', 0.4) +
            norm_retail_panic_slope * emotion_driven_risk_appetite_weights.get('retail_panic_slope', 0.3) +
            norm_market_sentiment_slope * emotion_driven_risk_appetite_weights.get('market_sentiment_slope', 0.3)
        ).clip(-1, 1)
        # 2.3 博弈效率与资金利用率 (V3.0 新增)
        mf_t0_efficiency_accel_weights = {"5": 1.0}
        mf_slippage_accel_weights = {"5": 1.0}
        mf_exec_alpha_accel_weights = {"5": 1.0}
        norm_main_force_t0_efficiency_accel = self._get_mtf_dynamic_score(df, 'main_force_t0_efficiency_D', [5], mf_t0_efficiency_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_main_force_slippage_inverse_accel = self._get_mtf_dynamic_score(df, 'main_force_slippage_index_D', [5], mf_slippage_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels) * -1
        norm_main_force_execution_alpha_accel = self._get_mtf_dynamic_score(df, 'main_force_execution_alpha_D', [5], mf_exec_alpha_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        game_efficiency_utilization = (
            norm_main_force_t0_efficiency_accel * game_efficiency_utilization_weights.get('main_force_t0_efficiency_accel', 0.4) +
            norm_main_force_slippage_inverse_accel * game_efficiency_utilization_weights.get('main_force_slippage_inverse_accel', 0.3) +
            norm_main_force_execution_alpha_accel * game_efficiency_utilization_weights.get('main_force_execution_alpha_accel', 0.3)
        ).clip(-1, 1)
        # 2.4 短期爆发力与持续性 (V3.0 强化)
        nmfnf_slope_weights = {"5": 1.0}
        nmfnf_accel_weights = {"5": 1.0}
        nmfnf_accel_slope_weights = {"5": 1.0}
        norm_nmfnf_slope_5 = self._get_mtf_dynamic_score(df, 'NMFNF_D', [5], nmfnf_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_nmfnf_accel_5 = self._get_mtf_dynamic_score(df, 'NMFNF_D', [5], nmfnf_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_nmfnf_accel_slope_5 = self._get_mtf_dynamic_score(df, 'SLOPE_5_NMFNF_D', [5], nmfnf_accel_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature", pre_fetched_data=all_pre_fetched_slopes_accels)
        short_term_explosiveness = (norm_nmfnf_slope_5 * 0.5 + norm_nmfnf_accel_5 * 0.3 + norm_nmfnf_accel_slope_5 * 0.2).clip(-1, 1)
        # 2.5 资金流驱动的题材热度
        norm_theme_hotness = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 融合敏捷资本得分
        agile_capital_score = (
            ofi_impact_directionality * agile_capital_weights.get('ofi_impact_directionality', 0.25) +
            emotion_driven_risk_appetite * agile_capital_weights.get('emotion_driven_risk_appetite', 0.2) +
            game_efficiency_utilization * agile_capital_weights.get('game_efficiency_utilization', 0.2) +
            short_term_explosiveness * agile_capital_weights.get('short_term_explosiveness', 0.2) +
            norm_theme_hotness * agile_capital_weights.get('theme_chasing', 0.15)
        )
        norm_rally_sell_distribution_intensity = get_adaptive_mtf_normalized_score(rally_sell_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_rally_buy_support_weakness = get_adaptive_mtf_normalized_score(rally_buy_support_weakness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(retail_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(retail_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        agile_capital_score = agile_capital_score \
                              - (norm_rally_sell_distribution_intensity * agile_capital_weights_v3_1.get('rally_sell_distribution_intensity', 0.1)) \
                              - (norm_rally_buy_support_weakness * agile_capital_weights_v3_1.get('rally_buy_support_weakness', 0.1)) \
                              - (norm_retail_buy_ofi * agile_capital_weights_v3_1.get('retail_buy_ofi', 0.05)) \
                              + (norm_retail_sell_ofi * agile_capital_weights_v3_1.get('retail_sell_ofi', 0.05))
        agile_capital_score = agile_capital_score.clip(-1, 1)
        # --- 3. 资本间意图博弈分析 (Inter-Capital Intent Game Analysis) (V3.0 新增) ---
        norm_mf_retail_battle_intensity = get_adaptive_mtf_normalized_score(mf_retail_battle_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_mf_retail_liquidity_swap_corr = get_adaptive_mtf_normalized_bipolar_score(mf_retail_liquidity_swap_corr_raw, df_index, tf_weights=tf_weights_ff)
        inter_capital_game_score = (
            norm_mf_retail_battle_intensity * inter_capital_game_weights.get('mf_retail_battle_intensity', 0.6) * np.sign(patient_capital_score - agile_capital_score) +
            norm_mf_retail_liquidity_swap_corr * inter_capital_game_weights.get('mf_retail_liquidity_swap_corr', 0.4)
        ).clip(-1, 1)
        # --- 4. 情境自适应动态权重与非线性融合 (Context-Adaptive Dynamic Weights & Non-linear Fusion) ---
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_liquidity_supply = get_adaptive_mtf_normalized_score(order_book_liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_liquidity_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw_context, df_index, tf_weights=tf_weights_ff)
        norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_strategic_phase = get_adaptive_mtf_normalized_bipolar_score(strategic_phase_raw, df_index, tf_weights=tf_weights_ff)
        norm_risk_reward_profile = get_adaptive_mtf_normalized_bipolar_score(risk_reward_profile_raw, df_index, tf_weights=tf_weights_ff)
        liquidity_mod = (1 + (norm_liquidity_supply + norm_liquidity_clearing_rate)/2 * capital_context_modulator_sensitivity.get('liquidity', 0.2)).clip(0.5, 1.5)
        volatility_mod = (1 - norm_volatility_instability * capital_context_modulator_sensitivity.get('volatility', 0.3)).clip(0.5, 1.5)
        sentiment_mod = (1 + norm_market_sentiment * capital_context_modulator_sensitivity.get('sentiment', 0.1)).clip(0.5, 1.5)
        trend_mod = (1 + norm_trend_vitality * capital_context_modulator_sensitivity.get('trend_vitality', 0.2)).clip(0.5, 1.5)
        market_phase_mod = (1 + norm_strategic_phase * capital_context_modulator_sensitivity.get('market_phase', 0.15)).clip(0.5, 1.5)
        risk_appetite_mod = (1 + norm_risk_reward_profile * capital_context_modulator_sensitivity.get('risk_appetite', 0.1)).clip(0.5, 1.5)
        dynamic_patient_weight = dynamic_fusion_weights.get('patient_base', 0.5) * (1 + trend_mod * dynamic_fusion_weights.get('trend_vitality_mod', 0.2) - volatility_mod * dynamic_fusion_weights.get('volatility_mod', 0.1) + market_phase_mod * dynamic_fusion_weights.get('market_phase_mod', 0.15) + risk_appetite_mod * dynamic_fusion_weights.get('risk_appetite_mod', 0.1))
        dynamic_agile_weight = dynamic_fusion_weights.get('agile_base', 0.5) * (1 - trend_mod * dynamic_fusion_weights.get('trend_vitality_mod', 0.2) + volatility_mod * dynamic_fusion_weights.get('volatility_mod', 0.1) - market_phase_mod * dynamic_fusion_weights.get('market_phase_mod', 0.15) - risk_appetite_mod * dynamic_fusion_weights.get('risk_appetite_mod', 0.1))
        total_dynamic_weights = dynamic_patient_weight + dynamic_agile_weight
        dynamic_patient_weight /= total_dynamic_weights
        dynamic_agile_weight /= total_dynamic_weights
        patient_modulated_score = patient_capital_score * dynamic_patient_weight * liquidity_mod * volatility_mod * sentiment_mod * (1 + inter_capital_game_score.clip(lower=0))
        agile_modulated_score = agile_capital_score * dynamic_agile_weight * liquidity_mod * volatility_mod * sentiment_mod * (1 + inter_capital_game_score.clip(upper=0).abs())
        capital_signature_score = np.tanh(patient_modulated_score - agile_modulated_score).pow(fusion_exponent).clip(-1, 1)
        print(f"    -> [资金流层] 资本属性 (capital_signature_score): {capital_signature_score.mean():.4f}")
        return capital_signature_score.astype(np.float32)

    def _diagnose_axiom_flow_structure_health(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 资金流向与风险校准版】资金流公理六：诊断“资金流结构健康度”
        - 核心升级:
            1. 严格遵循“仅针对【资金】类原始数据进行分析”的原则，移除对筹码层（main_force_vpoc_D）和价格层（close_D）的依赖。
            2. 修正了流动性（bid_side_liquidity_D, ask_side_liquidity_D）和卖方效率（sell_flow_efficiency_index_D等）的归一化方向，使其更符合“健康度”的定义。
            3. 引入了新的资金流结构风险指标：资金流基尼系数（main_force_flow_gini_D）和订单簿不稳定性（order_book_imbalance_D的波动率），替代了非资金流的结构杠杆。
            4. 升级了各子分数的融合方式，采用健壮的加权几何平均（_robust_geometric_mean），以增强协同效应和非线性特征。
            5. 增加了详细的探针输出，方便调试和理解计算过程。
            6. 修正了流量平稳度，使其同时考虑波动性和资金流向。
            7. 修正了资金流集中度风险的逻辑，高基尼系数现在正确地表示低健康度。
        """
        print(f"    -> [资金流层] 正在诊断 资金流公理六：诊断“资金流结构健康度 (V1.3 · 资金流向与风险校准版)”...")
        df_index = df.index
        p_conf_ff = self.p_conf_ff
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        afsh_params = get_param_value(p_conf_ff.get('axiom_flow_structure_health_params'), {})
        # probe_enabled = get_param_value(afsh_params.get('probe_enabled'), False) # 移除探针相关变量
        # current_probe_date = None # 移除探针相关变量
        # if probe_enabled and self.probe_dates: # 移除探针相关逻辑
        #     probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
        #     for date in reversed(df_index):
        #         if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
        #             current_probe_date = date
        #             break
        # if probe_enabled: # 移除探针相关逻辑
        #     if current_probe_date:
        #         print(f"        [探针] 资金流结构健康度诊断启动。探针日期: {current_probe_date.strftime('%Y-%m-%d')}")
        #     else:
        #         print(f"        [探针] 资金流结构健康度诊断启动。probe_enabled为True，但当前DataFrame不包含任何指定探针日期。")
        flow_steadiness_params = get_param_value(afsh_params.get('flow_steadiness_params'), {})
        net_flow_std_window = get_param_value(flow_steadiness_params.get('net_flow_std_window'), 21)
        net_flow_direction_window = get_param_value(flow_steadiness_params.get('net_flow_direction_window'), 5)
        flow_steadiness_norm_tf_weights = get_param_value(flow_steadiness_params.get('normalization_tf_weights'), tf_weights_ff)
        flow_efficiency_params = get_param_value(afsh_params.get('flow_efficiency_params'), {})
        net_flow_mean_window = get_param_value(flow_efficiency_params.get('net_flow_mean_window'), 21)
        price_volatility_window = get_param_value(flow_efficiency_params.get('price_volatility_window'), 14)
        base_efficiency_weights = get_param_value(flow_efficiency_params.get('base_efficiency_weights'), {'net_flow_mean_atr_ratio': 1.0})
        efficiency_enhancement_weights = get_param_value(flow_efficiency_params.get('enhancement_weights'), {
            'buy_flow_efficiency_index': 0.2, 'sell_flow_efficiency_index': 0.2,
            'buy_order_book_clearing_rate': 0.15, 'sell_order_book_clearing_rate': 0.15,
            'vwap_buy_control_strength': 0.15, 'vwap_sell_control_strength': 0.15
        })
        flow_efficiency_norm_tf_weights = get_param_value(flow_efficiency_params.get('normalization_tf_weights'), tf_weights_ff)
        structural_risk_params = get_param_value(afsh_params.get('structural_risk_params'), {})
        liquidity_weights = get_param_value(structural_risk_params.get('liquidity_weights'), {'bid_side_liquidity': 0.5, 'ask_side_liquidity': 0.5})
        flow_gini_weights = get_param_value(structural_risk_params.get('flow_gini_weights'), {'main_force_flow_gini': 1.0})
        order_book_stability_params = get_param_value(structural_risk_params.get('order_book_stability_weights'), {'order_book_imbalance_std_window': 21, 'order_book_imbalance': 1.0})
        order_book_imbalance_std_window = get_param_value(order_book_stability_params.get('order_book_imbalance_std_window'), 21)
        flow_credibility_weights = get_param_value(structural_risk_params.get('flow_credibility_weights'), {'flow_credibility_index': 1.0})
        structural_risk_norm_tf_weights = get_param_value(structural_risk_params.get('normalization_tf_weights'), tf_weights_ff)
        final_fusion_weights = get_param_value(afsh_params.get('final_fusion_weights'), {'flow_steadiness': 0.3, 'enhanced_flow_efficiency': 0.4, 'structural_risk_filter': 0.3})
        required_signals = [
            'main_force_net_flow_calibrated_D', 'ATR_14_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D',
            'buy_order_book_clearing_rate_D', 'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'main_force_flow_gini_D', 'order_book_imbalance_D', 'flow_credibility_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_structure_health"):
            return pd.Series(0.0, index=df.index)
        raw_data_cache = {}
        for signal_name in required_signals:
            raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_flow_structure_health")
        net_flow_raw = raw_data_cache['main_force_net_flow_calibrated_D']
        atr_raw = raw_data_cache['ATR_14_D']
        buy_flow_efficiency_raw = raw_data_cache['buy_flow_efficiency_index_D']
        sell_flow_efficiency_raw = raw_data_cache['sell_flow_efficiency_index_D']
        buy_order_book_clearing_rate_raw = raw_data_cache['buy_order_book_clearing_rate_D']
        sell_order_book_clearing_rate_raw = raw_data_cache['sell_order_book_clearing_rate_D']
        vwap_buy_control_strength_raw = raw_data_cache['vwap_buy_control_strength_D']
        vwap_sell_control_strength_raw = raw_data_cache['vwap_sell_control_strength_D']
        bid_side_liquidity_raw = raw_data_cache['bid_side_liquidity_D']
        ask_side_liquidity_raw = raw_data_cache['ask_side_liquidity_D']
        main_force_flow_gini_raw = raw_data_cache['main_force_flow_gini_D']
        order_book_imbalance_raw = raw_data_cache['order_book_imbalance_D']
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 原始数据获取完成。")
        #     print(f"          - main_force_net_flow_calibrated_D: {net_flow_raw.loc[current_probe_date]:.4f}")
        #     print(f"          - ATR_14_D: {atr_raw.loc[current_probe_date]:.4f}")
        #     print(f"          - buy_flow_efficiency_index_D: {buy_flow_efficiency_raw.loc[current_probe_date]:.4f}")
        #     print(f"          - sell_flow_efficiency_index_D: {sell_flow_efficiency_raw.loc[current_probe_date]:.4f}")
        #     print(f"          - main_force_flow_gini_D: {main_force_flow_gini_raw.loc[current_probe_date]:.4f}")
        #     print(f"          - order_book_imbalance_D: {order_book_imbalance_raw.loc[current_probe_date]:.4f}")
        # --- 1. 流量平稳度 (Flow Steadiness) - 引入方向性 ---
        flow_volatility = net_flow_raw.rolling(window=net_flow_std_window, min_periods=1).std().fillna(0)
        norm_flow_volatility_health = get_adaptive_mtf_normalized_score(flow_volatility, df_index, flow_steadiness_norm_tf_weights, ascending=False) # 低波动性 = 高健康度
        net_flow_direction_mean = net_flow_raw.rolling(window=net_flow_direction_window, min_periods=1).mean().fillna(0)
        norm_net_flow_direction_health = get_adaptive_mtf_normalized_score(net_flow_direction_mean, df_index, flow_steadiness_norm_tf_weights, ascending=True) # 正向流 = 高健康度
        # 融合波动性和方向性：只有当资金流平稳且方向为正时，才算健康
        norm_flow_steadiness = (norm_flow_volatility_health * norm_net_flow_direction_health).pow(0.5) # 几何平均融合
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 原始流量波动率 (flow_volatility): {flow_volatility.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 流量波动健康度 (norm_flow_volatility_health): {norm_flow_volatility_health.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 流量方向健康度 (norm_net_flow_direction_health): {norm_net_flow_direction_health.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 流量平稳度 (norm_flow_steadiness): {norm_flow_steadiness.loc[current_probe_date]:.4f}")
        # --- 2. 流量效率 (Flow Efficiency) ---
        net_flow_mean = net_flow_raw.rolling(window=net_flow_mean_window, min_periods=1).mean().fillna(0)
        price_volatility_mean = atr_raw.rolling(window=price_volatility_window, min_periods=1).mean().replace(0, 1e-9).fillna(1e-9)
        base_flow_efficiency_raw = (net_flow_mean / price_volatility_mean).replace([np.inf, -np.inf], 0).fillna(0)
        norm_base_flow_efficiency = get_adaptive_mtf_normalized_bipolar_score(base_flow_efficiency_raw, df_index, flow_efficiency_norm_tf_weights)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 原始基础流量效率 (base_flow_efficiency_raw): {base_flow_efficiency_raw.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 归一化基础流量效率 (norm_base_flow_efficiency): {norm_base_flow_efficiency.loc[current_probe_date]:.4f}")
        norm_buy_flow_efficiency = get_adaptive_mtf_normalized_score(buy_flow_efficiency_raw, df_index, flow_efficiency_norm_tf_weights, ascending=True)
        norm_sell_flow_efficiency = get_adaptive_mtf_normalized_score(sell_flow_efficiency_raw, df_index, flow_efficiency_norm_tf_weights, ascending=False) # 低卖方效率 = 高健康度
        norm_buy_order_book_clearing_rate = get_adaptive_mtf_normalized_score(buy_order_book_clearing_rate_raw, df_index, flow_efficiency_norm_tf_weights, ascending=True)
        norm_sell_order_book_clearing_rate = get_adaptive_mtf_normalized_score(sell_order_book_clearing_rate_raw, df_index, flow_efficiency_norm_tf_weights, ascending=False) # 低卖方清算率 = 高健康度
        norm_vwap_buy_control_strength = get_adaptive_mtf_normalized_score(vwap_buy_control_strength_raw, df_index, flow_efficiency_norm_tf_weights, ascending=True)
        norm_vwap_sell_control_strength = get_adaptive_mtf_normalized_score(vwap_sell_control_strength_raw, df_index, flow_efficiency_norm_tf_weights, ascending=False) # 低卖方控制力 = 高健康度
        efficiency_components = {
            'base_efficiency': (norm_base_flow_efficiency + 1) / 2, # 转换为0-1健康度
            'buy_flow_efficiency': norm_buy_flow_efficiency,
            'sell_flow_efficiency': norm_sell_flow_efficiency,
            'buy_order_book_clearing_rate': norm_buy_order_book_clearing_rate,
            'sell_order_book_clearing_rate': norm_sell_order_book_clearing_rate,
            'vwap_buy_control_strength': norm_vwap_buy_control_strength,
            'vwap_sell_control_strength': norm_vwap_sell_control_strength
        }
        efficiency_component_weights = {
            'base_efficiency': base_efficiency_weights.get('net_flow_mean_atr_ratio', 1.0),
            'buy_flow_efficiency': efficiency_enhancement_weights.get('buy_flow_efficiency_index', 0.2),
            'sell_flow_efficiency': efficiency_enhancement_weights.get('sell_flow_efficiency_index', 0.2), # 权重现在为正
            'buy_order_book_clearing_rate': efficiency_enhancement_weights.get('buy_order_book_clearing_rate', 0.15),
            'sell_order_book_clearing_rate': efficiency_enhancement_weights.get('sell_order_book_clearing_rate', 0.15), # 权重现在为正
            'vwap_buy_control_strength': efficiency_enhancement_weights.get('vwap_buy_control_strength', 0.15),
            'vwap_sell_control_strength': efficiency_enhancement_weights.get('vwap_sell_control_strength', 0.15) # 权重现在为正
        }
        enhanced_flow_efficiency_unipolar = _robust_geometric_mean(efficiency_components, efficiency_component_weights, df_index)
        enhanced_flow_efficiency = (enhanced_flow_efficiency_unipolar * 2 - 1).clip(-1, 1)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 流量效率 (enhanced_flow_efficiency): {enhanced_flow_efficiency.loc[current_probe_date]:.4f}")
        # --- 3. 结构风险过滤器 (Structural Risk Filter) ---
        norm_bid_side_liquidity = get_adaptive_mtf_normalized_score(bid_side_liquidity_raw, df_index, structural_risk_norm_tf_weights, ascending=True)
        norm_ask_side_liquidity = get_adaptive_mtf_normalized_score(ask_side_liquidity_raw, df_index, structural_risk_norm_tf_weights, ascending=True)
        liquidity_support_score = (
            norm_bid_side_liquidity * liquidity_weights.get('bid_side_liquidity', 0.5) +
            norm_ask_side_liquidity * liquidity_weights.get('ask_side_liquidity', 0.5)
        ).clip(0, 1)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 流动性支持 (liquidity_support_score): {liquidity_support_score.loc[current_probe_date]:.4f}")
        # 高Gini = 高风险 = 低健康度
        norm_flow_gini_risk = get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, structural_risk_norm_tf_weights, ascending=True)
        flow_concentration_health_score = (1 - norm_flow_gini_risk) * flow_gini_weights.get('main_force_flow_gini', 1.0)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 原始资金流基尼系数 (main_force_flow_gini_D): {main_force_flow_gini_raw.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 资金流集中度健康度 (flow_concentration_health_score): {flow_concentration_health_score.loc[current_probe_date]:.4f}")
        order_book_imbalance_volatility = order_book_imbalance_raw.rolling(window=order_book_imbalance_std_window, min_periods=1).std().fillna(0)
        norm_order_book_stability = 1 - get_adaptive_mtf_normalized_score(order_book_imbalance_volatility, df_index, structural_risk_norm_tf_weights, ascending=True)
        order_book_stability_health_score = norm_order_book_stability * order_book_stability_params.get('order_book_imbalance', 1.0)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 订单簿不稳定性 (order_book_imbalance_volatility): {order_book_imbalance_volatility.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 订单簿稳定性健康度 (order_book_stability_health_score): {order_book_stability_health_score.loc[current_probe_date]:.4f}")
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, structural_risk_norm_tf_weights, ascending=True)
        flow_credibility_score = norm_flow_credibility * flow_credibility_weights.get('flow_credibility_index', 1.0)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 资金流可信度 (flow_credibility_score): {flow_credibility_score.loc[current_probe_date]:.4f}")
        structural_risk_components = {
            'liquidity_support': liquidity_support_score,
            'flow_concentration_health': flow_concentration_health_score,
            'order_book_stability_health': order_book_stability_health_score,
            'flow_credibility': flow_credibility_score
        }
        structural_risk_component_weights = {k: 1.0 for k in structural_risk_components.keys()} # 默认等权重
        structural_risk_filter = _robust_geometric_mean(structural_risk_components, structural_risk_component_weights, df_index)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 结构风险过滤器 (structural_risk_filter): {structural_risk_filter.loc[current_probe_date]:.4f}")
        # --- 4. 最终融合 ---
        final_components = {
            'flow_steadiness': norm_flow_steadiness,
            'enhanced_flow_efficiency': (enhanced_flow_efficiency + 1) / 2, # 转换为0-1健康度
            'structural_risk_filter': structural_risk_filter
        }
        final_fusion_weights_adjusted = {
            'flow_steadiness': final_fusion_weights.get('flow_steadiness', 0.3),
            'enhanced_flow_efficiency': final_fusion_weights.get('enhanced_flow_efficiency', 0.4),
            'structural_risk_filter': final_fusion_weights.get('structural_risk_filter', 0.3)
        }
        flow_structure_health_score_unipolar = _robust_geometric_mean(final_components, final_fusion_weights_adjusted, df_index)
        flow_structure_health_score = (flow_structure_health_score_unipolar * 2 - 1).clip(-1, 1)
        # if probe_enabled and current_probe_date: # 移除探针相关逻辑
        #     print(f"        [探针] 最终资金流结构健康度 (flow_structure_health_score): {flow_structure_health_score.loc[current_probe_date]:.4f}")
        #     print(f"        [探针] 资金流结构健康度诊断完成。")
        print(f"    -> [资金流层] 最终资金流结构健康度 (flow_structure_health_score): {flow_structure_health_score.mean():.4f}")
        return flow_structure_health_score.astype(np.float32)

    def _calculate_mtf_cohesion_divergence(self, df: pd.DataFrame, signal_base_name: str, short_periods: List[int], long_periods: List[int], is_bipolar: bool, tf_weights: Dict, pre_fetched_data: Optional[Dict[str, pd.Series]] = None) -> pd.Series:
        """
        【V4.1 升级 · 效率优化版】计算双极性多时间框架的共振/背离因子。
        - 核心优化: 增加了 `pre_fetched_data` 参数，允许预先传入数据，避免重复调用 `_get_safe_series`。
        - 新增: `tf_weights` 参数，允许传入不同的时间框架权重。
        """
        method_name_str = "_calculate_mtf_cohesion_divergence"
        short_slope_scores = []
        short_accel_scores = []
        long_slope_scores = []
        long_accel_scores = []
        # 获取短期斜率和加速度
        for p in short_periods:
            slope_col = f'SLOPE_{p}_{signal_base_name}'
            accel_col = f'ACCEL_{p}_{signal_base_name}'
            # 优先从预取数据中获取
            if pre_fetched_data and slope_col in pre_fetched_data:
                slope_raw = pre_fetched_data[slope_col]
            else:
                slope_raw = self._get_safe_series(df, df, slope_col, 0.0, method_name_str)
            if pre_fetched_data and accel_col in pre_fetched_data:
                accel_raw = pre_fetched_data[accel_col]
            else:
                accel_raw = self._get_safe_series(df, df, accel_col, 0.0, method_name_str)
            # 修改结束
            if is_bipolar:
                short_slope_scores.append(get_adaptive_mtf_normalized_bipolar_score(slope_raw, df.index, tf_weights))
                short_accel_scores.append(get_adaptive_mtf_normalized_bipolar_score(accel_raw, df.index, tf_weights))
            else:
                short_slope_scores.append(get_adaptive_mtf_normalized_score(slope_raw, df.index, ascending=True, tf_weights=tf_weights))
                short_accel_scores.append(get_adaptive_mtf_normalized_score(accel_raw, df.index, ascending=True, tf_weights=tf_weights))
        # 获取长期斜率和加速度
        for p in long_periods:
            slope_col = f'SLOPE_{p}_{signal_base_name}'
            accel_col = f'ACCEL_{p}_{signal_base_name}'
            # 优先从预取数据中获取
            if pre_fetched_data and slope_col in pre_fetched_data:
                slope_raw = pre_fetched_data[slope_col]
            else:
                slope_raw = self._get_safe_series(df, df, slope_col, 0.0, method_name_str)
            if pre_fetched_data and accel_col in pre_fetched_data:
                accel_raw = pre_fetched_data[accel_col]
            else:
                accel_raw = self._get_safe_series(df, df, accel_col, 0.0, method_name_str)
            # 修改结束
            if is_bipolar:
                long_slope_scores.append(get_adaptive_mtf_normalized_bipolar_score(slope_raw, df.index, tf_weights))
                long_accel_scores.append(get_adaptive_mtf_normalized_bipolar_score(accel_raw, df.index, tf_weights))
            else:
                long_slope_scores.append(get_adaptive_mtf_normalized_score(slope_raw, df.index, ascending=True, tf_weights=tf_weights))
                long_accel_scores.append(get_adaptive_mtf_normalized_score(accel_raw, df.index, ascending=True, tf_weights=tf_weights))
        # 平均短期和长期分数
        avg_short_slope = sum(short_slope_scores) / len(short_slope_scores) if short_slope_scores else pd.Series(0.0, index=df.index)
        avg_short_accel = sum(short_accel_scores) / len(short_accel_scores) if short_accel_scores else pd.Series(0.0, index=df.index)
        avg_long_slope = sum(long_slope_scores) / len(long_slope_scores) if long_slope_scores else pd.Series(0.0, index=df.index)
        avg_long_accel = sum(long_accel_scores) / len(long_accel_scores) if long_accel_scores else pd.Series(0.0, index=df.index)
        # 计算双极性共振/背离分数
        # 1. 方向一致性：如果短期和长期方向一致，则为正；相反则为负。
        direction_alignment = np.sign(avg_short_slope) * np.sign(avg_long_slope)
        # 2. 强度一致性：短期和长期强度越接近，一致性越高。
        strength_cohesion_slope = (1 - (avg_short_slope.abs() - avg_long_slope.abs()).abs()).clip(0, 1)
        strength_cohesion_accel = (1 - (avg_short_accel.abs() - avg_long_accel.abs()).abs()).abs().clip(0, 1) # 修正为abs()
        # 综合强度一致性
        strength_cohesion = (strength_cohesion_slope + strength_cohesion_accel) / 2
        # 3. 最终双极性共振分数：方向 * 强度
        mtf_resonance_score = strength_cohesion * direction_alignment
        print(f"    -> [资金流层] 双极性多时间框架的共振/背离因子: {mtf_resonance_score.mean():.4f}")
        return mtf_resonance_score.astype(np.float32)

    def _diagnose_fund_flow_divergence_signals(self, df: pd.DataFrame, norm_window: int, axiom_divergence: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V4.5 · 幂运算精度与情境惩罚强化版】诊断资金流看涨/看跌背离信号。
        - 核心修复: 解决了 `bullish_divergence_score` 幂运算后值异常小的 Bug，确保浮点数精度和计算逻辑的正确性。
        - 业务逻辑增强:
            1. 强化“前期弱势惩罚”机制：当近期股价有显著下跌时，对看涨背离信号进行更严格的惩罚。
            2. 强化“诱多欺骗敏感度”：将行为层提供的 `deception_lure_long_intensity_D` 信号整合到看涨背离的纯度调制器中，当诱多强度高时，大幅降低看涨信号的纯度。
            3. 强化“趋势延续性”调制：在趋势不健康时，更有效地降低看涨信号的可靠性。
            4. 引入“散户狂热”反向指标：当散户情绪过于狂热时，惩罚看涨信号。
        """
        print(f"    -> [资金流层] 正在诊断 资金流公理四：诊断“资金流看涨/看跌背离”")
        method_name = "_diagnose_fund_flow_divergence_signals"
        df_index = df.index
        # 调试信息构建
        is_debug_enabled = get_param_value(self.debug_params.get('should_probe'), False) and get_param_value(self.debug_params.get('enabled'), False)
        probe_ts = None
        if is_debug_enabled and self.probe_dates:
            probe_dates_dt = [pd.to_datetime(d).normalize() for d in self.probe_dates]
            for date in reversed(df_index):
                if pd.to_datetime(date).tz_localize(None).normalize() in probe_dates_dt:
                    probe_ts = date
                    break
        debug_info_tuple = (is_debug_enabled, probe_ts, method_name)
        # 直接使用在 __init__ 中加载的配置
        p_conf_ff = self.p_conf_ff
        self.tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ffd_params = get_param_value(p_conf_ff.get('fund_flow_divergence_params'), {})
        purity_weights = get_param_value(ffd_params.get('purity_weights'), {"deception_inverse": 0.5, "wash_trade_inverse": 0.5})
        confirmation_weights = get_param_value(ffd_params.get('confirmation_weights'), {"conviction_positive": 0.5, "flow_momentum_positive": 0.5})
        context_modulator_weights = get_param_value(ffd_params.get('context_modulator_weights'), {"strategic_posture": 0.4, "flow_credibility": 0.3, "retail_panic_fomo_context": 0.3})
        non_linear_exponent_base = get_param_value(ffd_params.get('non_linear_exponent'), 1.2)
        bullish_divergence_threshold = get_param_value(ffd_params.get('bullish_divergence_threshold'), 0.1)
        bearish_divergence_threshold = get_param_value(ffd_params.get('bearish_divergence_threshold'), 0.1)
        retail_panic_fomo_sensitivity = get_param_value(ffd_params.get('retail_panic_fomo_sensitivity'), 0.5)
        mtf_resonance_factor_weights = get_param_value(ffd_params.get('mtf_resonance_factor_weights'), {"nmfnf_cohesion": 0.6, "conviction_cohesion": 0.4})
        micro_macro_divergence_weights = get_param_value(ffd_params.get('micro_macro_divergence_weights'), {"micro_intent_strength": 0.5, "macro_flow_momentum": 0.5})
        dynamic_context_modulator_sensitivity = get_param_value(ffd_params.get('dynamic_context_modulator_sensitivity'), {"volatility_instability": 0.2, "flow_credibility": 0.15, "trend_vitality": 0.1, "market_sentiment": 0.1})
        micro_intent_signals_weights = get_param_value(ffd_params.get('micro_intent_signals_weights'), {"order_book_imbalance": 0.3, "exhaustion_rate": 0.3, "micro_impact_elasticity": 0.2, "main_force_t0_efficiency": 0.2})
        bullish_exponent_context_factors = get_param_value(ffd_params.get('bullish_exponent_context_factors'), {"trend_vitality": 0.7, "volatility_inverse": 0.3})
        bearish_exponent_context_factors = get_param_value(ffd_params.get('bearish_exponent_context_factors'), {"volatility_instability": 0.6, "flow_credibility_inverse": 0.4})
        purity_context_mod_factors = get_param_value(ffd_params.get('purity_context_mod_factors'), {"market_sentiment": 0.5, "flow_credibility": 0.5})
        micro_buy_power_weights = get_param_value(ffd_params.get('micro_buy_power_weights'), {})
        micro_sell_power_weights = get_param_value(ffd_params.get('micro_sell_power_weights'), {})
        # 新增：前期弱势惩罚参数
        prior_weakness_penalty_params = get_param_value(ffd_params.get('prior_weakness_penalty_params'), {})
        prior_weakness_enabled = get_param_value(prior_weakness_penalty_params.get('enabled'), False)
        pct_change_window = get_param_value(prior_weakness_penalty_params.get('pct_change_window'), 3)
        pct_change_threshold = get_param_value(prior_weakness_penalty_params.get('pct_change_threshold'), -0.03)
        downward_momentum_weight = get_param_value(prior_weakness_penalty_params.get('downward_momentum_weight'), 0.5)
        prior_weakness_penalty_factor = get_param_value(prior_weakness_penalty_params.get('penalty_factor'), 0.5)
        # 新增：诱多欺骗惩罚因子
        deception_lure_long_penalty_factor = get_param_value(ffd_params.get('deception_lure_long_penalty_factor'), 0.5)
        required_signals = [
            'SCORE_FF_AXIOM_CONVICTION', 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 'SCORE_FF_STRATEGIC_POSTURE',
            'flow_credibility_index_D', 'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D',
            'SLOPE_5_deception_index_D', 'SLOPE_13_deception_index_D', 'SLOPE_21_deception_index_D',
            'SLOPE_5_wash_trade_intensity_D', 'SLOPE_13_wash_trade_intensity_D', 'SLOPE_21_wash_trade_intensity_D',
            'SLOPE_5_NMFNF_D', 'SLOPE_13_NMFNF_D', 'SLOPE_21_NMFNF_D', 'SLOPE_55_NMFNF_D',
            'ACCEL_5_NMFNF_D', 'ACCEL_13_NMFNF_D', 'ACCEL_21_NMFNF_D', 'ACCEL_55_NMFNF_D',
            'SLOPE_5_main_force_conviction_index_D', 'SLOPE_13_main_force_conviction_index_D', 'SLOPE_21_main_force_conviction_index_D', 'SLOPE_55_main_force_conviction_index_D',
            'ACCEL_5_main_force_conviction_index_D', 'ACCEL_13_main_force_conviction_index_D', 'ACCEL_21_main_force_conviction_index_D', 'ACCEL_55_main_force_conviction_index_D',
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'micro_impact_elasticity_D', 'main_force_t0_efficiency_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'trend_vitality_index_D', 'market_sentiment_score_D',
            'dip_buy_absorption_strength_D', 'dip_sell_pressure_resistance_D',
            'panic_sell_volume_contribution_D', 'panic_buy_absorption_contribution_D',
            'opening_buy_strength_D', 'opening_sell_strength_D',
            'pre_closing_buy_posture_D', 'pre_closing_sell_posture_D',
            'closing_auction_buy_ambush_D', 'closing_auction_sell_ambush_D',
            'main_force_t0_buy_efficiency_D', 'main_force_t0_sell_efficiency_D',
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D',
            'buy_order_book_clearing_rate_D', 'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'main_force_vwap_up_guidance_D', 'main_force_vwap_down_guidance_D',
            'vwap_cross_up_intensity_D', 'vwap_cross_down_intensity_D',
            'main_force_on_peak_sell_flow_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D',
            'pct_change_D', # 用于前期弱势惩罚
            'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', # 用于前期弱势惩罚
            'deception_lure_long_intensity_D' # 用于诱多欺骗敏感度
        ]
        if not self._validate_required_signals(df, required_signals, method_name, atomic_states=self.strategy.atomic_states):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        # 收集所有需要预取的信号基础名称和周期
        short_periods = [5, 13]
        long_periods = [21, 55]
        all_periods = list(set(short_periods + long_periods + [5, 13, 21, 55])) # 包含所有可能用到的斜率/加速度周期
        signal_bases_to_prefetch = [
            'deception_index_D', 'wash_trade_intensity_D', 'NMFNF_D', 'main_force_conviction_index_D'
        ]
        for signal_base in signal_bases_to_prefetch:
            for p in all_periods:
                all_pre_fetched_slopes_accels[f'SLOPE_{p}_{signal_base}'] = self._get_safe_series(df, df, f'SLOPE_{p}_{signal_base}', 0.0, method_name=method_name)
                all_pre_fetched_slopes_accels[f'ACCEL_{p}_{signal_base}'] = self._get_safe_series(df, df, f'ACCEL_{p}_{signal_base}', 0.0, method_name=method_name)
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name not in all_pre_fetched_slopes_accels and signal_name not in self.strategy.atomic_states: # 避免重复获取
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name=method_name)
        # 基础背离
        bullish_base_divergence, bearish_base_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        # --- 调试探针：检查 bullish_base_divergence 的值 ---
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"        [探针] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"          - axiom_divergence: {axiom_divergence.loc[probe_ts]:.4f}")
            print(f"          - bullish_base_divergence (from axiom_divergence): {bullish_base_divergence.loc[probe_ts]:.4f}")
        # 确认信号
        axiom_conviction = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_CONVICTION', 0.0, method_name=method_name)
        axiom_flow_momentum = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0, method_name=method_name)
        # 纯度过滤信号
        deception_slope_5_raw = raw_data_cache.get('SLOPE_5_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_5_deception_index_D'))
        deception_slope_13_raw = raw_data_cache.get('SLOPE_13_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_13_deception_index_D'))
        deception_slope_21_raw = raw_data_cache.get('SLOPE_21_deception_index_D', all_pre_fetched_slopes_accels.get('SLOPE_21_deception_index_D'))
        wash_trade_slope_5_raw = raw_data_cache.get('SLOPE_5_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_5_wash_trade_intensity_D'))
        wash_trade_slope_13_raw = raw_data_cache.get('SLOPE_13_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_13_wash_trade_intensity_D'))
        wash_trade_slope_21_raw = raw_data_cache.get('SLOPE_21_wash_trade_intensity_D', all_pre_fetched_slopes_accels.get('SLOPE_21_wash_trade_intensity_D'))
        # 情境校准信号
        strategic_posture = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_STRATEGIC_POSTURE', 0.0, method_name=method_name)
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        retail_panic_raw = raw_data_cache['retail_panic_surrender_index_D']
        retail_fomo_raw = raw_data_cache['retail_fomo_premium_index_D']
        # V4.0 新增原始数据
        order_book_imbalance_raw = raw_data_cache['order_book_imbalance_D']
        buy_exhaustion_raw = raw_data_cache['buy_quote_exhaustion_rate_D']
        sell_exhaustion_raw = raw_data_cache['sell_quote_exhaustion_rate_D']
        micro_impact_elasticity_raw = raw_data_cache['micro_impact_elasticity_D']
        main_force_t0_efficiency_raw = raw_data_cache['main_force_t0_efficiency_D']
        volatility_instability_raw = raw_data_cache['VOLATILITY_INSTABILITY_INDEX_21d_D']
        trend_vitality_raw = raw_data_cache['trend_vitality_index_D']
        market_sentiment_raw = raw_data_cache['market_sentiment_score_D']
        # V4.1 获取新增资金指标
        dip_buy_absorption_strength_raw = raw_data_cache['dip_buy_absorption_strength_D']
        dip_sell_pressure_resistance_raw = raw_data_cache['dip_sell_pressure_resistance_D']
        panic_sell_volume_contribution_raw = raw_data_cache['panic_sell_volume_contribution_D']
        panic_buy_absorption_contribution_raw = raw_data_cache['panic_buy_absorption_contribution_D']
        opening_buy_strength_raw = raw_data_cache['opening_buy_strength_D']
        opening_sell_strength_raw = raw_data_cache['opening_sell_strength_D']
        pre_closing_buy_posture_raw = raw_data_cache['pre_closing_buy_posture_D']
        pre_closing_sell_posture_raw = raw_data_cache['pre_closing_sell_posture_D']
        closing_auction_buy_ambush_raw = raw_data_cache['closing_auction_buy_ambush_D']
        closing_auction_sell_ambush_raw = raw_data_cache['closing_auction_sell_ambush_D']
        main_force_t0_buy_efficiency_raw = raw_data_cache['main_force_t0_buy_efficiency_D']
        main_force_t0_sell_efficiency_raw = raw_data_cache['main_force_t0_sell_efficiency_D']
        buy_flow_efficiency_index_raw = raw_data_cache['buy_flow_efficiency_index_D']
        sell_flow_efficiency_index_raw = raw_data_cache['sell_flow_efficiency_index_D']
        buy_order_book_clearing_rate_raw = raw_data_cache['buy_order_book_clearing_rate_D']
        sell_order_book_clearing_rate_raw = raw_data_cache['sell_order_book_clearing_rate_D']
        vwap_buy_control_strength_raw = raw_data_cache['vwap_buy_control_strength_D']
        vwap_sell_control_strength_raw = raw_data_cache['vwap_sell_control_strength_D']
        main_force_vwap_up_guidance_raw = raw_data_cache['main_force_vwap_up_guidance_D']
        main_force_vwap_down_guidance_raw = raw_data_cache['main_force_vwap_down_guidance_D']
        vwap_cross_up_intensity_raw = raw_data_cache['vwap_cross_up_intensity_D']
        vwap_cross_down_intensity_raw = raw_data_cache['vwap_cross_down_intensity_D']
        main_force_on_peak_sell_flow_raw = raw_data_cache['main_force_on_peak_sell_flow_D']
        bid_side_liquidity_raw = raw_data_cache['bid_side_liquidity_D']
        ask_side_liquidity_raw = raw_data_cache['ask_side_liquidity_D']
        pct_change_raw = raw_data_cache['pct_change_D'] # 用于前期弱势惩罚
        downward_momentum_raw = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM', 0.0, method_name=method_name) # 用于前期弱势惩罚
        deception_lure_long_raw = raw_data_cache['deception_lure_long_intensity_D'] # 用于诱多欺骗敏感度
        # --- 1. 纯度过滤 (Purity Filter) ---
        norm_deception_slope_5 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_5_raw, df_index, self.tf_weights_ff)
        norm_deception_slope_13 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_13_raw, df_index, self.tf_weights_ff)
        norm_deception_slope_21 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_21_raw, df_index, self.tf_weights_ff)
        norm_deception_multi_tf = (
            norm_deception_slope_5 * 0.5 +
            norm_deception_slope_13 * 0.3 +
            norm_deception_slope_21 * 0.2
        ).clip(-1, 1)
        norm_wash_trade_slope_5 = get_adaptive_mtf_normalized_score(wash_trade_slope_5_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_slope_13 = get_adaptive_mtf_normalized_score(wash_trade_slope_13_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_slope_21 = get_adaptive_mtf_normalized_score(wash_trade_slope_21_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_multi_tf = (
            norm_wash_trade_slope_5 * 0.5 +
            norm_wash_trade_slope_13 * 0.3 +
            norm_wash_trade_slope_21 * 0.2
        ).clip(0, 1)
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, self.tf_weights_ff)
        norm_flow_credibility_purity = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        purity_context_mod = (
            (1 + norm_market_sentiment.abs() * np.sign(norm_market_sentiment) * purity_context_mod_factors.get('market_sentiment', 0.5)) *
            (1 + (norm_flow_credibility_purity - 0.5) * purity_context_mod_factors.get('flow_credibility', 0.5))
        ).clip(0.5, 1.5)
        # 增强诱多欺骗敏感度：将 deception_lure_long_intensity_D 整合到看涨背离的纯度调制器中
        norm_deception_lure_long = get_adaptive_mtf_normalized_score(deception_lure_long_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        bullish_purity_modulator = (1 + norm_deception_multi_tf.clip(upper=0).abs() * purity_weights.get('deception_inverse', 0.5) * purity_context_mod) * \
                                   (1 - norm_wash_trade_multi_tf * purity_weights.get('wash_trade_inverse', 0.5) * purity_context_mod) * \
                                   (1 - norm_deception_lure_long * deception_lure_long_penalty_factor) # 新增诱多惩罚
        bearish_purity_modulator = (1 + norm_deception_multi_tf.clip(lower=0) * purity_weights.get('deception_inverse', 0.5) * purity_context_mod) * \
                                   (1 - norm_wash_trade_multi_tf * purity_weights.get('wash_trade_inverse', 0.5) * purity_context_mod)
        # --- 2. 意图确认 (Intent Confirmation) ---
        bullish_conviction_confirm = axiom_conviction.clip(lower=0) * confirmation_weights.get('conviction_positive', 0.5)
        bearish_conviction_confirm = axiom_conviction.clip(upper=0).abs() * confirmation_weights.get('conviction_positive', 0.5)
        bullish_flow_momentum_confirm = axiom_flow_momentum.clip(lower=0) * confirmation_weights.get('flow_momentum_positive', 0.5)
        bearish_flow_momentum_confirm = axiom_flow_momentum.clip(upper=0).abs() * confirmation_weights.get('flow_momentum_positive', 0.5)
        bullish_confirmation_score = (bullish_conviction_confirm + bullish_flow_momentum_confirm).clip(0, 1)
        bearish_confirmation_score = (bearish_conviction_confirm + bearish_flow_momentum_confirm).clip(0, 1)
        # --- 3. 情境校准 (Contextual Calibration) ---
        norm_strategic_posture = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_STRATEGIC_POSTURE', 0.0, method_name=method_name)
        bullish_posture_mod = norm_strategic_posture.clip(lower=0) * context_modulator_weights.get('strategic_posture', 0.4)
        bearish_posture_mod = norm_strategic_posture.clip(upper=0).abs() * context_modulator_weights.get('strategic_posture', 0.4)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        credibility_mod = norm_flow_credibility * context_modulator_weights.get('flow_credibility', 0.3)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        bullish_retail_context_mod = norm_retail_panic * retail_panic_fomo_sensitivity * context_modulator_weights.get('retail_panic_fomo_context', 0.3)
        bearish_retail_context_mod = norm_retail_fomo * retail_panic_fomo_sensitivity * context_modulator_weights.get('retail_panic_fomo_context', 0.3)
        # 新增：前期弱势惩罚
        prior_weakness_penalty = pd.Series(0.0, index=df_index)
        if prior_weakness_enabled:
            recent_pct_change_min = pct_change_raw.rolling(window=pct_change_window, min_periods=1).min().fillna(0)
            is_significant_drop = (recent_pct_change_min < pct_change_threshold).astype(float)
            norm_downward_momentum = get_adaptive_mtf_normalized_score(downward_momentum_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
            # 强化惩罚逻辑：当前期弱势和下跌动能同时存在时，惩罚力度更大
            prior_weakness_penalty = (is_significant_drop * (1 + norm_downward_momentum * downward_momentum_weight * 2)) * prior_weakness_penalty_factor
            prior_weakness_penalty = prior_weakness_penalty.clip(0, 1) # 确保惩罚因子在0-1之间
        # 新增：趋势延续性调制
        norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        # 调整趋势延续性调制器的计算，使其在趋势活力低时能更有效地降低看涨信号
        # 例如，当 norm_trend_vitality 接近 0 时，modulator 接近 0.1；当 norm_trend_vitality 接近 1 时，modulator 接近 1.0
        trend_continuity_modulator = (0.1 + norm_trend_vitality * 0.9).clip(0.1, 1.0) # 确保最低为0.1，最高为1.0
        # 新增：散户狂热反向指标
        retail_fomo_penalty = norm_retail_fomo * 0.5 # 散户狂热时，惩罚看涨信号
        bullish_context_modulator = (1 + bullish_posture_mod + credibility_mod + bullish_retail_context_mod - prior_weakness_penalty - retail_fomo_penalty).clip(0.1, 2.0) # 整合前期弱势惩罚和散户狂热惩罚
        bearish_context_modulator = (1 + bearish_posture_mod + credibility_mod + bearish_retail_context_mod).clip(0.1, 2.0)
        # --- V4.0 升级: 双极性多时间框架共振/背离因子 (Bipolar MTF Resonance/Divergence Factor) ---
        short_periods = [5, 13]
        long_periods = [21, 55]
        # 传递预取数据
        nmfnf_bipolar_resonance_factor = self._calculate_mtf_cohesion_divergence(df, 'NMFNF_D', short_periods, long_periods, True, self.tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        conviction_bipolar_resonance_factor = self._calculate_mtf_cohesion_divergence(df, 'main_force_conviction_index_D', short_periods, long_periods, True, self.tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        mtf_bipolar_resonance_factor = (
            nmfnf_bipolar_resonance_factor * mtf_resonance_factor_weights.get('nmfnf_cohesion', 0.6) +
            conviction_bipolar_resonance_factor * mtf_resonance_factor_weights.get('conviction_cohesion', 0.4)
        ).clip(-1, 1)
        # --- V4.0 升级: 更全面的微观意图强度 (Comprehensive Micro-Intent Strength) ---
        norm_order_book_imbalance = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, self.tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=self.tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_micro_impact_elasticity = get_adaptive_mtf_normalized_bipolar_score(micro_impact_elasticity_raw, df_index, self.tf_weights_ff)
        norm_main_force_t0_efficiency = get_adaptive_mtf_normalized_bipolar_score(main_force_t0_efficiency_raw, df_index, self.tf_weights_ff)
        # V4.1 整合新增资金指标到微观意图强度
        norm_dip_buy_absorption_strength = get_adaptive_mtf_normalized_score(dip_buy_absorption_strength_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_panic_buy_absorption_contribution = get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_opening_buy_strength = get_adaptive_mtf_normalized_score(opening_buy_strength_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_pre_closing_buy_posture = get_adaptive_mtf_normalized_score(pre_closing_buy_posture_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_closing_auction_buy_ambush = get_adaptive_mtf_normalized_score(closing_auction_buy_ambush_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_bid_side_liquidity = get_adaptive_mtf_normalized_score(bid_side_liquidity_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_main_force_t0_buy_efficiency = get_adaptive_mtf_normalized_score(main_force_t0_buy_efficiency_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_buy_flow_efficiency_index = get_adaptive_mtf_normalized_score(buy_flow_efficiency_index_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_buy_order_book_clearing_rate = get_adaptive_mtf_normalized_score(buy_order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_vwap_buy_control_strength = get_adaptive_mtf_normalized_score(vwap_buy_control_strength_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_main_force_vwap_up_guidance = get_adaptive_mtf_normalized_score(main_force_vwap_up_guidance_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_vwap_cross_up_intensity = get_adaptive_mtf_normalized_score(vwap_cross_up_intensity_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_dip_sell_pressure_resistance = get_adaptive_mtf_normalized_score(dip_sell_pressure_resistance_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_panic_sell_volume_contribution = get_adaptive_mtf_normalized_score(panic_sell_volume_contribution_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_opening_sell_strength = get_adaptive_mtf_normalized_score(opening_sell_strength_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_pre_closing_sell_posture = get_adaptive_mtf_normalized_score(pre_closing_sell_posture_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_closing_auction_sell_ambush = get_adaptive_mtf_normalized_score(closing_auction_sell_ambush_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_ask_side_liquidity = get_adaptive_mtf_normalized_score(ask_side_liquidity_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_main_force_on_peak_sell_flow = get_adaptive_mtf_normalized_score(main_force_on_peak_sell_flow_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_main_force_t0_sell_efficiency = get_adaptive_mtf_normalized_score(main_force_t0_sell_efficiency_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_sell_flow_efficiency_index = get_adaptive_mtf_normalized_score(sell_flow_efficiency_index_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_sell_order_book_clearing_rate = get_adaptive_mtf_normalized_score(sell_order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_vwap_sell_control_strength = get_adaptive_mtf_normalized_score(vwap_sell_control_strength_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_main_force_vwap_down_guidance = get_adaptive_mtf_normalized_score(main_force_vwap_down_guidance_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_vwap_cross_down_intensity = get_adaptive_mtf_normalized_score(vwap_cross_down_intensity_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        total_micro_buy_strength = (
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
            norm_vwap_cross_up_intensity * micro_buy_power_weights.get('vwap_cross_up_intensity', 0.05)
        ).clip(0, 1)
        total_micro_sell_strength = (
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
            norm_vwap_cross_down_intensity * micro_sell_power_weights.get('vwap_cross_down_intensity', 0.05)
        ).clip(0, 1)
        micro_intent_strength = (
            norm_order_book_imbalance * micro_intent_signals_weights.get('order_book_imbalance', 0.3) +
            (norm_sell_exhaustion - norm_buy_exhaustion) * micro_intent_signals_weights.get('exhaustion_rate', 0.3) +
            norm_micro_impact_elasticity * micro_intent_signals_weights.get('micro_impact_elasticity', 0.2) +
            norm_main_force_t0_efficiency * micro_intent_signals_weights.get('main_force_t0_efficiency', 0.2) +
            (total_micro_buy_strength - total_micro_sell_strength) * 0.5
        ).clip(-1, 1)
        micro_macro_divergence_factor = (
            micro_intent_strength * micro_macro_divergence_weights.get('micro_intent_strength', 0.5) -
            axiom_flow_momentum * micro_macro_divergence_weights.get('macro_flow_momentum', 0.5)
        ).clip(-1, 1)
        bullish_micro_macro_mod = (1 + micro_macro_divergence_factor.clip(lower=0))
        bearish_micro_macro_mod = (1 + micro_macro_divergence_factor.clip(upper=0).abs())
        # --- V4.0 升级: 看涨/看跌专属动态非线性指数 (Bullish/Bearish Adaptive Non-linear Exponents) ---
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        # norm_trend_vitality 已经在上面计算过
        norm_flow_credibility_exp = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        bullish_exponent_mod_factor = (
            norm_trend_vitality * bullish_exponent_context_factors.get('trend_vitality', 0.7) +
            (1 - norm_volatility_instability) * bullish_exponent_context_factors.get('volatility_inverse', 0.3)
        ).clip(0, 1)
        bullish_dynamic_non_linear_exponent = non_linear_exponent_base * (1 + bullish_exponent_mod_factor * dynamic_context_modulator_sensitivity.get('trend_vitality', 0.1))
        bearish_exponent_mod_factor = (
            norm_volatility_instability * bearish_exponent_context_factors.get('volatility_instability', 0.6) +
            (1 - norm_flow_credibility_exp) * bearish_exponent_context_factors.get('flow_credibility_inverse', 0.4)
        ).clip(0, 1)
        bearish_dynamic_non_linear_exponent = non_linear_exponent_base * (1 + bearish_exponent_mod_factor * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))
        # --- 4. 非线性融合 (Non-linear Fusion) ---
        # 计算产品前，确保所有 Series 都已填充 NaN
        bullish_product_components = [
            bullish_base_divergence,
            (bullish_purity_modulator * purity_context_mod).clip(0.1, 2.0),
            (bullish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0),
            (bullish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0),
            (1 + mtf_bipolar_resonance_factor.clip(lower=0)),
            bullish_micro_macro_mod,
            trend_continuity_modulator # 新增趋势延续性调制
        ]
        # 初始化产品为第一个组件，并确保其不为0，除非原始意图就是0
        # 修复Bug：确保基数在幂运算前不会因为 NaN 或 0 而导致结果异常
        bullish_product_before_pow = bullish_product_components[0].fillna(0.0).astype(np.float64) # 确保为float64
        # 逐个乘以其他组件，并确保每个组件都填充NaN为1.0（乘法中性元素）
        for comp in bullish_product_components[1:]:
            bullish_product_before_pow *= comp.fillna(1.0).astype(np.float64) # 确保为float64
        # 修复Bug：在进行幂运算前，对基数进行 clip(lower=1e-9) 处理，避免 0 的幂运算问题
        # 并且确保幂运算结果的 dtype 为 float32
        bullish_divergence_score = bullish_product_before_pow.clip(lower=1e-9).pow(bullish_dynamic_non_linear_exponent.astype(np.float64)).astype(np.float32).clip(0, 1)
        # --- 调试探针：产品计算结果 ---
        if is_debug_enabled and probe_ts and probe_ts in df_index:
            print(f"        [探针] {method_name} @ {probe_ts.strftime('%Y-%m-%d')}")
            print(f"          - bullish_base_divergence: {bullish_base_divergence.loc[probe_ts]:.8f}")
            print(f"          - bullish_purity_modulator * purity_context_mod: {(bullish_purity_modulator * purity_context_mod).clip(0.1, 2.0).loc[probe_ts]:.8f}")
            print(f"          - bullish_confirmation_score * (1 + norm_volatility_instability * ...): {(bullish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0).loc[probe_ts]:.8f}")
            print(f"          - bullish_context_modulator * (1 + norm_flow_credibility * ...): {(bullish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0).loc[probe_ts]:.8f}")
            print(f"          - (1 + mtf_bipolar_resonance_factor.clip(lower=0)): {(1 + mtf_bipolar_resonance_factor.clip(lower=0)).loc[probe_ts]:.8f}")
            print(f"          - bullish_micro_macro_mod: {bullish_micro_macro_mod.loc[probe_ts]:.8f}")
            print(f"          - trend_continuity_modulator: {trend_continuity_modulator.loc[probe_ts]:.8f}")
            print(f"          - bullish_product_before_pow: {bullish_product_before_pow.loc[probe_ts]:.8f}")
            print(f"          - bullish_dynamic_non_linear_exponent: {bullish_dynamic_non_linear_exponent.loc[probe_ts]:.8f}")
            print(f"          - bullish_divergence_score (after pow and clip): {bullish_divergence_score.loc[probe_ts]:.8f}")
        # Bug Fix: 确保如果 bullish_base_divergence 为 0，则最终分数也为 0
        # 使用一个小的 epsilon 来处理浮点数比较
        bullish_divergence_score = bullish_divergence_score.where(bullish_base_divergence > 1e-9, 0.0)
        bearish_product_components = [
            bearish_base_divergence,
            (bearish_purity_modulator * purity_context_mod).clip(0.1, 2.0),
            (bearish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0),
            (bearish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0),
            (1 + mtf_bipolar_resonance_factor.clip(upper=0).abs()),
            bearish_micro_macro_mod
        ]
        bearish_product_before_pow = bearish_product_components[0].fillna(0.0).astype(np.float64) # 确保为float64
        for comp in bearish_product_components[1:]:
            bearish_product_before_pow *= comp.fillna(1.0).astype(np.float64) # 确保为float64
        bearish_divergence_score = bearish_product_before_pow.clip(lower=1e-9).pow(bearish_dynamic_non_linear_exponent.astype(np.float64)).astype(np.float32).clip(0, 1)
        bearish_divergence_score = bearish_divergence_score.where(bearish_base_divergence > 1e-9, 0.0) # 同样对看跌信号进行门控
        bullish_divergence_score = bullish_divergence_score.where(bullish_divergence_score > bullish_divergence_threshold, 0.0)
        bearish_divergence_score = bearish_divergence_score.where(bearish_divergence_score > bearish_divergence_threshold, 0.0)
        print(f"    -> [资金流层] 看涨 (Bullish): {bullish_divergence_score.mean():.4f}")
        print(f"    -> [资金流层] 看跌 (Bearish): {bearish_divergence_score.mean():.4f}")
        return bullish_divergence_score.astype(np.float32), bearish_divergence_score.astype(np.float32)

    def _diagnose_axiom_intent_purity(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 意图韧性与适应性版】资金流公理：意图纯度。
        旨在量化主导资金流意图的清晰度、执行效率、抗操纵性、韧性与适应性，并捕捉其动态演化。
        它通过多时间维度动态捕捉、更丰富的资金流原始数据融合、非线性与自适应的判定逻辑，
        融合了流量方向清晰度、执行质量与效率、欺骗与操纵过滤、微观意图凝聚力、意图稳定性与抗干扰性以及情境强化六大维度。
        高分代表主导资金流意图清晰、执行高效、不易被操纵、具备韧性与适应性；
        负分代表意图模糊、执行低效、存在显著操纵或缺乏韧性。
        """
        df_index = df.index
        p_conf_ff = self.p_conf_ff
        aip_params = get_param_value(p_conf_ff.get('axiom_intent_purity_params'), {})
        print(f"    -> [资金流层] 正在诊断 资金流公理：意图纯度 (V3.0 · 意图韧性与适应性版)...")
        tf_weights_ff = self.tf_weights_ff
        flow_clarity_weights = get_param_value(aip_params.get('flow_clarity_weights'), {})
        execution_quality_weights = get_param_value(aip_params.get('execution_quality_weights'), {})
        deception_filter_weights = get_param_value(aip_params.get('deception_filter_weights'), {})
        micro_intent_cohesion_weights = get_param_value(aip_params.get('micro_intent_cohesion_weights'), {})
        intent_stability_weights = get_param_value(aip_params.get('intent_stability_weights'), {})
        context_reinforcement_weights = get_param_value(aip_params.get('context_reinforcement_weights'), {})
        mtf_periods_short = get_param_value(aip_params.get('mtf_periods_short'), [5, 13])
        mtf_periods_long = get_param_value(aip_params.get('mtf_periods_long'), [21, 34, 55])
        mtf_dynamic_weights = get_param_value(aip_params.get('mtf_dynamic_weights'), {"short": 0.6, "long": 0.4})
        main_dimension_adaptive_weights = get_param_value(aip_params.get('main_dimension_adaptive_weights'), {})
        final_fusion_exponent_base = get_param_value(aip_params.get('final_fusion_exponent_base'), 1.2)
        final_fusion_exponent_mod_sensitivity = get_param_value(aip_params.get('final_fusion_exponent_mod_sensitivity'), {})
        intent_divergence_mod_sensitivity = get_param_value(aip_params.get('intent_divergence_mod_sensitivity'), 0.5)
        smoothing_ema_span = get_param_value(aip_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(aip_params.get('dynamic_evolution_base_weights'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(aip_params.get('dynamic_evolution_context_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(aip_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(aip_params.get('dynamic_evolution_context_modulator_2'), 'flow_credibility_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(aip_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        # --- 信号依赖校验 ---
        required_signals = [
            'main_force_flow_directionality_D', 'main_force_flow_gini_D', 'NMFNF_D', 'main_force_ofi_D',
            'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D', 'SMART_MONEY_HM_NET_BUY_D',
            'main_force_t0_efficiency_D', 'main_force_slippage_index_D', 'main_force_execution_alpha_D',
            'vwap_control_strength_D', 'order_book_clearing_rate_D', 'microstructure_efficiency_index_D',
            'micro_price_impact_asymmetry_D', 'market_impact_cost_D', 'buy_sweep_intensity_D', 'sell_sweep_intensity_D',
            'deception_index_D', 'wash_trade_intensity_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'covert_accumulation_signal_D', 'covert_distribution_signal_D', 'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D',
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'liquidity_authenticity_score_D', 'order_flow_imbalance_score_D',
            'flow_credibility_index_D', 'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'trend_vitality_index_D', 'strategic_phase_score_D', 'panic_selling_cascade_D',
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name,
            'SCORE_FF_AXIOM_DIVERGENCE' # 用于意图背离惩罚/奖励
        ]
        # 动态添加MTF斜率和加速度信号
        signal_bases_for_mtf = [
            'main_force_flow_directionality_D', 'main_force_flow_gini_D', 'NMFNF_D', 'main_force_ofi_D',
            'net_lg_amount_calibrated_D', 'SMART_MONEY_HM_NET_BUY_D',
            'main_force_t0_efficiency_D', 'main_force_slippage_index_D', 'main_force_execution_alpha_D',
            'vwap_control_strength_D', 'order_book_clearing_rate_D', 'microstructure_efficiency_index_D',
            'micro_price_impact_asymmetry_D', 'market_impact_cost_D', 'buy_sweep_intensity_D', 'sell_sweep_intensity_D',
            'deception_index_D', 'wash_trade_intensity_D', 'deception_lure_long_intensity_D', 'deception_lure_short_intensity_D',
            'covert_accumulation_signal_D', 'covert_distribution_signal_D',
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'liquidity_authenticity_score_D', 'order_flow_imbalance_score_D',
            'flow_credibility_index_D', 'market_sentiment_score_D', 'VOLATILITY_INSTABILITY_INDEX_21d_D',
            'trend_vitality_index_D', 'strategic_phase_score_D', 'panic_selling_cascade_D'
        ]
        all_mtf_periods = list(set(mtf_periods_short + mtf_periods_long))
        for signal_base in signal_bases_for_mtf:
            for p in all_mtf_periods:
                required_signals.append(f'SLOPE_{p}_{signal_base}')
                required_signals.append(f'ACCEL_{p}_{signal_base}')
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_intent_purity", atomic_states=self.strategy.atomic_states):
            return pd.Series(0.0, index=df.index)
        # 预取所有斜率和加速度数据到单个字典
        all_pre_fetched_slopes_accels = {}
        for signal_base in signal_bases_for_mtf:
            for p in all_mtf_periods:
                all_pre_fetched_slopes_accels[f'SLOPE_{p}_{signal_base}'] = self._get_safe_series(df, df, f'SLOPE_{p}_{signal_base}', 0.0, method_name="_diagnose_axiom_intent_purity")
                all_pre_fetched_slopes_accels[f'ACCEL_{p}_{signal_base}'] = self._get_safe_series(df, df, f'ACCEL_{p}_{signal_base}', 0.0, method_name="_diagnose_axiom_intent_purity")
        # --- 原始数据获取 (用于探针和计算) ---
        raw_data_cache = {}
        for signal_name in required_signals:
            if signal_name not in all_pre_fetched_slopes_accels and signal_name not in self.strategy.atomic_states:
                raw_data_cache[signal_name] = self._get_safe_series(df, df, signal_name, 0.0, method_name="_diagnose_axiom_intent_purity")
        # --- 1. 流量方向清晰度 (Flow Directional Clarity) ---
        norm_main_force_flow_directionality_mtf = self._calculate_mtf_cohesion_divergence(df, 'main_force_flow_directionality_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_main_force_flow_gini_inverted_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'main_force_flow_gini_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_nmfnf_net_flow_mtf = self._calculate_mtf_cohesion_divergence(df, 'NMFNF_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_main_force_ofi_mtf = self._calculate_mtf_cohesion_divergence(df, 'main_force_ofi_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_net_lg_xl_flow_mtf = self._calculate_mtf_cohesion_divergence(df, 'net_lg_amount_calibrated_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_smart_money_net_buy_mtf = self._calculate_mtf_cohesion_divergence(df, 'SMART_MONEY_HM_NET_BUY_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        flow_clarity_components = {
            'main_force_flow_directionality_mtf': (norm_main_force_flow_directionality_mtf + 1) / 2,
            'main_force_flow_gini_inverted_mtf': norm_main_force_flow_gini_inverted_mtf,
            'nmfnf_net_flow_mtf': (norm_nmfnf_net_flow_mtf + 1) / 2,
            'main_force_ofi_mtf': (norm_main_force_ofi_mtf + 1) / 2,
            'net_lg_xl_flow_mtf': (norm_net_lg_xl_flow_mtf + 1) / 2,
            'smart_money_net_buy_mtf': (norm_smart_money_net_buy_mtf + 1) / 2
        }
        flow_clarity_score_unipolar = _robust_geometric_mean(flow_clarity_components, flow_clarity_weights, df_index)
        flow_clarity_score = (flow_clarity_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 2. 执行质量与效率 (Execution Quality & Efficiency) ---
        norm_main_force_t0_efficiency_mtf = self._calculate_mtf_cohesion_divergence(df, 'main_force_t0_efficiency_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_main_force_slippage_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'main_force_slippage_index_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_main_force_execution_alpha_mtf = self._calculate_mtf_cohesion_divergence(df, 'main_force_execution_alpha_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_vwap_control_strength_mtf = self._calculate_mtf_cohesion_divergence(df, 'vwap_control_strength_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_order_book_clearing_rate_mtf = self._calculate_mtf_cohesion_divergence(df, 'order_book_clearing_rate_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_microstructure_efficiency_mtf = self._calculate_mtf_cohesion_divergence(df, 'microstructure_efficiency_index_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_market_impact_cost_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'market_impact_cost_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_buy_sweep_intensity_mtf = self._calculate_mtf_cohesion_divergence(df, 'buy_sweep_intensity_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_sell_sweep_intensity_mtf = self._calculate_mtf_cohesion_divergence(df, 'sell_sweep_intensity_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        sweep_intensity_bipolar_mtf = (norm_buy_sweep_intensity_mtf - norm_sell_sweep_intensity_mtf).clip(-1, 1)
        execution_quality_components = {
            'main_force_t0_efficiency_mtf': (norm_main_force_t0_efficiency_mtf + 1) / 2,
            'main_force_slippage_inverse_mtf': norm_main_force_slippage_inverse_mtf,
            'main_force_execution_alpha_mtf': (norm_main_force_execution_alpha_mtf + 1) / 2,
            'vwap_control_strength_mtf': (norm_vwap_control_strength_mtf + 1) / 2,
            'order_book_clearing_rate_mtf': norm_order_book_clearing_rate_mtf,
            'microstructure_efficiency_mtf': (norm_microstructure_efficiency_mtf + 1) / 2,
            'market_impact_cost_inverse_mtf': norm_market_impact_cost_inverse_mtf,
            'sweep_intensity_bipolar_mtf': (sweep_intensity_bipolar_mtf + 1) / 2
        }
        execution_quality_score_unipolar = _robust_geometric_mean(execution_quality_components, execution_quality_weights, df_index)
        execution_quality_score = (execution_quality_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 3. 欺骗与操纵过滤 (Deception & Manipulation Filter) ---
        norm_deception_index_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'deception_index_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_wash_trade_intensity_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'wash_trade_intensity_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_deception_lure_long_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'deception_lure_long_intensity_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_deception_lure_short_mtf = self._calculate_mtf_cohesion_divergence(df, 'deception_lure_short_intensity_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_covert_accumulation_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'covert_accumulation_signal_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_covert_distribution_mtf = self._calculate_mtf_cohesion_divergence(df, 'covert_distribution_signal_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        deception_filter_components = {
            'deception_index_inverse_mtf': norm_deception_index_inverse_mtf,
            'wash_trade_intensity_inverse_mtf': norm_wash_trade_intensity_inverse_mtf,
            'deception_lure_long_inverse_mtf': norm_deception_lure_long_inverse_mtf,
            'deception_lure_short_mtf': norm_deception_lure_short_mtf,
            'covert_accumulation_inverse_mtf': norm_covert_accumulation_inverse_mtf,
            'covert_distribution_mtf': norm_covert_distribution_mtf
        }
        deception_filter_score_unipolar = _robust_geometric_mean(deception_filter_components, deception_filter_weights, df_index)
        deception_filter_score = (deception_filter_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 4. 微观意图凝聚力 (Micro-Intent Cohesion) ---
        norm_order_book_imbalance_mtf = self._calculate_mtf_cohesion_divergence(df, 'order_book_imbalance_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_buy_exhaustion_mtf = self._calculate_mtf_cohesion_divergence(df, 'buy_quote_exhaustion_rate_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_sell_exhaustion_mtf = self._calculate_mtf_cohesion_divergence(df, 'sell_quote_exhaustion_rate_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        buy_sell_exhaustion_bipolar_mtf = (norm_sell_exhaustion_mtf - norm_buy_exhaustion_mtf).clip(-1, 1)
        norm_liquidity_authenticity_mtf = self._calculate_mtf_cohesion_divergence(df, 'liquidity_authenticity_score_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_order_flow_imbalance_score_mtf = self._calculate_mtf_cohesion_divergence(df, 'order_flow_imbalance_score_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_micro_price_impact_asymmetry_mtf = self._calculate_mtf_cohesion_divergence(df, 'micro_price_impact_asymmetry_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        micro_intent_cohesion_components = {
            'order_book_imbalance_mtf': (norm_order_book_imbalance_mtf + 1) / 2,
            'buy_sell_exhaustion_bipolar_mtf': (buy_sell_exhaustion_bipolar_mtf + 1) / 2,
            'liquidity_authenticity_mtf': norm_liquidity_authenticity_mtf,
            'order_flow_imbalance_score_mtf': (norm_order_flow_imbalance_score_mtf + 1) / 2, # 修正此处变量名
            'micro_price_impact_asymmetry_mtf': (norm_micro_price_impact_asymmetry_mtf + 1) / 2
        }
        micro_intent_cohesion_score_unipolar = _robust_geometric_mean(micro_intent_cohesion_components, micro_intent_cohesion_weights, df_index)
        micro_intent_cohesion_score = (micro_intent_cohesion_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 5. 意图稳定性与抗干扰性 (Intent Stability & Resilience) ---
        nmfnf_raw = raw_data_cache['NMFNF_D']
        nmfnf_volatility = nmfnf_raw.rolling(window=norm_window, min_periods=1).std().fillna(0)
        norm_nmfnf_volatility_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'NMFNF_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        flow_credibility_raw = raw_data_cache['flow_credibility_index_D']
        flow_credibility_volatility = flow_credibility_raw.rolling(window=norm_window, min_periods=1).std().fillna(0)
        norm_flow_credibility_stability_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'flow_credibility_index_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_panic_selling_cascade_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'panic_selling_cascade_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_market_impact_resilience_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'market_impact_cost_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        intent_stability_components = {
            'nmfnf_volatility_inverse_mtf': norm_nmfnf_volatility_inverse_mtf,
            'flow_credibility_stability_mtf': norm_flow_credibility_stability_mtf,
            'panic_selling_cascade_inverse_mtf': norm_panic_selling_cascade_inverse_mtf,
            'market_impact_resilience_mtf': norm_market_impact_resilience_mtf
        }
        intent_stability_score_unipolar = _robust_geometric_mean(intent_stability_components, intent_stability_weights, df_index)
        intent_stability_score = (intent_stability_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 6. 情境强化 (Contextual Reinforcement) ---
        norm_flow_credibility_index_mtf = self._calculate_mtf_cohesion_divergence(df, 'flow_credibility_index_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_market_sentiment_score_mtf = self._calculate_mtf_cohesion_divergence(df, 'market_sentiment_score_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_volatility_instability_inverse_mtf = 1 - self._calculate_mtf_cohesion_divergence(df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_trend_vitality_mtf = self._calculate_mtf_cohesion_divergence(df, 'trend_vitality_index_D', mtf_periods_short, mtf_periods_long, False, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        norm_strategic_phase_mtf = self._calculate_mtf_cohesion_divergence(df, 'strategic_phase_score_D', mtf_periods_short, mtf_periods_long, True, tf_weights_ff, pre_fetched_data=all_pre_fetched_slopes_accels)
        context_reinforcement_components = {
            'flow_credibility_index_mtf': norm_flow_credibility_index_mtf,
            'market_sentiment_score_mtf': (norm_market_sentiment_score_mtf + 1) / 2,
            'volatility_instability_inverse_mtf': norm_volatility_instability_inverse_mtf,
            'trend_vitality_mtf': norm_trend_vitality_mtf,
            'strategic_phase_mtf': (norm_strategic_phase_mtf + 1) / 2
        }
        context_reinforcement_score_unipolar = _robust_geometric_mean(context_reinforcement_components, context_reinforcement_weights, df_index)
        context_reinforcement_score = (context_reinforcement_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 7. 主维度自适应融合 (Adaptive Fusion of Main Dimensions) ---
        norm_volatility_instability_raw = get_adaptive_mtf_normalized_score(raw_data_cache['VOLATILITY_INSTABILITY_INDEX_21d_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_market_sentiment_raw = get_adaptive_mtf_normalized_bipolar_score(raw_data_cache['market_sentiment_score_D'], df_index, tf_weights=tf_weights_ff)
        norm_flow_credibility_raw = get_adaptive_mtf_normalized_score(raw_data_cache['flow_credibility_index_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_trend_vitality_raw = get_adaptive_mtf_normalized_score(raw_data_cache['trend_vitality_index_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        volatility_sensitivity = main_dimension_adaptive_weights.get('volatility_sensitivity', 0.2)
        sentiment_sensitivity = main_dimension_adaptive_weights.get('sentiment_sensitivity', 0.15)
        flow_credibility_sensitivity = main_dimension_adaptive_weights.get('flow_credibility_sensitivity', 0.1)
        trend_vitality_sensitivity = main_dimension_adaptive_weights.get('trend_vitality_sensitivity', 0.1)
        # 动态调整权重
        dynamic_flow_clarity_weight = main_dimension_adaptive_weights.get('flow_clarity_base', 0.2) * (1 - norm_volatility_instability_raw * volatility_sensitivity + norm_flow_credibility_raw * flow_credibility_sensitivity)
        dynamic_execution_quality_weight = main_dimension_adaptive_weights.get('execution_quality_base', 0.2) * (1 + norm_volatility_instability_raw * volatility_sensitivity + norm_flow_credibility_raw * flow_credibility_sensitivity)
        dynamic_deception_filter_weight = main_dimension_adaptive_weights.get('deception_filter_base', 0.15) * (1 + norm_volatility_instability_raw * volatility_sensitivity - norm_flow_credibility_raw * flow_credibility_sensitivity)
        dynamic_micro_intent_cohesion_weight = main_dimension_adaptive_weights.get('micro_intent_cohesion_base', 0.15) * (1 + norm_market_sentiment_raw.abs() * sentiment_sensitivity + norm_trend_vitality_raw * trend_vitality_sensitivity)
        dynamic_intent_stability_weight = main_dimension_adaptive_weights.get('intent_stability_base', 0.15) * (1 - norm_volatility_instability_raw * volatility_sensitivity + norm_trend_vitality_raw * trend_vitality_sensitivity)
        dynamic_context_reinforcement_weight = main_dimension_adaptive_weights.get('context_reinforcement_base', 0.15) * (1 + norm_market_sentiment_raw.abs() * sentiment_sensitivity)
        total_dynamic_weights = dynamic_flow_clarity_weight + dynamic_execution_quality_weight + dynamic_deception_filter_weight + dynamic_micro_intent_cohesion_weight + dynamic_intent_stability_weight + dynamic_context_reinforcement_weight
        dynamic_flow_clarity_weight /= total_dynamic_weights
        dynamic_execution_quality_weight /= total_dynamic_weights
        dynamic_deception_filter_weight /= total_dynamic_weights
        dynamic_micro_intent_cohesion_weight /= total_dynamic_weights
        dynamic_intent_stability_weight /= total_dynamic_weights
        dynamic_context_reinforcement_weight /= total_dynamic_weights
        final_components_unipolar = {
            'flow_clarity': (flow_clarity_score + 1) / 2,
            'execution_quality': (execution_quality_score + 1) / 2,
            'deception_filter': (deception_filter_score + 1) / 2,
            'micro_intent_cohesion': (micro_intent_cohesion_score + 1) / 2,
            'intent_stability': (intent_stability_score + 1) / 2,
            'context_reinforcement': (context_reinforcement_score + 1) / 2
        }
        final_fusion_weights_dynamic = {
            'flow_clarity': dynamic_flow_clarity_weight,
            'execution_quality': dynamic_execution_quality_weight,
            'deception_filter': dynamic_deception_filter_weight,
            'micro_intent_cohesion': dynamic_micro_intent_cohesion_weight,
            'intent_stability': dynamic_intent_stability_weight,
            'context_reinforcement': dynamic_context_reinforcement_weight
        }
        base_intent_purity_score_unipolar = _robust_geometric_mean(final_components_unipolar, final_fusion_weights_dynamic, df_index)
        base_intent_purity_score = (base_intent_purity_score_unipolar * 2 - 1).clip(-1, 1)
        # --- 8. 动态融合指数与意图背离调制 (Dynamic Fusion Exponent & Intent Divergence Modulation) ---
        norm_volatility_instability_exp = get_adaptive_mtf_normalized_score(raw_data_cache['VOLATILITY_INSTABILITY_INDEX_21d_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_trend_vitality_exp = get_adaptive_mtf_normalized_score(raw_data_cache['trend_vitality_index_D'], df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_market_sentiment_exp = get_adaptive_mtf_normalized_bipolar_score(raw_data_cache['market_sentiment_score_D'], df_index, tf_weights=tf_weights_ff)
        dynamic_fusion_exponent = final_fusion_exponent_base * (
            1 + (1 - norm_volatility_instability_exp) * final_fusion_exponent_mod_sensitivity.get('volatility_instability', 0.3) +
            norm_trend_vitality_exp * final_fusion_exponent_mod_sensitivity.get('trend_vitality', 0.2) +
            norm_market_sentiment_exp.abs() * final_fusion_exponent_mod_sensitivity.get('market_sentiment', 0.1)
        )
        # 意图背离惩罚/奖励
        axiom_divergence = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_DIVERGENCE', 0.0, method_name="_diagnose_axiom_intent_purity")
        intent_divergence_modulator = 1 - axiom_divergence.abs() * intent_divergence_mod_sensitivity
        intent_divergence_modulator = intent_divergence_modulator.clip(0.5, 1.0) # 确保只惩罚，不奖励
        # 计算幂运算的基数
        base_for_power = base_intent_purity_score * intent_divergence_modulator
        # 处理负数基数和非整数指数，避免产生 NaN
        # 步骤：1. 获取基数的符号；2. 对基数的绝对值进行幂运算；3. 重新应用符号；4. 裁剪到 [-1, 1] 范围
        # dynamic_fusion_exponent 已经是 Series，确保其类型为浮点数
        signed_base = np.sign(base_for_power)
        abs_powered = base_for_power.abs().pow(dynamic_fusion_exponent)
        # 重新应用符号，并确保结果在 [-1, 1] 范围内
        final_modulated_score = (signed_base * abs_powered).clip(-1, 1)
        # --- 9. 意图纯度动态演化与前瞻性增强 (Dynamic Evolution & Foresight Enhancement) ---
        smoothed_base_score = final_modulated_score.ewm(span=smoothing_ema_span, adjust=False).mean()
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
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.6) * (1 - combined_evolution_context_mod)
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        final_score_components = {
            'base_score': (final_modulated_score + 1) / 2,
            'velocity': (norm_velocity + 1) / 2,
            'acceleration': (norm_acceleration + 1) / 2
        }
        final_score_weights = {
            'base_score': dynamic_base_score_weight,
            'velocity': dynamic_velocity_weight,
            'acceleration': dynamic_acceleration_weight
        }
        final_score_unipolar = _robust_geometric_mean(final_score_components, final_score_weights, df_index)
        final_score = (final_score_unipolar * 2 - 1).clip(-1, 1)
        print(f"    -> [资金流层] 意图纯度: {final_score_unipolar.mean():.4f}")
        return final_score.astype(np.float32)










