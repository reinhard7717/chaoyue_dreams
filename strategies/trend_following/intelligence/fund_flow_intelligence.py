import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar, get_adaptive_mtf_normalized_score

class FundFlowIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化资金流情报模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, data_source: Union[pd.DataFrame, Dict[str, pd.Series]], column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        【V1.1 · 上下文修复版】安全地从DataFrame或字典中获取Series，如果不存在则打印警告并返回默认Series。
        - 核心修复: 接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        df_index = df.index # 使用传入的 df.index
        if isinstance(data_source, pd.DataFrame):
            if column_name not in data_source.columns:
                print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少DataFrame数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            return data_source[column_name]
        elif isinstance(data_source, dict):
            if column_name not in data_source:
                print(f"    -> [资金流情报警告] 方法 '{method_name}' 缺少字典数据 '{column_name}'，使用默认值 {default_value}。")
                return pd.Series(default_value, index=df_index)
            series = data_source[column_name]
            if isinstance(series, pd.Series):
                return series.reindex(df_index, fill_value=default_value)
            else:
                return pd.Series(series, index=df_index)
        else:
            print(f"    -> [资金流情报警告] 方法 '{method_name}' 接收到未知数据源类型 {type(data_source)}，无法获取 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df_index)

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: List[str], method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
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
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。")
            return {}
        print("启动【V29.0 · 拐点洞察版】资金流情报分析...") # 修改: 更新版本号和名称
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        # --- 1. 计算所有原子公理 ---
        axiom_capital_signature = self._diagnose_axiom_capital_signature(df, norm_window)
        axiom_flow_structure_health = self._diagnose_axiom_flow_structure_health(df, norm_window)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_flow_momentum = self._diagnose_axiom_flow_momentum(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        # 修改代码行：移除多余的参数
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        # --- 2. 战略态势的向量合成 ---
        fusion_weights = get_param_value(p_conf.get('posture_fusion_weights'), {})
        penalty_factor = get_param_value(p_conf.get('dissonance_penalty_factor'), 0.25)
        attack_weights = fusion_weights.get('attack_group', {})
        structure_weights = fusion_weights.get('structure_group', {})
        context_weights = fusion_weights.get('context_group', {})
        attack_base = (axiom_conviction * attack_weights.get('conviction', 0.6) +
                       axiom_flow_momentum * attack_weights.get('flow_momentum', 0.4))
        attack_dissonance = abs(axiom_conviction - axiom_flow_momentum) / 2
        attack_score = attack_base * (1 - attack_dissonance * penalty_factor)
        structure_base = (axiom_consensus * structure_weights.get('consensus', 0.6) +
                          axiom_flow_structure_health * structure_weights.get('flow_health', 0.4))
        structure_dissonance = abs(axiom_consensus - axiom_flow_structure_health) / 2
        structure_score = structure_base * (1 - structure_dissonance * penalty_factor)
        context_modulator = (1 +
                             axiom_capital_signature * context_weights.get('capital_signature', 0.1) +
                             axiom_divergence * context_weights.get('divergence', 0.1)
                             ).clip(0.5, 1.5)
        posture_core = attack_score * (1 + structure_score) / 2
        strategic_posture_score = (posture_core * context_modulator).clip(-1, 1)
        # --- 3. 新增：和谐拐点计算 ---
        posture_velocity = strategic_posture_score.diff().fillna(0)
        posture_acceleration = posture_velocity.diff().fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_score(posture_velocity, df.index, ascending=True, tf_weights={3:1.0})
        norm_acceleration = get_adaptive_mtf_normalized_score(posture_acceleration, df.index, ascending=True, tf_weights={3:1.0})
        # 核心裁决：速度和加速度必须同时为正
        harmony_inflection_score = (norm_velocity.clip(lower=0) * norm_acceleration.clip(lower=0)).pow(0.5)
        # --- 4. 状态赋值 ---
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        all_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        all_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        all_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score.astype(np.float32)
        all_states['SCORE_FF_HARMONY_INFLECTION'] = harmony_inflection_score.astype(np.float32) # 新增: 和谐拐点机会信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        print(f"【V29.0 · 拐点洞察版】分析完成，生成 {len(all_states)} 个资金流原子及融合信号。") # 修改: 更新日志
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.0 · 多维时空与微观意图版】资金流公理四：诊断“资金流内部分歧与意图张力”
        - 核心升级1: 多时间框架张力融合：对核心分歧、结构性张力、诡道意图张力等维度，计算其在多个时间框架（5, 13, 21, 34, 55日）下的表现并加权融合。
        - 核心升级2: 微观资金流意图张力：引入订单簿不平衡、微观冲击弹性、买卖盘枯竭率等微观资金流信号，深化对主力盘口意图的洞察。
        - 核心升级3: 非线性融合与情境自适应权重：采用更复杂的非线性融合函数，并根据市场整体资金流健康度、波动性等情境因子动态调整各张力维度的融合权重。
        - 核心升级4: 张力演化趋势与预警：对最终张力分数进行速度和加速度分析，评估张力的演化趋势和潜在转折点。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资金流内部分歧与意图张力 (V4.0 · 多维时空与微观意图版)”公理...")
        # --- 探针: 原始输入 ---
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        df_index = df.index
        probe_date = None
        is_probe_active = False
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            temp_probe_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if temp_probe_date in df_index:
                probe_date = temp_probe_date
                is_probe_active = True
                print(f"    -> [资金流内部分歧与意图张力探针] @ {probe_date.date()}:")
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ad_params = get_param_value(p_conf_ff.get('axiom_divergence_params'), {})
        divergence_slope_periods = get_param_value(ad_params.get('divergence_slope_periods'), [5, 13, 21, 34, 55])
        raw_divergence_slope_weights = get_param_value(ad_params.get('divergence_slope_weights'), {"5": 0.4, "13": 0.3, "21": 0.2, "34": 0.05, "55": 0.05})
        # 过滤掉非数值的权重，例如 'description' 字段
        divergence_slope_weights = {k: v for k, v in raw_divergence_slope_weights.items() if isinstance(v, (int, float))}
        energy_injection_signals_weights = get_param_value(ad_params.get('energy_injection_signals'), {'main_force_activity_ratio_D': 0.4, 'main_force_ofi_D': 0.3, 'micro_impact_elasticity_D': 0.3})
        persistence_window = get_param_value(ad_params.get('persistence_window'), 13)
        amplification_factor = get_param_value(ad_params.get('amplification_factor'), 1.5)
        retail_sentiment_mod_sensitivity = get_param_value(ad_params.get('retail_sentiment_mod_sensitivity'), 0.2)
        deception_mod_sensitivity = get_param_value(ad_params.get('deception_mod_sensitivity'), 0.3)
        raw_divergence_component_weights = get_param_value(ad_params.get('divergence_component_weights'), {'core_divergence': 0.3, 'structural_tension': 0.25, 'deceptive_tension': 0.25, 'micro_intent_tension': 0.2})
        # 过滤掉非数值的权重，例如 'description' 字段
        divergence_component_weights = {k: v for k, v in raw_divergence_component_weights.items() if isinstance(v, (int, float))}
        micro_intent_tension_signals_weights = get_param_value(ad_params.get('micro_intent_tension_signals'), {'order_book_imbalance_D': 0.5, 'buy_quote_exhaustion_rate_D': 0.25, 'sell_quote_exhaustion_rate_D': 0.25})
        non_linear_fusion_exponent = get_param_value(ad_params.get('non_linear_fusion_exponent'), 0.8)
        adaptive_weight_modulator_signal_1_name = get_param_value(ad_params.get('adaptive_weight_modulator_signal_1'), 'flow_credibility_index_D')
        adaptive_weight_modulator_signal_2_name = get_param_value(ad_params.get('adaptive_weight_modulator_signal_2'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        adaptive_weight_sensitivity_credibility = get_param_value(ad_params.get('adaptive_weight_sensitivity_credibility'), 0.2)
        adaptive_weight_sensitivity_volatility = get_param_value(ad_params.get('adaptive_weight_sensitivity_volatility'), 0.1)
        smoothing_ema_span = get_param_value(ad_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ad_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(ad_params.get('dynamic_evolution_context_modulator_signal_1'), 'main_force_conviction_index_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(ad_params.get('dynamic_evolution_context_sensitivity_1'), 0.2)
        # --- 辅助函数: 融合多时间框架斜率 ---
        # 辅助函数已移除，直接在下方计算
        # --- 信号依赖校验 ---
        required_signals = []
        for p in divergence_slope_periods:
            required_signals.extend([
                f'SLOPE_{p}_NMFNF_D',
                f'SLOPE_{p}_main_force_conviction_index_D',
                f'SLOPE_{p}_net_lg_amount_calibrated_D',
                f'SLOPE_{p}_retail_net_flow_calibrated_D',
                f'SLOPE_{p}_deception_index_D',
                f'SLOPE_{p}_wash_trade_intensity_D'
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
            dynamic_evolution_context_modulator_signal_1_name
        ])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        nmfnf_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_NMFNF_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        mf_conviction_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        lg_flow_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        retail_flow_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_retail_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        retail_fomo_premium_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        retail_panic_surrender_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        deception_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_deception_index_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_divergence")
        wash_trade_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_divergence")
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence")
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence")
        mf_activity_ratio_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', 0.0, method_name="_diagnose_axiom_divergence")
        mf_ofi_raw = self._get_safe_series(df, df, 'main_force_ofi_D', 0.0, method_name="_diagnose_axiom_divergence")
        micro_impact_elasticity_raw = self._get_safe_series(df, df, 'micro_impact_elasticity_D', 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_1_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_2_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_divergence")
        dynamic_evolution_context_modulator_1_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_divergence")
        if is_probe_active:
            print(f"       - 原料: retail_fomo_premium_index_D (raw): {retail_fomo_premium_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: retail_panic_surrender_index_D (raw): {retail_panic_surrender_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: flow_credibility_index_D (raw): {flow_credibility_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: wash_trade_intensity_D (raw): {wash_trade_intensity_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: main_force_activity_ratio_D (raw): {mf_activity_ratio_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: main_force_ofi_D (raw): {mf_ofi_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: micro_impact_elasticity_D (raw): {micro_impact_elasticity_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: order_book_imbalance_D (raw): {order_book_imbalance_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: buy_quote_exhaustion_rate_D (raw): {buy_exhaustion_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: sell_quote_exhaustion_rate_D (raw): {sell_exhaustion_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: {adaptive_weight_modulator_signal_1_name} (raw): {adaptive_weight_modulator_1_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: {adaptive_weight_modulator_signal_2_name} (raw): {adaptive_weight_modulator_2_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: {dynamic_evolution_context_modulator_signal_1_name} (raw): {dynamic_evolution_context_modulator_1_raw.loc[probe_date]:.4f}")
            for p in divergence_slope_periods:
                print(f"       - 原料: SLOPE_{p}_NMFNF_D (raw): {nmfnf_slopes_raw[p].loc[probe_date]:.4f}")
                print(f"       - 原料: SLOPE_{p}_main_force_conviction_index_D (raw): {mf_conviction_slopes_raw[p].loc[probe_date]:.4f}")
                print(f"       - 原料: SLOPE_{p}_net_lg_amount_calibrated_D (raw): {lg_flow_slopes_raw[p].loc[probe_date]:.4f}")
                print(f"       - 原料: SLOPE_{p}_retail_net_flow_calibrated_D (raw): {retail_flow_slopes_raw[p].loc[probe_date]:.4f}")
                print(f"       - 原料: SLOPE_{p}_deception_index_D (raw): {deception_slopes_raw[p].loc[probe_date]:.4f}")
                print(f"       - 原料: SLOPE_{p}_wash_trade_intensity_D (raw): {wash_trade_slopes_raw[p].loc[probe_date]:.4f}")
        # --- 1. 核心分歧向量 (Core Divergence Vector) ---
        norm_nmfnf_slope_mtf = pd.Series(0.0, index=df_index)
        norm_mf_conviction_slope_mtf = pd.Series(0.0, index=df_index)
        total_slope_weight = sum(divergence_slope_weights.values())
        if total_slope_weight > 0:
            for p_str, weight in divergence_slope_weights.items():
                p = int(p_str)
                norm_nmfnf_slope_mtf += get_adaptive_mtf_normalized_bipolar_score(nmfnf_slopes_raw[p], df_index, tf_weights_ff) * weight
                norm_mf_conviction_slope_mtf += get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slopes_raw[p], df_index, tf_weights_ff) * weight
            norm_nmfnf_slope_mtf /= total_slope_weight
            norm_mf_conviction_slope_mtf /= total_slope_weight
        core_divergence_score = (norm_nmfnf_slope_mtf - norm_mf_conviction_slope_mtf).clip(-1, 1)
        if is_probe_active:
            print(f"       - 过程: norm_nmfnf_slope_mtf (CDV): {norm_nmfnf_slope_mtf.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_mf_conviction_slope_mtf (CDV): {norm_mf_conviction_slope_mtf.loc[probe_date]:.4f}")
            print(f"       - 过程: core_divergence_score: {core_divergence_score.loc[probe_date]:.4f}")
        # --- 2. 结构性张力 (Structural Tension) ---
        norm_lg_flow_slope_mtf = pd.Series(0.0, index=df_index)
        norm_retail_flow_slope_mtf = pd.Series(0.0, index=df_index)
        if total_slope_weight > 0:
            for p_str, weight in divergence_slope_weights.items():
                p = int(p_str)
                norm_lg_flow_slope_mtf += get_adaptive_mtf_normalized_bipolar_score(lg_flow_slopes_raw[p], df_index, tf_weights_ff) * weight
                norm_retail_flow_slope_mtf += get_adaptive_mtf_normalized_bipolar_score(retail_flow_slopes_raw[p], df_index, tf_weights_ff) * weight
            norm_lg_flow_slope_mtf /= total_slope_weight
            norm_retail_flow_slope_mtf /= total_slope_weight
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        structural_divergence_base = (norm_lg_flow_slope_mtf - norm_retail_flow_slope_mtf)
        retail_modulator = (1 - norm_retail_fomo * retail_sentiment_mod_sensitivity) + (norm_retail_panic * retail_sentiment_mod_sensitivity)
        structural_tension_score = (structural_divergence_base * retail_modulator).clip(-1, 1)
        if is_probe_active:
            print(f"       - 过程: norm_lg_flow_slope_mtf (ST): {norm_lg_flow_slope_mtf.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_retail_flow_slope_mtf (ST): {norm_retail_flow_slope_mtf.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_retail_fomo (ST): {norm_retail_fomo.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_retail_panic (ST): {norm_retail_panic.loc[probe_date]:.4f}")
            print(f"       - 过程: retail_modulator (ST): {retail_modulator.loc[probe_date]:.4f}")
            print(f"       - 过程: structural_tension_score: {structural_tension_score.loc[probe_date]:.4f}")
        # --- 3. 诡道意图张力 (Deceptive Intent Tension) ---
        norm_deception_slope_mtf = pd.Series(0.0, index=df_index)
        norm_wash_trade_slope_mtf = pd.Series(0.0, index=df_index)
        if total_slope_weight > 0:
            for p_str, weight in divergence_slope_weights.items():
                p = int(p_str)
                norm_deception_slope_mtf += get_adaptive_mtf_normalized_bipolar_score(deception_slopes_raw[p], df_index, tf_weights_ff) * weight
                norm_wash_trade_slope_mtf += get_adaptive_mtf_normalized_score(wash_trade_slopes_raw[p], df_index, ascending=True, tf_weights=tf_weights_ff) * weight
            norm_deception_slope_mtf /= total_slope_weight
            norm_wash_trade_slope_mtf /= total_slope_weight
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        deceptive_divergence_base = (norm_flow_credibility - norm_deception_slope_mtf.abs() * np.sign(norm_deception_slope_mtf))
        wash_trade_modulator = (1 - norm_wash_trade_slope_mtf * deception_mod_sensitivity)
        deceptive_tension_score = (deceptive_divergence_base * wash_trade_modulator).clip(-1, 1)
        if is_probe_active:
            print(f"       - 过程: norm_flow_credibility (DIT): {norm_flow_credibility.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_deception_slope_mtf (DIT): {norm_deception_slope_mtf.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_wash_trade_slope_mtf (DIT): {norm_wash_trade_slope_mtf.loc[probe_date]:.4f}")
            print(f"       - 过程: wash_trade_modulator (DIT): {wash_trade_modulator.loc[probe_date]:.4f}")
            print(f"       - 过程: deceptive_tension_score: {deceptive_tension_score.loc[probe_date]:.4f}")
        # --- 4. 微观意图张力 (Micro-Flow Intent Tension) (V4.0 新增) ---
        norm_order_book_imbalance = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        micro_exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        micro_intent_tension_score = (
            norm_order_book_imbalance * micro_intent_tension_signals_weights.get('order_book_imbalance_D', 0.5) +
            micro_exhaustion_score * (micro_intent_tension_signals_weights.get('buy_quote_exhaustion_rate_D', 0.25) + micro_intent_tension_signals_weights.get('sell_quote_exhaustion_rate_D', 0.25))
        ).clip(-1, 1)
        if is_probe_active:
            print(f"       - 过程: norm_order_book_imbalance (MIT): {norm_order_book_imbalance.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_buy_exhaustion (MIT): {norm_buy_exhaustion.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_sell_exhaustion (MIT): {norm_sell_exhaustion.loc[probe_date]:.4f}")
            print(f"       - 过程: micro_exhaustion_score (MIT): {micro_exhaustion_score.loc[probe_date]:.4f}")
            print(f"       - 过程: micro_intent_tension_score: {micro_intent_tension_score.loc[probe_date]:.4f}")
        # --- 5. 能量注入与持续性 (Energy Injection & Persistence) ---
        norm_mf_activity = get_adaptive_mtf_normalized_score(mf_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_mf_ofi = get_adaptive_mtf_normalized_score(mf_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_micro_impact_elasticity = get_adaptive_mtf_normalized_score(micro_impact_elasticity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        energy_injection = (
            norm_mf_activity * energy_injection_signals_weights.get('main_force_activity_ratio_D', 0.4) +
            norm_mf_ofi * energy_injection_signals_weights.get('main_force_ofi_D', 0.3) +
            norm_micro_impact_elasticity * energy_injection_signals_weights.get('micro_impact_elasticity_D', 0.3)
        ).clip(0, 1)
        combined_divergence_abs = (
            core_divergence_score.abs() * divergence_component_weights.get('core_divergence', 0.3) +
            structural_tension_score.abs() * divergence_component_weights.get('structural_tension', 0.25) +
            deceptive_tension_score.abs() * divergence_component_weights.get('deceptive_tension', 0.25) +
            micro_intent_tension_score.abs() * divergence_component_weights.get('micro_intent_tension', 0.2)
        )
        persistence = combined_divergence_abs.rolling(window=persistence_window, min_periods=max(1, int(persistence_window * 0.2))).std().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence, df_index, ascending=True, tf_weights=tf_weights_ff)
        eip_score = (energy_injection * norm_persistence).pow(0.5).clip(0, 1)
        if is_probe_active:
            print(f"       - 过程: norm_mf_activity (EIP): {norm_mf_activity.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_mf_ofi (EIP): {norm_mf_ofi.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_micro_impact_elasticity (EIP): {norm_micro_impact_elasticity.loc[probe_date]:.4f}")
            print(f"       - 过程: energy_injection (EIP): {energy_injection.loc[probe_date]:.4f}")
            print(f"       - 过程: combined_divergence_abs (EIP): {combined_divergence_abs.loc[probe_date]:.4f}")
            print(f"       - 过程: persistence (EIP): {persistence.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_persistence (EIP): {norm_persistence.loc[probe_date]:.4f}")
            print(f"       - 过程: eip_score: {eip_score.loc[probe_date]:.4f}")
        # --- 6. 非线性融合与情境自适应权重 ---
        norm_adaptive_weight_modulator_1 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_1_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_adaptive_weight_modulator_2 = get_adaptive_mtf_normalized_score(adaptive_weight_modulator_2_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        total_base_weight = sum(divergence_component_weights.values()) # 修正: 确保这里只对数值求和
        if total_base_weight == 0:
            return pd.Series(0.0, index=df.index)
        dynamic_core_divergence_weight = divergence_component_weights.get('core_divergence', 0.3) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility - norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility)
        dynamic_structural_tension_weight = divergence_component_weights.get('structural_tension', 0.25) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility)
        dynamic_deceptive_tension_weight = divergence_component_weights.get('deceptive_tension', 0.25) * (1 - norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility)
        dynamic_micro_intent_tension_weight = divergence_component_weights.get('micro_intent_tension', 0.2) * (1 + norm_adaptive_weight_modulator_1 * adaptive_weight_sensitivity_credibility + norm_adaptive_weight_modulator_2 * adaptive_weight_sensitivity_volatility)
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
        if is_probe_active:
            print(f"       - 过程: norm_adaptive_weight_modulator_1: {norm_adaptive_weight_modulator_1.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_adaptive_weight_modulator_2: {norm_adaptive_weight_modulator_2.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_core_divergence_weight: {dynamic_core_divergence_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_structural_tension_weight: {dynamic_structural_tension_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_deceptive_tension_weight: {dynamic_deceptive_tension_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_micro_intent_tension_weight: {dynamic_micro_intent_tension_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: fused_divergence_base: {fused_divergence_base.loc[probe_date]:.4f}")
            print(f"       - 过程: base_tension_score (before dynamic evolution): {base_tension_score.loc[probe_date]:.4f}")
        # --- 7. 张力演化趋势与预警 (Tension Evolution & Early Warning) ---
        smoothed_base_score = base_tension_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_1_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
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
        if is_probe_active:
            print(f"       - 过程: smoothed_base_score: {smoothed_base_score.loc[probe_date]:.4f}")
            print(f"       - 过程: velocity: {velocity.loc[probe_date]:.4f}")
            print(f"       - 过程: acceleration: {acceleration.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_velocity: {norm_velocity.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_acceleration: {norm_acceleration.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_dynamic_evolution_context_1: {norm_dynamic_evolution_context_1.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_base_score_weight: {dynamic_base_score_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_velocity_weight: {dynamic_velocity_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_acceleration_weight: {dynamic_acceleration_weight.loc[probe_date]:.4f}")
            print(f"       - 结果: final_score: {final_score.loc[probe_date]:.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.0 · 意图推断与情境预测版】资金流公理一：诊断“战场控制权”
        - 核心升级1: 诡道博弈深度情境感知：引入市场情绪作为诡道调制器的情境因子，对欺骗指数和对倒强度的影响进行动态校准，更精准识别主力诡道意图。
        - 核心升级2: 微观盘口意图推断：融合盘口枯竭率与微观结构效率，形成更具洞察力的微观意图分数，捕捉主力在盘口上的真实攻防。
        - 核心升级3: 多维度情境自适应权重：扩展动态权重调制器，引入市场情绪，使宏观资金流向与微观盘口控制力的融合权重在不同市场情境下更具适应性。
        - 核心升级4: 资金流结构与效率非线性建模：在各子分数融合中引入更多非线性函数，捕捉资金流各组件间更复杂的交互作用。
        - 核心升级5: 预测性与前瞻性增强：根据市场情境动态调整战场控制权速度和加速度的融合权重，使其在趋势转折点更具前瞻性。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“战场控制权 (V5.0 · 意图推断与情境预测版)”公理...")
        df_index = df.index
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ac_params = get_param_value(p_conf_ff.get('axiom_consensus_params'), {})
        # V5.0 诡道博弈深度情境感知参数
        deception_mod_enabled = get_param_value(ac_params.get('deception_mod_enabled'), True)
        deception_penalty_sensitivity = get_param_value(ac_params.get('deception_penalty_sensitivity'), 0.6)
        wash_trade_penalty_sensitivity = get_param_value(ac_params.get('wash_trade_penalty_sensitivity'), 0.4)
        conviction_threshold_deception = get_param_value(ac_params.get('conviction_threshold_deception'), 0.2)
        flow_credibility_threshold = get_param_value(ac_params.get('flow_credibility_threshold'), 0.5)
        deception_context_modulator_signal_name = get_param_value(ac_params.get('deception_context_modulator_signal'), 'market_sentiment_score_D')
        deception_context_sensitivity = get_param_value(ac_params.get('deception_context_sensitivity'), 0.3)
        # V5.0 多维度情境自适应权重参数
        dynamic_weight_mod_enabled = get_param_value(ac_params.get('dynamic_weight_mod_enabled'), True)
        macro_flow_base_weight = get_param_value(ac_params.get('macro_flow_base_weight'), 0.4)
        micro_control_base_weight = get_param_value(ac_params.get('micro_control_base_weight'), 0.6)
        dynamic_weight_modulator_signal_1_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_2'), 'SLOPE_5_NMFNF_D')
        dynamic_weight_modulator_signal_3_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_3'), 'market_sentiment_score_D')
        dynamic_weight_sensitivity_volatility = get_param_value(ac_params.get('dynamic_weight_sensitivity_volatility'), 0.3)
        dynamic_weight_sensitivity_flow_slope = get_param_value(ac_params.get('dynamic_weight_sensitivity_flow_slope'), 0.2)
        dynamic_weight_sensitivity_sentiment = get_param_value(ac_params.get('dynamic_weight_sensitivity_sentiment'), 0.1)
        # V5.0 微观盘口意图推断参数
        asymmetric_micro_control_enabled = get_param_value(ac_params.get('asymmetric_micro_control_enabled'), True)
        exhaustion_boost_factor = get_param_value(ac_params.get('exhaustion_boost_factor'), 0.2)
        exhaustion_penalty_factor = get_param_value(ac_params.get('exhaustion_penalty_factor'), 0.3)
        micro_intent_fusion_weights = get_param_value(ac_params.get('micro_intent_fusion_weights'), {'imbalance': 0.4, 'efficiency': 0.3, 'exhaustion': 0.3})
        # V5.0 预测性与前瞻性增强参数
        smoothing_ema_span = get_param_value(ac_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ac_params.get('dynamic_evolution_base_weights'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_name = get_param_value(ac_params.get('dynamic_evolution_context_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_evolution_context_sensitivity = get_param_value(ac_params.get('dynamic_evolution_context_sensitivity'), 0.2)
        # --- 信号依赖校验 ---
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'order_book_imbalance_D', 'microstructure_efficiency_index_D', 'wash_trade_intensity_D',
            'deception_index_D', 'main_force_conviction_index_D', 'flow_credibility_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name,
            dynamic_weight_modulator_signal_3_name, # V5.0 动态权重依赖
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            deception_context_modulator_signal_name, # V5.0 诡道情境依赖
            dynamic_evolution_context_modulator_signal_name # V5.0 动态演化情境依赖
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_consensus"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        main_force_flow_raw = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        retail_flow_raw = self._get_safe_series(df, df, 'retail_net_flow_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_consensus")
        ofi_impact_raw = self._get_safe_series(df, df, 'microstructure_efficiency_index_D', 0.0, method_name="_diagnose_axiom_consensus")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_consensus")
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_conviction_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_consensus")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_consensus")
        volatility_instability_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_consensus")
        flow_slope_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_consensus")
        market_sentiment_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_consensus")
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_consensus")
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_consensus")
        deception_context_modulator_raw = self._get_safe_series(df, df, deception_context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_consensus")
        dynamic_evolution_context_modulator_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_name, 0.0, method_name="_diagnose_axiom_consensus")
        # --- 1. 宏观资金流向 (Macro Fund Flow) ---
        flow_consensus_score = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_raw - retail_flow_raw, df_index, tf_weights_ff)
        # --- 2. 微观盘口意图推断 (Micro Order Book Intent Inference) ---
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff)
        impact_score = get_adaptive_mtf_normalized_bipolar_score(ofi_impact_raw, df_index, tf_weights_ff)
        # V5.0 微观盘口意图推断：融合枯竭率
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 枯竭率越低越好
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 枯竭率越高越好
        # 枯竭率综合得分 (双极性)
        exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        # V5.0 非线性融合微观意图
        micro_intent_score = (
            imbalance_score * micro_intent_fusion_weights.get('imbalance', 0.4) +
            impact_score * micro_intent_fusion_weights.get('efficiency', 0.3) +
            exhaustion_score * micro_intent_fusion_weights.get('exhaustion', 0.3)
        ).clip(-1, 1)
        # V4.0 微观盘口控制力非对称增强 (现在作用于 micro_intent_score)
        micro_control_modulator = pd.Series(1.0, index=df_index)
        if asymmetric_micro_control_enabled:
            # 买盘枯竭低 & 卖盘枯竭高 -> 增强微观控制力 (买方强势)
            boost_mask = (norm_buy_exhaustion > 0.5) & (norm_sell_exhaustion > 0.5)
            micro_control_modulator.loc[boost_mask] = 1 + (norm_buy_exhaustion.loc[boost_mask] * norm_sell_exhaustion.loc[boost_mask]) * exhaustion_boost_factor
            # 买盘枯竭高 & 卖盘枯竭低 -> 惩罚微观控制力 (卖方强势)
            penalty_mask = (norm_buy_exhaustion < 0.5) & (norm_sell_exhaustion < 0.5)
            micro_control_modulator.loc[penalty_mask] = 1 - (norm_buy_exhaustion.loc[penalty_mask] * norm_sell_exhaustion.loc[penalty_mask]) * exhaustion_penalty_factor
            micro_control_modulator = micro_control_modulator.clip(0.5, 1.5)
        micro_control_score = micro_intent_score * micro_control_modulator
        # --- 3. 诡道博弈深度情境感知与调制 (Deceptive Game Integration & Contextual Modulation) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_mod_enabled:
            norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights_ff)
            norm_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(deception_context_modulator_raw, df_index, tf_weights=tf_weights_ff)
            # V5.0 诡道博弈深度情境感知：市场情绪调制惩罚/奖励强度
            sentiment_mod_factor = (1 + norm_market_sentiment.abs() * deception_context_sensitivity * np.sign(norm_market_sentiment))
            # 基础惩罚：对倒强度
            deception_modulator = deception_modulator * (1 - norm_wash_trade * wash_trade_penalty_sensitivity * sentiment_mod_factor.clip(0.5, 1.5))
            # 欺骗指数调制
            # 正向欺骗 (诱多) 惩罚控制权，在市场情绪高涨时惩罚更重
            bull_trap_mask = (norm_deception > 0)
            deception_modulator.loc[bull_trap_mask] = deception_modulator.loc[bull_trap_mask] * (1 - norm_deception.loc[bull_trap_mask] * deception_penalty_sensitivity * sentiment_mod_factor.loc[bull_trap_mask].clip(0.5, 1.5))
            # 负向欺骗 (诱空) 在主力信念强且可信度高时，可能为洗盘，缓解惩罚或增强
            bear_trap_mitigation_mask = (norm_deception < 0) & (norm_conviction > conviction_threshold_deception) & (norm_flow_credibility > flow_credibility_threshold)
            deception_modulator.loc[bear_trap_mitigation_mask] = deception_modulator.loc[bear_trap_mitigation_mask] * (1 + norm_deception.loc[bear_trap_mitigation_mask].abs() * deception_penalty_sensitivity * 0.5 * sentiment_mod_factor.loc[bear_trap_mitigation_mask].clip(0.5, 1.5)) # 缓解一半惩罚
            # 全局可信度校准 (V5.0 增强：可信度低时，诡道调制效果减弱)
            deception_modulator = deception_modulator * (1 + (norm_flow_credibility - 0.5) * 0.5)
            # V5.0 资金流可信度作为信任门槛
            low_credibility_mask = (norm_flow_credibility < flow_credibility_threshold)
            deception_modulator.loc[low_credibility_mask] = deception_modulator.loc[low_credibility_mask] * (norm_flow_credibility.loc[low_credibility_mask] / flow_credibility_threshold).clip(0.1, 1.0) # 可信度低时，进一步惩罚
            deception_modulator = deception_modulator.clip(0.01, 2.0) # 限制调制范围，最低可惩罚至0.01
        # --- 4. 多维度情境自适应权重 (Adaptive Macro-Micro Weighting) ---
        dynamic_macro_weight = pd.Series(macro_flow_base_weight, index=df_index)
        dynamic_micro_weight = pd.Series(micro_control_base_weight, index=df_index)
        if dynamic_weight_mod_enabled:
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_flow_slope = get_adaptive_mtf_normalized_bipolar_score(flow_slope_raw, df_index, tf_weights=tf_weights_ff)
            norm_market_sentiment_dw = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            # 波动性高或资金流趋势不明确时，增加微观权重，降低宏观权重
            # 波动性低或资金流趋势明确时，增加宏观权重，降低微观权重
            # V5.0 市场情绪也影响权重：情绪高涨时，宏观权重可能更重要；情绪低迷时，微观权重更重要
            mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) + \
                         (norm_flow_slope.abs() * dynamic_weight_sensitivity_flow_slope * np.sign(norm_flow_slope)) + \
                         (norm_market_sentiment_dw * dynamic_weight_sensitivity_sentiment)
            dynamic_macro_weight = dynamic_macro_weight * (1 + mod_factor)
            dynamic_micro_weight = dynamic_micro_weight * (1 - mod_factor)
            # 归一化动态权重
            sum_dynamic_weights = dynamic_macro_weight + dynamic_micro_weight
            dynamic_macro_weight = dynamic_macro_weight / sum_dynamic_weights
            dynamic_micro_weight = dynamic_micro_weight / sum_dynamic_weights
            dynamic_macro_weight = dynamic_macro_weight.clip(0.1, 0.9) # 限制权重范围
            dynamic_micro_weight = dynamic_micro_weight.clip(0.1, 0.9)
        # --- 5. 融合基础战场控制权 (V5.0 资金流结构与效率非线性建模) ---
        # 使用 tanh 进一步非线性化融合结果
        base_battlefield_control_score = np.tanh(
            flow_consensus_score * dynamic_macro_weight +
            micro_control_score * dynamic_micro_weight
        )
        # 应用诡道调制器
        base_battlefield_control_score = base_battlefield_control_score * deception_modulator
        # --- 6. 战场控制权动态演化与前瞻性增强 (Dynamic Evolution & Foresight Enhancement) ---
        smoothed_base_score = base_battlefield_control_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        # V5.0 预测性与前瞻性增强：根据情境动态调整速度和加速度权重
        norm_dynamic_evolution_context = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 波动不稳定性越低，动态权重越高
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.2) * (1 + norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        dynamic_base_weight = dynamic_evolution_base_weights.get('base_score', 0.6) * (1 - norm_dynamic_evolution_context * dynamic_evolution_context_sensitivity)
        # 确保权重和为1
        total_dynamic_weights = dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        # 融合基础分、速度和加速度 (使用几何平均，非线性融合)
        final_score = (
            (base_battlefield_control_score.add(1)/2).pow(dynamic_base_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.0 · 多维时空与自适应粒度版】资金流公理二：诊断“信念韧性”
        - 核心升级1: 核心信念强度：融合主力信念和聪明钱净买入的多时间框架斜率，深化资金流纯度，提升信念判断的跨周期稳健性。
        - 核心升级2: 诡道博弈韧性调制：引入欺骗指数和对倒强度的多时间框架斜率，并根据资金流可信度和市场流动性动态调整惩罚因子，同时引入核心信念强度进行反馈调制，更精微地评估诡道影响。
        - 核心升级3: 信念传导效率：融合主力执行Alpha和资金流效率的多时间框架斜率，引入日内VWAP偏离指数衡量执行成本，并结合大单压力与价格冲击形成“投入产出比”，全面评估效率。
        - 核心升级4: 动态情境自适应权重：扩展情境因子（如趋势活力），并根据波动性和流动性动态调整高频信号权重，实现粒度自适应的非线性权重调制。
        - 核心升级5: 信念演化趋势：对最终信念分数进行多周期平滑，并结合趋势活力评估惯性与转折预警。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“信念韧性 (V4.0 · 多维时空与自适应粒度版)”公理...")
        df_index = df.index
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ac_params = get_param_value(p_conf_ff.get('axiom_conviction_params'), {})
        # V4.0 核心信念强度参数
        core_conviction_weights = get_param_value(ac_params.get('core_conviction_weights'), {'main_force_conviction_slope_5': 0.2, 'main_force_conviction_slope_13': 0.2, 'main_force_conviction_slope_21': 0.1, 'smart_money_net_buy_slope_5': 0.15, 'smart_money_net_buy_slope_13': 0.15, 'flow_credibility': 0.1, 'intraday_large_order_flow': 0.1})
        # V4.0 诡道博弈韧性调制参数
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
        # V4.0 信念传导效率参数
        transmission_efficiency_weights = get_param_value(ac_params.get('transmission_efficiency_weights'), {'main_force_execution_alpha_slope_5': 0.2, 'main_force_execution_alpha_slope_13': 0.1, 'flow_efficiency_slope_5': 0.2, 'flow_efficiency_slope_13': 0.1, 'intraday_price_impact': 0.2, 'large_order_pressure': 0.1, 'intraday_vwap_deviation': 0.1})
        # V4.0 动态情境自适应权重参数
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
        # V4.0 信念演化趋势参数
        smoothing_ema_span = get_param_value(ac_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ac_params.get('dynamic_evolution_base_weights'), {'base_score': 0.5, 'velocity': 0.3, 'acceleration': 0.2})
        dynamic_evolution_context_modulator_signal_1_name = get_param_value(ac_params.get('dynamic_evolution_context_modulator_signal'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_evolution_context_sensitivity_1 = get_param_value(ac_params.get('dynamic_evolution_context_sensitivity'), 0.2)
        dynamic_evolution_context_modulator_signal_2_name = get_param_value(ac_params.get('dynamic_evolution_context_modulator_signal_2'), 'trend_vitality_index_D')
        dynamic_evolution_context_sensitivity_2 = get_param_value(ac_params.get('dynamic_evolution_context_sensitivity_2'), 0.1)
        # --- 信号依赖校验 (仅资金类信号) ---
        required_signals = [
            'SLOPE_5_main_force_conviction_index_D', 'SLOPE_13_main_force_conviction_index_D', 'SLOPE_21_main_force_conviction_index_D',
            'SLOPE_5_SMART_MONEY_HM_NET_BUY_D', 'SLOPE_13_SMART_MONEY_HM_NET_BUY_D',
            'flow_credibility_index_D',
            'buy_lg_amount_D', 'buy_elg_amount_D', 'sell_lg_amount_D', 'sell_elg_amount_D', # 用于合成 intraday_large_order_flow
            'peak_exchange_purity_D',
            'SLOPE_5_deception_index_D', 'SLOPE_13_deception_index_D', 'SLOPE_21_deception_index_D',
            'SLOPE_5_wash_trade_intensity_D', 'SLOPE_13_wash_trade_intensity_D', 'SLOPE_21_wash_trade_intensity_D',
            'main_force_cost_advantage_D', resilience_context_modulator_signal_1_name, resilience_context_modulator_signal_3_name,
            'SLOPE_5_main_force_execution_alpha_D', 'SLOPE_13_main_force_execution_alpha_D',
            'SLOPE_5_flow_efficiency_index_D', 'SLOPE_13_flow_efficiency_index_D',
            'micro_price_impact_asymmetry_D', 'large_order_pressure_D', 'intraday_vwap_div_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name, dynamic_weight_modulator_signal_3_name, dynamic_weight_modulator_signal_4_name,
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_conviction"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        # Core Conviction
        mf_conviction_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        mf_conviction_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        mf_conviction_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        sm_net_buy_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_axiom_conviction")
        sm_net_buy_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_axiom_conviction")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        buy_lg_amount_raw = self._get_safe_series(df, df, 'buy_lg_amount_D', 0.0, method_name="_diagnose_axiom_conviction")
        buy_elg_amount_raw = self._get_safe_series(df, df, 'buy_elg_amount_D', 0.0, method_name="_diagnose_axiom_conviction")
        sell_lg_amount_raw = self._get_safe_series(df, df, 'sell_lg_amount_D', 0.0, method_name="_diagnose_axiom_conviction")
        sell_elg_amount_raw = self._get_safe_series(df, df, 'sell_elg_amount_D', 0.0, method_name="_diagnose_axiom_conviction")
        intraday_large_order_flow_synthesized = (buy_lg_amount_raw + buy_elg_amount_raw) - (sell_lg_amount_raw + sell_elg_amount_raw)
        main_force_flow_purity_raw = self._get_safe_series(df, df, 'peak_exchange_purity_D', 0.0, method_name="_diagnose_axiom_conviction")
        # Deceptive Resilience
        deception_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_deception_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        deception_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_deception_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        deception_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_deception_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        wash_trade_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_conviction")
        wash_trade_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_conviction")
        wash_trade_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_conviction")
        main_force_cost_advantage_raw = self._get_safe_series(df, df, resilience_context_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_conviction")
        market_sentiment_raw = self._get_safe_series(df, df, resilience_context_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_conviction")
        market_liquidity_raw = self._get_safe_series(df, df, resilience_context_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_conviction")
        # Transmission Efficiency
        mf_exec_alpha_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_main_force_execution_alpha_D', 0.0, method_name="_diagnose_axiom_conviction")
        mf_exec_alpha_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_main_force_execution_alpha_D', 0.0, method_name="_diagnose_axiom_conviction")
        flow_efficiency_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_flow_efficiency_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        flow_efficiency_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_flow_efficiency_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        intraday_price_impact_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_axiom_conviction")
        large_order_pressure_raw = self._get_safe_series(df, df, 'large_order_pressure_D', 0.0, method_name="_diagnose_axiom_conviction")
        intraday_vwap_deviation_raw = self._get_safe_series(df, df, 'intraday_vwap_div_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        # Dynamic Weighting & Evolution Context
        volatility_instability_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_conviction")
        market_sentiment_dw_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_conviction")
        market_liquidity_dw_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_conviction")
        trend_vitality_dw_raw = self._get_safe_series(df, df, dynamic_weight_modulator_signal_4_name, 0.0, method_name="_diagnose_axiom_conviction")
        dynamic_evolution_context_modulator_1_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_conviction")
        dynamic_evolution_context_modulator_2_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_conviction")
        # --- 1. 核心信念强度 (Core Conviction Strength) ---
        norm_mf_conviction_slope_5 = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slope_5_raw, df_index, tf_weights_ff)
        norm_mf_conviction_slope_13 = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slope_13_raw, df_index, tf_weights_ff)
        norm_mf_conviction_slope_21 = get_adaptive_mtf_normalized_bipolar_score(mf_conviction_slope_21_raw, df_index, tf_weights_ff)
        norm_sm_net_buy_slope_5 = get_adaptive_mtf_normalized_bipolar_score(sm_net_buy_slope_5_raw, df_index, tf_weights_ff)
        norm_sm_net_buy_slope_13 = get_adaptive_mtf_normalized_bipolar_score(sm_net_buy_slope_13_raw, df_index, tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_intraday_large_order_flow = get_adaptive_mtf_normalized_bipolar_score(intraday_large_order_flow_synthesized, df_index, tf_weights_ff)
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
        # --- 2. 诡道博弈韧性调制 (Deceptive Resilience Modulation) ---
        deceptive_resilience_modulator = pd.Series(1.0, index=df_index)
        if deceptive_resilience_mod_enabled:
            # V4.0 多时间框架诡道
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
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights_ff)
            norm_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights_ff)
            norm_market_liquidity = get_adaptive_mtf_normalized_score(market_liquidity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            # 情境因子对惩罚/奖励的敏感度调制
            sentiment_mod = (1 + norm_market_sentiment.abs() * resilience_context_sensitivity_sentiment * np.sign(norm_market_sentiment))
            cost_advantage_mod = (1 + norm_cost_advantage.abs() * resilience_context_sensitivity_cost_advantage * np.sign(norm_cost_advantage))
            liquidity_mod = (1 + (norm_market_liquidity - 0.5) * resilience_context_sensitivity_liquidity)
            # V4.0 信念反馈调制：核心信念强度影响诡道惩罚
            conviction_feedback_mod = (1 - core_conviction_score.abs() * conviction_feedback_sensitivity * np.sign(core_conviction_score)) # 信念越强，惩罚越轻；信念越弱，惩罚越重
            # 基础惩罚：对倒强度，受情绪、流动性和信念反馈影响
            deceptive_resilience_modulator = deceptive_resilience_modulator * (1 - norm_wash_trade_multi_tf * wash_trade_penalty_factor * sentiment_mod.clip(0.5, 1.5) * liquidity_mod.clip(0.5, 1.5) * conviction_feedback_mod.clip(0.5, 1.5))
            # 欺骗指数调制：
            bull_trap_mask = (norm_deception_multi_tf > 0)
            deceptive_resilience_modulator.loc[bull_trap_mask] = deceptive_resilience_modulator.loc[bull_trap_mask] * (1 - norm_deception_multi_tf.loc[bull_trap_mask] * deception_penalty_factor * sentiment_mod.loc[bull_trap_mask].clip(0.5, 1.5) * liquidity_mod.loc[bull_trap_mask].clip(0.5, 1.5) * conviction_feedback_mod.loc[bull_trap_mask].clip(0.5, 1.5))
            bear_trap_resilience_mask = (norm_deception_multi_tf < 0) & (norm_cost_advantage > 0.5) & (norm_market_sentiment < -0.5) & (norm_market_liquidity < 0.5) & (core_conviction_score > 0.2) # 只有核心信念也强时才认为是洗盘
            deceptive_resilience_modulator.loc[bear_trap_resilience_mask] = deceptive_resilience_modulator.loc[bear_trap_resilience_mask] * (1 + norm_deception_multi_tf.loc[bear_trap_resilience_mask].abs() * deception_penalty_factor * cost_advantage_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5) * (1 - liquidity_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5)))
            deceptive_resilience_modulator = deceptive_resilience_modulator.clip(0.01, 2.0)
        # --- 3. 信念传导效率 (Conviction Transmission Efficiency) ---
        norm_mf_exec_alpha_slope_5 = get_adaptive_mtf_normalized_bipolar_score(mf_exec_alpha_slope_5_raw, df_index, tf_weights_ff)
        norm_mf_exec_alpha_slope_13 = get_adaptive_mtf_normalized_bipolar_score(mf_exec_alpha_slope_13_raw, df_index, tf_weights_ff)
        norm_flow_efficiency_slope_5 = get_adaptive_mtf_normalized_bipolar_score(flow_efficiency_slope_5_raw, df_index, tf_weights_ff)
        norm_flow_efficiency_slope_13 = get_adaptive_mtf_normalized_bipolar_score(flow_efficiency_slope_13_raw, df_index, tf_weights_ff)
        norm_intraday_price_impact = get_adaptive_mtf_normalized_bipolar_score(intraday_price_impact_raw, df_index, tf_weights_ff)
        norm_large_order_pressure = get_adaptive_mtf_normalized_score(large_order_pressure_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_intraday_vwap_deviation = get_adaptive_mtf_normalized_score(intraday_vwap_deviation_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # VWAP偏离越小越好
        # V4.0 投入产出比
        efficiency_ratio = (norm_intraday_price_impact + (1 - norm_intraday_vwap_deviation)) / (norm_large_order_pressure + 1e-9) # 产出 / 投入
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
            # 波动性高时，更关注诡道韧性；市场情绪高涨时，更关注核心信念和传导效率；流动性差时，诡道韧性权重增加；趋势强劲时，核心信念和传导效率权重增加
            mod_factor_volatility = (norm_volatility_instability - 0.5) * dynamic_weight_sensitivity_volatility
            mod_factor_sentiment = norm_market_sentiment_dw * dynamic_weight_sensitivity_sentiment
            mod_factor_liquidity = (0.5 - norm_market_liquidity_dw) * dynamic_weight_sensitivity_liquidity
            mod_factor_trend_vitality = (norm_trend_vitality_dw - 0.5) * dynamic_weight_sensitivity_trend_vitality
            # 调整权重
            dynamic_core_conviction_weight = dynamic_core_conviction_weight * (1 + mod_factor_sentiment - mod_factor_volatility + mod_factor_trend_vitality)
            dynamic_deceptive_resilience_weight = dynamic_deceptive_resilience_weight * (1 + mod_factor_volatility + mod_factor_liquidity - mod_factor_sentiment - mod_factor_trend_vitality)
            dynamic_transmission_efficiency_weight = dynamic_transmission_efficiency_weight * (1 + mod_factor_sentiment - mod_factor_liquidity + mod_factor_trend_vitality)
            # 归一化动态权重
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
        # V4.0 预测性与前瞻性增强：根据情境动态调整速度和加速度权重
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_1_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 波动不稳定性越低，动态权重越高
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_2_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 趋势强度越高，动态权重越高
        # 综合情境因子
        combined_evolution_context_mod = (norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
                                          norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2)
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        # 确保权重和为1
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        # 融合基础分、速度和加速度 (使用几何平均，非线性融合)
        final_score = (
            (base_conviction_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V6.0 · 深度情境与结构洞察版】资金流公理三：诊断“资金流纯度与动能”
        - 核心升级1: 基础动能深化：融合资金净流量和聪明钱净买入的多时间框架（含55日）速度与加速度，捕捉宏观动能。
        - 核心升级2: 诡道纯度精修：引入欺骗指数的多周期斜率和加速度，并根据资金流基尼系数和散户FOMO溢价指数动态调整对倒和欺骗惩罚。
        - 核心升级3: 环境感知增强：引入波动不稳定性、趋势活力和价格成交量熵作为环境调制因子。
        - 核心升级4: 结构洞察升级：融合主力资金流方向性、资金流基尼系数和散户资金主导指数，评估资金流质量。
        - 核心升级5: 动态融合优化：引入市场情绪和资金流可信度作为动态演化权重的额外情境因子。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资金流纯度与动能 (V6.0 · 深度情境与结构洞察版)”公理...")
        df_index = df.index
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        fm_params = get_param_value(p_conf_ff.get('axiom_flow_momentum_params'), {})
        # V6.0 基础动能深化参数
        base_momentum_weights = get_param_value(fm_params.get('base_momentum_weights'), {'nmfnf_slope_5': 0.2, 'nmfnf_slope_13': 0.15, 'nmfnf_slope_21': 0.1, 'nmfnf_slope_55': 0.05, 'nmfnf_accel_5': 0.2, 'nmfnf_accel_13': 0.15, 'nmfnf_accel_21': 0.1, 'nmfnf_accel_55': 0.05})
        smart_money_momentum_weights = get_param_value(fm_params.get('smart_money_momentum_weights'), {'sm_net_buy_slope_5': 0.6, 'sm_net_buy_accel_5': 0.4})
        # V6.0 诡道纯度精修参数
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
        # V6.0 环境感知增强参数
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
        # V6.0 结构洞察升级参数
        structural_momentum_weights = get_param_value(fm_params.get('structural_momentum_weights'), {'large_order_flow_slope_5': 0.2, 'large_order_flow_accel_5': 0.15, 'main_force_flow_directionality': 0.2, 'main_force_flow_gini': 0.15, 'retail_flow_slope_5': -0.1, 'retail_flow_accel_5': -0.05, 'retail_flow_dominance': -0.15})
        flow_quality_signal_name = get_param_value(fm_params.get('flow_quality_signal'), 'main_force_flow_gini_D')
        retail_dominance_signal_name = get_param_value(fm_params.get('retail_dominance_signal'), 'retail_flow_dominance_index_D')
        # V6.0 动态融合优化参数
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
        # --- 信号依赖校验 ---
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
            dynamic_evolution_context_modulator_signal_3_name, dynamic_evolution_context_modulator_signal_4_name
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_momentum"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        # 基础动能深化
        nmfnf_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_slope_55_raw = self._get_safe_series(df, df, 'SLOPE_55_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_accel_5_raw = self._get_safe_series(df, df, 'ACCEL_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_accel_13_raw = self._get_safe_series(df, df, 'ACCEL_13_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_accel_21_raw = self._get_safe_series(df, df, 'ACCEL_21_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        nmfnf_accel_55_raw = self._get_safe_series(df, df, 'ACCEL_55_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        sm_net_buy_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        sm_net_buy_accel_5_raw = self._get_safe_series(df, df, 'ACCEL_5_SMART_MONEY_HM_NET_BUY_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        # 诡道纯度精修
        wash_trade_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        wash_trade_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        wash_trade_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        deception_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_deception_index_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        deception_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_deception_index_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        deception_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_deception_index_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        main_force_conviction_raw = self._get_safe_series(df, df, purity_context_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        flow_credibility_raw = self._get_safe_series(df, df, purity_context_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        main_force_flow_gini_raw = self._get_safe_series(df, df, purity_context_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        retail_fomo_premium_raw = self._get_safe_series(df, df, purity_context_modulator_signal_4_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        purity_auxiliary_raw = self._get_safe_series(df, df, purity_auxiliary_signal_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        # 环境感知增强
        liquidity_supply_raw = self._get_safe_series(df, df, 'order_book_liquidity_supply_D', 1.0, method_name="_diagnose_axiom_flow_momentum")
        liquidity_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_order_book_liquidity_supply_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        liquidity_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_order_book_liquidity_supply_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        liquidity_impact_raw = self._get_safe_series(df, df, liquidity_impact_signal_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        volatility_instability_raw = self._get_safe_series(df, df, environment_context_signal_1_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        trend_vitality_raw = self._get_safe_series(df, df, environment_context_signal_2_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        price_volume_entropy_raw = self._get_safe_series(df, df, environment_context_signal_3_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        # 结构洞察升级
        lg_flow_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        lg_flow_accel_5_raw = self._get_safe_series(df, df, 'ACCEL_5_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        xl_flow_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_net_xl_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        xl_flow_accel_5_raw = self._get_safe_series(df, df, 'ACCEL_5_net_xl_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        retail_flow_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_retail_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        retail_flow_accel_5_raw = self._get_safe_series(df, df, 'ACCEL_5_retail_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        main_force_flow_directionality_raw = self._get_safe_series(df, df, 'main_force_flow_directionality_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        flow_quality_raw = self._get_safe_series(df, df, flow_quality_signal_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        retail_dominance_raw = self._get_safe_series(df, df, retail_dominance_signal_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        # 动态融合优化
        dynamic_evolution_context_modulator_1_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        dynamic_evolution_context_modulator_2_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        dynamic_evolution_context_modulator_3_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
        dynamic_evolution_context_modulator_4_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_signal_4_name, 0.0, method_name="_diagnose_axiom_flow_momentum")
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
            norm_deception_slope_5 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_5_raw, df_index, tf_weights_ff)
            norm_deception_slope_13 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_13_raw, df_index, tf_weights_ff)
            norm_deception_slope_21 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_21_raw, df_index, tf_weights_ff)
            norm_deception_multi_tf = (
                norm_deception_slope_5 * deception_slope_weights.get('slope_5', 0.5) +
                norm_deception_slope_13 * deception_slope_weights.get('slope_13', 0.3) +
                norm_deception_slope_21 * deception_slope_weights.get('slope_21', 0.2)
            ).clip(-1, 1)
            norm_main_force_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_main_force_flow_gini = get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 基尼系数越低越好
            norm_retail_fomo_premium = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_purity_auxiliary = get_adaptive_mtf_normalized_score(purity_auxiliary_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            # 情境调制因子
            conviction_mod = (1 + norm_main_force_conviction.abs() * purity_context_sensitivity_conviction * np.sign(norm_main_force_conviction))
            credibility_mod = (1 + (norm_flow_credibility - 0.5) * purity_context_sensitivity_credibility)
            gini_mod = (1 + (norm_main_force_flow_gini - 0.5) * purity_context_sensitivity_gini) # 基尼系数低（好）时，调制器 > 1
            fomo_mod = (1 - norm_retail_fomo_premium * purity_context_sensitivity_fomo) # FOMO高（坏）时，调制器 < 1
            # 基础惩罚：对倒强度
            purity_modulator = purity_modulator * (1 - norm_wash_trade_multi_tf * purity_penalty_factor_wash_trade * conviction_mod.clip(0.5, 1.5) * credibility_mod.clip(0.5, 1.5) * gini_mod.clip(0.5, 1.5) * fomo_mod.clip(0.5, 1.5))
            # 欺骗指数调制
            bull_trap_mask = (norm_deception_multi_tf > 0)
            purity_modulator.loc[bull_trap_mask] = purity_modulator.loc[bull_trap_mask] * (1 - norm_deception_multi_tf.loc[bull_trap_mask] * purity_penalty_factor_deception * conviction_mod.loc[bull_trap_mask].clip(0.5, 1.5) * credibility_mod.loc[bull_trap_mask].clip(0.5, 1.5) * gini_mod.loc[bull_trap_mask].clip(0.5, 1.5) * fomo_mod.loc[bull_trap_mask].clip(0.5, 1.5))
            # 良性对倒/欺骗缓解：当主力信念强、可信度高、资金流集中且辅助信号（如T0效率）也高时，缓解惩罚
            benign_mask = (norm_wash_trade_multi_tf < 0.5) & (norm_deception_multi_tf < 0.5) & (norm_main_force_conviction > 0.5) & (norm_flow_credibility > 0.5) & (norm_main_force_flow_gini < 0.5) & (norm_purity_auxiliary > 0.5)
            purity_modulator.loc[benign_mask] = purity_modulator.loc[benign_mask] * (1 + purity_mitigation_factor * norm_purity_auxiliary.loc[benign_mask].clip(0.5, 1.5))
            purity_modulator = purity_modulator.clip(0.01, 2.0)
        # --- 3. 环境感知增强 (Enhanced Contextual Modulator) ---
        context_modulator = pd.Series(1.0, index=df_index)
        if contextual_modulator_enabled:
            norm_liquidity_supply = get_adaptive_mtf_normalized_score(liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_liquidity_slope_5 = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_5_raw, df_index, tf_weights_ff)
            norm_liquidity_slope_13 = get_adaptive_mtf_normalized_bipolar_score(liquidity_slope_13_raw, df_index, tf_weights_ff)
            norm_liquidity_slope_multi_tf = (
                norm_liquidity_slope_5 * liquidity_slope_weights.get('slope_5', 0.6) +
                norm_liquidity_slope_13 * liquidity_slope_weights.get('slope_13', 0.4)
            ).clip(-1, 1)
            norm_liquidity_impact = get_adaptive_mtf_normalized_bipolar_score(liquidity_impact_raw, df_index, tf_weights_ff)
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_price_volume_entropy = get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 熵越低越好
            # 流动性水平调制
            level_mod = (1 + (norm_liquidity_supply - 0.5) * liquidity_mod_sensitivity_level)
            # 流动性斜率调制
            slope_mod = (1 + norm_liquidity_slope_multi_tf * liquidity_mod_sensitivity_slope)
            # 冲击弹性调制
            impact_mod = (1 + norm_liquidity_impact) # 冲击弹性越高，动能放大效果越好
            # 波动不稳定性调制 (高波动不利于动能持续)
            volatility_mod = (1 - norm_volatility_instability * environment_mod_sensitivity_volatility)
            # 趋势活力调制 (高活力有利于动能持续)
            trend_mod = (1 + norm_trend_vitality * environment_mod_sensitivity_trend_vitality)
            # 价格成交量熵调制 (低熵有利于动能持续)
            entropy_mod = (1 + norm_price_volume_entropy * environment_mod_sensitivity_entropy)
            context_modulator = level_mod * slope_mod * impact_mod * volatility_mod * trend_mod * entropy_mod
            context_modulator = context_modulator.clip(0.5, 2.0)
        # --- 4. 结构洞察升级 (Upgraded Structural Momentum) ---
        norm_lg_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(lg_flow_slope_5_raw, df_index, tf_weights_ff)
        norm_lg_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(lg_flow_accel_5_raw, df_index, tf_weights_ff)
        norm_xl_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(xl_flow_slope_5_raw, df_index, tf_weights_ff)
        norm_xl_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(xl_flow_accel_5_raw, df_index, tf_weights_ff)
        norm_retail_flow_slope_5 = get_adaptive_mtf_normalized_bipolar_score(retail_flow_slope_5_raw, df_index, tf_weights_ff)
        norm_retail_flow_accel_5 = get_adaptive_mtf_normalized_bipolar_score(retail_flow_accel_5_raw, df_index, tf_weights_ff)
        norm_main_force_flow_directionality = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_directionality_raw, df_index, tf_weights_ff)
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
            large_order_flow_momentum * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) + # 使用大单权重作为基础
            norm_main_force_flow_directionality * structural_momentum_weights.get('main_force_flow_directionality', 0.2) +
            norm_flow_quality * structural_momentum_weights.get('main_force_flow_gini', 0.15) +
            retail_flow_momentum * structural_momentum_weights.get('retail_flow_slope_5', -0.1) + # 散户动能
            norm_retail_dominance * structural_momentum_weights.get('retail_flow_dominance', -0.15) # 散户主导惩罚
        ).clip(-1, 1)
        # --- 5. 融合基础动能、纯度、环境和结构性动能 ---
        # 几何平均融合，突出共振效应
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
        # V6.0 预测性与前瞻性增强：根据情境动态调整速度和加速度权重
        norm_dynamic_evolution_context_1 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_1_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 波动不稳定性越低，动态权重越高
        norm_dynamic_evolution_context_2 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_2_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 趋势强度越高，动态权重越高
        norm_dynamic_evolution_context_3 = get_adaptive_mtf_normalized_bipolar_score(dynamic_evolution_context_modulator_3_raw, df_index, tf_weights=tf_weights_ff) # 市场情绪
        norm_dynamic_evolution_context_4 = get_adaptive_mtf_normalized_score(dynamic_evolution_context_modulator_4_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 资金流可信度
        # 综合情境因子
        combined_evolution_context_mod = (
            norm_dynamic_evolution_context_1 * dynamic_evolution_context_sensitivity_1 +
            norm_dynamic_evolution_context_2 * dynamic_evolution_context_sensitivity_2 +
            norm_dynamic_evolution_context_3.abs() * dynamic_evolution_context_sensitivity_3 + # 情绪波动越大，越需要关注动态
            norm_dynamic_evolution_context_4 * dynamic_evolution_context_sensitivity_4
        )
        dynamic_velocity_weight = dynamic_evolution_base_weights.get('velocity', 0.3) * (1 + combined_evolution_context_mod)
        dynamic_acceleration_weight = dynamic_evolution_base_weights.get('acceleration', 0.2) * (1 + combined_evolution_context_mod)
        dynamic_base_score_weight = dynamic_evolution_base_weights.get('base_score', 0.5) * (1 - combined_evolution_context_mod)
        # 确保权重和为1
        total_dynamic_weights = dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight
        dynamic_base_score_weight /= total_dynamic_weights
        dynamic_velocity_weight /= total_dynamic_weights
        dynamic_acceleration_weight /= total_dynamic_weights
        # 融合基础分、速度和加速度 (使用几何平均，非线性融合)
        final_score = (
            (base_flow_momentum_score.add(1)/2).pow(dynamic_base_score_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_score_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_capital_signature(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 逻辑精炼版】资金流公理五：诊断“资本属性”
        - 核心逻辑: 通过分析资金流的行为模式，区分趋势是由“耐心资本”（机构）还是“敏捷资本”（游资）主导。
        - A股特性: 机构建仓如“温水煮青蛙”，游资点火如“烈火烹油”，两者后续走势预期截然不同。
        - V1.1 优化: 修正了“耐心资本”画像中对资金平稳度的计算，从衡量“波动率的变化”升级为直接衡量“波动率的大小”，更精准地刻画其稳定性。
        """
        print("    -> [资金流层] 正在诊断“资本属性”公理...")
        required_signals = [
            'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D', 'main_force_ofi_D',
            'trade_count_D', 'THEME_HOTNESS_SCORE_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_capital_signature"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. “耐心资本”（机构）画像
        institutional_flow = self._get_safe_series(df, df, 'net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_capital_signature") + self._get_safe_series(df, df, 'net_xl_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # 证据一：持续稳定的净流入
        flow_consistency = institutional_flow.rolling(window=21).mean()
        # 证据二：流入过程的波动率低 (V1.1 修正)
        flow_volatility = institutional_flow.rolling(window=21).std().fillna(0) # 新增: 直接计算资金流的波动率
        flow_steadiness = 1 - get_adaptive_mtf_normalized_score(flow_volatility, df_index, ascending=True, tf_weights=tf_weights_ff) # 修改: 对波动率本身进行归一化，低波动=高平稳度
        patient_capital_score = (
            get_adaptive_mtf_normalized_score(flow_consistency, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.7 +
            flow_steadiness * 0.3 # 修改: 使用新的平稳度得分
        ).clip(0, 1)
        # 2. “敏捷资本”（游资）画像
        # 证据一：高频盘口冲击力
        ofi = self._get_safe_series(df, df, 'main_force_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # 证据二：成交笔数爆发
        trade_count = self._get_safe_series(df, df, 'trade_count_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        # 证据三：与题材热度高度相关
        theme_hotness = self._get_safe_series(df, df, 'THEME_HOTNESS_SCORE_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        agile_capital_score = (
            get_adaptive_mtf_normalized_score(ofi.abs(), df_index, ascending=True, tf_weights=tf_weights_ff) * 0.4 +
            get_adaptive_mtf_normalized_score(trade_count, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.3 +
            get_adaptive_mtf_normalized_score(theme_hotness, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.3
        ).clip(0, 1)
        # 3. 融合
        capital_signature_score = patient_capital_score - agile_capital_score
        return capital_signature_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_flow_structure_health(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】资金流公理六：诊断“资金流结构健康度”
        - 核心逻辑: 融合流量的平稳度、效率、成本凝聚力与结构风险，评估资金流模式的可持续性。
        - A股特性: 旨在区分“一日游”式的脉冲行情与具备坚实基础的、可持续的趋势。
        """
        print("    -> [资金流层] 正在诊断“资金流结构健康度”公理...")
        required_signals = [
            'main_force_net_flow_calibrated_D', 'ATR_14_D',
            'main_force_vpoc_D', 'close_D', 'structural_leverage_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_structure_health"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. 流量平稳度 (Flow Steadiness)
        net_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        flow_volatility = net_flow.rolling(window=21).std().fillna(0)
        norm_flow_steadiness = 1 - get_adaptive_mtf_normalized_score(flow_volatility, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 2. 流量效率 (Flow Efficiency)
        price_volatility = self._get_safe_series(df, df, 'ATR_14_D', 1.0, method_name="_diagnose_axiom_flow_structure_health").replace(0, 1e-9)
        flow_efficiency_raw = net_flow.rolling(window=21).mean() / price_volatility.rolling(window=21).mean()
        norm_flow_efficiency = get_adaptive_mtf_normalized_bipolar_score(flow_efficiency_raw, df_index, tf_weights_ff)
        # 3. 成本凝聚力 (Cost Cohesion)
        vpoc = self._get_safe_series(df, df, 'main_force_vpoc_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        close = self._get_safe_series(df, df, 'close_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        cost_divergence = ((close - vpoc) / close).abs().fillna(0)
        norm_cost_cohesion = 1 - get_adaptive_mtf_normalized_score(cost_divergence, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 4. 结构风险过滤器 (Structural Risk Filter)
        structural_leverage = self._get_safe_series(df, df, 'structural_leverage_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        risk_filter = 1 - get_adaptive_mtf_normalized_score(structural_leverage, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 5. 融合
        health_core = (norm_flow_steadiness * 0.4 + norm_cost_cohesion * 0.6)
        # 使用 np.sign(norm_flow_efficiency) 确保当资金为净流出时，健康度指标也呈负向贡献
        flow_structure_health_score = (norm_flow_efficiency * 0.5 + health_core * np.sign(norm_flow_efficiency) * 0.5) * risk_filter
        return flow_structure_health_score.clip(-1, 1).astype(np.float32)

