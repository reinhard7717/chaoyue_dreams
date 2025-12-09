import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
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

    def _get_mtf_dynamic_score(self, df: pd.DataFrame, signal_base_name: str, periods_list: list, weights_dict: dict, is_bipolar: bool, is_accel: bool = False) -> pd.Series:
        mtf_scores = []
        numeric_weights = {k: v for k, v in weights_dict.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=df.index)
        for period_str, weight in numeric_weights.items():
            period = int(period_str)
            prefix = 'ACCEL' if is_accel else 'SLOPE'
            col_name = f'{prefix}_{period}_{signal_base_name}'
            raw_data = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_divergence")
            if is_bipolar:
                norm_score = get_adaptive_mtf_normalized_bipolar_score(raw_data, df.index, self.tf_weights_ff)
            else:
                norm_score = get_adaptive_mtf_normalized_score(raw_data, df.index, ascending=True, tf_weights=self.tf_weights_ff)
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
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
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
        # --- 4. 新增：资金流看涨/看跌背离信号 ---
        # 将所有原子公理存储到 self.strategy.atomic_states，以便 _diagnose_fund_flow_divergence_signals 可以获取
        self.strategy.atomic_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        self.strategy.atomic_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        self.strategy.atomic_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        self.strategy.atomic_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        self.strategy.atomic_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        self.strategy.atomic_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score
        bullish_divergence, bearish_divergence = self._diagnose_fund_flow_divergence_signals(df, norm_window, axiom_divergence)
        # --- 5. 状态赋值 ---
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        all_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        all_states['SCORE_FF_AXIOM_FLOW_STRUCTURE_HEALTH'] = axiom_flow_structure_health
        all_states['SCORE_FF_STRATEGIC_POSTURE'] = strategic_posture_score.astype(np.float32)
        all_states['SCORE_FF_HARMONY_INFLECTION'] = harmony_inflection_score.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        print(f"【V29.0 · 拐点洞察版】分析完成，生成 {len(all_states)} 个资金流原子及融合信号。")
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V5.0 · 动态共振与微观脉冲版】资金流公理四：诊断“资金流内部分歧与意图张力”
        - 核心升级1: 多时间框架斜率与加速度深度融合：对核心分歧、结构性张力、诡道意图张力等维度，计算其在多个时间框架（5, 13, 21, 34, 55日）下的斜率和加速度，并进行加权融合，形成各维度的“动态张力”分数。
        - 核心升级2: 微观资金流动态脉冲分析：对订单簿不平衡、买卖盘枯竭率等微观资金流信号，计算其多时间框架的斜率和加速度，形成“微观动态脉冲”分数。
        - 核心升级3: 情境自适应能量注入与持续性：引入市场波动性、资金流可信度等情境因子，动态调制能量注入和持续性分数的权重。
        - 核心升级4: 非线性融合与动态权重优化：引入更多情境因子动态调整各张力维度权重，并探索更复杂的非线性融合函数。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资金流内部分歧与意图张力 (V5.0 · 动态共振与微观脉冲版)”公理...")
        df_index = df.index
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        self.tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
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
        # --- 信号依赖校验 ---
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
            if isinstance(mod_params, dict) and 'signal' in mod_params: # 修正: 确保mod_params是字典且包含'signal'键
                required_signals.append(mod_params['signal'])
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        # 核心分歧向量
        nmfnf_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_NMFNF_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        nmfnf_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_NMFNF_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        mf_conviction_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        mf_conviction_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        # 结构性张力
        lg_flow_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        lg_flow_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        retail_flow_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_retail_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        retail_flow_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_retail_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        retail_fomo_premium_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        retail_panic_surrender_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        # 诡道意图张力
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_divergence")
        deception_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_deception_index_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        deception_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_deception_index_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_divergence")
        wash_trade_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        wash_trade_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        # 微观意图张力
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_divergence")
        order_book_imbalance_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        order_book_imbalance_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence")
        buy_exhaustion_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        buy_exhaustion_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence")
        sell_exhaustion_slopes_raw = {p: self._get_safe_series(df, df, f'SLOPE_{p}_sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_slope_periods}
        sell_exhaustion_accels_raw = {p: self._get_safe_series(df, df, f'ACCEL_{p}_sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_divergence") for p in divergence_accel_periods}
        # 能量注入
        mf_activity_ratio_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', 0.0, method_name="_diagnose_axiom_divergence")
        mf_ofi_raw = self._get_safe_series(df, df, 'main_force_ofi_D', 0.0, method_name="_diagnose_axiom_divergence")
        micro_impact_elasticity_raw = self._get_safe_series(df, df, 'micro_impact_elasticity_D', 0.0, method_name="_diagnose_axiom_divergence")
        # 能量注入情境调制器
        energy_modulator_signals = {}
        for mod_name, mod_params in energy_injection_context_modulators.items():
            if isinstance(mod_params, dict) and 'signal' in mod_params: # 修正: 确保mod_params是字典且包含'signal'键
                energy_modulator_signals[mod_name] = self._get_safe_series(df, df, mod_params['signal'], 0.0, method_name="_diagnose_axiom_divergence")
        # 自适应权重调制器
        adaptive_weight_modulator_1_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_1_name, 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_2_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_2_name, 0.0, method_name="_diagnose_axiom_divergence")
        adaptive_weight_modulator_3_raw = self._get_safe_series(df, df, adaptive_weight_modulator_signal_3_name, 0.0, method_name="_diagnose_axiom_divergence")
        # 动态演化情境因子
        dynamic_evolution_context_modulator_1_raw = self._get_safe_series(df, df, dynamic_evolution_context_modulator_1_name, 0.0, method_name="_diagnose_axiom_divergence")
        # --- 1. 核心分歧向量 (Core Divergence Vector) ---
        norm_nmfnf_slope_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_slope_periods, divergence_slope_weights, True, False)
        norm_nmfnf_accel_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_accel_periods, divergence_accel_weights, True, True)
        norm_mf_conviction_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_slope_periods, divergence_slope_weights, True, False)
        norm_mf_conviction_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_accel_periods, divergence_accel_weights, True, True)
        nmfnf_dynamic_score = (norm_nmfnf_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_nmfnf_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        mf_conviction_dynamic_score = (norm_mf_conviction_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_mf_conviction_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        core_divergence_score = (nmfnf_dynamic_score - mf_conviction_dynamic_score).clip(-1, 1)
        # --- 2. 结构性张力 (Structural Tension) ---
        norm_lg_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False)
        norm_lg_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True)
        norm_retail_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False)
        norm_retail_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True)
        lg_flow_dynamic_score = (norm_lg_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_lg_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        retail_flow_dynamic_score = (norm_retail_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_retail_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        structural_divergence_base = (lg_flow_dynamic_score - retail_flow_dynamic_score)
        retail_modulator = (1 - norm_retail_fomo * retail_sentiment_mod_sensitivity) + (norm_retail_panic * retail_sentiment_mod_sensitivity)
        structural_tension_score = (structural_divergence_base * retail_modulator).clip(-1, 1)
        # --- 3. 诡道意图张力 (Deceptive Intent Tension) ---
        norm_deception_slope_mtf = self._get_mtf_dynamic_score(df, 'deception_index_D', divergence_slope_periods, divergence_slope_weights, True, False)
        norm_deception_accel_mtf = self._get_mtf_dynamic_score(df, 'deception_index_D', divergence_accel_periods, divergence_accel_weights, True, True)
        norm_wash_trade_slope_mtf = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', divergence_slope_periods, divergence_slope_weights, False, False)
        norm_wash_trade_accel_mtf = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', divergence_accel_periods, divergence_accel_weights, False, True)
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
        norm_obi_slope_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_slope_periods, divergence_slope_weights, True, False)
        norm_obi_accel_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_accel_periods, divergence_accel_weights, True, True)
        obi_dynamic_pulse = (norm_obi_slope_mtf * obi_dynamic_params.get('slope', 0.6) + norm_obi_accel_mtf * obi_dynamic_params.get('accel', 0.4)).clip(-1, 1)
        buy_exh_dynamic_params = micro_intent_dynamic_signals.get('buy_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        norm_buy_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False)
        norm_buy_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True)
        buy_exh_dynamic_pulse = (norm_buy_exh_slope_mtf * buy_exh_dynamic_params.get('slope', 0.5) + norm_buy_exh_accel_mtf * buy_exh_dynamic_params.get('accel', 0.5)).clip(0, 1)
        sell_exh_dynamic_params = micro_intent_dynamic_signals.get('sell_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        norm_sell_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False)
        norm_sell_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True)
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
            if isinstance(mod_params, dict) and 'signal' in mod_params: # 修正: 确保mod_params是字典且包含'signal'键
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
        【V2.0 · 动态意图与结构韧性版】资金流公理五：诊断“资本属性”
        - 核心升级：1) 耐心资本画像深化：综合多时间框架净流入持久性、成本控制与效率、隐蔽吸筹迹象及资金流纯度，评估机构的结构韧性与成本优势。
                    2) 敏捷资本画像锐化：聚焦高频冲击力与方向性、成交活跃度与集中度、短期动能爆发及资金流驱动的题材热度，精准捕捉游资的脉冲冲击与短期博弈。
                    3) 动态情境调制：根据市场流动性、波动性、情绪和趋势活力等情境因子，动态调整两种资本的权重和敏感度。
                    4) 融合与非线性建模：采用更复杂的非线性函数融合，评估两种资本的相对强度和主导性。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资本属性 (V2.0 · 动态意图与结构韧性版)”公理...")
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
                print(f"    -> [资本属性探针] @ {probe_date.date()}:")
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        acs_params = get_param_value(p_conf_ff.get('axiom_capital_signature_params'), {})
        patient_capital_weights = get_param_value(acs_params.get('patient_capital_weights'), {"mtf_flow_persistence": 0.3, "cost_efficiency": 0.2, "covert_accumulation": 0.2, "flow_purity": 0.15, "structural_resilience": 0.15})
        agile_capital_weights = get_param_value(acs_params.get('agile_capital_weights'), {"ofi_impact": 0.3, "activity_concentration": 0.25, "short_term_momentum": 0.25, "theme_chasing": 0.2})
        mtf_periods_patient_flow = get_param_value(acs_params.get('mtf_periods_patient_flow'), {"short": [5, 13], "long": [21, 55]})
        mtf_periods_agile_ofi = get_param_value(acs_params.get('mtf_periods_agile_ofi'), {"short": [5, 13], "long": [21]})
        capital_context_modulator_sensitivity = get_param_value(acs_params.get('capital_context_modulator_sensitivity'), {"liquidity": 0.2, "volatility": 0.3, "sentiment": 0.1, "trend_vitality": 0.2})
        fusion_exponent = get_param_value(acs_params.get('fusion_exponent'), 1.0)
        dynamic_fusion_weights = get_param_value(acs_params.get('dynamic_fusion_weights'), {"patient_base": 0.5, "agile_base": 0.5, "trend_vitality_mod": 0.2, "volatility_mod": 0.1})
        # --- 信号依赖校验 ---
        required_signals = [
            'net_lg_amount_calibrated_D', 'net_xl_amount_calibrated_D',
            'SLOPE_5_net_lg_amount_calibrated_D', 'ACCEL_5_net_lg_amount_calibrated_D',
            'SLOPE_13_net_lg_amount_calibrated_D', 'ACCEL_13_net_lg_amount_calibrated_D',
            'SLOPE_21_net_lg_amount_calibrated_D', 'ACCEL_21_net_lg_amount_calibrated_D',
            'SLOPE_55_net_lg_amount_calibrated_D', 'ACCEL_55_net_lg_amount_calibrated_D',
            'main_force_cost_advantage_D', 'main_force_vwap_guidance_D', 'main_force_execution_alpha_D',
            'covert_accumulation_signal_D', 'suppressive_accumulation_intensity_D', 'flow_credibility_index_D',
            'chip_health_score_D', 'control_solidity_index_D',
            'main_force_ofi_D',
            'SLOPE_5_main_force_ofi_D', 'ACCEL_5_main_force_ofi_D',
            'SLOPE_13_main_force_ofi_D', 'ACCEL_13_main_force_ofi_D',
            'SLOPE_21_main_force_ofi_D', 'ACCEL_21_main_force_ofi_D',
            'micro_price_impact_asymmetry_D', 'trade_count_D', 'main_force_activity_ratio_D',
            'main_force_flow_gini_D',
            'SLOPE_5_NMFNF_D', 'ACCEL_5_NMFNF_D',
            'THEME_HOTNESS_SCORE_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'order_book_liquidity_supply_D', 'order_book_clearing_rate_D',
            'market_sentiment_score_D', 'trend_vitality_index_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_capital_signature"):
            return pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        # 耐心资本相关
        net_lg_amount_raw = self._get_safe_series(df, df, 'net_lg_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: net_lg_amount_calibrated_D (raw): {net_lg_amount_raw.loc[probe_date]:.4f}")
        net_xl_amount_raw = self._get_safe_series(df, df, 'net_xl_amount_calibrated_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: net_xl_amount_calibrated_D (raw): {net_xl_amount_raw.loc[probe_date]:.4f}")
        main_force_cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: main_force_cost_advantage_D (raw): {main_force_cost_advantage_raw.loc[probe_date]:.4f}")
        main_force_vwap_guidance_raw = self._get_safe_series(df, df, 'main_force_vwap_guidance_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: main_force_vwap_guidance_D (raw): {main_force_vwap_guidance_raw.loc[probe_date]:.4f}")
        main_force_execution_alpha_raw = self._get_safe_series(df, df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: main_force_execution_alpha_D (raw): {main_force_execution_alpha_raw.loc[probe_date]:.4f}")
        covert_accumulation_signal_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: covert_accumulation_signal_D (raw): {covert_accumulation_signal_raw.loc[probe_date]:.4f}")
        suppressive_accumulation_intensity_raw = self._get_safe_series(df, df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: suppressive_accumulation_intensity_D (raw): {suppressive_accumulation_intensity_raw.loc[probe_date]:.4f}")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: flow_credibility_index_D (raw): {flow_credibility_raw.loc[probe_date]:.4f}")
        chip_health_score_raw = self._get_safe_series(df, df, 'chip_health_score_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: chip_health_score_D (raw): {chip_health_score_raw.loc[probe_date]:.4f}")
        control_solidity_index_raw = self._get_safe_series(df, df, 'control_solidity_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: control_solidity_index_D (raw): {control_solidity_index_raw.loc[probe_date]:.4f}")
        # 敏捷资本相关
        main_force_ofi_raw = self._get_safe_series(df, df, 'main_force_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: main_force_ofi_D (raw): {main_force_ofi_raw.loc[probe_date]:.4f}")
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: micro_price_impact_asymmetry_D (raw): {micro_price_impact_asymmetry_raw.loc[probe_date]:.4f}")
        trade_count_raw = self._get_safe_series(df, df, 'trade_count_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: trade_count_D (raw): {trade_count_raw.loc[probe_date]:.4f}")
        main_force_activity_ratio_raw = self._get_safe_series(df, df, 'main_force_activity_ratio_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: main_force_activity_ratio_D (raw): {main_force_activity_ratio_raw.loc[probe_date]:.4f}")
        main_force_flow_gini_raw = self._get_safe_series(df, df, 'main_force_flow_gini_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: main_force_flow_gini_D (raw): {main_force_flow_gini_raw.loc[probe_date]:.4f}")
        nmfnf_raw = self._get_safe_series(df, df, 'NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: NMFNF_D (raw): {nmfnf_raw.loc[probe_date]:.4f}")
        theme_hotness_raw = self._get_safe_series(df, df, 'THEME_HOTNESS_SCORE_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: THEME_HOTNESS_SCORE_D (raw): {theme_hotness_raw.loc[probe_date]:.4f}")
        # 情境调制相关
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: VOLATILITY_INSTABILITY_INDEX_21d_D (raw): {volatility_instability_raw.loc[probe_date]:.4f}")
        order_book_liquidity_supply_raw = self._get_safe_series(df, df, 'order_book_liquidity_supply_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: order_book_liquidity_supply_D (raw): {order_book_liquidity_supply_raw.loc[probe_date]:.4f}")
        order_book_clearing_rate_raw = self._get_safe_series(df, df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: order_book_clearing_rate_D (raw): {order_book_clearing_rate_raw.loc[probe_date]:.4f}")
        market_sentiment_raw = self._get_safe_series(df, df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: market_sentiment_score_D (raw): {market_sentiment_raw.loc[probe_date]:.4f}")
        trend_vitality_raw = self._get_safe_series(df, df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        if is_probe_active: print(f"       - 原料: trend_vitality_index_D (raw): {trend_vitality_raw.loc[probe_date]:.4f}")
        # --- 1. 耐心资本 (Patient Capital) - 结构韧性与成本优势 ---
        # 1.1 多时间框架净流入持久性
        institutional_flow = net_lg_amount_raw + net_xl_amount_raw
        norm_institutional_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('short', []), {}, True, False) # 使用lg作为代表
        norm_institutional_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('short', []), {}, True, True)
        norm_institutional_flow_long_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('long', []), {}, True, False)
        flow_persistence = (
            norm_institutional_flow_slope_mtf * 0.4 +
            norm_institutional_flow_accel_mtf * 0.3 +
            norm_institutional_flow_long_slope_mtf * 0.3
        ).clip(-1, 1)
        if is_probe_active: print(f"       - 过程: flow_persistence (Patient): {flow_persistence.loc[probe_date]:.4f}")
        # 1.2 成本控制与效率
        norm_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights_ff)
        norm_vwap_guidance = get_adaptive_mtf_normalized_bipolar_score(main_force_vwap_guidance_raw, df_index, tf_weights=tf_weights_ff)
        norm_execution_alpha = get_adaptive_mtf_normalized_bipolar_score(main_force_execution_alpha_raw, df_index, tf_weights=tf_weights_ff)
        cost_efficiency = (norm_cost_advantage * 0.4 + norm_vwap_guidance * 0.3 + norm_execution_alpha * 0.3).clip(-1, 1)
        if is_probe_active: print(f"       - 过程: cost_efficiency (Patient): {cost_efficiency.loc[probe_date]:.4f}")
        # 1.3 隐蔽吸筹与资金流纯度
        norm_covert_accumulation = get_adaptive_mtf_normalized_score(covert_accumulation_signal_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_suppressive_accumulation = get_adaptive_mtf_normalized_score(suppressive_accumulation_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        covert_accumulation = (norm_covert_accumulation * 0.4 + norm_suppressive_accumulation * 0.3 + norm_flow_credibility * 0.3).clip(0, 1)
        if is_probe_active: print(f"       - 过程: covert_accumulation (Patient): {covert_accumulation.loc[probe_date]:.4f}")
        # 1.4 结构韧性
        norm_chip_health = get_adaptive_mtf_normalized_score(chip_health_score_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_control_solidity = get_adaptive_mtf_normalized_score(control_solidity_index_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        structural_resilience = (norm_chip_health * 0.5 + norm_control_solidity * 0.5).clip(0, 1)
        if is_probe_active: print(f"       - 过程: structural_resilience (Patient): {structural_resilience.loc[probe_date]:.4f}")
        # 融合耐心资本得分
        patient_capital_score = (
            flow_persistence * patient_capital_weights.get('mtf_flow_persistence', 0.3) +
            cost_efficiency * patient_capital_weights.get('cost_efficiency', 0.2) +
            covert_accumulation * patient_capital_weights.get('covert_accumulation', 0.2) +
            norm_flow_credibility * patient_capital_weights.get('flow_purity', 0.15) + # 资金流纯度直接作为耐心资本的组成部分
            structural_resilience * patient_capital_weights.get('structural_resilience', 0.15)
        ).clip(-1, 1)
        if is_probe_active: print(f"       - 过程: patient_capital_score: {patient_capital_score.loc[probe_date]:.4f}")
        # --- 2. 敏捷资本 (Agile Capital) - 脉冲冲击与短期博弈 ---
        # 2.1 高频冲击力与方向性
        norm_ofi_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_ofi_D', mtf_periods_agile_ofi.get('short', []), {}, True, False)
        norm_ofi_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_ofi_D', mtf_periods_agile_ofi.get('short', []), {}, True, True)
        norm_price_impact_asymmetry = get_adaptive_mtf_normalized_bipolar_score(micro_price_impact_asymmetry_raw, df_index, tf_weights=tf_weights_ff)
        ofi_impact = (norm_ofi_slope_mtf * 0.4 + norm_ofi_accel_mtf * 0.3 + norm_price_impact_asymmetry * 0.3).clip(-1, 1)
        if is_probe_active: print(f"       - 过程: ofi_impact (Agile): {ofi_impact.loc[probe_date]:.4f}")
        # 2.2 成交活跃度与集中度
        norm_trade_count = get_adaptive_mtf_normalized_score(trade_count_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_mf_activity_ratio = get_adaptive_mtf_normalized_score(main_force_activity_ratio_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_mf_flow_gini_inverse = 1 - get_adaptive_mtf_normalized_score(main_force_flow_gini_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 基尼系数越低，集中度越高，越是敏捷资本特征
        activity_concentration = (norm_trade_count * 0.3 + norm_mf_activity_ratio * 0.4 + norm_mf_flow_gini_inverse * 0.3).clip(0, 1)
        if is_probe_active: print(f"       - 过程: activity_concentration (Agile): {activity_concentration.loc[probe_date]:.4f}")
        # 2.3 短期动能爆发
        norm_nmfnf_slope_5 = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, df, 'SLOPE_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature"), df_index, tf_weights=tf_weights_ff)
        norm_nmfnf_accel_5 = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, df, 'ACCEL_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature"), df_index, tf_weights=tf_weights_ff)
        short_term_momentum = (norm_nmfnf_slope_5 * 0.6 + norm_nmfnf_accel_5 * 0.4).clip(-1, 1)
        if is_probe_active: print(f"       - 过程: short_term_momentum (Agile): {short_term_momentum.loc[probe_date]:.4f}")
        # 2.4 资金流驱动的题材热度
        norm_theme_hotness = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        if is_probe_active: print(f"       - 过程: norm_theme_hotness (Agile): {norm_theme_hotness.loc[probe_date]:.4f}")
        # 融合敏捷资本得分
        agile_capital_score = (
            ofi_impact * agile_capital_weights.get('ofi_impact', 0.3) +
            activity_concentration * agile_capital_weights.get('activity_concentration', 0.25) +
            short_term_momentum * agile_capital_weights.get('short_term_momentum', 0.25) +
            norm_theme_hotness * agile_capital_weights.get('theme_chasing', 0.2)
        ).clip(-1, 1)
        if is_probe_active: print(f"       - 过程: agile_capital_score: {agile_capital_score.loc[probe_date]:.4f}")
        # --- 3. 动态情境调制 (Dynamic Contextual Modulation) ---
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        if is_probe_active: print(f"       - 过程: norm_volatility_instability (Context): {norm_volatility_instability.loc[probe_date]:.4f}")
        norm_liquidity_supply = get_adaptive_mtf_normalized_score(order_book_liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        if is_probe_active: print(f"       - 原料: order_book_liquidity_supply_D (raw): {order_book_liquidity_supply_raw.loc[probe_date]:.4f}")
        norm_liquidity_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        if is_probe_active: print(f"       - 原料: order_book_clearing_rate_D (raw): {order_book_clearing_rate_raw.loc[probe_date]:.4f}")
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
        if is_probe_active: print(f"       - 原料: market_sentiment_score_D (raw): {market_sentiment_raw.loc[probe_date]:.4f}")
        norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        if is_probe_active: print(f"       - 原料: trend_vitality_index_D (raw): {trend_vitality_raw.loc[probe_date]:.4f}")
        # 情境调制器：影响最终融合权重
        liquidity_mod = (1 + (norm_liquidity_supply + norm_liquidity_clearing_rate)/2 * capital_context_modulator_sensitivity.get('liquidity', 0.2)).clip(0.5, 1.5)
        volatility_mod = (1 - norm_volatility_instability * capital_context_modulator_sensitivity.get('volatility', 0.3)).clip(0.5, 1.5)
        sentiment_mod = (1 + norm_market_sentiment * capital_context_modulator_sensitivity.get('sentiment', 0.1)).clip(0.5, 1.5)
        trend_mod = (1 + norm_trend_vitality * capital_context_modulator_sensitivity.get('trend_vitality', 0.2)).clip(0.5, 1.5)
        if is_probe_active:
            print(f"       - 过程: liquidity_mod (Context): {liquidity_mod.loc[probe_date]:.4f}")
            print(f"       - 过程: volatility_mod (Context): {volatility_mod.loc[probe_date]:.4f}")
            print(f"       - 过程: sentiment_mod (Context): {sentiment_mod.loc[probe_date]:.4f}")
            print(f"       - 过程: trend_mod (Context): {trend_mod.loc[probe_date]:.4f}")
        # --- 4. 融合与非线性建模 (Fusion & Non-linear Modeling) ---
        # 动态调整融合权重
        dynamic_patient_weight = dynamic_fusion_weights.get('patient_base', 0.5) * (1 + trend_mod * dynamic_fusion_weights.get('trend_vitality_mod', 0.2) - volatility_mod * dynamic_fusion_weights.get('volatility_mod', 0.1))
        dynamic_agile_weight = dynamic_fusion_weights.get('agile_base', 0.5) * (1 - trend_mod * dynamic_fusion_weights.get('trend_vitality_mod', 0.2) + volatility_mod * dynamic_fusion_weights.get('volatility_mod', 0.1))
        total_dynamic_weights = dynamic_patient_weight + dynamic_agile_weight
        dynamic_patient_weight /= total_dynamic_weights
        dynamic_agile_weight /= total_dynamic_weights
        if is_probe_active:
            print(f"       - 过程: dynamic_patient_weight (Fusion): {dynamic_patient_weight.loc[probe_date]:.4f}")
            print(f"       - 过程: dynamic_agile_weight (Fusion): {dynamic_agile_weight.loc[probe_date]:.4f}")
        # 最终融合，使用 tanh 非线性化，并考虑情境调制
        capital_signature_score = np.tanh(
            (patient_capital_score * dynamic_patient_weight * liquidity_mod * volatility_mod * sentiment_mod) -
            (agile_capital_score * dynamic_agile_weight * liquidity_mod * volatility_mod * sentiment_mod)
        ).pow(fusion_exponent).clip(-1, 1)
        if is_probe_active: print(f"       - 结果: capital_signature_score: {capital_signature_score.loc[probe_date]:.4f}")
        return capital_signature_score.astype(np.float32)

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

    def _get_mtf_dynamic_score(self, df: pd.DataFrame, signal_base_name: str, periods_list: list, weights_dict: dict, is_bipolar: bool, is_accel: bool = False) -> pd.Series:
        mtf_scores = []
        numeric_weights = {k: v for k, v in weights_dict.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        if total_weight == 0:
            return pd.Series(0.0, index=df.index)
        for period_str, weight in numeric_weights.items():
            period = int(period_str)
            prefix = 'ACCEL' if is_accel else 'SLOPE'
            col_name = f'{prefix}_{period}_{signal_base_name}'
            raw_data = self._get_safe_series(df, df, col_name, 0.0, method_name="_diagnose_axiom_divergence")
            if is_bipolar:
                norm_score = get_adaptive_mtf_normalized_bipolar_score(raw_data, df.index, self.tf_weights_ff)
            else:
                norm_score = get_adaptive_mtf_normalized_score(raw_data, df.index, ascending=True, tf_weights=self.tf_weights_ff)
            mtf_scores.append(norm_score * weight)
        if not mtf_scores:
            return pd.Series(0.0, index=df.index)
        return sum(mtf_scores) / total_weight

    def _calculate_mtf_cohesion_divergence(self, df: pd.DataFrame, signal_base_name: str, short_periods: List[int], long_periods: List[int], is_bipolar: bool, tf_weights: Dict, probe_date: pd.Timestamp, is_probe_active: bool, method_name: str) -> pd.Series:
        """
        【V4.0 升级】计算双极性多时间框架的共振/背离因子。
        分析短期和长期斜率/加速度的一致性及其方向。
        返回 -1 到 1 的分数，正值表示看涨共振，负值表示看跌共振。
        """
        short_slope_scores = []
        short_accel_scores = []
        long_slope_scores = []
        long_accel_scores = []
        # 获取短期斜率和加速度
        for p in short_periods:
            slope_col = f'SLOPE_{p}_{signal_base_name}'
            accel_col = f'ACCEL_{p}_{signal_base_name}'
            slope_raw = self._get_safe_series(df, df, slope_col, 0.0, method_name)
            accel_raw = self._get_safe_series(df, df, accel_col, 0.0, method_name)
            if is_probe_active:
                print(f"       - 原料: {slope_col} (raw): {slope_raw.loc[probe_date]:.4f}")
                print(f"       - 原料: {accel_col} (raw): {accel_raw.loc[probe_date]:.4f}")
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
            slope_raw = self._get_safe_series(df, df, slope_col, 0.0, method_name)
            accel_raw = self._get_safe_series(df, df, accel_col, 0.0, method_name)
            if is_probe_active:
                print(f"       - 原料: {slope_col} (raw): {slope_raw.loc[probe_date]:.4f}")
                print(f"       - 原料: {accel_col} (raw): {accel_raw.loc[probe_date]:.4f}")
            if is_bipolar:
                long_slope_scores.append(get_adaptive_mtf_normalized_bipolar_score(slope_raw, df.index, tf_weights))
                long_accel_scores.append(get_adaptive_mtf_normalized_bipolar_score(accel_raw, df.index, tf_weights))
            else:
                long_slope_scores.append(get_adaptive_mtf_normalized_score(raw_data, df.index, ascending=True, tf_weights=tf_weights))
                long_accel_scores.append(get_adaptive_mtf_normalized_score(raw_data, df.index, ascending=True, tf_weights=tf_weights))
        # 平均短期和长期分数
        avg_short_slope = sum(short_slope_scores) / len(short_slope_scores) if short_slope_scores else pd.Series(0.0, index=df.index)
        avg_short_accel = sum(short_accel_scores) / len(short_accel_scores) if short_accel_scores else pd.Series(0.0, index=df.index)
        avg_long_slope = sum(long_slope_scores) / len(long_slope_scores) if long_slope_scores else pd.Series(0.0, index=df.index)
        avg_long_accel = sum(long_accel_scores) / len(long_accel_scores) if long_accel_scores else pd.Series(0.0, index=df.index)
        if is_probe_active:
            print(f"       - 过程: avg_short_slope ({signal_base_name}): {avg_short_slope.loc[probe_date]:.4f}")
            print(f"       - 过程: avg_short_accel ({signal_base_name}): {avg_short_accel.loc[probe_date]:.4f}")
            print(f"       - 过程: avg_long_slope ({signal_base_name}): {avg_long_slope.loc[probe_date]:.4f}")
            print(f"       - 过程: avg_long_accel ({signal_base_name}): {avg_long_accel.loc[probe_date]:.4f}")
        # 计算双极性共振/背离分数
        # 1. 方向一致性：如果短期和长期方向一致，则为正；相反则为负。
        direction_alignment = np.sign(avg_short_slope) * np.sign(avg_long_slope)
        # 2. 强度一致性：短期和长期强度越接近，一致性越高。
        # 使用几何平均来衡量强度一致性，并确保其为正值
        strength_cohesion_slope = (1 - (avg_short_slope.abs() - avg_long_slope.abs()).abs()).clip(0, 1)
        strength_cohesion_accel = (1 - (avg_short_accel.abs() - avg_long_accel.abs()).abs()).clip(0, 1)
        # 综合强度一致性
        strength_cohesion = (strength_cohesion_slope + strength_cohesion_accel) / 2
        # 3. 最终双极性共振分数：方向 * 强度
        # 如果方向一致，则强度一致性越高，分数越高（正向或负向）
        # 如果方向不一致，则分数趋近于0
        mtf_resonance_score = strength_cohesion * direction_alignment
        return mtf_resonance_score.astype(np.float32)

    def _diagnose_fund_flow_divergence_signals(self, df: pd.DataFrame, norm_window: int, axiom_divergence: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V4.0 · 动态共振与微观意图精炼版】诊断资金流看涨/看跌背离信号。
        - 核心升级：1) 双极性多时间框架共振/背离因子：将MTF共振因子升级为-1到1的双极性分数，直接指示看涨/看跌共振或冲突。
                    2) 更全面的微观意图强度：整合更多微观资金流信号（如冲击弹性、T0效率）及其动态，构建更鲁棒的微观意图分数。
                    3) 看涨/看跌专属动态非线性指数：为看涨和看跌信号分别设置动态调整的非线性指数，实现更精细的放大/抑制。
                    4) 情境自适应纯度调制：根据市场情绪和资金流可信度动态调整纯度过滤的敏感度。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资金流看涨/看跌背离信号 (V4.0 · 动态共振与微观意图精炼版)”...")
        df_index = df.index
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        self.tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ffd_params = get_param_value(p_conf_ff.get('fund_flow_divergence_params'), {})
        purity_weights = get_param_value(ffd_params.get('purity_weights'), {"deception_inverse": 0.5, "wash_trade_inverse": 0.5})
        confirmation_weights = get_param_value(ffd_params.get('confirmation_weights'), {"conviction_positive": 0.5, "flow_momentum_positive": 0.5})
        context_modulator_weights = get_param_value(ffd_params.get('context_modulator_weights'), {"strategic_posture": 0.4, "flow_credibility": 0.3, "retail_panic_fomo_context": 0.3})
        non_linear_exponent_base = get_param_value(ffd_params.get('non_linear_exponent'), 1.2)
        bullish_divergence_threshold = get_param_value(ffd_params.get('bullish_divergence_threshold'), 0.1)
        bearish_divergence_threshold = get_param_value(ffd_params.get('bearish_divergence_threshold'), 0.1)
        retail_panic_fomo_sensitivity = get_param_value(ffd_params.get('retail_panic_fomo_sensitivity'), 0.5)
        # V4.0 新增参数
        mtf_resonance_factor_weights = get_param_value(ffd_params.get('mtf_resonance_factor_weights'), {"nmfnf_cohesion": 0.6, "conviction_cohesion": 0.4})
        micro_macro_divergence_weights = get_param_value(ffd_params.get('micro_macro_divergence_weights'), {"micro_intent_strength": 0.5, "macro_flow_momentum": 0.5})
        dynamic_context_modulator_sensitivity = get_param_value(ffd_params.get('dynamic_context_modulator_sensitivity'), {"volatility_instability": 0.2, "flow_credibility": 0.15, "trend_vitality": 0.1, "market_sentiment": 0.1})
        micro_intent_signals_weights = get_param_value(ffd_params.get('micro_intent_signals_weights'), {"order_book_imbalance": 0.3, "exhaustion_rate": 0.3, "micro_impact_elasticity": 0.2, "main_force_t0_efficiency": 0.2})
        bullish_exponent_context_factors = get_param_value(ffd_params.get('bullish_exponent_context_factors'), {"trend_vitality": 0.7, "volatility_inverse": 0.3})
        bearish_exponent_context_factors = get_param_value(ffd_params.get('bearish_exponent_context_factors'), {"volatility_instability": 0.6, "flow_credibility_inverse": 0.4})
        purity_context_mod_factors = get_param_value(ffd_params.get('purity_context_mod_factors'), {"market_sentiment": 0.5, "flow_credibility": 0.5})
        # --- 信号依赖校验 ---
        required_signals = [
            'SCORE_FF_AXIOM_CONVICTION', 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 'SCORE_FF_STRATEGIC_POSTURE',
            'flow_credibility_index_D', 'retail_panic_surrender_index_D', 'retail_fomo_premium_index_D',
            'SLOPE_5_deception_index_D', 'SLOPE_13_deception_index_D', 'SLOPE_21_deception_index_D',
            'SLOPE_5_wash_trade_intensity_D', 'SLOPE_13_wash_trade_intensity_D', 'SLOPE_21_wash_trade_intensity_D',
            # V4.0 新增依赖
            'SLOPE_5_NMFNF_D', 'SLOPE_13_NMFNF_D', 'SLOPE_21_NMFNF_D', 'SLOPE_55_NMFNF_D',
            'ACCEL_5_NMFNF_D', 'ACCEL_13_NMFNF_D', 'ACCEL_21_NMFNF_D', 'ACCEL_55_NMFNF_D',
            'SLOPE_5_main_force_conviction_index_D', 'SLOPE_13_main_force_conviction_index_D', 'SLOPE_21_main_force_conviction_index_D', 'SLOPE_55_main_force_conviction_index_D',
            'ACCEL_5_main_force_conviction_index_D', 'ACCEL_13_main_force_conviction_index_D', 'ACCEL_21_main_force_conviction_index_D', 'ACCEL_55_main_force_conviction_index_D',
            'order_book_imbalance_D', 'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D', # 微观意图信号
            'micro_impact_elasticity_D', 'main_force_t0_efficiency_D', # V4.0 更全面的微观意图信号
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'trend_vitality_index_D', 'market_sentiment_score_D' # 动态情境调制器
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_fund_flow_divergence_signals", atomic_states=self.strategy.atomic_states):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)
        # --- 原始数据获取 (用于探针和计算) ---
        # 基础背离
        bullish_base_divergence, bearish_base_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        # 确认信号
        axiom_conviction = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_CONVICTION', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        axiom_flow_momentum = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        # 纯度过滤信号
        deception_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_deception_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        deception_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_deception_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        deception_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_deception_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        wash_trade_slope_5_raw = self._get_safe_series(df, df, 'SLOPE_5_wash_trade_intensity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        wash_trade_slope_13_raw = self._get_safe_series(df, df, 'SLOPE_13_wash_trade_intensity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        wash_trade_slope_21_raw = self._get_safe_series(df, df, 'SLOPE_21_wash_trade_intensity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        # 情境校准信号
        strategic_posture = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_STRATEGIC_POSTURE', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        flow_credibility_raw = self._get_safe_series(df, df, 'flow_credibility_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        retail_panic_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        retail_fomo_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        # V4.0 新增原始数据
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        micro_impact_elasticity_raw = self._get_safe_series(df, df, 'micro_impact_elasticity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        main_force_t0_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_efficiency_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        trend_vitality_raw = self._get_safe_series(df, df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        market_sentiment_raw = self._get_safe_series(df, df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        # --- 1. 纯度过滤 (Purity Filter) ---
        # 欺骗指数多时间框架融合
        norm_deception_slope_5 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_5_raw, df_index, self.tf_weights_ff)
        norm_deception_slope_13 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_13_raw, df_index, self.tf_weights_ff)
        norm_deception_slope_21 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_21_raw, df_index, self.tf_weights_ff)
        norm_deception_multi_tf = (
            norm_deception_slope_5 * 0.5 +
            norm_deception_slope_13 * 0.3 +
            norm_deception_slope_21 * 0.2
        ).clip(-1, 1)
        # 对倒强度多时间框架融合
        norm_wash_trade_slope_5 = get_adaptive_mtf_normalized_score(wash_trade_slope_5_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_slope_13 = get_adaptive_mtf_normalized_score(wash_trade_slope_13_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_slope_21 = get_adaptive_mtf_normalized_score(wash_trade_slope_21_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_multi_tf = (
            norm_wash_trade_slope_5 * 0.5 +
            norm_wash_trade_slope_13 * 0.3 +
            norm_wash_trade_slope_21 * 0.2
        ).clip(0, 1)
        # V4.0 情境自适应纯度调制
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, self.tf_weights_ff)
        norm_flow_credibility_purity = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        purity_context_mod = (
            (1 + norm_market_sentiment.abs() * np.sign(norm_market_sentiment) * purity_context_mod_factors.get('market_sentiment', 0.5)) *
            (1 + (norm_flow_credibility_purity - 0.5) * purity_context_mod_factors.get('flow_credibility', 0.5))
        ).clip(0.5, 1.5)
        if is_probe_active: print(f"       - 过程: purity_context_mod (V4.0): {purity_context_mod.loc[probe_date]:.4f}")
        # 看涨纯度：欺骗指数负向（诱空）且对倒强度低
        bullish_purity_modulator = (1 + norm_deception_multi_tf.clip(upper=0).abs() * purity_weights.get('deception_inverse', 0.5) * purity_context_mod) * (1 - norm_wash_trade_multi_tf * purity_weights.get('wash_trade_inverse', 0.5) * purity_context_mod)
        # 看跌纯度：欺骗指数正向（诱多）且对倒强度低
        bearish_purity_modulator = (1 + norm_deception_multi_tf.clip(lower=0) * purity_weights.get('deception_inverse', 0.5) * purity_context_mod) * (1 - norm_wash_trade_multi_tf * purity_weights.get('wash_trade_inverse', 0.5) * purity_context_mod)
        # --- 2. 意图确认 (Intent Confirmation) ---
        # 信念韧性确认
        bullish_conviction_confirm = axiom_conviction.clip(lower=0) * confirmation_weights.get('conviction_positive', 0.5)
        bearish_conviction_confirm = axiom_conviction.clip(upper=0).abs() * confirmation_weights.get('conviction_positive', 0.5)
        # 资金流纯度与动能确认
        bullish_flow_momentum_confirm = axiom_flow_momentum.clip(lower=0) * confirmation_weights.get('flow_momentum_positive', 0.5)
        bearish_flow_momentum_confirm = axiom_flow_momentum.clip(upper=0).abs() * confirmation_weights.get('flow_momentum_positive', 0.5)
        # 综合确认
        bullish_confirmation_score = (bullish_conviction_confirm + bullish_flow_momentum_confirm).clip(0, 1)
        bearish_confirmation_score = (bearish_conviction_confirm + bearish_flow_momentum_confirm).clip(0, 1)
        # --- 3. 情境校准 (Contextual Calibration) ---
        # 资金流战略态势
        norm_strategic_posture = get_adaptive_mtf_normalized_bipolar_score(strategic_posture, df_index, self.tf_weights_ff)
        bullish_posture_mod = norm_strategic_posture.clip(lower=0) * context_modulator_weights.get('strategic_posture', 0.4)
        bearish_posture_mod = norm_strategic_posture.clip(upper=0).abs() * context_modulator_weights.get('strategic_posture', 0.4)
        # 资金流可信度
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        credibility_mod = norm_flow_credibility * context_modulator_weights.get('flow_credibility', 0.3)
        # 散户情绪情境
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        # 恐慌时增强看涨，狂热时增强看跌
        bullish_retail_context_mod = norm_retail_panic * retail_panic_fomo_sensitivity * context_modulator_weights.get('retail_panic_fomo_context', 0.3)
        bearish_retail_context_mod = norm_retail_fomo * retail_panic_fomo_sensitivity * context_modulator_weights.get('retail_panic_fomo_context', 0.3)
        # 综合情境调制器
        bullish_context_modulator = (1 + bullish_posture_mod + credibility_mod + bullish_retail_context_mod).clip(0.5, 2.0)
        bearish_context_modulator = (1 + bearish_posture_mod + credibility_mod + bearish_retail_context_mod).clip(0.5, 2.0)
        # --- V4.0 升级: 双极性多时间框架共振/背离因子 (Bipolar MTF Resonance/Divergence Factor) ---
        # 短期和长期时间框架定义
        short_periods = [5, 13]
        long_periods = [21, 55]
        nmfnf_bipolar_resonance_factor = self._calculate_mtf_cohesion_divergence(df, 'NMFNF_D', short_periods, long_periods, True, self.tf_weights_ff, probe_date, is_probe_active, "_diagnose_fund_flow_divergence_signals")
        conviction_bipolar_resonance_factor = self._calculate_mtf_cohesion_divergence(df, 'main_force_conviction_index_D', short_periods, long_periods, True, self.tf_weights_ff, probe_date, is_probe_active, "_diagnose_fund_flow_divergence_signals")
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
        micro_intent_strength = (
            norm_order_book_imbalance * micro_intent_signals_weights.get('order_book_imbalance', 0.3) +
            (norm_sell_exhaustion - norm_buy_exhaustion) * micro_intent_signals_weights.get('exhaustion_rate', 0.3) +
            norm_micro_impact_elasticity * micro_intent_signals_weights.get('micro_impact_elasticity', 0.2) +
            norm_main_force_t0_efficiency * micro_intent_signals_weights.get('main_force_t0_efficiency', 0.2)
        ).clip(-1, 1)
        # 微观-宏观背离因子：比较微观意图强度与宏观资金流动能
        micro_macro_divergence_factor = (
            micro_intent_strength * micro_macro_divergence_weights.get('micro_intent_strength', 0.5) -
            axiom_flow_momentum * micro_macro_divergence_weights.get('macro_flow_momentum', 0.5)
        ).clip(-1, 1)
        # 将其转换为一个放大因子，正向背离放大看涨，负向背离放大看跌
        bullish_micro_macro_mod = (1 + micro_macro_divergence_factor.clip(lower=0))
        bearish_micro_macro_mod = (1 + micro_macro_divergence_factor.clip(upper=0).abs())
        # --- V4.0 升级: 看涨/看跌专属动态非线性指数 (Bullish/Bearish Adaptive Non-linear Exponents) ---
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_flow_credibility_exp = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        # 看涨指数：趋势活力越高，波动性越低，指数越大
        bullish_exponent_mod_factor = (
            norm_trend_vitality * bullish_exponent_context_factors.get('trend_vitality', 0.7) +
            (1 - norm_volatility_instability) * bullish_exponent_context_factors.get('volatility_inverse', 0.3)
        ).clip(0, 1)
        bullish_dynamic_non_linear_exponent = non_linear_exponent_base * (1 + bullish_exponent_mod_factor * dynamic_context_modulator_sensitivity.get('trend_vitality', 0.1))
        # 看跌指数：波动性越高，资金流可信度越低，指数越大
        bearish_exponent_mod_factor = (
            norm_volatility_instability * bearish_exponent_context_factors.get('volatility_instability', 0.6) +
            (1 - norm_flow_credibility_exp) * bearish_exponent_context_factors.get('flow_credibility_inverse', 0.4)
        ).clip(0, 1)
        bearish_dynamic_non_linear_exponent = non_linear_exponent_base * (1 + bearish_exponent_mod_factor * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))
        # --- 4. 非线性融合 (Non-linear Fusion) ---
        # 看涨背离
        bullish_divergence_score = (
            bullish_base_divergence *
            (bullish_purity_modulator * purity_context_mod).clip(0.1, 2.0) * # 动态调整纯度敏感度
            (bullish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0) * # 动态调整确认敏感度
            (bullish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0) * # 动态调整情境敏感度
            (1 + mtf_bipolar_resonance_factor.clip(lower=0)) * # 引入双极性MTF共振因子，只取正向放大看涨
            bullish_micro_macro_mod # 引入微观-宏观背离因子
        ).pow(bullish_dynamic_non_linear_exponent).clip(0, 1) # 动态调整非线性指数
        # 看跌背离
        bearish_divergence_score = (
            bearish_base_divergence *
            (bearish_purity_modulator * purity_context_mod).clip(0.1, 2.0) * # 动态调整纯度敏感度
            (bearish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0) * # 动态调整确认敏感度
            (bearish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0) * # 动态调整情境敏感度
            (1 + mtf_bipolar_resonance_factor.clip(upper=0).abs()) * # 引入双极性MTF共振因子，只取负向放大看跌
            bearish_micro_macro_mod # 引入微观-宏观背离因子
        ).pow(bearish_dynamic_non_linear_exponent).clip(0, 1) # 动态调整非线性指数
        # 应用阈值
        bullish_divergence_score = bullish_divergence_score.where(bullish_divergence_score > bullish_divergence_threshold, 0.0)
        bearish_divergence_score = bearish_divergence_score.where(bearish_divergence_score > bearish_divergence_threshold, 0.0)
        return bullish_divergence_score.astype(np.float32), bearish_divergence_score.astype(np.float32)











