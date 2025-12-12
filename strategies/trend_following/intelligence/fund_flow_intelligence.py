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
        # 修正: 在初始化时加载 tf_weights_ff，确保其始终可用
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        self.tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})

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

    def _get_mtf_dynamic_score(self, df: pd.DataFrame, signal_base_name: str, periods_list: list, weights_dict: dict, is_bipolar: bool, is_accel: bool = False, method_name: str = "未知方法") -> pd.Series:
        """
        【V1.2 · 修复与增强版】计算多时间框架的动态得分。
        - 核心修复: 修复了 `else` 分支中 `raw_data` 未定义的 bug。
        - 核心增强: 增加了 `method_name` 参数，以便在 `_get_safe_series` 中提供更详细的警告信息。
        """
        mtf_scores = []
        numeric_weights = {k: v for k, v in weights_dict.items() if isinstance(v, (int, float))}
        total_weight = sum(numeric_weights.values())
        if total_weight == 0:
            # 如果权重总和为0，则返回一个全为0的Series，避免除以零错误
            return pd.Series(0.0, index=df.index)
        for period_str, weight in numeric_weights.items():
            period = int(period_str)
            prefix = 'ACCEL' if is_accel else 'SLOPE'
            col_name = f'{prefix}_{period}_{signal_base_name}'
            raw_data = self._get_safe_series(df, df, col_name, 0.0, method_name=method_name) # 修正: 传入 method_name
            if is_bipolar:
                norm_score = get_adaptive_mtf_normalized_bipolar_score(raw_data, df.index, self.tf_weights_ff)
            else:
                norm_score = get_adaptive_mtf_normalized_score(raw_data, df.index, ascending=True, tf_weights=self.tf_weights_ff) # 修正: 确保使用 raw_data
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
        print(f"    -> [资金流情报校验] 计算“资金流内部分歧与意图张力(SCORE_FF_AXIOM_CONVICTION)” 分数：{axiom_conviction.mean():.4f}")
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
        # 修正: 移除重复加载 self.tf_weights_ff，因为它已在 __init__ 中加载
        # self.tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
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
        norm_nmfnf_slope_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence")
        norm_nmfnf_accel_mtf = self._get_mtf_dynamic_score(df, 'NMFNF_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence")
        norm_mf_conviction_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence")
        norm_mf_conviction_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_conviction_index_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence")
        nmfnf_dynamic_score = (norm_nmfnf_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_nmfnf_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        mf_conviction_dynamic_score = (norm_mf_conviction_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_mf_conviction_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        core_divergence_score = (nmfnf_dynamic_score - mf_conviction_dynamic_score).clip(-1, 1)
        # --- 2. 结构性张力 (Structural Tension) ---
        norm_lg_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence")
        norm_lg_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence")
        norm_retail_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence")
        norm_retail_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'retail_net_flow_calibrated_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence")
        lg_flow_dynamic_score = (norm_lg_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_lg_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        retail_flow_dynamic_score = (norm_retail_flow_slope_mtf * slope_accel_fusion_weights.get('slope', 0.6) + norm_retail_flow_accel_mtf * slope_accel_fusion_weights.get('accel', 0.4)).clip(-1, 1)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_premium_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_surrender_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        structural_divergence_base = (lg_flow_dynamic_score - retail_flow_dynamic_score)
        retail_modulator = (1 - norm_retail_fomo * retail_sentiment_mod_sensitivity) + (norm_retail_panic * retail_sentiment_mod_sensitivity)
        structural_tension_score = (structural_divergence_base * retail_modulator).clip(-1, 1)
        # --- 3. 诡道意图张力 (Deceptive Intent Tension) ---
        norm_deception_slope_mtf = self._get_mtf_dynamic_score(df, 'deception_index_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence")
        norm_deception_accel_mtf = self._get_mtf_dynamic_score(df, 'deception_index_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence")
        norm_wash_trade_slope_mtf = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name="_diagnose_axiom_divergence")
        norm_wash_trade_accel_mtf = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name="_diagnose_axiom_divergence")
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
        norm_obi_slope_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_slope_periods, divergence_slope_weights, True, False, method_name="_diagnose_axiom_divergence")
        norm_obi_accel_mtf = self._get_mtf_dynamic_score(df, 'order_book_imbalance_D', divergence_accel_periods, divergence_accel_weights, True, True, method_name="_diagnose_axiom_divergence")
        obi_dynamic_pulse = (norm_obi_slope_mtf * obi_dynamic_params.get('slope', 0.6) + norm_obi_accel_mtf * obi_dynamic_params.get('accel', 0.4)).clip(-1, 1)
        buy_exh_dynamic_params = micro_intent_dynamic_signals.get('buy_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        norm_buy_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name="_diagnose_axiom_divergence")
        norm_buy_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'buy_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name="_diagnose_axiom_divergence")
        buy_exh_dynamic_pulse = (norm_buy_exh_slope_mtf * buy_exh_dynamic_params.get('slope', 0.5) + norm_buy_exh_accel_mtf * buy_exh_dynamic_params.get('accel', 0.5)).clip(0, 1)
        sell_exh_dynamic_params = micro_intent_dynamic_signals.get('sell_quote_exhaustion_rate_D', {"slope": 0.5, "accel": 0.5, "weight": 0.15})
        norm_sell_exh_slope_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_slope_periods, divergence_slope_weights, False, False, method_name="_diagnose_axiom_divergence")
        norm_sell_exh_accel_mtf = self._get_mtf_dynamic_score(df, 'sell_quote_exhaustion_rate_D', divergence_accel_periods, divergence_accel_weights, False, True, method_name="_diagnose_axiom_divergence")
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
        【V5.1 · 意图推断与情境预测版】资金流公理一：诊断“战场控制权”
        - 核心升级1: 诡道博弈深度情境感知：引入市场情绪作为诡道调制器的情境因子，对欺骗指数和对倒强度的影响进行动态校准，更精准识别主力诡道意图。
        - 核心升级2: 微观盘口意图推断：融合盘口枯竭率与微观结构效率，形成更具洞察力的微观意图分数，捕捉主力在盘口上的真实攻防。
        - 核心升级3: 多维度情境自适应权重：扩展动态权重调制器，引入市场情绪，使宏观资金流向与微观盘口控制力的融合权重在不同市场情境下更具适应性。
        - 核心升级4: 资金流结构与效率非线性建模：在各子分数融合中引入更多非线性函数，捕捉资金流各组件间更复杂的交互作用。
        - 核心升级5: 预测性与前瞻性增强：根据市场情境动态调整战场控制权速度和加速度的融合权重，使其在趋势转折点更具前瞻性。
        - 核心升级6: 新增资金指标整合：
            - 盘口买卖力量：dip_buy_absorption_strength, panic_buy_absorption_contribution, opening_buy_strength, pre_closing_buy_posture, closing_auction_buy_ambush, main_force_buy_ofi, retail_buy_ofi, bid_side_liquidity
            - 盘口买卖效率：main_force_t0_buy_efficiency, buy_flow_efficiency_index, buy_order_book_clearing_rate, vwap_buy_control_strength, main_force_vwap_up_guidance, vwap_cross_up_intensity
            - 盘口卖出力量：dip_sell_pressure_resistance, panic_sell_volume_contribution, opening_sell_strength, pre_closing_sell_posture, closing_auction_sell_ambush, main_force_sell_ofi, retail_sell_ofi, ask_side_liquidity, main_force_on_peak_sell_flow
            - 盘口卖出效率：main_force_t0_sell_efficiency, sell_flow_efficiency_index, sell_order_book_clearing_rate, vwap_sell_control_strength, main_force_vwap_down_guidance, vwap_cross_down_intensity
            - 对倒行为：wash_trade_buy_volume, wash_trade_sell_volume
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“战场控制权 (V5.1 · 意图推断与情境预测版)”公理...")
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

        # [新增代码行] V5.1 新增微观盘口意图推断权重
        micro_buy_power_weights = get_param_value(ac_params.get('micro_buy_power_weights'), {
            'dip_buy_absorption_strength': 0.1, 'panic_buy_absorption_contribution': 0.1,
            'opening_buy_strength': 0.1, 'pre_closing_buy_posture': 0.1, 'closing_auction_buy_ambush': 0.1,
            'main_force_buy_ofi': 0.1, 'retail_buy_ofi': 0.05, 'bid_side_liquidity': 0.05,
            'main_force_t0_buy_efficiency': 0.05, 'buy_flow_efficiency_index': 0.05,
            'buy_order_book_clearing_rate': 0.05, 'vwap_buy_control_strength': 0.05,
            'main_force_vwap_up_guidance': 0.05, 'vwap_cross_up_intensity': 0.05,
            'wash_trade_buy_volume': -0.05 # 对倒买入视为负面
        })
        micro_sell_power_weights = get_param_value(ac_params.get('micro_sell_power_weights'), {
            'dip_sell_pressure_resistance': 0.1, 'panic_sell_volume_contribution': 0.1,
            'opening_sell_strength': 0.1, 'pre_closing_sell_posture': 0.1, 'closing_auction_sell_ambush': 0.1,
            'main_force_sell_ofi': 0.1, 'retail_sell_ofi': 0.05, 'ask_side_liquidity': 0.05,
            'main_force_t0_sell_efficiency': 0.05, 'sell_flow_efficiency_index': 0.05,
            'sell_order_book_clearing_rate': 0.05, 'vwap_sell_control_strength': 0.05,
            'main_force_vwap_down_guidance': 0.05, 'vwap_cross_down_intensity': 0.05,
            'main_force_on_peak_sell_flow': 0.05,
            'wash_trade_sell_volume': -0.05 # 对倒卖出视为负面
        })

        # --- 信号依赖校验 ---
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'order_book_imbalance_D', 'microstructure_efficiency_index_D', 'wash_trade_intensity_D',
            'deception_index_D', 'main_force_conviction_index_D', 'flow_credibility_index_D',
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name,
            dynamic_weight_modulator_signal_3_name, # V5.0 动态权重依赖
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            deception_context_modulator_signal_name, # V5.0 诡道情境依赖
            dynamic_evolution_context_modulator_signal_name, # V5.0 动态演化情境依赖
            # [新增代码行] V5.1 新增资金指标
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
            'bid_side_liquidity_D', 'ask_side_liquidity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_consensus"):
            return pd.Series(0.0, index=df.index)

        # --- 原始数据获取 (用于探针和计算) ---
        print(f"        -> [探针] 正在获取原始数据...")
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

        # [新增代码行] V5.1 获取新增资金指标
        dip_buy_absorption_strength_raw = self._get_safe_series(df, df, 'dip_buy_absorption_strength_D', 0.0, method_name="_diagnose_axiom_consensus")
        dip_sell_pressure_resistance_raw = self._get_safe_series(df, df, 'dip_sell_pressure_resistance_D', 0.0, method_name="_diagnose_axiom_consensus")
        panic_sell_volume_contribution_raw = self._get_safe_series(df, df, 'panic_sell_volume_contribution_D', 0.0, method_name="_diagnose_axiom_consensus")
        panic_buy_absorption_contribution_raw = self._get_safe_series(df, df, 'panic_buy_absorption_contribution_D', 0.0, method_name="_diagnose_axiom_consensus")
        opening_buy_strength_raw = self._get_safe_series(df, df, 'opening_buy_strength_D', 0.0, method_name="_diagnose_axiom_consensus")
        opening_sell_strength_raw = self._get_safe_series(df, df, 'opening_sell_strength_D', 0.0, method_name="_diagnose_axiom_consensus")
        pre_closing_buy_posture_raw = self._get_safe_series(df, df, 'pre_closing_buy_posture_D', 0.0, method_name="_diagnose_axiom_consensus")
        pre_closing_sell_posture_raw = self._get_safe_series(df, df, 'pre_closing_sell_posture_D', 0.0, method_name="_diagnose_axiom_consensus")
        closing_auction_buy_ambush_raw = self._get_safe_series(df, df, 'closing_auction_buy_ambush_D', 0.0, method_name="_diagnose_axiom_consensus")
        closing_auction_sell_ambush_raw = self._get_safe_series(df, df, 'closing_auction_sell_ambush_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_t0_buy_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_buy_efficiency_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_t0_sell_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_sell_efficiency_D', 0.0, method_name="_diagnose_axiom_consensus")
        buy_flow_efficiency_index_raw = self._get_safe_series(df, df, 'buy_flow_efficiency_index_D', 0.0, method_name="_diagnose_axiom_consensus")
        sell_flow_efficiency_index_raw = self._get_safe_series(df, df, 'sell_flow_efficiency_index_D', 0.0, method_name="_diagnose_axiom_consensus")
        buy_order_book_clearing_rate_raw = self._get_safe_series(df, df, 'buy_order_book_clearing_rate_D', 0.0, method_name="_diagnose_axiom_consensus")
        sell_order_book_clearing_rate_raw = self._get_safe_series(df, df, 'sell_order_book_clearing_rate_D', 0.0, method_name="_diagnose_axiom_consensus")
        vwap_buy_control_strength_raw = self._get_safe_series(df, df, 'vwap_buy_control_strength_D', 0.0, method_name="_diagnose_axiom_consensus")
        vwap_sell_control_strength_raw = self._get_safe_series(df, df, 'vwap_sell_control_strength_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_vwap_up_guidance_raw = self._get_safe_series(df, df, 'main_force_vwap_up_guidance_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_vwap_down_guidance_raw = self._get_safe_series(df, df, 'main_force_vwap_down_guidance_D', 0.0, method_name="_diagnose_axiom_consensus")
        vwap_cross_up_intensity_raw = self._get_safe_series(df, df, 'vwap_cross_up_intensity_D', 0.0, method_name="_diagnose_axiom_consensus")
        vwap_cross_down_intensity_raw = self._get_safe_series(df, df, 'vwap_cross_down_intensity_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_on_peak_sell_flow_raw = self._get_safe_series(df, df, 'main_force_on_peak_sell_flow_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_buy_ofi_raw = self._get_safe_series(df, df, 'main_force_buy_ofi_D', 0.0, method_name="_diagnose_axiom_consensus")
        main_force_sell_ofi_raw = self._get_safe_series(df, df, 'main_force_sell_ofi_D', 0.0, method_name="_diagnose_axiom_consensus")
        retail_buy_ofi_raw = self._get_safe_series(df, df, 'retail_buy_ofi_D', 0.0, method_name="_diagnose_axiom_consensus")
        retail_sell_ofi_raw = self._get_safe_series(df, df, 'retail_sell_ofi_D', 0.0, method_name="_diagnose_axiom_consensus")
        wash_trade_buy_volume_raw = self._get_safe_series(df, df, 'wash_trade_buy_volume_D', 0.0, method_name="_diagnose_axiom_consensus")
        wash_trade_sell_volume_raw = self._get_safe_series(df, df, 'wash_trade_sell_volume_D', 0.0, method_name="_diagnose_axiom_consensus")
        bid_side_liquidity_raw = self._get_safe_series(df, df, 'bid_side_liquidity_D', 0.0, method_name="_diagnose_axiom_consensus")
        ask_side_liquidity_raw = self._get_safe_series(df, df, 'ask_side_liquidity_D', 0.0, method_name="_diagnose_axiom_consensus")

        # --- 1. 宏观资金流向 (Macro Fund Flow) ---
        flow_consensus_score = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_raw - retail_flow_raw, df_index, tf_weights_ff)
        print(f"        -> [探针] 宏观资金流向 (flow_consensus_score): {flow_consensus_score.iloc[-1]:.4f}")

        # --- 2. 微观盘口意图推断 (Micro Order Book Intent Inference) ---
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff)
        impact_score = get_adaptive_mtf_normalized_bipolar_score(ofi_impact_raw, df_index, tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 枯竭率越低越好
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 枯竭率越高越好
        exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)

        # [新增代码行] V5.1 整合新增资金指标到微观盘口意图推断
        # 买方力量
        norm_dip_buy_absorption_strength = get_adaptive_mtf_normalized_score(dip_buy_absorption_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_panic_buy_absorption_contribution = get_adaptive_mtf_normalized_score(panic_buy_absorption_contribution_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_opening_buy_strength = get_adaptive_mtf_normalized_score(opening_buy_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_pre_closing_buy_posture = get_adaptive_mtf_normalized_score(pre_closing_buy_posture_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_closing_auction_buy_ambush = get_adaptive_mtf_normalized_score(closing_auction_buy_ambush_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(main_force_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(retail_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_bid_side_liquidity = get_adaptive_mtf_normalized_score(bid_side_liquidity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 买方效率
        norm_main_force_t0_buy_efficiency = get_adaptive_mtf_normalized_score(main_force_t0_buy_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_flow_efficiency_index = get_adaptive_mtf_normalized_score(buy_flow_efficiency_index_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_order_book_clearing_rate = get_adaptive_mtf_normalized_score(buy_order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_buy_control_strength = get_adaptive_mtf_normalized_score(vwap_buy_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_vwap_up_guidance = get_adaptive_mtf_normalized_score(main_force_vwap_up_guidance_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_cross_up_intensity = get_adaptive_mtf_normalized_score(vwap_cross_up_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(wash_trade_buy_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)

        # 卖方力量
        norm_dip_sell_pressure_resistance = get_adaptive_mtf_normalized_score(dip_sell_pressure_resistance_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_panic_sell_volume_contribution = get_adaptive_mtf_normalized_score(panic_sell_volume_contribution_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_opening_sell_strength = get_adaptive_mtf_normalized_score(opening_sell_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_pre_closing_sell_posture = get_adaptive_mtf_normalized_score(pre_closing_sell_posture_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_closing_auction_sell_ambush = get_adaptive_mtf_normalized_score(closing_auction_sell_ambush_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(main_force_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(retail_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_ask_side_liquidity = get_adaptive_mtf_normalized_score(ask_side_liquidity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_on_peak_sell_flow = get_adaptive_mtf_normalized_score(main_force_on_peak_sell_flow_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 卖方效率
        norm_main_force_t0_sell_efficiency = get_adaptive_mtf_normalized_score(main_force_t0_sell_efficiency_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_flow_efficiency_index = get_adaptive_mtf_normalized_score(sell_flow_efficiency_index_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_order_book_clearing_rate = get_adaptive_mtf_normalized_score(sell_order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_sell_control_strength = get_adaptive_mtf_normalized_score(vwap_sell_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_vwap_down_guidance = get_adaptive_mtf_normalized_score(main_force_vwap_down_guidance_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_cross_down_intensity = get_adaptive_mtf_normalized_score(vwap_cross_down_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(wash_trade_sell_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)

        # 综合买方力量和效率
        total_buy_power = (
            norm_dip_buy_absorption_strength * micro_buy_power_weights.get('dip_buy_absorption_strength', 0.1) +
            norm_panic_buy_absorption_contribution * micro_buy_power_weights.get('panic_buy_absorption_contribution', 0.1) +
            norm_opening_buy_strength * micro_buy_power_weights.get('opening_buy_strength', 0.1) +
            norm_pre_closing_buy_posture * micro_buy_power_weights.get('pre_closing_buy_posture', 0.1) +
            norm_closing_auction_buy_ambush * micro_buy_power_weights.get('closing_auction_buy_ambush', 0.1) +
            norm_main_force_buy_ofi * micro_buy_power_weights.get('main_force_buy_ofi', 0.1) +
            norm_retail_buy_ofi * micro_buy_power_weights.get('retail_buy_ofi', 0.05) +
            norm_bid_side_liquidity * micro_buy_power_weights.get('bid_side_liquidity', 0.05) +
            norm_main_force_t0_buy_efficiency * micro_buy_power_weights.get('main_force_t0_buy_efficiency', 0.05) +
            norm_buy_flow_efficiency_index * micro_buy_power_weights.get('buy_flow_efficiency_index', 0.05) +
            norm_buy_order_book_clearing_rate * micro_buy_power_weights.get('buy_order_book_clearing_rate', 0.05) +
            norm_vwap_buy_control_strength * micro_buy_power_weights.get('vwap_buy_control_strength', 0.05) +
            norm_main_force_vwap_up_guidance * micro_buy_power_weights.get('main_force_vwap_up_guidance', 0.05) +
            norm_vwap_cross_up_intensity * micro_buy_power_weights.get('vwap_cross_up_intensity', 0.05) +
            norm_wash_trade_buy_volume * micro_buy_power_weights.get('wash_trade_buy_volume', -0.05)
        ).clip(0, 1)
        print(f"        -> [探针] 综合买方力量和效率 (total_buy_power): {total_buy_power.iloc[-1]:.4f}")

        # 综合卖方力量和效率
        total_sell_power = (
            norm_dip_sell_pressure_resistance * micro_sell_power_weights.get('dip_sell_pressure_resistance', 0.1) +
            norm_panic_sell_volume_contribution * micro_sell_power_weights.get('panic_sell_volume_contribution', 0.1) +
            norm_opening_sell_strength * micro_sell_power_weights.get('opening_sell_strength', 0.1) +
            norm_pre_closing_sell_posture * micro_sell_power_weights.get('pre_closing_sell_posture', 0.1) +
            norm_closing_auction_sell_ambush * micro_sell_power_weights.get('closing_auction_sell_ambush', 0.1) +
            norm_main_force_sell_ofi * micro_sell_power_weights.get('main_force_sell_ofi', 0.1) +
            norm_retail_sell_ofi * micro_sell_power_weights.get('retail_sell_ofi', 0.05) +
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
        print(f"        -> [探针] 综合卖方力量和效率 (total_sell_power): {total_sell_power.iloc[-1]:.4f}")

        # V5.1 综合买卖力量，形成更精细的微观控制力
        micro_control_score_v5_1 = (total_buy_power - total_sell_power).clip(-1, 1)
        print(f"        -> [探针] V5.1 微观控制力 (micro_control_score_v5_1): {micro_control_score_v5_1.iloc[-1]:.4f}")

        # V5.0 微观盘口意图推断：融合枯竭率
        # 枯竭率综合得分 (双极性)
        exhaustion_score = (norm_sell_exhaustion - norm_buy_exhaustion).clip(-1, 1)
        print(f"        -> [探针] 枯竭率综合得分 (exhaustion_score): {exhaustion_score.iloc[-1]:.4f}")

        # V5.0 非线性融合微观意图 (现在加入 V5.1 的精细化微观控制力)
        micro_intent_score = (
            imbalance_score * micro_intent_fusion_weights.get('imbalance', 0.4) +
            impact_score * micro_intent_fusion_weights.get('efficiency', 0.3) +
            exhaustion_score * micro_intent_fusion_weights.get('exhaustion', 0.3) +
            micro_control_score_v5_1 * 0.5 # [修改代码行] 增加 V5.1 的微观控制力，权重可调
        ).clip(-1, 1)
        print(f"        -> [探针] 融合后的微观意图得分 (micro_intent_score): {micro_intent_score.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 经过非对称增强的微观控制力 (micro_control_score): {micro_control_score.iloc[-1]:.4f}")

        # --- 3. 诡道博弈深度情境感知与调制 (Deceptive Game Integration & Contextual Modulation) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_mod_enabled:
            norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights_ff)
            norm_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(deception_context_modulator_raw, df_index, tf_weights=tf_weights_ff)
            sentiment_mod_factor = (1 + norm_market_sentiment.abs() * deception_context_sensitivity * np.sign(norm_market_sentiment))
            deception_modulator = deception_modulator * (1 - norm_wash_trade * wash_trade_penalty_sensitivity * sentiment_mod_factor.clip(0.5, 1.5))
            bull_trap_mask = (norm_deception > 0)
            deception_modulator.loc[bull_trap_mask] = deception_modulator.loc[bull_trap_mask] * (1 - norm_deception.loc[bull_trap_mask] * deception_penalty_sensitivity * sentiment_mod_factor.loc[bull_trap_mask].clip(0.5, 1.5))
            bear_trap_mitigation_mask = (norm_deception < 0) & (norm_conviction > conviction_threshold_deception) & (norm_flow_credibility > flow_credibility_threshold)
            deception_modulator.loc[bear_trap_mitigation_mask] = deception_modulator.loc[bear_trap_mitigation_mask] * (1 + norm_deception.loc[bear_trap_mitigation_mask].abs() * deception_penalty_sensitivity * 0.5 * sentiment_mod_factor.loc[bear_trap_mitigation_mask].clip(0.5, 1.5))
            deception_modulator = deception_modulator * (1 + (norm_flow_credibility - 0.5) * 0.5)
            low_credibility_mask = (norm_flow_credibility < flow_credibility_threshold)
            deception_modulator.loc[low_credibility_mask] = deception_modulator.loc[low_credibility_mask] * (norm_flow_credibility.loc[low_credibility_mask] / flow_credibility_threshold).clip(0.1, 1.0)
            deception_modulator = deception_modulator.clip(0.01, 2.0)
        print(f"        -> [探针] 诡道调制器 (deception_modulator): {deception_modulator.iloc[-1]:.4f}")

        # --- 4. 多维度情境自适应权重 (Adaptive Macro-Micro Weighting) ---
        dynamic_macro_weight = pd.Series(macro_flow_base_weight, index=df_index)
        dynamic_micro_weight = pd.Series(micro_control_base_weight, index=df_index)
        if dynamic_weight_mod_enabled:
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_flow_slope = get_adaptive_mtf_normalized_bipolar_score(flow_slope_raw, df_index, tf_weights=tf_weights_ff)
            norm_market_sentiment_dw = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
            mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) + \
                         (norm_flow_slope.abs() * dynamic_weight_sensitivity_flow_slope * np.sign(norm_flow_slope)) + \
                         (norm_market_sentiment_dw * dynamic_weight_sensitivity_sentiment)
            dynamic_macro_weight = dynamic_macro_weight * (1 + mod_factor)
            dynamic_micro_weight = dynamic_micro_weight * (1 - mod_factor)
            sum_dynamic_weights = dynamic_macro_weight + dynamic_micro_weight
            dynamic_macro_weight = dynamic_macro_weight / sum_dynamic_weights
            dynamic_micro_weight = dynamic_micro_weight / sum_dynamic_weights
            dynamic_macro_weight = dynamic_macro_weight.clip(0.1, 0.9)
            dynamic_micro_weight = dynamic_micro_weight.clip(0.1, 0.9)
        print(f"        -> [探针] 动态宏观权重 (dynamic_macro_weight): {dynamic_macro_weight.iloc[-1]:.4f}")
        print(f"        -> [探针] 动态微观权重 (dynamic_micro_weight): {dynamic_micro_weight.iloc[-1]:.4f}")

        # --- 5. 融合基础战场控制权 (V5.0 资金流结构与效率非线性建模) ---
        base_battlefield_control_score = np.tanh(
            flow_consensus_score * dynamic_macro_weight +
            micro_control_score * dynamic_micro_weight
        )
        base_battlefield_control_score = base_battlefield_control_score * deception_modulator
        print(f"        -> [探针] 基础战场控制权得分 (base_battlefield_control_score): {base_battlefield_control_score.iloc[-1]:.4f}")

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
        final_score = (
            (base_battlefield_control_score.add(1)/2).pow(dynamic_base_weight) *
            (norm_velocity.add(1)/2).pow(dynamic_velocity_weight) *
            (norm_acceleration.add(1)/2).pow(dynamic_acceleration_weight)
        ).pow(1 / (dynamic_base_weight + dynamic_velocity_weight + dynamic_acceleration_weight)) * 2 - 1
        print(f"        -> [探针] 最终战场控制权得分 (final_score): {final_score.iloc[-1]:.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.1 · 多维时空与自适应粒度版】资金流公理二：诊断“信念韧性”
        - 核心升级1: 核心信念强度：融合主力信念和聪明钱净买入的多时间框架斜率，深化资金流纯度，提升信念判断的跨周期稳健性。
        - 核心升级2: 诡道博弈韧性调制：引入欺骗指数和对倒强度的多时间框架斜率，并根据资金流可信度和市场流动性动态调整惩罚因子，同时引入核心信念强度进行反馈调制，更精微地评估诡道影响。
        - 核心升级3: 信念传导效率：融合主力执行Alpha和资金流效率的多时间框架斜率，引入日内VWAP偏离指数衡量执行成本，并结合大单压力与价格冲击形成“投入产出比”，全面评估效率。
        - 核心升级4: 动态情境自适应权重：扩展情境因子（如趋势活力），并根据波动性和流动性动态调整高频信号权重，实现粒度自适应的非线性权重调制。
        - 核心升级5: 信念演化趋势：对最终信念分数进行多周期平滑，并结合趋势活力评估惯性与转折预警。
        - 核心升级6: 新增资金指标整合：
            - 拉升卖出/买入弱点：rally_sell_distribution_intensity, rally_buy_support_weakness
            - 主力/散户OFII：main_force_buy_ofi, main_force_sell_ofi, retail_buy_ofi, retail_sell_ofi
            - 对倒买卖量：wash_trade_buy_volume, wash_trade_sell_volume
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“信念韧性 (V4.1 · 多维时空与自适应粒度版)”公理...")
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

        # [新增代码行] V4.1 新增核心信念强度权重
        core_conviction_weights_v4_1 = get_param_value(ac_params.get('core_conviction_weights_v4_1'), {
            'rally_sell_distribution_intensity': -0.1, 'rally_buy_support_weakness': -0.1,
            'main_force_buy_ofi': 0.1, 'main_force_sell_ofi': -0.1,
            'retail_buy_ofi': -0.05, 'retail_sell_ofi': 0.05,
            'wash_trade_buy_volume': -0.05, 'wash_trade_sell_volume': -0.05
        })

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
            dynamic_evolution_context_modulator_signal_1_name, dynamic_evolution_context_modulator_signal_2_name,
            # [新增代码行] V4.1 新增资金指标
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_conviction"):
            return pd.Series(0.0, index=df.index)

        # --- 原始数据获取 (用于探针和计算) ---
        print(f"        -> [探针] 正在获取原始数据...")
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

        # [新增代码行] V4.1 获取新增资金指标
        rally_sell_distribution_intensity_raw = self._get_safe_series(df, df, 'rally_sell_distribution_intensity_D', 0.0, method_name="_diagnose_axiom_conviction")
        rally_buy_support_weakness_raw = self._get_safe_series(df, df, 'rally_buy_support_weakness_D', 0.0, method_name="_diagnose_axiom_conviction")
        main_force_buy_ofi_raw = self._get_safe_series(df, df, 'main_force_buy_ofi_D', 0.0, method_name="_diagnose_axiom_conviction")
        main_force_sell_ofi_raw = self._get_safe_series(df, df, 'main_force_sell_ofi_D', 0.0, method_name="_diagnose_axiom_conviction")
        retail_buy_ofi_raw = self._get_safe_series(df, df, 'retail_buy_ofi_D', 0.0, method_name="_diagnose_axiom_conviction")
        retail_sell_ofi_raw = self._get_safe_series(df, df, 'retail_sell_ofi_D', 0.0, method_name="_diagnose_axiom_conviction")
        wash_trade_buy_volume_raw = self._get_safe_series(df, df, 'wash_trade_buy_volume_D', 0.0, method_name="_diagnose_axiom_conviction")
        wash_trade_sell_volume_raw = self._get_safe_series(df, df, 'wash_trade_sell_volume_D', 0.0, method_name="_diagnose_axiom_conviction")

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
        print(f"        -> [探针] 核心信念强度基础分 (core_conviction_score): {core_conviction_score.iloc[-1]:.4f}")

        # [新增代码行] V4.1 整合新增资金指标到核心信念强度
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
        print(f"        -> [探针] 整合新增指标后的核心信念强度 (core_conviction_score): {core_conviction_score.iloc[-1]:.4f}")

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
            sentiment_mod = (1 + norm_market_sentiment.abs() * resilience_context_sensitivity_sentiment * np.sign(norm_market_sentiment))
            cost_advantage_mod = (1 + norm_cost_advantage.abs() * resilience_context_sensitivity_cost_advantage * np.sign(norm_cost_advantage))
            liquidity_mod = (1 + (norm_market_liquidity - 0.5) * resilience_context_sensitivity_liquidity)
            conviction_feedback_mod = (1 - core_conviction_score.abs() * conviction_feedback_sensitivity * np.sign(core_conviction_score))
            deceptive_resilience_modulator = deceptive_resilience_modulator * (1 - norm_wash_trade_multi_tf * wash_trade_penalty_factor * sentiment_mod.clip(0.5, 1.5) * liquidity_mod.clip(0.5, 1.5) * conviction_feedback_mod.clip(0.5, 1.5))
            bull_trap_mask = (norm_deception_multi_tf > 0)
            deceptive_resilience_modulator.loc[bull_trap_mask] = deceptive_resilience_modulator.loc[bull_trap_mask] * (1 - norm_deception_multi_tf.loc[bull_trap_mask] * deception_penalty_factor * sentiment_mod.loc[bull_trap_mask].clip(0.5, 1.5) * liquidity_mod.loc[bull_trap_mask].clip(0.5, 1.5) * conviction_feedback_mod.loc[bull_trap_mask].clip(0.5, 1.5))
            bear_trap_resilience_mask = (norm_deception_multi_tf < 0) & (norm_cost_advantage > 0.5) & (norm_market_sentiment < -0.5) & (norm_market_liquidity < 0.5) & (core_conviction_score > 0.2)
            deceptive_resilience_modulator.loc[bear_trap_resilience_mask] = deceptive_resilience_modulator.loc[bear_trap_resilience_mask] * (1 + norm_deception_multi_tf.loc[bear_trap_resilience_mask].abs() * deception_penalty_factor * cost_advantage_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5) * (1 - liquidity_mod.loc[bear_trap_resilience_mask].clip(0.5, 1.5)))
            deceptive_resilience_modulator = deceptive_resilience_modulator.clip(0.01, 2.0)
        print(f"        -> [探针] 诡道博弈韧性调制器 (deceptive_resilience_modulator): {deceptive_resilience_modulator.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 信念传导效率得分 (transmission_efficiency_score): {transmission_efficiency_score.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 动态核心信念权重 (dynamic_core_conviction_weight): {dynamic_core_conviction_weight.iloc[-1]:.4f}")
        print(f"        -> [探针] 动态诡道韧性权重 (dynamic_deceptive_resilience_weight): {dynamic_deceptive_resilience_weight.iloc[-1]:.4f}")
        print(f"        -> [探针] 动态传导效率权重 (dynamic_transmission_efficiency_weight): {dynamic_transmission_efficiency_weight.iloc[-1]:.4f}")

        # --- 5. 融合基础信念分数 (V4.0 非线性建模) ---
        base_conviction_score = np.tanh(
            core_conviction_score * dynamic_core_conviction_weight * deceptive_resilience_modulator +
            transmission_efficiency_score * dynamic_transmission_efficiency_weight
        ).clip(-1, 1)
        print(f"        -> [探针] 基础信念分数 (base_conviction_score): {base_conviction_score.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 最终信念韧性得分 (final_score): {final_score.iloc[-1]:.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V6.1 · 深度情境与结构洞察版】资金流公理三：诊断“资金流纯度与动能”
        - 核心升级1: 基础动能深化：融合资金净流量和聪明钱净买入的多时间框架（含55日）速度与加速度，捕捉宏观动能。
        - 核心升级2: 诡道纯度精修：引入欺骗指数的多周期斜率和加速度，并根据资金流基尼系数和散户FOMO溢价指数动态调整对倒和欺骗惩罚。
        - 核心升级3: 环境感知增强：引入波动不稳定性、趋势活力和价格成交量熵作为环境调制因子。
        - 核心升级4: 结构洞察升级：融合主力资金流方向性、资金流基尼系数和散户资金主导指数，评估资金流质量。
        - 核心升级5: 动态融合优化：引入市场情绪和资金流可信度作为动态演化权重的额外情境因子。
        - 核心升级6: 新增资金指标整合：
            - 拉升卖出/买入弱点：rally_sell_distribution_intensity, rally_buy_support_weakness
            - 主力/散户OFII：main_force_buy_ofi, main_force_sell_ofi, retail_buy_ofi, retail_sell_ofi
            - 对倒买卖量：wash_trade_buy_volume, wash_trade_sell_volume
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资金流纯度与动能 (V6.1 · 深度情境与结构洞察版)”公理...")
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
        # [新增代码行] V6.1 新增结构性动能权重
        structural_momentum_weights_v6_1 = get_param_value(fm_params.get('structural_momentum_weights_v6_1'), {
            'rally_sell_distribution_intensity': -0.1, 'rally_buy_support_weakness': -0.1,
            'main_force_buy_ofi': 0.1, 'main_force_sell_ofi': -0.1,
            'retail_buy_ofi': -0.05, 'retail_sell_ofi': 0.05,
            'wash_trade_buy_volume': -0.05, 'wash_trade_sell_volume': -0.05
        })
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
            dynamic_evolution_context_modulator_signal_3_name, dynamic_evolution_context_modulator_signal_4_name,
            # [新增代码行] V6.1 新增资金指标
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_momentum"):
            return pd.Series(0.0, index=df.index)

        # --- 原始数据获取 (用于探针和计算) ---
        print(f"        -> [探针] 正在获取原始数据...")
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

        # [新增代码行] V6.1 获取新增资金指标
        rally_sell_distribution_intensity_raw = self._get_safe_series(df, df, 'rally_sell_distribution_intensity_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        rally_buy_support_weakness_raw = self._get_safe_series(df, df, 'rally_buy_support_weakness_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        main_force_buy_ofi_raw = self._get_safe_series(df, df, 'main_force_buy_ofi_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        main_force_sell_ofi_raw = self._get_safe_series(df, df, 'main_force_sell_ofi_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        retail_buy_ofi_raw = self._get_safe_series(df, df, 'retail_buy_ofi_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        retail_sell_ofi_raw = self._get_safe_series(df, df, 'retail_sell_ofi_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        wash_trade_buy_volume_raw = self._get_safe_series(df, df, 'wash_trade_buy_volume_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        wash_trade_sell_volume_raw = self._get_safe_series(df, df, 'wash_trade_sell_volume_D', 0.0, method_name="_diagnose_axiom_flow_momentum")

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
        print(f"        -> [探针] 基础动能得分 (base_momentum_score): {base_momentum_score.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 纯度调制器 (purity_modulator): {purity_modulator.iloc[-1]:.4f}")

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
            norm_price_volume_entropy = get_adaptive_mtf_normalized_score(price_volume_entropy_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
            level_mod = (1 + (norm_liquidity_supply - 0.5) * liquidity_mod_sensitivity_level)
            slope_mod = (1 + norm_liquidity_slope_multi_tf * liquidity_mod_sensitivity_slope)
            impact_mod = (1 + norm_liquidity_impact)
            volatility_mod = (1 - norm_volatility_instability * environment_mod_sensitivity_volatility)
            trend_mod = (1 + norm_trend_vitality * environment_mod_sensitivity_trend_vitality)
            entropy_mod = (1 + norm_price_volume_entropy * environment_mod_sensitivity_entropy)
            context_modulator = level_mod * slope_mod * impact_mod * volatility_mod * trend_mod * entropy_mod
            context_modulator = context_modulator.clip(0.5, 2.0)
        print(f"        -> [探针] 环境感知调制器 (context_modulator): {context_modulator.iloc[-1]:.4f}")

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
            large_order_flow_momentum * structural_momentum_weights.get('large_order_flow_slope_5', 0.2) +
            norm_main_force_flow_directionality * structural_momentum_weights.get('main_force_flow_directionality', 0.2) +
            norm_flow_quality * structural_momentum_weights.get('main_force_flow_gini', 0.15) +
            retail_flow_momentum * structural_momentum_weights.get('retail_flow_slope_5', -0.1) +
            norm_retail_dominance * structural_momentum_weights.get('retail_flow_dominance', -0.15)
        ).clip(-1, 1)
        print(f"        -> [探针] 结构性动能基础分 (structural_momentum_score): {structural_momentum_score.iloc[-1]:.4f}")

        # [新增代码行] V6.1 整合新增资金指标到结构性动能
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
        print(f"        -> [探针] 整合新增指标后的结构性动能 (structural_momentum_score): {structural_momentum_score.iloc[-1]:.4f}")

        # --- 5. 融合基础动能、纯度、环境和结构性动能 ---
        base_flow_momentum_score = (
            (base_momentum_score.add(1)/2).pow(0.3) *
            (purity_modulator.add(1)/2).pow(0.2) *
            (context_modulator.add(1)/2).pow(0.2) *
            (structural_momentum_score.add(1)/2).pow(0.3)
        ).pow(1 / (0.3 + 0.2 + 0.2 + 0.3)) * 2 - 1
        print(f"        -> [探针] 基础资金流纯度与动能得分 (base_flow_momentum_score): {base_flow_momentum_score.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 最终资金流纯度与动能得分 (final_score): {final_score.iloc[-1]:.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_capital_signature(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 意图博弈与结构演化版】资金流公理五：诊断“资本属性”
        - 核心升级：1) 耐心资本画像深化：引入隐蔽性与反侦察能力、成本结构与筹码分布、资金流结构韧性等维度。
                    2) 敏捷资本画像锐化：聚焦情绪驱动与风险偏好、博弈效率与资金利用率、短期爆发力与持续性等维度。
                    3) 资本间意图博弈分析：量化主力散户博弈强度、流动性争夺，通过动态博弈模型判断市场主导性。
                    4) 情境自适应动态权重与非线性融合：根据市场阶段、风险偏好等情境，动态调整各组件权重和融合方式。
                    5) 新增资金指标整合：
                        - 主力/散户OFII：main_force_buy_ofi, main_force_sell_ofi, retail_buy_ofi, retail_sell_ofi
                        - 对倒买卖量：wash_trade_buy_volume, wash_trade_sell_volume
                        - 拉升卖出/买入弱点：rally_sell_distribution_intensity, rally_buy_support_weakness
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资本属性 (V3.1 · 意图博弈与结构演化版)”公理...")
        df_index = df.index
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
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

        # [新增代码行] V3.1 新增耐心资本和敏捷资本的权重
        patient_capital_weights_v3_1 = get_param_value(acs_params.get('patient_capital_weights_v3_1'), {
            'main_force_buy_ofi': 0.1, 'main_force_sell_ofi': -0.1,
            'wash_trade_buy_volume': -0.05, 'wash_trade_sell_volume': -0.05
        })
        agile_capital_weights_v3_1 = get_param_value(acs_params.get('agile_capital_weights_v3_1'), {
            'rally_sell_distribution_intensity': -0.1, 'rally_buy_support_weakness': -0.1,
            'retail_buy_ofi': 0.1, 'retail_sell_ofi': -0.1
        })

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
            'SLOPE_5_covert_accumulation_signal_D', 'ACCEL_5_covert_accumulation_signal_D',
            'SLOPE_5_suppressive_accumulation_intensity_D', 'ACCEL_5_suppressive_accumulation_intensity_D',
            'SLOPE_5_deception_index_D', 'SLOPE_5_wash_trade_intensity_D', # 用于诡道共振
            'cost_structure_skewness_D', 'chip_fatigue_index_D', 'winner_loser_momentum_D',
            'structural_leverage_D', 'structural_node_count_D',
            'main_force_ofi_D',
            'SLOPE_5_main_force_ofi_D', 'ACCEL_5_main_force_ofi_D',
            'SLOPE_13_main_force_ofi_D', 'ACCEL_13_main_force_ofi_D',
            'SLOPE_21_main_force_ofi_D', 'ACCEL_21_main_force_ofi_D',
            'micro_price_impact_asymmetry_D', 'trade_count_D', 'main_force_activity_ratio_D',
            'main_force_flow_gini_D',
            'SLOPE_5_NMFNF_D', 'ACCEL_5_NMFNF_D', 'ACCEL_5_SLOPE_5_NMFNF_D', # 三阶导数
            'THEME_HOTNESS_SCORE_D',
            'SLOPE_5_retail_fomo_premium_index_D', 'SLOPE_5_retail_panic_surrender_index_D',
            'SLOPE_5_market_sentiment_score_D',
            'ACCEL_5_main_force_t0_efficiency_D', 'ACCEL_5_main_force_slippage_index_D', 'ACCEL_5_main_force_execution_alpha_D',
            'mf_retail_battle_intensity_D', 'mf_retail_liquidity_swap_corr_D',
            'VOLATILITY_INSTABILITY_INDEX_21d_D', 'order_book_liquidity_supply_D', 'order_book_clearing_rate_D',
            'market_sentiment_score_D', 'trend_vitality_index_D', 'strategic_phase_score_D', 'risk_reward_profile_D',
            # [新增代码行] V3.1 新增资金指标
            'rally_sell_distribution_intensity_D', 'rally_buy_support_weakness_D',
            'main_force_buy_ofi_D', 'main_force_sell_ofi_D',
            'retail_buy_ofi_D', 'retail_sell_ofi_D',
            'wash_trade_buy_volume_D', 'wash_trade_sell_volume_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_capital_signature"):
            return pd.Series(0.0, index=df.index)

        # --- 原始数据获取 (用于探针和计算) ---
        print(f"        -> [探针] 正在获取原始数据...")
        # 耐心资本相关
        main_force_cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        main_force_vwap_guidance_raw = self._get_safe_series(df, df, 'main_force_vwap_guidance_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        main_force_execution_alpha_raw = self._get_safe_series(df, df, 'main_force_execution_alpha_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        cost_structure_skewness_raw = self._get_safe_series(df, df, 'cost_structure_skewness_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        chip_fatigue_index_raw = self._get_safe_series(df, df, 'chip_fatigue_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        winner_loser_momentum_raw = self._get_safe_series(df, df, 'winner_loser_momentum_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        structural_leverage_raw = self._get_safe_series(df, df, 'structural_leverage_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        structural_node_count_raw = self._get_safe_series(df, df, 'structural_node_count_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        covert_accumulation_signal_raw = self._get_safe_series(df, df, 'covert_accumulation_signal_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        suppressive_accumulation_intensity_raw = self._get_safe_series(df, df, 'suppressive_accumulation_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        deception_index_raw = self._get_safe_series(df, df, 'deception_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")

        # 敏捷资本相关
        micro_price_impact_asymmetry_raw = self._get_safe_series(df, df, 'micro_price_impact_asymmetry_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        theme_hotness_raw = self._get_safe_series(df, df, 'THEME_HOTNESS_SCORE_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        retail_fomo_premium_index_raw = self._get_safe_series(df, df, 'retail_fomo_premium_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        retail_panic_surrender_index_raw = self._get_safe_series(df, df, 'retail_panic_surrender_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        main_force_t0_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_efficiency_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        main_force_slippage_index_raw = self._get_safe_series(df, df, 'main_force_slippage_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        nmfnf_raw = self._get_safe_series(df, df, 'NMFNF_D', 0.0, method_name="_diagnose_axiom_capital_signature")

        # 资本间意图博弈相关
        mf_retail_battle_intensity_raw = self._get_safe_series(df, df, 'mf_retail_battle_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        mf_retail_liquidity_swap_corr_raw = self._get_safe_series(df, df, 'mf_retail_liquidity_swap_corr_D', 0.0, method_name="_diagnose_axiom_capital_signature")

        # 情境调制相关
        volatility_instability_raw = self._get_safe_series(df, df, 'VOLATILITY_INSTABILITY_INDEX_21d_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        order_book_liquidity_supply_raw = self._get_safe_series(df, df, 'order_book_liquidity_supply_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        order_book_clearing_rate_raw = self._get_safe_series(df, df, 'order_book_clearing_rate_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        market_sentiment_raw = self._get_safe_series(df, df, 'market_sentiment_score_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        trend_vitality_raw = self._get_safe_series(df, df, 'trend_vitality_index_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        strategic_phase_raw = self._get_safe_series(df, df, 'strategic_phase_score_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        risk_reward_profile_raw = self._get_safe_series(df, df, 'risk_reward_profile_D', 0.0, method_name="_diagnose_axiom_capital_signature")

        # [新增代码行] V3.1 获取新增资金指标
        rally_sell_distribution_intensity_raw = self._get_safe_series(df, df, 'rally_sell_distribution_intensity_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        rally_buy_support_weakness_raw = self._get_safe_series(df, df, 'rally_buy_support_weakness_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        main_force_buy_ofi_raw = self._get_safe_series(df, df, 'main_force_buy_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        main_force_sell_ofi_raw = self._get_safe_series(df, df, 'main_force_sell_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        retail_buy_ofi_raw = self._get_safe_series(df, df, 'retail_buy_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        retail_sell_ofi_raw = self._get_safe_series(df, df, 'retail_sell_ofi_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        wash_trade_buy_volume_raw = self._get_safe_series(df, df, 'wash_trade_buy_volume_D', 0.0, method_name="_diagnose_axiom_capital_signature")
        wash_trade_sell_volume_raw = self._get_safe_series(df, df, 'wash_trade_sell_volume_D', 0.0, method_name="_diagnose_axiom_capital_signature")

        # --- 1. 耐心资本 (Patient Capital) - 意图博弈与结构演化 ---
        print(f"        -> [探针] 计算耐心资本画像...")
        # 1.1 多时间框架净流入持久性
        patient_flow_slope_weights_short = {str(p): 1/len(mtf_periods_patient_flow.get('short', [1])) for p in mtf_periods_patient_flow.get('short', [1])}
        patient_flow_accel_weights_short = {str(p): 1/len(mtf_periods_patient_flow.get('short', [1])) for p in mtf_periods_patient_flow.get('short', [1])}
        patient_flow_slope_weights_long = {str(p): 1/len(mtf_periods_patient_flow.get('long', [1])) for p in mtf_periods_patient_flow.get('long', [1])}
        norm_institutional_flow_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('short', []), patient_flow_slope_weights_short, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_institutional_flow_accel_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('short', []), patient_flow_accel_weights_short, True, True, method_name="_diagnose_axiom_capital_signature")
        norm_institutional_flow_long_slope_mtf = self._get_mtf_dynamic_score(df, 'net_lg_amount_calibrated_D', mtf_periods_patient_flow.get('long', []), patient_flow_slope_weights_long, True, False, method_name="_diagnose_axiom_capital_signature")
        flow_persistence = (
            norm_institutional_flow_slope_mtf * 0.4 +
            norm_institutional_flow_accel_mtf * 0.3 +
            norm_institutional_flow_long_slope_mtf * 0.3
        ).clip(-1, 1)
        print(f"            -> [探针] 流量持久性 (flow_persistence): {flow_persistence.iloc[-1]:.4f}")

        # 1.2 成本控制与效率
        norm_cost_advantage = get_adaptive_mtf_normalized_bipolar_score(main_force_cost_advantage_raw, df_index, tf_weights=tf_weights_ff)
        norm_vwap_guidance = get_adaptive_mtf_normalized_bipolar_score(main_force_vwap_guidance_raw, df_index, tf_weights=tf_weights_ff)
        norm_execution_alpha = get_adaptive_mtf_normalized_bipolar_score(main_force_execution_alpha_raw, df_index, tf_weights=tf_weights_ff)
        cost_efficiency = (norm_cost_advantage * 0.4 + norm_vwap_guidance * 0.3 + norm_execution_alpha * 0.3).clip(-1, 1)
        print(f"            -> [探针] 成本效率 (cost_efficiency): {cost_efficiency.iloc[-1]:.4f}")

        # 1.3 隐蔽性与反侦察能力 (V3.0 新增)
        covert_acc_slope_weights = {"5": 1.0}
        suppressive_acc_slope_weights = {"5": 1.0}
        deception_slope_weights = {"5": 1.0}
        wash_trade_slope_weights = {"5": 1.0}
        norm_covert_accumulation_slope = self._get_mtf_dynamic_score(df, 'covert_accumulation_signal_D', [5], covert_acc_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_suppressive_accumulation_slope = self._get_mtf_dynamic_score(df, 'suppressive_accumulation_intensity_D', [5], suppressive_acc_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_deception_slope = self._get_mtf_dynamic_score(df, 'deception_index_D', [5], deception_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_wash_trade_slope = self._get_mtf_dynamic_score(df, 'wash_trade_intensity_D', [5], wash_trade_slope_weights, False, False, method_name="_diagnose_axiom_capital_signature")
        deception_wash_trade_inverse = (1 - norm_deception_slope.abs() - norm_wash_trade_slope).clip(0, 1)
        covertness_anti_recon = (
            norm_covert_accumulation_slope * covertness_anti_recon_weights.get('covert_accumulation_slope', 0.4) +
            norm_suppressive_accumulation_slope * covertness_anti_recon_weights.get('suppressive_accumulation_slope', 0.3) +
            deception_wash_trade_inverse * covertness_anti_recon_weights.get('deception_wash_trade_inverse', 0.3)
        ).clip(0, 1)
        print(f"            -> [探针] 隐蔽性与反侦察能力 (covertness_anti_recon): {covertness_anti_recon.iloc[-1]:.4f}")

        # 1.4 成本结构与筹码分布 (V3.0 新增)
        norm_cost_structure_skewness = get_adaptive_mtf_normalized_bipolar_score(cost_structure_skewness_raw, df_index, tf_weights=tf_weights_ff)
        norm_chip_fatigue = get_adaptive_mtf_normalized_score(chip_fatigue_index_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_winner_loser_momentum = get_adaptive_mtf_normalized_bipolar_score(winner_loser_momentum_raw, df_index, tf_weights=tf_weights_ff)
        chip_structure_control = (
            norm_cost_structure_skewness * chip_structure_control_weights.get('cost_structure_skewness', 0.4) +
            norm_chip_fatigue * chip_structure_control_weights.get('chip_fatigue', 0.3) +
            norm_winner_loser_momentum * chip_structure_control_weights.get('winner_loser_momentum', 0.3)
        ).clip(-1, 1)
        print(f"            -> [探针] 筹码结构控制 (chip_structure_control): {chip_structure_control.iloc[-1]:.4f}")

        # 1.5 资金流结构韧性 (V3.0 新增)
        norm_structural_leverage = get_adaptive_mtf_normalized_score(structural_leverage_raw, df_index, ascending=False, tf_weights=tf_weights_ff)
        norm_structural_node_count = get_adaptive_mtf_normalized_score(structural_node_count_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        flow_structure_resilience = (norm_structural_leverage * 0.5 + norm_structural_node_count * 0.5).clip(0, 1)
        print(f"            -> [探针] 资金流结构韧性 (flow_structure_resilience): {flow_structure_resilience.iloc[-1]:.4f}")

        # 融合耐心资本得分
        patient_capital_score = (
            flow_persistence * patient_capital_weights.get('mtf_flow_persistence', 0.25) +
            cost_efficiency * patient_capital_weights.get('cost_efficiency', 0.2) +
            covertness_anti_recon * patient_capital_weights.get('covertness_anti_recon', 0.2) +
            chip_structure_control * patient_capital_weights.get('chip_structure_control', 0.2) +
            flow_structure_resilience * patient_capital_weights.get('flow_structure_resilience', 0.15)
        )
        # [新增代码行] V3.1 整合新增资金指标到耐心资本
        norm_main_force_buy_ofi = get_adaptive_mtf_normalized_score(main_force_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_main_force_sell_ofi = get_adaptive_mtf_normalized_score(main_force_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_buy_volume = get_adaptive_mtf_normalized_score(wash_trade_buy_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_wash_trade_sell_volume = get_adaptive_mtf_normalized_score(wash_trade_sell_volume_raw, df_index, ascending=True, tf_weights=tf_weights_ff)

        patient_capital_score = patient_capital_score + \
                                (norm_main_force_buy_ofi * patient_capital_weights_v3_1.get('main_force_buy_ofi', 0.1)) - \
                                (norm_main_force_sell_ofi * patient_capital_weights_v3_1.get('main_force_sell_ofi', -0.1)) - \
                                (norm_wash_trade_buy_volume * patient_capital_weights_v3_1.get('wash_trade_buy_volume', -0.05)) - \
                                (norm_wash_trade_sell_volume * patient_capital_weights_v3_1.get('wash_trade_sell_volume', -0.05))
        patient_capital_score = patient_capital_score.clip(-1, 1)
        print(f"            -> [探针] 整合新增指标后的耐心资本得分 (patient_capital_score): {patient_capital_score.iloc[-1]:.4f}")

        # --- 2. 敏捷资本 (Agile Capital) - 情绪驱动与博弈效率 ---
        print(f"        -> [探针] 计算敏捷资本画像...")
        # 2.1 高频冲击力与方向性
        agile_ofi_slope_weights_short = {str(p): 1/len(mtf_periods_agile_ofi.get('short', [1])) for p in mtf_periods_agile_ofi.get('short', [1])}
        agile_ofi_accel_weights_short = {str(p): 1/len(mtf_periods_agile_ofi.get('short', [1])) for p in mtf_periods_agile_ofi.get('short', [1])}
        norm_ofi_slope_mtf = self._get_mtf_dynamic_score(df, 'main_force_ofi_D', mtf_periods_agile_ofi.get('short', []), agile_ofi_slope_weights_short, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_ofi_accel_mtf = self._get_mtf_dynamic_score(df, 'main_force_ofi_D', mtf_periods_agile_ofi.get('short', []), agile_ofi_accel_weights_short, True, True, method_name="_diagnose_axiom_capital_signature")
        norm_price_impact_asymmetry = get_adaptive_mtf_normalized_bipolar_score(micro_price_impact_asymmetry_raw, df_index, tf_weights=tf_weights_ff)
        ofi_impact_directionality = (norm_ofi_slope_mtf * 0.4 + norm_ofi_accel_mtf * 0.3 + norm_price_impact_asymmetry * 0.3).clip(-1, 1)
        print(f"            -> [探针] OFI冲击力方向性 (ofi_impact_directionality): {ofi_impact_directionality.iloc[-1]:.4f}")

        # 2.2 情绪驱动与风险偏好 (V3.0 新增)
        retail_fomo_slope_weights = {"5": 1.0}
        retail_panic_slope_weights = {"5": 1.0}
        market_sentiment_slope_weights = {"5": 1.0}
        norm_retail_fomo_slope = self._get_mtf_dynamic_score(df, 'retail_fomo_premium_index_D', [5], retail_fomo_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_retail_panic_slope = self._get_mtf_dynamic_score(df, 'retail_panic_surrender_index_D', [5], retail_panic_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_market_sentiment_slope = self._get_mtf_dynamic_score(df, 'market_sentiment_score_D', [5], market_sentiment_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        emotion_driven_risk_appetite = (
            norm_retail_fomo_slope * emotion_driven_risk_appetite_weights.get('retail_fomo_slope', 0.4) +
            norm_retail_panic_slope * emotion_driven_risk_appetite_weights.get('retail_panic_slope', 0.3) +
            norm_market_sentiment_slope * emotion_driven_risk_appetite_weights.get('market_sentiment_slope', 0.3)
        ).clip(-1, 1)
        print(f"            -> [探针] 情绪驱动风险偏好 (emotion_driven_risk_appetite): {emotion_driven_risk_appetite.iloc[-1]:.4f}")

        # 2.3 博弈效率与资金利用率 (V3.0 新增)
        mf_t0_efficiency_accel_weights = {"5": 1.0}
        mf_slippage_accel_weights = {"5": 1.0}
        mf_exec_alpha_accel_weights = {"5": 1.0}
        norm_main_force_t0_efficiency_accel = self._get_mtf_dynamic_score(df, 'main_force_t0_efficiency_D', [5], mf_t0_efficiency_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature")
        norm_main_force_slippage_inverse_accel = self._get_mtf_dynamic_score(df, 'main_force_slippage_index_D', [5], mf_slippage_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature") * -1
        norm_main_force_execution_alpha_accel = self._get_mtf_dynamic_score(df, 'main_force_execution_alpha_D', [5], mf_exec_alpha_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature")
        game_efficiency_utilization = (
            norm_main_force_t0_efficiency_accel * game_efficiency_utilization_weights.get('main_force_t0_efficiency_accel', 0.4) +
            norm_main_force_slippage_inverse_accel * game_efficiency_utilization_weights.get('main_force_slippage_inverse_accel', 0.3) +
            norm_main_force_execution_alpha_accel * game_efficiency_utilization_weights.get('main_force_execution_alpha_accel', 0.3)
        ).clip(-1, 1)
        print(f"            -> [探针] 博弈效率与资金利用率 (game_efficiency_utilization): {game_efficiency_utilization.iloc[-1]:.4f}")

        # 2.4 短期爆发力与持续性 (V3.0 强化)
        nmfnf_slope_weights = {"5": 1.0}
        nmfnf_accel_weights = {"5": 1.0}
        nmfnf_accel_slope_weights = {"5": 1.0}
        norm_nmfnf_slope_5 = self._get_mtf_dynamic_score(df, 'NMFNF_D', [5], nmfnf_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        norm_nmfnf_accel_5 = self._get_mtf_dynamic_score(df, 'NMFNF_D', [5], nmfnf_accel_weights, True, True, method_name="_diagnose_axiom_capital_signature")
        norm_nmfnf_accel_slope_5 = self._get_mtf_dynamic_score(df, 'SLOPE_5_NMFNF_D', [5], nmfnf_accel_slope_weights, True, False, method_name="_diagnose_axiom_capital_signature")
        short_term_explosiveness = (norm_nmfnf_slope_5 * 0.5 + norm_nmfnf_accel_5 * 0.3 + norm_nmfnf_accel_slope_5 * 0.2).clip(-1, 1)
        print(f"            -> [探针] 短期爆发力 (short_term_explosiveness): {short_term_explosiveness.iloc[-1]:.4f}")

        # 2.5 资金流驱动的题材热度
        norm_theme_hotness = get_adaptive_mtf_normalized_score(theme_hotness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        print(f"            -> [探针] 题材热度 (norm_theme_hotness): {norm_theme_hotness.iloc[-1]:.4f}")

        # 融合敏捷资本得分
        agile_capital_score = (
            ofi_impact_directionality * agile_capital_weights.get('ofi_impact_directionality', 0.25) +
            emotion_driven_risk_appetite * agile_capital_weights.get('emotion_driven_risk_appetite', 0.2) +
            game_efficiency_utilization * agile_capital_weights.get('game_efficiency_utilization', 0.2) +
            short_term_explosiveness * agile_capital_weights.get('short_term_explosiveness', 0.2) +
            norm_theme_hotness * agile_capital_weights.get('theme_chasing', 0.15)
        )
        # [新增代码行] V3.1 整合新增资金指标到敏捷资本
        norm_rally_sell_distribution_intensity = get_adaptive_mtf_normalized_score(rally_sell_distribution_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_rally_buy_support_weakness = get_adaptive_mtf_normalized_score(rally_buy_support_weakness_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_buy_ofi = get_adaptive_mtf_normalized_score(retail_buy_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_retail_sell_ofi = get_adaptive_mtf_normalized_score(retail_sell_ofi_raw, df_index, ascending=True, tf_weights=tf_weights_ff)

        agile_capital_score = agile_capital_score - \
                              (norm_rally_sell_distribution_intensity * agile_capital_weights_v3_1.get('rally_sell_distribution_intensity', -0.1)) - \
                              (norm_rally_buy_support_weakness * agile_capital_weights_v3_1.get('rally_buy_support_weakness', -0.1)) + \
                              (norm_retail_buy_ofi * agile_capital_weights_v3_1.get('retail_buy_ofi', 0.1)) - \
                              (norm_retail_sell_ofi * agile_capital_weights_v3_1.get('retail_sell_ofi', -0.1))
        agile_capital_score = agile_capital_score.clip(-1, 1)
        print(f"            -> [探针] 整合新增指标后的敏捷资本得分 (agile_capital_score): {agile_capital_score.iloc[-1]:.4f}")

        # --- 3. 资本间意图博弈分析 (Inter-Capital Intent Game Analysis) (V3.0 新增) ---
        norm_mf_retail_battle_intensity = get_adaptive_mtf_normalized_score(mf_retail_battle_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_mf_retail_liquidity_swap_corr = get_adaptive_mtf_normalized_bipolar_score(mf_retail_liquidity_swap_corr_raw, df_index, tf_weights=tf_weights_ff)
        inter_capital_game_score = (
            norm_mf_retail_battle_intensity * inter_capital_game_weights.get('mf_retail_battle_intensity', 0.6) * np.sign(patient_capital_score - agile_capital_score) +
            norm_mf_retail_liquidity_swap_corr * inter_capital_game_weights.get('mf_retail_liquidity_swap_corr', 0.4)
        ).clip(-1, 1)
        print(f"        -> [探针] 资本间意图博弈得分 (inter_capital_game_score): {inter_capital_game_score.iloc[-1]:.4f}")

        # --- 4. 情境自适应动态权重与非线性融合 (Context-Adaptive Dynamic Weights & Non-linear Fusion) ---
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_liquidity_supply = get_adaptive_mtf_normalized_score(order_book_liquidity_supply_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_liquidity_clearing_rate = get_adaptive_mtf_normalized_score(order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, tf_weights=tf_weights_ff)
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
        print(f"        -> [探针] 动态耐心资本权重 (dynamic_patient_weight): {dynamic_patient_weight.iloc[-1]:.4f}")
        print(f"        -> [探针] 动态敏捷资本权重 (dynamic_agile_weight): {dynamic_agile_weight.iloc[-1]:.4f}")

        patient_modulated_score = patient_capital_score * dynamic_patient_weight * liquidity_mod * volatility_mod * sentiment_mod * (1 + inter_capital_game_score.clip(lower=0))
        agile_modulated_score = agile_capital_score * dynamic_agile_weight * liquidity_mod * volatility_mod * sentiment_mod * (1 + inter_capital_game_score.clip(upper=0).abs())
        capital_signature_score = np.tanh(patient_modulated_score - agile_modulated_score).pow(fusion_exponent).clip(-1, 1)
        print(f"        -> [探针] 最终资本属性得分 (capital_signature_score): {capital_signature_score.iloc[-1]:.4f}")
        return capital_signature_score.astype(np.float32)

    def _diagnose_axiom_flow_structure_health(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.1 · 新增】资金流公理六：诊断“资金流结构健康度”
        - 核心逻辑: 融合流量的平稳度、效率、成本凝聚力与结构风险，评估资金流模式的可持续性。
        - A股特性: 旨在区分“一日游”式的脉冲行情与具备坚实基础的、可持续的趋势。
        - 核心升级1: 流量效率增强：引入买卖流效率、订单簿清算率、VWAP控制强度等，更精细评估资金流效率。
        - 核心升级2: 结构风险过滤器增强：引入买卖方流动性，评估订单簿流动性风险。
        """
        print("    -> [资金流层] 正在诊断“资金流结构健康度 (V1.1)”公理...")
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        afsh_params = get_param_value(p_conf_ff.get('axiom_flow_structure_health_params'), {})

        # [新增代码行] V1.1 新增流量效率和结构风险权重
        flow_efficiency_weights = get_param_value(afsh_params.get('flow_efficiency_weights'), {
            'buy_flow_efficiency_index': 0.2, 'sell_flow_efficiency_index': -0.2,
            'buy_order_book_clearing_rate': 0.15, 'sell_order_book_clearing_rate': -0.15,
            'vwap_buy_control_strength': 0.15, 'vwap_sell_control_strength': -0.15
        })
        structural_risk_weights = get_param_value(afsh_params.get('structural_risk_weights'), {
            'bid_side_liquidity': -0.1, 'ask_side_liquidity': -0.1
        })

        required_signals = [
            'main_force_net_flow_calibrated_D', 'ATR_14_D',
            'main_force_vpoc_D', 'close_D', 'structural_leverage_D',
            # [新增代码行] V1.1 新增资金指标
            'buy_flow_efficiency_index_D', 'sell_flow_efficiency_index_D',
            'buy_order_book_clearing_rate_D', 'sell_order_book_clearing_rate_D',
            'vwap_buy_control_strength_D', 'vwap_sell_control_strength_D',
            'bid_side_liquidity_D', 'ask_side_liquidity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_structure_health"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index

        # --- 原始数据获取 (用于探针和计算) ---
        print(f"        -> [探针] 正在获取原始数据...")
        # 1. 流量平稳度 (Flow Steadiness)
        net_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        flow_volatility = net_flow.rolling(window=21).std().fillna(0)
        norm_flow_steadiness = 1 - get_adaptive_mtf_normalized_score(flow_volatility, df_index, ascending=True, tf_weights=tf_weights_ff)
        print(f"        -> [探针] 流量平稳度 (norm_flow_steadiness): {norm_flow_steadiness.iloc[-1]:.4f}")

        # 2. 流量效率 (Flow Efficiency)
        price_volatility = self._get_safe_series(df, df, 'ATR_14_D', 1.0, method_name="_diagnose_axiom_flow_structure_health").replace(0, 1e-9)
        flow_efficiency_raw = net_flow.rolling(window=21).mean() / price_volatility.rolling(window=21).mean()
        norm_flow_efficiency = get_adaptive_mtf_normalized_bipolar_score(flow_efficiency_raw, df_index, tf_weights_ff)
        print(f"        -> [探针] 基础流量效率 (norm_flow_efficiency): {norm_flow_efficiency.iloc[-1]:.4f}")

        # [新增代码行] V1.1 整合新增资金指标到流量效率
        norm_buy_flow_efficiency_index = get_adaptive_mtf_normalized_score(buy_flow_efficiency_index_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_flow_efficiency_index = get_adaptive_mtf_normalized_score(sell_flow_efficiency_index_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_buy_order_book_clearing_rate = get_adaptive_mtf_normalized_score(buy_order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_sell_order_book_clearing_rate = get_adaptive_mtf_normalized_score(sell_order_book_clearing_rate_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_buy_control_strength = get_adaptive_mtf_normalized_score(vwap_buy_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        norm_vwap_sell_control_strength = get_adaptive_mtf_normalized_score(vwap_sell_control_strength_raw, df_index, ascending=True, tf_weights=tf_weights_ff)

        enhanced_flow_efficiency = (
            norm_flow_efficiency +
            norm_buy_flow_efficiency_index * flow_efficiency_weights.get('buy_flow_efficiency_index', 0.2) -
            norm_sell_flow_efficiency_index * flow_efficiency_weights.get('sell_flow_efficiency_index', -0.2) +
            norm_buy_order_book_clearing_rate * flow_efficiency_weights.get('buy_order_book_clearing_rate', 0.15) -
            norm_sell_order_book_clearing_rate * flow_efficiency_weights.get('sell_order_book_clearing_rate', -0.15) +
            norm_vwap_buy_control_strength * flow_efficiency_weights.get('vwap_buy_control_strength', 0.15) -
            norm_vwap_sell_control_strength * flow_efficiency_weights.get('vwap_sell_control_strength', -0.15)
        ).clip(-1, 1)
        print(f"        -> [探针] 增强流量效率 (enhanced_flow_efficiency): {enhanced_flow_efficiency.iloc[-1]:.4f}")

        # 3. 成本凝聚力 (Cost Cohesion)
        vpoc = self._get_safe_series(df, df, 'main_force_vpoc_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        close = self._get_safe_series(df, df, 'close_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        cost_divergence = ((close - vpoc) / close).abs().fillna(0)
        norm_cost_cohesion = 1 - get_adaptive_mtf_normalized_score(cost_divergence, df_index, ascending=True, tf_weights=tf_weights_ff)
        print(f"        -> [探针] 成本凝聚力 (norm_cost_cohesion): {norm_cost_cohesion.iloc[-1]:.4f}")

        # 4. 结构风险过滤器 (Structural Risk Filter)
        structural_leverage = self._get_safe_series(df, df, 'structural_leverage_D', 0.0, method_name="_diagnose_axiom_flow_structure_health")
        risk_filter_base = 1 - get_adaptive_mtf_normalized_score(structural_leverage, df_index, ascending=True, tf_weights=tf_weights_ff)
        print(f"        -> [探针] 基础结构风险过滤器 (risk_filter_base): {risk_filter_base.iloc[-1]:.4f}")

        # [新增代码行] V1.1 整合新增资金指标到结构风险过滤器
        norm_bid_side_liquidity = get_adaptive_mtf_normalized_score(bid_side_liquidity_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 流动性越低风险越高
        norm_ask_side_liquidity = get_adaptive_mtf_normalized_score(ask_side_liquidity_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 流动性越低风险越高

        enhanced_risk_filter = risk_filter_base + \
                               (norm_bid_side_liquidity * structural_risk_weights.get('bid_side_liquidity', -0.1)) + \
                               (norm_ask_side_liquidity * structural_risk_weights.get('ask_side_liquidity', -0.1))
        enhanced_risk_filter = enhanced_risk_filter.clip(0, 1)
        print(f"        -> [探针] 增强结构风险过滤器 (enhanced_risk_filter): {enhanced_risk_filter.iloc[-1]:.4f}")

        # 5. 融合
        health_core = (norm_flow_steadiness * 0.4 + norm_cost_cohesion * 0.6)
        flow_structure_health_score = (enhanced_flow_efficiency * 0.5 + health_core * np.sign(enhanced_flow_efficiency) * 0.5) * enhanced_risk_filter
        print(f"        -> [探针] 最终资金流结构健康度得分 (flow_structure_health_score): {flow_structure_health_score.iloc[-1]:.4f}")
        return flow_structure_health_score.clip(-1, 1).astype(np.float32)

    def _calculate_mtf_cohesion_divergence(self, df: pd.DataFrame, signal_base_name: str, short_periods: List[int], long_periods: List[int], is_bipolar: bool, tf_weights: Dict) -> pd.Series:
        """
        【V4.0 升级 · 探针移除版】计算双极性多时间框架的共振/背离因子。
        分析短期和长期斜率/加速度的一致性及其方向。
        返回 -1 到 1 的分数，正值表示看涨共振，负值表示看跌共振。
        """
        method_name_str = "_calculate_mtf_cohesion_divergence" # 硬编码方法名
        short_slope_scores = []
        short_accel_scores = []
        long_slope_scores = []
        long_accel_scores = []
        # 获取短期斜率和加速度
        for p in short_periods:
            slope_col = f'SLOPE_{p}_{signal_base_name}'
            accel_col = f'ACCEL_{p}_{signal_base_name}'
            slope_raw = self._get_safe_series(df, df, slope_col, 0.0, method_name_str)
            accel_raw = self._get_safe_series(df, df, accel_col, 0.0, method_name_str)
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
            slope_raw = self._get_safe_series(df, df, slope_col, 0.0, method_name_str)
            accel_raw = self._get_safe_series(df, df, accel_col, 0.0, method_name_str)
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
        strength_cohesion_accel = (1 - (avg_short_accel.abs() - avg_long_accel.abs()).abs()).clip(0, 1)
        # 综合强度一致性
        strength_cohesion = (strength_cohesion_slope + strength_cohesion_accel) / 2
        # 3. 最终双极性共振分数：方向 * 强度
        mtf_resonance_score = strength_cohesion * direction_alignment
        return mtf_resonance_score.astype(np.float32)

    def _diagnose_fund_flow_divergence_signals(self, df: pd.DataFrame, norm_window: int, axiom_divergence: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        【V4.1 · 动态共振与微观意图精炼版】诊断资金流看涨/看跌背离信号。
        - 核心升级：1) 双极性多时间框架共振/背离因子：将MTF共振因子升级为-1到1的双极性分数，直接指示看涨/看跌共振或冲突。
                    2) 更全面的微观意图强度：整合更多微观资金流信号（如冲击弹性、T0效率）及其动态，构建更鲁棒的微观意图分数。
                    3) 看涨/看跌专属动态非线性指数：为看涨和看跌信号分别设置动态调整的非线性指数，实现更精细的放大/抑制。
                    4) 情境自适应纯度调制：根据市场情绪和资金流可信度动态调整纯度过滤的敏感度。
                    5) 新增资金指标整合：
                        - 盘口买卖力量：dip_buy_absorption_strength, panic_buy_absorption_contribution, opening_buy_strength, pre_closing_buy_posture, closing_auction_buy_ambush, bid_side_liquidity
                        - 盘口买卖效率：main_force_t0_buy_efficiency, buy_flow_efficiency_index, buy_order_book_clearing_rate, vwap_buy_control_strength, main_force_vwap_up_guidance, vwap_cross_up_intensity
                        - 盘口卖出力量：dip_sell_pressure_resistance, panic_sell_volume_contribution, opening_sell_strength, pre_closing_sell_posture, closing_auction_sell_ambush, ask_side_liquidity, main_force_on_peak_sell_flow
                        - 盘口卖出效率：main_force_t0_sell_efficiency, sell_flow_efficiency_index, sell_order_book_clearing_rate, vwap_sell_control_strength, main_force_vwap_down_guidance, vwap_cross_down_intensity
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“资金流看涨/看跌背离信号 (V4.1 · 动态共振与微观意图精炼版)”...")
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

        # [新增代码行] V4.1 新增微观意图强度权重
        micro_buy_power_weights = get_param_value(ffd_params.get('micro_buy_power_weights'), {
            'dip_buy_absorption_strength': 0.1, 'panic_buy_absorption_contribution': 0.1,
            'opening_buy_strength': 0.1, 'pre_closing_buy_posture': 0.1, 'closing_auction_buy_ambush': 0.1,
            'bid_side_liquidity': 0.05, 'main_force_t0_buy_efficiency': 0.05,
            'buy_flow_efficiency_index': 0.05, 'buy_order_book_clearing_rate': 0.05,
            'vwap_buy_control_strength': 0.05, 'main_force_vwap_up_guidance': 0.05,
            'vwap_cross_up_intensity': 0.05
        })
        micro_sell_power_weights = get_param_value(ffd_params.get('micro_sell_power_weights'), {
            'dip_sell_pressure_resistance': 0.1, 'panic_sell_volume_contribution': 0.1,
            'opening_sell_strength': 0.1, 'pre_closing_sell_posture': 0.1, 'closing_auction_sell_ambush': 0.1,
            'ask_side_liquidity': 0.05, 'main_force_on_peak_sell_flow': 0.05,
            'main_force_t0_sell_efficiency': 0.05, 'sell_flow_efficiency_index': 0.05,
            'sell_order_book_clearing_rate': 0.05, 'vwap_sell_control_strength': 0.05,
            'main_force_vwap_down_guidance': 0.05, 'vwap_cross_down_intensity': 0.05
        })

        # --- 信号依赖校验 ---
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
            # [新增代码行] V4.1 新增资金指标
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
            'bid_side_liquidity_D', 'ask_side_liquidity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_fund_flow_divergence_signals", atomic_states=self.strategy.atomic_states):
            return pd.Series(0.0, index=df.index), pd.Series(0.0, index=df.index)

        # --- 原始数据获取 (用于探针和计算) ---
        print(f"        -> [探针] 正在获取原始数据...")
        # 基础背离
        bullish_base_divergence, bearish_base_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        print(f"        -> [探针] 基础看涨背离 (bullish_base_divergence): {bullish_base_divergence.iloc[-1]:.4f}")
        print(f"        -> [探针] 基础看跌背离 (bearish_base_divergence): {bearish_base_divergence.iloc[-1]:.4f}")

        # 确认信号
        axiom_conviction = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_CONVICTION', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        axiom_flow_momentum = self._get_safe_series(df, self.strategy.atomic_states, 'SCORE_FF_AXIOM_FLOW_MOMENTUM', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        print(f"        -> [探针] 信念公理 (axiom_conviction): {axiom_conviction.iloc[-1]:.4f}")
        print(f"        -> [探针] 资金流纯度与动能公理 (axiom_flow_momentum): {axiom_flow_momentum.iloc[-1]:.4f}")

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

        # [新增代码行] V4.1 获取新增资金指标
        dip_buy_absorption_strength_raw = self._get_safe_series(df, df, 'dip_buy_absorption_strength_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        dip_sell_pressure_resistance_raw = self._get_safe_series(df, df, 'dip_sell_pressure_resistance_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        panic_sell_volume_contribution_raw = self._get_safe_series(df, df, 'panic_sell_volume_contribution_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        panic_buy_absorption_contribution_raw = self._get_safe_series(df, df, 'panic_buy_absorption_contribution_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        opening_buy_strength_raw = self._get_safe_series(df, df, 'opening_buy_strength_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        opening_sell_strength_raw = self._get_safe_series(df, df, 'opening_sell_strength_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        pre_closing_buy_posture_raw = self._get_safe_series(df, df, 'pre_closing_buy_posture_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        pre_closing_sell_posture_raw = self._get_safe_series(df, df, 'pre_closing_sell_posture_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        closing_auction_buy_ambush_raw = self._get_safe_series(df, df, 'closing_auction_buy_ambush_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        closing_auction_sell_ambush_raw = self._get_safe_series(df, df, 'closing_auction_sell_ambush_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        main_force_t0_buy_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_buy_efficiency_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        main_force_t0_sell_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_sell_efficiency_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        buy_flow_efficiency_index_raw = self._get_safe_series(df, df, 'buy_flow_efficiency_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        sell_flow_efficiency_index_raw = self._get_safe_series(df, df, 'sell_flow_efficiency_index_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        buy_order_book_clearing_rate_raw = self._get_safe_series(df, df, 'buy_order_book_clearing_rate_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        sell_order_book_clearing_rate_raw = self._get_safe_series(df, df, 'sell_order_book_clearing_rate_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        vwap_buy_control_strength_raw = self._get_safe_series(df, df, 'vwap_buy_control_strength_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        vwap_sell_control_strength_raw = self._get_safe_series(df, df, 'vwap_sell_control_strength_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        main_force_vwap_up_guidance_raw = self._get_safe_series(df, df, 'main_force_vwap_up_guidance_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        main_force_vwap_down_guidance_raw = self._get_safe_series(df, df, 'main_force_vwap_down_guidance_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        vwap_cross_up_intensity_raw = self._get_safe_series(df, df, 'vwap_cross_up_intensity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        vwap_cross_down_intensity_raw = self._get_safe_series(df, df, 'vwap_cross_down_intensity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        main_force_on_peak_sell_flow_raw = self._get_safe_series(df, df, 'main_force_on_peak_sell_flow_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        bid_side_liquidity_raw = self._get_safe_series(df, df, 'bid_side_liquidity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")
        ask_side_liquidity_raw = self._get_safe_series(df, df, 'ask_side_liquidity_D', 0.0, method_name="_diagnose_fund_flow_divergence_signals")

        # --- 1. 纯度过滤 (Purity Filter) ---
        norm_deception_slope_5 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_5_raw, df_index, self.tf_weights_ff)
        norm_deception_slope_13 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_13_raw, df_index, self.tf_weights_ff)
        norm_deception_slope_21 = get_adaptive_mtf_normalized_bipolar_score(deception_slope_21_raw, df_index, self.tf_weights_ff)
        norm_deception_multi_tf = (
            norm_deception_slope_5 * 0.5 +
            norm_deception_slope_13 * 0.3 +
            norm_deception_slope_21 * 0.2
        ).clip(-1, 1)
        print(f"        -> [探针] 欺骗指数多时间框架融合 (norm_deception_multi_tf): {norm_deception_multi_tf.iloc[-1]:.4f}")

        norm_wash_trade_slope_5 = get_adaptive_mtf_normalized_score(wash_trade_slope_5_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_slope_13 = get_adaptive_mtf_normalized_score(wash_trade_slope_13_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_slope_21 = get_adaptive_mtf_normalized_score(wash_trade_slope_21_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_wash_trade_multi_tf = (
            norm_wash_trade_slope_5 * 0.5 +
            norm_wash_trade_slope_13 * 0.3 +
            norm_wash_trade_slope_21 * 0.2
        ).clip(0, 1)
        print(f"        -> [探针] 对倒强度多时间框架融合 (norm_wash_trade_multi_tf): {norm_wash_trade_multi_tf.iloc[-1]:.4f}")

        norm_market_sentiment = get_adaptive_mtf_normalized_bipolar_score(market_sentiment_raw, df_index, self.tf_weights_ff)
        norm_flow_credibility_purity = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        purity_context_mod = (
            (1 + norm_market_sentiment.abs() * np.sign(norm_market_sentiment) * purity_context_mod_factors.get('market_sentiment', 0.5)) *
            (1 + (norm_flow_credibility_purity - 0.5) * purity_context_mod_factors.get('flow_credibility', 0.5))
        ).clip(0.5, 1.5)
        print(f"        -> [探针] 纯度情境调制器 (purity_context_mod): {purity_context_mod.iloc[-1]:.4f}")

        bullish_purity_modulator = (1 + norm_deception_multi_tf.clip(upper=0).abs() * purity_weights.get('deception_inverse', 0.5) * purity_context_mod) * (1 - norm_wash_trade_multi_tf * purity_weights.get('wash_trade_inverse', 0.5) * purity_context_mod)
        bearish_purity_modulator = (1 + norm_deception_multi_tf.clip(lower=0) * purity_weights.get('deception_inverse', 0.5) * purity_context_mod) * (1 - norm_wash_trade_multi_tf * purity_weights.get('wash_trade_inverse', 0.5) * purity_context_mod)
        print(f"        -> [探针] 看涨纯度调制器 (bullish_purity_modulator): {bullish_purity_modulator.iloc[-1]:.4f}")
        print(f"        -> [探针] 看跌纯度调制器 (bearish_purity_modulator): {bearish_purity_modulator.iloc[-1]:.4f}")

        # --- 2. 意图确认 (Intent Confirmation) ---
        bullish_conviction_confirm = axiom_conviction.clip(lower=0) * confirmation_weights.get('conviction_positive', 0.5)
        bearish_conviction_confirm = axiom_conviction.clip(upper=0).abs() * confirmation_weights.get('conviction_positive', 0.5)
        bullish_flow_momentum_confirm = axiom_flow_momentum.clip(lower=0) * confirmation_weights.get('flow_momentum_positive', 0.5)
        bearish_flow_momentum_confirm = axiom_flow_momentum.clip(upper=0).abs() * confirmation_weights.get('flow_momentum_positive', 0.5)
        bullish_confirmation_score = (bullish_conviction_confirm + bullish_flow_momentum_confirm).clip(0, 1)
        bearish_confirmation_score = (bearish_conviction_confirm + bearish_flow_momentum_confirm).clip(0, 1)
        print(f"        -> [探针] 看涨确认得分 (bullish_confirmation_score): {bullish_confirmation_score.iloc[-1]:.4f}")
        print(f"        -> [探针] 看跌确认得分 (bearish_confirmation_score): {bearish_confirmation_score.iloc[-1]:.4f}")

        # --- 3. 情境校准 (Contextual Calibration) ---
        norm_strategic_posture = get_adaptive_mtf_normalized_bipolar_score(strategic_posture, df_index, self.tf_weights_ff)
        bullish_posture_mod = norm_strategic_posture.clip(lower=0) * context_modulator_weights.get('strategic_posture', 0.4)
        bearish_posture_mod = norm_strategic_posture.clip(upper=0).abs() * context_modulator_weights.get('strategic_posture', 0.4)
        norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        credibility_mod = norm_flow_credibility * context_modulator_weights.get('flow_credibility', 0.3)
        norm_retail_panic = get_adaptive_mtf_normalized_score(retail_panic_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_retail_fomo = get_adaptive_mtf_normalized_score(retail_fomo_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        bullish_retail_context_mod = norm_retail_panic * retail_panic_fomo_sensitivity * context_modulator_weights.get('retail_panic_fomo_context', 0.3)
        bearish_retail_context_mod = norm_retail_fomo * retail_panic_fomo_sensitivity * context_modulator_weights.get('retail_panic_fomo_context', 0.3)
        bullish_context_modulator = (1 + bullish_posture_mod + credibility_mod + bullish_retail_context_mod).clip(0.5, 2.0)
        bearish_context_modulator = (1 + bearish_posture_mod + credibility_mod + bearish_retail_context_mod).clip(0.5, 2.0)
        print(f"        -> [探针] 看涨情境调制器 (bullish_context_modulator): {bullish_context_modulator.iloc[-1]:.4f}")
        print(f"        -> [探针] 看跌情境调制器 (bearish_context_modulator): {bearish_context_modulator.iloc[-1]:.4f}")

        # --- V4.0 升级: 双极性多时间框架共振/背离因子 (Bipolar MTF Resonance/Divergence Factor) ---
        short_periods = [5, 13]
        long_periods = [21, 55]
        nmfnf_bipolar_resonance_factor = self._calculate_mtf_cohesion_divergence(df, 'NMFNF_D', short_periods, long_periods, True, self.tf_weights_ff)
        conviction_bipolar_resonance_factor = self._calculate_mtf_cohesion_divergence(df, 'main_force_conviction_index_D', short_periods, long_periods, True, self.tf_weights_ff)
        mtf_bipolar_resonance_factor = (
            nmfnf_bipolar_resonance_factor * mtf_resonance_factor_weights.get('nmfnf_cohesion', 0.6) +
            conviction_bipolar_resonance_factor * mtf_resonance_factor_weights.get('conviction_cohesion', 0.4)
        ).clip(-1, 1)
        print(f"        -> [探针] MTF双极性共振因子 (mtf_bipolar_resonance_factor): {mtf_bipolar_resonance_factor.iloc[-1]:.4f}")

        # --- V4.0 升级: 更全面的微观意图强度 (Comprehensive Micro-Intent Strength) ---
        norm_order_book_imbalance = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, self.tf_weights_ff)
        norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=self.tf_weights_ff)
        norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_micro_impact_elasticity = get_adaptive_mtf_normalized_bipolar_score(micro_impact_elasticity_raw, df_index, self.tf_weights_ff)
        norm_main_force_t0_efficiency = get_adaptive_mtf_normalized_bipolar_score(main_force_t0_efficiency_raw, df_index, self.tf_weights_ff)

        # [新增代码行] V4.1 整合新增资金指标到微观意图强度
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
        print(f"        -> [探针] 总微观买方强度 (total_micro_buy_strength): {total_micro_buy_strength.iloc[-1]:.4f}")

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
        print(f"        -> [探针] 总微观卖方强度 (total_micro_sell_strength): {total_micro_sell_strength.iloc[-1]:.4f}")

        micro_intent_strength = (
            norm_order_book_imbalance * micro_intent_signals_weights.get('order_book_imbalance', 0.3) +
            (norm_sell_exhaustion - norm_buy_exhaustion) * micro_intent_signals_weights.get('exhaustion_rate', 0.3) +
            norm_micro_impact_elasticity * micro_intent_signals_weights.get('micro_impact_elasticity', 0.2) +
            norm_main_force_t0_efficiency * micro_intent_signals_weights.get('main_force_t0_efficiency', 0.2) +
            (total_micro_buy_strength - total_micro_sell_strength) * 0.5 # [修改代码行] 整合 V4.1 的买卖力量
        ).clip(-1, 1)
        print(f"        -> [探针] 综合微观意图强度 (micro_intent_strength): {micro_intent_strength.iloc[-1]:.4f}")

        micro_macro_divergence_factor = (
            micro_intent_strength * micro_macro_divergence_weights.get('micro_intent_strength', 0.5) -
            axiom_flow_momentum * micro_macro_divergence_weights.get('macro_flow_momentum', 0.5)
        ).clip(-1, 1)
        print(f"        -> [探针] 微观-宏观背离因子 (micro_macro_divergence_factor): {micro_macro_divergence_factor.iloc[-1]:.4f}")

        bullish_micro_macro_mod = (1 + micro_macro_divergence_factor.clip(lower=0))
        bearish_micro_macro_mod = (1 + micro_macro_divergence_factor.clip(upper=0).abs())
        print(f"        -> [探针] 看涨微观-宏观调制 (bullish_micro_macro_mod): {bullish_micro_macro_mod.iloc[-1]:.4f}")
        print(f"        -> [探针] 看跌微观-宏观调制 (bearish_micro_macro_mod): {bearish_micro_macro_mod.iloc[-1]:.4f}")

        # --- V4.0 升级: 看涨/看跌专属动态非线性指数 (Bullish/Bearish Adaptive Non-linear Exponents) ---
        norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
        norm_trend_vitality = get_adaptive_mtf_normalized_score(trend_vitality_raw, df_index, ascending=True, tf_weights=self.tf_weights_ff)
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
        print(f"        -> [探针] 看涨动态非线性指数 (bullish_dynamic_non_linear_exponent): {bullish_dynamic_non_linear_exponent.iloc[-1]:.4f}")
        print(f"        -> [探针] 看跌动态非线性指数 (bearish_dynamic_non_linear_exponent): {bearish_dynamic_non_linear_exponent.iloc[-1]:.4f}")

        # --- 4. 非线性融合 (Non-linear Fusion) ---
        bullish_divergence_score = (
            bullish_base_divergence *
            (bullish_purity_modulator * purity_context_mod).clip(0.1, 2.0) *
            (bullish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0) *
            (bullish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0) *
            (1 + mtf_bipolar_resonance_factor.clip(lower=0)) *
            bullish_micro_macro_mod
        ).pow(bullish_dynamic_non_linear_exponent).clip(0, 1)
        bearish_divergence_score = (
            bearish_base_divergence *
            (bearish_purity_modulator * purity_context_mod).clip(0.1, 2.0) *
            (bearish_confirmation_score * (1 + norm_volatility_instability * dynamic_context_modulator_sensitivity.get('volatility_instability', 0.2))).clip(0.1, 2.0) *
            (bearish_context_modulator * (1 + norm_flow_credibility * dynamic_context_modulator_sensitivity.get('flow_credibility', 0.15))).clip(0.1, 2.0) *
            (1 + mtf_bipolar_resonance_factor.clip(upper=0).abs()) *
            bearish_micro_macro_mod
        ).pow(bearish_dynamic_non_linear_exponent).clip(0, 1)
        bullish_divergence_score = bullish_divergence_score.where(bullish_divergence_score > bullish_divergence_threshold, 0.0)
        bearish_divergence_score = bearish_divergence_score.where(bearish_divergence_score > bearish_divergence_threshold, 0.0)
        print(f"        -> [探针] 最终看涨背离得分 (bullish_divergence_score): {bullish_divergence_score.iloc[-1]:.4f}")
        print(f"        -> [探针] 最终看跌背离得分 (bearish_divergence_score): {bearish_divergence_score.iloc[-1]:.4f}")
        return bullish_divergence_score.astype(np.float32), bearish_divergence_score.astype(np.float32)











