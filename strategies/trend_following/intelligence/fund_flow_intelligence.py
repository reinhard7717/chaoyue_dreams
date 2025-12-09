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
        axiom_conviction = self._diagnose_axiom_conviction(
            df, norm_window,
            capital_signature_score=axiom_capital_signature,
            flow_health_score=axiom_flow_structure_health
        )
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
        【V2.0 · 价资张力版】资金流公理四：诊断“价资张力”
        - 核心逻辑: 基于弹性势能模型，融合分歧向量、持续性与能量注入，量化积蓄的反转势能。
        - A股特性: 背离的威力不在于一瞬间，而在于持续的、耗费巨资的“拔河”。此模型旨在识别这种高势能状态。
        """
        print("    -> [资金流层] 正在诊断“价资张力”公理...")
        required_signals = ['SLOPE_5_close_D', 'SLOPE_5_NMFNF_D', 'volume_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. 分歧向量
        price_trend = self._get_safe_series(df, df, 'SLOPE_5_close_D', 0.0, method_name="_diagnose_axiom_divergence")
        flow_trend = self._get_safe_series(df, df, 'SLOPE_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_price_trend = get_adaptive_mtf_normalized_bipolar_score(price_trend, df_index, tf_weights_ff)
        norm_flow_trend = get_adaptive_mtf_normalized_bipolar_score(flow_trend, df_index, tf_weights_ff)
        disagreement_vector = (norm_flow_trend - norm_price_trend).clip(-2, 2) / 2
        # 2. 张力强度 (持续性 * 能量注入)
        persistence = disagreement_vector.rolling(window=13, min_periods=5).std().fillna(0)
        norm_persistence = get_adaptive_mtf_normalized_score(persistence, df_index, ascending=True, tf_weights=tf_weights_ff)
        volume = self._get_safe_series(df, df, 'volume_D', 0.0, method_name="_diagnose_axiom_divergence")
        norm_volume = get_adaptive_mtf_normalized_score(volume, df_index, ascending=True, tf_weights=tf_weights_ff)
        energy_injection = norm_volume * disagreement_vector.abs()
        tension_magnitude = (norm_persistence * energy_injection).pow(0.5)
        # 3. 融合
        tension_score = disagreement_vector * (1 + tension_magnitude * 1.5) # 1.5是放大系数
        return tension_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.0 · 诡道动态博弈版】资金流公理一：诊断“战场控制权”
        - 核心升级1: 诡道博弈深度融合与情境调制：引入欺骗指数、主力信念和资金流可信度，对对倒强度进行非对称调制，更精准识别和应对主力通过诡道手段对盘面控制权的干扰。
        - 核心升级2: 宏观与微观动态权重自适应：根据波动不稳定性、资金流短期趋势等情境因子，动态调整宏观资金流向与微观盘口控制力的融合权重，使信号自适应市场动态。
        - 核心升级3: 微观盘口控制力非对称增强：引入买卖盘口枯竭率，更精细地捕捉盘口买卖力量的真实对比和效率，提高微观控制力的判别准确性。
        - 核心升级4: 战场控制权动态演化：对战场控制权分数进行平滑处理，并计算其速度和加速度，将动态信息融入最终分数，增强信号的前瞻性。
        - 探针增强: 详细输出所有原始数据、关键计算节点、结果的值，以便于检查和调试。
        """
        print("    -> [资金流层] 正在诊断“战场控制权 (V4.0 · 诡道动态博弈版)”公理...")
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
                print(f"    -> [战场控制权探针] @ {probe_date.date()}:")
        # --- 参数加载 ---
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        ac_params = get_param_value(p_conf_ff.get('axiom_consensus_params'), {})
        # V4.0 诡道博弈深度融合参数
        deception_mod_enabled = get_param_value(ac_params.get('deception_mod_enabled'), True)
        deception_penalty_sensitivity = get_param_value(ac_params.get('deception_penalty_sensitivity'), 0.6)
        wash_trade_penalty_sensitivity = get_param_value(ac_params.get('wash_trade_penalty_sensitivity'), 0.4)
        conviction_threshold_deception = get_param_value(ac_params.get('conviction_threshold_deception'), 0.2)
        flow_credibility_threshold = get_param_value(ac_params.get('flow_credibility_threshold'), 0.5)
        # V4.0 宏观与微观动态权重自适应参数
        dynamic_weight_mod_enabled = get_param_value(ac_params.get('dynamic_weight_mod_enabled'), True)
        macro_flow_base_weight = get_param_value(ac_params.get('macro_flow_base_weight'), 0.4)
        micro_control_base_weight = get_param_value(ac_params.get('micro_control_base_weight'), 0.6)
        dynamic_weight_modulator_signal_1_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_1'), 'VOLATILITY_INSTABILITY_INDEX_21d_D')
        dynamic_weight_modulator_signal_2_name = get_param_value(ac_params.get('dynamic_weight_modulator_signal_2'), 'SLOPE_5_NMFNF_D')
        dynamic_weight_sensitivity_volatility = get_param_value(ac_params.get('dynamic_weight_sensitivity_volatility'), 0.3)
        dynamic_weight_sensitivity_flow_slope = get_param_value(ac_params.get('dynamic_weight_sensitivity_flow_slope'), 0.2)
        # V4.0 微观盘口控制力非对称增强参数
        asymmetric_micro_control_enabled = get_param_value(ac_params.get('asymmetric_micro_control_enabled'), True)
        exhaustion_boost_factor = get_param_value(ac_params.get('exhaustion_boost_factor'), 0.2)
        exhaustion_penalty_factor = get_param_value(ac_params.get('exhaustion_penalty_factor'), 0.3)
        # V4.0 战场控制权动态演化参数
        smoothing_ema_span = get_param_value(ac_params.get('smoothing_ema_span'), 5)
        dynamic_evolution_base_weights = get_param_value(ac_params.get('dynamic_evolution_base_weights'), {'base_score': 0.6, 'velocity': 0.2, 'acceleration': 0.2})
        # --- 信号依赖校验 ---
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'order_book_imbalance_D', 'microstructure_efficiency_index_D', 'wash_trade_intensity_D',
            'deception_index_D', 'main_force_conviction_index_D', 'flow_credibility_index_D', # V4.0 诡道依赖
            dynamic_weight_modulator_signal_1_name, dynamic_weight_modulator_signal_2_name, # V4.0 动态权重依赖
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D' # V4.0 微观增强依赖
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
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_consensus")
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_consensus")
        if is_probe_active:
            print(f"       - 原料: main_force_net_flow_calibrated_D (raw): {main_force_flow_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: retail_net_flow_calibrated_D (raw): {retail_flow_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: order_book_imbalance_D (raw): {order_book_imbalance_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: microstructure_efficiency_index_D (raw): {ofi_impact_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: wash_trade_intensity_D (raw): {wash_trade_intensity_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: deception_index_D (raw): {deception_index_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: main_force_conviction_index_D (raw): {main_force_conviction_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: flow_credibility_index_D (raw): {flow_credibility_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: {dynamic_weight_modulator_signal_1_name} (raw): {volatility_instability_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: {dynamic_weight_modulator_signal_2_name} (raw): {flow_slope_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: buy_quote_exhaustion_rate_D (raw): {buy_exhaustion_raw.loc[probe_date]:.4f}")
            print(f"       - 原料: sell_quote_exhaustion_rate_D (raw): {sell_exhaustion_raw.loc[probe_date]:.4f}")
        # --- 1. 宏观资金流向 (Macro Fund Flow) ---
        flow_consensus_score = get_adaptive_mtf_normalized_bipolar_score(main_force_flow_raw - retail_flow_raw, df_index, tf_weights_ff)
        if is_probe_active:
            print(f"       - 过程: flow_consensus_score: {flow_consensus_score.loc[probe_date]:.4f}")
        # --- 2. 微观盘口控制力 (Micro Order Book Control) ---
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff)
        impact_score = get_adaptive_mtf_normalized_bipolar_score(ofi_impact_raw, df_index, tf_weights_ff)
        # 基础微观控制分
        micro_control_base_score = (imbalance_score.abs() * impact_score.abs()).pow(0.5) * np.sign(imbalance_score)
        # V4.0 微观盘口控制力非对称增强
        micro_control_modulator = pd.Series(1.0, index=df_index)
        if asymmetric_micro_control_enabled:
            norm_buy_exhaustion = get_adaptive_mtf_normalized_score(buy_exhaustion_raw, df_index, ascending=False, tf_weights=tf_weights_ff) # 枯竭率越低越好
            norm_sell_exhaustion = get_adaptive_mtf_normalized_score(sell_exhaustion_raw, df_index, ascending=True, tf_weights=tf_weights_ff) # 枯竭率越高越好
            # 买盘枯竭低 & 卖盘枯竭高 -> 增强微观控制力 (买方强势)
            boost_mask = (norm_buy_exhaustion > 0.5) & (norm_sell_exhaustion > 0.5)
            micro_control_modulator.loc[boost_mask] = 1 + (norm_buy_exhaustion.loc[boost_mask] * norm_sell_exhaustion.loc[boost_mask]) * exhaustion_boost_factor
            # 买盘枯竭高 & 卖盘枯竭低 -> 惩罚微观控制力 (卖方强势)
            penalty_mask = (norm_buy_exhaustion < 0.5) & (norm_sell_exhaustion < 0.5)
            micro_control_modulator.loc[penalty_mask] = 1 - (norm_buy_exhaustion.loc[penalty_mask] * norm_sell_exhaustion.loc[penalty_mask]) * exhaustion_penalty_factor
            micro_control_modulator = micro_control_modulator.clip(0.5, 1.5)
        micro_control_score = micro_control_base_score * micro_control_modulator
        if is_probe_active:
            print(f"       - 过程: imbalance_score: {imbalance_score.loc[probe_date]:.4f}")
            print(f"       - 过程: impact_score: {impact_score.loc[probe_date]:.4f}")
            print(f"       - 过程: micro_control_base_score: {micro_control_base_score.loc[probe_date]:.4f}")
            if asymmetric_micro_control_enabled:
                print(f"       - 过程: norm_buy_exhaustion: {norm_buy_exhaustion.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_sell_exhaustion: {norm_sell_exhaustion.loc[probe_date]:.4f}")
                print(f"       - 过程: micro_control_modulator: {micro_control_modulator.loc[probe_date]:.4f}")
            print(f"       - 过程: micro_control_score: {micro_control_score.loc[probe_date]:.4f}")
        # --- 3. 诡道博弈深度融合与情境调制 (Deceptive Game Integration & Contextual Modulation) ---
        deception_modulator = pd.Series(1.0, index=df_index)
        if deception_mod_enabled:
            norm_wash_trade = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_deception = get_adaptive_mtf_normalized_bipolar_score(deception_index_raw, df_index, tf_weights=tf_weights_ff)
            norm_conviction = get_adaptive_mtf_normalized_bipolar_score(main_force_conviction_raw, df_index, tf_weights=tf_weights_ff)
            norm_flow_credibility = get_adaptive_mtf_normalized_score(flow_credibility_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            # 基础惩罚：对倒强度
            deception_modulator = deception_modulator * (1 - norm_wash_trade * wash_trade_penalty_sensitivity)
            # 欺骗指数调制
            # 正向欺骗 (诱多) 惩罚控制权
            bull_trap_mask = (norm_deception > 0)
            deception_modulator.loc[bull_trap_mask] = deception_modulator.loc[bull_trap_mask] * (1 - norm_deception.loc[bull_trap_mask] * deception_penalty_sensitivity)
            # 负向欺骗 (诱空) 在主力信念强且可信度高时，可能为洗盘，缓解惩罚或增强
            bear_trap_mitigation_mask = (norm_deception < 0) & (norm_conviction > conviction_threshold_deception) & (norm_flow_credibility > flow_credibility_threshold)
            deception_modulator.loc[bear_trap_mitigation_mask] = deception_modulator.loc[bear_trap_mitigation_mask] * (1 + norm_deception.loc[bear_trap_mitigation_mask].abs() * deception_penalty_sensitivity * 0.5) # 缓解一半惩罚
            # 全局可信度校准
            deception_modulator = deception_modulator * (1 + (norm_flow_credibility - 0.5) * 0.5) # 可信度越高，诡道调制越有效
            deception_modulator = deception_modulator.clip(0.1, 2.0) # 限制调制范围
        if is_probe_active:
            if deception_mod_enabled:
                print(f"       - 过程: norm_wash_trade: {norm_wash_trade.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_deception: {norm_deception.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_conviction: {norm_conviction.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_flow_credibility: {norm_flow_credibility.loc[probe_date]:.4f}")
            print(f"       - 过程: deception_modulator: {deception_modulator.loc[probe_date]:.4f}")
        # --- 4. 宏观与微观动态权重自适应 (Adaptive Macro-Micro Weighting) ---
        dynamic_macro_weight = pd.Series(macro_flow_base_weight, index=df_index)
        dynamic_micro_weight = pd.Series(micro_control_base_weight, index=df_index)
        if dynamic_weight_mod_enabled:
            norm_volatility_instability = get_adaptive_mtf_normalized_score(volatility_instability_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
            norm_flow_slope = get_adaptive_mtf_normalized_bipolar_score(flow_slope_raw, df_index, tf_weights=tf_weights_ff)
            # 波动性高或资金流趋势不明确时，增加微观权重，降低宏观权重
            # 波动性低或资金流趋势明确时，增加宏观权重，降低微观权重
            mod_factor = (norm_volatility_instability * dynamic_weight_sensitivity_volatility) + \
                         (norm_flow_slope.abs() * dynamic_weight_sensitivity_flow_slope * np.sign(norm_flow_slope)) # 资金流趋势越强，宏观权重越高
            dynamic_macro_weight = dynamic_macro_weight * (1 + mod_factor)
            dynamic_micro_weight = dynamic_micro_weight * (1 - mod_factor)
            # 归一化动态权重
            sum_dynamic_weights = dynamic_macro_weight + dynamic_micro_weight
            dynamic_macro_weight = dynamic_macro_weight / sum_dynamic_weights
            dynamic_micro_weight = dynamic_micro_weight / sum_dynamic_weights
            dynamic_macro_weight = dynamic_macro_weight.clip(0.2, 0.8) # 限制权重范围
            dynamic_micro_weight = dynamic_micro_weight.clip(0.2, 0.8)
        if is_probe_active:
            if dynamic_weight_mod_enabled:
                print(f"       - 过程: norm_volatility_instability: {norm_volatility_instability.loc[probe_date]:.4f}")
                print(f"       - 过程: norm_flow_slope: {norm_flow_slope.loc[probe_date]:.4f}")
                print(f"       - 过程: dynamic_macro_weight: {dynamic_macro_weight.loc[probe_date]:.4f}")
                print(f"       - 过程: dynamic_micro_weight: {dynamic_micro_weight.loc[probe_date]:.4f}")
        # --- 5. 融合基础战场控制权 ---
        # 宏观资金流向 * 动态宏观权重 + 微观盘口控制力 * 动态微观权重
        base_battlefield_control_score = (
            flow_consensus_score * dynamic_macro_weight +
            micro_control_score * dynamic_micro_weight
        )
        # 应用诡道调制器
        base_battlefield_control_score = base_battlefield_control_score * deception_modulator
        if is_probe_active:
            print(f"       - 过程: base_battlefield_control_score (before dynamic evolution): {base_battlefield_control_score.loc[probe_date]:.4f}")
        # --- 6. 战场控制权动态演化 (Dynamic Evolution of Battlefield Control) ---
        smoothed_base_score = base_battlefield_control_score.ewm(span=smoothing_ema_span, adjust=False).mean()
        velocity = smoothed_base_score.diff(1).fillna(0)
        acceleration = velocity.diff(1).fillna(0)
        norm_velocity = get_adaptive_mtf_normalized_bipolar_score(velocity, df_index, tf_weights=tf_weights_ff)
        norm_acceleration = get_adaptive_mtf_normalized_bipolar_score(acceleration, df_index, tf_weights=tf_weights_ff)
        # 融合基础分、速度和加速度
        final_score = (
            (base_battlefield_control_score.add(1)/2).pow(dynamic_evolution_base_weights.get('base_score', 0.6)) *
            (norm_velocity.add(1)/2).pow(dynamic_evolution_base_weights.get('velocity', 0.2)) *
            (norm_acceleration.add(1)/2).pow(dynamic_evolution_base_weights.get('acceleration', 0.2))
        ).pow(1 / (dynamic_evolution_base_weights.get('base_score', 0.6) + dynamic_evolution_base_weights.get('velocity', 0.2) + dynamic_evolution_base_weights.get('acceleration', 0.2))) * 2 - 1
        if is_probe_active:
            print(f"       - 过程: smoothed_base_score: {smoothed_base_score.loc[probe_date]:.4f}")
            print(f"       - 过程: velocity: {velocity.loc[probe_date]:.4f}")
            print(f"       - 过程: acceleration: {acceleration.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_velocity: {norm_velocity.loc[probe_date]:.4f}")
            print(f"       - 过程: norm_acceleration: {norm_acceleration.loc[probe_date]:.4f}")
            print(f"       - 结果: final_score: {final_score.loc[probe_date]:.4f}")
        return final_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int, capital_signature_score: pd.Series, flow_health_score: pd.Series) -> pd.Series:
        """
        【V4.0 · 信念质量调制版】资金流公理二：诊断“攻击性意图”
        - 核心逻辑: 融合“闪电战”意图与“阵地战”决心，评估资金的主动攻击意愿。
        - V4.0 升级: 引入“信念质量调节器”，使用“资本属性”和“资金流结构健康度”对原始攻击意图进行非线性调制。
                      旨在放大由“耐心资本”在“健康结构”上发起的攻击，抑制“敏捷资本”在“脆弱结构”上的攻击。
        """
        print("    -> [资金流层] 正在诊断“攻击性意图”公理...")
        required_signals = [
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'large_order_support_D', 'large_order_pressure_D', 'main_force_cost_advantage_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_conviction"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. “闪电战”意图 (主动攻击)
        buy_exhaustion = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_conviction")
        sell_exhaustion = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_conviction")
        blitz_intent_score = get_adaptive_mtf_normalized_bipolar_score(buy_exhaustion - sell_exhaustion, df_index, tf_weights_ff)
        # 2. “阵地战”决心 (大单攻防)
        large_support = self._get_safe_series(df, df, 'large_order_support_D', 0.0, method_name="_diagnose_axiom_conviction")
        large_pressure = self._get_safe_series(df, df, 'large_order_pressure_D', 0.0, method_name="_diagnose_axiom_conviction")
        trench_warfare_score = get_adaptive_mtf_normalized_bipolar_score(large_support - large_pressure, df_index, tf_weights_ff)
        # 3. 辅助证据 (成本优势)
        cost_advantage = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_axiom_conviction")
        cost_advantage_score = get_adaptive_mtf_normalized_bipolar_score(cost_advantage, df_index, tf_weights_ff)
        # 4. 融合，得到原始意图分
        aggressive_intent_score = (
            blitz_intent_score * 0.6 +
            trench_warfare_score * 0.3 +
            cost_advantage_score * 0.1
        )
        # 5. 新增：信念质量调节器
        modulator_weights = get_param_value(p_conf_ff.get('conviction_modulator_weights'), {'capital_signature': 0.2, 'flow_health': 0.3})
        quality_modulator = (1 +
                             capital_signature_score * modulator_weights.get('capital_signature', 0.2) +
                             flow_health_score * modulator_weights.get('flow_health', 0.3)
                             ).clip(0.5, 1.5)
        final_modulated_score = aggressive_intent_score * quality_modulator
        return final_modulated_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V4.0 · 纯度与动能版】资金流公理三：诊断“资金流纯度与动能”
        - 核心逻辑: 融合流量、加速度、对倒强度与盘口流动性，评估资金流的真实动能。
        - A股特性: 动能不仅要看“速度”，更要看“含金量”和“环境”。此模型旨在识别纯净、高效的资金流爆发。
        """
        print("    -> [资金流层] 正在诊断“资金流纯度与动能”公理...")
        required_signals = [
            'SLOPE_5_NMFNF_D', 'SLOPE_21_NMFNF_D', 'wash_trade_intensity_D',
            'order_book_liquidity_supply_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_momentum"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. 基础动能 (多周期斜率)
        slope_5 = self._get_safe_series(df, df, 'SLOPE_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        slope_21 = self._get_safe_series(df, df, 'SLOPE_21_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        norm_slope_5 = get_adaptive_mtf_normalized_bipolar_score(slope_5, df_index, tf_weights_ff)
        norm_slope_21 = get_adaptive_mtf_normalized_bipolar_score(slope_21, df_index, tf_weights_ff)
        base_momentum = (norm_slope_5 * 0.7 + norm_slope_21 * 0.3)
        # 2. 纯度过滤器
        wash_trade = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        purity_filter = 1 - get_adaptive_mtf_normalized_score(wash_trade, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 3. 环境调节器
        liquidity_supply = self._get_safe_series(df, df, 'order_book_liquidity_supply_D', 1.0, method_name="_diagnose_axiom_flow_momentum")
        liquidity_amplifier = 1 / liquidity_supply.replace(0, 1e-9).clip(0.5, 2.0) # 反比关系，并限制范围
        # 4. 融合
        true_momentum = base_momentum * purity_filter * liquidity_amplifier
        return true_momentum.clip(-1, 1).astype(np.float32)

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

