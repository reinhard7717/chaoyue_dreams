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
        【V21.8 · 资金流吸筹拐点意图参数传递版】资金流情报分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出资金流领域的原子公理信号和资金流背离信号。
        - 移除信号: SCORE_FUND_FLOW_BULLISH_RESONANCE, SCORE_FUND_FLOW_BEARISH_RESONANCE, BIPOLAR_FUND_FLOW_DOMAIN_HEALTH, SCORE_FUND_FLOW_BOTTOM_REVERSAL, SCORE_FUND_FLOW_TOP_REVERSAL。
        - 【更新】将 `_diagnose_axiom_increment` 替换为 `_diagnose_axiom_flow_momentum`。
        - 【新增】调用 `_diagnose_fund_flow_accumulation_inflection_intent` 方法，生成资金流吸筹拐点意图信号。
        - 【修复】将 `axiom_flow_momentum` 和 `axiom_consensus` 作为参数传递给 `_diagnose_fund_flow_accumulation_inflection_intent`，解决其获取不到当前日数据的问题。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。")
            return {}
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        axiom_flow_momentum = self._diagnose_axiom_flow_momentum(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        # 将当前计算出的 axiom_flow_momentum 和 axiom_consensus 传递给 _diagnose_fund_flow_accumulation_inflection_intent
        fund_flow_inflection_intent = self._diagnose_fund_flow_accumulation_inflection_intent(
            df, norm_window, axiom_flow_momentum, axiom_consensus
        )
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        all_states['PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT'] = fund_flow_inflection_intent
        # 引入资金流层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.2 · 上下文修复版】资金流公理四：诊断“资金背离”
        - 【V1.2 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights = get_param_value(p_conf.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        main_force_flow_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_divergence"), df.index, tf_weights)
        divergence_score = (main_force_flow_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.1 · 情报校验加固版】资金流公理一：诊断“战场控制权”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算，提升了健壮性。
        - 核心升级: 不再仅依赖资金流向，而是引入高频盘口指标来衡量主力对市场的实际控制力。
        - 新增权重: 引入`order_book_imbalance`（盘口压力）和`imbalance_effectiveness`（压力有效性）作为核心判断依据。
        - 引入惩罚: 引入`wash_trade_intensity`（对倒强度）作为负向调节因子，惩罚虚假繁荣。
        """
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'order_book_imbalance_D', 'imbalance_effectiveness_D', 'wash_trade_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_consensus"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        main_force_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        retail_flow = self._get_safe_series(df, df, 'retail_net_flow_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        raw_bipolar_series = main_force_flow - retail_flow
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        consensus_score_base = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df_index, tf_weights_ff, sensitivity=1.0)
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_consensus")
        imbalance_effectiveness_raw = self._get_safe_series(df, df, 'imbalance_effectiveness_D', 0.0, method_name="_diagnose_axiom_consensus")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_consensus")
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff, sensitivity=0.5)
        effectiveness_score = get_adaptive_mtf_normalized_bipolar_score(imbalance_effectiveness_raw, df_index, tf_weights_ff, sensitivity=0.5)
        wash_trade_penalty = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=False, tf_weights=tf_weights_ff).clip(0, 1)
        control_power_score = (imbalance_score * 0.6 + effectiveness_score * 0.4).clip(-1, 1)
        consensus_score = (
            consensus_score_base * 0.4 +
            control_power_score * 0.6
        ) * wash_trade_penalty
        return consensus_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.1 · 情报校验加固版】资金流公理二：诊断“攻击性意图”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 将信念的度量从被动的成本优势，升级为主动的攻击行为。
        - 核心证据: 引入`buy_quote_exhaustion_rate`和`sell_quote_exhaustion_rate`，构建“主动攻击得分”，作为信念的最强表征。
        - 辅助证据: 引入`large_order_support`和`large_order_pressure`，构建“阵地战决心”因子。
        - 权重重构: 大幅提升主动攻击证据的权重，重塑信念的定义。
        """
        required_signals = [
            'main_force_conviction_index_D', 'main_force_cost_advantage_D', 'hidden_accumulation_intensity_D',
            'buy_quote_exhaustion_rate_D', 'sell_quote_exhaustion_rate_D',
            'large_order_support_D', 'large_order_pressure_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_conviction"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        conviction_index_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', 0.0, method_name="_diagnose_axiom_conviction")
        cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', 0.0, method_name="_diagnose_axiom_conviction")
        hidden_accumulation_raw = self._get_safe_series(df, df, 'hidden_accumulation_intensity_D', 0.0, method_name="_diagnose_axiom_conviction")
        conviction_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(conviction_index_raw, df_index, tf_weights_ff, sensitivity=10.0)
        cost_advantage_bipolar = get_adaptive_mtf_normalized_bipolar_score(cost_advantage_raw, df_index, tf_weights_ff, sensitivity=100.0)
        hidden_accumulation_score = get_adaptive_mtf_normalized_score(hidden_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights_ff)
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_conviction")
        sell_exhaustion_raw = self._get_safe_series(df, df, 'sell_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_axiom_conviction")
        large_support_raw = self._get_safe_series(df, df, 'large_order_support_D', 0.0, method_name="_diagnose_axiom_conviction")
        large_pressure_raw = self._get_safe_series(df, df, 'large_order_pressure_D', 0.0, method_name="_diagnose_axiom_conviction")
        aggressive_action_raw = buy_exhaustion_raw - sell_exhaustion_raw
        aggressive_action_score = get_adaptive_mtf_normalized_bipolar_score(aggressive_action_raw, df_index, tf_weights_ff, sensitivity=0.5)
        positional_warfare_raw = large_support_raw - large_pressure_raw
        positional_warfare_score = get_adaptive_mtf_normalized_bipolar_score(positional_warfare_raw, df_index, tf_weights_ff, sensitivity=0.5)
        raw_bipolar_series = (
            aggressive_action_score * 0.5 +
            positional_warfare_score * 0.2 +
            cost_advantage_bipolar * 0.15 +
            conviction_index_bipolar * 0.1 +
            hidden_accumulation_score * 0.05
        ).clip(-1, 1)
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df_index, tf_weights_ff, sensitivity=1.0)
        return conviction_score.astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 情报校验加固版】资金流公理三：诊断“资金流纯度与动能”
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 将动能的评估从纯粹的“量”升级为“量”与“质”的结合。
        - 引入纯度: 强化`wash_trade_intensity`的负向权重，严厉惩罚含有杂质的资金流动。
        - 引入环境: 引入`order_book_liquidity_supply`作为环境因子，在流动性稀薄时放大动能得分，反之则削弱。
        """
        required_signals = [
            'main_force_net_flow_calibrated_D', 'total_market_value_D', 'SLOPE_5_NMFNF_D',
            'SLOPE_21_NMFNF_D', 'wash_trade_intensity_D', 'order_book_imbalance_D',
            'order_book_liquidity_supply_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_flow_momentum"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        main_force_net_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        total_market_value = self._get_safe_series(df, df, 'total_market_value_D', 1e9, method_name="_diagnose_axiom_flow_momentum")
        nmfnf = (main_force_net_flow / total_market_value.replace(0, 1e9)).fillna(0)
        slope_5_nmfnf = self._get_safe_series(df, df, 'SLOPE_5_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        slope_21_nmfnf = self._get_safe_series(df, df, 'SLOPE_21_NMFNF_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_flow_momentum")
        liquidity_supply_raw = self._get_safe_series(df, df, 'order_book_liquidity_supply_D', 1.0, method_name="_diagnose_axiom_flow_momentum")
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(nmfnf, df_index, tf_weights_ff, sensitivity=0.001)
        slope_5_nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(slope_5_nmfnf, df_index, tf_weights_ff, sensitivity=0.0001)
        slope_21_nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(slope_21_nmfnf, df_index, tf_weights_ff, sensitivity=0.00005)
        wash_trade_penalty_score = get_adaptive_mtf_normalized_score(wash_trade_intensity_raw, df_index, ascending=False, tf_weights=tf_weights_ff).clip(0, 1)
        order_book_imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff, sensitivity=0.5)
        liquidity_amplifier = 1 / liquidity_supply_raw.replace(0, 1e-9).clip(0.5, 2.0)
        liquidity_amplifier_score = get_adaptive_mtf_normalized_score(liquidity_amplifier, df_index, ascending=True, tf_weights=tf_weights_ff).clip(0.5, 1.5)
        base_momentum = (
            nmfnf_score * 0.4 +
            slope_5_nmfnf_score * 0.3 +
            slope_21_nmfnf_score * 0.2 +
            order_book_imbalance_score * 0.1
        ).clip(-1, 1)
        flow_momentum_score = base_momentum * liquidity_amplifier_score * wash_trade_penalty_score
        return flow_momentum_score.clip(-1, 1).astype(np.float32)

    def _diagnose_fund_flow_accumulation_inflection_intent(self, df: pd.DataFrame, norm_window: int, flow_momentum_current: pd.Series, consensus_score_current: pd.Series) -> pd.Series:
        """
        【V2.1 · 情报校验加固版】识别主力从隐蔽吸筹转向公开强攻的转折信号。
        - 核心增强: 增加了前置信号校验，确保所有依赖数据存在后才执行计算。
        - 核心升级: 引入高频指标作为转折点的核心确认证据，而非简单的加分项。
        - 核心证据 (强攻): `buy_quote_exhaustion_rate`的飙升是确认主力开始“抢筹”的关键。
        - 核心证据 (摊牌): `large_order_pressure`的减弱是确认主力放弃伪装的信号。
        - 触发逻辑重构: 只有在“强攻”和“摊牌”的高频证据出现时，拐点意图信号才会被激活并赋予高分。
        """
        required_signals = ['buy_quote_exhaustion_rate_D', 'large_order_pressure_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_fund_flow_accumulation_inflection_intent"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_inflection = get_params_block(self.strategy, 'fund_flow_inflection_params', {})
        tf_weights_inflection = get_param_value(p_conf_inflection.get('tf_fusion_weights'), {5: 0.5, 13: 0.3, 21: 0.2})
        psai = self._get_safe_series(df, self.strategy.df_indicators, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        main_force_flow = self._get_safe_series(df, self.strategy.df_indicators, 'FUND_FLOW_MAIN_FORCE_FLOW', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        buy_exhaustion_raw = self._get_safe_series(df, df, 'buy_quote_exhaustion_rate_D', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        large_pressure_raw = self._get_safe_series(df, df, 'large_order_pressure_D', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        flow_momentum = flow_momentum_current
        psai_high_threshold = get_param_value(p_conf_inflection.get('psai_high_threshold'), 0.5)
        mf_flow_positive_threshold = get_param_value(p_conf_inflection.get('mf_flow_positive_threshold'), 0.0)
        buy_exhaustion_threshold = get_param_value(p_conf_inflection.get('buy_exhaustion_threshold'), 0.7)
        large_pressure_low_threshold = get_param_value(p_conf_inflection.get('large_pressure_low_threshold'), 0.3)
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
        inflection_intent_score_normalized = get_adaptive_mtf_normalized_score(inflection_intent_score, df_index, ascending=True, tf_weights=tf_weights_inflection).clip(0, 1)
        self.strategy.df_indicators['PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT'] = inflection_intent_score_normalized
        return inflection_intent_score_normalized.astype(np.float32)



