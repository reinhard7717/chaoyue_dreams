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
        df_index = df.index # [代码修改] 使用传入的 df.index
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
        【V1.13 · 上下文修复版】资金流公理一：诊断“共识与分歧”
        - 【V1.13 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        df_index = df.index
        buy_sm_amount = self._get_safe_series(df, df, 'buy_sm_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_sm_amount = self._get_safe_series(df, df, 'sell_sm_amount_D', 0, method_name="_diagnose_axiom_consensus")
        buy_md_amount = self._get_safe_series(df, df, 'buy_md_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_md_amount = self._get_safe_series(df, df, 'sell_md_amount_D', 0, method_name="_diagnose_axiom_consensus")
        buy_lg_amount = self._get_safe_series(df, df, 'buy_lg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_lg_amount = self._get_safe_series(df, df, 'sell_lg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        buy_elg_amount = self._get_safe_series(df, df, 'buy_elg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        sell_elg_amount = self._get_safe_series(df, df, 'sell_elg_amount_D', 0, method_name="_diagnose_axiom_consensus")
        net_sm_amount = buy_sm_amount - sell_sm_amount
        net_md_amount = buy_md_amount - sell_md_amount
        net_lg_amount = buy_lg_amount - sell_lg_amount
        net_elg_amount = buy_elg_amount - sell_elg_amount
        main_force_flow = net_elg_amount + net_lg_amount
        retail_flow = net_md_amount + net_sm_amount
        raw_bipolar_series = main_force_flow - retail_flow
        battle_intensity_raw = self._get_safe_series(df, df, 'mf_retail_battle_intensity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_consensus")
        main_force_ofi_raw = self._get_safe_series(df, df, 'main_force_ofi_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_consensus")
        retail_ofi_raw = self._get_safe_series(df, df, 'retail_ofi_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_consensus")
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        battle_intensity_factor = get_adaptive_mtf_normalized_score(battle_intensity_raw, df_index, ascending=True, tf_weights=tf_weights_ff).clip(0, 1)
        sm_md_net_flow = net_sm_amount + net_md_amount
        lg_xl_net_flow = net_lg_amount + net_elg_amount
        pct_change = self._get_safe_series(df, df, 'pct_change_D', 0.0, method_name="_diagnose_axiom_consensus")
        split_order_accumulation_raw = pd.Series(0.0, index=df_index)
        condition_4_optimized = (sm_md_net_flow > lg_xl_net_flow.abs()) | np.isclose(sm_md_net_flow, lg_xl_net_flow.abs(), atol=1e-5)
        condition_mask = (pct_change <= 0) & (sm_md_net_flow > 0) & (lg_xl_net_flow <= 0) & condition_4_optimized
        # ... (探针代码省略)
        split_order_accumulation_raw.loc[condition_mask] = (sm_md_net_flow - lg_xl_net_flow).loc[condition_mask]
        # ... (探针代码省略)
        normalized_split_factor_series = get_adaptive_mtf_normalized_score(split_order_accumulation_raw, df_index, ascending=True, tf_weights=tf_weights_ff).clip(0, 1)
        split_order_accumulation_factor = normalized_split_factor_series.where(split_order_accumulation_raw > 0, 0)
        self.strategy.df_indicators['FUND_FLOW_MAIN_FORCE_FLOW'] = main_force_flow
        self.strategy.df_indicators['FUND_FLOW_RETAIL_FLOW'] = retail_flow
        self.strategy.df_indicators['FUND_FLOW_SM_MD_NET_FLOW'] = sm_md_net_flow
        self.strategy.df_indicators['FUND_FLOW_LG_XL_NET_FLOW'] = lg_xl_net_flow
        self.strategy.df_indicators['PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY'] = split_order_accumulation_factor
        consensus_score_base = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df_index, tf_weights_ff, sensitivity=1.0)
        main_force_ofi_score = get_adaptive_mtf_normalized_bipolar_score(main_force_ofi_raw, df_index, tf_weights_ff, sensitivity=0.5)
        retail_ofi_score = get_adaptive_mtf_normalized_bipolar_score(retail_ofi_raw, df_index, tf_weights_ff, sensitivity=0.5)
        consensus_score = (
            consensus_score_base * 0.6 +
            split_order_accumulation_factor * 0.2 +
            main_force_ofi_score * 0.15 -
            retail_ofi_score * 0.05
        ).clip(-1, 1)
        # ... (探针代码省略)
        return consensus_score.astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.6 · 上下文修复版】资金流公理二：诊断“信念与决心”
        - 【V1.6 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        df_index = df.index
        conviction_index_raw = self._get_safe_series(df, df, 'main_force_conviction_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        cost_advantage_raw = self._get_safe_series(df, df, 'main_force_cost_advantage_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        t0_efficiency_raw = self._get_safe_series(df, df, 'main_force_t0_efficiency_D', pd.Series(0.5, index=df_index), method_name="_diagnose_axiom_conviction")
        price_impact_raw = self._get_safe_series(df, df, 'main_force_price_impact_ratio_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        microstructure_efficiency_raw = self._get_safe_series(df, df, 'microstructure_efficiency_index_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        hidden_accumulation_raw = self._get_safe_series(df, df, 'hidden_accumulation_intensity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_conviction")
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        conviction_index_bipolar = get_adaptive_mtf_normalized_bipolar_score(conviction_index_raw, df_index, tf_weights_ff, sensitivity=10.0)
        cost_advantage_bipolar = get_adaptive_mtf_normalized_bipolar_score(cost_advantage_raw, df_index, tf_weights_ff, sensitivity=100.0)
        t0_efficiency_bipolar = get_adaptive_mtf_normalized_bipolar_score(t0_efficiency_raw, df_index, tf_weights_ff, sensitivity=0.5)
        price_impact_bipolar = get_adaptive_mtf_normalized_bipolar_score(price_impact_raw, df_index, tf_weights_ff, sensitivity=10.0)
        microstructure_efficiency_score = get_adaptive_mtf_normalized_bipolar_score(microstructure_efficiency_raw, df_index, tf_weights_ff, sensitivity=0.5)
        hidden_accumulation_score = get_adaptive_mtf_normalized_bipolar_score(hidden_accumulation_raw, df_index, tf_weights_ff, sensitivity=0.5)
        raw_bipolar_series = (
            conviction_index_bipolar * 0.3 +
            cost_advantage_bipolar * 0.3 +
            price_impact_bipolar * 0.15 +
            hidden_accumulation_score * 0.1 -
            t0_efficiency_bipolar * 0.1 -
            microstructure_efficiency_score * 0.05
        ).clip(-1, 1)
        conviction_score = get_adaptive_mtf_normalized_bipolar_score(raw_bipolar_series, df_index, tf_weights_ff, sensitivity=1.0)
        # ... (探针代码省略)
        return conviction_score.astype(np.float32)

    def _diagnose_axiom_flow_momentum(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V2.3 · 上下文修复版】资金流公理三：诊断“资金流动量”
        - 【V2.3 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        df_index = df.index
        required_signals = [
            'main_force_net_flow_calibrated_D', 'total_market_value_D',
            'SLOPE_5_NMFNF_D', 'SLOPE_21_NMFNF_D',
            'wash_trade_intensity_D', 'order_book_imbalance_D'
        ]
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            print(f"    -> [资金流动量探针] 警告: 缺少核心信号 {missing_signals}，使用默认值0.0。")
            return pd.Series(0.0, index=df_index)
        main_force_net_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        total_market_value = self._get_safe_series(df, df, 'total_market_value_D', pd.Series(1e9, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        nmfnf = (main_force_net_flow / total_market_value.replace(0, 1e9)).fillna(0)
        slope_5_nmfnf = self._get_safe_series(df, df, 'SLOPE_5_NMFNF_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        slope_21_nmfnf = self._get_safe_series(df, df, 'SLOPE_21_NMFNF_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        wash_trade_intensity_raw = self._get_safe_series(df, df, 'wash_trade_intensity_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        order_book_imbalance_raw = self._get_safe_series(df, df, 'order_book_imbalance_D', pd.Series(0.0, index=df_index), method_name="_diagnose_axiom_flow_momentum")
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(nmfnf, df_index, tf_weights_ff, sensitivity=0.001)
        slope_5_nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(slope_5_nmfnf, df_index, tf_weights_ff, sensitivity=0.0001)
        slope_21_nmfnf_score = get_adaptive_mtf_normalized_bipolar_score(slope_21_nmfnf, df_index, tf_weights_ff, sensitivity=0.00005)
        wash_trade_intensity_score = get_adaptive_mtf_normalized_bipolar_score(wash_trade_intensity_raw * -1, df_index, tf_weights_ff, sensitivity=0.5)
        order_book_imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance_raw, df_index, tf_weights_ff, sensitivity=0.5)
        flow_momentum_score = (
            nmfnf_score * 0.3 +
            slope_5_nmfnf_score * 0.25 +
            slope_21_nmfnf_score * 0.2 +
            order_book_imbalance_score * 0.15 +
            wash_trade_intensity_score * 0.1
        ).clip(-1, 1)
        # ... (探针代码省略)
        return flow_momentum_score.astype(np.float32)

    def _diagnose_fund_flow_accumulation_inflection_intent(self, df: pd.DataFrame, norm_window: int, flow_momentum_current: pd.Series, consensus_score_current: pd.Series) -> pd.Series:
        """
        【V1.4 · 上下文修复版】识别主力从隐蔽吸筹转向公开抢筹的资金流迹象。
        - 【V1.4 修复】在调用 _get_safe_series 时传递 df 参数。
        """
        df_index = df.index
        inflection_intent_score = pd.Series(0.0, index=df_index)
        # ... (探针代码省略)
        # 1. 获取核心资金流信号
        psai = self._get_safe_series(df, self.strategy.df_indicators, 'PROCESS_META_SPLIT_ORDER_ACCUMULATION_INTENSITY', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        main_force_flow = self._get_safe_series(df, self.strategy.df_indicators, 'FUND_FLOW_MAIN_FORCE_FLOW', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        lg_xl_net_flow = self._get_safe_series(df, self.strategy.df_indicators, 'FUND_FLOW_LG_XL_NET_FLOW', 0.0, method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        flow_momentum = flow_momentum_current
        consensus_score = consensus_score_current
        large_order_pressure_raw = self._get_safe_series(df, df, 'large_order_pressure_D', pd.Series(0.0, index=df_index), method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        large_order_support_raw = self._get_safe_series(df, df, 'large_order_support_D', pd.Series(0.0, index=df_index), method_name="_diagnose_fund_flow_accumulation_inflection_intent")
        # ... (后续逻辑省略)
        p_conf_inflection = get_params_block(self.strategy, 'fund_flow_inflection_params', {})
        psai_high_threshold = get_param_value(p_conf_inflection.get('psai_high_threshold'), 0.5)
        mf_flow_positive_threshold = get_param_value(p_conf_inflection.get('mf_flow_positive_threshold'), 0.0)
        lg_xl_flow_positive_threshold = get_param_value(p_conf_inflection.get('lg_xl_flow_positive_threshold'), 0.0)
        flow_momentum_positive_threshold = get_param_value(p_conf_inflection.get('flow_momentum_positive_threshold'), 0.0)
        consensus_score_positive_threshold = get_param_value(p_conf_inflection.get('consensus_score_positive_threshold'), 0.0)
        large_order_pressure_threshold = get_param_value(p_conf_inflection.get('large_order_pressure_threshold'), 0.5)
        large_order_support_threshold = get_param_value(p_conf_inflection.get('large_order_support_threshold'), 0.5)
        cond_psai_high = (psai > psai_high_threshold)
        cond_mf_overt_buying = (main_force_flow > mf_flow_positive_threshold) & (lg_xl_net_flow > lg_xl_flow_positive_threshold)
        cond_flow_momentum_positive = (flow_momentum > flow_momentum_positive_threshold)
        cond_consensus_improving = (consensus_score > consensus_score_positive_threshold) | (consensus_score.diff(1) > 0)
        cond_large_order_favorable = (large_order_pressure_raw < large_order_pressure_threshold) & (large_order_support_raw > large_order_support_threshold)
        base_intent_mask = cond_psai_high & cond_mf_overt_buying
        inflection_intent_score.loc[base_intent_mask] = 0.5
        inflection_intent_score.loc[base_intent_mask & cond_flow_momentum_positive] += 0.2
        inflection_intent_score.loc[base_intent_mask & cond_consensus_improving] += 0.3
        inflection_intent_score.loc[base_intent_mask & cond_large_order_favorable] += 0.2
        inflection_intent_score = inflection_intent_score.clip(0, 1)
        tf_weights_inflection = get_param_value(p_conf_inflection.get('tf_fusion_weights'), {5: 0.5, 13: 0.3, 21: 0.2})
        inflection_intent_score_normalized = get_adaptive_mtf_normalized_score(inflection_intent_score, df_index, ascending=True, tf_weights=tf_weights_inflection).clip(0, 1)
        self.strategy.df_indicators['PROCESS_META_FUND_FLOW_ACCUMULATION_INFLECTION_INTENT'] = inflection_intent_score_normalized
        # ... (探针代码省略)
        return inflection_intent_score_normalized.astype(np.float32)



