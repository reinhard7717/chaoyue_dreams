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
        【V24.0 · 资本属性公理版】资金流情报分析总指挥
        - 核心新增: 引入第五大公理——“资本属性公理”，旨在识别驱动趋势的资金是“耐心资本”（机构）
                      还是“敏捷资本”（游资），为策略提供全新的决策维度。
        """
        p_conf = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 资金流情报引擎在配置中被禁用，跳过分析。")
            return {}
        print("启动【V24.0 · 资本属性公理版】资金流情报分析...")
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_consensus = self._diagnose_axiom_consensus(df, norm_window)
        axiom_conviction = self._diagnose_axiom_conviction(df, norm_window)
        axiom_flow_momentum = self._diagnose_axiom_flow_momentum(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        # [新增代码行] 调用新增的资本属性公理诊断方法
        axiom_capital_signature = self._diagnose_axiom_capital_signature(df, norm_window)
        all_states['SCORE_FF_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_FF_AXIOM_CONSENSUS'] = axiom_consensus
        all_states['SCORE_FF_AXIOM_CONVICTION'] = axiom_conviction
        all_states['SCORE_FF_AXIOM_FLOW_MOMENTUM'] = axiom_flow_momentum
        # [新增代码行] 将新的公理分数添加到状态字典
        all_states['SCORE_FF_AXIOM_CAPITAL_SIGNATURE'] = axiom_capital_signature
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_FUND_FLOW_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_FUND_FLOW_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        print(f"【V24.0 · 资本属性公理版】分析完成，生成 {len(all_states)} 个资金流原子信号。")
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
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [价资张力探针] @ {probe_date_for_loop.date()}:")
                print(f"       - disagreement_vector: {disagreement_vector.loc[probe_date_for_loop]:.4f}")
                print(f"       - tension_magnitude: {tension_magnitude.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_tension_score: {tension_score.loc[probe_date_for_loop]:.4f}")
        return tension_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_consensus(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · NaN修复版】资金流公理一：诊断“战场控制权”
        - 核心修复: 修正了 `micro_control_score` 的计算公式，通过对乘积因子取绝对值，
                      避免了对负数开平方根而导致的NaN污染问题，确保了信号的健壮性。
        """
        print("    -> [资金流层] 正在诊断“战场控制权”公理...")
        required_signals = [
            'main_force_net_flow_calibrated_D', 'retail_net_flow_calibrated_D',
            'order_book_imbalance_D', 'ofi_price_impact_factor_D', 'wash_trade_intensity_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_consensus"):
            return pd.Series(0.0, index=df.index)
        df_index = df.index
        p_conf_ff = get_params_block(self.strategy, 'fund_flow_ultimate_params', {})
        tf_weights_ff = get_param_value(p_conf_ff.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        # 1. 宏观资金流向
        main_force_flow = self._get_safe_series(df, df, 'main_force_net_flow_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        retail_flow = self._get_safe_series(df, df, 'retail_net_flow_calibrated_D', 0, method_name="_diagnose_axiom_consensus")
        flow_consensus_score = get_adaptive_mtf_normalized_bipolar_score(main_force_flow - retail_flow, df_index, tf_weights_ff)
        # 2. 微观盘口控制力
        order_book_imbalance = self._get_safe_series(df, df, 'order_book_imbalance_D', 0.0, method_name="_diagnose_axiom_consensus")
        ofi_impact = self._get_safe_series(df, df, 'ofi_price_impact_factor_D', 0.0, method_name="_diagnose_axiom_consensus")
        imbalance_score = get_adaptive_mtf_normalized_bipolar_score(order_book_imbalance, df_index, tf_weights_ff)
        impact_score = get_adaptive_mtf_normalized_bipolar_score(ofi_impact, df_index, tf_weights_ff)
        # 修正数学公式，先取绝对值再开方，避免NaN
        micro_control_score = (imbalance_score.abs() * impact_score.abs()).pow(0.5) * np.sign(imbalance_score)
        # 3. 纯度过滤器 (惩罚项)
        wash_trade_intensity = self._get_safe_series(df, df, 'wash_trade_intensity_D', 0.0, method_name="_diagnose_axiom_consensus")
        purity_filter = 1 - get_adaptive_mtf_normalized_score(wash_trade_intensity, df_index, ascending=True, tf_weights=tf_weights_ff)
        # 4. 融合
        battlefield_control_score = (flow_consensus_score * 0.4 + micro_control_score * 0.6) * purity_filter
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [战场控制权探针] @ {probe_date_for_loop.date()}:")
                print(f"       - flow_consensus_score: {flow_consensus_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - micro_control_score: {micro_control_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - purity_filter: {purity_filter.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_control_score: {battlefield_control_score.loc[probe_date_for_loop]:.4f}")
        return battlefield_control_score.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_conviction(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 攻击性意图版】资金流公理二：诊断“攻击性意图”
        - 核心逻辑: 融合“闪电战”意图（盘口消耗率）与“阵地战”决心（大单攻防），评估资金的主动攻击意愿。
        - A股特性: 真正的信念不是口头上的，而是用真金白银砸出来的。此模型旨在量化这种“砸盘”的力度。
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
        # 4. 融合
        aggressive_intent_score = (
            blitz_intent_score * 0.6 +
            trench_warfare_score * 0.3 +
            cost_advantage_score * 0.1
        )
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [攻击性意图探针] @ {probe_date_for_loop.date()}:")
                print(f"       - blitz_intent_score (闪电战): {blitz_intent_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - trench_warfare_score (阵地战): {trench_warfare_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - cost_advantage_score (成本): {cost_advantage_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_intent_score: {aggressive_intent_score.loc[probe_date_for_loop]:.4f}")
        return aggressive_intent_score.clip(-1, 1).astype(np.float32)

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
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [资金流纯度与动能探针] @ {probe_date_for_loop.date()}:")
                print(f"       - base_momentum: {base_momentum.loc[probe_date_for_loop]:.4f}")
                print(f"       - purity_filter: {purity_filter.loc[probe_date_for_loop]:.4f}")
                print(f"       - liquidity_amplifier: {liquidity_amplifier.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_true_momentum: {true_momentum.loc[probe_date_for_loop]:.4f}")
        return true_momentum.clip(-1, 1).astype(np.float32)

    def _diagnose_axiom_capital_signature(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】资金流公理五：诊断“资本属性”
        - 核心逻辑: 通过分析资金流的行为模式，区分趋势是由“耐心资本”（机构）还是“敏捷资本”（游资）主导。
        - A股特性: 机构建仓如“温水煮青蛙”，游资点火如“烈火烹油”，两者后续走势预期截然不同。
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
        institutional_flow = self._get_safe_series(df, df, 'net_lg_amount_calibrated_D', 0.0) + self._get_safe_series(df, df, 'net_xl_amount_calibrated_D', 0.0)
        # 证据一：持续稳定的净流入
        flow_consistency = institutional_flow.rolling(window=21).mean()
        # 证据二：流入过程的波动率低
        flow_steadiness = 1 - institutional_flow.rolling(window=21).std().pct_change().fillna(0).abs()
        patient_capital_score = (
            get_adaptive_mtf_normalized_score(flow_consistency, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.7 +
            get_adaptive_mtf_normalized_score(flow_steadiness, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.3
        ).clip(0, 1)
        # 2. “敏捷资本”（游资）画像
        # 证据一：高频盘口冲击力
        ofi = self._get_safe_series(df, df, 'main_force_ofi_D', 0.0)
        # 证据二：成交笔数爆发
        trade_count = self._get_safe_series(df, df, 'trade_count_D', 0.0)
        # 证据三：与题材热度高度相关
        theme_hotness = self._get_safe_series(df, df, 'THEME_HOTNESS_SCORE_D', 0.0)
        agile_capital_score = (
            get_adaptive_mtf_normalized_score(ofi.abs(), df_index, ascending=True, tf_weights=tf_weights_ff) * 0.4 +
            get_adaptive_mtf_normalized_score(trade_count, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.3 +
            get_adaptive_mtf_normalized_score(theme_hotness, df_index, ascending=True, tf_weights=tf_weights_ff) * 0.3
        ).clip(0, 1)
        # 3. 融合
        capital_signature_score = patient_capital_score - agile_capital_score
        # [新增] 调试探针
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df_index:
                print(f"    -> [资本属性探针] @ {probe_date_for_loop.date()}:")
                print(f"       - patient_capital_score (耐心资本): {patient_capital_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - agile_capital_score (敏捷资本): {agile_capital_score.loc[probe_date_for_loop]:.4f}")
                print(f"       - final_signature_score: {capital_signature_score.loc[probe_date_for_loop]:.4f}")
        return capital_signature_score.clip(-1, 1).astype(np.float32)



