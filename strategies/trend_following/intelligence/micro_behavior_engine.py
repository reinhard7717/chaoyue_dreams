import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

class MicroBehaviorEngine:
    """
    【V2.0 · 三大公理重构版】
    - 核心升级: 废弃旧的复杂诊断模型，引入基于主力微观操盘本质的“伪装、试探、效率”三大公理。
                使引擎更聚焦、逻辑更清晰、信号更纯粹。
    """
    def __init__(self, strategy_instance):
        """
        初始化微观行为诊断引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [微观行为情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def _get_signal(self, df: pd.DataFrame, signal_name: str, default_value: float = 0.0) -> pd.Series:
        """
        【V1.0】信号获取哨兵方法
        - 核心职责: 安全地从DataFrame获取信号。
        - 预警机制: 如果信号不存在，打印明确的警告信息，并返回一个包含默认值的Series，以防止程序崩溃。
        """
        if signal_name not in df.columns:
            print(f"    -> [微观行为引擎警告] 依赖信号 '{signal_name}' 在数据帧中不存在，将使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[signal_name]

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # [修改] 调整校验信息为“微观行为情报校验”
            print(f"    -> [微观行为情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_micro_behavior_synthesis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.5 · 纯粹原子版】微观行为诊断引擎总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出微观行为领域的原子公理信号和微观背离信号。
        - 移除信号: SCORE_MICRO_BEHAVIOR_BULLISH_RESONANCE, SCORE_MICRO_BEHAVIOR_BEARISH_RESONANCE, BIPOLAR_MICRO_BEHAVIOR_DOMAIN_HEALTH, SCORE_MICRO_BEHAVIOR_BOTTOM_REVERSAL, SCORE_MICRO_BEHAVIOR_TOP_REVERSAL。
        """
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            print("-> [指挥覆盖探针] 微观行为引擎在配置中被禁用，跳过分析。")
            return {}
        all_states = {}
        norm_window = get_param_value(p_conf.get('norm_window'), 55)
        axiom_deception = self._diagnose_axiom_deception(df, norm_window)
        axiom_probe = self._diagnose_axiom_probe(df, norm_window)
        axiom_efficiency = self._diagnose_axiom_efficiency(df, norm_window)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        all_states['SCORE_MICRO_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_MICRO_AXIOM_DECEPTION'] = axiom_deception
        all_states['SCORE_MICRO_AXIOM_PROBE'] = axiom_probe
        all_states['SCORE_MICRO_AXIOM_EFFICIENCY'] = axiom_efficiency
        # 引入微观行为层面的看涨/看跌背离信号
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_MICRO_BEHAVIOR_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_MICRO_BEHAVIOR_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 信号校验增强版】微观行为公理四：诊断“微观背离”
        - 核心修复: 将依赖信号 'active_buy_amount_D' 和 'active_sell_amount_D' 修正为实际存在的
                      'active_buying_support_D' 和 'active_selling_pressure_D'。
        - 核心逻辑: 诊断价格行为与微观订单流（如主动买卖盘）之间的背离。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `price_trend` 和 `order_flow_trend` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['pct_change_D', 'active_buying_support_D', 'active_selling_pressure_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        price_trend = get_adaptive_mtf_normalized_bipolar_score(self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_divergence"), df.index, default_weights)
        active_buy_support = self._get_safe_series(df, 'active_buying_support_D', method_name="_diagnose_axiom_divergence")
        active_sell_pressure = self._get_safe_series(df, 'active_selling_pressure_D', method_name="_diagnose_axiom_divergence")
        active_buy_sell_diff = active_buy_support - active_sell_pressure
        order_flow_trend = get_adaptive_mtf_normalized_bipolar_score(active_buy_sell_diff.diff(1), df.index, default_weights)
        divergence_score = (order_flow_trend - price_trend).clip(-1, 1)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_deception(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.5 · 信号校验增强版】微观行为公理一：诊断“伪装与欺骗”
        - 核心修复: 使用 _get_safe_series 方法安全获取所有依赖信号，防止因信号缺失而崩溃。
        - 逻辑修正: 明确使用 'SLOPE_5_winner_concentration_90pct_D' 作为筹码集中度变化的证据。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `deception_score` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = [
            'main_force_net_flow_calibrated_D', 'SLOPE_5_winner_concentration_90pct_D',
            'SLOPE_5_observed_large_order_size_avg_D', 'SLOPE_5_NMFNF_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_deception"):
            return pd.Series(0.0, index=df.index)
        main_force_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', method_name="_diagnose_axiom_deception")
        main_force_outflow = -main_force_flow_raw.clip(upper=0)
        chip_concentration_slope = self._get_safe_series(df, 'SLOPE_5_winner_concentration_90pct_D', method_name="_diagnose_axiom_deception")
        chip_concentration_increase = chip_concentration_slope.clip(lower=0)
        flow_vs_chip_deception = main_force_outflow * chip_concentration_increase
        granularity_slope = self._get_safe_series(df, 'SLOPE_5_observed_large_order_size_avg_D', method_name="_diagnose_axiom_deception")
        granularity_decrease = -granularity_slope.clip(upper=0)
        control_leverage_slope = self._get_safe_series(df, 'SLOPE_5_NMFNF_D', method_name="_diagnose_axiom_deception")
        control_increase = control_leverage_slope.clip(lower=0)
        granularity_vs_control_deception = granularity_decrease * control_increase
        raw_deception_score = flow_vs_chip_deception + granularity_vs_control_deception
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        deception_score = get_adaptive_mtf_normalized_bipolar_score(raw_deception_score, df.index, default_weights)
        return deception_score.astype(np.float32)

    def _diagnose_axiom_probe(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 信号校验增强版】微观行为公理二：诊断“试探与确认”
        - 核心修复: 使用 _get_safe_series 方法安全获取所有依赖信号。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `probe_score` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['high_D', 'low_D', 'open_D', 'close_D', 'main_force_net_flow_calibrated_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_probe"):
            return pd.Series(0.0, index=df.index)
        total_range = (self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_probe") - self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_probe")).replace(0, np.nan)
        upper_shadow_ratio = ((self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_probe") - np.maximum(self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_probe"), self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_probe"))) / total_range).fillna(0)
        main_force_flow_raw = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', method_name="_diagnose_axiom_probe")
        main_force_not_outflow = main_force_flow_raw.clip(lower=0)
        probe_up_score = upper_shadow_ratio * main_force_not_outflow
        lower_shadow_ratio = ((np.minimum(self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_probe"), self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_probe")) - self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_probe")) / total_range).fillna(0)
        main_force_inflow = main_force_flow_raw.clip(lower=0)
        probe_down_score = lower_shadow_ratio * main_force_inflow
        breakout_high = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_probe") > self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_probe").rolling(21).max().shift(1)).astype(float)
        main_force_not_inflow = -main_force_flow_raw.clip(upper=0)
        fake_breakout_score = breakout_high * main_force_not_inflow
        raw_probe_score = probe_up_score + probe_down_score - fake_breakout_score
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        probe_score = get_adaptive_mtf_normalized_bipolar_score(raw_probe_score, df.index, default_weights)
        return probe_score.astype(np.float32)

    def _diagnose_axiom_efficiency(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V1.3 · 信号校验增强版】微观行为公理三：诊断“成本与效率”
        - 核心修复: 使用 _get_safe_series 方法安全获取所有依赖信号。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `efficiency_score` 的归一化方式改为多时间维度自适应归一化。
        - [新增] 在方法入口处添加信号校验逻辑。
        """
        required_signals = ['amount_D', 'pct_change_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_efficiency"):
            return pd.Series(0.0, index=df.index)
        amount_series = self._get_safe_series(df, 'amount_D', method_name="_diagnose_axiom_efficiency")
        amount_ma = amount_series.rolling(norm_window).mean().replace(0, np.nan)
        amount_input = (amount_series / amount_ma).fillna(1.0)
        pct_change_series = self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_efficiency")
        price_output = pct_change_series.abs() * 100
        k = 0.1
        raw_efficiency_score = np.sign(pct_change_series) * (price_output - k * amount_input)
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {}) # 借用行为层的MTF权重配置
        p_mtf = get_param_value(p_conf.get('mtf_normalization_params'), {})
        default_weights = get_param_value(p_mtf.get('default_weights'), {'weights': {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}})
        efficiency_score = get_adaptive_mtf_normalized_bipolar_score(raw_efficiency_score, df.index, default_weights)
        return efficiency_score.astype(np.float32)


