import pandas as pd
import numpy as np
from typing import Dict, Any
from strategies.trend_following.utils import get_params_block, get_param_value, get_adaptive_mtf_normalized_bipolar_score, bipolar_to_exclusive_unipolar

class PatternIntelligence:
    """
    【V8.0 · 清洁版】形态智能引擎
    - 核心修改: 移除了所有用于调试的print探针和过程性打印，恢复静默运行模式。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _get_safe_series(self, df: pd.DataFrame, column_name: str, default_value: Any = 0.0, method_name: str = "未知方法") -> pd.Series:
        """
        安全地从DataFrame获取Series，如果不存在则打印警告并返回默认Series。
        """
        if column_name not in df.columns:
            print(f"    -> [形态情报警告] 方法 '{method_name}' 缺少数据 '{column_name}'，使用默认值 {default_value}。")
            return pd.Series(default_value, index=df.index)
        return df[column_name]

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
        """
        【V1.1 · 上下文修复版】安全地从原子状态库或主数据帧中获取分数。
        - 【V1.1 修复】接收 df 参数，并使用其索引创建默认 Series，确保上下文一致。
        """
        if name in self.strategy.atomic_states:
            return self.strategy.atomic_states[name]
        elif name in self.strategy.df_indicators.columns:
            return self.strategy.df_indicators[name]
        else:
            print(f"    -> [形态情报警告] 原子信号 '{name}' 不存在，使用默认值 {default}。")
            return pd.Series(default, index=df.index)

    def _validate_required_signals(self, df: pd.DataFrame, required_signals: list, method_name: str) -> bool:
        """
        【V1.0 · 战前情报校验】内部辅助方法，用于在方法执行前验证所有必需的数据信号是否存在。
        """
        missing_signals = [s for s in required_signals if s not in df.columns]
        if missing_signals:
            # 调整校验信息为“形态情报校验”
            print(f"    -> [形态情报校验] 方法 '{method_name}' 启动失败：缺少核心信号 {missing_signals}。")
            return False
        return True

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.5 · 多方炮集成版】形态分析总指挥
        - 核心升级: 废弃原子层面的“共振”和“领域健康度”信号。
        - 核心职责: 只输出形态领域的原子公理信号和形态背离信号。
        - 移除信号: SCORE_PATTERN_BULLISH_RESONANCE, SCORE_PATTERN_BEARISH_RESONANCE, BIPOLAR_PATTERN_DOMAIN_HEALTH, SCORE_PATTERN_BOTTOM_REVERSAL, SCORE_PATTERN_TOP_REVERSAL。
        - 【新增】诊断“回踩确认二次启动”形态和“多方炮”形态。
        """
        all_states = {}
        p_conf = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        norm_window = get_param_value(p_conf.get('norm_window'), 60)
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        axiom_reversal = self._diagnose_axiom_reversal(df, norm_window)
        axiom_breakout = self._diagnose_axiom_breakout(df, norm_window)
        all_states['SCORE_PATTERN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_PATTERN_AXIOM_REVERSAL'] = axiom_reversal
        all_states['SCORE_PATTERN_AXIOM_BREAKOUT'] = axiom_breakout
        bullish_divergence, bearish_divergence = bipolar_to_exclusive_unipolar(axiom_divergence)
        all_states['SCORE_PATTERN_BULLISH_DIVERGENCE'] = bullish_divergence.astype(np.float32)
        all_states['SCORE_PATTERN_BEARISH_DIVERGENCE'] = bearish_divergence.astype(np.float32)
        # 诊断“回踩确认二次启动”形态
        axiom_pullback_confirmation = self._diagnose_axiom_pullback_confirmation(df)
        all_states['SCORE_PATTERN_PULLBACK_CONFIRMATION'] = axiom_pullback_confirmation
        # 新增行: 诊断“多方炮”形态
        axiom_duofangpao = self._diagnose_axiom_duofangpao(df)
        all_states['SCORE_PATTERN_DUOFANGPAO'] = axiom_duofangpao
        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.2 · 信号校验增强版】形态公理一：诊断“背离”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `divergence_score` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['SLOPE_13_close_D', 'SLOPE_13_intraday_vwap_div_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_divergence"):
            return pd.Series(0.0, index=df.index)
        price_slope = self._get_safe_series(df, 'SLOPE_13_close_D', 0, method_name="_diagnose_axiom_divergence")
        momentum_slope = self._get_safe_series(df, 'SLOPE_13_intraday_vwap_div_index_D', 0, method_name="_diagnose_axiom_divergence")
        bullish_divergence_strength = ((price_slope < 0) & (momentum_slope > 0)).astype(float)
        bearish_divergence_strength = ((price_slope > 0) & (momentum_slope < 0)).astype(float)
        raw_divergence_score = bullish_divergence_strength - bearish_divergence_strength
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}) # 借用筹码的MTF权重配置
        divergence_score = get_adaptive_mtf_normalized_bipolar_score(raw_divergence_score, df.index, tf_weights_pattern)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_reversal(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.2 · 信号校验增强版】形态公理二：诊断“反转”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `reversal_score` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['counterparty_exhaustion_index_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_reversal"):
            return pd.Series(0.0, index=df.index)
        raw_reversal_score = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0, method_name="_diagnose_axiom_reversal")
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        reversal_score = get_adaptive_mtf_normalized_bipolar_score(raw_reversal_score, df.index, tf_weights_pattern)
        return reversal_score.astype(np.float32)

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.2 · 信号校验增强版】形态公理三：诊断“突破”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `breakout_score` 的归一化方式改为多时间维度自适应归一化。
        """
        required_signals = ['close_D', 'dynamic_consolidation_high_D', 'dynamic_consolidation_low_D', 'breakout_quality_score_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_breakout"):
            return pd.Series(0.0, index=df.index)
        is_breakout_up = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") > self._get_safe_series(df, 'dynamic_consolidation_high_D', np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        is_breakout_down = (self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_breakout") < self._get_safe_series(df, 'dynamic_consolidation_low_D', -np.inf, method_name="_diagnose_axiom_breakout")).astype(float)
        breakout_direction = is_breakout_up - is_breakout_down
        breakout_quality = self._get_safe_series(df, 'breakout_quality_score_D', pd.Series(0, index=df.index), method_name="_diagnose_axiom_breakout")
        raw_breakout_score = breakout_direction * breakout_quality
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        breakout_score = get_adaptive_mtf_normalized_bipolar_score(raw_breakout_score, df.index, tf_weights_pattern)
        return breakout_score.astype(np.float32)

    def _diagnose_axiom_pullback_confirmation(self, df: pd.DataFrame) -> pd.Series:
        """
        【V5.7 · 信号校验增强版】形态公理四：诊断“回踩确认二次启动”形态
        - 【V5.6 修复】在调用 _get_atomic_score 时传递 df 参数。
        """
        required_signals = [
            'open_D', 'close_D', 'high_D', 'low_D', 'pct_change_D', 'volume_D', 'VOL_MA_5_D', 'VOL_MA_21_D',
            'main_force_net_flow_calibrated_D', 'SLOPE_5_winner_concentration_90pct_D', 'large_order_pressure_D',
            'large_order_support_D', 'hidden_accumulation_intensity_D', 'dip_absorption_power_D',
            'upper_shadow_selling_pressure_D', 'lower_shadow_absorption_strength_D', 'winner_stability_index_D',
            'control_solidity_index_D', 'main_force_ofi_D', 'retail_ofi_D', 'wash_trade_intensity_D',
            'closing_price_deviation_score_D'
        ]
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_pullback_confirmation"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        pullback_confirmation_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        n_pre_A = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pre_A'), 10)
        n_pullback_max = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pullback_max'), 13)
        open_D = self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_pullback_confirmation")
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_pullback_confirmation")
        high_D = self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_pullback_confirmation")
        low_D = self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_pullback_confirmation")
        pct_change_D = self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_pullback_confirmation")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma21_D = self._get_safe_series(df, 'VOL_MA_21_D', method_name="_diagnose_axiom_pullback_confirmation")
        main_force_net_flow_calibrated_D = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', method_name="_diagnose_axiom_pullback_confirmation")
        short_term_concentration_90pct_D_slope_5d = self._get_safe_series(df, 'SLOPE_5_winner_concentration_90pct_D', method_name="_diagnose_axiom_pullback_confirmation")
        large_order_pressure_D = self._get_safe_series(df, 'large_order_pressure_D', method_name="_diagnose_axiom_pullback_confirmation")
        large_order_support_D = self._get_safe_series(df, 'large_order_support_D', method_name="_diagnose_axiom_pullback_confirmation")
        hidden_accumulation_intensity_D = self._get_safe_series(df, 'hidden_accumulation_intensity_D', method_name="_diagnose_axiom_pullback_confirmation")
        absorption_strength_index_D = self._get_safe_series(df, 'dip_absorption_power_D', method_name="_diagnose_axiom_pullback_confirmation")
        upper_shadow_selling_pressure_D = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', method_name="_diagnose_axiom_pullback_confirmation")
        lower_shadow_absorption_strength_D = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', method_name="_diagnose_axiom_pullback_confirmation")
        winner_conviction_index_D = self._get_safe_series(df, 'winner_stability_index_D', method_name="_diagnose_axiom_pullback_confirmation")
        main_force_control_leverage_D = self._get_safe_series(df, 'control_solidity_index_D', method_name="_diagnose_axiom_pullback_confirmation")
        score_struct_axiom_trend_form = self._get_atomic_score(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', 0.0)
        main_force_ofi_D = self._get_safe_series(df, 'main_force_ofi_D', method_name="_diagnose_axiom_pullback_confirmation")
        retail_ofi_D = self._get_safe_series(df, 'retail_ofi_D', method_name="_diagnose_axiom_pullback_confirmation")
        wash_trade_intensity_D = self._get_safe_series(df, 'wash_trade_intensity_D', method_name="_diagnose_axiom_pullback_confirmation")
        closing_conviction_score_D = self._get_safe_series(df, 'closing_price_deviation_score_D', method_name="_diagnose_axiom_pullback_confirmation")
        max_vol_ma = pd.concat([vol_ma5_D, vol_ma21_D], axis=1).max(axis=1)
        effective_volume_D = volume_D * (1 - wash_trade_intensity_D.fillna(0).clip(0, 1))
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_target_date = None 
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_target_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
        for i in range(n_pre_A, len(df_index)):
            day_A_idx = i
            current_A_date = df_index[day_A_idx]
            pre_A_slice_start = day_A_idx - n_pre_A
            if pre_A_slice_start < 0:
                continue
            is_relevant_a_day_for_probe = False
            if probe_target_date is not None and current_A_date < probe_target_date:
                if (probe_target_date - current_A_date).days <= n_pullback_max + 1:
                    is_relevant_a_day_for_probe = True
            day_A_pct_change = pct_change_D.iloc[day_A_idx]
            day_A_volume = volume_D.iloc[day_A_idx]
            day_A_max_vol_ma = max_vol_ma.iloc[day_A_idx]
            day_A_main_force_flow = main_force_net_flow_calibrated_D.iloc[day_A_idx]
            day_A_chip_conc_slope = short_term_concentration_90pct_D_slope_5d.iloc[day_A_idx]
            day_A_close = close_D.iloc[day_A_idx]
            day_A_trend_form_score = score_struct_axiom_trend_form.iloc[day_A_idx]
            cond_A_pct_change = day_A_pct_change > 0.01
            cond_A_volume = day_A_volume > day_A_max_vol_ma * 1.2
            cond_A_mf_flow = day_A_main_force_flow > 0
            cond_A_chip_conc = day_A_chip_conc_slope > 0
            if not (cond_A_pct_change and cond_A_volume and cond_A_mf_flow and cond_A_chip_conc):
                continue
            for j in range(day_A_idx + 1, min(day_A_idx + n_pullback_max + 2, len(df_index))):
                day_B_idx = j
                if day_B_idx <= day_A_idx:
                    continue
                current_B_date = df_index[day_B_idx]
                is_probing_this_b_day = (probe_target_date is not None and current_B_date == probe_target_date)
                if not is_probing_this_b_day:
                    continue 
                day_B_pct_change = pct_change_D.iloc[day_B_idx]
                day_B_volume = volume_D.iloc[day_B_idx]
                day_B_max_vol_ma = max_vol_ma.iloc[day_B_idx]
                day_B_main_force_flow = main_force_net_flow_calibrated_D.iloc[day_B_idx]
                day_B_chip_conc_slope = short_term_concentration_90pct_D_slope_5d.iloc[day_B_idx]
                day_B_mf_control_leverage = main_force_control_leverage_D.iloc[day_B_idx]
                day_B_close = close_D.iloc[day_B_idx]
                day_B_trend_form_score = score_struct_axiom_trend_form.iloc[day_B_idx]
                cond_B_pct_change = day_B_pct_change > 0.01
                cond_B_volume = day_B_volume > day_B_max_vol_ma * 1.2
                cond_B_mf_flow = day_B_main_force_flow > 0
                cond_B_chip_conc = day_B_chip_conc_slope > 0
                cond_B_mf_leverage = day_B_mf_control_leverage > 0
                print(f"       - B日条件: pct_change > 1% ({cond_B_pct_change}), volume > 1.2*MA_VOL ({cond_B_volume}), MF_Flow > 0 ({cond_B_mf_flow}), Chip_Conc_Slope > 0 ({cond_B_chip_conc}), MF_Leverage > 0 ({cond_B_mf_leverage})")
                if not (cond_B_pct_change and cond_B_volume and cond_B_mf_flow and cond_B_chip_conc and cond_B_mf_leverage):
                    print(f"       - B日条件不满足，跳过。")
                    continue
                cond_B_close_higher_A = day_B_close > day_A_close
                cond_B_trend_better_A = day_B_trend_form_score > day_A_trend_form_score
                print(f"       - B日与A日关系: B_close > A_close ({cond_B_close_higher_A}), B_trend_score > A_trend_score ({cond_B_trend_better_A})")
                if not (cond_B_close_higher_A and cond_B_trend_better_A):
                    print(f"       - B日与A日关系不满足，跳过。")
                    continue
                pullback_slice_start = day_A_idx + 1
                pullback_slice_end = day_B_idx
                if pullback_slice_start >= pullback_slice_end:
                    print(f"       - 回调期长度不足，跳过。")
                    continue
                pullback_effective_volume = effective_volume_D.iloc[pullback_slice_start:pullback_slice_end] 
                pullback_raw_volume = volume_D.iloc[pullback_slice_start:pullback_slice_end] 
                pullback_max_vol_ma = max_vol_ma.iloc[pullback_slice_start:pullback_slice_end]
                pullback_pct_change = pct_change_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_main_force_ofi = main_force_ofi_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_retail_ofi = retail_ofi_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_hidden_accumulation = hidden_accumulation_intensity_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_large_order_support = large_order_support_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_large_order_pressure = large_order_pressure_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_lower_shadow_absorption = lower_shadow_absorption_strength_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_upper_shadow_selling_pressure = upper_shadow_selling_pressure_D.iloc[pullback_slice_start:pullback_slice_end]
                pullback_closing_conviction_score = closing_conviction_score_D.iloc[pullback_slice_start:pullback_slice_end]
                print(f"       - 回调期 ({df_index[pullback_slice_start].date()} to {df_index[pullback_slice_end-1].date()}):")
                for k in range(pullback_slice_start, pullback_slice_end):
                    pb_date = df_index[k].date()
                    pb_volume = volume_D.iloc[k]
                    pb_wash_intensity = wash_trade_intensity_D.iloc[k]
                    pb_effective_volume = effective_volume_D.iloc[k]
                    pb_max_vol_ma = max_vol_ma.iloc[k]
                    pb_pct_change = pct_change_D.iloc[k]
                    pb_is_shrunk_day = pb_effective_volume < pb_max_vol_ma
                    pb_is_churn_day = (pb_effective_volume >= pb_max_vol_ma * 0.8) and (pb_effective_volume < pb_max_vol_ma * 1.5)
                    pb_no_bearish_vol_breakout_day = not ((pb_pct_change < 0) and (pb_effective_volume > pb_max_vol_ma))
                    print(f"         - 日期 {pb_date}: 原始量: {pb_volume:.0f}, 对倒强度: {pb_wash_intensity:.4f}, 有效量: {pb_effective_volume:.0f}, 最大均量: {pb_max_vol_ma:.0f}")
                    print(f"           -> 有效量缩量 ({pb_is_shrunk_day}), 有效量放量换手 ({pb_is_churn_day}), 无放量下跌 ({pb_no_bearish_vol_breakout_day})")
                is_volume_shrunk = (pullback_effective_volume < pullback_max_vol_ma).all()
                is_volume_churn = (pullback_effective_volume >= pullback_max_vol_ma * 0.8).all() and (pullback_effective_volume < pullback_max_vol_ma * 1.5).all()
                no_bearish_volume_breakout = ~((pullback_pct_change < 0) & (pullback_effective_volume > pullback_max_vol_ma)).any()
                print(f"         - 回调期整体判断: 有效量能缩量 ({is_volume_shrunk}), 有效量能放量换手 ({is_volume_churn}), 无放量下跌 ({no_bearish_volume_breakout})")
                if not no_bearish_volume_breakout:
                    print(f"       - 回调期存在放量下跌，跳过。")
                    continue
                health_pullback_score = 0.0
                cond_mf_ofi_positive = not pullback_main_force_ofi.empty and (pullback_main_force_ofi > 0).all()
                if cond_mf_ofi_positive:
                    health_pullback_score += 0.2
                cond_retail_ofi_negative = not pullback_retail_ofi.empty and (pullback_retail_ofi < 0).all()
                if cond_retail_ofi_negative:
                    health_pullback_score += 0.2
                cond_hidden_acc_strong = not pullback_hidden_accumulation.empty and (pullback_hidden_accumulation > 0.5).any()
                if cond_hidden_acc_strong:
                    health_pullback_score += 0.1
                cond_large_order_support_strong = not pullback_large_order_support.empty and (pullback_large_order_support > 0.7).all()
                if cond_large_order_support_strong:
                    health_pullback_score += 0.1
                cond_large_order_pressure_weak = not pullback_large_order_pressure.empty and (pullback_large_order_pressure < 0.3).all()
                if cond_large_order_pressure_weak:
                    health_pullback_score += 0.1
                cond_lower_shadow_strong = not pullback_lower_shadow_absorption.empty and (pullback_lower_shadow_absorption > 0.5).any()
                if cond_lower_shadow_strong:
                    health_pullback_score += 0.1
                cond_upper_shadow_weak = not pullback_upper_shadow_selling_pressure.empty and (pullback_upper_shadow_selling_pressure < 0.3).all()
                if cond_upper_shadow_weak:
                    health_pullback_score += 0.1
                cond_closing_conviction_strong = not pullback_closing_conviction_score.empty and (pullback_closing_conviction_score > 0.5).all()
                if cond_closing_conviction_strong:
                    health_pullback_score += 0.1
                print(f"         - 健康洗盘分数: {health_pullback_score:.2f}")
                print(f"           - MF_OFI > 0 ({cond_mf_ofi_positive}), Retail_OFI < 0 ({cond_retail_ofi_negative}), Hidden_Acc > 0.5 ({cond_hidden_acc_strong})")
                print(f"           - Large_Order_Support > 0.7 ({cond_large_order_support_strong}), Large_Order_Pressure < 0.3 ({cond_large_order_pressure_weak})")
                print(f"           - Lower_Shadow_Abs > 0.5 ({cond_lower_shadow_strong}), Upper_Shadow_Sell < 0.3 ({cond_upper_shadow_weak}), Closing_Conviction > 0.5 ({cond_closing_conviction_strong})")
                is_healthy_pullback = False
                if is_volume_shrunk and health_pullback_score > 0.3: 
                    is_healthy_pullback = True
                elif is_volume_churn and health_pullback_score > 0.5: 
                    is_healthy_pullback = True
                print(f"         - 是否健康洗盘 ({is_healthy_pullback})")
                if is_healthy_pullback:
                    pullback_confirmation_score.iloc[day_B_idx] = 1.0
                    print(f"       - 成功匹配回踩确认二次启动形态！")
                    break
                else:
                    print(f"       - 健康洗盘条件不满足，跳过。")
        return pullback_confirmation_score.astype(np.float32)

    def _diagnose_axiom_duofangpao(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.1 · 信号校验增强版】形态公理五：诊断“多方炮”形态
        - 核心逻辑: 识别经典的“多方炮”K线组合。
        - 信号输出: 在形态的第三根K线日输出1.0；否则输出0.0。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        required_signals = ['open_D', 'close_D', 'high_D', 'low_D', 'volume_D', 'VOL_MA_5_D']
        if not self._validate_required_signals(df, required_signals, "_diagnose_axiom_duofangpao"):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        df_index = df.index
        duofangpao_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        open_D = self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_duofangpao")
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_duofangpao")
        high_D = self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_duofangpao")
        low_D = self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_duofangpao")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_duofangpao")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_duofangpao")
        for i in range(2, len(df_index)): 
            k1_open, k1_close, k1_high, k1_low, k1_volume = open_D.iloc[i-2], close_D.iloc[i-2], high_D.iloc[i-2], low_D.iloc[i-2], volume_D.iloc[i-2]
            k1_vol_ma5 = vol_ma5_D.iloc[i-2]
            k2_open, k2_close, k2_high, k2_low, k2_volume = open_D.iloc[i-1], close_D.iloc[i-1], high_D.iloc[i-1], low_D.iloc[i-1], volume_D.iloc[i-1]
            k2_vol_ma5 = vol_ma5_D.iloc[i-1]
            k3_open, k3_close, k3_high, k3_low, k3_volume = open_D.iloc[i], close_D.iloc[i], high_D.iloc[i], low_D.iloc[i], volume_D.iloc[i]
            k3_vol_ma5 = vol_ma5_D.iloc[i]
            cond1_price = k1_close > k1_open 
            cond1_volume = k1_volume > k1_vol_ma5 * 1.2 
            if not (cond1_price and cond1_volume):
                continue
            cond2_volume = k2_volume < k2_vol_ma5 * 0.8 
            cond2_body_small = abs(k2_close - k2_open) / (k2_high - k2_low + 1e-9) < 0.5 
            cond2_within_k1_range = k2_low >= k1_low and k2_high <= k1_high 
            if not (cond2_volume and cond2_body_small and cond2_within_k1_range):
                continue
            cond3_price = k3_close > k3_open 
            cond3_volume = k3_volume > k3_vol_ma5 * 1.2 
            cond3_close_higher_than_k1 = k3_close > k1_close 
            if cond3_price and cond3_volume and cond3_close_higher_than_k1:
                duofangpao_score.iloc[i] = 1.0
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_date_for_loop = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            if probe_date_for_loop is not None and probe_date_for_loop in df.index:
                print(f"    -> [多方炮探针] @ {probe_date_for_loop.date()}:")
                print(f"       - SCORE_PATTERN_DUOFANGPAO: {duofangpao_score.loc[probe_date_for_loop]:.4f}")
        return duofangpao_score.astype(np.float32)















