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
        【V3.1 · 清洁与多时间维度归一化版】形态公理一：诊断“背离”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `divergence_score` 的归一化方式改为多时间维度自适应归一化。
        """
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
        【V3.1 · 清洁与多时间维度归一化版】形态公理二：诊断“反转”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `reversal_score` 的归一化方式改为多时间维度自适应归一化。
        """
        raw_reversal_score = self._get_safe_series(df, 'counterparty_exhaustion_index_D', 0, method_name="_diagnose_axiom_reversal")
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        tf_weights_pattern = get_param_value(p_conf_pattern.get('tf_fusion_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
        reversal_score = get_adaptive_mtf_normalized_bipolar_score(raw_reversal_score, df.index, tf_weights_pattern)
        return reversal_score.astype(np.float32)

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.1 · 清洁与多时间维度归一化版】形态公理三：诊断“突破”
        - 核心修复: 增加对所有依赖数据的存在性检查。
        - 【优化】将 `breakout_score` 的归一化方式改为多时间维度自适应归一化。
        """
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
        【V5.5 · 纯粹量能洗盘识别版 - 探针增强 & 修复数据缺失处理】形态公理四：诊断“回踩确认二次启动”形态
        - 核心逻辑: 识别股价在量能萎缩后放量突破，随后缩量回调或放量换手洗盘，再放量突破的二次启动形态。
                    回调阶段融入微观资金流、筹码结构、行为效率等高级指标，量化“健康洗盘”特征。
                    新增B点收盘价高于A点收盘价，以及B点结构趋势相对于A点向上的判断。
                    在回调阶段，引入“主力对倒强度”来计算“有效成交量”，以更准确地判断量能的纯粹性。
        - 信号输出: 在形态的“二次启动日”（B日）输出1.0；否则输出0.0。
        - 核心修复: 移除冗余的required_cols检查，依赖_get_safe_series提供默认值。修正SLOPE列名。
        - 探针增强: 增加详细的print探针，用于调试和验证关键逻辑，探针输出集中在目标B日，并细化回调期量能探针。
        """
        df_index = df.index
        pullback_confirmation_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        p_conf_pattern = get_params_block(self.strategy, 'pattern_params', {})
        n_pre_A = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pre_A'), 10)
        n_pullback_max = get_param_value(p_conf_pattern.get('pullback_confirmation_n_pullback_max'), 13)
        # 获取核心K线数据
        open_D = self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_pullback_confirmation")
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_pullback_confirmation")
        high_D = self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_pullback_confirmation")
        low_D = self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_pullback_confirmation")
        pct_change_D = self._get_safe_series(df, 'pct_change_D', method_name="_diagnose_axiom_pullback_confirmation")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_pullback_confirmation")
        vol_ma21_D = self._get_safe_series(df, 'VOL_MA_21_D', method_name="_diagnose_axiom_pullback_confirmation")
        # 获取高级指标
        main_force_net_flow_calibrated_D = self._get_safe_series(df, 'main_force_net_flow_calibrated_D', method_name="_diagnose_axiom_pullback_confirmation")
        # [代码修改开始] 使用 SLOPE_5_winner_concentration_90pct_D 替代 SLOPE_5_short_term_concentration_90pct_D
        short_term_concentration_90pct_D_slope_5d = self._get_safe_series(df, 'SLOPE_5_winner_concentration_90pct_D', method_name="_diagnose_axiom_pullback_confirmation")
        large_order_pressure_D = self._get_safe_series(df, 'large_order_pressure_D', method_name="_diagnose_axiom_pullback_confirmation")
        large_order_support_D = self._get_safe_series(df, 'large_order_support_D', method_name="_diagnose_axiom_pullback_confirmation")
        hidden_accumulation_intensity_D = self._get_safe_series(df, 'hidden_accumulation_intensity_D', method_name="_diagnose_axiom_pullback_confirmation")
        # [代码修改开始] 使用 dip_absorption_power_D 替代 absorption_strength_index_D
        absorption_strength_index_D = self._get_safe_series(df, 'dip_absorption_power_D', method_name="_diagnose_axiom_pullback_confirmation")
        upper_shadow_selling_pressure_D = self._get_safe_series(df, 'upper_shadow_selling_pressure_D', method_name="_diagnose_axiom_pullback_confirmation")
        lower_shadow_absorption_strength_D = self._get_safe_series(df, 'lower_shadow_absorption_strength_D', method_name="_diagnose_axiom_pullback_confirmation")
        # [代码修改开始] 使用 winner_stability_index_D 替代 winner_conviction_index_D
        winner_conviction_index_D = self._get_safe_series(df, 'winner_stability_index_D', method_name="_diagnose_axiom_pullback_confirmation")
        # [代码修改开始] 使用 control_solidity_index_D 替代 main_force_control_leverage_D
        main_force_control_leverage_D = self._get_safe_series(df, 'control_solidity_index_D', method_name="_diagnose_axiom_pullback_confirmation")
        score_struct_axiom_trend_form = self._get_safe_series(df, 'SCORE_STRUCT_AXIOM_TREND_FORM', method_name="_diagnose_axiom_pullback_confirmation")
        main_force_ofi_D = self._get_safe_series(df, 'main_force_ofi_D', method_name="_diagnose_axiom_pullback_confirmation")
        retail_ofi_D = self._get_safe_series(df, 'retail_ofi_D', method_name="_diagnose_axiom_pullback_confirmation")
        wash_trade_intensity_D = self._get_safe_series(df, 'wash_trade_intensity_D', method_name="_diagnose_axiom_pullback_confirmation")
        # [代码修改开始] 使用 closing_price_deviation_score_D 替代 closing_conviction_score_D
        closing_conviction_score_D = self._get_safe_series(df, 'closing_price_deviation_score_D', method_name="_diagnose_axiom_pullback_confirmation")
        # [代码修改结束]
        max_vol_ma = pd.concat([vol_ma5_D, vol_ma21_D], axis=1).max(axis=1)
        # 计算有效成交量 (纯粹量能)
        effective_volume_D = volume_D * (1 - wash_trade_intensity_D.fillna(0).clip(0, 1))
        # 调试探针配置
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_target_date = None # 目标B日，当此日期为B日时，打印详细探针
        if probe_dates_str:
            probe_date_naive = pd.to_datetime(probe_dates_str[0])
            probe_target_date = probe_date_naive.tz_localize(df_index.tz) if df_index.tz else probe_date_naive
            # 新增行: 打印当前DataFrame的所有列名，仅在探针日期存在时
            if probe_target_date is not None and probe_target_date in df.index:
                print(f"    -> [回踩确认二次启动探针] @ {probe_target_date.date()}: 当前DataFrame包含的列: {df.columns.tolist()}")
        for i in range(n_pre_A, len(df_index)):
            day_A_idx = i
            current_A_date = df_index[day_A_idx]
            pre_A_slice_start = day_A_idx - n_pre_A
            if pre_A_slice_start < 0:
                continue
            # 确定当前A日是否与目标B日相关，以便打印探针信息
            is_relevant_a_day_for_probe = False
            if probe_target_date is not None and current_A_date < probe_target_date:
                # A日必须在目标B日之前，且在B日的回溯窗口内 (day_A_idx + 1 to day_B_idx)
                # max_pullback_days = n_pullback_max + 1 (inclusive of B-day)
                if (probe_target_date - current_A_date).days <= n_pullback_max + 1:
                    is_relevant_a_day_for_probe = True
            # 1. A日之前N天量能萎缩 (使用原始量能)
            pre_A_volume_atrophy = (volume_D.iloc[pre_A_slice_start:day_A_idx] < max_vol_ma.iloc[pre_A_slice_start:day_A_idx]).all()
            if is_relevant_a_day_for_probe:
                print(f"    -> [回踩确认二次启动探针] 目标B日为 {probe_target_date.date()}，正在检查候选A日 @ {current_A_date.date()}:")
                print(f"       - A日之前 {n_pre_A} 天量能萎缩 ({pre_A_volume_atrophy})")
            if not pre_A_volume_atrophy:
                if is_relevant_a_day_for_probe:
                    print(f"       - A日条件: 量能萎缩不满足，跳过此A日。")
                continue
            # 2. A日放量上涨突破 (强化A日条件，使用原始量能)
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
            if is_relevant_a_day_for_probe:
                print(f"       - A日条件: pct_change > 1% ({cond_A_pct_change}), volume > 1.2*MA_VOL ({cond_A_volume}), MF_Flow > 0 ({cond_A_mf_flow}), Chip_Conc_Slope > 0 ({cond_A_chip_conc})")
            if not (cond_A_pct_change and cond_A_volume and cond_A_mf_flow and cond_A_chip_conc):
                if is_relevant_a_day_for_probe:
                    print(f"       - A日条件: 放量上涨突破不满足，跳过此A日。")
                continue
            # 3. 寻找B日
            for j in range(day_A_idx + 1, min(day_A_idx + n_pullback_max + 2, len(df_index))):
                day_B_idx = j
                if day_B_idx <= day_A_idx:
                    continue
                current_B_date = df_index[day_B_idx]
                is_probing_this_b_day = (probe_target_date is not None and current_B_date == probe_target_date)
                if not is_probing_this_b_day:
                    continue # 只有当当前B日是目标探针日期时，才进行详细打印
                # 如果执行到这里，说明 current_B_date 是 probe_target_date，并且 A日条件已通过。
                print(f"    -> [回踩确认二次启动探针] 正在尝试匹配模式，目标B日 @ {current_B_date.date()}:")
                print(f"       - 候选A日 @ {df_index[day_A_idx].date()} (已通过A日条件检查):")
                # 4. B日再次放量上涨突破 (强化B日条件，使用原始量能)
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
                # 5. 细化B点与A点的关系：B点收盘高于A点收盘，且B点结构趋势相对于A点向上
                cond_B_close_higher_A = day_B_close > day_A_close
                cond_B_trend_better_A = day_B_trend_form_score > day_A_trend_form_score
                print(f"       - B日与A日关系: B_close > A_close ({cond_B_close_higher_A}), B_trend_score > A_trend_score ({cond_B_trend_better_A})")
                if not (cond_B_close_higher_A and cond_B_trend_better_A):
                    print(f"       - B日与A日关系不满足，跳过。")
                    continue
                # 6. A到B日之间缩量回调或放量换手洗盘，且健康洗盘
                pullback_slice_start = day_A_idx + 1
                pullback_slice_end = day_B_idx
                if pullback_slice_start >= pullback_slice_end:
                    print(f"       - 回调期长度不足，跳过。")
                    continue
                # 获取回调期数据
                pullback_effective_volume = effective_volume_D.iloc[pullback_slice_start:pullback_slice_end] # 使用有效成交量
                pullback_raw_volume = volume_D.iloc[pullback_slice_start:pullback_slice_end] # 原始量能用于MA比较
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
                # 基础回调条件：缩量或放量换手，且无放量下跌
                is_volume_shrunk = (pullback_effective_volume < pullback_max_vol_ma).all()
                is_volume_churn = (pullback_effective_volume >= pullback_max_vol_ma * 0.8).all() and (pullback_effective_volume < pullback_max_vol_ma * 1.5).all()
                no_bearish_volume_breakout = ~((pullback_pct_change < 0) & (pullback_effective_volume > pullback_max_vol_ma)).any()
                print(f"         - 回调期整体判断: 有效量能缩量 ({is_volume_shrunk}), 有效量能放量换手 ({is_volume_churn}), 无放量下跌 ({no_bearish_volume_breakout})")
                if not no_bearish_volume_breakout:
                    print(f"       - 回调期存在放量下跌，跳过。")
                    continue
                # 深度博弈：健康洗盘特征
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
                # 判断是缩量洗盘还是放量换手洗盘
                is_healthy_pullback = False
                if is_volume_shrunk and health_pullback_score > 0.3: # 缩量洗盘，健康分数要求低一些
                    is_healthy_pullback = True
                elif is_volume_churn and health_pullback_score > 0.5: # 放量换手洗盘，健康分数要求高一些
                    is_healthy_pullback = True
                print(f"         - 是否健康洗盘 ({is_healthy_pullback})")
                if is_healthy_pullback:
                    pullback_confirmation_score.iloc[day_B_idx] = 1.0
                    print(f"       - 成功匹配回踩确认二次启动形态！")
                    break
                else:
                    print(f"       - 健康洗盘条件不满足，跳过。")
        if probe_target_date is not None and probe_target_date in df.index:
            print(f"    -> [回踩确认二次启动探针] 最终结果 @ {probe_target_date.date()}:")
            print(f"       - SCORE_PATTERN_PULLBACK_CONFIRMATION: {pullback_confirmation_score.loc[probe_target_date]:.4f}")
        return pullback_confirmation_score.astype(np.float32)

    def _diagnose_axiom_duofangpao(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0】形态公理五：诊断“多方炮”形态
        - 核心逻辑: 识别经典的“多方炮”K线组合。
        - 信号输出: 在形态的第三根K线日输出1.0；否则输出0.0。
        - 核心修复: 增加对所有依赖数据的存在性检查。
        """
        df_index = df.index
        duofangpao_score = pd.Series(0.0, index=df_index, dtype=np.float32)
        # 获取核心K线数据
        open_D = self._get_safe_series(df, 'open_D', method_name="_diagnose_axiom_duofangpao")
        close_D = self._get_safe_series(df, 'close_D', method_name="_diagnose_axiom_duofangpao")
        high_D = self._get_safe_series(df, 'high_D', method_name="_diagnose_axiom_duofangpao")
        low_D = self._get_safe_series(df, 'low_D', method_name="_diagnose_axiom_duofangpao")
        volume_D = self._get_safe_series(df, 'volume_D', method_name="_diagnose_axiom_duofangpao")
        vol_ma5_D = self._get_safe_series(df, 'VOL_MA_5_D', method_name="_diagnose_axiom_duofangpao")
        # 检查所有必要的列是否存在
        required_cols = ['open_D', 'close_D', 'high_D', 'low_D', 'volume_D', 'VOL_MA_5_D']
        for col in required_cols:
            if col not in df.columns:
                print(f"    -> [形态情报警告] 缺少核心K线数据 '{col}'，无法诊断多方炮形态。")
                return duofangpao_score
        for i in range(2, len(df_index)): # 从第三根K线开始遍历
            # K线1 (i-2日)
            k1_open, k1_close, k1_high, k1_low, k1_volume = open_D.iloc[i-2], close_D.iloc[i-2], high_D.iloc[i-2], low_D.iloc[i-2], volume_D.iloc[i-2]
            k1_vol_ma5 = vol_ma5_D.iloc[i-2]
            # K线2 (i-1日)
            k2_open, k2_close, k2_high, k2_low, k2_volume = open_D.iloc[i-1], close_D.iloc[i-1], high_D.iloc[i-1], low_D.iloc[i-1], volume_D.iloc[i-1]
            k2_vol_ma5 = vol_ma5_D.iloc[i-1]
            # K线3 (i日)
            k3_open, k3_close, k3_high, k3_low, k3_volume = open_D.iloc[i], close_D.iloc[i], high_D.iloc[i], low_D.iloc[i], volume_D.iloc[i]
            k3_vol_ma5 = vol_ma5_D.iloc[i]
            # 条件1: K线1为放量阳线
            cond1_price = k1_close > k1_open # 阳线
            cond1_volume = k1_volume > k1_vol_ma5 * 1.2 # 放量 (1.2倍均量)
            if not (cond1_price and cond1_volume):
                continue
            # 条件2: K线2为缩量阴线或小实体K线，且在K线1范围内
            cond2_volume = k2_volume < k2_vol_ma5 * 0.8 # 缩量 (0.8倍均量)
            cond2_body_small = abs(k2_close - k2_open) / (k2_high - k2_low + 1e-9) < 0.5 # 小实体 (实体小于K线总长度的50%)
            cond2_within_k1_range = k2_low >= k1_low and k2_high <= k1_high # 在K线1范围内
            if not (cond2_volume and cond2_body_small and cond2_within_k1_range):
                continue
            # 条件3: K线3为放量阳线，且收盘价高于K线1收盘价
            cond3_price = k3_close > k3_open # 阳线
            cond3_volume = k3_volume > k3_vol_ma5 * 1.2 # 放量 (1.2倍均量)
            cond3_close_higher_than_k1 = k3_close > k1_close # 收盘价高于K线1收盘价
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















