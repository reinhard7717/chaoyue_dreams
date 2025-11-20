import pandas as pd
import numpy as np
import json
from strategies.trend_following.utils import get_params_block, calculate_holographic_dynamics, get_param_value, normalize_score

class BehavioralProbes:
    """
    【探针模块】行为情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.behavioral_intel = intel_layer.behavioral_intel
    def _deploy_prometheus_torch_probe(self, probe_date: pd.Timestamp):
        """
        【V2.6 · 结构行为同步版】“普罗米修斯火炬”探针
        - 核心升级: 与主引擎 `_calculate_structural_behavior_health` V4.2 版完全同步。
                      在“链路层2”中，精确复刻了“日内质量分”协议，引入“净有效强度”。
        """
        print("\n" + "="*35 + f" [行为探针] 正在点燃 🔥【普罗米修斯火炬 · 行为引擎解剖 V2.6】🔥 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        print("\n  [链路层 1] 最终输出 (Final Output)")
        bull_res_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        bear_res_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BEARISH_RESONANCE'), probe_date, 0.0)
        bottom_rev_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_BOTTOM_REVERSAL'), probe_date, 0.0)
        top_rev_actual = get_val(atomic.get('SCORE_STRUCT_BEHAVIOR_TOP_REVERSAL'), probe_date, 0.0)
        print(f"    - 【看涨共振】: {bull_res_actual:.4f}")
        print(f"    - 【看跌共振】: {bear_res_actual:.4f}")
        print(f"    - 【底部反转】: {bottom_rev_actual:.4f}")
        print(f"    - 【顶部反转】: {top_rev_actual:.4f}")
        print("\n  [链路层 2] 复合状态构建 (来自 _calculate_structural_behavior_health)")
        gap_up = df['open_D'] > df['pre_close_D']
        body_up = df['close_D'] > df['open_D']
        trajectory_score = pd.Series(0.0, index=df.index)
        trajectory_score.loc[gap_up & body_up] = 1.0
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        lower_shadow = (np.minimum(df['open_D'], df['close_D']) - df['low_D']).clip(lower=0)
        shadow_modifier = ((lower_shadow - upper_shadow) / kline_range).fillna(0)
        day_quality_score = (trajectory_score * 0.7 + shadow_modifier * 0.3).clip(-1, 1)
        quality_adjustment_factor = (1 + day_quality_score) / 2
        positive_day_strength_raw = df['pct_change_D'].clip(0)
        net_effective_bullish_strength = (positive_day_strength_raw * 0.5) + (positive_day_strength_raw * quality_adjustment_factor * 0.5)
        positive_day_strength = normalize_score(net_effective_bullish_strength, df.index, norm_window) * (net_effective_bullish_strength > 0)
        print("\n    - [日内质量分协议计算]:")
        print(f"      - 日内质量分: {get_val(day_quality_score, probe_date):.4f}")
        print(f"      - 净有效看涨强度(归一化): {get_val(positive_day_strength, probe_date):.4f}")
        csi_score = get_val(normalize_score(df.get('closing_strength_index_D'), df.index, norm_window), probe_date)
        bull_div_score = get_val(normalize_score(df.get('flow_divergence_mf_vs_retail_D').clip(0), df.index, norm_window), probe_date)
        auction_power_score = get_val(normalize_score(df.get('final_hour_momentum_D').clip(0), df.index, norm_window), probe_date)
        trend_eff_score = get_val(normalize_score(df.get('intraday_trend_efficiency_D'), df.index, norm_window), probe_date)
        efficiency_holo_bull, _ = calculate_holographic_dynamics(df, 'intraday_trend_efficiency_D', norm_window)
        gini_holo_bull, _ = calculate_holographic_dynamics(df, 'intraday_volume_gini_D', norm_window)
        eff_holo_bull_val = get_val(efficiency_holo_bull, probe_date)
        gini_holo_bull_val = get_val(gini_holo_bull, probe_date)
        bullish_d_intensity = (eff_holo_bull_val + gini_holo_bull_val) / 2.0
        bullish_composite_state = (get_val(positive_day_strength, probe_date) * csi_score * (1 + bull_div_score) * auction_power_score * trend_eff_score * bullish_d_intensity)**(1/6)
        print(f"    - [看涨复合状态]: {bullish_composite_state:.4f}")
        print("\n--- “普罗米修斯火炬”探针解剖完毕 ---")
    def _deploy_pressure_transmutation_probe(self, probe_date: pd.Timestamp):
        """
        【V1.0】广义抛压嬗变探针
        - 核心使命: 深度解剖`behavioral_intelligence`模块内部从“识别广义抛压”到“嬗变为吸收反转机会”的全链路逻辑。
        """
        print("\n" + "="*35 + f" [行为探针] 正在启用 💎【广义抛压嬗变探针 V1.0】💎 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.behavioral_intel
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, np.ndarray)): return default
            if isinstance(series, np.ndarray):
                idx_loc = df.index.get_loc(date, method='nearest')
                return series[idx_loc] if idx_loc < len(series) else default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_parent = get_params_block(self.strategy, 'kline_pattern_params', {})
        p_reversal = get_params_block(p_parent, 'absorption_reversal_params', {})
        judgment_threshold = get_param_value(p_reversal.get('judgment_threshold'), 0.7)
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_risk = get_val(atomic.get('SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW'), probe_date, 0.0)
        actual_opp = get_val(atomic.get('SCORE_OPPORTUNITY_ABSORPTION_REVERSAL'), probe_date, 0.0)
        print(f"    - 【最终抛压风险】: {actual_risk:.4f}")
        print(f"    - 【吸收反转机会分】: {actual_opp:.4f}")
        print("\n  [链路层 2] 原始广义抛压风险重算 (Raw General Pressure)")
        day_quality_score = engine._calculate_day_quality_score(df)
        recalc_provisional_signals = engine._diagnose_kline_patterns(df, day_quality_score)
        recalc_provisional_pressure = get_val(recalc_provisional_signals.get('PROVISIONAL_GENERAL_PRESSURE_RISK'), probe_date, 0.0)
        actual_provisional_pressure = get_val(atomic.get('PROVISIONAL_GENERAL_PRESSURE_RISK'), probe_date, 0.0)
        print(f"    - 【探针重算原始抛压】: {recalc_provisional_pressure:.4f}")
        print(f"    - [内部验证]: 系统值 {actual_provisional_pressure:.4f} vs. 探针重算 {recalc_provisional_pressure:.4f} -> {'✅ 一致' if np.isclose(actual_provisional_pressure, recalc_provisional_pressure) else '❌ 不一致'}")
        print("\n  [链路层 3] 主力意图诊断重算 (Main Force Intent)")
        recalc_intent_signals = engine._diagnose_upper_shadow_intent(df)
        recalc_intent_diagnosis = get_val(recalc_intent_signals.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'), probe_date, 0.0)
        actual_intent_diagnosis = get_val(atomic.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'), probe_date, 0.0)
        print(f"    - 【探针重算意图诊断】: {recalc_intent_diagnosis:.4f}")
        print(f"    - [内部验证]: 系统值 {actual_intent_diagnosis:.4f} vs. 探针重算 {recalc_intent_diagnosis:.4f} -> {'✅ 一致' if np.isclose(actual_intent_diagnosis, recalc_intent_diagnosis) else '❌ 不一致'}")
        print("\n  [链路层 4] 抛压分析与机会嬗变 (Pressure Analysis & Opportunity Transmutation)")
        is_absorption_reversal = recalc_intent_diagnosis >= judgment_threshold
        print(f"    - [审判阈值]: {judgment_threshold:.2f}")
        print(f"    - [审判结果]: 意图分 {recalc_intent_diagnosis:.4f} >= {judgment_threshold:.2f} -> {'✅ 构成吸收反转' if is_absorption_reversal else '❌ 未构成吸收反转'}")
        if is_absorption_reversal:
            recalc_final_risk = 0.0
            recalc_opportunity = recalc_provisional_pressure * recalc_intent_diagnosis
        else:
            recalc_final_risk = recalc_provisional_pressure * (1 - recalc_intent_diagnosis)
            recalc_opportunity = 0.0
        recalc_final_risk = np.clip(recalc_final_risk, 0, 1)
        recalc_opportunity = np.clip(recalc_opportunity, 0, 1)
        print(f"    - 【探针重算最终风险】: {recalc_final_risk:.4f}")
        print(f"    - 【探针重算机会分】: {recalc_opportunity:.4f}")
        print("\n--- “广义抛压嬗变探针”解剖完毕 ---")
    def _deploy_liquidity_dynamics_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0】流动性动态探针
        - 核心职责: 深度解剖全新的双极性流动性信号，验证“锁仓惜售”和“流动性枯竭/恐慌抛售”的计算逻辑。
        """
        print("\n" + "="*25 + f" [行为探针] 正在启用 💧【流动性动态探针 V1.0】💧 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        norm_window = 55
        print("\n  [链路层 1] 最终系统输出 (Final System Outputs)")
        opp_score = get_val(atomic.get('SCORE_OPPORTUNITY_LOCKUP_RALLY'), probe_date)
        risk_score = get_val(atomic.get('SCORE_RISK_LIQUIDITY_DRAIN'), probe_date)
        bipolar_score = get_val(atomic.get('BEHAVIOR_BIPOLAR_LIQUIDITY_DYNAMICS'), probe_date)
        print(f"    - 【锁仓惜售机会分】: {opp_score:.4f}")
        print(f"    - 【流动性枯竭风险分】: {risk_score:.4f}")
        print(f"    - (内部)双极性动态分: {bipolar_score:.4f}")
        print("\n  [链路层 2] 原始证据值 (Raw Evidence Values)")
        raw_turnover = get_val(df.get('turnover_rate_D'), probe_date)
        raw_pct_change = get_val(df.get('pct_change_D'), probe_date)
        raw_vol = get_val(df.get('volume_D'), probe_date)
        raw_vol_ma5 = get_val(df.get('VOL_MA_5_D'), probe_date)
        raw_vol_ma21 = get_val(df.get('VOL_MA_21_D'), probe_date)
        raw_prev_vol = get_val(df.get('volume_D').shift(1), probe_date)
        print(f"    - [换手率]: {raw_turnover:.4f}%")
        print(f"    - [涨跌幅]: {raw_pct_change:.4f}%")
        print(f"    - [成交量]: {raw_vol:.0f}")
        print(f"    - [5日均量]: {raw_vol_ma5:.0f}")
        print(f"    - [21日均量]: {raw_vol_ma21:.0f}")
        print(f"    - [昨日成交量]: {raw_prev_vol:.0f}")
        print("\n  [链路层 3] 证据归一化与融合 (Evidence Normalization & Fusion)")
        low_turnover_score = get_val(normalize_score(df.get('turnover_rate_D'), df.index, norm_window, ascending=False), probe_date)
        vol_spike_cond1 = df['volume_D'] > np.maximum(df.get('VOL_MA_5_D', 0), df.get('VOL_MA_21_D', 0))
        vol_spike_cond2 = df['volume_D'] > (df['volume_D'].shift(1).fillna(0) * 2)
        volume_spike_score = get_val(normalize_score((vol_spike_cond1 | vol_spike_cond2).astype(float), df.index, norm_window, ascending=True), probe_date)
        day_direction = np.sign(raw_pct_change)
        print(f"    - [低换手率得分]: {low_turnover_score:.4f}")
        print(f"    - [成交量爆发得分]: {volume_spike_score:.4f}")
        print(f"    - [日内方向]: {'上涨' if day_direction > 0 else '下跌' if day_direction < 0 else '平盘'}")
        bullish_lockup_score = low_turnover_score * (1 if day_direction > 0 else 0)
        bearish_vacuum_score = low_turnover_score * (1 if day_direction < 0 else 0)
        bearish_panic_score = volume_spike_score * (1 if day_direction < 0 else 0)
        final_bearish_score = max(bearish_vacuum_score, bearish_panic_score)
        print(f"    - [机会面: 锁仓惜售]: {bullish_lockup_score:.4f} (低换手 * 上涨)")
        print(f"    - [风险面A: 无人问津]: {bearish_vacuum_score:.4f} (低换手 * 下跌)")
        print(f"    - [风险面B: 恐慌杀跌]: {bearish_panic_score:.4f} (成交量爆发 * 下跌)")
        print(f"    - [最终风险组件]: max({bearish_vacuum_score:.4f}, {bearish_panic_score:.4f}) = {final_bearish_score:.4f}")
        print("\n  [链路层 4] 终极对质 (Final Verdict)")
        recalc_bipolar = bullish_lockup_score - final_bearish_score
        recalc_opp = max(0, recalc_bipolar)
        recalc_risk = abs(min(0, recalc_bipolar))
        print(f"    - 【探针重算双极性分】: {bullish_lockup_score:.4f} - {final_bearish_score:.4f} = {recalc_bipolar:.4f}")
        print(f"    - [机会分对比]: 系统值 {opp_score:.4f} vs. 探针值 {recalc_opp:.4f} -> {'✅ 逻辑闭环' if np.isclose(opp_score, recalc_opp) else '❌ 不一致'}")
        print(f"    - [风险分对比]: 系统值 {risk_score:.4f} vs. 探针值 {recalc_risk:.4f} -> {'✅ 逻辑闭环' if np.isclose(risk_score, recalc_risk) else '❌ 不一致'}")
        print("\n--- “流动性动态探针”解剖完毕 ---")









