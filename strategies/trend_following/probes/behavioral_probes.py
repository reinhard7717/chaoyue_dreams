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

    def _deploy_liquidity_vacuum_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0 · 流动性真空版】穿透式解剖 SCORE_RISK_LIQUIDITY_VACUUM 信号
        - 核心职责: 验证“流动性真空”风险信号的三大支柱，确保其计算的准确性。
        """
        # [代码新增开始]
        print("\n" + "="*25 + f" [行为探针] 正在启用 🌀【流动性真空探针 V1.0】🌀 " + "="*25)
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        signal_name = 'SCORE_RISK_LIQUIDITY_VACUUM'
        
        def get_val(series, date, default=0.0):
            val = series.get(date)
            return default if pd.isna(val) else val

        # 1. 获取最终系统输出
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        system_score = get_val(atomic_states.get(signal_name, pd.Series(0.0, index=df.index)), probe_date)
        print(f"    - 【最终信号分】: {system_score:.4f}")

        # 2. 重算快照分
        print("\n  [链路层 2] 快照分重算 (Snapshot Recalculation)")
        p_atomic = get_params_block(self.strategy, 'price_volume_atomic_params', {})
        norm_window = get_param_value(p_atomic.get('norm_window'), 55)

        # 证据一: 交易深度不足 (低换手率)
        turnover_raw = df.get('turnover_rate_D', pd.Series(10.0, index=df.index))
        low_turnover_risk = normalize_score(turnover_raw, df.index, norm_window, ascending=False)
        
        # 证据二: 交易意愿低迷 (持续缩量)
        vol_vs_ma5 = df['volume_D'] / df.get('VOL_MA_5_D', df['volume_D'])
        vol_vs_ma55 = df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D'])
        sustained_shrink_raw = vol_vs_ma5.fillna(1.0) + vol_vs_ma55.fillna(1.0)
        sustained_shrink_risk = normalize_score(sustained_shrink_raw, df.index, norm_window, ascending=False)

        # 证据三: 市场脆弱性 (高日内波动)
        fragility_raw = df.get('intraday_volatility_D', pd.Series(0.0, index=df.index))
        fragility_risk = normalize_score(fragility_raw, df.index, norm_window, ascending=True)

        # 融合快照分
        probe_snapshot_score = (low_turnover_risk * sustained_shrink_risk * fragility_risk)**(1/3)
        probe_snapshot_val = get_val(probe_snapshot_score, probe_date)
        print(f"    - 【探针重算快照分】: {probe_snapshot_val:.4f}")

        # 3. 终极对质
        print("\n  [链路层 3] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {system_score:.4f} vs. 探针重算值 {probe_snapshot_val:.4f} -> {'✅ 一致' if np.isclose(system_score, probe_snapshot_val) else '❌ 不一致'}")

        # 4. 证据链分解
        print("\n  [链路层 4] 证据链分解 (Component Dissection)")
        print(f"    - [支柱一: 低换手率] 原始值: {get_val(turnover_raw, probe_date):.2f}%, 归一化风险分: {get_val(low_turnover_risk, probe_date):.4f}")
        print(f"    - [支柱二: 持续缩量] 原始值: {get_val(sustained_shrink_raw, probe_date):.2f}, 归一化风险分: {get_val(sustained_shrink_risk, probe_date):.4f}")
        print(f"    - [支柱三: 市场脆弱性] 原始值: {get_val(fragility_raw, probe_date):.2f}, 归一化风险分: {get_val(fragility_risk, probe_date):.4f}")
        
        print("\n--- “流动性真空探针”解剖完毕 ---")










