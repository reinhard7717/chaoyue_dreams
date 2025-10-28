# 文件: strategies/trend_following/probes/micro_behavior_probes.py
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar

class MicroBehaviorProbes:
    """
    【V1.0 · 新增】微观行为探针模块
    - 核心职责: 提供对 micro_behavior_engine.py 中复杂信号的穿透式解剖能力。
    """
    def __init__(self, intelligence_layer_instance):
        self.intelligence_layer = intelligence_layer_instance
        self.strategy = intelligence_layer_instance.strategy

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def _deploy_euphoric_acceleration_transmutation_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0 · 新增】亢奋加速嬗变探针
        - 核心职责: 深度解剖“亢奋加速”信号从一个中性的“亢奋事件”，
                      在“看涨上下文护盾”的调节下，最终嬗变为“风险”或“机会”的全过程。
        """
        print("\n" + "="*25 + f" [微观探针] 正在启用 ⚛️【亢奋加速嬗变探针 V1.0】⚛️ " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.intelligence_layer.cognitive_intel.micro_behavior_engine
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        # --- 链路层 1: 最终系统输出 ---
        print("\n  [链路层 1] 最终系统输出 (Final System Outputs)")
        actual_risk_score = get_val(atomic.get('COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'), probe_date, 0.0)
        actual_opp_score = get_val(atomic.get('COGNITIVE_OPPORTUNITY_IGNITION_ACCELERATION'), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_risk_score:.4f}")
        print(f"    - 【最终机会分】: {actual_opp_score:.4f}")
        # --- 链路层 2: 原始亢奋事件重算 ---
        print("\n  [链路层 2] 原始亢奋事件探测 (Raw Euphoric Event Detection)")
        # 探针中简化重算过程，直接获取最终的原始分
        periods = [5, 13, 21, 55]
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        total_weight = sum(tf_weights.values())
        recalc_raw_euphoric_score = 0.0
        # 此处为了探针清晰，我们不完全重算，而是假设能拿到中间结果
        # 在真实场景中，可能需要更复杂的重算或日志记录
        # 这里我们用一个近似值或最终值来代表
        # 假设 dynamic_raw_euphoric_score 能够被某种方式获取
        # 为了演示，我们从最终结果反推
        if (actual_risk_score + actual_opp_score) > 1e-6:
             recalc_raw_euphoric_score = actual_risk_score + actual_opp_score
        print(f"    - 【探针重算-原始亢奋事件分】: {recalc_raw_euphoric_score:.4f} (此为风险与机会之和)")
        # --- 链路层 3: 看涨上下文护盾解剖 (Bullish Context Shield Dissection) ---
        print("\n  [链路层 3] 看涨上下文护盾解剖 (Bullish Context Shield Dissection)")
        # 护盾支柱 I: 深度底部区域
        bottom_zone_series = self._get_atomic_score(df, 'SCORE_CONTEXT_DEEP_BOTTOM_ZONE', 0.0)
        bottom_zone_val = get_val(bottom_zone_series, probe_date)
        print(f"    - [护盾支柱 I: 深度底部] -> 得分: {bottom_zone_val:.4f}")
        # 护盾支柱 II: 筹码吸筹锁仓
        chip_lockdown_series = self._get_atomic_score(df, 'SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN', 0.0)
        chip_lockdown_val = get_val(chip_lockdown_series, probe_date)
        print(f"    - [护盾支柱 II: 筹码锁仓] -> 得分: {chip_lockdown_val:.4f}")
        # 护盾支柱 III: 赢家信念
        winner_conviction_raw = self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION', 0.0)
        winner_conviction_val_raw = get_val(winner_conviction_raw, probe_date)
        winner_conviction_val = (np.clip(winner_conviction_val_raw, -1, 1) * 0.5 + 0.5)
        print(f"    - [护盾支柱 III: 赢家信念] -> 原始值: {winner_conviction_val_raw:.4f}, 归一化后: {winner_conviction_val:.4f}")
        # 融合护盾
        recalc_shield_score = (bottom_zone_val * chip_lockdown_val * winner_conviction_val)**(1/3)
        print(f"    - 【探针重算-护盾总分】: ({bottom_zone_val:.2f} * {chip_lockdown_val:.2f} * {winner_conviction_val:.2f})^(1/3) = {recalc_shield_score:.4f}")
        # --- 链路层 4: 最终嬗变裁决 (Final Transmutation Adjudication) ---
        print("\n  [链路层 4] 最终嬗变裁决 (Final Transmutation Adjudication)")
        recalc_risk_score = (recalc_raw_euphoric_score * (1 - recalc_shield_score)).clip(0, 1)
        recalc_opp_score = (recalc_raw_euphoric_score * recalc_shield_score).clip(0, 1)
        print(f"    - 【探针重算-风险分】: {recalc_raw_euphoric_score:.4f} * (1 - {recalc_shield_score:.4f}) = {recalc_risk_score:.4f}")
        print(f"    - 【探针重算-机会分】: {recalc_raw_euphoric_score:.4f} * {recalc_shield_score:.4f} = {recalc_opp_score:.4f}")
        # --- 链路层 5: 终极对质 ---
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        risk_match = np.isclose(actual_risk_score, recalc_risk_score)
        opp_match = np.isclose(actual_opp_score, recalc_opp_score)
        print(f"    - [风险分对比]: 系统值 {actual_risk_score:.4f} vs. 探针值 {recalc_risk_score:.4f} -> {'✅ 一致' if risk_match else '❌ 不一致'}")
        print(f"    - [机会分对比]: 系统值 {actual_opp_score:.4f} vs. 探针值 {recalc_opp_score:.4f} -> {'✅ 一致' if opp_match else '❌ 不一致'}")
        print("\n--- “亢奋加速嬗变探针”解剖完毕 ---")

    def _deploy_profit_taking_pressure_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.0 · 燃料转化版】利润兑现压力风险探针
        - 核心升级: 深度解剖“燃料转化”逻辑，展示风险如何被削弱，以及机会如何被增强。
        """
        print("\n" + "="*25 + f" [微观探针] 正在启用 💸【利润兑现压力探针 V2.0】💸 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.intelligence_layer.cognitive_intel.micro_behavior_engine
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_PROFIT_TAKING_PRESSURE'
        print("\n  [链路层 1] 最终系统输出 (Final System Outputs)")
        actual_final_risk = get_val(atomic.get(signal_name), probe_date, 0.0)
        actual_absorption_opp = get_val(atomic.get('SCORE_OPPORTUNITY_ABSORPTION_REVERSAL'), probe_date, 0.0)
        actual_lockdown_opp = get_val(atomic.get('SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN'), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_risk:.4f}")
        print(f"    - 【最终吸收机会分】: {actual_absorption_opp:.4f}")
        print(f"    - 【最终锁仓机会分】: {actual_lockdown_opp:.4f}")
        print("\n  [链路层 2] 原始风险分重算 (Raw Risk Recalculation)")
        periods = [1, 5, 13, 21, 55]
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        recalc_raw_fused_series = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        if total_weight > 0:
            for p_tactical in periods:
                p_context = periods[periods.index(p_tactical) + 1] if periods.index(p_tactical) + 1 < len(periods) else p_tactical
                urgency_q = engine._calculate_4d_metric_quality(df, 'profit_taking_urgency_D', p_tactical, p_context, ascending=True)
                premium_q = engine._calculate_4d_metric_quality(df, 'profit_realization_premium_D', p_tactical, p_context, ascending=True)
                snapshot_s = (urgency_q * premium_q)**0.5
                period_score = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_s)
                weight = tf_weights.get(p_tactical, 0) / total_weight
                recalc_raw_fused_series += period_score * weight
        recalc_raw_fused_score = get_val(recalc_raw_fused_series, probe_date, 0.0)
        print(f"    - 【探针重算-原始风险分】: {recalc_raw_fused_score:.4f}")
        print("\n  [链路层 3] 燃料转化机制 (Fuel Transmutation Mechanism)")
        # 获取转化前的原始机会分
        # 注意：这里我们无法直接获取转化前的原始值，只能从最终值反推，这在探针中是可接受的近似
        # 假设燃料转化是最后一步，我们可以从最终值反推
        absorption_opp_val = get_val(atomic.get('SCORE_OPPORTUNITY_ABSORPTION_REVERSAL'), probe_date, 0.0)
        lockdown_opp_val = get_val(atomic.get('SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN'), probe_date, 0.0)
        print(f"    - [护盾 I: 吸收反转机会] -> 原始得分: {absorption_opp_val:.4f}")
        print(f"    - [护盾 II: 底部锁仓机会] -> 原始得分: {lockdown_opp_val:.4f}")
        recalc_shield_score = max(absorption_opp_val, lockdown_opp_val)
        print(f"    - 【探针重算-抑制护盾分】: max({absorption_opp_val:.4f}, {lockdown_opp_val:.4f}) = {recalc_shield_score:.4f}")
        recalc_fuel_generated = recalc_raw_fused_score * recalc_shield_score
        print(f"    - 【探针重算-转化燃料量】: {recalc_raw_fused_score:.4f} * {recalc_shield_score:.4f} = {recalc_fuel_generated:.4f}")
        print("\n  [链路层 4] 最终裁决与对质 (Final Adjudication & Verdict)")
        recalc_final_risk = recalc_raw_fused_score * (1.0 - recalc_shield_score)
        print(f"    - 【探针重算-最终风险分】: {recalc_raw_fused_score:.4f} * (1.0 - {recalc_shield_score:.4f}) = {recalc_final_risk:.4f}")
        total_opp_strength = absorption_opp_val + lockdown_opp_val
        safe_total_opp_strength = 1.0 if total_opp_strength == 0 else total_opp_strength
        absorption_weight = absorption_opp_val / safe_total_opp_strength
        lockdown_weight = lockdown_opp_val / safe_total_opp_strength
        recalc_enhanced_absorption = (absorption_opp_val + recalc_fuel_generated * absorption_weight).clip(0, 1)
        recalc_enhanced_lockdown = (lockdown_opp_val + recalc_fuel_generated * lockdown_weight).clip(0, 1)
        print(f"    - 【探针重算-增强后吸收机会】: {absorption_opp_val:.4f} + {recalc_fuel_generated:.4f} * {absorption_weight:.2f} = {recalc_enhanced_absorption:.4f}")
        print(f"    - 【探针重算-增强后锁仓机会】: {lockdown_opp_val:.4f} + {recalc_fuel_generated:.4f} * {lockdown_weight:.2f} = {recalc_enhanced_lockdown:.4f}")
        risk_match = np.isclose(actual_final_risk, recalc_final_risk)
        absorption_match = np.isclose(actual_absorption_opp, recalc_enhanced_absorption)
        lockdown_match = np.isclose(actual_lockdown_opp, recalc_enhanced_lockdown)
        print(f"    - [风险分对比]: 系统值 {actual_final_risk:.4f} vs. 探针值 {recalc_final_risk:.4f} -> {'✅ 一致' if risk_match else '❌ 不一致'}")
        print(f"    - [吸收机会对比]: 系统值 {actual_absorption_opp:.4f} vs. 探针值 {recalc_enhanced_absorption:.4f} -> {'✅ 一致' if absorption_match else '❌ 不一致'}")
        print(f"    - [锁仓机会对比]: 系统值 {actual_lockdown_opp:.4f} vs. 探针值 {recalc_enhanced_lockdown:.4f} -> {'✅ 一致' if lockdown_match else '❌ 不一致'}")
        print("\n--- “利润兑现压力探针”解剖完毕 ---")















