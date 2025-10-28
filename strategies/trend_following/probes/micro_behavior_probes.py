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
















