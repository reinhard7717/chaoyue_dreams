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

    def _deploy_peak_rejection_risk_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0 · 新增】穿透式解剖 COGNITIVE_RISK_PEAK_REJECTION 信号
        - 核心目标: 彻底查清为何在上涨日仍会产生高额风险分。
        """
        print("\n" + "="*25 + f" [微观探针] 正在启用 🏔️【高位遇阻风险探针 V1.0】🏔️ " + "="*25)
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        signal_name = 'COGNITIVE_RISK_PEAK_REJECTION'

        def get_val(series, date, default=0.0):
            val = series.get(date)
            return default if pd.isna(val) else val

        print(f"\n  [链路层 1] 最终系统输出 (Final System Output)")
        system_score = get_val(atomic_states.get(signal_name, pd.Series(0.0, index=df.index)), probe_date)
        print(f"    - 【最终信号分】: {system_score:.4f}")

        print("\n  [链路层 2] 证据链重算与剖析 (Evidence Chain Recalculation)")
        p_risk = get_params_block(self.strategy, 'post_peak_downturn_risk_params', {})
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})

        # --- 支柱一：战略位置 ---
        print("\n    --- [支柱一: 战略位置] ---")
        peak_proximity_threshold = get_param_value(p_risk.get('peak_proximity_threshold'), 0.95)
        price_to_peak_ratio = df.get('price_to_peak_ratio_D', pd.Series(0.0, index=df.index))
        is_near_peak_score_series = (price_to_peak_ratio > peak_proximity_threshold).astype(float)
        is_near_peak_val = get_val(is_near_peak_score_series, probe_date)
        print(f"      - 价格/近期高点比率 (price_to_peak_ratio_D): {get_val(price_to_peak_ratio, probe_date):.4f}")
        print(f"      - 高位区域阈值: {peak_proximity_threshold:.2f}")
        print(f"      - 【支柱一得分 (是否处于高位区域)】: {is_near_peak_val:.4f} ({'是' if is_near_peak_val > 0 else '否'})")

        # --- 支柱二：战术行为 ---
        print("\n    --- [支柱二: 战术行为] ---")
        rejection_quality_score_series = self._get_atomic_score(df, 'SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW', 0.0)
        rejection_quality_val = get_val(rejection_quality_score_series, probe_date)
        print(f"      - 【支柱二得分 (当日抛压质量)】: {rejection_quality_val:.4f}")

        # --- 安全阀：趋势健康度 ---
        print("\n    --- [安全阀: 趋势健康度] ---")
        ma_health_score_series = self.intelligence_layer.cognitive_intel.micro_behavior_engine._calculate_ma_health(df, p_conf, 55)
        ma_health_val = get_val(ma_health_score_series, probe_date)
        trend_suppression_factor_series = (1 - ma_health_score_series.clip(0, 1))
        trend_suppression_val = get_val(trend_suppression_factor_series, probe_date)
        print(f"      - MA健康度分数: {ma_health_val:.4f}")
        print(f"      - 【趋势抑制因子】: {trend_suppression_val:.4f}")

        print("\n  [链路层 3] 快照分融合 (Snapshot Fusion)")
        snapshot_score_series = is_near_peak_score_series * rejection_quality_score_series * trend_suppression_factor_series
        snapshot_val = get_val(snapshot_score_series, probe_date)
        print(f"    - 计算: {is_near_peak_val:.4f} (高位) * {rejection_quality_val:.4f} (抛压) * {trend_suppression_val:.4f} (抑制)")
        print(f"    - 【探针重算快照分】: {snapshot_val:.4f}")

        print("\n  [链路层 4] 关系元分析 (Relational Meta-Analysis)")
        final_risk_score_series = self.intelligence_layer.cognitive_intel.micro_behavior_engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_score_series)
        final_risk_val = get_val(final_risk_score_series, probe_date)
        print(f"    - 【探针重算最终动态风险分】: {final_risk_val:.4f}")

        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {system_score:.4f} vs. 探针重算值 {final_risk_val:.4f} -> {'✅ 逻辑闭环' if np.isclose(system_score, final_risk_val, atol=1e-4) else '❌ 存在偏差'}")
        print("\n--- “高位遇阻风险探针”解剖完毕 ---")

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))
