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
        【探针 V2.0 · 过度拉伸版】穿透式解剖 COGNITIVE_RISK_PEAK_REJECTION 信号
        - 核心升级: 同步生产代码V6.0的“过度拉伸”逻辑，并增加对“动态熔断”的检查。
        """
        print("\n" + "="*25 + f" [微观探针] 正在启用 🏔️【高位遇阻风险探针 V2.0】🏔️ " + "="*25)
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

        # [代码修改开始]
        # --- 支柱一：战略位置 (重构为“过度拉伸”模型) ---
        print("\n    --- [支柱一: 战略位置 (过度拉伸模型)] ---")
        high_position_threshold = get_param_value(p_risk.get('high_position_threshold'), 0.7)
        ma55 = df.get('EMA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, np.nan)
        stretch_from_ma55_score_series = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        is_overextended_score_series = (stretch_from_ma55_score_series > high_position_threshold).astype(float)
        is_overextended_val = get_val(is_overextended_score_series, probe_date)
        print(f"      - 55日均线 (EMA_55_D): {get_val(ma55, probe_date):.2f}")
        print(f"      - 55日内高点: {get_val(rolling_high_55d, probe_date):.2f}")
        print(f"      - 近期平均波幅: {get_val(wave_channel_height, probe_date):.2f}")
        print(f"      - 当前价格拉伸度: {get_val(stretch_from_ma55_score_series, probe_date):.4f}")
        print(f"      - 高位区域阈值: {high_position_threshold:.2f}")
        print(f"      - 【支柱一得分 (是否过度拉伸)】: {is_overextended_val:.4f} ({'是' if is_overextended_val > 0 else '否'})")

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
        snapshot_score_series = is_overextended_score_series * rejection_quality_score_series * trend_suppression_factor_series
        snapshot_val = get_val(snapshot_score_series, probe_date)
        print(f"    - 计算: {is_overextended_val:.4f} (拉伸) * {rejection_quality_val:.4f} (抛压) * {trend_suppression_val:.4f} (抑制)")
        print(f"    - 【探针重算快照分】: {snapshot_val:.4f}")

        print("\n  [链路层 4] 关系元分析 (Relational Meta-Analysis)")
        # 检查动态熔断
        is_flat = snapshot_score_series.loc[:probe_date].tail(5).std() < 1e-6
        print(f"    - [熔断检查]: 快照分近期是否平坦? {'是' if is_flat else '否'}")
        if is_flat:
            print("      -> 触发动态熔断，元分析应输出中性分 0.5。")
        final_risk_score_series = self.intelligence_layer.cognitive_intel.micro_behavior_engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_score_series)
        final_risk_val = get_val(final_risk_score_series, probe_date)
        print(f"    - 【探针重算最终动态风险分】: {final_risk_val:.4f}")
        # [代码修改结束]

        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {system_score:.4f} vs. 探针重算值 {final_risk_val:.4f} -> {'✅ 逻辑闭环' if np.isclose(system_score, final_risk_val, atol=1e-4) else '❌ 存在偏差'}")
        print("\n--- “高位遇阻风险探针”解剖完毕 ---")

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))
