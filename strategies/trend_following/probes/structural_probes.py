# 文件: strategies/trend_following/intelligence/probes/structural_probes.py
import pandas as pd
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value

class StructuralProbes:
    """
    【探针模块】结构情报专属探针
    """
    def __init__(self, intel_layer):
        self.strategy = intel_layer.strategy
        self.structural_intel = intel_layer.structural_intel

    def _deploy_structural_health_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.3 · 动态周期组版】结构健康度探针
        - 核心修复: 废除了硬编码的周期分组逻辑。现在探针会动态地、精确地复制主引擎中
                      `period_groups` 的划分逻辑，确保在任何 `periods` 配置下都能正确重算。
        """
        # [代码修改开始]
        print("\n" + "="*25 + f" [结构探针] 正在启用 🏛️【结构健康度探针 V1.3】🏛️ " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.structural_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'SCORE_STRUCTURE_BULLISH_RESONANCE'
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终信号分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 整体健康度融合 (Overall Health Fusion)")
        overall_health = atomic.get('__STRUCTURE_overall_health', {})
        s_bull_overall = overall_health.get('s_bull', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {})
        # 精确复制主引擎的周期分组逻辑
        period_groups = {
            'short': [p for p in periods if p <= 5],
            'medium': [p for p in periods if 5 < p <= 21],
            'long': [p for p in periods if p > 21]
        }
        recalc_bullish_resonance = 0.0
        total_weight = sum(resonance_tf_weights.values())
        if total_weight > 0:
            for p_key, weight in resonance_tf_weights.items():
                # 使用动态生成的周期组
                group_periods = period_groups.get(p_key, [])
                group_scores = [get_val(s_bull_overall.get(p), probe_date) for p in group_periods if p in s_bull_overall]
                avg_group_score = np.mean(group_scores) if group_scores else 0.0
                recalc_bullish_resonance += avg_group_score * (weight / total_weight)
                print(f"    - [周期组 {p_key}] 平均健康度: {avg_group_score:.4f}, 权重贡献: {(avg_group_score * (weight / total_weight)):.4f}")
        print(f"    - 【探针重算看涨共振分】: {recalc_bullish_resonance:.4f}")
        # 最终裁决时，需要模拟 bipolar_to_exclusive_unipolar 的行为
        # 因为我们已经移除了阈值，所以直接比较即可
        final_recalc_score = recalc_bullish_resonance
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针重算值 {final_recalc_score:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, final_recalc_score) else '❌ 不一致'}")
        print("\n--- “结构健康度探针”解剖完毕 ---")
        # [代码修改结束]

    def _deploy_structural_pillar_fusion_probe(self, probe_date: pd.Timestamp, period: int = 13):
        """
        【探针 V1.1 · 结构势能同步版】结构支柱融合探针
        - 核心同步: 与 `StructuralIntelligence` V19.0 的重构保持一致。
          1. 将探针解剖的第四支柱从“形态确认”更新为“结构稳定性”。
          2. 调用新的 `_calculate_structural_stability_health` 方法进行重算。
        """
        # [代码修改开始]
        print("\n" + "="*25 + f" [结构探针] 正在启用 🔬【结构支柱融合探针 V1.1 (p={period})】🔬 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.structural_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        # 更新支柱名称列表
        pillar_names_in_order = ['trend_integrity', 'mtf_cohesion', 'breakout_potential', 'structural_stability']
        weights_in_order = [pillar_weights.get(name, 0.25) for name in pillar_names_in_order]
        periods_list = get_param_value(get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('periods'), [1, 5, 13, 21, 55])
        norm_window = 55
        print("\n  [链路层 1] 各支柱健康度重算 (Pillar Health Recalculation)")
        ti_s_bull, _, _, daily_bipolar = engine._calculate_trend_integrity_health(df, periods_list, norm_window)
        mtf_s_bull, _, _ = engine._calculate_mtf_cohesion_health(df, periods_list, norm_window, daily_bipolar)
        bp_s_bull, _, _ = engine._calculate_breakout_potential_health(df, periods_list, norm_window)
        # 调用新的支柱计算方法
        ss_s_bull, _, _ = engine._calculate_structural_stability_health(df, periods_list, norm_window)
        # 更新支柱健康度字典
        pillar_health_series = {
            'trend_integrity': ti_s_bull.get(period),
            'mtf_cohesion': mtf_s_bull.get(period),
            'breakout_potential': bp_s_bull.get(period),
            'structural_stability': ss_s_bull.get(period)
        }
        pillar_health_values = {name: get_val(series, probe_date) for name, series in pillar_health_series.items()}
        print("\n  [链路层 2] 融合过程解剖 (Fusion Process Dissection)")
        recalc_fused_score = 0.0
        total_weight = sum(weights_in_order)
        if total_weight > 0:
            for i, name in enumerate(pillar_names_in_order):
                score = pillar_health_values.get(name, 0.5)
                weight = weights_in_order[i]
                contribution = score * (weight / total_weight)
                recalc_fused_score += contribution
                # 更新打印的支柱名称
                print(f"    - [支柱: {name:<22}] 得分: {score:.4f} | 权重: {weight:.2f} | 贡献: {contribution:.4f}")
        else:
            scores = list(pillar_health_values.values())
            recalc_fused_score = np.mean(scores) if scores else 0.5
        print(f"    - 【探针重算融合分 (p={period})】: {recalc_fused_score:.4f}")
        print("\n  [链路层 3] 终极对质 (Final Verdict)")
        overall_health = atomic.get('__STRUCTURE_overall_health', {})
        s_bull_overall = overall_health.get('s_bull', {})
        system_fused_score = get_val(s_bull_overall.get(period), probe_date)
        print(f"    - [对比]: 系统值 {system_fused_score:.4f} vs. 探针重算值 {recalc_fused_score:.4f} -> {'✅ 一致' if np.isclose(system_fused_score, recalc_fused_score) else '❌ 不一致'}")
        print("\n--- “结构支柱融合探针”解剖完毕 ---")
        # [代码修改结束]











