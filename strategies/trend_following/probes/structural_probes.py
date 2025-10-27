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
        【探针 V1.2 · 修复版】结构健康度探针
        - 核心修复: 1. 修正了对`_calculate_trend_integrity_health`的调用，以匹配其新的四元组返回值。
                      2. 修正了对`_calculate_mtf_cohesion_health`的调用，正确传递了`daily_bipolar_snapshot`。
        """
        # [代码修改开始]
        print("\n" + "="*25 + f" [结构探针] 正在启用 🏛️【结构健康度探针 V1.2】🏛️ " + "="*25)
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
        recalc_bullish_resonance = 0.0
        total_weight = sum(resonance_tf_weights.values())
        if total_weight > 0:
            for p_key, weight in resonance_tf_weights.items():
                period_group = periods[:2] if p_key == 'short' else (periods[2:4] if p_key == 'medium' else periods[4:])
                group_scores = [get_val(s_bull_overall.get(p), probe_date) for p in period_group if p in s_bull_overall]
                avg_group_score = np.mean(group_scores) if group_scores else 0.0
                recalc_bullish_resonance += avg_group_score * (weight / total_weight)
                print(f"    - [周期组 {p_key}] 平均健康度: {avg_group_score:.4f}, 权重贡献: {(avg_group_score * (weight / total_weight)):.4f}")
        print(f"    - 【探针重算看涨共振分】: {recalc_bullish_resonance:.4f}")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针重算值 {recalc_bullish_resonance:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_bullish_resonance) else '❌ 不一致'}")
        print("\n  [链路层 3] 四大支柱健康度解剖 (以 p=13 周期为例)")
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        pillar_names_in_order = ['trend_integrity', 'mtf_cohesion', 'breakout_potential', 'pattern_confirmation']
        norm_window = 55
        ti_s_bull, _, _, daily_bipolar_snapshot = engine._calculate_trend_integrity_health(df, periods, norm_window)
        mtf_s_bull, _, _ = engine._calculate_mtf_cohesion_health(df, periods, norm_window, daily_bipolar_snapshot)
        bp_s_bull, _, _ = engine._calculate_breakout_potential_health(df, periods, norm_window)
        pc_s_bull, _, _ = engine._calculate_pattern_health(df, periods, norm_window)
        pillar_bull_health = {
            'trend_integrity': ti_s_bull,
            'mtf_cohesion': mtf_s_bull,
            'breakout_potential': bp_s_bull,
            'pattern_confirmation': pc_s_bull
        }
        p = 13
        overall_health_p13 = get_val(s_bull_overall.get(p), probe_date)
        print(f"    - 整体健康度 (p=13): {overall_health_p13:.4f}")
        pillar_values_p13 = {}
        for name in pillar_names_in_order:
            pillar_values_p13[name] = get_val(pillar_bull_health[name].get(p), probe_date)
            print(f"      - [支柱: {name}] 健康度: {pillar_values_p13[name]:.4f}")
        print("\n  [链路层 4] 支柱融合重算 (p=13 周期)")
        valid_components = [(pillar_values_p13[name], pillar_weights.get(name, 0.25)) for name in pillar_names_in_order]
        components = [item[0] for item in valid_components]
        weights_for_period = [item[1] for item in valid_components]
        total_pillar_weight = sum(weights_for_period)
        if total_pillar_weight > 0:
            normalized_weights = np.array(weights_for_period) / total_pillar_weight
            safe_components = np.maximum(components, 1e-9)
            recalc_overall_health_p13 = np.exp(np.sum(normalized_weights * np.log(safe_components)))
        else:
            recalc_overall_health_p13 = np.mean(components) if components else 0.0
        print(f"    - [融合公式]: exp(sum(log(score) * weight))")
        print(f"    - 【探针重算整体健康度 (p=13)】: {recalc_overall_health_p13:.4f}")
        print(f"    - [对比]: 系统值 {overall_health_p13:.4f} vs. 探针重算值 {recalc_overall_health_p13:.4f} -> {'✅ 一致' if np.isclose(overall_health_p13, recalc_overall_health_p13) else '❌ 不一致'}")
        print("\n--- “结构健康度探针”解剖完毕 ---")
        # [代码修改结束]











