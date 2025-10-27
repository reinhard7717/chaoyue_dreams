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
        print("\n--- “结构健康度探针”解剖完毕 ---")

    def _deploy_structural_pillar_fusion_probe(self, probe_date: pd.Timestamp, period: int = 13):
        """
        【探针 V1.0 · 新增】结构支柱融合探针
        - 核心使命: 深度解剖四大支柱是如何融合成最终的`overall_health`分数的。
        - 解剖链路: 1. 重算各支柱健康度 -> 2. 获取权重 -> 3. 模拟加权算术平均融合 -> 4. 对比系统值
        """
        # [代码新增开始]
        print("\n" + "="*25 + f" [结构探针] 正在启用 🔬【结构支柱融合探针 V1.0 (p={period})】🔬 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.structural_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        pillar_names_in_order = ['trend_integrity', 'mtf_cohesion', 'breakout_potential', 'pattern_confirmation']
        weights_in_order = [pillar_weights.get(name, 0.25) for name in pillar_names_in_order]
        periods_list = get_param_value(get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('periods'), [1, 5, 13, 21, 55])
        norm_window = 55
        print("\n  [链路层 1] 各支柱健康度重算 (Pillar Health Recalculation)")
        ti_s_bull, _, _, daily_bipolar = engine._calculate_trend_integrity_health(df, periods_list, norm_window)
        mtf_s_bull, _, _ = engine._calculate_mtf_cohesion_health(df, periods_list, norm_window, daily_bipolar)
        bp_s_bull, _, _ = engine._calculate_breakout_potential_health(df, periods_list, norm_window)
        pc_s_bull, _, _ = engine._calculate_pattern_health(df, periods_list, norm_window)
        pillar_health_series = {
            'trend_integrity': ti_s_bull.get(period),
            'mtf_cohesion': mtf_s_bull.get(period),
            'breakout_potential': bp_s_bull.get(period),
            'pattern_confirmation': pc_s_bull.get(period)
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
        # [代码新增结束]











