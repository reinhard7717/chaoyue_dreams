# 文件: strategies/trend_following/intelligence/probes/structural_probes.py
import pandas as pd
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class StructuralProbes:
    """
    【探针模块】结构情报专属探针
    """
    def __init__(self, intel_layer):
        self.strategy = intel_layer.strategy
        self.structural_intel = intel_layer.structural_intel

    def _deploy_structural_health_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.0 · 信号流穿透版】结构健康度探针
        - 核心升级: 探针现在能完全模拟 `transmute_health_to_ultimate_signals` 的内部计算流程，
                      包括 s_bull/s_bear 的净值计算、多时间框架融合、以及最终的单双极性转换，
                      从而实现对信号流的端到端穿透式解剖。
        """
        # [代码修改开始]
        print("\n" + "="*25 + f" [结构探针] 正在启用 🏛️【结构健康度探针 V2.0】🏛️ " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        from strategies.trend_following.utils import bipolar_to_exclusive_unipolar
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'SCORE_STRUCTURE_BULLISH_RESONANCE'
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终信号分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 周期健康度净值计算 (Per-Period Net Health)")
        overall_health = atomic.get('__STRUCTURE_overall_health', {})
        s_bull_overall = overall_health.get('s_bull', {})
        s_bear_overall = overall_health.get('s_bear', {})
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        bipolar_health_by_period = {}
        for p in periods:
            s_bull = get_val(s_bull_overall.get(p), probe_date)
            s_bear = get_val(s_bear_overall.get(p), probe_date)
            net_health = s_bull - s_bear
            bipolar_health_by_period[p] = net_health
            print(f"    - [周期 {p}d] s_bull: {s_bull:.4f}, s_bear: {s_bear:.4f} -> 净值: {net_health:.4f}")
        print("\n  [链路层 3] 整体健康度融合 (Overall Health Fusion)")
        resonance_tf_weights = get_param_value(p_synthesis.get('resonance_tf_weights'), {})
        period_groups = {
            'short': [p for p in periods if p <= 5],
            'medium': [p for p in periods if 5 < p <= 21],
            'long': [p for p in periods if p > 21]
        }
        recalc_final_bipolar_resonance = 0.0
        total_weight = sum(resonance_tf_weights.values())
        if total_weight > 0:
            for p_key, weight in resonance_tf_weights.items():
                group_periods = period_groups.get(p_key, [])
                group_scores = [bipolar_health_by_period.get(p, 0.0) for p in group_periods]
                avg_group_score = np.mean(group_scores) if group_scores else 0.0
                recalc_final_bipolar_resonance += avg_group_score * (weight / total_weight)
                print(f"    - [周期组 {p_key}] 平均净值: {avg_group_score:.4f}, 权重贡献: {(avg_group_score * (weight / total_weight)):.4f}")
        print(f"    - 【探针重算融合净值】: {recalc_final_bipolar_resonance:.4f}")
        print("\n  [链路层 4] 最终信号转换 (Final Signal Transmutation)")
        # 模拟 bipolar_to_exclusive_unipolar 的转换
        recalc_bullish_resonance, _ = bipolar_to_exclusive_unipolar(pd.Series([recalc_final_bipolar_resonance]))
        final_recalc_score = recalc_bullish_resonance.iloc[0]
        print(f"    - [转换函数]: bipolar_to_exclusive_unipolar({recalc_final_bipolar_resonance:.4f})")
        print(f"    - 【探针重算看涨共振分】: {final_recalc_score:.4f}")
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
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

    def _deploy_structural_pillar_dissection_probe(self, probe_date: pd.Timestamp, pillar_name: str = 'trend_integrity', period: int = 13):
        """
        【探针 V1.2 · 统一命名同步版】结构支柱穿透式解剖探针
        - 核心修复: 调用重命名后的 `_perform_relational_meta_analysis`。
        """
        print("\n" + "="*25 + f" [结构探针] 正在启用 🧬【结构支柱解剖探针 V1.2 ({pillar_name})】🧬 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.structural_intel
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        if pillar_name == 'trend_integrity':
            print("\n  [链路层 1] 趋势完整性支柱解剖 (Trend Integrity Dissection)")
            p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
            fusion_weights = get_param_value(p_conf.get('ma_health_fusion_weights'), {})
            ma_periods = [5, 13, 21, 55]
            norm_window = 55
            required_cols = [f'MA_{p}_D' for p in ma_periods]
            if not all(col in df.columns for col in required_cols):
                print("    - 缺少必要的MA列，无法进行解剖。")
                return
            bull_alignment = np.mean([(df[f'MA_{ma_periods[i]}_D'].loc[probe_date] > df[f'MA_{ma_periods[i+1]}_D'].loc[probe_date]) for i in range(len(ma_periods) - 1)])
            bear_alignment = np.mean([(df[f'MA_{ma_periods[i]}_D'].loc[probe_date] < df[f'MA_{ma_periods[i+1]}_D'].loc[probe_date]) for i in range(len(ma_periods) - 1)])
            slope_cols = [f'SLOPE_{p}_MA_{p}_D' for p in ma_periods if f'SLOPE_{p}_MA_{p}_D' in df.columns]
            bull_velocity = np.mean([get_val(normalize_score(df[col], df.index, norm_window, ascending=True), probe_date) for col in slope_cols]) if slope_cols else 0.5
            bear_velocity = np.mean([get_val(normalize_score(df[col], df.index, norm_window, ascending=False), probe_date) for col in slope_cols]) if slope_cols else 0.5
            ma_std_series = pd.Series(np.std(df[required_cols].values / df['close_D'].values[:, np.newaxis], axis=1), index=df.index)
            bull_relational = 1.0 - get_val(normalize_score(ma_std_series, df.index, norm_window, ascending=True), probe_date)
            bear_relational = get_val(normalize_score(ma_std_series, df.index, norm_window, ascending=True), probe_date)
            print(f"    - [看涨组件] 排列: {bull_alignment:.4f}, 速度: {bull_velocity:.4f}, 关系: {bull_relational:.4f}")
            print(f"    - [看跌组件] 排列: {bear_alignment:.4f}, 速度: {bear_velocity:.4f}, 关系: {bear_relational:.4f}")
            bull_score_val = (bull_alignment * fusion_weights.get('alignment', 0.15) + bull_velocity * fusion_weights.get('slope', 0.15) + bull_relational * fusion_weights.get('relational', 0.25))
            bear_score_val = (bear_alignment * fusion_weights.get('alignment', 0.15) + bear_velocity * fusion_weights.get('slope', 0.15) + bear_relational * fusion_weights.get('relational', 0.25))
            print(f"    - 【融合看涨分】: {bull_score_val:.4f}")
            print(f"    - 【融合看跌分】: {bear_score_val:.4f}")
            bipolar_snapshot_val = (bull_score_val - bear_score_val)
            print(f"    - 【双极性快照分】: {bipolar_snapshot_val:.4f}")
            _, _, _, bipolar_snapshot_series = engine._calculate_trend_integrity_health(df, [period], norm_window)
            # [代码修改开始]
            final_dynamic_score_series = engine._perform_relational_meta_analysis(df, bipolar_snapshot_series)
            # [代码修改结束]
            final_dynamic_score_val = get_val(final_dynamic_score_series, probe_date)
            print(f"    - 【关系元分析后动态分】: {final_dynamic_score_val:.4f}")
            from strategies.trend_following.utils import bipolar_to_exclusive_unipolar
            final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score_series)
            print(f"    - 【最终单极性转换后】 看涨分: {get_val(final_bull_score, probe_date):.4f}, 看跌分: {get_val(final_bear_score, probe_date):.4f}")
            ti_s_bull, _, _, _ = engine._calculate_trend_integrity_health(df, [period], norm_window)
            system_pillar_score = get_val(ti_s_bull.get(period), probe_date)
            print(f"    - [对比]: 系统支柱分 {system_pillar_score:.4f} vs. 探针重算分 {get_val(final_bull_score, probe_date):.4f} -> {'✅ 一致' if np.isclose(system_pillar_score, get_val(final_bull_score, probe_date)) else '❌ 不一致'}")
        else:
            print(f"    - 探针暂不支持解剖 '{pillar_name}' 支柱。")
        print("\n--- “结构支柱解剖探针”解剖完毕 ---")










