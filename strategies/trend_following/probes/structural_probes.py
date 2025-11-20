# 文件: strategies/trend_following/intelligence/probes/structural_probes.py
import pandas as pd
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class StructuralProbes:
    """
    【探针模块】结构情报专属探针
    """
    def __init__(self, intel_layer):
        self.intelligence_layer = intel_layer
        self.strategy = intel_layer.strategy
        self.structural_intel = intel_layer.structural_intel
    def _deploy_structural_health_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.0 · 信号流穿透版】结构健康度探针
        - 核心升级: 探针现在能完全模拟 `transmute_health_to_ultimate_signals` 的内部计算流程，
                      包括 s_bull/s_bear 的净值计算、多时间框架融合、以及最终的单双极性转换，
                      从而实现对信号流的端到端穿透式解剖。
        """
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
        
    def _deploy_structural_pillar_fusion_probe(self, probe_date: pd.Timestamp, period: int = 13):
        """
        【探针 V2.0 · V20引擎同步版】结构支柱融合探针
        - 核心重构: 完全同步 `StructuralIntelligence` V20.0 的融合逻辑。
          1. 不再直接融合各支柱的 s_bull 分，而是先计算各支柱的 s_bull - s_bear 双极性净值。
          2. 对这些双极性净值进行加权融合。
          3. 将最终的融合净值通过 bipolar_to_exclusive_unipolar 转换为最终的看涨分进行对比。
        - 收益: 确保探针的重算逻辑与主引擎完全一致，消除因逻辑不同步导致的“假警报”。
        """
        print("\n" + "="*25 + f" [结构探针] 正在启用 🔬【结构支柱融合探针 V2.0 (p={period})】🔬 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.structural_intel
        from strategies.trend_following.utils import bipolar_to_exclusive_unipolar
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        p_conf = get_params_block(self.strategy, 'structural_ultimate_params', {})
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {})
        pillar_names_in_order = ['trend_integrity', 'mtf_cohesion', 'breakout_potential', 'structural_stability']
        weights_in_order = [pillar_weights.get(name, 0.25) for name in pillar_names_in_order]
        periods_list = get_param_value(get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {}).get('periods'), [1, 5, 13, 21, 55])
        norm_window = 55
        print("\n  [链路层 1] 各支柱健康度净值重算 (Pillar Net Health Recalculation)")
        # 获取每个支柱的 s_bull 和 s_bear
        ti_s_bull, ti_s_bear, _, daily_bipolar = engine._calculate_trend_integrity_health(df, periods_list, norm_window)
        mtf_s_bull, mtf_s_bear, _ = engine._calculate_mtf_cohesion_health(df, periods_list, norm_window, daily_bipolar)
        bp_s_bull, bp_s_bear, _ = engine._calculate_breakout_potential_health(df, periods_list, norm_window)
        ss_s_bull, ss_s_bear, _ = engine._calculate_structural_stability_health(df, periods_list, norm_window)
        # 计算每个支柱在 probe_date 的双极性净值
        pillar_bipolar_values = {
            'trend_integrity': get_val(ti_s_bull.get(period), probe_date) - get_val(ti_s_bear.get(period), probe_date),
            'mtf_cohesion': get_val(mtf_s_bull.get(period), probe_date) - get_val(mtf_s_bear.get(period), probe_date),
            'breakout_potential': get_val(bp_s_bull.get(period), probe_date) - get_val(bp_s_bear.get(period), probe_date),
            'structural_stability': get_val(ss_s_bull.get(period), probe_date) - get_val(ss_s_bear.get(period), probe_date)
        }
        print("\n  [链路层 2] 融合过程解剖 (Fusion Process Dissection)")
        recalc_fused_bipolar_score = 0.0
        total_weight = sum(weights_in_order)
        if total_weight > 0:
            for i, name in enumerate(pillar_names_in_order):
                bipolar_score = pillar_bipolar_values.get(name, 0.0)
                weight = weights_in_order[i]
                contribution = bipolar_score * (weight / total_weight)
                recalc_fused_bipolar_score += contribution
                print(f"    - [支柱: {name:<22}] 净值: {bipolar_score:.4f} | 权重: {weight:.2f} | 贡献: {contribution:.4f}")
        else:
            scores = list(pillar_bipolar_values.values())
            recalc_fused_bipolar_score = np.mean(scores) if scores else 0.0
        print(f"    - 【探针重算融合净值 (p={period})】: {recalc_fused_bipolar_score:.4f}")
        print("\n  [链路层 3] 最终信号转换 (Final Signal Transmutation)")
        recalc_bull_score, _ = bipolar_to_exclusive_unipolar(pd.Series([recalc_fused_bipolar_score]))
        final_recalc_score = recalc_bull_score.iloc[0]
        print(f"    - 【探针重算看涨分】: {final_recalc_score:.4f}")
        print("\n  [链路层 4] 终极对质 (Final Verdict)")
        overall_health = atomic.get('__STRUCTURE_overall_health', {})
        s_bull_overall = overall_health.get('s_bull', {})
        system_fused_score = get_val(s_bull_overall.get(period), probe_date)
        print(f"    - [对比]: 系统值 {system_fused_score:.4f} vs. 探针重算值 {final_recalc_score:.4f} -> {'✅ 一致' if np.isclose(system_fused_score, final_recalc_score) else '❌ 不一致'}")
        print("\n--- “结构支柱融合探针”解剖完毕 ---")
    def _deploy_structural_pillar_dissection_probe(self, probe_date: pd.Timestamp, pillar_name: str = 'trend_integrity', period: int = 13):
        """
        【探针 V1.4 · 堡垒完整度同步版】结构支柱穿透式解剖探针
        - 核心同步: 'structural_stability' 支柱的解剖逻辑已与主引擎 V2.0 的“堡垒完整度 vs 围城压力”模型完全同步。
        """
        print("\n" + "="*25 + f" [结构探针] 正在启用 🧬【结构支柱解剖探针 V1.4 ({pillar_name})】🧬 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.structural_intel
        from strategies.trend_following.utils import bipolar_to_exclusive_unipolar, normalize_score
        def get_val(series, date, default=0.0):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        if pillar_name == 'trend_integrity':
            # ... (trend_integrity 解剖逻辑保持不变)
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
            final_dynamic_score_series = engine._perform_relational_meta_analysis(df, bipolar_snapshot_series)
            final_dynamic_score_val = get_val(final_dynamic_score_series, probe_date)
            print(f"    - 【关系元分析后动态分】: {final_dynamic_score_val:.4f}")
            final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score_series)
            print(f"    - 【最终单极性转换后】 看涨分: {get_val(final_bull_score, probe_date):.4f}, 看跌分: {get_val(final_bear_score, probe_date):.4f}")
            ti_s_bull, _, _, _ = engine._calculate_trend_integrity_health(df, [period], norm_window)
            system_pillar_score = get_val(ti_s_bull.get(period), probe_date)
            print(f"    - [对比]: 系统支柱分 {system_pillar_score:.4f} vs. 探针重算分 {get_val(final_bull_score, probe_date):.4f} -> {'✅ 一致' if np.isclose(system_pillar_score, get_val(final_bull_score, probe_date)) else '❌ 不一致'}")
        elif pillar_name == 'structural_stability':
            print("\n  [链路层 1] 结构稳定性支柱解剖 (Structural Stability Dissection)")
            norm_window = 55
            default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
            # 1. 重算看涨组件: 堡垒完整度 (Fortress Integrity)
            print("    --- [看涨组件: 堡垒完整度] ---")
            static_support_dist_s = normalize_score(df.get('support_below_D', default_series), df.index, norm_window)
            static_support_vol_s = normalize_score(df.get('support_below_volume_D', default_series), df.index, norm_window)
            static_fortress_s = (static_support_dist_s * static_support_vol_s)**0.5
            realized_support_s = normalize_score(df.get('realized_support_intensity_D', default_series), df.index, norm_window)
            peak_defense_s = normalize_score(df.get('peak_defense_intensity_D', default_series), df.index, norm_window)
            main_force_support_s = normalize_score(df.get('main_force_support_strength_D', default_series), df.index, norm_window)
            dynamic_defense_s = (realized_support_s * peak_defense_s * main_force_support_s)**(1/3)
            bull_snapshot_series = (static_fortress_s * 0.4 + dynamic_defense_s * 0.6)
            print(f"      - [静态城墙] 支撑距离: {get_val(static_support_dist_s, probe_date):.4f}, 支撑量: {get_val(static_support_vol_s, probe_date):.4f} -> 融合分: {get_val(static_fortress_s, probe_date):.4f}")
            print(f"      - [动态防御] 已实现支撑: {get_val(realized_support_s, probe_date):.4f}, 主峰防御: {get_val(peak_defense_s, probe_date):.4f}, 主力支撑: {get_val(main_force_support_s, probe_date):.4f} -> 融合分: {get_val(dynamic_defense_s, probe_date):.4f}")
            print(f"    - 【融合看涨快照分】: {get_val(bull_snapshot_series, probe_date):.4f}")
            # 2. 重算看跌组件: 围城压力 (Siege Pressure)
            print("    --- [看跌组件: 围城压力] ---")
            static_pressure_dist_s = normalize_score(df.get('pressure_above_D', default_series), df.index, norm_window, ascending=False)
            static_pressure_vol_s = normalize_score(df.get('pressure_above_volume_D', default_series), df.index, norm_window)
            static_siege_s = (static_pressure_dist_s * static_pressure_vol_s)**0.5
            realized_pressure_s = normalize_score(df.get('realized_pressure_intensity_D', default_series), df.index, norm_window)
            main_force_pressure_s = normalize_score(df.get('main_force_distribution_pressure_D', default_series), df.index, norm_window)
            dynamic_assault_s = (realized_pressure_s * main_force_pressure_s)**0.5
            bear_snapshot_series = (static_siege_s * 0.4 + dynamic_assault_s * 0.6)
            print(f"      - [静态兵力] 压力距离: {get_val(static_pressure_dist_s, probe_date):.4f}, 压力量: {get_val(static_pressure_vol_s, probe_date):.4f} -> 融合分: {get_val(static_siege_s, probe_date):.4f}")
            print(f"      - [动态攻击] 已实现压力: {get_val(realized_pressure_s, probe_date):.4f}, 主力派发: {get_val(main_force_pressure_s, probe_date):.4f} -> 融合分: {get_val(dynamic_assault_s, probe_date):.4f}")
            print(f"    - 【融合看跌快照分】: {get_val(bear_snapshot_series, probe_date):.4f}")
            # 3. 计算双极性快照分
            bipolar_snapshot_series = (bull_snapshot_series - bear_snapshot_series).clip(-1, 1)
            print(f"    - 【双极性快照分】: {get_val(bipolar_snapshot_series, probe_date):.4f}")
            # 4. 执行关系元分析
            final_dynamic_score_series = engine._perform_relational_meta_analysis(df, bipolar_snapshot_series)
            print(f"    - 【关系元分析后动态分】: {get_val(final_dynamic_score_series, probe_date):.4f}")
            # 5. 最终转换与对比
            final_bull_score, final_bear_score = bipolar_to_exclusive_unipolar(final_dynamic_score_series)
            ss_s_bull, ss_s_bear, _ = engine._calculate_structural_stability_health(df, [period], norm_window)
            system_s_bull = get_val(ss_s_bull.get(period), probe_date)
            system_s_bear = get_val(ss_s_bear.get(period), probe_date)
            print(f"    - 【最终单极性转换后】 看涨分: {get_val(final_bull_score, probe_date):.4f}, 看跌分: {get_val(final_bear_score, probe_date):.4f}")
            print(f"    - [对比-看涨]: 系统支柱分 {system_s_bull:.4f} vs. 探针重算分 {get_val(final_bull_score, probe_date):.4f} -> {'✅ 一致' if np.isclose(system_s_bull, get_val(final_bull_score, probe_date)) else '❌ 不一致'}")
            print(f"    - [对比-看跌]: 系统支柱分 {system_s_bear:.4f} vs. 探针重算分 {get_val(final_bear_score, probe_date):.4f} -> {'✅ 一致' if np.isclose(system_s_bear, get_val(final_bear_score, probe_date)) else '❌ 不一致'}")
        else:
            print(f"    - 探针暂不支持解剖 '{pillar_name}' 支柱。")
        print("\n--- “结构支柱解剖探针”解剖完毕 ---")











