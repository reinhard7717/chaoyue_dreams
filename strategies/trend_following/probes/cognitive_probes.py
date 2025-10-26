import pandas as pd
import numpy as np
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score

class CognitiveProbes:
    """
    【探针模块】认知情报专属探针
    """
    def __init__(self, intel_layer):
        self.strategy = intel_layer.strategy
        self.cognitive_intel = intel_layer.cognitive_intel

    def _deploy_thanatos_scythe_probe(self, probe_date: pd.Timestamp):
        """
        【V1.1 · 时空同步版】“塔纳托斯之镰”探针 - 真实撤退风险全要素解剖
        - 核心修复: 完全同步主引擎V5.3版的逻辑，确保探针与主引擎使用相同的“盈利盘信念”数据源进行重算。
        """
        print("\n--- [探针] 正在启用: 💀【真实撤退风险解剖 V1.1】💀 ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        norm_window = 55
        p = 5
        def get_val(series, date, default=np.nan):
            if series is None or not isinstance(series, (pd.Series, dict)): return default
            if isinstance(series, dict): return series.get(date, default)
            return series.get(date, default)
        print("\n  [链路层 1] 最终裁决 (Final Verdict)")
        final_score = get_val(atomic.get('COGNITIVE_SCORE_TRUE_RETREAT_RISK'), probe_date, 0.0)
        print(f"    - 【最终风险值】: {final_score:.4f}")
        print("\n  [链路层 2] 前提条件 (Premise): 近期派发强度")
        to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True))**0.5
        to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) * normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False))**0.5
        short_term_transfer_snapshot = (to_main - to_retail).astype(np.float32)
        recent_distribution_strength = (short_term_transfer_snapshot.rolling(3).mean().clip(-1, 0) * -1).astype(np.float32)
        print(f"    - 【近期派发强度】: {get_val(recent_distribution_strength, probe_date):.4f}")
        print("\n  [链路层 3] 证据链 (Evidence Chain): 真实撤退的四大迹象")
        trend_quality_context = get_val(atomic.get('COGNITIVE_SCORE_TREND_QUALITY'), probe_date, 0.0)
        trend_decay_context = 1.0 - trend_quality_context
        panic_absorption_score = get_val(atomic.get('SCORE_MICRO_PANIC_ABSORPTION'), probe_date, 0.0)
        no_absorption_score = 1.0 - panic_absorption_score
        winner_conviction_0_1 = (get_val(atomic.get('PROCESS_META_WINNER_CONVICTION'), probe_date, 0.0) * 0.5 + 0.5)
        winner_capitulation_score = (1.0 - winner_conviction_0_1) ** 0.7
        dyn_bullish_resonance = get_val(atomic.get('SCORE_DYN_BULLISH_RESONANCE'), probe_date, 0.0)
        behavior_bullish_resonance = get_val(atomic.get('SCORE_BEHAVIOR_BULLISH_RESONANCE'), probe_date, 0.0)
        reversal_dynamic_quality = (dyn_bullish_resonance * behavior_bullish_resonance)**0.5
        bull_trap_evidence = 1.0 - reversal_dynamic_quality
        retreat_evidence_chain = (trend_decay_context * no_absorption_score * winner_capitulation_score * bull_trap_evidence)
        print(f"    - 【证据链总强度】: {retreat_evidence_chain:.4f}")
        print("\n--- “真实撤退风险探针”解剖完毕 ---")

    def _deploy_liquidity_trap_probe(self, probe_date: pd.Timestamp):
        """
        【V2.0 · 全息诊断版】流动性陷阱风险探针
        - 核心升级: 同步主引擎的全息诊断逻辑，对每个证据进行多时间维度(MTF)的融合计算。
        """
        print("\n" + "="*35 + f" [认知探针] 正在启用 🌊【流动性陷阱风险探针 V2.0】🌊 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        periods = get_param_value(p_cognitive.get('expansion_engine_periods'), [1, 5, 13, 21, 55])
        tf_weights = get_param_value(p_cognitive.get('expansion_engine_tf_weights'), {"1": 0.05, "5": 0.2, "13": 0.3, "21": 0.3, "55": 0.15})
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get('COGNITIVE_RISK_LIQUIDITY_TRAP'), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 全息证据链 (Holographic Evidence Chain)")
        evidence_sources = {
            '卖压背景': atomic.get('SCORE_FF_BEARISH_RESONANCE'),
            '流动性枯竭': atomic.get('SCORE_RISK_LIQUIDITY_DRAIN'),
            '市场脆弱性': df.get('intraday_volatility_D'),
        }
        fused_evidence_scores = {}
        for name, series in evidence_sources.items():
            fused_score_val = 0.0
            if series is not None and total_weight > 0:
                for p in periods:
                    weight = numeric_tf_weights.get(p, 0) / total_weight
                    norm_val = get_val(normalize_score(series, df.index, p), probe_date)
                    fused_score_val += norm_val * weight
            fused_evidence_scores[name] = fused_score_val
        print("\n  [链路层 3] 快照分重算 (Snapshot Score Recalculation)")
        evidence1_norm = fused_evidence_scores['卖压背景']
        evidence2_norm = fused_evidence_scores['流动性枯竭']
        evidence3_norm = fused_evidence_scores['市场脆弱性']
        recalc_snapshot_score = (evidence1_norm * evidence2_norm * evidence3_norm)**(1/3)
        print(f"    - 【探针重算快照分】: {recalc_snapshot_score:.4f}")
        print("\n--- “流动性陷阱风险探针”解剖完毕 ---")

    def _deploy_profit_taking_pressure_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0】利润兑现压力风险探针
        - 核心使命: 深度解剖重构后的 COGNITIVE_RISK_PROFIT_TAKING_PRESSURE 信号。
        - 解剖链路: 1. 最终风险分 -> 2. 跨周期融合 -> 3. 单周期元分析 -> 4. 核心证据快照分 -> 5. 四维质量评估。
        """
        print("\n" + "="*35 + f" [认知探针] 正在启用 💸【利润兑现压力风险探针 V1.0】💸 " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel.micro_behavior_engine # 引擎实例现在位于微观行为引擎下
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_PROFIT_TAKING_PRESSURE'
        # 链路层 1: 最终输出
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        # 链路层 2: 核心证据四维质量评估 (以 p=5 周期为例)
        print("\n  [链路层 2] 核心证据四维质量评估 (p=5 周期为例)")
        p_tactical, p_context = 5, 13
        # 证据一: 了结紧迫度
        urgency_quality_series = engine._calculate_4d_metric_quality(df, 'profit_taking_urgency_D', p_tactical, p_context, ascending=True)
        urgency_quality_val = get_val(urgency_quality_series, probe_date)
        print(f"    - [证据一: 了结紧迫度] 质量分: {urgency_quality_val:.4f}")
        # 证据二: 兑现溢价
        premium_quality_series = engine._calculate_4d_metric_quality(df, 'profit_realization_premium_D', p_tactical, p_context, ascending=True)
        premium_quality_val = get_val(premium_quality_series, probe_date)
        print(f"    - [证据二: 兑现溢价] 质量分: {premium_quality_val:.4f}")
        # 链路层 3: 快照分融合
        print("\n  [链路层 3] 快照分融合 (p=5 周期)")
        snapshot_score_val = (urgency_quality_val * premium_quality_val)**0.5
        print(f"    - [融合公式]: (紧迫度质量 * 溢价质量) ** 0.5")
        print(f"    - 【探针重算快照分】: ({urgency_quality_val:.4f} * {premium_quality_val:.4f})**0.5 = {snapshot_score_val:.4f}")
        # 链路层 4: 关系元分析
        print("\n  [链路层 4] 关系元分析 (p=5 周期)")
        snapshot_series = (urgency_quality_series * premium_quality_series)**0.5
        recalc_period_score_series = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_series)
        recalc_period_score_val = get_val(recalc_period_score_series, probe_date)
        print(f"    - 【探针重算周期风险分】: {recalc_period_score_val:.4f}")
        # 链路层 5: 跨周期融合
        print("\n  [链路层 5] 跨周期融合")
        # 重新执行完整的计算以获得所有周期的分数
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        pressure_scores_by_period = {}
        for i, p_tac in enumerate(sorted_periods):
            p_con = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tac
            urgency_q = engine._calculate_4d_metric_quality(df, 'profit_taking_urgency_D', p_tac, p_con, ascending=True)
            premium_q = engine._calculate_4d_metric_quality(df, 'profit_realization_premium_D', p_tac, p_con, ascending=True)
            snapshot_s = (urgency_q * premium_q)**0.5
            pressure_scores_by_period[p_tac] = engine._perform_micro_behavior_relational_meta_analysis(df, snapshot_s)
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        recalc_final_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.values())
        calc_str = []
        for p in periods:
            weight = tf_weights.get(p, 0) / total_weight
            score_val = get_val(pressure_scores_by_period.get(p), probe_date, 0.0)
            recalc_final_score += score_val * weight
            calc_str.append(f"({score_val:.2f}*{weight:.2f})")
        print(f"    - [融合公式]: Σ (周期分 * 权重)")
        print(f"    - [计算过程]: {' + '.join(calc_str)} = {recalc_final_score.get(probe_date):.4f}")
        print("\n  [链路层 6] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_final_score.get(probe_date):.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_final_score.get(probe_date)) else '❌ 不一致'}")
        print("\n--- “利润兑现压力风险探针”解剖完毕 ---")

    def _deploy_ultimate_top_reversal_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0 · 新增】终极顶部反转探针
        - 核心使命: 解剖 COGNITIVE_RISK_ULTIMATE_TOP_REVERSAL 信号，追溯其核心风险源。
        """
        # [代码新增开始]
        print("\n" + "="*35 + f" [认知探针] 正在启用 🛡️【终极顶部反转探针 V1.0】🛡️ " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_ULTIMATE_TOP_REVERSAL'
        # 链路层 1: 最终输出
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        # 链路层 2: 核心风险源分析
        print("\n  [链路层 2] 核心风险源分析 (Component Analysis)")
        risk_signals = {
            'CONTEXT_TOP_SCORE': engine._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0),
            'SCORE_RISK_UPTHRUST_DISTRIBUTION': engine._get_atomic_score(df, 'SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0),
            'SCORE_BOARD_HEAVEN_EARTH': engine._get_atomic_score(df, 'SCORE_BOARD_HEAVEN_EARTH', 0.0),
            'COGNITIVE_RISK_RETAIL_FOMO_MAIN_FORCE_RETREAT': engine._get_atomic_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_MAIN_FORCE_RETREAT', 0.0),
            'COGNITIVE_RISK_LTP_HIGH_DISTRIBUTION': engine._get_atomic_score(df, 'COGNITIVE_RISK_LTP_HIGH_DISTRIBUTION', 0.0),
            'SCORE_RISK_ICARUS_FALL': engine._get_atomic_score(df, 'SCORE_RISK_ICARUS_FALL', 0.0),
            'COGNITIVE_RISK_TRUE_RETREAT_RISK': engine._get_atomic_score(df, 'COGNITIVE_SCORE_TRUE_RETREAT_RISK', 0.0),
            'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION': engine._get_atomic_score(df, 'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', 0.0),
            'COGNITIVE_RISK_SIREN_SONG': engine._get_atomic_score(df, 'COGNITIVE_RISK_SIREN_SONG', 0.0),
            'COGNITIVE_RISK_OLYMPUS_CRUMBLING': engine._get_atomic_score(df, 'COGNITIVE_RISK_OLYMPUS_CRUMBLING', 0.0),
            'COGNITIVE_RISK_CYCLICAL_TOP': engine._get_atomic_score(df, 'COGNITIVE_RISK_CYCLICAL_TOP', 0.0)
        }
        component_scores_on_date = {name: get_val(series, probe_date, 0.0) for name, series in risk_signals.items()}
        if not any(component_scores_on_date.values()):
            print("    - 未找到任何风险分项。")
            return
        max_contributor = max(component_scores_on_date, key=component_scores_on_date.get)
        max_score = component_scores_on_date[max_contributor]
        print(f"    - 主要风险源: 【{max_contributor}】 (分值: {max_score:.4f})")
        print("    - 各分项得分详情:")
        for name, score in sorted(component_scores_on_date.items(), key=lambda item: item[1], reverse=True):
            if score > 0.01:
                print(f"      - {name}: {score:.4f}")
        # 链路层 3: 重算与对质
        print("\n  [链路层 3] 探针重算与对质 (Recalculation & Verdict)")
        recalc_score = max(component_scores_on_date.values())
        print(f"    - [融合公式]: max(各分项得分)")
        print(f"    - 【探针重算风险分】: {recalc_score:.4f}")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_score:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_score) else '❌ 不一致'}")
        print("\n--- “终极顶部反转探针”解剖完毕 ---")
        # [代码新增结束]

    def _deploy_ultimate_top_reversal_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V1.0】终极顶部反转探针
        - 核心使命: 解剖 COGNITIVE_RISK_ULTIMATE_TOP_REVERSAL 信号，追溯其核心风险源。
        """
        print("\n" + "="*35 + f" [认知探针] 正在启用 🛡️【终极顶部反转探针 V1.0】🛡️ " + "="*35)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_ULTIMATE_TOP_REVERSAL'
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 核心风险源分析 (Component Analysis)")
        risk_signals = {
            'CONTEXT_TOP_SCORE': engine._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0),
            'SCORE_RISK_UPTHRUST_DISTRIBUTION': engine._get_atomic_score(df, 'SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0),
            'SCORE_BOARD_HEAVEN_EARTH': engine._get_atomic_score(df, 'SCORE_BOARD_HEAVEN_EARTH', 0.0),
            'COGNITIVE_RISK_RETAIL_FOMO_MAIN_FORCE_RETREAT': engine._get_atomic_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_MAIN_FORCE_RETREAT', 0.0),
            'COGNITIVE_RISK_LTP_HIGH_DISTRIBUTION': engine._get_atomic_score(df, 'COGNITIVE_RISK_LTP_HIGH_DISTRIBUTION', 0.0),
            'SCORE_RISK_ICARUS_FALL': engine._get_atomic_score(df, 'SCORE_RISK_ICARUS_FALL', 0.0),
            'COGNITIVE_RISK_TRUE_RETREAT_RISK': engine._get_atomic_score(df, 'COGNITIVE_SCORE_TRUE_RETREAT_RISK', 0.0),
            'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION': engine._get_atomic_score(df, 'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', 0.0),
            'COGNITIVE_RISK_SIREN_SONG': engine._get_atomic_score(df, 'COGNITIVE_RISK_SIREN_SONG', 0.0),
            'COGNITIVE_RISK_OLYMPUS_CRUMBLING': engine._get_atomic_score(df, 'COGNITIVE_RISK_OLYMPUS_CRUMBLING', 0.0),
            'COGNITIVE_RISK_CYCLICAL_TOP': engine._get_atomic_score(df, 'COGNITIVE_RISK_CYCLICAL_TOP', 0.0)
        }
        component_scores_on_date = {name: get_val(series, probe_date, 0.0) for name, series in risk_signals.items()}
        if not any(component_scores_on_date.values()):
            print("    - 未找到任何风险分项。")
            return
        max_contributor = max(component_scores_on_date, key=component_scores_on_date.get)
        max_score = component_scores_on_date[max_contributor]
        print(f"    - 主要风险源: 【{max_contributor}】 (分值: {max_score:.4f})")
        print("    - 各分项得分详情:")
        for name, score in sorted(component_scores_on_date.items(), key=lambda item: item[1], reverse=True):
            if score > 0.01:
                print(f"      - {name}: {score:.4f}")
        print("\n  [链路层 3] 探针重算与对质 (Recalculation & Verdict)")
        recalc_score = max(component_scores_on_date.values())
        print(f"    - [融合公式]: max(各分项得分)")
        print(f"    - 【探针重算风险分】: {recalc_score:.4f}")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_score:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_score) else '❌ 不一致'}")
        print("\n--- “终极顶部反转探针”解剖完毕 ---")

    def _deploy_ltp_high_distribution_probe(self, probe_date: pd.Timestamp):
        """
        【探针 V2.0 · 风险逻辑重构版】长期获利盘高位派发风险探针
        - 核心重构: 同步主引擎的风险逻辑重构，解剖全新的、基于“主力派发给散户”的证据链。
        """
        # [代码修改开始]
        print("\n" + "="*25 + f" [认知探针] 正在启用 📉【长期获利盘高位派发风险探针 V2.0】📉 " + "="*25)
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        engine = self.cognitive_intel
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        periods = get_param_value(p_cognitive.get('expansion_engine_periods'), [1, 5, 13, 21, 55])
        tf_weights = get_param_value(p_cognitive.get('expansion_engine_tf_weights'), {})
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        def get_val(series, date, default=np.nan):
            if series is None: return default
            val = series.get(date)
            return default if pd.isna(val) else val
        signal_name = 'COGNITIVE_RISK_LTP_HIGH_DISTRIBUTION'
        print("\n  [链路层 1] 最终系统输出 (Final System Output)")
        actual_final_score = get_val(atomic.get(signal_name), probe_date, 0.0)
        print(f"    - 【最终风险分】: {actual_final_score:.4f}")
        print("\n  [链路层 2] 核心证据全息诊断 (Holographic Diagnosis of Core Evidence)")
        # 更新为新的证据链
        components = [
            {'source': 'atomic', 'name': 'CONTEXT_TOP_SCORE', 'cn_name': '顶部上下文'},
            {'source': 'df', 'name': 'main_force_rally_distribution_D', 'cn_name': '主力拉高出货'},
            {'source': 'df', 'name': 'retail_chasing_accumulation_D', 'cn_name': '散户追涨抬轿'},
            {'source': 'df', 'name': 'main_force_net_flow_consensus_D', 'transform': 'neg_clip', 'cn_name': '主力净流出'},
        ]
        fused_evidence_scores = {}
        component_details = []
        for comp in components:
            source_series_raw = None
            if comp['source'] == 'df':
                source_series_raw = df.get(comp['name'], pd.Series(0.0, index=df.index))
            elif comp['source'] == 'atomic':
                source_series_raw = engine._get_atomic_score(df, comp['name'], 0.0)
            
            transformed_series = source_series_raw.copy()
            if comp.get('transform') == 'neg_clip':
                transformed_series = -transformed_series.clip(upper=0)
            
            fused_component_series = pd.Series(0.0, index=df.index)
            if total_weight > 0:
                for p in periods:
                    weight = numeric_tf_weights.get(p, 0) / total_weight
                    normalized_series = normalize_score(transformed_series, df.index, p)
                    fused_component_series += normalized_series * weight
            else:
                fused_component_series = normalize_score(transformed_series, df.index, 55)
            fused_evidence_scores[comp['name']] = fused_component_series
            raw_val = get_val(source_series_raw, probe_date, 0.0)
            fused_val = get_val(fused_component_series, probe_date, 0.0)
            component_details.append(f"    - [{comp['cn_name']}] 原始值: {raw_val:.4f} -> 全息诊断分: {fused_val:.4f}")
        print('\n'.join(component_details))
        print("\n  [链路层 3] 快照分融合 (Snapshot Score Fusion)")
        snapshot_components = list(fused_evidence_scores.values())
        snapshot_series = pd.Series(np.prod(np.stack([s.values for s in snapshot_components], axis=0), axis=0) ** (1.0 / len(snapshot_components)), index=df.index)
        recalc_snapshot_score = get_val(snapshot_series, probe_date, 0.0)
        print(f"    - [融合公式]: (证据1 * 证据2 * ...)^(1/N)")
        print(f"    - 【探针重算快照分】: {recalc_snapshot_score:.4f}")
        print("\n  [链路层 4] 动态元分析 (Dynamic Meta-Analysis)")
        recalc_final_series = engine._perform_cognitive_relational_meta_analysis(df, snapshot_series)
        recalc_final_score = get_val(recalc_final_series, probe_date, 0.0)
        print(f"    - 【探针重算最终风险分】: {recalc_final_score:.4f}")
        print("\n  [链路层 5] 终极对质 (Final Verdict)")
        print(f"    - [对比]: 系统最终值 {actual_final_score:.4f} vs. 探针正确值 {recalc_final_score:.4f} -> {'✅ 一致' if np.isclose(actual_final_score, recalc_final_score) else '❌ 不一致'}")
        print("\n--- “长期获利盘高位派发风险探针”解剖完毕 ---")
        # [代码修改结束]










