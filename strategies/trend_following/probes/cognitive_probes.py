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
