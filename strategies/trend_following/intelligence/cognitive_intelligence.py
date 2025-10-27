# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_unified_score, calculate_context_scores, normalize_to_bipolar
from strategies.trend_following.intelligence.micro_behavior_engine import MicroBehaviorEngine
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.intelligence.intraday_behavior_engine import IntradayBehaviorEngine

class MainForceState(Enum):
    """
    定义主力行为序列的各个状态。
    """
    IDLE = 0           # 闲置/观察期
    ACCUMULATING = 1   # 吸筹期
    WASHING = 2        # 洗盘期
    MARKUP = 3         # 拉升期
    DISTRIBUTING = 4   # 派发期
    COLLAPSE = 5       # 崩盘期

class CognitiveIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化顶层认知合成模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        self.micro_behavior_engine = MicroBehaviorEngine(strategy_instance)
        self.tactic_engine = TacticEngine(strategy_instance)
        self.intraday_behavior_engine = IntradayBehaviorEngine(strategy_instance)

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """
        【健壮性修复版】安全地从原子状态库中获取分数。
        """
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def synthesize_cognitive_scores(self, df: pd.DataFrame, pullback_enhancements: Dict) -> pd.DataFrame:
        """
        【V15.0 · 综合风险重构版】顶层认知总分合成模块
        - 核心升级:
          1. [架构重构] 废除所有零散的 _synthesize_* 扩展信号方法，统一由全新的 `_synthesize_cognitive_expansion_engine` 引擎生成。
          2. [分析升维] 所有扩展信号现在都将经过“关系元分析”，具备“状态-速度-加速度”的动态洞察力。
        - 本次修改:
          - [名称净化] 将 `_diagnose_ultimate_top_reversal` 重命名为 `_diagnose_comprehensive_top_risk`。
          - [逻辑加固] 新的终极顶部风险引擎将采用“三柱-神盾”架构，融合更多高优先级风险信号，并引入趋势韧性抑制机制。
        """
        df = self.synthesize_trend_quality_score(df)
        df = self.synthesize_pullback_states(df)
        df = self.synthesize_structural_fusion_scores(df)
        df = self.synthesize_ultimate_confirmation_scores(df)
        df = self.synthesize_ignition_resonance_score(df)
        df = self.synthesize_reversal_resonance_scores(df)
        df = self.synthesize_tactical_reversal_resonance(df)
        df = self.synthesize_industry_synergy_signals(df)
        df = self.synthesize_mean_reversion_signals(df)
        df = self.synthesize_state_process_synergy(df)
        self.synthesize_trend_acceleration_cascade(df)
        self.synthesize_tactical_opportunity_fusion(df)
        suppression_vs_retreat_states = self._diagnose_suppression_vs_retreat(df)
        self.strategy.atomic_states.update(suppression_vs_retreat_states)
        cyclical_risk_states = self._calculate_cyclical_top_risk(df)
        self.strategy.atomic_states.update(cyclical_risk_states)
        self.strategy.atomic_states.update(self._synthesize_cognitive_expansion_engine(df))
        self.strategy.atomic_states['strategy_instance_ref'] = self.strategy
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        del self.strategy.atomic_states['strategy_instance_ref']
        self.strategy.atomic_states['CONTEXT_BOTTOM_SCORE'] = bottom_context_score
        self.strategy.atomic_states['CONTEXT_TOP_SCORE'] = top_context_score
        # 调用重构后的综合顶部风险诊断引擎
        comprehensive_top_risk_states = self._diagnose_comprehensive_top_risk(df)
        self.strategy.atomic_states.update(comprehensive_top_risk_states)
        bullish_signal_names = [
            'COGNITIVE_SCORE_IGNITION_RESONANCE',
            'COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE',
            'COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE',
            'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING',
            'COGNITIVE_SCORE_REVERSAL_RELIABILITY',
            'COGNITIVE_SCORE_STATE_PROCESS_SYNERGY',
            'COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE',
            'COGNITIVE_SCORE_TACTICAL_REVERSAL_RESONANCE',
            'COGNITIVE_SCORE_TACTICAL_OPPORTUNITY_FUSION',
            'SCORE_CHIP_TRUE_ACCUMULATION',
            'COGNITIVE_SCORE_TACTICAL_SUPPRESSION',
            'SMART_MONEY_SYNERGY_BUY_D',
            'SCORE_BEHAVIOR_SMART_INTRADAY_TRADING',
            'SCORE_STRUCTURAL_FAULT_BREAKTHROUGH',
            'SCORE_STRUCTURAL_CONSOLIDATION_BREAKOUT',
            'SCORE_FOUNDATION_CHIP_FAULT_BREAKOUT',
            'SCORE_MICRO_HERMES_GAMBIT',
            'COGNITIVE_SCORE_LEADER_DRIVES_SECTOR_RISE',
            'COGNITIVE_SCORE_INDUSTRY_RECESSION_INDIVIDUAL_STRENGTH',
            'COGNITIVE_SCORE_SENTIMENT_TECH_RESONANCE',
            'COGNITIVE_SCORE_LEADER_BREAKOUT_AWAKENING',
            'COGNITIVE_SCORE_POLICY_DRIVEN_BREAKOUT',
            'COGNITIVE_SCORE_LIMIT_DOWN_REVERSAL',
            'COGNITIVE_SCORE_DESPAIR_EMOTION_REVERSAL',
            'COGNITIVE_SCORE_PROFIT_MAKING_LOCKUP',
            'COGNITIVE_SCORE_MAIN_FORCE_COST_ADVANTAGE',
            'COGNITIVE_SCORE_ACCUMULATION_COMPRESSION_BREAKOUT',
            'COGNITIVE_SCORE_CHIP_PEAK_PLATFORM_SUPPORT',
            'COGNITIVE_SCORE_CHIP_FAULT_ACCELERATION',
            'COGNITIVE_SCORE_MAIN_FORCE_LOW_COST_ACCUMULATION',
            'COGNITIVE_SCORE_PANIC_SELLING_ABSORPTION',
            'COGNITIVE_SCORE_MAIN_FORCE_BREAKOUT_CONFIRMATION',
            'COGNITIVE_SCORE_VOLATILITY_COMPRESSION_ACCUMULATION',
            'COGNITIVE_SCORE_CHIP_STRUCTURE_STABLE_TREND',
            'COGNITIVE_SCORE_BOTTOM_POWER_TRANSFER',
            'COGNITIVE_SCORE_BREAKOUT_VALIDATION_CONFIRM',
        ]
        valid_bullish_scores = []
        for signal_name in bullish_signal_names:
            signal_series = self._get_atomic_score(df, signal_name)
            if signal_series is not None and not signal_series.empty:
                valid_bullish_scores.append(signal_series.values)
        if valid_bullish_scores:
            cognitive_bullish_score = np.maximum.reduce(valid_bullish_scores)
        else:
            cognitive_bullish_score = np.zeros(len(df.index), dtype=np.float32)
        self.strategy.atomic_states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)
        fused_risk_states = self.synthesize_fused_risk_scores(df)
        self.strategy.atomic_states.update(fused_risk_states)
        self.synthesize_chimera_conflict_score(df)
        return df

    def synthesize_state_process_synergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.4 · 意图洞察版】状态-过程协同融合引擎
        - 核心升级: 将“主力紧迫度”、“聪明钱成本优势”、“盈亏流量背离”三大“意图洞察型”
                      过程信号全部纳入“过程共识分”的计算，极大提升了对主力真实意图的洞察力。
        """
        states = {}
        state_bullish_signals = [
            'SCORE_CHIP_BULLISH_RESONANCE', 'SCORE_BEHAVIOR_BULLISH_RESONANCE',
            'SCORE_FF_BULLISH_RESONANCE', 'SCORE_STRUCTURE_BULLISH_RESONANCE',
            'SCORE_DYN_BULLISH_RESONANCE', 'SCORE_FOUNDATION_BULLISH_RESONANCE'
        ]
        state_scores = [self._get_atomic_score(df, sig, 0.5).values for sig in state_bullish_signals]
        state_consensus_score = pd.Series(
            np.prod(np.stack(state_scores, axis=0), axis=0) ** (1.0 / len(state_scores)),
            index=df.index, dtype=np.float32
        )
        states['COGNITIVE_INTERNAL_STATE_CONSENSUS'] = state_consensus_score
        # 引入所有“意图洞察型”过程信号
        process_bullish_signals = [
            'PROCESS_META_PV_REL_BULLISH_TURN',
            'PROCESS_META_PF_REL_BULLISH_TURN',
            'PROCESS_STRATEGY_CHIP_VS_BEHAVIOR_SYNC',
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_STEALTH_ACCUMULATION',
            'PROCESS_META_WINNER_CONVICTION',
            'PROCESS_META_LOSER_CAPITULATION',
            'PROCESS_META_MAIN_FORCE_URGENCY',
            'PROCESS_META_SMART_MONEY_COST_ADVANTAGE',
            'PROCESS_META_PROFIT_VS_FLOW',
            'PROCESS_META_PRICE_VS_RETAIL_CAPITULATION' # 替换了旧的幽灵信号
        ]
        process_scores = [(self._get_atomic_score(df, sig, 0.0).clip(-1, 1) * 0.5 + 0.5).values for sig in process_bullish_signals]
        process_consensus_score = pd.Series(
            np.prod(np.stack(process_scores, axis=0), axis=0) ** (1.0 / len(process_scores)),
            index=df.index, dtype=np.float32
        )
        states['COGNITIVE_INTERNAL_PROCESS_CONSENSUS'] = process_consensus_score
        synergy_score = (state_consensus_score * process_consensus_score).astype(np.float32)
        states['COGNITIVE_SCORE_STATE_PROCESS_SYNERGY'] = synergy_score
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_trend_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V5.2 · 逻辑链路修复版】趋势质量融合评分模块
        - 核心修复: 修正了 `direct_trend_health_score` 变量仅在 else 分支中定义的致命错误。
                      将其移出 if/else 结构，确保在所有逻辑路径下都能被正确计算和赋值，
                      解决了 UnboundLocalError 问题。
        """
        
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        domain_scores_map = {
            'behavior': 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'STRUCT_BEHAVIOR_TOP_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE'),
            'fund_flow': get_unified_score(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE'),
            'structural': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE'),
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE'),
            'regime': (self._get_atomic_score(df, 'SCORE_TRENDING_REGIME') * self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT'))**0.5,
            'cyclical': 1.0 - self._get_atomic_score(df, 'SCORE_CYCLICAL_REGIME')
        }
        # 将 direct_trend_health_score 的计算移出 if/else 块，确保它总能被定义
        direct_trend_health_score = self._calculate_cognitive_trend_health(df)
        consensus_snapshot_score = pd.Series(0.0, index=df.index, dtype=np.float64)
        total_weight = 0.0
        valid_weights = {name: w for name, w in weights.items() if w > 0 and name in domain_scores_map}
        total_weight = sum(valid_weights.values())
        if total_weight > 0:
            for name, score_series in domain_scores_map.items():
                weight = valid_weights.get(name, 0)
                if weight > 0:
                    consensus_snapshot_score += score_series.fillna(0.5) * (weight / total_weight)
        else:
            consensus_snapshot_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        trend_quality_snapshot_score = (direct_trend_health_score * consensus_snapshot_score)**0.5
        final_trend_quality_score = self._perform_cognitive_relational_meta_analysis(df, trend_quality_snapshot_score)
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = final_trend_quality_score.astype(np.float32)
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.1 · 动态分析版】认知层回踩状态合成模块
        - 核心升级: 引入关系元分析，对“健康回踩”和“压制性回踩”的快照分进行动态锻造。
        """
        states = {}
        is_pullback_day = (df['pct_change_D'] < 0).astype(float)
        constructive_context_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1).fillna(0.0)
        shrinking_volume_score = self._get_atomic_score(df, 'SCORE_VOL_WEAKENING_DROP', 0.0)
        cycle_trough_score = (1 - self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE', 0.0).fillna(0.0)) / 2.0
        winner_holding_tight_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_TOP_REVERSAL')
        chip_stable_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BEARISH_RESONANCE')
        # --- 健康回踩 ---
        healthy_pullback_snapshot_score = (
            is_pullback_day * constructive_context_score *
            gentle_drop_score * shrinking_volume_score *
            winner_holding_tight_score * chip_stable_score *
            (1 + cycle_trough_score * 0.5)
        )
        # 动态锻造
        final_healthy_pullback_score = self._perform_cognitive_relational_meta_analysis(df, healthy_pullback_snapshot_score)
        states['COGNITIVE_SCORE_PULLBACK_HEALTHY'] = final_healthy_pullback_score.astype(np.float32)
        # --- 压制性回踩 ---
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1).fillna(0.0)
        panic_selling_score = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BEARISH_RESONANCE')
        suppressive_pullback_snapshot_score = (
            is_pullback_day * constructive_context_score *
            significant_drop_score * panic_selling_score * winner_holding_tight_score
        )
        final_suppressive_pullback_score = self._perform_cognitive_relational_meta_analysis(df, suppressive_pullback_snapshot_score)
        states['COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE'] = final_suppressive_pullback_score.astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_fused_risk_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.2 · 快照发布版】风险元融合模块
        - 核心升级: 将内部计算的“风险快照分”发布到原子状态库，
                      命名为 COGNITIVE_INTERNAL_RISK_SNAPSHOT，确保系统的完全透明和可验证性。
        """
        states = {}
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            states['COGNITIVE_FUSED_RISK_SCORE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        risk_categories = p_fused_risk.get('risk_categories', {})
        all_required_signals = {s for signals in risk_categories.values() if isinstance(signals, dict) for s in signals if s != "说明"}
        signal_numpy_cache = {
            sig_name: self._get_atomic_score(df, sig_name, 0.0).values
            for sig_name in all_required_signals
        }
        default_numpy_array = np.zeros(len(df.index), dtype=np.float32)
        fused_dimension_scores = {}
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            category_signal_scores = []
            for signal_name, signal_params in signals.items():
                if signal_name == "说明": continue
                atomic_score_np = signal_numpy_cache.get(signal_name, default_numpy_array)
                processed_score = 1.0 - atomic_score_np if signal_params.get('inverse', False) else atomic_score_np
                category_signal_scores.append(processed_score * signal_params.get('weight', 1.0))
            if category_signal_scores:
                stacked_scores = np.stack(category_signal_scores, axis=0)
                dimension_risk_values = np.maximum.reduce(stacked_scores, axis=0)
                dimension_risk_score = pd.Series(dimension_risk_values, index=df.index, dtype=np.float32)
                fused_dimension_scores[category_name] = dimension_risk_score
                states[f'FUSED_RISK_SCORE_{category_name.upper()}'] = dimension_risk_score
            else:
                fused_dimension_scores[category_name] = pd.Series(0.0, index=df.index, dtype=np.float32)
        valid_scores_np = [score.values for score in fused_dimension_scores.values() if not score.empty]
        if valid_scores_np:
            stacked_scores = np.stack(valid_scores_np, axis=0)
            total_fused_risk_values = np.maximum.reduce(stacked_scores, axis=0)
            total_fused_risk_score = pd.Series(total_fused_risk_values, index=df.index, dtype=np.float32)
        else:
            total_fused_risk_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = p_resonance.get('core_risk_dimensions', [])
            min_dims = p_resonance.get('min_dimensions_for_resonance', 2)
            threshold = p_resonance.get('risk_score_threshold', 0.6)
            penalty_multiplier = p_resonance.get('penalty_multiplier', 1.2)
            high_risk_dimension_count = np.sum([
                (fused_dimension_scores[dim].values > threshold)
                for dim in core_dims if dim in fused_dimension_scores
            ], axis=0)
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            total_fused_risk_score_values = np.where(
                is_resonance_triggered,
                total_fused_risk_score.values * penalty_multiplier,
                total_fused_risk_score.values
            )
            total_fused_risk_score = pd.Series(total_fused_risk_score_values, index=df.index, dtype=np.float32)
            states['FUSED_RISK_RESONANCE_PENALTY_ACTIVE'] = pd.Series(is_resonance_triggered, index=df.index)
        trend_quality_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        healthy_pullback_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_PULLBACK_HEALTHY', 0.0)
        aegis_shield_strength = pd.Series(np.maximum(trend_quality_score.values, healthy_pullback_score.values), index=df.index).clip(0, 1)
        states['COGNITIVE_CONTEXT_AEGIS_SHIELD_STRENGTH'] = aegis_shield_strength.astype(np.float32)
        suppression_factor = 1.0 - aegis_shield_strength
        risk_snapshot_score = (total_fused_risk_score * suppression_factor).clip(0, 2.0).astype(np.float32)
        states['COGNITIVE_INTERNAL_RISK_SNAPSHOT'] = risk_snapshot_score
        final_dynamic_risk_score = self._perform_cognitive_relational_meta_analysis(df, risk_snapshot_score)
        states['COGNITIVE_FUSED_RISK_SCORE'] = final_dynamic_risk_score
        return states

    def synthesize_chimera_conflict_score(self, df: pd.DataFrame) -> None:
        """
        【V3.1 · 资金流分歧版】多维冲突诊断引擎
        - 核心升级: 引入“资金流分歧度”作为独立的冲突维度。
        - 新核心逻辑: 最终冲突分 = (信号冲突分 * 权重) + (资金流分歧冲突分 * 权重)。
        """
        states = {}
        p_chimera = get_params_block(self.strategy, 'chimera_conflict_params', {})
        weights = get_param_value(p_chimera.get('fusion_weights'), {'signal_conflict': 0.6, 'flow_divergence': 0.4})
        bullish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_BULLISH_SCORE', 0.0).clip(0, 1)
        bearish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_FUSED_RISK_SCORE', 0.0).clip(0, 1)
        signal_conflict_score = np.minimum(bullish_score_normalized, bearish_score_normalized)
        norm_window = 55
        div_ts_ths_abs = df.get('divergence_ts_ths_D', pd.Series(0, index=df.index)).abs()
        div_ts_dc_abs = df.get('divergence_ts_dc_D', pd.Series(0, index=df.index)).abs()
        div_ths_dc_abs = df.get('divergence_ths_dc_D', pd.Series(0, index=df.index)).abs()
        div_ts_ths_score = normalize_score(div_ts_ths_abs, df.index, norm_window, ascending=True)
        div_ts_dc_score = normalize_score(div_ts_dc_abs, df.index, norm_window, ascending=True)
        div_ths_dc_score = normalize_score(div_ths_dc_abs, df.index, norm_window, ascending=True)
        flow_divergence_conflict_score = np.maximum.reduce([
            div_ts_ths_score.values,
            div_ts_dc_score.values,
            div_ths_dc_score.values
        ])
        flow_divergence_conflict_score = pd.Series(flow_divergence_conflict_score, index=df.index)
        final_conflict_score = (
            signal_conflict_score * weights.get('signal_conflict', 0.6) +
            flow_divergence_conflict_score * weights.get('flow_divergence', 0.4)
        )
        states['COGNITIVE_SCORE_CHIMERA_CONFLICT'] = final_conflict_score.clip(0, 1).astype(np.float32)
        self.strategy.atomic_states.update(states)

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.1 · 动态共识版】结构化元信号融合模块
        - 核心革命: 引入两阶段认知。第一阶段计算“共识快照分”，第二阶段对“共识”本身进行关系元分析。
        - 新核心逻辑: 最终分数 = 动态分析(加权几何均值(基础, 结构, 行为))
        - 收益: 最终信号不仅反映共识强度，更反映共识形成的速度与加速度，实现认知前瞻。
        """
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        fusion_weights = get_param_value(p_cognitive.get('cognitive_fusion_weights'), {
            'foundation': 0.3, 'structure': 0.3, 'behavior': 0.4
        })
        signal_sources = {
            'foundation': {
                'bullish': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BULLISH_RESONANCE'),
                'bearish': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BEARISH_RESONANCE'),
                'bottom': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL'),
                'top': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_TOP_REVERSAL'),
            },
            'structure': {
                'bullish': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE'),
                'bearish': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BEARISH_RESONANCE'),
                'bottom': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL'),
                'top': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL'),
            },
            'behavior': {
                'bullish': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BULLISH_RESONANCE'),
                'bearish': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BEARISH_RESONANCE'),
                'bottom': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_RESONANCE'),
                'top': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL'),
            }
        }
        fusion_types = ['bullish', 'bearish', 'bottom', 'top']
        output_names = {
            'bullish': 'COGNITIVE_FUSION_BULLISH_RESONANCE',
            'bearish': 'COGNITIVE_FUSION_BEARISH_RESONANCE',
            'bottom': 'COGNITIVE_FUSION_BOTTOM_REVERSAL',
            'top': 'COGNITIVE_FUSION_TOP_REVERSAL'
        }
        for f_type in fusion_types:
            scores_to_fuse = []
            weights_to_fuse = []
            for domain, signals in signal_sources.items():
                weight = fusion_weights.get(domain, 0.33)
                if weight > 0:
                    scores_to_fuse.append(signals[f_type].values)
                    weights_to_fuse.append(weight)
            if not scores_to_fuse:
                states[output_names[f_type]] = default_score.copy()
                continue
            stacked_scores = np.stack(scores_to_fuse, axis=0)
            weights_array = np.array(weights_to_fuse)
            weights_array /= weights_array.sum()
            safe_scores = np.maximum(stacked_scores, 1e-9)
            log_signals = np.log(safe_scores)
            weighted_log_sum = np.sum(log_signals * weights_array[:, np.newaxis], axis=0)
            consensus_snapshot_score = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
            final_fused_score = self._perform_cognitive_relational_meta_analysis(df, consensus_snapshot_score)
            states[output_names[f_type]] = final_fused_score
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_ultimate_confirmation_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.1 · 终极授权版】终极确认融合模块
        - 核心修复: 在调用 calculate_context_scores 前，为其注入 'strategy_instance_ref'。
        """
        states = {}
        atomic = self.strategy.atomic_states
        required_fusion_signals = [
            'COGNITIVE_FUSION_BULLISH_RESONANCE', 'COGNITIVE_FUSION_BEARISH_RESONANCE',
            'COGNITIVE_FUSION_BOTTOM_REVERSAL', 'COGNITIVE_FUSION_TOP_REVERSAL'
        ]
        if any(s not in atomic for s in required_fusion_signals):
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        fusion_bullish = atomic.get('COGNITIVE_FUSION_BULLISH_RESONANCE', default_series)
        fusion_bearish = atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE', default_series)
        fusion_bottom = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL', default_series)
        fusion_top = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL', default_series)
        pattern_bullish = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_BULLISH_RESONANCE')
        pattern_bearish = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_BEARISH_RESONANCE')
        pattern_bottom = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_BOTTOM_REVERSAL')
        pattern_top = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_TOP_REVERSAL')
        # 注入计算所需的上下文引用
        atomic['strategy_instance_ref'] = self.strategy
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        del atomic['strategy_instance_ref'] # 移除临时引用
        states['COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION'] = (fusion_bullish * pattern_bullish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION'] = (fusion_bearish * pattern_bearish).astype(np.float32)
        ultimate_bottom_raw = fusion_bottom * pattern_bottom
        states['COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION'] = (ultimate_bottom_raw * bottom_context_score).astype(np.float32)
        ultimate_top_raw = fusion_top * pattern_top
        states['COGNITIVE_ULTIMATE_TOP_CONFIRMATION'] = (ultimate_top_raw * top_context_score).astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_ignition_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.2 · 依赖净化版】多域点火共振分数合成模块
        - 核心重构: 移除了对已废弃的原子剧本信号的依赖，使融合逻辑更纯粹。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 移除了对已废弃的原子剧本信号的依赖
        chip_consensus_ignition = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        behavioral_ignition = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BULLISH_RESONANCE')
        structural_breakout = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_ignition = get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE')
        volatility_breakout = self._get_atomic_score(df, 'SCORE_VOL_BREAKOUT_POTENTIAL', 0.0)
        fund_flow_ignition = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        
        general_ignition_resonance = (
            behavioral_ignition * structural_breakout * mechanics_ignition *
            chip_consensus_ignition * fund_flow_ignition * volatility_breakout
        )
        
        # 简化融合逻辑，核心是各大领域的看涨共振
        ignition_resonance_score = general_ignition_resonance.astype(np.float32)
        
        states['COGNITIVE_SCORE_IGNITION_RESONANCE'] = pd.Series(ignition_resonance_score, index=df.index)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_reversal_resonance_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V5.0 · 权力上收版】多域反转共振分数合成模块
        - 核心升级: 将外部上下文(bottom/top_context_score)的应用权力从专业层上收到本模块。
                      现在由认知层统一对“纯粹的”专业反转信号进行战略价值评估。
        """
        states = {}
        default_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = p.get('bottom_resonance_weights', {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3, 'ff': 0.2})
        top_weights = p.get('top_resonance_weights', {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3, 'ff': 0.2})
        
        # 在认知层统一获取外部上下文
        bottom_context_score = self._get_atomic_score(df, 'CONTEXT_BOTTOM_SCORE', 0.0)
        top_context_score = self._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0)
        
        
        # --- 底部反转共振分数 ---
        bottom_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BOTTOM_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL'),
            'ff': get_unified_score(self.strategy.atomic_states, df.index, 'FF_BOTTOM_REVERSAL')
        }
        bottom_scores_np = [s.values for d, s in bottom_sources.items() if bottom_weights.get(d, 0) > 0]
        bottom_weights_np = [w for d, w in bottom_weights.items() if d in bottom_sources and w > 0]
        if bottom_scores_np:
            weights_array = np.array(bottom_weights_np)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(bottom_scores_np, axis=0)
            # 得到纯粹的、未经调制的底部反转共振分
            bottom_reversal_raw = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            
            reversal_reliability_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_REVERSAL_RELIABILITY', 0.0)
            reliability_bonus_factor = get_param_value(p_cognitive.get('reversal_reliability_bonus_factor'), 0.5)
            reliability_amplifier = 1.0 + (reversal_reliability_score.values * reliability_bonus_factor)
            capitulation_potential_score = self._get_atomic_score(df, 'SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL', 0.0)
            capitulation_bonus_factor = get_param_value(p_cognitive.get('capitulation_potential_bonus_factor'), 0.5)
            capitulation_amplifier = 1.0 + (capitulation_potential_score.values * capitulation_bonus_factor)
            
            # 在这里，由认知层统一应用外部上下文进行战略价值评估
            bottom_reversal_snapshot_values = bottom_reversal_raw * bottom_context_score.values * reliability_amplifier * capitulation_amplifier
            
            
            bottom_reversal_snapshot_score = pd.Series(bottom_reversal_snapshot_values, index=df.index, dtype=np.float32).clip(0, 1)
            final_bottom_reversal_score = self._perform_cognitive_relational_meta_analysis(df, bottom_reversal_snapshot_score)
            states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE'] = final_bottom_reversal_score
        else:
            states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE'] = default_score.copy()
            
        # --- 顶部反转共振分数 ---
        top_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_TOP_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_TOP_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_TOP_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL'),
            'ff': get_unified_score(self.strategy.atomic_states, df.index, 'FF_TOP_REVERSAL')
        }
        top_scores_np = [s.values for d, s in top_sources.items() if top_weights.get(d, 0) > 0]
        top_weights_np = [w for d, w in top_weights.items() if d in top_sources and w > 0]
        if top_scores_np:
            weights_array = np.array(top_weights_np)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(top_scores_np, axis=0)
            # 得到纯粹的顶部反转共振分
            top_reversal_raw = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            
            # 在这里，由认知层统一应用外部上下文进行战略价值评估
            top_reversal_snapshot_values = top_reversal_raw * top_context_score.values
            
            
            top_reversal_snapshot_score = pd.Series(top_reversal_snapshot_values, index=df.index, dtype=np.float32).clip(0, 1)
            final_top_reversal_score = self._perform_cognitive_relational_meta_analysis(df, top_reversal_snapshot_score)
            states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE'] = final_top_reversal_score
        else:
            states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE'] = default_score.copy()
            
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_industry_synergy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.4 · 健壮性升级版】行业-个股协同元融合引擎
        - 核心重构: 更新消费的风险信号为最新的、正确的融合风险信号。
        - 健壮性升级(V1.4): 对行业和个股的看涨/看跌分数的计算增加了健壮性检查。
                           现在会动态检查每个构成信号是否存在，只对有效的信号进行融合，
                           避免了因上游信号缺失可能导致的计算偏差。
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        # 动态构建有效的信号列表并进行融合
        # --- 行业看涨分 ---
        industry_bullish_signals = [
            atomic.get('SCORE_INDUSTRY_MARKUP', default_score),
            atomic.get('SCORE_INDUSTRY_PREHEAT', default_score)
        ]
        valid_industry_bullish = [s.values for s in industry_bullish_signals if s is not None and not s.empty]
        industry_bullish_score = np.maximum.reduce(valid_industry_bullish) if valid_industry_bullish else default_score.values
        # --- 行业看跌分 ---
        industry_bearish_signals = [
            atomic.get('SCORE_INDUSTRY_STAGNATION', default_score),
            atomic.get('SCORE_INDUSTRY_DOWNTREND', default_score)
        ]
        valid_industry_bearish = [s.values for s in industry_bearish_signals if s is not None and not s.empty]
        industry_bearish_score = np.maximum.reduce(valid_industry_bearish) if valid_industry_bearish else default_score.values
        # --- 个股看涨分 ---
        stock_bullish_signals = [
            atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE', default_score),
            self._get_atomic_score(df, 'SCORE_VOL_BREAKOUT_POTENTIAL', 0.0)
        ]
        valid_stock_bullish = [s.values for s in stock_bullish_signals if s is not None and not s.empty]
        stock_bullish_score = np.maximum.reduce(valid_stock_bullish) if valid_stock_bullish else default_score.values
        # --- 个股看跌分 ---
        stock_bearish_signals = [
            atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE', default_score),
            atomic.get('COGNITIVE_FUSION_TOP_REVERSAL', default_score)
        ]
        valid_stock_bearish = [s.values for s in stock_bearish_signals if s is not None and not s.empty]
        stock_bearish_score = np.maximum.reduce(valid_stock_bearish) if valid_stock_bearish else default_score.values
        
        synergy_offense_score = pd.Series(industry_bullish_score, index=df.index) * pd.Series(stock_bullish_score, index=df.index)
        states['COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE'] = synergy_offense_score.astype(np.float32)
        synergy_risk_score = pd.Series(industry_bearish_score, index=df.index) * pd.Series(stock_bearish_score, index=df.index)
        states['COGNITIVE_SCORE_INDUSTRY_SYNERGY_RISK'] = synergy_risk_score.astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_mean_reversion_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 返回值修复版】均值回归网格交易策略信号合成模块
        """
        states = {}
        p = get_params_block(self.strategy, 'mean_reversion_grid_params', {})
        if not get_param_value(p.get('enabled'), True):
            return df # 如果禁用，直接返回 df
        cyclical_regime_threshold = get_param_value(p.get('cyclical_regime_threshold'), 0.4)
        trending_regime_threshold = get_param_value(p.get('trending_regime_threshold'), 0.45)
        is_cyclical_regime = self._get_atomic_score(df, 'SCORE_CYCLICAL_REGIME') > cyclical_regime_threshold
        is_not_trending_regime = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT') < trending_regime_threshold
        context_is_ranging_market = (is_cyclical_regime & is_not_trending_regime).astype(float)
        states['CONTEXT_RANGING_MARKET'] = context_is_ranging_market.astype(np.float32)
        bbp = df.get('BBP_21_2.0_D', pd.Series(0.5, index=df.index)).fillna(0.5)
        buy_opportunity_score = (1 - bbp.clip(0, 1)).astype(np.float32)
        states['SCORE_OPP_MEAN_REVERSION_BUY'] = buy_opportunity_score
        final_playbook_score = context_is_ranging_market * buy_opportunity_score
        states['SCORE_PLAYBOOK_MEAN_REVERSION_GRID_BUY_A'] = final_playbook_score.astype(np.float32)
        
        # 更新 atomic_states 并返回 df 以维持调用链
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_trend_acceleration_cascade(self, df: pd.DataFrame) -> None:
        """
        【V3.1 · 动态级联版】趋势加速级联诊断引擎
        - 核心革命: 引入二阶导数(加速度)和关系动态(协同性)分析。
        - 优化说明: 逻辑高度定制，通过Numpy向量化操作实现斜率、加速度计算及后续融合，保证了复杂逻辑下的执行效率。
        """
        states = {}
        norm_window = 55
        slope_period = 3
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        fusion_weights = p_cognitive.get('cascade_fusion_weights', {'slope': 0.6, 'accel': 0.4})
        w_slope = fusion_weights.get('slope', 0.6)
        w_accel = fusion_weights.get('accel', 0.4)
        def get_dynamic_health(series: pd.Series) -> pd.Series:
            slope = series.diff(slope_period).fillna(0)
            accel = slope.diff(slope_period).fillna(0)
            slope_score = normalize_score(slope.clip(lower=0), df.index, norm_window)
            accel_score = normalize_score(accel.clip(lower=0), df.index, norm_window)
            dynamic_health_score = (slope_score * w_slope + accel_score * w_accel)
            return dynamic_health_score
        health_cache = self.strategy.atomic_states.get('__BEHAVIOR_overall_health', {})
        s_bull = health_cache.get('s_bull', {})
        d_intensity = health_cache.get('d_intensity', {})
        relational_power = self._get_atomic_score(df, 'SCORE_ATOMIC_RELATIONAL_DYNAMICS', 0.5)
        short_term_health = np.maximum(s_bull.get(5, pd.Series(0.5, index=df.index)), relational_power) * d_intensity.get(5, pd.Series(0.5, index=df.index))
        medium_term_health = np.maximum(s_bull.get(21, pd.Series(0.5, index=df.index)), relational_power) * d_intensity.get(21, pd.Series(0.5, index=df.index))
        short_term_dynamic_health = get_dynamic_health(short_term_health)
        medium_term_dynamic_health = get_dynamic_health(medium_term_health)
        temporal_cascade_score = (short_term_dynamic_health * medium_term_dynamic_health)**0.5
        states['COGNITIVE_INTERNAL_TEMPORAL_CASCADE'] = temporal_cascade_score.astype(np.float32)
        resonance_signals = {
            'behavior': self._get_atomic_score(df, 'SCORE_BEHAVIOR_BULLISH_RESONANCE'),
            'chip': self._get_atomic_score(df, 'SCORE_CHIP_BULLISH_RESONANCE'),
            'ff': self._get_atomic_score(df, 'SCORE_FF_BULLISH_RESONANCE'),
            'structure': self._get_atomic_score(df, 'SCORE_STRUCTURE_BULLISH_RESONANCE'),
            'dyn': self._get_atomic_score(df, 'SCORE_DYN_BULLISH_RESONANCE'),
        }
        domain_dynamic_health_scores = [get_dynamic_health(signal) for signal in resonance_signals.values()]
        if domain_dynamic_health_scores:
            stacked_health = np.stack([s.values for s in domain_dynamic_health_scores], axis=0)
            average_dynamic_health = pd.Series(np.mean(stacked_health, axis=0), index=df.index)
            std_dev_health = pd.Series(np.std(stacked_health, axis=0), index=df.index)
            normalized_std_dev = normalize_score(std_dev_health, df.index, norm_window, ascending=True)
            relational_cohesion_score = 1.0 - normalized_std_dev
            domain_cascade_score = (average_dynamic_health * relational_cohesion_score).clip(0, 1)
        else:
            domain_cascade_score = pd.Series(0.0, index=df.index)
        states['COGNITIVE_INTERNAL_DOMAIN_CASCADE'] = domain_cascade_score.astype(np.float32)
        final_cascade_score = (temporal_cascade_score * domain_cascade_score).astype(np.float32)
        states['COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE'] = final_cascade_score
        self.strategy.atomic_states.update(states)

    def _diagnose_comprehensive_top_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.1 · 信号源升级版】综合顶部风险诊断引擎
        - 核心升级: 废弃消费原始的、模棱两可的 SCORE_RISK_ICARUS_FALL 信号。
                      全面换装为消费由 _transmute_pressure_into_opportunity 引擎产出的、
                      经过主力意图审判的 SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW 信号。
        - 收益: 确保了顶层风险引擎的每一个输入都是经过深度加工的高质量情报，提升了最终裁决的准确性。
        """
        states = {}
        signal_name = 'COGNITIVE_RISK_COMPREHENSIVE_TOP'
        # [代码修改开始]
        # --- 亢奋/高潮支柱 ---
        euphoric_pillar_signals = {
            'EUPHORIC_ACCELERATION': self._get_atomic_score(df, 'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', 0.0),
            # 信号源升级：使用经过意图审判的“上影线抛压风险”替换原始的“伊卡洛斯之坠”
            'SELLING_PRESSURE': self._get_atomic_score(df, 'SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW', 0.0),
            'BOARD_HEAVEN_EARTH': self._get_atomic_score(df, 'SCORE_BOARD_HEAVEN_EARTH', 0.0),
        }
        # [代码修改结束]
        euphoric_risk_score = np.maximum.reduce([s.values for s in euphoric_pillar_signals.values()])
        # --- 派发/背叛支柱 ---
        distribution_pillar_signals = {
            'MAIN_FORCE_INTENT_DUEL': self._get_atomic_score(df, 'COGNITIVE_RISK_MAIN_FORCE_HIGH_COST_VS_DISTRIBUTION', 0.0),
            'UPTHRUST_DISTRIBUTION': self._get_atomic_score(df, 'SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0),
            'RETAIL_FOMO_RETREAT': self._get_atomic_score(df, 'COGNITIVE_RISK_RETAIL_FOMO_MAIN_FORCE_RETREAT', 0.0),
            'TRUE_RETREAT': self._get_atomic_score(df, 'COGNITIVE_SCORE_TRUE_RETREAT_RISK', 0.0),
        }
        distribution_risk_score = np.maximum.reduce([s.values for s in distribution_pillar_signals.values()])
        # --- 结构/周期支柱 ---
        structural_pillar_signals = {
            'CONTEXT_TOP': self._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0),
            'CYCLICAL_TOP': self._get_atomic_score(df, 'COGNITIVE_RISK_CYCLICAL_TOP', 0.0),
        }
        structural_risk_score = np.maximum.reduce([s.values for s in structural_pillar_signals.values()])
        # --- 融合与抑制 ---
        raw_fused_risk = np.maximum.reduce([euphoric_risk_score, distribution_risk_score, structural_risk_score])
        shield_score = self._calculate_trend_resilience_shield(df)
        final_risk_values = raw_fused_risk * (1.0 - shield_score.values)
        states[signal_name] = pd.Series(np.clip(final_risk_values, 0, 1), index=df.index, dtype=np.float32)
        return states

    def synthesize_tactical_reversal_resonance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 资金流补完版】战术反转共振融合引擎
        - 核心修复: 将被遗忘的 `SCORE_FF_TACTICAL_REVERSAL` 信号纳入融合，补完资金流维度。
        """
        states = {}
        default_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        tactical_weights = p.get('tactical_resonance_weights', {'mechanics': 0.2, 'chip': 0.2, 'foundation': 0.1, 'behavior': 0.2, 'structure': 0.2, 'ff': 0.1})
        
        tactical_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_TACTICAL_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_TACTICAL_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_TACTICAL_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TACTICAL_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TACTICAL_REVERSAL'),
            # 补完缺失的资金流战术反转信号
            'ff': get_unified_score(self.strategy.atomic_states, df.index, 'SCORE_FF_TACTICAL_REVERSAL')
        }
        
        tactical_scores_np = [s.values for d, s in tactical_sources.items() if tactical_weights.get(d, 0) > 0]
        tactical_weights_np = [w for d, w in tactical_weights.items() if d in tactical_sources and w > 0]
        
        if tactical_scores_np:
            weights_array = np.array(tactical_weights_np)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(tactical_scores_np, axis=0)
            safe_scores = np.maximum(stacked_scores, 1e-9)
            weighted_log_sum = np.sum(np.log(safe_scores) * weights_array[:, np.newaxis], axis=0)
            tactical_reversal_values = np.exp(weighted_log_sum)
            tactical_reversal_score = pd.Series(tactical_reversal_values, index=df.index, dtype=np.float32)
        else:
            tactical_reversal_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        states['COGNITIVE_SCORE_TACTICAL_REVERSAL_RESONANCE'] = tactical_reversal_score
        
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_tactical_opportunity_fusion(self, df: pd.DataFrame) -> None:
        """
        【V1.0 · 新增】战术机会融合引擎
        - 核心职责: 融合来自基础层和微观层的、被遗忘的战术机会信号，形成统一的认知层战术机会分。
        """
        states = {}
        
        # 融合基础层的战术机会信号
        vol_compression_opp = self._get_atomic_score(df, 'SCORE_FOUNDATION_VOL_COMPRESSION_OPP', 0.0)
        ignition_confirmation = self._get_atomic_score(df, 'SCORE_FOUNDATION_IGNITION_CONFIRMATION', 0.0)
        
        # 融合微观行为层的战术机会信号 (点火确认分)
        micro_ignition_confirmation = self._get_atomic_score(df, 'SCORE_IGNITION_CONFIRMATION', 0.0)
        
        # 使用 maximum.reduce 捕捉最强的战术机会
        fused_tactical_opportunity = np.maximum.reduce([
            vol_compression_opp.values,
            ignition_confirmation.values,
            micro_ignition_confirmation.values
        ])
        
        states['COGNITIVE_SCORE_TACTICAL_OPPORTUNITY_FUSION'] = pd.Series(fused_tactical_opportunity, index=df.index, dtype=np.float32)
        self.strategy.atomic_states.update(states)

    async def synthesize_intraday_confirmation(self, df: pd.DataFrame) -> None:
        """
        【V1.1 · 路径修正版】日内微观确认合成模块
        - 核心修正: 修正了获取分钟数据时对 indicator_service 的访问路径。
        """
        # 这是一个简化的触发逻辑：当任何一个核心底部信号大于0.5时，就触发日内分析
        # 未来可以设计更精细的触发器
        trigger_series = (self._get_atomic_score(df, 'COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE') > 0.5)
        dates_to_check = df.index[trigger_series]
        
        if dates_to_check.empty:
            return # 如果没有需要检查的日期，直接返回
        
        print(f"    -> [日内引擎触发] 发现 {len(dates_to_check)} 个潜在机会日，启动微观分析...")
        
        for trade_date in dates_to_check:
            # 修正访问路径：通过 self.strategy.orchestrator 访问顶层的 indicator_service
            df_minute = await self.strategy.orchestrator.indicator_service.stock_trade_dao.get_intraday_kline_by_date(
                self.strategy.stock_code, 
                trade_date.date()
            )
            
            # 调用日内行为引擎进行深度诊断
            intraday_scores = await self.intraday_behavior_engine.run_intraday_diagnostics(df_minute)
            
            # 将返回的日内战术分数注入到对应日期的atomic_states中
            for score_name, score_value in intraday_scores.items():
                # 确保该信号的Series存在
                if score_name not in self.strategy.atomic_states:
                    self.strategy.atomic_states[score_name] = pd.Series(0.0, index=df.index, dtype=np.float32)
                # 更新当天的分数
                self.strategy.atomic_states[score_name].loc[trade_date] = score_value
        
        print(f"    -> [日内引擎] 微观分析完成。")

    def _synthesize_cognitive_expansion_engine(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.9 · 最终净化版】认知扩展信号统一合成引擎
        - 核心修复: 赋予引擎智能。当消费的组件来源是 'atomic' (原子信号)时，直接使用其值，
                      不再进行冗余且错误的MTF二次融合。MTF融合仅用于处理 'df' 来源的原始指标。
                      此修改从根本上解决了“二次加工”导致的逻辑错误。
        """
        states = {}
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        signal_specific_configs = get_param_value(p_cognitive.get('expansion_signal_specific_configs'), {})
        periods = get_param_value(p_cognitive.get('expansion_engine_periods'), [1, 5, 13, 21, 55])
        tf_weights = get_param_value(p_cognitive.get('expansion_engine_tf_weights'), {
            "1": 0.05, "5": 0.2, "13": 0.3, "21": 0.3, "55": 0.15
        })
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        expansion_signal_configs = {
            'COGNITIVE_RISK_LIQUIDITY_TRAP': {
                'description': '【V3.0 · 协议修正版】融合“主力持续出逃”、“流动性真空”和“买盘真空”三大核心证据。采用算术平均，避免湮灭。',
                'components': [
                    {'source': 'df', 'name': 'main_force_net_flow_consensus_sum_5d_D', 'transform': 'neg_clip_abs', 'weight': 0.4},
                    {'source': 'atomic', 'name': 'SCORE_RISK_LIQUIDITY_VACUUM', 'weight': 0.4},
                    {'source': 'df', 'name': 'realized_support_intensity_D', 'transform': 'inverse', 'weight': 0.2},
                ]
            },
        }
        df['is_limit_up'] = df.get('close_D', 0) >= df.get('up_limit_D', np.inf) * 0.995
        df['volume_spike'] = df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D'])
        df['significant_gap_up'] = (df['open_D'] - df['pre_close_D']) / df['pre_close_D'].replace(0, np.nan)
        df['gap_not_filled'] = (df['low_D'] > df['pre_close_D'])
        df['touched_limit_down'] = (df['low_D'] <= df.get('down_limit_D', 0) * 1.005)
        for signal_name, config in expansion_signal_configs.items():
            specific_config = signal_specific_configs.get(signal_name, {})
            fusion_method = specific_config.get('fusion_method', 'arithmetic')
            fused_component_scores = []
            component_weights = []
            gate_scores = []
            for comp in config.get('components', []):
                source_series_raw = None
                if comp['source'] == 'df':
                    source_series_raw = df.get(comp['name'], pd.Series(0.0, index=df.index))
                elif comp['source'] == 'atomic':
                    source_series_raw = self._get_atomic_score(df, comp['name'], 0.0)
                if source_series_raw is None or source_series_raw.empty:
                    source_series_raw = pd.Series(0.0, index=df.index)
                transformed_series = source_series_raw.copy()
                transform = comp.get('transform')
                params = comp.get('params', ())
                if transform == 'inverse':
                    transformed_series = 1.0 - transformed_series
                elif transform == 'inverse_proximity':
                    transformed_series = 1.0 - (1.0 - transformed_series).clip(0, 1)
                elif transform == 'neg_clip':
                    transformed_series = -transformed_series.clip(upper=0)
                elif transform == 'pos_clip':
                    transformed_series = transformed_series.clip(lower=0)
                elif transform == 'neg_clip_abs':
                    transformed_series = transformed_series.clip(upper=0).abs()
                elif transform == 'is_positive':
                    transformed_series = (transformed_series > 0).astype(float)
                elif transform == 'shift':
                    transformed_series = transformed_series.shift(params[0]).fillna(0)
                elif transform == 'shift_lt':
                    transformed_series = transformed_series.shift(params[0]).fillna(params[1]) < params[1]
                # [代码修改开始]
                # 智能处理：仅对原始指标('df')应用MTF融合，对成品原子信号('atomic')直接使用
                if comp['source'] == 'atomic':
                    fused_component_series = transformed_series
                else: # comp['source'] == 'df'
                    fused_component_series = pd.Series(0.0, index=df.index)
                    if total_weight > 0:
                        for p in periods:
                            weight = numeric_tf_weights.get(p, 0) / total_weight
                            normalized_series = normalize_score(transformed_series, df.index, p)
                            fused_component_series += normalized_series * weight
                    else:
                        fused_component_series = normalize_score(transformed_series, df.index, 55)
                # [代码修改结束]
                if comp.get('is_gate', False):
                    gate_scores.append(fused_component_series.values)
                else:
                    fused_component_scores.append(fused_component_series.values)
                    component_weights.append(comp.get('weight', 1.0))
            if not fused_component_scores:
                snapshot_score_values = np.ones(len(df.index), dtype=np.float32)
            else:
                stacked_scores = np.stack(fused_component_scores, axis=0)
                if fusion_method == 'geometric':
                    snapshot_score_values = np.prod(stacked_scores, axis=0) ** (1.0 / len(fused_component_scores))
                else:
                    snapshot_score_values = np.average(stacked_scores, axis=0, weights=np.array(component_weights))
            if gate_scores:
                for gate in gate_scores:
                    snapshot_score_values *= gate
            snapshot_score = pd.Series(snapshot_score_values, index=df.index, dtype=np.float32)
            final_dynamic_score = self._perform_cognitive_relational_meta_analysis(df, snapshot_score)
            states[signal_name] = final_dynamic_score
        states.update(self._diagnose_main_force_high_cost_vs_distribution(df))
        return states

    def _perform_cognitive_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V4.1 · 加速度修复版】认知层专用的关系元分析核心引擎
        - 核心修复: 修正了“加速度”计算的致命逻辑错误。加速度是速度的一阶导数，
                      因此其计算应为 relationship_trend.diff(1)，而不是错误的 diff(meta_window)。
                      此修复将从根本上恢复关系元分析的数学正确性。
        """
        
        p_conf = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 权重调整为状态主导
        w_state = get_param_value(p_meta.get('state_weight'), 0.6)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.2)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.2)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        bipolar_snapshot = (snapshot_score * 2 - 1).clip(-1, 1)
        relationship_trend = bipolar_snapshot.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        
        # [代码修改开始]
        # 致命错误修复：加速度是速度(trend)的一阶导数，应使用 diff(1) 而不是 diff(meta_window)
        relationship_accel = relationship_trend.diff(1).fillna(0)
        # [代码修改结束]
        
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        bullish_state = bipolar_snapshot.clip(0, 1)
        bullish_velocity = velocity_score.clip(0, 1)
        bullish_acceleration = acceleration_score.clip(0, 1)
        total_bullish_force = (
            bullish_state * w_state +
            bullish_velocity * w_velocity +
            bullish_acceleration * w_acceleration
        )
        bearish_state = (bipolar_snapshot.clip(-1, 0) * -1)
        bearish_velocity = (velocity_score.clip(-1, 0) * -1)
        bearish_acceleration = (acceleration_score.clip(-1, 0) * -1)
        total_bearish_force = (
            bearish_state * w_state +
            bearish_velocity * w_velocity +
            bearish_acceleration * w_acceleration
        )
        net_force = (total_bullish_force - total_bearish_force).clip(-1, 1)
        # 植入“状态主导协议”护栏
        final_bipolar_score = np.where(bipolar_snapshot >= 0, net_force.clip(lower=0), net_force.clip(upper=0))
        final_unipolar_score = (pd.Series(final_bipolar_score, index=df.index) + 1) / 2.0
        return final_unipolar_score.astype(np.float32)

    def _calculate_aegis_shield_context(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】“雅典娜的神盾”上下文分数计算引擎
        - 核心职责: 评估当前上升趋势的健康度和强韧性，生成一个[0, 1]区间的“神盾分数”。
                      分数越高，代表趋势越健康，对常规顶部风险信号的抑制能力越强。
        - 核心逻辑: 融合四大支柱——趋势质量、结构完整性、波动率状态、价格位置。
        """
        # 从配置中读取神盾的构建参数
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_shield = get_param_value(p_cognitive.get('aegis_shield_params'), {})
        if not get_param_value(p_shield.get('enabled'), True):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        
        weights = get_param_value(p_shield.get('fusion_weights'), {})
        
        # --- 神盾的四大支柱 ---
        # 支柱一: 趋势质量 (Trend Quality) - 趋势的综合健康度
        trend_quality_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        
        # 支柱二: 结构完整性 (Structural Integrity) - 趋势的骨架是否坚固
        structural_integrity_score = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        
        # 支柱三: 波动率状态 (Volatility State) - 市场是冷静蓄力还是恐慌混乱
        # 健康的趋势中继，通常伴随着波动率的压缩
        volatility_compression_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_VOL_COMPRESSION_FUSED', 0.0)
        
        # 支柱四: 价格位置 (Price Location) - 价格是否处于危险的“超买”区域
        # 我们需要一个“不过热”的分数，所以用 1 减去过热分
        bias_overheat_score = self._get_atomic_score(df, 'FUSED_RISK_SCORE_PRICE_LOCATION', 0.0)
        not_overheated_score = (1.0 - bias_overheat_score).clip(0, 1)
        
        # --- 融合四大支柱，铸造神盾 ---
        pillars = {
            'trend_quality': trend_quality_score,
            'structural_integrity': structural_integrity_score,
            'volatility_compression': volatility_compression_score,
            'price_location': not_overheated_score
        }
        
        scores_to_fuse = []
        weights_to_fuse = []
        for name, score in pillars.items():
            weight = weights.get(name, 0.25)
            if weight > 0:
                scores_to_fuse.append(score.values)
                weights_to_fuse.append(weight)
        
        if not scores_to_fuse:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
            
        # 使用加权几何平均数进行融合，要求各方面都比较健康
        weights_array = np.array(weights_to_fuse)
        weights_array /= weights_array.sum()
        stacked_scores = np.stack(scores_to_fuse, axis=0)
        safe_scores = np.maximum(stacked_scores, 1e-9) # 避免log(0)
        
        aegis_shield_values = np.exp(np.sum(np.log(safe_scores) * weights_array[:, np.newaxis], axis=0))
        
        return pd.Series(aegis_shield_values, index=df.index, dtype=np.float32)

    def _diagnose_suppression_vs_retreat(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.4 · 信号替代修复版】“真伪识别：打压 vs 撤退”诊断引擎
        - 核心修复: 使用 `retail_capitulation_distribution_D` 和 `profit_taking_urgency_D` 作为替代品，
                      修复了因信号缺失导致的计算错误，并使证据链逻辑更清晰。
        """
        states = {}
        norm_window = 55
        p = 5
        to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) *
                   normalize_score(df.get(f'SLOPE_{p}_retail_capitulation_distribution_D'), df.index, norm_window, ascending=True))**0.5
        to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) *
                     normalize_score(df.get(f'SLOPE_{p}_profit_taking_urgency_D'), df.index, norm_window, ascending=True))**0.5
        short_term_transfer_snapshot = (to_main - to_retail).astype(np.float32)
        recent_distribution_strength = (short_term_transfer_snapshot.rolling(3).mean().clip(-1, 0) * -1).astype(np.float32)
        chip_reversal_raw = self._get_atomic_score(df, 'SCORE_CHIP_BOTTOM_REVERSAL', 0.0)
        behavior_reversal_raw = self._get_atomic_score(df, 'SCORE_BEHAVIOR_BOTTOM_REVERSAL', 0.0)
        dyn_reversal_raw = self._get_atomic_score(df, 'SCORE_DYN_BOTTOM_REVERSAL', 0.0)
        reversal_strength = np.maximum.reduce([
            chip_reversal_raw.values,
            behavior_reversal_raw.values,
            dyn_reversal_raw.values
        ])
        reversal_strength = pd.Series(reversal_strength, index=df.index, dtype=np.float32)
        dyn_bullish_resonance = self._get_atomic_score(df, 'SCORE_DYN_BULLISH_RESONANCE', 0.0)
        behavior_bullish_resonance = self._get_atomic_score(df, 'SCORE_BEHAVIOR_BULLISH_RESONANCE', 0.0)
        reversal_dynamic_quality = (dyn_bullish_resonance * behavior_bullish_resonance)**0.5
        winner_conviction_0_1 = (self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION', 0.0).clip(-1, 1) * 0.5 + 0.5)
        winner_belief_score = winner_conviction_0_1
        winner_capitulation_score = (1.0 - winner_conviction_0_1) ** 0.7
        trend_quality_context = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        panic_absorption_score = self._get_atomic_score(df, 'SCORE_MICRO_PANIC_ABSORPTION', 0.0)
        structural_support_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_BOTTOM_CONFIRMED', 0.0)
        absorption_evidence_chain = (
            trend_quality_context *
            panic_absorption_score *
            winner_belief_score *
            (1 + structural_support_score * 0.5)
        )
        tactical_suppression_score = (
            recent_distribution_strength *
            reversal_strength *
            reversal_dynamic_quality *
            absorption_evidence_chain
        ).clip(0, 1)
        states['COGNITIVE_SCORE_TACTICAL_SUPPRESSION'] = tactical_suppression_score.astype(np.float32)
        trend_decay_context = 1.0 - trend_quality_context
        no_absorption_score = 1.0 - panic_absorption_score
        bull_trap_evidence = 1.0 - reversal_dynamic_quality
        retreat_evidence_chain = (
            trend_decay_context *
            no_absorption_score *
            winner_capitulation_score *
            bull_trap_evidence
        )
        true_retreat_score = (recent_distribution_strength * retreat_evidence_chain).clip(0, 1)
        states['COGNITIVE_SCORE_TRUE_RETREAT_RISK'] = true_retreat_score.astype(np.float32)
        return states

    def _calculate_cognitive_trend_health(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】五维趋势健康度评估引擎
        - 核心逻辑: 作为认知层的专属趋势评估器，从“排列、速度、加速度、关系、元动力”
                      五个维度对趋势进行最全面、最权威的直接评估。
        """
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_health = get_param_value(p_cognitive.get('trend_health_fusion_weights'), {})
        weights = {
            'alignment': get_param_value(p_health.get('alignment'), 0.2),
            'velocity': get_param_value(p_health.get('velocity'), 0.15),
            'acceleration': get_param_value(p_health.get('acceleration'), 0.15),
            'relational': get_param_value(p_health.get('relational'), 0.2),
            'meta_dynamics': get_param_value(p_health.get('meta_dynamics'), 0.3)
        }
        norm_window = 55
        ma_periods = [5, 13, 21, 55, 89]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods if f'EMA_{p}_D' in df.columns]
        if len(ma_cols) < 2:
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods if f'SLOPE_{p}_EMA_{p}_D' in df.columns]
        velocity_health = np.mean([normalize_score(df[col], df.index, norm_window) for col in slope_cols], axis=0) if slope_cols else np.full(len(df.index), 0.5)
        accel_cols = [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods if f'ACCEL_{p}_EMA_{p}_D' in df.columns]
        acceleration_health = np.mean([normalize_score(df[col], df.index, norm_window) for col in accel_cols], axis=0) if accel_cols else np.full(len(df.index), 0.5)
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True)
        meta_dynamics_cols = ['SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D']
        valid_meta_cols = [col for col in meta_dynamics_cols if col in df.columns]
        meta_dynamics_health = np.mean([normalize_score(df[col], df.index, norm_window) for col in valid_meta_cols], axis=0) if valid_meta_cols else np.full(len(df.index), 0.5)
        scores = np.stack([alignment_health, velocity_health, acceleration_health, relational_health, meta_dynamics_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()
        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _calculate_cyclical_top_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】计算“周期顶”风险
        - 核心逻辑: 融合“主导周期强度”和“当前相位”，当市场处于一个强周期的波峰位置时，风险分会显著提高。
        """
        states = {}
        cycle_power = self._get_atomic_score(df, 'DOMINANT_CYCLE_POWER', 0.0)
        # 将相位[-1, 1]映射为分数[0, 1]，-1(波谷)对应0分，+1(波峰)对应1分
        cycle_phase_score = (self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE', 0.0) + 1) / 2.0
        
        # 周期顶风险 = 周期强度 * 相位位置
        cyclical_top_risk = (cycle_power * cycle_phase_score).clip(0, 1)
        
        states['COGNITIVE_RISK_CYCLICAL_TOP'] = cyclical_top_risk.astype(np.float32)
        print(f"      -> [CognitiveIntelligence:_calculate_cyclical_top_risk] 已生成周期顶风险信号。")
        return states

    def _diagnose_main_force_high_cost_vs_distribution(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 全息意图对决版】主力意图对决风险诊断引擎
        - 核心升级: 采纳MTF（多时间框架）分析，在每个周期上独立进行“决心 vs 背叛”的对决，
                      然后加权融合成一个更可靠的“综合净意图”，最后进行风险裁决。
                      这解决了单一维度判断的战略短视和信号脆弱问题。
        """
        states = {}
        signal_name = 'COGNITIVE_RISK_MAIN_FORCE_HIGH_COST_VS_DISTRIBUTION'
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        periods = get_param_value(p_cognitive.get('expansion_engine_periods'), [1, 5, 13, 21, 55])
        tf_weights = get_param_value(p_cognitive.get('expansion_engine_tf_weights'), {})
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        bipolar_intent_by_period = {}
        # 1. 在每个时间框架上独立进行意图对决
        for p in periods:
            # 获取看涨证据：主力追高买入的决心
            urgency_score_raw = self._get_atomic_score(df, 'PROCESS_META_MAIN_FORCE_URGENCY', 0.0)
            urgency_score_norm = normalize_score(urgency_score_raw, df.index, p)
            # 获取看跌证据：主力拉高派发的背叛
            distribution_score_raw = df.get('main_force_rally_distribution_D', pd.Series(0.0, index=df.index))
            distribution_score_norm = normalize_score(distribution_score_raw, df.index, p)
            # 计算该周期的“净意图分”
            bipolar_intent_by_period[p] = urgency_score_norm - distribution_score_norm
        # 2. 加权融合所有周期的“净意图分”
        final_bipolar_intent = pd.Series(0.0, index=df.index)
        if total_weight > 0:
            for p in periods:
                weight = numeric_tf_weights.get(p, 0) / total_weight
                final_bipolar_intent += bipolar_intent_by_period.get(p, 0.0) * weight
        # 3. 风险裁决：只取“背叛 > 决心”的部分作为风险
        risk_score = -(final_bipolar_intent.clip(upper=0))
        states[signal_name] = risk_score.astype(np.float32)
        return states

    def _calculate_trend_resilience_shield(self, df: pd.DataFrame) -> pd.Series:
        """
        【V2.0 · 动态韧性版】“趋势韧性神盾”计算引擎
        - 核心重构: 废除脆弱的“加权几何平均数”，换用更具韧性的“加权算术平均数”，避免因单点故障导致神盾完全失效。
        - 战略升维: 引入关系元分析。最终神盾分数 = 静态韧性 * (1 + 动态韧性)，使其能感知趋势的“生命力”。
        """
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_shield = get_param_value(p_cognitive.get('trend_resilience_shield_params'), {})
        if not get_param_value(p_shield.get('enabled'), True):
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        weights = get_param_value(p_shield.get('fusion_weights'), {})
        # --- 神盾的四大支柱 ---
        pillars = {
            'trend_quality': self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0),
            'structural_health': self._get_atomic_score(df, 'SCORE_STRUCTURE_BULLISH_RESONANCE', 0.0),
            'fund_flow_health': self._get_atomic_score(df, 'SCORE_FF_BULLISH_RESONANCE', 0.0),
            'chip_health': self._get_atomic_score(df, 'SCORE_CHIP_BULLISH_RESONANCE', 0.0)
        }
        # --- 阶段一：计算静态韧性分 (Static Resilience) ---
        static_resilience_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        total_weight = sum(weights.get(name, 0) for name in pillars.keys())
        if total_weight > 0:
            for name, score in pillars.items():
                weight = weights.get(name, 0.25)
                static_resilience_score += score * (weight / total_weight)
        # --- 阶段二：计算动态韧性分 (Dynamic Resilience) ---
        # 对静态韧性分本身进行关系元分析，只取其看涨的动态部分
        p_meta = get_param_value(p_cognitive.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        meta_window = 5
        relationship_trend = static_resilience_score.diff(meta_window).fillna(0)
        velocity_score_bipolar = normalize_to_bipolar(relationship_trend, df.index, norm_window)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score_bipolar = normalize_to_bipolar(relationship_accel, df.index, norm_window)
        # 只取动态分的正向部分作为加成
        dynamic_resilience_bonus = (
            velocity_score_bipolar.clip(0, 1) * w_velocity +
            acceleration_score_bipolar.clip(0, 1) * w_acceleration
        )
        # --- 阶段三：最终融合 ---
        # 最终神盾分数 = 静态韧性 * (1 + 动态韧性加成)
        final_shield_score = (static_resilience_score * (1 + dynamic_resilience_bonus)).clip(0, 1)
        return final_shield_score.astype(np.float32)

















