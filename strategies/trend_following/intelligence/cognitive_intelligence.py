# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, fuse_multi_level_scores
from strategies.trend_following.intelligence.micro_behavior_engine import MicroBehaviorEngine
from strategies.trend_following.intelligence.tactic_engine import TacticEngine

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

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """
        【健壮性修复版】安全地从原子状态库中获取分数。
        """
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def synthesize_cognitive_scores(self, df: pd.DataFrame, pullback_enhancements: Dict) -> pd.DataFrame:
        """
        【V2.3 · 认知升级版】顶层认知总分合成模块
        - 核心重构 (本次修改):
          - [信号消费] 全面审查并更新了所有认知融合模块，确保它们消费的是最新、最可靠的终极原子信号。
          - [周期整合] 将 `CyclicalIntelligence` 产出的FFT周期信号整合到“趋势质量”和“回踩”诊断中。
        - 收益: 认知层的决策质量实现了质的飞跃，能够更深刻地理解市场状态。
        """
        # print("        -> [顶层认知总分合成模块 V2.3 · 认知升级版] 启动...") # 更新版本号
        
        # --- 步骤 0: 预处理，确保所有底层信号已就绪 ---
        # 在一个理想的架构中，这一步由更高层的 `IntelligenceLayer` 保证。
        # 此处我们假设 `self.strategy.atomic_states` 已被所有底层引擎填充。

        # --- 步骤 1: 调用微观行为引擎，生成深层行为模式信号 ---
        micro_behavior_states = self.micro_behavior_engine.run_micro_behavior_synthesis(df)
        self.strategy.atomic_states.update(micro_behavior_states)

        # --- 步骤 2: 调用战术引擎，生成具体战术信号 ---
        tactic_states = self.tactic_engine.run_tactic_synthesis(df, pullback_enhancements)
        self.strategy.atomic_states.update(tactic_states)
        self.strategy.playbook_states.update({k: v for k, v in tactic_states.items() if k.startswith('PLAYBOOK_')})

        # --- 步骤 3: 执行本模块的核心认知融合任务 ---
        # 首先调用新的波动率合成器，确保其信号可被下游消费
        volatility_states = self._synthesize_volatility_signals(df)
        self.strategy.atomic_states.update(volatility_states)

        # 确保后续调用顺序正确
        df = self.synthesize_trend_quality_score(df)
        df = self.synthesize_pullback_states(df)
        df = self.synthesize_structural_fusion_scores(df)
        df = self.synthesize_ultimate_confirmation_scores(df)
        df = self.synthesize_ignition_resonance_score(df)
        df = self.synthesize_reversal_resonance_scores(df)
        df = self.synthesize_industry_synergy_signals(df)
        df = self.synthesize_mean_reversion_signals(df)
        
        # --- 步骤 4: 汇总所有S级的“机会”与“风险”类认知分数 ---
        bullish_scores = [
            self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE_S').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_CONSOLIDATION_BREAKOUT_OPP_A').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE_S').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_REVERSAL_RELIABILITY').values,
        ]
        cognitive_bullish_score = np.maximum.reduce(bullish_scores)
        self.strategy.atomic_states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)

        fused_risk_states = self.synthesize_fused_risk_scores(df)
        self.strategy.atomic_states.update(fused_risk_states)
        
        # print("        -> [顶层认知总分合成模块 V2.3] 认知升级完成。")
        return df

    def synthesize_trend_quality_score(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.3 · 重构版】趋势质量融合评分模块
        """
        # 调用 utils.fuse_multi_level_scores
        behavior_health_score = 1.0 - fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        fund_flow_health_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        structural_health_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_health_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE')
        
        # ... (后续逻辑保持不变) ...
        regime_health_score_hurst = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME')
        regime_health_score_fft = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT')
        regime_health_score = (regime_health_score_hurst + regime_health_score_fft) / 2.0
        
        p_chip_pillars = get_params_block(self.strategy, 'trend_quality_params', {}).get('chip_pillar_weights', {})
        chip_health_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        total_chip_weight = 0.0
        chip_pillar_names = ['quantitative', 'advanced', 'internal', 'holder', 'fault']
        for pillar_name in chip_pillar_names:
            weight = p_chip_pillars.get(pillar_name, 0.2)
            pillar_health_signal_name = f'SCORE_CHIP_PILLAR_{pillar_name.upper()}_HEALTH'
            pillar_score = self._get_atomic_score(df, pillar_health_signal_name, 0.0)
            chip_health_score += pillar_score * weight
            total_chip_weight += weight
        if total_chip_weight > 0:
            chip_health_score /= total_chip_weight

        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        
        trend_quality_score = (
            behavior_health_score * weights.get('behavior', 0.20) +
            chip_health_score * weights.get('chip', 0.30) +
            fund_flow_health_score * weights.get('fund_flow', 0.15) +
            structural_health_score * weights.get('structural', 0.15) + 
            mechanics_health_score * weights.get('mechanics', 0.10) +
            regime_health_score * weights.get('regime', 0.10)
        )
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = trend_quality_score.astype(np.float32)
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.5 · 重构版】认知层回踩状态合成模块
        """
        states = {}
        is_pullback_day = (df['pct_change_D'] < 0).astype(float)
        constructive_context_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1).fillna(0.0)
        shrinking_volume_score = self._get_atomic_score(df, 'SCORE_VOL_WEAKENING_DROP', 0.0) 
        
        cycle_trough_score = (1 - self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE', 0.0).fillna(0.0)) / 2.0 
        
        # 调用 utils.fuse_multi_level_scores
        winner_holding_tight_score = 1.0 - fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'TOP_REVERSAL')
        chip_stable_score = 1.0 - fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FALLING_RESONANCE')
        
        healthy_pullback_score = (
            is_pullback_day * constructive_context_score *
            gentle_drop_score * shrinking_volume_score *
            winner_holding_tight_score * chip_stable_score *
            (1 + cycle_trough_score * 0.5)
        )
        states['COGNITIVE_SCORE_PULLBACK_HEALTHY_S'] = healthy_pullback_score.astype(np.float32)
        
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1).fillna(0.0)
        # 调用 utils.fuse_multi_level_scores
        panic_selling_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_BEARISH_RESONANCE')
        suppressive_pullback_score = (
            is_pullback_day * constructive_context_score *
            significant_drop_score * panic_selling_score * winner_holding_tight_score
        )
        states['COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE_S'] = suppressive_pullback_score.astype(np.float32)
        
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_fused_risk_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.3 · 性能优化版】风险元融合模块
        - 本次优化:
          - [效率] 重构了维度内融合的计算流程。通过“预处理-批量获取-缓存计算”的模式，
                    将原先在循环中对Pandas Series的反复查找和转换，优化为对预先缓存的
                    NumPy数组的直接访问，显著降低了函数调用和数据转换开销，提升了执行效率。
        """
        states = {}
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            states['COGNITIVE_FUSED_RISK_SCORE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
            
        risk_categories = p_fused_risk.get('risk_categories', {})
        p_fusion_params = p_fused_risk.get('intra_dimension_fusion_params', {})
        secondary_risk_discount = p_fusion_params.get('secondary_risk_discount', 0.3)
        
        # --- 优化步骤 1: 预处理，收集所有需要的信号并批量获取 ---
        all_required_signals = set()
        for category_name, signals in risk_categories.items():
            if category_name != "说明":
                all_required_signals.update(s for s in signals if s != "说明")

        # 批量获取Series并转换为Numpy数组，存入缓存字典
        signal_numpy_cache = {
            sig_name: self._get_atomic_score(df, sig_name, 0.0).values
            for sig_name in all_required_signals
        }
        default_numpy_array = np.zeros(len(df.index), dtype=np.float32)

        # --- 维度内融合 (Intra-dimension Fusion) ---
        fused_dimension_scores = {}
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            
            category_signal_scores = []
            for signal_name, signal_params in signals.items():
                if signal_name == "说明": continue
                
                # 直接从缓存的Numpy数组中读取数据
                atomic_score_np = signal_numpy_cache.get(signal_name, default_numpy_array)
                
                is_inverse = signal_params.get('inverse', False)
                processed_score = 1.0 - atomic_score_np if is_inverse else atomic_score_np
                weight = signal_params.get('weight', 1.0)
                final_signal_score = processed_score * weight
                category_signal_scores.append(final_signal_score)
                
            if category_signal_scores:
                stacked_scores = np.stack(category_signal_scores, axis=1)
                sorted_scores = np.sort(stacked_scores, axis=1)
                primary_risk_values = sorted_scores[:, -1]
                secondary_risk_values = sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else 0
                dimension_risk_values = primary_risk_values + secondary_risk_values * secondary_risk_discount
                dimension_risk_score = pd.Series(dimension_risk_values, index=df.index, dtype=np.float32)
                fused_dimension_scores[category_name] = dimension_risk_score
                states[f'FUSED_RISK_SCORE_{category_name.upper()}'] = dimension_risk_score
            else:
                fused_dimension_scores[category_name] = pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 维度间融合 (Inter-dimension Fusion) & 风险共振惩罚 ---
        p_dynamic_weighting = p_fused_risk.get('dynamic_weighting_params', {})
        base_weights = p_dynamic_weighting.get('base_weights', {})
        context_adjustments = p_dynamic_weighting.get('context_adjustments', {})
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        
        total_fused_risk_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        is_early_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_EARLY', pd.Series(False, index=df.index))
        is_late_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_LATE', pd.Series(False, index=df.index))
        
        for category_name, weight in base_weights.items():
            if category_name in fused_dimension_scores:
                current_weight = pd.Series(weight, index=df.index)
                if get_param_value(p_dynamic_weighting.get('enabled'), True):
                    early_adjustments = context_adjustments.get("CONTEXT_TREND_STAGE_EARLY", {})
                    if category_name in early_adjustments:
                        current_weight = current_weight.where(~is_early_stage, current_weight * early_adjustments[category_name])
                    late_adjustments = context_adjustments.get("CONTEXT_TREND_STAGE_LATE", {})
                    if category_name in late_adjustments:
                        current_weight = current_weight.where(~is_late_stage, current_weight * late_adjustments[category_name])
                total_fused_risk_score += fused_dimension_scores[category_name] * current_weight
                
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = get_param_value(p_resonance.get('core_risk_dimensions'), [])
            min_dims = get_param_value(p_resonance.get('min_dimensions_for_resonance'), 2)
            threshold = get_param_value(p_resonance.get('risk_score_threshold'), 150)
            penalty_multiplier = get_param_value(p_resonance.get('penalty_multiplier'), 1.2)
            high_risk_dimension_count = pd.Series(0, index=df.index)
            for dim in core_dims:
                if dim in fused_dimension_scores:
                    high_risk_dimension_count += (fused_dimension_scores[dim] > threshold).astype(int)
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            total_fused_risk_score = total_fused_risk_score.where(~is_resonance_triggered, total_fused_risk_score * penalty_multiplier)
            states['FUSED_RISK_RESONANCE_PENALTY_ACTIVE'] = is_resonance_triggered
            
        states['COGNITIVE_FUSED_RISK_SCORE'] = total_fused_risk_score.astype(np.float32)
        return states

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.4 终极结构层信号适配版】结构化元信号融合模块
        """
        # print("        -> [结构化元信号融合模块 V2.4 终极结构层信号适配版] 启动...")
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        foundation_bullish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FOUNDATION_BULLISH_RESONANCE')
        foundation_bearish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FOUNDATION_BEARISH_RESONANCE')
        foundation_bottom = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL')
        foundation_top = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FOUNDATION_TOP_REVERSAL')
        structure_bullish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        structure_bearish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_BEARISH_RESONANCE')
        structure_bottom = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        structure_top = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL')
        behavior_bullish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_BULLISH_RESONANCE')
        behavior_bearish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_BEARISH_RESONANCE')
        behavior_bottom = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL')
        behavior_top = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        all_scores = [
            foundation_bullish, foundation_bearish, foundation_bottom, foundation_top,
            structure_bullish, structure_bearish, structure_bottom, structure_top,
            behavior_bullish, behavior_bearish, behavior_bottom, behavior_top
        ]
        for i, score in enumerate(all_scores):
            if not isinstance(score, pd.Series) or score.index.empty:
                all_scores[i] = default_score.copy()
            else:
                all_scores[i] = score.reindex(df.index).fillna(0.0)
        (foundation_bullish, foundation_bearish, foundation_bottom, foundation_top,
         structure_bullish, structure_bearish, structure_bottom, structure_top,
         behavior_bullish, behavior_bearish, behavior_bottom, behavior_top) = all_scores
        states['COGNITIVE_FUSION_BULLISH_RESONANCE_S'] = (foundation_bullish * structure_bullish * behavior_bullish).astype(np.float32)
        states['COGNITIVE_FUSION_BEARISH_RESONANCE_S'] = (foundation_bearish * structure_bearish * behavior_bearish).astype(np.float32)
        states['COGNITIVE_FUSION_BOTTOM_REVERSAL_S'] = (foundation_bottom * structure_bottom * behavior_bottom).astype(np.float32)
        states['COGNITIVE_FUSION_TOP_REVERSAL_S'] = (foundation_top * structure_top * behavior_top).astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_ultimate_confirmation_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.2 架构统一版】终极确认融合模块
        """
        # print("        -> [终极确认融合模块 V1.2 架构统一版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        required_fusion_signals = [
            'COGNITIVE_FUSION_BULLISH_RESONANCE_S', 'COGNITIVE_FUSION_BEARISH_RESONANCE_S',
            'COGNITIVE_FUSION_BOTTOM_REVERSAL_S', 'COGNITIVE_FUSION_TOP_REVERSAL_S'
        ]
        missing_fusion_signals = [s for s in required_fusion_signals if s not in atomic]
        if missing_fusion_signals:
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        fusion_bullish = atomic.get('COGNITIVE_FUSION_BULLISH_RESONANCE_S', default_series)
        fusion_bearish = atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE_S', default_series)
        fusion_bottom = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL_S', default_series)
        fusion_top = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL_S', default_series)
        pattern_bullish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'PATTERN_BULLISH_RESONANCE')
        pattern_bearish = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'PATTERN_BEARISH_RESONANCE')
        pattern_bottom = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'PATTERN_BOTTOM_REVERSAL')
        pattern_top = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'PATTERN_TOP_REVERSAL')
        states['COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION_S'] = (fusion_bullish * pattern_bullish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION_S'] = (fusion_bearish * pattern_bearish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION_S'] = (fusion_bottom * pattern_bottom).astype(np.float32)
        states['COGNITIVE_ULTIMATE_TOP_CONFIRMATION_S'] = (fusion_top * pattern_top).astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_ignition_resonance_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.0 王牌信号增强版】多域点火共振分数合成模块
        """
        # print("        -> [多域点火共振分数合成模块 V2.0 王牌信号增强版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        chip_playbook_ignition = self._get_atomic_score(df, 'SCORE_CHIP_PLAYBOOK_VACUUM_BREAKOUT', 0.0)
        chip_consensus_ignition = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        behavioral_ignition = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_BULLISH_RESONANCE')
        structural_breakout = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_ignition = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE')
        volatility_breakout = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'VOL_BREAKOUT')
        fund_flow_ignition_old = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        fund_flow_conviction_breakout = self._get_atomic_score(df, 'SCORE_FF_PLAYBOOK_CONVICTION_BREAKOUT', 0.0)
        general_ignition_resonance = (
            behavioral_ignition * structural_breakout * mechanics_ignition *
            chip_consensus_ignition * fund_flow_ignition_old * volatility_breakout
        )
        ignition_resonance_score = np.maximum.reduce([
            chip_playbook_ignition.values, 
            general_ignition_resonance.values,
            fund_flow_conviction_breakout.values
        ]).astype(np.float32)
        states['COGNITIVE_SCORE_IGNITION_RESONANCE_S'] = pd.Series(ignition_resonance_score, index=df.index)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_reversal_resonance_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V2.6 融合逻辑修复版】多域反转共振分数合成模块
        """
        # print("        -> [多域反转共振分数合成模块 V2.6 融合逻辑修复版] 启动...")
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = get_param_value(p.get('bottom_resonance_weights'), {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        top_weights = get_param_value(p.get('top_resonance_weights'), {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        mechanics_bottom_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'DYN_BOTTOM_REVERSAL')
        chip_bottom_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL')
        foundation_bottom_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL')
        behavior_bottom_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL')
        structure_bottom_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        total_bottom_score = pd.Series(0.0, index=df.index)
        total_bottom_weight = 0.0
        bottom_sources = {
            'mechanics': mechanics_bottom_score, 'chip': chip_bottom_score, 'foundation': foundation_bottom_score,
            'behavior': behavior_bottom_score, 'structure': structure_bottom_score
        }
        for domain, weight in bottom_weights.items():
            if domain in bottom_sources and weight > 0:
                total_bottom_score += bottom_sources[domain] * weight
                total_bottom_weight += weight
        bottom_reversal_score = total_bottom_score / total_bottom_weight if total_bottom_weight > 0 else default_score
        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE_S'] = bottom_reversal_score.astype(np.float32)
        mechanics_top_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'DYN_TOP_REVERSAL')
        chip_top_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'CHIP_TOP_REVERSAL')
        foundation_top_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FOUNDATION_TOP_REVERSAL')
        behavior_top_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        structure_top_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL')
        total_top_score = pd.Series(0.0, index=df.index)
        total_top_weight = 0.0
        top_sources = {
            'mechanics': mechanics_top_score, 'chip': chip_top_score, 'foundation': foundation_top_score,
            'behavior': behavior_top_score, 'structure': structure_top_score
        }
        for domain, weight in top_weights.items():
            if domain in top_sources and weight > 0:
                total_top_score += top_sources[domain] * weight
                total_top_weight += weight
        top_reversal_score = total_top_score / total_top_weight if total_top_weight > 0 else default_score
        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE_S'] = top_reversal_score.astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_industry_synergy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.1 · 返回值修复版】行业-个股协同元融合引擎
        """
        states = {}
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        score_markup = atomic.get('SCORE_INDUSTRY_MARKUP', default_score)
        score_preheat = atomic.get('SCORE_INDUSTRY_PREHEAT', default_score)
        industry_bullish_score = np.maximum(score_markup, score_preheat)
        score_stagnation = atomic.get('SCORE_INDUSTRY_STAGNATION', default_score)
        score_downtrend = atomic.get('SCORE_INDUSTRY_DOWNTREND', default_score)
        industry_bearish_score = np.maximum(score_stagnation, score_downtrend)
        stock_ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE_S', default_score)
        stock_breakout_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'VOL_BREAKOUT')
        stock_bullish_score = np.maximum(stock_ignition_score, stock_breakout_score)
        stock_breakdown_score = atomic.get('COGNITIVE_SCORE_BREAKDOWN_RESONANCE_S', default_score)
        stock_distribution_score = atomic.get('COGNITIVE_SCORE_RISK_TOP_DISTRIBUTION', default_score)
        stock_bearish_score = np.maximum(stock_breakdown_score, stock_distribution_score)
        synergy_offense_score = pd.Series(industry_bullish_score, index=df.index) * pd.Series(stock_bullish_score, index=df.index)
        states['COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE_S'] = synergy_offense_score.astype(np.float32)
        synergy_risk_score = pd.Series(industry_bearish_score, index=df.index) * pd.Series(stock_bearish_score, index=df.index)
        states['COGNITIVE_SCORE_INDUSTRY_SYNERGY_RISK_S'] = synergy_risk_score.astype(np.float32)
        
        # 更新 atomic_states 并返回 df 以维持调用链
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

    # 新增一个专门的波动率认知信号合成方法
    def _synthesize_volatility_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 预判能力版】波动率认知信号合成模块
        - 核心职责: 整合基础波动率信息，生成更高级的“压缩分”和“突破潜力分”。
        - 算法升级: “突破潜力”融合了BBW的斜率和加速度，具备更强的前瞻性。
        """
        states = {}
        norm_window = 120 # 考虑从参数获取

        # 1. 波动压缩分 (Fused Compression Score)
        atr_ratio = (df['ATR_14_D'] / df['close_D']).fillna(0.0)
        vol_compression_atr = 1 - normalize_score(atr_ratio, df.index, norm_window)
        bbw = df.get('BBW_21_2.0_D', pd.Series(0.5, index=df.index))
        vol_compression_bbw = 1 - normalize_score(bbw, df.index, norm_window)
        fused_compression_score = (vol_compression_atr * 0.4 + vol_compression_bbw * 0.6)
        
        # 2. 波动突破潜力 (Volatility Breakout Potential) - 预判能力升级版
        bbw_slope_score = normalize_score(df.get('SLOPE_5_BBW_21_2.0_D'), df.index, norm_window, ascending=True)
        bbw_accel_score = normalize_score(df.get('ACCEL_5_BBW_21_2.0_D'), df.index, norm_window, ascending=True)
        
        # 融合斜率和加速度，赋予前瞻性
        expansion_score = (bbw_slope_score * 0.6 + bbw_accel_score * 0.4)
        
        breakout_potential = (fused_compression_score * expansion_score)**0.5

        states['COGNITIVE_SCORE_VOL_COMPRESSION_FUSED'] = fused_compression_score.astype(np.float32)
        states['SCORE_VOL_BREAKOUT_POTENTIAL_S'] = breakout_potential.astype(np.float32)
        
        return states













