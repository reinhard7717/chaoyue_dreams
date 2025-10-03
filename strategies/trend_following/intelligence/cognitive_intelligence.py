# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_unified_score
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
        【V2.9 · 职责净化版】顶层认知总分合成模块
        """
        # print("        -> [顶层认知总分合成模块 V2.9 · 职责净化版] 启动...")
        # --- 步骤 0: 预处理 ---
        micro_behavior_states = self.micro_behavior_engine.run_micro_behavior_synthesis(df)
        self.strategy.atomic_states.update(micro_behavior_states)
        tactic_states = self.tactic_engine.run_tactic_synthesis(df, pullback_enhancements)
        self.strategy.atomic_states.update(tactic_states)
        self.strategy.playbook_states.update({k: v for k, v in tactic_states.items() if k.startswith('PLAYBOOK_')})
        # --- 步骤 1: 执行本模块的核心认知融合任务 ---
        df = self.synthesize_trend_quality_score(df)
        df = self.synthesize_pullback_states(df)
        df = self.synthesize_structural_fusion_scores(df)
        df = self.synthesize_ultimate_confirmation_scores(df)
        df = self.synthesize_ignition_resonance_score(df)
        df = self.synthesize_reversal_resonance_scores(df)
        df = self.synthesize_industry_synergy_signals(df)
        df = self.synthesize_mean_reversion_signals(df)
        df = self.synthesize_state_process_synergy(df)
        self.synthesize_trend_acceleration_cascade(df)
        # 在所有依赖项计算完毕后，调用“天使长”诊断引擎
        archangel_states = self._diagnose_archangel_top_reversal(df)
        self.strategy.atomic_states.update(archangel_states)
        # --- 步骤 2: 汇总所有“机会”与“风险”类认知分数 ---
        bullish_scores = [
            self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_REVERSAL_RELIABILITY').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_STATE_PROCESS_SYNERGY').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE').values,
        ]
        cognitive_bullish_score = np.maximum.reduce(bullish_scores)
        self.strategy.atomic_states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)
        fused_risk_states = self.synthesize_fused_risk_scores(df)
        self.strategy.atomic_states.update(fused_risk_states)
        self.synthesize_chimera_conflict_score(df)
        # print("        -> [顶层认知总分合成模块 V2.9] 认知升级完成。")
        return df

    def synthesize_state_process_synergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.3 · 创世纪版】状态-过程协同融合引擎
        - 核心升级: 将四个全新的“创世纪”过程信号全部纳入“过程共识分”的计算，
                      使顶层认知能够理解市场的真实博弈。
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

        # 引入所有“创世纪”过程信号
        process_bullish_signals = [
            'PROCESS_META_PV_REL_BULLISH_TURN',
            'PROCESS_META_PF_REL_BULLISH_TURN',
            'PROCESS_STRATEGY_CHIP_VS_BEHAVIOR_SYNC',
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_STEALTH_ACCUMULATION',
            'PROCESS_META_WINNER_CONVICTION',
            'PROCESS_META_LOSER_CAPITULATION'
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
        【V2.6 · 架构对齐版】趋势质量融合评分模块
        - 核心重构: 修复了筹码健康度计算的逻辑断层，使其直接消费筹码层的终极共振信号。
        """
        # 使用新的 get_unified_score 函数
        behavior_health_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        fund_flow_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        structural_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE')
        regime_health_score_hurst = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME')
        regime_health_score_fft = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT')
        regime_health_score = (regime_health_score_hurst * regime_health_score_fft)**0.5 # 使用几何平均
        
        # 趋势质量中的“筹码健康度”现在直接消费筹码层终极信号，不再重复计算
        chip_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')

        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        
        # 最终融合从加法改为乘法
        domain_scores = [
            behavior_health_score, chip_health_score, fund_flow_health_score,
            structural_health_score, mechanics_health_score, regime_health_score
        ]
        domain_weights_config = [
            weights.get('behavior', 0.20), weights.get('chip', 0.30), weights.get('fund_flow', 0.15),
            weights.get('structural', 0.15), weights.get('mechanics', 0.10), weights.get('regime', 0.10)
        ]
        
        valid_scores = []
        valid_weights = []
        for score, weight in zip(domain_scores, domain_weights_config):
            if weight > 0:
                valid_scores.append(score.values)
                valid_weights.append(weight)

        if valid_scores:
            weights_array = np.array(valid_weights)
            weights_array /= weights_array.sum() # 归一化权重
            stacked_scores = np.stack(valid_scores, axis=0)
            trend_quality_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            trend_quality_score = pd.Series(trend_quality_values, index=df.index, dtype=np.float32)
        else:
            trend_quality_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = trend_quality_score.astype(np.float32)
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.6 · 信号净化版】认知层回踩状态合成模块
        - 核心重构: 使用 get_unified_score 消费唯一的终极信号，并净化输出信号名。
        """
        states = {}
        is_pullback_day = (df['pct_change_D'] < 0).astype(float)
        constructive_context_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        
        gentle_drop_score = (1 - (df['pct_change_D'].abs() / 0.05)).clip(0, 1).fillna(0.0)
        shrinking_volume_score = self._get_atomic_score(df, 'SCORE_VOL_WEAKENING_DROP', 0.0) 
        
        cycle_trough_score = (1 - self._get_atomic_score(df, 'DOMINANT_CYCLE_PHASE', 0.0).fillna(0.0)) / 2.0 
        
        # 使用新的 get_unified_score 函数，并假设使用CHIP领域信号作为判断依据
        winner_holding_tight_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_TOP_REVERSAL')
        chip_stable_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BEARISH_RESONANCE')
        
        healthy_pullback_score = (
            is_pullback_day * constructive_context_score *
            gentle_drop_score * shrinking_volume_score *
            winner_holding_tight_score * chip_stable_score *
            (1 + cycle_trough_score * 0.5)
        )
        # 移除信号名中的_S后缀
        states['COGNITIVE_SCORE_PULLBACK_HEALTHY'] = healthy_pullback_score.astype(np.float32)
        
        significant_drop_score = (df['pct_change_D'].abs() / 0.07).clip(0, 1).fillna(0.0)
        # 使用新的 get_unified_score 函数
        panic_selling_score = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BEARISH_RESONANCE')
        suppressive_pullback_score = (
            is_pullback_day * constructive_context_score *
            significant_drop_score * panic_selling_score * winner_holding_tight_score
        )
        # 移除信号名中的_S后缀
        states['COGNITIVE_SCORE_PULLBACK_SUPPRESSIVE'] = suppressive_pullback_score.astype(np.float32)
        
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_fused_risk_scores(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.4 · 赫利俄斯版】风险元融合模块
        - 核心革命: 彻底废除“加权求和”的风险融合逻辑，升级为“加权几何平均”。
                      确保最终的 COGNITIVE_FUSED_RISK_SCORE 被严格归一化到 [0, 1] 区间，
                      解决了风险分“通货膨胀”和不可比的根本性设计缺陷。
        """
        states = {}
        p_fused_risk = get_params_block(self.strategy, 'fused_risk_scoring')
        if not get_param_value(p_fused_risk.get('enabled'), True):
            states['COGNITIVE_FUSED_RISK_SCORE'] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        risk_categories = p_fused_risk.get('risk_categories', {})
        p_fusion_params = p_fused_risk.get('intra_dimension_fusion_params', {})
        secondary_risk_discount = p_fusion_params.get('secondary_risk_discount', 0.3)
        all_required_signals = set()
        for category_name, signals in risk_categories.items():
            if category_name != "说明":
                all_required_signals.update(s for s in signals if s != "说明")
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
        p_dynamic_weighting = p_fused_risk.get('dynamic_weighting_params', {})
        base_weights = p_dynamic_weighting.get('base_weights', {})
        context_adjustments = p_dynamic_weighting.get('context_adjustments', {})
        p_resonance = p_fused_risk.get('resonance_penalty_params', {})
        is_early_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_EARLY', pd.Series(False, index=df.index))
        is_late_stage = self.strategy.atomic_states.get('CONTEXT_TREND_STAGE_LATE', pd.Series(False, index=df.index))
        # 核心修改：从加权求和改为加权几何平均
        valid_scores = []
        valid_weights = []
        for category_name, base_weight in base_weights.items():
            if category_name in fused_dimension_scores and base_weight > 0:
                current_weight = pd.Series(base_weight, index=df.index)
                if get_param_value(p_dynamic_weighting.get('enabled'), True):
                    early_adjustments = context_adjustments.get("CONTEXT_TREND_STAGE_EARLY", {})
                    if category_name in early_adjustments:
                        current_weight = current_weight.where(~is_early_stage, current_weight * early_adjustments[category_name])
                    late_adjustments = context_adjustments.get("CONTEXT_TREND_STAGE_LATE", {})
                    if category_name in late_adjustments:
                        current_weight = current_weight.where(~is_late_stage, current_weight * late_adjustments[category_name])
                valid_scores.append(fused_dimension_scores[category_name].values)
                valid_weights.append(current_weight.values)
        if valid_scores:
            stacked_scores = np.stack(valid_scores, axis=0)
            stacked_weights = np.stack(valid_weights, axis=0)
            # 归一化权重
            total_weights = np.sum(stacked_weights, axis=0)
            normalized_weights = stacked_weights / total_weights
            # 计算加权几何平均
            # 为避免log(0)错误，给stacked_scores增加一个极小值
            total_fused_risk_values = np.exp(np.sum(normalized_weights * np.log(stacked_scores + 1e-9), axis=0))
            total_fused_risk_score = pd.Series(total_fused_risk_values, index=df.index, dtype=np.float32)
        else:
            total_fused_risk_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        if get_param_value(p_resonance.get('enabled'), True):
            core_dims = get_param_value(p_resonance.get('core_risk_dimensions'), [])
            min_dims = get_param_value(p_resonance.get('min_dimensions_for_resonance'), 2)
            threshold = get_param_value(p_resonance.get('risk_score_threshold'), 0.6) # 阈值也应适配[0,1]
            penalty_multiplier = get_param_value(p_resonance.get('penalty_multiplier'), 1.2)
            high_risk_dimension_count = pd.Series(0, index=df.index)
            for dim in core_dims:
                if dim in fused_dimension_scores:
                    high_risk_dimension_count += (fused_dimension_scores[dim] > threshold).astype(int)
            is_resonance_triggered = (high_risk_dimension_count >= min_dims)
            total_fused_risk_score = total_fused_risk_score.where(~is_resonance_triggered, (total_fused_risk_score * penalty_multiplier).clip(0, 1))
            states['FUSED_RISK_RESONANCE_PENALTY_ACTIVE'] = is_resonance_triggered
        states['COGNITIVE_FUSED_RISK_SCORE'] = total_fused_risk_score.astype(np.float32)
        return states

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V2.5 · 信号净化版】结构化元信号融合模块
        - 核心重构: 使用 get_unified_score 消费唯一的终极信号，并净化输出信号名。
        """
        # print("        -> [结构化元信号融合模块 V2.5 信号净化版] 启动...")
        states = {}
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # 全面使用 get_unified_score 消费净化后的信号
        foundation_bullish = get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BULLISH_RESONANCE')
        foundation_bearish = get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BEARISH_RESONANCE')
        foundation_bottom = get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL')
        foundation_top = get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_TOP_REVERSAL')
        structure_bullish = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        structure_bearish = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BEARISH_RESONANCE')
        structure_bottom = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        structure_top = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL')
        behavior_bullish = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BULLISH_RESONANCE')
        behavior_bearish = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BEARISH_RESONANCE')
        behavior_bottom = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL')
        behavior_top = get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        
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
        
        # 移除所有输出信号名中的_S后缀
        states['COGNITIVE_FUSION_BULLISH_RESONANCE'] = (foundation_bullish * structure_bullish * behavior_bullish).astype(np.float32)
        states['COGNITIVE_FUSION_BEARISH_RESONANCE'] = (foundation_bearish * structure_bearish * behavior_bearish).astype(np.float32)
        states['COGNITIVE_FUSION_BOTTOM_REVERSAL'] = (foundation_bottom * structure_bottom * behavior_bottom).astype(np.float32)
        states['COGNITIVE_FUSION_TOP_REVERSAL'] = (foundation_top * structure_top * behavior_top).astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_ultimate_confirmation_scores(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        【V1.3 · 信号净化版】终极确认融合模块
        - 核心重构: 使用 get_unified_score 消费唯一的终极信号，并净化输出信号名。
        """
        # print("        -> [终极确认融合模块 V1.3 信号净化版] 启动...")
        states = {}
        atomic = self.strategy.atomic_states
        # 更新依赖信号名，移除_S后缀
        required_fusion_signals = [
            'COGNITIVE_FUSION_BULLISH_RESONANCE', 'COGNITIVE_FUSION_BEARISH_RESONANCE',
            'COGNITIVE_FUSION_BOTTOM_REVERSAL', 'COGNITIVE_FUSION_TOP_REVERSAL'
        ]
        missing_fusion_signals = [s for s in required_fusion_signals if s not in atomic]
        if missing_fusion_signals:
            return df
        default_series = pd.Series(0.0, index=df.index, dtype=np.float32)
        # 更新获取的信号名，移除_S后缀
        fusion_bullish = atomic.get('COGNITIVE_FUSION_BULLISH_RESONANCE', default_series)
        fusion_bearish = atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE', default_series)
        fusion_bottom = atomic.get('COGNITIVE_FUSION_BOTTOM_REVERSAL', default_series)
        fusion_top = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL', default_series)
        
        # 使用新的 get_unified_score 函数
        pattern_bullish = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_BULLISH_RESONANCE')
        pattern_bearish = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_BEARISH_RESONANCE')
        pattern_bottom = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_BOTTOM_REVERSAL')
        pattern_top = get_unified_score(self.strategy.atomic_states, df.index, 'PATTERN_TOP_REVERSAL')
        
        # 移除所有输出信号名中的_S后缀
        states['COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION'] = (fusion_bullish * pattern_bullish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION'] = (fusion_bearish * pattern_bearish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION'] = (fusion_bottom * pattern_bottom).astype(np.float32)
        states['COGNITIVE_ULTIMATE_TOP_CONFIRMATION'] = (fusion_top * pattern_top).astype(np.float32)
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
        【V2.8 · 信号净化版】多域反转共振分数合成模块
        - 核心重构: 使用 get_unified_score 消费唯一的终极信号，并净化输出信号名。
        """
        states = {}
        default_score = pd.Series(0.5, index=df.index, dtype=np.float32) # 默认值改为0.5
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = get_param_value(p.get('bottom_resonance_weights'), {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        top_weights = get_param_value(p.get('top_resonance_weights'), {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        
        # 全面使用 get_unified_score 消费净化后的信号
        bottom_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BOTTOM_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        }
        
        # 底部反转融合从加法改为乘法
        bottom_scores = []
        bottom_weight_values = []
        for domain, weight in bottom_weights.items():
            if domain in bottom_sources and weight > 0:
                bottom_scores.append(bottom_sources[domain].values)
                bottom_weight_values.append(weight)
        
        if bottom_scores:
            weights_array = np.array(bottom_weight_values)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(bottom_scores, axis=0)
            bottom_reversal_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            bottom_reversal_score = pd.Series(bottom_reversal_values, index=df.index, dtype=np.float32)
        else:
            bottom_reversal_score = default_score

        # 移除信号名中的_S后缀
        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE'] = bottom_reversal_score.astype(np.float32)
        
        # 全面使用 get_unified_score 消费净化后的信号
        top_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_TOP_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_TOP_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_TOP_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL')
        }

        # 顶部反转融合从加法改为乘法
        top_scores = []
        top_weight_values = []
        for domain, weight in top_weights.items():
            if domain in top_sources and weight > 0:
                top_scores.append(top_sources[domain].values)
                top_weight_values.append(weight)

        if top_scores:
            weights_array = np.array(top_weight_values)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(top_scores, axis=0)
            top_reversal_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            top_reversal_score = pd.Series(top_reversal_values, index=df.index, dtype=np.float32)
        else:
            top_reversal_score = default_score

        # 移除信号名中的_S后缀
        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE'] = top_reversal_score.astype(np.float32)
        self.strategy.atomic_states.update(states)
        return df

    def synthesize_industry_synergy_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.3 · 依赖净化版】行业-个股协同元融合引擎
        - 核心重构: 更新消费的风险信号为最新的、正确的融合风险信号。
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
        
        stock_ignition_score = atomic.get('COGNITIVE_SCORE_IGNITION_RESONANCE', default_score)
        stock_breakout_score = self._get_atomic_score(df, 'SCORE_VOL_BREAKOUT_POTENTIAL', 0.0)
        
        stock_bullish_score = np.maximum(stock_ignition_score, stock_breakout_score)
        
        # 将消费的信号更新为最新的、正确的融合风险信号
        stock_breakdown_score = atomic.get('COGNITIVE_FUSION_BEARISH_RESONANCE', default_score)
        stock_distribution_score = atomic.get('COGNITIVE_FUSION_TOP_REVERSAL', default_score)
        stock_bearish_score = np.maximum(stock_breakdown_score, stock_distribution_score)
        
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
        【V3.0 · 赫拉织布机版】趋势加速级联 (涡轮增压) 诊断引擎
        - 核心革命: 借鉴ProcessIntelligence思想，引入二阶导数(加速度)和关系动态(协同性)分析。
        - 新核心逻辑:
          1. 对每个领域，同时计算其斜率(速度)和加速度。
          2. 将速度分和加速度分加权融合成该领域的“动态健康分”。
          3. 计算所有领域“动态健康分”的均值（代表整体加速水平）和标准差（代表分歧度）。
          4. 最终级联分 = 平均动态健康分 * (1 - 标准差)，同时奖励“加速”和“协同”。
        """
        states = {}
        norm_window = 55
        slope_period = 3 
        # 从配置中获取新的融合权重
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        fusion_weights = get_param_value(p_cognitive.get('cascade_fusion_weights'), {'slope': 0.6, 'accel': 0.4})
        w_slope = fusion_weights.get('slope', 0.6)
        w_accel = fusion_weights.get('accel', 0.4)

        # 辅助函数，用于计算单个序列的动态健康分
        def get_dynamic_health(series: pd.Series) -> pd.Series:
            slope = series.diff(slope_period).fillna(0)
            accel = slope.diff(slope_period).fillna(0)
            
            # 归一化时，只考虑正向变化
            slope_score = normalize_score(slope.clip(lower=0), df.index, norm_window)
            accel_score = normalize_score(accel.clip(lower=0), df.index, norm_window)
            
            # 融合速度与加速度
            dynamic_health_score = (slope_score * w_slope + accel_score * w_accel)
            return dynamic_health_score

        # --- 1. 时序级联诊断 (Temporal Cascade) ---
        health_cache = self.strategy.atomic_states.get('__BEHAVIOR_overall_health', {})
        s_bull = health_cache.get('s_bull', {})
        d_intensity = health_cache.get('d_intensity', {})
        relational_power = self.strategy.atomic_states.get('SCORE_ATOMIC_RELATIONAL_DYNAMICS', pd.Series(0.5, index=df.index))
        short_term_health = np.maximum(s_bull.get(5, pd.Series(0.5, index=df.index)), relational_power) * d_intensity.get(5, pd.Series(0.5, index=df.index))
        medium_term_health = np.maximum(s_bull.get(21, pd.Series(0.5, index=df.index)), relational_power) * d_intensity.get(21, pd.Series(0.5, index=df.index))
        
        # 使用新的辅助函数计算动态健康分
        short_term_dynamic_health = get_dynamic_health(short_term_health)
        medium_term_dynamic_health = get_dynamic_health(medium_term_health)
        
        temporal_cascade_score = (short_term_dynamic_health * medium_term_dynamic_health)**0.5
        states['COGNITIVE_INTERNAL_TEMPORAL_CASCADE'] = temporal_cascade_score.astype(np.float32)

        # --- 2. 领域级联诊断 (Domain Cascade) ---
        resonance_signals = {
            'behavior': self._get_atomic_score(df, 'SCORE_BEHAVIOR_BULLISH_RESONANCE'),
            'chip': self._get_atomic_score(df, 'SCORE_CHIP_BULLISH_RESONANCE'),
            'ff': self._get_atomic_score(df, 'SCORE_FF_BULLISH_RESONANCE'),
            'structure': self._get_atomic_score(df, 'SCORE_STRUCTURE_BULLISH_RESONANCE'),
            'dyn': self._get_atomic_score(df, 'SCORE_DYN_BULLISH_RESONANCE'),
        }
        
        # 计算每个领域的动态健康分
        domain_dynamic_health_scores = []
        for name, signal in resonance_signals.items():
            dynamic_health = get_dynamic_health(signal)
            states[f'COGNITIVE_INTERNAL_DYN_HEALTH_{name.upper()}'] = dynamic_health.astype(np.float32)
            domain_dynamic_health_scores.append(dynamic_health)
        
        # 计算“关系协同分”
        if domain_dynamic_health_scores:
            # 将所有领域的动态健康分堆叠起来
            stacked_health = np.stack([s.values for s in domain_dynamic_health_scores], axis=0)
            
            # 计算平均动态健康分（代表整体加速水平）
            average_dynamic_health = pd.Series(np.mean(stacked_health, axis=0), index=df.index)
            
            # 计算标准差（代表分歧度），并归一化
            std_dev_health = pd.Series(np.std(stacked_health, axis=0), index=df.index)
            normalized_std_dev = normalize_score(std_dev_health, df.index, norm_window, ascending=True)
            
            # 协同分 = 1 - 分歧度
            relational_cohesion_score = 1.0 - normalized_std_dev
            
            # 最终领域级联分 = 整体加速水平 * 协同性
            domain_cascade_score = (average_dynamic_health * relational_cohesion_score).clip(0, 1)
        else:
            domain_cascade_score = pd.Series(0.0, index=df.index)
            
        states['COGNITIVE_INTERNAL_DOMAIN_CASCADE'] = domain_cascade_score.astype(np.float32)

        # --- 3. 最终融合 ---
        final_cascade_score = (temporal_cascade_score * domain_cascade_score).astype(np.float32)
        states['COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE'] = final_cascade_score
        
        self.strategy.atomic_states.update(states)

    def synthesize_chimera_conflict_score(self, df: pd.DataFrame) -> None:
        """
        【V1.2 · 奇美拉之力版】奇美拉冲突诊断引擎
        - 核心修复: 移除了对 COGNITIVE_BULLISH_SCORE 错误的除以1000的操作。
                      该分数已经是[0,1]区间的归一化值，无需再次缩放。
        - 收益: 确保了“奇美拉冲突”能够正确反映多空力量的真实冲突强度。
        """
        states = {}
        # 移除了错误的 / 1000.0 操作，因为 COGNITIVE_BULLISH_SCORE 已经是归一化分数
        bullish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_BULLISH_SCORE', 0.0).clip(0, 1)
        bearish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_FUSED_RISK_SCORE', 0.0)
        conflict_score = np.minimum(bullish_score_normalized, bearish_score_normalized).clip(0, 1)
        states['COGNITIVE_SCORE_CHIMERA_CONFLICT'] = conflict_score.astype(np.float32)
        self.strategy.atomic_states.update(states)

    def _diagnose_archangel_top_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 地狱三头犬版】“天使长”顶部反转诊断引擎
        - 核心职责: 融合三大致命顶部风险信号，识别最高优先级的离场信号。
        - 融合算法: 采用“主次风险融合算法”，最终风险 = 主要风险 + (次要风险 * 折扣因子)。
        - 归属: 此方法作为顶层认知融合的一部分，正式归属于认知情报模块。
        """
        states = {}
        # 从配置中获取次要风险的折扣因子
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        p_archangel = p_judge.get('archangel_fusion_params', {})
        secondary_risk_discount = get_param_value(p_archangel.get('secondary_risk_discount'), 0.4)
        # 从原子状态库中调集“天使军团”
        upthrust_risk = self.strategy.atomic_states.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index))
        heaven_earth_risk = self.strategy.atomic_states.get('SCORE_BOARD_HEAVEN_EARTH', pd.Series(0.0, index=df.index))
        post_peak_risk = self.strategy.atomic_states.get('COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN', pd.Series(0.0, index=df.index))
        # 升级为“地狱三头犬”融合逻辑
        # 1. 将三个风险信号堆叠成一个NumPy数组
        risk_matrix = np.stack([
            upthrust_risk.values,
            heaven_earth_risk.values,
            post_peak_risk.values
        ], axis=0)
        # 2. 沿信号轴（axis=0）对每日的风险进行排序
        sorted_risks = np.sort(risk_matrix, axis=0)
        # 3. 提取主要风险（最高分）和次要风险（第二高分）
        primary_risk = sorted_risks[-1]
        secondary_risk = sorted_risks[-2]
        # 4. 应用主次风险融合公式
        archangel_score_values = primary_risk + (secondary_risk * secondary_risk_discount)
        # 确保最终分数不会超过1.0
        archangel_score = np.clip(archangel_score_values, 0, 1)
        states['SCORE_ARCHANGEL_TOP_REVERSAL'] = pd.Series(archangel_score, index=df.index, dtype=np.float32)
        return states






















