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
        【V4.1 · 健壮性升级版】顶层认知总分合成模块
        - 核心修复: 彻底移除在融合风险分计算后，使用 np.maximum 将“真实撤退风险”并入“融合风险总分”的冗余逻辑。
                      此举解决了“风险被重复计算”和“奇美拉冲突分被错误放大”两大致命BUG，确保每个风险信号在最终裁决时只被审判一次。
        - 健壮性升级(V4.1): 重构了看涨总分的计算逻辑。不再静态地假设所有信号都存在，而是动态地构建一个有效的信号列表，
                           只将实际存在且非空的信号纳入最终的 `maximum.reduce` 计算，增强了系统的容错能力。
        """
        micro_behavior_states = self.micro_behavior_engine.run_micro_behavior_synthesis(df)
        self.strategy.atomic_states.update(micro_behavior_states)
        tactic_states = self.tactic_engine.run_tactic_synthesis(df, pullback_enhancements)
        self.strategy.atomic_states.update(tactic_states)
        self.strategy.playbook_states.update({k: v for k, v in tactic_states.items() if k.startswith('PLAYBOOK_')})
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
        self.strategy.atomic_states['strategy_instance_ref'] = self.strategy
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        del self.strategy.atomic_states['strategy_instance_ref']
        self.strategy.atomic_states['CONTEXT_BOTTOM_SCORE'] = bottom_context_score
        self.strategy.atomic_states['CONTEXT_TOP_SCORE'] = top_context_score
        archangel_states = self._diagnose_archangel_top_reversal(df)
        self.strategy.atomic_states.update(archangel_states)
        # 动态构建有效的看涨信号列表
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
        ]
        valid_bullish_scores = []
        for signal_name in bullish_signal_names:
            # 安全地获取信号，如果信号不存在或为空，则不会被加入列表
            signal_series = self._get_atomic_score(df, signal_name)
            if signal_series is not None and not signal_series.empty:
                valid_bullish_scores.append(signal_series.values)
        if valid_bullish_scores:
            # 只有在找到有效信号时才进行计算
            cognitive_bullish_score = np.maximum.reduce(valid_bullish_scores)
        else:
            # 如果没有任何有效的看涨信号，则总分为0
            cognitive_bullish_score = np.zeros(len(df.index), dtype=np.float32)
        
        self.strategy.atomic_states['COGNITIVE_BULLISH_SCORE'] = pd.Series(cognitive_bullish_score, index=df.index, dtype=np.float32)
        fused_risk_states = self.synthesize_fused_risk_scores(df)
        self.strategy.atomic_states.update(fused_risk_states)
        self.synthesize_chimera_conflict_score(df)
        return df

    def synthesize_state_process_synergy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.3 · 创世纪版】状态-过程协同融合引擎
        - 核心升级: 将四个全新的“创世纪”过程信号全部纳入“过程共识分”的计算，
                      使顶层认知能够理解市场的真实博弈。
        - 优化说明: 使用Numpy向量化计算几何平均数，提高计算效率和数值稳定性。
        """
        states = {} # 新增(规范): 初始化用于存储结果的字典
        
        # --- 状态共识分计算 ---
        # 状态层面的多头信号列表
        state_bullish_signals = [
            'SCORE_CHIP_BULLISH_RESONANCE', 'SCORE_BEHAVIOR_BULLISH_RESONANCE',
            'SCORE_FF_BULLISH_RESONANCE', 'SCORE_STRUCTURE_BULLISH_RESONANCE',
            'SCORE_DYN_BULLISH_RESONANCE', 'SCORE_FOUNDATION_BULLISH_RESONANCE'
        ]
        # 使用列表推导式和.values高效获取所有信号的Numpy数组
        state_scores = [self._get_atomic_score(df, sig, 0.5).values for sig in state_bullish_signals]
        
        # 使用np.stack将列表转换为2D数组，然后计算几何平均数，避免pandas开销
        # 几何平均数可以惩罚那些存在极低分数的项，要求各维度协同看多
        state_consensus_score = pd.Series(
            np.prod(np.stack(state_scores, axis=0), axis=0) ** (1.0 / len(state_scores)),
            index=df.index, dtype=np.float32
        )
        states['COGNITIVE_INTERNAL_STATE_CONSENSUS'] = state_consensus_score

        # --- 过程共识分计算 ---
        # 引入所有“创世纪”过程信号，这些信号描述了市场参与者行为的动态演变
        process_bullish_signals = [
            'PROCESS_META_PV_REL_BULLISH_TURN',
            'PROCESS_META_PF_REL_BULLISH_TURN',
            'PROCESS_STRATEGY_CHIP_VS_BEHAVIOR_SYNC',
            'PROCESS_META_POWER_TRANSFER',
            'PROCESS_META_STEALTH_ACCUMULATION',
            'PROCESS_META_WINNER_CONVICTION',
            'PROCESS_META_LOSER_CAPITULATION'
        ]
        # 将信号值从[-1, 1]区间归一化到[0, 1]区间，并直接获取Numpy数组
        # (score * 0.5 + 0.5) 是将[-1, 1]映射到[0, 1]的标准方法
        process_scores = [(self._get_atomic_score(df, sig, 0.0).clip(-1, 1) * 0.5 + 0.5).values for sig in process_bullish_signals]
        
        # 同样使用Numpy高效计算过程信号的几何平均数
        process_consensus_score = pd.Series(
            np.prod(np.stack(process_scores, axis=0), axis=0) ** (1.0 / len(process_scores)),
            index=df.index, dtype=np.float32
        )
        states['COGNITIVE_INTERNAL_PROCESS_CONSENSUS'] = process_consensus_score

        # --- 最终协同分数合成 ---
        # 状态共识与过程共识相乘，代表静态优势与动态演变的协同效应
        synergy_score = (state_consensus_score * process_consensus_score).astype(np.float32)
        states['COGNITIVE_SCORE_STATE_PROCESS_SYNERGY'] = synergy_score
        
        self.strategy.atomic_states.update(states) # 新增(规范): 统一更新原子状态库
        return df

    def synthesize_trend_quality_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.0 · 赫淮斯托斯熔炉版】趋势质量融合评分模块
        - 核心升级: 引入关系元分析，对融合后的“趋势质量快照分”进行动态锻造。
        """
        # --- 1. 获取各维度健康度分数 ---
        behavior_health_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        fund_flow_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        structural_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE')
        regime_health_score_hurst = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME')
        regime_health_score_fft = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT')
        regime_health_score = (regime_health_score_hurst * regime_health_score_fft)**0.5
        chip_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        # --- 2. 读取权重配置 ---
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        # --- 3. 静态融合，计算“趋势质量快照分” ---
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
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(valid_scores, axis=0)
            trend_quality_snapshot_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            trend_quality_snapshot_score = pd.Series(trend_quality_snapshot_values, index=df.index, dtype=np.float32)
        else:
            trend_quality_snapshot_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        # 动态锻造
        # --- 4. 对“趋势质量快照分”进行关系元分析，得到最终动态分数 ---
        final_trend_quality_score = self._perform_cognitive_relational_meta_analysis(df, trend_quality_snapshot_score)
        self.strategy.atomic_states['COGNITIVE_SCORE_TREND_QUALITY'] = final_trend_quality_score.astype(np.float32)
        
        return df

    def synthesize_pullback_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V3.0 · 赫淮斯托斯熔炉版】认知层回踩状态合成模块
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
        【V5.1 · 赫尔墨斯信使协议版】风险元融合模块
        - 核心升级: 签署“赫尔墨斯信使协议”，将内部计算的“风险快照分”发布到原子状态库，
                      命名为 COGNITIVE_INTERNAL_RISK_SNAPSHOT，确保系统的完全透明和可验证性。
        - 收益: 解决了风险探针因无法获取中间变量而导致重算结果恒为0的致命BUG。
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
        # 部署“赫尔墨斯信使协议”，发布内部快照分
        states['COGNITIVE_INTERNAL_RISK_SNAPSHOT'] = risk_snapshot_score

        final_dynamic_risk_score = self._perform_cognitive_relational_meta_analysis(df, risk_snapshot_score)
        states['COGNITIVE_FUSED_RISK_SCORE'] = final_dynamic_risk_score
        return states

    def synthesize_chimera_conflict_score(self, df: pd.DataFrame) -> None:
        """
        【V3.0 · 战神阿瑞斯版】奇美拉冲突诊断引擎
        - 核心升级: 引入“资金流分歧度”作为独立的冲突维度。
        - 新核心逻辑: 最终冲突分 = (信号冲突分 * 权重) + (资金流分歧冲突分 * 权重)。
                      当市场的看涨/看跌信号本身就充满矛盾，同时资金流的多个信息源也互相矛盾时，
                      系统将发出最高等级的“奇美拉冲突”警报。
        """
        states = {}
        p_chimera = get_params_block(self.strategy, 'chimera_conflict_params', {})
        weights = get_param_value(p_chimera.get('fusion_weights'), {'signal_conflict': 0.6, 'flow_divergence': 0.4})

        # --- 维度一: 信号冲突分 (Signal-level Conflict) ---
        bullish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_BULLISH_SCORE', 0.0).clip(0, 1)
        bearish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_FUSED_RISK_SCORE', 0.0).clip(0, 1)
        signal_conflict_score = np.minimum(bullish_score_normalized, bearish_score_normalized)
        
        # 引入资金流分歧度作为新的冲突维度
        # --- 维度二: 资金流分歧冲突分 (Flow Divergence Conflict) ---
        # 分歧度的绝对值越大，代表冲突越严重。我们对所有分歧度指标取绝对值，然后归一化，再取最大值。
        norm_window = 55
        div_ts_ths_abs = df.get('divergence_ts_ths_D', pd.Series(0, index=df.index)).abs()
        div_ts_dc_abs = df.get('divergence_ts_dc_D', pd.Series(0, index=df.index)).abs()
        div_ths_dc_abs = df.get('divergence_ths_dc_D', pd.Series(0, index=df.index)).abs()
        
        # 对每个分歧度的绝对值进行归一化
        div_ts_ths_score = normalize_score(div_ts_ths_abs, df.index, norm_window, ascending=True)
        div_ts_dc_score = normalize_score(div_ts_dc_abs, df.index, norm_window, ascending=True)
        div_ths_dc_score = normalize_score(div_ths_dc_abs, df.index, norm_window, ascending=True)
        
        # 取最强的分歧作为该维度的冲突分
        flow_divergence_conflict_score = np.maximum.reduce([
            div_ts_ths_score.values,
            div_ts_dc_score.values,
            div_ths_dc_score.values
        ])
        flow_divergence_conflict_score = pd.Series(flow_divergence_conflict_score, index=df.index)

        # --- 最终融合: 加权融合两大冲突维度 ---
        final_conflict_score = (
            signal_conflict_score * weights.get('signal_conflict', 0.6) +
            flow_divergence_conflict_score * weights.get('flow_divergence', 0.4)
        )
        
        
        states['COGNITIVE_SCORE_CHIMERA_CONFLICT'] = final_conflict_score.clip(0, 1).astype(np.float32)
        self.strategy.atomic_states.update(states)

    def synthesize_structural_fusion_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V4.0 · 普罗米修斯之火协议版】结构化元信号融合模块
        - 核心革命: 引入两阶段认知。第一阶段计算“共识快照分”，第二阶段对“共识”本身进行关系元分析。
        - 新核心逻辑: final_score = MetaAnalysis(GeometricMean(foundation, structure, behavior))
        - 收益: 最终信号不仅反映共识强度，更反映共识形成的速度与加速度，实现真正的认知前瞻。
        """
        # print("        -> [结构化元信号融合模块 V4.0 普罗米修斯之火协议版] 启动...") # 打印版本信息
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
                'bottom': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL'),
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
        # 实现两阶段认知
        for f_type in fusion_types:
            # --- 阶段一：计算“共识快照分” ---
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
            # 这是“共识”在当前时刻的快照分
            consensus_snapshot_score = pd.Series(np.exp(weighted_log_sum), index=df.index, dtype=np.float32)
            # --- 阶段二：对“共识”本身进行关系元分析 ---
            # 调用认知层专属的元分析引擎
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
        【V4.0 · 赫淮斯托斯熔炉版】多域反转共振分数合成模块
        - 核心升级: 引入关系元分析，对融合后的“反转共振快照分”进行动态锻造。
        """
        states = {}
        default_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = p.get('bottom_resonance_weights', {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        top_weights = p.get('top_resonance_weights', {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        self.strategy.atomic_states['strategy_instance_ref'] = self.strategy
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        del self.strategy.atomic_states['strategy_instance_ref']
        # --- 底部反转共振分数 ---
        bottom_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BOTTOM_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_BOTTOM_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_BOTTOM_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BOTTOM_REVERSAL')
        }
        bottom_scores_np = [s.values for d, s in bottom_sources.items() if bottom_weights.get(d, 0) > 0]
        bottom_weights_np = [w for d, w in bottom_weights.items() if d in bottom_sources and w > 0]
        if bottom_scores_np:
            weights_array = np.array(bottom_weights_np)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(bottom_scores_np, axis=0)
            bottom_reversal_values_raw = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            reversal_reliability_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_REVERSAL_RELIABILITY', 0.0)
            reliability_bonus_factor = get_param_value(p_cognitive.get('reversal_reliability_bonus_factor'), 0.5)
            reliability_amplifier = 1.0 + (reversal_reliability_score.values * reliability_bonus_factor)
            capitulation_potential_score = self._get_atomic_score(df, 'SCORE_CHIP_CONTEXT_CAPITULATION_POTENTIAL', 0.0)
            capitulation_bonus_factor = get_param_value(p_cognitive.get('capitulation_potential_bonus_factor'), 0.5)
            capitulation_amplifier = 1.0 + (capitulation_potential_score.values * capitulation_bonus_factor)
            # 计算快照分并进行动态锻造
            bottom_reversal_snapshot_values = bottom_reversal_values_raw * bottom_context_score.values * reliability_amplifier * capitulation_amplifier
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
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TOP_REVERSAL')
        }
        top_scores_np = [s.values for d, s in top_sources.items() if top_weights.get(d, 0) > 0]
        top_weights_np = [w for d, w in top_weights.items() if d in top_sources and w > 0]
        if top_scores_np:
            weights_array = np.array(top_weights_np)
            weights_array /= weights_array.sum()
            stacked_scores = np.stack(top_scores_np, axis=0)
            top_reversal_values_raw = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            # 计算快照分并进行动态锻造
            top_reversal_snapshot_values = top_reversal_values_raw * top_context_score.values
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
        【V3.0 · 赫拉织布机版】趋势加速级联 (涡轮增压) 诊断引擎
        - 核心革命: 借鉴ProcessIntelligence思想，引入二阶导数(加速度)和关系动态(协同性)分析。
        - 优化说明: 逻辑高度定制，通过Numpy向量化操作实现斜率、加速度计算及后续融合，保证了复杂逻辑下的执行效率。
        """
        states = {}
        norm_window = 55
        slope_period = 3 
        p_cognitive = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        fusion_weights = p_cognitive.get('cascade_fusion_weights', {'slope': 0.6, 'accel': 0.4})
        w_slope = fusion_weights.get('slope', 0.6)
        w_accel = fusion_weights.get('accel', 0.4)

        # 定义一个内联辅助函数，用于计算单个序列的“动态健康分”
        # 该分数融合了信号的增长速度（斜率）和增长加速度
        def get_dynamic_health(series: pd.Series) -> pd.Series:
            # 使用pandas内置的diff()高效计算斜率（一阶导数）和加速度（二阶导数）
            slope = series.diff(slope_period).fillna(0)
            accel = slope.diff(slope_period).fillna(0)
            
            # 归一化时，只考虑正向变化（增长和加速），体现“涡轮增压”的题意
            slope_score = normalize_score(slope.clip(lower=0), df.index, norm_window)
            accel_score = normalize_score(accel.clip(lower=0), df.index, norm_window)
            
            # 加权融合速度分和加速度分
            dynamic_health_score = (slope_score * w_slope + accel_score * w_accel)
            return dynamic_health_score

        # --- 1. 时序级联诊断 (Temporal Cascade) ---
        # 评估健康度信号在时间维度上的加速情况
        health_cache = self.strategy.atomic_states.get('__BEHAVIOR_overall_health', {})
        s_bull = health_cache.get('s_bull', {})
        d_intensity = health_cache.get('d_intensity', {})
        relational_power = self._get_atomic_score(df, 'SCORE_ATOMIC_RELATIONAL_DYNAMICS', 0.5)
        # 融合短期和中期健康度信号
        short_term_health = np.maximum(s_bull.get(5, pd.Series(0.5, index=df.index)), relational_power) * d_intensity.get(5, pd.Series(0.5, index=df.index))
        medium_term_health = np.maximum(s_bull.get(21, pd.Series(0.5, index=df.index)), relational_power) * d_intensity.get(21, pd.Series(0.5, index=df.index))
        
        # 计算短期和中期健康度的动态健康分
        short_term_dynamic_health = get_dynamic_health(short_term_health)
        medium_term_dynamic_health = get_dynamic_health(medium_term_health)
        
        # 融合时间维度上的加速信号，要求短中期趋势都在加速
        temporal_cascade_score = (short_term_dynamic_health * medium_term_dynamic_health)**0.5
        states['COGNITIVE_INTERNAL_TEMPORAL_CASCADE'] = temporal_cascade_score.astype(np.float32)

        # --- 2. 领域级联诊断 (Domain Cascade) ---
        # 评估不同分析维度（领域）之间的协同加速情况
        resonance_signals = {
            'behavior': self._get_atomic_score(df, 'SCORE_BEHAVIOR_BULLISH_RESONANCE'),
            'chip': self._get_atomic_score(df, 'SCORE_CHIP_BULLISH_RESONANCE'),
            'ff': self._get_atomic_score(df, 'SCORE_FF_BULLISH_RESONANCE'),
            'structure': self._get_atomic_score(df, 'SCORE_STRUCTURE_BULLISH_RESONANCE'),
            'dyn': self._get_atomic_score(df, 'SCORE_DYN_BULLISH_RESONANCE'),
        }
        
        domain_dynamic_health_scores = [get_dynamic_health(signal) for signal in resonance_signals.values()]
        
        if domain_dynamic_health_scores:
            # 将所有领域的动态健康分堆叠成Numpy数组，进行高效的跨领域分析
            stacked_health = np.stack([s.values for s in domain_dynamic_health_scores], axis=0)
            
            # 计算平均动态健康分，代表整体加速水平
            average_dynamic_health = pd.Series(np.mean(stacked_health, axis=0), index=df.index)
            
            # 计算标准差，代表各领域加速水平的分歧度
            std_dev_health = pd.Series(np.std(stacked_health, axis=0), index=df.index)
            normalized_std_dev = normalize_score(std_dev_health, df.index, norm_window, ascending=True)
            
            # 协同分 = 1 - 分歧度。分歧越小，协同性越高。
            relational_cohesion_score = 1.0 - normalized_std_dev
            
            # 最终领域级联分 = 整体加速水平 * 协同性。要求大家一起加速，而不是个别领域单兵突进。
            domain_cascade_score = (average_dynamic_health * relational_cohesion_score).clip(0, 1)
        else:
            domain_cascade_score = pd.Series(0.0, index=df.index)
            
        states['COGNITIVE_INTERNAL_DOMAIN_CASCADE'] = domain_cascade_score.astype(np.float32)

        # --- 3. 最终融合 ---
        # 融合时间级联和领域级联，得到最终的趋势加速级联分数
        final_cascade_score = (temporal_cascade_score * domain_cascade_score).astype(np.float32)
        states['COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE'] = final_cascade_score
        
        self.strategy.atomic_states.update(states)

    def _diagnose_archangel_top_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.1 · 幽灵信号斩断版】“天使长”顶部反转诊断引擎
        - 核心修复: 不再重复调用 calculate_context_scores，而是直接从 atomic_states 中消费
                      已经计算好的、最权威的 top_context_score，彻底解决信号黑洞问题。
        """
        states = {}
        # 不再重复计算，而是直接从状态库中消费最权威的信号
        top_context_score = self._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0)
        upthrust_risk = self._get_atomic_score(df, 'SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0)
        heaven_earth_risk = self._get_atomic_score(df, 'SCORE_BOARD_HEAVEN_EARTH', 0.0)
        post_peak_risk = self._get_atomic_score(df, 'COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN', 0.0)
        risk_matrix = np.stack([
            upthrust_risk.values,
            heaven_earth_risk.values,
            post_peak_risk.values,
            top_context_score.values
        ], axis=0)
        archangel_score_values = np.maximum.reduce(risk_matrix, axis=0)
        archangel_score = np.clip(archangel_score_values, 0, 1)
        states['SCORE_ARCHANGEL_TOP_REVERSAL'] = pd.Series(archangel_score, index=df.index, dtype=np.float32)
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

    def _perform_cognitive_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】认知层专用的关系元分析核心引擎 (普罗米修斯之火)
        - 核心逻辑: 对“领域间共识”的快照分进行动态调制，分析共识形成的速度与加速度。
        """
        # 从认知层专属配置中获取权重
        p_conf = get_params_block(self.strategy, 'cognitive_intelligence_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 使用“阿瑞斯之怒”的加法模型
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        # 第一维度：状态分 (State Score) - “共识”的当前状态
        state_score = snapshot_score.clip(0, 1)
        # 第二维度：速度分 (Velocity Score) - “共识”的变化速度
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 第三维度：加速度分 (Acceleration Score) - “共识”的变化加速度
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 终极融合：加法赋权
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1)
        return final_score.astype(np.float32)

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
        【V5.3 · 时空同步版】“真伪识别：打压 vs 撤退”诊断引擎
        - 核心修复: 重构了变量计算顺序，确保在构建“打压”和“撤退”证据链时，
                      明确使用同一个“盈利盘信念”信号的两个对立面（信念坚定 vs 信念瓦解），
                      彻底解决了因逻辑交叉引用导致的探针重算偏差问题。
        """
        states = {}
        norm_window = 55
        p = 5
        # --- 步骤1: 量化“近期派发强度”证据 ---
        to_main = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=True) *
                   normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=True))**0.5
        to_retail = (normalize_score(df.get(f'SLOPE_{p}_cost_divergence_D'), df.index, norm_window, ascending=False) *
                     normalize_score(df.get(f'SLOPE_{p}_turnover_from_losers_ratio_D'), df.index, norm_window, ascending=False))**0.5
        short_term_transfer_snapshot = (to_main - to_retail).astype(np.float32)
        recent_distribution_strength = (short_term_transfer_snapshot.rolling(3).mean().clip(-1, 0) * -1).astype(np.float32)
        # --- 步骤2: 量化“当日反转强度”与“动态质量”证据 ---
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
        # --- 步骤3: 准备核心对立证据：“盈利盘信念” ---
        # 修改开始(V5.3): 明确定义信念的两个对立面
        winner_conviction_0_1 = (self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION', 0.0).clip(-1, 1) * 0.5 + 0.5)
        winner_belief_score = winner_conviction_0_1 # 信念坚定分，越高越好
        winner_capitulation_score = (1.0 - winner_conviction_0_1) ** 0.7 # 信念瓦解/投降分，越高越糟
        # 修改结束(V5.3)
        # --- 步骤4: 构建“战术性打压”的证据链 (看涨) ---
        trend_quality_context = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0)
        panic_absorption_score = self._get_atomic_score(df, 'SCORE_MICRO_PANIC_ABSORPTION', 0.0)
        structural_support_score = self._get_atomic_score(df, 'SCORE_FOUNDATION_BOTTOM_CONFIRMED', 0.0)
        absorption_evidence_chain = (
            trend_quality_context *
            panic_absorption_score *
            winner_belief_score * # 使用“信念坚定分”
            (1 + structural_support_score * 0.5)
        )
        tactical_suppression_score = (
            recent_distribution_strength *
            reversal_strength *
            reversal_dynamic_quality *
            absorption_evidence_chain
        ).clip(0, 1)
        states['COGNITIVE_SCORE_TACTICAL_SUPPRESSION'] = tactical_suppression_score.astype(np.float32)
        # --- 步骤5: 构建“真实撤退/牛市陷阱”的证据链 (看跌) ---
        trend_decay_context = 1.0 - trend_quality_context
        no_absorption_score = 1.0 - panic_absorption_score
        bull_trap_evidence = 1.0 - reversal_dynamic_quality
        retreat_evidence_chain = (
            trend_decay_context *
            no_absorption_score *
            winner_capitulation_score * # 使用“信念瓦解分”
            bull_trap_evidence
        )
        true_retreat_score = (recent_distribution_strength * retreat_evidence_chain).clip(0, 1)
        states['COGNITIVE_SCORE_TRUE_RETREAT_RISK'] = true_retreat_score.astype(np.float32)
        return states


















