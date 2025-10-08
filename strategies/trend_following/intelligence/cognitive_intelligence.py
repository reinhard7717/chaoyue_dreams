# 文件: strategies/trend_following/intelligence/cognitive_intelligence.py
# 顶层认知合成模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from enum import Enum
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_unified_score, calculate_context_scores
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
        【V3.2 · 指挥链审查版】顶层认知总分合成模块
        - 核心升级: 部署“指挥链审查”探针，监控对微观行为引擎的调用。
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
        archangel_states = self._diagnose_archangel_top_reversal(df)
        self.strategy.atomic_states.update(archangel_states)
        bullish_scores = [
            self._get_atomic_score(df, 'COGNITIVE_SCORE_IGNITION_RESONANCE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_INDUSTRY_SYNERGY_OFFENSE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_REVERSAL_RELIABILITY').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_STATE_PROCESS_SYNERGY').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_ACCELERATION_CASCADE').values,
            self._get_atomic_score(df, 'COGNITIVE_SCORE_TACTICAL_REVERSAL_RESONANCE').values,
        ]
        cognitive_bullish_score = np.maximum.reduce(bullish_scores)
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
        【V2.6 · 架构对齐版】趋势质量融合评分模块
        - 核心重构: 修复了筹码健康度计算的逻辑断层，使其直接消费筹码层的终极共振信号。
        - 优化说明: 采用加权几何平均数进行融合，比加法更能体现“质量”的综合性，
                      并使用Numpy进行高效的向量化计算。
        """
        # --- 1. 获取各维度健康度分数 ---
        # 使用 get_unified_score 获取各领域标准化的健康度分数
        behavior_health_score = 1.0 - get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TOP_REVERSAL')
        fund_flow_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BULLISH_RESONANCE')
        structural_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_BULLISH_RESONANCE')
        mechanics_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'DYN_BULLISH_RESONANCE')
        
        # 趋势状态健康度通过融合Hurst指数和FFT分析结果得出，使用几何平均以求共识
        regime_health_score_hurst = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME')
        regime_health_score_fft = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT')
        regime_health_score = (regime_health_score_hurst * regime_health_score_fft)**0.5
        
        # 筹码健康度直接消费筹码层的最终看多共振信号，避免重复计算
        chip_health_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')

        # --- 2. 读取权重配置 ---
        p = get_params_block(self.strategy, 'trend_quality_params', {})
        weights = p.get('domain_weights', {})
        
        # --- 3. 加权几何平均融合 ---
        # 将所有维度分数和配置的权重一一对应
        domain_scores = [
            behavior_health_score, chip_health_score, fund_flow_health_score,
            structural_health_score, mechanics_health_score, regime_health_score
        ]
        domain_weights_config = [
            weights.get('behavior', 0.20), weights.get('chip', 0.30), weights.get('fund_flow', 0.15),
            weights.get('structural', 0.15), weights.get('mechanics', 0.10), weights.get('regime', 0.10)
        ]
        
        # 过滤掉权重为0的项，并直接处理Numpy数组以提高效率
        valid_scores = []
        valid_weights = []
        for score, weight in zip(domain_scores, domain_weights_config):
            if weight > 0:
                valid_scores.append(score.values)
                valid_weights.append(weight)

        if valid_scores:
            weights_array = np.array(valid_weights)
            weights_array /= weights_array.sum() # 权重归一化，确保总权重为1
            stacked_scores = np.stack(valid_scores, axis=0)
            # 计算加权几何平均数。该算法能有效惩罚任何一个维度的短板。
            # 公式: G = (s1^w1 * s2^w2 * ... * sn^wn)
            trend_quality_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            trend_quality_score = pd.Series(trend_quality_values, index=df.index, dtype=np.float32)
        else:
            trend_quality_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        
        # --- 4. 保存结果 ---
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
        【V3.6 · 忒弥斯校准协议版】风险元融合模块
        - 核心升级: 签署“忒弥斯校准协议”，废除“加权几何平均”的风险融合逻辑。
        - 新核心逻辑: 采用“取最大值”(`np.maximum.reduce`)的方式融合各维度风险。
                      最终的 COGNITIVE_FUSED_RISK_SCORE 现在由最强的那个风险维度定义。
        - 防御性编程 (新增): 对最终输出的风险分增加 clip(0, 2.0) 操作，防止极端值溢出。
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
        secondary_risk_discount = p_fused_risk.get('intra_dimension_fusion_params', {}).get('secondary_risk_discount', 0.3)
        for category_name, signals in risk_categories.items():
            if category_name == "说明": continue
            category_signal_scores = []
            for signal_name, signal_params in signals.items():
                if signal_name == "说明": continue
                atomic_score_np = signal_numpy_cache.get(signal_name, default_numpy_array)
                processed_score = 1.0 - atomic_score_np if signal_params.get('inverse', False) else atomic_score_np
                category_signal_scores.append(processed_score * signal_params.get('weight', 1.0))
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
        # 新增防御性编程：对最终的原始风险分进行范围裁剪，防止极端值
        states['COGNITIVE_FUSED_RISK_SCORE'] = total_fused_risk_score.clip(0, 2.0).astype(np.float32)
        return states

    def synthesize_chimera_conflict_score(self, df: pd.DataFrame) -> None:
        """
        【V1.3 · 忒弥斯校准协议版】奇美拉冲突诊断引擎
        - 核心革命: 签署“忒弥斯校准协议”，为风险分和看涨分建立统一的度量衡。
        - 新核心逻辑:
          1. 从配置中读取新的 `chimera_risk_normalization_base` 参数。
          2. 在计算冲突前，将原始的 `COGNITIVE_FUSED_RISK_SCORE` 除以此基准值，将其归一化到[0,1]区间。
          3. 使用两个都已归一化的分数来计算 `np.minimum`，确保比较的公平性。
        - 收益: 彻底解决了因度量衡不统一导致的奇美拉冲突分被严重低估的致命BUG。
        """
        states = {}
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        # 从配置中读取新的风险归一化基准值
        risk_norm_base = get_param_value(p_judge.get('chimera_risk_normalization_base'), 1000.0)
        bullish_score_normalized = self._get_atomic_score(df, 'COGNITIVE_BULLISH_SCORE', 0.0).clip(0, 1)
        # 获取原始风险分，并进行归一化处理
        raw_risk_score = self._get_atomic_score(df, 'COGNITIVE_FUSED_RISK_SCORE', 0.0)
        bearish_score_normalized = (raw_risk_score / risk_norm_base).clip(0, 1)
        # 现在比较的是两个都在[0,1]区间的、度量衡统一的分数
        conflict_score = np.minimum(bullish_score_normalized, bearish_score_normalized).clip(0, 1)
        states['COGNITIVE_SCORE_CHIMERA_CONFLICT'] = conflict_score.astype(np.float32)
        self.strategy.atomic_states.update(states)

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
        【V2.0 · 波塞冬三叉戟协议版】终极确认融合模块
        - 核心革命: 签署“波塞冬三叉戟协议”，为终极底部确认信号装上“雅典娜之盾”最终否决权。
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
        
        # 获取“雅典娜之盾”的裁决
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)

        states['COGNITIVE_ULTIMATE_BULLISH_CONFIRMATION'] = (fusion_bullish * pattern_bullish).astype(np.float32)
        states['COGNITIVE_ULTIMATE_BEARISH_CONFIRMATION'] = (fusion_bearish * pattern_bearish).astype(np.float32)
        
        # 应用“雅典娜之盾”的最终否决权
        ultimate_bottom_raw = fusion_bottom * pattern_bottom
        states['COGNITIVE_ULTIMATE_BOTTOM_CONFIRMATION'] = (ultimate_bottom_raw * bottom_context_score).astype(np.float32)
        
        # 应用“雅典娜之盾”的最终否决权 (顶部版本)
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
        【V3.0 · 波塞冬三叉戟协议版】多域反转共振分数合成模块
        - 核心革命: 签署“波塞冬三叉戟协议”，为底部反转共振信号装上“雅典娜之盾”最终否决权。
        - 新核心逻辑: 最终的底部反转共振分，必须乘以 bottom_context_score，确保只在宏观底部区域才有效。
        """
        states = {}
        default_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        bottom_weights = p.get('bottom_resonance_weights', {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        top_weights = p.get('top_resonance_weights', {'mechanics': 0.3, 'chip': 0.3, 'foundation': 0.2, 'behavior': 0.2, 'structure': 0.3})
        
        # 获取“雅典娜之盾”的裁决
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)

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
            # 应用“雅典娜之盾”的最终否决权
            bottom_reversal_values = bottom_reversal_values_raw * bottom_context_score.values
            bottom_reversal_score = pd.Series(bottom_reversal_values, index=df.index, dtype=np.float32)
        else:
            bottom_reversal_score = default_score.copy()

        states['COGNITIVE_SCORE_BOTTOM_REVERSAL_RESONANCE'] = bottom_reversal_score

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
            # 应用“雅典娜之盾”的最终否决权 (顶部版本)
            top_reversal_values_raw = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
            top_reversal_values = top_reversal_values_raw * top_context_score.values
            top_reversal_score = pd.Series(top_reversal_values, index=df.index, dtype=np.float32)
        else:
            top_reversal_score = default_score.copy()

        states['COGNITIVE_SCORE_TOP_REVERSAL_RESONANCE'] = top_reversal_score
        
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
        【V3.0 · 启示录四骑士版】“天使长”顶部反转诊断引擎
        - 核心革命: 引入第四位骑士——`top_context_score`，即由“乌拉诺斯穹顶”和“历史高点”
                      共同铸就的“结构性压力”分，形成四位一体的终极风险裁决。
        - 融合算法升级: 废除“主次风险融合算法”，升级为更稳健、更灵敏的“取最大值”融合。
                          最终风险 = MAX(上冲派发, 天地板, 高位回落, 结构性压力)。
        """
        states = {}
        # 步骤一：获取所有需要的信号，包括新的 top_context_score
        # 主动调用 calculate_context_scores 获取最核心的上下文分数
        _, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        upthrust_risk = self._get_atomic_score(df, 'SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0)
        heaven_earth_risk = self._get_atomic_score(df, 'SCORE_BOARD_HEAVEN_EARTH', 0.0)
        post_peak_risk = self._get_atomic_score(df, 'COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN', 0.0)
        # 步骤二：“启示录四骑士”融合算法
        # 将四个风险信号的Numpy数组堆叠成一个2D矩阵
        risk_matrix = np.stack([
            upthrust_risk.values,
            heaven_earth_risk.values,
            post_peak_risk.values,
            top_context_score.values  # 引入第四位骑士
        ], axis=0)
        # 使用 np.maximum.reduce 高效地找出每日最强的那个风险信号作为最终裁决
        archangel_score_values = np.maximum.reduce(risk_matrix, axis=0)
        # 步骤三：结果处理与保存
        archangel_score = np.clip(archangel_score_values, 0, 1)
        states['SCORE_ARCHANGEL_TOP_REVERSAL'] = pd.Series(archangel_score, index=df.index, dtype=np.float32)
        return states

    def synthesize_tactical_reversal_resonance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【V1.0 · 新增 · 赫尔墨斯之杖】战术反转共振融合引擎
        - 核心职责: 融合所有情报域的 TACTICAL_REVERSAL 信号，形成统一的、高价值的认知层战术共振信号。
        - 融合算法: 加权几何平均，确保只有多领域共振时才能产生高分。
        """
        states = {}
        default_score = pd.Series(0.5, index=df.index, dtype=np.float32)
        p = get_params_block(self.strategy, 'reversal_resonance_params', {})
        # 为战术反转共振定义独立的权重
        tactical_weights = p.get('tactical_resonance_weights', {'mechanics': 0.2, 'chip': 0.2, 'foundation': 0.1, 'behavior': 0.2, 'structure': 0.2, 'ff': 0.1})
        
        # --- 战术反转共振分数 ---
        tactical_sources = {
            'mechanics': get_unified_score(self.strategy.atomic_states, df.index, 'DYN_TACTICAL_REVERSAL'),
            'chip': get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_TACTICAL_REVERSAL'),
            'foundation': get_unified_score(self.strategy.atomic_states, df.index, 'FOUNDATION_TACTICAL_REVERSAL'),
            'behavior': get_unified_score(self.strategy.atomic_states, df.index, 'BEHAVIOR_TACTICAL_REVERSAL'),
            'structure': get_unified_score(self.strategy.atomic_states, df.index, 'STRUCTURE_TACTICAL_REVERSAL'),
            'ff': get_unified_score(self.strategy.atomic_states, df.index, 'FF_TACTICAL_REVERSAL')
        }
        
        tactical_scores_np = [s.values for d, s in tactical_sources.items() if tactical_weights.get(d, 0) > 0]
        tactical_weights_np = [w for d, w in tactical_weights.items() if d in tactical_sources and w > 0]
        
        if tactical_scores_np:
            weights_array = np.array(tactical_weights_np)
            weights_array /= weights_array.sum() # 权重归一化
            stacked_scores = np.stack(tactical_scores_np, axis=0)
            # 使用数值稳定的log-exp方法计算加权几何平均
            safe_scores = np.maximum(stacked_scores, 1e-9)
            weighted_log_sum = np.sum(np.log(safe_scores) * weights_array[:, np.newaxis], axis=0)
            tactical_reversal_values = np.exp(weighted_log_sum)
            tactical_reversal_score = pd.Series(tactical_reversal_values, index=df.index, dtype=np.float32)
        else:
            tactical_reversal_score = pd.Series(0.0, index=df.index, dtype=np.float32)

        states['COGNITIVE_SCORE_TACTICAL_REVERSAL_RESONANCE'] = tactical_reversal_score
        
        self.strategy.atomic_states.update(states)
        return df






















