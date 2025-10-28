# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_unified_score, calculate_holographic_dynamics, normalize_to_bipolar

class MicroBehaviorEngine:
    """
    微观行为诊断引擎
    - 核心职责: 诊断微观层面的、复杂的、但又非常具体的市场行为模式。
                这些模式通常是多个基础信号的精巧组合，用于识别主力的特定意图。
    - 来源: 从臃肿的 CognitiveIntelligence 模块中拆分而来。
    """
    def __init__(self, strategy_instance):
        """
        初始化微观行为诊断引擎。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance

    def _get_atomic_score(self, df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
        """安全地从原子状态库中获取分数。"""
        return self.strategy.atomic_states.get(name, pd.Series(default, index=df.index))

    def run_micro_behavior_synthesis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.8 · 伊卡洛斯之坠版】微观行为诊断引擎总指挥
        - 核心升级: 引入“伊卡洛斯之坠”诊断引擎，取代旧的“拉升派发”模型，
                      以更强大的多维证据链，专门识别主力在高位通过诱多完成派发的风险。
        """
        all_states = {}
        def update_states(new_states: Dict[str, pd.Series]):
            if new_states:
                all_states.update(new_states)
        update_states(self.synthesize_early_momentum_ignition(df))
        update_states(self.diagnose_deceptive_retail_flow(df))
        update_states(self.synthesize_microstructure_dynamics(df)) # 此方法已被重构
        update_states(self._synthesize_profit_taking_pressure_risk(df)) # 新增独立的风险引擎调用
        update_states(self.synthesize_euphoric_acceleration_risk(df))
        update_states(self.diagnose_hermes_gambit(df))
        update_states(self._diagnose_consolidation_breakout(df))
        early_ignition_score = all_states.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION', self._get_atomic_score(df, 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'))
        update_states(self.synthesize_reversal_reliability_score(
            df, early_ignition_score=early_ignition_score
        ))
        return all_states

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.1 · 商神杖激活版】早期动能点火诊断模块
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {}) # 读取微观行为模块的专属配置
        # 步骤一：计算原始的、纯粹的微观行为分数
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        body_size = (df['close_D'] - df['open_D']).clip(lower=0)
        body_strength_score = (body_size / candle_range).fillna(0.0)
        position_in_range_score = ((df['close_D'] - df['low_D']) / candle_range).fillna(0.0)
        momentum_strength_score = (df['pct_change_D'] / 0.10).clip(0, 1).fillna(0.0)
        raw_ignition_score = (body_strength_score * position_in_range_score * momentum_strength_score)
        # 步骤二：获取均线趋势上下文分数
        # 调用全新的、功能更强大的四维均线健康度评估引擎
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        # 核心思想：只有在健康的均线结构下发生的点火，才是有效的点火。
        snapshot_score = raw_ignition_score * ma_health_score # 使用新的 ma_health_score
        # 步骤四：对快照分进行关系元分析，得到最终的动态调制分数
        final_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        states['COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'] = final_score.astype(np.float32)
        return states

    def diagnose_deceptive_retail_flow(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.0 · 筹码升维版】隐秘吸筹诊断引擎
        - 核心升维: 对“筹码归集”支柱的诊断，从单一的“斜率”维度，升维至“状态+速度+加速度”三位一体的四维时空分析，
                      与筹码情报引擎的分析范式完全对齐，极大提升了对真实筹码集中的识别能力。
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        deception_scores_by_period = {}
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            def get_fused_pillar_score(metric_name: str, ascending: bool, period_t: int, period_c: int) -> pd.Series:
                tactical_score = normalize_score(df.get(metric_name), df.index, window=period_t, ascending=ascending)
                context_score = normalize_score(df.get(metric_name), df.index, window=period_c, ascending=ascending)
                return (tactical_score * context_score)**0.5
            disguise_score = get_fused_pillar_score(f'trade_granularity_impact_D', ascending=False, period_t=p_tactical, period_c=p_context)
            
            # 对“筹码归集”支柱进行三位一体升维
            chip_metric = 'concentration_90pct_D'
            chip_static = df.get(chip_metric, 0)
            chip_slope = df.get(f'SLOPE_{p_tactical}_{chip_metric}', 0)
            chip_accel = df.get(f'ACCEL_{p_tactical}_{chip_metric}', 0)
            # 战术层
            tactical_chip_static = normalize_score(chip_static, df.index, p_tactical, ascending=True)
            tactical_chip_slope = normalize_score(chip_slope, df.index, p_tactical, ascending=True)
            tactical_chip_accel = normalize_score(chip_accel, df.index, p_tactical, ascending=True)
            tactical_chip_quality = (tactical_chip_static * tactical_chip_slope * tactical_chip_accel)**(1/3)
            # 上下文层
            context_chip_static = normalize_score(chip_static, df.index, p_context, ascending=True)
            context_chip_slope = normalize_score(chip_slope, df.index, p_context, ascending=True)
            context_chip_accel = normalize_score(chip_accel, df.index, p_context, ascending=True)
            context_chip_quality = (context_chip_static * context_chip_slope * context_chip_accel)**(1/3)
            accumulation_score = (tactical_chip_quality * context_chip_quality)**0.5
            

            vpa_inefficiency = get_fused_pillar_score('VPA_EFFICIENCY_D', ascending=False, period_t=p_tactical, period_c=p_context)
            price_stagnation = 1.0 - get_fused_pillar_score(df.get(f'SLOPE_{p_tactical}_close_D', pd.Series(0, index=df.index)).abs(), ascending=True, period_t=p_tactical, period_c=p_context)
            suppression_score = (vpa_inefficiency * price_stagnation)**0.5
            cost_advantage_score = get_fused_pillar_score('main_buy_cost_advantage_D', ascending=False, period_t=p_tactical, period_c=p_context)
            snapshot_score = (disguise_score * accumulation_score * suppression_score * cost_advantage_score)**(1/4)
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            deception_scores_by_period[p_tactical] = period_score
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += deception_scores_by_period.get(p_tactical, 0.0) * weight
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION'] = final_fused_score.clip(0, 1).astype(np.float32)
        return states

    def diagnose_hermes_gambit(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.0 · 资金流升维版】“赫尔墨斯诡计”诊断引擎 (压单吸筹识别)
        - 核心升维: 对“表象矛盾”证据链中主力资金流出信号进行四维质量评估。
        """
        states = {}
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        hermes_scores_by_period = {}
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # --- 证据1: 表象矛盾 (分层计算) ---
            # 使用四维质量评估主力资金流出
            # 主力资金流出质量 (MFNF, ascending=False)
            main_force_outflow_quality = self._calculate_4d_metric_quality(
                df, 'main_force_net_flow_consensus', p, context_p, ascending=False
            )
            # 资金源分歧度 (发散度越高越好，ascending=True)
            tactical_div_ts_ths = normalize_score(df.get('divergence_ts_ths_D', pd.Series(0, index=df.index)), df.index, window=p, ascending=True)
            tactical_div_ts_dc = normalize_score(df.get('divergence_ts_dc_D', pd.Series(0, index=df.index)), df.index, window=p, ascending=True)
            context_div_ts_ths = normalize_score(df.get('divergence_ts_ths_D', pd.Series(0, index=df.index)), df.index, window=context_p, ascending=True)
            context_div_ts_dc = normalize_score(df.get('divergence_ts_dc_D', pd.Series(0, index=df.index)), df.index, window=context_p, ascending=True)
            fused_source_divergence = (np.maximum(tactical_div_ts_ths, tactical_div_ts_dc) * np.maximum(context_div_ts_ths, context_div_ts_dc))**0.5
            # 融合：高质量的流出表象 * 高质量的分歧
            contradiction_flow_score = (main_force_outflow_quality * fused_source_divergence)**0.5
            
            # --- 证据2: 价量矛盾 (分层计算) ---
            tactical_volume_spike = normalize_score(df['volume_D'], df.index, window=p, ascending=True)
            tactical_price_stagnation = 1.0 - normalize_score(df['pct_change_D'].abs(), df.index, window=p, ascending=True)
            tactical_contradiction_pv = (tactical_volume_spike * tactical_price_stagnation)**0.5
            context_volume_spike = normalize_score(df['volume_D'], df.index, window=context_p, ascending=True)
            context_price_stagnation = 1.0 - normalize_score(df['pct_change_D'].abs(), df.index, window=context_p, ascending=True)
            context_contradiction_pv = (context_volume_spike * context_price_stagnation)**0.5
            contradiction_pv_score = (tactical_contradiction_pv * context_contradiction_pv)**0.5
            # --- 证据3: 结果矛盾 (筹码集中度上升，已在筹码升维中修改) ---
            chip_metric = 'concentration_90pct'
            chip_static = df.get(f'{chip_metric}_D', 0)
            chip_slope = df.get(f'SLOPE_{p}_{chip_metric}_D', 0)
            chip_accel = df.get(f'ACCEL_{p}_{chip_metric}_D', 0)
            tactical_chip_static = normalize_score(chip_static, df.index, p, ascending=True)
            tactical_chip_slope = normalize_score(chip_slope, df.index, p, ascending=True)
            tactical_chip_accel = normalize_score(chip_accel, df.index, p, ascending=True)
            tactical_chip_quality = (tactical_chip_static * tactical_chip_slope * tactical_chip_accel)**(1/3)
            context_chip_static = normalize_score(chip_static, df.index, context_p, ascending=True)
            context_chip_slope = normalize_score(df.get(f'SLOPE_{context_p}_{chip_metric}_D', 0), df.index, context_p, ascending=True)
            context_chip_accel = normalize_score(df.get(f'ACCEL_{context_p}_{chip_metric}_D', 0), df.index, context_p, ascending=True)
            context_chip_quality = (context_chip_static * context_chip_slope * context_chip_accel)**(1/3)
            chip_concentration_rising_score = (tactical_chip_quality * context_chip_quality)**0.5
            # --- 证据4: 环境矛盾 (静态上下文，无需分层) ---
            trend_quality_context = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.5)
            # --- 最终融合，生成“瞬时关系快照分” ---
            hermes_gambit_snapshot = (
                contradiction_flow_score *
                contradiction_pv_score *
                chip_concentration_rising_score *
                trend_quality_context
            )**(1/4)
            # --- 对快照分进行关系元分析，捕捉其动态变化 ---
            final_period_score = self._perform_micro_behavior_relational_meta_analysis(df, hermes_gambit_snapshot)
            hermes_scores_by_period[p] = final_period_score
        # --- 跨周期融合，生成最终信号 ---
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.1, 55: 0.1} # 赋予中短期更高的权重
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p in periods:
                weight = tf_weights.get(p, 0) / total_weight
                final_fused_score += hermes_scores_by_period.get(p, 0.0) * weight
        states['SCORE_MICRO_HERMES_GAMBIT'] = final_fused_score.clip(0, 1).astype(np.float32)
        return states

    def synthesize_reversal_reliability_score(self, df: pd.DataFrame, early_ignition_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V6.0 · 分层印证版】高质量战备可靠性诊断引擎
        - 核心升级: 引入“分层动态印证”框架。对构成可靠性的核心原子信号（如位置、超卖、趋势潜力）进行分层验证，提升最终信号的稳健性。
        """
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        # 引入分层印证框架
        periods = [5, 13, 21, 55] # 此处使用稍长周期
        sorted_periods = sorted(periods)
        reliability_scores_by_period = {}
        # 静态信号，循环外计算
        chip_accumulation_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        chip_reversal_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL')
        conviction_strengthening_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING')
        shareholder_turnover_score = np.maximum.reduce([chip_accumulation_score.values, chip_reversal_score.values, conviction_strengthening_score.values])
        shareholder_quality_score = pd.Series(shareholder_turnover_score, index=df.index, dtype=np.float32)
        states['SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT'] = shareholder_quality_score
        vol_compression_score = get_unified_score(self.strategy.atomic_states, df.index, 'VOL_COMPRESSION')
        if len(early_ignition_score) != len(df.index):
            early_ignition_score = early_ignition_score.reindex(df.index, fill_value=0.0)
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # --- 背景分 (分层计算) ---
            tactical_price_pos = 1.0 - normalize_score(df['close_D'], df.index, window=p_tactical, ascending=True)
            tactical_rsi_w = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=p_tactical, ascending=False)
            tactical_background = np.maximum(tactical_price_pos, tactical_rsi_w)
            context_price_pos = 1.0 - normalize_score(df['close_D'], df.index, window=p_context, ascending=True)
            context_rsi_w = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=p_context, ascending=False)
            context_background = np.maximum(context_price_pos, context_rsi_w)
            background_score = (tactical_background * context_background)**0.5
            # --- 趋势潜力分 (分层计算) ---
            fft_trend_score = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT', 0.0)
            fft_trend_slope = fft_trend_score.diff(5).fillna(0)
            tactical_trend_potential = normalize_score(fft_trend_slope.clip(lower=0), df.index, window=p_tactical, ascending=True)
            context_trend_potential = normalize_score(fft_trend_slope.clip(lower=0), df.index, window=p_context, ascending=True)
            trend_potential_score = (tactical_trend_potential * context_trend_potential)**0.5
            # --- 重新组装 ---
            ignition_weights = get_param_value(p.get('ignition_weights'), {'early': 0.5, 'vol': 0.2, 'potential': 0.3})
            ignition_confirmation_score = ((early_ignition_score ** ignition_weights['early']) * (vol_compression_score ** ignition_weights['vol']) * (trend_potential_score ** ignition_weights['potential']))
            main_reliability_weights = get_param_value(p.get('main_reliability_weights'), {'shareholder': 0.5, 'ignition': 0.5})
            main_score = ((shareholder_quality_score ** main_reliability_weights['shareholder']) * (ignition_confirmation_score ** main_reliability_weights['ignition']))
            bonus_factor = get_param_value(p.get('reversal_reliability_bonus_factor'), 0.5)
            raw_reliability_score = (main_score * (1 + background_score * bonus_factor)).clip(0, 1)
            snapshot_score = raw_reliability_score * ma_health_score
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            reliability_scores_by_period[p_tactical] = period_score
        # --- 跨周期融合 ---
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += reliability_scores_by_period.get(p_tactical, 0.0) * weight
        states['COGNITIVE_SCORE_REVERSAL_RELIABILITY'] = final_fused_score.clip(0, 1).astype(np.float32)
        # 临时保留部分原子信号输出
        states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = background_score.astype(np.float32) if 'background_score' in locals() else pd.Series(0.0, index=df.index)
        states['INTERNAL_SCORE_TREND_POTENTIAL'] = trend_potential_score.astype(np.float32) if 'trend_potential_score' in locals() else pd.Series(0.0, index=df.index)
        states['SCORE_IGNITION_CONFIRMATION'] = ignition_confirmation_score.astype(np.float32) if 'ignition_confirmation_score' in locals() else pd.Series(0.0, index=df.index)
        
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V14.0 · 双星归位版】市场微观结构动态诊断引擎
        - 核心重构: 彻底废除原有的非对称、易冲突的逻辑。
                      1. 创建一个统一的、双极性的“权力转移”动态分。
                      2. 将此双极性分数拆分为两个绝对互斥的看涨机会和看跌风险信号。
                      3. 将原有的复杂风险逻辑剥离至独立的 `_synthesize_profit_taking_pressure_risk` 引擎。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        recent_reversal_context = self._get_atomic_score(df, 'SCORE_CONTEXT_RECENT_REVERSAL', 0.0)
        risk_suppression_factor = (1.0 - recent_reversal_context).clip(0, 1)
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        
        # 全新的、统一的、对称的逻辑
        power_shift_scores = {}
        conviction_scores = {}
        granularity_impact_metric = 'trade_granularity_impact_D'
        concentration_metric = 'trade_concentration_index_D'
        conviction_metric = 'main_force_conviction_ratio_D'

        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical

            # --- 1. 计算纯粹的“权力转移”双极性分数 ---
            # 证据(UP): 颗粒度/集中度提升
            granularity_up = normalize_score(df.get(f'SLOPE_{p_tactical}_{granularity_impact_metric}'), df.index, window=p_tactical, ascending=True)
            dominance_up = normalize_score(df.get(f'SLOPE_{p_tactical}_{concentration_metric}'), df.index, window=p_tactical, ascending=True)
            evidence_up = (granularity_up * dominance_up)**0.5
            
            # 证据(DOWN): 颗粒度/集中度下降
            granularity_down = normalize_score(df.get(f'SLOPE_{p_tactical}_{granularity_impact_metric}'), df.index, window=p_tactical, ascending=False)
            dominance_down = normalize_score(df.get(f'SLOPE_{p_tactical}_{concentration_metric}'), df.index, window=p_tactical, ascending=False)
            evidence_down = (granularity_down * dominance_down)**0.5

            # 生成双极性分数: 正代表权力流向主力，负代表流向散户
            bipolar_power_shift_snapshot = evidence_up - evidence_down
            power_shift_scores[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, bipolar_power_shift_snapshot)

            # --- 2. 计算纯粹的“主力信念”双极性分数 ---
            conviction_up = normalize_score(df.get(f'SLOPE_{p_tactical}_{conviction_metric}'), df.index, window=p_tactical, ascending=True)
            conviction_down = normalize_score(df.get(f'SLOPE_{p_tactical}_{conviction_metric}'), df.index, window=p_tactical, ascending=False)
            bipolar_conviction_snapshot = conviction_up - conviction_down
            conviction_scores[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, bipolar_conviction_snapshot)

        # --- 跨周期融合 ---
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        def fuse_bipolar_scores(score_dict):
            final_score = pd.Series(0.0, index=df.index)
            total_weight = sum(tf_weights.get(p, 0) for p in periods)
            if total_weight > 0:
                for p in periods:
                    weight = tf_weights.get(p, 0) / total_weight
                    final_score += score_dict.get(p, 0.0) * weight
            return final_score.clip(-1, 1) # 确保结果在[-1, 1]

        final_bipolar_power_shift = fuse_bipolar_scores(power_shift_scores)
        final_bipolar_conviction = fuse_bipolar_scores(conviction_scores)

        # --- 拆分为两个互斥的单极信号 ---
        # 权力转移
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = (final_bipolar_power_shift.clip(lower=0) * ma_health_score).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = (final_bipolar_power_shift.clip(upper=0).abs() * (1 - ma_health_score) * risk_suppression_factor).astype(np.float32)
        
        # 主力信念
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = (final_bipolar_conviction.clip(lower=0) * ma_health_score).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = (final_bipolar_conviction.clip(upper=0).abs() * (1 - ma_health_score) * risk_suppression_factor).astype(np.float32)
        
        return states

    def _synthesize_profit_taking_pressure_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 核心指标重构版】“利润兑现压力”风险诊断引擎
        - 核心重构: 废弃旧的、基于间接推断的逻辑。
                      改为直接消费专门设计的核心指标 'profit_taking_urgency_D' 和 'profit_realization_premium_D'。
        - 核心逻辑: 对“了结紧迫度”和“兑现溢价”两大核心证据进行独立的三维动态分析，然后加权融合，
                      最终生成一个高保真、高可信度的利润兑现风险信号。
        """
        states = {}
        signal_name = 'COGNITIVE_RISK_PROFIT_TAKING_PRESSURE'
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        pressure_scores_by_period = {}
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # 证据维度一：获利盘了结紧迫度 (profit_taking_urgency_D)
            urgency_quality = self._calculate_4d_metric_quality(
                df, 'profit_taking_urgency_D', p_tactical, p_context, ascending=True
            )
            # 证据维度二：利润兑现溢价 (profit_realization_premium_D)
            premium_quality = self._calculate_4d_metric_quality(
                df, 'profit_realization_premium_D', p_tactical, p_context, ascending=True
            )
            # 融合两大核心证据，生成风险快照分
            snapshot_score = (urgency_quality * premium_quality)**0.5
            # 对快照分进行关系元分析，捕捉其动态变化
            pressure_scores_by_period[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        # 跨周期融合
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.15, 55: 0.05}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p in periods:
                weight = tf_weights.get(p, 0) / total_weight
                final_fused_score += pressure_scores_by_period.get(p, 0.0) * weight
        states[signal_name] = final_fused_score.clip(0, 1).astype(np.float32)
        return states

    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.0 · 最终决战版】亢奋加速风险/机会诊断引擎
        - 核心重构: 彻底重写函数，确保逻辑清晰、无懈可击。
        - 作战流程:
          1. 计算原始的、中性的“亢奋事件”分。
          2. 构建“看涨上下文护盾”，融合底部区域、筹码锁仓、赢家信念三大情报。
          3. 执行“嬗变”裁决：
             - 最终风险 = 原始亢奋事件 * (1 - 护盾分)
             - 最终机会 = 原始亢奋事件 * 护盾分
        - 收益: 能够正确区分底部的“点火机会”和顶部的“派发风险”，解决了将健康启动误判为风险的致命缺陷。
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True):
            return states
        # --- 步骤1: 计算原始亢奋事件分 (Raw Euphoric Event Score) ---
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        epsilon = 1e-9
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        euphoric_scores_by_period = {}
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_bias = normalize_score(df.get(f'BIAS_{p_tactical}_D', pd.Series(0.5, index=df.index)).abs(), df.index, window=p_tactical, ascending=True)
            tactical_vol_ratio = (df['volume_D'] / (df.get(f'VOL_MA_{p_tactical}_D', df['volume_D']) + epsilon)).fillna(1.0)
            tactical_vol_spike = normalize_score(tactical_vol_ratio, df.index, window=p_tactical, ascending=True)
            tactical_atr_ratio = (df.get('ATR_14_D', pd.Series(0.0, index=df.index)) / (df['close_D'] + epsilon)).fillna(0.0)
            tactical_volatility = normalize_score(tactical_atr_ratio, df.index, window=p_tactical, ascending=True)
            context_bias = normalize_score(df.get(f'BIAS_{p_context}_D', pd.Series(0.5, index=df.index)).abs(), df.index, window=p_context, ascending=True)
            context_vol_ratio = (df['volume_D'] / (df.get(f'VOL_MA_{p_context}_D', df['volume_D']) + epsilon)).fillna(1.0)
            context_vol_spike = normalize_score(context_vol_ratio, df.index, window=p_context, ascending=True)
            context_atr_ratio = (df.get('ATR_14_D', pd.Series(0.0, index=df.index)) / (df['close_D'] + epsilon)).fillna(0.0)
            context_volatility = normalize_score(context_atr_ratio, df.index, window=p_context, ascending=True)
            bias_score = (tactical_bias * context_bias)**0.5
            volume_spike_score = (tactical_vol_spike * context_vol_spike)**0.5
            volatility_score = (tactical_volatility * context_volatility)**0.5
            total_range = (df['high_D'] - df['low_D']).replace(0, epsilon)
            upper_shadow = (df['high_D'] - np.maximum(df['close_D'], df['open_D'])).clip(lower=0)
            upthrust_score = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
            ma55 = df.get('EMA_55_D', df['close_D'])
            rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
            wave_channel_height = (rolling_high_55d - ma55).replace(0, epsilon)
            stretch_from_ma55_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
            ma55_is_rising = (ma55 > ma55.shift(3)).astype(float)
            bias_55d = df.get('BIAS_55_D', pd.Series(0.5, index=df.index))
            price_is_near_ma55 = (bias_55d.abs() < 0.15).astype(float)
            bbw_d = df.get('BBW_21_2.0_D', pd.Series(0.5, index=df.index))
            volatility_was_low = (bbw_d.shift(1) < bbw_d.rolling(60).quantile(0.3)).astype(float)
            safe_launch_context_score = (ma55_is_rising * price_is_near_ma55 * volatility_was_low)
            raw_risk_factors = (bias_score * volume_spike_score * volatility_score * upthrust_score)**(1/4)
            raw_euphoric_risk_score = (raw_risk_factors * stretch_from_ma55_score * (1 - safe_launch_context_score))
            snapshot_score = raw_euphoric_risk_score * ma_health_score
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            euphoric_scores_by_period[p_tactical] = period_score
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        dynamic_raw_euphoric_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                dynamic_raw_euphoric_score += euphoric_scores_by_period.get(p_tactical, 0.0) * weight
        # --- 步骤2: 构建“看涨上下文护盾” (Bullish Context Shield) ---
        bottom_zone_context = self._get_atomic_score(df, 'SCORE_CONTEXT_DEEP_BOTTOM_ZONE', 0.0)
        chip_lockdown_context = self._get_atomic_score(df, 'SCORE_CHIP_BOTTOM_ACCUMULATION_LOCKDOWN', 0.0)
        winner_conviction_raw = self._get_atomic_score(df, 'PROCESS_META_WINNER_CONVICTION', 0.0)
        winner_conviction_context = (winner_conviction_raw.clip(-1, 1) * 0.5 + 0.5)
        bullish_context_shield = (
            bottom_zone_context *
            chip_lockdown_context *
            winner_conviction_context
        )**(1/3)
        # --- 步骤3: 执行“嬗变”裁决 (Transmutation Adjudication) ---
        final_risk_score = (dynamic_raw_euphoric_score * (1 - bullish_context_shield)).clip(0, 1)
        ignition_opportunity_score = (dynamic_raw_euphoric_score * bullish_context_shield).clip(0, 1)
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score.astype(np.float32)
        states['COGNITIVE_OPPORTUNITY_IGNITION_ACCELERATION'] = ignition_opportunity_score.astype(np.float32)
        return states

    def _perform_micro_behavior_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V5.3 · 加速度校准版】微观行为专用的关系元分析核心引擎
        - 核心修复: 修正了“加速度”计算的致命逻辑错误。加速度是速度的一阶导数，
                      因此其计算应为 relationship_trend.diff(1)，而不是错误的 diff(meta_window)。
        """
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.6)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.2)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.2)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        bipolar_snapshot = snapshot_score.clip(0, 1)
        relationship_trend = bipolar_snapshot.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # [代码修改开始]
        # 致命错误修复：加速度是速度(trend)的一阶导数，应使用 diff(1)
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
        final_bipolar_score = np.where(bipolar_snapshot >= 0, net_force.clip(lower=0), net_force.clip(upper=0))
        final_unipolar_score = (pd.Series(final_bipolar_score, index=df.index) + 1) / 2.0
        return final_unipolar_score.astype(np.float32)

    def _calculate_ma_health(self, df: pd.DataFrame, params: dict, norm_window: int) -> pd.Series:
        """
        【V1.4 · 双极性输出版】“赫尔墨斯的商神杖”四维均线健康度评估引擎
        - 核心升级: 遵循统一范式，输出一个[-1, 1]的双极性健康分。
                      +1 代表极度看涨，-1 代表极度看跌，0 代表中性。
        """
        p_ma_health = get_param_value(params.get('ma_health_fusion_weights'), {})
        weights = {
            'alignment': get_param_value(p_ma_health.get('alignment'), 0.15),
            'slope': get_param_value(p_ma_health.get('slope'), 0.15),
            'accel': get_param_value(p_ma_health.get('accel'), 0.2),
            'relational': get_param_value(p_ma_health.get('relational'), 0.5)
        }
        ma_periods = [5, 13, 21, 55]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.0, index=df.index, dtype=np.float32) # 默认返回0.0 (中性)
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        # 1. 排列健康度 (Alignment Health)
        bull_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] > df[f'EMA_{ma_periods[i+1]}_D']).astype(float) for i in range(len(ma_periods) - 1)]
        bear_alignment_scores = [(df[f'EMA_{ma_periods[i]}_D'] < df[f'EMA_{ma_periods[i+1]}_D']).astype(float) for i in range(len(ma_periods) - 1)]
        bull_alignment_health = np.mean(bull_alignment_scores, axis=0) if bull_alignment_scores else np.full(len(df.index), 0.5)
        bear_alignment_health = np.mean(bear_alignment_scores, axis=0) if bear_alignment_scores else np.full(len(df.index), 0.5)
        # 2. 速度健康度 (Slope Health)
        slope_cols = [f'SLOPE_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in slope_cols):
            normalized_slopes = [normalize_to_bipolar(df[col], df.index, norm_window).values for col in slope_cols]
            slope_health = np.mean(np.stack(normalized_slopes, axis=0), axis=0)
        else:
            slope_health = np.full(len(df.index), 0.0)
        # 3. 加速度健康度 (Accel Health)
        accel_cols = [f'ACCEL_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in accel_cols):
            normalized_accels = [normalize_to_bipolar(df[col], df.index, norm_window).values for col in accel_cols]
            accel_health = np.mean(np.stack(normalized_accels, axis=0), axis=0)
        else:
            accel_health = np.full(len(df.index), 0.0)
        # 4. 关系健康度 (Relational Health)
        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health_raw = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True).values
        relational_health = (relational_health_raw * 2 - 1).clip(-1, 1) # 转换为双极性
        # 5. 融合：排列分单独计算，其余为双极性
        # 融合逻辑：将排列分转换为双极性，然后进行加权平均
        bipolar_alignment = bull_alignment_health - bear_alignment_health
        scores = np.stack([bipolar_alignment, slope_health, accel_health, relational_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()
        # 使用加权算术平均进行融合
        final_score_values = np.sum(scores * weights_array[:, np.newaxis], axis=0)
        return pd.Series(final_score_values, index=df.index, dtype=np.float32).clip(-1, 1)

    def _calculate_4d_metric_quality(self, df: pd.DataFrame, metric_name: str, p: int, context_p: int, ascending: bool) -> pd.Series:
        """
        【V1.2 · 健壮路径协议版】计算原子指标的四维质量分。
        - 核心修正: 建立健壮的列名处理协议。函数内部会自动处理'_D'后缀，
                      确保无论调用者传入 'metric' 还是 'metric_D' 都能正确找到数据列，
                      从根本上解决因命名不一致导致的 AttributeError。
        """
        # 建立健壮的列名处理协议，确保总能正确找到带 '_D' 后缀的列
        base_metric_name = metric_name[:-2] if metric_name.endswith('_D') else metric_name
        full_metric_name = f"{base_metric_name}_D"
        # 状态
        static = df.get(full_metric_name, 0)
        # 速度 (战术层)
        slope = df.get(f'SLOPE_{p}_{full_metric_name}', 0)
        # 加速度 (战术层)
        accel = df.get(f'ACCEL_{p}_{full_metric_name}', 0)
        # 战术层 (p)
        tactical_static = normalize_score(static, df.index, p, ascending=ascending)
        tactical_slope = normalize_score(slope, df.index, p, ascending=ascending)
        tactical_accel = normalize_score(accel, df.index, p, ascending=ascending)
        tactical_quality = (tactical_static * tactical_slope * tactical_accel)**(1/3)
        # 战略/上下文层 (context_p)
        context_static = normalize_score(static, df.index, context_p, ascending=ascending)
        context_slope = normalize_score(df.get(f'SLOPE_{context_p}_{full_metric_name}', 0), df.index, context_p, ascending=ascending)
        context_accel = normalize_score(df.get(f'ACCEL_{context_p}_{full_metric_name}', 0), df.index, context_p, ascending=ascending)
        context_quality = (context_static * context_slope * context_accel)**(1/3)
        # 最终融合 (战术 * 战略)
        return (tactical_quality * context_quality)**0.5
    
    def _diagnose_consolidation_breakout(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】“结构性盘整突破”诊断引擎
        - 核心逻辑: 识别价格在经历充分盘整后，放量向上突破盘整区间的关键结构性事件。
        """
        #
        states = {}
        signal_name = 'SCORE_STRUCTURAL_CONSOLIDATION_BREAKOUT'
        p_conf = get_params_block(self.strategy, 'consolidation_breakout_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return states
        min_duration = get_param_value(p_conf.get('min_consolidation_duration'), 5)
        vol_ma_period = get_param_value(p_conf.get('volume_confirmation_ma_period'), 21)
        min_closing_strength = get_param_value(p_conf.get('min_closing_strength'), 0.7)
        # 证据1: 战备状态 (处于盘整期且持续时间足够)
        is_consolidating = df.get('is_consolidation_D', pd.Series(0, index=df.index)) == 1
        duration_met = df.get('dynamic_consolidation_duration_D', pd.Series(0, index=df.index)) >= min_duration
        setup_condition = is_consolidating & duration_met
        # 证据2: 突破信号 (收盘价突破盘整上轨)
        breakout_condition = df['close_D'] > df.get('dynamic_consolidation_high_D', pd.Series(np.inf, index=df.index))
        # 证据3: 力量确认 (成交量 & 收盘强度)
        vol_ma_col = f'VOL_MA_{vol_ma_period}_D'
        volume_confirmation = df['volume_D'] > df.get(vol_ma_col, pd.Series(np.inf, index=df.index))
        closing_strength = df.get('closing_strength_index_D', pd.Series(0.0, index=df.index)) > min_closing_strength
        confirmation_condition = volume_confirmation & closing_strength
        # 融合所有条件，生成瞬时快照分
        snapshot_score = (setup_condition & breakout_condition & confirmation_condition).astype(float)
        # 对快照分进行关系元分析，得到最终的动态信号
        final_signal_dict = self._perform_micro_behavior_relational_meta_analysis(df=df, snapshot_score=snapshot_score)
        states[signal_name] = final_signal_dict
        return states
        



