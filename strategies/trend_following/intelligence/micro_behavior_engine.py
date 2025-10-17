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
        update_states(self.synthesize_microstructure_dynamics(df))
        update_states(self.synthesize_euphoric_acceleration_risk(df))
        update_states(self.synthesize_post_peak_downturn_risk(df))
        update_states(self.diagnose_hermes_gambit(df))
        # 调用全新的“伊卡洛斯之坠”诊断引擎
        update_states(self.diagnose_icarus_fall_risk(df))
        
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
        【V5.0 · 物理事实重构版】隐秘吸筹诊断引擎 (原名：伪装散户吸筹)
        - 核心重构: 彻底废除旧的、基于模糊表象的逻辑。基于“物理事实”构建全新的四维证据链，以识别主力在压制价格的同时，
                      通过行为伪装（拆单）实现筹码归集的高级战术。
        - 新证据链:
          1. 行为伪装: “交易颗粒度影响力”极低，呈现散户化特征。
          2. 筹码归集: 筹码集中度斜率为正，发生事实上的集中。
          3. 价格压制: VPA效率低下，成交量无法推升价格。
          4. 成本优势: 主力以低于市场均价的成本吸筹。
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        # 引入分层印证框架和新的四维证据链
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        deception_scores_by_period = {}
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # --- 分层计算四大支柱证据 ---
            def get_fused_pillar_score(metric_name: str, ascending: bool, period_t: int, period_c: int) -> pd.Series:
                tactical_score = normalize_score(df.get(metric_name), df.index, window=period_t, ascending=ascending)
                context_score = normalize_score(df.get(metric_name), df.index, window=period_c, ascending=ascending)
                return (tactical_score * context_score)**0.5
            # 支柱一: 行为伪装 (交易颗粒度影响力低)
            disguise_score = get_fused_pillar_score(f'trade_granularity_impact_D', ascending=False, period_t=p_tactical, period_c=p_context)
            # 支柱二: 筹码归集 (集中度斜率高)
            accumulation_score = get_fused_pillar_score(f'SLOPE_{p_tactical}_concentration_90pct_D', ascending=True, period_t=p_tactical, period_c=p_context)
            # 支柱三: 价格压制 (VPA效率低 + 价格平稳)
            vpa_inefficiency = get_fused_pillar_score('VPA_EFFICIENCY_D', ascending=False, period_t=p_tactical, period_c=p_context)
            price_stagnation = 1.0 - get_fused_pillar_score(df.get(f'SLOPE_{p_tactical}_close_D', pd.Series(0, index=df.index)).abs(), ascending=True, period_t=p_tactical, period_c=p_context)
            suppression_score = (vpa_inefficiency * price_stagnation)**0.5
            # 支柱四: 成本优势 (主力买入成本低于收盘价)
            cost_advantage_score = get_fused_pillar_score('main_buy_cost_advantage_D', ascending=False, period_t=p_tactical, period_c=p_context)
            # --- 融合四大支柱，生成“瞬时关系快照分” ---
            snapshot_score = (disguise_score * accumulation_score * suppression_score * cost_advantage_score)**(1/4)
            # --- 对快照分进行关系元分析，得到该周期的动态分数 ---
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            deception_scores_by_period[p_tactical] = period_score
        # --- 跨周期融合，生成最终信号 ---
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
        【V3.0 · 分层印证版】“赫尔墨斯诡计”诊断引擎 (压单吸筹识别)
        - 核心升级: 借鉴筹码情报引擎的“分层动态印证”框架。每个战术周期的行为，都由其更长一级周期的趋势进行印证，形成动态共振，以提升信号的可靠性。
        - 印证链: 1日由5日印证，5日由13日印证，以此类推。
        - 证据链:
          1. 表象矛盾: 主力资金共识为净流出，但Tushare(物理)与THS/DC(情绪)之间存在巨大正向分歧。
          2. 价量矛盾: 成交量放大 vs 价格滞涨或下跌。
          3. 结果矛盾: 主力看似卖出 vs 筹码集中度反而上升。
          4. 环境矛盾: 短期价格走弱 vs 长期趋势依然健康。
        """
        states = {}
        # 引入分层印证框架
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        hermes_scores_by_period = {}
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # --- 证据1: 表象矛盾 (分层计算) ---
            # 战术层 (p)
            tactical_main_force_outflow = normalize_score(df.get('main_force_net_flow_consensus_D', pd.Series(0, index=df.index)), df.index, window=p, ascending=False)
            tactical_div_ts_ths = normalize_score(df.get('divergence_ts_ths_D', pd.Series(0, index=df.index)), df.index, window=p, ascending=True)
            tactical_div_ts_dc = normalize_score(df.get('divergence_ts_dc_D', pd.Series(0, index=df.index)), df.index, window=p, ascending=True)
            tactical_source_divergence = np.maximum(tactical_div_ts_ths, tactical_div_ts_dc)
            tactical_contradiction_flow = (tactical_main_force_outflow * tactical_source_divergence)**0.5
            # 上下文层 (context_p)
            context_main_force_outflow = normalize_score(df.get('main_force_net_flow_consensus_D', pd.Series(0, index=df.index)), df.index, window=context_p, ascending=False)
            context_div_ts_ths = normalize_score(df.get('divergence_ts_ths_D', pd.Series(0, index=df.index)), df.index, window=context_p, ascending=True)
            context_div_ts_dc = normalize_score(df.get('divergence_ts_dc_D', pd.Series(0, index=df.index)), df.index, window=context_p, ascending=True)
            context_source_divergence = np.maximum(context_div_ts_ths, context_div_ts_dc)
            context_contradiction_flow = (context_main_force_outflow * context_source_divergence)**0.5
            # 融合
            contradiction_flow_score = (tactical_contradiction_flow * context_contradiction_flow)**0.5
            # --- 证据2: 价量矛盾 (分层计算) ---
            # 战术层 (p)
            tactical_volume_spike = normalize_score(df['volume_D'], df.index, window=p, ascending=True)
            tactical_price_stagnation = 1.0 - normalize_score(df['pct_change_D'].abs(), df.index, window=p, ascending=True)
            tactical_contradiction_pv = (tactical_volume_spike * tactical_price_stagnation)**0.5
            # 上下文层 (context_p)
            context_volume_spike = normalize_score(df['volume_D'], df.index, window=context_p, ascending=True)
            context_price_stagnation = 1.0 - normalize_score(df['pct_change_D'].abs(), df.index, window=context_p, ascending=True)
            context_contradiction_pv = (context_volume_spike * context_price_stagnation)**0.5
            # 融合
            contradiction_pv_score = (tactical_contradiction_pv * context_contradiction_pv)**0.5
            # --- 证据3: 结果矛盾 (分层计算) ---
            # 战术层 (p)
            tactical_chip_rising = normalize_score(df.get(f'SLOPE_{p}_concentration_90pct_D', pd.Series(0, index=df.index)).clip(lower=0), df.index, window=p, ascending=True)
            # 上下文层 (context_p)
            context_chip_rising = normalize_score(df.get(f'SLOPE_{context_p}_concentration_90pct_D', pd.Series(0, index=df.index)).clip(lower=0), df.index, window=context_p, ascending=True)
            # 融合
            chip_concentration_rising_score = (tactical_chip_rising * context_chip_rising)**0.5
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

    def diagnose_icarus_fall_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 分层印证版】“伊卡洛斯之坠”风险诊断引擎
        - 核心升级: 引入“分层动态印证”框架。将四重证据链的计算分层，在多个时间维度上进行交叉验证，极大提升了风险识别的准确性。
        """
        states = {}
        p = get_params_block(self.strategy, 'icarus_fall_params', {})
        if not get_param_value(p.get('enabled'), True):
            return states
        # 引入分层印证框架
        periods = [1, 5, 13, 21, 55]
        sorted_periods = sorted(periods)
        icarus_scores_by_period = {}
        previous_trend_quality = self._get_atomic_score(df, 'COGNITIVE_SCORE_TREND_QUALITY', 0.0).shift(1).fillna(0.5)
        top_context_score = self._get_atomic_score(df, 'CONTEXT_TOP_SCORE', 0.0)
        betrayal_evidence = (previous_trend_quality * top_context_score)**0.5
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # --- 证据链 1: 虚假的繁荣 (分层计算) ---
            # 战术层
            tactical_rally = normalize_score(df['pct_change_D'].clip(lower=0), df.index, window=p_tactical, ascending=True)
            tactical_inefficient_vol = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, window=p_tactical, ascending=False)
            tactical_vol_spike = normalize_score(df['volume_D'] / df.get(f'VOL_MA_{p_tactical}_D', df['volume_D']), df.index, window=p_tactical, ascending=True)
            close_position_in_range = ((df['close_D'] - df['low_D']) / (df['high_D'] - df['low_D']).replace(0, np.nan)).fillna(0.5)
            closing_weakness_score = 1.0 - close_position_in_range
            tactical_high_level_dist = (tactical_rally * closing_weakness_score * tactical_inefficient_vol)**(1/3)
            is_at_limit_up = (df['close_D'] >= df.get('up_limit_D', np.inf) * 0.995).astype(float)
            tactical_limit_up_deception = is_at_limit_up * tactical_vol_spike
            tactical_alluring_rally = np.maximum(tactical_high_level_dist, tactical_limit_up_deception)
            # 上下文层
            context_rally = normalize_score(df['pct_change_D'].clip(lower=0), df.index, window=p_context, ascending=True)
            context_inefficient_vol = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, window=p_context, ascending=False)
            context_vol_spike = normalize_score(df['volume_D'] / df.get(f'VOL_MA_{p_context}_D', df['volume_D']), df.index, window=p_context, ascending=True)
            context_high_level_dist = (context_rally * closing_weakness_score * context_inefficient_vol)**(1/3)
            context_limit_up_deception = is_at_limit_up * context_vol_spike
            context_alluring_rally = np.maximum(context_high_level_dist, context_limit_up_deception)
            # 融合
            alluring_rally_evidence = (tactical_alluring_rally * context_alluring_rally)**0.5
            # --- 证据链 2: 矛盾的博弈 (分层计算) ---
            tactical_profit = normalize_score(df.get('main_force_intraday_profit_D'), df.index, window=p_tactical, ascending=True)
            tactical_outflow = normalize_score(df.get('main_force_net_flow_consensus_D'), df.index, window=p_tactical, ascending=False)
            tactical_fomo = normalize_score(df.get('retail_net_flow_consensus_D'), df.index, window=p_tactical, ascending=True)
            tactical_contradiction = (tactical_profit * tactical_outflow * tactical_fomo)**(1/3)
            context_profit = normalize_score(df.get('main_force_intraday_profit_D'), df.index, window=p_context, ascending=True)
            context_outflow = normalize_score(df.get('main_force_net_flow_consensus_D'), df.index, window=p_context, ascending=False)
            context_fomo = normalize_score(df.get('retail_net_flow_consensus_D'), df.index, window=p_context, ascending=True)
            context_contradiction = (context_profit * context_outflow * context_fomo)**(1/3)
            contradiction_evidence = (tactical_contradiction * context_contradiction)**0.5
            # --- 证据链 4: 恶化的结构 (分层计算) ---
            tactical_chip_decay = normalize_score(df.get(f'SLOPE_{p_tactical}_concentration_90pct_D'), df.index, window=p_tactical, ascending=False)
            context_chip_decay = normalize_score(df.get(f'SLOPE_{p_context}_concentration_90pct_D'), df.index, window=p_context, ascending=False)
            chip_concentration_decay_score = (tactical_chip_decay * context_chip_decay)**0.5
            # --- 最终融合，生成“瞬时风险快照分” ---
            snapshot_score = (alluring_rally_evidence * contradiction_evidence * betrayal_evidence * chip_concentration_decay_score)**(1/4)
            # --- 对“风险关系”进行元分析 ---
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            icarus_scores_by_period[p_tactical] = period_score
        # --- 跨周期融合，生成最终信号 ---
        tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.1, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += icarus_scores_by_period.get(p_tactical, 0.0) * weight
        states['SCORE_RISK_ICARUS_FALL'] = final_fused_score.clip(0, 1).astype(np.float32)
        
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
        【V13.0 · 影响力归一化版】市场微观结构动态诊断引擎
        - 核心升维: 权力转移的“过程证据”现在基于“交易颗粒度影响力”(trade_granularity_impact)的动态变化。
                      该指标通过流通市值归一化，彻底消除了股价和市值的尺度干扰，使信号具备了跨市场的普适性。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        recent_reversal_context = self._get_atomic_score(df, 'SCORE_CONTEXT_RECENT_REVERSAL', 0.0)
        risk_suppression_factor = (1.0 - recent_reversal_context).clip(0, 1)
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        power_shift_up_scores = {}
        conviction_up_scores = {}
        power_shift_down_scores = {}
        conviction_down_scores = {}
        # 定义新的、基于“影响力”的颗粒度指标名称
        granularity_impact_metric = 'trade_granularity_impact_D'
        
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # --- 看涨信号计算 (分层) ---
            # 使用新的“影响力”指标
            # 权力转移(主力)
            tactical_granularity_up = normalize_score(df.get(f'SLOPE_{p_tactical}_{granularity_impact_metric}'), df.index, window=p_tactical, ascending=True)
            tactical_dominance_up = normalize_score(df.get(f'SLOPE_{p_tactical}_trade_concentration_index_D'), df.index, window=p_tactical, ascending=True)
            context_granularity_up = normalize_score(df.get(f'SLOPE_{p_context}_{granularity_impact_metric}'), df.index, window=p_context, ascending=True)
            context_dominance_up = normalize_score(df.get(f'SLOPE_{p_context}_trade_concentration_index_D'), df.index, window=p_context, ascending=True)
            granularity_holo_up, _ = calculate_holographic_dynamics(df, granularity_impact_metric, p_context)
            dominance_holo_up, _ = calculate_holographic_dynamics(df, 'trade_concentration_index_D', p_context)
            
            fused_power_shift_raw = (tactical_granularity_up * context_granularity_up * tactical_dominance_up * context_dominance_up)**0.25 * granularity_holo_up * dominance_holo_up
            snapshot_power_shift = fused_power_shift_raw * ma_health_score
            power_shift_up_scores[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_power_shift)
            # 主力信念加强
            tactical_conviction_up = normalize_score(df.get(f'SLOPE_{p_tactical}_main_force_conviction_ratio_D'), df.index, window=p_tactical, ascending=True)
            context_conviction_up = normalize_score(df.get(f'SLOPE_{p_context}_main_force_conviction_ratio_D'), df.index, window=p_context, ascending=True)
            conviction_holo_up, _ = calculate_holographic_dynamics(df, 'main_force_conviction_ratio_D', p_context)
            fused_conviction_raw = (tactical_conviction_up * context_conviction_up)**0.5 * conviction_holo_up
            snapshot_conviction = fused_conviction_raw * ma_health_score
            conviction_up_scores[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_conviction)
            # --- 看跌风险信号计算 (分层) ---
            # 使用新的“影响力”指标
            # 证据维度一：权力转移(散户) - 过程证据
            tactical_granularity_down = normalize_score(df.get(f'SLOPE_{p_tactical}_{granularity_impact_metric}'), df.index, window=p_tactical, ascending=False)
            tactical_dominance_down = normalize_score(df.get(f'SLOPE_{p_tactical}_trade_concentration_index_D'), df.index, window=p_tactical, ascending=False)
            context_granularity_down = normalize_score(df.get(f'SLOPE_{p_context}_{granularity_impact_metric}'), df.index, window=p_context, ascending=False)
            context_dominance_down = normalize_score(df.get(f'SLOPE_{p_context}_trade_concentration_index_D'), df.index, window=p_context, ascending=False)
            _, granularity_holo_down = calculate_holographic_dynamics(df, granularity_impact_metric, p_context)
            _, dominance_holo_down = calculate_holographic_dynamics(df, 'trade_concentration_index_D', p_context)
            
            fused_power_shift_process_evidence = (tactical_granularity_down * context_granularity_down * tactical_dominance_down * context_dominance_down)**0.25 * granularity_holo_down * dominance_holo_down
            # 证据维度二：主力日内T+0派发获利 - 战术结果证据
            tactical_profit_distribute = normalize_score(df.get(f'SLOPE_{p_tactical}_main_force_intraday_profit_D'), df.index, window=p_tactical, ascending=True)
            tactical_cost_battle_loss = normalize_score(df.get(f'SLOPE_{p_tactical}_market_cost_battle_D'), df.index, window=p_tactical, ascending=False)
            context_profit_distribute = normalize_score(df.get(f'SLOPE_{p_context}_main_force_intraday_profit_D'), df.index, window=p_context, ascending=True)
            context_cost_battle_loss = normalize_score(df.get(f'SLOPE_{p_context}_market_cost_battle_D'), df.index, window=p_context, ascending=False)
            fused_tactical_profit_taking_evidence = ((tactical_profit_distribute * context_profit_distribute)**0.5 * (tactical_cost_battle_loss * context_cost_battle_loss)**0.5)
            # 证据维度三：主力多日持仓派发获利 - 战略结果证据
            past_buy_cost = df.get('avg_cost_main_buy_D').shift(p_tactical)
            profit_margin = (df['close_D'] - past_buy_cost) / past_buy_cost.replace(0, np.nan)
            selling_action = -df.get('main_force_net_flow_consensus_D').clip(upper=0)
            tactical_strategic_margin = normalize_score(profit_margin, df.index, window=p_tactical, ascending=True)
            tactical_strategic_sell = normalize_score(selling_action, df.index, window=p_tactical, ascending=True)
            past_buy_cost_context = df.get('avg_cost_main_buy_D').shift(p_context)
            profit_margin_context = (df['close_D'] - past_buy_cost_context) / past_buy_cost_context.replace(0, np.nan)
            context_strategic_margin = normalize_score(profit_margin_context, df.index, window=p_context, ascending=True)
            context_strategic_sell = normalize_score(selling_action, df.index, window=p_context, ascending=True)
            fused_strategic_profit_taking_evidence = ((tactical_strategic_margin * context_strategic_margin)**0.5 * (tactical_strategic_sell * context_strategic_sell)**0.5)
            # 三位一体融合
            fused_power_shift_risk_raw = (fused_power_shift_process_evidence * fused_tactical_profit_taking_evidence * fused_strategic_profit_taking_evidence)**(1/3)
            snapshot_power_shift_risk = fused_power_shift_risk_raw * (1 - ma_health_score)
            power_shift_down_scores[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_power_shift_risk)
            # 主力信念瓦解 (逻辑不变)
            tactical_conviction_down = normalize_score(df.get(f'SLOPE_{p_tactical}_main_force_conviction_ratio_D'), df.index, window=p_tactical, ascending=False)
            context_conviction_down = normalize_score(df.get(f'SLOPE_{p_context}_main_force_conviction_ratio_D'), df.index, window=p_context, ascending=False)
            _, conviction_holo_down = calculate_holographic_dynamics(df, 'main_force_conviction_ratio_D', p_context)
            fused_conviction_risk_raw = (tactical_conviction_down * context_conviction_down)**0.5 * conviction_holo_down
            snapshot_conviction_risk = fused_conviction_risk_raw * (1 - ma_health_score)
            conviction_down_scores[p_tactical] = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_conviction_risk)
        # --- 跨周期融合 ---
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        def fuse_scores(score_dict):
            final_score = pd.Series(0.0, index=df.index)
            total_weight = sum(tf_weights.get(p, 0) for p in periods)
            if total_weight > 0:
                for p in periods:
                    weight = tf_weights.get(p, 0) / total_weight
                    final_score += score_dict.get(p, 0.0) * weight
            return final_score.clip(0, 1)
        final_power_shift_score = fuse_scores(power_shift_up_scores)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = final_power_shift_score.astype(np.float32)
        final_conviction_score = fuse_scores(conviction_up_scores)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = final_conviction_score.astype(np.float32)
        final_power_shift_risk = fuse_scores(power_shift_down_scores)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = (final_power_shift_risk * risk_suppression_factor).astype(np.float32)
        final_conviction_risk = fuse_scores(conviction_down_scores)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = (final_conviction_risk * risk_suppression_factor).astype(np.float32)
        return states

    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 分层印证版】亢奋加速风险诊断引擎
        - 核心升级: 引入“分层动态印证”框架。对构成亢奋风险的各项原子指标（乖离、成交量、波动率等）进行多时间维度的分层验证。
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        epsilon = 1e-9
        # 引入分层印证框架
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        euphoric_scores_by_period = {}
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # --- 亢奋风险因子 (分层计算) ---
            # 战术层
            tactical_bias = normalize_score(df[f'BIAS_{p_tactical}_D'].abs(), df.index, window=p_tactical, ascending=True) if f'BIAS_{p_tactical}_D' in df else pd.Series(0.5, index=df.index)
            tactical_vol_ratio = (df['volume_D'] / (df.get(f'VOL_MA_{p_tactical}_D', df['volume_D']) + epsilon)).fillna(1.0)
            tactical_vol_spike = normalize_score(tactical_vol_ratio, df.index, window=p_tactical, ascending=True)
            tactical_atr_ratio = (df['ATR_14_D'] / (df['close_D'] + epsilon)).fillna(0.0)
            tactical_volatility = normalize_score(tactical_atr_ratio, df.index, window=p_tactical, ascending=True)
            # 上下文层
            context_bias = normalize_score(df[f'BIAS_{p_context}_D'].abs(), df.index, window=p_context, ascending=True) if f'BIAS_{p_context}_D' in df else pd.Series(0.5, index=df.index)
            context_vol_ratio = (df['volume_D'] / (df.get(f'VOL_MA_{p_context}_D', df['volume_D']) + epsilon)).fillna(1.0)
            context_vol_spike = normalize_score(context_vol_ratio, df.index, window=p_context, ascending=True)
            context_atr_ratio = (df['ATR_14_D'] / (df['close_D'] + epsilon)).fillna(0.0)
            context_volatility = normalize_score(context_atr_ratio, df.index, window=p_context, ascending=True)
            # 融合
            bias_score = (tactical_bias * context_bias)**0.5
            volume_spike_score = (tactical_vol_spike * context_vol_spike)**0.5
            volatility_score = (tactical_volatility * context_volatility)**0.5
            # --- 静态因子 (无需分层) ---
            total_range = (df['high_D'] - df['low_D']).replace(0, epsilon)
            upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
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
            # --- 重新组装 ---
            raw_risk_factors = (bias_score * volume_spike_score * volatility_score * upthrust_score)**(1/4)
            raw_euphoric_risk_score = (raw_risk_factors * stretch_from_ma55_score * (1 - safe_launch_context_score))
            snapshot_score = raw_euphoric_risk_score * ma_health_score
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            euphoric_scores_by_period[p_tactical] = period_score
        # --- 跨周期融合 ---
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += euphoric_scores_by_period.get(p_tactical, 0.0) * weight
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_fused_score.clip(0, 1).astype(np.float32)
        
        return states

    def synthesize_post_peak_downturn_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V4.0 · 分层印证版】高位回落风险 (Post-Peak Downturn Risk) 诊断引擎
        - 核心升级: 引入“分层动态印证”框架。对构成回落风险的严重性因子（下跌幅度、成交量、破位深度）进行多时间维度的分层验证。
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'post_peak_downturn_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        high_position_threshold = get_param_value(p_risk.get('high_position_threshold'), 0.7)
        peak_echo_window = get_param_value(p_risk.get('peak_echo_window'), 5)
        # 引入分层印证框架
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        downturn_scores_by_period = {}
        ma_health_score = self._calculate_ma_health(df, p_conf, 55)
        # 静态因子
        ma55 = df.get('EMA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, np.nan)
        stretch_from_ma55_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        is_at_high_position = (stretch_from_ma55_score > high_position_threshold)
        recently_at_peak_context = is_at_high_position.rolling(window=peak_echo_window, min_periods=1).max().astype(float)
        is_falling_today = (df['pct_change_D'] < 0).astype(float)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # --- 严重性评分 (分层计算) ---
            fall_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            # 战术层
            tactical_fall_mag = normalize_score(fall_magnitude, df.index, window=p_tactical, ascending=True)
            tactical_vol_ratio = (df['volume_D'] / df.get(f'VOL_MA_{p_tactical}_D', df['volume_D'])).fillna(1.0)
            tactical_vol_spike = normalize_score(tactical_vol_ratio, df.index, window=p_tactical, ascending=True)
            ema_tactical = df.get(f'EMA_{p_tactical}_D', df['close_D'])
            tactical_breakdown_pct = ((ema_tactical - df['close_D']) / ema_tactical).clip(lower=0).fillna(0)
            tactical_break_ema = normalize_score(tactical_breakdown_pct, df.index, window=p_tactical, ascending=True)
            tactical_severity = (tactical_fall_mag * tactical_vol_spike * tactical_break_ema)**(1/3)
            # 上下文层
            context_fall_mag = normalize_score(fall_magnitude, df.index, window=p_context, ascending=True)
            context_vol_ratio = (df['volume_D'] / df.get(f'VOL_MA_{p_context}_D', df['volume_D'])).fillna(1.0)
            context_vol_spike = normalize_score(context_vol_ratio, df.index, window=p_context, ascending=True)
            ema_context = df.get(f'EMA_{p_context}_D', df['close_D'])
            context_breakdown_pct = ((ema_context - df['close_D']) / ema_context).clip(lower=0).fillna(0)
            context_break_ema = normalize_score(context_breakdown_pct, df.index, window=p_context, ascending=True)
            context_severity = (context_fall_mag * context_vol_spike * context_break_ema)**(1/3)
            # 融合
            severity_score = (tactical_severity * context_severity)**0.5
            # --- 重新组装 ---
            raw_downturn_risk_score = (recently_at_peak_context * is_falling_today * severity_score)
            snapshot_score = raw_downturn_risk_score * (1 - ma_health_score)
            period_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
            downturn_scores_by_period[p_tactical] = period_score
        # --- 跨周期融合 ---
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += downturn_scores_by_period.get(p_tactical, 0.0) * weight
        states['COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN'] = final_fused_score.clip(0, 1).astype(np.float32)
        
        return states

    def _perform_micro_behavior_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V2.0 · 阿瑞斯之怒协议版】微观行为专用的关系元分析核心引擎
        - 核心革命: 响应“重变化、轻状态”的哲学，从“状态 * (1 + 动态)”的乘法模型，升级为
                      “(状态*权重) + (速度*权重) + (加速度*权重)”的加法模型。
        - 核心目标: 即使静态分很低，只要动态（尤其是加速度）足够强，也能产生高分，真正捕捉“拐点”。
        """
        # 引入新的权重体系和加法融合模型
        # 从配置中获取新的加法模型权重
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的权重体系，直接作用于最终分数，而非杠杆
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4) # 赋予加速度最高权重
        # 核心参数
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        # 第一维度：状态分 (State Score) - 范围 [0, 1]
        state_score = snapshot_score.clip(0, 1)
        # 第二维度：速度分 (Velocity Score) - 范围 [-1, 1]
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 第三维度：加速度分 (Acceleration Score) - 范围 [-1, 1]
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # 终极融合：从乘法调制升级为加法赋权
        # 旧的乘法模型: dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        # 旧的乘法模型: final_score = (state_score * dynamic_leverage).clip(0, 1)
        # 新的加法模型:
        final_score = (
            state_score * w_state +
            velocity_score * w_velocity +
            acceleration_score * w_acceleration
        ).clip(0, 1) # clip确保分数在[0, 1]范围内
        return final_score.astype(np.float32)

    def _calculate_ma_health(self, df: pd.DataFrame, params: dict, norm_window: int) -> pd.Series:
        """
        【V1.0 · 新增】“赫尔墨斯的商神杖”四维均线健康度评估引擎
        - 核心职责: 严格按照 ma_health_fusion_weights 配置，计算并融合均线健康度的四大维度。
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
            return pd.Series(0.5, index=df.index, dtype=np.float32)

        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)

        slope_cols = [f'SLOPE_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in slope_cols):
            slope_values = np.stack([df[col].values for col in slope_cols], axis=0)
            slope_health = np.mean(normalize_score(pd.Series(slope_values.flatten()), df.index, norm_window).values.reshape(slope_values.shape), axis=0)
        else:
            slope_health = np.full(len(df.index), 0.5)

        accel_cols = [f'ACCEL_5_{col}' for col in ma_cols]
        if all(col in df.columns for col in accel_cols):
            accel_values = np.stack([df[col].values for col in accel_cols], axis=0)
            accel_health = np.mean(normalize_score(pd.Series(accel_values.flatten()), df.index, norm_window).values.reshape(accel_values.shape), axis=0)
        else:
            accel_health = np.full(len(df.index), 0.5)

        ma_std = np.std(ma_values / df['close_D'].values[:, np.newaxis].T, axis=0)
        relational_health = 1.0 - normalize_score(pd.Series(ma_std, index=df.index), df.index, norm_window, ascending=True)

        scores = np.stack([alignment_health, slope_health, accel_health, relational_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()

        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)




