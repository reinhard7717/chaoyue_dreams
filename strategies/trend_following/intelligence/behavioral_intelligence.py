# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, create_persistent_state, normalize_score, calculate_context_scores, calculate_holographic_dynamics

class BehavioralIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        # K线形态识别器可能需要在这里初始化或传入
        self.pattern_recognizer = strategy_instance.pattern_recognizer
        self.tactic_engine = TacticEngine(strategy_instance)

    def run_behavioral_analysis_command(self) -> Dict[str, pd.Series]:
        """
        【V3.4 · 职责净化版】行为情报模块总指挥
        - 核心重构: 移除了对 _diagnose_archangel_top_reversal 的调用。
                      “天使长”作为顶层认知信号，其诊断职责已正式移交认知情报模块。
        """
        df = self.strategy.df_indicators
        all_behavioral_states = {}
        internal_atomic_signals = self._generate_all_atomic_signals(df)
        if internal_atomic_signals:
            self.strategy.atomic_states.update(internal_atomic_signals)
            all_behavioral_states.update(internal_atomic_signals)
        ultimate_behavioral_states = self.diagnose_ultimate_behavioral_signals(df, atomic_signals=internal_atomic_signals)
        if ultimate_behavioral_states:
            all_behavioral_states.update(ultimate_behavioral_states)
        return all_behavioral_states

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        【V26.0 · 宙斯之雷版】
        - 核心革命: 废除对快照分的元分析，改为调用“最高神谕融合引擎”，
                      直接对多个、已完全进化的动态原子信号进行最终裁决。
        """
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        # 步骤一：获取“宙斯之雷”引擎的专属参数
        p_fusion = get_param_value(p_conf.get('supreme_fusion_params'), {})
        
        # 步骤二：调用“宙斯之雷”引擎，传入所有相关的原子信号
        # 我们将所有潜在的底部信号都交给引擎去裁决
        bottom_formation_score = self._supreme_fusion_engine(
            df=df,
            signals_to_fuse=atomic_signals,
            params=p_fusion
        )
        
        # 将“宙斯之雷”的最终裁决结果作为唯一的、权威的通用底部形态信号
        self.strategy.atomic_states['SCORE_UNIVERSAL_BOTTOM_PATTERN'] = bottom_formation_score.astype(np.float32)
        
        # 后续逻辑保持不变，它们现在消费的是由“宙斯之雷”锻造的、最高规格的神谕信号
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        reversal_echo_window = get_param_value(p_conf.get('reversal_echo_window'), 3)
        recent_reversal_context = bottom_formation_score.rolling(window=reversal_echo_window, min_periods=1).max()
        self.strategy.atomic_states['SCORE_CONTEXT_RECENT_REVERSAL'] = recent_reversal_context.astype(np.float32)
        
        price_s_bull, price_s_bear, price_d_intensity = self._calculate_price_health(df, norm_window, max(1, norm_window // 5), periods)
        vol_s_bull, vol_s_bear, vol_d_intensity = self._calculate_volume_health(df, norm_window, max(1, norm_window // 5), periods)
        kline_s_bull, kline_s_bear, kline_d_intensity = self._calculate_kline_pattern_health(df, atomic_signals, norm_window, max(1, norm_window // 5), periods)
        
        overall_health = {}
        pillar_weights = get_param_value(p_conf.get('pillar_weights'), {'price': 0.4, 'volume': 0.3, 'kline': 0.3})
        dim_weights_array = np.array([pillar_weights['price'], pillar_weights['volume'], pillar_weights['kline']])
        for health_type, health_sources in [
            ('s_bull', [price_s_bull, vol_s_bull, kline_s_bull]),
            ('s_bear', [price_s_bear, vol_s_bear, kline_s_bear]),
            ('d_intensity', [price_d_intensity, vol_d_intensity, kline_d_intensity])
        ]:
            overall_health[health_type] = {}
            for p in periods:
                if not all(health_sources) or not all(p in h for h in health_sources):
                    fused_values = np.full(len(df.index), 0.5)
                else:
                    stacked_values = np.stack([
                        health_sources[0][p].values, health_sources[1][p].values, health_sources[2][p].values
                    ], axis=0)
                    fused_values = np.prod(stacked_values ** dim_weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        
        self.strategy.atomic_states['__BEHAVIOR_overall_health'] = overall_health
        
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="BEHAVIOR"
        )
        states.update(ultimate_signals)
        return states

    # ==============================================================================
    # 以下为新增的原子信号中心和降级的原子诊断引擎
    # ==============================================================================

    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.3 · 双引擎版】原子信号中心，负责生产所有基础行为信号。"""
        atomic_signals = {}
        params = self.strategy.params
        
        atomic_signals.update(self._diagnose_atomic_bottom_formation(df))
        
        # 架构升级：同时运行两个独立的、针对不同场景的反转引擎
        epic_reversal_states = self._diagnose_atomic_rebound_reversal(df)
        continuation_reversal_states = self._diagnose_atomic_continuation_reversal(df)
        
        epic_score = epic_reversal_states.get('SCORE_ATOMIC_REBOUND_REVERSAL', pd.Series(0.0, index=df.index))
        continuation_score = continuation_reversal_states.get('SCORE_ATOMIC_CONTINUATION_REVERSAL', pd.Series(0.0, index=df.index))
        
        # 最终的探底回升信号是两种反转模式中的最强者
        final_rebound_score = np.maximum(epic_score, continuation_score)
        atomic_signals['SCORE_ATOMIC_REBOUND_REVERSAL'] = final_rebound_score.astype(np.float32)
        # 将延续性反转的独立分数也存入，以供调试
        atomic_signals.update(continuation_reversal_states)
        
        atomic_signals.update(self._diagnose_kline_patterns(df))
        atomic_signals.update(self._diagnose_advanced_atomic_signals(df))
        atomic_signals.update(self._diagnose_board_patterns(df))
        atomic_signals.update(self._diagnose_price_volume_atomics(df))
        atomic_signals.update(self._diagnose_volume_price_dynamics(df, params))
        
        upthrust_score = self._diagnose_upthrust_distribution(df, params)
        atomic_signals[upthrust_score.name] = upthrust_score
        
        ma_breakdown_score = self._diagnose_ma_breakdown(df, params)
        atomic_signals[ma_breakdown_score.name] = ma_breakdown_score
        
        return atomic_signals

    def _calculate_price_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V4.0 · 全息动态升级版】计算价格维度的三维健康度
        - 核心升级: 将 d_intensity 的计算方式从旧的绝对值模式升级为全新的“全息动态”模式，
                      使其能更灵敏地捕捉价格动能的真实变化。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        bbp = df.get('BBP_21_2.0_D', pd.Series(0.5, index=df.index)).fillna(0.5).clip(0, 1)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.5)

        is_positive_day = df['pct_change_D'] > 0
        bullish_score_on_up_day = (bbp * close_position_in_range)**0.5
        reversal_potential_score = close_position_in_range.where(df['pct_change_D'] < 0, 0)
        static_bull_score = pd.Series(
            np.where(is_positive_day, bullish_score_on_up_day, reversal_potential_score),
            index=df.index
        )

        static_bear_score = ((1.0 - bbp) * (1.0 - close_position_in_range))**0.5

        # 使用全新的全息动态引擎计算动态强度分
        price_holo_bull, price_holo_bear = calculate_holographic_dynamics(df, 'close_D', norm_window)
        # 动态强度分现在同时考虑看涨和看跌的动态变化
        unified_d_intensity = (price_holo_bull + price_holo_bear) / 2.0

        for p in periods:
            s_bull[p] = static_bull_score.astype(np.float32)
            s_bear[p] = static_bear_score.astype(np.float32)
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_volume_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V7.0 · 全息动态升级版】计算成交量维度的三维健康度
        - 核心升级: 将 d_intensity 的计算方式从旧的绝对值模式升级为全新的“全息动态”模式，
                      使其能更灵敏地捕捉成交量能的真实变化。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}

        if 'pct_change_D' not in df.columns or 'volume_D' not in df.columns:
            for p in periods:
                s_bull[p] = s_bear[p] = pd.Series(0.5, index=df.index)
                d_intensity[p] = pd.Series(0.5, index=df.index)
            return s_bull, s_bear, d_intensity

        volume_increase_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=True)
        price_stagnation_score = 1 - normalize_score(df['pct_change_D'].abs(), df.index, norm_window, ascending=True)
        stagnation_path_score = volume_increase_score * price_stagnation_score
        price_drop_score = normalize_score(df['pct_change_D'].clip(upper=0).abs(), df.index, norm_window, ascending=True)
        breakdown_path_score = (price_drop_score * volume_increase_score).where(df['pct_change_D'] < 0, 0)
        static_bear_score = np.maximum(stagnation_path_score, breakdown_path_score)
        
        price_increase_score = normalize_score(df['pct_change_D'].clip(lower=0), df.index, norm_window, ascending=True)
        yang_score = (price_increase_score * volume_increase_score)
        
        liquidity_drain_risk = self.strategy.atomic_states.get('SCORE_RISK_LIQUIDITY_DRAIN', pd.Series(0.0, index=df.index))
        yin_score = ((1.0 - static_bear_score) * (1.0 - liquidity_drain_risk))
        
        shrinking_volume_score = 1.0 - volume_increase_score
        selling_exhaustion_score = (shrinking_volume_score * price_drop_score)

        exhaustion_reversal_score = self.strategy.atomic_states.get('SCORE_BULLISH_EXHAUSTION_REVERSAL', pd.Series(0.0, index=df.index))
        
        is_positive_day = df['pct_change_D'] > 0
        bullish_score_on_down_day = np.maximum.reduce([
            yin_score, 
            selling_exhaustion_score,
            exhaustion_reversal_score
        ])
        static_bull_score_np = np.where(is_positive_day, yang_score, bullish_score_on_down_day)
        static_bull_score = pd.Series(static_bull_score_np, index=df.index)

        # 使用全新的全息动态引擎计算动态强度分
        vol_holo_bull, vol_holo_bear = calculate_holographic_dynamics(df, 'volume_D', norm_window)
        # 动态强度分现在同时考虑看涨和看跌的动态变化
        unified_d_intensity = (vol_holo_bull + vol_holo_bear) / 2.0

        for p in periods:
            s_bull[p] = static_bull_score.astype(np.float32)
            s_bear[p] = static_bear_score.astype(np.float32)
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_kline_pattern_health(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], norm_window: int, min_periods: int, periods: list) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.5 · 动态分统一版】计算K线形态维度的三维健康度
        - 核心重构: 废除 d_bull 和 d_bear，统一返回中性的“动态强度分” d_intensity。
        """
        # 更新方法签名和初始化
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        strong_close = normalize_score(atomic_signals.get('SCORE_PRICE_POSITION_IN_RANGE', pd.Series(0.5, index=df.index)), df.index, norm_window, True, min_periods)
        gap_support = normalize_score(atomic_signals.get('SCORE_GAP_SUPPORT_ACTIVE', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        earth_heaven = normalize_score(atomic_signals.get('SCORE_BOARD_EARTH_HEAVEN', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        gentle_rise_raw = df['pct_change_D'].clip(0, 0.03) / 0.03
        gentle_rise = normalize_score(gentle_rise_raw, df.index, norm_window, True, min_periods)
        static_bull_score = pd.Series(np.maximum.reduce([strong_close.values, gap_support.values, earth_heaven.values, gentle_rise.values]), index=df.index).astype(np.float32)

        weak_close = 1.0 - strong_close
        upthrust = normalize_score(atomic_signals.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        heaven_earth = normalize_score(atomic_signals.get('SCORE_BOARD_HEAVEN_EARTH', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        sharp_drop = normalize_score(atomic_signals.get('SCORE_KLINE_SHARP_DROP', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        static_bear_score = pd.Series(np.maximum.reduce([weak_close.values, upthrust.values, heaven_earth.values, sharp_drop.values]), index=df.index).astype(np.float32)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            # 计算统一的、中性的动态强度分 d_intensity
            # K线形态的动态分衡量的是“静态分自身的变化强度”
            bull_slope_strength = static_bull_score.diff(p).fillna(0).abs()
            bear_slope_strength = static_bear_score.diff(p).fillna(0).abs()
            # 取两者中变化更剧烈的作为动态强度
            intensity_slope = np.maximum(bull_slope_strength, bear_slope_strength)
            d_intensity[p] = normalize_score(intensity_slope, df.index, norm_window, ascending=True)
            
        # 返回三元组
        return s_bull, s_bear, d_intensity

    # 以下方法被降级为私有，作为原子信号的生产者
    def _diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        p_gap = p.get('gap_support_params', {})
        if get_param_value(p_gap.get('enabled'), True):
            persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
            gap_up_event = df['low_D'] > df['high_D'].shift(1)
            gap_high = df['high_D'].shift(1).where(gap_up_event).ffill()
            price_fills_gap = df['close_D'] < gap_high
            gap_support_state = create_persistent_state(df=df, entry_event_series=gap_up_event, persistence_days=persistence_days, break_condition_series=price_fills_gap, state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE')
            support_distance = (df['low_D'] - gap_high).clip(lower=0)
            normalization_base = (df['close_D'] * 0.1).replace(0, np.nan)
            support_strength_score = (support_distance / normalization_base).clip(0, 1).fillna(0)
            states['SCORE_GAP_SUPPORT_ACTIVE'] = (support_strength_score * gap_support_state).astype(np.float32)
        p_atomic = p.get('atomic_behavior_params', {})
        if get_param_value(p_atomic.get('enabled'), True):
            if 'pct_change_D' in df.columns:
                norm_window = get_param_value(p_atomic.get('norm_window'), 120)
                min_periods = max(1, norm_window // 5)
                drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
                sharp_drop_score = drop_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
                states['SCORE_KLINE_SHARP_DROP'] = sharp_drop_score.astype(np.float32)
        return states

    def _diagnose_advanced_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        states = {}
        p = get_params_block(self.strategy, 'advanced_atomic_params', {}) 
        if not get_param_value(p.get('enabled'), True): return states
        price_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        close_position_in_range = ((df['close_D'] - df['low_D']) / price_range).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RANGE'] = close_position_in_range.astype(np.float32)
        is_up_day = df['pct_change_D'] > 0
        is_down_day = df['pct_change_D'] < 0
        up_streak = (is_up_day.groupby((is_up_day != is_up_day.shift()).cumsum()).cumcount() + 1) * is_up_day
        down_streak = (is_down_day.groupby((is_down_day != is_down_day.shift()).cumsum()).cumcount() + 1) * is_down_day
        states['COUNT_CONSECUTIVE_UP_STREAK'] = up_streak.astype(np.int16)
        states['COUNT_CONSECUTIVE_DOWN_STREAK'] = down_streak.astype(np.int16)
        return states

    def _diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        states = {}
        p = get_params_block(self.strategy, 'board_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = get_param_value(p.get('price_buffer'), 0.005)
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        theoretical_max_range = (limit_up_price - limit_down_price).replace(0, np.nan)
        strength_score = (day_range / theoretical_max_range).clip(0, 1).fillna(0)
        low_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['low_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        close_near_limit_up_score = ((df['close_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        high_near_limit_up_score = ((df['high_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        close_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['close_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        states['SCORE_BOARD_EARTH_HEAVEN'] = (strength_score * low_near_limit_down_score * close_near_limit_up_score).astype(np.float32)
        states['SCORE_BOARD_HEAVEN_EARTH'] = (strength_score * high_near_limit_up_score * close_near_limit_down_score).astype(np.float32)
        return states

    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.0 · 关系元分析版】上冲派发风险诊断引擎
        - 核心升级: 采用“关系元分析”范式，寻找“派发风险关系”的加速恶化拐点。
        """
        p = get_params_block(self.strategy, 'upthrust_distribution_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        
        # 第一维度：计算“瞬时关系快照分”
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        ma_col = f'EMA_{overextension_ma_period}_D'
        if not all(col in df.columns for col in ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col]):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        norm_window = get_param_value(p.get('norm_window'), 55)
        min_periods = max(1, norm_window // 5)
        
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        overextension_score = overextension_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        volume_score = df['volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / total_range).fillna(0.5)
        weak_close_score = 1 - close_position_in_range
        
        # --- 融合三大支柱，得到单一的“瞬时风险关系分” ---
        snapshot_score = (overextension_score * volume_score * weak_close_score).astype(np.float32)
        
        # 第二维度：调用核心引擎，分析“风险关系”的拐点
        final_signal_dict = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_RISK_UPTHRUST_DISTRIBUTION'
        )
        
        # 返回 Series，保持原方法签名
        return final_signal_dict.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION'))

    def _diagnose_ma_breakdown(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V2.0 · 关系元分析版】均线破位风险诊断引擎
        - 核心升级: 采用“关系元分析”范式，捕捉“破位关系”的加速恶化拐点。
        """
        p = get_params_block(self.strategy, 'structure_breakdown_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN')
        
        breakdown_ma_period = get_param_value(p.get('breakdown_ma_period'), 21)
        ma_col = f'EMA_{breakdown_ma_period}_D'
        if not all(col in df.columns for col in ['close_D', 'volume_D', ma_col]):
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN')
            
        norm_window = get_param_value(p.get('norm_window'), 55)
        min_periods = max(1, norm_window // 5)

        # 第一维度：计算“瞬时破位关系快照分”
        # 破位深度分
        breakdown_depth = ((df[ma_col] - df['close_D']) / df[ma_col].replace(0, np.nan)).fillna(0)
        breakdown_depth_score = breakdown_depth.where(df['close_D'] < df[ma_col], 0).clip(0)
        normalized_depth_score = normalize_score(breakdown_depth_score, df.index, norm_window, ascending=True)
        
        # 成交量放大分
        volume_increase_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=True)
        
        # 融合得到快照分
        snapshot_score = (normalized_depth_score * volume_increase_score).astype(np.float32)

        # 第二维度：调用核心引擎，分析“风险关系”的拐点
        final_signal_dict = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_BEHAVIOR_MA_BREAKDOWN'
        )
        
        return final_signal_dict.get('SCORE_BEHAVIOR_MA_BREAKDOWN', pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN'))

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析版】
        - 核心升级: 对“VPA滞涨风险”信号采用“关系元分析”范式重构。
        """
        states = {}
        required_cols = ['volume_D', 'VOL_MA_21_D', 'pct_change_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D', 'VPA_EFFICIENCY_D', 'SLOPE_5_VPA_EFFICIENCY_D']
        if any(col not in df.columns for col in required_cols): return states
        
        p_vpa = params.get('vpa_dynamics_params', {})
        norm_window = get_param_value(p_vpa.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        
        # 对 VPA 滞涨风险进行关系元分析
        # 第一维度：计算“瞬时滞涨关系快照分”
        volume_ratio = (df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)).fillna(1.0)
        huge_volume_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        price_stagnant_score = 1 - normalize_score(df['pct_change_D'].abs(), df.index, norm_window, ascending=True)
        stagnation_snapshot_score = (huge_volume_score * price_stagnant_score).astype(np.float32)
        
        # 第二维度：调用核心引擎
        stagnation_risk_states = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=stagnation_snapshot_score,
            signal_name='SCORE_RISK_VPA_STAGNATION'
        )
        states.update(stagnation_risk_states)

        # 保持其他信号的原子性
        efficiency_decline_magnitude = df['SLOPE_5_VPA_EFFICIENCY_D'].where(df['SLOPE_5_VPA_EFFICIENCY_D'] < 0, 0).abs()
        efficiency_decline_score = efficiency_decline_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
        states['SCORE_RISK_VPA_EFFICIENCY_DECLINING'] = efficiency_decline_score.astype(np.float32)
        
        volume_accel_magnitude = df['ACCEL_5_volume_D'].where(df['ACCEL_5_volume_D'] > 0, 0)
        volume_accelerating_score = volume_accel_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
        states['SCORE_RISK_VPA_VOLUME_ACCELERATING'] = volume_accelerating_score.astype(np.float32)
        
        return states

    def _diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 关系元分析版】
        - 核心升级: 对“流动性枯竭风险”信号采用“关系元分析”范式重构。
        """
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states
        norm_window = get_param_value(p.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        
        lookback_period = get_param_value(p.get('range_lookback'), 20)
        rolling_high = df['high_D'].rolling(window=lookback_period).max()
        rolling_low = df['low_D'].rolling(window=lookback_period).min()
        price_range = (rolling_high - rolling_low).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - rolling_low) / price_range).clip(0, 1).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RECENT_RANGE'] = close_position_in_range.astype(np.float32)
        
        vol_ma_col = 'VOL_MA_21_D'
        if vol_ma_col in df.columns and 'pct_change_D' in df.columns:
            drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            price_drop_score = drop_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
            volume_ratio = (df['volume_D'] / df[vol_ma_col].replace(0, np.nan)).fillna(1.0)
            volume_shrink_score = (1 - volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
            states['SCORE_VOL_WEAKENING_DROP'] = (price_drop_score * volume_shrink_score).astype(np.float32)

        # 对“流动性枯竭风险”进行关系元分析重构
        p_drain = p.get('liquidity_drain_params', {})
        drain_window = get_param_value(p_drain.get('window'), 20)
        
        # 第一维度：计算“瞬时流动性枯竭快照分”
        # 价格下跌幅度分
        price_drop_magnitude = df['pct_change_D'].clip(upper=0).abs()
        price_drop_score = normalize_score(price_drop_magnitude, df.index, drain_window, ascending=True)
        # 成交量萎缩幅度分
        volume_shrink_score = normalize_score(df['volume_D'], df.index, drain_window, ascending=False)
        # 融合得到快照分
        drain_snapshot_score = (price_drop_score * volume_shrink_score).astype(np.float32)

        # 第二维度：调用核心引擎
        liquidity_drain_states = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=drain_snapshot_score,
            signal_name='SCORE_RISK_LIQUIDITY_DRAIN'
        )
        states.update(liquidity_drain_states)

        p_rev = p.get('exhaustion_reversal_params', {})
        rev_window = get_param_value(p_rev.get('window'), 5)
        
        is_stabilizing = (df['low_D'] >= df['low_D'].rolling(rev_window).min()).astype(int)
        price_volatility = df.get('ATR_14_D', pd.Series(df['high_D'] - df['low_D'], index=df.index))
        stabilization_score = is_stabilizing * normalize_score(price_volatility, df.index, norm_window, ascending=False)
        volume_dry_up_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=False)
        context_score = 1 - close_position_in_range
        snapshot_score = (stabilization_score * volume_dry_up_score * context_score)
        
        exhaustion_reversal_states = self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_BULLISH_EXHAUSTION_REVERSAL'
        )
        states.update(exhaustion_reversal_states)
        
        return states

    def _diagnose_atomic_bottom_formation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 · 纯净输出版】原子级“底部形态”诊断引擎
        - 核心升级: 只输出最终的元分析信号。
        """
        # 第一维度：计算“瞬时关系快照分”
        ma55 = df.get('EMA_55_D', df['close_D'])
        distance_from_ma55 = (df['close_D'] - ma55) / ma55
        lifeline_proximity_score = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2)
        
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        was_rsi_oversold = (rsi.rolling(window=10).min() < 35).astype(float)
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        pessimism_exhaustion_score = np.maximum(was_rsi_oversold, deep_bottom_context_score)

        vol_compression_score = normalize_score(df.get('BBW_21_2.0_D'), df.index, 60, ascending=False)

        snapshot_score = pd.Series(
            (lifeline_proximity_score * pessimism_exhaustion_score * vol_compression_score),
            index=df.index
        ).astype(np.float32)

        # 只返回最终信号字典
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_BOTTOM_FORMATION'
        )

    def _diagnose_atomic_rebound_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.2 · 纯净输出版】原子级“史诗探底回升”诊断引擎
        - 核心升级: 只输出最终的元分析信号。
        """
        # 第一维度：计算“瞬时关系快照分”
        p_rebound = get_params_block(self.strategy, 'panic_selling_setup_params', {})
        despair_context_score = self._calculate_despair_context_score(df, p_rebound)
        structural_test_score = self.tactic_engine.calculate_structural_test_score(df, p_rebound)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.0)
        is_recovering_today = (df['pct_change_D'] > -0.01).astype(float)
        confirmation_score = (close_position_in_range * is_recovering_today)

        snapshot_score = (despair_context_score * structural_test_score * confirmation_score).astype(np.float32)
        
        # 只返回最终信号字典
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_REBOUND_REVERSAL'
        )

    def _calculate_despair_context_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.0 · 新增/移植】“冥河之渡”多维绝望背景诊断引擎
        - 来源: 从 behavioral_intelligence 完美移植而来，作为“最终审判”计划的一部分。
        - 核心职责: 为本模块提供与行为引擎完全一致的、最高规格的绝望背景计算能力。
        """
        # --- 步骤 1: 获取参数 ---
        despair_periods = get_param_value(params.get('despair_periods'), {'short': (21, 5), 'mid': (60, 21), 'long': (250, 60)})
        despair_weights = get_param_value(params.get('despair_weights'), {'short': 0.2, 'mid': 0.3, 'long': 0.5})
        
        period_scores = []
        period_weight_values = []

        # --- 步骤 2: 遍历所有绝望周期，独立计算分数 ---
        for name, (drawdown_period, roc_period) in despair_periods.items():
            # 2.1 计算该周期的“坠落深度”
            rolling_peak = df['high_D'].rolling(window=drawdown_period, min_periods=max(1, drawdown_period//2)).max()
            drawdown_from_peak = (rolling_peak - df['close_D']) / rolling_peak.replace(0, np.nan)
            magnitude_score = normalize_score(drawdown_from_peak.clip(lower=0), df.index, window=drawdown_period, ascending=True)
            
            # 2.2 计算该周期的“坠落速度”
            price_roc = df['close_D'].pct_change(roc_period)
            velocity_score = normalize_score(price_roc, df.index, window=drawdown_period, ascending=False)
            
            # 2.3 融合得到该周期的绝望分数
            period_despair_score = (magnitude_score * velocity_score)**0.5
            
            period_scores.append(period_despair_score.values)
            period_weight_values.append(despair_weights.get(name, 0.0))

        # --- 步骤 3: 对所有周期的绝望分数进行加权几何平均 ---
        if not period_scores:
            return pd.Series(0.0, index=df.index)

        weights_array = np.array(period_weight_values)
        total_weights = weights_array.sum()
        if total_weights > 0:
            weights_array /= total_weights
        else:
            weights_array = np.full_like(weights_array, 1.0 / len(weights_array))

        stacked_scores = np.stack(period_scores, axis=0)
        
        final_score_values = np.prod(stacked_scores ** weights_array[:, np.newaxis], axis=0)
        
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _diagnose_atomic_continuation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 关系元分析版】原子级“延续性反转”诊断引擎
        - 核心升级: 采用“关系元分析”范式，寻找“延续性反转关系”的向上拐点。
        """
        # 第一维度：计算“瞬时关系快照分”
        p_continuation = get_params_block(self.strategy, 'continuation_reversal_params', {})
        ma_periods = get_param_value(p_continuation.get('ma_periods'), [5, 13, 21, 55])
        uptrending_ma_count = pd.Series(0, index=df.index)
        for p in ma_periods:
            ma_col = f'EMA_{p}_D'
            if ma_col in df:
                uptrending_ma_count += (df[ma_col] > df[ma_col].shift(1)).astype(int)
        trend_alignment_score = uptrending_ma_count / len(ma_periods)
        structural_test_score = self.tactic_engine.calculate_structural_test_score(df, p_continuation)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.0)
        confirmation_score = close_position_in_range

        # --- 融合三大支柱，得到单一的“瞬时关系分” ---
        snapshot_score = (trend_alignment_score * structural_test_score * confirmation_score).astype(np.float32)
        
        # 第二维度：调用核心引擎，分析“关系”的拐点
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_CONTINUATION_REVERSAL'
        )

    def _perform_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series, signal_name: str) -> Dict[str, pd.Series]:
        """
        【V2.0 · 赫拉织布机V2版】关系元分析核心引擎
        - 核心升级: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        - 新核心逻辑:
          1. 状态分(State): 瞬时关系快照分，是价值基石。
          2. 速度分(Velocity): 关系分趋势，归一化到[-1, 1]。
          3. 加速度分(Acceleration): 关系分趋势的趋势，归一化到[-1, 1]。
          4. 最终分 = 状态分 * (1 + 速度分*w_vel + 加速度分*w_accel)，动态决定最终价值。
        """
        states = {}
        # [代码新增] 从配置中获取新的动态杠杆权重
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.6)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)

        # --- 从ProcessIntelligence借鉴的核心参数 ---
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0

        # 第一维度：状态分 (State Score)
        state_score = snapshot_score.clip(0, 1)

        # 第二维度：速度分 (Velocity Score)
        relationship_trend = snapshot_score.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 第三维度：加速度分 (Acceleration Score)
        relationship_accel = relationship_trend.diff(meta_window).fillna(0)
        acceleration_score = normalize_to_bipolar(
            series=relationship_accel, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )

        # 终极融合：动态价值调制
        # 计算动态杠杆
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        
        # 应用公式：最终分 = 状态分 * 动态杠杆
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
        states[signal_name] = final_score.astype(np.float32)
        return states

    def _supreme_fusion_engine(self, df: pd.DataFrame, signals_to_fuse: Dict[str, pd.Series], params: Dict) -> pd.Series:
        """
        【V1.0 · 新增】最高神谕融合引擎 (宙斯之雷)
        - 核心职责: 对多个动态原子信号进行加权协同融合。
        """
        fusion_weights = get_param_value(params.get('fusion_weights'), {})
        synergy_bonus_factor = get_param_value(params.get('synergy_bonus_factor'), 0.5)

        valid_signals = []
        weights = []
        
        for name, weight in fusion_weights.items():
            signal_name_full = f'SCORE_ATOMIC_{name}'
            if signal_name_full in signals_to_fuse and weight > 0:
                # 使用 .values 确保 numpy 操作的性能和对齐
                valid_signals.append(signals_to_fuse[signal_name_full].values)
                weights.append(weight)

        if not valid_signals:
            return pd.Series(0.0, index=df.index, dtype=np.float32)

        # --- 基础融合：加权几何平均 ---
        stacked_signals = np.stack(valid_signals, axis=0)
        weights_array = np.array(weights)
        # 归一化权重
        total_weight = weights_array.sum()
        if total_weight > 0:
            normalized_weights = weights_array / total_weight
        else:
            normalized_weights = np.full_like(weights_array, 1.0 / len(weights_array))
        
        # 为避免 log(0) 错误，给信号值增加一个极小量
        safe_signals = np.maximum(stacked_signals, 1e-9)
        log_signals = np.log(safe_signals)
        weighted_log_sum = np.sum(log_signals * normalized_weights[:, np.newaxis], axis=0)
        base_fusion_score = np.exp(weighted_log_sum)

        # --- 协同奖励：当多个信号同时活跃时给予奖励 ---
        # 这里我们简化为取最强的两个信号的乘积作为协同基础
        if stacked_signals.shape[0] >= 2:
            # 沿信号轴排序，取最后两个（最大的）
            sorted_signals = np.sort(stacked_signals, axis=0)
            synergy_base = sorted_signals[-1] * sorted_signals[-2]
            synergy_bonus = (synergy_base**0.5) * synergy_bonus_factor
        else:
            synergy_bonus = 0.0
            
        # --- 最终融合 ---
        final_score = (base_fusion_score * (1 + synergy_bonus)).clip(0, 1)
        
        return pd.Series(final_score, index=df.index, dtype=np.float32)







