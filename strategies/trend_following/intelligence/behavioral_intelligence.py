# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
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
        【V24.0 · 圣杯契约版】
        - 核心革命: 不再读取本地的、重复的合成参数，而是从最高指挥部获取唯一的“圣杯”配置
                      (`ultimate_signal_synthesis_params`)，并将其传递给中央合成引擎。
        """
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 获取中央“圣杯”配置
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        grinding_bottom_score = atomic_signals.get('SCORE_ATOMIC_BOTTOM_FORMATION', pd.Series(0.0, index=df.index))
        rebound_bottom_score = atomic_signals.get('SCORE_ATOMIC_REBOUND_REVERSAL', pd.Series(0.0, index=df.index))
        bottom_formation_score = np.maximum(grinding_bottom_score, rebound_bottom_score)
        self.strategy.atomic_states['SCORE_UNIVERSAL_BOTTOM_PATTERN'] = bottom_formation_score.astype(np.float32)
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
        # 传入唯一的“圣杯”配置
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
        【V2.0 · 宙斯版】上冲派发风险诊断引擎
        - 核心革命: 1. 彻底废除僵化的 `upper_shadow_ratio_min` 教条，不再要求上影线达到特定比例。
                      2. 逻辑回归本质：风险 = 高位环境(overextension) * 巨大努力(volume) * 糟糕结果(weak_close)。
        - 收益: 极大提升了对各类顶部派发形态（无论上影线长短）的识别能力，稳健性与实战性飙升。
        """
        p = get_params_block(self.strategy, 'upthrust_distribution_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        ma_col = f'EMA_{overextension_ma_period}_D'

        if not all(col in df.columns for col in ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col]):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
            
        norm_window = get_param_value(p.get('norm_window'), 55)
        min_periods = max(1, norm_window // 5)
        
        # 支柱一: 高位环境 (Overextension) - 价格已远离均线，处于风险区域
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        overextension_score = overextension_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        
        # 支柱二: 巨大努力 (Volume) - 成交量显著放大
        volume_score = df['volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        
        # 支柱三: 糟糕结果 (Weak Close) - 收盘价在日内位置偏低
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / total_range).fillna(0.5)
        weak_close_score = 1 - close_position_in_range
        
        # 废除上影线逻辑，采用更稳健的三支柱融合
        # upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        # upper_shadow_ratio = (upper_shadow / total_range).fillna(0.0)
        # scaling_range = max(1.0 - upper_shadow_ratio_min, 0.001)
        # upper_shadow_score = ((upper_shadow_ratio - upper_shadow_ratio_min) / scaling_range).clip(0, 1).fillna(0)
        
        # 新的核心公式，直接融合三个本质支柱
        final_score = (overextension_score * volume_score * weak_close_score).astype(np.float32)
        final_score.name = 'SCORE_RISK_UPTHRUST_DISTRIBUTION'
        return final_score

    def _diagnose_ma_breakdown(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.2 · 参数访问修复版】
        - 核心修复: 修正了参数获取逻辑，使用正确的 get_params_block 工具。
        """
        # 修正参数获取逻辑，使用正确的get_params_block工具，并修复了误导性的参数名
        p = get_params_block(self.strategy, 'structure_breakdown_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN')
        
        breakdown_ma_period = get_param_value(p.get('breakdown_ma_period'), 21)
        ma_col = f'EMA_{breakdown_ma_period}_D'

        if not all(col in df.columns for col in ['close_D', ma_col]):
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN')
            
        breakdown_depth = ((df[ma_col] - df['close_D']) / df[ma_col].replace(0, np.nan)).fillna(0)
        breakdown_depth = breakdown_depth.where(df['close_D'] < df[ma_col], 0).clip(0)
        # 使用配置中定义的norm_window，而不是硬编码
        norm_window = get_param_value(p.get('norm_window'), 55)
        min_periods = max(1, norm_window // 5)
        score = breakdown_depth.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        final_score = (score * (breakdown_depth > 0)).astype(np.float32)
        final_score.name = 'SCORE_BEHAVIOR_MA_BREAKDOWN'
        return final_score

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        states = {}
        required_cols = ['volume_D', 'VOL_MA_21_D', 'pct_change_D', 'SLOPE_5_volume_D', 'ACCEL_5_volume_D', 'VPA_EFFICIENCY_D', 'SLOPE_5_VPA_EFFICIENCY_D']
        if any(col not in df.columns for col in required_cols): return states
        p_vpa = params.get('vpa_dynamics_params', {})
        norm_window = get_param_value(p_vpa.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        volume_ratio = (df['volume_D'] / df['VOL_MA_21_D'].replace(0, np.nan)).fillna(1.0)
        huge_volume_score = volume_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        price_stagnant_score = (1 - df['pct_change_D'].abs().rolling(window=norm_window, min_periods=min_periods).rank(pct=True)).fillna(0.5)
        states['SCORE_RISK_VPA_STAGNATION'] = (huge_volume_score * price_stagnant_score).astype(np.float32)
        efficiency_decline_magnitude = df['SLOPE_5_VPA_EFFICIENCY_D'].where(df['SLOPE_5_VPA_EFFICIENCY_D'] < 0, 0).abs()
        efficiency_decline_score = efficiency_decline_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
        states['SCORE_RISK_VPA_EFFICIENCY_DECLINING'] = efficiency_decline_score.astype(np.float32)
        volume_accel_magnitude = df['ACCEL_5_volume_D'].where(df['ACCEL_5_volume_D'] > 0, 0)
        volume_accelerating_score = volume_accel_magnitude.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.0)
        states['SCORE_RISK_VPA_VOLUME_ACCELERATING'] = volume_accelerating_score.astype(np.float32)
        return states

    def _diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.1 · 数据契约修正版】
        - 核心修正: 修复了因依赖不存在的 'tr_D' 列而导致的 KeyError。
        - 战术替换: 在“衰竭反转”信号的“姿态测试”中，使用数据层提供的标准指标 'ATR_14_D' 替代 'tr_D' 来衡量价格波动性。
          这不仅解决了崩溃问题，还提升了代码的健壮性和专业性。
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

        p_drain = p.get('liquidity_drain_params', {})
        drain_window = get_param_value(p_drain.get('window'), 20)
        price_trend_score = normalize_score(df.get(f'SLOPE_{drain_window}_close_D'), df.index, norm_window, ascending=False)
        volume_trend_score = normalize_score(df.get(f'SLOPE_{drain_window}_volume_D'), df.index, norm_window, ascending=False)
        liquidity_drain_score = (price_trend_score * volume_trend_score)**0.5
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = liquidity_drain_score.astype(np.float32)

        p_rev = p.get('exhaustion_reversal_params', {})
        rev_window = get_param_value(p_rev.get('window'), 5)
        
        # --- 姿态测试: 价格在短期内拒绝创新低，波动收窄 ---
        is_stabilizing = (df['low_D'] >= df['low_D'].rolling(rev_window).min()).astype(int)
        
        # 使用 'ATR_14_D' 替换不存在的 'tr_D'
        # ATR_14_D 是一个更标准、更可靠的波动率指标
        price_volatility = df.get('ATR_14_D', pd.Series(df['high_D'] - df['low_D'], index=df.index)) # 使用get增加健壮性，如果ATR也没有，则用当日振幅作为备用
        
        # 波动率越低，企稳分数越高 (ascending=False)
        stabilization_score = is_stabilizing * normalize_score(price_volatility, df.index, norm_window, ascending=False)
        
        # --- 极态测试: 成交量达到近期地量水平 ---
        volume_dry_up_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=False)
        
        # --- 上下文: 发生在深跌之后更有意义 ---
        context_score = 1 - close_position_in_range
        
        # --- 融合三个测试，得到最终的衰竭反转分 ---
        exhaustion_reversal_score = (stabilization_score * volume_dry_up_score * context_score)
        states['SCORE_BULLISH_EXHAUSTION_REVERSAL'] = exhaustion_reversal_score.astype(np.float32)
        
        return states

    def _diagnose_atomic_bottom_formation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 信号净化版】原子级“底部形态”诊断引擎
        - 核心修复: 净化了输出信号的名称，移除了 '_S' 后缀。
        """
        states = {}
        norm_window = 60 

        # --- 支柱一: 生命线支撑 (Proximity to MA55 Lifeline) ---
        ma55 = df.get('EMA_55_D', df['close_D'])
        distance_from_ma55 = (df['close_D'] - ma55) / ma55
        lifeline_proximity_score = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2)
        
        # --- 支柱二: 悲观情绪衰竭 (Pessimism Exhaustion) ---
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        was_rsi_oversold = (rsi.rolling(window=10).min() < 35).astype(float)
        
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        
        pessimism_exhaustion_score = np.maximum(was_rsi_oversold, deep_bottom_context_score)

        # --- 支柱三: 波动率压缩 (Volatility Squeeze) ---
        vol_compression_score = normalize_score(df.get('BBW_21_2.0_D'), df.index, norm_window, ascending=False)

        # --- 最终融合 ---
        final_score = (lifeline_proximity_score * pessimism_exhaustion_score * vol_compression_score).astype(np.float32)
        
        states['SCORE_ATOMIC_BOTTOM_FORMATION'] = final_score
        
        return states

    def _diagnose_atomic_rebound_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 冥河之渡版】原子级“史诗探底回升”诊断引擎
        - 核心革命: “绝望背景”支柱升级为多周期、带权重的立体化模型，能够感知不同层次的绝望。
        - 架构升级: 复杂的“绝望背景”计算逻辑被封装到全新的 `_calculate_despair_context_score` 方法中。
        - 收益: 对绝望的理解从“点”升级为“体”，极大增强了对复杂市场环境的适应性。
        """
        states = {}
        p_rebound = get_params_block(self.strategy, 'panic_selling_setup_params', {})

        # --- 支柱一: 调用全新的“冥河之渡”引擎计算多维绝望背景分 ---
        despair_context_score = self._calculate_despair_context_score(df, p_rebound)

        # --- 支柱二: 结构测试 (调用“绝对领域”引擎，保持不变) ---
        structural_test_score = self._calculate_structural_test_score(df, p_rebound)

        # --- 支柱三: 当日确认 (确认今日K线收盘企稳，保持不变) ---
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.0)
        is_recovering_today = (df['pct_change_D'] > -0.01).astype(float)
        confirmation_score = (close_position_in_range * is_recovering_today)

        # --- 最终融合: 三者皆备，方为有效的探底回升 ---
        rebound_reversal_score = (despair_context_score * structural_test_score * confirmation_score).astype(np.float32)
        
        states['SCORE_ATOMIC_REBOUND_REVERSAL'] = rebound_reversal_score
        return states

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
        【V1.1 · 奥林匹斯熔炉版】原子级“延续性反转”诊断引擎
        - 核心革命: 彻底废除简陋的“生命线测试”逻辑，直接调用完整的“绝对领域”引擎
                      (`_calculate_structural_test_score`)作为第二支柱。
        - 新核心逻辑: 延续性反转 = 上升趋势确认 × 全功能结构测试 × 反转姿态确认
        - 收益: 赋予了延续性反转引擎与史诗级反转引擎同等级别的、最强大的结构感知能力。
        """
        states = {}
        p_continuation = get_params_block(self.strategy, 'continuation_reversal_params', {})
        
        # --- 支柱一: 上升趋势确认 (Uptrend Context) ---
        ma_periods = get_param_value(p_continuation.get('ma_periods'), [5, 13, 21, 55])
        uptrending_ma_count = pd.Series(0, index=df.index)
        for p in ma_periods:
            ma_col = f'EMA_{p}_D'
            if ma_col in df:
                uptrending_ma_count += (df[ma_col] > df[ma_col].shift(1)).astype(int)
        trend_alignment_score = uptrending_ma_count / len(ma_periods)

        # --- 支柱二: 全功能结构测试 (Full-featured Structural Test) ---
        # 废除之前简陋的、只包含均线的测试逻辑。
        # 直接调用与“史诗反转”引擎同源的、最强大的“绝对领域”引擎。
        # 这将自动包含对多周期前低、SBC低点的加权、共振测试。
        structural_test_score = self._calculate_structural_test_score(df, p_continuation)

        # --- 支柱三: 反转姿态确认 (Reversal Posture) ---
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.0)
        confirmation_score = close_position_in_range

        # --- 最终融合 ---
        continuation_reversal_score = (trend_alignment_score * structural_test_score * confirmation_score).astype(np.float32)
        
        states['SCORE_ATOMIC_CONTINUATION_REVERSAL'] = continuation_reversal_score
        return states

    def _calculate_structural_test_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V1.0 · 新增/移植】“绝对领域”结构共振测试引擎 (战神之矛版)
        - 来源: 从 TacticEngine 完美移植而来，作为“圣物普降”计划的一部分。
        - 核心职责: 为本模块提供与先知引擎完全一致的、最高规格的结构测试能力。
        """
        # --- 步骤 1: 获取参数，定义支撑矩阵 ---
        support_periods = get_param_value(params.get('support_lookback_periods'), [5, 10, 21, 55])
        period_weights = get_param_value(params.get('support_period_weights'), {5: 0.6, 10: 0.8, 21: 1.0, 55: 1.2, 'sbc': 1.5})
        support_tolerance_pct = get_param_value(params.get('support_tolerance_pct'), 0.01)
        confluence_bonus_factor = get_param_value(params.get('confluence_bonus_factor'), 0.2)
        sbc_threshold_pct = get_param_value(params.get('sbc_threshold_pct'), 0.05)

        # 锻造“战神之矛”：识别并追踪“重要看涨K线”的低点
        is_sbc = (df['pct_change_D'] > sbc_threshold_pct) & (df['volume_D'] > df.get('VOL_MA_21_D', 0))
        recent_sbc_low = df['low_D'].where(is_sbc).ffill()

        # 将“战神之矛”加入支撑矩阵
        support_levels = {f'EMA_{p}_D': df.get(f'EMA_{p}_D') for p in [5, 10, 21]}
        for p in support_periods:
            support_levels[f'PrevLow{p}'] = df['low_D'].shift(1).rolling(p, min_periods=max(1, p//2)).min()
        support_levels['RecentSBCLow'] = recent_sbc_low.shift(1)

        valid_supports = {k: v for k, v in support_levels.items() if v is not None and not v.empty}
        if not valid_supports:
            return pd.Series(0.0, index=df.index)
        
        supports_df = pd.concat(valid_supports, axis=1)

        # --- 步骤 2: 计算“结构共振”强度 ---
        confluence_df = pd.DataFrame(1.0, index=df.index, columns=supports_df.columns)
        for col_i in supports_df.columns:
            for col_j in supports_df.columns:
                if col_i == col_j: continue
                is_close = (supports_df[col_i] - supports_df[col_j]).abs() / supports_df[col_i].replace(0, np.nan) < support_tolerance_pct
                confluence_df[col_i] += is_close.astype(float)
        
        confluence_bonus_df = 1.0 + (confluence_df - 1) * confluence_bonus_factor

        # --- 步骤 3: 计算所有支撑位的加权、共振调整后的测试分数 ---
        all_test_scores = []
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        atr = df.get('ATR_14_D', day_range)

        for name, support_series in valid_supports.items():
            if 'SBC' in name:
                weight = period_weights.get('sbc', 1.5)
            else:
                period = int(''.join(filter(str.isdigit, name))) if any(char.isdigit() for char in name) else 21
                weight = period_weights.get(period, 1.0)
            
            confluence_bonus = confluence_bonus_df[name]

            # 3.1 计算“被接住”分数
            tolerance_buffer = (support_series * support_tolerance_pct).replace(0, np.nan)
            distance = (df['low_D'] - support_series).abs()
            base_proximity_score = np.exp(-((distance / tolerance_buffer)**2)).fillna(0)
            weighted_proximity_score = base_proximity_score * weight * confluence_bonus
            all_test_scores.append(weighted_proximity_score)

            # 3.2 计算“破位收回”分数 (仅对前低和SBC低点有效)
            if 'PrevLow' in name or 'SBC' in name:
                is_spring = (df['low_D'] < support_series) & (df['close_D'] > support_series)
                reclaim_strength = ((df['close_D'] - support_series) / day_range).clip(0, 1)
                base_reclaim_score = (is_spring * reclaim_strength).fillna(0)
                weighted_reclaim_score = base_reclaim_score * weight * confluence_bonus
                all_test_scores.append(weighted_reclaim_score)

        # --- 步骤 4: 融合所有测试分数，取当日最强的结构事件 ---
        if not all_test_scores:
            return pd.Series(0.0, index=df.index)
            
        final_score_matrix = pd.concat(all_test_scores, axis=1)
        final_structural_test_score = final_score_matrix.max(axis=1, skipna=True).fillna(0.0)
        
        return final_structural_test_score.clip(0, 1)









