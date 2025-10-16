# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, create_persistent_state, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics

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
        【V27.0 · 分层印证版】行为终极信号诊断引擎
        - 核心升级: 全面采纳“分层动态印证”框架。价格、成交量、K线三大支柱的健康度计算，均升级为战术周期与上下文周期的动态共振模式。
        - 架构重构: 废除旧的 _calculate_*_health 辅助方法，将分层逻辑直接整合到本方法中，使诊断流程更内聚、更清晰。
        """
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        # 全面重构为分层印证框架
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        sorted_periods = sorted(periods)
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        # 步骤一：调用“宙斯之雷”引擎获取权威的底部形态信号
        p_fusion = get_param_value(p_conf.get('supreme_fusion_params'), {})
        bottom_formation_score = self._supreme_fusion_engine(df=df, signals_to_fuse=atomic_signals, params=p_fusion)
        self.strategy.atomic_states['SCORE_UNIVERSAL_BOTTOM_PATTERN'] = bottom_formation_score
        # 步骤二：计算并存储近期反转上下文
        reversal_echo_window = get_param_value(p_conf.get('reversal_echo_window'), 3)
        recent_reversal_context = bottom_formation_score.rolling(window=reversal_echo_window, min_periods=1).max()
        self.strategy.atomic_states['SCORE_CONTEXT_RECENT_REVERSAL'] = recent_reversal_context.astype(np.float32)
        # 步骤三：为每个周期，通过分层印证计算三大支柱的健康度
        overall_health = {'s_bull': {}, 's_bear': {}, 'd_intensity': {}}
        price_holo_bull, price_holo_bear = calculate_holographic_dynamics(df, 'close_D', norm_window)
        vol_holo_bull, vol_holo_bear = calculate_holographic_dynamics(df, 'volume_D', norm_window)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            # --- 1. 价格健康度 (分层计算) ---
            bbp = df.get('BBP_21_2.0_D', pd.Series(0.5, index=df.index)).fillna(0.5).clip(0, 1)
            day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            close_pos = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.5)
            is_pos_day = df['pct_change_D'] > 0
            bull_score_up = (bbp * close_pos)**0.5
            reversal_potential = close_pos.where(df['pct_change_D'] < 0, 0)
            price_s_bull = pd.Series(np.where(is_pos_day, bull_score_up, reversal_potential), index=df.index, dtype=np.float32)
            price_s_bear = ((1.0 - bbp) * (1.0 - close_pos))**0.5
            price_d_intensity = ((price_holo_bull + price_holo_bear) / 2.0)
            # --- 2. 成交量健康度 (分层计算) ---
            tactical_vol_inc = normalize_score(df['volume_D'], df.index, p, ascending=True)
            context_vol_inc = normalize_score(df['volume_D'], df.index, context_p, ascending=True)
            vol_inc = (tactical_vol_inc * context_vol_inc)**0.5
            tactical_price_stag = 1 - normalize_score(df['pct_change_D'].abs(), df.index, p, ascending=True)
            context_price_stag = 1 - normalize_score(df['pct_change_D'].abs(), df.index, context_p, ascending=True)
            price_stag = (tactical_price_stag * context_price_stag)**0.5
            tactical_price_drop = normalize_score(df['pct_change_D'].clip(upper=0).abs(), df.index, p, ascending=True)
            context_price_drop = normalize_score(df['pct_change_D'].clip(upper=0).abs(), df.index, context_p, ascending=True)
            price_drop = (tactical_price_drop * context_price_drop)**0.5
            stagnation_path = vol_inc * price_stag
            breakdown_path = (price_drop * vol_inc).where(df['pct_change_D'] < 0, 0)
            vol_s_bear = np.maximum(stagnation_path, breakdown_path)
            price_inc = normalize_score(df['pct_change_D'].clip(lower=0), df.index, p, ascending=True)
            yang_score = (price_inc * vol_inc)
            yin_score = (1.0 - vol_s_bear) * (1.0 - atomic_signals.get('SCORE_RISK_LIQUIDITY_DRAIN', 0.0))
            selling_exhaustion = (1.0 - vol_inc) * price_drop
            exhaustion_reversal = atomic_signals.get('SCORE_BULLISH_EXHAUSTION_REVERSAL', 0.0)
            bull_down_day = np.maximum.reduce([yin_score.values, selling_exhaustion.values, exhaustion_reversal.values])
            vol_s_bull = pd.Series(np.where(is_pos_day, yang_score, bull_down_day), index=df.index)
            vol_d_intensity = ((vol_holo_bull + vol_holo_bear) / 2.0)
            # --- 3. K线形态健康度 (分层计算) ---
            strong_close = normalize_score(atomic_signals.get('SCORE_PRICE_POSITION_IN_RANGE', 0.5), df.index, p, True)
            gap_support = normalize_score(atomic_signals.get('SCORE_GAP_SUPPORT_ACTIVE', 0.0), df.index, p, True)
            earth_heaven = normalize_score(atomic_signals.get('SCORE_BOARD_EARTH_HEAVEN', 0.0), df.index, p, True)
            gentle_rise = normalize_score((df['pct_change_D'].clip(0, 0.03) / 0.03), df.index, p, True)
            kline_s_bull = pd.Series(np.maximum.reduce([strong_close.values, gap_support.values, earth_heaven.values, gentle_rise.values]), index=df.index)
            weak_close = 1.0 - strong_close
            upthrust = normalize_score(atomic_signals.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', 0.0), df.index, p, True)
            heaven_earth = normalize_score(atomic_signals.get('SCORE_BOARD_HEAVEN_EARTH', 0.0), df.index, p, True)
            sharp_drop = normalize_score(atomic_signals.get('SCORE_KLINE_SHARP_DROP', 0.0), df.index, p, True)
            kline_s_bear = pd.Series(np.maximum.reduce([weak_close.values, upthrust.values, heaven_earth.values, sharp_drop.values]), index=df.index)
            bull_slope = kline_s_bull.diff(p).fillna(0).abs()
            bear_slope = kline_s_bear.diff(p).fillna(0).abs()
            kline_d_intensity = normalize_score(np.maximum(bull_slope, bear_slope), df.index, norm_window, ascending=True)
            # --- 4. 融合三大支柱健康度 ---
            pillar_weights = get_param_value(p_conf.get('pillar_weights'), {'price': 0.4, 'volume': 0.3, 'kline': 0.3})
            weights_array = np.array(list(pillar_weights.values()))
            weights_array /= weights_array.sum()
            for ht, hs in [('s_bull', [price_s_bull, vol_s_bull, kline_s_bull]), ('s_bear', [price_s_bear, vol_s_bear, kline_s_bear]), ('d_intensity', [price_d_intensity, vol_d_intensity, kline_d_intensity])]:
                stacked_values = np.stack([s.values for s in hs], axis=0)
                fused_values = np.prod(stacked_values ** weights_array[:, np.newaxis], axis=0)
                overall_health[ht][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        self.strategy.atomic_states['__BEHAVIOR_overall_health'] = overall_health
        # 步骤四：调用终极信号合成引擎
        ultimate_signals = transmute_health_to_ultimate_signals(df=df, atomic_states=self.strategy.atomic_states, overall_health=overall_health, params=p_synthesis, domain_prefix="BEHAVIOR")
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
        
        return atomic_signals

    def _calculate_price_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V4.1 · 赫尔墨斯之翼优化版】计算价格维度的三维健康度
        - 性能优化: 将不依赖于周期`p`的静态分(s_bull, s_bear)和动态分(d_intensity)的计算完全移出循环，
                      避免了(len(periods) - 1)次重复计算，大幅提升效率。
        - 核心逻辑: 保持原有的全息动态计算模式不变。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        # --- 将所有与周期p无关的计算提前至循环外 ---
        # 1. 计算静态看涨分 (s_bull)
        bbp = df.get('BBP_21_2.0_D', pd.Series(0.5, index=df.index)).fillna(0.5).clip(0, 1)
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / day_range).clip(0, 1).fillna(0.5)

        is_positive_day = df['pct_change_D'] > 0
        # 上涨日的看涨分：布林百分比与日内收盘位置的几何平均
        bullish_score_on_up_day = (bbp * close_position_in_range)**0.5
        # 下跌日的看涨分(潜在反转)：仅考虑日内收盘位置，收盘越高，反转潜力越大
        reversal_potential_score = close_position_in_range.where(df['pct_change_D'] < 0, 0)
        # 融合得到最终的静态看涨分
        static_bull_score = pd.Series(
            np.where(is_positive_day, bullish_score_on_up_day, reversal_potential_score),
            index=df.index,
            dtype=np.float32 # 直接指定数据类型，减少内存占用
        )

        # 2. 计算静态看跌分 (s_bear)
        # 看跌分：(1-布林百分比) 与 (1-日内收盘位置) 的几何平均，代表收盘弱势
        static_bear_score = ((1.0 - bbp) * (1.0 - close_position_in_range))**0.5
        static_bear_score = static_bear_score.astype(np.float32) # 指定数据类型

        # 3. 计算统一的动态强度分 (d_intensity)
        # 使用全息动态引擎计算价格的看涨和看跌动能
        price_holo_bull, price_holo_bear = calculate_holographic_dynamics(df, 'close_D', norm_window)
        # 统一的动态强度分是两种动能的平均值，反映价格变化的活跃程度
        unified_d_intensity = ((price_holo_bull + price_holo_bear) / 2.0).astype(np.float32)

        # --- 循环内仅进行高效的字典赋值操作 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            # 所有周期共享同一个、更高级的动态强度分
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_volume_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V7.1 · 赫尔墨斯之翼优化版】计算成交量维度的三维健康度
        - 性能优化: 将不依赖于周期`p`的静态分(s_bull, s_bear)和动态分(d_intensity)的计算完全移出循环，
                      避免了(len(periods) - 1)次重复计算，极大提升了执行效率。
        - 核心逻辑: 保持原有的全息动态计算模式不变。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        if 'pct_change_D' not in df.columns or 'volume_D' not in df.columns:
            # 如果缺少关键列，快速返回默认值
            default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
            for p in periods:
                s_bull[p] = s_bear[p] = d_intensity[p] = default_series
            return s_bull, s_bear, d_intensity
        # --- 将所有与周期p无关的计算提前至循环外 ---

        # 1. 计算静态看跌分 (s_bear)
        volume_increase_score = normalize_score(df['volume_D'], df.index, norm_window, ascending=True)
        price_stagnation_score = 1 - normalize_score(df['pct_change_D'].abs(), df.index, norm_window, ascending=True)
        # 路径1: 放量滞涨
        stagnation_path_score = volume_increase_score * price_stagnation_score
        price_drop_score = normalize_score(df['pct_change_D'].clip(upper=0).abs(), df.index, norm_window, ascending=True)
        # 路径2: 放量下跌
        breakdown_path_score = (price_drop_score * volume_increase_score).where(df['pct_change_D'] < 0, 0)
        # 看跌分取两种路径中的更强者
        static_bear_score = np.maximum(stagnation_path_score, breakdown_path_score).astype(np.float32)
        
        # 2. 计算静态看涨分 (s_bull)
        price_increase_score = normalize_score(df['pct_change_D'].clip(lower=0), df.index, norm_window, ascending=True)
        # 上涨日的看涨分：价涨量增（健康的阳）
        yang_score = (price_increase_score * volume_increase_score)
        
        liquidity_drain_risk = self.strategy.atomic_states.get('SCORE_RISK_LIQUIDITY_DRAIN', pd.Series(0.0, index=df.index))
        # 下跌日的看涨分基础：价跌但未放量破位（健康的阴）
        yin_score = ((1.0 - static_bear_score) * (1.0 - liquidity_drain_risk))
        
        shrinking_volume_score = 1.0 - volume_increase_score
        # 下跌日的看涨分补充1：缩量下跌（卖盘衰竭）
        selling_exhaustion_score = (shrinking_volume_score * price_drop_score)
        # 下跌日的看涨分补充2：外部计算的衰竭反转信号
        exhaustion_reversal_score = self.strategy.atomic_states.get('SCORE_BULLISH_EXHAUSTION_REVERSAL', pd.Series(0.0, index=df.index))
        
        is_positive_day = df['pct_change_D'] > 0
        # 下跌日的看涨分取三种可能中的最强者
        bullish_score_on_down_day = np.maximum.reduce([
            yin_score.values, 
            selling_exhaustion_score.values,
            exhaustion_reversal_score.values
        ])
        # 融合得到最终的静态看涨分
        static_bull_score_np = np.where(is_positive_day, yang_score, bullish_score_on_down_day)
        static_bull_score = pd.Series(static_bull_score_np, index=df.index, dtype=np.float32)

        # 3. 计算统一的动态强度分 (d_intensity)
        vol_holo_bull, vol_holo_bear = calculate_holographic_dynamics(df, 'volume_D', norm_window)
        unified_d_intensity = ((vol_holo_bull + vol_holo_bear) / 2.0).astype(np.float32)

        # --- 循环内仅进行高效的字典赋值操作 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            d_intensity[p] = unified_d_intensity
            
        return s_bull, s_bear, d_intensity

    def _calculate_kline_pattern_health(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], norm_window: int, min_periods: int, periods: list) -> Tuple[Dict, Dict, Dict]:
        """
        【V2.6 · 赫尔墨斯之翼优化版】计算K线形态维度的三维健康度
        - 性能优化: 将不依赖于周期`p`的静态分(s_bull, s_bear)计算移出循环，避免重复计算。
                      动态分(d_intensity)因其计算依赖于`p`(`diff(p)`)，故保留在循环内。
        - 核心逻辑: 保持原有的动态分计算逻辑不变。
        """
        s_bull, s_bear, d_intensity = {}, {}, {}
        
        # --- 将静态分的计算提前至循环外 ---
        strong_close = normalize_score(atomic_signals.get('SCORE_PRICE_POSITION_IN_RANGE', pd.Series(0.5, index=df.index)), df.index, norm_window, True, min_periods)
        gap_support = normalize_score(atomic_signals.get('SCORE_GAP_SUPPORT_ACTIVE', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        earth_heaven = normalize_score(atomic_signals.get('SCORE_BOARD_EARTH_HEAVEN', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        # 温和上涨：将日涨幅限制在3%以内进行归一化，鼓励稳定上涨而非暴涨
        gentle_rise_raw = df['pct_change_D'].clip(0, 0.03) / 0.03
        gentle_rise = normalize_score(gentle_rise_raw, df.index, norm_window, True, min_periods)
        # 静态看涨分：取强势收盘、缺口支撑、地天板、温和上涨中的最强者
        static_bull_score = pd.Series(np.maximum.reduce([strong_close.values, gap_support.values, earth_heaven.values, gentle_rise.values]), index=df.index, dtype=np.float32)

        weak_close = 1.0 - strong_close
        upthrust = normalize_score(atomic_signals.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        heaven_earth = normalize_score(atomic_signals.get('SCORE_BOARD_HEAVEN_EARTH', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        sharp_drop = normalize_score(atomic_signals.get('SCORE_KLINE_SHARP_DROP', pd.Series(0.0, index=df.index)), df.index, norm_window, True, min_periods)
        # 静态看跌分：取弱势收盘、上冲派发、天地板、急跌中的最强者
        static_bear_score = pd.Series(np.maximum.reduce([weak_close.values, upthrust.values, heaven_earth.values, sharp_drop.values]), index=df.index, dtype=np.float32)

        # --- 循环内仅保留必须依赖周期p的计算 ---
        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            
            # [代码保持] 动态强度分的计算依赖于周期p，因此保留在循环内
            # K线形态的动态分衡量的是“静态分自身的变化强度”
            bull_slope_strength = static_bull_score.diff(p).fillna(0).abs()
            bear_slope_strength = static_bear_score.diff(p).fillna(0).abs()
            # 取看涨和看跌分中变化更剧烈的一方作为动态强度
            intensity_slope = np.maximum(bull_slope_strength, bear_slope_strength)
            d_intensity[p] = normalize_score(intensity_slope, df.index, norm_window, ascending=True)
            
        return s_bull, s_bear, d_intensity

    # 以下方法被降级为私有，作为原子信号的生产者
    def _diagnose_kline_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 分层印证版】诊断K线原子形态
        - 核心升级: 对“急跌”信号的计算引入“分层动态印证”框架，使其对不同时间尺度的下跌趋势更敏感。
        - 保持不变: 缺口支撑的逻辑是事件驱动型，不适用分层框架，保持原逻辑。
        """
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        # --- 缺口支撑信号计算 (逻辑不变) ---
        p_gap = p.get('gap_support_params', {})
        if get_param_value(p_gap.get('enabled'), True):
            persistence_days = get_param_value(p_gap.get('persistence_days'), 10)
            gap_up_mask = df['low_D'] > df['high_D'].shift(1)
            gap_high = df['high_D'].shift(1).where(gap_up_mask).ffill()
            price_fills_gap_mask = df['close_D'] < gap_high
            gap_support_state = create_persistent_state(df=df, entry_event_series=gap_up_mask, persistence_days=persistence_days, break_condition_series=price_fills_gap_mask, state_name='KLINE_STATE_GAP_SUPPORT_ACTIVE')
            support_distance = (df['low_D'] - gap_high).clip(lower=0)
            normalization_base = (df['close_D'] * 0.1).replace(0, np.nan)
            support_strength_score = (support_distance / normalization_base).clip(0, 1).fillna(0)
            states['SCORE_GAP_SUPPORT_ACTIVE'] = (support_strength_score * gap_support_state).astype(np.float32)
        # --- 急跌信号计算 (应用分层印证) ---
        # [代码修改开始]
        p_atomic = p.get('atomic_behavior_params', {})
        if get_param_value(p_atomic.get('enabled'), True) and 'pct_change_D' in df.columns:
            periods = [1, 5, 13, 21, 55]
            sorted_periods = sorted(periods)
            sharp_drop_scores = {}
            drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            for i, p_tactical in enumerate(sorted_periods):
                p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
                tactical_score = normalize_score(drop_magnitude, df.index, p_tactical, ascending=True)
                context_score = normalize_score(drop_magnitude, df.index, p_context, ascending=True)
                sharp_drop_scores[p_tactical] = (tactical_score * context_score)**0.5
            # 跨周期融合
            tf_weights = {1: 0.1, 5: 0.4, 13: 0.3, 21: 0.1, 55: 0.1}
            final_fused_score = pd.Series(0.0, index=df.index)
            total_weight = sum(tf_weights.get(p, 0) for p in periods)
            if total_weight > 0:
                for p_tactical in periods:
                    weight = tf_weights.get(p_tactical, 0) / total_weight
                    final_fused_score += sharp_drop_scores.get(p_tactical, 0.0) * weight
            states['SCORE_KLINE_SHARP_DROP'] = final_fused_score.clip(0, 1).astype(np.float32)
        
        return states

    def _diagnose_advanced_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 赫尔墨斯之翼优化版】诊断高级原子信号
        - 内存优化: 对连涨/连跌天数的计数结果使用`np.int16`存储，减少内存占用。
        - 核心逻辑: 保持高效的向量化连胜/连败计算逻辑不变。
        """
        states = {}
        p = get_params_block(self.strategy, 'advanced_atomic_params', {}) 
        if not get_param_value(p.get('enabled'), True): return states
        
        # 计算收盘价在当日振幅中的位置，值域[0, 1]
        price_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        close_position_in_range = ((df['close_D'] - df['low_D']) / price_range).fillna(0.5)
        states['SCORE_PRICE_POSITION_IN_RANGE'] = close_position_in_range.astype(np.float32)
        
        # 高效计算连涨/连跌天数
        is_up_day = df['pct_change_D'] > 0
        is_down_day = df['pct_change_D'] < 0
        # 使用groupby和cumcount的经典向量化技巧计算连胜
        up_streak = (is_up_day.groupby((is_up_day != is_up_day.shift()).cumsum()).cumcount() + 1) * is_up_day
        down_streak = (is_down_day.groupby((is_down_day != is_down_day.shift()).cumsum()).cumcount() + 1) * is_down_day
        
        # 使用更节省内存的整数类型
        states['COUNT_CONSECUTIVE_UP_STREAK'] = up_streak.astype(np.int16)
        states['COUNT_CONSECUTIVE_DOWN_STREAK'] = down_streak.astype(np.int16)
        
        return states

    def _diagnose_board_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 · 赫尔墨斯之翼优化版】诊断天地板/地天板模式
        - 性能优化: 预先计算涨跌停价，并将多个评分因子的计算向量化，避免重复计算和中间Series的创建。
        - 核心逻辑: 保持评分的乘法融合逻辑不变。
        """
        states = {}
        p = get_params_block(self.strategy, 'board_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
        
        # --- 预先计算所有基础变量 ---
        prev_close = df['close_D'].shift(1)
        limit_up_threshold = get_param_value(p.get('limit_up_threshold'), 0.098)
        limit_down_threshold = get_param_value(p.get('limit_down_threshold'), -0.098)
        price_buffer = get_param_value(p.get('price_buffer'), 0.005)
        
        # 计算理论上的涨停价和跌停价
        limit_up_price = prev_close * (1 + limit_up_threshold)
        limit_down_price = prev_close * (1 + limit_down_threshold)
        
        # --- 向量化计算所有评分因子 ---
        # 1. 振幅强度分：当日振幅占理论最大振幅的比例
        day_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        theoretical_max_range = (limit_up_price - limit_down_price).replace(0, np.nan)
        strength_score = (day_range / theoretical_max_range).clip(0, 1).fillna(0)
        
        # 2. 地天板相关因子
        # 最低价接近跌停价的程度
        low_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['low_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 收盘价接近涨停价的程度
        close_near_limit_up_score = ((df['close_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        
        # 3. 天地板相关因子
        # 最高价接近涨停价的程度
        high_near_limit_up_score = ((df['high_D'] - limit_up_price * (1 - price_buffer)) / (limit_up_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        # 收盘价接近跌停价的程度
        close_near_limit_down_score = ((limit_down_price * (1 + price_buffer) - df['close_D']) / (limit_down_price * price_buffer).replace(0, np.nan)).clip(0, 1).fillna(0)
        
        # --- 最终融合 ---
        # 地天板得分 = 振幅强度 * 触及跌停 * 收于涨停
        states['SCORE_BOARD_EARTH_HEAVEN'] = (strength_score * low_near_limit_down_score * close_near_limit_up_score).astype(np.float32)
        # 天地板得分 = 振幅强度 * 触及涨停 * 收于跌停
        states['SCORE_BOARD_HEAVEN_EARTH'] = (strength_score * high_near_limit_up_score * close_near_limit_down_score).astype(np.float32)
        
        return states

    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V4.0 · 分层印证版】上冲派发风险诊断引擎
        - 核心升级: 引入“分层动态印证”框架。对构成“上冲派发”的乖离率、成交量、弱势收盘三大支柱进行多时间维度的分层验证。
        """
        p = get_params_block(self.strategy, 'upthrust_distribution_params', {})
        signal_name = 'SCORE_RISK_UPTHRUST_DISTRIBUTION'
        default_series = pd.Series(0.0, index=df.index, name=signal_name, dtype=np.float32)
        if not get_param_value(p.get('enabled'), False):
            return default_series
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        ma_col = f'EMA_{overextension_ma_period}_D'
        if not all(col in df.columns for col in ['high_D', 'low_D', 'close_D', 'volume_D', ma_col]):
            return default_series
        # 引入分层印证框架
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        upthrust_scores_by_period = {}
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        close_position_in_range = ((df['close_D'] - df['low_D']) / total_range).fillna(0.5)
        weak_close_score = 1 - close_position_in_range
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # 战术层
            tactical_overextension = normalize_score(overextension_ratio, df.index, p_tactical, ascending=True)
            tactical_volume = normalize_score(df['volume_D'], df.index, p_tactical, ascending=True)
            # 上下文层
            context_overextension = normalize_score(overextension_ratio, df.index, p_context, ascending=True)
            context_volume = normalize_score(df['volume_D'], df.index, p_context, ascending=True)
            # 融合
            fused_overextension = (tactical_overextension * context_overextension)**0.5
            fused_volume = (tactical_volume * context_volume)**0.5
            # 生成快照分
            snapshot_score = (fused_overextension * fused_volume * weak_close_score).astype(np.float32)
            # 对每个周期的快照分进行元分析
            period_signal = self._perform_relational_meta_analysis(df=df, snapshot_score=snapshot_score, signal_name=f"{signal_name}_{p_tactical}")
            upthrust_scores_by_period[p_tactical] = period_signal.get(f"{signal_name}_{p_tactical}", pd.Series(0.0, index=df.index))
        # 跨周期融合
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += upthrust_scores_by_period.get(p_tactical, 0.0) * weight
        final_fused_score.name = signal_name
        
        return final_fused_score.clip(0, 1).astype(np.float32)

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict) -> Dict[str, pd.Series]:
        """
        【V3.0 · 分层印证版】VPA动态诊断引擎
        - 核心升级: 对“VPA滞涨风险”信号引入“分层动态印证”框架，对放量和滞涨两个维度进行分层验证。
        """
        states = {}
        required_cols = ['volume_D', 'VOL_MA_21_D', 'pct_change_D']
        if any(col not in df.columns for col in required_cols): return states
        # 引入分层印证框架
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        stagnation_scores_by_period = {}
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            # 战术层
            tactical_volume_ratio = (df['volume_D'] / df[f'VOL_MA_{p_tactical}_D'].replace(0, np.nan)).fillna(1.0) if f'VOL_MA_{p_tactical}_D' in df else pd.Series(1.0, index=df.index)
            tactical_huge_volume = normalize_score(tactical_volume_ratio, df.index, p_tactical, ascending=True)
            tactical_price_stagnant = 1 - normalize_score(df['pct_change_D'].abs(), df.index, p_tactical, ascending=True)
            # 上下文层
            context_volume_ratio = (df['volume_D'] / df[f'VOL_MA_{p_context}_D'].replace(0, np.nan)).fillna(1.0) if f'VOL_MA_{p_context}_D' in df else pd.Series(1.0, index=df.index)
            context_huge_volume = normalize_score(context_volume_ratio, df.index, p_context, ascending=True)
            context_price_stagnant = 1 - normalize_score(df['pct_change_D'].abs(), df.index, p_context, ascending=True)
            # 融合
            fused_huge_volume = (tactical_huge_volume * context_huge_volume)**0.5
            fused_price_stagnant = (tactical_price_stagnant * context_price_stagnant)**0.5
            # 生成快照分
            stagnation_snapshot_score = (fused_huge_volume * fused_price_stagnant).astype(np.float32)
            # 对每个周期的快照分进行元分析
            signal_name = f"SCORE_RISK_VPA_STAGNATION_{p_tactical}"
            period_signal = self._perform_relational_meta_analysis(df=df, snapshot_score=stagnation_snapshot_score, signal_name=signal_name)
            stagnation_scores_by_period[p_tactical] = period_signal.get(signal_name, pd.Series(0.0, index=df.index))
        # 跨周期融合
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_fused_score = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                weight = tf_weights.get(p_tactical, 0) / total_weight
                final_fused_score += stagnation_scores_by_period.get(p_tactical, 0.0) * weight
        states['SCORE_RISK_VPA_STAGNATION'] = final_fused_score.clip(0, 1).astype(np.float32)
        
        return states

    def _diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.0 · 分层印证版】量价原子信号诊断引擎
        - 核心升级: 对“缩量下跌”、“流动性枯竭风险”、“卖盘衰竭反转”三大信号全面引入“分层动态印证”框架。
        """
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states
        # 引入分层印证框架
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        # --- 信号一: 缩量下跌 (SCORE_VOL_WEAKENING_DROP) ---
        if 'pct_change_D' in df.columns:
            weakening_drop_scores = {}
            drop_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
            for i, p_tactical in enumerate(sorted_periods):
                p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
                tactical_price_drop = normalize_score(drop_magnitude, df.index, p_tactical, ascending=True)
                tactical_vol_shrink = normalize_score(df['volume_D'], df.index, p_tactical, ascending=False)
                context_price_drop = normalize_score(drop_magnitude, df.index, p_context, ascending=True)
                context_vol_shrink = normalize_score(df['volume_D'], df.index, p_context, ascending=False)
                fused_price_drop = (tactical_price_drop * context_price_drop)**0.5
                fused_vol_shrink = (tactical_vol_shrink * context_vol_shrink)**0.5
                weakening_drop_scores[p_tactical] = (fused_price_drop * fused_vol_shrink)
            final_weakening_drop = pd.Series(0.0, index=df.index)
            total_weight = sum(tf_weights.get(p, 0) for p in periods)
            if total_weight > 0:
                for p_tactical in periods:
                    final_weakening_drop += weakening_drop_scores.get(p_tactical, 0.0) * (tf_weights.get(p_tactical, 0) / total_weight)
            states['SCORE_VOL_WEAKENING_DROP'] = final_weakening_drop.clip(0, 1).astype(np.float32)
        # --- 信号二: 流动性枯竭风险 (SCORE_RISK_LIQUIDITY_DRAIN) ---
        drain_scores = {}
        price_drop_magnitude = df['pct_change_D'].clip(upper=0).abs()
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_price_drop = normalize_score(price_drop_magnitude, df.index, p_tactical, ascending=True)
            tactical_vol_shrink = normalize_score(df['volume_D'], df.index, p_tactical, ascending=False)
            context_price_drop = normalize_score(price_drop_magnitude, df.index, p_context, ascending=True)
            context_vol_shrink = normalize_score(df['volume_D'], df.index, p_context, ascending=False)
            fused_price_drop = (tactical_price_drop * context_price_drop)**0.5
            fused_vol_shrink = (tactical_vol_shrink * context_vol_shrink)**0.5
            snapshot_score = (fused_price_drop * fused_vol_shrink).astype(np.float32)
            period_signal = self._perform_relational_meta_analysis(df, snapshot_score, f"drain_{p_tactical}")
            drain_scores[p_tactical] = period_signal.get(f"drain_{p_tactical}", pd.Series(0.0, index=df.index))
        final_drain_risk = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                final_drain_risk += drain_scores.get(p_tactical, 0.0) * (tf_weights.get(p_tactical, 0) / total_weight)
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = final_drain_risk.clip(0, 1).astype(np.float32)
        # --- 信号三: 卖盘衰竭反转 (SCORE_BULLISH_EXHAUSTION_REVERSAL) ---
        exhaustion_scores = {}
        close_position_in_range = self.strategy.atomic_states.get('SCORE_PRICE_POSITION_IN_RANGE', pd.Series(0.5, index=df.index))
        context_score = 1 - close_position_in_range
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            is_stabilizing = (df['low_D'] >= df['low_D'].rolling(p_tactical).min()).astype(int)
            price_volatility = df.get('ATR_14_D', pd.Series(df['high_D'] - df['low_D'], index=df.index))
            tactical_stabilization = is_stabilizing * normalize_score(price_volatility, df.index, p_tactical, ascending=False)
            tactical_vol_dry_up = normalize_score(df['volume_D'], df.index, p_tactical, ascending=False)
            context_stabilization = is_stabilizing * normalize_score(price_volatility, df.index, p_context, ascending=False)
            context_vol_dry_up = normalize_score(df['volume_D'], df.index, p_context, ascending=False)
            fused_stabilization = (tactical_stabilization * context_stabilization)**0.5
            fused_vol_dry_up = (tactical_vol_dry_up * context_vol_dry_up)**0.5
            snapshot_score = (fused_stabilization * fused_vol_dry_up * context_score)
            period_signal = self._perform_relational_meta_analysis(df, snapshot_score, f"exhaust_{p_tactical}")
            exhaustion_scores[p_tactical] = period_signal.get(f"exhaust_{p_tactical}", pd.Series(0.0, index=df.index))
        final_exhaustion_rev = pd.Series(0.0, index=df.index)
        total_weight = sum(tf_weights.get(p, 0) for p in periods)
        if total_weight > 0:
            for p_tactical in periods:
                final_exhaustion_rev += exhaustion_scores.get(p_tactical, 0.0) * (tf_weights.get(p_tactical, 0) / total_weight)
        states['SCORE_BULLISH_EXHAUSTION_REVERSAL'] = final_exhaustion_rev.clip(0, 1).astype(np.float32)
        
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
        【V3.0 · 阿瑞斯之怒协议版】关系元分析核心引擎
        - 核心革命: 响应“重变化、轻状态”的哲学，从“状态 * (1 + 动态)”的乘法模型，升级为
                      “(状态*权重) + (速度*权重) + (加速度*权重)”的加法模型。
        - 核心目标: 即使静态分很低，只要动态（尤其是加速度）足够强，也能产生高分，真正捕捉“拐点”。
        """
        states = {}
        # 引入新的权重体系和加法融合模型
        # 从配置中获取新的加法模型权重
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        # 新的权重体系，直接作用于最终分数，而非杠杆
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4) # 赋予加速度最高权重
        # --- 从ProcessIntelligence借鉴的核心参数 ---
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







