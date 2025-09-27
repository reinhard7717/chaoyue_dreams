# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.utils import get_params_block, get_param_value, create_persistent_state, normalize_score, calculate_context_scores

class BehavioralIntelligence:
    def __init__(self, strategy_instance):
        """
        初始化行为与模式识别模块。
        :param strategy_instance: 策略主实例的引用。
        """
        self.strategy = strategy_instance
        # K线形态识别器可能需要在这里初始化或传入
        self.pattern_recognizer = strategy_instance.pattern_recognizer

    def run_behavioral_analysis_command(self) -> None:
        """
        【V3.1 · 原子信号暴露版】行为情报模块总指挥
        - 核心修复: 将内部生成的原子信号(internal_atomic_signals)显式更新到全局状态(self.strategy.atomic_states)，
                      以确保它们能被法医探针等下游模块正确访问，彻底修复探针因找不到中间信号而崩溃的BUG。
        """
        # print("      -> [行为情报模块总指挥 V3.1 · 原子信号暴露版] 启动...")
        df = self.strategy.df_indicators
        
        # 步骤1: 生成内部原子信号
        internal_atomic_signals = self._generate_all_atomic_signals(df)
        
        # 步骤2: 立即将内部原子信号暴露到全局状态，供下游消费
        if internal_atomic_signals:
            self.strategy.atomic_states.update(internal_atomic_signals)
        
        # 步骤3: 调用终极信号引擎，并传入已计算的原子信号以避免重复计算
        ultimate_behavioral_states = self.diagnose_ultimate_behavioral_signals(df, atomic_signals=internal_atomic_signals)
        
        # 步骤4: 更新终极信号到全局状态
        if ultimate_behavioral_states:
            self.strategy.atomic_states.update(ultimate_behavioral_states)
            # print(f"      -> [行为情报模块总指挥 V3.1] 分析完毕，共生成 {len(ultimate_behavioral_states)} 个终极行为信号。")

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series] = None) -> Dict[str, pd.Series]:
        """
        【V9.7 · 依赖注入版】终极行为信号诊断模块
        - 核心升级: 接受一个可选的 `atomic_signals` 字典作为参数。如果传入，则直接使用；
                      否则，自行计算。这使得上层模块可以预先计算并注入原子信号，避免重复工作。
        """
        # 如果没有从外部传入预先计算好的原子信号，则自行计算
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        # [代码修改结束]
        
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        dimension_weights = {'price': 0.4, 'volume': 0.3, 'kline': 0.3}
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 55) # 保持55天周期
        min_periods = max(1, norm_window // 5)
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)
        top_context_bonus_factor = get_param_value(p_conf.get('top_context_bonus_factor'), 0.8)
        
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        
        price_s_bull, price_d_bull, price_s_bear, price_d_bear = self._calculate_price_health(df, norm_window, min_periods, dynamic_weights, periods)
        vol_s_bull, vol_d_bull, vol_s_bear, vol_d_bear = self._calculate_volume_health(df, norm_window, min_periods, dynamic_weights, periods)
        kline_s_bull, kline_d_bull, kline_s_bear, kline_d_bear = self._calculate_kline_pattern_health(df, atomic_signals, norm_window, min_periods, periods)
        
        overall_health = {}
        dim_weights_array = np.array([dimension_weights['price'], dimension_weights['volume'], dimension_weights['kline']])
        
        for health_type, health_sources in [
            ('s_bull', [price_s_bull, vol_s_bull, kline_s_bull]),
            ('d_bull', [price_d_bull, vol_d_bull, kline_d_bull]),
            ('s_bear', [price_s_bear, vol_s_bear, kline_s_bear]),
            ('d_bear', [price_d_bear, vol_d_bear, kline_d_bear])
        ]:
        
            overall_health[health_type] = {}
            for p in periods:
                stacked_values = np.stack([
                    health_sources[0][p].values, health_sources[1][p].values, health_sources[2][p].values
                ], axis=0)
                fused_values = np.sum(stacked_values * dim_weights_array[:, np.newaxis], axis=0)
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)
        
        self.strategy.atomic_states['__BEHAVIOR_overall_health'] = overall_health
        
        bullish_resonance_health = {p: overall_health['s_bull'][p] * overall_health['d_bull'][p] for p in periods}
        bullish_short_force_res = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, 0.5)
        overall_bullish_resonance = (bullish_short_force_res * resonance_tf_weights['short'] + bullish_medium_trend_res * resonance_tf_weights['medium'] + bullish_long_inertia_res * resonance_tf_weights['long'])
        
        bullish_reversal_health = {p: overall_health['s_bear'][p] * overall_health['d_bull'][p] for p in periods}
        bullish_short_force_rev = (bullish_reversal_health.get(1, 0.5) * bullish_reversal_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_reversal_health.get(13, 0.5) * bullish_reversal_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_reversal_health.get(55, 0.5)
        overall_bullish_reversal_trigger = (bullish_short_force_rev * reversal_tf_weights['short'] + bullish_medium_trend_rev * reversal_tf_weights['medium'] + bullish_long_inertia_rev * reversal_tf_weights['long'])
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_score * bottom_context_bonus_factor)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['s_bear'][p] * overall_health['d_bear'][p] for p in periods}
        bearish_short_force_res = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, 0.5)
        overall_bearish_resonance = (bearish_short_force_res * resonance_tf_weights['short'] + bearish_medium_trend_res * resonance_tf_weights['medium'] + bearish_long_inertia_res * resonance_tf_weights['long'])

        bearish_reversal_health = {p: overall_health['s_bull'][p] * overall_health['d_bear'][p] for p in periods}
        bearish_short_force_rev = (bearish_reversal_health.get(1, 0.5) * bearish_reversal_health.get(5, 0.5))**0.5
        bearish_medium_trend_rev = (bearish_reversal_health.get(13, 0.5) * bearish_reversal_health.get(21, 0.5))**0.5
        bearish_long_inertia_rev = bearish_reversal_health.get(55, 0.5)
        overall_bearish_reversal_trigger = (bearish_short_force_rev * reversal_tf_weights['short'] + bearish_medium_trend_rev * reversal_tf_weights['medium'] + bearish_long_inertia_rev * reversal_tf_weights['long'])
        final_top_reversal_score = (overall_bearish_reversal_trigger * (1 + top_context_score * top_context_bonus_factor)).clip(0, 1)
        
        for prefix, score in [('SCORE_BEHAVIOR_BULLISH_RESONANCE', overall_bullish_resonance), ('SCORE_BEHAVIOR_BOTTOM_REVERSAL', final_bottom_reversal_score),
                              ('SCORE_BEHAVIOR_BEARISH_RESONANCE', overall_bearish_resonance), ('SCORE_BEHAVIOR_TOP_REVERSAL', final_top_reversal_score)]:
            states[f'{prefix}_S_PLUS'] = score.astype(np.float32)
            states[f'{prefix}_S'] = (score * 0.8).astype(np.float32)
            states[f'{prefix}_A'] = (score * 0.6).astype(np.float32)
            states[f'{prefix}_B'] = (score * 0.4).astype(np.float32)
        
        return states

    # ==============================================================================
    # 以下为新增的原子信号中心和降级的原子诊断引擎
    # ==============================================================================

    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """【V1.1 · 数据流修复版】原子信号中心，负责生产所有基础行为信号。"""
        atomic_signals = {}
        params = self.strategy.params
        
        atomic_signals.update(self._diagnose_kline_patterns(df))
        atomic_signals.update(self._diagnose_advanced_atomic_signals(df))
        atomic_signals.update(self._diagnose_board_patterns(df))
        atomic_signals.update(self._diagnose_price_volume_atomics(df))
        atomic_signals.update(self._diagnose_volume_price_dynamics(df, params))
        
        upthrust_score = self._diagnose_upthrust_distribution(df, params)
        atomic_signals[upthrust_score.name] = upthrust_score
        
        # 将 df 作为参数传递给 _diagnose_ma_breakdown 方法
        ma_breakdown_score = self._diagnose_ma_breakdown(df, params)
        atomic_signals[ma_breakdown_score.name] = ma_breakdown_score
        
        return atomic_signals

    def _calculate_price_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, dynamic_weights: Dict, periods: list) -> tuple:
        """【V1.1 · 重构版】计算价格维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        for p in periods:
            # 调用 utils.normalize_score
            s_bull[p] = normalize_score(df.get(f'price_vs_ma_{p}_D'), df.index, norm_window, ascending=True)
            s_bear[p] = normalize_score(df.get(f'price_vs_ma_{p}_D'), df.index, norm_window, ascending=False)
            price_mom = normalize_score(df.get(f'SLOPE_{p}_close_D'), df.index, norm_window, ascending=True)
            price_accel = normalize_score(df.get(f'ACCEL_{p}_close_D'), df.index, norm_window, ascending=True)
            d_bull[p] = price_mom * dynamic_weights['slope'] + price_accel * dynamic_weights['accel']
            price_mom_neg = normalize_score(df.get(f'SLOPE_{p}_close_D'), df.index, norm_window, ascending=False)
            price_accel_neg = normalize_score(df.get(f'ACCEL_{p}_close_D'), df.index, norm_window, ascending=False)
            d_bear[p] = price_mom_neg * dynamic_weights['slope'] + price_accel_neg * dynamic_weights['accel']
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_volume_health(self, df: pd.DataFrame, norm_window: int, min_periods: int, dynamic_weights: Dict, periods: list) -> tuple:
        """【V1.2 · 逻辑修复版】计算成交量维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        for p in periods:
            # 看涨逻辑：成交量高于均线为佳
            s_bull[p] = normalize_score(df.get(f'volume_vs_ma_{p}_D'), df.index, norm_window, ascending=True)
            # 看跌逻辑：成交量低于均线为佳（缩量），修正 ascending 参数
            s_bear[p] = normalize_score(df.get(f'volume_vs_ma_{p}_D'), df.index, norm_window, ascending=False)
            
            # 看涨动态：成交量斜率、加速度增加为佳
            vol_mom = normalize_score(df.get(f'SLOPE_{p}_volume_D'), df.index, norm_window, ascending=True)
            vol_accel = normalize_score(df.get(f'ACCEL_{p}_volume_D'), df.index, norm_window, ascending=True)
            d_bull[p] = vol_mom * dynamic_weights['slope'] + vol_accel * dynamic_weights['accel']
            
            # 看跌动态：成交量在下跌中放大是风险，斜率和加速度增加为坏，修正 ascending 参数为 True
            vol_mom_neg = normalize_score(df.get(f'SLOPE_{p}_volume_D'), df.index, norm_window, ascending=True)
            vol_accel_neg = normalize_score(df.get(f'ACCEL_{p}_volume_D'), df.index, norm_window, ascending=True)
            d_bear[p] = vol_mom_neg * dynamic_weights['slope'] + vol_accel_neg * dynamic_weights['accel']
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_kline_pattern_health(self, df: pd.DataFrame, atomic_signals: Dict[str, pd.Series], norm_window: int, min_periods: int, periods: list) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        【V2.1 · 逻辑革命版】
        - 核心修复: 将原子信号的融合逻辑从“相乘”(AND)彻底修改为“取最大值”(OR)。
                      这解决了因单一信号为0而导致整个支柱评分为0的致命缺陷。
        """
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        
        # --- 看涨静态分 (Bullish Static Score) ---
        strong_close = normalize_score(atomic_signals.get('SCORE_ATOMIC_STRONG_CLOSE'), df.index, norm_window, True, min_periods)
        gap_support = normalize_score(atomic_signals.get('SCORE_ATOMIC_GAP_SUPPORT'), df.index, norm_window, True, min_periods)
        earth_heaven = normalize_score(atomic_signals.get('SCORE_OPP_EARTH_HEAVEN_BOARD'), df.index, norm_window, True, min_periods)
        gentle_rise = normalize_score(atomic_signals.get('SCORE_ATOMIC_GENTLE_RISE'), df.index, norm_window, True, min_periods)
        
        # [代码修改] 从“与”逻辑(相乘)修改为“或”逻辑(取最大值)
        static_bull_score = pd.Series(np.maximum.reduce([
            strong_close.values, gap_support.values, earth_heaven.values, gentle_rise.values
        ]), index=df.index).astype(np.float32)

        # --- 看跌静态分 (Bearish Static Score) ---
        weak_close = normalize_score(atomic_signals.get('SCORE_ATOMIC_WEAK_CLOSE'), df.index, norm_window, True, min_periods)
        upthrust = normalize_score(atomic_signals.get('SCORE_RISK_UPTHRUST_DISTRIBUTION'), df.index, norm_window, True, min_periods)
        heaven_earth = normalize_score(atomic_signals.get('SCORE_RISK_HEAVEN_EARTH_BOARD'), df.index, norm_window, True, min_periods)
        sharp_drop = normalize_score(atomic_signals.get('SCORE_ATOMIC_SHARP_DROP'), df.index, norm_window, True, min_periods)
        
        # [代码修改] 从“与”逻辑(相乘)修改为“或”逻辑(取最大值)
        static_bear_score = pd.Series(np.maximum.reduce([
            weak_close.values, upthrust.values, heaven_earth.values, sharp_drop.values
        ]), index=df.index).astype(np.float32)

        # 动态分保持不变
        dynamic_bull_score = normalize_score(atomic_signals.get('SCORE_BEHAVIOR_MA_BREAKTHROUGH'), df.index, norm_window, True, min_periods)
        dynamic_bear_score = normalize_score(atomic_signals.get('SCORE_BEHAVIOR_MA_BREAKDOWN'), df.index, norm_window, True, min_periods)

        for p in periods:
            s_bull[p] = static_bull_score
            s_bear[p] = static_bear_score
            d_bull[p] = dynamic_bull_score
            d_bear[p] = dynamic_bear_score
            
        return s_bull, d_bull, s_bear, d_bear

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
        # 修正参数获取逻辑，使用正确的get_params_block工具，并修复了误导性的参数名
        p = get_params_block(self.strategy, 'upthrust_distribution_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        upper_shadow_ratio_min = get_param_value(p.get('upper_shadow_ratio_min'), 0.5)
        ma_col = f'EMA_{overextension_ma_period}_D'
        if not all(col in df.columns for col in ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col]):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        # 使用配置中定义的norm_window，而不是硬编码
        norm_window = get_param_value(p.get('norm_window'), 55)
        min_periods = max(1, norm_window // 5)
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        overextension_score = overextension_ratio.rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        total_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upper_shadow_ratio = (upper_shadow / total_range).fillna(0.0)
        scaling_range = max(1.0 - upper_shadow_ratio_min, 0.001)
        upper_shadow_score = ((upper_shadow_ratio - upper_shadow_ratio_min) / scaling_range).clip(0, 1).fillna(0)
        volume_score = df['volume_D'].rolling(window=norm_window, min_periods=min_periods).rank(pct=True).fillna(0.5)
        close_position_in_range = ((df['close_D'] - df['low_D']) / total_range).fillna(0.5)
        weak_close_score = 1 - close_position_in_range
        final_score = (overextension_score * upper_shadow_score * volume_score * weak_close_score).astype(np.float32)
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
        return states
