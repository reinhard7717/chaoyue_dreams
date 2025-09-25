# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict
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
        【V3.0 · 终极净化版】行为情报模块总指挥
        - 核心重构 (本次修改):
          - [架构净化] 彻底废除旧的总指挥模式，本方法不再调用任何零散的原子诊断引擎。
          - [单一职责] 本方法的唯一职责是调用唯一的终极信号引擎 `diagnose_ultimate_behavioral_signals`，
                       并将其产出的最终信号更新到策略状态中。
        - 收益: 实现了与其他所有情报模块完全统一的、最纯粹的架构范式，数据流清晰，职责单一。
        """
        # print("      -> [行为情报模块总指挥 V3.0 · 终极净化版] 启动...") # 更新版本号和说明
        df = self.strategy.df_indicators
        ultimate_behavioral_states = self.diagnose_ultimate_behavioral_signals(df)
        if ultimate_behavioral_states:
            self.strategy.atomic_states.update(ultimate_behavioral_states)
            # print(f"      -> [行为情报模块总指挥 V3.0] 分析完毕，共生成 {len(ultimate_behavioral_states)} 个终极行为信号。")

    def diagnose_ultimate_behavioral_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.3 · 性能优化版】终极行为信号诊断模块
        - 核心重构: 彻底修改底部反转信号的合成逻辑，将“情景分”从“硬性门控”改为“奖励因子”。
        - 本次优化:
          - [效率] 重构了第4步“全局健康度融合”逻辑。通过将各维度健康分预先堆叠为NumPy数组，
                    再利用向量化操作进行加权求和，取代了原先在for循环中对Pandas Series的重复计算，
                    大幅提升了计算效率并减少了内存占用。
        """
        atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        
        # --- 1. 定义权重与参数 ---
        dynamic_weights = {'slope': 0.6, 'accel': 0.4}
        dimension_weights = {'price': 0.4, 'volume': 0.3, 'kline': 0.3}
        resonance_tf_weights = {'short': 0.2, 'medium': 0.5, 'long': 0.3}
        reversal_tf_weights = {'short': 0.6, 'medium': 0.3, 'long': 0.1}
        periods = get_param_value(p_conf.get('periods', [1, 5, 13, 21, 55]))
        norm_window = get_param_value(p_conf.get('norm_window'), 120)
        min_periods = max(1, norm_window // 5)
        bottom_context_bonus_factor = get_param_value(p_conf.get('bottom_context_bonus_factor'), 0.5)

        # --- 2. 调用公共函数计算上下文分数 ---
        bottom_context_score, top_context_score = calculate_context_scores(df, self.strategy.atomic_states)
        
        # --- 3. 调用所有行为健康度组件计算器 ---
        price_s_bull, price_d_bull, price_s_bear, price_d_bear = self._calculate_price_health(df, norm_window, min_periods, dynamic_weights, periods)
        vol_s_bull, vol_d_bull, vol_s_bear, vol_d_bear = self._calculate_volume_health(df, norm_window, min_periods, dynamic_weights, periods)
        kline_s_bull, kline_d_bull, kline_s_bear, kline_d_bear = self._calculate_kline_pattern_health(df, atomic_signals, norm_window, min_periods, periods)

        # --- 4. 独立融合，生成四个全局健康度 (向量化版本) ---
        overall_health = {}
        # 将各维度权重整理成一个列表，顺序与下面的数组堆叠顺序保持一致
        dim_weights_array = np.array([dimension_weights['price'], dimension_weights['volume'], dimension_weights['kline']])
        
        # 遍历四种健康度类型
        for health_type, health_sources in [
            ('bullish_static', [price_s_bull, vol_s_bull, kline_s_bull]),
            ('bullish_dynamic', [price_d_bull, vol_d_bull, kline_d_bull]),
            ('bearish_static', [price_s_bear, vol_s_bear, kline_s_bear]),
            ('bearish_dynamic', [price_d_bear, vol_d_bear, kline_d_bear])
        ]:
            overall_health[health_type] = {}
            # 遍历所有周期
            for p in periods:
                # 将三个维度的健康分Series的Numpy数组堆叠成一个 (3, N) 的矩阵
                # N是数据行数
                stacked_values = np.stack([
                    health_sources[0][p].values,
                    health_sources[1][p].values,
                    health_sources[2][p].values
                ], axis=0)
                
                # 使用NumPy的向量化乘法和加法，一次性完成加权求和
                # (3, N) 矩阵与 (3,) 权重的乘法(利用广播机制) -> (3, N) -> 沿axis=0求和 -> (N,)
                fused_values = np.sum(stacked_values * dim_weights_array[:, np.newaxis], axis=0)
                
                # 将最终的Numpy数组转换回Pandas Series
                overall_health[health_type][p] = pd.Series(fused_values, index=df.index, dtype=np.float32)

        # --- 5. 终极信号合成 (逻辑不变) ---
        bullish_resonance_health = {p: overall_health['bullish_static'][p] * overall_health['bullish_dynamic'][p] for p in periods}
        bullish_short_force_res = (bullish_resonance_health.get(1, 0.5) * bullish_resonance_health.get(5, 0.5))**0.5
        bullish_medium_trend_res = (bullish_resonance_health.get(13, 0.5) * bullish_resonance_health.get(21, 0.5))**0.5
        bullish_long_inertia_res = bullish_resonance_health.get(55, 0.5)
        overall_bullish_resonance = (bullish_short_force_res * resonance_tf_weights['short'] + bullish_medium_trend_res * resonance_tf_weights['medium'] + bullish_long_inertia_res * resonance_tf_weights['long'])
        
        bullish_dynamic_health = overall_health['bullish_dynamic']
        bullish_short_force_rev = (bullish_dynamic_health.get(1, 0.5) * bullish_dynamic_health.get(5, 0.5))**0.5
        bullish_medium_trend_rev = (bullish_dynamic_health.get(13, 0.5) * bullish_dynamic_health.get(21, 0.5))**0.5
        bullish_long_inertia_rev = bullish_dynamic_health.get(55, 0.5)
        overall_bullish_reversal_trigger = (bullish_short_force_rev * reversal_tf_weights['short'] + bullish_medium_trend_rev * reversal_tf_weights['medium'] + bullish_long_inertia_rev * reversal_tf_weights['long'])
        
        final_bottom_reversal_score = (overall_bullish_reversal_trigger * (1 + bottom_context_score * bottom_context_bonus_factor)).clip(0, 1)

        bearish_resonance_health = {p: overall_health['bearish_static'][p] * overall_health['bearish_dynamic'][p] for p in periods}
        bearish_short_force_res = (bearish_resonance_health.get(1, 0.5) * bearish_resonance_health.get(5, 0.5))**0.5
        bearish_medium_trend_res = (bearish_resonance_health.get(13, 0.5) * bearish_resonance_health.get(21, 0.5))**0.5
        bearish_long_inertia_res = bearish_resonance_health.get(55, 0.5)
        overall_bearish_resonance = (bearish_short_force_res * resonance_tf_weights['short'] + bearish_medium_trend_res * resonance_tf_weights['medium'] + bearish_long_inertia_res * resonance_tf_weights['long'])

        bearish_dynamic_health = overall_health['bearish_dynamic']
        bearish_short_force_rev = (bearish_dynamic_health.get(1, 0.5) * bearish_dynamic_health.get(5, 0.5))**0.5
        bearish_medium_trend_rev = (bearish_dynamic_health.get(13, 0.5) * bearish_dynamic_health.get(21, 0.5))**0.5
        bearish_long_inertia_rev = bearish_dynamic_health.get(55, 0.5)
        overall_bearish_reversal_trigger = (bearish_short_force_rev * reversal_tf_weights['short'] + bearish_medium_trend_rev * reversal_tf_weights['medium'] + bearish_long_inertia_rev * reversal_tf_weights['long'])
        final_top_reversal_score = top_context_score * overall_bearish_reversal_trigger

        # --- 6. 赋值 (逻辑不变) ---
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
        """【V1.1 · 重构版】计算成交量维度的四维健康度"""
        s_bull, d_bull, s_bear, d_bear = {}, {}, {}, {}
        for p in periods:
            # 调用 utils.normalize_score
            s_bull[p] = normalize_score(df.get(f'volume_vs_ma_{p}_D'), df.index, norm_window, ascending=True)
            s_bear[p] = normalize_score(df.get(f'volume_vs_ma_{p}_D'), df.index, norm_window, ascending=True)
            vol_mom = normalize_score(df.get(f'SLOPE_{p}_volume_D'), df.index, norm_window, ascending=True)
            vol_accel = normalize_score(df.get(f'ACCEL_{p}_volume_D'), df.index, norm_window, ascending=True)
            d_bull[p] = vol_mom * dynamic_weights['slope'] + vol_accel * dynamic_weights['accel']
            d_bear[p] = vol_mom * dynamic_weights['slope'] + vol_accel * dynamic_weights['accel']
        return s_bull, d_bull, s_bear, d_bear

    def _calculate_kline_pattern_health(self, df: pd.DataFrame, atomic_signals: Dict, norm_window: int, min_periods: int, periods: list) -> tuple:
        """
        【V1.1 · 性能优化版】计算K线形态与行为模式维度的四维健康度
        - 本次优化:
          - [效率] 将静态分和动态分的赋值操作移出 for 循环。由于这些分数对于所有周期 p 都是相同的，
                    因此在循环外计算一次，然后使用字典推导式生成结果，避免了不必要的重复赋值。
        """
        default_series = pd.Series(0.5, index=df.index, dtype=np.float32)
        
        strong_close = atomic_signals.get('SCORE_PRICE_POSITION_IN_RANGE', default_series)
        hammer_body = (df['close_D'] - df['open_D']).abs().replace(0, 0.0001)
        lower_shadow = (df[['open_D', 'close_D']].min(axis=1) - df['low_D']).clip(0)
        hammer_score = ((lower_shadow / hammer_body - 1.8) / 3).clip(0, 1).fillna(0)
        earth_heaven = atomic_signals.get('SCORE_BOARD_EARTH_HEAVEN', default_series)
        static_bull_score = (strong_close * hammer_score * earth_heaven)**(1/3)

        weak_close = 1.0 - strong_close
        upthrust = atomic_signals.get('SCORE_RISK_UPTHRUST_DISTRIBUTION', default_series)
        heaven_earth = atomic_signals.get('SCORE_BOARD_HEAVEN_EARTH', default_series)
        sharp_drop = atomic_signals.get('SCORE_KLINE_SHARP_DROP', default_series)
        static_bear_score = (weak_close * upthrust * heaven_earth * sharp_drop)**(1/4)

        up_streak = atomic_signals.get('COUNT_CONSECUTIVE_UP_STREAK', pd.Series(0, index=df.index)).astype(float)
        dynamic_bull_score = (up_streak / 5.0).clip(0, 1)
        down_streak = atomic_signals.get('COUNT_CONSECUTIVE_DOWN_STREAK', pd.Series(0, index=df.index)).astype(float)
        dynamic_bear_score = (down_streak / 5.0).clip(0, 1)

        # 使用字典推导式一次性生成所有周期的分数，避免在循环中重复赋值
        s_bull = {p: static_bull_score for p in periods}
        s_bear = {p: static_bear_score for p in periods}
        d_bull = {p: dynamic_bull_score for p in periods}
        d_bear = {p: dynamic_bear_score for p in periods}
        
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

    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        p = exit_params.get('upthrust_distribution_params', {})
        if not get_param_value(p.get('enabled'), False):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        overextension_ma_period = get_param_value(p.get('overextension_ma_period'), 55)
        upper_shadow_ratio_min = get_param_value(p.get('upper_shadow_ratio_min'), 0.5)
        ma_col = f'EMA_{overextension_ma_period}_D'
        if not all(col in df.columns for col in ['open_D', 'high_D', 'low_D', 'close_D', 'volume_D', ma_col]):
            return pd.Series(0.0, index=df.index, name='SCORE_RISK_UPTHRUST_DISTRIBUTION')
        norm_window = get_param_value(p.get('norm_window'), 120)
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

    def _diagnose_ma_breakdown(self, df: pd.DataFrame, exit_params: dict) -> pd.Series:
        """
        【V1.1 · 依赖注入修复版】
        - 核心修复: 方法签名增加了 df 参数，不再依赖 self.strategy.df 或 self.strategy.df_indicators。
                    所有对 DataFrame 的访问都改为使用传入的 df 参数，彻底解决了 AttributeError。
        """
        p = exit_params.get('structure_breakdown_params', {})
        if not get_param_value(p.get('enabled'), False):
            # 使用传入的 df.index，不再访问 self.strategy.df
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN')
        
        breakdown_ma_period = get_param_value(p.get('breakdown_ma_period'), 21)
        ma_col = f'EMA_{breakdown_ma_period}_D'

        if not all(col in df.columns for col in ['close_D', ma_col]):
            # 使用传入的 df.index
            return pd.Series(0.0, index=df.index, name='SCORE_BEHAVIOR_MA_BREAKDOWN')
            
        breakdown_depth = ((df[ma_col] - df['close_D']) / df[ma_col].replace(0, np.nan)).fillna(0)
        breakdown_depth = breakdown_depth.where(df['close_D'] < df[ma_col], 0).clip(0)
        norm_window = get_param_value(p.get('norm_window'), 120)
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
