# 文件: strategies/trend_following/intelligence/behavioral_intelligence.py
# 行为与模式识别模块
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from strategies.trend_following.intelligence.tactic_engine import TacticEngine
from strategies.trend_following.utils import transmute_health_to_ultimate_signals, get_params_block, get_param_value, create_persistent_state, normalize_score, normalize_to_bipolar, calculate_holographic_dynamics, bipolar_to_exclusive_unipolar

class BehavioralIntelligence:
    """
    【V28.0 · 结构升维版】
    - 核心升级: 废弃了旧的 _calculate_price_health, _calculate_volume_health, _calculate_kline_pattern_health 方法。
                所有健康度计算已统一由全新的 _calculate_structural_behavior_health 引擎负责。
    """
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
        【V28.0 · 结构升维版】行为终极信号诊断引擎
        - 核心升级: 废除旧的、基于OHLCV的价、量、K线三支柱健康度计算。
                      全面转向调用全新的`_calculate_structural_behavior_health`方法，
                      该方法直接消费由`AdvancedStructuralMetricsService`提供的高保真微观结构指标。
        """
        if atomic_signals is None:
            atomic_signals = self._generate_all_atomic_signals(df)
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        if not get_param_value(p_conf.get('enabled'), True): return states
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        # 废除旧的三大支柱健康度计算，调用全新的结构行为健康度引擎
        # 步骤一：调用“宙斯之雷”引擎获取权威的底部形态信号 (逻辑保留)
        p_fusion = get_param_value(p_conf.get('supreme_fusion_params'), {})
        bottom_formation_score = self._supreme_fusion_engine(df=df, signals_to_fuse=atomic_signals, params=p_fusion)
        self.strategy.atomic_states['SCORE_UNIVERSAL_BOTTOM_PATTERN'] = bottom_formation_score
        # 步骤二：计算并存储近期反转上下文 (逻辑保留)
        reversal_echo_window = get_param_value(p_conf.get('reversal_echo_window'), 3)
        recent_reversal_context = bottom_formation_score.rolling(window=reversal_echo_window, min_periods=1).max()
        self.strategy.atomic_states['SCORE_CONTEXT_RECENT_REVERSAL'] = recent_reversal_context.astype(np.float32)
        # 步骤三：【核心重构】调用全新的、基于高保真结构指标的健康度计算引擎
        overall_health = self._calculate_structural_behavior_health(df, p_conf)
        self.strategy.atomic_states['__BEHAVIOR_overall_health'] = overall_health
        # 步骤四：调用终极信号合成引擎 (逻辑保留，但消费的是更高质量的健康度数据)
        # 注意：我们将 domain_prefix 修改为 "STRUCT_BEHAVIOR" 以匹配新的信号字典定义
        ultimate_signals = transmute_health_to_ultimate_signals(
            df=df,
            atomic_states=self.strategy.atomic_states,
            overall_health=overall_health,
            params=p_synthesis,
            domain_prefix="STRUCT_BEHAVIOR"
        )
        states.update(ultimate_signals)
        
        return states
    # ==============================================================================
    # 以下为新增的原子信号中心和降级的原子诊断引擎
    # ==============================================================================
    def _generate_all_atomic_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 风险调光器版】原子信号中心
        - 核心升级: 实施“风险调光器”协议。
                      1. 分别调用方法生成`PROVISIONAL_RISK_UPPER_SHADOW`和`SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS`。
                      2. 调用新的融合器`_fuse_upper_shadow_signals`生成最终的、经过调制的风险信号。
        """
        atomic_signals = {}
        params = self.strategy.params
        day_quality_score = self._calculate_day_quality_score(df)
        atomic_signals.update(self._diagnose_atomic_bottom_formation(df))
        epic_reversal_states = self._diagnose_atomic_rebound_reversal(df)
        continuation_reversal_states = self._diagnose_atomic_continuation_reversal(df)
        epic_score = epic_reversal_states.get('SCORE_ATOMIC_REBOUND_REVERSAL', pd.Series(0.0, index=df.index))
        continuation_score = continuation_reversal_states.get('SCORE_ATOMIC_CONTINUATION_REVERSAL', pd.Series(0.0, index=df.index))
        final_rebound_score = np.maximum(epic_score, continuation_score)
        atomic_signals['SCORE_ATOMIC_REBOUND_REVERSAL'] = final_rebound_score.astype(np.float32)
        atomic_signals.update(continuation_reversal_states)
        # [代码修改开始] 实施“风险调光器”协议
        # 步骤1: 生成临时的原始信号
        provisional_kline_signals = self._diagnose_kline_patterns(df, day_quality_score)
        provisional_risk = provisional_kline_signals.get('PROVISIONAL_RISK_UPPER_SHADOW', pd.Series(0.0, index=df.index))
        atomic_signals.update(provisional_kline_signals) # 将其他kline信号(如缺口)加入
        # 步骤2: 生成意图诊断信号
        intent_signals = self._diagnose_upper_shadow_intent(df)
        intent_diagnosis = intent_signals.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS', pd.Series(0.0, index=df.index))
        atomic_signals.update(intent_signals)
        # 步骤3: 调用融合器生成最终风险信号
        fused_signals = self._fuse_upper_shadow_signals(provisional_risk, intent_diagnosis)
        atomic_signals.update(fused_signals)
        # [代码修改结束]
        atomic_signals.update(self._diagnose_advanced_atomic_signals(df))
        atomic_signals.update(self._diagnose_board_patterns(df))
        atomic_signals.update(self._diagnose_price_volume_atomics(df))
        atomic_signals.update(self._diagnose_volume_price_dynamics(df, params, day_quality_score))
        upthrust_score_series = self._diagnose_upthrust_distribution(df, params, day_quality_score)
        atomic_signals[upthrust_score_series.name] = upthrust_score_series
        atomic_signals.update(self._diagnose_smart_intraday_trading(df))
        atomic_signals.update(self._diagnose_structural_fault_breakthrough(df))
        return atomic_signals

    def _calculate_structural_behavior_health(self, df: pd.DataFrame, params: dict) -> Dict[str, Dict[int, pd.Series]]:
        """
        【V4.2 · 阿波罗战车版】结构与行为健康度计算核心引擎
        - 核心革命: 签署“阿波罗战车”协议，引入“日内四象限博弈”分析。
                      1. [轨迹四象限] 根据“跳空方向”和“实体方向”，将日内走势划分为四象限，并赋予“轨迹得分”。
                      2. [日内质量分] 融合“轨迹得分”和“影线修正分”，得到对K线质量的最终审判。
                      3. [终极强度融合] 使用“日内质量分”来调制“日间总涨跌幅”，计算出最终的“净有效强度”。
        - 收益: 能够精准解读“高开高走”、“低开高走”等不同日内轨迹的战术含义，评估结果更符合市场博弈的真实情况。
        """
        p_synthesis = get_params_block(self.strategy, 'ultimate_signal_synthesis_params', {})
        periods = get_param_value(p_synthesis.get('periods'), [1, 5, 13, 21, 55])
        sorted_periods = sorted(periods)
        norm_window = get_param_value(p_synthesis.get('norm_window'), 55)
        p_meta = get_param_value(params.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        s_bull, s_bear, d_intensity = {}, {}, {}
        # 实施“阿波罗战车”协议
        # --- 步骤1: 计算日内质量分 (Intraday Quality Score) ---
        gap_up = df['open_D'] > df['pre_close_D']
        gap_down = df['open_D'] < df['pre_close_D']
        body_up = df['close_D'] > df['open_D']
        body_down = df['close_D'] < df['open_D']
        # 1.1 轨迹得分 (Trajectory Score)
        trajectory_score = pd.Series(0.0, index=df.index)
        trajectory_score.loc[gap_up & body_up] = 1.0    # 高开高走
        trajectory_score.loc[gap_down & body_up] = 0.8   # 低开高走
        trajectory_score.loc[gap_up & body_down] = -0.8  # 高开低走
        trajectory_score.loc[gap_down & body_down] = -1.0  # 低开低走
        # 1.2 影线修正分 (Shadow Modifier)
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        lower_shadow = (np.minimum(df['open_D'], df['close_D']) - df['low_D']).clip(lower=0)
        shadow_modifier = ((lower_shadow - upper_shadow) / kline_range).fillna(0)
        # 1.3 融合得到日内质量分
        day_quality_score = (trajectory_score * 0.7 + shadow_modifier * 0.3).clip(-1, 1)
        # --- 步骤2: 计算净有效强度 ---
        quality_adjustment_factor = (1 + day_quality_score) / 2 # 将[-1, 1]映射到[0, 1]
        positive_day_strength_raw = df['pct_change_D'].clip(0)
        negative_day_strength_raw = df['pct_change_D'].clip(upper=0).abs()
        # 结果(50%) + 过程(50%)
        net_effective_bullish_strength = (positive_day_strength_raw * 0.5) + (positive_day_strength_raw * quality_adjustment_factor * 0.5)
        net_effective_bearish_strength = (negative_day_strength_raw * 0.5) + (negative_day_strength_raw * (1 - quality_adjustment_factor) * 0.5)
        # --- 步骤3: 应用“宙斯之雷”协议，归一化“净有效强度” ---
        positive_day_strength = normalize_score(net_effective_bullish_strength, df.index, norm_window) * (net_effective_bullish_strength > 0)
        negative_day_strength = normalize_score(net_effective_bearish_strength, df.index, norm_window) * (net_effective_bearish_strength > 0)
        
        efficiency_holo_bull, efficiency_holo_bear = calculate_holographic_dynamics(df, 'intraday_trend_efficiency_D', norm_window)
        gini_holo_bull, gini_holo_bear = calculate_holographic_dynamics(df, 'intraday_volume_gini_D', norm_window)
        bullish_d_intensity = ((efficiency_holo_bull + gini_holo_bull) / 2.0).astype(np.float32)
        bearish_d_intensity = ((efficiency_holo_bear + gini_holo_bear) / 2.0).astype(np.float32)
        closing_strength_score = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        closing_weakness_score = 1.0 - closing_strength_score
        bullish_divergence = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        auction_power = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        trend_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        bearish_divergence = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window)
        auction_weakness = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window)
        trend_inefficiency = 1 - trend_efficiency
        bullish_composite_state = (
            positive_day_strength * closing_strength_score * (1 + bullish_divergence) *
            auction_power * trend_efficiency * bullish_d_intensity
        )**(1/6)
        bearish_composite_state = (
            negative_day_strength * closing_weakness_score * (1 + bearish_divergence) *
            auction_weakness * trend_inefficiency * bearish_d_intensity
        )**(1/6)
        bipolar_composite_state = (bullish_composite_state - bearish_composite_state).clip(-1, 1)
        for i, p in enumerate(sorted_periods):
            context_p = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p
            state_norm_tactical = normalize_to_bipolar(bipolar_composite_state, df.index, p)
            slope_raw = bipolar_composite_state.diff(p).fillna(0)
            slope_norm_tactical = normalize_to_bipolar(slope_raw, df.index, p)
            accel_raw = slope_raw.diff(1).fillna(0)
            accel_norm_tactical = normalize_to_bipolar(accel_raw, df.index, p)
            tactical_health_bipolar = (
                state_norm_tactical * w_state +
                slope_norm_tactical * w_velocity +
                accel_norm_tactical * w_acceleration
            ).clip(-1, 1)
            state_norm_context = normalize_to_bipolar(bipolar_composite_state, df.index, context_p)
            slope_norm_context = normalize_to_bipolar(slope_raw, df.index, context_p)
            accel_norm_context = normalize_to_bipolar(accel_raw, df.index, context_p)
            context_health_bipolar = (
                state_norm_context * w_state +
                slope_norm_context * w_velocity +
                accel_norm_context * w_acceleration
            ).clip(-1, 1)
            final_dynamic_bipolar_health = (tactical_health_bipolar + context_health_bipolar) / 2.0
            s_bull[p] = final_dynamic_bipolar_health.astype(np.float32)
            s_bear[p] = pd.Series(0.0, index=df.index, dtype=np.float32)
        for p in periods:
            d_intensity[p] = pd.Series(1.0, index=df.index, dtype=np.float32)
        return {'s_bull': s_bull, 's_bear': s_bear, 'd_intensity': d_intensity}

    # 以下方法被降级为私有，作为原子信号的生产者
    def _diagnose_kline_patterns(self, df: pd.DataFrame, day_quality_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V3.7 · 职责分离版】诊断K线原子形态
        - 核心重构: 此方法不再输出最终的`SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW`。
                      根据“风险调光器”协议，它现在只负责计算并输出一个临时的、未经上下文调制的
                      原始抛压分 `PROVISIONAL_RISK_UPPER_SHADOW`，供下游融合器使用。
        """
        states = {}
        p = get_params_block(self.strategy, 'kline_pattern_params')
        if not get_param_value(p.get('enabled'), False): return states
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
        p_pressure = p.get('selling_pressure_params', {})
        # [代码修改开始] 信号名称变更为临时信号
        signal_name = 'PROVISIONAL_RISK_UPPER_SHADOW'
        # [代码修改结束]
        if get_param_value(p_pressure.get('enabled'), True):
            periods = get_param_value(p_pressure.get('periods'), [5, 13, 21, 55])
            sorted_periods = sorted(periods)
            pressure_scores = {}
            kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
            upper_shadow_ratio = (upper_shadow / kline_range).fillna(0)
            for p_tactical in sorted_periods:
                tactical_pressure = normalize_score(upper_shadow_ratio, df.index, p_tactical, ascending=True)
                tactical_volume = normalize_score(df['volume_D'], df.index, p_tactical, ascending=True)
                pressure_scores[p_tactical] = (tactical_pressure * tactical_volume)**0.5
            tf_weights = get_param_value(p_pressure.get('tf_weights'), {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1})
            final_fused_snapshot_score = pd.Series(0.0, index=df.index)
            numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
            total_weight = sum(numeric_tf_weights.values())
            if total_weight > 0:
                for p_tactical in periods:
                    weight = numeric_tf_weights.get(p_tactical, 0) / total_weight
                    final_fused_snapshot_score += pressure_scores.get(p_tactical, 0.0) * weight
            states[signal_name] = final_fused_snapshot_score.clip(0, 1).astype(np.float32)
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
        【V1.3 · 前线换装版】诊断天地板/地天板模式
        - 核心升级: 废弃 `auction_conviction_index_D`，换装为 `final_hour_momentum_D`
                      作为更可靠的收盘意图确认因子。
        """
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
        # 换装新式武器
        # 使用尾盘动能作为收盘意图的确认因子
        auction_bullish_confirmation = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, 55)
        auction_bearish_confirmation = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, 55)

        states['SCORE_BOARD_EARTH_HEAVEN'] = (strength_score * low_near_limit_down_score * close_near_limit_up_score * (1 + auction_bullish_confirmation)).clip(0, 1).astype(np.float32)
        states['SCORE_BOARD_HEAVEN_EARTH'] = (strength_score * high_near_limit_up_score * close_near_limit_down_score * (1 + auction_bearish_confirmation)).clip(0, 1).astype(np.float32)
        return states

    def _diagnose_upthrust_distribution(self, df: pd.DataFrame, params: dict, day_quality_score: pd.Series) -> pd.Series:
        """
        【V7.4 · 雅典娜敕令版】上冲派发风险诊断引擎
        - 核心修复: 贯彻“雅典娜敕令”，在风险驱动因子（weak_close_score）计算的源头，
                      就使用K线质量分进行过滤。确保只有在K线质量为负的日子里，
                      “上冲派发”风险才会被纳入计算，实现逻辑的源头纯粹性。
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
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        upthrust_scores_by_period = {}
        # [代码修改开始] 实施“雅典娜敕令”，在源头过滤风险驱动因子
        is_bearish_quality_day_mask = (day_quality_score <= 0).astype(int)
        # 只有在K线质量为负的日子里，才计算收盘疲弱分
        weak_close_score = (1.0 - normalize_score(df.get('closing_strength_index_D', 0.5), df.index, 55)) * is_bearish_quality_day_mask
        # [代码修改结束]
        overextension_ratio = (df['close_D'] / df[ma_col] - 1).clip(0)
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_overextension = normalize_score(overextension_ratio, df.index, p_tactical, ascending=True)
            tactical_volume = normalize_score(df['volume_D'], df.index, p_tactical, ascending=True)
            context_overextension = normalize_score(overextension_ratio, df.index, p_context, ascending=True)
            context_volume = normalize_score(df['volume_D'], df.index, p_context, ascending=True)
            fused_overextension = (tactical_overextension * context_overextension)**0.5
            fused_volume = (tactical_volume * context_volume)**0.5
            upthrust_scores_by_period[p_tactical] = (fused_overextension * fused_volume * weak_close_score).astype(np.float32)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_snapshot_score = pd.Series(0.0, index=df.index)
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        if total_weight > 0:
            for p_tactical in periods:
                weight = numeric_tf_weights.get(p_tactical, 0) / total_weight
                final_snapshot_score += upthrust_scores_by_period.get(p_tactical, 0.0) * weight
        final_snapshot_score.name = signal_name
        return final_snapshot_score.clip(0, 1).astype(np.float32)

    def _diagnose_volume_price_dynamics(self, df: pd.DataFrame, params: dict, day_quality_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V6.4 · 雅典娜敕令版】VPA动态诊断引擎
        - 核心修复: 贯彻“雅典娜敕令”，在风险驱动因子（tactical_price_stagnant）计算的源头，
                      就使用K线质量分进行过滤。确保只有在K线质量为负的日子里，
                      “VPA滞涨”风险才会被纳入计算，实现逻辑的源头纯粹性。
        """
        states = {}
        signal_name = 'SCORE_RISK_VPA_STAGNATION'
        required_cols = ['volume_D', 'VOL_MA_21_D', 'pct_change_D', 'intraday_trend_efficiency_D']
        if any(col not in df.columns for col in required_cols):
            states[signal_name] = pd.Series(0.0, index=df.index)
            return states
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        stagnation_scores_by_period = {}
        # [代码新增开始] 实施“雅典娜敕令”
        is_bearish_quality_day_mask = (day_quality_score <= 0).astype(int)
        # [代码新增结束]
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_volume_ratio = (df['volume_D'] / df[f'VOL_MA_{p_tactical}_D'].replace(0, np.nan)).fillna(1.0) if f'VOL_MA_{p_tactical}_D' in df else pd.Series(1.0, index=df.index)
            tactical_huge_volume = normalize_score(tactical_volume_ratio, df.index, p_tactical, ascending=True)
            # [代码修改开始] 在源头过滤风险驱动因子
            tactical_low_efficiency = 1 - normalize_score(df.get('intraday_trend_efficiency_D', 0.5), df.index, p_tactical, ascending=True)
            tactical_high_volatility = normalize_score(df.get('intraday_volatility_D', 0.0), df.index, p_tactical, ascending=True)
            # 只有在K线质量为负的日子里，才计算价格滞涨分
            tactical_price_stagnant = ((tactical_low_efficiency * tactical_high_volatility)**0.5) * is_bearish_quality_day_mask
            # [代码修改结束]
            context_volume_ratio = (df['volume_D'] / df[f'VOL_MA_{p_context}_D'].replace(0, np.nan)).fillna(1.0) if f'VOL_MA_{p_context}_D' in df else pd.Series(1.0, index=df.index)
            context_huge_volume = normalize_score(context_volume_ratio, df.index, p_context, ascending=True)
            # [代码修改开始] 在源头过滤风险驱动因子
            context_low_efficiency = 1 - normalize_score(df.get('intraday_trend_efficiency_D', 0.5), df.index, p_context, ascending=True)
            context_high_volatility = normalize_score(df.get('intraday_volatility_D', 0.0), df.index, p_context, ascending=True)
            # 只有在K线质量为负的日子里，才计算价格滞涨分
            context_price_stagnant = ((context_low_efficiency * context_high_volatility)**0.5) * is_bearish_quality_day_mask
            # [代码修改结束]
            fused_huge_volume = (tactical_huge_volume * context_huge_volume)**0.5
            fused_price_stagnant = (tactical_price_stagnant * context_price_stagnant)**0.5
            stagnation_scores_by_period[p_tactical] = (fused_huge_volume * fused_price_stagnant).astype(np.float32)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
        final_snapshot_score = pd.Series(0.0, index=df.index)
        numeric_tf_weights = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight = sum(numeric_tf_weights.values())
        if total_weight > 0:
            for p_tactical in periods:
                weight = numeric_tf_weights.get(p_tactical, 0) / total_weight
                final_snapshot_score += stagnation_scores_by_period.get(p_tactical, 0.0) * weight
        states[signal_name] = final_snapshot_score.clip(0, 1).astype(np.float32)
        return states

    def _diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V12.2 · 宙斯敕令版】量价原子信号诊断引擎
        - 核心修复: 贯彻“宙斯敕令”，修复“流动性枯竭风险”信号的逻辑。
                      对于这类“事件状态”型风险信号，废除使用复杂的“关系元分析”引擎，
                      直接使用其多周期融合的“快照分”作为最终信号。
        """
        states = {}
        p = get_params_block(self.strategy, 'price_volume_atomic_params')
        if not get_param_value(p.get('enabled'), True): return states
        periods = [5, 13, 21, 55]
        sorted_periods = sorted(periods)
        tf_weights = {5: 0.4, 13: 0.3, 21: 0.2, 55: 0.1}
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
            numeric_tf_weights_weak = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
            total_weight_weak = sum(numeric_tf_weights_weak.values())
            if total_weight_weak > 0:
                for p_tactical in periods:
                    final_weakening_drop += weakening_drop_scores.get(p_tactical, 0.0) * (numeric_tf_weights_weak.get(p_tactical, 0) / total_weight_weak)
            states['SCORE_VOL_WEAKENING_DROP'] = final_weakening_drop.clip(0, 1).astype(np.float32)
        drain_scores_by_period = {}
        price_drop_magnitude = df['pct_change_D'].clip(upper=0).abs()
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_price_drop = normalize_score(price_drop_magnitude, df.index, p_tactical, ascending=True)
            tactical_vol_shrink = normalize_score(df['volume_D'], df.index, p_tactical, ascending=False)
            context_price_drop = normalize_score(price_drop_magnitude, df.index, p_context, ascending=True)
            context_vol_shrink = normalize_score(df['volume_D'], df.index, p_context, ascending=False)
            fused_price_drop = (tactical_price_drop * context_price_drop)**0.5
            fused_vol_shrink = (tactical_vol_shrink * context_vol_shrink)**0.5
            drain_scores_by_period[p_tactical] = (fused_price_drop * fused_vol_shrink).astype(np.float32)
        final_drain_snapshot = pd.Series(0.0, index=df.index)
        numeric_tf_weights_drain = {int(k): v for k, v in tf_weights.items() if str(k).isdigit()}
        total_weight_drain = sum(numeric_tf_weights_drain.values())
        if total_weight_drain > 0:
            for p_tactical in periods:
                final_drain_snapshot += drain_scores_by_period.get(p_tactical, 0.0) * (numeric_tf_weights_drain.get(p_tactical, 0) / total_weight_drain)
        # [代码修改开始] 签署“宙斯敕令”：废除对“流动性枯竭风险”快照分进行不当的元分析
        # 旧的错误逻辑:
        # drain_signal_dict = self._perform_relational_meta_analysis(df, final_drain_snapshot, "SCORE_RISK_LIQUIDITY_DRAIN")
        # states.update(drain_signal_dict)
        # 新的正确逻辑:
        states['SCORE_RISK_LIQUIDITY_DRAIN'] = final_drain_snapshot.clip(0, 1).astype(np.float32)
        # [代码修改结束]
        norm_window = 55
        vol_dry_up = normalize_score(df['volume_D'], df.index, norm_window, ascending=False)
        bottom_support_power = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        bullish_divergence = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        auction_power = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        exhaustion_snapshot = (
            vol_dry_up * bottom_support_power * auction_power * (1 + bullish_divergence)
        ).clip(0, 1).astype(np.float32)
        exhaustion_signal_dict = self._perform_relational_meta_analysis(df, exhaustion_snapshot, "SCORE_BULLISH_EXHAUSTION_REVERSAL")
        states.update(exhaustion_signal_dict)
        return states

    def _diagnose_atomic_bottom_formation(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.3 · 前线换装版】原子级“底部形态”诊断引擎
        - 核心升级: 彻底清除对所有“幽灵信号”的依赖，全面换装为新式高保真指标。
        """
        # --- 支柱1: 宏观背景 (逻辑不变) ---
        ma55 = df.get('EMA_55_D', df['close_D'])
        distance_from_ma55 = (df['close_D'] - ma55) / ma55
        lifeline_proximity_score = np.exp(-((distance_from_ma55 - 0.015) / 0.03)**2)
        rsi = df.get('RSI_13_D', pd.Series(50, index=df.index))
        was_rsi_oversold = (rsi.rolling(window=10).min() < 35).astype(float)
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        pessimism_exhaustion_score = np.maximum(was_rsi_oversold, deep_bottom_context_score)
        macro_context_score = (lifeline_proximity_score * pessimism_exhaustion_score)**0.5
        # --- 支柱2: 静态设置 (逻辑不变) ---
        vol_compression_score = normalize_score(df.get('BBW_21_2.0_D', 1.0), df.index, 60, ascending=False)
        # 全面换装新式武器
        # --- 支柱3: 微观结构确认 (看到真实的转折力量) ---
        norm_window = 55
        closing_strength = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        vwap_dominance = normalize_score(df.get('close_vs_vwap_ratio_D', pd.Series(1.0, index=df.index)), df.index, norm_window)
        reversal_strength = (closing_strength * vwap_dominance)**0.5
        bullish_divergence = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        auction_power = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        trend_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        micro_confirmation_score = (
            reversal_strength * (1 + bullish_divergence) * auction_power * trend_efficiency
        )**(1/4)

        # --- 融合三大支柱 ---
        snapshot_score = (macro_context_score * vol_compression_score * micro_confirmation_score).astype(np.float32)
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_BOTTOM_FORMATION'
        )

    def _diagnose_atomic_rebound_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V6.3 · 前线换装版】原子级“史诗探底回升”诊断引擎
        - 核心升级: 彻底清除对所有“幽灵信号”的依赖，全面换装为新式高保真指标。
        """
        # --- 支柱1 & 2: 绝望背景 和 结构测试 (逻辑不变) ---
        p_rebound = get_params_block(self.strategy, 'panic_selling_setup_params', {})
        despair_context_score = self._calculate_despair_context_score(df, p_rebound)
        structural_test_score = self.tactic_engine.calculate_structural_test_score(df, p_rebound)
        # 全面换装新式武器
        # --- 支柱3: 微观结构确认 (全新的“反转质量”评分) ---
        norm_window = 55
        closing_strength = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        vwap_dominance = normalize_score(df.get('close_vs_vwap_ratio_D', pd.Series(1.0, index=df.index)), df.index, norm_window)
        reversal_intensity = (closing_strength * vwap_dominance)**0.5
        is_positive_day = df['pct_change_D'] > 0
        trend_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        efficient_rise = trend_efficiency.where(is_positive_day, 0)
        auction_power = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        confirmation_score = (reversal_intensity * efficient_rise * auction_power)**(1/3)

        # --- 融合三大支柱 ---
        snapshot_score = (despair_context_score * structural_test_score * confirmation_score).astype(np.float32)
        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_REBOUND_REVERSAL'
        )

    def _diagnose_atomic_continuation_reversal(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V5.4 · 阿波罗战车升级版】原子级“延续性反转”诊断引擎
        - 核心升级: 废弃了原有的、粗糙的`uptrending_ma_count`逻辑。
                      全面换装为调用全新的`_calculate_trend_health_score`引擎，
                      从“排列、速度、加速度、元动力”四个维度对趋势背景进行精准评估。
        """
        # --- 支柱1 & 2: 趋势背景 和 结构支撑 (逻辑不变) ---
        p_continuation = get_params_block(self.strategy, 'continuation_reversal_params', {})
        # 使用全新的四维趋势健康度评分替换旧的趋势对齐分
        trend_health_score = self._calculate_trend_health_score(df)

        structural_test_score = self.tactic_engine.calculate_structural_test_score(df, p_continuation)
        # --- 支柱3: 微观反转质量 (Micro-Behavioral Reversal Quality) ---
        norm_window = 55
        closing_strength = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        vwap_dominance = normalize_score(df.get('close_vs_vwap_ratio_D', pd.Series(1.0, index=df.index)), df.index, norm_window)
        reversal_intensity = (closing_strength * vwap_dominance)**0.5
        trend_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.0, index=df.index)), df.index, norm_window)
        bullish_divergence_bonus = normalize_score(df.get('flow_divergence_mf_vs_retail_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window) * 0.5
        auction_power = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(0), df.index, norm_window)
        confirmation_score = ((reversal_intensity * trend_efficiency * auction_power)**(1/3) + bullish_divergence_bonus).clip(0, 1)
        # --- 融合三大支柱 ---
        # 使用新的 trend_health_score 进行融合
        snapshot_score = (trend_health_score * structural_test_score * confirmation_score).astype(np.float32)

        return self._perform_relational_meta_analysis(
            df=df,
            snapshot_score=snapshot_score,
            signal_name='SCORE_ATOMIC_CONTINUATION_REVERSAL'
        )

    def _diagnose_smart_intraday_trading(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】“日内聪明钱”诊断引擎
        - 核心逻辑: 融合四大日内行为指标，高精度识别由专业交易者主导的、具有持续性的上涨行为。
                      1. 高执行Alpha: 低买高卖的能力。
                      2. 强劲收盘: 巩固日内战果。
                      3. 下午盘强于上午盘: 持续的买入意愿。
                      4. 高效趋势: 流畅的攻击效率。
        """
        #
        states = {}
        signal_name = 'SCORE_BEHAVIOR_SMART_INTRADAY_TRADING'
        norm_window = 55
        # 证据1: 高执行Alpha
        execution_alpha = normalize_score(df.get('intraday_execution_alpha_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据2: 强劲收盘
        closing_strength = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window, ascending=True)
        # 证据3: 下午盘强于上午盘
        afternoon_power = normalize_score(df.get('am_pm_vwap_ratio_D', pd.Series(1.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据4: 高效趋势
        trend_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.5, index=df.index)), df.index, norm_window, ascending=True)
        # 融合四大证据，生成瞬时快照分
        snapshot_score = (
            execution_alpha * closing_strength * afternoon_power * trend_efficiency
        )**(1/4)
        # 对快照分进行关系元分析，得到最终的动态信号
        final_signal_dict = self._perform_relational_meta_analysis(df=df, snapshot_score=snapshot_score, signal_name=signal_name)
        states.update(final_signal_dict)
        return states

    def _diagnose_structural_fault_breakthrough(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】“结构性断层突破”诊断引擎
        - 核心逻辑: 融合断层突破强度、断层真空度、成交量和收盘强度，
                      识别价格突破筹码真空区的关键结构性战机。
        """
        #
        states = {}
        signal_name = 'SCORE_STRUCTURAL_FAULT_BREAKTHROUGH'
        norm_window = 55
        # 证据1: 断层突破强度
        breakthrough_intensity = normalize_score(df.get('fault_breakthrough_intensity_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据2: 断层质量 (真空度)
        fault_quality = normalize_score(df.get('chip_fault_vacuum_percent_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据3: 力量确认 (成交量)
        volume_confirmation = normalize_score(df.get('volume_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=True)
        # 证据4: 战果确认 (收盘强度)
        closing_confirmation = normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window, ascending=True)
        # 融合四大证据，生成瞬时快照分
        snapshot_score = (
            breakthrough_intensity * fault_quality * volume_confirmation * closing_confirmation
        )**(1/4)
        # 对快照分进行关系元分析，得到最终的动态信号
        final_signal_dict = self._perform_relational_meta_analysis(df=df, snapshot_score=snapshot_score, signal_name=signal_name)
        states.update(final_signal_dict)
        return states

    def _diagnose_upper_shadow_intent(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】上影线意图诊断引擎
        - 核心使命: 揭示长上影线背后的真实意图——是主力派发（散户站岗），还是主力洗盘吸筹。
        - 诊断逻辑: 融合“资金流向”、“筹码结果”、“成本代价”三维证据，输出一个[-1, 1]的双极性分数。
                      正分代表看涨意图（吸筹），负分代表看跌意图（派发）。
        """
        states = {}
        signal_name = 'SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS'
        p_parent = get_params_block(self.strategy, 'kline_pattern_params', {})
        p = get_params_block(p_parent, 'upper_shadow_intent_params', {})
        if not get_param_value(p.get('enabled'), True):
            states[signal_name] = pd.Series(0.0, index=df.index, dtype=np.float32)
            return states
        norm_window = get_param_value(p.get('norm_window'), 55)
        weights = get_param_value(p.get('fusion_weights'), {})
        w_flow = get_param_value(weights.get('flow_divergence'), 0.5)
        w_conc = get_param_value(weights.get('concentration_change'), 0.3)
        w_profit = get_param_value(weights.get('profit_profile'), 0.2)
        # 1. 计算上影线比例，作为触发条件
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        upper_shadow_ratio = (upper_shadow / kline_range).fillna(0)
        # 仅在有显著上影线的日子进行诊断
        trigger_mask = upper_shadow_ratio > get_param_value(p.get('min_upper_shadow_ratio'), 0.4)
        # 2. 归一化三维证据
        # 证据A: 资金流向 (主力vs散户)
        flow_divergence_score = normalize_to_bipolar(df.get('flow_divergence_mf_vs_retail_D', 0), df.index, norm_window)
        # 证据B: 筹码结果 (集中度变化)
        concentration_change = df.get('concentration_90pct_D', pd.Series(0.0, index=df.index)).diff().fillna(0)
        concentration_change_score = normalize_to_bipolar(concentration_change, df.index, norm_window)
        # 证据C: 成本代价 (主力日内盈亏，注意取反)
        # 主力亏钱买入是看涨信号，所以对原始利润取反
        profit_profile_score = normalize_to_bipolar(-df.get('main_force_intraday_profit_D', 0), df.index, norm_window)
        # 3. 加权融合
        final_intent_score = (
            flow_divergence_score * w_flow +
            concentration_change_score * w_conc +
            profit_profile_score * w_profit
        ).clip(-1, 1)
        # 4. 应用触发条件
        states[signal_name] = (final_intent_score * trigger_mask).astype(np.float32)
        return states

    def _calculate_trend_health_score(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】“阿波罗的战车”四维趋势健康度评估引擎
        - 核心逻辑: 从“排列、速度、加速度、元动力”四个维度评估趋势的健康度。
                      1. 排列 (Alignment): 均线是否呈多头排列。
                      2. 速度 (Velocity): 均线趋势的强度 (周期匹配导数)。
                      3. 加速度 (Acceleration): 均线趋势的加速能力 (周期匹配二阶导数)。
                      4. 元动力 (Meta-Dynamics): 长期趋势的短期变化率 (跨周期导数)，用于捕捉拐点。
        """
        #
        norm_window = 55
        ma_periods = [5, 13, 21, 55, 89]
        ma_cols = [f'EMA_{p}_D' for p in ma_periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index, dtype=np.float32)
        ma_values = np.stack([df[col].values for col in ma_cols], axis=0)
        # 维度1: 排列健康度 (Alignment Health)
        alignment_bools = ma_values[:-1] > ma_values[1:]
        alignment_health = np.mean(alignment_bools, axis=0) if alignment_bools.size > 0 else np.full(len(df.index), 0.5)
        # 维度2: 速度健康度 (Velocity Health) - 周期匹配导数
        slope_cols = [f'SLOPE_{p}_EMA_{p}_D' for p in ma_periods]
        if all(col in df.columns for col in slope_cols):
            slope_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in slope_cols], axis=0)
            velocity_health = np.mean(slope_values, axis=0)
        else:
            velocity_health = np.full(len(df.index), 0.5)
        # 维度3: 加速度健康度 (Acceleration Health) - 周期匹配二阶导数
        accel_cols = [f'ACCEL_{p}_EMA_{p}_D' for p in ma_periods]
        if all(col in df.columns for col in accel_cols):
            accel_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in accel_cols], axis=0)
            acceleration_health = np.mean(accel_values, axis=0)
        else:
            acceleration_health = np.full(len(df.index), 0.5)
        # 维度4: 元动力健康度 (Meta-Dynamics Health) - 跨周期导数
        meta_dynamics_cols = [
            'SLOPE_5_EMA_55_D', 'SLOPE_13_EMA_89_D', 'SLOPE_21_EMA_144_D'
        ]
        if all(col in df.columns for col in meta_dynamics_cols):
            meta_values = np.stack([normalize_score(df[col], df.index, norm_window) for col in meta_dynamics_cols], axis=0)
            meta_dynamics_health = np.mean(meta_values, axis=0)
        else:
            meta_dynamics_health = np.full(len(df.index), 0.5)
        # 最终融合：加权几何平均
        weights = {'alignment': 0.3, 'velocity': 0.2, 'acceleration': 0.2, 'meta_dynamics': 0.3}
        scores = np.stack([alignment_health, velocity_health, acceleration_health, meta_dynamics_health], axis=0)
        weights_array = np.array(list(weights.values()))
        weights_array /= weights_array.sum()
        final_score_values = np.prod(scores ** weights_array[:, np.newaxis], axis=0)
        return pd.Series(final_score_values, index=df.index, dtype=np.float32)

    def _calculate_despair_context_score(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """
        【V3.2 · 前线换装版】“冥河之渡”多维绝望背景诊断引擎
        - 核心升级: 废弃 `auction_conviction_index_D`，换装为 `final_hour_momentum_D`
                      来量化“竞价确认恐慌”。
        """
        despair_periods = get_param_value(params.get('despair_periods'), {'short': (21, 5), 'mid': (60, 21), 'long': (250, 60)})
        despair_weights = get_param_value(params.get('despair_weights'), {'short': 0.2, 'mid': 0.3, 'long': 0.5})
        period_scores = []
        period_weight_values = []
        is_negative_day = df['pct_change_D'] < 0
        norm_window = 55
        # 换装新式武器
        panic_efficiency = normalize_score(df.get('intraday_trend_efficiency_D', pd.Series(0.0, index=df.index)), df.index, norm_window).where(is_negative_day, 0)
        closing_weakness = 1.0 - normalize_score(df.get('closing_strength_index_D', pd.Series(0.5, index=df.index)), df.index, norm_window)
        # 使用尾盘动能的负向部分代表竞价恐慌
        auction_panic = normalize_score(df.get('final_hour_momentum_D', pd.Series(0.0, index=df.index)).clip(upper=0).abs(), df.index, norm_window)

        panic_behavior_score = (panic_efficiency * closing_weakness * auction_panic)**(1/3)
        for name, (drawdown_period, roc_period) in despair_periods.items():
            rolling_peak = df['high_D'].rolling(window=drawdown_period, min_periods=max(1, drawdown_period//2)).max()
            drawdown_from_peak = (rolling_peak - df['close_D']) / rolling_peak.replace(0, np.nan)
            magnitude_score = normalize_score(drawdown_from_peak.clip(lower=0), df.index, window=drawdown_period, ascending=True)
            price_roc = df['close_D'].pct_change(roc_period)
            velocity_score = normalize_score(price_roc, df.index, window=drawdown_period, ascending=False)
            period_despair_score = (magnitude_score * velocity_score * panic_behavior_score)**(1/3)
            period_scores.append(period_despair_score.values)
            period_weight_values.append(despair_weights.get(name, 0.0))
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

    def _calculate_day_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】“赫尔墨斯敕令”核心裁决者
        - 核心职责: 提炼出的独立方法，专门用于计算单日K线的“质量分”。
                      该分数综合了“轨迹四象限”和“影线博弈”，是判断K线内在看涨/看跌属性的统一标准。
        - 返回值: 一个在[-1, 1]区间的Series，正值代表看涨质量，负值代表看跌质量。
        """
        gap_up = df['open_D'] > df['pre_close_D']
        gap_down = df['open_D'] < df['pre_close_D']
        body_up = df['close_D'] > df['open_D']
        body_down = df['close_D'] < df['open_D']
        trajectory_score = pd.Series(0.0, index=df.index)
        trajectory_score.loc[gap_up & body_up] = 1.0
        trajectory_score.loc[gap_down & body_up] = 0.8
        trajectory_score.loc[gap_up & body_down] = -0.8
        trajectory_score.loc[gap_down & body_down] = -1.0
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        lower_shadow = (np.minimum(df['open_D'], df['close_D']) - df['low_D']).clip(lower=0)
        shadow_modifier = ((lower_shadow - upper_shadow) / kline_range).fillna(0)
        day_quality_score = (trajectory_score * 0.7 + shadow_modifier * 0.3).clip(-1, 1)
        return day_quality_score

    def _fuse_upper_shadow_signals(self, provisional_risk: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】上影线信号融合器（风险调光器）
        - 核心使命: 应用“风险调光器”协议，融合原始抛压风险和意图诊断分。
        - 核心公式: 最终风险分 = 原始抛压风险分 * (1 - 上影线意图诊断分)
        - 产出: 生成最终的、经过上下文调制的 `SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW` 信号。
        """
        states = {}
        final_risk_score = (provisional_risk * (1 - intent_diagnosis)).clip(0, 2) # 允许风险翻倍
        # 为了计分系统的兼容性，最终输出到计分系统时可以再clip(0,1)
        # 但这里保留原始计算值，以便未来更精细的风险模型使用
        states['SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW'] = final_risk_score.astype(np.float32)
        return states





