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
        【V28.1 · 状态隔离版】行为终极信号诊断引擎
        - 核心修复: 将复杂的 overall_health 字典存入一个内部专用的键名，防止其污染通用的 atomic_states 存储空间，
                      从而解决了下游模块因数据类型不匹配而崩溃的问题。
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
        # 将复杂的字典存入一个内部专用的键，避免污染
        self.strategy.atomic_states['_internal_behavior_health_dict'] = overall_health
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
        【V2.2 · 终极净化版】原子信号中心
        - 核心重构: 彻底移除了对已被废弃的 `_fuse_upper_shadow_signals` 的调用，消除了逻辑冗余。
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
        # 实施全链路逻辑集权，并移除冗余调用
        # 步骤1: 生成临时的广义抛压信号
        provisional_kline_signals = self._diagnose_kline_patterns(df, day_quality_score)
        provisional_pressure = provisional_kline_signals.get('PROVISIONAL_GENERAL_PRESSURE_RISK', pd.Series(0.0, index=df.index))
        atomic_signals.update(provisional_kline_signals)
        # 步骤2: 生成意图诊断信号
        intent_signals = self._diagnose_upper_shadow_intent(df)
        intent_diagnosis = intent_signals.get('SCORE_UPPER_SHADOW_INTENT_DIAGNOSIS', pd.Series(0.0, index=df.index))
        atomic_signals.update(intent_signals)
        # 步骤3: 调用嬗变引擎，生成最终的风险和机会信号
        final_pressure_signals = self._transmute_pressure_into_opportunity(provisional_pressure, intent_diagnosis)
        atomic_signals.update(final_pressure_signals)
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
        【V3.8 · 广义抛压版】诊断K线原子形态
        - 核心升级: 从 foundation_intelligence 模块接管了“广义抛压”的计算逻辑。
                      现在负责生成一个更全面的 `PROVISIONAL_GENERAL_PRESSURE_RISK` 临时信号。
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
        # 实施广义抛压计算逻辑
        p_pressure = p.get('selling_pressure_params', {})
        signal_name = 'PROVISIONAL_GENERAL_PRESSURE_RISK'
        if get_param_value(p_pressure.get('enabled'), True):
            periods = get_param_value(p_pressure.get('periods'), [5, 13, 21, 55])
            sorted_periods = sorted(periods)
            pressure_scores = {}
            # 广义抛压证据1: 上影线
            kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
            upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
            upper_shadow_ratio = (upper_shadow / kline_range).fillna(0)
            # 广义抛压证据2: 价跌
            candle_body_down = (df['open_D'] - df['close_D']).clip(lower=0)
            for p_tactical in sorted_periods:
                # 归一化上影线和价跌
                tactical_shadow = normalize_score(upper_shadow_ratio, df.index, p_tactical, ascending=True)
                tactical_price_down = normalize_score(candle_body_down, df.index, p_tactical, ascending=True)
                # 融合价跌和上影线作为抛压组件
                selling_pressure_component = np.maximum(tactical_shadow, tactical_price_down)
                # 归一化成交量
                tactical_volume = normalize_score(df['volume_D'], df.index, p_tactical, ascending=True)
                # 融合抛压和成交量
                pressure_scores[p_tactical] = (selling_pressure_component * tactical_volume)**0.5
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
        # 实施“雅典娜敕令”，在源头过滤风险驱动因子
        is_bearish_quality_day_mask = (day_quality_score <= 0).astype(int)
        # 只有在K线质量为负的日子里，才计算收盘疲弱分
        weak_close_score = (1.0 - normalize_score(df.get('closing_strength_index_D', 0.5), df.index, 55)) * is_bearish_quality_day_mask
        
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
        # 实施“雅典娜敕令”
        is_bearish_quality_day_mask = (day_quality_score <= 0).astype(int)
        
        for i, p_tactical in enumerate(sorted_periods):
            p_context = sorted_periods[i + 1] if i + 1 < len(sorted_periods) else p_tactical
            tactical_volume_ratio = (df['volume_D'] / df[f'VOL_MA_{p_tactical}_D'].replace(0, np.nan)).fillna(1.0) if f'VOL_MA_{p_tactical}_D' in df else pd.Series(1.0, index=df.index)
            tactical_huge_volume = normalize_score(tactical_volume_ratio, df.index, p_tactical, ascending=True)
            # 在源头过滤风险驱动因子
            tactical_low_efficiency = 1 - normalize_score(df.get('intraday_trend_efficiency_D', 0.5), df.index, p_tactical, ascending=True)
            tactical_high_volatility = normalize_score(df.get('intraday_volatility_D', 0.0), df.index, p_tactical, ascending=True)
            # 只有在K线质量为负的日子里，才计算价格滞涨分
            tactical_price_stagnant = ((tactical_low_efficiency * tactical_high_volatility)**0.5) * is_bearish_quality_day_mask
            
            context_volume_ratio = (df['volume_D'] / df[f'VOL_MA_{p_context}_D'].replace(0, np.nan)).fillna(1.0) if f'VOL_MA_{p_context}_D' in df else pd.Series(1.0, index=df.index)
            context_huge_volume = normalize_score(context_volume_ratio, df.index, p_context, ascending=True)
            # 在源头过滤风险驱动因子
            context_low_efficiency = 1 - normalize_score(df.get('intraday_trend_efficiency_D', 0.5), df.index, p_context, ascending=True)
            context_high_volatility = normalize_score(df.get('intraday_volatility_D', 0.0), df.index, p_context, ascending=True)
            # 只有在K线质量为负的日子里，才计算价格滞涨分
            context_price_stagnant = ((context_low_efficiency * context_high_volatility)**0.5) * is_bearish_quality_day_mask
            
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

    def _diagnose_trend_context(self, df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】行为层专属的趋势上下文诊断引擎
        - 核心逻辑: 基于关键移动平均线的位置和斜率，生成一个[-1, 1]区间的双极性趋势分数。
                      +1 代表明确的上升趋势，-1 代表明确的下降趋势，0 附近代表震荡或趋势不明。
        - 输出: 一个代表当前趋势方向和强度的双极性分数。
        """
        p = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        trend_ma_period = get_param_value(p.get('trend_context_ma_period'), 55)
        ma_col = f'EMA_{trend_ma_period}_D'
        slope_col = f'SLOPE_5_{ma_col}' # 使用5日斜率判断短期趋势方向
        if ma_col not in df.columns or slope_col not in df.columns:
            return pd.Series(0.0, index=df.index, dtype=np.float32)
        # 证据一：价格是否在趋势线上方
        price_above_ma = (df['close_D'] > df[ma_col]).astype(float) * 2 - 1 # 转换为[-1, 1]
        # 证据二：趋势线斜率方向
        ma_slope = df[slope_col]
        slope_direction = normalize_to_bipolar(ma_slope, df.index, window=trend_ma_period)
        # 融合：位置权重60%，斜率权重40%
        trend_context_score = (price_above_ma * 0.6 + slope_direction * 0.4)
        return trend_context_score.clip(-1, 1).astype(np.float32)

    def _diagnose_price_volume_atomics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V13.7 · 趋势上下文版】量价原子信号诊断引擎
        - 核心升级 (流动性真空风险):
          1. [数据加固] 废弃不稳定的 `intraday_volatility_D`，改用 `(high-low)/pre_close` 计算的真实波动率。
          2. [逻辑升维] 引入趋势上下文判断。流动性真空风险现在只在下降或震荡趋势中被激活。
                         在明确的上升趋势中，该风险将被抑制，因为缩量和低换手此时是积极信号。
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
        norm_window = get_param_value(p.get('norm_window'), 55)
        # --- 流动性真空风险 V2.4 (趋势上下文版) ---
        low_turnover_score = normalize_score(df.get('turnover_rate_D', pd.Series(0.0, index=df.index)), df.index, norm_window, ascending=False)
        vol_ratio = df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D']).replace(0, np.nan)
        sustained_shrink_score = normalize_score(vol_ratio.fillna(1.0), df.index, norm_window, ascending=False)
        calculated_volatility = (df['high_D'] - df['low_D']) / df['pre_close_D'].replace(0, np.nan)
        fragility_score = normalize_score(calculated_volatility.fillna(0.0), df.index, norm_window, ascending=True)
        liquidity_vacuum_snapshot = (low_turnover_score * sustained_shrink_score * fragility_score)**(1/3)
        trend_context_score = self._diagnose_trend_context(df)
        # [代码修改开始]
        # 风险抑制因子：只有在明确的上升趋势中(score > 0)，风险才被抑制。下降或震荡趋势中(score <= 0)，风险被完全表达。
        risk_suppression_factor = 1.0 - trend_context_score.clip(lower=0)
        # [代码修改结束]
        final_liquidity_vacuum_risk = liquidity_vacuum_snapshot * risk_suppression_factor
        states['SCORE_RISK_LIQUIDITY_VACUUM'] = final_liquidity_vacuum_risk.clip(0, 1).astype(np.float32)
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
        【V2.1 · 阿瑞斯之矛版】上影线意图诊断引擎
        - 核心升级: 签署“阿瑞斯之矛”协议，聚焦核心意图，剔除次要证据的干扰。
                      1. [精简证据]: 移除“筹码结果分(concentration_change)”这一噪声较大的证据。
                      2. [聚焦核心]: 诊断逻辑完全聚焦于“主力净流向”和“主力成本代价”两大核心意图指标。
                      3. [重配权重]: 将权重集中于两大核心证据，使诊断更果断、更精准。
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
        # 实施“阿瑞斯之矛”协议，更新权重和证据
        w_flow = get_param_value(weights.get('main_force_flow'), 0.6)
        w_profit = get_param_value(weights.get('profit_profile'), 0.4)
        kline_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D'])).clip(lower=0)
        upper_shadow_ratio = (upper_shadow / kline_range).fillna(0)
        trigger_mask = upper_shadow_ratio > get_param_value(p.get('min_upper_shadow_ratio'), 0.4)
        # 实施“阿瑞斯之矛”协议，精简证据链
        # 证据A: 主力净流向 (绝对意图)
        main_force_flow_score = normalize_to_bipolar(df.get('main_force_net_flow_consensus_D', 0), df.index, norm_window)
        # 证据B: 成本代价 (信念强度)
        profit_profile_score = normalize_to_bipolar(-df.get('main_force_intraday_profit_D', 0), df.index, norm_window)
        # 加权融合
        final_intent_score = (
            main_force_flow_score * w_flow +
            profit_profile_score * w_profit
        ).clip(-1, 1)
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
        【V4.1 · 加速度修复版】行为层专用的关系元分析核心引擎
        - 核心修复: 修正了“加速度”计算的致命逻辑错误。加速度是速度的一阶导数，
                      因此其计算应为 relationship_trend.diff(1)，而不是错误的 diff(meta_window)。
        """
        states = {}
        p_conf = get_params_block(self.strategy, 'behavioral_dynamics_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_state = get_param_value(p_meta.get('state_weight'), 0.3)
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.3)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)
        norm_window = 55
        meta_window = 5
        bipolar_sensitivity = 1.0
        bipolar_snapshot = (snapshot_score * 2 - 1).clip(-1, 1)
        relationship_trend = bipolar_snapshot.diff(meta_window).fillna(0)
        velocity_score = normalize_to_bipolar(
            series=relationship_trend, target_index=df.index,
            window=norm_window, sensitivity=bipolar_sensitivity
        )
        # [代码修改开始]
        # 致命错误修复：加速度是速度(trend)的一阶导数，应使用 diff(1) 而不是 diff(meta_window)
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
        final_score = (total_bullish_force - total_bearish_force).clip(-1, 1)
        unipolar_score = (final_score + 1) / 2.0
        states[signal_name] = unipolar_score.astype(np.float32)
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

    def _transmute_pressure_into_opportunity(self, provisional_pressure: pd.Series, intent_diagnosis: pd.Series) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】抛压嬗变引擎
        - 核心使命: 融合临时的广义抛压分和权威的主力意图诊断分，完成从风险到机会的最终嬗变。
        - 核心公式: 1. 若意图分 > 阈值，则风险归零，创造机会分。
                      2. 否则，风险 = 原始抛压 * (1 - 意图分)。
        - 产出: 生成最终的 `SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW` 和 `SCORE_OPPORTUNITY_ABSORPTION_REVERSAL`。
        """
        states = {}
        p_parent = get_params_block(self.strategy, 'kline_pattern_params', {})
        p_reversal = get_params_block(p_parent, 'absorption_reversal_params', {})
        judgment_threshold = get_param_value(p_reversal.get('judgment_threshold'), 0.7)
        is_absorption_reversal = intent_diagnosis >= judgment_threshold
        # 计算最终风险
        final_risk_array = np.where(
            is_absorption_reversal,
            0.0,
            (provisional_pressure * (1 - intent_diagnosis))
        )
        # 计算机会分
        opportunity_score_array = np.where(
            is_absorption_reversal,
            provisional_pressure * intent_diagnosis,
            0.0
        )
        # 统一命名最终的抛压风险信号
        states['SCORE_RISK_SELLING_PRESSURE_UPPER_SHADOW'] = pd.Series(final_risk_array, index=provisional_pressure.index).clip(0, 1).astype(np.float32)
        states['SCORE_OPPORTUNITY_ABSORPTION_REVERSAL'] = pd.Series(opportunity_score_array, index=provisional_pressure.index).clip(0, 1).astype(np.float32)
        return states




