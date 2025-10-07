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
        【V2.5 · 指挥链审查版】微观行为诊断引擎总指挥
        - 核心升级: 部署“指挥链审查”探针，确认本模块是否被成功调用。
        """
        # [代码新增] 指挥链审查探针 - 级别 3
        print("    -> [指挥链探针-3] MicroBehaviorEngine: run_micro_behavior_synthesis 已被调用。")
        all_states = {}
        def update_states(new_states: Dict[str, pd.Series]):
            if new_states:
                all_states.update(new_states)
        update_states(self.synthesize_early_momentum_ignition(df))
        update_states(self.diagnose_deceptive_retail_flow(df))
        update_states(self.synthesize_microstructure_dynamics(df))
        update_states(self.synthesize_euphoric_acceleration_risk(df))
        update_states(self.synthesize_post_peak_downturn_risk(df))
        early_ignition_score = all_states.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION', self._get_atomic_score(df, 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'))
        # [代码新增] 指挥链审查探针 - 级别 4
        print("    -> [指挥链探针-4] MicroBehaviorEngine: 即将调用 synthesize_reversal_reliability_score...")
        update_states(self.synthesize_reversal_reliability_score(
            df, early_ignition_score=early_ignition_score
        ))
        return all_states

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V9.0 · 关系元分析版】早期动能点火诊断模块
        """
        states = {}
        
        # 步骤一：计算原始的、纯粹的微观行为分数
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        body_size = (df['close_D'] - df['open_D']).clip(lower=0)
        body_strength_score = (body_size / candle_range).fillna(0.0)
        position_in_range_score = ((df['close_D'] - df['low_D']) / candle_range).fillna(0.0)
        momentum_strength_score = (df['pct_change_D'] / 0.10).clip(0, 1).fillna(0.0)
        raw_ignition_score = (body_strength_score * position_in_range_score * momentum_strength_score)

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        # 核心思想：只有在健康的均线结构下发生的点火，才是有效的点火。
        snapshot_score = raw_ignition_score * ma_context_score

        # 步骤四：对快照分进行关系元分析，得到最终的动态调制分数
        final_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        
        states['COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'] = final_score.astype(np.float32)
        return states

    def diagnose_deceptive_retail_flow(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 关系元分析版】伪装散户吸筹诊断引擎
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 步骤一：计算原始的、纯粹的微观行为分数
        retail_inflow_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BEARISH_RESONANCE')
        chip_concentration_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=False)
        price_suppression_score = normalize_score(df.get('SLOPE_5_close_D').abs(), df.index, norm_window, ascending=False)
        vpa_inefficiency_score = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)
        raw_deception_score = (
            retail_inflow_score * chip_concentration_score *
            price_suppression_score * vpa_inefficiency_score
        )

        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])

        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        # 核心思想：伪装吸筹在均线结构“看起来很差”的时候进行，才最具欺骗性和威力。
        snapshot_score = raw_deception_score * (1 - ma_context_score)

        # 步骤四：对快照分进行关系元分析，得到最终的动态调制分数
        final_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION'] = final_score.astype(np.float32)
        return states


    def synthesize_reversal_reliability_score(self, df: pd.DataFrame, early_ignition_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V5.1 · 放大器侦测版】高质量战备可靠性诊断引擎
        - 核心升级: 部署“放大器侦测探针”，在目标日期打印出底部反转可靠性分数的放大过程。
        """
        # [代码新增] 指挥链审查探针 - 级别 5
        print("    -> [指挥链探针-5] MicroBehaviorEngine: synthesize_reversal_reliability_score 已被调用。")
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True):
            # [代码新增] 指挥链审查探针 - 级别 5.1 (配置禁用)
            print("    -> [指挥链探针-5.1] MicroBehaviorEngine: synthesize_reversal_reliability_score 因配置禁用而退出。")
            return states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        norm_window = get_param_value(p.get('norm_window'), 120)
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        rsi_w_oversold_score = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
        background_score = np.maximum(deep_bottom_context_score, rsi_w_oversold_score).astype(np.float32)
        states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = background_score
        chip_accumulation_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        chip_reversal_score = get_unified_score(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL')
        conviction_strengthening_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING')
        shareholder_turnover_score = np.maximum.reduce([
            chip_accumulation_score.values, chip_reversal_score.values, conviction_strengthening_score.values
        ])
        shareholder_quality_score = pd.Series(shareholder_turnover_score, index=df.index, dtype=np.float32)
        states['SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT'] = shareholder_quality_score
        fft_trend_score = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT', 0.0)
        fft_trend_slope = fft_trend_score.diff(5).fillna(0)
        trend_potential_score = normalize_score(fft_trend_slope.clip(lower=0), df.index, window=norm_window, ascending=True, default_value=0.0)
        states['INTERNAL_SCORE_TREND_POTENTIAL'] = trend_potential_score.astype(np.float32)
        vol_compression_score = get_unified_score(self.strategy.atomic_states, df.index, 'VOL_COMPRESSION')
        ignition_weights = get_param_value(p.get('ignition_weights'), {'early': 0.5, 'vol': 0.2, 'potential': 0.3})
        if len(early_ignition_score) != len(df.index):
            early_ignition_score = early_ignition_score.reindex(df.index, fill_value=0.0)
        ignition_confirmation_score = (
            (early_ignition_score ** ignition_weights['early']) *
            (vol_compression_score ** ignition_weights['vol']) *
            (trend_potential_score ** ignition_weights['potential'])
        ).astype(np.float32)
        states['SCORE_IGNITION_CONFIRMATION'] = ignition_confirmation_score
        main_reliability_weights = get_param_value(p.get('main_reliability_weights'), {'shareholder': 0.5, 'ignition': 0.5})
        main_score = (
            (shareholder_quality_score ** main_reliability_weights['shareholder']) *
            (ignition_confirmation_score ** main_reliability_weights['ignition'])
        )
        bonus_factor = get_param_value(p.get('reversal_reliability_bonus_factor'), 0.5)
        raw_reliability_score = (main_score * (1 + background_score * bonus_factor)).clip(0, 1)
        # [代码新增] 部署“放大器侦测探针”
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d) for d in probe_dates_str]
        for date in probe_dates:
            if date in df.index and date.date() == pd.to_datetime('2025-09-17').date():
                print(f"\n      -> [放大器侦测探针 @ {date.date()}] 信号: COGNITIVE_SCORE_REVERSAL_RELIABILITY")
                print(f"         - 基础分 (main_score): {main_score.loc[date]:.4f}")
                print(f"         - 上下文分 (background_score): {background_score.loc[date]:.4f}")
                print(f"         - 奖励因子 (bonus_factor): {bonus_factor:.2f}")
                print(f"         - 放大公式: Base * (1 + Context * Factor)")
                print(f"         - 计算过程: {main_score.loc[date]:.4f} * (1 + {background_score.loc[date]:.4f} * {bonus_factor:.2f})")
                print(f"         - 放大后分数 (raw_reliability_score): {raw_reliability_score.loc[date]:.4f}")
                print(f"         - 贡献增量 (Amplification): {(raw_reliability_score.loc[date] - main_score.loc[date]):.4f}\n")
        snapshot_score = raw_reliability_score * self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        final_reliability_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        states['COGNITIVE_SCORE_REVERSAL_RELIABILITY'] = final_reliability_score.astype(np.float32)
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 关系元分析版】市场微观结构动态诊断引擎
        """
        states = {}
        norm_window = 120
        # 获取均线趋势上下文分数，供所有子信号使用
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # --- 看涨信号计算 (已在上一轮改造) ---
        granularity_momentum_up = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=True)
        dominance_momentum_up = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=True)
        granularity_holo_up, _ = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
        dominance_holo_up, _ = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
        raw_power_shift_score = (granularity_momentum_up * granularity_holo_up * dominance_momentum_up * dominance_holo_up)
        snapshot_power_shift = raw_power_shift_score * ma_context_score
        final_power_shift_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_power_shift)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = final_power_shift_score.astype(np.float32)
        conviction_momentum_strengthening = normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=True)
        conviction_holo_up, _ = calculate_holographic_dynamics(df, 'main_force_conviction_ratio', norm_window)
        raw_conviction_score = (conviction_momentum_strengthening * conviction_holo_up)
        snapshot_conviction = raw_conviction_score * ma_context_score
        final_conviction_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_conviction)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = final_conviction_score.astype(np.float32)
        # --- 看跌风险信号计算 (应用关系元分析) ---
        recent_reversal_context = self._get_atomic_score(df, 'SCORE_CONTEXT_RECENT_REVERSAL', 0.0)
        risk_suppression_factor = (1.0 - recent_reversal_context).clip(0, 1)
        # 1. COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL
        granularity_momentum_down = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=False)
        dominance_momentum_down = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=False)
        _, granularity_holo_down = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
        _, dominance_holo_down = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
        power_shift_to_retail_risk_raw = (granularity_momentum_down * granularity_holo_down * dominance_momentum_down * dominance_holo_down)
        # 构建关系快照分：权力向散户转移的风险，在均线结构恶化时最危险
        snapshot_power_shift_risk = power_shift_to_retail_risk_raw * (1 - ma_context_score)
        # 对关系进行元分析
        final_power_shift_risk = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_power_shift_risk)
        # 应用抑制力场
        power_shift_to_retail_risk = (final_power_shift_risk * risk_suppression_factor).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = power_shift_to_retail_risk
        # 2. COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING
        conviction_momentum_weakening = normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=False)
        _, conviction_holo_down = calculate_holographic_dynamics(df, 'main_force_conviction_ratio', norm_window)
        conviction_weakening_risk_raw = (conviction_momentum_weakening * conviction_holo_down)
        # 构建关系快照分：主力信念瓦解的风险，在均线结构恶化时最危险
        snapshot_conviction_risk = conviction_weakening_risk_raw * (1 - ma_context_score)
        # 对关系进行元分析
        final_conviction_risk = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_conviction_risk)
        # 应用抑制力场
        conviction_weakening_risk = (final_conviction_risk * risk_suppression_factor).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = conviction_weakening_risk
        return states

    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 关系元分析版】亢奋加速风险诊断引擎
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)
        epsilon = 1e-9
        # 步骤一：计算原始的、纯粹的亢奋风险分数
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
        bias_score = normalize_score(df['BIAS_21_D'].abs(), df.index, norm_window, ascending=True)
        volume_ratio = (df['volume_D'] / (df.get('VOL_MA_55_D', df['volume_D']) + epsilon)).fillna(1.0)
        volume_spike_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        atr_ratio = (df['ATR_14_D'] / (df['close_D'] + epsilon)).fillna(0.0)
        volatility_score = normalize_score(atr_ratio, df.index, norm_window, ascending=True)
        total_range = (df['high_D'] - df['low_D']).replace(0, epsilon)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upthrust_score = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
        raw_risk_factors = (bias_score * volume_spike_score * volatility_score * upthrust_score)**(1/4)
        raw_euphoric_risk_score = (raw_risk_factors * stretch_from_ma55_score * (1 - safe_launch_context_score))
        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        # 核心思想：亢奋风险发生在健康的均线结构背景下，代表“盛极而衰”的转折风险。
        snapshot_score = raw_euphoric_risk_score * ma_context_score
        # 步骤四：对快照分进行关系元分析，得到最终的动态调制分数
        final_risk_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score.astype(np.float32)
        return states

    def synthesize_post_peak_downturn_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V3.0 · 关系元分析版】高位回落风险 (Post-Peak Downturn Risk) 诊断引擎
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'post_peak_downturn_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)
        high_position_threshold = get_param_value(p_risk.get('high_position_threshold'), 0.7)
        peak_echo_window = get_param_value(p_risk.get('peak_echo_window'), 5)
        # 步骤一：计算原始的、纯粹的高位回落风险分数
        ma55 = df.get('EMA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, np.nan)
        stretch_from_ma55_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        is_at_high_position = (stretch_from_ma55_score > high_position_threshold)
        recently_at_peak_context = is_at_high_position.rolling(window=peak_echo_window, min_periods=1).max().astype(float)
        is_falling_today = (df['pct_change_D'] < 0).astype(float)
        fall_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
        fall_magnitude_score = normalize_score(fall_magnitude, df.index, norm_window, ascending=True)
        volume_ratio = (df['volume_D'] / df.get('VOL_MA_21_D', df['volume_D'])).fillna(1.0)
        volume_spike_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        ema5 = df.get('EMA_5_D', df['close_D'])
        breakdown_depth_pct = ((ema5 - df['close_D']) / ema5).clip(lower=0).fillna(0)
        break_ema5_score = normalize_score(breakdown_depth_pct, df.index, norm_window, ascending=True)
        severity_score = (fall_magnitude_score * volume_spike_score * break_ema5_score)**(1/3)
        raw_downturn_risk_score = (recently_at_peak_context * is_falling_today * severity_score)
        # 步骤二：获取均线趋势上下文分数
        ma_context_score = self._calculate_ma_trend_context(df, [5, 13, 21, 55])
        # 步骤三：构建融合了趋势上下文的“瞬时关系快照分”
        # 核心思想：高位回落的风险，在均线结构也开始恶化时最为致命。
        snapshot_score = raw_downturn_risk_score * (1 - ma_context_score)
        # 步骤四：对快照分进行关系元分析，得到最终的动态调制分数
        final_risk_score = self._perform_micro_behavior_relational_meta_analysis(df, snapshot_score)
        states['COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN'] = final_risk_score.astype(np.float32)
        return states

    def _perform_micro_behavior_relational_meta_analysis(self, df: pd.DataFrame, snapshot_score: pd.Series) -> pd.Series:
        """
        【V1.0 · 新增】微观行为专用的关系元分析核心引擎 (赫拉织布机V2)
        - 核心逻辑: 实现“状态 * (1 + 动态杠杆)”的动态价值调制范式。
        """
        # 从配置中获取动态杠杆权重
        p_conf = get_params_block(self.strategy, 'micro_behavior_params', {})
        p_meta = get_param_value(p_conf.get('relational_meta_analysis_params'), {})
        w_velocity = get_param_value(p_meta.get('velocity_weight'), 0.6)
        w_acceleration = get_param_value(p_meta.get('acceleration_weight'), 0.4)

        # 核心参数
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
        dynamic_leverage = 1 + (velocity_score * w_velocity) + (acceleration_score * w_acceleration)
        final_score = (state_score * dynamic_leverage).clip(0, 1)
        
        return final_score.astype(np.float32)

    def _calculate_ma_trend_context(self, df: pd.DataFrame, periods: list) -> pd.Series:
        """
        【V1.0 · 新增】计算均线趋势上下文分数
        - 核心逻辑: 评估短期、中期、长期均线的排列和价格位置，输出一个统一的趋势健康分。
        """
        # 确保所有需要的均线都存在
        ma_cols = [f'EMA_{p}_D' for p in periods]
        if not all(col in df.columns for col in ma_cols):
            return pd.Series(0.5, index=df.index)

        # 均线排列健康度
        alignment_scores = []
        for i in range(len(periods) - 1):
            short_ma = df[f'EMA_{periods[i]}_D']
            long_ma = df[f'EMA_{periods[i+1]}_D']
            alignment_scores.append((short_ma > long_ma).astype(float))
        
        alignment_health = np.mean(alignment_scores, axis=0) if alignment_scores else np.full(len(df.index), 0.5)

        # 价格位置健康度 (价格应在所有均线之上)
        position_scores = [(df['close_D'] > df[col]).astype(float) for col in ma_cols]
        position_health = np.mean(position_scores, axis=0) if position_scores else np.full(len(df.index), 0.5)

        # 融合得到最终的趋势上下文分数
        ma_context_score = pd.Series((alignment_health * position_health)**0.5, index=df.index)
        return ma_context_score.astype(np.float32)




