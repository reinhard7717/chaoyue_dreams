# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, get_unified_score, calculate_holographic_dynamics

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
        【V2.4 · 净化协议版】微观行为诊断引擎总指挥
        - 核心革命: 移除了子引擎内部对 self.strategy.atomic_states 的直接写入操作。
                      现在引擎遵循“纯函数”原则，只负责计算并返回结果，将状态更新的权力
                      完全交还给上层调用者，彻底解决了“越权写入”导致的“状态污染”问题。
        """
        all_states = {}
        # [代码修改] 简化了 update_states 辅助函数，只更新局部字典 all_states
        def update_states(new_states: Dict[str, pd.Series]):
            if new_states:
                all_states.update(new_states)
                # [代码删除] 移除了对 self.strategy.atomic_states 的直接写入，这是非法的“越权”行为
                # self.strategy.atomic_states.update(new_states)
        update_states(self.synthesize_early_momentum_ignition(df))
        update_states(self.diagnose_deceptive_retail_flow(df))
        update_states(self.synthesize_microstructure_dynamics(df))
        update_states(self.synthesize_euphoric_acceleration_risk(df))
        update_states(self.synthesize_post_peak_downturn_risk(df))
        # [代码新增] 为了让下游的 synthesize_reversal_reliability_score 能消费到最新的信号，
        # 我们需要临时将当前计算出的状态合并到 df 中，或者直接传入。这里选择传入。
        early_ignition_score = all_states.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION', self._get_atomic_score(df, 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'))
        update_states(self.synthesize_reversal_reliability_score(
            df, early_ignition_score=early_ignition_score
        ))
        return all_states

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.1 · 信号净化版】早期动能点火诊断模块
        """
        # print("        -> [早期动能点火诊断模块 V8.1 · 信号净化版] 启动...") # 更新版本号
        states = {}
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        body_size = (df['close_D'] - df['open_D']).clip(lower=0)
        body_strength_score = (body_size / candle_range).fillna(0.0)
        position_in_range_score = ((df['close_D'] - df['low_D']) / candle_range).fillna(0.0)
        momentum_strength_score = (df['pct_change_D'] / 0.10).clip(0, 1).fillna(0.0)
        final_score = (body_strength_score * position_in_range_score * momentum_strength_score).astype(np.float32)
        # 移除信号名中的 '_A' 后缀
        states['COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION'] = final_score
        return states

    def diagnose_deceptive_retail_flow(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.3 · 信号净化版】伪装散户吸筹诊断引擎
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = get_param_value(p.get('norm_window'), 120)
        retail_inflow_score = get_unified_score(self.strategy.atomic_states, df.index, 'FF_BEARISH_RESONANCE')
        
        chip_concentration_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=False)
        price_suppression_score = normalize_score(df.get('SLOPE_5_close_D').abs(), df.index, norm_window, ascending=False)
        vpa_inefficiency_score = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)
        
        final_score = (
            retail_inflow_score * chip_concentration_score *
            price_suppression_score * vpa_inefficiency_score
        ).astype(np.float32)
        
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION'] = final_score
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V7.2 · 回声版】市场微观结构动态诊断引擎
        - 核心升级: 引入“风险抑制力场”。在“反转回声”的持续时间内，主动抑制“主力信念瓦解”等潜在的误判风险信号。
        """
        states = {}
        norm_window = 120
        
        # 获取“反转回声”信号，并创建抑制因子
        recent_reversal_context = self._get_atomic_score(df, 'SCORE_CONTEXT_RECENT_REVERSAL', 0.0)
        risk_suppression_factor = (1.0 - recent_reversal_context).clip(0, 1)

        # --- 看涨信号计算 (保持不变) ---
        granularity_momentum_up = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=True)
        dominance_momentum_up = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=True)
        granularity_holo_up, _ = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
        dominance_holo_up, _ = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
        power_shift_to_main_force_score = (granularity_momentum_up * granularity_holo_up * dominance_momentum_up * dominance_holo_up).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = power_shift_to_main_force_score
        
        conviction_momentum_strengthening = normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=True)
        conviction_holo_up, _ = calculate_holographic_dynamics(df, 'main_force_conviction_ratio', norm_window)
        conviction_strengthening_opp = (conviction_momentum_strengthening * conviction_holo_up).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = conviction_strengthening_opp

        # --- 看跌风险信号计算 (应用抑制力场) ---
        granularity_momentum_down = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=False)
        dominance_momentum_down = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=False)
        _, granularity_holo_down = calculate_holographic_dynamics(df, 'avg_order_value', norm_window)
        _, dominance_holo_down = calculate_holographic_dynamics(df, 'trade_concentration_index', norm_window)
        power_shift_to_retail_risk_raw = (granularity_momentum_down * granularity_holo_down * dominance_momentum_down * dominance_holo_down)
        
        # 应用抑制力场
        power_shift_to_retail_risk = (power_shift_to_retail_risk_raw * risk_suppression_factor).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = power_shift_to_retail_risk
        
        conviction_momentum_weakening = normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=False)
        _, conviction_holo_down = calculate_holographic_dynamics(df, 'main_force_conviction_ratio', norm_window)
        conviction_weakening_risk_raw = (conviction_momentum_weakening * conviction_holo_down)
        
        # 应用抑制力场
        conviction_weakening_risk = (conviction_weakening_risk_raw * risk_suppression_factor).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = conviction_weakening_risk
        
        return states

    def synthesize_reversal_reliability_score(self, df: pd.DataFrame, early_ignition_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V4.9 · 终极哲学统一版】高质量战备可靠性诊断引擎
        - 核心修复: 1. 将所有内部融合逻辑从“加权求和”彻底修改为“加权几何平均”。
                      2. 增加.clip(0, 1)确保最终分数不会因奖励因子而突破上限。
        """
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
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
            chip_accumulation_score.values,
            chip_reversal_score.values,
            conviction_strengthening_score.values
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

        # 将点火确认分的加权求和改为加权几何平均
        ignition_confirmation_score = (
            (early_ignition_score ** ignition_weights['early']) *
            (vol_compression_score ** ignition_weights['vol']) *
            (trend_potential_score ** ignition_weights['potential'])
        ).astype(np.float32)
        states['SCORE_IGNITION_CONFIRMATION'] = ignition_confirmation_score
        
        main_reliability_weights = get_param_value(p.get('main_reliability_weights'), {'shareholder': 0.5, 'ignition': 0.5})
        # 将主分数的加权求和改为加权几何平均
        main_score = (
            (shareholder_quality_score ** main_reliability_weights['shareholder']) *
            (ignition_confirmation_score ** main_reliability_weights['ignition'])
        )
        bonus_factor = get_param_value(p.get('reversal_reliability_bonus_factor'), 0.5)
        # 增加.clip(0, 1)确保最终分数不会因奖励因子而突破上限
        final_reliability_score = (main_score * (1 + background_score * bonus_factor)).clip(0, 1).astype(np.float32)
        states['COGNITIVE_SCORE_REVERSAL_RELIABILITY'] = final_reliability_score
        
        return states

    # “亢奋加速风险”诊断引擎
    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.2 · 防御性编程版】亢奋加速风险诊断引擎
        - 核心修复: 对所有除法操作增加epsilon(1e-9)保护，彻底杜绝因“除以零”产生inf值的可能性。
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)
        epsilon = 1e-9 # 定义一个极小值用于防止除以零

        # --- 步骤 1: 【智能上下文】计算以MA55为基准的“波段伸展度” ---
        ma55 = df.get('EMA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        # 增加epsilon保护
        wave_channel_height = (rolling_high_55d - ma55).replace(0, epsilon)
        stretch_from_ma55_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)

        # --- 步骤 2: 【安全港豁免】判断是否处于“主升浪启动区” ---
        ma55_is_rising = (ma55 > ma55.shift(3)).astype(float)
        
        bias_55d = df.get('BIAS_55_D', pd.Series(0.5, index=df.index))
        price_is_near_ma55 = (bias_55d.abs() < 0.15).astype(float)
        bbw_d = df.get('BBW_21_2.0_D', pd.Series(0.5, index=df.index))
        volatility_was_low = (bbw_d.shift(1) < bbw_d.rolling(60).quantile(0.3)).astype(float)
        safe_launch_context_score = (ma55_is_rising * price_is_near_ma55 * volatility_was_low)

        # --- 步骤 3: 计算原始风险因子 ---
        bias_score = normalize_score(df['BIAS_21_D'].abs(), df.index, norm_window, ascending=True)
        # 增加epsilon保护
        volume_ratio = (df['volume_D'] / (df.get('VOL_MA_55_D', df['volume_D']) + epsilon)).fillna(1.0)
        volume_spike_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        # 增加epsilon保护
        atr_ratio = (df['ATR_14_D'] / (df['close_D'] + epsilon)).fillna(0.0)
        volatility_score = normalize_score(atr_ratio, df.index, norm_window, ascending=True)
        # 增加epsilon保护
        total_range = (df['high_D'] - df['low_D']).replace(0, epsilon)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upthrust_score = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
        raw_risk_score = (bias_score * volume_spike_score * volatility_score * upthrust_score)**(1/4)

        # --- 步骤 4: 最终风险裁定 ---
        final_risk_score = (raw_risk_score * stretch_from_ma55_score * (1 - safe_launch_context_score)).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score
        
        return states

    # “高位回落风险”诊断引擎
    def synthesize_post_peak_downturn_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.0 · 新增】高位回落风险 (Post-Peak Downturn Risk) 诊断引擎
        - 核心职责: 识别股价在经历一轮上涨到达高位后，开始回落的风险。
        - 算法:
          1. 上下文: 前一日处于波段高位。
          2. 触发器: 当日股价下跌。
          3. 严重性: 结合下跌幅度、成交量放大程度、是否跌破短期均线等因素。
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'post_peak_downturn_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)

        # --- 步骤 1: 上下文 - 确认前一日处于高位 ---
        ma55 = df.get('EMA_55_D', df['close_D'])
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        wave_channel_height = (rolling_high_55d - ma55).replace(0, np.nan)
        stretch_from_ma55_score = ((df['close_D'] - ma55) / wave_channel_height).clip(0, 1).fillna(0.5)
        
        high_pos_threshold = get_param_value(p_risk.get('high_position_threshold'), 0.7)
        was_in_high_position_yesterday = (stretch_from_ma55_score.shift(1) > high_pos_threshold).astype(float)

        # --- 步骤 2: 触发器 - 确认当日正在下跌 ---
        is_falling_today = (df['pct_change_D'] < 0).astype(float)

        # --- 步骤 3: 严重性评估 ---
        # 3.1 下跌幅度
        fall_magnitude = df['pct_change_D'].where(df['pct_change_D'] < 0, 0).abs()
        fall_magnitude_score = normalize_score(fall_magnitude, df.index, norm_window, ascending=True)
        
        # 3.2 成交量放大
        volume_ratio = (df['volume_D'] / df.get('VOL_MA_21_D', df['volume_D'])).fillna(1.0)
        volume_spike_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        
        # 3.3 跌破短期均线
        ema5 = df.get('EMA_5_D', df['close_D'])
        break_ema5_score = (df['close_D'] < ema5).astype(float)

        # --- 步骤 4: 最终风险裁定 ---
        severity_score = (fall_magnitude_score * volume_spike_score * break_ema5_score)**(1/3)
        final_risk_score = (was_in_high_position_yesterday * is_falling_today * severity_score).astype(np.float32)
        
        states['COGNITIVE_SCORE_RISK_POST_PEAK_DOWNTURN'] = final_risk_score
        return states






