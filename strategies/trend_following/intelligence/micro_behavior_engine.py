# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, fuse_multi_level_scores, create_persistent_state

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

    def _normalize_score(self, series: pd.Series, window: int = 120, ascending: bool = True, default=0.5) -> pd.Series:
        """辅助函数：将一个Series进行滚动窗口排名归一化，生成0-1分。"""
        if series is None or series.empty:
            return pd.Series(default, index=self.strategy.df_indicators.index)
        min_periods = max(1, window // 5)
        rank = series.rolling(window=window, min_periods=min_periods).rank(pct=True)
        score = rank if ascending else 1 - rank
        return score.fillna(default).astype(np.float32)

    def _fuse_multi_level_scores(self, df: pd.DataFrame, base_name: str, weights: Dict[str, float] = None) -> pd.Series:
        """融合S+/S/A/B等多层置信度分数的辅助函数。"""
        # [代码修改] 此处直接调用 utils 中的公共函数，保持一致性
        return fuse_multi_level_scores(df, self.strategy.atomic_states, base_name, weights)

    def run_micro_behavior_synthesis(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 · 信号适配版】微观行为诊断引擎总指挥
        - 核心重构 (本次修改):
          - [信号适配] 全面审查并更新了所有方法，确保它们消费的是最新的终极原子信号。
          - [周期整合] 将 `CyclicalIntelligence` 产出的FFT周期信号整合到“高质量战备可靠性”诊断中。
        """
        print("      -> [微观行为诊断引擎 V2.0 · 信号适配版] 启动...") # [代码修改] 更新版本号
        all_states = {}
        
        # 依次调用所有微观行为合成方法
        early_momentum_states = self.synthesize_early_momentum_ignition(df)
        all_states.update(early_momentum_states)
        
        # 立即更新原子状态库，以便下游方法可以消费刚刚生成的信号
        self.strategy.atomic_states.update(early_momentum_states)
        
        all_states.update(self.diagnose_deceptive_retail_flow(df))
        all_states.update(self.synthesize_microstructure_dynamics(df))
        all_states.update(self.synthesize_euphoric_acceleration_risk(df))
        
        # 将 early_ignition_score 作为参数传入
        reversal_states = self.synthesize_reversal_reliability_score(
            df, early_ignition_score=early_momentum_states.get('COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A')
        )
        all_states.update(reversal_states)
        
        print(f"      -> [微观行为诊断引擎] 分析完毕，共生成 {len(all_states)} 个微观行为信号。")
        return all_states

    def synthesize_early_momentum_ignition(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 纯粹形态版】早期动能点火诊断模块
        - 核心职责: 作为一个纯粹的、强大的“大阳线质量分”识别器。
        """
        # print("        -> [早期动能点火诊断模块 V8.0 · 纯粹形态版] 启动...")
        states = {}
        candle_range = (df['high_D'] - df['low_D']).replace(0, np.nan)
        body_size = (df['close_D'] - df['open_D']).clip(lower=0)
        body_strength_score = (body_size / candle_range).fillna(0.0)
        position_in_range_score = ((df['close_D'] - df['low_D']) / candle_range).fillna(0.0)
        momentum_strength_score = (df['pct_change_D'] / 0.10).clip(0, 1).fillna(0.0)
        final_score = (body_strength_score * position_in_range_score * momentum_strength_score).astype(np.float32)
        states['COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A'] = final_score
        return states

    def diagnose_deceptive_retail_flow(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 VPA增强版】伪装散户吸筹诊断引擎
        - 核心逻辑: 散户资金流入 & 筹码集中 & 价格压制 & VPA低效
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = get_param_value(p.get('norm_window'), 120)
        # [代码修改] 消费新的终极资金流信号
        retail_inflow_score = self._fuse_multi_level_scores(df, 'FF_BEARISH_RESONANCE') # 散户流入是资金流的看跌信号
        chip_concentration_score = self._normalize_score(df.get('SLOPE_5_concentration_90pct_D'), norm_window, ascending=False)
        price_suppression_score = self._normalize_score(df.get('SLOPE_5_close_D').abs(), norm_window, ascending=False)
        vpa_inefficiency_score = self._normalize_score(df.get('VPA_EFFICIENCY_D'), norm_window, ascending=False)
        
        final_score = (
            retail_inflow_score * chip_concentration_score *
            price_suppression_score * vpa_inefficiency_score
        ).astype(np.float32)
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION_S'] = final_score
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.0 完全对称版】市场微观结构动态诊断引擎
        - 核心逻辑: 基于交易颗粒度、主导权、主力信念的动态变化进行诊断。
        """
        states = {}
        norm_window = 120
        granularity_momentum_up = self._normalize_score(df.get('SLOPE_5_avg_order_value_D'), norm_window, ascending=True)
        granularity_accel_up = self._normalize_score(df.get('ACCEL_5_avg_order_value_D'), norm_window, ascending=True)
        dominance_momentum_up = self._normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), norm_window, ascending=True)
        dominance_accel_up = self._normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), norm_window, ascending=True)
        power_shift_to_main_force_score = (granularity_momentum_up * granularity_accel_up * dominance_momentum_up * dominance_accel_up).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = power_shift_to_main_force_score
        
        granularity_momentum_down = self._normalize_score(df.get('SLOPE_5_avg_order_value_D'), norm_window, ascending=False)
        granularity_accel_down = self._normalize_score(df.get('ACCEL_5_avg_order_value_D'), norm_window, ascending=False)
        dominance_momentum_down = self._normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), norm_window, ascending=False)
        dominance_accel_down = self._normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), norm_window, ascending=False)
        power_shift_to_retail_risk = (granularity_momentum_down * granularity_accel_down * dominance_momentum_down * dominance_accel_down).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = power_shift_to_retail_risk
        
        conviction_momentum_weakening = self._normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), norm_window, ascending=False)
        conviction_accel_weakening = self._normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), norm_window, ascending=False)
        conviction_weakening_risk = (conviction_momentum_weakening * conviction_accel_weakening).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = conviction_weakening_risk
        
        conviction_momentum_strengthening = self._normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), norm_window, ascending=True)
        conviction_accel_strengthening = self._normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), norm_window, ascending=True)
        conviction_strengthening_opp = (conviction_momentum_strengthening * conviction_accel_strengthening).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = conviction_strengthening_opp
        return states

    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.1 上下文净化版】亢奋加速风险诊断引擎
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)

        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        top_context_score = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)

        bias_score = self._normalize_score(df['BIAS_21_D'].abs(), norm_window, ascending=True)
        volume_ratio = (df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D'])).fillna(1.0)
        volume_spike_score = self._normalize_score(volume_ratio, norm_window, ascending=True)
        atr_ratio = (df['ATR_14_D'] / df['close_D']).fillna(0.0)
        volatility_score = self._normalize_score(atr_ratio, norm_window, ascending=True)
        total_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upthrust_score = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
        
        raw_risk_score = (bias_score * volume_spike_score * volatility_score * upthrust_score)**(1/4)
        final_risk_score = (raw_risk_score * top_context_score).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score
        return states

    def synthesize_reversal_reliability_score(self, df: pd.DataFrame, early_ignition_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V4.6 · FFT周期整合版】高质量战备可靠性诊断引擎
        - 核心升级 (本次修改):
          - [周期整合] 引入FFT趋势潜力分，作为“转折点火”的确认维度之一。
        """
        # print("        -> [高质量战备可靠性诊断引擎 V4.6 · FFT周期整合版] 启动...") # [代码修改] 更新版本号
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        price_pos_yearly = self._normalize_score(df['close_D'], window=250, ascending=True, default=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        rsi_w_oversold_score = self._normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), window=52, ascending=False, default=0.5)
        background_score = np.maximum(deep_bottom_context_score, rsi_w_oversold_score).astype(np.float32)
        states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = background_score
        
        # [代码修改] 全面适配新的终极信号
        chip_accumulation_score = self._fuse_multi_level_scores(df, 'CHIP_BULLISH_RESONANCE')
        chip_reversal_score = self._fuse_multi_level_scores(df, 'CHIP_BOTTOM_REVERSAL')
        conviction_strengthening_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING')
        shareholder_turnover_score = np.maximum.reduce([
            chip_accumulation_score.values,
            chip_reversal_score.values,
            conviction_strengthening_score.values
        ])
        shareholder_quality_score = pd.Series(shareholder_turnover_score, index=df.index, dtype=np.float32)
        states['SCORE_SHAREHOLDER_QUALITY_IMPROVEMENT'] = shareholder_quality_score
        
        # [代码修改] 引入FFT趋势潜力分
        fft_trend_score = self._get_atomic_score(df, 'SCORE_TRENDING_REGIME_FFT', 0.0)
        fft_trend_slope = fft_trend_score.diff(5).fillna(0)
        trend_potential_score = self._normalize_score(fft_trend_slope.clip(lower=0), window=norm_window, ascending=True, default=0.0)
        states['INTERNAL_SCORE_TREND_POTENTIAL'] = trend_potential_score.astype(np.float32)
        
        vol_compression_score = self._fuse_multi_level_scores(df, 'VOL_COMPRESSION')
        ignition_weights = get_param_value(p.get('ignition_weights'), {'early': 0.5, 'vol': 0.2, 'potential': 0.3})
        ignition_confirmation_score = (
            early_ignition_score * ignition_weights['early'] +
            vol_compression_score * ignition_weights['vol'] +
            trend_potential_score * ignition_weights['potential']
        ).astype(np.float32)
        states['SCORE_IGNITION_CONFIRMATION'] = ignition_confirmation_score
        
        main_reliability_weights = get_param_value(p.get('main_reliability_weights'), {'shareholder': 0.5, 'ignition': 0.5})
        main_score = (
            shareholder_quality_score * main_reliability_weights['shareholder'] +
            ignition_confirmation_score * main_reliability_weights['ignition']
        )
        bonus_factor = get_param_value(p.get('reversal_reliability_bonus_factor'), 0.5)
        final_reliability_score = (main_score * (1 + background_score * bonus_factor)).astype(np.float32)
        states['COGNITIVE_SCORE_REVERSAL_RELIABILITY'] = final_reliability_score
        
        return states
