# 文件: strategies/trend_following/intelligence/micro_behavior_engine.py
# 微观行为诊断引擎
import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, fuse_multi_level_scores, normalize_score

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
        【V2.1 · 状态同步修复版】微观行为诊断引擎总指挥
        - 核心修复: 修正了状态更新逻辑。现在，每个子模块生成的信号都会被立即更新到
                    全局的 `self.strategy.atomic_states` 中，而不仅仅是暂存在局部变量里。
                    这解决了下游方法因无法获取上游即时计算的信号而导致崩溃的问题。
        """
        # 更新版本号和日志
        print("      -> [微观行为诊断引擎 V2.1 · 状态同步修复版] 启动...")
        all_states = {}

        # [代码新增] 定义一个辅助函数来简化状态更新流程
        def update_states(new_states: Dict[str, pd.Series]):
            """同时更新局部和全局状态字典"""
            if new_states:
                all_states.update(new_states)
                self.strategy.atomic_states.update(new_states)

        # 依次调用所有微观行为合成方法，并使用辅助函数立即更新状态
        update_states(self.synthesize_early_momentum_ignition(df))
        
        # 现在，下游方法可以安全地消费上面生成的信号了
        update_states(self.diagnose_deceptive_retail_flow(df))
        update_states(self.synthesize_microstructure_dynamics(df))
        update_states(self.synthesize_euphoric_acceleration_risk(df))
        
        # 在调用 reversal_reliability_score 之前，它所依赖的所有信号都已存在于 atomic_states 中
        # 需要从 self.strategy.atomic_states 中获取 early_ignition_score，因为它刚刚被更新
        early_ignition_score = self._get_atomic_score(df, 'COGNITIVE_SCORE_EARLY_MOMENTUM_IGNITION_A')
        update_states(self.synthesize_reversal_reliability_score(
            df, early_ignition_score=early_ignition_score
        ))
        
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
        【V2.2 · 重构版】伪装散户吸筹诊断引擎
        """
        states = {}
        p = get_params_block(self.strategy, 'deceptive_flow_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
        norm_window = get_param_value(p.get('norm_window'), 120)
        retail_inflow_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'FF_BEARISH_RESONANCE')
        
        # 调用 utils.normalize_score 并传入 df.index
        chip_concentration_score = normalize_score(df.get('SLOPE_5_concentration_90pct_D'), df.index, norm_window, ascending=False)
        price_suppression_score = normalize_score(df.get('SLOPE_5_close_D').abs(), df.index, norm_window, ascending=False)
        vpa_inefficiency_score = normalize_score(df.get('VPA_EFFICIENCY_D'), df.index, norm_window, ascending=False)
        
        final_score = (
            retail_inflow_score * chip_concentration_score *
            price_suppression_score * vpa_inefficiency_score
        ).astype(np.float32)
        states['SCORE_COGNITIVE_DECEPTIVE_RETAIL_ACCUMULATION_S'] = final_score
        return states

    def synthesize_microstructure_dynamics(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V2.1 · 重构版】市场微观结构动态诊断引擎
        """
        states = {}
        norm_window = 120
        # 全部改为调用 utils.normalize_score 并传入 df.index
        granularity_momentum_up = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=True)
        granularity_accel_up = normalize_score(df.get('ACCEL_5_avg_order_value_D'), df.index, norm_window, ascending=True)
        dominance_momentum_up = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=True)
        dominance_accel_up = normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), df.index, norm_window, ascending=True)
        power_shift_to_main_force_score = (granularity_momentum_up * granularity_accel_up * dominance_momentum_up * dominance_accel_up).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_POWER_SHIFT_TO_MAIN_FORCE'] = power_shift_to_main_force_score
        
        granularity_momentum_down = normalize_score(df.get('SLOPE_5_avg_order_value_D'), df.index, norm_window, ascending=False)
        granularity_accel_down = normalize_score(df.get('ACCEL_5_avg_order_value_D'), df.index, norm_window, ascending=False)
        dominance_momentum_down = normalize_score(df.get('SLOPE_5_trade_concentration_index_D'), df.index, norm_window, ascending=False)
        dominance_accel_down = normalize_score(df.get('ACCEL_5_trade_concentration_index_D'), df.index, norm_window, ascending=False)
        power_shift_to_retail_risk = (granularity_momentum_down * granularity_accel_down * dominance_momentum_down * dominance_accel_down).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL'] = power_shift_to_retail_risk
        
        conviction_momentum_weakening = normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=False)
        conviction_accel_weakening = normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=False)
        conviction_weakening_risk = (conviction_momentum_weakening * conviction_accel_weakening).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'] = conviction_weakening_risk
        
        conviction_momentum_strengthening = normalize_score(df.get('SLOPE_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=True)
        conviction_accel_strengthening = normalize_score(df.get('ACCEL_5_main_force_conviction_ratio_D'), df.index, norm_window, ascending=True)
        conviction_strengthening_opp = (conviction_momentum_strengthening * conviction_accel_strengthening).astype(np.float32)
        states['COGNITIVE_SCORE_OPP_MAIN_FORCE_CONVICTION_STRENGTHENING'] = conviction_strengthening_opp
        return states

    def synthesize_euphoric_acceleration_risk(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V1.2 · 重构版】亢奋加速风险诊断引擎
        """
        states = {}
        p_risk = get_params_block(self.strategy, 'euphoric_risk_params', {})
        if not get_param_value(p_risk.get('enabled'), True): return states
        norm_window = get_param_value(p_risk.get('norm_window'), 120)

        rolling_low_55d = df['low_D'].rolling(window=55, min_periods=21).min()
        rolling_high_55d = df['high_D'].rolling(window=55, min_periods=21).max()
        price_range_55d = (rolling_high_55d - rolling_low_55d).replace(0, 1e-9)
        top_context_score = ((df['close_D'] - rolling_low_55d) / price_range_55d).clip(0, 1).fillna(0.5)

        # 全部改为调用 utils.normalize_score 并传入 df.index
        bias_score = normalize_score(df['BIAS_21_D'].abs(), df.index, norm_window, ascending=True)
        volume_ratio = (df['volume_D'] / df.get('VOL_MA_55_D', df['volume_D'])).fillna(1.0)
        volume_spike_score = normalize_score(volume_ratio, df.index, norm_window, ascending=True)
        atr_ratio = (df['ATR_14_D'] / df['close_D']).fillna(0.0)
        volatility_score = normalize_score(atr_ratio, df.index, norm_window, ascending=True)
        total_range = (df['high_D'] - df['low_D']).replace(0, 1e-9)
        upper_shadow = (df['high_D'] - np.maximum(df['open_D'], df['close_D']))
        upthrust_score = (upper_shadow / total_range).clip(0, 1).fillna(0.0)
        
        raw_risk_score = (bias_score * volume_spike_score * volatility_score * upthrust_score)**(1/4)
        final_risk_score = (raw_risk_score * top_context_score).astype(np.float32)
        states['COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = final_risk_score
        return states

    def synthesize_reversal_reliability_score(self, df: pd.DataFrame, early_ignition_score: pd.Series) -> Dict[str, pd.Series]:
        """
        【V4.8 · 重构版】高质量战备可靠性诊断引擎
        """
        states = {}
        p = get_params_block(self.strategy, 'reversal_reliability_params', {})
        if not get_param_value(p.get('enabled'), True): return states
        
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        norm_window = get_param_value(p.get('norm_window'), 120)
        
        # 全部改为调用 utils.normalize_score 并传入 df.index
        price_pos_yearly = normalize_score(df['close_D'], df.index, window=250, ascending=True, default_value=0.5)
        deep_bottom_context_score = 1.0 - price_pos_yearly
        rsi_w_oversold_score = normalize_score(df.get('RSI_13_W', pd.Series(50, index=df.index)), df.index, window=52, ascending=False, default_value=0.5)
        background_score = np.maximum(deep_bottom_context_score, rsi_w_oversold_score).astype(np.float32)
        states['SCORE_CONTEXT_DEEP_BOTTOM_ZONE'] = background_score
        
        chip_accumulation_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'CHIP_BULLISH_RESONANCE')
        chip_reversal_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'CHIP_BOTTOM_REVERSAL')
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
        
        vol_compression_score = fuse_multi_level_scores(self.strategy.atomic_states, df.index, 'VOL_COMPRESSION')
        ignition_weights = get_param_value(p.get('ignition_weights'), {'early': 0.5, 'vol': 0.2, 'potential': 0.3})
        
        if len(early_ignition_score) != len(df.index):
            early_ignition_score = early_ignition_score.reindex(df.index, fill_value=0.0)

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










