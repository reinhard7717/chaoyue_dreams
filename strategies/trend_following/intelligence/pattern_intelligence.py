# 文件: strategies/trend_following/intelligence/pattern_intelligence.py

import pandas as pd
import numpy as np
from typing import Dict
from strategies.trend_following.utils import get_params_block, get_param_value, normalize_score, normalize_to_bipolar, bipolar_to_exclusive_unipolar

class PatternIntelligence:
    """
    【V8.0 · 清洁版】形态智能引擎
    - 核心修改: 移除了所有用于调试的print探针和过程性打印，恢复静默运行模式。
    """
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def run_pattern_analysis_command(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        【V8.0 · 清洁版】形态分析总指挥
        """
        # 移除了所有过程性print语句
        all_states = {}
        p_conf = get_params_block(self.strategy, 'pattern_params', {})
        if not get_param_value(p_conf.get('enabled'), True):
            return {}
        
        norm_window = get_param_value(p_conf.get('norm_window'), 60)

        # --- 步骤一: 诊断三大公理 ---
        axiom_divergence = self._diagnose_axiom_divergence(df, norm_window)
        axiom_reversal = self._diagnose_axiom_reversal(df, norm_window)
        axiom_breakout = self._diagnose_axiom_breakout(df, norm_window)

        all_states['SCORE_PATTERN_AXIOM_DIVERGENCE'] = axiom_divergence
        all_states['SCORE_PATTERN_AXIOM_REVERSAL'] = axiom_reversal
        all_states['SCORE_PATTERN_AXIOM_BREAKOUT'] = axiom_breakout

        # --- 步骤二: 融合三大公理，合成终极信号 ---
        axiom_weights = get_param_value(p_conf.get('axiom_weights'), {
            'divergence': 0.4, 'reversal': 0.4, 'breakout': 0.2
        })
        
        bipolar_health = (
            axiom_divergence * axiom_weights['divergence'] +
            axiom_reversal * axiom_weights['reversal'] +
            axiom_breakout * axiom_weights['breakout']
        ).clip(-1, 1)

        bullish_resonance, bearish_resonance = bipolar_to_exclusive_unipolar(bipolar_health)
        all_states['SCORE_PATTERN_BULLISH_RESONANCE'] = bullish_resonance
        all_states['SCORE_PATTERN_BEARISH_RESONANCE'] = bearish_resonance

        return all_states

    def _diagnose_axiom_divergence(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 清洁版】形态公理一：诊断“背离”
        """
        # 移除了所有过程性print语句
        price_slope = df.get('SLOPE_13_close_D', pd.Series(0, index=df.index))
        momentum_slope = df.get('SLOPE_13_intraday_vwap_div_index_D', pd.Series(0, index=df.index))
        bullish_divergence_strength = ((price_slope < 0) & (momentum_slope > 0)).astype(float)
        bearish_divergence_strength = ((price_slope > 0) & (momentum_slope < 0)).astype(float)
        raw_divergence_score = bullish_divergence_strength - bearish_divergence_strength
        divergence_score = normalize_to_bipolar(raw_divergence_score, df.index, window=norm_window)
        return divergence_score.astype(np.float32)

    def _diagnose_axiom_reversal(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 清洁版】形态公理二：诊断“反转”
        """
        # 移除了所有过程性print语句和探针
        raw_reversal_score = df.get('counterparty_exhaustion_index_D', pd.Series(0, index=df.index))
        reversal_score = normalize_to_bipolar(raw_reversal_score, df.index, window=norm_window)
        return reversal_score.astype(np.float32)

    def _diagnose_axiom_breakout(self, df: pd.DataFrame, norm_window: int) -> pd.Series:
        """
        【V3.0 · 清洁版】形态公理三：诊断“突破”
        """
        # 移除了所有过程性print语句和探针
        is_breakout_up = (df['close_D'] > df.get('dynamic_consolidation_high_D', np.inf)).astype(float)
        is_breakout_down = (df['close_D'] < df.get('dynamic_consolidation_low_D', -np.inf)).astype(float)
        breakout_direction = is_breakout_up - is_breakout_down
        breakout_quality = df.get('breakout_quality_score_D', pd.Series(0, index=df.index))
        raw_breakout_score = breakout_direction * breakout_quality
        breakout_score = normalize_to_bipolar(raw_breakout_score, df.index, window=norm_window)
        return breakout_score.astype(np.float32)
