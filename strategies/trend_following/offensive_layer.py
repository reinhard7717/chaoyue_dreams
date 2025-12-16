# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
    def calculate_entry_score(self, trigger_events: Dict, bottom_context_score: pd.Series, top_context_score: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V520.2 · 探针清理版】
        - 核心重构: 保持“统一情报总线”的设计。
        - 清理: 移除了用于调试的“真理探针”相关代码。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        all_available_signals = self.strategy.atomic_states.copy()
        all_available_signals.update(self.strategy.playbook_states)
        total_score = pd.Series(0.0, index=df.index)
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        # 移除了整个“真理探针 V2.0”的调试代码块
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            signal_series = all_available_signals.get(signal_name)
            if signal_series is None or not isinstance(signal_series, pd.Series):
                continue
            is_playbook_signal = 'PLAYBOOK' in signal_name
            processed_signal_series = signal_series.astype(float)
            scoring_mode = meta.get('scoring_mode', 'unipolar')
            context_role = meta.get('context_role', 'neutral')
            positive_score = abs(meta.get('score', 0))
            penalty_weight = abs(meta.get('penalty_weight', 0))
            if penalty_weight == 0 and positive_score > 0:
                penalty_weight = positive_score
            if positive_score == 0 and penalty_weight > 0:
                positive_score = penalty_weight
            if positive_score == 0 and penalty_weight == 0:
                continue
            bonus_amount = pd.Series(0.0, index=df.index)
            if scoring_mode == 'bipolar':
                opportunity_part = processed_signal_series.clip(lower=0)
                if context_role == 'bottom_opportunity':
                    suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    opportunity_part *= damper
                bonus_amount += opportunity_part * positive_score
                risk_part = processed_signal_series.clip(upper=0).abs()
                if context_role == 'top_risk':
                    suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    risk_part *= damper
                bonus_amount -= risk_part * penalty_weight
            else: # unipolar
                unipolar_series = processed_signal_series.clip(lower=0)
                if meta.get('type') == 'risk':
                    if context_role == 'top_risk':
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount -= unipolar_series * penalty_weight
                else:
                    if context_role == 'bottom_opportunity':
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount += unipolar_series * positive_score
            total_score += bonus_amount
            score_details_df[signal_name] = bonus_amount
        return total_score.fillna(0), score_details_df.fillna(0)


















