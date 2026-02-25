# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict, bottom_context_score: pd.Series, top_context_score: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V521.0 · 极性绝对正交解算版】
        - 核心修复: 彻底修复了由于 if-elif 分支互斥导致的 penalty_weight 被丢弃的致命数值黑洞 BUG。
        - 极性强压: 对所有的 penalty_weight 施加严格的 -abs() 负向锁定，确保风险项绝对不能向多头得分渗透。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        all_available_signals = self.strategy.atomic_states.copy()
        all_available_signals.update(self.strategy.playbook_states)
        total_offensive_score = pd.Series(0.0, index=df.index)
        total_risk_sum = pd.Series(0.0, index=df.index)
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        trend_quality = all_available_signals.get('FUSION_BIPOLAR_TREND_QUALITY', pd.Series(0.0, index=df.index)).fillna(0.0)
        cognitive_risk_trend_exhaustion = all_available_signals.get('COGNITIVE_RISK_TREND_EXHAUSTION', pd.Series(0.0, index=df.index)).fillna(0.0)
        trend_quality_damper = (trend_quality + 1) / 2
        trend_exhaustion_damper = 1 - cognitive_risk_trend_exhaustion * 0.8
        trend_exhaustion_damper = trend_exhaustion_damper.clip(lower=0.2)
        price_momentum_damper = pd.concat([trend_quality_damper, trend_exhaustion_damper], axis=1).min(axis=1).clip(lower=0.1, upper=1.0)
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            signal_series = all_available_signals.get(signal_name)
            if signal_series is None or not isinstance(signal_series, pd.Series):
                continue
            processed_signal_series = signal_series.astype(float)
            scoring_mode = meta.get('scoring_mode', 'unipolar')
            context_role = meta.get('context_role', 'neutral')
            configured_score = meta.get('score')
            configured_penalty_weight = meta.get('penalty_weight')
            positive_score = 0.0
            penalty_weight = 0.0
            if scoring_mode == 'bipolar':
                if configured_score is not None:
                    positive_score = abs(float(configured_score))
                elif configured_penalty_weight is not None:
                    positive_score = abs(float(configured_penalty_weight))
                if configured_penalty_weight is not None:
                    penalty_weight = -abs(float(configured_penalty_weight))
                elif configured_score is not None:
                    penalty_weight = -abs(float(configured_score))
            else:
                if configured_score is not None:
                    positive_score = abs(float(configured_score))
                if configured_penalty_weight is not None:
                    penalty_weight = -abs(float(configured_penalty_weight))
            bonus_amount_for_signal = pd.Series(0.0, index=df.index)
            if scoring_mode == 'bipolar':
                opportunity_part = processed_signal_series.clip(lower=0)
                bonus_amount_for_signal += opportunity_part * positive_score
                risk_part = processed_signal_series.clip(upper=0).abs()
                if context_role == 'top_risk':
                    suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    risk_part *= damper
                bonus_amount_for_signal += risk_part * penalty_weight
            else:
                if meta.get('type') == 'risk':
                    risk_contribution_series = processed_signal_series.clip(lower=0)
                    if context_role == 'top_risk':
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        risk_contribution_series *= damper
                    bonus_amount_for_signal = risk_contribution_series * penalty_weight
                else:
                    unipolar_opportunity_series = processed_signal_series.clip(lower=0)
                    if context_role == 'bottom_opportunity':
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_opportunity_series *= damper
                    bonus_amount_for_signal = unipolar_opportunity_series * positive_score
            total_offensive_score += bonus_amount_for_signal.clip(lower=0)
            total_risk_sum += bonus_amount_for_signal.clip(upper=0).abs()
            score_details_df[signal_name] = bonus_amount_for_signal
        return total_offensive_score.fillna(0), total_risk_sum.fillna(0), score_details_df.fillna(0)


















