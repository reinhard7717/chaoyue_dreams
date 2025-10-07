# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict, bottom_context_score: pd.Series, top_context_score: pd.Series) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V513.0 · 统一计分法典版】
        - 核心革命: 签署“统一计分法典”，本层现在负责计算所有信号（包括进攻和风险）的贡献值。
        - 核心逻辑:
          1. 移除对信号类型的过滤，遍历所有定义了 `score` 的信号。
          2. 风险信号在字典中被定义为负分，因此本方法计算出的 `total_score` 是真实的“净战斗力得分”。
        - 收益: 彻底解决了风险贡献从未被计入总分的根本性设计缺陷。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        total_score = pd.Series(0.0, index=df.index)
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            score_value = meta.get('score', 0)
            # 核心修改：只要定义了 score，就进行计算，不再按 type 过滤
            if score_value != 0:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    processed_signal_series = signal_series.astype(float)
                    context_role = meta.get('context_role', 'neutral')
                    if context_role == 'top_risk' and score_value < 0:
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        processed_signal_series *= damper
                    elif context_role == 'bottom_opportunity' and score_value > 0:
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        processed_signal_series *= damper
                    scoring_mode = meta.get('scoring_mode', 'bipolar')
                    if scoring_mode == 'unipolar':
                        processed_signal_series = processed_signal_series.clip(lower=0)
                    bonus_amount = processed_signal_series * score_value
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount
        return total_score.fillna(0).astype(int), score_details_df.fillna(0)


















