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
        【V516.0 · 作战状态恢复版】
        - 核心升级: 移除所有为调查“724分事件”而部署的调试探针，恢复代码至纯净、高效的作战状态。
        - 收益: 代码清晰度、执行效率达到最佳。
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
            if score_value == 0:
                score_value = meta.get('penalty_weight', 0)
            if score_value != 0:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    processed_signal_series = signal_series.astype(float)
                    context_role = meta.get('context_role', 'neutral')
                    
                    # 修正风险信号的压制逻辑
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
        # 保持浮点数以便观察，最终的取整在 judgment_layer 中完成
        return total_score.fillna(0), score_details_df.fillna(0)


















