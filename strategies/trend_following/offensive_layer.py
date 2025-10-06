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
        【V512.0 · 阿瑞斯精准打击协议版】
        - 核心革命: 签署“阿瑞斯精准打击协议”，修复“赫淮斯托斯之锤”的逻辑瑕疵。
        - 核心逻辑:
          1. 废除错误的 `clip(lower=...)` 逻辑。
          2. 采用全新的 `where` 条件判断，确保上下文阻尼器只在分数高于阈值时才被激活。
        - 收益: 实现了对风险信号的精准外科手术式切除，避免了在非底部区域错误压制风险的重大BUG。
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
            signal_type = meta.get('type')
            score_value = meta.get('score', 0)
            # [代码修改] 将 'predictive' 类型也纳入处理范围，因为高潮衰竭风险需要被压制
            if score_value != 0 and signal_type in ['positional', 'dynamic', 'playbook', 'process', 'risk', 'predictive']:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    processed_signal_series = signal_series.astype(float)
                    context_role = meta.get('context_role', 'neutral')
                    if context_role == 'top_risk' and score_value < 0:
                        # [代码修改] 阿瑞斯精准打击：使用 where 条件判断，只在高于阈值时进行压制
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        processed_signal_series *= damper
                    elif context_role == 'bottom_opportunity' and score_value > 0:
                        # [代码修改] 阿瑞斯精准打击：对底部机会应用同样的精准逻辑
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


















