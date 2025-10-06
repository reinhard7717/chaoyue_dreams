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
        【V511.0 · 赫淮斯托斯协议版】
        - 核心革命: 签署“赫淮斯托斯协议”，将上下文感知能力直接锻造进计分核心。
        - 核心逻辑:
          1. [接收神谕] 本方法现在接收由更高层传递的权威上下文分数。
          2. [锻造规则] 在计分循环内部，直接使用上下文分数对不合时宜的风险信号进行压制。
             - 当底部上下文确认时，所有“顶部风险”信号的原始值将被削弱或归零。
             - 当顶部上下文确认时，所有“底部机会”信号的原始值将被削弱或归零。
        - 收益: 从计分的源头根除了“战略悖论”，确保了总分的逻辑一致性。
        """
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        total_score = pd.Series(0.0, index=df.index)
        # [代码新增] 定义上下文压制参数
        p_context_suppression = get_params_block(self.strategy, 'contextual_suppression_params', {})
        bottom_context_threshold = get_param_value(p_context_suppression.get('bottom_context_threshold'), 0.9)
        top_context_threshold = get_param_value(p_context_suppression.get('top_context_threshold'), 0.9)
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            signal_type = meta.get('type')
            score_value = meta.get('score', 0)
            if signal_type == 'predictive':
                continue
            if score_value != 0 and signal_type in ['positional', 'dynamic', 'playbook', 'process', 'risk']: # 确保 risk 类型也被处理
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    processed_signal_series = signal_series.astype(float)
                    # [代码新增] 赫淮斯托斯锻造核心：应用上下文阻尼器
                    context_role = meta.get('context_role', 'neutral') # 从配置中获取信号的角色
                    if context_role == 'top_risk' and score_value < 0:
                        # 如果是顶部风险，使用底部上下文进行压制
                        damper = (1.0 - bottom_context_score.clip(lower=bottom_context_threshold)).fillna(1.0)
                        processed_signal_series *= damper
                    elif context_role == 'bottom_opportunity' and score_value > 0:
                        # 如果是底部机会，使用顶部上下文进行压制
                        damper = (1.0 - top_context_score.clip(lower=top_context_threshold)).fillna(1.0)
                        processed_signal_series *= damper
                    scoring_mode = meta.get('scoring_mode', 'bipolar')
                    if scoring_mode == 'unipolar':
                        processed_signal_series = processed_signal_series.clip(lower=0)
                    bonus_amount = processed_signal_series * score_value
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount
        return total_score.fillna(0).astype(int), score_details_df.fillna(0)


















