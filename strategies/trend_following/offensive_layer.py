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
        【V513.2 · 度量衡统一法案版】
        - 核心修正: 移除返回前的 .astype(int) 转换，确保分数在计算链路中保持为高精度浮点数。
        - 收益: 根除了因过早截断小数而导致的源头性分数误差。
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
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d).date() for d in probe_dates_str]
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            score_value = meta.get('score', 0)
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
        for date in total_score.index:
            if date.date() in probe_dates:
                # 打印更精确的浮点数
                print(f"      -> [阿里阿德涅之线 @ {date.date()}] (OffensiveLayer) 计分完成，返回前的 total_score: {total_score.loc[date]:.4f}")
        # 移除 .astype(int)，保持浮点数精度
        return total_score.fillna(0), score_details_df.fillna(0)


















