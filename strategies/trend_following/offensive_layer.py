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
        【V517.1 · 健壮性增强版】
        - 核心修复: 在处理信号前增加 isinstance(signal_series, pd.Series) 检查，
                      从根本上防止因 atomic_states 中存在非序列化数据（如字典）而导致的崩溃。
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
            signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
            # [代码修改开始] 增加对 signal_series 类型的严格检查
            if signal_series is None or not isinstance(signal_series, pd.Series):
                continue
            # [代码修改结束]
            processed_signal_series = signal_series.astype(float)
            scoring_mode = meta.get('scoring_mode', 'unipolar') # 默认改为unipolar更安全
            context_role = meta.get('context_role', 'neutral')
            # 获取分数和惩罚权重，确保它们是正数
            positive_score = abs(meta.get('score', 0))
            penalty_weight = abs(meta.get('penalty_weight', 0))
            # 如果 penalty_weight 未定义，则使用 score 作为惩罚权重
            if penalty_weight == 0 and positive_score > 0:
                penalty_weight = positive_score
            # 如果 score 未定义，则使用 penalty_weight 作为分数
            if positive_score == 0 and penalty_weight > 0:
                positive_score = penalty_weight
            if positive_score == 0 and penalty_weight == 0:
                continue
            bonus_amount = pd.Series(0.0, index=df.index)
            if scoring_mode == 'bipolar':
                # --- 双极性信号处理 ---
                # 1. 处理正向部分（机会）
                opportunity_part = processed_signal_series.clip(lower=0)
                if context_role == 'bottom_opportunity':
                    suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    opportunity_part *= damper
                bonus_amount += opportunity_part * positive_score
                # 2. 处理负向部分（风险）
                risk_part = processed_signal_series.clip(upper=0).abs()
                if context_role == 'top_risk':
                    suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                    damper = 1.0 - suppression_factor
                    risk_part *= damper
                bonus_amount -= risk_part * penalty_weight # 永远是减分
            else: # unipolar
                # --- 单极性信号处理 ---
                unipolar_series = processed_signal_series.clip(lower=0)
                # 判断是机会还是风险
                if meta.get('type') == 'risk': # 风险信号
                    if context_role == 'top_risk':
                        suppression_factor = bottom_context_score.where(bottom_context_score >= bottom_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount -= unipolar_series * penalty_weight # 永远是减分
                else: # 机会信号
                    if context_role == 'bottom_opportunity':
                        suppression_factor = top_context_score.where(top_context_score >= top_context_threshold, 0.0)
                        damper = 1.0 - suppression_factor
                        unipolar_series *= damper
                    bonus_amount += unipolar_series * positive_score # 永远是加分
            total_score += bonus_amount
            score_details_df[signal_name] = bonus_amount
        # 保持浮点数以便观察，最终的取整在 judgment_layer 中完成
        return total_score.fillna(0), score_details_df.fillna(0)


















