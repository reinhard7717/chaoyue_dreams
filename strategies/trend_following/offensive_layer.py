# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from .utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V509.0 · 先知的低语版】
        - 核心革命: 恢复了先知信号的双重身份。它既能独立触发“先知入场”，也能作为常规过程信号为“买入信号”贡献分数。
        - 核心逻辑: 移除了对 'predictive' 类型的硬编码过滤，恢复了基于信号类型列表的通用计分逻辑。
        - 收益: 使得不够强的预测信号也能作为“低语”为常规决策提供佐证，增强了系统的决策细腻度。
        """
        print("        -> [进攻方案评估中心 V509.0 · 先知的低语版] 启动...")
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        
        total_score = pd.Series(0.0, index=df.index)
        
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            
            signal_type = meta.get('type')
            score_value = meta.get('score', 0)
            
            # 移除对 'predictive' 类型的硬编码过滤，恢复通用计分逻辑
            if score_value != 0 and signal_type in ['positional', 'dynamic', 'playbook', 'process']:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    
                    processed_signal_series = signal_series.astype(float)

                    scoring_mode = meta.get('scoring_mode', 'bipolar')
                    if scoring_mode == 'unipolar':
                        processed_signal_series = processed_signal_series.clip(lower=0)
                    bonus_amount = processed_signal_series * score_value
                    
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount

        return total_score.fillna(0).astype(int), score_details_df.fillna(0)


















