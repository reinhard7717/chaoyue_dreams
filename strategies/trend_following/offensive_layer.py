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
        【V507.0 · 神权归还版】
        - 核心革命: 进攻层被剥夺了对“预测性”信号的计分权。
        - 核心逻辑: 在计分循环中，明确跳过所有 type 为 'predictive' 的信号。
        - 收益: 确保了“先知”的神谕不再作为普通进攻项被计分和稀释，其价值完全回归到对最高指挥部的决策引导上。
        """
        print("        -> [进攻方案评估中心 V507.0 · 神权归还版] 启动...")
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
            
            # [代码修改] 明确将 'predictive' 类型排除在计分循环之外
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


















