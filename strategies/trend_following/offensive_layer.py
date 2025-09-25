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
        【V503.0 · 配置驱动重构版】
        - 核心重构: 不再读取本地配置，而是遍历所有原子信号，根据 signal_dictionary.json
                      中定义的 'type' 和 'score' 动态计算进攻分。
        """
        print("        -> [进攻方案评估中心 V503.0 · 配置驱动重构版] 启动...") # [代码修改]
        df = self.strategy.df_indicators
        score_details_df = pd.DataFrame(index=df.index)
        
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        atomic_states = self.strategy.atomic_states
        playbook_states = self.strategy.playbook_states
        
        total_score = pd.Series(0.0, index=df.index)
        
        # [代码修改] 遍历所有信号，根据配置动态计分
        for signal_name, meta in score_map.items():
            if not isinstance(meta, dict): continue
            
            # 只处理进攻型信号
            signal_type = meta.get('type')
            score_value = meta.get('score', 0)
            
            if score_value > 0 and signal_type in ['positional', 'dynamic', 'playbook']:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    bonus_amount = signal_series.astype(float) * score_value
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount

        return total_score.astype(int), score_details_df.fillna(0)
