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
        【V505.0 · 单极圣谕版】
        - 核心革命: 引入 "scoring_mode" 概念，彻底解决“事件型”过程信号被错误计为负分的问题。
        - 核心逻辑: 在计分前，检查信号是否被标记为 "unipolar"。如果是，则先将 [-1, 1] 的
                      原始分裁剪为 [0, 1]，确保只奖励事件的发生，不惩罚事件的未发生。
        """
        print("        -> [进攻方案评估中心 V505.0 · 单极圣谕版] 启动...")
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
            
            if score_value != 0 and signal_type in ['positional', 'dynamic', 'playbook', 'process']:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    
                    # 引入“单极圣谕”裁决逻辑
                    scoring_mode = meta.get('scoring_mode', 'bipolar') # 默认为双极
                    
                    # 确保信号序列是浮点数类型以进行数学运算
                    processed_signal_series = signal_series.astype(float)

                    if scoring_mode == 'unipolar':
                        # 对于“事件型”信号，只取其正值部分进行计分
                        processed_signal_series = processed_signal_series.clip(lower=0)
                        # print(f"DEBUG: Signal {signal_name} is unipolar. Original min: {signal_series.min()}, Clipped min: {processed_signal_series.min()}")

                    # 核心计分逻辑现在基于处理后的信号序列
                    bonus_amount = processed_signal_series * score_value
                    
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount

        return total_score.fillna(0).astype(int), score_details_df.fillna(0)



















