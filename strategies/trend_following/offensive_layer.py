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
        【V506.0 · 神权归还版】
        - 核心革命: 引入 "predictive" 信号类型。对于此类信号，不再进行乘法打折。
        - 核心逻辑: 如果信号类型为 'predictive' 且原始分大于0，则直接赋予其在字典中定义的全部额定分数，
                      确保“先知”的神谕被无条件、全额执行。
        """
        print("        -> [进攻方案评估中心 V506.0 · 神权归还版] 启动...") # 修改: 更新版本号
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
            
            if score_value != 0 and signal_type in ['positional', 'dynamic', 'playbook', 'process', 'predictive']: # 修改: 增加 'predictive' 类型
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    
                    processed_signal_series = signal_series.astype(float)

                    # [代码新增] 新增对 'predictive' 类型的特殊处理
                    if signal_type == 'predictive':
                        # 对于预测型信号，只要分数大于0，就触发全额奖励
                        bonus_amount = (processed_signal_series > 0) * score_value
                    else:
                        # 保持对其他类型的原有逻辑
                        scoring_mode = meta.get('scoring_mode', 'bipolar')
                        if scoring_mode == 'unipolar':
                            processed_signal_series = processed_signal_series.clip(lower=0)
                        bonus_amount = processed_signal_series * score_value
                    
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount

        return total_score.fillna(0).astype(int), score_details_df.fillna(0)


















