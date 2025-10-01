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
        【V510.0 · 协定版】
        - 核心革命: 签署神圣协定，重建进攻层的神圣屏障。
        - 核心逻辑: 恢复对 'predictive' 类型信号的绝对豁免权，确保计分系统永远不处理神谕本身。
                      “先知的低语”现在由一个独立的 'process' 类型信号 (SCORE_SETUP_PANIC_SELLING) 承载。
        - 收益: 彻底分离了神谕的决策功能和其物质基础的计分功能，实现了系统架构的最终和谐。
        """
        # print("        -> [进攻方案评估中心 V510.0 · 协定版] 启动...")
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
            
            # 重建神圣屏障：如果信号类型是 'predictive'，则直接跳过，不参与任何计分。
            if signal_type == 'predictive':
                continue

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


















