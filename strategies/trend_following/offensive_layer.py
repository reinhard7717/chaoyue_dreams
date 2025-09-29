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
        【V504.0 · 过程感知版】
        - 核心升级: 新增对 'process' 类型信号的识别和计分能力。
                      这使得 ProcessIntelligence 引擎的 [-1, 1] 双极分数可以被正确计入总分。
        """
        print("        -> [进攻方案评估中心 V504.0 · 过程感知版] 启动...")
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
            
            # 在计分类型中增加 'process'
            # 这样，进攻层就能识别并处理我们新定义的过程信号了
            if score_value != 0 and signal_type in ['positional', 'dynamic', 'playbook', 'process']:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    # 核心计分逻辑：
                    # - 对于 [0,1] 信号, 结果是 [0, score_value]
                    # - 对于 [-1,1] 过程信号, 结果是 [-score_value, score_value]，完美实现加减分
                    bonus_amount = signal_series.astype(float) * score_value
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount

        if total_score.hasnans:
            debug_params = get_params_block(self.strategy, 'debug_params', {})
            if get_param_value(debug_params.get('enable_nan_probe'), False):
                nan_dates = total_score[total_score.isna()].index
                if not nan_dates.empty:
                    first_nan_date = nan_dates[0]
                    nan_signal_name = "Unknown"
                    for col in score_details_df.columns:
                        if pd.isna(score_details_df.loc[first_nan_date, col]):
                            nan_signal_name = col
                            break
                    self.strategy.intelligence_layer.deploy_nan_forensics_probe(first_nan_date, nan_signal_name)

        return total_score.fillna(0).astype(int), score_details_df.fillna(0)



















