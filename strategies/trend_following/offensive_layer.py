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
        【V503.2 · 法医探针版】
        - 核心升级: 在填充NaN之前，增加了一个“NaN法医探针”的触发逻辑。
                      如果检测到 total_score 中有NaN，并且配置中启用了探针，
                      它会调用 intelligence_layer 的探针进行深度诊断，然后再填充NaN以防止崩溃。
        """
        print("        -> [进攻方案评估中心 V503.2 · 法医探针版] 启动...") # [代码修改] 更新版本号和说明
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
            
            if score_value > 0 and signal_type in ['positional', 'dynamic', 'playbook']:
                signal_series = atomic_states.get(signal_name, playbook_states.get(signal_name))
                if signal_series is not None and not signal_series.empty:
                    bonus_amount = signal_series.astype(float) * score_value
                    total_score += bonus_amount
                    score_details_df[signal_name] = bonus_amount

        # [代码修改] 核心升级：先诊断，后修复
        if total_score.hasnans:
            debug_params = get_params_block(self.strategy, 'debug_params', {})
            if get_param_value(debug_params.get('enable_nan_probe'), False):
                # 找到第一个出现NaN的日期
                nan_dates = total_score[total_score.isna()].index
                if not nan_dates.empty:
                    first_nan_date = nan_dates[0]
                    # 找到是哪个信号在这一天贡献了NaN
                    nan_signal_name = "Unknown"
                    for col in score_details_df.columns:
                        if pd.isna(score_details_df.loc[first_nan_date, col]):
                            nan_signal_name = col
                            break
                    # 调用法医探针
                    self.strategy.intelligence_layer.deploy_nan_forensics_probe(first_nan_date, nan_signal_name)

        # 无论是否诊断，最后都执行防御性填充，确保流程不中断
        return total_score.fillna(0).astype(int), score_details_df.fillna(0)



















