# 文件: strategies/trend_following/exit_layer.py
# 离场层
import pandas as pd
from .utils import get_params_block, get_param_value

class ExitLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_exit_signals(self):
        print("      -> [离场指令部 V292.0] 启动，正在执行代码净化与离场计算...")
        df = self.strategy.df_indicators
        
        df['exit_signal_code'] = 0
        df['alert_level'] = 0
        df['alert_reason'] = ''

        exit_params = get_params_block(self.strategy, 'exit_strategy_params')
        if not get_param_value(exit_params.get('enabled'), True):
            return

        is_potential_buy_day = df['entry_score'] > 0
        risk_score = df['risk_score']
        
        warning_params = exit_params.get('warning_threshold_params', {})
        for level_name, level_info in sorted(warning_params.items(), key=lambda item: item[1]['level']):
            threshold = level_info['level']
            cn_name = level_info['cn_name']
            condition = (risk_score >= threshold) & (~is_potential_buy_day)
            df.loc[condition, 'alert_level'] = level_info.get('level', 0)
            df.loc[condition, 'alert_reason'] = cn_name
        
        exit_threshold_params = exit_params.get('exit_threshold_params', {})
        for level_name, level_info in exit_threshold_params.items():
            threshold = level_info['level']
            code = level_info['code']
            cn_name = level_info['cn_name']
            condition = (risk_score >= threshold) & (~is_potential_buy_day)
            df.loc[condition, 'exit_signal_code'] = code
            df.loc[condition, 'alert_level'] = level_info.get('level', 0) 
            df.loc[condition, 'alert_reason'] = cn_name
        
        print(f"        -> 风险与离场信号计算完成。")
