# 文件: strategies/trend_following/exit_layer.py
# 离场层
import pandas as pd
from .utils import get_params_block, get_param_value

class ExitLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_exit_signals(self):
        """
        【V293.0 主动净化版】
        - 核心修复: 在计算前，主动将 exit/alert 相关列重置为默认值，
                    彻底杜绝因 pandas 填充机制导致的历史信号污染问题。
        """
        # print("      -> [离场指令部 V293.0 主动净化版] 启动...")
        df = self.strategy.df_indicators
        
        # --- 【核心修复】主动净化 ---
        # 在进行任何计算之前，先将所有输出列重置为干净的初始状态。
        df['exit_signal_code'] = 0
        df['alert_level'] = 0
        df['alert_reason'] = '' # 使用空字符串作为默认值

        exit_params = get_params_block(self.strategy, 'exit_strategy_params')
        if not get_param_value(exit_params.get('enabled'), True):
            return

        # 1. 读取专属的“致命离场”配置
        critical_params = get_params_block(self.strategy, 'critical_exit_params')
        critical_rules = critical_params.get('signals', {})
        
        # 2. 计算“致命风险分”
        critical_risk_score = pd.Series(0.0, index=df.index)
        default_series = pd.Series(False, index=df.index)
        for rule_name, score in critical_rules.items():
            signal_series = self.strategy.atomic_states.get(rule_name, default_series)
            if signal_series.any():
                critical_risk_score.loc[signal_series] += score

        # 3. 使用“致命风险分”来判断离场
        is_potential_buy_day = df['entry_score'] > 0

        # 这里的阈值，现在是相对于“致命风险分”的
        exit_threshold_params = exit_params.get('exit_threshold_params', {})
        for level_name, level_info in exit_threshold_params.items():
            threshold = level_info['level']
            code = level_info['code']
            cn_name = level_info['cn_name']

            condition = (critical_risk_score >= threshold) # & (~is_potential_buy_day)
            df.loc[condition, 'exit_signal_code'] = code
            df.loc[condition, 'alert_level'] = level_info.get('level', 0) 
            df.loc[condition, 'alert_reason'] = cn_name

        # 4. 【重要】预警信号现在由 WarningLayer 的总风险分 (risk_score) 决定
        #    我们在这里只负责读取并应用它，不再自己计算
        risk_score_from_warning_layer = df['risk_score']
        warning_params = exit_params.get('warning_threshold_params', {})
        for level_name, level_info in sorted(warning_params.items(), key=lambda item: item[1]['level']):
            threshold = level_info['level']
            cn_name = level_info['cn_name']

            condition = (risk_score_from_warning_layer >= threshold) # & (~is_potential_buy_day)
            # 只更新那些还没有被更高等级的“离场信号”覆盖的记录
            df.loc[condition & (df['exit_signal_code'] == 0), 'alert_level'] = level_info.get('level', 0)
            df.loc[condition & (df['exit_signal_code'] == 0), 'alert_reason'] = cn_name

        # print(f"        -> 风险与离场信号计算完成。")
