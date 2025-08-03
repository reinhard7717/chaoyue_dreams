# 文件: strategies/trend_following/exit_layer.py
# 离场层
import pandas as pd
from typing import Tuple # 导入 Tuple
from .utils import get_params_block, get_param_value

class ExitLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_critical_risks(self) -> pd.DataFrame:
        """
        【V400.0 ORM适配版】
        - 核心重构: 此方法的职责被极大简化。它不再计算任何最终信号（如exit_code），
                    而是只专注于计算“致命风险”的构成，并返回一个包含这些风险详情的
                    DataFrame，供 WarningLayer 合并使用。
        - 返回值变更: 返回一个 pd.DataFrame。
        """
        # print("      -> [致命风险评估部 V400.0] 启动...")
        df = self.strategy.df_indicators
        
        # 1. 读取“致命离场”配置
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        critical_params = scoring_params.get('critical_exit_params', {})
        critical_rules = critical_params.get('signals', {})
        
        # 2. 计算“致命风险”详情DataFrame
        critical_risk_details_df = pd.DataFrame(index=df.index)
        default_series = pd.Series(False, index=df.index)

        for rule_name, score in critical_rules.items():
            signal_series = self.strategy.atomic_states.get(rule_name, default_series)
            if signal_series.any():
                # 直接在DataFrame中记录每个致命风险及其分数
                critical_risk_details_df[rule_name] = signal_series * score
        
        # 3. 返回详情DataFrame
        # 所有关于阈值判断、设置 exit_code/alert_level 的逻辑都已移除，
        # 这些判断现在由 JudgmentLayer 基于总风险分统一处理。
        return critical_risk_details_df

