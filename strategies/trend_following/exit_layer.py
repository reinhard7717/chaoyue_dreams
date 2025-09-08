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
        【V400.2 配置驱动版】
        - 核心修改: 移除了所有硬编码的风险信号和分数，现在完全由配置文件驱动。
                    这确保了风险评估的逻辑与配置一致，解决了分数来源不透明的问题。
        """
        df = self.strategy.df_indicators
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        critical_params = scoring_params.get('critical_exit_params', {})
        critical_rules = critical_params.get('signals', {})
        critical_risk_details_df = pd.DataFrame(index=df.index)
        default_series = pd.Series(False, index=df.index)
        # 循环遍历从配置中加载的规则，而不是硬编码的字典
        for rule_name, score in critical_rules.items():
            # 增加对 "说明_" 前缀的过滤，确保只处理真正的信号规则
            if rule_name.startswith("说明_"):
                continue
            signal_series = self.strategy.atomic_states.get(rule_name, default_series)
            if signal_series.any():
                # 使用 .add() 方法安全地合并分数，以防万一有重复定义的风险信号
                if rule_name in critical_risk_details_df.columns:
                    critical_risk_details_df[rule_name] = critical_risk_details_df[rule_name].add(signal_series * score, fill_value=0)
                else:
                    critical_risk_details_df[rule_name] = signal_series * score
        # 删除了所有硬编码的 behavioral_traps 和 hardcoded_critical_risks 字典及其处理逻辑
        return critical_risk_details_df

