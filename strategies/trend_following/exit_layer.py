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
        【V400.1 风险融合版】
        """
        df = self.strategy.df_indicators
        
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        critical_params = scoring_params.get('critical_exit_params', {})
        critical_rules = critical_params.get('signals', {})
        
        critical_risk_details_df = pd.DataFrame(index=df.index)
        default_series = pd.Series(False, index=df.index)

        # 直接将两个终极风险信号识别为致命风险
        # 假设这两个信号在配置文件中也有对应的分数
        critical_rules['RISK_CHIP_STRUCTURE_CRITICAL_FAILURE'] = critical_rules.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', 500)
        critical_rules['STRUCTURE_TOPPING_DANGER_S'] = critical_rules.get('STRUCTURE_TOPPING_DANGER_S', 500)
        # 将周线战略顶部风险也识别为致命风险
        critical_rules['CONTEXT_STRATEGIC_TOPPING_RISK_W'] = critical_rules.get('CONTEXT_STRATEGIC_TOPPING_RISK_W', 500)

        for rule_name, score in critical_rules.items():
            signal_series = self.strategy.atomic_states.get(rule_name, default_series)
            if signal_series.any():
                critical_risk_details_df[rule_name] = signal_series * score
        
        return critical_risk_details_df

