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

        for rule_name, score in critical_rules.items():
            signal_series = self.strategy.atomic_states.get(rule_name, default_series)
            if signal_series.any():
                critical_risk_details_df[rule_name] = signal_series * score
        # --- 步骤2: 硬编码终极安全网，监控最高级别的S+级风险信号 ---
        # 目的: 确保策略对最危险的陷阱有本能的规避能力，防止因配置疏忽导致灾难性亏损。
        hardcoded_critical_risks = {
            "RISK_STATIC_DYN_FUSION_TRAP_S_PLUS": 1000,  # 静态-动态融合陷阱 (诱多)
            "RISK_FUND_FLOW_DECEPTIVE_RALLY_S_PLUS": 1000, # 主力缺席下的诱多式拉升
            "RISK_CHIP_EUPHORIA_TRAP_S": 1000,           # 高位狂欢的亢奋陷阱
            "RISK_STRUCTURE_MTF_EXHAUSTION_S_PLUS": 1000, # 战略衰竭下的战术诱多
            "RISK_BEHAVIOR_PANIC_FLEEING_S": 1000,       # 获利盘恐慌加速出逃
        }

        for risk_name, score in hardcoded_critical_risks.items():
            signal_series = self.strategy.atomic_states.get(risk_name, default_series)
            if signal_series.any():
                # 使用 .add() 方法以安全地合并分数，避免覆盖已存在的分数
                if risk_name in critical_risk_details_df.columns:
                    critical_risk_details_df[risk_name] = critical_risk_details_df[risk_name].add(signal_series * score, fill_value=0)
                else:
                    critical_risk_details_df[risk_name] = signal_series * score
                print(f"          -> [终极安全网] 侦测到致命风险 “{risk_name}”，触发紧急离场！")
        
        return critical_risk_details_df

