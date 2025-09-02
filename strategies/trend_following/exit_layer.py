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
                
        # --- 步骤1.5: 新增来自行为/认知层的S级致命风险，作为配置的补充和强化 ---
        # 目的: 硬编码整合最危险的几种“陷阱”信号，确保策略的最终安全。
        behavioral_traps = {
            # S级风险: 长期派发背景下的短期诱多拉升，是典型的出货陷阱。
            "RISK_BEHAVIOR_DECEPTIVE_RALLY_LONG_TERM_S": 1700,
            # S级风险: 股价处于高位，但价格、筹码、获利盘等多维度出现背离，是顶部强烈信号。
            "RISK_STATIC_HIGH_ALTITUDE_MULTI_DIVERGENCE_S": 1650,
            # S级风险: 在战略派发的大背景下出现的任何拉升，都应被视为最高级别的诱多陷阱。
            "RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S": 1800,
        }

        for risk_name, score in behavioral_traps.items():
            signal_series = self.strategy.atomic_states.get(risk_name, default_series)
            if signal_series.any():
                critical_risk_details_df[risk_name] = signal_series * score
                print(f"          -> [行为陷阱侦测] 侦测到致命风险 “{risk_name}”，触发紧急离场！")

        # --- 步骤2: 硬编码终极安全网，监控最高级别的S+级风险信号 ---
        # 目的: 确保策略对最危险的陷阱有本能的规避能力，防止因配置疏忽导致灾难性亏损。
        hardcoded_critical_risks = {
            "RISK_STATIC_DYN_FUSION_TRAP_S_PLUS": 1000,  # 静态-动态融合陷阱 (诱多)
            "RISK_FUND_FLOW_DECEPTIVE_RALLY_S_PLUS": 1000, # 主力缺席下的诱多式拉升
            "RISK_STRUCTURE_MTF_EXHAUSTION_S_PLUS": 1000, # 战略衰竭下的战术诱多
            "RISK_BEHAVIOR_PANIC_FLEEING_S": 1000,       # 获利盘恐慌加速出逃
            # S级风险: 亢奋顶点+结构瓦解，是顶部结构性崩塌的强烈信号。
            "RISK_STATIC_DYN_COLLAPSE_S": 1900,
            # S级风险: 高位派发陷阱，多重证据确认主力在高位出货。
            "SCENARIO_HIGH_ALTITUDE_DISTRIBUTION_TRAP_S": 1850,
            # S级风险: 市场引擎失速，价格上涨但资金效率崩溃，是上涨动能终结的强烈信号。
            "RISK_DYN_MARKET_ENGINE_STALLING_S": 1950,
            # S级风险: 获利盘恐慌加速，是市场情绪崩溃、踩踏式下跌的预警。
            "RISK_DYN_PANIC_SELLING_ACCELERATING_S": 1920,
            # S级风险: 结构性衰竭反弹，短期上涨但长期筹码结构瓦解，是典型的诱多陷阱。
            "RISK_DYN_STRUCTURAL_WEAKNESS_RALLY_S": 1900,
            # S级风险: 认知层合成的顶部危险结构信号，代表多重风险共振，是结构性顶部的强烈信号。
            "STRUCTURE_TOPPING_DANGER_S": 1980,
            # S级风险: 高位狂欢亢奋陷阱，市场最乐观时主力加速派发，是极度危险的离场信号。
            "RISK_CHIP_EUPHORIA_TRAP_S": 1960,
            # S级风险: 多维共振超涨，日线和周线同时严重超涨，是形成重要顶部的强烈共振信号，必须无条件离场。
            "RISK_STRUCTURE_MTF_OVEREXTENDED_RESONANCE_S": 1940,
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

