# 文件: strategies/trend_following/warning_layer.py
# 预警层
import pandas as pd
from typing import Dict, List, Tuple
from .utils import get_params_block, get_param_value

class WarningLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.risk_playbook_blueprints = self._get_risk_playbook_blueprints()

    def calculate_risk_score(self) -> Tuple[pd.Series, pd.DataFrame]:
        print("        -> [最高风险裁决所 V291.0 联合作战版] 启动...")
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        
        risk_params = get_params_block(self.strategy, 'four_layer_scoring_params').get('risk_scoring', {})
        risk_rules = risk_params.get('signals', {})
        risk_score_df = pd.DataFrame(0, index=df.index, columns=list(risk_rules.keys()))
        total_risk_score = pd.Series(0.0, index=df.index)
        default_series = pd.Series(False, index=df.index)

        strategic_opportunity_signals = {
            'CHIP_DYN_ACCEL_CONCENTRATING': atomic_states.get('CHIP_DYN_ACCEL_CONCENTRATING', default_series),
            'BOX_STATE_HEALTHY_ACCUMULATION': atomic_states.get('BOX_STATE_HEALTHY_ACCUMULATION', default_series),
            'COGNITIVE_PATTERN_LOCK_CHIP_RALLY': atomic_states.get('COGNITIVE_PATTERN_LOCK_CHIP_RALLY', default_series),
            'STRUCTURE_BREAKOUT_EVE_S': atomic_states.get('STRUCTURE_BREAKOUT_EVE_S', default_series)
        }
        has_strategic_opportunity = pd.Series(False, index=df.index)
        for signal in strategic_opportunity_signals.values():
            has_strategic_opportunity |= signal

        print("          -> [第一阶段] 正在执行“外科手术式”局部对冲...")
        for rule_name, score in risk_rules.items():
            signal_series = atomic_states.get(rule_name, default_series)
            is_veto_risk_rule = rule_name in ['CONTEXT_RECENT_DISTRIBUTION_PRESSURE', 'COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN']
            if is_veto_risk_rule and signal_series.any():
                final_score = pd.Series(0.0, index=df.index)
                mitigated_score = get_param_value(risk_params.get('mitigated_veto_score'), 300)
                final_score.loc[signal_series & has_strategic_opportunity] = mitigated_score
                final_score.loc[signal_series & ~has_strategic_opportunity] = score
                risk_score_df[rule_name] = final_score
                total_risk_score += final_score
            else:
                risk_score_df.loc[signal_series, rule_name] = score
                total_risk_score += signal_series * score

        healthy_trend_state = 'STRUCTURE_MAIN_UPTREND_WAVE_S'
        trend_reduction_factor = 0.7 # 在S级主升浪中，非致命风险的重要性降低30%
        
        if healthy_trend_state in atomic_states:
            trend_condition = atomic_states[healthy_trend_state]
            if trend_condition.any():
                # 创建一个布尔掩码，标记哪些风险规则不是“一票否决”级的临界风险
                critical_risks = {"CONTEXT_RECENT_DISTRIBUTION_PRESSURE", "COGNITIVE_RISK_DYNAMIC_DECEPTIVE_CHURN", 
                                "COGNITIVE_RISK_BREAKOUT_DISTRIBUTION", "RISK_CONTEXT_LONG_TERM_DISTRIBUTION"}
                non_critical_columns = [col for col in risk_score_df.columns if col not in critical_risks]
                
                # 只对非临界风险应用折减
                risk_score_df.loc[trend_condition, non_critical_columns] *= trend_reduction_factor
                print(f"          -> [趋势折减已执行！] 已对 {trend_condition.sum()} 天的非临界风险应用了 {trend_reduction_factor} 的折减系数。")
        
        # 重新计算总分
        total_risk_score = risk_score_df.sum(axis=1)
        
        print("          -> [第二阶段] 正在评估是否执行“战略覆盖”...")
        if has_strategic_opportunity.any():
            strategic_coverage_factor = get_param_value(risk_params.get('strategic_coverage_factor'), 0.3)
            total_risk_score = total_risk_score.where(~has_strategic_opportunity, total_risk_score * strategic_coverage_factor)
            print(f"          -> [战略覆盖已执行！] 已对 {has_strategic_opportunity.sum()} 天的总风险分应用了 {strategic_coverage_factor} 的覆盖系数。")
        
        return total_risk_score, risk_score_df

    def _get_risk_playbook_blueprints(self) -> List[Dict]:
        """
        【V202.0 动态防御版】
        - 核心升级: 新增了三个基于动态筹码斜率的风险剧本，用于量化趋势衰竭的风险。
        """
        return [
            # --- 结构性风险 (Structure Risk) ---
            {'name': 'STRUCTURE_BREAKDOWN', 'cn_name': '【结构】关键支撑破位', 'score': 100},
            {'name': 'UPTHRUST_DISTRIBUTION', 'cn_name': '【结构】冲高派发', 'score': 80},
            # --- 动能衰竭风险 (Momentum Exhaustion) ---
            {'name': 'CHIP_EXHAUSTION', 'cn_name': '【动能】筹码成本加速衰竭', 'score': 60},
            {'name': 'CHIP_DIVERGENCE', 'cn_name': '【动能】筹码顶背离', 'score': 70},
            # ▼▼▼ 动态风险剧本 ▼▼▼
            {
                'name': 'PROFIT_EVAPORATION', 'cn_name': '【动态】获利盘蒸发', 'score': 75,
                'comment': '总获利盘斜率转负，市场赚钱效应快速消失，是强烈的离场信号。'
            },
            {
                'name': 'PEAK_WEAKENING', 'cn_name': '【动态】主峰根基动摇', 'score': 55,
                'comment': '主筹码峰的稳定性或占比开始下降，主力阵地可能在瓦解。'
            },
            {
                'name': 'RESISTANCE_BUILDING', 'cn_name': '【动态】上方压力积聚', 'score': 35,
                'comment': '上方套牢盘不减反增，表明进攻受阻，后续突破难度加大。'
            }
        ]
