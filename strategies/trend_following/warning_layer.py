# 文件: strategies/trend_following/warning_layer.py
# 预警层
import pandas as pd
from typing import Dict, List, Tuple
from .utils import get_params_block, get_param_value

class WarningLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        self.risk_playbook_blueprints = self._get_risk_playbook_blueprints()

    def _diagnose_holding_health(self, risk_score_df: pd.DataFrame) -> pd.Series:
        """
        【V339.0 新增】持仓健康诊断大脑 (Holding Brain)
        - 核心职责: 对比当日与前一日的风险构成，生成结构化的“风险变化摘要”。
        """
        # print("          -> [持仓大脑 V339.0] 启动，正在进行风险变化分析...")
        
        # 获取昨日的风险构成
        risk_score_df_yesterday = risk_score_df.shift(1).fillna(0)
        
        # 初始化一个空的Series，用于存放诊断摘要
        health_diagnostics = pd.Series([{} for _ in range(len(risk_score_df))], index=risk_score_df.index)

        # 逐日进行对比分析
        for idx in risk_score_df.index:
            today_risks = set(risk_score_df.columns[risk_score_df.loc[idx] > 0])
            yesterday_risks = set(risk_score_df_yesterday.columns[risk_score_df_yesterday.loc[idx] > 0])
            
            new_risks = list(today_risks - yesterday_risks)
            persistent_risks = list(today_risks.intersection(yesterday_risks))
            resolved_risks = list(yesterday_risks - today_risks)
            
            # 只在有风险或风险变化时才记录
            if new_risks or persistent_risks or resolved_risks:
                health_diagnostics.at[idx] = {
                    'new': new_risks,
                    'persistent': persistent_risks,
                    'resolved': resolved_risks
                }

        return health_diagnostics

    def calculate_risk_score(self, critical_risk_details: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        """
        【V400.0 ORM适配版】
        - 核心升级: 接收由 ExitLayer 计算的“致命风险”详情DataFrame，并将其与自身的
                    “常规风险”合并，形成一个统一的、完整的风险构成详情DataFrame，
                    供新的报告层使用。
        - 参数变更: 新增 critical_risk_details: pd.DataFrame 参数。
        """
        # print("        -> [最高风险裁决所 V400.0 ORM适配版] 启动...")
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        risk_params = scoring_params.get('holding_warning_params', {})
        risk_rules = risk_params.get('signals', {})
        
        # --- 步骤 1: 计算“常规风险” (逻辑不变) ---
        regular_risk_score = pd.Series(0.0, index=df.index)
        risk_details_df = pd.DataFrame(index=df.index) # 这个DataFrame现在只包含常规风险
        default_series = pd.Series(False, index=df.index)

        for rule_name, score in risk_rules.items():
            signal_series = atomic_states.get(rule_name, default_series)
            if signal_series.any():
                regular_risk_score.loc[signal_series] += score
                risk_details_df[rule_name] = signal_series * score
        
        # --- 步骤 2: 【核心改造】合并致命风险与常规风险 ---
        # 将传入的致命风险分，加到常规风险分上，得到未调整前的总风险分
        # critical_risk_details.sum(axis=1) 是致命风险的总分
        base_total_risk_score = regular_risk_score + critical_risk_details.sum(axis=1)
        
        # 将两个详情DataFrame合并，形成完整的风险构成
        # 使用 .add() 并设置 fill_value=0 可以安全地合并，即使有重叠列或缺失值
        combined_risk_details_df = risk_details_df.add(critical_risk_details, fill_value=0)
        
        # --- 步骤 3: 应用“战场环境”调节器 (逻辑不变，但作用于合并后的数据) ---
        risk_multiplier = pd.Series(1.0, index=df.index)
        is_mean_reversion = atomic_states.get('FRACTAL_STATE_MEAN_REVERSION', default_series)
        is_random_walk = atomic_states.get('FRACTAL_STATE_RANDOM_WALK', default_series)
        is_unstable_market = is_mean_reversion | is_random_walk
        
        if is_unstable_market.any():
            instability_multiplier = 1.3
            risk_multiplier.loc[is_unstable_market] *= instability_multiplier
            # print(f"          -> [风险放大器] 已为 {is_unstable_market.sum()} 天的“不稳定市场”应用 {instability_multiplier}x 风险乘数。")

        is_strong_trend = atomic_states.get('FRACTAL_STATE_STRONG_TREND', default_series)
        if is_strong_trend.any():
            core_risks = {
                "CONTEXT_RECENT_DISTRIBUTION_PRESSURE", "COGNITIVE_RISK_BREAKOUT_DISTRIBUTION",
                "RISK_CONTEXT_LONG_TERM_DISTRIBUTION", "FRACTAL_RISK_TOP_DIVERGENCE",
                "STRUCTURE_TOPPING_DANGER_S"
            }
            trend_reduction_factor = 0.7
            # 注意：现在是对合并后的详情DataFrame进行操作
            for col in combined_risk_details_df.columns:
                if col not in core_risks:
                    combined_risk_details_df.loc[is_strong_trend, col] *= trend_reduction_factor
            # print(f"          -> [风险对冲器] 已为 {is_strong_trend.sum()} 天的“强趋势市场”期间，对非核心风险应用了 {trend_reduction_factor}x 折减系数。")

        # 重新计算应用了“风险对冲器”后的总分
        adjusted_total_risk_score = combined_risk_details_df.sum(axis=1)
        
        # 将“风险放大器”的全局乘数应用到总分上
        adjusted_total_risk_score *= risk_multiplier
        
        # --- 步骤 4: 战略机会覆盖 (逻辑不变) ---
        has_strategic_opportunity = atomic_states.get('COGNITIVE_PATTERN_LOCK_CHIP_RALLY', default_series)
        if has_strategic_opportunity.any():
            strategic_coverage_factor = get_param_value(risk_params.get('strategic_coverage_factor'), 0.3)
            adjusted_total_risk_score = adjusted_total_risk_score.where(~has_strategic_opportunity, adjusted_total_risk_score * strategic_coverage_factor)
            # print(f"          -> [战略覆盖已执行！] 已对 {has_strategic_opportunity.sum()} 天的总风险分应用了 {strategic_coverage_factor} 的覆盖系数。")
        
        # --- 步骤 5: 生成风险变化摘要 ---
        # 使用合并后的完整风险详情来生成摘要
        risk_change_summary = self._diagnose_holding_health(combined_risk_details_df)

        # print("        -> [最高风险裁决所 V400.0] 风险评估完成。")
        # 返回最终调整后的总风险分，和合并后的完整风险详情
        return adjusted_total_risk_score, combined_risk_details_df, risk_change_summary

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
