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
        """
        【V292.0 赫斯特指数增强版】
        - 核心升级: 深度集成赫斯特指数，实现风险的“环境自适应”调节。
          1. 将分形风险直接纳入计分。
          2. 根据市场宏观状态（趋势/震荡）动态放大或缩小风险权重。
        """
        print("        -> [最高风险裁决所 V292.0 赫斯特指数增强版] 启动...")
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        
        risk_params = get_params_block(self.strategy, 'four_layer_scoring_params').get('risk_scoring', {})
        risk_rules = risk_params.get('signals', {})
        
        # --- 步骤 1: 基础风险分计算 (逻辑不变) ---
        # 直接从配置文件中读取所有风险规则并计算基础分
        total_risk_score = pd.Series(0.0, index=df.index)
        risk_score_df = pd.DataFrame(index=df.index)
        default_series = pd.Series(False, index=df.index)

        for rule_name, score in risk_rules.items():
            signal_series = atomic_states.get(rule_name, default_series)
            if signal_series.any():
                total_risk_score.loc[signal_series] += score
                risk_score_df[rule_name] = signal_series * score
        
        # --- 步骤 2: 【核心改造】应用“战场环境”调节器 ---
        # 我们将创建一个全局的风险乘数，根据赫斯特指数的状态进行调整
        risk_multiplier = pd.Series(1.0, index=df.index)

        # 2.1 风险放大器: 在震荡或随机市场中，放大风险
        is_mean_reversion = atomic_states.get('FRACTAL_STATE_MEAN_REVERSION', default_series)
        is_random_walk = atomic_states.get('FRACTAL_STATE_RANDOM_WALK', default_series)
        is_unstable_market = is_mean_reversion | is_random_walk
        
        if is_unstable_market.any():
            instability_multiplier = 1.3 # 风险权重提升30%
            risk_multiplier.loc[is_unstable_market] *= instability_multiplier
            print(f"          -> [风险放大器] 已为 {is_unstable_market.sum()} 天的“不稳定市场”应用 {instability_multiplier}x 风险乘数。")

        # 2.2 风险对冲器: 在强趋势市场中，折减非核心风险
        is_strong_trend = atomic_states.get('FRACTAL_STATE_STRONG_TREND', default_series)
        if is_strong_trend.any():
            # 定义哪些风险是“核心/结构性”风险，它们不应被折减
            core_risks = {
                "CONTEXT_RECENT_DISTRIBUTION_PRESSURE",
                "COGNITIVE_RISK_BREAKOUT_DISTRIBUTION",
                "RISK_CONTEXT_LONG_TERM_DISTRIBUTION",
                "FRACTAL_RISK_TOP_DIVERGENCE", # 新增的分形风险也是核心风险
                "STRUCTURE_TOPPING_DANGER_S"
            }
            
            # 对非核心风险应用折减系数
            trend_reduction_factor = 0.7 # 非核心风险重要性降低30%
            for col in risk_score_df.columns:
                if col not in core_risks:
                    # 使用 .loc 对特定条件下的列进行乘法操作
                    risk_score_df.loc[is_strong_trend, col] *= trend_reduction_factor
            
            print(f"          -> [风险对冲器] 已为 {is_strong_trend.sum()} 天的“强趋势市场”期间，对非核心风险应用了 {trend_reduction_factor}x 折减系数。")

        # 重新计算应用了“风险对冲器”后的总分
        total_risk_score = risk_score_df.sum(axis=1)
        
        # 将“风险放大器”的全局乘数应用到总分上
        total_risk_score *= risk_multiplier

        # --- 步骤 3: 战略机会覆盖 (逻辑微调) ---
        # 这里的逻辑可以保持，但现在它是在经过环境调节后的风险分基础上进行覆盖
        has_strategic_opportunity = atomic_states.get('COGNITIVE_PATTERN_LOCK_CHIP_RALLY', default_series) # 简化为最强的机会信号
        if has_strategic_opportunity.any():
            strategic_coverage_factor = get_param_value(risk_params.get('strategic_coverage_factor'), 0.3)
            total_risk_score = total_risk_score.where(~has_strategic_opportunity, total_risk_score * strategic_coverage_factor)
            print(f"          -> [战略覆盖已执行！] 已对 {has_strategic_opportunity.sum()} 天的总风险分应用了 {strategic_coverage_factor} 的覆盖系数。")
        
        print("        -> [最高风险裁决所 V292.0] 风险评估完成。")
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
