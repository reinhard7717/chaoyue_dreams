# 文件: strategies/trend_following/warning_layer.py
# 预警层
import pandas as pd
import numpy as np
from scipy.stats import linregress
from typing import Dict, List, Tuple
from .utils import get_params_block, get_param_value

class WarningLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        self.risk_metadata = scoring_params.get('score_type_map', {})

    # 持仓健康诊断大脑
    def _diagnose_risk_momentum(self, total_risk_score_series: pd.Series) -> pd.Series:
        """
        【V503.0 新增】风险动量引擎 (Risk Momentum Engine)
        - 核心职责: 计算总风险分的“斜率”(变化速度)和“加速度”(速度的变化)，
                    以实现对风险趋势的预测性分析。
        """
        # print("          -> [风险动量引擎 V503.0] 启动，正在计算风险势能...")
        p = get_params_block(self.strategy, 'four_layer_scoring_params')
        momentum_params = p.get('holding_warning_params', {}).get('risk_momentum_params', {})
        
        if not get_param_value(momentum_params.get('enabled'), False):
            return pd.Series([{} for _ in range(len(total_risk_score_series))], index=total_risk_score_series.index)

        window = get_param_value(momentum_params.get('slope_window'), 3)
        accel_threshold = get_param_value(momentum_params.get('accel_threshold'), 20.0)

        # 1. 计算风险分的斜率 (速度)
        risk_slope = total_risk_score_series.rolling(window).apply(
            lambda y: linregress(np.arange(len(y)), y).slope if len(y.dropna()) == window else np.nan,
            raw=False
        ).fillna(0)

        # 2. 计算风险分的加速度 (速度的变化)
        risk_accel = risk_slope.diff().fillna(0)

        # 3. 定义动量状态
        is_escalating = (risk_slope > 0) & (risk_accel > accel_threshold)
        is_decelerating = (risk_slope > 0) & (risk_accel < -accel_threshold)
        is_improving = risk_slope < 0

        # 4. 生成量化的动量报告
        momentum_summary = pd.Series([{} for _ in range(len(total_risk_score_series))], index=total_risk_score_series.index)
        for idx in total_risk_score_series.index:
            state = "STABLE" # 默认稳定
            if is_escalating.at[idx]:
                state = "ESCALATING" # 加速恶化
            elif is_decelerating.at[idx]:
                state = "DECELERATING" # 减速恶化
            elif is_improving.at[idx]:
                state = "IMPROVING" # 改善中
            
            # 只在有显著动量时记录，避免报告冗余
            if state != "STABLE":
                momentum_summary.at[idx] = {
                    'momentum_state': state,
                    'risk_slope': round(risk_slope.at[idx], 2),
                    'risk_accel': round(risk_accel.at[idx], 2)
                }
        
        return momentum_summary

    def _diagnose_risk_dynamics(self, combined_risk_details_df: pd.DataFrame) -> pd.Series:
        """
        【V502.0 定量诊断大脑】
        - 核心升级: 实现从“定性”到“定量”的飞跃。
        - 产出变更: risk_change_summary 中的 new/persistent/resolved 不再是简单的名称列表，
                    而是包含 name, cn_name, score, prev_score, change, change_pct
                    等丰富量化信息的对象列表。
        """
        # print("          -> [定量诊断大脑 V502.0] 启动，正在进行风险量化分析...")
        
        risk_df_yesterday = combined_risk_details_df.shift(1).fillna(0)
        risk_change_summary = pd.Series([{} for _ in range(len(combined_risk_details_df))], index=combined_risk_details_df.index)

        # 获取所有出现过的风险信号列名
        all_risk_columns = combined_risk_details_df.columns

        for idx in combined_risk_details_df.index:
            # 如果当天和昨天都没有任何风险，则跳过，提高效率
            if combined_risk_details_df.loc[idx].sum() == 0 and risk_df_yesterday.loc[idx].sum() == 0:
                continue

            new_risks_details = []
            persistent_risks_details = []
            resolved_risks_details = []

            for risk_name in all_risk_columns:
                today_score = combined_risk_details_df.at[idx, risk_name]
                prev_score = risk_df_yesterday.at[idx, risk_name]

                # 如果今天和昨天该风险的分数都为0，则跳过
                if today_score == 0 and prev_score == 0:
                    continue

                # 计算变化量和变化率
                change = today_score - prev_score
                if prev_score != 0:
                    change_pct = (change / prev_score) * 100
                else:
                    # 如果前一天分数为0，则变化率为无穷大或无意义，记为 99999.0 以便排序
                    change_pct = 99999.0 if change > 0 else 0.0
                
                # 准备风险详情对象
                risk_detail_obj = {
                    'name': risk_name,
                    'cn_name': self.risk_metadata.get('cn_name', risk_name), # 从缓存的元数据获取中文名
                    'score': round(today_score, 2),
                    'prev_score': round(prev_score, 2),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2)
                }

                # 根据分数变化，将风险归类
                if today_score > 0 and prev_score == 0:
                    new_risks_details.append(risk_detail_obj)
                elif today_score > 0 and prev_score > 0:
                    persistent_risks_details.append(risk_detail_obj)
                elif today_score == 0 and prev_score > 0:
                    resolved_risks_details.append(risk_detail_obj)
            
            # 只有在确实有风险变化时才记录
            if new_risks_details or persistent_risks_details or resolved_risks_details:
                # 对列表进行排序，让最重要的变化排在前面
                # 排序规则：优先按分数变化的绝对值降序
                new_risks_details.sort(key=lambda x: abs(x['change']), reverse=True)
                persistent_risks_details.sort(key=lambda x: abs(x['change']), reverse=True)
                resolved_risks_details.sort(key=lambda x: abs(x['change']), reverse=True)

                risk_change_summary.at[idx] = {
                    'new': new_risks_details,
                    'persistent': persistent_risks_details,
                    'resolved': resolved_risks_details
                }

        return risk_change_summary

    def calculate_risk_score(self, critical_risk_details: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.Series]:
        """
        【V503.1 风险融合版】
        """
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        warning_params = scoring_params.get('holding_warning_params', {})
        warning_rules = warning_params.get('signals', {})
        
        risk_details_df = pd.DataFrame(index=df.index)
        default_series = pd.Series(False, index=df.index)
        for rule_name, score in warning_rules.items():
            signal_series = atomic_states.get(rule_name, default_series)
            if signal_series.any():
                risk_details_df[rule_name] = signal_series * score
        
        combined_risk_details_df = risk_details_df.add(critical_risk_details, fill_value=0)
        
        risk_multiplier = pd.Series(1.0, index=df.index)
        is_mean_reversion = atomic_states.get('FRACTAL_STATE_MEAN_REVERSION', default_series)
        is_random_walk = atomic_states.get('FRACTAL_STATE_RANDOM_WALK', default_series)
        is_unstable_market = is_mean_reversion | is_random_walk
        
        if is_unstable_market.any():
            instability_multiplier = 1.3
            risk_multiplier.loc[is_unstable_market] *= instability_multiplier

        is_strong_trend = atomic_states.get('FRACTAL_STATE_STRONG_TREND', default_series)
        if is_strong_trend.any():
            # 在强趋势中，我们只关心最核心的、不可被趋势消化的风险
            core_risks = {
                "RISK_CHIP_STRUCTURE_CRITICAL_FAILURE",
                "STRUCTURE_TOPPING_DANGER_S",
                "CONTEXT_RECENT_DISTRIBUTION_PRESSURE", 
                "COGNITIVE_RISK_BREAKOUT_DISTRIBUTION",
                "FRACTAL_RISK_TOP_DIVERGENCE",
            }
            trend_reduction_factor = 0.7
            for col in combined_risk_details_df.columns:
                if col not in core_risks:
                    combined_risk_details_df.loc[is_strong_trend, col] *= trend_reduction_factor

        adjusted_total_risk_score = combined_risk_details_df.sum(axis=1)
        adjusted_total_risk_score *= risk_multiplier
        
        has_strategic_opportunity = atomic_states.get('COGNITIVE_PATTERN_LOCK_CHIP_RALLY', default_series)
        if has_strategic_opportunity.any():
            strategic_coverage_factor = get_param_value(warning_params.get('strategic_coverage_factor'), 0.3)
            adjusted_total_risk_score = adjusted_total_risk_score.where(~has_strategic_opportunity, adjusted_total_risk_score * strategic_coverage_factor)

        momentum_summary = self._diagnose_risk_momentum(adjusted_total_risk_score)
        composition_summary = self._diagnose_risk_dynamics(combined_risk_details_df)

        final_health_summary = pd.Series([{} for _ in range(len(df))], index=df.index)
        for idx in df.index:
            final_report = composition_summary.at[idx]
            momentum_report = momentum_summary.at[idx]
            if momentum_report:
                final_report['momentum'] = momentum_report
            final_health_summary.at[idx] = final_report

        return adjusted_total_risk_score, combined_risk_details_df, final_health_summary

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
