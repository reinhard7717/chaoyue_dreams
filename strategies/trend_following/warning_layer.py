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
        【V503.0 新增】【代码优化】风险动量引擎 (Risk Momentum Engine)
        - 核心职责: 计算总风险分的“斜率”(变化速度)和“加速度”(速度的变化)，
                    以实现对风险趋势的预测性分析。
        - 优化说明: 1. 在 .rolling().apply() 中设置 raw=True，将NumPy数组直接传递给计算函数，减少开销。
                    2. 使用 np.select 和列表推导式替代了用于生成最终报告的 for 循环，实现了全向量化。
        """
        # print("          -> [风险动量引擎 V503.0] 启动，正在计算风险势能...")
        p = get_params_block(self.strategy, 'four_layer_scoring_params')
        momentum_params = p.get('holding_warning_params', {}).get('risk_momentum_params', {})
        
        if not get_param_value(momentum_params.get('enabled'), False):
            return pd.Series([{} for _ in range(len(total_risk_score_series))], index=total_risk_score_series.index)

        window = get_param_value(momentum_params.get('slope_window'), 3)
        accel_threshold = get_param_value(momentum_params.get('accel_threshold'), 20.0)

        # 定义一个辅助函数，用于在 raw=True 模式下安全地计算斜率
        def calculate_slope(y: np.ndarray) -> float:
            if np.isnan(y).any() or len(y) < window:
                return np.nan
            return linregress(np.arange(len(y)), y).slope

        # 1. 计算风险分的斜率 (速度)，使用 raw=True 提升性能
        risk_slope = total_risk_score_series.rolling(window).apply(calculate_slope, raw=True).fillna(0)

        # 2. 计算风险分的加速度 (速度的变化)
        risk_accel = risk_slope.diff().fillna(0)

        # 3. 定义动量状态
        is_escalating = (risk_slope > 0) & (risk_accel > accel_threshold)
        is_decelerating = (risk_slope > 0) & (risk_accel < -accel_threshold)
        is_improving = risk_slope < 0

        # 4. 使用向量化操作生成量化的动量报告
        conditions = [is_escalating, is_decelerating, is_improving]
        choices = ["ESCALATING", "DECELERATING", "IMPROVING"]
        states = np.select(conditions, choices, default="STABLE")
        
        # 使用列表推导式构建报告字典列表，只为非稳定状态构建
        reports = [
            {'momentum_state': state, 'risk_slope': round(slope, 2), 'risk_accel': round(accel, 2)} if state != "STABLE" else {}
            for state, slope, accel in zip(states, risk_slope, risk_accel)
        ]
        
        return pd.Series(reports, index=total_risk_score_series.index)

    def _diagnose_risk_dynamics(self, combined_risk_details_df: pd.DataFrame) -> pd.Series:
        """
        【V502.1 健壮性修复版】
        - 核心修复: 修复了当DataFrame索引有名称时，`reset_index().melt()` 因找不到 'index' 列而崩溃的问题。
        - 修复逻辑: 通过 `reset_index(names='trade_time')` 强制将索引列命名为 'trade_time'，
                    并在后续的 melt, merge, groupby 操作中统一使用此名称，消除了对默认行为的依赖。
        """
        # print("          -> [定量诊断大脑 V502.1 健壮性修复版] 启动，正在进行风险量化分析...")
        
        # 使用全向量化操作替代 for 循环
        if combined_risk_details_df.empty:
            return pd.Series([{} for _ in range(len(combined_risk_details_df))], index=combined_risk_details_df.index)

        
        # 强制将索引转换为一个名为 'trade_time' 的列，避免依赖不确定的默认列名 'index'。
        # 步骤1: 将当日和昨日的风险数据转换为长格式
        risk_today_long = combined_risk_details_df.reset_index(names='trade_time').melt(
            id_vars='trade_time', var_name='risk_name', value_name='score'
        )
        risk_yesterday_long = combined_risk_details_df.shift(1).reset_index(names='trade_time').melt(
            id_vars='trade_time', var_name='risk_name', value_name='prev_score'
        )
        
        # 步骤2: 合并数据，使每一行包含当日和昨日的分数
        # 更新 merge 的 on 参数以匹配新的列名 'trade_time'。
        merged_risks = pd.merge(risk_today_long, risk_yesterday_long, on=['trade_time', 'risk_name']).fillna(0)
        
        
        # 步骤3: 过滤掉没有风险变化的行以提高效率
        active_risks = merged_risks[(merged_risks['score'] > 0) | (merged_risks['prev_score'] > 0)].copy()
        if active_risks.empty:
            return pd.Series([{} for _ in range(len(combined_risk_details_df))], index=combined_risk_details_df.index)

        # 步骤4: 向量化计算变化量、变化率和风险类别
        active_risks['change'] = active_risks['score'] - active_risks['prev_score']
        active_risks['change_pct'] = (active_risks['change'] / active_risks['prev_score'].replace(0, np.nan) * 100).fillna(99999.0)
        
        conditions = [
            (active_risks['score'] > 0) & (active_risks['prev_score'] == 0),
            (active_risks['score'] > 0) & (active_risks['prev_score'] > 0),
            (active_risks['score'] == 0) & (active_risks['prev_score'] > 0)
        ]
        choices = ['new', 'persistent', 'resolved']
        active_risks['category'] = np.select(conditions, choices, default=None)
        
        # 步骤5: 准备用于格式化的元数据
        active_risks['cn_name'] = active_risks['risk_name'].apply(lambda x: self.risk_metadata.get(x, {}).get('cn_name', x))
        active_risks['abs_change'] = active_risks['change'].abs()

        # 步骤6: 按日期和类别分组，并将每个组转换为字典列表
        def format_group(group):
            return group[['risk_name', 'cn_name', 'score', 'prev_score', 'change', 'change_pct']].rename(columns={'risk_name': 'name'}).round(2).to_dict('records')

        
        # 更新 groupby 的分组键以匹配新的列名 'trade_time'。
        grouped = active_risks.sort_values('abs_change', ascending=False).groupby(['trade_time', 'category']).apply(format_group)
        
        
        # 步骤7: 将分组结果重新组合成最终的 Series
        final_summary = grouped.unstack(level='category').apply(lambda row: row.dropna().to_dict(), axis=1)
        
        # 确保所有日期都有记录，即使是空字典
        return final_summary.reindex(combined_risk_details_df.index, fill_value={})

    def calculate_risk_score(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V600.0 统一风险计分中心版】
        - 核心重构: 本函数现在是唯一的风险计分中心。它不再区分“致命”和“常规”风险。
        - 统一配置驱动: 从配置中加载所有风险信号（包括原致命风险和常规风险）及其权重。
        - 数值化计分: 采用 `风险贡献分 = 信号强度 * 权重` 的方式计算，完美适配纯数值化系统。
        - 简化输出: 返回一个总的 risk_score Series 和一个包含所有风险项详情的 DataFrame。
        """
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        
        # 1. 从配置中加载所有风险信号及其权重
        critical_params = scoring_params.get('critical_exit_params', {})
        warning_params = scoring_params.get('holding_warning_params', {})
        
        # 合并致命风险和常规风险的配置
        all_risk_rules = {}
        all_risk_rules.update(critical_params.get('signals', {}))
        all_risk_rules.update(warning_params.get('signals', {}))

        risk_details_df = pd.DataFrame(index=df.index)
        default_series = pd.Series(0.0, index=df.index)

        def get_clipped_score(signal_name):
            """辅助函数：获取信号分并确保其非负"""
            return atomic_states.get(signal_name, default_series).clip(lower=0)

        # 2. 循环遍历所有风险规则，计算风险贡献分
        for rule_name, weight in all_risk_rules.items():
            if rule_name.startswith("说明_"):
                continue
            
            # 获取数值化的信号强度 (0-1)
            signal_strength = get_clipped_score(rule_name)
            
            if signal_strength.any():
                # 计算该风险项的最终贡献分
                risk_contribution = signal_strength * weight
                risk_details_df[rule_name] = risk_contribution

        # 3. 计算总风险分，并应用市场状态乘数
        total_risk_score = risk_details_df.sum(axis=1)
        
        risk_multiplier = pd.Series(1.0, index=df.index)
        
        # 市场不稳定时，放大风险
        is_mean_reversion = get_clipped_score('FRACTAL_STATE_MEAN_REVERSION') > 0
        is_random_walk = get_clipped_score('FRACTAL_STATE_RANDOM_WALK') > 0
        is_unstable_market = is_mean_reversion | is_random_walk
        if is_unstable_market.any():
            instability_multiplier = get_param_value(warning_params.get('instability_multiplier'), 1.3)
            risk_multiplier.loc[is_unstable_market] *= instability_multiplier
            
        # 强趋势市场中，对非核心风险进行折减
        is_strong_trend = get_clipped_score('FRACTAL_STATE_STRONG_TREND') > 0
        if is_strong_trend.any():
            core_risks = set(get_param_value(warning_params.get('core_risks_in_trend'), []))
            trend_reduction_factor = get_param_value(warning_params.get('trend_reduction_factor'), 0.7)
            for col in risk_details_df.columns:
                if col not in core_risks:
                    # 直接在详情DF上修改，这样总分计算时会自动体现
                    risk_details_df.loc[is_strong_trend, col] *= trend_reduction_factor
        
        # 重新计算应用了趋势折减后的总分
        adjusted_total_risk_score = risk_details_df.sum(axis=1)
        
        # 应用最终的乘数
        adjusted_total_risk_score *= risk_multiplier

        # 战略机会覆盖，最终折减风险
        has_strategic_opportunity = get_clipped_score('COGNITIVE_PATTERN_LOCK_CHIP_RALLY') > 0
        if has_strategic_opportunity.any():
            strategic_coverage_factor = get_param_value(warning_params.get('strategic_coverage_factor'), 0.3)
            adjusted_total_risk_score = adjusted_total_risk_score.where(
                ~has_strategic_opportunity, 
                adjusted_total_risk_score * strategic_coverage_factor
            )
        
        # 4. 返回总风险分和风险详情
        # 注意：原有的健康度诊断逻辑 (_diagnose_risk_momentum, _diagnose_risk_dynamics) 已被解耦
        # 主流程现在只关心总风险分和其构成。
        return adjusted_total_risk_score, risk_details_df

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
