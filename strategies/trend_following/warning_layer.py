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

        # --- 代码修改开始 ---
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
        # --- 代码修改结束 ---
        
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

        # --- 代码修改开始 ---
        # 更新 groupby 的分组键以匹配新的列名 'trade_time'。
        grouped = active_risks.sort_values('abs_change', ascending=False).groupby(['trade_time', 'category']).apply(format_group)
        # --- 代码修改结束 ---
        
        # 步骤7: 将分组结果重新组合成最终的 Series
        final_summary = grouped.unstack(level='category').apply(lambda row: row.dropna().to_dict(), axis=1)
        
        # 确保所有日期都有记录，即使是空字典
        return final_summary.reindex(combined_risk_details_df.index, fill_value={})

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
        
        # --- 精英原子风险计分 (Elite Atomic Risk Scoring) ---
        # 目的: 硬编码计入最关键的S级风险信号，防止因配置疏忽而遗漏。
        elite_atomic_risks = {
            'RISK_STATIC_DYN_COLLAPSE_S': 500,          # 静态-动态融合崩塌
            'RISK_HIGH_VOL_DIVERGENT_RALLY_S': 450,     # 高波动区的背离诱多
            'RISK_DYN_STRUCTURAL_WEAKNESS_RALLY_S': 400,# 结构性衰竭反弹
            'RISK_MTF_RSI_BEARISH_DIVERGENCE_S': 350,   # 周线与日线RSI顶背离
            'RISK_MA_DEATH_CROSS_CONFIRMED_S': 300,     # 均线死亡交叉确认
            # A级风险: 获利盘的平均利润在减少，是趋势弱化的重要早期预警。
            'RISK_BEHAVIOR_PROFIT_CUSHION_SHRINKING_A': 350,
            # B级风险: 上方套牢盘越来越多，形成阻力，表明上涨乏力。
            'RISK_BEHAVIOR_BUILDING_OVERHEAD_PRESSURE_B': 300,
            # A级风险: 短中长周期都在派发，是系统性出货的明确信号。
            'RISK_CHIP_DIVERGING_RESONANCE_A': 400,
            # A级风险: 主力堡垒看似稳固，但内部已开始瓦解，是危险的背离信号。
            'SCENARIO_FORTRESS_INTERNAL_COLLAPSE_A': 420,
            # S级风险: 战略派发背景下的任何拉升都应被视为高风险事件。
            'RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S': 550,
            # S级风险: 市场引擎失速，上涨效率崩溃，是趋势即将终结的强烈信号。
            'RISK_DYN_MARKET_ENGINE_STALLING_S': 600,
            # S级风险: 获利盘恐慌加速，是市场情绪崩溃、踩踏式下跌的预警。
            'RISK_DYN_PANIC_SELLING_ACCELERATING_S': 580,
            # S级风险: 认知层合成的顶部危险结构信号，代表多重风险共振。
            'STRUCTURE_TOPPING_DANGER_S': 520,
            # A级风险: 放量杀跌，是恐慌或主力出货的直接体现，是强烈的风险预警。
            'RISK_VOL_PRICE_SPIKE_DOWN_A': 480,
            # F级风险: 认知层判定的下跌通道，是绝对的逆风环境，风险极高。
            'STRUCTURE_BEARISH_CHANNEL_F': 450,
            # B级风险: MACD死叉，经典的短期动能转弱信号。
            'RISK_TRIGGER_MACD_DEATH_CROSS_B': 250,
            # A级风险: 主峰高位派发嫌疑，在高位区域发生激烈换手但价格滞涨，是典型的派发行为。
            'RISK_PEAK_BATTLE_DISTRIBUTION_A': 460,
            # B级风险: 散户狂热风险，股价大涨但主要由散户买盘驱动，是情绪过热的危险信号。
            'RISK_FUND_FLOW_RETAIL_FOMO_B': 310,
            # S级风险: 结构性长期超涨，股价长期严重偏离均线，回归压力巨大，结构不稳定。
            'RISK_STRUCTURE_OVEREXTENDED_LONG_TERM_S': 470,
            # S级风险: 多维共振超涨，日线和周线同时严重超涨，是极度危险的顶部共振信号。
            'RISK_STRUCTURE_MTF_OVEREXTENDED_RESONANCE_S': 530,
            # B级风险: 市场处于均值回归状态，追涨策略的风险显著增加，突破很可能是陷阱。
            'STRUCTURE_REGIME_MEAN_REVERTING': 280,
        }
        for risk_name, score in elite_atomic_risks.items():
            signal_series = atomic_states.get(risk_name, default_series)
            if signal_series.any():
                current_score = risk_details_df.get(risk_name, pd.Series(0.0, index=df.index))
                risk_details_df[risk_name] = current_score.add(signal_series * score, fill_value=0)
                # print(f"          -> [精英原子风险] 侦测到高危信号 “{risk_name}”，增加 {score} 风险分！")

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
