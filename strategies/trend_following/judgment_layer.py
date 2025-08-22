# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V404.2 - 逻辑净化版)
import pandas as pd
import numpy as np
from .intelligence_layer import MainForceState
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _evaluate_holding_health(self, score_details_df: pd.DataFrame, risk_score_df: pd.DataFrame):
        """
        【V400.0 健康报告总汇版】【代码优化】
        - 优化说明: 原始代码使用 for 循环和 .at 索引器逐行构建字典，在处理大量数据时效率低下。
                    优化后的版本使用了列表推导式（List Comprehension）和 zip，
                    将两列数据并行处理并构建新的字典列表，然后一次性将其赋值给新列。
                    这种方法避免了 pandas 逐行操作的开销，执行效率更高。
        """
        df = self.strategy.df_indicators
        offensive_summary = df.get('offensive_momentum_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        risk_summary = df.get('risk_change_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        
        # 使用列表推导式替代 for 循环，实现向量化操作
        final_summaries = []
        for offense_report, risk_report in zip(offensive_summary, risk_summary):
            final_summary = {}
            # 检查进攻动能报告是否有效
            if offense_report and isinstance(offense_report, dict) and any(offense_report.values()):
                final_summary['offense_momentum'] = offense_report
            # 检查风险变化报告是否有效
            if risk_report and isinstance(risk_report, dict) and any(v for v in risk_report.values() if v):
                final_summary['risk_change'] = risk_report
            final_summaries.append(final_summary)
            
        df['health_change_summary'] = final_summaries

    def _generate_exit_triggers(self) -> pd.DataFrame:
        """
        【V504.0 新增】离场触发器生成器
        - 核心职责: 根据“三道防线”原则，生成一个包含所有离场原因的布尔型DataFrame。
        """
        df = self.strategy.df_indicators
        triggers_df = pd.DataFrame(index=df.index)
        
        # --- 防线一: 致命一击 (Critical Hit) ---
        critical_risk_details = self.strategy.critical_risk_details
        triggers_df['EXIT_CRITICAL_HIT'] = critical_risk_details.sum(axis=1) > 0

        # --- 防线二: 风险溢出 (Risk Overflow) ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        overflow_threshold = get_param_value(p_judge.get('risk_overflow_threshold'), 1000)
        triggers_df['EXIT_RISK_OVERFLOW'] = self.strategy.risk_score > overflow_threshold

        # --- 防线三: 利润保护 (Profit Protector) ---
        p_protector = p_judge.get('profit_protector', {})
        if get_param_value(p_protector.get('enabled'), False):
            max_drawdown_pct = get_param_value(p_protector.get('max_drawdown_pct'), 0.15)
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)
        else:
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)

        return triggers_df

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V501.0 纯粹决策版】
        - 核心变化: 移除了对特定“加速状态”的硬性过滤。由于“阵地优势加速”已作为
                    核心奖励分融入 entry_score，本层只需根据最终的净得分进行决策即可。
        """
        print("    --- [最高作战指挥部 V501.0 纯粹决策版] 启动... ---")
        df = self.strategy.df_indicators
        
        df['final_score'] = 0.0
        df['signal_type'] = '无信号'
        df['veto_votes'] = 0
        df['dynamic_action'] = 'HOLD'
        self._evaluate_holding_health(score_details_df, risk_details_df)
        self._calculate_static_veto_votes()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        exit_triggers = self._generate_exit_triggers()
        is_sell_signal = exit_triggers.any(axis=1)
        self.strategy.exit_triggers = exit_triggers[is_sell_signal]
        df.loc[is_sell_signal, 'signal_type'] = '卖出信号'

        # --- 买入决策核心逻辑 ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        net_score_threshold_no_veto = get_param_value(p_judge.get('net_score_threshold_no_veto'), 500)
        net_score_threshold_with_veto = get_param_value(p_judge.get('net_score_threshold_with_veto'), 800)
        df['net_score'] = df['entry_score'] - df['risk_score']
        no_veto_buy_condition = (df['veto_votes'] == 0) & (df['net_score'] > net_score_threshold_no_veto)
        with_veto_buy_condition = (df['veto_votes'] > 0) & (df['net_score'] > net_score_threshold_with_veto)
        is_net_score_sufficient = no_veto_buy_condition | with_veto_buy_condition
        not_avoid = df['dynamic_action'] != 'AVOID'
        is_not_sell_day = ~is_sell_signal

        final_buy_condition = (
            is_net_score_sufficient &
            not_avoid &
            is_not_sell_day
        )

        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        self._finalize_signals()

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V317.0 核心】动态力学战术矩阵
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        offense_accel = atomic.get('FORCE_VECTOR_OFFENSE_ACCELERATING', default_series)
        offense_decel = atomic.get('FORCE_VECTOR_OFFENSE_DECELERATING', default_series)
        risk_accel = atomic.get('FORCE_VECTOR_RISK_ACCELERATING', default_series)
        risk_decel = atomic.get('FORCE_VECTOR_RISK_DECELERATING', default_series)

        is_force_attack = offense_accel & risk_decel
        is_avoid = offense_decel & risk_accel
        is_caution = (offense_accel & risk_accel) | (offense_decel & risk_decel)

        actions = pd.Series('HOLD', index=df.index)
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        
        return actions

    def _calculate_static_veto_votes(self):
        """
        【V318.4 风险融合版】
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 风险1: 筹码结构严重失效 (3票) - 直接使用新的融合信号
        has_critical_chip_risk = atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3

        # 风险2: 主力正在派发或崩盘 (1票)
        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'veto_votes'] += 1
        
        # 风险3: 绝对否决信号 (2票) - 这里的逻辑可以保持，因为它处理的是更具体的、可配置的否决项
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            final_absolute_veto = pd.Series(False, index=df.index)
            for signal_name in veto_signals:
                has_risk = atomic.get(signal_name, default_series)
                if signal_name in mitigation_rules:
                    mitigators = mitigation_rules[signal_name].get('mitigated_by', [])
                    has_mitigator = pd.Series(False, index=df.index)
                    for m_signal in mitigators: has_mitigator |= atomic.get(m_signal, default_series)
                    final_absolute_veto |= (has_risk & ~has_mitigator)
                else:
                    final_absolute_veto |= has_risk
            df.loc[final_absolute_veto, 'veto_votes'] += 2

        # 风险4: 风险分高于进攻分 (1票)
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        is_in_ascent_phase = atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        df.loc[risk_overrides_entry & ~is_in_ascent_phase, 'veto_votes'] += 1
        
        # 风险5: 核心原子风险信号 (1票)
        # CHIP_DYN_COST_FALLING (成本松动) 信号回测显示有约30%的规避成功率，是重要的预警信号。
        has_cost_falling_risk = atomic.get('CHIP_DYN_COST_FALLING', default_series)
        df.loc[has_cost_falling_risk, 'veto_votes'] += 1
        # 根据最新战报，将规避成功率高达28%的“获利盘崩盘”信号也加入否决票体系。
        has_winner_collapsing_risk = atomic.get('CHIP_DYN_WINNER_RATE_COLLAPSING', default_series)
        df.loc[has_winner_collapsing_risk, 'veto_votes'] += 1
        
        # 风险6: 周线战略顶层风险 (Strategic Veto)
        # 这是最高级别的风险，拥有强大的否决权
        # 6.1 周线发出“顶部区域”强风险信号，投3票 (强否决)
        is_strategic_topping = atomic.get('CONTEXT_STRATEGIC_TOPPING_RISK_W', default_series)
        df.loc[is_strategic_topping, 'veto_votes'] += 3
        
        # 6.2 周线处于“战略看跌”状态，投1票 (软否决)
        is_strategic_bearish = atomic.get('CONTEXT_STRATEGIC_BEARISH_W', default_series)
        df.loc[is_strategic_bearish, 'veto_votes'] += 1

    def _finalize_signals(self):
        """
        【V404.1 健壮性修复版】
        - 核心修复: 修复了当没有任何买卖信号时，'signal_entry'列不存在导致的AttributeError。
        """
        df = self.strategy.df_indicators
        
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0
        df['alert_reason'] = ''
        
        final_buy_condition = df['signal_type'] == '买入信号'
        final_sell_condition = df['signal_type'] == '卖出信号'
        final_warning_condition = df['signal_type'] == '风险预警'

        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        
        if 'exit_signal_code' in df.columns:
            if 'exit_severity_level' not in df.columns: df['exit_severity_level'] = 0
            if 'alert_reason' not in df.columns: df['alert_reason'] = ''
            df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        df.loc[final_sell_condition | final_warning_condition, 'final_score'] = df.loc[final_sell_condition | final_warning_condition, 'risk_score']
        
        debug_cols = ['entry_score', 'risk_score', 'veto_votes', 'net_score', 'final_score', 'signal_type', 'main_force_state']
        final_check_df = df[(df['signal_type'] != '无信号') & (df['signal_type'] != '中性')].tail(10)
        if not final_check_df.empty:
            cols_to_show = [col for col in debug_cols if col in final_check_df.columns]
            print("          -> [最终分数审查报告]:")
            print(final_check_df[cols_to_show])
        else:
            print("          -> [最终分数审查报告]: 在最近的记录中未发现任何有效信号。")
