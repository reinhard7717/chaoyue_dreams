# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V404.1 - 健壮性修复版)
import pandas as pd
import numpy as np
from .intelligence_layer import MainForceState
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _evaluate_holding_health(self, score_details_df: pd.DataFrame, risk_score_df: pd.DataFrame):
        """
        【V400.0 健康报告总汇版】
        """
        df = self.strategy.df_indicators
        df['health_change_summary'] = [{} for _ in range(len(df))]
        offensive_summary = df.get('offensive_momentum_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        risk_summary = df.get('risk_change_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        for idx in df.index:
            final_summary = {}
            offense_report = offensive_summary.at[idx]
            if offense_report and isinstance(offense_report, dict) and any(offense_report.values()):
                final_summary['offense_momentum'] = offense_report
            risk_report = risk_summary.at[idx]
            if risk_report and isinstance(risk_report, dict) and any(v for v in risk_report.values() if v):
                final_summary['risk_change'] = risk_report
            if final_summary:
                df.at[idx, 'health_change_summary'] = final_summary

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V404.0 A股实战版】
        """
        print("    --- [最高作战指挥部 V404.0 A股实战版] 启动... ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        df['final_score'] = 0.0
        df['signal_type'] = '无信号'
        df['veto_votes'] = 0
        df['dynamic_action'] = 'HOLD'
        df['max_allowed_votes'] = 0

        self._evaluate_holding_health(score_details_df, risk_details_df)
        self._calculate_static_veto_votes()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        
        exit_params = self.strategy.unified_config.get('exit_strategy_params', {})
        exit_thresholds = exit_params.get('exit_threshold_params', {})
        warning_thresholds = exit_params.get('warning_threshold_params', {})
        all_thresholds = []
        if exit_thresholds:
            for level_info in exit_thresholds.values():
                all_thresholds.append({'level': level_info['level'], 'type': '卖出信号'})
        if warning_thresholds:
            for level_info in warning_thresholds.values():
                all_thresholds.append({'level': level_info['level'], 'type': '风险预警'})
        sorted_thresholds = sorted(all_thresholds, key=lambda x: x['level'], reverse=True)
        for rule in sorted_thresholds:
            threshold = rule['level']
            signal_type = rule['type']
            condition = (df['risk_score'] >= threshold) & (df['signal_type'] == '无信号')
            df.loc[condition, 'signal_type'] = signal_type

        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {}).keys()
        valid_pos_cols = [col for col in positional_rules if col in score_details_df.columns]
        positional_score = score_details_df[valid_pos_cols].sum(axis=1) if valid_pos_cols else pd.Series(0.0, index=df.index)

        buy_params = get_params_block(self.strategy, 'dynamic_mechanics_params').get('tactical_matrix_rules', {})
        min_positional_score = get_param_value(buy_params.get('min_positional_score'), 200)
        is_foundation_solid = positional_score >= min_positional_score

        hard_veto_signals = get_param_value(buy_params.get('hard_veto_signals'), [])
        has_hard_veto = pd.Series(False, index=df.index)
        for signal in hard_veto_signals:
            has_hard_veto |= atomic.get(signal, default_series)

        tolerance_tiers = get_param_value(buy_params.get('tolerance_tiers'), [])
        for tier in sorted(tolerance_tiers, key=lambda x: x.get('score_min', 0)):
            score_min = tier.get('score_min', 0)
            score_max = tier.get('score_max', float('inf'))
            max_votes = tier.get('max_veto_votes', 0)
            condition = (df['entry_score'] >= score_min) & (df['entry_score'] <= score_max)
            df.loc[condition, 'max_allowed_votes'] = max_votes
        is_veto_tolerated = df['veto_votes'] <= df['max_allowed_votes']

        is_score_positive = df['entry_score'] > df['risk_score']
        not_avoid = df['dynamic_action'] != 'AVOID'
        is_not_risk_day = df['signal_type'] == '无信号'

        final_buy_condition = (
            is_foundation_solid &
            ~has_hard_veto &
            is_score_positive &
            is_veto_tolerated &
            not_avoid &
            is_not_risk_day
        )
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # 调用最终净化方法
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
        【V318.1 风控回归版】
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        has_critical_chip_risk = atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3

        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'veto_votes'] += 1
        
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            chip_risks_in_veto = {"RISK_CAPITAL_STRUCT_MAIN_FORCE_DISTRIBUTING", "CONTEXT_RECENT_DISTRIBUTION_PRESSURE"}
            veto_signals = [s for s in get_param_value(veto_params.get('veto_signals'), []) if s not in chip_risks_in_veto]
            
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

        risk_overrides_entry = df['risk_score'] > df['entry_score']
        is_in_ascent_phase = atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        df.loc[risk_overrides_entry & ~is_in_ascent_phase, 'veto_votes'] += 1
        
        is_opportunity_fading = atomic.get('SCORE_DYN_OPPORTUNITY_FADING', default_series)
        df.loc[is_opportunity_fading, 'veto_votes'] += 1
        
        is_risk_escalating = atomic.get('SCORE_DYN_RISK_ESCALATING', default_series)
        df.loc[is_risk_escalating, 'veto_votes'] += 1

    def _finalize_signals(self):
        """
        【V404.1 健壮性修复版】
        - 核心修复: 修复了当没有任何买卖信号时，'signal_entry'列不存在导致的AttributeError。
        """
        df = self.strategy.df_indicators
        
        # --- 代码修改开始 ---
        # [修改原因] 修复 AttributeError。确保 signal_entry 列总是存在，即使没有任何买入或卖出信号。
        # 在进行任何条件赋值之前，先用默认值 False 初始化该列。
        df['signal_entry'] = False
        # --- 代码修改结束 ---
        
        final_buy_condition = df['signal_type'] == '买入信号'
        final_sell_condition = df['signal_type'] == '卖出信号'
        final_warning_condition = df['signal_type'] == '风险预警'

        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        
        # 确保买入信号日不携带任何卖出或预警信息
        # 增加健壮性检查，防止因列不存在而报错。
        if 'exit_signal_code' in df.columns:
            # 确保 exit_severity_level 和 alert_reason 也存在
            if 'exit_severity_level' not in df.columns: df['exit_severity_level'] = 0
            if 'alert_reason' not in df.columns: df['alert_reason'] = ''
            df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        df.loc[final_sell_condition | final_warning_condition, 'final_score'] = df.loc[final_sell_condition | final_warning_condition, 'risk_score']
        
        print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        debug_cols = ['entry_score', 'risk_score', 'veto_votes', 'max_allowed_votes', 'final_score', 'signal_type', 'main_force_state']
        final_check_df = df[(df['signal_type'] != '无信号') & (df['signal_type'] != '中性')].tail(10)
        if not final_check_df.empty:
            cols_to_show = [col for col in debug_cols if col in final_check_df.columns]
            print("          -> [最终分数审查报告]:")
            print(final_check_df[cols_to_show])
        else:
            print("          -> [最终分数审查报告]: 在最近的记录中未发现任何有效信号。")
