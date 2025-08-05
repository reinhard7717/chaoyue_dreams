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

    def _generate_exit_triggers(self) -> pd.DataFrame:
        """
        【V504.0 新增】离场触发器生成器
        - 核心职责: 根据“三道防线”原则，生成一个包含所有离场原因的布尔型DataFrame。
        """
        # print("        -> [离场触发器 V504.0] 启动，正在检查三道防线...")
        df = self.strategy.df_indicators
        triggers_df = pd.DataFrame(index=df.index)
        
        # --- 防线一: 致命一击 (Critical Hit) ---
        # 检查是否存在任何一个致命风险信号。critical_risk_details 是由 WarningLayer 计算并传入的。
        critical_risk_details = self.strategy.critical_risk_details
        triggers_df['EXIT_CRITICAL_HIT'] = critical_risk_details.sum(axis=1) > 0
        if triggers_df['EXIT_CRITICAL_HIT'].any():
            print(f"          -> [防线一] 侦测到 {triggers_df['EXIT_CRITICAL_HIT'].sum()} 天存在“致命一击”风险。")

        # --- 防线二: 风险溢出 (Risk Overflow) ---
        # 从新的 judgment_params 读取配置
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        overflow_threshold = get_param_value(p_judge.get('risk_overflow_threshold'), 1000)
        triggers_df['EXIT_RISK_OVERFLOW'] = self.strategy.risk_score > overflow_threshold
        if triggers_df['EXIT_RISK_OVERFLOW'].any():
            print(f"          -> [防线二] 侦测到 {triggers_df['EXIT_RISK_OVERFLOW'].sum()} 天总风险分超过阈值 {overflow_threshold}。")

        # --- 防线三: 利润保护 (Profit Protector) ---
        p_protector = p_judge.get('profit_protector', {})
        if get_param_value(p_protector.get('enabled'), False):
            max_drawdown_pct = get_param_value(p_protector.get('max_drawdown_pct'), 0.15)
            # 注意：此处的实现是简化的，完整的利润保护需要与持仓跟踪模块联动。
            # 在当前的回测框架下，我们暂时将其设置为False，但保留了逻辑框架。
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)
            print(f"          -> [防线三] 利润保护器已启用 (最大回撤 {max_drawdown_pct*100}%)。注意：完整功能需与持仓状态联动。")
        else:
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)

        return triggers_df

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V504.0 三道防线集成版】
        - 核心修改: 集成了新的“三道防线”离场逻辑，并确保其与现有买入逻辑正确衔接。
        """
        # print("    --- [最高作战指挥部 V504.0 三道防线集成版] 启动... ---")
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
        
        # --- 代码修改开始 ---
        # [修改原因] V504.0 架构升级：用新的“三道防线”逻辑替换旧的离场阈值逻辑。
        # 1. 调用新的离场触发器生成器
        exit_triggers = self._generate_exit_triggers()
        is_sell_signal = exit_triggers.any(axis=1)
        
        # 2. 将触发卖出的具体原因保存到 strategy 对象中，用于后续分析
        self.strategy.exit_triggers = exit_triggers[is_sell_signal]
        
        # 3. 根据触发结果设置信号类型
        df.loc[is_sell_signal, 'signal_type'] = '卖出信号'
        # --- 代码修改结束 ---

        # (旧的 exit_threshold_params 和 warning_threshold_params 逻辑已被上面的新逻辑取代)

        # --- 买入逻辑 (基本保持不变，但增加了卖出信号的否决) ---
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
        
        # --- 代码修改开始 ---
        # [修改原因] V504.0 架构升级：确保卖出信号优先，当天有卖出信号则不能买入。
        is_not_sell_day = ~is_sell_signal
        # --- 代码修改结束 ---

        final_buy_condition = (
            is_foundation_solid &
            ~has_hard_veto &
            is_score_positive &
            is_veto_tolerated &
            not_avoid &
            is_not_sell_day # [修改] 使用新的条件替换旧的 is_not_risk_day
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
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0
        df['alert_reason'] = ''
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
        
        # print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        debug_cols = ['entry_score', 'risk_score', 'veto_votes', 'max_allowed_votes', 'final_score', 'signal_type', 'main_force_state']
        final_check_df = df[(df['signal_type'] != '无信号') & (df['signal_type'] != '中性')].tail(10)
        if not final_check_df.empty:
            cols_to_show = [col for col in debug_cols if col in final_check_df.columns]
            print("          -> [最终分数审查报告]:")
            print(final_check_df[cols_to_show])
        else:
            print("          -> [最终分数审查报告]: 在最近的记录中未发现任何有效信号。")
