# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V317.0 - 动态力学版)
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
        - 核心重构: 此方法不再独立计算，而是成为一个“报告总汇”。
                    它从主DataFrame中读取由进攻层和预警层预先计算好的
                    “进攻动能摘要”和“风险变化摘要”，并将它们合并成一个
                    结构清晰的、最终的 health_change_summary。
        """
        # print("        -> [健康报告总汇 V400.0] 启动，正在整合攻防报告...")
        df = self.strategy.df_indicators
        
        # --- 1. 初始化最终的健康报告列 ---
        df['health_change_summary'] = [{} for _ in range(len(df))]

        # --- 2. 读取预计算好的摘要 ---
        # offensive_momentum_summary 由 OffensiveLayer 计算
        offensive_summary = df.get('offensive_momentum_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        # risk_change_summary 由 WarningLayer 计算
        risk_summary = df.get('risk_change_summary', pd.Series([{} for _ in range(len(df))], index=df.index))

        # --- 3. 逐日合并成最终报告 ---
        for idx in df.index:
            final_summary = {}
            
            # 获取当天的进攻动能报告
            offense_report = offensive_summary.at[idx]
            if offense_report and isinstance(offense_report, dict) and any(offense_report.values()):
                final_summary['offense_momentum'] = offense_report
            
            # 获取当天的风险变化报告
            risk_report = risk_summary.at[idx]
            if risk_report and isinstance(risk_report, dict) and any(v for v in risk_report.values() if v):
                final_summary['risk_change'] = risk_report

            # 只有在有内容时才赋值
            if final_summary:
                df.at[idx, 'health_change_summary'] = final_summary

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V404.0 A股实战版】
        - 核心思想: 地基优先，底线思维。买入必须基于稳固的筹码和结构基础。
        - 新流程:
          1. 优先标记所有“卖出”和“预警”信号，确保风险优先。
          2. 计算“阵地分”，这是买入资格的“准入门槛”。
          3. 引入“硬性否决”机制，对S级风险零容忍。
          4. 在满足“地基稳固”和“无硬性否决”的前提下，再综合评估总分和弹性否决票。
        """
        print("    --- [最高作战指挥部 V404.0 A股实战版] 启动... ---")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        # --- 步骤 1: 初始化 ---
        df['final_score'] = 0.0
        df['signal_type'] = '无信号'
        df['veto_votes'] = 0
        df['dynamic_action'] = 'HOLD'
        df['max_allowed_votes'] = 0

        # --- 步骤 2: 评估健康度与计算否决票 (逻辑不变) ---
        self._evaluate_holding_health(score_details_df, risk_details_df)
        self._calculate_static_veto_votes()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        
        # --- 步骤 3: 【风险优先】标记卖出和预警信号 (逻辑不变) ---
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

        # --- 步骤 4: 【地基审查】计算“阵地分”并设定准入门槛 ---
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {}).keys()
        valid_pos_cols = [col for col in positional_rules if col in score_details_df.columns]
        positional_score = score_details_df[valid_pos_cols].sum(axis=1) if valid_pos_cols else pd.Series(0.0, index=df.index)

        buy_params = get_params_block(self.strategy, 'dynamic_mechanics_params').get('tactical_matrix_rules', {})
        min_positional_score = get_param_value(buy_params.get('min_positional_score'), 200) # 地基分门槛
        is_foundation_solid = positional_score >= min_positional_score

        # --- 步骤 5: 【硬性否决】识别不可容忍的S级风险 ---
        hard_veto_signals = get_param_value(buy_params.get('hard_veto_signals'), [])
        has_hard_veto = pd.Series(False, index=df.index)
        for signal in hard_veto_signals:
            has_hard_veto |= atomic.get(signal, default_series)

        # --- 步骤 6: 【弹性否决】应用基于总分的弹性否决票规则 (逻辑不变) ---
        tolerance_tiers = get_param_value(buy_params.get('tolerance_tiers'), [])
        for tier in sorted(tolerance_tiers, key=lambda x: x.get('score_min', 0)):
            score_min = tier.get('score_min', 0)
            score_max = tier.get('score_max', float('inf'))
            max_votes = tier.get('max_veto_votes', 0)
            condition = (df['entry_score'] >= score_min) & (df['entry_score'] <= score_max)
            df.loc[condition, 'max_allowed_votes'] = max_votes
        is_veto_tolerated = df['veto_votes'] <= df['max_allowed_votes']

        # --- 步骤 7: 【最终决策】整合所有条件形成最终买入信号 ---
        is_score_positive = df['entry_score'] > df['risk_score']
        not_avoid = df['dynamic_action'] != 'AVOID'
        is_not_risk_day = df['signal_type'] == '无信号'

        final_buy_condition = (
            is_foundation_solid &      # 1. 地基必须稳固 (硬性要求)
            ~has_hard_veto &           # 2. 没有S级硬性否决风险 (硬性要求)
            is_score_positive &        # 3. 进攻分 > 风险分
            is_veto_tolerated &        # 4. 常规否决票在容忍范围内
            not_avoid &                # 5. 动态力学非“规避”
            is_not_risk_day            # 6. 当天未被标记为卖出/预警
        )

        # --- 步骤 8: 标记买入信号 ---
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V317.0 核心】动态力学战术矩阵
        根据进攻和风险的“双向加速度”状态，返回四种战术指令之一：
        - FORCE_ATTACK (强攻): 进攻加速，风险减速。这是最佳战机。
        - PROCEED_WITH_CAUTION (暂缓/谨慎前行): 双向加速或双向减速。
        - AVOID (规避): 进攻减速，风险加速。这是最危险的陷阱。
        - HOLD (死守): 默认状态。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        offense_accel = atomic.get('FORCE_VECTOR_OFFENSE_ACCELERATING', default_series)
        offense_decel = atomic.get('FORCE_VECTOR_OFFENSE_DECELERATING', default_series)
        risk_accel = atomic.get('FORCE_VECTOR_RISK_ACCELERATING', default_series)
        risk_decel = atomic.get('FORCE_VECTOR_RISK_DECELERATING', default_series)

        # 1. 黄金象限 (Golden Quadrant): 进攻加速，风险减速
        is_force_attack = offense_accel & risk_decel
        
        # 2. 死亡象限 (Death Quadrant): 进攻减速，风险加速
        is_avoid = offense_decel & risk_accel
        
        # 3. 缠斗象限 (Contested Quadrant): 其他情况
        is_caution = (offense_accel & risk_accel) | (offense_decel & risk_decel)

        # 生成最终行动指令序列
        actions = pd.Series('HOLD', index=df.index)
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        
        # if is_force_attack.any(): print(f"          -> [战术矩阵] 在 {is_force_attack.sum()} 天内发出“强攻”指令！")
        # if is_avoid.any(): print(f"          -> [战术矩阵] 在 {is_avoid.sum()} 天内发出“规避”指令！")
        
        return actions

    def _calculate_static_veto_votes(self):
        """
        【V318.1 风控回归版】
        - 核心修复: 将 `risk_score > entry_score` 这个基础风控重新纳入否决票体系。
                    它现在作为常规风险，与其他风险因素一起参与投票。
        """
        # print("        -> [联席会议] 正在进行静态否决票评估...")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 1. 筹码地基风险 (3票)
        has_critical_chip_risk = atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3

        # 2. 主力行为风险 (1票)
        is_in_distribution_phase = df['main_force_state'].isin([MainForceState.DISTRIBUTING.value, MainForceState.COLLAPSE.value])
        df.loc[is_in_distribution_phase, 'veto_votes'] += 1
        
        # 3. 绝对否决权风险 (2票)
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

        # 4. 【核心修复】常规风险 (每项1票)
        # 4.1 风险/收益倒挂
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        is_in_ascent_phase = atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        df.loc[risk_overrides_entry & ~is_in_ascent_phase, 'veto_votes'] += 1
        
        # 4.2 机会衰退
        is_opportunity_fading = atomic.get('SCORE_DYN_OPPORTUNITY_FADING', default_series)
        df.loc[is_opportunity_fading, 'veto_votes'] += 1
        
        # 4.3 风险抬头
        is_risk_escalating = atomic.get('SCORE_DYN_RISK_ESCALATING', default_series)
        df.loc[is_risk_escalating, 'veto_votes'] += 1

    def _finalize_signals(self):
        """
        【V319.0 终极决策修复版 - 净化模块】
        - 职责: 在最终决策完成后，为信号赋予最终分数并进行净化。
        """
        df = self.strategy.df_indicators
        df['signal_entry'] = False
        
        # 1. 识别最终的信号类型
        final_buy_condition = df['signal_type'] == '买入信号'
        final_sell_condition = df['signal_type'] == '卖出信号'
        final_warning_condition = df['signal_type'] == '风险预警'

        # 2. 为买入信号赋值并净化
        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        # 确保买入信号日不携带任何卖出或预警信息
        if 'exit_signal_code' in df.columns:
            df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        # 3. 为卖出及预警信号赋值
        df.loc[final_sell_condition | final_warning_condition, 'final_score'] = df.loc[final_sell_condition | final_warning_condition, 'risk_score']
        
        # 4. 打印最终审查报告
        print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        # 增加 'max_allowed_votes' 列到调试输出，方便检查否决票逻辑
        debug_cols = ['entry_score', 'risk_score', 'veto_votes', 'max_allowed_votes', 'final_score', 'signal_type', 'main_force_state']
        final_check_df = df[(df['signal_type'] != '无信号') & (df['signal_type'] != '中性')].tail(10)
        if not final_check_df.empty:
            # 确保所有调试列都存在
            cols_to_show = [col for col in debug_cols if col in final_check_df.columns]
            print("          -> [最终分数审查报告]:")
            print(final_check_df[cols_to_show])
        else:
            print("          -> [最终分数审查报告]: 在最近的记录中未发现任何有效信号。")












