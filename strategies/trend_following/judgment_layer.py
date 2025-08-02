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
        【V339.1 攻防动态版】持仓大脑 - 评估持仓健康度
        - 核心重构: 不再计算静态的健康分，而是动态对比当日与前一日的
                    “进攻分数构成”和“风险分数构成”，捕捉“买入逻辑”的
                    衰减和“风险”的出现，生成结构化的攻防变化摘要。
        """
        # print("        -> [持仓大脑 V339.1] 启动，正在进行攻防动态对比分析...")
        df = self.strategy.df_indicators
        
        # --- 1. 获取昨日的攻防态势 ---
        score_details_df_yesterday = score_details_df.shift(1).fillna(0)
        risk_score_df_yesterday = risk_score_df.shift(1).fillna(0)
        
        # --- 2. 初始化诊断摘要存储列 ---
        # 我们将把这个丰富的诊断结果，直接存储在df中，供后续模块使用
        df['health_change_summary'] = [{} for _ in range(len(df))]

        # --- 3. 逐日进行对比分析 ---
        for idx in df.index:
            # 进攻端分析
            today_offense = set(score_details_df.columns[score_details_df.loc[idx] > 0])
            yesterday_offense = set(score_details_df_yesterday.columns[score_details_df_yesterday.loc[idx] > 0])
            
            new_strengths = list(today_offense - yesterday_offense)
            fading_strengths = list(yesterday_offense - today_offense)

            # 风险端分析
            today_risks = set(risk_score_df.columns[risk_score_df.loc[idx] > 0])
            yesterday_risks = set(risk_score_df_yesterday.columns[risk_score_df_yesterday.loc[idx] > 0])
            
            new_risks = list(today_risks - yesterday_risks)
            resolved_risks = list(yesterday_risks - today_risks)
            
            # 只有在攻防态势发生变化时才记录
            if new_strengths or fading_strengths or new_risks or resolved_risks:
                summary = {
                    'offense_change': {
                        'new': new_strengths,
                        'fading': fading_strengths
                    },
                    'risk_change': {
                        'new': new_risks,
                        'resolved': resolved_risks
                    }
                }
                # 使用 .at 来精确赋值
                df.at[idx, 'health_change_summary'] = summary

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_score_df: pd.DataFrame):
        """
        【V319.0 终极决策修复版】
        - 核心修复: 实现了严格的、多层次的风险否决逻辑，确保高风险或
                    存在否决票的买入信号被正确地过滤掉。
        - 流程简化: 简化了信号生成逻辑，使其更清晰、更健壮。
        """
        # print("    --- [最高作战指挥部 V319.0 终极决策修复版] 启动... ---")
        df = self.strategy.df_indicators
        
        # --- 步骤 1: 初始化所有决策相关列 ---
        df['final_score'] = 0.0
        df['signal_type'] = '无信号'
        df['veto_votes'] = 0
        df['dynamic_action'] = 'HOLD'

        # --- 步骤 2: 评估持仓健康度 (独立模块) ---
        self._evaluate_holding_health(score_details_df, risk_score_df)
        
        # --- 步骤 3: 静态否决票评估 ---
        self._calculate_static_veto_votes()

        # --- 步骤 4: 动态力学矩阵裁决 ---
        df['dynamic_action'] = self._get_dynamic_combat_action()
        
        # --- 步骤 5: 【核心决策逻辑】形成最终买入条件 ---
        # 条件A: 基础风控，进攻分必须大于风险分
        is_score_positive = df['entry_score'] > df['risk_score']
        
        # 条件B: 否决票必须为0
        no_veto_votes = df['veto_votes'] == 0
        
        # 条件C: 动态力学指令不能是“规避”
        not_avoid = df['dynamic_action'] != 'AVOID'
        
        # 最终买入条件：必须同时满足以上所有条件
        final_buy_condition = is_score_positive & no_veto_votes & not_avoid

        # --- 步骤 6: 标记最终信号类型 ---
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # --- 步骤 7: 调用离场层，计算并标记卖出信号 ---
        # 这会填充 df['exit_signal_code'] 等列
        self.strategy.exit_layer.calculate_exit_signals()
        is_exit_signal = (df.get('exit_signal_code', 0) > 0)
        # 卖出信号的优先级高于“无信号”
        df.loc[is_exit_signal, 'signal_type'] = '卖出信号'
        
        # 标记“风险预警”信号
        is_warning_signal = (df.get('alert_level', 0) > 0)
        # 关键条件: 只在当前信号类型仍为“无信号”时，才将其标记为“风险预警”，以避免覆盖掉更高级别的“买入”或“卖出”信号。
        df.loc[is_warning_signal & (df['signal_type'] == '无信号'), 'signal_type'] = '风险预警'

        # --- 步骤 8: 最终净化与分数赋值 ---
        self._finalize_signals()

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
        
        # 1. 识别最终的信号类型
        final_buy_condition = df['signal_type'] == '买入信号'
        final_sell_condition = df['signal_type'] == '卖出信号'

        # 2. 为买入信号赋值并净化
        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        # 确保买入信号日不携带任何卖出或预警信息
        df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        # 3. 为卖出信号赋值
        df.loc[final_sell_condition, 'final_score'] = df.loc[final_sell_condition, 'risk_score'] # 卖出信号的最终分是风险分
        df.loc[final_sell_condition, 'signal_entry'] = False
        
        # 4. 打印最终审查报告
        print("        -> [决策单元] 决策完成。正在进行最终分数审查...")
        final_check_df = df[(df['signal_type'] != '无信号') & (df['signal_type'] != '中性')].tail(10)
        if not final_check_df.empty:
            print("          -> [最终分数审查报告]:")
            print(final_check_df[['entry_score', 'risk_score', 'veto_votes', 'final_score', 'signal_type', 'main_force_state']])
        else:
            print("          -> [最终分数审查报告]: 在最近的记录中未发现任何有效信号。")












