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
        【V400.0 ORM适配版】
        - 核心重构: 决策逻辑现在完全基于由 WarningLayer 提供的、已合并了所有
                    致命风险和常规风险的最终 risk_score。
        - 流程简化: 不再需要单独调用 ExitLayer。而是直接使用配置文件中的
                    exit_threshold_params 和 warning_threshold_params 来对
                    最终的 risk_score 进行分级，从而决定信号是“卖出”还是“预警”。
        """
        # print("    --- [最高作战指挥部 V400.0 ORM适配版] 启动... ---")
        df = self.strategy.df_indicators
        
        # --- 步骤 1: 初始化所有决策相关列 ---
        df['final_score'] = 0.0
        df['signal_type'] = '无信号'
        df['veto_votes'] = 0
        df['dynamic_action'] = 'HOLD'

        # --- 步骤 2: 评估持仓健康度 (现在由 WarningLayer 内部完成) ---
        self._evaluate_holding_health(score_details_df, risk_details_df)

        # --- 步骤 3 & 4: 否决票与动态力学 (逻辑不变) ---
        self._calculate_static_veto_votes()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        
        # --- 步骤 5: 【核心决策逻辑】形成最终买入条件 (逻辑不变) ---
        is_score_positive = df['entry_score'] > df['risk_score']
        # 此处可以加入您之前实现的弹性否决票逻辑
        no_veto_votes = df['veto_votes'] == 0 # 简化示例，可替换
        not_avoid = df['dynamic_action'] != 'AVOID'
        final_buy_condition = is_score_positive & no_veto_votes & not_avoid

        # --- 步骤 6: 标记初步信号类型 ---
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # --- 步骤 7: 【核心改造】基于总风险分，统一进行卖出和预警判断 ---
        # 不再调用 self.strategy.exit_layer.calculate_exit_signals()
        
        # 从配置加载卖出和预警的阈值
        exit_params = self.strategy.unified_config.get('exit_strategy_params', {})
        exit_thresholds = exit_params.get('exit_threshold_params', {})
        warning_thresholds = exit_params.get('warning_threshold_params', {})
        
        # 按阈值从高到低排序，确保优先判断卖出信号
        all_thresholds = []
        for level_info in exit_thresholds.values():
            all_thresholds.append({'level': level_info['level'], 'type': '卖出信号'})
        for level_info in warning_thresholds.values():
            all_thresholds.append({'level': level_info['level'], 'type': '风险预警'})
        
        sorted_thresholds = sorted(all_thresholds, key=lambda x: x['level'], reverse=True)

        # 遍历阈值，对 risk_score 进行分级
        for rule in sorted_thresholds:
            threshold = rule['level']
            signal_type = rule['type']
            
            # 条件：风险分达到阈值，且当前没有被更高优先级的信号标记
            # （买入信号优先级最高，在循环外已标记）
            condition = (df['risk_score'] >= threshold) & (df['signal_type'] == '无信号')
            df.loc[condition, 'signal_type'] = signal_type

        # --- 步骤 8: 最终净化与分数赋值 (逻辑不变) ---
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
        final_warning_condition = df['signal_type'] == '风险预警'

        # 2. 为买入信号赋值并净化
        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        # 确保买入信号日不携带任何卖出或预警信息
        df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        # 3. 为卖出及预警信号赋值
        df.loc[final_sell_condition | final_warning_condition, 'final_score'] = df.loc[final_sell_condition | final_warning_condition, 'risk_score']
        df.loc[final_sell_condition, 'signal_entry'] = False
        
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












