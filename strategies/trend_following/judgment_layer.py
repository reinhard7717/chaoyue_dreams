# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层
import pandas as pd
import numpy as np
from typing import Tuple
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V515.0 · 审判日版】
        - 核心升级: 1. 新增并调用 _adjudicate_risk_level 方法，将所有风险信号整合成统一的、分级的 ALERT_LEVEL。
                      2. 决策逻辑现在直接消费 ALERT_LEVEL，当警报等级过高时，一票否决买入信号。
        - 收益: 建立了清晰、果断、等级分明的风险指挥中心，实现了从“感知风险”到“应对风险”的飞跃。
        """
        print("    --- [最高作战指挥部 V515.0 · 审判日版] 启动...")
        df = self.strategy.df_indicators
        
        # 调用新的风险裁决者，并将结果存入df
        df['alert_level'], df['alert_reason'] = self._adjudicate_risk_level()
        
        df['risk_penalty_score'], penalty_components_df = self._calculate_risk_penalty_score(risk_details_df)
        reportable_risk_df = risk_details_df.copy()
        if not penalty_components_df.empty:
            reportable_risk_df.update(penalty_components_df)

        df['dynamic_action'] = self._get_dynamic_combat_action()
        
        chimera_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index))
        confidence_damper = 1.0 - chimera_conflict_score
        df['final_score'] = (df['entry_score'] * confidence_damper) - df['risk_penalty_score']
        
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 400)
        
        df['signal_type'] = '无信号'
        
        is_score_sufficient = df['final_score'] > final_score_threshold
        
        # 使用 alert_level >= 3 作为最高否决条件
        is_veto_by_alert = df['alert_level'] >= 3
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        
        exit_triggers_df = self.strategy.exit_triggers
        is_hard_exit_veto = exit_triggers_df.any(axis=1)
        
        if is_hard_exit_veto.any():
            strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
            tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
            df.loc[strategic_exit_mask, 'signal_type'] = '战略失效离场'
            df.loc[tactical_exit_mask, 'signal_type'] = '趋势破位离场'
            df.loc[is_hard_exit_veto, 'final_score'] = 0
            
        # 将原有的 dynamic_veto 替换为新的 alert_veto
        alert_veto_condition = is_score_sufficient & is_veto_by_alert & ~is_hard_exit_veto
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0

        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, reportable_risk_df)
        self._finalize_signals()

    def _calculate_risk_penalty_score(self, risk_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V511.1 · 重复惩罚修复版】计算总风险惩罚分
        - 核心修复:
          - [BUG修复] 彻底修复了“亢奋加速风险”被重复计算两次的致命BUG。
          - [逻辑优化] 简化了计算流程，现在统一在循环中处理所有风险信号，
                        不再有特殊的外部处理逻辑，确保每个风险信号只被惩罚一次。
        - 收益: 恢复了风险惩罚体系的平衡，避免了单一风险项权重被不合理放大的问题。
        """
        df = self.strategy.df_indicators
        
        if risk_details_df.empty:
            return pd.Series(0.0, index=df.index), pd.DataFrame(index=df.index)

        total_penalty_score = pd.Series(0.0, index=df.index)
        penalty_components_df = pd.DataFrame(index=df.index)
        
        score_map = get_params_block(self.strategy, 'score_type_map', {})

        # 遍历 risk_details_df 中的每一列（即每一个风险信号）
        for signal_name in risk_details_df.columns:
            # 确保信号在配置中定义，并且有 penalty_weight
            if signal_name in score_map and 'penalty_weight' in score_map[signal_name]:
                raw_risk_score = risk_details_df[signal_name]
                penalty_weight = score_map[signal_name]['penalty_weight']
                
                # 计算该信号贡献的惩罚分
                penalty_amount = raw_risk_score * penalty_weight
                
                # 累加到总惩罚分中
                total_penalty_score += penalty_amount
                
                # 记录每个分量的惩罚，用于报告
                penalty_components_df[signal_name] = penalty_amount
        
        # 删除了对亢奋风险的重复计算逻辑，因为它已经在上面的循环中被统一处理了。
        
        return total_penalty_score, penalty_components_df

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> pd.Series:
        """
        【V2.9 · 数据结构修复版】生成人类可读的信号摘要。
        - 核心修复: 不再返回预格式化的字符串列表，而是返回一个字典列表，
                    每个字典包含 'name' 和 'score'，以供下游程序化处理。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        def process_details_df(details_df):
            if details_df.empty:
                return pd.Series(dtype=object)
            
            long_df = details_df.melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            long_df = long_df[long_df['score'] > 0].copy()
            if long_df.empty:
                return pd.Series(dtype=object)

            date_col_name = long_df.columns[0]
            
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items() if isinstance(v, dict)}
            long_df['cn_name'] = long_df['signal'].map(cn_name_map).fillna(long_df['signal'])
            
            # 不再创建预格式化的字符串，而是创建一个字典
            long_df['summary_dict'] = long_df.apply(
                lambda row: {'name': row['cn_name'], 'score': int(row['score'])},
                axis=1
            )
            
            # 返回字典的列表，而不是字符串的列表
            return long_df.groupby(date_col_name)['summary_dict'].apply(list)

        offense_summaries = process_details_df(score_details_df)
        risk_summaries = process_details_df(risk_details_df)

        summary_df = pd.DataFrame({'offense': offense_summaries, 'risk': risk_summaries}).reindex(self.strategy.df_indicators.index)
        
        return summary_df.apply(
            lambda row: {'offense': row['offense'] if isinstance(row['offense'], list) else [], 'risk': row['risk'] if isinstance(row['risk'], list) else []},
            axis=1
        )

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V319.0 · 终极信号适配版】动态力学战术矩阵
        - 核心重构: 全面消费由 DynamicMechanicsEngine 生成的终极信号。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        offensive_resonance_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE_S_PLUS', default_score)
        risk_expansion_score = atomic.get('SCORE_DYN_BEARISH_RESONANCE_S_PLUS', default_score)
        
        is_force_attack = offensive_resonance_score > 0.6
        is_avoid = risk_expansion_score > 0.6
        is_caution = (offensive_resonance_score > 0.4) & (risk_expansion_score > 0.4)
        
        actions = pd.Series('HOLD', index=df.index)
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        return actions

    def _finalize_signals(self):
        """
        【V404.3 清理与对齐版】
        """
        df = self.strategy.df_indicators
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        
        final_buy_condition = df['signal_type'] == '买入信号'
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, 'exit_signal_code'] = 0

    def _adjudicate_risk_level(self) -> Tuple[pd.Series, pd.Series]:
        """
        【V1.0 · 新增】风险裁决者 (Risk Adjudicator)
        - 核心职责: 将所有零散的风险信号，整合成统一的、分级的、可直接执行的战术警报等级。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        # 1. 定义风险审判庭及其成员
        risk_categories = {
            'TOP_REVERSAL': [
                'SCORE_BEHAVIOR_TOP_REVERSAL', 'SCORE_CHIP_TOP_REVERSAL', 'SCORE_FF_TOP_REVERSAL',
                'SCORE_STRUCTURE_TOP_REVERSAL', 'SCORE_DYN_TOP_REVERSAL', 'SCORE_FOUNDATION_TOP_REVERSAL'
            ],
            'BEARISH_RESONANCE': [
                'SCORE_BEHAVIOR_BEARISH_RESONANCE', 'SCORE_CHIP_BEARISH_RESONANCE', 'SCORE_FF_BEARISH_RESONANCE',
                'SCORE_STRUCTURE_BEARISH_RESONANCE', 'SCORE_DYN_BEARISH_RESONANCE', 'SCORE_FOUNDATION_BEARISH_RESONANCE'
            ],
            'MICRO_RISK': [
                'COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL', 'COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'
            ]
        }
        
        # 2. 计算每个审判庭的风险强度 (取最大值)
        fused_risks = {}
        for category, signals in risk_categories.items():
            signal_scores = [atomic.get(s, pd.Series(0.0, index=df.index)) for s in signals]
            fused_risks[category] = np.maximum.reduce(signal_scores)

        # 3. 定义警报等级的触发阈值
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        p_alerts = p_judge.get('alert_level_thresholds', {})
        
        level_3_threshold = get_param_value(p_alerts.get('level_3_top_reversal'), 0.8)
        level_2_threshold = get_param_value(p_alerts.get('level_2_bearish_resonance'), 0.7)
        level_1_threshold = get_param_value(p_alerts.get('level_1_micro_risk'), 0.6)

        # 4. 根据阈值进行最终审判
        conditions = [
            fused_risks['TOP_REVERSAL'] > level_3_threshold,
            fused_risks['BEARISH_RESONANCE'] > level_2_threshold,
            fused_risks['MICRO_RISK'] > level_1_threshold,
        ]
        
        choices_level = [3, 2, 1]
        choices_reason = ['红色警报: 顶部反转风险', '橙色警报: 看跌共振风险', '黄色警报: 微观结构风险']
        
        alert_level = pd.Series(np.select(conditions, choices_level, default=0), index=df.index)
        alert_reason = pd.Series(np.select(conditions, choices_reason, default=''), index=df.index)
        
        # 将裁决结果存入原子状态，供全局使用
        self.strategy.atomic_states['ALERT_LEVEL'] = alert_level.astype(np.int8)
        self.strategy.atomic_states['ALERT_REASON'] = alert_reason
        
        return alert_level, alert_reason

















