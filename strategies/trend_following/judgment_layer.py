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
        【V527.0 · 宙斯天平版】
        - 核心革命: 彻底废除旧的计分逻辑，恢复并强化“进攻分 - 风险惩罚分”的核心裁决机制。
        - 新核心公式: 最终得分 = (总进攻分 * 信心阻尼器) - 总风险惩罚分
        """
        df = self.strategy.df_indicators
        df['alert_level'], df['alert_reason'], fused_risks_df = self._adjudicate_risk_level()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        chimera_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index))
        confidence_damper = 1.0 - chimera_conflict_score
        
        # 调用复活的风险惩罚计算器，得到总风险惩罚分
        total_risk_penalty, _ = self._calculate_risk_penalty_score(risk_details_df)
        
        # 应用“宙斯天平”计分公式
        df['final_score'] = (df['entry_score'] * confidence_damper) - total_risk_penalty
        
        df['risk_score'] = self.strategy.atomic_states.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).fillna(0.0)
        
        p_trend_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        p_prophet_judge = get_params_block(self.strategy, 'prophet_oracle', {}).get('judgment_params', {})
        
        final_score_threshold = get_param_value(p_trend_judge.get('final_score_threshold'), 400)
        
        df['signal_type'] = '无信号'

        is_score_sufficient = df['final_score'] > final_score_threshold
        is_veto_by_alert = df['alert_level'] >= 3
        exit_triggers_df = self.strategy.exit_triggers
        is_hard_exit_veto = exit_triggers_df.any(axis=1)
        
        prophet_entry_threshold = get_param_value(p_prophet_judge.get('prophet_entry_threshold'), 0.6)
        prophet_score_multiplier = get_param_value(p_prophet_judge.get('prophet_score_multiplier'), 1000)
        
        predictive_opp_score = self.strategy.atomic_states.get('PREDICTIVE_OPP_CAPITULATION_REVERSAL', pd.Series(0.0, index=df.index))

        is_prophet_entry = (predictive_opp_score > prophet_entry_threshold)
        df.loc[is_prophet_entry, 'signal_type'] = '先知入场'
        
        df.loc[is_prophet_entry, 'final_score'] = (predictive_opp_score * prophet_score_multiplier).astype(int)

        is_not_prophet_day = ~is_prophet_entry
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index)) & is_not_prophet_day
        tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask & is_not_prophet_day
        is_effective_hard_exit = (strategic_exit_mask | tactical_exit_mask)

        df.loc[strategic_exit_mask, 'signal_type'] = '战略失效离场'
        df.loc[tactical_exit_mask, 'signal_type'] = '趋势破位离场'
        df.loc[is_effective_hard_exit, 'final_score'] = 0

        is_mundane_day = is_not_prophet_day & ~is_effective_hard_exit
        
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert & is_mundane_day
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        
        alert_veto_condition = is_score_sufficient & is_veto_by_alert & is_mundane_day
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0
        
        # 确保最终得分在任何情况下都转换为整数
        df['final_score'] = df['final_score'].fillna(0).astype(int)
        
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df, df['signal_type'])
        self._finalize_signals()

    def _calculate_risk_penalty_score(self, risk_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V512.0 · 宙斯天平版】风险惩罚分数计算器
        - 核心革命: “复活”此方法，使其成为最终得分计算的关键一环。
        - 新逻辑: 遍历所有风险信号，根据其原始分(0-1)和在字典中定义的'penalty_weight'，计算出总的风险惩罚分。
        """
        # 移除废除声明，正式启用此方法
        df_index = self.strategy.df_indicators.index
        if risk_details_df.empty:
            return pd.Series(0.0, index=df_index), pd.DataFrame(index=df_index)
        
        total_penalty_score = pd.Series(0.0, index=df_index)
        
        # 遍历 risk_details_df 的每一列（即每个风险信号）
        for risk_name in risk_details_df.columns:
            # 从配置中查找该风险的惩罚权重
            signal_meta = self.risk_metadata.get(risk_name, {})
            penalty_weight = signal_meta.get('penalty_weight', 0.0)
            
            # 如果有权重，则计算惩罚分并累加
            if penalty_weight > 0:
                risk_series = risk_details_df[risk_name].fillna(0.0)
                penalty_score = risk_series * penalty_weight
                total_penalty_score += penalty_score
                
        # 返回计算出的总惩罚分，第二个返回值保持为空DataFrame以兼容旧接口
        return total_penalty_score.reindex(df_index).fillna(0.0), pd.DataFrame(index=df_index)

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, signal_type_series: pd.Series) -> pd.Series:
        """
        【V3.7 · 赏罚分明版】生成人类可读的信号摘要。
        - 核心革命: 修正了进攻项的筛选逻辑，从 != 0 升级为 > 0。
        - 收益: 确保只有得分大于零的进攻项才会被记录和显示，彻底解决了负分项被错误归类为“激活项”的问题。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        def process_details_df(details_df, is_risk_df=False):
            if details_df.empty: return pd.Series(dtype=object)
            long_df = details_df.melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            
            if not is_risk_df:
                # 将筛选条件从 != 0 修正为 > 0，确保只记录真正的“进攻项”。
                long_df = long_df[long_df['score'].fillna(0).astype(int) > 0].copy()
            else:
                # 风险信号的逻辑保持不变，任何大于0的风险都应被记录
                long_df = long_df[(long_df['score'].fillna(0) * 1000).astype(int) > 0].copy()

            if long_df.empty: return pd.Series(dtype=object)
            date_col_name = long_df.columns[0]
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items() if isinstance(v, dict)}
            long_df['cn_name'] = long_df['signal'].map(cn_name_map).fillna(long_df['signal'])
            if is_risk_df:
                long_df['summary_dict'] = long_df.apply(lambda row: {'name': row['cn_name'], 'score': int(row['score'] * 1000.0)}, axis=1)
            else:
                long_df['summary_dict'] = long_df.apply(lambda row: {'name': row['cn_name'], 'score': int(row['score'])}, axis=1)
            return long_df.groupby(date_col_name)['summary_dict'].apply(list)

        offense_summaries = process_details_df(score_details_df, is_risk_df=False)
        risk_summaries = process_details_df(risk_details_df, is_risk_df=True)

        summary_df = pd.DataFrame({'offense': offense_summaries, 'risk': risk_summaries}).reindex(self.strategy.df_indicators.index)
        
        def generate_final_summary(row):
            final_signal_type = signal_type_series.get(row.name)
            if final_signal_type in ['买入信号', '先知入场']:
                offense_list = row['offense'] if isinstance(row['offense'], list) else []
            else:
                offense_list = []
            risk_list = row['risk'] if isinstance(row['risk'], list) else []
            return {'offense': offense_list, 'risk': risk_list}

        return summary_df.apply(generate_final_summary, axis=1)

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
        【V522.0 · 统一号令版】
        - 核心革命: 重建指挥链。signal_entry 成为所有入场信号的唯一官方旗帜。
        - 核心逻辑: 无论是“买入信号”还是“先知入场”，都会将 signal_entry 设置为 True。
        - 收益: 为下游的 simulation_layer 提供了单一、明确的建仓指令。
        """
        df = self.strategy.df_indicators
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        
        # 统一号令：任何一种入场信号，都必须升起'signal_entry'旗帜。
        final_buy_condition = (df['signal_type'] == '买入信号') | (df['signal_type'] == '先知入场')
        
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, 'exit_signal_code'] = 0

    def _adjudicate_risk_level(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V2.5 · 雅典娜的精准版】风险裁决者 (Risk Adjudicator)
        - 核心升级: 采纳指挥官的精准洞察，将“先知离场”的上下文过滤器从宽泛的 EMA55 收紧为严格的 EMA5。
        - 核心逻辑: 仅当股价处于短期强势（收盘价 > EMA5）时，才允许“高潮衰竭”风险触发。
                      这彻底解决了在“下跌中继”状态下，因成交量放大而错误触发顶部风险的致命缺陷。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        risk_categories = {
            'ARCHANGEL_RISK': ['SCORE_ARCHANGEL_TOP_REVERSAL'],
            'TOP_REVERSAL': ['SCORE_BEHAVIOR_TOP_REVERSAL', 'SCORE_CHIP_TOP_REVERSAL', 'SCORE_FF_TOP_REVERSAL', 'SCORE_STRUCTURE_TOP_REVERSAL', 'SCORE_DYN_TOP_REVERSAL', 'SCORE_FOUNDATION_TOP_REVERSAL'],
            'BEARISH_RESONANCE': ['SCORE_BEHAVIOR_BEARISH_RESONANCE', 'SCORE_CHIP_BEARISH_RESONANCE', 'SCORE_FF_BEARISH_RESONANCE', 'SCORE_STRUCTURE_BEARISH_RESONANCE', 'SCORE_DYN_BEARISH_RESONANCE', 'SCORE_FOUNDATION_BEARISH_RESONANCE'],
            'MICRO_RISK': ['COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL', 'COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING'],
            'EUPHORIA_RISK': ['COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION']
        }
        fused_risks = {}
        for category, signals in risk_categories.items():
            signal_scores = [atomic.get(s, pd.Series(0.0, index=df.index)).reindex(df.index).fillna(0.0) for s in signals]
            fused_risks[category] = np.maximum.reduce(signal_scores)
        fused_risks_df = pd.DataFrame(fused_risks, index=df.index)
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        p_alerts = p_judge.get('alert_level_thresholds', {})
        predictive_exhaustion_risk = atomic.get('PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION', pd.Series(0, index=df.index))
        prophet_threshold = get_param_value(p_judge.get('prophet_alert_threshold'), 0.7)
        level_3_archangel_threshold = get_param_value(p_alerts.get('level_3_archangel_threshold'), 0.7)
        level_3_threshold = get_param_value(p_alerts.get('level_3_top_reversal'), 0.8)
        level_2_resonance_threshold = get_param_value(p_alerts.get('level_2_bearish_resonance'), 0.7)
        level_2_euphoria_threshold = get_param_value(p_alerts.get('level_2_euphoria_risk'), 0.75)
        level_1_threshold = get_param_value(p_alerts.get('level_1_micro_risk'), 0.6)
        
        # 将上下文过滤器从 EMA_55_D 升级为更精准的 EMA_5_D
        is_uptrend_context = df.get('close_D', 0) > df.get('EMA_5_D', 0)
        
        conditions = [
            (predictive_exhaustion_risk > prophet_threshold) & is_uptrend_context,
            fused_risks_df['ARCHANGEL_RISK'] > level_3_archangel_threshold,
            fused_risks_df['TOP_REVERSAL'] > level_3_threshold,
            (fused_risks_df['BEARISH_RESONANCE'] > level_2_resonance_threshold) | (fused_risks_df['EUPHORIA_RISK'] > level_2_euphoria_threshold),
            fused_risks_df['MICRO_RISK'] > level_1_threshold,
        ]
        choices_level = [3, 3, 3, 2, 1]
        choices_reason = [
            '红色警报: 先知-高潮衰竭',
            '红色警报: 天使长-明确顶部形态',
            '红色警报: 顶部反转风险', 
            '橙色警报: 共振或亢奋风险', 
            '黄色警报: 微观结构风险'
        ]
        alert_level = pd.Series(np.select(conditions, choices_level, default=0), index=df.index)
        alert_reason = pd.Series(np.select(conditions, choices_reason, default=''), index=df.index)
        self.strategy.atomic_states['ALERT_LEVEL'] = alert_level.astype(np.int8)
        self.strategy.atomic_states['ALERT_REASON'] = alert_reason
        return alert_level, alert_reason, fused_risks_df

















