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
        【V522.0 · 最后的敕令版】
        - 核心升级: 在调用史官(_get_human_readable_summary)时，授予其“阅读圣旨”(df['signal_type'])的权力。
        """
        print("    --- [最高作战指挥部 V522.0 · 最后的敕令版] 启动...")
        df = self.strategy.df_indicators
        df['alert_level'], df['alert_reason'], fused_risks_df = self._adjudicate_risk_level()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        chimera_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index))
        confidence_damper = 1.0 - chimera_conflict_score
        df['final_score'] = (df['entry_score'] * confidence_damper)
        df['risk_score'] = self.strategy.atomic_states.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).fillna(0.0)
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 400)
        
        df['signal_type'] = '无信号'

        # 步骤一：硬性离场信号拥有最高否决权
        exit_triggers_df = self.strategy.exit_triggers
        is_hard_exit_veto = exit_triggers_df.any(axis=1)
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
        tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
        
        df.loc[strategic_exit_mask, 'signal_type'] = '战略失效离场'
        df.loc[tactical_exit_mask, 'signal_type'] = '趋势破位离场'
        df.loc[is_hard_exit_veto, 'final_score'] = 0

        # 步骤二：在没有硬性离场的前提下，进行进攻决策
        is_not_hard_exit = ~is_hard_exit_veto
        is_score_sufficient = df['final_score'] > final_score_threshold
        is_veto_by_alert = df['alert_level'] >= 3
        
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert & is_not_hard_exit
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        
        prophet_entry_threshold = get_param_value(p_judge.get('prophet_entry_threshold'), 0.04)
        predictive_opp_score = self.strategy.atomic_states.get('PREDICTIVE_OPP_CAPITULATION_REVERSAL', pd.Series(0.0, index=df.index))
        is_prophet_entry = (predictive_opp_score > prophet_entry_threshold) & ~is_veto_by_alert & is_not_hard_exit
        
        df.loc[is_prophet_entry, 'signal_type'] = '先知入场'

        # 步骤三：最后处理风险否决
        alert_veto_condition = is_score_sufficient & is_veto_by_alert & is_not_hard_exit
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0
        
        # [代码修改] 调用被授予“阅读圣旨”权力的新版史官
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df, df['signal_type'])
        self._finalize_signals()

    def _calculate_risk_penalty_score(self, risk_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V511.1 · 已废除】
        - 此方法已被“创世纪 XI · 独裁者”计划废除。其功能由 _adjudicate_risk_level 方法完全取代。
        """
        # [代码删除] 整个方法的内容都被删除，只留下废除声明
        return pd.Series(0.0, index=self.strategy.df_indicators.index), pd.DataFrame(index=self.strategy.df_indicators.index)

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, signal_type_series: pd.Series) -> pd.Series:
        """
        【V3.3 · 最后的敕令版】生成人类可读的信号摘要。
        - 核心革命: 史官被授予“阅读圣旨”(signal_type_series)的权力。
        - 核心法则: 只有当天的最终信号是“买入信号”时，才记录进攻项细节。对于“先知入场”等所有其他信号，进攻项列表必须为空。
        - 收益: 彻底根除战报污染，确保历史记录的纯净与真实。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        def process_details_df(details_df, is_risk_df=False):
            if details_df.empty: return pd.Series(dtype=object)
            long_df = details_df.melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            if not is_risk_df: long_df = long_df[long_df['score'] != 0].copy()
            else: long_df = long_df[long_df['score'] > 0].copy()
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
        
        # [代码新增] 颁布并执行“最后的敕令”
        def generate_final_summary(row):
            # 检查当天的“圣旨”（最终信号类型）
            final_signal_type = signal_type_series.get(row.name)
            
            # 法则一：只有当信号是“买入信号”时，才记录进攻项。
            if final_signal_type == '买入信号':
                offense_list = row['offense'] if isinstance(row['offense'], list) else []
            else:
                # 法则二：对于所有其他信号，进攻项列表必须为空。
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
        【V404.3 清理与对齐版】
        """
        df = self.strategy.df_indicators
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        
        final_buy_condition = df['signal_type'] == '买入信号'
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, 'exit_signal_code'] = 0

    def _adjudicate_risk_level(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V2.3 · 先知计划版】风险裁决者 (Risk Adjudicator)
        - 核心升级: 引入最高优先级的“预测性风险”，一旦触发，直接发布最高警报。
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
        # 从 judgment_params 中获取参数
        p_judge = get_params_block(self.strategy, 'judgment_params', {})
        p_alerts = p_judge.get('alert_level_thresholds', {})
        # 获取先知引擎的预测风险和阈值
        predictive_exhaustion_risk = atomic.get('PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION', pd.Series(0, index=df.index))
        prophet_threshold = get_param_value(p_judge.get('prophet_alert_threshold'), 0.7)
        level_3_archangel_threshold = get_param_value(p_alerts.get('level_3_archangel_threshold'), 0.7)
        level_3_threshold = get_param_value(p_alerts.get('level_3_top_reversal'), 0.8)
        level_2_resonance_threshold = get_param_value(p_alerts.get('level_2_bearish_resonance'), 0.7)
        level_2_euphoria_threshold = get_param_value(p_alerts.get('level_2_euphoria_risk'), 0.75)
        level_1_threshold = get_param_value(p_alerts.get('level_1_micro_risk'), 0.6)
        # 将先知神谕的判断条件置于最高优先级
        conditions = [
            predictive_exhaustion_risk > prophet_threshold,
            fused_risks_df['ARCHANGEL_RISK'] > level_3_archangel_threshold,
            fused_risks_df['TOP_REVERSAL'] > level_3_threshold,
            (fused_risks_df['BEARISH_RESONANCE'] > level_2_resonance_threshold) | (fused_risks_df['EUPHORIA_RISK'] > level_2_euphoria_threshold),
            fused_risks_df['MICRO_RISK'] > level_1_threshold,
        ]
        # 增加先知神谕对应的警报等级和原因
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

















