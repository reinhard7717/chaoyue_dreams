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
        【V523.0 · 教皇敕令版】
        - 核心革命: 建立绝对决策优先级。先知入场拥有最高进攻决策权。
        - 核心逻辑: 1. 优先判断“先知入场”，一旦触发，立即设置信号并强制将final_score归零。
                      2. 仅在“先知入场”未触发时，才继续判断常规“买入信号”和“风险否决”。
        - 收益: 彻底解决了“先知入场”与“买入信号”在同一天触发时的逻辑冲突和分数污染问题。
        """
        print("    --- [最高作战指挥部 V523.0 · 教皇敕令版] 启动...")
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

        # 重构决策优先级，颁布“教皇敕令”
        # 准备所有判断条件
        is_not_hard_exit = ~is_hard_exit_veto
        is_score_sufficient = df['final_score'] > final_score_threshold
        is_veto_by_alert = df['alert_level'] >= 3
        
        # 决策优先级 1: 先知入场 (拥有最高权威，无视常规风险否决)
        prophet_entry_threshold = get_param_value(p_judge.get('prophet_entry_threshold'), 0.6) # 阈值可以根据新模型适当调整
        predictive_opp_score = self.strategy.atomic_states.get('PREDICTIVE_OPP_CAPITULATION_REVERSAL', pd.Series(0.0, index=df.index))
        # 移除了 `& ~is_veto_by_alert`，赋予神谕无上权力
        is_prophet_entry = (predictive_opp_score > prophet_entry_threshold) & is_not_hard_exit
        df.loc[is_prophet_entry, 'signal_type'] = '先知入场'
        df.loc[is_prophet_entry, 'final_score'] = 0 # 教皇敕令：神谕降临之日，凡人的分数皆为虚无

        # 决策优先级 2: 常规买入 (必须在非先知入场日)
        is_not_prophet_entry = ~is_prophet_entry
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert & is_not_hard_exit & is_not_prophet_entry
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        
        # 决策优先级 3: 风险否决 (必须在非先知入场日)
        alert_veto_condition = is_score_sufficient & is_veto_by_alert & is_not_hard_exit & is_not_prophet_entry
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0
        
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
        【V3.5 · 绝对过滤版】生成人类可读的信号摘要。
        - 核心升级: 升级了过滤逻辑，确保只有在最终报告中显示为非零的信号才被记录。
        - 核心法则: 风险信号的过滤标准从 score > 0 升级为 (score * 1000).astype(int) > 0。
        - 收益: 彻底杜绝了因浮点数精度问题导致的“幽灵信号”（显示为0分的激活项）。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        def process_details_df(details_df, is_risk_df=False):
            if details_df.empty: return pd.Series(dtype=object)
            long_df = details_df.melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            
            # 升级过滤逻辑，确保只有最终显示为非零的信号才被包含
            if not is_risk_df:
                # 对于进攻项，最终显示为 int(score)，所以按此过滤
                long_df = long_df[long_df['score'].astype(int) != 0].copy()
            else:
                # 对于风险项，最终显示为 int(score * 1000)，所以按此过滤
                long_df = long_df[(long_df['score'] * 1000).astype(int) > 0].copy()

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

















