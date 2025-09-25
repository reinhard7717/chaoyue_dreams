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
        【V512.0 · 终极决策透明化版】
        - 核心修复: 引入了对“硬性离场信号”的感知。当买入决策被硬性离场信号
                      (如趋势破位)否决时，不仅将 signal_type 标记为 '趋势否决'，
                      同时强制将其 final_score 置为 0。
        - 收益: 彻底解决了“高分卖出”的报告悖论，实现了决策逻辑与最终报告的完美统一。
        """
        print("    --- [最高作战指挥部 V512.0 · 终极决策透明化版] 启动... ---") # [代码修改] 更新版本号
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        df['risk_penalty_score'], penalty_components_df = self._calculate_risk_penalty_score(risk_details_df)

        reportable_risk_df = risk_details_df.copy()
        if not penalty_components_df.empty:
            reportable_risk_df.update(penalty_components_df)

        df['dynamic_action'] = self._get_dynamic_combat_action()
        
        # 步骤1: 计算原始最终分
        df['final_score'] = df['entry_score'] - df['risk_penalty_score']
        
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 400)
        is_score_sufficient = df['final_score'] > final_score_threshold
        
        # 步骤2: 识别所有“一票否决”条件
        is_dynamic_veto = df['dynamic_action'] == 'AVOID'
        # [代码新增] 从 exit_triggers 中识别硬性离场否决
        is_hard_exit_veto = self.strategy.exit_triggers.any(axis=1)
        
        # 步骤3: 根据决策逻辑，确定最终信号类型
        df['signal_type'] = '无信号'
        
        # 只有在分数足够且没有任何否决的情况下，才产生买入信号
        final_buy_condition = is_score_sufficient & ~is_dynamic_veto & ~is_hard_exit_veto
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # [代码修改] 扩展否决逻辑，使其更具体
        vetoed_by_dynamic = is_score_sufficient & is_dynamic_veto
        df.loc[vetoed_by_dynamic, 'signal_type'] = '风险否决'
        
        vetoed_by_hard_exit = is_score_sufficient & is_hard_exit_veto & ~is_dynamic_veto
        df.loc[vetoed_by_hard_exit, 'signal_type'] = '趋势否决' # 更具体的否决原因
        
        # [代码修改] 核心修复：对所有被否决的信号执行“分数清零”
        # 这一步确保了报告中的分数与最终决策严格一致
        all_veto_conditions = vetoed_by_dynamic | vetoed_by_hard_exit
        df.loc[all_veto_conditions, 'final_score'] = 0
        
        # 步骤4: 生成报告并完成
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, reportable_risk_df)
        self._finalize_signals()

    def _calculate_risk_penalty_score(self, risk_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V505.0 · 配置驱动终极版】计算风险惩罚分
        - 核心重构: 逻辑已在V504.0中净化，此版本确认其完全依赖 signal_dictionary.json
                      中的 'penalty_weight' 配置，无需任何修改。
        """
        if risk_details_df.empty:
            return pd.Series(0.0, index=self.strategy.df_indicators.index), pd.DataFrame(index=self.strategy.df_indicators.index)

        score_map = get_params_block(self.strategy, 'score_type_map', {})
        penalty_components = {}

        for signal_name in risk_details_df.columns:
            raw_score = risk_details_df[signal_name].clip(lower=0)
            
            signal_meta = score_map.get(signal_name, {})
            penalty_weight = signal_meta.get('penalty_weight', 0)
            
            if penalty_weight > 0:
                weighted_score = raw_score * penalty_weight
                penalty_components[signal_name] = weighted_score
        
        if not penalty_components:
            return pd.Series(0.0, index=risk_details_df.index), pd.DataFrame(index=risk_details_df.index)

        penalty_components_df = pd.DataFrame(penalty_components)
        total_penalty_score = penalty_components_df.sum(axis=1).fillna(0)
        
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
