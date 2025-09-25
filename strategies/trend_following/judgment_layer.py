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
        【V508.0 · 风险流重构版】
        - 核心重构: 全面适配 WarningLayer V3.0 的新数据流。
                      现在接收由 WarningLayer 提供的、包含所有原始风险分的 risk_details_df，
                      并将其传递给重构后的 _calculate_risk_penalty_score 进行加权和报告。
        """
        print("    --- [最高作战指挥部 V508.0 · 风险流重构版] 启动... ---") # 更新版本号
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        # risk_details_df 现在是包含所有原始风险分的完整“起诉书”
        # 将其传递给“法官” _calculate_risk_penalty_score 进行审判
        df['risk_penalty_score'], reportable_risk_df = self._calculate_risk_penalty_score(risk_details_df)

        # --- 步骤 2: 获取亢奋风险并计算衰减因子 ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        euphoria_risk_score = atomic.get('COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', pd.Series(0.0, index=df.index))
        
        # 将亢奋风险的“显示分”也加入到报告DF中
        # 注意：这里的 euphora_risk_score 是原始分(0-1)，而 reportable_risk_df 中的已经是加权惩罚分
        # 为了统一，我们应该在 _calculate_risk_penalty_score 中处理它
        attenuation_factor = (euphoria_risk_score * get_param_value(p_judge.get('euphoria_attenuation_multiplier'), 2.0)).clip(0, 1)

        # --- 步骤 3, 4, 5 保持不变 ---
        df['dynamic_action'] = self._get_dynamic_combat_action()
        euphoria_veto_threshold = get_param_value(p_judge.get('euphoria_veto_threshold'), 0.85)
        df.loc[euphoria_risk_score > euphoria_veto_threshold, 'dynamic_action'] = 'AVOID'
        df['final_score'] = df['entry_score'] * (1 - attenuation_factor) - df['risk_penalty_score']
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 400)
        is_score_sufficient = df['final_score'] > final_score_threshold
        not_avoid = df['dynamic_action'] != 'AVOID'
        final_buy_condition = is_score_sufficient & not_avoid
        df['signal_type'] = '无信号'
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # --- 步骤 6: 生成报告与清理 ---
        # reportable_risk_df 现在包含了所有带权重的惩罚分，报告将完全透明
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, reportable_risk_df)
        self._finalize_signals()

    def _calculate_risk_penalty_score(self, risk_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V503.0 · 配置驱动重构版】计算风险惩罚分
        - 核心重构: 不再硬编码信号列表。而是遍历传入的 risk_details_df 的所有列（信号），
                      从 score_type_map 配置中查找各自的 'penalty_weight'，
                      计算加权惩罚分，并就地更新 DataFrame 用于报告。
        """
        if risk_details_df.empty:
            return pd.Series(0.0, index=self.strategy.df_indicators.index), risk_details_df

        score_map = get_params_block(self.strategy, 'score_type_map', {})
        reportable_risk_df = risk_details_df.copy()

        # 遍历传入的、包含所有原始风险分的DataFrame的列
        for signal_name in reportable_risk_df.columns:
            raw_score = reportable_risk_df[signal_name].clip(lower=0)
            
            # 从配置中查找该信号的惩罚权重
            signal_meta = score_map.get(signal_name, {})
            penalty_weight = signal_meta.get('penalty_weight', 0) # 默认权重为0
            
            if penalty_weight > 0:
                # 计算加权惩罚分
                weighted_score = raw_score * penalty_weight
                # 就地更新DataFrame，用于最终报告
                reportable_risk_df[signal_name] = weighted_score
            else:
                # 如果没有配置权重，则该项不参与惩罚，在报告中分值也为0
                reportable_risk_df[signal_name] = 0.0
        
        # 总惩罚分是所有加权惩罚分之和
        total_penalty_score = reportable_risk_df.sum(axis=1).fillna(0)
        
        return total_penalty_score, reportable_risk_df

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
