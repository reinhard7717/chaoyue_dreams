# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层
import pandas as pd
import numpy as np
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V507.0 · 惩罚透明化重构版】
        - 核心修复: 重构了风险惩罚分的计算与报告流程。现在 `_calculate_risk_penalty_score`
                      会返回更新后的 risk_details_df，确保所有惩罚分项都在报告中可见。
        """
        print("    --- [最高作战指挥部 V507.0 · 惩罚透明化重构版] 启动... ---") # 更新版本号
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        
        # 重构风险惩罚分的计算流程
        # 现在，risk_details_df 会被传入并被更新后返回
        df['risk_penalty_score'], risk_details_df = self._calculate_risk_penalty_score(risk_details_df)

        # --- 步骤 2: 获取亢奋风险并计算衰减因子 (逻辑不变，但现在 risk_details_df 已包含其他惩罚分) ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        euphoria_risk_score = atomic.get('COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION', pd.Series(0.0, index=df.index))
        
        euphoria_display_score = (euphoria_risk_score * 100).clip(0, 1000)
        # 使用 .loc 避免 SettingWithCopyWarning
        risk_details_df.loc[:, 'COGNITIVE_SCORE_RISK_EUPHORIC_ACCELERATION'] = euphoria_display_score
        
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
        # 此处调用的 _get_human_readable_summary 现在会消费一个包含了所有惩罚分项的 risk_details_df
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df)
        self._finalize_signals()

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

    def _calculate_risk_penalty_score(self, risk_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V502.0 · 惩罚透明化重构版】计算风险惩罚分
        - 核心重构: 此方法现在接收 risk_details_df，将计算出的带权重的惩罚分填入其中，
                      然后返回总惩罚分和更新后的 risk_details_df。
        """
        atomic = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=self.strategy.df_indicators.index)
        risk_components = {}
        
        # 创建一个副本以安全地修改
        reportable_risk_df = risk_details_df.copy()

        def get_clipped_score(signal_name):
            return atomic.get(signal_name, default_series).clip(lower=0)

        # --- 风险1: 绝对否决信号 (权重: 300) ---
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            for signal_name in veto_signals:
                weighted_score = get_clipped_score(signal_name) * 300
                risk_components[signal_name] = weighted_score
                reportable_risk_df[signal_name] = weighted_score # 将带权重的分值写入报告DF

        # --- 风险2: 顶层认知风险 ---
        cognitive_risk_score = get_clipped_score('COGNITIVE_FUSED_RISK_SCORE') * 1.0
        risk_components['COGNITIVE_FUSED_RISK_SCORE'] = cognitive_risk_score
        reportable_risk_df['COGNITIVE_FUSED_RISK_SCORE'] = cognitive_risk_score # 写入报告DF

        # --- 风险3: 各情报层S+级看跌共振风险 (权重: 150) ---
        bearish_resonance_signals = [
            'SCORE_BEHAVIOR_BEARISH_RESONANCE_S_PLUS', 'SCORE_CHIP_BEARISH_RESONANCE_S_PLUS',
            'SCORE_DYN_BEARISH_RESONANCE_S_PLUS', 'SCORE_FF_BEARISH_RESONANCE_S_PLUS',
            'SCORE_STRUCTURE_BEARISH_RESONANCE_S_PLUS', 'SCORE_FOUNDATION_BEARISH_RESONANCE_S_PLUS'
        ]
        for signal_name in bearish_resonance_signals:
            weighted_score = get_clipped_score(signal_name) * 150
            risk_components[signal_name] = weighted_score
            reportable_risk_df[signal_name] = weighted_score # 写入报告DF

        # --- 风险4: 行业周期风险 ---
        industry_stagnation_score = get_clipped_score('SCORE_INDUSTRY_STAGNATION') * 200
        risk_components['RISK_INDUSTRY_STAGNATION'] = industry_stagnation_score
        reportable_risk_df['SCORE_INDUSTRY_STAGNATION'] = industry_stagnation_score # 写入报告DF

        industry_downtrend_score = get_clipped_score('SCORE_INDUSTRY_DOWNTREND') * 300
        risk_components['RISK_INDUSTRY_DOWNTREND'] = industry_downtrend_score
        reportable_risk_df['SCORE_INDUSTRY_DOWNTREND'] = industry_downtrend_score # 写入报告DF
        
        total_penalty_score = pd.DataFrame(risk_components).sum(axis=1).fillna(0)
        
        # 返回总惩罚分和更新后的报告DF
        return total_penalty_score, reportable_risk_df

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
