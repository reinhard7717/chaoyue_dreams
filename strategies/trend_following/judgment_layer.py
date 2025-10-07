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
        【V532.0 · 幽灵驱逐协议版】
        - 核心革命: 执行“幽灵驱逐协议”，彻底清除所有在主计分逻辑后对 final_score 和 signal_details_cn 进行覆写的潜伏代码。
        - 核心逻辑: 严格遵循“计算 -> 裁决 -> 报告 -> 终结”的线性流程，杜绝任何形式的逻辑回环和事后篡改。
        - 收益: 确保了 TrendFollow 策略的战报纯洁性，根除了导致分数被神秘篡改的“幽灵BUG”。
        """
        print("    --- [最高作战指挥部 V532.0 · 幽灵驱逐协议版] 启动...") # 修改版本号
        df = self.strategy.df_indicators
        
        # 步骤 1: 计算最终得分 (基于已包含风险的 entry_score)
        chimera_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index))
        dominant_signal_type = self._get_dominant_offense_type(score_details_df)
        is_reversal_day = (dominant_signal_type == 'positional')
        dynamic_chimera_score = chimera_conflict_score.where(~is_reversal_day, chimera_conflict_score * 0.5)
        confidence_damper = 1.0 - dynamic_chimera_score
        df['final_score'] = (df['entry_score'] * confidence_damper)
        
        # 步骤 2: 进行风险裁决
        df['alert_level'], df['alert_reason'], fused_risks_df = self._adjudicate_risk_level()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        df['risk_score'] = self.strategy.atomic_states.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).fillna(0.0)
        
        # 步骤 3: 根据得分和风险裁决，生成最终信号类型
        p_judge_common = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge_common.get('final_score_threshold'), 400)
        
        df['signal_type'] = '无信号'
        is_score_sufficient = df['final_score'] > final_score_threshold
        is_veto_by_alert = df['alert_level'] >= 3
        
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        
        alert_veto_condition = is_score_sufficient & is_veto_by_alert
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0 # 被否决的信号，最终分清零
        
        exit_triggers_df = self.strategy.exit_triggers
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
        tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
        
        df.loc[strategic_exit_mask & ~potential_buy_condition, 'signal_type'] = '战略失效离场'
        df.loc[tactical_exit_mask & ~potential_buy_condition, 'signal_type'] = '趋势破位离场'
        
        # [代码净化] 确保在此之后，没有任何逻辑会再次修改 'final_score' 或 'signal_type'。
        # 所有与“先知”相关的判断、覆写逻辑必须被彻底清除。
        
        # 步骤 4: 生成人类可读的摘要报告 (基于最终确定的信号类型)
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df, df['signal_type'])
        
        # 步骤 5: 终结信号，为下游提供统一接口
        self._finalize_signals()

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame, signal_type_series: pd.Series) -> pd.Series:
        """
        【V3.8 · 赫尔墨斯净化协议版】
        - 核心修复: 在进行数值计算和类型转换前，强制对所有可能为NaN的值进行净化处理。
        - 核心逻辑: 在 get_risk_contribution 函数中，对 row['score'] 和 base_score 使用 .fillna(0) 或默认值。
        - 收益: 彻底解决了因上游信号出现 NaN 值而导致的 ValueError 运行时崩溃问题。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        def process_details_df(details_df, is_risk_df=False):
            if details_df is None or details_df.empty: return pd.Series(dtype=object)
            active_cols = details_df.columns[(details_df != 0).any()]
            if active_cols.empty: return pd.Series(dtype=object)
            long_df = details_df[active_cols].melt(ignore_index=False, var_name='signal', value_name='score').reset_index()
            # [代码修改] 在处理前先用fillna(0)净化score列，避免后续操作因NaN出错
            long_df = long_df[long_df['score'].fillna(0) != 0].copy()
            if long_df.empty: return pd.Series(dtype=object)
            date_col_name = long_df.columns[0]
            cn_name_map = {k: v.get('cn_name', k) for k, v in score_map.items() if isinstance(v, dict)}
            long_df['cn_name'] = long_df['signal'].map(cn_name_map).fillna(long_df['signal'])
            if is_risk_df:
                def get_risk_contribution(row):
                    meta = score_map.get(row['signal'], {})
                    # [代码修改] 对 base_score 和 row['score'] 进行健壮性处理
                    base_score = meta.get('score', 0) if isinstance(meta, dict) else 0
                    current_score = row.get('score', 0)
                    # 确保在乘法前两者都是有效数值
                    if pd.isna(current_score):
                        current_score = 0.0
                    return int(current_score * base_score)
                long_df['contribution'] = long_df.apply(get_risk_contribution, axis=1)
            else:
                # [代码修改] 对进攻项也进行净化
                long_df['contribution'] = long_df['score'].fillna(0).astype(int)
            # [代码修改] 过滤掉贡献值为0的项
            long_df = long_df[long_df['contribution'] != 0]
            long_df['summary_dict'] = long_df.apply(lambda row: {'name': row['cn_name'], 'score': row['contribution']}, axis=1)
            return long_df.groupby(date_col_name)['summary_dict'].apply(list)
        offense_summaries = process_details_df(score_details_df, is_risk_df=False)
        risk_summaries = process_details_df(risk_details_df, is_risk_df=True)
        summary_df = pd.DataFrame({'offense': offense_summaries, 'risk': risk_summaries}).reindex(self.strategy.df_indicators.index)
        def generate_final_summary(row):
            final_signal_type = signal_type_series.get(row.name)
            if final_signal_type == '买入信号':
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
        
        # 注意：这里的信号名可能需要根据你的 intelligence layer 的最终输出进行调整
        offensive_resonance_score = atomic.get('SCORE_DYN_BULLISH_RESONANCE', default_score)
        risk_expansion_score = atomic.get('SCORE_DYN_BEARISH_RESONANCE', default_score)
        
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
        
        # 统一号令：只为“买入信号”升起'signal_entry'旗帜。
        final_buy_condition = (df['signal_type'] == '买入信号')
        
        df.loc[final_buy_condition, 'signal_entry'] = True
        df.loc[final_buy_condition, 'exit_signal_code'] = 0

    def _adjudicate_risk_level(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V2.7 · API标准用法修正案版】风险裁决者 (Risk Adjudicator)
        - 核心修复: 修正了对 get_param_value 函数的错误调用，解决了因传递过多参数导致的 TypeError。
        - 核心逻辑: 严格遵循“先用 .get() 从字典取值，再用 get_param_value 解析”的标准用法。
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
            fused_risks[category] = np.maximum.reduce(signal_scores) if signal_scores else pd.Series(0.0, index=df.index)
        fused_risks_df = pd.DataFrame(fused_risks, index=df.index)
        p_judge = get_params_block(self.strategy, 'judgment_day_params', {})
        predictive_exhaustion_risk = atomic.get('PREDICTIVE_RISK_CLIMACTIC_RUN_EXHAUSTION', pd.Series(0, index=df.index))
        # [代码修改] 修正所有 get_param_value 的调用方式
        prophet_threshold = get_param_value(p_judge.get('prophet_alert_threshold'), 0.7)
        archangel_threshold = get_param_value(p_judge.get('archangel_alert_threshold'), 0.7)
        top_reversal_threshold = get_param_value(p_judge.get('top_reversal_alert_threshold'), 0.8)
        resonance_threshold = get_param_value(p_judge.get('bearish_resonance_alert_threshold'), 0.7)
        euphoria_threshold = get_param_value(p_judge.get('euphoria_alert_threshold'), 0.75)
        micro_risk_threshold = get_param_value(p_judge.get('micro_risk_alert_threshold'), 0.6)
        is_uptrend_context = df.get('close_D', 0) > df.get('EMA_5_D', 0)
        conditions = [
            (predictive_exhaustion_risk > prophet_threshold) & is_uptrend_context,
            fused_risks_df['ARCHANGEL_RISK'] > archangel_threshold,
            fused_risks_df['TOP_REVERSAL'] > top_reversal_threshold,
            (fused_risks_df['BEARISH_RESONANCE'] > resonance_threshold) | (fused_risks_df['EUPHORIA_RISK'] > euphoria_threshold),
            fused_risks_df['MICRO_RISK'] > micro_risk_threshold,
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

    def _get_dominant_offense_type(self, score_details_df: pd.DataFrame) -> pd.Series:
        """
        【V1.0 · 新增】识别每日最强的进攻信号及其类型 ('positional' 或 'dynamic')。
        """
        if score_details_df is None or score_details_df.empty:
            return pd.Series('unknown', index=self.strategy.df_indicators.index)

        # 获取信号字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        # 找出得分最高的信号列名
        dominant_signal_names = score_details_df.idxmax(axis=1)
        
        # 创建一个从信号名到类型的映射
        signal_to_type_map = {
            name: meta.get('type', 'unknown') 
            for name, meta in score_map.items() 
            if isinstance(meta, dict)
        }
        
        # 将最强信号名映射到其类型
        dominant_types = dominant_signal_names.map(signal_to_type_map).fillna('unknown')
        
        return dominant_types.reindex(self.strategy.df_indicators.index).fillna('unknown')
















