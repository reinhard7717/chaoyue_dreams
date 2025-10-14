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
        【V536.0 · 神盾战报版】
        - 核心升级: 为被“神盾协议”否决的离场信号，赋予专属的“神盾防御”信号类型。
        - 收益: 将“被动不离场”的隐性行为，转化为“主动防御成功”的显性战报，为报告层提供关键情报。
        """
        print("    --- [最高作战指挥部 V536.0 · 神盾战报版] 启动...")
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d).date() for d in probe_dates_str]
        chimera_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_CHIMERA_CONFLICT', pd.Series(0.0, index=df.index))
        dominant_signal_type = self._get_dominant_offense_type(score_details_df)
        is_reversal_day = (dominant_signal_type == 'positional')
        dynamic_chimera_score = chimera_conflict_score.where(~is_reversal_day, chimera_conflict_score * 0.5)
        confidence_damper = 1.0 - dynamic_chimera_score
        df['final_score'] = df['entry_score'] * confidence_damper
        df['alert_level'], df['alert_reason'], fused_risks_df = self._adjudicate_risk_level()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        df['risk_score'] = self.strategy.atomic_states.get('COGNITIVE_FUSED_RISK_SCORE', pd.Series(0.0, index=df.index)).fillna(0.0)
        p_judge_common = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge_common.get('final_score_threshold'), 400)
        df['signal_type'] = '无信号'
        is_score_sufficient = df['final_score'] > final_score_threshold
        is_veto_by_alert = df['alert_level'] >= 3
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        alert_veto_condition = is_score_sufficient & is_veto_by_alert
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0.0
        exit_triggers_df = self.strategy.exit_triggers
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
        # --- 神盾协议 (Aegis Protocol) 升级开始 ---
        gaia_bedrock_score = atomic.get('SCORE_FOUNDATION_BOTTOM_CONFIRMED', pd.Series(0.0, index=df.index))
        is_aegis_shield_active = (gaia_bedrock_score > 0.1)
        # 捕获原始的、未经神盾过滤的战术离场信号
        raw_tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
        # 定义被神盾否决的离场条件，并赋予新的信号类型
        aegis_defense_condition = raw_tactical_exit_mask & is_aegis_shield_active
        df.loc[aegis_defense_condition, 'signal_type'] = '神盾防御'

        # 最终的战术离场信号，是原始信号中未被神盾防御的部分
        tactical_exit_mask = raw_tactical_exit_mask & ~is_aegis_shield_active
        # --- 神盾协议 (Aegis Protocol) 升级结束 ---
        df.loc[strategic_exit_mask & ~potential_buy_condition, 'signal_type'] = '战略失效离场'
        df.loc[tactical_exit_mask & ~potential_buy_condition, 'signal_type'] = '趋势破位离场'
        df['final_score'] = df['final_score'].fillna(0).round().astype(int)
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df)
        self._finalize_signals()

    def _get_human_readable_summary(self, details_df: pd.DataFrame) -> pd.Series:
        """
        【V5.0 · 雅努斯归位协议版】
        - 核心革命: 部署“雅努斯归位协议”，彻底重构信号细节报告的生成逻辑。
        - 新核心逻辑:
          1. 只接收一个包含所有得分的 details_df。
          2. 对每一天，遍历所有激活的信号（得分非零）。
          3. 在方法内部根据得分的正负，将信号正确地分类到 'offense' 或 'risk' 列表。
          4. 确保每个信号的 'raw_score' 和 'base_score' 都被准确查找并记录。
        - 收益: 根除了因信号错误分类和元数据丢失导致的探针重算失败的最终根源。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        atomic = self.strategy.atomic_states
        df = self.strategy.df_indicators
        # 将所有列转换为数值类型，无法转换的填充为0
        details_df_numeric = details_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        def generate_summary_for_day(row):
            offense_list = []
            risk_list = []
            active_signals = row[row != 0]
            for signal_name, contribution in active_signals.items():
                meta = score_map.get(signal_name, {})
                is_risk = contribution < 0
                # 确定原始分数的来源
                raw_score_source = meta.get('raw_score_source', signal_name)
                raw_score = atomic.get(raw_score_source, pd.Series(0.0, index=df.index)).get(row.name, 0.0)
                # 确定基础分
                base_score_key = 'penalty_weight' if is_risk else 'score'
                base_score = meta.get(base_score_key, 0)
                # 创建信号字典
                signal_dict = {
                    'name': meta.get('cn_name', signal_name),
                    'score': int(round(contribution)),
                    'raw_score': raw_score,
                    'base_score': base_score
                }
                # 根据得分正负放入正确的列表
                if is_risk:
                    risk_list.append(signal_dict)
                else:
                    offense_list.append(signal_dict)
            return {'offense': offense_list, 'risk': risk_list}
        # 对每一行（每一天）应用这个总结函数
        return details_df_numeric.apply(generate_summary_for_day, axis=1)

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
        【V2.8 · 数据纯净法案版】风险裁决者 (Risk Adjudicator)
        - 核心修复: 不再将字符串类型的 alert_reason 存入 atomic_states，确保其只包含数值状态。
        - 核心逻辑: alert_reason 作为最终描述性结果，直接返回给 make_final_decisions 方法处理。
        - 收益: 根除了因数据类型污染导致的下游模块（如探针）崩溃问题。
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
        # 只将数值型的 alert_level 存入 atomic_states
        self.strategy.atomic_states['ALERT_LEVEL'] = alert_level.astype(np.int8)
        # [代码删除] 不再将字符串类型的 alert_reason 存入 atomic_states
        # self.strategy.atomic_states['ALERT_REASON'] = alert_reason
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
















