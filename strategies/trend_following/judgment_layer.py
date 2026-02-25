# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层
import pandas as pd
import numpy as np
from typing import Tuple
from strategies.trend_following.utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V538.4 · 决策层命名规范版】
        - 核心重构: 移除了 Chimera (喀迈拉)、Aegis (神盾)、Gaia (盖亚) 等异教徒与神话命名。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        debug_params = get_params_block(self.strategy, 'debug_params', {})
        probe_dates_str = debug_params.get('probe_dates', [])
        probe_dates = [pd.to_datetime(d).date() for d in probe_dates_str]
        strategy_conflict_score = self.strategy.atomic_states.get('COGNITIVE_SCORE_STRATEGY_CONFLICT', pd.Series(0.0, index=df.index))
        dominant_signal_type = self._get_dominant_offense_type(score_details_df)
        is_reversal_day = (dominant_signal_type == 'positional')
        dynamic_conflict_score = strategy_conflict_score.where(~is_reversal_day, strategy_conflict_score * 0.5)
        confidence_damper = 1.0 - dynamic_conflict_score
        total_offensive_score = df['entry_score']
        total_risk_sum = df['total_risk_sum']
        net_score = total_offensive_score - total_risk_sum
        df['final_score'] = net_score * confidence_damper
        df['dynamic_action'] = self._get_dynamic_combat_action()
        df['risk_score'] = total_risk_sum.fillna(0.0)
        p_judge_common = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge_common.get('final_score_threshold'), 400)
        df['signal_type'] = '无信号'
        is_score_sufficient = df['final_score'] > final_score_threshold
        potential_buy_condition = is_score_sufficient
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        exit_triggers_df = self.strategy.exit_triggers
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
        macro_bedrock_score = atomic.get('SCORE_FOUNDATION_BOTTOM_CONFIRMED', pd.Series(0.0, index=df.index))
        is_macro_bottom_support_active = (macro_bedrock_score > 0.1)
        raw_tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
        macro_defense_condition = raw_tactical_exit_mask & is_macro_bottom_support_active
        df.loc[macro_defense_condition, 'signal_type'] = '底座防御'
        df.loc[strategic_exit_mask & ~potential_buy_condition, 'signal_type'] = '战略失效离场'
        df['final_score'] = df['final_score'].fillna(0).round().astype(int)
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df)
        self._finalize_signals()

    def _get_human_readable_summary(self, details_df: pd.DataFrame) -> pd.Series:
        """
        【V5.3 · 统一情报总线版 & 风险项调试增强版】
        - 核心修复: 在查找原始分(raw_score)时，不再只依赖 atomic_states。
                      同样建立一个临时的“统一情报总线”来合并 atomic_states 和 playbook_states，
                      确保任何来源的信号都能被正确回溯。
        - 调试增强: 增加对 SCORE_CHIP_AXIOM_HOLDER_SENTIMENT 信号在 summary 生成前的检查。
        """
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        all_available_signals = self.strategy.atomic_states.copy()
        all_available_signals.update(self.strategy.playbook_states)
        df = self.strategy.df_indicators
        details_df_numeric = details_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        def generate_summary_for_day(row):
            offense_list = []
            risk_list = []
            # # --- 调试增强: 检查特定日期和信号的原始数据 ---
            # if not df.empty and row.name.date() == pd.to_datetime('2025-12-10').date():
            #     print(f"    -> [JudgmentLayer Debug] _get_human_readable_summary processing row for {row.name.strftime('%Y-%m-%d')}")
            #     if 'SCORE_CHIP_AXIOM_HOLDER_SENTIMENT' in row.index:
            #         print(f"        - SCORE_CHIP_AXIOM_HOLDER_SENTIMENT contribution in row: {row['SCORE_CHIP_AXIOM_HOLDER_SENTIMENT']:.2f}")
            #     else:
            #         print(f"        - SCORE_CHIP_AXIOM_HOLDER_SENTIMENT column NOT FOUND in row for {row.name.strftime('%Y-%m-%d')}")
            #     print(f"        - All non-zero contributions in row: {row[row != 0].to_dict()}")
            # --- 调试增强结束 ---
            active_signals = row[row != 0]
            for signal_name, contribution in active_signals.items():
                meta = score_map.get(signal_name, {})
                is_risk = contribution < 0
                raw_score_source = meta.get('raw_score_source', signal_name)
                raw_score = all_available_signals.get(raw_score_source, pd.Series(0.0, index=df.index)).get(row.name, 0.0)
                base_score_key = 'penalty_weight' if is_risk else 'score'
                base_score = meta.get(base_score_key, 0)
                signal_dict = {
                    'name': meta.get('cn_name', signal_name),
                    'internal_name': signal_name,
                    'score': int(round(contribution)),
                    'raw_score': raw_score,
                    'base_score': base_score
                }
                if is_risk:
                    risk_list.append(signal_dict)
                else:
                    offense_list.append(signal_dict)
            return {'offense': offense_list, 'risk': risk_list}
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

    def _get_dominant_offense_type(self, score_details_df: pd.DataFrame) -> pd.Series:
        """
        【V1.0】识别每日最强的进攻信号及其类型 ('positional' 或 'dynamic')。
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
















