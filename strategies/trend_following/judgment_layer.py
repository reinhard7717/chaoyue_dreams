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
        【V538.1 · 进攻风险分离与全面惩罚版】
        - 核心革命: final_score 现在是 (总进攻分 - 总风险惩罚) * confidence_damper。
        - 核心增强: df['risk_score'] 现在记录所有负向信号的绝对值之和 (total_risk_sum)。
        - 收益: 实现了从“基于规则的被动离场”到“基于状态的主动防御”的哲学飞跃。
        """
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
        # 获取总进攻分和总风险惩罚分
        total_offensive_score = df['entry_score'] # 此时 entry_score 已经是 OffensiveLayer 返回的总进攻分
        total_risk_sum = df['total_risk_sum'] # 从 IntelligenceLayer 传递过来的总风险惩罚分
        # 计算净得分：总进攻分 - 总风险惩罚
        net_score = total_offensive_score - total_risk_sum
        # 应用信心阻尼器到净得分
        df['final_score'] = net_score * confidence_damper
        df['alert_level'], df['alert_reason'], fused_risks_df = self._adjudicate_risk_level()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        # df['risk_score'] 现在直接使用 total_risk_sum，代表所有负向信号的绝对值之和
        df['risk_score'] = total_risk_sum.fillna(0.0) 
        p_judge_common = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge_common.get('final_score_threshold'), 400)
        df['signal_type'] = '无信号'
        is_score_sufficient = df['final_score'] > final_score_threshold
        # 核心否决逻辑
        is_veto_by_alert = df['alert_level'] >= get_param_value(p_judge_common.get('veto_alert_level'), 3)
        potential_buy_condition = is_score_sufficient & ~is_veto_by_alert
        df.loc[potential_buy_condition, 'signal_type'] = '买入信号'
        alert_veto_condition = is_score_sufficient & is_veto_by_alert
        df.loc[alert_veto_condition, 'signal_type'] = '风险否决'
        df.loc[alert_veto_condition, 'final_score'] = 0.0 # 风险否决时最终分归零
        exit_triggers_df = self.strategy.exit_triggers
        strategic_exit_mask = exit_triggers_df.get('EXIT_STRATEGY_INVALIDATED', pd.Series(False, index=df.index))
        gaia_bedrock_score = atomic.get('SCORE_FOUNDATION_BOTTOM_CONFIRMED', pd.Series(0.0, index=df.index))
        is_aegis_shield_active = (gaia_bedrock_score > 0.1)
        raw_tactical_exit_mask = exit_triggers_df.get('EXIT_TREND_BROKEN', pd.Series(False, index=df.index)) & ~strategic_exit_mask
        aegis_defense_condition = raw_tactical_exit_mask & is_aegis_shield_active
        df.loc[aegis_defense_condition, 'signal_type'] = '神盾防御'
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

    def _adjudicate_risk_level(self) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """
        【V2.11 · 全面风险感知与分级警报版】风险裁决者 (Risk Adjudicator)
        - 核心升级: 扩展风险类别，将更多高优先级、高影响力的风险信号纳入警报裁决体系。
        - 核心逻辑: 根据信号的性质和重要性，将其归类到不同的警报等级（3级红色、2级橙色、1级黄色）。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        # 扩展风险类别，将更多关键风险信号纳入警报裁决体系
        risk_categories = {
            # 3级红色警报：最高优先级，系统性风险或明确顶部信号
            'ARCHANGEL_RISK': ['SCORE_ARCHANGEL_TOP_REVERSAL'], # 假设此信号存在且为终极顶部反转
            'COGNITIVE_SYSTEMIC_RISK': [ # 认知层面的系统性风险或重大派发陷阱
                'COGNITIVE_RISK_KEY_SUPPORT_BREAK',
                'COGNITIVE_RISK_HIGH_LEVEL_STRUCTURAL_COLLAPSE',
                'COGNITIVE_RISK_BULL_TRAP_DISTRIBUTION',
                'COGNITIVE_RISK_RETAIL_FOMO_RETREAT',
                'COGNITIVE_RISK_HARVEST_CONFIRMATION',
                'COGNITIVE_RISK_LONG_TERM_PROFIT_DISTRIBUTION',
                'COGNITIVE_RISK_LIQUIDITY_TRAP',
                'COGNITIVE_RISK_TREND_EXHAUSTION',
                'FUSION_RISK_DISTRIBUTION_PRESSURE',
                'FUSION_RISK_STAGNATION',
                'PROCESS_FUSION_TREND_EXHAUSTION_SYNDROME',
                'SCORE_CHIP_RISK_DISTRIBUTION_WHISPER',
                'SCORE_RISK_BREAKOUT_FAILURE_CASCADE'
            ],
            # 2级橙色警报：显著风险，需要高度关注
            'EUPHORIA_RISK': ['COGNITIVE_SCORE_RISK_EUPHORIA_ACCELERATION'], # 假设此信号存在
            'DECEPTION_RISK': ['SCORE_FF_DECEPTION_RISK'], # 资金流诡道风险
            'BEARISH_DIVERGENCE_RISK': [ # 各层面的看跌背离
                'SCORE_BEHAVIOR_BEARISH_DIVERGENCE_QUALITY',
                'SCORE_BEHAVIOR_BEARISH_DIVERGENCE',
                'SCORE_FUND_FLOW_BEARISH_DIVERGENCE',
                'SCORE_STRUCTURE_BEARISH_DIVERGENCE',
                'SCORE_PATTERN_BEARISH_DIVERGENCE',
                'SCORE_DYNAMIC_MECHANICS_BEARISH_DIVERGENCE',
                'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE' # 负向部分代表风险
            ],
            'LIQUIDITY_DRAIN_RISK': [ # 流动性枯竭或结构空心化
                'SCORE_RISK_LIQUIDITY_DRAIN',
                'SCORE_CHIP_HOLLOWING_OUT_RISK',
                'PROCESS_RISK_VPA_EFFICIENCY_DECAY', # VPA效率衰减可能导致流动性问题
                'PROCESS_STRATEGY_DYN_VS_CHIP_DECAY', # 力学筹码共振看跌
                'COGNITIVE_RISK_CYCLICAL_TOP' # 周期顶部可能伴随流动性问题
            ],
            'TOP_REVERSAL_PROCESS_RISK': [ # 过程层面的顶部反转信号
                'PROCESS_META_FOUNDATION_TOP_REVERSAL',
                'PROCESS_META_STRUCTURE_TOP_REVERSAL',
                'PROCESS_META_PATTERN_TOP_REVERSAL',
                'PROCESS_META_DYNAMIC_MECHANICS_TOP_REVERSAL',
                'PROCESS_META_CHIP_TOP_REVERSAL',
                'PROCESS_META_FUND_FLOW_TOP_REVERSAL',
                'PROCESS_META_MICRO_BEHAVIOR_TOP_REVERSAL',
                'PROCESS_META_BEHAVIOR_TOP_REVERSAL'
            ],
            # 1级黄色警报：早期预警或一般性风险
            'MICRO_STRUCTURAL_RISK': [ # 微观或结构层面的早期风险
                'COGNITIVE_SCORE_RISK_POWER_SHIFT_TO_RETAIL', # 假设此信号存在
                'COGNITIVE_SCORE_RISK_MAIN_FORCE_CONVICTION_WEAKENING' # 假设此信号存在
            ],
            'EARLY_WARNING_RISK': [ # 其他早期或一般性风险
                'SCORE_BEHAVIOR_PRICE_DOWNWARD_MOMENTUM',
                'SCORE_RISK_UNRESOLVED_PRESSURE',
                'SCORE_CHIP_RETAIL_VULNERABILITY',
                'COGNITIVE_RISK_MARKET_UNCERTAINTY',
                'COGNITIVE_RISK_T0_ARBITRAGE_PRESSURE',
                'PROCESS_META_HOT_SECTOR_COOLING',
                'PROCESS_META_WINNER_CONVICTION_DECAY',
                'SCORE_BEHAVIOR_DISTRIBUTION_INTENT' # 派发意图
            ]
        }
        fused_risks = {}
        for category, signals in risk_categories.items():
            signal_scores = [atomic.get(s, pd.Series(0.0, index=df.index)).reindex(df.index).fillna(0.0) for s in signals]
            # 对于 PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE 这样的双极信号，我们只关心其负向风险部分
            if category == 'BEARISH_DIVERGENCE_RISK' and 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE' in signals:
                # 确保只取负值（代表风险）的绝对值
                processed_signal_scores = [s.clip(upper=0).abs() if s.name == 'PROCESS_META_PRICE_VS_MOMENTUM_DIVERGENCE' else s for s in signal_scores]
                fused_risks[category] = np.maximum.reduce(processed_signal_scores) if processed_signal_scores else pd.Series(0.0, index=df.index)
            else:
                fused_risks[category] = np.maximum.reduce(signal_scores) if signal_scores else pd.Series(0.0, index=df.index)
        fused_risks_df = pd.DataFrame(fused_risks, index=df.index)
        p_judge = get_params_block(self.strategy, 'judgment_day_params', {})
        # 获取所有风险类别的阈值
        archangel_threshold = get_param_value(p_judge.get('archangel_alert_threshold'), 0.7)
        cognitive_systemic_threshold = get_param_value(p_judge.get('cognitive_systemic_alert_threshold'), 0.7)
        euphoria_threshold = get_param_value(p_judge.get('euphoria_alert_threshold'), 0.75)
        deception_risk_threshold = get_param_value(p_judge.get('deception_risk_alert_threshold'), 0.7)
        bearish_divergence_threshold = get_param_value(p_judge.get('bearish_divergence_alert_threshold'), 0.6)
        liquidity_drain_threshold = get_param_value(p_judge.get('liquidity_drain_alert_threshold'), 0.6)
        top_reversal_process_threshold = get_param_value(p_judge.get('top_reversal_process_alert_threshold'), 0.6)
        micro_structural_threshold = get_param_value(p_judge.get('micro_structural_alert_threshold'), 0.5)
        early_warning_threshold = get_param_value(p_judge.get('early_warning_alert_threshold'), 0.5)
        is_uptrend_context = df.get('close_D', 0) > df.get('EMA_5_D', 0)
        conditions = [
            fused_risks_df['ARCHANGEL_RISK'] > archangel_threshold,
            fused_risks_df['COGNITIVE_SYSTEMIC_RISK'] > cognitive_systemic_threshold,
            fused_risks_df['EUPHORIA_RISK'] > euphoria_threshold,
            fused_risks_df['DECEPTION_RISK'] > deception_risk_threshold,
            fused_risks_df['BEARISH_DIVERGENCE_RISK'] > bearish_divergence_threshold,
            fused_risks_df['LIQUIDITY_DRAIN_RISK'] > liquidity_drain_threshold,
            fused_risks_df['TOP_REVERSAL_PROCESS_RISK'] > top_reversal_process_threshold,
            fused_risks_df['MICRO_STRUCTURAL_RISK'] > micro_structural_threshold,
            fused_risks_df['EARLY_WARNING_RISK'] > early_warning_threshold,
        ]
        choices_level = [3, 3, 2, 2, 2, 2, 2, 1, 1] # 警报等级
        choices_reason = [
            '红色警报: 明确顶部形态',
            '红色警报: 系统性风险或重大派发',
            '橙色警报: 亢奋风险',
            '橙色警报: 资金流诡道风险',
            '橙色警报: 熊市背离风险',
            '橙色警报: 流动性枯竭或结构空心化',
            '橙色警报: 过程顶部反转风险',
            '黄色警报: 微观结构风险',
            '黄色警报: 早期预警或一般性风险'
        ]
        alert_level = pd.Series(np.select(conditions, choices_level, default=0), index=df.index)
        alert_reason = pd.Series(np.select(conditions, choices_reason, default=''), index=df.index)
        self.strategy.atomic_states['ALERT_LEVEL'] = alert_level.astype(np.int8)
        return alert_level, alert_reason, fused_risks_df

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
















