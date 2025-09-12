# 文件: strategies/trend_following/judgment_layer.py
# 统合判断层 (V404.2 - 逻辑净化版)
import pandas as pd
import numpy as np
from .utils import get_params_block, get_param_value

class JudgmentLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def _evaluate_holding_health(self, score_details_df: pd.DataFrame, risk_score_df: pd.DataFrame):
        """
        【V400.0 健康报告总汇版】【代码优化】
        - 优化说明: 原始代码使用 for 循环和 .at 索引器逐行构建字典，在处理大量数据时效率低下。
                    优化后的版本使用了列表推导式（List Comprehension）和 zip，
                    将两列数据并行处理并构建新的字典列表，然后一次性将其赋值给新列。
                    这种方法避免了 pandas 逐行操作的开销，执行效率更高。
        """
        df = self.strategy.df_indicators
        offensive_summary = df.get('offensive_momentum_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        risk_summary = df.get('risk_change_summary', pd.Series([{} for _ in range(len(df))], index=df.index))
        # 使用列表推导式替代 for 循环，实现向量化操作
        def create_summary(offense_report, risk_report):
            # 定义一个内部辅助函数来构建单个报告字典
            final_summary = {}
            # 检查进攻动能报告是否有效且非空
            if offense_report and isinstance(offense_report, dict) and any(offense_report.values()):
                final_summary['offense_momentum'] = offense_report
            # 检查风险变化报告是否有效且非空
            if risk_report and isinstance(risk_report, dict) and any(v for v in risk_report.values() if v):
                final_summary['risk_change'] = risk_report
            return final_summary
        # 使用列表推导式和zip并行处理两个Series，并调用辅助函数
        final_summaries = [create_summary(o, r) for o, r in zip(offensive_summary, risk_summary)]
        df['health_change_summary'] = final_summaries

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> pd.Series:
        """
        【V2.1 · 最终修复版】生成人类可读的信号摘要。
        - 核心修复 (本次修改):
          - [根除BUG] 彻底重写了信号名处理逻辑，使其与 reporting_layer 中的健壮逻辑完全一致。
          - 现在会正确、智能地剥离所有已知前缀（如 `SETUP_`, `TRIGGER_` 等），然后再去信号字典中查找中文名。
        - 收益: 解决了因错误的信号名处理逻辑可能导致的、隐晦的Pandas数据污染问题，这是导致下游报告层接收到全零数据的最终根源。
        """
        # 加载信号与中文名的映射字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        summaries = []
        
        # 新增行：定义已知的前缀列表，与 reporting_layer 保持完全一致
        prefixes_to_strip = ['SETUP_', 'TRIGGER_', 'PLAYBOOK_', 'DYN_', 'STRATEGIC_']

        # 使用主DataFrame的索引进行迭代，确保覆盖所有日期
        for idx in self.strategy.df_indicators.index:
            day_summary = {'offense': [], 'risk': []}
            
            # 安全地处理进攻项
            if idx in score_details_df.index:
                active_offense_signals = score_details_df.loc[idx]
                active_offense_signals = active_offense_signals[active_offense_signals > 0].sort_values(ascending=False)
                for signal, score in active_offense_signals.items():
                    # --- 新增开始：使用健壮的前缀剥离逻辑 ---
                    base_signal_name = signal
                    for prefix in prefixes_to_strip:
                        if signal.startswith(prefix):
                            base_signal_name = signal[len(prefix):]
                            break
                    # --- 新增结束 ---
                    
                    # 修改行：使用正确的基础信号名进行查找
                    cn_name = score_map.get(base_signal_name, {}).get('cn_name', base_signal_name)
                    day_summary['offense'].append(f"{cn_name} ({int(score)})")

            # 安全地处理风险项
            if idx in risk_details_df.index:
                active_risk_signals = risk_details_df.loc[idx]
                active_risk_signals = active_risk_signals[active_risk_signals > 0].sort_values(ascending=False)
                for signal, score in active_risk_signals.items():
                    # 风险信号通常没有前缀，直接查找即可
                    cn_name = score_map.get(signal, {}).get('cn_name', signal)
                    day_summary['risk'].append(f"{cn_name} ({int(score)})")
            
            summaries.append(day_summary)
            
        return pd.Series(summaries, index=self.strategy.df_indicators.index)

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V502.1 职责净化版】
        - 核心架构升级: 废除“否决票”机制，采用“风险惩罚分”模型。
        - 决策逻辑简化: 最终决策逻辑简化为 `最终得分 = 进攻分 - 风险惩罚分`。
        - 职责净化: 移除了对 `_generate_exit_triggers` 的调用，硬性离场决策完全交由上层模块处理。
        """
        print("    --- [最高作战指挥部 V502.1 职责净化版] 启动... ---")
        df = self.strategy.df_indicators
        
        df['risk_penalty_score'] = 0.0
        self._calculate_risk_penalty_score()

        df['signal_type'] = '无信号'
        df['dynamic_action'] = 'HOLD'
        self._evaluate_holding_health(score_details_df, risk_details_df)
        df['dynamic_action'] = self._get_dynamic_combat_action()
        # --- 买入决策核心逻辑 (纯数值化) ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 300)
        df['final_score'] = df['entry_score'] - df['risk_penalty_score']
        is_score_sufficient = df['final_score'] > final_score_threshold
        not_avoid = df['dynamic_action'] != 'AVOID'
        final_buy_condition = (
            is_score_sufficient &
            not_avoid
        )
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df)
        self._finalize_signals()

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V318.0 力学信号适配版】动态力学战术矩阵
        - 核心重构 (本次修改):
          - [信号适配] 废除了对旧版、布尔型力学信号的依赖。
          - 全面升级为消费由 DynamicMechanicsEngine V5.0+ 生成的、经过深度交叉验证的S级数值化信号。
          - `FORCE_ATTACK` (强攻): 由 `SCORE_FV_OFFENSIVE_RESONANCE_S` (进攻共振) 驱动。
          - `AVOID` (规避): 由 `SCORE_FV_RISK_EXPANSION_S` (风险扩张) 驱动。
        - 收益: 战术决策的依据更可靠，能更精确地识别“健康上涨”与“风险上涨”的本质区别。
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_score = pd.Series(0.0, index=df.index, dtype=np.float32)
        
        # --- 全面使用新版S级数值化信号 ---
        # 获取进攻共振S级分数，代表“纯粹的进攻”
        offensive_resonance_score = atomic.get('SCORE_FV_OFFENSIVE_RESONANCE_S', default_score)
        # 获取风险扩张S级分数，代表“高位滞涨/出货”风险
        risk_expansion_score = atomic.get('SCORE_FV_RISK_EXPANSION_S', default_score)

        # 定义基于数值化分数的战术状态
        # 当进攻共振分数很高时，采取强攻姿态
        is_force_attack = offensive_resonance_score > 0.6
        # 当风险扩张分数很高时，采取规避姿态
        is_avoid = risk_expansion_score > 0.6
        # 当两者分数都高时，代表多空激战，应谨慎；两者都低则代表方向不明，也应谨慎
        is_caution = (offensive_resonance_score > 0.4) & (risk_expansion_score > 0.4)

        actions = pd.Series('HOLD', index=df.index)
        # 注意赋值顺序，AVOID的优先级最高
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        
        return actions

    def _calculate_risk_penalty_score(self):
        """
        【V400.0 纯数值化版】计算风险惩罚分
        - 核心转变: 不再计算离散的“否决票”，而是累加连续的“风险惩罚分”。
        - 计算逻辑: risk_penalty_score += risk_signal_score * weight
        - 风险缓解: 采用数值化缓解 `risk_score * (1 - mitigator_score)`
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(0.0, index=df.index)

        def get_clipped_score(signal_name):
            """辅助函数：获取信号分并确保其非负"""
            return atomic.get(signal_name, default_series).clip(lower=0)

        # 风险1: 筹码结构严重失效 (权重: 300)
        risk_score = get_clipped_score('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE')
        df['risk_penalty_score'] += risk_score * 300
        
        # 风险3: 绝对否决信号 (权重: 200)
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            for signal_name in veto_signals:
                risk_score = get_clipped_score(signal_name)
                
                # 处理风险缓解
                if signal_name in mitigation_rules:
                    mitigators = mitigation_rules[signal_name].get('mitigated_by', [])
                    # 取最强的那个缓解信号作为最终缓解分
                    mitigator_score = pd.Series(0.0, index=df.index)
                    for m_signal in mitigators:
                        current_mitigator_score = get_clipped_score(m_signal)
                        mitigator_score = np.maximum(mitigator_score, current_mitigator_score)
                    
                    # 数值化缓解：风险分 * (1 - 缓解分)
                    net_risk_score = risk_score * (1 - mitigator_score)
                    df['risk_penalty_score'] += net_risk_score * 200
                else:
                    df['risk_penalty_score'] += risk_score * 200

        # 风险4: 风险分高于进攻分 (权重: 100)
        # 这里是一个状态判断，不是一个0-1的信号，所以我们将其转换为一个0或1的布尔分数
        risk_overrides_score = (df['risk_score'] > df['entry_score']).astype(float)
        is_in_ascent_phase = get_clipped_score('STRUCTURE_POST_ACCUMULATION_ASCENT_C')
        # 只有在非上升期，此风险才生效
        net_risk_score = risk_overrides_score * (1 - is_in_ascent_phase)
        df['risk_penalty_score'] += net_risk_score * 100
        
        # 风险5: 核心原子风险信号 (权重: 100)
        df['risk_penalty_score'] += get_clipped_score('CHIP_DYN_COST_FALLING') * 100
        df['risk_penalty_score'] += get_clipped_score('CHIP_DYN_WINNER_RATE_COLLAPSING') * 100
        
        # 风险6: 周线战略顶层风险 (权重: 300 for topping, 100 for bearish)
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_STRATEGIC_TOPPING_RISK_W') * 300
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_STRATEGIC_BEARISH_W') * 100
        
        # 风险7 & 8: 战略级筹码长期发散 (权重: 200 for diverging, 300 for accelerating)
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_CHIP_LONG_TERM_DIVERGENCE_D') * 200
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_CHIP_LONG_TERM_ACCEL_DIVERGENCE_D') * 300
        
        # 风险9: 高级S级风险信号 (权重: 200 for divergence, 300 for fleeing)
        df['risk_penalty_score'] += get_clipped_score('RISK_MTF_RSI_BEARISH_DIVERGENCE_S') * 200
        df['risk_penalty_score'] += get_clipped_score('SCORE_BEHAVIOR_BEARISH_RESONANCE_S_PLUS') * 300
        
        # 风险10: 更多S级陷阱信号 (权重: 300)
        df['risk_penalty_score'] += get_clipped_score('RISK_STATIC_DYN_COLLAPSE_S') * 300
        df['risk_penalty_score'] += get_clipped_score('RISK_FUND_FLOW_DECEPTIVE_RALLY_S_PLUS') * 300
        
        # 风险11: 顶层战略筹码风险 (权重: 200 for distribution, 300 for trap)
        df['risk_penalty_score'] += get_clipped_score('CONTEXT_CHIP_STRATEGIC_DISTRIBUTION') * 200
        df['risk_penalty_score'] += get_clipped_score('RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S') * 300
        
        # 风险12: 高级行为与结构陷阱 (权重: 300)
        df['risk_penalty_score'] += get_clipped_score('RISK_STATIC_HIGH_ALTITUDE_MULTI_DIVERGENCE_S') * 300
        df['risk_penalty_score'] += get_clipped_score('RISK_BEHAVIOR_DECEPTIVE_RALLY_LONG_TERM_S') * 300

        # 风险13: 力学层与认知层S级风险 (权重: 300)
        df['risk_penalty_score'] += get_clipped_score('RISK_DYN_MARKET_ENGINE_STALLING_S') * 300
        df['risk_penalty_score'] += get_clipped_score('RISK_DYN_STRUCTURAL_WEAKNESS_RALLY_S') * 300
        topping_danger_score = get_clipped_score('SCORE_STRUCTURE_TOPPING_DANGER_S')
        # 将0-1的评分转换为一个阈值判断后的0-1信号
        has_topping_danger = (topping_danger_score > 0.6).astype(float) * topping_danger_score
        df['risk_penalty_score'] += has_topping_danger * 300
        
        # 风险14: 基础层与结构层风险
        df['risk_penalty_score'] += get_clipped_score('RISK_VOL_PRICE_SPIKE_DOWN_A') * 300 # 权重300
        df['risk_penalty_score'] += get_clipped_score('STRUCTURE_BEARISH_CHANNEL_F') * 300 # 权重300
        df['risk_penalty_score'] += get_clipped_score('RISK_TRIGGER_MACD_DEATH_CROSS_B') * 100 # 权重100

        # 风险15: 筹码结构瓦解风险 (权重: 300)
        df['risk_penalty_score'] += get_clipped_score('SCENARIO_FORTRESS_INTERNAL_COLLAPSE_A') * 300
        df['risk_penalty_score'] += get_clipped_score('RISK_PEAK_BATTLE_DISTRIBUTION_A') * 300

        # 风险16: 资金流结构风险 (权重: 200)
        df['risk_penalty_score'] += get_clipped_score('RISK_FUND_FLOW_RETAIL_FOMO_B') * 200

        # 风险17: 结构性风险与市场状态
        df['risk_penalty_score'] += get_clipped_score('RISK_STRUCTURE_OVEREXTENDED_LONG_TERM_S') * 200 # 权重200
        df['risk_penalty_score'] += get_clipped_score('RISK_STRUCTURE_MTF_OVEREXTENDED_RESONANCE_S') * 300 # 权重300
        df['risk_penalty_score'] += get_clipped_score('STRUCTURE_REGIME_MEAN_REVERTING') * 200 # 权重200
    
    def _finalize_signals(self):
        """
        【V404.3 清理与对齐版】
        - 核心清理: 移除了已定义的但未使用的 `final_sell_condition` 和 `final_warning_condition` 变量，
                      使代码更整洁。
        - 逻辑确认: 确认本函数的核心职责是根据 `signal_type` 列，最终确定 `signal_entry` 等
                      用于回测引擎的布尔标志位，而不再修改 `final_score`。
        """
        df = self.strategy.df_indicators
        
        # 1. 初始化用于回测引擎的标准列
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0
        df['alert_reason'] = ''
        
        # 2. 根据最终决策的信号类型，设置买入标志位
        final_buy_condition = df['signal_type'] == '买入信号'
        df.loc[final_buy_condition, 'signal_entry'] = True
        
        # 3. 对于买入信号当天，确保没有遗留的卖出信息
        #    (这是一个健壮性检查，确保逻辑清晰)
        if 'exit_signal_code' in df.columns:
            df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']














