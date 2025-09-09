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

    def _generate_exit_triggers(self) -> pd.DataFrame:
        """
        【V504.0 新增】离场触发器生成器
        - 核心职责: 根据“三道防线”原则，生成一个包含所有离场原因的布尔型DataFrame。
        """
        df = self.strategy.df_indicators
        triggers_df = pd.DataFrame(index=df.index)
        
        # --- 防线一: 致命一击 (Critical Hit) ---
        critical_risk_details = self.strategy.critical_risk_details
        triggers_df['EXIT_CRITICAL_HIT'] = critical_risk_details.sum(axis=1) > 0

        # --- 防线二: 风险溢出 (Risk Overflow) ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        overflow_threshold = get_param_value(p_judge.get('risk_overflow_threshold'), 1000)
        triggers_df['EXIT_RISK_OVERFLOW'] = self.strategy.risk_score > overflow_threshold

        # --- 防线三: 趋势破位 (Trend Broken) ---
        #  实现“第三道防线”，即基于移动平均线的技术性移动止盈/止损。
        #           这是保护利润和控制回撤的关键纪律。
        p_pos_mgmt = get_params_block(self.strategy, 'position_management_params')
        p_trailing = p_pos_mgmt.get('trailing_stop', {})
        triggers_df['EXIT_TREND_BROKEN'] = pd.Series(False, index=df.index) # 默认不触发
        if get_param_value(p_trailing.get('enabled'), False):
            model = get_param_value(p_trailing.get('trailing_model'))
            if model == 'MOVING_AVERAGE':
                ma_type = get_param_value(p_trailing.get('ma_type'), 'EMA').upper()
                ma_period = get_param_value(p_trailing.get('ma_period'), 20)
                ma_col = f'{ma_type}_{ma_period}_D'
                if ma_col in df.columns:
                    # 当日收盘价低于移动平均线，则触发趋势破位信号
                    triggers_df['EXIT_TREND_BROKEN'] = df['close_D'] < df[ma_col]
                    # print(f"    -> [第三道防线] 已激活：趋势破位监控 (基于 {ma_col})。")
                else:
                    print(f"    -> [第三道防线-警告] 无法找到移动平均线列: {ma_col}，趋势破位监控未激活。")

        # --- 防线四: 利润保护 (Profit Protector - 暂未完全实现) ---
        p_protector = p_judge.get('profit_protector', {})
        if get_param_value(p_protector.get('enabled'), False):
            max_drawdown_pct = get_param_value(p_protector.get('max_drawdown_pct'), 0.15)
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)
        else:
            triggers_df['EXIT_PROFIT_PROTECT'] = pd.Series(False, index=df.index)

        return triggers_df

    def _get_human_readable_summary(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame) -> pd.Series:
        """
        【新增】生成人类可读的信号摘要。
        - 核心职责: 遍历每日激活的进攻和风险信号，查询配置文件中的中文名，并格式化为字符串。
        """
        # 加载信号与中文名的映射字典
        score_map = get_params_block(self.strategy, 'score_type_map', {})
        
        summaries = []
        # 迭代每一天的数据
        for idx in score_details_df.index:
            day_summary = {'offense': [], 'risk': []}
            
            # 处理进攻项
            active_offense_signals = score_details_df.loc[idx]
            active_offense_signals = active_offense_signals[active_offense_signals > 0].sort_values(ascending=False)
            for signal, score in active_offense_signals.items():
                # 从信号名中提取基础名称 (例如从 DYN_SCORE_... 提取 SCORE_...)
                base_signal_name = signal.split('_', 1)[1] if '_' in signal else signal
                cn_name = score_map.get(base_signal_name, {}).get('cn_name', base_signal_name)
                day_summary['offense'].append(f"{cn_name} ({int(score)})")

            # 处理风险项
            active_risk_signals = risk_details_df.loc[idx]
            active_risk_signals = active_risk_signals[active_risk_signals > 0].sort_values(ascending=False)
            for signal, score in active_risk_signals.items():
                base_signal_name = signal.split('_', 1)[1] if '_' in signal else signal
                cn_name = score_map.get(base_signal_name, {}).get('cn_name', base_signal_name)
                day_summary['risk'].append(f"{cn_name} ({int(score)})")
            
            summaries.append(day_summary)
            
        return pd.Series(summaries, index=score_details_df.index)

    def make_final_decisions(self, score_details_df: pd.DataFrame, risk_details_df: pd.DataFrame):
        """
        【V502.0 纯数值化决策版】
        - 核心架构升级: 废除“否决票”机制，采用“风险惩罚分”模型。
        - 决策逻辑简化: 最终决策逻辑简化为 `最终得分 = 进攻分 - 风险惩罚分`。
                        如果最终得分高于入场阈值，则产生买入信号。
                        这使得决策过程更平滑、更稳健，并从根本上解决了类型不匹配问题。
        """
        print("    --- [最高作战指挥部 V502.0 纯数值化决策版] 启动... ---")
        df = self.strategy.df_indicators
        # 初始化新的风险惩罚分列，并调用新的计算函数
        df['risk_penalty_score'] = 0.0
        self._calculate_risk_penalty_score()
        df['signal_type'] = '无信号'
        df['dynamic_action'] = 'HOLD'
        self._evaluate_holding_health(score_details_df, risk_details_df)
        df['dynamic_action'] = self._get_dynamic_combat_action()
        exit_triggers = self._generate_exit_triggers()
        is_sell_signal = exit_triggers.any(axis=1)
        self.strategy.exit_triggers = exit_triggers
        df.loc[is_sell_signal, 'signal_type'] = '卖出信号'
        # --- 买入决策核心逻辑 (纯数值化) ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        # 使用单一的最终得分阈值
        final_score_threshold = get_param_value(p_judge.get('final_score_threshold'), 300)
        df['final_score'] = df['entry_score'] - df['risk_penalty_score']
        is_score_sufficient = df['final_score'] > final_score_threshold
        not_avoid = df['dynamic_action'] != 'AVOID'
        is_not_sell_day = ~is_sell_signal
        final_buy_condition = (
            is_score_sufficient &
            not_avoid &
            is_not_sell_day
        )
        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df)
        self._finalize_signals()

    def _get_dynamic_combat_action(self) -> pd.Series:
        """
        【V317.0 核心】动态力学战术矩阵
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)
        
        offense_accel = atomic.get('FORCE_VECTOR_OFFENSE_ACCELERATING', default_series)
        offense_decel = atomic.get('FORCE_VECTOR_OFFENSE_DECELERATING', default_series)
        risk_accel = atomic.get('FORCE_VECTOR_RISK_ACCELERATING', default_series)
        risk_decel = atomic.get('FORCE_VECTOR_RISK_DECELERATING', default_series)

        is_force_attack = offense_accel & risk_decel
        is_avoid = offense_decel & risk_accel
        is_caution = (offense_accel & risk_accel) | (offense_decel & risk_decel)

        actions = pd.Series('HOLD', index=df.index)
        actions.loc[is_force_attack] = 'FORCE_ATTACK'
        actions.loc[is_avoid] = 'AVOID'
        actions.loc[is_caution] = 'PROCEED_WITH_CAUTION'
        
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
        df['risk_penalty_score'] += get_clipped_score('RISK_BEHAVIOR_PANIC_FLEEING_S') * 300
        
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














