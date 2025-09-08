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
        【V501.1 中文日志版】
        - 核心变化: 移除了对特定“加速状态”的硬性过滤。由于“阵地优势加速”已作为
                    核心奖励分融入 entry_score，本层只需根据最终的净得分进行决策即可。
        - 新增功能: 调用辅助函数生成人类可读的信号详情，并存入 'signal_details_cn' 列。
        """
        # print("    --- [最高作战指挥部 V501.1 中文日志版] 启动... ---") # 代码修改: 更新版本号
        df = self.strategy.df_indicators
        
        df['final_score'] = 0.0
        df['signal_type'] = '无信号'
        df['veto_votes'] = 0
        df['dynamic_action'] = 'HOLD'
        self._evaluate_holding_health(score_details_df, risk_details_df)
        self._calculate_static_veto_votes()
        df['dynamic_action'] = self._get_dynamic_combat_action()
        exit_triggers = self._generate_exit_triggers()
        is_sell_signal = exit_triggers.any(axis=1)
        self.strategy.exit_triggers = exit_triggers
        df.loc[is_sell_signal, 'signal_type'] = '卖出信号'

        # --- 买入决策核心逻辑 ---
        p_judge = get_params_block(self.strategy, 'four_layer_scoring_params').get('judgment_params', {})
        net_score_threshold_no_veto = get_param_value(p_judge.get('net_score_threshold_no_veto'), 500)
        net_score_threshold_with_veto = get_param_value(p_judge.get('net_score_threshold_with_veto'), 800)
        df['net_score'] = df['entry_score'] - df['risk_score']
        no_veto_buy_condition = (df['veto_votes'] == 0) & (df['net_score'] > net_score_threshold_no_veto)
        with_veto_buy_condition = (df['veto_votes'] > 0) & (df['net_score'] > net_score_threshold_with_veto)
        is_net_score_sufficient = no_veto_buy_condition | with_veto_buy_condition
        not_avoid = df['dynamic_action'] != 'AVOID'
        is_not_sell_day = ~is_sell_signal

        final_buy_condition = (
            is_net_score_sufficient &
            not_avoid &
            is_not_sell_day
        )

        df.loc[final_buy_condition, 'signal_type'] = '买入信号'
        
        # --- 代码新增: 生成并存储中文信号详情 ---
        df['signal_details_cn'] = self._get_human_readable_summary(score_details_df, risk_details_df)
        # --- 代码新增结束 ---

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

    def _calculate_static_veto_votes(self):
        """
        【V318.4 风险融合版】
        """
        df = self.strategy.df_indicators
        atomic = self.strategy.atomic_states
        default_series = pd.Series(False, index=df.index)

        # 风险1: 筹码结构严重失效 (3票) - 直接使用新的融合信号
        has_critical_chip_risk = atomic.get('RISK_CHIP_STRUCTURE_CRITICAL_FAILURE', default_series)
        df.loc[has_critical_chip_risk, 'veto_votes'] += 3
        
        # 风险3: 绝对否决信号 (2票) - 这里的逻辑可以保持，因为它处理的是更具体的、可配置的否决项
        veto_params = get_params_block(self.strategy, 'absolute_veto_params')
        if get_param_value(veto_params.get('enabled'), True):
            mitigation_rules = get_param_value(veto_params.get('mitigation_rules'), {})
            veto_signals = get_param_value(veto_params.get('veto_signals'), [])
            final_absolute_veto = pd.Series(False, index=df.index)
            for signal_name in veto_signals:
                has_risk = atomic.get(signal_name, default_series)
                if signal_name in mitigation_rules:
                    mitigators = mitigation_rules[signal_name].get('mitigated_by', [])
                    has_mitigator = pd.Series(False, index=df.index)
                    for m_signal in mitigators: has_mitigator |= atomic.get(m_signal, default_series)
                    final_absolute_veto |= (has_risk & ~has_mitigator)
                else:
                    final_absolute_veto |= has_risk
            df.loc[final_absolute_veto, 'veto_votes'] += 2

        # 风险4: 风险分高于进攻分 (1票)
        risk_overrides_entry = df['risk_score'] > df['entry_score']
        is_in_ascent_phase = atomic.get('STRUCTURE_POST_ACCUMULATION_ASCENT_C', default_series)
        df.loc[risk_overrides_entry & ~is_in_ascent_phase, 'veto_votes'] += 1
        
        # 风险5: 核心原子风险信号 (1票)
        # CHIP_DYN_COST_FALLING (成本松动) 信号回测显示有约30%的规避成功率，是重要的预警信号。
        has_cost_falling_risk = atomic.get('CHIP_DYN_COST_FALLING', default_series)
        df.loc[has_cost_falling_risk, 'veto_votes'] += 1
        # 根据最新战报，将规避成功率高达28%的“获利盘崩盘”信号也加入否决票体系。
        has_winner_collapsing_risk = atomic.get('CHIP_DYN_WINNER_RATE_COLLAPSING', default_series)
        df.loc[has_winner_collapsing_risk, 'veto_votes'] += 1
        
        # 风险6: 周线战略顶层风险 (Strategic Veto)
        # 这是最高级别的风险，拥有强大的否决权
        # 6.1 周线发出“顶部区域”强风险信号，投3票 (强否决)
        is_strategic_topping = atomic.get('CONTEXT_STRATEGIC_TOPPING_RISK_W', default_series)
        df.loc[is_strategic_topping, 'veto_votes'] += 3
        
        # 6.2 周线处于“战略看跌”状态，投1票 (软否决)
        is_strategic_bearish = atomic.get('CONTEXT_STRATEGIC_BEARISH_W', default_series)
        df.loc[is_strategic_bearish, 'veto_votes'] += 1
        
        # 风险7: 战略级筹码长期发散 (Strategic Chip Divergence)
        # 如果日线长周期筹码集中度持续发散，即使短期有买入信号，也应谨慎。
        is_long_term_chip_diverging = atomic.get('CONTEXT_CHIP_LONG_TERM_DIVERGENCE_D', default_series)
        df.loc[is_long_term_chip_diverging, 'veto_votes'] += 2 # 给予2票否决，因为长期筹码发散是严重风险

        # 风险8: 战略级筹码长期发散加速 (Strategic Chip Accelerated Divergence)
        # 长期发散且加速，是更严重的风险。
        is_strategic_chip_accel_diverging = atomic.get('CONTEXT_CHIP_LONG_TERM_ACCEL_DIVERGENCE_D', default_series)
        df.loc[is_strategic_chip_accel_diverging, 'veto_votes'] += 3 # 给予3票否决，最高级别风险
        
        # 风险9: 高级S级风险信号 (Advanced S-Grade Risks)
        # 9.1 周线与日线RSI顶背离，是经典的顶部信号，给予2票否决
        has_mtf_divergence = atomic.get('RISK_MTF_RSI_BEARISH_DIVERGENCE_S', default_series)
        df.loc[has_mtf_divergence, 'veto_votes'] += 2

        # 9.2 获利盘恐慌加速出逃，是市场情绪崩溃的强烈信号，给予3票强否决
        has_panic_fleeing = atomic.get('RISK_BEHAVIOR_PANIC_FLEEING_S', default_series)
        df.loc[has_panic_fleeing, 'veto_votes'] += 3
        
        # --- 风险10: 更多S级陷阱信号 (More S-Grade Trap Signals) ---
        # 10.1 静态-动态融合崩塌信号，是结构性风险的强烈预警，给予3票强否决
        has_static_dyn_collapse = atomic.get('RISK_STATIC_DYN_COLLAPSE_S', default_series)
        df.loc[has_static_dyn_collapse, 'veto_votes'] += 3

        # 10.2 主力缺席的诱多式拉升，是典型的出货陷阱，给予3票强否决
        has_deceptive_rally = atomic.get('RISK_FUND_FLOW_DECEPTIVE_RALLY_S_PLUS', default_series)
        df.loc[has_deceptive_rally, 'veto_votes'] += 3
        
        # --- 风险11: 顶层战略筹码风险 (Top-Level Strategic Chip Risk) ---
        # 这是基于最长周期（55日）的宏观判断，拥有极高的否决优先级。
        # 11.1 宏观背景风险：如果股票处于长达一个季度的“战略派发期”，这是一个非常危险的宏观背景，给予2票否决。
        is_strategic_distribution = atomic.get('CONTEXT_CHIP_STRATEGIC_DISTRIBUTION', default_series)
        df.loc[is_strategic_distribution, 'veto_votes'] += 2

        # 11.2 宏观陷阱确认：如果在“战略派发期”这个恶劣天气下，还出现了“拉升”这种看似利好的行为，
        # 这就是我们刚刚在认知层合成的S级“诱多陷阱”，必须给予最强的3票否决。
        has_strategic_trap = atomic.get('RISK_STRATEGIC_DISTRIBUTION_RALLY_TRAP_S', default_series)
        df.loc[has_strategic_trap, 'veto_votes'] += 3
        
        # --- 风险12: 高级行为与结构陷阱 (Advanced Behavioral & Structural Traps) ---
        # 12.1 高位多重背离派发，是强烈的顶部信号，给予3票强否决。
        has_high_altitude_divergence = atomic.get('RISK_STATIC_HIGH_ALTITUDE_MULTI_DIVERGENCE_S', default_series)
        df.loc[has_high_altitude_divergence, 'veto_votes'] += 3

        # 12.2 长期派发背景下的诱多拉升，是典型的出货陷阱，给予3票强否决。
        has_deceptive_rally_long_term = atomic.get('RISK_BEHAVIOR_DECEPTIVE_RALLY_LONG_TERM_S', default_series)
        df.loc[has_deceptive_rally_long_term, 'veto_votes'] += 3

        # --- 风险13: 新增力学层与认知层S级风险否决票 ---
        # 13.1 市场引擎失速，是上涨动能终结的强烈信号，给予3票强否决。
        has_engine_stalling = atomic.get('RISK_DYN_MARKET_ENGINE_STALLING_S', default_series)
        df.loc[has_engine_stalling, 'veto_votes'] += 3

        # 13.2 结构性衰竭反弹，是典型的诱多陷阱，给予3票强否决。
        has_structural_weakness_rally = atomic.get('RISK_DYN_STRUCTURAL_WEAKNESS_RALLY_S', default_series)
        df.loc[has_structural_weakness_rally, 'veto_votes'] += 3

        # 13.3 认知层合成的顶部危险结构信号，代表多重风险共振，给予3票强否决。
        default_score = pd.Series(0.0, index=df.index) # 新增一个默认分数Series
        topping_danger_score = atomic.get('SCORE_STRUCTURE_TOPPING_DANGER_S', default_score)
        has_topping_danger = topping_danger_score > 0.6 # 使用与原布尔信号相同的阈值
        df.loc[has_topping_danger, 'veto_votes'] += 3
        
        # --- 风险14: 新增基础层与结构层风险否决票 ---
        # 14.1 放量杀跌，是恐慌或主力出货的直接体现，给予3票强否决。
        has_volume_spike_down = atomic.get('RISK_VOL_PRICE_SPIKE_DOWN_A', default_series)
        df.loc[has_volume_spike_down, 'veto_votes'] += 3

        # 14.2 处于下跌通道，是绝对的逆风环境，给予3票强否决。
        is_in_bearish_channel = atomic.get('STRUCTURE_BEARISH_CHANNEL_F', default_series)
        df.loc[is_in_bearish_channel, 'veto_votes'] += 3

        # 14.3 MACD死叉，经典的短期动能转弱信号，给予1票软否决。
        has_macd_death_cross = atomic.get('RISK_TRIGGER_MACD_DEATH_CROSS_B', default_series)
        df.loc[has_macd_death_cross, 'veto_votes'] += 1

        # --- 风险15: 新增筹码结构瓦解风险否决票 ---
        # 15.1 堡垒内部瓦解，主力高控盘成为派发的掩护，风险极高，给予3票强否决。
        has_fortress_collapse = atomic.get('SCENARIO_FORTRESS_INTERNAL_COLLAPSE_A', default_series)
        df.loc[has_fortress_collapse, 'veto_votes'] += 3

        # 15.2 主峰高位派发嫌疑，是典型的主力出货行为，给予3票强否决。
        has_peak_distribution = atomic.get('RISK_PEAK_BATTLE_DISTRIBUTION_A', default_series)
        df.loc[has_peak_distribution, 'veto_votes'] += 3

        # --- 风险16: 新增资金流结构风险否决票 ---
        # 16.1 散户狂热风险，上涨由散户情绪驱动，根基不稳，是潜在的顶部信号，给予2票否决。
        has_retail_fomo = atomic.get('RISK_FUND_FLOW_RETAIL_FOMO_B', default_series)
        df.loc[has_retail_fomo, 'veto_votes'] += 2

        # --- 风险17: 新增结构性风险与市场状态否决票 ---
        # 17.1 结构性长期超涨，股价严重偏离均线，结构不稳定，给予2票否决。
        has_overextended = atomic.get('RISK_STRUCTURE_OVEREXTENDED_LONG_TERM_S', default_series)
        df.loc[has_overextended, 'veto_votes'] += 2
        # 17.2 多维共振超涨，日线周线同时超涨，是极强的顶部风险信号，给予3票强否决。
        has_mtf_overextended = atomic.get('RISK_STRUCTURE_MTF_OVEREXTENDED_RESONANCE_S', default_series)
        df.loc[has_mtf_overextended, 'veto_votes'] += 3
        # 17.3 市场处于均值回归状态，与趋势跟踪策略的根本逻辑相悖，给予2票否决。
        is_mean_reverting = atomic.get('STRUCTURE_REGIME_MEAN_REVERTING', default_series)
        df.loc[is_mean_reverting, 'veto_votes'] += 2

    def _finalize_signals(self):
        """
        【V404.1 健壮性修复版】
        - 核心修复: 修复了当没有任何买卖信号时，'signal_entry'列不存在导致的AttributeError。
        """
        df = self.strategy.df_indicators
        
        df['signal_entry'] = False
        df['exit_signal_code'] = 0
        df['exit_severity_level'] = 0
        df['alert_reason'] = ''
        
        final_buy_condition = df['signal_type'] == '买入信号'
        final_sell_condition = df['signal_type'] == '卖出信号'
        final_warning_condition = df['signal_type'] == '风险预警'

        df.loc[final_buy_condition, 'final_score'] = df.loc[final_buy_condition, 'entry_score']
        df.loc[final_buy_condition, 'signal_entry'] = True
        
        if 'exit_signal_code' in df.columns:
            if 'exit_severity_level' not in df.columns: df['exit_severity_level'] = 0
            if 'alert_reason' not in df.columns: df['alert_reason'] = ''
            df.loc[final_buy_condition, ['exit_signal_code', 'exit_severity_level', 'alert_reason']] = [0, 0, '']

        df.loc[final_sell_condition | final_warning_condition, 'final_score'] = df.loc[final_sell_condition | final_warning_condition, 'risk_score']
        
        # debug_cols = ['entry_score', 'risk_score', 'veto_votes', 'net_score', 'final_score', 'signal_type', 'main_force_state']
        # final_check_df = df[(df['signal_type'] != '无信号') & (df['signal_type'] != '中性')].tail(10)
        # if not final_check_df.empty:
        #     cols_to_show = [col for col in debug_cols if col in final_check_df.columns]
        #     print("          -> [最终分数审查报告]:")
        #     print(final_check_df[cols_to_show])
        # else:
        #     print("          -> [最终分数审查报告]: 在最近的记录中未发现任何有效信号。")
