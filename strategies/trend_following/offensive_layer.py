# 文件: strategies/trend_following/offensive_layer.py
# 进攻层
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from .utils import get_params_block, get_param_value

class OffensiveLayer:
    def __init__(self, strategy_instance):
        self.strategy = strategy_instance

    def calculate_entry_score(self, trigger_events: Dict) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V339.0 精英制胜版】
        - 核心升级: 实现了“动能催化剂”逻辑。动能信号的分数只有在“纯阵地分”达到
                    `min_positional_score_for_dynamic` 阈值时才会被激活。
        - 核心哲学: 贯彻“精英制胜”原则，确保动能信号只作为锦上添花的催化剂，
                    而非构建决策的核心，大幅提升信号的信噪比。
        """
        # print("        -> [进攻方案评估中心 V339.0 精英制胜版] 启动...")
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        entry_score = pd.Series(0.0, index=df.index)
        score_details_df = pd.DataFrame(index=df.index)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        default_series = pd.Series(False, index=df.index)
        # --- 1. 评估“战法火力” (Composite Scoring) ---
        composite_rules = scoring_params.get('composite_scoring', {}).get('rules', [])
        for rule in composite_rules:
            rule_name = rule.get('name')
            score = rule.get('score', 0)
            required_states = rule.get('all_of', [])
            any_of_states = rule.get('any_of', [])
            forbidden_states = rule.get('none_of', [])
            final_condition = pd.Series(True, index=df.index)
            if required_states:
                for state in required_states:
                    final_condition &= atomic_states.get(state, default_series)
            if any_of_states:
                any_condition = pd.Series(False, index=df.index)
                for state in any_of_states:
                    any_condition |= atomic_states.get(state, default_series)
                final_condition &= any_condition
            if forbidden_states:
                for state in forbidden_states:
                    final_condition &= ~atomic_states.get(state, default_series)
            if final_condition.any():
                entry_score.loc[final_condition] += score
                score_details_df[rule_name] = final_condition * score
        # --- 2. 评估“阵地火力” (Positional Scoring) ---
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {})
        for signal_name, score in positional_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score
        # --- 3. 计算纯粹的“阵地分”，这是后续所有条件判断的核心基石 ---
        valid_pos_cols = [col for col in positional_rules.keys() if col in score_details_df.columns]
        positional_score = score_details_df[valid_pos_cols].sum(axis=1) if valid_pos_cols else pd.Series(0.0, index=df.index)

        # --- 开始：实现“动能催化剂”逻辑 ---
        # --- 4. 评估“动能火力” (Dynamic Scoring)，带前置条件判断 ---
        dynamic_params = scoring_params.get('dynamic_scoring', {})
        dynamic_rules = dynamic_params.get('positive_signals', {})
        # 4.1 读取“安全锁”参数：激活催化剂所需的最低阵地分
        min_pos_score_for_dyn_params = dynamic_params.get('min_positional_score_for_dynamic', {})
        min_pos_score_for_dyn = get_param_value(min_pos_score_for_dyn_params, 300)
        # 4.2 创建前置条件：只有阵地分达标的日子，动能信号才有资格计分
        dynamic_precondition_met = (positional_score >= min_pos_score_for_dyn)
        if dynamic_precondition_met.any():
            # print(f"          -> [动能催化剂] 在 {dynamic_precondition_met.sum()} 天满足阵地分门槛(>={min_pos_score_for_dyn})，动能信号被激活。")
            for signal_name, score in dynamic_rules.items():
                signal_series = atomic_states.get(signal_name, default_series)
                # 核心逻辑：信号必须同时满足自身触发条件 和 阵地分达标的前置条件
                final_dynamic_condition = signal_series & dynamic_precondition_met
                if final_dynamic_condition.any():
                    entry_score.loc[final_dynamic_condition] += score
                    score_details_df[signal_name] = final_dynamic_condition * score
        # --- 5. 评估“阵地优势加速度”火力 (带安全开关的涡轮增压引擎) ---
        p_hybrid = scoring_params.get('positional_acceleration_hybrid_params', {})
        if get_param_value(p_hybrid.get('enabled'), True):
            positional_change = positional_score.diff(1).fillna(0)
            positional_accel = positional_change.diff(1).fillna(0)
            min_base_score = get_param_value(p_hybrid.get('min_base_score'), 400)
            min_score_increase = get_param_value(p_hybrid.get('min_score_increase'), 150)
            multiplier = get_param_value(p_hybrid.get('score_multiplier'), 2.0)
            max_bonus = get_param_value(p_hybrid.get('max_bonus_score'), 800)
            is_base_strong = positional_score.shift(1) >= min_base_score
            is_increase_significant = positional_change >= min_score_increase
            is_accelerating = positional_accel > 0
            launch_condition = is_base_strong & is_increase_significant & is_accelerating
            if launch_condition.any():
                accel_bonus_score = (positional_accel * multiplier).clip(upper=max_bonus)
                final_bonus = accel_bonus_score.where(launch_condition, 0)
                entry_score += final_bonus
                score_details_df['SCORE_POS_ACCEL_HYBRID_BONUS'] = final_bonus
                # print(f"          -> [混合奖励模型] 已为 {launch_condition.sum()} 天满足“三重保险”的加速信号施加了动态奖励分！")
        # --- 6. 评估“触发器火力” ---
        trigger_rules = scoring_params.get('trigger_events', {}).get('scoring', {})
        enhancement_params = scoring_params.get('trigger_enhancement_params', {})
        is_enhancement_enabled = get_param_value(enhancement_params.get('enabled'), False)
        min_positional_score = get_param_value(enhancement_params.get('min_positional_score_for_trigger'), 350)
        precondition_met = (positional_score >= min_positional_score) if is_enhancement_enabled else pd.Series(True, index=df.index)
        for signal_name, score in trigger_rules.items():
            signal_series = trigger_events.get(signal_name, default_series)
            final_trigger_condition = signal_series & precondition_met
            if final_trigger_condition.any():
                entry_score.loc[final_trigger_condition] += score
                score_details_df[signal_name] = final_trigger_condition * score
        # --- 7. 评估“剧本火力” (Playbook Scoring) ---
        playbook_rules = scoring_params.get('playbook_scoring', {})
        if playbook_rules:
            for playbook_name, score in playbook_rules.items():
                playbook_series = self.strategy.playbook_states.get(playbook_name, default_series)
                if playbook_series.any():
                    entry_score.loc[playbook_series] += score
                    score_details_df[playbook_name] = playbook_series * score
        # --- 7.1. 评估“精英剧本火力” (Elite Playbook Scoring) ---
        # 目的: 为情报层合成的最高级别S+和S++剧本赋予决定性的高分。
        elite_playbook_rules = {
            'PLAYBOOK_PRIME_BREAKOUT_S_PLUS_PLUS': 1500, # S++级王牌战法，给予最高分
            'PLAYBOOK_RESONANCE_IGNITION_S': 1200,       # S级主升浪共振，确定性极高
            'PLAYBOOK_BOTTOM_REVERSAL_S': 1000          # S级深度反转，值得重仓
        }
        for playbook_name, score in elite_playbook_rules.items():
            playbook_series = self.strategy.playbook_states.get(playbook_name, default_series)
            if playbook_series.any():
                entry_score.loc[playbook_series] += score
                score_details_df[playbook_name] = playbook_series * score
                print(f"          -> [精英剧本火力] 侦测到王牌剧本 “{playbook_name}”，增加 {score} 分！")
        # --- 7.2. 评估“精英原子信号火力” (Elite Atomic Signal Scoring) ---
        # 目的: 确保所有由底层逻辑共振产生的S级以上机会信号，都能直接贡献决定性分数。
        elite_atomic_rules = {
            # 结构-动态融合信号 (最高级别)
            'OPP_STATIC_DYN_FUSION_IGNITION_S_PLUS': 1500, # 静态-动态融合引爆点
            # 资金流-动态融合信号
            'OPP_FUND_FLOW_STATIC_DYN_CONFLUENCE_S_PLUS': 1300, # 主力控盘下的三维共振
            # 多时间维度共振信号
            'OPP_STRUCTURE_MTF_IGNITION_S_PLUS': 1200, # 战略点火，战术强攻
            'OPP_FUND_FLOW_MTF_DYN_ALIGNMENT_S': 1000, # 多维资金动态协同
            # 静态-多动态协同信号
            'OPP_SQUEEZE_MULTI_CONFIRMED_BREAKOUT_S': 900, # 极致压缩后的协同突破
            'OPP_STATIC_DYN_BREAKTHROUGH_S': 800, # 阵地战协同突破
            # S级机会: 价格超跌 + 多重底部共振信号，是确定性极高的左侧反转信号。
            'OPP_STATIC_LONG_TERM_BOTTOM_REVERSAL_S': 1400,
            # A级机会: 主峰强力吸筹，代表主力在关键成本区不计成本地猛烈抢筹，是后续行情启动的强烈信号。
            'OPP_CHIP_INTENSE_ABSORPTION_A': 470,
            # A级机会: 上方压力锐减，表明突破前的最后障碍正在被快速清除，是极强的突破前兆。
            'CHIP_DYN_PRESSURE_RAPIDLY_DECREASING_A': 490,
            # S+级机会: 战略性协同建仓，筹码锁定且资金持续流入，是最高质量的主力建仓信号。
            'CHIP_FUND_FLOW_ACCUMULATION_STRATEGIC_S_PLUS': 1450,
        }
        for signal_name, score in elite_atomic_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score
                print(f"          -> [精英原子火力] 侦测到王牌信号 “{signal_name}”，增加 {score} 分！")
        # --- 7.3. 评估“行为与结构机会火力” (Behavioral & Structural Opportunity Scoring) ---
        # 目的: 为行为层和结构层识别出的高质量A级机会信号赋予基础阵地分。
        behavioral_opportunity_rules = {
            # A级机会: 主力利用恐慌盘进行洗盘吸筹的经典行为。
            'OPP_BEHAVIOR_WASH_OUT_ACCUMULATION_A': 450,
            # A级机会: 极致压缩后的蓄势待发，是突破前的典型信号。
            'OPP_STATIC_EXTREME_SQUEEZE_ACCUMULATION_A': 460,
        }
        for signal_name, score in behavioral_opportunity_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score

        # --- 7.4. 评估“精英动态与认知信号火力” (Elite Dynamic & Cognitive Signal Scoring) ---
        # 目的: 确保所有由底层力学共振或顶层认知判断产生的高确定性机会信号，都能直接贡献决定性分数。
        elite_dynamic_cognitive_rules = {
            # S级机会: 内外共振·主升确认，内部筹码结构与外部价格表现完美共振，是主升浪最强信号。
            'OPP_DYN_INTERNAL_EXTERNAL_RESONANCE_S': 1100,
            # S级机会: 全周期趋势共振，短中长期力量形成合力，是最健康的上涨形态。
            'OPP_DYN_TREND_RESONANCE_S': 1050,
            # A级机会: 市场引擎点火，价格与资金效率同步加速，上涨健康且高效。
            'OPP_DYN_MARKET_ENGINE_IGNITION_A': 550,
            # A级机会: 长周期底部拐点，下跌动能衰竭且短期趋势反转，是理想的左侧信号。
            'OPP_DYN_LONG_CYCLE_INFLECTION_A': 500,
        }
        for signal_name, score in elite_dynamic_cognitive_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                # 使用 .add() 方法以安全地合并分数，避免覆盖
                current_score = score_details_df.get(signal_name, pd.Series(0.0, index=df.index))
                score_details_df[signal_name] = current_score.add(signal_series * score, fill_value=0)
                entry_score = entry_score.add(signal_series * score, fill_value=0)
                print(f"          -> [精英动态火力] 侦测到王牌信号 “{signal_name}”，增加 {score} 分！")

        # --- 7.5. 评估“经典与复合信号火力” (Classic & Composite Signal Scoring) ---
        # 目的: 确保所有具备高市场共识度的经典信号和高质量的复合信号，都能直接贡献分数。
        classic_composite_rules = {
            # A级机会: 认知层合成的侵略性协同进攻信号，代表多重动能共振且处于安全区。
            'DYN_AGGRESSIVE_OFFENSE_A': 520,
            # A级机会: 放量上涨，量价齐升，是强烈的买盘确认信号。
            'VOL_PRICE_SPIKE_UP_A': 480,
            # B级机会: MACD金叉，经典的短期动能转强信号。
            'OSC_TRIGGER_MACD_GOLDEN_CROSS_B': 250,
            # A级机会: 持续净流入，表明买盘具有连续性，而非一日游的脉冲行为。
            'FUND_FLOW_SUSTAINED_INFLOW_A': 480,
            # A级机会: 高强度净流入，代表买盘意愿坚决，其强度足以主导当日价格趋势。
            'FUND_FLOW_HIGH_INTENSITY_INFLOW_A': 510,
            # A级环境: 市场处于趋势状态，宏观环境有利于趋势跟踪策略，是重要的顺风信号。
            'STRUCTURE_REGIME_TRENDING': 450,
        }
        for signal_name, score in classic_composite_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                current_score = score_details_df.get(signal_name, pd.Series(0.0, index=df.index))
                score_details_df[signal_name] = current_score.add(signal_series * score, fill_value=0)
                entry_score = entry_score.add(signal_series * score, fill_value=0)
                print(f"          -> [经典复合火力] 侦测到信号 “{signal_name}”，增加 {score} 分！")
                
        entry_score, score_details_df = self._apply_contextual_bonus_score(entry_score, score_details_df)
        # --- 8. 评估“周线战略背景”火力 (Strategic Context Bonus) ---
        strategic_bonus_params = scoring_params.get('strategic_context_scoring', {})
        if get_param_value(strategic_bonus_params.get('enabled'), True):
            bullish_bonus = get_param_value(strategic_bonus_params.get('bullish_bonus'), 200)
            is_bullish = atomic_states.get('CONTEXT_STRATEGIC_BULLISH_W', default_series)
            if is_bullish.any():
                entry_score.loc[is_bullish] += bullish_bonus
                score_details_df['STRATEGIC_BULLISH_BONUS_W'] = is_bullish * bullish_bonus
            ignition_bonus = get_param_value(strategic_bonus_params.get('ignition_bonus'), 100)
            is_ignition = atomic_states.get('CONTEXT_STRATEGIC_IGNITION_W', default_series)
            if is_ignition.any():
                entry_score.loc[is_ignition] += ignition_bonus
                score_details_df['STRATEGIC_IGNITION_BONUS_W'] = is_ignition * ignition_bonus
        # --- 9. 评估“日线长周期筹码战略背景”火力 (Daily Long-Term Chip Context Bonus) ---
        # 这是对宏观战局的最终判断。如果大部队正在战略集结，那么任何战术进攻的成功率都会大增。
        chip_context_params = scoring_params.get('chip_context_scoring', {})
        if get_param_value(chip_context_params.get('enabled'), True):
            # 顺风加成：如果处于长达一个季度的战略吸筹期，给予显著加分。
            gathering_bonus = get_param_value(chip_context_params.get('strategic_gathering_bonus'), 300)
            is_gathering = atomic_states.get('CONTEXT_CHIP_STRATEGIC_GATHERING', default_series)
            if is_gathering.any():
                entry_score.loc[is_gathering] += gathering_bonus
                score_details_df['STRATEGIC_GATHERING_BONUS_D'] = is_gathering * gathering_bonus
                print(f"          -> [战略顺风] 侦测到处于“战略吸筹期”，为 {is_gathering.sum()} 天的信号增加 {gathering_bonus} 分！")
        # --- 10. 评估“主升浪黄金航道”背景火力 (Main Uptrend Wave Context Bonus) ---
        # 这是对当前战局的最终判断。如果已确认进入主升浪，那么任何战术进攻的成功率都会大增。
        main_uptrend_params = scoring_params.get('main_uptrend_context_scoring', {})
        if get_param_value(main_uptrend_params.get('enabled'), True):
            # 顺风加成：如果认知层确认当前处于S级主升浪黄金航道，给予显著加分。
            uptrend_bonus = get_param_value(main_uptrend_params.get('uptrend_bonus'), 350)
            is_in_main_uptrend = atomic_states.get('STRUCTURE_MAIN_UPTREND_WAVE_S', default_series)
            if is_in_main_uptrend.any():
                entry_score.loc[is_in_main_uptrend] += uptrend_bonus
                score_details_df['CONTEXT_MAIN_UPTREND_BONUS_S'] = is_in_main_uptrend * uptrend_bonus
                print(f"          -> [战局顺风] 侦测到处于“主升浪黄金航道”，为 {is_in_main_uptrend.sum()} 天的信号增加 {uptrend_bonus} 分！")
        # --- 11. 评估“堡垒式主升浪”背景火力 (Fortress Uptrend Context Bonus) ---
        # 这是最高质量的战局判断。如果已确认进入由S级堡垒结构支撑的主升浪，
        # 意味着主力高度控盘，上涨确定性极高，应给予最高级别的环境加成。
        fortress_uptrend_params = scoring_params.get('fortress_uptrend_context_scoring', {})
        if get_param_value(fortress_uptrend_params.get('enabled'), True):
            # S+级顺风加成：如果认知层确认当前处于堡垒式主升浪，给予决定性加分。
            fortress_bonus = get_param_value(fortress_uptrend_params.get('fortress_bonus'), 500)
            is_in_fortress_uptrend = atomic_states.get('STRUCTURE_FORTRESS_UPTREND_S_PLUS', default_series)
            if is_in_fortress_uptrend.any():
                entry_score.loc[is_in_fortress_uptrend] += fortress_bonus
                score_details_df['CONTEXT_FORTRESS_UPTREND_BONUS_S_PLUS'] = is_in_fortress_uptrend * fortress_bonus
                print(f"          -> [S+战局顺风] 侦测到处于“堡垒式主升浪”，为 {is_in_fortress_uptrend.sum()} 天的信号增加 {fortress_bonus} 分！")
        return entry_score, score_details_df

    def _diagnose_offensive_momentum(self, entry_score: pd.Series, score_details_df: pd.DataFrame) -> pd.Series:
        """
        【V501.1 修复版】进攻动能诊断大脑
        - 核心修复: 修正了 get_param_value 函数的调用错误，确保能正确读取参数。
        - 核心变化: “阵地优势加速”的判断和计分已移至 `calculate_entry_score`。
                    本方法回归其核心职责：诊断【总分】的动态，主要用于生成“机会衰退”等风险类信号。
        """
        print("          -> [进攻动能诊断大脑 V501.1 修复版] 启动，正在诊断总分动态...")
        
        # --- 步骤 1: 计算总分(entry_score)的动态，用于风险控制（机会衰退）---
        score_change = entry_score.diff(1).fillna(0)
        score_accel = score_change.diff(1).fillna(0)
        
        # 状态: 【机会衰退】(否决票来源)
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        is_opportunity_fading = ((score_change > 0) & (score_accel < 0)) | (score_change <= 0)
        
        momentum_params = scoring_params.get('momentum_diagnostics_params', {})
        fading_score_threshold = get_param_value(momentum_params.get('fading_score_threshold'), 500)
        
        self.strategy.atomic_states['SCORE_DYN_OPPORTUNITY_FADING'] = is_opportunity_fading & (entry_score.shift(1) > fading_score_threshold)

        # 状态: 【风险抬头】(否决票来源)
        risk_score = self.strategy.df_indicators.get('risk_score', pd.Series(0.0, index=entry_score.index))
        risk_change = risk_score.diff(1).fillna(0)
        risk_accel = risk_change.diff(1).fillna(0)
        is_risk_escalating = (risk_change > 0) & (risk_accel > 0)
        self.strategy.atomic_states['SCORE_DYN_RISK_ESCALATING'] = is_risk_escalating
        
        # --- 步骤 2: 生成用于调试的详细诊断报告 ---
        diagnostics = pd.Series([{} for _ in range(len(entry_score))], index=entry_score.index)
        # 重新计算阵地分及其变化，仅为生成报告
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {}).keys()
        valid_pos_cols = [col for col in positional_rules if col in score_details_df.columns]
        positional_score = score_details_df[valid_pos_cols].sum(axis=1) if valid_pos_cols else pd.Series(0.0, index=score_details_df.index)
        positional_change = positional_score.diff(1).fillna(0)
        dynamic_rules = scoring_params.get('dynamic_scoring', {}).get('positive_signals', {}).keys()
        valid_dyn_cols = [col for col in dynamic_rules if col in score_details_df.columns]
        dynamic_score = score_details_df[valid_dyn_cols].sum(axis=1) if valid_dyn_cols else pd.Series(0.0, index=score_details_df.index)
        dynamic_change = dynamic_score.diff(1).fillna(0)
        stall_condition = (score_change <= 0) & (entry_score.shift(1) > 0)
        decel_condition = (score_change > 0) & (score_accel < 0)
        base_erosion_condition = (positional_change < 0)
        divergence_condition = (positional_change <= 0) & (dynamic_change > 0) & (entry_score > 0)
        for idx in entry_score.index:
            report = {}
            if stall_condition.at[idx]: report['stall'] = f"进攻停滞(总分变化: {score_change.at[idx]:.0f})"
            if decel_condition.at[idx]: report['deceleration'] = f"进攻减速(加速度: {score_accel.at[idx]:.0f})"
            if base_erosion_condition.at[idx]: report['base_erosion'] = f"阵地侵蚀(阵地分变化: {positional_change.at[idx]:.0f})"
            if divergence_condition.at[idx]: report['divergence'] = "结构性背离(动能分虚高)"
            if report: diagnostics.at[idx] = report
        return diagnostics

    def _apply_contextual_bonus_score(self, entry_score: pd.Series, score_details_df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        【V4.1 接力战法版】战术环境奖励模块
        - 核心改造: 引入“衰减式奖励”模型，奖励在触发日达到峰值，随后线性递减。
                    同时保留了对简单“固定加分”模型的支持，并通过'deprecated'标志位来禁用过时规则。
        """
        # 从配置文件获取上下文奖励的参数块
        bonus_params = get_params_block(self.strategy, 'contextual_bonus_params')
        # 检查该模块是否启用，若未启用则直接返回，不进行任何操作
        if not get_param_value(bonus_params.get('enabled'), False):
            return entry_score, score_details_df
        # 获取所有奖励规则的列表
        bonus_rules = bonus_params.get('bonuses', [])
        # 遍历每一条奖励规则
        for rule in bonus_rules:
            # 从规则中获取核心参数
            state_name = rule.get('if_state')
            bonus_signal_name = rule.get('signal_name')
            # 增加健壮性检查。如果规则中缺少必要的键，则跳过此规则，防止程序因配置错误而崩溃。
            if not (state_name and bonus_signal_name) or rule.get('deprecated', False):
                continue
            condition = self.strategy.atomic_states.get(state_name, pd.Series(False, index=entry_score.index)).shift(1).fillna(False)
            if not condition.any():
                continue
            # 判断此规则是“衰减模型”还是“固定加分模型”
            if rule.get('decay_model', False):
                max_bonus = rule.get('max_bonus_score', 0)
                decay_days = rule.get('decay_days', 1)
                if max_bonus <= 0 or decay_days <= 0 or not condition.any():
                    continue
                # 步骤1: 创建一个临时的Series来存储此规则产生的奖励分
                bonus_series = pd.Series(0.0, index=entry_score.index)
                # 步骤2: 使用辅助函数生成一个0-1的、带衰减效果的影响力序列
                # 这个函数已经内置了处理窗口重叠的逻辑（取最大影响力）
                influence_series = self.strategy.cognitive_intel._create_decaying_influence_series(condition, decay_days)
                # 步骤3: 将影响力序列乘以最大奖励分，得到最终的奖励分数序列
                bonus_series = influence_series * max_bonus
                # 步骤4: 将计算好的奖励分数序列一次性地应用到总分和详情中
                entry_score += bonus_series
                if bonus_signal_name not in score_details_df.columns:
                    score_details_df[bonus_signal_name] = 0.0
                score_details_df[bonus_signal_name] += bonus_series
                # print(f"          -> [衰减奖励] 已为 “{state_name}” 事件应用了峰值为 {max_bonus}，持续 {decay_days} 天的衰减奖励。")
            else:
                # --- 固定加分模型逻辑 ---
                bonus_value = rule.get('add_score', 0)
                # 只有在条件触发且奖励分大于0时才执行
                if condition.any() and bonus_value != 0:
                    # 将固定的奖励分加到总分上
                    entry_score.loc[condition] += bonus_value
                    # 在详情中记录这个加分项
                    score_details_df[bonus_signal_name] = condition * bonus_value
                    # print(f"          -> [环境奖励] 已为 {condition.sum()} 天的“{state_name}”期间应用 {bonus_value} 分固定奖励。")
            
        return entry_score, score_details_df












