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
            print(f"          -> [动能催化剂] 在 {dynamic_precondition_met.sum()} 天满足阵地分门槛(>={min_pos_score_for_dyn})，动能信号被激活。")
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
                print(f"          -> [衰减奖励] 已为 “{state_name}” 事件应用了峰值为 {max_bonus}，持续 {decay_days} 天的衰减奖励。")
            else:
                # --- 固定加分模型逻辑 ---
                bonus_value = rule.get('add_score', 0)
                # 只有在条件触发且奖励分大于0时才执行
                if condition.any() and bonus_value > 0:
                    # 将固定的奖励分加到总分上
                    entry_score.loc[condition] += bonus_value
                    # 在详情中记录这个加分项
                    score_details_df[bonus_signal_name] = condition * bonus_value
                    print(f"          -> [环境奖励] 已为 {condition.sum()} 天的“{state_name}”期间应用 {bonus_value} 分固定奖励。")
                # 显式处理固定加分模型
                fixed_bonus = rule.get('add_score', 0)
                if fixed_bonus > 0:
                    # 奖励只施加在满足前置条件后的“当天”
                    bonus_series = condition * fixed_bonus
                    entry_score += bonus_series
                    score_details_df[bonus_signal_name] = bonus_series
            
        return entry_score, score_details_df












