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
        【V337.2 武器库升级版】
        - 核心升级: 新增了对 playbook_states 的处理逻辑，激活了“均值回归”等剧本的计分能力。
        - 核心加固: 增加了对“安全开关”逻辑中 foundation_signals 的健壮性检查。
        """
        print("        -> [进攻方案评估中心 V337.2 武器库升级版] 启动...") # MODIFIED: 修改版本号
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

        # --- 2. 评估“阵地/动能火力” (Atomic Scoring) ---
        positional_rules = scoring_params.get('positional_scoring', {}).get('positive_signals', {})
        for signal_name, score in positional_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score
        dynamic_rules = scoring_params.get('dynamic_scoring', {}).get('positive_signals', {})
        for signal_name, score in dynamic_rules.items():
            signal_series = atomic_states.get(signal_name, default_series)
            if signal_series.any():
                entry_score.loc[signal_series] += score
                score_details_df[signal_name] = signal_series * score

        # --- 3. 计算纯粹的“阵地分”用于后续逻辑 ---
        valid_pos_cols = [col for col in positional_rules.keys() if col in score_details_df.columns]
        positional_score = score_details_df[valid_pos_cols].sum(axis=1) if valid_pos_cols else pd.Series(0.0, index=df.index)

        # --- 4. 评估“阵地优势加速度”火力 (带安全开关的涡轮增压引擎) ---
        p_hybrid = scoring_params.get('positional_acceleration_hybrid_params', {})
        if get_param_value(p_hybrid.get('enabled'), True):
            # 4.1 计算动态
            positional_change = positional_score.diff(1).fillna(0)
            positional_accel = positional_change.diff(1).fillna(0)

            # 4.2 获取实战参数 (三重保险 + 奖励参数)
            min_base_score = get_param_value(p_hybrid.get('min_base_score'), 400)
            min_score_increase = get_param_value(p_hybrid.get('min_score_increase'), 150)
            multiplier = get_param_value(p_hybrid.get('score_multiplier'), 2.0)
            max_bonus = get_param_value(p_hybrid.get('max_bonus_score'), 800) # 新增：奖励上限

            # 4.3 应用三重保险过滤器
            is_base_strong = positional_score.shift(1) >= min_base_score
            is_increase_significant = positional_change >= min_score_increase
            is_accelerating = positional_accel > 0
            
            # 最终的“发射许可”条件
            launch_condition = is_base_strong & is_increase_significant & is_accelerating

            if launch_condition.any():
                # 4.4 计算奖励分，并施加“安全上限”
                accel_bonus_score = (positional_accel * multiplier).clip(upper=max_bonus)
                
                # 4.5 将奖励分施加到总分和详情中
                final_bonus = accel_bonus_score.where(launch_condition, 0)
                entry_score += final_bonus
                score_details_df['SCORE_POS_ACCEL_HYBRID_BONUS'] = final_bonus
                print(f"          -> [混合奖励模型] 已为 {launch_condition.sum()} 天满足“三重保险”的加速信号施加了动态奖励分！")

        # --- 5. 评估“触发器火力” ---
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

        # [修改原因] 适配 PlaybookEngine V2.0 的输出。playbook_states 现在是一个纯粹的布尔序列字典。
        # --- 6. 评估“剧本火力” (Playbook Scoring) ---
        playbook_rules = scoring_params.get('playbook_scoring', {})
        if playbook_rules:
            for playbook_name, score in playbook_rules.items():
                playbook_series = self.strategy.playbook_states.get(playbook_name, default_series)
                if playbook_series.any():
                    entry_score.loc[playbook_series] += score
                    score_details_df[playbook_name] = playbook_series * score
        
        entry_score, score_details_df = self._apply_contextual_bonus_score(entry_score, score_details_df)
        return entry_score, score_details_df

    def _get_playbook_blueprints(self) -> List[Dict]:
        """
        【V83.0 新增】剧本蓝图知识库
        - 职责: 定义所有剧本的静态属性（名称、家族、评分规则等）。
        - 特性: 这是一个纯粹的数据结构，不依赖任何动态数据，可以在初始化时被安全地缓存。
        """
        return [
            # --- 反转/逆势家族 (REVERSAL_CONTRARIAN) ---
            {
                'name': 'ABYSS_GAZE_S', 'cn_name': '【S级】深渊凝视', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 320, 'side': 'left', 'comment': 'S级: 市场极度恐慌后的第一个功能性反转。'
            },
            {
                'name': 'CAPITULATION_PIT_REVERSAL', 'cn_name': '【动态】投降坑反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup_score', 'side': 'left', 'comment': '根据坑的深度和反转K强度动态给分。', 'allow_memory': False,
                'setup_score_key': 'SETUP_SCORE_CAPITULATION_PIT',
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 160, 'score_multiplier': 1.0, 'trigger_bonus': {'TRIGGER_REVERSAL_CONFIRMATION_CANDLE': 50}}
            },
            {
                'name': 'CAPITAL_DIVERGENCE_REVERSAL', 'cn_name': '【A-级】资本逆行者', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 230, 'side': 'left', 'comment': 'A-级: 在“底背离”机会窗口中出现的反转确认。'
            },
            {
                'name': 'BEAR_TRAP_RALLY', 'cn_name': '【C+级】熊市反弹', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 180, 'side': 'left', 'comment': 'C+级: 熊市背景下，价格首次钝化，并由长期趋势企稳信号触发。'
            },
            {
                'name': 'WASHOUT_REVERSAL_A', 'cn_name': '【A级】巨阴洗盘反转', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 260, 'side': 'left', 'comment': 'A级: 极端洗盘后的拉升前兆。', 'allow_memory': False
            },
            {
                'name': 'BOTTOM_STABILIZATION_B', 'cn_name': '【B级】底部企稳', 'family': 'REVERSAL_CONTRARIAN',
                'type': 'setup', 'score': 190, 'side': 'left', 'comment': 'B级: 股价严重超卖偏离均线后，出现企稳阳线。'
            },
            # --- 趋势/动能家族 (TREND_MOMENTUM) ---
            {
                'name': 'PLAYBOOK_BREAKOUT_PULLBACK_RELAY_S_PLUS', 
                'cn_name': '【S+级】突破-回踩接力', 
                'family': 'TREND_MOMENTUM',
                'type': 'setup', 
                'score': 800, # 给予一个压倒性的高分
                'side': 'right', 
                'comment': 'S+级: 初升浪启动后，首次出现健康的、由筹码支撑的回踩。这是最高置信度的趋势确认和介入信号。'
            },
            {
                'name': 'HEALTHY_PULLBACK_S', 
                'cn_name': '【S级】健康回踩吸筹', 
                'family': 'TREND_MOMENTUM',
                'type': 'setup', 
                'score': 450, # 给予一个非常高的分数
                'side': 'right', 
                'comment': 'S级: 在主升浪中，股价回踩但筹码结构稳固，是极佳的加仓或建仓机会。'
            },
            {
                'name': 'CHIP_PLATFORM_PULLBACK', 'cn_name': '【S-级】筹码平台回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 330, 'side': 'right', 
                'comment': 'S-级: 股价回踩由筹码峰形成的、位于趋势线上方的稳固平台，并获得支撑。这是极高质量的“空中加油”信号。'
            },
            {
                'name': 'TREND_EMERGENCE_B_PLUS', 'cn_name': '【B+级】右侧萌芽', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 210, 'comment': 'B+级: 填补战术真空。在左侧反转后，短期均线走好但尚未站上长线时的首次介入机会。'
            },
            {
                'name': 'DEEP_ACCUMULATION_BREAKOUT', 'cn_name': '【动态】潜龙出海', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': '根据深度吸筹分数动态给分。若伴随筹码点火，则确定性更高。',
                'setup_score_key': 'SETUP_SCORE_DEEP_ACCUMULATION',
                'scoring_rules': { 'min_setup_score_to_trigger': 51, 'base_score': 200, 'score_multiplier': 1.5, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'ENERGY_COMPRESSION_BREAKOUT', 
                'cn_name': '【动态】能量压缩突破', 
                'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 
                'side': 'right', 
                'comment': '根据当天前提和历史准备共同给分。若伴随筹码点火或力学确认，则确定性更高。',
                # 确认条件：要求突破必须发生在“惯量减小”（筹码锁定）的背景下
                'confirmation_states': ['MECHANICS_INERTIA_DECREASING'], 
                'scoring_rules': { 
                    'base_score': 150, 
                    'min_score_to_trigger': 180, 
                    'conditions': {'VOL_STATE_SQUEEZE_WINDOW': 50, 'CHIP_STATE_CONCENTRATION_SQUEEZE': 30, 'MA_STATE_CONVERGING': 20}, 
                    'setup_bonus': {'ENERGY_COMPRESSION': 0.2, 'HEALTHY_MARKUP': 0.1}, 
                    'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} 
                }
            },
            {
                'name': 'PLATFORM_SUPPORT_PULLBACK', 
                'cn_name': '【动态】多维支撑回踩', 
                'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 
                'side': 'right', 
                'comment': '根据平台质量(筹码结构、趋势背景)和支撑级别(均线/筹码峰)进行动态评分。',
                'setup_score_key': 'SETUP_SCORE_PLATFORM_QUALITY',
                'scoring_rules': {
                    'min_setup_score_to_trigger': 50,  # 要求平台质量分至少达到50
                    'base_score': 180,                 # 给予一个稳健的基础分
                    'score_multiplier': 1.2            # 允许根据平台质量分进行加成
                }
            },
            {
                'name': 'HEALTHY_BOX_BREAKOUT', 
                'cn_name': '【A-级】健康箱体突破', 
                'family': 'TREND_MOMENTUM',
                'type': 'setup', 
                'score': 220, 
                'side': 'right', 
                'comment': 'A-级: 在一个位于趋势均线上方的健康箱体内盘整后，发生的向上突破。',
                # 确认条件：要求箱体突破必须得到“趋势正在形成”的分形确认
                'confirmation_states': ['FRACTAL_EVENT_TREND_FORMING']
            },
            {
                'name': 'HEALTHY_MARKUP_A', 'cn_name': '【A级】健康主升浪', 'family': 'TREND_MOMENTUM',
                'type': 'setup_score', 'side': 'right', 'comment': 'A级: 在主升浪中回踩或延续，根据主升浪健康分动态加成。',
                'setup_score_key': 'SETUP_SCORE_HEALTHY_MARKUP',
                'scoring_rules': { 
                    'min_setup_score_to_trigger': 60, # 要求主升浪健康分至少达到60
                    'base_score': 240, 
                    'score_multiplier': 1.2, # 允许根据健康分进行加成
                    'conditions': { 'MA_STATE_DIVERGING': 20, 'OSC_STATE_MACD_BULLISH': 15, 'MA_STATE_PRICE_ABOVE_SHORT_MA': 0 } 
                }
            },
            {
                'name': 'N_SHAPE_CONTINUATION_A', 'cn_name': '【A级】N字板接力', 'family': 'TREND_MOMENTUM',
                'type': 'precondition_score', 'side': 'right', 'comment': 'A级: 强势股的经典趋势中继信号。若伴随筹码点火，则确定性更高。',
                'scoring_rules': { 'base_score': 250, 'min_score_to_trigger': 250, 'conditions': {'SETUP_SCORE_N_SHAPE_CONTINUATION_ABOVE_80': 0}, 'trigger_bonus': {'CHIP_EVENT_IGNITION': 60} }
            },
            {
                'name': 'GAP_SUPPORT_PULLBACK_B_PLUS', 'cn_name': '【B+级】缺口支撑回踩', 'family': 'TREND_MOMENTUM',
                'type': 'setup', 'score': 190, 'side': 'right', 'comment': 'B+级: 回踩前期跳空缺口获得支撑并反弹。'
            },
            # --- 特殊事件家族 (SPECIAL_EVENT) ---
            {
                'name': 'EARTH_HEAVEN_BOARD', 'cn_name': '【S+】地天板', 'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 'score': 380, 'side': 'left', 'comment': '市场情绪的极致反转。'
            },
            {
                'name': 'FAULT_REBIRTH_S', 
                'cn_name': '【S级】断层新生', 
                'family': 'SPECIAL_EVENT',
                'type': 'event_driven', 
                'score': 350, 
                'side': 'left', 
                'comment': 'S级: 识别因成本断层导致的筹码结构重置，是极高价值的特殊事件信号。'
            },
            # --- 均值回归家族 (MEAN_REVERSION) ---
            {
                'name': 'MEAN_REVERSION_BOUNCE_A', 
                'cn_name': '【A级】恐慌坑反弹', 
                'family': 'MEAN_REVERSION',
                'type': 'setup', 
                'score': 280, # 给予一个较高的分数
                'side': 'left', # 这是一个典型的左侧交易信号
                'comment': 'A级: 在统计学超卖、卖盘衰竭和多头反击的共振点入场。'
            },
        ]

    def _create_persistent_state_from_events(self, entry_event_series: pd.Series, persistence_days: int, break_condition_series: pd.Series) -> pd.Series:
        """
        【新增工具函数 V401.1】创建一个基于事件的持续性状态。
        这是将理想化数学模型改造为实战策略的关键工具。
        - 状态一旦由 entry_event_series 触发，将持续 persistence_days 天。
        - 如果在持续期间 break_condition_series 为 True，则状态立即中断。
        - 如果在状态持续期间，新的 entry_event 发生，则重置持续天数。
        - 采用循环实现，确保逻辑清晰和准确，对于单只股票的回测性能足够。
        
        Args:
            entry_event_series (pd.Series): 触发状态开始的布尔序列 (瞬时事件)。
            persistence_days (int): 状态希望持续的天数。
            break_condition_series (pd.Series): 立即中断状态的布尔序列。

        Returns:
            pd.Series: 代表持续状态的布尔序列。
        """
        in_state = False
        days_left = 0
        output_state = pd.Series(False, index=entry_event_series.index)
        
        # 为了效率，将Series转为numpy array进行迭代
        entry_events = entry_event_series.to_numpy()
        break_conditions = break_condition_series.to_numpy()
        output_array = output_state.to_numpy()

        for i in range(len(entry_events)):
            # 步骤 1: 检查中断条件，如果满足，立即退出状态
            if in_state and break_conditions[i]:
                in_state = False
                days_left = 0
                
            # 步骤 2: 检查新的入口事件，如果触发，则进入/重置状态
            if entry_events[i]:
                in_state = True
                days_left = persistence_days
                
            # 步骤 3: 维持状态
            if in_state and days_left > 0:
                output_array[i] = True
                days_left -= 1
            else:
                # 如果持续时间结束，也退出状态
                in_state = False
                
        return pd.Series(output_array, index=entry_event_series.index)

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
        
        # --- 【代码修改】修复 get_param_value 参数数量错误 ---
        # [错误原因] 原代码向 get_param_value 传入了3个参数，导致 TypeError。
        # [修复逻辑] 正确的调用方式是两步：1. 先获取上一级的参数字典。 2. 再从该字典中获取目标参数，并将其和默认值传给 get_param_value。
        momentum_params = scoring_params.get('momentum_diagnostics_params', {})
        fading_score_threshold = get_param_value(momentum_params.get('fading_score_threshold'), 500)
        # --- 【代码修改】结束 ---
        
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
            
            # [新增代码] 增加健壮性检查。如果规则中缺少必要的键，则跳过此规则，防止程序因配置错误而崩溃。
            if not (state_name and bonus_signal_name):
                continue

            # [新增代码] 检查规则是否已被标记为“已废弃”。如果是，则跳过此规则。
            # 这使得我们可以在不删除旧配置的情况下，安全地禁用它，便于未来复查。
            if rule.get('deprecated', False):
                continue

            # 从原子状态池中获取触发条件序列
            condition = self.strategy.atomic_states.get(state_name, pd.Series(False, index=entry_score.index))
            
            # 判断此规则是“衰减模型”还是“固定加分模型”
            if rule.get('decay_model', False):
                # --- 衰减奖励模型逻辑 ---
                max_bonus = rule.get('max_bonus_score', 0)
                decay_days = rule.get('decay_days', 1)
                
                # 确保衰减参数有效
                if max_bonus <= 0 or decay_days <= 0:
                    continue

                # 找到所有触发事件的索引位置，这是衰减的起点
                trigger_indices = np.where(condition)[0]
                
                # 遍历每一个触发点
                for start_idx in trigger_indices:
                    # 计算每日的衰减量
                    daily_decay = max_bonus / decay_days
                    # 从触发点开始，向后应用衰减的奖励
                    for i in range(decay_days):
                        current_idx = start_idx + i
                        # 防止索引越界
                        if current_idx >= len(entry_score):
                            break
                        
                        # 计算当天的奖励分数（线性衰减）
                        current_bonus = max_bonus - (i * daily_decay)
                        
                        # 获取当前日期索引
                        current_date = entry_score.index[current_idx]
                        # 将计算出的奖励分加到总分上
                        entry_score.at[current_date] += current_bonus
                        
                        # 在分数详情中记录这个加分项
                        # 如果是第一次记录，需要先初始化该列
                        if bonus_signal_name not in score_details_df.columns:
                            score_details_df[bonus_signal_name] = 0.0
                        score_details_df.at[current_date, bonus_signal_name] += current_bonus
                
                if len(trigger_indices) > 0:
                    print(f"          -> [衰减奖励] 已为 {len(trigger_indices)} 次“{state_name}”事件应用了峰值为 {max_bonus}，持续 {decay_days} 天的衰减奖励。")

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
            
        return entry_score, score_details_df












