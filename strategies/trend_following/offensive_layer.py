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
        【V401.0 三位一体版】进攻方案评估中心
        - 核心重构: 废除旧的混合计分模型，全面升级为“三位一体”计分体系，将“战备”、“触发”和“协同”分离。
        - 核心逻辑:
          1.  【战备分】: 对高质量的元融合信号和战备状态进行独立计分，构筑开仓的“地基”。
          2.  【触发器分】: 对强力的、独立的触发事件本身进行加分，作为点火的“催化剂”。
          3.  【协同奖励分】: 当“战备”与“触发”完美结合（即剧本成功执行）时，给予额外的“协同奖励”。
          4.  【总分合成】: 最终进攻分 = 战备分 + 触发器分 + 协同奖励分 + 其他加成项。
        - 收益: 评分体系更透明、可解释，且能更早地发现“万事俱备，只欠东风”的潜在机会。
        """
        # print("        -> [进攻方案评估中心 V401.0 三位一体版] 启动...")
        df = self.strategy.df_indicators
        atomic_states = self.strategy.atomic_states
        score_details_df = pd.DataFrame(index=df.index)
        
        # --- 1. 加载三位一体计分模型参数 ---
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        if not get_param_value(scoring_params.get('enabled'), True):
            return pd.Series(0.0, index=df.index), score_details_df
        
        context_params = scoring_params.get('contextual_setup_scoring', {})
        trigger_params = scoring_params.get('trigger_event_scoring', {})
        playbook_params = scoring_params.get('playbook_synergy_scoring', {})
            
        # --- 2. 计算【第一层：环境与战备分】(Contextual & Setup Score) ---
        context_score = pd.Series(0.0, index=df.index)
        if get_param_value(context_params.get('enabled'), True):
            for signal_name, score in context_params.get('positive_signals', {}).items():
                signal_series = atomic_states.get(signal_name, pd.Series(False, index=df.index))
                if signal_series.any():
                    bonus_amount = signal_series.astype(float) * score
                    context_score += bonus_amount
                    score_details_df[f"SETUP_{signal_name}"] = bonus_amount
        score_details_df['SCORE_SETUP'] = context_score
        print(f"          -> [第一层] 环境与战备分计算完毕，基础分峰值: {context_score.max():.0f}")

        # --- 3. 计算【第二层：独立触发器加分】(Trigger Event Score) ---
        trigger_score = pd.Series(0.0, index=df.index)
        if get_param_value(trigger_params.get('enabled'), True):
            for trigger_name, score in trigger_params.get('positive_signals', {}).items():
                trigger_series = trigger_events.get(trigger_name, pd.Series(False, index=df.index))
                if trigger_series.any():
                    bonus_amount = trigger_series.astype(float) * score
                    trigger_score += bonus_amount
                    score_details_df[f"TRIGGER_{trigger_name}"] = bonus_amount
        score_details_df['SCORE_TRIGGER'] = trigger_score
        print(f"          -> [第二层] 独立触发器分计算完毕，加分峰值: {trigger_score.max():.0f}")

        # --- 4. 计算【第三层：剧本协同奖励分】(Playbook Synergy Score) ---
        playbook_score = pd.Series(0.0, index=df.index)
        if get_param_value(playbook_params.get('enabled'), True):
            for playbook_name, score in playbook_params.get('positive_signals', {}).items():
                playbook_series = self.strategy.playbook_states.get(playbook_name, pd.Series(False, index=df.index))
                if playbook_series.any():
                    bonus_amount = playbook_series.astype(float) * score
                    playbook_score += bonus_amount
                    score_details_df[f"PLAYBOOK_{playbook_name}"] = bonus_amount
        score_details_df['SCORE_PLAYBOOK_SYNERGY'] = playbook_score
        print(f"          -> [第三层] 剧本协同奖励分计算完毕，奖励峰值: {playbook_score.max():.0f}")

        # --- 5. 合成总进攻分 (Total Entry Score) ---
        # 总分 = 战备分 + 触发器分 + 协同奖励分
        entry_score = context_score + trigger_score + playbook_score
        
        # --- 6. [保留逻辑] 应用其他独立的加分模块 (如动能分、环境加成等) ---
        # 注意：这些模块现在是在三位一体总分的基础上进行加成
        
        # 6.1 应用“上下文环境”奖励分 (调用独立的奖励模块)
        entry_score, score_details_df = self._apply_contextual_bonus_score(entry_score, score_details_df)
        
        # 6.2 应用“动态动能”加分
        dynamic_params = scoring_params.get('dynamic_scoring', {})
        if get_param_value(dynamic_params.get('enabled'), True):
            min_setup_score = get_param_value(dynamic_params.get('min_positional_score_for_dynamic'), 300)
            # 只有在地基分（战备分）达标时，才激活后续的动能加分
            can_add_dynamic_score = context_score >= min_setup_score
            for signal_name, score in dynamic_params.get('positive_signals', {}).items():
                signal_series = atomic_states.get(signal_name, pd.Series(False, index=df.index))
                if signal_series.any():
                    bonus_amount = (signal_series & can_add_dynamic_score).astype(float) * score
                    entry_score += bonus_amount
                    score_details_df[f"DYN_{signal_name}"] = bonus_amount

        print(f"        -> [进攻方案评估中心] 最终合成完毕，总进攻分峰值: {entry_score.max():.0f}")
        return entry_score, score_details_df

    def _diagnose_offensive_momentum(self, entry_score: pd.Series, score_details_df: pd.DataFrame) -> pd.Series:
        """
        【V401.0 三位一体适配版】进攻动能诊断大脑
        - 核心重构: 适配全新的三位一体评分体系。
        - 核心职责: 诊断【总分】和【核心维度】的动态变化，生成“机会衰退”和“结构性背离”等风险信号。
        - 核心逻辑:
          1.  基于总分(entry_score)的斜率和加速度，判断整体进攻动能是增强、减速还是停滞。
          2.  基于核心维度(现在是'SCORE_SETUP')的分数变化，与总分变化进行交叉验证，识别“结构性背离”风险
              (例如，核心战备分下降，但次要的触发器或协同分上升导致总分虚高)。
        """
        print("          -> [进攻动能诊断大脑 V401.0 三位一体适配版] 启动，正在诊断分数动态...")
        # --- 步骤 1: 诊断总分(entry_score)的动态，用于风险控制（机会衰退）---
        score_change = entry_score.diff(1).fillna(0)
        score_accel = score_change.diff(1).fillna(0)
        # 状态: 【机会衰退】(否决票来源) - 当总分增长停滞或减速时触发
        scoring_params = get_params_block(self.strategy, 'four_layer_scoring_params')
        momentum_params = scoring_params.get('momentum_diagnostics_params', {})
        fading_score_threshold = get_param_value(momentum_params.get('fading_score_threshold'), 800)
        is_opportunity_fading = ((score_change > 0) & (score_accel < 0)) | (score_change <= 0)
        self.strategy.atomic_states['SCORE_DYN_OPPORTUNITY_FADING'] = is_opportunity_fading & (entry_score.shift(1) > fading_score_threshold)
        # 状态: 【风险抬头】(否决票来源) - 逻辑不变
        risk_score = self.strategy.df_indicators.get('risk_score', pd.Series(0.0, index=entry_score.index))
        risk_change = risk_score.diff(1).fillna(0)
        risk_accel = risk_change.diff(1).fillna(0)
        is_risk_escalating = (risk_change > 0) & (risk_accel > 0)
        self.strategy.atomic_states['SCORE_DYN_RISK_ESCALATING'] = is_risk_escalating
        # --- 步骤 2: 生成用于调试的详细诊断报告 ---
        diagnostics = pd.Series([{} for _ in range(len(entry_score))], index=entry_score.index)
        # 诊断核心维度分数的变化，以识别结构性问题
        core_score = score_details_df.get('SCORE_SETUP', pd.Series(0.0, index=entry_score.index))
        core_score_change = core_score.diff(1).fillna(0)
        # 定义各种诊断条件
        stall_condition = (score_change <= 0) & (entry_score.shift(1) > 0)
        decel_condition = (score_change > 0) & (score_accel < 0)
        core_erosion_condition = (core_score_change < 0)
        divergence_condition = (core_score_change <= 0) & (score_change > 0) & (entry_score > 0)
        # 填充报告
        for idx in entry_score.index:
            report = {}
            if stall_condition.at[idx]: report['stall'] = f"进攻停滞(总分变化: {score_change.at[idx]:.0f})"
            if decel_condition.at[idx]: report['deceleration'] = f"进攻减速(加速度: {score_accel.at[idx]:.0f})"
            if core_erosion_condition.at[idx]: report['core_erosion'] = f"核心侵蚀(战备分变化: {core_score_change.at[idx]:.2f})"
            if divergence_condition.at[idx]: report['divergence'] = "结构性背离(总分虚高)"
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
















